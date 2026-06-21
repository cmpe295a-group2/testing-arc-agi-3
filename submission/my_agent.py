"""
SJSU CMPE 295A — Group 2 · ARC-AGI-3 frame-only agent (v1, target ≈ 0.46).

Clean from-scratch build of the proven frame-only recipe (graph-exploration +
online frame-change CNN). The scored Kaggle harness serves the 110 private games
from a `gateway:8001` server in ONLINE mode → the agent only ever sees FRAMES
(no game object, no source), so this agent NEVER touches a simulator. See
planning/KAGGLE_HARNESS.md.

Design (per game; the CNN + replay buffer persist across levels, the graph is
level-local):
  1. ExplorationPolicy — a transition graph keyed by frame hash. Each state node
     tracks its candidate actions, which have been tried, and the observed edges.
     Untried actions are preferred; when the current state is exhausted, BFS the
     known edges to the nearest state that still has an untried action (frontier
     exploration). Plan steps are verified: if reality diverges from the stored
     edge, that edge is deleted and we replan (handles hidden state / animation).
  2. ChangeNet — a small CNN trained online to predict P(action changes the frame)
     with a novelty-weighted target. Used as the PRIOR to rank untried actions and
     as the sampling distribution once the graph is exhausted.

Why this scores: RHAE punishes wasted actions, so the whole point is "never
knowingly repeat a transition" — the graph guarantees we don't, and the CNN biases
exploration toward state-changing actions so we burn fewer moves discovering them.

Improvements over the public 0.42–0.46 references: ACTION7 is a first-class action
(not dropped); torch/SDK imports are guarded so the exploration core is unit-tested
offline (submission/test_explore.py); cleaner edge-verification.

This file is self-contained: on Kaggle it is written to
ARC-AGI-3-Agents/agents/templates/my_agent.py and registered as `myagent`.
"""
import hashlib
import logging
import os
import random
import time
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

# --- optional heavy deps (absent on the Windows dev box → logic stays testable) ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    _HAS_TORCH = True
except Exception:                                   # pragma: no cover
    _HAS_TORCH = False

# arcengine (GameAction/GameState) is decoupled from the agents SDK: the SDK's
# `agents/__init__` eager-imports LLM templates that crash offline, but arcengine is a
# clean pure-python wheel that also runs on the dev box — so we can drive the agent on
# the 25 public games locally without the SDK.
try:
    from arcengine import GameAction, GameState
    _HAS_ARC = True
except Exception:                                   # pragma: no cover
    _HAS_ARC = False

try:
    from agents.agent import Agent
    _HAS_SDK = True
except Exception:                                   # pragma: no cover - dev/test only
    _HAS_SDK = False

    class Agent:                                    # minimal stand-in for offline tests
        def __init__(self, *a, **kw):
            pass


# ============================ constants ============================
GRID = 64
SIMPLE_IDS = [1, 2, 3, 4, 5, 7]      # ACTION1-5 + ACTION7 (all "simple" actions)
N_SIMPLE = len(SIMPLE_IDS)           # action keys 0..N_SIMPLE-1 map to SIMPLE_IDS
CLICK_BASE = N_SIMPLE                # a click key is CLICK_BASE + y*GRID + x (ACTION6)
FEAT_CH = 21                         # 16 colour one-hot + bg + rarity + edge + row + col

# Global wall-clock budget shared by every game agent in the process.
_PROCESS_START = time.time()
GLOBAL_BUDGET_S = float(os.environ.get("ARC_GLOBAL_BUDGET_S", 8.5 * 3600))
EXPECTED_GAMES = int(os.environ.get("ARC_EXPECTED_GAMES", 6))
SAFETY_MARGIN_S = 240.0


def _budget_left():
    return GLOBAL_BUDGET_S - (time.time() - _PROCESS_START)


# ============================ candidate generation (pure numpy) ============================
def click_candidates(frame: np.ndarray, cap: int = 48) -> list:
    """Click action keys worth trying: for every non-background colour blob, its centroid,
    its 4 EXTREME cells (top/bottom/left/right-most — interactive edges & corners that the
    centroid misses), and a coarse interior subsample. Clicking empty background is almost
    never useful, so it is skipped. cap is kept at 48 (more candidates spread the fixed
    action budget too thin and lowered coverage). ARC_SIMPLE_CLICKS=1 drops the extremes."""
    simple = bool(os.environ.get("ARC_SIMPLE_CLICKS"))
    cnt = np.bincount(frame.ravel(), minlength=16)
    bg = int(cnt.argmax())
    pts = []
    for c in range(16):
        if c == bg or cnt[c] == 0 or cnt[c] > 3200:
            continue
        ys, xs = np.where(frame == c)
        cy, cx = int(np.median(ys)), int(np.median(xs))
        j = int((np.abs(ys - cy) + np.abs(xs - cx)).argmin())   # snap centroid to a real cell
        pts.append((int(ys[j]), int(xs[j])))
        if not simple:
            for idx in (ys.argmin(), ys.argmax(), xs.argmin(), xs.argmax()):
                pts.append((int(ys[idx]), int(xs[idx])))         # extreme edge/corner cells
        step = max(1, len(ys) // 6)
        for k in range(0, len(ys), step):
            pts.append((int(ys[k]), int(xs[k])))
    seen, keys = set(), []
    for (y, x) in pts:
        key = CLICK_BASE + y * GRID + x
        if key not in seen:
            seen.add(key)
            keys.append(key)
        if len(keys) >= cap:
            break
    return keys


def candidate_keys(frame: np.ndarray, avail_ids: set) -> list:
    """All action keys to consider in a state, given the available action ids."""
    keys = [i for i, aid in enumerate(SIMPLE_IDS) if aid in avail_ids]
    if 6 in avail_ids:
        keys += click_candidates(frame)
    return keys or list(range(N_SIMPLE))


# ============================ exploration policy (pure python — TESTABLE offline) ============================
class _Node:
    __slots__ = ("cands", "tried", "edges")

    def __init__(self, cands):
        self.cands = list(cands)     # candidate action keys
        self.tried = set()           # keys already executed from this state
        self.edges = {}              # key -> resulting frame hash


class ExplorationPolicy:
    """Frame-hash transition graph + frontier exploration. No torch, no SDK — driven
    by (frame_hash, candidates, optional prior). One instance per game level."""

    def __init__(self, epsilon=0.05, exhaust_threshold=40, max_sweeps=8, frontier_limit=2500):
        self.epsilon = epsilon
        self.exhaust_threshold = exhaust_threshold
        self.max_sweeps = max_sweeps
        self.frontier_limit = frontier_limit
        self.reset_level()

    def reset_level(self):
        self.graph = {}
        self.visited = set()
        self.plan = deque()                 # [(akey, expected_next_hash), ...]
        self.no_change_streak = 0
        self.sweeps = 0
        self._pending = None                # (src_hash, akey, expected_hash) of last planned step

    # -- called at the START of each turn to close out the previous transition --
    def observe(self, prev_hash, akey, curr_hash):
        """Record the edge prev_hash --akey--> curr_hash. Returns the CNN training
        target for that transition (1 novel, 0.6 changed-seen, 0 no-change)."""
        changed = curr_hash != prev_hash
        novel = changed and curr_hash not in self.visited
        if changed:
            self.visited.add(curr_hash)
            self.no_change_streak = 0
        else:
            self.no_change_streak += 1
        node = self.graph.get(prev_hash)
        if node is not None:
            node.tried.add(akey)
            node.edges[akey] = curr_hash
        # verify the previous planned step; a divergent edge is unreliable → drop it
        if self._pending is not None:
            src, pak, expected = self._pending
            if src == prev_hash and pak == akey and curr_hash != expected:
                s = self.graph.get(src)
                if s is not None and pak in s.edges:
                    del s.edges[pak]
                self.plan.clear()
            self._pending = None
        return 1.0 if novel else (0.6 if changed else 0.0)

    # -- choose the next action key for the current state --
    def select(self, frame_hash, candidates, prior=None):
        node = self.graph.get(frame_hash)
        if node is None:
            node = _Node(candidates)
            self.graph[frame_hash] = node
            self.visited.add(frame_hash)

        if self.plan:                                    # execute a planned route
            akey, expected = self.plan.popleft()
            self._pending = (frame_hash, akey, expected)
            return akey, "plan"

        untried = [k for k in node.cands if k not in node.tried]
        if untried:                                      # prefer untried, CNN-ranked
            if prior is not None and random.random() >= self.epsilon:
                akey = max(untried, key=lambda k: prior.get(k, 0.0))
            else:
                akey = random.choice(untried)
            self._pending = None
            return akey, "explore"

        path = self._plan_to_frontier(frame_hash)        # route to nearest untried state
        if path:
            self.plan = path
            akey, expected = self.plan.popleft()
            self._pending = (frame_hash, akey, expected)
            return akey, "frontier"

        # graph exhausted: mechanics may be time/state dependent → re-open tried sets a
        # few times, otherwise sample by the CNN prior.
        if self.no_change_streak > self.exhaust_threshold and self.sweeps < self.max_sweeps:
            self.sweeps += 1
            for nd in self.graph.values():
                nd.tried.clear()
            self.no_change_streak = 0
        keys = node.cands or list(range(N_SIMPLE))
        if prior is not None:
            ps = np.array([max(prior.get(k, 1e-4), 1e-4) for k in keys], dtype=np.float64)
            ps /= ps.sum()
            akey = int(np.random.choice(keys, p=ps))
        else:
            akey = random.choice(keys)
        self._pending = None
        return akey, "model"

    def _plan_to_frontier(self, start_hash):
        """BFS over known edges to the nearest node with an untried candidate."""
        parents = {start_hash: None}
        q = deque([start_hash])
        goal, n = None, 0
        while q and n < self.frontier_limit:
            h = q.popleft()
            n += 1
            node = self.graph.get(h)
            if node is None:
                continue
            if h != start_hash and any(k not in node.tried for k in node.cands):
                goal = h
                break
            for akey, h2 in node.edges.items():
                if h2 == h or h2 in parents:             # skip self-loops / seen
                    continue
                parents[h2] = (h, akey)
                q.append(h2)
        if goal is None:
            return None
        path = []
        h = goal
        while parents[h] is not None:
            ph, akey = parents[h]
            path.append((akey, h))
            h = ph
        path.reverse()
        return deque(path)


# ============================ tabular change prior (pure python — TESTABLE, no torch) ============================
class ChangeStats:
    """A cheap tabular stand-in for the change-CNN, used when no GPU is available.

    Tracks P(action changes the frame) per action FEATURE — the simple-action id, or the
    COLOUR of the clicked cell — with Laplace smoothing toward 0.5. Keying clicks by the
    clicked colour generalises across positions ("clicking red things changes the board")
    so it learns within a handful of tries. Used to rank untried actions and to weight the
    sampling distribution once the graph is exhausted, so the agent stops burning moves on
    no-op clicks (which both deepens reach within the action budget and lifts RHAE)."""

    def __init__(self, smoothing=2.0):
        self.tries = {}
        self.changes = {}
        self.k = smoothing

    def update(self, key, changed):
        self.tries[key] = self.tries.get(key, 0) + 1
        if changed:
            self.changes[key] = self.changes.get(key, 0) + 1

    def rate(self, key):
        n = self.tries.get(key, 0)
        c = self.changes.get(key, 0)
        return (c + 0.5 * self.k) / (n + self.k)


def action_feature(frame: np.ndarray, akey: int):
    """Feature key for the change prior: ('s', simple_id) or ('c', clicked-cell colour)."""
    if akey < N_SIMPLE:
        return ("s", akey)
    ci = akey - CLICK_BASE
    return ("c", int(frame[ci // GRID, ci % GRID]))


# ============================ change-prediction CNN (torch) ============================
if _HAS_TORCH:
    def featurize(frames: "torch.Tensor") -> "torch.Tensor":
        """(B,64,64) int64 → (B,21,64,64): colour one-hot + bg mask + rarity + edge map
        + normalised row/col. Identical at train and inference time."""
        B = frames.shape[0]
        oh = F.one_hot(frames.clamp(0, 15), 16).permute(0, 3, 1, 2).float()
        cnt = oh.sum(dim=[2, 3])
        bg = cnt.argmax(dim=1)
        bg_mask = (frames == bg.view(B, 1, 1)).float().unsqueeze(1)
        mx = cnt.max(dim=1, keepdim=True)[0].clamp(min=1.0)
        rarity = (oh * (1.0 - cnt / mx).view(B, 16, 1, 1)).sum(1, keepdim=True)
        f = frames.unsqueeze(1).float()
        pad = F.pad(f, (1, 1, 1, 1), mode="replicate")
        edge = ((f != pad[:, :, :-2, 1:-1]) | (f != pad[:, :, 2:, 1:-1]) |
                (f != pad[:, :, 1:-1, :-2]) | (f != pad[:, :, 1:-1, 2:])).float()
        rp = torch.linspace(0, 1, GRID, device=frames.device).view(1, 1, GRID, 1).expand(B, 1, GRID, GRID)
        cp = torch.linspace(0, 1, GRID, device=frames.device).view(1, 1, 1, GRID).expand(B, 1, GRID, GRID)
        return torch.cat([oh, bg_mask, rarity, edge, rp, cp], dim=1)

    class _ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
            self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

        def forward(self, x):
            return F.relu(x + self.c2(F.relu(self.c1(x))))

    class ChangeNet(nn.Module):
        """Trunk + simple-action head (N_SIMPLE) + 64×64 coordinate head; each logit is
        'this action changes the frame (novelty-weighted)'."""
        def __init__(self, in_ch=FEAT_CH):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
            self.res1 = _ResBlock(128)
            self.res2 = _ResBlock(128)
            self.a_pool = nn.AdaptiveAvgPool2d(4)
            self.a_fc1 = nn.Linear(128 * 16, 256)
            self.a_fc2 = nn.Linear(256, N_SIMPLE)
            self.drop = nn.Dropout(0.1)
            self.c_dec = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 1, 1))

        def forward(self, x):
            f = self.res2(self.res1(self.stem(x)))
            a = self.a_pool(f).flatten(1)
            a_logits = self.a_fc2(self.drop(F.relu(self.a_fc1(a))))
            c_logits = self.c_dec(f).squeeze(1)
            return a_logits, c_logits


# ============================ SDK agent (glue) ============================
class MyAgent(Agent):
    """Wires the frame stream to ExplorationPolicy + ChangeNet. Implements the SDK's
    is_done / choose_action. One instance per game; the net/buffer persist across levels."""

    MAX_ACTIONS = float("inf")
    _MAX_FRAMES = 10
    _games_started = 0

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        MyAgent._games_started += 1
        _sd = os.environ.get("ARC_SEED")                 # ARC_SEED → reproducible A/B runs
        if _sd is not None:
            seed = (int(_sd) * 2654435761 + hash(getattr(self, "game_id", "g"))) % (2 ** 31 - 1)
        else:
            seed = (int(time.time() * 1e6) + hash(getattr(self, "game_id", "g"))) % (2 ** 31 - 1)
        random.seed(seed)
        np.random.seed(seed)
        # The tabular change/novelty prior empirically HURT graph-only coverage (greedy
        # novelty funnels exploration; uniform covers more) — kept but OFF by default,
        # only worth revisiting blended with the spatial CNN. A/B: 13/10 uniform vs 8/7 prior.
        self._use_change_prior = bool(os.environ.get("ARC_ENABLE_CHANGE_PRIOR"))

        self.game_start = time.time()
        remaining = max(1, EXPECTED_GAMES - MyAgent._games_started + 1)
        self.game_budget = max(300.0, (_budget_left() - SAFETY_MARGIN_S) / remaining)

        self.policy = ExplorationPolicy()
        self.level = -1
        self.prev_frame = self.prev_hash = self.prev_akey = None
        self.err_streak = 0
        self._fallback_i = 0
        self._acts = 0

        # CNN (optional) + online replay buffer (persist across levels)
        self.net = self.opt = None
        self.device = None
        if _HAS_TORCH and not os.environ.get("ARC_DISABLE_CNN"):
            seed_t = seed
            torch.manual_seed(seed_t)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.net = ChangeNet().to(self.device)
            self._load_pretrained()
            self.opt = optim.Adam(self.net.parameters(), lr=3e-4)
        self.buf = deque(maxlen=60000)       # (frame uint8, akey, target)
        self.buf_seen = set()
        self.bsz = 64
        self.train_every = 8
        self.stats = ChangeStats()           # tabular change prior (used when the CNN is off)

    # ---- weights (optional warm start from a Kaggle dataset) ----
    def _load_pretrained(self):
        for wp in ("/kaggle/input/arc-agi3-pretrained/pretrained.pt", "pretrained.pt"):
            try:
                if os.path.exists(wp):
                    state = torch.load(wp, map_location=self.device, weights_only=True)
                    ms = self.net.state_dict()
                    for k in list(state.keys()):
                        if k in ms and state[k].shape == ms[k].shape:
                            ms[k] = state[k]
                    self.net.load_state_dict(ms)
                    logger.info("loaded pretrained weights from %s", wp)
                    return
            except Exception:
                pass

    # ---- SDK plumbing ----
    def append_frame(self, f):
        self.frames.append(f)
        if len(self.frames) > self._MAX_FRAMES:
            self.frames = self.frames[-self._MAX_FRAMES:]
        if getattr(f, "guid", None):
            self.guid = f.guid
        if hasattr(self, "recorder") and not getattr(self, "is_playback", False):
            import json
            self.recorder.record(json.loads(f.model_dump_json()))

    @staticmethod
    def _lvl(f):
        return getattr(f, "score", None) or f.levels_completed

    @staticmethod
    def _raw(fd):
        return np.array(fd.frame, dtype=np.uint8)[-1]    # last sub-frame, values 0..15

    @staticmethod
    def _hash(frame):
        return hashlib.md5(frame.tobytes()).hexdigest()[:16]

    @staticmethod
    def _avail(lf):
        out = set()
        for a in getattr(lf, "available_actions", None) or []:
            out.add(a.value if hasattr(a, "value") else int(a))
        return out or {1, 2, 3, 4, 5, 6, 7}

    # ---- CNN prior + online training ----
    def _prior(self, frame, keys):
        if self.net is None:
            return None
        with torch.no_grad():
            t = torch.from_numpy(frame.astype(np.int64)).unsqueeze(0).to(self.device)
            a_logits, c_logits = self.net(featurize(t))
            a_p = torch.sigmoid(a_logits[0]).cpu().numpy()
            c_p = torch.sigmoid(c_logits[0]).cpu().numpy()
        out = {}
        for k in keys:
            if k < N_SIMPLE:
                out[k] = float(a_p[k])
            else:
                ci = k - CLICK_BASE
                out[k] = float(c_p[ci // GRID, ci % GRID])
        return out

    def _buffer(self, frame, akey, target):
        dedup = self.prev_hash + ":" + str(akey)
        if dedup not in self.buf_seen:
            self.buf_seen.add(dedup)
            self.buf.append((frame.copy(), akey, target))

    def _train(self):
        if self.net is None or len(self.buf) < self.bsz:
            return
        idx = np.random.choice(len(self.buf), self.bsz, replace=False)
        frames = np.stack([self.buf[i][0] for i in idx]).astype(np.int64)
        keys = [self.buf[i][1] for i in idx]
        targs = torch.tensor([self.buf[i][2] for i in idx], dtype=torch.float32, device=self.device)
        a_logits, c_logits = self.net(featurize(torch.from_numpy(frames).to(self.device)))
        sel = torch.empty(self.bsz, device=self.device)
        for i, k in enumerate(keys):
            if k < N_SIMPLE:
                sel[i] = a_logits[i, k]
            else:
                ci = k - CLICK_BASE
                sel[i] = c_logits[i, ci // GRID, ci % GRID]
        loss = F.binary_cross_entropy_with_logits(sel, targs)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    # ---- action construction ----
    def _to_action(self, akey, reason):
        if akey < N_SIMPLE:
            act = GameAction.from_id(SIMPLE_IDS[akey])
            act.reasoning = reason
            return act
        ci = akey - CLICK_BASE
        act = GameAction.ACTION6
        act.set_data({"x": int(ci % GRID), "y": int(ci // GRID)})   # always fresh click data
        act.reasoning = reason
        return act

    def _fallback(self):
        self._fallback_i = (self._fallback_i + 1) % N_SIMPLE
        return self._to_action(self._fallback_i, "fallback")

    # ---- SDK hooks ----
    def is_done(self, frames, lf):
        try:
            if lf.state is GameState.WIN:
                return True
            if _budget_left() < SAFETY_MARGIN_S:
                return True
            return (time.time() - self.game_start) >= self.game_budget
        except Exception:
            return True

    def choose_action(self, frames, lf):
        try:
            return self._choose(lf)
        except Exception as e:
            self.err_streak += 1
            if self.err_streak <= 3:
                logger.exception("choose_action error")
            self.policy.plan.clear()
            self.prev_hash = None
            a = self._fallback()
            a.reasoning = f"err:{type(e).__name__}"
            return a

    def _choose(self, lf):
        lvl = self._lvl(lf)
        if lvl != self.level:                            # new level: reset graph, keep net
            self.level = lvl
            self.policy.reset_level()
            self.prev_frame = self.prev_hash = self.prev_akey = None
            for _ in range(min(10, len(self.buf) // self.bsz)):
                self._train()

        if lf.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.policy.plan.clear()
            self.prev_hash = None
            a = GameAction.RESET
            a.reasoning = "reset"
            return a

        if not lf.frame:                                 # no frame: act safely, don't pollute graph
            self.prev_hash = None
            return self._fallback()

        frame = self._raw(lf)
        h = self._hash(frame)

        if self.prev_hash is not None and self.prev_akey is not None:   # close prev transition
            target = self.policy.observe(self.prev_hash, self.prev_akey, h)
            self._buffer(self.prev_frame, self.prev_akey, target)
            # train the prior on NOVELTY (reaching a new state), not mere churn (target>=1.0)
            self.stats.update(action_feature(self.prev_frame, self.prev_akey), target >= 1.0)

        keys = candidate_keys(frame, self._avail(lf))
        # CNN prior on the GPU; else the cheap tabular novelty prior; else uniform (A/B baseline)
        if self.net is not None:
            prior = self._prior(frame, keys)
        elif self._use_change_prior:
            prior = {k: self.stats.rate(action_feature(frame, k)) for k in keys}
        else:
            prior = None
        akey, reason = self.policy.select(h, keys, prior)

        self.prev_frame, self.prev_hash, self.prev_akey = frame, h, akey
        self._acts += 1
        if self._acts % self.train_every == 0:
            self._train()
        self.err_streak = 0
        return self._to_action(akey, reason)
