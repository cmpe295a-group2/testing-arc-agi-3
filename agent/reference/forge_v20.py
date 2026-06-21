# =====================================================================
# REFERENCE ONLY — "FORGE v20" frame-only agent (CHRONOS, Kaggle LB 0.42).
# Saved 2026-06-20 as our base template. Runs on the Kaggle gateway harness
# (online mode), NOT standalone on Windows. See planning/KAGGLE_HARNESS.md.
#
# Why this is the base: its own header confirms OUR analysis — Kaggle grades in
# COMPETITION/online mode, game source is NOT reachable, so the deepcopy/BFS/
# transfer stack is DELETED and the agent is pure frame-only:
#   (1) per-game transition graph keyed by frame hash (Blind-Squirrel frontier
#       exploration: prefer untried actions; when a state is exhausted, plan a
#       path through known edges to the nearest state with untried actions);
#   (2) an online-trained CNN (ChangeNet) that predicts whether an action will
#       change the frame (novelty-weighted), used to rank untried actions and to
#       sample when the graph is exhausted (StochasticGoose-style).
# Our plan: start from this, then push past 0.46 (the public ceiling; LB #1 Tufa
# = 1.21) with better exploration / a learned world-model / the LLM path.
# =====================================================================
import hashlib
import logging
import os
import random
import time
import traceback
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent import Agent
from arcengine import GameAction, GameState

logger = logging.getLogger(__name__)

# Global time budget, shared across every game played in this process.
_PROCESS_START = time.time()
GLOBAL_BUDGET_S = float(os.environ.get("FORGE_GLOBAL_BUDGET_S", 8 * 3600))
EXPECTED_GAMES = int(os.environ.get("FORGE_EXPECTED_GAMES", 6))
SAFETY_MARGIN_S = 240.0

GRID = 64
N_SIMPLE = 5                       # ACTION1..ACTION5
CLICK_BASE = N_SIMPLE              # click action keys are CLICK_BASE + y*64 + x
FEAT_CH = 21                       # 16 one-hot + bg + rarity + edge + row + col


def _now_left():
    return GLOBAL_BUDGET_S - (time.time() - _PROCESS_START)


# Vectorised featurisation (identical at train and inference time)
def featurize(frames: torch.Tensor) -> torch.Tensor:
    """frames: (B, 64, 64) int64 on the target device -> (B, 21, 64, 64)."""
    B = frames.shape[0]
    oh = F.one_hot(frames.clamp(0, 15), 16).permute(0, 3, 1, 2).float()
    cnt = oh.sum(dim=[2, 3])                                   # (B, 16)
    bg = cnt.argmax(dim=1)                                     # (B,)
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


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = F.relu(self.c1(x))
        h = self.c2(h)
        return F.relu(x + h)


class ChangeNet(nn.Module):
    """Shared trunk + 5-way simple-action head + 64x64 coordinate head. Each output
    is the logit of 'this action changes the frame (weighted towards novel changes)'."""
    def __init__(self, in_ch=FEAT_CH):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )
        self.res1 = ResBlock(128)
        self.res2 = ResBlock(128)
        self.a_pool = nn.AdaptiveAvgPool2d(4)
        self.a_fc1 = nn.Linear(128 * 16, 256)
        self.a_fc2 = nn.Linear(256, N_SIMPLE)
        self.drop = nn.Dropout(0.1)
        self.c_dec = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        f = self.res2(self.res1(self.stem(x)))
        a = self.a_pool(f).flatten(1)
        a_logits = self.a_fc2(self.drop(F.relu(self.a_fc1(a))))     # (B, 5)
        c_logits = self.c_dec(f).squeeze(1)                          # (B, 64, 64)
        return a_logits, c_logits


class Node:
    __slots__ = ("cands", "tried", "edges")

    def __init__(self, cands):
        self.cands = cands      # list[int] action keys worth trying here
        self.tried = set()      # action keys already executed from this state
        self.edges = {}         # action key -> resulting state hash


class MyAgent(Agent):
    MAX_ACTIONS = float("inf")
    _MAX_FRAMES = 10
    _games_started = 0

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        MyAgent._games_started += 1
        seed = (int(time.time() * 1e6) + hash(self.game_id)) % (2 ** 31 - 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.game_start = time.time()
        remaining_games = max(1, EXPECTED_GAMES - MyAgent._games_started + 1)
        self.game_budget = max(300.0, (_now_left() - SAFETY_MARGIN_S) / remaining_games)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            try:
                mps = torch.backends.mps.is_available()
            except Exception:
                mps = False
            self.device = torch.device("mps" if mps else "cpu")

        self.net = ChangeNet().to(self.device)
        self._load_pretrained()
        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)
        self.buf = deque(maxlen=60000)        # (frame uint8, akey, target)
        self.buf_seen = set()
        self.bsz = 64
        self.train_every = 8

        self.level = -1
        self.graph = {}                        # hash -> Node
        self.visited = set()
        self.plan = deque()                    # [(akey, expected_next_hash), ...]
        self.no_change_streak = 0
        self.sweeps_cleared = 0

        self.prev_frame = None
        self.prev_hash = None
        self.prev_akey = None
        self.pending_expected = None
        self.pending_src = None
        self.pending_akey_done = None

        self.simple_actions = [GameAction.ACTION1, GameAction.ACTION2,
                               GameAction.ACTION3, GameAction.ACTION4,
                               GameAction.ACTION5]
        self.err_streak = 0
        self._fallback_i = 0
        self._acts = 0

    def _load_pretrained(self):
        for wp in ("/kaggle/input/forge-pretrained-weights/pretrained_weights.pt",
                   "pretrained_weights.pt"):
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

    def append_frame(self, f):
        self.frames.append(f)
        if len(self.frames) > self._MAX_FRAMES:
            self.frames = self.frames[-self._MAX_FRAMES:]
        if f.guid:
            self.guid = f.guid
        if hasattr(self, "recorder") and not self.is_playback:
            import json
            self.recorder.record(json.loads(f.model_dump_json()))

    def _lvl(self, f):
        return getattr(f, "score", None) or f.levels_completed

    @staticmethod
    def _raw(fd):
        """Last sub-frame as uint8 (values 0..15)."""
        return np.array(fd.frame, dtype=np.uint8)[-1]

    @staticmethod
    def _hash(frame):
        return hashlib.md5(frame.tobytes()).hexdigest()[:16]

    @staticmethod
    def _norm_avail(avail):
        out = set()
        for a in avail or []:
            out.add(a.value if hasattr(a, "value") else int(a))
        return out or {1, 2, 3, 4, 5, 6}

    def _click_candidates(self, frame, cap=48):
        """Centroids plus a coarse subsample of every non-background colour."""
        cnt = np.bincount(frame.ravel(), minlength=16)
        bg = int(cnt.argmax())
        cands = []
        for c in range(16):
            if c == bg or cnt[c] == 0 or cnt[c] > 3200:
                continue
            ys, xs = np.where(frame == c)
            cy, cx = int(np.median(ys)), int(np.median(xs))
            d = np.abs(ys - cy) + np.abs(xs - cx)
            j = int(d.argmin())
            cands.append((int(ys[j]), int(xs[j])))
            step = max(1, len(ys) // 6)
            for k in range(0, len(ys), step):
                cands.append((int(ys[k]), int(xs[k])))
        seen, keys = set(), []
        for (y, x) in cands:
            key = CLICK_BASE + y * GRID + x
            if key not in seen:
                seen.add(key)
                keys.append(key)
            if len(keys) >= cap:
                break
        return keys

    def _candidates(self, frame, avail):
        av = self._norm_avail(avail)
        keys = [i - 1 for i in (1, 2, 3, 4, 5) if i in av]
        if 6 in av:
            keys += self._click_candidates(frame)
        return keys

    @torch.no_grad()
    def _score(self, frame, keys):
        t = torch.from_numpy(frame.astype(np.int64)).unsqueeze(0).to(self.device)
        a_logits, c_logits = self.net(featurize(t))
        a_p = torch.sigmoid(a_logits[0]).cpu().numpy()           # (5,)
        c_p = torch.sigmoid(c_logits[0]).cpu().numpy()           # (64, 64)
        out = {}
        for k in keys:
            if k < N_SIMPLE:
                out[k] = float(a_p[k])
            else:
                ci = k - CLICK_BASE
                out[k] = float(c_p[ci // GRID, ci % GRID])
        return out

    def _record(self, prev_frame, prev_hash, akey, curr_hash):
        changed = curr_hash != prev_hash
        novel = changed and (curr_hash not in self.visited)
        if changed:
            self.visited.add(curr_hash)
            self.no_change_streak = 0
        else:
            self.no_change_streak += 1
        target = 1.0 if novel else (0.6 if changed else 0.0)

        dedup = prev_hash + ":" + str(akey)
        if dedup not in self.buf_seen:
            self.buf_seen.add(dedup)
            self.buf.append((prev_frame.copy(), akey, target))

        node = self.graph.get(prev_hash)
        if node is not None:
            node.tried.add(akey)
            node.edges[akey] = curr_hash

    def _train(self):
        if len(self.buf) < self.bsz:
            return
        idx = np.random.choice(len(self.buf), self.bsz, replace=False)
        frames = np.stack([self.buf[i][0] for i in idx]).astype(np.int64)
        keys = [self.buf[i][1] for i in idx]
        targs = torch.tensor([self.buf[i][2] for i in idx],
                             dtype=torch.float32, device=self.device)
        x = featurize(torch.from_numpy(frames).to(self.device))
        a_logits, c_logits = self.net(x)
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

    def _plan_to_frontier(self, start_hash, limit=2500):
        """Shortest path through known edges to a node with untried actions."""
        parents = {start_hash: None}
        q = deque([start_hash])
        goal = None
        n = 0
        while q and n < limit:
            h = q.popleft()
            n += 1
            node = self.graph.get(h)
            if node is None:
                continue
            if h != start_hash and any(k not in node.tried for k in node.cands):
                goal = h
                break
            for akey, h2 in node.edges.items():
                if h2 == h or h2 in parents:
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

    def _to_action(self, akey, reason):
        if akey < N_SIMPLE:
            act = self.simple_actions[akey]
            act.reasoning = reason
            return act
        ci = akey - CLICK_BASE
        y, x = ci // GRID, ci % GRID
        act = GameAction.ACTION6
        act.set_data({"x": int(x), "y": int(y)})   # always overwrite stale data
        act.reasoning = reason
        return act

    def _safe_fallback(self):
        self._fallback_i = (self._fallback_i + 1) % N_SIMPLE
        return self._to_action(self._fallback_i, "fallback")

    def is_done(self, frames, lf):
        try:
            if lf.state is GameState.WIN:
                return True
            if _now_left() < SAFETY_MARGIN_S:
                return True
            return (time.time() - self.game_start) >= self.game_budget
        except Exception:
            return True

    def choose_action(self, frames, lf):
        try:
            action = self._choose(lf)
            self.err_streak = 0
            return action
        except Exception as e:
            self.err_streak += 1
            if self.err_streak <= 3:
                traceback.print_exc()
            self.plan.clear()
            self.prev_hash = None
            a = self._safe_fallback()
            a.reasoning = f"err:{type(e).__name__}"
            return a

    def _choose(self, lf):
        lvl = self._lvl(lf)

        if lvl != self.level:                    # level change: reset level-local only
            self.level = lvl
            self.graph.clear()
            self.visited.clear()
            self.plan.clear()
            self.prev_frame = None
            self.prev_hash = None
            self.prev_akey = None
            self.pending_expected = None
            self.pending_src = None
            self.pending_akey_done = None
            self.no_change_streak = 0
            self.sweeps_cleared = 0
            for _ in range(min(10, len(self.buf) // self.bsz)):
                self._train()

        if lf.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.plan.clear()
            self.prev_hash = None
            self.pending_expected = None
            a = GameAction.RESET
            a.reasoning = "reset"
            return a

        if not lf.frame:
            self.prev_hash = None
            self.pending_expected = None
            return self._safe_fallback()

        frame = self._raw(lf)
        h = self._hash(frame)

        if self.prev_hash is not None and self.prev_akey is not None:
            self._record(self.prev_frame, self.prev_hash, self.prev_akey, h)

        node = self.graph.get(h)
        if node is None:
            avail = getattr(lf, "available_actions", None)
            node = Node(self._candidates(frame, avail))
            self.graph[h] = node
            self.visited.add(h)

        if self.pending_expected is not None:   # plan verification
            if h != self.pending_expected:
                src = self.graph.get(self.pending_src) if self.pending_src else None
                if src is not None and self.pending_akey_done in src.edges:
                    del src.edges[self.pending_akey_done]
                self.plan.clear()
            self.pending_expected = None
            self.pending_src = None
            self.pending_akey_done = None

        if self.plan:                            # plan execution
            akey, expected = self.plan.popleft()
            self.pending_expected = expected
            self.pending_src = h
            self.pending_akey_done = akey
            self._set_pending(frame, h, akey)
            return self._to_action(akey, "plan")

        untried = [k for k in node.cands if k not in node.tried]
        if untried:                              # untried actions here
            scores = self._score(frame, untried)
            if random.random() < 0.05:
                akey = random.choice(untried)
            else:
                akey = max(untried, key=lambda k: scores[k])
            self._set_pending(frame, h, akey)
            self._maybe_train()
            return self._to_action(akey, "explore")

        path = self._plan_to_frontier(h)         # navigate to nearest untried frontier
        if path:
            self.plan = path
            akey, expected = self.plan.popleft()
            self._set_pending(frame, h, akey)
            return self._to_action(akey, "frontier")

        if self.no_change_streak > 40 and self.sweeps_cleared < 8:   # exhausted graph
            self.sweeps_cleared += 1
            for nd in self.graph.values():
                nd.tried.clear()
            self.no_change_streak = 0

        keys = node.cands or list(range(N_SIMPLE))
        scores = self._score(frame, keys)
        ks = list(scores.keys())
        ps = np.array([max(scores[k], 1e-4) for k in ks])
        ps = ps / ps.sum()
        akey = int(np.random.choice(ks, p=ps))
        self._set_pending(frame, h, akey)
        self._maybe_train()
        return self._to_action(akey, "model")

    def _set_pending(self, frame, h, akey):
        self.prev_frame = frame
        self.prev_hash = h
        self.prev_akey = akey
        self._acts += 1

    def _maybe_train(self):
        if self._acts % self.train_every == 0:
            self._train()
