"""
Forward-Model Search Agent for ARC-AGI-3
========================================

SJSU CMPE 295A, Group 2 — primary engine ("the best algorithm").

THE THESIS
----------
ARC-AGI-3 ships each game as a *deterministic, in-process Python object*
(`arcengine.ARCBaseGame`). In OFFLINE mode the wrapper hands us that live object
(`arc_env._game`). So we can `copy.deepcopy` it, search the copy *for free*
(simulated steps never touch the scorecard), find a near-optimal action sequence,
and execute ONLY that sequence on the real, scored instance.

This converts ARC-AGI-3's defining obstacle — "the goal is never told" — into a
*checkable* objective: a rollout that makes `_game._score` tick up (a level was
completed) or sets `_game._state == WIN` is, by definition, a solution.

VERIFIED API FACTS (read from the shipped wheels: arcengine 0.9.3 / arc_agi 0.9.8)
----------------------------------------------------------------------------------
- `ARCBaseGame.perform_action(action_input, raw=True) -> FrameDataRaw`  [base_game.py:189]
  Mutates the game in place; the ONLY way state advances. `@final`.
- `_game._score`  = levels_completed (bumped by `next_level()`)         [base_game.py:412]
- `_game._state`  = GameState {NOT_PLAYED, NOT_FINISHED, WIN, GAME_OVER} [enums.py:34]
- `_game._current_level_index`, `_game._action_count`, `_game._available_actions`
- `_game._get_valid_actions() -> list[ActionInput]`  the click-pruning oracle
  (keyboard 1..5 as SimpleAction; ACTION6 expands to sprite-cell clicks)  [base_game.py:480]
- `arc_env.step(action: GameAction, data: dict, reasoning) -> FrameDataRaw`  [local_wrapper.py:211]
  builds `ActionInput(id=action, data=data)` then `_game.perform_action(...)`.
- Scorecard updates only in `EnvironmentWrapper._set_last_response`, gated on
  `scorecard_manager and resp.guid and len(resp.frame)>0`                 [wrapper.py:186]
  => a deepcopy's `perform_action` is invisible to scoring. THE EXPLOIT.
- Offline load: `LocalEnvironmentWrapper(EnvironmentInfo, ...)` execs `<id>.py`,
  instantiates the `ARCBaseGame` subclass (e.g. cd82 -> class `Cd82`), stores `._game`.
- ACTION7 is a 7th *simple* action (NOT undo).                            [enums.py:59]

WHY WE DON'T SUBCLASS THE SDK `Agent`
-------------------------------------
`from agents.agent import Agent` first runs `agents/__init__.py`, which EAGERLY
imports the LLM templates (openai/langgraph/smolagents) [agents/__init__.py:8-15].
None of those ship in the offline wheels, so that import CRASHES on the Kaggle image.
We therefore drive `arc_env` with our own thin loop (identical semantics to the SDK
`Agent.main()` loop) and keep the policy decoupled. See `SDKAgentAdapter` note below
for wiring into the swarm when submitting.

SCORING (RHAE), so search optimizes the right thing                       [scorecard.py:168]
- per-level score = min(1.15, (human_baseline / agent_actions) ** 2)   (0 if not completed)
- per-game       = level-weighted (1-indexed) average, capped by completion
- => SHORTEST solving sequence wins. BFS (shortest-path) is the natural fit.

BUILD ORDER (this file implements B-1 + B2; scaffolds B1/B3)
- B-1  Simulability Probe ...... Day-1 go/no-go gate (deepcopy free + deterministic).
- B1   Novelty explorer ........ perception + candidate enumeration (substrate).
- B2   Forward-model BFS/IDDFS .. THE core: search the clone, execute the plan.   <-- here
- B3   Best-first + win-inference  defeats deep levels (heuristic-guided).         <-- scaffold

RUNTIME NOTE
------------
The engine wheels are Linux `cp312` (numpy/pillow compiled for Linux). Develop and
run this on the pinned Kaggle Linux image — it will NOT import on the Windows dev box.
The logic is written against the verified API and is target-correct.
"""

from __future__ import annotations

import collections
import contextlib
import copy
import heapq
import json
import logging
import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# --- Engine imports (available only where the wheels are installed: Kaggle Linux) ---
try:
    from arcengine import ARCBaseGame, ActionInput, GameAction, GameState
    _ENGINE_AVAILABLE = True
    _IMPORT_ERROR: Optional[BaseException] = None
except Exception as _e:  # pragma: no cover - dev box without the Linux wheels
    ARCBaseGame = object       # type: ignore[assignment,misc]
    ActionInput = object       # type: ignore[assignment,misc]
    GameAction = None          # type: ignore[assignment]
    GameState = None           # type: ignore[assignment]
    _ENGINE_AVAILABLE = False
    _IMPORT_ERROR = _e

log = logging.getLogger("fwd_model_agent")


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class SearchConfig:
    """Knobs for the forward-model search. Tune per-game / per-budget."""
    max_depth: int = 220              # depth ceiling (deep levels reach ~190-424)
    node_budget: int = 400_000        # max clones expanded per plan call
    time_budget_s: float = 8.0        # used when a plan fn is called directly
    bfs_budget_s: float = 2.5         # cascade stage 1: optimal BFS (shallow levels)
    novelty_budget_s: float = 25.0    # cascade stage 2: novelty best-first (deep levels)
    bucket_stride: int = 4            # Go-Explore coarse bucket = frame[::stride, ::stride]
    max_click_candidates: int = 64    # cap on ACTION6 click targets per state (tame the branch)
    rng_guard: bool = True            # snapshot/restore global RNG around every rollout (lf52)


# =============================================================================
# B-1 substrate: RNG guard, state view, canonical hashing
# =============================================================================
@contextlib.contextmanager
def rng_snapshot():
    """Freeze global numpy + stdlib RNG across a rollout, then restore it.

    Some games touch global RNG (e.g. `lf52` does an unseeded `np.random.shuffle`
    in __init__). We `deepcopy` an *already-constructed* instance so its shuffled
    layout is frozen; this guard additionally ensures simulating on a clone cannot
    perturb the real instance's future global draws.
    """
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


@dataclass(frozen=True)
class StateView:
    """Cheap, comparable snapshot of the engine state that matters for search."""
    level_index: int
    score: int                 # == levels_completed
    state_name: str            # GameState.name
    frame_key: bytes           # bytes of the resting (last) rendered frame
    hidden_key: bytes = b""    # _get_hidden_state() bytes — disambiguates states that
                               # render identically (e.g. ls20 fog / partial observability)
    sig: tuple = ()            # value-hash of the game's scalar attrs — captures internal
                               # progress NOT shown in the frame (e.g. sc25's hidden flags)

    @property
    def is_win(self) -> bool:
        return self.state_name == "WIN"

    @property
    def is_over(self) -> bool:
        return self.state_name == "GAME_OVER"


def _resting_frame(game) -> np.ndarray:
    """Render the current (resting) frame of a game object as an ndarray."""
    return game.camera.render(game.current_level.get_sprites())


# Attributes that depend on the PATH (how we got here), not the STATE (where we are).
# Excluded from the dedup signature so equivalent positions reached by different routes
# still merge — otherwise a move-counter would make every state unique and explode search.
_PATH_ATTRS = frozenset({"_action", "_action_count", "_full_reset",
                         "_next_level", "_action_complete"})


def _scalar_sig(game) -> tuple:
    """Value-hash of the game's scalar attributes (int/float/bool/str/scalar-tuple).

    Captures internal progress that the rendered frame does NOT show — the sc25 case,
    where every action flips an obfuscated boolean while the pixels stay identical, so
    a frame-only signature wrongly prunes every action as a no-op. Lists/objects (sprite
    & level data) are skipped here; their visible effect is already in the frame.
    """
    out = []
    for k, v in game.__dict__.items():
        if k in _PATH_ATTRS:
            continue
        if isinstance(v, (int, float, bool, str)):
            out.append((k, v))
        elif isinstance(v, tuple) and all(isinstance(x, (int, float, bool, str)) for x in v):
            out.append((k, v))
    return tuple(sorted(out))


def view_of(game, last_frame: Optional[np.ndarray] = None) -> StateView:
    """Build a StateView from a live/cloned game object.

    The dedup/cycle key is the GAME STATE identity (level, score, rendered frame, engine
    hidden state, and scalar internal attrs) — deliberately NOT the path length, so
    equivalent positions reached by different routes are merged.
    """
    frame = last_frame if last_frame is not None else _resting_frame(game)
    try:
        hidden = np.ascontiguousarray(game._get_hidden_state()).tobytes()
    except Exception:
        hidden = b""
    return StateView(
        level_index=int(game._current_level_index),
        score=int(game._score),
        state_name=game._state.name,
        frame_key=np.ascontiguousarray(frame).tobytes(),
        hidden_key=hidden,
        sig=_scalar_sig(game),
    )


# =============================================================================
# B1 substrate: candidate-action enumeration (perception + click pruning)
# =============================================================================
def enumerate_candidate_actions(game, cfg: SearchConfig) -> list:
    """The branching set for search at the current state, as `ActionInput`s.

    Simple actions (1..5, 7) come straight from the per-frame legal-action filter
    `_available_actions` — note the engine's `_get_valid_actions()` oracle only
    expands ids 1..5 and 6, so it OMITS ACTION7 and we must add it ourselves
    [base_game.py:491]. Click targets (ACTION6) come from the oracle, which
    restricts clicks to meaningful sprite cells (the never-exposed pruning we get
    for free in-process). RESET (0) is intentionally excluded from search.

    TODO(B1): if the oracle yields no clicks for a click game, fall back to
    object-centroid clicks (connected-component segmentation over the frame).
    """
    avail = list(game._available_actions)
    simple = [ActionInput(id=GameAction.from_id(aid)) for aid in avail
              if aid in (1, 2, 3, 4, 5, 7)]

    clicks: list = []
    if 6 in avail:
        try:
            raw_clicks = list(game._get_valid_actions())
        except Exception:  # the oracle is internal; be defensive
            raw_clicks = []
        seen: set = set()
        for ai in raw_clicks:
            aid = ai.id.value if hasattr(ai.id, "value") else int(ai.id)
            if aid != 6:
                continue
            d = ai.data or {}
            key = (d.get("x"), d.get("y"))
            if key in seen:
                continue
            seen.add(key)
            clicks.append(ai)
        # TODO(B1): rank clicks (object size / novelty / CNN prior) before truncating.
        clicks = clicks[: cfg.max_click_candidates]
    return simple + clicks


def _apply(game, ai) -> np.ndarray:
    """Apply an action to a (cloned) game in place; return its resting frame.

    `perform_action` renders an animation; the last frame is the resting state.
    """
    raw = game.perform_action(ai, raw=True)
    frames = raw.frame
    return frames[-1] if frames else _resting_frame(game)


# =============================================================================
# B2 (core): forward-model search — find the shortest level-completing plan
# =============================================================================
@dataclass
class SearchResult:
    plan: list = field(default_factory=list)   # list[ActionInput]
    solved: bool = False                       # reached a higher _score or WIN
    nodes: int = 0
    elapsed_s: float = 0.0
    reason: str = ""


def plan_to_next_level(root_game, cfg: SearchConfig,
                       heuristic: Optional[Callable] = None) -> SearchResult:
    """Search a deepcopy of `root_game` for the SHORTEST action sequence that
    completes the current level (or wins the game).

    * BFS by default (shortest-path == fewest actions == best RHAE).
    * If `heuristic` is given, run best-first (A*-ish) — the B3 hook for deep levels.
    * Success := a clone's `_score` exceeds the root's, or state == WIN.
    * Prune GAME_OVER branches, no-ops, and already-seen canonical states.
    * Honor node + wall-clock budgets; on exhaustion return the best partial prefix.
    """
    t0 = time.time()
    target_score = int(root_game._score)
    use_best_first = heuristic is not None

    with rng_snapshot() if cfg.rng_guard else contextlib.nullcontext():
        root = copy.deepcopy(root_game)
    root_view = view_of(root)

    counter = 0
    frontier: list = [(0.0, counter, root, [], root_view)] if use_best_first else []
    bfs: "collections.deque" = collections.deque(
        [] if use_best_first else [(root, [], root_view)])

    seen: set = {root_view}
    nodes = 0
    best_partial: list = []
    best_h = float("inf")      # for heuristic-aware MPC partial selection

    def out_of_budget() -> bool:
        return nodes >= cfg.node_budget or (time.time() - t0) >= cfg.time_budget_s

    while True:
        if use_best_first:
            if not frontier or out_of_budget():
                break
            _, _, game, plan, view = heapq.heappop(frontier)
        else:
            if not bfs or out_of_budget():
                break
            game, plan, view = bfs.popleft()

        if len(plan) >= cfg.max_depth:
            continue

        for ai in enumerate_candidate_actions(game, cfg):
            if out_of_budget():
                break
            with rng_snapshot() if cfg.rng_guard else contextlib.nullcontext():
                child = copy.deepcopy(game)
                last_frame = _apply(child, ai)
            nodes += 1
            cview = view_of(child, last_frame)
            new_plan = plan + [ai]

            # Success: a level was completed (or the whole game won).
            if int(child._score) > target_score or cview.is_win:
                return SearchResult(plan=new_plan, solved=True, nodes=nodes,
                                    elapsed_s=time.time() - t0, reason="level_completed")

            # Dead ends / no-ops / cycles: prune.
            if cview.is_over:        # lethal move
                continue
            if cview == view:        # action changed nothing (no-op) -> blacklist
                continue
            if cview in seen:
                continue
            seen.add(cview)

            if use_best_first:
                counter += 1
                h = float(heuristic(child, last_frame))
                if h < best_h:                          # MPC: commit toward the most-promising node
                    best_h, best_partial = h, new_plan
                heapq.heappush(frontier, (len(new_plan) + h, counter, child, new_plan, cview))
            else:
                if len(new_plan) > len(best_partial):   # BFS: no progress signal -> deepest reach
                    best_partial = new_plan
                bfs.append((child, new_plan, cview))

    return SearchResult(plan=best_partial, solved=False, nodes=nodes,
                        elapsed_s=time.time() - t0,
                        reason="budget_exhausted" if out_of_budget() else "frontier_empty")


# =============================================================================
# B3 (core): novelty-guided best-first search (Go-Explore frontier)
# =============================================================================
def _coarse_bucket(frame: np.ndarray, level_index: int, stride: int) -> tuple:
    """A Go-Explore 'cell': a coarse spatial signature of the frame. States in the
    same bucket are treated as similar; rarely-bucketed states count as 'novel'."""
    ds = np.ascontiguousarray(frame[::stride, ::stride])
    return (level_index, ds.tobytes())


def _agent_centroid(frame: np.ndarray, agent_color: int) -> tuple:
    """Centroid (x, y) of the agent-colored cells, or (-1, -1) if absent."""
    ys, xs = np.where(frame == agent_color)
    if len(xs) == 0:
        return (-1, -1)
    return (int(xs.mean()), int(ys.mean()))


def infer_agent_color(game, cfg: SearchConfig):
    """B3 (reach): identify the 'agent' = the non-background color whose footprint
    moves most under actions. Cheap (~16 one-ply clones), general, no goal needed.

    Used to bucket novelty search by AGENT POSITION (a few hundred reachable cells)
    instead of the whole frame (b^d) — so the search covers the reachable space and
    lands on the goal for routing/reach games (bp35-style). Returns a color, or None.
    """
    root = _resting_frame(game)
    bg = int(np.bincount(np.ascontiguousarray(root).ravel().astype(np.int64),
                         minlength=16).argmax())
    moved: dict = {}
    for ai in enumerate_candidate_actions(game, cfg)[:16]:
        with rng_snapshot() if cfg.rng_guard else contextlib.nullcontext():
            c = copy.deepcopy(game)
            f = _apply(c, ai)
        diff = f != root
        if not diff.any():
            continue
        cols = set(int(x) for x in np.unique(f[diff])) | set(int(x) for x in np.unique(root[diff]))
        for col in cols:
            if col == bg or col < 0:
                continue
            moved[col] = moved.get(col, 0) + int(np.logical_xor(f == col, root == col).sum())
    return max(moved, key=moved.get) if moved else None


def make_match_heuristic(game, cfg: SearchConfig):
    """B3 (pixel/silhouette match): if the level shows a STATIC target template plus a
    DYNAMIC canvas, return a goal-distance heuristic h(state) = number of canvas cells
    that mismatch the best-aligned target. Dense + monotone → lets best-first prune
    toward the goal on deep 'make the canvas match the target' games (cd82/re86/tr87).

    Inference (no labels, general): probe ~14 actions; STATIC = cells unchanged by every
    probe, DYNAMIC = cells changed by some probe. target = static non-background cells.
    Align the canvas to the target by their centroid offset, fix that offset, and score
    mismatches. Returns None (→ fall back to novelty) if no clear template/canvas split.
    """
    root = _resting_frame(game)
    h_, w_ = root.shape
    bg = int(np.bincount(root.ravel().astype(np.int64), minlength=16).argmax())
    static = np.ones((h_, w_), dtype=bool)
    dyn = np.zeros((h_, w_), dtype=bool)
    for ai in enumerate_candidate_actions(game, cfg)[:14]:
        with rng_snapshot() if cfg.rng_guard else contextlib.nullcontext():
            c = copy.deepcopy(game)
            f = _apply(c, ai)
        if f.shape != root.shape:
            return None
        changed = f != root
        static &= ~changed
        dyn |= changed

    target_mask = static & (root != bg)
    if int(target_mask.sum()) < 6 or int(dyn.sum()) < 6:
        return None

    ty, tx = np.where(target_mask)
    dyy, dxx = np.where(dyn)
    off_y = int(round(dyy.mean() - ty.mean()))
    off_x = int(round(dxx.mean() - tx.mean()))

    dyn_idx = np.argwhere(dyn)                      # canvas cells (N,2)
    src_y, src_x = dyn_idx[:, 0] - off_y, dyn_idx[:, 1] - off_x
    valid = (src_y >= 0) & (src_y < h_) & (src_x >= 0) & (src_x < w_)
    src_y, src_x, dst = src_y[valid], src_x[valid], dyn_idx[valid]
    keep = target_mask[src_y, src_x]               # only canvas cells whose source is a target cell
    src_y, src_x, dst = src_y[keep], src_x[keep], dst[keep]
    if len(dst) < 6:
        return None
    want = root[src_y, src_x]                       # desired value at each canvas cell
    dy_r, dx_r = dst[:, 0], dst[:, 1]

    def h(child, frame):
        return float(np.count_nonzero(frame[dy_r, dx_r] != want))

    return h


def plan_novelty(root_game, cfg: SearchConfig, agent_color=None) -> SearchResult:
    """B3 engine: novelty-guided best-first search.

    BFS is breadth-bound and stalls on deep levels (baseline 44+). This prioritizes
    expanding states in the LEAST-visited coarse bucket, diving toward unexplored
    configurations and reaching deep winning states with far fewer node expansions.
    Win detection stays EXACT (_score tick / WIN). Fully general — no goal or agent
    identification required; it simply rewards reaching novel states.

    NOTE: solutions may be longer than optimal (novelty != shortest path). That's an
    acceptable trade: completion dominates RHAE, and a long-but-completing plan still
    scores. (TODO: trace-compress the found plan before executing.)
    """
    t0 = time.time()
    target_score = int(root_game._score)
    with rng_snapshot() if cfg.rng_guard else contextlib.nullcontext():
        root = copy.deepcopy(root_game)
    root_view = view_of(root)

    def bucket(frame, lvl):
        if agent_color is not None:                        # reach mode: novelty over agent position
            cx, cy = _agent_centroid(frame, agent_color)
            return (lvl, cx // 2, cy // 2)
        return _coarse_bucket(frame, lvl, cfg.bucket_stride)  # generic: novelty over coarse frame

    bucket_count: "collections.Counter" = collections.Counter()
    bucket_count[bucket(_resting_frame(root), root_view.level_index)] += 1

    counter = 0
    # frontier item: (bucket_visit_count, depth, tiebreak, game, plan, view)
    frontier: list = [(0, 0, counter, root, [], root_view)]
    seen: set = {root_view}
    nodes = 0
    best_partial: list = []
    best_novelty = float("inf")

    def out_of_budget() -> bool:
        return nodes >= cfg.node_budget or (time.time() - t0) >= cfg.time_budget_s

    while frontier and not out_of_budget():
        _, _, _, game, plan, view = heapq.heappop(frontier)
        if len(plan) >= cfg.max_depth:
            continue
        for ai in enumerate_candidate_actions(game, cfg):
            if out_of_budget():
                break
            with rng_snapshot() if cfg.rng_guard else contextlib.nullcontext():
                child = copy.deepcopy(game)
                last_frame = _apply(child, ai)
            nodes += 1
            cview = view_of(child, last_frame)
            new_plan = plan + [ai]

            if int(child._score) > target_score or cview.is_win:
                return SearchResult(plan=new_plan, solved=True, nodes=nodes,
                                    elapsed_s=time.time() - t0, reason="level_completed")
            if cview.is_over or cview == view or cview in seen:
                continue
            seen.add(cview)

            b = bucket(last_frame, cview.level_index)
            cnt = bucket_count[b]
            bucket_count[b] = cnt + 1          # prefer states in least-visited buckets
            counter += 1
            heapq.heappush(frontier, (cnt, len(new_plan), counter, child, new_plan, cview))
            if cnt < best_novelty or (cnt == best_novelty and len(new_plan) > len(best_partial)):
                best_novelty = cnt             # most-novel (and, on ties, deepest) partial
                best_partial = new_plan         # used for the MPC fallback step

    return SearchResult(plan=best_partial, solved=False, nodes=nodes,
                        elapsed_s=time.time() - t0,
                        reason="budget_exhausted" if out_of_budget() else "frontier_empty")


# =============================================================================
# B3 (unified): Active-Inference best-first search
# =============================================================================
# Free-energy gate: two weight profiles. EXPLORE behaves like novelty (no goal
# gradient yet); EXPLOIT behaves like goal-directed A* once a progress gradient
# appears (template present, or structural progress recurs). Risk/loop fixed on.
# NOTE: COST (depth penalty) must stay ~0 during deep search — completion dominates
# RHAE, and shallow optimality is already handled by the separate BFS-short stage.
# A larger COST regressed deep reach (m0r0 2->0): 0.1*depth40 swamps the ~[0,1] signals.
_AI_EXPLORE = dict(INFO=1.0, PROG=0.3, COST=0.0, LOOP=1.0, RISK=0.5, CHURN=0.15)
_AI_EXPLOIT = dict(INFO=0.2, PROG=1.0, COST=0.02, LOOP=1.0, RISK=0.5, CHURN=0.05)
_PROG_OBJ, _PROG_PAL, _PROG_REACH, _PROG_TMPL = 1.0, 0.6, 0.5, 1.0


def plan_active_inference(root_game, cfg: SearchConfig, agent_color=None, match_h=None):
    """Unified goal‑agnostic best‑first search (the Seed‑IQ / Active‑Inference idea,
    but on our FREE exact world model — no learned/verified dynamics needed).

    priority (min‑heap, lower = expanded first) =
        W_COST*depth + W_LOOP*loop + W_RISK*risk − (W_PROG*progress + W_INFO*infogain)

    * progress  — generic STRUCTURAL progress toward an UNKNOWN goal (no labels):
                  object‑mass cleared (collect/erase/sokoban), a colour class fully
                  cleared, agent reach displacement, OR template‑mismatch reduction.
    * infogain  — novelty (least‑visited bucket) + a small structural‑churn prior.
    * loop/risk — soft penalties for near‑repeats and lethal neighbourhoods.
    A free‑energy gate flips EXPLORE→EXPLOIT once a real gradient appears.
    WIN‑detection stays EXACT (a clone's _score tick / WIN) and is checked before
    any score is computed — the heuristic only ORDERS the frontier, never decides
    success. Subsumes plan_novelty + the match/reach heuristics into one engine.
    """
    t0 = time.time()
    target = int(root_game._score)
    with rng_snapshot() if cfg.rng_guard else contextlib.nullcontext():
        root = copy.deepcopy(root_game)
    root_frame = _resting_frame(root)
    root_view = view_of(root, root_frame)
    start = _agent_centroid(root_frame, agent_color) if agent_color is not None else (-1, -1)
    m0 = float(match_h(root, root_frame)) if match_h is not None else 1.0

    def bucket(frame, lvl):
        if agent_color is not None:
            cx, cy = _agent_centroid(frame, agent_color)
            return (lvl, cx // 2, cy // 2)
        return _coarse_bucket(frame, lvl, cfg.bucket_stride)

    bucket_count: "collections.Counter" = collections.Counter()
    bucket_count[bucket(root_frame, root_view.level_index)] += 1
    recent: "collections.deque" = collections.deque(maxlen=16)   # global near-loop window

    exploit = match_h is not None            # template gradient present -> exploit, else explore
    counter = 0
    # frontier item: (priority, depth, tie, game, plan, view, frame, mismatch)
    frontier: list = [(0.0, 0, counter, root, [], root_view, root_frame, m0)]
    seen: set = {root_view}
    nodes = 0
    best_partial: list = []

    def oob():
        return nodes >= cfg.node_budget or (time.time() - t0) >= cfg.time_budget_s

    while frontier and not oob():
        _, _, _, game, plan, view, pf, pmis = heapq.heappop(frontier)
        if len(plan) >= cfg.max_depth:
            continue
        bg = int(np.bincount(pf.ravel().astype(np.int64), minlength=16).argmax())

        # Expand ALL children first so risk = fraction of lethal siblings (free).
        kids: list = []
        n_over = 0
        for ai in enumerate_candidate_actions(game, cfg):
            if oob():
                break
            with rng_snapshot() if cfg.rng_guard else contextlib.nullcontext():
                child = copy.deepcopy(game)
                cf = _apply(child, ai)
            nodes += 1
            cview = view_of(child, cf)
            if int(child._score) > target or cview.is_win:      # EXACT win — return now
                return SearchResult(plan + [ai], True, nodes, time.time() - t0, "level_completed")
            if cview.is_over:                                    # hard prune lethal
                n_over += 1
                continue
            if cview == view or cview in seen:                   # hard prune no-op / cycle
                continue
            kids.append((ai, child, cf, cview))
        n_kids = n_over + len(kids)
        risk = (n_over / n_kids) if n_kids else 0.0
        W = _AI_EXPLOIT if exploit else _AI_EXPLORE

        for ai, child, cf, cview in kids:
            seen.add(cview)
            new_plan = plan + [ai]

            # --- progress (goal-agnostic) ---
            # Template games: template-mismatch reduction drives (exploit, dense+monotone).
            # All others: agent-reach bias only. Generic object-mass / palette progress
            # was empirically too noisy — it flipped the exploit gate on board-changing
            # reach games and regressed them (m0r0 2->0) — so it is NOT used as a driver.
            if match_h is not None:
                m_now = float(match_h(child, cf))
                progress = max(0.0, pmis - m_now) / max(1.0, m0)
                child_mis = m_now
            else:
                ax, ay = _agent_centroid(cf, agent_color) if agent_color is not None else (-1, -1)
                p_reach = 0.0 if ax < 0 else (abs(ax - start[0]) + abs(ay - start[1])) / 128.0
                progress = min(1.0, _PROG_REACH * p_reach)
                child_mis = pmis

            # --- infogain (novelty + churn) ---
            b = bucket(cf, cview.level_index)
            cnt = bucket_count[b]
            bucket_count[b] = cnt + 1
            nov = 1.0 / (1.0 + cnt)
            churn = int(np.count_nonzero((cf != bg) ^ (pf != bg)))
            infogain = nov + W["CHURN"] * float(np.log1p(churn))

            # --- loop ---
            loop = 2.0 if cview.frame_key in recent else 0.0
            recent.append(cview.frame_key)

            priority = (W["COST"] * len(new_plan) + W["LOOP"] * loop + W["RISK"] * risk
                        - (W["PROG"] * progress + W["INFO"] * infogain))

            counter += 1
            heapq.heappush(frontier, (priority, len(new_plan), counter, child, new_plan, cview, cf, child_mis))
            if len(new_plan) > len(best_partial):
                best_partial = new_plan

    return SearchResult(plan=best_partial, solved=False, nodes=nodes,
                        elapsed_s=time.time() - t0,
                        reason="budget_exhausted" if oob() else "frontier_empty")


# =============================================================================
# B3 (win-condition inference): infer the unstated GOAL, use it as the search
# heuristic. This is the research-confirmed lever past goal-agnostic search
# (ARC_AGI_3_SOTA_research.md). We own the dynamics for free (deepcopy), so unlike
# SOTA executable-world-models we need only the GOAL, not the transition function.
#
# Backend 1 — mine_win_hypothesis(): NON-LLM hindsight miner. Learn the goal from a
#   level we DID solve and transfer it to this game's deeper levels (levels of one
#   game share the mechanic). Fully offline + testable now.
# Backend 2 — LLMWinSynthesizer: the SOTA path. A LOCAL code LLM proposes a Python
#   win_distance(); we verify it against rollout-labelled frames and feed back
#   counter-examples (propose->verify->refine). The LLM call is abstract (plug a GPU
#   model on Kaggle); the verifier + sandbox are exercised offline.
# =============================================================================
@dataclass
class WinHypothesis:
    """A learned goal + the distance heuristic it induces. distance(frame)->float is
    0 at the goal and grows with distance, so best-first prunes toward it."""
    kind: str               # 'clear_color' | 'reach_color' | 'none'
    color: int = -1
    agent_color: int = -1

    def is_active(self) -> bool:
        return self.kind in ("clear_color", "reach_color") and self.color >= 0

    def distance(self, frame: np.ndarray) -> float:
        f = np.asarray(frame)
        if self.kind == "clear_color":                       # goal = remove all of `color`
            return float(np.count_nonzero(f == self.color))  # remaining target cells
        if self.kind == "reach_color":                       # goal = agent onto a `color` cell
            tys, txs = np.where(f == self.color)
            if txs.size == 0:                                # target gone -> reached/consumed
                return 0.0
            ays, axs = np.where(f == self.agent_color)
            if axs.size == 0:
                return 64.0
            acx, acy = axs.mean(), ays.mean()
            return float(np.min(np.abs(txs - acx) + np.abs(tys - acy)))  # nearest target
        return 0.0

    def heuristic(self):
        """Adapter to plan_to_next_level's heuristic(game, frame) signature."""
        return (lambda game, frame: self.distance(frame)) if self.is_active() else None


def mine_win_hypothesis(f0: np.ndarray, f_prewin: np.ndarray, agent_color) -> WinHypothesis:
    """Hindsight: compare the level-start frame `f0` to the pre-win frame `f_prewin`
    (the resting frame just before the winning action) to guess the goal.

      * CLEAR-color: a non-background color whose population fell substantially on the
        way to the win → goal is consuming/clearing it (collect/eat games).
      * REACH-color: else, a RARE static (unchanged) non-bg color → a landmark the
        agent routes to (exit/door games).
    Both transfer across a game's levels because the mechanic is shared. Falls back to
    'none' (→ no heuristic, search stays goal-agnostic) when nothing is distinctive.
    """
    f0 = np.asarray(f0); f1 = np.asarray(f_prewin)
    ac = int(agent_color) if agent_color is not None else -1
    bg = int(np.bincount(f0.ravel().astype(np.int64), minlength=16).argmax())

    def counts(f):
        u, c = np.unique(f, return_counts=True)
        return {int(k): int(v) for k, v in zip(u, c) if int(k) not in (bg, ac) and int(k) >= 0}

    c0, c1 = counts(f0), counts(f1)
    drops = {col: c0[col] - c1.get(col, 0) for col in c0 if c0[col] - c1.get(col, 0) > 0}
    if drops:                                                # (a) clear-color
        col = max(drops, key=drops.get)
        if drops[col] >= max(1, 0.25 * c0[col]):             # a meaningful fraction cleared
            return WinHypothesis("clear_color", color=col, agent_color=ac)
    if ac >= 0:                                              # (b) reach-color landmark
        statics = {col: c0[col] for col in c0 if c1.get(col, 0) == c0[col]}
        if statics:
            col = min(statics, key=statics.get)              # rarest static = most goal-like
            if statics[col] <= 8:                            # a small landmark, not a wall/region
                return WinHypothesis("reach_color", color=col, agent_color=ac)
    return WinHypothesis("none")


class LLMWinSynthesizer:
    """SOTA path (offline GPU): a LOCAL code LLM writes a Python `win_distance(frame)`
    that is 0 at the goal and decreasing toward it; we VERIFY it against rollout-
    labelled frames from the free simulator and feed counter-examples back, Rodionov-
    style (propose -> verify -> refine). The LLM call is injected (plug a GPU model on
    Kaggle via `llm_complete`); the sandbox + verifier are exercised offline."""

    def __init__(self, llm_complete=None, max_rounds: int = 4):
        self.llm_complete = llm_complete     # callable(prompt:str)->code:str; None => unavailable
        self.max_rounds = max_rounds

    @staticmethod
    def _safe_exec(code: str):
        # Light sandbox: our OWN local model emits this (not adversarial web input), and
        # numpy needs real builtins (__import__) to run, so we allow them. The real guard
        # is verify() — incorrect code is rejected, never trusted. Harden with
        # RestrictedPython only if this is ever fed untrusted code.
        ns = {"np": np}
        exec(compile(code, "<win_distance>", "exec"), ns)
        fn = ns.get("win_distance")
        if not callable(fn):
            raise ValueError("generated code defines no win_distance(frame)")
        return fn

    @staticmethod
    def verify(fn, labelled):
        """labelled = [(frame, is_goal)]. Good ⇔ every goal frame scores strictly below
        every non-goal frame. Returns (ok, reason_or_None)."""
        goal = [float(fn(f)) for f, g in labelled if g]
        other = [float(fn(f)) for f, g in labelled if not g]
        if not goal or not other:
            return False, "need both goal and non-goal samples"
        if max(goal) < min(other):
            return True, None
        return False, f"goal<= {max(goal):.2f} not below non-goal>= {min(other):.2f}"

    def synthesize(self, observations, labelled):
        """Return a verified win_distance fn, or None (→ caller falls back to mining)."""
        if self.llm_complete is None:
            return None
        feedback = ""
        for _ in range(self.max_rounds):
            try:
                fn = self._safe_exec(self.llm_complete(self._prompt(observations, feedback)))
                ok, why = self.verify(fn, labelled)
            except Exception as e:                           # bad codegen -> refine
                ok, why, fn = False, f"{type(e).__name__}: {e}", None
            if ok:
                return fn
            feedback = f"Previous attempt failed: {why}. Return ONLY a corrected win_distance."
        return None

    @staticmethod
    def _prompt(observations, feedback):
        return ("Frames from an unknown 64x64 grid game are integer color arrays (numpy).\n"
                "Write `def win_distance(frame):` returning a float that is 0 at the goal and\n"
                "strictly larger the further from it. Use only numpy as `np`.\n"
                f"Observations: {observations}\n{feedback}\nReturn ONLY the function.")


# =============================================================================
# B2 policy: plan once per level, execute the cached plan action-by-action.
# Decoupled from the SDK Agent (see module docstring). Drives `arc_env` directly.
# =============================================================================
class ForwardModelPolicy:
    """Stateful policy. `choose_action` returns the next `ActionInput` to execute.

    Plan lifecycle:
      * if a cached plan exists, pop & return its next action;
      * else deepcopy `arc_env._game`, search for the level-completing plan, cache it;
      * if search SOLVED, commit the whole plan; if not, commit only the first step
        (Model-Predictive-Control: act once, then re-plan from the advanced state);
      * on GAME_OVER, RESET (engine does a level_reset) and re-plan.
    """

    def __init__(self, cfg: Optional[SearchConfig] = None, use_heuristic: bool = False,
                 win_inference: bool = True):
        self.cfg = cfg or SearchConfig()
        self.use_heuristic = use_heuristic
        self.win_inference = win_inference     # B3c: mine the goal from solved levels, transfer it
        self.win_hyp: Optional[WinHypothesis] = None
        self._plan: list = []
        self._reset_ai = None  # lazily built (needs the engine enums at runtime)

    def reset(self) -> None:
        self._plan = []

    def is_done(self, latest) -> bool:
        return latest is not None and latest.state == GameState.WIN

    def choose_action(self, arc_env, latest):
        """Return the next ActionInput to send to `arc_env.step`."""
        if self._reset_ai is None:
            self._reset_ai = ActionInput(id=GameAction.RESET)

        # Recover from a lethal state by resetting the current level.
        if latest is not None and latest.state == GameState.GAME_OVER:
            self._plan = []
            return self._reset_ai

        if not self._plan:
            game = getattr(arc_env, "_game", None)
            if game is None:
                # RED branch: no in-process game (frozen/remote step). Fall back.
                # TODO: route to a CNN / learned-world-model fallback policy here.
                log.warning("arc_env._game unavailable — RESET fallback (RED branch).")
                return self._reset_ai

            result = self._plan_level(game)
            log.info("plan: solved=%s len=%d nodes=%d %.2fs (%s)",
                     result.solved, len(result.plan), result.nodes,
                     result.elapsed_s, result.reason)

            if result.solved:
                self._plan = list(result.plan)                  # commit full solution
            elif result.plan:
                self._plan = [result.plan[0]]                   # MPC: one step, then re-plan
            else:
                cands = enumerate_candidate_actions(game, self.cfg)
                self._plan = [cands[0]] if cands else [self._reset_ai]

        return self._plan.pop(0)

    def _plan_level(self, game):
        """Cascade: optimal BFS first (shallow levels), then novelty best-first (deep).

        BFS gives the SHORTEST solution (best RHAE) when the level is shallow enough
        to reach within its small budget; otherwise novelty best-first dives deep to
        find a completing sequence. Completion dominates RHAE, so a longer novelty
        solution still beats not solving the level.
        """
        cfg = self.cfg
        bfs = plan_to_next_level(game, replace(cfg, time_budget_s=cfg.bfs_budget_s))
        if bfs.solved:
            self._mine(game, bfs.plan)                   # learn the goal from this easy win
            return bfs                                   # shallow levels: BFS is optimal (best RHAE)
        # B3c: learned win-predicate (cross-level transfer) — the tightest goal signal we
        # have once any level is solved; try it FIRST on deep levels.
        if self.win_inference and self.win_hyp is not None and self.win_hyp.is_active():
            wp = plan_to_next_level(game, replace(cfg, time_budget_s=cfg.novelty_budget_s),
                                    heuristic=self.win_hyp.heuristic())
            if wp.solved:
                return wp
        # B3a: pixel/silhouette-match heuristic (template games: cd82/re86/tr87).
        match_h = make_match_heuristic(game, cfg)
        if match_h is not None:
            mh = plan_to_next_level(game, replace(cfg, time_budget_s=cfg.novelty_budget_s),
                                    heuristic=match_h)
            if mh.solved:
                self._mine(game, mh.plan)
                return mh
        # B3b: reach search — agent-position novelty (the one B3 win so far: m0r0).
        agent_color = infer_agent_color(game, cfg)
        nov = plan_novelty(game, replace(cfg, time_budget_s=cfg.novelty_budget_s),
                           agent_color=agent_color)
        if nov.solved or len(nov.plan) >= len(bfs.plan):
            if nov.solved:
                self._mine(game, nov.plan)
            return nov
        return bfs

    def _mine(self, game, plan) -> None:
        """Hindsight win-condition inference: after solving a level, replay its plan on a
        clone to capture the level-start and pre-win frames, mine the goal, and cache it
        as `self.win_hyp` for this game's deeper levels. One-shot (first solve only)."""
        if not self.win_inference or self.win_hyp is not None or not plan:
            return
        try:
            agent_color = infer_agent_color(game, self.cfg)
            clone = copy.deepcopy(game)
            s0 = int(clone._score)
            f0 = _resting_frame(clone)
            prewin = f0
            for ai in plan:
                f = _apply(clone, ai)
                if int(clone._score) > s0:               # this action won; `prewin` precedes it
                    break
                prewin = f
            self.win_hyp = mine_win_hypothesis(f0, prewin, agent_color)
            log.info("mined win hypothesis: %s", self.win_hyp)
        except Exception as e:
            log.warning("win-mining failed: %s", e)
            self.win_hyp = WinHypothesis("none")
        # NOTE: plan_active_inference() unifies the above into one scorer (per the design
        # panel) but empirically did NOT beat this cascade — it regressed reach games
        # (m0r0 2->0) because its loop/risk/churn terms perturb the clean Go-Explore
        # ordering, and it supplies no better GOAL signal (the actual limiter). Kept as an
        # available engine for future tuning; not the default. See research: the real lever
        # is win-condition INFERENCE, not a fancier search framework.


# --- SDK swarm wiring (for the actual submission) ----------------------------
# To run under the official Swarm, wrap this policy in an `Agent` subclass WITHOUT
# importing `agents` (its __init__ eager-imports LLM templates that crash offline).
# Load agent.py in isolation, e.g.:
#     import importlib.util, sys, types
#     pkg = types.ModuleType("agents"); pkg.__path__ = [".../ARC-AGI-3-Agents/agents"]
#     sys.modules["agents"] = pkg                      # stub the package (skip __init__)
#     spec = importlib.util.spec_from_file_location("agents.agent", ".../agents/agent.py")
#     mod = importlib.util.module_from_spec(spec); sys.modules["agents.agent"] = mod
#     spec.loader.exec_module(mod); Agent = mod.Agent
# then subclass Agent; in choose_action, call policy.choose_action(self.arc_env, raw)
# and convert the ActionInput to a GameAction via `a = ai.id; a.set_data({**ai.data,
# "game_id": self.game_id}); return a`.  (TODO: harden + verify on the grader image.)


# =============================================================================
# B-1: the Day-1 Simulability Probe (run this BEFORE trusting the planner)
# =============================================================================
def simulability_probe(arc_env, n_actions: int = 6) -> dict:
    """Verify the load-bearing assumption: is `_game` exposed, free to clone, and
    deterministic? Run on every dev game (and re-run on the real grader image).

    Checks:
      1. EXPOSED      — arc_env._game is a live ARCBaseGame.
      2. ISOLATED     — stepping a deepcopy does NOT change the original's
                        _action_count / _score (=> simulated steps are free).
      3. DETERMINISTIC— two deepcopies + same action sequence => identical frames.
    GREEN => forward-model search is the primary engine.
    """
    verdict = {"exposed": False, "isolated": False, "deterministic": False,
               "color": "RED", "detail": ""}
    game = getattr(arc_env, "_game", None)
    if game is None or not isinstance(game, ARCBaseGame):
        verdict["detail"] = "arc_env._game is None / not an ARCBaseGame"
        return verdict
    verdict["exposed"] = True

    orig_count, orig_score = int(game._action_count), int(game._score)
    cands = enumerate_candidate_actions(game, SearchConfig())
    if not cands:
        verdict["detail"] = "no candidate actions to probe"
        return verdict
    seq = [cands[i % len(cands)] for i in range(n_actions)]

    # ISOLATION: step a clone; the original must be untouched.
    with rng_snapshot():
        c1 = copy.deepcopy(game)
        frames_a = [np.ascontiguousarray(_apply(c1, ai)).tobytes() for ai in seq]
    verdict["isolated"] = (int(game._action_count) == orig_count
                           and int(game._score) == orig_score)

    # DETERMINISM: a second clone + same sequence must match byte-for-byte.
    with rng_snapshot():
        c2 = copy.deepcopy(game)
        frames_b = [np.ascontiguousarray(_apply(c2, ai)).tobytes() for ai in seq]
    verdict["deterministic"] = frames_a == frames_b

    if verdict["isolated"] and verdict["deterministic"]:
        verdict["color"] = "GREEN"
    elif verdict["isolated"]:
        verdict["color"] = "YELLOW"   # free but stochastic -> deepcopy-only + RNG guard
    verdict["detail"] = (f"exposed={verdict['exposed']} isolated={verdict['isolated']} "
                         f"deterministic={verdict['deterministic']}")
    return verdict


# =============================================================================
# Offline harness — load a public game and drive it with our own loop.
# (Dev convenience; the real submission wires the policy into the SDK swarm.)
# =============================================================================
def load_local_wrapper(game_dir: str | Path, seed: int = 0):
    """Build a LocalEnvironmentWrapper for environment_files/<id>/<ver>/ (no scorecard).

    `metadata.json` provides game_id/default_fps/tags/baseline_actions; `class_name`
    defaults from game_id (e.g. "cd82" -> class `Cd82`); the file is "<id>.py".
    """
    try:
        from arc_agi.local_wrapper import LocalEnvironmentWrapper
        from arc_agi.models import EnvironmentInfo
    except ImportError as e:
        # arc_agi/__init__ eager-imports .api/.base/.remote_wrapper, which pull flask +
        # requests (both ARE in the offline wheels — ensure they're installed on the image).
        raise RuntimeError(
            "arc_agi import failed — its package __init__ eager-imports flask (.api) and "
            f"requests (.base/.remote_wrapper). Install flask+requests on the image. ({e})"
        ) from e

    game_dir = Path(game_dir)
    meta = json.loads((game_dir / "metadata.json").read_text(encoding="utf-8"))
    info = EnvironmentInfo(
        game_id=str(meta["game_id"]).split("-")[0],   # "cd82-fb555c5d" -> "cd82"
        title=meta.get("title"),
        default_fps=meta.get("default_fps"),
        tags=meta.get("tags"),
        baseline_actions=meta.get("baseline_actions"),
        local_dir=str(game_dir),
    )
    return LocalEnvironmentWrapper(
        environment_info=info,
        logger=log,
        scorecard_id="local-dev",
        seed=seed,
        scorecard_manager=None,    # <- no scoring side effects in dev
    )


def rhae_level_score(baseline_actions: int, agent_actions: int) -> float:
    """The shipped per-level RHAE: min(1.15, (baseline/agent)^2), 0 if not solved."""
    if agent_actions <= 0:
        return 0.0
    return min(1.15, (baseline_actions / agent_actions) ** 2)


def run_episode(arc_env, policy: ForwardModelPolicy, max_actions: int) -> dict:
    """Self-contained agent loop (same semantics as the SDK `Agent.main`) driving
    `arc_env` directly. The wrapper auto-RESETs in __init__, so we start from its
    last response. Returns a per-game summary.
    """
    latest = arc_env.observation_space          # FrameDataRaw from the initial reset
    actions = 0
    while actions <= max_actions:
        if policy.is_done(latest):
            break
        ai = policy.choose_action(arc_env, latest)
        raw = arc_env.step(ai.id, data=dict(ai.data or {}), reasoning=None)
        if raw is None:
            log.warning("arc_env.step returned None; stopping.")
            break
        latest = raw
        actions += 1
        if latest.state == GameState.WIN:
            break

    return {
        "state": latest.state.name if latest is not None else "NONE",
        "levels_completed": int(latest.levels_completed) if latest is not None else 0,
        "win_levels": int(latest.win_levels) if latest is not None else 0,
        "actions_taken": actions,
    }


def run_local(game_dir: str | Path, cfg: Optional[SearchConfig] = None,
              max_actions: int = 5_000) -> dict:
    """Probe, then run the forward-model policy on one local game; return a summary.

    `max_actions` mirrors the server termination rule (~5x human baseline); the SDK
    default of 80 is far too low for real games (baselines reach 578).
    """
    if not _ENGINE_AVAILABLE:
        raise RuntimeError(
            "arcengine/arc_agi not importable here (Linux cp312 wheels). "
            f"Run on the Kaggle Linux image. Import error: {_IMPORT_ERROR!r}")

    cfg = cfg or SearchConfig()
    wrapper = load_local_wrapper(game_dir)

    probe = simulability_probe(wrapper)
    log.info("Simulability probe: %s", probe)
    if probe["color"] == "RED":
        return {"game_dir": str(game_dir), "probe": probe, "note": "RED — pivot to fallback"}

    t0 = time.time()
    summary = run_episode(wrapper, ForwardModelPolicy(cfg), max_actions)
    summary["wall_clock_s"] = round(time.time() - t0, 2)
    summary["game_dir"] = str(game_dir)
    summary["probe_color"] = probe["color"]
    return summary


if __name__ == "__main__":  # pragma: no cover
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Forward-model agent — local runner")
    ap.add_argument("game_dir", help="path to environment_files/<id>/<ver>/")
    ap.add_argument("--time-budget", type=float, default=8.0)
    ap.add_argument("--max-depth", type=int, default=64)
    ap.add_argument("--max-actions", type=int, default=5000)
    args = ap.parse_args()
    cfg = SearchConfig(time_budget_s=args.time_budget, max_depth=args.max_depth)
    print(json.dumps(run_local(args.game_dir, cfg, max_actions=args.max_actions), indent=2))
