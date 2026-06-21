"""
Offline unit tests for the torch/SDK-free exploration core of my_agent.py.
Runs on the Windows dev box (numpy only). Validates the algorithm before it ever
touches the Kaggle gateway.

    python submission/test_explore.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from my_agent import (ExplorationPolicy, _Node, ChangeStats, action_feature,  # noqa: E402
                      candidate_keys, click_candidates,
                      SIMPLE_IDS, CLICK_BASE, N_SIMPLE, GRID, _HAS_TORCH, _HAS_SDK)
import numpy as np  # noqa: E402


# --- a tiny deterministic "game": 4 states in a corridor, 2 actions each ---
class MockEnv:
    # action 0 = forward (A->B->C->D, D absorbs); action 1 = no-op (stay)
    FWD = {"A": "B", "B": "C", "C": "D", "D": "D"}

    def __init__(self):
        self.state = "A"

    def candidates(self):
        return [0, 1]

    def step(self, akey):
        self.state = self.FWD[self.state] if akey == 0 else self.state
        return self.state


def test_corridor_coverage():
    """Drive the policy through the corridor; it must discover every reachable state and
    record the durable forward edges (sweeps may clear `tried`, but `edges` persist)."""
    random.seed(0)
    np.random.seed(0)
    env = MockEnv()
    pol = ExplorationPolicy(epsilon=0.0)
    prev_h = prev_ak = None
    reasons = []
    for _ in range(60):
        h = env.state
        if prev_h is not None:
            pol.observe(prev_h, prev_ak, h)
        ak, reason = pol.select(h, env.candidates(), prior=None)
        reasons.append(reason)
        prev_h, prev_ak = h, ak
        env.step(ak)
    pol.observe(prev_h, prev_ak, env.state)

    assert pol.visited >= {"A", "B", "C", "D"}, f"coverage failed: {pol.visited}"
    assert pol.graph["A"].edges[0] == "B", pol.graph["A"].edges
    assert pol.graph["B"].edges[0] == "C", pol.graph["B"].edges
    assert pol.graph["C"].edges[0] == "D", pol.graph["C"].edges
    assert "explore" in reasons, f"never explored untried-first: {set(reasons)}"
    print("  ok: corridor fully discovered, forward edges recorded")


def test_frontier_routing():
    """When the current state is exhausted but another reachable state has an untried
    action, _plan_to_frontier must return the shortest route to it."""
    pol = ExplorationPolicy()
    pol.graph["A"] = _Node([0]); pol.graph["A"].tried = {0}; pol.graph["A"].edges = {0: "B"}
    pol.graph["B"] = _Node([0]); pol.graph["B"].tried = {0}; pol.graph["B"].edges = {0: "C"}
    pol.graph["C"] = _Node([0, 1]); pol.graph["C"].tried = {0}   # action 1 still untried
    path = pol._plan_to_frontier("A")
    assert path is not None, "no route found to the untried frontier"
    assert [ak for ak, _ in path] == [0, 0], f"wrong route: {list(path)}"
    assert path[-1][1] == "C", "route does not end at the untried state"
    print("  ok: frontier BFS routes A->B->C to the untried action")


def test_edge_verification_deletes_bad_edge():
    """A planned step whose outcome diverges from the stored edge must drop that edge."""
    pol = ExplorationPolicy()
    pol.select("A", [0, 1], prior={0: 1.0, 1: 0.0})    # create node A
    pol.observe("A", 0, "B")                           # record A --0--> B
    assert pol.graph["A"].edges[0] == "B"
    pol._pending = ("A", 0, "B")                       # we expected B from the plan...
    pol.observe("A", 0, "X")                           # ...reality gave X
    assert 0 not in pol.graph["A"].edges, "stale edge not deleted"
    assert len(pol.plan) == 0, "plan not cleared after divergence"
    print("  ok: divergent edge deleted, plan reset")


def test_no_change_streak_and_target():
    """observe returns 1.0 for novel, 0.6 for changed-but-seen, 0.0 for no-change."""
    pol = ExplorationPolicy()
    pol.select("A", [0, 1, 2])
    assert pol.observe("A", 0, "B") == 1.0             # novel
    assert pol.observe("A", 1, "B") == 0.6             # changed but B already seen
    pol.select("B", [0])
    assert pol.observe("B", 0, "B") == 0.0             # no-change (self)
    assert pol.no_change_streak == 1
    print("  ok: novelty targets + no-change streak correct")


def test_candidate_generation():
    """Click candidates target coloured blobs (not background); simple ids respect avail."""
    frame = np.zeros((64, 64), dtype=np.uint8)         # background = 0
    frame[10:14, 20:24] = 5                            # a 4x4 blob of colour 5
    clicks = click_candidates(frame)
    assert clicks, "no click candidates on a non-empty frame"
    for key in clicks:                                 # every click lands on the blob region
        ci = key - CLICK_BASE
        y, x = ci // 64, ci % 64
        assert 10 <= y < 14 and 20 <= x < 24, f"click off-blob at ({y},{x})"
    keys = candidate_keys(frame, {1, 3, 6})            # only ACTION1, ACTION3, ACTION6 available
    simple = [k for k in keys if k < len(SIMPLE_IDS)]
    assert set(SIMPLE_IDS[k] for k in simple) == {1, 3}, "simple-action filtering wrong"
    print("  ok: candidate generation targets blobs + respects availability")


def test_change_stats_prior():
    """The tabular prior must learn which action features change the frame and rank a
    proven-changing feature above a proven-no-op one."""
    st = ChangeStats()
    for _ in range(10):
        st.update(("s", 0), True)      # ACTION1 always changes
        st.update(("s", 1), False)     # ACTION2 never changes
    assert st.rate(("s", 0)) > 0.8, st.rate(("s", 0))
    assert st.rate(("s", 1)) < 0.2, st.rate(("s", 1))
    assert abs(st.rate(("s", 2)) - 0.5) < 1e-9, "unseen feature must sit at the 0.5 prior"

    # click features key by the clicked cell's colour, generalising across positions
    frame = np.zeros((GRID, GRID), dtype=np.uint8)
    frame[5, 7] = 9
    assert action_feature(frame, 0) == ("s", 0)
    assert action_feature(frame, CLICK_BASE + 5 * GRID + 7) == ("c", 9)
    print("  ok: change prior learns changing vs no-op features; click keyed by colour")


if __name__ == "__main__":
    print(f"env: torch={_HAS_TORCH} sdk={_HAS_SDK} (both False is fine for these tests)\n")
    test_corridor_coverage()
    test_frontier_routing()
    test_change_stats_prior()
    test_edge_verification_deletes_bad_edge()
    test_no_change_streak_and_target()
    test_candidate_generation()
    print("\nALL TESTS PASSED")
