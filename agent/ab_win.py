"""A/B test for win-condition inference (B3c): run each game with win_inference OFF
vs ON and compare levels solved + which hypothesis was mined. Also smoke-tests the
LLMWinSynthesizer sandbox+verifier with a stub 'LLM'.

Usage: python agent/ab_win.py [bfs_s] [novelty_s] [cap_s] [game ...]
"""
import glob
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "arc-prize-2026-arc-agi-3" / "arc_agi_3_wheels"
                       / "arcengine-0.9.3-py3-none-any.whl"))
sys.path.insert(0, str(ROOT / "agent"))
logging.disable(logging.CRITICAL)

import numpy as np  # noqa: E402
import forward_model_agent as fma  # noqa: E402
from try_run import load_game, MiniEnv  # noqa: E402
from arcengine import GameState  # noqa: E402


def solve(ver_dir, win_inf, bfs, nov, cap):
    game, _ = load_game(ver_dir)
    env = MiniEnv(game)
    pol = fma.ForwardModelPolicy(fma.SearchConfig(bfs_budget_s=bfs, novelty_budget_s=nov),
                                 win_inference=win_inf)
    n = 0
    t0 = time.time()
    while env.observation_space.state != GameState.WIN and time.time() - t0 < cap:
        r = pol._plan_level(env._game)
        if not r.solved:
            break
        for ai in r.plan:
            env.step(ai.id, data=dict(ai.data or {}))
        n += 1
    return n, pol.win_hyp, time.time() - t0


def smoke():
    # WinHypothesis distance sanity
    f = np.zeros((8, 8), dtype=int); f[0, 0] = 5; f[7, 7] = 3   # agent=5 at (0,0), target=3 at (7,7)
    reach = fma.WinHypothesis("reach_color", color=3, agent_color=5)
    clear = fma.WinHypothesis("clear_color", color=3)
    assert reach.distance(f) == 14.0, reach.distance(f)         # Manhattan (0,0)->(7,7)
    assert clear.distance(f) == 1.0                              # one target cell remains
    # LLM synthesizer: stub 'LLM' returns a correct win_distance; verifier must accept it
    g0 = np.zeros((4, 4), dtype=int); g0[0, 0] = 7               # goal frame (no target color 2)
    n0 = np.zeros((4, 4), dtype=int); n0[1, 1] = 2               # non-goal (target present)
    labelled = [(g0, True), (n0, False)]
    stub = lambda prompt: "def win_distance(frame):\n    return float((frame == 2).sum())"
    fn = fma.LLMWinSynthesizer(llm_complete=stub).synthesize("frames", labelled)
    assert fn is not None and fn(g0) == 0.0 and fn(n0) == 1.0
    # no-LLM path returns None (caller falls back to mining)
    assert fma.LLMWinSynthesizer(llm_complete=None).synthesize("x", labelled) is None
    print("smoke: WinHypothesis + LLMWinSynthesizer verifier OK\n")


def main():
    a = sys.argv[1:]
    bfs = float(a[0]) if a and a[0].replace(".", "").isdigit() else 2.5
    nov = float(a[1]) if len(a) > 1 and a[1].replace(".", "").isdigit() else 15.0
    cap = float(a[2]) if len(a) > 2 and a[2].replace(".", "").isdigit() else 80.0
    games = [x for x in a if not x.replace(".", "").isdigit()] or \
            ["m0r0", "cd82", "vc33", "tu93", "ft09", "sp80"]
    smoke()
    print(f"# A/B win-inference | bfs {bfs}s nov {nov}s cap {cap:.0f}s")
    print(f"{'game':5} {'off':>4} {'on':>4}  mined-hypothesis")
    print("-" * 56)
    for g in games:
        ds = glob.glob(str(ROOT / "arc-prize-2026-arc-agi-3" / "environment_files" / g / "*"))
        if not ds:
            print(f"{g:5}  (not found)"); continue
        off, _, _ = solve(Path(ds[0]), False, bfs, nov, cap)
        on, hyp, _ = solve(Path(ds[0]), True, bfs, nov, cap)
        flag = "  <== +" + str(on - off) if on > off else ("  <== -" + str(off - on) if on < off else "")
        h = f"{hyp.kind}(c={hyp.color})" if hyp and hyp.is_active() else (hyp.kind if hyp else "none")
        print(f"{g:5} {off:>4} {on:>4}  {h}{flag}")


if __name__ == "__main__":
    main()
