"""
Plug a bespoke `win_distance(frame)` into the forward-model search and measure how many
levels it cracks vs the goal-agnostic cascade.

This ISOLATES win-condition INFERENCE from search: a predicate is derived from the game's
own win logic, so if even a (near-perfect) win_distance can't crack a deep level, the wall
is SEARCH DEPTH, not goal knowledge. If it can, win-condition inference is the real lever
and the LLM-synthesis path (LLMWinSynthesizer) is worth building for the Kaggle GPU.

Per level it runs BOTH arms and advances with the better:
  * cascade  — the shipped goal-agnostic cascade (BFS -> match -> reach)
  * predicate — best-first with the supplied win_distance as the heuristic
Reports which levels ONLY the predicate solved (the win-inference payoff).

Usage:
  python agent/test_predicate.py <game_id> <predicate.py> [budget_s]   # A/B test
  python agent/test_predicate.py <game_id> --dump [budget_s]           # show reached frames
predicate.py must define  win_distance(frame) -> float  (0 at goal, larger further away).
"""
import glob
import importlib.util
import logging
import sys
import time
from dataclasses import replace
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


def _ver_dir(game_id):
    ds = glob.glob(str(ROOT / "arc-prize-2026-arc-agi-3" / "environment_files" / game_id / "*"))
    if not ds:
        sys.exit(f"game {game_id} not found")
    return Path(ds[0])


def _load_pred(path):
    spec = importlib.util.spec_from_file_location("pred", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if not hasattr(m, "win_distance"):
        sys.exit(f"{path} defines no win_distance(frame)")
    return m.win_distance


def _grid(frame, stride=2):
    """Compact text render of a frame (downsampled) for an agent to read."""
    f = np.asarray(frame)[::stride, ::stride]
    return "\n".join("".join(format(int(c), "x") if 0 <= c < 16 else "?" for c in row) for row in f)


def dump(game_id, budget):
    """Advance with the cascade and print each reached level's start frame + palette."""
    game, meta = load_game(_ver_dir(game_id))
    env = MiniEnv(game)
    cfg = fma.SearchConfig(bfs_budget_s=2.5, novelty_budget_s=budget)
    pol = fma.ForwardModelPolicy(cfg, win_inference=False)
    total = int(env.observation_space.win_levels)
    ac = fma.infer_agent_color(env._game, cfg)
    print(f"# {game_id}: {total} levels | inferred agent_color={ac} | baselines={meta.get('baseline_actions')}")
    while env.observation_space.state != GameState.WIN:
        lvl = int(env._game._score)
        f = fma._resting_frame(env._game)
        u, c = np.unique(f, return_counts=True)
        print(f"\n--- Level {lvl + 1} start | palette {dict(zip(u.tolist(), c.tolist()))}")
        print(_grid(f))
        r = pol._plan_level(env._game)
        if not r.solved:
            print(f"--- cascade STALLS at L{lvl + 1} ({r.reason}) — this is the level to crack")
            break
        for ai in r.plan:
            env.step(ai.id, data=dict(ai.data or {}))


def ab(game_id, pred_path, budget):
    wd = _load_pred(pred_path)
    game, meta = load_game(_ver_dir(game_id))
    env = MiniEnv(game)
    cfg = fma.SearchConfig(bfs_budget_s=2.5, novelty_budget_s=budget)
    pol = fma.ForwardModelPolicy(cfg, win_inference=False)
    total = int(env.observation_space.win_levels)
    pred_only, both, casc_only, rows = [], [], [], []
    t0 = time.time()
    while env.observation_space.state != GameState.WIN:
        lvl = int(env._game._score) + 1
        rc = pol._plan_level(env._game)
        try:
            rp = fma.plan_to_next_level(env._game, replace(cfg, time_budget_s=budget),
                                        heuristic=lambda g, f: float(wd(f)))
        except Exception as e:
            rp = fma.SearchResult(plan=[], solved=False, nodes=0, elapsed_s=0.0,
                                  reason=f"pred_error:{type(e).__name__}")
        cs, ps = rc.solved, rp.solved
        if ps and (not cs or len(rp.plan) <= len(rc.plan)):
            pick = rp
        elif cs:
            pick = rc
        else:
            rows.append(f"  L{lvl}: STALL (casc:{rc.reason} pred:{rp.reason})")
            break
        tag = "both" if (cs and ps) else ("PRED-ONLY" if ps else "casc")
        (pred_only if tag == "PRED-ONLY" else both if tag == "both" else casc_only).append(lvl)
        rows.append(f"  L{lvl}: {tag:9} pick={len(pick.plan)}a (casc {'Y' if cs else 'n'}/{len(rc.plan)}a, "
                    f"pred {'Y' if ps else 'n'}/{len(rp.plan)}a)")
        for ai in pick.plan:
            env.step(ai.id, data=dict(ai.data or {}))
    solved = len(pred_only) + len(both) + len(casc_only)
    print(f"{game_id}: {solved}/{total} levels in {time.time() - t0:.0f}s | "
          f"PRED-ONLY (win-inference cracked these) = {pred_only or 'NONE'}")
    print("\n".join(rows))
    return solved, pred_only


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    game_id, arg2 = sys.argv[1], sys.argv[2]
    if arg2 == "--dump":
        dump(game_id, float(sys.argv[3]) if len(sys.argv) > 3 else 40.0)
    else:
        ab(game_id, arg2, float(sys.argv[3]) if len(sys.argv) > 3 else 40.0)


if __name__ == "__main__":
    main()
