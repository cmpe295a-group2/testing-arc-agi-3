"""
Parallel game runner + budget governor for the forward-model ARC-AGI-3 agent.

The search is CPU-bound (deepcopy + simulate are pure-Python → GIL-bound), so we
parallelize ACROSS GAMES with a PROCESS pool (not threads). With C usable cores and
G games in a wall-clock budget T, each game gets ~ T*C/G of compute — e.g. 9h on a
4-core box over 110 private games ≈ 20 min/game, far more than a single-thread sweep,
which lets the borderline levels (stuck only for lack of time, not depth) get solved.

Windows-testable (uses only the pure-python arcengine wheel via the MiniEnv shim, no
arc_agi/flask). The Kaggle submission wires the same process-pool + governor around
the SDK's OFFLINE LocalEnvironmentWrapper — see planning/SUBMISSION.md.

Usage:
    python agent/run_parallel.py [--workers N] [--total-budget-min M] [--bfs S] [--novelty S] [--cap S]
"""
import argparse
import glob
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Make arcengine (pure-python wheel) + the agent package importable in EVERY worker:
# this module-level setup re-runs on import in each spawned process.
sys.path.insert(0, str(ROOT / "arc-prize-2026-arc-agi-3" / "arc_agi_3_wheels"
                       / "arcengine-0.9.3-py3-none-any.whl"))
sys.path.insert(0, str(ROOT / "agent"))

ENV_GLOB = str(ROOT / "arc-prize-2026-arc-agi-3" / "environment_files" / "*" / "*")


def eval_one(task):
    """Worker: solve one game level-by-level under a per-game time cap. Picklable
    (module-level fn, plain-tuple arg, dict return) so it survives spawn on Windows."""
    game_dir, bfs_s, nov_s, cap_s = task
    logging.disable(logging.CRITICAL)
    import forward_model_agent as fma
    from try_run import load_game, MiniEnv
    from arcengine import GameState

    gid = Path(game_dir).parent.name
    try:
        game, meta = load_game(Path(game_dir))
    except Exception as e:  # one bad game must not kill the pool
        return {"game": gid, "solved": 0, "total": 0, "rhae": 0.0, "time_s": 0.0,
                "stop": f"ERR {type(e).__name__}"}

    env = MiniEnv(game)
    total = int(env.observation_space.win_levels)
    baselines = meta.get("baseline_actions") or []
    pol = fma.ForwardModelPolicy(fma.SearchConfig(bfs_budget_s=bfs_s, novelty_budget_s=nov_s))
    rows = []
    t0 = time.time()
    stop = "cap"
    while env.observation_space.state != GameState.WIN and time.time() - t0 < cap_s:
        lvl = int(env._game._score)
        r = pol._plan_level(env._game)
        if not r.solved:
            stop = f"L{lvl + 1}:{r.reason}"
            break
        for ai in r.plan:
            env.step(ai.id, data=dict(ai.data or {}))
        base = baselines[lvl] if lvl < len(baselines) else None
        rows.append((lvl + 1, len(r.plan), base,
                     fma.rhae_level_score(base, len(r.plan)) if base else 0.0))
    if env.observation_space.state == GameState.WIN:
        stop = "WIN"

    allw = sum(range(1, total + 1)) or 1
    weighted = sum(l * s for l, _, _, s in rows) / allw
    cap = sum(l for l, _, _, _ in rows) / allw
    return {"game": gid, "solved": len(rows), "total": total,
            "rhae": min(weighted, cap), "time_s": time.time() - t0, "stop": stop}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--total-budget-min", type=float, default=0.0,
                    help="if >0, derive per-game cap = total*workers/n_games (budget governor)")
    ap.add_argument("--bfs", type=float, default=2.5)
    ap.add_argument("--novelty", type=float, default=10.0)
    ap.add_argument("--cap", type=float, default=45.0, help="per-game cap (ignored if --total-budget-min>0)")
    args = ap.parse_args()

    dirs = sorted(glob.glob(ENV_GLOB))
    cap = args.cap
    if args.total_budget_min > 0:                     # budget governor
        cap = args.total_budget_min * 60.0 * args.workers / max(1, len(dirs))
    tasks = [(d, args.bfs, args.novelty, cap) for d in dirs]

    print(f"# {len(dirs)} games | {args.workers} workers | cap {cap:.0f}s/game "
          f"| BFS {args.bfs}s -> novelty {args.novelty}s")
    print(f"{'game':5} {'solved':>7} {'gRHAE':>6} {'time':>6}  stop")
    print("-" * 50)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(eval_one, tasks):
            results.append(r)
            print(f"{r['game']:5} {r['solved']:>3}/{r['total']:<3} {r['rhae']:>6.3f} "
                  f"{r['time_s']:>5.0f}s  {r['stop']}")
    wall = time.time() - t0

    solved = sum(r["solved"] for r in results)
    levels = sum(r["total"] for r in results)
    cpu = sum(r["time_s"] for r in results)
    mean_rhae = sum(r["rhae"] for r in results) / len(results) if results else 0.0
    print("-" * 50)
    print(f"TOTAL: {solved}/{levels} levels | mean game-RHAE {mean_rhae:.3f}")
    print(f"wall {wall:.0f}s | cpu-time {cpu:.0f}s | speedup ~{cpu / max(wall, 0.1):.1f}x "
          f"on {args.workers} workers")


if __name__ == "__main__":
    main()
