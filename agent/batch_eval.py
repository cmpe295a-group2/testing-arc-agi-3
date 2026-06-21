"""
Batch coverage eval — run the forward-model cascade on all 25 public games and
report levels solved + per-game RHAE. Windows-friendly (needs only `arcengine`).

Usage:  python agent/batch_eval.py  [bfs_s] [novelty_s] [per_game_cap_s]
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

import forward_model_agent as fma         # noqa: E402
from try_run import load_game, MiniEnv    # noqa: E402
from arcengine import GameState           # noqa: E402

BASE = ROOT / "arc-prize-2026-arc-agi-3" / "environment_files"


def game_rhae(rows, total_levels: int) -> float:
    """Per-game score = min(completion_cap, level-weighted avg), weight = level idx."""
    allw = sum(range(1, total_levels + 1)) or 1
    weighted = sum(lvl * s for lvl, _, _, s in rows) / allw     # uncompleted contribute 0
    cap = sum(lvl for lvl, _, _, _ in rows) / allw              # completion cap
    return min(weighted, cap)


def eval_game(game_dir: Path, cfg: "fma.SearchConfig", per_game_cap: float) -> dict:
    game, meta = load_game(game_dir)
    env = MiniEnv(game)
    total = env.observation_space.win_levels
    baselines = meta.get("baseline_actions") or []
    pol = fma.ForwardModelPolicy(cfg)
    rows: list = []
    t0 = time.time()
    stop = ""
    while env.observation_space.state != GameState.WIN and time.time() - t0 < per_game_cap:
        lvl = int(env._game._score)                       # 0-indexed level being solved
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
    elif not stop:
        stop = "time_cap"
    return {"solved": len(rows), "total": total, "tags": meta.get("tags"),
            "rhae": game_rhae(rows, total), "time_s": time.time() - t0, "stop": stop}


def main():
    bfs_s = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    nov_s = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    cap = float(sys.argv[3]) if len(sys.argv) > 3 else 45.0
    cfg = fma.SearchConfig(bfs_budget_s=bfs_s, novelty_budget_s=nov_s)

    dirs = sorted(glob.glob(str(BASE / "*" / "*")))
    print(f"# cascade: BFS {bfs_s}s -> novelty {nov_s}s/level, cap {cap}s/game, {len(dirs)} games\n")
    print(f"{'game':5} {'solved':>7} {'gRHAE':>6} {'tags':16} {'time':>6}  stop")
    print("-" * 62)
    summ = []
    t_all = time.time()
    for d in dirs:
        gid = Path(d).parent.name
        try:
            r = eval_game(Path(d), cfg, cap)
        except Exception as e:
            print(f"{gid:5}     ERR  {type(e).__name__}: {str(e)[:42]}")
            summ.append((gid, 0, 0, 0.0))
            continue
        print(f"{gid:5} {r['solved']:>3}/{r['total']:<3} {r['rhae']:>6.3f} "
              f"{str(r['tags']):16} {r['time_s']:>5.0f}s  {r['stop']}")
        summ.append((gid, r["solved"], r["total"], r["rhae"]))

    tot_solved = sum(s for _, s, _, _ in summ)
    tot_levels = sum(t for _, _, t, _ in summ)
    mean_rhae = sum(x[3] for x in summ) / len(summ) if summ else 0.0
    print("-" * 62)
    print(f"TOTAL: {tot_solved}/{tot_levels} levels solved | mean game-RHAE {mean_rhae:.3f} "
          f"| {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
