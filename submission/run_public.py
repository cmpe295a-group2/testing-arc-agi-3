"""
Run the submission agent (my_agent.MyAgent) on the 25 PUBLIC games locally, end-to-end.

The 110 scored private games are remote (frame-only via gateway), but the 25 public games
load in-process from the arcengine wheel, so we can drive the REAL agent against them
without the SDK/gateway. This exercises the full choose_action loop and reports how many
levels the frame-only exploration completes.

NOTE: the dev box has no torch → the agent runs in GRAPH-ONLY mode (no ChangeNet prior),
i.e. a LOWER BOUND on the real (GPU, CNN-guided) agent. It still validates the integration
and the exploration's ability to complete shallow levels.

    python submission/run_public.py [game ...] [--workers N] [--max-actions M] [--time-budget S]
"""
import argparse
import glob
import inspect
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "arc-prize-2026-arc-agi-3" / "arc_agi_3_wheels"
                       / "arcengine-0.9.3-py3-none-any.whl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
ENV_GLOB = str(ROOT / "arc-prize-2026-arc-agi-3" / "environment_files" / "*" / "*")


def load_game(game_dir, seed=0):
    meta = json.loads((game_dir / "metadata.json").read_text(encoding="utf-8"))
    gid = str(meta["game_id"]).split("-")[0]
    cls_name = gid[:4][0].upper() + gid[:4][1:]
    ns = {"__name__": f"game_{gid}"}
    exec(compile((game_dir / f"{gid}.py").read_text(encoding="utf-8"), f"{gid}.py", "exec"), ns)
    cls = ns[cls_name]
    game = cls(seed=seed) if "seed" in inspect.signature(cls).parameters else cls()
    return game, meta


def run_one(task):
    game_dir, max_actions, time_budget = task
    logging.disable(logging.CRITICAL)
    from arcengine import ActionInput, GameAction, GameState
    import my_agent as M

    gid = Path(game_dir).parent.name
    try:
        game, meta = load_game(Path(game_dir))
    except Exception as e:
        return {"game": gid, "solved": 0, "total": 0, "actions": 0, "time_s": 0.0,
                "stop": f"ERR {type(e).__name__}"}

    total = int(game.perform_action(ActionInput(id=GameAction.RESET), raw=True).win_levels)
    agent = M.MyAgent()                       # stub SDK base; no torch -> graph-only
    agent.frames = []
    raw = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)

    t0 = time.time()
    acts = 0
    while raw.state != GameState.WIN and acts < max_actions and (time.time() - t0) < time_budget:
        try:
            action = agent.choose_action([], raw)
        except Exception as e:
            return {"game": gid, "solved": int(game._score), "total": total, "actions": acts,
                    "time_s": time.time() - t0, "stop": f"agent_err:{type(e).__name__}:{e}"[:60]}
        ad = getattr(action, "action_data", None)         # set_data stores a ComplexAction
        if ad is None:
            data = {}
        elif isinstance(ad, dict):
            data = ad
        elif hasattr(ad, "model_dump"):
            data = ad.model_dump()
        else:
            data = {"x": getattr(ad, "x", 0), "y": getattr(ad, "y", 0)}
        raw = game.perform_action(ActionInput(id=action, data=data), raw=True)
        acts += 1

    stop = "WIN" if raw.state == GameState.WIN else ("cap_actions" if acts >= max_actions else "cap_time")
    return {"game": gid, "solved": int(game._score), "total": total, "actions": acts,
            "time_s": time.time() - t0, "stop": stop}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-actions", type=int, default=4000)
    ap.add_argument("--time-budget", type=float, default=90.0)
    ap.add_argument("--no-cnn", action="store_true",
                    help="force graph-only (skip ChangeNet) — fast benchmarking even if torch is installed")
    ap.add_argument("--seed", type=int, default=None, help="fixed per-game seed for reproducible A/B")
    ap.add_argument("--change-prior", action="store_true",
                    help="enable the tabular novelty prior (default OFF — it regressed coverage)")
    ap.add_argument("--simple-clicks", action="store_true",
                    help="use the original centroid-only click candidates (A/B against the richer set)")
    args = ap.parse_args()
    if args.no_cnn:
        os.environ["ARC_DISABLE_CNN"] = "1"            # inherited by spawned workers
    if args.seed is not None:
        os.environ["ARC_SEED"] = str(args.seed)
    if args.change_prior:
        os.environ["ARC_ENABLE_CHANGE_PRIOR"] = "1"
    if args.simple_clicks:
        os.environ["ARC_SIMPLE_CLICKS"] = "1"

    try:
        import torch  # noqa: F401
        mode = "GRAPH-ONLY (--no-cnn)" if args.no_cnn else f"CNN on {'cuda' if torch.cuda.is_available() else 'CPU (slow!)'}"
    except Exception:
        mode = "GRAPH-ONLY (no torch)"

    dirs = sorted(glob.glob(ENV_GLOB))
    if args.games:
        dirs = [d for d in dirs if Path(d).parent.name in set(args.games)]
    tasks = [(d, args.max_actions, args.time_budget) for d in dirs]

    print(f"# {len(dirs)} games | {mode} | cap {args.max_actions} actions / "
          f"{args.time_budget:.0f}s | workers {args.workers}")
    print(f"{'game':5} {'solved':>7} {'acts':>6} {'time':>6}  stop")
    print("-" * 48)
    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(run_one, tasks):
                results.append(r)
                print(f"{r['game']:5} {r['solved']:>3}/{r['total']:<3} {r['actions']:>6} "
                      f"{r['time_s']:>5.0f}s  {r['stop']}")
    else:
        for t in tasks:
            r = run_one(t)
            results.append(r)
            print(f"{r['game']:5} {r['solved']:>3}/{r['total']:<3} {r['actions']:>6} "
                  f"{r['time_s']:>5.0f}s  {r['stop']}")

    solved = sum(r["solved"] for r in results)
    levels = sum(r["total"] for r in results)
    games_with_progress = sum(1 for r in results if r["solved"] > 0)
    print("-" * 48)
    print(f"TOTAL: {solved}/{levels} levels | {games_with_progress}/{len(results)} games with ≥1 level "
          f"(graph-only lower bound; CNN + GPU would lift this)")


if __name__ == "__main__":
    main()
