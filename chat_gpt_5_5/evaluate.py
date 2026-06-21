from __future__ import annotations

import argparse
import time

from .agent import AgentConfig, GeneralistAgent
from .engine import EvaluationEnvironment, RunReport, game_rhae, iter_environment_directories


def evaluate_one(directory, anonymous_index: int, mode: str, config: AgentConfig,
                 action_budget: int, time_budget: float) -> tuple[RunReport, float, str]:
    env = EvaluationEnvironment.load(directory, anonymous_index)
    agent = GeneralistAgent(config)
    obs = env.observe()
    level_actions = 0
    per_level: list[int] = []
    actions = 0
    error = ""
    started = time.perf_counter()

    while not obs.won and actions < action_budget and time.perf_counter() - started < time_budget:
        before_level = obs.levels_completed
        try:
            simulator = env.simulator() if mode == "simulator" else None
            action = agent.act(obs, simulator)
            obs = env.step(action)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
        actions += 1
        level_actions += 1
        if obs.levels_completed > before_level:
            per_level.append(level_actions)
            level_actions = 0

    report = RunReport(
        label=env.label,
        solved=obs.levels_completed,
        total=obs.win_levels,
        state=obs.state,
        actions=actions,
        per_level_actions=per_level,
        search_nodes=agent.search_nodes,
        error=error,
    )
    baselines = list(env.metadata.get("baseline_actions") or [])
    score = game_rhae(per_level, baselines, obs.win_levels)
    public_label = str(env.metadata.get("game_id", directory.parent.name)).split("-", 1)[0]
    return report, score, public_label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the anonymous, game-agnostic ARC-AGI-3 agent"
    )
    parser.add_argument("games", nargs="*", help="harness-only filter; never passed to the policy")
    parser.add_argument("--mode", choices=("simulator", "online"), default="simulator",
                        help="simulator = black-box fork/step; online = official observations only")
    parser.add_argument("--bfs", type=float, default=0.20, help="shortest-path search seconds per plan")
    parser.add_argument("--search", type=float, default=1.80, help="guided search seconds per plan")
    parser.add_argument("--cap", type=float, default=45.0, help="wall-clock seconds per environment")
    parser.add_argument("--actions", type=int, default=1200, help="real action budget per environment")
    parser.add_argument("--nodes", type=int, default=120_000, help="simulated nodes per plan")
    parser.add_argument("--depth", type=int, default=180)
    parser.add_argument("--clicks", type=int, default=48)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = AgentConfig(
        bfs_seconds=args.bfs,
        search_seconds=args.search,
        node_budget=args.nodes,
        max_depth=args.depth,
        max_clicks=args.clicks,
    )
    directories = iter_environment_directories(args.games)
    print(f"# mode={args.mode}; anonymous policy; no IDs/source/traces; {len(directories)} environments")
    print(f"{'public':6} {'anonymous':9} {'levels':>7} {'state':>12} {'real':>6} {'sim-nodes':>10} {'RHAE':>7}  detail")
    print("-" * 88)
    rows = []
    for index, directory in enumerate(directories, start=1):
        started = time.perf_counter()
        try:
            report, score, public_label = evaluate_one(
                directory, index, args.mode, config, args.actions, args.cap
            )
            detail = report.error or ("WIN" if report.won else "budget/unsolved")
            print(f"{public_label:6} {report.label:9} {report.solved:>3}/{report.total:<3} {report.state:>12} "
                  f"{report.actions:>6} {report.search_nodes:>10} {score:>6.2%}  {detail} "
                  f"({time.perf_counter() - started:.1f}s)")
            rows.append((report, score))
        except Exception as exc:
            print(f"{directory.parent.name:6} env-{index:03d} ERROR {type(exc).__name__}: {exc}")

    solved = sum(report.solved for report, _ in rows)
    total = sum(report.total for report, _ in rows)
    won = sum(report.won for report, _ in rows)
    mean_rhae = sum(score for _, score in rows) / len(rows) if rows else 0.0
    print("-" * 88)
    print(f"TOTAL {solved}/{total} levels; {won}/{len(rows)} games won; mean RHAE {mean_rhae:.2%}")
    # Partial completion is a benchmark result, not a process failure.
    return 0 if len(rows) == len(directories) and not any(r.error for r, _ in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
