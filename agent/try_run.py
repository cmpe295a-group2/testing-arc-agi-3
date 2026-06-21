"""
Standalone dev runner for forward_model_agent.py (Windows-friendly).

Loads a real public game straight from the pure-python `arcengine` wheel (no
`arc_agi` / flask / requests needed) via a minimal env shim, then exercises the
agent's Day-1 probe + forward-model search on it.

Usage:
    python agent/try_run.py environment_files/vc33/<ver>  [per_plan_budget_s] [max_actions]
"""
import inspect
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "arc-prize-2026-arc-agi-3" / "arc_agi_3_wheels"
                       / "arcengine-0.9.3-py3-none-any.whl"))
sys.path.insert(0, str(ROOT / "agent"))

from arcengine import ActionInput, GameAction  # noqa: E402  (from the wheel on sys.path)
import forward_model_agent as fma              # noqa: E402


def load_game(game_dir: Path, seed: int = 0):
    """Mirror LocalEnvironmentWrapper._load_game_class without the wrapper."""
    meta = json.loads((game_dir / "metadata.json").read_text(encoding="utf-8"))
    gid = str(meta["game_id"]).split("-")[0]
    cls_name = gid[:4][0].upper() + gid[:4][1:]          # "cd82" -> "Cd82"
    src = (game_dir / f"{gid}.py").read_text(encoding="utf-8")
    ns: dict = {"__name__": f"game_{gid}"}
    exec(compile(src, f"{gid}.py", "exec"), ns)
    cls = ns[cls_name]
    game = cls(seed=seed) if "seed" in inspect.signature(cls).parameters else cls()
    return game, meta


class MiniEnv:
    """Minimal stand-in for LocalEnvironmentWrapper (no scorecard / recording)."""
    def __init__(self, game):
        self._game = game
        self._last = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)

    @property
    def observation_space(self):
        return self._last

    def step(self, action, data=None, reasoning=None):
        self._last = self._game.perform_action(
            ActionInput(id=action, data=data or {}), raw=True)
        return self._last


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    from arcengine import GameState

    game_dir = Path(sys.argv[1])
    bfs_b = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
    nov_b = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0

    game, meta = load_game(game_dir)
    env = MiniEnv(game)
    obs = env.observation_space
    baselines = meta.get("baseline_actions") or []
    print(f"\n=== {meta['game_id']}  tags={meta.get('tags')}  win_levels={obs.win_levels}  "
          f"baselines={baselines}")
    print("PROBE:", json.dumps(fma.simulability_probe(env)))
    print(f"--- solving level-by-level (BFS {bfs_b}s -> novelty {nov_b}s cascade) ---")

    cfg = fma.SearchConfig(bfs_budget_s=bfs_b, novelty_budget_s=nov_b)
    policy = fma.ForwardModelPolicy(cfg)

    rhae: list = []
    t_start = time.time()
    while env.observation_space.state != GameState.WIN:
        lvl = int(env._game._score)                      # 0-indexed level being solved
        r = policy._plan_level(env._game)
        if not r.solved:
            print(f"L{lvl + 1}: STUCK after {r.nodes} nodes / {r.elapsed_s:.1f}s ({r.reason})")
            break
        for ai in r.plan:                                # execute the solving plan for real
            env.step(ai.id, data=dict(ai.data or {}))
        base = baselines[lvl] if lvl < len(baselines) else None
        s = fma.rhae_level_score(base, len(r.plan)) if base else 0.0
        rhae.append(s)
        print(f"L{lvl + 1}: SOLVED in {len(r.plan):>3} actions (human {base}) "
              f"-> RHAE {s:.2f}  | {r.nodes} nodes {r.elapsed_s:.1f}s")

    done = int(env._game._score)
    st = env.observation_space.state.name
    print(f"\n=== {done}/{obs.win_levels} levels solved in {time.time() - t_start:.1f}s | state={st}")
    if rhae:
        print(f"=== mean per-level RHAE (solved levels): {sum(rhae) / len(rhae):.2f}")


if __name__ == "__main__":
    main()
