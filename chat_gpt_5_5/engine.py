"""Local evaluation adapter.

This module is outside the policy boundary.  It loads a public environment only
to reproduce the competition runtime, then exposes the policy-facing
``SimulatorView`` containing exactly three operations: observe, fork, and step.
No game identifier, source, metadata, or Python game object crosses that boundary.
"""

from __future__ import annotations

import copy
import inspect
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = ROOT / "arc-prize-2026-arc-agi-3"
ENVIRONMENTS = BENCHMARK / "environment_files"
ENGINE_WHEEL = BENCHMARK / "arc_agi_3_wheels" / "arcengine-0.9.3-py3-none-any.whl"
if str(ENGINE_WHEEL) not in sys.path:
    sys.path.insert(0, str(ENGINE_WHEEL))

from arcengine import ActionInput, GameAction  # noqa: E402

from .core import Action, Observation, last_frame  # noqa: E402


@dataclass(slots=True)
class RunReport:
    label: str
    solved: int
    total: int
    state: str
    actions: int
    per_level_actions: list[int] = field(default_factory=list)
    search_nodes: int = 0
    error: str = ""

    @property
    def won(self) -> bool:
        return self.state == "WIN" and self.solved == self.total


def iter_environment_directories(selected: Iterable[str] = ()) -> list[Path]:
    wanted = set(selected)
    directories = sorted(p for game in ENVIRONMENTS.iterdir() if game.is_dir() for p in game.iterdir() if p.is_dir())
    if wanted:
        directories = [p for p in directories if p.parent.name in wanted]
    return directories


def _load_backend(directory: Path, seed: int = 0) -> tuple["_GameBackend", dict[str, Any]]:
    """Trusted harness operation: instantiate the environment under test."""

    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    short_id = str(metadata["game_id"]).split("-", 1)[0]
    class_name = short_id[0].upper() + short_id[1:]
    source_path = directory / f"{short_id}.py"
    namespace: dict[str, Any] = {"__name__": f"evaluation_environment_{directory.parent.name}"}
    exec(compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec"), namespace)
    cls = namespace[class_name]
    game = cls(seed=seed) if "seed" in inspect.signature(cls).parameters else cls()
    backend = _GameBackend(game)
    backend.step(Action(0))
    return backend, metadata


class _GameBackend:
    """Owns the actual environment.  Never pass this object to the policy."""

    __slots__ = ("__game", "__raw")

    def __init__(self, game: Any, raw: Any = None) -> None:
        self.__game = game
        self.__raw = raw

    def observe(self) -> Observation:
        raw = self.__raw
        if raw is None:
            raise RuntimeError("environment has not been reset")
        available = []
        for item in raw.available_actions or []:
            available.append(int(item.value if hasattr(item, "value") else item))
        state = raw.state.name if hasattr(raw.state, "name") else str(raw.state)
        return Observation(
            frame=last_frame(raw.frame),
            available_actions=tuple(available),
            levels_completed=int(raw.levels_completed),
            win_levels=int(raw.win_levels),
            state=state,
        )

    def step(self, action: Action) -> Observation:
        self.__raw = self.__game.perform_action(
            ActionInput(id=GameAction.from_id(int(action.id)), data=action.data), raw=True
        )
        return self.observe()

    def fork_view(self) -> "SimulatorView":
        return SimulatorView(_GameBackend(copy.deepcopy(self.__game), copy.deepcopy(self.__raw)))


class SimulatorView:
    """Capability-limited simulator handed to the policy.

    Deliberately has no game/source/id/metadata accessors.  ``fork`` clones the
    current black-box state and simulation steps return the same Observation type
    as real steps.
    """

    __slots__ = ("__backend",)

    def __init__(self, backend: _GameBackend) -> None:
        self.__backend = backend

    def observe(self) -> Observation:
        return self.__backend.observe()

    def fork(self) -> "SimulatorView":
        return self.__backend.fork_view()

    def step(self, action: Action) -> Observation:
        return self.__backend.step(action)


@dataclass(slots=True)
class EvaluationEnvironment:
    """Evaluator-owned pair: anonymous policy view plus private scoring metadata."""

    label: str
    backend: _GameBackend
    metadata: dict[str, Any]

    @classmethod
    def load(cls, directory: Path, anonymous_index: int, seed: int = 0) -> "EvaluationEnvironment":
        backend, metadata = _load_backend(directory, seed=seed)
        return cls(label=f"env-{anonymous_index:03d}", backend=backend, metadata=metadata)

    def observe(self) -> Observation:
        return self.backend.observe()

    def step(self, action: Action) -> Observation:
        return self.backend.step(action)

    def simulator(self) -> SimulatorView:
        return self.backend.fork_view()


def game_rhae(per_level_actions: list[int], baselines: list[int], total_levels: int) -> float:
    weights = sum(range(1, total_levels + 1)) or 1
    weighted = 0.0
    for level, actions in enumerate(per_level_actions, start=1):
        if level > len(baselines) or actions <= 0:
            continue
        weighted += level * min(1.15, (baselines[level - 1] / actions) ** 2)
    completion_cap = sum(range(1, len(per_level_actions) + 1)) / weights
    return min(completion_cap, weighted / weights)
