from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class Action:
    """An action in the public ARC-AGI-3 action space.

    ACTION6 is the only parameterized action.  Coordinates are absent for every
    other action.  The policy intentionally carries no environment identifier.
    """

    id: int
    x: int | None = None
    y: int | None = None

    @property
    def data(self) -> dict[str, int]:
        if self.id == 6 and self.x is not None and self.y is not None:
            return {"x": int(self.x), "y": int(self.y)}
        return {}

    @property
    def key(self) -> tuple[int, int | None, int | None]:
        return (self.id, self.x, self.y)


@dataclass(frozen=True, slots=True)
class Observation:
    """The complete information boundary visible to the policy."""

    frame: np.ndarray
    available_actions: tuple[int, ...]
    levels_completed: int
    win_levels: int
    state: str

    @property
    def terminal(self) -> bool:
        return self.state in {"WIN", "GAME_OVER"}

    @property
    def won(self) -> bool:
        return self.state == "WIN"


class BlackBoxSimulator(Protocol):
    """Optional generative interface; it exposes no game implementation details."""

    def observe(self) -> Observation: ...

    def fork(self) -> "BlackBoxSimulator": ...

    def step(self, action: Action) -> Observation: ...


def last_frame(frames: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    arr = np.asarray(frames)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D grid or frame sequence, got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.int8)


def state_key(obs: Observation, context: tuple[tuple[int, int | None, int | None], ...] = ()) -> bytes:
    """Observable belief-state key.

    A short action context distinguishes hidden-state progress when two states
    render identically.  This is still observation-only: no engine attributes are
    inspected.
    """

    header = bytes((obs.levels_completed & 255, len(context) & 255))
    trail = repr(context[-6:]).encode("ascii")
    return header + obs.frame.tobytes() + trail
