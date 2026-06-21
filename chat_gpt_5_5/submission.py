"""Thin competition-runtime adapter for :class:`GeneralistAgent`.

The adapter is intentionally outside the audited policy boundary.  It converts
SDK frame objects to the narrow Observation type and, when explicitly enabled and
available, wraps a deepcopy of the live game as a black-box simulator.  It never
passes that object to the policy.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from arcengine import ActionInput, GameAction
except ModuleNotFoundError:  # local repository; competition image installs it normally
    root = Path(__file__).resolve().parent.parent
    wheel = root / "arc-prize-2026-arc-agi-3" / "arc_agi_3_wheels" / "arcengine-0.9.3-py3-none-any.whl"
    sys.path.insert(0, str(wheel))
    from arcengine import ActionInput, GameAction

from .agent import AgentConfig, GeneralistAgent
from .core import Action, Observation, last_frame


def _observation(frame_data: Any) -> Observation:
    available = tuple(
        int(value.value if hasattr(value, "value") else value)
        for value in (frame_data.available_actions or [])
    )
    state = frame_data.state.name if hasattr(frame_data.state, "name") else str(frame_data.state)
    return Observation(
        frame=last_frame(np.asarray(frame_data.frame)),
        available_actions=available,
        levels_completed=int(frame_data.levels_completed),
        win_levels=int(frame_data.win_levels),
        state=state,
    )


class _ForkView:
    __slots__ = ("__game", "__observation")

    def __init__(self, game: Any, observation: Observation) -> None:
        self.__game = game
        self.__observation = observation

    def observe(self) -> Observation:
        return self.__observation

    def fork(self) -> "_ForkView":
        return _ForkView(copy.deepcopy(self.__game), self.__observation)

    def step(self, action: Action) -> Observation:
        raw = self.__game.perform_action(
            ActionInput(id=GameAction.from_id(action.id), data=action.data), raw=True
        )
        self.__observation = _observation(raw)
        return self.__observation


class CompetitionPolicy:
    """Drop-in policy object for an SDK Agent subclass.

    Call ``choose_action(latest_frame, arc_env)`` from the SDK's method of the
    same name.  Do not pass ``game_id`` into this object.
    """

    def __init__(self, config: AgentConfig | None = None, allow_black_box_clone: bool = True) -> None:
        self.policy = GeneralistAgent(config)
        self.allow_black_box_clone = allow_black_box_clone

    def choose_action(self, latest_frame: Any, arc_env: Any = None) -> GameAction:
        obs = _observation(latest_frame)
        simulator = None
        if self.allow_black_box_clone and arc_env is not None:
            live_game = getattr(arc_env, "_game", None)
            if live_game is not None:
                simulator = _ForkView(copy.deepcopy(live_game), obs)
        selected = self.policy.act(obs, simulator)
        action = GameAction.from_id(selected.id)
        if selected.data:
            action.set_data(selected.data)
        action.reasoning = {
            "policy": "anonymous-object-graph",
            "simulator": simulator is not None,
            "search_nodes": self.policy.last_plan.nodes,
            "goal_hypothesis": self.policy.memory.goal.kind,
        }
        return action
