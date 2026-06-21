from __future__ import annotations

import copy
import unittest

import numpy as np

from .agent import AgentConfig, GeneralistAgent
from .audit import audit_policy
from .core import Action, Observation
from .perception import candidate_actions, connected_components


class LineWorld:
    """Anonymous toy mechanic used to verify real rule discovery/search."""

    def __init__(self, x: int = 0, level: int = 0, state: str = "NOT_FINISHED") -> None:
        self.x = x
        self.level = level
        self.state = state

    def observe(self) -> Observation:
        frame = np.zeros((8, 8), dtype=np.int8)
        frame[4, self.x] = 2
        frame[4, 7] = 3
        return Observation(frame, (3, 4), self.level, 1, self.state)

    def fork(self) -> "LineWorld":
        return copy.deepcopy(self)

    def step(self, action: Action) -> Observation:
        if action.id == 3:
            self.x = max(0, self.x - 1)
        elif action.id == 4:
            self.x = min(7, self.x + 1)
        if self.x == 7:
            self.level = 1
            self.state = "WIN"
        return self.observe()


class HiddenCounterWorld:
    """Three identical-looking actions are required; tests belief-state context."""

    def __init__(self, counter: int = 0) -> None:
        self.counter = counter

    def observe(self) -> Observation:
        frame = np.zeros((6, 6), dtype=np.int8)
        frame[2:4, 2:4] = 9
        won = self.counter >= 3
        return Observation(frame, (1,), int(won), 1, "WIN" if won else "NOT_FINISHED")

    def fork(self) -> "HiddenCounterWorld":
        return copy.deepcopy(self)

    def step(self, action: Action) -> Observation:
        if action.id == 1:
            self.counter += 1
        return self.observe()


class GeneralAgentTests(unittest.TestCase):
    def test_policy_leakage_audit(self) -> None:
        self.assertEqual(audit_policy(), [])

    def test_components_and_click_proposals(self) -> None:
        frame = np.zeros((10, 10), dtype=np.int8)
        frame[2:4, 2:4] = 7
        frame[7, 8] = 5
        obs = Observation(frame, (6,), 0, 1, "NOT_FINISHED")
        comps = connected_components(frame)
        self.assertEqual({c.color for c in comps}, {5, 7})
        clicks = candidate_actions(obs)
        self.assertTrue(clicks)
        self.assertTrue(all(a.id == 6 and a.x is not None and a.y is not None for a in clicks))

    def test_black_box_planner_solves_unseen_line_world(self) -> None:
        world = LineWorld()
        cfg = AgentConfig(bfs_seconds=0.5, search_seconds=0.5, max_depth=20)
        agent = GeneralistAgent(cfg)
        obs = world.observe()
        for _ in range(12):
            if obs.won:
                break
            action = agent.act(obs, world)
            obs = world.step(action)
        self.assertTrue(obs.won)
        self.assertGreater(agent.search_nodes, 0)

    def test_observation_only_policy_learns_action_effect(self) -> None:
        world = LineWorld(x=5)
        agent = GeneralistAgent(AgentConfig())
        obs = world.observe()
        for _ in range(25):
            if obs.won:
                break
            obs = world.step(agent.act(obs, simulator=None))
        self.assertTrue(obs.won)

    def test_hidden_progress_is_not_pruned_as_noop(self) -> None:
        world = HiddenCounterWorld()
        agent = GeneralistAgent(AgentConfig(bfs_seconds=0.5, search_seconds=0.1, max_depth=8))
        obs = world.observe()
        action = agent.act(obs, world)
        self.assertEqual(action.id, 1)
        self.assertTrue(agent.last_plan.solved)
        self.assertEqual(len(agent.last_plan.actions), 3)


if __name__ == "__main__":
    unittest.main()
