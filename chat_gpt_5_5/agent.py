from __future__ import annotations

import collections
import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .core import Action, BlackBoxSimulator, Observation, state_key
from .perception import (
    abstraction_key,
    candidate_actions,
    connected_components,
    features,
    transition_value,
)


@dataclass(slots=True)
class AgentConfig:
    bfs_seconds: float = 0.20
    search_seconds: float = 1.80
    node_budget: int = 120_000
    max_depth: int = 180
    max_clicks: int = 48
    history_context: int = 6
    online_exploration: float = 1.4


@dataclass(slots=True)
class PlanResult:
    actions: list[Action] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    solved: bool = False
    nodes: int = 0
    elapsed: float = 0.0
    reason: str = ""


@dataclass(slots=True)
class GoalHypothesis:
    """A mechanic hypothesis learned only after an observed level completion."""

    kind: str = "none"
    color: int = -1
    agent_color: int = -1
    confidence: float = 0.0

    @property
    def active(self) -> bool:
        return self.kind in {"clear", "reach"} and self.color >= 0

    def distance(self, frame: np.ndarray) -> float:
        grid = np.asarray(frame)
        if self.kind == "clear":
            return float(np.count_nonzero(grid == self.color))
        if self.kind == "reach" and self.agent_color >= 0:
            target = np.argwhere(grid == self.color)
            agent = np.argwhere(grid == self.agent_color)
            if target.size == 0:
                return 0.0
            if agent.size == 0:
                return 128.0
            ay, ax = agent.mean(axis=0)
            return float(np.min(np.abs(target[:, 0] - ay) + np.abs(target[:, 1] - ax)))
        return 0.0


class MechanicMemory:
    """Cross-level knowledge learned within one anonymous environment."""

    def __init__(self) -> None:
        self.goal = GoalHypothesis()
        self.effect_sum: collections.Counter[tuple] = collections.Counter()
        self.effect_n: collections.Counter[tuple] = collections.Counter()
        self.successful_action_classes: collections.Counter[tuple] = collections.Counter()

    @staticmethod
    def action_class(action: Action, frame: np.ndarray) -> tuple:
        if action.id != 6 or action.x is None or action.y is None:
            return (action.id,)
        y = min(max(int(action.y), 0), frame.shape[0] - 1)
        x = min(max(int(action.x), 0), frame.shape[1] - 1)
        return (6, int(frame[y, x]))

    def observe_effect(self, action: Action, before: Observation, after: Observation) -> None:
        cls = self.action_class(action, before.frame)
        value = transition_value(before.frame, after.frame)
        if after.state == "GAME_OVER":
            value -= 4.0
        if after.levels_completed > before.levels_completed:
            value += 10.0
            self.successful_action_classes[cls] += 1
        self.effect_sum[cls] += value
        self.effect_n[cls] += 1

    def action_prior(self, action: Action, frame: np.ndarray) -> float:
        cls = self.action_class(action, frame)
        n = self.effect_n[cls]
        empirical = self.effect_sum[cls] / n if n else 0.0
        return empirical + 1.5 * self.successful_action_classes[cls]

    def learn_goal(self, frames: list[np.ndarray]) -> None:
        """Infer a transferable goal from a completed easy level.

        `frames[-1]` is the frame immediately before the winning action.  The
        post-action frame belongs to the next level and is deliberately excluded.
        """

        if len(frames) < 2:
            return
        start = np.asarray(frames[0])
        prewin = np.asarray(frames[-1])
        fs, fp = features(start), features(prewin)
        ignored = {fs.background}

        # Clear/collect hypothesis: a colour loses most of its mass before win.
        candidates: list[tuple[float, int]] = []
        for color in range(16):
            if color in ignored or fs.color_counts[color] == 0:
                continue
            drop = fs.color_counts[color] - fp.color_counts[color]
            ratio = drop / fs.color_counts[color]
            if drop > 0 and ratio >= 0.40:
                candidates.append((ratio + min(drop, 32) / 64.0, color))
        if candidates:
            confidence, color = max(candidates)
            self.goal = GoalHypothesis("clear", color=color, confidence=min(1.0, confidence))
            return

        # Reach hypothesis: identify the colour whose mask moved most, then the
        # nearest rare static landmark in the pre-win state.
        move_score: dict[int, int] = collections.defaultdict(int)
        for a, b in zip(frames, frames[1:]):
            for color in range(16):
                move_score[color] += int(np.count_nonzero((a == color) ^ (b == color)))
        move_score.pop(fs.background, None)
        if not move_score:
            return
        agent_color = max(move_score, key=move_score.get)
        agent_cells = np.argwhere(prewin == agent_color)
        if agent_cells.size == 0:
            return
        ay, ax = agent_cells.mean(axis=0)
        landmarks: list[tuple[float, int]] = []
        for color in range(16):
            count = fs.color_counts[color]
            if color in (fs.background, agent_color) or not 1 <= count <= 16:
                continue
            # Static over the successful trajectory is more likely a target.
            if any(np.count_nonzero((a == color) ^ (b == color)) for a, b in zip(frames, frames[1:])):
                continue
            cells = np.argwhere(prewin == color)
            if cells.size:
                dist = float(np.min(np.abs(cells[:, 0] - ay) + np.abs(cells[:, 1] - ax)))
                landmarks.append((dist + count / 32.0, color))
        if landmarks:
            _, color = min(landmarks)
            self.goal = GoalHypothesis("reach", color=color, agent_color=agent_color, confidence=0.55)


@dataclass(slots=True)
class _SearchNode:
    simulator: BlackBoxSimulator
    observation: Observation
    actions: list[Action]
    observations: list[Observation]
    context: tuple[tuple[int, int | None, int | None], ...]


class BlackBoxPlanner:
    """Searches observations and black-box transitions, never engine internals."""

    def __init__(self, config: AgentConfig, memory: MechanicMemory) -> None:
        self.config = config
        self.memory = memory

    def plan(self, root: BlackBoxSimulator) -> PlanResult:
        started = time.perf_counter()
        root_obs = root.observe()
        bfs = self._bfs(root, root_obs, started, self.config.bfs_seconds)
        if bfs.solved:
            return bfs
        remaining_start = time.perf_counter()
        guided = self._guided(root, root_obs, remaining_start, self.config.search_seconds)
        guided.nodes += bfs.nodes
        guided.elapsed = time.perf_counter() - started
        if guided.solved or len(guided.actions) >= len(bfs.actions):
            return guided
        bfs.elapsed = time.perf_counter() - started
        return bfs

    def _children(self, node: _SearchNode) -> list[tuple[Action, BlackBoxSimulator, Observation, tuple]]:
        actions = candidate_actions(node.observation, self.config.max_clicks)
        actions.sort(key=lambda a: self.memory.action_prior(a, node.observation.frame), reverse=True)
        out = []
        for action in actions:
            child = node.simulator.fork()
            obs = child.step(action)
            context = (node.context + (action.key,))[-self.config.history_context:]
            # Only retain action history when pixels did not expose the state
            # change; otherwise exact frame identity is sufficient and cycles merge.
            if not np.array_equal(obs.frame, node.observation.frame):
                context = ()
            out.append((action, child, obs, context))
        return out

    @staticmethod
    def _success(root_level: int, obs: Observation) -> bool:
        return obs.won or obs.levels_completed > root_level

    def _bfs(self, root: BlackBoxSimulator, root_obs: Observation, started: float, seconds: float) -> PlanResult:
        queue = collections.deque([_SearchNode(root.fork(), root_obs, [], [root_obs], ())])
        seen = {state_key(root_obs)}
        nodes = 0
        best = queue[0]
        while queue and nodes < self.config.node_budget and time.perf_counter() - started < seconds:
            node = queue.popleft()
            if len(node.actions) >= self.config.max_depth:
                continue
            for action, child, obs, context in self._children(node):
                nodes += 1
                actions = node.actions + [action]
                observations = node.observations + [obs]
                if self._success(root_obs.levels_completed, obs):
                    return PlanResult(actions, observations, True, nodes,
                                      time.perf_counter() - started, "bfs_win")
                if obs.state == "GAME_OVER":
                    continue
                key = state_key(obs, context)
                if key in seen:
                    continue
                seen.add(key)
                nxt = _SearchNode(child, obs, actions, observations, context)
                queue.append(nxt)
                if len(actions) > len(best.actions):
                    best = nxt
        return PlanResult(best.actions, best.observations, False, nodes,
                          time.perf_counter() - started, "bfs_budget")

    def _guided(self, root: BlackBoxSimulator, root_obs: Observation,
                started: float, seconds: float) -> PlanResult:
        root_node = _SearchNode(root.fork(), root_obs, [], [root_obs], ())
        counter = 0
        frontier: list[tuple[float, int, _SearchNode]] = [(0.0, counter, root_node)]
        seen = {state_key(root_obs)}
        buckets: collections.Counter[bytes] = collections.Counter()
        buckets[abstraction_key(root_obs.frame)] += 1
        nodes = 0
        best = root_node
        best_score = -math.inf
        goal0 = max(1.0, self.memory.goal.distance(root_obs.frame))
        while frontier and nodes < self.config.node_budget and time.perf_counter() - started < seconds:
            _, _, node = heapq.heappop(frontier)
            if len(node.actions) >= self.config.max_depth:
                continue
            for action, child, obs, context in self._children(node):
                nodes += 1
                actions = node.actions + [action]
                observations = node.observations + [obs]
                if self._success(root_obs.levels_completed, obs):
                    return PlanResult(actions, observations, True, nodes,
                                      time.perf_counter() - started, "guided_win")
                if obs.state == "GAME_OVER":
                    continue
                key = state_key(obs, context)
                if key in seen:
                    continue
                seen.add(key)
                bucket = abstraction_key(obs.frame)
                visits = buckets[bucket]
                buckets[bucket] += 1
                info = 1.0 / (1.0 + visits)
                effect = transition_value(node.observation.frame, obs.frame)
                prior = self.memory.action_prior(action, node.observation.frame)
                goal_progress = 0.0
                if self.memory.goal.active:
                    goal_progress = (goal0 - self.memory.goal.distance(obs.frame)) / goal0
                # Higher utility is better; heap priority is its negative.  A tiny
                # depth cost avoids loops without killing long solutions.
                utility = 2.4 * goal_progress + 1.2 * info + 0.6 * effect + 0.15 * prior - 0.002 * len(actions)
                nxt = _SearchNode(child, obs, actions, observations, context)
                counter += 1
                heapq.heappush(frontier, (-utility, counter, nxt))
                if utility > best_score or (utility == best_score and len(actions) > len(best.actions)):
                    best_score, best = utility, nxt
        return PlanResult(best.actions, best.observations, False, nodes,
                          time.perf_counter() - started, "guided_budget")


@dataclass(slots=True)
class _OnlineEdge:
    action: Action
    destination: bytes
    reward: float
    terminal: bool


class GeneralistAgent:
    """Anonymous, game-agnostic online learner.

    The constructor has no game-id parameter.  All persistent knowledge is reset
    between environments and learned again from that environment's observations.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.memory = MechanicMemory()
        self.planner = BlackBoxPlanner(self.config, self.memory)
        self.cached_plan: collections.deque[Action] = collections.deque()
        self.previous: Observation | None = None
        self.previous_action: Action | None = None
        self.level_frames: list[np.ndarray] = []
        self.level_actions: list[Action] = []
        self.nodes: dict[bytes, Observation] = {}
        self.edges: dict[bytes, dict[tuple, _OnlineEdge]] = collections.defaultdict(dict)
        self.attempts: collections.Counter[tuple[bytes, tuple]] = collections.Counter()
        self.total_steps = 0
        self.search_nodes = 0
        self.last_plan = PlanResult()

    def _observable_key(self, obs: Observation) -> bytes:
        return state_key(obs)

    def observe(self, obs: Observation) -> None:
        key = self._observable_key(obs)
        self.nodes[key] = obs
        if not self.level_frames:
            self.level_frames.append(obs.frame.copy())
        if self.previous is not None and self.previous_action is not None:
            before_key = self._observable_key(self.previous)
            self.memory.observe_effect(self.previous_action, self.previous, obs)
            reward = transition_value(self.previous.frame, obs.frame)
            if obs.state == "GAME_OVER":
                reward -= 4.0
            if obs.levels_completed > self.previous.levels_completed:
                reward += 10.0
                # The current frame belongs to the next level.  Learn only from
                # frames through the pre-win state.
                self.memory.learn_goal(self.level_frames)
                self.level_frames = [obs.frame.copy()]
                self.level_actions = []
                self.cached_plan.clear()
                self.nodes.clear()
                self.edges.clear()
                self.attempts.clear()
            else:
                self.level_frames.append(obs.frame.copy())
                self.edges[before_key][self.previous_action.key] = _OnlineEdge(
                    self.previous_action, key, reward, obs.terminal
                )
        self.previous = obs
        self.previous_action = None

    def act(self, obs: Observation, simulator: BlackBoxSimulator | None = None) -> Action:
        self.observe(obs)
        if obs.state in {"NOT_PLAYED", "GAME_OVER"}:
            action = Action(0)
        elif self.cached_plan:
            action = self.cached_plan.popleft()
        elif simulator is not None:
            self.last_plan = self.planner.plan(simulator)
            self.search_nodes += self.last_plan.nodes
            if self.last_plan.solved:
                self.cached_plan.extend(self.last_plan.actions)
                action = self.cached_plan.popleft()
            elif self.last_plan.actions:
                # Model-predictive exploration: commit one information-rich step,
                # observe reality, then re-plan.
                action = self.last_plan.actions[0]
            else:
                action = self._online_action(obs)
        else:
            action = self._online_action(obs)
        self.previous_action = action
        self.level_actions.append(action)
        self.total_steps += 1
        return action

    def _online_action(self, obs: Observation) -> Action:
        key = self._observable_key(obs)
        actions = candidate_actions(obs, self.config.max_clicks)
        if not actions:
            return Action(0)
        untried = [a for a in actions if self.attempts[(key, a.key)] == 0]
        if untried:
            untried.sort(key=lambda a: self.memory.action_prior(a, obs.frame), reverse=True)
            chosen = untried[0]
        else:
            total = 1 + sum(self.attempts[(key, a.key)] for a in actions)
            scored = []
            for action in actions:
                n = self.attempts[(key, action.key)]
                edge = self.edges[key].get(action.key)
                mean = edge.reward if edge is not None else self.memory.action_prior(action, obs.frame)
                if edge is not None and edge.terminal:
                    mean -= 8.0
                ucb = mean + self.config.online_exploration * math.sqrt(math.log(total + 1) / max(1, n))
                scored.append((ucb, action))
            chosen = max(scored, key=lambda item: item[0])[1]
        self.attempts[(key, chosen.key)] += 1
        return chosen
