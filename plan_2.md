Below is the English list I’d use as a serious ARC-AGI-3 strategy map. I’m ranking by **private-set generalization + action efficiency + implementation practicality**. ARC-AGI-3 rewards agents that can explore, model, infer goals, and plan in novel interactive environments, and the official competition has no internet during evaluation, so the winning system needs self-contained reasoning/code, not online lookup. ([ARC Prize][1])

## Ranked Algorithm List for ARC-AGI-3

### 1. Hybrid Executable World Model + Planner

**Core idea:** Build an internal Python simulator of the game from observed transitions, verify it against past frames, then plan inside that simulator before acting.

**Tactical strategy:**

* Segment the grid into objects, colors, player candidates, walls, goals, buttons, hazards.
* Probe actions carefully to infer action semantics.
* Generate candidate transition rules.
* Compile those rules into an executable simulator.
* Run BFS/A*/MCTS inside the simulator.
* Execute only a few real actions, then re-check predictions.
* If prediction fails, patch the simulator.

**Why it is strong:** It directly targets the main ARC-AGI-3 challenge: learn the game rules fast and act efficiently. Recent executable-world-model work reports promising results on the 25 public ARC-AGI-3 games using verifier-driven Python world models. ([arXiv][2])

---

### 2. Workspace Optimization / Multi-Agent Scientist System

**Core idea:** Use multiple specialized agents that share a structured workspace: one explores, one builds the model, one critiques, one plans, one repairs failures.

**Tactical strategy:**

* Maintain a written/structured “lab notebook” of hypotheses.
* Store failed predictions as counterexamples.
* Assign roles:

  * Explorer: chooses probes.
  * Modeler: writes transition rules.
  * Critic: finds contradictions.
  * Planner: searches for solution paths.
  * Router: decides when to switch strategy.
* Update the workspace like training weights: evidence acts like data, prediction errors act like gradients.

**Why it is strong:** DreamTeam-style workspace optimization reportedly improved protocol-matched ARC-AGI-3 public performance while using fewer environment actions. ([arXiv][3])

---

### 3. Object-Centric Graph Exploration

**Core idea:** Convert every frame into an object graph and explore unknown state-action edges systematically.

**Tactical strategy:**

* Hash states by object positions and relationships, not raw pixels only.
* Track which actions have been tried from each abstract state.
* Prefer actions that lead to novel states.
* Avoid repeating transitions with no information gain.
* Build a directed graph: `state -> action -> next_state`.
* Detect loops, traps, doors, keys, switches, teleports, push mechanics.

**Why it is strong:** It is cheap, reliable, and does not depend on LLM reasoning. It should be a core baseline in the system.

---

### 4. Active Probe / Bayesian Experiment Design

**Core idea:** Choose actions that maximally reduce uncertainty about the game rules.

**Tactical strategy:**

* Maintain multiple hypotheses:

  * ACTION1 = up?
  * ACTION5 = interact?
  * Red = hazard?
  * Yellow = target?
  * ACTION6 clicks objects?
* Pick the next probe that distinguishes the most hypotheses.
* Example: if ACTION1 might be “up” or “rotate,” test it near a wall/object where the two outcomes differ.
* Stop probing once enough confidence exists to plan.

**Why it is strong:** Scoring heavily punishes wasted actions, so good probing is critical.

---

### 5. Program Synthesis / DSL Rule Induction

**Core idea:** Learn compact symbolic rules that explain observed transitions.

**Tactical strategy:**

* Define a small DSL for common game mechanics:

  * movement
  * collision
  * push/pull
  * collect
  * toggle
  * transform color
  * fill target
  * teleport
  * click object
* Search for the simplest program that explains observed state changes.
* Prefer shorter rules using MDL-style simplicity bias.
* Use the learned program as the world model.

**Example learned rule:**

```python
if action == ACTION1 and cell_above(player) != WALL:
    player.y -= 1

if player.pos == key.pos:
    key.collected = True
    door.open = True
```

**Why it is strong:** ARC tasks usually reward abstraction. A compact rule is more likely to generalize to later levels.

---

### 6. Hierarchical Skill Library / Options

**Core idea:** Learn reusable macro-actions and apply them across levels.

**Tactical strategy:**

* Build skills like:

  * `move_to(object)`
  * `click_object(center)`
  * `push_block_to(target)`
  * `collect_all(color)`
  * `avoid_hazard(color)`
  * `toggle_switch_then_enter_door`
* After level 1, reuse skills in level 2+.
* Compress low-level actions into high-level plans.

**Why it is strong:** ARC-AGI-3 weights later levels more heavily, so fast reuse after early discovery can greatly improve score.

---

### 7. Portfolio Meta-Controller

**Core idea:** Do not rely on one solver. Route each game to the most suitable algorithm.

**Tactical strategy:**

* Detect game type from early frames:

  * avatar navigation
  * click-based puzzle
  * cellular automaton
  * Sokoban-like pushing
  * collect-and-deliver
  * pattern transformation
  * hidden-rule puzzle
* Run cheap solvers first.
* Escalate to expensive LLM/world-model methods only when needed.
* Keep a time/action budget per solver.

**Why it is strong:** Private games will be diverse. A portfolio protects against single-method failure.

---

### 8. Model Predictive Control with BFS/A*/MCTS

**Core idea:** Plan inside the learned model, but only execute a short prefix before re-validating.

**Tactical strategy:**

* Search candidate plans in the simulator.
* Execute 1–3 real actions.
* Compare predicted frame vs actual frame.
* If matched, continue.
* If mismatched, repair model and replan.
* Use BFS/A* for deterministic navigation.
* Use MCTS for uncertain mechanics.

**Why it is strong:** It avoids committing to a long hallucinated plan.

---

### 9. State Abstraction and Canonicalization

**Core idea:** Compress raw frames into canonical symbolic states.

**Tactical strategy:**

* Ignore irrelevant background pixels.
* Normalize object order.
* Track relative positions instead of absolute positions when possible.
* Merge visually different but behaviorally equivalent states.
* Use canonical hashes to avoid redundant exploration.

**Why it is strong:** Raw grid states explode combinatorially. Abstraction makes search feasible.

---

### 10. Object Affordance Discovery

**Core idea:** Learn what each object can do or be used for.

**Tactical strategy:**

* For every object/color/component, estimate:

  * passable or solid
  * collectible or static
  * dangerous or safe
  * clickable or not
  * movable or fixed
  * target or tool
* Probe object interactions one by one.
* Build an affordance table.

**Example:**

```text
Blue square: likely player
Gray cells: walls
Yellow cells: goal/target
Red cells: hazard
Green cells: switch or collectible
```

**Why it is strong:** Human players solve many games by quickly identifying affordances.

---

### 11. Goal Inference / Inverse Planning

**Core idea:** Infer what winning probably means before seeing a win.

**Tactical strategy:**

* Look for goal-like visual cues:

  * target tiles
  * exit doors
  * symmetric patterns
  * empty slots
  * objects matching colors/shapes
  * counters or progress bars
* Generate candidate goals:

  * reach tile
  * collect all items
  * match pattern
  * fill targets
  * remove hazards
  * align objects
* Test plans against each candidate goal.
* Prefer goals that explain level design.

**Why it is strong:** Waiting until accidental WIN is too inefficient.

---

### 12. Novelty Search / Go-Explore Style Exploration

**Core idea:** Prioritize reaching new states instead of maximizing immediate reward.

**Tactical strategy:**

* Maintain archive of novel states.
* Prefer actions that create new object arrangements.
* Detect promising states: near goals, opened doors, changed colors, collected items.
* Use RESET only when exploration is clearly stuck.
* Replay known path to return to promising states if possible.

**Why it is useful:** Some games require discovering mechanics before the goal becomes obvious.

---

### 13. Causal Intervention Testing

**Core idea:** Identify cause-effect relationships by controlled experiments.

**Tactical strategy:**

* Change one variable at a time.
* Example:

  * stand near button, press ACTION5
  * stand away from button, press ACTION5
  * click object center
  * click empty space
* Compare deltas.
* Separate local effects from global effects.
* Store causal rules.

**Why it is strong:** ARC-AGI-3 punishes agents that observe local action effects but fail to form a global model, a known failure mode for frontier systems. ([arXiv][4])

---

### 14. Visual Delta Compiler

**Core idea:** Turn frame differences into structured transition facts.

**Tactical strategy:**

* After each action, compute:

  * which cells changed
  * which objects moved
  * which objects appeared/disappeared
  * which colors transformed
  * whether camera/grid shifted
* Convert deltas into semantic events:

  * “player moved left”
  * “red object vanished”
  * “door opened”
  * “counter decreased”
  * “wall became passable”

**Why it is strong:** It gives all higher-level algorithms clean input.

---

### 15. Constraint Solver / SAT-SMT Planning

**Core idea:** For puzzle-like games, formulate the solution as constraints.

**Tactical strategy:**

* Variables: object positions, target assignments, action sequence.
* Constraints:

  * cannot cross walls
  * blocks must end on targets
  * colors must match
  * all collectibles must be collected
* Use bounded search over action length.
* Increase horizon gradually.

**Why it is useful:** Works well for alignment, matching, filling, and Sokoban-like puzzles.

---

### 16. Cellular Automaton / Rule-Based Grid Induction

**Core idea:** Detect games where the whole grid evolves by local rules.

**Tactical strategy:**

* Test whether cells update based on neighbor patterns.
* Learn rules like:

  * color A spreads to adjacent B
  * object disappears after contact
  * line extends until wall
  * cells toggle every step
* Use local neighborhood templates.
* Simulate forward.

**Why it is useful:** Some ARC-style tasks are not avatar games; they are transformation systems.

---

### 17. Local LLM as Hypothesis Generator

**Core idea:** Use a local LLM as a scientist, not as the real-time controller.

**Tactical strategy:**

* Feed it compressed traces, not full raw grids every step.
* Ask it to propose:

  * possible goal
  * action meanings
  * object roles
  * next probe
  * simulator patch
* Validate every LLM idea with code and observations.
* Never trust the LLM directly for final action unless verified.

**Why it is useful:** LLMs are good at generating hypotheses, but weak at reliable low-level control.

---

### 18. Coding-Agent Rule Repair

**Core idea:** Let a coding model write and repair transition-model code.

**Tactical strategy:**

* Keep a file like `world_model.py`.
* After each failed prediction, provide:

  * previous frame
  * action
  * predicted frame
  * actual frame
  * diff
* Ask the coding model to patch the model.
* Run unit tests on all previous transitions.
* Reject patches that break older cases.

**Why it is strong:** This is basically software debugging applied to game-rule discovery.

---

### 19. Memory Across Levels

**Core idea:** Carry learned mechanics from earlier levels of the same game.

**Tactical strategy:**

* Store:

  * action meanings
  * object meanings
  * goal type
  * successful macro-skills
  * failure traps
* On next level, start with known hypotheses.
* Quickly verify if rules still hold.
* Avoid re-learning from scratch.

**Why it is strong:** Multiple levels usually share mechanics. Level transfer is essential for high score.

---

### 20. Cheap Heuristic Solvers

**Core idea:** Use simple universal tricks before expensive reasoning.

**Tactical strategy:**

* Try each simple action once.
* If a player is detected, map movement actions.
* If ACTION6 exists, click object centers.
* Click unusual colors first.
* Move toward goal-like cells.
* Avoid repeated no-op actions.
* Stop if GAME_OVER risk is detected.

**Why it is useful:** Some public/private games may be solved by simple behavior. Cheap wins matter.

---

### 21. Risk-Aware Exploration

**Core idea:** Avoid actions likely to cause GAME_OVER unless necessary.

**Tactical strategy:**

* Mark colors/objects associated with bad outcomes.
* Detect irreversible state changes.
* Prefer reversible probes.
* Use RESET only when stuck or after a controlled experiment.
* Maintain a safety score for each action.

**Why it is important:** Random exploration can destroy a run.

---

### 22. Action-Cost-Aware Planning

**Core idea:** Optimize not just for winning, but for winning with human-like action count.

**Tactical strategy:**

* Penalize long plans.
* Prefer direct paths.
* Avoid redundant probing after confidence is high.
* Use macro-actions internally, but count real actions carefully.
* Choose the shortest successful plan, not the most certain one if certainty difference is small.

**Why it is critical:** The score is based on relative human action efficiency, so long solutions are punished even if they eventually win. ([ARC Prize][1])

---

### 23. Self-Play Replay Analysis

**Core idea:** Use public games to build better generic debugging tools, not hard-coded game solutions.

**Tactical strategy:**

* Run many agents on public games.
* Analyze:

  * where they waste actions
  * where model prediction fails
  * which probes are most useful
  * which abstractions transfer
* Improve general algorithms, not game-specific code.

**Why it is useful:** Public games are development tools, but private games require generalization.

---

### 24. Learned Meta-Prior from Public Games

**Core idea:** Learn general priors about ARC-AGI-3 game design without memorizing games.

**Tactical strategy:**

* Learn which object features often matter:

  * unique colors
  * border objects
  * small movable components
  * symmetric targets
  * repeated patterns
* Train a lightweight classifier for:

  * player candidate
  * wall candidate
  * goal candidate
  * clickable object candidate
* Use this only as a prior, then verify by interaction.

**Why it is useful:** It speeds exploration while avoiding hard-coded public-game overfit.

---

### 25. Pure LLM ReAct Controller

**Core idea:** Let an LLM observe frames and choose the next action directly.

**Tactical strategy:**

* Prompt with current frame and action history.
* Ask for next action.
* Repeat until done.

**Why I rank it low:** Frontier AI systems have historically scored extremely low on ARC-AGI-3 relative to humans, so pure LLM action selection is unlikely to be enough. ([arXiv][4])

---

### 26. Model-Free Reinforcement Learning

**Core idea:** Learn a policy online from reward.

**Tactical strategy:**

* Try actions.
* Update policy from success/failure.
* Explore until reward appears.

**Why I rank it very low:** The environment is sparse-reward, low-sample, and novel per game. Online model-free RL is too sample-inefficient here.

---

## My Recommended Stack

```text
Tier 1: Universal cheap solvers
- action probing
- visual diff compiler
- movement detection
- object center clicking

Tier 2: Object-centric exploration
- state graph
- novelty search
- affordance table
- risk-aware probing

Tier 3: World model
- symbolic/DSL rules
- executable simulator
- verifier
- coding-agent repair

Tier 4: Planner
- BFS/A*
- MCTS when uncertain
- model predictive control
- macro-skill library

Tier 5: Meta-controller
- route games to specialized solvers
- manage action budget
- decide when to use local LLM
```

## My Top 5 to Actually Build First

1. **Visual Delta Compiler**
2. **Object-Centric Graph Explorer**
3. **Executable World Model + Verifier**
4. **Short-Horizon Planner / MPC**
5. **Local LLM Rule-Repair Assistant**

That combination gives the best chance: it explores like a scientist, learns rules like a programmer, and acts like a planner.

[1]: https://arcprize.org/competitions/2026/arc-agi-3?utm_source=chatgpt.com "ARC Prize 2026 - ARC-AGI-3 Competition"
[2]: https://arxiv.org/abs/2605.05138?utm_source=chatgpt.com "Executable World Models for ARC-AGI-3 in the Era of Coding Agents"
[3]: https://arxiv.org/abs/2605.09650?utm_source=chatgpt.com "Workspace Optimization: How to Train Your Agent"
[4]: https://arxiv.org/abs/2603.24621?utm_source=chatgpt.com "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence"
