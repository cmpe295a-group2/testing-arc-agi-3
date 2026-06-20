ARC-AGI-3 tests exploration, modeling, goal-setting, and planning/execution; submissions have no internet during evaluation; and scoring is capped by completed levels and penalizes inefficient action counts. That means the best agents should spend compute to understand the environment, build a model, search for a solution, then execute efficiently.

Ranked Algorithm List for ARC-AGI-3
Rank	Algorithm	Core Strategy	Why It Matters	Main Risk
1	Object-Centric State Graph Planner	Convert each 64×64 grid into objects, components, positions, colors, and relationships. Build a graph of observed states and action transitions. Use graph search to find paths to progress or win states.	ARC-AGI-3 is not just perception; it requires remembering what actions do and planning across states. State graph agents already performed well in the preview competition.	Bad object detection can poison the graph.
2	Systematic Exploration + Novelty Search	Prioritize actions that produce unseen states, unseen object movements, new colors, new click effects, or changes in available actions. Avoid loops and no-op actions.	The goal is not explicitly given, so the agent must discover what matters through interaction.	Exploration can waste too many scored actions if not separated from planning.
3	Model-Based Test-Time Planning	Learn or infer a transition model: state + action -> next_state. Then search inside that model before committing to real actions.	Since scoring depends heavily on completed levels and action efficiency, planning before acting is critical.	Learned models can be wrong in edge cases.
4	Executable World Model Agent	Maintain a Python-like executable model of the game dynamics. Verify the model against observed transitions. Refactor it when contradicted. Plan through the model.	Recent ARC-AGI-3 research reports strong public-game results using verifier-driven executable world models.	Hard to make reliable offline without strong coding/reasoning support.
5	Explore → Verify → Plan Loop	Explore small action sets, form hypotheses, verify them with targeted probes, then switch to planning once the rule is understood.	This mirrors how humans solve these games: not random action spam, but controlled experimentation.	Needs good hypothesis management.
6	Map-Then-Act Agent	First build a structured “cognitive map” of the environment: passable cells, objects, affordances, buttons, goals, hazards, portals. Then solve using that map.	The MAP paper argues that agents fail when they act before understanding the environment. It reports gains on ARC-AGI-3-style tasks.	Mapping can be expensive or incomplete.
7	Action-Effect Learning	Learn which actions actually change the frame, which are no-ops, which move the agent, which interact, which click cells matter.	Preview winner StochasticGoose used a CNN action-learning approach to predict useful actions and beat random exploration.	It may learn “what changes” but not “what wins.”
8	Click Candidate Pruning	For coordinate actions, avoid trying all 4096 cells. Try object centers, changed cells, colored components, borders, corners, symmetry points, highlighted areas, and cells near the agent.	Click action space can explode. Smart pruning is essential.	If the true click target is visually subtle, pruning may miss it.
9	Hierarchical Macro Discovery	Compress repeated action sequences into reusable macros: move-to-object, push-object, collect-item, toggle-switch, match-pattern, open-door, return-home.	Later levels likely compose earlier mechanics. Macros help transfer from easy levels to harder ones.	Wrong macros can overfit early levels.
10	A / BFS / Dijkstra Pathfinding Module*	When the game resembles navigation, extract walls, movable agent, targets, doors, hazards, and solve with classical pathfinding.	Many grid games contain navigation subproblems. Classical search is fast and reliable.	Fails when rules are non-spatial or hidden.
11	Sokoban/Push-Puzzle Solver	Detect pushable objects, blockers, target cells, deadlocks, and solve with domain-specific search.	Some ARC-like environments may hide push mechanics. Specialized solvers can be extremely efficient.	Too narrow if the game is not push-based.
12	Pattern Transformation Solver	Detect whether the level requires arranging colors, copying shapes, completing symmetry, matching counts, or transforming patterns. Use constraint search over object transforms.	ARC heritage strongly favors abstract visual transformations.	Hard to infer goal from interaction alone.
13	Goal Progress Heuristic Learner	Learn signals correlated with progress: level completion, new animation, object disappearance, door opening, score-like state changes, reduced distance, matched color count.	Sparse rewards need intermediate progress estimates.	False progress signals can mislead search.
14	ResNet/CNN Value Model	Train a small neural model to rank (state, action) pairs based on previous successful transitions. Use it as a heuristic, not as the full policy.	Blind Squirrel preview agent used a state graph plus small value model to rank actions toward milestones.	Needs enough data per game; can overfit noisy exploration.
15	Monte Carlo Tree Search with Object-Level Actions	Use MCTS, but only after reducing the action space to meaningful object-level actions or candidate clicks.	Good for uncertain dynamics and delayed reward.	Pure pixel/action MCTS is too expensive.
16	Beam Search over Action Sequences	Keep the top-K most promising partial action sequences using novelty, progress, and model confidence.	Simpler than full MCTS and often effective for puzzle search.	Beam can prune the correct path early.
17	Program Synthesis / DSL Solver	Define a small DSL: find_agent, find_targets, move_to, click_object, match_shape, toggle_until, avoid_color. Search over programs.	Good for generalizing compositional rules across levels.	DSL design is hard; too narrow loses, too broad explodes.
18	Local LLM Hypothesis Router	Use a local LLM to read compressed logs and suggest game type, possible mechanics, and which solver to try. Do not let it control every action.	Frontier LLMs score poorly as direct agents, but they can still help route strategies and explain patterns. The technical report says frontier AI systems scored below 1% while humans solved 100%.	Slow, hallucination-prone, and hard to trust without verifier.
19	Workspace Optimization / Multi-Agent Blackboard	Maintain shared artifacts: transition logs, object maps, failed hypotheses, discovered rules, candidate solvers, successful traces. Multiple modules update the workspace.	DreamTeam-style workspace optimization improved public ARC-AGI-3 performance by coordinating hypothesis, planning, probing, and failure routing.	Complex engineering.
20	Trace Compression / Plan Minimization	After finding a winning action sequence, remove unnecessary actions using replay tests: delete chunks, shorten paths, replace loops with direct macros.	Scoring punishes inefficient action counts, so shortening the solution can dramatically improve score.	Requires reliable replay or simulation.
21	Portfolio Solver / Algorithm Router	Run multiple solvers: pathfinding, click search, object manipulation, pattern matching, MCTS, value model, LLM hypothesis. Route based on observed mechanics.	No single algorithm will solve all private games. A portfolio is more robust.	Needs good routing and time budgeting.
22	Synthetic Game Pretraining	Generate many small ARC-like interactive grid games. Train perception, action-effect prediction, novelty search, and planner selection.	Useful for learning general priors before private games.	Synthetic distribution may not match private games.
23	Model-Free RL	Train a policy network to act directly from grid frames.	Simple conceptually and may learn some affordances.	Weak for unseen games with sparse rewards; not ideal as main strategy.
24	Pure Frontier/Local LLM Agent	Feed frames/logs to an LLM and ask it to choose the next action.	Easy to implement.	Not competitive. ARC-AGI-3 was designed to expose failures of direct frontier agents.
25	Random / Evolutionary Action Search	Generate many random action sequences, keep those that produce changes, mutate promising traces.	Can discover simple mechanics accidentally.	Very inefficient and likely poor under action-based scoring.
My Recommended Winning Stack

The strongest practical strategy is not one algorithm. It is a portfolio architecture:

1. Object perception
2. State hashing and state graph memory
3. Systematic exploration and no-op pruning
4. Action-effect learning
5. Goal/progress inference
6. Planner selection
7. Model-based search / executable world model
8. Trace compression
9. Efficient final execution
What I Would Build First

Phase 1: Core engine

Object segmentation
State hashing
Transition graph
No-op detection
Loop detection
Click candidate generator

Phase 2: Solvers

Navigation solver
Click/toggle solver
Object interaction solver
Pattern matching solver
Beam search
MCTS with object-level actions

Phase 3: Intelligence layer

Progress heuristic
Value model
Local LLM hypothesis router
Workspace memory
Failure router

Phase 4: Score optimization

Replay winning traces
Remove redundant actions
Compress paths
Transfer learned rules to later levels

My final ranking: Object-Centric State Graph Planner + Systematic Exploration + Model-Based Planning + Trace Compression is the best base. Local LLMs can help, but only as a hypothesis/router module, not as the main action policy.