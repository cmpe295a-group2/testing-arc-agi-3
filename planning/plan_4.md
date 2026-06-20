ARC-AGI-3 Overview:
Create AI agent to play simple video game with no user input
Must figure out rules, goal, and strategy by itself
Simple 2D grids with 64x64 cells, cell values are 0-15 representing colors/states
7 actions (up/down/left/right, interact, click, undo)
No internet allowed (only local LLMs)

Things we can work on:
Perception
How to interpret the grid best? 
Turn pixels into objects like player, walls, key etc
Exploration
Action space pruning
Maintaining state graph (reinforcement learning?)
Memory optimization
Hashing visited states, prune actions that lead nowhere
Goal inference
What does each game want - how to decide without input
Planning/execution?
Shortest path after deciding rules

Plan:
Review current literature
Previous years ARC prizes, world model papers, etc
Set up simple model (random choice)
Play through some games using a random choice model to get idea of how challenge works
Choose 4 prototypes (one for each member)?
Each member can build a small agent using a different base idea
Compare at the end based on different heuristics


Things to do:
1. Update schedule to match our schedules, create milestones
Project statement
Current lit
2. Limit scope to certain part of project



Background Research
Jun 19, 2026


ARC-AGI-2

Last year’s competition
The goal is to take a large amount of input -> output demonstration pairs and then infer the transformation rule and produce the output
No interaction, no exploration
Grand prize remained unclaimed -> no one was able to meet minimum qualifications for prize - this shows still plenty of room for improvement

Top 2 results

NVARC - 24.03% accuracy - github.com/1ytic/NVARC 
Wanted to use LLM’s but couldn’t fit in runtime, so moved reasoning into an offline synthetic-data pipeline to train a small 4B model
For synthetic data generation, they took human-written descriptions of existing puzzles and broke them into structured parts (like how to draw grid, transformation steps, rules, etc) and then used that to generate thousands of new puzzles with known answers 
Used test-time training - when new puzzles appeared at test time they would run a few training steps before solving
Used a Tiny Recursive Model (TRM) along with the 4B LLM and combined answers
Main concept - push heavy lifting of LLM to an offline data-making and training pipeline, and then use a small specialized LLM for solving. Can use synthetic environment generation in ours
This concept assumes we have examples input-> output pairs, but in ARC-AGI-3 we have no examples.

The ARChitects - 16.5% accuracy - https://arxiv.org/abs/2505.07859
Another LLM approach, with two twists
First change is instead of writing tokens left to right, one at a time, they used a 2D-aware masked diffusion model - starts with a blank grid and fills everything in at once while following the row/column layout. Better for grids/images than sentences
Used recursive self-refinement for submissions - produce draft answer, improve it, repeat. Combine this with perspective-based scoring (solve the same puzzle under multiple transformed views - rotations, flips, etc) and keep the most confident answer.
We can use part 2 for our project but part 1 is about creating a grid, which in AGI-3 our goal is produce a sequence of actions, not just a single grid.


Top 3 papers
1 TRM — Tiny Recursive Model - https://arxiv.org/abs/2510.04871
A 7M-parameter single network with separate answer state y and latent state z. For up to N_sup = 16 deep-supervised improvement steps it (i) recursively updates the latent z given the question x, current answer y, and current z, then (ii) updates the answer y from z. This progressively refines the answer, fixing earlier errors, in a hugely parameter-efficient, overfitting-resistant way. 45% on ARC-AGI-1, 8% on ARC-AGI-2. Builds on the Hierarchical Reasoning Model (HRM).
2 SOAR — Self-improving evolutionary program synthesis - https://arxiv.org/abs/2507.14172
"Self-improving Operators for Automated program Refinements." An LLM proposes candidate programs; a verifier scores them on training pairs; the LLM is then fine-tuned on its own successful search traces, and the loop repeats. +52% on open-source ARC-AGI-1 solver performance, with no hand-engineered DSL and no solution dataset. The clearest demonstration that a generate→verify→self-train loop can autonomously improve.
3 CompressARC — ARC-AGI Without Pretraining - https://arxiv.org/abs/2512.06104
76K parameters, no pretraining, no dataset, no search. One randomly-initialized network is trained only at test time on a single puzzle by minimizing description length (MDL) — Liao shows a VAE loss with decoder regularization can substitute for combinatorial search. 20% on ARC-AGI-1, ~20 min/puzzle on one RTX 4070. The purest expression of "simplicity bias as the learning signal."
What transfers?

The refinement loop itself. The strongest ARC-AGI-3 agent (SingularityNET executable world model) is a refinement loop: observe → build model → verify against observations → refactor toward simplicity (MDL) → plan → act. ARC-AGI-2's central lesson is the right mental model for ARC-AGI-3.
Synthetic data/environment generation. NVARC's "generate the distribution, train offline, ship small" is the template for any trained ARC-AGI-3 agent under the offline constraint — except you generate synthetic interactive environments, not static puzzles, to meta-train an in-context/RL agent. This is the single most transferable lesson and likely necessary for a competitive offline entry.
Test-time training. Fine-tune a small model on the transitions you collect within the episode (self-supervised next-frame prediction, value, or policy), adapting per-environment at test time — the interactive analogue of TTT-on-examples.
MDL / simplicity bias. CompressARC's MDL objective and SingularityNET's "refactor to simpler abstractions" are the same idea; a good ARC-AGI-3 world model should be the compressive one that generalizes to later levels.
Generate-and-verify + ensembling/voting. Propose multiple world models or plans, keep the one the verifier (plan-executor / observation-consistency check) accepts; ensemble/vote across augmented views. Directly counters the "premature commitment to a wrong model" failure mode.
Evolutionary program synthesis. SOAR/Berman/Pang's evolve-and-verify is a strong recipe for inducing the mechanics/win-condition program of an environment.
Small-model + offline discipline. The whole NVARC philosophy is the answer to the no-internet Kaggle sandbox.
Tiny recursive nets (TRM/HRM) — repurpose the architecture as the world-model's next-frame predictor, a value head, or a policy that runs fast under contest compute, rather than as the end-to-end solver.

ARC-AGI-3

Overall goal is a refinement loop: Propose model -> get feedback -> improve -> repeat
Not many current available algorithms as competition is still going on
2 main types - search/exploration which is fast but very inefficient and gets low scores, and the LLM family that is slow/brittle and extremely api dependent.

Potential Solutions

Search & state graph exploration
Build a graph of observed states and explore it, using pruning/looping actions. Low/no training involved
Blind Squirrel (Will Dick, preview 2nd, 6.71%, 13 levels): builds a directed state graph from frames; prunes actions that loop or don't change state; when score improves, back-labels states with distance-to-milestone and trains a small ResNet18 value model to rank (state, action) pairs. A clean hybrid of explicit graph + lightweight learned value function. ~109K actions.
"Explore It Till You Solve It" → "Graph-Based Exploration for ARC-AGI-3" (Evgenii Rudakov, preview HM, 3.64%; later arXiv 2512.24156): a training-free graph-based exploration system. Frames as nodes, actions as edges, hash-merged states. Explicitly motivated by curiosity-driven exploration (ICM) and Go-Explore. Reported 3rd on the private preview leaderboard, median 30/52 levels across 6 games. Conclusion: systematic state tracking matters more than model size for this task.
GuidedRandomAgent (2.24%): rule-based "smart random."

				
Learned exploration / reinforcement learning
StochasticGoose (Dries Smit / Tufa Labs, advised by Jack Cole/MindsAI; preview 1st, 12.58%, 18 levels): a 4-layer CNN that predicts which actions will change the frame, trained online with a simple RL signal, to bias exploration away from no-ops. Still fundamentally "smart random" — it used ~256K actions. Its full-launch score collapsed to 0.25% (see §1.3), roughly frontier-LLM level. 
LLM / LRM harnesses (research track - API dependent)
Duke "Hill-Climbing ARC-AGI-3" (Fox, Wang, Rosu, Dhingra, 2026): wraps a large reasoning model and lets it execute arbitrary Python to selectively retrieve and transform its action history — directly attacking the core problem that 64×64 frames blow up the context window. Solved all 3 public games at near-human action counts; reported the best public score for a while.
Arcgentica (Symbolica AI): an orchestrator–subagent architecture; subagents return compressed text summaries so the top-level planner's context stays bounded. Solved all 3 public games.
Play Zero / Tomas Engine (preview): LLM-driven; limited, crash-prone.

Note: The shared lesson from all three: context management is the binding constraint for LLM agents here. A naive rolling window of 64×64 frames exhausts the budget almost immediately. 


Programmatic / executable world models
"Executable World Models for ARC-AGI-3 in the Era of Coding Agents" (S. Rodionov, SingularityNET, arXiv 2605.05138, Jun 2026): the strongest publicly reported result. A coding agent (Codex CLI + GPT-5.5) maintains an executable Python world model (state representation, transition prediction, goal-check, planner), verifies it reproduces observed transitions, refactors it toward simpler abstractions (a practical proxy for a Minimum-Description-Length simplicity bias), and plans through the model before spending real actions. A plan executor simulates a plan in the model, executes it in the real game, and stops the instant a predicted frame diverges from the observed frame — i.e., execution doubles as an online falsification test of the model.
Results on the 25 public games: GPT-5.5 → 15/25 fully solved, mean RHAE 58.12%; GPT-5.4 → 8/25, 41.29%. (Public-only; private generalization untested.)
Failure mode identified: premature commitment to a wrong/over-specific world model. Proposed fixes: competing-hypothesis tracking, falsification-driven action choice, parallel multi-model protocols, and a reusable "agentic skills" library (BFS, A*, constraint solving, backtracking, subgoal decomposition, state abstraction).
Caveat: it runs on a hosted frontier API → not Kaggle-eval-eligible, but it is a paper-track / community-leaderboard exemplar and a blueprint for an offline version using a small open-weight coder.
	Component 1: Perception & State Representation
The core challenge here is that an uncompressed 64x64 grid changes too often and overwhelms the memory of most AI models.
DOCX
Literature Approach - Object Parsing: Translate the raw grid into text-based object lists, identifying entities like the player, walls, and keys.
 DOCX
Literature Approach - Context Bounding: Use subagents to return compressed text summaries of the grid so a top-level planner's context window does not blow up.
 DOCX
Novel Approach - Convolutional Autoencoders: Train a lightweight, offline neural network strictly to compress the 64x64x16 grid into a tiny, dense mathematical vector, passing only that vector to the reasoning model.
Novel Approach - Quadtree Compression: Mathematically chunk large areas of solid background colors into single blocks, reducing the grid from 4,096 individual pixels to a handful of geometric shapes.
Component 2: Exploration Strategies
Randomly pushing buttons wastes actions and results in severe penalties in the competition. The agent needs a smart way to map the environment.
DOCX
Literature Approach - Directed State Graphs: Treat each visual frame as a node and each action as an edge to build an explicit map of the game.
 DOCX
Literature Approach - Hash Merging & Pruning: Hash visited states and aggressively prune any actions that loop back to previous states or cause no change (no-ops).
 DOCX
Literature Approach - Learned Exploration: Train a small CNN online using a simple reinforcement learning signal to predict and favor actions that actually alter the frame.
 DOCX
Literature Approach - Active Hypothesis Testing: Program the agent to deliberately choose actions that will disambiguate or falsify competing theories about how the world works.
 DOCX
Novel Approach - Novelty-Driven MCTS: Implement a Monte Carlo Tree Search (MCTS) that ignores the game's "score" entirely and instead rewards the agent purely for discovering a state it has never seen before.
Component 3: World Modeling & Rule Induction
Once the agent explores, it needs to figure out the hidden rules and win conditions of the level without any human instructions.
DOCX
Literature Approach - Executable Python Models: Have an agent write an explicit Python program representing the game's state and transition rules.
 DOCX
Literature Approach - Falsification & Refactoring: If the Python model predicts a frame that differs from the actual game, the agent must refactor the code toward simpler abstractions (Minimum-Description-Length).
 DOCX
Literature Approach - Bayesian Inference: Use probabilistic generative models over programs to simulate a game engine and decompose win conditions into subgoals.
 DOCX
Novel Approach - Neuro-symbolic Verifiers: Have a small offline LLM propose a game rule in natural language, then use a strict, deterministic logic solver (like Z3) to mathematically prove whether that rule holds true for every past frame the agent has seen.
Component 4: Synthetic Generation & Meta-Training
Because internet access is blocked, you cannot use massive models. The agent must be trained offline to learn how to adapt quickly before the competition begins.
DOCX
Literature Approach - Structured Decomposition: Take human-written descriptions of existing puzzles, break them into structured parts (grid drawing, rules), and generate thousands of new training environments.
 DOCX
Literature Approach - In-Context Meta-RL: Train a model across procedurally generated worlds with hidden rules so it learns to adapt on human-timescales during actual testing.
 DOCX
Literature Approach - Self-Improving Evolution: Use a loop where a model proposes candidate programs, a verifier scores them, and the model fine-tunes itself on its own successful search traces.
 DOCX
Novel Approach - Adversarial Game Generation: Create a two-agent training loop where a "Creator" agent tries to generate a valid ARC-AGI-3 level that a "Solver" agent fails to beat, slowly ramping up the complexity of the rules as the Solver gets smarter.
