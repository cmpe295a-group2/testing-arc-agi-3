# Research log: from answer replay to genuine online learning

Date: 2026-06-20

## Correction

The former implementation claimed 25/25 public games and 183/183 levels. The
execution was real, but the mechanism was not a general solver:

- 23 environments replayed published action traces indexed by public ID;
- two environments used environment-specific repaired policies;
- an unseen ID had no policy and would fail immediately.

That result is invalid for the actual objective—private/OOD leaderboard
performance. The trace cache and fixed policies were deleted, not hidden behind a
flag. `audit.py` now makes their return a test failure.

## Evaluation contract

Policy input is restricted to public observations. The optional simulator exposes
only `fork`, `step(Action)`, and `observe()`; it does not expose the Python object,
source, game ID, metadata, hidden score fields, or internal valid-action generator.

Two scores must always be reported separately:

1. **simulator** — anonymous black-box planning when a clone capability exists;
2. **online** — observation-only learning where every probe is a real action.

The public evaluator may know the environment label after the run for reporting.
The policy cannot.

## Research reviewed

| Work | Useful evidence | Consequence here |
|---|---|---|
| ARC Prize methodology and technical report | Goal is unstated; RHAE rewards completion and action efficiency | Learn mechanics online and measure completion honestly |
| Rudakov et al., *Graph-Based Exploration for ARC-AGI-3* (arXiv:2512.24156) | Training-free state graphs work, but exhaustive exploration collapses in large spaces | Keep the graph; add object actions, hypotheses, and goal-directed ordering |
| Rodionov et al., *Executable World Models for ARC-AGI-3* (arXiv:2605.05138) | Executable models + verifier + divergence stopping substantially outperform blind exploration | Use a narrow black-box model interface and verify every real transition |
| ARC Prize preview learnings | Frame-change and state-graph priors were the strongest non-language baselines | Learn action effects and information value online |

The current web search endpoint returned HTTP 403 during this rewrite. These are
the primary sources already fetched and archived in the repository research notes;
no new secondary claim was introduced to compensate for the failed search.

## Approaches tested

| Approach | Result | Decision |
|---|---:|---|
| Blind deepcopy BFS/novelty prototype | approximately 16/183 public levels in earlier experiments | Retain only as a shallow/local planner |
| Stored public traces and hand repairs | 183/183, but zero meaningful OOD evidence | Deleted as answer leakage |
| Observation-only UCB graph | 0/19 levels on a quick three-game sample | Valid fallback, but too weak alone |
| Anonymous black-box object search | 2/19 levels on the same quick sample | Valid and non-zero; full benchmark required |
| Anonymous black-box object search, full 25 | **2/183 levels, 0/25 games, 0.25% mean RHAE** | Honest baseline; model induction is now the bottleneck |

Quick-sample settings were intentionally small: three environments, 15 seconds
per environment for simulator mode and five seconds for online mode. They are
smoke measurements, not final scores.

The full run used 0.10 seconds of BFS plus 0.90 seconds of guided search per plan,
an eight-second wall-clock cap per environment, depth 120, 32 click proposals, and
50,000 simulated nodes per plan. Only two tutorial levels completed. Longer search
did solve additional tutorial levels in the earlier three-game smoke run, but it
did not change the central diagnosis: unguided state-space expansion is not the
private solution.

## Implemented algorithm

1. Parse every frame into connected objects and palette/object-count features.
2. Propose all simple actions and object-centred clicks; never call a hidden click
   oracle or enumerate policy tables.
3. Try shallow BFS because it gives shortest solutions and strong RHAE when the
   tutorial is genuinely shallow.
4. Continue with novelty-guided best-first search. Order states by abstract novelty,
   observed action value, structural change, and any cross-level goal hypothesis.
5. When a level completes, learn clear/reach hypotheses from the real successful
   trajectory and transfer only that abstract mechanic to later levels.
6. When no simulator exists, learn transition edges and action values from real
   interaction using UCB exploration.

## Known limitations

- Exact online world-model synthesis is not yet implemented; the current learned
  model is a transition graph plus small goal-hypothesis portfolio.
- Component-centroid clicks miss controls requiring background or exact empty-cell
  coordinates.
- Frame identity aliases hidden states; short action context helps but is not a
  full belief-state model.
- Goal mining currently covers clear/collect and reach/landmark families, not
  arithmetic, ordering, program execution, physics, or multi-object constraints.
- Global search still suffers exponential depth. A black-box simulator gives more
  experience, not automatic abstraction.
- The private wrapper may forbid or hide cloning. Online mode is therefore the
  portability floor and currently has very low measured ability.

## Highest-potential next work

Build a small executable DSL learned from transition tuples:

- object motion and collision rules;
- counters, inventories, toggles, and transformations;
- candidate win predicates;
- counterexample-guided selection of the next probe;
- symbolic planning plus macro compilation;
- divergence checks after every real action.

Promotion gate: the DSL must improve a source-hidden mechanic-family holdout. A
gain on the same public IDs is insufficient.

## Primary sources

- https://docs.arcprize.org/methodology
- https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings
- https://arxiv.org/abs/2512.24156
- https://arxiv.org/abs/2605.05138
