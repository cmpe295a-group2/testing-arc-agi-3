# `chat_gpt_5_5`: game-agnostic ARC-AGI-3 agent

This directory no longer contains the public replay solver. That implementation
produced a real 25/25 replay result, but it selected stored answers by public game
ID and therefore measured coverage, not intelligence. The traces and two fixed
policies have been removed.

The replacement is an anonymous online-learning agent. It receives only:

- the latest grid frame;
- public available-action IDs;
- `levels_completed`, `win_levels`, and terminal state;
- optionally, a capability-limited black-box `fork / step / observe` simulator.

It never receives a game ID, Python source, metadata, recorded solution, hidden
state, valid-click oracle, or environment object.

## What it learns

1. **Object proposals** — 4-connected components, palette counts, bounding boxes,
   and object-centred click candidates.
2. **Action effects** — empirical information gain, lethal actions, and successful
   action classes, learned again inside each anonymous environment.
3. **State graph** — observable state hashes plus short action context for hidden
   progress that produces identical frames.
4. **Goal hypotheses** — after an easy level is genuinely solved, mine reusable
   `clear colour` or `reach landmark` hypotheses from the successful trajectory.
5. **Planning** — short BFS for optimal shallow solutions, followed by
   goal/novelty-guided black-box search. If no simulator exists, use online UCB
   exploration over observed transitions.

No rule or coordinate is specific to any of the 25 public environments.

## Run

```powershell
# Prove the policy contains no answer-table leakage
.\.venv-arc\Scripts\python.exe -m chat_gpt_5_5.audit

# Unit tests, including unseen toy mechanics
.\.venv-arc\Scripts\python.exe -m unittest chat_gpt_5_5.test_agent -v

# Strong mode: black-box simulator, without source/object introspection
.\.venv-arc\Scripts\python.exe -m chat_gpt_5_5.evaluate --mode simulator

# Strict fallback: only real observations and real actions
.\.venv-arc\Scripts\python.exe -m chat_gpt_5_5.evaluate --mode online
```

The evaluator uses public source only inside the trusted harness to instantiate an
environment, exactly as a runtime loader must. It then assigns `env-001`,
`env-002`, ... and constructs `GeneralistAgent()` with no identity argument.
Metadata is retained outside the policy solely for post-run RHAE reporting.

## Measured baseline

Full anonymous simulator-mode run on 2026-06-20:

```text
settings: BFS 0.10 s + guided search 0.90 s/plan; 8 s/environment
result:   2/183 levels; 0/25 games; mean game RHAE 0.25%
solved:   one tutorial level in two environments
```

The observation-only three-environment smoke run solved `0/19` levels. These
numbers are intentionally not mixed with the deleted 183/183 replay result.

## Files

| File | Responsibility |
|---|---|
| `core.py` | Narrow `Action`, `Observation`, and black-box simulator protocol |
| `perception.py` | Components, abstract features, click proposals, novelty |
| `agent.py` | Mechanic memory, goal inference, planner, online policy |
| `engine.py` | Trusted local loader and capability-limited simulator adapter |
| `evaluate.py` | Anonymous public benchmark; no win requirement |
| `submission.py` | Thin official-runtime adapter; clone capability is optional |
| `audit.py` | Fails on public IDs, source readers, traces, or fixed-policy artifacts |
| `test_agent.py` | Unit and unseen-mechanic tests |
| `RESEARCH_LOG.md` | Evidence, failed approaches, limitations, next experiments |
| `BENCHMARK_2026-06-20.md` | Full per-environment anonymous benchmark result |

## Claim boundary

This code is a legitimate general algorithm, but it is **not guaranteed to solve
every private game**. No current method has established that capability. Public
performance is reported even when low; a score is accepted only when produced by
the anonymous policy under the leakage audit.

Simulator mode also remains conditional: if the competition wrapper does not
permit cloning the live state, only online mode is submission-valid. The policy
code is shared by both modes; simulator mode changes the amount of safe planning
experience, not the rules available to the agent.

`submission.CompetitionPolicy` is the integration point for an official SDK
`Agent` subclass. Construct it without a game ID, then call
`policy.choose_action(latest_frame, self.arc_env)`. Set
`allow_black_box_clone=False` to force the sealed-grader path.
