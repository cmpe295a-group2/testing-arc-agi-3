# SJSU 295A Group 2 — ARC-AGI-3 submission agent (v1, target ≈ 0.46)

Clean from-scratch **frame-only** agent for the Kaggle "ARC Prize 2026 — ARC-AGI-3"
private leaderboard. The scored notebook plays the 110 hidden games against a
`gateway:8001` server in ONLINE mode, so the agent only ever sees frames (no game
object, no source) — see [`../planning/KAGGLE_HARNESS.md`](../planning/KAGGLE_HARNESS.md).

## What it does
Per game (CNN + replay buffer persist across levels; the graph is level-local):
1. **ExplorationPolicy** — a transition graph keyed by frame hash. Prefer untried
   actions; when a state is exhausted, BFS the known edges to the nearest state with
   an untried action (frontier exploration). Planned steps are verified and bad edges
   dropped (handles hidden state / animation).
2. **ChangeNet** — a small CNN trained online to predict *P(action changes the frame)*
   with a novelty-weighted target; used to rank untried actions and to sample once the
   graph is exhausted.

This is the proven public recipe (graph + online frame-change CNN, the 0.42–0.46
family) rebuilt clean, **plus** first-class ACTION7 support and guarded imports so the
exploration core is unit-tested offline.

## Files
- `my_agent.py` — the agent (single self-contained file; becomes `my_agent.py` on Kaggle).
- `test_explore.py` — offline unit tests for the torch/SDK-free exploration core.
  `python submission/test_explore.py` → **ALL TESTS PASSED** (run on the dev box).

## Build status
- ✅ `python -m py_compile submission/my_agent.py` — clean.
- ✅ `python submission/test_explore.py` — exploration graph, frontier routing, edge
  verification, novelty targets, and candidate generation all pass (numpy-only, no GPU).
- ✅ CNN path smoke-tested (`ChangeNet` featurize→forward→train→prior; shapes correct).
- ✅ `python submission/run_public.py` — the FULL agent runs end-to-end against the 25
  in-process public games. **Graph-only (no CNN): ≈ 10–12/183 levels, 7–8/25 games**
  (controlled seeds 1/2/3 → 9/12/11; high run-to-run variance ±3). LOWER bound — the dev
  box has no GPU so the ChangeNet prior is off; on the Kaggle RTX 6000 it lifts coverage
  and efficiency. `--no-cnn` forces the fast graph-only benchmark (the CNN runs but is
  ~70× too slow to evaluate on CPU).
- 🧪 Graph-only exploration tuning (controlled A/B, fixed seeds):
  - **Tabular novelty/change prior: REJECTED** — it regressed coverage (13/10 uniform vs
    8/7 prior across 2 seeds). Greedy novelty funnels exploration; uniform covers more.
    The CNN's real value is SPATIAL click selection, which a colour-keyed table can't
    replicate. Kept behind `ARC_ENABLE_CHANGE_PRIOR` (off by default), maybe useful only
    blended with the CNN.
  - **Richer click candidates (blob extremes, cap 48): kept** — neutral-to-slightly-better
    at fixed cap, never worse (`ARC_SIMPLE_CLICKS=1` reverts). Raising the cap hurt
    (spreads the fixed action budget too thin).
  - Conclusion: goal-agnostic graph-only is near its ceiling (~10–12, matching the
    deepcopy engine's ~12–16); the learned spatial CNN on GPU is the real lever.
- ⏳ Real (CNN + GPU) score: measured by submitting on Kaggle (the gateway + RTX 6000).

## How to submit (Kaggle notebook)
Create a GPU notebook attached to the competition, with these cells. **Cell 2's body
after `%%writefile` is the entire contents of `my_agent.py`** (single source of truth —
paste it verbatim).

**Cell 1 — install (offline wheels):**
```python
!pip install -q --no-index --find-links \
    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \
    arc-agi python-dotenv
```

**Cell 2 — write the agent:**
```python
%%writefile /kaggle/working/my_agent.py
# <<< paste the full contents of submission/my_agent.py here >>>
```

**Cell 3 — scored-run launcher (only runs during a real submission):**
```python
import os
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    !curl --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 600 \
        http://gateway:8001/api/games
    !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \
        /kaggle/working/ARC-AGI-3-Agents
    !cp /kaggle/working/my_agent.py \
        /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py
    with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
        f.write(
            "from typing import Type\n"
            "from dotenv import load_dotenv\n"
            "from .agent import Agent, Playback\n"
            "from .swarm import Swarm\n"
            "from .templates.random_agent import Random\n"
            "from .templates.my_agent import MyAgent\n"
            "load_dotenv()\n"
            'AVAILABLE_AGENTS: dict[str, Type[Agent]] = {"random": Random, "myagent": MyAgent}\n')
    with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
        f.write(
            "SCHEME=http\nHOST=gateway\nPORT=8001\n"
            "ARC_API_KEY=test-key-123\nARC_BASE_URL=http://gateway:8001/\n"
            "OPERATION_MODE=online\nRECORDINGS_DIR=/kaggle/working/server_recording\n")
    !cd /kaggle/working/ARC-AGI-3-Agents && MPLBACKEND=agg python main.py --agent myagent
```

**Cell 4 — placeholder submission for the commit (non-rerun) path:**
```python
import os
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    import pandas as pd
    pd.DataFrame([['1_0', '1', True, 1]],
                 columns=['row_id', 'game_id', 'end_of_game', 'score']
                 ).to_parquet('/kaggle/working/submission.parquet', index=False)
```

Notes:
- The agents `__init__.py` is rewritten to import only `random` + `myagent` so the LLM
  template modules (openai/langgraph/smolagents) are never imported (they're absent
  offline and would crash the run).
- `ARC_GLOBAL_BUDGET_S` / `ARC_EXPECTED_GAMES` env vars tune the per-game time budget
  (default: 8.5 h shared, ~6 games-in-flight). The Swarm runs games in parallel; the
  budget is shared across the process.
- Optional warm start: drop a `pretrained.pt` (matching `ChangeNet`) as a Kaggle dataset
  at `/kaggle/input/arc-agi3-pretrained/` — the agent loads it if present, else trains
  from scratch online.

## Roadmap
- **Phase 1 (this):** match the 0.42–0.46 family — clean graph + online CNN, valid
  end-to-end submission. ← we are here (logic verified; submit to get the real score).
- **Phase 2 (beat 0.46):** the unshared frontier — smarter exploration (count-based /
  intrinsic-reward, better click-candidate proposals), a learned forward/world model to
  plan toward novelty, and the local open-weight-LLM world-model path on the RTX 6000.
  Use the 25 public games + our deepcopy engine (`../agent/`) as the offline lab to
  pretrain `ChangeNet` and to measure changes before spending a daily submission.
