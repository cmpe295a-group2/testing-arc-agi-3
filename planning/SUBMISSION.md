# ARC‑AGI‑3 — Offline Submission & Speed Packaging

**Team:** SJSU CMPE 295A, Group 2 · **Updated:** 2026‑06‑20
**Track:** Kaggle "ARC Prize 2026 — ARC‑AGI‑3", offline notebook · no internet · ≤ 9 h · single RTX 6000 (Ada, 48 GB).

> ⚠️ **SUPERSEDED IN PART — read `KAGGLE_HARNESS.md` first.** A real 0.46 submission confirms the scored notebook runs in **`OPERATION_MODE=online` against a `gateway:8001` server** (RemoteEnvironmentWrapper) → **frame-only, no `_game`, no private source on disk.** The deepcopy/source engine below works ONLY on the 25 public dev games (in-process `LocalEnvironmentWrapper`), NOT on the scored private set. Use this doc for the dev/training harness and the deepcopy machinery; use `KAGGLE_HARNESS.md` for the actual submission scaffold and build a **frame-only `Agent`** for scoring.

> This is the build/run recipe for packaging `agent/forward_model_agent.py` into the Kaggle offline submission, with the **process‑pool + budget governor** that turns the 9 h budget into ~20 min/game (vs a single‑thread sweep). Grounded in the verified engine source (see `ARC_AGI_3_SOTA_research.md`).

---

## 0. Why this matters (measured)

The search is **CPU‑bound** (deepcopy + simulate are pure‑Python → GIL‑bound), so a process pool across games gives near‑linear speedup, and the extra per‑game compute lifts coverage:

| Run (25 public games) | per‑game budget | levels | mean game‑RHAE | wall (6 workers) | speedup |
|---|---|---|---|---|---|
| sequential | ~8 s | 8/183 | 1.6% | 207 s | 1× |
| parallel | ~25 s | 8/183 | 1.7% | 63 s | **5.1×** |
| parallel | ~70 s | **16/183** | **3.2%** | 259 s | **5.3×** |

→ The 9 h real budget gives **~T·C/G ≈ 20 min/game** (4 cores, 110 private games) — far more than any sweep above, which solves the *borderline* levels (stuck only for lack of time, e.g. **tu93 0→4/9** at 70 s). Deep levels (depth ~44) still need win‑condition inference; compute alone won't crack them.

`agent/run_parallel.py` implements this and is Windows‑testable (arcengine‑only via the MiniEnv shim).

---

## 1. The offline environment

- **Install wheels with NO internet:** `pip install --no-index --find-links arc_agi_3_wheels/ arc-agi arcengine numpy pydantic flask requests pillow …`. All deps are in `arc_agi_3_wheels/` (Linux `cp312`). **`arc_agi/__init__` eager‑imports flask + requests** — they're in the wheels, install them. **No torch/tensorflow** in the wheels (none needed by our agent).
- **Develop/CI on the pinned Kaggle Linux image**, not the Windows dev box (wheels are Linux `cp312`; our `.venv-arc` on Windows only loads the pure‑python `arcengine` for logic tests).
- **No network anywhere** in the submission path. Do **not** import `main.py` (HTTP to `three.arcprize.org`) or `agents/__init__` (see §3).

## 2. Building the env in OFFLINE mode

Two options, both local:
- **Direct (what our dev runner does):** build a `LocalEnvironmentWrapper(EnvironmentInfo(...), scorecard_manager=...)` per game from `environment_files/<id>/<ver>/` (`metadata.json` → game_id/tags/baseline_actions; class name derives `cd82`→`Cd82`; file `<id>.py`). `wrapper._game` is the live `ARCBaseGame`. [verified: local_wrapper.py]
- **Via the SDK:** `Arcade(OperationMode.OFFLINE)` + `.make(game)` — but force `OFFLINE`/`NORMAL` (never `COMPETITION`, which routes to a remote HTTP wrapper with no `_game`). The default `Arcade()` mode is `NORMAL` (local+API) — set OFFLINE explicitly.

## 3. ⚠️ The `agents/__init__` footgun (hard requirement)

`from agents.agent import Agent` first runs `agents/__init__.py`, which **eager‑imports** `llm_agents`/`langgraph_*`/`smolagents` (→ openai/langgraph/smolagents — **not in the offline wheels** → ImportError). Two safe options:
- **Don't subclass the SDK Agent at all** — our `ForwardModelPolicy` drives `arc_env.step(id, data)` directly via `run_episode` (identical loop semantics). This is the default and the cleanest offline path.
- **If the grader requires an SDK `Agent` subclass** (swarm), load `agents/agent.py` in isolation without the package `__init__` (stub the package — see the `SDKAgentAdapter` note in `forward_model_agent.py`), then subclass it and have `choose_action` call `policy.choose_action(...)`.

## 4. Parallelism + budget governor

- **Process pool** (not threads — GIL): one worker per usable core, each worker builds its OWN OFFLINE wrapper for its assigned game and runs the policy. `agent/run_parallel.py` is the reference (swap the MiniEnv for a `LocalEnvironmentWrapper(scorecard_manager=<real>)` on the grader).
- **Budget governor:** `per_game_cap = total_budget_s × workers / n_games` (e.g. `9h×4/110 ≈ 29 min/game`). Per game, the cascade caps each level at `bfs_budget_s` (BFS, optimal shallow) then `novelty_budget_s` (deep). On exhaustion it executes the best partial and moves on — **partial completion still scores** (completion is the dominant RHAE lever). Bias remaining budget toward later levels (weight `w_l = l`).
- **Checkpoint** per‑game best trajectories so a near‑timeout run still submits best‑so‑far.

## 5. Day‑1 Simulability Probe — run BEFORE trusting forward‑model search

The whole engine hinges on `deepcopy(arc_env._game)` working on the REAL grader. `simulability_probe()` checks: `_game` exposed · stepping a clone leaves the original's `_action_count`/`_score` unchanged (free) · two clones + same actions → identical frames (deterministic). 🔴 **Risk (verified):** the harness already blocks **re‑instantiating** a 2nd game client in competition mode (SingularityNET paper); `deepcopy` of the live object is a *different* mechanism and may survive, but it's the same risk class. **GREEN → forward‑model search is primary; RED → fall back to the CNN / learned‑deterministic‑model engine.** Never ship the deepcopy simulator as the SOLE engine. [see ARC_AGI_3_SOTA_research.md §4]

## 6. Determinism, RNG guard, license

- **Seed all global RNG** at startup; same submission → same score.
- **RNG guard** (`rng_snapshot`) around every rollout — `lf52` does an unseeded global `np.random.shuffle`; **prefer `deepcopy` over replay‑from‑RESET** (replay diverges on `lf52`).
- **License: CC0 or MIT‑0** (prize requirement) — include full source, build instructions, wheel manifest. Winners must open‑source.

## 7. Submission entrypoint (sketch)

```python
# submission notebook (Linux, offline)
# 1. pip install --no-index --find-links arc_agi_3_wheels/ arc-agi arcengine numpy pydantic flask requests pillow
# 2. enumerate environment_files/<id>/<ver>/
# 3. ProcessPoolExecutor(max_workers=cores): for each game ->
#       wrapper = LocalEnvironmentWrapper(EnvironmentInfo(...), scorecard_manager=mgr)  # OFFLINE
#       probe = simulability_probe(wrapper)            # GREEN/RED fork
#       run_episode(wrapper, ForwardModelPolicy(cfg_with_governor_cap), max_actions=5*sum(baselines))
# 4. merge per-game scorecards -> total RHAE -> write submission
```

---

## Files
- [agent/forward_model_agent.py](../agent/forward_model_agent.py) — the agent (probe + BFS + novelty + reach + match heuristics + the optional Active‑Inference engine)
- [agent/run_parallel.py](../agent/run_parallel.py) — process‑pool + budget governor (5.3× speedup measured)
- [agent/batch_eval.py](../agent/batch_eval.py) · [agent/try_run.py](../agent/try_run.py) — sequential eval + single‑game dev runners

*Coverage today: ~16/183 public levels (3.2% mean RHAE) at 70 s/game, goal‑agnostic search, RHAE capped where solved. The ceiling beyond this is win‑condition inference (ARC_AGI_3_SOTA_research.md), not more compute.*
