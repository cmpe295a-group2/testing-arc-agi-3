# ARC Prize 2026 — ARC‑AGI‑3 · Master Strategy Document

**Team:** SJSU CMPE 295A, Group 2 · **Repo:** `arc-agi-3-agent` · **Date:** 2026‑06‑19
**Competition:** [ARC Prize 2026 — ARC‑AGI‑3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) (Kaggle, offline)

> This document is the team's single source of truth for how we win. It is grounded in (a) a full read of the starter repo and the bundled `arcengine 0.9.3` / `arc_agi 0.9.8` wheels, (b) reverse‑engineering of 8 of the 25 public game environments, (c) the official [ARC‑AGI‑3 Technical Report (Apr 2026)](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf) and the [30‑day preview learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings), and (d) a multi‑agent research + adversarial‑verification pass over the candidate algorithm families. Every load‑bearing claim is tagged with its evidence status.

---

## 0. Executive Summary (read this first)

**The thesis in one sentence:** *ARC‑AGI‑3 is a search‑and‑planning problem wearing the costume of a perception problem — and because the competition ships the games as deterministic, in‑process Python objects, the environment can serve as its own perfect simulator, turning "the goal is never told" into "the goal is detectable in a free rollout."*

**The five facts that dictate everything:**

1. **Offline.** No internet at inference. Every LLM template in the repo (OpenAI/LangGraph/smolagents) is **dead weight** — delete it from the submission path. The games run **fully locally** via the bundled wheels.
2. **Efficiency scoring (RHAE).** Per level `S = min(1.15, (h/a)²)` (human actions `h` over agent actions `a`, **squared**), gated by a **completion cap** weighted toward later levels. *Completion is the dominant lever; efficiency is a squared multiplier you cannot bank.*
3. **The goal is never stated.** Only Core‑Knowledge priors (objectness, geometry, physics, agentness). No language, numbers, or icons. Win conditions must be **discovered by interaction**.
4. **Non‑LLM approaches lead; frontier LLMs score <1%.** Humans solve 100%. The bottleneck is **strategy/exploration**, not perception.
5. **The eval set is intentionally OOD.** 25 public / 55 semi‑private / **55 fully‑private** (the scored set). Overfitting to the 25 public games is **futile and harmful** — only *engine‑level* generality transfers.

**The plan:** Build a general **forward‑model search agent** that (i) segments each 64×64 frame into objects, (ii) clones the in‑process game (`copy.deepcopy(arc_env._game)`) to get a free, exact simulator, (iii) searches that simulator — guided by novelty for goal discovery and by an inferred win‑condition heuristic for efficiency — to find a near‑optimal action sequence, and (iv) executes **only** that sequence on the scored instance. A small on‑GPU CNN and a learned deterministic transition cache are wired in as fallbacks.

**The one decisive fork (Day‑1, hard gate):** the entire strategy depends on whether the Kaggle grading harness hands us the live `_game` object for `deepcopy`. This is **code‑verified on the shipped wheels** but **not yet confirmed on the real grader**. Phase 1 is a go/no‑go probe. `GREEN` → forward‑model search is the primary engine and the path to 100%. `RED` → pivot to the learned‑world‑model + CNN fallback immediately.

**Evidence legend:** ✅ verified from source/report · 🧪 strongly indicated, confirm Day‑1 · ⚠️ external claim, not source‑verifiable here.

---

## 1. Project Overview & Code Breakdown

### 1.1 What this project is

We are building an **autonomous agent** that, dropped into a never‑before‑seen turn‑based grid game with **no instructions**, must (1) **explore** to learn the controls and mechanics, (2) **model** the dynamics, (3) **infer the goal** on its own, and (4) **plan and execute** an efficient solution — all in a single life, offline, scored against human action efficiency.

**The competition contract** (✅ Technical Report + Kaggle):

| Property | Value |
|---|---|
| Submission | Kaggle Notebook, **no internet** at inference |
| Compute | ≤ **9 hours**, single **RTX 6000 (Ada, 48 GB)**‑class GPU |
| Datasets | 25 public demo · 55 semi‑private · **55 fully‑private (scored)**, >1000 levels |
| Observation | one or more **64×64 grids**, each cell ∈ {0..15} (16 colors); a frame may be a short animation sequence |
| Actions | `RESET` + `ACTION1..5` (keys) + `ACTION6` (click x,y ∈ 0..63) + `ACTION7` (a 7th simple action) — each game uses a **subset** |
| Win | complete **all levels** (≥6/env; level 1 is an easy tutorial); difficulty grows by **composition** |
| Prizes | **$700K** Grand (first to 100%) · $75K Top Score · $75K Milestones (**Jun 30** & **Sep 30 2026**); winners open‑source under **CC0/MIT‑0** |
| Status quo (⚠️ report) | humans 100%; Opus 4.6 0.50%, Gemini 3.1 Pro 0.40%, GPT 5.4 0.20%, Grok 4.20 0.10% (official, no harness). Preview community: non‑LLM CNN 12.58%, state‑graph agent 6.71%, LLMs ~3–4% |

### 1.2 The mechanics we must get right (priority order)

1. **Completion before efficiency.** The environment cap = `(Σ weights of completed levels) / (Σ all weights)`, weights `w_l = l`. Finishing a *later* level is worth more than shaving actions off an *earlier* one. You cannot trade efficiency on easy levels for failure on hard ones. ✅
2. **Explore on copies, execute on the scored instance.** The scorecard mutates **only** via the wrapper's `step()`; simulated rollouts on a clone are free. This is the whole game. 🧪
3. **Goal discovery is the hard part.** The only honest reward signal is `levels_completed` ticking / `state == WIN`. Everything keys off detecting that in simulation. ✅
4. **Generality is mandatory.** Build for the engine, never for a specific public game. ✅

### 1.3 Repo structure & file‑by‑file breakdown

All paths under `arc-prize-2026-arc-agi-3/`. (✅ verified against source + extracted wheels.)

```
arc-prize-2026-arc-agi-3/
├── ARC-AGI-3-Agents/        # official agent SDK (the scaffold)
│   ├── agents/
│   │   ├── agent.py         # abstract Agent + Playback  ← the ONLY reusable core
│   │   ├── swarm.py         # multi-game orchestration (Arcade.make + threads)
│   │   ├── recorder.py      # JSONL record/replay (the human-replay harness)
│   │   ├── __init__.py      # agent registry (eagerly imports ALL templates ⚠️)
│   │   └── templates/       # random + LLM/LangGraph/smolagents (ALL API-based → DEAD offline)
│   ├── main.py              # ONLINE runner (HTTP to three.arcprize.org) → DEAD offline
│   └── tests/
├── arc_agi_3_wheels/        # offline wheels: arcengine, arc_agi, numpy, pydantic, pillow… (NO torch)
└── environment_files/       # the 25 public games as standalone obfuscated Python + metadata.json
    └── <id>/<ver>/<id>.py
```

| File | What it does | Relevance to the offline submission |
|---|---|---|
| `agents/agent.py` | Abstract `Agent`: the main loop `while not is_done and action_counter ≤ MAX_ACTIONS: choose_action → take_action → append_frame`. Two extension points: **`choose_action`** (your policy) and **`is_done`**. `take_action → arc_env.step(...)` hits the in‑process game. `_convert_raw_frame_data` materializes `frame`, `state`, `levels_completed`, `win_levels`, `available_actions`. | **The only structural code we keep.** Subclass it. ⚠️ **`MAX_ACTIONS = 80` is the most dangerous default** — real baselines reach 578; raise it to the server rule `5·h`. The `X-API-Key` headers are inert offline. |
| `agents/agent.py::Playback` | Replays a recorded JSONL trajectory action‑by‑action (`MAX_ACTIONS = 1e6`). | The official **"human‑replay = 100%"** harness. Useful as a *deterministic‑replay sanity tool*, **scores 0 on unseen envs** (memorized actions don't transfer). |
| `agents/swarm.py` | Builds N agents (one per game) on N threads; opens/closes a scorecard; constructs `Arcade()` (**default `NORMAL` = local+API**). | Reuse the `Arcade.make(...)` wiring; **force `OperationMode.OFFLINE`**. Replace threaded fan‑out with sequential, search‑heavy per‑env compute for the 9 h budget. |
| `agents/recorder.py` | Appends `{timestamp, data}` JSONL events; lists/loads recordings. | Our **trajectory format** + replay test rig. |
| `agents/__init__.py` | Reflects over `Agent.__subclasses__()` to build `AVAILABLE_AGENTS`; **eagerly imports** `llm_agents`, `multimodal`, `smolagents`, `langgraph_*`. | ⚠️ Those imports pull `openai`/`smolagents`/`langgraph` at load → **import error on a no‑internet image**. **Do not import this module**; register only our agent + `Random`/`Playback`. |
| `main.py` | Loads `.env` (→ `three.arcprize.org`), **HTTP‑GETs the game list**, runs the swarm, logs an online scorecard URL. | **Useless offline.** Replace with a notebook entrypoint that builds `Arcade(OFFLINE)`, enumerates `environment_files/`, and drives our agent. |
| `templates/llm_agents.py`, `reasoning_agent.py`, `multimodal.py`, `smolagents.py`, `langgraph_*` | LLM agents calling hosted APIs (`OpenAIClient`, `ChatOpenAI`, `OpenAIServerModel`). Show the action/observation prompt format and the click x/y schema. | **All dead offline** (no local‑weights path). Keep only as reference for the action schema and the (useful) idea of a textual frame/observation step. Delete from the submission. |
| `templates/random_agent.py` | Uniform‑random legal action; resets on `NOT_PLAYED`/`GAME_OVER`. | Our **B0 sanity baseline** and pipeline smoke test. |
| `arc_agi_3_wheels/*` | `arcengine 0.9.3`, `arc_agi 0.9.8`, `numpy 2.4.4`, `pydantic`, `pillow`, `flask`, `requests`, `matplotlib`. **No torch/tf/jax.** Linux `cp312` manylinux. | The games run **fully locally** from these — no server. ⚠️ If we need a GPU net, we must **self‑package a CUDA torch wheel**. Develop/CI on the **pinned Kaggle Linux image**, not the Windows dev box. |
| `environment_files/<id>/<id>.py` | Each public game as an `arcengine.ARCBaseGame` subclass, with **obfuscated identifiers** (random names) so the rules can't simply be read. | The agent must learn each game **by interaction**, not by reading code. These 25 are our **dev + held‑out validation** set (never a training target). |
| `environment_files/<id>/metadata.json` | `default_fps`, `tags` (`keyboard`/`click`/`keyboard_click`), **`baseline_actions`** (per‑level baseline counts; range ≈ 8 → 578). | `baseline_actions` ≈ the local stand‑in for `h` and the per‑level **search‑depth budget** (`5·h`). |

### 1.4 The engine internals that matter (✅ from `arcengine`/`arc_agi` wheels)

- **`GameAction`** enum: `RESET(0)`, `ACTION1..5` (`SimpleAction`), `ACTION6` (`ComplexAction` with `x,y ∈ [0,63]`), `ACTION7`. → **8 action types; the click is 4096‑wide.**
- **`FrameData`**: `frame: list[64×64 grid]`, `state`, `levels_completed`, `win_levels`, `available_actions: list[int]` (**the per‑frame legal‑action filter — primary pruning signal**), `action_input.reasoning` (opaque ≤16 KB JSON echoed back → usable scratch memory).
- **`ARCBaseGame`**: holds `_levels`, `_current_level_index`, `_score`, `_state`, `_available_actions`, `_seed`. Exposes `perform_action()`, `try_move()`, `next_level()`, `_get_valid_actions()`, **`_get_valid_clickable_actions()`** (restricts clicks to meaningful sprite cells — *internal, never sent over the API*), `_get_hidden_state()`.
- **`LocalEnvironmentWrapper`** (`arc_agi/local_wrapper.py`): `exec`s the game source, instantiates `self._game` **in‑process**, and `step()` calls `self._game.perform_action(...)`. **The scorecard updates only in `EnvironmentWrapper._set_last_response` (i.e., only on `wrapper.step()`).** 🧪 ← *this is the basis of the free‑simulator exploit.*
- **`OperationMode`**: `OFFLINE` (local files only, exposes `_game`), `NORMAL` (local+API), `ONLINE`/`COMPETITION` (route to a **remote** HTTP wrapper — **no `_game`, cannot run offline**). ⚠️ The submission must use **`OFFLINE`**, never `COMPETITION`.
- **`EnvironmentScoreCalculator`** (`arc_agi/scorecard.py`): the **exact RHAE** implementation (see §3/§2.0) — `min(1.15, (baseline/actions)²)`, level‑index weights, completion cap. ✅

---

## 2. Research: What Works & What Does Not

**Bottom line:** approaches that **exploit the deterministic in‑process simulator** win; approaches that try to **learn or reason** their way to the goal from scratch lose. Every verdict below follows from three verified properties.

### 2.0 The three properties that determine everything

1. **The goal is never told.** Win conditions are inferable only by observing which interactions fire `next_level()`/`win()` vs `lose()`. (e.g., `bp35`: "gravity‑bound avatar reaches the gem tile" — discoverable only behaviorally.) ✅
2. **Scoring is completion‑dominated and efficiency‑squared.** `S = min(1.15,(h/a)²)`; env score `= min(completion_cap, weighted_avg)`, `w_l = l`. 2× human actions = **25%**; 10× = **1%**. Partial competence decays multiplicatively across 6–10 sequential levels. ✅
3. **The environment is a deterministic, in‑process Python object.** The engine core has **no RNG**; of the 8 games analyzed, all are `deepcopy`‑replicable; the only two suite‑wide randomness cases are safe under `deepcopy` (`tr87` = instance‑local *seeded* `random.Random`; `lf52` = global `np.random.shuffle` once in `__init__`, frozen by a deepcopy snapshot). 🧪

### 2.1 What works

**A. Forward‑model planning, env as its own perfect simulator — THE primary engine (fit ~82).**
`copy.deepcopy(arc_env._game)`, search/plan on the copy *for free*, execute only the discovered near‑optimal path. It is the **only** paradigm that converts "the goal is never told" into a **checkable** objective (detect `levels_completed` in a rollout) and attacks **both** scoring levers at once (completion via search + near‑optimal action count via the exact model). It **generalizes to OOD mechanics by construction** (it searches the *actual* rules — nothing to overfit) and mirrors the ARC team's own internal state‑graph validation (Technical Report §3.5.2). In‑process you can call `_get_valid_clickable_actions()` to collapse the 4096‑wide click branch to a handful of sprite‑center clicks. *Per‑game confirmation:* every analyzed game is narrow‑branching but **deep** (e.g., `ls20` ≈ 4 actions/state to depth ~190) — exactly where blind search (4¹⁹⁰) is hopeless but **informed search on an exact model** is tractable.

**B. Object‑centric perception + state‑graph explore‑and‑prune — the supporting front‑end.** Connected‑component segmentation over the 64×64×16 grid (the engine *is* object‑centric: sprites with tags/layers) + canonical frame‑hash dedup + dead‑end/cycle detection. This is empirically the strongest *family* on the real benchmark (⚠️ Blind Squirrel state‑graph agent 6.71%). Substrate, not standalone engine.

**C. A small local CNN as a perception/click‑proposal aide.** 16‑channel one‑hot 64×64 → conv torso → per‑action heads + a **convolutional 64×64 coordinate head** (the StochasticGoose design that ⚠️ *won the preview at 12.58%* — the best result any approach has posted). Scores 4096 click targets cheaply with 2D inductive bias; biases exploration toward frame‑changing actions. Use as a search/click prior; it is the GPU‑resident **fallback policy** if `_game` is sealed.

**D. Go‑Explore / novelty exploration — contribute only the archive.** Return‑to‑state is exact and free via deepcopy; novelty/count/RND/frame‑change bonuses are language‑free (Core‑Knowledge‑safe) and drive goal discovery. But where deepcopy works, plain best‑first dominates it; keep the archive + novelty‑prioritized frontier, drop the RL robustification phase.

### 2.2 What does not work (and why)

**A. Pure LLM/VLM agents — wrong primary approach, structurally and empirically.**
(1) **Mechanically impossible offline** — every template hard‑imports a hosted API; no local‑weights path. (2) **Empirically dead even at the frontier** — ⚠️ frontier models <1% official. (3) **The bottleneck is strategy/exploration, not perception** — the *same* TR87 env scores ⚠️ **0% off‑the‑shelf → 97.1% with a handcrafted harness**; if perception were the wall, a harness couldn't add 97 points. The grids are machine‑trivial to ingest; the hard part is discovering an unstated compositional goal and executing a 20–500‑step *exact* plan under squared‑efficiency scoring — a search problem. LLMs are also penalized by the metric (one costly pass per action; NL reasoning isn't rewarded; trial‑and‑error wandering is exactly what `(h/a)²` punishes). **Nuance:** that 97.1% harness is env‑specific and **does not transfer** (sibling BP35 stays 0%). Real‑world linguistic priors are *actively harmful* on abstract grids — **drop VLM‑as‑policy outright.**

**B. Learned world models (DreamerV3/MuZero/EfficientZero/IRIS) trained at test time — right family, wrong problem (fit ~18).** Strictly worse than possessing the exact model for free: why approximate the transition function when `deepcopy(_game)` *is* it? Fatal flaws: sample‑hunger (10⁴–10⁸ steps vs a `5·h` single‑life budget; **no pretrain on the OOD eval**); model error is catastrophic under squared efficiency + completion cap; it burns the scarce resource (real actions) to rebuild a model you already own. **Reserve only** a learned *deterministic transition cache* for the unlikely case: partial observability **and** non‑copyable object **and** replay blocked.

**C. Model‑free deep RL at test time (PPO/DQN/Rainbow/IMPALA/R2D2) — the single worst‑matched paradigm (fit ~8).** Needs 10⁵–10⁸ steps; budget is tens‑to‑hundreds in one life, every learning step burns scored budget, and random `P(win)` ranges from ~1/355 to <1/10,000 → **no reward to bootstrap**. The metric punishes RL's core mechanism (a miracle solve exactly at `5·h` = `(1/5)² = 4%`). No pretrain on the OOD eval. *Salvage:* AlphaZero‑style policy/value heads distilled **offline** from public self‑play can seed search priors — that's value‑guided planning, not model‑free RL as the engine.

**D. Meta‑RL / in‑context RL (RL², AMAGO, RELIC, Algorithm Distillation, DPT) — perfect lexical match, disqualifying distribution mismatch (fit ~18).** Meta‑RL amortizes over a *known* training distribution; here the private set is **intentionally OOD**, you **cannot pretrain on eval mechanics**, and there is **no adequate training distribution** (25 correlated public envs vs the hundreds–thousands of diverse tasks meta‑RL needs). At most: a tiny distilled net that *orders* candidate moves fed to the real engine.

**E. Handcrafting/tuning/training on the 25 public envs — the dominant overfitting trap.** ⚠️ TR87→Duke (0→97.1%) vs BP35 (0→0%) proves env‑specific harnesses are worth 97 points where built and **zero** on a sibling. Doomed: memorized action sequences (Playback scores 0 on unseen), hardcoded tags/sprite taxonomies/layouts/click priors, DSL primitive libraries tuned to public mechanics. **The correct distinction:** *solution‑level* handcrafting fails; *engine‑level* generality transfers — public and private run the **identical engine** (same grids, 7‑action space, `try_move`/collision/`next_level`, `available_actions`, click restriction). Learn engine‑level structure; validate cold on a held‑out public split.

**F. Program/DSL synthesis of *dynamics* — largely redundant; self‑defeating standalone (fit ~34).** Inducing a symbolic transition model is redundant against the exact simulator; single‑life gives almost no examples (and no negatives) for sample‑hungry synthesis; hand‑authored DSLs miss OOD mechanics. **Keep only** symbolic **win‑predicate induction** + object/affordance abstraction as a *module feeding the planner*.

### 2.3 The one load‑bearing assumption (verify Day‑1)

The whole ranking pivots on: **does the real Kaggle harness expose the in‑process `_game` for `deepcopy`, or only a frozen `step()`?** ✅ In `Arcade.make()`, only `OFFLINE`/`NORMAL` return a `LocalEnvironmentWrapper` carrying `._game`; `COMPETITION` routes to a remote HTTP wrapper that **can't run offline at all** — so the competition *must* run via the offline local wheels it ships (strongly implying the local path is available). ⚠️ Two hardening risks: the report says the protocol was patched to reject *starting a second game client even in local mode* (targets re‑instantiation; whether in‑process `deepcopy` is also caught is the open question); and `lf52`'s unseeded global RNG means **prefer `deepcopy` over replay‑from‑RESET** and snapshot/restore `np.random.get_state()` around every rollout. **Because of this single dependency, never ship perfect‑sim as the sole engine.**

---

## 3. Algorithm Ranking

Decisive, but **conditional on the Day‑1 fact** above. The stack is designed so each engine degrades gracefully into the next.

| Rank | Paradigm | Fit | Role | One‑line justification |
|---|---|---|---|---|
| **1** | **Forward‑model tree/graph search** (deepcopy‑as‑perfect‑simulator: BFS / IDA* / best‑first / MCTS; reward = `levels_completed`) | **82** | **Primary engine** (if Day‑1 `_game` access) | Converts "goal never told" into a *checkable* objective via free exact rollouts; drives executed action counts to ≤ human baseline. |
| **2** | **Small local CNN perception** (action‑effect + 64×64 coordinate head, segmenter, frame‑diff) | **42** | Perception/pruning aide; **fallback policy** if `_game` sealed | Only paradigm with a real ARC‑AGI‑3 lead (⚠️12.58%); tames the 4096‑wide click branch. |
| **3** | **Go‑Explore + novelty/count exploration** (cell archive, return‑to‑state, RND, frame‑change reward) | **38** | Exploration substrate feeding the search | Return‑to‑state is free/exact via deepcopy; language‑free novelty drives goal discovery — but it covers states, it doesn't reason. |
| **4** | **Neuro‑symbolic object‑centric perception + state graph** (CC segmentation, canonical hash, dead‑end prune) | **38** | State‑abstraction front‑end (Blind Squirrel‑style) | Correct shared abstraction; its symbolic *planner* is redundant when deepcopy works, unreliable when it doesn't. |
| **5** | **Program/DSL synthesis & win‑predicate induction** | **34** | Symbolic goal‑induction module only | Inducing the *win condition* gives search a dense subgoal; inducing *dynamics* is dominated by the exact simulator. |
| **6** | **Learned world models** (DreamerV3 / MuZero / IRIS, test‑time) | **18** | Thin contingency fallback (only if `_game` sealed **and** replay blocked) | Right family, wrong problem — you already own the exact model for free. |
| **7** | **Meta‑RL / in‑context RL** (RL², AMAGO, RELIC, AD) | **18** | Not recommended; at most a distilled move‑ordering prior | Amortizes over a *known* distribution; the eval is intentionally OOD and pretraining on it is forbidden. |
| **8** | **Model‑free deep RL** (PPO, DQN/Rainbow, IMPALA, R2D2) | **8** | Not recommended; at most an offline‑distilled prior to seed search | Needs 10⁵–10⁸ interactions vs a ~10²–10³ single‑life budget; squared‑efficiency punishes its core exploration. |

*Pure online‑API LLM/VLM agents are excluded entirely — inert offline; a local open‑weights model is at most a perception/prior, never the decision‑maker.*

**Why #1 is #1 (fit 82, not higher):** highest ceiling, attacks both scoring levers, generalizes by construction, matches the preview evidence — but capped at 82 by four adversarially‑confirmed risks: (1) **harness access is load‑bearing and runtime‑unverified**; (2) **search horizon explodes** on weighted late levels (baselines 192–578) with sparse end‑of‑level reward → heuristics + macro‑actions + subgoals are mandatory; (3) **replay‑from‑RESET is unsafe** (`lf52` global RNG) → deepcopy‑only + RNG guard; (4) the **click‑pruning oracle is never exposed** → click branch must be tamed by simulating clicks or by the CNN (#2). **Build it first; never ship it alone.**

---

## 4. Baselines & Winning Ideas

An executable build order. Each baseline is useful alone **and** becomes a reusable component of the winning stack. Build in order; each gates the next.

> **B‑(−1) — The Simulability Probe (Day‑1, before any planner code).** The stack forks on one fact: does the harness hand you the live `_game` (OFFLINE) or a frozen/remote `step()` (COMPETITION → HTTP, can't run offline)? Write a ~30‑line probe: `g2 = copy.deepcopy(env._game); g2.perform_action(a); assert env._game frame unchanged; step g2 twice and compare frames; verify scorecard delta == 0`. Run on all 25 public envs. `_game` exposed + faithful → B2/B3 are primary. Sealed → B1 + CNN is primary. **Also** snapshot/restore `np.random.get_state()`/`random.getstate()` around every rollout (`lf52` diverges on replay; deepcopy is safe). *Do not write planner code until this resolves.*

### 4.1 Baselines (build immediately, in order)

**B0 — Random sanity agent.** *Tests:* end‑to‑end plumbing — load each env via `Arcade.make(..., OFFLINE)`, drive the loop, read `FrameData`, compute the local RHAE score, produce a scorecard. Calibration floor (random solves a non‑tutorial level <1/10,000).
```
loop until done or counter > budget:
    a = sample_uniform(frame.available_actions)   # for ACTION6 sample x,y in 0..63
    frame = env.step(a)
score = local_RHAE(scorecard)
```
*Critical fix:* raise `MAX_ACTIONS=80` → per‑level `5·h` from `metadata.baseline_actions`. *Expected RHAE:* ≈0%; confirms scorecard isolation (simulated steps → delta 0).

**B1 — Frame‑change / novelty‑greedy explorer.** *Tests:* how far pure interaction gets you; builds the **perception + click‑pruning substrate** every later baseline reuses (the morally‑equivalent reimplementation of the ⚠️ preview leaders). **Mandatory fallback** if the probe fails.
```
archive = {}                                  # canonical frame-hash -> visit count
loop:
    cands = enumerate_candidate_actions(frame)         # object-centric, see below
    score each by: frame changed? + novelty(result hash) + did levels_completed tick?
    pick highest-scoring untried candidate (novelty tie-break)
    if no-op (frame unchanged) -> blacklist in this state
    if levels_completed increased -> lock in the macro that caused it
```
- **Candidate enumeration is the key primitive.** Never sample 4096 raw clicks. For `ACTION6`, segment the frame into objects (connected components over 64×64×16) and propose **one click per object centroid** — reconstructing what `_get_valid_clickable_actions` does internally (never exposed). Collapses click branching to a handful on `vc33/bp35/lf52/tn36/cd82`.
- **No‑op detection by frame equality** is mandatory for keyboard games with static/uninformative `available_actions` (`ls20`=[1,2,3,4], `re86`=[1..5], `tr87`=[1,2,3,4]): a move into a wall is wasted and won't be pruned for you.
- *Expected RHAE:* low‑single to low‑double digits; solves tutorials + shallow click puzzles, **stalls on deep composed (high‑weight) levels** — the ceiling of "explore better than random."

**B2 — Forward‑model BFS/IDDFS planner (deepcopy/replay).** *Tests:* the core thesis — the free perfect simulator turns "unknown goal" into a *checkable* one and "1% efficiency penalty" into near‑optimal counts.
```
def plan_level(real_game):
    root = copy.deepcopy(real_game)            # free, unscored
    target = root.levels_completed
    seen = {canonical_hash(root)}
    for depth_limit in itertools.count(1):     # IDDFS bounds memory on deep keyboard games
        stack/queue = [(root, [])]
        while frontier:
            g, path = pop()
            for a in enumerate_candidate_actions(g.frame):   # B1's pruned set
                with rng_snapshot(): c = deepcopy(g); c.perform_action(a)
                if c.levels_completed > target: return path + [a]   # WIN detected in sim
                h = canonical_hash(c)
                if h not in seen and len(path) < depth_limit:
                    seen.add(h); push((c, path+[a]))
execute_only(plan_level(real_game))            # the ONLY scored actions
```
- Canonical state hash (64×64 grid + level index + visible budget) for cycle detection (essential on `re86`/`ls20`). **Works as‑is** on narrow‑branch/moderate‑depth games (`cd82`, `vc33`, `tn36`, `bp35`) → near‑cap RHAE per completed level. **Provably fails** on `ls20` L6 (depth ~192, b=4 → 4¹⁹²) and `re86` L7 (depth ~424): sparse reward gives blind search nothing to climb → **why B3 exists.**
- *Expected RHAE:* **the big jump** — solves early/mid levels + the narrow‑branch click family at/near the 115% cap; completion cap banks most of each env's score; stalls on deep levels until B3.

**B3 — Object‑centric state‑graph + best‑first search with learned win‑condition detection.** *Tests:* defeating the depth wall + sparse reward. Combines B1's perception, B2's free simulator, and a **heuristic guide** so search is goal‑directed (mirrors ARC's §3.5.2).
```
def plan_level_bestfirst(real_game):
    root = deepcopy(real_game); pq = PQ([(0, root, [])]); seen = {hash(root)}
    while pq:
        _, g, path = pq.pop()
        for a in enumerate_candidate_actions(g.frame):
            with rng_snapshot(): c = deepcopy(g); c.perform_action(a)
            if c.levels_completed > root.levels_completed: return path+[a]
            h = canonical_hash(c)
            if h in seen: continue
            seen.add(h)
            f = (len(path)+1) + heuristic(c)      # inferred-goal distance
            pq.push(f, (c, path+[a]))
```
- **Heuristic = inferred win‑condition distance**, derived per‑frame, *not* hardcoded. The recurring transferable schema across all 8 analyses is **"make a controlled object/canvas match a visible target":**
  - *Pixel/silhouette match* (`re86`, `cd82`, `tr87`): mismatching goal cells between the assembled/painted region and the target template → dense monotone signal → per‑piece/region A*.
  - *Object‑pose match* (`tn36`, `ls20`): sum of attribute mismatches (shape+color+rotation; or (x,y,scale,rotation,color) 5‑tuple) to the ghost target.
  - *Routing/reach* (`bp35`, `vc33`, `lf52`): Manhattan/reachability distance of agent/blocks to goal cell(s) under the (possibly flipped/gravity) physics.
- **Learned win‑condition detector:** a tiny classifier / frame‑diff rule miner that recognizes *which on‑screen object is the target* and *what predicate flips `next_level()`*, by correlating frame features with `levels_completed` ticks across simulated branches → auto‑selects the heuristic and provides a dense surrogate reward.
- **Subgoal decomposition** to tame depth: `ls20` per‑pad (route through needed cyclers → stand on pad); `re86` per‑piece A* to its slot; `tr87` per‑tape‑slot shortest cyclic presses + cursor TSP. Search each subgoal to ≤30 depth instead of one ~400‑depth search.
- *Expected RHAE:* **the win condition** — deep high‑weight levels become tractable, completion cap → 1.0, efficiency → cap. Credible path to 100% on simulable envs.

### 4.2 Winning ideas (innovations that turn baselines into a 100% agent)

1. **The perfect‑simulator planner (primary engine).** Treat `_game` as its own oracle; run all exploration/search on `deepcopy`s (free), execute only the discovered near‑optimal path. The single highest‑ceiling idea and the only one that converts ARC‑AGI‑3's defining obstacle into a detectable objective. **Load‑bearing & conditional** on the Day‑1 probe; keep B1 + a learned model as the documented fallback.
2. **General win‑condition inference (not hardcoding).** Never read the obfuscated source. Infer the goal from the only honest signals — `levels_completed`/`WIN` in rollouts + frame‑diff. Detect which of {pixel‑match, pose‑match, reach‑match} the env is, then plug the matching heuristic. Generalizes to the OOD private set because it searches actual rules.
3. **Object‑centric abstraction.** Segment every frame into objects/relations; plan in object space, not pixels. Collapses branching, supplies the canonical hash for cycle detection, and encodes the only priors the games use (objectness/geometry/agentness) → transfers across unseen mechanics.
4. **Click‑target pruning.** Reconstruct the never‑exposed `_get_valid_clickable_actions`: propose clicks only at object centroids / interactive sprite cells; validate on the simulator (a no‑op click is pruned). Turns the 4096‑wide branch into a handful per state — what makes B2/B3 affordable in 9 h.
5. **Action‑effect CNN (GPU heuristic + fallback engine).** 16‑ch one‑hot 64×64 → conv → per‑action heads + a convolutional 64×64 coordinate head. (a) **Search guide:** order candidate actions/clicks fed to B3's frontier. (b) **Primary fallback:** if the probe fails, this *is* the engine (⚠️ led the preview at 12.58%). Train **online per env** (no pretrain‑freeze on public → no overfit). Self‑package a CUDA torch wheel (none in the offline wheels). **No VLM** — language/real‑world priors mislead on abstract grids.
6. **Level‑to‑level transfer.** Later levels *compose* earlier mechanics, and `completion_cap` weights them most. **Cache** the win‑condition model, object taxonomy, action semantics, and macro‑actions learned on early levels and reapply to later ones — converting partial completion into full completion (the dominant lever). Under partial observability (`ls20` L7 fog), plan on the full deterministic simulator state, not the rendered frame.

**Net build order:** B‑(−1) probe → B0 plumbing → B1 perception+novelty (+fallback) → B2 deepcopy BFS/IDDFS (big jump) → B3 best‑first + win‑inference + subgoals (the 100% path). Ideas 1–6 are the components B2/B3 are assembled from; their generality is what survives the OOD private set.

---

## 5. Execution Plan (roadmap to ~100%)

**Fork‑structured.** Phase 1 is a Day‑1 go/no‑go gate; everything downstream assumes its outcome. Guiding principles in every phase: **completion > efficiency**; **explore on copies, execute on the scored instance**; **generality is mandatory — validate on a held‑out public split.**

### Phase 0 — Offline harness & test rig (Days 0–5)
1. **Build the offline venv** from `arc_agi_3_wheels/` with `--no-index` in clean Python 3.12; pin the **exact Kaggle Linux image** (wheels are Linux `cp312`) and verify there. Confirm **zero** network/DL imports.
2. **Instantiate envs in `OFFLINE` mode, never `COMPETITION`** (the latter routes to remote HTTP, no `_game`). Load `environment_files/<id>/<id>.py` + `metadata.json`.
3. **Raise `MAX_ACTIONS`** → per‑level `5·h` from `baseline_actions` (mirrors the server termination rule); make it a config knob.
4. **Build the scoring/replay rig** — re‑implement RHAE (cross‑checked vs `EnvironmentScoreCalculator`): per‑level `min(1.15,(h/a)²)`, level‑index weights, completion cap, env score, total = mean. Input = JSONL trajectory; output = per‑level/env/total. The regression oracle for every later phase.
5. **Wire the Playback agent** as a sanity check: record a hand‑solved env, replay, confirm ~100% (validates the rig against "human replay = 100%").
6. **Freeze a held‑out generalization split now** — 18 dev / 7 frozen‑holdout by mechanic family (keyboard/click/keyboard_click; pattern‑match/maze/physics/grammar). The holdout is never inspected or tuned against — the only honest OOD proxy.

*Exit:* all 25 load+step offline; rig reproduces Playback 100%; holdout frozen; `5·h` enforced. *Deliverable:* `harness/`, `rig/`, `splits.json`.

### Phase 1 — DAY‑1 CRITICAL EXPERIMENT: verify the perfect simulator (Day 1, parallel with Phase 0)
Run on all 25 public games (and re‑run on the real Kaggle harness the moment available):
1. **Scorecard isolation** — `deepcopy(_game)`, N `perform_action` on the copy; assert original `_action_count` and the wrapper scorecard are unchanged. *Pass = simulated steps are free.*
2. **Deepcopy determinism** — two deepcopies, same action sequence → byte‑identical `frame`/`state`/`levels_completed`.
3. **Replay‑from‑RESET determinism — and where it diverges** — fresh instances, replay a prefix, compare. **Expected to fail on `lf52`** (global `np.random.shuffle`) → proves **deepcopy is the only robust primitive**; document per game.
4. **RNG/IO audit + global‑state guard** — grep every env for `random`, `np.random`, `time`, `datetime`, `open`, `socket`, `os.`; implement+test snapshot/restore of `np.random.get_state()`/`random.getstate()` around every rollout. Assume the OOD private set has *more* hazards; make the guard unconditional.

**Decision tree:** **GREEN** (`_game` reachable + free + deterministic, expected) → forward‑model search is primary; proceed to Phases 2–4 (the ~100% path). **YELLOW** (reachable but stochastic on some envs) → deepcopy‑only + mandatory RNG guard; flag stochastic envs for the learned fallback. **RED** (frozen/remote `step()`, no `_game`) → pivot primary engine to a learned‑deterministic world model + CNN proposer; re‑scope immediately.

*Surface, don't bury:* ⚠️ the report says the protocol rejects *starting a second game client even in local mode* — `deepcopy` of the live object is a different mechanism and may survive, **but this is the one genuinely open question; re‑test on the real image.** *Exit:* GREEN/YELLOW/RED locked with per‑game evidence; RNG guard done. *Deliverable:* `probe_results.md`, `sim/forward_model.py`.

### Phase 2 — Explorer: goal discovery & state‑graph builder (Weeks 2–4)
1. **Object‑centric perception** (CC/flood‑fill segmentation → objects, agent token, adjacency/containment; Core‑Knowledge only). A feature layer, not a rule inducer.
2. **Directed state‑graph explore‑and‑prune** (Blind Squirrel + ARC §3.5.2): canonical frame hash dedup; expand frontier on the deepcopy model; **re‑query `_get_valid_actions()`/`_get_valid_clickable_actions()` every step** to collapse clicks.
3. **Dense reward oracle from the simulator** — `levels_completed`/`WIN` detectable in rollouts converts "infer the goal" → "detect the goal"; prioritize the frontier with frame‑change/RND novelty.
4. **No‑op/dead‑end detection** via frame‑equality; handle multi‑tick animation stalls.

*Validate on the held‑out 7.* *Exit:* explorer reaches `WIN` on ≥1 level of every dev game + ≥1 holdout via interaction only. *Deliverable:* `explorer/`.

### Phase 3 — Planner: search‑then‑execute for near‑optimal action counts (Weeks 4–7)
1. **Engine selection by horizon:** short levels → bounded BFS/IDDFS; deep levels (`baseline ≥ ~100`) → best‑first/A*/IDA*; wide‑branch click games → MCTS(UCT).
2. **Heuristics from perception, not hardcoding** — frame‑distance to the discovered goal, object‑to‑target distance, satisfied‑subgoal count; expressed over abstract objects/relations to transfer OOD.
3. **Subgoal decomposition & macro‑actions** to tame depth‑~190 search; reuse a solved early‑level strategy on structurally‑identical later levels (the dominant scoring lever).
4. **Compute‑bounded search** — hard wall‑clock per (env,level); on exhaustion, execute the best partial path (partial completion still moves the cap).
5. **Execute minimally** — run only the found plan on the scored env, respecting `5·h`; verify executed count ≤ `h` where possible (RHAE → cap).

*Exit:* ≥80% of dev‑game *levels* completed end‑to‑end at ≤ baseline counts; measurable completion on holdout. *Deliverable:* `planner/`.

### Phase 4 — Perception & world‑model fallback hardening (Weeks 6–9, overlaps Phase 3)
1. **Local CNN action‑effect/click proposer** (StochasticGoose‑style) — search prior/click ranker only, trains online per game from scratch.
2. **Learned deterministic world model as RED fallback** — a tabular/transition‑cached deterministic model (cheap; dynamics are discrete+deterministic) before any Dreamer/MuZero stack; behind a Phase‑1 feature flag.
3. **Package a GPU stack offline only if needed** — no torch/tf/jax in the wheels; if the CNN/WM is on the critical path, self‑package a CUDA torch wheel for the RTX 6000 and verify it imports offline. If GREEN, prefer a pure‑numpy proposer and **skip the GPU dependency** to cut packaging risk.
4. **Drop all VLM ideas.**

*Exit:* RED fallback runs end‑to‑end on ≥1 env; CNN proposer beats random click selection. *Deliverable:* `fallback/`, conditional torch recipe.

### Phase 5 — Generalization hardening (Weeks 9–12)
1. **Holdout‑only gate** — promote no change that improves dev‑18 but not holdout‑7; track a generalization‑gap metric.
2. **Hunt & delete env‑specific constants** — grep for any public tag string, layout, sprite name, magic baseline → replace with frame/object‑derived computation.
3. **Stress‑test the RNG guard + deepcopy on adversarial synthetic envs** (per‑step RNG, time seeding, non‑deepcopyable handles) → guard fails loudly + fallback engages, never silent corruption.
4. **Mechanic‑family ablation** — solve each family (pattern‑reconstruction, maze/attribute, physics/gravity, sliding‑block, grammar/tape, platformer) without per‑family code.

*Exit:* holdout‑7 completion within a small stable gap of dev‑18; zero hardcoded public constants in audit. *Deliverable:* `generalization_report.md`.

### Phase 6 — Compute‑budget management (9 h / single RTX 6000) (continuous; locked Weeks 11–13)
The binding constraint is **wall‑clock search time** (CPU‑bound; GPU mostly idle unless the CNN is active).
1. **Global time governor** ≈ `9h / (N_envs × avg_levels)` → few‑hundred‑ms‑to‑few‑seconds per level; hard deadline returns best partial.
2. **Make state copy cheap** — profile `deepcopy` (64×64 sprite arrays aren't free); prefer incremental save/restore of the engine's mutable state over a full deepcopy in hot loops (there is **no engine‑level undo** — `ACTION7` is just a 7th simple action); hash to avoid re‑expansion.
3. **Allocate time by score leverage** — bias toward completing *later* levels (a failed later level zeroes subsequent weight).
4. **Parallelize across envs** within the single‑GPU/CPU budget; cap per‑env time so one pathological env can't starve the rest.
5. **Checkpoint** per‑env best trajectories so a near‑timeout run still submits best‑so‑far.

*Exit:* full dry‑run over 25 public envs completes within margin. *Deliverable:* `budget/`, timing report.

### Phase 7 — Submission packaging & open‑source (Weeks 12–13, then maintenance)
1. **Offline notebook** — install `--no-index` from bundled wheels (+ conditional torch); confirm **no** network call anywhere (delete the `llm_agents` family from the path).
2. **Self‑contained harness** — build the env in OFFLINE/local mode and reach `_game` (GREEN) or fall back (RED); don't depend on the grader handing a live object unless re‑verified on the real image.
3. **Determinism** — seed all global RNG at startup; same submission → same score.
4. **License** — **CC0 or MIT‑0** (prize requirement); include full source, build instructions, wheel manifest.
5. **End‑to‑end rehearsal** on the public set through the actual notebook before each milestone.

*Exit:* notebook runs offline start‑to‑finish under budget, scores via the rig, license present. *Deliverable:* `submission.ipynb`, `LICENSE`, `BUILD.md`.

### Risks & mitigations

| # | Risk | L | I | Mitigation |
|---|---|---|---|---|
| R1 | Grader hides `_game` (frozen/remote `step()`) — kills perfect‑sim | Med | Crit | Phase‑1 RED fork pre‑built (cached WM + CNN behind a flag); re‑verify on the real image ASAP. |
| R2 | Organizers patch second‑instance/deepcopy between versions | Med | Crit | Use deepcopy‑of‑live‑object (≠ re‑instantiation); keep RED fallback live; track wheel releases (solution is open‑source → assume the exploit can be neutralized). |
| R3 | Hidden/global RNG in OOD envs breaks deepcopy/replay | Med | High | Unconditional `np.random`/`random` save‑restore guard; deepcopy‑only; adversarial synthetic tests. |
| R4 | Search horizon explosion (baselines→578, 4096 clicks) | High | High | Subgoal decomposition, macro‑actions, learned heuristics/click proposer, time governor → best‑partial. |
| R5 | Public‑overfit fails to transfer (private OOD) | High | High | Holdout‑7 gate, constant‑hunt audit, mechanic‑family ablation; engine‑level generality only. |
| R6 | 9 h compute overrun | Med | High | Time governor, incremental save/restore instead of full deepcopy, per‑env caps, checkpointed best‑so‑far. |
| R7 | Offline DL packaging (no torch in wheels) fails on Kaggle | Med | Med | Prefer pure‑numpy proposer when GREEN; else pin CUDA wheel to the exact image and rehearse import offline. |
| R8 | Goal discovery stalls on a novel mechanic | Med | High | Novelty‑guided explorer + dense `levels_completed` oracle; partial completion still scores. |
| R9 | Linux/Windows / image drift (wheels are Linux cp312) | Low | Med | Develop/CI on the pinned Kaggle Linux image, not the Windows dev box. |

### Milestone timeline (aligned to Jun 30 / Sep 30 2026)

| Date | Milestone | Target state |
|---|---|---|
| **T0 + Day 1** | **Phase 1 gate** | GREEN/YELLOW/RED locked; offline harness booting. |
| +1 wk | Phase 0 done | Rig reproduces Playback 100%; budget loop + holdout split done. |
| +4 wks | Phase 2 done | Explorer wins ≥1 level on every dev game + ≥1 holdout via interaction only. |
| **Jun 30 2026 — Milestone #1** | Phase 3 partial | Search‑then‑execute completing a solid majority of dev‑game levels; competitive partial‑completion on the private set; notebook runs offline under budget. **Bank Milestone #1 with a working general search agent.** |
| +9 wks | Phase 4–5 | Fallback live; holdout gap small/stable; public‑overfit audited out. |
| **Sep 30 2026 — Milestone #2** | Phase 5–6 done | Full stack (explorer + planner + fallback) generalizing across families, finishing the private set in 9 h/RTX6000. **Maximize completion‑weighted score; push the 100% bar.** |
| Post‑Sep 30 | Phase 7 maint. | Re‑verify against new wheel releases; tune efficiency toward the cap; finalize open‑source. |

**Critical‑path summary:** Day‑1 Phase 1 decides everything. GREEN → Jun 30 is a search‑agent banking exercise, Sep 30 is a generalization + completion push toward 100%. RED → re‑scope to the learned‑world‑model fallback on Day 1 and treat Milestone #1 as its proving ground.

---

## Appendix A — The 25 public environments

Columns: id · fps · control tag · ~levels (= # baseline entries). Concept/win known only for the 8 reverse‑engineered (✅).

| id | fps | tags | ~lvls | Notes (✅ = analyzed) |
|---|---|---|---|---|
| ar25 | 6 | keyboard_click | 8 | |
| **bp35** | 20 | keyboard_click | 9 | ✅ gravity platformer; win = avatar reaches gem tile; ⚠️ Opus 0% with & without harness (hard) |
| **cd82** | 30 | keyboard_click | 6 | ✅ smallest; geometry/pattern reconstruction by directional paint; win = canvas matches target (off‑diagonal) |
| cn04 | 5 | keyboard_click | 6 | |
| dc22 | 15 | keyboard_click | 6 | |
| ft09 | 8 | (none) | 6 | |
| g50t | 30 | keyboard | 7 | |
| ka59 | 10 | keyboard_click | 7 | largest file (831 KB) |
| **lf52** | 30 | click | 10 | ✅ sokoban/sliding‑block match; **unseeded global `np.random` in `__init__`** → deepcopy‑only |
| lp85 | 20 | click | 8 | |
| **ls20** | 30 | keyboard | 7 | ✅ canonical (report Fig2/3); object‑attribute matching; L1 P(win)=1/355; L7 fog; depth ~190 |
| m0r0 | 7 | keyboard_click | 6 | |
| r11l | 30 | click | 6 | |
| **re86** | 25 | keyboard_click | 8 | ✅ polyomino assembly; win = pixel‑match to goal; report Fig4 RHAE example; L7 depth ~424 |
| s5i5 | 15 | click | 8 | |
| sb26 | 30 | keyboard_click | 8 | |
| sc25 | 20 | keyboard_click | 6 | |
| sk48 | 30 | keyboard_click | 8 | |
| sp80 | 10 | keyboard_click | 6 | |
| su15 | 20 | click | 9 | |
| **tn36** | 3 | click | 7 | ✅ object transform (rigid+scale+recolor) to match ghost; clicks = program‑selector buttons + object cells |
| **tr87** | 10 | keyboard | 6 | ✅ tape/grammar rewrite; per‑level seeded RNG (safe); ⚠️ Opus 0%→97.1% with handcrafted "Duke" harness |
| tu93 | 30 | keyboard_click | 9 | |
| **vc33** | 20 | click | 7 | ✅ physics (per‑level gravity vector); win = `next_level()` predicate; report: L6 ~50 actions vs L1 <5 |
| wa30 | 5 | keyboard | 9 | |

## Appendix B — Sources

- [ARC Prize 2026 — ARC‑AGI‑3 (Kaggle)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) · [ARC‑AGI‑3 competition page](https://arcprize.org/competitions/2026/arc-agi-3)
- [ARC‑AGI‑3 Technical Report (PDF, Apr 2026)](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf) — design, RHAE formula, dataset composition, frontier scores
- [ARC‑AGI‑3 Preview: 30‑Day Learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings) — preview agent results (StochasticGoose 12.58%, Blind Squirrel 6.71%)
- Source: bundled `arcengine 0.9.3` / `arc_agi 0.9.8` wheels + `ARC-AGI-3-Agents/` (this repo)

---

*Generated with [Claude Code](https://claude.com/claude-code). Confidence tags: ✅ source/report‑verified · 🧪 confirm Day‑1 · ⚠️ external/not source‑verifiable. The strategy is sound but **load‑bearing on the Day‑1 simulability probe** — run it before committing engineering.*
