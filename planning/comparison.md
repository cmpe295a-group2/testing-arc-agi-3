# ARC‑AGI‑3 — Strategy & Algorithm Comparison and Ranking

**Team:** SJSU CMPE 295A, Group 2 · **Date:** 2026‑06‑19
**Competition:** [ARC Prize 2026 — ARC‑AGI‑3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) (Kaggle, offline)

> **Purpose.** Four strategy documents were produced for this competition: two external "ranked algorithm list" plans (`plan_1.md`, `plan_2.md`), the team's own literature‑review + discussion note (`research.md`), and the source‑grounded `master_plan.md`. This document (1) compares the four head‑to‑head against a fixed rubric derived from the competition's actual mechanics, (2) **ranks the four plans**, (3) puts `research` and the Master Plan in a focused duel, and (4) merges every distinct algorithm into **one unified master ranking** scored by fitness to ARC‑AGI‑3.

---

## 0. TL;DR

- **Best plan: `master_plan.md`** — the only document that reads the shipped engine source and finds the decisive move (the in‑process game is its own free, exact simulator via `deepcopy(_game)`), and the only one that treats the offline rule, the squared‑efficiency metric, and OOD generalization as hard design constraints.
- **`plan_1.md`** is the strongest *self‑contained* external list — a realistic, preview‑grounded portfolio that ranks the right substrate (object‑centric state‑graph + exploration) first and keeps LLMs in their place.
- **`plan_2.md`** is the most exhaustive (26 algorithms) but over‑engineered, leans on offline‑hostile coding‑agent loops, and its citations look unreliable.
- **`research.md`** is the best‑*researched* — the only doc bringing the real ARC‑AGI‑3 preview leaderboard and the ARC‑AGI‑2 winning lineage — but it is an earlier‑stage scoping note that misses the deepcopy exploit. **It is also the most valuable document to *merge* into the Master Plan**, because it is the best blueprint for the branch where the Master Plan is weakest.

**Plan ranking:** `master_plan` (94) ≫ `plan_1` (63) ≈ `plan_2` (65) ≈ `research` (60).

**The single most important idea any plan contains** is in the Master Plan alone: *because the games ship as deterministic in‑process Python objects, you can clone the live game and search it for free — turning "the goal is never told" into "the goal is detectable in a rollout."* No external plan finds this, and it is worth more than every other technique combined.

---

## 1. The rubric — what ARC‑AGI‑3 actually rewards

| # | Constraint | Consequence for an algorithm |
|---|---|---|
| **C1 — Offline** | No internet at inference; bundled LLM templates hard‑import hosted APIs. | LLM‑as‑policy is **dead weight**. A local model is at most a prior, never the decision‑maker. |
| **C2 — RHAE scoring** | Per level `S = min(1.15,(h/a)²)`, completion‑capped, later levels weigh more. | **Completion dominates; efficiency is squared and cannot be banked.** Trial‑and‑error is punished quadratically. |
| **C3 — Goal never stated** | Only Core‑Knowledge priors; win conditions discovered by interaction. | Need a mechanism that **detects** `levels_completed`/`WIN`, not one that assumes a reward. |
| **C4 — Non‑LLM leads, frontier LLMs <1%** | Humans 100%; bottleneck is strategy/exploration, not perception. | Perception/reasoning‑heavy LLM stacks are mis‑aimed. Search/planning is the lever. |
| **C5 — Eval set is OOD** | 25 public / 55 semi‑private / **55 fully‑private (scored)**. | Overfitting public games is **harmful**. Only **engine‑level** generality transfers. |
| **C6 — Deterministic in‑process object** | `arcengine` core has no RNG; games are `deepcopy`‑replicable. | The environment **can be its own perfect simulator** — the highest‑leverage exploit available. |

**The eight evaluation criteria** (0–10 each; **Engine exploitation** and **Scoring‑awareness** weighted ×1.5): Offline‑awareness · Scoring‑awareness · Goal‑discovery · Anti‑LLM realism · Generalization/OOD · Engine exploitation · Implementation realism · Evidence grounding.

---

## 2. Side‑by‑side comparison of the four plans

| Dimension | **master_plan** | **plan_1** | **plan_2** | **research** |
|---|---|---|---|---|
| **Format** | Full strategy: ranking + execution plan + risk register | 25 ranked algos + build phases | 26 ranked algos + deep tactics | Literature review + 4‑prototype scoping note |
| **#1 pick** | Forward‑model search (deepcopy = perfect sim) | Object‑centric state‑graph planner | Executable world model + planner | Refinement loop (executable/learned world models + synthetic meta‑training) |
| **Finds the deepcopy exploit?** | **Yes — the spine** | No | Partial (MPC re‑validation) | No |
| **Offline rule handled?** | Explicit: deletes LLM templates, OFFLINE mode | LLM as router only | LLM as hypothesis gen only | "Small‑model + offline" (NVARC template), but proven wins are API‑dependent |
| **RHAE / efficiency** | Exact formula; completion‑dominance is the spine | Trace compression noted | Action‑cost‑aware + trace compression | Cites RHAE % / action counts; no engagement with the squared term |
| **OOD / anti‑overfit** | Held‑out split + constant‑hunt audit | Warns synthetic≠private | Self‑play tooling + meta‑prior | Aware (cites StochasticGoose collapse) but synthetic‑meta‑training bet is OOD‑risky |
| **Evidence grounding** | Source‑verified wheels + ✅/🧪/⚠️ tags | Preview results cited | arXiv IDs look fabricated | **Best landscape grounding**: real leaderboard + ARC‑AGI‑2 lineage |
| **Risk analysis** | 9‑row register + Day‑1 go/no‑go gate | Light | Per‑algo notes | None (scoping plan) |
| **Biggest strength** | Decisive insight + rigor + falsifiable gate | Realistic, well‑prioritized | Breadth and tactical depth | Research depth + best RED‑path blueprint |
| **Biggest weakness** | Load‑bearing on one runtime‑unverified assumption (mitigated) | Misses the free‑simulator exploit | Over‑engineered; offline‑hostile loops; shaky citations | Earliest‑stage; misses the exploit; light on the exact metric |

---

## 3. Ranking the four plans

Scored on the eight criteria (0–10; **Engine exploitation** and **Scoring‑awareness** ×1.5).

| Criterion (weight) | master_plan | plan_1 | plan_2 | research |
|---|:--:|:--:|:--:|:--:|
| Offline‑awareness ×1.0 | 10 | 7 | 7 | 8 |
| Scoring‑awareness ×1.5 | 10 | 7 | 8 | 5 |
| Goal‑discovery ×1.0 | 10 | 7 | 8 | 7 |
| Anti‑LLM realism ×1.0 | 10 | 8 | 7 | 6 |
| Generalization / OOD ×1.0 | 10 | 7 | 7 | 6 |
| Engine exploitation ×1.5 | 10 | 3 | 4 | 2 |
| Implementation realism ×1.0 | 9 | 8 | 7 | 6 |
| Evidence grounding ×1.0 | 10 | 6 | 5 | 9 |
| **Normalized / 100** | **94** | **63** | **65** | **60** |

### Final plan ranking

**🥇 1. `master_plan.md` — 94/100.** The only plan grounded in the actual engine and the only one that finds the decisive exploit. Correctly orders the paradigms, handles all six constraints, and de‑risks itself with a falsifiable Day‑1 gate and pre‑built fallbacks. *The backbone.*

**🥈 2. `plan_1.md` — 63/100.** The best self‑contained external list: right instincts (state graph + exploration on top, LLM as router only), preview‑aligned, clean phased build. Blind spot: the free simulator. Fold the Master Plan's deepcopy spine into it and it becomes very strong.

**🥉 3. `plan_2.md` — 65/100 (ranked 3rd on alignment).** The most thorough and the best *parts bin* — mine it for probing, causal testing, delta compiler, MPC repair, constraint solving. Demoted below `plan_1` because its headline (build+verify an executable world model with a coding agent) is offline‑fragile, dominated by the free simulator, and its citations are untrustworthy.

**4. `research.md` — 60/100.** The best‑researched document, but an earlier‑stage scoping note that misses the exploit and barely engages the exact scoring metric — hence 4th as a *self‑contained strategy*. **Yet it is the #1 document to merge:** its executable‑world‑model + falsification program is the best available blueprint for the RED branch where the Master Plan is thin. See §4.

---

## 4. Head‑to‑head: `research` vs the Master Plan

`research` is different in kind from `plan_1`/`plan_2` — a literature‑grounded research note, not a ranked list — so it earns a focused duel.

**Headline finding: they are not rivals — they are the two halves of the same fork.** The Master Plan's whole strategy pivots on one Day‑1 question: *does the grader hand us the live in‑process game object to `deepcopy`?*

- **🟢 GREEN — deepcopy works** → **the Master Plan's forward‑model search dominates.** The env is its own free, exact simulator; a perfect model you own beats any model you have to learn or write.
- **🔴 RED — deepcopy sealed** → you must build the model the hard way, and **`research` is the most developed blueprint for exactly that**: executable Python world models, falsification/MDL refactoring, synthetic‑environment meta‑training, test‑time training, Z3 verifiers.

The Master Plan's own RED fallback was thin ("learned deterministic transition cache + CNN"). `research` fills it out into a genuine research program. **Merge them.**

### Criterion scorecard (master_plan vs research)

| Criterion | master_plan | research |
|---|:--:|:--:|
| Offline‑awareness | 10 | 8 |
| Scoring‑awareness | 10 | 5 |
| Goal / rule induction | 10 | 7 |
| Anti‑LLM realism | 10 | 6 |
| Generalization / OOD | 10 | 6 |
| Engine exploitation | 10 | 2 |
| Implementation realism | 9 | 6 |
| Evidence grounding | 10 | 9 |

The one place `research` nearly ties is **evidence grounding**: the Master Plan is grounded deepest (the engine source), but `research` owns the broadest *competitive‑landscape* intel — it is the only document that knows who is winning the real benchmark, with what, and at what action cost.

### Two datapoints `research` contributes — and what they prove

- **🔴 StochasticGoose: 12.58% → 0.25%.** The preview's #1 learned‑exploration CNN collapsed to roughly frontier‑LLM level at full launch. *Proves:* learned exploration **overfits and does not generalize** to the private OOD set — a direct vindication of the Master Plan's "search the actual rules, nothing to overfit," and a warning against `research`'s own synthetic‑meta‑training bet.
- **🟢 SingularityNET executable world model: 15/25 solved, 58.12% mean RHAE.** The strongest publicly reported ARC‑AGI‑3 result — a world‑model refinement loop, no deepcopy exploit. *Proves:* the "honest path" can go far, so `research`'s direction is credible — **but** it ran on a hosted frontier API (not Kaggle‑eligible) and is public‑only. Where deepcopy works, the Master Plan still dominates it; where it doesn't, this is the bar to beat offline.

### What to mine from `research`

- **Executable world model + falsification loop** — the formal RED‑branch engine the Master Plan lacked.
- **Z3 / neuro‑symbolic verifier** — proves an induced win‑rule against every past frame; hardens the Master Plan's win‑condition inference.
- **Conv‑autoencoder / quadtree state compression** — cheaper canonical hashing for the state graph under the 9 h budget.
- **ARC‑AGI‑2 transfer lessons** — MDL/simplicity bias, generate‑and‑verify, evolutionary program synthesis (SOAR), tiny recursive nets (TRM) as fast value/policy heads.
- *Hold at arm's length:* synthetic‑environment meta‑training and adversarial game generation — intellectually strong, but OOD‑risky (ARC‑AGI‑3 has no example pairs, which `research` itself notes) and off the critical path for an offline entry.

---

## 5. Unified master ranking of all algorithms

Every distinct algorithm across the four documents, deduplicated into paradigm families and scored by **fitness to ARC‑AGI‑3**. "Role" is how it should be used in a winning stack. Source: **M**=master_plan · **1**=plan_1 · **2**=plan_2 · **R**=research.

### Tier S — the decisive engine

| Fit | Algorithm | Role | Source |
|:--:|---|---|:--:|
| **82** | **Forward‑model tree/graph search** — `deepcopy(_game)` as a free exact simulator; BFS/IDA*/best‑first/MCTS, reward = `levels_completed` | **Primary engine** (if Day‑1 `_game` access). Converts the unstated goal into a checkable objective; drives executed action counts to ≤ human baseline. | M |

### Tier A — strong, transferable substrate (build regardless)

| Fit | Algorithm | Role | Source |
|:--:|---|---|:--:|
| **42** | **Small local CNN** (action‑effect heads + 64×64 conv coordinate head) | Perception/click aide; **fallback policy** if `_game` sealed. Only approach with a real ARC‑AGI‑3 lead (preview 12.58%). | M,1,R |
| **40** | **Object‑centric state graph** (CC segmentation, canonical hash, dead‑end prune) | The shared front‑end every method reuses. Strongest *family* on the real benchmark (Blind Squirrel 6.71%). | M,1,2,R |
| **38** | **Systematic exploration + novelty / Go‑Explore** | Exploration substrate driving goal discovery; language‑free. Return‑to‑state is free/exact via deepcopy. | M,1,2,R |
| **36** | **Win‑condition / goal inference** (classify pixel/pose/reach from `levels_completed` ticks) | The heuristic that makes deep search goal‑directed — *the* generalizing idea. | M,1,2,R |
| **35** | **Click‑target pruning** (clicks only at object centroids; validate on sim) | Reconstructs the hidden `_get_valid_clickable_actions`; collapses the 4096‑wide branch. | M,1,2,R |
| **34** | **Hierarchical macros / skills + level‑to‑level transfer** | Tames deep search and converts partial → full completion (the dominant scoring lever). | M,1,2,R |
| **32** | **Trace compression / plan minimization** (replay‑verify) | Pure efficiency squeeze → directly improves the squared `(h/a)²` term. Cheap; run last. | 1,2 |

### Tier B — useful modules (conditional or supporting)

| Fit | Algorithm | Role | Source |
|:--:|---|---|:--:|
| **34** | **Program / DSL synthesis** — *win‑predicate induction only* | Keep the symbolic goal inducer; drop symbolic dynamics (dominated by the exact sim). | M,1,2,R |
| **30** | **Model Predictive Control** (plan in sim, execute prefix, re‑validate, repair) | Right discipline when the model is approximate; collapses to "just execute" when the model is exact. | 2,R |
| **30** | **Executable world model + verifier** | **Strong only if deepcopy is unavailable.** Otherwise strictly dominated. The formal RED‑branch engine. | 1,2,R |
| **29** | **Active probing / Bayesian experiment design / causal testing** | Cheap mechanic discovery before search; squared efficiency rewards minimizing wasted probes. | 2,R |
| **28** | **Visual delta compiler** (cell/object diffs → semantic events) | Clean input layer feeding every higher method; low‑risk infrastructure. | 2 |
| **28** | **Affordance discovery** (passable/hazard/clickable/movable tables) | Bootstraps perception priors; must be *verified by interaction*, never hardcoded. | 2,R |
| **27** | **Classical pathfinding** (A*/BFS/Dijkstra on extracted walls/agent/goal) | Fast, reliable solver for the navigation subproblems that recur inside many levels. | 1,2,R |
| **27** | **Portfolio meta‑controller / router** (detect game type, budget per solver) | Robustness across diverse private games; run cheap solvers first, escalate. | 1,2 |
| **26** | **Constraint / SAT‑SMT planning** (alignment, matching, Sokoban; Z3) | Specialized power solver for the puzzle‑constraint subclass; also verifies induced win‑rules. | 2,R |
| **26** | **Value / heuristic model** (small net / ResNet18 / TRM ranks (state,action)) | Search heuristic only (Blind Squirrel‑style), never the full policy. | 1,2,R |
| **25** | **Risk‑aware + action‑cost‑aware planning** (avoid GAME_OVER, penalize long plans) | Directly serves the metric; bolt onto the planner's cost function. | 2 |
| **24** | **Memory across levels** (carry action/object/goal semantics forward) | Subsumed by level‑to‑level transfer; essential for the completion cap. | 1,2,R |

### Tier C — weak fit / contingency only

| Fit | Algorithm | Role | Source |
|:--:|---|---|:--:|
| **18** | **Learned world models at test time** (DreamerV3 / MuZero / IRIS) | Right family, wrong problem — you already own the exact model. Thin RED‑fork fallback only. | 1,2 |
| **18** | **Workspace optimization / multi‑agent blackboard** (Arcgentica / DreamTeam‑style) | Heavy engineering; its value needs an offline LLM to shine. Optional orchestration layer, not an engine. | 1,2,R |
| **18** | **Meta‑RL / in‑context RL** (Decision Transformer, RL², AMAGO, AD) | Amortizes over a *known* distribution; the eval is intentionally OOD. Distilled move‑ordering prior at most. | 2,R |
| **16** | **Local LLM as hypothesis router** | Non‑policy aide only; every output code‑verified. Slow, hallucination‑prone. | 1,2,R |
| **15** | **Synthetic game pretraining / learned meta‑prior** (incl. adversarial game generation) | Speeds exploration *if* the synthetic distribution overlaps private mechanics — a big if. Prior only, then verify. | 1,2,R |

### Tier D — poor fit / avoid as primary

| Fit | Algorithm | Role | Source |
|:--:|---|---|:--:|
| **8** | **Model‑free deep RL** (PPO/DQN/Rainbow/IMPALA) | Needs 10⁵–10⁸ interactions vs a ~10²–10³ single‑life budget; squared efficiency punishes its core mechanism. | 1,2 |
| **6** | **Pure LLM ReAct / frontier agent as policy** | Mechanically dead offline; empirically <1% even at the frontier. Excluded as a decision‑maker. | 1,2,R |
| **3** | **Random / evolutionary action search** | Calibration floor only (random solves a non‑tutorial level <1/10,000). | 1,R |

---

## 6. Verdict & recommended synthesis

**Adopt `master_plan.md` as the backbone**, then graft in the best modules from the external plans:

1. **Engine (Master):** forward‑model search on `deepcopy(_game)` — gated by the Day‑1 simulability probe.
2. **GREEN →** forward‑model search is the primary engine; `research`'s Z3 verifier and state‑compression ideas become supporting modules.
3. **RED →** promote `research`'s executable‑world‑model + falsification + offline‑training program to the primary engine; it is far more developed than the Master Plan's original RED stub.
4. **Substrate (all plans agree):** object‑centric state graph + canonical hashing + novelty exploration + click pruning + win‑condition inference. Build first; reuse everywhere.
5. **Efficiency layer (plan_1 / plan_2):** trace compression and action‑cost‑aware planning — cheap, amplifies the squared `(h/a)²` term.
6. **Discovery modules (plan_2 / research):** active/Bayesian probing, causal intervention testing, visual delta compiler, Z3 win‑rule verification.
7. **Keep LLMs out of the action loop entirely** — at most a local, code‑verified hypothesis router, never the policy.
8. **Hold at arm's length:** synthetic‑environment meta‑training and adversarial game generation — OOD‑risky and off the critical path.

**One‑line synthesis:** The Master Plan supplies the winning engine, the rigor, and the Day‑1 gate; `plan_1` supplies the realistic portfolio scaffolding; `plan_2` supplies the richest module toolbox; `research` supplies the competitive‑landscape evidence and the best blueprint for the branch the Master Plan can't yet fill. Use the Master Plan to decide *which* engine to build on Day 1, and use `research` to build the one it couldn't.

---

*Generated with [Claude Code](https://claude.com/claude-code). Fit scores and plan scores are this analysis's composite judgments against the §1 rubric; the deepcopy‑simulator advantage that separates the Master Plan is load‑bearing on the Day‑1 probe and should be verified on the real grader before committing engineering.*
