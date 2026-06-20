# ARC‑AGI‑3 — Strategy & Algorithm Comparison and Ranking

**Team:** SJSU CMPE 295A, Group 2 · **Date:** 2026‑06‑19
**Competition:** [ARC Prize 2026 — ARC‑AGI‑3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) (Kaggle, offline)

> **Purpose.** Four strategy documents were produced for this competition: three external "ranked algorithm list" plans (`plan_1.md`, `plan_2.md`, `plan_3.md`) and the source‑grounded `ARC_AGI_3_MASTER_STRATEGY.md`. This document (1) compares the four head‑to‑head against a fixed rubric derived from the competition's actual mechanics, (2) **ranks the four plans**, and (3) merges every distinct algorithm they propose into **one unified master ranking** scored by fitness to ARC‑AGI‑3.

---

## 0. TL;DR

- **Best plan: `ARC_AGI_3_MASTER_STRATEGY.md`** — it is the only document that reads the shipped engine source and discovers the decisive move (the in‑process game is its own free, exact simulator via `deepcopy(_game)`), and the only one that correctly treats the offline rule, the squared‑efficiency metric, and OOD generalization as hard design constraints rather than footnotes.
- **`plan_1.md` is the strongest of the three external lists** — a realistic, preview‑evidence‑grounded portfolio that ranks the right substrate (object‑centric state‑graph + exploration) first and keeps LLMs in their place.
- **`plan_2.md` is the most exhaustive** (26 algorithms, deep tactics) but slightly over‑engineered and leans on offline‑hostile coding‑agent loops; its citations look unreliable.
- **`plan_3.md` is the weakest** — only 5 algorithms, and its top two are LLM‑driven, which is exactly what ARC‑AGI‑3 was built to defeat and what the offline rule disables.

**Plan ranking:** `MASTER_STRATEGY` ≫ `plan_1` > `plan_2` ≫ `plan_3`.

**The single most important idea any of the plans contains** is in the Master Strategy alone: *because the games ship as deterministic in‑process Python objects, you can clone the live game and search it for free — turning "the goal is never told" into "the goal is detectable in a rollout."* No external plan finds this, and it is worth more than every other technique combined.

---

## 1. The rubric — what ARC‑AGI‑3 actually rewards

Any algorithm or plan should be judged against the five facts that dictate the competition. These are the scoring axes used throughout this document.

| # | Constraint | Consequence for an algorithm |
|---|---|---|
| **C1 — Offline** | No internet at inference; every bundled LLM template hard‑imports a hosted API. | LLM‑as‑policy is **dead weight**. A local open‑weights model is at most a prior, never the decision‑maker. |
| **C2 — RHAE scoring** | Per level `S = min(1.15, (h/a)²)`, gated by a completion cap weighted toward later levels. | **Completion dominates; efficiency is squared and cannot be banked.** Trial‑and‑error wandering is punished quadratically. |
| **C3 — Goal never stated** | Only Core‑Knowledge priors; win conditions discovered by interaction. | Need a mechanism that **detects** `levels_completed`/`WIN`, not one that assumes a reward. |
| **C4 — Non‑LLM leads, frontier LLMs <1%** | Humans 100%; bottleneck is strategy/exploration, not perception. | Perception‑heavy or reasoning‑heavy LLM stacks are **mis‑aimed**. Search/planning is the lever. |
| **C5 — Eval set is OOD** | 25 public / 55 semi‑private / **55 fully‑private (scored)**. | Overfitting to public games is **harmful**. Only **engine‑level** generality transfers. |
| **C6 — Engine is a deterministic in‑process object** | `arcengine` core has no RNG; games are `deepcopy`‑replicable. | The environment **can be its own perfect simulator** — the highest‑leverage exploit available. |

**The eight evaluation criteria** (used to score plans in §3 and algorithms in §4), each 0–10:

1. **Offline‑awareness** (C1) 2. **Scoring‑awareness** (C2) 3. **Goal‑discovery mechanism** (C3) 4. **Anti‑LLM realism** (C4) 5. **Generalization / OOD discipline** (C5) 6. **Engine exploitation** (C6) 7. **Implementation realism / actionability** 8. **Evidence grounding**.

---

## 2. Side‑by‑side comparison of the four plans

| Dimension | **MASTER_STRATEGY** | **plan_1** | **plan_2** | **plan_3** |
|---|---|---|---|---|
| **Format** | Full strategy doc: ranking + execution plan + risk register + appendices | 25 ranked algos + recommended stack + build phases | 26 ranked algos w/ tactical detail + recommended stack | 5 ranked algos in a table |
| **#1 pick** | Forward‑model tree/graph search (deepcopy = perfect simulator) | Object‑Centric State Graph Planner | Hybrid Executable World Model + Planner | LLM‑Driven Program Synthesis |
| **Core thesis** | The env is its own free, exact simulator; search it, execute only the optimal path | Build object state graph, explore, plan; portfolio of solvers | Build & verify an executable Python world model, plan inside it (MPC) | Use offline LLMs to synthesize code / guide MCTS |
| **Finds the deepcopy exploit?** | **Yes — the entire spine** | No (treats model as something to *learn*) | Partially (MPC re‑validation is adjacent, but rebuilds the model) | No |
| **Offline rule handled?** | Explicitly: deletes LLM templates, flags import errors, OFFLINE mode | Mentioned; LLM kept as router only | Mentioned; LLM as hypothesis generator only | Weak — top 2 picks are LLM‑centric |
| **RHAE / efficiency** | Exact formula re‑implemented; completion‑dominance is the spine | Trace compression + action efficiency noted | Action‑cost‑aware planning + trace compression | Not addressed |
| **OOD / anti‑overfit** | Held‑out split, constant‑hunt audit, engine‑level‑only rule | Noted; warns synthetic≠private | Self‑play tooling + meta‑prior + overfit warning | "Code generalizes" claim only |
| **Evidence grounding** | Source‑verified wheels + Tech Report + preview, with ✅/🧪/⚠️ tags | Preview results (StochasticGoose, Blind Squirrel) | arXiv citations (**IDs look fabricated/unverifiable**) | Generic, unsourced |
| **Risk analysis** | 9‑row risk register + Day‑1 go/no‑go gate + fallback forks | Light (per‑algo "main risk" column) | Per‑algo "main risk" notes | None |
| **Biggest strength** | Decisive insight + rigor + falsifiable Day‑1 gate | Realistic, well‑prioritized, evidence‑aligned | Breadth and tactical depth | Concise; program‑synthesis‑as‑generalizer is a fair point |
| **Biggest weakness** | Load‑bearing on one runtime‑unverified assumption (mitigated by the probe + fallbacks) | Misses the free‑simulator exploit | Over‑engineered; offline‑hostile coding loops; shaky citations | Misaligned with offline + anti‑LLM nature; shallow coverage |

### What each plan gets uniquely right

- **MASTER_STRATEGY** — Only doc to (a) reverse‑engineer the wheels and prove the engine is deepcopy‑replicable, (b) catch the `MAX_ACTIONS=80` footgun, (c) catch the `lf52` global‑RNG hazard that makes replay‑from‑RESET unsafe, (d) reconstruct the never‑exposed `_get_valid_clickable_actions` to tame the 4096‑wide click branch, and (e) structure the whole project around a single falsifiable Day‑1 probe.
- **plan_1** — Correctly identifies that the *empirically proven* approaches on the real benchmark (state‑graph agents, action‑effect CNN) belong at the top, and keeps the LLM strictly as a hypothesis router. Its phased build order is clean and shippable.
- **plan_2** — The richest toolbox: active/Bayesian probing, causal intervention testing, visual delta compiler, MPC with model repair, constraint/SAT solving, risk‑aware and action‑cost‑aware planning. Many of these are genuinely useful *modules* the Master Strategy under‑specifies.
- **plan_3** — Names program synthesis as "the ultimate generalizer," which is directionally true for ARC's abstraction heritage — but bets the top of the list on it being LLM‑driven, which the offline rule breaks.

### What each plan gets wrong

- **MASTER_STRATEGY** — Everything pivots on the deepcopy assumption holding on the *real* grader (not just the shipped wheels). It says so and pre‑builds the RED fallback, but the risk is real.
- **plan_1** — Frames the core as "learn or infer a transition model" (#3 Model‑Based Test‑Time Planning) without realizing the exact model is *handed to you for free*. It would burn scored actions rebuilding what `deepcopy` already gives.
- **plan_2** — Its #1 (executable world model) and its "coding‑agent rule repair" loop assume a capable code‑generation model in the loop, which is slow and offline‑fragile; building/verifying a Python simulator is strictly dominated by cloning the real one. The fabricated‑looking arXiv IDs (`2605.05138`, `2605.09650`, `2603.24621`) undermine trust in the evidence base.
- **plan_3** — Ranks four of five paradigms (LLM program synthesis, LLM‑MCTS, DreamerV3 world models, in‑context meta‑RL) that are either disabled offline or are known poor‑fits (sample‑hunger, OOD distribution mismatch). It never mentions the metric, the offline rule's effect on LLMs, or the engine.

---

## 3. Ranking the four plans

Scored on the eight criteria from §1 (0–10 each; **Engine exploitation** and **Scoring‑awareness** are weighted ×1.5 because they are the highest‑leverage axes for this specific competition).

| Criterion (weight) | MASTER | plan_1 | plan_2 | plan_3 |
|---|:--:|:--:|:--:|:--:|
| Offline‑awareness ×1.0 | 10 | 7 | 7 | 3 |
| Scoring‑awareness ×1.5 | 10 | 7 | 8 | 2 |
| Goal‑discovery ×1.0 | 10 | 7 | 8 | 4 |
| Anti‑LLM realism ×1.0 | 10 | 8 | 7 | 2 |
| Generalization / OOD ×1.0 | 10 | 7 | 7 | 4 |
| Engine exploitation ×1.5 | 10 | 3 | 4 | 2 |
| Implementation realism ×1.0 | 9 | 8 | 7 | 4 |
| Evidence grounding ×1.0 | 10 | 6 | 5 | 3 |
| **Weighted total / 105** | **99** | **66** | **68.5** | **30** |
| **Normalized / 100** | **94** | **63** | **65** | **29** |

> Note: on the raw weighted total `plan_2` (68.5) edges `plan_1` (66) on breadth/efficiency‑awareness, but the gap is inside the noise and `plan_1` is the safer, more *aligned* bet for an offline, anti‑LLM, OOD competition. The final ordering below weights alignment and shippability over raw coverage.

### Final plan ranking

**🥇 1. `ARC_AGI_3_MASTER_STRATEGY.md` — 94/100.**
The only plan grounded in the actual engine, and the only one that finds the decisive exploit. It correctly orders the paradigms, handles all six competition constraints, and de‑risks itself with a falsifiable Day‑1 gate and pre‑built fallbacks. *Use this as the backbone.*

**🥈 2. `plan_1.md` — 63/100.**
The best external list. Right instincts (state graph + exploration on top, LLM as router only), preview‑evidence‑aligned, and a clean phased build. Its blind spot is the free simulator — fold the Master Strategy's deepcopy spine into it and it becomes very strong.

**🥉 3. `plan_2.md` — 65/100 (ranked 3rd on alignment).**
The most thorough and the best *parts bin* — mine it for the probing, causal‑testing, delta‑compiler, MPC‑repair, and constraint‑solver modules. Demoted below `plan_1` because its headline approach (build+verify an executable world model with a coding agent) is offline‑fragile and dominated by the free simulator, and its citations are untrustworthy.

**4. `plan_3.md` — 29/100.**
Concise but fundamentally mis‑aimed: it bets on LLM‑driven and sample‑hungry world‑model/meta‑RL methods that the offline rule disables and the metric punishes. Useful only as a reminder that program‑synthesis *of the win condition* (not LLM‑driven, not of dynamics) is worth keeping as a module.

---

## 4. Unified master ranking of all algorithms

Every distinct algorithm proposed across the four documents, deduplicated into paradigm families and scored by **fitness to ARC‑AGI‑3** (the §1 rubric). "Fit" is a 0–100 composite; "Role" is how it should actually be used in a winning stack. Source column shows which plan(s) proposed it (**M**=Master, **1/2/3**=plans).

### Tier S — the decisive engine

| Fit | Algorithm | Role | Source |
|:--:|---|---|:--:|
| **82** | **Forward‑model tree/graph search** — `deepcopy(_game)` as a free exact simulator; BFS/IDA*/best‑first/MCTS with reward = `levels_completed` | **Primary engine** (if Day‑1 `_game` access). Converts the unstated goal into a checkable objective and drives executed action counts to ≤ human baseline. | **M** |

> This single family is the difference between the Master Strategy and every other plan. It attacks *both* scoring levers (completion via search, near‑optimal counts via the exact model) and generalizes by construction (it searches the actual rules — nothing to overfit).

### Tier A — strong, transferable substrate (build these regardless)

| Fit | Algorithm | Role | Source |
|:--:|---|---|:--:|
| **42** | **Small local CNN** (action‑effect heads + 64×64 conv coordinate head) | Perception/click‑proposal aide; **fallback policy** if `_game` is sealed. Only approach with a real ARC‑AGI‑3 lead (preview 12.58%). | M, 1 |
| **40** | **Object‑centric state graph** (CC segmentation, canonical hash, `state→action→next_state` edges, dead‑end/loop prune) | The shared front‑end every method reuses; supplies the cycle‑detection hash. Empirically the strongest *family* on the real benchmark (Blind Squirrel 6.71%). | M, 1, 2 |
| **38** | **Systematic exploration + novelty / Go‑Explore** (return‑to‑state, RND, frame‑change bonus) | Exploration substrate that drives goal discovery; language‑free, so Core‑Knowledge‑safe. Return‑to‑state is free/exact via deepcopy. | M, 1, 2 |
| **36** | **Win‑condition / goal inference** (correlate frame features with `levels_completed` ticks; classify pixel‑match vs pose‑match vs reach‑match) | The heuristic that makes deep search goal‑directed; *the* generalizing idea. Inferring the *goal* is gold; inferring *dynamics* is redundant against the free sim. | M, 1, 2 |
| **35** | **Click‑target pruning** (clicks only at object centroids / interactive cells; validate on sim) | Reconstructs the never‑exposed `_get_valid_clickable_actions`; collapses the 4096‑wide branch to a handful — what makes search affordable in 9 h. | M, 1, 2 |
| **34** | **Hierarchical macros / skills + level‑to‑level transfer** (`move_to`, `push_to`, `toggle_then_enter`; cache & reuse across levels) | Tames depth‑~190 search and converts partial completion into full completion (the dominant scoring lever, since later levels weigh most). | M, 1, 2 |
| **32** | **Trace compression / plan minimization** (delete chunks, shorten paths, replace loops, replay‑verify) | Pure efficiency squeeze on a found solution → directly improves the squared `(h/a)²` term. Cheap, high‑value, always run last. | 1, 2 |

### Tier B — useful modules (conditional or supporting)

| Fit | Algorithm | Role | Source |
|:--:|---|---|:--:|
| **34** | **Program / DSL synthesis** — *win‑predicate induction* only | Keep the symbolic *goal* inducer (dense subgoal); drop symbolic *dynamics* (dominated by the exact sim). | M, 1, 2, 3 |
| **30** | **Model Predictive Control** (plan in sim, execute short prefix, re‑validate, repair) | The right *discipline* when the model is approximate; collapses to "just execute the plan" when the model is exact (deepcopy). | 2 |
| **30** | **Executable world model + verifier** (build/verify a Python simulator from transitions) | **Strong only if deepcopy is unavailable.** Otherwise strictly dominated — why approximate the transition function when you own it for free? | 1, 2 |
| **29** | **Active probing / Bayesian experiment design / causal intervention** (one‑variable‑at‑a‑time, max‑info probes) | Excellent for *cheap* mechanic discovery before search; squared efficiency rewards minimizing wasted probes. | 2 |
| **28** | **Visual delta compiler** (cell/object diffs → semantic events) | Clean input layer feeding every higher method; low‑risk infrastructure. | 2 |
| **28** | **Affordance discovery** (passable/solid, collectible, hazard, clickable, movable tables) | Bootstraps perception priors; must be *verified by interaction*, never hardcoded (OOD risk). | 2 |
| **27** | **Classical pathfinding** (A*/BFS/Dijkstra on extracted walls/agent/goal) | Fast, reliable solver for the navigation subproblems that recur inside many levels. | 1, 2 |
| **27** | **Portfolio meta‑controller / algorithm router** (detect game type, route to solver, budget per solver) | Robustness across diverse private games; run cheap solvers first, escalate. Good engineering hygiene, not an engine. | 1, 2 |
| **26** | **Constraint / SAT‑SMT planning** (alignment, matching, filling, Sokoban) | Specialized power solver for the puzzle‑constraint subclass; bounded‑horizon. | 2 |
| **26** | **Value / heuristic model** (small net ranks (state,action) toward milestones) | Search heuristic only (Blind Squirrel‑style), never the full policy; needs per‑game data. | 1, 2 |
| **25** | **Risk‑aware + action‑cost‑aware planning** (avoid GAME_OVER, penalize long plans) | Directly serves the metric; bolt onto the planner's cost function. | 2 |
| **24** | **Memory across levels** (carry action/object/goal semantics forward) | Subsumed by "level‑to‑level transfer"; essential for the completion cap. | 1, 2 |

### Tier C — weak fit / contingency only

| Fit | Algorithm | Role | Source |
|:--:|---|---|:--:|
| **18** | **Learned world models trained at test time** (DreamerV3 / MuZero / IRIS) | Right family, wrong problem — you already own the exact model. Thin RED‑fork fallback only. | 1, 2, 3 |
| **18** | **Workspace optimization / multi‑agent blackboard** ("scientist" roles, shared lab notebook) | Heavy engineering; its value (hypothesis routing) needs an offline LLM to shine. Optional orchestration layer, not an engine. | 1, 2 |
| **18** | **Meta‑RL / in‑context RL** (Decision Transformer, RL², AMAGO, AD) | Amortizes over a *known* distribution; the eval is intentionally OOD and pretraining on it is forbidden. At most a distilled move‑ordering prior. | 2, 3 |
| **16** | **Local LLM as hypothesis router** (compressed traces → game‑type / probe suggestions) | Non‑policy aide only; every output must be code‑verified. Slow and hallucination‑prone. | 1, 2, 3 |
| **15** | **Synthetic game pretraining / learned meta‑prior** (train priors on generated ARC‑like games) | Speeds exploration *if* the synthetic distribution overlaps private mechanics — a big if. Prior only, then verify. | 1, 2 |

### Tier D — poor fit / avoid as primary

| Fit | Algorithm | Role | Source |
|:--:|---|---|:--:|
| **12** | **LLM‑driven program synthesis as the engine** | Concept is sound (code generalizes), but offline + slow + LLM‑weak‑on‑ARC. Demote the *LLM* part; keep symbolic win‑predicate synthesis (Tier B). | 3 |
| **10** | **Active Inference / discrete world models as primary** | Same sample‑hunger trap as learned world models; matching human "what if I move this?" is exactly what the *free simulator* already does, better. | 3 |
| **8** | **Model‑free deep RL** (PPO/DQN/Rainbow/IMPALA) | Needs 10⁵–10⁸ interactions vs a ~10²–10³ single‑life budget; squared efficiency punishes its core mechanism. Offline‑distilled prior at most. | 1, 2 |
| **6** | **Pure LLM ReAct / frontier agent as policy** | Mechanically dead offline; empirically <1% even at the frontier. Excluded as a decision‑maker. | 1, 2, 3 |
| **3** | **Random / evolutionary action search** | Calibration floor only (random solves a non‑tutorial level <1/10,000). | 1 |

---

## 5. Verdict & recommended synthesis

**Adopt `ARC_AGI_3_MASTER_STRATEGY.md` as the backbone**, then graft in the best modules from the external plans:

1. **Engine (from Master):** forward‑model search on `deepcopy(_game)` — gated by the Day‑1 simulability probe, with the CNN + learned‑deterministic‑model RED fallback pre‑built.
2. **Substrate (consensus across all plans):** object‑centric state graph + canonical hashing + novelty exploration + click pruning + win‑condition inference. Every plan ranks this family high; build it first and reuse everywhere.
3. **Efficiency layer (from plan_1 / plan_2):** trace compression / plan minimization and action‑cost‑aware planning — cheap, directly amplifies the squared `(h/a)²` term.
4. **Discovery modules (from plan_2):** active/Bayesian probing, causal intervention testing, and the visual delta compiler — the best‑specified parts of any external plan; they make the cheap exploration phase information‑dense before the expensive search.
5. **Orchestration (from plan_1 / plan_2):** a portfolio meta‑controller that routes per‑game and budgets the 9 h, with memory/skill transfer across levels to bank the completion cap.
6. **Keep LLMs out of the action loop entirely** (all plans except plan_3 agree). At most, a local model as an offline, code‑verified hypothesis router — never the policy.

**The one‑line synthesis:** *The Master Strategy supplies the winning engine and the rigor; plan_1 supplies the realistic portfolio scaffolding; plan_2 supplies the richest module toolbox; plan_3 supplies a cautionary example of what the offline, anti‑LLM, squared‑efficiency, OOD design of ARC‑AGI‑3 specifically defeats.*

---

*Generated with [Claude Code](https://claude.com/claude-code). Fit scores and plan scores are this analysis's composite judgments against the §1 rubric; the deepcopy‑simulator advantage that separates the Master Strategy is load‑bearing on the Day‑1 probe and should be verified on the real grader before committing engineering.*
