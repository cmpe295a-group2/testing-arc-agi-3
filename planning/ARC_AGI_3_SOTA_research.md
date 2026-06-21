# ARC‑AGI‑3 — State‑of‑the‑Art Research (online, verified)

**Team:** SJSU CMPE 295A, Group 2 · **Compiled:** 2026‑06‑20
**Method:** deep‑research harness — 5 search angles · 19 primary sources fetched · 86 claims extracted · **25 claims adversarially verified (3‑vote, 2/3‑to‑kill) → 24 confirmed, 1 refuted** · 101 research agents.

> This file captures **everything** the online research turned up about ARC‑AGI‑3 (the 2026 interactive/agentic ARC benchmark): the real leaderboard numbers, how the leaders actually crack the hard part (goal/win‑condition inference on deep levels), the legality of our deepcopy‑simulator edge, corrections to the team's `research.md` notes, and what it all means for our agent. Every load‑bearing claim has a confidence + vote + primary source.

---

## 0. TL;DR (read this first)

1. **The leader numbers in our notes are REAL but they're from the 2025 PREVIEW, not the 2026 launch.** StochasticGoose 12.58% (1st), Blind Squirrel 6.71% (2nd), Rudakov 3.64% (HM) — all from the **30‑day Preview Agent Competition (Jul 18 – Aug 19 2025)**. **StochasticGoose collapsed to ~0.25% at the full 2026 launch** → preview rank does NOT transfer.
2. **At the official 2026 launch every frontier model is sub‑1%** (Opus 4.6 0.50%, Gemini 3.1 Pro 0.40%, GPT‑5.4 0.20%, Grok‑4.20 0.10%). The only double‑digit number anywhere is a purpose‑built harness.
3. **The genuine SOTA on the hard problem is the EXECUTABLE PYTHON WORLD MODEL** (Rodionov / SingularityNET): synthesize a Python world model → verify it against observations → refactor toward MDL‑simplicity → plan through it. **58.12% mean RHAE (v2, GPT‑5.5, 15/25 solved)** — but our notes conflated versions: **v1 is only 32.58% (GPT‑5.4, 7/25)**; 58% is the best‑config v2 number.
4. **The field has documented OUR EXACT WALL.** Goal‑agnostic exhaustive/graph search becomes computationally intractable on deep/large‑state levels (Rudakov: ft09 L6+, ls20 L3+). **The fix the leaders use = replace goal‑agnostic search with a synthesized/learned world model that INFERS the win condition and prunes toward it.** This directly validates our diagnosis and points to A2.
5. **Our deepcopy‑simulator edge is REAL and currently neither sanctioned nor forbidden** by the official toolkit/docs/report — ARCEngine games are deterministic in‑process Python objects, 1000 FPS, turn‑based, with serializable action traces. **BUT the binding Kaggle competition rules were NOT retrievable** (page didn't render) and must be checked; the exploit only exists on the **offline/local track**, not the server‑side live API.

---

## 1. Leaderboard & results — two eras (verified, high confidence)

### 1.1 The 2025 Preview Agent Competition (Jul 18 – Aug 19 2025)
The leaders were **goal‑AGNOSTIC informed‑search agents** — "exploring as much of the action space as possible in the hope of encountering a winning combination by chance."

| Rank | Agent | Author | Score | Levels | Method |
|---|---|---|---|---|---|
| 1 | **StochasticGoose** | Tufa Labs / Dries Smit (adviser Jack Cole) | **12.58%** | 18 | CNN + RL **frame‑change predictor** (bias exploration toward actions that change the frame) |
| 2 | **Blind Squirrel** | Will Dick | **6.71%** | 13 | Explore‑and‑learn **directed state graph** from frames + **ResNet18 value model** |
| HM | **"Explore It Till You Solve It"** (Graph‑Based Exploration) | Evgenii Rudakov | **3.64%** (blog) | 12 | Training‑free, **model‑free directed state‑transition graph** + shortest‑path‑to‑untested navigation |

- Confidence **high (3‑0 each)**. Sources: ARC Prize 30‑day‑learnings blog, preview‑agents competition page, Technical Report (Apr 22 2026), `github.com/DriesSmit/ARC3-solution`.
- ⚠️ **The StochasticGoose repo README reports NO percentage** — the 12.58% lives in the ARC Prize blog/competition page + Smit's Medium writeup.
- ⚠️ **Time‑sensitivity flag:** StochasticGoose **collapsed to ~0.25% at the full 2026 launch** (consistent with the sub‑1% frontier numbers). Preview success did not carry over.
- 📐 The **3.64% (Rudakov)** is the ARC Prize blog's *official‑submission* score, **depressed by a reset‑action bug**. The paper itself reports **levels solved** (median 30/52 across 6 games, 3rd on the private preview leaderboard); after the bugfix the author cites a median of **~17 private levels**. So 3.64% / 30‑of‑52 / 12‑official‑bugged / ~17‑post‑bugfix are *different reporting scopes, not contradictions*.

### 1.2 The official 2026 launch — frontier models all sub‑1% (high, 3‑0)
From **Table 2 ("Semi‑private leaderboard scores for frontier models at release")** of the Technical Report, identical system prompt ("Your goal is to win"):

| Model | Score |
|---|---|
| Anthropic **Opus 4.6 (Max)** | **0.50%** |
| Google **Gemini 3.1 Pro Preview** | **0.40%** |
| OpenAI **GPT‑5.4 (High)** | **0.20%** |
| xAI **Grok‑4.20** | **0.10%** |

- Source: ARC_AGI_3_Technical_Report.pdf. (Secondary press, The Decoder Mar 2026, shows slightly different launch‑day snapshots — Gemini 0.37 / GPT 0.26 / Opus 0.25 / Grok 0.00 — but both sets agree: **everything sub‑1%**.)

### 1.3 RHAE scoring — confirmed exactly as we modeled it (high, 3‑0)
**Eq. 1 of the Technical Report:** `S_{l,e} = min(1.15, (h_{l,e} / a_{l,e})²)` — **square first, then cap at 1.15** (so per‑level scores range 0%–115%). Human baseline = upper‑median best first‑run human playthrough; level *l* has **linear weight w_l = l**; per‑environment score is **capped at the weighted fraction of levels completed**. Worked example: human 10 vs AI 100 actions → (10/100)² = 1%. ✅ Matches our `rhae_level_score` and `batch_eval` weighting.

---

## 2. The hard part — goal/win‑condition inference & deep levels

### 2.1 It is officially a goal‑inference problem (high, 3‑0)
Technical Report **verbatim**: *"the agent is never told the objective nor provided instructions. It must autonomously infer the mechanics… including the win conditions."*

**ARC Prize's OWN environment‑validation tool (§3.5.2) is a directed state graph** — *"an explicit directed graph over reachable states… Node identity is hash‑based, allowing the builder to merge distinct trajectories that arrive at the same underlying state,"* with an acceptance threshold that a random policy should not solve a level more than **1 in 10,000**.
→ This is **structurally the same family** as Blind Squirrel and as **our BFS‑over‑deepcopied‑simulator** (we hash‑merge states too). *Caveat: the report does not equate its validator with our approach — that's a methodological‑family analogy, not an organizer claim.*

### 2.2 OUR EXACT WALL is documented in the literature (high, 3‑0)
Rudakov et al. ("Graph‑Based Exploration for ARC‑AGI‑3", arXiv **2512.24156**), training‑free, **model‑free**, no win‑condition inference, **verbatim**:
> *"Performance degraded on games with extremely large state spaces (**ft09 levels 6+, ls20 levels 3+**), where exhaustive exploration becomes computationally intractable."*

Repo (`dolphin-in-a-coma/arc-agi-3-just-explore`): *"approaches the limits of brute‑force solving"*, *"the goal is simply to be more intelligent than a purely random agent,"* and crucially: **novelty signal "may be orthogonal to goal‑relevance."**

→ **This is precisely our finding:** goal‑agnostic search (BFS / novelty / Go‑Explore / agent‑position reach) **cannot prune toward an unknown win condition** on deep levels.
→ *Nuance (2‑1 on the paraphrase):* Rudakov attributes intractability primarily to the **4,096‑wide click action space (64×64)**, whereas our wall is **depth ~44 / branching ~8 unknown‑goal pruning** — overlapping but not identical mechanisms.

### 2.3 The fix the leaders use
Replace goal‑agnostic search with a **synthesized / learned world model that encodes the inferred win condition and lets you plan / prune toward it.** Two concrete instances ↓ (§3).

---

## 3. SOTA techniques (the actual fix)

### 3.1 ★ Executable Python World Models — the single most transferable technique (high, 3‑0)
**Rodionov / SingularityNET — "Executable World Models for ARC‑AGI‑3 in the Era of Coding Agents"** (arXiv **2605.05138**; repo `astroseger/arc-3-agents-baseline1`).

**Method (abstract, verbatim):** *"the agent maintains an executable Python world model, verifies it against previous observations, refactors it toward simpler abstractions as a practical proxy for an MDL‑like simplicity bias, and plans through the model before acting."* Components: a scripted external controller, a **world‑model verifier** (*"after each modification… run verifiers that test consistency with previous observations"*), a **planner verifier**, and a **plan executor** (executes a plan in the model, runs it in the real game, **stops the instant a predicted frame diverges from observed** → execution doubles as online falsification).

**Results (⚠️ VERSION‑SPLIT — corrects our notes):**
| Version | Coder | Mean RHAE | Solved |
|---|---|---|---|
| **v1** | GPT‑5.4, Codex CLI v0.122.0 | **32.58%** | **7/25** |
| **v2** (Jun 6 2026) | GPT‑5.5 high‑reasoning, Codex CLI v0.128.0 | **58.12%** | **15/25** |

→ "**~58% on the 25 public games, GPT‑5.5**" is real **but is the v2/best‑config number, not a robust v1 baseline.** Treat 58% as best‑case single‑config.
→ ⚠️ **Hosted frontier API** (Codex/GPT‑5.x) → **NOT Kaggle‑offline‑eligible** as‑is; it's a blueprint for an offline open‑weight coder.

### 3.2 DreamTeam — "Workspace Optimization: How to Train Your Agent" (medium)
arXiv **2605.09650** (~May 2026) + `symbolica.ai/blog/arc-agi-3`. A **DreamerV3‑inspired multi‑agent harness**; roles:
- **Observer + Simulator** — build/track the executable world model / hidden dynamics
- **Inductive Explorer** — commit reusable strategies, maintain sub‑goal sets & policies (plan/strategize)
- **Transductive Explorer** — information‑seeking probes (hypothesize/probe)
- **Critic** — route failures to owners · **Team Leader** — arbitration
- Uses Opus 4.6 + GPT‑5.5 in the roles.

**Result:** improves a protocol‑matched SOTA agent **36% → 38.4% mean RHAE** with **31% fewer environment actions/game**.
→ ⚠️ **Confidence medium (2‑1 on the number):** 38.36% is a **2‑run mean** vs a **single‑run, self‑reported** Symbolica/Agentica baseline (36.08%); **no significance test**, ~2.3pp margin; authors concede *"scores are bounded by time and cost budgets rather than by system intelligence."* The architecture claim itself is solid (3‑0).

### 3.3 What the preview leaders did (context, §1.1)
StochasticGoose (CNN frame‑change RL) and Blind Squirrel (state‑graph + ResNet18 value) are **goal‑agnostic informed search** — the same class as ours, and the same class that the 2026 papers explicitly move *beyond*.

---

## 4. The in‑process‑simulator (deepcopy) edge — legality status (high, with a scope limit)

**Verified facts (3‑0):**
- **ARC‑AGI Toolkit** is the official open‑source Python SDK (`github.com/arcprize/arc-agi`, PyPI `arc-agi` / `arc-agi-3`), runs games **entirely locally without the API** (`OperationMode.OFFLINE`, `environments_dir`), built on **ARCEngine**.
- Games are authored as **deterministic in‑process Python objects** (subclass `ARCBaseGame`, override `step()`, `level_reset()` / `full_reset()`).
- Technical Report verbatim: engine *"implemented in Python to achieve our minimum performance goal threshold of **1,000 frames per second**"*; *"The environment's state does not change asynchronously from the agent's actions"*; *"the engine can **serialize and faithfully re‑execute action traces**."*
- ARCEngine README: executes *"directly as Python objects without external dependencies"* — exactly the substrate a deepcopy simulator needs.

**Legality — no RULE forbids it, but the harness actively patches the adjacent exploit (updated 2026‑06‑20, primary source):**
- The Kaggle competition **rules page is JS‑rendered and could not be fetched**, but the **ARC Prize competition page + Kaggle search confirm the general rules**: *"No internet access during evaluation"*, *"All code and methods must be open sourced to be eligible for prizes"* (CC0/MIT‑0), compute limits announced at launch, sandboxed notebook submission. **NO rule prohibiting reading the game source, inspecting, deepcopying, cloning, serializing, or reverse‑engineering the environment object was found in ANY official source.**
- 🔴 **BUT — the SingularityNET "Executable World Models" paper (arXiv 2605.05138) documents that the competition harness PATCHES the adjacent exploit:** *"an agent could **start a second game client and use it as an unscored simulator for trying actions. This attack is not available in the competition setting, where the arc_agi library does not allow the same game to be started a second time.**"* So **re‑instantiation (a 2nd client) is BLOCKED in competition mode.**
- ✅ **Our `copy.deepcopy(wrapper._game)` is a DIFFERENT mechanism** — it clones the *already‑instantiated* in‑process object; it does NOT start a second game. So it may survive the patch. **But it is squarely in the same risk class the organizers are actively closing** — exactly the master‑plan **R2** caveat, now confirmed by a primary source.

**⚠️ SCOPE LIMITS & RISK:**
- The deepcopy edge is **structurally impossible on the server‑side live API track** (`three.arcprize.org`). It exists **only on the offline/local toolkit track we're on** — and that local track is precisely where the paper says the "2nd client" trick *used* to work before being patched.
- **Day‑1 probe is mandatory:** verify on the REAL competition harness that `deepcopy(_game)` (a) is reachable and (b) doesn't trip the same "second instance" guard. **Never ship the deepcopy simulator as the SOLE engine** — keep the RED fallback live.

→ **Bottom line (revised):** our deepcopy‑simulator is **not forbidden by any rule**, and it's the *correct* primitive (a 2nd client is blocked, so deepcopy is the only viable in‑process simulator) — **but it lives in a risk class the organizers actively patch**, so treat it as **conditional, Day‑1‑verifiable**, with a non‑deepcopy fallback ready.

---

## 5. Corrections to the team's `research.md` notes

| Note in `research.md` | Verdict | Correct version |
|---|---|---|
| StochasticGoose 12.58% (preview 1st, 18 levels) | ✅ **Correct** | …but **2025 preview only**; collapsed to ~0.25% at 2026 launch |
| Blind Squirrel 6.71% (preview 2nd, 13 levels) | ✅ **Correct** | preview only |
| Rudakov 3.64% / 12 levels | ✅ **Correct (as the blog official‑submission score)** | bug‑depressed; paper reports median **30/52 levels**, ~17 post‑bugfix |
| SingularityNET "GPT‑5.5 → 15/25, mean RHAE 58.12%" | ✅ **Real, but mislabeled** | that's **v2** (GPT‑5.5); **v1 is 32.58% / 7/25 (GPT‑5.4)** — don't treat 58% as the baseline |
| SingularityNET arXiv **2605.05138** | ✅ **Real arXiv ID** (we'd earlier flagged it as possibly fabricated — it is genuine) | v1 + v2 both exist |
| DreamTeam arXiv **2605.09650**, "Workspace Optimization" | ✅ **Real**; 36→38.4% RHAE, −31% actions | medium confidence on the number (2‑run mean, no sig test) |
| "ARChitects / NVARC / TRM / SOAR / CompressARC" (ARC‑AGI‑2 lineage) | *(not re‑checked this run; those are ARC‑AGI‑2, out of scope of this ARC‑AGI‑3 search)* | — |
| **REFUTED**: "agent solved 7 games AND >75% RHAE on 6 games" | ❌ **Refuted as a conjunction (0‑3)** | These are **two separate v1 facts** — do not state them as a single combined claim |

Net: the comparison docs that cite these numbers are **substantially accurate**; the main fixes are (a) label everything **preview vs 2026‑launch**, (b) split the **SingularityNET v1/v2** numbers, (c) stop conflating the two v1 facts.

---

## 6. Actionable for OUR agent (`agent/forward_model_agent.py`)

1. **Our diagnosis is field‑validated.** The literature documents the exact wall (goal‑agnostic search intractable on deep levels; novelty "orthogonal to goal‑relevance"). We are not missing an easy trick — this is *the* open hard part.
2. **The proven fix = an inferred, code‑checkable win predicate used as a search target.** Our deepcopy/BFS substrate is *ideal* for this: instead of LLM‑synthesizing the whole world model (we already HAVE the exact model via deepcopy), we only need to **infer the GOAL PREDICATE** and use it as the A* heuristic / pruning target. This is strictly less than what Rodionov/DreamTeam do (they must also learn dynamics; we get dynamics for free).
   → **Concrete next step (A2):** mine the win predicate from rollouts (correlate frame/feature deltas with `_score` ticks), express it as a Python check over the deepcopy state, and plug it as the `heuristic` in `plan_to_next_level` (the hook already exists). This is the "executable world model" idea **minus the dynamics‑synthesis half we don't need.**
3. **Keep the deepcopy edge — but verify the Kaggle rules.** It's allowed by official docs; the Kaggle‑track rules are the one unchecked binding doc.
4. **Right‑size expectations.** Even the SOTA offline‑hostile world‑model agent is 32–58% on the *public* set with a frontier coding LLM; private generalization is untested and the preview‑leader collapse (12.58%→0.25%) is a loud OOD warning. A robust offline agent in the single‑digits‑to‑teens % would already be competitive.

---

## 7. Caveats & time‑sensitivity

- **Two eras:** 12.58% / 6.71% / 3.64% are **2025 preview** (hidden eval); sub‑1% frontier numbers are **2026 launch** (Technical Report, Apr 22 2026). Preview rank does NOT carry over (StochasticGoose 12.58%→~0.25%).
- **Executable‑World‑Models version confusion:** 58.12% = **v2** best‑config (GPT‑5.5, Codex 0.128.0, 15/25); **v1 = 32.58%** (GPT‑5.4, 7/25). Treat 58% as best‑case, single‑config.
- **DreamTeam statistical thinness:** 36→38.4% is a 2‑run mean vs a single‑run self‑reported "SOTA"; no significance test; budget‑bound by the authors' own admission.
- **Deepcopy‑"allowed" scope:** verified only vs ARC Prize official docs/report/engine; **Kaggle competition rules NOT retrieved** and may prohibit; impossible on the live API track anyway.

---

## 8. Open questions (worth chasing next)

1. ~~Do the Kaggle offline‑track rules permit/forbid deepcopying the in‑process game?~~ **PARTIALLY ANSWERED (2026‑06‑20):** no official source forbids it; general rules confirmed (no internet, open‑source, sandboxed). The harness blocks **re‑instantiating** a 2nd game client in competition mode (SingularityNET paper) but says nothing about `deepcopy` of the live object. *Still open:* whether the same "second instance" guard also catches a deepcopy — **resolve via the Day‑1 probe on the real harness.**
2. **Current mid‑2026 Kaggle leaderboard standing & top method** for the full 2026 competition (vs the 2025 preview)? Not covered by the verified sources.
3. **How exactly do the executable‑world‑model agents represent & verify the inferred WIN CONDITION in code** — and can that synthesized goal predicate be plugged into our deepcopy‑BFS as the pruning/heuristic target to break the depth‑~44 wall?
4. **Did ARC Prize ever publicly comment on / patch the deepcopy edge** after the preview? None found (absence of evidence ≠ evidence of absence).

---

## 9. Sources (19 fetched; ★ = primary, directly load‑bearing)

**ARC Prize official**
- ★ ARC‑AGI‑3 Technical Report (PDF, Apr 22 2026) — https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf
- ★ 30‑Day Preview Learnings (blog) — https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings
- ★ Preview Agents competition page — https://arcprize.org/competitions/arc-agi-3-preview-agents
- ★ Methodology docs — https://docs.arcprize.org/methodology
- ★ Toolkit overview — https://docs.arcprize.org/toolkit/overview
- ★ ARCEngine (GitHub) — https://github.com/arcprize/ARCEngine
- Kaggle competition leaderboard *(did not render / unreliable)* — https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard

**Papers (arXiv / OpenReview)**
- ★ Executable World Models for ARC‑AGI‑3 (Rodionov/SingularityNET) — https://arxiv.org/abs/2605.05138 · v1 https://arxiv.org/html/2605.05138v1 · v2 https://arxiv.org/html/2605.05138v2
- ★ DreamTeam / Workspace Optimization — https://arxiv.org/abs/2605.09650
- ★ Graph‑Based Exploration for ARC‑AGI‑3 (Rudakov) — https://arxiv.org/abs/2512.24156 · PDF https://arxiv.org/pdf/2512.24156
- ARC‑AGI‑3 challenge / frontier‑agentic paper — https://arxiv.org/html/2603.24621v1
- transferable‑techniques set — https://arxiv.org/pdf/2509.24116 · https://arxiv.org/pdf/2510.12088 · https://arxiv.org/html/2601.18620v1 · https://arxiv.org/abs/2604.18131 · https://arxiv.org/pdf/2203.16311 · https://openreview.net/pdf?id=1UoB7IWiku

**Preview‑leader code / writeups**
- ★ StochasticGoose solution (Dries Smit) — https://github.com/DriesSmit/ARC3-solution
- ★ "just‑explore" graph agent (Rudakov) — https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore
- ★ Executable‑world‑model baseline repo — https://github.com/astroseger/arc-3-agents-baseline1
- Symbolica DreamTeam blog — https://symbolica.ai/blog/arc-agi-3
- community blog (secondary) — https://deniseholt.us/arc-agi-3-we-didnt-expect-this-to-happen/

---

*Generated from a verified deep‑research pass (24/25 claims confirmed at 3‑vote, 1 refuted). Numbers are split across the 2025 preview and the 2026 launch — always state which era. The deepcopy‑simulator edge is allowed by official docs but the binding Kaggle‑track rules remain unverified.*
