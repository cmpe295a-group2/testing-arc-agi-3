# ARC-AGI-3 Game Rules — the simple version

*Written for SJSU 295A Group 2. Everything here is verified against the real engine code
(`arcengine/base_game.py`) and the Kaggle Data/Scoring pages.*

---

## 1. Structure: Game → Level → Action

- **Game:** one small puzzle "world" — a 64×64 grid where every cell is a colour number
  0–15. The competition is scored on **110 hidden games** (never seen before). The 25
  public games are only for practice.
- **Level:** each game has **several levels, played IN ORDER, getting harder**. You must
  beat level 1 to reach level 2. The number of levels **varies by game** — e.g. for the
  public set: vc33 has 7 levels, tu93 has 9, lf52 has 10. (Private games unknown, but
  similar.)
- **Action:** each turn the agent picks **one action**:
  | Action | Meaning |
  |---|---|
  | RESET | Replay the current level from the start |
  | ACTION1–5, ACTION7 | 6 "simple" actions (e.g. up/down/left/right/interact) |
  | ACTION6 | Click on a cell (x, y) |
  - **Nobody tells you what an action does.** The agent must **try things to find out** —
    and the meaning is different in every game.
- **3 game states:**
  - `NOT_FINISHED` — still playing.
  - `WIN` — beat **every level** → the game is done.
  - `GAME_OVER` — failed / died (a wrong move) → must **RESET** to retry that level.

---

## 2. How scoring works (RHAE) — the 3 layers

Scoring measures **two things**: (a) **how many levels you beat**, and (b) **how few
moves you used vs a human**.

For **each level you beat:**
```
level_score = (human's moves  /  agent's moves)²      (capped at 1.15 = 115%)
```
*(The engine caps a level at 115%, i.e. you can score slightly over 100% by using fewer moves than the human — verified in `scorecard.py`. The Kaggle page's "cap 1.0" is a simplification.)*
The three layers stack up like this:
```
LEVEL score = (human moves / agent moves)²        ← per level, 0 to 1
     ↓ averaged (later levels weigh more)
GAME score  = weighted average of its levels        ← per game, 0 to 1
     ↓ averaged (every game counts equally)
TOTAL score = average of ALL games' scores          ← 0% to 100%
```

So the answer to "is it per game or cumulative across games?" → **per game, then the
game scores are AVERAGED. It is NOT a running sum from game 1 to game 110.** A game you
never finish scores 0 but still counts in the average — which is why, with 110 games
mostly at 0, the **world #1 is only ~1.2%** (the benchmark is brutally hard).

### Why fewer moves matters so much (squared!)
If a level takes a human **30 moves**:
| Agent uses | level_score |
|---|---|
| 30 (same as human) | **1.00** (100%) |
| 60 (twice as many) | 0.25 (25%) |
| 300 (ten times) | **0.01** (1%) |

→ **Wasted moves crush the score fast (because it's squared). Wasting moves is the enemy.**

---

## 3. Worked example: agent vs human

Suppose there are just **2 games**, each with **1 level**, and a **human beats each level
in 10 moves**.

**Game 1** — the agent plays 10 moves, then RESET, then 10 more moves, then WIN.
- ⚠️ **RESET does NOT clear the move counter** (verified in the engine). So this level cost
  **20 moves** (10 wasted + 10 real), not 10.
- `game1_score = (10 / 20)² = 0.25`

**Game 2** — the agent plays 10 moves, then WIN.
- `game2_score = (10 / 10)² = 1.00`

**TOTAL = average of the games = (0.25 + 1.00) / 2 = 0.625 = 62.5%**

Notice how the RESET in Game 1 **cost points**: without those 10 wasted moves it would have
been 1.00, but it dropped to 0.25. Restarting never erases moves already spent.

### Same idea, a multi-level game
A game with **3 levels**, where the agent beats levels 1 and 2 but gets stuck on level 3:
```
game_score = (1×score_L1 + 2×score_L2 + 3×0) / (1 + 2 + 3)
```
Level 3 is unfinished (score 0) **and** carries the **highest weight (3)**, so it drags the
game score down hard. → **Later levels are worth more than earlier ones.** Beating deep
levels is where the real points are.

---

## 4. The details that make it HARD for the agent

1. **No instructions** — the goal is unknown, and what each action does is unknown. Must probe.
2. **Every game has different rules** — ACTION6/click might mean "eat" in one game and "open
   a door" in another; you can't carry knowledge from one game to the next.
3. **Frame-only** — at scoring time the agent **cannot** read the game's code/objects (the
   game runs on a separate server); it only receives the 64×64 grid and must infer everything.
4. **Wasted-move penalty (squared)** — you cannot brute-force; grinding moves destroys your score.
5. **Deep levels need long move sequences** — later levels need dozens-to-hundreds of correct
   moves in order; blind trial-and-error can't reach them.
6. **Unseen private games** — the 110 scored games are never-before-seen; you can't memorize.
7. **GAME_OVER (deadly moves)** — some moves lose instantly; the agent must learn to avoid them.
8. **Hidden state** — some games change internal state **while the picture stays the same**
   (e.g. sc25), so the agent can't tell two different situations apart from the frame.
9. **Huge click space** — ACTION6 has 64×64 = **4096 possible cells**; finding the right click
   is a needle in a haystack.
10. **9-hour budget shared across all games** — each game gets only a few minutes of exploration.

→ Because of 4 + 5 + 8 + 9, **blind exploration (like our current graph agent) only beats
~10/183 public levels**. Doing better requires **learning** (a CNN that predicts which moves
are useful) — which is why a GPU is needed. This matches exactly what we measured.

---

*See also: [KAGGLE_RULES.md](KAGGLE_RULES.md) (legal rules), [KAGGLE_HARNESS.md](KAGGLE_HARNESS.md)
(how Kaggle grades), [submission/README.md](../submission/README.md) (our agent).*
