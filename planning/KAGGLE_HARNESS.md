# How the Kaggle ARC-AGI-3 grading harness ACTUALLY works

**Saved 2026-06-20.** Reverse-engineered from (a) the competition Overview/Data/Code-Requirements and (b) a REAL submitted notebook ("ARC26-3 Agent v15 — Best Score 0.46", revised from kaggle.com/code/ashvinsingh). This is the authoritative reference; it supersedes the OFFLINE-mode assumption in `SUBMISSION.md`.

---

## ⭐ THE DECISIVE FINDING — grading is FRAME-ONLY via a gateway server (CONFIRMED, not estimated)

The reference notebook's scored-run cell is the smoking gun:

```python
if os.getenv("KAGGLE_IS_COMPETITION_RERUN") == "1":
    BASE = Path("/kaggle/working/ARC-AGI-3-Agents")
    SRC  = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents")
    # handshake with a SEPARATE server container:
    subprocess.run(["curl","--fail",...,"http://gateway:8001/api/games"])
    shutil.copytree(SRC, BASE)
    # inject agent + rewrite agents/__init__.py (only random + my_agent → avoids the LLM-template import footgun)
    env_file.write_text("\n".join([
        "SCHEME=http", "HOST=gateway", "PORT=8001",
        "ARC_API_KEY=test-key-123",
        "ARC_BASE_URL=http://gateway:8001/",
        "OPERATION_MODE=online",                 # <<< ONLINE / REMOTE wrapper
        "RECORDINGS_DIR=/kaggle/working/server_recording"]))
    subprocess.run(["python","main.py","--agent","myagent"], cwd=BASE, check=True)
```

What this proves:

1. **The 110 private games run on a SEPARATE server container `gateway:8001`.** The agent talks to it over HTTP. The agent's own process does NOT hold the game objects.
2. **`OPERATION_MODE=online` → `RemoteEnvironmentWrapper`** (HTTP, `requests`). Per the wheel source, that wrapper has **no `_game` and no source on disk**. So the scored agent is **frame-only**: it receives frames (JSON grids) and returns actions. **This CONFIRMS what we estimated at ~82% — now ~certain, from a real 0.46 submission.**
3. **The deepcopy/source exploit is DEAD on private.** The reference agent ships an elaborate `BFSSolver` that tries to find each game's `.py` (`environment_info.local_dir`, then globs `/tmp/.../{gid}.py`, `/kaggle/.../{gid}.py`). In online mode `local_dir` is `None` and private source lives only inside the `gateway` container, so the BFS finds source **only for the 25 public dev games**, never the private ones → it falls back to a **CNN policy**. The realized private score (0.46) is essentially that frame-only CNN. The BFS is dev-theater that does not transfer — exactly the StochasticGoose public→private collapse pattern, and exactly what our own analysis predicted for our deepcopy engine.
4. **"Internet disabled" + "from the API" are reconciled:** the "API" is the *local* `gateway:8001` container on the Kaggle competition network, not the public internet.

### What this locks in for our strategy
- **Abandon deepcopy/source as the private engine** (confirmed: games are remote, frame-only). Keep our search machinery ONLY to run *inside* a learned/synthesized world model, and as a DEV tool on the 25 public games (which DO load in-process via `LocalEnvironmentWrapper`).
- **Build a frame-only `Agent` subclass** (`choose_action(frames, latest_frame)`), inject it into `ARC-AGI-3-Agents/agents/templates/`, register in a rewritten `agents/__init__.py`, talk to `gateway:8001`.
- **Real score calibration (ground truth):** this BFS+CNN agent = **0.46**; leaderboard pack 0.46–0.70; leader Tufa Labs (StochasticGoose's lab) **1.21**. A modest frame-only CNN already reaches ~0.46. The leader's edge is better frame-only reasoning (exploration/memory/world-model), NOT an exploit.

---

## Code Requirements (from Overview)
- Submission = **Notebook**; **CPU or GPU ≤ 9 h** run-time.
- **Internet access disabled** at evaluation. Freely & publicly available external data + **pre-trained models allowed** (→ open-weight LLM OK, must be a Kaggle dataset, no download at run).
- **RTX 6000 (g4-standard-48, 48 GB) available** for this competition's notebooks; RTX sessions must have internet disabled. (H100/P100/T4 also in pool per the reference author.)
- Submission file auto-generated once the agent acts on any game.

## Data facts (from Data tab)
- Agent receives **frames = JSON** (grid ≤ 64×64, values 0–15, (0,0) top-left) + metadata; replies with actions. States: NOT_FINISHED / WIN / GAME_OVER.
- **Actions:** RESET, ACTION1–5 (simple), ACTION6 (needs x,y), ACTION7 (extra simple). Meaning differs per game → must be discovered by exploration.
- **Private set = 110 games the agent has NEVER seen; half → Public LB, half → Private LB** (≈55/55). The 25 `environment_files/` games are **dev only** (not scored).
- **Scoring:** per-level `= min(human_actions/agent_actions, 1.0)²` (cap **1.0**, i.e. 100% — note: our older notes said 1.15; the Kaggle metric caps at 1.0); per-game = level-index-weighted average; total = average over games; **0–100%**, completion-dominated.
- Agent contract: `is_done(frames, latest_frame)`, `choose_action(frames, latest_frame)`. A **Swarm** runs agent instances across all games in parallel. Lifecycle: get games from API → open scorecard → RESET + act per game → close scorecard.

---

## Reference submission notebook (verbatim — "ARC26-3 Agent v15", best 0.46)

> Approach = CPU-bound BFS planner (deepcopy the game class loaded from source) + GPU CNN fallback (CBAM + action-effect attention) + a chooser that prefers BFS, falls back to CNN, and transfers earlier-level solutions. NOTE: the BFS only works where game source is on disk (public dev games), so on the private gateway set the CNN carries the score.

### Cell 1 — install (offline, from competition wheels)
```python
!pip install -q --no-index --find-links \
    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \
    arc-agi python-dotenv
!pip check
```

### Cell 2 — `/kaggle/working/my_agent.py` (the agent)
```python
# Refactoring/optimizing "FORGE v19"
import copy, glob, hashlib, importlib.util, logging, os, random, time, traceback
from collections import deque
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState, ActionInput
logger = logging.getLogger(__name__)

# ==================== BFS SOLVER ====================
class BFSSolver:
    """Offline BFS solver using direct game class instantiation."""
    def __init__(self, game_path, game_class_name, scan_timeout=3, bfs_timeout=120):
        self.game_path = game_path; self.class_name = game_class_name
        self.scan_timeout = scan_timeout; self.bfs_timeout = bfs_timeout
        self.game_cls = None; self.solutions = {}  # level_idx -> action list
        self._run_log = {"steps":0,"bfs_used":False,"bfs_success":False,"cnn_steps":0,
                         "explore_steps":0,"revisit_count":0,"reward_sum":0.0}
        self._seen_states = set()
        self._run_id = hash((time.time(), getattr(self,"game_id","unknown")))
        self._last_logged_score = None

    def load(self):
        try:
            spec = importlib.util.spec_from_file_location('game_mod', self.game_path)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            self.game_cls = getattr(mod, self.class_name); return True
        except Exception as e:
            logger.warning(f"BFS: Failed to load game class: {e}"); return False

    def _state_hash(self, g, frame, hidden_fields=None):
        fh = hashlib.md5(frame.tobytes()).hexdigest()[:16]
        if hidden_fields:
            extras=[]
            for field_name in hidden_fields:
                try:
                    v=getattr(g,field_name,None)
                    if v is not None: extras.append(f"{field_name}={v}")
                except: pass
            if extras: return fh+"|"+"|".join(extras)
        return fh

    def _probe_hidden_fields(self, game, actions):
        """Discover scalar fields that change without pixel change (hidden state)."""
        if not actions: return []
        initial={k:v for k,v in game.__dict__.items()
                 if isinstance(v,(int,float,bool)) and not k.startswith('__')}
        changing=set(); frame0=game.get_pixels(0,0,64,64)
        for act_id,data in actions[:10]:
            g=copy.deepcopy(game)
            try:
                ai=ActionInput(id=GameAction.from_id(act_id),data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                g.perform_action(ai,raw=True)
            except: continue
            for k,v in g.__dict__.items():
                if isinstance(v,(int,float,bool)) and not k.startswith('__'):
                    if k in initial and v!=initial[k] and k not in ('_action_count','_full_reset','_action_complete'):
                        changing.add(k)
        hidden=[]
        for f in changing:
            if f.startswith('_') and f not in ('_current_level_index','_score'): continue
            hidden.append(f)
        return sorted(hidden)

    def _scan_actions(self, game, f0, bg):
        """Scan effective actions. Returns [(action_id, data)]."""
        avail=game._available_actions; actions=[]
        for a in [a for a in avail if a<=5]:
            g=copy.deepcopy(game)
            try:
                r=g.perform_action(ActionInput(id=GameAction.from_id(a)),raw=True)
                if r.frame and np.sum(f0!=np.array(r.frame[-1]))>0: actions.append((a,None))
            except: pass
        if 6 in avail:
            t0=time.time(); seen_effects=set()
            for y in range(0,64,2):
                if time.time()-t0>self.scan_timeout: break
                for x in range(0,64,2):
                    if f0[y,x]==bg: continue
                    g=copy.deepcopy(game)
                    try:
                        r=g.perform_action(ActionInput(id=GameAction.ACTION6,data={'x':x,'y':y,'game_id':'bfs'}),raw=True)
                        if not r.frame: continue
                        f=np.array(r.frame[-1]); diff=np.sum(f0!=f)
                        if diff>0:
                            eh=hashlib.md5(f.tobytes()).hexdigest()[:12]
                            if eh not in seen_effects: seen_effects.add(eh); actions.append((6,{'x':x,'y':y,'game_id':'bfs'}))
                    except: pass
        return actions

    def solve_level(self, level_idx, max_states=500000, prev_solution=None):
        if not self.game_cls: return None
        game=self.game_cls(); game.set_level(level_idx)
        game.perform_action(ActionInput(id=GameAction.RESET),raw=True)
        r0=game.perform_action(ActionInput(id=GameAction.RESET),raw=True)
        if not r0.frame: return None
        f0=np.asarray(r0.frame[-1],dtype=np.uint8); bg=int(np.bincount(f0.flatten(),minlength=16).argmax())
        if prev_solution and level_idx>0:
            tr=self._try_transfer(game,level_idx,prev_solution,f0)
            if tr: return tr
        actions=self._scan_actions(game,f0,bg)
        if not actions:   # warm-up an action to unlock others
            for warmup_id in [a for a in game._available_actions if a<=4]:
                g=copy.deepcopy(game)
                try:
                    g.perform_action(ActionInput(id=GameAction.from_id(warmup_id)),raw=True)
                    fa=np.asarray(g.get_pixels(0,0,64,64),dtype=np.uint8)
                    actions=self._scan_actions(g,fa,bg)
                    if actions: game=g; f0=fa; break
                except: pass
        if not actions: return None
        # FAST BFS: carry full game state forward (deepcopy), hash frames, dedup, depth<=30
        visited={hash(f0.tobytes())}; queue=deque([(game,[],0)]); t0=time.time(); explored=0
        while queue and explored<max_states and (time.time()-t0)<self.bfs_timeout:
            g,hist,depth=queue.popleft()
            for act_id,data in actions:
                try:
                    g2=copy.deepcopy(g)
                    ai=ActionInput(id=GameAction.from_id(act_id),data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r=g2.perform_action(ai,raw=True); explored+=1
                    if not r.frame: continue
                    f=np.asarray(r.frame[-1],dtype=np.uint8); h=hash(f.tobytes())
                    if h in visited: continue
                    visited.add(h); new_hist=hist+[(act_id,data)]
                    if r.levels_completed>level_idx or g2._current_level_index>level_idx:
                        self.solutions[level_idx]=new_hist; return new_hist
                    if depth<30: queue.append((g2,new_hist,depth+1))
                except: continue
        # ... (early-exit + hidden-field RETRY pass with the same BFS, omitted for brevity)
        return None

    def _try_transfer(self, game, level_idx, prev_solution, f1):
        """Replay prev level's actions; else object-centroid offset (dx,dy) on click coords."""
        # (direct replay; then match same-color objects between levels, shift ACTION6 x/y by mean offset)
        ...

def find_game_source_and_class(game_id, arc_env=None):
    """Find the game .py + class. Uses arc_env.environment_info.local_dir, else globs
    /tmp/*/{gid}/*/{gid}.py, /kaggle/*/{gid}*/{gid}.py, **/game_sources/**/{gid}.py.
    -> In ONLINE/gateway mode local_dir is None and private source is absent, so this
       only resolves for the 25 public dev games."""
    ...

# ==================== CNN FALLBACK (GPU) ====================
class CBAM(nn.Module): ...            # channel+spatial attention
class ActionEffectAttention(nn.Module): ...  # attends over (frame-diff, action, reward) memory
class ForgeNet(nn.Module):           # 4 conv blocks -> CBAM -> action head (5) + click head (HxW)
    def __init__(s, in_ch=26, g=64): ...
    def forward(s, x, mem_diffs=None, mem_actions=None, mem_rewards=None): ...

# ==================== AGENT ====================
class MyAgent(Agent):
    MAX_ACTIONS=float('inf'); _MAX_FRAMES=10
    def __init__(s,*a,**kw):
        super().__init__(*a,**kw)
        # ... (original CNN/init code) ; plus self-assessment metrics dict
    def choose_action(s, frames, lf):
        try:
            lvl=s._lvl(lf)
            if lvl!=s.cl: s._level_start_time=time.time(); s._level_step_count=0; s.cl=lvl
            raw=s._raw(lf)
            if s._bfs_solution and s._bfs_step<len(s._bfs_solution):   # BFS path (public only)
                act_id,data=s._bfs_solution[s._bfs_step]; s._bfs_step+=1
                action=GameAction.from_id(act_id); action.reasoning="bfs"; return action
            return super().choose_action(frames, lf)                   # CNN path (carries private)
        except Exception:
            traceback.print_exc(); return random.choice([GameAction.ACTION1])
```

### Cell 3 — scored-run launcher (the harness contract — see top of this file)
```python
import os, shutil, subprocess
from pathlib import Path
if os.getenv("KAGGLE_IS_COMPETITION_RERUN") == "1":
    BASE=Path("/kaggle/working/ARC-AGI-3-Agents"); SRC=Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents")
    AGENT_SRC=Path("/kaggle/working/my_agent.py"); AGENT_DST=BASE/"agents/templates/my_agent.py"
    subprocess.run(["curl","--fail","--retry","60","--retry-all-errors","--retry-delay","5",
                    "--retry-max-time","300","http://gateway:8001/api/games"], check=False)
    if BASE.exists(): shutil.rmtree(BASE)
    shutil.copytree(SRC, BASE)
    shutil.copy(AGENT_SRC, AGENT_DST)
    (BASE/"agents/__init__.py").write_text(
        "from typing import Type\nfrom dotenv import load_dotenv\n"
        "from .agent import Agent, Playback\nfrom .swarm import Swarm\n"
        "from .templates.random_agent import Random\nfrom .templates.my_agent import MyAgent\n\n"
        "load_dotenv()\n\nAVAILABLE_AGENTS: dict[str, Type[Agent]] = {\n"
        '    "random": Random,\n    "myagent": MyAgent\n}\n')
    (BASE/".env").write_text("\n".join(["SCHEME=http","HOST=gateway","PORT=8001",
        "ARC_API_KEY=test-key-123","ARC_BASE_URL=http://gateway:8001/",
        "OPERATION_MODE=online","RECORDINGS_DIR=/kaggle/working/server_recording"])+"\n")
    subprocess.run(["python","main.py","--agent","myagent"], cwd=str(BASE),
                   env={**os.environ,"MPLBACKEND":"agg","PYTHONUNBUFFERED":"1"}, check=True)
```

### Cell 4 — placeholder submission for the non-rerun (commit) path
```python
import os
if os.getenv("KAGGLE_IS_COMPETITION_RERUN") != "1":
    import pandas as pd
    pd.DataFrame([{"row_id":"debug_0","game_id":"1","end_of_game":True,"score":0}]) \
      .to_parquet("/kaggle/working/submission.parquet", index=False)
```

---

## Public notebook landscape (scores, 2026-06-20) — what works, what doesn't

The public scene is a handful of base agents (FORGE / ashvin / CHRONOS / StochasticGoose) plus many forks, all converging at **0.18–0.46**. The leaderboard #1 (Tufa Labs 1.21) does NOT publish — the gap from 0.46 → 1.21 is the unshared secret sauce.

| Score | Notebook | Approach |
|---|---|---|
| **1.21** | Tufa Labs (LB #1, private) | unknown / unpublished |
| 0.46 | [ARC26-3] Agent v15 (ashvin) | deepcopy-BFS (public-only) + CNN fallback |
| 0.46 | Persistent Memory BFS | env-graph BFS + memory |
| 0.43 | FORGE (private fork) | graph + CNN |
| **0.42** | **Ash / FORGE v20 (CHRONOS)** | **graph exploration + online ChangeNet CNN — BFS DELETED, pure frame-only (our base)** |
| 0.42 | Hybrid Solver (BFS+CNN+Heuristics) | graph + CNN + heuristics |
| 0.39 | FORGE (orig) | graph + CNN |
| 0.36 | v31 Code Pro | graph + CNN |
| 0.35 | FORGE v16 Trigger-aware BFS | trigger-aware exploration |
| 0.34 | ShotInTheDark | exploration |
| 0.32 | StochasticGoose++ CNN Frame-Change | CNN frame-change |
| 0.30 | Redpill: Latent Imaginations / Dream Rollouts | world-model dreaming |
| 0.29 | MCTS Solver | MCTS |
| **0.25** | StochasticGoose (official sample) | CNN frame-change baseline |
| **0.19** | Just Explore (sample) | pure exploration |
| **0.18** | Random Agent (sample) | random valid actions |
| 0.06 | DreamerV3 + ICM | model-based RL |

**Read of the landscape:**
- **Floor 0.18 = random**; pure exploration ≈ 0.19. So ~0.18 comes "for free" from acting at all (trivial L1s).
- **The workhorse is graph-exploration + online frame-change CNN (FORGE/StochasticGoose family) → 0.32–0.46.** This is the proven frame-only approach and our base.
- **Complexity ≠ score:** DreamerV3 (0.06), MCTS (0.29), latent-dream (0.30) all UNDERPERFORM the simple graph+CNN. Don't over-engineer.
- **"BFS" in 0.42–0.46 notebooks = environment-graph BFS (exploration), not the deepcopy simulator** (which is dead on private — v15's deepcopy-BFS and v20's no-BFS score the same ~0.42–0.46, confirming the BFS adds ~nothing on private).
- **Public ceiling ≈ 0.46; LB #1 = 1.21.** To place (top-5 prize ≈ pack 0.46–0.70) we need a clean frame-only agent ~matching the family; to WIN we need something past 0.46 (better exploration / learned world-model / the LLM path) — the unshared frontier.

