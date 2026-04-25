"""router.py — FastAPI router for Wordle RLVR env.

Mounts under /wordle/* prefix:
  POST /wordle/reset         body: {seed?, target_word?}  -> {session_id, observation}
  POST /wordle/step          body: {session_id, guess}    -> {observation, reward_breakdown}
  POST /wordle/grade         body: {session_id}            -> grade dict
  GET  /wordle/health                                       -> service status
  GET  /wordle/ui                                           -> minimal HTML game

Session state held in-process (uvicorn single-worker assumed). For multi-worker
deploy, swap dict for redis. Sessions auto-expire after 1h.
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from typing import Optional

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import HTMLResponse
except ImportError:
    APIRouter = None  # type: ignore
    HTTPException = Exception  # type: ignore
    HTMLResponse = None  # type: ignore

from .env import (
    WordleAction, WordleResetRequest, WordleState,
    reset as env_reset, step as env_step, grade as env_grade,
    WORD_LIST,
)

logger = logging.getLogger(__name__)

router = APIRouter() if APIRouter is not None else None

_MAX_SESSIONS = 200
_SESSION_TTL_S = 3600
_sessions: "OrderedDict[str, tuple[float, WordleState]]" = OrderedDict()


def _gc_sessions() -> None:
    now = time.time()
    expired = [sid for sid, (ts, _) in _sessions.items()
                if now - ts > _SESSION_TTL_S]
    for sid in expired:
        _sessions.pop(sid, None)
    while len(_sessions) > _MAX_SESSIONS:
        _sessions.popitem(last=False)


class StepRequest(BaseModel):
    session_id: str
    guess: str = Field(..., min_length=1, max_length=10)


class SessionRequest(BaseModel):
    session_id: str


if router is not None:

    @router.get("/wordle/health")
    def wordle_health() -> dict:
        return {
            "status": "ok",
            "n_active_sessions": len(_sessions),
            "n_words_in_dict": len(WORD_LIST),
            "max_sessions": _MAX_SESSIONS,
            "session_ttl_s": _SESSION_TTL_S,
            "openenv_compliant": True,
            "rlvr": True,
            "reward_components": [
                "green_credit", "yellow_credit", "solve_bonus",
                "timeout_penalty", "format_gate", "dictionary_gate",
            ],
            "anti_hack_layers": ["format_gate", "dictionary_gate", "timeout"],
        }

    @router.post("/wordle/reset", tags=["wordle"])
    def wordle_reset(req: WordleResetRequest) -> dict:
        _gc_sessions()
        state, obs = env_reset(req)
        sid = uuid.uuid4().hex
        _sessions[sid] = (time.time(), state)
        return {
            "session_id": sid,
            "observation": obs.model_dump(),
            "task_type": "wordle_5_letter",
            "horizon": 6,
            "n_actions_per_step": "any 5-letter word",
        }

    @router.post("/wordle/step", tags=["wordle"])
    def wordle_step(req: StepRequest) -> dict:
        s = _sessions.get(req.session_id)
        if s is None:
            raise HTTPException(status_code=404,
                                detail="session_id not found or expired")
        _, state = s
        try:
            action = WordleAction(guess=req.guess.lower().strip())
        except Exception as e:  # noqa: BLE001
            return {
                "observation": None,
                "reward": -0.20,
                "components": {"format_gate_pydantic": -0.20},
                "defense": "pydantic_format_gate",
                "rejected": str(e)[:200],
            }
        new_state, obs, breakdown = env_step(state, action)
        _sessions[req.session_id] = (time.time(), new_state)
        return {
            "observation": obs.model_dump(),
            "reward": breakdown["reward"],
            "components": breakdown.get("components", {}),
            "defense": breakdown.get("defense"),
            "done": new_state.won or new_state.lost,
        }

    @router.post("/wordle/grade", tags=["wordle"])
    def wordle_grade(req: SessionRequest) -> dict:
        s = _sessions.get(req.session_id)
        if s is None:
            raise HTTPException(status_code=404,
                                detail="session_id not found or expired")
        _, state = s
        return env_grade(state)

    @router.get("/wordle/ui", include_in_schema=False)
    def wordle_ui():
        if HTMLResponse is None:
            raise HTTPException(status_code=500, detail="HTMLResponse unavailable")
        return HTMLResponse(_HTML)


_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Wordle RLVR · SupplyMind canonical demo</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body { background:#0a0e1a; color:#e8edf6; font-family:system-ui,sans-serif; }
  .tile { width:48px; height:48px; display:flex; align-items:center;
          justify-content:center; font-weight:700; text-transform:uppercase;
          margin:2px; border-radius:6px; font-size:20px; }
  .tile.green { background:#34d399; color:#0a0e1a; }
  .tile.yellow{ background:#fbbf24; color:#0a0e1a; }
  .tile.gray  { background:#3a3f4d; color:#aab3c5; }
  .tile.empty { border:1px solid #232a3d; }
  .row { display:flex; }
  input { background:#131826; border:1px solid #232a3d; color:#e8edf6;
          padding:8px 12px; border-radius:6px; width:120px; }
  button { background:linear-gradient(135deg,#22d3ee,#a78bfa);
            color:#0a0e1a; padding:8px 16px; border-radius:6px;
            font-weight:700; cursor:pointer; }
</style></head>
<body class="min-h-screen p-6">
<div class="max-w-2xl mx-auto">
  <h1 class="text-2xl font-bold mb-1">Wordle RLVR
    <span class="text-cyan-400">·</span>
    <span class="text-zinc-400 text-sm">SupplyMind canonical hackathon-guide demo</span>
  </h1>
  <p class="text-zinc-400 text-sm mb-4">
    OpenEnv-compliant. Multi-component reward (green/yellow credit + solve bonus).
    Anti-hack layers: format gate · dictionary gate · timeout. Programmatically
    verifiable (RLVR per Meta OpenEnv guide §11). GRPO-trainable via TRL.
  </p>
  <div id="board" class="my-3"></div>
  <div class="flex gap-2 items-end my-3">
    <input id="g" maxlength="5" placeholder="guess" autofocus>
    <button onclick="doGuess()">submit</button>
    <button onclick="reset()" style="background:#3a3f4d;color:#e8edf6">new game</button>
  </div>
  <div id="status" class="text-sm mt-2"></div>
  <pre id="dbg" class="text-xs text-zinc-500 mt-3 whitespace-pre-wrap"></pre>
</div>
<script>
let SID = null;
async function reset(){
  const r = await fetch('/wordle/reset',{method:'POST',headers:{'Content-Type':'application/json'},
    body: JSON.stringify({seed: Math.floor(Math.random()*1e9)})});
  const j = await r.json(); SID = j.session_id;
  document.getElementById('board').innerHTML = '';
  document.getElementById('status').textContent = 'new game · ' + j.observation.guesses_remaining + ' guesses left';
  document.getElementById('dbg').textContent = '';
}
async function doGuess(){
  const guess = document.getElementById('g').value;
  document.getElementById('g').value = '';
  const r = await fetch('/wordle/step',{method:'POST',headers:{'Content-Type':'application/json'},
    body: JSON.stringify({session_id: SID, guess})});
  const j = await r.json();
  if (j.rejected) {
    document.getElementById('status').innerHTML =
      '<span class="text-red-400">rejected by '+j.defense+': '+j.rejected+'</span>';
    return;
  }
  const fb = j.observation.last_feedback || [];
  const row = document.createElement('div');
  row.className = 'row';
  fb.forEach(f => {
    const t = document.createElement('div');
    t.className = 'tile ' + f.state;
    t.textContent = f.letter;
    row.appendChild(t);
  });
  document.getElementById('board').appendChild(row);
  let s = 'reward ' + j.reward.toFixed(3);
  if (j.observation.won) s = '<span class="text-emerald-400">WON · '+s+' · '+j.defense+'</span>';
  if (j.observation.lost) s = '<span class="text-red-400">LOST · target was '+j.observation.target_revealed+'</span>';
  s += ' · ' + j.observation.guesses_remaining + ' left';
  document.getElementById('status').innerHTML = s;
  document.getElementById('dbg').textContent =
    'components: ' + JSON.stringify(j.components, null, 2);
}
document.getElementById('g').addEventListener('keypress', e => {
  if (e.key === 'Enter') doGuess();
});
reset();
</script>
</body></html>"""
