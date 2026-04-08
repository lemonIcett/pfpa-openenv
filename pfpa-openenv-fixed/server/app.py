"""
PFPA OpenEnv — FastAPI Server
Implements the full OpenEnv spec: /reset, /step, /state, /tasks, /health
Also serves a plain HTML web UI at /web for manual testing (no Gradio needed).
"""
from __future__ import annotations
import os
import json
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from environment.models import (
    ResetRequest, StepRequest, StepResponse, TasksResponse, TaskInfo,
    EnvironmentState
)
from environment.scenarios import build_scenario, TASK_CATALOG
from environment.grader import PFPAGrader

app = FastAPI(
    title="PFPA OpenEnv",
    description=(
        "A real-world personal productivity AI environment for reinforcement learning. "
        "The agent learns to triage context signals, manage predictions, and automate workflows "
        "— mirroring the PFPA (Prompt-Free Proactive AI) system."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory environment state (single-session per container)
_state: EnvironmentState | None = None
_grader = PFPAGrader()


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "pfpa-productivity-env", "version": "1.0.0"}


# ── Tasks ──────────────────────────────────────────────────────────────────────

@app.get("/tasks", response_model=TasksResponse)
def list_tasks():
    """List all available tasks with their difficulty and description."""
    return TasksResponse(tasks=list(TASK_CATALOG.values()))


# ── Reset ──────────────────────────────────────────────────────────────────────

@app.post("/reset", response_model=EnvironmentState)
def reset(body: ResetRequest = None):
    global _state
    task = (body.task_id if body and body.task_id else "signal_triage_easy")
    seed = random.randint(0, 2**31)
    try:
        _state = build_scenario(task, seed=seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _state


# ── Step ───────────────────────────────────────────────────────────────────────

@app.post("/step", response_model=StepResponse)
def step(body: StepRequest):
    global _state
    if _state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call POST /reset first.")
    if _state.done:
        raise HTTPException(status_code=400, detail="Episode is done. Call POST /reset to start a new episode.")
    reward, done, info = _grader.grade(_state, body.action)
    return StepResponse(state=_state, reward=reward, done=done, info=info)


# ── State ──────────────────────────────────────────────────────────────────────

@app.get("/state", response_model=EnvironmentState)
def get_state():
    if _state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call POST /reset first.")
    return _state


# ── Root ───────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "PFPA OpenEnv",
        "description": "Personal Productivity AI — Real-world RL environment",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state", "/web"],
        "tasks": list(TASK_CATALOG.keys()),
    }


# ── Web UI (pure HTML — no Gradio dependency) ──────────────────────────────────

_TASK_OPTIONS = "".join(
    f'<option value="{k}">{k}</option>' for k in TASK_CATALOG.keys()
)

_WEB_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PFPA OpenEnv — Test UI</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#f5f5f5;color:#222;padding:2rem}
h1{font-size:1.4rem;font-weight:600;margin-bottom:.25rem}
p.sub{color:#666;font-size:.9rem;margin-bottom:2rem}
.tabs{display:flex;gap:.5rem;margin-bottom:1.5rem}
.tab-btn{padding:.5rem 1.2rem;border:1px solid #ccc;border-radius:6px;background:#fff;cursor:pointer;font-size:.9rem}
.tab-btn.active{background:#1a1a1a;color:#fff;border-color:#1a1a1a}
.panel{display:none;background:#fff;border:1px solid #ddd;border-radius:8px;padding:1.5rem;max-width:800px}
.panel.active{display:block}
label{font-size:.85rem;font-weight:500;display:block;margin-bottom:.4rem;color:#444}
select,textarea{width:100%;padding:.5rem .7rem;border:1px solid #ccc;border-radius:6px;font-size:.9rem;font-family:inherit;margin-bottom:1rem}
textarea{font-family:monospace;resize:vertical}
button.run{background:#1a1a1a;color:#fff;border:none;padding:.55rem 1.4rem;border-radius:6px;cursor:pointer;font-size:.9rem}
button.run:hover{background:#333}
.out-label{font-size:.85rem;font-weight:500;color:#444;margin-top:1rem;margin-bottom:.4rem}
pre{background:#f0f0f0;border:1px solid #ddd;border-radius:6px;padding:1rem;font-size:.82rem;overflow-x:auto;white-space:pre-wrap;word-break:break-word;min-height:60px}
.badge{display:inline-block;font-size:.75rem;padding:2px 8px;border-radius:20px;margin-left:.5rem}
.ok{background:#d4edda;color:#155724}.err{background:#f8d7da;color:#721c24}
</style>
</head>
<body>
<h1>PFPA OpenEnv <span id="status-badge" class="badge">checking...</span></h1>
<p class="sub">Manual test UI — Reset, Step, and inspect State without writing any code.</p>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('reset')">Reset</button>
  <button class="tab-btn" onclick="switchTab('step')">Step</button>
  <button class="tab-btn" onclick="switchTab('state')">Get State</button>
</div>

<div id="tab-reset" class="panel active">
  <label for="task-select">Task ID</label>
  <select id="task-select">TASK_OPTIONS_PLACEHOLDER</select>
  <button class="run" onclick="doReset()">Reset Environment</button>
  <div class="out-label">Response</div>
  <pre id="reset-out">—</pre>
</div>

<div id="tab-step" class="panel">
  <label for="action-input">Action JSON</label>
  <textarea id="action-input" rows="8">{
  "action": "execute_prediction",
  "prediction_id": "pred-001"
}</textarea>
  <button class="run" onclick="doStep()">Send Step</button>
  <div class="out-label">Response (reward / done / info)</div>
  <pre id="step-out">—</pre>
</div>

<div id="tab-state" class="panel">
  <button class="run" onclick="doState()">Fetch Current State</button>
  <div class="out-label">Current State</div>
  <pre id="state-out">—</pre>
</div>

<script>
function switchTab(name){
  var tabs=['reset','step','state'];
  document.querySelectorAll('.tab-btn').forEach(function(b,i){b.classList.toggle('active',tabs[i]===name);});
  document.querySelectorAll('.panel').forEach(function(p){p.classList.remove('active');});
  document.getElementById('tab-'+name).classList.add('active');
}
async function doReset(){
  var pre=document.getElementById('reset-out');
  pre.textContent='Loading...';
  try{
    var r=await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:document.getElementById('task-select').value})});
    pre.textContent=JSON.stringify(await r.json(),null,2);
  }catch(e){pre.textContent='Error: '+e;}
}
async function doStep(){
  var pre=document.getElementById('step-out');
  var payload;
  try{payload=JSON.parse(document.getElementById('action-input').value);}
  catch(e){pre.textContent='Invalid JSON: '+e;return;}
  pre.textContent='Loading...';
  try{
    var r=await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:payload})});
    pre.textContent=JSON.stringify(await r.json(),null,2);
  }catch(e){pre.textContent='Error: '+e;}
}
async function doState(){
  var pre=document.getElementById('state-out');
  pre.textContent='Loading...';
  try{
    var r=await fetch('/state');
    pre.textContent=JSON.stringify(await r.json(),null,2);
  }catch(e){pre.textContent='Error: '+e;}
}
(async function(){
  var badge=document.getElementById('status-badge');
  try{
    var r=await fetch('/health');
    var d=await r.json();
    if(d.status==='ok'){badge.textContent='live';badge.className='badge ok';}
    else{badge.textContent='error';badge.className='badge err';}
  }catch(e){badge.textContent='offline';badge.className='badge err';}
})();
</script>
</body>
</html>"""

_WEB_HTML = _WEB_HTML.replace("TASK_OPTIONS_PLACEHOLDER", _TASK_OPTIONS)


@app.get("/web", response_class=HTMLResponse)
def web_ui():
    """Plain HTML test UI — no Gradio required."""
    return HTMLResponse(content=_WEB_HTML)
