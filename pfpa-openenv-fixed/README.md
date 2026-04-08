---
title: PFPA OpenEnv
emoji: 🧠
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# PFPA OpenEnv — Personal Productivity AI Environment

[![OpenEnv Spec](https://img.shields.io/badge/OpenEnv-1.0-blue)](https://github.com/raun/openenv-course)
[![Python 3.11](https://img.shields.io/badge/python-3.11-green)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)](https://fastapi.tiangolo.com)

## Overview

PFPA OpenEnv is a **real-world reinforcement learning environment** based on the PFPA (Prompt-Free Proactive AI) desktop application — a system that monitors context signals (calendar events, emails, Slack messages, file activity) and takes proactive actions on behalf of the user.

An AI agent in this environment must learn to:
- **Triage** incoming context signals by urgency and relevance
- **Execute** correct AI-generated predictions (schedule meetings, draft emails, send messages)
- **Dismiss** low-confidence trap predictions to avoid false positives
- **Create workflow rules** for recurring automation patterns
- **Resolve** calendar conflicts and prioritize competing tasks

This mirrors real-world executive AI assistant behavior. Signals and predictions are **shuffled on every `reset()` call** so the agent must learn generalizable triage strategies rather than memorising a fixed sequence.

---

## Environment Specification

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Health check — returns 200 if live |
| `GET`  | `/tasks` | List all available tasks |
| `POST` | `/reset` | Reset to a new episode `{"task_id": "..."}` |
| `POST` | `/step`  | Apply action `{"action": {...}}` |
| `GET`  | `/state` | Get current state without advancing |
| `GET`  | `/`      | API overview |

### Observation Space

```json
{
  "task_id": "signal_triage_easy",
  "step_number": 3,
  "max_steps": 10,
  "done": false,
  "pending_signals": [...],       // ContextSignal[]
  "active_predictions": [...],    // Prediction[] — agent must decide execute/dismiss
  "calendar_events": [...],       // CalendarEvent[] — may have is_conflicting=true
  "recent_actions": [...],        // ActionRecord[] — last 10 actions taken
  "workflow_rules": [...],        // WorkflowRule[] — automation rules created
  "current_time": "2026-04-05T10:00:00+05:30",
  "user_context": {"timezone": "Asia/Kolkata", "work_hours_start": 9, ...},
  "cumulative_reward": 0.45
}
```

### Action Space (Categorical)

| Action | Required Params | Reward |
|--------|----------------|--------|
| `execute_prediction` | `prediction_id` | +0.1–0.5 (green) / −0.1–0.3 (red trap) |
| `dismiss_prediction` | `prediction_id` | +0.12 (red) / −0.1 (green missed) |
| `create_calendar_event` | `title`, `start_time` | +0.08–0.15 |
| `draft_email` | `to`, `subject` | +0.07–0.15 |
| `send_slack_message` | `channel`, `message` | +0.06–0.15 |
| `set_workflow_rule` | `trigger_type`, `condition`, `action_template` | +0.20 |
| `snooze_signal` | `signal_id` | +0.03 |
| `noop` | — | 0.0 / −0.05 if urgent signals pending |

---

## Tasks

### Task 1: Signal Triage — Easy (`signal_triage_easy`)
- **Signals:** 5 (calendar, email, Slack)
- **Max Steps:** 10
- **Goal:** Execute the 2 green-confidence predictions; dismiss the 1 red-confidence trap
- **Reward Range:** 0.0–1.0

### Task 2: Workflow Optimization — Medium (`workflow_optimization_medium`)
- **Signals:** 12 (mixed urgency across all signal types)
- **Max Steps:** 20
- **Goal:** Resolve calendar conflict, unblock dev team, escalate production issue, create 1 workflow rule
- **Reward Range:** 0.0–1.0

### Task 3: Crisis Management — Hard (`crisis_management_hard`)
- **Signals:** 25+ (signal storm with trap predictions mixed in)
- **Max Steps:** 40
- **Goal:** Correctly prioritize 7 critical actions, dismiss 3 traps, create workflow automation
- **Reward Range:** 0.0–1.0

---

## Reward Function

Rewards are designed with **partial progress signals** — every correct decision earns reward, not just episode completion:

- **Correct execution** of a green prediction: `+reward_if_correct` (0.15–0.50)
- **Correct dismissal** of a red prediction: `+0.12`
- **False positive** (executing a red prediction): `-penalty_if_wrong` (0.15–0.30)
- **Missed opportunity** (dismissing a green prediction): `-penalty * 0.5`
- **Workflow rule creation**: `+0.20` (forward-thinking behavior)
- **NOOP with urgent signals**: `-0.05 per urgent signal` (opportunity cost)

Final per-step reward is clipped to `[0.0, 1.0]`. Cumulative reward is cumulative, capped at `1.0`.

---

## Quick Start

### Option A — uv (recommended)

```bash
# Clone from HF Spaces
git clone https://huggingface.co/spaces/YOUR_USERNAME/pfpa-openenv
cd pfpa-openenv

# Install dependencies and run server
uv sync
uv run uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### Option B — pip

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### Option C — Docker

```bash
docker build -t pfpa-openenv .
docker run -p 7860:7860 pfpa-openenv
```

### Run Baseline Inference

```bash
API_BASE_URL="https://router.huggingface.co/v1" \
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
HF_TOKEN="hf_..." \
ENV_URL="http://localhost:7860" \
python inference.py
```

### Quick Test

```bash
# Health check
curl http://localhost:7860/health

# List tasks
curl http://localhost:7860/tasks

# Reset to easy task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "signal_triage_easy"}'

# Execute a prediction
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action": "execute_prediction", "prediction_id": "pred-001"}}'
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | HF inference router URL (`https://router.huggingface.co/v1`) |
| `MODEL_NAME` | Yes | HF-hosted model identifier (e.g. `meta-llama/Llama-3.1-8B-Instruct`) |
| `HF_TOKEN` | Yes | Your Hugging Face token (`hf_...`) |
| `ENV_URL` | No | OpenEnv server URL (default: `http://localhost:7860`) |

---

## Project Structure

```
pfpa-openenv/
├── app.py                    # FastAPI application (OpenEnv API)
├── inference.py              # Baseline LLM agent (required by spec)
├── openenv.yaml              # OpenEnv spec configuration
├── pyproject.toml            # Python package + uv dependency management
├── requirements.txt          # pip-compatible dependency list
├── Dockerfile
├── README.md
└── environment/
    ├── __init__.py
    ├── models.py             # Pydantic typed models (all domain types)
    ├── client.py             # OpenEnv EnvClient implementation
    ├── scenarios.py          # Task scenarios (easy/medium/hard) with shuffle
    └── grader.py             # Reward logic and episode grading
```

---

## Connection to PFPA Application

This environment is derived from the **PFPA v2** (Prompt-Free Proactive AI) desktop application — a real Electron + React app that:

- Monitors 5 signal types in real-time (calendar, email, Slack, files, browser)
- Generates AI-powered action predictions using Claude
- Executes approved actions automatically (calendar events, email drafts, Slack messages)
- Tracks prediction accuracy and builds user behavior models

The OpenEnv environment faithfully simulates the decision-making challenge at the core of PFPA: given a stream of context signals and a set of AI predictions, what actions maximize user productivity while minimizing automation errors?

---

## Baseline Results

| Task | Score | Steps Used |
|------|-------|-----------|
| Signal Triage (Easy) | 0.74 | 5/10 |
| Workflow Optimization (Medium) | 0.61 | 14/20 |
| Crisis Management (Hard) | 0.48 | 28/40 |
| **Average** | **0.61** | — |

*Baseline uses `meta-llama/Llama-3.1-8B-Instruct` via HF inference router with zero-shot chain-of-thought prompting.*

---

## License

MIT
