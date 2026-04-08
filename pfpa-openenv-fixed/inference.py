# """
# PFPA OpenEnv — inference.py
# Baseline inference script using the Hugging Face Inference Router via OpenAI-compatible client.

# Required env vars:
#   API_BASE_URL  = https://router.huggingface.co/v1   (HF inference router)
#   MODEL_NAME    = meta-llama/Llama-3.1-8B-Instruct   (any HF-hosted model)
#   HF_TOKEN      = your Hugging Face token (hf_...)
#   ENV_URL       = http://localhost:7860              (or your HF Space URL)

# Usage:
#     API_BASE_URL="https://router.huggingface.co/v1" \
#     MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
#     HF_TOKEN="hf_..." \
#     ENV_URL="http://localhost:7860" \
#     python inference.py
# """
# from __future__ import annotations
# import os
# import sys
# import json
# import time
# import requests
# from openai import OpenAI

# # ── Configuration ──────────────────────────────────────────────────────────────
# API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
# MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
# HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
# ENV_URL = os.environ.get("ENV_URL", "https://ravenscroft-PFPA.hf.space")

# TASKS = [
#     "signal_triage_easy",
#     "workflow_optimization_medium",
#     "crisis_management_hard",
# ]

# # ── OpenAI-compatible client pointed at HF router ──────────────────────────────
# client = OpenAI(
#     base_url=API_BASE_URL,
#     api_key=HF_TOKEN or "placeholder",
# )

# SYSTEM_PROMPT = """You are an AI agent operating a personal productivity environment (PFPA OpenEnv).

# Your goal: observe the environment state and take the best action to maximize reward.

# ACTIONS you can take:
# - execute_prediction: Execute a pending prediction (use for green/high-confidence ones)
# - dismiss_prediction: Dismiss a prediction (use for red/low-confidence trap predictions)
# - create_calendar_event: Create a calendar event (requires: title, start_time)
# - draft_email: Draft an email (requires: to, subject)
# - send_slack_message: Send a Slack message (requires: channel, message)
# - set_workflow_rule: Create automation rule (requires: trigger_type, condition, action_template)
# - snooze_signal: Snooze a signal (requires: signal_id)
# - noop: Do nothing (only if nothing urgent)

# STRATEGY:
# 1. Execute green-confidence (>=75%) predictions first, especially for urgent signals
# 2. Dismiss red-confidence (<45%) predictions — they are TRAPS, never execute them
# 3. Resolve calendar conflicts by creating new events
# 4. Create workflow rules for recurring alert patterns (big reward bonus)
# 5. Prioritize: system health > revenue risk > legal > scheduling

# Respond with ONLY a valid JSON object, no markdown fences, no explanation:
# {
#   "action": "<action_name>",
#   "prediction_id": "<id or null>",
#   "signal_id": "<id or null>",
#   "title": "<string or null>",
#   "start_time": "<ISO8601 or null>",
#   "duration_minutes": null,
#   "attendees": null,
#   "to": "<email or null>",
#   "subject": "<string or null>",
#   "body_outline": null,
#   "channel": "<string or null>",
#   "message": "<string or null>",
#   "trigger_type": "<signal_type or null>",
#   "condition": "<string or null>",
#   "action_template": "<string or null>",
#   "snooze_minutes": null
# }"""


# def call_llm(state: dict) -> dict:
#     """Call the LLM with the current environment state and get an action."""
#     pending_preds = [p for p in state.get("active_predictions", []) if p.get("status") == "pending"]
#     urgent_signals = [s for s in state.get("pending_signals", []) if s.get("urgency") == "high" and not s.get("processed")]
#     conflicts = [e for e in state.get("calendar_events", []) if e.get("is_conflicting")]

#     summary = {
#         "task_id": state.get("task_id"),
#         "step": state.get("step_number"),
#         "cumulative_reward": state.get("cumulative_reward"),
#         "pending_predictions": [
#             {
#                 "id": p["id"],
#                 "description": p["description"],
#                 "confidence": p["confidence"],
#                 "confidence_level": p["confidence_level"],
#                 "action_type": p["action_type"],
#             }
#             for p in pending_preds[:8]
#         ],
#         "urgent_signals": [
#             {"id": s["id"], "title": s["title"], "signal_type": s["signal_type"]}
#             for s in urgent_signals[:5]
#         ],
#         "calendar_conflicts": [
#             {"id": e["id"], "title": e["title"], "start_time": e["start_time"]}
#             for e in conflicts[:3]
#         ],
#         "workflow_rules_count": len(state.get("workflow_rules", [])),
#     }

#     user_msg = f"Current environment state:\n{json.dumps(summary, indent=2)}\n\nChoose the single best action right now."

#     response = client.chat.completions.create(
#         model=MODEL_NAME,
#         max_tokens=300,
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user",   "content": user_msg},
#         ],
#     )
#     raw = response.choices[0].message.content.strip()

#     # Strip markdown code fences if present
#     if "```" in raw:
#         parts = raw.split("```")
#         for part in parts:
#             part = part.strip()
#             if part.startswith("json"):
#                 part = part[4:].strip()
#             try:
#                 return json.loads(part)
#             except Exception:
#                 continue
#     return json.loads(raw)


# def run_task(task_id: str) -> dict:
#     """Run one full episode for a task. Returns score info."""
#     reset_resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
#     reset_resp.raise_for_status()
#     state = reset_resp.json()

#     # [START] — one per task, emitted here only (not duplicated in main)
#     print(json.dumps({
#         "event": "[START]",
#         "task_id": task_id,
#         "max_steps": state["max_steps"],
#         "signals": len(state["pending_signals"]),
#         "predictions": len(state["active_predictions"]),
#     }), flush=True)

#     total_reward = 0.0
#     steps = []

#     for step_num in range(state["max_steps"]):
#         if state.get("done"):
#             break

#         try:
#             action = call_llm(state)
#         except Exception as e:
#             print(f"[WARN] LLM error at step {step_num}: {e}", file=sys.stderr)
#             action = {"action": "noop"}

#         try:
#             step_resp = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
#             step_resp.raise_for_status()
#         except Exception as e:
#             print(f"[WARN] Step error at step {step_num}: {e}", file=sys.stderr)
#             break

#         result       = step_resp.json()
#         reward       = result["reward"]
#         done         = result["done"]
#         info         = result["info"]
#         state        = result["state"]
#         total_reward = state["cumulative_reward"]

#         log_entry = {
#             "event": "[STEP]",
#             "task_id": task_id,
#             "step": step_num + 1,
#             "action": action.get("action"),
#             "target": action.get("prediction_id") or action.get("signal_id") or "",
#             "reward": reward,
#             "cumulative_reward": total_reward,
#             "outcome": info.get("outcome", ""),
#             "done": done,
#         }
#         print(json.dumps(log_entry), flush=True)
#         steps.append(log_entry)

#         if done:
#             break
#         time.sleep(0.3)

#     # [TASK_END] — one per task (distinct from the run-level [END] in main)
#     print(json.dumps({
#         "event": "[TASK_END]",
#         "task_id": task_id,
#         "total_steps": len(steps),
#         "final_score": total_reward,
#         "max_possible": 1.0,
#         "normalized_score": round(total_reward, 4),
#     }), flush=True)

#     return {"task_id": task_id, "score": total_reward, "steps": len(steps)}


# def main():
#     results = []
#     for task_id in TASKS:
#         try:
#             result = run_task(task_id)
#             results.append(result)
#         except Exception as e:
#             print(f"[ERROR] Task {task_id} failed: {e}", file=sys.stderr)
#             results.append({"task_id": task_id, "score": 0.0, "error": str(e)})
#         time.sleep(1)

#     avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
#     print(json.dumps({
#         "event": "[END]",
#         "run": "pfpa-openenv-baseline",
#         "model": MODEL_NAME,
#         "env_url": ENV_URL,
#         "results": results,
#         "average_score": round(avg_score, 4),
#         "tasks_completed": len(results),
#     }), flush=True)


# if __name__ == "__main__":
#     main()
# | Before (JSON)                                         | After (Plain Text)                                                            |
# | ----------------------------------------------------- | ----------------------------------------------------------------------------- |
# | `{"event": "[START]", "task_id": "X", ...}`           | `[START] task=X max_steps=10 signals=5 predictions=3`                         |
# | `{"event": "[STEP]", "task_id": "X", "step": 1, ...}` | `[STEP] task=X step=1 action=Y target=Z reward=0.5 cumulative=0.5 done=false` |
# | `{"event": "[END]", ...}`                             | `[END] task=X score=1.0 steps=3`                                              |


"""
PFPA OpenEnv — inference.py
Baseline inference script using the Hugging Face Inference Router via OpenAI-compatible client.
"""
from __future__ import annotations
import os
import sys
import json
import time
import requests
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "https://ravenscroft-PFPA.hf.space")

TASKS = [
    "signal_triage_easy",
    "workflow_optimization_medium",
    "crisis_management_hard",
]

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "placeholder")

SYSTEM_PROMPT = """You are an AI agent operating a personal productivity environment (PFPA OpenEnv).
Your goal: observe the environment state and take the best action to maximize reward.
ACTIONS you can take:
- execute_prediction: Execute a pending prediction (use for green/high-confidence ones)
- dismiss_prediction: Dismiss a prediction (use for red/low-confidence trap predictions)
- create_calendar_event: Create a calendar event (requires: title, start_time)
- draft_email: Draft an email (requires: to, subject)
- send_slack_message: Send a Slack message (requires: channel, message)
- set_workflow_rule: Create automation rule (requires: trigger_type, condition, action_template)
- snooze_signal: Snooze a signal (requires: signal_id)
- noop: Do nothing (only if nothing urgent)

STRATEGY:
1. Execute green-confidence (>=75%) predictions first, especially for urgent signals
2. Dismiss red-confidence (<45%) predictions — they are TRAPS, never execute them
3. Resolve calendar conflicts by creating new events
4. Create workflow rules for recurring alert patterns (big reward bonus)
5. Prioritize: system health > revenue risk > legal > scheduling

Respond with ONLY a valid JSON object, no markdown fences, no explanation:
{
  "action": "<action_name>",
  "prediction_id": "<id or null>",
  "signal_id": "<id or null>",
  "title": "<string or null>",
  "start_time": "<ISO8601 or null>",
  "duration_minutes": null,
  "attendees": null,
  "to": "<email or null>",
  "subject": "<string or null>",
  "body_outline": null,
  "channel": "<string or null>",
  "message": "<string or null>",
  "trigger_type": "<signal_type or null>",
  "condition": "<string or null>",
  "action_template": "<string or null>",
  "snooze_minutes": null
}"""


def call_llm(state: dict) -> dict:
    """Call the LLM with the current environment state and get an action."""
    pending_preds = [p for p in state.get("active_predictions", []) if p.get("status") == "pending"]
    urgent_signals = [s for s in state.get("pending_signals", []) if s.get("urgency") == "high" and not s.get("processed")]
    conflicts = [e for e in state.get("calendar_events", []) if e.get("is_conflicting")]

    summary = {
        "task_id": state.get("task_id"),
        "step": state.get("step_number"),
        "cumulative_reward": state.get("cumulative_reward"),
        "pending_predictions": [
            {
                "id": p["id"],
                "description": p["description"],
                "confidence": p["confidence"],
                "confidence_level": p["confidence_level"],
                "action_type": p["action_type"],
            }
            for p in pending_preds[:8]
        ],
        "urgent_signals": [
            {"id": s["id"], "title": s["title"], "signal_type": s["signal_type"]}
            for s in urgent_signals[:5]
        ],
        "calendar_conflicts": [
            {"id": e["id"], "title": e["title"], "start_time": e["start_time"]}
            for e in conflicts[:3]
        ],
        "workflow_rules_count": len(state.get("workflow_rules", [])),
    }

    user_msg = f"Current environment state:\n{json.dumps(summary, indent=2)}\n\nChoose the single best action right now."

    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=300,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
    )
    raw = response.choices[0].message.content.strip()

    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except Exception:
                continue
    return json.loads(raw)


def run_task(task_id: str) -> dict:
    """Run one full episode for a task. Returns score info."""
    reset_resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    reset_resp.raise_for_status()
    state = reset_resp.json()

    # [START] — PLAIN TEXT FORMAT
    print(f"[START] task={task_id} max_steps={state['max_steps']} signals={len(state['pending_signals'])} predictions={len(state['active_predictions'])}", flush=True)

    total_reward = 0.0
    step_count = 0

    for step_num in range(state["max_steps"]):
        if state.get("done"):
            break

        try:
            action = call_llm(state)
        except Exception as e:
            print(f"[WARN] LLM error at step {step_num}: {e}", file=sys.stderr, flush=True)
            action = {"action": "noop"}

        try:
            step_resp = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
            step_resp.raise_for_status()
        except Exception as e:
            print(f"[WARN] Step error at step {step_num}: {e}", file=sys.stderr, flush=True)
            break

        result       = step_resp.json()
        reward       = result["reward"]
        done         = result["done"]
        state        = result["state"]
        total_reward = state["cumulative_reward"]
        step_count   = step_num + 1

        target = action.get("prediction_id") or action.get("signal_id") or ""
        
        # [STEP] — PLAIN TEXT FORMAT
        print(f"[STEP] task={task_id} step={step_count} action={action.get('action')} target={target} reward={reward} cumulative={total_reward} done={done}", flush=True)

        if done:
            break
        time.sleep(0.3)

    # [END] task — PLAIN TEXT FORMAT
    print(f"[END] task={task_id} score={total_reward} steps={step_count}", flush=True)

    return {"task_id": task_id, "score": total_reward, "steps": step_count}


def main():
    results = []
    for task_id in TASKS:
        try:
            result = run_task(task_id)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}", file=sys.stderr, flush=True)
            results.append({"task_id": task_id, "score": 0.0, "error": str(e)})
        time.sleep(1)

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"[SUMMARY] average_score={avg_score:.4f} tasks_completed={len(results)}", flush=True)


if __name__ == "__main__":
    main()