"""
PFPA OpenEnv — Pre-Submission Validation Script
Run this before submitting to verify all OpenEnv spec requirements.

Usage:
    python validate.py [--url http://localhost:7860]
"""
import sys
import json
import argparse
import requests

def check(name: str, passed: bool, detail: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))
    return passed


def validate(url: str) -> bool:
    all_passed = True
    print(f"\n{'='*60}")
    print(f"PFPA OpenEnv Pre-Submission Validation")
    print(f"Target: {url}")
    print(f"{'='*60}\n")

    # ── 1. Health ──────────────────────────────────────────────────────────
    print("1. Health Check")
    try:
        r = requests.get(f"{url}/health", timeout=10)
        all_passed &= check("GET /health returns 200", r.status_code == 200, f"status={r.status_code}")
        data = r.json()
        all_passed &= check("Response contains 'status: ok'", data.get("status") == "ok")
    except Exception as e:
        all_passed &= check("GET /health reachable", False, str(e))

    # ── 2. Tasks ───────────────────────────────────────────────────────────
    print("\n2. Task Listing")
    try:
        r = requests.get(f"{url}/tasks", timeout=10)
        all_passed &= check("GET /tasks returns 200", r.status_code == 200)
        tasks = r.json().get("tasks", [])
        all_passed &= check("At least 3 tasks defined", len(tasks) >= 3, f"found {len(tasks)}")
        difficulties = {t["difficulty"] for t in tasks}
        all_passed &= check("Tasks span easy/medium/hard", {"easy","medium","hard"}.issubset(difficulties))
        for t in tasks:
            rr = t.get("reward_range", [])
            all_passed &= check(
                f"Task '{t['id']}' reward_range is [0.0, 1.0]",
                len(rr) == 2 and rr[0] == 0.0 and rr[1] == 1.0,
            )
    except Exception as e:
        all_passed &= check("GET /tasks parseable", False, str(e))

    # ── 3. Reset ───────────────────────────────────────────────────────────
    print("\n3. Reset — All Tasks")
    task_ids = ["signal_triage_easy", "workflow_optimization_medium", "crisis_management_hard"]
    state = None
    for task_id in task_ids:
        try:
            r = requests.post(f"{url}/reset", json={"task_id": task_id}, timeout=15)
            all_passed &= check(f"POST /reset task={task_id} returns 200", r.status_code == 200)
            s = r.json()
            all_passed &= check(f"  State has pending_signals", len(s.get("pending_signals", [])) > 0)
            all_passed &= check(f"  State has active_predictions", len(s.get("active_predictions", [])) > 0)
            all_passed &= check(f"  State.done is False initially", s.get("done") == False)
            if task_id == "signal_triage_easy":
                state = s
        except Exception as e:
            all_passed &= check(f"POST /reset {task_id}", False, str(e))

    # ── 4. State ───────────────────────────────────────────────────────────
    print("\n4. GET /state")
    try:
        r = requests.get(f"{url}/state", timeout=10)
        all_passed &= check("GET /state returns 200", r.status_code == 200)
        s = r.json()
        required_fields = ["task_id","step_number","max_steps","done","pending_signals",
                          "active_predictions","calendar_events","current_time","cumulative_reward"]
        for field in required_fields:
            all_passed &= check(f"  State has field '{field}'", field in s)
    except Exception as e:
        all_passed &= check("GET /state parseable", False, str(e))

    # ── 5. Step ────────────────────────────────────────────────────────────
    print("\n5. POST /step — Action Execution")
    try:
        # Reset easy task
        r = requests.post(f"{url}/reset", json={"task_id": "signal_triage_easy"}, timeout=15)
        state = r.json()
        preds = state.get("active_predictions", [])
        green_pred = next((p for p in preds if p.get("confidence_level") == "green"), None)
        red_pred   = next((p for p in preds if p.get("confidence_level") == "red"), None)

        if green_pred:
            r = requests.post(f"{url}/step",
                json={"action": {"action": "execute_prediction", "prediction_id": green_pred["id"]}},
                timeout=15)
            all_passed &= check("POST /step execute green prediction returns 200", r.status_code == 200)
            result = r.json()
            all_passed &= check("  Response has 'reward' field", "reward" in result)
            all_passed &= check("  Response has 'done' field", "done" in result)
            all_passed &= check("  Response has 'state' field", "state" in result)
            all_passed &= check("  Response has 'info' field", "info" in result)
            reward = result.get("reward", -1)
            all_passed &= check("  Reward in [0.0, 1.0]", 0.0 <= reward <= 1.0, f"reward={reward}")
            all_passed &= check("  Positive reward for green prediction", reward > 0, f"reward={reward}")

        if red_pred:
            r = requests.post(f"{url}/step",
                json={"action": {"action": "dismiss_prediction", "prediction_id": red_pred["id"]}},
                timeout=15)
            all_passed &= check("POST /step dismiss red prediction returns 200", r.status_code == 200)
            result = r.json()
            reward = result.get("reward", -1)
            all_passed &= check("  Positive reward for correct dismissal", reward > 0, f"reward={reward}")

        # Test noop
        r = requests.post(f"{url}/step", json={"action": {"action": "noop"}}, timeout=15)
        all_passed &= check("POST /step noop returns 200", r.status_code == 200)

        # Test invalid action
        r = requests.post(f"{url}/step", json={"action": {"action": "fly_to_mars"}}, timeout=15)
        all_passed &= check("POST /step invalid action handled gracefully", r.status_code == 200)

    except Exception as e:
        all_passed &= check("Step execution", False, str(e))

    # ── 6. Grader scores ───────────────────────────────────────────────────
    print("\n6. Grader Score Verification — All Tasks")
    for task_id in task_ids:
        try:
            requests.post(f"{url}/reset", json={"task_id": task_id}, timeout=15)
            # Execute noop and check score is in range
            r = requests.post(f"{url}/step", json={"action": {"action": "noop"}}, timeout=15)
            result = r.json()
            reward = result.get("reward", -1)
            all_passed &= check(
                f"Task '{task_id}' reward in [0.0, 1.0]",
                0.0 <= reward <= 1.0,
                f"reward={reward}"
            )
        except Exception as e:
            all_passed &= check(f"Task '{task_id}' grader", False, str(e))

    # ── 7. openenv.yaml ────────────────────────────────────────────────────
    print("\n7. openenv.yaml")
    import os
    yaml_exists = os.path.exists("openenv.yaml")
    all_passed &= check("openenv.yaml exists", yaml_exists)
    if yaml_exists:
        with open("openenv.yaml") as f:
            content = f.read()
        for key in ["name", "tasks", "observation_space", "action_space", "api"]:
            all_passed &= check(f"  openenv.yaml has '{key}'", key in content)

    # ── 8. inference.py ────────────────────────────────────────────────────
    print("\n8. inference.py")
    inf_exists = os.path.exists("inference.py")
    all_passed &= check("inference.py exists in root", inf_exists)
    if inf_exists:
        with open("inference.py") as f:
            inf_content = f.read()
        for marker in ["[START]", "[STEP]", "[END]", "API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]:
            all_passed &= check(f"  inference.py contains '{marker}'", marker in inf_content)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL CHECKS PASSED — Ready to submit!")
    else:
        print("⚠️  SOME CHECKS FAILED — Fix before submitting.")
    print(f"{'='*60}\n")
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:7860", help="Environment URL")
    args = parser.parse_args()
    ok = validate(args.url)
    sys.exit(0 if ok else 1)
