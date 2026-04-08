"""
PFPA OpenEnv — One-Click HuggingFace Spaces Deployer
Run this ONCE on your local machine to:
  1. Create the HF Space
  2. Upload all files
  3. Set API secrets
  4. Verify the deployment is live

Usage:
    pip install huggingface_hub requests
    python deploy_to_hf.py
"""
import os
import sys
import time
import requests
from pathlib import Path
from huggingface_hub import HfApi, SpaceStage

# ── CREDENTIALS — loaded from environment variables ────────────────────────────
# Copy .env.example to .env, fill in your values, then run:
#   export $(cat .env | xargs) && python deploy_to_hf.py
HF_USERNAME  = os.environ.get("HF_USERNAME", "")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
SPACE_NAME   = os.environ.get("SPACE_NAME", "pfpa-openenv")
REPO_ID      = f"{HF_USERNAME}/{SPACE_NAME}"

# HuggingFace inference router (free — no OpenAI key needed)
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

if not HF_USERNAME or not HF_TOKEN:
    print("❌  Missing credentials. Set HF_USERNAME and HF_TOKEN in your .env file.")
    sys.exit(1)

# ── FILES TO UPLOAD ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent

FILES = [
    "app.py",
    "inference.py",
    "validate.py",
    "openenv.yaml",
    "requirements.txt",
    "Dockerfile",
    "README.md",
    ".env.example",
    "environment/__init__.py",
    "environment/client.py",
    "environment/models.py",
    "environment/scenarios.py",
    "environment/grader.py",
]


def step(msg):
    print(f"\n{'─'*55}")
    print(f"  {msg}")
    print(f"{'─'*55}")


def ok(msg):   print(f"  ✅  {msg}")
def fail(msg): print(f"  ❌  {msg}"); sys.exit(1)
def info(msg): print(f"  ℹ️   {msg}")


def main():
    api = HfApi(token=HF_TOKEN)

    # ── 1. Verify token ────────────────────────────────────────────────────────
    step("1 / 5  Verifying HuggingFace token")
    try:
        user = api.whoami()
        ok(f"Logged in as: {user['name']}")
    except Exception as e:
        fail(f"Token invalid or network error: {e}")

    # ── 2. Create Space ────────────────────────────────────────────────────────
    step("2 / 5  Creating HuggingFace Space")
    try:
        api.create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True,
        )
        ok(f"Space ready: https://huggingface.co/spaces/{REPO_ID}")
    except Exception as e:
        fail(f"Failed to create space: {e}")

    # ── 3. Upload files ────────────────────────────────────────────────────────
    step("3 / 5  Uploading files")
    for rel_path in FILES:
        local = SCRIPT_DIR / rel_path
        if not local.exists():
            fail(f"Missing file: {local}")
        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=rel_path,
                repo_id=REPO_ID,
                repo_type="space",
            )
            ok(f"Uploaded: {rel_path}")
        except Exception as e:
            fail(f"Upload failed for {rel_path}: {e}")

    # ── 4. Set secrets ─────────────────────────────────────────────────────────
    step("4 / 5  Setting Space secrets")
    secrets = {
        "HF_TOKEN":     HF_TOKEN,
        "API_BASE_URL": API_BASE_URL,
        "MODEL_NAME":   MODEL_NAME,
        "ENV_URL":      f"https://{HF_USERNAME}-{SPACE_NAME}.hf.space",
    }
    for key, value in secrets.items():
        try:
            api.add_space_secret(repo_id=REPO_ID, key=key, value=value)
            ok(f"Secret set: {key}")
        except Exception as e:
            info(f"Could not set {key} via SDK ({e}) — set manually in Space settings")

    # ── 5. Wait for build & verify ─────────────────────────────────────────────
    step("5 / 5  Waiting for Space to build (this takes ~60–90 seconds)")
    space_url = f"https://{HF_USERNAME}-{SPACE_NAME}.hf.space"
    health_url = f"{space_url}/health"

    for attempt in range(24):  # up to ~4 minutes
        try:
            r = requests.get(health_url, timeout=10)
            if r.status_code == 200 and r.json().get("status") == "ok":
                ok(f"Space is LIVE: {space_url}")
                break
        except Exception:
            pass
        dots = "." * ((attempt % 3) + 1)
        print(f"  ⏳  Building{dots} ({attempt * 10}s)", end="\r", flush=True)
        time.sleep(10)
    else:
        info("Build taking longer than expected — check build logs at:")
        info(f"  https://huggingface.co/spaces/{REPO_ID}")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print("  🎉  DEPLOYMENT COMPLETE")
    print(f"{'═'*55}")
    print(f"  Space URL  :  {space_url}")
    print(f"  Health     :  {space_url}/health")
    print(f"  Tasks      :  {space_url}/tasks")
    print(f"  HF Page    :  https://huggingface.co/spaces/{REPO_ID}")
    print()
    print("  ► PASTE THIS URL on the hackathon dashboard:")
    print(f"     {space_url}")
    print(f"{'═'*55}\n")

    # ── Quick smoke test ───────────────────────────────────────────────────────
    print("  Running quick smoke test...")
    try:
        r = requests.post(f"{space_url}/reset", json={"task_id": "signal_triage_easy"}, timeout=30)
        assert r.status_code == 200
        state = r.json()
        assert len(state["pending_signals"]) > 0
        ok(f"POST /reset → signals={len(state['pending_signals'])}, preds={len(state['active_predictions'])}")

        r = requests.post(f"{space_url}/step",
            json={"action": {"action": "execute_prediction", "prediction_id": "pred-001"}},
            timeout=30)
        assert r.status_code == 200
        result = r.json()
        ok(f"POST /step  → reward={result['reward']} outcome={result['info'].get('outcome')}")
        print()
        print("  ✅  All checks passed — you are READY TO SUBMIT!")
    except Exception as e:
        info(f"Smoke test failed: {e} — but Space may still be building. Try again in 1 min.")


if __name__ == "__main__":
    main()
