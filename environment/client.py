"""
PFPA OpenEnv — Client
Typed environment client for connecting to a running PFPA OpenEnv server.

Usage (sync):
    from environment.client import PFPAEnvClient
    from environment.models import ActionPayload

    with PFPAEnvClient(base_url="http://localhost:7860").sync() as env:
        state = env.reset(task_id="signal_triage_easy")
        action = ActionPayload(action="noop")
        result = env.step(action)
        print(result.observation)   # EnvironmentState as dict
        print(result.reward)
        print(result.done)

Usage (async):
    import asyncio
    from environment.client import PFPAEnvClient
    from environment.models import ActionPayload

    async def run():
        async with PFPAEnvClient(base_url="http://localhost:7860") as env:
            state = await env.reset(task_id="signal_triage_easy")
            action = ActionPayload(action="execute_prediction", prediction_id="pred-001")
            result = await env.step(action)

    asyncio.run(run())

From HuggingFace Spaces:
    env = PFPAEnvClient(base_url="https://<username>-pfpa-openenv.hf.space").sync()
    with env:
        state = env.reset()
"""
from __future__ import annotations
from typing import Any, Dict, Optional
from openenv import GenericEnvClient
from environment.models import ActionPayload, EnvironmentState


class PFPAEnvClient(GenericEnvClient):
    """
    Typed client for the PFPA OpenEnv personal productivity environment.

    Wraps GenericEnvClient with PFPA-specific reset/step helpers and
    returns typed EnvironmentState objects instead of raw dicts.

    Inherits from GenericEnvClient which provides:
      - .sync()                  → SyncEnvClient wrapper for non-async code
      - .from_docker_image(...)  → connect via local Docker image
      - .from_env(...)           → connect via HuggingFace Hub Space
    """

    def _step_payload(self, action: ActionPayload | Dict[str, Any]) -> Dict[str, Any]:
        """Convert an ActionPayload (or raw dict) to the server's expected format."""
        if isinstance(action, ActionPayload):
            # Wrap in the {"action": {...}} envelope the server expects
            return {"action": action.model_dump(exclude_none=False)}
        if isinstance(action, dict):
            # If caller already passed the full envelope, pass through
            if "action" in action and isinstance(action["action"], dict):
                return action
            # Otherwise wrap it
            return {"action": action}
        raise TypeError(f"Unsupported action type: {type(action)}")

    # ── Convenience typed helpers ──────────────────────────────────────────────

    def reset_typed(
        self,
        task_id: str = "signal_triage_easy",
    ) -> EnvironmentState:
        """
        Reset the environment and return a typed EnvironmentState.
        Use this instead of the raw reset() if you want IDE autocomplete.

        Note: This is a sync helper — use inside a .sync() context.
        """
        import requests as _req
        base = self._base_url.replace("ws://", "http://").replace("wss://", "https://")
        r = _req.post(f"{base}/reset", json={"task_id": task_id}, timeout=30)
        r.raise_for_status()
        return EnvironmentState(**r.json())

    def step_typed(self, action: ActionPayload) -> Dict[str, Any]:
        """
        Take a step with a typed ActionPayload and return the raw StepResponse dict.
        Use inside a .sync() context.
        """
        import requests as _req
        base = self._base_url.replace("ws://", "http://").replace("wss://", "https://")
        r = _req.post(
            f"{base}/step",
            json={"action": action.model_dump(exclude_none=False)},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def state_typed(self) -> EnvironmentState:
        """Return the current environment state as a typed EnvironmentState."""
        import requests as _req
        base = self._base_url.replace("ws://", "http://").replace("wss://", "https://")
        r = _req.get(f"{base}/state", timeout=15)
        r.raise_for_status()
        return EnvironmentState(**r.json())
