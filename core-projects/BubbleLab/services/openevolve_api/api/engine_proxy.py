"""
Engine proxy for the OpenEvolve decomposition workflow engine (:8001).

This module centralises the HTTPX call the ``:8000`` OpenEvolve API service makes
to the ``:8001`` Decomposition-Workflow engine. The ``:8000`` service acts as the
unified authority for the BubbleLab UI: settings/plan storage is served from the
``:8000`` store, while *running* a workflow is delegated to ``:8001`` via
``POST /workflows/run``.

Auth: the frontend sends ``X-API-Key`` (value of ``OPENEVOLVE_API_KEY``). The
header is forwarded verbatim so ``:8001``'s ``verify_api_key`` dependency can
validate it. No secrets are hardcoded.
"""

import os

import httpx

# Engine base URL. Both servers should resolve the same environment value so the
# shared OPENEVOLVE_API_KEY is honoured end-to-end. Default to the :8001 port.
ENGINE_API_BASE_URL = os.getenv(
    "ENGINE_API_BASE_URL",
    os.getenv("OPENEVOLVE_ENGINE_URL", "http://localhost:8001"),
).rstrip("/")

# Hard timeout so a downed engine fails fast instead of hanging the orchestrator.
ENGINE_HTTP_TIMEOUT = float(os.getenv("ENGINE_HTTP_TIMEOUT", "60"))


async def run_workflow_on_engine(
    problem_statement: str,
    config: dict,
    api_key: "str | None" = None,
) -> dict:
    """
    Forward a workflow run to the ``:8001`` engine.

    Args:
        problem_statement: The workflow problem statement (from the `:8000` store).
        config: Merged settings/plan + caller-supplied config dict.
        api_key: Inbound ``X-API-Key`` header value, forwarded to ``:8001``.

    Returns:
        The JSON body returned by ``:8001`` (``{workflow_id, status, tenant_id}``).

    Raises:
        httpx.HTTPStatusError: when ``:8001`` responds with a non-2xx status.
        httpx.HTTPError: on connection/transport failures.
    """
    url = f"{ENGINE_API_BASE_URL}/workflows/run"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    payload = {
        "problem_statement": problem_statement,
        "team_ids": [],
        "gauntlet_ids": [],
        "config": config or {},
    }

    async with httpx.AsyncClient(timeout=ENGINE_HTTP_TIMEOUT) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
