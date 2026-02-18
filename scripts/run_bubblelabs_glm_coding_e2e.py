#!/usr/bin/env python3
"""Run a real BubbleLabs -> OpenEvolve end-to-end workflow using Z.AI GLM coding endpoint."""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import requests


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


def _mask(secret: str, head: int = 6, tail: int = 4) -> str:
    if not secret:
        return "<empty>"
    if len(secret) <= head + tail:
        return "*" * len(secret)
    return f"{secret[:head]}...{secret[-tail:]}"


def _expect_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _extract_model_ids(payload: Any) -> List[str]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [item.get("id") for item in data if isinstance(item, dict) and item.get("id")]
    if isinstance(payload, list):
        return [item.get("id") for item in payload if isinstance(item, dict) and item.get("id")]
    return []


def _choose_glm_model(api_base: str, zai_api_key: str, requested_model: Optional[str]) -> str:
    if requested_model:
        return requested_model

    headers = {"Authorization": f"Bearer {zai_api_key}"}
    try:
        response = requests.get(f"{api_base}/models", headers=headers, timeout=30)
        if response.ok:
            model_ids = _extract_model_ids(response.json())
            preferred = [
                "glm-4.7",
                "glm-5",
                "glm-4.6",
                "glm-4-plus",
                "glm-4",
            ]
            for model in preferred:
                if model in model_ids:
                    return model
            if model_ids:
                return model_ids[0]
    except Exception:
        pass

    return "glm-4.7"


def _provider_smoke(api_base: str, zai_api_key: str, model: str) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {zai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply exactly: GLM_CODING_OK"}],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    response = requests.post(
        f"{api_base}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )

    snippet = response.text[:300]
    if not response.ok:
        raise RuntimeError(
            f"Z.AI provider call failed: HTTP {response.status_code} body={snippet}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Z.AI provider returned non-JSON response: {snippet}") from exc

    try:
        content = body["choices"][0]["message"]["content"]
    except Exception:
        content = str(body)[:300]

    return {
        "status_code": response.status_code,
        "model": model,
        "content": content,
    }


def _request_json(method: str, url: str, headers: Dict[str, str], payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    response = requests.request(method=method, url=url, headers=headers, json=payload, timeout=60)
    if not response.ok:
        raise RuntimeError(f"{method} {url} failed: HTTP {response.status_code} body={response.text[:400]}")
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"{method} {url} returned non-JSON payload: {response.text[:300]}") from exc


def _redact_workflow_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    redacted = json.loads(json.dumps(payload))
    try:
        key = redacted["parameters"]["openevolve_parameters"].get("api_key")
        if isinstance(key, str):
            redacted["parameters"]["openevolve_parameters"]["api_key"] = _mask(key)
    except Exception:
        pass
    return redacted


def _run_workflow(
    api_url: str,
    openevolve_api_key: str,
    zai_api_key: str,
    zai_api_base: str,
    model_id: str,
) -> Dict[str, Any]:
    headers = {
        "X-API-Key": openevolve_api_key,
        "Authorization": f"Bearer {openevolve_api_key}",
        "Content-Type": "application/json",
    }

    suffix = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    definition_id = None
    instance_id = None

    try:
        definition = _request_json(
            "POST",
            f"{api_url}/bubblelabs/workflow-definitions",
            headers,
            {
                "name": f"glm-coding-e2e-{suffix}",
                "description": "Real BubbleLabs/OpenEvolve E2E with Z.AI coding endpoint",
                "workflow_type": "evolution",
                "parameters": {
                    "max_iterations": 1,
                    "population_size": 2,
                    "temperature": 0.0,
                    "openevolve_parameters": {
                        "api_key": zai_api_key,
                        "api_base": zai_api_base,
                        "model_id": model_id,
                        "max_iterations": 1,
                        "population_size": 2,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_tokens": 64,
                    },
                },
            },
        )
        definition_id = definition["definition_id"]

        instance = _request_json(
            "POST",
            f"{api_url}/bubblelabs/workflow-instances",
            headers,
            {
                "definition_id": definition_id,
                "instance_name": f"glm-e2e-instance-{suffix}",
                "inputs": {
                    "problem_statement": "Generate a concise plan title for BubbleLabs integration."
                },
            },
        )
        instance_id = instance["instance_id"]

        _request_json(
            "POST",
            f"{api_url}/bubblelabs/workflow-instances/{instance_id}/parameters",
            headers,
            {
                "parameters": {
                    "max_iterations": 1,
                    "population_size": 2,
                    "temperature": 0.0,
                    "openevolve_parameters": {
                        "api_key": zai_api_key,
                        "api_base": zai_api_base,
                        "model_id": model_id,
                        "max_iterations": 1,
                        "population_size": 2,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_tokens": 64,
                    },
                }
            },
        )

        _request_json(
            "POST",
            f"{api_url}/bubblelabs/workflow-instances/{instance_id}/start",
            headers,
            {},
        )

        terminal = {"completed", "failed", "stopped", "cancelled"}
        deadline = time.time() + 180
        status_payload = {}
        while time.time() < deadline:
            status_payload = _request_json(
                "GET",
                f"{api_url}/bubblelabs/workflow-instances/{instance_id}",
                headers,
                None,
            )
            state = (
                status_payload.get("status", {}).get("status")
                if isinstance(status_payload.get("status"), dict)
                else status_payload.get("status")
            )
            if state in terminal:
                return status_payload
            time.sleep(1.0)

        raise RuntimeError("Workflow polling timed out before reaching terminal status")
    finally:
        if instance_id:
            try:
                requests.delete(
                    f"{api_url}/bubblelabs/workflow-instances/{instance_id}",
                    headers=headers,
                    timeout=30,
                )
            except Exception:
                pass


def main() -> int:
    _load_dotenv_if_available()

    zai_api_key = _expect_env("ZAI_API_KEY")
    zai_api_base = os.getenv("ZAI_API_BASE", "https://api.z.ai/api/coding/paas/v4").rstrip("/")
    requested_model = os.getenv("ZAI_MODEL_ID", "").strip() or None

    openevolve_api_url = os.getenv("OPENEVOLVE_API_URL", "http://127.0.0.1:8000").rstrip("/")
    openevolve_api_key = _expect_env("OPENEVOLVE_API_KEY")

    print(
        json.dumps(
            {
                "stage": "config",
                "zai_api_base": zai_api_base,
                "zai_api_key": _mask(zai_api_key),
                "openevolve_api_url": openevolve_api_url,
                "openevolve_api_key": _mask(openevolve_api_key),
            },
            indent=2,
        )
    )

    model_id = _choose_glm_model(zai_api_base, zai_api_key, requested_model)
    print(json.dumps({"stage": "model_selection", "model_id": model_id}, indent=2))

    smoke = _provider_smoke(zai_api_base, zai_api_key, model_id)
    print(json.dumps({"stage": "provider_smoke", **smoke}, indent=2))

    workflow_status = _run_workflow(
        api_url=openevolve_api_url,
        openevolve_api_key=openevolve_api_key,
        zai_api_key=zai_api_key,
        zai_api_base=zai_api_base,
        model_id=model_id,
    )
    safe_payload = _redact_workflow_payload(workflow_status)
    print(json.dumps({"stage": "workflow_terminal", "payload": safe_payload}, indent=2, default=str))

    state = (
        workflow_status.get("status", {}).get("status")
        if isinstance(workflow_status.get("status"), dict)
        else workflow_status.get("status")
    )
    if state != "completed":
        err = (
            workflow_status.get("status", {}).get("error_message")
            if isinstance(workflow_status.get("status"), dict)
            else None
        )
        raise RuntimeError(f"Workflow did not complete successfully. terminal_status={state} error={err}")

    print(json.dumps({"stage": "result", "status": "ok", "terminal_status": state}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"stage": "result", "status": "failed", "error": str(exc)}, indent=2))
        raise
