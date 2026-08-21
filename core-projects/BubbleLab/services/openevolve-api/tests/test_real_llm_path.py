"""
GATED integration test for the REAL (live provider) LLM path of ``/api/v1/*``.

The ``/api/v1/*`` engine path runs the deterministic OFFLINE mock LLM by default
(``llm_mode == "mock"``, see ``tests/test_openevolve_v1_e2e.py``), so the live
branch of ``core.openevolve_bridge._build_llm_models`` is never exercised. This
module closes that gap WITHOUT making the normal suite depend on credentials or
network access.

Opt in by exporting real provider credentials::

    OPENEVOLVE_REAL_LLM_PROVIDER=openai         # required: openai|openrouter|groq|deepseek|google|anthropic|...
    OPENEVOLVE_REAL_LLM_API_KEY=sk-...          # required (falls back to <PROVIDER>_API_KEY)
    OPENEVOLVE_REAL_LLM_MODEL=gpt-4o-mini       # optional, provider default otherwise
    OPENEVOLVE_REAL_LLM_API_BASE=https://...    # optional, provider default otherwise
    OPENEVOLVE_REAL_LLM_TIMEOUT_SECONDS=600     # optional wall-clock ceiling (default 600)

Without those vars every test here SKIPS with an explicit reason, so the default
suite stays green, offline and free.

Run with:
    python -m pytest tests/test_real_llm_path.py -q -p no:pytest_ethereum
    python -m pytest -m real_llm -q -p no:pytest_ethereum        # opt-in selection
"""

import os
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Isolate the workflow DB before the service modules import (they open it eagerly).
os.environ.setdefault(
    "WORKFLOW_DB_PATH",
    str(Path(tempfile.gettempdir()) / "openevolve_api_real_llm_workflows.db"),
)
os.environ.setdefault("OPENEVOLVE_BRIDGE_ENABLED", "1")

if "openevolve_api" not in sys.modules:
    package = types.ModuleType("openevolve_api")
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules["openevolve_api"] = package

pytest.importorskip(
    "openevolve", reason="the real openevolve library must be installed (pip install -e ../../../openevolve)"
)

from fastapi.testclient import TestClient  # noqa: E402

import openevolve_api.api.openevolve_v1 as v1  # noqa: E402
from openevolve_api.core.openevolve_bridge import _build_llm_models  # noqa: E402
from openevolve_api.main import app  # noqa: E402

# Opt-in markers: never selected by a plain ``pytest`` run that filters on them.
pytestmark = [pytest.mark.real_llm, pytest.mark.integration, pytest.mark.slow]


# --------------------------------------------------------------------------- #
# Environment gate
# --------------------------------------------------------------------------- #
ENV_PROVIDER = "OPENEVOLVE_REAL_LLM_PROVIDER"
ENV_API_KEY = "OPENEVOLVE_REAL_LLM_API_KEY"
ENV_MODEL = "OPENEVOLVE_REAL_LLM_MODEL"
ENV_API_BASE = "OPENEVOLVE_REAL_LLM_API_BASE"
ENV_TIMEOUT = "OPENEVOLVE_REAL_LLM_TIMEOUT_SECONDS"

# provider -> (default model, default OpenAI-compatible api_base)
_PROVIDER_DEFAULTS: Dict[str, tuple] = {
    "openai": ("gpt-4o-mini", "https://api.openai.com/v1"),
    "openrouter": ("openai/gpt-4o-mini", "https://openrouter.ai/api/v1"),
    "groq": ("llama-3.1-8b-instant", "https://api.groq.com/openai/v1"),
    "deepseek": ("deepseek-chat", "https://api.deepseek.com/v1"),
    "google": ("gemini-2.0-flash", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    "gemini": ("gemini-2.0-flash", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    "anthropic": ("claude-3-5-haiku-latest", "https://api.anthropic.com/v1"),
}

# Tiny by construction: a real provider is billed per call, so keep the run to a
# couple of generations over a very small population.
REAL_ITERATIONS = 2
REAL_POPULATION_SIZE = 3
DEFAULT_TIMEOUT_SECONDS = 600.0


def _env(name: str) -> str:
    value = os.getenv(name)
    return value.strip() if isinstance(value, str) else ""


def _resolve_real_llm() -> Optional[Dict[str, Any]]:
    """Build the bridge ``llm`` dict from env, or ``None`` when not configured."""
    provider = _env(ENV_PROVIDER).lower()
    if not provider or provider == "mock":
        return None

    api_key = _env(ENV_API_KEY) or _env(f"{provider.upper()}_API_KEY")
    if not api_key:
        return None

    default_model, default_base = _PROVIDER_DEFAULTS.get(provider, ("", ""))
    model = _env(ENV_MODEL) or default_model
    api_base = _env(ENV_API_BASE) or default_base
    if not model or not api_base:
        return None

    return {
        "name": model,
        "provider": provider,
        "api_key": api_key,
        "api_base": api_base,
        "temperature": 0.7,
        "max_tokens": 2048,
    }


def _missing_env() -> List[str]:
    """Human-readable list of what still has to be set to opt in."""
    provider = _env(ENV_PROVIDER).lower()
    missing: List[str] = []
    if not provider:
        missing.append(ENV_PROVIDER)
    elif provider == "mock":
        missing.append(f"{ENV_PROVIDER} (is 'mock'; a real provider name is required)")
    if not (_env(ENV_API_KEY) or (provider and _env(f"{provider.upper()}_API_KEY"))):
        missing.append(ENV_API_KEY)
    if provider and provider not in _PROVIDER_DEFAULTS:
        if not _env(ENV_MODEL):
            missing.append(f"{ENV_MODEL} (no default for provider '{provider}')")
        if not _env(ENV_API_BASE):
            missing.append(f"{ENV_API_BASE} (no default for provider '{provider}')")
    return missing


def _timeout_seconds() -> float:
    try:
        return max(30.0, float(_env(ENV_TIMEOUT)))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def _require_real_llm() -> Dict[str, Any]:
    """Skip unless real provider credentials are configured in the environment."""
    llm = _resolve_real_llm()
    if llm is None:
        pytest.skip(
            "real LLM path not exercised: live provider credentials are not configured. "
            f"Set {ENV_PROVIDER} + {ENV_API_KEY} (optionally {ENV_MODEL}, {ENV_API_BASE}, "
            f"{ENV_TIMEOUT}) to opt in. Missing: {', '.join(_missing_env()) or 'unknown'}"
        )
    return llm


# --------------------------------------------------------------------------- #
# Tiny program / evaluator (fast + cheap for a live run)
# --------------------------------------------------------------------------- #
INITIAL_PROGRAM = '''"""Tiny candidate program for the gated real-LLM run."""

# EVOLVE-BLOCK-START
def solve(x):
    """Return a value as close to 42 as possible."""
    return x + 1
# EVOLVE-BLOCK-END


def run():
    """Stable entry point used by the evaluator."""
    return solve(40)
'''

EVALUATOR = '''"""Tiny evaluator: score how close run() gets to 42."""

import importlib.util


def evaluate(program_path):
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    if spec is None or spec.loader is None:
        return {"combined_score": 0.0, "runs": 0.0}
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        value = float(module.run())
    except Exception as exc:  # unparsable / crashing candidate
        return {"combined_score": 0.0, "runs": 0.0, "error": str(exc)[:200]}
    return {
        "combined_score": round(1.0 / (1.0 + abs(value - 42.0)), 6),
        "value": value,
        "runs": 1.0,
    }
'''


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# Gate / wiring checks (no network)
# --------------------------------------------------------------------------- #
def test_env_credentials_select_the_live_llm_backend():
    """The env-provided config must route the bridge to a LIVE model, not the mock."""
    llm = _require_real_llm()

    models, mode = _build_llm_models({"llm": llm}, {})
    assert mode == "live", f"bridge stayed in mock mode for provider {llm['provider']!r}"
    assert len(models) == 1
    assert models[0].name == llm["name"]
    assert str(models[0].api_key)  # credentials actually reached the model config


def test_evolve_endpoint_forwards_live_llm_config(client, monkeypatch):
    """``POST /api/v1/evolve`` must carry the live ``llm`` block to the bridge."""
    llm = _require_real_llm()

    captured: Dict[str, Any] = {}

    def fake_run(request):
        captured["request"] = dict(request)
        return {
            "best_code": "x = 1",
            "best_score": 1.0,
            "metrics": {},
            "generations": 1,
            "engine": "openevolve",
            "engine_entrypoint": "openevolve.api.run_evolution",
            "llm_mode": "live",
            "iterations": 1,
            "population_size": REAL_POPULATION_SIZE,
            "mock_llm_calls": 0,
            "duration_seconds": 0.01,
            "started_at": "",
            "completed_at": "",
        }

    monkeypatch.setattr(v1, "run_openevolve_workflow", fake_run)

    resp = client.post(
        "/api/v1/evolve",
        json={
            "initial_program": INITIAL_PROGRAM,
            "evaluator": EVALUATOR,
            "config": {"max_iterations": 1, "population_size": REAL_POPULATION_SIZE},
            "llm": llm,
        },
    )
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["run_id"]

    deadline = time.time() + 15
    while time.time() < deadline:
        run = client.get(f"/api/v1/runs/{run_id}")
        assert run.status_code == 200, run.text
        if run.json()["status"] in ("completed", "failed"):
            break
        time.sleep(0.2)

    assert captured.get("request", {}).get("llm") == llm


# --------------------------------------------------------------------------- #
# The real run
# --------------------------------------------------------------------------- #
def test_real_llm_evolution_completes_over_api_v1(client):
    """Drive ONE real evolution run through ``/api/v1/*`` against a live provider."""
    llm = _require_real_llm()
    timeout = _timeout_seconds()

    # a. Start the run through the /api/v1 engine dialect.
    resp = client.post(
        "/api/v1/evolve",
        json={
            "initial_program": INITIAL_PROGRAM,
            "evaluator": EVALUATOR,
            "config": {
                "max_iterations": REAL_ITERATIONS,
                "population_size": REAL_POPULATION_SIZE,
                "log_level": "WARNING",
            },
            "llm": llm,
            "timeout_seconds": timeout,
        },
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "running"
    run_id = body["run_id"]
    assert isinstance(run_id, str) and run_id

    # b. Poll until the live engine finishes (generous ceiling: real API latency).
    run_body: Dict[str, Any] = {}
    deadline = time.time() + timeout + 30
    while time.time() < deadline:
        run = client.get(f"/api/v1/runs/{run_id}")
        assert run.status_code == 200, run.text
        run_body = run.json()
        if run_body["status"] in ("completed", "failed"):
            break
        time.sleep(2.0)

    assert run_body.get("status") not in (None, "running"), (
        f"real LLM run {run_id} did not finish within {timeout + 30:.0f}s"
    )
    assert run_body["status"] == "completed", (
        f"real LLM run failed against provider {llm['provider']}/{llm['name']}: "
        f"{run_body.get('error')}"
    )

    # c. Prove the LIVE path ran and produced a real program.
    result = run_body["result"]
    assert result is not None
    assert result["engine"] == "openevolve"
    assert result["engine_entrypoint"] == "openevolve.api.run_evolution"
    assert result["llm_mode"] == "live", "run fell back to the offline mock LLM"
    assert isinstance(result["best_code"], str) and result["best_code"].strip()
    assert isinstance(result["metrics"], dict)
    assert result["iterations"] == REAL_ITERATIONS
    assert result["population_size"] == REAL_POPULATION_SIZE
    assert result["duration_seconds"] > 0
    # The mock backend must not have produced any completion for a live run.
    assert result["mock_llm_calls"] in (0, None)
