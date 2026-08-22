"""
Route-contract test for the OpenEvolve decomposition surface.

Exercises ``/api/decomposition/*`` via ``fastapi.testclient.TestClient`` with NO
live server and NO network access. Every endpoint is asserted to return a
non-500 response and, where it returns JSON, to parse as JSON.

The decomposition engine modules (``problem_analyzer`` / ``decomposition_engine``)
live outside the service package; ``api/decomposition.py`` puts their directories
on ``sys.path`` at import time so the endpoints below return genuine analysis
rather than 500/empty. If the engine modules are ever genuinely unavailable the
endpoint degrades to HTTP 501 with a clear message, which this test also accepts
as a non-500 outcome.

Run with:
    python -m pytest tests/test_route_contract_decomposition.py -q
"""

import os
import sys
import types
import tempfile
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Isolate the workflow DB before service modules import (they open it eagerly).
os.environ.setdefault(
    "WORKFLOW_DB_PATH",
    str(Path(tempfile.gettempdir()) / "openevolve_api_decomposition_contract_workflows.db"),
)
os.environ.setdefault("OPENEVOLVE_BRIDGE_ENABLED", "1")

if "openevolve_api" not in sys.modules:
    package = types.ModuleType("openevolve_api")
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules["openevolve_api"] = package

from fastapi.testclient import TestClient  # noqa: E402

from openevolve_api.main import app  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# Route specification table (decomposition surface only)
# --------------------------------------------------------------------------- #
ROUTE_SPECS = [
    # Mounted under /api/decomposition
    dict(name="decomposition.executions.list", method="GET",
         path="/api/decomposition/decomposition/executions"),
    dict(name="decomposition.plan", method="POST",
         path="/api/decomposition/plan",
         body=lambda r: {
             "problem_statement": "Design a fault-tolerant payment service that "
                                  "scales to 1M requests/min and survives region failure.",
             "title": "Payment Service",
             "strategy": "hierarchical",
         }),
    dict(name="decomposition.execute", method="POST",
         path="/api/decomposition/workflows/wf_dummy/execute-decomposition",
         body=lambda r: {
             "problem_statement": "Refactor the ingestion pipeline for throughput.",
             "decomposition_method": "hierarchical",
             "granularity": "medium",
         }),
    # Unknown execution status -> 404 (a clean 4xx is NOT a 500 crash)
    dict(name="decomposition.executions.status", method="GET",
         path="/api/decomposition/decomposition/executions/nonexistent_id/status",
         tolerant=True),
]


def _is_json(resp, spec):
    ctype = resp.headers.get("content-type", "")
    return "application/json" in ctype or resp.status_code >= 400


def test_decomposition_route_contract(client):
    results = []

    for spec in ROUTE_SPECS:
        method = spec["method"]
        body = spec.get("body")
        payload = body({}) if callable(body) else (body if body is not None else {})
        url = spec["path"]

        try:
            if method == "GET":
                resp = client.get(url)
            elif method == "POST":
                resp = client.post(url, json=payload)
            else:
                results.append((spec, None, "SKIPPED", "bad method"))
                continue
        except Exception as exc:  # pragma: no cover - server crash guard
            results.append((spec, None, "CRASH", f"{type(exc).__name__}: {exc}"))
            continue

        status = resp.status_code

        # A genuine 500 is the ONLY hard failure. 501 (engine unavailable) and
        # 404 (unknown resource) are tolerated as DEGRADED/NOT-FOUND.
        if status == 500:
            category = "FAIL"
        elif status >= 500:
            category = "DEGRADED"
        else:
            category = "OK"

        parse_ok = True
        if category != "FAIL" and 200 <= status < 300 and _is_json(resp, spec):
            try:
                resp.json()
            except Exception:
                parse_ok = False
                category = "FAIL"

        note = "" if parse_ok else "body did not parse as JSON"
        results.append((spec, status, category, note))

    print("\n=== OpenEvolve decomposition route-contract coverage ===")
    hdr = f"{'ROUTE':<36}{'METH':<6}{'STATUS':<7}{'RESULT':<10}NOTE"
    print(hdr)
    print("-" * len(hdr))
    counts = {}
    for spec, status, category, note in results:
        counts[category] = counts.get(category, 0) + 1
        st = "" if status is None else str(status)
        print(f"{spec['name']:<36}{spec['method']:<6}{st:<7}{category:<10}{note}")
    print("-" * len(hdr))
    print("Summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    hard_failures = [r for r in results if r[2] == "FAIL" or r[2] == "CRASH"]
    assert not hard_failures, (
        "Decomposition route-contract violations (genuine 500/crash or "
        "unparseable JSON): "
        + ", ".join(f"{r[0]['name']}({r[2]})" for r in hard_failures)
    )


def test_decomposition_plan_returns_real_analysis(client):
    """The /plan endpoint must return an actual problem + plan payload (200)."""
    resp = client.post(
        "/api/decomposition/plan",
        json={
            "problem_statement": "Build a recommendation engine for an e-commerce site.",
            "title": "RecSys",
            "strategy": "hierarchical",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "problem" in data, "missing 'problem' in payload"
    assert "plan" in data, "missing 'plan' in payload"
    problem = data["problem"]
    plan = data["plan"]
    assert isinstance(problem.get("success_criteria"), list)
    assert isinstance(plan.get("sub_problems"), list)
    assert len(plan["sub_problems"]) >= 1, "decomposition produced no sub-problems"
    assert plan.get("strategy") in {"hierarchical", "semantic", "flow_based"}
