"""
Comprehensive backend route-contract test for the OpenEvolve FastAPI service.

It exercises EVERY mounted router group under the service (the BubbleLab
integration surface) via ``fastapi.testclient.TestClient`` with NO live server
and NO network access.

For each mounted group a representative request is performed (preferring GET
list/health endpoints; for POST-only groups a minimal valid payload is sent)
and the response is asserted to be a genuine non-500 (a 500 is treated as a
real crash bug) and, where the route returns JSON, that the body parses.

Optional external stacks (determinism engine, LeanAide proxy) may legitimately
return 5xx "unavailable" codes in a hermetic environment — those are tolerated
as DEGRADED, never as a 500 crash.

Run with:
    python -m pytest tests/test_route_contract.py -q -p no:pytest_ethereum
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
    str(Path(tempfile.gettempdir()) / "openevolve_api_route_contract_workflows.db"),
)
# Keep the real openevolve bridge enabled flag set; the bridge import itself is
# stubbed in this hermetic environment, but we re-stub it explicitly below for
# the /api/v1 dialect so /api/v1/evolve returns a clean 2xx instead of a 500.
os.environ.setdefault("OPENEVOLVE_BRIDGE_ENABLED", "1")

if "openevolve_api" not in sys.modules:
    package = types.ModuleType("openevolve_api")
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules["openevolve_api"] = package

from fastapi.testclient import TestClient  # noqa: E402

from openevolve_api.main import app  # noqa: E402
from openevolve_api.api import openevolve_v1 as _v1  # noqa: E402
from openevolve_api.api import workflows as _wf  # noqa: E402


@pytest.fixture(autouse=True)
def _stub_openevolve_bridge():
    """Make the /api/v1 engine dialect return 2xx without a real engine.

    In the hermetic test env the openevolve bridge import fails, so
    ``OPENEVOLVE_BRIDGE_AVAILABLE`` is False and ``/api/v1/evolve`` would emit a
    controlled 500. We flip the flag and stub the worker entrypoint so the
    dialect routes resolve without spawning threads or hitting the network.
    """
    original_flag = _v1.OPENEVOLVE_BRIDGE_AVAILABLE
    original_run = _v1.run_openevolve_workflow
    _v1.OPENEVOLVE_BRIDGE_AVAILABLE = True
    _v1.run_openevolve_workflow = lambda req: {"status": "ok", "ran": True}
    try:
        yield
    finally:
        _v1.OPENEVOLVE_BRIDGE_AVAILABLE = original_flag
        _v1.run_openevolve_workflow = original_run


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# Route specification table
# --------------------------------------------------------------------------- #
# name           human label
# method         GET/POST/PUT/DELETE
# path           full path (may contain {placeholders})
# body           optional JSON body (callable receiving resources)
# needs          resource keys required to resolve the path
# degraded_ok    True => a 5xx here is tolerated as an external-stack dependency
# tolerant       True => only assert "not 500 + parses", no shape checks
# json_ok        False => route is verified by existence only (SSE stream,
#                which cannot be exercised via TestClient without blocking)

ROUTE_SPECS = [
    # ---- Health & Info ----
    dict(name="health", method="GET", path="/health", json_ok=True),

    # ---- Workflows ----
    dict(name="workflows.list", method="GET", path="/api/workflows",
         params="page=1&page_size=10"),

    # ---- Teams ----
    dict(name="teams.list", method="GET", path="/api/teams"),

    # ---- Gauntlets ----
    dict(name="gauntlets.list", method="GET", path="/api/gauntlets"),

    # ---- Executions ----
    dict(name="executions.get", method="GET", path="/api/executions/{execution_id}",
         needs=["execution_id"], tolerant=True),

    # ---- Settings ----
    dict(name="settings.llm", method="GET", path="/api/settings/llm"),
    dict(name="settings.determinism", method="GET", path="/api/settings/determinism"),

    # ---- Decomposition ----
    dict(name="decomposition.executions", method="GET",
         path="/api/decomposition/decomposition/executions"),

    # ---- Parameters ----
    dict(name="parameters.schema", method="GET", path="/api/parameters/schema"),

    # ---- Monitoring ----
    dict(name="monitoring.health", method="GET", path="/api/monitoring/health"),

    # ---- Validation ----
    dict(name="validation.rules", method="GET", path="/api/validation/rules"),

    # ---- Analytics ----
    dict(name="analytics.performance-metrics", method="GET",
         path="/api/analytics/performance-metrics"),

    # ---- CrewAI ----
    dict(name="crewai.workflows", method="GET", path="/api/crewai/workflows"),

    # ---- Version Control ----
    dict(name="version-control.versions", method="GET",
         path="/api/version-control/versions"),

    # ---- Evaluators ----
    dict(name="evaluators.list", method="GET", path="/api/evaluators"),

    # ---- Integrated (POST-only) ----
    dict(name="integrated.run", method="POST", path="/api/integrated/run",
         body=lambda r: {"content_type": "protocol", "red_team_models": ["a"],
                          "blue_team_models": ["b"]}),

    # ---- LeanAide proxy (degrades if backend down) ----
    dict(name="leanaide.status", method="GET",
         path="/api/bubblelabs/leanaide/status", degraded_ok=True),

    # ---- Knowledge ----
    dict(name="knowledge.artifacts", method="GET", path="/api/knowledge/artifacts"),

    # ---- ICR ----
    dict(name="icr.refinement-needed", method="GET",
         path="/icr/events/refinement-needed"),

    # ---- Determinism (POST-only, needs optional external stack) ----
    dict(name="determinism.generate", method="POST", path="/determinism/generate",
         body=lambda r: {"prompt": "hello", "mode": "cloud", "provider": "openai",
                          "model": "gpt-4"},
         degraded_ok=True),

    # ---- OpenEvolve /api/v1 dialect (real engine dialect) ----
    dict(name="openevolve_v1.health", method="GET", path="/api/v1/health"),
    dict(name="openevolve_v1.evolve", method="POST", path="/api/v1/evolve",
         body=lambda r: {"initial_program": "def f(): return 1",
                          "evaluator": "def ev(s, p): return 1.0"}),

    # ---- BubbleLabs control plane ----
    dict(name="bubblelabs.control.catalog", method="GET",
         path="/bubblelabs/control/catalog"),

    # ---- SSE streaming endpoint (verified by route-existence; cannot be
    #      exercised via TestClient without blocking on the open stream) ----
    dict(name="stream.workflow", method="GET",
         path="/stream/workflow/{workflow_id}", json_ok=False),
]


def _create_resources(client):
    res: dict = {}
    wf = client.post(
        "/api/workflows",
        json={"name": "rc-wf", "description": "rc", "workflow_type": "sovereign",
              "parameters": {}},
    )
    if wf.status_code < 300:
        res["workflow_id"] = wf.json().get("id")
        ex = client.post(
            "/api/executions",
            json={"workflow_id": res["workflow_id"],
                  "problem_statement": "Compute 1+1.", "context": ""},
        )
        if ex.status_code < 300:
            res["execution_id"] = ex.json().get("execution_id")
    return res


def _is_json(resp, spec):
    if spec.get("json_ok") is False:
        return True  # non-JSON (SSE) bodies are not required to parse as JSON
    ctype = resp.headers.get("content-type", "")
    return "application/json" in ctype or resp.status_code >= 400


def test_route_contract(client):
    resources = _create_resources(client)
    results = []

    for spec in ROUTE_SPECS:
        needed = spec.get("needs") or []
        missing = [n for n in needed if not resources.get(n)]
        if missing:
            results.append((spec, None, "SKIPPED", f"missing {missing}"))
            continue

        path = spec["path"]
        for key in needed:
            path = path.replace("{" + key + "}", str(resources[key]))

        body = None
        if spec.get("body") is not None:
            body = spec["body"](resources)

        method = spec["method"]
        params = spec.get("params", "")
        url = path + (f"?{params}" if params else "")

        # SSE streaming route: verify it is actually mounted (path template
        # present in the app) rather than issuing a request that would block.
        if spec.get("json_ok") is False:
            mounted = any(
                getattr(r, "path", None) == path
                for r in app.routes
            )
            results.append((spec, None, "OK" if mounted else "FAIL",
                            "" if mounted else "route not mounted"))
            continue

        try:
            if method == "GET":
                resp = client.get(url)
            elif method == "POST":
                resp = client.post(url, json=body if body is not None else {})
            elif method == "PUT":
                resp = client.put(url, json=body if body is not None else {})
            elif method == "DELETE":
                resp = client.delete(url)
            else:
                results.append((spec, None, "SKIPPED", "bad method"))
                continue
        except Exception as exc:  # pragma: no cover - server crash guard
            results.append((spec, None, "CRASH", f"{type(exc).__name__}: {exc}"))
            continue

        status = resp.status_code

        # A genuine 500 is the ONLY hard failure. 502/503/504 on routes that
        # depend on optional external stacks are tolerated as DEGRADED.
        if status == 500:
            category = "FAIL"
        elif status >= 500:
            category = "DEGRADED" if spec.get("degraded_ok") else "FAIL"
        else:
            category = "OK"

        # Where JSON is expected, ensure the body parses.
        parse_ok = True
        if category != "FAIL" and 200 <= status < 300 and _is_json(resp, spec):
            try:
                resp.json()
            except Exception:
                parse_ok = False
                category = "FAIL"

        note = "" if parse_ok else "body did not parse as JSON"
        results.append((spec, status, category, note))

    # ---- Print per-route summary table ----
    print("\n=== OpenEvolve mounted route-contract coverage ===")
    hdr = f"{'ROUTE':<34}{'METH':<6}{'STATUS':<7}{'RESULT':<10}NOTE"
    print(hdr)
    print("-" * len(hdr))
    counts = {}
    for spec, status, category, note in results:
        counts[category] = counts.get(category, 0) + 1
        st = "" if status is None else str(status)
        print(f"{spec['name']:<34}{spec['method']:<6}{st:<7}{category:<10}{note}")
    print("-" * len(hdr))
    print("Summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"Total mounted groups covered: {len(results)}")

    hard_failures = [r for r in results if r[2] == "FAIL" or r[2] == "CRASH"]
    assert not hard_failures, (
        "Route-contract violations (genuine 500/crash or unparseable JSON): "
        + ", ".join(f"{r[0]['name']}({r[2]})" for r in hard_failures)
    )
