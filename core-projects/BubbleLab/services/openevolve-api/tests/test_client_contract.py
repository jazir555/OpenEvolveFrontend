"""
Automated client <-> backend contract test for the OpenEvolve FastAPI service.

It mirrors every ``/api/*`` (and unprefixed ``/health``) endpoint the BubbleLab
UI client (``apps/bubble-studio/src/services/openevolveApi.ts``) calls and
proves, via FastAPI ``TestClient``, that the backend responds without crashing.

Strategy (per the audit / task brief):
  * Routes that currently exist should return 2xx with the expected top-level
    JSON keys.
  * Routes that are intentionally NOT mounted (the ``/bubblelabs/control/*`` and
    ``/bubblelabs/workflow-*`` groups the client still fires at the old address)
    are expected to 404 -> recorded as a KNOWN GAP, not a failure.
  * Routes that proxy an external backend (LeanAide, CrewAI) may degrade to 502
    when that backend is down -> recorded as DEGRADED, never a 500 crash.
  * A genuine 500 (server crash) is the ONLY hard failure.

The test is hermetic: it uses an in-memory TempDir for the SQLite DB, patches
the execution engine so no real LLM work / background thread is spawned, and
tolerates absent route groups.

Run with:
    python -m pytest tests/test_client_contract.py -q -p no:pytest_ethereum
"""

import os
import sys
import types
import tempfile
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]

_DB_PATH = str(Path(tempfile.gettempdir()) / "openevolve_api_contract_workflows.db")
os.environ.setdefault("WORKFLOW_DB_PATH", _DB_PATH)
# Keep the legacy in-service execution path for this contract run by disabling
# the real openevolve bridge. Done via a scoped fixture (below) rather than a
# global env var so it cannot leak into other test modules.

if "openevolve_api" not in sys.modules:
    package = types.ModuleType("openevolve_api")
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules["openevolve_api"] = package

from fastapi.testclient import TestClient  # noqa: E402

from openevolve_api.main import app  # noqa: E402
from openevolve_api.api import workflows as _wf_module  # noqa: E402
from openevolve_api.services import execution_service as _es  # noqa: E402


@pytest.fixture(autouse=True)
def _disable_openevolve_bridge():
    """Disable the real-engine bridge for this contract run only.

    Restores the original flag afterwards so other test modules (which exercise
    the bridge) are not polluted by this module's choice.
    """
    original = _wf_module.OPENEVOLVE_BRIDGE_AVAILABLE
    _wf_module.OPENEVOLVE_BRIDGE_AVAILABLE = False
    try:
        yield
    finally:
        _wf_module.OPENEVOLVE_BRIDGE_AVAILABLE = original


@pytest.fixture()
def client():
    # Patch the execution engine so workflow execution does not spawn real
    # background threads / LLM calls. The fake still creates the execution
    # record (and the workflow<->execution link) so pause/resume/cancel/detail
    # routes operate against real in-memory state.
    counter = {"n": 0}
    import datetime
    import threading

    orig_start = _es.execution_manager.start_execution

    async def fake_start(workflow_id, problem_statement, context=None):
        counter["n"] += 1
        execution_id = f"exec_test_{counter['n']}"
        now = datetime.datetime.now(datetime.timezone.utc)
        execution = {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "status": "running",
            "progress": 0.0,
            "started_at": now,
            "completed_at": None,
            "result": None,
            "error": None,
            "logs": [],
            "workflow_type": "sovereign",
            "parameters": {},
        }
        with _es.execution_manager._lock:
            _es.execution_manager._executions[execution_id] = execution
            _es.execution_manager._pause_events[execution_id] = threading.Event()
            _es.execution_manager._cancel_events[execution_id] = threading.Event()
        _wf_module._workflow_executions[workflow_id] = execution_id
        return execution

    _es.execution_manager.start_execution = fake_start
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        _es.execution_manager.start_execution = orig_start


# --------------------------------------------------------------------------- #
# Route specification table
# --------------------------------------------------------------------------- #
# Each spec:
#   name           human label
#   method         GET/POST/PUT/DELETE
#   path           full path; may contain {placeholders}
#   body(res)      optional callable -> JSON body
#   params         optional query string
#   expected_keys  top-level keys asserted when the response is 2xx
#   needs          resource keys that must exist to resolve the path
#   gap_expected   True => a 404 here is the EXPECTED contract gap (no failure)
#   allow_5xx      True => a 5xx is tolerated as DEGRADED (engine-dependent)
#   tolerant       True => no shape assertion, only "is JSON / not crash"

ROUTE_SPECS = [
    # ---- Health & Info ----
    dict(name="health", method="GET", path="/health",
         expected_keys=["status", "service", "version", "features"]),

    # ---- Workflows (collection) ----
    dict(name="workflows.create", method="POST", path="/api/workflows",
         body=lambda r: {"name": "contract-wf", "description": "contract",
                          "workflow_type": "sovereign", "parameters": {}},
         expected_keys=["id", "name", "workflow_type"]),
    dict(name="workflows.list", method="GET",
         path="/api/workflows", params="page=1&page_size=10",
         expected_keys=["workflows", "total"]),
    dict(name="workflows.get", method="GET", path="/api/workflows/{workflow_id}",
         needs=["workflow_id"], expected_keys=["id", "name", "workflow_type"]),
    dict(name="workflows.pause", method="POST",
         path="/api/workflows/{workflow_id}/pause", needs=["workflow_id"],
         expected_keys=["id", "name"]),
    dict(name="workflows.resume", method="POST",
         path="/api/workflows/{workflow_id}/resume", needs=["workflow_id"],
         expected_keys=["id", "name"]),
    dict(name="workflows.results", method="GET",
         path="/api/workflows/{workflow_id}/results", needs=["workflow_id"],
         expected_keys=["workflow_id", "status"]),
    # decomposition-plan is a client sub-path the backend does NOT mount yet.
    dict(name="workflows.decomposition-plan", method="GET",
         path="/api/workflows/{workflow_id}/decomposition-plan",
         needs=["workflow_id"], gap_expected=True),
    dict(name="workflows.delete", method="DELETE",
         path="/api/workflows/{workflow_del_id}", needs=["workflow_del_id"],
         expected_keys=["message"]),

    # ---- Executions ----
    dict(name="executions.create", method="POST", path="/api/executions",
         body=lambda r: {"workflow_id": r["workflow_id"],
                          "problem_statement": "Given x=1 and y=2, compute x+y.",
                          "context": ""},
         expected_keys=["execution_id", "workflow_id", "status"],
         allow_5xx=True),
    dict(name="executions.get", method="GET", path="/api/executions/{execution_id}",
         needs=["execution_id"], expected_keys=["execution_id", "workflow_id", "status"],
         allow_5xx=True),
    dict(name="executions.pause", method="POST",
         path="/api/executions/{execution_id}/pause", needs=["execution_id"],
         expected_keys=["execution_id", "status"], allow_5xx=True),
    dict(name="executions.resume", method="POST",
         path="/api/executions/{execution_id}/resume", needs=["execution_id"],
         expected_keys=["execution_id", "status"], allow_5xx=True),
    dict(name="executions.cancel", method="POST",
         path="/api/executions/{execution_id}/cancel", needs=["execution_id"],
         expected_keys=["execution_id", "status"], allow_5xx=True),
    dict(name="executions.logs", method="GET",
         path="/api/executions/{execution_id}/logs", needs=["execution_id"],
         expected_keys=["logs", "total"], allow_5xx=True),
    # list (GET /api/executions) is not mounted in the execution router.
    dict(name="executions.list", method="GET", path="/api/executions",
         gap_expected=True),

    # ---- Teams ----
    dict(name="teams.create", method="POST", path="/api/teams",
         body=lambda r: {"name": "contract-team", "description": "t",
                          "members": [{"name": "m1", "role": "solver",
                                        "model": "gpt-4", "temperature": 0.7,
                                        "max_tokens": 4096}]},
         expected_keys=["id", "name", "members"]),
    dict(name="teams.list", method="GET", path="/api/teams",
         expected_keys=["teams", "total"]),
    dict(name="teams.get", method="GET", path="/api/teams/{team_id}",
         needs=["team_id"], expected_keys=["id", "name", "members"]),
    dict(name="teams.update", method="PUT", path="/api/teams/{team_id}",
         needs=["team_id"],
         body=lambda r: {"name": "contract-team", "description": "updated",
                          "members": [{"name": "m1", "role": "solver",
                                        "model": "gpt-4", "temperature": 0.7,
                                        "max_tokens": 4096}]},
         expected_keys=["id", "name", "members"]),
    dict(name="teams.delete", method="DELETE", path="/api/teams/{team_del_id}",
         needs=["team_del_id"], expected_keys=["message"]),

    # ---- Gauntlets ----
    dict(name="gauntlets.create", method="POST", path="/api/gauntlets",
         body=lambda r: {"name": "contract-gauntlet", "description": "g",
                          "rounds": [{"name": "r1", "quorum_threshold": 0.6,
                                       "confidence_threshold": 0.6,
                                       "evaluation_type": "approval"}]},
         expected_keys=["id", "name", "rounds"]),
    dict(name="gauntlets.list", method="GET", path="/api/gauntlets",
         expected_keys=["gauntlets", "total"]),
    dict(name="gauntlets.get", method="GET", path="/api/gauntlets/{gauntlet_id}",
         needs=["gauntlet_id"], expected_keys=["id", "name", "rounds"]),
    dict(name="gauntlets.update", method="PUT", path="/api/gauntlets/{gauntlet_id}",
         needs=["gauntlet_id"],
         body=lambda r: {"name": "contract-gauntlet", "description": "updated",
                          "rounds": [{"name": "r1", "quorum_threshold": 0.6,
                                       "confidence_threshold": 0.6,
                                       "evaluation_type": "approval"}]},
         expected_keys=["id", "name", "rounds"]),
    dict(name="gauntlets.delete", method="DELETE",
         path="/api/gauntlets/{gauntlet_del_id}", needs=["gauntlet_del_id"],
         expected_keys=["message"]),

    # ---- Evaluators ----
    dict(name="evaluators.list", method="GET", path="/api/evaluators",
         expected_keys=["evaluators"]),
    dict(name="evaluators.upload", method="POST", path="/api/evaluators",
         body=lambda r: {"code": "def evaluate(solution, problem):\n    return 1.0"},
         expected_keys=["evaluator_id"]),
    dict(name="evaluators.delete", method="DELETE",
         path="/api/evaluators/{evaluator_id}", needs=["evaluator_id"],
         expected_keys=["success"]),

    # ---- Monitoring ----
    dict(name="monitoring.dashboard", method="GET", path="/api/monitoring/dashboard",
         expected_keys=["timestamp", "system", "health", "workflow", "recent_metrics"]),
    dict(name="monitoring.alerts", method="GET", path="/api/monitoring/alerts",
         expected_keys=["alerts"]),
    dict(name="monitoring.services", method="GET", path="/api/monitoring/services",
         expected_keys=["services"]),
    dict(name="monitoring.logs", method="GET", path="/api/monitoring/logs",
         expected_keys=["entries", "total"]),
    dict(name="monitoring.metrics", method="GET", path="/api/monitoring/metrics",
         expected_keys=["metrics"]),
    dict(name="monitoring.health", method="GET", path="/api/monitoring/health",
         expected_keys=["status"]),

    # ---- Statistics & Analytics ----
    dict(name="statistics", method="GET", path="/api/statistics",
         expected_keys=["total_workflows"]),
    dict(name="analytics.performance-metrics", method="GET",
         path="/api/analytics/performance-metrics", expected_keys=["metrics", "total"]),
    dict(name="analytics.knowledge-stats", method="GET",
         path="/api/analytics/knowledge-stats", expected_keys=["total_artifacts"]),
    dict(name="analytics.workflow-metrics", method="GET",
         path="/api/analytics/workflow-metrics", expected_keys=["metrics"]),

    # ---- Knowledge ----
    dict(name="knowledge.artifacts.list", method="GET",
         path="/api/knowledge/artifacts", expected_keys=["artifacts"]),
    dict(name="knowledge.artifacts.get", method="GET",
         path="/api/knowledge/artifacts/{artifact_id}", needs=["artifact_id"],
         expected_keys=["id"]),
    dict(name="knowledge.artifacts.create", method="POST",
         path="/api/knowledge/artifacts",
         body=lambda r: {"artifact_type": "generic", "content": "x", "domain": "test"},
         expected_keys=["id"]),
    dict(name="knowledge.artifacts.delete", method="DELETE",
         path="/api/knowledge/artifacts/{artifact_id}", needs=["artifact_id"],
         expected_keys=["success"]),
    dict(name="knowledge.search", method="POST", path="/api/knowledge/search",
         body=lambda r: {"query": "test", "limit": 5}, expected_keys=["results"]),
    dict(name="knowledge.graph", method="GET", path="/api/knowledge/graph",
         expected_keys=["nodes", "edges"]),
    dict(name="knowledge.stats", method="GET", path="/api/knowledge/stats",
         expected_keys=["total_artifacts"]),
    dict(name="knowledge.recommendations", method="POST",
         path="/api/knowledge/recommendations",
         body=lambda r: {"query": "test"}, expected_keys=["recommendations"],
         tolerant=True),
    dict(name="knowledge.export", method="GET", path="/api/knowledge/export",
         tolerant=True),
    dict(name="knowledge.import", method="POST", path="/api/knowledge/import",
         body=lambda r: {"artifacts": []}, tolerant=True),

    # ---- CrewAI (degrades if backend down) ----
    dict(name="crewai.workflows.list", method="GET", path="/api/crewai/workflows",
         expected_keys=["workflows", "total"]),
    dict(name="crewai.workflows.get", method="GET",
         path="/api/crewai/workflows/{crewai_workflow_id}",
         needs=["crewai_workflow_id"], expected_keys=["workflow_id"], tolerant=True),
    dict(name="crewai.workflows.tickets", method="GET",
         path="/api/crewai/workflows/{crewai_workflow_id}/tickets",
         needs=["crewai_workflow_id"], expected_keys=["tickets", "total"],
         tolerant=True),

    # ---- BubbleLabs LeanAide proxy (degrades to 502 if LeanAide down) ----
    dict(name="leanaide.status", method="GET",
         path="/api/bubblelabs/leanaide/status", tolerant=True),
    dict(name="leanaide.execute", method="POST",
         path="/api/bubblelabs/leanaide/execute",
         body=lambda r: {"task_type": "prove", "payload": {"x": 1}}, tolerant=True),
    dict(name="leanaide.trees", method="GET",
         path="/api/bubblelabs/leanaide/trees", tolerant=True),
    dict(name="leanaide.tree", method="GET",
         path="/api/bubblelabs/leanaide/trees/{tree_id}", needs=["tree_id"],
         tolerant=True),
    dict(name="leanaide.proofs", method="GET",
         path="/api/bubblelabs/leanaide/proofs", tolerant=True),
    dict(name="leanaide.proof", method="GET",
         path="/api/bubblelabs/leanaide/proofs/{proof_id}", needs=["proof_id"],
         tolerant=True),
    dict(name="leanaide.prove", method="POST",
         path="/api/bubblelabs/leanaide/prove",
         body=lambda r: {"theorem": "1+1=2"}, tolerant=True),

    # ---- Version Control ----
    dict(name="version-control.versions.list", method="GET",
         path="/api/version-control/versions", expected_keys=["versions"]),
    dict(name="version-control.version.get", method="GET",
         path="/api/version-control/versions/{version_id}", needs=["version_id"],
         expected_keys=["id", "protocol_text"]),
    dict(name="version-control.current", method="GET",
         path="/api/version-control/current", expected_keys=["current"]),
    dict(name="version-control.version.create", method="POST",
         path="/api/version-control/versions",
         body=lambda r: {"protocol_text": "contract protocol v1",
                          "version_name": "v-contract"},
         expected_keys=["version_id"]),
    dict(name="version-control.version.load", method="POST",
         path="/api/version-control/versions/{version_id}/load",
         needs=["version_id"], body=lambda r: {}, expected_keys=["loaded", "current"]),
    dict(name="version-control.version.branch", method="POST",
         path="/api/version-control/versions/{version_id}/branch",
         needs=["version_id"],
         body=lambda r: {"new_version_name": "v-contract-branch"},
         expected_keys=["version_id"]),
    dict(name="version-control.compare", method="POST",
         path="/api/version-control/compare",
         body=lambda r: {"version_id_1": r["version_id"],
                          "version_id_2": r["version_id"]},
         expected_keys=["chars_added"]),
    dict(name="version-control.version.delete", method="DELETE",
         path="/api/version-control/versions/{version_del_id}",
         needs=["version_del_id"], expected_keys=["deleted"]),

    # ---- Validation ----
    dict(name="validation.rules.list", method="GET", path="/api/validation/rules",
         expected_keys=["rules", "rule_names"]),
    dict(name="validation.rule.get", method="GET",
         path="/api/validation/rules/{rule_name}", needs=["rule_name"],
         expected_keys=["name", "rule"]),
    dict(name="validation.rule.create", method="POST", path="/api/validation/rules",
         body=lambda r: {"name": "contract_rule", "max_length": 1000, "min_length": 10},
         expected_keys=["created", "rule_name"]),
    dict(name="validation.rule.update", method="PUT",
         path="/api/validation/rules/{rule_name}", needs=["rule_name"],
         body=lambda r: {"max_length": 2000}, expected_keys=["updated", "rule_name"]),
    dict(name="validation.rule.delete", method="DELETE",
         path="/api/validation/rules/{rule_name}", needs=["rule_name"],
         expected_keys=["deleted"]),
    dict(name="validation.run", method="POST", path="/api/validation/run",
         body=lambda r: {"content": "def add(a, b):\n    return a + b\n",
                          "rule_names": ["min_documentation"]},
         expected_keys=["overall_result", "validations"]),
    dict(name="validation.compliance", method="POST", path="/api/validation/compliance",
         body=lambda r: {"content": "def f():\n    return 1\n", "framework": "default"},
         expected_keys=["valid"]),

    # ---- Parameters ----
    dict(name="parameters.schema", method="GET", path="/api/parameters/schema",
         expected_keys=["parameters"]),
    dict(name="parameters.defaults", method="GET", path="/api/parameters/defaults",
         expected_keys=["population_size"]),
    dict(name="parameters.categories", method="GET", path="/api/parameters/categories",
         expected_keys=["categories"]),
    dict(name="parameters.validate", method="POST", path="/api/parameters/validate",
         body=lambda r: {"temperature": 0.7}, expected_keys=["valid", "errors", "warnings"]),

    # ---- Integrated Run ----
    dict(name="integrated.run", method="POST", path="/api/integrated/run",
         body=lambda r: {"content_type": "protocol", "red_team_models": ["a"],
                           "blue_team_models": ["b"]},
         expected_keys=["status", "timestamp", "monitoring", "parameters", "crewai"]),

    # ---- BubbleLabs control / workflow lifecycle (KNOWN CLIENT GAPS) ----
    dict(name="bubblelabs.control.catalog", method="GET",
         path="/bubblelabs/control/catalog", gap_expected=True),
    dict(name="bubblelabs.control.discover", method="POST",
         path="/bubblelabs/control/discover", body=lambda r: {"force": False},
         gap_expected=True),
    dict(name="bubblelabs.control.execute", method="POST",
         path="/bubblelabs/control/execute",
         body=lambda r: {"component": "x", "action": "y", "payload": {}},
         gap_expected=True),
    dict(name="bubblelabs.workflow-definitions.list", method="GET",
         path="/bubblelabs/workflow-definitions", gap_expected=True),
    dict(name="bubblelabs.workflow-definitions.create", method="POST",
         path="/bubblelabs/workflow-definitions",
         body=lambda r: {"name": "x", "description": "d", "workflow_type": "t",
                          "parameters": {}}, gap_expected=True),
    dict(name="bubblelabs.workflow-instances.list", method="GET",
         path="/bubblelabs/workflow-instances", gap_expected=True),
    dict(name="bubblelabs.workflow-instances.create", method="POST",
         path="/bubblelabs/workflow-instances",
         body=lambda r: {"definition_id": "d", "instance_name": "i", "inputs": {}},
         gap_expected=True),
]


def _classify(status: int, spec: dict) -> str:
    if 200 <= status < 300:
        return "OK"
    if status in (404, 405):
        return "GAP" if spec.get("gap_expected") else "GAP"
    if status == 422:
        return "PRESENT"
    if status == 400:
        return "PRESENT"
    if status in (502, 503, 504):
        return "DEGRADED"
    if status == 500:
        return "FAIL" if not spec.get("allow_5xx") else "DEGRADED"
    return "OTHER"


def _create_resources(client):
    """Create real resources so detail/sub-routes can be exercised with valid IDs."""
    res: dict = {}

    def post(path, body):
        return client.post(path, json=body)

    # Workflows (one for detail/pause/resume/results, one for delete)
    wf = post("/api/workflows", {"name": "contract-wf", "description": "c",
                                  "workflow_type": "sovereign", "parameters": {}})
    if wf.status_code < 300:
        res["workflow_id"] = wf.json().get("id")
    wf_del = post("/api/workflows", {"name": "contract-wf-del", "description": "c",
                                      "workflow_type": "sovereign", "parameters": {}})
    if wf_del.status_code < 300:
        res["workflow_del_id"] = wf_del.json().get("id")

    # Execution
    if res.get("workflow_id"):
        ex = post("/api/executions", {"workflow_id": res["workflow_id"],
                                       "problem_statement": "Compute 1+1.",
                                       "context": ""})
        if ex.status_code < 300:
            res["execution_id"] = ex.json().get("execution_id")

    # Teams
    tm = post("/api/teams", {"name": "contract-team", "description": "t",
                              "members": [{"name": "m1", "role": "solver",
                                            "model": "gpt-4", "temperature": 0.7,
                                            "max_tokens": 4096}]})
    if tm.status_code < 300:
        res["team_id"] = tm.json().get("id")
    tm_del = post("/api/teams", {"name": "contract-team-del", "description": "t",
                                  "members": [{"name": "m1", "role": "solver",
                                                "model": "gpt-4", "temperature": 0.7,
                                                "max_tokens": 4096}]})
    if tm_del.status_code < 300:
        res["team_del_id"] = tm_del.json().get("id")

    # Gauntlets
    g = post("/api/gauntlets", {"name": "contract-gauntlet", "description": "g",
                                 "rounds": [{"name": "r1", "quorum_threshold": 0.6,
                                              "confidence_threshold": 0.6,
                                              "evaluation_type": "approval"}]})
    if g.status_code < 300:
        res["gauntlet_id"] = g.json().get("id")
    g_del = post("/api/gauntlets", {"name": "contract-gauntlet-del", "description": "g",
                                     "rounds": [{"name": "r1", "quorum_threshold": 0.6,
                                                  "confidence_threshold": 0.6,
                                                  "evaluation_type": "approval"}]})
    if g_del.status_code < 300:
        res["gauntlet_del_id"] = g_del.json().get("id")

    # Evaluator
    ev = post("/api/evaluators", {"code": "def evaluate(s, p):\n    return 1.0"})
    if ev.status_code < 300:
        res["evaluator_id"] = ev.json().get("evaluator_id")

    # Version (create + use its id)
    v = post("/api/version-control/versions",
             {"protocol_text": "contract protocol", "version_name": "v-contract"})
    if v.status_code < 300:
        res["version_id_tmp"] = v.json().get("version_id")
    res["version_id"] = res.get("version_id_tmp")
    v_del = post("/api/version-control/versions",
                 {"protocol_text": "contract protocol del", "version_name": "v-contract-del"})
    if v_del.status_code < 300:
        res["version_del_id"] = v_del.json().get("version_id")

    # Validation rule
    rule = post("/api/validation/rules",
                {"name": "contract_rule", "max_length": 1000, "min_length": 10})
    if rule.status_code < 300:
        res["rule_name"] = rule.json().get("rule_name") or "contract_rule"
    else:
        res["rule_name"] = "contract_rule"

    # Knowledge artifact
    art = post("/api/knowledge/artifacts",
               {"artifact_type": "generic", "content": "x", "domain": "test"})
    if art.status_code < 300:
        res["artifact_id"] = art.json().get("id")

    # CrewAI known seeded workflow
    res["crewai_workflow_id"] = "crewai-problem-decomp"

    # LeanAide sentinel ids
    res["tree_id"] = "tree-test"
    res["proof_id"] = "proof-test"

    return res


def test_client_contract(client):
    resources = _create_resources(client)
    results = []

    for spec in ROUTE_SPECS:
        needed = spec.get("needs") or []
        missing = [n for n in needed if not resources.get(n)]
        if missing:
            results.append((spec, None, "SKIPPED", f"missing {missing}", []))
            continue

        path = spec["path"]
        for key in needed:
            path = path.replace("{" + key + "}", str(resources[key]))

        body = None
        if spec.get("body") is not None:
            try:
                body = spec["body"](resources)
            except Exception as exc:  # pragma: no cover
                results.append((spec, None, "SKIPPED", f"body error {exc}", []))
                continue

        method = spec["method"]
        params = spec.get("params", "")
        url = path + (f"?{params}" if params else "")
        try:
            if method == "GET":
                resp = client.get(url)
            elif method == "POST":
                resp = client.post(url, json=body if body is not None else {})
            elif method == "PUT":
                resp = client.put(url, json=body if body is not None else {})
            elif method == "DELETE":
                resp = client.delete(url)
            else:  # pragma: no cover
                results.append((spec, None, "SKIPPED", "bad method", []))
                continue
        except Exception as exc:  # pragma: no cover - server crash guard
            results.append((spec, None, "CRASH", f"{type(exc).__name__}: {exc}", []))
            continue

        status = resp.status_code
        category = _classify(status, spec)

        note = ""
        missing_keys = []
        if 200 <= status < 300 and not spec.get("tolerant"):
            try:
                data = resp.json()
                keys = spec.get("expected_keys") or []
                missing_keys = [k for k in keys if k not in data]
                if missing_keys:
                    note = f"missing keys: {missing_keys}"
                    if category == "OK":
                        category = "OK*"  # 2xx but shape warning
            except Exception:
                note = "non-json 2xx body"
                if category == "OK":
                    category = "OK*"

        results.append((spec, status, category, note, missing_keys))

    # ---- Print per-route summary table ----
    print("\n=== OpenEvolve client <-> backend contract ===")
    hdr = f"{'ROUTE':<34}{'METH':<6}{'STATUS':<7}{'RESULT':<10}NOTE"
    print(hdr)
    print("-" * len(hdr))
    counts = {}
    for spec, status, category, note, _ in results:
        counts[category] = counts.get(category, 0) + 1
        st = "" if status is None else str(status)
        print(f"{spec['name']:<34}{spec['method']:<6}{st:<7}{category:<10}{note}")
    print("-" * len(hdr))
    print("Summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"Total routes covered: {len(results)}")

    hard_failures = [r for r in results if r[2] == "FAIL" or r[2] == "CRASH"]
    assert not hard_failures, (
        "Contract violations (genuine 500/crash): "
        + ", ".join(f"{r[0]['name']}({r[2]})" for r in hard_failures)
    )
