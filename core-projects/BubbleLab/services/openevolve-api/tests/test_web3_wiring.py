from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import tempfile
import types

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "WORKFLOW_DB_PATH",
    str(Path(tempfile.gettempdir()) / "openevolve_api_test_workflows.db"),
)
if "openevolve_api" not in sys.modules:
    package = types.ModuleType("openevolve_api")
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules["openevolve_api"] = package

from openevolve_api.api import decomposition as decomposition_api
from openevolve_api.api import workflows as workflows_api
from openevolve_api.models import WorkflowCreate, WorkflowResponse, WorkflowStatus


def test_workflow_create_normalizes_web3_aliases():
    workflow = WorkflowCreate(
        name="alias",
        description="alias normalization",
        problem_statement="Audit this vault smart contract for reentrancy and flash loan risks.",
        content_type="text",
        teams=[],
        gauntlets=[],
        workflow_type="smart_contract_audit",
    )
    assert workflow.workflow_type == "web3"


@pytest.mark.asyncio
async def test_workflow_list_filter_accepts_web3_aliases(monkeypatch):
    now = datetime.now(timezone.utc)
    web3_workflow = WorkflowResponse(
        id="wf_web3",
        name="web3",
        description="web3",
        problem_statement="Audit contracts",
        content_type="text",
        teams=[],
        gauntlets=[],
        status=WorkflowStatus.CREATED,
        created_at=now,
        updated_at=now,
        user_id="u",
        tenant_id="t",
        parameters={},
        workflow_type="web3",
    )
    sovereign_workflow = WorkflowResponse(
        id="wf_sovereign",
        name="sov",
        description="sov",
        problem_statement="Decompose",
        content_type="text",
        teams=[],
        gauntlets=[],
        status=WorkflowStatus.CREATED,
        created_at=now,
        updated_at=now,
        user_id="u",
        tenant_id="t",
        parameters={},
        workflow_type="sovereign",
    )
    monkeypatch.setattr(
        workflows_api,
        "_workflows",
        {
            web3_workflow.id: web3_workflow,
            sovereign_workflow.id: sovereign_workflow,
        },
        raising=False,
    )

    result = await workflows_api.list_workflows(
        page=1,
        page_size=10,
        workflow_type="defi",
        status_filter=None,
    )
    assert result.total == 1
    assert result.workflows[0].workflow_type == "web3"


class _DummyDomainContext:
    def __init__(self):
        self.domain = "general"
        self.domain_knowledge = {}


class _DummyProblem:
    def __init__(self):
        self.domain_context = _DummyDomainContext()
        self.metadata = {}

    def to_dict(self):
        return {
            "domain_context": {
                "domain": self.domain_context.domain,
                "domain_knowledge": self.domain_context.domain_knowledge,
            },
            "metadata": self.metadata,
        }


class _DummyAnalyzer:
    def __init__(self, openevolve_client_config=None):
        self.openevolve_client_config = openevolve_client_config or {}

    def analyze_problem(self, problem_text: str, title: str = ""):
        _ = problem_text, title
        return _DummyProblem()


class _DummyPlan:
    def __init__(self):
        self.metadata = {}

    def to_dict(self):
        return {"metadata": self.metadata}


class _DummyEngine:
    def __init__(self, problem_analyzer=None, enable_adaptive_selection=True, maker_config=None):
        self.problem_analyzer = problem_analyzer
        self.enable_adaptive_selection = enable_adaptive_selection
        self.maker_config = maker_config or {}

    def decompose(self, problem, strategy=None):
        _ = problem, strategy
        return _DummyPlan()


@pytest.mark.asyncio
async def test_decomposition_plan_includes_web3_artifacts(monkeypatch):
    monkeypatch.setattr(decomposition_api, "ProblemAnalyzer", _DummyAnalyzer)
    monkeypatch.setattr(decomposition_api, "DecompositionEngine", _DummyEngine)
    monkeypatch.setattr(
        decomposition_api,
        "get_setting",
        lambda _key: {
            "default_domain_hint": "general",
            "default_domain_artifacts": {"from_defaults": True},
            "web3_ingestion_enabled": False,
            "web3_project_path": ".",
            "web3_run_fuzzing": True,
        },
    )
    monkeypatch.setattr(decomposition_api, "WEB3_INGESTION_AVAILABLE", True)
    monkeypatch.setattr(
        decomposition_api,
        "get_mcp_tool_inventory",
        lambda: {"web3_tools": ["web3_ingest_contract_audit_stack"]},
    )
    monkeypatch.setattr(
        decomposition_api,
        "web3_ingest_contract_audit_stack",
        lambda **_kwargs: {
            "success": True,
            "contracts": ["Vault", "Oracle"],
            "entanglement_matrix": {"Vault": ["Oracle"]},
        },
    )

    request = decomposition_api.DecompositionPlanRequest(
        problem_statement="Audit this Solidity vault for flash loan and oracle manipulation exploits.",
        domain_hint="defi",
        domain_artifacts={"entry_point": "withdraw"},
        web3_ingestion_enabled=True,
        web3_project_path="./contracts",
        web3_run_fuzzing=False,
    )
    result = await decomposition_api.create_decomposition_plan(request)

    assert result["problem"]["domain_context"]["domain"] == "web3"
    assert result["problem"]["metadata"]["domain_hint"] == "web3"
    assert (
        result["plan"]["metadata"]["domain_artifacts"]["entanglement_matrix"]["Vault"]
        == ["Oracle"]
    )
    assert result["plan"]["metadata"]["web3_ingestion"]["success"] is True
    assert result["plan"]["metadata"]["web3"]["project_path"] == "./contracts"
