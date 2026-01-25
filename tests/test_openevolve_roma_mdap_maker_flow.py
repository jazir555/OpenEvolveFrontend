import json
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from generic_maker_integration import GenericSolution
import hephaestus_integration
from hephaestus_integration import HephaestusWorkflowSync
from workflow_structures import ModelConfig, Team, WorkflowState


def _load_workflow_engine():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    engine_path = repo_root / "workflow_engine.py"
    spec = importlib.util.spec_from_file_location("workflow_engine_local", engine_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


workflow_engine = _load_workflow_engine()


def test_recursive_decomposition_and_hephaestus_labels(monkeypatch):
    from hephaestus_unified_bridge import get_unified_bridge_status

    status = get_unified_bridge_status()
    assert status["roma_mdap_maker_bridge_available"] is True

    call_count = {"value": 0}

    async def _fake_run_generic_maker(*args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            payload = {
                "is_atomic": False,
                "subproblem_1": {
                    "description": "Design module A",
                    "ai_suggested_evolution_mode": "standard",
                    "ai_suggested_complexity_score": 5,
                    "ai_suggested_evaluation_prompt": "Check module A completeness"
                },
                "subproblem_2": {
                    "description": "Design module B",
                    "ai_suggested_evolution_mode": "standard",
                    "ai_suggested_complexity_score": 5,
                    "ai_suggested_evaluation_prompt": "Check module B completeness"
                },
                "composition": "Merge module A and B outputs into a final design",
                "complexity": 7
            }
        else:
            payload = {
                "is_atomic": True,
                "subproblem_1": None,
                "subproblem_2": None,
                "composition": "",
                "complexity": 3
            }

        return GenericSolution(
            task_id="test",
            solution=json.dumps(payload),
            quality_score=0.9
        )

    monkeypatch.setattr(workflow_engine, "run_generic_maker", _fake_run_generic_maker)

    team = Team(
        name="Planner",
        role="Blue",
        members=[ModelConfig(model_id="gpt-test", api_key="test-key")],
        decomposition_system_prompt="Decompose the task",
        decomposition_user_prompt_template="Decompose: {{problem_statement}}\n{{analyzed_context}}"
    )

    workflow_state = WorkflowState(
        workflow_id="wf_test",
        workflow_type="sovereign",
        problem_statement="Build a system",
        current_stage="AI-Assisted Decomposition",
        maker_enabled=True,
        mdap_enabled=True,
        maker_config={"decomposition_depth": 2}
    )

    analyzed_context = {
        "summary": "Test",
        "mdap_enabled": True,
        "maker_enabled": True,
        "mdap_config": {},
        "maker_config": {"decomposition_depth": 2}
    }

    plan = workflow_engine.run_decomposition_with_reliability(
        workflow_state.problem_statement,
        analyzed_context,
        team,
        workflow_state
    )

    assert len(plan.sub_problems) == 3
    sub1, sub2, composition = plan.sub_problems
    assert composition.dependencies == [sub1.id, sub2.id]
    assert "compose subproblem results" in composition.description.lower()

    mock_client = SimpleNamespace()
    created = {}

    def _create_ticket(title, description, ticket_type=None, assignee=None, labels=None):
        created[title] = {"labels": labels or []}
        return f"ticket-{len(created)}"

    mock_client.create_ticket = _create_ticket
    sync = HephaestusWorkflowSync(mock_client)
    sync.create_subproblem_tickets("wf_test", plan.sub_problems, workflow_epic_id="epic-1")

    composition_ticket = next(
        labels for title, labels in created.items() if "Compose subproblem results" in title
    )
    label_set = set(composition_ticket["labels"])
    assert f"depends-on-{sub1.id}" in label_set
    assert f"depends-on-{sub2.id}" in label_set
    assert "composition-node" in label_set


def test_full_flow_with_mocked_hephaestus(monkeypatch):
    call_count = {"value": 0}

    async def _fake_run_generic_maker(*args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            payload = {
                "is_atomic": False,
                "subproblem_1": {
                    "description": "Analyze requirements",
                    "ai_suggested_evolution_mode": "standard",
                    "ai_suggested_complexity_score": 5,
                    "ai_suggested_evaluation_prompt": "Check analysis completeness"
                },
                "subproblem_2": {
                    "description": "Draft solution",
                    "ai_suggested_evolution_mode": "standard",
                    "ai_suggested_complexity_score": 5,
                    "ai_suggested_evaluation_prompt": "Check draft quality"
                },
                "composition": "Combine analysis and draft into final response",
                "complexity": 7
            }
        else:
            payload = {
                "is_atomic": True,
                "subproblem_1": None,
                "subproblem_2": None,
                "composition": "",
                "complexity": 3
            }

        return GenericSolution(
            task_id="test",
            solution=json.dumps(payload),
            quality_score=0.9
        )

    class FakeHephaestusClient:
        def __init__(self, api_base, api_key, project_id):
            self.created = {}
            self.updated = []

        def create_ticket(self, title, description, ticket_type=None, assignee=None, labels=None):
            ticket_id = f"ticket-{len(self.created) + 1}"
            self.created[title] = {"labels": labels or [], "description": description}
            return ticket_id

        def update_ticket(self, ticket_id, status=None, assignee=None, description=None):
            self.updated.append(
                {
                    "ticket_id": ticket_id,
                    "status": status,
                    "assignee": assignee,
                    "description": description,
                }
            )
            return True

        def get_ticket(self, ticket_id):
            return {"id": ticket_id, "status": "todo"}

        def get_tickets_by_label(self, label):
            return []

    monkeypatch.setattr(workflow_engine, "run_generic_maker", _fake_run_generic_maker)
    monkeypatch.setattr(hephaestus_integration, "HephaestusClient", FakeHephaestusClient)

    team = Team(
        name="Planner",
        role="Blue",
        members=[ModelConfig(model_id="gpt-test", api_key="test-key")],
        decomposition_system_prompt="Decompose the task",
        decomposition_user_prompt_template="Decompose: {{problem_statement}}\n{{analyzed_context}}"
    )

    workflow_state = WorkflowState(
        workflow_id="wf_mock",
        workflow_type="sovereign",
        problem_statement="Build a system",
        current_stage="AI-Assisted Decomposition",
        maker_enabled=True,
        mdap_enabled=True,
        maker_config={"decomposition_depth": 2}
    )

    analyzed_context = {
        "summary": "Test",
        "mdap_enabled": True,
        "maker_enabled": True,
        "mdap_config": {},
        "maker_config": {"decomposition_depth": 2}
    }

    workflow_state.decomposition_plan = workflow_engine.run_decomposition_with_reliability(
        workflow_state.problem_statement,
        analyzed_context,
        team,
        workflow_state
    )

    manager = hephaestus_integration.setup_hephaestus_integration(
        workflow_state,
        api_base="http://hephaestus.local",
        api_key="mock-key",
        project_id="mock-project"
    )

    assert manager is not None
    assert workflow_state.hephaestus_workflow_id
    assert len(workflow_state.id_to_ticket_id_map) == len(workflow_state.decomposition_plan.sub_problems)

    epic_labels = list(manager.client.created.values())[0]["labels"]
    assert "mdap-enabled" in epic_labels
    assert "maker-enabled" in epic_labels
    assert "composition-nodes" in epic_labels

    composition_titles = [
        title for title in manager.client.created
        if "Compose subproblem results" in title
    ]
    assert composition_titles
