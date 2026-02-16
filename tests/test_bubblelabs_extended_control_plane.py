from types import SimpleNamespace
from pathlib import Path

from bubblelabs_extended_integration import BubbleLabsExtendedIntegration


def test_control_catalog_exposes_components(monkeypatch):
    integration = BubbleLabsExtendedIntegration(config={"use_cav_nlp": False})

    def fake_initialize_all():
        integration._initialized = True
        return {}

    monkeypatch.setattr(integration, "initialize_all", fake_initialize_all)
    catalog = integration.get_control_catalog()

    assert catalog["success"] is True
    components = catalog["components"]
    assert "ace" in components
    assert "web3" in components
    assert "cav_nlp" in components
    assert "openevolve_workflows" in components
    assert "create_skillbook" in components["ace"]
    assert "audit_exploit_verification" in components["web3"]
    assert "create_instance" in components["openevolve_workflows"]


def test_execute_control_action_dispatches_to_component_method():
    integration = BubbleLabsExtendedIntegration(config={"use_cav_nlp": False})
    integration._initialized = True
    integration._ace_bridge = SimpleNamespace(
        create_skillbook=lambda name, skills: {"success": True, "name": name, "skills_count": len(skills)},
        extract_patterns=lambda workflow_results: {"success": True, "patterns_extracted": len(workflow_results)},
    )

    result = integration.execute_control_action(
        component="ace",
        action="create_skillbook",
        payload={"name": "integration-skillbook", "skills": [{"id": "s1"}]},
    )

    assert result["success"] is True
    assert result["component"] == "ace"
    assert result["action"] == "create_skillbook"
    assert result["result"]["name"] == "integration-skillbook"
    assert result["result"]["skills_count"] == 1


def test_execute_control_action_handles_unknown_component_and_action():
    integration = BubbleLabsExtendedIntegration(config={"use_cav_nlp": False})
    integration._initialized = True

    unknown_component = integration.execute_control_action(
        component="does_not_exist",
        action="anything",
        payload={},
    )
    assert unknown_component["success"] is False
    assert "Unknown component" in unknown_component["error"]

    unknown_action = integration.execute_control_action(
        component="ace",
        action="does_not_exist",
        payload={},
    )
    assert unknown_action["success"] is False
    assert "Unknown action" in unknown_action["error"]


def test_execute_control_action_initializes_bridge_on_demand(monkeypatch):
    integration = BubbleLabsExtendedIntegration(config={"use_cav_nlp": False})
    integration._initialized = True
    integration._z3_bridge = None

    z3_stub = SimpleNamespace(
        solve_constraints=lambda variables, constraints: {
            "success": True,
            "variables_count": len(variables),
            "constraints_count": len(constraints),
        }
    )

    monkeypatch.setattr(integration, "_init_bridge_by_name", lambda name: z3_stub if name == "z3" else None)

    result = integration.execute_control_action(
        component="z3",
        action="solve_constraints",
        payload={"variables": [{"name": "x"}], "constraints": ["x > 0"]},
    )

    assert result["success"] is True
    assert result["result"]["variables_count"] == 1
    assert result["result"]["constraints_count"] == 1


def test_auto_discovery_adds_module_to_catalog(monkeypatch, tmp_path: Path):
    module_path = tmp_path / "openevolve_sample_integration.py"
    module_path.write_text(
        "\n".join(
            [
                "def get_status(target: str = 'ok'):",
                "    return {'success': True, 'target': target}",
            ]
        ),
        encoding="utf-8",
    )

    integration = BubbleLabsExtendedIntegration(
        config={"use_cav_nlp": False, "auto_discovery_root": str(tmp_path)}
    )
    integration._initialized = True
    monkeypatch.setattr(
        integration,
        "_discover_integration_module_files",
        lambda: [module_path],
    )

    discovery = integration.refresh_auto_discovery(force=True)
    assert discovery["success"] is True
    assert discovery["components"] == 1

    catalog = integration.get_control_catalog()
    assert "openevolve_sample_integration" in catalog["components"]
    assert "get_status" in catalog["components"]["openevolve_sample_integration"]


def test_execute_control_action_runs_auto_discovered_function(monkeypatch, tmp_path: Path):
    module_path = tmp_path / "bubblelabs_runtime_integration.py"
    module_path.write_text(
        "\n".join(
            [
                "def run_check(value: int):",
                "    return {'success': True, 'value': value}",
            ]
        ),
        encoding="utf-8",
    )

    integration = BubbleLabsExtendedIntegration(
        config={"use_cav_nlp": False, "auto_discovery_root": str(tmp_path)}
    )
    integration._initialized = True
    monkeypatch.setattr(
        integration,
        "_discover_integration_module_files",
        lambda: [module_path],
    )
    integration.refresh_auto_discovery(force=True)

    result = integration.execute_control_action(
        component="bubblelabs_runtime_integration",
        action="run_check",
        payload={"value": 7},
    )

    assert result["success"] is True
    assert result["auto_discovered"] is True
    assert result["result"]["value"] == 7


def test_execute_control_action_openevolve_workflows(monkeypatch):
    class StubWorkflowIntegration:
        def create_workflow_definition(self, name, description, workflow_type, parameters):
            _ = name, description, workflow_type, parameters
            return "def-123"

        def create_workflow_instance(self, definition_id, instance_name, inputs, parameters=None):
            _ = definition_id, instance_name, inputs, parameters
            return "inst-456"

        def list_workflow_definitions(self):
            return [{"id": "def-123"}]

        def list_workflow_instances(self):
            return [{"instance_id": "inst-456"}]

        def get_workflow_instance_status(self, instance_id):
            return {"instance_id": instance_id, "status": "created"}

        def start_workflow_instance(self, instance_id):
            return {"instance_id": instance_id, "status": "pending"}

        def sync_parameters_to_workflow(self, instance_id, parameters):
            return {"instance_id": instance_id, "updated_count": len(parameters)}

    integration = BubbleLabsExtendedIntegration(config={"use_cav_nlp": False})
    integration._initialized = True
    monkeypatch.setattr(
        integration,
        "_get_openevolve_workflow_integration",
        lambda: StubWorkflowIntegration(),
    )

    create_def = integration.execute_control_action(
        component="openevolve_workflows",
        action="create_definition",
        payload={"name": "wf", "description": "d", "workflow_type": "sovereign", "parameters": {}},
    )
    assert create_def["success"] is True
    assert create_def["result"]["definition_id"] == "def-123"

    create_inst = integration.execute_control_action(
        component="openevolve_workflows",
        action="create_instance",
        payload={"definition_id": "def-123", "instance_name": "inst", "inputs": {}},
    )
    assert create_inst["success"] is True
    assert create_inst["result"]["instance_id"] == "inst-456"

    start_inst = integration.execute_control_action(
        component="openevolve_workflows",
        action="start_instance",
        payload={"instance_id": "inst-456"},
    )
    assert start_inst["success"] is True
    assert start_inst["result"]["status"] == "pending"

    sync_params = integration.execute_control_action(
        component="openevolve_workflows",
        action="sync_parameters",
        payload={"instance_id": "inst-456", "parameters": {"max_iterations": 10}},
    )
    assert sync_params["success"] is True
    assert sync_params["result"]["updated_count"] == 1
