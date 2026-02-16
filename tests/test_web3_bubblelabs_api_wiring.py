import pytest
import time

pytestmark = pytest.mark.timeout(60)  # This test has slow imports

from openevolve_bubblelabs_api import OpenEvolveBubbleLabsIntegration, validate_workflow_type


def test_validate_workflow_type_supports_web3_aliases():
    assert validate_workflow_type("web3") == "web3"
    assert validate_workflow_type("smart_contract_audit") == "web3"
    assert validate_workflow_type("defi") == "web3"


def test_web3_workflow_definition_injects_default_web3_parameters():
    integration = OpenEvolveBubbleLabsIntegration()
    definition_id = integration.create_workflow_definition(
        name="web3-audit",
        description="Audit smart contracts",
        workflow_type="web3",
        parameters={},
    )
    definition = integration.get_workflow_definition(definition_id)
    assert definition is not None
    params = definition["parameters"]
    assert params.get("domain_hint") == "web3"
    assert isinstance(params.get("web3"), dict)
    assert params.get("formal_verification_mode") == "hybrid"


def test_web3_workflow_instance_forwards_web3_parameters_to_runtime_state():
    integration = OpenEvolveBubbleLabsIntegration()
    definition_id = integration.create_workflow_definition(
        name="web3-runtime",
        description="Runtime forwarding",
        workflow_type="web3",
        parameters={
            "web3": {"enabled": True, "project_path": ".", "run_fuzzing": True},
            "domain_artifacts": {"contracts": ["Vault"]},
            "solver_generation_gauntlet": "dummy_solver_gauntlet",
        },
    )
    instance_id = integration.create_workflow_instance(
        definition_id=definition_id,
        instance_name="instance-1",
        inputs={"problem_statement": "Audit Vault for reentrancy"},
    )
    state = integration.workflow_instances[instance_id]
    assert state.workflow_type == "sovereign"
    assert state.openevolve_parameters.get("domain_hint") == "web3"
    assert state.openevolve_parameters.get("domain_artifacts", {}).get("contracts") == ["Vault"]
    assert state.openevolve_parameters.get("web3", {}).get("enabled") is True


def test_sync_parameters_accepts_web3_payload():
    integration = OpenEvolveBubbleLabsIntegration()
    definition_id = integration.create_workflow_definition(
        name="web3-sync",
        description="Sync test",
        workflow_type="web3",
        parameters={},
    )
    instance_id = integration.create_workflow_instance(
        definition_id=definition_id,
        instance_name="instance-sync",
        inputs={"problem_statement": "Audit oracle/vault entanglement"},
    )
    result = integration.sync_parameters_to_workflow(
        instance_id=instance_id,
        parameters={
            "domain_artifacts": {"dependencies": {"Vault": ["Oracle"]}},
            "web3": {"enabled": True, "project_path": "./contracts"},
        },
    )
    assert "updated_count" in result
    state = integration.workflow_instances[instance_id]
    assert state.openevolve_parameters.get("domain_artifacts", {}).get("dependencies", {}).get("Vault") == ["Oracle"]
    assert state.openevolve_parameters.get("web3", {}).get("project_path") == "./contracts"


def test_sync_safe_parameter_is_available_in_runtime_store_when_not_schema_attribute():
    integration = OpenEvolveBubbleLabsIntegration()
    definition_id = integration.create_workflow_definition(
        name="default-sync",
        description="Default sync test",
        workflow_type="default",
        parameters={},
    )
    instance_id = integration.create_workflow_instance(
        definition_id=definition_id,
        instance_name="instance-default",
        inputs={"problem_statement": "Improve edge-case handling"},
    )
    result = integration.sync_parameters_to_workflow(
        instance_id=instance_id,
        parameters={"max_iterations": 77},
    )
    assert result["updated_count"] >= 1
    state = integration.workflow_instances[instance_id]
    assert state.openevolve_parameters.get("max_iterations") == 77


def test_workflow_lifecycle_start_to_completion_with_thread_execution(monkeypatch):
    integration = OpenEvolveBubbleLabsIntegration()
    definition_id = integration.create_workflow_definition(
        name="lifecycle-default",
        description="Lifecycle path",
        workflow_type="default",
        parameters={},
    )
    instance_id = integration.create_workflow_instance(
        definition_id=definition_id,
        instance_name="instance-life",
        inputs={"problem_statement": "Run lifecycle flow"},
    )

    def fake_execute(workflow_state):
        workflow_state.status = "running"
        workflow_state.current_stage = "fake_execute"
        workflow_state.progress = 1.0
        workflow_state.execution_time = 0.01
        workflow_state.end_time = time.time()
        workflow_state.status = "completed"

    monkeypatch.setattr(integration, "_execute_workflow_thread", fake_execute)
    start_result = integration.start_workflow_instance(instance_id)
    assert start_result["status"] in {"pending", "completed"}

    thread = integration.running_threads[instance_id]
    thread.join(timeout=2)

    status = integration.get_workflow_instance_status(instance_id)
    assert status["status"] == "completed"
    assert status["current_stage"] == "fake_execute"


def test_restart_preserves_openevolve_runtime_parameters(monkeypatch):
    integration = OpenEvolveBubbleLabsIntegration()
    definition_id = integration.create_workflow_definition(
        name="restart-web3",
        description="restart state copy",
        workflow_type="web3",
        parameters={
            "web3": {"enabled": True, "project_path": "./contracts"},
            "domain_artifacts": {"contracts": ["Vault"]},
            "max_iterations": 25,
        },
    )
    instance_id = integration.create_workflow_instance(
        definition_id=definition_id,
        instance_name="instance-restart",
        inputs={"problem_statement": "Audit Vault"},
    )

    monkeypatch.setattr(
        integration,
        "start_workflow_instance",
        lambda new_id: {"message": "Workflow started", "instance_id": new_id, "status": "pending"},
    )
    result = integration.restart_workflow_instance(instance_id)
    new_instance_id = result["new_instance_id"]
    new_state = integration.workflow_instances[new_instance_id]

    assert new_state.openevolve_parameters.get("web3", {}).get("project_path") == "./contracts"
    assert new_state.openevolve_parameters.get("domain_artifacts", {}).get("contracts") == ["Vault"]
    assert new_state.openevolve_parameters.get("max_iterations") == 25


def test_running_thread_is_cleaned_up_after_workflow_completes(monkeypatch):
    integration = OpenEvolveBubbleLabsIntegration()
    definition_id = integration.create_workflow_definition(
        name="cleanup-default",
        description="thread cleanup",
        workflow_type="default",
        parameters={},
    )
    instance_id = integration.create_workflow_instance(
        definition_id=definition_id,
        instance_name="instance-cleanup",
        inputs={"problem_statement": "Check thread cleanup"},
    )

    import evolution

    monkeypatch.setattr(evolution, "run_evolution_loop", lambda *args, **kwargs: "ok")
    integration.start_workflow_instance(instance_id)
    thread = integration.running_threads.get(instance_id)
    if thread is not None:
        thread.join(timeout=2)

    assert instance_id not in integration.running_threads
    assert integration.workflow_instances[instance_id].status == "completed"
