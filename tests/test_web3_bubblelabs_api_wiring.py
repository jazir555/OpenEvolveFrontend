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
