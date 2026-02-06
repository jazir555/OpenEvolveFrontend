import api_server
from api_server import WorkflowCreateRequest, app
from bubblelabs_security import validate_workflow_type as validate_sec_workflow_type
from bubblelabs_validation import validate_workflow_type as validate_val_workflow_type
from bubblelabs_extended_integration import BubbleLabsExtendedIntegration
from bubblelabs_integration import BubbleLabsIntegration
import decomposition_mcp_tools as decomp_mcp_tools
import bubblelabs_mcp_tools as mcp_tools


def test_web3_api_routes_are_registered():
    paths = {route.path for route in app.routes}
    expected_paths = {
        "/web3/status",
        "/web3/mcp-tool-inventory",
        "/web3/ingest",
        "/web3/ingest/slither",
        "/web3/ingest/foundry",
        "/web3/invariants/translate",
        "/web3/exploits/symbolic-witness",
        "/web3/audit/exploit-verification",
        "/bubblelabs/web3/status",
        "/bubblelabs/web3/ingest",
        "/bubblelabs/web3/invariants/translate",
        "/bubblelabs/web3/exploits/symbolic-witness",
        "/bubblelabs/web3/audit/exploit-verification",
    }
    assert expected_paths.issubset(paths)


def test_workflow_create_request_accepts_web3_config():
    request = WorkflowCreateRequest(
        problem_statement="Audit a Solidity Vault contract for flash loan and reentrancy exploits.",
        content_analyzer_team="content_analyzer",
        planner_team="planner",
        solver_team="solver",
        patcher_team="patcher",
        assembler_team="assembler",
        sub_problem_red_gauntlet="sub_red",
        sub_problem_gold_gauntlet="sub_gold",
        final_red_gauntlet="final_red",
        final_gold_gauntlet="final_gold",
        solver_generation_gauntlet="solver_gen",
        domain_hint="web3",
        web3={
            "enabled": True,
            "project_path": ".",
            "run_fuzzing": True,
        },
    )
    assert request.domain_hint == "web3"
    assert request.web3.get("enabled") is True


def test_workflow_create_request_normalizes_web3_aliases():
    request = WorkflowCreateRequest(
        problem_statement="Audit this smart contract system for flash-loan and oracle attacks.",
        content_analyzer_team="content_analyzer",
        planner_team="planner",
        solver_team="solver",
        patcher_team="patcher",
        assembler_team="assembler",
        sub_problem_red_gauntlet="sub_red",
        sub_problem_gold_gauntlet="sub_gold",
        final_red_gauntlet="final_red",
        final_gold_gauntlet="final_gold",
        solver_generation_gauntlet="solver_gen",
        workflow_type="smart_contract_audit",
        domain_hint="defi",
    )
    assert request.workflow_type == "web3"
    assert request.domain_hint == "web3"


def test_api_create_workflow_sets_web3_runtime_defaults(monkeypatch):
    class _StubTeamManager:
        def get_team(self, _name):
            return object()

    class _StubGauntletManager:
        def get_gauntlet(self, _name):
            return object()

    monkeypatch.setattr(api_server, "get_tenant_team_manager", lambda _tenant: _StubTeamManager())
    monkeypatch.setattr(api_server, "get_tenant_gauntlet_manager", lambda _tenant: _StubGauntletManager())
    monkeypatch.setattr(api_server, "record_audit_event", lambda *args, **kwargs: None)

    original_workflows = dict(api_server.workflows)
    try:
        request = WorkflowCreateRequest(
            problem_statement="Audit this Solidity vault for flash-loan, oracle, and reentrancy exploits.",
            content_analyzer_team="content_analyzer",
            planner_team="planner",
            solver_team="solver",
            patcher_team="patcher",
            assembler_team="assembler",
            sub_problem_red_gauntlet="sub_red",
            sub_problem_gold_gauntlet="sub_gold",
            final_red_gauntlet="final_red",
            final_gold_gauntlet="final_gold",
            solver_generation_gauntlet="solver_gen",
            workflow_type="smart_contract_audit",
            domain_hint="defi",
            web3={"project_path": "./contracts", "run_fuzzing": False},
        )
        user = api_server.AuthUser(api_key="test-key", role=api_server.UserRole.USER, name="tester")
        response = api_server.create_workflow(request=request, user=user, tenant_id="default")
        created = api_server.workflows[response.workflow_id]

        assert created.workflow_type == "web3"
        assert created.openevolve_parameters.get("domain_hint") == "web3"
        assert created.openevolve_parameters.get("web3", {}).get("project_path") == "./contracts"
        assert created.openevolve_parameters.get("formal_verification_mode") == "hybrid"
    finally:
        api_server.workflows.clear()
        api_server.workflows.update(original_workflows)


def test_bubblelabs_web3_status_shape():
    integration = BubbleLabsExtendedIntegration(config={"use_cav_nlp": False})
    status = integration.get_web3_status()
    assert "available" in status
    assert "ingestion_available" in status
    assert "formal_available" in status
    assert "web3_formal_available" in status
    assert "web3_formal_verification_available" in status
    assert "audit_exploit_verification_available" in status
    assert "composite_exploit_verification" in status.get("capabilities", [])
    assert "formal_capabilities" in status
    assert "tool_inventory" in status


def test_bubblelabs_leanaide_bridge_status_exposes_audit_flag():
    import bubblelabs_extended_integration as bubblelabs_ext

    bridge = bubblelabs_ext.LeanAideIntegrationBridge()
    status = bridge.get_status()
    assert "web3_formal_available" in status
    assert "web3_formal_verification_available" in status
    assert "web3_formal_tools" in status
    assert "formal_capabilities" in status
    assert "audit_exploit_verification_available" in status


def test_bubblelabs_security_validation_accepts_web3_aliases():
    assert validate_sec_workflow_type("web3") == "web3"
    assert validate_sec_workflow_type("smart_contract_audit") == "web3"


def test_bubblelabs_validation_accepts_web3_aliases():
    assert validate_val_workflow_type("web3") == "web3"
    assert validate_val_workflow_type("defi") == "web3"


def test_bubblelabs_integration_creates_web3_definition_graph():
    integration = BubbleLabsIntegration()
    definition = integration.create_workflow_definition_from_openevolve(
        problem_statement="Audit vault/oracle contracts and produce exploit witnesses.",
        team_config={
            "content_analyzer_team": "content_analyzer",
            "planner_team": "planner",
            "solver_team": "solver",
            "assembler_team": "assembler",
        },
        gauntlet_config={
            "sub_problem_red_gauntlet": "sub_red",
            "final_gold_gauntlet": "final_gold",
        },
        workflow_type="web3",
        web3_config={"enabled": True, "project_path": "./contracts"},
    )
    node_ids = {node["id"] for node in definition.nodes}
    assert "web3_static_ingestion" in node_ids
    assert "web3_formal_translation" in node_ids
    assert definition.metadata.get("workflow_type") == "openevolve_web3_audit"


def test_mcp_create_workflow_forwards_web3_type_and_config(monkeypatch):
    captured = {}

    class _StubDefinition:
        id = "def-1"
        name = "stub"
        description = "stub"
        nodes = []
        edges = []
        metadata = {}

    class _StubIntegration:
        def create_workflow_definition_from_openevolve(self, **kwargs):
            captured.update(kwargs)
            return _StubDefinition()

    monkeypatch.setattr(mcp_tools, "BUBBLELABS_AVAILABLE", True)
    monkeypatch.setattr(mcp_tools, "get_shared_bubblelabs", lambda: _StubIntegration())
    result = mcp_tools.create_bubblelabs_workflow(
        problem_statement="Audit contracts",
        workflow_type="smart_contract_audit",
        web3_config={"project_path": "./contracts", "run_fuzzing": False},
    )
    assert result["success"] is True
    assert captured.get("workflow_type") == "web3"
    assert captured.get("web3_config", {}).get("project_path") == "./contracts"


def test_mcp_inventory_exposes_web3_formal_tools(monkeypatch):
    monkeypatch.setattr(
        decomp_mcp_tools,
        "_get_web3_formal_inventory",
        lambda: {
            "available": True,
            "tools": [
                "z3_translate_solidity_invariant",
                "z3_solve_smart_contract_exploit_witness",
                "z3_web3_audit_exploit_verification",
            ],
            "formal_capabilities": {
                "solidity_invariant_translation": True,
                "symbolic_exploit_witness": True,
                "composite_exploit_verification": True,
            },
        },
    )
    inventory = decomp_mcp_tools.get_mcp_tool_inventory()
    assert {
        "z3_translate_solidity_invariant",
        "z3_solve_smart_contract_exploit_witness",
        "z3_web3_audit_exploit_verification",
    }.issubset(
        set(inventory.get("web3_formal_tools", []))
    )
    assert {
        "z3_translate_solidity_invariant",
        "z3_solve_smart_contract_exploit_witness",
        "z3_web3_audit_exploit_verification",
    }.issubset(
        set(inventory.get("web3_tools", []))
    )
    assert inventory.get("web3_ingestion_available") is True
    assert inventory.get("formal_capabilities", {}).get("composite_exploit_verification") is True
    assert inventory.get("web3_formal_available") is True
    assert inventory.get("audit_exploit_verification_available") is True


def test_mcp_inventory_infers_formal_tools_from_capabilities(monkeypatch):
    monkeypatch.setattr(
        decomp_mcp_tools,
        "_get_web3_formal_inventory",
        lambda: {
            "available": False,
            "tools": [],
            "formal_capabilities": {
                "solidity_invariant_translation": True,
                "symbolic_exploit_witness": True,
                "composite_exploit_verification": True,
            },
        },
    )
    inventory = decomp_mcp_tools.get_mcp_tool_inventory()
    assert {
        "z3_translate_solidity_invariant",
        "z3_solve_smart_contract_exploit_witness",
        "z3_web3_audit_exploit_verification",
    }.issubset(set(inventory.get("web3_formal_tools", [])))
    assert inventory.get("web3_formal_available") is True
    assert inventory.get("audit_exploit_verification_available") is True
    status = decomp_mcp_tools.get_decomposition_status()
    formal_tools = set(status["mcp_tool_inventory"].get("web3_formal_tools", []))
    assert "z3_web3_audit_exploit_verification" in formal_tools
    assert status["web3_ingestion_available"] is True
    assert status["web3_formal_available"] is True
    assert status["audit_exploit_verification_available"] is True
    assert "web3_ingest_contract_audit_stack" in status["web3_ingestion_tools"]
    assert status["mcp_tool_inventory"]["web3_formal_available"] is True
    assert status["mcp_tool_inventory"]["audit_exploit_verification_available"] is True
    assert status["mcp_tool_inventory"]["formal_capabilities"]["composite_exploit_verification"] is True


def test_api_web3_status_exposes_formal_tools_from_inventory(monkeypatch):
    monkeypatch.setattr(
        api_server,
        "get_mcp_tool_inventory",
        lambda: {
            "web3_tools": [
                "web3_ingest_contract_audit_stack",
                "z3_translate_solidity_invariant",
                "z3_web3_audit_exploit_verification",
            ],
            "web3_ingestion_tools": ["web3_ingest_contract_audit_stack"],
            "web3_formal_tools": [
                "z3_translate_solidity_invariant",
                "z3_web3_audit_exploit_verification",
            ],
            "formal_capabilities": {
                "composite_exploit_verification": True,
            },
        },
    )
    status = api_server.web3_status()
    assert "audit_exploit_verification_available" in status
    assert "web3_formal_available" in status
    assert status["web3_formal_tools"] == [
        "z3_translate_solidity_invariant",
        "z3_web3_audit_exploit_verification",
    ]
    assert status["web3_ingestion_tools"] == ["web3_ingest_contract_audit_stack"]
    assert status["formal_capabilities"]["composite_exploit_verification"] is True


def test_api_web3_status_infers_tool_lists_when_inventory_omits_lists(monkeypatch):
    monkeypatch.setattr(
        api_server,
        "get_mcp_tool_inventory",
        lambda: {
            "formal_capabilities": {
                "solidity_invariant_translation": True,
                "symbolic_exploit_witness": True,
                "composite_exploit_verification": True,
            }
        },
    )
    status = api_server.web3_status()
    assert {
        "z3_translate_solidity_invariant",
        "z3_solve_smart_contract_exploit_witness",
        "z3_web3_audit_exploit_verification",
    }.issubset(set(status["web3_formal_tools"]))
    assert "web3_ingest_contract_audit_stack" in status["web3_ingestion_tools"]


def test_api_web3_status_infers_available_flag_from_formal_capabilities(monkeypatch):
    monkeypatch.setattr(api_server, "WEB3_INGESTION_AVAILABLE", False)
    monkeypatch.setattr(api_server, "WEB3_FORMAL_VERIFICATION_AVAILABLE", False)
    monkeypatch.setattr(
        api_server,
        "get_mcp_tool_inventory",
        lambda: {
            "web3_tools": [],
            "web3_ingestion_tools": [],
            "web3_formal_tools": [],
            "formal_capabilities": {
                "solidity_invariant_translation": True,
                "symbolic_exploit_witness": True,
                "composite_exploit_verification": True,
            },
        },
    )
    status = api_server.web3_status()
    assert status["available"] is True
    assert status["web3_formal_verification_available"] is True
    assert status["web3_formal_available"] is True
    assert status["audit_exploit_verification_available"] is True


def test_api_web3_audit_endpoint_returns_verified_exploit(monkeypatch):
    monkeypatch.setattr(api_server, "WEB3_INGESTION_AVAILABLE", False)
    monkeypatch.setattr(api_server, "WEB3_FORMAL_VERIFICATION_AVAILABLE", True)
    monkeypatch.setattr(
        api_server,
        "translate_solidity_assignment_to_z3",
        lambda **kwargs: {"constraints": ["new_balance == old_balance - amount"], "invariants": ["new_balance >= 0"]},
    )
    monkeypatch.setattr(
        api_server,
        "verify_solidity_invariant_translation",
        lambda **kwargs: {"proven": True},
    )
    monkeypatch.setattr(
        api_server,
        "solve_smart_contract_exploit_witness",
        lambda **kwargs: {"satisfiable": True, "model": {"amount": 1}},
    )
    request = api_server.Web3AuditExploitRequest(
        project_path="./contracts",
        run_fuzzing=False,
        statement="balance[msg.sender] -= amount;",
        verify_translation=True,
    )
    user = api_server.AuthUser(api_key="test-key", role=api_server.UserRole.USER, name="tester")
    result = api_server.web3_audit_exploit_verification(request=request, user=user)
    assert result["verified_exploit"] is True


def test_z3_mcp_server_registers_web3_formal_tools():
    from z3_mcp_tools import get_z3_mcp_server

    server = get_z3_mcp_server()
    tool_names = {tool["name"] for tool in server.list_tools()}
    assert {
        "z3_translate_solidity_invariant",
        "z3_solve_smart_contract_exploit_witness",
        "z3_web3_audit_exploit_verification",
    }.issubset(tool_names)


def test_z3_leanaide_classifier_detects_web3_audit_as_hybrid():
    from z3_leanaide_openevolve_integration import (
        IntegratedProblemClassifier,
        ProblemCategory,
        WorkflowIntegrationConfig,
    )

    classifier = IntegratedProblemClassifier(WorkflowIntegrationConfig())
    result = classifier.classify(
        "Audit this Solidity vault for flash-loan and reentrancy exploits and verify invariants in Lean."
    )
    assert result.category == ProblemCategory.HYBRID
    assert result.recommended_solver == "combined"


def test_bubblelabs_web3_audit_endpoint_forwards_request(monkeypatch):
    captured = {}

    class _StubIntegration:
        def web3_audit_exploit_verification(self, **kwargs):
            captured.update(kwargs)
            return {"success": True, "source": "bubblelabs_stub"}

    monkeypatch.setattr(api_server, "BUBBLELABS_AVAILABLE", True)
    monkeypatch.setattr(api_server, "get_extended_integration", lambda: _StubIntegration())

    request = api_server.Web3AuditExploitRequest(
        project_path="./contracts",
        run_fuzzing=False,
        statement="balance[msg.sender] -= amount;",
        non_negative_target=True,
        max_withdraw_expr="deposits[msg.sender] + yield[msg.sender]",
        verify_translation=True,
        assume_non_negative_amount=True,
        additional_constraints=["contract_balance_post < contract_balance_pre"],
        timeout_seconds=12.0,
    )
    user = api_server.AuthUser(api_key="test-key", role=api_server.UserRole.USER, name="tester")
    result = api_server.bubblelabs_web3_audit_exploit_verification(request=request, user=user)

    assert result["success"] is True
    assert result["source"] == "bubblelabs_stub"
    assert captured["project_path"] == "./contracts"
    assert captured["run_fuzzing"] is False
    assert captured["statement"] == "balance[msg.sender] -= amount;"
    assert captured["timeout_seconds"] == 12.0


def test_bubblelabs_extended_integration_web3_audit_orchestration(monkeypatch):
    integration = BubbleLabsExtendedIntegration(config={"use_cav_nlp": False})

    monkeypatch.setattr(
        integration,
        "web3_ingest_contract_stack",
        lambda **kwargs: {"success": True, "phase": "ingestion", "kwargs": kwargs},
    )
    monkeypatch.setattr(
        integration,
        "web3_translate_solidity_invariant",
        lambda **kwargs: {"success": True, "phase": "translation", "kwargs": kwargs},
    )
    monkeypatch.setattr(
        integration,
        "web3_solve_exploit_witness",
        lambda **kwargs: {
            "success": True,
            "phase": "witness",
            "kwargs": kwargs,
            "result": {"satisfiable": True, "model": {"amount": 1}},
        },
    )

    result = integration.web3_audit_exploit_verification(
        project_path="./contracts",
        run_fuzzing=False,
        statement="balance[msg.sender] -= amount;",
        additional_constraints=["user_deposit == 0"],
        timeout_seconds=9.5,
    )

    assert result["success"] is True
    assert result["ingestion"]["phase"] == "ingestion"
    assert result["translation"]["phase"] == "translation"
    assert result["exploit_witness"]["phase"] == "witness"
    assert result["verified_exploit"] is True


def test_bubblelabs_web3_status_infers_available_from_formal_capabilities(monkeypatch):
    integration = BubbleLabsExtendedIntegration(config={"use_cav_nlp": False})
    import bubblelabs_extended_integration as bubblelabs_ext

    monkeypatch.setattr(bubblelabs_ext, "WEB3_INGESTION_AVAILABLE", False)
    monkeypatch.setattr(bubblelabs_ext, "WEB3_FORMAL_AVAILABLE", False)
    monkeypatch.setattr(
        bubblelabs_ext,
        "get_mcp_tool_inventory",
        lambda: {
            "web3_tools": [],
            "web3_ingestion_tools": [],
            "web3_formal_tools": [],
            "formal_capabilities": {
                "solidity_invariant_translation": True,
                "symbolic_exploit_witness": True,
                "composite_exploit_verification": True,
            },
        },
    )
    status = integration.get_web3_status()
    assert status["available"] is True
    assert status["formal_available"] is True
    assert status["audit_exploit_verification_available"] is True
