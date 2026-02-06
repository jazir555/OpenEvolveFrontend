from api_server import WorkflowCreateRequest, app
from bubblelabs_security import validate_workflow_type as validate_sec_workflow_type
from bubblelabs_validation import validate_workflow_type as validate_val_workflow_type
from bubblelabs_extended_integration import BubbleLabsExtendedIntegration


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


def test_bubblelabs_web3_status_shape():
    integration = BubbleLabsExtendedIntegration(config={"use_cav_nlp": False})
    status = integration.get_web3_status()
    assert "available" in status
    assert "ingestion_available" in status
    assert "formal_available" in status
    assert "tool_inventory" in status


def test_bubblelabs_security_validation_accepts_web3_aliases():
    assert validate_sec_workflow_type("web3") == "web3"
    assert validate_sec_workflow_type("smart_contract_audit") == "web3"


def test_bubblelabs_validation_accepts_web3_aliases():
    assert validate_val_workflow_type("web3") == "web3"
    assert validate_val_workflow_type("defi") == "web3"
