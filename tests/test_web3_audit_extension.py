import pytest

from decomposition_mcp_tools import get_mcp_tool_inventory
from enhanced_recomposition_engine import (
    EnhancedRecompositionEngine,
    RecompositionConfig,
    SubProblemSolution,
)
from universal_decomposition_engine import ProblemDomain, UniversalDecompositionEngine
from z3prover_integration import translate_solidity_assignment_to_z3


def test_universal_decomposition_applies_web3_extension():
    engine = UniversalDecompositionEngine()
    plan = engine.decompose(
        problem_statement=(
            "Audit a Solidity Vault contract entangled with an Oracle and AMM pool. "
            "Use Slither and Foundry to detect reentrancy and flash loan exploits."
        ),
        domain=ProblemDomain.WEB3,
        max_subproblems=8,
    )

    assert "web3" in plan.metadata.get("domain_extensions_applied", [])
    assert any(sp.metadata.get("domain_extension") == "web3" for sp in plan.sub_problems)
    assert "web3" in plan.metadata
    assert "dependency_hints" in plan.metadata["web3"]


def test_solidity_assignment_translation_produces_expected_invariants():
    translation = translate_solidity_assignment_to_z3("balance[msg.sender] -= amount;")
    constraints = translation["constraints"]
    invariants = translation["invariants"]
    variable_names = {v["name"] for v in translation["variables"]}

    assert "new_balance == old_balance - (amount)" in constraints
    assert "new_balance >= 0" in invariants
    assert {"old_balance", "new_balance", "amount"}.issubset(variable_names)


def test_mcp_inventory_exposes_web3_ingestion_tools():
    inventory = get_mcp_tool_inventory()
    expected_tools = {
        "web3_ingest_slither_static_analysis",
        "web3_ingest_foundry_fuzzing",
        "web3_ingest_contract_audit_stack",
    }
    assert expected_tools.issubset(set(inventory["tools"]))
    assert expected_tools.issubset(set(inventory["web3_tools"]))


def test_enhanced_recomposition_runs_defi_gauntlet_for_web3_context():
    config = RecompositionConfig(enable_defi_gauntlet=True)
    engine = EnhancedRecompositionEngine(config=config)

    sub_solutions = {
        "sp_withdraw": SubProblemSolution(
            sub_problem_id="sp_withdraw",
            solution_content=(
                "function withdraw(uint amount) external { "
                "msg.sender.call{value: amount}(\"\"); "
                "balance[msg.sender] -= amount; }"
            ),
            quality_score=0.8,
            completeness=0.8,
            correctness=0.8,
            clarity=0.8,
            metadata={"domain_extension": "web3"},
            keywords=["Withdraw"],
        )
    }

    solution = engine.assemble(
        sub_solutions=sub_solutions,
        problem_id="web3_problem",
        decomposition_plan_id="web3_plan",
        dependency_graph={},
    )

    assert "defi_gauntlet" in solution.metadata
    result = solution.metadata["defi_gauntlet"]
    vector_ids = {v["id"] for v in result["attack_vectors"]}
    assert {"flash_loan_attack", "reentrancy_attack", "symbolic_execution_probe"}.issubset(vector_ids)
    assert result["high_findings"] >= 1
