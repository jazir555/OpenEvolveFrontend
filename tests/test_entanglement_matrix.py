import pytest

from dataclasses import dataclass, field
from typing import Dict, Any, List

from utils.entanglement_utils import build_symbolic_entanglement_matrix, serialize_entanglement_matrix
from universal_recomposition_engine import (
    UniversalRecompositionEngine,
    DecompositionPlan,
    ProblemDefinition,
    SubProblem as UniversalSubProblem,
    ComplexityScore,
    SuccessCriterion,
    Constraint,
    SubProblemSolution,
)
from decomposition_recomposition_integration import SolutionSolver, SolverConfig
from workflow_structures import Team, ModelConfig


@dataclass
class _MiniSubProblem:
    id: str
    title: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def test_build_symbolic_entanglement_matrix_is_symmetric():
    sub_problems = [
        _MiniSubProblem(id="sp1", title="Define Foo", description="Expose Foo interface"),
        _MiniSubProblem(id="sp2", title="Use Foo", description="Integrate Foo API"),
        _MiniSubProblem(id="sp3", title="Independent", description="No overlap here"),
    ]
    matrix, symbols_by_id = build_symbolic_entanglement_matrix(sub_problems)
    serialized = serialize_entanglement_matrix(matrix)

    assert "sp1" in serialized and "sp2" in serialized
    assert "sp2" in serialized["sp1"]
    assert "sp1" in serialized["sp2"]
    assert "sp3" in serialized
    assert "sp1" in symbols_by_id and "sp2" in symbols_by_id


def test_entanglement_invalidation_propagates():
    problem = ProblemDefinition(
        id="prob1",
        title="Entanglement Test",
        description="Test entanglement invalidation",
        domain="software",
        complexity_score=ComplexityScore(1, 1, 1, 1, 1),
        constraints=[Constraint(id="c1", description="none", type="general", severity="soft")],
        success_criteria=[SuccessCriterion(id="s1", description="ok", metric="done", threshold=1.0)],
    )
    sp1 = UniversalSubProblem(
        id="sp1",
        parent_id="prob1",
        title="Component A",
        description="Must enable feature X",
        type="implementation",
        complexity_score=ComplexityScore(1, 1, 1, 1, 1),
        dependencies=[],
        success_criteria=[],
    )
    sp2 = UniversalSubProblem(
        id="sp2",
        parent_id="prob1",
        title="Component B",
        description="Must not enable feature X",
        type="implementation",
        complexity_score=ComplexityScore(1, 1, 1, 1, 1),
        dependencies=[],
        success_criteria=[],
    )
    plan = DecompositionPlan(
        id="plan1",
        original_problem=problem,
        sub_problems=[sp1, sp2],
        strategy_used="semantic",
        dependency_graph={},
        execution_order=["sp1", "sp2"],
        metadata={
            "entanglement_matrix": {"sp1": ["sp2"], "sp2": ["sp1"]}
        },
    )

    sub_solutions = {
        "sp1": SubProblemSolution(
            sub_problem_id="sp1",
            solution_content="We must enable feature X.",
            quality_score=0.8,
        ),
        "sp2": SubProblemSolution(
            sub_problem_id="sp2",
            solution_content="We must not enable feature X.",
            quality_score=0.8,
        ),
    }

    engine = UniversalRecompositionEngine()
    result = engine.assemble(
        plan=plan,
        sub_solutions=sub_solutions,
        detect_conflicts=True,
        resolve_conflicts=False,
    )

    sp1_meta = result.sub_solutions["sp1"].metadata
    sp2_meta = result.sub_solutions["sp2"].metadata

    assert sp1_meta.get("needs_consistency_refinement") is True
    assert sp2_meta.get("needs_consistency_refinement") is True
    assert "entanglement_invalidation" in sp1_meta
    assert "entanglement_invalidation" in sp2_meta


def test_simple_solver_propagates_entanglement_metadata():
    # Create a minimal ModelConfig for the team
    model_config = ModelConfig(
        model_id="test-model",
        api_key="test-key"
    )

    # Create solver with properly configured Team
    solver = SolutionSolver(
        team=Team(
            name="test_team",
            role="Blue",
            members=[model_config]
        ),
        config=SolverConfig(
            mode="decomposition_first",
            max_decomposition_depth=3,
            max_subproblems=5
        )
    )

    # The solve method expects a string problem description and context dict,
    # not a SubProblem object. Pass entanglement metadata in context.
    problem_description = "Expose authentication interface for client integrations."
    context = {
        "entangled_with": ["sp-beta"],
        "entanglement_symbols": ["auth", "token"],
        "entanglement_source": "symbolic_overlap",
        "input_contracts": ["User credentials"],
        "output_contracts": ["Access token"],
    }

    solution_text, metadata = solver.solve(problem_description, context)

    # Verify entanglement metadata is propagated in the returned metadata
    assert metadata.get("entangled_with") == ["sp-beta"]
    assert metadata.get("entanglement_symbols") == ["auth", "token"]
    assert metadata.get("entanglement_source") == "symbolic_overlap"
    assert metadata.get("input_contracts") == ["User credentials"]
    assert metadata.get("output_contracts") == ["Access token"]
