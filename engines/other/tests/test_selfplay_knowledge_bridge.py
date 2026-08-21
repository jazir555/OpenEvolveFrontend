import sys
from pathlib import Path

# Make ``engines/other`` importable so the bridge (and psv_selfplay) resolve.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "engines" / "other"))

from psv_selfplay import MathematicalProblem, SolutionAttempt  # noqa: E402
from selfplay_knowledge_bridge import (  # noqa: E402
    generate_knowledge_enhanced_specification,
    solve_with_knowledge_context,
    verify_with_knowledge,
)


class FakeKnowledgeEngine:
    """Lightweight, network-free stand-in for the real KnowledgeEngine."""

    def __init__(self, hits=None):
        self.hits = hits or [
            {"text": "Similar problem: solve x^2 + y^2 = 25 for integers."},
            {"summary": "Use substitution for Diophantine equations."},
        ]
        self.retrieve_calls = []

    def retrieve_knowledge(self, query, limit=10):
        self.retrieve_calls.append((query, limit))
        return list(self.hits)


class MissingApiEngine:
    """Engine that has no compatible query/retrieve method at all."""

    pass


def _make_problem(statement="Find integer solutions to x^2 + y^2 = 25."):
    return MathematicalProblem(
        id="p1",
        statement=statement,
        domain="number_theory",
        difficulty=0.5,
    )


def test_generate_enhanced_specification_calls_engine():
    engine = FakeKnowledgeEngine()
    problem = _make_problem()
    result = generate_knowledge_enhanced_specification(problem, engine)

    assert engine.retrieve_calls, "engine.retrieve_knowledge was not called"
    assert result["problem_id"] == "p1"
    assert result["domain"] == "number_theory"
    assert result["enhanced"] is True
    assert "Knowledge-Enhanced Context" in result["enhanced_specification"]
    assert result["knowledge"]["available"] is True
    assert result["knowledge"]["via"] == "retrieve_knowledge"
    assert len(result["sources"]) == len(engine.hits)


def test_solve_with_knowledge_context_calls_engine():
    engine = FakeKnowledgeEngine()
    problem = _make_problem()
    def my_solver():
        pass

    result = solve_with_knowledge_context(problem, engine, solver_model=my_solver)

    assert engine.retrieve_calls
    assert result["context_available"] is True
    assert result["solver_model"] == "my_solver"
    assert len(result["solver_hints"]) == len(engine.hits)
    assert "Retrieved knowledge" in result["solver_context"]
    assert result["knowledge"]["engine_type"] == "FakeKnowledgeEngine"


def test_verify_with_knowledge_supports_facts():
    # One fact exactly matches the solution text -> supported.
    engine = FakeKnowledgeEngine(
        hits=[{"text": "The answer is x=3, y=4"}]
    )
    solution = SolutionAttempt(
        problem_id="p1",
        solution="We find x=3 and y=4. The answer is x=3, y=4.",
        solver_id="solver-1",
    )
    result = verify_with_knowledge(solution, engine)

    assert engine.retrieve_calls
    assert result["solution_id"] == "solver-1"
    assert result["verified_by_knowledge"] is True
    assert "The answer is x=3, y=4" in result["supported_facts"]
    assert result["conflicts"] == []


def test_verify_with_knowledge_reports_conflicts():
    engine = FakeKnowledgeEngine(
        hits=[{"text": "The correct result is 42"}]
    )
    solution = SolutionAttempt(
        problem_id="p1",
        solution="My result is 7.",
        solver_id="solver-2",
    )
    result = verify_with_knowledge(solution, engine)
    assert result["verified_by_knowledge"] is False
    assert result["knowledge_available"] is True
    assert "The correct result is 42" in result["conflicts"]


def test_graceful_degradation_no_engine():
    problem = _make_problem()
    spec = generate_knowledge_enhanced_specification(problem, None)
    assert spec["enhanced"] is False
    assert spec["knowledge"]["available"] is False
    assert "Knowledge-Enhanced Context" not in spec["enhanced_specification"]

    ctx = solve_with_knowledge_context(problem, None)
    assert ctx["context_available"] is False
    assert "No knowledge retrieved" in ctx["solver_context"]

    solution = SolutionAttempt(problem_id="p1", solution="x=3", solver_id="s3")
    ver = verify_with_knowledge(solution, None)
    assert ver["knowledge_available"] is False
    assert ver["verified_by_knowledge"] is False
    assert ver["conflicts"] == []


def test_graceful_degradation_engine_without_api():
    engine = MissingApiEngine()
    problem = _make_problem()
    spec = generate_knowledge_enhanced_specification(problem, engine)
    assert spec["knowledge"]["available"] is False
    assert spec["knowledge"]["engine_type"] == "MissingApiEngine"
    assert spec["enhanced"] is False

    ver = verify_with_knowledge(
        SolutionAttempt(problem_id="p1", solution="x=3", solver_id="s4"), engine
    )
    assert ver["knowledge_available"] is False
    assert ver["verified_by_knowledge"] is False
