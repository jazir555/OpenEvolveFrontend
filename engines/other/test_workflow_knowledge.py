"""
Offline tests for engines/other/workflow_knowledge.py.

Run:
    python -m pytest engines/other/test_workflow_knowledge.py -q -p no:pytest_ethereum

All tests are offline; no API keys required.
"""

import json
import os
import sys
import tempfile

# Flat-style: make engines/other and the repo root importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from workflow_structures import (  # noqa: E402
    CritiqueReport,
    DecompositionPlan,
    KnowledgeArtifact,
    PerformanceMetrics,
    SolutionAttempt,
    SubProblem,
    VerificationReport,
    WorkflowState,
)

import workflow_knowledge as wk  # noqa: E402


def _build_fake_run():
    """Build minimal real-dataclass objects for a fake successful run."""
    sub = SubProblem(id="sp_1", description="Solve sub problem 1")
    attempt = SolutionAttempt(
        sub_problem_id="sp_1",
        content="def solve(): return 42",
        status="verified",
        quality_metrics={"correctness": 0.9, "clarity": 0.8},
        team_id="blue_team",
        solution_approach="implement_directly",
    )
    critique_fail = CritiqueReport(
        solution_attempt_id="sp_1",
        gauntlet_name="red_gauntlet",
        is_approved=False,
        identified_flaws=[{"type": "ambiguity", "severity": 0.6}],
        overall_score=0.4,
    )
    verify_pass = VerificationReport(
        solution_attempt_id="sp_1",
        gauntlet_name="gold_gauntlet",
        is_approved=True,
        average_score=0.92,
    )
    plan = DecompositionPlan(
        id="plan_1",
        problem_statement="Build a robust calculator.",
        strategy="semantic",
        sub_problems=[sub],
        planner_team_name="planner_team",
        assembler_team_name="assembler_team",
        final_gold_team_gauntlet_name="gold_gauntlet",
    )
    state = WorkflowState(
        workflow_id="wf_1",
        workflow_type="decomposition",
        problem_statement="Build a robust calculator.",
        current_stage="knowledge_extraction",
        decomposition_plan=plan,
        sub_problem_solutions={"sp_1": attempt},
    )
    state.all_critique_reports.append(critique_fail)
    state.all_verification_reports.append(verify_pass)
    return state, plan, [attempt], [critique_fail, verify_pass]


def test_extract_workflow_knowledge_nonempty():
    state, plan, attempts, reports = _build_fake_run()
    artifacts = wk.extract_workflow_knowledge(state, plan, attempts, reports)
    assert isinstance(artifacts, list)
    assert len(artifacts) > 0
    for a in artifacts:
        assert isinstance(a, KnowledgeArtifact)
        assert a.artifact_id
        assert a.source_workflow_id == "wf_1"
        assert a.source_stage == 6
    # Expect at least: decomposition + solution pattern + critique insight + gauntlet.
    types = {a.artifact_type for a in artifacts}
    assert "decomposition_strategy" in types
    assert "solution_pattern" in types
    assert "critique_insight" in types
    assert "gauntlet_effectiveness" in types


def test_jsonl_fallback_written_when_engine_unavailable():
    # Simulate knowledge_engine being absent by forcing ImportError on next import.
    saved = sys.modules.get("knowledge_engine", None)
    sys.modules["knowledge_engine"] = None
    try:
        state, plan, attempts, reports = _build_fake_run()
        tmp = tempfile.mkdtemp()
        fallback = os.path.join(tmp, "fallback.jsonl")
        artifacts = wk.extract_workflow_knowledge(
            state, plan, attempts, reports, fallback_path=fallback
        )
        assert len(artifacts) > 0
        assert os.path.exists(fallback)
        with open(fallback, "r", encoding="utf-8") as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        assert len(lines) == len(artifacts)
        rec = json.loads(lines[0])
        assert "artifact_id" in rec or "artifact_type" in rec
    finally:
        if saved is None:
            sys.modules.pop("knowledge_engine", None)
        else:
            sys.modules["knowledge_engine"] = saved


def test_learning_store_record_and_best():
    store = wk.WorkflowLearningStore()
    store.record_outcome("blue", "g1", "standard", "math", True, 0.9)
    store.record_outcome("blue", "g1", "standard", "math", True, 0.8)
    store.record_outcome("red", "g2", "aggressive", "math", False, 0.2)
    store.record_outcome("blue", "g1", "standard", "logic", True, 0.95)

    best_math = store.best_strategy_for("math")
    assert best_math is not None
    assert best_math["team"] == "blue"
    assert best_math["gauntlet"] == "g1"
    assert best_math["evolution_mode"] == "standard"
    assert best_math["success_rate"] == 1.0

    assert store.best_strategy_for("unknown_type") is None
    assert store.stats()["total_outcomes"] == 4

    # File-backed persistence round-trips.
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "learn.json")
    store2 = wk.WorkflowLearningStore(path=p)
    store2.record_outcome("b", "g", "std", "t", True, 1.0)
    store3 = wk.WorkflowLearningStore(path=p)
    assert store3.best_strategy_for("t") is not None


def test_aggregate_workflow_metrics_sane():
    run = {
        "workflow_id": "wf_x",
        "execution_time": 10.0,
        "total_sub_problems": 4,
        "solved_count": 3,
        "failed_count": 1,
        "error_count": 0,
        "quality_scores": [0.7, 0.8, 0.9],
        "resource_usage": {"tokens": 1234},
        "critiques_total": 4,
        "verifications_total": 3,
        "refinement_loops": 2,
        "step_durations": {"plan": 1.0, "solve": 5.0, "verify": 4.0},
    }
    metrics = wk.aggregate_workflow_metrics(run)
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.workflow_id == "wf_x"
    assert metrics.execution_time == 10.0
    assert abs(metrics.success_rate - 0.75) < 1e-9
    assert metrics.throughput == 0.3
    assert metrics.latency == 2.5
    assert abs(metrics.quality_score - 0.8) < 1e-9
    assert metrics.error_count == 0
    # Serializability.
    serial = json.loads(json.dumps(metrics.to_dict(), default=str))
    assert serial["success_rate"] == 0.75

    steps = wk.collect_step_metrics(run)
    assert steps["sub_problems_total"] == 4
    assert steps["sub_problems_solved"] == 3
    assert abs(steps["avg_quality"] - 0.8) < 1e-9
    assert steps["stage_durations"]["solve"] == 5.0


if __name__ == "__main__":
    test_extract_workflow_knowledge_nonempty()
    test_jsonl_fallback_written_when_engine_unavailable()
    test_learning_store_record_and_best()
    test_aggregate_workflow_metrics_sane()
    print("All flat tests passed.")
