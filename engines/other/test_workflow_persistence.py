"""
Round-trip acceptance test for the OpenEvolve workflow run persistence layer.

Verifies:
  * pickle -> unpickle of a realistic WorkflowState (Set, Enum, datetime-like)
    including the ad-hoc ``error`` attribute
  * upsert_run / get_run / list_runs / delete_run
  * append_audit / get_audit_logs

Run:  pytest engines/other/test_workflow_persistence.py
       (use ``-p no:pytest_ethereum`` if the repo-wide web3 plugin breaks collection)
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

# Ensure repo root + flat module dir are importable (so ``openevolve`` and the
# sibling modules resolve).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openevolve.kernel.schema import (  # noqa: E402
    WorkflowState,
    SubProblem,
    SubProblemType,
    SubProblemStatus,
    DecompositionPlan,
    DecompositionStrategy,
)

from workflow_persistence import WorkflowRunStore  # noqa: E402


def _make_state():
    """Build a realistic WorkflowState with nested dataclasses / Set / Enum."""
    sub = SubProblem(
        id="sp-1",
        title="Sub-problem one",
        description="Solve the thing",
        type=SubProblemType.ANALYSIS,
        status=SubProblemStatus.PENDING,
        dependencies=["sp-0"],
    )
    state = WorkflowState(
        workflow_id="wf-test-001",
        workflow_type="sovereign_decomposition",
        problem_statement="Prove the Riemann hypothesis lite",
        current_stage="INITIALIZING",
        tenant_id="tenant-xyz",
        status="running",
        progress=0.42,
        start_time=1700000000.123,
    )
    # Set, Enum, nested dataclass, and an attrdict of solutions.
    state.solved_sub_problem_ids = {"sp-0", "sp-1"}
    state.decomposition_plan = DecompositionPlan(
        problem_statement="Prove the Riemann hypothesis lite",
        strategy=DecompositionStrategy.SEMANTIC,
        sub_problems=[sub],
    )
    state.error = "boom"  # ad-hoc attribute set at runtime by the engine
    return state


def _fresh_store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)  # let the store create it cleanly
    store = WorkflowRunStore(db_path=path)
    store.init_database()
    store.apply_migrations()
    return store, path


def test_pickle_roundtrip_preserves_fields():
    import pickle

    state = _make_state()
    restored = pickle.loads(pickle.dumps(state))

    assert restored.workflow_id == "wf-test-001"
    assert restored.problem_statement == "Prove the Riemann hypothesis lite"
    assert restored.status == "running"
    assert restored.progress == 0.42
    assert restored.start_time == 1700000000.123
    # Set survives
    assert isinstance(restored.solved_sub_problem_ids, set)
    assert restored.solved_sub_problem_ids == {"sp-0", "sp-1"}
    # Enum survives (nested dataclass field)
    assert restored.decomposition_plan.sub_problems[0].type == SubProblemType.ANALYSIS
    # ad-hoc error attribute survives
    assert getattr(restored, "error", None) == "boom"


def test_upsert_get_list_delete():
    store, path = _fresh_store()
    try:
        state = _make_state()
        store.upsert_run(
            state,
            scalars={"created_at": "2024-01-01T00:00:00", "updated_at": "2024-01-02T00:00:00"},
        )

        # get_run
        got = store.get_run("wf-test-001")
        assert got is not None
        assert got.workflow_id == "wf-test-001"
        assert got.solved_sub_problem_ids == {"sp-0", "sp-1"}
        assert getattr(got, "error", None) == "boom"
        assert got.start_time == 1700000000.123

        # list_runs
        listed = store.list_runs()
        assert len(listed) == 1
        assert listed[0].workflow_id == "wf-test-001"

        # tenant-scoped list
        assert len(store.list_runs("tenant-xyz")) == 1
        assert len(store.list_runs("other-tenant")) == 0

        # upsert again should update, not duplicate
        state.status = "completed"
        state.progress = 1.0
        store.upsert_run(state)
        assert len(store.list_runs()) == 1
        assert store.get_run("wf-test-001").status == "completed"

        # delete
        assert store.delete_run("wf-test-001") is True
        assert store.get_run("wf-test-001") is None
        assert len(store.list_runs()) == 0
        # deleting a missing id is a no-op
        assert store.delete_run("nope") is False
    finally:
        store.get_connection().close()
        for ext in ("", "-wal", "-shm"):
            try:
                os.remove(path + ext)
            except OSError:
                pass


def test_audit_log_roundtrip():
    store, path = _fresh_store()
    try:
        store.append_audit({
            "timestamp": "2024-01-01T00:00:00",
            "user": "alice",
            "role": "admin",
            "operation": "CREATE_WORKFLOW",
            "resource": "workflow",
            "resource_id": "wf-test-001",
            "success": True,
            "details": {"tenant_id": "tenant-xyz", "n": 3},
        })
        logs = store.get_audit_logs()
        assert len(logs) == 1
        entry = logs[0]
        assert entry["user"] == "alice"
        assert entry["operation"] == "CREATE_WORKFLOW"
        assert entry["success"] is True
        assert entry["details"] == {"tenant_id": "tenant-xyz", "n": 3}
    finally:
        store.get_connection().close()
        for ext in ("", "-wal", "-shm"):
            try:
                os.remove(path + ext)
            except OSError:
                pass


def test_get_run_missing_returns_none():
    store, path = _fresh_store()
    try:
        assert store.get_run("does-not-exist") is None
    finally:
        store.get_connection().close()
        for ext in ("", "-wal", "-shm"):
            try:
                os.remove(path + ext)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
