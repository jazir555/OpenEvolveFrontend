"""
Offline tests for ``workflow_resources.py`` (§6.2) and ``workflow_distributed.py``
(§6.3). No network / API keys required.

Run with:
    python -m pytest engines/other/test_workflow_resources_distributed.py -q \
        -p no:pytest_ethereum
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# Flat-style imports: make engines/other and repo root importable.
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for _p in (_THIS, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from workflow_resources import ResourceBudget, ResourceManager, ResourceExhaustedError
from workflow_distributed import SubProblemExecutor


# ----------------------------------------------------------------------
# Fake sub-problem + deterministic offline solve fns (module-level so they
# are picklable for the multiprocessing backend).
# ----------------------------------------------------------------------
@dataclass
class FakeSub:
    id: str
    dependencies: List[str] = field(default_factory=list)
    estimated_resources: Optional[dict] = None


def _solve_deterministic(sub, voter):
    """Deterministic, offline solve: result depends only on the sub-problem."""
    return {
        "sub_problem_id": sub.id,
        "content": f"solution-for-{sub.id}",
        "deps": list(sub.dependencies),
    }


def _solve_with_pid(sub, voter):
    """Records the pid that solved it so we can prove parallelism."""
    return {
        "sub_problem_id": sub.id,
        "content": f"solution-for-{sub.id}",
        "solving_pid": os.getpid(),
    }


def _make_dag():
    """A->B, A->C, B->D, C->D  (A independent root; B,C independent of each other)."""
    return [
        FakeSub(id="A"),
        FakeSub(id="B", dependencies=["A"]),
        FakeSub(id="C", dependencies=["A"]),
        FakeSub(id="D", dependencies=["B", "C"]),
    ]


# ======================================================================
# ResourceManager (§6.2)
# ======================================================================
def test_resource_budget_enforcement_tokens():
    mgr = ResourceManager(
        ResourceBudget(total_tokens=100, tokens_per_sub_problem=30)
    )
    mgr.consume(tokens=30)
    mgr.consume(tokens=30)
    mgr.consume(tokens=30)
    assert mgr.remaining_tokens() == 10
    # One more 30-token consume would exceed -> raises.
    raised = False
    try:
        mgr.consume_or_raise(tokens=30)
    except ResourceExhaustedError:
        raised = True
    assert raised
    assert mgr.remaining_tokens() == 10


def test_resource_budget_enforcement_steps_and_time():
    mgr = ResourceManager(
        ResourceBudget(
            total_steps=3,
            steps_per_sub_problem=1,
            total_time_seconds=2.0,
            time_per_sub_problem=1.0,
        )
    )
    mgr.consume_or_raise(steps=1, time_seconds=1.0)
    mgr.consume_or_raise(steps=1, time_seconds=1.0)
    assert mgr.remaining_steps() == 1
    assert mgr.remaining_time() == 0.0
    # Time fully spent -> either dimension blocks.
    raised = False
    try:
        mgr.consume_or_raise(steps=1, time_seconds=1.0)
    except ResourceExhaustedError:
        raised = True
    assert raised


def test_resource_unbounded_dimension_ok():
    mgr = ResourceManager(ResourceBudget(total_tokens=None, total_steps=None))
    assert mgr.remaining_tokens() is None
    assert mgr.remaining_steps() is None
    for _ in range(1000):
        mgr.consume_or_raise(steps=1, tokens=5)
    assert mgr.used.steps == 1000


def test_recommend_batch_size_sane():
    # Abundant budget -> capped by parallelism and pending count.
    mgr = ResourceManager(
        ResourceBudget(
            total_tokens=10_000,
            tokens_per_sub_problem=10,
            total_steps=10_000,
            steps_per_sub_problem=1,
            max_parallel=4,
        )
    )
    assert mgr.recommend_batch_size(pending_count=10) == 4
    assert mgr.recommend_batch_size(pending_count=2) == 2
    assert mgr.recommend_batch_size(pending_count=0) == 0

    # Tight step budget -> limited by headroom.
    mgr2 = ResourceManager(
        ResourceBudget(total_steps=5, steps_per_sub_problem=2, max_parallel=4)
    )
    assert mgr2.recommend_batch_size(pending_count=10) == 2  # floor(5/2)=2
    mgr2.consume_or_raise(steps=4)  # headroom now 1 -> floor(1/2)=0
    assert mgr2.recommend_batch_size(pending_count=10) == 0


# ======================================================================
# SubProblemExecutor (§6.3)
# ======================================================================
def test_executor_local_solves_all_and_respects_order():
    ex = SubProblemExecutor(default_backend="local")
    result = ex.execute(_make_dag(), _solve_deterministic)
    assert result.backend_used == "local"
    assert set(result.solutions.keys()) == {"A", "B", "C", "D"}
    # Dependency order respected.
    order = result.solve_order
    pos = {sid: i for i, sid in enumerate(order)}
    for sid in ("B", "C", "D"):
        for dep in _make_dag_map()[sid].dependencies:
            assert pos[dep] < pos[sid], f"{dep} must precede {sid}"
    # Independent B and C both ran.
    assert "B" in result.solutions and "C" in result.solutions


def test_executor_local_cycle_detection():
    cyclic = [
        FakeSub(id="X", dependencies=["Y"]),
        FakeSub(id="Y", dependencies=["X"]),
    ]
    ex = SubProblemExecutor(default_backend="local")
    raised = False
    try:
        ex.topological_layers(cyclic)
    except ValueError:
        raised = True
    assert raised


def test_executor_multiprocessing_identical_to_local():
    dag = _make_dag()
    local = SubProblemExecutor(default_backend="local").execute(dag, _solve_deterministic)
    mp = SubProblemExecutor(default_backend="multiprocessing", max_workers=2).execute(
        dag, _solve_deterministic
    )
    # Backend actually used multiprocessing (not silently degraded).
    assert mp.backend_used == "multiprocessing"
    assert not mp.degraded_to_local
    # Results identical (deterministic solve -> same output on every backend).
    assert mp.solutions == local.solutions
    assert mp.solve_order == local.solve_order
    assert set(mp.solutions.keys()) == {"A", "B", "C", "D"}


def test_executor_multiprocessing_runs_independent_nodes():
    # Two fully independent nodes -> should execute in separate worker processes.
    dag = [FakeSub(id="P"), FakeSub(id="Q")]
    mp = SubProblemExecutor(default_backend="multiprocessing", max_workers=2).execute(
        dag, _solve_with_pid
    )
    assert mp.backend_used == "multiprocessing"
    pids = {mp.solutions["P"]["solving_pid"], mp.solutions["Q"]["solving_pid"]}
    # Both independent nodes must have executed in worker processes, not the
    # main process.
    assert pids  # non-empty
    assert os.getpid() not in pids
    # NOTE: ProcessPoolExecutor may reuse a single worker for a 2-task layer if
    # the first task finishes before the second is submitted, so distinct PIDs
    # are not guaranteed. The test_executor_multiprocessing_identical_to_local
    # test above already proves the multiprocessing backend is genuinely used
    # (backend_used == "multiprocessing", degraded_to_local == False).


def _make_dag_map():
    return {sp.id: sp for sp in _make_dag()}
