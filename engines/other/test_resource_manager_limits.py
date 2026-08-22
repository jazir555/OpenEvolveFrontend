"""
Tests for the per-sub-problem / step / parallelism enforcement added to
``ResourceManager`` and ``ResourceLimits``.
"""
from __future__ import annotations

import pytest

from resource_manager import (
    ResourceManager,
    ResourceLimits,
    ResourceLimitExceeded,
    create_resource_limits_from_config,
)


def test_max_parallel_enforced_no_overshoot():
    rm = ResourceManager(
        ResourceLimits(max_parallel=2, allow_overshoot=False)
    )
    rm.acquire_slot()
    rm.acquire_slot()
    with pytest.raises(ResourceLimitExceeded):
        rm.acquire_slot()


def test_max_parallel_enforced_overshoot():
    rm = ResourceManager(
        ResourceLimits(max_parallel=1, allow_overshoot=True)
    )
    rm.acquire_slot()
    # Should not raise when overshoot allowed.
    rm.acquire_slot()
    rm.release_slot()
    rm.release_slot()
    assert rm._parallel_active == 0


def test_max_parallel_sub_problem_slot_contextmanager():
    rm = ResourceManager(ResourceLimits(max_parallel=1, allow_overshoot=False))
    with rm.sub_problem_slot():
        pass
    assert rm._parallel_active == 0
    with pytest.raises(ResourceLimitExceeded):
        with rm.sub_problem_slot():
            with rm.sub_problem_slot():
                pass


def test_steps_per_sub_problem_enforced():
    rm = ResourceManager(ResourceLimits(steps_per_sub_problem=2, allow_overshoot=False))
    rm.record_sub_problem("sp1", steps=1)
    rm.record_sub_problem("sp1", steps=1)
    # Cumulative is now 2, so the next step trips it.
    with pytest.raises(ResourceLimitExceeded):
        rm.record_sub_problem("sp1", steps=1)


def test_steps_per_sub_problem_overshoot():
    rm = ResourceManager(ResourceLimits(steps_per_sub_problem=1, allow_overshoot=True))
    rm.record_sub_problem("sp1", steps=1)
    rm.record_sub_problem("sp1", steps=5)  # overshoots but allowed
    assert rm._per_sub_problem["sp1"]["steps"] == 6


def test_tokens_per_sub_problem_enforced():
    rm = ResourceManager(ResourceLimits(tokens_per_sub_problem=10, allow_overshoot=False))
    rm.record_sub_problem("sp1", tokens=6)
    with pytest.raises(ResourceLimitExceeded):
        rm.record_sub_problem("sp1", tokens=6)


def test_time_per_sub_problem_enforced():
    rm = ResourceManager(ResourceLimits(time_per_sub_problem_seconds=1.0, allow_overshoot=False))
    rm.record_sub_problem("sp1", seconds=0.6)
    with pytest.raises(ResourceLimitExceeded):
        rm.record_sub_problem("sp1", seconds=0.6)


def test_max_steps_total_enforced():
    rm = ResourceManager(ResourceLimits(max_steps=3, allow_overshoot=False))
    rm.record_sub_problem("a", steps=1)
    rm.record_sub_problem("b", steps=1)
    rm.record_sub_problem("c", steps=1)
    with pytest.raises(ResourceLimitExceeded):
        rm.record_sub_problem("d", steps=1)


def test_usage_summary_exposes_new_fields():
    rm = ResourceManager(ResourceLimits(max_parallel=4, allow_overshoot=True))
    rm.acquire_slot()
    rm.record_sub_problem("sp1", tokens=5, steps=1, seconds=0.5)
    summary = rm.get_usage_summary()
    assert summary["steps"] == 1
    assert summary["parallel_active"] == 1
    assert summary["computed_time_seconds"] == 0.5
    assert summary["allow_overshoot"] is True
    assert summary["per_sub_problem_usage"]["sp1"]["tokens"] == 5


def test_create_from_config_maps_decomposition_keys():
    cfg = {
        "total_steps": 10,
        "max_parallel": 3,
        "tokens_per_sub_problem": 100,
        "time_per_sub_problem": 5.0,
        "steps_per_sub_problem": 2,
        "allow_overshoot": True,
        "total_tokens": 500,
        "total_time_seconds": 60,
    }
    limits = create_resource_limits_from_config(cfg)
    assert limits.max_steps == 10
    assert limits.max_parallel == 3
    assert limits.tokens_per_sub_problem == 100
    assert limits.time_per_sub_problem_seconds == 5.0
    assert limits.steps_per_sub_problem == 2
    assert limits.allow_overshoot is True
    assert limits.max_tokens == 500
    assert limits.max_execution_time_seconds == 60


def test_create_from_config_empty_returns_no_limits():
    limits = create_resource_limits_from_config({})
    assert isinstance(limits, ResourceLimits)
    assert limits.max_steps is None
    assert limits.max_parallel is None
