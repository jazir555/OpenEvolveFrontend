"""
Unit tests for adaptive strategy selector tracker behavior.
"""

from adaptive_strategy_selector import StrategyPerformanceTracker


def test_record_attempt_first_entry_updates_without_division_error() -> None:
    tracker = StrategyPerformanceTracker()

    tracker.record_attempt(
        strategy_name="api_health",
        success=True,
        quality_score=100.0,
        metadata={"endpoint": "health"},
    )

    data = tracker.get_strategy_data("api_health")
    assert data is not None
    assert data.total_attempts == 1
    assert data.success_count == 1
    assert data.failure_count == 0
    assert data.average_quality == 100.0
