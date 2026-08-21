"""
Offline tests for the adaptive metric and dynamic strategy selection.

These feed synthetic fitness/population histories and assert that:
  - ``compute_adaptive_metric`` returns a real number in [0, 1];
  - the adaptive metric correctly distinguishes improving vs. stagnating runs;
  - ``select_strategy`` returns a valid, sensible choice that explores more
    under stagnation and exploits under convergence.
"""

import math

from openevolve.config.config_metrics import (
    compute_adaptive_metric,
    compute_adaptive_metrics,
)
from openevolve.config.dynamic_strategy import (
    DynamicStrategySwitcher,
    SystemMode,
    select_strategy,
)


IMPROVING_HISTORY = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
STAGNANT_HISTORY = [0.80, 0.80, 0.80, 0.80, 0.80, 0.80]
REGRESSING_HISTORY = [0.80, 0.70, 0.60, 0.50, 0.40, 0.30]


def test_adaptive_metric_is_real_number():
    metric = compute_adaptive_metric(IMPROVING_HISTORY)
    assert isinstance(metric, float)
    assert math.isfinite(metric)
    assert 0.0 <= metric <= 1.0


def test_adaptive_metric_distinguishes_runs():
    improving = compute_adaptive_metric(IMPROVING_HISTORY)
    stagnant = compute_adaptive_metric(STAGNANT_HISTORY)
    regressing = compute_adaptive_metric(REGRESSING_HISTORY)

    # Improving run shows progress -> low stagnation metric.
    assert improving == 0.0
    # Stagnant run never improves -> fully stagnated.
    assert stagnant == 1.0
    # Regression is also treated as stagnation.
    assert regressing >= 0.8
    assert stagnant > improving


def test_adaptive_metrics_detail():
    detail = compute_adaptive_metrics(IMPROVING_HISTORY, population_scores=[0.1, 0.4, 0.7])
    assert detail.stagnation_generations == 0
    assert detail.convergence_slope > 0
    assert detail.diversity > 0

    detail_stag = compute_adaptive_metrics(STAGNANT_HISTORY)
    assert detail_stag.stagnation_generations == len(STAGNANT_HISTORY) - 1


def test_adaptive_metric_insufficient_data():
    metric = compute_adaptive_metric([0.5])
    assert metric == 0.0


def test_select_strategy_explores_under_stagnation():
    stagnant = select_strategy(stagnation_index=1.0, diversity=0.0)
    converging = select_strategy(stagnation_index=0.0, diversity=0.5)

    # Stagnation must drive more exploration than convergence.
    assert stagnant["exploration"] > converging["exploration"]
    assert stagnant["mutation_rate"] > converging["mutation_rate"]
    assert stagnant["selection_pressure"] < converging["selection_pressure"]

    # Low diversity + stagnation should switch to a diversity-seeking mode.
    assert stagnant["strategy"] == SystemMode.QD

    # A converging run should stay on the current (default) strategy.
    assert converging["strategy"] == SystemMode.OPENEVOLVE


def test_select_strategy_returns_valid_dict():
    result = select_strategy(stagnation_index=0.3, diversity=0.2)
    assert isinstance(result, dict)
    assert "strategy" in result
    assert "mutation_rate" in result
    assert "crossover_rate" in result
    assert "selection_pressure" in result
    assert "exploration" in result
    assert "reason" in result
    assert isinstance(result["strategy"], SystemMode)
    assert 0.0 <= result["mutation_rate"] <= 1.0
    assert 0.0 <= result["crossover_rate"] <= 1.0
    assert 0.0 <= result["selection_pressure"] <= 1.0


def test_select_strategy_is_deterministic():
    a = select_strategy(stagnation_index=0.7, diversity=0.1)
    b = select_strategy(stagnation_index=0.7, diversity=0.1)
    assert a == b


def test_select_strategy_switcher_method():
    switcher = DynamicStrategySwitcher(SystemMode.OPENEVOLVE)
    result = switcher.select_strategy(stagnation_index=1.0, diversity=0.0)
    assert isinstance(result, dict)
    assert result["strategy"] == SystemMode.QD


def test_capture_current_state_uses_real_metrics():
    switcher = DynamicStrategySwitcher(SystemMode.OPENEVOLVE)
    switcher.current_state = {"fitness_history": STAGNANT_HISTORY}
    import asyncio
    state = asyncio.run(switcher._capture_current_state())
    assert state["metrics"]["stagnation_index"] == 1.0
