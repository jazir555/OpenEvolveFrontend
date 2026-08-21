"""
Offline tests for the evolution-context self-adaptive operators.

Covers:
  * Parameters change under stagnation vs. improvement.
  * Parameters always stay within configured bounds.
  * Enabling ``Config.adaptive_parameters`` in a tiny end-to-end evolution run
    (with a stubbed worker) does not crash and actually retunes the live
    database selection ratios.
"""

import asyncio
import uuid

import pytest

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.process_parallel import (
    ProcessParallelController,
    SerializableResult,
)
from openevolve.self_adaptive import SelfAdaptiveOperators


IMPROVING_HISTORY = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
STAGNANT_HISTORY = [0.80, 0.80, 0.80, 0.80, 0.80, 0.80]


def _run_operator(history, population_scores):
    op = SelfAdaptiveOperators()
    for gen, score in enumerate(history):
        op.update(score, population_scores, generation=gen)
    return op


def test_params_change_under_stagnation_vs_improvement():
    # Improvement run with a *spread-out* population -> exploit (low mutation,
    # high selection pressure).
    improving = _run_operator(IMPROVING_HISTORY, [0.1, 0.5, 0.9])
    # Stagnation run with a *collapsed* population -> explore (high mutation,
    # low selection pressure).
    stagnant = _run_operator(STAGNANT_HISTORY, [0.8, 0.8, 0.8])

    assert stagnant.params["mutation_rate"] > improving.params["mutation_rate"]
    assert stagnant.params["selection_pressure"] < improving.params["selection_pressure"]

    # Elitism tracks selection pressure, so it should also differ.
    assert stagnant.params["elitism"] < improving.params["elitism"]


def test_params_stay_within_bounds():
    op = SelfAdaptiveOperators()
    # Hammer it with extreme signals for many generations.
    for gen in range(50):
        score = 0.9 if gen % 2 == 0 else 0.2
        pop = [0.9, 0.9, 0.9] if gen % 2 == 0 else [0.1, 0.2, 0.3]
        op.update(score, pop, generation=gen)

    for key, value in op.get_params().items():
        assert 0.0 <= value <= 1.0, f"{key}={value} out of bounds"


def test_operator_is_deterministic():
    a = _run_operator(IMPROVING_HISTORY, [0.1, 0.5, 0.9])
    b = _run_operator(IMPROVING_HISTORY, [0.1, 0.5, 0.9])
    assert a.get_params() == b.get_params()


def test_flag_off_is_default_disabled():
    cfg = Config()
    assert cfg.adaptive_parameters is False
    assert cfg.self_adaptive.enabled is False


def _make_eval_file(tmp_path):
    path = tmp_path / "eval.py"
    path.write_text("def evaluate(program_path):\n    return {'score': 1.0}\n")
    return str(path)


def _seed_database(database):
    for i in range(6):
        database.add(
            Program(
                id=str(uuid.uuid4()),
                code=f"x={i}",
                language="python",
                generation=0,
                metrics={"score": float(i) / 5.0},
                iteration_found=0,
            )
        )


def test_tiny_evolution_run_with_adaptive_parameters(monkeypatch, tmp_path):
    """Enable the flag and run a few generations with a stubbed worker.

    This exercises the real wiring: the controller builds the
    ``SelfAdaptiveOperators`` from ``Config.adaptive_parameters`` and applies the
    retuned parameters to the live database each generation.
    """

    def fake_worker(iteration, db_snapshot, parent_id, inspiration_ids):
        child = Program(
            id=str(uuid.uuid4()),
            code="x=1",
            language="python",
            parent_id=parent_id,
            generation=1,
            metrics={"score": float(iteration + 1) / 10.0},
            iteration_found=iteration,
            metadata={"island": db_snapshot.get("sampling_island", 0)},
        )
        return SerializableResult(
            child_program_dict=child.to_dict(),
            parent_id=parent_id,
            iteration=iteration,
            target_island=db_snapshot.get("sampling_island"),
        )

    # Avoid constructing any real LLM ensemble / evaluator in in-process mode.
    monkeypatch.setattr(
        "openevolve.process_parallel._lazy_init_worker_components",
        lambda: None,
    )
    monkeypatch.setattr(
        "openevolve.process_parallel._run_iteration_worker", fake_worker
    )

    config = Config(adaptive_parameters=True)
    config.evaluator.parallel_evaluations = 1  # force in-process mode
    config.max_iterations = 6
    config.database.num_islands = 1

    database = ProgramDatabase(config.database)
    _seed_database(database)
    initial_exploration = config.database.exploration_ratio

    controller = ProcessParallelController(
        config, _make_eval_file(tmp_path), database
    )
    # The adapter must have been created from the flag.
    assert controller.self_adaptive is not None

    controller.start()
    try:
        asyncio.run(controller.run_evolution(1, 6, None))
    finally:
        controller.stop()

    # Adaptation must have moved the live selection ratios away from defaults.
    assert controller.self_adaptive.last_metrics is not None
    assert config.database.exploration_ratio != initial_exploration or (
        config.database.elite_selection_ratio != config.self_adaptive.initial_elitism
    )
    # Still valid selection ratios (sum of explore+exploit <= 1).
    assert 0.0 <= config.database.exploration_ratio <= 1.0
    assert 0.0 <= config.database.exploitation_ratio <= 1.0
    assert (
        config.database.exploration_ratio + config.database.exploitation_ratio
        <= 1.0 + 1e-9
    )
