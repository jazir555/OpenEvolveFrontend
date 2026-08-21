"""Offline unit tests for the CMA-ES continuous optimizer.

No LLM / network required. Minimizes simple continuous benchmarks (sphere,
shifted sphere, ill-conditioned ellipsoid, Rosenbrock) and asserts that the
optimum is approached, that the best-so-far history improves monotonically,
and that the selection/config wiring routes to CMA-ES without crashing.
"""

import numpy as np
import pytest

from openevolve.cmaes import (
    CMAES,
    CMAESResult,
    cmaes_selection,
    default_population_size,
    evolve,
    recombination_weights,
    run_cmaes,
)
from openevolve.selection import run_cmaes as run_cmaes_via_selection, select_mo


def sphere(x):
    """Simple quadratic: global minimum 0 at the origin."""
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


def shifted_sphere(x, target=2.5):
    x = np.asarray(x, dtype=float)
    return float(np.sum((x - target) ** 2))


def ellipsoid(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    coeff = 1e3 ** (np.arange(n) / max(n - 1, 1))
    return float(np.sum(coeff * x ** 2))


def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


# --------------------------------------------------------------------------- #
# Core optimization behavior
# --------------------------------------------------------------------------- #
def test_sphere_reaches_optimum():
    result = evolve(sphere, dim=5, generations=200, random_state=0)
    assert isinstance(result, CMAESResult)
    assert result.best_fitness < 1e-8, f"sphere not minimized: {result.best_fitness}"
    assert np.allclose(result.best_solution, np.zeros(5), atol=1e-3)
    assert result.evaluations > 0


def test_history_improves_monotonically():
    result = evolve(sphere, dim=4, generations=60, random_state=1)
    history = result.history
    assert len(history) == result.generations
    assert history[0] >= history[-1], "best-so-far history must not increase"
    assert history[-1] < history[0], "CMA-ES should make progress on the sphere"
    diffs = np.diff(np.asarray(history))
    assert np.all(diffs <= 1e-12), "best-so-far history must be non-increasing"
    # Per-generation bests are recorded alongside the best-so-far curve.
    assert len(result.generation_best) == len(history)


def test_shifted_optimum_is_located():
    result = evolve(
        shifted_sphere,
        dim=3,
        generations=250,
        sigma0=1.0,
        random_state=2,
    )
    assert np.allclose(result.best_solution, np.full(3, 2.5), atol=1e-2), (
        f"expected mean near 2.5, got {result.best_solution}"
    )
    assert result.best_fitness < 1e-6


def test_ill_conditioned_ellipsoid_progresses():
    """Covariance adaptation should handle a 1e3-conditioned quadratic."""
    result = evolve(ellipsoid, dim=5, generations=400, sigma0=1.0, random_state=3)
    assert result.best_fitness < 1e-6, f"ellipsoid fitness {result.best_fitness}"
    # The adapted covariance must become anisotropic (not the identity).
    cond = np.linalg.cond(result.covariance)
    assert cond > 5.0, f"covariance did not adapt (cond={cond})"


def test_rosenbrock_makes_substantial_progress():
    result = evolve(
        rosenbrock,
        dim=4,
        generations=800,
        sigma0=0.3,
        x0=[-1.0, -1.0, -1.0, -1.0],
        random_state=4,
    )
    assert result.best_fitness < 1e-2, f"rosenbrock fitness {result.best_fitness}"


# --------------------------------------------------------------------------- #
# Determinism, bounds, ask/tell API
# --------------------------------------------------------------------------- #
def test_seeded_runs_are_reproducible():
    a = evolve(sphere, dim=3, generations=30, random_state=42)
    b = evolve(sphere, dim=3, generations=30, random_state=42)
    assert a.best_fitness == b.best_fitness
    assert np.allclose(a.best_solution, b.best_solution)


def test_bounds_are_respected_and_constrained_optimum_found():
    lower = np.full(3, 1.0)
    upper = np.full(3, 4.0)
    result = evolve(
        sphere,
        dim=3,
        generations=120,
        sigma0=0.5,
        bounds=(lower, upper),
        random_state=5,
    )
    assert np.all(result.best_solution >= lower - 1e-9)
    assert np.all(result.best_solution <= upper + 1e-9)
    # Constrained optimum of the sphere in [1, 4]^3 is the corner (1, 1, 1).
    assert np.allclose(result.best_solution, lower, atol=1e-2)
    assert np.all(result.mean >= lower - 1e-9) and np.all(result.mean <= upper + 1e-9)


def test_scalar_bounds_spelling():
    result = evolve(sphere, dim=2, generations=40, bounds=(-2.0, 2.0), random_state=6)
    assert np.all(np.abs(result.best_solution) <= 2.0 + 1e-9)


def test_per_dimension_pair_bounds_spelling():
    es = CMAES(dim=3, bounds=[[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], random_state=7)
    assert np.allclose(es.lower, [0.0, 0.0, 0.0])
    assert np.allclose(es.upper, [1.0, 2.0, 3.0])
    samples = es.ask()
    assert np.all(samples >= es.lower - 1e-12)
    assert np.all(samples <= es.upper + 1e-12)


def test_ask_tell_loop_updates_state():
    es = CMAES(dim=4, sigma0=0.6, random_state=8)
    sigma_start = es.sigma
    for _ in range(40):
        xs = es.ask()
        assert xs.shape == (es.pop_size, 4)
        es.tell(xs, [sphere(x) for x in xs])
    assert es.generation == 40
    assert es.evaluations == 40 * es.pop_size
    assert es.sigma != sigma_start
    assert es.best_fitness < sphere(np.ones(4))
    # Evolution paths and covariance have been updated away from their init.
    assert np.linalg.norm(es.p_sigma) > 0.0
    assert not np.allclose(es.C, np.eye(4))


def test_strategy_parameters_are_standard():
    es = CMAES(dim=10, random_state=0)
    assert es.pop_size == default_population_size(10) == 10
    assert es.mu == es.pop_size // 2
    weights = recombination_weights(es.pop_size, es.mu)
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(np.diff(weights) < 0), "weights must decrease with rank"
    assert 1.0 < es.mu_eff <= es.mu
    assert 0.0 < es.c_sigma < 1.0
    assert 0.0 < es.c_c < 1.0
    assert 0.0 < es.c_1 < 1.0
    assert 0.0 <= es.c_mu < 1.0
    assert es.c_1 + es.c_mu <= 1.0


def test_tol_fitness_early_stop():
    result = evolve(sphere, dim=3, generations=1000, tol_fitness=1e-6, random_state=9)
    assert result.stop_reason in ("tol_fitness", "tol_fun", "tol_x")
    assert result.generations < 1000
    assert result.best_fitness <= 1e-6 or result.stop_reason != "tol_fitness"


def test_non_finite_objective_does_not_crash():
    def nasty(x):
        if float(np.sum(x)) > 0:
            return float("nan")
        raise RuntimeError("boom")

    result = evolve(nasty, dim=2, generations=10, random_state=10)
    assert np.isfinite(result.sigma)
    assert result.generations >= 1


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        CMAES(dim=0)
    with pytest.raises(ValueError):
        CMAES(dim=2, sigma0=0.0)
    with pytest.raises(ValueError):
        CMAES(dim=2, bounds=(np.array([1.0, 1.0]), np.array([0.0, 0.0])))


# --------------------------------------------------------------------------- #
# Wiring: selection dispatch + config validation
# --------------------------------------------------------------------------- #
def test_cmaes_selection_keeps_best_ranked_rows():
    objectives = np.array(
        [[5.0, 5.0], [0.0, 0.0], [3.0, 3.0], [1.0, 1.0]], dtype=float
    )
    selected = cmaes_selection(objectives, population_size=2)
    assert selected == [1, 3]
    assert cmaes_selection(objectives, population_size=0) == []
    assert len(cmaes_selection(objectives, population_size=99)) == 4
    assert cmaes_selection(np.empty((0, 2)), population_size=3) == []


def test_select_mo_dispatches_to_cmaes():
    objectives = np.array([[4.0], [1.0], [2.0], [9.0]], dtype=float)
    for name in ("cmaes", "CMA-ES", " cma-es "):
        selected = select_mo(objectives, 2, method=name)
        assert selected == [1, 2], name
    # Existing modes remain intact.
    assert len(select_mo(np.random.default_rng(0).random((8, 2)), 4, method="nsga2")) == 4
    assert len(select_mo(np.random.default_rng(0).random((8, 3)), 4, method="nsga3")) == 4


def test_selection_module_run_cmaes_entrypoint():
    result = run_cmaes_via_selection(sphere, dim=3, generations=50, random_state=11)
    assert result.best_fitness < 1e-3
    direct = run_cmaes(sphere, dim=3, generations=50, random_state=11)
    assert np.isclose(result.best_fitness, direct.best_fitness)


def test_mo_config_accepts_cmaes_and_dispatches():
    from openevolve.unified.config import CMAESConfig, MOConfig

    cfg = MOConfig(selection_method="CMA-ES")
    assert cfg.selection_method == "cmaes"
    objectives = np.array([[7.0], [0.5], [3.0]], dtype=float)
    assert cfg.select(objectives, 1) == [1]

    with pytest.raises(ValueError):
        MOConfig(selection_method="not-a-method")

    # Other selection methods still validate.
    assert MOConfig(selection_method="nsga-iii").selection_method == "nsga3"
    assert MOConfig(selection_method="novelty").selection_method == "novelty_search"

    run_cfg = CMAESConfig(dim=3, generations=60, sigma0=0.5, random_state=12)
    result = run_cfg.run(sphere)
    assert result.best_fitness < 1e-3
    assert result.history[0] >= result.history[-1]
