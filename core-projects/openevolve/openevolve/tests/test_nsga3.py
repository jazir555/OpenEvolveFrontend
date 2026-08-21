"""Offline unit tests for NSGA-III selection (no LLM required)."""

import numpy as np
from typing import Sequence

from openevolve.nsga3 import (
    NSGAIII,
    fast_non_dominated_sort,
    generate_reference_points,
    nsga3_selection,
)
from openevolve.selection import nsga2_selection, select_mo
from openevolve.unified.config import MOConfig


def _selected_not_dominated_by_others(
    objectives: np.ndarray, selected: Sequence[int]
) -> bool:
    """No non-selected individual may dominate a selected individual.

    This is the correct environmental-selection invariant: kept lower fronts
    may be dominated by kept higher fronts, but nothing that was *dropped*
    should dominate something that was *kept*.
    """
    obj = np.asarray(objectives, dtype=float)
    all_idx = set(range(obj.shape[0]))
    dropped = all_idx - set(selected)
    for s in selected:
        for t in dropped:
            if np.all(obj[t] <= obj[s]) and np.any(obj[t] < obj[s]):
                return False
    return True


def _random_objectives(n, m, seed):
    rng = np.random.default_rng(seed)
    return rng.random((n, m))


def test_reference_point_generation():
    rp = generate_reference_points(3, 4)
    assert rp.shape[0] == 15
    assert np.allclose(np.sum(rp, axis=1), 1.0)
    assert np.all(rp >= 0)


def test_reference_point_layered():
    rp = generate_reference_points(3, [12, 4])
    assert rp.shape[0] == 91
    assert np.allclose(np.sum(rp, axis=1), 1.0)


def test_fast_non_dominated_sort():
    obj = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [0.0, 1.0],
        [1.0, 0.0],
    ])
    fronts = fast_non_dominated_sort(obj)
    assert fronts[0] == [0]
    assert fronts[-1] == [1]


def test_nsga3_selects_non_dominated():
    rng = np.random.default_rng(0)
    combined = rng.random((200, 3))
    selected = nsga3_selection(combined, population_size=100, divisions=4, random_state=1)
    assert len(selected) == 100
    assert _selected_not_dominated_by_others(combined, selected)


def test_nsga3_reference_points_associated():
    m = NSGAIII(num_objectives=3, divisions=4, population_size=100)
    rng = np.random.default_rng(2)
    combined = rng.random((200, 3))
    sel = m.environmental_selection(combined, random_state=3)
    assert len(sel) == 100
    ideal = np.min(combined, axis=0)
    intercepts = np.array([combined[:, j].max() for j in range(3)])
    membership, _ = _associate_alias(combined, m.reference_points, ideal, intercepts)
    used = set(membership.tolist())
    assert len(used) > 1


def _associate_alias(objectives, reference_points, ideal, intercepts):
    from openevolve.nsga3 import _associate
    return _associate(objectives, reference_points, ideal, intercepts)


def test_nsga3_runs_generations():
    rng = np.random.default_rng(7)
    pop = rng.random((100, 3))
    for _ in range(5):
        offspring = np.clip(pop + rng.normal(0, 0.05, pop.shape), 0, 1)
        combined = np.vstack([pop, offspring])
        sel = nsga3_selection(combined, population_size=100, divisions=4, random_state=11)
        pop = combined[sel]
    assert pop.shape == (100, 3)
    assert _selected_not_dominated_by_others(pop, range(len(pop)))


def test_nsga2_kept_working():
    obj = np.random.default_rng(4).random((120, 2))
    sel = nsga2_selection(obj, population_size=50, random_state=5)
    assert len(sel) == 50
    assert _selected_not_dominated_by_others(obj, sel)


def test_dispatch_routes_nsga3():
    obj = np.random.default_rng(8).random((150, 3))
    sel = select_mo(obj, population_size=75, method="nsga3", divisions=4, random_state=9)
    assert len(sel) == 75
    assert _selected_not_dominated_by_others(obj, sel)


def test_moconfig_wiring():
    cfg = MOConfig(selection_method="nsga-iii")
    assert cfg.selection_method == "nsga3"
    obj = np.random.default_rng(12).random((160, 3))
    sel = cfg.select(obj, population_size=80, divisions=4, random_state=13)
    assert len(sel) == 80
    assert _selected_not_dominated_by_others(obj, sel)
