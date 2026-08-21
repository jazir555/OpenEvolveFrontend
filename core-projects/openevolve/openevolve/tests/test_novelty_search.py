"""Offline unit tests for Novelty Search (no LLM required)."""

import numpy as np

from openevolve.novelty_search import (
    BehaviorArchive,
    NoveltySearch,
    compute_novelty,
    novelty_selection,
)
from openevolve.selection import select_mo
from openevolve.unified.config import MOConfig, NoveltySearchConfig


def _random_behaviors(n, dim, seed):
    rng = np.random.default_rng(seed)
    return rng.random((n, dim))


def test_compute_novelty_inf_with_empty_archive():
    b = [0.1, 0.2, 0.3]
    assert compute_novelty(b, [], k=5) == float("inf")


def test_compute_novelty_near_duplicate_low():
    archive = [[0.5, 0.5], [0.55, 0.45], [0.49, 0.52]]
    dup = [0.5, 0.5]
    far = [1.0, 0.0]
    n_dup = compute_novelty(dup, archive, k=3)
    n_far = compute_novelty(far, archive, k=3)
    assert n_dup < n_far


def test_compute_novelty_distance_metrics():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(compute_novelty(a, [b], k=1, distance_metric="euclidean") - np.sqrt(2)) < 1e-9
    assert abs(compute_novelty(a, [b], k=1, distance_metric="manhattan") - 2.0) < 1e-9
    assert abs(compute_novelty(a, [b], k=1, distance_metric="cosine") - 1.0) < 1e-9


def test_archive_grows_on_admission():
    arch = BehaviorArchive(k_neighbors=3, novelty_threshold=0.1, archive_size_limit=None)
    assert len(arch) == 0
    # Dispersed points are all novel relative to an empty/small archive.
    points = _random_behaviors(20, 3, seed=1)
    added = 0
    for p in points:
        if arch.maybe_add(p):
            added += 1
    assert len(arch) == added
    assert added > 1  # archive grows


def test_threshold_gating_blocks_duplicates():
    arch = BehaviorArchive(k_neighbors=3, novelty_threshold=0.5, archive_size_limit=None)
    seed_point = [0.5, 0.5, 0.5]
    assert arch.maybe_add(seed_point) is True
    # Near-duplicates fall below the threshold and are rejected.
    rejected = 0
    for p in [[0.51, 0.5, 0.5], [0.5, 0.49, 0.5], [0.5, 0.5, 0.51]]:
        if not arch.maybe_add(p):
            rejected += 1
    assert rejected == 3
    assert len(arch) == 1


def test_archive_size_limit_enforced():
    arch = BehaviorArchive(k_neighbors=2, novelty_threshold=0.0, archive_size_limit=5)
    for p in _random_behaviors(15, 2, seed=2):
        arch.maybe_add(p)  # threshold 0 admits everything
    assert len(arch) == 5


def test_novelty_decreases_for_near_duplicates():
    ns = NoveltySearch(k_neighbors=5, novelty_threshold=0.0)
    base = [0.5, 0.5, 0.5, 0.5]
    ns.archive.append(np.array(base, dtype=float))
    n_close = ns.calculate_novelty([0.5, 0.5, 0.5, 0.51])
    n_far = ns.calculate_novelty([0.0, 1.0, 0.0, 1.0])
    assert n_close < n_far


def test_update_archive_threshold_gate():
    ns = NoveltySearch(k_neighbors=3, novelty_threshold=0.6)
    assert ns.update_archive([0.5, 0.5]) is True
    # A near-duplicate is below threshold -> not added.
    assert ns.update_archive([0.5, 0.5]) is False
    assert len(ns.archive) == 1


def test_select_returns_valid_indices():
    ns = NoveltySearch(k_neighbors=3, novelty_threshold=0.0)
    behaviors = _random_behaviors(50, 4, seed=3)
    sel = ns.select(behaviors, population_size=10, random_state=0)
    assert len(sel) == 10
    assert set(sel).issubset(set(range(50)))
    assert len(set(sel)) == 10  # unique indices


def test_select_keeps_all_when_population_small():
    ns = NoveltySearch(k_neighbors=2, novelty_threshold=0.0)
    behaviors = _random_behaviors(5, 3, seed=4)
    sel = ns.select(behaviors, population_size=100, random_state=0)
    assert sel == [0, 1, 2, 3, 4]


def test_novelty_selection_wrapper():
    behaviors = _random_behaviors(60, 3, seed=5)
    sel = novelty_selection(
        behaviors,
        population_size=15,
        k=4,
        novelty_threshold=0.0,
        distance_metric="euclidean",
        random_state=0,
    )
    assert len(sel) == 15
    assert set(sel).issubset(set(range(60)))


def test_dispatch_routes_novelty_search():
    behaviors = _random_behaviors(80, 3, seed=6)
    sel = select_mo(behaviors, population_size=20, method="novelty_search", random_state=0)
    assert len(sel) == 20
    assert set(sel).issubset(set(range(80)))


def test_moconfig_novelty_wiring():
    cfg = MOConfig(selection_method="novelty_search")
    assert cfg.selection_method == "novelty_search"
    behaviors = _random_behaviors(90, 3, seed=7)
    sel = cfg.select(behaviors, population_size=25, random_state=0)
    assert len(sel) == 25
    assert set(sel).issubset(set(range(90)))


def test_novelty_search_config_defaults():
    cfg = NoveltySearchConfig(enabled=True)
    assert cfg.k_neighbors == 10
    assert cfg.novelty_threshold == 0.5
    assert cfg.distance_metric == "euclidean"


def test_novelty_search_config_distance_validation():
    import pytest

    with pytest.raises(ValueError):
        NoveltySearchConfig(distance_metric="hamming")
