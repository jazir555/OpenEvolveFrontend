"""Novelty Search: a Quality-Diversity (QD) algorithm.

Implements the behavior-characteristic archive variant of Novelty Search
(Lehman & Stanley, 2011) for OpenEvolve. An individual's novelty is the
average distance from its *behavior descriptor* (a fixed-length vector) to
its k nearest neighbors in the union of the archive and the current
population. Individuals are admitted to the archive only when they exceed a
configured ``novelty_threshold``.

Pure-numpy, dependency free beyond numpy. The selection routine follows the
same objective-matrix / selection contract as NSGA-II/NSGA-III in
``openevolve.selection``: it takes a 2D array (rows = individuals, columns =
behavior features) and returns the indices selected for the next generation.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Union

import numpy as np


# --------------------------------------------------------------------------- #
# Distance metrics
# --------------------------------------------------------------------------- #
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean (L2) distance between two behavior vectors."""
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan (L1) distance between two behavior vectors."""
    return float(np.sum(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (1 - cosine similarity) between two behavior vectors.

    Zero vectors are treated as identical (distance 0).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        # Both zero -> identical; otherwise one-sided zero is maximally distant.
        if norm_a == 0.0 and norm_b == 0.0:
            return 0.0
        return 1.0
    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    return float(1.0 - similarity)


_DISTANCE_METRICS = {
    "euclidean": euclidean_distance,
    "l2": euclidean_distance,
    "manhattan": manhattan_distance,
    "l1": manhattan_distance,
    "cosine": cosine_distance,
}


def get_distance_metric(name: str):
    """Resolve a distance-metric callable by name (case-insensitive)."""
    key = name.strip().lower() if name else "euclidean"
    if key not in _DISTANCE_METRICS:
        raise ValueError(
            f"Unknown distance_metric {name!r}. "
            f"Supported: {sorted(_DISTANCE_METRICS)}"
        )
    return _DISTANCE_METRICS[key]


# --------------------------------------------------------------------------- #
# Core novelty computation
# --------------------------------------------------------------------------- #
def compute_novelty(
    behavior: Sequence[float],
    reference_behaviors: Sequence[Sequence[float]],
    k: int = 10,
    distance_metric: str = "euclidean",
) -> float:
    """Average distance from ``behavior`` to its ``k`` nearest references.

    Args:
        behavior: the individual's behavior descriptor (1D vector).
        reference_behaviors: archive/population behavior vectors.
        k: number of nearest neighbors to average over.
        distance_metric: name of the distance metric to use.

    Returns:
        ``float('inf')`` if there are no reference behaviors, otherwise the
        mean distance to the ``k`` closest references.
    """
    refs = [np.asarray(r, dtype=float) for r in reference_behaviors]
    if not refs:
        return float("inf")

    dist_fn = get_distance_metric(distance_metric)
    b = np.asarray(behavior, dtype=float)

    distances = np.array([dist_fn(b, r) for r in refs], dtype=float)
    k_eff = min(k, len(distances))
    nearest = np.partition(distances, k_eff - 1)[:k_eff]
    return float(np.mean(nearest))


# --------------------------------------------------------------------------- #
# Behavior-characteristic archive
# --------------------------------------------------------------------------- #
class BehaviorArchive:
    """Stores behavior vectors of previously evaluated solutions."""

    def __init__(
        self,
        k_neighbors: int = 10,
        novelty_threshold: float = 0.5,
        archive_size_limit: Optional[int] = None,
        distance_metric: str = "euclidean",
    ) -> None:
        self.k_neighbors = k_neighbors
        self.novelty_threshold = novelty_threshold
        self.archive_size_limit = archive_size_limit
        self.distance_metric = distance_metric
        self.archive: List[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.archive)

    def novelty(self, behavior: Sequence[float], k: Optional[int] = None) -> float:
        """Novelty of ``behavior`` relative to the current archive only."""
        k_eff = k if k is not None else self.k_neighbors
        return compute_novelty(behavior, self.archive, k=k_eff,
                               distance_metric=self.distance_metric)

    def maybe_add(
        self, behavior: Sequence[float], k: Optional[int] = None
    ) -> bool:
        """Admit ``behavior`` to the archive if it is novel enough.

        Returns ``True`` if the behavior was added. When the archive exceeds
        ``archive_size_limit`` the oldest entries are evicted.
        """
        k_eff = k if k is not None else self.k_neighbors
        novelty = compute_novelty(behavior, self.archive, k=k_eff,
                                  distance_metric=self.distance_metric)
        if novelty < self.novelty_threshold:
            return False
        self.archive.append(np.asarray(behavior, dtype=float))
        if (
            self.archive_size_limit is not None
            and len(self.archive) > self.archive_size_limit
        ):
            excess = len(self.archive) - self.archive_size_limit
            self.archive = self.archive[excess:]
        return True

    def add(self, behavior: Sequence[float]) -> None:
        """Unconditionally add a behavior vector to the archive."""
        self.archive.append(np.asarray(behavior, dtype=float))
        if (
            self.archive_size_limit is not None
            and len(self.archive) > self.archive_size_limit
        ):
            excess = len(self.archive) - self.archive_size_limit
            self.archive = self.archive[excess:]


# --------------------------------------------------------------------------- #
# Novelty Search variant
# --------------------------------------------------------------------------- #
class NoveltySearch:
    """Standalone Novelty Search QD variant.

    Manages a behavior archive and provides selection that prioritizes
    behavioral novelty. Novelty of an individual is measured against the
    *union* of the archive and the current population (so early, sparse
    generations are compared against their peers), then the most novel
    individuals are kept for the next generation and admitted to the archive
    subject to ``novelty_threshold``.
    """

    def __init__(
        self,
        k_neighbors: int = 10,
        novelty_threshold: float = 0.5,
        archive_size_limit: Optional[int] = None,
        distance_metric: str = "euclidean",
    ) -> None:
        self.k_neighbors = k_neighbors
        self.novelty_threshold = novelty_threshold
        self.archive_size_limit = archive_size_limit
        self.distance_metric = distance_metric
        self.archive: List[np.ndarray] = []

    def _as_matrix(self, behaviors: Sequence[Sequence[float]]) -> List[np.ndarray]:
        return [np.asarray(b, dtype=float) for b in behaviors]

    def calculate_novelty(
        self,
        behavior: Sequence[float],
        population_behaviors: Optional[Sequence[Sequence[float]]] = None,
        k: Optional[int] = None,
    ) -> float:
        """Novelty vs archive + (optionally) the current population.

        A duplicate of an archived behavior yields near-zero novelty.
        """
        k_eff = k if k is not None else self.k_neighbors
        references: List[np.ndarray] = list(self.archive)
        if population_behaviors is not None:
            references.extend(self._as_matrix(population_behaviors))
        return compute_novelty(behavior, references, k=k_eff,
                               distance_metric=self.distance_metric)

    def update_archive(self, behavior: Sequence[float]) -> bool:
        """Add ``behavior`` to the archive if it passes the novelty gate."""
        novelty = self.calculate_novelty(behavior)
        if novelty < self.novelty_threshold:
            return False
        self.archive.append(np.asarray(behavior, dtype=float))
        if (
            self.archive_size_limit is not None
            and len(self.archive) > self.archive_size_limit
        ):
            excess = len(self.archive) - self.archive_size_limit
            self.archive = self.archive[excess:]
        return True

    def select(
        self,
        behaviors: Sequence[Sequence[float]],
        population_size: int,
        random_state: Optional[Union[int, random.Random]] = None,
    ) -> List[int]:
        """Select the most novel individuals and update the archive.

        Args:
            behaviors: 2D array of behavior vectors (rows = individuals).
            population_size: number of individuals to keep.
            random_state: optional seed / Random for tie-breaking.

        Returns:
            Indices (into ``behaviors``) selected for the next generation.
        """
        rng = (
            random.Random(random_state)
            if isinstance(random_state, int)
            else (random_state or random.Random())
        )
        behaviors = self._as_matrix(behaviors)
        n = len(behaviors)
        if n == 0:
            return []

        if population_size >= n:
            # Everyone is kept; still populate the archive with novel entries.
            for i in range(n):
                self.update_archive(behaviors[i])
            return list(range(n))

        # Novelty of each individual vs the whole population + current archive.
        novelties = np.array(
            [self.calculate_novelty(behaviors[i], behaviors) for i in range(n)],
            dtype=float,
        )

        # Greedy farthest-point style selection: pick the most novel, add it
        # to the archive as we go so subsequent references include it.
        order = sorted(range(n), key=lambda i: (-novelties[i], rng.random()))
        selected: List[int] = []
        for i in order:
            if len(selected) >= population_size:
                break
            self.update_archive(behaviors[i])
            selected.append(i)
        return selected


# --------------------------------------------------------------------------- #
# Selection-contract wrapper (mirrors nsga3_selection / select_mo)
# --------------------------------------------------------------------------- #
def novelty_selection(
    behaviors: np.ndarray,
    population_size: int,
    k: int = 10,
    novelty_threshold: float = 0.5,
    archive_size_limit: Optional[int] = None,
    distance_metric: str = "euclidean",
    random_state: Optional[int] = None,
) -> List[int]:
    """Convenience wrapper around :class:`NoveltySearch`.

    Mirrors the NSGA-II/III selection contract: a 2D matrix of behavior
    descriptors in and a list of selected indices out.
    """
    searcher = NoveltySearch(
        k_neighbors=k,
        novelty_threshold=novelty_threshold,
        archive_size_limit=archive_size_limit,
        distance_metric=distance_metric,
    )
    return searcher.select(behaviors, population_size, random_state=random_state)
