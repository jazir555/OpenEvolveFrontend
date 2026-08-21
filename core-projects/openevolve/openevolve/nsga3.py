"""NSGA-III: Non-dominated Sorting Genetic Algorithm III.

Reference-point based many-objective evolutionary selection (Deb & Jain, 2014).
Pure-numpy, dependency free beyond numpy. Operates on objective vectors
(minimization assumed) regardless of the surrounding evolution loop, so it can
be slotted into OpenEvolve's multi-objective selection path.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


def _lattice_points(num_objectives: int, p: int) -> np.ndarray:
    points: List[List[int]] = []

    def recurse(current: List[int], remaining: int, dims: int) -> None:
        if dims == 1:
            current.append(remaining)
            points.append(list(current))
            current.pop()
            return
        for i in range(remaining + 1):
            current.append(i)
            recurse(current, remaining - i, dims - 1)
            current.pop()

    recurse([], p, num_objectives)
    return np.asarray(points, dtype=float) / float(p)


def generate_reference_points(
    num_objectives: int,
    divisions: Union[int, Sequence[int]],
    divisions_inner: int = 0,
) -> np.ndarray:
    """Das & Dennis structured reference points on the simplex.

    `divisions` may be a single integer (one layer) or a sequence of integers
    (multiple layers: outer, then inner subdivisions). When `divisions_inner`
    is given alongside a single outer `divisions`, a two-layer set is produced.
    """
    if isinstance(divisions, (list, tuple)):
        layers = [int(d) for d in divisions]
    else:
        if divisions_inner and divisions_inner > 0:
            layers = [int(divisions), int(divisions_inner)]
        else:
            layers = [int(divisions)]

    ref_sets = [_lattice_points(num_objectives, p) for p in layers]
    reference = np.vstack(ref_sets)
    _, unique_idx = np.unique(np.round(reference, 12), axis=0, return_index=True)
    return reference[np.sort(unique_idx)]


def dominates_min(a: np.ndarray, b: np.ndarray) -> bool:
    """True if `a` dominates `b` under minimization."""
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """Fast non-dominated sort. Returns fronts as lists of indices."""
    objectives = np.asarray(objectives, dtype=float)
    n = objectives.shape[0]
    dominated: List[List[int]] = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates_min(objectives[p], objectives[q]):
                dominated[p].append(q)
            elif dominates_min(objectives[q], objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt: List[int] = []
        for p in fronts[i]:
            for q in dominated[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)

    if not fronts[-1]:
        fronts.pop()
    return fronts


def _extreme_intercepts(
    objectives: np.ndarray, ideal: np.ndarray
) -> np.ndarray:
    """Intercepts along each axis via the achievement scalarizing function."""
    n_obj = objectives.shape[1]
    translated = objectives - ideal
    intercepts = np.amin(translated, axis=0)
    for j in range(n_obj):
        w = np.full(n_obj, 1e-6)
        w[j] = 1.0
        asf = np.max(translated / w, axis=1)
        best = int(np.argmin(asf))
        intercepts[j] = translated[best, j]
    max_obj = np.amax(translated, axis=0)
    intercepts[intercepts < 1e-6] = max_obj[intercepts < 1e-6]
    return intercepts


def _associate(
    objectives: np.ndarray,
    reference_points: np.ndarray,
    ideal: np.ndarray,
    intercepts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    translated = objectives - ideal
    denom = intercepts - ideal
    denom[denom == 0] = 1.0
    normalized = translated / denom

    membership = np.zeros(objectives.shape[0], dtype=int)
    distances = np.zeros(objectives.shape[0], dtype=float)
    for i in range(objectives.shape[0]):
        d = normalized[i]
        diffs = reference_points - d
        proj = (np.sum(diffs * reference_points, axis=1) /
                np.sum(reference_points * reference_points, axis=1))
        perp = np.linalg.norm(diffs - proj[:, None] * reference_points, axis=1)
        k = int(np.argmin(perp))
        membership[i] = k
        distances[i] = perp[k]
    return membership, distances


class NSGAIII:
    """NSGA-III environmental selection over objective vectors."""

    def __init__(
        self,
        num_objectives: int,
        divisions: Union[int, Sequence[int]] = 4,
        divisions_inner: int = 0,
        population_size: int = 100,
    ) -> None:
        self.num_objectives = num_objectives
        self.divisions = divisions
        self.divisions_inner = divisions_inner
        self.population_size = population_size
        self.reference_points = generate_reference_points(
            num_objectives, divisions, divisions_inner
        )

    def environmental_selection(
        self,
        combined_objectives: np.ndarray,
        population_size: Optional[int] = None,
        random_state: Optional[random.Random] = None,
    ) -> List[int]:
        combined = np.asarray(combined_objectives, dtype=float)
        pop_size = population_size or self.population_size
        if isinstance(random_state, int):
            rng = random.Random(random_state)
        else:
            rng = random_state or random.Random()
        num_ref = self.reference_points.shape[0]

        fronts = fast_non_dominated_sort(combined)
        selected: List[int] = []
        last_front: Optional[List[int]] = None
        for front in fronts:
            if len(selected) + len(front) <= pop_size:
                selected.extend(front)
            else:
                last_front = front
                break

        if last_front is None:
            return selected[:pop_size]

        remaining = pop_size - len(selected)
        if remaining <= 0:
            return selected[:pop_size]

        ideal = np.min(combined, axis=0)
        intercepts = _extreme_intercepts(combined, ideal)
        last_obj = combined[last_front]
        last_membership, last_dist = _associate(
            last_obj, self.reference_points, ideal, intercepts
        )

        niche_count = np.zeros(num_ref, dtype=int)
        if selected:
            sel_obj = combined[selected]
            sel_membership, _ = _associate(
                sel_obj, self.reference_points, ideal, intercepts
            )
            for k in sel_membership:
                niche_count[k] += 1

        pool_membership = list(last_membership)
        pool_dist = list(last_dist)
        pool_indices = list(last_front)
        chosen: List[int] = []

        while remaining > 0 and pool_membership:
            assoc_counts = np.zeros(num_ref, dtype=int)
            for k in pool_membership:
                assoc_counts[k] += 1

            min_niche = int(np.min(niche_count + (assoc_counts == 0) * (num_ref + 1)))
            candidates = [
                k for k in range(num_ref)
                if niche_count[k] == min_niche and assoc_counts[k] > 0
            ]
            if not candidates:
                candidates = [k for k in range(num_ref) if assoc_counts[k] > 0]
            if not candidates:
                break
            j = rng.choice(candidates)

            positions = [idx for idx, k in enumerate(pool_membership) if k == j]
            if niche_count[j] == 0:
                best_local = min(positions, key=lambda p: pool_dist[p])
            else:
                best_local = min(positions, key=lambda p: pool_dist[p])

            pick = positions[positions.index(best_local)]
            chosen.append(pool_indices[pick])
            niche_count[j] += 1
            del pool_membership[pick]
            del pool_dist[pick]
            del pool_indices[pick]
            remaining -= 1

        selected.extend(chosen)
        return selected[:pop_size]


def nsga3_selection(
    objectives: np.ndarray,
    population_size: int,
    divisions: Union[int, Sequence[int]] = 4,
    divisions_inner: int = 0,
    random_state: Optional[int] = None,
) -> List[int]:
    """Convenience wrapper mirroring the selection-loop contract.

    Takes a 2D array of objective vectors (rows = individuals, columns =
    objectives, minimization assumed) and returns the selected individual
    indices for the next generation.
    """
    objectives = np.asarray(objectives, dtype=float)
    num_objectives = objectives.shape[1]
    selector = NSGAIII(
        num_objectives=num_objectives,
        divisions=divisions,
        divisions_inner=divisions_inner,
        population_size=population_size,
    )
    rng = random.Random(random_state) if random_state is not None else None
    return selector.environmental_selection(
        objectives, population_size=population_size, random_state=rng
    )
