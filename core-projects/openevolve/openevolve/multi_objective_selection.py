"""Generic NSGA-style multi-objective selection helpers.

Implements :class:`MultiObjectiveSelection` from
``docs/architecture/EVOLUTION_ALGORITHM_ENHANCEMENT_SPEC.md`` (section
"Multi-Objective Evolution / Multi-Objective Selection") as a reusable,
algorithm-agnostic utility: fast non-dominated sorting, crowding-distance
assignment, the crowded-comparison operator, binary-tournament mating
selection and NSGA-II environmental (truncation) selection.

Unlike :mod:`openevolve.nsga3` -- which is specialised for reference-point
based many-objective selection -- this module operates on plain objective
matrices and supports per-objective *senses* (minimise / maximise), so it can
be reused by the enhanced evolution engine, by MAP-Elites variants that carry
auxiliary objectives, and by any caller that just needs Pareto ranking.

Pure-numpy, dependency free beyond numpy. The selection routines follow the
same contract as ``openevolve.selection.select_mo``: a 2D array in (rows =
individuals, columns = objectives) and a list of selected indices out.
Objectives are minimised by default; pass ``senses`` to mix directions.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from openevolve.nsga3 import dominates_min, fast_non_dominated_sort

SenseLike = Optional[Union[str, Sequence[Union[str, int, bool, float]]]]

_MAXIMIZE_TOKENS = {"max", "maximize", "maximise", "maximum", "up", "+", "high"}
_MINIMIZE_TOKENS = {"min", "minimize", "minimise", "minimum", "down", "-", "low"}


# --------------------------------------------------------------------------- #
# Objective-matrix helpers
# --------------------------------------------------------------------------- #
def as_objective_matrix(objectives: object) -> np.ndarray:
    """Coerce ``objectives`` into a 2D float matrix (rows = individuals)."""
    matrix = np.asarray(objectives, dtype=float)
    if matrix.size == 0:
        return matrix.reshape(0, 0) if matrix.ndim < 2 else matrix
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    return matrix


def normalize_senses(num_objectives: int, senses: SenseLike = None) -> np.ndarray:
    """Return a sign vector: ``+1`` for minimise, ``-1`` for maximise.

    ``senses`` accepts ``None`` (all minimise), a single token applied to every
    objective (``"min"`` / ``"max"``), or a per-objective sequence of tokens,
    booleans (``True`` = maximise) or numbers (negative = maximise).
    """
    if senses is None:
        return np.ones(num_objectives, dtype=float)

    if isinstance(senses, str):
        tokens: Sequence[Union[str, int, bool, float]] = [senses] * num_objectives
    else:
        tokens = list(senses)
        if len(tokens) == 1 and num_objectives > 1:
            tokens = list(tokens) * num_objectives
        if len(tokens) != num_objectives:
            raise ValueError(
                f"senses has {len(tokens)} entries but there are "
                f"{num_objectives} objectives"
            )

    signs = np.ones(num_objectives, dtype=float)
    for i, token in enumerate(tokens):
        if isinstance(token, str):
            key = token.strip().lower()
            if key in _MAXIMIZE_TOKENS:
                signs[i] = -1.0
            elif key in _MINIMIZE_TOKENS:
                signs[i] = 1.0
            else:
                raise ValueError(
                    f"Unknown objective sense {token!r}. "
                    f"Use one of {sorted(_MINIMIZE_TOKENS | _MAXIMIZE_TOKENS)}."
                )
        elif isinstance(token, bool):
            signs[i] = -1.0 if token else 1.0
        else:
            signs[i] = -1.0 if float(token) < 0 else 1.0
    return signs


def apply_senses(objectives: object, senses: SenseLike = None) -> np.ndarray:
    """Return a minimisation-oriented copy of ``objectives``."""
    matrix = as_objective_matrix(objectives)
    if matrix.size == 0:
        return matrix
    signs = normalize_senses(matrix.shape[1], senses)
    return matrix * signs


def dominates(
    a: Sequence[float],
    b: Sequence[float],
    senses: SenseLike = None,
) -> bool:
    """True if objective vector ``a`` Pareto-dominates ``b``."""
    vec_a = np.asarray(a, dtype=float).reshape(-1)
    vec_b = np.asarray(b, dtype=float).reshape(-1)
    if vec_a.shape != vec_b.shape:
        raise ValueError("dominates() requires objective vectors of equal length")
    signs = normalize_senses(vec_a.shape[0], senses)
    return dominates_min(vec_a * signs, vec_b * signs)


def crowding_distance(
    objectives: object,
    front: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """NSGA-II crowding distance for the rows of ``front``.

    Returns an array aligned with ``front`` (position ``i`` holds the distance
    of ``front[i]``). Boundary solutions in each objective receive
    ``float('inf')``; fronts of size <= 2 are all-infinite. When ``front`` is
    omitted every row of ``objectives`` is used.
    """
    matrix = as_objective_matrix(objectives)
    if matrix.size == 0:
        return np.zeros(0, dtype=float)
    indices = list(range(matrix.shape[0])) if front is None else list(front)
    dist = np.zeros(len(indices), dtype=float)
    if len(indices) == 0:
        return dist
    if len(indices) <= 2:
        dist[:] = float("inf")
        return dist

    sub = matrix[indices]
    for obj in range(matrix.shape[1]):
        order = np.argsort(sub[:, obj], kind="stable")
        dist[order[0]] = float("inf")
        dist[order[-1]] = float("inf")
        obj_range = sub[order[-1], obj] - sub[order[0], obj]
        if obj_range == 0:
            continue
        for i in range(1, len(indices) - 1):
            if np.isinf(dist[order[i]]):
                continue
            dist[order[i]] += (
                sub[order[i + 1], obj] - sub[order[i - 1], obj]
            ) / obj_range
    return dist


def pareto_front(objectives: object, senses: SenseLike = None) -> List[int]:
    """Indices of the non-dominated (first) front."""
    signed = apply_senses(objectives, senses)
    if signed.size == 0:
        return []
    fronts = fast_non_dominated_sort(signed)
    return list(fronts[0]) if fronts else []


# --------------------------------------------------------------------------- #
# Multi-objective selection
# --------------------------------------------------------------------------- #
class MultiObjectiveSelection:
    """NSGA-style Pareto selection over objective matrices.

    Args:
        senses: per-objective direction (see :func:`normalize_senses`). The
            default minimises every objective, matching ``select_mo``.
        tournament_size: candidates per mating tournament in
            :meth:`select_parents`.
        random_state: seed or :class:`random.Random` for tie-breaking.

    The class is stateless with respect to the population: every method takes
    the objective matrix explicitly, so the same instance can be reused across
    generations (only the RNG carries state).
    """

    def __init__(
        self,
        senses: SenseLike = None,
        tournament_size: int = 2,
        random_state: Optional[Union[int, random.Random]] = None,
    ) -> None:
        self.senses = senses
        self.tournament_size = max(2, int(tournament_size))
        self.rng = (
            random_state
            if isinstance(random_state, random.Random)
            else random.Random(random_state)
        )

    # -- core ranking ----------------------------------------------------- #
    def signed_objectives(self, objectives: object) -> np.ndarray:
        """Minimisation-oriented view of ``objectives`` honouring ``senses``."""
        return apply_senses(objectives, self.senses)

    def fast_nondominated_sort(self, objectives: object) -> List[List[int]]:
        """Fast non-dominated sort. Returns fronts as lists of row indices."""
        signed = self.signed_objectives(objectives)
        if signed.size == 0:
            return []
        return fast_non_dominated_sort(signed)

    def dominates(self, a: Sequence[float], b: Sequence[float]) -> bool:
        """Domination test using this selector's ``senses``."""
        return dominates(a, b, self.senses)

    def pareto_front(self, objectives: object) -> List[int]:
        """Indices of the first (non-dominated) front."""
        fronts = self.fast_nondominated_sort(objectives)
        return list(fronts[0]) if fronts else []

    def assign_crowding_distance(
        self, objectives: object, front: Optional[Sequence[int]] = None
    ) -> np.ndarray:
        """Crowding distances for ``front`` (aligned with ``front`` order)."""
        return crowding_distance(self.signed_objectives(objectives), front)

    def rank_and_crowding(self, objectives: object) -> Tuple[np.ndarray, np.ndarray]:
        """Per-individual (front rank, crowding distance).

        Rank 0 is the Pareto front. Both arrays are indexed by row.
        """
        signed = self.signed_objectives(objectives)
        n = signed.shape[0] if signed.size else 0
        ranks = np.zeros(n, dtype=int)
        distances = np.zeros(n, dtype=float)
        if n == 0:
            return ranks, distances
        for rank, front in enumerate(fast_non_dominated_sort(signed)):
            dist = crowding_distance(signed, front)
            for pos, idx in enumerate(front):
                ranks[idx] = rank
                distances[idx] = dist[pos]
        return ranks, distances

    def crowded_compare(
        self,
        i: int,
        j: int,
        ranks: np.ndarray,
        distances: np.ndarray,
    ) -> int:
        """NSGA-II crowded-comparison operator.

        Returns ``-1`` when ``i`` is preferred, ``1`` when ``j`` is preferred
        and ``0`` when neither dominates on (rank, crowding).
        """
        if ranks[i] < ranks[j]:
            return -1
        if ranks[i] > ranks[j]:
            return 1
        if distances[i] > distances[j]:
            return -1
        if distances[i] < distances[j]:
            return 1
        return 0

    def sort_by_rank(self, objectives: object) -> List[int]:
        """All indices ordered best-first by (rank asc, crowding desc)."""
        ranks, distances = self.rank_and_crowding(objectives)
        n = len(ranks)
        return sorted(
            range(n),
            key=lambda i: (
                int(ranks[i]),
                -(distances[i] if np.isfinite(distances[i]) else float("inf")),
                self.rng.random(),
            ),
        )

    # -- selection -------------------------------------------------------- #
    def select_parents(
        self,
        objectives: object,
        selection_size: int,
        with_replacement: bool = True,
    ) -> List[int]:
        """Mating selection via binary (k-ary) crowded tournaments.

        Args:
            objectives: 2D objective matrix.
            selection_size: number of parent indices to return.
            with_replacement: when False the same individual is never returned
                twice (the pool is capped at the population size).

        Returns:
            Selected row indices, in the order they were drawn.
        """
        signed = self.signed_objectives(objectives)
        n = signed.shape[0] if signed.size else 0
        if n == 0 or selection_size <= 0:
            return []

        ranks, distances = self.rank_and_crowding(signed)
        if not with_replacement:
            selection_size = min(selection_size, n)

        available = list(range(n))
        selected: List[int] = []
        while len(selected) < selection_size and available:
            k = min(self.tournament_size, len(available))
            candidates = self.rng.sample(available, k)
            winner = candidates[0]
            for challenger in candidates[1:]:
                if self.crowded_compare(challenger, winner, ranks, distances) < 0:
                    winner = challenger
            selected.append(winner)
            if not with_replacement:
                available.remove(winner)
        return selected

    def environmental_selection(
        self,
        objectives: object,
        population_size: Optional[int] = None,
    ) -> List[int]:
        """NSGA-II survivor selection: fill fronts, truncate by crowding.

        Args:
            objectives: 2D objective matrix of the combined population.
            population_size: survivors to keep (default: everyone).

        Returns:
            Indices kept for the next generation.
        """
        signed = self.signed_objectives(objectives)
        if signed.size == 0:
            return []
        n = signed.shape[0]
        pop_size = n if population_size is None else int(population_size)
        if pop_size >= n:
            return list(range(n))
        if pop_size <= 0:
            return []

        fronts = fast_non_dominated_sort(signed)
        selected: List[int] = []
        last_front: Optional[List[int]] = None
        for front in fronts:
            if len(selected) + len(front) <= pop_size:
                selected.extend(front)
            else:
                last_front = list(front)
                break

        if last_front is None:
            return selected[:pop_size]

        remaining = pop_size - len(selected)
        if remaining <= 0:
            return selected[:pop_size]

        dist = crowding_distance(signed, last_front)
        order = sorted(
            range(len(last_front)),
            key=lambda p: (
                -(dist[p] if np.isfinite(dist[p]) else float("inf")),
                self.rng.random(),
            ),
        )
        selected.extend(last_front[p] for p in order[:remaining])
        return selected[:pop_size]

    # -- contract alias --------------------------------------------------- #
    def select(self, objectives: object, population_size: int) -> List[int]:
        """Alias of :meth:`environmental_selection` (``select_mo`` contract)."""
        return self.environmental_selection(objectives, population_size)


# --------------------------------------------------------------------------- #
# Selection-contract wrapper (mirrors nsga3_selection / novelty_selection)
# --------------------------------------------------------------------------- #
def multi_objective_selection(
    objectives: np.ndarray,
    population_size: int,
    senses: SenseLike = None,
    random_state: Optional[int] = None,
) -> List[int]:
    """Convenience wrapper around :class:`MultiObjectiveSelection`.

    Mirrors the NSGA-II/III selection contract: a 2D objective matrix in
    (rows = individuals, columns = objectives, minimisation unless ``senses``
    says otherwise) and the surviving indices out.
    """
    selector = MultiObjectiveSelection(senses=senses, random_state=random_state)
    return selector.environmental_selection(objectives, population_size)


def nondominated_ranks(
    objectives: np.ndarray,
    senses: SenseLike = None,
) -> np.ndarray:
    """Per-individual Pareto front index (0 = non-dominated front)."""
    selector = MultiObjectiveSelection(senses=senses)
    ranks, _ = selector.rank_and_crowding(objectives)
    return ranks


def pareto_archive_update(
    archive: List[Tuple[int, Sequence[float]]],
    candidates: Iterable[Tuple[int, Sequence[float]]],
    senses: SenseLike = None,
    max_size: Optional[int] = None,
) -> List[Tuple[int, Sequence[float]]]:
    """Insert ``candidates`` into a non-dominated ``archive``.

    Each entry is an ``(identifier, objective_vector)`` pair. Dominated and
    duplicate entries are dropped. When ``max_size`` is given the archive is
    thinned by crowding distance (most crowded entries removed first).
    """
    entries: List[Tuple[int, Sequence[float]]] = list(archive)
    for candidate in candidates:
        cand_id, cand_obj = candidate
        cand_vec = np.asarray(cand_obj, dtype=float).reshape(-1)
        dominated_by_archive = False
        kept: List[Tuple[int, Sequence[float]]] = []
        for entry_id, entry_obj in entries:
            entry_vec = np.asarray(entry_obj, dtype=float).reshape(-1)
            if entry_id == cand_id or np.array_equal(entry_vec, cand_vec):
                dominated_by_archive = True
                kept.append((entry_id, entry_obj))
                continue
            if dominates(entry_vec, cand_vec, senses):
                dominated_by_archive = True
                kept.append((entry_id, entry_obj))
            elif not dominates(cand_vec, entry_vec, senses):
                kept.append((entry_id, entry_obj))
        entries = kept
        if not dominated_by_archive:
            entries.append((cand_id, cand_vec))

    if max_size is not None and len(entries) > max_size:
        matrix = np.array(
            [np.asarray(obj, dtype=float).reshape(-1) for _, obj in entries],
            dtype=float,
        )
        dist = crowding_distance(apply_senses(matrix, senses))
        order = sorted(
            range(len(entries)),
            key=lambda i: -(dist[i] if np.isfinite(dist[i]) else float("inf")),
        )
        entries = [entries[i] for i in order[:max_size]]
    return entries
