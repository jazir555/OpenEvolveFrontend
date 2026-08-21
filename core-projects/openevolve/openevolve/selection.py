"""Multi-objective selection dispatch for OpenEvolve.

Provides `select_mo`, which routes the configured `selection_method` to the
appropriate environmental-selection routine. NSGA-II is implemented here
(fast non-dominated sort + crowding distance) and NSGA-III is delegated to
`openevolve.nsga3`. Objective vectors are minimization; callers should negate
or invert maximize objectives before passing them in.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from openevolve.neat import neat_selection
from openevolve.nsga3 import fast_non_dominated_sort, nsga3_selection
from openevolve.novelty_search import novelty_selection
from openevolve.symbolic_regression import SymbolicRegressionResult, evolve as _sr_evolve


def _crowding_distance(objectives: np.ndarray, front: List[int]) -> np.ndarray:
    m = objectives.shape[1]
    dist = np.zeros(len(front), dtype=float)
    if len(front) <= 2:
        dist[:] = float("inf")
        return dist
    sub = objectives[front]
    for obj in range(m):
        order = np.argsort(sub[:, obj])
        dist[order[0]] = float("inf")
        dist[order[-1]] = float("inf")
        obj_range = sub[order[-1], obj] - sub[order[0], obj]
        if obj_range == 0:
            continue
        for i in range(1, len(front) - 1):
            dist[order[i]] += (
                sub[order[i + 1], obj] - sub[order[i - 1], obj]
            ) / obj_range
    return dist


def nsga2_selection(
    objectives: np.ndarray,
    population_size: int,
    random_state: Optional[int] = None,
) -> List[int]:
    objectives = np.asarray(objectives, dtype=float)
    fronts = fast_non_dominated_sort(objectives)
    selected: List[int] = []
    last_front: Optional[List[int]] = None
    for front in fronts:
        if len(selected) + len(front) <= population_size:
            selected.extend(front)
        else:
            last_front = front
            break

    if last_front is None:
        return selected[:population_size]

    remaining = population_size - len(selected)
    if remaining <= 0:
        return selected[:population_size]

    dist = _crowding_distance(objectives, last_front)
    order = np.argsort(-dist)
    selected.extend([last_front[i] for i in order[:remaining]])
    return selected[:population_size]


def select_mo(
    objectives: np.ndarray,
    population_size: int,
    method: str = "nsga2",
    divisions: Union[int, Sequence[int]] = 4,
    random_state: Optional[int] = None,
) -> List[int]:
    """Dispatch multi-objective environmental selection by method name.

    Args:
        objectives: 2D array, rows are individuals, columns are objectives.
        population_size: number of individuals to keep for the next generation.
        method: "nsga2" or "nsga3" (also accepts "nsga-iii").
        divisions: reference-point divisions for nsga3.
        random_state: optional seed for reproducible tie-breaking.
    """
    normalized = method.strip().lower().replace("-", "").replace(" ", "")
    if normalized in ("nsga3", "nsgaiii"):
        return nsga3_selection(
            objectives,
            population_size,
            divisions=divisions,
            random_state=random_state,
        )
    if normalized == "nsga2":
        return nsga2_selection(
            objectives, population_size, random_state=random_state
        )
    if normalized in ("noveltysearch", "novelty_search", "novelty"):
        # The "objectives" matrix is interpreted as behavior descriptors;
        # selection keeps the most novel individuals.
        return novelty_selection(
            objectives,
            population_size,
            random_state=random_state,
        )
    if normalized in ("neat", "neuroevolution"):
        # The "objectives" matrix is interpreted as behavior/feature vectors;
        # NEAT-style speciation selects diverse, high-fitness individuals.
        return neat_selection(
            objectives,
            population_size,
            random_state=random_state,
        )
    raise ValueError(
        f"Unsupported multi-objective selection_method: {method!r}. "
        "Supported: nsga2, nsga3, novelty_search, neat."
    )


def run_symbolic_regression(
    data: Tuple[Sequence[Sequence[float]], Sequence[float]],
    generations: int = 100,
    pop_size: int = 200,
    function_set: Optional[Sequence[str]] = None,
    const_range: Tuple[float, float] = (-1.0, 1.0),
    init_depth: Tuple[int, int] = (2, 6),
    max_depth: int = 10,
    tournament_size: int = 3,
    p_crossover: float = 0.8,
    p_mutation: float = 0.15,
    p_reproduction: float = 0.05,
    parsimony_coefficient: float = 0.001,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> "SymbolicRegressionResult":
    """Run Genetic Programming symbolic regression.

    Thin delegation to :func:`openevolve.symbolic_regression.evolve`, mirroring
    how multi-objective ``select_mo`` delegates to ``nsga3`` / ``novelty_search``.
    Returns the best recovered expression (string + callable) and its fitness.
    """
    return _sr_evolve(
        data,
        generations=generations,
        pop_size=pop_size,
        function_set=function_set,
        const_range=const_range,
        init_depth=init_depth,
        max_depth=max_depth,
        tournament_size=tournament_size,
        p_crossover=p_crossover,
        p_mutation=p_mutation,
        p_reproduction=p_reproduction,
        parsimony_coefficient=parsimony_coefficient,
        random_state=random_state,
        verbose=verbose,
    )
