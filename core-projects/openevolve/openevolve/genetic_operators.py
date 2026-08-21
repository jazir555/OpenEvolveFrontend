"""
Genetic operators for the OpenEvolve evolution loop.

These operators make the documented core-evolution parameters
``mutation_rate``, ``crossover_rate``, ``selection_method``,
``elitism`` and ``selection_pressure`` real, configurable knobs. They are
applied ONLY when ``Config.use_genetic_operators`` is ``True``. With the flag
off the evolution loop is byte-for-byte unchanged (the operators are never
imported or called).

All operators are pure functions that operate on ``Program`` objects / code
strings and accept an injectable ``random.Random`` for reproducibility.
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional, Sequence

from openevolve.database import Program
from openevolve.utils.metrics_utils import get_fitness_score

logger = logging.getLogger(__name__)


def _fitness(program: Program, feature_dimensions: Sequence[str]) -> float:
    """Fitness score for a program (always >= 0)."""
    return max(
        get_fitness_score(program.metrics, list(feature_dimensions)),
        0.0,
    )


def select_parent(
    population: Sequence[Program],
    method: str = "tournament",
    selection_pressure: float = 1.0,
    feature_dimensions: Optional[Sequence[str]] = None,
    rng: Optional[random.Random] = None,
) -> Program:
    """Select a parent program using the configured selection strategy.

    - ``tournament``: tournament of size ``max(2, round(selection_pressure * 3))``;
      the fittest candidate wins.
    - ``roulette``: fitness-proportional sampling; ``selection_pressure`` sharpens
      the distribution by raising each fitness to that power.
    - ``rank``: linear rank-based sampling (elitism-aware: top ranks weighted
      highest).

    Falls back to uniform random selection for tiny / empty populations and
    normalizes gracefully when fitnesses are non-positive.
    """
    rng = rng or random
    feature_dimensions = list(feature_dimensions or [])
    progs = list(population)
    if not progs:
        raise ValueError("select_parent called on an empty population")
    if len(progs) == 1:
        return progs[0]

    method = (method or "tournament").strip().lower()

    if method in ("roulette", "rank"):
        if method == "roulette":
            pressure = max(selection_pressure, 0.0)
            weights = [(_fitness(p, feature_dimensions) + 1e-6) ** pressure for p in progs]
        else:  # rank
            ranked = sorted(
                progs,
                key=lambda p: _fitness(p, feature_dimensions),
                reverse=True,
            )
            n = len(ranked)
            weights = [(n - i) / n for i in range(n)]
        total = sum(weights)
        if total <= 0:
            weights = [1.0 / len(progs)] * len(progs)
        else:
            weights = [w / total for w in weights]
        return rng.choices(progs, weights=weights, k=1)[0]

    # default: tournament
    tourney_size = max(2, int(round(selection_pressure * 3)))
    tourney_size = min(tourney_size, len(progs))
    candidates = rng.sample(progs, tourney_size)
    return max(
        candidates,
        key=lambda p: _fitness(p, feature_dimensions),
    )


def crossover(
    parent_a: Program,
    parent_b: Program,
    rate: float,
    rng: Optional[random.Random] = None,
) -> str:
    """Crossover two parent programs into a single candidate code string.

    Returns ``parent_a.code`` unchanged when ``rate <= 0`` (callers are also
    free to gate the call on ``rng.random() < rate``). Otherwise performs a real
    one-point line crossover: the first ``k`` lines of A followed by the
    remaining lines of B, where ``k`` is chosen uniformly.
    """
    if rate <= 0:
        return parent_a.code
    rng = rng or random
    a_lines = parent_a.code.splitlines()
    b_lines = parent_b.code.splitlines()
    if not a_lines or not b_lines or len(a_lines) < 2:
        return parent_a.code
    k = rng.randint(1, len(a_lines) - 1)
    return "\n".join(a_lines[:k] + b_lines[k:])


def mutate_code(
    code: str,
    rate: float,
    rng: Optional[random.Random] = None,
) -> str:
    """Apply a simple, syntactically-safe code-level mutation with prob ``rate``.

    Inserts a unique comment line at a random position. Returns the code
    unchanged when ``rate <= 0`` or the random roll fails. The mutation is
    reversible (a comment) so it never breaks program syntax, making it safe to
    exercise offline.
    """
    if rate <= 0:
        return code
    rng = rng or random
    if rng.random() >= rate:
        return code
    lines = code.splitlines()
    if not lines:
        lines = [""]
    idx = rng.randrange(len(lines))
    lines.insert(idx, f"# evolved mutation {rng.randint(0, 10 ** 6)}")
    return "\n".join(lines)


def mutation_temperature_scale(
    base_temperature: float,
    mutation_rate: float,
) -> float:
    """Scale the LLM sampling temperature by the mutation rate.

    Higher ``mutation_rate`` -> higher temperature (more variation). When
    ``mutation_rate`` is 0 the base temperature is returned unchanged.
    """
    if mutation_rate <= 0:
        return base_temperature
    return min(2.0, float(base_temperature) * (1.0 + mutation_rate))


def elite_programs(
    population: Sequence[Program],
    elite_count: int,
    feature_dimensions: Optional[Sequence[str]] = None,
) -> List[Program]:
    """Return the top ``elite_count`` programs by fitness (high to low)."""
    if elite_count <= 0:
        return []
    feature_dimensions = list(feature_dimensions or [])
    progs = sorted(
        population,
        key=lambda p: _fitness(p, feature_dimensions),
        reverse=True,
    )
    return progs[:elite_count]
