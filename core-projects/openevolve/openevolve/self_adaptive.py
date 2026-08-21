"""
Self-Adaptive Operators for OpenEvolve evolution runs.

Implements :class:`SelfAdaptiveOperators`, the evolution-context counterpart of
the ``SelfAdaptiveOperators`` / ``AdaptationEngine`` described in
``docs/architecture/EVOLUTION_ALGORITHM_ENHANCEMENT_SPEC.md``.

It keeps an internal set of operator parameters
(``mutation_rate``, ``crossover_rate``, ``selection_pressure``, ``elitism``)
and adjusts them every generation based on the run's progress. The progress
signals come from the *already-implemented* adaptive metrics in
``openevolve.config.config_metrics`` (deterministic and dependency-free), so we
do not recompute fitness slopes / diversity ourselves.

The controller instantiates this (only when ``Config.adaptive_parameters`` is
enabled) and calls :meth:`SelfAdaptiveOperators.update` once per generation with
the best-fitness history and the current population scores. The returned
parameters are then applied to the live ``ProgramDatabase`` selection ratios so
the adaptation genuinely affects the run. With the flag disabled the default
path is left completely untouched.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from openevolve.config.config_metrics import compute_adaptive_metrics
from openevolve.config import SelfAdaptiveConfig

logger = logging.getLogger(__name__)


def _clamp(value: float, bounds) -> float:
    lo, hi = bounds
    return max(lo, min(hi, value))


class SelfAdaptiveOperators:
    """
    Stateful self-adaptive operator controller.

    Maintains ``params`` across generations and nudges them toward an adapted
    target derived from the run's adaptive metrics. The adaptation rules mirror
    the spec's ``AdaptationEngine``:

    - More stagnation -> explore more (higher mutation, lower selection
      pressure, so the run escapes local optima).
    - High population diversity -> exploit (lower mutation, higher selection
      pressure).
    - Low diversity -> preserve spread (more mutation / exploration).
    - Elitism tracks selection pressure so good solutions are retained more
      aggressively while exploiting.

    All updates are deterministic: the same history always yields the same
    parameters, and every parameter is kept within its configured bounds.
    """

    PARAM_KEYS = (
        "mutation_rate",
        "crossover_rate",
        "selection_pressure",
        "elitism",
    )

    def __init__(self, config: Optional[SelfAdaptiveConfig] = None):
        cfg = config or SelfAdaptiveConfig()

        self.bounds = {
            "mutation_rate": cfg.mutation_rate_bounds,
            "crossover_rate": cfg.crossover_rate_bounds,
            "selection_pressure": cfg.selection_pressure_bounds,
            "elitism": cfg.elitism_bounds,
        }

        self.params = {
            "mutation_rate": _clamp(cfg.initial_mutation_rate, self.bounds["mutation_rate"]),
            "crossover_rate": _clamp(cfg.initial_crossover_rate, self.bounds["crossover_rate"]),
            "selection_pressure": _clamp(
                cfg.initial_selection_pressure, self.bounds["selection_pressure"]
            ),
            "elitism": _clamp(cfg.initial_elitism, self.bounds["elitism"]),
        }

        self.diversity_low = cfg.diversity_low_threshold
        self.diversity_high = cfg.diversity_high_threshold
        self.learning_rate = cfg.learning_rate
        self.window = cfg.window

        self.fitness_history: List[float] = []
        self.generation = 0
        self.last_metrics = None

    def update(
        self,
        best_score: float,
        population_scores: Optional[List[float]] = None,
        generation: Optional[int] = None,
    ) -> Dict[str, float]:
        """Adapt operator parameters for one generation.

        Args:
            best_score: Best fitness achieved so far this generation.
            population_scores: Fitness of the current population individuals.
            generation: Current generation index (for call-site symmetry).

        Returns:
            The updated operator parameters (all within configured bounds).
        """
        if generation is not None:
            self.generation = generation

        self.fitness_history.append(float(best_score))

        metrics = compute_adaptive_metrics(
            self.fitness_history,
            population_scores,
            iteration=self.generation,
            window=self.window,
        )
        self.last_metrics = metrics

        stagnation = metrics.stagnation_index
        diversity = metrics.diversity
        improvement = metrics.improvement_rate
        lr = self.learning_rate

        m = self.params

        # More stagnation -> explore more (higher mutation, lower selection).
        m["mutation_rate"] *= 1.0 + lr * stagnation
        m["selection_pressure"] *= 1.0 - lr * stagnation

        # High diversity -> exploit (less mutation, more selection pressure).
        if diversity > self.diversity_high:
            m["mutation_rate"] *= 1.0 - lr
            m["selection_pressure"] *= 1.0 + lr
        # Low diversity -> preserve spread (more mutation / exploration).
        elif diversity < self.diversity_low:
            m["mutation_rate"] *= 1.0 + lr

        # Crossover tracks improvement (more recombination while progressing).
        m["crossover_rate"] *= 1.0 + 0.1 * improvement

        # Elitism follows selection pressure so good solutions are retained
        # more aggressively when we are exploiting.
        m["elitism"] = 0.05 + 0.25 * m["selection_pressure"]

        # Clamp everything back into configured bounds.
        for key in self.params:
            self.params[key] = round(_clamp(self.params[key], self.bounds[key]), 6)

        return dict(self.params)

    def get_params(self) -> Dict[str, float]:
        """Return the current operator parameters (a copy)."""
        return dict(self.params)

    def reset(self) -> None:
        """Reset history and parameters to their initial state."""
        self.fitness_history = []
        self.generation = 0
        self.last_metrics = None
        cfg = SelfAdaptiveConfig()
        self.params = {
            "mutation_rate": _clamp(cfg.initial_mutation_rate, self.bounds["mutation_rate"]),
            "crossover_rate": _clamp(cfg.initial_crossover_rate, self.bounds["crossover_rate"]),
            "selection_pressure": _clamp(
                cfg.initial_selection_pressure, self.bounds["selection_pressure"]
            ),
            "elitism": _clamp(cfg.initial_elitism, self.bounds["elitism"]),
        }
