"""Enhanced Evolution Engine: orchestrates the enhanced algorithm variants.

Implements :class:`EnhancedEvolutionEngine` from
``docs/architecture/EVOLUTION_ALGORITHM_ENHANCEMENT_SPEC.md`` (section
"Enhanced Evolution Algorithms / Core Evolution Engine"). It composes the
already-implemented algorithm components -- :class:`openevolve.map_elites.MAPElites`
(quality-diversity), :class:`openevolve.multi_objective_selection.MultiObjectiveSelection`
and :mod:`openevolve.nsga3` / :mod:`openevolve.novelty_search` (multi-objective /
quality-diversity selection), and :class:`openevolve.differentiable_architecture_search.DifferentiableArchitectureSearch`
(neuroevolution) -- into a single, standalone evolution loop covering population
management, selection, variation, replacement, archive, diversity, convergence,
adaptation and (optional) knowledge integration.

Every optional dependency is imported behind a feature flag so a missing module
or an unconfigured component degrades gracefully instead of crashing the run.
Pure-numpy; no heavy external dependencies.

Quick start::

    engine = EnhancedEvolutionEngine(
        population_size=50,
        max_generations=100,
        selection_method="nsga2",
        random_state=0,
    )
    result = engine.evolve(evaluate=lambda g: {"fitness": -float(g @ g)})
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from openevolve import multi_objective_selection as _mos
from openevolve.nsga3 import nsga3_selection
from openevolve.novelty_search import novelty_selection

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Optional component availability (feature flags)
# --------------------------------------------------------------------------- #
try:
    from openevolve.map_elites import MAPElites, MAPElitesConfig

    _HAS_MAP_ELITES = True
except Exception:  # pragma: no cover - module always present in this dist
    MAPElites = None  # type: ignore
    MAPElitesConfig = None  # type: ignore
    _HAS_MAP_ELITES = False

try:
    from openevolve.differentiable_architecture_search import (
        DifferentiableArchitectureSearch,
    )

    _HAS_DARTS = True
except Exception:  # pragma: no cover
    DifferentiableArchitectureSearch = None  # type: ignore
    _HAS_DARTS = False

try:
    # Adaptive operators are wired into the controller; degrade to a built-in
    # adapter if the package is ever unavailable.
    from openevolve.self_adaptive import SelfAdaptiveOperators

    _HAS_SELF_ADAPTIVE = True
except Exception:  # pragma: no cover
    SelfAdaptiveOperators = None  # type: ignore
    _HAS_SELF_ADAPTIVE = False


# --------------------------------------------------------------------------- #
# Individuals / population
# --------------------------------------------------------------------------- #
@dataclass
class Individual:
    """A single solution tracked by the engine.

    ``genotype`` is opaque to the engine (any object the ``evaluate`` callable
    understands, e.g. a numpy vector or a program id). The evolution
    bookkeeping uses ``fitness`` (maximise), ``objectives`` (a minimisation
    vector for multi-objective selection) and an optional ``behavior_descriptor``
    (fixed-length vector for quality-diversity archives).
    """

    genotype: Any
    fitness: float = 0.0
    objectives: List[float] = field(default_factory=list)
    behavior_descriptor: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    uid: int = 0

    def objective_vector(self) -> List[float]:
        if self.objectives:
            return list(self.objectives)
        return [max(0.0, -self.fitness)]


@dataclass
class EnhancedEvolutionConfig:
    """Configuration for :class:`EnhancedEvolutionEngine`."""

    population_size: int = 50
    max_generations: int = 100
    initial_population_size: Optional[int] = None
    offspring_size: Optional[int] = None

    # Selection / replacement
    selection_method: str = "nsga2"  # nsga2 | nsga3 | novelty | map_elites | multi_objective | tournament | random
    selection_pressure: float = 1.0
    tournament_size: int = 3
    elitism: float = 0.1  # fraction of top individuals copied verbatim
    senses: Optional[Sequence[Union[str, int, bool, float]]] = None
    nsga3_divisions: Union[int, Sequence[int]] = 4

    # Variation
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    mutation_scale: float = 0.1
    variation_fn: Optional[Callable[[Any, "EnhancedEvolutionConfig", random.Random], Any]] = None

    # Quality-diversity
    use_map_elites: bool = False
    use_novelty_archive: bool = False
    behavior_descriptors: Sequence[Any] = field(default_factory=list)
    map_elites_strategy: str = "improvement"
    novelty_k: int = 15
    novelty_threshold: float = 0.5
    archive_size_limit: Optional[int] = None

    # Diversity / convergence
    diversity_threshold: float = 1e-3
    convergence_threshold: float = 1e-6
    convergence_window: int = 15
    stall_window: int = 20
    terminate_on_convergence: bool = True

    # Adaptation / knowledge
    use_adaptation: bool = True
    use_knowledge_integration: bool = False
    knowledge_integrator: Optional[Any] = None

    # Neuroevolution (DARTS) search at the end of a run
    run_darts: bool = False
    darts_epochs: int = 30

    random_state: Optional[int] = None

    def resolve(self) -> "EnhancedEvolutionConfig":
        if self.initial_population_size is None:
            self.initial_population_size = self.population_size
        if self.offspring_size is None:
            self.offspring_size = max(1, self.population_size // 2)
        return self


# --------------------------------------------------------------------------- #
# Variation (default numeric mutation / crossover)
# --------------------------------------------------------------------------- #
def _default_variation(
    genotype: Any, config: EnhancedEvolutionConfig, rng: random.Random
) -> Any:
    """Generic numeric-aware variation: Gaussian mutation + blend crossover.

    Works on numpy arrays, lists/tuples and scalars. Non-numeric genotypes are
    returned unchanged (callers that evolve code should supply their own
    ``variation_fn``).
    """
    if isinstance(genotype, np.ndarray):
        arr = genotype.astype(float).copy()
        flat_mask = np.array([rng.random() for _ in range(arr.size)])
        mask = flat_mask.reshape(arr.shape) < config.mutation_rate
        flat_noise = np.array([rng.gauss(0.0, 1.0) for _ in range(arr.size)])
        noise = flat_noise.reshape(arr.shape) * config.mutation_scale
        return arr + mask * noise
    if isinstance(genotype, (list, tuple)):
        arr = np.asarray(genotype, dtype=float)
        return _default_variation(arr, config, rng)
    if isinstance(genotype, (int, float)):
        if rng.random() < config.mutation_rate:
            return genotype + rng.gauss(0.0, config.mutation_scale)
        return genotype
    return genotype


def _crossover(
    a: Any, b: Any, config: EnhancedEvolutionConfig, rng: random.Random
) -> Any:
    if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
        return a if rng.random() < 0.5 else b
    alpha = rng.random()
    return alpha * a + (1.0 - alpha) * b


# --------------------------------------------------------------------------- #
# Archive manager (MAP-Elites QD grid + novelty archive + Pareto fallback)
# --------------------------------------------------------------------------- #
class ArchiveManager:
    """Maintains elite / diverse solutions across generations.

    - When ``behavior_descriptors`` are configured and MAP-Elites is enabled, a
      :class:`MAPElites` QD grid stores the fittest individual per behavior cell.
    - Otherwise a Pareto archive keeps the current non-dominated set.
    - A novelty archive (behavior descriptors) is kept when enabled.
    """

    def __init__(self, config: EnhancedEvolutionConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng
        self.map_elites: Optional[Any] = None
        if config.use_map_elites and _HAS_MAP_ELITES and config.behavior_descriptors:
            try:
                me_cfg = MAPElitesConfig(
                    behavior_descriptors=list(config.behavior_descriptors),
                    selection_strategy=config.map_elites_strategy,
                    archive_size_limit=config.archive_size_limit,
                )
                self.map_elites = MAPElites(config=me_cfg, random_state=rng.randint(0, 2**31))
            except Exception as exc:  # pragma: no cover
                logger.warning("MAP-Elites archive disabled: %s", exc)
                self.map_elites = None
        self.pareto: List[Individual] = []
        self.novelty_archive: List[List[float]] = []

    def update(self, population: Sequence[Individual]) -> Dict[str, Any]:
        if self.map_elites is not None:
            for ind in population:
                if ind.behavior_descriptor:
                    self.map_elites.add(
                        ind.fitness,
                        ind.behavior_descriptor,
                        genotype=ind.genotype,
                        metadata={"uid": ind.uid},
                    )
        # Pareto archive: keep non-dominated individuals only.
        candidates = list(self.pareto) + list(population)
        self.pareto = self._filter_pareto(candidates)
        if self.config.archive_size_limit and len(self.pareto) > self.config.archive_size_limit:
            self.pareto = sorted(self.pareto, key=lambda i: -i.fitness)[: self.config.archive_size_limit]

        if self.config.use_novelty_archive:
            for ind in population:
                if ind.behavior_descriptor:
                    self.novelty_archive.append(list(ind.behavior_descriptor))
            limit = self.config.archive_size_limit
            if limit and len(self.novelty_archive) > limit:
                self.novelty_archive = self.novelty_archive[-limit:]

        return self.statistics()

    def _filter_pareto(self, individuals: Sequence[Individual]) -> List[Individual]:
        if not individuals:
            return []
        vecs = np.asarray([i.objective_vector() for i in individuals], dtype=float)
        if vecs.ndim == 1:
            vecs = vecs.reshape(-1, 1)
        fronts = _mos.fast_non_dominated_sort(
            _mos.apply_senses(vecs, self.config.senses)
        )
        return [individuals[i] for i in fronts[0]] if fronts else list(individuals)

    def statistics(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "pareto_size": len(self.pareto),
            "novelty_archive_size": len(self.novelty_archive),
        }
        if self.map_elites is not None:
            stats.update(self.map_elites.archive_statistics())
        return stats

    def get_archive(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "pareto": [
                {"uid": i.uid, "fitness": i.fitness, "objectives": i.objective_vector()}
                for i in self.pareto
            ],
            "novelty_archive": list(self.novelty_archive),
        }
        if self.map_elites is not None:
            out["map_elites"] = {str(k): v.fitness for k, v in self.map_elites.grid.cells.items()}
            out["map_elites_coverage"] = self.map_elites.coverage()
        return out


# --------------------------------------------------------------------------- #
# Diversity metrics
# --------------------------------------------------------------------------- #
class DiversityMetrics:
    """Computes population diversity from objectives / behavior descriptors."""

    def calculate(self, population: Sequence[Individual]) -> Dict[str, float]:
        if not population:
            return {"entropy": 0.0, "spread": 0.0, "mean_fitness": 0.0, "std_fitness": 0.0}
        fitness = np.asarray([p.fitness for p in population], dtype=float)
        metrics: Dict[str, float] = {
            "mean_fitness": float(np.mean(fitness)),
            "std_fitness": float(np.std(fitness)),
        }
        # Normalised Shannon entropy over binned fitness.
        if fitness.max() > fitness.min():
            bins = np.histogram(fitness, bins=10)[0].astype(float)
            bins = bins[bins > 0]
            p = bins / bins.sum()
            metrics["entropy"] = float(-np.sum(p * np.log(p)))
        else:
            metrics["entropy"] = 0.0
        try:
            obj_matrix = np.asarray([p.objective_vector() for p in population], dtype=float)
            if obj_matrix.ndim > 1:
                metrics["spread"] = float(np.mean(np.std(obj_matrix, axis=0)))
            else:
                metrics["spread"] = 0.0
        except Exception:  # pragma: no cover
            metrics["spread"] = 0.0
        return metrics


# --------------------------------------------------------------------------- #
# Convergence detector
# --------------------------------------------------------------------------- #
class ConvergenceDetector:
    """Detects stagnation / convergence from best-fitness history."""

    def __init__(self, config: EnhancedEvolutionConfig) -> None:
        self.config = config
        self.best_history: List[float] = []

    def check_convergence(self, best_fitness: float, generation: int) -> bool:
        self.best_history.append(best_fitness)
        window = self.config.convergence_window
        if len(self.best_history) < window:
            return False
        recent = self.best_history[-window:]
        improvement = abs(recent[-1] - max(recent[:-1]))
        return improvement <= self.config.convergence_threshold

    def stalled(self) -> bool:
        window = self.config.stall_window
        if len(self.best_history) < window:
            return False
        return self.best_history[-1] <= max(self.best_history[-window:]) + self.config.convergence_threshold


# --------------------------------------------------------------------------- #
# Adaptation engine
# --------------------------------------------------------------------------- #
class AdaptationEngine:
    """Adjusts mutation_rate / selection_pressure from population signals.

    Prefers :class:`openevolve.self_adaptive.SelfAdaptiveOperators` when present
    (so it integrates with the controller's adaptive metrics); otherwise uses a
    built-in density/plateau heuristic matching the spec's ``AdaptationEngine``.
    """

    def __init__(self, config: EnhancedEvolutionConfig) -> None:
        self.config = config
        self.self_adaptive = None
        if _HAS_SELF_ADAPTIVE:
            try:
                self.self_adaptive = SelfAdaptiveOperators()
            except Exception:  # pragma: no cover
                self.self_adaptive = None

    def adapt(
        self,
        population: Sequence[Individual],
        diversity: Dict[str, float],
        converged: bool,
    ) -> Dict[str, float]:
        if self.self_adaptive is not None:
            try:
                scores = [p.fitness for p in population]
                params = self.self_adaptive.update(scores)
                self.config.mutation_rate = float(params.get("mutation_rate", self.config.mutation_rate))
                self.config.selection_pressure = float(
                    params.get("selection_pressure", self.config.selection_pressure)
                )
                self.config.crossover_rate = float(params.get("crossover_rate", self.config.crossover_rate))
                return dict(params)
            except Exception as exc:  # pragma: no cover
                logger.debug("SelfAdaptiveOperators failed, using built-in: %s", exc)

        # Built-in heuristic adaptation.
        entropy = diversity.get("entropy", 1.0)
        mutation_rate = self.config.mutation_rate
        selection_pressure = self.config.selection_pressure
        crossover_rate = self.config.crossover_rate

        if entropy < self.config.diversity_threshold:
            mutation_rate = min(mutation_rate * 1.1, 0.9)
        else:
            mutation_rate = max(mutation_rate * 0.95, 0.01)
        if converged:
            selection_pressure = max(selection_pressure * 0.9, 0.5)
        else:
            selection_pressure = min(selection_pressure * 1.05, 5.0)

        self.config.mutation_rate = mutation_rate
        self.config.selection_pressure = selection_pressure
        self.config.crossover_rate = crossover_rate
        return {
            "mutation_rate": mutation_rate,
            "selection_pressure": selection_pressure,
            "crossover_rate": crossover_rate,
        }


# --------------------------------------------------------------------------- #
# Knowledge integrator (optional)
# --------------------------------------------------------------------------- #
class KnowledgeIntegrator:
    """Thin optional hook for knowledge-based guidance.

    If ``config.knowledge_integrator`` is supplied it is used directly; else an
    external ``openevolve.agents.investment.knowledge_integrator`` is attempted
    but never required. ``integrate`` is a safe no-op when unavailable.
    """

    def __init__(self, config: EnhancedEvolutionConfig) -> None:
        self.config = config
        self._integrator = config.knowledge_integrator
        if self._integrator is None and config.use_knowledge_integration:
            try:  # pragma: no cover - optional dependency
                from openevolve.agents.investment.knowledge_integrator import (
                    KnowledgeIntegrator as _KI,
                )

                self._integrator = _KI()
            except Exception as exc:
                logger.warning("Knowledge integrator unavailable: %s", exc)
                self._integrator = None

    def integrate(self, population: Sequence[Individual], generation: int) -> Optional[Any]:
        if self._integrator is None:
            return None
        try:
            if hasattr(self._integrator, "integrate"):
                return self._integrator.integrate(population, generation)
            if callable(self._integrator):
                return self._integrator(population, generation)
        except Exception as exc:  # pragma: no cover
            logger.warning("Knowledge integration step failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class EnhancedEvolutionEngine:
    """Standalone orchestrator for the enhanced evolution algorithm variants.

    The engine owns a population of :class:`Individual` objects and runs the
    standard loop -- initialize, evaluate, select parents, vary, evaluate
    offspring, replace, archive, measure diversity, detect convergence, adapt,
    integrate knowledge -- delegating the algorithm-specific work to the
    composed components above and to the existing ``openevolve`` selection
    modules.

    Args:
        config: an :class:`EnhancedEvolutionConfig`, or keyword overrides.
        population_size / max_generations / selection_method / mutation_rate /
            crossover_rate / selection_pressure / elitism / random_state:
            convenience overrides forwarded to the config.
        evaluate: callable ``genotype -> result``. ``result`` may be a dict with
            keys ``fitness``, ``objectives``, ``behavior_descriptor``; a bare
            float (fitness); or an :class:`EvaluationResult`. When omitted the
            engine can still run structure-only loops (e.g. for DARTS).
    """

    def __init__(self, *args, config: Optional[EnhancedEvolutionConfig] = None, **kwargs):
        if config is None:
            config = EnhancedEvolutionConfig()
        if config is None:
            config = EnhancedEvolutionConfig()
        # Apply keyword overrides.
        for key in (
            "population_size",
            "max_generations",
            "initial_population_size",
            "offspring_size",
            "selection_method",
            "selection_pressure",
            "tournament_size",
            "elitism",
            "senses",
            "nsga3_divisions",
            "mutation_rate",
            "crossover_rate",
            "mutation_scale",
            "variation_fn",
            "use_map_elites",
            "use_novelty_archive",
            "behavior_descriptors",
            "map_elites_strategy",
            "novelty_k",
            "novelty_threshold",
            "archive_size_limit",
            "diversity_threshold",
            "convergence_threshold",
            "convergence_window",
            "stall_window",
            "terminate_on_convergence",
            "use_adaptation",
            "use_knowledge_integration",
            "knowledge_integrator",
            "run_darts",
            "darts_epochs",
            "random_state",
        ):
            if key in kwargs and kwargs[key] is not None:
                setattr(config, key, kwargs[key])
        config = config.resolve()
        self.config = config
        self.rng = random.Random(config.random_state)
        self._uid_counter = 0

        self.diversity_metrics = DiversityMetrics()
        self.convergence_detector = ConvergenceDetector(config)
        self.archive_manager = ArchiveManager(config, self.rng)
        self.adaptation_engine = AdaptationEngine(config) if config.use_adaptation else None
        self.knowledge_integrator = KnowledgeIntegrator(config)
        self.population: List[Individual] = []
        self.history: List[Dict[str, float]] = []
        self.generation = 0

    # -- population management ------------------------------------------- #
    def _next_uid(self) -> int:
        self._uid_counter += 1
        return self._uid_counter

    def initialize_population(
        self, seed_genotypes: Optional[Sequence[Any]] = None
    ) -> List[Individual]:
        population: List[Individual] = []
        size = self.config.initial_population_size or self.config.population_size
        if seed_genotypes:
            for g in seed_genotypes[:size]:
                population.append(Individual(genotype=g, uid=self._next_uid()))
        else:
            # Default numeric genotypes (so the engine is usable standalone).
            dim = 2
            for _ in range(size):
                genotype = np.asarray(
                    [self.rng.gauss(0.0, 1.0) for _ in range(dim)], dtype=float
                )
                population.append(Individual(genotype=genotype, uid=self._next_uid()))
        self.population = population
        return population

    # -- evaluation ------------------------------------------------------- #
    def evaluate_individual(self, individual: Individual, evaluate: Callable) -> Individual:
        result = evaluate(individual.genotype)
        if isinstance(result, dict):
            individual.fitness = float(result.get("fitness", individual.fitness))
            if "objectives" in result and result["objectives"] is not None:
                individual.objectives = [float(o) for o in result["objectives"]]
            if "behavior_descriptor" in result and result["behavior_descriptor"] is not None:
                individual.behavior_descriptor = [float(b) for b in result["behavior_descriptor"]]
            if "metadata" in result and result["metadata"]:
                individual.metadata.update(result["metadata"])
        elif isinstance(result, (int, float)):
            individual.fitness = float(result)
        elif hasattr(result, "metrics"):
            metrics = getattr(result, "metrics", {}) or {}
            individual.fitness = float(metrics.get("combined_score", individual.fitness))
        else:
            raise TypeError(f"Unsupported evaluation result: {type(result)!r}")
        return individual

    def evaluate_population(
        self, population: Sequence[Individual], evaluate: Callable
    ) -> List[Individual]:
        return [self.evaluate_individual(ind, evaluate) for ind in population]

    # -- selection / replacement ------------------------------------------ #
    def _objective_matrix(self, population: Sequence[Individual]) -> np.ndarray:
        matrix = np.asarray([p.objective_vector() for p in population], dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        return matrix

    def select_parents(self, population: Sequence[Individual]) -> List[Individual]:
        if not population:
            return []
        matrix = self._objective_matrix(population)
        selector = _mos.MultiObjectiveSelection(
            senses=self.config.senses,
            tournament_size=self.config.tournament_size,
            random_state=self.rng.randint(0, 2**31),
        )
        n_parents = self.config.offspring_size or (len(population) // 2)
        idx = selector.select_parents(matrix, n_parents, with_replacement=True)
        return [population[i] for i in idx]

    def replace_population(
        self, parents: Sequence[Individual], offspring: Sequence[Individual]
    ) -> List[Individual]:
        combined = list(parents) + list(offspring)
        matrix = self._objective_matrix(combined)
        method = (self.config.selection_method or "nsga2").lower()
        pop_size = self.config.population_size

        # Elitism: carry over the very best individuals verbatim.
        elite_n = int(round(self.config.elitism * pop_size))
        elites: List[Individual] = []
        if elite_n > 0:
            order = sorted(range(len(combined)), key=lambda i: -combined[i].fitness)
            elites = [combined[i] for i in order[:elite_n]]

        try:
            if method in ("nsga3",) and matrix.shape[1] > 1:
                survivors_idx = nsga3_selection(
                    matrix,
                    pop_size - len(elites),
                    divisions=self.config.nsga3_divisions,
                    random_state=self.rng.randint(0, 2**31),
                )
            elif method == "novelty" and self.config.behavior_descriptors:
                behaviors = np.asarray(
                    [p.behavior_descriptor for p in combined], dtype=float
                )
                if behaviors.ndim == 1:
                    behaviors = behaviors.reshape(-1, 1)
                survivors_idx = novelty_selection(
                    behaviors,
                    pop_size - len(elites),
                    k=self.config.novelty_k,
                    random_state=self.rng.randint(0, 2**31),
                )
            elif method in ("multi_objective", "nsga2", "map_elites"):
                survivors_idx = _mos.multi_objective_selection(
                    matrix,
                    pop_size - len(elites),
                    senses=self.config.senses,
                    random_state=self.rng.randint(0, 2**31),
                )
            else:  # tournament / random / fallback
                selector = _mos.MultiObjectiveSelection(
                    senses=self.config.senses,
                    tournament_size=self.config.tournament_size,
                    random_state=self.rng.randint(0, 2**31),
                )
                survivors_idx = selector.environmental_selection(matrix, pop_size - len(elites))
        except Exception as exc:  # pragma: no cover
            logger.warning("Selection '%s' failed, using crowding fallback: %s", method, exc)
            survivors_idx = _mos.multi_objective_selection(matrix, pop_size - len(elites))

        survivors = [combined[i] for i in survivors_idx]
        return elites + survivors

    # -- variation -------------------------------------------------------- #
    def vary(self, parents: Sequence[Individual]) -> List[Individual]:
        offspring: List[Individual] = []
        variation_fn = self.config.variation_fn or _default_variation
        n = max(1, self.config.offspring_size or len(parents))
        if not parents:
            return offspring
        for _ in range(n):
            if len(parents) >= 2 and self.rng.random() < self.config.crossover_rate:
                a, b = self.rng.sample(list(parents), 2)
                child_genotype = _crossover(a.genotype, b.genotype, self.config, self.rng)
            else:
                a = self.rng.choice(list(parents))
                child_genotype = a.genotype
            child_genotype = variation_fn(child_genotype, self.config, self.rng)
            offspring.append(Individual(genotype=child_genotype, uid=self._next_uid()))
        return offspring

    # -- termination ------------------------------------------------------ #
    def should_terminate(self, converged: bool) -> bool:
        if self.generation >= self.config.max_generations:
            return True
        if converged and self.config.terminate_on_convergence:
            return True
        return False

    # -- main loop -------------------------------------------------------- #
    def evolve(
        self,
        evaluate: Optional[Callable] = None,
        initial_population: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        """Run the enhanced evolution loop.

        Returns a result dict with the final population, archive, diversity
        metrics, convergence generation and total generations.
        """
        population = self.initialize_population(initial_population)
        if evaluate is not None:
            population = self.evaluate_population(population, evaluate)
        self.population = population

        converged = False
        for generation in range(self.config.max_generations):
            self.generation = generation

            parents = self.select_parents(population)
            offspring = self.vary(parents)
            if evaluate is not None:
                offspring = self.evaluate_population(offspring, evaluate)

            population = self.replace_population(population, offspring)
            if evaluate is not None:
                # Survivors keep their cached evaluation; re-evaluate only if
                # they were freshly created this generation (offspring).
                pass

            archive_stats = self.archive_manager.update(population)
            diversity = self.diversity_metrics.calculate(population)
            best_fitness = max((p.fitness for p in population), default=0.0)
            converged = self.convergence_detector.check_convergence(best_fitness, generation)

            if self.adaptation_engine is not None:
                self.adaptation_engine.adapt(population, diversity, converged)

            if self.config.use_knowledge_integration:
                self.knowledge_integrator.integrate(population, generation)

            self.history.append(
                {
                    "generation": float(generation),
                    "best_fitness": best_fitness,
                    "mean_fitness": diversity["mean_fitness"],
                    "entropy": diversity["entropy"],
                    "archive_pareto": float(archive_stats.get("pareto_size", 0)),
                    **{f"archive_{k}": v for k, v in archive_stats.items() if k != "pareto_size"},
                }
            )

            if self.should_terminate(converged):
                break

        result = {
            "final_population": population,
            "best": max(population, key=lambda p: p.fitness) if population else None,
            "archive": self.archive_manager.get_archive(),
            "diversity_metrics": diversity if population else {},
            "convergence_generation": self.generation if converged else None,
            "total_generations": self.generation + 1,
            "history": self.history,
        }

        if self.config.run_darts and _HAS_DARTS and evaluate is not None:
            result["darts"] = self._run_darts(population)
        return result

    # -- optional neuroevolution (DARTS) ---------------------------------- #
    def _run_darts(self, population: Sequence[Individual]) -> Optional[Dict[str, Any]]:
        """Train a differentiable supernet on population descriptors (if numeric)."""
        try:
            descriptors = [
                np.asarray(p.behavior_descriptor, dtype=float)
                for p in population
                if p.behavior_descriptor
            ]
            if len(descriptors) < 4:
                return None
            X = np.vstack(descriptors)
            y = np.asarray([p.fitness for p in population], dtype=float)
            das = DifferentiableArchitectureSearch(
                input_dim=X.shape[1],
                output_dim=1,
                search_epochs=self.config.darts_epochs,
                random_state=self.rng.randint(0, 2**31),
            )
            return das.search(X, y)
        except Exception as exc:  # pragma: no cover
            logger.warning("DARTS search skipped: %s", exc)
            return None


# --------------------------------------------------------------------------- #
# Functional entry point (mirrors run_cmaes / run_darts style)
# --------------------------------------------------------------------------- #
def evolve(
    evaluate: Callable,
    population_size: int = 50,
    max_generations: int = 100,
    selection_method: str = "nsga2",
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """One-shot functional wrapper around :class:`EnhancedEvolutionEngine`."""
    engine = EnhancedEvolutionEngine(
        population_size=population_size,
        max_generations=max_generations,
        selection_method=selection_method,
        random_state=random_state,
        **kwargs,
    )
    return engine.evolve(evaluate=evaluate)
