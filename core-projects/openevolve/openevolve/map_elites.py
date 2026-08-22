"""MAP-Elites: a Quality-Diversity (QD) optimizer.

Implements :class:`MAPElites` from
``docs/architecture/EVOLUTION_ALGORITHM_ENHANCEMENT_SPEC.md`` (section
"Quality-Diversity Optimization / MAP-Elites Implementation"). It maintains an
n-dimensional *behavior descriptor* grid archive: every cell is indexed by a
behavior characterisation (e.g. program runtime, memory, accuracy) and stores
the single *elite* with the highest fitness seen in that cell. Because only
one elite is kept per cell, MAP-Elites simultaneously searches for high
fitness and for behavioral diversity.

The coordinate mapping mirrors ``openevolve.database.ProgramDatabase``'s
MAP-Elites grid: each descriptor dimension has ``(min, max, bins)`` bounds and
a value is normalised to ``[0, 1]`` then floored into a bin index.

Pure-numpy, dependency free beyond numpy. The :func:`map_elites_selection`
wrapper follows the same selection-contract as ``nsga3_selection`` /
``novelty_selection``: a 2D matrix in, selected indices out. See
:func:`openevolve.selection.select_mo` for how it is dispatched.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from openevolve.multi_objective_selection import crowding_distance

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Behavior space / grid
# --------------------------------------------------------------------------- #
@dataclass
class BehaviorDescriptor:
    """Definition of one behavior-descriptor dimension."""

    name: str
    min_value: float = 0.0
    max_value: float = 1.0
    bins: int = 20
    # Optional reference to a metric path on an individual (e.g. "metrics.latency").
    metric: Optional[str] = None

    def clip(self, value: float) -> float:
        return max(self.min_value, min(self.max_value, float(value)))

    def to_bin(self, value: float) -> int:
        norm = (self.clip(value) - self.min_value) / max(
            self.max_value - self.min_value, 1e-12
        )
        idx = int(norm * (self.bins - 1))
        return max(0, min(self.bins - 1, idx))

    @classmethod
    def from_config(
        cls, name: str, cfg: Any, default_bins: int = 20
    ) -> "BehaviorDescriptor":
        if isinstance(cfg, BehaviorDescriptor):
            return cfg
        if isinstance(cfg, dict):
            return cls(
                name=name,
                min_value=float(cfg.get("min", 0.0)),
                max_value=float(cfg.get("max", 1.0)),
                bins=int(cfg.get("bins", default_bins)),
                metric=cfg.get("metric"),
            )
        if isinstance(cfg, (list, tuple)):
            if len(cfg) >= 3 and isinstance(cfg[0], str):
                # ("name", lo, hi, bins?)
                parts = cfg[1:]
            else:
                parts = cfg
            lo, hi = parts[0], parts[1]
            bins = parts[2] if len(parts) > 2 else default_bins
            return cls(name=name, min_value=float(lo), max_value=float(hi), bins=int(bins))
        raise TypeError(f"Cannot build BehaviorDescriptor from {cfg!r}")


class BehaviorSpace:
    """n-dimensional behaviour space with grid coordinate mapping."""

    def __init__(
        self,
        descriptors: Sequence[Union[BehaviorDescriptor, Dict[str, Any], Any]],
        default_bins: int = 20,
    ) -> None:
        self.descriptors: List[BehaviorDescriptor] = [
            (
                d
                if isinstance(d, BehaviorDescriptor)
                else BehaviorDescriptor.from_config(f"dim{i}", d, default_bins)
            )
            for i, d in enumerate(descriptors)
        ]
        self.ndim = len(self.descriptors)
        if self.ndim == 0:
            raise ValueError("BehaviorSpace requires at least one descriptor")

    @property
    def grid_resolution(self) -> Tuple[int, ...]:
        return tuple(d.bins for d in self.descriptors)

    @property
    def total_cells(self) -> int:
        prod = 1
        for d in self.descriptors:
            prod *= d.bins
        return prod

    def __len__(self) -> int:
        return self.ndim

    def to_coordinates(self, descriptor: Sequence[float]) -> Tuple[int, ...]:
        """Map a behavior descriptor vector to integer grid coordinates."""
        if len(descriptor) != self.ndim:
            raise ValueError(
                f"descriptor has {len(descriptor)} dims but space has {self.ndim}"
            )
        coords = []
        for value, desc in zip(descriptor, self.descriptors):
            coords.append(desc.to_bin(float(value)))
        return tuple(coords)

    def is_valid_coords(self, coords: Sequence[int]) -> bool:
        if len(coords) != self.ndim:
            return False
        return all(
            0 <= int(c) < desc.bins for c, desc in zip(coords, self.descriptors)
        )

    def get_neighborhood_offsets(self) -> List[Tuple[int, ...]]:
        """All non-zero offsets within a 3x3x... (Moore) neighborhood."""
        offsets: List[Tuple[int, ...]] = []
        for i in range(3 ** self.ndim):
            offset: List[int] = []
            temp = i
            for _ in range(self.ndim):
                offset.append((temp % 3) - 1)
                temp //= 3
            if any(o != 0 for o in offset):
                offsets.append(tuple(offset))
        return offsets

    def count_nearby_empty_cells(
        self, coords: Tuple[int, ...], occupied: Sequence[Tuple[int, ...]]
    ) -> int:
        occupied_set = set(occupied)
        count = 0
        for offset in self.get_neighborhood_offsets():
            neighbor = tuple(c + o for c, o in zip(coords, offset))
            if self.is_valid_coords(neighbor) and neighbor not in occupied_set:
                count += 1
        return count


# --------------------------------------------------------------------------- #
# Archive
# --------------------------------------------------------------------------- #
@dataclass
class Elite:
    """A single solution stored in a MAP-Elites cell."""

    fitness: float
    behavior_descriptor: Tuple[float, ...]
    coordinates: Tuple[int, ...]
    genotype: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "Elite") -> bool:
        return self.fitness < other.fitness


class Grid:
    """Sparse n-dimensional grid of :class:`Elite` objects."""

    def __init__(self, dimensions: BehaviorSpace) -> None:
        self.space = dimensions
        self.cells: Dict[Tuple[int, ...], Elite] = {}

    def __len__(self) -> int:
        return len(self.cells)

    def __contains__(self, coords: Tuple[int, ...]) -> bool:
        return coords in self.cells

    def __iter__(self):
        return iter(self.cells.values())

    def get_cell(self, coords: Tuple[int, ...]) -> Optional[Elite]:
        return self.cells.get(coords)

    def set_cell(self, coords: Tuple[int, ...], elite: Elite) -> None:
        self.cells[coords] = elite

    def occupied_coordinates(self) -> List[Tuple[int, ...]]:
        return list(self.cells.keys())


@dataclass
class MAPElitesConfig:
    """Configuration for :class:`MAPElites`."""

    behavior_descriptors: Sequence[Any] = field(default_factory=list)
    default_bins: int = 20
    selection_strategy: str = "improvement"
    mutation_strategy: str = "gaussian"
    max_generations: int = 100
    initial_population_size: int = 200
    offspring_per_generation: int = 50
    archive_size_limit: Optional[int] = None
    distance_metric: str = "euclidean"

    def normalize_descriptors(self) -> List[BehaviorDescriptor]:
        if not self.behavior_descriptors:
            raise ValueError("MAPElitesConfig.behavior_descriptors is empty")
        return [
            BehaviorDescriptor.from_config(f"dim{i}", d, self.default_bins)
            for i, d in enumerate(self.behavior_descriptors)
        ]


# --------------------------------------------------------------------------- #
# MAP-Elites optimizer
# --------------------------------------------------------------------------- #
class MAPElites:
    """MAP-Elites quality-diversity optimizer.

    Args:
        config: a :class:`MAPElitesConfig`, or keyword overrides accepted by it.
        behavior_descriptors: convenience alias for
            ``config.behavior_descriptors`` (list of descriptor specs, see
            :meth:`BehaviorDescriptor.from_config`).
        selection_strategy: ``random`` / ``improvement`` (curiosity) /
            ``novelty`` / ``fitness`` / ``tournament``.
        mutation_strategy: ``gaussian`` / ``uniform`` / ``iso_line_dd`` /
            or a callable ``f(elite) -> child_descriptor``.
        max_generations: number of evolution steps.
        initial_population_size / offspring_per_generation: population sizes.
        archive_size_limit: cap on stored elites (oldest evicted) or ``None``.
        distance_metric: metric for the ``novelty`` selection strategy.
    """

    def __init__(
        self,
        config: Optional[MAPElitesConfig] = None,
        behavior_descriptors: Optional[Sequence[Any]] = None,
        selection_strategy: str = "improvement",
        mutation_strategy: str = "gaussian",
        max_generations: int = 100,
        initial_population_size: int = 200,
        offspring_per_generation: int = 50,
        archive_size_limit: Optional[int] = None,
        distance_metric: str = "euclidean",
        random_state: Optional[Union[int, random.Random]] = None,
    ) -> None:
        if config is None:
            config = MAPElitesConfig()
        if behavior_descriptors is not None:
            config.behavior_descriptors = list(behavior_descriptors)
        if config.selection_strategy == "improvement" and selection_strategy != "improvement":
            config.selection_strategy = selection_strategy
        else:
            config.selection_strategy = selection_strategy
        config.mutation_strategy = mutation_strategy
        config.max_generations = max_generations
        config.initial_population_size = initial_population_size
        config.offspring_per_generation = offspring_per_generation
        config.archive_size_limit = archive_size_limit
        config.distance_metric = distance_metric

        self.config = config
        descriptors = config.normalize_descriptors()
        self.behavior_space = BehaviorSpace(descriptors, config.default_bins)
        self.grid = Grid(self.behavior_space)

        self.rng = (
            random_state
            if isinstance(random_state, random.Random)
            else random.Random(random_state)
        )
        self.generation = 0
        self.history: List[Dict[str, float]] = []

        self._mutation_fn: Optional[Callable[[Elite], Any]] = None
        if not isinstance(mutation_strategy, str):
            self._mutation_fn = mutation_strategy

    # -- archive management ---------------------------------------------- #
    def initialize_archive(self) -> None:
        """(Re)create an empty n-dimensional grid archive."""
        self.grid = Grid(self.behavior_space)
        self.generation = 0
        self.history = []

    def descriptor_to_coordinates(
        self, descriptor: Sequence[float]
    ) -> Tuple[int, ...]:
        return self.behavior_space.to_coordinates(descriptor)

    def add(
        self,
        fitness: float,
        behavior_descriptor: Sequence[float],
        genotype: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Insert a solution; replaces the cell elite only if fitter.

        Returns True when the solution became the new elite of its cell.
        """
        descriptor = tuple(float(v) for v in behavior_descriptor)
        coords = self.descriptor_to_coordinates(descriptor)
        elite = Elite(
            fitness=float(fitness),
            behavior_descriptor=descriptor,
            coordinates=coords,
            genotype=genotype,
            metadata=metadata or {},
        )
        current = self.grid.get_cell(coords)
        if current is None or elite.fitness > current.fitness:
            self.grid.set_cell(coords, elite)
            self._enforce_size_limit()
            return True
        return False

    def add_many(self, solutions: Sequence[Tuple[float, Sequence[float], Any]]) -> int:
        """Batch insert ``(fitness, descriptor, genotype)`` triples.

        Returns the number of cells whose elite changed.
        """
        updated = 0
        for fitness, descriptor, genotype in solutions:
            if self.add(fitness, descriptor_for_genotype(genotype, descriptor), genotype):
                updated += 1
        return updated

    def _enforce_size_limit(self) -> None:
        limit = self.config.archive_size_limit
        if limit is not None and len(self.grid) > limit:
            # Evict the lowest-fitness elites (MAP-Elites rarely needs this).
            ordered = sorted(
                self.grid.cells.values(), key=lambda e: e.fitness
            )
            excess = len(ordered) - limit
            for elite in ordered[:excess]:
                del self.grid.cells[elite.coordinates]

    def update_archive(self, individual: Any) -> bool:
        """Adapter for ``individual.fitness`` / ``individual.behavior_descriptor``
        style objects (mirrors the spec's ``update_archive``)."""
        fitness = getattr(individual, "fitness", None)
        descriptor = getattr(individual, "behavior_descriptor", None)
        if fitness is None or descriptor is None:
            raise ValueError("individual needs .fitness and .behavior_descriptor")
        return self.add(fitness, descriptor, genotype=individual)

    def __len__(self) -> int:
        return len(self.grid)

    def __contains__(self, coords: Tuple[int, ...]) -> bool:
        return coords in self.grid

    def __iter__(self):
        return iter(self.grid.cells.values())

    def elites(self) -> List[Elite]:
        return list(self.grid.cells.values())

    def best(self) -> Optional[Elite]:
        if not self.grid.cells:
            return None
        return max(self.grid.cells.values(), key=lambda e: e.fitness)

    def get(self, coords: Tuple[int, ...]) -> Optional[Elite]:
        return self.grid.get_cell(coords)

    def coverage(self) -> float:
        """Fraction of grid cells currently occupied (0.0 - 1.0)."""
        return len(self.grid) / float(self.behavior_space.total_cells)

    def qd_score(self) -> float:
        """Sum of fitness across all occupied cells."""
        return sum(e.fitness for e in self.grid.cells.values())

    def archive_statistics(self) -> Dict[str, float]:
        return {
            "covered_cells": float(len(self.grid)),
            "total_cells": float(self.behavior_space.total_cells),
            "coverage": self.coverage(),
            "qd_score": self.qd_score(),
            "best_fitness": self.best().fitness if self.best() else 0.0,
            "generation": float(self.generation),
        }

    # -- selection helpers ---------------------------------------------- #
    def select_parent(self) -> Elite:
        strategy = (self.config.selection_strategy or "improvement").lower()
        if strategy == "random":
            return self._select_random_parent()
        if strategy == "novelty":
            return self._select_novelty_parent()
        if strategy == "fitness":
            return self._select_fitness_parent()
        if strategy == "tournament":
            return self._select_tournament_parent()
        # default: improvement / curiosity
        return self._select_improvement_parent()

    def _select_random_parent(self) -> Elite:
        occupied = self.grid.occupied_coordinates()
        if not occupied:
            raise ValueError("No occupied cells in archive")
        return self.grid.get_cell(self.rng.choice(occupied))

    def _select_improvement_parent(self) -> Elite:
        occupied = self.grid.occupied_coordinates()
        if not occupied:
            raise ValueError("No occupied cells in archive")
        candidates = [
            (self.grid.get_cell(c), self.behavior_space.count_nearby_empty_cells(c, occupied))
            for c in occupied
        ]
        candidates.sort(key=lambda t: t[1], reverse=True)
        # Among the most promising, add a little randomness to avoid cycling.
        top_k = candidates[: max(1, len(candidates) // 5)]
        return self.rng.choice(top_k)[0]

    def _select_novelty_parent(self) -> Elite:
        elites = self.elites()
        occupied = self.grid.occupied_coordinates()
        best = None
        best_score = -1.0
        for elite, coords in zip(elites, occupied):
            nearest = float("inf")
            for other in elites:
                if other is elite:
                    continue
                d = float(
                    np.linalg.norm(
                        np.asarray(elite.behavior_descriptor)
                        - np.asarray(other.behavior_descriptor)
                    )
                )
                nearest = min(nearest, d)
            score = np.inf if not np.isfinite(nearest) else nearest
            if score > best_score:
                best_score = score
                best = elite
        return best if best is not None else self._select_random_parent()

    def _select_fitness_parent(self) -> Elite:
        elites = self.elites()
        if not elites:
            raise ValueError("No occupied cells in archive")
        weights = [max(e.fitness, 1e-9) for e in elites]
        total = sum(weights)
        return self.rng.choices(elites, weights=[w / total for w in weights], k=1)[0]

    def _select_tournament_parent(self, k: int = 3) -> Elite:
        elites = self.elites()
        if len(elites) <= k:
            return self._select_fitness_parent()
        pool = self.rng.sample(elites, k)
        return max(pool, key=lambda e: e.fitness)

    # -- variation ------------------------------------------------------- #
    def mutate(
        self, parent: Elite, generation: Optional[int] = None
    ) -> Tuple[float, Any]:
        """Return ``(child_descriptor, child_genotype)`` for a parent elite."""
        if self._mutation_fn is not None:
            child_genotype = self._mutation_fn(parent)
            child_descriptor = extract_descriptor(child_genotype, self.behavior_space)
            return tuple(float(v) for v in child_descriptor), child_genotype

        descriptor = np.asarray(parent.behavior_descriptor, dtype=float)
        rng = self.rng
        strategy = self.config.mutation_strategy.lower()

        if strategy == "uniform":
            span = np.array(
                [d.max_value - d.min_value for d in self.behavior_space.descriptors]
            )
            child = descriptor + rng.random(self.behavior_space.ndim) * span * 0.1
        elif strategy == "iso_line_dd":
            # Iso+Line DD: interpolate two parents, then perturb isotropically.
            second = self._select_random_parent()
            other = np.asarray(second.behavior_descriptor, dtype=float)
            iso = 0.5 * (descriptor + other)
            sigma_iso = max(
                float(np.linalg.norm(other - descriptor)) * 0.1, 1e-3
            )
            line = iso + rng.gauss(0.0, sigma_iso) * (descriptor - other)
            child = line
        else:  # gaussian (default)
            sigma = 0.1 * np.array(
                [max(d.max_value - d.min_value, 1e-6) for d in self.behavior_space.descriptors]
            )
            child = descriptor + rng.gauss(0.0, 1.0) * sigma

        child = np.clip(
            child,
            [d.min_value for d in self.behavior_space.descriptors],
            [d.max_value for d in self.behavior_space.descriptors],
        )
        child_descriptor = tuple(float(v) for v in child)
        child_genotype = (
            self._clone_genotype(parent.genotype, child_descriptor)
            if parent.genotype is not None
            else child_descriptor
        )
        return child_descriptor, child_genotype

    def _clone_genotype(self, genotype: Any, descriptor: Sequence[float]) -> Any:
        # Genotypes are opaque; return as-is so callers can attach descriptors.
        return genotype

    # -- main loop ------------------------------------------------------- #
    def evolve(
        self,
        evaluate: Callable[[Any], Tuple[float, Sequence[float]]],
        initial_population: Optional[Sequence[Any]] = None,
        generation_callback: Optional[Callable[["MAPElites"], None]] = None,
    ) -> Dict[str, Any]:
        """Run MAP-Elites until ``max_generations`` is reached.

        Args:
            evaluate: maps a genotype to ``(fitness, behavior_descriptor)``.
            initial_population: optional seed genotypes (evaluated first).
            generation_callback: optional hook invoked after each generation.

        Returns:
            A result dict with ``archive``, ``coverage``, ``qd_score`` and the
            ``best`` elite.
        """
        self.initialize_archive()

        seed = list(initial_population or [])
        if seed:
            for genotype in seed[: self.config.initial_population_size]:
                fitness, descriptor = evaluate(genotype)
                self.add(fitness, descriptor, genotype=genotype)

        for generation in range(self.config.max_generations):
            self.generation = generation
            for _ in range(self.config.offspring_per_generation):
                if len(self.grid) == 0:
                    # Cold start: random genotype when no archive yet.
                    fitness, descriptor = evaluate(None)
                    self.add(fitness, descriptor, genotype=descriptor)
                    continue
                parent = self.select_parent()
                child_descriptor, child_genotype = self.mutate(parent, generation)
                fitness, evaluated_descriptor = evaluate(child_genotype)
                # The evaluator may refine the descriptor; prefer it.
                descriptor = (
                    evaluated_descriptor
                    if evaluated_descriptor is not None
                    else child_descriptor
                )
                self.add(fitness, descriptor, genotype=child_genotype)

            self.history.append(self.archive_statistics())
            if generation_callback is not None:
                generation_callback(self)

        return {
            "archive": self.grid.cells,
            "coverage": self.coverage(),
            "qd_score": self.qd_score(),
            "best": self.best(),
            "generations": self.config.max_generations,
            "history": self.history,
        }

    def ask(self) -> Optional[Any]:
        """Request a parent genotype to evaluate (interactive / ask-tell API)."""
        if len(self.grid) == 0:
            return None
        return self.select_parent().genotype

    def tell(
        self,
        genotype: Any,
        fitness: float,
        behavior_descriptor: Sequence[float],
    ) -> bool:
        """Report an evaluated individual (interactive / ask-tell API)."""
        return self.add(fitness, behavior_descriptor, genotype=genotype)


# --------------------------------------------------------------------------- #
# Descriptor extraction helpers (for opaque genotypes)
# --------------------------------------------------------------------------- #
def extract_descriptor(genotype: Any, space: BehaviorSpace) -> Tuple[float, ...]:
    """Pull a behavior descriptor from a genotype dict/object if possible."""
    if isinstance(genotype, (list, tuple)) and len(genotype) == space.ndim:
        try:
            return tuple(float(v) for v in genotype)
        except (TypeError, ValueError):
            pass
    if isinstance(genotype, dict) and "behavior_descriptor" in genotype:
        return tuple(float(v) for v in genotype["behavior_descriptor"])
    if hasattr(genotype, "behavior_descriptor"):
        bd = getattr(genotype, "behavior_descriptor")
        return tuple(float(v) for v in bd)
    # Fall back to random descriptor inside bounds.
    rng = random.Random(id(genotype))
    return tuple(
        rng.uniform(d.min_value, d.max_value) for d in space.descriptors
    )


def descriptor_for_genotype(genotype: Any, descriptor: Sequence[float]) -> Sequence[float]:
    """Use descriptor as-is, or attempt to set it on a mutable genotype."""
    if isinstance(genotype, dict) and "behavior_descriptor" not in genotype:
        genotype["behavior_descriptor"] = list(descriptor)
    return descriptor


# --------------------------------------------------------------------------- #
# Selection-contract wrapper (mirrors nsga3_selection / novelty_selection)
# --------------------------------------------------------------------------- #
def map_elites_selection(
    objectives: np.ndarray,
    population_size: int,
    dimensions: Optional[int] = None,
    random_state: Optional[int] = None,
) -> List[int]:
    """Discrete ``select_mo`` wrapper: keep the best elite per behavior cell.

    ``objectives`` is interpreted as ``[fitness, behavior_1, ..., behavior_n]``.
    Each individual is binned into a MAP-Elites cell using the (fitness,
    behavior) columns; only the fitter individual of each cell survives, then
    the fittest ``population_size`` cells are returned. When no behavior
    dimensions are supplied the full vector is treated as the descriptor.
    """
    matrix = np.asarray(objectives, dtype=float)
    if matrix.size == 0:
        return []
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    n = matrix.shape[0]

    if dimensions is None:
        # Heuristic: last 2 columns are behaviors, first is fitness.
        n_behavior = min(2, matrix.shape[1] - 1)
    else:
        n_behavior = int(dimensions)
    n_behavior = min(n_behavior, max(0, matrix.shape[1] - 1))

    rng = random.Random(random_state)
    # Discretise behavior columns into bins.
    if n_behavior == 0:
        order = sorted(range(n), key=lambda i: -float(matrix[i, 0]))
        return order[:population_size]

    behavior_cols = matrix[:, 1 : 1 + n_behavior]
    bins_per_col = 12
    cell_keys = {}
    for i in range(n):
        coords = []
        for j in range(n_behavior):
            lo = float(np.min(behavior_cols[:, j]))
            hi = float(np.max(behavior_cols[:, j]))
            span = hi - lo if hi > lo else 1.0
            coords.append(int((behavior_cols[i, j] - lo) / span * (bins_per_col - 1)))
        cell_keys[i] = tuple(coords)

    best_in_cell: Dict[Tuple[int, ...], int] = {}
    for i in range(n):
        key = cell_keys[i]
        if key not in best_in_cell or float(matrix[i, 0]) > float(
            matrix[best_in_cell[key], 0]
        ):
            best_in_cell[key] = i

    survivors = list(best_in_cell.values())
    survivors.sort(key=lambda i: -float(matrix[i, 0]))
    return survivors[:population_size]
