"""leanaide_evolution - evolutionary search over Lean proof strategies.

Flat-script module providing ``Tactic`` (shared with ``leanaide_mcts``), a
``LeanProof`` data structure and a compact genetic/evolutionary engine used by
``mcts_evolved_policies`` / ``mcts_evolutionary_nodes`` via
``from leanaide_evolution import Tactic, LeanProof, ...``.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from leanaide_mcts import Tactic  # noqa: E402  (shared definition)


class MutationType(str, Enum):
    POINT = "point"
    INSERT = "insert"
    DELETE = "delete"
    SWAP = "swap"


class CrossoverMethod(str, Enum):
    ONE_POINT = "one_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"


class SelectionMethod(str, Enum):
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"


@dataclass
class LeanProof:
    """A candidate proof: an ordered sequence of tactics plus a fitness score."""

    tactics: List[Tactic] = field(default_factory=list)
    fitness: float = 0.0
    theorem: str = ""
    proven: bool = False

    def to_string(self) -> str:
        return "\n".join(t.to_string() for t in self.tactics)

    @classmethod
    def from_string(cls, text: str, theorem: str = "") -> "LeanProof":
        tactics = [Tactic.parse(line) for line in text.splitlines() if line.strip()]
        return cls(tactics=tactics, theorem=theorem)


@dataclass
class EvolutionResult:
    best: Optional[LeanProof]
    generations: int = 0
    history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_tactics": [t.to_string() for t in (self.best.tactics if self.best else [])],
            "generations": self.generations,
            "history": list(self.history),
        }


@dataclass
class Population:
    individuals: List[LeanProof] = field(default_factory=list)

    def best(self) -> Optional[LeanProof]:
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda p: p.fitness)

    def add(self, proof: LeanProof) -> None:
        self.individuals.append(proof)


@dataclass
class FitnessFunction:
    func: Callable[[LeanProof], float]

    def evaluate(self, proof: LeanProof) -> float:
        proof.fitness = float(self.func(proof))
        return proof.fitness


class GeneticOperator:
    """Base genetic operator (strategy pattern)."""


class MutationOperator(GeneticOperator):
    def __init__(self, mutation_rate: float = 0.2, seed: int = 0,
                 vocabulary: Optional[List[str]] = None):
        self.mutation_rate = mutation_rate
        self._rng = random.Random(seed)
        self.vocabulary = vocabulary or ["simp", "intro", "rw", "exact", "apply", "induction"]

    def mutate(self, proof: LeanProof) -> LeanProof:
        new_tactics = list(proof.tactics)
        for i, t in enumerate(new_tactics):
            if self._rng.random() < self.mutation_rate:
                kind = self._rng.choice(list(MutationType))
                if kind == MutationType.POINT:
                    new_tactics[i] = Tactic(self._rng.choice(self.vocabulary))
                elif kind == MutationType.INSERT and new_tactics:
                    new_tactics.insert(i, Tactic(self._rng.choice(self.vocabulary)))
                elif kind == MutationType.DELETE and len(new_tactics) > 1:
                    new_tactics.pop(i)
                elif kind == MutationType.SWAP and len(new_tactics) > 1:
                    j = self._rng.randrange(len(new_tactics))
                    new_tactics[i], new_tactics[j] = new_tactics[j], new_tactics[i]
        return LeanProof(tactics=new_tactics, theorem=proof.theorem)


class CrossoverOperator(GeneticOperator):
    def __init__(self, method: CrossoverMethod = CrossoverMethod.ONE_POINT):
        self.method = method

    def crossover(self, a: LeanProof, b: LeanProof) -> LeanProof:
        ta, tb = a.tactics, b.tactics
        if not ta or not tb:
            return LeanProof(tactics=list(ta or tb), theorem=a.theorem)
        if self.method == CrossoverMethod.ONE_POINT:
            k = min(len(ta), len(tb)) // 2
            child = ta[:k] + tb[k:]
        elif self.method == CrossoverMethod.TWO_POINT:
            k1, k2 = sorted(self._pair(len(ta), len(tb)))
            child = ta[:k1] + tb[k1:k2] + ta[k2:]
        else:
            child = [self._pick(x, y) for x, y in zip(ta, tb)]
        return LeanProof(tactics=child, theorem=a.theorem)

    @staticmethod
    def _pair(n: int, m: int):
        a, b = sorted([max(1, n // 3), max(1, m // 3)])
        return a, min(max(a, b), max(n, m))

    @staticmethod
    def _pick(x, y):
        return x


class SelectionOperator(GeneticOperator):
    def __init__(self, method: SelectionMethod = SelectionMethod.TOURNAMENT, seed: int = 0):
        self.method = method
        self._rng = random.Random(seed)

    def select(self, population: Population, k: int) -> List[LeanProof]:
        inds = population.individuals
        if not inds:
            return []
        if self.method == SelectionMethod.TOURNAMENT:
            out = []
            for _ in range(k):
                contenders = self._rng.sample(inds, min(3, len(inds)))
                out.append(max(contenders, key=lambda p: p.fitness))
            return out
        inds_sorted = sorted(inds, key=lambda p: p.fitness, reverse=True)
        return inds_sorted[:k]


@dataclass
class LeanProofStrategy:
    mutation: MutationOperator
    crossover: CrossoverOperator
    selection: SelectionOperator
    population_size: int = 20
    generations: int = 10


class LeanProofEvolutionEngine:
    """Compact evolutionary search over tactic sequences."""

    def __init__(self, strategy: Optional[LeanProofStrategy] = None, seed: int = 0):
        self._rng = random.Random(seed)
        if strategy is None:
            strategy = LeanProofStrategy(
                mutation=MutationOperator(seed=seed),
                crossover=CrossoverOperator(),
                selection=SelectionOperator(seed=seed),
            )
        self.strategy = strategy

    def evolve(self, seed_proof: LeanProof, fitness: FitnessFunction,
               generations: Optional[int] = None) -> EvolutionResult:
        gens = generations or self.strategy.generations
        pop = Population([seed_proof] + [
            LeanProof(tactics=list(seed_proof.tactics), theorem=seed_proof.theorem)
            for _ in range(self.strategy.population_size - 1)
        ])
        history: List[float] = []
        for g in range(gens):
            scored = [fitness.evaluate(p) for p in pop.individuals]
            history.append(max(scored) if scored else 0.0)
            parents = self.strategy.selection.select(pop, 2)
            if len(parents) < 2:
                break
            child = self.strategy.crossover.crossover(parents[0], parents[1])
            child = self.strategy.mutation.mutate(child)
            # elitism: replace weakest
            weakest = min(pop.individuals, key=lambda p: p.fitness)
            pop.individuals.remove(weakest)
            pop.add(child)
        return EvolutionResult(best=pop.best(), generations=gens, history=history)
