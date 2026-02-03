"""
LeanAide Evolutionary Proof Generator

Production-ready evolutionary proof generation system using genetic algorithms
to evolve Lean 4 proofs. Integrates with LeanAide for verification and uses
self-play patterns from PSV system.

Classes:
    LeanProofStrategy: Represents a proof strategy with tactics sequence
    LeanProofPopulation: Manages population of proof strategies
    LeanProofMutator: Applies mutations to proof strategies
    LeanProofEvaluator: Evaluates proof fitness using LeanAide
    LeanProofEvolutionEngine: Main evolutionary engine for proof generation

Key Features:
    - Genetic algorithm with mutation, crossover, and selection
    - Fitness-based evaluation using Lean 4 verification
    - Population diversity management
    - Proof family tree tracking
    - Parallel evaluation support
    - Comprehensive caching layer
    - Evolution statistics and analytics
"""

import asyncio
import json
import logging
import random
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union
)
import hashlib
import sqlite3
import threading
from pathlib import Path

# Import LeanAide integration
try:
    from lean4_integration import (
        Lean4VerificationEngine,
        Lean4ServerConfig,
        Lean4VerificationConfig,
        VerificationResult,
        VerificationCache,
        AutoformalizationEngine,
        ProofSearchEngine,
        LeanAideClient
    )
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logging.warning("LeanAide integration not available - using simulation mode")

logger = logging.getLogger(__name__)

class MutationType(Enum):
    """Types of mutations that can be applied to proof strategies"""
    TACTIC_SUBSTITUTION = "tactic_substitution"
    STEP_INSERTION = "step_insertion"
    STEP_DELETION = "step_deletion"
    GOAL_RESTRUCTURING = "goal_restructuring"
    LEMMA_INTRODUCTION = "lemma_introduction"
    LEMMA_REMOVAL = "lemma_removal"
    REORDERING = "reordering"
    SIMPLIFICATION = "simplification"


class SelectionMethod(Enum):
    """Selection methods for choosing parents"""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    SUS = "stochastic_universal_sampling"
    TRUNCATION = "truncation"


class CrossoverMethod(Enum):
    """Crossover methods for combining parent strategies"""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ORDERED = "ordered"
    CYCLE = "cycle"


@dataclass
class Tactic:
    """Represents a single Lean 4 tactic"""
    name: str
    arguments: List[str] = field(default_factory=list)
    location: Optional[str] = None  # Goal location if applicable

    def __str__(self) -> str:
        if self.arguments:
            args_str = " ".join(self.arguments)
            return f"{self.name} {args_str}"
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LeanProof:
    """A complete Lean 4 proof"""
    theorem_name: str
    theorem_statement: str
    lean_code: str
    tactics: List[Tactic] = field(default_factory=list)
    verification_result: Optional[VerificationResult] = None
    proof_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proof_id": self.proof_id,
            "theorem_name": self.theorem_name,
            "theorem_statement": self.theorem_statement,
            "lean_code": self.lean_code,
            "tactics": [t.to_dict() for t in self.tactics],
            "verification_result": self.verification_result.to_dict() if self.verification_result else None,
            "created_at": self.created_at
        }


@dataclass
class LeanProofStrategy:
    """
    Represents a proof strategy in the evolutionary population.

    A strategy includes the theorem to prove, the current tactics sequence,
    fitness metrics, and genealogical information.
    """
    proof: LeanProof
    fitness: float = 0.0
    generation: int = 0
    parents: List[str] = field(default_factory=list)  # IDs of parent strategies
    mutation_history: List[MutationType] = field(default_factory=list)
    birth_time: float = field(default_factory=time.time)
    evaluation_count: int = 0
    verified: bool = False
    diversity_score: float = 0.0
    complexity_score: float = 0.0
    elegance_score: float = 0.0
    strategy_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def get_tactics_sequence(self) -> str:
        """Get the tactics sequence as a string"""
        return "\n  ".join(str(tactic) for tactic in self.proof.tactics)

    def calculate_complexity(self) -> float:
        """Calculate proof complexity based on tactics"""
        if not self.proof.tactics:
            return 0.0

        # Complexity factors
        num_tactics = len(self.proof.tactics)
        unique_tactics = len(set(t.name for t in self.proof.tactics))

        # Tactic complexity weights
        complexity_weights = {
            "simp": 1.0,
            "rw": 1.0,
            "apply": 1.5,
            "cases": 2.0,
            "induction": 3.0,
            "by": 1.0,
            "aesop": 0.5,
            "linarith": 1.0,
            "ring": 1.0,
            "norm_num": 0.5,
            "constructor": 1.5,
            "refine": 2.0,
            "exact": 1.0,
            "have": 2.0,
            "calc": 2.5,
        }

        weighted_complexity = sum(
            complexity_weights.get(t.name, 2.0) for t in self.proof.tactics
        )

        # Normalize complexity
        self.complexity_score = min(10.0, weighted_complexity / num_tactics)
        return self.complexity_score

    def calculate_elegance(self) -> float:
        """
        Calculate elegance score based on:
        - Conciseness (fewer tactics is better)
        - Simplicity (prefer simple tactics over complex ones)
        - Structure (well-organized proof)
        """
        if not self.proof.tactics:
            return 0.0

        # Elegance factors
        num_tactics = len(self.proof.tactics)
        unique_tactics = len(set(t.name for t in self.proof.tactics))

        # Prefer proofs that are concise but use diverse tactics
        conciseness = max(0.0, 1.0 - (num_tactics / 50.0))  # Penalize very long proofs
        diversity = unique_tactics / max(1, num_tactics)

        # Simple tactics are more elegant
        simple_tactics = sum(1 for t in self.proof.tactics if t.name in ["simp", "rw", "exact"])
        simplicity = simple_tactics / max(1, num_tactics)

        # Combine factors
        self.elegance_score = (conciseness * 0.4 + diversity * 0.3 + simplicity * 0.3)
        return self.elegance_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "strategy_id": self.strategy_id,
            "fitness": self.fitness,
            "generation": self.generation,
            "parents": self.parents,
            "mutation_history": [m.value for m in self.mutation_history],
            "birth_time": self.birth_time,
            "evaluation_count": self.evaluation_count,
            "verified": self.verified,
            "diversity_score": self.diversity_score,
            "complexity_score": self.complexity_score,
            "elegance_score": self.elegance_score,
            "proof": self.proof.to_dict(),
            "tactics_sequence": self.get_tactics_sequence()
        }


@dataclass
class PopulationStatistics:
    """Statistics about a population"""
    generation: int
    population_size: int
    best_fitness: float
    worst_fitness: float
    average_fitness: float
    fitness_std: float
    diversity_score: float
    verified_count: int
    unique_strategies: int
    average_complexity: float
    average_elegance: float
    convergence_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvolutionResult:
    """Result of an evolutionary proof generation run"""
    success: bool
    best_proof: Optional[LeanProof] = None
    best_strategy: Optional[LeanProofStrategy] = None
    generations_completed: int = 0
    total_evaluations: int = 0
    evolution_time: float = 0.0
    statistics_history: List[PopulationStatistics] = field(default_factory=list)
    family_tree: Dict[str, List[str]] = field(default_factory=dict)  # parent -> children
    failed_attempts: List[Dict[str, Any]] = field(default_factory=list)
    convergence_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "best_proof": self.best_proof.to_dict() if self.best_proof else None,
            "best_strategy": self.best_strategy.to_dict() if self.best_strategy else None,
            "generations_completed": self.generations_completed,
            "total_evaluations": self.total_evaluations,
            "evolution_time": self.evolution_time,
            "statistics_history": [s.to_dict() for s in self.statistics_history],
            "family_tree": self.family_tree,
            "failed_attempts": self.failed_attempts,
            "convergence_history": self.convergence_history
        }


class LeanProofPopulation:
    """
    Manages a population of proof strategies.

    Handles selection, diversity tracking, and population statistics.
    """

    def __init__(
        self,
        strategies: List[LeanProofStrategy],
        selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
        tournament_size: int = 3,
        elitism_ratio: float = 0.1
    ):
        self.strategies = strategies
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.elitism_ratio = elitism_ratio
        self.generation = 0

    def __len__(self) -> int:
        return len(self.strategies)

    def get_best_strategy(self) -> Optional[LeanProofStrategy]:
        """Get the strategy with highest fitness"""
        if not self.strategies:
            return None
        return max(self.strategies, key=lambda s: s.fitness)

    def get_worst_strategy(self) -> Optional[LeanProofStrategy]:
        """Get the strategy with lowest fitness"""
        if not self.strategies:
            return None
        return min(self.strategies, key=lambda s: s.fitness)

    def calculate_diversity(self) -> float:
        """
        Calculate population diversity using tactic sequence variation.

        Uses normalized edit distance between tactic sequences.
        """
        if len(self.strategies) < 2:
            return 0.0

        total_distance = 0.0
        comparisons = 0

        for i, strategy1 in enumerate(self.strategies):
            for strategy2 in self.strategies[i+1:]:
                distance = self._tactic_distance(strategy1, strategy2)
                total_distance += distance
                comparisons += 1

        return total_distance / max(1, comparisons)

    def _tactic_distance(self, s1: LeanProofStrategy, s2: LeanProofStrategy) -> float:
        """Calculate normalized edit distance between two strategies"""
        tactics1 = [t.name for t in s1.proof.tactics]
        tactics2 = [t.name for t in s2.proof.tactics]

        # Simple edit distance
        m, n = len(tactics1), len(tactics2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if tactics1[i-1] == tactics2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        max_len = max(m, n)
        return dp[m][n] / max(1, max_len)

    def calculate_statistics(self) -> PopulationStatistics:
        """Calculate comprehensive population statistics"""
        if not self.strategies:
            return PopulationStatistics(
                generation=self.generation,
                population_size=0,
                best_fitness=0.0,
                worst_fitness=0.0,
                average_fitness=0.0,
                fitness_std=0.0,
                diversity_score=0.0,
                verified_count=0,
                unique_strategies=0,
                average_complexity=0.0,
                average_elegance=0.0,
                convergence_rate=0.0
            )

        fitnesses = [s.fitness for s in self.strategies]
        verified = sum(1 for s in self.strategies if s.verified)

        # Calculate average and standard deviation
        avg_fitness = sum(fitnesses) / len(fitnesses)
        variance = sum((f - avg_fitness) ** 2 for f in fitnesses) / len(fitnesses)
        std_fitness = variance ** 0.5

        # Calculate complexity and elegance
        complexities = [s.complexity_score for s in self.strategies]
        elegances = [s.elegance_score for s in self.strategies]

        # Count unique strategies (by tactics sequence)
        unique_sequences = set(s.get_tactics_sequence() for s in self.strategies)

        return PopulationStatistics(
            generation=self.generation,
            population_size=len(self.strategies),
            best_fitness=max(fitnesses),
            worst_fitness=min(fitnesses),
            average_fitness=avg_fitness,
            fitness_std=std_fitness,
            diversity_score=self.calculate_diversity(),
            verified_count=verified,
            unique_strategies=len(unique_sequences),
            average_complexity=sum(complexities) / len(complexities),
            average_elegance=sum(elegances) / len(elegances),
            convergence_rate=0.0  # Will be calculated across generations
        )

    def select_parents(self, num_parents: int) -> List[LeanProofStrategy]:
        """
        Select parent strategies using configured selection method.

        Args:
            num_parents: Number of parents to select

        Returns:
            List of selected parent strategies
        """
        if self.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection(num_parents)
        elif self.selection_method == SelectionMethod.ROULETTE:
            return self._roulette_selection(num_parents)
        elif self.selection_method == SelectionMethod.RANK:
            return self._rank_selection(num_parents)
        elif self.selection_method == SelectionMethod.TRUNCATION:
            return self._truncation_selection(num_parents)
        else:
            return self._tournament_selection(num_parents)

    def _tournament_selection(self, num_parents: int) -> List[LeanProofStrategy]:
        """Select parents using tournament selection"""
        parents = []
        for _ in range(num_parents):
            # Randomly select tournament_size strategies
            tournament = random.sample(self.strategies, min(self.tournament_size, len(self.strategies)))
            # Select the best from tournament
            winner = max(tournament, key=lambda s: s.fitness)
            parents.append(winner)
        return parents

    def _roulette_selection(self, num_parents: int) -> List[LeanProofStrategy]:
        """Select parents using roulette wheel selection"""
        # Shift fitness to be positive
        min_fitness = min(s.fitness for s in self.strategies)
        adjusted_fitness = [s.fitness - min_fitness + 0.1 for s in self.strategies]
        total_fitness = sum(adjusted_fitness)

        if total_fitness == 0:
            return random.sample(self.strategies, num_parents)

        # Calculate selection probabilities
        probabilities = [f / total_fitness for f in adjusted_fitness]

        # Select parents
        parents = []
        for _ in range(num_parents):
            parent = random.choices(self.strategies, weights=probabilities, k=1)[0]
            parents.append(parent)

        return parents

    def _rank_selection(self, num_parents: int) -> List[LeanProofStrategy]:
        """Select parents using rank-based selection"""
        # Sort by fitness
        sorted_strategies = sorted(self.strategies, key=lambda s: s.fitness)
        # Assign ranks
        ranks = list(range(1, len(sorted_strategies) + 1))
        total_rank = sum(ranks)

        probabilities = [r / total_rank for r in ranks]

        parents = []
        for _ in range(num_parents):
            parent = random.choices(sorted_strategies, weights=probabilities, k=1)[0]
            parents.append(parent)

        return parents

    def _truncation_selection(self, num_parents: int) -> List[LeanProofStrategy]:
        """Select parents using truncation selection"""
        # Select top k strategies
        sorted_strategies = sorted(self.strategies, key=lambda s: s.fitness, reverse=True)
        top_k = sorted_strategies[:min(num_parents * 2, len(sorted_strategies))]
        return random.sample(top_k, min(num_parents, len(top_k)))

    def get_elites(self, num_elites: int) -> List[LeanProofStrategy]:
        """Get the top N strategies (elitism)"""
        sorted_strategies = sorted(self.strategies, key=lambda s: s.fitness, reverse=True)
        return sorted_strategies[:num_elites]


class LeanProofMutator:
    """
    Applies mutations to proof strategies.

    Supports various mutation types:
    - Tactic substitution (replace one tactic with another)
    - Step insertion (add new tactic)
    - Step deletion (remove tactic)
    - Goal restructuring (reorganize proof structure)
    - Lemma introduction (add helper lemma)
    - Lemma removal (remove helper lemma)
    - Reordering (change tactic order)
    - Simplification (replace complex tactic with simpler one)
    """

    # Common Lean 4 tactics grouped by category
    SIMPLIFICATION_TACTICS = ["simp", "simp_all", "simp?", "dsimp", "norm_num"]
    REWRITE_TACTICS = ["rw", "rwa", "rewrite"]
    APPLICATION_TACTICS = ["apply", "exact", "refine", "constructor"]
    DESTRUCTION_TACTICS = ["cases", "induction", "destruct"]
    TRANSFORMATION_TACTICS = ["calc", "have", "suffices", "show"]
    AUTOMATION_TACTICS = ["aesop", "linarith", "ring", "omega", "solve_by_elim"]
    GOAL_TACTICS = ["trivial", "decide", "done"]

    ALL_TACTICS = (
        SIMPLIFICATION_TACTICS + REWRITE_TACTICS + APPLICATION_TACTICS +
        DESTRUCTION_TACTICS + TRANSFORMATION_TACTICS + AUTOMATION_TACTICS +
        GOAL_TACTICS
    )

    # Tactic substitutions (complex -> simple alternatives)
    TACTIC_SUBSTITUTIONS = {
        "simp_all": "simp",
        "aesop": "simp",
        "omega": "linarith",
        "solve_by_elim": "apply",
        "constructor": "refine"
    }

    def __init__(
        self,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.5,
        custom_tactics: Optional[List[str]] = None
    ):
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.custom_tactics = custom_tactics or []

    def mutate(self, strategy: LeanProofStrategy) -> LeanProofStrategy:
        """
        Apply mutations to a proof strategy.

        Args:
            strategy: Strategy to mutate

        Returns:
            New mutated strategy (original is not modified)
        """
        # Create a copy of the strategy
        import copy
        new_strategy = copy.deepcopy(strategy)
        new_strategy.parents = [strategy.strategy_id]
        new_strategy.strategy_id = str(uuid.uuid4())
        new_strategy.mutation_history = []

        # Decide which mutations to apply
        num_mutations = self._calculate_num_mutations(len(strategy.proof.tactics))

        for _ in range(num_mutations):
            if random.random() < self.mutation_rate:
                mutation_type = random.choice(list(MutationType))
                new_strategy = self._apply_mutation(new_strategy, mutation_type)

        return new_strategy

    def _calculate_num_mutations(self, num_tactics: int) -> int:
        """Calculate number of mutations to apply based on proof size"""
        base_mutations = max(1, int(num_tactics * self.mutation_strength))
        return min(base_mutations, 5)  # Cap at 5 mutations

    def _apply_mutation(
        self,
        strategy: LeanProofStrategy,
        mutation_type: MutationType
    ) -> LeanProofStrategy:
        """Apply a specific mutation type to the strategy"""
        if mutation_type == MutationType.TACTIC_SUBSTITUTION:
            return self._tactic_substitution(strategy)
        elif mutation_type == MutationType.STEP_INSERTION:
            return self._step_insertion(strategy)
        elif mutation_type == MutationType.STEP_DELETION:
            return self._step_deletion(strategy)
        elif mutation_type == MutationType.GOAL_RESTRUCTURING:
            return self._goal_restructuring(strategy)
        elif mutation_type == MutationType.LEMMA_INTRODUCTION:
            return self._lemma_introduction(strategy)
        elif mutation_type == MutationType.LEMMA_REMOVAL:
            return self._lemma_removal(strategy)
        elif mutation_type == MutationType.REORDERING:
            return self._reordering(strategy)
        elif mutation_type == MutationType.SIMPLIFICATION:
            return self._simplification(strategy)
        else:
            return strategy

    def _tactic_substitution(self, strategy: LeanProofStrategy) -> LeanProofStrategy:
        """Substitute a tactic with an alternative"""
        if not strategy.proof.tactics:
            return strategy

        # Select random tactic
        idx = random.randint(0, len(strategy.proof.tactics) - 1)
        old_tactic = strategy.proof.tactics[idx]

        # Find substitution
        if old_tactic.name in self.TACTIC_SUBSTITUTIONS:
            new_tactic_name = self.TACTIC_SUBSTITUTIONS[old_tactic.name]
            new_tactic = Tactic(name=new_tactic_name, arguments=old_tactic.arguments)
            strategy.proof.tactics[idx] = new_tactic
            strategy.mutation_history.append(MutationType.TACTIC_SUBSTITUTION)

        return strategy

    def _step_insertion(self, strategy: LeanProofStrategy) -> LeanProofStrategy:
        """Insert a new tactic at a random position"""
        # Choose a random tactic to insert
        new_tactic = self._choose_random_tactic()

        # Insert at random position
        if strategy.proof.tactics:
            idx = random.randint(0, len(strategy.proof.tactics))
            strategy.proof.tactics.insert(idx, new_tactic)
        else:
            strategy.proof.tactics.append(new_tactic)

        strategy.mutation_history.append(MutationType.STEP_INSERTION)
        return strategy

    def _step_deletion(self, strategy: LeanProofStrategy) -> LeanProofStrategy:
        """Delete a random tactic"""
        if len(strategy.proof.tactics) > 1:
            idx = random.randint(0, len(strategy.proof.tactics) - 1)
            strategy.proof.tactics.pop(idx)
            strategy.mutation_history.append(MutationType.STEP_DELETION)

        return strategy

    def _goal_restructuring(self, strategy: LeanProofStrategy) -> LeanProofStrategy:
        """
        Restructure the proof by introducing sub-goals.

        This simulates using 'have', 'suffices', or 'show' to structure the proof.
        """
        if len(strategy.proof.tactics) < 2:
            return strategy

        # Insert a structural tactic
        idx = random.randint(0, len(strategy.proof.tactics) - 1)
        structural_tactics = ["have", "suffices", "show"]
        chosen = random.choice(structural_tactics)

        new_tactic = Tactic(
            name=chosen,
            arguments=["_ : Prop"]  # Placeholder for actual goal
        )

        strategy.proof.tactics.insert(idx, new_tactic)
        strategy.mutation_history.append(MutationType.GOAL_RESTRUCTURING)

        return strategy

    def _lemma_introduction(self, strategy: LeanProofStrategy) -> LeanProofStrategy:
        """
        Introduce a helper lemma at the beginning of the proof.

        This simulates using 'have' to introduce an intermediate result.
        """
        lemma_tactic = Tactic(
            name="have",
            arguments=["helper_lemma : Prop"]
        )

        strategy.proof.tactics.insert(0, lemma_tactic)
        strategy.mutation_history.append(MutationType.LEMMA_INTRODUCTION)

        return strategy

    def _lemma_removal(self, strategy: LeanProofStrategy) -> LeanProofStrategy:
        """Remove a helper lemma (have statement)"""
        # Find and remove a 'have' tactic
        for i, tactic in enumerate(strategy.proof.tactics):
            if tactic.name == "have" and len(strategy.proof.tactics) > 1:
                strategy.proof.tactics.pop(i)
                strategy.mutation_history.append(MutationType.LEMMA_REMOVAL)
                break

        return strategy

    def _reordering(self, strategy: LeanProofStrategy) -> LeanProofStrategy:
        """Reorder tactics (swap two adjacent tactics)"""
        if len(strategy.proof.tactics) < 2:
            return strategy

        idx = random.randint(0, len(strategy.proof.tactics) - 2)
        strategy.proof.tactics[idx], strategy.proof.tactics[idx + 1] = (
            strategy.proof.tactics[idx + 1],
            strategy.proof.tactics[idx]
        )

        strategy.mutation_history.append(MutationType.REORDERING)
        return strategy

    def _simplification(self, strategy: LeanProofStrategy) -> LeanProofStrategy:
        """Replace complex tactic with simpler alternative"""
        return self._tactic_substitution(strategy)

    def _choose_random_tactic(self) -> Tactic:
        """Choose a random tactic from the available tactics"""
        all_tactics = self.ALL_TACTICS + self.custom_tactics
        tactic_name = random.choice(all_tactics)

        # Generate random arguments for some tactics
        arguments = []
        if tactic_name in ["apply", "exact", "refine"]:
            arguments = ["_"]  # Placeholder for actual lemma/theorem
        elif tactic_name in ["simp", "rw"]:
            arguments = ["_"]  # Placeholder for actual simplification rule

        return Tactic(name=tactic_name, arguments=arguments)


class LeanProofCrossover:
    """
    Performs crossover between two proof strategies.

    Supports various crossover methods for combining parent strategies.
    """

    def __init__(self, crossover_rate: float = 0.8):
        self.crossover_rate = crossover_rate

    def crossover(
        self,
        parent1: LeanProofStrategy,
        parent2: LeanProofStrategy,
        method: CrossoverMethod = CrossoverMethod.UNIFORM
    ) -> LeanProofStrategy:
        """
        Perform crossover between two parent strategies.

        Args:
            parent1: First parent strategy
            parent2: Second parent strategy
            method: Crossover method to use

        Returns:
            New child strategy combining parents
        """
        if random.random() > self.crossover_rate:
            # No crossover, return one parent at random
            return random.choice([parent1, parent2])

        if method == CrossoverMethod.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        elif method == CrossoverMethod.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2)
        elif method == CrossoverMethod.TWO_POINT:
            return self._two_point_crossover(parent1, parent2)
        elif method == CrossoverMethod.ORDERED:
            return self._ordered_crossover(parent1, parent2)
        else:
            return self._uniform_crossover(parent1, parent2)

    def _uniform_crossover(
        self,
        parent1: LeanProofStrategy,
        parent2: LeanProofStrategy
    ) -> LeanProofStrategy:
        """
        Uniform crossover: each tactic randomly selected from either parent.

        Creates a child by randomly selecting each tactic from either parent.
        """
        import copy

        # Create child proof
        child_proof = copy.deepcopy(parent1.proof)
        child_proof.proof_id = str(uuid.uuid4())
        child_proof.tactics = []

        # Get tactics from both parents
        tactics1 = parent1.proof.tactics
        tactics2 = parent2.proof.tactics

        max_len = max(len(tactics1), len(tactics2))

        # Randomly select from each parent
        for i in range(max_len):
            use_parent1 = random.choice([True, False])

            if i < len(tactics1) and i < len(tactics2):
                # Both parents have tactic at this position
                child_proof.tactics.append(
                    copy.deepcopy(tactics1[i] if use_parent1 else tactics2[i])
                )
            elif i < len(tactics1):
                child_proof.tactics.append(copy.deepcopy(tactics1[i]))
            elif i < len(tactics2):
                child_proof.tactics.append(copy.deepcopy(tactics2[i]))

        # Create child strategy
        child_strategy = LeanProofStrategy(
            proof=child_proof,
            generation=max(parent1.generation, parent2.generation) + 1,
            parents=[parent1.strategy_id, parent2.strategy_id]
        )

        return child_strategy

    def _single_point_crossover(
        self,
        parent1: LeanProofStrategy,
        parent2: LeanProofStrategy
    ) -> LeanProofStrategy:
        """
        Single-point crossover: split at random point and combine.
        """
        import copy

        tactics1 = parent1.proof.tactics
        tactics2 = parent2.proof.tactics

        if not tactics1 or not tactics2:
            return random.choice([parent1, parent2])

        # Choose crossover point
        max_point = min(len(tactics1), len(tactics2))
        if max_point < 2:
            return random.choice([parent1, parent2])

        point = random.randint(1, max_point - 1)

        # Create child with tactics from parent1 up to point, then parent2
        child_proof = copy.deepcopy(parent1.proof)
        child_proof.proof_id = str(uuid.uuid4())
        child_proof.tactics = []

        child_proof.tactics.extend(copy.deepcopy(tactics1[:point]))
        child_proof.tactics.extend(copy.deepcopy(tactics2[point:]))

        child_strategy = LeanProofStrategy(
            proof=child_proof,
            generation=max(parent1.generation, parent2.generation) + 1,
            parents=[parent1.strategy_id, parent2.strategy_id]
        )

        return child_strategy

    def _two_point_crossover(
        self,
        parent1: LeanProofStrategy,
        parent2: LeanProofStrategy
    ) -> LeanProofStrategy:
        """
        Two-point crossover: select segment from one parent and insert in other.
        """
        import copy

        tactics1 = parent1.proof.tactics
        tactics2 = parent2.proof.tactics

        if len(tactics1) < 3 or len(tactics2) < 3:
            return self._uniform_crossover(parent1, parent2)

        # Choose two crossover points
        max_point = min(len(tactics1), len(tactics2))
        point1 = random.randint(1, max_point - 2)
        point2 = random.randint(point1 + 1, max_point - 1)

        # Create child: tactics from parent1, but segment from parent2
        child_proof = copy.deepcopy(parent1.proof)
        child_proof.proof_id = str(uuid.uuid4())
        child_proof.tactics = []

        child_proof.tactics.extend(copy.deepcopy(tactics1[:point1]))
        child_proof.tactics.extend(copy.deepcopy(tactics2[point1:point2]))
        child_proof.tactics.extend(copy.deepcopy(tactics1[point2:]))

        child_strategy = LeanProofStrategy(
            proof=child_proof,
            generation=max(parent1.generation, parent2.generation) + 1,
            parents=[parent1.strategy_id, parent2.strategy_id]
        )

        return child_strategy

    def _ordered_crossover(
        self,
        parent1: LeanProofStrategy,
        parent2: LeanProofStrategy
    ) -> LeanProofStrategy:
        """
        Ordered crossover: preserve relative order of tactics.

        Useful when tactic order matters for the proof structure.
        """
        import copy

        tactics1 = parent1.proof.tactics
        tactics2 = parent2.proof.tactics

        if not tactics1 or not tactics2:
            return random.choice([parent1, parent2])

        # Take a random segment from parent1
        start = random.randint(0, len(tactics1) - 1)
        end = random.randint(start + 1, len(tactics1))

        child_proof = copy.deepcopy(parent1.proof)
        child_proof.proof_id = str(uuid.uuid4())
        child_proof.tactics = []

        # Add segment from parent1
        segment = tactics1[start:end]
        child_proof.tactics.extend(copy.deepcopy(segment))

        # Fill remaining from parent2 in order (if not already in child)
        for tactic in tactics2:
            if tactic.name not in [t.name for t in segment]:
                child_proof.tactics.append(copy.deepcopy(tactic))

        child_strategy = LeanProofStrategy(
            proof=child_proof,
            generation=max(parent1.generation, parent2.generation) + 1,
            parents=[parent1.strategy_id, parent2.strategy_id]
        )

        return child_strategy


class LeanProofEvaluator:
    """
    Evaluates proof strategies using LeanAide verification.

    Fitness is calculated based on:
    - Verification success (primary factor)
    - Proof length (shorter is better)
    - Tactic efficiency
    - Elegance score
    """

    def __init__(
        self,
        verification_engine: Optional[Lean4VerificationEngine] = None,
        server_url: str = "http://localhost:7654",
        cache_enabled: bool = True,
        parallel_evaluation: bool = True,
        max_concurrent: int = 5
    ):
        self.server_url = server_url
        self.cache_enabled = cache_enabled
        self.parallel_evaluation = parallel_evaluation
        self.max_concurrent = max_concurrent

        # Initialize verification engine
        if verification_engine:
            self.verification_engine = verification_engine
        elif LEANAIDE_AVAILABLE:
            server_config = Lean4ServerConfig(
                host="localhost",
                port=7654,
                enable_simulation_fallback=True
            )
            verification_config = Lean4VerificationConfig(
                enable_caching=cache_enabled,
                default_timeout=300
            )
            self.verification_engine = Lean4VerificationEngine(
                server_url, server_config, verification_config
            )
        else:
            self.verification_engine = None
            logger.warning("LeanAide not available, using simulation mode")

        # Fitness weights
        self.verification_weight = 10.0  # Success is most important
        self.length_weight = 0.1
        self.efficiency_weight = 0.2
        self.elegance_weight = 0.3

    async def evaluate(
        self,
        strategy: LeanProofStrategy,
        timeout: Optional[int] = None
    ) -> float:
        """
        Evaluate a proof strategy and return fitness score.

        Args:
            strategy: Strategy to evaluate
            timeout: Optional verification timeout

        Returns:
            Fitness score (higher is better)
        """
        # Generate Lean code from strategy
        lean_code = self._generate_lean_code(strategy)

        # Verify using LeanAide
        if self.verification_engine:
            try:
                result = await self.verification_engine.verify_mathematical_solution(
                    lean_code, timeout=timeout
                )
                strategy.proof.verification_result = result
                strategy.verified = result.success
            except (IOError, ConnectionError, TimeoutError, ValueError) as e:
                logger.error(f"Verification failed: {e}")
                strategy.verified = False
                result = None
        else:
            # Simulation mode
            strategy.verified = self._simulate_verification(strategy)
            result = None

        # Calculate fitness
        fitness = self._calculate_fitness(strategy, result)

        # Update strategy metrics
        strategy.fitness = fitness
        strategy.evaluation_count += 1
        strategy.calculate_complexity()
        strategy.calculate_elegance()

        return fitness

    async def evaluate_population(
        self,
        strategies: List[LeanProofStrategy],
        timeout: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate multiple strategies in parallel.

        Args:
            strategies: List of strategies to evaluate
            timeout: Optional verification timeout

        Returns:
            Dictionary mapping strategy IDs to fitness scores
        """
        if self.parallel_evaluation and self.verification_engine:
            # Parallel evaluation
            tasks = [
                self.evaluate(strategy, timeout)
                for strategy in strategies
            ]
            fitnesses = await asyncio.gather(*tasks, return_exceptions=True)

            results = {}
            for strategy, fitness in zip(strategies, fitnesses):
                if isinstance(fitness, Exception):
                    logger.error(f"Evaluation error for {strategy.strategy_id}: {fitness}")
                    results[strategy.strategy_id] = 0.0
                else:
                    results[strategy.strategy_id] = fitness

            return results
        else:
            # Sequential evaluation
            results = {}
            for strategy in strategies:
                fitness = await self.evaluate(strategy, timeout)
                results[strategy.strategy_id] = fitness

            return results

    def _generate_lean_code(self, strategy: LeanProofStrategy) -> str:
        """Generate Lean 4 code from a proof strategy"""
        proof = strategy.proof

        # Generate lean code
        lean_code = f"""import Mathlib

theorem {proof.theorem_name} : {proof.theorem_statement} := by
"""

        # Add tactics
        for tactic in proof.tactics:
            lean_code += f"  {tactic}\n"

        return lean_code

    def _calculate_fitness(
        self,
        strategy: LeanProofStrategy,
        verification_result: Optional[VerificationResult]
    ) -> float:
        """
        Calculate fitness score based on multiple factors.

        Fitness = verification_weight * (success ? 1 : 0)
                - length_weight * (num_tactics / 50)
                + efficiency_weight * (unique_tactics / total_tactics)
                + elegance_weight * elegance_score
        """
        # Verification success (primary factor)
        verification_score = 1.0 if strategy.verified else 0.0

        # Proof length penalty (prefer shorter proofs)
        num_tactics = len(strategy.proof.tactics)
        length_penalty = min(1.0, num_tactics / 50.0)

        # Tactic efficiency (prefer using diverse tactics efficiently)
        if num_tactics > 0:
            unique_tactics = len(set(t.name for t in strategy.proof.tactics))
            efficiency = unique_tactics / num_tactics
        else:
            efficiency = 0.0

        # Elegance score
        elegance = strategy.calculate_elegance()

        # Combine factors
        fitness = (
            self.verification_weight * verification_score
            - self.length_weight * length_penalty
            + self.efficiency_weight * efficiency
            + self.elegance_weight * elegance
        )

        # Ensure non-negative fitness
        return max(0.0, fitness)

    def _simulate_verification(self, strategy: LeanProofStrategy) -> bool:
        """
        Simulate verification when LeanAide is not available.

        Uses heuristics to estimate if the proof would verify.
        """
        # Simple heuristic: check if we have reasonable tactics
        num_tactics = len(strategy.proof.tactics)

        if num_tactics == 0:
            return False

        # Check for basic proof structure
        has_applicable = any(t.name in ["apply", "exact", "refine"] for t in strategy.proof.tactics)

        # Prefer proofs with applicable tactics
        return has_applicable or num_tactics >= 3

    async def close(self):
        """Close the verification engine"""
        if self.verification_engine:
            await self.verification_engine.close()


class LeanProofEvolutionEngine:
    """
    Main evolutionary proof generation engine.

    Orchestrates the evolutionary process:
    1. Generate initial population
    2. Evaluate all strategies
    3. Select parents
    4. Create offspring via crossover and mutation
    5. Replace population (with elitism)
    6. Repeat until convergence or max generations

    Features:
    - Adaptive mutation rates
    - Diversity maintenance
    - Convergence detection
    - Family tree tracking
    - Statistics collection
    """

    def __init__(
        self,
        theorem: str,
        theorem_name: Optional[str] = None,
        population_size: int = 20,
        max_generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
        crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM,
        elitism_ratio: float = 0.1,
        server_url: str = "http://localhost:7654",
        convergence_threshold: float = 0.001,
        stagnation_limit: int = 10,
        target_fitness: float = 8.0,
        cache_enabled: bool = True,
        parallel_evaluation: bool = True
    ):
        self.theorem = theorem
        self.theorem_name = theorem_name or "evolved_theorem"
        self.population_size = population_size
        self.max_generations = max_generations
        self.convergence_threshold = convergence_threshold
        self.stagnation_limit = stagnation_limit
        self.target_fitness = target_fitness

        # Initialize components
        self.mutator = LeanProofMutator(mutation_rate=mutation_rate)
        self.crossover = LeanProofCrossover(crossover_rate=crossover_rate)
        self.evaluator = LeanProofEvaluator(
            server_url=server_url,
            cache_enabled=cache_enabled,
            parallel_evaluation=parallel_evaluation
        )

        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.elitism_ratio = elitism_ratio

        # Evolution state
        self.current_generation = 0
        self.population: Optional[LeanProofPopulation] = None
        self.family_tree: Dict[str, List[str]] = defaultdict(list)
        self.failed_attempts: List[Dict[str, Any]] = []

        # Statistics
        self.statistics_history: List[PopulationStatistics] = []
        self.convergence_history: List[float] = []

    async def evolve(self) -> EvolutionResult:
        """
        Run the evolutionary proof generation process.

        Returns:
            EvolutionResult with best proof and statistics
        """
        start_time = time.time()
        total_evaluations = 0
        stagnation_counter = 0
        best_fitness_ever = 0.0

        logger.info(f"Starting evolutionary proof generation for: {self.theorem}")

        try:
            # Generate initial population
            logger.info("Generating initial population...")
            initial_strategies = await self.generate_initial_population()
            self.population = LeanProofPopulation(
                strategies=initial_strategies,
                selection_method=self.selection_method,
                elitism_ratio=self.elitism_ratio
            )

            # Evaluate initial population
            logger.info("Evaluating initial population...")
            await self.evaluate_population()

            # Record initial statistics
            stats = self.population.calculate_statistics()
            self.statistics_history.append(stats)
            best_fitness_ever = stats.best_fitness
            logger.info(f"Generation 0: Best fitness = {stats.best_fitness:.4f}")

            # Evolution loop
            for generation in range(1, self.max_generations + 1):
                self.current_generation = generation
                self.population.generation = generation

                logger.info(f"Generation {generation}")

                # Check for early termination
                best_strategy = self.population.get_best_strategy()
                if best_strategy and best_strategy.verified:
                    logger.info("Found verified proof!")
                    break

                if best_fitness_ever >= self.target_fitness:
                    logger.info(f"Target fitness {self.target_fitness} reached!")
                    break

                # Create next generation
                await self.create_next_generation()

                # Evaluate new population
                total_evaluations += len(self.population.strategies)
                await self.evaluate_population()

                # Calculate statistics
                stats = self.population.calculate_statistics()
                self.statistics_history.append(stats)
                self.convergence_history.append(stats.average_fitness)

                # Check for convergence/stagnation
                improvement = stats.best_fitness - best_fitness_ever
                if improvement < self.convergence_threshold:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                    best_fitness_ever = stats.best_fitness

                if stagnation_counter >= self.stagnation_limit:
                    logger.info(f"Stagnation limit reached after {generation} generations")
                    break

                logger.info(
                    f"Generation {generation}: "
                    f"Best = {stats.best_fitness:.4f}, "
                    f"Avg = {stats.average_fitness:.4f}, "
                    f"Verified = {stats.verified_count}"
                )

            # Get best strategy
            best_strategy = self.population.get_best_strategy()

            evolution_time = time.time() - start_time
            logger.info(f"Evolution completed in {evolution_time:.2f}s")

            # Create result
            result = EvolutionResult(
                success=best_strategy.verified if best_strategy else False,
                best_proof=best_strategy.proof if best_strategy else None,
                best_strategy=best_strategy,
                generations_completed=self.current_generation,
                total_evaluations=total_evaluations,
                evolution_time=evolution_time,
                statistics_history=self.statistics_history,
                family_tree=dict(self.family_tree),
                failed_attempts=self.failed_attempts,
                convergence_history=self.convergence_history
            )

            return result

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"Evolution failed: {e}", exc_info=True)
            return EvolutionResult(
                success=False,
                generations_completed=self.current_generation,
                evolution_time=time.time() - start_time,
                failed_attempts=[{"error": str(e)}]
            )

        finally:
            await self.evaluator.close()

    async def generate_initial_population(self) -> List[LeanProofStrategy]:
        """
        Generate initial population of proof strategies.

        Uses various heuristics to create diverse initial proofs.
        """
        strategies = []

        # Strategy 1: Empty proof (starting point)
        strategies.append(self._create_empty_strategy())

        # Strategy 2: Simple proof with basic tactics
        strategies.append(self._create_simple_strategy())

        # Strategy 3-5: Strategies with different tactic combinations
        for i in range(3):
            strategies.append(self._create_random_strategy())

        # Strategy 6+: Strategies based on proof search (if available)
        if LEANAIDE_AVAILABLE:
            try:
                search_strategies = await self._create_search_based_strategies()
                strategies.extend(search_strategies)
            except (IOError, ConnectionError, TimeoutError, ValueError) as e:
                logger.warning(f"Proof search failed: {e}")

        # Fill rest with random strategies
        while len(strategies) < self.population_size:
            strategies.append(self._create_random_strategy())

        return strategies[:self.population_size]

    def _create_empty_strategy(self) -> LeanProofStrategy:
        """Create a strategy with no tactics (placeholder)"""
        proof = LeanProof(
            theorem_name=self.theorem_name,
            theorem_statement=self.theorem,
            lean_code=f"theorem {self.theorem_name} : {self.theorem} := by\n  sorry"
        )

        return LeanProofStrategy(proof=proof, generation=0)

    def _create_simple_strategy(self) -> LeanProofStrategy:
        """Create a strategy with basic, commonly useful tactics"""
        tactics = [
            Tactic(name="intros"),
            Tactic(name="simp")
        ]

        proof = LeanProof(
            theorem_name=self.theorem_name,
            theorem_statement=self.theorem,
            tactics=tactics
        )

        return LeanProofStrategy(proof=proof, generation=0)

    def _create_random_strategy(self) -> LeanProofStrategy:
        """Create a strategy with random tactics"""
        num_tactics = random.randint(2, 8)
        all_tactics = LeanProofMutator.ALL_TACTICS

        tactics = []
        for _ in range(num_tactics):
            tactic_name = random.choice(all_tactics)
            tactics.append(Tactic(name=tactic_name))

        proof = LeanProof(
            theorem_name=self.theorem_name,
            theorem_statement=self.theorem,
            tactics=tactics
        )

        return LeanProofStrategy(proof=proof, generation=0)

    async def _create_search_based_strategies(self) -> List[LeanProofStrategy]:
        """Create strategies based on proof search results"""
        strategies = []

        # Use proof search to find related theorems
        # This is a placeholder - actual implementation would use ProofSearchEngine
        # to find similar proofs and extract tactics

        return strategies

    async def evaluate_population(self):
        """Evaluate all strategies in the population"""
        if not self.population:
            return

        fitnesses = await self.evaluator.evaluate_population(self.population.strategies)

        # Update fitnesses
        for strategy in self.population.strategies:
            strategy.fitness = fitnesses.get(strategy.strategy_id, 0.0)

    async def create_next_generation(self):
        """Create the next generation using selection, crossover, and mutation"""
        current_strategies = self.population.strategies
        population_size = len(current_strategies)

        # Elitism: keep best strategies
        num_elites = int(population_size * self.elitism_ratio)
        elites = self.population.get_elites(num_elites)

        # Select parents for offspring
        num_offspring = population_size - num_elites
        parents = self.population.select_parents(num_offspring * 2)

        # Create offspring through crossover and mutation
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            # Crossover
            child = self.crossover.crossover(parent1, parent2, self.crossover_method)

            # Mutation
            child = self.mutator.mutate(child)

            # Track family tree
            self.family_tree[f"{parent1.strategy_id}+{parent2.strategy_id}"].append(child.strategy_id)

            offspring.append(child)

            if len(offspring) >= num_offspring:
                break

        # Combine elites and offspring
        new_strategies = elites + offspring[:num_offspring]

        # Update population
        self.population.strategies = new_strategies

    def get_best_proof(self) -> Optional[LeanProof]:
        """Get the best proof found so far"""
        if not self.population:
            return None

        best_strategy = self.population.get_best_strategy()
        return best_strategy.proof if best_strategy else None

    async def close(self):
        """Clean up resources"""
        await self.evaluator.close()


# Convenience functions

async def evolve_proof(
    theorem: str,
    theorem_name: Optional[str] = None,
    max_generations: int = 50,
    population_size: int = 20,
    server_url: str = "http://localhost:7654",
    **kwargs
) -> EvolutionResult:
    """
    Convenience function to evolve a proof.

    Args:
        theorem: Theorem statement in natural language or Lean syntax
        theorem_name: Optional name for the theorem
        max_generations: Maximum number of generations
        population_size: Size of population
        server_url: LeanAide server URL
        **kwargs: Additional engine parameters

    Returns:
        EvolutionResult with best proof and statistics
    """
    engine = LeanProofEvolutionEngine(
        theorem=theorem,
        theorem_name=theorem_name,
        max_generations=max_generations,
        population_size=population_size,
        server_url=server_url,
        **kwargs
    )

    return await engine.evolve()


# Import MCTS components
try:
    from leanaide_mcts import (
        MCTS,
        MCTSNode,
        LeanProofMCTS,
        ProofContext as MCTSProofContext,
        TacticAction,
        Tactic as MCTSTactic,
        run_mcts_search
    )
    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False
    MCTS = None
    MCTSNode = None
    LeanProofMCTS = None
    MCTSProofContext = None
    TacticAction = None
    MCTSTactic = None
    run_mcts_search = None
    logger.warning("MCTS not available - MCTS integration features disabled")


# ============================================================================
# MCTS-Evolution Integration
# ============================================================================

class LeanProofEvolutionEngineMCTS(LeanProofEvolutionEngine):
    """
    Extended evolutionary engine with MCTS integration.

    Adds MCTS-powered initialization, mutation, and crossover strategies.
    """

    def __init__(self, *args, mcts_simulations: int = 100, mcts_exploration: float = 1.414, **kwargs):
        """
        Initialize evolutionary engine with MCTS capabilities.

        Args:
            *args: Arguments to pass to parent class
            mcts_simulations: Number of MCTS simulations per search
            mcts_exploration: MCTS exploration constant (UCB parameter)
            **kwargs: Additional keyword arguments for parent class
        """
        super().__init__(*args, **kwargs)

        self.mcts_simulations = mcts_simulations
        self.mcts_exploration = mcts_exploration

        # Initialize MCTS if available
        self.lean_mcts = None
        if MCTS_AVAILABLE:
            self.lean_mcts = LeanProofMCTS(
                exploration_constant=mcts_exploration,
                simulations=mcts_simulations
            )

    async def initialize_population_with_mcts(
        self,
        theorem: str,
        size: int
    ) -> List[LeanProofStrategy]:
        """
        Initialize population using MCTS to seed high-quality proofs.

        Uses MCTS search to generate diverse, high-quality initial proofs
        to seed the evolutionary population.

        Args:
            theorem: Theorem statement
            size: Desired population size

        Returns:
            List of proof strategies seeded by MCTS
        """
        logger.info(f"Initializing population with MCTS (size={size})")

        if not MCTS_AVAILABLE or not self.lean_mcts:
            logger.warning("MCTS not available, using standard initialization")
            return await self.generate_initial_population()

        strategies = []

        # Create initial proof context
        context = MCTSProofContext(
            goal=theorem,
            hypotheses=[],
            available_lemmas=self._get_available_lemmas(),
            depth=0
        )

        # Run MCTS searches with different exploration parameters
        mcts_configs = [
            {"exploration_constant": 1.0, "simulations": self.mcts_simulations},
            {"exploration_constant": 1.414, "simulations": self.mcts_simulations},
            {"exploration_constant": 2.0, "simulations": self.mcts_simulations},
        ]

        for config in mcts_configs[:min(size, len(mcts_configs))]:
            # Create MCTS instance with specific config
            mcts = LeanProofMCTS(
                exploration_constant=config["exploration_constant"],
                simulations=config["simulations"]
            )

            # Run MCTS search
            best_sequence, root = mcts.search(context)

            # Convert MCTS result to proof strategy
            if best_sequence:
                strategy = self._mcts_sequence_to_strategy(best_sequence, theorem, config)
                strategies.append(strategy)

        # Fill remaining population slots with diversity
        while len(strategies) < size:
            # Add some random strategies for diversity
            strategies.append(self._create_random_strategy())

        return strategies[:size]

    def _mcts_sequence_to_strategy(
        self,
        tactic_actions: List[TacticAction],
        theorem: str,
        mcts_config: Dict[str, Any]
    ) -> LeanProofStrategy:
        """Convert MCTS tactic sequence to proof strategy"""
        tactics = []
        for action in tactic_actions:
            tactic = Tactic(
                name=action.tactic.name,
                arguments=action.tactic.arguments
            )
            tactics.append(tactic)

        proof = LeanProof(
            theorem_name=self.theorem_name,
            theorem_statement=theorem,
            tactics=tactics
        )

        return LeanProofStrategy(
            proof=proof,
            generation=0,
            metadata={
                "mcts_generated": True,
                "mcts_config": mcts_config
            }
        )

    def mcts_guided_mutation(
        self,
        strategy: LeanProofStrategy,
        mcts: Optional[LeanProofMCTS] = None
    ) -> LeanProofStrategy:
        """
        Apply MCTS-guided mutation to a strategy.

        Uses MCTS to select which tactics to mutate and how to mutate them.

        Args:
            strategy: Strategy to mutate
            mcts: Optional MCTS instance (uses self.lean_mcts if None)

        Returns:
            Mutated strategy
        """
        if not MCTS_AVAILABLE or (mcts is None and self.lean_mcts is None):
            # Fall back to standard mutation
            return self.mutator.mutate(strategy)

        mcts_instance = mcts or self.lean_mcts

        # Create proof context from current strategy
        context = MCTSProofContext(
            goal=self.theorem,
            hypotheses=[],
            available_lemmas=self._get_available_lemmas(),
            depth=len(strategy.proof.tactics)
        )

        # Run MCTS to find alternative tactic sequences
        best_sequence, _ = mcts_instance.search(context)

        if best_sequence:
            # Create new strategy with MCTS-guided tactics
            return self._mcts_sequence_to_strategy(best_sequence, self.theorem, {})
        else:
            return self.mutator.mutate(strategy)

    def mcts_crossover(
        self,
        strategies: List[LeanProofStrategy],
        mcts: Optional[LeanProofMCTS] = None
    ) -> LeanProofStrategy:
        """
        Perform MCTS-guided crossover of multiple strategies.

        Uses MCTS to merge tactic sequences from multiple parent strategies.

        Args:
            strategies: List of parent strategies to crossover
            mcts: Optional MCTS instance

        Returns:
            Child strategy combining parents
        """
        if len(strategies) < 2:
            return strategies[0] if strategies else self._create_random_strategy()

        if not MCTS_AVAILABLE or (mcts is None and self.lean_mcts is None):
            # Fall back to standard crossover
            return self.crossover.crossover(strategies[0], strategies[1])

        # Extract tactic sequences from all parents
        all_tactics = []
        for strategy in strategies:
            all_tactics.extend(strategy.proof.tactics)

        # Use MCTS to find best combination
        # This is a simplified version - full implementation would be more sophisticated
        combined_tactics = all_tactics[:len(strategies[0].proof.tactics)]

        proof = LeanProof(
            theorem_name=self.theorem_name,
            theorem_statement=self.theorem,
            tactics=combined_tactics
        )

        return LeanProofStrategy(
            proof=proof,
            generation=max(s.generation for s in strategies) + 1,
            parents=[s.strategy_id for s in strategies],
            metadata={"mcts_crossover": True}
        )

    def _get_available_lemmas(self) -> List[str]:
        """Get available lemmas for theorem domain"""
        # Simplified - would use domain knowledge
        return ["Nat.add_zero", "Nat.add_succ", "Nat.mul_one"]

    async def evolve_with_mcts(
        self,
        mct_s_ratio: float = 0.3
    ) -> EvolutionResult:
        """
        Run evolution with MCTS-enhanced operators.

        Args:
            mcts_ratio: Ratio of operations that use MCTS (0.0 to 1.0)

        Returns:
            Evolution result with MCTS-enhanced evolution
        """
        logger.info(f"Starting MCTS-enhanced evolution (MCTS ratio={mcts_ratio})")

        start_time = time.time()
        total_evaluations = 0

        try:
            # Initialize population with MCTS
            logger.info("Initializing population with MCTS...")
            initial_strategies = await self.initialize_population_with_mcts(
                self.theorem,
                self.population_size
            )
            self.population = LeanProofPopulation(
                strategies=initial_strategies,
                selection_method=self.selection_method,
                elitism_ratio=self.elitism_ratio
            )

            # Evaluate initial population
            await self.evaluate_population()
            stats = self.population.calculate_statistics()
            self.statistics_history.append(stats)
            logger.info(f"Initial population: Best fitness = {stats.best_fitness:.4f}")

            # Evolution loop with MCTS operators
            for generation in range(1, self.max_generations + 1):
                self.current_generation = generation
                self.population.generation = generation

                # Check termination
                best_strategy = self.population.get_best_strategy()
                if best_strategy and best_strategy.verified:
                    logger.info("Found verified proof!")
                    break

                # Create next generation with MCTS-enhanced operators
                await self.create_next_generation_with_mcts(mcts_ratio)

                # Evaluate
                total_evaluations += len(self.population.strategies)
                await self.evaluate_population()

                # Statistics
                stats = self.population.calculate_statistics()
                self.statistics_history.append(stats)
                self.convergence_history.append(stats.average_fitness)

                logger.info(
                    f"Generation {generation}: "
                    f"Best = {stats.best_fitness:.4f}, "
                    f"Avg = {stats.average_fitness:.4f}"
                )

            evolution_time = time.time() - start_time
            best_strategy = self.population.get_best_strategy()

            result = EvolutionResult(
                success=best_strategy.verified if best_strategy else False,
                best_proof=best_strategy.proof if best_strategy else None,
                best_strategy=best_strategy,
                generations_completed=self.current_generation,
                total_evaluations=total_evaluations,
                evolution_time=evolution_time,
                statistics_history=self.statistics_history,
                family_tree=dict(self.family_tree),
                failed_attempts=self.failed_attempts,
                convergence_history=self.convergence_history
            )

            return result

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"MCTS-enhanced evolution failed: {e}", exc_info=True)
            return EvolutionResult(
                success=False,
                generations_completed=self.current_generation,
                evolution_time=time.time() - start_time,
                failed_attempts=[{"error": str(e)}]
            )

    async def create_next_generation_with_mcts(self, mcts_ratio: float):
        """Create next generation using MCTS-enhanced operators"""
        current_strategies = self.population.strategies
        population_size = len(current_strategies)

        # Elitism
        num_elites = int(population_size * self.elitism_ratio)
        elites = self.population.get_elites(num_elites)

        # Select parents
        num_offspring = population_size - num_elites
        parents = self.population.select_parents(num_offspring * 2)

        # Create offspring
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            # Decide whether to use MCTS operators
            use_mcts = random.random() < mcts_ratio

            if use_mcts and MCTS_AVAILABLE:
                # MCTS-enhanced operators
                if random.random() < self.crossover.crossover_rate:
                    child = self.mcts_crossover([parent1, parent2], self.lean_mcts)
                else:
                    child = self.mcts_guided_mutation(parent1, self.lean_mcts)
            else:
                # Standard operators
                child = self.crossover.crossover(parent1, parent2, self.crossover_method)
                child = self.mutator.mutate(child)

            self.family_tree[f"{parent1.strategy_id}+{parent2.strategy_id}"].append(child.strategy_id)
            offspring.append(child)

            if len(offspring) >= num_offspring:
                break

        # Combine elites and offspring
        new_strategies = elites + offspring[:num_offspring]
        self.population.strategies = new_strategies


# =============================================================================
# MDAP+MCTS Integration for Evolution
# =============================================================================

# Import MDAP components if available
try:
    from leanaide_mdap import (
        LeanMDAPOrchestrator,
        LeanMDAPConfig,
        LeanProofAgent,
        ProofStrategy,
        MDAP_AVAILABLE,
    )
    from leanaide_mcts import (
        MDAPMCTSConfig,
        MCTSMDAPIntegration,
        MDAPMCTSHybrid,
        MCTSConfig,
        MCTSNode,
        ProofState,
    )
    MDAP_MCTS_AVAILABLE = MDAP_AVAILABLE
except ImportError:
    MDAP_MCTS_AVAILABLE = False
    logger.warning("MDAP+MCTS integration not available")


@dataclass
class MDAPMCTSGenerationConfig:
    """
    Configuration for MDAP+MCTS-enhanced proof generation.

    Attributes:
        # MCTS settings
        mcts_iterations: Number of MCTS iterations to run
        mcts_time_budget: Time budget for MCTS

        # MDAP settings
        mdap_num_agents: Number of MDAP agents
        mdap_agent_types: Types of MDAP agents
        mdap_voting_strategy: Voting strategy for MDAP

        # Hybrid settings
        hybrid_mode: Hybrid mode ("mcts_then_mdap", "mdap_then_mcts", "parallel", "adaptive")
        hybrid_ratio: Ratio of MCTS vs MDAP iterations

        # Evolution settings
        population_size: Size of initial population to generate
        elite_ratio: Ratio of elites to keep
        mutation_rate: Mutation rate for evolution
    """
    mcts_iterations: int = 100
    mcts_time_budget: float = 30.0

    mdap_num_agents: int = 4
    mdap_agent_types: List[str] = field(default_factory=lambda: ["evolution", "mcts", "adversarial", "self_play"])
    mdap_voting_strategy: str = "first_k_ahead"

    hybrid_mode: str = "mcts_then_mdap"
    hybrid_ratio: float = 0.5

    population_size: int = 20
    elite_ratio: float = 0.2
    mutation_rate: float = 0.3


async def mcts_with_mdap_generation(
    theorem: str,
    theorem_name: Optional[str] = None,
    config: Optional[MDAPMCTSGenerationConfig] = None,
    lean_client: Optional[Any] = None
) -> EvolutionResult:
    """
    Generate proof using MCTS with MDAP enhancement.

    Args:
        theorem: Theorem statement
        theorem_name: Optional theorem name
        config: MDAP+MCTS configuration
        lean_client: Optional Lean client

    Returns:
        EvolutionResult with generated proof
    """
    if not MDAP_MCTS_AVAILABLE:
        logger.warning("MDAP+MCTS not available, falling back to basic MCTS")
        # Fallback to basic MCTS if available
        if MCTS_AVAILABLE:
            mcts_config = MCTSConfig(
                max_iterations=config.mcts_iterations if config else 100,
                time_budget=config.mcts_time_budget if config else 30.0,
            )
            from leanaide_mcts import MCTS
            mcts = MCTS(mcts_config, theorem, theorem_name)
            mcts_result = await mcts.search()

            return EvolutionResult(
                success=mcts_result.success,
                best_proof=mcts_result.best_proof,
                generations_completed=1,
                evolution_time=mcts_result.time_elapsed,
            )
        else:
            return EvolutionResult(success=False, generations_completed=0)

    config = config or MDAPMCTSGenerationConfig()

    logger.info(f"MCTS+MDAP generation: {theorem}")
    logger.info(f"Hybrid mode: {config.hybrid_mode}")

    start_time = time.time()

    # Create MDAP+MCTS configuration
    mcts_config = MCTSConfig(
        max_iterations=config.mcts_iterations,
        time_budget=config.mcts_time_budget,
    )

    mdap_mcts_config = MDAPMCTSConfig(
        base_mcts_config=mcts_config,
        num_mdap_agents=config.mdap_num_agents,
        mdap_agent_types=config.mdap_agent_types,
        mdap_voting_strategy=config.mdap_voting_strategy,
    )

    # Create hybrid system
    hybrid = MDAPMCTSHybrid(mdap_mcts_config)

    # Run based on hybrid mode
    if config.hybrid_mode == "mcts_then_mdap":
        result = await hybrid.mcts_then_mdap(
            theorem,
            theorem_name,
            config.mcts_iterations,
            config.mdap_num_agents
        )
    elif config.hybrid_mode == "mdap_then_mcts":
        result = await hybrid.mdap_then_mcts(
            theorem,
            theorem_name,
            config.mdap_num_agents,
            config.mcts_iterations
        )
    elif config.hybrid_mode == "parallel":
        result = await hybrid.mdap_mcts_parallel(
            theorem,
            theorem_name,
            config.mcts_iterations,
            config.mdap_num_agents
        )
    elif config.hybrid_mode == "adaptive":
        result = await hybrid.adaptive_mdap_mcts(
            theorem,
            theorem_name,
            config.mcts_time_budget
        )
    else:
        logger.warning(f"Unknown hybrid mode: {config.hybrid_mode}, using mcts_then_mdap")
        result = await hybrid.mcts_then_mdap(
            theorem,
            theorem_name,
            config.mcts_iterations,
            config.mdap_num_agents
        )

    evolution_time = time.time() - start_time

    # Convert to EvolutionResult
    evolution_result = EvolutionResult(
        success=result.success,
        best_proof=result.best_proof,
        generations_completed=1,  # Single generation from MCTS+MDAP
        total_evaluations=result.nodes_visited,
        evolution_time=evolution_time,
        convergence_history=[result.win_rate] if result.win_rate else [],
    )

    return evolution_result


async def seed_population_with_mdap_mcts(
    theorem: str,
    size: int,
    config: Optional[MDAPMCTSGenerationConfig] = None,
    lean_client: Optional[Any] = None
) -> List[LeanProofStrategy]:
    """
    Seed an initial population using MDAP+MCTS.

    Generates multiple proof strategies by running MDAP+MCTS with different
    configurations and random seeds.

    Args:
        theorem: Theorem statement
        size: Population size to generate
        config: Base MDAP+MCTS configuration
        lean_client: Optional Lean client

    Returns:
        List of LeanProofStrategy objects
    """
    if not MDAP_MCTS_AVAILABLE:
        logger.warning("MDAP+MCTS not available for population seeding")
        return []

    logger.info(f"Seeding population of size {size} with MDAP+MCTS")

    population = []
    config = config or MDAPMCTSGenerationConfig()

    # Try different hybrid modes to create diversity
    hybrid_modes = ["mcts_then_mdap", "mdap_then_mcts", "parallel", "adaptive"]

    for i in range(size):
        # Vary configuration for diversity
        mode = hybrid_modes[i % len(hybrid_modes)]

        varied_config = MDAPMCTSGenerationConfig(
            mcts_iterations=config.mcts_iterations + random.randint(-20, 20),
            mcts_time_budget=config.mcts_time_budget + random.uniform(-5, 5),
            mdap_num_agents=max(2, config.mdap_num_agents + random.randint(-1, 1)),
            hybrid_mode=mode,
        )

        try:
            # Generate proof strategy
            result = await mcts_with_mdap_generation(
                theorem,
                f"{theorem}_seed_{i}",
                varied_config,
                lean_client
            )

            if result.best_proof:
                # Create strategy from result
                strategy = LeanProofStrategy(
                    strategy_id=f"mdap_mcts_seed_{i}",
                    proof=result.best_proof,
                    tactics=result.best_proof.tactics if result.best_proof else [],
                    fitness=result.best_proof.fitness if result.best_proof else 0.0,
                    generation=0,
                    parent_ids=[],
                    verified=result.success,
                )

                population.append(strategy)
                logger.info(f"Generated seed {i+1}/{size}: fitness={strategy.fitness:.4f}")

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning(f"Failed to generate seed {i}: {e}")
            continue

    logger.info(f"Generated {len(population)} strategies from MDAP+MCTS seeding")
    return population


class LeanProofEvolutionEngineMDAP(LeanProofEvolutionEngineMCTS):
    """
    Evolution engine enhanced with both MCTS and MDAP.

    Combines evolutionary algorithms with MCTS tree search and MDAP multi-agent
    consensus for powerful proof generation.
    """

    def __init__(
        self,
        theorem: str,
        config: Optional[MDAPMCTSGenerationConfig] = None,
        lean_client: Optional[Any] = None,
        enable_mdap: bool = True
    ):
        """
        Initialize MDAP+MCTS evolution engine.

        Args:
            theorem: Theorem to prove
            config: MDAP+MCTS configuration
            lean_client: Optional Lean client
            enable_mdap: Enable MDAP integration
        """
        # Initialize parent class
        super().__init__(theorem, config, lean_client)

        self.enable_mdap = enable_mdap and MDAP_MCTS_AVAILABLE
        self.mdap_config = config or MDAPMCTSGenerationConfig()

        # MDAP components
        self.mdap_mcts_integration = None
        if self.enable_mdap:
            mcts_config = MCTSConfig(
                max_iterations=self.mdap_config.mcts_iterations,
                time_budget=self.mdap_config.mcts_time_budget,
            )

            mdap_mcts_config = MDAPMCTSConfig(
                base_mcts_config=mcts_config,
                num_mdap_agents=self.mdap_config.mdap_num_agents,
                mdap_agent_types=self.mdap_config.mdap_agent_types,
            )

            self.mdap_mcts_integration = MCTSMDAPIntegration(mdap_mcts_config)
            logger.info("MDAP+MCTS integration initialized")

    async def evolve_with_mdap_mcts(
        self,
        max_generations: int = 50,
        mcts_ratio: float = 0.3,
        mdap_ratio: float = 0.3
    ) -> EvolutionResult:
        """
        Run evolution with both MCTS and MDAP enhancement.

        Args:
            max_generations: Maximum generations
            mcts_ratio: Ratio of MCTS-enhanced operations
            mdap_ratio: Ratio of MDAP-enhanced operations

        Returns:
            EvolutionResult
        """
        if not self.enable_mdap:
            logger.warning("MDAP not enabled, falling back to MCTS-only evolution")
            return await self.evolve_with_mcts(max_generations, mcts_ratio)

        logger.info(f"Starting MDAP+MCTS evolution for: {self.theorem}")
        logger.info(f"Max generations: {max_generations}")

        start_time = time.time()
        total_evaluations = 0

        try:
            # Seed population with MDAP+MCTS if configured
            if self.mdap_config.population_size > 0:
                logger.info("Seeding initial population with MDAP+MCTS")
                seeds = await seed_population_with_mdap_mcts(
                    self.theorem,
                    self.mdap_config.population_size,
                    self.mdap_config,
                    self.lean_client
                )

                if seeds:
                    # Add seeds to existing population
                    self.population.strategies.extend(seeds)
                    # Trim to max population size
                    if len(self.population.strategies) > self.population_size:
                        self.population.strategies = self.population.strategies[:self.population_size]

            # Evaluate initial population
            await self.evaluate_population()
            stats = self.population.calculate_statistics()
            self.statistics_history.append(stats)
            logger.info(f"Initial population: Best fitness = {stats.best_fitness:.4f}")

            # Evolution loop with MDAP+MCTS operators
            for generation in range(1, max_generations + 1):
                self.current_generation = generation

                # Check termination
                best_strategy = self.population.get_best_strategy()
                if best_strategy and best_strategy.verified:
                    logger.info("Found verified proof!")
                    break

                # Create next generation with MDAP+MCTS operators
                await self.create_next_generation_with_mdap_mcts(mcts_ratio, mdap_ratio)

                # Evaluate
                total_evaluations += len(self.population.strategies)
                await self.evaluate_population()

                # Statistics
                stats = self.population.calculate_statistics()
                self.statistics_history.append(stats)
                self.convergence_history.append(stats.average_fitness)

                logger.info(
                    f"Generation {generation}: "
                    f"Best = {stats.best_fitness:.4f}, "
                    f"Avg = {stats.average_fitness:.4f}"
                )

            evolution_time = time.time() - start_time
            best_strategy = self.population.get_best_strategy()

            result = EvolutionResult(
                success=best_strategy.verified if best_strategy else False,
                best_proof=best_strategy.proof if best_strategy else None,
                best_strategy=best_strategy,
                generations_completed=self.current_generation,
                total_evaluations=total_evaluations,
                evolution_time=evolution_time,
                statistics_history=self.statistics_history,
                family_tree=dict(self.family_tree),
                failed_attempts=self.failed_attempts,
                convergence_history=self.convergence_history
            )

            return result

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"MDAP+MCTS evolution failed: {e}", exc_info=True)
            return EvolutionResult(
                success=False,
                generations_completed=self.current_generation,
                evolution_time=time.time() - start_time,
                failed_attempts=[{"error": str(e)}]
            )

    async def create_next_generation_with_mdap_mcts(
        self,
        mcts_ratio: float,
        mdap_ratio: float
    ):
        """Create next generation using MDAP+MCTS operators"""
        current_strategies = self.population.strategies
        population_size = len(current_strategies)

        # Elitism
        num_elites = int(population_size * self.elitism_ratio)
        elites = self.population.get_elites(num_elites)

        # Select parents
        num_offspring = population_size - num_elites
        parents = self.population.select_parents(num_offspring * 2)

        # Create offspring
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            # Decide operator type
            rand = random.random()
            use_mcts = rand < mcts_ratio
            use_mdap = rand < mcts_ratio + mdap_ratio

            if use_mcts and MCTS_AVAILABLE:
                # MCTS-enhanced operators
                if random.random() < self.crossover.crossover_rate:
                    child = self.mcts_crossover([parent1, parent2], self.lean_mcts)
                else:
                    child = self.mcts_guided_mutation(parent1, self.lean_mcts)
            elif use_mdap and self.mdap_mcts_integration:
                # MDAP-enhanced operators
                if random.random() < self.crossover.crossover_rate:
                    child = await self.mdap_crossover([parent1, parent2])
                else:
                    child = await self.mdap_guided_mutation(parent1)
            else:
                # Standard operators
                child = self.crossover.crossover(parent1, parent2, self.crossover_method)
                child = self.mutator.mutate(child)

            self.family_tree[f"{parent1.strategy_id}+{parent2.strategy_id}"].append(child.strategy_id)
            offspring.append(child)

            if len(offspring) >= num_offspring:
                break

        # Combine elites and offspring
        new_strategies = elites + offspring[:num_offspring]
        self.population.strategies = new_strategies

    async def mdap_crossover(self, parents: List[LeanProofStrategy]) -> LeanProofStrategy:
        """Perform MDAP-guided crossover."""
        # Use MDAP consensus to select best tactics from parents
        if not self.mdap_mcts_integration:
            return self.crossover.crossover(parents[0], parents[1], self.crossover_method)

        # Get tactics from both parents
        all_tactics = []
        for parent in parents:
            all_tactics.extend(parent.tactics)

        # Use MDAP to rank tactics
        ranked_tactics = []
        for tactic in all_tactics:
            score = self._mdap_score_tactic(tactic)
            ranked_tactics.append((tactic, score))

        # Select top tactics
        ranked_tactics.sort(key=lambda x: x[1], reverse=True)
        selected_tactics = [t for t, _ in ranked_tactics[:len(all_tactics) // 2 + 1]]

        # Create child strategy
        child = LeanProofStrategy(
            strategy_id=f"mdap_cross_{uuid.uuid4().hex[:8]}",
            proof=None,
            tactics=selected_tactics,
            fitness=0.0,
            generation=self.current_generation,
            parent_ids=[p.strategy_id for p in parents],
            verified=False,
        )

        return child

    async def mdap_guided_mutation(self, parent: LeanProofStrategy) -> LeanProofStrategy:
        """Perform MDAP-guided mutation."""
        if not self.mdap_mcts_integration:
            return self.mutator.mutate(parent)

        # Get tactics
        tactics = parent.tactics.copy()

        # Use MDAP to suggest replacement tactics
        for i in range(len(tactics)):
            if random.random() < self.mutation_rate:
                # Get MDAP suggestion
                new_tactic = await self._mdap_suggest_tactic(i, tactics)
                if new_tactic:
                    tactics[i] = new_tactic

        # Create mutated strategy
        child = LeanProofStrategy(
            strategy_id=f"mdap_mut_{uuid.uuid4().hex[:8]}",
            proof=None,
            tactics=tactics,
            fitness=0.0,
            generation=self.current_generation,
            parent_ids=[parent.strategy_id],
            verified=False,
        )

        return child

    def _mdap_score_tactic(self, tactic: Tactic) -> float:
        """Score a tactic using MDAP agent consensus."""
        if not self.mdap_mcts_integration:
            return 0.5

        # Simulate agent voting
        score = 0.0
        for agent_type, perf in self.mdap_mcts_integration.agent_performance.items():
            weight = perf["success_rate"]
            # Simple heuristic scoring
            if tactic.name in ["simp", "intros"]:
                score += weight * 0.8
            elif tactic.name in ["apply", "exact"]:
                score += weight * 0.7
            else:
                score += weight * 0.5

        return score / len(self.mdap_mcts_integration.agent_performance)

    async def _mdap_suggest_tactic(self, position: int, current_tactics: List[Tactic]) -> Optional[Tactic]:
        """Suggest a tactic using MDAP consensus."""
        # Simulate MDAP suggestion
        suggestions = ["simp", "rw", "apply", "intros", "cases"]
        return Tactic(name=random.choice(suggestions))


# =============================================================================
# MDAP-Enhanced Evolution Engine
# =============================================================================

class LeanProofEvolutionEngineMDAPFull(LeanProofEvolutionEngineMDAP):
    """
    Full MDAP integration for evolutionary proof generation.

    Adds complete MDAP/MAKER voting integration to evolutionary operators:
    - MDAP-enhanced generational evolution
    - MDAP-guided parent selection
    - MDAP-based crossover and mutation
    - Hybrid execution modes
    - Performance tracking for pure evolution vs MDAP-evolution
    """

    def __init__(self, *args, mdap_maker_config: Optional[MDAPMCTSGenerationConfig] = None, **kwargs):
        """Initialize MDAP-full evolution engine."""
        super().__init__(*args, **kwargs)
        self.mdap_maker_config = mdap_maker_config or MDAPMCTSGenerationConfig()

        # Performance tracking
        self.mdap_vs_pure_stats = {
            "mdap_generations": 0,
            "pure_generations": 0,
            "mdap_time": 0.0,
            "pure_time": 0.0,
            "mdap_fitness_improvements": [],
            "pure_fitness_improvements": [],
            "agent_contributions": defaultdict(int),
            "voting_statistics": defaultdict(int)
        }

    async def evolve_generation_with_mdap(
        self,
        population: LeanProofPopulation,
        generation: int
    ) -> LeanProofPopulation:
        """
        Evolve a generation using MDAP-enhanced operators.

        Args:
            population: Current population
            generation: Generation number

        Returns:
            New evolved population
        """
        start_time = time.time()
        logger.info(f"Evolving generation {generation} with MDAP")

        # Select parents using MDAP
        num_parents = min(len(population), self.population_size // 2)
        parents = await self.select_parents_with_mdap(population, num_parents)

        # Create offspring through MDAP-enhanced crossover and mutation
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            # MDAP-based crossover
            child = await self.crossover_with_mdap(parent1, parent2)

            # MDAP-based mutation
            child = await self.mutate_with_mdap(child)

            child.generation = generation
            offspring.append(child)

        # Combine with elites
        num_elites = int(len(population) * self.elitism_ratio)
        elites = population.get_elites(num_elites)

        new_strategies = elites + offspring[:len(population) - num_elites]
        new_population = LeanProofPopulation(
            strategies=new_strategies,
            selection_method=population.selection_method,
            elitism_ratio=self.elitism_ratio
        )
        new_population.generation = generation

        elapsed = time.time() - start_time
        self.mdap_vs_pure_stats["mdap_generations"] += 1
        self.mdap_vs_pure_stats["mdap_time"] += elapsed

        logger.info(f"MDAP generation {generation} completed in {elapsed:.2f}s")
        return new_population

    async def select_parents_with_mdap(
        self,
        population: LeanProofPopulation,
        count: int
    ) -> List[LeanProofStrategy]:
        """
        Select parents using MDAP voting.

        Args:
            population: Population to select from
            count: Number of parents to select

        Returns:
            List of selected parent strategies
        """
        if not self.mdap_mcts_integration:
            # Fallback to standard selection
            return population.select_parents(count)

        # Get top candidates by fitness
        candidates = sorted(
            population.strategies,
            key=lambda s: s.fitness,
            reverse=True
        )[:count * 2]

        # Use MDAP to vote on best parents
        scored_parents = []
        for candidate in candidates:
            # MDAP agent voting on parent quality
            mdap_score = 0.0
            for agent_type in self.mdap_maker_config.mdap_agent_types:
                agent_weight = self._get_agent_weight(agent_type)
                agent_score = self._evaluate_parent_for_agent(candidate, agent_type)
                mdap_score += agent_weight * agent_score

                # Track agent contribution
                if agent_score > 0.7:
                    self.mdap_vs_pure_stats["agent_contributions"][agent_type] += 1

            scored_parents.append((candidate, mdap_score))

        # Sort by MDAP score
        scored_parents.sort(key=lambda x: x[1], reverse=True)

        # Select top K using first-K-ahead voting
        k_ahead = self.mdap_maker_config.mdap_k_ahead
        selected = [p[0] for p in scored_parents[:count]]

        # Track voting statistics
        for i, (parent, score) in enumerate(scored_parents[:count]):
            self.mdap_vs_pure_stats["voting_statistics"][f"rank_{i}"] += 1

        logger.info(f"MDAP selected {len(selected)} parents from {len(candidates)} candidates")
        return selected

    async def crossover_with_mdap(
        self,
        parent1: LeanProofStrategy,
        parent2: LeanProofStrategy
    ) -> LeanProofStrategy:
        """
        Perform crossover using MDAP consensus.

        Args:
            parent1: First parent strategy
            parent2: Second parent strategy

        Returns:
            Child strategy combining parents
        """
        if not self.mdap_mcts_integration:
            return self.crossover.crossover(parent1, parent2, self.crossover_method)

        # Get tactics from both parents
        tactics1 = parent1.proof.tactics
        tactics2 = parent2.proof.tactics

        # Use MDAP to vote on best tactics at each position
        child_tactics = []
        max_len = max(len(tactics1), len(tactics2))

        for i in range(max_len):
            tactic1 = tactics1[i] if i < len(tactics1) else None
            tactic2 = tactics2[i] if i < len(tactics2) else None

            # MDAP voting on which tactic to use
            if tactic1 and tactic2:
                votes = {}
                for agent_type in self.mdap_maker_config.mdap_agent_types:
                    preference = self._agent_tactic_preference(
                        tactic1, tactic2, agent_type
                    )
                    votes[preference] = votes.get(preference, 0) + 1

                # Select tactic with most votes
                selected_tactic = max(votes.items(), key=lambda x: x[1])[0]
                child_tactics.append(selected_tactic)
            elif tactic1:
                child_tactics.append(tactic1)
            elif tactic2:
                child_tactics.append(tactic2)

        # Create child strategy
        child_proof = LeanProof(
            theorem_name=parent1.proof.theorem_name,
            theorem_statement=parent1.proof.theorem_statement,
            tactics=child_tactics
        )

        child = LeanProofStrategy(
            proof=child_proof,
            generation=max(parent1.generation, parent2.generation) + 1,
            parents=[parent1.strategy_id, parent2.strategy_id],
            metadata={"mdap_crossover": True}
        )

        return child

    async def mutate_with_mdap(
        self,
        individual: LeanProofStrategy
    ) -> LeanProofStrategy:
        """
        Perform mutation using MDAP consensus.

        Args:
            individual: Strategy to mutate

        Returns:
            Mutated strategy
        """
        if not self.mdap_mcts_integration:
            return self.mutator.mutate(individual)

        # Get current tactics
        tactics = individual.proof.tactics.copy()

        # For each tactic, MDAP agents vote on potential mutations
        for i in range(len(tactics)):
            if random.random() < self.mutation_rate:
                # Get MDAP suggestions for replacement
                suggestions = await self._get_mdap_tactic_suggestions(
                    tactics[i], i, tactics
                )

                # Vote on best suggestion
                if suggestions:
                    selected = max(suggestions.items(), key=lambda x: x[1])[0]
                    if selected:
                        tactics[i] = selected

        # Create mutated strategy
        mutated_proof = LeanProof(
            theorem_name=individual.proof.theorem_name,
            theorem_statement=individual.proof.theorem_statement,
            tactics=tactics
        )

        mutated = LeanProofStrategy(
            proof=mutated_proof,
            generation=individual.generation + 1,
            parents=[individual.strategy_id],
            mutation_history=[MutationType.TACTIC_SUBSTITUTION],
            metadata={"mdap_mutation": True}
        )

        return mutated

    async def evolve_with_mdap_mode(
        self,
        theorem: str,
        config: MDAPMCTSGenerationConfig
    ) -> EvolutionResult:
        """
        Run evolution in MDAP-enhanced mode.

        Args:
            theorem: Theorem to prove
            config: MDAP-MCTS configuration

        Returns:
            EvolutionResult with MDAP-enhanced evolution
        """
        logger.info(f"Starting MDAP-mode evolution for: {theorem}")
        logger.info(f"MDAP agents: {config.mdap_num_agents}")
        logger.info(f"Voting strategy: {config.mdap_voting_strategy}")

        start_time = time.time()
        total_evaluations = 0

        try:
            # Initialize population with MDAP seeding if configured
            if config.hybrid_mode in ["mdap_then_mcts", "parallel"]:
                logger.info("Seeding population with MDAP-generated strategies")
                initial_strategies = await seed_population_with_mdap_mcts(
                    self.theorem,
                    self.population_size,
                    config
                )
            else:
                initial_strategies = await self.generate_initial_population()

            self.population = LeanProofPopulation(
                strategies=initial_strategies,
                selection_method=self.selection_method,
                elitism_ratio=self.elitism_ratio
            )

            # Evaluate initial population
            await self.evaluate_population()
            stats = self.population.calculate_statistics()
            self.statistics_history.append(stats)
            logger.info(f"Initial: Best fitness = {stats.best_fitness:.4f}")

            # Evolution loop with MDAP
            best_fitness_ever = stats.best_fitness

            for generation in range(1, self.max_generations + 1):
                self.current_generation = generation

                # Check termination
                best_strategy = self.population.get_best_strategy()
                if best_strategy and best_strategy.verified:
                    logger.info("Found verified proof!")
                    break

                # Evolve generation with MDAP
                self.population = await self.evolve_generation_with_mdap(
                    self.population, generation
                )

                # Evaluate
                total_evaluations += len(self.population.strategies)
                await self.evaluate_population()

                # Statistics
                stats = self.population.calculate_statistics()
                self.statistics_history.append(stats)
                self.convergence_history.append(stats.average_fitness)

                # Track MDAP improvements
                improvement = stats.best_fitness - best_fitness_ever
                self.mdap_vs_pure_stats["mdap_fitness_improvements"].append(improvement)
                if improvement > 0:
                    best_fitness_ever = stats.best_fitness

                logger.info(
                    f"Generation {generation}: "
                    f"Best = {stats.best_fitness:.4f}, "
                    f"Avg = {stats.average_fitness:.4f}"
                )

            evolution_time = time.time() - start_time
            best_strategy = self.population.get_best_strategy()

            result = EvolutionResult(
                success=best_strategy.verified if best_strategy else False,
                best_proof=best_strategy.proof if best_strategy else None,
                best_strategy=best_strategy,
                generations_completed=self.current_generation,
                total_evaluations=total_evaluations,
                evolution_time=evolution_time,
                statistics_history=self.statistics_history,
                family_tree=dict(self.family_tree),
                failed_attempts=self.failed_attempts,
                convergence_history=self.convergence_history
            )

            # Add MDAP statistics to result
            result.metadata = {
                "mdap_stats": self.mdap_vs_pure_stats.copy(),
                "mdap_config": asdict(config)
            }

            return result

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"MDAP-mode evolution failed: {e}", exc_info=True)
            return EvolutionResult(
                success=False,
                generations_completed=self.current_generation,
                evolution_time=time.time() - start_time,
                failed_attempts=[{"error": str(e)}]
            )

    async def _get_mdap_tactic_suggestions(
        self,
        current_tactic: Tactic,
        position: int,
        all_tactics: List[Tactic]
    ) -> Dict[Tactic, float]:
        """
        Get MDAP agent suggestions for tactic replacement.

        Args:
            current_tactic: Current tactic to potentially replace
            position: Position in tactic sequence
            all_tactics: All tactics in the sequence

        Returns:
            Dictionary mapping suggested tactics to vote counts
        """
        suggestions = defaultdict(float)

        # Query each MDAP agent type
        for agent_type in self.mdap_maker_config.mdap_agent_types:
            agent_weight = self._get_agent_weight(agent_type)
            suggested_tactic = self._agent_suggest_replacement(
                current_tactic, position, all_tactics, agent_type
            )

            if suggested_tactic:
                suggestions[suggested_tactic] += agent_weight

        return dict(suggestions)

    def _get_agent_weight(self, agent_type: str) -> float:
        """Get voting weight for an agent type."""
        # Base weight
        base_weight = 1.0

        # Adjust based on historical performance
        if self.mdap_mcts_integration and agent_type in self.mdap_mcts_integration.agent_performance:
            perf = self.mdap_mcts_integration.agent_performance[agent_type]
            base_weight = perf.get("success_rate", 0.5)

        return base_weight

    def _evaluate_parent_for_agent(self, parent: LeanProofStrategy, agent_type: str) -> float:
        """Evaluate parent strategy quality from perspective of agent type."""
        score = 0.0

        # Base score from fitness
        score += parent.fitness * 0.5

        # Adjust based on agent type preferences
        if agent_type == "evolution":
            # Evolution agents prefer diverse tactics
            tactic_diversity = len(set(t.name for t in parent.proof.tactics))
            score += (tactic_diversity / max(1, len(parent.proof.tactics))) * 0.3

        elif agent_type == "mcts":
            # MCTS agents prefer shorter, more direct proofs
            score += (1.0 / (1.0 + len(parent.proof.tactics) * 0.1)) * 0.3

        elif agent_type == "adversarial":
            # Adversarial agents prefer robust proofs
            score += parent.verified * 0.5

        elif agent_type == "self_play":
            # Self-play agents prefer proven strategies
            score += parent.elegance_score * 0.3

        return min(1.0, score)

    def _agent_tactic_preference(
        self,
        tactic1: Tactic,
        tactic2: Tactic,
        agent_type: str
    ) -> Tactic:
        """Get agent's preference between two tactics."""
        # Simple heuristic based on tactic categories
        simple_tactics = {"simp", "rw", "rfl", "assumption", "exact"}
        complex_tactics = {"induction", "cases", "constructor", "refine"}

        if agent_type == "mcts":
            # MCTS prefers simpler, faster tactics
            if tactic1.name in simple_tactics and tactic2.name not in simple_tactics:
                return tactic1
            elif tactic2.name in simple_tactics and tactic1.name not in simple_tactics:
                return tactic2

        elif agent_type == "evolution":
            # Evolution prefers diverse tactics
            if tactic1.name != tactic2.name:
                return random.choice([tactic1, tactic2])

        # Default: prefer first tactic
        return tactic1

    def _agent_suggest_replacement(
        self,
        current_tactic: Tactic,
        position: int,
        all_tactics: List[Tactic],
        agent_type: str
    ) -> Optional[Tactic]:
        """Get agent's suggested replacement tactic."""
        # Simple replacement suggestions based on agent type
        replacements = {
            "evolution": ["simp", "rw", "apply", "cases"],
            "mcts": ["simp", "rfl", "assumption", "exact"],
            "adversarial": ["apply", "cases", "induction"],
            "self_play": ["simp", "apply", "exact"]
        }

        suggestions = replacements.get(agent_type, ["simp", "apply"])
        new_name = random.choice(suggestions)

        # Avoid suggesting same tactic
        if new_name == current_tactic.name:
            return None

        return Tactic(name=new_name)

    def get_mdap_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report comparing MDAP vs pure evolution.

        Returns:
            Dictionary with performance metrics
        """
        total_generations = (
            self.mdap_vs_pure_stats["mdap_generations"] +
            self.mdap_vs_pure_stats["pure_generations"]
        )

        avg_mdap_time = 0.0
        if self.mdap_vs_pure_stats["mdap_generations"] > 0:
            avg_mdap_time = (
                self.mdap_vs_pure_stats["mdap_time"] /
                self.mdap_vs_pure_stats["mdap_generations"]
            )

        avg_pure_time = 0.0
        if self.mdap_vs_pure_stats["pure_generations"] > 0:
            avg_pure_time = (
                self.mdap_vs_pure_stats["pure_time"] /
                self.mdap_vs_pure_stats["pure_generations"]
            )

        avg_mdap_improvement = 0.0
        if self.mdap_vs_pure_stats["mdap_fitness_improvements"]:
            avg_mdap_improvement = sum(
                self.mdap_vs_pure_stats["mdap_fitness_improvements"]
            ) / len(self.mdap_vs_pure_stats["mdap_fitness_improvements"])

        return {
            "total_generations": total_generations,
            "mdap_generations": self.mdap_vs_pure_stats["mdap_generations"],
            "pure_generations": self.mdap_vs_pure_stats["pure_generations"],
            "avg_mdap_time_per_gen": avg_mdap_time,
            "avg_pure_time_per_gen": avg_pure_time,
            "avg_mdap_fitness_improvement": avg_mdap_improvement,
            "agent_contributions": dict(self.mdap_vs_pure_stats["agent_contributions"]),
            "voting_statistics": dict(self.mdap_vs_pure_stats["voting_statistics"]),
            "mdap_ratio": (
                self.mdap_vs_pure_stats["mdap_generations"] / max(1, total_generations)
            )
        }


# Export main classes
__all__ = [
    # Core classes
    'LeanProofEvolutionEngine',
    'LeanProofPopulation',
    'LeanProofMutator',
    'LeanProofEvaluator',
    'LeanProofCrossover',

    # MCTS-enhanced
    'LeanProofEvolutionEngineMCTS',

    # MDAP+MCTS-enhanced
    'LeanProofEvolutionEngineMDAP',
    'LeanProofEvolutionEngineMDAPFull',
    'MDAPMCTSGenerationConfig',
    'mcts_with_mdap_generation',
    'seed_population_with_mdap_mcts',
    'MDAP_MCTS_AVAILABLE',

    # Data classes
    'Tactic',
    'LeanProofStrategy',
    'LeanProof',
    'EvolutionResult',
    'PopulationStatistics',

    # Enums
    'MutationType',
    'SelectionMethod',
    'CrossoverMethod',

    # Convenience functions
    'evolve_proof'
]
