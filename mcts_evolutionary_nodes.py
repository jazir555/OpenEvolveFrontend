"""
Evolutionary Monte Carlo Tree Search (Evolutionary MCTS)

This module implements "Evolutionary Monte Carlo Tree Search" where each MCTS node
uses evolutionary algorithms for richer exploration than simple random rollouts.

The key innovation is that instead of doing random rollouts during the simulation
phase, each MCTS node maintains an evolutionary population of action sequences
and evolves them to estimate node values more accurately.

Core Components:
    EvolutionaryNode: MCTS node with evolutionary population
    ActionSequence: Genome representation (action sequences)
    SequenceCrossover: Crossover operators for sequences
    SequenceMutation: Mutation operators for sequences
    SequenceSelection: Selection operators for evolutionary algorithm
    SequenceEvaluator: Fitness evaluation for sequences
    EvolutionaryMCTS: Main MCTS with evolutionary simulations
    AdaptiveEvolutionController: Dynamic control of evolution parameters
    DistributedEvolutionaryMCTS: Parallel evolution support

Author: OpenEvolve
Created: 2025-12-30
"""

import asyncio
import json
import logging
import math
import random
import time
import uuid
import hashlib
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union, TypeVar
)
import sqlite3
from pathlib import Path

# Import MCTS components
try:
    from leanaide_mcts import (
        MCTSNode,
        MCTSTree,
        ProofState,
        Tactic,
        MCTSConfig,
        MCTSResult,
        RolloutPolicy,
        MCTSSelection,
        MCTSExpansion,
        MCTSSimulation,
        MCTSBackpropagation,
    )
    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False
    logging.warning("MCTS components not available")

# Import LeanAide client
try:
    from leanaide_client import LeanAideClient
    from leanaide_evolution import (
        LeanProof,
        LeanProofStrategy,
        MutationType,
        SelectionMethod,
        CrossoverMethod,
    )
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logging.warning("LeanAide integration not available")

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

T = TypeVar('T')


# =============================================================================
# Action Sequence Representation (Genome)
# =============================================================================

@dataclass
class ActionSequence:
    """
    Represents a sequence of tactics (genome for evolution).

    An action sequence is the basic unit of evolution in the evolutionary MCTS.
    It represents a partial proof path from a node to some descendant state.
    """

    actions: List[Tactic]
    fitness: float = 0.0
    depth: int = 0
    proof_complete: bool = False
    sequence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    creation_time: float = field(default_factory=time.time)
    evaluation_count: int = 0
    valid: bool = True

    def length(self) -> int:
        """Return sequence length."""
        return len(self.actions)

    def is_valid(self) -> bool:
        """Check if sequence is valid."""
        return self.valid and len(self.actions) > 0

    def copy(self) -> 'ActionSequence':
        """Create deep copy."""
        return ActionSequence(
            actions=[t for t in self.actions],
            fitness=self.fitness,
            depth=self.depth,
            proof_complete=self.proof_complete,
            parent_ids=self.parent_ids.copy(),
            generation=self.generation,
            evaluation_count=self.evaluation_count,
            valid=self.valid
        )

    def to_string(self) -> str:
        """Convert to Lean code string."""
        return "\n  ".join(str(tactic) for tactic in self.actions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sequence_id": self.sequence_id,
            "actions": [t.to_dict() for t in self.actions],
            "fitness": self.fitness,
            "depth": self.depth,
            "proof_complete": self.proof_complete,
            "parent_ids": self.parent_ids,
            "generation": self.generation,
            "creation_time": self.creation_time,
            "evaluation_count": self.evaluation_count,
            "valid": self.valid,
            "lean_code": self.to_string()
        }

    def calculate_hash(self) -> str:
        """Calculate hash of sequence for caching."""
        tactic_str = ",".join(f"{t.name}:{t.arguments}" for t in self.actions)
        return hashlib.md5(tactic_str.encode()).hexdigest()[:16]


@dataclass
class ProofContext:
    """
    Context for proof evaluation.

    Contains information needed to evaluate action sequences.
    """

    theorem: str
    goals: List[str]
    hypotheses: List[str]
    available_tactics: List[str]
    depth_limit: int = 100
    current_depth: int = 0

    def copy(self) -> 'ProofContext':
        """Create deep copy."""
        return ProofContext(
            theorem=self.theorem,
            goals=self.goals.copy(),
            hypotheses=self.hypotheses.copy(),
            available_tactics=self.available_tactics.copy(),
            depth_limit=self.depth_limit,
            current_depth=self.current_depth
        )


# =============================================================================
# Evolutionary Node State
# =============================================================================

class EvolutionaryNode(MCTSNode):
    """
    MCTS node that maintains an evolutionary population.

    Each node stores a population of action sequences that are evolved
    to provide better value estimates than simple rollouts.
    """

    def __init__(
        self,
        state: ProofState,
        parent: Optional['EvolutionaryNode'] = None,
        action: Optional[str] = None,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_count: int = 2
    ):
        """
        Initialize an evolutionary MCTS node.

        Args:
            state: Proof state at this node
            parent: Parent node
            action: Action that led to this node
            population_size: Size of evolutionary population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_count: Number of elites to preserve each generation
        """
        # Initialize base MCTS node
        super().__init__(state, parent, action)

        # Evolutionary state
        self.rollout_population: List[ActionSequence] = []
        self.population_fitness: List[float] = []
        self.population_generation: int = 0

        # Evolution parameters
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count

        # Performance tracking
        self.best_sequence: Optional[ActionSequence] = None
        self.best_fitness: float = 0.0
        self.convergence_history: List[float] = []

        # Metadata
        self.evolution_initialized: bool = False
        self.total_evolutions: int = 0
        self.evolution_time: float = 0.0

    def is_population_converged(self, threshold: float = 0.95) -> bool:
        """
        Check if population has converged.

        Convergence is measured by:
        1. Low variance in fitness
        2. Little improvement over recent generations

        Args:
            threshold: Convergence threshold (0-1)

        Returns:
            True if population is converged
        """
        if len(self.convergence_history) < 3:
            return False

        # Check if fitness has plateaued
        recent_fitness = self.convergence_history[-5:]
        if len(recent_fitness) < 2:
            return False

        # Calculate variance
        avg = sum(recent_fitness) / len(recent_fitness)
        variance = sum((f - avg) ** 2 for f in recent_fitness) / len(recent_fitness)

        # Converged if variance is low
        return variance < (1 - threshold)

    def evolve_population(self, generations: int = 1) -> float:
        """
        Evolve population for N generations.

        This is a placeholder - actual evolution is handled by
        the EvolutionaryMCTS class which has access to operators.

        Args:
            generations: Number of generations to evolve

        Returns:
            Best fitness after evolution
        """
        # This method is called by EvolutionaryMCTS
        # The actual evolution logic is in the main MCTS class
        self.population_generation += generations
        return self.best_fitness

    def update_population(self, population: List[ActionSequence]) -> None:
        """
        Update the node's population.

        Args:
            population: New population to store
        """
        self.rollout_population = population
        self.population_fitness = [seq.fitness for seq in population]

        # Update best sequence
        if population:
            best = max(population, key=lambda s: s.fitness)
            if best.fitness > self.best_fitness:
                self.best_fitness = best.fitness
                self.best_sequence = best

        # Track convergence
        if self.population_fitness:
            avg_fitness = sum(self.population_fitness) / len(self.population_fitness)
            self.convergence_history.append(avg_fitness)

    def get_population_diversity(self) -> float:
        """
        Calculate population diversity using sequence variation.

        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if len(self.rollout_population) < 2:
            return 0.0

        total_distance = 0.0
        comparisons = 0

        for i, seq1 in enumerate(self.rollout_population):
            for seq2 in self.rollout_population[i+1:]:
                # Calculate distance based on action sequences
                distance = self._sequence_distance(seq1, seq2)
                total_distance += distance
                comparisons += 1

        return total_distance / max(1, comparisons)

    def _sequence_distance(self, seq1: ActionSequence, seq2: ActionSequence) -> float:
        """Calculate normalized distance between two sequences."""
        actions1 = [a.name for a in seq1.actions]
        actions2 = [a.name for a in seq2.actions]

        # Simple edit distance
        m, n = len(actions1), len(actions2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if actions1[i-1] == actions2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        max_len = max(m, n)
        return dp[m][n] / max(1, max_len)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "population_size": len(self.rollout_population),
            "population_generation": self.population_generation,
            "best_fitness": self.best_fitness,
            "best_sequence": self.best_sequence.to_dict() if self.best_sequence else None,
            "convergence_history": self.convergence_history[-10:],  # Last 10 entries
            "evolution_initialized": self.evolution_initialized,
            "total_evolutions": self.total_evolutions,
            "population_diversity": self.get_population_diversity()
        })
        return base_dict


# =============================================================================
# Evolutionary Operators for Sequences
# =============================================================================

class SequenceCrossover:
    """
    Crossover operators for action sequences.

    Implements various crossover strategies for combining parent sequences
    to create offspring sequences.
    """

    def __init__(self, context_aware: bool = True):
        """
        Initialize crossover operators.

        Args:
            context_aware: Use context-aware crossover that respects tactic boundaries
        """
        self.context_aware = context_aware

    def one_point_crossover(
        self,
        parent1: ActionSequence,
        parent2: ActionSequence
    ) -> Tuple[ActionSequence, ActionSequence]:
        """
        Single-point crossover.

        Select a random crossover point and swap tails after that point.

        Args:
            parent1: First parent sequence
            parent2: Second parent sequence

        Returns:
            Two offspring sequences
        """
        len1, len2 = len(parent1.actions), len(parent2.actions)

        # Select crossover point with bounds checking
        if len1 < 2 or len2 < 2:
            # Return clones if arrays are too small for crossover
            return parent1.clone(), parent2.clone()
        
        point1 = random.randint(1, max(1, len1 - 1))
        point2 = random.randint(1, max(1, len2 - 1))

        # Create offspring
        offspring1_actions = parent1.actions[:point1] + parent2.actions[point2:]
        offspring2_actions = parent2.actions[:point2] + parent1.actions[point1:]

        offspring1 = ActionSequence(
            actions=offspring1_actions,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.sequence_id, parent2.sequence_id]
        )

        offspring2 = ActionSequence(
            actions=offspring2_actions,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.sequence_id, parent2.sequence_id]
        )

        return offspring1, offspring2

    def uniform_crossover(
        self,
        parent1: ActionSequence,
        parent2: ActionSequence
    ) -> Tuple[ActionSequence, ActionSequence]:
        """
        Uniform crossover.

        Randomly select from each parent at each position.

        Args:
            parent1: First parent sequence
            parent2: Second parent sequence

        Returns:
            Two offspring sequences
        """
        max_len = max(len(parent1.actions), len(parent2.actions))

        offspring1_actions = []
        offspring2_actions = []

        for i in range(max_len):
            # Randomly choose which parent contributes to which offspring
            if random.random() < 0.5:
                offspring1_actions.append(parent1.actions[i] if i < len(parent1.actions) else parent2.actions[i])
                offspring2_actions.append(parent2.actions[i] if i < len(parent2.actions) else parent1.actions[i])
            else:
                offspring1_actions.append(parent2.actions[i] if i < len(parent2.actions) else parent1.actions[i])
                offspring2_actions.append(parent1.actions[i] if i < len(parent1.actions) else parent2.actions[i])

        offspring1 = ActionSequence(
            actions=offspring1_actions,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.sequence_id, parent2.sequence_id]
        )

        offspring2 = ActionSequence(
            actions=offspring2_actions,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.sequence_id, parent2.sequence_id]
        )

        return offspring1, offspring2

    def context_aware_crossover(
        self,
        parent1: ActionSequence,
        parent2: ActionSequence,
        context: ProofContext
    ) -> Tuple[ActionSequence, ActionSequence]:
        """
        Context-aware crossover that respects proof structure.

        Prefer crossover at tactic boundaries and maintain semantic coherence.

        Args:
            parent1: First parent sequence
            parent2: Second parent sequence
            context: Proof context for semantic awareness

        Returns:
            Two offspring sequences
        """
        # Find good crossover points (tactic boundaries)
        points1 = self._find_crossover_points(parent1, context)
        points2 = self._find_crossover_points(parent2, context)

        if not points1 or not points2:
            # Fallback to one-point crossover
            return self.one_point_crossover(parent1, parent2)

        # Select crossover points
        point1 = random.choice(points1)
        point2 = random.choice(points2)

        # Create offspring
        offspring1_actions = parent1.actions[:point1] + parent2.actions[point2:]
        offspring2_actions = parent2.actions[:point2] + parent1.actions[point1:]

        offspring1 = ActionSequence(
            actions=offspring1_actions,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.sequence_id, parent2.sequence_id]
        )

        offspring2 = ActionSequence(
            actions=offspring2_actions,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.sequence_id, parent2.sequence_id]
        )

        return offspring1, offspring2

    def _find_crossover_points(
        self,
        sequence: ActionSequence,
        context: ProofContext
    ) -> List[int]:
        """
        Find good crossover points in a sequence.

        Good points are after tactics that complete logical steps.

        Args:
            sequence: Action sequence
            context: Proof context

        Returns:
            List of valid crossover points
        """
        points = []

        # Tactic boundaries (after certain tactics)
        boundary_tactics = ["simp", "rw", "apply", "exact", "constructor"]

        for i, tactic in enumerate(sequence.actions):
            # Prefer crossover after boundary tactics
            if tactic.name in boundary_tactics:
                points.append(i + 1)

            # Also consider points before induction/cases
            if tactic.name in ["induction", "cases"]:
                points.append(i)

        # Always include midpoint
        if len(sequence.actions) > 2:
            mid = len(sequence.actions) // 2
            if mid not in points:
                points.append(mid)

        return points


class SequenceMutation:
    """
    Mutation operators for action sequences.

    Implements various mutation strategies to introduce variation
    into the population.
    """

    # Common Lean 4 tactics
    BASIC_TACTICS = [
        "intros", "simp", "rw", "apply", "exact", "refine",
        "cases", "induction", "constructor", "exists",
        "have", "suffices", "show", "calc",
        "aesop", "linarith", "ring", "omega", "norm_num",
        "trivial", "decide", "done"
    ]

    def __init__(self, available_tactics: Optional[List[str]] = None):
        """
        Initialize mutation operators.

        Args:
            available_tactics: List of available tactics (uses BASIC_TACTICS if None)
        """
        self.available_tactics = available_tactics or self.BASIC_TACTICS

    def tactic_substitution(
        self,
        sequence: ActionSequence,
        available_tactics: Optional[List[str]] = None
    ) -> ActionSequence:
        """
        Replace a random tactic with another.

        Args:
            sequence: Sequence to mutate
            available_tactics: Optional list of tactics to choose from

        Returns:
            Mutated sequence
        """
        if not sequence.actions:
            return sequence

        tactics = available_tactics or self.available_tactics
        mutated = sequence.copy()

        # Select random position and new tactic
        if not mutated.actions:
            return sequence
        pos = random.randint(0, len(mutated.actions) - 1)
        new_tactic = Tactic(name=random.choice(tactics))

        mutated.actions[pos] = new_tactic
        mutated.parent_ids = [sequence.sequence_id]

        return mutated

    def tactic_insertion(
        self,
        sequence: ActionSequence,
        available_tactics: Optional[List[str]] = None
    ) -> ActionSequence:
        """
        Insert a new tactic at random position.

        Args:
            sequence: Sequence to mutate
            available_tactics: Optional list of tactics to choose from

        Returns:
            Mutated sequence
        """
        tactics = available_tactics or self.available_tactics
        mutated = sequence.copy()

        # Select random position and tactic
        pos = random.randint(0, len(mutated.actions)) if mutated.actions else 0
        new_tactic = Tactic(name=random.choice(tactics))

        mutated.actions.insert(pos, new_tactic)
        mutated.parent_ids = [sequence.sequence_id]

        return mutated

    def tactic_deletion(self, sequence: ActionSequence) -> ActionSequence:
        """
        Delete a random tactic.

        Args:
            sequence: Sequence to mutate

        Returns:
            Mutated sequence
        """
        if len(sequence.actions) <= 1:
            return sequence

        mutated = sequence.copy()

        # Select random position to delete
        if not mutated.actions:
            return sequence
        pos = random.randint(0, len(mutated.actions) - 1)
        mutated.actions.pop(pos)

        mutated.parent_ids = [sequence.sequence_id]

        return mutated

    def subsequence_reorder(self, sequence: ActionSequence) -> ActionSequence:
        """
        Reorder a subsequence.

        Selects a random subsequence and reverses or shuffles it.

        Args:
            sequence: Sequence to mutate

        Returns:
            Mutated sequence
        """
        if len(sequence.actions) < 2:
            return sequence

        mutated = sequence.copy()

        # Select random subsequence
        if len(mutated.actions) < 2:
            return sequence
        start = random.randint(0, len(mutated.actions) - 2)
        end = random.randint(start + 1, len(mutated.actions) - 1)

        # Reverse subsequence
        mutated.actions[start:end+1] = reversed(mutated.actions[start:end+1])

        mutated.parent_ids = [sequence.sequence_id]

        return mutated

    def lemma_injection(
        self,
        sequence: ActionSequence,
        lemmas: List[str]
    ) -> ActionSequence:
        """
        Inject lemma application.

        Adds a 'have' tactic to introduce a lemma.

        Args:
            sequence: Sequence to mutate
            lemmas: List of available lemmas

        Returns:
            Mutated sequence
        """
        if not lemmas or not sequence.actions:
            return sequence

        mutated = sequence.copy()

        # Select random lemma and position
        lemma = random.choice(lemmas)
        pos = random.randint(0, len(mutated.actions))

        # Create 'have' tactic
        have_tactic = Tactic(
            name="have",
            arguments=[lemma]
        )

        mutated.actions.insert(pos, have_tactic)
        mutated.parent_ids = [sequence.sequence_id]

        return mutated

    def adaptive_mutation(
        self,
        sequence: ActionSequence,
        mutation_rate: float,
        available_tactics: Optional[List[str]] = None
    ) -> ActionSequence:
        """
        Apply multiple mutations with adaptive rate.

        Args:
            sequence: Sequence to mutate
            mutation_rate: Probability of mutation at each position
            available_tactics: Optional list of tactics

        Returns:
            Mutated sequence
        """
        mutated = sequence.copy()
        mutated.parent_ids = [sequence.sequence_id]

        for _ in range(len(mutated.actions)):
            if random.random() < mutation_rate:
                # Apply random mutation
                mutation_type = random.choice([
                    'substitution', 'insertion', 'deletion', 'reorder'
                ])

                if mutation_type == 'substitution':
                    mutated = self.tactic_substitution(mutated, available_tactics)
                elif mutation_type == 'insertion':
                    mutated = self.tactic_insertion(mutated, available_tactics)
                elif mutation_type == 'deletion':
                    mutated = self.tactic_deletion(mutated)
                elif mutation_type == 'reorder':
                    mutated = self.subsequence_reorder(mutated)

        return mutated


class SequenceSelection:
    """
    Selection operators for action sequences.

    Implements various parent selection strategies for the evolutionary algorithm.
    """

    def tournament_selection(
        self,
        population: List[ActionSequence],
        tournament_size: int = 3
    ) -> ActionSequence:
        """
        Select using tournament selection.

        Args:
            population: Population to select from
            tournament_size: Number of individuals in tournament

        Returns:
            Selected sequence
        """
        if not population:
            raise ValueError("Cannot select from empty population")

        tournament = random.sample(
            population,
            min(tournament_size, len(population))
        )

        return max(tournament, key=lambda s: s.fitness)

    def fitness_proportionate_selection(
        self,
        population: List[ActionSequence]
    ) -> ActionSequence:
        """
        Roulette wheel selection.

        Args:
            population: Population to select from

        Returns:
            Selected sequence
        """
        if not population:
            raise ValueError("Cannot select from empty population")

        # Shift fitness to be positive
        min_fitness = min(s.fitness for s in population)
        adjusted_fitness = [s.fitness - min_fitness + 0.001 for s in population]
        total_fitness = sum(adjusted_fitness)

        if total_fitness == 0:
            # Fix: Check if population is empty before random.choice
            if not population:
                raise ValueError("Cannot select from empty population: no individuals available")
            return random.choice(population)

        # Calculate selection probabilities
        probabilities = [f / total_fitness for f in adjusted_fitness]

        # Fix: Check population not empty before random.choices
        if not population:
            raise ValueError("Cannot select from empty population: no individuals available")
        return random.choices(population, weights=probabilities, k=1)[0]

    def rank_selection(self, population: List[ActionSequence]) -> ActionSequence:
        """
        Select based on rank, not absolute fitness.

        Args:
            population: Population to select from

        Returns:
            Selected sequence
        """
        if not population:
            raise ValueError("Cannot select from empty population")

        # Sort by fitness
        sorted_pop = sorted(population, key=lambda s: s.fitness)

        # Assign ranks
        ranks = list(range(1, len(sorted_pop) + 1))
        total_rank = sum(ranks)

        probabilities = [r / total_rank for r in ranks]

        return random.choices(sorted_pop, weights=probabilities, k=1)[0]

    def elitist_selection(
        self,
        population: List[ActionSequence],
        elite_count: int
    ) -> List[ActionSequence]:
        """
        Select top performers.

        Args:
            population: Population to select from
            elite_count: Number of elites to select

        Returns:
            List of elite sequences
        """
        sorted_pop = sorted(population, key=lambda s: s.fitness, reverse=True)
        return sorted_pop[:elite_count]


# =============================================================================
# Fitness Evaluation
# =============================================================================

class SequenceEvaluator:
    """
    Evaluate action sequences for fitness.

    Fitness is based on:
    1. Proof completeness (did we solve it?)
    2. Goal proximity (how close to goal?)
    3. Tactic quality (are tactics appropriate?)
    4. Depth efficiency (not too long?)
    5. Semantic validity (does it make sense?)
    """

    def __init__(self, leanaide_client: Optional[LeanAideClient] = None):
        """
        Initialize evaluator.

        Args:
            leanaide_client: Optional LeanAide client for formal verification
        """
        self.leanaide_client = leanaide_client

    def evaluate(
        self,
        sequence: ActionSequence,
        context: ProofContext
    ) -> float:
        """
        Evaluate sequence quality.

        Args:
            sequence: Action sequence to evaluate
            context: Proof context

        Returns:
            Fitness score (0-1, higher is better)
        """
        # Factors
        completeness = self._evaluate_completeness(sequence, context)
        proximity = self._evaluate_goal_proximity(sequence, context)
        quality = self._evaluate_tactic_quality(sequence, context)
        efficiency = self._evaluate_efficiency(sequence, context)

        # Weighted combination
        fitness = (
            0.4 * completeness +
            0.3 * proximity +
            0.2 * quality +
            0.1 * efficiency
        )

        sequence.fitness = fitness
        sequence.evaluation_count += 1

        return fitness

    def _evaluate_completeness(
        self,
        sequence: ActionSequence,
        context: ProofContext
    ) -> float:
        """Evaluate proof completeness."""
        if sequence.proof_complete:
            return 1.0

        # Heuristic: check if all goals are resolved
        # In real implementation, would verify with Lean
        if not context.goals:
            return 1.0

        return 0.0

    def _evaluate_goal_proximity(
        self,
        sequence: ActionSequence,
        context: ProofContext
    ) -> float:
        """Evaluate how close we are to solving goals."""
        # Heuristic: count goals resolved
        initial_goals = len(context.goals)

        if initial_goals == 0:
            return 1.0

        # Estimate goal reduction
        goals_remaining = max(0, initial_goals - len(sequence.actions) // 2)
        reduction = (initial_goals - goals_remaining) / initial_goals

        return reduction

    def _evaluate_tactic_quality(
        self,
        sequence: ActionSequence,
        context: ProofContext
    ) -> float:
        """Evaluate quality of tactics used."""
        if not sequence.actions:
            return 0.0

        # Count high-quality tactics
        quality_tactics = ["simp", "rw", "apply", "exact", "aesop"]
        quality_count = sum(1 for t in sequence.actions if t.name in quality_tactics)

        return quality_count / len(sequence.actions)

    def _evaluate_efficiency(
        self,
        sequence: ActionSequence,
        context: ProofContext
    ) -> float:
        """Evaluate depth efficiency."""
        if not sequence.actions:
            return 0.0

        # Prefer shorter sequences
        ideal_length = 10
        actual_length = len(sequence.actions)

        if actual_length <= ideal_length:
            return 1.0

        # Penalize excessive length
        penalty = (actual_length - ideal_length) / ideal_length
        return max(0.0, 1.0 - penalty)

    async def evaluate_with_leanaide(
        self,
        sequence: ActionSequence,
        leanaide_client: LeanAideClient
    ) -> float:
        """
        Evaluate using LeanAide formal verification.

        Args:
            sequence: Action sequence to evaluate
            leanaide_client: LeanAide client

        Returns:
            Fitness score with formal verification bonus
        """
        # Base evaluation
        context = ProofContext(
            theorem="",  # Would be filled in
            goals=[],
            hypotheses=[],
            available_tactics=[]
        )
        base_fitness = self.evaluate(sequence, context)

        # Try formal verification
        if LEANAIDE_AVAILABLE and leanaide_client:
            try:
                # Convert to Lean code
                lean_code = sequence.to_string()

                # Verify with LeanAide
                result = await leanaide_client.elaborate(lean_code)

                if result.success:
                    # Bonus for verified proofs
                    base_fitness *= 1.5
                    sequence.proof_complete = True
                else:
                    # Penalty for errors
                    base_fitness *= 0.8

            except Exception as e:
                logger.warning(f"LeanAide verification failed: {e}")

        return min(1.0, base_fitness)


# =============================================================================
# Evolutionary MCTS
# =============================================================================

class EvolutionaryMCTS:
    """
    MCTS with evolutionary simulations at each node.

    Instead of random rollouts, each node evolves a population of
    action sequences to estimate node values more accurately.
    """

    def __init__(
        self,
        population_size: int = 20,
        evolution_generations: int = 5,
        exploration_constant: float = 1.414,
        mcts_simulations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_count: int = 2
    ):
        """
        Initialize Evolutionary MCTS.

        Args:
            population_size: Size of evolutionary population at each node
            evolution_generations: Number of generations to evolve per simulation
            exploration_constant: UCT exploration constant
            mcts_simulations: Number of MCTS simulations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_count: Number of elites to preserve
        """
        self.population_size = population_size
        self.evolution_generations = evolution_generations
        self.exploration_constant = exploration_constant
        self.mcts_simulations = mcts_simulations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count

        # Initialize operators
        self.crossover = SequenceCrossover(context_aware=True)
        self.mutation = SequenceMutation()
        self.selection = SequenceSelection()
        self.evaluator = SequenceEvaluator()

        # Statistics
        self.total_evolutions = 0
        self.total_evaluations = 0

    async def search(
        self,
        initial_context: ProofContext,
        leanaide_client: Optional[LeanAideClient] = None
    ) -> MCTSResult:
        """
        Search using evolutionary MCTS.

        Args:
            initial_context: Initial proof context
            leanaide_client: Optional LeanAide client for verification

        Returns:
            MCTSResult with best proof found
        """
        start_time = time.time()

        # Initialize root node
        initial_state = ProofState(
            goals=initial_context.goals,
            context=initial_context.hypotheses
        )

        root = EvolutionaryNode(
            state=initial_state,
            population_size=self.population_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            elite_count=self.elite_count
        )

        # Initialize root population
        self.initialize_node_population(root, initial_context)

        # Create tree
        tree = EvolutionaryTree(root)

        # Run MCTS iterations
        for i in range(self.mcts_simulations):
            # Selection: Select leaf using UCT
            leaf = self._select_leaf(root)

            # Expansion: Expand leaf if needed
            if not leaf.is_terminal:
                await self._expand_node(leaf, initial_context)

            # Evolutionary Simulation: Evolve population at leaf
            reward = await self.evolutionary_simulation(
                leaf,
                initial_context,
                self.evolution_generations
            )

            # Backpropagation: Update statistics
            self._backpropagate(leaf, reward)

        # Compile result
        elapsed = time.time() - start_time
        return self._compile_result(root, tree, elapsed)

    def initialize_node_population(
        self,
        node: EvolutionaryNode,
        context: ProofContext
    ) -> None:
        """
        Initialize evolutionary population at node.

        Args:
            node: Node to initialize
            context: Proof context
        """
        population = []

        # Generate random action sequences
        for _ in range(self.population_size):
            sequence = self._generate_random_sequence(context)
            sequence.fitness = self.evaluator.evaluate(sequence, context)
            population.append(sequence)

        # Store in node
        node.update_population(population)
        node.evolution_initialized = True

        self.total_evaluations += len(population)

    def _generate_random_sequence(
        self,
        context: ProofContext,
        max_length: int = 10
    ) -> ActionSequence:
        """Generate a random action sequence."""
        length = random.randint(1, max_length)
        actions = []

        for _ in range(length):
            tactic = Tactic(name=random.choice(context.available_tactics))
            actions.append(tactic)

        return ActionSequence(actions=actions, depth=length)

    async def evolutionary_simulation(
        self,
        node: EvolutionaryNode,
        context: ProofContext,
        generations: int
    ) -> float:
        """
        Run evolution at this node.

        Args:
            node: Node to evolve at
            context: Proof context
            generations: Number of generations

        Returns:
            Fitness of best individual
        """
        # Evolve population
        await self.evolve_at_node(node, context, generations)

        # Return best fitness
        return node.best_fitness

    async def evolve_at_node(
        self,
        node: EvolutionaryNode,
        context: ProofContext,
        generations: int
    ) -> None:
        """
        Evolve population at a node.

        Args:
            node: Node with population to evolve
            context: Proof context
            generations: Number of generations to run
        """
        self.total_evolutions += 1
        node.total_evolutions += 1

        for gen in range(generations):
            # 1. Selection
            parents = self._select_parents(node)

            # 2. Crossover
            offspring = await self._crossover_parents(parents, context)

            # 3. Mutation
            offspring = self._mutate_offspring(offspring, context)

            # 4. Evaluation
            for child in offspring:
                child.fitness = self.evaluator.evaluate(child, context)
                self.total_evaluations += 1

            # 5. Survival selection
            new_population = self._select_survivors(
                node.rollout_population,
                offspring
            )

            # Update node
            node.update_population(new_population)

            # Track convergence
            avg_fitness = sum(s.fitness for s in new_population) / len(new_population)
            node.convergence_history.append(avg_fitness)

    def _select_parents(
        self,
        node: EvolutionaryNode,
        num_parents: int = 10
    ) -> List[ActionSequence]:
        """Select parents for reproduction."""
        parents = []
        population = node.rollout_population

        for _ in range(num_parents):
            parent = self.selection.tournament_selection(
                population,
                tournament_size=3
            )
            parents.append(parent)

        return parents

    async def _crossover_parents(
        self,
        parents: List[ActionSequence],
        context: ProofContext
    ) -> List[ActionSequence]:
        """Perform crossover to create offspring."""
        offspring = []

        # Pair up parents
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            # Crossover with probability
            if random.random() < self.crossover_rate:
                # Use context-aware crossover
                child1, child2 = self.crossover.context_aware_crossover(
                    parent1,
                    parent2,
                    context
                )
                offspring.extend([child1, child2])
            else:
                # Parents survive unchanged
                offspring.extend([parent1.copy(), parent2.copy()])

        return offspring

    def _mutate_offspring(
        self,
        offspring: List[ActionSequence],
        context: ProofContext
    ) -> List[ActionSequence]:
        """Apply mutation to offspring."""
        mutated = []

        for child in offspring:
            if random.random() < self.mutation_rate:
                mutated_child = self.mutation.adaptive_mutation(
                    child,
                    self.mutation_rate,
                    context.available_tactics
                )
                mutated.append(mutated_child)
            else:
                mutated.append(child)

        return mutated

    def _select_survivors(
        self,
        current_population: List[ActionSequence],
        offspring: List[ActionSequence]
    ) -> List[ActionSequence]:
        """Select survivors for next generation."""
        combined = current_population + offspring

        # Elitism: keep best
        elites = self.selection.elitist_selection(
            combined,
            self.elite_count
        )

        # Select rest via tournament
        remaining_count = self.population_size - self.elite_count
        survivors = elites.copy()

        while len(survivors) < self.population_size:
            survivor = self.selection.tournament_selection(
                combined,
                tournament_size=3
            )
            survivors.append(survivor)

        return survivors[:self.population_size]

    def _select_leaf(self, root: EvolutionaryNode) -> EvolutionaryNode:
        """Select leaf node using UCT."""
        current = root

        while current.children and not current.is_terminal:
            # Select best child using UCT
            current = current.best_child(self.exploration_constant)

        return current

    async def _expand_node(
        self,
        node: EvolutionaryNode,
        context: ProofContext
    ) -> None:
        """Expand node by creating children."""
        if node.is_terminal or node.is_fully_expanded_node():
            return

        # Get untried actions
        if not node.untried_actions:
            node.untried_actions = context.available_tactics[:10]

        if not node.untried_actions:
            node.is_terminal = True
            return

        # Create child node
        action = node.untried_actions.pop(0)

        # Apply action to get new state
        new_state = self._apply_action(node.state, action)

        # Create child node
        child = EvolutionaryNode(
            state=new_state,
            parent=node,
            action=action,
            population_size=self.population_size
        )

        # Add to tree
        node.add_child(action, child)

    def _apply_action(
        self,
        state: ProofState,
        action: str
    ) -> ProofState:
        """Apply action to get new state (simplified)."""
        new_state = ProofState(
            goals=state.goals.copy(),
            context=state.context.copy(),
            tactics_sequence=state.tactics_sequence.copy(),
            depth=state.depth + 1
        )

        # Add tactic
        tactic = Tactic(name=action)
        new_state.tactics_sequence.append(tactic)

        # Simulate goal reduction
        if action in ["simp", "aesop"] and new_state.goals:
            new_state.goals = new_state.goals[:-1]

        new_state.is_complete = len(new_state.goals) == 0

        return new_state

    def _backpropagate(
        self,
        node: EvolutionaryNode,
        reward: float
    ) -> None:
        """Backpropagate reward up the tree."""
        current = node

        while current is not None:
            current.N += 1
            current.W += reward
            current.Q = current.W / current.N
            current = current.parent

    def _compile_result(
        self,
        root: EvolutionaryNode,
        tree: 'EvolutionaryTree',
        elapsed: float
    ) -> MCTSResult:
        """Compile final result."""
        # Get best path
        best_path = tree.get_best_path()

        # Create proof
        best_proof = self._create_proof_from_path(best_path)

        return MCTSResult(
            best_proof=best_proof,
            success=best_proof is not None and best_path[-1].is_terminal,
            search_iterations=self.mcts_simulations,
            time_elapsed=elapsed,
            nodes_visited=tree.total_nodes,
            tree_depth=tree.max_depth,
            win_rate=root.best_fitness,
            confidence=root.Q,
            proof_path=best_path,
            search_statistics={
                "total_evolutions": self.total_evolutions,
                "total_evaluations": self.total_evaluations,
                "root_visits": root.N,
                "root_value": root.Q
            },
            tree_statistics=tree.get_statistics()
        )

    def _create_proof_from_path(self, path: List[EvolutionaryNode]) -> Optional[LeanProof]:
        """Create a LeanProof from a path."""
        if not path:
            return None

        # Collect all tactics
        all_tactics = []
        for node in path:
            all_tactics.extend(node.state.tactics_sequence)

        # Generate Lean code
        lean_code = "\n  ".join(str(t) for t in all_tactics)

        return LeanProof(
            theorem_name="evolutionary_mcts_proof",
            theorem_statement="",
            lean_code=lean_code,
            tactics=all_tactics
        )


# =============================================================================
# Evolutionary Tree
# =============================================================================

class EvolutionaryTree(MCTSTree):
    """
    Manages the Evolutionary MCTS search tree.

    Extends MCTSTree with evolutionary-specific statistics.
    """

    def __init__(self, root: EvolutionaryNode):
        """Initialize evolutionary tree."""
        super().__init__(root)
        self.root = root

    def get_best_path(self) -> List[EvolutionaryNode]:
        """Get best path from root to leaf."""
        path = [self.root]
        current = self.root

        while current.children:
            current = current.best_child(c_param=0.0)
            path.append(current)

        return path

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tree statistics."""
        base_stats = super().get_statistics()

        # Add evolutionary-specific stats
        total_population_size = 0
        total_evolutions = 0

        queue = deque([self.root])
        visited = set()

        while queue:
            node = queue.popleft()
            if node.hash in visited:
                continue
            visited.add(node.hash)

            if isinstance(node, EvolutionaryNode):
                total_population_size += len(node.rollout_population)
                total_evolutions += node.total_evolutions

            queue.extend(node.children.values())

        base_stats.update({
            "total_population_size": total_population_size,
            "total_evolutions": total_evolutions,
            "avg_population_size": total_population_size / max(1, len(visited))
        })

        return base_stats


# =============================================================================
# Adaptive Evolution Control
# =============================================================================

class AdaptiveEvolutionController:
    """
    Control evolution parameters dynamically.

    Adjusts population size, generations, and mutation rates
    based on node importance, depth, and convergence.
    """

    def __init__(self):
        """Initialize adaptive controller."""
        self.node_performance: Dict[str, Dict] = {}

    def should_evolve_at_node(
        self,
        node: EvolutionaryNode,
        depth: int
    ) -> bool:
        """
        Decide whether to evolve at this node.

        Args:
            node: Node to check
            depth: Current depth in tree

        Returns:
            True if evolution should be performed
        """
        # Always evolve if not initialized
        if not node.evolution_initialized:
            return True

        # Skip if converged
        if node.is_population_converged():
            return False

        # More evolution at important nodes
        if node.N > 100:
            return True

        # Less evolution at deep nodes
        if depth > 20:
            return False

        return True

    def get_evolution_generations(
        self,
        node: EvolutionaryNode,
        depth: int
    ) -> int:
        """
        Determine how many generations to run.

        Args:
            node: Node to evolve at
            depth: Current depth

        Returns:
            Number of generations to run
        """
        base_generations = 5

        # More generations at highly visited nodes
        if node.N > 100:
            base_generations += 3

        # More generations at root
        if depth == 0:
            base_generations += 2

        # Fewer generations if converged
        if node.is_population_converged(threshold=0.8):
            base_generations = max(1, base_generations // 2)

        return base_generations

    def get_population_size(
        self,
        node: EvolutionaryNode,
        depth: int
    ) -> int:
        """
        Determine population size.

        Args:
            node: Node to evolve at
            depth: Current depth

        Returns:
            Population size to use
        """
        base_size = 20

        # Larger populations at important nodes
        if node.N > 100:
            base_size += 10

        # Larger populations at root
        if depth == 0:
            base_size += 10

        # Smaller populations at deep nodes
        if depth > 15:
            base_size = max(10, base_size // 2)

        return base_size

    def get_mutation_rate(
        self,
        node: EvolutionaryNode,
        generation: int
    ) -> float:
        """
        Determine mutation rate.

        Args:
            node: Node being evolved
            generation: Current generation number

        Returns:
            Mutation rate to use
        """
        base_rate = 0.1

        # Increase mutation if population is stagnant
        if len(node.convergence_history) > 5:
            recent = node.convergence_history[-5:]
            if max(recent) - min(recent) < 0.05:
                base_rate *= 1.5  # Increase mutation

        # Decrease mutation over time
        decay = 0.99 ** generation
        base_rate *= decay

        return min(0.5, max(0.01, base_rate))


# =============================================================================
# Distributed Evolution
# =============================================================================

class DistributedEvolutionaryMCTS:
    """
    Parallel evolution at multiple nodes.

    Distributes evolutionary computation across multiple workers
    for faster proof search.
    """

    def __init__(
        self,
        base_mcts: EvolutionaryMCTS,
        max_workers: int = 4
    ):
        """
        Initialize distributed evolutionary MCTS.

        Args:
            base_mcts: Base evolutionary MCTS instance
            max_workers: Maximum number of parallel workers
        """
        self.base_mcts = base_mcts
        self.max_workers = max_workers

    async def distributed_search(
        self,
        initial_context: ProofContext,
        max_workers: Optional[int] = None
    ) -> MCTSResult:
        """
        Search with parallel evolution.

        Args:
            initial_context: Initial proof context
            max_workers: Optional override for worker count

        Returns:
            MCTSResult with best proof
        """
        workers = max_workers or self.max_workers
        start_time = time.time()

        # Initialize root
        initial_state = ProofState(
            goals=initial_context.goals,
            context=initial_context.hypotheses
        )

        root = EvolutionaryNode(
            state=initial_state,
            population_size=self.base_mcts.population_size
        )

        self.base_mcts.initialize_node_population(root, initial_context)

        # Run MCTS iterations with parallel evolution
        for i in range(self.base_mcts.mcts_simulations):
            # Select nodes needing evolution
            nodes_to_evolve = self._identify_nodes_for_evolution(root)

            if not nodes_to_evolve:
                break

            # Distribute evolution tasks
            if len(nodes_to_evolve) > 1:
                await self._parallel_evolve(nodes_to_evolve, initial_context, workers)
            elif len(nodes_to_evolve) == 1:
                # Fix: Only access [0] if list has exactly 1 element
                await self.base_mcts.evolve_at_node(
                    nodes_to_evolve[0],
                    initial_context,
                    self.base_mcts.evolution_generations
                )
            # else: no nodes to evolve, skip

        # Compile result
        elapsed = time.time() - start_time
        tree = EvolutionaryTree(root)
        return self.base_mcts._compile_result(root, tree, elapsed)

    def _identify_nodes_for_evolution(
        self,
        root: EvolutionaryNode
    ) -> List[EvolutionaryNode]:
        """Identify nodes that need evolution."""
        nodes = []
        queue = [root]
        visited = set()

        controller = AdaptiveEvolutionController()

        while queue:
            node = queue.pop(0)

            if node.hash in visited:
                continue
            visited.add(node.hash)

            if controller.should_evolve_at_node(node, node.depth):
                nodes.append(node)

            queue.extend(node.children.values())

        return nodes

    async def _parallel_evolve(
        self,
        nodes: List[EvolutionaryNode],
        context: ProofContext,
        workers: int
    ) -> None:
        """Evolve multiple nodes in parallel."""
        tasks = []

        for node in nodes:
            task = self.base_mcts.evolve_at_node(
                node,
                context,
                self.base_mcts.evolution_generations
            )
            tasks.append(task)

        # Run in parallel with semaphore
        semaphore = asyncio.Semaphore(workers)

        async def bounded_task(task):
            async with semaphore:
                return await task

        await asyncio.gather(*[bounded_task(t) for t in tasks])


# =============================================================================
# LeanAide Integration
# =============================================================================

class EvolutionaryMCTSWithLeanAide(EvolutionaryMCTS):
    """
    Evolutionary MCTS with LeanAide verification.

    Uses LeanAide for formal verification during evolution,
    providing better fitness estimates and guiding search.
    """

    def __init__(
        self,
        leanaide_client: LeanAideClient,
        **kwargs
    ):
        """
        Initialize with LeanAide client.

        Args:
            leanaide_client: LeanAide client for verification
            **kwargs: Additional arguments for EvolutionaryMCTS
        """
        super().__init__(**kwargs)
        self.leanaide_client = leanaide_client

    async def search_with_verification(
        self,
        theorem: str,
        context: ProofContext
    ) -> MCTSResult:
        """
        Search with formal verification.

        Args:
            theorem: Theorem to prove
            context: Proof context

        Returns:
            MCTSResult with verified proof
        """
        # Enhanced evaluator with LeanAide
        self.evaluator = SequenceEvaluator(self.leanaide_client)

        # Run search
        result = await self.search(context, self.leanaide_client)

        # Verify best proof
        if result.best_proof:
            verified = await self._verify_proof(result.best_proof)

            if verified:
                result.success = True
                logger.info("Proof verified with LeanAide!")

        return result

    async def evolve_at_node(
        self,
        node: EvolutionaryNode,
        context: ProofContext,
        generations: int
    ) -> None:
        """
        Evolve population with LeanAide verification.

        Periodically verifies best sequences with Lean.
        """
        for gen in range(generations):
            # Standard evolution
            await super().evolve_at_node(node, context, 1)

            # Verify best sequence every few generations
            if gen % 3 == 0 and node.best_sequence:
                await self._verify_and_update(node, context)

    async def _verify_and_update(
        self,
        node: EvolutionaryNode,
        context: ProofContext
    ) -> None:
        """Verify best sequence and update fitness."""
        if not node.best_sequence or not self.leanaide_client:
            return

        try:
            # Verify with LeanAide
            fitness = await self.evaluator.evaluate_with_leanaide(
                node.best_sequence,
                self.leanaide_client
            )

            # Update if better
            if fitness > node.best_fitness:
                node.best_fitness = fitness

        except Exception as e:
            logger.warning(f"Verification failed: {e}")

    async def _verify_proof(self, proof: LeanProof) -> bool:
        """Verify proof with LeanAide."""
        if not self.leanaide_client:
            return False

        try:
            result = await self.leanaide_client.elaborate(proof.lean_code)
            return result.success
        except Exception as e:
            logger.warning(f"Proof verification failed: {e}")
            return False


# =============================================================================
# Memory and Computation Management
# =============================================================================

class EvolutionaryNodeCache:
    """
    Cache evolved populations to avoid recomputation.

    Stores populations by state hash so they can be reused
    when the same state is reached via different paths.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached populations
        """
        self.cache: Dict[str, EvolutionaryNode] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get_or_compute(
        self,
        state_hash: str,
        compute_fn: Callable[[], EvolutionaryNode]
    ) -> EvolutionaryNode:
        """
        Get cached node or compute new one.

        Args:
            state_hash: Hash of the state
            compute_fn: Function to compute node if not cached

        Returns:
            Evolutionary node (cached or newly computed)
        """
        if state_hash in self.cache:
            self.hits += 1
            return self.cache[state_hash]

        self.misses += 1

        # Compute new node
        node = compute_fn()

        # Add to cache
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[state_hash] = node

        return node

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / max(1, total)

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


# =============================================================================
# Utility Functions
# =============================================================================

def create_action_sequence_from_tactics(
    tactics: List[str],
    generation: int = 0
) -> ActionSequence:
    """
    Create an ActionSequence from a list of tactic names.

    Args:
        tactics: List of tactic names
        generation: Generation number

    Returns:
        ActionSequence object
    """
    actions = [Tactic(name=t) for t in tactics]
    return ActionSequence(
        actions=actions,
        generation=generation
    )


def create_evolutionary_mcts(
    population_size: int = 20,
    evolution_generations: int = 5,
    **kwargs
) -> EvolutionaryMCTS:
    """
    Convenience function to create an EvolutionaryMCTS instance.

    Args:
        population_size: Size of evolutionary population
        evolution_generations: Generations per simulation
        **kwargs: Additional arguments

    Returns:
        EvolutionaryMCTS instance
    """
    return EvolutionaryMCTS(
        population_size=population_size,
        evolution_generations=evolution_generations,
        **kwargs
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core data classes
    'ActionSequence',
    'ProofContext',
    'EvolutionaryNode',

    # Operators
    'SequenceCrossover',
    'SequenceMutation',
    'SequenceSelection',
    'SequenceEvaluator',

    # Main algorithms
    'EvolutionaryMCTS',
    'DistributedEvolutionaryMCTS',
    'EvolutionaryMCTSWithLeanAide',

    # Control and utilities
    'AdaptiveEvolutionController',
    'EvolutionaryNodeCache',
    'EvolutionaryTree',

    # Utility functions
    'create_action_sequence_from_tactics',
    'create_evolutionary_mcts',
]


# =============================================================================
# Example Usage
# =============================================================================

async def main():
    """Example usage of Evolutionary MCTS."""

    print("=" * 80)
    print("Evolutionary MCTS Example")
    print("=" * 80)

    # Create proof context
    context = ProofContext(
        theorem="forall (a b : Nat), a + b = b + a",
        goals=["prove a + b = b + a"],
        hypotheses=[],
        available_tactics=[
            "intros", "simp", "rw", "apply", "exact",
            "induction", "cases", "linarith", "ring"
        ]
    )

    # Create evolutionary MCTS
    emcts = create_evolutionary_mcts(
        population_size=20,
        evolution_generations=5,
        mcts_simulations=100
    )

    # Run search
    result = await emcts.search(context)

    # Print results
    print("\n" + "=" * 80)
    print("Evolutionary MCTS Results")
    print("=" * 80)
    print(f"\nSuccess: {result.success}")
    print(f"Time: {result.time_elapsed:.2f}s")
    print(f"Nodes visited: {result.nodes_visited}")
    print(f"Win rate: {result.win_rate:.4f}")
    print(f"Total evolutions: {result.search_statistics.get('total_evolutions', 0)}")
    print(f"Total evaluations: {result.search_statistics.get('total_evaluations', 0)}")

    if result.best_proof:
        print("\nBest proof found:")
        print(result.best_proof.lean_code)


if __name__ == "__main__":
    asyncio.run(main())
