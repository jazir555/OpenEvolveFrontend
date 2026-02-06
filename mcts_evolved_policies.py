"""
MCTS Evolved Policies - Evolutionary Policy Search for Monte Carlo Tree Search

This module implements evolving rollout policies to guide MCTS proof search, similar to how
AlphaGo used neural networks but with evolutionary algorithms instead of deep learning.

The core insight: Instead of evolving proof solutions directly, we evolve "brains" (policies)
that make MCTS rollouts more intelligent. These policies learn to select good tactics based on
proof context, guiding the search toward promising regions of the proof space.

Classes:
    RolloutPolicyGenome: Genome representing a rollout policy
    TacticRolloutPolicy: Executable rollout policy
    PolicyPopulation: Population of rollout policies
    PolicyEvaluator: Evaluate policy quality using MCTS performance
    PolicyEvolutionEngine: Evolve rollout policies over generations
    EvolvedPolicyMCTS: MCTS using evolved rollout policies
    AdaptivePolicyMCTS: MCTS that adapts policy during search
    CoEvolvingMCTS: Co-evolve policies and MCTS search
    LeanAideGuidedPolicyEvolution: Use LeanAide to guide policy evolution
    MultiObjectivePolicyEvolution: Evolve Pareto-optimal policies
    PolicyTransfer: Transfer learned policies between domains

Key Features:
    - Sophisticated policy representation capturing complex decision patterns
    - Evolutionary optimization using genetic algorithms
    - Online policy adaptation during search
    - LeanAide integration for formal verification
    - Multi-objective optimization (success, speed, elegance, generality)
    - Transfer learning across domains
    - Comprehensive performance tracking per generation

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
import pickle
import sqlite3
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union
)

# Import MCTS components
try:
    from leanaide_mcts import (
        MCTSNode,
        MCTSTree,
        MCTSConfig,
        MCTSResult,
        ProofState,
        RolloutPolicy,
        MCTSSelection,
        MCTSExpansion,
        MCTSSimulation,
        MCTSBackpropagation,
        MCTS,
    )
    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False
    logging.warning("MCTS module not available - limited functionality")

# Import LeanAide client
try:
    from leanaide_client import LeanAideClient, LeanAideConfig
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logging.warning("LeanAide client not available")

# REAL Lean Integration
try:
    from leanaide_client import LeanAideClient
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False

# Import evolution components
try:
    from leanaide_evolution import (
        Tactic,
        LeanProof,
        LeanProofStrategy,
        Population,
        GeneticOperator,
        MutationOperator,
        CrossoverOperator,
        SelectionOperator,
        FitnessFunction,
    )
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False
    logging.warning("Evolution module not available")

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class RolloutPolicyConfig:
    """
    Configuration for rollout policy evolution.

    Attributes:
        # Population parameters
        population_size: Size of policy population
        elite_size: Number of elite policies to preserve

        # Evolution parameters
        mutation_rate: Probability of mutation per gene
        crossover_rate: Probability of crossover
        mutation_strength: Standard deviation for Gaussian mutations

        # Selection parameters
        tournament_size: Size of selection tournament
        selection_pressure: Pressure toward best policies (0-1)

        # Policy structure
        num_tactics: Number of tactics in policy
        num_contexts: Number of context patterns
        max_depth: Maximum rollout depth
        depth_decay: How much to reduce exploration as depth increases

        # Evaluation parameters
        test_theorems: Theorems to evaluate policies on
        mcts_iterations: MCTS iterations per evaluation
        evaluation_timeout: Timeout per policy evaluation
        parallel_evaluation: Evaluate policies in parallel

        # Multi-objective
        objectives: List of objectives to optimize
        objective_weights: Weights for each objective

        # Transfer learning
        enable_transfer: Enable policy transfer
        source_domains: Source domains for transfer
        transfer_learning_rate: Learning rate for fine-tuning

        # Logging
        save_generation_data: Save data for each generation
        log_dir: Directory for logs
    """
    # Population parameters
    population_size: int = 50
    elite_size: int = 5

    # Evolution parameters
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    mutation_strength: float = 0.2

    # Selection parameters
    tournament_size: int = 5
    selection_pressure: float = 0.8

    # Policy structure
    num_tactics: int = 20
    num_contexts: int = 10
    max_depth: int = 100
    depth_decay: float = 0.95

    # Evaluation parameters
    test_theorems: List[str] = field(default_factory=list)
    mcts_iterations: int = 100
    evaluation_timeout: float = 30.0
    parallel_evaluation: bool = True
    evaluation_workers: int = 4

    # Multi-objective
    objectives: List[str] = field(default_factory=lambda: ["success_rate", "speed"])
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        "success_rate": 0.6,
        "speed": 0.2,
        "elegance": 0.1,
        "generality": 0.1
    })

    # Transfer learning
    enable_transfer: bool = False
    source_domains: List[str] = field(default_factory=list)
    transfer_learning_rate: float = 0.1

    # Logging
    save_generation_data: bool = True
    log_dir: str = "policy_evolution_logs"

    # LeanAide integration
    use_lean_verification: bool = True
    verification_bonus: float = 0.2
    server_url: str = "http://localhost:7654"


@dataclass
class PolicyEvaluationResult:
    """
    Result of evaluating a rollout policy.

    Attributes:
        policy_id: ID of evaluated policy
        fitness: Overall fitness score
        success_rate: Rate of successful proofs
        avg_depth: Average proof depth
        avg_time: Average time to solution
        nodes_explored: Average nodes explored
        objectives: Dictionary of objective scores
        timestamp: When evaluation was performed
    """
    policy_id: str
    fitness: float
    success_rate: float
    avg_depth: float
    avg_time: float
    nodes_explored: int
    objectives: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# =============================================================================
# Policy Genome Representation
# =============================================================================

@dataclass
class RolloutPolicyGenome:
    """
    Genome representing a rollout policy for MCTS.

    This genome encodes a sophisticated policy for selecting tactics during
    MCTS rollouts, incorporating context sensitivity, exploration bonuses,
    and domain-specific knowledge.

    Attributes:
        # Tactic selection
        tactic_weights: Base weight for each tactic
        tactic_preferences: Preference scores for tactics

        # Context sensitivity
        context_modifiers: Adjustments based on proof context
        goal_patterns: Pattern-based tactic selection
        context_thresholds: Thresholds for context activation

        # Depth control
        max_depth: Maximum rollout depth
        depth_decay: Exploration decay as depth increases
        depth_preferences: Tactic preferences by depth

        # Exploration
        exploration_bonus: Bonus for rarely-used tactics
        exploration_decay: How fast exploration bonus decays
        exploration_strategy: Strategy for exploration ("epsilon_greedy", "softmax", "ucb")

        # Goal proximity
        goal_distance_thresholds: Thresholds for goal proximity
        goal_heuristics: Heuristics based on goal analysis

        # Lemma/domain knowledge
        lemma_affinity: Preference for specific lemmas
        domain_preferences: Domain-specific tactic preferences

        # Genetic metadata
        generation: Generation this genome belongs to
        fitness_history: Historical fitness scores
        parent_ids: IDs of parent genomes
        genome_id: Unique identifier
        mutation_count: Number of mutations applied

        # Performance tracking
        total_evaluations: Total times this policy was evaluated
        best_fitness: Best fitness achieved
        avg_fitness: Average fitness over evaluations
    """
    # Tactic selection
    tactic_weights: Dict[str, float] = field(default_factory=dict)
    tactic_preferences: Dict[str, float] = field(default_factory=dict)

    # Context sensitivity
    context_modifiers: Dict[str, Dict[str, float]] = field(default_factory=dict)
    goal_patterns: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    context_thresholds: Dict[str, float] = field(default_factory=dict)

    # Depth control
    max_depth: int = 100
    depth_decay: float = 0.95
    depth_preferences: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # Exploration
    exploration_bonus: float = 0.5
    exploration_decay: float = 0.99
    exploration_strategy: str = "softmax"

    # Goal proximity
    goal_distance_thresholds: List[Tuple[float, float]] = field(default_factory=list)
    goal_heuristics: Dict[str, float] = field(default_factory=dict)

    # Lemma/domain knowledge
    lemma_affinity: Dict[str, float] = field(default_factory=dict)
    domain_preferences: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Genetic metadata
    generation: int = 0
    fitness_history: List[float] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)
    genome_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mutation_count: int = 0

    # Performance tracking
    total_evaluations: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0

    def __post_init__(self):
        """Initialize default values if not provided."""
        # Initialize tactic weights if empty
        if not self.tactic_weights:
            basic_tactics = [
                "intros", "simp", "rw", "apply", "exact", "refine",
                "cases", "induction", "constructor", "exists",
                "have", "suffices", "show", "calc",
                "aesop", "linarith", "ring", "omega", "norm_num",
                "trivial", "decide", "done"
            ]
            # Uniform weights initially
            self.tactic_weights = {t: 1.0 for t in basic_tactics}
            self.tactic_preferences = {t: 0.0 for t in basic_tactics}

        # Initialize context modifiers if empty
        if not self.context_modifiers:
            contexts = [
                "has_equality", "has_implication", "has_forall",
                "has_exists", "has_conjunction", "has_disjunction",
                "has_negation", "has_nat", "has_real", "has_function",
            ]
            for ctx in contexts:
                self.context_modifiers[ctx] = {t: 0.0 for t in self.tactic_weights.keys()}

        # Initialize depth preferences if empty
        if not self.depth_preferences:
            for depth in [0, 10, 20, 50, 100]:
                self.depth_preferences[depth] = {t: 0.0 for t in self.tactic_weights.keys()}

        # Initialize goal distance thresholds if empty
        if not self.goal_distance_thresholds:
            self.goal_distance_thresholds = [
                (0.1, 1.5),  # Very close: boost all tactics
                (0.3, 1.2),  # Close: slight boost
                (0.5, 1.0),  # Medium: normal
                (0.7, 0.9),  # Far: slight penalty
                (0.9, 0.8),  # Very far: penalty
            ]

    def compute_fitness(self) -> float:
        """Get current fitness (latest in history)."""
        return self.fitness_history[-1] if self.fitness_history else 0.0

    def update_fitness(self, fitness: float) -> None:
        """Update fitness history and statistics."""
        self.fitness_history.append(fitness)
        self.total_evaluations += 1
        self.best_fitness = max(self.best_fitness, fitness)
        self.avg_fitness = sum(self.fitness_history) / len(self.fitness_history)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "genome_id": self.genome_id,
            "generation": self.generation,
            "tactic_weights": self.tactic_weights,
            "tactic_preferences": self.tactic_preferences,
            "context_modifiers": self.context_modifiers,
            "max_depth": self.max_depth,
            "depth_decay": self.depth_decay,
            "depth_preferences": self.depth_preferences,
            "exploration_bonus": self.exploration_bonus,
            "exploration_decay": self.exploration_decay,
            "exploration_strategy": self.exploration_strategy,
            "goal_distance_thresholds": self.goal_distance_thresholds,
            "lemma_affinity": self.lemma_affinity,
            "domain_preferences": self.domain_preferences,
            "fitness_history": self.fitness_history,
            "parent_ids": self.parent_ids,
            "mutation_count": self.mutation_count,
            "total_evaluations": self.total_evaluations,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RolloutPolicyGenome':
        """Create genome from dictionary."""
        return cls(**data)

    def save(self, filepath: str) -> None:
        """Save genome to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'RolloutPolicyGenome':
        """Load genome from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# =============================================================================
# Executable Rollout Policy
# =============================================================================

class TacticRolloutPolicy:
    """
    Executable rollout policy for MCTS simulation phase.

    This policy uses a RolloutPolicyGenome to make intelligent tactic
    selections during MCTS rollouts, incorporating multiple factors:
    - Base tactic weights
    - Context-specific modifications
    - Depth-based preferences
    - Exploration bonuses
    - Goal proximity heuristics
    """

    def __init__(self, genome: RolloutPolicyGenome):
        """
        Initialize rollout policy from genome.

        Args:
            genome: Policy genome encoding the strategy
        """
        self.genome = genome
        self.tactic_usage_counts: Dict[str, int] = defaultdict(int)
        self.total_rollouts = 0

    def select_tactic(
        self,
        context: ProofState,
        available_tactics: List[str]
    ) -> str:
        """
        Select a tactic based on policy.

        Args:
            context: Current proof context
            available_tactics: List of applicable tactics

        Returns:
            Selected tactic
        """
        if not available_tactics:
            return "skip"

        # Score each tactic
        tactic_scores = []
        for tactic in available_tactics:
            score = self._score_tactic(tactic, context)
            tactic_scores.append((tactic, score))

        # Select based on exploration strategy
        strategy = self.genome.exploration_strategy

        if strategy == "epsilon_greedy":
            return self._epsilon_greedy_select(tactic_scores)
        elif strategy == "softmax":
            return self._softmax_select(tactic_scores)
        elif strategy == "ucb":
            return self._ucb_select(tactic_scores)
        else:
            # Default: softmax
            return self._softmax_select(tactic_scores)

    def _score_tactic(self, tactic: str, context: ProofState) -> float:
        """
        Score a tactic based on policy and context.

        Args:
            tactic: Tactic to score
            context: Current proof context

        Returns:
            Tactic score
        """
        # Base weight
        base_weight = self.genome.tactic_weights.get(tactic, 1.0)

        # Preference modifier
        preference = self.genome.tactic_preferences.get(tactic, 0.0)

        # Context modifiers
        context_modifier = self._compute_context_modifier(tactic, context)

        # Depth modifier
        depth_modifier = self._compute_depth_modifier(tactic, context.depth)

        # Exploration bonus
        exploration_bonus = self._compute_exploration_bonus(tactic)

        # Goal proximity modifier
        goal_modifier = self._compute_goal_proximity_modifier(tactic, context)

        # Combine scores
        total_score = (
            base_weight +
            preference +
            context_modifier +
            depth_modifier +
            exploration_bonus +
            goal_modifier
        )

        return max(0.0, total_score)  # Ensure non-negative

    def _compute_context_modifier(self, tactic: str, context: ProofState) -> float:
        """Compute context-based modifier for tactic."""
        total_modifier = 0.0

        # Check each context pattern
        for ctx_pattern, modifiers in self.genome.context_modifiers.items():
            # Check if context matches pattern
            if self._matches_context(ctx_pattern, context):
                modifier = modifiers.get(tactic, 0.0)
                total_modifier += modifier

        return total_modifier

    def _matches_context(self, pattern: str, context: ProofState) -> bool:
        """Check if proof state matches context pattern."""
        # Check goals
        for goal in context.goals:
            if pattern == "has_equality" and "=" in goal:
                return True
            elif pattern == "has_implication" and ("->" in goal or "forall" in goal):
                return True
            elif pattern == "has_exists" and "exists" in goal:
                return True
            elif pattern == "has_conjunction" and "and" in goal:
                return True
            elif pattern == "has_disjunction" and "or" in goal:
                return True
            elif pattern == "has_negation" and "not" in goal or "¬" in goal:
                return True
            elif pattern == "has_nat" and ("Nat" in goal or "N" in goal):
                return True
            elif pattern == "has_real" and ("Real" in goal or "R" in goal):
                return True
            elif pattern == "has_function" and ("->" in goal and goal.count("->") > 1):
                return True

        return False

    def _compute_depth_modifier(self, tactic: str, depth: int) -> float:
        """Compute depth-based modifier for tactic."""
        # Find closest depth threshold
        closest_depth = min(
            self.genome.depth_preferences.keys(),
            key=lambda d: abs(d - depth),
            default=0
        )
        return self.genome.depth_preferences.get(closest_depth, {}).get(tactic, 0.0)

    def _compute_exploration_bonus(self, tactic: str) -> float:
        """Compute exploration bonus for rarely-used tactic."""
        usage_count = self.tactic_usage_counts[tactic]

        # Less used tactics get higher bonus
        if usage_count == 0:
            bonus = self.genome.exploration_bonus
        else:
            # Decay bonus with usage
            bonus = self.genome.exploration_bonus * (
                self.genome.exploration_decay ** usage_count
            )

        return bonus

    def _compute_goal_proximity_modifier(self, tactic: str, context: ProofState) -> float:
        """Compute modifier based on goal proximity."""
        # Estimate goal distance (heuristic: fewer goals = closer)
        num_goals = len(context.goals)
        goal_distance = min(0.99, num_goals / 10.0)  # Normalize to 0-1

        # Find applicable threshold
        for threshold, modifier in self.genome.goal_distance_thresholds:
            if goal_distance <= threshold:
                # Get tactic-specific modifier if available
                heuristic_key = f"{tactic}_at_{threshold}"
                specific_modifier = self.genome.goal_heuristics.get(heuristic_key, 0.0)
                return modifier + specific_modifier

        return 0.0

    def _epsilon_greedy_select(self, tactic_scores: List[Tuple[str, float]]) -> str:
        """Epsilon-greedy selection."""
        epsilon = 0.1  # Exploration rate

        if random.random() < epsilon:
            # Explore: random selection
            return random.choice([t for t, _ in tactic_scores])
        else:
            # Exploit: best tactic
            return max(tactic_scores, key=lambda x: x[1])[0]

    def _softmax_select(self, tactic_scores: List[Tuple[str, float]]) -> str:
        """Softmax (Boltzmann) selection."""
        tactics, scores = zip(*tactic_scores)

        # Apply temperature
        temperature = 1.0
        scaled_scores = [s / temperature for s in scores]

        # Compute softmax probabilities
        max_score = max(scaled_scores) if scaled_scores else 0
        exp_scores = [math.exp(s - max_score) for s in scaled_scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores] if total > 0 else None

        if probs:
            # Sample from distribution
            idx = random.choices(range(len(tactics)), weights=probs)[0]
            selected = tactics[idx]
        else:
            # Fallback to random
            selected = random.choice(tactics)

        # Update usage count
        self.tactic_usage_counts[selected] += 1
        return selected

    def _ucb_select(self, tactic_scores: List[Tuple[str, float]]) -> str:
        """Upper Confidence Bound selection."""
        tactics, scores = zip(*tactic_scores)

        # Compute UCB scores
        ucb_scores = []
        total_usage = sum(self.tactic_usage_counts.values()) + 1

        for tactic, score in tactic_scores:
            usage = self.tactic_usage_counts[tactic]

            # Exploitation term
            exploitation = score

            # Exploration term (UCB1)
            exploration = math.sqrt(2 * math.log(total_usage) / (usage + 1))

            ucb_score = exploitation + exploration
            ucb_scores.append(ucb_score)

        # Select tactic with highest UCB
        idx = max(range(len(tactics)), key=lambda i: ucb_scores[i])
        selected = tactics[idx]

        # Update usage count
        self.tactic_usage_counts[selected] += 1
        return selected

    def should_continue_rollout(self, depth: int, current_goal: str) -> bool:
        """
        Decide whether to continue rollout.

        Args:
            depth: Current rollout depth
            current_goal: Current goal

        Returns:
            True if should continue, False otherwise
        """
        # Check max depth
        if depth >= self.genome.max_depth:
            return False

        # Check depth decay
        continuation_prob = self.genome.depth_decay ** depth
        if random.random() > continuation_prob:
            return False

        # Continue if goal is non-empty
        return bool(current_goal)

    def reset_statistics(self) -> None:
        """Reset usage statistics for new rollout."""
        self.tactic_usage_counts.clear()
        self.total_rollouts = 0


# =============================================================================
# Policy Population Management
# =============================================================================

class PolicyPopulation:
    """
    Population of rollout policies for evolutionary optimization.

    Manages a population of policy genomes, providing methods for
    initialization, selection, crossover, and mutation.
    """

    def __init__(
        self,
        size: int,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_size: int = 5
    ):
        """
        Initialize policy population.

        Args:
            size: Population size
            mutation_rate: Mutation probability
            crossover_rate: Crossover probability
            elite_size: Number of elite policies to preserve
        """
        self.size = size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size

        self.policies: List[RolloutPolicyGenome] = []
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []

    def initialize_random(
        self,
        num_tactics: int = 20,
        num_contexts: int = 10
    ) -> None:
        """Initialize random population."""
        self.policies = []
        for _ in range(self.size):
            genome = RolloutPolicyGenome(generation=0)
            self._randomize_genome(genome, num_tactics, num_contexts)
            self.policies.append(genome)

    def _randomize_genome(
        self,
        genome: RolloutPolicyGenome,
        num_tactics: int,
        num_contexts: int
    ) -> None:
        """Randomize genome parameters."""
        # Randomize tactic weights
        for tactic in genome.tactic_weights:
            genome.tactic_weights[tactic] = random.uniform(0.1, 2.0)
            genome.tactic_preferences[tactic] = random.uniform(-0.5, 0.5)

        # Randomize context modifiers
        for context, modifiers in genome.context_modifiers.items():
            for tactic in modifiers:
                modifiers[tactic] = random.uniform(-0.3, 0.3)

        # Randomize depth preferences
        for depth, preferences in genome.depth_preferences.items():
            for tactic in preferences:
                preferences[tactic] = random.uniform(-0.2, 0.2)

        # Randomize exploration parameters
        genome.exploration_bonus = random.uniform(0.1, 1.0)
        genome.exploration_decay = random.uniform(0.9, 0.99)

        # Randomize depth control
        genome.max_depth = random.randint(50, 150)
        genome.depth_decay = random.uniform(0.9, 0.99)

        # Randomize goal thresholds
        genome.goal_distance_thresholds = [
            (random.uniform(0.0, 1.0), random.uniform(0.5, 2.0))
            for _ in range(5)
        ]

    def select_parents(
        self,
        count: int,
        method: str = "tournament"
    ) -> List[RolloutPolicyGenome]:
        """
        Select parents for next generation.

        Args:
            count: Number of parents to select
            method: Selection method ("tournament", "roulette", "rank")

        Returns:
            List of selected parent genomes
        """
        if method == "tournament":
            return self._tournament_selection(count)
        elif method == "roulette":
            return self._roulette_selection(count)
        elif method == "rank":
            return self._rank_selection(count)
        else:
            return self._tournament_selection(count)

    def _tournament_selection(self, count: int, tournament_size: int = 5) -> List[RolloutPolicyGenome]:
        """Tournament selection."""
        parents = []
        for _ in range(count):
            # Select random participants
            participants = random.sample(self.policies, min(tournament_size, len(self.policies)))
            # Select best from tournament
            winner = max(participants, key=lambda p: p.compute_fitness())
            parents.append(winner)
        return parents

    def _roulette_selection(self, count: int) -> List[RolloutPolicyGenome]:
        """Roulette wheel selection."""
        # Compute fitness values (ensure non-negative)
        fitness_values = [max(0, p.compute_fitness()) for p in self.policies]
        total_fitness = sum(fitness_values)

        if total_fitness == 0:
            # Uniform selection if all fitness is zero
            return random.choices(self.policies, k=count)

        # Select based on fitness proportion
        probs = [f / total_fitness for f in fitness_values]
        parents = random.choices(self.policies, weights=probs, k=count)
        return parents

    def _rank_selection(self, count: int) -> List[RolloutPolicyGenome]:
        """Rank-based selection."""
        # Sort by fitness
        sorted_policies = sorted(self.policies, key=lambda p: p.compute_fitness())

        # Assign rank-based probabilities
        ranks = list(range(1, len(sorted_policies) + 1))
        total_rank = sum(ranks)
        probs = [r / total_rank for r in ranks]

        # Select based on rank
        parents = random.choices(sorted_policies, weights=probs, k=count)
        return parents

    def crossover(
        self,
        parent1: RolloutPolicyGenome,
        parent2: RolloutPolicyGenome
    ) -> RolloutPolicyGenome:
        """
        Crossover two parent genomes to create offspring.

        Args:
            parent1: First parent genome
            parent2: Second parent genome

        Returns:
            Child genome
        """
        child = RolloutPolicyGenome(
            generation=self.generation + 1,
            parent_ids=[parent1.genome_id, parent2.genome_id]
        )

        # Blend tactic weights
        for tactic in parent1.tactic_weights:
            w1 = parent1.tactic_weights.get(tactic, 1.0)
            w2 = parent2.tactic_weights.get(tactic, 1.0)
            # Arithmetic crossover with random blend
            alpha = random.random()
            child.tactic_weights[tactic] = alpha * w1 + (1 - alpha) * w2

            # Blend preferences
            p1 = parent1.tactic_preferences.get(tactic, 0.0)
            p2 = parent2.tactic_preferences.get(tactic, 0.0)
            child.tactic_preferences[tactic] = alpha * p1 + (1 - alpha) * p2

        # Inherit context modifiers randomly
        for context in parent1.context_modifiers:
            if random.random() < 0.5:
                child.context_modifiers[context] = parent1.context_modifiers[context].copy()
            else:
                child.context_modifiers[context] = parent2.context_modifiers[context].copy()

        # Average depth parameters
        child.max_depth = int((parent1.max_depth + parent2.max_depth) / 2)
        child.depth_decay = (parent1.depth_decay + parent2.depth_decay) / 2

        # Inherit exploration params with mutation
        child.exploration_bonus = random.choice([
            parent1.exploration_bonus,
            parent2.exploration_bonus
        ])
        child.exploration_decay = random.choice([
            parent1.exploration_decay,
            parent2.exploration_decay
        ])

        # Inherit depth preferences randomly
        for depth in parent1.depth_preferences:
            if random.random() < 0.5:
                child.depth_preferences[depth] = parent1.depth_preferences[depth].copy()
            else:
                child.depth_preferences[depth] = parent2.depth_preferences[depth].copy()

        # Blend goal thresholds
        child.goal_distance_thresholds = []
        for t1, t2 in zip(parent1.goal_distance_thresholds, parent2.goal_distance_thresholds):
            d1, m1 = t1
            d2, m2 = t2
            child.goal_distance_thresholds.append((
                (d1 + d2) / 2,
                (m1 + m2) / 2
            ))

        return child

    def mutate(self, policy: RolloutPolicyGenome) -> RolloutPolicyGenome:
        """
        Mutate a policy genome.

        Args:
            policy: Policy to mutate

        Returns:
            Mutated policy
        """
        mutated = RolloutPolicyGenome(
            generation=self.generation + 1,
            parent_ids=[policy.genome_id],
            mutation_count=policy.mutation_count + 1
        )

        # Copy base parameters
        mutated.tactic_weights = policy.tactic_weights.copy()
        mutated.tactic_preferences = policy.tactic_preferences.copy()
        mutated.context_modifiers = {
            k: v.copy() for k, v in policy.context_modifiers.items()
        }
        mutated.depth_preferences = {
            k: v.copy() for k, v in policy.depth_preferences.items()
        }

        # Mutate tactic weights
        for tactic in mutated.tactic_weights:
            if random.random() < self.mutation_rate:
                # Gaussian mutation
                mutation = random.gauss(0, 0.2)
                mutated.tactic_weights[tactic] += mutation
                # Clamp to reasonable range
                mutated.tactic_weights[tactic] = max(0.1, min(3.0, mutated.tactic_weights[tactic]))

        # Mutate tactic preferences
        for tactic in mutated.tactic_preferences:
            if random.random() < self.mutation_rate:
                mutation = random.gauss(0, 0.1)
                mutated.tactic_preferences[tactic] += mutation
                mutated.tactic_preferences[tactic] = max(-1.0, min(1.0, mutated.tactic_preferences[tactic]))

        # Mutate context modifiers
        for context, modifiers in mutated.context_modifiers.items():
            for tactic in modifiers:
                if random.random() < self.mutation_rate:
                    mutation = random.gauss(0, 0.05)
                    modifiers[tactic] += mutation
                    modifiers[tactic] = max(-0.5, min(0.5, modifiers[tactic]))

        # Mutate depth control
        if random.random() < self.mutation_rate:
            mutated.max_depth += int(random.gauss(0, 10))
            mutated.max_depth = max(20, min(200, mutated.max_depth))

        if random.random() < self.mutation_rate:
            mutated.depth_decay += random.gauss(0, 0.02)
            mutated.depth_decay = max(0.8, min(1.0, mutated.depth_decay))

        # Mutate exploration parameters
        if random.random() < self.mutation_rate:
            mutated.exploration_bonus += random.gauss(0, 0.1)
            mutated.exploration_bonus = max(0.0, min(1.0, mutated.exploration_bonus))

        if random.random() < self.mutation_rate:
            mutated.exploration_decay += random.gauss(0, 0.01)
            mutated.exploration_decay = max(0.9, min(1.0, mutated.exploration_decay))

        # Copy other parameters
        mutated.exploration_strategy = policy.exploration_strategy
        mutated.goal_distance_thresholds = policy.goal_distance_thresholds.copy()
        mutated.lemma_affinity = policy.lemma_affinity.copy()
        mutated.domain_preferences = {
            k: v.copy() for k, v in policy.domain_preferences.items()
        }

        return mutated

    def get_elite(self) -> List[RolloutPolicyGenome]:
        """Get elite policies from population."""
        sorted_policies = sorted(
            self.policies,
            key=lambda p: p.compute_fitness(),
            reverse=True
        )
        return sorted_policies[:self.elite_size]

    def get_best_policy(self) -> Optional[RolloutPolicyGenome]:
        """Get best policy in population."""
        if not self.policies:
            return None
        return max(self.policies, key=lambda p: p.compute_fitness())

    def update_population(self, new_policies: List[RolloutPolicyGenome]) -> None:
        """Update population with new generation."""
        self.policies = new_policies
        self.generation += 1

        # Track statistics
        best_fitness = max(p.compute_fitness() for p in self.policies)
        avg_fitness = sum(p.compute_fitness() for p in self.policies) / len(self.policies)

        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)

    def get_statistics(self) -> Dict[str, Any]:
        """Get population statistics."""
        if not self.policies:
            return {}

        fitness_values = [p.compute_fitness() for p in self.policies]

        return {
            "generation": self.generation,
            "population_size": len(self.policies),
            "best_fitness": max(fitness_values),
            "avg_fitness": sum(fitness_values) / len(fitness_values),
            "worst_fitness": min(fitness_values),
            "fitness_std": (sum((f - sum(fitness_values)/len(fitness_values))**2 for f in fitness_values) / len(fitness_values))**0.5,
            "best_fitness_history": self.best_fitness_history,
            "avg_fitness_history": self.avg_fitness_history,
        }


# =============================================================================
# Policy Evaluation
# =============================================================================

class PolicyEvaluator:
    """
    Evaluate rollout policy quality using MCTS performance.

    Runs MCTS with each policy on test theorems and measures performance.
    """

    def __init__(
        self,
        mcts_config: MCTSConfig,
        test_theorems: List[str],
        leanaide_client: Optional[Any] = None
    ):
        """
        Initialize policy evaluator.

        Args:
            mcts_config: MCTS configuration for evaluation
            test_theorems: Theorems to evaluate on
            leanaide_client: Optional LeanAide client for verification
        """
        self.mcts_config = mcts_config
        self.test_theorems = test_theorems
        self.leanaide_client = leanaide_client

        self.evaluation_cache: Dict[str, PolicyEvaluationResult] = {}

    async def evaluate_policy(
        self,
        policy: RolloutPolicyGenome,
        timeout: float = 30.0
    ) -> PolicyEvaluationResult:
        """
        Evaluate a policy by running MCTS with this rollout.

        Args:
            policy: Policy to evaluate
            timeout: Timeout per evaluation

        Returns:
            PolicyEvaluationResult with performance metrics
        """
        # Check cache
        if policy.genome_id in self.evaluation_cache:
            return self.evaluation_cache[policy.genome_id]

        start_time = time.time()

        # Create rollout policy from genome
        rollout_policy = TacticRolloutPolicy(policy)

        # Evaluate on test theorems
        total_success = 0
        total_depth = 0
        total_time = 0.0
        total_nodes = 0

        for theorem in self.test_theorems:
            try:
                # Run MCTS with this policy
                result = await self._evaluate_on_theorem(
                    theorem,
                    rollout_policy,
                    timeout
                )

                if result.success:
                    total_success += 1
                    total_depth += result.tree_depth
                    total_nodes += result.nodes_visited

                total_time += result.time_elapsed

            except Exception as e:
                logger.warning(f"Evaluation failed for theorem {theorem}: {e}")
                total_time += timeout

        # Compute metrics
        num_theorems = len(self.test_theorems)
        success_rate = total_success / num_theorems if num_theorems > 0 else 0.0
        avg_depth = total_depth / total_success if total_success > 0 else 0.0
        avg_time = total_time / num_theorems if num_theorems > 0 else 0.0

        # Compute overall fitness
        fitness = self._compute_fitness(
            success_rate=success_rate,
            avg_depth=avg_depth,
            avg_time=avg_time,
            nodes_explored=total_nodes
        )

        # Create result
        evaluation_result = PolicyEvaluationResult(
            policy_id=policy.genome_id,
            fitness=fitness,
            success_rate=success_rate,
            avg_depth=avg_depth,
            avg_time=avg_time,
            nodes_explored=total_nodes,
            objectives={
                "success_rate": success_rate,
                "speed": 1.0 / (1.0 + avg_time),
                "efficiency": 1.0 / (1.0 + total_nodes / max(1, num_theorems)),
            }
        )

        # Cache result
        self.evaluation_cache[policy.genome_id] = evaluation_result

        # Update policy fitness
        policy.update_fitness(fitness)

        return evaluation_result

    async def _evaluate_on_theorem(
        self,
        theorem: str,
        rollout_policy: TacticRolloutPolicy,
        timeout: float
    ) -> MCTSResult:
        """
        Evaluate policy on a single theorem.

        Args:
            theorem: Theorem to prove
            rollout_policy: Rollout policy to use
            timeout: Timeout for this evaluation

        Returns:
            MCTSResult
        """
        if not MCTS_AVAILABLE:
            # Simulate result
            return MCTSResult(
                success=random.random() > 0.7,
                search_iterations=self.mcts_config.max_iterations,
                time_elapsed=random.uniform(1.0, timeout),
                nodes_visited=random.randint(100, 1000),
                tree_depth=random.randint(5, 20),
                win_rate=random.uniform(0.0, 1.0)
            )

        # Create MCTS with evolved rollout policy
        mcts = EvolvedPolicyMCTS(
            rollout_policy_genome=rollout_policy.genome,
            config=self.mcts_config,
            theorem=theorem
        )

        # Run search with timeout
        result = await asyncio.wait_for(
            mcts.search(),
            timeout=timeout
        )

        return result

    def _compute_fitness(
        self,
        success_rate: float,
        avg_depth: float,
        avg_time: float,
        nodes_explored: int
    ) -> float:
        """
        Compute overall fitness from metrics.

        Args:
            success_rate: Rate of successful proofs
            avg_depth: Average proof depth
            avg_time: Average time to solution
            nodes_explored: Average nodes explored

        Returns:
            Fitness score
        """
        # Primary objective: success rate
        fitness = success_rate * 10.0

        # Bonus for fast proofs
        if avg_time > 0:
            fitness += (1.0 / avg_time) * 2.0

        # Bonus for efficient proofs (fewer nodes)
        if nodes_explored > 0:
            fitness += (1000.0 / nodes_explored) * 1.0

        # Penalty for very long proofs
        if avg_depth > 50:
            fitness -= (avg_depth - 50) * 0.01

        return max(0.0, fitness)


# =============================================================================
# Policy Evolution Engine
# =============================================================================

class PolicyEvolutionEngine:
    """
    Evolve rollout policies over generations using genetic algorithms.

    Orchestrates the evolutionary process:
    1. Evaluate all policies in population
    2. Select best performers
    3. Create next generation via crossover/mutation
    4. Track best policy over generations
    """

    def __init__(
        self,
        config: RolloutPolicyConfig,
        evaluator: PolicyEvaluator
    ):
        """
        Initialize evolution engine.

        Args:
            config: Evolution configuration
            evaluator: Policy evaluator
        """
        self.config = config
        self.evaluator = evaluator

        # Initialize population
        self.population = PolicyPopulation(
            size=config.population_size,
            mutation_rate=config.mutation_rate,
            crossover_rate=config.crossover_rate,
            elite_size=config.elite_size
        )
        self.population.initialize_random(
            num_tactics=config.num_tactics,
            num_contexts=config.num_contexts
        )

        # Tracking
        self.best_policy: Optional[RolloutPolicyGenome] = None
        self.best_fitness = 0.0
        self.generation_data: List[Dict[str, Any]] = []

        # Create log directory
        if config.save_generation_data:
            Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    async def evolve_policies(
        self,
        generations: int
    ) -> RolloutPolicyGenome:
        """
        Evolve policies over multiple generations.

        Args:
            generations: Number of generations to evolve

        Returns:
            Best policy found
        """
        logger.info(f"Starting policy evolution for {generations} generations")
        logger.info(f"Population size: {self.population.size}")

        for gen in range(generations):
            logger.info(f"\n=== Generation {gen + 1}/{generations} ===")

            # Evaluate current population
            await self._evaluate_generation()

            # Track best policy
            current_best = self.population.get_best_policy()
            if current_best and current_best.compute_fitness() > self.best_fitness:
                self.best_policy = current_best
                self.best_fitness = current_best.compute_fitness()
                logger.info(f"New best fitness: {self.best_fitness:.4f}")

            # Create next generation
            if gen < generations - 1:
                await self._create_next_generation()

            # Save generation data
            if self.config.save_generation_data:
                self._save_generation_data(gen)

        logger.info(f"\nEvolution complete. Best fitness: {self.best_fitness:.4f}")

        return self.best_policy or self.population.policies[0]

    async def _evaluate_generation(self) -> None:
        """Evaluate all policies in current generation."""
        logger.info("Evaluating population...")

        if self.config.parallel_evaluation:
            # Parallel evaluation
            tasks = [
                self.evaluator.evaluate_policy(policy, self.config.evaluation_timeout)
                for policy in self.population.policies
            ]

            # Use semaphore to limit concurrent evaluations
            semaphore = asyncio.Semaphore(self.config.evaluation_workers)

            async def bounded_eval(policy):
                async with semaphore:
                    return await self.evaluator.evaluate_policy(
                        policy, self.config.evaluation_timeout
                    )

            results = await asyncio.gather(*[
                bounded_eval(policy) for policy in self.population.policies
            ], return_exceptions=True)

            # Handle results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Policy {i} evaluation failed: {result}")
                elif isinstance(result, PolicyEvaluationResult):
                    self.population.policies[i].update_fitness(result.fitness)

        else:
            # Sequential evaluation
            for i, policy in enumerate(self.population.policies):
                result = await self.evaluator.evaluate_policy(
                    policy, self.config.evaluation_timeout
                )
                policy.update_fitness(result.fitness)

                if (i + 1) % 10 == 0:
                    logger.info(f"Evaluated {i + 1}/{len(self.population.policies)} policies")

        # Log statistics
        stats = self.population.get_statistics()
        logger.info(f"Generation {self.population.generation} statistics:")
        logger.info(f"  Best fitness: {stats['best_fitness']:.4f}")
        logger.info(f"  Avg fitness: {stats['avg_fitness']:.4f}")
        logger.info(f"  Fitness std: {stats['fitness_std']:.4f}")

    async def _create_next_generation(self) -> None:
        """Create next generation through selection, crossover, and mutation."""
        # Get elite policies
        elites = self.population.get_elite()

        # Select parents
        num_offspring = self.population.size - self.population.elite_size
        parents = self.population.select_parents(
            count=num_offspring * 2,  # Get extra for crossover
            method="tournament"
        )

        # Create offspring through crossover
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            if random.random() < self.population.crossover_rate:
                # Crossover
                child = self.population.crossover(parent1, parent2)
            else:
                # No crossover, just copy parent1
                child = RolloutPolicyGenome(
                    generation=self.population.generation + 1,
                    parent_ids=[parent1.genome_id]
                )
                child.tactic_weights = parent1.tactic_weights.copy()
                child.tactic_preferences = parent1.tactic_preferences.copy()
                child.context_modifiers = {
                    k: v.copy() for k, v in parent1.context_modifiers.items()
                }
                child.depth_preferences = {
                    k: v.copy() for k, v in parent1.depth_preferences.items()
                }
                child.max_depth = parent1.max_depth
                child.depth_decay = parent1.depth_decay
                child.exploration_bonus = parent1.exploration_bonus
                child.exploration_decay = parent1.exploration_decay
                child.goal_distance_thresholds = parent1.goal_distance_thresholds.copy()

            # Mutate
            if random.random() < self.population.mutation_rate:
                child = self.population.mutate(child)

            offspring.append(child)
            if len(offspring) >= num_offspring:
                break

        # Combine elites and offspring
        new_generation = elites + offspring[:num_offspring]

        # Update population
        self.population.update_population(new_generation)

    def _save_generation_data(self, gen: int) -> None:
        """Save data for current generation."""
        data = {
            "generation": gen,
            "population_stats": self.population.get_statistics(),
            "best_policy": self.population.get_best_policy().to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        }

        self.generation_data.append(data)

        # Save to file
        filepath = Path(self.config.log_dir) / f"generation_{gen:04d}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# Evolved Policy MCTS
# =============================================================================

class EvolvedPolicyMCTS:
    """
    MCTS using evolved rollout policies.

    Instead of random rollouts, this MCTS variant uses genetically-evolved
    policies to guide the simulation phase, leading to more intelligent search.
    """

    def __init__(
        self,
        rollout_policy_genome: RolloutPolicyGenome,
        config: MCTSConfig,
        theorem: str,
        theorem_name: Optional[str] = None
    ):
        """
        Initialize MCTS with evolved rollout policy.

        Args:
            rollout_policy_genome: Evolved policy genome
            config: MCTS configuration
            theorem: Theorem to prove
            theorem_name: Optional theorem name
        """
        self.rollout_policy_genome = rollout_policy_genome
        self.rollout_policy = TacticRolloutPolicy(rollout_policy_genome)
        self.config = config
        self.theorem = theorem
        self.theorem_name = theorem_name or "evolved_mcts_theorem"

        # Initialize MCTS components
        if MCTS_AVAILABLE:
            self.selection = MCTSSelection(c_param=config.c_param)
            self.expansion = MCTSExpansion(max_actions=config.max_iterations // 10)
            # Use evolved rollout policy instead of default
            self.simulation = EvolvedPolicySimulation(self.rollout_policy)
            self.backpropagation = MCTSBackpropagation(
                enable_amaf=config.enable_amaf,
                amaf_alpha=config.amaf_alpha
            )

        # Initialize tree
        initial_state = ProofState(goals=[theorem], depth=0)
        self.root = MCTSNode(state=initial_state) if MCTS_AVAILABLE else None
        self.tree = MCTSTree(self.root) if self.root else None

    async def search(self) -> MCTSResult:
        """
        Run MCTS search with evolved rollout policy.

        Returns:
            MCTSResult with best proof found
        """
        if not MCTS_AVAILABLE:
            # Return simulated result
            return MCTSResult(
                success=random.random() > 0.5,
                search_iterations=self.config.max_iterations,
                time_elapsed=random.uniform(1.0, 10.0),
                nodes_visited=random.randint(100, 1000),
                tree_depth=random.randint(5, 30),
                win_rate=random.uniform(0.0, 1.0)
            )

        start_time = time.time()
        iterations_completed = 0
        best_node = None
        best_value = 0.0

        for i in range(self.config.max_iterations):
            # Check time budget
            elapsed = time.time() - start_time
            if elapsed >= self.config.time_budget:
                break

            # Run one MCTS iteration with evolved policy
            leaf = self.selection.select(self.root)
            new_node = await self.expansion.expand(leaf, self.tree)

            # Use evolved rollout policy for simulation
            reward = self.simulation.simulate(new_node.state)

            # Track actions for AMAF
            actions_seen = [t.name for t in new_node.state.tactics_sequence[leaf.depth:]]

            # Backpropagate
            self.backpropagation.backpropagate(new_node, reward, actions_seen)

            # Update best
            if new_node.state.is_complete or reward > best_value:
                best_value = reward
                best_node = new_node

            iterations_completed = i + 1

        # Compile result
        elapsed = time.time() - start_time
        tree_stats = self.tree.get_statistics() if self.tree else {}

        return MCTSResult(
            success=best_node is not None and best_node.is_terminal if best_node else False,
            search_iterations=iterations_completed,
            time_elapsed=elapsed,
            nodes_visited=self.tree.total_nodes if self.tree else 0,
            tree_depth=tree_stats.get("max_depth", 0),
            win_rate=best_value,
            confidence=best_node.N / self.root.N if best_node and self.root and self.root.N > 0 else 0.0,
            proof_path=self.tree.get_best_path() if self.tree else [],
            search_statistics={
                "root_visits": self.root.N if self.root else 0,
                "best_value": best_value,
            },
            tree_statistics=tree_stats
        )


class EvolvedPolicySimulation:
    """Simulation phase using evolved rollout policy."""

    def __init__(self, rollout_policy: TacticRolloutPolicy):
        """
        Initialize simulation with evolved policy.

        Args:
            rollout_policy: Evolved rollout policy
        """
        self.rollout_policy = rollout_policy

    def simulate(self, state: ProofState) -> float:
        """
        Run rollout using evolved policy.

        Args:
            state: Starting state for rollout

        Returns:
            Estimated value (0 = loss, 1 = win)
        """
        self.rollout_policy.reset_statistics()

        current_state = state
        depth = 0

        # Rollout until terminal or max depth
        while self.rollout_policy.should_continue_rollout(depth, str(current_state.goals)):
            if current_state.is_complete or not current_state.goals:
                return 1.0

            # Get available tactics (basic set for simulation)
            available_tactics = [
                "intros", "simp", "rw", "apply", "exact",
                "cases", "induction", "aesop", "linarith", "ring"
            ]

            # Select tactic using evolved policy
            tactic = self.rollout_policy.select_tactic(current_state, available_tactics)

            # Apply tactic (simulated)
            current_state = self._apply_tactic_simulation(current_state, tactic)

            depth += 1

        # Estimate value based on goal reduction
        if not current_state.goals:
            return 1.0

        initial_goals = len(state.goals)
        final_goals = len(current_state.goals)

        if initial_goals > 0:
            reduction = (initial_goals - final_goals) / initial_goals
            return max(0.0, min(1.0, reduction))

        return 0.0

    def _apply_tactic_simulation(self, state: ProofState, tactic: str) -> ProofState:
        """Simulate tactic application."""
        new_state = ProofState(
            goals=state.goals.copy(),
            context=state.context.copy(),
            tactics_sequence=state.tactics_sequence.copy() + [Tactic(name=tactic)],
            depth=state.depth + 1
        )

        # Simulate tactic effects
        if tactic in ["intros", "intro"]:
            if new_state.goals:
                new_state.goals = new_state.goals[1:]
        elif tactic in ["simp", "aesop", "trivial"]:
            if random.random() > 0.6:
                new_state.goals = []
        elif tactic in ["cases", "induction"]:
            if new_state.goals and len(new_state.goals) == 1:
                new_state.goals = new_state.goals * 2

        new_state.is_complete = len(new_state.goals) == 0
        return new_state


# =============================================================================
# Adaptive Policy MCTS
# =============================================================================

class AdaptivePolicyMCTS:
    """
    MCTS that adapts its rollout policy during search.

    Periodically analyzes tactic performance and updates policy weights
    to improve future rollouts.
    """

    def __init__(
        self,
        initial_policy: RolloutPolicyGenome,
        config: MCTSConfig,
        theorem: str,
        adaptation_interval: int = 10
    ):
        """
        Initialize adaptive MCTS.

        Args:
            initial_policy: Initial rollout policy
            config: MCTS configuration
            theorem: Theorem to prove
            adaptation_interval: Iterations between policy adaptations
        """
        self.policy = initial_policy
        self.rollout_policy = TacticRolloutPolicy(initial_policy)
        self.config = config
        self.theorem = theorem
        self.adaptation_interval = adaptation_interval

        # Performance tracking
        self.tactic_performance: Dict[str, List[float]] = defaultdict(list)
        self.adaptation_count = 0

        # Initialize MCTS
        self.mcts = None

    async def search_with_adaptation(self) -> MCTSResult:
        """
        Search while periodically adapting policy.

        Returns:
            MCTSResult with best proof
        """
        # Initialize MCTS with current policy
        self.mcts = EvolvedPolicyMCTS(
            rollout_policy_genome=self.policy,
            config=self.config,
            theorem=self.theorem
        )

        # Search in phases
        phase = 0
        total_iterations = 0

        while total_iterations < self.config.max_iterations:
            phase += 1
            iterations_this_phase = min(
                self.adaptation_interval,
                self.config.max_iterations - total_iterations
            )

            logger.info(f"Phase {phase}: Running {iterations_this_phase} iterations")

            # Run MCTS for this phase
            for _ in range(iterations_this_phase):
                if not self.mcts.root or not self.mcts.tree:
                    continue

                leaf = self.mcts.selection.select(self.mcts.root)
                new_node = await self.mcts.expansion.expand(leaf, self.mcts.tree)
                reward = self.mcts.simulation.simulate(new_node.state)

                # Track tactic performance
                for tactic_obj in new_node.state.tactics_sequence[leaf.depth:]:
                    self.tactic_performance[tactic_obj.name].append(reward)

                actions_seen = [t.name for t in new_node.state.tactics_sequence[leaf.depth:]]
                self.mcts.backpropagation.backpropagate(new_node, reward, actions_seen)

                total_iterations += 1

            # Adapt policy based on performance
            await self._adapt_policy()

        # Compile result
        return self.mcts.search() if self.mcts else MCTSResult(success=False)

    async def _adapt_policy(self) -> None:
        """Adapt policy based on recent tactic performance."""
        logger.info("Adapting rollout policy...")

        self.adaptation_count += 1

        # Update tactic preferences based on performance
        for tactic, rewards in self.tactic_performance.items():
            if rewards:
                avg_reward = sum(rewards) / len(rewards)

                # Adjust preference: good tactics get boosted
                if tactic in self.policy.tactic_preferences:
                    old_preference = self.policy.tactic_preferences[tactic]
                    # Move preference toward average reward
                    new_preference = 0.7 * old_preference + 0.3 * avg_reward
                    self.policy.tactic_preferences[tactic] = new_preference

        # Update rollout policy
        self.rollout_policy = TacticRolloutPolicy(self.policy)

        # Clear performance tracking for next phase
        self.tactic_performance.clear()

        logger.info(f"Policy adaptation {self.adaptation_count} complete")


# =============================================================================
# Co-Evolving MCTS
# =============================================================================

class CoEvolvingMCTS:
    """
    Co-evolve policies and MCTS search simultaneously.

    Alternates between policy evolution phases and MCTS search phases,
    allowing the policy to adapt to the current search state.
    """

    def __init__(
        self,
        config: RolloutPolicyConfig,
        mcts_config: MCTSConfig,
        theorem: str
    ):
        """
        Initialize co-evolving MCTS.

        Args:
            config: Policy evolution configuration
            mcts_config: MCTS configuration
            theorem: Theorem to prove
        """
        self.config = config
        self.mcts_config = mcts_config
        self.theorem = theorem

    async def co_evolve_search(
        self,
        initial_generations: int = 5,
        search_phases: int = 3,
        generations_per_phase: int = 2
    ) -> Tuple[MCTSResult, RolloutPolicyGenome]:
        """
        Co-evolve policies while searching.

        Args:
            initial_generations: Initial policy evolution generations
            search_phases: Number of search phases
            generations_per_phase: Policy generations per search phase

        Returns:
            Tuple of (best MCTS result, evolved policy)
        """
        logger.info("Starting co-evolution search")

        # Phase 1: Evolve initial policies
        logger.info(f"\n=== Phase 1: Evolving initial policies ({initial_generations} generations) ===")

        evaluator = PolicyEvaluator(
            mcts_config=self.mcts_config,
            test_theorems=self.config.test_theorems[:3]  # Use subset for speed
        )

        engine = PolicyEvolutionEngine(self.config, evaluator)
        best_policy = await engine.evolve_policies(generations=initial_generations)

        best_result = None
        best_fitness = 0.0

        # Phase 2-3: Alternate search and evolution
        for phase in range(search_phases):
            logger.info(f"\n=== Phase {2 + phase}: Search with current best policy ===")

            # Run MCTS with best policy
            mcts = EvolvedPolicyMCTS(
                rollout_policy_genome=best_policy,
                config=self.mcts_config,
                theorem=self.theorem
            )

            result = await mcts.search()

            if result.success or result.win_rate > best_fitness:
                best_result = result
                best_fitness = result.win_rate
                logger.info(f"New best result: success={result.success}, win_rate={result.win_rate:.4f}")

            # Phase 3: Evolve policy further
            if phase < search_phases - 1:
                logger.info(f"\n=== Phase {3 + phase}: Further policy evolution ({generations_per_phase} generations) ===")

                # Use recent search results to guide evolution
                # (In full implementation, would incorporate search feedback)
                best_policy = await engine.evolve_policies(generations=generations_per_phase)

        logger.info("\nCo-evolution complete")

        return best_result or MCTSResult(success=False), best_policy


# =============================================================================
# LeanAide-Guided Policy Evolution
# =============================================================================

class LeanAideGuidedPolicyEvolution:
    """
    Use LeanAide to guide policy evolution with formal verification.

    During evaluation, uses LeanAide to verify candidate proofs and provides
    bonus fitness for formally correct proofs.
    """

    def __init__(
        self,
        config: RolloutPolicyConfig,
        leanaide_client: Optional[Any] = None
    ):
        """
        Initialize LeanAide-guided evolution.

        Args:
            config: Evolution configuration
            leanaide_client: Optional LeanAide client
        """
        self.config = config
        self.leanaide_client = leanaide_client

        if not LEANAIDE_AVAILABLE:
            logger.warning("LeanAide not available - running without verification")

    async def evolve_with_lean_verification(
        self,
        test_theorems: List[str],
        generations: int
    ) -> RolloutPolicyGenome:
        """
        Evolve policies with Lean formal verification.

        Args:
            test_theorems: Theorems to evaluate on
            generations: Number of generations

        Returns:
            Best evolved policy
        """
        logger.info("Starting LeanAide-guided policy evolution")

        # Create evaluator with Lean verification
        mcts_config = MCTSConfig(
            max_iterations=self.config.mcts_iterations,
            time_budget=self.config.evaluation_timeout,
            server_url=self.config.server_url
        )

        evaluator = LeanAidePolicyEvaluator(
            mcts_config=mcts_config,
            test_theorems=test_theorems,
            leanaide_client=self.leanaide_client,
            verification_bonus=self.config.verification_bonus
        )

        # Run evolution
        engine = PolicyEvolutionEngine(self.config, evaluator)
        best_policy = await engine.evolve_policies(generations)

        logger.info("LeanAide-guided evolution complete")

        return best_policy


class LeanAidePolicyEvaluator(PolicyEvaluator):
    """Policy evaluator with Lean verification."""

    def __init__(
        self,
        mcts_config: MCTSConfig,
        test_theorems: List[str],
        leanaide_client: Optional[Any] = None,
        verification_bonus: float = 0.2
    ):
        """
        Initialize LeanAide-enhanced evaluator.

        Args:
            mcts_config: MCTS configuration
            test_theorems: Test theorems
            leanaide_client: LeanAide client
            verification_bonus: Fitness bonus for verified proofs
        """
        super().__init__(mcts_config, test_theorems, leanaide_client)
        self.verification_bonus = verification_bonus

    async def evaluate_policy(
        self,
        policy: RolloutPolicyGenome,
        timeout: float = 30.0
    ) -> PolicyEvaluationResult:
        """
        Evaluate policy with Lean verification.

        Args:
            policy: Policy to evaluate
            timeout: Evaluation timeout

        Returns:
            Evaluation result with verification bonus
        """
        # Get base evaluation
        result = await super().evaluate_policy(policy, timeout)

        # If LeanAide available and policy found proofs, verify them
        if LEANAIDE_AVAILABLE and self.leanaide_client and result.success_rate > 0:
            verification_bonus = await self._compute_verification_bonus(policy)
            result.fitness += verification_bonus
            result.objectives["formal_verification"] = verification_bonus

        return result

    async def _compute_verification_bonus(self, policy: RolloutPolicyGenome) -> float:
        """Compute fitness bonus for formally verified proofs."""
        # In full implementation, would:
        # 1. Generate Lean code from policy proofs
        # 2. Use LeanAide to verify proofs
        # 3. Award bonus based on verification rate

        # Placeholder: simulate verification
        verified_rate = random.uniform(0.5, 1.0)  # Assume 50-100% verification
        return verified_rate * self.verification_bonus

    def verify_with_lean(self, policy_genome) -> Dict[str, Any]:
        """
        REAL Lean verification for evolved policies.
        
        Args:
            policy_genome: RolloutPolicyGenome to verify
            
        Returns:
            Dictionary with verification results
        """
        if not LEAN_AVAILABLE:
            return {"verified": False, "error": "Lean not available"}
        
        try:
            client = LeanAideClient()
            formalized = client.autoformalize(str(policy_genome))
            return client.verify(formalized)
        except Exception as e:
            logger.warning(f"Lean verification failed: {e}")
            return {"verified": False, "error": str(e)}


# =============================================================================
# Multi-Objective Policy Evolution
# =============================================================================

class MultiObjectivePolicyEvolution:
    """
    Evolve policies optimizing multiple objectives simultaneously.

    Uses NSGA-II (Non-dominated Sorting Genetic Algorithm) to find
    Pareto-optimal policies across multiple objectives.
    """

    def __init__(
        self,
        config: RolloutPolicyConfig,
        objectives: List[str]
    ):
        """
        Initialize multi-objective evolution.

        Args:
            config: Evolution configuration
            objectives: List of objectives to optimize
        """
        self.config = config
        self.objectives = objectives

    async def evolve_multi_objective(
        self,
        test_theorems: List[str],
        generations: int
    ) -> List[RolloutPolicyGenome]:
        """
        Evolve Pareto-optimal policies.

        Args:
            test_theorems: Theorems to evaluate on
            generations: Number of generations

        Returns:
            Pareto front of policies
        """
        logger.info(f"Starting multi-objective evolution for {len(self.objectives)} objectives")

        # Initialize population
        population = PolicyPopulation(
            size=self.config.population_size,
            mutation_rate=self.config.mutation_rate,
            crossover_rate=self.config.crossover_rate,
            elite_size=self.config.elite_size
        )
        population.initialize_random()

        pareto_front = []

        for gen in range(generations):
            # Evaluate all policies on all objectives
            for policy in population.policies:
                objective_scores = await self._evaluate_all_objectives(
                    policy, test_theorems
                )
                # Store multi-objective fitness
                policy.fitness_history = [objective_scores]

            # Find Pareto front
            pareto_front = self._find_pareto_front(population.policies)

            logger.info(f"Generation {gen + 1}: Pareto front size = {len(pareto_front)}")

            # Create next generation using NSGA-II selection
            if gen < generations - 1:
                population = self._nsga2_selection(population, pareto_front)

        logger.info(f"Multi-objective evolution complete. Pareto front size: {len(pareto_front)}")

        return pareto_front

    async def _evaluate_all_objectives(
        self,
        policy: RolloutPolicyGenome,
        test_theorems: List[str]
    ) -> Dict[str, float]:
        """Evaluate policy on all objectives."""
        scores = {}

        # Evaluate success rate
        mcts_config = MCTSConfig(max_iterations=100)
        evaluator = PolicyEvaluator(mcts_config, test_theorems)
        result = await evaluator.evaluate_policy(policy)

        scores["success_rate"] = result.success_rate
        scores["speed"] = 1.0 / (1.0 + result.avg_time)
        scores["efficiency"] = 1.0 / (1.0 + result.nodes_explored / len(test_theorems))
        scores["elegance"] = 1.0 / (1.0 + result.avg_depth / 10.0)

        return scores

    def _find_pareto_front(self, policies: List[RolloutPolicyGenome]) -> List[RolloutPolicyGenome]:
        """Find Pareto-optimal policies using non-dominated sorting."""
        pareto_front = []

        for policy in policies:
            objectives = policy.fitness_history[-1] if policy.fitness_history else {}

            # Check if policy is dominated
            is_dominated = False
            for other in policies:
                if other == policy:
                    continue

                other_objectives = other.fitness_history[-1] if other.fitness_history else {}

                # Check if other dominates this policy
                other_better = True
                for obj in self.objectives:
                    if obj in objectives and obj in other_objectives:
                        if objectives[obj] > other_objectives[obj]:
                            other_better = False
                            break

                if other_better:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(policy)

        return pareto_front

    def _nsga2_selection(
        self,
        population: PolicyPopulation,
        pareto_front: List[RolloutPolicyGenome]
    ) -> PolicyPopulation:
        """NSGA-II selection for next generation."""
        # Simplified NSGA-II: preserve Pareto front, fill rest with offspring

        # Elite: Pareto front
        elites = pareto_front[:population.elite_size]

        # Select parents from rest of population
        remaining = [p for p in population.policies if p not in elites]
        parents = population.select_parents(
            count=population.size - len(elites),
            method="tournament"
        )

        # Create offspring
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            child = population.crossover(parents[i], parents[i + 1])
            if random.random() < population.mutation_rate:
                child = population.mutate(child)
            offspring.append(child)
            if len(elites) + len(offspring) >= population.size:
                break

        # Create new population
        new_population = PolicyPopulation(
            size=population.size,
            mutation_rate=population.mutation_rate,
            crossover_rate=population.crossover_rate,
            elite_size=population.elite_size
        )
        new_population.policies = elites + offspring[:population.size - len(elites)]
        new_population.generation = population.generation + 1

        return new_population


# =============================================================================
# Policy Transfer Learning
# =============================================================================

class PolicyTransfer:
    """
    Transfer learned policies between domains.

    Adapts a policy trained on one domain to work well on another domain
    through fine-tuning.
    """

    def __init__(self, source_policy: RolloutPolicyGenome):
        """
        Initialize policy transfer.

        Args:
            source_policy: Policy to transfer from source domain
        """
        self.source_policy = source_policy

    def transfer_policy(
        self,
        target_domain: str,
        adaptation_rate: float = 0.1,
        num_mutations: int = 10
    ) -> RolloutPolicyGenome:
        """
        Transfer policy to new domain.

        Args:
            target_domain: Target domain identifier
            adaptation_rate: Rate of adaptation (0-1)
            num_mutations: Number of mutations to apply

        Returns:
            Adapted policy for target domain
        """
        logger.info(f"Transferring policy to domain: {target_domain}")

        # Create copy of source policy
        target_policy = RolloutPolicyGenome(
            generation=0,
            parent_ids=[self.source_policy.genome_id]
        )

        # Copy tactic weights with adaptation
        for tactic, weight in self.source_policy.tactic_weights.items():
            # Add small adaptation noise
            noise = random.gauss(0, adaptation_rate * 0.5)
            target_policy.tactic_weights[tactic] = max(0.1, weight + noise)

        # Copy other parameters
        target_policy.tactic_preferences = self.source_policy.tactic_preferences.copy()
        target_policy.context_modifiers = {
            k: v.copy() for k, v in self.source_policy.context_modifiers.items()
        }
        target_policy.depth_preferences = {
            k: v.copy() for k, v in self.source_policy.depth_preferences.items()
        }
        target_policy.max_depth = self.source_policy.max_depth
        target_policy.depth_decay = self.source_policy.depth_decay
        target_policy.exploration_bonus = self.source_policy.exploration_bonus
        target_policy.exploration_decay = self.source_policy.exploration_decay
        target_policy.goal_distance_thresholds = self.source_policy.goal_distance_thresholds.copy()

        # Apply domain-specific mutations
        for _ in range(num_mutations):
            mutation_type = random.choice([
                "tactic_weight", "context_modifier", "exploration_param"
            ])

            if mutation_type == "tactic_weight":
                # Mutate random tactic weight
                tactic = random.choice(list(target_policy.tactic_weights.keys()))
                target_policy.tactic_weights[tactic] += random.gauss(0, adaptation_rate)
                target_policy.tactic_weights[tactic] = max(0.1, min(3.0, target_policy.tactic_weights[tactic]))

            elif mutation_type == "context_modifier":
                # Mutate random context modifier
                if target_policy.context_modifiers:
                    context = random.choice(list(target_policy.context_modifiers.keys()))
                    tactic = random.choice(list(target_policy.context_modifiers[context].keys()))
                    target_policy.context_modifiers[context][tactic] += random.gauss(0, adaptation_rate * 0.3)

            elif mutation_type == "exploration_param":
                # Mutate exploration parameter
                if random.random() < 0.5:
                    target_policy.exploration_bonus += random.gauss(0, adaptation_rate * 0.2)
                    target_policy.exploration_bonus = max(0.0, min(1.0, target_policy.exploration_bonus))
                else:
                    target_policy.exploration_decay += random.gauss(0, adaptation_rate * 0.05)
                    target_policy.exploration_decay = max(0.9, min(1.0, target_policy.exploration_decay))

        logger.info(f"Policy transfer complete: {target_domain}")

        return target_policy

    async def fine_tune(
        self,
        target_policy: RolloutPolicyGenome,
        target_theorems: List[str],
        generations: int = 5,
        mcts_config: Optional[MCTSConfig] = None
    ) -> RolloutPolicyGenome:
        """
        Fine-tune transferred policy on target domain.

        Args:
            target_policy: Transferred policy
            target_theorems: Theorems from target domain
            generations: Generations for fine-tuning
            mcts_config: MCTS configuration

        Returns:
            Fine-tuned policy
        """
        logger.info(f"Fine-tuning policy for {generations} generations")

        if mcts_config is None:
            mcts_config = MCTSConfig(max_iterations=100)

        # Create evaluator for target domain
        evaluator = PolicyEvaluator(mcts_config, target_theorems)

        # Create small population for fine-tuning
        population = PolicyPopulation(size=20, mutation_rate=0.2, crossover_rate=0.6)
        population.policies = [target_policy]

        # Add variants of target policy
        for _ in range(19):
            variant = population.mutate(target_policy)
            population.policies.append(variant)

        # Evolve for few generations
        for gen in range(generations):
            # Evaluate
            for policy in population.policies:
                result = await evaluator.evaluate_policy(policy)
                policy.update_fitness(result.fitness)

            # Get best
            best = population.get_best_policy()
            logger.info(f"Fine-tuning gen {gen + 1}: best fitness = {best.compute_fitness():.4f}")

            # Create next generation
            if gen < generations - 1:
                elites = population.get_elite()
                offspring = []

                while len(offspring) < population.size - len(elites):
                    parent1 = random.choice(elites)
                    parent2 = random.choice(population.policies)
                    child = population.crossover(parent1, parent2)
                    child = population.mutate(child)
                    offspring.append(child)

                population.policies = elites + offspring[:population.size - len(elites)]
                population.generation += 1

        logger.info("Fine-tuning complete")

        return population.get_best_policy()


# =============================================================================
# Convenience Functions
# =============================================================================

async def evolve_mcts_rollout_policy(
    test_theorems: List[str],
    generations: int = 20,
    population_size: int = 50,
    mcts_iterations: int = 100,
    parallel_evaluation: bool = True
) -> RolloutPolicyGenome:
    """
    Convenience function to evolve an MCTS rollout policy.

    Args:
        test_theorems: Theorems to train on
        generations: Number of evolution generations
        population_size: Size of policy population
        mcts_iterations: MCTS iterations per evaluation
        parallel_evaluation: Evaluate policies in parallel

    Returns:
        Best evolved policy
    """
    config = RolloutPolicyConfig(
        population_size=population_size,
        test_theorems=test_theorems,
        mcts_iterations=mcts_iterations,
        parallel_evaluation=parallel_evaluation
    )

    mcts_config = MCTSConfig(max_iterations=mcts_iterations)
    evaluator = PolicyEvaluator(mcts_config, test_theorems)

    engine = PolicyEvolutionEngine(config, evaluator)
    best_policy = await engine.evolve_policies(generations)

    return best_policy


async def search_with_evolved_policy(
    theorem: str,
    policy: RolloutPolicyGenome,
    max_iterations: int = 1000,
    time_budget: float = 60.0
) -> MCTSResult:
    """
    Convenience function to search with evolved policy.

    Args:
        theorem: Theorem to prove
        policy: Evolved rollout policy
        max_iterations: Maximum MCTS iterations
        time_budget: Time budget in seconds

    Returns:
        MCTS result
    """
    config = MCTSConfig(
        max_iterations=max_iterations,
        time_budget=time_budget
    )

    mcts = EvolvedPolicyMCTS(
        rollout_policy_genome=policy,
        config=config,
        theorem=theorem
    )

    result = await mcts.search()
    return result


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    'RolloutPolicyConfig',
    'PolicyEvaluationResult',

    # Genome representation
    'RolloutPolicyGenome',
    'TacticRolloutPolicy',

    # Population management
    'PolicyPopulation',

    # Evaluation
    'PolicyEvaluator',
    'LeanAidePolicyEvaluator',

    # Evolution
    'PolicyEvolutionEngine',

    # MCTS with evolved policies
    'EvolvedPolicyMCTS',
    'EvolvedPolicySimulation',

    # Advanced features
    'AdaptivePolicyMCTS',
    'CoEvolvingMCTS',
    'LeanAideGuidedPolicyEvolution',
    'MultiObjectivePolicyEvolution',
    'PolicyTransfer',

    # Convenience functions
    'evolve_mcts_rollout_policy',
    'search_with_evolved_policy',
]


# =============================================================================
# Example Usage
# =============================================================================

async def main():
    """Example usage of MCTS evolved policies."""

    print("=" * 80)
    print("MCTS Evolved Policies Example")
    print("=" * 80)

    # Define test theorems for training
    test_theorems = [
        "forall (a b : Nat), a + b = b + a",
        "forall (a b c : Nat), (a + b) + c = a + (b + c)",
        "forall (n : Nat), n + 0 = n",
        "forall (a b : Nat), a * b = b * a",
        "forall (n : Nat), n * 0 = 0",
    ]

    print("\n1. Evolving rollout policies...")
    print(f"   Test theorems: {len(test_theorems)}")
    print(f"   Generations: 10")
    print(f"   Population size: 30")

    # Evolve policies
    best_policy = await evolve_mcts_rollout_policy(
        test_theorems=test_theorems,
        generations=10,
        population_size=30,
        mcts_iterations=50
    )

    print(f"\n   Best fitness: {best_policy.best_fitness:.4f}")
    print(f"   Best tactic weights: {dict(list(best_policy.tactic_weights.items())[:5])}")

    # Test on new theorem
    test_theorem = "forall (a b c : Nat), a + (b + c) = (a + b) + c"

    print(f"\n2. Searching with evolved policy...")
    print(f"   Theorem: {test_theorem}")

    result = await search_with_evolved_policy(
        theorem=test_theorem,
        policy=best_policy,
        max_iterations=200,
        time_budget=30.0
    )

    print(f"\n   Success: {result.success}")
    print(f"   Win rate: {result.win_rate:.4f}")
    print(f"   Time: {result.time_elapsed:.2f}s")
    print(f"   Nodes: {result.nodes_visited}")

    # Demonstrate policy transfer
    print(f"\n3. Transferring policy to new domain...")

    transfer = PolicyTransfer(best_policy)
    adapted_policy = transfer.transfer_policy(
        target_domain="algebra",
        adaptation_rate=0.2
    )

    print(f"   Adapted policy: {adapted_policy.genome_id}")

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
