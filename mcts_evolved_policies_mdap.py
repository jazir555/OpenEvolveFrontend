"""
MCTS Evolved Policies with MDAP/MAKER Integration

This module integrates MDAP (Multi-Agent voting) and MAKER (zero-error guarantees) with
the evolved rollout policies approach for MCTS proof search.

Core Concept:
Evolve rollout policies using MDAP multi-agent evaluation and MAKER voting for
consensus-based policy selection, creating a robust hybrid system where:
- Evolution searches for better policies
- Multi-agent evaluation provides robustness
- MAKER voting ensures zero-error convergence

Key Features:
1. MDAP-Enhanced Policy Genome: Multi-agent weights and preferences
2. Multi-Agent Policy Evaluator: Each policy evaluated by multiple agents
3. MAKER Voting for Policy Selection: Consensus-based selection
4. MDAP Policy Evolution: Evolution with multi-agent evaluation
5. MDAP-Enhanced Evolved Policy MCTS: Multi-agent consensus during search
6. Decomposition-Enhanced Policies: MDAP task decomposition integration
7. LeanAide Integration with MDAP: Formal verification guides evolution
8. Red-Flagging for Invalid Policies: Filter out low-quality policies
9. Performance Tracking: Track across agents and generations

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

# Import MCTS evolved policies base
try:
    from mcts_evolved_policies import (
        RolloutPolicyGenome,
        RolloutPolicyConfig,
        PolicyEvaluationResult,
        PolicyEvaluator,
        PolicyPopulation,
        PolicyEvolutionEngine,
        TacticRolloutPolicy,
        EvolvedPolicyMCTS,
        MCTSConfig,
        MCTSResult,
        MCTSNode,
        ProofState,
        Tactic,
    )
    EVOLVED_POLICIES_AVAILABLE = True
except ImportError:
    EVOLVED_POLICIES_AVAILABLE = False
    logging.warning("MCTS evolved policies module not available")

# Import MDAP/MAKER components
try:
    from mdap_engine import (
        MDAPConfig,
        MDAPRunResult,
        MDAPVoteResult,
        MDAPStep,
        MDAPTask,
        RedFlagger,
        RedFlagRules,
    )
    MDAP_AVAILABLE = True
except ImportError:
    MDAP_AVAILABLE = False
    logging.warning("MDAP engine not available")

try:
    from mdap_maker_complete import (
        VoteCollector,
        VotingEngine,
        TaskDecomposition,
        MAKERRunMetrics,
    )
    MAKER_AVAILABLE = True
except ImportError:
    MAKER_AVAILABLE = False
    logging.warning("MAKER complete implementation not available")

# Import LeanAide client
try:
    from leanaide_client import LeanAideClient, LeanAideConfig
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logging.warning("LeanAide client not available")

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class MDAPPolicyConfig:
    """
    Configuration for MDAP-enhanced policy evolution.

    Attributes:
        # MDAP parameters
        num_agents: Number of agents for multi-agent evaluation
        voting_strategy: Strategy for MAKER voting (first_k_ahead, majority, weighted)
        k_ahead: K parameter for first-to-ahead-by-k voting
        consensus_threshold: Minimum consensus level for decision

        # Agent specialization
        enable_specialization: Allow agents to specialize in certain tactics
        specialization_rate: Probability of developing specialization
        agent_diversity: Encourage diversity among agents

        # Decomposition
        enable_decomposition: Enable MDAP task decomposition
        decomposition_depth: Maximum depth for decomposition
        atomic_threshold: Threshold for treating subtask as atomic

        # Red-flagging
        enable_red_flagging: Enable red-flagging for invalid policies
        max_policy_depth: Maximum allowed policy depth
        min_diversity: Minimum tactic diversity required

        # Performance tracking
        track_agent_performance: Track individual agent performance
        track_consensus_history: Track consensus over time
        save_agent_statistics: Save per-agent statistics

        # LeanAide integration
        use_lean_verification: Use Lean formal verification
        verification_bonus: Fitness bonus for verified policies
        verification_threshold: Minimum fitness for verification
    """
    # MDAP parameters
    num_agents: int = 5
    voting_strategy: str = "first_k_ahead"
    k_ahead: int = 3
    consensus_threshold: float = 0.75

    # Agent specialization
    enable_specialization: bool = True
    specialization_rate: float = 0.1
    agent_diversity: float = 0.8

    # Decomposition
    enable_decomposition: bool = True
    decomposition_depth: int = 3
    atomic_threshold: float = 0.9

    # Red-flagging
    enable_red_flagging: bool = True
    max_policy_depth: int = 100
    min_diversity: float = 0.2

    # Performance tracking
    track_agent_performance: bool = True
    track_consensus_history: bool = True
    save_agent_statistics: bool = True

    # LeanAide integration
    use_lean_verification: bool = True
    verification_bonus: float = 0.5
    verification_threshold: float = 0.8


@dataclass
class MDAPPolicyEvaluation:
    """
    Result of multi-agent policy evaluation.

    Attributes:
        policy_id: ID of evaluated policy
        consensus_fitness: Fitness based on agent consensus
        agent_results: Results from each individual agent
        voting_details: Details about voting process
        agreement_level: Level of agreement among agents (0-1)
        confidence: Confidence in consensus result
        generation: Generation number
        decomposition_used: Whether decomposition was used
        red_flags: Number of red flags raised
    """
    policy_id: str
    consensus_fitness: float
    agent_results: List[Dict[str, Any]]
    voting_details: Dict[str, Any]
    agreement_level: float
    confidence: float
    generation: int
    decomposition_used: bool = False
    red_flags: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class VotingDetails:
    """
    Details about MAKER voting process.

    Attributes:
        voting_strategy: Strategy used
        total_rounds: Number of voting rounds
        votes_per_candidate: Vote counts
        winner: Winning candidate
        winning_margin: Margin of victory
        consensus_reached: Whether consensus was achieved
        agreement_distribution: Distribution of agreement
        agent_participation: Which agents participated
    """
    voting_strategy: str
    total_rounds: int
    votes_per_candidate: Dict[str, int]
    winner: str
    winning_margin: int
    consensus_reached: bool
    agreement_distribution: Dict[str, float]
    agent_participation: List[int]
    tiebreaker_used: bool = False


@dataclass
class AgentSpecialization:
    """
    Agent specialization information.

    Attributes:
        agent_id: Agent identifier
        specialized_tactics: Tactics this agent specializes in
        specialization_strength: Strength of specialization (0-1)
        preferred_contexts: Contexts agent prefers
        performance_metrics: Historical performance
    """
    agent_id: str
    specialized_tactics: List[str]
    specialization_strength: float
    preferred_contexts: List[str]
    performance_metrics: Dict[str, float]


# =============================================================================
# MDAP-Enhanced Policy Genome
# =============================================================================

class MDAPRolloutPolicyGenome(RolloutPolicyGenome):
    """
    Policy genome with MDAP multi-agent weights and capabilities.

    Extends RolloutPolicyGenome with:
    - Agent-specific preferences and weights
    - MAKER voting parameters
    - Decomposition capabilities
    - Multi-agent consensus tracking
    """

    def __init__(self, **kwargs):
        """Initialize MDAP-enhanced policy genome."""
        super().__init__(**kwargs)

        # MDAP-specific additions
        self.agent_preferences: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.agent_confidence: Dict[str, float] = defaultdict(lambda: 0.5)
        self.agent_specialization: Dict[str, List[str]] = defaultdict(list)

        # MAKER voting parameters
        self.voting_strategy: str = kwargs.get('voting_strategy', 'first_k_ahead')
        self.consensus_threshold: float = kwargs.get('consensus_threshold', 0.75)
        self.k_ahead: int = kwargs.get('k_ahead', 3)

        # Decomposition preferences
        self.enable_decomposition: bool = kwargs.get('enable_decomposition', True)
        self.decomposition_depth: int = kwargs.get('decomposition_depth', 3)

        # Multi-agent history
        self.agent_performance_history: Dict[str, List[float]] = defaultdict(list)
        self.consensus_history: List[float] = []

    def get_agent_weights(self, agent_id: str) -> Dict[str, float]:
        """
        Get tactic weights for specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Dictionary mapping tactics to weights for this agent
        """
        # Base weights modified by agent preferences
        agent_weights = self.tactic_weights.copy()

        # Apply agent-specific preferences
        if agent_id in self.agent_preferences:
            for tactic, preference in self.agent_preferences[agent_id].items():
                agent_weights[tactic] = agent_weights.get(tactic, 1.0) + preference

        # Apply specialization
        if agent_id in self.agent_specialization:
            for tactic in self.agent_specialization[agent_id]:
                if tactic in agent_weights:
                    agent_weights[tactic] *= 1.5

        return agent_weights

    def compute_agent_consensus(
        self,
        agent_votes: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Compute consensus across agents for tactic selection.

        Args:
            agent_votes: Dictionary mapping agent_id to their voted tactics

        Returns:
            Dictionary mapping tactics to consensus scores
        """
        consensus_scores = defaultdict(float)

        # Count votes for each tactic
        vote_counts = defaultdict(int)
        for agent_id, tactics in agent_votes.items():
            for tactic in tactics:
                vote_counts[tactic] += 1

        # Weight votes by agent confidence
        for agent_id, tactics in agent_votes.items():
            confidence = self.agent_confidence.get(agent_id, 0.5)
            for tactic in tactics:
                consensus_scores[tactic] += confidence

        # Normalize
        if consensus_scores:
            max_score = max(consensus_scores.values())
            if max_score > 0:
                for tactic in consensus_scores:
                    consensus_scores[tactic] /= max_score

        return dict(consensus_scores)

    def get_agent_policy_variant(self, agent_id: str) -> 'MDAPRolloutPolicyGenome':
        """
        Get a policy variant for a specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Policy genome specialized for this agent
        """
        variant = MDAPRolloutPolicyGenome()

        # Copy base parameters
        variant.tactic_weights = self.get_agent_weights(agent_id)
        variant.tactic_preferences = self.tactic_preferences.copy()
        variant.context_modifiers = {
            k: v.copy() for k, v in self.context_modifiers.items()
        }
        variant.max_depth = self.max_depth
        variant.depth_decay = self.depth_decay
        variant.depth_preferences = {
            k: v.copy() for k, v in self.depth_preferences.items()
        }
        variant.exploration_bonus = self.exploration_bonus
        variant.exploration_decay = self.exploration_decay
        variant.exploration_strategy = self.exploration_strategy

        # Agent-specific parameters
        variant.agent_preferences = self.agent_preferences.copy()
        variant.agent_confidence = self.agent_confidence.copy()
        variant.agent_specialization = self.agent_specialization.copy()

        # Metadata
        variant.generation = self.generation
        variant.parent_ids = self.parent_ids.copy()
        variant.genome_id = f"{self.genome_id}_agent_{agent_id}"

        return variant

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'agent_preferences': dict(self.agent_preferences),
            'agent_confidence': dict(self.agent_confidence),
            'agent_specialization': dict(self.agent_specialization),
            'voting_strategy': self.voting_strategy,
            'consensus_threshold': self.consensus_threshold,
            'k_ahead': self.k_ahead,
            'enable_decomposition': self.enable_decomposition,
            'decomposition_depth': self.decomposition_depth,
            'agent_performance_history': dict(self.agent_performance_history),
            'consensus_history': self.consensus_history,
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MDAPRolloutPolicyGenome':
        """Create genome from dictionary."""
        genome = cls(**data)
        return genome


# =============================================================================
# Multi-Agent Policy Evaluator
# =============================================================================

class MDAPPolicyEvaluator(PolicyEvaluator):
    """
    Evaluate policies using multiple MDAP agents.

    Each policy is evaluated by multiple agents, and MAKER voting is used
    to reach consensus on the policy's fitness.
    """

    def __init__(
        self,
        mcts_config: MCTSConfig,
        test_theorems: List[str],
        num_agents: int = 5,
        voting_strategy: str = "first_k_ahead",
        leanaide_client: Optional[Any] = None,
        mdap_config: Optional[MDAPPolicyConfig] = None
    ):
        """
        Initialize MDAP policy evaluator.

        Args:
            mcts_config: MCTS configuration
            test_theorems: Theorems to evaluate on
            num_agents: Number of agents for evaluation
            voting_strategy: MAKER voting strategy
            leanaide_client: Optional LeanAide client
            mdap_config: MDAP configuration
        """
        super().__init__(mcts_config, test_theorems, leanaide_client)

        self.num_agents = num_agents
        self.voting_strategy = voting_strategy
        self.mdap_config = mdap_config or MDAPPolicyConfig()

        # Agent specializations
        self.agent_specializations: Dict[str, AgentSpecialization] = {}
        self._initialize_agent_specializations()

        # MAKER voting engine
        if MAKER_AVAILABLE:
            self.vote_collector = VoteCollector()
            self.voting_engine = VotingEngine(
                vote_collector=self.vote_collector,
                enable_first_to_ahead=(voting_strategy == "first_k_ahead")
            )

    def _initialize_agent_specializations(self) -> None:
        """Initialize agent specializations."""
        tactic_groups = [
            ["intros", "simp", "rw"],
            ["apply", "exact", "refine"],
            ["cases", "induction", "constructor"],
            ["aesop", "linarith", "ring"],
            ["exists", "have", "show"]
        ]

        for i in range(self.num_agents):
            agent_id = f"agent_{i}"
            specialized_tactics = tactic_groups[i % len(tactic_groups)]

            self.agent_specializations[agent_id] = AgentSpecialization(
                agent_id=agent_id,
                specialized_tactics=specialized_tactics,
                specialization_strength=0.7 + random.uniform(-0.1, 0.1),
                preferred_contexts=random.sample([
                    "has_equality", "has_implication", "has_forall",
                    "has_exists", "has_conjunction"
                ], k=2),
                performance_metrics={}
            )

    async def evaluate_policy_mdap(
        self,
        policy: MDAPRolloutPolicyGenome,
        test_theorems: List[str],
        mcts_config: MCTSConfig,
        timeout: float = 30.0
    ) -> MDAPPolicyEvaluation:
        """
        Evaluate policy using MDAP multi-agent approach.

        Args:
            policy: Policy to evaluate
            test_theorems: Theorems to evaluate on
            mcts_config: MCTS configuration
            timeout: Timeout per agent evaluation

        Returns:
            MDAPPolicyEvaluation with multi-agent results
        """
        logger.info(f"Evaluating policy {policy.genome_id} with {self.num_agents} agents")

        agent_results = []

        # Evaluate with each agent
        for agent_id in range(self.num_agents):
            agent_id_str = f"agent_{agent_id}"

            # Get agent-specific policy variant
            agent_policy = policy.get_agent_policy_variant(agent_id_str)

            # Apply agent specialization
            if agent_id_str in self.agent_specializations:
                spec = self.agent_specializations[agent_id_str]
                agent_policy.agent_specialization[agent_id_str] = spec.specialized_tactics

            # Run MCTS with agent policy
            result = await self._run_mcts_with_policy(
                agent_policy,
                test_theorems,
                mcts_config,
                timeout
            )

            # Collect metrics
            agent_results.append({
                "agent_id": agent_id_str,
                "success_rate": result.success_rate,
                "avg_depth": result.avg_depth,
                "time": result.avg_time,
                "nodes_explored": result.nodes_explored,
                "fitness": result.fitness,
                "objectives": result.objectives
            })

            logger.debug(f"Agent {agent_id_str}: fitness={result.fitness:.4f}")

        # Apply MAKER voting for consensus
        consensus = self._apply_maker_voting(agent_results, policy)

        # Compute agreement level
        agreement_level = self._compute_agreement_level(agent_results)

        # Create evaluation result
        evaluation = MDAPPolicyEvaluation(
            policy_id=policy.genome_id,
            consensus_fitness=consensus["fitness"],
            agent_results=agent_results,
            voting_details=consensus["details"],
            agreement_level=agreement_level,
            confidence=consensus["confidence"],
            generation=policy.generation,
            decomposition_used=policy.enable_decomposition,
            red_flags=consensus.get("red_flags", 0)
        )

        # Update policy fitness
        policy.update_fitness(consensus["fitness"])
        policy.consensus_history.append(consensus["fitness"])

        # Update agent performance history
        for result in agent_results:
            agent_id = result["agent_id"]
            policy.agent_performance_history[agent_id].append(result["fitness"])

        logger.info(
            f"Policy evaluation complete: consensus={consensus['fitness']:.4f}, "
            f"agreement={agreement_level:.2f}"
        )

        return evaluation

    async def _run_mcts_with_policy(
        self,
        policy: MDAPRolloutPolicyGenome,
        test_theorems: List[str],
        mcts_config: MCTSConfig,
        timeout: float
    ) -> PolicyEvaluationResult:
        """Run MCTS evaluation with given policy."""
        # Use base evaluator
        rollout_policy = TacticRolloutPolicy(policy)

        total_success = 0
        total_depth = 0
        total_time = 0.0
        total_nodes = 0

        for theorem in test_theorems:
            try:
                # Simulate MCTS run (placeholder)
                if EVOLVED_POLICIES_AVAILABLE:
                    result = await self._evaluate_on_theorem(
                        theorem,
                        rollout_policy,
                        timeout
                    )
                else:
                    # Fallback simulation
                    result = self._simulate_evaluation(theorem, policy, timeout)

                if result.success:
                    total_success += 1
                    total_depth += result.tree_depth
                    total_nodes += result.nodes_visited

                total_time += result.time_elapsed

            except Exception as e:
                logger.warning(f"Evaluation failed for theorem {theorem}: {e}")
                total_time += timeout

        # Compute metrics
        num_theorems = len(test_theorems)
        success_rate = total_success / num_theorems if num_theorems > 0 else 0.0
        avg_depth = total_depth / total_success if total_success > 0 else 0.0
        avg_time = total_time / num_theorems if num_theorems > 0 else 0.0

        fitness = self._compute_fitness(
            success_rate=success_rate,
            avg_depth=avg_depth,
            avg_time=avg_time,
            nodes_explored=total_nodes
        )

        return PolicyEvaluationResult(
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

    def _simulate_evaluation(
        self,
        theorem: str,
        policy: MDAPRolloutPolicyGenome,
        timeout: float
    ) -> MCTSResult:
        """Simulate evaluation when MCTS not available."""
        # Simulate based on policy quality
        base_success = random.uniform(0.3, 0.7)

        # Boost based on policy fitness
        if policy.fitness_history:
            avg_fitness = sum(policy.fitness_history) / len(policy.fitness_history)
            base_success *= (1.0 + avg_fitness * 0.5)

        return MCTSResult(
            success=random.random() < base_success,
            search_iterations=100,
            time_elapsed=random.uniform(1.0, min(timeout, 10.0)),
            nodes_visited=random.randint(100, 1000),
            tree_depth=random.randint(5, 20),
            win_rate=base_success
        )

    def _apply_maker_voting(
        self,
        agent_results: List[Dict[str, Any]],
        policy: MDAPRolloutPolicyGenome
    ) -> Dict[str, Any]:
        """
        Apply MAKER voting to combine agent results.

        Args:
            agent_results: Results from each agent
            policy: Policy being evaluated

        Returns:
            Consensus result with voting details
        """
        # Collect votes for fitness ranges
        fitness_votes = defaultdict(int)

        for result in agent_results:
            # Discretize fitness into ranges
            fitness = result["fitness"]
            fitness_range = int(fitness * 10) / 10  # Round to 0.1 precision
            fitness_votes[fitness_range] += 1

        # Apply voting strategy
        if self.voting_strategy == "first_k_ahead":
            winner = self._first_k_ahead_voting(fitness_votes, policy.k_ahead)
        elif self.voting_strategy == "majority":
            winner = self._majority_voting(fitness_votes)
        elif self.voting_strategy == "weighted":
            winner = self._weighted_voting(agent_results, fitness_votes)
        else:
            winner = max(fitness_votes.keys())

        # Compute confidence
        total_votes = sum(fitness_votes.values())
        confidence = fitness_votes[winner] / total_votes if total_votes > 0 else 0.0

        # Voting details
        details = VotingDetails(
            voting_strategy=self.voting_strategy,
            total_rounds=len(agent_results),
            votes_per_candidate=dict(fitness_votes),
            winner=str(winner),
            winning_margin=fitness_votes[winner] - max(
                [v for k, v in fitness_votes.items() if k != winner],
                default=0
            ),
            consensus_reached=confidence >= policy.consensus_threshold,
            agreement_distribution={k: v/total_votes for k, v in fitness_votes.items()},
            agent_participation=list(range(len(agent_results)))
        )

        return {
            "fitness": winner,
            "confidence": confidence,
            "details": asdict(details),
            "red_flags": 0
        }

    def _first_k_ahead_voting(
        self,
        votes: Dict[float, int],
        k: int
    ) -> float:
        """First-to-ahead-by-k voting."""
        while True:
            # Check if any candidate is ahead by k
            for candidate, count in votes.items():
                max_other = max(
                    [v for c, v in votes.items() if c != candidate],
                    default=0
                )
                if count >= max_other + k:
                    return candidate

            # If no clear winner, add random vote (simulation)
            random_candidate = random.choice(list(votes.keys()))
            votes[random_candidate] += 1

            # Prevent infinite loop
            if sum(votes.values()) > 100:
                return max(votes.keys(), key=votes.get)

    def _majority_voting(self, votes: Dict[float, int]) -> float:
        """Simple majority voting."""
        return max(votes.keys(), key=votes.get)

    def _weighted_voting(
        self,
        agent_results: List[Dict[str, Any]],
        votes: Dict[float, int]
    ) -> float:
        """Weighted voting based on agent confidence."""
        weighted_scores = defaultdict(float)

        for result in agent_results:
            fitness = result["fitness"]
            fitness_range = int(fitness * 10) / 10
            # Weight by success rate
            weighted_scores[fitness_range] += result["success_rate"]

        return max(weighted_scores.keys(), key=weighted_scores.get)

    def _compute_agreement_level(
        self,
        agent_results: List[Dict[str, Any]]
    ) -> float:
        """Compute level of agreement among agents (0-1)."""
        if not agent_results:
            return 0.0

        fitness_values = [r["fitness"] for r in agent_results]

        if len(fitness_values) == 1:
            return 1.0

        # Compute coefficient of variation (normalized)
        mean_fitness = sum(fitness_values) / len(fitness_values)
        if mean_fitness == 0:
            return 0.0

        std_fitness = (sum((f - mean_fitness)**2 for f in fitness_values) / len(fitness_values))**0.5
        cv = std_fitness / mean_fitness if mean_fitness > 0 else float('inf')

        # Convert to agreement score (lower CV = higher agreement)
        agreement = max(0.0, 1.0 - cv)
        return agreement


# =============================================================================
# MAKER Voting for Policy Selection
# =============================================================================

class PolicyVotingEngine:
    """
    MAKER voting engine for selecting best policies.

    Implements first-to-ahead-by-k voting for policy selection from
    a population of candidates.
    """

    def __init__(
        self,
        k_ahead: int = 3,
        max_agents: int = 7,
        voting_strategy: str = "first_k_ahead"
    ):
        """
        Initialize policy voting engine.

        Args:
            k_ahead: K parameter for first-to-ahead-by-k voting
            max_agents: Maximum number of agents
            voting_strategy: Voting strategy to use
        """
        self.k_ahead = k_ahead
        self.max_agents = max_agents
        self.voting_strategy = voting_strategy

        if MAKER_AVAILABLE:
            self.vote_collector = VoteCollector()
            self.voting_engine = VotingEngine(
                vote_collector=self.vote_collector,
                enable_first_to_ahead=(voting_strategy == "first_k_ahead")
            )

    def vote_on_best_policy(
        self,
        policies: List[MDAPRolloutPolicyGenome],
        evaluations: List[MDAPPolicyEvaluation]
    ) -> Tuple[MDAPRolloutPolicyGenome, VotingDetails]:
        """
        Use MAKER voting to select best policy.

        Args:
            policies: List of candidate policies
            evaluations: Corresponding evaluations

        Returns:
            Tuple of (best_policy, voting_details)
        """
        logger.info(f"Voting on best policy from {len(policies)} candidates")

        # Collect votes from all agents
        votes: Dict[str, int] = defaultdict(int)

        for eval in evaluations:
            for agent_result in eval.agent_results:
                # Agent votes for policy based on performance
                if agent_result["success_rate"] > 0.7:
                    policy_id = eval.policy_id
                    votes[policy_id] += 1

                    # Check if ahead by k
                    max_other = max(
                        [v for pid, v in votes.items() if pid != policy_id],
                        default=0
                    )

                    if votes[policy_id] >= max_other + self.k_ahead:
                        # Winner found
                        best_policy = next(
                            (p for p in policies if p.genome_id == policy_id),
                            policies[0]
                        )

                        details = VotingDetails(
                            voting_strategy=self.voting_strategy,
                            total_rounds=sum(votes.values()),
                            votes_per_candidate=dict(votes),
                            winner=policy_id,
                            winning_margin=votes[policy_id] - max_other,
                            consensus_reached=True,
                            agreement_distribution={
                                k: v/sum(votes.values()) for k, v in votes.items()
                            },
                            agent_participation=list(range(len(evaluations)))
                        )

                        logger.info(f"Winner found: {policy_id} with {votes[policy_id]} votes")

                        return best_policy, details

        # If no clear winner, return highest voted
        winner_id = max(votes.keys(), key=lambda k: votes[k])
        best_policy = next(
            (p for p in policies if p.genome_id == winner_id),
            policies[0]
        )

        max_other = max(
            [v for pid, v in votes.items() if pid != winner_id],
            default=0
        )

        details = VotingDetails(
            voting_strategy=self.voting_strategy,
            total_rounds=sum(votes.values()),
            votes_per_candidate=dict(votes),
            winner=winner_id,
            winning_margin=votes[winner_id] - max_other,
            consensus_reached=votes[winner_id] >= max_other + self.k_ahead,
            agreement_distribution={
                k: v/sum(votes.values()) for k, v in votes.items()
            },
            agent_participation=list(range(len(evaluations))),
            tiebreaker_used=True
        )

        logger.info(f"Best policy (no consensus): {winner_id} with {votes[winner_id]} votes")

        return best_policy, details


# =============================================================================
# MDAP Policy Evolution Engine
# =============================================================================

class MDAPPolicyEvolutionEngine(PolicyEvolutionEngine):
    """
    Evolve policies with MDAP multi-agent evaluation.

    Extends PolicyEvolutionEngine to use MDAP evaluation and MAKER voting
    for parent selection and policy evolution.
    """

    def __init__(
        self,
        config: RolloutPolicyConfig,
        evaluator: MDAPPolicyEvaluator,
        mdap_config: Optional[MDAPPolicyConfig] = None
    ):
        """
        Initialize MDAP evolution engine.

        Args:
            config: Evolution configuration
            evaluator: MDAP policy evaluator
            mdap_config: MDAP configuration
        """
        # Initialize base engine
        super().__init__(config, evaluator)

        self.mdap_config = mdap_config or MDAPPolicyConfig()
        self.mdap_evaluator = evaluator

        # Voting engine
        self.voting_engine = PolicyVotingEngine(
            k_ahead=self.mdap_config.k_ahead,
            voting_strategy=self.mdap_config.voting_strategy
        )

        # Red flagger
        if self.mdap_config.enable_red_flagging:
            self.red_flagger = PolicyRedFlagger()
        else:
            self.red_flagger = None

    async def evolve_policies_mdap(
        self,
        initial_population: int,
        generations: int,
        test_theorems: List[str],
        mcts_config: MCTSConfig,
        num_agents: int = 5,
        voting_strategy: str = "first_k_ahead"
    ) -> MDAPRolloutPolicyGenome:
        """
        Evolve policies using MDAP evaluation.

        Args:
            initial_population: Initial population size
            generations: Number of generations
            test_theorems: Theorems for evaluation
            mcts_config: MCTS configuration
            num_agents: Number of MDAP agents
            voting_strategy: MAKER voting strategy

        Returns:
            Best evolved policy
        """
        logger.info(f"Starting MDAP policy evolution: {generations} generations, {initial_population} policies")

        # Initialize MDAP population
        population = self._initialize_mdap_population(initial_population)

        best_policy = None
        best_consensus = 0.0

        for generation in range(generations):
            logger.info(f"\n=== Generation {generation + 1}/{generations} ===")

            # Evaluate all policies with MDAP
            evaluations = []
            for policy in population:
                # Check red flags
                if self.red_flagger:
                    is_flagged, flags = self.red_flagger.check_policy(policy, None)
                    if is_flagged:
                        logger.warning(f"Policy {policy.genome_id} red-flagged: {flags}")
                        # Give very low fitness
                        policy.update_fitness(0.0)
                        continue

                # Evaluate with MDAP
                eval_result = await self.mdap_evaluator.evaluate_policy_mdap(
                    policy,
                    test_theorems,
                    mcts_config
                )
                policy.fitness = eval_result.consensus_fitness
                evaluations.append(eval_result)

            # Track best
            current_best = max(population, key=lambda p: p.compute_fitness())
            if current_best.compute_fitness() > best_consensus:
                best_policy = current_best
                best_consensus = current_best.compute_fitness()
                logger.info(f"New best consensus fitness: {best_consensus:.4f}")

            # Log generation statistics
            self._log_mdap_generation_stats(population, evaluations)

            # Create next generation
            if generation < generations - 1:
                population = await self._create_next_generation_mdap(
                    population,
                    evaluations,
                    voting_strategy
                )

        logger.info(f"\nMDAP evolution complete. Best consensus fitness: {best_consensus:.4f}")

        return best_policy or population[0]

    def _initialize_mdap_population(
        self,
        size: int
    ) -> List[MDAPRolloutPolicyGenome]:
        """Initialize MDAP policy population."""
        population = []

        for _ in range(size):
            policy = MDAPRolloutPolicyGenome(generation=0)

            # Randomize parameters
            self._randomize_mdap_policy(policy)

            population.append(policy)

        return population

    def _randomize_mdap_policy(self, policy: MDAPRolloutPolicyGenome) -> None:
        """Randomize MDAP policy parameters."""
        # Randomize base policy
        for tactic in policy.tactic_weights:
            policy.tactic_weights[tactic] = random.uniform(0.1, 2.0)
            policy.tactic_preferences[tactic] = random.uniform(-0.5, 0.5)

        # Randomize agent preferences
        num_agents = self.mdap_config.num_agents
        for agent_id in range(num_agents):
            agent_str = f"agent_{agent_id}"
            for tactic in policy.tactic_weights:
                policy.agent_preferences[agent_str][tactic] = random.uniform(-0.3, 0.3)
            policy.agent_confidence[agent_str] = random.uniform(0.3, 0.9)

            # Add specialization
            if random.random() < self.mdap_config.specialization_rate:
                tactics = list(policy.tactic_weights.keys())
                specialized = random.sample(tactics, k=random.randint(1, 3))
                policy.agent_specialization[agent_str] = specialized

        # Randomize depth and exploration
        policy.max_depth = random.randint(50, 150)
        policy.depth_decay = random.uniform(0.9, 0.99)
        policy.exploration_bonus = random.uniform(0.1, 1.0)
        policy.exploration_decay = random.uniform(0.9, 0.99)

        # Randomize voting parameters
        policy.voting_strategy = random.choice([
            "first_k_ahead", "majority", "weighted"
        ])
        policy.consensus_threshold = random.uniform(0.6, 0.9)
        policy.k_ahead = random.randint(2, 5)

        # Randomize decomposition
        policy.enable_decomposition = random.random() < 0.7
        policy.decomposition_depth = random.randint(2, 4)

    async def _create_next_generation_mdap(
        self,
        population: List[MDAPRolloutPolicyGenome],
        evaluations: List[MDAPPolicyEvaluation],
        voting_strategy: str
    ) -> List[MDAPRolloutPolicyGenome]:
        """Create next generation using MAKER voting for parent selection."""
        # Select elites
        elites = sorted(
            population,
            key=lambda p: p.compute_fitness(),
            reverse=True
        )[:self.config.elite_size]

        # Select parents using voting
        parents = self._select_parents_with_voting(
            population,
            evaluations,
            voting_strategy
        )

        # Create offspring
        offspring = []
        num_offspring = self.config.population_size - len(elites)

        while len(offspring) < num_offspring:
            parent1, parent2 = random.sample(parents, 2)

            # Crossover
            if random.random() < self.population.crossover_rate:
                child1, child2 = self.mdap_crossover(parent1, parent2)
            else:
                child1 = parent1.get_agent_policy_variant("new")
                child2 = parent2.get_agent_policy_variant("new")

            # Mutation
            if random.random() < self.population.mutation_rate:
                child1 = self.mdap_mutate(child1)
            if random.random() < self.population.mutation_rate:
                child2 = self.mdap_mutate(child2)

            offspring.extend([child1, child2])

        # Combine
        next_gen = elites + offspring[:num_offspring]

        # Update generation
        for policy in next_gen:
            policy.generation += 1

        return next_gen

    def _select_parents_with_voting(
        self,
        population: List[MDAPRolloutPolicyGenome],
        evaluations: List[MDAPPolicyEvaluation],
        voting_strategy: str
    ) -> List[MDAPRolloutPolicyGenome]:
        """Select parents using MAKER voting."""
        # Use voting to rank policies
        _, voting_details = self.voting_engine.vote_on_best_policy(
            population,
            evaluations
        )

        # Sort by vote count
        vote_counts = voting_details.votes_per_candidate

        sorted_population = sorted(
            population,
            key=lambda p: vote_counts.get(p.genome_id, 0),
            reverse=True
        )

        # Select top half as parents
        num_parents = max(4, len(population) // 2)
        return sorted_population[:num_parents]

    def mdap_crossover(
        self,
        parent1: MDAPRolloutPolicyGenome,
        parent2: MDAPRolloutPolicyGenome
    ) -> Tuple[MDAPRolloutPolicyGenome, MDAPRolloutPolicyGenome]:
        """
        MDAP-aware crossover of two policies.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Tuple of two children
        """
        child1 = MDAPRolloutPolicyGenome(
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.genome_id, parent2.genome_id]
        )

        child2 = MDAPRolloutPolicyGenome(
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.genome_id, parent2.genome_id]
        )

        # Crossover tactic weights
        for tactic in parent1.tactic_weights:
            w1 = parent1.tactic_weights.get(tactic, 1.0)
            w2 = parent2.tactic_weights.get(tactic, 1.0)
            alpha = random.random()

            child1.tactic_weights[tactic] = alpha * w1 + (1 - alpha) * w2
            child2.tactic_weights[tactic] = (1 - alpha) * w1 + alpha * w2

        # Crossover agent preferences
        for agent_id in parent1.agent_preferences:
            if agent_id in parent2.agent_preferences:
                for tactic in parent1.agent_preferences[agent_id]:
                    p1 = parent1.agent_preferences[agent_id].get(tactic, 0.0)
                    p2 = parent2.agent_preferences[agent_id].get(tactic, 0.0)

                    child1.agent_preferences[agent_id][tactic] = (p1 + p2) / 2
                    child2.agent_preferences[agent_id][tactic] = (p1 + p2) / 2

        # Crossover other parameters
        child1.max_depth = random.choice([parent1.max_depth, parent2.max_depth])
        child2.max_depth = random.choice([parent1.max_depth, parent2.max_depth])

        child1.depth_decay = (parent1.depth_decay + parent2.depth_decay) / 2
        child2.depth_decay = (parent1.depth_decay + parent2.depth_decay) / 2

        # Crossover voting parameters
        child1.voting_strategy = random.choice([parent1.voting_strategy, parent2.voting_strategy])
        child2.voting_strategy = random.choice([parent1.voting_strategy, parent2.voting_strategy])

        child1.k_ahead = random.choice([parent1.k_ahead, parent2.k_ahead])
        child2.k_ahead = random.choice([parent1.k_ahead, parent2.k_ahead])

        return child1, child2

    def mdap_mutate(
        self,
        policy: MDAPRolloutPolicyGenome
    ) -> MDAPRolloutPolicyGenome:
        """
        MDAP-aware mutation of policy.

        Args:
            policy: Policy to mutate

        Returns:
            Mutated policy
        """
        mutated = policy.get_agent_policy_variant("mutated")
        mutated.mutation_count = policy.mutation_count + 1

        # Mutate tactic weights
        for tactic in mutated.tactic_weights:
            if random.random() < self.population.mutation_rate:
                mutation = random.gauss(0, 0.2)
                mutated.tactic_weights[tactic] += mutation
                mutated.tactic_weights[tactic] = max(0.1, min(3.0, mutated.tactic_weights[tactic]))

        # Mutate agent preferences
        for agent_id in mutated.agent_preferences:
            if random.random() < self.population.mutation_rate:
                for tactic in mutated.agent_preferences[agent_id]:
                    if random.random() < self.population.mutation_rate:
                        mutation = random.gauss(0, 0.1)
                        mutated.agent_preferences[agent_id][tactic] += mutation
                        mutated.agent_preferences[agent_id][tactic] = max(-0.5, min(0.5, mutated.agent_preferences[agent_id][tactic]))

        # Mutate agent confidence
        for agent_id in mutated.agent_confidence:
            if random.random() < self.population.mutation_rate * 0.5:
                mutation = random.gauss(0, 0.05)
                mutated.agent_confidence[agent_id] += mutation
                mutated.agent_confidence[agent_id] = max(0.0, min(1.0, mutated.agent_confidence[agent_id]))

        # Mutate depth and exploration
        if random.random() < self.population.mutation_rate:
            mutated.max_depth += int(random.gauss(0, 10))
            mutated.max_depth = max(20, min(200, mutated.max_depth))

        if random.random() < self.population.mutation_rate:
            mutated.depth_decay += random.gauss(0, 0.02)
            mutated.depth_decay = max(0.8, min(1.0, mutated.depth_decay))

        # Mutate voting parameters
        if random.random() < self.population.mutation_rate * 0.3:
            mutated.k_ahead = random.choice([2, 3, 4, 5])

        if random.random() < self.population.mutation_rate * 0.2:
            mutated.voting_strategy = random.choice(["first_k_ahead", "majority", "weighted"])

        return mutated

    def _log_mdap_generation_stats(
        self,
        population: List[MDAPRolloutPolicyGenome],
        evaluations: List[MDAPPolicyEvaluation]
    ) -> None:
        """Log MDAP-specific generation statistics."""
        fitness_values = [p.compute_fitness() for p in population]

        # Compute consensus statistics
        consensus_values = [e.consensus_fitness for e in evaluations]
        agreement_values = [e.agreement_level for e in evaluations]
        red_flags = sum(e.red_flags for e in evaluations)

        logger.info(f"  Population size: {len(population)}")
        logger.info(f"  Best fitness: {max(fitness_values):.4f}")
        logger.info(f"  Avg fitness: {sum(fitness_values)/len(fitness_values):.4f}")
        logger.info(f"  Best consensus: {max(consensus_values):.4f}")
        logger.info(f"  Avg consensus: {sum(consensus_values)/len(consensus_values):.4f}")
        logger.info(f"  Avg agreement: {sum(agreement_values)/len(agreement_values):.2f}")
        logger.info(f"  Red flags: {red_flags}")


# =============================================================================
# MDAP-Enhanced Evolved Policy MCTS
# =============================================================================

class MDAPEvolvedPolicyMCTS(EvolvedPolicyMCTS):
    """
    MCTS with MDAP-enhanced evolved policies.

    At each decision point, multiple agents suggest tactics and MAKER voting
    is used to reach consensus on the best tactic.
    """

    def __init__(
        self,
        policy: MDAPRolloutPolicyGenome,
        num_agents: int = 5,
        voting_strategy: str = "first_k_ahead",
        config: Optional[MCTSConfig] = None,
        theorem: Optional[str] = None
    ):
        """
        Initialize MDAP-enhanced MCTS.

        Args:
            policy: MDAP-enhanced policy genome
            num_agents: Number of agents for consensus
            voting_strategy: MAKER voting strategy
            config: MCTS configuration
            theorem: Theorem to prove
        """
        if EVOLVED_POLICIES_AVAILABLE:
            super().__init__(policy, config or MCTSConfig(), theorem or "")
        else:
            self.policy = policy
            self.config = config or MCTSConfig()
            self.theorem = theorem or ""

        self.num_agents = num_agents
        self.voting_strategy = voting_strategy

        # Create voting engine
        self.voting_engine = PolicyVotingEngine(
            k_ahead=policy.k_ahead,
            voting_strategy=voting_strategy
        )

        # Agent policies
        self.agent_policies = {
            f"agent_{i}": policy.get_agent_policy_variant(f"agent_{i}")
            for i in range(num_agents)
        }

        # Rollout policies for each agent
        if EVOLVED_POLICIES_AVAILABLE:
            self.agent_rollout_policies = {
                agent_id: TacticRolloutPolicy(agent_policy)
                for agent_id, agent_policy in self.agent_policies.items()
            }

    async def search_mdap(
        self,
        context: Optional[ProofState] = None,
        leanaide_client: Optional[Any] = None
    ) -> MCTSResult:
        """
        Search using MDAP-enhanced policy.

        Args:
            context: Initial proof context
            leanaide_client: Optional LeanAide client

        Returns:
            MCTS result
        """
        logger.info(f"Starting MDAP-enhanced MCTS search with {self.num_agents} agents")

        if not EVOLVED_POLICIES_AVAILABLE:
            # Simulate result
            return MCTSResult(
                success=random.random() > 0.5,
                search_iterations=self.config.max_iterations,
                time_elapsed=random.uniform(1.0, 10.0),
                nodes_visited=random.randint(100, 1000),
                tree_depth=random.randint(5, 30),
                win_rate=random.uniform(0.0, 1.0)
            )

        # Initialize root
        if context is None:
            context = ProofState(goals=[self.theorem], depth=0)

        root = MCTSNode(state=context)
        best_node = None
        best_value = 0.0

        start_time = time.time()

        for iteration in range(self.config.max_iterations):
            # Check time budget
            if time.time() - start_time >= self.config.time_budget:
                break

            # Selection
            node = self._select(root)

            # Expansion
            if not node.is_fully_expanded():
                node = self._expand(node)

            # Multi-agent simulation
            value = await self._mdap_simulation(node, context)

            # Backpropagation
            self._backpropagate(node, value)

            # Track best
            if value > best_value:
                best_value = value
                best_node = node

        elapsed = time.time() - start_time

        # Compile result
        return MCTSResult(
            success=best_node is not None and best_node.is_terminal if best_node else False,
            search_iterations=iteration + 1,
            time_elapsed=elapsed,
            nodes_visited=root.N if root else 0,
            tree_depth=self._get_tree_depth(root) if root else 0,
            win_rate=best_value,
            proof_path=self._get_best_path(root) if root else []
        )

    async def _mdap_simulation(
        self,
        node: MCTSNode,
        context: ProofState
    ) -> float:
        """
        Simulation with multi-agent consensus.

        Args:
            node: Current node
            context: Proof context

        Returns:
            Simulated value (0-1)
        """
        agent_suggestions = []

        # Each agent suggests tactic
        for agent_id, rollout_policy in self.agent_rollout_policies.items():
            # Get agent-specific tactic suggestion
            available_tactics = self._get_available_tactics(node.state)
            tactic = rollout_policy.select_tactic(node.state, available_tactics)

            agent_suggestions.append({
                "agent_id": agent_id,
                "tactic": tactic,
                "confidence": self.policy.agent_confidence.get(agent_id, 0.5)
            })

        # MAKER voting on best tactic
        selected_tactic = self._apply_maker_voting_tactics(agent_suggestions)

        # Execute selected tactic
        result = await self._execute_tactic(selected_tactic, node.state)

        return result.value

    def _apply_maker_voting_tactics(
        self,
        agent_suggestions: List[Dict[str, Any]]
    ) -> str:
        """
        Apply MAKER voting to select best tactic.

        Args:
            agent_suggestions: List of agent suggestions

        Returns:
            Selected tactic
        """
        # Collect votes
        tactic_votes = defaultdict(int)

        for suggestion in agent_suggestions:
            tactic = suggestion["tactic"]
            confidence = suggestion["confidence"]
            tactic_votes[tactic] += confidence

        # First-to-k-ahead voting
        k = self.policy.k_ahead

        while True:
            for tactic, votes in tactic_votes.items():
                max_other = max(
                    [v for t, v in tactic_votes.items() if t != tactic],
                    default=0
                )
                if votes >= max_other + k:
                    return tactic

            # Add some randomness if no clear winner
            random_suggestion = random.choice(agent_suggestions)
            tactic_votes[random_suggestion["tactic"]] += 1

            # Prevent infinite loop
            if sum(tactic_votes.values()) > 50:
                return max(tactic_votes.keys(), key=tactic_votes.get)

    def _get_available_tactics(self, state: ProofState) -> List[str]:
        """Get available tactics for current state."""
        basic_tactics = [
            "intros", "simp", "rw", "apply", "exact",
            "cases", "induction", "constructor", "exists",
            "have", "suffices", "show", "calc",
            "aesop", "linarith", "ring", "omega", "norm_num"
        ]
        return basic_tactics

    async def _execute_tactic(
        self,
        tactic: str,
        state: ProofState
    ) -> MCTSResult:
        """Execute tactic and return result."""
        # Simulate tactic execution
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
            if random.random() > 0.7:
                new_state.goals = []
        elif tactic in ["cases", "induction"]:
            if new_state.goals and len(new_state.goals) == 1:
                new_state.goals = new_state.goals * 2

        new_state.is_complete = len(new_state.goals) == 0

        # Return result
        value = 1.0 if new_state.is_complete else 0.5

        return MCTSResult(
            success=new_state.is_complete,
            search_iterations=1,
            time_elapsed=0.1,
            nodes_visited=1,
            tree_depth=new_state.depth,
            win_rate=value
        )

    def _select(self, node: MCTSNode) -> MCTSNode:
        """UCB selection."""
        while not node.is_leaf:
            # Select child with highest UCB
            children = [child for child in node.children.values() if child is not None]

            if not children:
                break

            # UCB1 formula
            log_total = math.log(node.N) if node.N > 0 else 0

            def ucb(child):
                if child.N == 0:
                    return float('inf')
                return child.Q / child.N + math.sqrt(2 * log_total / child.N)

            node = max(children, key=ucb)

        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding a child."""
        # Find unexpanded action
        available_tactics = self._get_available_tactics(node.state)

        for tactic in available_tactics:
            if tactic not in node.children:
                # Create new child
                new_state = ProofState(
                    goals=node.state.goals.copy(),
                    context=node.state.context.copy(),
                    tactics_sequence=node.state.tactics_sequence.copy(),
                    depth=node.state.depth + 1
                )

                child = MCTSNode(state=new_state, parent=node)
                node.children[tactic] = child
                return child

        return node

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate value up the tree."""
        while node is not None:
            node.N += 1
            node.Q += value
            node = node.parent

    def _get_tree_depth(self, node: MCTSNode) -> int:
        """Get maximum depth of tree from node."""
        if not node.children:
            return node.state.depth

        return max(
            self._get_tree_depth(child)
            for child in node.children.values()
            if child is not None
        )

    def _get_best_path(self, node: MCTSNode) -> List[str]:
        """Get best path from root to leaf."""
        path = []

        while node.children:
            # Select child with highest visits
            best_child = max(
                [c for c in node.children.values() if c is not None],
                key=lambda c: c.N,
                default=None
            )

            if best_child is None:
                break

            # Find tactic that led to this child
            for tactic, child in node.children.items():
                if child == best_child:
                    path.append(tactic)
                    break

            node = best_child

        return path


# =============================================================================
# Decomposition-Enhanced Policies
# =============================================================================

class DecompositionEnhancedPolicy:
    """
    Policy enhanced with MDAP task decomposition.

    Decomposes complex theorems into subtasks and solves each with evolved policies.
    """

    def __init__(
        self,
        base_policy: MDAPRolloutPolicyGenome,
        decomposer: Optional[Any] = None,
        max_depth: int = 3
    ):
        """
        Initialize decomposition-enhanced policy.

        Args:
            base_policy: Base MDAP policy
            decomposer: Optional MDAP decomposer
            max_depth: Maximum decomposition depth
        """
        self.base_policy = base_policy
        self.decomposer = decomposer
        self.max_depth = max_depth

    async def execute_with_decomposition(
        self,
        theorem: str,
        context: Optional[ProofState] = None,
        mcts_config: Optional[MCTSConfig] = None
    ) -> MCTSResult:
        """
        Execute proof with decomposition.

        Args:
            theorem: Theorem to prove
            context: Initial proof context
            mcts_config: MCTS configuration

        Returns:
            MCTS result
        """
        logger.info(f"Executing with decomposition: {theorem}")

        if not self.base_policy.enable_decomposition:
            # Direct execution without decomposition
            return await self._execute_direct(theorem, context, mcts_config)

        # Decompose theorem
        subtasks = await self._decompose_theorem(theorem, depth=0)

        if not subtasks or len(subtasks) <= 1:
            # Treat as atomic
            return await self._execute_direct(theorem, context, mcts_config)

        logger.info(f"Decomposed into {len(subtasks)} subtasks")

        # Solve each subtask
        results = []
        for i, subtask in enumerate(subtasks):
            logger.info(f"Solving subtask {i + 1}/{len(subtasks)}")

            result = await self._solve_subtask(
                subtask,
                context,
                mcts_config
            )

            results.append(result)

            if not result.success:
                logger.warning(f"Subtask {i + 1} failed")
                # Could continue or abort

        # Combine results
        return self._combine_results(results)

    async def _decompose_theorem(
        self,
        theorem: str,
        depth: int
    ) -> List[str]:
        """
        Decompose theorem into subtasks.

        Args:
            theorem: Theorem to decompose
            depth: Current decomposition depth

        Returns:
            List of subtask theorems
        """
        if depth >= self.max_depth:
            return [theorem]

        # Simple decomposition heuristic
        # Look for conjunctions, implications, etc.

        if " and " in theorem.lower() or " ∧ " in theorem:
            # Split on conjunction
            parts = theorem.split(" and ") if " and " in theorem else theorem.split(" ∧ ")
            subtasks = []

            for part in parts:
                further_subtasks = await self._decompose_theorem(part.strip(), depth + 1)
                subtasks.extend(further_subtasks)

            return subtasks

        elif " -> " in theorem or " → " in theorem:
            # For implication, try to prove consequent assuming antecedent
            if " -> " in theorem:
                parts = theorem.split(" -> ")
            else:
                parts = theorem.split(" → ")

            if len(parts) == 2:
                # Focus on proving consequent
                return await self._decompose_theorem(parts[1].strip(), depth + 1)

        # No decomposition possible
        return [theorem]

    async def _solve_subtask(
        self,
        subtask: str,
        context: Optional[ProofState],
        mcts_config: Optional[MCTSConfig]
    ) -> MCTSResult:
        """Solve a single subtask."""
        # Create MCTS with base policy
        mcts = MDAPEvolvedPolicyMCTS(
            policy=self.base_policy,
            num_agents=5,
            voting_strategy=self.base_policy.voting_strategy,
            config=mcts_config or MCTSConfig(),
            theorem=subtask
        )

        result = await mcts.search_mdap(context)

        return result

    async def _execute_direct(
        self,
        theorem: str,
        context: Optional[ProofState],
        mcts_config: Optional[MCTSConfig]
    ) -> MCTSResult:
        """Execute without decomposition."""
        mcts = MDAPEvolvedPolicyMCTS(
            policy=self.base_policy,
            num_agents=5,
            voting_strategy=self.base_policy.voting_strategy,
            config=mcts_config or MCTSConfig(),
            theorem=theorem
        )

        return await mcts.search_mdap(context)

    def _combine_results(self, results: List[MCTSResult]) -> MCTSResult:
        """Combine results from multiple subtasks."""
        if not results:
            return MCTSResult(success=False)

        # Overall success if all succeeded
        all_success = all(r.success for r in results)

        # Combine metrics
        total_time = sum(r.time_elapsed for r in results)
        total_nodes = sum(r.nodes_visited for r in results)
        max_depth = max(r.tree_depth for r in results)

        # Average win rate
        avg_win_rate = sum(r.win_rate for r in results) / len(results)

        # Combine proof paths
        combined_path = []
        for result in results:
            if result.proof_path:
                combined_path.extend(result.proof_path)

        return MCTSResult(
            success=all_success,
            search_iterations=sum(r.search_iterations for r in results),
            time_elapsed=total_time,
            nodes_visited=total_nodes,
            tree_depth=max_depth,
            win_rate=avg_win_rate,
            proof_path=combined_path
        )


# =============================================================================
# LeanAide Integration with MDAP
# =============================================================================

class LeanAideMDAPPolicyEvolution:
    """
    Policy evolution with LeanAide and MDAP.

    Uses Lean formal verification to guide policy evolution and MDAP
    for robust evaluation.
    """

    def __init__(
        self,
        mdap_config: MDAPPolicyConfig,
        leanaide_client: Optional[LeanAideClient] = None
    ):
        """
        Initialize LeanAide-MDAP evolution.

        Args:
            mdap_config: MDAP configuration
            leanaide_client: Optional LeanAide client
        """
        self.mdap_config = mdap_config
        self.leanaide_client = leanaide_client

        if not LEANAIDE_AVAILABLE:
            logger.warning("LeanAide not available")

    async def evolve_with_verification(
        self,
        test_theorems: List[str],
        generations: int,
        num_agents: int = 5,
        mcts_config: Optional[MCTSConfig] = None
    ) -> MDAPRolloutPolicyGenome:
        """
        Evolve policies with Lean formal verification and MDAP.

        Args:
            test_theorems: Theorems for evaluation
            generations: Number of generations
            num_agents: Number of MDAP agents
            mcts_config: MCTS configuration

        Returns:
            Best evolved policy
        """
        logger.info("Starting LeanAide-MDAP policy evolution")

        # Create evaluator
        mcts_config = mcts_config or MCTSConfig()
        evaluator = MDAPPolicyEvaluator(
            mcts_config=mcts_config,
            test_theorems=test_theorems,
            num_agents=num_agents,
            leanaide_client=self.leanaide_client
        )

        # Create evolution engine
        config = RolloutPolicyConfig(
            population_size=30,
            test_theorems=test_theorems
        )

        engine = MDAPPolicyEvolutionEngine(
            config=config,
            evaluator=evaluator,
            mdap_config=self.mdap_config
        )

        best_policy = None

        for generation in range(generations):
            logger.info(f"\n=== Generation {generation + 1}/{generations} ===")

            # Evaluate with MDAP
            evaluations = await self._evaluate_with_mdap(
                engine,
                test_theorems,
                mcts_config
            )

            # Verify best policies with Lean
            if LEANAIDE_AVAILABLE and self.leanaide_client:
                for policy, eval in zip(engine.population.policies, evaluations):
                    if eval.consensus_fitness > self.mdap_config.verification_threshold:
                        # Verify with Lean
                        verification = await self._verify_policy_with_lean(
                            policy,
                            test_theorems[0]  # Verify on first theorem
                        )

                        # Bonus for verified proofs
                        if verification.get("is_valid", False):
                            policy.update_fitness(
                                policy.compute_fitness() * (1 + self.mdap_config.verification_bonus)
                            )
                            logger.info(f"Policy {policy.genome_id} verified - bonus applied")

            # Track best
            current_best = max(
                engine.population.policies,
                key=lambda p: p.compute_fitness()
            )

            if best_policy is None or current_best.compute_fitness() > best_policy.compute_fitness():
                best_policy = current_best
                logger.info(f"New best policy: fitness={best_policy.compute_fitness():.4f}")

            # Create next generation
            if generation < generations - 1:
                await self._create_next_generation(engine, evaluations)

        logger.info("LeanAide-MDAP evolution complete")

        return best_policy or engine.population.policies[0]

    async def _evaluate_with_mdap(
        self,
        engine: MDAPPolicyEvolutionEngine,
        test_theorems: List[str],
        mcts_config: MCTSConfig
    ) -> List[MDAPPolicyEvaluation]:
        """Evaluate population with MDAP."""
        evaluations = []

        for policy in engine.population.policies:
            eval_result = await engine.mdap_evaluator.evaluate_policy_mdap(
                policy,
                test_theorems,
                mcts_config
            )
            evaluations.append(eval_result)

        return evaluations

    async def _verify_policy_with_lean(
        self,
        policy: MDAPRolloutPolicyGenome,
        theorem: str
    ) -> Dict[str, Any]:
        """
        Verify policy-generated proof with Lean.

        Args:
            policy: Policy to verify
            theorem: Theorem to prove

        Returns:
            Verification result
        """
        if not LEANAIDE_AVAILABLE or not self.leanaide_client:
            # Simulate verification
            return {
                "is_valid": random.random() > 0.5,
                "confidence": random.uniform(0.5, 1.0)
            }

        # Generate proof from policy
        mcts = MDAPEvolvedPolicyMCTS(
            policy=policy,
            theorem=theorem
        )

        result = await mcts.search_mdap()

        if result.success and result.proof_path:
            # Convert proof to Lean code
            lean_code = self._policy_to_lean_code(policy, result.proof_path)

            # Verify with LeanAide
            try:
                verification = await self.leanaide_client.verify_proof(
                    lean_code,
                    theorem
                )

                return {
                    "is_valid": verification.is_valid,
                    "confidence": verification.confidence,
                    "errors": verification.errors
                }
            except Exception as e:
                logger.warning(f"Lean verification failed: {e}")
                return {"is_valid": False, "error": str(e)}

        return {"is_valid": False}

    def _policy_to_lean_code(
        self,
        policy: MDAPRolloutPolicyGenome,
        proof_path: List[str]
    ) -> str:
        """Convert policy proof to Lean code."""
        # Simple conversion
        lean_lines = [
            "theorem proved :",
            "  by",
        ]

        for tactic in proof_path:
            lean_lines.append(f"    {tactic}")

        lean_lines.append("    done")

        return "\n".join(lean_lines)

    async def _create_next_generation(
        self,
        engine: MDAPPolicyEvolutionEngine,
        evaluations: List[MDAPPolicyEvaluation]
    ) -> None:
        """Create next generation."""
        # Select elites
        elites = sorted(
            engine.population.policies,
            key=lambda p: p.compute_fitness(),
            reverse=True
        )[:engine.config.elite_size]

        # Select parents
        parents = engine._select_parents_with_voting(
            engine.population.policies,
            evaluations,
            self.mdap_config.voting_strategy
        )

        # Create offspring
        offspring = []
        num_offspring = engine.config.population_size - len(elites)

        while len(offspring) < num_offspring:
            parent1, parent2 = random.sample(parents, 2)

            if random.random() < engine.population.crossover_rate:
                child1, child2 = engine.mdap_crossover(parent1, parent2)
            else:
                child1 = parent1.get_agent_policy_variant("new")
                child2 = parent2.get_agent_policy_variant("new")

            if random.random() < engine.population.mutation_rate:
                child1 = engine.mdap_mutate(child1)
            if random.random() < engine.population.mutation_rate:
                child2 = engine.mdap_mutate(child2)

            offspring.extend([child1, child2])

        # Update population
        next_gen = elites + offspring[:num_offspring]
        engine.population.policies = next_gen
        engine.population.generation += 1


# =============================================================================
# Policy Red-Flagger
# =============================================================================

class PolicyRedFlagger:
    """
    Red-flag invalid or low-quality policies.

    Identifies policies that should be flagged for:
    - Invalid tactic combinations
    - Excessive depth
    - Low diversity
    - Low agent confidence
    """

    def __init__(self):
        """Initialize policy red flagger."""
        self.flagged_policies: Set[str] = set()
        self.flagging_history: Dict[str, List[str]] = defaultdict(list)

    def check_policy(
        self,
        policy: MDAPRolloutPolicyGenome,
        context: Optional[Any]
    ) -> Tuple[bool, List[str]]:
        """
        Check if policy should be red-flagged.

        Args:
            policy: Policy to check
            context: Optional proof context

        Returns:
            Tuple of (is_flagged, list of reasons)
        """
        flags = []

        # Check for invalid tactic combinations
        if self._has_invalid_combinations(policy):
            flags.append("Invalid tactic combinations")

        # Check for excessive depth
        if policy.max_depth > 100:
            flags.append(f"Excessive depth: {policy.max_depth}")

        # Check for low diversity
        diversity = self._compute_diversity(policy)
        if diversity < 0.2:
            flags.append(f"Low tactic diversity: {diversity:.2f}")

        # Check consensus level
        if policy.agent_confidence:
            min_confidence = min(policy.agent_confidence.values())
            if min_confidence < 0.3:
                flags.append(f"Low agent confidence: {min_confidence:.2f}")

        # Check for NaN or infinite values
        if self._has_invalid_values(policy):
            flags.append("Invalid values (NaN or infinite)")

        is_flagged = len(flags) > 0

        if is_flagged:
            self.flagged_policies.add(policy.genome_id)
            self.flagging_history[policy.genome_id].extend(flags)

        return is_flagged, flags

    def _has_invalid_combinations(self, policy: MDAPRolloutPolicyGenome) -> bool:
        """Check for invalid tactic combinations."""
        # Check if weights are valid
        for tactic, weight in policy.tactic_weights.items():
            if not isinstance(weight, (int, float)) or math.isnan(weight) or math.isinf(weight):
                return True
            if weight < 0 or weight > 10:
                return True

        return False

    def _compute_diversity(self, policy: MDAPRolloutPolicyGenome) -> float:
        """Compute tactic diversity (entropy)."""
        weights = list(policy.tactic_weights.values())

        if not weights:
            return 0.0

        # Normalize
        total = sum(weights)
        if total == 0:
            return 0.0

        probs = [w / total for w in weights]

        # Compute entropy
        entropy = -sum(p * math.log(p) if p > 0 else 0 for p in probs)

        # Normalize to 0-1
        max_entropy = math.log(len(weights))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return normalized_entropy

    def _has_invalid_values(self, policy: MDAPRolloutPolicyGenome) -> bool:
        """Check for NaN or infinite values."""
        # Check tactic weights
        for weight in policy.tactic_weights.values():
            if math.isnan(weight) or math.isinf(weight):
                return True

        # Check agent confidence
        for confidence in policy.agent_confidence.values():
            if math.isnan(confidence) or math.isinf(confidence):
                return True

        # Check fitness history
        for fitness in policy.fitness_history:
            if math.isnan(fitness) or math.isinf(fitness):
                return True

        return False


# =============================================================================
# Performance Tracking
# =============================================================================

class MDAPPolicyPerformanceTracker:
    """
    Track policy performance across agents and generations.

    Maintains history of evaluations, agent statistics, and consensus trends.
    """

    def __init__(self):
        """Initialize performance tracker."""
        self.history: Dict[str, List[Dict]] = defaultdict(list)
        self.agent_statistics: Dict[str, Dict] = defaultdict(dict)
        self.consensus_history: List[float] = []
        self.generation_stats: List[Dict] = []

    def track_evaluation(
        self,
        policy_id: str,
        evaluation: MDAPPolicyEvaluation
    ):
        """
        Track evaluation results.

        Args:
            policy_id: Policy identifier
            evaluation: Evaluation result
        """
        record = {
            "generation": evaluation.generation,
            "consensus_fitness": evaluation.consensus_fitness,
            "agreement_level": evaluation.agreement_level,
            "agent_performance": evaluation.agent_results,
            "confidence": evaluation.confidence,
            "red_flags": evaluation.red_flags,
            "timestamp": evaluation.timestamp
        }

        self.history[policy_id].append(record)
        self.consensus_history.append(evaluation.consensus_fitness)

        # Update agent statistics
        for agent_result in evaluation.agent_results:
            agent_id = agent_result["agent_id"]

            if agent_id not in self.agent_statistics:
                self.agent_statistics[agent_id] = {
                    "total_evaluations": 0,
                    "total_fitness": 0.0,
                    "best_fitness": 0.0,
                    "evaluation_history": []
                }

            stats = self.agent_statistics[agent_id]
            stats["total_evaluations"] += 1
            stats["total_fitness"] += agent_result["fitness"]
            stats["best_fitness"] = max(
                stats["best_fitness"],
                agent_result["fitness"]
            )
            stats["evaluation_history"].append({
                "policy_id": policy_id,
                "fitness": agent_result["fitness"],
                "success_rate": agent_result["success_rate"]
            })

    def track_generation(
        self,
        generation: int,
        population: List[MDAPRolloutPolicyGenome],
        evaluations: List[MDAPPolicyEvaluation]
    ):
        """
        Track generation-level statistics.

        Args:
            generation: Generation number
            population: Policy population
            evaluations: Evaluation results
        """
        fitness_values = [e.consensus_fitness for e in evaluations]
        agreement_values = [e.agreement_level for e in evaluations]

        gen_stats = {
            "generation": generation,
            "population_size": len(population),
            "best_fitness": max(fitness_values) if fitness_values else 0.0,
            "avg_fitness": sum(fitness_values) / len(fitness_values) if fitness_values else 0.0,
            "worst_fitness": min(fitness_values) if fitness_values else 0.0,
            "fitness_std": (
                (sum((f - sum(fitness_values)/len(fitness_values))**2 for f in fitness_values) / len(fitness_values))**0.5
                if fitness_values else 0.0
            ),
            "avg_agreement": sum(agreement_values) / len(agreement_values) if agreement_values else 0.0,
            "total_red_flags": sum(e.red_flags for e in evaluations),
            "timestamp": datetime.utcnow().isoformat()
        }

        self.generation_stats.append(gen_stats)

    def get_policy_trajectory(self, policy_id: str) -> List[Dict]:
        """
        Get performance trajectory for policy.

        Args:
            policy_id: Policy identifier

        Returns:
            List of evaluation records
        """
        return self.history.get(policy_id, [])

    def get_agent_statistics(self, agent_id: str) -> Optional[Dict]:
        """
        Get statistics for specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent statistics dictionary
        """
        return self.agent_statistics.get(agent_id)

    def get_generation_statistics(self) -> List[Dict]:
        """Get all generation statistics."""
        return self.generation_stats

    def get_consensus_trajectory(self) -> List[float]:
        """Get consensus fitness over time."""
        return self.consensus_history

    def generate_report(self) -> str:
        """Generate performance report."""
        lines = [
            "=" * 80,
            "MDAP Policy Performance Report",
            "=" * 80,
            "",
            f"Total policies tracked: {len(self.history)}",
            f"Total agents: {len(self.agent_statistics)}",
            f"Total generations: {len(self.generation_stats)}",
            ""
        ]

        if self.generation_stats:
            latest = self.generation_stats[-1]
            lines.extend([
                "Latest Generation:",
                f"  Generation: {latest['generation']}",
                f"  Best fitness: {latest['best_fitness']:.4f}",
                f"  Average fitness: {latest['avg_fitness']:.4f}",
                f"  Average agreement: {latest['avg_agreement']:.2f}",
                f"  Red flags: {latest['total_red_flags']}",
                ""
            ])

        if self.agent_statistics:
            lines.append("Agent Performance:")
            for agent_id, stats in self.agent_statistics.items():
                avg_fitness = stats["total_fitness"] / stats["total_evaluations"]
                lines.append(
                    f"  {agent_id}: avg={avg_fitness:.4f}, "
                    f"best={stats['best_fitness']:.4f}, "
                    f"evals={stats['total_evaluations']}"
                )
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

async def evolve_mdap_policy(
    test_theorems: List[str],
    generations: int = 20,
    population_size: int = 30,
    num_agents: int = 5,
    voting_strategy: str = "first_k_ahead",
    enable_decomposition: bool = True
) -> MDAPRolloutPolicyGenome:
    """
    Convenience function to evolve MDAP-enhanced policy.

    Args:
        test_theorems: Theorems to train on
        generations: Number of generations
        population_size: Policy population size
        num_agents: Number of MDAP agents
        voting_strategy: MAKER voting strategy
        enable_decomposition: Enable task decomposition

    Returns:
        Best evolved policy
    """
    mdap_config = MDAPPolicyConfig(
        num_agents=num_agents,
        voting_strategy=voting_strategy,
        enable_decomposition=enable_decomposition
    )

    config = RolloutPolicyConfig(
        population_size=population_size,
        test_theorems=test_theorems
    )

    mcts_config = MCTSConfig(max_iterations=100)

    evaluator = MDAPPolicyEvaluator(
        mcts_config=mcts_config,
        test_theorems=test_theorems,
        num_agents=num_agents,
        voting_strategy=voting_strategy
    )

    engine = MDAPPolicyEvolutionEngine(
        config=config,
        evaluator=evaluator,
        mdap_config=mdap_config
    )

    best_policy = await engine.evolve_policies_mdap(
        initial_population=population_size,
        generations=generations,
        test_theorems=test_theorems,
        mcts_config=mcts_config,
        num_agents=num_agents,
        voting_strategy=voting_strategy
    )

    return best_policy


async def search_with_mdap_policy(
    theorem: str,
    policy: MDAPRolloutPolicyGenome,
    max_iterations: int = 1000,
    time_budget: float = 60.0,
    use_decomposition: bool = True
) -> MCTSResult:
    """
    Convenience function to search with MDAP-enhanced policy.

    Args:
        theorem: Theorem to prove
        policy: MDAP-enhanced policy
        max_iterations: Maximum MCTS iterations
        time_budget: Time budget in seconds
        use_decomposition: Use task decomposition

    Returns:
        MCTS result
    """
    if use_decomposition and policy.enable_decomposition:
        # Use decomposition-enhanced policy
        decomp_policy = DecompositionEnhancedPolicy(
            base_policy=policy,
            max_depth=policy.decomposition_depth
        )

        mcts_config = MCTSConfig(
            max_iterations=max_iterations,
            time_budget=time_budget
        )

        result = await decomp_policy.execute_with_decomposition(
            theorem=theorem,
            mcts_config=mcts_config
        )

        return result
    else:
        # Direct MDAP search
        mcts = MDAPEvolvedPolicyMCTS(
            policy=policy,
            num_agents=5,
            voting_strategy=policy.voting_strategy,
            config=MCTSConfig(
                max_iterations=max_iterations,
                time_budget=time_budget
            ),
            theorem=theorem
        )

        result = await mcts.search_mdap()

        return result


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    'MDAPPolicyConfig',
    'MDAPPolicyEvaluation',
    'VotingDetails',
    'AgentSpecialization',

    # Genome
    'MDAPRolloutPolicyGenome',

    # Evaluation
    'MDAPPolicyEvaluator',

    # Voting
    'PolicyVotingEngine',

    # Evolution
    'MDAPPolicyEvolutionEngine',

    # MCTS
    'MDAPEvolvedPolicyMCTS',

    # Decomposition
    'DecompositionEnhancedPolicy',

    # LeanAide Integration
    'LeanAideMDAPPolicyEvolution',

    # Red-flagging
    'PolicyRedFlagger',

    # Tracking
    'MDAPPolicyPerformanceTracker',

    # Convenience functions
    'evolve_mdap_policy',
    'search_with_mdap_policy',
]


# =============================================================================
# Example Usage
# =============================================================================

async def main():
    """Example usage of MDAP-enhanced evolved policies."""

    print("=" * 80)
    print("MDAP/MAKER Evolved Policies Example")
    print("=" * 80)

    # Define test theorems
    test_theorems = [
        "forall (a b : Nat), a + b = b + a",
        "forall (a b c : Nat), (a + b) + c = a + (b + c)",
        "forall (n : Nat), n + 0 = n",
    ]

    print(f"\n1. Evolving MDAP-enhanced policies...")
    print(f"   Test theorems: {len(test_theorems)}")
    print(f"   Generations: 10")
    print(f"   Agents: 5")

    # Evolve MDAP policy
    best_policy = await evolve_mdap_policy(
        test_theorems=test_theorems,
        generations=10,
        population_size=20,
        num_agents=5,
        voting_strategy="first_k_ahead"
    )

    print(f"\n   Best fitness: {best_policy.compute_fitness():.4f}")
    print(f"   Consensus history: {best_policy.consensus_history[-5:]}")

    # Test on new theorem
    test_theorem = "forall (a b c : Nat), a + (b + c) = (a + b) + c"

    print(f"\n2. Searching with MDAP policy...")
    print(f"   Theorem: {test_theorem}")

    result = await search_with_mdap_policy(
        theorem=test_theorem,
        policy=best_policy,
        max_iterations=200,
        use_decomposition=True
    )

    print(f"\n   Success: {result.success}")
    print(f"   Win rate: {result.win_rate:.4f}")
    print(f"   Time: {result.time_elapsed:.2f}s")

    # Track performance
    print(f"\n3. Performance tracking...")

    tracker = MDAPPolicyPerformanceTracker()
    tracker.track_generation(
        generation=10,
        population=[best_policy],
        evaluations=[MDAPPolicyEvaluation(
            policy_id=best_policy.genome_id,
            consensus_fitness=best_policy.compute_fitness(),
            agent_results=[],
            voting_details={},
            agreement_level=0.8,
            confidence=0.9,
            generation=10
        )]
    )

    report = tracker.generate_report()
    print(report)

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
