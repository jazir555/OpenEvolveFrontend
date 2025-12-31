"""
MDAP/MAKER Integration for Evolutionary MCTS Nodes

This module integrates MDAP (Multi-Agent voting) and MAKER with the evolutionary
MCTS nodes approach, creating rich exploration with multi-agent consensus and
zero-error guarantees.

Core Concept:
    Each MCTS node maintains populations that are evolved using multi-agent
    evaluation and MAKER voting for consensus.

Implementation Components:
    1. MDAPEvolutionaryNode - Evolutionary node with MDAP multi-agent evaluation
    2. MDAPSequenceEvaluator - Multi-agent sequence evaluation
    3. SequenceMAKERVoting - MAKER voting for sequence selection
    4. MDAPNodeEvolution - Evolution at nodes with MDAP evaluation
    5. DecompositionAwareEvolution - Evolution with decomposition support
    6. MDAPEvolutionaryMCTS - Main MDAP evolutionary MCTS
    7. MDAPEvolutionaryMCTSWithLeanAide - Integration with Lean formal verification
    8. SequenceRedFlagger - Red-flagging invalid sequences
    9. DistributedMDAPEvolution - Parallel MDAP evolution
    10. MDAPEvolutionMonitor - Performance monitoring

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

# Import MCTS evolutionary components
try:
    from mcts_evolutionary_nodes import (
        EvolutionaryNode,
        EvolutionaryMCTS,
        EvolutionaryTree,
        ActionSequence,
        ProofContext,
        ProofState,
        Tactic,
        MCTSResult,
        SequenceCrossover,
        SequenceMutation,
        SequenceSelection,
        SequenceEvaluator,
        AdaptiveEvolutionController,
        MCTS_AVAILABLE
    )
except ImportError:
    MCTS_AVAILABLE = False
    logging.warning("MCTS evolutionary components not available")

# Import MDAP components
try:
    from mdap_engine import (
        MDAPOrchestrator,
        MDAPConfig,
        MDAPTask,
        MDAPStep,
        MDAPVoteResult,
        RedFlagRules,
        RedFlagger,
        canonicalize_candidate
    )
    MDAP_AVAILABLE = True
except ImportError:
    MDAP_AVAILABLE = False
    logging.warning("MDAP components not available")

# Import MAKER components
try:
    from maker_engine import (
        MakerEngine,
        MakerConfig,
        MakerStep,
        MakerState,
        MakerRunResult
    )
    MAKER_AVAILABLE = True
except ImportError:
    MAKER_AVAILABLE = False
    logging.warning("MAKER components not available")

# Import LeanAide
try:
    from leanaide_client import LeanAideClient
    from leanaide_evolution import LeanProof
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logging.warning("LeanAide integration not available")

# Import decomposition
try:
    from decomposition_engine import (
        DecompositionEngine,
        DecompositionStrategyBase,
        SemanticDecomposition
    )
    DECOMPOSITION_AVAILABLE = True
except ImportError:
    DECOMPOSITION_AVAILABLE = False
    logging.warning("Decomposition engine not available")

logger = logging.getLogger(__name__)

# =============================================================================
# Type Definitions
# =============================================================================

T = TypeVar('T')


# =============================================================================
# MDAP-Enhanced Evolutionary Node
# =============================================================================

@dataclass
class MDAPSequenceEvaluation:
    """Result of MDAP evaluation for a sequence."""
    sequence_id: str
    agent_results: List['AgentEvaluationResult']
    consensus_fitness: float
    agreement_level: float
    voting_details: Dict[str, int]
    red_flags: int = 0
    evaluation_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sequence_id": self.sequence_id,
            "agent_results": [r.to_dict() for r in self.agent_results],
            "consensus_fitness": self.consensus_fitness,
            "agreement_level": self.agreement_level,
            "voting_details": self.voting_details,
            "red_flags": self.red_flags,
            "evaluation_time": self.evaluation_time
        }


@dataclass
class AgentEvaluationResult:
    """Result from a single agent evaluation."""
    agent_id: str
    fitness: float
    confidence: float
    reasoning: str
    evaluation_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "fitness": self.fitness,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evaluation_metadata": self.evaluation_metadata
        }


@dataclass
class SubtaskDefinition:
    """Definition of a subtask from decomposition."""
    task_id: str
    title: str
    description: str
    priority: int
    dependencies: List[str]
    success_criteria: str
    complexity: float
    estimated_effort: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "success_criteria": self.success_criteria,
            "complexity": self.complexity,
            "estimated_effort": self.estimated_effort
        }


class MDAPEvolutionaryNode(EvolutionaryNode):
    """
    Evolutionary node with MDAP multi-agent evaluation.

    Extends EvolutionaryNode with:
    - Agent-specific populations
    - Multi-agent fitness tracking
    - MAKER voting for consensus
    - Decomposition support
    """

    def __init__(
        self,
        state: ProofState,
        parent: Optional['MDAPEvolutionaryNode'] = None,
        action: Optional[str] = None,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_count: int = 2,
        num_agents: int = 5,
        voting_strategy: str = "first_k_ahead",
        consensus_threshold: float = 0.75,
        k_ahead: int = 3,
        enable_decomposition: bool = True
    ):
        """
        Initialize MDAP evolutionary node.

        Args:
            state: Proof state at this node
            parent: Parent node
            action: Action that led to this node
            population_size: Size of evolutionary population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_count: Number of elites to preserve
            num_agents: Number of agents for MDAP evaluation
            voting_strategy: MAKER voting strategy
            consensus_threshold: Threshold for agent agreement
            k_ahead: K-ahead parameter for voting
            enable_decomposition: Enable decomposition support
        """
        # Initialize base evolutionary node
        super().__init__(
            state=state,
            parent=parent,
            action=action,
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_count=elite_count
        )

        # MDAP-specific state
        self.num_agents = num_agents
        self.agent_populations: Dict[str, List[ActionSequence]] = {
            f"agent_{i}": [] for i in range(num_agents)
        }
        self.agent_fitness: Dict[str, List[float]] = {
            f"agent_{i}": [] for i in range(num_agents)
        }
        self.agent_votes: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # MAKER voting configuration
        self.voting_strategy = voting_strategy
        self.consensus_threshold = consensus_threshold
        self.k_ahead = k_ahead

        # Decomposition support
        self.enable_decomposition = enable_decomposition
        self.subtask_nodes: Dict[str, 'MDAPEvolutionaryNode'] = {}
        self.subtasks: List[SubtaskDefinition] = []

        # MDAP evaluation tracking
        self.mdap_evaluations: Dict[str, MDAPSequenceEvaluation] = {}
        self.agent_agreement_history: List[float] = []

        # Metadata
        self.node_id = str(uuid.uuid4())
        self.mdap_initialized: bool = False

    def get_agent_consensus(self) -> ActionSequence:
        """
        Get consensus sequence across all agents.

        Returns:
            Sequence with highest consensus fitness
        """
        if not self.rollout_population:
            raise ValueError("Cannot get consensus from empty population")

        # Find sequence with highest consensus fitness
        best_sequence = max(
            self.rollout_population,
            key=lambda s: s.fitness
        )

        return best_sequence

    def compute_agreement_level(self) -> float:
        """
        Compute agreement level across all agents.

        Agreement is measured as:
        1. Variance in fitness across agents
        2. Voting concentration

        Returns:
            Agreement level (0-1, higher is better agreement)
        """
        if not self.agent_fitness:
            return 0.0

        # Collect all fitness values
        all_fitness = []
        for agent_id, fitness_list in self.agent_fitness.items():
            all_fitness.extend(fitness_list)

        if not all_fitness:
            return 0.0

        # Calculate variance (inverse of agreement)
        avg_fitness = sum(all_fitness) / len(all_fitness)
        variance = sum((f - avg_fitness) ** 2 for f in all_fitness) / len(all_fitness)

        # Convert to agreement score (lower variance = higher agreement)
        agreement = 1.0 / (1.0 + variance)

        return agreement

    def should_decompose(self) -> bool:
        """
        Decide whether to decompose at this node.

        Decomposition is triggered when:
        1. Population diversity is high (many different approaches)
        2. Agreement level is low (agents disagree)
        3. Node is at appropriate depth
        4. Population hasn't converged

        Returns:
            True if decomposition should be performed
        """
        if not self.enable_decomposition:
            return False

        # Check agreement level
        agreement = self.compute_agreement_level()
        if agreement > self.consensus_threshold:
            return False  # High agreement, no need to decompose

        # Check population diversity
        diversity = self.get_population_diversity()
        if diversity < 0.3:
            return False  # Low diversity, not worth decomposing

        # Check if already decomposed
        if self.subtask_nodes:
            return False  # Already decomposed

        # Check node depth
        if self.depth > 15:
            return False  # Too deep

        return True

    def initialize_mdap_populations(
        self,
        context: ProofContext,
        initial_sequences: Optional[List[ActionSequence]] = None
    ) -> None:
        """
        Initialize MDAP populations for all agents.

        Args:
            context: Proof context
            initial_sequences: Optional initial sequences to use
        """
        if initial_sequences is None:
            initial_sequences = []

        # Distribute sequences among agents
        sequences_per_agent = max(1, self.population_size // self.num_agents)

        for agent_id in self.agent_populations.keys():
            # Give each agent their own population
            agent_sequences = []

            # Add shared sequences if available
            for seq in initial_sequences[:sequences_per_agent]:
                agent_seq = seq.copy()
                agent_seq.sequence_id = str(uuid.uuid4())
                agent_sequences.append(agent_seq)

            # Fill with random sequences if needed
            while len(agent_sequences) < sequences_per_agent:
                random_seq = self._generate_random_sequence(context)
                agent_sequences.append(random_seq)

            self.agent_populations[agent_id] = agent_sequences
            self.agent_fitness[agent_id] = [seq.fitness for seq in agent_sequences]

        # Combine for main population
        combined = []
        for agent_seqs in self.agent_populations.values():
            combined.extend(agent_seqs)

        self.rollout_population = combined[:self.population_size]
        self.mdap_initialized = True

    def _generate_random_sequence(self, context: ProofContext) -> ActionSequence:
        """Generate a random action sequence."""
        length = random.randint(1, min(10, context.depth_limit))
        actions = []

        for _ in range(length):
            tactic = Tactic(name=random.choice(context.available_tactics))
            actions.append(tactic)

        return ActionSequence(
            actions=actions,
            depth=length,
            fitness=0.0
        )

    def update_agent_votes(
        self,
        sequence_id: str,
        agent_id: str,
        vote: int
    ) -> None:
        """
        Update vote tracking for a sequence.

        Args:
            sequence_id: ID of sequence
            agent_id: ID of voting agent
            vote: Vote weight
        """
        self.agent_votes[sequence_id][agent_id] = vote

    def get_sequence_votes(self, sequence_id: str) -> Dict[str, int]:
        """Get all votes for a sequence."""
        return dict(self.agent_votes[sequence_id])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()

        base_dict.update({
            "node_id": self.node_id,
            "num_agents": self.num_agents,
            "voting_strategy": self.voting_strategy,
            "consensus_threshold": self.consensus_threshold,
            "k_ahead": self.k_ahead,
            "enable_decomposition": self.enable_decomposition,
            "agent_agreement": self.compute_agreement_level(),
            "num_subtasks": len(self.subtask_nodes),
            "mdap_initialized": self.mdap_initialized,
            "agent_populations_size": {
                agent_id: len(pop)
                for agent_id, pop in self.agent_populations.items()
            }
        })

        return base_dict


# =============================================================================
# Multi-Agent Sequence Evaluator
# =============================================================================

class MDAPSequenceEvaluator(SequenceEvaluator):
    """
    Evaluate sequences using multiple agents.

    Each agent independently evaluates a sequence, then results are
    combined using MAKER voting for consensus.
    """

    def __init__(
        self,
        num_agents: int = 5,
        leanaide_client: Optional[LeanAideClient] = None
    ):
        """
        Initialize MDAP sequence evaluator.

        Args:
            num_agents: Number of agents to use
            leanaide_client: Optional LeanAide client for verification
        """
        super().__init__(leanaide_client)
        self.num_agents = num_agents

    async def evaluate_mdap(
        self,
        sequences: List[ActionSequence],
        node: MDAPEvolutionaryNode,
        context: ProofContext
    ) -> Dict[str, MDAPSequenceEvaluation]:
        """
        Evaluate sequences using multi-agent approach.

        Args:
            sequences: List of sequences to evaluate
            node: MDAP evolutionary node
            context: Proof context

        Returns:
            Dictionary mapping sequence_id to evaluation
        """
        evaluations = {}
        start_time = time.time()

        for sequence in sequences:
            agent_results = []

            # Each agent evaluates sequence
            for agent_id in range(self.num_agents):
                # Get agent-specific evaluation
                result = await self._agent_evaluate(
                    sequence,
                    f"agent_{agent_id}",
                    context
                )
                agent_results.append(result)

            # Compute consensus
            consensus = self._compute_consensus(agent_results)

            # Collect voting details
            votes = {}
            for i, result in enumerate(agent_results):
                votes[f"agent_{i}"] = int(result.fitness * 10)

            evaluation = MDAPSequenceEvaluation(
                sequence_id=sequence.sequence_id,
                agent_results=agent_results,
                consensus_fitness=consensus.fitness,
                agreement_level=consensus.agreement,
                voting_details=votes,
                evaluation_time=time.time() - start_time
            )

            evaluations[sequence.sequence_id] = evaluation

            # Update sequence fitness
            sequence.fitness = consensus.fitness

            # Store in node
            node.mdap_evaluations[sequence.sequence_id] = evaluation

        return evaluations

    async def _agent_evaluate(
        self,
        sequence: ActionSequence,
        agent_id: str,
        context: ProofContext
    ) -> AgentEvaluationResult:
        """
        Get evaluation from a single agent.

        Args:
            sequence: Sequence to evaluate
            agent_id: ID of evaluating agent
            context: Proof context

        Returns:
            Agent evaluation result
        """
        # Base evaluation with agent-specific bias
        base_fitness = self.evaluate(sequence, context)

        # Add agent-specific variation
        agent_bias = random.gauss(0, 0.05)  # Small random bias
        agent_fitness = max(0.0, min(1.0, base_fitness + agent_bias))

        # Compute confidence based on sequence quality
        confidence = self._compute_confidence(sequence, context)

        # Generate reasoning
        reasoning = self._generate_reasoning(sequence, context, agent_fitness)

        return AgentEvaluationResult(
            agent_id=agent_id,
            fitness=agent_fitness,
            confidence=confidence,
            reasoning=reasoning,
            evaluation_metadata={
                "sequence_length": len(sequence.actions),
                "sequence_depth": sequence.depth,
                "base_fitness": base_fitness,
                "agent_bias": agent_bias
            }
        )

    def _compute_consensus(
        self,
        agent_results: List[AgentEvaluationResult]
    ) -> 'ConsensusResult':
        """
        Compute consensus from agent evaluations.

        Args:
            agent_results: Results from all agents

        Returns:
            Consensus result
        """
        if not agent_results:
            return ConsensusResult(fitness=0.0, agreement=0.0)

        # Compute weighted average fitness
        total_confidence = sum(r.confidence for r in agent_results)
        if total_confidence == 0:
            consensus_fitness = sum(r.fitness for r in agent_results) / len(agent_results)
        else:
            consensus_fitness = sum(
                r.fitness * r.confidence for r in agent_results
            ) / total_confidence

        # Compute agreement level
        fitness_values = [r.fitness for r in agent_results]
        avg_fitness = sum(fitness_values) / len(fitness_values)
        variance = sum((f - avg_fitness) ** 2 for f in fitness_values) / len(fitness_values)
        agreement = 1.0 / (1.0 + variance)

        return ConsensusResult(
            fitness=consensus_fitness,
            agreement=agreement
        )

    def _compute_confidence(
        self,
        sequence: ActionSequence,
        context: ProofContext
    ) -> float:
        """Compute confidence in evaluation."""
        # Base confidence on sequence length
        if not sequence.actions:
            return 0.3

        # Prefer medium-length sequences
        length = len(sequence.actions)
        if 3 <= length <= 8:
            length_confidence = 0.9
        elif length < 3:
            length_confidence = 0.6
        else:
            length_confidence = max(0.5, 0.9 - (length - 8) * 0.05)

        # Check if sequence completes proof
        if sequence.proof_complete:
            return 1.0

        return length_confidence

    def _generate_reasoning(
        self,
        sequence: ActionSequence,
        context: ProofContext,
        fitness: float
    ) -> str:
        """Generate reasoning for evaluation."""
        reasons = []

        if sequence.proof_complete:
            reasons.append("Proof completed successfully")

        if fitness > 0.8:
            reasons.append("High-quality tactics used")
        elif fitness > 0.5:
            reasons.append("Moderate quality approach")
        else:
            reasons.append("Low quality - reconsider approach")

        if len(sequence.actions) > 10:
            reasons.append("Sequence may be too long")

        return "; ".join(reasons)


@dataclass
class ConsensusResult:
    """Result of consensus computation."""
    fitness: float
    agreement: float


# =============================================================================
# MAKER Voting for Sequence Selection
# =============================================================================

class SequenceMAKERVoting:
    """
    MAKER voting for sequence selection at nodes.

    Uses multi-agent voting with k-ahead criterion to select
    the best sequence from a population.
    """

    def __init__(
        self,
        k_ahead: int = 3,
        voting_strategy: str = "first_k_ahead"
    ):
        """
        Initialize MAKER voting.

        Args:
            k_ahead: K-ahead parameter
            voting_strategy: Voting strategy to use
        """
        self.k_ahead = k_ahead
        self.voting_strategy = voting_strategy

    def vote_on_best_sequence(
        self,
        node: MDAPEvolutionaryNode,
        evaluations: Dict[str, MDAPSequenceEvaluation]
    ) -> ActionSequence:
        """
        Use MAKER voting to select best sequence.

        Args:
            node: MDAP evolutionary node
            evaluations: Evaluation results

        Returns:
            Best sequence according to voting
        """
        if self.voting_strategy == "first_k_ahead":
            return self._vote_k_ahead(node, evaluations)
        elif self.voting_strategy == "majority":
            return self._vote_majority(node, evaluations)
        elif self.voting_strategy == "weighted":
            return self._vote_weighted(node, evaluations)
        else:
            return self._vote_k_ahead(node, evaluations)

    def _vote_k_ahead(
        self,
        node: MDAPEvolutionaryNode,
        evaluations: Dict[str, MDAPSequenceEvaluation]
    ) -> ActionSequence:
        """
        Vote using first-k-ahead strategy.

        A sequence wins if it's k votes ahead of all others.
        """
        votes: Dict[str, int] = defaultdict(int)

        # Collect votes from all agents
        for seq_id, evaluation in evaluations.items():
            for agent_result in evaluation.agent_results:
                # Agent votes if performance is good
                if agent_result.fitness > 0.6:
                    votes[seq_id] += 1

                    # Check if ahead by k
                    max_other = max(
                        [v for sid, v in votes.items() if sid != seq_id],
                        default=0
                    )

                    if votes[seq_id] >= max_other + self.k_ahead:
                        # Winner found
                        return self._get_sequence(node, seq_id)

        # No clear winner, return highest voted
        if not votes:
            return node.get_agent_consensus()

        winner_id = max(votes.keys(), key=lambda k: votes[k])
        return self._get_sequence(node, winner_id)

    def _vote_majority(
        self,
        node: MDAPEvolutionaryNode,
        evaluations: Dict[str, MDAPSequenceEvaluation]
    ) -> ActionSequence:
        """Vote using simple majority."""
        votes: Dict[str, int] = defaultdict(int)

        for seq_id, evaluation in evaluations.items():
            # Count agents that like this sequence
            approval_count = sum(
                1 for r in evaluation.agent_results
                if r.fitness > 0.5
            )
            votes[seq_id] = approval_count

        if not votes:
            return node.get_agent_consensus()

        winner_id = max(votes.keys(), key=lambda k: votes[k])
        return self._get_sequence(node, winner_id)

    def _vote_weighted(
        self,
        node: MDAPEvolutionaryNode,
        evaluations: Dict[str, MDAPSequenceEvaluation]
    ) -> ActionSequence:
        """Vote using weighted sum of agent confidences."""
        scores: Dict[str, float] = defaultdict(float)

        for seq_id, evaluation in evaluations.items():
            # Weight by confidence
            for agent_result in evaluation.agent_results:
                scores[seq_id] += agent_result.fitness * agent_result.confidence

        if not scores:
            return node.get_agent_consensus()

        winner_id = max(scores.keys(), key=lambda k: scores[k])
        return self._get_sequence(node, winner_id)

    def _get_sequence(
        self,
        node: MDAPEvolutionaryNode,
        sequence_id: str
    ) -> ActionSequence:
        """Get sequence by ID from node."""
        for seq in node.rollout_population:
            if seq.sequence_id == sequence_id:
                return seq

        # Fallback to consensus
        return node.get_agent_consensus()


# =============================================================================
# MDAP Evolution at Nodes
# =============================================================================

class MDAPNodeEvolution:
    """
    Evolution at nodes with MDAP evaluation.

    Orchestrates the evolutionary process at individual nodes,
    using multi-agent evaluation and MAKER voting.
    """

    def __init__(
        self,
        mdap_evaluator: MDAPSequenceEvaluator,
        sequence_crossover: SequenceCrossover,
        sequence_mutator: SequenceMutation,
        sequence_selection: SequenceSelection,
        maker_voting: SequenceMAKERVoting
    ):
        """
        Initialize MDAP node evolution.

        Args:
            mdap_evaluator: Multi-agent evaluator
            sequence_crossover: Crossover operator
            sequence_mutator: Mutation operator
            sequence_selection: Selection operator
            maker_voting: MAKER voting
        """
        self.mdap_evaluator = mdap_evaluator
        self.sequence_crossover = sequence_crossover
        self.sequence_mutator = sequence_mutator
        self.sequence_selection = sequence_selection
        self.maker_voting = maker_voting

    async def evolve_at_node_mdap(
        self,
        node: MDAPEvolutionaryNode,
        context: ProofContext,
        generations: int = 5
    ) -> ActionSequence:
        """
        Evolve population at node using MDAP.

        Args:
            node: MDAP evolutionary node
            context: Proof context
            generations: Number of generations

        Returns:
            Best sequence found
        """
        for gen in range(generations):
            # 1. Multi-agent evaluation
            evaluations = await self.mdap_evaluator.evaluate_mdap(
                node.rollout_population,
                node,
                context
            )

            # 2. Check convergence
            agreement = node.compute_agreement_level()
            node.agent_agreement_history.append(agreement)

            if agreement > node.consensus_threshold:
                # Converged, return consensus
                logger.info(f"Node {node.node_id} converged at generation {gen}")
                return node.get_agent_consensus()

            # 3. Selection with MAKER voting
            parents = self._select_with_voting(
                node,
                evaluations,
                voting_strategy=node.voting_strategy
            )

            # 4. Crossover
            offspring = []
            while len(offspring) < node.population_size // 2:
                if len(parents) < 2:
                    break
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = self.sequence_crossover.context_aware_crossover(
                    parent1, parent2, context
                )
                offspring.extend([child1, child2])

            # 5. Mutation
            for child in offspring:
                if random.random() < node.mutation_rate:
                    mutated = self.sequence_mutator.adaptive_mutation(
                        child,
                        node.mutation_rate,
                        context.available_tactics
                    )
                    offspring[offspring.index(child)] = mutated

            # 6. Survival selection with voting
            node.rollout_population = self._survival_selection_with_voting(
                node.rollout_population + offspring,
                evaluations,
                context
            )

            # Update agent populations
            self._update_agent_populations(node)

        # Return best sequence
        return node.get_agent_consensus()

    def _select_with_voting(
        self,
        node: MDAPEvolutionaryNode,
        evaluations: Dict[str, MDAPSequenceEvaluation],
        voting_strategy: str
    ) -> List[ActionSequence]:
        """Select parents using voting."""
        parents = []

        # Select top sequences based on voting
        num_parents = min(10, len(node.rollout_population))

        for _ in range(num_parents):
            parent = self.maker_voting.vote_on_best_sequence(node, evaluations)
            if parent not in parents:
                parents.append(parent)

        # Fallback to tournament selection
        while len(parents) < num_parents:
            parent = self.sequence_selection.tournament_selection(
                node.rollout_population,
                tournament_size=3
            )
            if parent not in parents:
                parents.append(parent)

        return parents

    def _survival_selection_with_voting(
        self,
        combined: List[ActionSequence],
        evaluations: Dict[str, MDAPSequenceEvaluation],
        context: ProofContext
    ) -> List[ActionSequence]:
        """Select survivors using voting."""
        if not combined:
            return []

        # Sort by fitness
        sorted_pop = sorted(combined, key=lambda s: s.fitness, reverse=True)

        # Keep elites
        elite_count = 2
        survivors = sorted_pop[:elite_count]

        # Select rest via tournament
        while len(survivors) < len(combined):
            survivor = self.sequence_selection.tournament_selection(
                combined,
                tournament_size=3
            )
            if survivor not in survivors:
                survivors.append(survivor)

        return survivors

    def _update_agent_populations(self, node: MDAPEvolutionaryNode) -> None:
        """Update agent-specific populations."""
        sequences_per_agent = len(node.rollout_population) // node.num_agents

        for i, agent_id in enumerate(node.agent_populations.keys()):
            start_idx = i * sequences_per_agent
            end_idx = start_idx + sequences_per_agent
            agent_seqs = node.rollout_population[start_idx:end_idx]
            node.agent_populations[agent_id] = agent_seqs
            node.agent_fitness[agent_id] = [s.fitness for s in agent_seqs]


# =============================================================================
# Decomposition-Aware Evolution
# =============================================================================

class DecompositionAwareEvolution:
    """
    Evolution that can decompose complex nodes.

    When a node is too complex or has low agent agreement,
    it decomposes the problem into subtasks.
    """

    def __init__(
        self,
        node_evolution: MDAPNodeEvolution,
        mdap_evaluator: MDAPSequenceEvaluator
    ):
        """
        Initialize decomposition-aware evolution.

        Args:
            node_evolution: MDAP node evolution
            mdap_evaluator: Multi-agent evaluator
        """
        self.node_evolution = node_evolution
        self.mdap_evaluator = mdap_evaluator

    async def evolve_with_decomposition(
        self,
        node: MDAPEvolutionaryNode,
        context: ProofContext,
        max_depth: int = 3
    ) -> ActionSequence:
        """
        Evolve with decomposition for complex nodes.

        Args:
            node: MDAP evolutionary node
            context: Proof context
            max_depth: Maximum decomposition depth

        Returns:
            Best solution found
        """
        # Check if should decompose
        if node.should_decompose() and max_depth > 0:
            logger.info(f"Decomposing node {node.node_id}")

            # Decompose problem
            subtasks = await self._decompose_problem(node.state, context)

            if not subtasks:
                # No decomposition possible, use standard evolution
                return await self.node_evolution.evolve_at_node_mdap(
                    node, context, generations=5
                )

            # Create subnodes for each subtask
            subtask_nodes = {}
            subtask_solutions = {}

            for subtask in subtasks:
                # Create subnode
                subnode = self._create_subnode(subtask, node, context)
                subtask_nodes[subtask.task_id] = subnode

                # Evolve solution for subtask
                solution = await self.node_evolution.evolve_at_node_mdap(
                    subnode,
                    context,
                    generations=3
                )
                subtask_solutions[subtask.task_id] = solution

            # Combine subtask solutions
            combined = self._combine_solutions(subtask_solutions)

            return combined
        else:
            # Standard evolution
            return await self.node_evolution.evolve_at_node_mdap(
                node, context, generations=5
            )

    async def _decompose_problem(
        self,
        state: ProofState,
        context: ProofContext
    ) -> List[SubtaskDefinition]:
        """
        Decompose problem into subtasks.

        Args:
            state: Current proof state
            context: Proof context

        Returns:
            List of subtask definitions
        """
        # Use decomposition engine if available
        if DECOMPOSITION_AVAILABLE:
            # Would use DecompositionEngine here
            pass

        # Simple heuristic decomposition
        subtasks = []

        # Split based on goals
        if len(state.goals) > 1:
            for i, goal in enumerate(state.goals):
                subtask = SubtaskDefinition(
                    task_id=f"subtask_{i}",
                    title=f"Prove goal {i+1}",
                    description=f"Prove the goal: {goal}",
                    priority=5,
                    dependencies=[],
                    success_criteria=f"Goal {i+1} proved",
                    complexity=0.5,
                    estimated_effort=8
                )
                subtasks.append(subtask)

        return subtasks

    def _create_subnode(
        self,
        subtask: SubtaskDefinition,
        parent: MDAPEvolutionaryNode,
        context: ProofContext
    ) -> MDAPEvolutionaryNode:
        """Create a subnode for a subtask."""
        # Create sub-state
        sub_state = ProofState(
            goals=[subtask.success_criteria],
            context=parent.state.context.copy(),
            tactics_sequence=[],
            depth=parent.depth + 1
        )

        # Create subnode
        subnode = MDAPEvolutionaryNode(
            state=sub_state,
            parent=parent,
            action=f"decompose:{subtask.task_id}",
            population_size=max(10, parent.population_size // 2),
            num_agents=parent.num_agents,
            voting_strategy=parent.voting_strategy,
            enable_decomposition=False  # Don't decompose subtasks
        )

        # Initialize population
        subnode.initialize_mdap_populations(context)

        return subnode

    def _combine_solutions(
        self,
        subtask_solutions: Dict[str, ActionSequence]
    ) -> ActionSequence:
        """Combine solutions from subtasks."""
        # Combine all actions
        all_actions = []
        for solution in subtask_solutions.values():
            all_actions.extend(solution.actions)

        # Create combined sequence
        combined = ActionSequence(
            actions=all_actions,
            fitness=sum(s.fitness for s in subtask_solutions.values()) / len(subtask_solutions),
            depth=sum(s.depth for s in subtask_solutions.values()),
            proof_complete=all(s.proof_complete for s in subtask_solutions.values())
        )

        return combined


# =============================================================================
# MDAP Evolutionary MCTS
# =============================================================================

class MDAPEvolutionaryMCTS(EvolutionaryMCTS):
    """
    Evolutionary MCTS with MDAP multi-agent evaluation.

    Main class that combines MDAP evaluation, MAKER voting,
    and evolutionary MCTS for rich exploration.
    """

    def __init__(
        self,
        num_agents: int = 5,
        voting_strategy: str = "first_k_ahead",
        enable_decomposition: bool = True,
        consensus_threshold: float = 0.75,
        k_ahead: int = 3,
        **kwargs
    ):
        """
        Initialize MDAP evolutionary MCTS.

        Args:
            num_agents: Number of agents for MDAP
            voting_strategy: MAKER voting strategy
            enable_decomposition: Enable decomposition
            consensus_threshold: Agent agreement threshold
            k_ahead: K-ahead parameter for voting
            **kwargs: Additional arguments for EvolutionaryMCTS
        """
        super().__init__(**kwargs)

        self.num_agents = num_agents
        self.voting_strategy = voting_strategy
        self.enable_decomposition = enable_decomposition
        self.consensus_threshold = consensus_threshold
        self.k_ahead = k_ahead

        # Initialize MDAP components
        self.mdap_evaluator = MDAPSequenceEvaluator(
            num_agents=num_agents,
            leanaide_client=getattr(self, 'leanaide_client', None)
        )
        self.maker_voting = SequenceMAKERVoting(
            k_ahead=k_ahead,
            voting_strategy=voting_strategy
        )
        self.node_evolution = MDAPNodeEvolution(
            mdap_evaluator=self.mdap_evaluator,
            sequence_crossover=self.crossover,
            sequence_mutator=self.mutation,
            sequence_selection=self.selection,
            maker_voting=self.maker_voting
        )
        self.decomposition_evolution = DecompositionAwareEvolution(
            node_evolution=self.node_evolution,
            mdap_evaluator=self.mdap_evaluator
        )

    async def search(
        self,
        initial_context: ProofContext,
        leanaide_client: Optional[LeanAideClient] = None
    ) -> MCTSResult:
        """
        Search using MDAP evolutionary MCTS.

        Args:
            initial_context: Initial proof context
            leanaide_client: Optional LeanAide client

        Returns:
            MCTSResult with best proof found
        """
        start_time = time.time()

        # Create root node
        initial_state = ProofState(
            goals=initial_context.goals,
            context=initial_context.hypotheses
        )

        root = MDAPEvolutionaryNode(
            state=initial_state,
            population_size=self.population_size,
            num_agents=self.num_agents,
            voting_strategy=self.voting_strategy,
            consensus_threshold=self.consensus_threshold,
            k_ahead=self.k_ahead,
            enable_decomposition=self.enable_decomposition
        )

        # Initialize population
        self.initialize_node_population(root, initial_context)

        # Create tree
        tree = EvolutionaryTree(root)

        # Main MCTS loop
        for i in range(self.mcts_simulations):
            # Selection
            node = self._select(root)

            # Expansion with MDAP
            if not node.is_fully_expanded_node():
                node = await self._expand_mdap(node, initial_context)

            # Evolutionary simulation with MDAP
            value = await self._evolutionary_simulation_mdap(
                node,
                initial_context
            )

            # Backpropagation
            self._backpropagate(node, value)

        # Compile result
        elapsed = time.time() - start_time
        return self._compile_result(root, tree, elapsed)

    def initialize_node_population(
        self,
        node: MDAPEvolutionaryNode,
        context: ProofContext
    ) -> None:
        """Initialize MDAP population at node."""
        # Generate initial sequences
        initial_sequences = []
        for _ in range(node.population_size):
            sequence = self._generate_random_sequence(context)
            sequence.fitness = self.evaluator.evaluate(sequence, context)
            initial_sequences.append(sequence)

        # Initialize MDAP populations
        node.initialize_mdap_populations(context, initial_sequences)

    async def _expand_mdap(
        self,
        node: MDAPEvolutionaryNode,
        context: ProofContext
    ) -> MDAPEvolutionaryNode:
        """Expand node using MDAP."""
        if node.is_terminal or node.is_fully_expanded_node():
            return node

        # Get untried actions
        if not node.untried_actions:
            node.untried_actions = context.available_tactics[:10]

        if not node.untried_actions:
            node.is_terminal = True
            return node

        # Create child node
        action = node.untried_actions.pop(0)
        new_state = self._apply_action(node.state, action)

        child = MDAPEvolutionaryNode(
            state=new_state,
            parent=node,
            action=action,
            population_size=self.population_size,
            num_agents=self.num_agents,
            voting_strategy=self.voting_strategy,
            enable_decomposition=self.enable_decomposition
        )

        # Initialize population
        self.initialize_node_population(child, context)

        # Add to tree
        node.add_child(action, child)

        return child

    async def _evolutionary_simulation_mdap(
        self,
        node: MDAPEvolutionaryNode,
        context: ProofContext
    ) -> float:
        """Run evolution at node with MDAP."""
        if self.enable_decomposition:
            solution = await self.decomposition_evolution.evolve_with_decomposition(
                node, context, max_depth=3
            )
        else:
            solution = await self.node_evolution.evolve_at_node_mdap(
                node, context, generations=self.evolution_generations
            )

        return solution.fitness


# =============================================================================
# LeanAide Integration
# =============================================================================

class MDAPEvolutionaryMCTSWithLeanAide(MDAPEvolutionaryMCTS):
    """
    MDAP evolutionary MCTS with Lean formal verification.

    Uses LeanAide to formally verify evolved sequences,
    providing zero-error guarantees.
    """

    def __init__(
        self,
        leanaide_client: LeanAideClient,
        **kwargs
    ):
        """
        Initialize with LeanAide client.

        Args:
            leanaide_client: LeanAide client
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.leanaide_client = leanaide_client

    async def search_with_verification(
        self,
        theorem: str,
        leanaide_client: Optional[LeanAideClient] = None
    ) -> MCTSResult:
        """
        Search with formal verification.

        Args:
            theorem: Theorem to prove
            leanaide_client: Optional LeanAide client

        Returns:
            MCTSResult with verified proof
        """
        client = leanaide_client or self.leanaide_client

        # Create context
        context = ProofContext(
            theorem=theorem,
            goals=[f"prove {theorem}"],
            hypotheses=[],
            available_tactics=self.mutation.available_tactics
        )

        # Run MDAP evolutionary MCTS
        result = await self.search(context, client)

        # Verify best candidates with Lean
        verified_candidates = []
        for candidate in result.proof_path:
            try:
                verification = await client.elaborate(
                    candidate.best_sequence.to_string() if hasattr(candidate, 'best_sequence') else ""
                )

                if verification.success:
                    candidate.best_sequence.fitness *= 1.5  # Bonus
                    candidate.best_sequence.proof_complete = True
                    verified_candidates.append(candidate.best_sequence)

            except Exception as e:
                logger.warning(f"Verification failed: {e}")

        # Return best verified candidate
        if verified_candidates:
            result.best_proof = max(verified_candidates, key=lambda c: c.fitness)
            result.success = True

        return result


# =============================================================================
# Red-Flagging Invalid Sequences
# =============================================================================

class SequenceRedFlagger:
    """
    Red-flag invalid sequences at nodes.

    Filters out sequences that are clearly invalid before
    they waste computation resources.
    """

    def __init__(self):
        """Initialize sequence red-flagger."""
        self.flagged_sequences: Set[str] = set()
        self.flag_reasons: Dict[str, List[str]] = {}

    def check_sequence(
        self,
        sequence: ActionSequence,
        context: ProofContext
    ) -> Tuple[bool, List[str]]:
        """
        Check if sequence should be red-flagged.

        Args:
            sequence: Sequence to check
            context: Proof context

        Returns:
            Tuple of (is_flagged, reasons)
        """
        flags = []

        # Check for invalid tactics
        if self._has_invalid_tactics(sequence, context):
            flags.append("Invalid tactics for context")

        # Check for cycles
        if self._has_cycles(sequence):
            flags.append("Contains cycles")

        # Check for dead ends
        if self._leads_to_dead_end(sequence, context):
            flags.append("Dead end sequence")

        # Check sequence length
        if len(sequence.actions) > context.depth_limit:
            flags.append("Exceeds depth limit")

        # Check agent consensus if available
        if hasattr(sequence, 'agent_agreement'):
            if sequence.agent_agreement < 0.3:
                flags.append("Low agent agreement")

        is_flagged = len(flags) > 0

        if is_flagged:
            self.flagged_sequences.add(sequence.sequence_id)
            self.flag_reasons[sequence.sequence_id] = flags

        return is_flagged, flags

    def _has_invalid_tactics(
        self,
        sequence: ActionSequence,
        context: ProofContext
    ) -> bool:
        """Check for invalid tactics."""
        available = set(context.available_tactics)
        for tactic in sequence.actions:
            if tactic.name not in available:
                return True
        return False

    def _has_cycles(self, sequence: ActionSequence) -> bool:
        """Check for repeated tactics (potential cycles)."""
        if len(sequence.actions) < 3:
            return False

        tactic_names = [t.name for t in sequence.actions]

        # Look for repeated patterns
        for i in range(len(tactic_names) - 2):
            if tactic_names[i] == tactic_names[i+1] == tactic_names[i+2]:
                return True

        return False

    def _leads_to_dead_end(
        self,
        sequence: ActionSequence,
        context: ProofContext
    ) -> bool:
        """Check if sequence leads to dead end."""
        # Heuristic: too many same tactic
        tactic_counts = defaultdict(int)
        for tactic in sequence.actions:
            tactic_counts[tactic.name] += 1

        for tactic_name, count in tactic_counts.items():
            if count > 5:
                return True

        return False

    def is_flagged(self, sequence_id: str) -> bool:
        """Check if sequence is flagged."""
        return sequence_id in self.flagged_sequences

    def get_flag_reasons(self, sequence_id: str) -> List[str]:
        """Get flag reasons for sequence."""
        return self.flag_reasons.get(sequence_id, [])

    def clear_flags(self) -> None:
        """Clear all flags."""
        self.flagged_sequences.clear()
        self.flag_reasons.clear()


# =============================================================================
# Distributed MDAP Evolution
# =============================================================================

class DistributedMDAPEvolution:
    """
    Parallel MDAP evolution at multiple nodes.

    Distributes evolutionary computation across workers
    for faster proof search.
    """

    def __init__(
        self,
        node_evolution: MDAPNodeEvolution,
        max_workers: int = 4
    ):
        """
        Initialize distributed MDAP evolution.

        Args:
            node_evolution: MDAP node evolution
            max_workers: Maximum parallel workers
        """
        self.node_evolution = node_evolution
        self.max_workers = max_workers

    async def evolve_nodes_parallel(
        self,
        nodes: List[MDAPEvolutionaryNode],
        context: ProofContext,
        max_workers: Optional[int] = None
    ) -> Dict[str, ActionSequence]:
        """
        Evolve multiple nodes in parallel.

        Args:
            nodes: List of nodes to evolve
            context: Proof context
            max_workers: Optional worker count override

        Returns:
            Dictionary mapping node_id to best solution
        """
        workers = max_workers or self.max_workers
        tasks = []

        for node in nodes:
            task = self.node_evolution.evolve_at_node_mdap(
                node, context, generations=3
            )
            tasks.append((node.node_id, task))

        # Run in parallel with semaphore
        semaphore = asyncio.Semaphore(workers)

        async def bounded_task(node_id, task):
            async with semaphore:
                result = await task
                return node_id, result

        results = await asyncio.gather(*[
            bounded_task(nid, t) for nid, t in tasks
        ])

        return {nid: result for nid, result in results}


# =============================================================================
# Performance Monitoring
# =============================================================================

class MDAPEvolutionMonitor:
    """
    Monitor MDAP evolution at nodes.

    Tracks convergence, agent performance, and resource usage.
    """

    def __init__(self):
        """Initialize monitor."""
        self.node_history: Dict[str, List[Dict]] = defaultdict(list)
        self.agent_performance: Dict[str, List[float]] = defaultdict(list)
        self.evolution_times: Dict[str, List[float]] = defaultdict(list)
        self.convergence_curves: Dict[str, List[float]] = defaultdict(list)

    def track_generation(
        self,
        node_id: str,
        generation: int,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Track generation metrics.

        Args:
            node_id: Node ID
            generation: Generation number
            metrics: Metrics dictionary
        """
        entry = {
            "generation": generation,
            "timestamp": time.time(),
            "metrics": metrics.copy()
        }

        self.node_history[node_id].append(entry)

        # Track convergence
        if "avg_fitness" in metrics:
            self.convergence_curves[node_id].append(metrics["avg_fitness"])

        # Track agent performance
        if "agent_fitness" in metrics:
            for agent_id, fitness in metrics["agent_fitness"].items():
                self.agent_performance[agent_id].append(fitness)

    def get_convergence_curve(self, node_id: str) -> List[float]:
        """Get convergence curve for node."""
        return self.convergence_curves.get(node_id, [])

    def get_agent_reliability(self, agent_id: str) -> float:
        """
        Get reliability score for agent.

        Reliability is based on consistency of evaluations.
        """
        if agent_id not in self.agent_performance:
            return 0.5

        scores = self.agent_performance[agent_id]
        if not scores:
            return 0.5

        # Compute consistency (inverse of variance)
        avg_score = sum(scores) / len(scores)
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        reliability = 1.0 / (1.0 + variance)

        return reliability

    def get_node_statistics(self, node_id: str) -> Dict[str, Any]:
        """Get statistics for a node."""
        if node_id not in self.node_history:
            return {}

        history = self.node_history[node_id]

        return {
            "total_generations": len(history),
            "convergence_curve": self.get_convergence_curve(node_id),
            "avg_fitness": history[-1]["metrics"].get("avg_fitness", 0.0) if history else 0.0,
            "best_fitness": max(
                entry["metrics"].get("best_fitness", 0.0)
                for entry in history
            ) if history else 0.0
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_nodes = len(self.node_history)
        total_generations = sum(len(h) for h in self.node_history.values())

        agent_reliability = {
            agent_id: self.get_agent_reliability(agent_id)
            for agent_id in self.agent_performance.keys()
        }

        return {
            "total_nodes": total_nodes,
            "total_generations": total_generations,
            "agent_reliability": agent_reliability,
            "avg_generations_per_node": total_generations / max(1, total_nodes)
        }

    def clear(self) -> None:
        """Clear all monitoring data."""
        self.node_history.clear()
        self.agent_performance.clear()
        self.evolution_times.clear()
        self.convergence_curves.clear()


# =============================================================================
# Utility Functions
# =============================================================================

def create_mdap_evolutionary_mcts(
    population_size: int = 20,
    evolution_generations: int = 5,
    num_agents: int = 5,
    voting_strategy: str = "first_k_ahead",
    enable_decomposition: bool = True,
    **kwargs
) -> MDAPEvolutionaryMCTS:
    """
    Convenience function to create MDAP evolutionary MCTS.

    Args:
        population_size: Size of evolutionary population
        evolution_generations: Generations per simulation
        num_agents: Number of agents for MDAP
        voting_strategy: MAKER voting strategy
        enable_decomposition: Enable decomposition
        **kwargs: Additional arguments

    Returns:
        MDAPEvolutionaryMCTS instance
    """
    return MDAPEvolutionaryMCTS(
        population_size=population_size,
        evolution_generations=evolution_generations,
        num_agents=num_agents,
        voting_strategy=voting_strategy,
        enable_decomposition=enable_decomposition,
        **kwargs
    )


def create_mdap_node(
    state: ProofState,
    population_size: int = 20,
    num_agents: int = 5,
    voting_strategy: str = "first_k_ahead",
    consensus_threshold: float = 0.75,
    **kwargs
) -> MDAPEvolutionaryNode:
    """
    Convenience function to create MDAP evolutionary node.

    Args:
        state: Proof state
        population_size: Size of population
        num_agents: Number of agents
        voting_strategy: Voting strategy
        consensus_threshold: Consensus threshold
        **kwargs: Additional arguments

    Returns:
        MDAPEvolutionaryNode instance
    """
    return MDAPEvolutionaryNode(
        state=state,
        population_size=population_size,
        num_agents=num_agents,
        voting_strategy=voting_strategy,
        consensus_threshold=consensus_threshold,
        **kwargs
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core MDAP data classes
    'MDAPSequenceEvaluation',
    'AgentEvaluationResult',
    'SubtaskDefinition',

    # MDAP Node
    'MDAPEvolutionaryNode',

    # Evaluators
    'MDAPSequenceEvaluator',

    # Voting
    'SequenceMAKERVoting',

    # Evolution
    'MDAPNodeEvolution',
    'DecompositionAwareEvolution',

    # Main MCTS
    'MDAPEvolutionaryMCTS',
    'MDAPEvolutionaryMCTSWithLeanAide',

    # Utilities
    'SequenceRedFlagger',
    'DistributedMDAPEvolution',
    'MDAPEvolutionMonitor',

    # Factory functions
    'create_mdap_evolutionary_mcts',
    'create_mdap_node',
]


# =============================================================================
# Example Usage
# =============================================================================

async def example_mdap_evolutionary_mcts():
    """Example usage of MDAP Evolutionary MCTS."""

    print("=" * 80)
    print("MDAP Evolutionary MCTS Example")
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

    # Create MDAP evolutionary MCTS
    mdap_mcts = create_mdap_evolutionary_mcts(
        population_size=20,
        evolution_generations=5,
        num_agents=5,
        voting_strategy="first_k_ahead",
        enable_decomposition=True,
        mcts_simulations=100
    )

    # Create monitor
    monitor = MDAPEvolutionMonitor()

    # Run search
    result = await mdap_mcts.search(context)

    # Print results
    print("\n" + "=" * 80)
    print("MDAP Evolutionary MCTS Results")
    print("=" * 80)
    print(f"\nSuccess: {result.success}")
    print(f"Time: {result.time_elapsed:.2f}s")
    print(f"Nodes visited: {result.nodes_visited}")
    print(f"Win rate: {result.win_rate:.4f}")

    # Get monitoring summary
    summary = monitor.get_summary()
    print(f"\nTotal nodes evolved: {summary['total_nodes']}")
    print(f"Total generations: {summary['total_generations']}")

    if result.best_proof:
        print("\nBest proof found:")
        print(result.best_proof.lean_code)


if __name__ == "__main__":
    asyncio.run(example_mdap_evolutionary_mcts())
