"""
LeanAide MDAP-Enhanced Evolutionary Operators

Production-ready MDAP (Multi-Agent Voting) integration with evolutionary proof generation.
Combines genetic algorithms with multi-agent consensus for superior proof search.

Architecture:
    MDAPLeanPopulation: Population with multi-agent voting
    MDAPLeanSelector: Selection enhanced with agent consensus
    MDAPLeanCrossover: Crossover guided by agent voting
    MDAPLeanMutator: Mutation suggested and voted by agents
    MDAPEvolutionEngine: Main evolutionary orchestration with MDAP

Key Features:
    - Multi-agent voting for fitness evaluation
    - Agent-guided crossover point selection
    - Agent-suggested mutations with voting
    - Red-flagging for invalid individuals
    - Agent performance tracking and weighting
    - First-K-ahead, majority, weighted voting strategies

Author: OpenEvolve
Created: 2025-12-30
"""

import asyncio
import json
import logging
import random
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union
)
import threading

# Import evolution components
try:
    from leanaide_evolution import (
        LeanProofStrategy,
        LeanProof,
        Tactic,
        LeanProofPopulation,
        LeanProofMutator,
        LeanProofCrossover,
        LeanProofEvaluator,
        LeanProofEvolutionEngine,
        EvolutionResult,
        PopulationStatistics,
        MutationType,
        SelectionMethod,
        CrossoverMethod
    )
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False
    logging.warning("Evolution module not available")

# Import MDAP components
try:
    from leanaide_mdap import (
        LeanProofAgent,
        ProofStrategy,
        LeanMDAPConfig,
        MDAP_AVAILABLE
    )
    from maker_engine import RedFlagRules, canonicalize_candidate
    MDAP_ENGINE_AVAILABLE = True
except ImportError:
    MDAP_ENGINE_AVAILABLE = False
    logging.warning("MDAP engine not available")

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class MDAPVotingStrategy(str, Enum):
    """Voting strategies for MDAP consensus"""
    FIRST_K_AHEAD = "first_k_ahead"
    MAJORITY = "majority"
    WEIGHTED_CONFIDENCE = "weighted_confidence"
    WEIGHTED_PERFORMANCE = "weighted_performance"
    CONDORCET = "condorcet"
    BORDA = "borda"


class AgentConsensusLevel(str, Enum):
    """Level of agent consensus"""
    UNANIMOUS = "unanimous"
    STRONG = "strong"  # >= 75% agreement
    MAJORITY = "majority"  # >= 50% agreement
    WEAK = "weak"  # < 50% agreement
    NO_CONSENSUS = "no_consensus"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AgentVote:
    """A single agent's vote on a proof strategy"""
    agent_id: str
    agent_type: str
    strategy_id: str
    fitness_score: float
    confidence: float
    rationale: str
    suggested_mutations: List[str] = field(default_factory=list)
    suggested_crossovers: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConsensusResult:
    """Result of agent consensus calculation"""
    strategy_id: str
    consensus_level: AgentConsensusLevel
    aggregate_fitness: float
    vote_distribution: Dict[str, int]  # agent_type -> count
    confidence: float
    agreement_ratio: float  # 0.0 to 1.0
    votes: List[AgentVote] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["consensus_level"] = self.consensus_level.value
        return data


@dataclass
class MutationSuggestion:
    """Suggested mutation from an agent"""
    agent_id: str
    mutation_type: MutationType
    position: int  # Position in tactic sequence
    old_tactic: Optional[str]
    new_tactic: str
    confidence: float
    rationale: str
    estimated_improvement: float

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["mutation_type"] = self.mutation_type.value
        return data


@dataclass
class CrossoverVote:
    """Agent vote on crossover strategy"""
    agent_id: str
    crossover_method: CrossoverMethod
    crossover_points: List[int]
    confidence: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["crossover_method"] = self.crossover_method.value
        return data


@dataclass
class MDAPEvolutionConfig:
    """Configuration for MDAP-enhanced evolution"""

    # Evolutionary parameters
    population_size: int = 20
    max_generations: int = 50
    mutation_rate: float = 0.2
    crossover_rate: float = 0.8
    elitism_count: int = 2
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM

    # MDAP agents
    selection_agents: List[str] = field(default_factory=lambda: ["evolution", "mcts", "direct"])
    selection_num_agents: int = 3
    selection_voting_strategy: MDAPVotingStrategy = MDAPVotingStrategy.FIRST_K_AHEAD
    selection_k_ahead: int = 3

    crossover_agents: List[str] = field(default_factory=lambda: ["evolution", "mcts"])
    crossover_num_agents: int = 2
    crossover_voting_strategy: MDAPVotingStrategy = MDAPVotingStrategy.MAJORITY

    mutation_agents: List[str] = field(default_factory=lambda: ["evolution", "adversarial"])
    mutation_num_agents: int = 2
    mutation_voting_strategy: MDAPVotingStrategy = MDAPVotingStrategy.WEIGHTED_CONFIDENCE

    # Red-flagging
    enable_red_flagging: bool = True
    prune_invalid: bool = True
    max_proof_length: int = 500
    min_confidence: float = 0.1

    # Agent tracking
    track_agent_performance: bool = True
    update_agent_weights: bool = True
    performance_window: int = 10  # Number of recent results to consider

    # Server
    server_url: str = "http://localhost:7654"
    cache_enabled: bool = True
    parallel_evaluation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["selection_method"] = self.selection_method.value
        data["crossover_method"] = self.crossover_method.value
        data["selection_voting_strategy"] = self.selection_voting_strategy.value
        data["crossover_voting_strategy"] = self.crossover_voting_strategy.value
        data["mutation_voting_strategy"] = self.mutation_voting_strategy.value
        return data


@dataclass
class MDAPResult:
    """Result of MDAP-enhanced evolution"""
    success: bool
    best_proof: Optional[LeanProof]
    best_strategy: Optional[LeanProofStrategy]
    generations_completed: int
    total_evaluations: int
    evolution_time: float
    statistics_history: List[PopulationStatistics]

    # MDAP-specific
    agent_performance: Dict[str, Dict[str, float]]
    consensus_history: List[ConsensusResult]
    voting_efficiency: float
    agent_agreement_rate: float
    red_flag_count: int

    # Lineage
    family_tree: Dict[str, List[str]]
    failed_attempts: List[Dict[str, Any]]
    convergence_history: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "best_proof": self.best_proof.to_dict() if self.best_proof else None,
            "best_strategy": self.best_strategy.to_dict() if self.best_strategy else None,
            "generations_completed": self.generations_completed,
            "total_evaluations": self.total_evaluations,
            "evolution_time": self.evolution_time,
            "statistics_history": [s.to_dict() for s in self.statistics_history],
            "agent_performance": self.agent_performance,
            "consensus_history": [c.to_dict() for c in self.consensus_history],
            "voting_efficiency": self.voting_efficiency,
            "agent_agreement_rate": self.agent_agreement_rate,
            "red_flag_count": self.red_flag_count,
            "family_tree": self.family_tree,
            "failed_attempts": self.failed_attempts,
            "convergence_history": self.convergence_history
        }


# =============================================================================
# MDAP-ENHANCED POPULATION
# =============================================================================

class MDAPLeanPopulation(LeanProofPopulation if EVOLUTION_AVAILABLE else object):
    """
    Population enhanced with MDAP multi-agent voting.

    Each individual can be evaluated by multiple agents, and their votes
    are aggregated using various voting strategies.
    """

    def __init__(
        self,
        strategies: List[LeanProofStrategy],
        agents: List['LeanProofAgent'],
        config: MDAPEvolutionConfig,
        selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
        tournament_size: int = 3,
        elitism_ratio: float = 0.1
    ):
        """
        Initialize MDAP-enhanced population.

        Args:
            strategies: Initial proof strategies
            agents: Available MDAP agents
            config: MDAP evolution configuration
            selection_method: Selection method for parent selection
            tournament_size: Tournament size for tournament selection
            elitism_ratio: Ratio of elites to preserve
        """
        if EVOLUTION_AVAILABLE:
            super().__init__(strategies, selection_method, tournament_size, elitism_ratio)
        else:
            self.strategies = strategies
            self.selection_method = selection_method
            self.tournament_size = tournament_size
            self.elitism_ratio = elitism_ratio
            self.generation = 0

        self.agents = agents
        self.config = config

        # Store agent votes for each strategy
        self.agent_votes: Dict[str, List[AgentVote]] = defaultdict(list)

        # Agent performance tracking
        self.agent_performance: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"success_rate": 0.5, "avg_confidence": 0.5, "total_votes": 0}
        )

    async def evaluate_with_mdap(
        self,
        individuals: List[LeanProofStrategy]
    ) -> Dict[str, ConsensusResult]:
        """
        Evaluate individuals using multiple MDAP agents.

        Args:
            individuals: Strategies to evaluate

        Returns:
            Dictionary mapping strategy IDs to consensus results
        """
        results = {}

        # Select agents for evaluation
        evaluation_agents = self._select_agents_for_task(
            self.config.selection_agents,
            self.config.selection_num_agents
        )

        # Evaluate each individual with all agents
        for individual in individuals:
            votes = []

            for agent in evaluation_agents:
                try:
                    # Get agent's evaluation
                    vote = await self._get_agent_vote(agent, individual)
                    votes.append(vote)
                except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                    logger.error(f"Agent {agent.agent_id} failed: {e}")
                    continue

            # Store votes
            self.agent_votes[individual.strategy_id] = votes

            # Calculate consensus
            consensus = self._calculate_consensus(individual, votes)
            results[individual.strategy_id] = consensus

            # Update individual fitness with consensus
            individual.fitness = consensus.aggregate_fitness

        return results

    def _select_agents_for_task(
        self,
        agent_types: List[str],
        num_agents: int
    ) -> List['LeanProofAgent']:
        """
        Select agents for a specific task.

        Args:
            agent_types: Preferred agent types
            num_agents: Number of agents to select

        Returns:
            List of selected agents
        """
        selected = []

        # Filter agents by type
        typed_agents = [a for a in self.agents if a.agent_type.value in agent_types]

        # If we have enough typed agents, use them
        if len(typed_agents) >= num_agents:
            selected = typed_agents[:num_agents]
        else:
            # Use typed agents + fill with others
            selected = typed_agents
            remaining = [a for a in self.agents if a not in selected]
            selected.extend(remaining[:num_agents - len(selected)])

        return selected

    async def _get_agent_vote(
        self,
        agent: 'LeanProofAgent',
        individual: LeanProofStrategy
    ) -> AgentVote:
        """
        Get an agent's vote on a strategy.

        Args:
            agent: Agent to query
            individual: Strategy to evaluate

        Returns:
            Agent's vote
        """
        # Estimate fitness based on agent's perspective
        fitness = agent.estimate_quality(individual.proof)

        # Generate rationale
        rationale = self._generate_agent_rationale(agent, individual)

        # Suggest potential improvements
        suggested_mutations = self._suggest_mutations(agent, individual)
        suggested_crossovers = self._suggest_crossover_points(agent, individual)

        return AgentVote(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type.value,
            strategy_id=individual.strategy_id,
            fitness_score=fitness,
            confidence=individual.proof.confidence if hasattr(individual.proof, 'confidence') else fitness,
            rationale=rationale,
            suggested_mutations=suggested_mutations,
            suggested_crossovers=suggested_crossovers,
            metadata={
                "agent_success_rate": agent.successful_proofs / max(1, agent.total_proofs_generated)
            }
        )

    def _generate_agent_rationale(
        self,
        agent: 'LeanProofAgent',
        individual: LeanProofStrategy
    ) -> str:
        """Generate rationale for agent's vote"""
        num_tactics = len(individual.proof.tactics)

        rationale_parts = [
            f"Agent {agent.agent_type.value} evaluated strategy with {num_tactics} tactics",
        ]

        if individual.verified:
            rationale_parts.append("Proof is verified")

        if individual.fitness > 5.0:
            rationale_parts.append("High fitness score indicates promising approach")
        elif individual.fitness < 2.0:
            rationale_parts.append("Low fitness score suggests improvements needed")

        return ". ".join(rationale_parts) + "."

    def _suggest_mutations(
        self,
        agent: 'LeanProofAgent',
        individual: LeanProofStrategy
    ) -> List[str]:
        """Suggest potential mutations"""
        suggestions = []

        # Basic suggestions based on strategy type
        if agent.agent_type.value == "evolution":
            suggestions.extend([
                "Consider adding 'simp' tactic",
                "Try 'rw' for rewrites",
                "Explore 'apply' with lemmas"
            ])
        elif agent.agent_type.value == "mcts":
            suggestions.extend([
                "Explore alternative tactic orderings",
                "Consider 'cases' for induction",
                "Try 'constructor' for existence"
            ])
        elif agent.agent_type.value == "adversarial":
            suggestions.extend([
                "Challenge assumptions",
                "Try counter-examples",
                "Test edge cases"
            ])

        return suggestions[:3]  # Limit to 3 suggestions

    def _suggest_crossover_points(
        self,
        agent: 'LeanProofAgent',
        individual: LeanProofStrategy
    ) -> List[int]:
        """Suggest potential crossover points"""
        num_tactics = len(individual.proof.tactics)

        if num_tactics <= 2:
            return [1]

        # Suggest points at 1/3 and 2/3 of the proof
        return [num_tactics // 3, 2 * num_tactics // 3]

    def _calculate_consensus(
        self,
        individual: LeanProofStrategy,
        votes: List[AgentVote]
    ) -> ConsensusResult:
        """
        Calculate consensus from agent votes.

        Args:
            individual: Strategy being evaluated
            votes: Agent votes

        Returns:
            Consensus result
        """
        if not votes:
            return ConsensusResult(
                strategy_id=individual.strategy_id,
                consensus_level=AgentConsensusLevel.NO_CONSENSUS,
                aggregate_fitness=0.0,
                vote_distribution={},
                confidence=0.0,
                agreement_ratio=0.0,
                votes=[]
            )

        # Aggregate votes using configured strategy
        strategy = self.config.selection_voting_strategy

        if strategy == MDAPVotingStrategy.FIRST_K_AHEAD:
            return self._consensus_first_k_ahead(individual, votes)
        elif strategy == MDAPVotingStrategy.MAJORITY:
            return self._consensus_majority(individual, votes)
        elif strategy == MDAPVotingStrategy.WEIGHTED_CONFIDENCE:
            return self._consensus_weighted_confidence(individual, votes)
        elif strategy == MDAPVotingStrategy.WEIGHTED_PERFORMANCE:
            return self._consensus_weighted_performance(individual, votes)
        else:
            return self._consensus_first_k_ahead(individual, votes)

    def _consensus_first_k_ahead(
        self,
        individual: LeanProofStrategy,
        votes: List[AgentVote]
    ) -> ConsensusResult:
        """First-K-ahead consensus"""
        # Group votes by fitness ranges
        fitness_groups = defaultdict(list)
        for vote in votes:
            score_range = int(vote.fitness_score)
            fitness_groups[score_range].append(vote)

        # Find K-ahead winner
        k = self.config.selection_k_ahead
        sorted_ranges = sorted(fitness_groups.keys(), reverse=True)

        if len(sorted_ranges) >= 2:
            leader_count = len(fitness_groups[sorted_ranges[0]])
            runner_up_count = len(fitness_groups[sorted_ranges[1]])

            if leader_count >= runner_up_count + k:
                # Strong consensus
                aggregate_fitness = sorted_ranges[0]
                consensus_level = AgentConsensusLevel.STRONG
            else:
                # Weak consensus
                aggregate_fitness = sorted_ranges[0]
                consensus_level = AgentConsensusLevel.WEAK
        else:
            aggregate_fitness = sorted_ranges[0] if sorted_ranges else 0.0
            consensus_level = AgentConsensusLevel.MAJORITY

        # Calculate agreement ratio
        agreement = len(fitness_groups[sorted_ranges[0]]) / len(votes) if votes else 0.0

        # Vote distribution
        vote_dist = {}
        for vote in votes:
            vote_dist[vote.agent_type] = vote_dist.get(vote.agent_type, 0) + 1

        return ConsensusResult(
            strategy_id=individual.strategy_id,
            consensus_level=consensus_level,
            aggregate_fitness=aggregate_fitness,
            vote_distribution=vote_dist,
            confidence=sum(v.confidence for v in votes) / len(votes),
            agreement_ratio=agreement,
            votes=votes
        )

    def _consensus_majority(
        self,
        individual: LeanProofStrategy,
        votes: List[AgentVote]
    ) -> ConsensusResult:
        """Simple majority consensus"""
        avg_fitness = sum(v.fitness_score for v in votes) / len(votes)
        avg_confidence = sum(v.confidence for v in votes) / len(votes)

        # Determine consensus level
        positive_votes = sum(1 for v in votes if v.fitness_score > 5.0)
        ratio = positive_votes / len(votes) if votes else 0.0

        if ratio >= 0.9:
            consensus_level = AgentConsensusLevel.UNANIMOUS
        elif ratio >= 0.75:
            consensus_level = AgentConsensusLevel.STRONG
        elif ratio >= 0.5:
            consensus_level = AgentConsensusLevel.MAJORITY
        else:
            consensus_level = AgentConsensusLevel.WEAK

        # Vote distribution
        vote_dist = {}
        for vote in votes:
            vote_dist[vote.agent_type] = vote_dist.get(vote.agent_type, 0) + 1

        return ConsensusResult(
            strategy_id=individual.strategy_id,
            consensus_level=consensus_level,
            aggregate_fitness=avg_fitness,
            vote_distribution=vote_dist,
            confidence=avg_confidence,
            agreement_ratio=ratio,
            votes=votes
        )

    def _consensus_weighted_confidence(
        self,
        individual: LeanProofStrategy,
        votes: List[AgentVote]
    ) -> ConsensusResult:
        """Confidence-weighted consensus"""
        total_weight = sum(v.confidence for v in votes)

        if total_weight == 0:
            return self._consensus_majority(individual, votes)

        # Weighted fitness
        weighted_fitness = sum(
            v.fitness_score * v.confidence for v in votes
        ) / total_weight

        # Vote distribution
        vote_dist = {}
        for vote in votes:
            vote_dist[vote.agent_type] = vote_dist.get(vote.agent_type, 0) + 1

        # Agreement based on confidence
        high_conf_votes = sum(1 for v in votes if v.confidence > 0.7)
        agreement = high_conf_votes / len(votes) if votes else 0.0

        return ConsensusResult(
            strategy_id=individual.strategy_id,
            consensus_level=AgentConsensusLevel.MAJORITY,
            aggregate_fitness=weighted_fitness,
            vote_distribution=vote_dist,
            confidence=sum(v.confidence for v in votes) / len(votes),
            agreement_ratio=agreement,
            votes=votes
        )

    def _consensus_weighted_performance(
        self,
        individual: LeanProofStrategy,
        votes: List[AgentVote]
    ) -> ConsensusResult:
        """Performance-weighted consensus"""
        # Get agent performance weights
        total_weight = 0.0
        weighted_fitness = 0.0

        for vote in votes:
            perf = self.agent_performance[vote.agent_id]
            weight = perf["success_rate"] * perf["avg_confidence"]
            total_weight += weight
            weighted_fitness += vote.fitness_score * weight

        if total_weight > 0:
            weighted_fitness /= total_weight

        # Vote distribution
        vote_dist = {}
        for vote in votes:
            vote_dist[vote.agent_type] = vote_dist.get(vote.agent_type, 0) + 1

        return ConsensusResult(
            strategy_id=individual.strategy_id,
            consensus_level=AgentConsensusLevel.MAJORITY,
            aggregate_fitness=weighted_fitness,
            vote_distribution=vote_dist,
            confidence=sum(v.confidence for v in votes) / len(votes),
            agreement_ratio=0.5,
            votes=votes
        )

    def get_agent_consensus(self) -> Dict[str, float]:
        """
        Get consensus scores for all strategies.

        Returns:
            Dictionary mapping strategy IDs to consensus scores
        """
        consensus_scores = {}

        for strategy_id, votes in self.agent_votes.items():
            if votes:
                # Average fitness as consensus score
                consensus_scores[strategy_id] = sum(
                    v.fitness_score for v in votes
                ) / len(votes)

        return consensus_scores

    def rank_by_voting(self) -> List[LeanProofStrategy]:
        """
        Rank strategies by agent voting.

        Returns:
            List of strategies ranked by consensus score
        """
        consensus_scores = self.get_agent_consensus()

        # Sort by consensus score
        ranked = sorted(
            self.strategies,
            key=lambda s: consensus_scores.get(s.strategy_id, 0.0),
            reverse=True
        )

        return ranked

    def select_parents_with_voting(
        self,
        population: 'LeanProofPopulation',
        num_parents: int
    ) -> List[LeanProofStrategy]:
        """
        Select parents using agent voting.

        Args:
            population: Population to select from
            num_parents: Number of parents to select

        Returns:
            List of selected parent strategies
        """
        # Rank by voting
        ranked = self.rank_by_voting()

        # Select top parents
        return ranked[:num_parents]

    def apply_red_flagging(self) -> List[LeanProofStrategy]:
        """
        Filter out red-flagged individuals.

        Returns:
            List of valid (non-flagged) strategies
        """
        if not self.config.enable_red_flagging:
            return self.strategies

        valid_strategies = []

        for strategy in self.strategies:
            is_flagged, reasons = self._check_red_flags(strategy)

            if is_flagged:
                logger.debug(f"Strategy {strategy.strategy_id} red-flagged: {reasons}")
            else:
                valid_strategies.append(strategy)

        return valid_strategies

    def _check_red_flags(self, strategy: LeanProofStrategy) -> Tuple[bool, List[str]]:
        """Check if strategy should be red-flagged"""
        reasons = []

        # Check proof length
        proof_length = len(strategy.proof.lean_code) if strategy.proof else 0
        if proof_length > self.config.max_proof_length:
            reasons.append(f"proof_too_long_{proof_length}")

        # Check confidence
        confidence = strategy.proof.confidence if hasattr(strategy.proof, 'confidence') else 0.0
        if confidence < self.config.min_confidence:
            reasons.append(f"low_confidence_{confidence:.3f}")

        # Check verification status
        if hasattr(strategy, 'verified') and not strategy.verified:
            reasons.append("not_verified")

        # Check for empty proof
        if not strategy.proof or not strategy.proof.tactics:
            reasons.append("empty_proof")

        # Check for too many mutations (potential instability)
        if len(strategy.mutation_history) > 10:
            reasons.append("excessive_mutations")

        return len(reasons) > 0, reasons

    def update_agent_performance(
        self,
        agent_id: str,
        success: bool,
        confidence: float
    ):
        """Update agent performance metrics"""
        perf = self.agent_performance[agent_id]

        # Exponential moving average
        alpha = 0.1
        new_success = 1.0 if success else 0.0

        perf["success_rate"] = (
            alpha * new_success + (1 - alpha) * perf["success_rate"]
        )
        perf["avg_confidence"] = (
            alpha * confidence + (1 - alpha) * perf["avg_confidence"]
        )
        perf["total_votes"] += 1


# =============================================================================
# MDAP-ENHANCED SELECTOR
# =============================================================================

class MDAPLeanSelector:
    """
    Enhanced selection using MDAP agent voting.

    Uses multiple agents to evaluate fitness and select the best individuals
    as parents for the next generation.
    """

    def __init__(
        self,
        agents: List['LeanProofAgent'],
        config: MDAPEvolutionConfig
    ):
        """
        Initialize MDAP selector.

        Args:
            agents: Available MDAP agents
            config: MDAP evolution configuration
        """
        self.agents = agents
        self.config = config
        self.agent_performance: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"success_rate": 0.5, "avg_confidence": 0.5, "selections": 0}
        )

    async def select_with_agent_votes(
        self,
        population: LeanProofPopulation,
        count: int
    ) -> List[LeanProofStrategy]:
        """
        Select strategies using agent voting.

        Args:
            population: Population to select from
            count: Number of strategies to select

        Returns:
            List of selected strategies
        """
        # Evaluate all strategies with agents
        mdap_pop = MDAPLeanPopulation(
            strategies=population.strategies,
            agents=self.agents,
            config=self.config
        )

        await mdap_pop.evaluate_with_mdap(population.strategies)

        # Select by voting
        selected = mdap_pop.select_parents_with_voting(population, count)

        # Update agent performance
        for strategy in selected:
            votes = mdap_pop.agent_votes.get(strategy.strategy_id, [])
            for vote in votes:
                success = strategy.verified if hasattr(strategy, 'verified') else False
                mdap_pop.update_agent_performance(
                    vote.agent_id,
                    success,
                    vote.confidence
                )

        return selected

    def tournament_with_voting(
        self,
        population: LeanProofPopulation,
        tournament_size: int,
        count: int
    ) -> List[LeanProofStrategy]:
        """
        Tournament selection enhanced with voting.

        Args:
            population: Population to select from
            tournament_size: Size of each tournament
            count: Number of parents to select

        Returns:
            List of selected strategies
        """
        selected = []

        for _ in range(count):
            # Randomly select tournament participants
            tournament = random.sample(
                population.strategies,
                min(tournament_size, len(population.strategies))
            )

            # Score each using agent votes
            scored = []
            for strategy in tournament:
                # Simple heuristic score based on fitness and consensus
                score = strategy.fitness
                if hasattr(strategy, 'diversity_score'):
                    score += strategy.diversity_score * 0.2

                scored.append((score, strategy))

            # Select winner
            scored.sort(reverse=True)
            winner = scored[0][1]
            selected.append(winner)

        return selected

    def rank_with_consensus(
        self,
        population: LeanProofPopulation
    ) -> List[LeanProofStrategy]:
        """
        Rank strategies by agent consensus.

        Args:
            population: Population to rank

        Returns:
            Ranked list of strategies
        """
        # Calculate consensus for each strategy
        scored = []

        for strategy in population.strategies:
            consensus_score = self.calculate_consensus_score(
                strategy,
                self.agents
            )
            scored.append((consensus_score, strategy))

        # Sort by consensus score
        scored.sort(key=lambda x: x[0], reverse=True)

        return [s for _, s in scored]

    def calculate_consensus_score(
        self,
        individual: LeanProofStrategy,
        agents: List['LeanProofAgent']
    ) -> float:
        """
        Calculate consensus score for an individual.

        Args:
            individual: Strategy to score
            agents: Agents to query

        Returns:
            Consensus score (higher is better)
        """
        if not agents:
            return individual.fitness

        # Get agent evaluations
        scores = []
        for agent in agents:
            try:
                score = agent.estimate_quality(individual.proof)
                scores.append(score)
            except (ValueError, TypeError, AttributeError):
                continue

        if not scores:
            return individual.fitness

        # Average score
        return sum(scores) / len(scores)


# =============================================================================
# MDAP-ENHANCED CROSSOVER
# =============================================================================

class MDAPLeanCrossover(LeanProofCrossover if EVOLUTION_AVAILABLE else object):
    """
    Crossover enhanced with MDAP agent voting on crossover points.

    Agents vote on:
    - Best crossover strategy (uniform, single-point, two-point, etc.)
    - Best crossover points in parent sequences
    - Which parent tactics to preserve
    """

    def __init__(
        self,
        agents: List['LeanProofAgent'],
        config: MDAPEvolutionConfig,
        crossover_rate: float = 0.8
    ):
        """
        Initialize MDAP crossover.

        Args:
            agents: Available MDAP agents
            config: MDAP evolution configuration
            crossover_rate: Probability of crossover
        """
        if EVOLUTION_AVAILABLE:
            super().__init__(crossover_rate)
        else:
            self.crossover_rate = crossover_rate

        self.agents = agents
        self.config = config

    async def crossover_with_agent_guidance(
        self,
        parent1: LeanProofStrategy,
        parent2: LeanProofStrategy,
        agents: Optional[List['LeanProofAgent']] = None
    ) -> LeanProofStrategy:
        """
        Perform crossover guided by agent voting.

        Args:
            parent1: First parent strategy
            parent2: Second parent strategy
            agents: Optional list of agents (uses self.agents if None)

        Returns:
            Child strategy
        """
        if random.random() > self.crossover_rate:
            # No crossover, return random parent
            return random.choice([parent1, parent2])

        agents = agents or self.agents

        # Vote on crossover strategy
        strategy = await self.vote_on_crossover_strategy(parent1, parent2, agents)

        # Vote on crossover points
        points = await self.vote_on_crossover_points(
            parent1.proof.tactics,
            parent2.proof.tactics,
            agents
        )

        # Perform voted crossover
        child = await self.perform_voted_crossover(parent1, parent2, strategy, points)

        return child

    async def vote_on_crossover_strategy(
        self,
        parent1: LeanProofStrategy,
        parent2: LeanProofStrategy,
        agents: List['LeanProofAgent']
    ) -> CrossoverMethod:
        """
        Vote on best crossover strategy.

        Args:
            parent1: First parent
            parent2: Second parent
            agents: Agents to query

        Returns:
            Selected crossover method
        """
        votes: List[CrossoverVote] = []

        # Collect votes from agents
        for agent in agents:
            try:
                vote = await self._get_crossover_strategy_vote(
                    agent, parent1, parent2
                )
                votes.append(vote)
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.error(f"Agent {agent.agent_id} failed to vote: {e}")

        if not votes:
            # Default to uniform crossover
            return CrossoverMethod.UNIFORM

        # Aggregate votes
        method_counts: Dict[CrossoverMethod, int] = defaultdict(int)
        method_confidence: Dict[CrossoverMethod, float] = defaultdict(float)

        for vote in votes:
            method_counts[vote.crossover_method] += 1
            method_confidence[vote.crossover_method] += vote.confidence

        # Select method with most votes
        best_method = max(method_counts, key=method_counts.get)

        logger.debug(
            f"Crossover strategy selected: {best_method.value} "
            f"({method_counts[best_method]}/{len(votes)} votes)"
        )

        return best_method

    async def _get_crossover_strategy_vote(
        self,
        agent: 'LeanProofAgent',
        parent1: LeanProofStrategy,
        parent2: LeanProofStrategy
    ) -> CrossoverVote:
        """Get agent's vote on crossover strategy"""
        # Heuristic selection based on parent characteristics
        len1 = len(parent1.proof.tactics)
        len2 = len(parent2.proof.tactics)

        # Prefer uniform for similar lengths
        if abs(len1 - len2) <= 2:
            method = CrossoverMethod.UNIFORM
            rationale = "Parents have similar lengths, uniform crossover preferred"
        # Prefer single-point for different lengths
        elif min(len1, len2) > 3:
            method = CrossoverMethod.SINGLE_POINT
            rationale = "Parents have different lengths, single-point crossover"
        # Default to uniform
        else:
            method = CrossoverMethod.UNIFORM
            rationale = "Default uniform crossover"

        return CrossoverVote(
            agent_id=agent.agent_id,
            crossover_method=method,
            crossover_points=[],
            confidence=0.7,
            rationale=rationale
        )

    async def vote_on_crossover_points(
        self,
        parent1_tactics: List[Tactic],
        parent2_tactics: List[Tactic],
        agents: List['LeanProofAgent']
    ) -> List[int]:
        """
        Vote on best crossover points.

        Args:
            parent1_tactics: Tactics from first parent
            parent2_tactics: Tactics from second parent
            agents: Agents to query

        Returns:
            List of selected crossover points
        """
        point_votes: Dict[int, int] = defaultdict(int)

        # Collect votes
        for agent in agents:
            try:
                points = await self._get_crossover_points_vote(
                    agent, parent1_tactics, parent2_tactics
                )
                for point in points:
                    point_votes[point] += 1
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.error(f"Agent {agent.agent_id} failed to vote on points: {e}")

        if not point_votes:
            # Default: midpoint
            return [max(1, min(len(parent1_tactics), len(parent2_tactics)) // 2)]

        # Select most voted points
        sorted_points = sorted(point_votes.items(), key=lambda x: x[1], reverse=True)

        # Return top 1-2 points
        num_points = min(2, len(sorted_points))
        selected = [p for p, _ in sorted_points[:num_points]]

        return sorted(selected)

    async def _get_crossover_points_vote(
        self,
        agent: 'LeanProofAgent',
        parent1_tactics: List[Tactic],
        parent2_tactics: List[Tactic]
    ) -> List[int]:
        """Get agent's vote on crossover points"""
        # Suggest points at 1/3 and 2/3
        len1 = len(parent1_tactics)
        len2 = len(parent2_tactics)
        min_len = min(len1, len2)

        if min_len <= 2:
            return [1]

        # Suggest strategic points
        points = [
            max(1, min_len // 3),
            max(1, 2 * min_len // 3)
        ]

        return points

    async def perform_voted_crossover(
        self,
        parent1: LeanProofStrategy,
        parent2: LeanProofStrategy,
        strategy: CrossoverMethod,
        points: List[int]
    ) -> LeanProofStrategy:
        """
        Perform crossover using voted strategy and points.

        Args:
            parent1: First parent
            parent2: Second parent
            strategy: Crossover method to use
            points: Crossover points

        Returns:
            Child strategy
        """
        import copy

        # Create child proof
        child_proof = copy.deepcopy(parent1.proof)
        child_proof.proof_id = str(uuid.uuid4())
        child_proof.tactics = []

        tactics1 = parent1.proof.tactics
        tactics2 = parent2.proof.tactics

        if strategy == CrossoverMethod.UNIFORM:
            # Uniform crossover at voted points
            for i in range(max(len(tactics1), len(tactics2))):
                if i < len(tactics1) and i < len(tactics2):
                    # Alternate based on points
                    use_parent1 = i % 2 == 0
                    child_proof.tactics.append(
                        copy.deepcopy(tactics1[i] if use_parent1 else tactics2[i])
                    )
                elif i < len(tactics1):
                    child_proof.tactics.append(copy.deepcopy(tactics1[i]))
                else:
                    child_proof.tactics.append(copy.deepcopy(tactics2[i]))

        elif strategy == CrossoverMethod.SINGLE_POINT:
            # Single-point crossover
            point = points[0] if points else max(1, min(len(tactics1), len(tactics2)) // 2)

            child_proof.tactics.extend(copy.deepcopy(tactics1[:point]))
            child_proof.tactics.extend(copy.deepcopy(tactics2[point:]))

        elif strategy == CrossoverMethod.TWO_POINT:
            # Two-point crossover
            if len(points) >= 2 and len(tactics1) > points[1] and len(tactics2) > points[1]:
                point1, point2 = points[0], points[1]

                child_proof.tactics.extend(copy.deepcopy(tactics1[:point1]))
                child_proof.tactics.extend(copy.deepcopy(tactics2[point1:point2]))
                child_proof.tactics.extend(copy.deepcopy(tactics1[point2:]))
            else:
                # Fall back to single-point
                return await self.perform_voted_crossover(
                    parent1, parent2, CrossoverMethod.SINGLE_POINT, points
                )

        else:
            # Default to uniform
            return await self.perform_voted_crossover(
                parent1, parent2, CrossoverMethod.UNIFORM, points
            )

        # Create child strategy
        child_strategy = LeanProofStrategy(
            proof=child_proof,
            generation=max(parent1.generation, parent2.generation) + 1,
            parents=[parent1.strategy_id, parent2.strategy_id]
        )

        return child_strategy


# =============================================================================
# MDAP-ENHANCED MUTATOR
# =============================================================================

class MDAPLeanMutator(LeanProofMutator if EVOLUTION_AVAILABLE else object):
    """
    Mutation enhanced with MDAP agent voting on mutation operations.

    Agents:
    - Suggest mutations (tactic changes, insertions, deletions)
    - Vote on best mutation
    - Apply selected mutation
    """

    def __init__(
        self,
        agents: List['LeanProofAgent'],
        config: MDAPEvolutionConfig,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.5
    ):
        """
        Initialize MDAP mutator.

        Args:
            agents: Available MDAP agents
            config: MDAP evolution configuration
            mutation_rate: Probability of mutation
            mutation_strength: Strength of mutations
        """
        if EVOLUTION_AVAILABLE:
            super().__init__(mutation_rate, mutation_strength)
        else:
            self.mutation_rate = mutation_rate
            self.mutation_strength = mutation_strength

        self.agents = agents
        self.config = config

    async def mutate_with_agent_guidance(
        self,
        individual: LeanProofStrategy,
        agents: Optional[List['LeanProofAgent']] = None
    ) -> LeanProofStrategy:
        """
        Apply mutation guided by agent voting.

        Args:
            individual: Strategy to mutate
            agents: Optional list of agents

        Returns:
            Mutated strategy
        """
        agents = agents or self.agents

        # Collect mutation suggestions
        suggestions = await self.collect_mutation_suggestions(individual, agents)

        if not suggestions:
            # No suggestions, return original
            return individual

        # Vote on best mutation
        best_mutation = await self.vote_on_mutation(suggestions)

        # Apply selected mutation
        mutated = await self.apply_mutation(individual, best_mutation)

        return mutated

    async def collect_mutation_suggestions(
        self,
        individual: LeanProofStrategy,
        agents: List['LeanProofAgent']
    ) -> List[MutationSuggestion]:
        """
        Collect mutation suggestions from agents.

        Args:
            individual: Strategy to mutate
            agents: Agents to query

        Returns:
            List of mutation suggestions
        """
        suggestions = []

        for agent in agents:
            try:
                agent_suggestions = await self._get_agent_mutations(agent, individual)
                suggestions.extend(agent_suggestions)
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.error(f"Agent {agent.agent_id} failed to suggest mutations: {e}")

        return suggestions

    async def _get_agent_mutations(
        self,
        agent: 'LeanProofAgent',
        individual: LeanProofStrategy
    ) -> List[MutationSuggestion]:
        """Get mutation suggestions from an agent"""
        suggestions = []
        tactics = individual.proof.tactics

        if not tactics:
            return suggestions

        # Suggest mutations for random positions
        num_suggestions = min(3, len(tactics))

        for _ in range(num_suggestions):
            position = random.randint(0, len(tactics) - 1)
            old_tactic = tactics[position]

            # Suggest new tactic based on agent type
            if agent.agent_type.value == "evolution":
                new_tactic = self._suggest_evolution_mutation(old_tactic)
            elif agent.agent_type.value == "mcts":
                new_tactic = self._suggest_mcts_mutation(old_tactic)
            elif agent.agent_type.value == "adversarial":
                new_tactic = self._suggest_adversarial_mutation(old_tactic)
            else:
                new_tactic = self._suggest_random_mutation()

            mutation_type = MutationType.TACTIC_SUBSTITUTION

            suggestions.append(MutationSuggestion(
                agent_id=agent.agent_id,
                mutation_type=mutation_type,
                position=position,
                old_tactic=old_tactic.name,
                new_tactic=new_tactic,
                confidence=random.uniform(0.5, 0.9),
                rationale=f"Agent {agent.agent_type.value} suggests substitution",
                estimated_improvement=random.uniform(0.0, 0.3)
            ))

        return suggestions

    def _suggest_evolution_mutation(self, old_tactic: Tactic) -> str:
        """Suggest mutation using evolutionary heuristics"""
        substitutions = {
            "simp": "simp_all",
            "simp_all": "simp",
            "rw": "simp",
            "apply": "exact",
            "exact": "refine",
            "constructor": "refine"
        }

        return substitutions.get(old_tactic.name, "simp")

    def _suggest_mcts_mutation(self, old_tactic: Tactic) -> str:
        """Suggest mutation using MCTS heuristics"""
        # Prefer exploratory tactics
        exploratory = ["cases", "induction", "constructor", "refine"]
        return random.choice(exploratory)

    def _suggest_adversarial_mutation(self, old_tactic: Tactic) -> str:
        """Suggest mutation using adversarial heuristics"""
        # Prefer challenging tactics
        challenging = ["aesop", "simp?", "by", "calc"]
        return random.choice(challenging)

    def _suggest_random_mutation(self) -> str:
        """Suggest random tactic"""
        all_tactics = ["simp", "rw", "apply", "exact", "cases", "induction", "constructor"]
        return random.choice(all_tactics)

    async def vote_on_mutation(
        self,
        suggestions: List[MutationSuggestion]
    ) -> MutationSuggestion:
        """
        Vote on best mutation suggestion.

        Args:
            suggestions: Mutation suggestions to vote on

        Returns:
            Selected mutation suggestion
        """
        if not suggestions:
            # Return empty suggestion
            return MutationSuggestion(
                agent_id="none",
                mutation_type=MutationType.TACTIC_SUBSTITUTION,
                position=0,
                old_tactic="",
                new_tactic="",
                confidence=0.0,
                rationale="No suggestions",
                estimated_improvement=0.0
            )

        # Score suggestions
        strategy = self.config.mutation_voting_strategy

        if strategy == MDAPVotingStrategy.WEIGHTED_CONFIDENCE:
            # Select by confidence
            scored = [(s.confidence, s) for s in suggestions]
            scored.sort(reverse=True)
            return scored[0][1]

        elif strategy == MDAPVotingStrategy.MAJORITY:
            # Group by mutation type, select most common
            type_counts: Dict[MutationType, List[MutationSuggestion]] = defaultdict(list)
            for s in suggestions:
                type_counts[s.mutation_type].append(s)

            # Select type with most suggestions
            most_common_type = max(type_counts, key=lambda k: len(type_counts[k]))
            candidates = type_counts[most_common_type]

            # Return highest confidence in that type
            return max(candidates, key=lambda s: s.confidence)

        else:
            # Default: select by estimated improvement
            scored = [(s.estimated_improvement, s) for s in suggestions]
            scored.sort(reverse=True)
            return scored[0][1]

    async def apply_mutation(
        self,
        individual: LeanProofStrategy,
        mutation: MutationSuggestion
    ) -> LeanProofStrategy:
        """
        Apply selected mutation.

        Args:
            individual: Strategy to mutate
            mutation: Mutation to apply

        Returns:
            Mutated strategy
        """
        import copy

        # Create copy
        mutated = copy.deepcopy(individual)
        mutated.strategy_id = str(uuid.uuid4())
        mutated.parents = [individual.strategy_id]
        mutated.mutation_history = []

        # Apply mutation
        if mutation.mutation_type == MutationType.TACTIC_SUBSTITUTION:
            if 0 <= mutation.position < len(mutated.proof.tactics):
                # Create new tactic
                new_tactic = Tactic(name=mutation.new_tactic)
                mutated.proof.tactics[mutation.position] = new_tactic
                mutated.mutation_history.append(MutationType.TACTIC_SUBSTITUTION)

        elif mutation.mutation_type == MutationType.STEP_INSERTION:
            new_tactic = Tactic(name=mutation.new_tactic)
            if mutated.proof.tactics:
                mutated.proof.tactics.insert(
                    min(mutation.position, len(mutated.proof.tactics)),
                    new_tactic
                )
            else:
                mutated.proof.tactics.append(new_tactic)
            mutated.mutation_history.append(MutationType.STEP_INSERTION)

        elif mutation.mutation_type == MutationType.STEP_DELETION:
            if 0 <= mutation.position < len(mutated.proof.tactics):
                mutated.proof.tactics.pop(mutation.position)
                mutated.mutation_history.append(MutationType.STEP_DELETION)

        # Update generation
        mutated.generation = individual.generation + 1

        return mutated


# =============================================================================
# MDAP-ENHANCED EVOLUTION ENGINE
# =============================================================================

class MDAPEvolutionEngine(LeanProofEvolutionEngine if EVOLUTION_AVAILABLE else object):
    """
    Main evolutionary engine enhanced with MDAP voting.

    Uses MDAP agents during:
    - Selection (multi-agent fitness evaluation)
    - Crossover (agent-guided crossover strategy and points)
    - Mutation (agent-suggested and voted mutations)
    - Evaluation (multi-agent consensus)
    """

    def __init__(
        self,
        theorem: str,
        theorem_name: Optional[str] = None,
        config: Optional[MDAPEvolutionConfig] = None,
        agents: Optional[List['LeanProofAgent']] = None
    ):
        """
        Initialize MDAP evolution engine.

        Args:
            theorem: Theorem to prove
            theorem_name: Optional theorem name
            config: MDAP evolution configuration
            agents: Optional list of MDAP agents
        """
        self.theorem = theorem
        self.theorem_name = theorem_name or "mdap_evolved_theorem"
        self.config = config or MDAPEvolutionConfig()

        # Initialize agents if not provided
        if agents:
            self.agents = agents
        else:
            self.agents = self._create_default_agents()

        # Initialize components
        self.selector = MDAPLeanSelector(self.agents, self.config)
        self.crossover = MDAPLeanCrossover(self.agents, self.config, self.config.crossover_rate)
        self.mutator = MDAPLeanMutator(self.agents, self.config, self.config.mutation_rate)

        if EVOLUTION_AVAILABLE:
            super().__init__(
                theorem=theorem,
                theorem_name=theorem_name,
                population_size=self.config.population_size,
                max_generations=self.config.max_generations,
                mutation_rate=self.config.mutation_rate,
                crossover_rate=self.config.crossover_rate,
                selection_method=self.config.selection_method,
                server_url=self.config.server_url,
                cache_enabled=self.config.cache_enabled,
                parallel_evaluation=self.config.parallel_evaluation
            )
        else:
            self.population_size = self.config.population_size
            self.max_generations = self.config.max_generations
            self.population = None
            self.current_generation = 0
            self.family_tree = defaultdict(list)
            self.failed_attempts = []
            self.statistics_history = []
            self.convergence_history = []

        # MDAP-specific tracking
        self.consensus_history: List[ConsensusResult] = []
        self.agent_performance: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"selections": 0, "crossovers": 0, "mutations": 0}
        )
        self.red_flag_count = 0

    def _create_default_agents(self) -> List['LeanProofAgent']:
        """Create default set of MDAP agents"""
        agents = []

        if MDAP_ENGINE_AVAILABLE:
            # Create agents for different strategies
            strategies_to_create = [
                ProofStrategy.EVOLUTION,
                ProofStrategy.MCTS,
                ProofStrategy.DIRECT
            ]

            for strategy in strategies_to_create:
                try:
                    agent = LeanProofAgent(
                        agent_id=f"{strategy.value}_agent",
                        agent_type=strategy,
                        config=None  # Use default config
                    )
                    agents.append(agent)
                except (ValueError, TypeError, AttributeError) as e:
                    logger.warning(f"Failed to create agent {strategy}: {e}")

        return agents

    async def evolve_with_mdap(self) -> MDAPResult:
        """
        Run MDAP-enhanced evolutionary proof generation.

        Returns:
            MDAPResult with best proof and metrics
        """
        start_time = time.time()
        total_evaluations = 0

        logger.info(f"Starting MDAP-enhanced evolution for: {self.theorem}")
        logger.info(f"Population size: {self.config.population_size}")
        logger.info(f"Max generations: {self.config.max_generations}")
        logger.info(f"Agents: {[a.agent_type.value for a in self.agents]}")

        try:
            # Generate initial population
            logger.info("Generating initial population...")
            if EVOLUTION_AVAILABLE:
                initial_strategies = await self.generate_initial_population()
            else:
                initial_strategies = self._generate_simple_population()

            # Create MDAP-enhanced population
            self.population = MDAPLeanPopulation(
                strategies=initial_strategies,
                agents=self.agents,
                config=self.config
            )

            # Evaluate with MDAP
            logger.info("Evaluating initial population with MDAP...")
            await self.evaluate_population_with_mdap()

            # Record initial statistics
            stats = self.population.calculate_statistics()
            self.statistics_history.append(stats)
            logger.info(f"Generation 0: Best fitness = {stats.best_fitness:.4f}")

            # Evolution loop
            for generation in range(1, self.config.max_generations + 1):
                self.current_generation = generation
                self.population.generation = generation

                logger.info(f"Generation {generation}")

                # Check for early termination
                best_strategy = self.population.get_best_strategy()
                if best_strategy and best_strategy.verified:
                    logger.info("Found verified proof!")
                    break

                # Create next generation with MDAP
                await self.next_generation_with_mdap(generation)

                # Evaluate with MDAP
                total_evaluations += len(self.population.strategies)
                await self.evaluate_population_with_mdap()

                # Calculate statistics
                stats = self.population.calculate_statistics()
                self.statistics_history.append(stats)
                self.convergence_history.append(stats.average_fitness)

                # Apply red-flagging
                if self.config.enable_red_flagging:
                    valid = self.population.apply_red_flagging()
                    filtered_count = len(self.population.strategies) - len(valid)
                    if filtered_count > 0:
                        self.red_flag_count += filtered_count
                        logger.info(f"Red-flagged {filtered_count} strategies")
                        self.population.strategies = valid

                logger.info(
                    f"Generation {generation}: "
                    f"Best = {stats.best_fitness:.4f}, "
                    f"Avg = {stats.average_fitness:.4f}, "
                    f"Verified = {stats.verified_count}"
                )

            # Get best strategy
            best_strategy = self.population.get_best_strategy()

            evolution_time = time.time() - start_time
            logger.info(f"MDAP evolution completed in {evolution_time:.2f}s")

            # Calculate MDAP-specific metrics
            voting_efficiency = self._calculate_voting_efficiency()
            agent_agreement = self._calculate_agent_agreement()

            # Create result
            result = MDAPResult(
                success=best_strategy.verified if best_strategy else False,
                best_proof=best_strategy.proof if best_strategy else None,
                best_strategy=best_strategy,
                generations_completed=self.current_generation,
                total_evaluations=total_evaluations,
                evolution_time=evolution_time,
                statistics_history=self.statistics_history,
                agent_performance=dict(self.agent_performance),
                consensus_history=self.consensus_history,
                voting_efficiency=voting_efficiency,
                agent_agreement_rate=agent_agreement,
                red_flag_count=self.red_flag_count,
                family_tree=dict(self.family_tree),
                failed_attempts=self.failed_attempts,
                convergence_history=self.convergence_history
            )

            return result

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"MDAP evolution failed: {e}", exc_info=True)
            return MDAPResult(
                success=False,
                best_proof=None,
                best_strategy=None,
                generations_completed=self.current_generation,
                total_evaluations=total_evaluations,
                evolution_time=time.time() - start_time,
                statistics_history=self.statistics_history,
                agent_performance=dict(self.agent_performance),
                consensus_history=[],
                voting_efficiency=0.0,
                agent_agreement_rate=0.0,
                red_flag_count=self.red_flag_count,
                family_tree=dict(self.family_tree),
                failed_attempts=[{"error": str(e)}],
                convergence_history=self.convergence_history
            )

    async def next_generation_with_mdap(self, generation: int):
        """
        Create next generation using MDAP-enhanced operators.

        Args:
            generation: Current generation number
        """
        current_strategies = self.population.strategies
        population_size = len(current_strategies)

        # Elitism: keep best strategies
        num_elites = self.config.elitism_count
        elites = self.population.get_elites(num_elites)

        # Select parents with MDAP voting
        num_offspring = population_size - num_elites
        parents = await self.selector.select_with_agent_votes(
            self.population,
            num_offspring * 2
        )

        # Create offspring through MDAP-enhanced crossover and mutation
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            # MDAP-enhanced crossover
            child = await self.crossover.crossover_with_agent_guidance(
                parent1, parent2, self.agents
            )

            # MDAP-enhanced mutation
            if random.random() < self.config.mutation_rate:
                child = await self.mutator.mutate_with_agent_guidance(
                    child, self.agents
                )

            # Track family tree
            self.family_tree[f"{parent1.strategy_id}+{parent2.strategy_id}"].append(
                child.strategy_id
            )

            offspring.append(child)

            if len(offspring) >= num_offspring:
                break

        # Combine elites and offspring
        new_strategies = elites + offspring[:num_offspring]

        # Update population
        self.population.strategies = new_strategies

    async def select_with_mdap(
        self,
        population: LeanProofPopulation
    ) -> List[LeanProofStrategy]:
        """
        Select strategies using MDAP agent voting.

        Args:
            population: Population to select from

        Returns:
            Selected strategies
        """
        return await self.selector.select_with_agent_votes(
            population,
            self.config.elitism_count * 2
        )

    async def crossover_with_mdap(
        self,
        parents: List[LeanProofStrategy]
    ) -> List[LeanProofStrategy]:
        """
        Perform crossover using MDAP guidance.

        Args:
            parents: Parent strategies

        Returns:
            Offspring strategies
        """
        offspring = []

        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            child = await self.crossover.crossover_with_agent_guidance(
                parent1, parent2, self.agents
            )

            offspring.append(child)

        return offspring

    async def mutate_with_mdap(
        self,
        population: LeanProofPopulation
    ) -> LeanProofPopulation:
        """
        Apply mutations using MDAP guidance.

        Args:
            population: Population to mutate

        Returns:
            Population with mutations applied
        """
        for strategy in population.strategies:
            if random.random() < self.config.mutation_rate:
                mutated = await self.mutator.mutate_with_agent_guidance(
                    strategy, self.agents
                )
                # Replace in population
                idx = population.strategies.index(strategy)
                population.strategies[idx] = mutated

        return population

    async def evaluate_population_with_mdap(self):
        """Evaluate population using MDAP agents"""
        if not self.population:
            return

        # Evaluate with MDAP
        results = await self.population.evaluate_with_mdap(self.population.strategies)

        # Record consensus
        for consensus in results.values():
            self.consensus_history.append(consensus)

    def _generate_simple_population(self) -> List[LeanProofStrategy]:
        """Generate simple initial population"""
        strategies = []

        for i in range(self.config.population_size):
            proof = LeanProof(
                theorem_name=self.theorem_name,
                theorem_statement=self.theorem,
                lean_code=f"theorem {self.theorem_name} : {self.theorem} := by\n  sorry",
                tactics=[]
            )

            strategy = LeanProofStrategy(
                proof=proof,
                generation=0,
                strategy_id=f"simple_{i}"
            )

            strategies.append(strategy)

        return strategies

    def _calculate_voting_efficiency(self) -> float:
        """Calculate voting efficiency metric"""
        if not self.consensus_history:
            return 0.0

        # Count strong consensus results
        strong_count = sum(
            1 for c in self.consensus_history
            if c.consensus_level in [AgentConsensusLevel.UNANIMOUS, AgentConsensusLevel.STRONG]
        )

        return strong_count / len(self.consensus_history)

    def _calculate_agent_agreement(self) -> float:
        """Calculate average agent agreement rate"""
        if not self.consensus_history:
            return 0.0

        total_agreement = sum(c.agreement_ratio for c in self.consensus_history)
        return total_agreement / len(self.consensus_history)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def evolve_with_mdap(
    theorem: str,
    theorem_name: Optional[str] = None,
    config: Optional[MDAPEvolutionConfig] = None,
    agents: Optional[List['LeanProofAgent']] = None
) -> MDAPResult:
    """
    Convenience function for MDAP-enhanced evolution.

    Args:
        theorem: Theorem statement
        theorem_name: Optional theorem name
        config: MDAP evolution configuration
        agents: Optional list of MDAP agents

    Returns:
        MDAPResult with best proof and metrics
    """
    engine = MDAPEvolutionEngine(
        theorem=theorem,
        theorem_name=theorem_name,
        config=config,
        agents=agents
    )

    return await engine.evolve_with_mdap()


def create_mdap_config(
    population_size: int = 20,
    max_generations: int = 50,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.8,
    selection_agents: Optional[List[str]] = None,
    **kwargs
) -> MDAPEvolutionConfig:
    """
    Create MDAP evolution configuration.

    Args:
        population_size: Size of population
        max_generations: Maximum generations
        mutation_rate: Mutation rate
        crossover_rate: Crossover rate
        selection_agents: Agent types for selection
        **kwargs: Additional configuration

    Returns:
        MDAPEvolutionConfig object
    """
    return MDAPEvolutionConfig(
        population_size=population_size,
        max_generations=max_generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        selection_agents=selection_agents or ["evolution", "mcts", "direct"],
        **kwargs
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def example_usage():
    """Example usage of MDAP-enhanced evolution"""

    print("=" * 80)
    print("MDAP-Enhanced Evolution Example")
    print("=" * 80)

    # Simple theorem
    theorem = "forall (n m : Nat), n + m = m + n"

    print(f"\nTheorem: {theorem}\n")

    # Create configuration
    config = create_mdap_config(
        population_size=10,
        max_generations=5,
        mutation_rate=0.2,
        selection_agents=["evolution", "mcts", "direct"]
    )

    # Run evolution
    result = await evolve_with_mdap(
        theorem=theorem,
        theorem_name="addition_commutativity",
        config=config
    )

    # Print results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    print(f"\nSuccess: {result.success}")
    print(f"Generations: {result.generations_completed}")
    print(f"Evaluations: {result.total_evaluations}")
    print(f"Time: {result.evolution_time:.2f}s")
    print(f"Voting Efficiency: {result.voting_efficiency:.3f}")
    print(f"Agent Agreement: {result.agent_agreement_rate:.3f}")
    print(f"Red Flags: {result.red_flag_count}")

    if result.best_proof:
        print("\n" + "=" * 80)
        print("Best Proof")
        print("=" * 80)
        print(f"\n{result.best_proof.lean_code}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    "MDAPLeanPopulation",
    "MDAPLeanSelector",
    "MDAPLeanCrossover",
    "MDAPLeanMutator",
    "MDAPEvolutionEngine",

    # Configuration
    "MDAPEvolutionConfig",
    "create_mdap_config",

    # Results
    "MDAPResult",
    "ConsensusResult",
    "AgentVote",
    "MutationSuggestion",
    "CrossoverVote",

    # Enums
    "MDAPVotingStrategy",
    "AgentConsensusLevel",

    # Convenience functions
    "evolve_with_mdap",
]
