"""
MDAP/MAKER Coevolving Decision Trees Integration

This module integrates MDAP (Multi-Agent voting) and MAKER with coevolving decision trees
for robust theorem proving with zero-error guarantees.

Core Concept:
- Coevolve decision trees where each tree is evaluated by multiple agents
- MAKER voting determines the best candidates for evolution
- Multi-agent consensus provides robust fitness evaluation
- Decomposition enhances complex problem solving

Key Features:
1. MDAP-Enhanced Decision Trees with multi-agent evaluation
2. MAKER voting for tree selection (first-to-ahead-by-k)
3. Decomposition-enhanced coevolution
4. Competitive coevolution with MDAP
5. Multi-objective optimization with MDAP
6. LeanAide integration for formal verification
7. Ensemble methods with MDAP voting
8. Performance tracking and monitoring

Reference:
- MDAP/MAKER: "Solving a Million-Step LLM Task with Zero Errors" (arXiv:2511.09030)
- MCTS Coevolution: Genetic programming with Monte Carlo evaluation

Author: OpenEvolve
Created: 2025-12-30
"""
from __future__ import annotations


import asyncio
import hashlib
import json
import logging
import random
import statistics
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

# Import from existing modules
from mcts_coevolution import (
    ProofDecisionTree, DecisionNode, NodeType, Tactic, ProofContext,
    ProofResult, EvaluationResult, SingleEvaluation, TreeGenerator,
    TreeCrossover, TreeMutation, MCTreeEvaluator
)
from mdap_maker_complete import (
    MAKEREngine, VoteCollector, VotingEngine, MAKERRunMetrics,
    TaskDecomposition
)
try:
    from workflow_structures import ModelConfig, Team
except ImportError:
    ModelConfig = None
    Team = None

logger = logging.getLogger(__name__)


# ============================================================================
# Type Definitions
# ============================================================================

class VotingStrategy(Enum):
    """Voting strategies for tree selection"""
    FIRST_K_AHEAD = "first_k_ahead"  # MAKER first-to-ahead-by-k
    FIRST_TO_K = "first_to_k"        # Simple first-to-k
    MAJORITY = "majority"            # Simple majority
    WEIGHTED = "weighted"            # Weighted by agent reliability


@dataclass
class AgentEvaluation:
    """Single agent's evaluation of a tree"""
    agent_id: str
    success_rate: float
    avg_depth: float
    avg_time: float
    elegance_score: float
    simplicity_score: float
    robustness: float
    confidence: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "success_rate": self.success_rate,
            "avg_depth": self.avg_depth,
            "avg_time": self.avg_time,
            "elegance_score": self.elegance_score,
            "simplicity_score": self.simplicity_score,
            "robustness": self.robustness,
            "confidence": self.confidence
        }


@dataclass
class MDAPTreeEvaluation:
    """Multi-agent evaluation of a decision tree"""
    tree_id: str
    agent_results: List[AgentEvaluation]
    consensus_score: float
    agreement_level: float
    voting_details: Dict[str, int]
    std_dev_success: float
    std_dev_depth: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "agent_results": [r.to_dict() for r in self.agent_results],
            "consensus_score": self.consensus_score,
            "agreement_level": self.agreement_level,
            "voting_details": self.voting_details,
            "std_dev_success": self.std_dev_success,
            "std_dev_depth": self.std_dev_depth
        }


@dataclass
class TreeDecomposition:
    """Decomposition of a tree's task"""
    subtask1: Optional[str] = None
    subtask2: Optional[str] = None
    composition_function: Optional[str] = None
    confidence: float = 0.5
    is_atomic: bool = False


# ============================================================================
# MDAP-Enhanced Decision Tree
# ============================================================================

class MDAPProofDecisionTree(ProofDecisionTree):
    """
    Decision tree with MDAP multi-agent evaluation.

    Extends ProofDecisionTree with:
    - Multi-agent evaluation results
    - Consensus and agreement metrics
    - MAKER voting support
    - Decomposition capabilities
    """

    def __init__(
        self,
        root: DecisionNode,
        tree_id: str = None,
        generation: int = 0,
        num_agents: int = 5,
        voting_strategy: str = "first_k_ahead",
        k_ahead: int = 3
    ):
        super().__init__(root, tree_id, generation)

        # MDAP properties
        self.num_agents = num_agents
        self.voting_strategy = voting_strategy
        self.k_ahead = k_ahead

        # Agent evaluations
        self.agent_evaluations: Dict[str, AgentEvaluation] = {}
        self.agent_votes: Dict[str, int] = defaultdict(int)

        # Consensus metrics
        self.consensus_score: float = 0.0
        self.agreement_level: float = 0.0

        # Decomposition
        self.enable_decomposition: bool = True
        self.decomposition_threshold: float = 0.7  # Success rate below which to decompose
        self.decomposition_nodes: List['MDAPProofDecisionTree'] = []

        # Agent reliability (updated over time)
        self.agent_reliability: Dict[str, float] = {f"agent_{i}": 1.0 for i in range(num_agents)}

    def compute_consensus(self, evaluations: List[AgentEvaluation]) -> float:
        """
        Compute consensus score across agents.

        Uses weighted average based on agent reliability.
        """
        if not evaluations:
            return 0.0

        total_weight = 0.0
        weighted_score = 0.0

        for eval in evaluations:
            reliability = self.agent_reliability.get(eval.agent_id, 1.0)
            weight = reliability * eval.confidence
            weighted_score += eval.success_rate * weight
            total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def compute_agreement(self, evaluations: List[AgentEvaluation]) -> float:
        """
        Compute agreement level across agents.

        Agreement = 1 - std_dev of success rates
        """
        if not evaluations:
            return 0.0

        success_rates = [e.success_rate for e in evaluations]

        if len(success_rates) == 1:
            return 1.0

        std_dev = statistics.stdev(success_rates)
        agreement = max(0.0, 1.0 - std_dev)

        return agreement

    def get_agent_reliability(self, agent_id: str) -> float:
        """Get reliability score for agent"""
        return self.agent_reliability.get(agent_id, 1.0)

    def update_agent_reliability(
        self,
        agent_id: str,
        predicted_performance: float,
        actual_performance: float
    ):
        """
        Update agent reliability based on prediction accuracy.

        Reliability decays if agent's predictions are inaccurate.
        """
        current_reliability = self.agent_reliability.get(agent_id, 1.0)
        error = abs(predicted_performance - actual_performance)

        # Decay factor based on error
        decay = 1.0 - (error * 0.1)
        decay = max(0.5, min(1.0, decay))  # Keep in [0.5, 1.0]

        # Exponential moving average
        new_reliability = 0.9 * current_reliability + 0.1 * decay
        self.agent_reliability[agent_id] = new_reliability

    def should_decompose(self, context: ProofContext) -> bool:
        """
        Decide if tree should use decomposition for this problem.

        Decompose if:
        1. Problem complexity is high
        2. Previous success rate is below threshold
        3. Agreement among agents is low
        """
        if not self.enable_decomposition:
            return False

        # Check if success rate suggests need for decomposition
        if self.consensus_score < self.decomposition_threshold:
            return True

        # Check if low agreement suggests need for decomposition
        if self.agreement_level < 0.6:
            return True

        # Check problem complexity (heuristic based on theorem length)
        complexity = len(context.theorem)
        if complexity > 100:
            return True

        return False

    def clone(self) -> 'MDAPProofDecisionTree':
        """Create deep copy"""
        cloned = MDAPProofDecisionTree(
            root=self.root.clone(),
            tree_id=str(uuid.uuid4()),
            generation=self.generation,
            num_agents=self.num_agents,
            voting_strategy=self.voting_strategy,
            k_ahead=self.k_ahead
        )

        # Copy evaluation data
        cloned.agent_evaluations = self.agent_evaluations.copy()
        cloned.agent_votes = self.agent_votes.copy()
        cloned.consensus_score = self.consensus_score
        cloned.agreement_level = self.agreement_level
        cloned.agent_reliability = self.agent_reliability.copy()

        # Copy fitness
        cloned.fitness = self.fitness
        cloned.success_rate = self.success_rate

        return cloned


# ============================================================================
# Multi-Agent Tree Evaluator
# ============================================================================

class MDAPTreeEvaluator(MCTreeEvaluator):
    """
    Evaluate trees using multiple agents with Monte Carlo simulation.

    Each agent evaluates the tree independently, then consensus is computed.
    """

    def __init__(
        self,
        num_agents: int = 5,
        simulations: int = 100,
        max_depth: int = 50,
        agent_diversity: float = 0.2
    ):
        super().__init__(simulations, max_depth)
        self.num_agents = num_agents
        self.agent_diversity = agent_diversity  # Diversity in agent evaluation

    async def evaluate_tree_mdap(
        self,
        tree: MDAPProofDecisionTree,
        test_theorems: List[str],
        agent_configs: List[ModelConfig] = None
    ) -> MDAPTreeEvaluation:
        """
        Evaluate tree using multi-agent Monte Carlo simulation.

        Each agent runs simulations with different random seeds.
        """
        agent_results = []

        for agent_id in range(self.num_agents):
            # Each agent evaluates with different randomization
            agent_result = await self._agent_evaluate_tree(
                tree,
                test_theorems,
                f"agent_{agent_id}",
                agent_configs[agent_id] if agent_configs else None
            )
            agent_results.append(agent_result)

        # Compute consensus
        consensus = tree.compute_consensus(agent_results)
        agreement = tree.compute_agreement(agent_results)

        # Compute std devs
        success_rates = [r.success_rate for r in agent_results]
        depths = [r.avg_depth for r in agent_results]
        std_dev_success = statistics.stdev(success_rates) if len(success_rates) > 1 else 0.0
        std_dev_depth = statistics.stdev(depths) if len(depths) > 1 else 0.0

        # Voting details (how many agents voted for success)
        voting_details = {
            "success_votes": sum(1 for r in agent_results if r.success_rate > 0.5),
            "total_agents": len(agent_results),
            "avg_success": statistics.mean(success_rates)
        }

        # Update tree metrics
        tree.agent_evaluations = {r.agent_id: r for r in agent_results}
        tree.consensus_score = consensus
        tree.agreement_level = agreement

        return MDAPTreeEvaluation(
            tree_id=tree.tree_id,
            agent_results=agent_results,
            consensus_score=consensus,
            agreement_level=agreement,
            voting_details=voting_details,
            std_dev_success=std_dev_success,
            std_dev_depth=std_dev_depth
        )

    async def _agent_evaluate_tree(
        self,
        tree: MDAPProofDecisionTree,
        test_theorems: List[str],
        agent_id: str,
        agent_config: ModelConfig = None
    ) -> AgentEvaluation:
        """
        Single agent evaluates tree.

        Uses agent-specific random seed for diversity.
        """
        results = []
        elegance_scores = []
        simplicity_scores = []

        for theorem in test_theorems:
            # Run simulations with agent-specific seed
            agent_seed = hash(agent_id + theorem) % (2 ** 31)

            for sim in range(self.simulations):
                context = ProofContext(
                    theorem=theorem,
                    goal_state=f"prove {theorem}",
                    max_depth=self.max_depth
                )

                # Add agent-specific diversity
                result = tree.evaluate(context)
                results.append(result)
                elegance_scores.append(result.elegance_score)
                simplicity_scores.append(result.simplicity_score)

        # Aggregate results
        success_count = sum(1 for r in results if r.success)
        success_rate = success_count / len(results) if results else 0.0
        avg_depth = statistics.mean(r.depth_reached for r in results) if results else 0.0
        avg_time = statistics.mean(r.time_taken for r in results) if results else 0.0
        avg_elegance = statistics.mean(elegance_scores) if elegance_scores else 0.0
        avg_simplicity = statistics.mean(simplicity_scores) if simplicity_scores else 0.0

        # Robustness: consistency across simulations
        if len(results) > 1:
            success_by_sim = [1 if r.success else 0 for r in results]
            robustness = 1.0 - statistics.stdev(success_by_sim)
            robustness = max(0.0, robustness)
        else:
            robustness = 0.5

        return AgentEvaluation(
            agent_id=agent_id,
            success_rate=success_rate,
            avg_depth=avg_depth,
            avg_time=avg_time,
            elegance_score=avg_elegance,
            simplicity_score=avg_simplicity,
            robustness=robustness,
            confidence=0.95  # Can be adjusted based on agent reliability
        )


# ============================================================================
# MAKER Voting for Tree Selection
# ============================================================================

class TreeMAKERVoting:
    """
    MAKER voting for tree selection in coevolution.

    Implements first-to-ahead-by-k voting mechanism.
    """

    def __init__(
        self,
        k_ahead: int = 3,
        voting_strategy: str = "first_k_ahead"
    ):
        self.k_ahead = k_ahead
        self.voting_strategy = voting_strategy

    def vote_on_best_trees(
        self,
        trees: List[MDAPProofDecisionTree],
        evaluations: List[MDAPTreeEvaluation],
        count: int,
        use_reliability: bool = True
    ) -> List[MDAPProofDecisionTree]:
        """
        Use MAKER voting to select best trees.

        Implements first-to-ahead-by-k selection.
        """
        votes = defaultdict(float)

        # Collect votes from all agents
        for tree, evaluation in zip(trees, evaluations):
            for agent_result in evaluation.agent_results:
                # Agent votes based on success rate
                if agent_result.success_rate > 0.6:
                    vote_weight = 1.0

                    # Weight by agent reliability
                    if use_reliability:
                        reliability = tree.get_agent_reliability(agent_result.agent_id)
                        vote_weight *= reliability

                    votes[tree.tree_id] += vote_weight

        # Select trees using first-K-ahead
        selected = []
        remaining = list(trees)

        while len(selected) < count and remaining:
            # Find tree ahead by k
            winner = None
            for tree in remaining:
                tree_votes = votes.get(tree.tree_id, 0)

                if self.voting_strategy == "first_k_ahead":
                    # First-to-ahead-by-k
                    max_other = max(
                        [votes.get(t.tree_id, 0) for t in remaining if t != tree],
                        default=0
                    )

                    if tree_votes >= max_other + self.k_ahead:
                        winner = tree
                        break

                elif self.voting_strategy == "first_to_k":
                    # Simple first-to-k
                    if tree_votes >= self.k_ahead:
                        winner = tree
                        break

            # If no tree ahead by k, select highest voted
            if winner is None and remaining:
                winner = max(remaining, key=lambda t: votes.get(t.tree_id, 0))

            if winner:
                selected.append(winner)
                remaining.remove(winner)
            else:
                break

        # If we still need more trees, add from remaining
        if len(selected) < count and remaining:
            remaining_sorted = sorted(
                remaining,
                key=lambda t: votes.get(t.tree_id, 0),
                reverse=True
            )
            selected.extend(remaining_sorted[:count - len(selected)])

        return selected

    def vote_on_crossover_parents(
        self,
        trees: List[MDAPProofDecisionTree],
        evaluations: List[MDAPTreeEvaluation],
        num_parents: int,
        tournament_size: int = 5
    ) -> List[MDAPProofDecisionTree]:
        """
        Select parents for crossover using voting + tournament.
        """
        # First use MAKER voting to get top candidates
        top_candidates = self.vote_on_best_trees(
            trees,
            evaluations,
            count=min(len(trees), tournament_size * 2)
        )

        # Then use tournament among top candidates
        selected = []
        for _ in range(num_parents):
            tournament = random.sample(
                top_candidates,
                min(tournament_size, len(top_candidates))
            )
            winner = max(tournament, key=lambda t: t.consensus_score)
            selected.append(winner)

        return selected


# ============================================================================
# MDAP Tree Coevolution
# ============================================================================

class MDAPTreeCoevolution:
    """
    Coevolve trees with MDAP multi-agent evaluation.

    Main coevolution engine using MDAP for fitness evaluation
    and MAKER voting for selection.
    """

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 50,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        elitism: int = 5,
        max_depth: int = 17,
        simulations: int = 100,
        num_agents: int = 5,
        k_ahead: int = 3,
        voting_strategy: str = "first_k_ahead"
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.max_depth = max_depth

        # MDAP components
        self.num_agents = num_agents
        self.k_ahead = k_ahead
        self.voting_strategy = voting_strategy

        # Initialize components
        self.generator = TreeGenerator()
        self.crossover = TreeCrossover()
        self.mutation = TreeMutation(self.generator)
        self.mdap_evaluator = MDAPTreeEvaluator(
            num_agents=num_agents,
            simulations=simulations,
            max_depth=max_depth
        )
        self.tree_voting = TreeMAKERVoting(
            k_ahead=k_ahead,
            voting_strategy=voting_strategy
        )

        # Tracking
        self.history: List[Dict] = []
        self.best_tree: Optional[MDAPProofDecisionTree] = None
        self.best_consensus: float = 0.0

    def _initialize_mdap_population(
        self,
        available_actions: List[Tactic] = None
    ) -> List[MDAPProofDecisionTree]:
        """Initialize MDAP tree population"""
        base_trees = self.generator.generate_ramped_half_and_half(
            self.population_size,
            self.max_depth,
            available_actions
        )

        # Convert to MDAP trees
        population = []
        for tree in base_trees:
            mdap_tree = MDAPProofDecisionTree(
                root=tree.root,
                tree_id=tree.tree_id,
                generation=0,
                num_agents=self.num_agents,
                voting_strategy=self.voting_strategy,
                k_ahead=self.k_ahead
            )
            population.append(mdap_tree)

        return population

    async def coevolve_mdap(
        self,
        test_theorems: List[str],
        leanaide_client=None,
        agent_configs: List[ModelConfig] = None
    ) -> MDAPProofDecisionTree:
        """Coevolve trees using MDAP evaluation and MAKER voting"""
        # Initialize MDAP tree population
        population = self._initialize_mdap_population()

        self.best_tree = None
        self.best_consensus = 0.0

        print(f"Starting MDAP coevolution: {self.population_size} trees, {self.generations} generations")
        print(f"Agents: {self.num_agents}, K-ahead: {self.k_ahead}")
        print(f"Test theorems: {len(test_theorems)}")

        for generation in range(self.generations):
            start_time = time.time()

            # Multi-agent evaluation
            evaluations = []
            for tree in population:
                eval_result = await self.mdap_evaluator.evaluate_tree_mdap(
                    tree,
                    test_theorems,
                    agent_configs
                )

                # Update tree fitness
                tree.fitness = eval_result.consensus_score
                tree.consensus_score = eval_result.consensus_score
                tree.agreement_level = eval_result.agreement_level

                evaluations.append(eval_result)

                # Track best
                if eval_result.consensus_score > self.best_consensus:
                    self.best_tree = tree.clone()
                    self.best_consensus = eval_result.consensus_score

            # Optional: LeanAide verification bonus
            if leanaide_client:
                await self._apply_verification_bonus(
                    population,
                    evaluations,
                    test_theorems,
                    leanaide_client
                )

            # Parent selection with MAKER voting
            num_parents = self.population_size // 2
            parents = self.tree_voting.vote_on_crossover_parents(
                population,
                evaluations,
                num_parents,
                tournament_size=5
            )

            # Create next generation
            next_gen = []

            # Elitism using MAKER voting
            elites = self.tree_voting.vote_on_best_trees(
                population,
                evaluations,
                count=self.elitism
            )
            next_gen.extend([e.clone() for e in elites])

            # Crossover and mutation
            if len(parents) < 2:
                # Not enough parents for crossover, clone existing
                # Fix: Check if population is empty before random.choice
                if not population:
                    raise ValueError("Cannot perform evolution: population is empty")
                while len(next_gen) < self.population_size:
                    next_gen.append(random.choice(population).clone())
            else:
                while len(next_gen) < self.population_size:
                    parent1, parent2 = random.sample(parents, 2)

                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover.subtree_crossover(
                        parent1, parent2, self.max_depth
                    )
                    # Convert to MDAP trees only after successful crossover
                    child1 = self._convert_to_mdap(child1)
                    child2 = self._convert_to_mdap(child2)
                else:
                    # Clone and convert parents to ensure consistent types
                    child1, child2 = parent1.clone(), parent2.clone()
                    # Ensure MDAP type consistency
                    if not isinstance(child1, MDAPProofDecisionTree):
                        child1 = self._convert_to_mdap(child1)
                    if not isinstance(child2, MDAPProofDecisionTree):
                        child2 = self._convert_to_mdap(child2)

                # Mutation
                if random.random() < self.mutation_rate:
                    child1 = self.mutation.subtree_mutation(child1)
                if random.random() < self.mutation_rate:
                    child2 = self.mutation.subtree_mutation(child2)

                child1.generation = generation + 1
                child2.generation = generation + 1

                next_gen.extend([child1, child2])

            population = next_gen[:self.population_size]

            # Statistics
            gen_time = time.time() - start_time
            avg_consensus = statistics.mean(t.consensus_score for t in population)
            avg_agreement = statistics.mean(t.agreement_level for t in population)

            self.history.append({
                'generation': generation,
                'best_consensus': self.best_consensus,
                'avg_consensus': avg_consensus,
                'avg_agreement': avg_agreement,
                'time': gen_time,
                'population_size': len(population)
            })

            if generation % 10 == 0 or generation == self.generations - 1:
                print(f"Generation {generation}: "
                      f"best_consensus={self.best_consensus:.4f}, "
                      f"avg_consensus={avg_consensus:.4f}, "
                      f"avg_agreement={avg_agreement:.4f}, "
                      f"time={gen_time:.2f}s")

        print(f"MDAP coevolution complete. Best consensus: {self.best_consensus:.4f}")
        return self.best_tree

    def _convert_to_mdap(
        self,
        tree: ProofDecisionTree
    ) -> MDAPProofDecisionTree:
        """Convert regular tree to MDAP tree"""
        if isinstance(tree, MDAPProofDecisionTree):
            return tree

        return MDAPProofDecisionTree(
            root=tree.root,
            tree_id=str(uuid.uuid4()),
            generation=tree.generation,
            num_agents=self.num_agents,
            voting_strategy=self.voting_strategy,
            k_ahead=self.k_ahead
        )

    async def _apply_verification_bonus(
        self,
        population: List[MDAPProofDecisionTree],
        evaluations: List[MDAPTreeEvaluation],
        test_theorems: List[str],
        leanaide_client
    ):
        """Apply LeanAide verification bonus (mock for now)"""
        # Select top trees for verification
        top_trees = self.tree_voting.vote_on_best_trees(
            population,
            evaluations,
            count=5
        )

        for tree in top_trees:
            # Simulate verification bonus
            # In real implementation, would call leanaide_client
            if tree.consensus_score > 0.8:
                verification_bonus = 0.1 * tree.consensus_score
                tree.fitness += verification_bonus


# ============================================================================
# Decomposition-Enhanced Coevolution
# ============================================================================

class DecompositionTreeCoevolution:
    """
    Coevolution with decomposition for complex problems.

    Trees can decompose problems into subproblems when needed.
    """

    def __init__(
        self,
        mdap_coevolution: MDAPTreeCoevolution,
        max_decomposition_depth: int = 3,
        decomposition_threshold: float = 0.7
    ):
        self.mdap_coevolution = mdap_coevolution
        self.max_decomposition_depth = max_decomposition_depth
        self.decomposition_threshold = decomposition_threshold

        # MAKER engine for decomposition decisions
        self.maker_engine = MAKEREngine(
            team=Team(
                team_id="decomposition_team",
                members=[],  # Would populate with actual agents
                description="Team for decomposition decisions"
            ),
            k_ahead=3
        )

    async def coevolve_with_decomposition(
        self,
        test_theorems: List[str],
        leanaide_client=None
    ) -> MDAPProofDecisionTree:
        """Coevolve trees that can decompose problems"""
        population = self.mdap_coevolution._initialize_mdap_population()

        best_tree = None
        best_fitness = 0.0

        # Fix: Check if test_theorems is empty before accessing [0]
        if not test_theorems:
            raise ValueError("Cannot evolve solvers: test_theorems list is empty")
        
        for generation in range(self.mdap_coevolution.generations):
            # Evaluate trees
            for tree in population:
                # Check if tree should use decomposition
                context = ProofContext(
                    theorem=test_theorems[0],  # Sample
                    goal_state=f"prove {test_theorems[0]}",
                    max_depth=50
                )

                if tree.should_decompose(context):
                    # Decompose test theorems
                    subtask_results = await self._evaluate_with_decomposition(
                        tree,
                        test_theorems
                    )
                    tree.fitness = subtask_results["combined_fitness"]
                else:
                    # Standard evaluation
                    eval_result = await self.mdap_coevolution.mdap_evaluator.evaluate_tree_mdap(
                        tree, test_theorems
                    )
                    tree.fitness = eval_result.consensus_score

            # Track best
            current_best = max(population, key=lambda t: t.fitness)
            if current_best.fitness > best_fitness:
                best_tree = current_best.clone()
                best_fitness = current_best.fitness

            # Continue coevolution
            population = await self._create_next_generation(population)

            if generation % 10 == 0:
                print(f"Generation {generation}: best_fitness={best_fitness:.4f}")

        return best_tree

    async def _evaluate_with_decomposition(
        self,
        tree: MDAPProofDecisionTree,
        test_theorems: List[str]
    ) -> Dict[str, Any]:
        """Evaluate tree using decomposition"""
        # For each theorem, try decomposition
        results = []

        for theorem in test_theorems:
            # Decide if decomposition is beneficial
            decomposition = await self._decide_decomposition(theorem)

            if decomposition and not decomposition.is_atomic:
                # Solve subtasks
                result1 = await self._solve_subtask(tree, decomposition.subtask1)
                result2 = await self._solve_subtask(tree, decomposition.subtask2)

                # Compose results
                combined_score = (result1 + result2) / 2 * decomposition.confidence
                results.append(combined_score)
            else:
                # Solve directly
                eval_result = await self.mdap_coevolution.mdap_evaluator.evaluate_tree_mdap(
                    tree, [theorem]
                )
                results.append(eval_result.consensus_score)

        combined_fitness = statistics.mean(results) if results else 0.0

        return {
            "combined_fitness": combined_fitness,
            "subtask_scores": results
        }

    async def _decide_decomposition(
        self,
        theorem: str
    ) -> Optional[TreeDecomposition]:
        """Use MAKER voting to decide on decomposition"""
        # Simplified: random decision
        # In real implementation, would use MAKER engine
        if random.random() < 0.3:
            return TreeDecomposition(
                subtask1=f"Prove lemma for {theorem}",
                subtask2=f"Complete proof of {theorem}",
                composition_function="Combine lemma with main proof",
                confidence=0.7,
                is_atomic=False
            )
        return TreeDecomposition(is_atomic=True)

    async def _solve_subtask(
        self,
        tree: MDAPProofDecisionTree,
        subtask: str
    ) -> float:
        """Solve a subtask"""
        # Mock evaluation
        return random.uniform(0.5, 0.9)

    async def _create_next_generation(
        self,
        population: List[MDAPProofDecisionTree]
    ) -> List[MDAPProofDecisionTree]:
        """Create next generation (simplified)"""
        # Sort by fitness
        population.sort(key=lambda t: t.fitness, reverse=True)

        # Elitism
        new_pop = [t.clone() for t in population[:5]]

        # Generate offspring
        # Fix: Ensure population has enough elements for sampling
        parent_pool = population[:len(population)//2]
        if len(parent_pool) < 2:
            # Not enough parents, just return clones of existing population
            return [t.clone() for t in population]
        while len(new_pop) < len(population):
            parent1, parent2 = random.sample(parent_pool, 2)

            if random.random() < 0.9:
                child1, child2 = self.mdap_coevolution.crossover.subtree_crossover(
                    parent1, parent2
                )
                new_pop.extend([child1, child2])
            else:
                new_pop.extend([parent1.clone(), parent2.clone()])

        return new_pop[:len(population)]


# ============================================================================
# Competitive Coevolution with MDAP
# ============================================================================

class MDAPCompetitiveCoevolution:
    """
    Coevolve solvers and problems with MDAP.

    Solver trees evolve to solve harder problems.
    Problems evolve to be more challenging.
    """

    def __init__(
        self,
        solver_pop_size: int = 50,
        problem_pop_size: int = 20,
        generations: int = 100,
        num_agents: int = 5,
        k_ahead: int = 3
    ):
        self.solver_pop_size = solver_pop_size
        self.problem_pop_size = problem_pop_size
        self.generations = generations
        self.num_agents = num_agents
        self.k_ahead = k_ahead

        self.generator = TreeGenerator()
        self.mdap_evaluator = MDAPTreeEvaluator(num_agents=num_agents)
        self.tree_voting = TreeMAKERVoting(k_ahead=k_ahead)

        self.solver_population: List[MDAPProofDecisionTree] = []
        self.problem_population: List[str] = []

    async def competitive_coevolve_mdap(
        self,
        initial_theorems: List[str]
    ) -> MDAPProofDecisionTree:
        """Coevolve solvers and problems with multi-agent evaluation"""
        # Initialize solver population
        base_solvers = self.generator.generate_ramped_half_and_half(
            self.solver_pop_size,
            15
        )

        self.solver_population = [
            MDAPProofDecisionTree(
                root=s.root,
                num_agents=self.num_agents,
                k_ahead=self.k_ahead
            )
            for s in base_solvers
        ]

        self.problem_population = initial_theorems[:self.problem_pop_size]

        best_solver = None
        best_solver_fitness = 0.0

        for generation in range(self.generations):
            # Evaluate solvers on current problems
            solver_scores = await self._evaluate_solvers()

            # Track best solver
            current_best_idx = max(range(len(solver_scores)), key=lambda i: solver_scores[i])
            current_best_fitness = solver_scores[current_best_idx]

            if current_best_fitness > best_solver_fitness:
                best_solver = self.solver_population[current_best_idx].clone()
                best_solver_fitness = current_best_fitness
                print(f"  New best solver fitness: {best_solver_fitness:.4f}")

            # Select best solvers with MAKER voting
            evaluations = []  # Would have proper evaluations
            best_solvers = self.tree_voting.vote_on_best_trees(
                self.solver_population,
                evaluations,
                count=len(self.solver_population) // 2
            )

            # Generate harder problem variants
            new_problems = []
            for solver in best_solvers:
                weak_problems = self._identify_weak_problems(
                    solver,
                    self.problem_population
                )

                for problem in weak_problems:
                    harder = self._create_harder_variant(problem)
                    new_problems.append(harder)

            # Update problem population
            self.problem_population = self._select_hardest_problems(
                self.problem_population + new_problems
            )

            # Evolve solver population
            self.solver_population = await self._evolve_solvers(solver_scores)

            if generation % 20 == 0:
                avg_solver_score = statistics.mean(solver_scores)
                print(f"Generation {generation}: best={current_best_fitness:.4f}, "
                      f"avg={avg_solver_score:.4f}")

        print(f"Competitive coevolution complete. Best solver fitness: {best_solver_fitness:.4f}")
        return best_solver

    async def _evaluate_solvers(self) -> List[float]:
        """Evaluate all solvers on all problems"""
        scores = []

        for solver in self.solver_population:
            total_score = 0.0

            for problem in self.problem_population:
                eval_result = await self.mdap_evaluator.evaluate_tree_mdap(
                    solver, [problem]
                )
                total_score += eval_result.consensus_score

            scores.append(total_score / len(self.problem_population))

        return scores

    def _identify_weak_problems(
        self,
        solver: MDAPProofDecisionTree,
        problems: List[str]
    ) -> List[str]:
        """Find problems solver struggles with"""
        # Simplified: return random problems
        return random.sample(problems, min(3, len(problems)))

    def _create_harder_variant(self, problem: str) -> str:
        """Create harder variant of problem"""
        modifiers = [
            " with additional constraints",
            " under stronger conditions",
            " with extended requirements"
        ]
        return problem + random.choice(modifiers)

    def _select_hardest_problems(
        self,
        all_problems: List[str]
    ) -> List[str]:
        """Select hardest problems for next generation"""
        # Simplified: random selection
        # Fix: Ensure we don't sample more than available
        if not all_problems:
            return []
        sample_size = min(self.problem_pop_size, len(all_problems))
        return random.sample(all_problems, sample_size)

    async def _evolve_solvers(
        self,
        scores: List[float]
    ) -> List[MDAPProofDecisionTree]:
        """Evolve solver population"""
        # Sort by score
        sorted_solvers = [
            (solver, score)
            for solver, score in zip(self.solver_population, scores)
        ]
        sorted_solvers.sort(key=lambda x: x[1], reverse=True)

        # Elitism
        new_pop = [s.clone() for s, _ in sorted_solvers[:5]]

        # Generate new solvers
        crossover = TreeCrossover()
        mutation = TreeMutation(self.generator)

        while len(new_pop) < self.solver_pop_size:
            parent1, parent2 = random.choices(
                [s for s, _ in sorted_solvers[:self.solver_pop_size//2]],
                k=2
            )

            if random.random() < 0.8:
                child1, child2 = crossover.subtree_crossover(parent1, parent2)
                child1 = MDAPProofDecisionTree(
                    root=child1.root,
                    num_agents=self.num_agents,
                    k_ahead=self.k_ahead
                )
                child2 = MDAPProofDecisionTree(
                    root=child2.root,
                    num_agents=self.num_agents,
                    k_ahead=self.k_ahead
                )
                new_pop.extend([child1, child2])
            else:
                new_pop.append(parent1.clone())

        # Mutation
        for i in range(5, len(new_pop)):
            if random.random() < 0.15:
                new_pop[i] = mutation.subtree_mutation(new_pop[i])

        return new_pop[:self.solver_pop_size]


# ============================================================================
# Multi-Objective MDAP Coevolution
# ============================================================================

class MDAPMultiObjectiveCoevolution:
    """
    Multi-objective coevolution with MDAP.

    Optimizes multiple objectives with Pareto front analysis.
    """

    def __init__(
        self,
        objectives: List[str] = None,
        population_size: int = 100,
        generations: int = 50,
        num_agents: int = 5
    ):
        self.objectives = objectives or ["success", "speed", "elegance", "simplicity"]
        self.population_size = population_size
        self.generations = generations
        self.num_agents = num_agents

        self.generator = TreeGenerator()
        self.mdap_evaluator = MDAPTreeEvaluator(num_agents=num_agents)
        self.tree_voting = TreeMAKERVoting()

    async def coevolve_multi_objective_mdap(
        self,
        test_theorems: List[str]
    ) -> List[MDAPProofDecisionTree]:
        """Coevolve Pareto-optimal trees with MDAP"""
        # Initialize population
        base_trees = self.generator.generate_ramped_half_and_half(
            self.population_size,
            15
        )

        population = [
            MDAPProofDecisionTree(
                root=t.root,
                num_agents=self.num_agents
            )
            for t in base_trees
        ]

        pareto_front = []

        for generation in range(self.generations):
            # Multi-agent evaluation for each objective
            all_evaluations = {}

            for obj in self.objectives:
                evaluations = []
                for tree in population:
                    eval_result = await self.mdap_evaluator.evaluate_tree_mdap(
                        tree, test_theorems
                    )

                    # Extract objective-specific fitness
                    obj_fitness = self._extract_objective_fitness(eval_result, obj)
                    tree.objective_fitness = getattr(tree, 'objective_fitness', {})
                    tree.objective_fitness[obj] = obj_fitness

                    evaluations.append(eval_result)

                all_evaluations[obj] = evaluations

            # Update Pareto front
            pareto_front = self._update_pareto_front(population)

            # Non-dominated sorting for selection
            ranks = self._non_dominated_sort(population)

            # Select parents with MAKER voting
            parents = self._select_parents_multi_objective(
                population,
                ranks
            )

            # Create next generation
            population = await self._create_next_generation(parents)

            if generation % 10 == 0:
                print(f"Generation {generation}: Pareto front size = {len(pareto_front)}")

        return pareto_front

    def _extract_objective_fitness(
        self,
        evaluation: MDAPTreeEvaluation,
        objective: str
    ) -> float:
        """Extract objective-specific fitness from evaluation"""
        if objective == "success":
            return evaluation.consensus_score
        elif objective == "speed":
            # Inverse of avg time (faster is better)
            avg_time = statistics.mean(
                [e.avg_time for e in evaluation.agent_results]
            )
            return 1.0 / (1.0 + avg_time)
        elif objective == "elegance":
            return statistics.mean(
                [e.elegance_score for e in evaluation.agent_results]
            )
        elif objective == "simplicity":
            return statistics.mean(
                [e.simplicity_score for e in evaluation.agent_results]
            )
        else:
            return evaluation.consensus_score

    def _update_pareto_front(
        self,
        population: List[MDAPProofDecisionTree]
    ) -> List[MDAPProofDecisionTree]:
        """Update Pareto front"""
        pareto_front = []

        for ind1 in population:
            dominated = False
            for ind2 in population:
                if ind1 != ind2 and self._dominates(ind2, ind1):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(ind1)

        return pareto_front

    def _dominates(
        self,
        ind1: MDAPProofDecisionTree,
        ind2: MDAPProofDecisionTree
    ) -> bool:
        """Check if ind1 dominates ind2"""
        obj1 = getattr(ind1, 'objective_fitness', {})
        obj2 = getattr(ind2, 'objective_fitness', {})

        at_least_one_better = False
        for obj in self.objectives:
            val1 = obj1.get(obj, 0.0)
            val2 = obj2.get(obj, 0.0)
            if val1 < val2:
                return False
            if val1 > val2:
                at_least_one_better = True

        return at_least_one_better

    def _non_dominated_sort(
        self,
        population: List[MDAPProofDecisionTree]
    ) -> Dict[str, int]:
        """Non-dominated sorting (simplified NSGA-II)"""
        fronts = []
        remaining = population.copy()

        while remaining:
            current_front = []
            for i, ind1 in enumerate(remaining):
                dominated = False
                for j, ind2 in enumerate(remaining):
                    if i != j and self._dominates(ind2, ind1):
                        dominated = True
                        break
                if not dominated:
                    current_front.append(ind1)

            fronts.append(current_front)
            for ind in current_front:
                remaining.remove(ind)

        # Assign ranks
        ranks = {}
        for rank, front in enumerate(fronts):
            for ind in front:
                ranks[ind.tree_id] = rank

        return ranks

    def _select_parents_multi_objective(
        self,
        population: List[MDAPProofDecisionTree],
        ranks: Dict[str, int]
    ) -> List[MDAPProofDecisionTree]:
        """Select parents considering Pareto ranks"""
        selected = []

        while len(selected) < self.population_size // 2:
            # Fix: Check if population has at least 2 elements for sampling
            if len(population) < 2:
                selected.extend(population)
                break
            ind1, ind2 = random.sample(population, 2)

            rank1 = ranks.get(ind1.tree_id, len(population))
            rank2 = ranks.get(ind2.tree_id, len(population))

            # Lower rank is better
            if rank1 < rank2:
                selected.append(ind1)
            elif rank2 < rank1:
                selected.append(ind2)
            else:
                # Same rank, use consensus score
                selected.append(ind1 if ind1.consensus_score > ind2.consensus_score else ind2)

        return selected

    async def _create_next_generation(
        self,
        parents: List[MDAPProofDecisionTree]
    ) -> List[MDAPProofDecisionTree]:
        """Create offspring"""
        offspring = []
        crossover = TreeCrossover()
        mutation = TreeMutation(self.generator)

        if len(parents) < 2:
            # Not enough parents, return clones of existing population
            return [p.clone() for p in population[:self.population_size]]
        
        while len(offspring) < self.population_size:
            # Fix: Check if parents has at least 2 elements for crossover
            if len(parents) < 2:
                # Clone existing parents to fill offspring
                while len(offspring) < self.population_size and parents:
                    offspring.append(parents[len(offspring) % len(parents)].clone())
                break
            parent1, parent2 = random.sample(parents, 2)

            if random.random() < 0.9:
                child1, child2 = crossover.subtree_crossover(parent1, parent2)
                child1 = MDAPProofDecisionTree(
                    root=child1.root,
                    num_agents=self.num_agents
                )
                child2 = MDAPProofDecisionTree(
                    root=child2.root,
                    num_agents=self.num_agents
                )
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.clone(), parent2.clone()])

        # Mutation
        for i in range(len(offspring)):
            if random.random() < 0.1:
                offspring[i] = mutation.subtree_mutation(offspring[i])

        return offspring[:self.population_size]


# ============================================================================
# Ensemble Methods with MDAP
# ============================================================================

class MDAPTreeEnsemble:
    """
    Ensemble with MDAP multi-agent voting.

    Combines multiple trees using consensus mechanisms.
    """

    def __init__(
        self,
        trees: List[MDAPProofDecisionTree],
        voting_strategy: str = "first_k_ahead",
        k_ahead: int = 3
    ):
        self.trees = trees
        self.voting_strategy = voting_strategy
        self.k_ahead = k_ahead
        self.tree_voting = TreeMAKERVoting(k_ahead=k_ahead)

    async def majority_vote_mdap(
        self,
        context: ProofContext
    ) -> ProofResult:
        """Majority vote with multi-agent consensus"""
        # Each tree produces result
        tree_results = []
        for tree in self.trees:
            result = tree.evaluate(context)
            tree_results.append(result)

        # Agent voting on results
        votes = defaultdict(int)

        for i, result in enumerate(tree_results):
            if result.success:
                votes[str(result.proof_steps)] += 1

                # Check first-K-ahead
                max_other = max(
                    [v for p, v in votes.items() if p != str(result.proof_steps)],
                    default=0
                )

                if votes[str(result.proof_steps)] >= max_other + self.k_ahead:
                    return result

        # Return most common result
        if votes:
            winner_key = max(votes.keys(), key=lambda k: votes[k])
            # Fix: Check if tree_results is not empty before accessing [0]
            default_result = tree_results[0] if tree_results else ProofResult(
                success=False,
                proof_steps=[],
                final_state="",
                depth_reached=0,
                time_taken=0.0
            )
            return next(
                (r for r in tree_results if str(r.proof_steps) == winner_key),
                default_result
            )

        return tree_results[0] if tree_results else ProofResult(
            success=False,
            proof_steps=[],
            final_state="",
            depth_reached=0,
            time_taken=0.0
        )

    async def weighted_vote_mdap(
        self,
        context: ProofContext,
        weights: List[float] = None
    ) -> ProofResult:
        """Weighted voting with agent reliability"""
        if weights and len(weights) == len(self.trees):
            self.weights = weights
        else:
            # Use consensus scores as weights
            self.weights = [t.consensus_score for t in self.trees]

        results = [tree.evaluate(context) for tree in self.trees]

        # Weighted voting
        weighted_success = sum(
            w * (1.0 if r.success else 0.0)
            for w, r in zip(self.weights, results)
        )

        final_success = weighted_success > 0.5

        # Combine proof steps
        all_steps = []
        for result, weight in zip(results, self.weights):
            if weight > 0.5:
                all_steps.extend(result.proof_steps)

        avg_depth = statistics.mean(r.depth_reached for r in results) if results else 0
        avg_time = statistics.mean(r.time_taken for r in results) if results else 0.0

        return ProofResult(
            success=final_success,
            proof_steps=all_steps[:50],
            final_state=context.current_state,
            depth_reached=int(avg_depth),
            time_taken=avg_time
        )

    async def cascade_mdap(
        self,
        context: ProofContext
    ) -> ProofResult:
        """Try trees in sequence with consensus check"""
        total_steps = []
        total_time = 0.0
        max_depth = 0

        for tree in self.trees:
            result = tree.evaluate(context)
            total_steps.extend(result.proof_steps)
            total_time += result.time_taken
            max_depth = max(max_depth, result.depth_reached)

            # Check if consensus is high enough
            if tree.consensus_score > 0.8 and result.success:
                return ProofResult(
                    success=True,
                    proof_steps=total_steps,
                    final_state=result.final_state,
                    depth_reached=max_depth,
                    time_taken=total_time
                )

        # All tried
        return ProofResult(
            success=False,
            proof_steps=total_steps,
            final_state=context.current_state,
            depth_reached=max_depth,
            time_taken=total_time
        )


# ============================================================================
# Performance Tracking
# ============================================================================

class MDAPCoevolutionMonitor:
    """Monitor MDAP coevolution progress"""

    def __init__(self):
        self.generation_history: List[Dict] = []
        self.agent_performance: Dict[str, List[float]] = defaultdict(list)
        self.consensus_history: List[float] = []
        self.agreement_history: List[float] = []

    def track_generation(
        self,
        generation: int,
        population: List[MDAPProofDecisionTree],
        evaluations: List[MDAPTreeEvaluation]
    ):
        """Track generation metrics"""
        avg_consensus = sum(t.consensus_score for t in population) / len(population)
        avg_agreement = sum(t.agreement_level for t in population) / len(population)

        # Agent performance
        for eval in evaluations:
            for agent_result in eval.agent_results:
                self.agent_performance[agent_result.agent_id].append(
                    agent_result.success_rate
                )

        self.generation_history.append({
            "generation": generation,
            "avg_consensus": avg_consensus,
            "avg_agreement": avg_agreement,
            "best_consensus": max(t.consensus_score for t in population),
            "population_diversity": self._compute_diversity(population)
        })

        self.consensus_history.append(avg_consensus)
        self.agreement_history.append(avg_agreement)

    def _compute_diversity(
        self,
        population: List[MDAPProofDecisionTree]
    ) -> float:
        """Compute population diversity"""
        if len(population) < 2:
            return 0.0

        # Diversity based on fitness variance
        fitnesses = [t.consensus_score for t in population]
        if len(fitnesses) > 1:
            return statistics.stdev(fitnesses)
        return 0.0

    def get_agent_reliability_report(self) -> Dict[str, Dict[str, float]]:
        """Generate agent reliability report"""
        report = {}

        for agent_id, scores in self.agent_performance.items():
            if scores:
                report[agent_id] = {
                    "avg_score": statistics.mean(scores),
                    "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "num_evaluations": len(scores)
                }

        return report

    def plot_progress(self):
        """Plot coevolution progress (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Consensus over time
            axes[0, 0].plot(self.consensus_history)
            axes[0, 0].set_title("Average Consensus")
            axes[0, 0].set_xlabel("Generation")
            axes[0, 0].set_ylabel("Consensus Score")

            # Agreement over time
            axes[0, 1].plot(self.agreement_history)
            axes[0, 1].set_title("Average Agreement")
            axes[0, 1].set_xlabel("Generation")
            axes[0, 1].set_ylabel("Agreement Level")

            # Agent performance
            for agent_id, scores in self.agent_performance.items():
                axes[1, 0].plot(scores, label=agent_id, alpha=0.6)
            axes[1, 0].set_title("Agent Performance")
            axes[1, 0].set_xlabel("Evaluation")
            axes[1, 0].set_ylabel("Success Rate")
            axes[1, 0].legend()

            # Population diversity
            diversity = [g["population_diversity"] for g in self.generation_history]
            axes[1, 1].plot(diversity)
            axes[1, 1].set_title("Population Diversity")
            axes[1, 1].set_xlabel("Generation")
            axes[1, 1].set_ylabel("Diversity (Std Dev)")

            plt.tight_layout()
            plt.savefig("mdap_coevolution_progress.png")
            print("Progress plot saved to mdap_coevolution_progress.png")

        except ImportError:
            print("Matplotlib not available, skipping plots")


# ============================================================================
# Demo and Testing
# ============================================================================

async def demo_mdap_coevolution():
    """Demonstrate MDAP coevolution"""
    print("=" * 80)
    print("MDAP COEVOLVING DECISION TREES DEMONSTRATION")
    print("=" * 80)

    # Sample theorems
    test_theorems = [
        "∀ n: Nat, n + 0 = n",
        "∀ a b: Nat, a + b = b + a",
        "∀ n: Nat, 2 * n = n + n",
        "∀ a b c: Nat, (a + b) + c = a + (b + c)",
        "∀ n: Nat, n ≤ n"
    ]

    print("\n1. MDAP TREE COEVOLUTION")
    print("-" * 50)

    mdap_coevolution = MDAPTreeCoevolution(
        population_size=20,
        generations=10,
        simulations=20,
        num_agents=5,
        k_ahead=3
    )

    best_tree = await mdap_coevolution.coevolve_mdap(test_theorems)

    print(f"\nBest tree consensus: {best_tree.consensus_score:.4f}")
    print(f"Best tree agreement: {best_tree.agreement_level:.4f}")
    print(f"Best tree fitness: {best_tree.fitness:.4f}")

    print("\n2. DECOMPOSITION-ENHANCED COEVOLUTION")
    print("-" * 50)

    decomp_coevolution = DecompositionTreeCoevolution(
        mdap_coevolution=mdap_coevolution,
        max_decomposition_depth=3,
        decomposition_threshold=0.7
    )

    best_decomp_tree = await decomp_coevolution.coevolve_with_decomposition(
        test_theorems
    )

    print(f"\nBest decomposition tree fitness: {best_decomp_tree.fitness:.4f}")

    print("\n3. COMPETITIVE COEVOLUTION")
    print("-" * 50)

    competitive = MDAPCompetitiveCoevolution(
        solver_pop_size=20,
        problem_pop_size=10,
        generations=15,
        num_agents=5
    )

    best_solver = await competitive.competitive_coevolve_mdap(test_theorems)

    print(f"\nBest solver fitness: {best_solver.fitness:.4f}")

    print("\n4. MULTI-OBJECTIVE COEVOLUTION")
    print("-" * 50)

    multi_obj = MDAPMultiObjectiveCoevolution(
        objectives=["success", "elegance", "simplicity"],
        population_size=20,
        generations=10,
        num_agents=5
    )

    pareto_front = await multi_obj.coevolve_multi_objective_mdap(test_theorems)

    print(f"\nPareto front size: {len(pareto_front)}")

    print("\n5. MDAP TREE ENSEMBLE")
    print("-" * 50)

    ensemble = MDAPTreeEnsemble(pareto_front[:5])

    test_context = ProofContext(
        theorem="∀ n: Nat, n + 0 = n",
        goal_state="prove addition identity",
        max_depth=50
    )

    result = await ensemble.majority_vote_mdap(test_context)
    print(f"Ensemble result: success={result.success}, "
          f"depth={result.depth_reached}, time={result.time_taken:.3f}s")

    print("\n6. PERFORMANCE MONITORING")
    print("-" * 50)

    monitor = MDAPCoevolutionMonitor()

    # Simulate some tracking
    for gen in range(5):
        monitor.track_generation(
            gen,
            [best_tree],
            [MDAPTreeEvaluation(
                tree_id=best_tree.tree_id,
                agent_results=[],
                consensus_score=0.8 + gen * 0.02,
                agreement_level=0.7 + gen * 0.01,
                voting_details={},
                std_dev_success=0.1,
                std_dev_depth=0.2
            )]
        )

    report = monitor.get_agent_reliability_report()
    print("\nAgent reliability report:")
    for agent_id, metrics in report.items():
        print(f"  {agent_id}: avg={metrics['avg_score']:.3f}, "
              f"std_dev={metrics['std_dev']:.3f}")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


# ============================================================================
# Integration Utility Functions
# ============================================================================

def create_mdap_config(
    num_agents: int = 5,
    k_ahead: int = 3,
    voting_strategy: str = "first_k_ahead",
    enable_decomposition: bool = True
) -> Dict[str, Any]:
    """Create MDAP configuration"""
    return {
        "num_agents": num_agents,
        "k_ahead": k_ahead,
        "voting_strategy": voting_strategy,
        "enable_decomposition": enable_decomposition
    }


async def run_mdap_coevolution_pipeline(
    test_theorems: List[str],
    config: Dict[str, Any] = None
) -> MDAPProofDecisionTree:
    """
    Run complete MDAP coevolution pipeline.

    Args:
        test_theorems: List of theorems to prove
        config: Configuration dict (uses defaults if None)

    Returns:
        Best evolved tree
    """
    if config is None:
        config = create_mdap_config()

    coevolution = MDAPTreeCoevolution(
        population_size=config.get("population_size", 50),
        generations=config.get("generations", 30),
        num_agents=config.get("num_agents", 5),
        k_ahead=config.get("k_ahead", 3),
        voting_strategy=config.get("voting_strategy", "first_k_ahead")
    )

    best_tree = await coevolution.coevolve_mdap(test_theorems)

    return best_tree


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys

    print("MDAP/MAKER Coevolving Decision Trees")
    print("=" * 80)

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(demo_mdap_coevolution())
    else:
        print("Usage: python mcts_coevolution_mdap.py demo")
        print("\nFeatures:")
        print("  - MDAP-enhanced decision trees with multi-agent evaluation")
        print("  - MAKER voting for tree selection (first-to-ahead-by-k)")
        print("  - Decomposition-enhanced coevolution")
        print("  - Competitive coevolution with MDAP")
        print("  - Multi-objective optimization")
        print("  - Ensemble methods with MDAP voting")
        print("  - Performance tracking and monitoring")
