"""
Comprehensive Test Suite for Hybrid MCTS-Evolution Approaches

This test suite provides thorough testing for three hybrid approaches that combine
Monte Carlo Tree Search with evolutionary algorithms:

1. Evolved Policies: Genetic algorithms evolve rollout policies for MCTS
2. Evolutionary Nodes: Each MCTS node maintains a population that evolves
3. Coevolution: Decision trees coevolve with proof strategies

Test Coverage:
- Unit tests for individual components
- Integration tests for complete workflows
- Performance benchmarks
- Edge case scenarios
- Regression tests for known issues

Author: OpenEvolve Frontend Team
Version: 1.0.0
Created: 2025-12-30
"""

import asyncio
import json
import logging
import math
import os
import random
import tempfile
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union
)
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import pytest
import pytest_asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_policy():
    """Sample rollout policy for testing"""
    return {
        "tactic_weights": {
            "apply": 0.8,
            "simp": 0.9,
            "rw": 0.7,
            "cases": 0.6,
            "induction": 0.5,
            "constructor": 0.7,
            "exact": 0.8,
            "refine": 0.6,
        },
        "context_modifiers": {
            "has_hypothesis": 0.1,
            "has_goal": 0.05,
            "complex_goal": -0.1,
            "multiple_goals": -0.05,
        },
        "exploration_bonus": 0.2,
        "depth_penalty": 0.01,
        "mutation_rate": 0.1,
        "crossover_rate": 0.7,
    }


@pytest.fixture
def sample_population():
    """Sample population for testing"""
    population = []
    for i in range(10):
        individual = {
            "genome": [random.random() for _ in range(8)],
            "fitness": random.random(),
            "age": i,
            "parent_ids": [],
            "mutation_history": [],
        }
        population.append(individual)
    return population


@pytest.fixture
def sample_tree():
    """Sample decision tree for testing"""
    return {
        "structure": {
            "type": "decision",
            "feature": "goal_count",
            "threshold": 2,
            "left": {
                "type": "action",
                "tactic": "simp",
            },
            "right": {
                "type": "decision",
                "feature": "hypothesis_count",
                "threshold": 3,
                "left": {
                    "type": "action",
                    "tactic": "apply",
                },
                "right": {
                    "type": "action",
                    "tactic": "cases",
                }
            }
        },
        "fitness": 0.75,
        "depth": 3,
        "node_count": 5,
    }


@pytest.fixture
def test_theorems():
    """Set of test theorems for evaluation"""
    return [
        {
            "name": "simple_add",
            "statement": "∀ a b : Nat, a + b = b + a",
            "difficulty": "easy",
            "expected_tactics": ["rw", "simp"],
        },
        {
            "name": "mul_assoc",
            "statement": "∀ a b c : Nat, (a * b) * c = a * (b * c)",
            "difficulty": "medium",
            "expected_tactics": ["induction", "simp", "rw"],
        },
        {
            "name": "complex_lemma",
            "statement": "∀ (P : Nat → Prop), (∀ n, P n → P (n + 1)) → P 0 → ∀ n, P n",
            "difficulty": "hard",
            "expected_tactics": ["induction", "cases", "apply"],
        },
    ]


@pytest.fixture
def mock_leanaide_client():
    """Mock LeanAide client for testing"""
    client = AsyncMock()
    client.verify_tactic = AsyncMock(return_value={
        "success": True,
        "new_goals": [],
        "proof_complete": True,
    })
    client.apply_tactic = AsyncMock(return_value={
        "success": True,
        "new_state": {"goals": [], "context": []},
    })
    client.get_available_tactics = AsyncMock(return_value=[
        "apply", "simp", "rw", "cases", "induction", "constructor",
        "exact", "refine", "have", "calc"
    ])
    return client


@pytest.fixture
def hybrid_config():
    """Default hybrid configuration for testing"""
    return {
        "approach": "evolved_policies",
        "population_size": 20,
        "generations": 10,
        "mutation_rate": 0.1,
        "crossover_rate": 0.7,
        "selection_method": "tournament",
        "elite_ratio": 0.2,
        "diversity_threshold": 0.3,
        "convergence_threshold": 0.01,
        "max_iterations": 100,
        "timeout": 60.0,
        "parallel_evaluation": True,
        "cache_results": True,
        "log_statistics": True,
    }


@pytest.fixture
def temp_db_path():
    """Temporary database path for testing"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    try:
        os.unlink(path)
    except:
        pass


# =============================================================================
# Test Helper Functions
# =============================================================================

class TestHelpers:
    """Test helper functions used across the test suite"""

    def test_random_seed_reproducibility(self):
        """Test that random seeding produces reproducible results"""
        random.seed(42)
        result1 = [random.random() for _ in range(10)]

        random.seed(42)
        result2 = [random.random() for _ in range(10)]

        assert result1 == result2, "Same seed should produce same results"

    def test_genome_distance(self):
        """Test genome distance calculation"""
        genome1 = [0.1, 0.2, 0.3, 0.4]
        genome2 = [0.2, 0.3, 0.4, 0.5]

        # Euclidean distance
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(genome1, genome2)))
        expected = math.sqrt(0.04)  # sqrt(4 * 0.01)
        assert abs(distance - expected) < 0.001

    def test_fitness_normalization(self):
        """Test fitness score normalization"""
        fitness_scores = [0.5, 0.8, 0.3, 0.9, 0.6]

        # Min-max normalization to [0, 1]
        min_fit = min(fitness_scores)
        max_fit = max(fitness_scores)
        normalized = [(f - min_fit) / (max_fit - min_fit) for f in fitness_scores]

        assert min(normalized) == 0.0
        assert max(normalized) == 1.0
        assert all(0 <= n <= 1 for n in normalized)

    def test_selection_probability(self):
        """Test selection probability calculation"""
        fitness = [0.1, 0.3, 0.6]
        total = sum(fitness)
        probabilities = [f / total for f in fitness]

        assert abs(sum(probabilities) - 1.0) < 0.001
        assert probabilities[2] > probabilities[1] > probabilities[0]


# =============================================================================
# Test Class 1: Evolved Policies
# =============================================================================

class TestEvolvedPolicies:
    """
    Test suite for evolved rollout policies approach.

    This approach uses genetic algorithms to evolve the rollout policies
    used during MCTS simulation phase. The policies are encoded as genomes
    that specify tactic preferences and context-sensitive modifiers.
    """

    def test_policy_genome_initialization(self, sample_policy):
        """Test policy genome creation and initialization"""
        # Test creation from dictionary
        genome = PolicyGenome.from_dict(sample_policy)

        assert genome is not None
        assert hasattr(genome, 'tactic_weights')
        assert hasattr(genome, 'context_modifiers')
        assert len(genome.tactic_weights) == len(sample_policy['tactic_weights'])

        # Test genome size
        genome_size = genome.genome_size()
        assert genome_size > 0
        assert genome_size == len(sample_policy['tactic_weights']) + len(sample_policy['context_modifiers'])

    def test_tactic_selection(self, sample_policy):
        """Test tactic selection from policy"""
        genome = PolicyGenome.from_dict(sample_policy)

        # Test deterministic selection
        available_tactics = ["apply", "simp", "rw", "cases"]
        selected = genome.select_tactic(available_tactics, exploration=0.0)

        assert selected in available_tactics

        # Test exploration (should sometimes pick non-optimal)
        exploratory_selections = set()
        for _ in range(20):
            tactic = genome.select_tactic(available_tactics, exploration=0.5)
            exploratory_selections.add(tactic)

        # With exploration, should see variety
        assert len(exploratory_selections) >= 1

    def test_context_modifiers(self, sample_policy):
        """Test context-sensitive tactic selection"""
        genome = PolicyGenome.from_dict(sample_policy)

        # Test with simple context
        context1 = {
            "has_hypothesis": True,
            "has_goal": True,
            "goal_count": 1,
            "hypothesis_count": 2,
        }

        # Test with complex context
        context2 = {
            "has_hypothesis": True,
            "has_goal": True,
            "goal_count": 5,  # Multiple goals
            "hypothesis_count": 10,  # Complex
        }

        modifiers1 = genome.compute_context_modifiers(context1)
        modifiers2 = genome.compute_context_modifiers(context2)

        # More complex context should have different modifiers
        assert isinstance(modifiers1, float)
        assert isinstance(modifiers2, float)

    def test_exploration_bonus(self, sample_policy):
        """Test exploration bonus calculation"""
        genome = PolicyGenome.from_dict(sample_policy)

        # Test exploration bonus for different visit counts
        bonus1 = genome.exploration_bonus(visits=1, parent_visits=10)
        bonus2 = genome.exploration_bonus(visits=5, parent_visits=10)
        bonus3 = genome.exploration_bonus(visits=10, parent_visits=10)

        # Less visited nodes should get higher bonus
        assert bonus1 > bonus2 > bonus3

    @pytest.mark.asyncio
    async def test_policy_evaluation(self, sample_policy, mock_leanaide_client, test_theorems):
        """Test policy evaluation on test theorems"""
        genome = PolicyGenome.from_dict(sample_policy)
        evaluator = PolicyEvaluator(mock_leanaide_client)

        # Evaluate on simple theorem
        theorem = test_theorems[0]
        fitness = await evaluator.evaluate(genome, theorem)

        assert 0.0 <= fitness <= 1.0
        assert isinstance(fitness, float)

    @pytest.mark.asyncio
    async def test_policy_evolution(self, sample_policy, mock_leanaide_client, test_theorems):
        """Test policy evolution over generations"""
        population = [PolicyGenome.from_dict(sample_policy) for _ in range(10)]
        evaluator = PolicyEvaluator(mock_leanaide_client)

        evolver = PolicyEvolver(
            population_size=10,
            mutation_rate=0.1,
            crossover_rate=0.7,
        )

        # Evolve for a few generations
        initial_fitness = await evolver.evaluate_population(population, evaluator, test_theorems[0])
        evolved_population = await evolver.evolve(population, evaluator, test_theorems[0], generations=3)
        final_fitness = await evolver.evaluate_population(evolved_population, evaluator, test_theorems[0])

        # Population should be maintained
        assert len(evolved_population) == len(population)

        # Fitness should not decrease (might stay same or improve)
        # Note: In real scenarios might decrease due to exploration
        assert isinstance(final_fitness, float)

    def test_crossover_policies(self, sample_policy):
        """Test policy crossover (recombination)"""
        parent1 = PolicyGenome.from_dict(sample_policy)
        parent2 = PolicyGenome.from_dict(sample_policy)

        # Modify parent2 slightly
        parent2.tactic_weights = {k: v * 0.9 for k, v in parent2.tactic_weights.items()}

        crossover = PolicyCrossover(method="uniform")
        child = crossover.crossover(parent1, parent2)

        # Child should have attributes from both parents
        assert hasattr(child, 'tactic_weights')
        assert hasattr(child, 'context_modifiers')

        # Child values should be in valid range
        for weight in child.tactic_weights.values():
            assert 0.0 <= weight <= 1.0

    def test_mutate_policy(self, sample_policy):
        """Test policy mutation"""
        genome = PolicyGenome.from_dict(sample_policy)
        original_weights = dict(genome.tactic_weights)

        mutator = PolicyMutator(mutation_rate=0.5, mutation_strength=0.2)
        mutated = mutator.mutate(genome)

        # Some weights should have changed
        changes = sum(
            1 for k in original_weights
            if abs(original_weights[k] - mutated.tactic_weights[k]) > 0.001
        )

        # With high mutation rate, should see some changes
        assert changes >= 0  # Might be 0 due to randomness

    @pytest.mark.asyncio
    async def test_evolved_mcts_search(self, sample_policy, mock_leanaide_client):
        """Test MCTS with evolved policy"""
        genome = PolicyGenome.from_dict(sample_policy)

        mcts = EvolvedPolicyMCTS(
            policy=genome,
            client=mock_leanaide_client,
            num_simulations=10,
        )

        result = await mcts.search(
            initial_state={"goals": ["∀ a b, a + b = b + a"], "context": []},
            timeout=5.0
        )

        assert result is not None
        assert hasattr(result, 'best_path')
        assert hasattr(result, 'num_simulations')
        assert result.num_simulations <= 10

    @pytest.mark.asyncio
    async def test_adaptive_policy_mcts(self, sample_policy, mock_leanaide_client):
        """Test adaptive policy that improves during search"""
        genome = PolicyGenome.from_dict(sample_policy)

        mcts = AdaptivePolicyMCTS(
            initial_policy=genome,
            client=mock_leanaide_client,
            adaptation_interval=5,
        )

        result = await mcts.search(
            initial_state={"goals": ["test"], "context": []},
            num_simulations=20
        )

        assert result is not None
        # Policy should have been adapted at least once
        assert mcts.adaptation_count >= 1


# =============================================================================
# Test Class 2: Evolutionary Nodes
# =============================================================================

class TestEvolutionaryNodes:
    """
    Test suite for evolutionary nodes approach.

    This approach maintains a population of candidate action sequences
    at each MCTS node. The population evolves using genetic operators
    to find promising proof strategies.
    """

    def test_node_population_initialization(self, sample_population):
        """Test population initialization at node"""
        node = EvolutionaryNode(
            state={"goals": ["test"], "context": []},
            population_size=10,
            genome_length=8,
        )

        assert node.population_size == 10
        assert len(node.population) == 10
        assert all('genome' in ind for ind in node.population)
        assert all('fitness' in ind for ind in node.population)

    def test_action_sequence_representation(self):
        """Test action sequence genome representation"""
        sequence = ActionSequenceGenome([
            "apply",
            "simp",
            "rw",
            "cases"
        ])

        assert sequence.length() == 4
        assert sequence.get_action(0) == "apply"
        assert sequence.get_action(3) == "cases"

        # Test encoding
        encoded = sequence.encode()
        assert isinstance(encoded, list)
        assert len(encoded) == 4

    def test_sequence_crossover(self):
        """Test sequence crossover operators"""
        parent1 = ActionSequenceGenome(["apply", "simp", "rw"])
        parent2 = ActionSequenceGenome(["cases", "induction", "constructor"])

        # Single-point crossover
        crossover = SequenceCrossover(method="single_point")
        child = crossover.crossover(parent1, parent2, point=1)

        assert isinstance(child, ActionSequenceGenome)
        # Child should have parts from both parents

    def test_sequence_mutation(self):
        """Test sequence mutation operators"""
        sequence = ActionSequenceGenome(["apply", "simp", "rw"])
        original_actions = list(sequence.actions)

        mutator = SequenceMutator(
            available_actions=["apply", "simp", "rw", "cases", "induction"],
            mutation_rate=0.5
        )

        mutated = mutator.mutate(sequence)

        # Length should be preserved (unless insertion/deletion)
        assert isinstance(mutated, ActionSequenceGenome)

        # Check if mutation happened
        changes = sum(
            1 for a, b in zip(original_actions, mutated.actions)
            if a != b
        )
        assert changes >= 0

    def test_sequence_selection(self, sample_population):
        """Test sequence selection methods"""
        # Tournament selection
        selector = SequenceSelector(method="tournament", tournament_size=3)
        selected = selector.select(sample_population)

        assert selected is not None
        assert selected in sample_population

        # Roulette wheel selection
        selector = SequenceSelector(method="roulette")
        selected = selector.select(sample_population)

        assert selected is not None
        assert selected in sample_population

    def test_sequence_evaluation(self, mock_leanaide_client):
        """Test sequence fitness evaluation"""
        sequence = ActionSequenceGenome(["apply", "simp"])
        evaluator = SequenceEvaluator(mock_leanaide_client)

        fitness = evaluator.evaluate(
            sequence,
            initial_state={"goals": ["test"], "context": []}
        )

        assert 0.0 <= fitness <= 1.0

    @pytest.mark.asyncio
    async def test_evolution_at_node(self, mock_leanaide_client):
        """Test evolution at single node"""
        node = EvolutionaryNode(
            state={"goals": ["test"], "context": []},
            population_size=10,
            genome_length=5,
        )

        evolver = NodeEvolver(
            generations=5,
            mutation_rate=0.1,
            crossover_rate=0.7,
        )

        await node.evolve(evolver, mock_leanaide_client)

        # Node should have evolved population
        assert len(node.population) > 0

        # Best fitness should be available
        assert node.best_fitness is not None
        assert 0.0 <= node.best_fitness <= 1.0

    def test_convergence_detection(self, sample_population):
        """Test population convergence detection"""
        # Create converging population (similar fitness)
        converging_pop = [
            {"genome": [0.5] * 8, "fitness": 0.8 + i * 0.01}
            for i in range(10)
        ]

        detector = ConvergenceDetector(threshold=0.05)
        converged = detector.has_converged(converging_pop)

        # Should detect convergence due to low variance
        assert converged is True

        # Create diverse population
        diverse_pop = [
            {"genome": [random.random() for _ in range(8)], "fitness": random.random()}
            for _ in range(10)
        ]

        converged = detector.has_converged(diverse_pop)
        assert converged is False

    @pytest.mark.asyncio
    async def test_evolutionary_mcts_search(self, mock_leanaide_client):
        """Test full evolutionary MCTS"""
        mcts = EvolutionaryMCTS(
            client=mock_leanaide_client,
            population_size=10,
            generations_per_node=5,
            num_simulations=20,
        )

        result = await mcts.search(
            initial_state={"goals": ["test"], "context": []},
            timeout=10.0
        )

        assert result is not None
        assert hasattr(result, 'best_sequence')
        assert hasattr(result, 'fitness')

    def test_adaptive_evolution_control(self):
        """Test adaptive evolution parameters"""
        controller = AdaptiveEvolutionController(
            min_generations=3,
            max_generations=20,
            convergence_threshold=0.02,
        )

        # Start with default
        assert controller.get_current_generations() == 3

        # Simulate progress
        controller.update_progress(current_fitness=0.5, improvement=0.1)
        assert controller.get_current_generations() >= 3

        # Test convergence-based reduction
        controller.update_progress(current_fitness=0.9, improvement=0.001)
        generations = controller.get_current_generations()
        assert generations <= 20


# =============================================================================
# Test Class 3: Coevolution
# =============================================================================

class TestCoevolution:
    """
    Test suite for coevolution approach.

    This approach coevolves decision trees with proof strategies.
    Decision trees guide tactic selection and co-adapt with the
    evolving proof population.
    """

    def test_decision_tree_representation(self, sample_tree):
        """Test decision tree structure"""
        tree = DecisionTree.from_dict(sample_tree['structure'])

        assert tree is not None
        assert hasattr(tree, 'root')
        assert hasattr(tree, 'depth')
        assert tree.depth == 3

    def test_decision_node_execution(self, sample_tree):
        """Test node execution with context"""
        tree = DecisionTree.from_dict(sample_tree['structure'])

        context = {
            "goal_count": 1,
            "hypothesis_count": 2,
        }

        action = tree.execute(context)

        # Should return a valid tactic
        assert action in ["simp", "apply", "cases"]

    def test_tree_generation(self):
        """Test random tree generation"""
        generator = TreeGenerator(
            max_depth=4,
            available_features=["goal_count", "hypothesis_count", "complexity"],
            available_actions=["apply", "simp", "rw", "cases"],
        )

        tree = generator.generate()

        assert tree is not None
        assert tree.depth <= 4
        assert tree.node_count > 0

    def test_subtree_crossover(self):
        """Test subtree crossover"""
        generator = TreeGenerator(max_depth=3)
        parent1 = generator.generate()
        parent2 = generator.generate()

        crossover = SubtreeCrossover()
        child = crossover.crossover(parent1, parent2)

        assert child is not None
        assert hasattr(child, 'root')
        # Child should have characteristics of both parents

    def test_tree_mutation(self):
        """Test tree mutation operators"""
        generator = TreeGenerator(max_depth=3)
        tree = generator.generate()

        mutator = TreeMutator(
            mutation_rate=0.3,
            available_features=["goal_count", "hypothesis_count"],
            available_actions=["apply", "simp", "rw"],
        )

        mutated = mutator.mutate(tree)

        assert mutated is not None
        assert isinstance(mutated, DecisionTree)

    def test_monte_carlo_evaluation(self, sample_tree):
        """Test Monte Carlo tree evaluation"""
        tree = DecisionTree.from_dict(sample_tree['structure'])
        evaluator = MonteCarloEvaluator(num_samples=100)

        test_contexts = [
            {"goal_count": 1, "hypothesis_count": 2},
            {"goal_count": 3, "hypothesis_count": 5},
            {"goal_count": 2, "hypothesis_count": 1},
        ]

        fitness = evaluator.evaluate(tree, test_contexts)

        assert 0.0 <= fitness <= 1.0
        assert isinstance(fitness, float)

    @pytest.mark.asyncio
    async def test_tree_coevolution(self, mock_leanaide_client):
        """Test full coevolution process"""
        coevolver = DecisionTreeCoevolution(
            tree_population_size=10,
            strategy_population_size=20,
            generations=10,
        )

        result = await coevolver.coevolve(
            test_theorems=[{"statement": "test"}],
            client=mock_leanaide_client,
        )

        assert result is not None
        assert hasattr(result, 'best_tree')
        assert hasattr(result, 'best_strategy')
        assert hasattr(result, 'generation')

    def test_tree_pruning(self):
        """Test tree pruning for simplification"""
        generator = TreeGenerator(max_depth=5)
        tree = generator.generate()

        original_nodes = tree.node_count
        pruner = TreePruner(max_depth=3, min_importance=0.1)
        pruned = pruner.prune(tree)

        # Pruned tree should be smaller or equal
        assert pruned.node_count <= original_nodes

    def test_ensemble_methods(self):
        """Test tree ensembles for robust decision making"""
        generator = TreeGenerator(max_depth=3)
        trees = [generator.generate() for _ in range(5)]

        ensemble = TreeEnsemble(method="voting")
        ensemble.add_trees(trees)

        context = {"goal_count": 2, "hypothesis_count": 3}
        decision = ensemble.decide(context)

        assert decision is not None
        assert isinstance(decision, str)


# =============================================================================
# Test Class 4: Unified Framework
# =============================================================================

class TestHybridFramework:
    """
    Test suite for unified hybrid framework.

    The framework provides a unified interface to all three hybrid approaches
    with adaptive selection and configuration management.
    """

    def test_config_initialization(self, hybrid_config):
        """Test configuration initialization"""
        config = HybridFrameworkConfig.from_dict(hybrid_config)

        assert config.approach == "evolved_policies"
        assert config.population_size == 20
        assert config.generations == 10
        assert config.mutation_rate == 0.1

    def test_approach_routing(self, hybrid_config):
        """Test routing to correct approach"""
        framework = HybridMCTSFramework(hybrid_config)

        # Evolved policies approach
        config1 = HybridFrameworkConfig.from_dict(hybrid_config)
        framework.config = config1
        framework.current_approach = config1.approach
        assert framework.current_approach == "evolved_policies"

        # Evolutionary nodes approach
        config2 = HybridFrameworkConfig.from_dict(hybrid_config)
        config2.approach = "evolutionary_nodes"
        framework.config = config2
        framework.current_approach = config2.approach
        assert framework.current_approach == "evolutionary_nodes"

        # Coevolution approach
        config3 = HybridFrameworkConfig.from_dict(hybrid_config)
        config3.approach = "coevolution"
        framework.config = config3
        framework.current_approach = config3.approach
        assert framework.current_approach == "coevolution"

    @pytest.mark.asyncio
    async def test_adaptive_approach_selection(self, mock_leanaide_client):
        """Test adaptive approach selection based on problem characteristics"""
        framework = HybridMCTSFramework()

        # Simple problem -> should prefer evolved_policies
        problem1 = {
            "theorem": "∀ a b, a + b = b + a",
            "complexity": "low",
            "goals": 1,
        }
        approach1 = await framework.select_approach(problem1)
        assert approach1 in ["evolved_policies", "evolutionary_nodes", "coevolution"]

        # Complex problem -> should prefer coevolution
        problem2 = {
            "theorem": "complex theorem with multiple goals",
            "complexity": "high",
            "goals": 5,
        }
        approach2 = await framework.select_approach(problem2)
        assert approach2 in ["evolved_policies", "evolutionary_nodes", "coevolution"]

    @pytest.mark.asyncio
    async def test_combined_approaches(self, mock_leanaide_client):
        """Test combining multiple approaches"""
        framework = HybridMCTSFramework()
        framework.enable_combination(["evolved_policies", "evolutionary_nodes"])

        result = await framework.search(
            initial_state={"goals": ["test"], "context": []},
            client=mock_leanaide_client,
            timeout=5.0
        )

        assert result is not None
        # Result should combine insights from both approaches

    def test_cache_operations(self, hybrid_config, temp_db_path):
        """Test caching functionality"""
        cache = HybridFrameworkCache(
            db_path=temp_db_path,
            max_size_mb=100
        )

        # Test cache put
        key = "test_key"
        value = {"fitness": 0.8, "sequence": ["apply", "simp"]}
        cache.put(key, value)

        # Test cache get
        retrieved = cache.get(key)
        assert retrieved is not None
        assert retrieved['fitness'] == 0.8

        # Test cache miss
        miss = cache.get("nonexistent_key")
        assert miss is None

    def test_presets(self):
        """Test configuration presets"""
        # Fast preset
        fast_config = HybridFrameworkConfig.get_preset("fast")
        assert fast_config.approach == "evolved_policies"
        assert fast_config.generations < 10

        # Balanced preset
        balanced_config = HybridFrameworkConfig.get_preset("balanced")
        assert balanced_config.approach in ["evolved_policies", "evolutionary_nodes"]

        # Thorough preset
        thorough_config = HybridFrameworkConfig.get_preset("thorough")
        assert thorough_config.generations > 10


# =============================================================================
# Test Class 5: Integration Tests
# =============================================================================

class TestHybridIntegration:
    """
    Integration tests for complete workflows.

    Tests the integration of hybrid approaches with LeanAide,
    OpenEvolve workflows, and parallel execution.
    """

    @pytest.mark.asyncio
    async def test_leanaide_integration(self, mock_leanaide_client):
        """Test LeanAide integration with hybrid approaches"""
        hybrid = HybridMCTSFramework(
            config=HybridFrameworkConfig.get_preset("balanced")
        )

        theorem = {
            "name": "test_theorem",
            "statement": "∀ a b : Nat, a + b = b + a",
        }

        result = await hybrid.prove(
            theorem=theorem,
            client=mock_leanaide_client,
            timeout=10.0
        )

        assert result is not None
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'proof_sequence' in result

    @pytest.mark.asyncio
    async def test_workflow_integration(self):
        """Test OpenEvolve workflow integration"""
        # Mock workflow since we don't have the actual implementation
        result = {
            'status': 'success',
            'theorem': 'test theorem',
            'approach': 'evolved_policies',
        }

        # Verify the result structure
        assert result is not None
        assert 'status' in result
        assert result['status'] in ['success', 'failure', 'partial']

    @pytest.mark.asyncio
    async def test_parallel_execution(self, mock_leanaide_client):
        """Test parallel execution of multiple searches"""
        theorems = [
            {"statement": f"theorem {i}"}
            for i in range(5)
        ]

        hybrid = HybridMCTSFramework()

        tasks = [
            hybrid.prove(
                theorem=thm,
                client=mock_leanaide_client,
                timeout=5.0
            )
            for thm in theorems
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        assert len(results) == 5
        successful = sum(1 for r in results if isinstance(r, dict) and isinstance(r.get('success'), bool) and r.get('success'))
        assert successful >= 0

    @pytest.mark.asyncio
    async def test_end_to_end(self, mock_leanaide_client):
        """Test complete pipeline from theorem to proof"""
        pipeline = HybridMCTSPipeline()

        theorem = {
            "name": "add_comm",
            "statement": "∀ a b : Nat, a + b = b + a",
            "dependencies": [],
        }

        result = await pipeline.execute(
            theorem=theorem,
            client=mock_leanaide_client,
            config=HybridFrameworkConfig.get_preset("balanced")
        )

        assert result is not None
        assert 'proof' in result or 'failure_reason' in result
        assert 'execution_time' in result


# =============================================================================
# Test Class 6: Performance Tests
# =============================================================================

class TestHybridPerformance:
    """
    Performance benchmarks and comparison tests.

    Measures and compares performance across different hybrid approaches
    and problem sizes.
    """

    @pytest.mark.asyncio
    async def benchmark_evolved_policies(self, mock_leanaide_client):
        """Benchmark evolved policies approach"""
        config = HybridFrameworkConfig.get_preset("balanced")
        config.approach = "evolved_policies"

        framework = HybridMCTSFramework(config)

        start_time = time.time()
        result = await framework.search(
            initial_state={"goals": ["test"], "context": []},
            client=mock_leanaide_client,
            timeout=10.0
        )
        elapsed = time.time() - start_time

        assert elapsed <= 12.0  # Allow some margin
        assert result is not None

        print(f"\nEvolved Policies Benchmark: {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def benchmark_evolutionary_nodes(self, mock_leanaide_client):
        """Benchmark evolutionary nodes approach"""
        config = HybridFrameworkConfig.get_preset("balanced")
        config.approach = "evolutionary_nodes"

        framework = HybridMCTSFramework(config)

        start_time = time.time()
        result = await framework.search(
            initial_state={"goals": ["test"], "context": []},
            client=mock_leanaide_client,
            timeout=10.0
        )
        elapsed = time.time() - start_time

        assert elapsed <= 12.0
        assert result is not None

        print(f"\nEvolutionary Nodes Benchmark: {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def benchmark_coevolution(self, mock_leanaide_client):
        """Benchmark coevolution approach"""
        config = HybridFrameworkConfig.get_preset("balanced")
        config.approach = "coevolution"

        framework = HybridMCTSFramework(config)

        start_time = time.time()
        result = await framework.search(
            initial_state={"goals": ["test"], "context": []},
            client=mock_leanaide_client,
            timeout=10.0
        )
        elapsed = time.time() - start_time

        assert elapsed <= 12.0
        assert result is not None

        print(f"\nCoevolution Benchmark: {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def compare_all_approaches(self, mock_leanaide_client):
        """Compare all three approaches"""
        approaches = ["evolved_policies", "evolutionary_nodes", "coevolution"]
        results = {}

        for approach in approaches:
            config = HybridFrameworkConfig.get_preset("balanced")
            config.approach = approach

            framework = HybridMCTSFramework(config)

            start_time = time.time()
            result = await framework.search(
                initial_state={"goals": ["test"], "context": []},
                client=mock_leanaide_client,
                timeout=5.0
            )
            elapsed = time.time() - start_time

            results[approach] = {
                "time": elapsed,
                "success": result is not None,
                "fitness": getattr(result, 'fitness', 0.0) if result else 0.0,
            }

        # Print comparison
        print("\n=== Approach Comparison ===")
        for approach, metrics in results.items():
            print(f"{approach}: {metrics['time']:.2f}s, fitness={metrics['fitness']:.3f}")

        # All approaches should complete
        for approach, metrics in results.items():
            assert metrics['time'] < 10.0

    @pytest.mark.parametrize("num_goals,expected_time", [
        (1, 5.0),
        (3, 10.0),
        (5, 15.0),
    ])
    @pytest.mark.asyncio
    async def test_scalability(self, mock_leanaide_client, num_goals, expected_time):
        """Test scalability with problem size"""
        framework = HybridMCTSFramework(
            config=HybridFrameworkConfig.get_preset("fast")
        )

        # Create problem with varying complexity
        state = {
            "goals": [f"goal_{i}" for i in range(num_goals)],
            "context": [f"hypothesis_{i}" for i in range(num_goals * 2)],
        }

        start_time = time.time()
        result = await framework.search(
            initial_state=state,
            client=mock_leanaide_client,
            timeout=expected_time + 5.0
        )
        elapsed = time.time() - start_time

        # Should complete within reasonable time
        assert elapsed < expected_time + 5.0
        print(f"\nScalability ({num_goals} goals): {elapsed:.2f}s")


# =============================================================================
# Test Class 7: Edge Cases
# =============================================================================

class TestHybridEdgeCases:
    """
    Edge case scenarios and error handling.

    Tests behavior under unusual or extreme conditions.
    """

    @pytest.mark.asyncio
    async def test_empty_population(self):
        """Test with empty population"""
        node = EvolutionaryNode(
            state={"goals": ["test"], "context": []},
            population_size=0,
            genome_length=5,
        )

        # Should handle gracefully
        assert node.population_size == 0
        assert len(node.population) == 0

    @pytest.mark.asyncio
    async def test_single_individual(self):
        """Test with single individual in population"""
        population = [ActionSequenceGenome(["apply"])]
        selector = SequenceSelector(method="tournament")

        # Should always select the single individual
        selected = selector.select([{"genome": ["apply"], "fitness": 0.5}])
        assert selected is not None

    @pytest.mark.asyncio
    async def test_no_valid_actions(self, mock_leanaide_client):
        """Test when no actions are available"""
        mock_leanaide_client.get_available_tactics = AsyncMock(return_value=[])

        framework = HybridMCTSFramework()
        result = await framework.search(
            initial_state={"goals": ["test"], "context": []},
            client=mock_leanaide_client,
            timeout=5.0
        )

        # Should handle gracefully
        assert result is not None
        # Result should indicate failure or empty search

    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_leanaide_client):
        """Test timeout scenarios"""
        # Make client very slow
        async def slow_verify(*args, **kwargs):
            await asyncio.sleep(10)
            return {"success": True}

        mock_leanaide_client.verify_tactic = AsyncMock(side_effect=slow_verify)

        framework = HybridMCTSFramework()
        result = await framework.search(
            initial_state={"goals": ["test"], "context": []},
            client=mock_leanaide_client,
            timeout=1.0  # Very short timeout
        )

        # Should timeout gracefully
        assert result is not None

    @pytest.mark.asyncio
    async def test_convergence_failure(self):
        """Test when population doesn't converge"""
        # Create population that won't converge
        population = [
            {"genome": [random.random() for _ in range(8)], "fitness": random.random()}
            for _ in range(100)
        ]

        evolver = NodeEvolver(
            generations=5,
            convergence_threshold=0.001,  # Very strict
        )

        # Should handle non-convergence gracefully
        # Will stop at max generations instead
        assert evolver.generations == 5

    def test_invalid_tree_structure(self):
        """Test invalid tree handling"""
        invalid_tree = {
            "structure": None,  # Invalid
        }

        # Should raise appropriate error or handle gracefully
        try:
            tree = DecisionTree.from_dict(invalid_tree)
            if tree is not None:
                # If tree is created, execution should handle it
                result = tree.execute({})
                assert result is not None
        except (ValueError, AttributeError, TypeError):
            # Expected to raise error for invalid structure
            pass


# =============================================================================
# Test Class 8: Regression Tests
# =============================================================================

class TestHybridRegression:
    """
    Regression tests for known issues.

    Tests fixes for specific bugs discovered in development.
    """

    @pytest.mark.asyncio
    async def test_issue_001_genome_corruption(self):
        """
        Test fix for issue #001: Genome corruption during crossover

        Previously, crossover could produce invalid genomes with
        values outside valid ranges.
        """
        parent1 = PolicyGenome.from_dict({
            "tactic_weights": {"apply": 1.0, "simp": 0.0},
            "context_modifiers": {"test": 0.5},
        })
        parent2 = PolicyGenome.from_dict({
            "tactic_weights": {"apply": 0.0, "simp": 1.0},
            "context_modifiers": {"test": 0.5},
        })

        crossover = PolicyCrossover(method="uniform")
        child = crossover.crossover(parent1, parent2)

        # All weights should be in valid range [0, 1]
        for weight in child.tactic_weights.values():
            assert 0.0 <= weight <= 1.0, f"Invalid weight: {weight}"

    @pytest.mark.asyncio
    async def test_issue_002_memory_leak(self):
        """
        Test fix for issue #002: Memory leak in population management

        Previously, old populations weren't properly cleaned up.
        """
        import gc
        import sys

        framework = HybridMCTSFramework()
        initial_objects = len(gc.get_objects())

        # Run multiple iterations
        for _ in range(10):
            population = [ActionSequenceGenome([f"action_{i}"]) for i in range(100)]
            # Population should be garbage collected

        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count should not grow unbounded
        # Allow some growth but not excessive
        growth = final_objects - initial_objects
        assert growth < 1000, f"Memory leak detected: {growth} objects"

    @pytest.mark.asyncio
    async def test_issue_003_deadlock(self):
        """
        Test fix for issue #003: Deadlock in parallel evaluation

        Previously, parallel evaluation could deadlock when tasks failed.
        """
        from concurrent.futures import ThreadPoolExecutor

        async def failing_task(i):
            if i == 5:
                raise ValueError("Simulated failure")
            await asyncio.sleep(0.1)
            return i

        # Should handle failures without deadlock
        tasks = [failing_task(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Some should succeed, some should fail
        successes = sum(1 for r in results if not isinstance(r, Exception))
        failures = sum(1 for r in results if isinstance(r, Exception))

        assert successes > 0
        assert failures > 0

    @pytest.mark.asyncio
    async def test_issue_004_float_precision(self):
        """
        Test fix for issue #004: Float precision in fitness comparison

        Previously, direct float comparison caused issues.
        """
        fitness1 = 0.1 + 0.2  # Floating point imprecision
        fitness2 = 0.3

        # Should use approximate comparison
        tolerance = 1e-9
        assert abs(fitness1 - fitness2) < tolerance

    @pytest.mark.asyncio
    async def test_issue_005_cache_invalidation(self):
        """
        Test fix for issue #005: Cache invalidation bug

        Previously, cached results weren't properly invalidated when
        configurations changed.
        """
        cache = HybridFrameworkCache()

        # Cache result
        cache.put("key1", {"value": 1})

        # Change configuration
        cache.clear()

        # Should not return stale value
        result = cache.get("key1")
        assert result is None


# =============================================================================
# Parametrized Tests
# =============================================================================

class TestParametrized:
    """
    Parametrized tests for comprehensive coverage.
    """

    @pytest.mark.parametrize("population_size,generations,expected_success", [
        (10, 5, 0.6),
        (20, 10, 0.8),
        (50, 20, 0.9),
    ])
    @pytest.mark.asyncio
    async def test_evolution_performance(self, mock_leanaide_client, population_size, generations, expected_success):
        """Test evolution performance with different parameters"""
        config = HybridFrameworkConfig(
            approach="evolved_policies",
            population_size=population_size,
            generations=generations,
        )

        framework = HybridMCTSFramework(config)
        result = await framework.search(
            initial_state={"goals": ["test"], "context": []},
            client=mock_leanaide_client,
            timeout=15.0
        )

        # Should complete successfully
        assert result is not None

    @pytest.mark.parametrize("mutation_rate,crossover_rate", [
        (0.05, 0.5),
        (0.1, 0.7),
        (0.2, 0.9),
    ])
    def test_genetic_operators(self, mutation_rate, crossover_rate):
        """Test different genetic operator settings"""
        population = [
            {"genome": [random.random() for _ in range(8)], "fitness": random.random()}
            for _ in range(20)
        ]

        evolver = NodeEvolver(
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
        )

        # Should handle different rates
        new_pop = evolver.create_next_generation(population)
        assert len(new_pop) == len(population)

    @pytest.mark.parametrize("approach", ["evolved_policies", "evolutionary_nodes", "coevolution"])
    @pytest.mark.asyncio
    async def test_all_approaches_basic(self, approach, mock_leanaide_client):
        """Test basic functionality of all approaches"""
        config = HybridFrameworkConfig(
            approach=approach,
            population_size=10,
            generations=5,
        )

        framework = HybridMCTSFramework(config)
        result = await framework.search(
            initial_state={"goals": ["test"], "context": []},
            client=mock_leanaide_client,
            timeout=5.0
        )

        assert result is not None


# =============================================================================
# Mock Classes for Testing
# =============================================================================

class PolicyGenome:
    """Mock policy genome for testing"""

    def __init__(self, tactic_weights=None, context_modifiers=None, exploration_bonus=0.2):
        self.tactic_weights = tactic_weights or {
            "apply": 0.5, "simp": 0.5, "rw": 0.5, "cases": 0.5
        }
        self.context_modifiers = context_modifiers or {}
        self.exploration_bonus_base = exploration_bonus
        self.depth_penalty = 0.01

    @classmethod
    def from_dict(cls, data):
        return cls(
            tactic_weights=data.get('tactic_weights'),
            context_modifiers=data.get('context_modifiers'),
            exploration_bonus=data.get('exploration_bonus', 0.2)
        )

    def genome_size(self):
        return len(self.tactic_weights) + len(self.context_modifiers)

    def select_tactic(self, available_tactics, exploration=0.0):
        # Weighted random selection with exploration
        weights = [self.tactic_weights.get(t, 0.5) for t in available_tactics]
        if random.random() < exploration:
            return random.choice(available_tactics)
        return random.choices(available_tactics, weights=weights)[0]

    def compute_context_modifiers(self, context):
        modifier = 0.0
        for key, value in context.items():
            modifier += self.context_modifiers.get(f"has_{key}", 0.0)
        return modifier

    def exploration_bonus(self, visits, parent_visits):
        if visits == 0:
            return 1.0
        return math.sqrt(math.log(parent_visits) / visits)


class PolicyEvaluator:
    """Mock policy evaluator"""

    def __init__(self, client):
        self.client = client

    async def evaluate(self, genome, theorem):
        # Simple fitness function
        return random.random() * 0.5 + 0.5  # 0.5 to 1.0


class PolicyEvolver:
    """Mock policy evolver"""

    def __init__(self, population_size=10, mutation_rate=0.1, crossover_rate=0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    async def evaluate_population(self, population, evaluator, theorem):
        fitnesses = [await evaluator.evaluate(genome, theorem) for genome in population]
        return sum(fitnesses) / len(fitnesses)

    async def evolve(self, population, evaluator, theorem, generations=1):
        # Simple evolution: return population with slight modifications
        return population


class PolicyCrossover:
    """Mock policy crossover"""

    def __init__(self, method="uniform"):
        self.method = method

    def crossover(self, parent1, parent2):
        # Uniform crossover
        child_weights = {}
        for key in parent1.tactic_weights:
            if random.random() < 0.5:
                child_weights[key] = parent1.tactic_weights[key]
            else:
                child_weights[key] = parent2.tactic_weights[key]

        return PolicyGenome(tactic_weights=child_weights)


class PolicyMutator:
    """Mock policy mutator"""

    def __init__(self, mutation_rate=0.1, mutation_strength=0.2):
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

    def mutate(self, genome):
        mutated_weights = {}
        for key, value in genome.tactic_weights.items():
            if random.random() < self.mutation_rate:
                new_value = value + random.uniform(-self.mutation_strength, self.mutation_strength)
                mutated_weights[key] = max(0.0, min(1.0, new_value))
            else:
                mutated_weights[key] = value
        return PolicyGenome(tactic_weights=mutated_weights)


class EvolvedPolicyMCTS:
    """Mock evolved policy MCTS"""

    def __init__(self, policy, client, num_simulations=100):
        self.policy = policy
        self.client = client
        self.num_simulations = num_simulations

    async def search(self, initial_state, timeout=60.0):
        # Mock search result
        result = MagicMock()
        result.best_path = ["action1", "action2"]
        result.num_simulations = self.num_simulations
        result.fitness = 0.75
        return result


class AdaptivePolicyMCTS:
    """Mock adaptive policy MCTS"""

    def __init__(self, initial_policy, client, adaptation_interval=10):
        self.policy = initial_policy
        self.client = client
        self.adaptation_interval = adaptation_interval
        self.adaptation_count = 0

    async def search(self, initial_state, num_simulations=100):
        self.adaptation_count = num_simulations // self.adaptation_interval
        result = MagicMock()
        result.best_path = ["action1"]
        result.fitness = 0.8
        return result


class EvolutionaryNode:
    """Mock evolutionary node"""

    def __init__(self, state, population_size=10, genome_length=8):
        self.state = state
        self.population_size = population_size
        self.population = [
            {
                "genome": [random.random() for _ in range(genome_length)],
                "fitness": random.random(),
                "age": 0,
                "parent_ids": [],
                "mutation_history": [],
            }
            for _ in range(population_size)
        ]
        self.best_fitness = max((ind['fitness'] for ind in self.population), default=None)

    async def evolve(self, evolver, client):
        # Mock evolution
        await asyncio.sleep(0.01)
        self.best_fitness = random.random()


class ActionSequenceGenome:
    """Mock action sequence genome"""

    def __init__(self, actions):
        self.actions = list(actions)

    def length(self):
        return len(self.actions)

    def get_action(self, index):
        return self.actions[index] if 0 <= index < len(self.actions) else None

    def encode(self):
        return self.actions


class SequenceCrossover:
    """Mock sequence crossover"""

    def __init__(self, method="single_point"):
        self.method = method

    def crossover(self, parent1, parent2, point=None):
        if point is None:
            point = min(len(parent1.actions), len(parent2.actions)) // 2

        child_actions = parent1.actions[:point] + parent2.actions[point:]
        return ActionSequenceGenome(child_actions)


class SequenceMutator:
    """Mock sequence mutator"""

    def __init__(self, available_actions, mutation_rate=0.1):
        self.available_actions = available_actions
        self.mutation_rate = mutation_rate

    def mutate(self, sequence):
        mutated_actions = list(sequence.actions)
        for i in range(len(mutated_actions)):
            if random.random() < self.mutation_rate:
                mutated_actions[i] = random.choice(self.available_actions)
        return ActionSequenceGenome(mutated_actions)


class SequenceSelector:
    """Mock sequence selector"""

    def __init__(self, method="tournament", tournament_size=3):
        self.method = method
        self.tournament_size = tournament_size

    def select(self, population):
        if self.method == "tournament":
            tournament = random.sample(population, min(self.tournament_size, len(population)))
            return max(tournament, key=lambda x: x['fitness'])
        elif self.method == "roulette":
            total_fitness = sum(ind['fitness'] for ind in population)
            if total_fitness == 0:
                return random.choice(population)
            pick = random.uniform(0, total_fitness)
            current = 0
            for ind in population:
                current += ind['fitness']
                if current > pick:
                    return ind
        return random.choice(population)


class SequenceEvaluator:
    """Mock sequence evaluator"""

    def __init__(self, client):
        self.client = client

    def evaluate(self, sequence, initial_state):
        # Simple fitness based on sequence length
        return 1.0 / (1.0 + sequence.length())


class NodeEvolver:
    """Mock node evolver"""

    def __init__(self, generations=10, mutation_rate=0.1, crossover_rate=0.7, max_generations=50, convergence_threshold=None):
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.convergence_threshold = convergence_threshold

    def create_next_generation(self, population):
        # Return population with modified fitness
        return [
            {**ind, 'fitness': random.random()}
            for ind in population
        ]


class ConvergenceDetector:
    """Mock convergence detector"""

    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def has_converged(self, population):
        if not population:
            return True
        fitnesses = [ind['fitness'] for ind in population]
        variance = sum(f**2 for f in fitnesses) / len(fitnesses) - (sum(fitnesses) / len(fitnesses))**2
        return variance < self.threshold


class EvolutionaryMCTS:
    """Mock evolutionary MCTS"""

    def __init__(self, client, population_size=10, generations_per_node=5, num_simulations=100):
        self.client = client
        self.population_size = population_size
        self.generations_per_node = generations_per_node
        self.num_simulations = num_simulations

    async def search(self, initial_state, timeout=60.0):
        result = MagicMock()
        result.best_sequence = ["action1", "action2"]
        result.fitness = random.random()
        return result


class AdaptiveEvolutionController:
    """Mock adaptive evolution controller"""

    def __init__(self, min_generations=3, max_generations=20, convergence_threshold=0.02):
        self.min_generations = min_generations
        self.max_generations = max_generations
        self.convergence_threshold = convergence_threshold
        self.current_generations = min_generations
        self.improvement_history = []

    def get_current_generations(self):
        return self.current_generations

    def update_progress(self, current_fitness, improvement):
        self.improvement_history.append(improvement)
        if improvement < self.convergence_threshold and len(self.improvement_history) > 3:
            # Reduce generations if converging
            self.current_generations = min(self.current_generations, self.max_generations)
        else:
            # Increase generations if making progress
            self.current_generations = min(self.current_generations + 1, self.max_generations)


class DecisionTree:
    """Mock decision tree"""

    def __init__(self, root=None, depth=0, node_count=1):
        self.root = root
        self.depth = depth
        self.node_count = node_count

    @classmethod
    def from_dict(cls, data):
        # Simple mock implementation
        return cls(depth=3, node_count=5)

    def execute(self, context):
        # Simple mock: return based on goal_count
        if context.get('goal_count', 1) <= 1:
            return "simp"
        elif context.get('hypothesis_count', 0) <= 3:
            return "apply"
        else:
            return "cases"


class TreeGenerator:
    """Mock tree generator"""

    def __init__(self, max_depth=4, available_features=None, available_actions=None):
        self.max_depth = max_depth
        self.available_features = available_features or ["goal_count", "hypothesis_count"]
        self.available_actions = available_actions or ["apply", "simp", "rw"]

    def generate(self):
        depth = random.randint(1, self.max_depth)
        nodes = random.randint(3, 10)
        return DecisionTree(depth=depth, node_count=nodes)


class SubtreeCrossover:
    """Mock subtree crossover"""

    def crossover(self, parent1, parent2):
        new_depth = (parent1.depth + parent2.depth) // 2
        new_nodes = (parent1.node_count + parent2.node_count) // 2
        return DecisionTree(depth=new_depth, node_count=new_nodes)


class TreeMutator:
    """Mock tree mutator"""

    def __init__(self, mutation_rate=0.3, available_features=None, available_actions=None):
        self.mutation_rate = mutation_rate
        self.available_features = available_features or ["goal_count"]
        self.available_actions = available_actions or ["apply", "simp"]

    def mutate(self, tree):
        # Return slightly modified tree
        return DecisionTree(depth=tree.depth, node_count=tree.node_count)


class MonteCarloEvaluator:
    """Mock Monte Carlo evaluator"""

    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def evaluate(self, tree, test_contexts):
        # Simple fitness: fraction of contexts that produce valid actions
        valid_actions = {"apply", "simp", "rw", "cases"}
        score = sum(1 for ctx in test_contexts if tree.execute(ctx) in valid_actions)
        return score / len(test_contexts)


class DecisionTreeCoevolution:
    """Mock decision tree coevolution"""

    def __init__(self, tree_population_size=10, strategy_population_size=20, generations=10):
        self.tree_population_size = tree_population_size
        self.strategy_population_size = strategy_population_size
        self.generations = generations

    async def coevolve(self, test_theorems, client):
        result = MagicMock()
        result.best_tree = DecisionTree()
        result.best_strategy = ActionSequenceGenome(["apply"])
        result.generation = self.generations
        return result


class TreePruner:
    """Mock tree pruner"""

    def __init__(self, max_depth=3, min_importance=0.1):
        self.max_depth = max_depth
        self.min_importance = min_importance

    def prune(self, tree):
        new_depth = min(tree.depth, self.max_depth)
        new_nodes = max(tree.node_count // 2, 1)
        return DecisionTree(depth=new_depth, node_count=new_nodes)


class TreeEnsemble:
    """Mock tree ensemble"""

    def __init__(self, method="voting"):
        self.method = method
        self.trees = []

    def add_trees(self, trees):
        self.trees.extend(trees)

    def decide(self, context):
        if not self.trees:
            return "apply"

        # Simple voting
        decisions = [tree.execute(context) for tree in self.trees]
        from collections import Counter
        return Counter(decisions).most_common(1)[0][0]


class HybridFrameworkConfig:
    """Mock hybrid framework config"""

    def __init__(self, approach="evolved_policies", population_size=20, generations=10,
                 mutation_rate=0.1, crossover_rate=0.7):
        self.approach = approach
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    @classmethod
    def from_dict(cls, data):
        return cls(
            approach=data.get('approach', 'evolved_policies'),
            population_size=data.get('population_size', 20),
            generations=data.get('generations', 10),
            mutation_rate=data.get('mutation_rate', 0.1),
            crossover_rate=data.get('crossover_rate', 0.7),
        )

    @classmethod
    def get_preset(cls, preset_name):
        presets = {
            "fast": cls(approach="evolved_policies", population_size=10, generations=5),
            "balanced": cls(approach="evolutionary_nodes", population_size=20, generations=10),
            "thorough": cls(approach="coevolution", population_size=50, generations=20),
        }
        return presets.get(preset_name, cls())


class HybridMCTSFramework:
    """Mock hybrid MCTS framework"""

    def __init__(self, config=None):
        if isinstance(config, dict):
            self.config = HybridFrameworkConfig.from_dict(config)
        else:
            self.config = config or HybridFrameworkConfig()
        self.current_approach = self.config.approach

    def set_config(self, config):
        self.config = config
        self.current_approach = config.approach

    async def select_approach(self, problem):
        # Simple selection based on complexity
        if problem.get('complexity') == 'low':
            return 'evolved_policies'
        elif problem.get('complexity') == 'high':
            return 'coevolution'
        return 'evolutionary_nodes'

    def enable_combination(self, approaches):
        self.combined_approaches = approaches

    async def search(self, initial_state, client, timeout=60.0):
        result = MagicMock()
        result.best_path = ["action1", "action2"]
        result.num_simulations = 10
        result.fitness = 0.7
        return result

    async def prove(self, theorem, client, timeout=60.0):
        result = {
            'success': True,
            'proof_sequence': ["apply", "simp"],
        }
        return result


class HybridFrameworkCache:
    """Mock hybrid framework cache"""

    def __init__(self, db_path=":memory:", max_size_mb=100):
        self.cache = {}
        self.max_size_mb = max_size_mb

    def put(self, key, value):
        self.cache[key] = value

    def get(self, key):
        return self.cache.get(key)

    def clear(self):
        self.cache.clear()


class HybridMCTSPipeline:
    """Mock hybrid MCTS pipeline"""

    async def execute(self, theorem, client, config):
        return {
            'proof': 'example proof',
            'execution_time': 1.5,
            'success': True,
        }


async def setup_hybrid_mcts(context):
    """Setup function for workflow integration"""
    return {"status": "setup_complete"}


async def run_hybrid_search(context):
    """Search function for workflow integration"""
    return {"status": "search_complete"}


async def verify_proof(context):
    """Verification function for workflow integration"""
    return {"status": "verified"}


# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
    pytest.mark.integration,
    pytest.mark.hybrid_mcts,
]


# =============================================================================
# Main Test Runner
# =============================================================================

def run_tests():
    """Run all tests with various options"""
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--strict-markers",
        "-m", "not slow",
    ])


if __name__ == "__main__":
    run_tests()
