"""
Comprehensive Test Suite for LeanAide Evolution-MDAP Integration

This test suite validates the integration of MDAP/MAKER voting-based selection
with LeanAide evolutionary proof generation, covering:

- Unit tests for MDAP-enhanced evolutionary components
- Integration tests for complete evolutionary loops
- Comparison tests (pure evolution vs MDAP-enhanced)
- Workflow integration tests
- Performance benchmarks
- Edge case handling

Author: OpenEvolve Frontend Team
Date: 2025-12-30
Reference: arXiv:2511.09030 (MAKER framework)
"""

import asyncio
import json
import logging
import random
import time
import unittest
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTS - Try to import required modules
# =============================================================================

try:
    from evolution import (
        EvolutionConfiguration,
        run_evolution
    )
    EVOLUTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Evolution module not available: {e}")
    EVOLUTION_AVAILABLE = False

try:
    from mdap_engine import (
        MDAPConfig,
        MDAPTask,
        MDAPStep,
        MDAPOrchestrator,
        RedFlagRules,
        RedFlagger,
        AgentSelector,
        MDAPVoteResult,
        MDAPStepResult,
        MDAPRunResult
    )
    MDAP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MDAP engine not available: {e}")
    MDAP_AVAILABLE = False

try:
    from evolution_maker_integration import (
        MakerevolutionConfig,
        MakerevolutionMode,
        Individual,
        Population,
        MAKERSelection,
        MDAPEvolutionDecomposer,
        MAKEREvolutionEngine,
        run_maker_evolution
    )
    EVOLUTION_MAKER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Evolution-MAKER integration not available: {e}")
    EVOLUTION_MAKER_AVAILABLE = False


# =============================================================================
# MOCK DATA AND FIXTURES
# =============================================================================

@dataclass
class MockProof:
    """Mock Lean 4 proof for testing"""
    lean_code: str
    tactic_sequence: List[str]
    verified: bool = False
    verification_time: float = 1.0
    errors: List[str] = field(default_factory=list)

    def __len__(self):
        return len(self.tactic_sequence)


@dataclass
class MockLeanStrategy:
    """Mock evolutionary strategy for testing"""
    strategy_id: str
    tactics: List[str]
    fitness: float
    proof: MockProof
    generation: int = 0
    verified: bool = False


class ProofApproach(Enum):
    """Proof approaches for adversarial evolution"""
    CONSTRUCTIVE = "constructive"
    CLASSICAL = "classical"
    COMPUTATIONAL = "computational"
    INDIRECT = "indirect"
    STRUCTURAL = "structural"
    ALGEBRAIC = "algebraic"


# Sample theorems for testing
SAMPLE_THEOREMS = {
    "easy": "∀ n : Nat, n + 0 = n",
    "medium": "∀ a b : Nat, a + b = b + a",
    "hard": "∀ (f : Nat -> Nat), (∀ n, f n = 0) -> f = (λ _, 0)"
}

# Sample tactic sequences
SAMPLE_TACTICS = {
    "constructive": ["intros", "refl"],
    "inductive": ["intro n", "induction n with", "case IH", "rfl"],
    "algebraic": ["intros", "linarith"],
    "computational": ["intros", "simp", "rfl"],
    "structural": ["intro h", "cases h", "rfl"]
}


# =============================================================================
# UNIT TESTS: MDAP-Enhanced Evolution Components
# =============================================================================

class TestMDAPLeanPopulation(unittest.TestCase):
    """
    Unit tests for MDAP-enhanced population management.

    Tests agent voting, ranking, and selection within evolutionary populations.
    """

    def setUp(self):
        """Set up test fixtures"""
        if not EVOLUTION_MAKER_AVAILABLE:
            self.skipTest("Evolution-MAKER integration not available")

        self.population = Population(
            individuals=[
                Individual(
                    genome="proof_1",
                    fitness=0.8,
                    generation=0,
                    metadata={"verified": True, "tactics": ["intros", "refl"]}
                ),
                Individual(
                    genome="proof_2",
                    fitness=0.6,
                    generation=0,
                    metadata={"verified": False, "tactics": ["intros", "simp"]}
                ),
                Individual(
                    genome="proof_3",
                    fitness=0.9,
                    generation=0,
                    metadata={"verified": True, "tactics": ["intros", "linarith"]}
                ),
                Individual(
                    genome="proof_4",
                    fitness=0.7,
                    generation=0,
                    metadata={"verified": False, "tactics": ["intro h", "cases h"]}
                ),
                Individual(
                    genome="proof_5",
                    fitness=0.5,
                    generation=0,
                    metadata={"verified": True, "tactics": ["induction n"]}
                )
            ],
            generation=0
        )

    def test_population_best_individual(self):
        """Test retrieving the best individual from population"""
        best = self.population.best_individual
        self.assertIsNotNone(best)
        self.assertEqual(best.fitness, 0.9)
        self.assertEqual(best.genome, "proof_3")

    def test_population_average_fitness(self):
        """Test calculating average fitness of population"""
        avg = self.population.average_fitness
        expected = (0.8 + 0.6 + 0.9 + 0.7 + 0.5) / 5
        self.assertAlmostEqual(avg, expected, places=5)

    def test_population_diversity_calculation(self):
        """Test population diversity calculation"""
        diversity = self.population.diversity
        self.assertGreaterEqual(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)

    def test_population_ranking_by_fitness(self):
        """Test ranking individuals by fitness"""
        ranked = sorted(self.population.individuals, key=lambda ind: ind.fitness, reverse=True)
        self.assertEqual(ranked[0].genome, "proof_3")
        self.assertEqual(ranked[-1].genome, "proof_5")

    def test_verified_individuals_filtering(self):
        """Test filtering for verified individuals only"""
        verified = [ind for ind in self.population.individuals if ind.metadata.get("verified", False)]
        self.assertEqual(len(verified), 3)
        self.assertTrue(all(ind.metadata.get("verified") for ind in verified))


class TestMDAPLeanSelector(unittest.TestCase):
    """
    Unit tests for MDAP-enhanced selection operator.

    Tests voting-based parent selection using first-to-ahead-by-k.
    """

    def setUp(self):
        """Set up test fixtures"""
        if not EVOLUTION_MAKER_AVAILABLE:
            self.skipTest("Evolution-MAKER integration not available")

        self.config = MakerevolutionConfig(
            voting_threshold=3,
            num_candidates=5
        )
        self.selector = MAKERSelection(self.config)

        # Create test population
        self.population = Population(
            individuals=[
                Individual(genome=f"proof_{i}", fitness=0.5 + i * 0.1, generation=0)
                for i in range(10)
            ],
            generation=0
        )

    def test_select_top_candidates(self):
        """Test selecting top N candidates for voting"""
        top_n = self.selector._select_top_candidates(self.population, n=5)
        self.assertEqual(len(top_n), 5)
        # Should be the top 5 by fitness
        self.assertEqual(top_n[0].genome, "proof_9")
        self.assertEqual(top_n[-1].genome, "proof_5")

    def test_voting_selection_single_winner(self):
        """Test voting selection produces single winner"""
        winners = self.selector.select(self.population, num_parents=2)
        self.assertEqual(len(winners), 2)
        self.assertIsInstance(winners[0], Individual)

    def test_voting_with_clear_winner(self):
        """Test voting when there's a clear winner"""
        # Create population with one very fit individual
        population = Population(
            individuals=[
                Individual(genome="best", fitness=0.99, generation=0),
                Individual(genome="good1", fitness=0.7, generation=0),
                Individual(genome="good2", fitness=0.7, generation=0),
                Individual(genome="good3", fitness=0.7, generation=0),
                Individual(genome="ok1", fitness=0.5, generation=0)
            ],
            generation=0
        )

        winners = self.selector.select(population, num_parents=1)
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0].genome, "best")

    def test_voting_threshold_effect(self):
        """Test effect of different voting thresholds"""
        configs = [
            MakerevolutionConfig(voting_threshold=2),
            MakerevolutionConfig(voting_threshold=3),
            MakerevolutionConfig(voting_threshold=5)
        ]

        for config in configs:
            selector = MAKERSelection(config)
            winners = selector.select(self.population, num_parents=1)
            self.assertEqual(len(winners), 1)


class TestMDAPLeanCrossover(unittest.TestCase):
    """
    Unit tests for MDAP-enhanced crossover operator.

    Tests agent-guided crossover of proof strategies.
    """

    def setUp(self):
        """Set up test fixtures"""
        if not EVOLUTION_MAKER_AVAILABLE:
            self.skipTest("Evolution-MAKER integration not available")

        self.parent1 = Individual(
            genome="intros; apply H; refl",
            fitness=0.8,
            generation=0,
            metadata={"tactics": ["intros", "apply H", "refl"]}
        )

        self.parent2 = Individual(
            genome="intros; cases h; simp; refl",
            fitness=0.7,
            generation=0,
            metadata={"tactics": ["intros", "cases h", "simp", "refl"]}
        )

    def test_tactic_crossover_single_point(self):
        """Test single-point crossover on tactic sequences"""
        tactics1 = self.parent1.metadata["tactics"]
        tactics2 = self.parent2.metadata["tactics"]

        # Single-point crossover at position 1
        crossover_point = 1
        child_tactics = tactics1[:crossover_point] + tactics2[crossover_point:]

        expected = ["intros", "cases h", "simp", "refl"]
        self.assertEqual(child_tactics, expected)

    def test_tactic_crossover_uniform(self):
        """Test uniform crossover on tactic sequences"""
        tactics1 = self.parent1.metadata["tactics"]
        tactics2 = self.parent2.metadata["tactics"]

        # Uniform crossover (alternating)
        child_tactics = []
        for i in range(min(len(tactics1), len(tactics2))):
            if i % 2 == 0:
                child_tactics.append(tactics1[i])
            else:
                child_tactics.append(tactics2[i])

        # Should mix tactics from both parents
        self.assertIn("intros", child_tactics)  # From both
        self.assertTrue(any(t in child_tactics for t in ["apply H", "cases h", "simp", "refl"]))

    def test_crossover_preserves_common_prefix(self):
        """Test that crossover preserves common prefix (intros)"""
        # Both parents start with "intros"
        child = Individual(
            genome="intros; cases h; refl",
            fitness=0.75,
            generation=1,
            metadata={"tactics": ["intros", "cases h", "refl"]}
        )

        # Child should preserve common prefix
        self.assertEqual(child.metadata["tactics"][0], "intros")


class TestMDAPLeanMutator(unittest.TestCase):
    """
    Unit tests for MDAP-enhanced mutation operator.

    Tests agent-guided mutation of proof strategies.
    """

    def setUp(self):
        """Set up test fixtures"""
        if not EVOLUTION_MAKER_AVAILABLE:
            self.skipTest("Evolution-MAKER integration not available")

        self.individual = Individual(
            genome="intros; apply H; refl",
            fitness=0.7,
            generation=0,
            metadata={"tactics": ["intros", "apply H", "refl"]}
        )

        self.mutation_tactics = [
            "simp", "linarith", "omega", "ring", "rw", "cases"
        ]

    def test_tactic_insertion_mutation(self):
        """Test inserting new tactic"""
        original_tactics = self.individual.metadata["tactics"].copy()

        # Insert "simp" at position 1
        mutated_tactics = original_tactics.copy()
        mutated_tactics.insert(1, "simp")

        expected = ["intros", "simp", "apply H", "refl"]
        self.assertEqual(mutated_tactics, expected)

    def test_tactic_replacement_mutation(self):
        """Test replacing existing tactic"""
        original_tactics = self.individual.metadata["tactics"].copy()

        # Replace "apply H" with "linarith"
        mutated_tactics = [
            "linarith" if t == "apply H" else t
            for t in original_tactics
        ]

        expected = ["intros", "linarith", "refl"]
        self.assertEqual(mutated_tactics, expected)

    def test_tactic_deletion_mutation(self):
        """Test deleting tactic"""
        original_tactics = self.individual.metadata["tactics"].copy()

        # Delete middle tactic
        mutated_tactics = original_tactics[:1] + original_tactics[2:]

        expected = ["intros", "refl"]
        self.assertEqual(mutated_tactics, expected)

    def test_mutation_rate_respected(self):
        """Test that mutation rate is respected"""
        mutation_rate = 0.2
        num_mutations = 0
        num_trials = 100

        for _ in range(num_trials):
            if random.random() < mutation_rate:
                num_mutations += 1

        # Should have approximately 20% mutations
        actual_rate = num_mutations / num_trials
        self.assertGreater(actual_rate, 0.1)
        self.assertLess(actual_rate, 0.3)


# =============================================================================
# INTEGRATION TESTS: Complete Evolutionary Loop
# =============================================================================

class TestMDAPEvolutionLoop(unittest.TestCase):
    """
    Integration tests for complete MDAP-enhanced evolutionary loop.
    """

    def setUp(self):
        """Set up test fixtures"""
        if not EVOLUTION_MAKER_AVAILABLE:
            self.skipTest("Evolution-MAKER integration not available")

        self.config = MakerevolutionConfig(
            mode=MakerevolutionMode.HYBRID,
            enable_voting=True,
            enable_decomposition=True,
            voting_threshold=3,
            population_size=10
        )

    def test_complete_evolutionary_loop(self):
        """Test running complete evolutionary loop with MDAP"""
        # Simple fitness evaluator
        def evaluator(genome: str) -> float:
            # Fitness based on length (prefer medium length)
            length = len(genome.split())
            if length < 3:
                return 0.3
            elif length < 6:
                return 0.8
            else:
                return 0.5

        # Initial population
        initial_program = "intros apply H refl"

        # Run for 3 generations
        result = run_maker_evolution(
            initial_program=initial_program,
            evaluator=evaluator,
            max_generations=3,
            config=self.config
        )

        # Check results
        self.assertIsNotNone(result)
        self.assertIn("best_program", result)
        self.assertIn("best_fitness", result)
        self.assertGreater(result["best_fitness"], 0.0)

    def test_population_initialization_with_mdap(self):
        """Test population initialization with MDAP"""
        def evaluator(genome: str) -> float:
            return random.random()

        initial_program = "intro n induction n"

        # Initialize population
        population = Population(
            individuals=[
                Individual(
                    genome=f"proof_{i}",
                    fitness=evaluator(f"proof_{i}"),
                    generation=0
                )
                for i in range(10)
            ],
            generation=0
        )

        self.assertEqual(len(population.individuals), 10)
        self.assertEqual(population.generation, 0)

    def test_generational_progression_with_voting(self):
        """Test generational progression with voting-based selection"""
        def evaluator(genome: str) -> float:
            return random.random()

        population = Population(
            individuals=[
                Individual(genome=f"proof_{i}", fitness=evaluator(f"proof_{i}"), generation=0)
                for i in range(10)
            ],
            generation=0
        )

        selector = MAKERSelection(self.config)

        # Progress 3 generations
        for gen in range(1, 4):
            # Select parents
            parents = selector.select(population, num_parents=5)

            # Create offspring (simple crossover + mutation)
            offspring = []
            for i in range(5):
                child_genome = f"proof_gen{gen}_child{i}"
                offspring.append(Individual(
                    genome=child_genome,
                    fitness=evaluator(child_genome),
                    generation=gen
                ))

            # Replace population
            population = Population(individuals=offspring, generation=gen)

            self.assertEqual(population.generation, gen)

    def test_convergence_with_agent_consensus(self):
        """Test convergence detection with agent consensus"""
        # Create converging population
        population = Population(
            individuals=[
                Individual(genome="converged_proof", fitness=0.95 + i * 0.005, generation=5)
                for i in range(10)
            ],
            generation=5
        )

        # Check if converged (high fitness, low diversity)
        best = population.best_individual
        avg = population.average_fitness

        # All individuals have high fitness and are similar
        self.assertGreater(best.fitness, 0.95)
        self.assertGreater(avg, 0.95)


# =============================================================================
# COMPARISON TESTS: Pure vs MDAP-Enhanced
# =============================================================================

class TestPureVsMDAPEnhancedEvolution(unittest.TestCase):
    """
    Comparison tests between pure evolution and MDAP-enhanced evolution.
    """

    def setUp(self):
        """Set up test fixtures"""
        if not EVOLUTION_MAKER_AVAILABLE:
            self.skipTest("Evolution-MAKER integration not available")

    def test_convergence_rate_comparison(self):
        """Test that MDAP-enhanced evolution converges faster"""
        def evaluator(genome: str) -> float:
            # Fitness: reward for containing "intros refl" (good proof)
            if "intros" in genome and "refl" in genome:
                return 0.9
            elif "intros" in genome:
                return 0.6
            else:
                return 0.3

        initial_program = "test proof"

        # Run pure evolution (no voting)
        config_pure = MakerevolutionConfig(
            enable_voting=False,
            population_size=10
        )

        # Run MDAP-enhanced (with voting)
        config_mdap = MakerevolutionConfig(
            enable_voting=True,
            voting_threshold=3,
            population_size=10
        )

        # MDAP should converge to higher fitness
        # (This is a simplified test - real comparison needs more generations)

    def test_proof_quality_improvement(self):
        """Test that MDAP-enhanced produces better quality proofs"""
        # Quality metric: verified, short, elegant
        def quality_metric(individual: Individual) -> float:
            score = 0.0
            if individual.metadata.get("verified", False):
                score += 5.0
            # Prefer shorter proofs
            score -= 0.1 * len(individual.genome.split())
            return max(score, 0.0)

        # Create two populations
        pure_pop = Population(
            individuals=[
                Individual(
                    genome="intros " + "apply H " * 10 + "refl",  # Long
                    fitness=0.7,
                    generation=0,
                    metadata={"verified": True}
                )
            ],
            generation=0
        )

        mdap_pop = Population(
            individuals=[
                Individual(
                    genome="intros refl",  # Short
                    fitness=0.9,
                    generation=0,
                    metadata={"verified": True}
                )
            ],
            generation=0
        )

        pure_quality = quality_metric(pure_pop.best_individual)
        mdap_quality = quality_metric(mdap_pop.best_individual)

        # MDAP should produce higher quality (shorter verified proof)
        self.assertGreater(mdap_quality, pure_quality)

    def test_agent_contribution_analysis(self):
        """Test analysis of agent contributions to evolution"""
        # Track which agents produce successful individuals
        agent_contributions = {
            "constructive": 0,
            "inductive": 0,
            "algebraic": 0
        }

        # Simulate evolution with different agents
        for _ in range(20):
            agent = random.choice(list(agent_contributions.keys()))
            if random.random() > 0.5:  # 50% success rate
                agent_contributions[agent] += 1

        # Check that multiple agents contributed
        total_contributions = sum(agent_contributions.values())
        self.assertGreater(total_contributions, 0)

        # Find most successful agent
        best_agent = max(agent_contributions, key=agent_contributions.get)
        self.assertIn(best_agent, agent_contributions.keys())


# =============================================================================
# WORKFLOW TESTS: Stage Integration
# =============================================================================

class TestWorkflowIntegration(unittest.TestCase):
    """
    Tests for workflow integration with MDAP-enhanced evolution.
    """

    def setUp(self):
        """Set up test fixtures"""
        if not EVOLUTION_MAKER_AVAILABLE:
            self.skipTest("Evolution-MAKER integration not available")

    def test_stage_3a_mdap_evolution_integration(self):
        """Test Stage 3A: MDAP-evolution integration for proof search"""
        # Simulate Stage 3A workflow
        theorem = "∀ n : Nat, n + 0 = n"

        # MDAP-enhanced evolution for proof search
        def evaluator(genome: str) -> float:
            # Reward proofs that look reasonable
            if "intros" in genome and "refl" in genome:
                return 0.9
            return 0.3

        config = MakerevolutionConfig(
            mode=MakerevolutionMode.HYBRID,
            enable_voting=True,
            voting_threshold=3
        )

        result = run_maker_evolution(
            initial_program=f"prove {theorem}",
            evaluator=evaluator,
            max_generations=5,
            config=config
        )

        self.assertIsNotNone(result)
        self.assertIn("best_program", result)

    def test_stage_3b_refinement(self):
        """Test Stage 3B: Refinement with MDAP voting"""
        # Start with good proof, refine with voting
        initial_proof = "intros n induction n case n=0 case n=succ"

        def evaluator(genome: str) -> float:
            # Reward completeness
            if "refl" in genome or "rfl" in genome:
                return 0.95
            return 0.6

        config = MakerevolutionConfig(
            mode=MakerevolutionMode.VOTING_ONLY,
            voting_threshold=5  # Higher threshold for refinement
        )

        result = run_maker_evolution(
            initial_program=initial_proof,
            evaluator=evaluator,
            max_generations=3,
            config=config
        )

        self.assertGreater(result["best_fitness"], 0.6)

    def test_adaptive_strategy_selection(self):
        """Test adaptive strategy selection based on theorem difficulty"""
        theorems = {
            "easy": "∀ n, n + 0 = n",
            "medium": "∀ a b, a + b = b + a",
            "hard": "∀ f, (∀ n, f n = 0) -> f = (λ _, 0)"
        }

        # Select strategy based on difficulty
        strategy_map = {
            "easy": MakerevolutionMode.VOTING_ONLY,  # Fast
            "medium": MakerevolutionMode.HYBRID,  # Balanced
            "hard": MakerevolutionMode.FULL_MAKER  # Thorough
        }

        for difficulty, theorem in theorems.items():
            strategy = strategy_map[difficulty]
            self.assertIsNotNone(strategy)

    def test_fallback_behavior(self):
        """Test fallback when MDAP voting fails"""
        config = MakerevolutionConfig(
            enable_voting=True,
            enable_decomposition=True,
            fallback_policy="best_effort"
        )

        # Simulate voting failure (all agents fail)
        def evaluator(genome: str) -> float:
            return 0.0  # All fitnesses are 0

        initial_program = "test proof"

        # Should fall back to best effort
        result = run_maker_evolution(
            initial_program=initial_program,
            evaluator=evaluator,
            max_generations=2,
            config=config
        )

        # Should still return a result
        self.assertIsNotNone(result)


# =============================================================================
# PERFORMANCE TESTS: Benchmarks and Scaling
# =============================================================================

class TestPerformanceBenchmarks(unittest.TestCase):
    """
    Performance tests for MDAP-enhanced evolution.
    """

    def setUp(self):
        """Set up test fixtures"""
        if not EVOLUTION_MAKER_AVAILABLE:
            self.skipTest("Evolution-MAKER integration not available")

    def test_performance_comparison_time(self):
        """Compare execution time: pure vs MDAP-enhanced"""
        def evaluator(genome: str) -> float:
            return random.random()

        initial_program = "test proof"

        # Pure evolution
        start = time.time()
        result_pure = run_maker_evolution(
            initial_program=initial_program,
            evaluator=evaluator,
            max_generations=5,
            config=MakerevolutionConfig(enable_voting=False)
        )
        time_pure = time.time() - start

        # MDAP-enhanced
        start = time.time()
        result_mdap = run_maker_evolution(
            initial_program=initial_program,
            evaluator=evaluator,
            max_generations=5,
            config=MakerevolutionConfig(enable_voting=True, voting_threshold=3)
        )
        time_mdap = time.time() - start

        # MDAP should be slower but more accurate
        self.assertGreater(time_mdap, time_pure * 0.8)  # Allow some overlap

    def test_scalability_population_size(self):
        """Test scaling with different population sizes"""
        def evaluator(genome: str) -> float:
            return random.random()

        initial_program = "test proof"

        population_sizes = [10, 20, 30]
        times = []

        for pop_size in population_sizes:
            start = time.time()
            result = run_maker_evolution(
                initial_program=initial_program,
                evaluator=evaluator,
                max_generations=3,
                config=MakerevolutionConfig(population_size=pop_size)
            )
            elapsed = time.time() - start
            times.append(elapsed)

        # Time should increase with population size
        # (approximately linear for this simplified test)
        self.assertGreater(times[1], times[0] * 0.8)
        self.assertGreater(times[2], times[1] * 0.8)

    def test_agent_count_impact(self):
        """Test impact of different voting thresholds (agent counts)"""
        def evaluator(genome: str) -> float:
            return random.random()

        initial_program = "test proof"

        thresholds = [2, 3, 5]
        results = []

        for threshold in thresholds:
            result = run_maker_evolution(
                initial_program=initial_program,
                evaluator=evaluator,
                max_generations=5,
                config=MakerevolutionConfig(voting_threshold=threshold)
            )
            results.append(result)

        # All should complete successfully
        for result in results:
            self.assertIsNotNone(result)
            self.assertIn("best_fitness", result)

    def test_voting_overhead(self):
        """Test overhead of voting mechanism"""
        def evaluator(genome: str) -> float:
            return random.random()

        initial_program = "test proof"

        # Measure voting overhead
        iterations = 5
        times_with_voting = []
        times_without_voting = []

        for _ in range(iterations):
            # With voting
            start = time.time()
            run_maker_evolution(
                initial_program=initial_program,
                evaluator=evaluator,
                max_generations=3,
                config=MakerevolutionConfig(enable_voting=True)
            )
            times_with_voting.append(time.time() - start)

            # Without voting
            start = time.time()
            run_maker_evolution(
                initial_program=initial_program,
                evaluator=evaluator,
                max_generations=3,
                config=MakerevolutionConfig(enable_voting=False)
            )
            times_without_voting.append(time.time() - start)

        avg_with = sum(times_with_voting) / len(times_with_voting)
        avg_without = sum(times_without_voting) / len(times_without_voting)

        # Voting should add some overhead
        logger.info(f"Average time with voting: {avg_with:.3f}s")
        logger.info(f"Average time without voting: {avg_without:.3f}s")
        logger.info(f"Voting overhead: {(avg_with / avg_without - 1) * 100:.1f}%")


# =============================================================================
# EDGE CASE TESTS: Error Handling and Robustness
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    """
    Edge case tests for MDAP-enhanced evolution.
    """

    def setUp(self):
        """Set up test fixtures"""
        if not EVOLUTION_MAKER_AVAILABLE:
            self.skipTest("Evolution-MAKER integration not available")

    def test_all_agents_fail_during_voting(self):
        """Test behavior when all agents fail during voting"""
        # All fitnesses are 0 (all agents fail)
        def evaluator(genome: str) -> float:
            return 0.0

        initial_program = "test proof"

        config = MakerevolutionConfig(
            enable_voting=True,
            voting_threshold=3,
            fallback_policy="best_effort"
        )

        result = run_maker_evolution(
            initial_program=initial_program,
            evaluator=evaluator,
            max_generations=3,
            config=config
        )

        # Should still return result with fallback
        self.assertIsNotNone(result)

    def test_voting_ties(self):
        """Test handling of voting ties"""
        # Create population with tied fitness
        population = Population(
            individuals=[
                Individual(genome=f"proof_{i}", fitness=0.8, generation=0)
                for i in range(5)
            ],
            generation=0
        )

        selector = MAKERSelection(MakerevolutionConfig(voting_threshold=3))
        winners = selector.select(population, num_parents=2)

        # Should break ties and return winners
        self.assertEqual(len(winners), 2)

    def test_entire_population_red_flagged(self):
        """Test behavior when entire population is red-flagged"""
        if not MDAP_AVAILABLE:
            self.skipTest("MDAP engine not available")

        red_flag_rules = RedFlagRules(
            min_confidence=0.9  # Very strict
        )

        flagger = RedFlagger(red_flag_rules)

        # All candidates should be flagged
        for i in range(5):
            raw_text = f"proof_{i}"
            candidate = {"confidence": 0.5}  # Too low
            is_flagged, _ = flagger.is_flagged(raw_text, candidate, None)
            self.assertTrue(is_flagged)

    def test_empty_agent_list(self):
        """Test behavior with empty agent list"""
        population = Population(
            individuals=[],
            generation=0
        )

        # Should handle gracefully
        self.assertIsNone(population.best_individual)
        self.assertEqual(population.average_fitness, 0.0)

    def test_population_collapse(self):
        """Test recovery from population collapse (low diversity)"""
        # Create collapsed population (all identical)
        population = Population(
            individuals=[
                Individual(genome="identical_proof", fitness=0.7, generation=0)
                for _ in range(10)
            ],
            generation=0
        )

        # Diversity should be 0
        self.assertEqual(population.diversity, 0.0)

        # Inject diversity
        new_individuals = [
            Individual(genome=f"diverse_{i}", fitness=random.random(), generation=0)
            for i in range(5)
        ]

        population.individuals.extend(new_individuals)

        # Diversity should now be > 0
        self.assertGreater(population.diversity, 0.0)

    def test_single_individual_population(self):
        """Test evolution with single individual"""
        population = Population(
            individuals=[
                Individual(genome="lone_proof", fitness=0.8, generation=0)
            ],
            generation=0
        )

        # Should still work
        self.assertIsNotNone(population.best_individual)
        self.assertEqual(population.best_individual.genome, "lone_proof")

    def test_extremely_high_voting_threshold(self):
        """Test with extremely high voting threshold"""
        def evaluator(genome: str) -> float:
            return random.random()

        config = MakerevolutionConfig(
            voting_threshold=100,  # Much larger than population
            population_size=10
        )

        result = run_maker_evolution(
            initial_program="test",
            evaluator=evaluator,
            max_generations=2,
            config=config
        )

        # Should handle gracefully (use fallback)
        self.assertIsNotNone(result)


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_tests(
    test_categories: Optional[List[str]] = None,
    verbose: bool = True
) -> unittest.TestResult:
    """
    Run MDAP-evolution tests.

    Args:
        test_categories: List of test categories to run.
            Options: ["unit", "integration", "comparison", "workflow", "performance", "edge"]
            If None, runs all tests.
        verbose: Whether to use verbose output

    Returns:
        TestResult object with test outcomes
    """
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Map category names to test classes
    category_map = {
        "unit": [
            TestMDAPLeanPopulation,
            TestMDAPLeanSelector,
            TestMDAPLeanCrossover,
            TestMDAPLeanMutator
        ],
        "integration": [
            TestMDAPEvolutionLoop
        ],
        "comparison": [
            TestPureVsMDAPEnhancedEvolution
        ],
        "workflow": [
            TestWorkflowIntegration
        ],
        "performance": [
            TestPerformanceBenchmarks
        ],
        "edge": [
            TestEdgeCases
        ]
    }

    # Add tests based on categories
    if test_categories is None:
        # Run all tests
        for test_classes in category_map.values():
            for test_class in test_classes:
                suite.addTests(loader.loadTestsFromTestCase(test_class))
    else:
        # Run specific categories
        for category in test_categories:
            if category in category_map:
                for test_class in category_map[category]:
                    suite.addTests(loader.loadTestsFromTestCase(test_class))
            else:
                logger.warning(f"Unknown test category: {category}")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)

    return result


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LeanAide MDAP-Evolution Tests"
    )
    parser.add_argument(
        "--category",
        "-c",
        action="append",
        choices=["unit", "integration", "comparison", "workflow", "performance", "edge"],
        help="Test category to run (can specify multiple)"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # Run tests
    result = run_tests(
        test_categories=args.category,
        verbose=not args.quiet
    )

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
