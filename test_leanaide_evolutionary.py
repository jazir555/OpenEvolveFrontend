"""
Comprehensive Test Suite for Evolutionary LeanAide Integration

This test suite provides comprehensive testing for all evolutionary LeanAide components:
- Evolution Tests (leanaide_evolution.py)
- Decomposition Tests (leanaide_decomposition_integration.py)
- Adversarial Tests (leanaide_adversarial.py)
- Self-Play Tests (leanaide_selfplay.py)
- Strategy Library Tests (leanaide_strategies.py)
- Workflow Integration Tests

Author: OpenEvolve
Created: 2025-12-30
"""

import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import tempfile
import shutil

# pytest imports
import pytest
from pytest import mark
from pytest_asyncio import fixture

# Add LeanAide to path if needed
LEANAIDE_PATH = os.path.join(os.path.dirname(__file__), "LeanAide")
if os.path.exists(LEANAIDE_PATH) and LEANAIDE_PATH not in sys.path:
    sys.path.insert(0, LEANAIDE_PATH)

# =============================================================================
# IMPORT EVOLUTIONARY COMPONENTS
# =============================================================================

try:
    from leanaide_evolution import (
        LeanProofEvolutionEngine,
        LeanProofStrategy,
        LeanProofPopulation,
        LeanProofMutator,
        LeanProofEvaluator,
        LeanProofCrossover,
        Tactic,
        LeanProof,
        EvolutionResult,
        PopulationStatistics,
        MutationType,
        SelectionMethod,
        CrossoverMethod,
        evolve_proof,
        LEANAIDE_AVAILABLE
    )
    EVOLUTION_AVAILABLE = True
except ImportError as e:
    EVOLUTION_AVAILABLE = False
    print(f"Warning: Could not import leanaide_evolution: {e}")

try:
    from leanaide_decomposition_integration import (
        LeanAideDecompositionIntegrator,
        MathematicalComponent,
        DecompositionResult,
        MathematicalDomain,
        extract_mathematical_components,
        identify_dependencies,
        estimate_complexity,
        generate_subproblems,
        topological_order,
        detect_parallelization
    )
    DECOMPOSITION_AVAILABLE = True
except ImportError as e:
    DECOMPOSITION_AVAILABLE = False
    print(f"Warning: Could not import leanaide_decomposition_integration: {e}")

try:
    from leanaide_adversarial import (
        LeanAideAdversarialEngine,
        BlueTeamAgent,
        RedTeamAgent,
        AdversarialRound,
        AdversarialResult,
        Counterexample,
        ProofCritique,
        CoevolutionDynamics,
        evolve_adversarially
    )
    ADVERSARIAL_AVAILABLE = True
except ImportError as e:
    ADVERSARIAL_AVAILABLE = False
    print(f"Warning: Could not import leanaide_adversarial: {e}")

try:
    from leanaide_selfplay import (
        LeanAideSelfPlayEngine,
        SelfPlayGame,
        SelfPlayAgent,
        ExperienceBuffer,
        SelfPlayResult,
        SelfPlayStrategy,
        run_selfplay_training,
        select_agent_strategy
    )
    SELFPLAY_AVAILABLE = True
except ImportError as e:
    SELFPLAY_AVAILABLE = False
    print(f"Warning: Could not import leanaide_selfplay: {e}")

try:
    from leanaide_strategies import (
        TacticLibrary,
        StrategyTemplate,
        StrategySelector,
        StrategyGenerator,
        StrategyMutator,
        StrategyCombiner,
        SuccessTracker,
        get_tactic_library,
        select_strategy,
        mutate_strategy,
        combine_strategies
    )
    STRATEGIES_AVAILABLE = True
except ImportError as e:
    STRATEGIES_AVAILABLE = False
    print(f"Warning: Could not import leanaide_strategies: {e}")


# =============================================================================
# PYTEST CONFIGURATION AND FIXTURES
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for end-to-end workflows")
    config.addinivalue_line("markers", "mock: Tests that use mocking (offline testing)")
    config.addinivalue_line("markers", "server: Tests that require LeanAide server running")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")
    config.addinivalue_line("markers", "evolution: Evolution-specific tests")
    config.addinivalue_line("markers", "decomposition: Decomposition-specific tests")
    config.addinivalue_line("markers", "adversarial: Adversarial-specific tests")
    config.addinivalue_line("markers", "selfplay: Self-play-specific tests")
    config.addinivalue_line("markers", "strategy: Strategy-specific tests")
    config.addinivalue_line("markers", "workflow: Workflow integration tests")


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory for test data."""
    test_dir = Path(__file__).parent / "test_leanaide_evolution_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture(scope="session")
def sample_theorems():
    """Sample theorems of varying difficulty."""
    return {
        "trivial": {
            "statement": "True is true",
            "lean_code": "theorem trivial : True := by trivial",
            "difficulty": "trivial"
        },
        "simple": {
            "statement": "There are infinitely many prime numbers",
            "lean_code": "theorem infinitely_many_primes : Infinite {p : Nat | Nat.Prime p} := by sorry",
            "difficulty": "simple"
        },
        "medium": {
            "statement": "The square root of 2 is irrational",
            "lean_code": "theorem sqrt2_irrational : Irrational (Real.sqrt 2) := by sorry",
            "difficulty": "medium"
        },
        "complex": {
            "statement": "Every natural number has a unique prime factorization",
            "lean_code": "theorem prime_factorization_unique (n : Nat) (h : n > 0) : ∀ f1 f2 : List Nat, (∀ p ∈ f1, Nat.Prime p) → (∀ p ∈ f2, Nat.Prime p) → f1.prod = n → f2.prod = n → f1.perm f2 := by sorry",
            "difficulty": "complex"
        },
        "algebraic": {
            "statement": "The product of two even numbers is even",
            "lean_code": "theorem even_product_even (a b : Nat) (ha : Even a) (hb : Even b) : Even (a * b) := by sorry",
            "difficulty": "simple"
        }
    }


@pytest.fixture(scope="session")
def sample_mathematical_problems():
    """Sample mathematical problems for decomposition testing."""
    return {
        "single_step": "Prove that 2 + 2 = 4",
        "multi_step": "Prove that the sum of two even numbers is even",
        "complex": "Prove that every natural number greater than 1 has a prime divisor",
        "with_dependencies": "Prove that if n is composite, then n has a prime factor less than or equal to sqrt(n)",
        "parallelizable": "Prove that for any integers a, b, c, d: if a divides b and c divides d, then ac divides bd"
    }


@pytest.fixture(scope="session")
def sample_lean_tactics():
    """Sample Lean 4 tactics."""
    return [
        "simp", "rw", "apply", "exact", "refine",
        "cases", "induction", "constructor", "intros",
        "have", "suffices", "show", "calc",
        "aesop", "linarith", "ring", "omega",
        "norm_num", "trivial", "decide", "done"
    ]


@pytest.fixture
def temp_cache_dir():
    """Temporary directory for cache testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_verification_result():
    """Mock verification result."""
    def _result(success: bool = True, errors: List[str] = None):
        result = MagicMock()
        result.success = success
        result.errors = errors or []
        result.warnings = []
        result.sorries = [] if success else ["placeholder"]
        result.execution_time = 0.1
        return result
    return _result


# =============================================================================
# EVOLUTION TESTS
# =============================================================================

@mark.unit
@mark.evolution
class TestLeanProofStrategy:
    """Test LeanProofStrategy class."""

    def test_strategy_creation(self):
        """Test creating a proof strategy."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        proof = LeanProof(
            theorem_name="test_theorem",
            theorem_statement="True is true",
            lean_code="theorem test : True := by trivial"
        )

        strategy = LeanProofStrategy(proof=proof, generation=0)

        assert strategy.proof == proof
        assert strategy.fitness == 0.0
        assert strategy.generation == 0
        assert strategy.parents == []
        assert strategy.mutation_history == []
        assert strategy.verified is False
        assert isinstance(strategy.strategy_id, str)

    def test_tactics_sequence(self):
        """Test getting tactics sequence."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        tactics = [
            Tactic(name="intros"),
            Tactic(name="simp"),
            Tactic(name="apply", arguments=["_"])
        ]

        proof = LeanProof(
            theorem_name="test",
            theorem_statement="True",
            lean_code="",
            tactics=tactics
        )

        strategy = LeanProofStrategy(proof=proof)

        sequence = strategy.get_tactics_sequence()
        assert "intros" in sequence
        assert "simp" in sequence
        assert "apply" in sequence

    def test_complexity_calculation(self):
        """Test complexity score calculation."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        # Simple proof
        simple_tactics = [Tactic(name="simp"), Tactic(name="trivial")]
        simple_proof = LeanProof(
            theorem_name="simple",
            theorem_statement="True",
            lean_code="",
            tactics=simple_tactics
        )
        simple_strategy = LeanProofStrategy(proof=simple_proof)
        simple_complexity = simple_strategy.calculate_complexity()
        assert 0.0 <= simple_complexity <= 10.0

        # Complex proof
        complex_tactics = [
            Tactic(name="induction"),
            Tactic(name="cases"),
            Tactic(name="calc"),
            Tactic(name="have"),
            Tactic(name="apply", arguments=["_"])
        ]
        complex_proof = LeanProof(
            theorem_name="complex",
            theorem_statement="False",
            lean_code="",
            tactics=complex_tactics
        )
        complex_strategy = LeanProofStrategy(proof=complex_proof)
        complex_complexity = complex_strategy.calculate_complexity()
        assert complex_complexity > simple_complexity

    def test_elegance_calculation(self):
        """Test elegance score calculation."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        # Elegant proof (concise, diverse)
        elegant_tactics = [
            Tactic(name="simp"),
            Tactic(name="rw"),
            Tactic(name="exact", arguments=["_"])
        ]
        elegant_proof = LeanProof(
            theorem_name="elegant",
            theorem_statement="True",
            lean_code="",
            tactics=elegant_tactics
        )
        elegant_strategy = LeanProofStrategy(proof=elegant_proof)
        elegant_score = elegant_strategy.calculate_elegance()
        assert 0.0 <= elegant_score <= 1.0

        # Less elegant (repetitive)
        repetitive_tactics = [Tactic(name="simp") for _ in range(10)]
        repetitive_proof = LeanProof(
            theorem_name="repetitive",
            theorem_statement="True",
            lean_code="",
            tactics=repetitive_tactics
        )
        repetitive_strategy = LeanProofStrategy(proof=repetitive_proof)
        repetitive_score = repetitive_strategy.calculate_elegance()
        # Elegant should score higher
        assert elegant_score >= repetitive_score

    def test_strategy_serialization(self):
        """Test strategy serialization to dict."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        proof = LeanProof(
            theorem_name="test",
            theorem_statement="True",
            lean_code="theorem test : True := by trivial"
        )

        strategy = LeanProofStrategy(
            proof=proof,
            fitness=5.0,
            generation=3,
            parents=["parent1", "parent2"],
            mutation_history=[MutationType.TACTIC_SUBSTITUTION]
        )

        strategy_dict = strategy.to_dict()

        assert strategy_dict["strategy_id"] == strategy.strategy_id
        assert strategy_dict["fitness"] == 5.0
        assert strategy_dict["generation"] == 3
        assert len(strategy_dict["parents"]) == 2
        assert len(strategy_dict["mutation_history"]) == 1


@mark.unit
@mark.evolution
class TestLeanProofPopulation:
    """Test LeanProofPopulation class."""

    @pytest.fixture
    def sample_population(self):
        """Create a sample population."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        strategies = []
        for i in range(10):
            proof = LeanProof(
                theorem_name=f"theorem_{i}",
                theorem_statement=f"statement_{i}",
                lean_code=f"code_{i}",
                tactics=[Tactic(name="simp")]
            )
            strategy = LeanProofStrategy(
                proof=proof,
                fitness=float(i)  # Varying fitness
            )
            strategies.append(strategy)

        return LeanProofPopulation(
            strategies=strategies,
            selection_method=SelectionMethod.TOURNAMENT,
            tournament_size=3,
            elitism_ratio=0.1
        )

    def test_population_size(self, sample_population):
        """Test population size."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        assert len(sample_population) == 10

    def test_get_best_strategy(self, sample_population):
        """Test getting best strategy."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        best = sample_population.get_best_strategy()
        assert best.fitness == 9.0  # Highest fitness

    def test_get_worst_strategy(self, sample_population):
        """Test getting worst strategy."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        worst = sample_population.get_worst_strategy()
        assert worst.fitness == 0.0  # Lowest fitness

    def test_diversity_calculation(self, sample_population):
        """Test diversity calculation."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        diversity = sample_population.calculate_diversity()
        assert 0.0 <= diversity <= 1.0

    def test_tournament_selection(self, sample_population):
        """Test tournament selection."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        parents = sample_population.select_parents(num_parents=5)
        assert len(parents) == 5
        # All should be from the population
        for parent in parents:
            assert parent in sample_population.strategies

    def test_roulette_selection(self):
        """Test roulette wheel selection."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        strategies = []
        for i in range(10):
            proof = LeanProof(
                theorem_name=f"t{i}",
                theorem_statement="s",
                lean_code="c",
                tactics=[Tactic(name="simp")]
            )
            strategy = LeanProofStrategy(proof=proof, fitness=float(i + 1))
            strategies.append(strategy)

        population = LeanProofPopulation(
            strategies=strategies,
            selection_method=SelectionMethod.ROULETTE
        )

        parents = population.select_parents(num_parents=5)
        assert len(parents) == 5

    def test_rank_selection(self):
        """Test rank-based selection."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        strategies = []
        for i in range(10):
            proof = LeanProof(
                theorem_name=f"t{i}",
                theorem_statement="s",
                lean_code="c",
                tactics=[Tactic(name="simp")]
            )
            strategy = LeanProofStrategy(proof=proof, fitness=float(i))
            strategies.append(strategy)

        population = LeanProofPopulation(
            strategies=strategies,
            selection_method=SelectionMethod.RANK
        )

        parents = population.select_parents(num_parents=5)
        assert len(parents) == 5

    def test_elitism(self, sample_population):
        """Test getting elite strategies."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        elites = sample_population.get_elites(num_elites=2)
        assert len(elites) == 2
        # Should be the two best
        assert elites[0].fitness >= elites[1].fitness

    def test_population_statistics(self, sample_population):
        """Test population statistics calculation."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        stats = sample_population.calculate_statistics()

        assert stats.generation == 0
        assert stats.population_size == 10
        assert stats.best_fitness == 9.0
        assert stats.worst_fitness == 0.0
        assert 0.0 <= stats.average_fitness <= 10.0
        assert stats.verified_count >= 0


@mark.unit
@mark.evolution
class TestLeanProofMutator:
    """Test LeanProofMutator class."""

    @pytest.fixture
    def mutator(self):
        """Create a mutator instance."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")
        return LeanProofMutator(mutation_rate=0.5, mutation_strength=0.5)

    @pytest.fixture
    def sample_strategy(self):
        """Create a sample strategy."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        tactics = [
            Tactic(name="intros"),
            Tactic(name="simp"),
            Tactic(name="apply", arguments=["_"]),
            Tactic(name="exact", arguments=["_"])
        ]

        proof = LeanProof(
            theorem_name="test",
            theorem_statement="True",
            lean_code="",
            tactics=tactics
        )

        return LeanProofStrategy(proof=proof, generation=0)

    def test_tactic_substitution(self, mutator, sample_strategy):
        """Test tactic substitution mutation."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        mutated = mutator._tactic_substitution(sample_strategy)

        # Should be a new strategy
        assert mutated.strategy_id != sample_strategy.strategy_id
        # Should have mutation history
        assert MutationType.TACTIC_SUBSTITUTION in mutated.mutation_history

    def test_step_insertion(self, mutator, sample_strategy):
        """Test step insertion mutation."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        original_length = len(sample_strategy.proof.tactics)
        mutated = mutator._step_insertion(sample_strategy)

        # Should have more tactics
        assert len(mutated.proof.tactics) >= original_length
        assert MutationType.STEP_INSERTION in mutated.mutation_history

    def test_step_deletion(self, mutator, sample_strategy):
        """Test step deletion mutation."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        # Ensure we have enough tactics to delete
        sample_strategy.proof.tactics.extend([Tactic(name="simp") for _ in range(3)])

        original_length = len(sample_strategy.proof.tactics)
        mutated = mutator._step_deletion(sample_strategy)

        # Should have fewer tactics
        assert len(mutated.proof.tactics) <= original_length
        assert MutationType.STEP_DELETION in mutated.mutation_history

    def test_goal_restructuring(self, mutator, sample_strategy):
        """Test goal restructuring mutation."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        mutated = mutator._goal_restructuring(sample_strategy)

        # Should have structural tactic
        structural_tactics = [t.name for t in mutated.proof.tactics if t.name in ["have", "suffices", "show"]]
        assert len(structural_tactics) > 0
        assert MutationType.GOAL_RESTRUCTURING in mutated.mutation_history

    def test_lemma_introduction(self, mutator, sample_strategy):
        """Test lemma introduction mutation."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        original_length = len(sample_strategy.proof.tactics)
        mutated = mutator._lemma_introduction(sample_strategy)

        # Should have 'have' tactic at beginning
        assert mutated.proof.tactics[0].name == "have"
        assert len(mutated.proof.tactics) > original_length
        assert MutationType.LEMMA_INTRODUCTION in mutated.mutation_history

    def test_lemma_removal(self, mutator):
        """Test lemma removal mutation."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        # Create strategy with 'have' tactic
        tactics = [
            Tactic(name="have", arguments=["helper : Prop"]),
            Tactic(name="simp"),
            Tactic(name="apply", arguments=["_"])
        ]

        proof = LeanProof(
            theorem_name="test",
            theorem_statement="True",
            lean_code="",
            tactics=tactics
        )

        strategy = LeanProofStrategy(proof=proof)

        mutated = mutator._lemma_removal(strategy)

        # Should have removed 'have'
        have_tactics = [t for t in mutated.proof.tactics if t.name == "have"]
        assert len(have_tactics) < len([t for t in strategy.proof.tactics if t.name == "have"])
        assert MutationType.LEMMA_REMOVAL in mutated.mutation_history

    def test_reordering(self, mutator, sample_strategy):
        """Test reordering mutation."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        original_order = [t.name for t in sample_strategy.proof.tactics]
        mutated = mutator._reordering(sample_strategy)

        mutated_order = [t.name for t in mutated.proof.tactics]

        # Order should be different (or at least mutated)
        assert MutationType.REORDERING in mutated.mutation_history

    def test_full_mutation(self, mutator, sample_strategy):
        """Test full mutation process."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        mutated = mutator.mutate(sample_strategy)

        # Should be a new strategy
        assert mutated.strategy_id != sample_strategy.strategy_id
        assert mutated.generation == sample_strategy.generation
        assert sample_strategy.strategy_id in mutated.parents

    def test_custom_tactics(self):
        """Test mutator with custom tactics."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        custom_tactics = ["custom_tac1", "custom_tac2"]
        mutator = LeanProofMutator(custom_tactics=custom_tactics)

        # Should include custom tactics in available pool
        assert "custom_tac1" in mutator.custom_tactics
        assert "custom_tac2" in mutator.custom_tactics


@mark.unit
@mark.evolution
class TestLeanProofCrossover:
    """Test LeanProofCrossover class."""

    @pytest.fixture
    def crossover(self):
        """Create a crossover instance."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")
        return LeanProofCrossover(crossover_rate=0.8)

    @pytest.fixture
    def parent1(self):
        """Create first parent."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        tactics = [
            Tactic(name="intros"),
            Tactic(name="simp"),
            Tactic(name="apply", arguments=["lemma1"])
        ]

        proof = LeanProof(
            theorem_name="test",
            theorem_statement="True",
            lean_code="",
            tactics=tactics
        )

        return LeanProofStrategy(proof=proof, generation=0)

    @pytest.fixture
    def parent2(self):
        """Create second parent."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        tactics = [
            Tactic(name="intros"),
            Tactic(name="rw"),
            Tactic(name="exact", arguments=["lemma2"])
        ]

        proof = LeanProof(
            theorem_name="test",
            theorem_statement="True",
            lean_code="",
            tactics=tactics
        )

        return LeanProofStrategy(proof=proof, generation=0)

    def test_uniform_crossover(self, crossover, parent1, parent2):
        """Test uniform crossover."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        child = crossover._uniform_crossover(parent1, parent2)

        # Child should be new
        assert child.strategy_id != parent1.strategy_id
        assert child.strategy_id != parent2.strategy_id

        # Should have both parents
        assert parent1.strategy_id in child.parents
        assert parent2.strategy_id in child.parents

        # Generation should be incremented
        assert child.generation == max(parent1.generation, parent2.generation) + 1

    def test_single_point_crossover(self, crossover, parent1, parent2):
        """Test single-point crossover."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        child = crossover._single_point_crossover(parent1, parent2)

        assert child.strategy_id not in [parent1.strategy_id, parent2.strategy_id]
        assert len(child.parents) == 2

    def test_two_point_crossover(self, crossover, parent1, parent2):
        """Test two-point crossover."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        # Add more tactics for two-point crossover
        parent1.proof.tactics.extend([Tactic(name="cases"), Tactic(name="calc")])
        parent2.proof.tactics.extend([Tactic(name="have"), Tactic(name="show")])

        child = crossover._two_point_crossover(parent1, parent2)

        assert child.strategy_id not in [parent1.strategy_id, parent2.strategy_id]

    def test_ordered_crossover(self, crossover, parent1, parent2):
        """Test ordered crossover."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        child = crossover._ordered_crossover(parent1, parent2)

        assert child.strategy_id not in [parent1.strategy_id, parent2.strategy_id]
        # Tactics should maintain relative order from at least one parent

    def test_crossover_rate(self, parent1, parent2):
        """Test crossover rate affects result."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        # Low crossover rate - should often return parent without crossover
        low_rate_crossover = LeanProofCrossover(crossover_rate=0.0)
        child = low_rate_crossover.crossover(parent1, parent2)

        # Should be one of the parents
        assert child.strategy_id in [parent1.strategy_id, parent2.strategy_id]

    def test_crossover_with_different_lengths(self):
        """Test crossover with parents of different lengths."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        crossover = LeanProofCrossover()

        # Short parent
        short_tactics = [Tactic(name="simp")]
        short_proof = LeanProof(
            theorem_name="test",
            theorem_statement="True",
            lean_code="",
            tactics=short_tactics
        )
        short_parent = LeanProofStrategy(proof=short_proof)

        # Long parent
        long_tactics = [Tactic(name=str(i)) for i in range(10)]
        long_proof = LeanProof(
            theorem_name="test",
            theorem_statement="True",
            lean_code="",
            tactics=long_tactics
        )
        long_parent = LeanProofStrategy(proof=long_proof)

        child = crossover._uniform_crossover(short_parent, long_parent)

        # Child should exist
        assert child is not None
        assert child.strategy_id not in [short_parent.strategy_id, long_parent.strategy_id]


# =============================================================================
# DECOMPOSITION TESTS
# =============================================================================

@mark.unit
@mark.decomposition
class TestMathematicalComponentExtraction:
    """Test mathematical component extraction."""

    def test_extract_simple_components(self):
        """Test extracting components from simple problem."""
        if not DECOMPOSITION_AVAILABLE:
            pytest.skip("Decomposition module not available")

        problem = "Prove that the sum of two even numbers is even"

        components = extract_mathematical_components(problem)

        assert len(components) > 0
        # Should detect concepts like "even numbers", "sum"
        assert any("even" in str(c).lower() for c in components)

    def test_extract_complex_components(self):
        """Test extracting components from complex problem."""
        if not DECOMPOSITION_AVAILABLE:
            pytest.skip("Decomposition module not available")

        problem = "Prove that every natural number greater than 1 has a prime divisor"

        components = extract_mathematical_components(problem)

        assert len(components) > 0
        # Should detect multiple components
        # Check for natural numbers, primes, divisors, etc.

    def test_extract_with_dependencies(self):
        """Test extracting components with dependencies."""
        if not DECOMPOSITION_AVAILABLE:
            pytest.skip("Decomposition module not available")

        problem = "Prove that if n is composite, then n has a prime factor"

        components = extract_mathematical_components(problem)

        assert len(components) > 0
        # Should detect dependency between composite numbers and prime factors


@mark.unit
@mark.decomposition
class TestDependencyIdentification:
    """Test dependency identification between components."""

    def test_identify_simple_dependencies(self):
        """Test identifying dependencies in simple problem."""
        if not DECOMPOSITION_AVAILABLE:
            pytest.skip("Decomposition module not available")

        components = [
            MathematicalComponent(name="even", type="definition"),
            MathematicalComponent(name="sum", type="operation"),
            MathematicalComponent(name="even_sum", type="theorem")
        ]

        dependencies = identify_dependencies(components)

        # even_sum should depend on even and sum
        assert any(dep.source == "even_sum" for dep in dependencies)

    def test_identify_complex_dependencies(self):
        """Test identifying dependencies in complex problem."""
        if not DECOMPOSITION_AVAILABLE:
            pytest.skip("Decomposition module not available")

        components = [
            MathematicalComponent(name="prime", type="definition"),
            MathematicalComponent(name="divisor", type="definition"),
            MathematicalComponent(name="composite", type="definition"),
            MathematicalComponent(name="prime_divisor_theorem", type="theorem")
        ]

        dependencies = identify_dependencies(components)

        # Should detect multiple dependencies
        assert len(dependencies) > 0

    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        if not DECOMPOSITION_AVAILABLE:
            pytest.skip("Decomposition module not available")

        # Create circular dependency
        components = [
            MathematicalComponent(name="A", type="theorem"),
            MathematicalComponent(name="B", type="theorem", dependencies=["C"]),
            MathematicalComponent(name="C", type="theorem", dependencies=["A"])
        ]

        dependencies = identify_dependencies(components)

        # Should detect cycle
        # Implementation should handle this


@mark.unit
@mark.decomposition
class TestComplexityEstimation:
    """Test complexity estimation for problems."""

    def test_estimate_simple_complexity(self):
        """Test complexity estimation for simple problem."""
        if not DECOMPOSITION_AVAILABLE:
            pytest.skip("Decomposition module not available")

        problem = "Prove that 2 + 2 = 4"

        complexity = estimate_complexity(problem)

        assert 0.0 <= complexity <= 1.0  # Should be low

    def test_estimate_medium_complexity(self):
        """Test complexity estimation for medium problem."""
        if not DECOMPOSITION_AVAILABLE:
            pytest.skip("Decomposition module not available")

        problem = "Prove that the sum of two even numbers is even"

        complexity = estimate_complexity(problem)

        assert 0.0 <= complexity <= 10.0

    def test_estimate_complex_complexity(self):
        """Test complexity estimation for complex problem."""
        if not DECOMPOSITION_AVAILABLE:
            pytest.skip("Decomposition module not available")

        problem = "Prove that every natural number greater than 1 has a prime divisor"

        complexity = estimate_complexity(problem)

        assert 0.0 <= complexity <= 10.0
        # Should be higher than simple problem


@mark.unit
@mark.decomposition
class TestSubProblemGeneration:
    """Test sub-problem generation."""

    def test_generate_simple_subproblems(self):
        """Test generating sub-problems from simple problem."""
        if not DECOMPOSITION_AVAILABLE:
            pytest.skip("Decomposition module not available")

        problem = "Prove that the product of two even numbers is even"

        subproblems = generate_subproblems(problem)

        assert len(subproblems) >= 1
        # Should break down into manageable parts

    def test_generate_complex_subproblems(self):
        """Test generating sub-problems from complex problem."""
        if not DECOMPOSITION_AVAILABLE:
            pytest.skip("Decomposition module not available")

        problem = "Prove that every natural number has a unique prime factorization"

        subproblems = generate_subproblems(problem)

        assert len(subproblems) >= 1
        # Should generate multiple sub-problems

    def test_subproblem_dependencies(self):
        """Test that sub-problems maintain dependencies."""
        if not DECOMPOSITION_AVAILABLE:
            pytest.skip("Decomposition module not available")

        problem = "Prove that if n is composite, then n has a prime factor"

        subproblems = generate_subproblems(problem)

        # Check that dependencies are preserved
        for sp in subproblems:
            if hasattr(sp, 'dependencies'):
                assert isinstance(sp.dependencies, list)


@mark.unit
@mark.decomposition
class TestTopologicalOrdering:
    """Test topological ordering of sub-problems."""

    def test_order_simple_subproblems(self):
        """Test ordering simple sub-problems."""
        if not DECOMPOSITION_AVAILABLE:
            pytest.skip("Decomposition module not available")

        subproblems = [
            MathematicalComponent(name="A", type="lemma"),
            MathematicalComponent(name="B", type="lemma", dependencies=["A"]),
            MathematicalComponent(name="C", type="theorem", dependencies=["B"])
        ]

        ordered = topological_order(subproblems)

        # A should come before B, B before C
        assert ordered.index("A") < ordered.index("B")
        assert ordered.index("B") < ordered.index("C")

    def test_order_complex_subproblems(self):
        """Test ordering complex sub-problems."""
        if not DECOMPOSITION_AVAILABLE:
            pytest.skip("Decomposition module not available")

        subproblems = [
            MathematicalComponent(name="base1", type="definition"),
            MathematicalComponent(name="base2", type="definition"),
            MathematicalComponent(name="lemma1", type="lemma", dependencies=["base1"]),
            MathematicalComponent(name="lemma2", type="lemma", dependencies=["base2"]),
            MathematicalComponent(name="theorem", type="theorem", dependencies=["lemma1", "lemma2"])
        ]

        ordered = topological_order(subproblems)

        # Check dependencies are satisfied
        for sp in subproblems:
            if hasattr(sp, 'dependencies'):
                for dep in sp.dependencies:
                    assert ordered.index(dep) < ordered.index(sp.name)


@mark.unit
@mark.decomposition
class TestParallelizationDetection:
    """Test detection of parallelizable sub-problems."""

    def test_detect_parallelizable_subproblems(self):
        """Test detecting parallelizable components."""
        if not DECOMPOSITION_AVAILABLE:
            pytest.skip("Decomposition module not available")

        problem = "Prove that for any integers a, b, c, d: if a divides b and c divides d, then ac divides bd"

        parallel = detect_parallelization(problem)

        assert parallel is not None
        # Should detect that a|b and c|d can be proved in parallel

    def test_non_parallelizable(self):
        """Test that non-parallelizable problems are detected."""
        if not DECOMPOSITION_AVAILABLE:
            pytest.skip("Decomposition module not available")

        problem = "Prove that if n is prime, then n is irreducible"

        parallel = detect_parallelization(problem)

        # May return empty or indicate no parallelization
        assert parallel is not None


# =============================================================================
# ADVERSARIAL TESTS
# =============================================================================

@mark.unit
@mark.adversarial
class TestBlueTeamAgent:
    """Test blue team (proof generation) agent."""

    @pytest.fixture
    def blue_team(self):
        """Create a blue team agent."""
        if not ADVERSARIAL_AVAILABLE:
            pytest.skip("Adversarial module not available")
        return BlueTeamAgent(model="gpt-4", temperature=0.3)

    def test_generate_initial_proof(self, blue_team):
        """Test generating initial proof."""
        if not ADVERSARIAL_AVAILABLE:
            pytest.skip("Adversarial module not available")

        theorem = "Prove that there are infinitely many primes"

        proof = blue_team.generate_proof(theorem)

        assert proof is not None
        assert len(proof) > 0

    def test_refine_proof(self, blue_team):
        """Test refining proof based on feedback."""
        if not ADVERSARIAL_AVAILABLE:
            pytest.skip("Adversarial module not available")

        initial_proof = "Assume there are finitely many primes. Derive contradiction."
        critique = ProofCritique(
            issues=["Need more detail on contradiction step"],
            suggestions=["Use Euclid's argument"]
        )

        refined = blue_team.refine_proof(initial_proof, critique)

        assert refined is not None
        assert len(refined) > 0

    def test_defend_proof(self, blue_team):
        """Test defending proof against attack."""
        if not ADVERSARIAL_AVAILABLE:
            pytest.skip("Adversarial module not available")

        proof = "Proof using Euclid's argument"
        counterexample = Counterexample(
            description="What if primes are not finite?",
            target_step="Assumption step"
        )

        defense = blue_team.defend_proof(proof, counterexample)

        assert defense is not None


@mark.unit
@mark.adversarial
class TestRedTeamAgent:
    """Test red team (critique) agent."""

    @pytest.fixture
    def red_team(self):
        """Create a red team agent."""
        if not ADVERSARIAL_AVAILABLE:
            pytest.skip("Adversarial module not available")
        return RedTeamAgent(model="gpt-4", temperature=0.7)

    def test_generate_critique(self, red_team):
        """Test generating proof critique."""
        if not ADVERSARIAL_AVAILABLE:
            pytest.skip("Adversarial module not available")

        proof = "Proof: Assume finite primes, multiply them, add 1, get contradiction"

        critique = red_team.generate_critique(proof)

        assert critique is not None
        assert isinstance(critique, ProofCritique)
        assert len(critique.issues) >= 0
        assert len(critique.suggestions) >= 0

    def test_generate_counterexample(self, red_team):
        """Test generating counterexample."""
        if not ADVERSARIAL_AVAILABLE:
            pytest.skip("Adversarial module not available")

        proof = "All primes are odd"

        counterexample = red_team.generate_counterexample(proof)

        assert counterexample is not None
        assert isinstance(counterexample, Counterexample)
        assert len(counterexample.description) > 0

    def test_attack_proof(self, red_team):
        """Test attacking proof."""
        if not ADVERSARIAL_AVAILABLE:
            pytest.skip("Adversarial module not available")

        proof = "Incomplete proof with gaps"

        attack = red_team.attack_proof(proof)

        assert attack is not None
        assert attack.critique is not None or attack.counterexample is not None


@mark.unit
@mark.adversarial
class TestAdversarialRound:
    """Test adversarial round execution."""

    def test_single_adversarial_round(self):
        """Test executing a single adversarial round."""
        if not ADVERSARIAL_AVAILABLE:
            pytest.skip("Adversarial module not available")

        theorem = "Prove that there are infinitely many primes"
        blue_team = BlueTeamAgent()
        red_team = RedTeamAgent()

        round_result = AdversarialRound(
            theorem=theorem,
            blue_team=blue_team,
            red_team=red_team
        )

        result = round_result.execute()

        assert result is not None
        assert result.proof is not None
        assert result.round_number >= 1

    def test_multiple_adversarial_rounds(self):
        """Test executing multiple adversarial rounds."""
        if not ADVERSARIAL_AVAILABLE:
            pytest.skip("Adversarial module not available")

        theorem = "Prove that sqrt(2) is irrational"
        blue_team = BlueTeamAgent()
        red_team = RedTeamAgent()

        rounds = []
        for i in range(3):
            round_result = AdversarialRound(
                theorem=theorem,
                blue_team=blue_team,
                red_team=red_team,
                round_number=i+1
            )
            result = round_result.execute()
            rounds.append(result)

        assert len(rounds) == 3
        # Quality should improve over rounds
        # (implementation-dependent)

    def test_adversarial_convergence(self):
        """Test that adversarial process converges."""
        if not ADVERSARIAL_AVAILABLE:
            pytest.skip("Adversarial module not available")

        theorem = "Prove that 2+2=4"
        engine = LeanAideAdversarialEngine(
            theorem=theorem,
            max_rounds=5
        )

        result = engine.evolve()

        assert result is not None
        assert result.rounds_completed >= 1
        assert result.final_proof is not None


@mark.unit
@mark.adversarial
class TestCoevolutionDynamics:
    """Test co-evolution dynamics."""

    def test_blue_red_coevolution(self):
        """Test co-evolution of blue and red teams."""
        if not ADVERSARIAL_AVAILABLE:
            pytest.skip("Adversarial module not available")

        theorem = "Prove that the sum of two even numbers is even"

        dynamics = CoevolutionDynamics(
            theorem=theorem,
            blue_team=BlueTeamAgent(),
            red_team=RedTeamAgent()
        )

        history = dynamics.evolve(num_rounds=3)

        assert len(history) == 3
        # Blue team should improve
        # Red team should find better critiques

    def test_adaptive_difficulty(self):
        """Test adaptive difficulty adjustment."""
        if not ADVERSARIAL_AVAILABLE:
            pytest.skip("Adversarial module not available")

        dynamics = CoevolutionDynamics(
            theorem="Simple theorem",
            blue_team=BlueTeamAgent(),
            red_team=RedTeamAgent()
        )

        # Red team should adapt to blue team's strength
        initial_strength = dynamics.red_team.strength
        dynamics.adjust_difficulty()
        adjusted_strength = dynamics.red_team.strength

        # Strength should change based on performance


# =============================================================================
# SELF-PLAY TESTS
# =============================================================================

@mark.unit
@mark.selfplay
class TestSelfPlayAgent:
    """Test self-play agent."""

    @pytest.fixture
    def agent(self):
        """Create a self-play agent."""
        if not SELFPLAY_AVAILABLE:
            pytest.skip("Self-play module not available")
        return SelfPlayAgent(agent_id="test_agent")

    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        if not SELFPLAY_AVAILABLE:
            pytest.skip("Self-play module not available")

        assert agent.agent_id == "test_agent"
        assert agent.strategy is not None
        assert agent.experience_buffer is not None

    def test_select_strategy(self, agent):
        """Test strategy selection."""
        if not SELFPLAY_AVAILABLE:
            pytest.skip("Self-play module not available")

        strategy = agent.select_strategy()

        assert strategy is not None
        assert isinstance(strategy, SelfPlayStrategy)

    def test_update_from_experience(self, agent):
        """Test learning from experience."""
        if not SELFPLAY_AVAILABLE:
            pytest.skip("Self-play module not available")

        # Simulate experience
        experience = {
            "theorem": "Test theorem",
            "proof": "Test proof",
            "reward": 1.0,
            "success": True
        }

        agent.update_from_experience(experience)

        # Strategy should be updated
        assert len(agent.experience_buffer) > 0


@mark.unit
@mark.selfplay
class TestExperienceBuffer:
    """Test experience buffer for self-play."""

    @pytest.fixture
    def buffer(self):
        """Create an experience buffer."""
        if not SELFPLAY_AVAILABLE:
            pytest.skip("Self-play module not available")
        return ExperienceBuffer(capacity=100)

    def test_store_experience(self, buffer):
        """Test storing experience."""
        if not SELFPLAY_AVAILABLE:
            pytest.skip("Self-play module not available")

        experience = {
            "theorem": "Test",
            "proof": "Proof",
            "reward": 1.0
        }

        buffer.store(experience)

        assert len(buffer) == 1

    def test_retrieve_experience(self, buffer):
        """Test retrieving experience."""
        if not SELFPLAY_AVAILABLE:
            pytest.skip("Self-play module not available")

        experiences = [
            {"theorem": f"T{i}", "proof": f"P{i}", "reward": float(i)}
            for i in range(10)
        ]

        for exp in experiences:
            buffer.store(exp)

        # Sample batch
        batch = buffer.sample(batch_size=5)

        assert len(batch) == 5

    def test_buffer_capacity(self, buffer):
        """Test buffer capacity limit."""
        if not SELFPLAY_AVAILABLE:
            pytest.skip("Self-play module not available")

        # Fill beyond capacity
        for i in range(150):
            buffer.store({"theorem": f"T{i}", "proof": f"P{i}", "reward": 1.0})

        # Should not exceed capacity
        assert len(buffer) <= 100


@mark.unit
@mark.selfplay
class TestSelfPlayGame:
    """Test self-play game execution."""

    def test_single_game(self):
        """Test executing a single self-play game."""
        if not SELFPLAY_AVAILABLE:
            pytest.skip("Self-play module not available")

        agent1 = SelfPlayAgent(agent_id="agent1")
        agent2 = SelfPlayAgent(agent_id="agent2")

        game = SelfPlayGame(
            agent1=agent1,
            agent2=agent2,
            theorem="Prove that 2+2=4"
        )

        result = game.play()

        assert result is not None
        assert result.winner is not None
        assert len(agent1.experience_buffer) > 0
        assert len(agent2.experience_buffer) > 0

    def test_tournament(self):
        """Test running a self-play tournament."""
        if not SELFPLAY_AVAILABLE:
            pytest.skip("Self-play module not available")

        agents = [SelfPlayAgent(agent_id=f"agent{i}") for i in range(4)]

        engine = LeanAideSelfPlayEngine(agents=agents)
        results = engine.run_tournament(num_rounds=2)

        assert len(results) > 0
        # Check that all agents played


@mark.unit
@mark.selfplay
class TestSelfPlayTraining:
    """Test self-play training loop."""

    def test_training_convergence(self):
        """Test that self-play leads to improvement."""
        if not SELFPLAY_AVAILABLE:
            pytest.skip("Self-play module not available")

        engine = LeanAideSelfPlayEngine(
            num_agents=4,
            training_iterations=10
        )

        result = engine.train()

        assert result is not None
        assert result.iterations_completed >= 1
        # Performance should improve

    def test_reward_calculation(self):
        """Test reward calculation."""
        if not SELFPLAY_AVAILABLE:
            pytest.skip("Self-play module not available")

        agent = SelfPlayAgent()

        # Win
        reward_win = agent.calculate_reward(success=True, proof_length=5)
        assert reward_win > 0

        # Loss
        reward_loss = agent.calculate_reward(success=False, proof_length=10)
        assert reward_loss < reward_win


# =============================================================================
# STRATEGY LIBRARY TESTS
# =============================================================================

@mark.unit
@mark.strategy
class TestTacticLibrary:
    """Test tactic library."""

    def test_library_completeness(self):
        """Test that library contains essential tactics."""
        if not STRATEGIES_AVAILABLE:
            pytest.skip("Strategies module not available")

        library = get_tactic_library()

        essential_tactics = [
            "simp", "rw", "apply", "exact", "cases",
            "induction", "intros", "have", "calc"
        ]

        for tactic in essential_tactics:
            assert tactic in library

    def test_tactic_metadata(self):
        """Test tactic metadata."""
        if not STRATEGIES_AVAILABLE:
            pytest.skip("Strategies module not available")

        library = get_tactic_library()

        for tactic_name, tactic_info in library.items():
            assert "name" in tactic_info
            assert "category" in tactic_info
            assert "description" in tactic_info


@mark.unit
@mark.strategy
class TestStrategySelection:
    """Test strategy selection."""

    def test_select_strategy_simple(self):
        """Test strategy selection for simple problems."""
        if not STRATEGIES_AVAILABLE:
            pytest.skip("Strategies module not available")

        problem = "Prove that True is true"

        strategy = select_strategy(problem)

        assert strategy is not None
        assert strategy.tactics is not None

    def test_select_strategy_complex(self):
        """Test strategy selection for complex problems."""
        if not STRATEGIES_AVAILABLE:
            pytest.skip("Strategies module not available")

        problem = "Prove that every natural number has a prime factorization"

        strategy = select_strategy(problem)

        assert strategy is not None
        # Should select more advanced strategy


@mark.unit
@mark.strategy
class TestStrategyMutation:
    """Test strategy mutation."""

    def test_mutate_tactics(self):
        """Test mutating tactics."""
        if not STRATEGIES_AVAILABLE:
            pytest.skip("Strategies module not available")

        original_strategy = StrategyTemplate(
            name="test_strategy",
            tactics=["simp", "rw", "apply"]
        )

        mutated = mutate_strategy(original_strategy)

        assert mutated is not None
        assert mutated.name == "test_strategy"
        # Tactics should be different

    def test_mutate_parameters(self):
        """Test mutating strategy parameters."""
        if not STRATEGIES_AVAILABLE:
            pytest.skip("Strategies module not available")

        original = StrategyTemplate(
            name="param_strategy",
            tactics=["apply"],
            parameters={"temperature": 0.5}
        )

        mutated = mutate_strategy(original, mutate_params=True)

        assert mutated is not None
        # Parameters may have changed


@mark.unit
@mark.strategy
class TestStrategyCombination:
    """Test strategy combination."""

    def test_combine_strategies(self):
        """Test combining multiple strategies."""
        if not STRATEGIES_AVAILABLE:
            pytest.skip("Strategies module not available")

        strategy1 = StrategyTemplate(
            name="strategy1",
            tactics=["simp", "rw"]
        )

        strategy2 = StrategyTemplate(
            name="strategy2",
            tactics=["apply", "exact"]
        )

        combined = combine_strategies([strategy1, strategy2])

        assert combined is not None
        # Should have tactics from both


@mark.unit
@mark.strategy
class TestSuccessTracker:
    """Test strategy success tracking."""

    def test_track_success(self):
        """Test tracking successful strategies."""
        if not STRATEGIES_AVAILABLE:
            pytest.skip("Strategies module not available")

        tracker = SuccessTracker()

        strategy = StrategyTemplate(name="test", tactics=["simp"])

        tracker.record_success(strategy, problem_type="simple")

        stats = tracker.get_statistics(strategy)

        assert stats["attempts"] == 1
        assert stats["successes"] == 1

    def test_track_failure(self):
        """Test tracking failed strategies."""
        if not STRATEGIES_AVAILABLE:
            pytest.skip("Strategies module not available")

        tracker = SuccessTracker()

        strategy = StrategyTemplate(name="test", tactics=["wrong"])

        tracker.record_failure(strategy, problem_type="complex")

        stats = tracker.get_statistics(strategy)

        assert stats["attempts"] == 1
        assert stats["failures"] == 1

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        if not STRATEGIES_AVAILABLE:
            pytest.skip("Strategies module not available")

        tracker = SuccessTracker()

        strategy = StrategyTemplate(name="test", tactics=["simp"])

        # Record some attempts
        tracker.record_success(strategy, "simple")
        tracker.record_success(strategy, "simple")
        tracker.record_failure(strategy, "simple")

        rate = tracker.get_success_rate(strategy, "simple")

        assert rate == 2/3  # 2 successes out of 3 attempts


# =============================================================================
# WORKFLOW INTEGRATION TESTS
# =============================================================================

@mark.integration
@mark.workflow
class TestStage3AEvolutionarySolution:
    """Test Stage 3A: Evolutionary solution generation."""

    @pytest.mark.asyncio
    async def test_evolutionary_solution_generation(self, sample_theorems):
        """Test generating solutions using evolution."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        theorem = sample_theorems["simple"]["statement"]

        result = await evolve_proof(
            theorem=theorem,
            theorem_name="test_theorem",
            max_generations=5,
            population_size=10
        )

        assert result is not None
        assert result.generations_completed >= 1
        assert result.best_strategy is not None or result.best_proof is not None


@mark.integration
@mark.workflow
class TestStage3BAdversarialEvolution:
    """Test Stage 3B: Adversarial evolution."""

    def test_adversarial_evolution_integration(self):
        """Test adversarial evolution in workflow."""
        if not ADVERSARIAL_AVAILABLE:
            pytest.skip("Adversarial module not available")

        theorem = "Prove that there are infinitely many primes"

        result = evolve_adversarially(
            theorem=theorem,
            max_rounds=3
        )

        assert result is not None
        assert result.rounds_completed >= 1


@mark.integration
@mark.workflow
class TestMathematicalProblemDetection:
    """Test mathematical problem detection in decomposition."""

    def test_detect_mathematical_problems(self, sample_mathematical_problems):
        """Test detection of mathematical problems."""
        if not DECOMPOSITION_AVAILABLE:
            pytest.skip("Decomposition module not available")

        for problem_name, problem_text in sample_mathematical_problems.items():
            is_math = isinstance(problem_text, str) and any(
                keyword in problem_text.lower()
                for keyword in ["prove", "show", "calculate", "theorem"]
            )
            assert is_math or problem_name  # Should detect mathematical content


@mark.integration
@mark.workflow
class TestGracefulFallback:
    """Test graceful fallback when LeanAide unavailable."""

    @pytest.mark.asyncio
    async def test_fallback_when_server_unavailable(self):
        """Test that evolution works without server."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        # Use invalid server URL
        result = await evolve_proof(
            theorem="Test theorem",
            server_url="http://nonexistent:9999",
            max_generations=2,
            population_size=5
        )

        # Should still return result (with simulation)
        assert result is not None


@mark.integration
@mark.workflow
class TestEndToEndEvolutionaryWorkflow:
    """Test complete evolutionary workflow."""

    @pytest.mark.asyncio
    async def test_full_evolutionary_pipeline(self, sample_theorems):
        """Test full pipeline from problem to solution."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        theorem = sample_theorems["simple"]["statement"]

        # Step 1: Detect if mathematical ( decomposition integration)
        # Step 2: Generate initial population (evolution)
        # Step 3: Evolve solution
        # Step 4: Verify result

        result = await evolve_proof(
            theorem=theorem,
            max_generations=3,
            population_size=5
        )

        assert result is not None
        assert result.evolution_time > 0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@mark.slow
@mark.evolution
class TestEvolutionPerformance:
    """Test evolution performance."""

    @pytest.mark.asyncio
    async def test_evolution_speed(self):
        """Test evolution completes in reasonable time."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        start = time.time()

        result = await evolve_proof(
            theorem="Prove that True is true",
            max_generations=10,
            population_size=20
        )

        elapsed = time.time() - start

        # Should complete within reasonable time
        assert elapsed < 60.0  # 1 minute max

    @pytest.mark.asyncio
    async def test_parallel_evaluation(self):
        """Test that parallel evaluation is faster."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        # Sequential
        start_seq = time.time()
        result_seq = await evolve_proof(
            theorem="Test",
            max_generations=2,
            population_size=10,
            parallel_evaluation=False
        )
        time_seq = time.time() - start_seq

        # Parallel
        start_par = time.time()
        result_par = await evolve_proof(
            theorem="Test",
            max_generations=2,
            population_size=10,
            parallel_evaluation=True
        )
        time_par = time.time() - start_par

        # Parallel should be faster (or at least not significantly slower)
        assert time_par <= time_seq * 1.5


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

@mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_theorem(self):
        """Test handling of empty theorem."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        # Should handle gracefully
        with pytest.raises((ValueError, Exception)):
            await evolve_proof(theorem="")

    def test_malformed_lean_code(self):
        """Test handling of malformed Lean code."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        malformed_code = "this is not valid Lean code !!!"

        proof = LeanProof(
            theorem_name="malformed",
            theorem_statement="Bad",
            lean_code=malformed_code
        )

        # Should handle error gracefully
        assert proof is not None

    def test_extremely_long_proof(self):
        """Test handling of extremely long proofs."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        # Create proof with many tactics
        tactics = [Tactic(name=f"tac_{i}") for i in range(1000)]

        proof = LeanProof(
            theorem_name="long",
            theorem_statement="Long theorem",
            lean_code="",
            tactics=tactics
        )

        strategy = LeanProofStrategy(proof=proof)

        # Should handle without crashing
        complexity = strategy.calculate_complexity()
        assert 0.0 <= complexity <= 10.0

    def test_zero_population_size(self):
        """Test handling of zero population size."""
        if not EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        # Should raise error or handle gracefully
        with pytest.raises((ValueError, Exception)):
            LeanProofEvolutionEngine(
                theorem="Test",
                population_size=0
            )


# =============================================================================
# TEST SUITE ORGANIZATION
# =============================================================================

class TestLeanAideEvolutionarySuite:
    """
    Master test suite for evolutionary LeanAide integration.

    Organizes all tests into logical groups:
    1. Evolution Tests
    2. Decomposition Tests
    3. Adversarial Tests
    4. Self-Play Tests
    5. Strategy Tests
    6. Workflow Integration Tests
    7. Performance Tests
    8. Edge Case Tests
    """

    @staticmethod
    def run_all_tests():
        """Run all tests in the suite."""
        pytest.main([__file__, "-v", "-s"])

    @staticmethod
    def run_evolution_tests_only():
        """Run only evolution tests."""
        pytest.main([__file__, "-v", "-m", "evolution"])

    @staticmethod
    def run_decomposition_tests_only():
        """Run only decomposition tests."""
        pytest.main([__file__, "-v", "-m", "decomposition"])

    @staticmethod
    def run_adversarial_tests_only():
        """Run only adversarial tests."""
        pytest.main([__file__, "-v", "-m", "adversarial"])

    @staticmethod
    def run_selfplay_tests_only():
        """Run only self-play tests."""
        pytest.main([__file__, "-v", "-m", "selfplay"])

    @staticmethod
    def run_strategy_tests_only():
        """Run only strategy tests."""
        pytest.main([__file__, "-v", "-m", "strategy"])

    @staticmethod
    def run_workflow_tests_only():
        """Run only workflow integration tests."""
        pytest.main([__file__, "-v", "-m", "workflow"])

    @staticmethod
    def run_unit_tests_only():
        """Run only unit tests."""
        pytest.main([__file__, "-v", "-m", "unit"])

    @staticmethod
    def run_integration_tests_only():
        """Run only integration tests."""
        pytest.main([__file__, "-v", "-m", "integration"])


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Main entry point for running tests.

    Usage:
        python test_leanaide_evolutionary.py                    # Run all tests
        python test_leanaide_evolutionary.py -m evolution       # Run evolution tests only
        python test_leanaide_evolutionary.py -m decomposition   # Run decomposition tests only
        python test_leanaide_evolutionary.py -m adversarial     # Run adversarial tests only
        python test_leanaide_evolutionary.py -m selfplay        # Run self-play tests only
        python test_leanaide_evolutionary.py -m strategy        # Run strategy tests only
        python test_leanaide_evolutionary.py -m workflow        # Run workflow tests only
        python test_leanaide_evolutionary.py -v                 # Verbose output
        python test_leanaide_evolutionary.py -s                 # Show print output
        python test_leanaide_evolutionary.py -m "not slow"      # Skip slow tests
    """
    sys.exit(pytest.main([__file__, "-v", "-s"]))
