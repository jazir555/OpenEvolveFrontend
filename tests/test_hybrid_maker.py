"""
Comprehensive Test Suite for Hybrid MAKER Integration System

This module provides extensive testing coverage for all hybrid strategies:
- MCTS-Then-MAKER
- MAKER-Then-Evolution
- MAKER-Adversarial
- Adaptive MAKER
- MAKER-MDAP Parallel
- Full MAKER Hybrid

Author: OpenEvolve Hybrid Testing Team
Created: 2025-01-07
Version: 1.0.0
"""

import asyncio
import pytest
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import time

# Import modules to test
try:
    from hybrid_maker_integration import (
        MAKERHybridMode,
        MAKERHybridConfig,
        MCTSThenMAKER,
        MAKERThenEvolution,
        MAKERAdversarialHybrid,
        AdaptiveMAKERHybrid,
        MAKERMDAPParallel,
        FullMAKERHybrid,
        run_maker_hybrid,
        get_maker_hybrid_capabilities,
        EvolutionResult
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_theorem():
    """Sample theorem for testing"""
    return "forall n m : nat, n + m = m + n"

@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return MAKERHybridConfig(
        enable_voting=True,
        voting_threshold=3,
        enable_decomposition=True,
        mcts_simulations=10,
        evolution_generations=5,
        population_size=10,
        adversarial_rounds=2
    )

@pytest.fixture
def sample_evolution_result():
    """Sample evolution result for testing"""
    return EvolutionResult(
        success=True,
        generations_completed=10,
        evolution_time=5.0,
        best_proof="simp\nrw [add_comm]\nrefl",
        best_fitness=0.85,
        convergence_history=[0.5, 0.6, 0.7, 0.8, 0.85]
    )

@pytest.fixture
def temp_data_dir(tmp_path):
    """Temporary directory for test data"""
    data_dir = tmp_path / "hybrid_data"
    data_dir.mkdir()
    return data_dir


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestMAKERHybridConfig:
    """Test suite for MAKERHybridConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = MAKERHybridConfig()

        assert config.enable_voting is True
        assert config.voting_threshold == 3
        assert config.enable_red_flagging is True
        assert config.enable_decomposition is True
        assert config.decomposition_depth == 3
        assert config.mcts_simulations == 100
        assert config.evolution_generations == 20
        assert config.population_size == 20

    def test_custom_config(self):
        """Test custom configuration values"""
        config = MAKERHybridConfig(
            voting_threshold=5,
            mcts_simulations=50,
            evolution_generations=30
        )

        assert config.voting_threshold == 5
        assert config.mcts_simulations == 50
        assert config.evolution_generations == 30

    def test_to_dict(self):
        """Test config to dictionary conversion"""
        config = MAKERHybridConfig(voting_threshold=7)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["voting_threshold"] == 7
        assert config_dict["enable_voting"] is True

    def test_from_dict(self):
        """Test config from dictionary creation"""
        config_dict = {
            "voting_threshold": 5,
            "mcts_simulations": 75,
            "enable_decomposition": False
        }

        config = MAKERHybridConfig(**config_dict)

        assert config.voting_threshold == 5
        assert config.mcts_simulations == 75
        assert config.enable_decomposition is False


# =============================================================================
# MCTS-THEN-MAKER TESTS
# =============================================================================

class TestMCTSThenMAKER:
    """Test suite for MCTSThenMAKER hybrid strategy"""

    @pytest.fixture
    def strategy(self):
        """Create MCTSThenMAKER instance"""
        return MCTSThenMAKER(
            mcts_simulations=10,
            maker_voting_threshold=3,
            population_size=5
        )

    def test_initialization(self, strategy):
        """Test strategy initialization"""
        assert strategy.name == "MCTS_Then_MAKER"
        assert strategy.mcts_simulations == 10
        assert strategy.maker_voting_threshold == 3
        assert strategy.population_size == 5

    def test_generate_proof_mock(self, strategy, sample_theorem):
        """Test proof generation with mocked components"""
        with patch('hybrid_maker_integration.MCTS_AVAILABLE', False):
            # Mock MCTS not available
            result = asyncio.run(strategy.generate_proof(sample_theorem))

            assert isinstance(result, EvolutionResult)
            assert result.success is False  # Should fail without MCTS

    def test_evaluate_sequence(self, strategy):
        """Test sequence evaluation"""
        # Mock action
        mock_action = Mock()
        mock_action.tactic.name = "simp"
        mock_action.tactic.arguments = []

        sequence = [mock_action] * 5
        fitness = strategy._evaluate_sequence(sequence)

        assert 0.0 <= fitness <= 1.0

    def test_sequence_to_string(self, strategy):
        """Test sequence to string conversion"""
        # Mock actions
        actions = []
        for i in range(3):
            action = Mock()
            action.tactic.name = f"tactic{i}"
            action.tactic.arguments = [f"arg{i}"] if i > 0 else []
            actions.append(action)

        result = strategy._sequence_to_string(actions)

        assert isinstance(result, str)
        assert len(result) > 0


# =============================================================================
# MAKER-THEN-EVOLUTION TESTS
# =============================================================================

class TestMAKERThenEvolution:
    """Test suite for MAKERThenEvolution hybrid strategy"""

    @pytest.fixture
    def strategy(self):
        """Create MAKERThenEvolution instance"""
        return MAKERThenEvolution(
            maker_voting_threshold=3,
            evolution_generations=5,
            population_size=10,
            initial_candidates=20
        )

    def test_initialization(self, strategy):
        """Test strategy initialization"""
        assert strategy.name == "MAKER_Then_Evolution"
        assert strategy.maker_voting_threshold == 3
        assert strategy.evolution_generations == 5
        assert strategy.population_size == 10

    def test_generate_candidate(self, strategy, sample_theorem):
        """Test candidate generation"""
        candidate = strategy._generate_candidate(sample_theorem, seed=42)

        assert isinstance(candidate, str)
        assert sample_theorem in candidate or "theorem" in candidate

    def test_evaluate_candidate(self, strategy):
        """Test candidate evaluation"""
        candidate = "theorem : test\nby\n  simp\n  rw [add_comm]"
        fitness = strategy._evaluate_candidate(candidate)

        assert 0.0 <= fitness <= 1.0

    def test_evaluate_empty_candidate(self, strategy):
        """Test evaluation of empty candidate"""
        fitness = strategy._evaluate_candidate("")
        assert fitness == 0.0


# =============================================================================
# MAKER-ADVERSARIAL TESTS
# =============================================================================

class TestMAKERAdversarialHybrid:
    """Test suite for MAKERAdversarialHybrid strategy"""

    @pytest.fixture
    def strategy(self):
        """Create MAKERAdversarialHybrid instance"""
        return MAKERAdversarialHybrid(
            adversarial_rounds=3,
            maker_voting_threshold=3,
            red_team_size=2,
            blue_team_size=2
        )

    def test_initialization(self, strategy):
        """Test strategy initialization"""
        assert strategy.name == "MAKER_Adversarial_Hybrid"
        assert strategy.adversarial_rounds == 3
        assert strategy.red_team_size == 2
        assert strategy.blue_team_size == 2

    def test_evaluate_defense_with_effectiveness(self, strategy):
        """Test defense evaluation with effectiveness attribute"""
        defense = Mock()
        defense.effectiveness = 0.9

        fitness = strategy._evaluate_defense(defense)

        assert fitness == 0.9

    def test_evaluate_defense_without_effectiveness(self, strategy):
        """Test defense evaluation without effectiveness attribute"""
        defense = Mock()
        delattr(defense, 'effectiveness')

        fitness = strategy._evaluate_defense(defense)

        assert 0.5 <= fitness <= 1.0


# =============================================================================
# ADAPTIVE MAKER TESTS
# =============================================================================

class TestAdaptiveMAKERHybrid:
    """Test suite for AdaptiveMAKERHybrid strategy"""

    @pytest.fixture
    def strategy(self):
        """Create AdaptiveMAKERHybrid instance"""
        return AdaptiveMAKERHybrid(
            diversity_threshold=0.3,
            convergence_threshold=0.95,
            max_generations=10
        )

    def test_initialization(self, strategy):
        """Test strategy initialization"""
        assert strategy.name == "Adaptive_MAKER_Hybrid"
        assert strategy.diversity_threshold == 0.3
        assert strategy.convergence_threshold == 0.95
        assert strategy.max_generations == 10

    def test_generate_candidate(self, strategy, sample_theorem):
        """Test candidate generation"""
        candidate = strategy._generate_candidate(sample_theorem, seed=42)

        assert isinstance(candidate, str)
        assert len(candidate) > 0

    def test_evaluate_candidate(self, strategy):
        """Test candidate evaluation"""
        candidate = "theorem : test\nby\n  simp\n  refl"
        fitness = strategy._evaluate_candidate(candidate)

        assert 0.0 <= fitness <= 1.0

    def test_mutate_individual(self, strategy):
        """Test individual mutation"""
        # Mock individual
        individual = Mock()
        individual.genome = "base_genome"
        individual.fitness = 0.5
        individual.generation = 0
        individual.metadata = {"test": "data"}

        mutated = strategy._mutate_individual(individual)

        assert mutated.genome != individual.genome
        assert mutated.generation == 1

    def test_crossover(self, strategy):
        """Test crossover operation"""
        parent1 = Mock()
        parent1.genome = "simp\nrw\nrefl"
        parent1.fitness = 0.7
        parent1.generation = 0
        parent1.metadata = {"id": 1}

        parent2 = Mock()
        parent2.genome = "induction\nsimp\nassumption"
        parent2.fitness = 0.8
        parent2.generation = 0
        parent2.metadata = {"id": 2}

        child = strategy._crossover(parent1, parent2)

        assert child.generation == 1
        assert isinstance(child.genome, str)
        assert "parent1" in child.metadata
        assert "parent2" in child.metadata


# =============================================================================
# MAKER-MDAP PARALLEL TESTS
# =============================================================================

class TestMAKERMDAPParallel:
    """Test suite for MAKERMDAPParallel strategy"""

    @pytest.fixture
    def strategy(self):
        """Create MAKERMDAPParallel instance"""
        return MAKERMDAPParallel(
            maker_voting_threshold=3,
            mdap_agents=4,
            combination_method="best_fitness"
        )

    def test_initialization(self, strategy):
        """Test strategy initialization"""
        assert strategy.name == "MAKER_MDAP_Parallel"
        assert strategy.maker_voting_threshold == 3
        assert strategy.mdap_agents == 4
        assert strategy.combination_method == "best_fitness"

    def test_generate_candidate(self, strategy, sample_theorem):
        """Test candidate generation"""
        candidate = strategy._generate_candidate(sample_theorem, seed=42)

        assert isinstance(candidate, str)
        assert len(candidate) > 0

    def test_evaluate_candidate(self, strategy):
        """Test candidate evaluation"""
        candidate = "simp\nrw\nrefl"
        fitness = strategy._evaluate_candidate(candidate)

        assert 0.0 <= fitness <= 1.0


# =============================================================================
# FULL MAKER HYBRID TESTS
# =============================================================================

class TestFullMAKERHybrid:
    """Test suite for FullMAKERHybrid strategy"""

    @pytest.fixture
    def strategy(self, sample_config):
        """Create FullMAKERHybrid instance"""
        return FullMAKERHybrid(config=sample_config)

    def test_initialization(self, strategy, sample_config):
        """Test strategy initialization"""
        assert strategy.name == "Full_MAKER_Hybrid"
        assert strategy.config == sample_config

    def test_generate_proof_mock(self, strategy, sample_theorem):
        """Test proof generation with all phases mocked"""
        # Mock all availability flags
        with patch('hybrid_maker_integration.MAKER_EVOLUTION_AVAILABLE', False):
            with patch('hybrid_maker_integration.MAKER_ADVERSARIAL_AVAILABLE', False):
                result = asyncio.run(strategy.generate_proof(sample_theorem))

                assert isinstance(result, EvolutionResult)


# =============================================================================
# MAIN ENTRY POINT TESTS
# =============================================================================

class TestMainEntryPoint:
    """Test suite for main entry point functions"""

    def test_run_maker_hybrid_mcts_mode(self, sample_theorem, sample_config):
        """Test run_maker_hybrid with MCTS mode"""
        with patch('hybrid_maker_integration.MCTS_AVAILABLE', False):
            result = asyncio.run(run_maker_hybrid(
                theorem=sample_theorem,
                mode=MAKERHybridMode.MCTS_THEN_MAKER,
                config=sample_config
            ))

            assert isinstance(result, EvolutionResult)

    def test_run_maker_hybrid_evolution_mode(self, sample_theorem, sample_config):
        """Test run_maker_hybrid with Evolution mode"""
        with patch('hybrid_maker_integration.MAKER_EVOLUTION_AVAILABLE', False):
            result = asyncio.run(run_maker_hybrid(
                theorem=sample_theorem,
                mode=MAKERHybridMode.MAKER_THEN_EVOLUTION,
                config=sample_config
            ))

            assert isinstance(result, EvolutionResult)

    def test_run_maker_hybrid_adversarial_mode(self, sample_theorem, sample_config):
        """Test run_maker_hybrid with Adversarial mode"""
        with patch('hybrid_maker_integration.MAKER_ADVERSARIAL_AVAILABLE', False):
            result = asyncio.run(run_maker_hybrid(
                theorem=sample_theorem,
                mode=MAKERHybridMode.MAKER_ADVERSARIAL,
                config=sample_config
            ))

            assert isinstance(result, EvolutionResult)

    def test_run_maker_hybrid_adaptive_mode(self, sample_theorem, sample_config):
        """Test run_maker_hybrid with Adaptive mode"""
        result = asyncio.run(run_maker_hybrid(
            theorem=sample_theorem,
            mode=MAKERHybridMode.ADAPTIVE_MAKER,
            config=sample_config
        ))

        assert isinstance(result, EvolutionResult)

    def test_run_maker_hybrid_parallel_mode(self, sample_theorem, sample_config):
        """Test run_maker_hybrid with Parallel mode"""
        with patch('hybrid_maker_integration.MDAP_AVAILABLE', False):
            result = asyncio.run(run_maker_hybrid(
                theorem=sample_theorem,
                mode=MAKERHybridMode.MAKER_MDAP_PARALLEL,
                config=sample_config
            ))

            assert isinstance(result, EvolutionResult)

    def test_run_maker_hybrid_full_mode(self, sample_theorem, sample_config):
        """Test run_maker_hybrid with Full mode"""
        with patch('hybrid_maker_integration.MAKER_EVOLUTION_AVAILABLE', False):
            with patch('hybrid_maker_integration.MAKER_ADVERSARIAL_AVAILABLE', False):
                result = asyncio.run(run_maker_hybrid(
                    theorem=sample_theorem,
                    mode=MAKERHybridMode.FULL_MAKER_HYBRID,
                    config=sample_config
                ))

                assert isinstance(result, EvolutionResult)

    def test_get_maker_hybrid_capabilities(self):
        """Test capability detection"""
        capabilities = get_maker_hybrid_capabilities()

        assert isinstance(capabilities, dict)
        assert "maker_hybrid_enabled" in capabilities
        assert "modes" in capabilities
        assert "strategies" in capabilities
        assert "paper" in capabilities

        # Check paper info
        assert capabilities["paper"]["arxiv"] == "2511.09030"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestHybridIntegration:
    """Integration tests for complete hybrid workflows"""

    @pytest.mark.asyncio
    async def test_mcts_then_maker_workflow(self, sample_theorem):
        """Test complete MCTS-Then-MAKER workflow"""
        strategy = MCTSThenMAKER(
            mcts_simulations=5,
            maker_voting_threshold=3
        )

        # Mock components not available
        with patch('hybrid_maker_integration.MCTS_AVAILABLE', False):
            result = await strategy.generate_proof(sample_theorem)

            assert isinstance(result, EvolutionResult)
            assert "generations_completed" in result.__dict__

    @pytest.mark.asyncio
    async def test_adaptive_workflow(self, sample_theorem):
        """Test complete adaptive MAKER workflow"""
        strategy = AdaptiveMAKERHybrid(
            diversity_threshold=0.3,
            convergence_threshold=0.95,
            max_generations=5
        )

        result = await strategy.generate_proof(sample_theorem)

        assert isinstance(result, EvolutionResult)
        assert hasattr(result, 'generations_completed')

    @pytest.mark.asyncio
    async def test_parallel_workflow(self, sample_theorem):
        """Test complete parallel MAKER-MDAP workflow"""
        strategy = MAKERMDAPParallel(
            maker_voting_threshold=3,
            mdap_agents=4
        )

        with patch('hybrid_maker_integration.MAKER_EVOLUTION_AVAILABLE', False):
            with patch('hybrid_maker_integration.MDAP_AVAILABLE', False):
                result = await strategy.generate_proof(sample_theorem)

                assert isinstance(result, EvolutionResult)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestHybridPerformance:
    """Performance and benchmark tests"""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_mcts_then_maker_performance(self, sample_theorem):
        """Test MCTS-Then-MAKER performance"""
        strategy = MCTSThenMAKER(
            mcts_simulations=10,
            maker_voting_threshold=3
        )

        start = time.time()

        with patch('hybrid_maker_integration.MCTS_AVAILABLE', False):
            result = await strategy.generate_proof(sample_theorem)

        duration = time.time() - start

        assert duration < 10.0  # Should complete in < 10s even with failures

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_evolution_scalability(self, sample_theorem):
        """Test evolution strategy scalability"""
        for pop_size in [10, 20, 30]:
            strategy = MAKERThenEvolution(
                population_size=pop_size,
                evolution_generations=3
            )

            with patch('hybrid_maker_integration.MAKER_EVOLUTION_AVAILABLE', False):
                result = await strategy.generate_proof(sample_theorem)

                assert isinstance(result, EvolutionResult)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_parallel_efficiency(self, sample_theorem):
        """Test parallel execution efficiency"""
        strategy = MAKERMDAPParallel(
            maker_voting_threshold=3,
            mdap_agents=4
        )

        start = time.time()

        with patch('hybrid_maker_integration.MAKER_EVOLUTION_AVAILABLE', False):
            result = await strategy.generate_proof(sample_theorem)

        parallel_time = time.time() - start

        # Parallel should be reasonably fast
        assert parallel_time < 15.0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestHybridEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_empty_theorem(self):
        """Test handling of empty theorem"""
        strategy = MCTSThenMAKER()

        with patch('hybrid_maker_integration.MCTS_AVAILABLE', False):
            result = await strategy.generate_proof("")

            assert isinstance(result, EvolutionResult)

    @pytest.mark.asyncio
    async def test_very_long_theorem(self):
        """Test handling of very long theorem"""
        long_theorem = "forall " + " ".join([f"n{i}" for i in range(100)]) + " : nat, " + " + ".join([f"n{i}" for i in range(100)])

        strategy = AdaptiveMAKERHybrid(max_generations=2)
        result = await strategy.generate_proof(long_theorem)

        assert isinstance(result, EvolutionResult)

    @pytest.mark.asyncio
    async def test_invalid_config(self):
        """Test handling of invalid configuration"""
        config = MAKERHybridConfig(
            voting_threshold=-1,  # Invalid
            mcts_simulations=0  # Invalid
        )

        strategy = MCTSThenMAKER(
            mcts_simulations=config.mcts_simulations,
            maker_voting_threshold=config.voting_threshold
        )

        with patch('hybrid_maker_integration.MCTS_AVAILABLE', False):
            result = await strategy.generate_proof("test theorem")

            assert isinstance(result, EvolutionResult)

    def test_evolution_result_serialization(self, sample_evolution_result):
        """Test EvolutionResult serialization"""
        # Convert to dict
        result_dict = {
            "success": sample_evolution_result.success,
            "generations_completed": sample_evolution_result.generations_completed,
            "evolution_time": sample_evolution_result.evolution_time,
            "best_proof": sample_evolution_result.best_proof,
            "best_fitness": sample_evolution_result.best_fitness,
            "convergence_history": sample_evolution_result.convergence_history
        }

        assert result_dict["success"] is True
        assert result_dict["best_fitness"] == 0.85

        # JSON serialization
        json_str = json.dumps(result_dict)
        assert len(json_str) > 0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
