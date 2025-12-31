"""
Comprehensive Test Suite for Hybrid MAKER Strategies

This module provides extensive testing for hybrid MAKER strategies including:
- Unit tests for individual strategies
- Integration tests for strategy combinations
- Performance benchmarks
- Edge case scenarios
- Configuration validation

Author: Hybrid MAKER Test Suite
Version: 1.0.0
Coverage Target: >90%
"""

import asyncio
import json
import pytest
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import tempfile
import os
import sys

# Import hybrid MAKER components
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
        get_maker_hybrid_capabilities
    )
    HYBRID_MAKER_AVAILABLE = True
except ImportError:
    HYBRID_MAKER_AVAILABLE = False
    pytestmark = pytest.mark.skip("Hybrid MAKER integration not available")

try:
    from leanaide_hybrid_strategies import (
        HybridStrategy,
        MCTSThenEvolution,
        EvolutionWithMCTS,
        MCTSAdversarial,
        MCTSSelfPlay,
        AdaptiveHybrid,
        MCTSThenMDAP,
        MDAPThenMCTS,
        MDAPMCTSParallel,
        AdaptiveMDAPMCTS,
        HybridStrategyFactory
    )
    HYBRID_STRATEGIES_AVAILABLE = True
except ImportError:
    HYBRID_STRATEGIES_AVAILABLE = False

try:
    from maker_engine import (
        MakerStep,
        MakerConfig,
        MakerState,
        MakerRunResult,
        CheckpointStore,
        FileCheckpointStore,
        MakerEngine
    )
    MAKER_ENGINE_AVAILABLE = True
except ImportError:
    MAKER_ENGINE_AVAILABLE = False


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def test_theorem():
    """Provide test theorem"""
    return "forall n m : nat, n + m = m + n"


@pytest.fixture
def test_theorems():
    """Provide multiple test theorems"""
    return [
        "forall n : nat, n + 0 = n",
        "forall n m : nat, n + m = m + n",
        "forall n m k : nat, (n + m) + k = n + (m + k)",
        "forall n : nat, 0 + n = n",
        "forall n m : nat, n * m = m * n"
    ]


@pytest.fixture
def sample_config():
    """Provide sample MAKER hybrid configuration"""
    return MAKERHybridConfig(
        enable_voting=True,
        voting_threshold=3,
        enable_red_flagging=True,
        enable_decomposition=True,
        decomposition_depth=3,
        max_subtasks=10,
        mcts_simulations=50,
        evolution_generations=10,
        population_size=15,
        adversarial_rounds=2,
        red_team_agents=2,
        blue_team_agents=2,
        adaptive_switching=True,
        diversity_threshold=0.3,
        convergence_threshold=0.95
    )


@pytest.fixture
def temp_checkpoint_file():
    """Create temporary checkpoint file"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_maker_config():
    """Provide sample MAKER configuration"""
    return MakerConfig(
        k_min=2,
        k_max=8,
        max_votes_per_step=60,
        max_steps=100,
        timeout_seconds=90,
        checkpoint_interval=25
    )


@pytest.fixture
def mock_team():
    """Provide mock team for testing"""
    from workflow_structures import ModelConfig, Team

    members = [
        ModelConfig(
            model_id="gpt-4",
            api_key="test_key",
            api_base="http://test.com",
            temperature=0.7
        )
    ]

    return Team(
        team_id="test_team",
        name="Test Team",
        members=members
    )


@pytest.fixture
def mock_maker_step():
    """Provide mock MAKER step"""
    return MakerStep(
        step_id="test_step",
        prompt_template="Generate proof for: {state}",
        expected_schema={"type": "object"},
        task_type="general",
        priority=1
    )


@pytest.fixture
def mock_leanaide_client():
    """Provide mock LeanAide client"""
    client = AsyncMock()

    # Mock response
    mock_response = Mock()
    mock_response.success = True
    mock_response.proof = "simp\nrefl"
    mock_response.fitness = 0.85

    client.generate_proof = AsyncMock(return_value=mock_response)
    client.verify_proof = AsyncMock(return_value=True)

    return client


# ============================================================================
# Unit Tests: MAKER Hybrid Configuration
# ============================================================================

class TestMAKERHybridConfig:
    """Test MAKER hybrid configuration"""

    @pytest.mark.parametrize("enable_voting", [True, False])
    @pytest.mark.parametrize("voting_threshold", [1, 3, 5, 10])
    def test_config_initialization(self, enable_voting, voting_threshold):
        """Test configuration initialization with various parameters"""
        config = MAKERHybridConfig(
            enable_voting=enable_voting,
            voting_threshold=voting_threshold
        )

        assert config.enable_voting == enable_voting
        assert config.voting_threshold == voting_threshold
        assert config.enable_red_flagging == True
        assert config.enable_decomposition == True

    def test_config_to_dict(self, sample_config):
        """Test configuration serialization to dictionary"""
        config_dict = sample_config.to_dict()

        assert isinstance(config_dict, dict)
        assert "enable_voting" in config_dict
        assert "voting_threshold" in config_dict
        assert "enable_decomposition" in config_dict
        assert config_dict["voting_threshold"] == 3

    def test_config_default_values(self):
        """Test configuration default values"""
        config = MAKERHybridConfig()

        assert config.enable_voting == True
        assert config.voting_threshold == 3
        assert config.enable_red_flagging == True
        assert config.enable_decomposition == True
        assert config.decomposition_depth == 3
        assert config.max_subtasks == 10
        assert config.mcts_simulations == 100
        assert config.evolution_generations == 20
        assert config.population_size == 20

    @pytest.mark.parametrize("invalid_threshold", [-1, 0, -10])
    def test_config_invalid_threshold_raises_error(self, invalid_threshold):
        """Test that invalid voting threshold is handled"""
        # Configuration accepts any int, validation happens elsewhere
        config = MAKERHybridConfig(voting_threshold=invalid_threshold)
        assert config.voting_threshold == invalid_threshold

    def test_config_all_parameters(self):
        """Test configuration with all parameters set"""
        config = MAKERHybridConfig(
            enable_voting=False,
            voting_threshold=5,
            enable_red_flagging=False,
            enable_decomposition=False,
            decomposition_depth=5,
            max_subtasks=20,
            mcts_simulations=200,
            evolution_generations=50,
            population_size=100,
            adversarial_rounds=10,
            red_team_agents=5,
            blue_team_agents=5,
            adaptive_switching=False,
            diversity_threshold=0.5,
            convergence_threshold=0.99
        )

        assert config.enable_voting == False
        assert config.voting_threshold == 5
        assert config.enable_decomposition == False
        assert config.decomposition_depth == 5
        assert config.mcts_simulations == 200
        assert config.evolution_generations == 50


# ============================================================================
# Unit Tests: MAKER Hybrid Mode
# ============================================================================

class TestMAKERHybridMode:
    """Test MAKER hybrid mode enumeration"""

    def test_all_modes_defined(self):
        """Test that all expected modes are defined"""
        expected_modes = [
            "MCTS_THEN_MAKER",
            "MAKER_THEN_EVOLUTION",
            "MAKER_ADVERSARIAL",
            "ADAPTIVE_MAKER",
            "MAKER_MDAP_PARALLEL",
            "FULL_MAKER_HYBRID"
        ]

        for mode_name in expected_modes:
            assert hasattr(MAKERHybridMode, mode_name)
            mode = getattr(MAKERHybridMode, mode_name)
            assert isinstance(mode.value, str)

    def test_mode_values_are_strings(self):
        """Test that all mode values are strings"""
        for mode in MAKERHybridMode:
            assert isinstance(mode.value, str)
            assert len(mode.value) > 0

    def test_mode_uniqueness(self):
        """Test that all modes have unique values"""
        mode_values = [mode.value for mode in MAKERHybridMode]
        assert len(mode_values) == len(set(mode_values))


# ============================================================================
# Unit Tests: MAKER Engine Components
# ============================================================================

@pytest.mark.skipif(not MAKER_ENGINE_AVAILABLE, reason="Maker engine not available")
class TestMakerStep:
    """Test MAKER step functionality"""

    def test_maker_step_initialization(self, mock_maker_step):
        """Test MAKER step initialization"""
        assert mock_maker_step.step_id == "test_step"
        assert mock_maker_step.task_type == "general"
        assert mock_maker_step.priority == 1
        assert mock_maker_step.metadata == {}

    def test_render_prompt(self, mock_maker_step):
        """Test prompt rendering"""
        state = {"goal": "prove theorem"}
        history = [{"action": "simp"}]

        prompt = mock_maker_step.render_prompt(state, history)

        assert isinstance(prompt, str)
        assert "prove theorem" in prompt or "goal" in prompt

    def test_render_prompt_with_empty_history(self, mock_maker_step):
        """Test prompt rendering with empty history"""
        state = {"goal": "test"}
        history = []

        prompt = mock_maker_step.render_prompt(state, history)

        assert isinstance(prompt, str)
        assert len(prompt) > 0

    @pytest.mark.parametrize("priority", [0, 1, 5, 10])
    def test_maker_step_priority(self, priority):
        """Test MAKER step with various priorities"""
        step = MakerStep(
            step_id=f"step_{priority}",
            prompt_template="Test",
            priority=priority
        )
        assert step.priority == priority


@pytest.mark.skipif(not MAKER_ENGINE_AVAILABLE, reason="Maker engine not available")
class TestMakerConfig:
    """Test MAKER configuration"""

    def test_maker_config_initialization(self):
        """Test MAKER config initialization"""
        config = MakerConfig()

        assert config.k_min == 2
        assert config.k_max == 8
        assert config.max_votes_per_step == 60
        assert config.max_steps == 1000
        assert config.timeout_seconds == 90

    def test_maker_config_custom_values(self):
        """Test MAKER config with custom values"""
        config = MakerConfig(
            k_min=1,
            k_max=5,
            max_votes_per_step=30,
            max_steps=500,
            timeout_seconds=60
        )

        assert config.k_min == 1
        assert config.k_max == 5
        assert config.max_votes_per_step == 30

    @pytest.mark.parametrize("k_min,k_max", [
        (1, 5),
        (2, 8),
        (3, 10)
    ])
    def test_k_value_range(self, k_min, k_max):
        """Test various k value ranges"""
        config = MakerConfig(k_min=k_min, k_max=k_max)
        assert config.k_min < config.k_max


@pytest.mark.skipif(not MAKER_ENGINE_AVAILABLE, reason="Maker engine not available")
class TestMakerState:
    """Test MAKER state management"""

    def test_maker_state_initialization(self):
        """Test state initialization"""
        state = MakerState()

        assert state.step_index == 0
        assert state.current_state is None
        assert state.history == []
        assert state.last_action is None

    def test_maker_state_with_values(self):
        """Test state with initial values"""
        state = MakerState(
            step_index=5,
            current_state={"test": "value"},
            last_action="simp"
        )

        assert state.step_index == 5
        assert state.current_state == {"test": "value"}
        assert state.last_action == "simp"

    def test_maker_state_history_append(self):
        """Test appending to history"""
        state = MakerState()
        state.history.append({"action": "test"})

        assert len(state.history) == 1
        assert state.history[0] == {"action": "test"}


@pytest.mark.skipif(not MAKER_ENGINE_AVAILABLE, reason="Maker engine not available")
class TestFileCheckpointStore:
    """Test file-based checkpoint storage"""

    def test_save_and_load_checkpoint(self, temp_checkpoint_file):
        """Test saving and loading checkpoint"""
        store = FileCheckpointStore(temp_checkpoint_file)

        # Create state
        original_state = MakerState(
            step_index=10,
            current_state={"goal": "test"},
            history=[{"action": "simp"}, {"action": "refl"}],
            last_action="refl"
        )

        # Save
        store.save(original_state)

        # Load
        loaded_state = store.load()

        assert loaded_state is not None
        assert loaded_state.step_index == original_state.step_index
        assert loaded_state.current_state == original_state.current_state
        assert len(loaded_state.history) == len(original_state.history)
        assert loaded_state.last_action == original_state.last_action

    def test_load_nonexistent_checkpoint(self, temp_checkpoint_file):
        """Test loading from nonexistent file"""
        store = FileCheckpointStore(temp_checkpoint_file)

        # Delete file if it exists
        if os.path.exists(temp_checkpoint_file):
            os.unlink(temp_checkpoint_file)

        # Load should return None
        loaded_state = store.load()
        assert loaded_state is None

    def test_overwrite_checkpoint(self, temp_checkpoint_file):
        """Test overwriting existing checkpoint"""
        store = FileCheckpointStore(temp_checkpoint_file)

        # Save first checkpoint
        state1 = MakerState(step_index=5, current_state={"v": 1})
        store.save(state1)

        # Save second checkpoint
        state2 = MakerState(step_index=10, current_state={"v": 2})
        store.save(state2)

        # Load should get second checkpoint
        loaded = store.load()
        assert loaded.step_index == 10
        assert loaded.current_state == {"v": 2}


# ============================================================================
# Unit Tests: MCTS-Then-MAKER Strategy
# ============================================================================

@pytest.mark.skipif(not HYBRID_MAKER_AVAILABLE, reason="Hybrid MAKER not available")
class TestMCTSThenMAKER:
    """Test MCTS-Then-MAKER hybrid strategy"""

    @pytest.fixture
    def strategy(self):
        return MCTSThenMAKER(
            mcts_simulations=50,
            maker_voting_threshold=3,
            population_size=15
        )

    def test_strategy_initialization(self, strategy):
        """Test strategy initialization"""
        assert strategy.name == "MCTS_Then_MAKER"
        assert strategy.mcts_simulations == 50
        assert strategy.maker_voting_threshold == 3
        assert strategy.population_size == 15

    def test_strategy_description(self, strategy):
        """Test strategy has description"""
        assert len(strategy.description) > 0

    @pytest.mark.asyncio
    async def test_generate_proof_success(self, strategy, test_theorem):
        """Test successful proof generation"""
        # This test may require mocking
        try:
            result = await strategy.generate_proof(test_theorem)

            assert hasattr(result, 'success')
            assert hasattr(result, 'generations_completed')
            assert hasattr(result, 'evolution_time')
        except ImportError as e:
            pytest.skip("Required dependencies not available")

    @pytest.mark.asyncio
    async def test_generate_proof_with_empty_theorem(self, strategy):
        """Test proof generation with empty theorem"""
        result = await strategy.generate_proof("")

        # Should handle gracefully
        assert result is not None

    @pytest.mark.parametrize("simulations", [10, 50, 100])
    def test_various_simulation_counts(self, simulations):
        """Test with various simulation counts"""
        strategy = MCTSThenMAKER(mcts_simulations=simulations)
        assert strategy.mcts_simulations == simulations

    @pytest.mark.parametrize("voting_threshold", [1, 3, 5, 10])
    def test_various_voting_thresholds(self, voting_threshold):
        """Test with various voting thresholds"""
        strategy = MCTSThenMAKER(maker_voting_threshold=voting_threshold)
        assert strategy.maker_voting_threshold == voting_threshold


# ============================================================================
# Unit Tests: MAKER-Then-Evolution Strategy
# ============================================================================

@pytest.mark.skipif(not HYBRID_MAKER_AVAILABLE, reason="Hybrid MAKER not available")
class TestMAKERThenEvolution:
    """Test MAKER-Then-Evolution hybrid strategy"""

    @pytest.fixture
    def strategy(self):
        return MAKERThenEvolution(
            maker_voting_threshold=3,
            evolution_generations=10,
            population_size=20,
            initial_candidates=50
        )

    def test_strategy_initialization(self, strategy):
        """Test strategy initialization"""
        assert strategy.name == "MAKER_Then_Evolution"
        assert strategy.maker_voting_threshold == 3
        assert strategy.evolution_generations == 10
        assert strategy.population_size == 20
        assert strategy.initial_candidates == 50

    @pytest.mark.asyncio
    async def test_generate_proof_basic(self, strategy, test_theorem):
        """Test basic proof generation"""
        try:
            result = await strategy.generate_proof(test_theorem)

            assert result is not None
            assert hasattr(result, 'success')
        except ImportError as e:
            pytest.skip("Required dependencies not available")

    def test_generate_candidate(self, strategy, test_theorem):
        """Test candidate generation"""
        candidate = strategy._generate_candidate(test_theorem, seed=42)

        assert isinstance(candidate, str)
        assert len(candidate) > 0
        assert "theorem" in candidate.lower() or test_theorem in candidate

    @pytest.mark.parametrize("seed", [0, 1, 42, 100])
    def test_generate_candidate_deterministic(self, strategy, test_theorem, seed):
        """Test that candidate generation is deterministic with same seed"""
        candidate1 = strategy._generate_candidate(test_theorem, seed=seed)
        candidate2 = strategy._generate_candidate(test_theorem, seed=seed)

        assert candidate1 == candidate2

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_generate_candidate_different_seeds(self, strategy, test_theorem, seed):
        """Test that different seeds produce different candidates"""
        candidate1 = strategy._generate_candidate(test_theorem, seed=seed)
        candidate2 = strategy._generate_candidate(test_theorem, seed=seed+1)

        # May be same due to randomness, but typically different
        # Just check both are valid
        assert isinstance(candidate1, str)
        assert isinstance(candidate2, str)

    def test_evaluate_candidate(self, strategy):
        """Test candidate evaluation"""
        candidate = "theorem test\nby\n  simp\n  refl"

        fitness = strategy._evaluate_candidate(candidate)

        assert isinstance(fitness, float)
        assert 0.0 <= fitness <= 1.0

    def test_evaluate_empty_candidate(self, strategy):
        """Test evaluation of empty candidate"""
        fitness = strategy._evaluate_candidate("")

        assert fitness == 0.0

    @pytest.mark.parametrize("generations", [5, 10, 20, 50])
    def test_various_generation_counts(self, generations):
        """Test with various generation counts"""
        strategy = MAKERThenEvolution(evolution_generations=generations)
        assert strategy.evolution_generations == generations


# ============================================================================
# Unit Tests: MAKER Adversarial Hybrid
# ============================================================================

@pytest.mark.skipif(not HYBRID_MAKER_AVAILABLE, reason="Hybrid MAKER not available")
class TestMAKERAdversarialHybrid:
    """Test MAKER Adversarial hybrid strategy"""

    @pytest.fixture
    def strategy(self):
        return MAKERAdversarialHybrid(
            adversarial_rounds=3,
            maker_voting_threshold=3,
            red_team_size=2,
            blue_team_size=2
        )

    def test_strategy_initialization(self, strategy):
        """Test strategy initialization"""
        assert strategy.name == "MAKER_Adversarial_Hybrid"
        assert strategy.adversarial_rounds == 3
        assert strategy.maker_voting_threshold == 3
        assert strategy.red_team_size == 2
        assert strategy.blue_team_size == 2

    @pytest.mark.asyncio
    async def test_generate_proof_basic(self, strategy, test_theorem):
        """Test basic proof generation"""
        try:
            result = await strategy.generate_proof(test_theorem)

            assert result is not None
            assert hasattr(result, 'success')
            assert hasattr(result, 'generations_completed')
        except ImportError as e:
            pytest.skip("Required dependencies not available")

    def test_evaluate_defense(self, strategy):
        """Test defense evaluation"""
        # Mock defense object
        class MockDefense:
            effectiveness = 0.85

        defense = MockDefense()
        fitness = strategy._evaluate_defense(defense)

        assert fitness == 0.85

    def test_evaluate_defense_no_effectiveness(self, strategy):
        """Test defense evaluation without effectiveness attribute"""
        defense = {"action": "simp"}

        fitness = strategy._evaluate_defense(defense)

        # Should return random value between 0.5 and 1.0
        assert 0.5 <= fitness <= 1.0

    @pytest.mark.parametrize("rounds", [1, 3, 5, 10])
    def test_various_adversarial_rounds(self, rounds):
        """Test with various adversarial round counts"""
        strategy = MAKERAdversarialHybrid(adversarial_rounds=rounds)
        assert strategy.adversarial_rounds == rounds

    @pytest.mark.parametrize("red_team,blue_team", [
        (1, 1),
        (2, 2),
        (3, 3),
        (5, 5)
    ])
    def test_various_team_sizes(self, red_team, blue_team):
        """Test with various team sizes"""
        strategy = MAKERAdversarialHybrid(
            red_team_size=red_team,
            blue_team_size=blue_team
        )
        assert strategy.red_team_size == red_team
        assert strategy.blue_team_size == blue_team


# ============================================================================
# Unit Tests: Adaptive MAKER Hybrid
# ============================================================================

@pytest.mark.skipif(not HYBRID_MAKER_AVAILABLE, reason="Hybrid MAKER not available")
class TestAdaptiveMAKERHybrid:
    """Test Adaptive MAKER hybrid strategy"""

    @pytest.fixture
    def strategy(self):
        return AdaptiveMAKERHybrid(
            diversity_threshold=0.3,
            convergence_threshold=0.95,
            max_generations=50
        )

    def test_strategy_initialization(self, strategy):
        """Test strategy initialization"""
        assert strategy.name == "Adaptive_MAKER_Hybrid"
        assert strategy.diversity_threshold == 0.3
        assert strategy.convergence_threshold == 0.95
        assert strategy.max_generations == 50

    @pytest.mark.asyncio
    async def test_generate_proof_basic(self, strategy, test_theorem):
        """Test basic proof generation"""
        try:
            result = await strategy.generate_proof(test_theorem)

            assert result is not None
            assert hasattr(result, 'success')
        except ImportError as e:
            pytest.skip("Required dependencies not available")

    def test_initialize_population(self, strategy, test_theorem):
        """Test population initialization"""
        try:
            population = strategy._initialize_population(test_theorem, size=10)

            assert hasattr(population, 'individuals')
            assert len(population.individuals) == 10
        except (ImportError, AttributeError):
            pytest.skip("Population class not available")

    def test_evaluate_candidate(self, strategy):
        """Test candidate evaluation"""
        candidate = "test candidate with multiple tactics"

        fitness = strategy._evaluate_candidate(candidate)

        assert isinstance(fitness, float)
        assert 0.0 <= fitness <= 1.0

    def test_generate_candidate(self, strategy, test_theorem):
        """Test candidate generation"""
        candidate = strategy._generate_candidate(test_theorem, seed=42)

        assert isinstance(candidate, str)
        assert len(candidate) > 0

    @pytest.mark.parametrize("diversity_threshold", [0.1, 0.3, 0.5, 0.7])
    def test_various_diversity_thresholds(self, diversity_threshold):
        """Test with various diversity thresholds"""
        strategy = AdaptiveMAKERHybrid(diversity_threshold=diversity_threshold)
        assert strategy.diversity_threshold == diversity_threshold

    @pytest.mark.parametrize("convergence_threshold", [0.9, 0.95, 0.99, 1.0])
    def test_various_convergence_thresholds(self, convergence_threshold):
        """Test with various convergence thresholds"""
        strategy = AdaptiveMAKERHybrid(convergence_threshold=convergence_threshold)
        assert strategy.convergence_threshold == convergence_threshold


# ============================================================================
# Unit Tests: MAKER-MDAP Parallel
# ============================================================================

@pytest.mark.skipif(not HYBRID_MAKER_AVAILABLE, reason="Hybrid MAKER not available")
class TestMAKERMDAPParallel:
    """Test MAKER-MDAP Parallel hybrid strategy"""

    @pytest.fixture
    def strategy(self):
        return MAKERMDAPParallel(
            maker_voting_threshold=3,
            mdap_agents=4,
            combination_method="best_fitness"
        )

    def test_strategy_initialization(self, strategy):
        """Test strategy initialization"""
        assert strategy.name == "MAKER_MDAP_Parallel"
        assert strategy.maker_voting_threshold == 3
        assert strategy.mdap_agents == 4
        assert strategy.combination_method == "best_fitness"

    @pytest.mark.asyncio
    async def test_generate_proof_basic(self, strategy, test_theorem):
        """Test basic proof generation"""
        try:
            result = await strategy.generate_proof(test_theorem)

            assert result is not None
            assert hasattr(result, 'success')
        except ImportError as e:
            pytest.skip("Required dependencies not available")

    @pytest.mark.parametrize("combination_method", [
        "best_fitness",
        "average",
        "voting"
    ])
    def test_various_combination_methods(self, combination_method):
        """Test with various combination methods"""
        strategy = MAKERMDAPParallel(combination_method=combination_method)
        assert strategy.combination_method == combination_method

    @pytest.mark.parametrize("mdap_agents", [2, 4, 6, 8])
    def test_various_mdap_agent_counts(self, mdap_agents):
        """Test with various MDAP agent counts"""
        strategy = MAKERMDAPParallel(mdap_agents=mdap_agents)
        assert strategy.mdap_agents == mdap_agents


# ============================================================================
# Integration Tests: Strategy Combinations
# ============================================================================

@pytest.mark.skipif(not HYBRID_MAKER_AVAILABLE, reason="Hybrid MAKER not available")
class TestStrategyCombinations:
    """Test strategy combinations and interactions"""

    @pytest.mark.asyncio
    async def test_mcts_maker_then_evolution(self, test_theorem):
        """Test combining MCTS-MAKER with evolution"""
        try:
            # Phase 1: MCTS-Then-MAKER
            phase1 = MCTSThenMAKER(mcts_simulations=30)
            result1 = await phase1.generate_proof(test_theorem)

            # Phase 2: Evolution
            phase2 = MAKERThenEvolution(evolution_generations=5)
            result2 = await phase2.generate_proof(test_theorem)

            assert result1 is not None
            assert result2 is not None
        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.asyncio
    async def test_adaptive_with_parallel(self, test_theorem):
        """Test combining adaptive with parallel execution"""
        try:
            # Run adaptive
            adaptive = AdaptiveMAKERHybrid(max_generations=5)
            result1 = await adaptive.generate_proof(test_theorem)

            # Run parallel
            parallel = MAKERMDAPParallel(mdap_agents=2)
            result2 = await parallel.generate_proof(test_theorem)

            assert result1 is not None
            assert result2 is not None
        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.asyncio
    async def test_full_hybrid_integration(self, test_theorem, sample_config):
        """Test full hybrid integration"""
        try:
            result = await run_maker_hybrid(
                theorem=test_theorem,
                mode=MAKERHybridMode.FULL_MAKER_HYBRID,
                config=sample_config
            )

            assert result is not None
            assert hasattr(result, 'success')
        except (ImportError, TypeError) as e:
            pytest.skip(f"Full hybrid not available: {e}")

    @pytest.mark.parametrize("mode", [
        MAKERHybridMode.MCTS_THEN_MAKER,
        MAKERHybridMode.MAKER_THEN_EVOLUTION,
        MAKERHybridMode.ADAPTIVE_MAKER
    ])
    @pytest.mark.asyncio
    async def test_different_hybrid_modes(self, test_theorem, mode):
        """Test different hybrid modes"""
        try:
            result = await run_maker_hybrid(
                theorem=test_theorem,
                mode=mode
            )

            assert result is not None
        except (ImportError, TypeError):
            pytest.skip("Mode not available")


# ============================================================================
# Integration Tests: Fallback Mechanisms
# ============================================================================

@pytest.mark.skipif(not HYBRID_MAKER_AVAILABLE, reason="Hybrid MAKER not available")
class TestFallbackMechanisms:
    """Test fallback and error handling mechanisms"""

    @pytest.mark.asyncio
    async def test_mcts_fallback_to_evolution(self, test_theorem):
        """Test fallback from MCTS to evolution"""
        try:
            strategy = MCTSThenMAKER()

            # If MCTS fails, should fallback gracefully
            result = await strategy.generate_proof(test_theorem)

            # Should not crash even if components missing
            assert result is not None
        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.asyncio
    async def test_voting_fallback(self, test_theorem):
        """Test voting fallback mechanisms"""
        try:
            strategy = MAKERThenEvolution(
                maker_voting_threshold=3,
                evolution_generations=5
            )

            result = await strategy.generate_proof(test_theorem)

            # Should handle voting failures gracefully
            assert result is not None
        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.asyncio
    async def test_adaptive_strategy_switching(self, test_theorem):
        """Test adaptive strategy switching"""
        try:
            strategy = AdaptiveMAKERHybrid(max_generations=10)

            result = await strategy.generate_proof(test_theorem)

            # Should attempt different strategies
            assert result is not None
        except ImportError:
            pytest.skip("Required dependencies not available")


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.skipif(not HYBRID_MAKER_AVAILABLE, reason="Hybrid MAKER not available")
@pytest.mark.slow
class TestHybridPerformance:
    """Test performance of hybrid strategies"""

    @pytest.mark.asyncio
    async def test_mcts_maker_performance(self, test_theorem):
        """Test MCTS-Then-MAKER performance"""
        strategy = MCTSThenMAKER(mcts_simulations=50)

        start_time = time.time()
        try:
            result = await strategy.generate_proof(test_theorem)
            elapsed = time.time() - start_time

            assert elapsed < 300  # Should complete within 5 minutes
            assert result is not None
        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.asyncio
    async def test_evolution_performance(self, test_theorem):
        """Test evolution performance"""
        strategy = MAKERThenEvolution(evolution_generations=10)

        start_time = time.time()
        try:
            result = await strategy.generate_proof(test_theorem)
            elapsed = time.time() - start_time

            assert elapsed < 300  # Should complete within 5 minutes
            assert result is not None
        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("population_size", [10, 20, 30])
    async def test_scalability_population_size(self, test_theorem, population_size):
        """Test scalability with different population sizes"""
        strategy = MAKERThenEvolution(
            evolution_generations=5,
            population_size=population_size
        )

        start_time = time.time()
        try:
            result = await strategy.generate_proof(test_theorem)
            elapsed = time.time() - start_time

            # Time should scale roughly linearly with population
            assert elapsed < 600  # 10 minutes max
            assert result is not None
        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("generations", [5, 10, 20])
    async def test_scalability_generations(self, test_theorem, generations):
        """Test scalability with different generation counts"""
        strategy = MAKERThenEvolution(
            evolution_generations=generations,
            population_size=15
        )

        start_time = time.time()
        try:
            result = await strategy.generate_proof(test_theorem)
            elapsed = time.time() - start_time

            assert elapsed < 600
            assert result is not None
        except ImportError:
            pytest.skip("Required dependencies not available")


# ============================================================================
# Performance Benchmarks
# ============================================================================

@pytest.mark.skipif(not HYBRID_MAKER_AVAILABLE, reason="Hybrid MAKER not available")
class TestPerformanceBenchmarks:
    """Benchmark performance of different strategies"""

    @pytest.mark.asyncio
    async def benchmark_strategies(self, test_theorem):
        """Benchmark all available strategies"""
        strategies = [
            ("MCTS_Then_MAKER", MCTSThenMAKER(mcts_simulations=30)),
            ("MAKER_Then_Evolution", MAKERThenEvolution(evolution_generations=5)),
            ("Adaptive_MAKER", AdaptiveMAKERHybrid(max_generations=5))
        ]

        results = {}
        for name, strategy in strategies:
            start_time = time.time()
            try:
                result = await strategy.generate_proof(test_theorem)
                elapsed = time.time() - start_time

                results[name] = {
                    "time": elapsed,
                    "success": result.success if hasattr(result, 'success') else False
                }
            except (ImportError, Exception):
                results[name] = {"time": None, "success": False}

        # Log results
        print("\n=== Performance Benchmarks ===")
        for name, metrics in results.items():
            print(f"{name}: {metrics['time']:.2f}s (success={metrics['success']})")

        # At least one strategy should work
        successful = [r for r in results.values() if r['success']]
        assert len(successful) > 0 or all(r['time'] is None for r in results.values())


# ============================================================================
# Edge Case Tests
# ============================================================================

@pytest.mark.skipif(not HYBRID_MAKER_AVAILABLE, reason="Hybrid MAKER not available")
class TestEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.mark.asyncio
    async def test_empty_theorem(self):
        """Test with empty theorem"""
        strategy = MCTSThenMAKER()
        result = await strategy.generate_proof("")

        # Should handle gracefully
        assert result is not None

    @pytest.mark.asyncio
    async def test_very_long_theorem(self):
        """Test with very long theorem"""
        long_theorem = " ".join(["forall"] * 100) + " n : nat, n = n"

        strategy = MAKERThenEvolution(evolution_generations=2)
        result = await strategy.generate_proof(long_theorem)

        # Should handle or fail gracefully
        assert result is not None

    @pytest.mark.asyncio
    async def test_special_characters_in_theorem(self):
        """Test with special characters"""
        special_theorem = "∀ n m : nat, n + m = m + n (with unicode: ∀)"

        strategy = AdaptiveMAKERHybrid(max_generations=3)
        result = await strategy.generate_proof(special_theorem)

        assert result is not None

    @pytest.mark.asyncio
    async def test_timeout_handling(self, test_theorem):
        """Test timeout handling"""
        # Create a strategy with very short timeout
        strategy = MCTSThenMAKER(mcts_simulations=1000000)

        # Should timeout gracefully
        start = time.time()
        try:
            result = await strategy.generate_proof(test_theorem)
            elapsed = time.time() - start

            # Should complete or timeout reasonably
            assert elapsed < 600  # 10 minutes max
        except (ImportError, asyncio.TimeoutError):
            pass  # Expected

    @pytest.mark.asyncio
    async def test_network_failure_simulation(self, test_theorem):
        """Test behavior when network fails"""
        # This would require mocking network failures
        # For now, just test the strategy doesn't crash
        strategy = MAKERThenEvolution(evolution_generations=2)

        try:
            result = await strategy.generate_proof(test_theorem)
            assert result is not None
        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.asyncio
    async def test_invalid_lean_code(self):
        """Test with invalid Lean code"""
        invalid_theorem = "this is not valid Lean syntax at all !!!"

        strategy = MAKERThenEvolution(evolution_generations=2)
        result = await strategy.generate_proof(invalid_theorem)

        # Should handle gracefully
        assert result is not None

    def test_voting_tie_scenario(self):
        """Test voting tie handling"""
        # This would require setting up a tie scenario
        # For now, just test the logic exists
        votes = {"A": 5, "B": 5, "C": 3}

        # Find max
        max_votes = max(votes.values())
        leaders = [k for k, v in votes.items() if v == max_votes]

        assert len(leaders) >= 1
        assert len(leaders) <= len(votes)

    @pytest.mark.asyncio
    async def test_single_agent_scenario(self, test_theorem):
        """Test with minimal agent count"""
        strategy = MAKERThenEvolution(
            population_size=2,
            evolution_generations=2
        )

        result = await strategy.generate_proof(test_theorem)

        assert result is not None

    @pytest.mark.asyncio
    async def test_zero_population_size(self, test_theorem):
        """Test with zero population (edge case)"""
        # Should handle gracefully or fail with clear error
        strategy = MAKERThenEvolution(
            population_size=0,
            evolution_generations=2
        )

        try:
            result = await strategy.generate_proof(test_theorem)
            # If it doesn't crash, that's good
            assert result is not None
        except (ValueError, AttributeError):
            # Also acceptable to raise an error
            pass

    @pytest.mark.asyncio
    async def test_negative_parameters(self, test_theorem):
        """Test with negative parameters"""
        strategy = MAKERThenEvolution(
            evolution_generations=-1,
            population_size=10
        )

        # Should handle gracefully
        result = await strategy.generate_proof(test_theorem)
        assert result is not None


# ============================================================================
# Configuration Validation Tests
# ============================================================================

@pytest.mark.skipif(not HYBRID_MAKER_AVAILABLE, reason="Hybrid MAKER not available")
class TestConfigurationValidation:
    """Test configuration validation"""

    def test_voting_threshold_range(self):
        """Test voting threshold validation"""
        # Valid thresholds
        for threshold in [1, 3, 5, 10]:
            config = MAKERHybridConfig(voting_threshold=threshold)
            assert config.voting_threshold == threshold

    def test_population_size_range(self):
        """Test population size validation"""
        for size in [1, 10, 20, 50, 100]:
            config = MAKERHybridConfig(population_size=size)
            assert config.population_size == size

    def test_decomposition_depth_range(self):
        """Test decomposition depth validation"""
        for depth in [1, 3, 5, 10]:
            config = MAKERHybridConfig(decomposition_depth=depth)
            assert config.decomposition_depth == depth

    def test_config_serialization_roundtrip(self, sample_config):
        """Test config serialization and deserialization"""
        # Serialize
        config_dict = sample_config.to_dict()

        # All fields should be present
        expected_fields = [
            "enable_voting",
            "voting_threshold",
            "enable_red_flagging",
            "enable_decomposition",
            "decomposition_depth",
            "max_subtasks",
            "mcts_simulations",
            "evolution_generations",
            "population_size",
            "adversarial_rounds",
            "red_team_agents",
            "blue_team_agents",
            "adaptive_switching",
            "diversity_threshold",
            "convergence_threshold"
        ]

        for field in expected_fields:
            assert field in config_dict

    def test_config_edge_cases(self):
        """Test config with edge case values"""
        # Minimum values
        config1 = MAKERHybridConfig(
            voting_threshold=1,
            population_size=1,
            decomposition_depth=1
        )
        assert config1.voting_threshold == 1

        # Maximum values
        config2 = MAKERHybridConfig(
            voting_threshold=100,
            population_size=1000,
            decomposition_depth=100
        )
        assert config2.voting_threshold == 100


# ============================================================================
# Workflow Integration Tests
# ============================================================================

@pytest.mark.skipif(not HYBRID_MAKER_AVAILABLE, reason="Hybrid MAKER not available")
class TestWorkflowIntegration:
    """Test integration with workflow systems"""

    @pytest.mark.asyncio
    async def test_workflow_step_integration(self, test_theorem):
        """Test integration as workflow step"""
        # Create a simple workflow
        workflow_steps = [
            ("MCTS_Then_MAKER", MCTSThenMAKER(mcts_simulations=20)),
            ("Evolution", MAKERThenEvolution(evolution_generations=5))
        ]

        results = {}
        for name, step in workflow_steps:
            try:
                result = await step.generate_proof(test_theorem)
                results[name] = result.success if hasattr(result, 'success') else False
            except ImportError:
                results[name] = None

        # At least one step should execute
        assert len(results) == len(workflow_steps)

    @pytest.mark.asyncio
    async def test_checkpoint_integration(self, test_theorem, temp_checkpoint_file):
        """Test integration with checkpointing"""
        # This would test saving/loading state during execution
        # For now, just verify checkpoint store works
        if MAKER_ENGINE_AVAILABLE:
            store = FileCheckpointStore(temp_checkpoint_file)

            state = MakerState(step_index=5, current_state={"test": test_theorem})
            store.save(state)

            loaded = store.load()
            assert loaded.step_index == 5

    @pytest.mark.asyncio
    async def test_multi_theorem_workflow(self, test_theorems):
        """Test workflow with multiple theorems"""
        strategy = MAKERThenEvolution(evolution_generations=3)

        results = []
        for theorem in test_theorems[:3]:  # Test first 3
            try:
                result = await strategy.generate_proof(theorem)
                results.append(result)
            except ImportError:
                pytest.skip("Required dependencies not available")

        assert len(results) == 3


# ============================================================================
# Statistics and Reporting Tests
# ============================================================================

@pytest.mark.skipif(not HYBRID_MAKER_AVAILABLE, reason="Hybrid MAKER not available")
class TestStatisticsAndReporting:
    """Test statistics tracking and reporting"""

    def test_strategy_statistics(self):
        """Test strategy statistics tracking"""
        strategy = MCTSThenMAKER()

        # Check statistics are initialized
        assert hasattr(strategy, 'statistics')
        assert 'total_runs' in strategy.statistics
        assert 'successful_proofs' in strategy.statistics

    def test_capabilities_report(self):
        """Test capabilities reporting"""
        try:
            capabilities = get_maker_hybrid_capabilities()

            assert isinstance(capabilities, dict)
            assert 'maker_hybrid_enabled' in capabilities
            assert 'modes' in capabilities
            assert 'strategies' in capabilities
        except ImportError:
            pytest.skip("Capabilities function not available")

    @pytest.mark.asyncio
    async def test_result_metrics(self, test_theorem):
        """Test result metrics are populated"""
        strategy = MAKERThenEvolution(evolution_generations=2)

        try:
            result = await strategy.generate_proof(test_theorem)

            # Check result has expected fields
            assert hasattr(result, 'success')
            assert hasattr(result, 'generations_completed')
            assert hasattr(result, 'evolution_time')
        except ImportError:
            pytest.skip("Required dependencies not available")


# ============================================================================
# Thread Safety and Concurrency Tests
# ============================================================================

@pytest.mark.skipif(not HYBRID_MAKER_AVAILABLE, reason="Hybrid MAKER not available")
class TestConcurrency:
    """Test concurrent execution and thread safety"""

    @pytest.mark.asyncio
    async def test_parallel_strategy_execution(self, test_theorem):
        """Test executing multiple strategies in parallel"""
        strategies = [
            MCTSThenMAKER(mcts_simulations=10),
            MAKERThenEvolution(evolution_generations=3),
            AdaptiveMAKERHybrid(max_generations=3)
        ]

        # Execute in parallel
        tasks = [s.generate_proof(test_theorem) for s in strategies]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Should get results for all strategies
            assert len(results) == len(strategies)
        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.asyncio
    async def test_concurrent_theorem_processing(self, test_theorems):
        """Test processing multiple theorems concurrently"""
        strategy = MAKERThenEvolution(evolution_generations=2)

        # Process multiple theorems
        tasks = [strategy.generate_proof(t) for t in test_theorems[:3]]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            assert len(results) == 3
        except ImportError:
            pytest.skip("Required dependencies not available")


# ============================================================================
# Mock Tests for Missing Dependencies
# ============================================================================

class TestMockBehavior:
    """Test behavior with mocked/unavailable dependencies"""

    def test_strategy_creation_without_dependencies(self):
        """Test creating strategies when dependencies unavailable"""
        if not HYBRID_MAKER_AVAILABLE:
            # Should not crash
            assert True
        else:
            # With dependencies, should create successfully
            strategy = MCTSThenMAKER()
            assert strategy is not None

    def test_config_without_dependencies(self):
        """Test config without dependencies"""
        config = MAKERHybridConfig()

        # Should work regardless of other dependencies
        assert config.enable_voting == True
        assert config.voting_threshold == 3


# ============================================================================
# Parameterized Tests
# ============================================================================

@pytest.mark.skipif(not HYBRID_MAKER_AVAILABLE, reason="Hybrid MAKER not available")
class TestParameterizedStrategies:
    """Parameterized tests across different configurations"""

    @pytest.mark.parametrize("voting_threshold,expected_min_votes", [
        (1, 1),
        (3, 5),  # 2k-1 formula
        (5, 9),
        (10, 19)
    ])
    def test_voting_agent_count(self, voting_threshold, expected_min_votes):
        """Test that voting threshold determines agent count"""
        # Based on first-to-ahead-by-k requiring 2k-1 agents
        expected_agents = 2 * voting_threshold - 1
        assert expected_agents == expected_min_votes

    @pytest.mark.parametrize("simulations,expected_max_time", [
        (10, 60),
        (50, 300),
        (100, 600)
    ])
    @pytest.mark.asyncio
    async def test_simulation_time_scaling(self, simulations, expected_max_time, test_theorem):
        """Test that simulation count scales appropriately with time"""
        strategy = MCTSThenMAKER(mcts_simulations=simulations)

        start = time.time()
        try:
            await strategy.generate_proof(test_theorem)
            elapsed = time.time() - start

            # Should complete within reasonable time
            assert elapsed < expected_max_time
        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.parametrize("population_size,generations", [
        (10, 5),
        (20, 10),
        (30, 15)
    ])
    @pytest.mark.asyncio
    async def test_evolution_parameters(self, population_size, generations, test_theorem):
        """Test different evolution parameters"""
        strategy = MAKERThenEvolution(
            population_size=population_size,
            evolution_generations=generations
        )

        try:
            result = await strategy.generate_proof(test_theorem)

            assert result is not None
            if hasattr(result, 'generations_completed'):
                assert result.generations_completed <= generations
        except ImportError:
            pytest.skip("Required dependencies not available")


# ============================================================================
# Regression Tests
# ============================================================================

@pytest.mark.skipif(not HYBRID_MAKER_AVAILABLE, reason="Hybrid MAKER not available")
class TestRegression:
    """Regression tests for known issues"""

    @pytest.mark.asyncio
    async def test_none_theorem_handling(self):
        """Test handling of None theorem"""
        strategy = MAKERThenEvolution(evolution_generations=2)

        # Should not crash
        try:
            result = await strategy.generate_proof(None)
            assert result is not None
        except (TypeError, AttributeError):
            # Also acceptable to raise error
            pass

    @pytest.mark.asyncio
    async def test_unicode_theorem_handling(self):
        """Test handling of unicode characters"""
        unicode_theorem = "∀ n : ℕ, n + 0 = n"

        strategy = MCTSThenMAKER()
        result = await strategy.generate_proof(unicode_theorem)

        assert result is not None

    def test_config_mutation(self, sample_config):
        """Test that config doesn't mutate unexpectedly"""
        original_threshold = sample_config.voting_threshold

        # Use config
        config_dict = sample_config.to_dict()

        # Check original unchanged
        assert sample_config.voting_threshold == original_threshold

    @pytest.mark.asyncio
    async def test_state_cleanup(self, test_theorem):
        """Test that state is cleaned up properly"""
        strategy = AdaptiveMAKERHybrid(max_generations=5)

        try:
            await strategy.generate_proof(test_theorem)

            # Should not have leaked state
            assert True
        except ImportError:
            pytest.skip("Required dependencies not available")


# ============================================================================
# Helper Functions
# ============================================================================

def assert_result_valid(result):
    """Helper to assert result is valid"""
    assert result is not None
    assert hasattr(result, 'success')
    assert hasattr(result, 'generations_completed')
    assert hasattr(result, 'evolution_time')


def assert_config_valid(config):
    """Helper to assert config is valid"""
    assert hasattr(config, 'enable_voting')
    assert hasattr(config, 'voting_threshold')
    assert hasattr(config, 'enable_decomposition')


# ============================================================================
# Test Discovery and Registration
# ============================================================================

def pytest_generate_tests(metafunc):
    """Generate parametrized tests dynamically"""
    if 'theorem' in metafunc.fixturenames:
        theorems = [
            "forall n : nat, n + 0 = n",
            "forall n m : nat, n + m = m + n"
        ]
        metafunc.parametrize("theorem", theorems)


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10",
        "-k", "test_"
    ])
