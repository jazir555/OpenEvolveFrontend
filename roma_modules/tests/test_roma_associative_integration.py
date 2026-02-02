"""
Unit tests for ROMA Associative Integration

Tests the ROMA-MDAP-MAKER + Associative integration functionality.

Author: OpenEvolve
Date: 2026-02-02
"""

import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass

import pytest

from roma_associative_integration import (
    ROMAMDAPMakerAssociativeEngine,
    ROMAMDAPMakerAssociativeConfig,
    create_romamdapmaker_associative_config,
    solve_with_romamdapmaker_associative,
    get_romamdapmaker_associative_status,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Create default configuration."""
    return ROMAMDAPMakerAssociativeConfig()


@pytest.fixture
def custom_config():
    """Create custom configuration."""
    return ROMAMDAPMakerAssociativeConfig(
        roma_max_depth_analysis=5,
        roma_max_depth_solving=3,
        mdap_enabled=False,
        provider="anthropic",
        model="claude-3-sonnet",
        temperature=0.3
    )


@pytest.fixture
def mock_llm_call():
    """Create mock LLM call function."""
    return Mock(return_value="Mock LLM response")


# =============================================================================
# ROMAMDAPMakerAssociativeConfig Tests
# =============================================================================

class TestROMAMDAPMakerAssociativeConfig:
    """Tests for ROMAMDAPMakerAssociativeConfig dataclass."""
    
    def test_default_config(self):
        """Test creating default configuration."""
        config = ROMAMDAPMakerAssociativeConfig()
        
        assert config.roma_max_depth_analysis == 3
        assert config.roma_max_depth_solving == 2
        assert config.roma_execution_mode == "recursive"
        assert config.roma_enable_checkpoints is False
        assert config.roma_enable_logging is True
        assert config.mdap_enabled is True
        assert config.mdap_k_ahead == 3
        assert config.mdap_max_samples == 100
        assert config.mdap_enable_red_flagging is True
        assert config.mdap_max_token_length == 750
        assert config.mdap_min_confidence == 0.2
        assert config.apply_maker_to_roma_atomic is True
        assert config.apply_maker_to_roma_planning is True
        assert config.aggregate_maker_results is True
        assert config.enable_hierarchical_voting is True
        assert config.enable_adaptive_k is True
        assert config.enable_caching is True
        assert config.cache_ttl_seconds == 3600
        assert config.cache_max_size == 10000
        assert config.max_retries == 3
        assert config.timeout_seconds == 300
        assert config.fallback_policy == "escalate_then_best_effort"
        assert config.use_associative_recomposition is True
        assert config.associative_max_retries == 3
        assert config.associative_use_agentjson is True
        assert config.enable_ground_truth is True
        assert config.ground_truth_storage_path == "roma_mdap_maker_ground_truth.json"
        assert config.apply_mdap_to_recomposed is True
        assert config.enable_hierarchical_validation is True
        assert config.use_evaluator_team is True
        assert config.evaluator_threshold == "standard_approval"
        assert config.evaluator_num_members == 3
        assert config.use_gauntlet_system is True
        assert config.gauntlet_difficulty == "adaptive"
        assert config.max_refinement_attempts == 3
        assert config.min_acceptance_score == 75.0
        assert config.provider == "openai"
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.1
    
    def test_custom_config(self, custom_config):
        """Test creating custom configuration."""
        assert custom_config.roma_max_depth_analysis == 5
        assert custom_config.roma_max_depth_solving == 3
        assert custom_config.mdap_enabled is False
        assert custom_config.provider == "anthropic"
        assert custom_config.model == "claude-3-sonnet"
        assert custom_config.temperature == 0.3
    
    def test_metadata_initialization(self):
        """Test metadata initialization."""
        config = ROMAMDAPMakerAssociativeConfig()
        assert config.metadata is not None
        assert isinstance(config.metadata, dict)


# =============================================================================
# ROMAMDAPMakerAssociativeEngine Tests
# =============================================================================

class TestROMAMDAPMakerAssociativeEngine:
    """Tests for ROMAMDAPMakerAssociativeEngine class."""
    
    def test_initialization_with_default_config(self):
        """Test engine initialization with default config."""
        config = ROMAMDAPMakerAssociativeConfig()
        engine = ROMAMDAPMakerAssociativeEngine(config)
        
        assert engine.config == config
        assert engine.initialized is False
    
    def test_initialization_with_custom_config(self, custom_config):
        """Test engine initialization with custom config."""
        engine = ROMAMDAPMakerAssociativeEngine(custom_config)
        
        assert engine.config == custom_config
    
    def test_initialization_without_config(self):
        """Test engine initialization without config."""
        engine = ROMAMDAPMakerAssociativeEngine()
        
        assert engine.config is not None
        assert isinstance(engine.config, ROMAMDAPMakerAssociativeConfig)
    
    def test_initialize_success(self):
        """Test successful initialization."""
        engine = ROMAMDAPMakerAssociativeEngine()
        result = engine.initialize()
        
        assert result is True
        assert engine.initialized is True
    
    def test_get_config(self, custom_config):
        """Test getting configuration."""
        engine = ROMAMDAPMakerAssociativeEngine(custom_config)
        result = engine.get_config()
        
        assert result == custom_config
    
    def test_plan_decomposition(self):
        """Test problem decomposition planning."""
        engine = ROMAMDAPMakerAssociativeEngine()
        result = engine.plan_decomposition(
            problem="Solve the equation x + 2 = 5",
            domain="mathematics"
        )
        
        assert "problem" in result
        assert result["problem"] == "Solve the equation x + 2 = 5"
        assert "domain" in result
        assert result["domain"] == "mathematics"
        assert "approach" in result
        assert result["approach"] == "associative"
        assert "confidence" in result
        assert isinstance(result["confidence"], float)
    
    def test_plan_decomposition_auto_initialize(self):
        """Test that plan_decomposition auto-initializes if needed."""
        engine = ROMAMDAPMakerAssociativeEngine()
        assert engine.initialized is False
        
        engine.plan_decomposition("test problem", "test domain")
        
        assert engine.initialized is True
    
    def test_solve_problem_simplified(self):
        """Test problem solving in simplified mode."""
        engine = ROMAMDAPMakerAssociativeEngine()
        result = engine.solve_problem("Test problem")
        
        assert "success" in result
        assert result["success"] is True
        assert "problem" in result
        assert result["problem"] == "Test problem"
        assert "solution" in result
        assert "confidence" in result
        assert "total_time" in result
        assert "error_free" in result
    
    def test_solve_problem_recursive_simplified(self):
        """Test recursive problem solving in simplified mode."""
        engine = ROMAMDAPMakerAssociativeEngine()
        result = engine.solve_problem_recursive("Test problem")
        
        assert "success" in result
        assert result["success"] is True
    
    def test_get_metrics(self):
        """Test getting metrics."""
        engine = ROMAMDAPMakerAssociativeEngine()
        metrics = engine.get_metrics()
        
        assert "total_problems_solved" in metrics
        assert "total_decomposition_time" in metrics
        assert "total_recomposition_time" in metrics
        assert "total_validation_time" in metrics
        assert "avg_confidence" in metrics
        assert "total_sub_solutions" in metrics
        assert "successful_recompositions" in metrics
        assert "failed_recompositions" in metrics
    
    def test_reset_metrics(self):
        """Test resetting metrics."""
        engine = ROMAMDAPMakerAssociativeEngine()
        engine._metrics["total_problems_solved"] = 10
        engine.reset_metrics()
        
        assert engine._metrics["total_problems_solved"] == 0


# =============================================================================
# create_romamdapmaker_associative_config Tests
# =============================================================================

class TestCreateConfig:
    """Tests for create_romamdapmaker_associative_config function."""
    
    def test_create_config_default(self):
        """Test creating default config."""
        config = create_romamdapmaker_associative_config()
        
        assert isinstance(config, ROMAMDAPMakerAssociativeConfig)
        assert config.roma_max_depth_analysis == 3
    
    def test_create_config_with_preset(self):
        """Test creating config with preset."""
        config = create_romamdapmaker_associative_config(preset="fast")
        
        assert isinstance(config, ROMAMDAPMakerAssociativeConfig)
    
    def test_create_config_with_overrides(self):
        """Test creating config with overrides."""
        config = create_romamdapmaker_associative_config(
            roma_max_depth_analysis=10,
            mdap_k_ahead=5,
            provider="anthropic"
        )
        
        assert config.roma_max_depth_analysis == 10
        assert config.mdap_k_ahead == 5
        assert config.provider == "anthropic"
    
    def test_create_config_with_kwargs(self):
        """Test creating config with kwargs."""
        config = create_romamdapmaker_associative_config(
            custom_param="value",
            another_param=123
        )
        
        assert isinstance(config, ROMAMDAPMakerAssociativeConfig)


# =============================================================================
# solve_with_romamdapmaker_associative Tests
# =============================================================================

class TestSolveWithAssociative:
    """Tests for solve_with_romamdapmaker_associative function."""
    
    def test_solve_default_config(self):
        """Test solving with default config."""
        result = solve_with_romamdapmaker_associative("Test problem")
        
        assert "success" in result
        assert result["success"] is True
    
    def test_solve_with_custom_config(self):
        """Test solving with custom config."""
        config = ROMAMDAPMakerAssociativeConfig(
            roma_max_depth_analysis=5
        )
        result = solve_with_romamdapmaker_associative(
            "Test problem",
            config=config
        )
        
        assert "success" in result
    
    def test_solve_with_context(self):
        """Test solving with context."""
        result = solve_with_romamdapmaker_associative(
            "Test problem",
            context={"domain": "mathematics"}
        )
        
        assert "success" in result
    
    def test_solve_with_llm_call_fn(self, mock_llm_call):
        """Test solving with custom LLM call function."""
        result = solve_with_romamdapmaker_associative(
            "Test problem",
            llm_call_fn=mock_llm_call
        )
        
        assert "success" in result
    
    def test_solve_recursive_true(self):
        """Test solving with recursive=True."""
        result = solve_with_romamdapmaker_associative(
            "Test problem",
            recursive=True
        )
        
        assert "success" in result
    
    def test_solve_recursive_false(self):
        """Test solving with recursive=False."""
        result = solve_with_romamdapmaker_associative(
            "Test problem",
            recursive=False
        )
        
        assert "success" in result


# =============================================================================
# get_romamdapmaker_associative_status Tests
# =============================================================================

class TestGetStatus:
    """Tests for get_romamdapmaker_associative_status function."""
    
    def test_get_status(self):
        """Test getting system status."""
        status = get_romamdapmaker_associative_status()
        
        assert "roma_mdap_maker_available" in status
        assert "associative_available" in status
        assert "ground_truth_available" in status
        assert "full_system_available" in status
        assert "components" in status
        assert "description" in status
    
    def test_get_status_components(self):
        """Test status components."""
        status = get_romamdapmaker_associative_status()
        
        assert "roma_mdap_maker" in status["components"]
        assert "associative_recomposition" in status["components"]
        assert "ground_truth_store" in status["components"]


# =============================================================================
# Async Tests
# =============================================================================

class TestAsyncMethods:
    """Tests for async methods."""
    
    @pytest.mark.asyncio
    async def test_async_solve_problem(self):
        """Test async problem solving."""
        engine = ROMAMDAPMakerAssociativeEngine()
        
        # In simplified mode, solve_problem is synchronous
        result = engine.solve_problem("Test problem")
        
        assert "success" in result


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for ROMA associative integration."""
    
    def test_full_workflow(self):
        """Test complete workflow from config to solve."""
        # Create config
        config = create_romamdapmaker_associative_config(
            roma_max_depth_analysis=3,
            mdap_enabled=True
        )
        
        # Create engine
        engine = ROMAMDAPMakerAssociativeEngine(config)
        
        # Initialize
        assert engine.initialize() is True
        
        # Plan decomposition
        plan = engine.plan_decomposition(
            "Solve x + 2 = 5",
            "mathematics"
        )
        assert "problem" in plan
        
        # Solve problem
        result = engine.solve_problem("Solve x + 2 = 5")
        assert "success" in result
        
        # Get metrics
        metrics = engine.get_metrics()
        assert "total_problems_solved" in metrics
    
    def test_config_modification(self):
        """Test that config can be modified."""
        config = ROMAMDAPMakerAssociativeConfig()
        engine = ROMAMDAPMakerAssociativeEngine(config)
        
        # Modify config
        config.roma_max_depth_analysis = 10
        config.mdap_enabled = False
        
        # Verify changes
        assert engine.config.roma_max_depth_analysis == 10
        assert engine.config.mdap_enabled is False


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
