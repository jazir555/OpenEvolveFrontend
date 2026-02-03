"""Tests for Resource Allocator."""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from adaptive_mdap.allocators.resource_allocator import (
    AdaptiveMDAPAllocator,
    AllocationContext,
    AllocationStats,
    SolveConfig,
    SolveStrategy,
)
from adaptive_mdap.core.types import ComplexityScore


class TestAllocationContext:
    """Tests for AllocationContext."""
    
    def test_default_context(self):
        """Test default context values."""
        context = AllocationContext()
        
        assert context.time_of_day is None
        assert context.system_load is None
        assert context.budget_remaining is None
        assert context.quality_requirements is None
    
    def test_custom_context(self):
        """Test custom context values."""
        context = AllocationContext(
            time_of_day="business_hours",
            system_load="low",
            budget_remaining=80.0,
            quality_requirements="strict",
        )
        
        assert context.time_of_day == "business_hours"
        assert context.system_load == "low"
        assert context.budget_remaining == 80.0
        assert context.quality_requirements == "strict"
    
    def test_from_system_state(self):
        """Test creating context from system state."""
        context = AllocationContext.from_system_state()
        
        assert context.time_of_day in ["business_hours", "off_hours"]
        assert context.system_load in ["high", "medium", "low"]
        assert context.budget_remaining == 100.0
        assert context.quality_requirements == "normal"


class TestAllocationStats:
    """Tests for AllocationStats."""
    
    def test_default_stats(self):
        """Test default statistics values."""
        stats = AllocationStats()
        
        assert stats.total_allocations == 0
        assert "direct" in stats.strategy_counts
        assert "mdap_light" in stats.strategy_counts
        assert "medium-low" in stats.complexity_band_counts
    
    def test_increment_strategy(self):
        """Test strategy counting."""
        stats = AllocationStats()
        
        stats.strategy_counts["direct"] += 1
        stats.strategy_counts["maker_full"] += 1
        
        assert stats.strategy_counts["direct"] == 1
        assert stats.strategy_counts["maker_full"] == 1
    
    def test_increment_complexity_band(self):
        """Test complexity band counting."""
        stats = AllocationStats()
        
        stats.complexity_band_counts["low"] += 1
        stats.complexity_band_counts["high"] += 1
        
        assert stats.complexity_band_counts["low"] == 1
        assert stats.complexity_band_counts["high"] == 1


class TestSolveConfig:
    """Tests for SolveConfig."""
    
    def test_valid_config(self):
        """Test creating valid config."""
        config = SolveConfig(
            strategy=SolveStrategy.DIRECT,
            n_agents=1,
            k_ahead=0,
            max_retries=1,
        )
        
        assert config.strategy == SolveStrategy.DIRECT
        assert config.n_agents == 1
    
    def test_invalid_n_agents(self):
        """Test that invalid n_agents raises error."""
        with pytest.raises(ValueError, match="n_agents must be > 0"):
            SolveConfig(
                strategy=SolveStrategy.DIRECT,
                n_agents=0,
                k_ahead=0,
                max_retries=1,
            )
    
    def test_invalid_k_ahead(self):
        """Test that negative k_ahead raises error."""
        with pytest.raises(ValueError, match="k_ahead must be >= 0"):
            SolveConfig(
                strategy=SolveStrategy.DIRECT,
                n_agents=1,
                k_ahead=-1,
                max_retries=1,
            )
    
    def test_invalid_max_retries(self):
        """Test that negative max_retries raises error."""
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            SolveConfig(
                strategy=SolveStrategy.DIRECT,
                n_agents=1,
                k_ahead=0,
                max_retries=-1,
            )


class TestAdaptiveMDAPAllocator:
    """Tests for AdaptiveMDAPAllocator."""
    
    def test_allocator_initialization(self):
        """Test allocator can be initialized."""
        allocator = AdaptiveMDAPAllocator()
        
        assert allocator is not None
        assert allocator._stats is not None
    
    def test_allocate_direct_strategy(self):
        """Test allocation of simple problem to DIRECT strategy."""
        allocator = AdaptiveMDAPAllocator()
        
        score = ComplexityScore(
            overall_score=0.1,  # Low complexity
            text_length_score=0.1,
            domain_rarity_score=0.1,
            depth_score=0.1,
            historical_error_score=0.1,
            dependency_score=0.1,
            feature_weights={},
        )
        
        decision = allocator.allocate(score)
        
        assert decision.complexity_score == 0.1
        assert decision.allocated_strategy == SolveStrategy.DIRECT
        assert decision.config.n_agents == 1
    
    def test_allocate_mdap_light_strategy(self):
        """Test allocation of medium problem to MDAP_LIGHT strategy."""
        allocator = AdaptiveMDAPAllocator()
        
        score = ComplexityScore(
            overall_score=0.3,  # Medium complexity
            text_length_score=0.3,
            domain_rarity_score=0.3,
            depth_score=0.3,
            historical_error_score=0.3,
            dependency_score=0.3,
            feature_weights={},
        )
        
        decision = allocator.allocate(score)
        
        assert decision.config.n_agents == 3
        assert decision.config.k_ahead == 1
    
    def test_allocate_maker_full_strategy(self):
        """Test allocation of complex problem to MAKER_FULL strategy."""
        allocator = AdaptiveMDAPAllocator()
        
        score = ComplexityScore(
            overall_score=0.8,  # High complexity
            text_length_score=0.8,
            domain_rarity_score=0.8,
            depth_score=0.8,
            historical_error_score=0.8,
            dependency_score=0.8,
            feature_weights={},
        )
        
        decision = allocator.allocate(score)
        
        assert decision.config.n_agents == 5
        assert decision.config.k_ahead == 2
    
    def test_allocate_maker_ultra_strategy(self):
        """Test allocation of very complex problem to MAKER_ULTRA strategy."""
        allocator = AdaptiveMDAPAllocator()
        
        score = ComplexityScore(
            overall_score=0.95,  # Very high complexity
            text_length_score=0.95,
            domain_rarity_score=0.95,
            depth_score=0.95,
            historical_error_score=0.95,
            dependency_score=0.95,
            feature_weights={},
        )
        
        decision = allocator.allocate(score)
        
        assert decision.allocated_strategy == SolveStrategy.MAKER_ULTRA
        assert decision.config.n_agents == 7
        assert decision.config.k_ahead == 3
    
    def test_estimated_cost(self):
        """Test cost estimation."""
        allocator = AdaptiveMDAPAllocator()
        
        score = ComplexityScore(
            overall_score=0.5,
            text_length_score=0.5,
            domain_rarity_score=0.5,
            depth_score=0.5,
            historical_error_score=0.5,
            dependency_score=0.5,
            feature_weights={},
        )
        
        decision = allocator.allocate(score)
        
        assert decision.estimated_cost > 0
        assert decision.estimated_quality > 0
    
    def test_allocate_with_context(self):
        """Test allocation with context awareness."""
        allocator = AdaptiveMDAPAllocator()
        
        score = ComplexityScore(
            overall_score=0.5,
            text_length_score=0.5,
            domain_rarity_score=0.5,
            depth_score=0.5,
            historical_error_score=0.5,
            dependency_score=0.5,
            feature_weights={},
        )
        
        context = AllocationContext(
            quality_requirements="strict",
            budget_remaining=50.0,
        )
        
        decision = allocator.allocate(score, context)
        
        # Strict quality might choose higher strategy
        assert decision is not None
    
    def test_strategy_counts_updated(self):
        """Test that strategy counts are updated."""
        allocator = AdaptiveMDAPAllocator()
        
        score = ComplexityScore(
            overall_score=0.1,
            text_length_score=0.1,
            domain_rarity_score=0.1,
            depth_score=0.1,
            historical_error_score=0.1,
            dependency_score=0.1,
            feature_weights={},
        )
        
        allocator.allocate(score)
        
        assert allocator._stats.strategy_counts["direct"] == 1
        assert allocator._stats.total_allocations == 1
    
    def test_threshold_boundaries(self):
        """Test allocation at threshold boundaries."""
        allocator = AdaptiveMDAPAllocator()
        
        # At 0.2 boundary
        score = ComplexityScore(
            overall_score=0.2,
            text_length_score=0.2,
            domain_rarity_score=0.2,
            depth_score=0.2,
            historical_error_score=0.2,
            dependency_score=0.2,
            feature_weights={},
        )
        
        decision = allocator.allocate(score)
        # Should be allocated to mdap_light or higher
        assert decision.config.n_agents >= 3


class TestSolveStrategy:
    """Tests for SolveStrategy enum."""
    
    def test_strategy_values(self):
        """Test strategy enum values."""
        assert SolveStrategy.DIRECT.value == "direct"
        assert SolveStrategy.MDAP_LIGHT.value == "mdap_light"
        assert SolveStrategy.MDAP_MEDIUM.value == "mdap_medium"
        assert SolveStrategy.MAKER_FULL.value == "maker_full"
        assert SolveStrategy.MAKER_ULTRA.value == "maker_ultra"
    
    def test_all_strategies_have_configs(self):
        """Test that all strategies have default configs."""
        allocator = AdaptiveMDAPAllocator()
        
        for strategy in SolveStrategy:
            config = allocator._get_default_config(strategy)
            assert config is not None
            assert config.strategy == strategy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
