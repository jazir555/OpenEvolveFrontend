"""
Unit tests for AdaptiveMDAPAllocator.
"""

import pytest
from adaptive_mdap.core.types import SolveStrategy
from adaptive_mdap.allocators.resource_allocator import (
    AdaptiveMDAPAllocator,
    AllocationContext,
    AllocationError,
)


class TestAllocationLogic:
    """Tests for core allocation logic."""
    
    def test_low_complexity_gets_direct(self, allocator):
        """Test low complexity gets DIRECT strategy."""
        config = allocator.allocate_resources(complexity_score=0.1)
        assert config.strategy == SolveStrategy.DIRECT
    
    def test_medium_low_gets_mdap_light(self, allocator):
        """Test medium-low complexity gets MDAP_LIGHT."""
        config = allocator.allocate_resources(complexity_score=0.3)
        assert config.strategy == SolveStrategy.MDAP_LIGHT
        
    def test_medium_gets_mdap_medium(self, allocator):
        """Test medium complexity gets MDAP_MEDIUM strategy."""
        config = allocator.allocate_resources(complexity_score=0.5)
        assert config.strategy == SolveStrategy.MDAP_MEDIUM
        
    def test_medium_high_gets_maker_full(self, allocator):
        """Test medium-high gets MAKER_FULL."""
        config = allocator.allocate_resources(complexity_score=0.7)
        assert config.strategy == SolveStrategy.MAKER_FULL
    
    def test_high_complexity_gets_maker_ultra(self, allocator):
        """Test high complexity gets MAKER_ULTRA strategy."""
        config = allocator.allocate_resources(complexity_score=0.9)
        assert config.strategy == SolveStrategy.MAKER_ULTRA
    
    def test_boundary_thresholds(self, allocator):
        """Test exact boundaries."""
        # thresholds: [0.2, 0.4, 0.6, 0.8]
        assert allocator.allocate_resources(0.2).strategy == SolveStrategy.MDAP_LIGHT
        assert allocator.allocate_resources(0.4).strategy == SolveStrategy.MDAP_MEDIUM
        assert allocator.allocate_resources(0.6).strategy == SolveStrategy.MAKER_FULL
        assert allocator.allocate_resources(0.8).strategy == SolveStrategy.MAKER_ULTRA


class TestStrategyConfigs:
    """Tests for strategy configurations."""
    
    def test_direct_config(self):
        """Test DIRECT strategy config."""
        config = AdaptiveMDAPAllocator.DEFAULT_CONFIGS[SolveStrategy.DIRECT]
        assert config.n_agents == 1
        assert config.k_ahead == 0
        assert config.max_retries == 1
    
    def test_mdap_light_config(self):
        """Test MDAP_LIGHT strategy config."""
        config = AdaptiveMDAPAllocator.DEFAULT_CONFIGS[SolveStrategy.MDAP_LIGHT]
        assert config.n_agents == 3
        assert config.k_ahead == 1
        assert config.max_retries == 2
    
    def test_maker_full_config(self):
        """Test MAKER_FULL strategy config."""
        config = AdaptiveMDAPAllocator.DEFAULT_CONFIGS[SolveStrategy.MAKER_FULL]
        assert config.n_agents == 5
        assert config.k_ahead == 2
        assert config.max_retries == 3


class TestThresholdValidation:
    """Tests for threshold validation."""
    
    def test_valid_thresholds(self):
        """Test valid thresholds are accepted."""
        allocator = AdaptiveMDAPAllocator(thresholds=[0.1, 0.3, 0.5, 0.7])
        assert allocator.thresholds == [0.1, 0.3, 0.5, 0.7]
    
    def test_invalid_threshold_count(self):
        """Test invalid threshold count raises error."""
        with pytest.raises(AllocationError, match="exactly 4 values"):
            AdaptiveMDAPAllocator(thresholds=[0.3, 0.5])
    
    def test_thresholds_out_of_order(self):
        """Test thresholds out of order raise error."""
        with pytest.raises(AllocationError, match="strictly increasing"):
            AdaptiveMDAPAllocator(thresholds=[0.2, 0.1, 0.4, 0.6])
    
    def test_thresholds_out_of_range(self):
        """Test thresholds out of [0, 1] raise error."""
        with pytest.raises(AllocationError, match="in \[0, 1\]"):
            AdaptiveMDAPAllocator(thresholds=[-0.1, 0.3, 0.5, 0.7])
        
        with pytest.raises(AllocationError, match="in \[0, 1\]"):
            AdaptiveMDAPAllocator(thresholds=[0.3, 0.5, 0.7, 1.1])


class TestStatistics:
    """Tests for allocation statistics."""
    
    def test_initial_stats_empty(self, allocator):
        """Test initial stats are empty."""
        stats = allocator.get_allocation_stats()
        assert stats["total_allocations"] == 0
        assert stats["estimated_savings_percent"] == 0.0
    
    def test_stats_after_allocations(self, allocator):
        """Test stats after some allocations."""
        # thresholds: [0.2, 0.4, 0.6, 0.8]
        allocator.allocate_resources(0.1)  # DIRECT
        allocator.allocate_resources(0.15) # DIRECT
        allocator.allocate_resources(0.5)  # MDAP_MEDIUM
        allocator.allocate_resources(0.9)  # MAKER_ULTRA
        
        stats = allocator.get_allocation_stats()
        assert stats["total_allocations"] == 4
        
        # Check strategy distribution
        dist = stats["strategy_distribution"]
        assert dist[SolveStrategy.DIRECT.value] == 0.5  # 2/4
    
    def test_savings_estimation(self, allocator):
        """Test savings estimation."""
        # All DIRECT would have maximum savings
        for _ in range(10):
            allocator.allocate_resources(0.1)
        
        stats = allocator.get_allocation_stats()
        assert stats["estimated_savings_percent"] > 0
    
    def test_stats_reset(self, allocator):
        """Test stats reset."""
        allocator.allocate_resources(0.5)
        allocator.reset_stats()
        
        stats = allocator.get_allocation_stats()
        assert stats["total_allocations"] == 0


class TestContextAwareAllocation:
    """Tests for context-aware allocation."""
    
    def test_high_load_favors_cheaper(self, allocator):
        """Test high load favors cheaper strategies."""
        allocator.enable_context_aware = True
        
        # Original t1=0.2. High load +0.05 -> 0.25
        # Complexity 0.23 originally MDAP_LIGHT, now DIRECT
        context = AllocationContext(system_load="high")
        config = allocator.allocate_resources(0.23, context=context)
        assert config.strategy == SolveStrategy.DIRECT
    
    def test_low_budget_favors_cheaper(self, allocator):
        """Test low budget favors cheaper strategies."""
        allocator.enable_context_aware = True
        
        # Original t4=0.8. Low budget +0.1 -> 0.9
        # Complexity 0.85 originally MAKER_ULTRA, now MAKER_FULL
        context = AllocationContext(budget_remaining=10)
        config = allocator.allocate_resources(0.85, context=context)
        assert config.strategy == SolveStrategy.MAKER_FULL
    
    def test_strict_quality_favors_expensive(self, allocator):
        """Test strict quality favors expensive strategies."""
        allocator.enable_context_aware = True
        
        # Original t1=0.2. Strict quality -0.1 -> 0.1
        # Complexity 0.15 originally DIRECT, now MDAP_LIGHT
        context = AllocationContext(quality_requirements="strict")
        config = allocator.allocate_resources(0.15, context=context)
        assert config.strategy == SolveStrategy.MDAP_LIGHT


class TestInvalidInputs:
    """Tests for invalid input handling."""
    
    def test_negative_complexity(self, allocator):
        """Test negative complexity is clamped."""
        config = allocator.allocate_resources(complexity_score=-0.5)
        # Should treat as 0.0 (DIRECT)
        assert config.strategy == SolveStrategy.DIRECT
    
    def test_over_one_complexity(self, allocator):
        """Test >1 complexity is clamped."""
        config = allocator.allocate_resources(complexity_score=1.5)
        # Should treat as 1.0 (MAKER_ULTRA)
        assert config.strategy == SolveStrategy.MAKER_ULTRA
    
    def test_nan_complexity(self, allocator):
        """Test NaN complexity uses default."""
        import math
        config = allocator.allocate_resources(complexity_score=float('nan'))
        # Should use default 0.5 (MDAP_MEDIUM)
        assert config.strategy == SolveStrategy.MDAP_MEDIUM


class TestThresholdUpdates:
    """Tests for threshold updates."""
    
    def test_update_thresholds(self, allocator):
        """Test updating thresholds."""
        allocator.update_thresholds([0.1, 0.3, 0.5, 0.7], reason="testing")
        assert allocator.thresholds == [0.1, 0.3, 0.5, 0.7]
    
    def test_update_thresholds_with_reset(self, allocator):
        """Test updating thresholds with stats reset."""
        allocator.allocate_resources(0.5)
        allocator.update_thresholds([0.1, 0.3, 0.5, 0.7], reset_stats=True)
        
        stats = allocator.get_allocation_stats()
        assert stats["total_allocations"] == 0
    
    def test_invalid_update_rejected(self, allocator):
        """Test invalid threshold update is rejected."""
        with pytest.raises(AllocationError):
            allocator.update_thresholds([0.8, 0.2])  # Out of order


class TestBatchAllocation:
    """Tests for batch allocation."""
    
    def test_batch_allocation(self, allocator):
        """Test allocating for multiple scores."""
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        configs = allocator.allocate_resources_batch(scores)
        
        assert len(configs) == 5
        assert configs[0].strategy == SolveStrategy.DIRECT
        assert configs[1].strategy == SolveStrategy.MDAP_LIGHT
        assert configs[2].strategy == SolveStrategy.MDAP_MEDIUM
        assert configs[3].strategy == SolveStrategy.MAKER_FULL
        assert configs[4].strategy == SolveStrategy.MAKER_ULTRA
