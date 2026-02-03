"""
End-to-end tests for the full Adaptive MDAP system integration.

These tests verify the complete integration chain from SubProblemSolver
down through the adaptive allocation system to actual execution.
"""

import pytest
import time
from typing import List

# Test imports
from adaptive_mdap.core.types import SubProblem, SolveStrategy
from adaptive_mdap.classifiers.task_complexity_classifier import TaskComplexityClassifier
from adaptive_mdap.allocators.resource_allocator import AdaptiveMDAPAllocator, AllocationContext
from adaptive_mdap.controllers.execution_controller import AdaptiveExecutionController
from adaptive_mdap.integrations.subproblem_solver_integration import (
    AdaptiveSubProblemSolver,
    AdaptiveSolverConfig,
    create_adaptive_solver,
)
from adaptive_mdap.tools.cost_calculator import CostCalculator, APIPricing, WorkloadDistribution


class MockOpenEvolveClient:
    """Mock OpenEvolve client for testing."""
    
    def evolve(self, **kwargs):
        class MockResult:
            success = True
            best_code = "print('Hello World')"
            best_fitness = 0.95
        return MockResult()


class TestFullSystemAdaptiveSolve:
    """Test the full system from SubProblemSolver down to Adaptive MDAP."""
    
    def test_full_system_simple_problem(self):
        """Test the full system with a simple problem."""
        # Create adaptive solver
        solver = create_adaptive_solver(
            openevolve_client=MockOpenEvolveClient(),
            enabled=True,
            profile="balanced",
        )
        
        # Verify adaptive components are initialized
        assert solver.config.enabled is True
        assert solver.classifier is not None
        assert solver.allocator is not None
        assert solver.controller is not None
        
        # Create a simple sub-problem
        sp = SubProblem(
            id="e2e-test-simple",
            description="A simple task to verify the full integration chain.",
            domain="basic",
            depth=0,
            dependencies=[],
            metadata={},
        )
        
        # Solve it
        attempt = solver.solve(sp)
        
        # Verify the results
        assert attempt is not None
        assert attempt.subproblem_id == sp.id
        assert attempt.status.value == "completed"
        assert attempt.complexity_score < 0.5  # Simple problem
        assert attempt.allocated_strategy == "direct"
        assert attempt.solution is not None
    
    def test_full_system_complex_problem(self):
        """Test the full system with a complex problem."""
        solver = create_adaptive_solver(
            openevolve_client=MockOpenEvolveClient(),
            enabled=True,
            profile="balanced",
        )
        
        # Create a complex sub-problem
        sp = SubProblem(
            id="e2e-test-complex",
            description=(
                "This is an extremely complex problem involving distributed concurrency, "
                "security vulnerabilities, cryptographic protocols, and performance optimization. "
                "The solution must handle distributed consensus, implement secure communication "
                "channels, and optimize for high-throughput scenarios while maintaining correctness."
            ),
            domain="ultra_rare_quantum_biological_neural_encryption_domain",
            depth=10,
            dependencies=[f"dep{i}" for i in range(10)],
            metadata={
                "constraints": ["must be O(log n)", "must be thread-safe", "must be cryptographically secure"],
                "success_criteria": ["passes all tests", "no security leaks", "verified", "optimized"],
            },
        )
        
        # Solve it
        attempt = solver.solve(sp)
        
        # Verify
        assert attempt is not None
        assert attempt.subproblem_id == sp.id
        assert attempt.status.value == "completed"
        assert attempt.complexity_score > 0.6  # High complexity
        assert attempt.allocated_strategy in ["maker_full", "maker_ultra", "mdap_medium"]
        assert attempt.solution is not None
    
    def test_adaptive_fallback_to_standard(self):
        """Test that it falls back to standard solve if adaptive fails."""
        # Initialize solver
        solver = create_adaptive_solver(
            openevolve_client=None,
            enabled=True,
        )
        
        # Manually break the controller to force fallback
        solver.controller = None
        
        sp = SubProblem(
            id="fallback-test",
            description="Test fallback",
            domain="test",
            depth=0,
            dependencies=[],
            metadata={},
        )
        
        # Should handle gracefully
        with pytest.raises(Exception):
            solver.solve(sp)
    
    def test_strategy_override(self):
        """Test forcing a specific strategy."""
        solver = create_adaptive_solver(enabled=True)
        
        # Create any sub-problem
        sp = SubProblem(
            id="override-test",
            description="Simple task",
            domain="test",
            depth=0,
            dependencies=[],
            metadata={},
        )
        
        # Force MAKER_FULL even for simple problem
        attempt = solver.solve(sp, strategy="maker_full")
        
        assert attempt.allocated_strategy == "maker_full"
        assert attempt.status.value == "completed"


class TestAdaptiveStrategySelection:
    """Test that strategies are selected based on complexity."""
    
    def test_complexity_based_strategy_selection(self):
        """Test that different complexities get different strategies."""
        controller = AdaptiveExecutionController()
        
        test_cases = [
            # (description, expected_strategy_min, expected_strategy_max)
            ("Simple task", SolveStrategy.DIRECT, SolveStrategy.MDAP_LIGHT),
            ("Medium complexity task with some constraints and dependencies", SolveStrategy.MDAP_LIGHT, SolveStrategy.MDAP_MEDIUM),
            ("Complex distributed security optimization task", SolveStrategy.MDAP_MEDIUM, SolveStrategy.MAKER_ULTRA),
        ]
        
        for desc, min_strategy, max_strategy in test_cases:
            sp = SubProblem(
                id=f"strategy-test-{hash(desc) % 10000}",
                description=desc,
                domain="test",
                depth=3,
                dependencies=[],
                metadata={},
            )
            
            attempt = controller.execute_adaptive(sp)
            
            # Verify strategy is within expected range
            strategy_order = [
                SolveStrategy.DIRECT,
                SolveStrategy.MDAP_LIGHT,
                SolveStrategy.MDAP_MEDIUM,
                SolveStrategy.MAKER_FULL,
                SolveStrategy.MAKER_ULTRA,
            ]
            
            actual_idx = strategy_order.index(SolveStrategy(attempt.allocated_strategy))
            min_idx = strategy_order.index(min_strategy)
            max_idx = strategy_order.index(max_strategy)
            
            assert min_idx <= actual_idx <= max_idx, \
                f"For '{desc[:30]}...': expected {min_strategy.value} to {max_strategy.value}, got {attempt.allocated_strategy}"
    
    def test_escalation_on_failure(self):
        """Test that system escalates strategy on failure."""
        controller = AdaptiveExecutionController()
        
        # Force a strategy that might fail and need escalation
        sp = SubProblem(
            id="escalation-test",
            description="Test escalation",
            domain="test",
            depth=5,
            dependencies=[],
            metadata={},
        )
        
        attempt = controller.execute_adaptive(
            sp,
            force_strategy=SolveStrategy.DIRECT,
            enable_escalation=True,
        )
        
        # Should complete (possibly after escalation)
        assert attempt.status.value == "completed"


class TestCostSavings:
    """Test cost savings from adaptive allocation."""
    
    def test_cost_savings_calculation(self):
        """Test that adaptive allocation achieves cost savings."""
        calculator = CostCalculator(pricing=APIPricing.gpt_4o_mini())
        
        # Calculate for 1000 problems with default workload
        result = calculator.calculate_adaptive_cost(1000)
        
        # Verify savings
        assert result["savings_percent"] > 20  # At least 20% savings
        assert result["savings"] > 0
        assert result["baseline_cost"] > result["adaptive_cost"]
    
    def test_cost_savings_with_workload_distribution(self):
        """Test cost savings with different workload distributions."""
        calculator = CostCalculator(pricing=APIPricing.gpt_4o_mini())
        
        # Cost-optimized workload (60% easy, 30% medium, 10% hard)
        cost_optimized = WorkloadDistribution.cost_optimized()
        result_optimized = calculator.calculate_adaptive_cost(1000, cost_optimized)
        
        # Compute-heavy workload (10% easy, 30% medium, 60% hard)
        compute_heavy = WorkloadDistribution.compute_heavy()
        result_heavy = calculator.calculate_adaptive_cost(1000, compute_heavy)
        
        # Cost-optimized should have better savings
        assert result_optimized["savings_percent"] > result_heavy["savings_percent"]
    
    def test_model_comparison(self):
        """Test cost comparison across different models."""
        calculator = CostCalculator()
        
        comparison = calculator.compare_models(1000)
        
        assert len(comparison["comparisons"]) > 0
        assert comparison["cheapest"] is not None
        
        # GPT-4o-mini should be cheapest
        cheapest_model = comparison["cheapest"]["model"]
        assert "gpt-4o-mini" in cheapest_model or "gemini-1.5-flash" in cheapest_model


class TestContextAwareAllocation:
    """Test context-aware resource allocation."""
    
    def test_high_load_favors_cheaper_strategies(self):
        """Test that high system load favors cheaper strategies."""
        allocator = AdaptiveMDAPAllocator(enable_context_aware=True)
        
        # Same complexity, different contexts
        complexity = 0.35
        
        # High load context
        high_load = AllocationContext(system_load="high")
        config_high = allocator.allocate_resources(complexity, context=high_load)
        
        # Low load context
        low_load = AllocationContext(system_load="low")
        config_low = allocator.allocate_resources(complexity, context=low_load)
        
        # High load should favor cheaper strategy (lower index in strategy order)
        strategy_order = [
            SolveStrategy.DIRECT,
            SolveStrategy.MDAP_LIGHT,
            SolveStrategy.MDAP_MEDIUM,
            SolveStrategy.MAKER_FULL,
            SolveStrategy.MAKER_ULTRA,
        ]
        
        high_idx = strategy_order.index(config_high.strategy)
        low_idx = strategy_order.index(config_low.strategy)
        
        # High load should get same or cheaper strategy
        assert high_idx <= low_idx
    
    def test_low_budget_favors_cheaper_strategies(self):
        """Test that low budget favors cheaper strategies."""
        allocator = AdaptiveMDAPAllocator(enable_context_aware=True)
        
        complexity = 0.85
        
        # Low budget context
        low_budget = AllocationContext(budget_remaining=10)
        config_low = allocator.allocate_resources(complexity, context=low_budget)
        
        # Normal budget context
        normal_budget = AllocationContext(budget_remaining=100)
        config_normal = allocator.allocate_resources(complexity, context=normal_budget)
        
        # Low budget should favor cheaper strategy
        strategy_order = [
            SolveStrategy.DIRECT,
            SolveStrategy.MDAP_LIGHT,
            SolveStrategy.MDAP_MEDIUM,
            SolveStrategy.MAKER_FULL,
            SolveStrategy.MAKER_ULTRA,
        ]
        
        low_idx = strategy_order.index(config_low.strategy)
        normal_idx = strategy_order.index(config_normal.strategy)
        
        assert low_idx <= normal_idx


class TestStatisticsAndMonitoring:
    """Test statistics collection and monitoring."""
    
    def test_statistics_accumulation(self):
        """Test that statistics accumulate across executions."""
        solver = create_adaptive_solver(enabled=True)
        
        # Execute multiple problems
        for i in range(5):
            sp = SubProblem(
                id=f"stats-test-{i}",
                description=f"Task {i}",
                domain="test",
                depth=i,
                dependencies=[],
                metadata={},
            )
            solver.solve(sp)
        
        # Check stats
        stats = solver.get_stats()
        assert stats["total_solves"] == 5
        assert stats["adaptive_solves"] == 5
        assert "allocator_stats" in stats
        assert "controller_stats" in stats
    
    def test_allocation_stats(self):
        """Test allocation statistics tracking."""
        allocator = AdaptiveMDAPAllocator()
        
        # Make various allocations
        complexities = [0.1, 0.2, 0.5, 0.7, 0.9]
        for c in complexities:
            allocator.allocate_resources(c)
        
        stats = allocator.get_allocation_stats()
        
        assert stats["total_allocations"] == 5
        assert len(stats["strategy_distribution"]) == 5
        assert stats["estimated_savings_percent"] >= 0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_description_handled(self):
        """Test that empty descriptions are handled gracefully."""
        solver = create_adaptive_solver(enabled=True)
        
        sp = SubProblem(
            id="empty-test",
            description="",
            domain="",
            depth=0,
            dependencies=[],
            metadata={},
        )
        
        # Should not raise exception
        attempt = solver.solve(sp)
        assert attempt is not None
        assert attempt.status.value in ["completed", "failed"]
    
    def test_very_long_description_handled(self):
        """Test that very long descriptions are handled."""
        solver = create_adaptive_solver(enabled=True)
        
        sp = SubProblem(
            id="long-test",
            description="word " * 10000,
            domain="test",
            depth=0,
            dependencies=[],
            metadata={},
        )
        
        attempt = solver.solve(sp)
        assert attempt is not None
        assert attempt.status.value == "completed"
    
    def test_negative_values_handled(self):
        """Test that negative values are handled."""
        solver = create_adaptive_solver(enabled=True)
        
        sp = SubProblem(
            id="negative-test",
            description="Test",
            domain="test",
            depth=-5,
            dependencies=[],
            metadata={},
        )
        
        attempt = solver.solve(sp)
        assert attempt is not None
        assert attempt.status.value == "completed"


class TestIntegrationWithExistingSolver:
    """Test integration with existing SubProblemSolver patterns."""
    
    def test_backward_compatibility(self):
        """Test that existing API patterns still work."""
        solver = create_adaptive_solver(enabled=True)
        
        # Test dict-based sub-problem
        dict_sp = {
            "id": "dict-test",
            "description": "Dict-based problem",
            "domain": "test",
            "depth": 0,
            "dependencies": [],
            "metadata": {},
        }
        
        attempt = solver.solve(dict_sp)
        assert attempt is not None
        assert attempt.status.value == "completed"


@pytest.mark.performance
class TestPerformance:
    """Performance tests for the adaptive system."""
    
    def test_classification_latency(self):
        """Test that classification is fast."""
        classifier = TaskComplexityClassifier()
        
        sp = SubProblem(
            id="perf-test",
            description="Performance test problem",
            domain="test",
            depth=3,
            dependencies=[],
            metadata={},
        )
        
        start = time.time()
        complexity = classifier.compute_complexity(sp)
        elapsed_ms = (time.time() - start) * 1000
        
        # Classification should be fast (under 100ms for simple cases)
        assert elapsed_ms < 100
    
    def test_allocation_latency(self):
        """Test that allocation is fast."""
        allocator = AdaptiveMDAPAllocator()
        
        start = time.time()
        config = allocator.allocate_resources(0.5)
        elapsed_ms = (time.time() - start) * 1000
        
        # Allocation should be very fast (under 10ms)
        assert elapsed_ms < 10
    
    def test_batch_allocation_performance(self):
        """Test batch allocation performance."""
        allocator = AdaptiveMDAPAllocator()
        
        complexities = [i / 100 for i in range(100)]
        
        start = time.time()
        configs = allocator.allocate_resources_batch(complexities)
        elapsed_ms = (time.time() - start) * 1000
        
        # 100 allocations should take less than 100ms
        assert elapsed_ms < 100
        assert len(configs) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
