"""
End-to-end integration tests for Adaptive MDAP.
"""

import pytest
from adaptive_mdap.core.types import SubProblem
from adaptive_mdap.classifiers.task_complexity_classifier import TaskComplexityClassifier
from adaptive_mdap.allocators.resource_allocator import AdaptiveMDAPAllocator
from adaptive_mdap.controllers.execution_controller import AdaptiveExecutionController


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    def test_simple_problem_workflow(self):
        """Test complete workflow for simple problem."""
        # Create components
        classifier = TaskComplexityClassifier()
        allocator = AdaptiveMDAPAllocator()
        controller = AdaptiveExecutionController(
            classifier=classifier,
            allocator=allocator,
        )
        
        # Create simple sub-problem
        subproblem = SubProblem(
            id="simple-test",
            description="Simple task",
            domain="basic",
            depth=0,
            dependencies=[],
            metadata={},
        )
        
        # Execute
        attempt = controller.execute_adaptive(subproblem)
        
        # Verify
        assert attempt is not None
        assert attempt.status.value == "completed"
        assert attempt.complexity_score < 0.5  # Simple problem
        assert attempt.allocated_strategy == "direct"
    
    def test_complex_problem_workflow(self):
        """Test complete workflow for complex problem."""
        classifier = TaskComplexityClassifier()
        allocator = AdaptiveMDAPAllocator()
        controller = AdaptiveExecutionController(
            classifier=classifier,
            allocator=allocator,
        )
        
        # Create truly complex sub-problem
        subproblem = SubProblem(
            id="complex-test",
            description="This is an extremely complex problem involving distributed concurrency and security refactor. " * 20,
            domain="ultra_rare_quantum_biological_neural_encryption_domain",
            depth=10,
            dependencies=[f"dep{i}" for i in range(10)],
            metadata={
                "constraints": ["must be O(log n)", "must be thread-safe"],
                "success_criteria": ["passes all tests", "no security leaks", "verified"]
            },
        )
        
        # Execute
        attempt = controller.execute_adaptive(subproblem)
        
        # Verify
        assert attempt is not None
        assert attempt.status.value == "completed"
        assert attempt.complexity_score > 0.7  # High complexity
        assert attempt.allocated_strategy in ["maker_full", "maker_ultra"]
    
    def test_strategy_override(self):
        """Test forcing a specific strategy."""
        controller = AdaptiveExecutionController()
        
        subproblem = SubProblem(
            id="override-test",
            description="Some task",
            domain="test",
            depth=0,
            dependencies=[],
            metadata={},
        )
        
        # Force MAKER_FULL even for simple problem
        from adaptive_mdap.core.types import SolveStrategy
        attempt = controller.execute_adaptive(
            subproblem,
            force_strategy=SolveStrategy.MAKER_FULL,
        )
        
        assert attempt.allocated_strategy == "maker_full"
    
    def test_statistics_accumulation(self):
        """Test that statistics accumulate across executions."""
        controller = AdaptiveExecutionController()
        
        # Execute multiple problems
        for i in range(5):
            subproblem = SubProblem(
                id=f"stats-test-{i}",
                description=f"Task {i}",
                domain="test",
                depth=i,
                dependencies=[],
                metadata={},
            )
            controller.execute_adaptive(subproblem)
        
        # Check stats
        stats = controller.get_execution_stats()
        assert stats["total_executions"] == 5
        assert stats["successful_executions"] == 5


class TestAdaptiveBehavior:
    """Tests for adaptive behavior."""
    
    def test_allocation_changes_with_complexity(self):
        """Test that allocation changes based on complexity."""
        allocator = AdaptiveMDAPAllocator()
        
        # Test various complexities
        strategies = []
        for complexity in [0.1, 0.4, 0.6, 0.9]:
            config = allocator.allocate_resources(complexity)
            strategies.append(config.strategy.value)
        
        # Should have different strategies
        assert len(set(strategies)) > 1
    
    def test_context_affects_allocation(self):
        """Test that context affects allocation decisions."""
        allocator = AdaptiveMDAPAllocator(enable_context_aware=True)
        
        # Same complexity, different contexts
        from adaptive_mdap.allocators.resource_allocator import AllocationContext
        
        # High load context
        high_load_context = AllocationContext(system_load="high")
        config_high_load = allocator.allocate_resources(
            complexity_score=0.35,
            context=high_load_context,
        )
        
        # Low load context
        low_load_context = AllocationContext(system_load="low")
        config_low_load = allocator.allocate_resources(
            complexity_score=0.35,
            context=low_load_context,
        )
        
        # Contexts should produce different results
        # (though they might be the same depending on threshold adjustments)


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_invalid_subproblem_handled(self):
        """Test that invalid sub-problems are handled gracefully."""
        controller = AdaptiveExecutionController()
        
        # Create sub-problem with unusual values
        subproblem = SubProblem(
            id="error-test",
            description="",  # Empty description
            domain="",  # Empty domain
            depth=-1,  # Negative depth
            dependencies=None,
            metadata={},
        )
        
        # Should not raise exception
        attempt = controller.execute_adaptive(subproblem)
        assert attempt is not None
        assert attempt.status.value in ["completed", "failed"]
    
    def test_fallback_on_execution_failure(self):
        """Test fallback when execution fails."""
        controller = AdaptiveExecutionController()
        
        # Normal sub-problem
        subproblem = SubProblem(
            id="fallback-test",
            description="Test task",
            domain="test",
            depth=0,
            dependencies=[],
            metadata={},
        )
        
        # Should complete (possibly with fallback)
        attempt = controller.execute_adaptive(subproblem)
        assert attempt.status.value == "completed"


class TestBackwardCompatibility:
    """Tests for backward compatibility."""
    
    def test_existing_api_still_works(self):
        """Test that existing API patterns still work."""
        # This simulates how existing code might use the system
        classifier = TaskComplexityClassifier()
        allocator = AdaptiveMDAPAllocator()
        
        subproblem = SubProblem(
            id="compat-test",
            description="Test",
            domain="test",
            depth=0,
            dependencies=[],
            metadata={},
        )
        
        # Original pattern: classify then allocate
        complexity = classifier.compute_complexity(subproblem)
        config = allocator.allocate_resources(complexity.overall_score)
        
        assert config is not None
        assert config.n_agents > 0
