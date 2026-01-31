"""Tests for Execution Controller."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import time

from adaptive_mdap.controllers.execution_controller import (
    AdaptiveExecutionController,
    SolutionStatus,
    SolutionAttempt,
)
from adaptive_mdap.core.types import (
    SubProblem,
    ComplexityScore,
    SolveConfig,
    SolveStrategy,
)


class TestSolutionStatus:
    """Tests for SolutionStatus enum."""
    
    def test_status_values(self):
        """Test status enum values."""
        assert SolutionStatus.PENDING.value == "pending"
        assert SolutionStatus.IN_PROGRESS.value == "in_progress"
        assert SolutionStatus.COMPLETED.value == "completed"
        assert SolutionStatus.FAILED.value == "failed"


class TestSolutionAttempt:
    """Tests for SolutionAttempt."""
    
    def test_attempt_creation(self):
        """Test creating a solution attempt."""
        attempt = SolutionAttempt(
            attempt_id="attempt-1",
            subproblem_id="sub-1",
            complexity_score=0.5,
            allocated_strategy="maker_full",
            n_agents=5,
            k_ahead=2,
            status=SolutionStatus.IN_PROGRESS,
            start_time=time.time(),
        )
        
        assert attempt.attempt_id == "attempt-1"
        assert attempt.status == SolutionStatus.IN_PROGRESS
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        start = time.time()
        time.sleep(0.01)  # 10ms
        
        attempt = SolutionAttempt(
            attempt_id="attempt-1",
            subproblem_id="sub-1",
            complexity_score=0.5,
            allocated_strategy="maker_full",
            n_agents=5,
            k_ahead=2,
            status=SolutionStatus.COMPLETED,
            start_time=start,
            end_time=time.time(),
        )
        
        assert attempt.duration_ms >= 10  # At least 10ms
    
    def test_duration_in_progress(self):
        """Test duration for in-progress attempt."""
        start = time.time() - 5  # 5 seconds ago
        
        attempt = SolutionAttempt(
            attempt_id="attempt-1",
            subproblem_id="sub-1",
            complexity_score=0.5,
            allocated_strategy="maker_full",
            n_agents=5,
            k_ahead=2,
            status=SolutionStatus.IN_PROGRESS,
            start_time=start,
        )
        
        # Should calculate duration from start time to now
        assert attempt.duration_ms >= 5000  # At least 5 seconds
    
    def test_metadata_default(self):
        """Test metadata defaults to empty dict."""
        attempt = SolutionAttempt(
            attempt_id="attempt-1",
            subproblem_id="sub-1",
            complexity_score=0.5,
            allocated_strategy="maker_full",
            n_agents=5,
            k_ahead=2,
            status=SolutionStatus.PENDING,
            start_time=time.time(),
        )
        
        assert attempt.metadata == {}
    
    def test_custom_metadata(self):
        """Test custom metadata."""
        attempt = SolutionAttempt(
            attempt_id="attempt-1",
            subproblem_id="sub-1",
            complexity_score=0.5,
            allocated_strategy="maker_full",
            n_agents=5,
            k_ahead=2,
            status=SolutionStatus.COMPLETED,
            start_time=time.time(),
            metadata={"model": "gpt-4", "tokens": 1000},
        )
        
        assert attempt.metadata["model"] == "gpt-4"
        assert attempt.metadata["tokens"] == 1000


class TestAdaptiveExecutionController:
    """Tests for AdaptiveExecutionController."""
    
    def test_controller_initialization(self):
        """Test controller can be initialized."""
        controller = AdaptiveExecutionController()
        
        assert controller.classifier is not None
        assert controller.allocator is not None
    
    def test_controller_with_custom_components(self):
        """Test controller with custom components."""
        classifier = Mock()
        allocator = Mock()
        
        controller = AdaptiveExecutionController(
            classifier=classifier,
            allocator=allocator,
        )
        
        assert controller.classifier == classifier
        assert controller.allocator == allocator
    
    def test_get_status(self):
        """Test getting controller status."""
        controller = AdaptiveExecutionController()
        
        status = controller.get_status()
        
        assert "initialized" in status
        assert status["initialized"] is True
    
    def test_solve_simple_subproblem(self):
        """Test solving a simple subproblem."""
        controller = AdaptiveExecutionController()
        
        subproblem = SubProblem(
            id="test-simple",
            description="Solve 2 + 2 = ?",
            domain="mathematics",
            depth=1,
            dependencies=[],
            metadata={},
        )
        
        # Mock the actual solving to avoid LLM calls
        with patch.object(controller, '_execute_solution') as mock_execute:
            mock_execute.return_value = {
                "solution": "4",
                "success": True,
                "duration_ms": 100.0,
            }
            
            result = controller.solve(subproblem)
            
            assert result is not None
            assert result["success"] is True
    
    def test_solve_complex_subproblem(self):
        """Test solving a complex subproblem."""
        controller = AdaptiveExecutionController()
        
        subproblem = SubProblem(
            id="test-complex",
            description="Implement a distributed consensus algorithm with Byzantine fault tolerance",
            domain="distributed_systems",
            depth=10,
            dependencies=["network", "crypto", "consensus"],
            metadata={},
        )
        
        with patch.object(controller, '_execute_solution') as mock_execute:
            mock_execute.return_value = {
                "solution": "Implemented algorithm",
                "success": True,
                "duration_ms": 5000.0,
            }
            
            result = controller.solve(subproblem)
            
            assert result is not None
            assert result["complexity_score"] > 0.5
    
    def test_execution_stats_tracking(self):
        """Test that execution stats are tracked."""
        controller = AdaptiveExecutionController()
        
        initial_stats = controller._execution_stats.copy()
        
        subproblem = SubProblem(
            id="test-stats",
            description="Simple problem",
            domain="mathematics",
            depth=1,
            dependencies=[],
            metadata={},
        )
        
        with patch.object(controller, '_execute_solution') as mock_execute:
            mock_execute.return_value = {
                "solution": "Solution",
                "success": True,
                "duration_ms": 100.0,
            }
            
            controller.solve(subproblem)
        
        # Stats should have been updated
        updated_stats = controller._execution_stats
        assert updated_stats.get("total_solves", 0) >= 1
    
    def test_failed_execution(self):
        """Test handling of failed execution."""
        controller = AdaptiveExecutionController()
        
        subproblem = SubProblem(
            id="test-fail",
            description="Problem that will fail",
            domain="mathematics",
            depth=1,
            dependencies=[],
            metadata={},
        )
        
        with patch.object(controller, '_execute_solution') as mock_execute:
            mock_execute.return_value = {
                "solution": None,
                "success": False,
                "error": "Test error",
                "duration_ms": 50.0,
            }
            
            result = controller.solve(subproblem)
            
            assert result["success"] is False
            assert result["error"] == "Test error"
    
    def test_retry_on_failure(self):
        """Test retry logic on failure."""
        controller = AdaptiveExecutionController()
        
        subproblem = SubProblem(
            id="test-retry",
            description="Problem requiring retry",
            domain="mathematics",
            depth=1,
            dependencies=[],
            metadata={},
        )
        
        call_count = 0
        
        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return {"success": False, "error": "Temporary failure"}
            return {"success": True, "solution": "Success", "duration_ms": 100.0}
        
        with patch.object(controller, '_execute_solution', side_effect=mock_execute):
            result = controller.solve(subproblem)
            
            assert result["success"] is True
            # Should have been called twice
            assert call_count == 2
    
    def test_subproblem_with_dependencies(self):
        """Test solving subproblem with dependencies."""
        controller = AdaptiveExecutionController()
        
        subproblem = SubProblem(
            id="test-deps",
            description="Problem with dependencies",
            domain="mathematics",
            depth=3,
            dependencies=["dep1", "dep2", "dep3"],
            metadata={},
        )
        
        with patch.object(controller, '_execute_solution') as mock_execute:
            mock_execute.return_value = {
                "solution": "Solution",
                "success": True,
                "duration_ms": 200.0,
            }
            
            result = controller.solve(subproblem)
            
            assert result is not None
            # Should have higher complexity due to dependencies
            assert result.get("complexity_score", 0) > 0


class TestControllerIntegration:
    """Integration tests for controller."""
    
    def test_full_solve_flow(self):
        """Test complete solve flow."""
        controller = AdaptiveExecutionController()
        
        subproblem = SubProblem(
            id="test-flow",
            description="Prove that the sum of two even numbers is even",
            domain="mathematics",
            depth=2,
            dependencies=[],
            metadata={},
        )
        
        with patch.object(controller, '_execute_solution') as mock_execute:
            mock_execute.return_value = {
                "solution": "Proof completed",
                "success": True,
                "duration_ms": 150.0,
                "metadata": {"model": "gpt-4o-mini"},
            }
            
            result = controller.solve(subproblem)
            
            # Verify result structure
            assert "subproblem_id" in result
            assert "complexity_score" in result
            assert "strategy_used" in result
            assert "success" in result
            assert "duration_ms" in result
    
    def test_classifier_and_allocator_integration(self):
        """Test that classifier and allocator work together."""
        controller = AdaptiveExecutionController()
        
        subproblem = SubProblem(
            id="test-integration",
            description="Implement quicksort algorithm",
            domain="computer_science",
            depth=5,
            dependencies=["sorting"],
            metadata={},
        )
        
        # Classify
        complexity = controller.classifier.classify(subproblem)
        
        # Allocate based on complexity
        decision = controller.allocator.allocate(complexity)
        
        # Verify
        assert complexity.overall_score > 0
        assert decision.complexity_score == complexity.overall_score
        assert decision.config is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
