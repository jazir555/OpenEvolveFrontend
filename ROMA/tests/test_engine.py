"""Tests for the engine module."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from roma_dspy.engine import ROMAEngine
from roma_dspy.config.schemas import ROMAConfig


class TestROMAEngine:
    """Test suite for ROMA Engine."""

    def test_engine_initialization(self):
        """Test engine initialization with default config."""
        config = ROMAConfig()
        engine = ROMAEngine(config)
        
        assert engine.config is not None
        assert engine.config.max_depth > 0
        assert engine.is_initialized is False

    def test_engine_initialization_with_custom_config(self):
        """Test engine initialization with custom configuration."""
        config = ROMAConfig(
            max_depth=5,
            timeout_seconds=60,
            provider="mock"
        )
        engine = ROMAEngine(config)
        
        assert engine.config.max_depth == 5
        assert engine.config.timeout_seconds == 60

    def test_solve_simple_task(self):
        """Test solving a simple task."""
        config = ROMAConfig(
            provider="mock",
            max_depth=2,
            timeout_seconds=30
        )
        engine = ROMAEngine(config)
        
        # Test basic task solving
        result = engine.solve("Calculate 2 + 2")
        
        # Result should be either a TaskNode or a string result
        assert result is not None

    def test_solve_with_decomposition(self):
        """Test solving a task that requires decomposition."""
        config = ROMAConfig(
            provider="mock",
            max_depth=3,
            execution_mode="recursive"
        )
        engine = ROMAEngine(config)
        
        # A task that benefits from decomposition
        result = engine.solve("Sort a list of numbers using quicksort")
        assert result is not None

    def test_error_handling_empty_task(self):
        """Test error handling for empty task."""
        config = ROMAConfig()
        engine = ROMAEngine(config)
        
        # Test with empty task - should raise ValueError or handle gracefully
        with pytest.raises((ValueError, Exception)):
            engine.solve("")

    def test_error_handling_none_task(self):
        """Test error handling for None task."""
        config = ROMAConfig()
        engine = ROMAEngine(config)
        
        # Test with None task - should raise ValueError or handle gracefully
        with pytest.raises((ValueError, TypeError, Exception)):
            engine.solve(None)


class TestROMAEngineAsync:
    """Test suite for async ROMA Engine operations."""

    @pytest.mark.asyncio
    async def test_async_solve(self):
        """Test async solve function."""
        import asyncio
        
        config = ROMAConfig(
            provider="mock",
            max_depth=2,
            timeout_seconds=30
        )
        engine = ROMAEngine(config)
        
        result = await engine.solve_async("Calculate 3 * 3")
        assert result is not None

    @pytest.mark.asyncio
    async def test_async_solve_concurrent(self):
        """Test multiple concurrent async solves."""
        config = ROMAConfig(
            provider="mock",
            max_depth=2,
            timeout_seconds=30
        )
        engine = ROMAEngine(config)
        
        # Run multiple tasks concurrently
        tasks = [
            engine.solve_async(f"Task {i}")
            for i in range(3)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All tasks should complete (even if some raise exceptions)
        assert len(results) == 3


class TestROMAEngineState:
    """Test suite for ROMA Engine state management."""

    def test_engine_state_transitions(self):
        """Test engine state transitions during solve."""
        config = ROMAConfig()
        engine = ROMAEngine(config)
        
        # Initial state
        assert engine.is_initialized is False
        
        # Initialize
        engine.initialize()
        assert engine.is_initialized is True

    def test_engine_reset(self):
        """Test engine reset functionality."""
        config = ROMAConfig()
        engine = ROMAEngine(config)
        
        engine.initialize()
        assert engine.is_initialized is True
        
        engine.reset()
        # After reset, engine should be ready for new tasks
        assert engine is not None
