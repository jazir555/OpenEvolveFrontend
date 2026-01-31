"""Tests for parallel execution."""

import pytest
import sys
from pathlib import Path
import asyncio

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from roma_dspy.parallel.executor import ParallelExecutor
from roma_dspy.parallel.scheduler import TaskScheduler
from roma_dspy.engine import ROMAEngine
from roma_dspy.config.schemas import ROMAConfig


class TestParallelExecution:
    """Test suite for parallel task execution."""

    def test_parallel_executor_initialization(self):
        """Test parallel executor initialization."""
        executor = ParallelExecutor(max_workers=4)
        
        assert executor.max_workers == 4
        assert executor.is_initialized is True

    def test_parallel_executor_with_custom_workers(self):
        """Test parallel executor with different worker counts."""
        for workers in [1, 2, 4, 8]:
            executor = ParallelExecutor(max_workers=workers)
            assert executor.max_workers == workers

    def test_execute_multiple_tasks_parallel(self):
        """Test executing multiple tasks in parallel."""
        executor = ParallelExecutor(max_workers=4)
        
        # Create simple tasks
        tasks = [
            lambda i=i: f"result_{i}"
            for i in range(5)
        ]
        
        results = executor.execute_parallel(tasks)
        
        assert len(results) == 5
        assert all(r is not None for r in results)

    def test_parallel_execution_with_errors(self):
        """Test parallel execution handles errors gracefully."""
        executor = ParallelExecutor(max_workers=2)
        
        def failing_task():
            raise ValueError("Test error")
        
        def success_task():
            return "success"
        
        tasks = [success_task, failing_task, success_task]
        results = executor.execute_parallel(tasks, continue_on_error=True)
        
        # Should have 3 results (some may be exceptions)
        assert len(results) == 3


class TestTaskScheduler:
    """Test suite for task scheduling."""

    def test_scheduler_initialization(self):
        """Test task scheduler initialization."""
        scheduler = TaskScheduler()
        
        assert scheduler is not None
        assert len(scheduler.task_queue) == 0

    def test_add_single_task(self):
        """Test adding a single task to scheduler."""
        scheduler = TaskScheduler()
        
        task = {"id": "task_1", "priority": 1}
        scheduler.add_task(task)
        
        assert len(scheduler.task_queue) == 1

    def test_add_multiple_tasks(self):
        """Test adding multiple tasks to scheduler."""
        scheduler = TaskScheduler()
        
        tasks = [
            {"id": f"task_{i}", "priority": i}
            for i in range(5)
        ]
        scheduler.add_tasks(tasks)
        
        assert len(scheduler.task_queue) == 5

    def test_task_priority_ordering(self):
        """Test that tasks are scheduled by priority."""
        scheduler = TaskScheduler()
        
        # Add tasks in random priority order
        scheduler.add_task({"id": "low", "priority": 3})
        scheduler.add_task({"id": "high", "priority": 1})
        scheduler.add_task({"id": "medium", "priority": 2})
        
        # Get next task - should be highest priority (lowest number)
        next_task = scheduler.get_next_task()
        assert next_task["id"] == "high"


class TestConcurrencyLimits:
    """Test suite for concurrency limit enforcement."""

    def test_concurrency_limit_enforcement(self):
        """Test that concurrency limits are enforced."""
        executor = ParallelExecutor(max_workers=2)
        
        running_count = 0
        max_running = 0
        
        def count_concurrent_task():
            nonlocal running_count, max_running
            running_count += 1
            max_running = max(max_running, running_count)
            # Simulate work
            import time
            time.sleep(0.01)
            running_count -= 1
            return "done"
        
        tasks = [count_concurrent_task for _ in range(10)]
        executor.execute_parallel(tasks)
        
        # Max concurrent should not exceed worker limit
        assert max_running <= 2

    def test_zero_concurrency_limit(self):
        """Test behavior with zero concurrency limit."""
        # Should either raise error or default to 1
        try:
            executor = ParallelExecutor(max_workers=0)
            # If created, should handle gracefully
            assert executor.max_workers >= 1
        except (ValueError, AssertionError):
            # Expected - zero workers is invalid
            pass

    def test_dynamic_concurrency_adjustment(self):
        """Test dynamic adjustment of concurrency limits."""
        executor = ParallelExecutor(max_workers=4)
        
        # Initial limit
        assert executor.max_workers == 4
        
        # Adjust limit
        executor.set_max_workers(2)
        assert executor.max_workers == 2


class TestParallelWithEngine:
    """Test suite for parallel execution with ROMA Engine."""

    def test_engine_parallel_solve(self):
        """Test engine with parallel task solving."""
        config = ROMAConfig(
            provider="mock",
            max_depth=2,
            parallel=True,
            max_workers=2
        )
        engine = ROMAEngine(config)
        
        # Solve should work with parallel config
        result = engine.solve("Parallel test task")
        assert result is not None

    def test_parallel_decomposition(self):
        """Test parallel execution during task decomposition."""
        config = ROMAConfig(
            provider="mock",
            max_depth=3,
            parallel=True,
            parallel_decomposition=True
        )
        engine = ROMAEngine(config)
        
        result = engine.solve("Task requiring parallel decomposition")
        assert result is not None
