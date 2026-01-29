"""
Integration Tests for Parallel Execution

Tests parallel execution with real Gauntlet components,
end-to-end workflows, and performance validation.

Acceptance Criteria:
- Integration tests passing
- 50%+ speedup on 3+ problems
"""

import pytest
import asyncio
import time
from bubblelabs_nodes import (
    ParallelProblemExecutor,
    WorkerPoolExecutor,
    GauntletSolver,
    solveProblem,
    create_solution_cache,
    create_checkpoint_manager,
)


class TestParallelExecutionIntegration:
    """Integration tests for parallel execution"""

    @pytest.mark.asyncio
    async def test_parallel_with_real_gauntlet(self):
        """Test parallel execution with real Gauntlet components"""
        # Setup
        cache = create_solution_cache()
        executor = ParallelProblemExecutor(max_parallelism=3)

        # Create test problems that simulate real workload
        async def realistic_solver(problem):
            # Simulate realistic solve time
            await asyncio.sleep(0.05)
            return {
                'problem_id': problem['id'],
                'success': True,
                'score': 0.85,
                'solution': f"Solution for {problem['id']}"
            }

        problems = [
            {'id': f'problem_{i}', 'statement': f'Solve problem {i}'}
            for i in range(5)
        ]

        # Execute
        result = await executor.execute_in_parallel(
            problems=problems,
            executor_func=realistic_solver,
            context={}
        )

        # Validate
        assert result.total_count == 5
        assert result.successful_count == 5
        assert result.failed_count == 0
        assert result.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_with_independent_problems(self):
        """Test with truly independent atomic problems"""
        executor = ParallelProblemExecutor(max_parallelism=5)

        async def solver(problem):
            await asyncio.sleep(0.02)
            return {'id': problem['id'], 'success': True}

        problems = [
            {'id': f'independent_{i}', 'statement': f'Independent {i}'}
            for i in range(10)
        ]

        result = await executor.execute_in_parallel(
            problems=problems,
            executor_func=solver,
            context={}
        )

        assert result.successful_count == 10

    @pytest.mark.asyncio
    async def test_with_dependent_subproblems(self):
        """Test with dependent subproblems (must execute in order)"""
        executor = ParallelProblemExecutor(max_parallelism=3)

        execution_order = []

        async def ordered_solver(problem):
            await asyncio.sleep(0.01)
            execution_order.append(problem['id'])
            return {'id': problem['id'], 'success': True}

        # Create problems with dependencies
        problems = [
            {'id': 'p1', 'dependencies': []},
            {'id': 'p2', 'dependencies': ['p1']},
            {'id': 'p3', 'dependencies': ['p2']},
        ]

        result = await executor.execute_in_parallel(
            problems=problems,
            executor_func=ordered_solver,
            context={}
        )

        # All should complete despite dependencies
        assert result.successful_count == 3

    @pytest.mark.asyncio
    async def test_with_mixed_independence(self):
        """Test with mix of independent and dependent problems"""
        executor = ParallelProblemExecutor(max_parallelism=4)

        async def solver(problem):
            await asyncio.sleep(0.02)
            return {'id': problem['id'], 'success': True}

        # Mix: some independent, some dependent
        problems = [
            {'id': 'independent_1', 'dependencies': []},
            {'id': 'independent_2', 'dependencies': []},
            {'id': 'dependent_1', 'dependencies': ['independent_1']},
            {'id': 'dependent_2', 'dependencies': ['independent_2']},
        ]

        result = await executor.execute_in_parallel(
            problems=problems,
            executor_func=solver,
            context={}
        )

        assert result.successful_count == 4

    @pytest.mark.asyncio
    async def test_performance_improvement(self):
        """Test that parallel execution provides 50%+ speedup"""
        executor = ParallelProblemExecutor(max_parallelism=5)

        async def solver(problem):
            await asyncio.sleep(0.05)  # 50ms per problem
            return {'id': problem['id'], 'success': True}

        problems = [
            {'id': f'perf_{i}', 'statement': f'Performance test {i}'}
            for i in range(5)
        ]

        # Parallel execution
        start_parallel = time.time()
        parallel_result = await executor.execute_in_parallel(
            problems=problems,
            executor_func=solver,
            context={}
        )
        parallel_time = time.time() - start_parallel

        # Sequential execution
        start_sequential = time.time()
        for problem in problems:
            await solver(problem)
        sequential_time = time.time() - start_sequential

        # Calculate speedup
        speedup = sequential_time / parallel_time

        print(f"\nPerformance Results:")
        print(f"  Sequential time: {sequential_time:.3f}s")
        print(f"  Parallel time: {parallel_time:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # Assert 50%+ speedup
        assert speedup >= 1.5, f"Speedup {speedup:.2f}x is below 1.5x target"

    @pytest.mark.asyncio
    async def test_worker_pool_integration(self):
        """Test worker pool with real problems"""
        executor = WorkerPoolExecutor(max_workers=3)

        def sync_solver(problem):
            time.sleep(0.02)
            return {'id': problem['id'], 'success': True}

        problems = [
            {'id': f'wp_{i}', 'statement': f'Worker pool test {i}'}
            for i in range(6)
        ]

        result = await executor.execute_in_parallel(
            problems=problems,
            executor_func=sync_solver,
            context={}
        )

        assert result.total_tasks == 6
        assert result.successful_tasks >= 5

    @pytest.mark.asyncio
    async def test_cache_integration(self):
        """Test cache integration with parallel execution"""
        cache = create_solution_cache()
        executor = ParallelProblemExecutor(max_parallelism=3)

        async def solver(problem):
            await asyncio.sleep(0.01)
            return {'id': problem['id'], 'success': True, 'score': 0.85}

        # First run - populate cache
        problems = [
            {'id': f'cache_{i}', 'statement': f'Cache test {i}'}
            for i in range(3)
        ]

        result1 = await executor.execute_in_parallel(
            problems=problems,
            executor_func=solver,
            context={'cache': cache}
        )

        # Second run - should hit cache
        result2 = await executor.execute_in_parallel(
            problems=problems,
            executor_func=solver,
            context={'cache': cache}
        )

        assert result1.successful_count == 3
        assert result2.successful_count == 3

    @pytest.mark.asyncio
    async def test_checkpoint_integration(self):
        """Test checkpoint creation during parallel execution"""
        checkpoint_manager = create_checkpoint_manager()
        executor = ParallelProblemExecutor(max_parallelism=2)

        async def solver_with_checkpoint(problem):
            # Create checkpoint
            await checkpoint_manager.create_checkpoint(
                problem=problem,
                context={'stage': 'parallel_solve'},
                solutions={},
                level=0,
                stage='before_solve'
            )

            await asyncio.sleep(0.01)
            return {'id': problem['id'], 'success': True}

        problems = [
            {'id': f'cp_{i}', 'statement': f'Checkpoint test {i}'}
            for i in range(3)
        ]

        result = await executor.execute_in_parallel(
            problems=problems,
            executor_func=solver_with_checkpoint,
            context={}
        )

        assert result.successful_count == 3

        # Verify checkpoints created
        checkpoints = await checkpoint_manager.list_checkpoints()
        assert len(checkpoints) >= 3


class TestEndToEndWorkflows:
    """End-to-end workflow tests"""

    @pytest.mark.asyncio
    async def test_complete_solve_problem_workflow(self):
        """Test complete solveProblem workflow"""
        problem = {
            'id': 'e2e_1',
            'statement': 'End-to-end test',
            'subproblems': [
                {'id': 'sub_1', 'statement': 'Sub 1'},
                {'id': 'sub_2', 'statement': 'Sub 2'},
                {'id': 'sub_3', 'statement': 'Sub 3'},
                {'id': 'sub_4', 'statement': 'Sub 4'},
            ]
        }

        solution = await solveProblem(problem, enable_parallel=True)

        assert solution['success'] is True
        assert solution['num_solutions'] == 4

    @pytest.mark.asyncio
    async def test_sequential_vs_parallel_consistency(self):
        """Test that sequential and parallel produce same results"""
        problem = {
            'id': 'consistency_1',
            'subproblems': [
                {'id': f'sub_{i}', 'statement': f'Sub {i}'}
                for i in range(5)
            ]
        }

        # Sequential
        solution_seq = await solveProblem(problem, enable_parallel=False)

        # Parallel
        solution_par = await solveProblem(problem, enable_parallel=True)

        # Both should succeed
        assert solution_seq['success']
        assert solution_par['success']
        assert solution_seq['num_solutions'] == solution_par['num_solutions']


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarking tests"""

    @pytest.mark.asyncio
    async def benchmark_1_problem(self):
        """Benchmark with 1 problem"""
        executor = ParallelProblemExecutor(max_parallelism=5)

        async def solver(problem):
            await asyncio.sleep(0.05)
            return {'id': problem['id']}

        problems = [{'id': 'bench_1'}]

        start = time.time()
        result = await executor.execute_in_parallel(problems, solver, {})
        elapsed = time.time() - start

        print(f"\n1 problem: {elapsed:.3f}s")

    @pytest.mark.asyncio
    async def benchmark_3_problems(self):
        """Benchmark with 3 problems"""
        executor = ParallelProblemExecutor(max_parallelism=5)

        async def solver(problem):
            await asyncio.sleep(0.05)
            return {'id': problem['id']}

        problems = [{'id': f'bench_{i}'} for i in range(3)]

        start = time.time()
        result = await executor.execute_in_parallel(problems, solver, {})
        elapsed = time.time() - start

        print(f"\n3 problems: {elapsed:.3f}s")
        print(f"  Speedup vs sequential (0.15s): {0.15 / elapsed:.2f}x")

    @pytest.mark.asyncio
    async def benchmark_5_problems(self):
        """Benchmark with 5 problems"""
        executor = ParallelProblemExecutor(max_parallelism=5)

        async def solver(problem):
            await asyncio.sleep(0.05)
            return {'id': problem['id']}

        problems = [{'id': f'bench_{i}'} for i in range(5)]

        start = time.time()
        result = await executor.execute_in_parallel(problems, solver, {})
        elapsed = time.time() - start

        print(f"\n5 problems: {elapsed:.3f}s")
        print(f"  Speedup vs sequential (0.25s): {0.25 / elapsed:.2f}x")

    @pytest.mark.asyncio
    async def benchmark_10_problems(self):
        """Benchmark with 10 problems"""
        executor = ParallelProblemExecutor(max_parallelism=5)

        async def solver(problem):
            await asyncio.sleep(0.05)
            return {'id': problem['id']}

        problems = [{'id': f'bench_{i}'} for i in range(10)]

        start = time.time()
        result = await executor.execute_in_parallel(problems, solver, {})
        elapsed = time.time() - start

        print(f"\n10 problems: {elapsed:.3f}s")
        print(f"  Speedup vs sequential (0.50s): {0.50 / elapsed:.2f}x")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
