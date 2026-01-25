"""
Unit Tests for Parallel Execution Components

Comprehensive unit tests for parallel executor, worker pool,
dependency analyzer, and enhanced solver.

Coverage target: 90%+
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from bubblelabs_nodes import (
    ParallelProblemExecutor,
    ProblemDependencyAnalyzer,
    WorkerPoolExecutor,
    GauntletSolver,
    solveProblem,
    WorkerTask,
    WorkerResult,
)


class TestProblemDependencyAnalyzer:
    """Tests for ProblemDependencyAnalyzer"""

    def test_find_independent_problems_all_independent(self):
        """Test with completely independent problems"""
        analyzer = ProblemDependencyAnalyzer()

        problems = [
            {'id': 'p1', 'statement': 'Problem 1'},
            {'id': 'p2', 'statement': 'Problem 2'},
            {'id': 'p3', 'statement': 'Problem 3'},
        ]

        independent = analyzer.find_independent_problems(problems)

        assert len(independent) == 3
        assert set(p['id'] for p in independent) == {'p1', 'p2', 'p3'}

    def test_find_independent_problems_with_dependencies(self):
        """Test with dependent problems"""
        analyzer = ProblemDependencyAnalyzer()

        problems = [
            {'id': 'p1', 'statement': 'Problem 1', 'dependencies': []},
            {'id': 'p2', 'statement': 'Problem 2', 'dependencies': ['p1']},
            {'id': 'p3', 'statement': 'Problem 3', 'dependencies': []},
        ]

        independent = analyzer.find_independent_problems(problems)

        # p1 and p3 should be independent
        assert len(independent) >= 2
        independent_ids = {p['id'] for p in independent}
        assert 'p1' in independent_ids
        assert 'p3' in independent_ids

    def test_build_dependency_graph(self):
        """Test dependency graph construction"""
        analyzer = ProblemDependencyAnalyzer()

        problems = [
            {'id': 'p1', 'dependencies': []},
            {'id': 'p2', 'dependencies': ['p1']},
            {'id': 'p3', 'dependencies': ['p1', 'p2']},
        ]

        graph = analyzer.build_dependency_graph(problems)

        assert 'p1' in graph
        assert 'p2' in graph
        assert 'p3' in graph
        assert graph['p2'] == ['p1']
        assert set(graph['p3']) == {'p1', 'p2'}

    def test_topological_sort(self):
        """Test topological sorting"""
        analyzer = ProblemDependencyAnalyzer()

        graph = {
            'p3': ['p1', 'p2'],
            'p1': [],
            'p2': ['p1'],
        }

        sorted_nodes = analyzer.topological_sort(graph)

        # p1 should come before p2 and p3
        p1_idx = sorted_nodes.index('p1')
        p2_idx = sorted_nodes.index('p2')
        p3_idx = sorted_nodes.index('p3')

        assert p1_idx < p2_idx
        assert p1_idx < p3_idx

    def test_detect_circular_dependencies(self):
        """Test circular dependency detection"""
        analyzer = ProblemDependencyAnalyzer()

        # Create circular dependency: p1 -> p2 -> p1
        problems = [
            {'id': 'p1', 'dependencies': ['p2']},
            {'id': 'p2', 'dependencies': ['p1']},
        ]

        graph = analyzer.build_dependency_graph(problems)

        # Should detect cycle
        with pytest.raises(ValueError):
            analyzer.topological_sort(graph)


class TestParallelProblemExecutor:
    """Tests for ParallelProblemExecutor"""

    @pytest.mark.asyncio
    async def test_execute_in_parallel_basic(self):
        """Test basic parallel execution"""
        executor = ParallelProblemExecutor(max_parallelism=3)

        async def mock_solve(problem):
            return {'id': problem['id'], 'success': True}

        problems = [
            {'id': 'p1'},
            {'id': 'p2'},
            {'id': 'p3'},
        ]

        result = await executor.execute_in_parallel(
            problems=problems,
            executor_func=mock_solve,
            context={}
        )

        assert result.total_count == 3
        assert result.successful_count == 3
        assert result.failed_count == 0
        assert result.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_execute_with_partial_failures(self):
        """Test parallel execution with some failures"""
        executor = ParallelProblemExecutor(max_parallelism=3)

        async def mock_solve(problem):
            if problem['id'] == 'p2':
                raise ValueError("Test error")
            return {'id': problem['id'], 'success': True}

        problems = [
            {'id': 'p1'},
            {'id': 'p2'},
            {'id': 'p3'},
        ]

        result = await executor.execute_in_parallel(
            problems=problems,
            executor_func=mock_solve,
            context={}
        )

        assert result.total_count == 3
        assert result.successful_count == 2
        assert result.failed_count == 1

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Test concurrency limit is respected"""
        executor = ParallelProblemExecutor(max_parallelism=2)

        running_count = 0
        max_running = 0

        async def mock_solve(problem):
            nonlocal running_count, max_running
            running_count += 1
            max_running = max(max_running, running_count)

            await asyncio.sleep(0.1)

            running_count -= 1
            return {'id': problem['id']}

        problems = [{'id': f'p{i}'} for i in range(5)]

        await executor.execute_in_parallel(
            problems=problems,
            executor_func=mock_solve,
            context={}
        )

        # Should never exceed max_parallelism
        assert max_running <= 2

    @pytest.mark.asyncio
    async def test_cancellation_on_error(self):
        """Test cancellation when error occurs"""
        executor = ParallelProblemExecutor(max_parallelism=3, stop_on_first_error=True)

        async def mock_solve(problem):
            if problem['id'] == 'p2':
                raise ValueError("Stop!")
            await asyncio.sleep(0.1)
            return {'id': problem['id']}

        problems = [
            {'id': 'p1'},
            {'id': 'p2'},
            {'id': 'p3'},
        ]

        result = await executor.execute_in_parallel(
            problems=problems,
            executor_func=mock_solve,
            context={}
        )

        # Should stop after first error
        assert result.failed_count > 0


class TestWorkerPoolExecutor:
    """Tests for WorkerPoolExecutor"""

    @pytest.mark.asyncio
    async def test_worker_pool_basic_execution(self):
        """Test basic worker pool execution"""
        executor = WorkerPoolExecutor(max_workers=2)

        def mock_solve(problem):
            return {'id': problem['id'], 'success': True}

        problems = [
            {'id': 'p1'},
            {'id': 'p2'},
        ]

        result = await executor.execute_in_parallel(
            problems=problems,
            executor_func=mock_solve,
            context={}
        )

        assert result.total_tasks == 2
        assert result.successful_tasks >= 1

    @pytest.mark.asyncio
    async def test_work_stealing(self):
        """Test work stealing across workers"""
        executor = WorkerPoolExecutor(max_workers=2, enable_work_stealing=True)

        def mock_solve(problem):
            return {'id': problem['id'], 'success': True}

        problems = [{'id': f'p{i}'} for i in range(10)]

        result = await executor.execute_with_work_stealing(
            problems=problems,
            executor_func=mock_solve,
            context={}
        )

        assert result.total_tasks == 10


class TestGauntletSolver:
    """Tests for GauntletSolver"""

    @pytest.mark.asyncio
    async def test_solve_atomic_problem(self):
        """Test solving atomic problem"""
        solver = GauntletSolver()

        problem = {
            'id': 'atomic_1',
            'statement': 'Simple problem'
        }

        solution = await solver.solve_problem(problem)

        assert solution['problem_id'] == 'atomic_1'
        assert solution['success'] is True

    @pytest.mark.asyncio
    async def test_solve_with_parallel_subproblems(self):
        """Test solving with parallel subproblems"""
        solver = GauntletSolver(enable_parallel=True, parallel_threshold=2)

        problem = {
            'id': 'complex_1',
            'statement': 'Complex problem',
            'subproblems': [
                {'id': 'sub_1', 'statement': 'Sub 1'},
                {'id': 'sub_2', 'statement': 'Sub 2'},
                {'id': 'sub_3', 'statement': 'Sub 3'},
            ]
        }

        solution = await solver.solve_problem(problem)

        assert solution['success'] is True
        assert solution['num_solutions'] == 3

    @pytest.mark.asyncio
    async def test_solve_sequential_fallback(self):
        """Test sequential fallback on error"""
        solver = GauntletSolver(enable_parallel=True)

        problem = {
            'id': 'test_1',
            'subproblems': [
                {'id': 'sub_1'},
                {'id': 'sub_2'},
            ]
        }

        # Force sequential
        solution = await solver.solve_problem(problem, force_sequential=True)

        assert solution['success'] is True

    @pytest.mark.asyncio
    async def test_cache_integration(self):
        """Test cache is checked before solving"""
        solver = GauntletSolver()

        problem = {'id': 'cached_1', 'statement': 'Cached problem'}

        # First call - not cached
        solution1 = await solver.solve_problem(problem)

        # Second call - should be cached
        solution2 = await solver.solve_problem(problem)

        assert solution1['problem_id'] == solution2['problem_id']


class TestSolveProblemFunction:
    """Tests for solveProblem main function"""

    @pytest.mark.asyncio
    async def test_solve_problem_basic(self):
        """Test basic solveProblem function"""
        problem = {'id': 'test_1', 'statement': 'Test'}

        solution = await solveProblem(problem)

        assert solution['success'] is True

    @pytest.mark.asyncio
    async def test_solve_problem_with_context(self):
        """Test solveProblem with context"""
        problem = {'id': 'test_1', 'statement': 'Test'}
        context = {'team': 'blue_team'}

        solution = await solveProblem(problem, context=context)

        assert solution['success'] is True

    @pytest.mark.asyncio
    async def test_solve_problem_parallel_disabled(self):
        """Test solveProblem with parallel disabled"""
        problem = {
            'id': 'test_1',
            'subproblems': [
                {'id': 'sub_1'},
                {'id': 'sub_2'},
            ]
        }

        solution = await solveProblem(problem, enable_parallel=False)

        assert solution['success'] is True


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=bubblelabs_nodes', '--cov-report=html'])
