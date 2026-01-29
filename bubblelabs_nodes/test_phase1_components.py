"""
Unit Tests for Phase 1 Gauntlet Components

Comprehensive test suite for:
- Parallel executor
- Solution cache
- Checkpoint manager
- Visualization components
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Import components to test
from bubblelabs_nodes.parallel_executor import (
    ParallelProblemExecutor,
    ProblemDependencyAnalyzer,
    ExecutionResult,
    ParallelExecutionSummary,
)

from bubblelabs_nodes.solution_cache import (
    AtomicSolutionCache,
    ProblemHasher,
    InMemoryCache,
    create_solution_cache,
)

from bubblelabs_nodes.checkpoint_manager import (
    CheckpointManager,
    CheckpointRepository,
    StateSerializer,
    PipelineState,
    CheckpointMetadata,
    create_checkpoint_manager,
)

from bubblelabs_nodes.visualization import (
    ProblemTreeBuilder,
    ProblemNode,
    ASCIITreeRenderer,
    visualize_problem,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def sample_problems():
    """Sample problems for testing"""
    return [
        {
            'id': 'problem_1',
            'statement': 'Solve problem 1',
            'requirements': ['fast', 'reliable'],
        },
        {
            'id': 'problem_2',
            'statement': 'Solve problem 2',
            'requirements': ['secure'],
        },
        {
            'id': 'problem_3',
            'statement': 'Solve problem 3',
            'requires': ['problem_1'],  # Depends on problem_1
        },
    ]


@pytest.fixture
def sample_problem_hierarchy():
    """Sample hierarchical problem"""
    return {
        'id': 'root_problem',
        'statement': 'Build a system',
        'status': 'complete',
        'score': 85,
        'subproblems': [
            {
                'id': 'subproblem_1',
                'statement': 'Design database',
                'status': 'complete',
                'score': 90,
                'subproblems': [
                    {
                        'id': 'atomic_1',
                        'statement': 'Choose DB engine',
                        'status': 'complete',
                        'score': 95,
                    }
                ]
            },
            {
                'id': 'subproblem_2',
                'statement': 'Build API',
                'status': 'in_progress',
                'score': 75,
            }
        ]
    }


# =============================================================================
# Parallel Executor Tests
# =============================================================================

class TestProblemDependencyAnalyzer:
    """Tests for ProblemDependencyAnalyzer"""

    def test_find_independent_problems(self, sample_problems):
        """Test finding independent problems"""
        analyzer = ProblemDependencyAnalyzer()
        independent, dependencies = analyzer.find_independent_problems(sample_problems)

        # problems 1 and 2 are independent, problem 3 depends on problem 1
        assert len(independent) == 2
        assert len(dependencies) == 1
        assert dependencies[0].parent_id == 'problem_1'
        assert dependencies[0].child_id == 'problem_3'

    def test_build_dependency_graph(self, sample_problems):
        """Test building dependency graph"""
        analyzer = ProblemDependencyAnalyzer()
        graph = analyzer.build_dependency_graph(sample_problems)

        assert 'problem_1' in graph
        assert 'problem_2' in graph
        assert 'problem_3' in graph
        assert 'problem_1' in graph['problem_3']  # problem_3 depends on problem_1

    def test_topological_sort(self, sample_problems):
        """Test topological sorting"""
        analyzer = ProblemDependencyAnalyzer()
        graph = analyzer.build_dependency_graph(sample_problems)
        sorted_problems = analyzer.topological_sort(sample_problems, graph)

        # problem_1 should come before problem_3
        ids = [p['id'] for p in sorted_problems]
        assert ids.index('problem_1') < ids.index('problem_3')

    def test_circular_dependency_detection(self):
        """Test circular dependency detection"""
        analyzer = ProblemDependencyAnalyzer()

        # Create circular dependency: 1 -> 2 -> 3 -> 1
        circular_problems = [
            {'id': '1', 'requires': ['3']},
            {'id': '2', 'requires': ['1']},
            {'id': '3', 'requires': ['2']},
        ]

        graph = analyzer.build_dependency_graph(circular_problems)

        with pytest.raises(ValueError, match="Circular dependency"):
            analyzer.topological_sort(circular_problems, graph)


class TestParallelProblemExecutor:
    """Tests for ParallelProblemExecutor"""

    @pytest.mark.asyncio
    async def test_execute_independent_problems(self, sample_problems):
        """Test executing independent problems in parallel"""
        executor = ParallelProblemExecutor(config={'max_concurrency': 2})

        async def mock_solve(problem, context):
            await asyncio.sleep(0.1)
            return f"Solution for {problem['id']}"

        summary = await executor.execute_in_parallel(
            sample_problems[:2],  # Only independent problems
            mock_solve,
            {}
        )

        assert summary.total_problems == 2
        assert summary.successful == 2
        assert summary.failed == 0
        assert summary.parallel_speedup > 1.0

    @pytest.mark.asyncio
    async def test_execute_with_dependencies(self, sample_problems):
        """Test executing problems with dependencies"""
        executor = ParallelProblemExecutor()

        async def mock_solve(problem, context):
            await asyncio.sleep(0.1)
            return f"Solution for {problem['id']}"

        summary = await executor.execute_in_parallel(
            sample_problems,
            mock_solve,
            {}
        )

        assert summary.total_problems == 3
        assert summary.successful == 3
        # Should respect dependencies (problem_3 waits for problem_1)

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling"""
        executor = ParallelProblemExecutor(config={'timeout': 0.1})

        async def slow_solve(problem, context):
            await asyncio.sleep(1)  # Slower than timeout
            return "Solution"

        summary = await executor.execute_in_parallel(
            [{'id': '1', 'statement': 'Slow problem'}],
            slow_solve,
            {}
        )

        assert summary.failed == 1
        assert 'timed out' in summary.errors[0].lower()


# =============================================================================
# Solution Cache Tests
# =============================================================================

class TestProblemHasher:
    """Tests for ProblemHasher"""

    def test_generate_hash(self):
        """Test hash generation"""
        hasher = ProblemHasher()
        problem = {
            'id': 'test_problem',
            'statement': 'Solve this',
            'requirements': ['fast'],
        }

        hash1 = hasher.generate_hash(problem)
        hash2 = hasher.generate_hash(problem)

        assert hash1 == hash2  # Same problem = same hash
        assert hash1.startswith('problem:')

    def test_normalize_problem(self):
        """Test problem normalization"""
        hasher = ProblemHasher()
        problem1 = {
            'id': 'different_id',  # Should be ignored
            'statement': 'Solve this',
            'timestamp': '2024-01-01',  # Should be ignored
            'requirements': ['fast'],
        }

        problem2 = {
            'id': 'another_id',  # Different ID
            'statement': 'Solve this',
            'requirements': ['fast'],
        }

        hash1 = hasher.generate_hash(problem1)
        hash2 = hasher.generate_hash(problem2)

        assert hash1 == hash2  # Should match after normalization


class TestInMemoryCache:
    """Tests for InMemoryCache"""

    @pytest.mark.asyncio
    async def test_cache_hit_miss(self):
        """Test cache hit and miss"""
        cache = InMemoryCache(max_size=10, ttl=60)

        # Miss
        result = await cache.get('nonexistent')
        assert result is None

        # Set and hit
        await cache.set('key', b'value', ttl=60)
        result = await cache.get('key')
        assert result == b'value'

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL expiration"""
        cache = InMemoryCache(max_size=10, ttl=1)

        await cache.set('key', b'value', ttl=1)
        assert await cache.get('key') == b'value'

        # Wait for expiration
        await asyncio.sleep(1.5)
        assert await cache.get('key') is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction"""
        cache = InMemoryCache(max_size=2, ttl=60)

        await cache.set('key1', b'value1', ttl=60)
        await cache.set('key2', b'value2', ttl=60)
        await cache.set('key3', b'value3', ttl=60)  # Should evict key1

        assert await cache.get('key1') is None
        assert await cache.get('key2') == b'value2'
        assert await cache.get('key3') == b'value3'


class TestAtomicSolutionCache:
    """Tests for AtomicSolutionCache"""

    @pytest.mark.asyncio
    async def test_cache_solve(self):
        """Test solving with cache"""
        cache = create_solution_cache(config={'max_size': 100})

        call_count = 0

        async def solve_func(problem):
            nonlocal call_count
            call_count += 1
            return f"Solution {call_count}"

        problem = {'id': 'test', 'statement': 'Solve'}

        # First call - cache miss
        result1 = await cache.solve(problem, solve_func)
        assert call_count == 1

        # Second call - cache hit
        result2 = await cache.solve(problem, solve_func)
        assert call_count == 1  # Should not increment
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_cache_statistics(self):
        """Test cache statistics"""
        cache = create_solution_cache()

        async def solve_func(problem):
            return "solution"

        problem = {'id': 'test', 'statement': 'Solve'}

        # Cache miss
        await cache.solve(problem, solve_func)
        # Cache hit
        await cache.solve(problem, solve_func)

        stats = cache.get_statistics()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5


# =============================================================================
# Checkpoint Manager Tests
# =============================================================================

class TestStateSerializer:
    """Tests for StateSerializer"""

    @pytest.mark.asyncio
    async def test_serialize_deserialize(self):
        """Test serialization and deserialization"""
        serializer = StateSerializer()

        state = PipelineState(
            problem={'id': 'test', 'statement': 'Solve'},
            context={'key': 'value'},
            solutions={'solution': 'answer'},
            metrics={'time': 1.0},
        )

        # Serialize
        data = await serializer.serialize(state)
        assert len(data) > 0

        # Deserialize
        restored = await serializer.deserialize(data)
        assert restored.problem == state.problem
        assert restored.context == state.context
        assert restored.solutions == state.solutions

    @pytest.mark.asyncio
    async def test_sanitize_context(self):
        """Test context sanitization"""
        serializer = StateSerializer()

        context = {
            'valid_data': 'value',
            'function': lambda x: x,  # Should be removed
            'class': object,  # Should be removed
        }

        sanitized = serializer._sanitize_context(context)
        assert 'valid_data' in sanitized
        assert 'function' not in sanitized
        assert 'class' not in sanitized


class TestCheckpointRepository:
    """Tests for CheckpointRepository"""

    @pytest.mark.asyncio
    async def test_save_load_file(self, temp_dir):
        """Test saving and loading checkpoints to file"""
        repo = CheckpointRepository(storage_type='file', storage_path=temp_dir)

        metadata = CheckpointMetadata(
            checkpoint_id='test_checkpoint',
            problem_id='test_problem',
            timestamp=datetime.utcnow(),
            level=0,
            stage='test',
        )

        await repo.save('test_checkpoint', b'test_data', metadata)
        result = await repo.load('test_checkpoint')

        assert result is not None
        data, loaded_metadata = result
        assert data == b'test_data'
        assert loaded_metadata.checkpoint_id == 'test_checkpoint'

    @pytest.mark.asyncio
    async def test_save_load_memory(self):
        """Test saving and loading checkpoints in memory"""
        repo = CheckpointRepository(storage_type='memory')

        metadata = CheckpointMetadata(
            checkpoint_id='test_checkpoint',
            problem_id='test_problem',
            timestamp=datetime.utcnow(),
            level=0,
            stage='test',
        )

        await repo.save('test_checkpoint', b'test_data', metadata)
        result = await repo.load('test_checkpoint')

        assert result is not None
        data, _ = result
        assert data == b'test_data'

    @pytest.mark.asyncio
    async def test_delete(self, temp_dir):
        """Test checkpoint deletion"""
        repo = CheckpointRepository(storage_type='file', storage_path=temp_dir)

        metadata = CheckpointMetadata(
            checkpoint_id='test_checkpoint',
            problem_id='test_problem',
            timestamp=datetime.utcnow(),
            level=0,
            stage='test',
        )

        await repo.save('test_checkpoint', b'test_data', metadata)
        assert await repo.load('test_checkpoint') is not None

        await repo.delete('test_checkpoint')
        assert await repo.load('test_checkpoint') is None


class TestCheckpointManager:
    """Tests for CheckpointManager"""

    @pytest.mark.asyncio
    async def test_create_checkpoint(self, temp_dir):
        """Test checkpoint creation"""
        repo = CheckpointRepository(storage_type='file', storage_path=temp_dir)
        manager = CheckpointManager(repository=repo)

        checkpoint_id = await manager.create_checkpoint(
            problem={'id': 'test', 'statement': 'Solve'},
            context={'stage': 'test'},
            level=0,
            stage='initial'
        )

        assert checkpoint_id is not None
        assert 'test' in checkpoint_id

    @pytest.mark.asyncio
    async def test_load_checkpoint(self, temp_dir):
        """Test checkpoint loading"""
        repo = CheckpointRepository(storage_type='file', storage_path=temp_dir)
        manager = CheckpointManager(repository=repo)

        # Create checkpoint
        checkpoint_id = await manager.create_checkpoint(
            problem={'id': 'test', 'statement': 'Solve'},
            context={'stage': 'test'},
            level=0,
            stage='initial'
        )

        # Load checkpoint
        state = await manager.load_checkpoint(checkpoint_id)
        assert state is not None
        assert state.problem['id'] == 'test'

    @pytest.mark.asyncio
    async def test_cleanup_checkpoints(self, temp_dir):
        """Test checkpoint cleanup"""
        repo = CheckpointRepository(storage_type='file', storage_path=temp_dir)
        manager = CheckpointManager(repository=repo, auto_cleanup=False)

        problem_id = 'test_problem'

        # Create multiple checkpoints
        for i in range(5):
            await manager.create_checkpoint(
                problem={'id': problem_id, 'statement': f'Solve {i}'},
                context={'stage': 'test'},
                level=0,
                stage=f'checkpoint_{i}'
            )

        # Cleanup, keeping only 2
        deleted = await manager.cleanup_checkpoints(problem_id, keep_last_n=2)
        assert deleted == 3

        # Verify only 2 remain
        checkpoints = await manager.list_checkpoints(problem_id)
        assert len(checkpoints) == 2


# =============================================================================
# Visualization Tests
# =============================================================================

class TestProblemTreeBuilder:
    """Tests for ProblemTreeBuilder"""

    def test_build_tree(self, sample_problem_hierarchy):
        """Test building a problem tree"""
        builder = ProblemTreeBuilder()
        tree = builder.build_tree(sample_problem_hierarchy)

        assert isinstance(tree, ProblemNode)
        assert tree.problem_id == 'root_problem'
        assert len(tree.children) == 2
        assert tree.children[0].problem_id == 'subproblem_1'
        assert tree.children[0].children[0].problem_id == 'atomic_1'

    def test_metadata_preservation(self, sample_problem_hierarchy):
        """Test that metadata is preserved"""
        builder = ProblemTreeBuilder()
        tree = builder.build_tree(sample_problem_hierarchy)

        assert tree.status == 'complete'
        assert tree.score == 85
        assert tree.children[0].score == 90


class TestASCIITreeRenderer:
    """Tests for ASCIITreeRenderer"""

    def test_render_simple_tree(self, sample_problem_hierarchy):
        """Test rendering a simple tree"""
        builder = ProblemTreeBuilder()
        tree = builder.build_tree(sample_problem_hierarchy)

        renderer = ASCIITreeRenderer()
        output = renderer.render(tree)

        assert isinstance(output, str)
        assert len(output) > 0
        assert 'Build a system' in output
        assert '└' in output or '├' in output  # Box characters

    def test_status_icons(self):
        """Test status icon rendering"""
        renderer = ASCIITreeRenderer()

        assert renderer._get_status_icon('complete') == '✅'
        assert renderer._get_status_icon('in_progress') == '🔄'
        assert renderer._get_status_icon('failed') == '❌'


class TestVisualizeProblem:
    """Tests for visualize_problem function"""

    def test_visualize_ascii(self, sample_problem_hierarchy):
        """Test ASCII visualization"""
        output = visualize_problem(sample_problem_hierarchy, format='ascii')
        assert isinstance(output, str)
        assert len(output) > 0

    def test_visualize_html(self, sample_problem_hierarchy):
        """Test HTML visualization"""
        output = visualize_problem(sample_problem_hierarchy, format='html')
        assert isinstance(output, str)
        assert '<html>' in output
        assert '<body>' in output

    def test_visualize_dot(self, sample_problem_hierarchy):
        """Test DOT visualization"""
        output = visualize_problem(sample_problem_hierarchy, format='dot')
        assert isinstance(output, str)
        assert 'digraph' in output


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase1Integration:
    """Integration tests for Phase 1 components working together"""

    @pytest.mark.asyncio
    async def test_parallel_with_cache(self, sample_problems):
        """Test parallel execution with caching"""
        executor = ParallelProblemExecutor()
        cache = create_solution_cache()

        call_count = 0

        async def solve_with_cache(problem, context):
            nonlocal call_count
            return await cache.solve(problem, lambda p: f"Solution {call_count + 1}")

        summary = await executor.execute_in_parallel(
            sample_problems[:2],
            solve_with_cache,
            {}
        )

        assert summary.successful == 2

        # Run again - should use cache
        summary2 = await executor.execute_in_parallel(
            sample_problems[:2],
            solve_with_cache,
            {}
        )

        assert summary2.successful == 2
        stats = cache.get_statistics()
        assert stats['hits'] > 0

    @pytest.mark.asyncio
    async def test_checkpointed_execution(self, temp_dir):
        """Test execution with checkpointing"""
        repo = CheckpointRepository(storage_type='file', storage_path=temp_dir)
        manager = CheckpointManager(repository=repo)

        # Create initial checkpoint
        checkpoint_id = await manager.create_checkpoint(
            problem={'id': 'test', 'statement': 'Solve'},
            context={'stage': 'initial'},
            level=0,
            stage='initial'
        )

        # Load and continue
        state = await manager.load_checkpoint(checkpoint_id)
        assert state is not None

        # Create completion checkpoint
        completion_id = await manager.create_checkpoint(
            problem=state.problem,
            context={'stage': 'complete'},
            solutions={'solution': 'answer'},
            level=0,
            stage='complete'
        )

        assert completion_id is not None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
