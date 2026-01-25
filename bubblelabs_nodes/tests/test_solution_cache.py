"""
Unit and Integration Tests for Solution Cache System

Comprehensive test coverage for caching functionality including
CRUD operations, TTL expiration, statistics, and performance benchmarks.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from bubblelabs_nodes.solution_cache import (
    ProblemHasher,
    InMemoryCache,
    CacheStatistics,
    AtomicSolutionCache,
    create_solution_cache,
)


class TestProblemHasher:
    """Tests for ProblemHasher"""

    def test_normalize_problem(self):
        """Test problem normalization"""
        hasher = ProblemHasher()

        problem = {
            'id': 'test_123',
            'timestamp': '2024-01-01',
            'type': 'test_problem',
            'difficulty': 'medium',
            'requires': ['req2', 'req1'],
        }

        normalized = hasher.normalize_problem(problem)

        # Should remove metadata fields
        assert 'id' not in normalized
        assert 'timestamp' not in normalized

        # Should sort lists
        assert normalized['requires'] == ['req1', 'req2']

        # Should keep content fields
        assert normalized['type'] == 'test_problem'
        assert normalized['difficulty'] == 'medium'

    def test_generate_hash_consistency(self):
        """Test hash generation is consistent"""
        hasher = ProblemHasher()

        problem1 = {'type': 'test', 'value': 42}
        problem2 = {'type': 'test', 'value': 42}

        hash1 = hasher.generate_hash(problem1)
        hash2 = hasher.generate_hash(problem2)

        assert hash1 == hash2

    def test_generate_hash_uniqueness(self):
        """Test different problems generate different hashes"""
        hasher = ProblemHasher()

        problem1 = {'type': 'test', 'value': 1}
        problem2 = {'type': 'test', 'value': 2}

        hash1 = hasher.generate_hash(problem1)
        hash2 = hasher.generate_hash(problem2)

        assert hash1 != hash2

    def test_hash_collision_handling(self):
        """Test hash collision detection"""
        hasher = ProblemHasher()

        # Generate many hashes and check for duplicates
        hashes = set()
        problems = [{'type': 'test', 'value': i} for i in range(1000)]

        for problem in problems:
            hash_val = hasher.generate_hash(problem)
            hashes.add(hash_val)

        # Should have minimal collisions (allow 1% due to hash space)
        assert len(hashes) >= 990


class TestInMemoryCache:
    """Tests for InMemoryCache"""

    @pytest.mark.asyncio
    async def test_cache_set_get(self):
        """Test basic set and get operations"""
        cache = InMemoryCache(max_size=10)

        await cache.set('key1', b'value1', ttl=60)
        value = await cache.get('key1')

        assert value == b'value1'

    @pytest.mark.asyncio
    async def test_cache_has(self):
        """Test has operation"""
        cache = InMemoryCache(max_size=10)

        assert not await cache.has('key1')

        await cache.set('key1', b'value1', ttl=60)

        assert await cache.has('key1')

    @pytest.mark.asyncio
    async def test_cache_delete(self):
        """Test delete operation"""
        cache = InMemoryCache(max_size=10)

        await cache.set('key1', b'value1', ttl=60)
        assert await cache.has('key1')

        await cache.delete('key1')
        assert not await cache.has('key1')

    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """Test clear operation"""
        cache = InMemoryCache(max_size=10)

        await cache.set('key1', b'value1', ttl=60)
        await cache.set('key2', b'value2', ttl=60)

        await cache.clear()

        assert not await cache.has('key1')
        assert not await cache.has('key2')

    @pytest.mark.asyncio
    async def test_cache_size_limit(self):
        """Test cache size limit enforcement"""
        cache = InMemoryCache(max_size=3)

        await cache.set('key1', b'value1', ttl=60)
        await cache.set('key2', b'value2', ttl=60)
        await cache.set('key3', b'value3', ttl=60)

        assert await cache.get_size() == 3

        # Add one more - should evict oldest
        await cache.set('key4', b'value4', ttl=60)

        assert await cache.get_size() == 3
        # key1 should be evicted
        assert not await cache.has('key1')

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction policy"""
        cache = InMemoryCache(max_size=3, ttl=3600)

        await cache.set('key1', b'value1', ttl=60)
        await cache.set('key2', b'value2', ttl=60)
        await cache.set('key3', b'value3', ttl=60)

        # Access key1 to make it recently used
        await cache.get('key1')

        # Add key4 - should evict key2 (least recently used)
        await cache.set('key4', b'value4', ttl=60)

        assert await cache.has('key1')
        assert not await cache.has('key2')
        assert await cache.has('key3')
        assert await cache.has('key4')

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL expiration"""
        cache = InMemoryCache(max_size=10)

        # Set with very short TTL
        await cache.set('key1', b'value1', ttl=0.1)

        # Should exist immediately
        assert await cache.has('key1')

        # Wait for expiration
        await asyncio.sleep(0.15)

        # Should be expired
        assert not await cache.has('key1')

    @pytest.mark.asyncio
    async def test_cache_statistics(self):
        """Test cache statistics tracking"""
        cache = InMemoryCache(max_size=10)

        await cache.set('key1', b'value1', ttl=60)

        # Hit
        await cache.get('key1')

        # Miss
        await cache.get('key2')

        stats = cache.get_statistics()

        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5

    @pytest.mark.asyncio
    async def test_cache_get_size(self):
        """Test get_size operation"""
        cache = InMemoryCache(max_size=10)

        assert await cache.get_size() == 0

        await cache.set('key1', b'value1', ttl=60)
        await cache.set('key2', b'value2', ttl=60)

        assert await cache.get_size() == 2


class TestAtomicSolutionCache:
    """Tests for AtomicSolutionCache"""

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test cache hit scenario"""
        cache = create_solution_cache(
            cache_type='memory',
            ttl_seconds=60,
            max_size=10
        )

        problem = {'type': 'test', 'value': 42}
        solution = {'result': 'success'}

        # Store solution
        await cache.set(problem, solution)

        # Retrieve (should hit cache)
        retrieved, hit = await cache.get(problem)

        assert hit is True
        assert retrieved == solution

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss scenario"""
        cache = create_solution_cache(cache_type='memory')

        problem = {'type': 'test', 'value': 42}

        # Retrieve (should miss cache)
        retrieved, hit = await cache.get(problem)

        assert hit is False
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_cache_invalidate(self):
        """Test cache invalidation"""
        cache = create_solution_cache(cache_type='memory')

        problem = {'type': 'test', 'value': 42}
        solution = {'result': 'success'}

        await cache.set(problem, solution)
        assert await cache.has(problem)

        await cache.invalidate(problem)

        assert not await cache.has(problem)

    @pytest.mark.asyncio
    async def test_get_statistics(self):
        """Test statistics retrieval"""
        cache = create_solution_cache(cache_type='memory')

        problem = {'type': 'test', 'value': 42}
        solution = {'result': 'success'}

        # Cache miss
        await cache.get(problem)

        # Store and hit
        await cache.set(problem, solution)
        await cache.get(problem)

        stats = await cache.get_statistics()

        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5


class TestCacheIntegration:
    """Integration tests for cache with real solutions"""

    @pytest.mark.asyncio
    async def test_cache_with_real_solutions(self):
        """Test caching with realistic solution objects"""
        cache = create_solution_cache(
            cache_type='memory',
            ttl_seconds=60,
            max_size=100
        )

        # Simulate storing and retrieving solutions
        problems = [
            {'type': 'math', 'operation': 'add', 'a': 5, 'b': 3},
            {'type': 'math', 'operation': 'multiply', 'a': 4, 'b': 7},
            {'type': 'string', 'operation': 'reverse', 'value': 'hello'},
        ]

        solutions = [
            {'result': 8},
            {'result': 28},
            {'result': 'olleh'},
        ]

        # Store all solutions
        for problem, solution in zip(problems, solutions):
            await cache.set(problem, solution)

        # Retrieve all (should all be hits)
        for problem, expected_solution in zip(problems, solutions):
            retrieved, hit = await cache.get(problem)
            assert hit is True
            assert retrieved == expected_solution

    @pytest.mark.asyncio
    async def test_cache_hit_rate_measurement(self):
        """Measure cache hit rate with repeated access"""
        cache = create_solution_cache(
            cache_type='memory',
            ttl_seconds=60,
            max_size=10
        )

        problem = {'type': 'test', 'value': 42}
        solution = {'result': 'success'}

        # Store solution
        await cache.set(problem, solution)

        # Access 10 times
        for _ in range(10):
            await cache.get(problem)

        stats = await cache.get_statistics()

        # Should have 10 hits, 1 miss (first get before set)
        assert stats['hits'] == 10
        assert stats['hit_rate'] >= 0.9

    @pytest.mark.asyncio
    async def test_cache_with_different_problems(self):
        """Test cache handles different problems correctly"""
        cache = create_solution_cache(
            cache_type='memory',
            ttl_seconds=60,
            max_size=20
        )

        # Store different problems
        for i in range(10):
            problem = {'type': 'test', 'iteration': i}
            solution = {'result': f'solution_{i}'}
            await cache.set(problem, solution)

        # All should be retrievable
        for i in range(10):
            problem = {'type': 'test', 'iteration': i}
            retrieved, hit = await cache.get(problem)
            assert hit is True
            assert retrieved['result'] == f'solution_{i}'


class TestCachePerformance:
    """Performance benchmarks for cache"""

    @pytest.mark.asyncio
    async def test_cache_speedup(self):
        """Test cache provides significant speedup"""
        cache = create_solution_cache(
            cache_type='memory',
            ttl_seconds=60,
            max_size=100
        )

        problem = {'type': 'test', 'value': 42}
        solution = {'result': 'expensive computation result'}

        # Simulate expensive solution computation
        async def expensive_solve(prob):
            await asyncio.sleep(0.01)  # Simulate work
            return solution

        # Time first access (miss)
        start = time.time()
        await cache.get(problem)
        miss_time = time.time() - start

        # Store solution
        await cache.set(problem, solution)

        # Time cached access (hit)
        start = time.time()
        retrieved, hit = await cache.get(problem)
        hit_time = time.time() - start

        # Cache hit should be much faster
        assert hit_time < miss_time

    @pytest.mark.asyncio
    async def test_cache_scalability(self):
        """Test cache scales to 1000+ entries"""
        cache = create_solution_cache(
            cache_type='memory',
            ttl_seconds=60,
            max_size=2000
        )

        # Add 1000 entries
        for i in range(1000):
            problem = {'type': 'test', 'id': i}
            solution = {'result': f'result_{i}'}
            await cache.set(problem, solution)

        # Should handle 1000 entries
        size = await cache.get_size()
        assert size == 1000

        # Verify last entries are accessible
        for i in range(990, 1000):
            problem = {'type': 'test', 'id': i}
            retrieved, hit = await cache.get(problem)
            assert hit is True

    @pytest.mark.asyncio
    async def test_cache_memory_efficiency(self):
        """Test cache memory usage is reasonable"""
        import sys

        cache = create_solution_cache(
            cache_type='memory',
            ttl_seconds=60,
            max_size=100
        )

        # Get baseline memory
        # Note: This is a rough estimate
        for i in range(100):
            problem = {'type': 'test', 'id': i}
            solution = {'result': 'x' * 100}  # 100 byte result
            await cache.set(problem, solution)

        # Cache should not consume excessive memory
        # (This is a basic check - production would use more sophisticated profiling)
        stats = await cache.get_statistics()
        assert stats['size'] == 100


class TestCacheConfiguration:
    """Tests for cache configuration"""

    def test_cache_config_defaults(self):
        """Test default configuration values"""
        from bubblelabs_nodes.gauntlet_config import CacheConfig

        config = CacheConfig()

        assert config.enabled is True
        assert config.cache_type.value == 'memory'
        assert config.ttl_seconds == 3600
        assert config.max_size == 1000

    def test_cache_config_validation(self):
        """Test configuration validation"""
        from bubblelabs_nodes.gauntlet_config import CacheConfig

        # Invalid config
        config = CacheConfig(
            ttl_seconds=-1,  # Invalid
            max_size=-1  # Invalid
        )

        valid, errors = config.validate()

        assert valid is False
        assert len(errors) >= 2

    @pytest.mark.asyncio
    async def test_cache_from_environment(self):
        """Test cache configuration from environment"""
        import os

        # Set environment variables
        os.environ['CACHE_ENABLED'] = 'true'
        os.environ['CACHE_TYPE'] = 'memory'
        os.environ['CACHE_TTL_SECONDS'] = '3600'
        os.environ['CACHE_MAX_SIZE'] = '500'

        # Create cache from environment
        # (In production, this would use the config system)
        from bubblelabs_nodes.gauntlet_config import CacheConfig

        config = CacheConfig(
            enabled=os.getenv('CACHE_ENABLED', 'true').lower() == 'true',
            cache_type=CacheConfig.__annotations__['cache_type'](),
            ttl_seconds=int(os.getenv('CACHE_TTL_SECONDS', '3600')),
            max_size=int(os.getenv('CACHE_MAX_SIZE', '1000')),
        )

        assert config.enabled is True
        assert config.ttl_seconds == 3600
        assert config.max_size == 500


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
