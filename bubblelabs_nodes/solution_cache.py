"""
Solution Caching System for OpenEvolve Gauntlet System

Provides intelligent caching of atomic solutions to dramatically speed up
repeated problems. Uses LRU eviction with configurable TTL and size limits.

Key Features:
- In-memory and Redis backends
- Problem hashing for cache keys
- Cache statistics (hit rate, miss rate)
- Configurable TTL and size limits
"""

from typing import Dict, Any, Optional, Protocol
import hashlib
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Metrics integration (lazy import to avoid circular dependency)
_metrics_collector = None

def get_metrics_collector():
    """Get the metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        try:
            from bubblelabs_nodes.gauntlet_metrics import get_metrics_collector as get_gm
            _metrics_collector = get_gm()
        except ImportError:
            logger.warning("Metrics collector not available")
            _metrics_collector = False  # Marker that import failed
    return _metrics_collector if _metrics_collector is not False else None


class CacheBackend(Protocol):
    """Protocol for cache backends"""

    async def get(self, key: str) -> Optional[bytes]:
        """Get value from cache"""
        ...

    async def set(self, key: str, value: bytes, ttl: int) -> bool:
        """Set value in cache with TTL in seconds"""
        ...

    async def has(self, key: str) -> bool:
        """Check if key exists in cache"""
        ...

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        ...

    async def clear(self) -> bool:
        """Clear all cache entries"""
        ...

    async def get_size(self) -> int:
        """Get current cache size"""
        ...


@dataclass
class CacheStatistics:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    sets: int = 0
    gets: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate"""
        return 1.0 - self.hit_rate


class ProblemHasher:
    """
    Generates consistent cache keys for problems.
    """

    def __init__(self):
        self.hash_cache = {}

    def normalize_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize problem for consistent hashing.

        Removes fields that shouldn't affect cache key (timestamps, IDs, etc.)
        """
        # Create a copy to avoid mutating original
        normalized = problem.copy()

        # Remove fields that shouldn't affect the solution
        normalized.pop('id', None)
        normalized.pop('timestamp', None)
        normalized.pop('execution_id', None)
        normalized.pop('cache_key', None)

        # Sort any lists/dicts for consistency
        if 'requires' in normalized and isinstance(normalized['requires'], list):
            normalized['requires'] = sorted(normalized['requires'])

        return normalized

    def generate_hash(self, problem: Dict[str, Any]) -> str:
        """
        Generate cache key hash for problem.

        Args:
            problem: Problem definition

        Returns:
            SHA256 hash as hex string
        """
        # Normalize the problem first
        normalized = self.normalize_problem(problem)

        # Convert to JSON string
        problem_json = json.dumps(normalized, sort_keys=True)

        # Generate hash
        hash_obj = hashlib.sha256(problem_json.encode('utf-8'))

        return f"problem:{hash_obj.hexdigest()}"

    def generate_hash_fast(self, problem: Dict[str, Any]) -> str:
        """
        Generate cache key hash using simpler (faster) method.
        """
        # For very simple problems, skip full JSON serialization
        key_parts = [
            problem.get('statement', ''),
            str(sorted(problem.get('requirements', []))),
            str(sorted(problem.get('constraints', [])))
        ]
        key_str = ':'.join(key_parts)
        return f"problem:fast:{hashlib.md5(key_str.encode()).hexdigest()}"


class InMemoryCache:
    """In-memory cache implementation with LRU eviction"""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl  # seconds
        self.cache: Dict[str, tuple[bytes, datetime]] = {}
        self.access_order: list[str] = []
        self.stats = CacheStatistics()

    async def get(self, key: str) -> Optional[bytes]:
        """Get value from cache"""
        self.stats.gets += 1

        if key not in self.cache:
            self.stats.misses += 1
            return None

        value, expiry = self.cache[key]

        # Check if expired
        if datetime.now() > expiry:
            # Expired
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
            self.stats.misses += 1
            return None

        # Update access order (LRU)
        self.access_order.remove(key)
        self.access_order.append(key)

        self.stats.hits += 1
        return value

    async def set(self, key: str, value: bytes, ttl: int) -> bool:
        """Set value in cache with TTL"""
        self.stats.sets += 1

        # Check size limit
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Evict least recently used
            if self.access_order:
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
                self.access_order.remove(lru_key)
                self.stats.evictions += 1

        # Set with expiry
        expiry = datetime.now() + timedelta(seconds=ttl or self.ttl)
        self.cache[key] = (value, expiry)

        # Add to access order if not present
        if key not in self.access_order:
            self.access_order.append(key)

        return True

    async def has(self, key: str) -> bool:
        """Check if key exists in cache"""
        if key not in self.cache:
            return False

        value, expiry = self.cache[key]
        return datetime.now() <= expiry

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
            return True
        return False

    async def clear(self) -> bool:
        """Clear all cache entries"""
        self.cache.clear()
        self.access_order.clear()
        return True

    async def get_size(self) -> int:
        """Get current cache size"""
        return len(self.cache)


class AtomicSolutionCache:
    """
    Main cache interface for atomic solutions.
    """

    def __init__(self, backend: CacheBackend, hasher: ProblemHasher = None):
        self.backend = backend
        self.hasher = hasher or ProblemHasher()
        self.enabled = True

    async def solve(self, problem: Dict[str, Any], solve_func: callable) -> Any:
        """
        Solve a problem with caching.

        Args:
            problem: Problem to solve
            solve_func: Async function to solve the problem

        Returns:
            Solution result
        """
        if not self.enabled:
            # Cache disabled, solve directly
            return await solve_func(problem)

        # Generate cache key
        cache_key = self.hasher.generate_hash(problem)

        # Try cache first
        cached = await self.backend.get(cache_key)
        if cached is not None:
            # Cache HIT - log and track metrics
            problem_id = problem.get('id', problem.get('statement', 'unknown'))
            logger.info(f"Cache HIT for problem: {problem_id}")

            # Track metrics
            collector = get_metrics_collector()
            if collector:
                collector.record_cache_operation(
                    operation='hit',
                    cache_type=type(self.backend).__name__,
                    key=cache_key[:16],  # First 16 chars of hash
                    metadata={'problem_id': str(problem_id)}
                )

            return json.loads(cached.decode('utf-8'))

        # Cache MISS - log and track metrics
        problem_id = problem.get('id', problem.get('statement', 'unknown'))
        logger.info(f"Cache MISS for problem: {problem_id}")

        # Track metrics
        collector = get_metrics_collector()
        if collector:
            collector.record_cache_operation(
                operation='miss',
                cache_type=type(self.backend).__name__,
                key=cache_key[:16],
                metadata={'problem_id': str(problem_id)}
            )
        solution = await solve_func(problem)

        # Store in cache
        try:
            solution_json = json.dumps(solution)
            await self.backend.set(cache_key, solution_json.encode('utf-8'), ttl=3600)
        except Exception as e:
            logger.warning(f"Failed to cache solution: {e}")

        return solution

    async def get(self, problem: Dict[str, Any]) -> Optional[Any]:
        """Get cached solution if available"""
        if not self.enabled:
            return None

        cache_key = self.hasher.generate_hash(problem)
        cached = await self.backend.get(cache_key)

        if cached:
            return json.loads(cached.decode('utf-8'))
        return None

    async def has(self, problem: Dict[str, Any]) -> bool:
        """Check if solution is cached"""
        if not self.enabled:
            return False

        cache_key = self.hasher.generate_hash(problem)
        return await self.backend.has(cache_key)

    async def invalidate(self, problem: Dict[str, Any]) -> bool:
        """Invalidate cached solution"""
        if not self.enabled:
            return False

        cache_key = self.hasher.generate_hash(problem)
        return await self.backend.delete(cache_key)

    async def clear(self) -> bool:
        """Clear all cached solutions"""
        return await self.backend.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if isinstance(self.backend, InMemoryCache):
            return {
                'enabled': self.enabled,
                'type': 'memory',
                'hits': self.backend.stats.hits,
                'misses': self.backend.stats.misses,
                'evictions': self.backend.stats.evictions,
                'hit_rate': self.backend.stats.hit_rate,
                'size': len(self.backend.cache),
                'max_size': self.backend.max_size,
            }
        return {
            'enabled': self.enabled,
            'type': 'unknown',
        }


def create_solution_cache(config: Dict[str, Any] = None) -> AtomicSolutionCache:
    """
    Factory function to create solution cache.

    Args:
        config: Cache configuration

    Returns:
        AtomicSolutionCache instance
    """
    config = config or {}
    backend = InMemoryCache(
        max_size=config.get('max_size', 1000),
        ttl=config.get('ttl', 3600)
    )

    hasher = ProblemHasher()

    return AtomicSolutionCache(backend, hasher)
