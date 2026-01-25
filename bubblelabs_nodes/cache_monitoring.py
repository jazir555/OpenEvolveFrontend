"""
Cache Monitoring and Metrics Integration

Provides comprehensive monitoring, metrics collection, and observability
for the solution cache system with Prometheus integration.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import logging
import time
from .solution_cache import AtomicSolutionCache
from .gauntlet_metrics import MetricsCollector, MetricType, get_metrics_collector

logger = logging.getLogger(__name__)


class CacheMonitor:
    """
    Monitor for solution cache performance and health.
    """

    def __init__(self, cache: AtomicSolutionCache, metrics_collector: MetricsCollector = None):
        self.cache = cache
        self.metrics = metrics_collector or get_metrics_collector()
        self.start_time = time.time()

    async def record_cache_hit(self, problem_hash: str, solution: Any):
        """Record a cache hit"""
        await self.metrics.increment_counter('cache_hit_count')
        await self.metrics.set_gauge('cache_hit_rate', await self._calculate_hit_rate())
        await self.metrics.set_gauge('cache_size', await self.cache.size())

        logger.debug(f"Cache hit for problem: {problem_hash[:16]}...")

    async def record_cache_miss(self, problem_hash: str):
        """Record a cache miss"""
        await self.metrics.increment_counter('cache_miss_count')
        await self.metrics.set_gauge('cache_hit_rate', await self._calculate_hit_rate())

        logger.debug(f"Cache miss for problem: {problem_hash[:16]}...")

    async def record_cache_eviction(self, problem_hash: str):
        """Record a cache eviction"""
        await self.metrics.increment_counter('cache_eviction_count')
        await self.metrics.set_gauge('cache_size', await self.cache.size())

        logger.info(f"Cache eviction for problem: {problem_hash[:16]}...")

    async def record_cache_set(self, problem_hash: str):
        """Record a cache set operation"""
        await self.metrics.increment_counter('cache_set_count')
        await self.metrics.set_gauge('cache_size', await self.cache.size())

    async def _calculate_hit_rate(self) -> float:
        """Calculate current hit rate"""
        stats = await self.cache.get_statistics()
        return stats['hit_rate']

    async def get_health_status(self) -> Dict[str, Any]:
        """Get cache health status"""
        stats = await self.cache.get_statistics()
        size = await self.cache.size()

        # Calculate health score
        hit_rate = stats['hit_rate']
        if hit_rate >= 0.5:
            health = 'healthy'
        elif hit_rate >= 0.3:
            health = 'degraded'
        else:
            health = 'unhealthy'

        return {
            'health': health,
            'size': size,
            'hit_rate': hit_rate,
            'miss_rate': stats['miss_rate'],
            'hits': stats['hits'],
            'misses': stats['misses'],
            'evictions': stats['evictions'],
            'uptime_seconds': time.time() - self.start_time,
        }


class StructuredCacheLogger:
    """
    Structured logging for cache operations.
    """

    def __init__(self, cache: AtomicSolutionCache):
        self.cache = cache

    async def log_cache_hit(self, problem_hash: str, solution_id: str):
        """Log cache hit with structured data"""
        logger.info(
            "cache_hit",
            extra={
                'cache': {
                    'event': 'hit',
                    'problem_hash': problem_hash[:16],
                    'solution_id': solution_id,
                    'timestamp': datetime.utcnow().isoformat(),
                }
            }
        )

    async def log_cache_miss(self, problem_hash: str):
        """Log cache miss with structured data"""
        logger.info(
            "cache_miss",
            extra={
                'cache': {
                    'event': 'miss',
                    'problem_hash': problem_hash[:16],
                    'timestamp': datetime.utcnow().isoformat(),
                }
            }
        )

    async def log_cache_eviction(self, problem_hash: str, reason: str = 'lru'):
        """Log cache eviction with structured data"""
        logger.warning(
            "cache_eviction",
            extra={
                'cache': {
                    'event': 'eviction',
                    'problem_hash': problem_hash[:16],
                    'reason': reason,
                    'timestamp': datetime.utcnow().isoformat(),
                }
            }
        )

    async def log_cache_set(self, problem_hash: str, ttl: int):
        """Log cache set with structured data"""
        logger.debug(
            "cache_set",
            extra={
                'cache': {
                    'event': 'set',
                    'problem_hash': problem_hash[:16],
                    'ttl': ttl,
                    'timestamp': datetime.utcnow().isoformat(),
                }
            }
        )


class MonitoredCache:
    """
    Wrapper that adds monitoring and logging to cache operations.
    """

    def __init__(self, cache: AtomicSolutionCache):
        self.cache = cache
        self.monitor = CacheMonitor(cache)
        self.logger = StructuredCacheLogger(cache)

    async def get(self, problem: Dict[str, Any]) -> tuple[Any, bool]:
        """Get with monitoring and logging"""
        from .solution_cache import ProblemHasher

        hasher = ProblemHasher()
        problem_hash = hasher.generate_hash(problem)

        # Get from cache
        result, hit = await self.cache.get(problem)

        if hit:
            await self.monitor.record_cache_hit(problem_hash, str(result))
            await self.logger.log_cache_hit(problem_hash, str(result)[:50])
        else:
            await self.monitor.record_cache_miss(problem_hash)
            await self.logger.log_cache_miss(problem_hash)

        return result, hit

    async def set(self, problem: Dict[str, Any], solution: Any) -> bool:
        """Set with monitoring and logging"""
        from .solution_cache import ProblemHasher

        hasher = ProblemHasher()
        problem_hash = hasher.generate_hash(problem)
        ttl = self.cache.config.ttl_seconds

        await self.logger.log_cache_set(problem_hash, ttl)
        success = await self.cache.set(problem, solution)

        if success:
            await self.monitor.record_cache_set(problem_hash)

        return success

    async def invalidate(self, problem: Dict[str, Any]) -> bool:
        """Invalidate with monitoring"""
        from .solution_cache import ProblemHasher

        hasher = ProblemHasher()
        problem_hash = hasher.generate_hash(problem)

        return await self.cache.invalidate(problem)

    async def has(self, problem: Dict[str, Any]) -> bool:
        """Check if problem is cached"""
        return await self.cache.has(problem)

    async def size(self) -> int:
        """Get cache size"""
        return await self.cache.size()

    async def clear(self) -> bool:
        """Clear cache with monitoring"""
        result = await self.cache.clear()
        if result:
            await self.metrics.set_gauge('cache_size', 0)
        return result

    async def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return await self.cache.get_statistics()


def create_monitored_cache(cache_config: Any = None) -> MonitoredCache:
    """
    Create a monitored cache instance.

    Args:
        cache_config: Optional cache configuration

    Returns:
        MonitoredCache instance
    """
    from .solution_cache import create_solution_cache

    if cache_config is None:
        from .gauntlet_config import CacheConfig
        cache_config = CacheConfig()

    base_cache = create_solution_cache(
        cache_type=cache_config.cache_type.value,
        ttl_seconds=cache_config.ttl_seconds,
        max_size=cache_config.max_size,
    )

    return MonitoredCache(base_cache)
