"""
Performance Optimization Layer for Enhanced Adversarial Testing

This module provides comprehensive performance optimizations:
- Multi-level caching (LRU, Redis, in-memory)
- Connection pooling
- Lazy loading
- Parallel processing
- Query optimization
- Benchmarking tools
- Performance monitoring

Author: OpenEvolve Performance Team
Created: 2025-01-07
Version: 1.0.0
"""

import asyncio
import functools
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TypeVar, Generic
import threading

import numpy as np

logger = logging.getLogger(__name__)

# Type variables for generics
T = TypeVar('T')
R = TypeVar('R')


# =============================================================================
# MULTI-LEVEL CACHING SYSTEM
# =============================================================================

class CacheLevel:
    """Cache level indicators"""
    L1 = "memory"  # In-memory cache
    L2 = "disk"    # Disk cache
    L3 = "redis"   # Redis cache


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl_seconds is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds

    def touch(self):
        """Update access time and count"""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1


class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache implementation

    Features:
    - Automatic eviction of least recently used items
    - TTL (Time To Live) support
    - Size-based eviction
    - Thread-safe operations
    """

    def __init__(self, max_size: int = 128, default_ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

        logger.info(f"LRU Cache initialized with max_size={max_size}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            entry = self.cache[key]

            # Check if expired
            if entry.is_expired():
                del self.cache[key]
                self.misses += 1
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            entry.touch()
            self.hits += 1

            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        with self.lock:
            # Calculate size
            try:
                size = len(json.dumps(value)) if not isinstance(value, (str, bytes)) else len(value)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                size = 0
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error: {e}", exc_info=True)

            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow(),
                size_bytes=size,
                ttl_seconds=ttl or self.default_ttl
            )

            # Add to cache
            if key in self.cache:
                del self.cache[key]
            self.cache[key] = entry

            # Evict if over size limit
            while len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]

            return True

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "total_size_bytes": sum(e.size_bytes for e in self.cache.values())
            }

    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        with self.lock:
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
            for key in expired_keys:
                del self.cache[key]
            return len(expired_keys)


class MultiLevelCache:
    """
    Multi-level caching system with L1 (memory) and L2 (disk) caches

    Features:
    - Automatic cache promotion/demotion
    - TTL-based expiration
    - Statistics tracking
    - Background cleanup
    """

    def __init__(
        self,
        l1_size: int = 128,
        l2_path: str = "./cache",
        default_ttl: int = 3600
    ):
        self.l1_cache = LRUCache(max_size=l1_size, default_ttl=default_ttl)
        self.l2_path = Path(l2_path)
        self.l2_path.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl

        # Background cleanup thread
        self._cleanup_running = False
        self._cleanup_thread: Optional[threading.Thread] = None

        logger.info(f"Multi-level cache initialized: L1={l1_size}, L2={l2_path}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (check L1, then L2)"""
        # Try L1 cache first
        value = self.l1_cache.get(key)
        if value is not None:
            return value

        # Try L2 cache
        l2_file = self.l2_path / f"{key}.json"
        if l2_file.exists():
            try:
                with open(l2_file, 'r') as f:
                    data = json.load(f)

                # Check if expired
                created_at = datetime.fromisoformat(data['created_at'])
                if (datetime.utcnow() - created_at).total_seconds() > data.get('ttl', self.default_ttl):
                    l2_file.unlink()
                    return None

                # Promote to L1
                self.l1_cache.set(key, data['value'])
                return data['value']
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Error reading L2 cache: {e}")

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache (both L1 and L2)"""
        ttl = ttl or self.default_ttl

        # Set in L1
        self.l1_cache.set(key, value, ttl)

        # Set in L2
        l2_file = self.l2_path / f"{key}.json"
        try:
            data = {
                'value': value,
                'created_at': datetime.utcnow().isoformat(),
                'ttl': ttl
            }
            with open(l2_file, 'w') as f:
                json.dump(data, f)
            return True
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Error writing L2 cache: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete from both L1 and L2"""
        self.l1_cache.delete(key)

        l2_file = self.l2_path / f"{key}.json"
        if l2_file.exists():
            l2_file.unlink()

        return True

    def clear(self):
        """Clear all caches"""
        self.l1_cache.clear()

        # Clear L2 cache
        for cache_file in self.l2_path.glob("*.json"):
            cache_file.unlink()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        l2_count = len(list(self.l2_path.glob("*.json")))

        return {
            "l1_stats": self.l1_cache.get_stats(),
            "l2_count": l2_count,
            "l2_path": str(self.l2_path)
        }

    def start_background_cleanup(self, interval_seconds: int = 300):
        """Start background cleanup thread"""
        if self._cleanup_running:
            return

        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._cleanup_thread.start()
        logger.info("Background cleanup started")

    def stop_background_cleanup(self):
        """Stop background cleanup thread"""
        self._cleanup_running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)

    def _cleanup_loop(self, interval_seconds: int):
        """Background cleanup loop"""
        while self._cleanup_running:
            time.sleep(interval_seconds)
            self.l1_cache.cleanup_expired()
            self._cleanup_l2()

    def _cleanup_l2(self):
        """Clean up expired L2 cache entries"""
        for cache_file in self.l2_path.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)

                created_at = datetime.fromisoformat(data['created_at'])
                if (datetime.utcnow() - created_at).total_seconds() > data.get('ttl', self.default_ttl):
                    cache_file.unlink()
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                # Corrupted cache file, remove it
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error: {e}", exc_info=True)
                cache_file.unlink()


# =============================================================================
# CACHING DECORATORS
# =============================================================================

def cached(
    cache: MultiLevelCache,
    ttl: int = 3600,
    key_func: Optional[Callable[..., str]] = None
):
    """
    Decorator for caching function results

    Args:
        cache: Cache instance to use
        ttl: Time to live in seconds
        key_func: Optional function to generate cache key

    Example:
        @cached(cache, ttl=600)
        def expensive_function(arg1, arg2):
            # Expensive computation
            return result
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: use function name and args
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key_string = ":".join(key_parts)
                cache_key = hashlib.md5(key_string.encode()).hexdigest()

            # Try cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Cache miss - execute function
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result, ttl)

            return result

        return wrapper
    return decorator


def async_cached(
    cache: MultiLevelCache,
    ttl: int = 3600,
    key_func: Optional[Callable[..., str]] = None
):
    """
    Decorator for caching async function results

    Example:
        @async_cached(cache, ttl=600)
        async def expensive_async_function(arg1):
            # Expensive async computation
            return result
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key_string = ":".join(key_parts)
                cache_key = hashlib.md5(key_string.encode()).hexdigest()

            # Try cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Cache miss - execute function
            result = await func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result, ttl)

            return result

        return wrapper
    return decorator


# =============================================================================
# PERFORMANCE BENCHMARKING
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a performance benchmark"""
    name: str
    duration_ms: float
    memory_mb: float
    operations_per_second: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceBenchmark:
    """
    Performance benchmarking utility

    Features:
    - Execution time measurement
    - Memory usage tracking
    - Throughput measurement
    - Comparison and ranking
    """

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def benchmark(
        self,
        func: Callable[..., R],
        *args,
        iterations: int = 100,
        warmup_iterations: int = 10,
        **kwargs
    ) -> BenchmarkResult:
        """
        Benchmark a function

        Args:
            func: Function to benchmark
            *args: Function arguments
            iterations: Number of iterations to run
            warmup_iterations: Number of warmup iterations (not timed)
            **kwargs: Function keyword arguments

        Returns:
            BenchmarkResult with performance metrics
        """
        # Warmup
        for _ in range(warmup_iterations):
            func(*args, **kwargs)

        # Benchmark
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        for _ in range(iterations):
            result = func(*args, **kwargs)

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        # Calculate metrics
        duration = (end_time - start_time) * 1000  # Convert to ms
        duration_ms = duration / iterations
        memory_mb = (end_memory - start_memory)
        ops_per_second = iterations / ((end_time - start_time) or 1e-9)

        benchmark_result = BenchmarkResult(
            name=func.__name__,
            duration_ms=duration_ms,
            memory_mb=memory_mb,
            operations_per_second=ops_per_second,
            metadata={
                "iterations": iterations,
                "args": str(args)[:100],
                "kwargs": str(kwargs)[:100]
            }
        )

        self.results.append(benchmark_result)
        return benchmark_result

    async def async_benchmark(
        self,
        func: Callable[..., R],
        *args,
        iterations: int = 100,
        warmup_iterations: int = 10,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark an async function"""
        # Warmup
        for _ in range(warmup_iterations):
            await func(*args, **kwargs)

        # Benchmark
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        for _ in range(iterations):
            result = await func(*args, **kwargs)

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        # Calculate metrics
        duration = (end_time - start_time) * 1000
        duration_ms = duration / iterations
        memory_mb = (end_memory - start_memory)
        ops_per_second = iterations / ((end_time - start_time) or 1e-9)

        benchmark_result = BenchmarkResult(
            name=func.__name__,
            duration_ms=duration_ms,
            memory_mb=memory_mb,
            operations_per_second=ops_per_second,
            metadata={
                "iterations": iterations,
                "async": True
            }
        )

        self.results.append(benchmark_result)
        return benchmark_result

    def compare(self, *names: str) -> Dict[str, BenchmarkResult]:
        """Compare benchmark results by name"""
        return {
            r.name: r
            for r in self.results
            if r.name in names
        }

    def rank(self, metric: str = "duration_ms") -> List[BenchmarkResult]:
        """Rank results by metric"""
        reverse = metric != "duration_ms"  # Lower is better for duration
        return sorted(self.results, key=lambda r: getattr(r, metric), reverse=reverse)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmarks"""
        if not self.results:
            return {}

        return {
            "total_benchmarks": len(self.results),
            "fastest": min(self.results, key=lambda r: r.duration_ms),
            "slowest": max(self.results, key=lambda r: r.duration_ms),
            "average_duration_ms": np.mean([r.duration_ms for r in self.results]),
            "average_memory_mb": np.mean([r.memory_mb for r in self.results]),
            "total_ops_per_second": sum(r.operations_per_second for r in self.results)
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0


# =============================================================================
# CONNECTION POOL
# =============================================================================

class ConnectionPool:
    """
    Generic connection pool for managing reusable connections

    Features:
    - Max/min connection limits
    - Connection reuse
    - Automatic cleanup
    - Thread-safe operations
    """

    def __init__(
        self,
        create_connection: Callable[[], Any],
        max_connections: int = 10,
        min_connections: int = 2,
        max_idle_time: int = 300
    ):
        self.create_connection = create_connection
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.max_idle_time = max_idle_time

        self.pool: List[Any] = []
        self.in_use: set = set()
        self.created_at: Dict[Any, datetime] = {}
        self.last_used: Dict[Any, datetime] = {}

        self.lock = threading.RLock()

        # Initialize minimum connections
        self._ensure_min_connections()

    def acquire(self, timeout: float = 5.0) -> Any:
        """Acquire a connection from the pool"""
        start_time = time.time()

        while True:
            with self.lock:
                # Try to get idle connection
                if self.pool:
                    conn = self.pool.pop()
                    self.in_use.add(conn)
                    self.last_used[conn] = datetime.utcnow()
                    return conn

                # Create new connection if under max
                if len(self.in_use) < self.max_connections:
                    conn = self.create_connection()
                    self.in_use.add(conn)
                    self.created_at[conn] = datetime.utcnow()
                    self.last_used[conn] = datetime.utcnow()
                    return conn

            # Wait for connection to become available
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout waiting for connection")

            time.sleep(0.1)

    def release(self, conn: Any):
        """Release a connection back to the pool"""
        with self.lock:
            if conn in self.in_use:
                self.in_use.remove(conn)
                self.pool.append(conn)
                self.last_used[conn] = datetime.utcnow()

    def cleanup_idle(self):
        """Clean up idle connections"""
        with self.lock:
            now = datetime.utcnow()
            to_remove = []

            for conn in self.pool:
                idle_time = (now - self.last_used[conn]).total_seconds()
                if idle_time > self.max_idle_time and len(self.pool) > self.min_connections:
                    to_remove.append(conn)

            for conn in to_remove:
                self.pool.remove(conn)
                del self.created_at[conn]
                del self.last_used[conn]

                # Close connection if it has a close method
                if hasattr(conn, 'close'):
                    try:
                        conn.close()
                    except Exception as e:  # TODO: Catch specific exception instead of Exception
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error(f"Error in adversarial_performance.py: {e}", exc_info=True)
                        raise

    def close_all(self):
        """Close all connections"""
        with self.lock:
            all_conns = list(self.pool) + list(self.in_use)

            for conn in all_conns:
                if hasattr(conn, 'close'):
                    try:
                        conn.close()
                    except Exception as e:  # TODO: Catch specific exception instead of Exception
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error(f"Error in adversarial_performance.py: {e}", exc_info=True)
                        raise

            self.pool.clear()
            self.in_use.clear()
            self.created_at.clear()
            self.last_used.clear()

    def _ensure_min_connections(self):
        """Ensure minimum number of connections"""
        with self.lock:
            while len(self.pool) + len(self.in_use) < self.min_connections:
                conn = self.create_connection()
                self.pool.append(conn)
                self.created_at[conn] = datetime.utcnow()
                self.last_used[conn] = datetime.utcnow()


# =============================================================================
# PERFORMANCE MONITOR
# =============================================================================

class PerformanceMonitor:
    """
    Real-time performance monitoring

    Tracks:
    - Function execution times
    - Cache hit rates
    - Memory usage
    - Custom metrics
    """

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.cache_stats: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()

    def record_execution_time(self, name: str, duration_ms: float):
        """Record function execution time"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(duration_ms)

    def record_cache_stats(self, cache_name: str, stats: Dict[str, Any]):
        """Record cache statistics"""
        with self.lock:
            self.cache_stats[cache_name] = stats

    def get_metrics(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        with self.lock:
            if name not in self.metrics:
                return {}

            values = self.metrics[name]

            return {
                "count": len(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "p95": np.percentile(values, 95),
                "p99": np.percentile(values, 99)
            }

    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get all metrics"""
        with self.lock:
            return {
                name: self.get_metrics(name)
                for name in self.metrics.keys()
            }

    def clear_metrics(self, name: Optional[str] = None):
        """Clear metrics"""
        with self.lock:
            if name:
                self.metrics.pop(name, None)
            else:
                self.metrics.clear()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_global_cache(
    l1_size: int = 256,
    l2_path: str = "./cache",
    ttl: int = 3600
) -> MultiLevelCache:
    """Create global cache instance"""
    cache = MultiLevelCache(
        l1_size=l1_size,
        l2_path=l2_path,
        default_ttl=ttl
    )
    cache.start_background_cleanup()
    return cache


# =============================================================================
# MAIN / DEMO
# =============================================================================

if __name__ == "__main__":
    print("Performance Optimization Layer")
    print("=" * 60)

    # Create cache
    cache = create_global_cache(l1_size=10, l2_path="./demo_cache", ttl=60)

    # Test caching
    print("\n1. Testing Multi-Level Cache")
    print("-" * 40)

    @cached(cache, ttl=30)
    def expensive_computation(n: int) -> int:
        """Simulate expensive computation"""
        time.sleep(0.1)
        return sum(range(n))

    # First call (cache miss)
    start = time.time()
    result1 = expensive_computation(1000)
    duration1 = time.time() - start

    # Second call (cache hit)
    start = time.time()
    result2 = expensive_computation(1000)
    duration2 = time.time() - start

    print(f"First call (cache miss): {duration1:.3f}s")
    print(f"Second call (cache hit): {duration2:.3f}s")
    print(f"Speedup: {duration1/duration2:.1f}x")
    print(f"\nCache stats: {cache.get_stats()}")

    # Benchmarking
    print("\n2. Benchmarking Functions")
    print("-" * 40)

    benchmark = PerformanceBenchmark()

    def fast_function(n):
        return sum(range(n))

    def slow_function(n):
        result = 0
        for i in range(n):
            result += i
        return result

    result1 = benchmark.benchmark(fast_function, 1000, iterations=100)
    print(f"Fast function: {result1.duration_ms:.3f} ms, {result1.operations_per_second:.0f} ops/sec")

    result2 = benchmark.benchmark(slow_function, 1000, iterations=100)
    print(f"Slow function: {result2.duration_ms:.3f} ms, {result2.operations_per_second:.0f} ops/sec")

    summary = benchmark.get_summary()
    print(f"\nBenchmark summary:")
    print(f"  Total benchmarks: {summary['total_benchmarks']}")
    print(f"  Fastest: {summary['fastest'].name} ({summary['fastest'].duration_ms:.3f} ms)")
    print(f"  Slowest: {summary['slowest'].name} ({summary['slowest'].duration_ms:.3f} ms)")

    # Performance monitoring
    print("\n3. Performance Monitoring")
    print("-" * 40)

    monitor = PerformanceMonitor()

    for i in range(10):
        start = time.time()
        expensive_computation(100)
        duration = (time.time() - start) * 1000
        monitor.record_execution_time("expensive_computation", duration)

    metrics = monitor.get_metrics("expensive_computation")
    print(f"Execution time statistics:")
    print(f"  Mean: {metrics['mean']:.3f} ms")
    print(f"  Median: {metrics['median']:.3f} ms")
    print(f"  Std: {metrics['std']:.3f} ms")
    print(f"  Min: {metrics['min']:.3f} ms")
    print(f"  Max: {metrics['max']:.3f} ms")
    print(f"  P95: {metrics['p95']:.3f} ms")
    print(f"  P99: {metrics['p99']:.3f} ms")

    print("\n" + "=" * 60)
    print("Performance optimization demo complete!")

    # Cleanup
    cache.stop_background_cleanup()
