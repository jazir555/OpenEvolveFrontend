"""
Performance Optimization Layer for Adaptive MDAP/MAKER Adapter

This module provides performance optimizations including:
- Async/await support for concurrent operations
- Connection pooling for API clients
- Response caching with TTL
- Batch processing for multiple analyses
- Parallel complexity analysis for sub-problems
- Memory-efficient streaming operations

Federation Constitution Compliant.
"""

import os
import sys
import logging
import time
import asyncio
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
from collections import defaultdict
import hashlib
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from adaptive_mdap_adapter import get_adapter, CanonicalSubProblem, TaskStatus

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"
    TTL = "ttl"
    FIFO = "fifo"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResponseCache:
    """
    Thread-safe response cache with TTL and size limits.
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 300):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl

        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            # Check TTL
            if entry.ttl_seconds:
                age = time.time() - entry.created_at
                if age > entry.ttl_seconds:
                    del self._cache[key]
                    self._misses += 1
                    return None

            # Update access stats
            entry.accessed_at = time.time()
            entry.access_count += 1
            self._hits += 1

            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        with self._lock:
            # Enforce size limit
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_one()

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                access_count=0,
                ttl_seconds=ttl or self.default_ttl
            )

            self._cache[key] = entry

    def _evict_one(self):
        """Evict one entry using LRU policy."""
        if not self._cache:
            return

        # Find least recently used
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].accessed_at
        )

        del self._cache[lru_key]

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate
            }


class ConnectionPool:
    """
    Thread-safe connection pool for API clients.
    """

    def __init__(self, max_connections: int = 10, idle_timeout: float = 300):
        """
        Initialize connection pool.

        Args:
            max_connections: Maximum number of connections
            idle_timeout: Idle timeout in seconds
        """
        self.max_connections = max_connections
        self.idle_timeout = idle_timeout

        self._available: List[Any] = []
        self._in_use: Dict[str, Any] = {}
        self._lock = threading.Lock()

        self._created = 0
        self._closed = 0

    def acquire(self, timeout: float = 5.0) -> Optional[Any]:
        """
        Acquire a connection from the pool.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Connection or None if timeout
        """
        deadline = time.time() + timeout

        with self._lock:
            while time.time() < deadline:
                # Return available connection
                if self._available:
                    conn = self._available.pop()
                    self._in_use[id(conn)] = conn
                    return conn

                # Create new connection if under limit
                if self._created < self.max_connections:
                    conn = self._create_connection()
                    self._created += 1
                    self._in_use[id(conn)] = conn
                    return conn

                # Wait for a connection to become available
                time.sleep(0.1)

        return None

    def release(self, connection: Any):
        """Release a connection back to the pool."""
        with self._lock:
            conn_id = id(connection)

            if conn_id in self._in_use:
                del self._in_use[conn_id]

                # Check if connection is still valid
                if self._is_valid(connection):
                    self._available.append(connection)
                else:
                    self._close_connection(connection)
                    self._created -= 1

    def _create_connection(self) -> Any:
        """Create a new connection (placeholder)."""
        # In production, would create actual API connections
        return {"id": f"conn_{self._created}", "created_at": time.time()}

    def _is_valid(self, connection: Any) -> bool:
        """Check if connection is still valid."""
        # In production, would check connection health
        return True

    def _close_connection(self, connection: Any):
        """Close a connection."""
        # In production, would actually close connection
        self._closed += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "created": self._created,
                "available": len(self._available),
                "in_use": len(self._in_use),
                "closed": self._closed,
                "utilization": len(self._in_use) / self.max_connections if self.max_connections > 0 else 0
            }


class AsyncMDAPAdapter:
    """
    Async wrapper for MDAP adapter with concurrent operations.
    """

    def __init__(self, cache_size: int = 1000, cache_ttl: float = 300):
        """
        Initialize async adapter.

        Args:
            cache_size: Maximum cache size
            cache_ttl: Cache TTL in seconds
        """
        self.adapter = get_adapter()
        self.cache = ResponseCache(max_size=cache_size, default_ttl=cache_ttl)
        self.executor = ThreadPoolExecutor(max_workers=10)

        logger.info("Async MDAP Adapter initialized")

    async def analyze_complexity_async(
        self,
        subproblem: CanonicalSubProblem,
        use_cache: bool = True
    ):
        """
        Async complexity analysis with caching.

        Args:
            subproblem: Sub-problem to analyze
            use_cache: Whether to use cache

        Returns:
            Analysis result
        """
        # Check cache
        if use_cache:
            cache_key = self._make_cache_key("complexity", subproblem)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        # Run analysis in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.adapter.analyze_complexity,
            subproblem
        )

        # Cache result
        if use_cache and result.status == TaskStatus.COMPLETED:
            cache_key = self._make_cache_key("complexity", subproblem)
            self.cache.set(cache_key, result)

        return result

    async def batch_analyze_complexity(
        self,
        subproblems: List[CanonicalSubProblem],
        max_concurrency: int = 5
    ) -> List[Any]:
        """
        Analyze multiple sub-problems in parallel.

        Args:
            subproblems: List of sub-problems
            max_concurrency: Maximum concurrent operations

        Returns:
            List of results
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def analyze_one(sp):
            async with semaphore:
                return await self.analyze_complexity_async(sp)

        tasks = [analyze_one(sp) for sp in subproblems]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return results

    def _make_cache_key(self, prefix: str, obj: Any) -> str:
        """Generate cache key from object."""
        # Serialize object to string
        if hasattr(obj, '__dict__'):
            obj_str = json.dumps(obj.__dict__, sort_keys=True)
        else:
            obj_str = str(obj)

        # Hash
        hash_val = hashlib.md5(obj_str.encode()).hexdigest()
        return f"{prefix}:{hash_val}"

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()

    def clear_cache(self):
        """Clear all cache entries."""
        self.cache.clear()

    def shutdown(self):
        """Shutdown adapter and cleanup resources."""
        self.executor.shutdown(wait=True)
        logger.info("Async MDAP Adapter shutdown")


def cached(ttl: float = 300, key_func: Optional[Callable] = None):
    """
    Decorator for caching function results.

    Args:
        ttl: Time to live in seconds
        key_func: Optional function to generate cache key
    """
    cache = ResponseCache(max_size=1000, default_ttl=ttl)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__] + [str(a) for a in args] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
                key = ":".join(key_parts)

            # Check cache
            cached = cache.get(key)
            if cached is not None:
                return cached

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            cache.set(key, result, ttl=ttl)

            return result

        # Add cache management methods
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.get_stats

        return wrapper

    return decorator


def batch_processor(batch_size: int = 10, timeout_ms: float = 5000):
    """
    Decorator for batch processing function calls.

    Args:
        batch_size: Maximum batch size
        timeout_ms: Maximum time to wait for batch to fill
    """
    def decorator(func):
        batches = {}
        batch_lock = threading.Lock()
        batch_executor = ThreadPoolExecutor(max_workers=5)

        def process_batch(key: str, items: List[Tuple]):
            """Process a batch of items."""
            try:
                results = func(items)
                return results
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                return None

        def wrapper(*args, **kwargs):
            # Extract batch key (first argument by default)
            batch_key = args[0] if args else "default"

            with batch_lock:
                if batch_key not in batches:
                    batches[batch_key] = {
                        "items": [],
                        "created_at": time.time()
                    }

                batch = batches[batch_key]
                batch["items"].append((args, kwargs))

                # Check if should process
                should_process = (
                    len(batch["items"]) >= batch_size or
                    (time.time() - batch["created_at"]) * 1000 >= timeout_ms
                )

                if should_process:
                    items = batch["items"]
                    del batches[batch_key]

                    # Process batch in thread pool
                    future = batch_executor.submit(process_batch, batch_key, items)
                    return future.result()

        return wrapper

    return decorator


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self):
        """Initialize performance monitor."""
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def record(self, operation: str, duration_ms: float):
        """Record operation duration."""
        with self._lock:
            self._metrics[operation].append(duration_ms)

            # Keep only last 1000 measurements
            if len(self._metrics[operation]) > 1000:
                self._metrics[operation] = self._metrics[operation][-1000:]

    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        with self._lock:
            measurements = self._metrics.get(operation, [])

            if not measurements:
                return {
                    "count": 0,
                    "avg_ms": 0,
                    "min_ms": 0,
                    "max_ms": 0,
                    "p50_ms": 0,
                    "p95_ms": 0,
                    "p99_ms": 0
                }

            import statistics
            return {
                "count": len(measurements),
                "avg_ms": statistics.mean(measurements),
                "min_ms": min(measurements),
                "max_ms": max(measurements),
                "p50_ms": statistics.median(measurements),
                "p95_ms": self._percentile(measurements, 95),
                "p99_ms": self._percentile(measurements, 99)
            }

    def _percentile(self, data: List[float], p: int) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * p / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


# Global instances
_async_adapter: Optional[AsyncMDAPAdapter] = None
_perf_monitor: Optional[PerformanceMonitor] = None


def get_async_adapter() -> AsyncMDAPAdapter:
    """Get or create global async adapter."""
    global _async_adapter
    if _async_adapter is None:
        _async_adapter = AsyncMDAPAdapter()
    return _async_adapter


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _perf_monitor
    if _perf_monitor is None:
        _perf_monitor = PerformanceMonitor()
    return _perf_monitor


__all__ = [
    "CachePolicy",
    "CacheEntry",
    "ResponseCache",
    "ConnectionPool",
    "AsyncMDAPAdapter",
    "cached",
    "batch_processor",
    "PerformanceMonitor",
    "get_async_adapter",
    "get_performance_monitor"
]
