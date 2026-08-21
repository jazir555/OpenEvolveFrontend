"""
Performance Optimization Module for OpenEvolve Frontend

This module provides comprehensive performance optimizations including:
1. Configuration caching to eliminate repeated config access
2. Performance monitoring decorators
3. LRU caching for expensive operations
4. Optimized data structure usage
5. Bulk operation patterns

Author: Performance Optimization Suite
Version: 1.0.0
"""
from __future__ import annotations


import time
import logging
from functools import wraps, lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib
import json

logger = logging.getLogger(__name__)


# ============================================================================
# Performance Monitoring
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics for function execution"""
    function_name: str
    execution_time: float
    timestamp: float
    args_hash: Optional[str] = None
    result_size: Optional[int] = None


class PerformanceMonitor:
    """Monitor and track performance metrics"""

    def __init__(self, enabled: bool = True, slow_threshold: float = 0.1):
        """
        Initialize performance monitor.

        Args:
            enabled: Whether monitoring is enabled
            slow_threshold: Threshold (seconds) for logging slow functions
        """
        self.enabled = enabled
        self.slow_threshold = slow_threshold
        self.metrics: List[PerformanceMetrics] = []
        self.function_stats: Dict[str, Dict[str, float]] = {}

    def record(self, func_name: str, execution_time: float, args_hash: str = None,
               result_size: int = None):
        """Record a performance metric"""
        if not self.enabled:
            return

        metric = PerformanceMetrics(
            function_name=func_name,
            execution_time=execution_time,
            timestamp=time.time(),
            args_hash=args_hash,
            result_size=result_size
        )
        self.metrics.append(metric)

        # Update statistics
        if func_name not in self.function_stats:
            self.function_stats[func_name] = {
                "count": 0,
                "total_time": 0.0,
                "max_time": 0.0,
                "min_time": float('inf')
            }

        stats = self.function_stats[func_name]
        stats["count"] += 1
        stats["total_time"] += execution_time
        stats["max_time"] = max(stats["max_time"], execution_time)
        stats["min_time"] = min(stats["min_time"], execution_time)

        # Log slow functions
        if execution_time > self.slow_threshold:
            logger.warning(
                f"Slow function detected: {func_name} took {execution_time:.3f}s "
                f"(threshold: {self.slow_threshold}s)"
            )

    def get_stats(self, func_name: str) -> Dict[str, float]:
        """Get statistics for a specific function"""
        if func_name not in self.function_stats:
            return {}

        stats = self.function_stats[func_name]
        return {
            "count": stats["count"],
            "total_time": stats["total_time"],
            "avg_time": stats["total_time"] / stats["count"],
            "max_time": stats["max_time"],
            "min_time": stats["min_time"] if stats["min_time"] != float('inf') else 0.0
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all monitored functions"""
        return {
            func_name: self.get_stats(func_name)
            for func_name in self.function_stats.keys()
        }

    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.function_stats.clear()


# Global performance monitor instance
_global_monitor = PerformanceMonitor()


def monitor_performance(
    func: Optional[Callable] = None,
    enabled: bool = True,
    slow_threshold: float = 0.1
) -> Callable:
    """
    Decorator to monitor function performance.

    Args:
        func: Function to monitor (if used as @monitor_performance)
        enabled: Whether monitoring is enabled
        slow_threshold: Threshold for logging slow functions

    Usage:
        @monitor_performance
        def my_function():
            pass

        or

        @monitor_performance(slow_threshold=0.5)
        def my_slow_function():
            pass
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not enabled:
                return f(*args, **kwargs)

            start_time = time.time()
            result = f(*args, **kwargs)
            elapsed_time = time.time() - start_time

            # Create args hash for caching insights
            args_str = str(args) + str(kwargs)
            args_hash = hashlib.md5(args_str.encode()).hexdigest()[:8]

            # Get result size if applicable
            result_size = None
            try:
                if hasattr(result, '__len__'):
                    result_size = len(result)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error in performance_optimizations.py: {e}", exc_info=True)
                raise

            _global_monitor.record(
                f.__name__,
                elapsed_time,
                args_hash=args_hash,
                result_size=result_size
            )

            return result

        return wrapper

    if func is not None:
        # Called as @monitor_performance
        return decorator(func)
    else:
        # Called as @monitor_performance(...)
        return decorator


def get_performance_stats() -> Dict[str, Dict[str, float]]:
    """Get all performance statistics"""
    return _global_monitor.get_all_stats()


def reset_performance_stats():
    """Reset all performance statistics"""
    _global_monitor.reset()


# ============================================================================
# Configuration Caching
# ============================================================================

class ConfigCache:
    """
    Cache for configuration objects to eliminate repeated attribute access.

    This is the SINGLE MOST IMPORTANT optimization for the codebase.
    It extracts all config values once and stores them in local variables.
    """

    def __init__(self, config_obj: Any):
        """
        Initialize cache from configuration object.

        Args:
            config_obj: Configuration dataclass or dict
        """
        self._config = config_obj
        self._cache: Dict[str, Any] = {}
        self._cached = False

    def extract(self, *attributes: str) -> 'ConfigCache':
        """
        Extract specific attributes into cache.

        Args:
            *attributes: Attribute names to extract

        Returns:
            Self for chaining

        Example:
            cache = ConfigCache(config).extract(
                'temperature', 'max_iterations', 'population_size'
            )
            temp = cache.temperature
            max_iter = cache.max_iterations
        """
        for attr in attributes:
            if hasattr(self._config, attr):
                self._cache[attr] = getattr(self._config, attr)
            elif isinstance(self._config, dict) and attr in self._config:
                self._cache[attr] = self._config[attr]
        return self

    def extract_all(self) -> 'ConfigCache':
        """Extract all attributes from config object"""
        if hasattr(self._config, '__dataclass_fields__'):
            # Dataclass object
            for field in self._config.__dataclass_fields__:
                self._cache[field] = getattr(self._config, field, None)
        elif isinstance(self._config, dict):
            # Dictionary
            self._cache.update(self._config)
        else:
            # Regular object - get all attributes
            for attr in dir(self._config):
                if not attr.startswith('_'):
                    try:
                        self._cache[attr] = getattr(self._config, attr)
                    except Exception as e:  # TODO: Catch specific exception instead of Exception
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error(f"Error in performance_optimizations.py: {e}", exc_info=True)
                        raise
        return self

    def __getattr__(self, name: str) -> Any:
        """Get cached attribute value"""
        if name in self._cache:
            return self._cache[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert cache to dictionary"""
        return self._cache.copy()

    def get(self, name: str, default: Any = None) -> Any:
        """Get cached attribute with default"""
        return self._cache.get(name, default)


def cache_config_loop(config_obj: Any, attributes: List[str]) -> Dict[str, Any]:
    """
    Extract config attributes for use in loops.

    This is the primary optimization pattern for eliminating
    repeated config access in loops.

    BEFORE (slow):
        for i in range(1000):
            temp = config.temperature  # 1000 attribute accesses!
            max_iter = config.max_iterations  # 1000 more!

    AFTER (fast):
        cached = cache_config_loop(config, ['temperature', 'max_iterations'])
        for i in range(1000):
            temp = cached['temperature']  # Dict lookup (fast)
            max_iter = cached['max_iterations']

    Args:
        config_obj: Configuration object
        attributes: List of attribute names to extract

    Returns:
        Dictionary with cached values
    """
    cache = ConfigCache(config_obj).extract(*attributes)
    return cache.to_dict()


# ============================================================================
# Result Caching
# ============================================================================

def cached_result(
    maxsize: int = 128,
    ttl: Optional[float] = None
) -> Callable:
    """
    Decorator for caching function results with optional TTL.

    Args:
        maxsize: Maximum cache size
        ttl: Time-to-live in seconds (None = no expiration)

    Usage:
        @cached_result(maxsize=256, ttl=300)
        def expensive_function(param):
            return complex_calculation(param)
    """
    cache = {}
    cache_timestamps = {}

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create cache key from arguments
            key = hashlib.md5(
                json.dumps([args, kwargs], sort_keys=True, default=str).encode()
            ).hexdigest()

            current_time = time.time()

            # Check cache
            if key in cache:
                if ttl is None or (current_time - cache_timestamps[key]) < ttl:
                    return cache[key]
                else:
                    # Expired
                    del cache[key]
                    del cache_timestamps[key]

            # Compute and cache
            result = func(*args, **kwargs)
            cache[key] = result
            cache_timestamps[key] = current_time

            # Enforce maxsize
            if len(cache) > maxsize:
                # Remove oldest entry
                oldest_key = min(cache_timestamps.keys(),
                               key=cache_timestamps.get)
                del cache[oldest_key]
                del cache_timestamps[oldest_key]

            return result

        return wrapper

    return decorator


# ============================================================================
# Bulk Operations
# ============================================================================

class BulkProcessor:
    """
    Processor for bulk operations to reduce individual call overhead.

    BEFORE (slow - N database calls):
        for item in items:
            db.save(item)

    AFTER (fast - 1 database call):
        processor = BulkProcessor(db.bulk_save, batch_size=100)
        processor.process_all(items)
    """

    def __init__(
        self,
        bulk_func: Callable,
        batch_size: int = 100,
        timeout: float = 30.0
    ):
        """
        Initialize bulk processor.

        Args:
            bulk_func: Function that processes a batch
            batch_size: Number of items per batch
            timeout: Timeout per batch in seconds
        """
        self.bulk_func = bulk_func
        self.batch_size = batch_size
        self.timeout = timeout

    @monitor_performance
    def process_all(self, items: List[Any]) -> List[Any]:
        """
        Process all items in batches.

        Args:
            items: Items to process

        Returns:
            Combined results from all batches
        """
        results = []

        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]

            start_time = time.time()
            batch_results = self.bulk_func(batch)

            if time.time() - start_time > self.timeout:
                logger.warning(f"Bulk processor timeout on batch {i // self.batch_size}")

            results.extend(batch_results if batch_results else [])

        return results

    def process_generator(self, items: List[Any]) -> Any:
        """
        Process items as a generator (yields batch results).

        Args:
            items: Items to process

        Yields:
            Results from each batch
        """
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            yield self.bulk_func(batch)


# ============================================================================
# Data Structure Optimizations
# ============================================================================

def optimize_for_membership(items: List[Any]) -> set:
    """
    Convert list to set for O(1) membership testing.

    BEFORE (slow - O(n) lookup):
        if item in large_list:
            pass

    AFTER (fast - O(1) lookup):
        large_set = optimize_for_membership(large_list)
        if item in large_set:
            pass

    Args:
        items: List to convert

    Returns:
        Set for fast membership testing
    """
    return set(items)


def optimize_for_lookup(data: List[Any], key_func: Callable) -> Dict[Any, Any]:
    """
    Convert list to dict for O(1) key-based lookup.

    BEFORE (slow - O(n) lookup):
        item = next(i for i in items if i.id == target_id)

    AFTER (fast - O(1) lookup):
        lookup = optimize_for_lookup(items, lambda x: x.id)
        item = lookup[target_id]

    Args:
        data: List of items
        key_func: Function to extract key from each item

    Returns:
        Dictionary mapping key to item
    """
    return {key_func(item): item for item in data}


# ============================================================================
# String Operations
# ============================================================================

def cached_string_operation(operation: str) -> Callable:
    """
    Cache expensive string operations like regex compilation.

    Args:
        operation: Regex pattern or operation string

    Returns:
        Cached operation function
    """
    import re

    compiled = re.compile(operation)

    def search(text: str) -> Optional[Any]:
        """Cached regex search"""
        return compiled.search(text)

    def match(text: str) -> Optional[Any]:
        """Cached regex match"""
        return compiled.match(text)

    def findall(text: str) -> List[Any]:
        """Cached regex findall"""
        return compiled.findall(text)

    return type('CachedRegex', (), {
        'search': search,
        'match': match,
        'findall': findall
    })()


# ============================================================================
# Benchmarking Utilities
# ============================================================================

@dataclass
class BenchmarkResult:
    """Result of benchmark comparison"""
    function_name: str
    old_time: float
    new_time: float
    improvement_percent: float
    speedup_factor: float


def benchmark(
    old_func: Callable,
    new_func: Callable,
    args: tuple = (),
    kwargs: Dict = None,
    iterations: int = 1000
) -> BenchmarkResult:
    """
    Benchmark two functions to measure improvement.

    Args:
        old_func: Original (slow) function
        new_func: Optimized (fast) function
        args: Arguments to pass to functions
        kwargs: Keyword arguments to pass to functions
        iterations: Number of iterations for benchmarking

    Returns:
        BenchmarkResult with comparison metrics
    """
    kwargs = kwargs or {}

    # Benchmark old function
    old_start = time.time()
    for _ in range(iterations):
        old_func(*args, **kwargs)
    old_time = time.time() - old_start

    # Benchmark new function
    new_start = time.time()
    for _ in range(iterations):
        new_func(*args, **kwargs)
    new_time = time.time() - new_start

    # Calculate improvement
    improvement_percent = ((old_time - new_time) / old_time) * 100
    speedup_factor = old_time / new_time

    return BenchmarkResult(
        function_name=old_func.__name__,
        old_time=old_time,
        new_time=new_time,
        improvement_percent=improvement_percent,
        speedup_factor=speedup_factor
    )


def print_benchmark_results(results: List[BenchmarkResult]):
    """Print benchmark results in a nice format"""
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)

    for result in results:
        print(f"\nFunction: {result.function_name}")
        print(f"  Old time: {result.old_time:.4f}s")
        print(f"  New time: {result.new_time:.4f}s")
        print(f"  Improvement: {result.improvement_percent:+.1f}%")
        print(f"  Speedup: {result.speedup_factor:.2f}x")

    print("\n" + "="*70)


# ============================================================================
# Main Entry Points
# ============================================================================

def enable_performance_monitoring(slow_threshold: float = 0.1):
    """Enable global performance monitoring"""
    global _global_monitor
    _global_monitor = PerformanceMonitor(enabled=True, slow_threshold=slow_threshold)


def disable_performance_monitoring():
    """Disable global performance monitoring"""
    global _global_monitor
    _global_monitor.enabled = False


def get_slow_functions(threshold: float = None) -> List[Tuple[str, float]]:
    """
    Get list of slow functions above threshold.

    Args:
        threshold: Minimum execution time (defaults to monitor threshold)

    Returns:
        List of (function_name, avg_time) tuples
    """
    threshold = threshold or _global_monitor.slow_threshold
    stats = _global_monitor.get_all_stats()

    slow_funcs = []
    for func_name, stats_dict in stats.items():
        avg_time = stats_dict.get('avg_time', 0)
        if avg_time > threshold:
            slow_funcs.append((func_name, avg_time))

    return sorted(slow_funcs, key=lambda x: x[1], reverse=True)


# Export key classes and functions
__all__ = [
    'monitor_performance',
    'ConfigCache',
    'cache_config_loop',
    'cached_result',
    'BulkProcessor',
    'optimize_for_membership',
    'optimize_for_lookup',
    'benchmark',
    'print_benchmark_results',
    'get_performance_stats',
    'reset_performance_stats',
    'get_slow_functions',
]
