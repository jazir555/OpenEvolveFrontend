"""
Performance Optimization Utilities

This module provides utilities for optimizing workflow performance including
parallel execution, batching, and resource management.
"""

import concurrent.futures
from typing import List, Callable, Any, Dict, Optional
import time
from functools import wraps
import threading


class ParallelExecutor:
    """Executes tasks in parallel using thread pools."""
    
    def __init__(self, max_workers: int = 5):
        """
        Initialize parallel executor.
        
        Args:
            max_workers: Maximum number of parallel workers
        """
        self.max_workers = max_workers
    
    def execute_parallel(
        self,
        func: Callable,
        items: List[Any],
        **kwargs
    ) -> List[Any]:
        """
        Execute function on items in parallel.
        
        Args:
            func: Function to execute
            items: List of items to process
            **kwargs: Additional arguments to pass to func
            
        Returns:
            List of results
        """
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(func, item, **kwargs): item
                for item in items
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing item: {e}")
                    results.append(None)
        
        return results
    
    def execute_parallel_with_progress(
        self,
        func: Callable,
        items: List[Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> List[Any]:
        """
        Execute function on items in parallel with progress tracking.
        
        Args:
            func: Function to execute
            items: List of items to process
            progress_callback: Callback function(completed, total)
            **kwargs: Additional arguments to pass to func
            
        Returns:
            List of results
        """
        results = [None] * len(items)
        completed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks with index
            future_to_index = {
                executor.submit(func, item, **kwargs): idx
                for idx, item in enumerate(items)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    print(f"Error processing item {idx}: {e}")
                    results[idx] = None
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(items))
        
        return results


class BatchProcessor:
    """Processes items in batches to optimize resource usage."""
    
    def __init__(self, batch_size: int = 10):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of items per batch
        """
        self.batch_size = batch_size
    
    def process_in_batches(
        self,
        func: Callable,
        items: List[Any],
        **kwargs
    ) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            func: Function to execute on each batch
            items: List of items to process
            **kwargs: Additional arguments to pass to func
            
        Returns:
            List of results
        """
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = func(batch, **kwargs)
            results.extend(batch_results)
        
        return results


def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.2f} seconds")
        return result
    return wrapper


def memoize(func):
    """Decorator to memoize function results."""
    cache = {}
    cache_lock = threading.Lock()
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from args and kwargs
        key = str(args) + str(sorted(kwargs.items()))
        
        with cache_lock:
            if key in cache:
                return cache[key]
        
        result = func(*args, **kwargs)
        
        with cache_lock:
            cache[key] = result
        
        return result
    
    return wrapper


class RateLimiter:
    """Rate limiter to control API call frequency."""
    
    def __init__(self, calls_per_second: float = 10.0):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_second: Maximum calls per second
        """
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limit."""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            
            if time_since_last_call < self.min_interval:
                sleep_time = self.min_interval - time_since_last_call
                time.sleep(sleep_time)
            
            self.last_call_time = time.time()
    
    def execute_with_rate_limit(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with rate limiting.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        self.wait_if_needed()
        return func(*args, **kwargs)


class PerformanceMonitor:
    """Monitors and reports performance metrics."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: Dict[str, List[float]] = {}
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float):
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Dictionary with min, max, avg, total
        """
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return {"min": 0, "max": 0, "avg": 0, "total": 0, "count": 0}
            
            values = self.metrics[name]
            return {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "total": sum(values),
                "count": len(values)
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {name: self.get_stats(name) for name in self.metrics.keys()}
    
    def clear(self):
        """Clear all metrics."""
        with self.lock:
            self.metrics = {}


# Global performance monitor
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def monitored_execution(metric_name: str):
    """Decorator to monitor execution time of a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            execution_time = end_time - start_time
            monitor = get_performance_monitor()
            monitor.record_metric(metric_name, execution_time)
            
            return result
        return wrapper
    return decorator


class OptimizationConfig:
    """Configuration for performance optimizations."""
    
    def __init__(
        self,
        enable_caching: bool = True,
        enable_parallelization: bool = True,
        max_parallel_workers: int = 5,
        batch_size: int = 10,
        rate_limit_calls_per_second: float = 10.0,
        cache_ttl_hours: int = 24
    ):
        """
        Initialize optimization configuration.
        
        Args:
            enable_caching: Enable LLM response caching
            enable_parallelization: Enable parallel execution
            max_parallel_workers: Maximum parallel workers
            batch_size: Batch size for batch processing
            rate_limit_calls_per_second: Rate limit for API calls
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.enable_caching = enable_caching
        self.enable_parallelization = enable_parallelization
        self.max_parallel_workers = max_parallel_workers
        self.batch_size = batch_size
        self.rate_limit_calls_per_second = rate_limit_calls_per_second
        self.cache_ttl_hours = cache_ttl_hours


# Default optimization configuration
DEFAULT_OPTIMIZATION_CONFIG = OptimizationConfig()
