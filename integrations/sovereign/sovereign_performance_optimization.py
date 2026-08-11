"""
Sovereign-Grade Problem Decomposition System - Performance Optimization
Implements caching, parallel processing, and performance monitoring.
"""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Callable
from functools import wraps, lru_cache
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import json

from sovereign_data_models import (
    ProblemDefinition, DecompositionPlan, SubProblem, Pattern, generate_id
)

logger = logging.getLogger(__name__)


class PerformanceCache:
    """High-performance caching layer for sovereign system."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize performance cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.logger = logging.getLogger(__name__)
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True)
        else:
            content = str(data)
        
        hash_obj = hashlib.md5(content.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key not in self.cache:
            self.miss_count += 1
            return None
        
        # Check TTL
        if key in self.access_times:
            age = (datetime.now() - self.access_times[key]).total_seconds()
            if age > self.ttl_seconds:
                self.invalidate(key)
                self.miss_count += 1
                return None
        
        self.hit_count += 1
        self.access_times[key] = datetime.now()
        return self.cache[key]['value']
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = {
            'value': value,
            'created_at': datetime.now()
        }
        self.access_times[key] = datetime.now()
    
    def invalidate(self, key: str) -> None:
        """Invalidate cache entry."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
    
    def invalidate_prefix(self, prefix: str) -> int:
        """Invalidate all entries with given prefix."""
        keys_to_remove = [k for k in self.cache.keys() if k.startswith(prefix)]
        for key in keys_to_remove:
            self.invalidate(key)
        return len(keys_to_remove)
    
    def clear(self) -> None:
        """Clear entire cache."""
        self.cache.clear()
        self.access_times.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times, key=self.access_times.get)
        self.invalidate(oldest_key)


# Global cache instance
_global_cache = PerformanceCache()


def cached(prefix: str, ttl_seconds: int = 3600):
    """
    Decorator for caching function results.
    
    Args:
        prefix: Cache key prefix
        ttl_seconds: Time-to-live in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from arguments
            cache_key_data = {
                'args': str(args),
                'kwargs': str(sorted(kwargs.items()))
            }
            cache_key = _global_cache._generate_key(prefix, cache_key_data)
            
            # Try to get from cache
            cached_result = _global_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {prefix}")
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            _global_cache.set(cache_key, result)
            logger.debug(f"Cache miss for {prefix}, stored result")
            
            return result
        
        return wrapper
    return decorator


class ParallelProcessor:
    """Parallel processing for independent operations."""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of parallel workers
        """
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
    
    async def process_subproblems_parallel(
        self,
        sub_problems: List[SubProblem],
        processor_func: Callable
    ) -> List[Any]:
        """
        Process sub-problems in parallel.
        
        Args:
            sub_problems: List of sub-problems to process
            processor_func: Function to process each sub-problem
            
        Returns:
            List of processing results
        """
        self.logger.info(f"Processing {len(sub_problems)} sub-problems in parallel")
        
        # Create tasks
        tasks = [
            asyncio.create_task(self._process_single(sp, processor_func))
            for sp in sub_problems
        ]
        
        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        self.logger.info(f"Completed {len(successful_results)}/{len(sub_problems)} successfully")
        
        return successful_results
    
    async def _process_single(self, sub_problem: SubProblem, processor_func: Callable) -> Any:
        """Process a single sub-problem."""
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, processor_func, sub_problem)
            return result
        except Exception as e:
            self.logger.error(f"Error processing sub-problem {sub_problem.id}: {e}")
            raise
    
    async def run_gauntlets_parallel(
        self,
        plan: DecompositionPlan,
        gauntlets: Dict[str, Callable]
    ) -> Dict[str, Any]:
        """
        Run multiple gauntlets in parallel.
        
        Args:
            plan: Decomposition plan to validate
            gauntlets: Dictionary of gauntlet name to gauntlet function
            
        Returns:
            Dictionary of gauntlet results
        """
        self.logger.info(f"Running {len(gauntlets)} gauntlets in parallel")
        
        # Create tasks for each gauntlet
        tasks = {
            name: asyncio.create_task(self._run_single_gauntlet(plan, gauntlet))
            for name, gauntlet in gauntlets.items()
        }
        
        # Wait for all gauntlets
        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                self.logger.error(f"Gauntlet {name} failed: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    async def _run_single_gauntlet(self, plan: DecompositionPlan, gauntlet: Callable) -> Any:
        """Run a single gauntlet."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, gauntlet, plan)


class PerformanceMonitor:
    """Monitor and track system performance."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.logger = logging.getLogger(__name__)
    
    def record_operation(self, operation_name: str, duration: float) -> None:
        """
        Record operation performance.
        
        Args:
            operation_name: Name of the operation
            duration: Duration in seconds
        """
        self.metrics[operation_name].append(duration)
        self.operation_counts[operation_name] += 1
    
    def get_stats(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        if operation_name not in self.metrics:
            return {
                'count': 0,
                'avg_duration': 0.0,
                'min_duration': 0.0,
                'max_duration': 0.0
            }
        
        durations = self.metrics[operation_name]
        
        return {
            'count': len(durations),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'total_duration': sum(durations)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {
            operation: self.get_stats(operation)
            for operation in self.metrics.keys()
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.operation_counts.clear()


# Global performance monitor
_global_monitor = PerformanceMonitor()


def timed(operation_name: str):
    """
    Decorator to time function execution.
    
    Args:
        operation_name: Name of the operation for tracking
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                _global_monitor.record_operation(operation_name, duration)
                logger.debug(f"{operation_name} took {duration:.3f}s")
        
        return wrapper
    return decorator


class LazyLoader:
    """Lazy loading for expensive operations."""
    
    def __init__(self):
        """Initialize lazy loader."""
        self._loaded: Dict[str, Any] = {}
        self._loaders: Dict[str, Callable] = {}
    
    def register(self, key: str, loader_func: Callable) -> None:
        """
        Register a lazy loader.
        
        Args:
            key: Unique key for the resource
            loader_func: Function to load the resource
        """
        self._loaders[key] = loader_func
    
    def get(self, key: str) -> Any:
        """
        Get resource, loading if necessary.
        
        Args:
            key: Resource key
            
        Returns:
            Loaded resource
        """
        if key in self._loaded:
            return self._loaded[key]
        
        if key not in self._loaders:
            raise KeyError(f"No loader registered for key: {key}")
        
        # Load resource
        logger.debug(f"Lazy loading resource: {key}")
        resource = self._loaders[key]()
        self._loaded[key] = resource
        
        return resource
    
    def is_loaded(self, key: str) -> bool:
        """Check if resource is loaded."""
        return key in self._loaded
    
    def unload(self, key: str) -> None:
        """Unload a resource."""
        if key in self._loaded:
            del self._loaded[key]
    
    def clear(self) -> None:
        """Clear all loaded resources."""
        self._loaded.clear()


class BatchProcessor:
    """Batch processing for database operations."""
    
    def __init__(self, batch_size: int = 100):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of items per batch
        """
        self.batch_size = batch_size
        self.pending_operations: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def add_operation(self, operation: Dict[str, Any]) -> None:
        """Add operation to batch."""
        self.pending_operations.append(operation)
        
        if len(self.pending_operations) >= self.batch_size:
            self.flush()
    
    def flush(self) -> int:
        """
        Execute all pending operations.
        
        Returns:
            Number of operations executed
        """
        if not self.pending_operations:
            return 0
        
        count = len(self.pending_operations)
        self.logger.info(f"Flushing {count} batched operations")
        
        # Execute operations (implementation would depend on database)
        # For now, just clear the queue
        self.pending_operations.clear()
        
        return count
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - flush on exit."""
        self.flush()


# Utility functions

def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    return _global_cache.get_stats()


def clear_cache() -> None:
    """Clear global cache."""
    _global_cache.clear()


def get_performance_stats() -> Dict[str, Dict[str, float]]:
    """Get global performance statistics."""
    return _global_monitor.get_all_stats()


def reset_performance_stats() -> None:
    """Reset global performance statistics."""
    _global_monitor.reset()
