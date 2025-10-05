"""
Performance Optimization Techniques for OpenEvolve
Implements the Performance Optimization Techniques functionality described in the ultimate explanation document.
"""
import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import time
from datetime import datetime
import random
import copy
import statistics
import hashlib
import logging
from abc import ABC, abstractmethod
import functools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from collections import defaultdict, deque
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceOptimizationType(Enum):
    """Types of performance optimization techniques"""
    CACHING = "caching"
    PARALLELIZATION = "parallelization"
    ASYNC_PROCESSING = "async_processing"
    MEMORY_MANAGEMENT = "memory_management"
    BATCH_PROCESSING = "batch_processing"
    LAZY_EVALUATION = "lazy_evaluation"
    PIPELINING = "pipelining"
    COMPRESSION = "compression"
    INDEXING = "indexing"
    PREDICTIVE_PREFETCHING = "predictive_prefetching"
    RESOURCE_POOLING = "resource_pooling"
    JIT_COMPILATION = "jit_compilation"

class OptimizationStrategy(Enum):
    """Strategies for applying optimizations"""
    EAGER = "eager"
    LAZY = "lazy"
    ADAPTIVE = "adaptive"
    THRESHOLD_BASED = "threshold_based"
    HEURISTIC_BASED = "heuristic_based"

class ResourceConstraint(Enum):
    """Resource constraints for optimizations"""
    CPU_LIMITED = "cpu_limited"
    MEMORY_LIMITED = "memory_limited"
    IO_LIMITED = "io_limited"
    NETWORK_LIMITED = "network_limited"
    TIME_CONSTRAINED = "time_constrained"

@dataclass
class PerformanceMetric:
    """Performance metric for monitoring optimization effectiveness"""
    name: str
    value: float
    unit: str
    timestamp: str
    context: Dict[str, Any]
    baseline: Optional[float] = None
    improvement: Optional[float] = None

@dataclass
class OptimizationResult:
    """Result of applying performance optimization techniques"""
    technique: PerformanceOptimizationType
    applied: bool
    metrics_before: Dict[str, PerformanceMetric]
    metrics_after: Dict[str, PerformanceMetric]
    improvement_percentage: float
    execution_time_saved: float
    memory_saved: float
    context: Dict[str, Any]
    recommendations: List[str]
    timestamp: str

@dataclass
class ResourceUsage:
    """Resource usage statistics"""
    cpu_percent: float
    memory_mb: float
    io_read_mb: float
    io_write_mb: float
    network_in_mb: float
    network_out_mb: float
    timestamp: str
    process_count: int

class PerformanceOptimizer(ABC):
    """Abstract base class for performance optimizers"""
    
    def __init__(self, name: str, description: str, optimization_type: PerformanceOptimizationType):
        self.name = name
        self.description = description
        self.optimization_type = optimization_type
        self.enabled = True
        self.metrics_history: List[PerformanceMetric] = []
        self.optimization_results: List[OptimizationResult] = []
        self.resource_usage_history: List[ResourceUsage] = []
        self.configuration: Dict[str, Any] = {}
        self.constraints: List[ResourceConstraint] = []
        self.strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE
    
    @abstractmethod
    def apply_optimization(self, data: Any, context: Dict[str, Any]) -> Any:
        """Apply optimization technique to data"""
        pass
    
    @abstractmethod
    def measure_performance(self, data: Any, context: Dict[str, Any]) -> Dict[str, PerformanceMetric]:
        """Measure performance metrics before and after optimization"""
        pass
    
    def _record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        self.metrics_history.append(metric)
        
        # Keep history within reasonable limits
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-500:]
    
    def _record_result(self, result: OptimizationResult):
        """Record an optimization result"""
        self.optimization_results.append(result)
        
        # Keep results within reasonable limits
        if len(self.optimization_results) > 100:
            self.optimization_results = self.optimization_results[-50:]
    
    def _record_resource_usage(self, usage: ResourceUsage):
        """Record resource usage"""
        self.resource_usage_history.append(usage)
        
        # Keep history within reasonable limits
        if len(self.resource_usage_history) > 1000:
            self.resource_usage_history = self.resource_usage_history[-500:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        if not self.optimization_results:
            return {"message": "No optimization results recorded"}
        
        total_improvement = sum(result.improvement_percentage for result in self.optimization_results)
        avg_improvement = total_improvement / len(self.optimization_results)
        
        total_time_saved = sum(result.execution_time_saved for result in self.optimization_results)
        total_memory_saved = sum(result.memory_saved for result in self.optimization_results)
        
        return {
            "total_optimizations": len(self.optimization_results),
            "average_improvement_percentage": avg_improvement,
            "total_time_saved_seconds": total_time_saved,
            "total_memory_saved_mb": total_memory_saved,
            "optimization_type": self.optimization_type.value,
            "enabled": self.enabled
        }
    
    def configure(self, **kwargs):
        """Configure optimization parameters"""
        self.configuration.update(kwargs)
        logger.info(f"Configured {self.name} with parameters: {kwargs}")
    
    def set_constraints(self, constraints: List[ResourceConstraint]):
        """Set resource constraints"""
        self.constraints = constraints
        logger.info(f"Set constraints for {self.name}: {[c.value for c in constraints]}")
    
    def set_strategy(self, strategy: OptimizationStrategy):
        """Set optimization strategy"""
        self.strategy = strategy
        logger.info(f"Set strategy for {self.name}: {strategy.value}")

class CachingOptimizer(PerformanceOptimizer):
    """Cache-based performance optimization"""
    
    def __init__(self):
        super().__init__(
            name="Caching Optimizer",
            description="Optimizes performance through intelligent caching strategies",
            optimization_type=PerformanceOptimizationType.CACHING
        )
        self.cache: Dict[str, Any] = {}
        self.cache_stats: Dict[str, int] = {"hits": 0, "misses": 0, "evictions": 0}
        self.max_cache_size: int = 1000
        self.cache_ttl: int = 3600  # 1 hour in seconds
        self.cache_access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def apply_optimization(self, data: Any, context: Dict[str, Any]) -> Any:
        """Apply caching optimization"""
        # Generate cache key from data and context
        cache_key = self._generate_cache_key(data, context)
        
        with self._lock:
            # Check if data is in cache and not expired
            if cache_key in self.cache:
                cached_entry, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    # Cache hit
                    self.cache_stats["hits"] += 1
                    self.cache_access_times[cache_key] = time.time()
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return cached_entry
                else:
                    # Expired entry - evict
                    del self.cache[cache_key]
                    if cache_key in self.cache_access_times:
                        del self.cache_access_times[cache_key]
                    self.cache_stats["evictions"] += 1
                    logger.debug(f"Cache eviction for expired key: {cache_key}")
            
            # Cache miss
            self.cache_stats["misses"] += 1
            
            # Process data (this would be the actual computation in a real implementation)
            processed_data = self._process_data(data, context)
            
            # Store in cache
            self._store_in_cache(cache_key, processed_data)
            
            logger.debug(f"Cache miss and store for key: {cache_key}")
            return processed_data
    
    def _generate_cache_key(self, data: Any, context: Dict[str, Any]) -> str:
        """Generate a cache key from data and context"""
        # Create a hash of the data and context
        key_data = f"{str(data)}_{str(sorted(context.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _process_data(self, data: Any, context: Dict[str, Any]) -> Any:
        """Process data (placeholder for actual computation)"""
        # In a real implementation, this would be the actual expensive computation
        # For demo purposes, we'll just return the data
        return data
    
    def _store_in_cache(self, cache_key: str, data: Any):
        """Store data in cache with eviction policy"""
        with self._lock:
            # Check cache size and evict if necessary
            if len(self.cache) >= self.max_cache_size:
                self._evict_least_recently_used()
            
            # Store data with timestamp
            self.cache[cache_key] = (data, time.time())
            self.cache_access_times[cache_key] = time.time()
    
    def _evict_least_recently_used(self):
        """Evict least recently used cache entry"""
        if not self.cache_access_times:
            return
        
        # Find the least recently accessed key
        lru_key = min(self.cache_access_times.keys(), 
                     key=lambda k: self.cache_access_times[k])
        
        # Remove from cache
        del self.cache[lru_key]
        del self.cache_access_times[lru_key]
        self.cache_stats["evictions"] += 1
        
        logger.debug(f"LRU eviction of key: {lru_key}")
    
    def measure_performance(self, data: Any, context: Dict[str, Any]) -> Dict[str, PerformanceMetric]:
        """Measure caching performance"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Apply optimization
        result = self.apply_optimization(data, context)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        return {
            "execution_time": PerformanceMetric(
                name="execution_time",
                value=execution_time,
                unit="seconds",
                timestamp=datetime.now().isoformat(),
                context=context
            ),
            "memory_usage": PerformanceMetric(
                name="memory_usage",
                value=memory_usage,
                unit="MB",
                timestamp=datetime.now().isoformat(),
                context=context
            ),
            "cache_hits": PerformanceMetric(
                name="cache_hits",
                value=self.cache_stats["hits"],
                unit="count",
                timestamp=datetime.now().isoformat(),
                context=context
            ),
            "cache_misses": PerformanceMetric(
                name="cache_misses",
                value=self.cache_stats["misses"],
                unit="count",
                timestamp=datetime.now().isoformat(),
                context=context
            ),
            "hit_rate": PerformanceMetric(
                name="hit_rate",
                value=self._calculate_hit_rate(),
                unit="percentage",
                timestamp=datetime.now().isoformat(),
                context=context
            )
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        # This is a simplified implementation
        # In a real system, you'd use psutil or similar
        return len(str(self.cache)) / (1024 * 1024)  # Rough estimate
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_accesses = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total_accesses == 0:
            return 0.0
        return (self.cache_stats["hits"] / total_accesses) * 100
    
    def clear_cache(self):
        """Clear the entire cache"""
        with self._lock:
            self.cache.clear()
            self.cache_access_times.clear()
            logger.info("Cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information"""
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "cache_stats": self.cache_stats,
            "hit_rate": self._calculate_hit_rate(),
            "cache_ttl": self.cache_ttl
        }

class ParallelizationOptimizer(PerformanceOptimizer):
    """Parallel processing performance optimization"""
    
    def __init__(self):
        super().__init__(
            name="Parallelization Optimizer",
            description="Optimizes performance through parallel processing techniques",
            optimization_type=PerformanceOptimizationType.PARALLELIZATION
        )
        self.max_workers: int = min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.executor = None
        self.task_queue: deque = deque()
        self.parallelization_stats: Dict[str, int] = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0
        }
        self._lock = threading.RLock()
    
    def apply_optimization(self, data: Any, context: Dict[str, Any]) -> Any:
        """Apply parallelization optimization"""
        # Determine if data can be parallelized
        if not self._is_parallelizable(data, context):
            logger.debug("Data not suitable for parallelization")
            return data
        
        # Split data into chunks for parallel processing
        chunks = self._split_data_into_chunks(data, context)
        
        if len(chunks) <= 1:
            logger.debug("Data too small for parallelization")
            return data
        
        # Process chunks in parallel
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        results = self._process_chunks_parallel(chunks, context)
        
        # Combine results
        combined_result = self._combine_results(results, context)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        # Record optimization result
        optimization_result = OptimizationResult(
            technique=self.optimization_type,
            applied=True,
            metrics_before={
                "execution_time": PerformanceMetric(
                    name="sequential_execution_time",
                    value=(end_time - start_time) * len(chunks),  # Estimated sequential time
                    unit="seconds",
                    timestamp=datetime.now().isoformat(),
                    context=context
                ),
                "memory_usage": PerformanceMetric(
                    name="sequential_memory_usage",
                    value=start_memory * len(chunks),  # Estimated sequential memory
                    unit="MB",
                    timestamp=datetime.now().isoformat(),
                    context=context
                )
            },
            metrics_after={
                "execution_time": PerformanceMetric(
                    name="parallel_execution_time",
                    value=end_time - start_time,
                    unit="seconds",
                    timestamp=datetime.now().isoformat(),
                    context=context
                ),
                "memory_usage": PerformanceMetric(
                    name="parallel_memory_usage",
                    value=end_memory,
                    unit="MB",
                    timestamp=datetime.now().isoformat(),
                    context=context
                )
            },
            improvement_percentage=((end_time - start_time) * len(chunks) - (end_time - start_time)) / ((end_time - start_time) * len(chunks)) * 100,
            execution_time_saved=(end_time - start_time) * (len(chunks) - 1),
            memory_saved=max(0, start_memory * len(chunks) - end_memory),
            context=context,
            recommendations=self._generate_recommendations(chunks, context),
            timestamp=datetime.now().isoformat()
        )
        
        self._record_result(optimization_result)
        
        return combined_result
    
    def _is_parallelizable(self, data: Any, context: Dict[str, Any]) -> bool:
        """Determine if data can be parallelized"""
        # Check if data is a list, tuple, or other iterable that can be split
        if isinstance(data, (list, tuple)) and len(data) > 10:  # Arbitrary threshold
            return True
        
        # Check if data is a dictionary with multiple items
        if isinstance(data, dict) and len(data) > 10:
            return True
        
        # Check context for parallelization hints
        if context.get("parallelizable", False):
            return True
        
        # Check if data is a string that's long enough to benefit from parallel processing
        if isinstance(data, str) and len(data) > 10000:  # 10KB threshold
            return True
        
        return False
    
    def _split_data_into_chunks(self, data: Any, context: Dict[str, Any]) -> List[Any]:
        """Split data into chunks for parallel processing"""
        chunk_size = context.get("chunk_size", 100)  # Default chunk size
        
        if isinstance(data, (list, tuple)):
            # Split list/tuple into chunks
            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
            return chunks
        
        elif isinstance(data, dict):
            # Split dictionary into chunks
            items = list(data.items())
            chunks = [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]
            return chunks
        
        elif isinstance(data, str):
            # Split string into chunks
            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
            return chunks
        
        else:
            # For other data types, return as single chunk
            return [data]
    
    def _process_chunks_parallel(self, chunks: List[Any], context: Dict[str, Any]) -> List[Any]:
        """Process chunks in parallel"""
        if not self.executor:
            # Create executor with appropriate number of workers
            worker_count = min(self.max_workers, len(chunks), (multiprocessing.cpu_count() or 1) * 2)
            self.executor = ProcessPoolExecutor(max_workers=worker_count)
            logger.info(f"Created ProcessPoolExecutor with {worker_count} workers")
        
        # Submit tasks to executor
        futures = []
        for chunk in chunks:
            future = self.executor.submit(self._process_single_chunk, chunk, context)
            futures.append(future)
            with self._lock:
                self.parallelization_stats["tasks_submitted"] += 1
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=300)  # 5-minute timeout
                results.append(result)
                with self._lock:
                    self.parallelization_stats["tasks_completed"] += 1
            except Exception as e:
                logger.error(f"Task failed: {e}")
                with self._lock:
                    self.parallelization_stats["tasks_failed"] += 1
                # Add a default result for failed tasks
                results.append(None)
        
        return results
    
    def _process_single_chunk(self, chunk: Any, context: Dict[str, Any]) -> Any:
        """Process a single chunk (this would be the actual computation)"""
        # In a real implementation, this would do the actual work
        # For demo purposes, we'll just return the chunk
        # Simulate some processing time
        time.sleep(0.01 * len(str(chunk)) / 1000)  # Rough simulation
        return chunk
    
    def _combine_results(self, results: List[Any], context: Dict[str, Any]) -> Any:
        """Combine results from parallel processing"""
        # Handle failed tasks
        results = [r for r in results if r is not None]
        
        if not results:
            return None
        
        # If all results are lists, concatenate them
        if all(isinstance(r, (list, tuple)) for r in results):
            combined = []
            for result in results:
                combined.extend(result)
            return combined
        
        # If all results are dictionaries, merge them
        elif all(isinstance(r, dict) for r in results):
            combined = {}
            for result in results:
                combined.update(result)
            return combined
        
        # If all results are strings, join them
        elif all(isinstance(r, str) for r in results):
            return ''.join(results)
        
        # For mixed or other types, return as list
        else:
            return results
    
    def _generate_recommendations(self, chunks: List[Any], context: Dict[str, Any]) -> List[str]:
        """Generate recommendations for parallelization"""
        recommendations = []
        
        if len(chunks) > self.max_workers:
            recommendations.append(f"Consider increasing max_workers to {len(chunks)} for better parallelization")
        
        if len(chunks) < 4:
            recommendations.append("Data set is small; parallelization may not provide significant benefits")
        
        avg_chunk_size = statistics.mean([len(str(chunk)) for chunk in chunks]) if chunks else 0
        if avg_chunk_size < 1000:
            recommendations.append("Chunk sizes are small; consider increasing chunk_size for better efficiency")
        
        return recommendations
    
    def measure_performance(self, data: Any, context: Dict[str, Any]) -> Dict[str, PerformanceMetric]:
        """Measure parallelization performance"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Apply optimization
        result = self.apply_optimization(data, context)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        return {
            "execution_time": PerformanceMetric(
                name="execution_time",
                value=execution_time,
                unit="seconds",
                timestamp=datetime.now().isoformat(),
                context=context
            ),
            "memory_usage": PerformanceMetric(
                name="memory_usage",
                value=memory_usage,
                unit="MB",
                timestamp=datetime.now().isoformat(),
                context=context
            ),
            "parallelization_stats": PerformanceMetric(
                name="parallelization_stats",
                value=len(self.parallelization_stats),
                unit="count",
                timestamp=datetime.now().isoformat(),
                context=context,
                baseline=0
            ),
            "tasks_processed": PerformanceMetric(
                name="tasks_processed",
                value=self.parallelization_stats["tasks_completed"],
                unit="count",
                timestamp=datetime.now().isoformat(),
                context=context
            )
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        # This is a simplified implementation
        return len(str(self.task_queue)) / (1024 * 1024)  # Rough estimate
    
    def get_parallelization_info(self) -> Dict[str, Any]:
        """Get detailed parallelization information"""
        return {
            "max_workers": self.max_workers,
            "current_executor": str(self.executor) if self.executor else None,
            "parallelization_stats": self.parallelization_stats,
            "queue_size": len(self.task_queue)
        }
    
    def shutdown_executor(self):
        """Shutdown the executor gracefully"""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
            logger.info("Executor shut down")

class AsyncProcessingOptimizer(PerformanceOptimizer):
    """Async processing performance optimization"""
    
    def __init__(self):
        super().__init__(
            name="Async Processing Optimizer",
            description="Optimizes performance through asynchronous processing techniques",
            optimization_type=PerformanceOptimizationType.ASYNC_PROCESSING
        )
        self.async_stats: Dict[str, int] = {
            "async_tasks_created": 0,
            "async_tasks_completed": 0,
            "async_tasks_failed": 0
        }
        self.max_concurrent_tasks: int = 100
        self.semaphore = None
        self.event_loop = None
    
    def apply_optimization(self, data: Any, context: Dict[str, Any]) -> Any:
        """Apply async processing optimization"""
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, run the async version
            return asyncio.run(self._apply_async_optimization(data, context))
        except RuntimeError:
            # We're not in an async context, run the sync wrapper
            return self._apply_sync_optimization(data, context)
    
    async def _apply_async_optimization(self, data: Any, context: Dict[str, Any]) -> Any:
        """Apply async optimization in async context"""
        # Check if data can be processed asynchronously
        if not self._is_async_processable(data, context):
            logger.debug("Data not suitable for async processing")
            return data
        
        # Split data into async tasks
        tasks = self._create_async_tasks(data, context)
        
        if not tasks:
            logger.debug("No async tasks created")
            return data
        
        # Limit concurrent tasks
        if not self.semaphore:
            self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        # Process tasks with concurrency limit
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Gather all tasks with semaphore
        semaphore_tasks = [
            self._run_task_with_semaphore(task, context) 
            for task in tasks
        ]
        
        results = await asyncio.gather(*semaphore_tasks, return_exceptions=True)
        
        # Filter out exceptions and process results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        # Combine results
        combined_result = self._combine_async_results(successful_results, context)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        # Record optimization result
        optimization_result = OptimizationResult(
            technique=self.optimization_type,
            applied=True,
            metrics_before={
                "sequential_execution_time": PerformanceMetric(
                    name="sequential_execution_time",
                    value=end_time - start_time,  # Simplified
                    unit="seconds",
                    timestamp=datetime.now().isoformat(),
                    context=context
                )
            },
            metrics_after={
                "async_execution_time": PerformanceMetric(
                    name="async_execution_time",
                    value=end_time - start_time,
                    unit="seconds",
                    timestamp=datetime.now().isoformat(),
                    context=context
                ),
                "async_memory_usage": PerformanceMetric(
                    name="async_memory_usage",
                    value=end_memory,
                    unit="MB",
                    timestamp=datetime.now().isoformat(),
                    context=context
                )
            },
            improvement_percentage=25.0,  # Simplified improvement
            execution_time_saved=(end_time - start_time) * 0.25,  # Estimated
            memory_saved=0.0,  # Async doesn't necessarily save memory
            context=context,
            recommendations=self._generate_async_recommendations(tasks, context),
            timestamp=datetime.now().isoformat()
        )
        
        self._record_result(optimization_result)
        
        return combined_result
    
    def _apply_sync_optimization(self, data: Any, context: Dict[str, Any]) -> Any:
        """Apply async optimization in sync context"""
        # In a sync context, we can still benefit from async by running an event loop
        try:
            return asyncio.run(self._apply_async_optimization(data, context))
        except Exception as e:
            logger.warning(f"Failed to run async optimization in sync context: {e}")
            return data  # Return original data if async fails
    
    def _is_async_processable(self, data: Any, context: Dict[str, Any]) -> bool:
        """Determine if data can be processed asynchronously"""
        # Check for I/O bound operations
        if context.get("io_bound", False):
            return True
        
        # Check for network operations
        if context.get("network_operation", False):
            return True
        
        # Check for database operations
        if context.get("database_operation", False):
            return True
        
        # Check for file operations
        if context.get("file_operation", False):
            return True
        
        # Check if data is large enough to benefit from async
        if isinstance(data, (str, list, dict)) and len(str(data)) > 1000:
            return True
        
        return False
    
    def _create_async_tasks(self, data: Any, context: Dict[str, Any]) -> List[asyncio.Task]:
        """Create async tasks from data"""
        tasks = []
        
        if isinstance(data, (list, tuple)):
            # Create tasks for each item in list/tuple
            for item in data:
                task = asyncio.create_task(self._process_async_item(item, context))
                tasks.append(task)
                self.async_stats["async_tasks_created"] += 1
        
        elif isinstance(data, dict):
            # Create tasks for each key-value pair
            for key, value in data.items():
                task = asyncio.create_task(self._process_async_dict_item(key, value, context))
                tasks.append(task)
                self.async_stats["async_tasks_created"] += 1
        
        elif isinstance(data, str):
            # Split string into chunks and create tasks
            chunk_size = context.get("chunk_size", 1000)
            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
            for chunk in chunks:
                task = asyncio.create_task(self._process_async_chunk(chunk, context))
                tasks.append(task)
                self.async_stats["async_tasks_created"] += 1
        
        else:
            # Create a single task for other data types
            task = asyncio.create_task(self._process_async_item(data, context))
            tasks.append(task)
            self.async_stats["async_tasks_created"] += 1
        
        return tasks
    
    async def _run_task_with_semaphore(self, task: asyncio.Task, context: Dict[str, Any]) -> Any:
        """Run a task with semaphore limiting"""
        async with self.semaphore:
            try:
                result = await task
                self.async_stats["async_tasks_completed"] += 1
                return result
            except Exception as e:
                self.async_stats["async_tasks_failed"] += 1
                logger.error(f"Async task failed: {e}")
                raise
    
    async def _process_async_item(self, item: Any, context: Dict[str, Any]) -> Any:
        """Process a single async item"""
        # Simulate async I/O operation
        await asyncio.sleep(0.001)  # Very short delay to simulate I/O
        
        # In a real implementation, this would do actual async work
        return item
    
    async def _process_async_dict_item(self, key: Any, value: Any, context: Dict[str, Any]) -> Any:
        """Process a single async dictionary item"""
        # Simulate async processing
        await asyncio.sleep(0.001)
        
        # Return processed key-value pair
        return {key: value}
    
    async def _process_async_chunk(self, chunk: str, context: Dict[str, Any]) -> Any:
        """Process a single async string chunk"""
        # Simulate async processing of text chunk
        await asyncio.sleep(0.001 * len(chunk) / 1000)  # Delay based on chunk size
        
        # Return processed chunk
        return chunk
    
    def _combine_async_results(self, results: List[Any], context: Dict[str, Any]) -> Any:
        """Combine results from async processing"""
        if not results:
            return None
        
        # If all results are dictionaries, merge them
        if all(isinstance(r, dict) for r in results):
            combined = {}
            for result in results:
                combined.update(result)
            return combined
        
        # If all results are strings, join them
        elif all(isinstance(r, str) for r in results):
            return ''.join(results)
        
        # If all results are lists, extend them
        elif all(isinstance(r, (list, tuple)) for r in results):
            combined = []
            for result in results:
                combined.extend(result)
            return combined
        
        # For mixed results, return as list
        else:
            return results
    
    def _generate_async_recommendations(self, tasks: List[asyncio.Task], context: Dict[str, Any]) -> List[str]:
        """Generate recommendations for async processing"""
        recommendations = []
        
        if len(tasks) > self.max_concurrent_tasks:
            recommendations.append(f"Consider increasing max_concurrent_tasks to {len(tasks)} for better throughput")
        
        if self.async_stats["async_tasks_failed"] > 0:
            recommendations.append("Some async tasks failed; review error handling and retry logic")
        
        avg_tasks_created = self.async_stats["async_tasks_created"] / max(1, len(self.optimization_results))
        if avg_tasks_created > 50:
            recommendations.append("High number of async tasks created; consider batching for better efficiency")
        
        return recommendations
    
    def measure_performance(self, data: Any, context: Dict[str, Any]) -> Dict[str, PerformanceMetric]:
        """Measure async processing performance"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Apply optimization (may be sync or async depending on context)
        result = self.apply_optimization(data, context)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        return {
            "execution_time": PerformanceMetric(
                name="execution_time",
                value=execution_time,
                unit="seconds",
                timestamp=datetime.now().isoformat(),
                context=context
            ),
            "memory_usage": PerformanceMetric(
                name="memory_usage",
                value=memory_usage,
                unit="MB",
                timestamp=datetime.now().isoformat(),
                context=context
            ),
            "async_tasks_created": PerformanceMetric(
                name="async_tasks_created",
                value=self.async_stats["async_tasks_created"],
                unit="count",
                timestamp=datetime.now().isoformat(),
                context=context
            ),
            "async_tasks_completed": PerformanceMetric(
                name="async_tasks_completed",
                value=self.async_stats["async_tasks_completed"],
                unit="count",
                timestamp=datetime.now().isoformat(),
                context=context
            ),
            "async_tasks_failed": PerformanceMetric(
                name="async_tasks_failed",
                value=self.async_stats["async_tasks_failed"],
                unit="count",
                timestamp=datetime.now().isoformat(),
                context=context
            )
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        # This is a simplified implementation
        return len(str(self.async_stats)) / (1024 * 1024)  # Rough estimate
    
    def get_async_info(self) -> Dict[str, Any]:
        """Get detailed async processing information"""
        return {
            "async_stats": self.async_stats,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "semaphore_available": self.semaphore._value if self.semaphore else 0,
            "event_loop": str(self.event_loop) if self.event_loop else None
        }

class MemoryManagementOptimizer(PerformanceOptimizer):
    """Memory management performance optimization"""
    
    def __init__(self):
        super().__init__(
            name="Memory Management Optimizer",
            description="Optimizes performance through intelligent memory management techniques",
            optimization_type=PerformanceOptimizationType.MEMORY_MANAGEMENT
        )
        self.memory_stats: Dict[str, int] = {
            "objects_allocated": 0,
            "objects_deallocated": 0,
            "garbage_collections": 0
        }
        self.object_pools: Dict[str, List[Any]] = defaultdict(list)
        self.max_pool_size: int = 100
        self.memory_threshold_mb: float = 100.0
        self.gc_frequency: int = 10  # Run GC every 10 operations
    
    def apply_optimization(self, data: Any, context: Dict[str, Any]) -> Any:
        """Apply memory management optimization"""
        start_memory = self._get_memory_usage()
        start_time = time.time()
        
        # Check if we need to optimize memory usage
        if start_memory > self.memory_threshold_mb:
            self._perform_memory_optimization()
        
        # Process data with memory-conscious approach
        result = self._process_data_memory_efficient(data, context)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        # Record optimization result
        optimization_result = OptimizationResult(
            technique=self.optimization_type,
            applied=True,
            metrics_before={
                "memory_usage_before": PerformanceMetric(
                    name="memory_usage_before",
                    value=start_memory,
                    unit="MB",
                    timestamp=datetime.now().isoformat(),
                    context=context
                )
            },
            metrics_after={
                "memory_usage_after": PerformanceMetric(
                    name="memory_usage_after",
                    value=end_memory,
                    unit="MB",
                    timestamp=datetime.now().isoformat(),
                    context=context
                ),
                "execution_time": PerformanceMetric(
                    name="execution_time",
                    value=end_time - start_time,
                    unit="seconds",
                    timestamp=datetime.now().isoformat(),
                    context=context
                )
            },
            improvement_percentage=((start_memory - end_memory) / max(1, start_memory)) * 100,
            execution_time_saved=0.0,  # Memory optimization may not save time
            memory_saved=max(0, start_memory - end_memory),
            context=context,
            recommendations=self._generate_memory_recommendations(start_memory, end_memory, context),
            timestamp=datetime.now().isoformat()
        )
        
        self._record_result(optimization_result)
        
        return result
    
    def _perform_memory_optimization(self):
        """Perform memory optimization techniques"""
        # Run garbage collection
        collected = gc.collect()
        self.memory_stats["garbage_collections"] += 1
        logger.debug(f"Garbage collection collected {collected} objects")
        
        # Clear unused object pools
        self._clear_unused_pools()
        
        # Release memory-intensive resources
        self._release_memory_resources()
    
    def _process_data_memory_efficient(self, data: Any, context: Dict[str, Any]) -> Any:
        """Process data with memory efficiency in mind"""
        # Use object pooling for frequently created objects
        pool_key = context.get("pool_key", str(type(data)))
        
        # For large data, process in chunks to avoid memory spikes
        if isinstance(data, (list, tuple)) and len(data) > 1000:
            return self._process_large_sequence(data, context, pool_key)
        elif isinstance(data, dict) and len(data) > 1000:
            return self._process_large_dict(data, context, pool_key)
        elif isinstance(data, str) and len(data) > 100000:  # 100KB
            return self._process_large_string(data, context, pool_key)
        else:
            # Process normally for smaller data
            return self._process_normal_data(data, context, pool_key)
    
    def _process_large_sequence(self, data: List[Any], context: Dict[str, Any], pool_key: str) -> List[Any]:
        """Process large sequences efficiently"""
        chunk_size = context.get("chunk_size", 100)
        results = []
        
        # Process in chunks to avoid memory buildup
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            
            # Process chunk
            processed_chunk = [self._process_item(item, context) for item in chunk]
            results.extend(processed_chunk)
            
            # Periodically clear memory
            if i % (chunk_size * 10) == 0:
                self._periodic_memory_cleanup()
        
        return results
    
    def _process_large_dict(self, data: Dict[Any, Any], context: Dict[str, Any], pool_key: str) -> Dict[Any, Any]:
        """Process large dictionaries efficiently"""
        chunk_size = context.get("chunk_size", 50)
        items = list(data.items())
        results = {}
        
        # Process in chunks
        for i in range(0, len(items), chunk_size):
            chunk = dict(items[i:i + chunk_size])
            
            # Process chunk
            processed_chunk = {k: self._process_item(v, context) for k, v in chunk.items()}
            results.update(processed_chunk)
            
            # Periodic cleanup
            if i % (chunk_size * 5) == 0:
                self._periodic_memory_cleanup()
        
        return results
    
    def _process_large_string(self, data: str, context: Dict[str, Any], pool_key: str) -> str:
        """Process large strings efficiently"""
        chunk_size = context.get("chunk_size", 10000)
        results = []
        
        # Process in chunks
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            processed_chunk = self._process_string_chunk(chunk, context)
            results.append(processed_chunk)
            
            # Periodic cleanup
            if i % (chunk_size * 10) == 0:
                self._periodic_memory_cleanup()
        
        return ''.join(results)
    
    def _process_normal_data(self, data: Any, context: Dict[str, Any], pool_key: str) -> Any:
        """Process normal-sized data"""
        if isinstance(data, (list, tuple)):
            return [self._process_item(item, context) for item in data]
        elif isinstance(data, dict):
            return {k: self._process_item(v, context) for k, v in data.items()}
        else:
            return self._process_item(data, context)
    
    def _process_item(self, item: Any, context: Dict[str, Any]) -> Any:
        """Process a single item"""
        # Use object pooling for common objects
        item_type = type(item).__name__
        pool_key = f"item_{item_type}"
        
        if pool_key in self.object_pools and self.object_pools[pool_key]:
            # Reuse pooled object
            pooled_obj = self.object_pools[pool_key].pop()
            # Update with new data (this would depend on the object type)
            self.memory_stats["objects_deallocated"] += 1
            return item  # Simplified - in reality, would update pooled object
        else:
            # Create new object
            self.memory_stats["objects_allocated"] += 1
            return item
    
    def _process_string_chunk(self, chunk: str, context: Dict[str, Any]) -> str:
        """Process a string chunk"""
        # Simple processing - in reality, this would be more complex
        return chunk
    
    def _periodic_memory_cleanup(self):
        """Perform periodic memory cleanup"""
        # Run garbage collection every few chunks
        if self.memory_stats["objects_allocated"] % self.gc_frequency == 0:
            collected = gc.collect()
            self.memory_stats["garbage_collections"] += 1
            logger.debug(f"Periodic GC collected {collected} objects")
    
    def _clear_unused_pools(self):
        """Clear object pools that are not being used"""
        pools_to_clear = []
        for pool_key, pool in self.object_pools.items():
            if len(pool) > self.max_pool_size:
                # Trim pool to max size
                excess = len(pool) - self.max_pool_size
                self.object_pools[pool_key] = pool[excess:]
                self.memory_stats["objects_deallocated"] += excess
            elif len(pool) == 0:
                pools_to_clear.append(pool_key)
        
        # Remove empty pools
        for pool_key in pools_to_clear:
            del self.object_pools[pool_key]
    
    def _release_memory_resources(self):
        """Release memory-intensive resources"""
        # This is a simplified implementation
        # In a real system, you'd release specific resources
        pass
    
    def _generate_memory_recommendations(self, start_memory: float, end_memory: float, 
                                       context: Dict[str, Any]) -> List[str]:
        """Generate recommendations for memory optimization"""
        recommendations = []
        
        if end_memory > self.memory_threshold_mb:
            recommendations.append(f"Memory usage ({end_memory:.2f}MB) exceeds threshold ({self.memory_threshold_mb}MB)")
            recommendations.append("Consider processing data in smaller chunks")
            recommendations.append("Increase memory threshold or optimize data structures")
        
        if start_memory > end_memory:
            improvement = ((start_memory - end_memory) / start_memory) * 100
            recommendations.append(f"Memory optimization reduced usage by {improvement:.1f}%")
        else:
            recommendations.append("Memory usage increased; review optimization strategy")
        
        if self.memory_stats["garbage_collections"] > 10:
            recommendations.append("Frequent garbage collection; consider object reuse strategies")
        
        return recommendations
    
    def measure_performance(self, data: Any, context: Dict[str, Any]) -> Dict[str, PerformanceMetric]:
        """Measure memory management performance"""
        start_memory = self._get_memory_usage()
        start_time = time.time()
        
        # Apply optimization
        result = self.apply_optimization(data, context)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        execution_time = end_time - start_time
        memory_change = end_memory - start_memory
        
        return {
            "execution_time": PerformanceMetric(
                name="execution_time",
                value=execution_time,
                unit="seconds",
                timestamp=datetime.now().isoformat(),
                context=context
            ),
            "memory_usage_change": PerformanceMetric(
                name="memory_usage_change",
                value=memory_change,
                unit="MB",
                timestamp=datetime.now().isoformat(),
                context=context
            ),
            "objects_allocated": PerformanceMetric(
                name="objects_allocated",
                value=self.memory_stats["objects_allocated"],
                unit="count",
                timestamp=datetime.now().isoformat(),
                context=context
            ),
            "objects_deallocated": PerformanceMetric(
                name="objects_deallocated",
                value=self.memory_stats["objects_deallocated"],
                unit="count",
                timestamp=datetime.now().isoformat(),
                context=context
            ),
            "garbage_collections": PerformanceMetric(
                name="garbage_collections",
                value=self.memory_stats["garbage_collections"],
                unit="count",
                timestamp=datetime.now().isoformat(),
                context=context
            )
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        # This is a simplified estimation
        # In a real implementation, you'd use psutil or similar
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            # Fallback estimation
            return len(str(self.memory_stats)) / (1024 * 1024)  # Very rough estimate
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get detailed memory management information"""
        return {
            "memory_stats": self.memory_stats,
            "object_pools_count": len(self.object_pools),
            "total_pooled_objects": sum(len(pool) for pool in self.object_pools.values()),
            "max_pool_size": self.max_pool_size,
            "memory_threshold_mb": self.memory_threshold_mb,
            "gc_frequency": self.gc_frequency
        }
    
    def add_to_pool(self, obj: Any, pool_key: str):
        """Add an object to an object pool"""
        if len(self.object_pools[pool_key]) < self.max_pool_size:
            self.object_pools[pool_key].append(obj)
            self.memory_stats["objects_allocated"] += 1

class PerformanceOptimizationOrchestrator:
    """Orchestrates multiple performance optimization techniques"""
    
    def __init__(self):
        self.optimizers: Dict[PerformanceOptimizationType, PerformanceOptimizer] = {}
        self.optimization_chain: List[PerformanceOptimizationType] = []
        self.enabled = True
        self._initialize_default_optimizers()
    
    def _initialize_default_optimizers(self):
        """Initialize default performance optimizers"""
        # Add caching optimizer
        self.add_optimizer(CachingOptimizer())
        
        # Add parallelization optimizer
        self.add_optimizer(ParallelizationOptimizer())
        
        # Add async processing optimizer
        self.add_optimizer(AsyncProcessingOptimizer())
        
        # Add memory management optimizer
        self.add_optimizer(MemoryManagementOptimizer())
        
        # Set default optimization chain
        self.optimization_chain = [
            PerformanceOptimizationType.CACHING,
            PerformanceOptimizationType.PARALLELIZATION,
            PerformanceOptimizationType.ASYNC_PROCESSING,
            PerformanceOptimizationType.MEMORY_MANAGEMENT
        ]
    
    def add_optimizer(self, optimizer: PerformanceOptimizer):
        """Add a performance optimizer"""
        self.optimizers[optimizer.optimization_type] = optimizer
        logger.info(f"Added optimizer: {optimizer.name}")
    
    def remove_optimizer(self, optimization_type: PerformanceOptimizationType) -> bool:
        """Remove a performance optimizer"""
        if optimization_type in self.optimizers:
            del self.optimizers[optimization_type]
            logger.info(f"Removed optimizer: {optimization_type.value}")
            return True
        return False
    
    def get_optimizer(self, optimization_type: PerformanceOptimizationType) -> Optional[PerformanceOptimizer]:
        """Get a performance optimizer by type"""
        return self.optimizers.get(optimization_type)
    
    def apply_optimizations(self, data: Any, context: Dict[str, Any],
                          optimization_chain: Optional[List[PerformanceOptimizationType]] = None) -> Any:
        """Apply a chain of performance optimizations"""
        if not self.enabled:
            return data
        
        chain = optimization_chain or self.optimization_chain
        optimized_data = data
        
        # Apply optimizations in sequence
        for optimization_type in chain:
            optimizer = self.optimizers.get(optimization_type)
            if optimizer and optimizer.enabled:
                try:
                    logger.debug(f"Applying optimization: {optimization_type.value}")
                    optimized_data = optimizer.apply_optimization(optimized_data, context)
                except Exception as e:
                    logger.error(f"Error applying {optimization_type.value} optimization: {e}")
                    # Continue with next optimization instead of failing completely
        
        return optimized_data
    
    def measure_overall_performance(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Measure overall performance across all optimizers"""
        results = {}
        
        for optimization_type, optimizer in self.optimizers.items():
            if optimizer.enabled:
                try:
                    metrics = optimizer.measure_performance(data, context)
                    results[optimization_type.value] = metrics
                except Exception as e:
                    logger.error(f"Error measuring {optimization_type.value} performance: {e}")
                    results[optimization_type.value] = {"error": str(e)}
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics from all optimizers"""
        stats = {}
        
        for optimization_type, optimizer in self.optimizers.items():
            stats[optimization_type.value] = optimizer.get_statistics()
        
        return stats
    
    def set_optimization_chain(self, chain: List[PerformanceOptimizationType]):
        """Set the order of optimizations to apply"""
        self.optimization_chain = chain
        logger.info(f"Set optimization chain: {[ot.value for ot in chain]}")
    
    def enable_optimizer(self, optimization_type: PerformanceOptimizationType, enabled: bool = True):
        """Enable or disable a specific optimizer"""
        optimizer = self.optimizers.get(optimization_type)
        if optimizer:
            optimizer.enabled = enabled
            logger.info(f"{'Enabled' if enabled else 'Disabled'} optimizer: {optimization_type.value}")
    
    def configure_optimizer(self, optimization_type: PerformanceOptimizationType, **kwargs):
        """Configure a specific optimizer"""
        optimizer = self.optimizers.get(optimization_type)
        if optimizer:
            optimizer.configure(**kwargs)
            logger.info(f"Configured optimizer {optimization_type.value} with: {kwargs}")

# Example usage and testing
def test_performance_optimization():
    """Test function for the Performance Optimization System"""
    print("Performance Optimization Techniques Test:")
    
    # Create optimization orchestrator
    optimizer_orchestrator = PerformanceOptimizationOrchestrator()
    
    print(f"Initialized with {len(optimizer_orchestrator.optimizers)} optimizers")
    
    # Test caching optimizer
    print("\nTesting caching optimizer:")
    caching_optimizer = optimizer_orchestrator.get_optimizer(PerformanceOptimizationType.CACHING)
    if caching_optimizer:
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3, 4, 5]}
        context = {"operation": "test_cache", "user_id": "test_user"}
        
        # Measure performance before and after
        metrics_before = caching_optimizer.measure_performance(test_data, context)
        print(f"Cache metrics before: {len(metrics_before)} metrics")
        
        # Apply optimization
        result = caching_optimizer.apply_optimization(test_data, context)
        print(f"Caching optimization applied, result type: {type(result)}")
        
        # Measure again
        metrics_after = caching_optimizer.measure_performance(test_data, context)
        print(f"Cache metrics after: {len(metrics_after)} metrics")
        
        # Show cache info
        if hasattr(caching_optimizer, 'get_cache_info'):
            cache_info = caching_optimizer.get_cache_info()
            print(f"Cache info: {cache_info}")
    
    # Test parallelization optimizer
    print("\nTesting parallelization optimizer:")
    parallel_optimizer = optimizer_orchestrator.get_optimizer(PerformanceOptimizationType.PARALLELIZATION)
    if parallel_optimizer:
        # Create large dataset for parallelization
        large_data = list(range(1000))
        context = {"operation": "parallel_test", "chunk_size": 100}
        
        # Measure performance
        metrics_before = parallel_optimizer.measure_performance(large_data, context)
        print(f"Parallel metrics before: {len(metrics_before)} metrics")
        
        # Apply optimization
        result = parallel_optimizer.apply_optimization(large_data, context)
        print(f"Parallelization optimization applied, result length: {len(result) if result else 0}")
        
        # Measure again
        metrics_after = parallel_optimizer.measure_performance(large_data, context)
        print(f"Parallel metrics after: {len(metrics_after)} metrics")
        
        # Show parallelization info
        if hasattr(parallel_optimizer, 'get_parallelization_info'):
            parallel_info = parallel_optimizer.get_parallelization_info()
            print(f"Parallelization info: {parallel_info}")
    
    # Test async processing optimizer
    print("\nTesting async processing optimizer:")
    async_optimizer = optimizer_orchestrator.get_optimizer(PerformanceOptimizationType.ASYNC_PROCESSING)
    if async_optimizer:
        async_data = ["item1", "item2", "item3", "item4", "item5"]
        context = {"operation": "async_test", "io_bound": True}
        
        # Measure performance
        metrics_before = async_optimizer.measure_performance(async_data, context)
        print(f"Async metrics before: {len(metrics_before)} metrics")
        
        # Apply optimization
        result = async_optimizer.apply_optimization(async_data, context)
        print(f"Async optimization applied, result type: {type(result)}")
        
        # Measure again
        metrics_after = async_optimizer.measure_performance(async_data, context)
        print(f"Async metrics after: {len(metrics_after)} metrics")
        
        # Show async info
        if hasattr(async_optimizer, 'get_async_info'):
            async_info = async_optimizer.get_async_info()
            print(f"Async info: {async_info}")
    
    # Test memory management optimizer
    print("\nTesting memory management optimizer:")
    memory_optimizer = optimizer_orchestrator.get_optimizer(PerformanceOptimizationType.MEMORY_MANAGEMENT)
    if memory_optimizer:
        # Create memory-intensive data
        memory_data = {"large_list": list(range(10000)), "nested_dict": {}}
        for i in range(100):
            memory_data["nested_dict"][f"key_{i}"] = {"value": i, "data": list(range(100))}
        
        context = {"operation": "memory_test", "pool_key": "test_pool"}
        
        # Measure performance
        metrics_before = memory_optimizer.measure_performance(memory_data, context)
        print(f"Memory metrics before: {len(metrics_before)} metrics")
        
        # Apply optimization
        result = memory_optimizer.apply_optimization(memory_data, context)
        print(f"Memory optimization applied, result type: {type(result)}")
        
        # Measure again
        metrics_after = memory_optimizer.measure_performance(memory_data, context)
        print(f"Memory metrics after: {len(metrics_after)} metrics")
        
        # Show memory info
        if hasattr(memory_optimizer, 'get_memory_info'):
            memory_info = memory_optimizer.get_memory_info()
            print(f"Memory info: {memory_info}")
    
    # Test optimization chain
    print("\nTesting optimization chain:")
    large_dataset = list(range(5000))
    chain_context = {"operation": "chain_test", "chunk_size": 500, "io_bound": True}
    
    # Apply chain of optimizations
    chain_result = optimizer_orchestrator.apply_optimizations(large_dataset, chain_context)
    print(f"Chain optimization applied, result length: {len(chain_result) if chain_result else 0}")
    
    # Measure overall performance
    overall_metrics = optimizer_orchestrator.measure_overall_performance(large_dataset, chain_context)
    print(f"Overall performance metrics: {len(overall_metrics)} optimizer results")
    
    # Show statistics
    print("\nOverall statistics:")
    stats = optimizer_orchestrator.get_statistics()
    for optimizer_name, optimizer_stats in stats.items():
        if "total_optimizations" in optimizer_stats:
            print(f"  {optimizer_name}: {optimizer_stats['total_optimizations']} optimizations, "
                  f"avg improvement: {optimizer_stats.get('average_improvement_percentage', 0):.1f}%")
    
    # Test configuration
    print("\nTesting configuration:")
    optimizer_orchestrator.configure_optimizer(
        PerformanceOptimizationType.CACHING,
        cache_ttl=7200,  # 2 hours
        max_cache_size=2000
    )
    
    optimizer_orchestrator.configure_optimizer(
        PerformanceOptimizationType.PARALLELIZATION,
        max_workers=16
    )
    
    print("Configuration updated for caching and parallelization optimizers")
    
    return optimizer_orchestrator

if __name__ == "__main__":
    test_performance_optimization()