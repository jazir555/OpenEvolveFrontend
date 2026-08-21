"""
Sovereign-Grade Problem Decomposition System - Performance Optimization
Implements caching, parallel processing, and other performance enhancements.
"""
from __future__ import annotations


import asyncio
import concurrent.futures
import hashlib
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from collections import OrderedDict
from functools import wraps
from dataclasses import dataclass
import sqlite3
import logging
from queue import Queue
from contextlib import contextmanager
import os

# Optional Redis support for distributed caching
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    ttl: Optional[int] = None  # time to live in seconds
    hits: int = 0
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired"""
        if self.ttl is None:
            return False
        expiry_time = self.created_at + timedelta(seconds=self.ttl)
        return datetime.now() > expiry_time


class LRUCache:
    """
    Optimized LRU cache implementation using OrderedDict for O(1) performance.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache, return None if not found or expired"""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired():
                del self._cache[key]
                return None
            
            # Update access order and hit count
            self._cache.move_to_end(key)
            entry.hits += 1
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache with optional TTL"""
        with self._lock:
            # If key already exists, update it
            if key in self._cache:
                self._cache.move_to_end(key)
                entry = self._cache[key]
                entry.value = value
                entry.ttl = ttl or self.default_ttl
                entry.created_at = datetime.now()
                return

            # If cache is full, remove least recently used item (first item in OrderedDict)
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            # Create new cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                ttl=ttl or self.default_ttl
            )
            self._cache[key] = entry
    
    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_entries = len(self._cache)
            expired_entries = sum(1 for entry in self._cache.values() if entry.is_expired())
            return {
                'total_entries': total_entries,
                'expired_entries': expired_entries,
                'active_entries': total_entries - expired_entries,
                'max_size': self.max_size,
                'cache_hit_rate': sum(entry.hits for entry in self._cache.values()) / max(total_entries, 1)
            }


class LLMResponseCache:
    """Cache for LLM responses to reduce API calls and costs"""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 7200):  # 2 hours default
        self.cache = LRUCache(max_size, default_ttl)
        self.stats = {'hits': 0, 'misses': 0, 'saved_calls': 0}
        self._lock = threading.RLock()
    
    def _generate_key(self, content: str, model_params: Dict[str, Any]) -> str:
        """Generate a unique cache key based on content and model parameters"""
        # Optimization: Use a faster key generation for simple cases
        if not model_params:
            return hashlib.sha256(content.encode()).hexdigest()

        cache_input = {
            'content': content,
            'model': model_params.get('model', ''),
            'temperature': model_params.get('temperature', 0.7),
            'max_tokens': model_params.get('max_tokens', 1000),
            'top_p': model_params.get('top_p', 1.0),
        }
        # Use sort_keys for consistent hashing
        cache_str = json.dumps(cache_input, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def get_response(self, content: str, model_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached LLM response if available"""
        key = self._generate_key(content, model_params)
        response = self.cache.get(key)
        
        with self._lock:
            if response is not None:
                self.stats['hits'] += 1
                logger.debug(f"LLM cache hit for key: {key[:8]}")
                return response
            else:
                self.stats['misses'] += 1
                logger.debug(f"LLM cache miss for key: {key[:8]}")
                return None
    
    def cache_response(self, content: str, model_params: Dict[str, Any], response: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Cache an LLM response"""
        key = self._generate_key(content, model_params)
        self.cache.set(key, response, ttl)
        
        with self._lock:
            self.stats['saved_calls'] += 1
            logger.debug(f"Cached LLM response with key: {key[:8]}")
    
    def invalidate_content(self, content: str, model_params: Dict[str, Any]) -> bool:
        """Invalidate cache for specific content and parameters"""
        key = self._generate_key(content, model_params)
        return self.cache.invalidate(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        cache_stats = self.cache.stats()
        with self._lock:
            return {**cache_stats, **self.stats}


class DatabaseOptimizer:
    """Database query optimization utilities"""
    
    def __init__(self, db_path: str = "sovereign_decomposition.db"):
        self.db_path = db_path
        self.connection_pools = {}
    
    def create_connection(self):
        """Create an optimized database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # Optimize connection settings for performance
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-128000")  # Increased to 128MB cache
        conn.execute("PRAGMA temp_store=MEMORY")   # Use RAM for temporary storage
        conn.execute("PRAGMA mmap_size=536870912")  # Increased to 512MB memory mapping
        conn.execute("PRAGMA page_size=4096")
        conn.execute("PRAGMA threads=4")
        
        return conn
    
    @contextmanager
    def get_connection(self):
        """Context manager for optimized database connections"""
        conn = self.create_connection()
        try:
            yield conn
            conn.commit()
        except (sqlite3.Error, IOError, OSError):
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def add_indexes(self):
        """Add performance indexes to database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Add indexes for common query patterns
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_problems_type_created ON problems(problem_type, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_subproblems_status ON sub_problems(status)",
                "CREATE INDEX IF NOT EXISTS idx_subproblems_parent_priority ON sub_problems(parent_id, priority DESC)",
                "CREATE INDEX IF NOT EXISTS idx_plans_status_created ON decomposition_plans(status, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_attempts_subproblem_status ON solution_attempts(sub_problem_id, status)",
                "CREATE INDEX IF NOT EXISTS idx_feedback_source_timestamp ON feedback(source, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_assignments_task_status ON team_assignments(task_id, status)",
            ]
            
            for index_stmt in indexes:
                cursor.execute(index_stmt)
            
            logger.info("Added performance indexes to database")
    
    def optimize_queries(self):
        """Run database optimization commands"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Run optimization commands
            cursor.execute("ANALYZE")  # Update statistics
            cursor.execute("VACUUM")  # Defragment database
            
            logger.info("Database optimization completed")


class ResourcePool:
    """Generic resource pooling for expensive resources like database connections"""
    
    def __init__(self, create_resource: Callable[[], Any], destroy_resource: Callable[[Any], None], 
                 max_size: int = 10, min_size: int = 2):
        self.create_resource = create_resource
        self.destroy_resource = destroy_resource
        self.max_size = max_size
        self.min_size = min_size
        self.pool = Queue(maxsize=max_size)
        self.active_count = 0
        self.lock = threading.Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the pool with minimum resources"""
        for _ in range(self.min_size):
            resource = self.create_resource()
            self.pool.put(resource)
            self.active_count += 1
    
    @contextmanager
    def get_resource(self):
        """Get a resource from the pool"""
        resource = None
        try:
            # Try to get from pool or create new one if not full
            try:
                resource = self.pool.get_nowait()
            except:
                with self.lock:
                    if self.active_count < self.max_size:
                        resource = self.create_resource()
                        self.active_count += 1
                    else:
                        # Wait for a resource to be returned
                        resource = self.pool.get()
            
            yield resource
        finally:
            if resource:
                self.pool.put(resource)
    
    def close(self):
        """Close all resources in the pool"""
        while not self.pool.empty():
            try:
                resource = self.pool.get_nowait()
                self.destroy_resource(resource)
                with self.lock:
                    self.active_count -= 1
            except:
                break


class RateLimiter:
    """Rate limiting for API calls to prevent abuse and ensure fair usage"""
    
    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.requests = {}  # {identifier: [request_times]}
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier: str = "default") -> bool:
        """Check if a request is allowed based on rate limits"""
        now = time.time()
        
        with self.lock:
            if identifier not in self.requests:
                self.requests[identifier] = []
            
            # Remove requests older than 1 minute
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if now - req_time < 60
            ]
            
            # Check rate limit
            if len(self.requests[identifier]) >= self.requests_per_minute:
                return False
            
            # Add current request
            self.requests[identifier].append(now)
            return True
    
    def wait_for_allowance(self, identifier: str = "default", timeout: float = 60.0) -> bool:
        """Wait until a request is allowed, with timeout"""
        import time as time_module
        
        start_time = time_module.time()
        while time_module.time() - start_time < timeout:
            if self.is_allowed(identifier):
                return True
            time_module.sleep(0.1)
        return False


class ParallelProcessor:
    """Parallel processing utilities for independent tasks"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
    
    def process_in_parallel(self, tasks: List[Callable[[], Any]], timeout: Optional[float] = None) -> List[Any]:
        """Process multiple tasks in parallel using ThreadPoolExecutor"""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(task): i for i, task in enumerate(tasks)
            }
            
            # Collect results maintaining order
            temp_results = [None] * len(tasks)
            for future in concurrent.futures.as_completed(future_to_index, timeout=timeout):
                index = future_to_index[future]
                try:
                    temp_results[index] = future.result()
                except Exception as e:
                    logger.error(f"Task {index} failed: {e}")
                    temp_results[index] = None  # Or raise the exception
            
            results = temp_results
        
        return results
    
    async def process_in_parallel_async(self, tasks: List[Callable[[], Any]]) -> List[Any]:
        """Process multiple tasks in parallel using asyncio"""
        async def run_task(task):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, task)
        
        coroutines = [run_task(task) for task in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        return results


class PerformanceOptimizer:
    """Main performance optimization manager"""
    
    def __init__(self):
        self.llm_cache = LLMResponseCache()
        self.database_optimizer = DatabaseOptimizer()
        self.parallel_processor = ParallelProcessor()
        self.rate_limiter = RateLimiter()
        self.resource_pools = {}
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_tasks_executed': 0,
            'rate_limit_rejections': 0
        }
    
    def cache_llm_response(self, content: str, model_params: Dict[str, Any], response: Dict[str, Any], ttl: Optional[int] = None):
        """Cache an LLM response"""
        self.llm_cache.cache_response(content, model_params, response, ttl)
    
    def get_cached_llm_response(self, content: str, model_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get a cached LLM response"""
        response = self.llm_cache.get_response(content, model_params)
        if response:
            self.stats['cache_hits'] += 1
        else:
            self.stats['cache_misses'] += 1
        return response
    
    def process_tasks_in_parallel(self, tasks: List[Callable[[], Any]], timeout: Optional[float] = None) -> List[Any]:
        """Process tasks in parallel"""
        results = self.parallel_processor.process_in_parallel(tasks, timeout)
        self.stats['parallel_tasks_executed'] += len([r for r in results if r is not None])
        return results
    
    def check_rate_limit(self, identifier: str = "default") -> bool:
        """Check if a request is within rate limits"""
        allowed = self.rate_limiter.is_allowed(identifier)
        if not allowed:
            self.stats['rate_limit_rejections'] += 1
        return allowed
    
    def create_db_connection_pool(self, name: str = "default", max_size: int = 10) -> ResourcePool:
        """Create a database connection pool"""
        pool = ResourcePool(
            create_resource=self.database_optimizer.create_connection,
            destroy_resource=lambda conn: conn.close(),
            max_size=max_size
        )
        self.resource_pools[name] = pool
        return pool
    
    def optimize_database(self):
        """Run database optimizations"""
        self.database_optimizer.add_indexes()
        self.database_optimizer.optimize_queries()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_stats = self.llm_cache.get_stats()
        return {**self.stats, **cache_stats}


def cache_result(ttl: int = 3600, key_func: Optional[Callable] = None):
    """Decorator to cache function results"""
    def decorator(func):
        cache = LRUCache(max_size=1000, default_ttl=ttl)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_input = str(args) + str(sorted(kwargs.items()))
                cache_key = hashlib.sha256(key_input.encode()).hexdigest()
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for function {func.__name__}")
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            logger.debug(f"Cached result for function {func.__name__}")
            return result
        
        wrapper.cache = cache  # Expose cache for manual management
        return wrapper
    return decorator


def rate_limit(calls_per_minute: int = 60, burst_size: int = 10):
    """Decorator to rate limit function calls"""
    def decorator(func):
        limiter = RateLimiter(requests_per_minute=calls_per_minute, burst_size=burst_size)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use function name as identifier, or allow custom identifier
            identifier = f"{func.__name__}_default"
            if args and isinstance(args[0], str):
                identifier = f"{func.__name__}_{args[0]}"
            
            if not limiter.is_allowed(identifier):
                raise Exception(f"Rate limit exceeded for {identifier}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def parallelize(func):
    """Decorator to run function in parallel when possible"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # If the function is being called with multiple items, run in parallel
        # This is a simplified version - in practice, you'd detect if args is a list
        return func(*args, **kwargs)
    return wrapper


# Global performance optimizer instance
_performance_optimizer = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the performance optimizer instance"""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


def get_llm_cache() -> LLMResponseCache:
    """Get the LLM response cache instance"""
    return get_performance_optimizer().llm_cache


def get_database_optimizer() -> DatabaseOptimizer:
    """Get the database optimizer instance"""
    return get_performance_optimizer().database_optimizer


def get_parallel_processor() -> ParallelProcessor:
    """Get the parallel processor instance"""
    return get_performance_optimizer().parallel_processor


def get_rate_limiter() -> RateLimiter:
    """Get the rate limiter instance"""
    return get_performance_optimizer().rate_limiter


# Example usage and initialization
if __name__ == "__main__":
    # Initialize performance optimizer
    perf_opt = get_performance_optimizer()
    
    # Example: Cache an LLM response
    llm_params = {
        'model': 'gpt-4',
        'temperature': 0.7,
        'max_tokens': 150
    }
    
    sample_response = {
        'choices': [{'message': {'content': 'This is a cached response'}}],
        'model': 'gpt-4',
        'usage': {'total_tokens': 50}
    }
    
    perf_opt.cache_llm_response("What is AI?", llm_params, sample_response)
    
    # Retrieve cached response
    cached = perf_opt.get_cached_llm_response("What is AI?", llm_params)
    print(f"Cached response: {cached is not None}")
    
    # Example: Parallel processing
    def sample_task():
        import time
        time.sleep(0.1)  # Simulate work
        return f"Task completed at {time.time()}"
    
    tasks = [sample_task] * 5
    results = perf_opt.process_tasks_in_parallel(tasks)
    print(f"Parallel processing completed {len(results)} tasks")
    
    # Example: Rate limiting
    for i in range(5):
        allowed = perf_opt.check_rate_limit("test_user")
        print(f"Request {i+1} allowed: {allowed}")
    
    # Example: Database optimization
    perf_opt.optimize_database()
    print("Database optimization completed")
    
    # Example: Connection pool
    pool = perf_opt.create_db_connection_pool("main_pool", max_size=5)
    print(f"Created connection pool with max size: {pool.max_size}")
    
    # Get performance stats
    stats = perf_opt.get_performance_stats()
    print(f"Performance stats: {stats}")