"""
Sovereign-Grade Problem Decomposition System - LLM Response Caching
Implements intelligent caching for expensive LLM calls to reduce costs and improve performance.
"""

import hashlib
import json
import time
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Callable, Tuple
from dataclasses import dataclass
import logging
from threading import Lock
from collections import OrderedDict
import sqlite3
import os


@dataclass(slots=True)
class CacheEntry:
    """Data class for cache entries."""
    key: str
    value: Any
    timestamp: datetime
    ttl: int  # Time-to-live in seconds
    hit_count: int = 1
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl


class LRUCache:
    """
    In-memory Least Recently Used cache implementation.
    Optimized with OrderedDict for O(1) performance.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries to store
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = Lock()
        self.logger = logging.getLogger(__name__)
        self._last_clean_time = 0
        self._clean_interval = 300  # Clean every 5 minutes
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            if entry.is_expired():
                del self.cache[key]
                return None
            
            # Update access order and hit count
            self.cache.move_to_end(key)
            entry.hit_count += 1
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (default 1 hour)
            
        Returns:
            True if set successfully
        """
        with self.lock:
            # If key already exists, update its position and value
            if key in self.cache:
                self.cache.move_to_end(key)
                self.cache[key] = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=datetime.now(),
                    ttl=ttl,
                    hit_count=self.cache[key].hit_count
                )
                return True

            # Clean expired entries periodically to avoid O(n) scan on every set
            current_time = time.time()
            if current_time - self._last_clean_time > self._clean_interval:
                self._clean_expired()
                self._last_clean_time = current_time
            
            # Remove oldest entry if at max size
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            
            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=datetime.now(),
                ttl=ttl
            )
            self.cache[key] = entry
            
            return True
    
    def _clean_expired(self):
        """Clean expired entries from cache."""
        # Note: OrderedDict iteration is O(n), but we only do it on set()
        # and only if needed. A more aggressive optimization could use a
        # separate TTL-sorted structure, but for simplicity we keep it here.
        expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]
        for key in expired_keys:
            del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_entries = len(self.cache)
            expired_entries = sum(1 for entry in self.cache.values() if entry.is_expired())
            total_hits = sum(entry.hit_count for entry in self.cache.values())
            
            return {
                'total_entries': total_entries,
                'expired_entries': expired_entries,
                'total_hits': total_hits,
                'cache_size': len(self.cache)
            }


class DatabaseCache:
    """Database-based cache implementation for persistence."""
    
    def __init__(self, db_path: str = "llm_cache.db"):
        """
        Initialize database cache.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_db()
        self.lock = Lock()
        self._hit_count_buffer = {}  # key -> count
        self._buffer_lock = Lock()
        self._last_flush_time = time.time()
        self._flush_interval = 60  # Flush hit counts every minute
    
    def _init_db(self):
        """Initialize database tables and optimize settings."""
        with sqlite3.connect(self.db_path) as conn:
            # Performance PRAGMAs for high-concurrency and throughput
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA busy_timeout=5000")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    timestamp TEXT,
                    ttl INTEGER,
                    hit_count INTEGER DEFAULT 1
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON llm_cache(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ttl ON llm_cache(ttl)")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from database cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        should_flush = False
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value, timestamp, ttl FROM llm_cache WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            
            if row:
                value, timestamp_str, ttl = row
                timestamp = datetime.fromisoformat(timestamp_str)
                
                if self._is_expired(timestamp, ttl):
                    # Remove expired entry
                    conn.execute("DELETE FROM llm_cache WHERE key = ?", (key,))
                    return None
                
                # Buffer hit count update to reduce I/O
                with self._buffer_lock:
                    self._hit_count_buffer[key] = self._hit_count_buffer.get(key, 0) + 1
                    if time.time() - self._last_flush_time > self._flush_interval:
                        should_flush = True
                
                # Deserialize value - Optimized fast path for JSON
                try:
                    # Optimized fast-path: check for JSON markers first
                    # Avoids unnecessary imports and nested try-except in hot paths
                    deserialized_value = None
                    if value.startswith(b'{') or value.startswith(b'['):
                        try:
                            import json
                            deserialized_value = json.loads(value.decode('utf-8'))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass

                    if deserialized_value is None:
                        # Fallback for complex/legacy data
                        import ast
                        try:
                            deserialized_value = ast.literal_eval(value.decode('utf-8'))
                        except (ValueError, SyntaxError, UnicodeDecodeError):
                            deserialized_value = pickle.loads(value)

                    # Periodic flush of hit counts - outside the main connection block to minimize contention
                    if should_flush:
                        self._flush_hit_counts()

                    return deserialized_value
                except Exception as e:
                    self.logger.error(f"Failed to deserialize cached value: {e}")
                    # Remove corrupted entry
                    conn.execute("DELETE FROM llm_cache WHERE key = ?", (key,))
                    return None
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Set value in database cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            
        Returns:
            True if set successfully
        """
        try:
            # Performance optimization: Use JSON for standard types, fallback to pickle
            try:
                serialized_value = json.dumps(value).encode('utf-8')
            except (TypeError, ValueError, OverflowError):
                # Fallback to pickle for complex objects
                serialized_value = pickle.dumps(value)
        except Exception as e:
            self.logger.error(f"Failed to serialize value for caching: {e}")
            return False
        
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO llm_cache 
                    (key, value, timestamp, ttl, hit_count)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    key, 
                    serialized_value,
                    datetime.now().isoformat(),
                    ttl,
                    1  # New entry starts with hit_count of 1
                ))
                conn.commit()
                return True
            except Exception as e:
                self.logger.error(f"Failed to cache value: {e}")
                return False
    
    def _flush_hit_counts(self):
        """Flush buffered hit counts to the database in a single transaction."""
        with self._buffer_lock:
            if not self._hit_count_buffer:
                return
            current_buffer = self._hit_count_buffer
            self._hit_count_buffer = {}
            self._last_flush_time = time.time()

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Optimized batch update
                conn.executemany(
                    "UPDATE llm_cache SET hit_count = hit_count + ? WHERE key = ?",
                    [(count, key) for key, count in current_buffer.items()]
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to flush hit counts: {e}")

    def _is_expired(self, timestamp: datetime, ttl: int) -> bool:
        """Check if cache entry is expired."""
        return (datetime.now() - timestamp).total_seconds() > ttl
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from database.
        
        Returns:
            Number of entries removed
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM llm_cache WHERE ? - timestamp > ttl",
                (int(datetime.now().timestamp()),)
            )
            removed_count = cursor.rowcount
            conn.commit()
            return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Count total entries
            cursor = conn.execute("SELECT COUNT(*) FROM llm_cache")
            total_entries = cursor.fetchone()[0]
            
            # Count expired entries
            cursor = conn.execute(
                "SELECT COUNT(*) FROM llm_cache WHERE ? - timestamp > ttl",
                (int(datetime.now().timestamp()),)
            )
            expired_entries = cursor.fetchone()[0]
            
            # Count total hits
            cursor = conn.execute("SELECT SUM(hit_count) FROM llm_cache")
            total_hits = cursor.fetchone()[0] or 0
            
            # Get database size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            return {
                'total_entries': total_entries,
                'expired_entries': expired_entries,
                'total_hits': total_hits,
                'db_size_bytes': db_size,
                'db_size_mb': db_size / (1024 * 1024)
            }


class HybridCache:
    """Hybrid cache combining in-memory and database storage."""
    
    def __init__(self, max_memory_size: int = 1000, db_path: str = "llm_cache.db"):
        """
        Initialize hybrid cache.
        
        Args:
            max_memory_size: Maximum size of in-memory cache
            db_path: Path to database file
        """
        self.memory_cache = LRUCache(max_memory_size)
        self.db_cache = DatabaseCache(db_path)
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from hybrid cache (check memory first, then database).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        # Check memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # If not in memory, check database and populate memory
        value = self.db_cache.get(key)
        if value is not None:
            # Store in memory cache (with shorter TTL for memory)
            self.memory_cache.set(key, value, ttl=1800)  # 30 minutes in memory
            return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Set value in both memory and database cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            
        Returns:
            True if set successfully
        """
        # Set in both caches
        mem_success = self.memory_cache.set(key, value, ttl=ttl)
        db_success = self.db_cache.set(key, value, ttl=ttl)
        
        return mem_success and db_success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid cache statistics."""
        mem_stats = self.memory_cache.get_stats()
        db_stats = self.db_cache.get_stats()
        
        return {
            'memory_cache': mem_stats,
            'database_cache': db_stats,
            'total_entries': mem_stats['total_entries'] + db_stats['total_entries'],
            'total_hits': mem_stats['total_hits'] + db_stats['total_hits']
        }


class LLMCacheManager:
    """Main cache manager for LLM responses."""
    
    def __init__(self, cache_backend: str = "hybrid", **config):
        """
        Initialize LLM cache manager.
        
        Args:
            cache_backend: Cache backend ('memory', 'database', or 'hybrid')
            **config: Backend-specific configuration
        """
        self.logger = logging.getLogger(__name__)
        
        if cache_backend == "memory":
            max_size = config.get('max_size', 1000)
            self.cache = LRUCache(max_size)
        elif cache_backend == "database":
            db_path = config.get('db_path', 'llm_cache.db')
            self.cache = DatabaseCache(db_path)
        else:  # hybrid
            max_memory_size = config.get('max_memory_size', 1000)
            db_path = config.get('db_path', 'llm_cache.db')
            self.cache = HybridCache(max_memory_size, db_path)
        
        self.logger.info(f"LLM Cache initialized with {cache_backend} backend")
    
    def _generate_cache_key(self, prompt: str, model_params: Dict[str, Any]) -> str:
        """
        Generate cache key from prompt and parameters.
        
        Args:
            prompt: Input prompt
            model_params: Model parameters
            
        Returns:
            Hash-based cache key
        """
        cache_input = {
            'prompt': prompt,
            'params': model_params
        }
        cache_input_json = json.dumps(cache_input, sort_keys=True)
        return hashlib.sha256(cache_input_json.encode()).hexdigest()
    
    def get_cached_response(self, prompt: str, model_params: Dict[str, Any]) -> Optional[Any]:
        """
        Get cached LLM response.
        
        Args:
            prompt: Input prompt
            model_params: Model parameters
            
        Returns:
            Cached response or None if not found
        """
        cache_key = self._generate_cache_key(prompt, model_params)
        return self.cache.get(cache_key)
    
    def cache_response(self, prompt: str, model_params: Dict[str, Any], response: Any, ttl: int = 3600) -> bool:
        """
        Cache LLM response.
        
        Args:
            prompt: Input prompt
            model_params: Model parameters
            response: LLM response to cache
            ttl: Time-to-live in seconds
            
        Returns:
            True if cached successfully
        """
        cache_key = self._generate_cache_key(prompt, model_params)
        return self.cache.set(cache_key, response, ttl=ttl)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def cleanup_expired(self) -> int:
        """Clean up expired cache entries (database only)."""
        if hasattr(self.cache, 'db_cache'):
            return self.cache.db_cache.cleanup_expired()
        return 0


def with_llm_caching(ttl: int = 3600, cache_params: Dict[str, Any] = None):
    """
    Decorator for intelligent LLM response caching.
    
    Args:
        ttl: Time-to-live for cache entries
        cache_params: Additional cache configuration
    """
    if cache_params is None:
        cache_params = {}
    
    # Create a shared cache manager instance
    cache_manager = LLMCacheManager(**cache_params)
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            # Extract prompt and model parameters from args/kwargs
            # This assumes the first argument is the prompt and the rest includes model params
            prompt = args[0] if args else ""
            
            # Extract model parameters (this is specific to your implementation)
            # In a real scenario, you'd extract these from the function signature
            model_params = {
                'model': kwargs.get('model', 'default'),
                'temperature': kwargs.get('temperature', 0.7),
                'max_tokens': kwargs.get('max_tokens', 1000),
                'top_p': kwargs.get('top_p', 1.0),
                'frequency_penalty': kwargs.get('frequency_penalty', 0.0),
                'presence_penalty': kwargs.get('presence_penalty', 0.0),
            }
            
            # Try to get cached response
            cached_response = cache_manager.get_cached_response(prompt, model_params)
            if cached_response is not None:
                # Record cache hit
                cache_manager.logger.info(f"Cache HIT for prompt: {prompt[:50]}...")
                return cached_response
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Cache the result
            if result is not None:
                cache_manager.cache_response(prompt, model_params, result, ttl)
                cache_manager.logger.info(f"Cache SET for prompt: {prompt[:50]}...")
            
            return result
        
        # Add cache manager to the wrapper for external access
        wrapper.cache_manager = cache_manager
        return wrapper
    
    return decorator


# Global cache manager instance
llm_cache_manager = LLMCacheManager(cache_backend="hybrid")


def integrate_with_existing_components():
    """
    Helper function to integrate caching with existing system components.
    This would typically be called during system initialization.
    """
    from problem_analyzer import ProblemAnalyzer
    from decomposition_engine import DecompositionEngine
    from sovereign_solution_orchestration import SolutionOrchestrator
    from sovereign_gauntlets import CoherenceGauntlet, CompletenessGauntlet, FeasibilityGauntlet, DependencyGauntlet
    
    # Example: Apply caching to problem analyzer methods
    original_analyze_problem = ProblemAnalyzer.analyze_problem
    ProblemAnalyzer.analyze_problem = with_llm_caching(
        ttl=3600,
        cache_params={'cache_backend': 'hybrid', 'max_memory_size': 500}
    )(original_analyze_problem)
    
    # Example: Apply caching to decomposition engine methods
    original_decompose = DecompositionEngine.decompose
    DecompositionEngine.decompose = with_llm_caching(
        ttl=7200,  # 2 hours for decomposition plans
        cache_params={'cache_backend': 'hybrid', 'max_memory_size': 300}
    )(original_decompose)
    
    # Example: Apply caching to gauntlet methods
    original_coherence_check = CoherenceGauntlet._check_coherence_with_llm
    CoherenceGauntlet._check_coherence_with_llm = with_llm_caching(
        ttl=1800,  # 30 minutes for gauntlet checks
        cache_params={'cache_backend': 'memory', 'max_size': 200}
    )(original_coherence_check)
    
    original_completeness_check = CompletenessGauntlet._check_completeness_with_llm
    CompletenessGauntlet._check_completeness_with_llm = with_llm_caching(
        ttl=1800,
        cache_params={'cache_backend': 'memory', 'max_size': 200}
    )(original_completeness_check)
    
    original_feasibility_check = FeasibilityGauntlet._check_feasibility_with_llm
    FeasibilityGauntlet._check_feasibility_with_llm = with_llm_caching(
        ttl=1800,
        cache_params={'cache_backend': 'memory', 'max_size': 200}
    )(original_feasibility_check)
    
    original_dependency_check = DependencyGauntlet._check_dependency_with_llm
    DependencyGauntlet._check_dependency_with_llm = with_llm_caching(
        ttl=1800,
        cache_params={'cache_backend': 'memory', 'max_size': 200}
    )(original_dependency_check)


# Example usage function
def example_usage():
    """Example of how to use the LLM caching system."""
    
    # Example 1: Direct usage of cache manager
    prompt = "Analyze this problem: How to build a scalable web application?"
    model_params = {
        'model': 'gpt-4',
        'temperature': 0.5,
        'max_tokens': 1000
    }
    
    # Check cache first
    cached_result = llm_cache_manager.get_cached_response(prompt, model_params)
    if cached_result:
        print("Using cached result")
        return cached_result
    
    # Simulate LLM call
    print("Making actual LLM call...")
    # result = some_llm_client.generate(prompt, **model_params)
    result = {"response": "Simulated LLM response", "timestamp": datetime.now().isoformat()}
    
    # Cache the result
    llm_cache_manager.cache_response(prompt, model_params, result)
    
    # Example 2: Using the decorator
    @with_llm_caching(ttl=3600)
    def analyze_content(content: str, model: str = "gpt-4", temperature: float = 0.7):
        # Simulate actual LLM call
        print(f"Actually calling LLM for: {content[:30]}...")
        return {"analysis": f"Analysis of '{content[:20]}...'", "model_used": model}
    
    # First call - will execute function
    result1 = analyze_content("This is some content to analyze")
    # Second call with same parameters - will return cached result
    result2 = analyze_content("This is some content to analyze")
    
    print(f"Results are same: {result1 == result2}")
    print(f"Cache stats: {llm_cache_manager.get_stats()}")
    
    return result


if __name__ == "__main__":
    example_usage()