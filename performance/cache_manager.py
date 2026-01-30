"""
Global Cache Manager for RESE

Implements Redis/memcached integration with intelligent invalidation.

Author: Agent M1
"""

import time
import json
import hashlib
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from collections import OrderedDict
from threading import RLock


@dataclass
class CacheEntry:
    """Cache entry with TTL and metadata"""
    value: Any
    timestamp: float = field(default_factory=time.time)
    ttl: float = 3600.0  # 1 hour default
    version: int = 0
    hit_count: int = 0
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        return time.time() - self.timestamp > self.ttl

    def touch(self):
        """Update timestamp on access"""
        self.timestamp = time.time()
        self.hit_count += 1


class InMemoryCache:
    """
    High-performance in-memory cache with LRU eviction.

    Features:
    - LRU eviction policy
    - TTL-based expiration
    - Size limits
    - Thread-safe
    """

    def __init__(self, max_size: int = 10000, default_ttl: float = 3600.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._stats["expirations"] += 1
                self._stats["misses"] += 1
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            entry.touch()

            self._stats["hits"] += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache"""
        with self._lock:
            # Estimate size
            size_bytes = len(str(value).encode())

            # Check if need to evict
            while len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()

            # Create entry
            entry = CacheEntry(
                value=value,
                ttl=ttl if ttl is not None else self.default_ttl,
                size_bytes=size_bytes
            )

            self._cache[key] = entry
            self._cache.move_to_end(key)

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.

        Returns number of keys invalidated.
        """
        with self._lock:
            to_delete = [key for key in self._cache if pattern in key]

            for key in to_delete:
                del self._cache[key]

            return len(to_delete)

    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()

    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if self._cache:
            key, _ = self._cache.popitem(last=False)
            self._stats["evictions"] += 1

    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        with self._lock:
            to_delete = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in to_delete:
                del self._cache[key]
                self._stats["expirations"] += 1

            return len(to_delete)

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0

            return {
                **self._stats,
                "size": len(self._cache),
                "hit_rate": hit_rate,
                "max_size": self.max_size,
            }


class CacheManager:
    """
    Global cache manager for RESE components.

    Provides unified caching interface with optional Redis backend.
    """

    def __init__(self,
                 use_redis: bool = False,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 max_memory_size: int = 10000):
        self.use_redis = use_redis
        self.redis_client = None

        if use_redis:
            try:
                import redis
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True
                )
            except ImportError:
                print("Redis not available, falling back to in-memory cache")
                self.use_redis = False

        # Always have in-memory cache as fallback/primary
        self.memory_cache = InMemoryCache(max_size=max_memory_size)

    def get(self, key: str, component: str = "default") -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key
            component: Component name (for namespacing)

        Returns:
            Cached value or None
        """
        full_key = f"{component}:{key}"

        # Try Redis first if enabled
        if self.use_redis and self.redis_client:
            try:
                value = self.redis_client.get(full_key)
                if value is not None:
                    return json.loads(value)
            except (redis.RedisError, json.JSONDecodeError, TypeError):
                pass

        # Fall back to memory cache
        return self.memory_cache.get(full_key)

    def set(self, key: str, value: Any,
            component: str = "default",
            ttl: float = 3600.0) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            component: Component name
            ttl: Time-to-live in seconds
        """
        full_key = f"{component}:{key}"

        # Set in memory cache
        self.memory_cache.set(full_key, value, ttl)

        # Set in Redis if enabled
        if self.use_redis and self.redis_client:
            try:
                serialized = json.dumps(value)
                self.redis_client.setex(full_key, int(ttl), serialized)
            except (redis.RedisError, json.JSONDecodeError, TypeError):
                pass

    def delete(self, key: str, component: str = "default") -> None:
        """Delete key from cache"""
        full_key = f"{component}:{key}"

        self.memory_cache.delete(full_key)

        if self.use_redis and self.redis_client:
            try:
                self.redis_client.delete(full_key)
            except (redis.RedisError, TypeError):
                pass

    def invalidate_constraint(self, constraint_id: str) -> None:
        """
        Invalidate all cache entries related to a constraint.

        Called when constraint is added/removed/modified.
        """
        # Invalidate contradiction checks
        pattern = f"contradiction:*:{constraint_id}"
        self.memory_cache.invalidate_pattern(pattern)
        pattern = f"contradiction:{constraint_id}:*"
        self.memory_cache.invalidate_pattern(pattern)

        # Invalidate dependency queries
        self.memory_cache.delete(f"dependencies:{constraint_id}")

        # Invalidate in Redis if enabled
        if self.use_redis and self.redis_client:
            try:
                # Redis scan for pattern deletion
                for key in self.redis_client.scan_iter(f"*:{constraint_id}"):
                    self.redis_client.delete(key)
                for key in self.redis_client.scan_iter(f"{constraint_id}:*"):
                    self.redis_client.delete(key)
            except (redis.RedisError, TypeError):
                pass

    def invalidate_batch(self, keys: List[str], component: str = "default") -> None:
        """Batch invalidate multiple keys"""
        for key in keys:
            self.delete(key, component)

    def clear_all(self) -> None:
        """Clear all caches"""
        self.memory_cache.clear()

        if self.use_redis and self.redis_client:
            try:
                self.redis_client.flushdb()
            except (redis.RedisError, TypeError):
                pass

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        stats = {
            "memory_cache": self.memory_cache.get_stats(),
            "redis_enabled": self.use_redis,
        }

        if self.use_redis and self.redis_client:
            try:
                info = self.redis_client.info("stats")
                stats["redis"] = {
                    "keys": self.redis_client.dbsize(),
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0),
                }
            except (redis.RedisError, TypeError):
                pass

        return stats

    @staticmethod
    def compute_hash(obj: Any) -> str:
        """Compute hash of object for cache key"""
        serialized = json.dumps(obj, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()[:16]
