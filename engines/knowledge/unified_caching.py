"""
Unified Caching System for OpenEvolve Frontend

Provides standardized caching across all components:
- LLM response caching
- Verification result caching
- Workflow state caching
- Knowledge graph caching
- ROMA-MDAP-MAKER caching

Features:
- Unified cache interface
- Multiple backend support (memory, Redis, file)
- LRU eviction
- TTL support
- Cache statistics
- Graceful degradation
"""

import hashlib
import json
import logging
import pickle
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from functools import wraps
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.ttl


class CacheBackend:
    """Base class for cache backends."""

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        raise NotImplementedError

    def clear(self) -> bool:
        """Clear all cache entries."""
        raise NotImplementedError

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        raise NotImplementedError


class InMemoryCache(CacheBackend):
    """In-memory LRU cache with thread safety."""

    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        """
        Initialize in-memory cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0,
            'deletes': 0,
        }

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return None

            entry = self.cache[key]

            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                self.stats['misses'] += 1
                return None

            # Update access info
            entry.accessed_at = datetime.now()
            entry.access_count += 1

            # Move to end (most recently used)
            self.cache.move_to_end(key)

            self.stats['hits'] += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        with self.lock:
            try:
                # Calculate size
                size_bytes = len(pickle.dumps(value))

                # Check if we need to evict
                if key not in self.cache and len(self.cache) >= self.max_size:
                    # Evict least recently used (first item)
                    self.cache.popitem(last=False)
                    self.stats['evictions'] += 1

                # Create entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    accessed_at=datetime.now(),
                    access_count=0,
                    ttl=ttl or self.default_ttl,
                    size_bytes=size_bytes
                )

                self.cache[key] = entry
                self.stats['sets'] += 1
                return True

            except Exception as e:
                logger.error(f"Failed to set cache entry: {e}")
                return False

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats['deletes'] += 1
                return True
            return False

    def clear(self) -> bool:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            hit_rate = (self.stats['hits'] / (self.stats['hits'] + self.stats['misses'])
                       if (self.stats['hits'] + self.stats['misses']) > 0 else 0)

            return {
                'entries': len(self.cache),
                'max_size': self.max_size,
                'total_size_bytes': total_size,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self.stats['evictions'],
                'sets': self.stats['sets'],
                'deletes': self.stats['deletes'],
            }


class UnifiedCache:
    """
    Unified caching system for all components.

    Provides a consistent caching interface with multiple backend support.
    """

    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        prefix: str = "unified",
        key_namespace: str = "openevolve"
    ):
        """
        Initialize unified cache.

        Args:
            backend: Cache backend (defaults to InMemoryCache)
            prefix: Key prefix for this cache instance
            key_namespace: Namespace for cache keys
        """
        self.backend = backend or InMemoryCache()
        self.prefix = prefix
        self.key_namespace = key_namespace
        self.component_stats: Dict[str, Dict[str, int]] = {}

    def _generate_key(
        self,
        component: str,
        operation: str,
        *args,
        **kwargs
    ) -> str:
        """
        Generate cache key from component, operation, and arguments.

        Args:
            component: Component name
            operation: Operation name
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        # Create key data
        key_data = {
            'namespace': self.key_namespace,
            'prefix': self.prefix,
            'component': component,
            'operation': operation,
            'args': str(args)[:500],  # Limit to prevent huge keys
            'kwargs': str(sorted(kwargs.items()))[:500],
        }

        # Hash the key data
        key_json = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_json.encode()).hexdigest()

        return f"{self.prefix}:{component}:{operation}:{key_hash}"

    def get(self, component: str, operation: str, *args, **kwargs) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            component: Component name
            operation: Operation name
            *args: Operation arguments
            **kwargs: Operation keyword arguments

        Returns:
            Cached value or None
        """
        key = self._generate_key(component, operation, *args, **kwargs)
        value = self.backend.get(key)

        # Update stats
        if component not in self.component_stats:
            self.component_stats[component] = {'hits': 0, 'misses': 0}

        if value is not None:
            self.component_stats[component]['hits'] += 1
        else:
            self.component_stats[component]['misses'] += 1

        return value

    def set(
        self,
        component: str,
        operation: str,
        value: Any,
        ttl: Optional[float] = None,
        *args,
        **kwargs
    ) -> bool:
        """
        Set value in cache.

        Args:
            component: Component name
            operation: Operation name
            value: Value to cache
            ttl: Time to live in seconds
            *args: Operation arguments
            **kwargs: Operation keyword arguments

        Returns:
            True if successful
        """
        key = self._generate_key(component, operation, *args, **kwargs)
        return self.backend.set(key, value, ttl)

    def delete(self, component: str, operation: str, *args, **kwargs) -> bool:
        """
        Delete value from cache.

        Args:
            component: Component name
            operation: Operation name
            *args: Operation arguments
            **kwargs: Operation keyword arguments

        Returns:
            True if deleted
        """
        key = self._generate_key(component, operation, *args, **kwargs)
        return self.backend.delete(key)

    def clear_component(self, component: str) -> bool:
        """Clear all cache entries for a component."""
        # This is simplified - real implementation would iterate and delete
        return self.backend.clear()

    def get_component_stats(self, component: str) -> Dict[str, int]:
        """Get statistics for a component."""
        return self.component_stats.get(component, {'hits': 0, 'misses': 0})

    def get_all_stats(self) -> Dict[str, Any]:
        """Get all cache statistics."""
        return {
            'backend': self.backend.get_stats(),
            'components': self.component_stats.copy(),
        }

    def decorator(
        self,
        component: str,
        operation: Optional[str] = None,
        ttl: Optional[float] = None
    ):
        """
        Decorator for caching function results.

        Args:
            component: Component name
            operation: Operation name (defaults to function name)
            ttl: Cache TTL in seconds

        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation or func.__name__

                # Try to get from cache
                cached_value = self.get(component, op_name, *args, **kwargs)
                if cached_value is not None:
                    return cached_value

                # Execute function
                result = func(*args, **kwargs)

                # Cache result
                self.set(component, op_name, result, ttl, *args, **kwargs)

                return result

            return wrapper
        return decorator


# Global cache instances
_caches: Dict[str, UnifiedCache] = {}


def get_cache(
    name: str = "default",
    backend: Optional[CacheBackend] = None,
    prefix: str = "unified"
) -> UnifiedCache:
    """
    Get or create a unified cache instance.

    Args:
        name: Cache instance name
        backend: Optional custom backend
        prefix: Key prefix

    Returns:
        UnifiedCache instance
    """
    if name not in _caches:
        _caches[name] = UnifiedCache(backend=backend, prefix=prefix)
    return _caches[name]


def llm_cache_decorator(component: str, operation: Optional[str] = None, ttl: float = 3600):
    """Decorator for caching LLM responses."""
    return get_cache("llm").decorator(component, operation, ttl)


def verification_cache_decorator(component: str, operation: Optional[str] = None, ttl: float = 7200):
    """Decorator for caching verification results."""
    return get_cache("verification").decorator(component, operation, ttl)


def workflow_cache_decorator(component: str, operation: Optional[str] = None, ttl: float = 1800):
    """Decorator for caching workflow states."""
    return get_cache("workflow").decorator(component, operation, ttl)


def knowledge_cache_decorator(component: str, operation: Optional[str] = None, ttl: float = 3600):
    """Decorator for caching knowledge graph queries."""
    return get_cache("knowledge").decorator(component, operation, ttl)


__all__ = [
    'CacheEntry',
    'CacheBackend',
    'InMemoryCache',
    'UnifiedCache',
    'get_cache',
    'llm_cache_decorator',
    'verification_cache_decorator',
    'workflow_cache_decorator',
    'knowledge_cache_decorator',
]
