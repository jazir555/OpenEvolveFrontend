"""Caching utilities for Adaptive MDAP."""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field
from threading import Lock
import logging

logger = logging.getLogger("adaptive_mdap.cache")


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    ttl_seconds: Optional[float] = None
    access_count: int = 0
    last_access: float = field(default_factory=time.time)


class BaseCache:
    """Base cache implementation with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats["misses"] += 1
                return None
            
            # Check TTL
            if entry.ttl_seconds is not None:
                if time.time() - entry.timestamp > entry.ttl_seconds:
                    del self._cache[key]
                    self._stats["misses"] += 1
                    return None
            
            # Update access stats
            entry.access_count += 1
            entry.last_access = time.time()
            self._stats["hits"] += 1
            
            return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None
    ) -> None:
        """Set value in cache."""
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl
        
        with self._lock:
            # Evict if at capacity and key is new
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            self._cache[key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl_seconds=ttl_seconds,
            )
    
    def delete(self, key: str) -> bool:
        """Delete key from cache. Returns True if key existed."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0.0
            return {
                **self._stats,
                "size": len(self._cache),
                "hit_rate": hit_rate,
                "total_requests": total,
            }
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_access)
        del self._cache[lru_key]
        self._stats["evictions"] += 1


class EmbeddingCache(BaseCache):
    """Cache for domain embeddings with disk persistence."""
    
    def __init__(
        self,
        cache_dir: str = ".cache/adaptive_mdap",
        max_size: int = 10000,
        default_ttl: float = 7 * 24 * 3600,  # 7 days
    ):
        super().__init__(max_size, default_ttl)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._disk_cache_path = self.cache_dir / "embeddings.pkl"
        self._load_from_disk()
    
    def _get_embedding_key(self, text: str) -> str:
        """Generate cache key for embedding text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding vector for text."""
        key = self._get_embedding_key(text)
        return self.get(key)
    
    def set_embedding(self, text: str, embedding: List[float]) -> None:
        """Store embedding vector for text."""
        key = self._get_embedding_key(text)
        self.set(key, embedding)
        self._save_to_disk()
    
    def _save_to_disk(self) -> None:
        """Save cache to disk."""
        try:
            with open(self._disk_cache_path, 'wb') as f:
                pickle.dump(dict(self._cache), f)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not self._disk_cache_path.exists():
            return
        
        try:
            with open(self._disk_cache_path, 'rb') as f:
                loaded = pickle.load(f)
                # Filter out expired entries
                now = time.time()
                for key, entry in loaded.items():
                    if entry.ttl_seconds is None or (now - entry.timestamp) < entry.ttl_seconds:
                        self._cache[key] = entry
            logger.info(f"Loaded {len(self._cache)} embeddings from disk cache")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")


class FeatureCache(BaseCache):
    """Cache for feature computations."""
    
    def __init__(self, max_size: int = 10000, default_ttl: float = 3600):  # 1 hour
        super().__init__(max_size, default_ttl)
    
    def get_features(self, subproblem_id: str) -> Optional[Dict[str, float]]:
        """Get cached features for sub-problem."""
        return self.get(f"features:{subproblem_id}")
    
    def set_features(self, subproblem_id: str, features: Dict[str, float]) -> None:
        """Cache features for sub-problem."""
        self.set(f"features:{subproblem_id}", features)
    
    def get_complexity(self, subproblem_id: str) -> Optional[float]:
        """Get cached complexity score."""
        return self.get(f"complexity:{subproblem_id}")
    
    def set_complexity(self, subproblem_id: str, score: float) -> None:
        """Cache complexity score."""
        self.set(f"complexity:{subproblem_id}", score)


# Global cache instances
_embedding_cache: Optional[EmbeddingCache] = None
_feature_cache: Optional[FeatureCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """Get the global embedding cache instance."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache


def get_feature_cache() -> FeatureCache:
    """Get the global feature cache instance."""
    global _feature_cache
    if _feature_cache is None:
        _feature_cache = FeatureCache()
    return _feature_cache


def get_cache_stats() -> Dict[str, Any]:
    """Get combined cache statistics for monitoring."""
    embedding_stats = get_embedding_cache().get_stats()
    feature_stats = get_feature_cache().get_stats()
    
    # Calculate combined hit rate
    total_hits = embedding_stats["hits"] + feature_stats["hits"]
    total_misses = embedding_stats["misses"] + feature_stats["misses"]
    total_requests = total_hits + total_misses
    combined_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
    
    return {
        "embedding_cache": embedding_stats,
        "feature_cache": feature_stats,
        "combined_hit_rate": combined_hit_rate,
        "total_size": embedding_stats["size"] + feature_stats["size"],
        "total_evictions": embedding_stats["evictions"] + feature_stats["evictions"],
    }
