"""
Query Cache

Caches query results for improved performance.

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import hashlib
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached query result"""
    query_hash: str
    results: List[Dict[str, Any]]
    timestamp: datetime
    ttl: int  # Time to live in seconds
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        age = (datetime.utcnow() - self.timestamp).total_seconds()
        return age > self.ttl
    
    def touch(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


class QueryCache:
    """
    Cache for query results
    
    Features:
    - In-memory caching with TTL
    - Persistent cache to disk
    - LRU eviction
    - Cache statistics
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600,  # 1 hour
        persist_path: Optional[str] = None
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.persist_path = Path(persist_path) if persist_path else None
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "insertions": 0,
        }
        
        # Load persisted cache
        if self.persist_path:
            self._load_from_disk()
    
    def _hash_query(self, query: str, params: Optional[Dict] = None) -> str:
        """Generate hash for query"""
        content = query
        if params:
            content += json.dumps(params, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(
        self,
        query: str,
        params: Optional[Dict] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached results for a query
        
        Args:
            query: Query string
            params: Query parameters
        
        Returns:
            Cached results or None if not found/expired
        """
        query_hash = self._hash_query(query, params)
        
        with self._lock:
            entry = self._cache.get(query_hash)
            
            if entry is None:
                self._stats["misses"] += 1
                return None
            
            if entry.is_expired():
                # Remove expired entry
                del self._cache[query_hash]
                self._stats["misses"] += 1
                return None
            
            # Cache hit
            entry.touch()
            self._stats["hits"] += 1
            
            return entry.results
    
    def set(
        self,
        query: str,
        results: List[Dict[str, Any]],
        params: Optional[Dict] = None,
        ttl: Optional[int] = None
    ):
        """
        Cache results for a query
        
        Args:
            query: Query string
            results: Results to cache
            params: Query parameters
            ttl: Time to live in seconds (default: self.default_ttl)
        """
        query_hash = self._hash_query(query, params)
        ttl = ttl or self.default_ttl
        
        with self._lock:
            # Check if we need to evict
            if len(self._cache) >= self.max_size and query_hash not in self._cache:
                self._evict_lru()
            
            # Store entry
            self._cache[query_hash] = CacheEntry(
                query_hash=query_hash,
                results=results,
                timestamp=datetime.utcnow(),
                ttl=ttl
            )
            
            self._stats["insertions"] += 1
        
        # Persist if configured
        if self.persist_path:
            self._save_to_disk()
    
    def invalidate(self, query: str, params: Optional[Dict] = None) -> bool:
        """Invalidate a cached query"""
        query_hash = self._hash_query(query, params)
        
        with self._lock:
            if query_hash in self._cache:
                del self._cache[query_hash]
                return True
            return False
    
    def invalidate_all(self):
        """Clear all cached entries"""
        with self._lock:
            self._cache.clear()
        
        logger.info("Cache cleared")
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate queries matching a pattern"""
        with self._lock:
            to_remove = [
                k for k in self._cache.keys()
                if pattern in k
            ]
            for k in to_remove:
                del self._cache[k]
        
        logger.info(f"Invalidated {len(to_remove)} cache entries matching '{pattern}'")
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self._cache:
            return
        
        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed
        )
        
        del self._cache[lru_key]
        self._stats["evictions"] += 1
    
    def _save_to_disk(self):
        """Persist cache to disk"""
        if not self.persist_path:
            return
        
        try:
            data = {
                "entries": [
                    {
                        "query_hash": e.query_hash,
                        "results": e.results,
                        "timestamp": e.timestamp.isoformat(),
                        "ttl": e.ttl,
                        "access_count": e.access_count,
                        "last_accessed": e.last_accessed.isoformat(),
                    }
                    for e in self._cache.values()
                    if not e.is_expired()
                ]
            }
            
            self.persist_path.write_text(json.dumps(data, indent=2))
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _load_from_disk(self):
        """Load cache from disk"""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        try:
            data = json.loads(self.persist_path.read_text())
            
            for entry_data in data.get("entries", []):
                entry = CacheEntry(
                    query_hash=entry_data["query_hash"],
                    results=entry_data["results"],
                    timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                    ttl=entry_data["ttl"],
                    access_count=entry_data.get("access_count", 0),
                    last_accessed=datetime.fromisoformat(
                        entry_data.get("last_accessed", entry_data["timestamp"])
                    )
                )
                
                if not entry.is_expired():
                    self._cache[entry.query_hash] = entry
            
            logger.info(f"Loaded {len(self._cache)} entries from cache")
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "insertions": self._stats["insertions"],
            }
    
    def cleanup(self):
        """Remove expired entries"""
        with self._lock:
            expired = [
                k for k, e in self._cache.items()
                if e.is_expired()
            ]
            for k in expired:
                del self._cache[k]
        
        logger.debug(f"Cleaned up {len(expired)} expired entries")
