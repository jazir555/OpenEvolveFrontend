"""
Knowledge Graph Query Result Cache and Statistics Collector.

Implements a TTL + size-bounded in-memory cache for query results, plus a
statistics collector that records cache hits, execution time and result
sizes and exposes monitoring hooks.

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

import time
import hashlib
import threading
import asyncio
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Result cache
# --------------------------------------------------------------------------- #
class CacheEntry:
    __slots__ = ("value", "expires_at", "size", "created_at")

    def __init__(self, value: Any, ttl: float, size: int):
        now = time.time()
        self.value = value
        self.expires_at = now + ttl
        self.size = size
        self.created_at = now


class ResultCache:
    """TTL- and size-guarded in-memory cache for query results."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.default_ttl = float(self.config.get("default_ttl", 300))
        self.max_entries = int(self.config.get("max_entries", 1000))
        self.max_entry_size = int(self.config.get("max_entry_size", 10000))
        self.enabled = bool(self.config.get("enabled", True))
        self._store: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._monitor_hooks: List[Callable[[str, Any], None]] = []

    # -- monitoring -------------------------------------------------------- #
    def add_monitor_hook(self, hook: Callable[[str, Any], None]) -> None:
        self._monitor_hooks.append(hook)

    def _fire(self, event: str, data: Any) -> None:
        for hook in self._monitor_hooks:
            try:
                hook(event, data)
            except Exception:
                pass

    # -- key generation ---------------------------------------------------- #
    @staticmethod
    def generate_cache_key(query: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        q = query.encode("utf-8")
        p = (str(sorted((parameters or {}).items()))).encode("utf-8")
        return hashlib.sha256(q + b"::" + p).hexdigest()

    # -- public API -------------------------------------------------------- #
    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            self._misses += 1
            return None
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.expires_at <= time.time():
                del self._store[key]
                self._evictions += 1
                self._misses += 1
                return None
            self._hits += 1
            self._fire("cache_hit", {"key": key, "size": entry.size})
            return entry.value

    async def get_async(self, key: str) -> Optional[Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get, key)

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        if not self.enabled:
            return False
        ttl = self.default_ttl if ttl is None else ttl
        size = self._estimate_size(value)
        if size > self.max_entry_size:
            self._fire("cache_skip", {"key": key, "reason": "too_large", "size": size})
            return False
        with self._lock:
            if len(self._store) >= self.max_entries:
                self._evict_one()
            self._store[key] = CacheEntry(value, ttl, size)
            self._fire("cache_set", {"key": key, "ttl": ttl, "size": size})
            return True

    async def set_async(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.set(key, value, ttl))

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)
        self._fire("cache_invalidate", {"key": key})

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
        self._fire("cache_clear", {})

    def _evict_one(self) -> None:
        # Simple LRU-ish eviction: drop soonest-to-expire entry.
        oldest_key = None
        oldest_exp = None
        for k, e in self._store.items():
            if oldest_exp is None or e.expires_at < oldest_exp:
                oldest_exp = e.expires_at
                oldest_key = k
        if oldest_key is not None:
            self._store.pop(oldest_key, None)
            self._evictions += 1

    @staticmethod
    def _estimate_size(value: Any) -> int:
        try:
            if isinstance(value, (list, tuple)):
                return len(value)
            if isinstance(value, dict):
                return sum(len(v) if isinstance(v, (list, tuple)) else 1 for v in value.values())
        except Exception:
            return 1
        return 1

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._store),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": (self._hits / (self._hits + self._misses))
                if (self._hits + self._misses) else 0.0,
            }


# --------------------------------------------------------------------------- #
# Statistics collector
# --------------------------------------------------------------------------- #
@dataclass
class QueryStatistic:
    query_id: str
    query_type: str
    execution_time_ms: float
    result_size: int
    cache_hit: bool
    backend: str
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None


class StatisticsCollector:
    """Collects query execution statistics and exposes monitoring hooks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_history = int(self.config.get("max_history", 10000))
        self._history: List[QueryStatistic] = []
        self._lock = threading.RLock()
        self._monitor_hooks: List[Callable[[QueryStatistic], None]] = []
        self._totals = {
            "executions": 0,
            "cache_hits": 0,
            "errors": 0,
            "total_time_ms": 0.0,
            "total_result_size": 0,
        }

    def add_monitor_hook(self, hook: Callable[[QueryStatistic], None]) -> None:
        self._monitor_hooks.append(hook)

    def record_cache_hit(self, query: str, execution_time_ms: float,
                         backend: str = "memory", result_size: int = 0) -> QueryStatistic:
        return self._record(query, execution_time_ms, result_size, True, backend)

    def record_query_execution(self, query: str, execution_time_ms: float,
                               result_size: int, backend: str = "memory",
                               error: Optional[str] = None) -> QueryStatistic:
        return self._record(query, execution_time_ms, result_size,
                            error is not None, backend, error)

    def _record(self, query: str, execution_time_ms: float, result_size: int,
                cache_hit: bool, backend: str, error: Optional[str] = None) -> QueryStatistic:
        stat = QueryStatistic(
            query_id=hashlib.sha256(query.encode()).hexdigest()[:16],
            query_type="read",
            execution_time_ms=execution_time_ms,
            result_size=result_size,
            cache_hit=cache_hit,
            backend=backend,
            error=error,
        )
        with self._lock:
            self._history.append(stat)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]
            self._totals["executions"] += 1
            self._totals["total_time_ms"] += execution_time_ms
            self._totals["total_result_size"] += result_size
            if cache_hit:
                self._totals["cache_hits"] += 1
            if error:
                self._totals["errors"] += 1
        for hook in self._monitor_hooks:
            try:
                hook(stat)
            except Exception:
                pass
        return stat

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            totals = dict(self._totals)
            totals["cache_hit_rate"] = (
                totals["cache_hits"] / totals["executions"]
                if totals["executions"] else 0.0)
            totals["avg_execution_time_ms"] = (
                totals["total_time_ms"] / totals["executions"]
                if totals["executions"] else 0.0)
            totals["error_rate"] = (
                totals["errors"] / totals["executions"]
                if totals["executions"] else 0.0)
            return totals

    def recent(self, n: int = 50) -> List[QueryStatistic]:
        with self._lock:
            return list(self._history[-n:])


__all__ = [
    "CacheEntry", "ResultCache", "QueryStatistic", "StatisticsCollector",
]
