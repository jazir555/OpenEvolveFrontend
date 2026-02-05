"""
Z3 Result Caching and Persistence Layer

Provides:
- Intelligent result caching with TTL
- Persistent storage of solutions and proofs
- Cache invalidation strategies
- Distributed cache support (Valkey)
- Result versioning
- Cache analytics

Author: OpenEvolve
Created: 2026-01-31
"""

import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import OrderedDict
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# Valkey Import with Graceful Fallback
# =============================================================================

VALKEY_AVAILABLE = False
valkey = None
ConnectionPool = None

# Try to import from valkey_integration first (our wrapper)
try:
    from valkey_integration import valkey as valkey_module, ConnectionPool as ValkeyPool, VALKEY_AVAILABLE as VK_AVAIL
    valkey = valkey_module
    ConnectionPool = ValkeyPool
    VALKEY_AVAILABLE = VK_AVAIL
    if VALKEY_AVAILABLE:
        logger.info("Valkey integration loaded successfully")
except ImportError:
    # Try direct valkey import
    try:
        import valkey as valkey_module
        from valkey.connection import ConnectionPool as ValkeyPool
        valkey = valkey_module
        ConnectionPool = ValkeyPool
        VALKEY_AVAILABLE = True
        logger.info("Valkey library imported directly")
    except ImportError:
        # Fallback to redis-py (Valkey-compatible)
        try:
            import redis
            from redis.connection import ConnectionPool as RedisPool
            valkey = redis
            ConnectionPool = RedisPool
            VALKEY_AVAILABLE = True
            logger.info("Using redis-py as Valkey-compatible backend")
        except ImportError:
            VALKEY_AVAILABLE = False
            logger.debug("Neither valkey nor redis-py installed. Distributed cache will not be available.")


# =============================================================================
# Cache Configuration
# =============================================================================

class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


@dataclass
class CacheConfig:
    """Configuration for caching."""
    max_size: int = 1000
    default_ttl: float = 3600  # 1 hour
    policy: CachePolicy = CachePolicy.LRU
    persistent_storage: bool = True
    db_path: str = "z3_cache.db"
    compression: bool = False
    checksum_verification: bool = True
    
    # Distributed cache settings (Valkey)
    distributed: bool = False
    valkey_host: Optional[str] = None
    valkey_port: int = 6379
    valkey_db: int = 0
    valkey_password: Optional[str] = None
    valkey_ssl: bool = False
    valkey_socket_timeout: float = 5.0
    valkey_socket_connect_timeout: float = 5.0
    valkey_max_connections: int = 50
    valkey_key_prefix: str = "z3_cache"


@dataclass
class CacheEntry:
    """Single cache entry."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    version: int = 1
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "size_bytes": self.size_bytes,
            "tags": self.tags,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], value: Any) -> "CacheEntry":
        """Create CacheEntry from dictionary."""
        return cls(
            key=data.get("key", ""),
            value=value,
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at"),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed", time.time()),
            size_bytes=data.get("size_bytes", 0),
            tags=data.get("tags", []),
            version=data.get("version", 1)
        )


# =============================================================================
# Valkey Cache Backend
# =============================================================================

class ValkeyCacheBackend:
    """
    Valkey-based distributed cache backend with connection pooling.
    
    Features:
    - Connection pooling for efficient resource usage
    - TTL support with native Valkey expiration
    - Tag-based invalidation using Valkey sets
    - Serialization/deserialization for dataclasses
    - Graceful fallback handling
    - Redis-compatible (works with Valkey, Redis, or redis-py)
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._valkey: Optional[Any] = None
        self._pool: Optional[Any] = None
        self._lock = threading.RLock()
        self._initialized = False
        
        if not VALKEY_AVAILABLE:
            logger.warning("Valkey/Redis not installed. Cannot use distributed cache.")
            return
        
        self._initialize_connection()
    
    def _initialize_connection(self) -> bool:
        """Initialize Valkey connection pool."""
        if self._initialized:
            return True
        
        try:
            # Build connection kwargs
            conn_kwargs = {
                "host": self.config.valkey_host or "localhost",
                "port": self.config.valkey_port,
                "db": self.config.valkey_db,
                "password": self.config.valkey_password,
                "socket_timeout": self.config.valkey_socket_timeout,
                "socket_connect_timeout": self.config.valkey_socket_connect_timeout,
                "max_connections": self.config.valkey_max_connections,
                "retry_on_timeout": True,
                "health_check_interval": 30
            }
            
            # Add SSL parameters only if SSL is enabled
            if self.config.valkey_ssl:
                conn_kwargs["ssl"] = True
            
            # Create connection pool
            self._pool = ConnectionPool(**conn_kwargs)
            
            # Create Valkey/Redis client
            self._valkey = valkey.Redis(connection_pool=self._pool)
            
            # Test connection
            self._valkey.ping()
            
            self._initialized = True
            logger.info(
                f"Valkey cache connected to {self.config.valkey_host}:{self.config.valkey_port}"
            )
            return True
            
        except Exception as e:
            logger.warning(f"Failed to connect to Valkey: {e}. Will fall back to SQLite.")
            self._cleanup_connection()
            return False
    
    def _cleanup_connection(self):
        """Clean up Valkey connection resources."""
        try:
            if self._pool:
                self._pool.disconnect()
        except Exception:
            pass
        self._valkey = None
        self._pool = None
        self._initialized = False
    
    def is_available(self) -> bool:
        """Check if Valkey connection is available."""
        if not VALKEY_AVAILABLE or not self._initialized or not self._valkey:
            return False
        try:
            return self._valkey.ping()
        except Exception:
            return False
    
    def _make_key(self, key: str) -> str:
        """Create Valkey key with prefix."""
        return f"{self.config.valkey_key_prefix}:{key}"
    
    def _make_tags_key(self, tag: str) -> str:
        """Create Valkey key for tag set."""
        return f"{self.config.valkey_key_prefix}:tags:{tag}"
    
    def _serialize_value(self, entry: CacheEntry) -> str:
        """Serialize cache entry to JSON string."""
        data = {
            "key": entry.key,
            "value": entry.value,
            "created_at": entry.created_at,
            "expires_at": entry.expires_at,
            "access_count": entry.access_count,
            "last_accessed": entry.last_accessed,
            "size_bytes": entry.size_bytes,
            "tags": entry.tags,
            "version": entry.version
        }
        return json.dumps(data, default=self._json_serializer)
    
    def _deserialize_value(self, data: str) -> Optional[CacheEntry]:
        """Deserialize JSON string to cache entry."""
        try:
            parsed = json.loads(data)
            return CacheEntry.from_dict(parsed, parsed.get("value"))
        except Exception as e:
            logger.error(f"Failed to deserialize cache entry: {e}")
            return None
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for special types."""
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry from Valkey."""
        if not self.is_available():
            return None
        
        try:
            with self._lock:
                valkey_key = self._make_key(key)
                data = self._valkey.get(valkey_key)
                
                if data is None:
                    return None
                
                entry = self._deserialize_value(data.decode('utf-8'))
                
                if entry and entry.is_expired():
                    # Entry expired, remove it
                    self._valkey.delete(valkey_key)
                    return None
                
                return entry
                
        except Exception as e:
            logger.error(f"Valkey get error: {e}")
            return None
    
    def set(self, entry: CacheEntry) -> bool:
        """Store cache entry in Valkey with TTL."""
        if not self.is_available():
            return False
        
        try:
            with self._lock:
                valkey_key = self._make_key(entry.key)
                data = self._serialize_value(entry)
                
                # Calculate TTL
                ttl_seconds = None
                if entry.expires_at:
                    ttl_seconds = int(entry.expires_at - time.time())
                    if ttl_seconds <= 0:
                        # Already expired, don't store
                        return False
                elif self.config.default_ttl > 0:
                    ttl_seconds = int(self.config.default_ttl)
                
                # Store with TTL if specified
                if ttl_seconds:
                    self._valkey.setex(valkey_key, ttl_seconds, data)
                else:
                    self._valkey.set(valkey_key, data)
                
                # Add to tag sets for tag-based invalidation
                for tag in entry.tags:
                    tag_key = self._make_tags_key(tag)
                    self._valkey.sadd(tag_key, entry.key)
                    # Set expiration on tag set too
                    if ttl_seconds:
                        self._valkey.expire(tag_key, ttl_seconds)
                
                return True
                
        except Exception as e:
            logger.error(f"Valkey set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete cache entry from Valkey."""
        if not self.is_available():
            return False
        
        try:
            with self._lock:
                valkey_key = self._make_key(key)
                
                # Get entry first to remove from tag sets
                data = self._valkey.get(valkey_key)
                if data:
                    entry = self._deserialize_value(data.decode('utf-8'))
                    if entry:
                        for tag in entry.tags:
                            tag_key = self._make_tags_key(tag)
                            self._valkey.srem(tag_key, entry.key)
                
                self._valkey.delete(valkey_key)
                return True
                
        except Exception as e:
            logger.error(f"Valkey delete error: {e}")
            return False
    
    def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate entries by tags."""
        if not self.is_available() or not tags:
            return 0
        
        try:
            with self._lock:
                keys_to_delete = set()
                
                # Collect keys from all matching tag sets
                for tag in tags:
                    tag_key = self._make_tags_key(tag)
                    keys = self._valkey.smembers(tag_key)
                    keys_to_delete.update(k.decode('utf-8') if isinstance(k, bytes) else k 
                                          for k in keys)
                    # Delete the tag set itself
                    self._valkey.delete(tag_key)
                
                # Delete all collected entries
                count = 0
                for key in keys_to_delete:
                    valkey_key = self._make_key(key)
                    if self._valkey.delete(valkey_key):
                        count += 1
                
                return count
                
        except Exception as e:
            logger.error(f"Valkey invalidate by tags error: {e}")
            return 0
    
    def clear(self) -> bool:
        """Clear all cache entries with the configured prefix."""
        if not self.is_available():
            return False
        
        try:
            with self._lock:
                pattern = f"{self.config.valkey_key_prefix}:*"
                cursor = 0
                
                while True:
                    cursor, keys = self._valkey.scan(
                        cursor=cursor, 
                        match=pattern, 
                        count=1000
                    )
                    if keys:
                        self._valkey.delete(*keys)
                    if cursor == 0:
                        break
                
                logger.info("Valkey cache cleared")
                return True
                
        except Exception as e:
            logger.error(f"Valkey clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Valkey cache statistics."""
        if not self.is_available():
            return {"error": "Valkey not available"}
        
        try:
            info = self._valkey.info()
            pattern = f"{self.config.valkey_key_prefix}:*"
            
            # Count keys with our prefix
            cursor = 0
            key_count = 0
            while True:
                cursor, keys = self._valkey.scan(
                    cursor=cursor,
                    match=pattern,
                    count=1000
                )
                key_count += len(keys)
                if cursor == 0:
                    break
            
            return {
                "connected": True,
                "used_memory_human": info.get("used_memory_human", "N/A"),
                "total_keys": info.get("db" + str(self.config.valkey_db), {}).get("keys", key_count),
                "cache_keys": key_count,
                "connected_clients": info.get("connected_clients", 0),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "backend": "valkey"
            }
            
        except Exception as e:
            logger.error(f"Valkey stats error: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close Valkey connection."""
        self._cleanup_connection()


# =============================================================================
# Cache Statistics
# =============================================================================

@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0
    avg_access_time_ms: float = 0.0
    
    def update_hit_rate(self):
        """Update hit rate."""
        total = self.hits + self.misses
        if total > 0:
            self.hit_rate = self.hits / total
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": f"{self.hit_rate:.2%}",
            "entry_count": self.entry_count,
            "total_size_mb": self.total_size_bytes / (1024 * 1024)
        }


# =============================================================================
# Result Cache
# =============================================================================

class Z3ResultCache:
    """
    Intelligent result cache for Z3 operations.
    
    Features:
    - Multiple eviction policies
    - Persistent storage (SQLite fallback)
    - Distributed cache support (Valkey with connection pooling)
    - TTL support
    - Tag-based invalidation
    - Statistics tracking
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self._db_conn = None
        self._valkey_backend: Optional[ValkeyCacheBackend] = None
        self._use_valkey = False
        
        # Initialize Valkey if configured and available
        if self.config.distributed and self.config.valkey_host:
            self._valkey_backend = ValkeyCacheBackend(self.config)
            if self._valkey_backend.is_available():
                self._use_valkey = True
                logger.info("Using Valkey as primary cache backend")
        
        # Initialize SQLite as fallback or for local caching
        if self.config.persistent_storage and not self._use_valkey:
            self._init_db()
        elif self.config.persistent_storage:
            # Use SQLite as secondary/backup storage
            self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for persistent cache."""
        self._db_path = Path(self.config.db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._db_conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        cursor = self._db_conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                value BLOB,
                created_at REAL,
                expires_at REAL,
                access_count INTEGER,
                last_accessed REAL,
                size_bytes INTEGER,
                tags TEXT,
                version INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_expires ON cache_entries(expires_at)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_tags ON cache_entries(tags)
        ''')
        
        self._db_conn.commit()
        
        # Load existing entries
        self._load_from_db()
    
    def _load_from_db(self):
        """Load cache entries from database."""
        if not self._db_conn:
            return
        
        try:
            cursor = self._db_conn.cursor()
            
            cursor.execute('''
                SELECT key, value, created_at, expires_at, access_count, 
                       last_accessed, size_bytes, tags, version
                FROM cache_entries
                WHERE expires_at IS NULL OR expires_at > ?
            ''', (time.time(),))
            
            for row in cursor.fetchall():
                key, value_json = row[0], row[1]
                
                try:
                    value = json.loads(value_json)
                    
                    entry = CacheEntry(
                        key=key,
                        value=value,
                        created_at=row[2],
                        expires_at=row[3],
                        access_count=row[4],
                        last_accessed=row[5],
                        size_bytes=row[6],
                        tags=json.loads(row[7]) if row[7] else [],
                        version=row[8]
                    )
                    
                    if not entry.is_expired():
                        self._cache[key] = entry
                        self._stats.total_size_bytes += entry.size_bytes
                
                except Exception as e:
                    logger.warning(f"Failed to load cache entry {key}: {e}")
            
            self._stats.entry_count = len(self._cache)
            logger.info(f"Loaded {self._stats.entry_count} cache entries from database")
        
        except Exception as e:
            logger.error(f"Failed to load cache from database: {e}")
    
    def _save_to_db(self, entry: CacheEntry):
        """Save entry to database."""
        if not self._db_conn:
            return
        
        try:
            cursor = self._db_conn.cursor()
            
            value_json = json.dumps(entry.value)
            
            cursor.execute('''
                INSERT OR REPLACE INTO cache_entries
                (key, value, created_at, expires_at, access_count, 
                 last_accessed, size_bytes, tags, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.key,
                value_json,
                entry.created_at,
                entry.expires_at,
                entry.access_count,
                entry.last_accessed,
                entry.size_bytes,
                json.dumps(entry.tags),
                entry.version
            ))
            
            self._db_conn.commit()
        
        except Exception as e:
            logger.error(f"Failed to save cache entry to database: {e}")
    
    def _generate_key(
        self,
        operation: str,
        params: Dict[str, Any]
    ) -> str:
        """Generate cache key from operation and parameters."""
        # Sort parameters for consistent hashing
        param_str = json.dumps(params, sort_keys=True)
        key_data = f"{operation}:{param_str}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(
        self,
        operation: str,
        params: Dict[str, Any],
        default: Any = None
    ) -> Tuple[bool, Any]:
        """
        Get cached result.
        
        Returns:
            Tuple of (cache_hit, value)
        """
        key = self._generate_key(operation, params)
        
        with self._lock:
            # Try Redis first if enabled
            if self._use_valkey and self._valkey_backend:
                entry = self._valkey_backend.get(key)
                if entry is not None:
                    self._stats.hits += 1
                    self._stats.update_hit_rate()
                    # Update in-memory cache for faster subsequent access
                    self._cache[key] = entry
                    return True, entry.value
            
            # Check local cache
            entry = self._cache.get(key)
            
            if entry is None:
                # Try loading from SQLite if Redis missed
                if self._db_conn and not self._use_valkey:
                    entry = self._load_entry_from_db(key)
                    if entry:
                        self._cache[key] = entry
                
                if entry is None:
                    self._stats.misses += 1
                    self._stats.update_hit_rate()
                    return False, default
            
            if entry.is_expired():
                self._evict_entry(key)
                self._stats.misses += 1
                self._stats.update_hit_rate()
                return False, default
            
            # Update access stats
            entry.touch()
            
            # Update LRU order
            if self.config.policy == CachePolicy.LRU:
                self._cache.move_to_end(key)
            
            self._stats.hits += 1
            self._stats.update_hit_rate()
            
            return True, entry.value
    
    def _load_entry_from_db(self, key: str) -> Optional[CacheEntry]:
        """Load a single entry from database."""
        if not self._db_conn:
            return None
        
        try:
            cursor = self._db_conn.cursor()
            cursor.execute('''
                SELECT key, value, created_at, expires_at, access_count, 
                       last_accessed, size_bytes, tags, version
                FROM cache_entries
                WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)
            ''', (key, time.time()))
            
            row = cursor.fetchone()
            if row:
                value = json.loads(row[1])
                return CacheEntry(
                    key=row[0],
                    value=value,
                    created_at=row[2],
                    expires_at=row[3],
                    access_count=row[4],
                    last_accessed=row[5],
                    size_bytes=row[6],
                    tags=json.loads(row[7]) if row[7] else [],
                    version=row[8]
                )
        except Exception as e:
            logger.error(f"Failed to load entry from database: {e}")
        
        return None
    
    def set(
        self,
        operation: str,
        params: Dict[str, Any],
        value: Any,
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Store result in cache.
        
        Args:
            operation: Operation name
            params: Operation parameters
            value: Result value
            ttl: Time to live in seconds (None for no expiration)
            tags: Tags for grouping/invalidation
        """
        key = self._generate_key(operation, params)
        
        # Convert dataclasses to dict if needed
        import dataclasses
        if hasattr(value, 'to_dict'):
            serializable_value = value.to_dict()
        elif dataclasses.is_dataclass(value):
            serializable_value = dataclasses.asdict(value)
        else:
            serializable_value = value

        # Calculate size
        try:
            size = len(json.dumps(serializable_value))
        except:
            size = 0
        
        expires_at = None
        if ttl is not None:
            expires_at = time.time() + ttl
        elif self.config.default_ttl > 0:
            expires_at = time.time() + self.config.default_ttl
        
        entry = CacheEntry(
            key=key,
            value=serializable_value,
            expires_at=expires_at,
            size_bytes=size,
            tags=tags or []
        )
        
        with self._lock:
            # Store in Redis if enabled
            if self._use_valkey and self._valkey_backend:
                if not self._valkey_backend.set(entry):
                    # Redis failed, fall back to SQLite
                    logger.warning("Redis set failed, falling back to SQLite")
                    self._save_to_db(entry)
                else:
                    # Also update local cache for consistency
                    self._cache[key] = entry
                    self._stats.total_size_bytes += size
                    return
            
            # Check if we need to evict (only for local cache)
            if len(self._cache) >= self.config.max_size and key not in self._cache:
                self._evict_one()
            
            # Store entry
            old_entry = self._cache.get(key)
            if old_entry:
                self._stats.total_size_bytes -= old_entry.size_bytes
            
            self._cache[key] = entry
            self._stats.total_size_bytes += size
            self._stats.entry_count = len(self._cache)
            
            # Save to database (SQLite fallback or secondary storage)
            if self._db_conn:
                self._save_to_db(entry)
    
    def invalidate(
        self,
        operation: Optional[str] = None,
        tags: Optional[List[str]] = None,
        older_than: Optional[float] = None
    ) -> int:
        """
        Invalidate cache entries.
        
        Args:
            operation: Invalidate specific operation
            tags: Invalidate entries with any of these tags
            older_than: Invalidate entries older than timestamp
            
        Returns:
            Number of entries invalidated
        """
        invalidated_count = 0
        
        with self._lock:
            # Invalidate in Redis if enabled
            if self._use_valkey and self._valkey_backend and tags:
                valkey_invalidated = self._valkey_backend.invalidate_by_tags(tags)
                invalidated_count += valkey_invalidated
            
            # Invalidate in local cache and SQLite
            to_remove = []
            
            for key, entry in self._cache.items():
                should_remove = False
                
                if operation and key.startswith(operation):
                    should_remove = True
                
                if tags and any(tag in entry.tags for tag in tags):
                    should_remove = True
                
                if older_than and entry.created_at < older_than:
                    should_remove = True
                
                if should_remove:
                    to_remove.append(key)
            
            for key in to_remove:
                self._evict_entry(key)
            
            invalidated_count += len(to_remove)
            return invalidated_count
    
    def _evict_one(self):
        """Evict single entry based on policy."""
        if not self._cache:
            return
        
        key_to_evict = None
        
        if self.config.policy == CachePolicy.LRU:
            # Evict least recently used (first in OrderedDict)
            key_to_evict = next(iter(self._cache))
        
        elif self.config.policy == CachePolicy.LFU:
            # Evict least frequently used
            key_to_evict = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
        
        elif self.config.policy == CachePolicy.FIFO:
            # Evict oldest
            key_to_evict = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        
        else:  # TTL
            # Evict soonest to expire
            now = time.time()
            expiring = [
                (k, e.expires_at) for k, e in self._cache.items()
                if e.expires_at is not None
            ]
            if expiring:
                key_to_evict = min(expiring, key=lambda x: x[1])[0]
            else:
                key_to_evict = next(iter(self._cache))
        
        if key_to_evict:
            self._evict_entry(key_to_evict)
            self._stats.evictions += 1
    
    def _evict_entry(self, key: str):
        """Remove entry from cache and database."""
        entry = self._cache.pop(key, None)
        
        if entry:
            self._stats.total_size_bytes -= entry.size_bytes
            self._stats.entry_count = len(self._cache)
            
            # Remove from database
            if self._db_conn:
                try:
                    cursor = self._db_conn.cursor()
                    cursor.execute(
                        "DELETE FROM cache_entries WHERE key = ?",
                        (key,)
                    )
                    self._db_conn.commit()
                except Exception as e:
                    logger.error(f"Failed to remove from database: {e}")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.entry_count = len(self._cache)
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                total_size_bytes=self._stats.total_size_bytes,
                entry_count=self._stats.entry_count,
                hit_rate=self._stats.hit_rate
            )
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            # Clear Redis if enabled
            if self._use_valkey and self._valkey_backend:
                self._valkey_backend.clear()
            
            self._cache.clear()
            self._stats = CacheStats()
            
            if self._db_conn:
                try:
                    cursor = self._db_conn.cursor()
                    cursor.execute("DELETE FROM cache_entries")
                    self._db_conn.commit()
                except Exception as e:
                    logger.error(f"Failed to clear database: {e}")

    def __del__(self):
        """Close database and Redis connections."""
        if hasattr(self, '_db_conn') and self._db_conn:
            try:
                self._db_conn.close()
            except:
                pass
        
        if hasattr(self, '_valkey_backend') and self._valkey_backend:
            try:
                self._valkey_backend.close()
            except:
                pass
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        with self._lock:
            info = {
                "stats": self._stats.to_dict(),
                "config": {
                    "max_size": self.config.max_size,
                    "policy": self.config.policy.value,
                    "persistent": self.config.persistent_storage,
                    "distributed": self.config.distributed,
                    "valkey_host": self.config.valkey_host,
                    "valkey_port": self.config.valkey_port
                },
                "backend": {
                    "valkey_enabled": self._use_valkey,
                    "valkey_available": self._valkey_backend.is_available() if self._valkey_backend else False,
                    "sqlite_enabled": self._db_conn is not None
                },
                "entries": [
                    entry.to_dict() for entry in list(self._cache.values())[:10]
                ]
            }
            
            # Add Redis stats if available
            if self._use_valkey and self._valkey_backend:
                info["valkey_stats"] = self._valkey_backend.get_stats()
            
            return info


# =============================================================================
# Decorator for Caching
# =============================================================================

class Cached:
    """Decorator for caching function results."""
    
    def __init__(
        self,
        cache: Optional[Z3ResultCache] = None,
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
        key_fn: Optional[Callable] = None
    ):
        self.cache = cache or get_z3_result_cache()
        self.ttl = ttl
        self.tags = tags
        self.key_fn = key_fn
    
    def __call__(self, func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            # Generate key
            if self.key_fn:
                key_data = self.key_fn(*args, **kwargs)
            else:
                key_data = {
                    "args": args,
                    "kwargs": kwargs
                }
            
            # Try cache
            hit, value = self.cache.get(func.__name__, key_data)
            
            if hit:
                return value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            self.cache.set(
                func.__name__,
                key_data,
                result,
                ttl=self.ttl,
                tags=self.tags
            )
            
            return result
        
        def sync_wrapper(*args, **kwargs):
            # Similar logic for sync functions
            if self.key_fn:
                key_data = self.key_fn(*args, **kwargs)
            else:
                key_data = {"args": args, "kwargs": kwargs}
            
            hit, value = self.cache.get(func.__name__, key_data)
            
            if hit:
                return value
            
            result = func(*args, **kwargs)
            
            self.cache.set(
                func.__name__,
                key_data,
                result,
                ttl=self.ttl,
                tags=self.tags
            )
            
            return result
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            async_wrapper.__name__ = func.__name__
            return async_wrapper
        else:
            sync_wrapper.__name__ = func.__name__
            return sync_wrapper


# =============================================================================
# Global Cache Instance
# =============================================================================

_result_cache: Optional[Z3ResultCache] = None


def get_z3_result_cache(config: Optional[CacheConfig] = None) -> Z3ResultCache:
    """Get global result cache instance."""
    global _result_cache
    if _result_cache is None:
        _result_cache = Z3ResultCache(config)
    return _result_cache


# =============================================================================
# Example Usage
# =============================================================================

def example_basic_caching():
    """Example: Basic caching."""
    cache = get_z3_result_cache()
    
    # Store result
    cache.set(
        "solve",
        {"x": 5, "constraints": ["x > 0"]},
        {"solution": {"x": 1}, "status": "sat"},
        ttl=3600,
        tags=["constraint", "simple"]
    )
    
    # Retrieve result
    hit, value = cache.get("solve", {"x": 5, "constraints": ["x > 0"]})
    
    print(f"Cache hit: {hit}")
    print(f"Value: {value}")
    
    # Get stats
    stats = cache.get_stats()
    print(f"\nCache stats: {stats.to_dict()}")


def example_cache_invalidation():
    """Example: Cache invalidation."""
    cache = get_z3_result_cache()
    
    # Add entries with tags
    cache.set("op1", {"a": 1}, "result1", tags=["tag1"])
    cache.set("op2", {"b": 2}, "result2", tags=["tag2"])
    cache.set("op3", {"c": 3}, "result3", tags=["tag1", "tag2"])
    
    print(f"Before invalidation: {cache.get_stats().entry_count} entries")
    
    # Invalidate by tag
    invalidated = cache.invalidate(tags=["tag1"])
    
    print(f"Invalidated {invalidated} entries")
    print(f"After invalidation: {cache.get_stats().entry_count} entries")


if __name__ == "__main__":
    print("Z3 Result Cache")
    print("=" * 50)
    
    example_basic_caching()
    print("\n" + "=" * 50)
    example_cache_invalidation()
