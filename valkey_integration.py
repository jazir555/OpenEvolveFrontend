#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Valkey Integration for OpenEvolve

This module provides Valkey-based storage for the OpenEvolve PES integration.
Valkey is a Redis fork maintained by the Linux Foundation with a permissive license.

Features:
- Redis-compatible API (Valkey is a drop-in replacement)
- Connection pooling
- Async support
- Automatic reconnection
- Distributed lock support

Author: OpenEvolve
Created: 2026-02-02
License: Apache-2.0 (Valkey) / MIT (this module)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from contextlib import asynccontextmanager


logger = logging.getLogger(__name__)


# =============================================================================
# Import Valkey or Fallback to Redis
# =============================================================================

VALKEY_AVAILABLE = False
valkey = None
ConnectionPool = None
Redis = None

try:
    # Try to import valkey (the new Redis-compatible library)
    import valkey
    VALKEY_AVAILABLE = True
    logger.info("Valkey library imported successfully")
except ImportError:
    try:
        # Fallback to redis-py with Valkey compatibility
        import redis
        from redis.connection import ConnectionPool
        from redis.asyncio import Redis as AsyncRedis
        
        # Create a Valkey-compatible wrapper
        valkey = redis
        VALKEY_AVAILABLE = True
        ConnectionPool = ConnectionPool
        Redis = AsyncRedis
        logger.info("Using redis-py as Valkey-compatible backend")
    except ImportError:
        logger.warning("Neither valkey nor redis-py available")
        logger.warning("Install valkey: pip install valkey")
        logger.warning("Or use redis-py: pip install redis")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ValkeyConfig:
    """Configuration for Valkey connection."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    ssl_cert_reqs: Optional[str] = None
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    decode_responses: bool = True
    
    # Valkey-specific options
    valkey_mode: bool = True  # Use Valkey-specific commands if available
    
    @property
    def url(self) -> str:
        """Generate connection URL."""
        scheme = "valkey" if self.valkey_mode else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ValkeyConfig":
        """Create config from dictionary."""
        return cls(
            host=config.get("host", "localhost"),
            port=config.get("port", 6379),
            db=config.get("db", 0),
            password=config.get("password"),
            ssl=config.get("ssl", False),
            max_connections=config.get("max_connections", 50),
            socket_timeout=config.get("socket_timeout", 5.0),
            socket_connect_timeout=config.get("socket_connect_timeout", 5.0),
        )


# =============================================================================
# Valkey Client Wrapper
# =============================================================================

class ValkeyClient:
    """
    Valkey client wrapper for OpenEvolve.
    
    Provides a simple interface for common operations:
    - Key-value storage
    - Hash operations
    - List operations
    - Pub/Sub
    - Distributed locks
    """
    
    def __init__(self, config: Optional[ValkeyConfig] = None):
        """
        Initialize Valkey client.
        
        Args:
            config: Valkey configuration (uses defaults if not provided)
        """
        self.config = config or ValkeyConfig()
        self._pool: Optional[ConnectionPool] = None
        self._client = None
        self._async_client = None
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize connection pool and client."""
        if not VALKEY_AVAILABLE:
            logger.warning("Valkey/Redis not available")
            return
        
        try:
            # Create connection pool
            self._pool = ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                ssl=self.config.ssl,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                decode_responses=self.config.decode_responses,
            )
            
            # Create sync client
            self._client = redis.Redis(connection_pool=self._pool)
            
            # Test connection
            self._client.ping()
            logger.info(f"Connected to Valkey at {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Valkey: {e}")
            self._client = None
    
    async def ainitialize(self) -> None:
        """Initialize async client."""
        if not VALKEY_AVAILABLE:
            return
        
        try:
            self._async_client = AsyncRedis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                ssl=self.config.ssl,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=self.config.decode_responses,
            )
            await self._async_client.ping()
            logger.info(f"Async Valkey connection established")
        except Exception as e:
            logger.error(f"Failed to connect to async Valkey: {e}")
            self._async_client = None
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        if not self._client:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            return False
    
    # =====================================================================
    # Key-Value Operations
    # =====================================================================
    
    def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        if not self._client:
            return None
        return self._client.get(key)
    
    def set(
        self,
        key: str,
        value: str,
        ex: Optional[int] = None,  # Expire in seconds
        px: Optional[int] = None,  # Expire in milliseconds
        nx: bool = False,  # Only set if not exists
        xx: bool = False,  # Only set if exists
    ) -> bool:
        """Set key-value pair."""
        if not self._client:
            return False
        return self._client.set(key, value, ex=ex, px=px, nx=nx, xx=xx)
    
    def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        if not self._client:
            return 0
        return self._client.delete(*keys)
    
    def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        if not self._client:
            return 0
        return self._client.exists(*keys)
    
    def expire(self, key: str, seconds: int) -> bool:
        """Set key expiration."""
        if not self._client:
            return False
        return self._client.expire(key, seconds)
    
    def ttl(self, key: str) -> int:
        """Get key TTL."""
        if not self._client:
            return -2
        return self._client.ttl(key)
    
    # =====================================================================
    # Hash Operations
    # =====================================================================
    
    def hget(self, name: str, key: str) -> Optional[str]:
        """Get field from hash."""
        if not self._client:
            return None
        return self._client.hget(name, key)
    
    def hset(self, name: str, key: str, value: str) -> int:
        """Set field in hash."""
        if not self._client:
            return 0
        return self._client.hset(name, key, value)
    
    def hgetall(self, name: str) -> Dict[str, str]:
        """Get all fields from hash."""
        if not self._client:
            return {}
        return self._client.hgetall(name) or {}
    
    def hdel(self, name: str, *keys: str) -> int:
        """Delete fields from hash."""
        if not self._client:
            return 0
        return self._client.hdel(name, *keys)
    
    # =====================================================================
    # List Operations
    # =====================================================================
    
    def lpush(self, name: str, *values: str) -> int:
        """Push values to list head."""
        if not self._client:
            return 0
        return self._client.lpush(name, *values)
    
    def rpush(self, name: str, *values: str) -> int:
        """Push values to list tail."""
        if not self._client:
            return 0
        return self._client.rpush(name, *values)
    
    def lpop(self, name: str) -> Optional[str]:
        """Pop value from list head."""
        if not self._client:
            return None
        return self._client.lpop(name)
    
    def rpop(self, name: str) -> Optional[str]:
        """Pop value from list tail."""
        if not self._client:
            return None
        return self._client.rpop(name)
    
    def lrange(self, name: str, start: int, end: int) -> List[str]:
        """Get range of values from list."""
        if not self._client:
            return []
        return self._client.lrange(name, start, end)
    
    def llen(self, name: str) -> int:
        """Get list length."""
        if not self._client:
            return 0
        return self._client.llen(name)
    
    # =====================================================================
    # Set Operations
    # =====================================================================
    
    def sadd(self, name: str, *values: str) -> int:
        """Add values to set."""
        if not self._client:
            return 0
        return self._client.sadd(name, *values)
    
    def srem(self, name: str, *values: str) -> int:
        """Remove values from set."""
        if not self._client:
            return 0
        return self._client.srem(name, *values)
    
    def smembers(self, name: str) -> set:
        """Get all set members."""
        if not self._client:
            return set()
        return self._client.smembers(name) or set()
    
    def sismember(self, name: str, value: str) -> bool:
        """Check if value is in set."""
        if not self._client:
            return False
        return self._client.sismember(name, value)
    
    # =====================================================================
    # Sorted Set Operations
    # =====================================================================
    
    def zadd(self, name: str, mapping: Dict[str, float], **kwargs) -> int:
        """Add members to sorted set."""
        if not self._client:
            return 0
        return self._client.zadd(name, mapping, **kwargs)
    
    def zrange(
        self,
        name: str,
        start: int,
        end: int,
        withscores: bool = False
    ) -> List[Union[str, Tuple[str, float]]]:
        """Get range from sorted set."""
        if not self._client:
            return []
        return self._client.zrange(name, start, end, withscores=withscores)
    
    def zscore(self, name: str, member: str) -> Optional[float]:
        """Get score of member."""
        if not self._client:
            return None
        return self._client.zscore(name, member)
    
    def zrem(self, name: str, *members: str) -> int:
        """Remove members from sorted set."""
        if not self._client:
            return 0
        return self._client.zrem(name, *members)
    
    # =====================================================================
    # Utility Operations
    # =====================================================================
    
    def ping(self) -> bool:
        """Check connection."""
        if not self._client:
            return False
        try:
            return self._client.ping()
        except Exception:
            return False
    
    def info(self) -> Dict[str, Any]:
        """Get server info."""
        if not self._client:
            return {}
        return self._client.info()
    
    def flushdb(self) -> bool:
        """Flush current database."""
        if not self._client:
            return False
        self._client.flushdb()
        return True
    
    # =====================================================================
    # Async Operations
    # =====================================================================
    
    async def aget(self, key: str) -> Optional[str]:
        """Async get value."""
        if not self._async_client:
            return None
        return await self._async_client.get(key)
    
    async def aset(
        self,
        key: str,
        value: str,
        ex: Optional[int] = None,
        px: Optional[int] = None,
    ) -> bool:
        """Async set value."""
        if not self._async_client:
            return False
        return await self._async_client.set(key, value, ex=ex, px=px)
    
    async def adelete(self, *keys: str) -> int:
        """Async delete keys."""
        if not self._async_client:
            return 0
        return await self._async_client.delete(*keys)
    
    async def aclose(self) -> None:
        """Close async connection."""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None
    
    def close(self) -> None:
        """Close sync connection."""
        if self._pool:
            self._pool.disconnect()
            self._pool = None
            self._client = None


# =============================================================================
# Distributed Lock
# =============================================================================

class ValkeyLock:
    """
    Distributed lock using Valkey.
    
    Features:
    - Automatic expiration
    - Non-blocking acquire with timeout
    - Safe release (only if we own the lock)
    """
    
    def __init__(
        self,
        client: ValkeyClient,
        name: str,
        timeout: float = 10.0,
        id: Optional[str] = None
    ):
        """
        Initialize lock.
        
        Args:
            client: Valkey client
            name: Lock name
            timeout: Lock timeout in seconds
            id: Lock identifier (generated if not provided)
        """
        self.client = client
        self.name = f"lock:{name}"
        self.timeout = int(timeout)
        self.id = id or f"{time.time()}:{hex(id(self))}"  # Use object id for uniqueness
        self._acquired = False
    
    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        """
        Acquire the lock.
        
        Args:
            blocking: Block until acquired
            timeout: Max time to wait (seconds, -1 for infinite)
            
        Returns:
            True if acquired, False otherwise
        """
        if not self.client._client:
            return False
        
        import time as time_module
        start_time = time_module.time()
        
        while True:
            # Try to acquire lock
            result = self.client._client.set(self.name, self.id, nx=True, ex=self.timeout)
            
            if result:
                self._acquired = True
                logger.debug(f"Acquired lock: {self.name}")
                return True
            
            if not blocking:
                return False
            
            # Check timeout
            if timeout > 0 and (time_module.time() - start_time) > timeout:
                return False
            
            # Wait a bit before retry
            time.sleep(0.01)
    
    def release(self) -> bool:
        """
        Release the lock.
        
        Only releases if we own the lock (id matches).
        
        Returns:
            True if released, False otherwise
        """
        if not self._acquired or not self.client._client:
            return False
        
        # Lua script for atomic check-and-delete
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        
        result = self.client._client.eval(script, 1, self.name, self.id)
        self._acquired = False
        
        if result:
            logger.debug(f"Released lock: {self.name}")
        
        return bool(result)
    
    def extend(self, additional_time: float) -> bool:
        """
        Extend lock timeout.
        
        Only works if we own the lock.
        
        Args:
            additional_time: Additional time in seconds
            
        Returns:
            True if extended, False otherwise
        """
        if not self._acquired or not self.client._client:
            return False
        
        # Lua script for atomic check-and-expire
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("expire", KEYS[1], ARGV[2])
        else
            return 0
        end
        """
        
        result = self.client._client.eval(script, 1, self.name, self.id, additional_time)
        return bool(result)
    
    def __enter__(self):
        """Context manager enter."""
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
    
    async def __aenter__(self):
        """Async context manager enter."""
        self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.release()
        return False


# =============================================================================
# Valkey-based Evolution Database
# =============================================================================

class ValkeyEvolutionDatabase:
    """
    Evolution database using Valkey for storage.
    
    Provides:
    - Population storage with sorted sets
    - Archive management
    - Solution history
    - Checkpoint support
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize database.
        
        Args:
            config: Database configuration with valkey settings
        """
        self.config = config
        self.valkey_config = ValkeyConfig.from_dict(config.get("valkey", {}))
        
        # Keys
        self.prefix = config.get("key_prefix", "evolution")
        self.population_key = f"{self.prefix}:population"
        self.archive_key = f"{self.prefix}:archive"
        self.history_key = f"{self.prefix}:history"
        self.best_key = f"{self.prefix}:best"
        self.iteration_key = f"{self.prefix}:iteration"
        
        # Initialize client
        self.client = ValkeyClient(self.valkey_config)
        
        # Population settings
        self.max_population = config.get("population_size", 100)
        self.max_archive = config.get("archive_size", 1000)
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to Valkey."""
        return self.client.is_connected
    
    async def connect(self) -> bool:
        """Connect to Valkey."""
        await self.client.ainitialize()
        return self.is_connected
    
    async def save_solution(
        self,
        solution_id: str,
        iteration: int,
        solution: str,
        score: float,
        fitness: float,
        features: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save a solution to the database.
        
        Uses sorted set for population with score as ordering.
        """
        if not self.is_connected:
            logger.warning("Not connected to Valkey")
            return False
        
        try:
            data = {
                "id": solution_id,
                "iteration": iteration,
                "solution": solution,
                "score": score,
                "fitness": fitness,
                "features": json.dumps(features),
                "metadata": json.dumps(metadata or {}),
                "timestamp": time.time()
            }
            
            # Store solution data as hash
            solution_key = f"{self.prefix}:solution:{solution_id}"
            for key, value in data.items():
                self.client.hset(solution_key, key, str(value))
            
            # Add to population sorted set (score as member for ordering)
            self.client.zadd(self.population_key, {solution_id: score})
            
            # Trim population to max size
            population = self.client.zrange(self.population_key, 0, -1)
            if len(population) > self.max_population:
                # Remove worst solutions
                to_remove = population[:-self.max_population]
                self.client.zrem(self.population_key, *to_remove)
            
            # Update best solution
            current_best = self.get_best_solution_id()
            if current_best is None or score > self.client.zscore(self.population_key, current_best):
                self.client.set(self.best_key, solution_id)
            
            # Update iteration counter
            current_iter = self.get_iteration()
            if iteration > current_iter:
                self.client.set(self.iteration_key, str(iteration))
            
            logger.debug(f"Saved solution {solution_id[:8]} with score {score:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save solution: {e}")
            return False
    
    async def get_best_solution(self) -> Optional[Dict[str, Any]]:
        """Get the best solution."""
        if not self.is_connected:
            return None
        
        best_id = self.client.get(self.best_key)
        if not best_id:
            # Get from population
            population = self.client.zrange(self.population_key, -1, -1, withscores=True)
            if population:
                best_id, score = population[0]
            else:
                return None
        
        return await self.get_solution(best_id)
    
    async def get_solution(self, solution_id: str) -> Optional[Dict[str, Any]]:
        """Get a solution by ID."""
        if not self.is_connected:
            return None
        
        solution_key = f"{self.prefix}:solution:{solution_id}"
        data = self.client.hgetall(solution_key)
        
        if not data:
            return None
        
        return {
            "id": solution_id,
            "iteration": int(data.get("iteration", 0)),
            "solution": data.get("solution", ""),
            "score": float(data.get("score", 0.0)),
            "fitness": float(data.get("fitness", 0.0)),
            "features": json.loads(data.get("features", "{}")),
            "metadata": json.loads(data.get("metadata", "{}")),
            "timestamp": float(data.get("timestamp", 0.0))
        }
    
    async def get_population(self, iteration: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get population solutions."""
        if not self.is_connected:
            return []
        
        population = self.client.zrange(self.population_key, 0, -1)
        solutions = []
        
        for sol_id in population:
            sol = await self.get_solution(sol_id)
            if sol and (iteration is None or sol["iteration"] == iteration):
                solutions.append(sol)
        
        return solutions
    
    def get_best_solution_id(self) -> Optional[str]:
        """Get best solution ID."""
        if not self.is_connected:
            return None
        return self.client.get(self.best_key)
    
    def get_iteration(self) -> int:
        """Get current iteration."""
        if not self.is_connected:
            return 0
        val = self.client.get(self.iteration_key)
        return int(val) if val else 0
    
    async def save_checkpoint(self, checkpoint_dir: str) -> bool:
        """
        Save checkpoint.
        
        For Valkey, we export data to files for persistence.
        """
        if not self.is_connected:
            return False
        
        try:
            import os
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Export population
            population = self.client.zrange(self.population_key, 0, -1, withscores=True)
            checkpoint_data = {
                "population": population,
                "best_id": self.client.get(self.best_key),
                "iteration": self.get_iteration(),
                "timestamp": time.time()
            }
            
            checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.json")
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Export solutions
            for sol_id, _ in population:
                sol = await self.get_solution(sol_id)
                if sol:
                    sol_file = os.path.join(checkpoint_dir, f"solution_{sol_id[:8]}.json")
                    with open(sol_file, "w") as f:
                        json.dump(sol, f, indent=2)
            
            logger.info(f"Checkpoint saved to {checkpoint_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    async def load_checkpoint(self, checkpoint_dir: str) -> bool:
        """Load checkpoint."""
        if not self.is_connected:
            return False
        
        try:
            checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.json")
            if not os.path.exists(checkpoint_file):
                logger.warning(f"Checkpoint file not found: {checkpoint_file}")
                return False
            
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
            
            # Restore population
            for sol_id, score in checkpoint_data.get("population", []):
                self.client.zadd(self.population_key, {sol_id: score})
            
            # Restore best
            best_id = checkpoint_data.get("best_id")
            if best_id:
                self.client.set(self.best_key, best_id)
            
            # Restore iteration
            self.client.set(self.iteration_key, str(checkpoint_data.get("iteration", 0)))
            
            logger.info(f"Checkpoint loaded from {checkpoint_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def memory_status(self) -> Dict[str, Any]:
        """Get memory system status."""
        if not self.is_connected:
            return {"global_status": {"population_size": 0, "best_score": 0.0, "current_iteration": 0}}
        
        population_size = self.client.zcard(self.population_key) if hasattr(self.client, 'zcard') else len(self.client.zrange(self.population_key, 0, -1))
        
        best_score = 0.0
        best = self.client.zrange(self.population_key, -1, -1, withscores=True)
        if best:
            best_score = best[0][1]
        
        return {
            "global_status": {
                "population_size": population_size,
                "best_score": best_score,
                "current_iteration": self.get_iteration()
            }
        }
    
    async def close(self) -> None:
        """Close database connection."""
        await self.client.aclose()
        self.client.close()


# =============================================================================
# Factory Functions
# =============================================================================

def create_valkey_client(
    host: str = "localhost",
    port: int = 6379,
    password: Optional[str] = None,
    **kwargs
) -> ValkeyClient:
    """
    Create a Valkey client.
    
    Args:
        host: Server host
        port: Server port
        password: Authentication password
        **kwargs: Additional configuration
        
    Returns:
        ValkeyClient instance
    """
    config = ValkeyConfig(
        host=host,
        port=port,
        password=password,
        **kwargs
    )
    return ValkeyClient(config)


def create_valkey_database(config: Dict[str, Any]) -> ValkeyEvolutionDatabase:
    """
    Create a Valkey-based evolution database.
    
    Args:
        config: Database configuration
        
    Returns:
        ValkeyEvolutionDatabase instance
    """
    return ValkeyEvolutionDatabase(config)


# =============================================================================
# Standalone Usage
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test_valkey():
        """Test Valkey integration."""
        print("Testing Valkey Integration...")
        
        # Create client
        client = create_valkey_client()
        print(f"Connected: {client.is_connected}")
        
        # Test operations
        client.set("test_key", "test_value")
        value = client.get("test_key")
        print(f"Get test_key: {value}")
        
        # Test lock
        with ValkeyLock(client, "test_lock", timeout=5.0) as lock:
            print(f"Lock acquired: {lock._acquired}")
        
        # Test database
        db_config = {
            "population_size": 100,
            "archive_size": 1000,
            "key_prefix": "test_evolution",
            "valkey": {"host": "localhost", "port": 6379}
        }
        
        db = create_valkey_database(db_config)
        print(f"Database connected: {await db.connect()}")
        
        # Save solution
        await db.save_solution(
            solution_id="test_solution_001",
            iteration=1,
            solution="def test(): pass",
            score=0.8,
            fitness=0.8,
            features={"complexity": 1.0},
            metadata={"source": "test"}
        )
        
        # Get best
        best = await db.get_best_solution()
        print(f"Best solution: {best['id'] if best else None}")
        
        await db.close()
        print("Test complete!")
    
    asyncio.run(test_valkey())
