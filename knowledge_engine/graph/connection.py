"""
Neo4j Connection Pool with Retry Logic

Handles connection pooling, retry policies, and fault tolerance.

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

import asyncio
import logging
import time
from typing import Optional, Callable, Any, List, Dict
from dataclasses import dataclass, field
from enum import Enum, auto
from contextlib import asynccontextmanager
import threading

logger = logging.getLogger(__name__)

# Try to import Neo4j driver
try:
    from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
    from neo4j.exceptions import (
        ServiceUnavailable, AuthError, ClientError,
        TransientError, DatabaseError
    )
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    AsyncGraphDatabase = None
    AsyncDriver = None
    AsyncSession = None
    ServiceUnavailable = Exception
    AuthError = Exception
    ClientError = Exception
    TransientError = Exception
    DatabaseError = Exception


class ConnectionState(Enum):
    """Connection state"""
    IDLE = auto()
    IN_USE = auto()
    CLOSED = auto()
    ERROR = auto()


@dataclass
class RetryPolicy:
    """Retry policy configuration"""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retryable_exceptions: tuple = field(default_factory=lambda: (
        ServiceUnavailable, TransientError, ConnectionError, TimeoutError
    ))
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        delay = self.initial_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)


@dataclass
class ConnectionConfig:
    """Neo4j connection configuration"""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    max_connections: int = 10
    connection_timeout: float = 30.0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)


class PooledConnection:
    """A pooled connection wrapper"""
    
    def __init__(self, connection_id: str, driver: Optional[Any] = None):
        self.id = connection_id
        self.driver = driver
        self.state = ConnectionState.IDLE
        self.created_at = time.time()
        self.last_used = time.time()
        self.use_count = 0
        self.error_count = 0
        self._lock = threading.RLock()
    
    def acquire(self) -> bool:
        """Try to acquire the connection"""
        with self._lock:
            if self.state == ConnectionState.IDLE:
                self.state = ConnectionState.IN_USE
                self.last_used = time.time()
                self.use_count += 1
                return True
            return False
    
    def release(self):
        """Release the connection back to pool"""
        with self._lock:
            self.state = ConnectionState.IDLE
            self.last_used = time.time()
    
    def mark_error(self):
        """Mark connection as having an error"""
        with self._lock:
            self.error_count += 1
            if self.error_count >= 3:
                self.state = ConnectionState.ERROR
    
    def close(self):
        """Close the connection"""
        with self._lock:
            if self.driver and hasattr(self.driver, 'close'):
                try:
                    asyncio.create_task(self.driver.close())
                except:
                    pass
            self.state = ConnectionState.CLOSED
    
    @property
    def is_healthy(self) -> bool:
        """Check if connection is healthy"""
        with self._lock:
            return self.state in (ConnectionState.IDLE, ConnectionState.IN_USE) and self.error_count < 3
    
    @property
    def idle_time(self) -> float:
        """Get idle time in seconds"""
        with self._lock:
            if self.state == ConnectionState.IDLE:
                return time.time() - self.last_used
            return 0.0


class ConnectionPool:
    """Neo4j connection pool with retry logic"""
    
    def __init__(self, config: Optional[ConnectionConfig] = None):
        self.config = config or ConnectionConfig()
        self._pool: List[PooledConnection] = []
        self._lock = threading.RLock()
        self._closed = False
        self._driver: Optional[Any] = None
        self._metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "queries_executed": 0,
            "queries_failed": 0,
            "retries": 0,
        }
    
    async def initialize(self) -> bool:
        """Initialize the connection pool"""
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j driver not available, using mock implementation")
            return True
        
        try:
            self._driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password),
                connection_timeout=self.config.connection_timeout,
                max_connection_pool_size=self.config.max_connections
            )
            
            # Verify connection
            await self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.config.uri}")
            
            # Create initial connections
            for i in range(min(3, self.config.max_connections)):
                await self._create_connection()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j connection: {e}")
            self._metrics["failed_connections"] += 1
            return False
    
    async def _create_connection(self) -> Optional[PooledConnection]:
        """Create a new pooled connection"""
        if not NEO4J_AVAILABLE or not self._driver:
            # Mock connection
            conn = PooledConnection(f"mock_{len(self._pool)}")
            with self._lock:
                self._pool.append(conn)
                self._metrics["total_connections"] += 1
            return conn
        
        try:
            conn_id = f"conn_{len(self._pool)}"
            conn = PooledConnection(conn_id, self._driver)
            
            with self._lock:
                self._pool.append(conn)
                self._metrics["total_connections"] += 1
            
            return conn
            
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            self._metrics["failed_connections"] += 1
            return None
    
    async def acquire(self) -> Optional[PooledConnection]:
        """Acquire a connection from the pool"""
        with self._lock:
            if self._closed:
                return None
            
            # Try to find idle connection
            for conn in self._pool:
                if conn.acquire():
                    self._metrics["active_connections"] += 1
                    return conn
            
            # Create new connection if under limit
            if len(self._pool) < self.config.max_connections:
                conn = await self._create_connection()
                if conn and conn.acquire():
                    self._metrics["active_connections"] += 1
                    return conn
        
        # Wait and retry
        await asyncio.sleep(0.1)
        return await self.acquire()
    
    def release(self, conn: PooledConnection):
        """Release a connection back to the pool"""
        conn.release()
        with self._lock:
            self._metrics["active_connections"] = max(0, self._metrics["active_connections"] - 1)
    
    @asynccontextmanager
    async def get_connection(self):
        """Context manager for acquiring/releasing connections"""
        conn = await self.acquire()
        if not conn:
            raise ConnectionError("Could not acquire connection from pool")
        
        try:
            yield conn
        finally:
            self.release(conn)
    
    async def execute_with_retry(
        self,
        query_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute a query with retry logic"""
        policy = self.config.retry_policy
        last_exception = None
        
        for attempt in range(policy.max_retries + 1):
            try:
                async with self.get_connection() as conn:
                    if NEO4J_AVAILABLE and conn.driver:
                        async with conn.driver.session(database=self.config.database) as session:
                            result = await query_func(session, *args, **kwargs)
                            self._metrics["queries_executed"] += 1
                            return result
                    else:
                        # Mock execution
                        return await query_func(None, *args, **kwargs)
                        
            except policy.retryable_exceptions as e:
                last_exception = e
                self._metrics["retries"] += 1
                
                if attempt < policy.max_retries:
                    delay = policy.get_delay(attempt)
                    logger.warning(f"Query failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Query failed after {policy.max_retries + 1} attempts: {e}")
            
            except Exception as e:
                # Non-retryable error
                logger.error(f"Query failed with non-retryable error: {e}")
                self._metrics["queries_failed"] += 1
                raise
        
        self._metrics["queries_failed"] += 1
        raise last_exception or ConnectionError("Max retries exceeded")
    
    async def run_cypher(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute a Cypher query"""
        async def _execute(session, query, parameters):
            if session is None:
                # Mock execution
                logger.debug(f"Mock execute: {query[:100]}...")
                return []
            
            result = await session.run(query, parameters or {})
            records = []
            async for record in result:
                records.append(dict(record))
            return records
        
        return await self.execute_with_retry(_execute, query, parameters)
    
    async def run_cypher_write(self, query: str, parameters: Optional[Dict] = None) -> Dict:
        """Execute a write Cypher query"""
        async def _execute(session, query, parameters):
            if session is None:
                # Mock execution
                logger.debug(f"Mock write: {query[:100]}...")
                return {"created": 0, "updated": 0, "deleted": 0}
            
            result = await session.run(query, parameters or {})
            summary = await result.consume()
            return {
                "created": summary.counters.nodes_created,
                "updated": summary.counters.properties_set,
                "deleted": summary.counters.nodes_deleted,
                "relationships_created": summary.counters.relationships_created,
            }
        
        return await self.execute_with_retry(_execute, query, parameters)
    
    async def health_check(self) -> bool:
        """Check if the connection pool is healthy"""
        if not NEO4J_AVAILABLE:
            return True  # Mock is always healthy
        
        try:
            async with self.get_connection() as conn:
                if conn.driver:
                    await conn.driver.verify_connectivity()
                    return True
                return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pool metrics"""
        with self._lock:
            metrics = self._metrics.copy()
            metrics["pool_size"] = len(self._pool)
            metrics["available_connections"] = sum(
                1 for c in self._pool if c.state == ConnectionState.IDLE
            )
            return metrics
    
    async def close(self):
        """Close all connections in the pool"""
        with self._lock:
            self._closed = True
            
            for conn in self._pool:
                conn.close()
            
            if self._driver and NEO4J_AVAILABLE:
                await self._driver.close()
            
            self._pool.clear()
        
        logger.info("Connection pool closed")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
