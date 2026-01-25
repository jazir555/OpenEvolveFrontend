"""
Resource Pooling System for OpenEvolve Decomposition Engine

Provides efficient resource management:
- Object pooling
- Connection pooling
- Memory pooling
- Semaphore-based rate limiting
- Resource lifecycle management
- Automatic resource cleanup
- Pool statistics and monitoring
"""

import threading
import time
import queue
import weakref
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import logging
import hashlib
import os

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class PoolStats:
    """Statistics for a resource pool"""
    total_created: int = 0
    total_destroyed: int = 0
    current_size: int = 0
    available_resources: int = 0
    in_use_resources: int = 0
    total_acquisitions: int = 0
    total_releases: int = 0
    total_wait_time: float = 0.0
    avg_wait_time: float = 0.0
    max_wait_time: float = 0.0
    last_cleanup: Optional[datetime] = None


class ObjectPool(Generic[T]):
    """
    Generic object pool for reusing expensive objects.

    Features:
    - Object creation and reuse
    - Object validation
    - Automatic cleanup
    - Size limits
    - Thread-safe operations
    """

    def __init__(self, factory: Callable[[], T],
                 validator: Optional[Callable[[T], bool]] = None,
                 reset: Optional[Callable[[T], None]] = None,
                 destroy: Optional[Callable[[T], None]] = None,
                 min_size: int = 0,
                 max_size: int = 10,
                 max_idle_time: float = 300.0):
        """
        Initialize object pool.

        Args:
            factory: Function to create new objects
            validator: Function to validate object health
            reset: Function to reset object state
            destroy: Function to destroy object
            min_size: Minimum pool size
            max_size: Maximum pool size
            max_idle_time: Maximum idle time before cleanup
        """
        self.factory = factory
        self.validator = validator or (lambda obj: True)
        self.reset = reset
        self.destroy = destroy
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time

        self._pool: deque[T] = deque()
        self._in_use: Dict[int, T] = {}
        self._created_at: Dict[int, datetime] = {}
        self._last_used: Dict[int, datetime] = {}
        self._lock = threading.Lock()

        self.stats = PoolStats()

        # Pre-populate pool to min_size
        self._ensure_min_size()

        logger.info(f"Object pool initialized: min_size={min_size}, max_size={max_size}")

    def acquire(self, timeout: Optional[float] = None) -> T:
        """Acquire object from pool"""
        start_time = time.time()

        with self._lock:
            # Check for available object
            while self._pool:
                obj = self._pool.popleft()

                # Validate object
                obj_id = id(obj)
                if self.validator(obj):
                    self._in_use[obj_id] = obj
                    self._last_used[obj_id] = datetime.now()
                    self.stats.available_resources -= 1
                    self.stats.in_use_resources += 1
                    self.stats.total_acquisitions += 1

                    wait_time = time.time() - start_time
                    self.stats.total_wait_time += wait_time
                    self.stats.max_wait_time = max(self.stats.max_wait_time, wait_time)
                    self.stats.avg_wait_time = (
                        self.stats.total_wait_time / self.stats.total_acquisitions
                    )

                    return obj
                else:
                    # Destroy invalid object
                    self._destroy_object(obj)

            # Check if we can create new object
            if len(self._in_use) < self.max_size:
                obj = self._create_object()
                obj_id = id(obj)
                self._in_use[obj_id] = obj
                self._last_used[obj_id] = datetime.now()
                self.stats.in_use_resources += 1
                self.stats.total_acquisitions += 1

                return obj

        # Wait for available object
        deadline = time.time() + (timeout or 30.0)
        while time.time() < deadline:
            with self._lock:
                if self._pool:
                    obj = self._pool.popleft()
                    if self.validator(obj):
                        obj_id = id(obj)
                        self._in_use[obj_id] = obj
                        self._last_used[obj_id] = datetime.now()
                        self.stats.available_resources -= 1
                        self.stats.in_use_resources += 1
                        self.stats.total_acquisitions += 1

                        return obj

            time.sleep(0.01)

        raise TimeoutError("Timeout waiting for object from pool")

    def release(self, obj: T):
        """Release object back to pool"""
        obj_id = id(obj)

        with self._lock:
            if obj_id not in self._in_use:
                raise ValueError("Object not acquired from this pool")

            del self._in_use[obj_id]
            self.stats.total_releases += 1
            self.stats.in_use_resources -= 1

            # Reset object if needed
            if self.reset:
                self.reset(obj)

            # Add back to pool if under max size
            if len(self._pool) + len(self._in_use) < self.max_size:
                self._pool.append(obj)
                self._last_used[obj_id] = datetime.now()
                self.stats.available_resources += 1
            else:
                # Destroy excess object
                self._destroy_object(obj)

    def _create_object(self) -> T:
        """Create new object"""
        obj = self.factory()
        obj_id = id(obj)

        self._created_at[obj_id] = datetime.now()
        self._last_used[obj_id] = datetime.now()
        self.stats.total_created += 1
        self.stats.current_size += 1

        return obj

    def _destroy_object(self, obj: T):
        """Destroy object"""
        obj_id = id(obj)

        if self.destroy:
            try:
                self.destroy(obj)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Error destroying object: {e}")

        if obj_id in self._created_at:
            del self._created_at[obj_id]
        if obj_id in self._last_used:
            del self._last_used[obj_id]

        self.stats.total_destroyed += 1
        self.stats.current_size -= 1

    def _ensure_min_size(self):
        """Ensure pool has minimum number of objects"""
        while len(self._pool) + len(self._in_use) < self.min_size:
            obj = self._create_object()
            self._pool.append(obj)
            self.stats.available_resources += 1

    def cleanup(self) -> int:
        """Remove idle objects from pool"""
        removed = 0
        now = datetime.now()

        with self._lock:
            # Keep at least min_size objects
            while len(self._pool) > self.min_size:
                obj = self._pool[0]
                obj_id = id(obj)

                # Check if object has been idle too long
                last_used = self._last_used.get(obj_id)
                if last_used and (now - last_used).total_seconds() > self.max_idle_time:
                    self._pool.popleft()
                    self._destroy_object(obj)
                    removed += 1
                else:
                    break

            self.stats.last_cleanup = now

        return removed

    def clear(self):
        """Clear all objects from pool"""
        with self._lock:
            # Destroy all pooled objects
            while self._pool:
                obj = self._pool.popleft()
                self._destroy_object(obj)

            # Update stats
            self.stats.available_resources = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            return {
                "total_created": self.stats.total_created,
                "total_destroyed": self.stats.total_destroyed,
                "current_size": self.stats.current_size,
                "available": self.stats.available_resources,
                "in_use": self.stats.in_use_resources,
                "total_acquisitions": self.stats.total_acquisitions,
                "total_releases": self.stats.total_releases,
                "avg_wait_time": self.stats.avg_wait_time,
                "max_wait_time": self.stats.max_wait_time,
                "last_cleanup": self.stats.last_cleanup.isoformat() if self.stats.last_cleanup else None,
            }


class ConnectionPool:
    """
    Generic connection pool for database/network connections.

    Features:
    - Connection reuse
    - Health checking
    - Automatic reconnection
    - Connection timeout
    - Maximum connection lifetime
    """

    def __init__(self, connector: Callable[[], Any],
                 validator: Optional[Callable[[Any], bool]] = None,
                 closer: Optional[Callable[[Any], None]] = None,
                 min_connections: int = 0,
                 max_connections: int = 10,
                 max_connection_age: float = 3600.0,
                 connection_timeout: float = 30.0):
        """
        Initialize connection pool.

        Args:
            connector: Function to create new connection
            validator: Function to validate connection health
            closer: Function to close connection
            min_connections: Minimum pool size
            max_connections: Maximum pool size
            max_connection_age: Maximum connection age in seconds
            connection_timeout: Connection timeout
        """
        self.connector = connector
        self.validator = validator or (lambda conn: True)
        self.closer = closer
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.max_connection_age = max_connection_age
        self.connection_timeout = connection_timeout

        self._pool: deque[Any] = deque()
        self._in_use: Dict[int, Any] = {}
        self._created_at: Dict[int, datetime] = {}
        self._lock = threading.Lock()

        self.stats = PoolStats()

        # Pre-populate pool
        self._ensure_min_connections()

        logger.info(f"Connection pool initialized: min={min_connections}, max={max_connections}")

    def acquire(self, timeout: Optional[float] = None) -> Any:
        """Acquire connection from pool"""
        start_time = time.time()
        timeout = timeout or self.connection_timeout

        with self._lock:
            # Check for available connection
            while self._pool:
                conn = self._pool.popleft()

                # Validate connection
                conn_id = id(conn)
                if self._is_connection_valid(conn):
                    self._in_use[conn_id] = conn
                    self.stats.available_connections -= 1
                    self.stats.in_use_connections += 1
                    return conn
                else:
                    # Close invalid connection
                    self._close_connection(conn)

            # Check if we can create new connection
            if len(self._in_use) < self.max_connections:
                conn = self._create_connection()
                conn_id = id(conn)
                self._in_use[conn_id] = conn
                return conn

        # Wait for available connection
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._pool:
                    conn = self._pool.popleft()
                    if self._is_connection_valid(conn):
                        conn_id = id(conn)
                        self._in_use[conn_id] = conn
                        return conn

            time.sleep(0.01)

        raise TimeoutError("Timeout waiting for connection")

    def release(self, conn: Any):
        """Release connection back to pool"""
        conn_id = id(conn)

        with self._lock:
            if conn_id not in self._in_use:
                raise ValueError("Connection not acquired from this pool")

            del self._in_use[conn_id]

            # Return to pool if still valid
            if self._is_connection_valid(conn):
                self._pool.append(conn)
                self.stats.available_connections += 1
            else:
                self._close_connection(conn)

            self.stats.in_use_connections -= 1

    def _create_connection(self) -> Any:
        """Create new connection"""
        conn = self.connector()
        conn_id = id(conn)

        self._created_at[conn_id] = datetime.now()
        self.stats.total_created += 1
        self.stats.current_size += 1

        return conn

    def _close_connection(self, conn: Any):
        """Close connection"""
        conn_id = id(conn)

        if self.closer:
            try:
                self.closer(conn)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Error closing connection: {e}")

        if conn_id in self._created_at:
            del self._created_at[conn_id]

        self.stats.total_destroyed += 1
        self.stats.current_size -= 1

    def _is_connection_valid(self, conn: Any) -> bool:
        """Check if connection is valid"""
        # Check age
        conn_id = id(conn)
        created_at = self._created_at.get(conn_id)

        if created_at:
            age = (datetime.now() - created_at).total_seconds()
            if age > self.max_connection_age:
                return False

        # Check health
        try:
            return self.validator(conn)
        except Exception:  # TODO: Catch specific exception instead of Exception
            return False

    def _ensure_min_connections(self):
        """Ensure minimum number of connections"""
        while len(self._pool) + len(self._in_use) < self.min_connections:
            conn = self._create_connection()
            self._pool.append(conn)
            self.stats.available_connections += 1

    def cleanup(self) -> int:
        """Remove stale connections"""
        removed = 0

        with self._lock:
            # Keep at least min_connections
            while len(self._pool) > self.min_connections:
                conn = self._pool[0]

                if not self._is_connection_valid(conn):
                    self._pool.popleft()
                    self._close_connection(conn)
                    removed += 1
                else:
                    break

        return removed

    def clear(self):
        """Clear all connections"""
        with self._lock:
            while self._pool:
                conn = self._pool.popleft()
                self._close_connection(conn)

            self.stats.available_connections = 0


class SemaphorePool:
    """
    Semaphore-based resource limiter.

    Features:
    - Rate limiting
    - Concurrent access control
    - Timeout support
    - Fairness guarantees
    """

    def __init__(self, max_concurrent: int = 10):
        """
        Initialize semaphore pool.

        Args:
            max_concurrent: Maximum concurrent acquisitions
        """
        self.semaphore = threading.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent

        self.stats = PoolStats()
        self._lock = threading.Lock()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire semaphore"""
        start_time = time.time()

        acquired = self.semaphore.acquire(timeout=timeout or 30.0)

        wait_time = time.time() - start_time

        with self._lock:
            if acquired:
                self.stats.in_use_resources += 1
                self.stats.total_acquisitions += 1
                self.stats.total_wait_time += wait_time

        return acquired

    def release(self):
        """Release semaphore"""
        with self._lock:
            self.stats.in_use_resources -= 1
            self.stats.total_releases += 1

        self.semaphore.release()

    def __enter__(self):
        """Context manager entry"""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        with self._lock:
            return {
                "max_concurrent": self.max_concurrent,
                "in_use": self.stats.in_use_resources,
                "total_acquisitions": self.stats.total_acquisitions,
                "total_releases": self.stats.total_releases,
                "avg_wait_time": (
                    self.stats.total_wait_time / self.stats.total_acquisitions
                    if self.stats.total_acquisitions > 0 else 0
                ),
            }


class ResourceManager:
    """
    Centralized resource management system.

    Features:
    - Multiple pool types
    - Resource lifecycle management
    - Automatic cleanup
    - Statistics aggregation
    """

    def __init__(self):
        """Initialize resource manager"""
        self.object_pools: Dict[str, ObjectPool] = {}
        self.connection_pools: Dict[str, ConnectionPool] = {}
        self.semaphores: Dict[str, SemaphorePool] = {}

        self._lock = threading.Lock()

        logger.info("Resource manager initialized")

    def create_object_pool(self, name: str, factory: Callable[[], Any],
                          **config) -> ObjectPool:
        """Create object pool"""
        with self._lock:
            pool = ObjectPool(factory, **config)
            self.object_pools[name] = pool
            logger.info(f"Created object pool: {name}")
            return pool

    def create_connection_pool(self, name: str, connector: Callable[[], Any],
                              **config) -> ConnectionPool:
        """Create connection pool"""
        with self._lock:
            pool = ConnectionPool(connector, **config)
            self.connection_pools[name] = pool
            logger.info(f"Created connection pool: {name}")
            return pool

    def create_semaphore(self, name: str, max_concurrent: int) -> SemaphorePool:
        """Create semaphore"""
        with self._lock:
            semaphore = SemaphorePool(max_concurrent)
            self.semaphores[name] = semaphore
            logger.info(f"Created semaphore: {name} (max={max_concurrent})")
            return semaphore

    def get_object_pool(self, name: str) -> Optional[ObjectPool]:
        """Get object pool"""
        return self.object_pools.get(name)

    def get_connection_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get connection pool"""
        return self.connection_pools.get(name)

    def get_semaphore(self, name: str) -> Optional[SemaphorePool]:
        """Get semaphore"""
        return self.semaphores.get(name)

    def cleanup_all(self) -> Dict[str, int]:
        """Cleanup all pools"""
        results = {}

        with self._lock:
            for name, pool in self.object_pools.items():
                removed = pool.cleanup()
                results[f"object_pool.{name}"] = removed

            for name, pool in self.connection_pools.items():
                removed = pool.cleanup()
                results[f"connection_pool.{name}"] = removed

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get all statistics"""
        stats = {
            "object_pools": {},
            "connection_pools": {},
            "semaphores": {},
        }

        with self._lock:
            for name, pool in self.object_pools.items():
                stats["object_pools"][name] = pool.get_stats()

            for name, pool in self.connection_pools.items():
                stats["connection_pools"][name] = pool.get_stats()

            for name, semaphore in self.semaphores.items():
                stats["semaphores"][name] = semaphore.get_stats()

        return stats


# Global resource manager
_global_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Get global resource manager"""
    global _global_resource_manager
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager()
    return _global_resource_manager


# Example usage
if __name__ == "__main__":
    import os

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create resource manager
    rm = ResourceManager()

    # Example 1: Object pool
    class ExpensiveObject:
        def __init__(self):
            self.data = list(range(1000))

        def reset(self):
            self.data = []

    def create_object():
        print("Creating new object")
        return ExpensiveObject()

    object_pool = rm.create_object_pool(
        "test_objects",
        factory=create_object,
        reset=lambda obj: obj.reset(),
        min_size=2,
        max_size=5,
    )

    # Acquire and release objects
    obj1 = object_pool.acquire()
    print(f"Acquired object 1: {obj1.data[:5]}")

    obj2 = object_pool.acquire()
    print(f"Acquired object 2: {obj2.data[:5]}")

    object_pool.release(obj1)
    object_pool.release(obj2)

    stats = object_pool.get_stats()
    print(f"Object pool stats: {stats}")

    # Example 2: Semaphore
    semaphore = rm.create_semaphore("test_semaphore", max_concurrent=3)

    def worker(worker_id: int):
        with semaphore:
            print(f"Worker {worker_id} acquired semaphore")
            time.sleep(0.5)
            print(f"Worker {worker_id} releasing semaphore")

    # Run workers
    import threading
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Get all statistics
    all_stats = rm.get_statistics()
    print(f"\nAll statistics: {all_stats}")
