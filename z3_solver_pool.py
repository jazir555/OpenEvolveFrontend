"""
Z3 Solver Pool Management System

Provides centralized tracking of Z3 solver instances:
- Active solver instance counting
- Queue depth tracking for pending operations
- Thread-safe metrics access
- Integration with performance monitoring and reliability checking

This module solves the hardcoded metrics issue in:
- z3_performance_monitor.py (lines 416-417)
- z3_reliability_checker.py

Author: OpenEvolve
Created: 2026-02-05
"""

import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable
from collections import deque
from contextlib import contextmanager
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class SolverState(Enum):
    """States a solver can be in."""
    IDLE = "idle"
    ACTIVE = "active"
    QUEUED = "queued"
    DESTROYED = "destroyed"


@dataclass
class SolverInstance:
    """Represents a single solver instance in the pool."""
    solver_id: str
    state: SolverState = SolverState.IDLE
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    operation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def touch(self):
        """Update last used timestamp."""
        self.last_used_at = time.time()
        self.operation_count += 1


@dataclass
class PoolMetrics:
    """Current metrics for the solver pool."""
    active_solvers: int = 0
    idle_solvers: int = 0
    queued_solvers: int = 0
    queue_depth: int = 0
    total_solvers_created: int = 0
    total_solvers_destroyed: int = 0
    total_operations: int = 0
    average_wait_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_solvers": self.active_solvers,
            "idle_solvers": self.idle_solvers,
            "queued_solvers": self.queued_solvers,
            "queue_depth": self.queue_depth,
            "total_solvers_created": self.total_solvers_created,
            "total_solvers_destroyed": self.total_solvers_destroyed,
            "total_operations": self.total_operations,
            "average_wait_time_ms": round(self.average_wait_time_ms, 2),
            "timestamp": self.timestamp
        }


# =============================================================================
# Z3 Solver Pool
# =============================================================================

class Z3SolverPool:
    """
    Thread-safe pool for tracking Z3 solver instances.
    
    Features:
    - Track active/idle/queued solver counts
    - Monitor queue depth for pending operations
    - Thread-safe metrics access
    - Context manager for automatic state tracking
    - Event callbacks for state changes
    
    Usage:
        # Register a solver when created
        pool = Z3SolverPool()
        solver_id = pool.register_solver()
        
        # Mark solver as active during operation
        with pool.active_operation(solver_id):
            # ... do solver work ...
            pass
        
        # Or manually manage state
        pool.mark_active(solver_id)
        # ... do work ...
        pool.mark_idle(solver_id)
        
        # Get current metrics
        metrics = pool.get_metrics()
        print(f"Active: {metrics.active_solvers}, Queue: {metrics.queue_depth}")
    """
    
    def __init__(self, max_pool_size: int = 10, max_queue_depth: int = 100):
        self._max_pool_size = max_pool_size
        self._max_queue_depth = max_queue_depth
        
        # Solver tracking
        self._solvers: Dict[str, SolverInstance] = {}
        self._operation_queue: deque = deque(maxlen=max_queue_depth)
        
        # Thread safety
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        
        # Metrics tracking
        self._total_created = 0
        self._total_destroyed = 0
        self._total_operations = 0
        self._wait_times: deque = deque(maxlen=1000)
        
        # Callbacks for state changes
        self._callbacks: List[Callable[[str, SolverState, SolverState], None]] = []
        
        self._shutdown = False
    
    # =====================================================================
    # Solver Registration
    # =====================================================================
    
    def register_solver(self, solver_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a new solver instance with the pool.
        
        Args:
            solver_id: Optional custom ID (auto-generated if not provided)
            metadata: Optional metadata about the solver
            
        Returns:
            The solver ID
        """
        import uuid
        
        if solver_id is None:
            solver_id = f"solver_{uuid.uuid4().hex[:8]}"
        
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Pool is shutdown")
            
            instance = SolverInstance(
                solver_id=solver_id,
                state=SolverState.IDLE,
                metadata=metadata or {}
            )
            
            self._solvers[solver_id] = instance
            self._total_created += 1
            
            logger.debug(f"Registered solver {solver_id}")
            return solver_id
    
    def unregister_solver(self, solver_id: str) -> bool:
        """
        Unregister a solver instance from the pool.
        
        Args:
            solver_id: The solver ID to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        with self._lock:
            if solver_id not in self._solvers:
                return False
            
            instance = self._solvers.pop(solver_id)
            instance.state = SolverState.DESTROYED
            self._total_destroyed += 1
            
            # Remove any queued operations for this solver
            self._operation_queue = deque(
                [op for op in self._operation_queue if op.get('solver_id') != solver_id],
                maxlen=self._max_queue_depth
            )
            
            self._condition.notify_all()
            
            logger.debug(f"Unregistered solver {solver_id}")
            return True
    
    # =====================================================================
    # State Management
    # =====================================================================
    
    def mark_active(self, solver_id: str) -> bool:
        """
        Mark a solver as actively processing.
        
        Args:
            solver_id: The solver ID
            
        Returns:
            True if state changed, False if solver not found
        """
        return self._transition_state(solver_id, SolverState.ACTIVE)
    
    def mark_idle(self, solver_id: str) -> bool:
        """
        Mark a solver as idle (ready for work).
        
        Args:
            solver_id: The solver ID
            
        Returns:
            True if state changed, False if solver not found
        """
        return self._transition_state(solver_id, SolverState.IDLE)
    
    def mark_queued(self, solver_id: str) -> bool:
        """
        Mark a solver as queued (waiting for resources).
        
        Args:
            solver_id: The solver ID
            
        Returns:
            True if state changed, False if solver not found
        """
        return self._transition_state(solver_id, SolverState.QUEUED)
    
    def _transition_state(self, solver_id: str, new_state: SolverState) -> bool:
        """Transition a solver to a new state."""
        with self._lock:
            if solver_id not in self._solvers:
                return False
            
            instance = self._solvers[solver_id]
            old_state = instance.state
            
            if old_state == new_state:
                return True
            
            instance.state = new_state
            instance.touch()
            
            # Track queue operations
            if new_state == SolverState.QUEUED:
                self._operation_queue.append({
                    'solver_id': solver_id,
                    'queued_at': time.time()
                })
            elif old_state == SolverState.QUEUED:
                # Calculate wait time
                for op in list(self._operation_queue):
                    if op.get('solver_id') == solver_id:
                        wait_time = (time.time() - op['queued_at']) * 1000  # ms
                        self._wait_times.append(wait_time)
                        self._operation_queue.remove(op)
                        break
            
            if new_state == SolverState.ACTIVE:
                self._total_operations += 1
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(solver_id, old_state, new_state)
                except Exception as e:
                    logger.warning(f"State change callback failed: {e}")
            
            self._condition.notify_all()
            return True
    
    @contextmanager
    def active_operation(self, solver_id: str, timeout: Optional[float] = None):
        """
        Context manager for tracking an active solver operation.
        
        Automatically transitions the solver from idle -> active -> idle.
        
        Args:
            solver_id: The solver ID
            timeout: Optional timeout for the operation
            
        Yields:
            The solver instance
            
        Example:
            with pool.active_operation(solver_id):
                result = solver.solve(constraints)
        """
        start_time = time.time()
        
        # Wait if pool is at capacity
        with self._lock:
            while (self.active_count >= self._max_pool_size and 
                   not self._shutdown and
                   (timeout is None or time.time() - start_time < timeout)):
                self.mark_queued(solver_id)
                wait_remaining = None if timeout is None else timeout - (time.time() - start_time)
                self._condition.wait(timeout=wait_remaining)
            
            if self._shutdown:
                raise RuntimeError("Pool is shutdown")
        
        self.mark_active(solver_id)
        try:
            instance = self._solvers.get(solver_id)
            yield instance
        finally:
            self.mark_idle(solver_id)
    
    # =====================================================================
    # Metrics Access
    # =====================================================================
    
    def get_metrics(self) -> PoolMetrics:
        """
        Get current pool metrics.
        
        Returns:
            PoolMetrics with current counts
        """
        with self._lock:
            active = sum(1 for s in self._solvers.values() if s.state == SolverState.ACTIVE)
            idle = sum(1 for s in self._solvers.values() if s.state == SolverState.IDLE)
            queued = sum(1 for s in self._solvers.values() if s.state == SolverState.QUEUED)
            
            avg_wait = 0.0
            if self._wait_times:
                avg_wait = sum(self._wait_times) / len(self._wait_times)
            
            return PoolMetrics(
                active_solvers=active,
                idle_solvers=idle,
                queued_solvers=queued,
                queue_depth=len(self._operation_queue),
                total_solvers_created=self._total_created,
                total_solvers_destroyed=self._total_destroyed,
                total_operations=self._total_operations,
                average_wait_time_ms=avg_wait,
                timestamp=time.time()
            )
    
    def get_active_count(self) -> int:
        """Get number of active solvers."""
        with self._lock:
            return sum(1 for s in self._solvers.values() if s.state == SolverState.ACTIVE)
    
    def get_queue_depth(self) -> int:
        """Get current queue depth."""
        with self._lock:
            return len(self._operation_queue)
    
    def get_solver_count(self) -> int:
        """Get total number of registered solvers."""
        with self._lock:
            return len(self._solvers)
    
    # =====================================================================
    # Callbacks
    # =====================================================================
    
    def add_state_callback(self, callback: Callable[[str, SolverState, SolverState], None]):
        """
        Add a callback for state changes.
        
        Args:
            callback: Function(solver_id, old_state, new_state) called on state changes
        """
        self._callbacks.append(callback)
    
    def remove_state_callback(self, callback: Callable[[str, SolverState, SolverState], None]):
        """Remove a state change callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    # =====================================================================
    # Pool Management
    # =====================================================================
    
    def get_all_solvers(self) -> List[SolverInstance]:
        """Get list of all solver instances."""
        with self._lock:
            return list(self._solvers.values())
    
    def get_solver(self, solver_id: str) -> Optional[SolverInstance]:
        """Get a specific solver instance."""
        with self._lock:
            return self._solvers.get(solver_id)
    
    def shutdown(self, wait: bool = True, timeout: float = 30.0):
        """
        Shutdown the pool and cleanup resources.
        
        Args:
            wait: Whether to wait for active operations to complete
            timeout: Maximum time to wait
        """
        with self._lock:
            self._shutdown = True
            
            if wait:
                start = time.time()
                while (self.active_count > 0 and 
                       time.time() - start < timeout):
                    self._condition.wait(timeout=0.1)
            
            # Clear all solvers
            self._solvers.clear()
            self._operation_queue.clear()
            self._callbacks.clear()
            
            logger.info("Z3SolverPool shutdown complete")
    
    @property
    def active_count(self) -> int:
        """Number of active solvers."""
        return self.get_active_count()
    
    @property
    def is_shutdown(self) -> bool:
        """Whether pool is shutdown."""
        with self._lock:
            return self._shutdown


# =============================================================================
# Global Singleton Instance
# =============================================================================

_solver_pool: Optional[Z3SolverPool] = None
_pool_lock = threading.Lock()


def get_solver_pool() -> Z3SolverPool:
    """
    Get the global solver pool instance.
    
    Returns:
        The global Z3SolverPool singleton
    """
    global _solver_pool
    if _solver_pool is None:
        with _pool_lock:
            if _solver_pool is None:
                _solver_pool = Z3SolverPool()
    return _solver_pool


def reset_solver_pool():
    """Reset the global solver pool (useful for testing)."""
    global _solver_pool
    with _pool_lock:
        if _solver_pool is not None:
            _solver_pool.shutdown(wait=False)
        _solver_pool = None


# =============================================================================
# Integration Helpers
# =============================================================================

def get_active_solvers_count() -> int:
    """Get count of active solvers from global pool."""
    return get_solver_pool().get_active_count()


def get_queue_depth_count() -> int:
    """Get queue depth from global pool."""
    return get_solver_pool().get_queue_depth()


def get_solver_metrics() -> Dict[str, Any]:
    """Get all solver metrics as dictionary."""
    return get_solver_pool().get_metrics().to_dict()


# =============================================================================
# Auto-registration Decorator
# =============================================================================

def register_with_pool(method):
    """
    Decorator to auto-register solver instances with the global pool.
    
    This should be used on __init__ methods of solver classes.
    
    Example:
        class MySolver:
            @register_with_pool
            def __init__(self):
                self.solver_id = None  # Will be set by decorator
                
            def solve(self):
                with get_solver_pool().active_operation(self.solver_id):
                    # ... solving logic ...
                    pass
    """
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        
        # Register with pool
        pool = get_solver_pool()
        solver_id = pool.register_solver(metadata={'class': self.__class__.__name__})
        self.solver_id = solver_id
        
        # Store original cleanup
        original_del = getattr(self, '__del__', None)
        
        def cleanup():
            pool.unregister_solver(solver_id)
            if original_del:
                original_del()
        
        self.__del__ = cleanup
        
        return result
    
    return wrapper


# =============================================================================
# Legacy Compatibility
# =============================================================================

# For backward compatibility with code expecting the old interface
Z3SolverRegistry = Z3SolverPool
get_solver_registry = get_solver_pool
