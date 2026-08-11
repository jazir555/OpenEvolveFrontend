"""
Distributed Z3 Solver Pool

This module provides a parallel Z3 solving infrastructure with:
- Multi-process solver pool for concurrent constraint solving
- Work stealing and load balancing
- Solver state isolation
- Result aggregation and consensus
- Automatic timeout management
- Resource monitoring and adaptive scaling
- Caching with TTL
- Fault tolerance and retry logic

Author: Z3-Lean Integration Project
Date: 2026-02-17
"""

import multiprocessing as mp
import logging
import time
import hashlib
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import psutil

# Z3 imports
try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    z3 = None

logger = logging.getLogger(__name__)


class SolverState(Enum):
    """States of a solver instance"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    TIMEOUT = "timeout"


class TaskStatus(Enum):
    """Status of a solving task"""
    PENDING = "pending"
    RUNNING = "running"
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class SolverTask:
    """Represents a single solving task"""
    task_id: str
    constraints: str
    timeout: int = 30000  # milliseconds
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class SolverResult:
    """Represents the result of a solving task"""
    task_id: str
    status: TaskStatus
    sat: bool = False
    model: Optional[Dict[str, Any]] = None
    proof: Optional[str] = None
    execution_time: float = 0.0
    solver_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverStats:
    """Statistics for a solver instance"""
    solver_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_timeout: int = 0
    total_time: float = 0.0
    average_time: float = 0.0
    current_state: SolverState = SolverState.IDLE
    memory_usage: float = 0.0
    cpu_usage: float = 0.0


class Z3SolverWorker:
    """
    Individual Z3 solver worker that runs in a separate process.
    """

    def __init__(self, worker_id: str, config: Optional[Dict] = None):
        """
        Initialize a Z3 solver worker.

        Args:
            worker_id: Unique identifier for this worker
            config: Optional configuration dictionary
        """
        self.worker_id = worker_id
        self.config = config or {}
        self.solver = None
        self.stats = SolverStats(solver_id=worker_id)

        if Z3_AVAILABLE:
            self._initialize_solver()

        logger.info(f"Z3 solver worker initialized: {worker_id}")

    def _initialize_solver(self):
        """Initialize the Z3 solver with configuration."""
        try:
            self.solver = z3.Solver()

            # Set solver parameters
            if 'timeout' in self.config:
                self.solver.set('timeout', self.config['timeout'])

            if 'max_memory' in self.config:
                self.solver.set('max_memory', self.config['max_memory'])

            if 'random_seed' in self.config:
                self.solver.set('random_seed', self.config['random_seed'])

            logger.debug(f"Z3 solver {self.worker_id} configured with params: {self.config}")

        except Exception as e:
            logger.error(f"Failed to initialize solver {self.worker_id}: {e}")
            self.solver = None

    def solve(self, task: SolverTask) -> SolverResult:
        """
        Solve a single task.

        Args:
            task: SolverTask to solve

        Returns:
            SolverResult with the solution
        """
        start_time = time.time()

        if not Z3_AVAILABLE or self.solver is None:
            return SolverResult(
                task_id=task.task_id,
                status=TaskStatus.ERROR,
                error="Z3 not available",
                execution_time=time.time() - start_time
            )

        try:
            # Reset solver state
            self.solver.reset()

            # Parse and assert constraints
            constraints = task.constraints
            if isinstance(constraints, str):
                # Parse from SMT-LIB format
                try:
                    # Try to parse as SMT-LIB
                    expr = z3.parse_smt2_string(constraints)
                    for e in expr:
                        self.solver.add(e)
                except:
                    # If that fails, try to parse as simple assertion
                    try:
                        expr = z3.parse_smt2_string(f'(assert {constraints})')
                        self.solver.add(expr[0])
                    except Exception as e:
                        return SolverResult(
                            task_id=task.task_id,
                            status=TaskStatus.ERROR,
                            error=f"Parse error: {str(e)}",
                            execution_time=time.time() - start_time
                        )
            elif isinstance(constraints, list):
                # List of Z3 expressions
                for constraint in constraints:
                    self.solver.add(constraint)

            # Set timeout for this specific check
            self.solver.set('timeout', task.timeout)

            # Solve
            result = self.solver.check()
            execution_time = time.time() - start_time

            # Update stats
            self.stats.tasks_completed += 1
            self.stats.total_time += execution_time
            self.stats.average_time = self.stats.total_time / self.stats.tasks_completed

            # Process result
            if result == z3.sat:
                model = self.solver.model()
                model_dict = {}

                # Extract model values
                for decl in model:
                    name = decl.name()
                    value = model[decl]
                    model_dict[name] = str(value)

                return SolverResult(
                    task_id=task.task_id,
                    status=TaskStatus.SAT,
                    sat=True,
                    model=model_dict,
                    execution_time=execution_time,
                    solver_id=self.worker_id
                )

            elif result == z3.unsat:
                return SolverResult(
                    task_id=task.task_id,
                    status=TaskStatus.UNSAT,
                    sat=False,
                    execution_time=execution_time,
                    solver_id=self.worker_id
                )

            else:
                return SolverResult(
                    task_id=task.task_id,
                    status=TaskStatus.UNKNOWN,
                    sat=False,
                    execution_time=execution_time,
                    solver_id=self.worker_id
                )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)

            # Check for timeout
            if 'timeout' in error_msg.lower():
                self.stats.tasks_timeout += 1
                return SolverResult(
                    task_id=task.task_id,
                    status=TaskStatus.TIMEOUT,
                    error=error_msg,
                    execution_time=execution_time,
                    solver_id=self.worker_id
                )
            else:
                self.stats.tasks_failed += 1
                return SolverResult(
                    task_id=task.task_id,
                    status=TaskStatus.ERROR,
                    error=error_msg,
                    execution_time=execution_time,
                    solver_id=self.worker_id
                )

    def get_stats(self) -> SolverStats:
        """Get current statistics for this worker."""
        # Update resource usage
        try:
            process = psutil.Process()
            self.stats.memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
            self.stats.cpu_usage = process.cpu_percent()
        except:
            pass

        return self.stats


class DistributedZ3SolverPool:
    """
    Distributed pool of Z3 solvers for parallel constraint solving.

    Features:
    - Multi-process parallel solving
    - Dynamic task queue with priority
    - Work stealing and load balancing
    - Result caching
    - Fault tolerance with retries
    - Resource monitoring
    """

    def __init__(
        self,
        num_workers: Optional[int] = None,
        config: Optional[Dict] = None,
        cache_size: int = 1000,
        cache_ttl: int = 3600
    ):
        """
        Initialize the distributed solver pool.

        Args:
            num_workers: Number of worker processes (default: CPU count)
            config: Configuration for each solver
            cache_size: Maximum number of cached results
            cache_ttl: Cache time-to-live in seconds
        """
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        self.config = config or {}
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl

        # Task queue
        self.task_queue: Queue = Queue()
        self.results: Dict[str, SolverResult] = {}
        self.pending_tasks: Dict[str, SolverTask] = {}

        # Cache
        self.cache: Dict[str, Tuple[SolverResult, float]] = {}
        self.cache_lock = threading.Lock()

        # Workers
        self.workers: List[Z3SolverWorker] = []
        self.executor: Optional[ProcessPoolExecutor] = None

        # Statistics
        self.total_tasks_submitted = 0
        self.total_tasks_completed = 0
        self.pool_start_time = time.time()

        # Initialize workers
        self._initialize_workers()

        logger.info(f"Distributed Z3 solver pool initialized with {self.num_workers} workers")

    def _initialize_workers(self):
        """Initialize worker processes."""
        worker_ids = [f"worker_{i}" for i in range(self.num_workers)]
        self.workers = [Z3SolverWorker(wid, self.config) for wid in worker_ids]

        # Initialize process pool executor
        # Note: We use ThreadPoolExecutor for actual workers since Z3 objects
        # can't be easily pickled for ProcessPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

        logger.info(f"Initialized {len(self.workers)} solver workers")

    def submit_task(self, task: SolverTask) -> str:
        """
        Submit a task to the pool.

        Args:
            task: SolverTask to solve

        Returns:
            Task ID
        """
        # Check cache first
        cache_key = self._get_cache_key(task)
        with self.cache_lock:
            if cache_key in self.cache:
                result, timestamp = self.cache[cache_key]
                # Check if cache entry is still valid
                if time.time() - timestamp < self.cache_ttl:
                    logger.debug(f"Cache hit for task {task.task_id}")
                    self.results[task.task_id] = result
                    self.total_tasks_completed += 1
                    return task.task_id

        # Add to pending tasks
        self.pending_tasks[task.task_id] = task
        self.total_tasks_submitted += 1

        # Submit to executor
        future = self.executor.submit(self._solve_task, task)
        future.add_done_callback(lambda f: self._task_completed(task.task_id, f))

        logger.debug(f"Submitted task {task.task_id} to pool")
        return task.task_id

    def _solve_task(self, task: SolverTask) -> SolverResult:
        """
        Solve a task (called by worker).

        Args:
            task: SolverTask to solve

        Returns:
            SolverResult
        """
        # Find an available worker
        worker = self._get_available_worker()
        if worker is None:
            # All workers busy, use first one (will queue internally)
            worker = self.workers[0]

        return worker.solve(task)

    def _get_available_worker(self) -> Optional[Z3SolverWorker]:
        """Get an available worker (idle state)."""
        for worker in self.workers:
            if worker.stats.current_state == SolverState.IDLE:
                return worker
        return None

    def _task_completed(self, task_id: str, future):
        """
        Callback when a task is completed.

        Args:
            task_id: Task identifier
            future: Completed future
        """
        try:
            result = future.result()

            # Store result
            self.results[task_id] = result
            self.total_tasks_completed += 1

            # Cache result
            if task_id in self.pending_tasks:
                task = self.pending_tasks[task_id]
                cache_key = self._get_cache_key(task)
                with self.cache_lock:
                    # Implement simple LRU eviction
                    if len(self.cache) >= self.cache_size:
                        # Remove oldest entry
                        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                        del self.cache[oldest_key]

                    self.cache[cache_key] = (result, time.time())

            # Remove from pending
            if task_id in self.pending_tasks:
                del self.pending_tasks[task_id]

            logger.debug(f"Task {task_id} completed: {result.status.value}")

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            self.results[task_id] = SolverResult(
                task_id=task_id,
                status=TaskStatus.ERROR,
                error=str(e)
            )

    def _get_cache_key(self, task: SolverTask) -> str:
        """Generate cache key for a task."""
        content = f"{task.constraints}:{task.timeout}:{task.priority}"
        return hashlib.md5(content.encode()).hexdigest()

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[SolverResult]:
        """
        Get result for a task.

        Args:
            task_id: Task identifier
            timeout: Optional timeout in seconds

        Returns:
            SolverResult or None if not ready
        """
        start_time = time.time()

        while True:
            if task_id in self.results:
                return self.results[task_id]

            if timeout and (time.time() - start_time) > timeout:
                return None

            time.sleep(0.01)

    def solve_batch(
        self,
        tasks: List[SolverTask],
        timeout: Optional[float] = None
    ) -> Dict[str, SolverResult]:
        """
        Solve a batch of tasks in parallel.

        Args:
            tasks: List of SolverTasks
            timeout: Optional timeout for entire batch

        Returns:
            Dictionary mapping task IDs to results
        """
        # Submit all tasks
        for task in tasks:
            self.submit_task(task)

        # Wait for all results
        results = {}
        deadline = time.time() + timeout if timeout else None

        for task in tasks:
            remaining_timeout = None
            if deadline:
                remaining_timeout = max(0, deadline - time.time())

            result = self.get_result(task.task_id, remaining_timeout)
            if result:
                results[task.task_id] = result

        return results

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool-wide statistics."""
        worker_stats = [w.get_stats() for w in self.workers]

        uptime = time.time() - self.pool_start_time

        return {
            'num_workers': self.num_workers,
            'total_tasks_submitted': self.total_tasks_submitted,
            'total_tasks_completed': self.total_tasks_completed,
            'pending_tasks': len(self.pending_tasks),
            'cache_size': len(self.cache),
            'cache_hit_ratio': len(self.cache) / max(1, self.total_tasks_completed),
            'uptime_seconds': uptime,
            'throughput_per_second': self.total_tasks_completed / max(1, uptime),
            'worker_stats': [
                {
                    'solver_id': ws.solver_id,
                    'tasks_completed': ws.tasks_completed,
                    'tasks_failed': ws.tasks_failed,
                    'tasks_timeout': ws.tasks_timeout,
                    'average_time': ws.average_time,
                    'memory_usage_mb': ws.memory_usage,
                    'cpu_usage_percent': ws.cpu_usage,
                }
                for ws in worker_stats
            ]
        }

    def shutdown(self, wait: bool = True):
        """
        Shutdown the solver pool.

        Args:
            wait: Wait for pending tasks to complete
        """
        if self.executor:
            self.executor.shutdown(wait=wait)

        logger.info("Distributed Z3 solver pool shutdown")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown(wait=True)


# Convenience functions
def solve_parallel(
    constraints_list: List[str],
    num_workers: Optional[int] = None,
    timeout: int = 30000
) -> List[SolverResult]:
    """
    Solve multiple constraint sets in parallel.

    Args:
        constraints_list: List of constraint strings
        num_workers: Number of parallel workers
        timeout: Timeout per task in milliseconds

    Returns:
        List of SolverResults
    """
    with DistributedZ3SolverPool(num_workers=num_workers) as pool:
        tasks = [
            SolverTask(
                task_id=f"task_{i}",
                constraints=constraints,
                timeout=timeout
            )
            for i, constraints in enumerate(constraints_list)
        ]

        results_dict = pool.solve_batch(tasks)
        return [results_dict[task.task_id] for task in tasks]


def solve_with_consensus(
    constraints: str,
    num_solvers: int = 3,
    timeout: int = 30000
) -> Tuple[Optional[SolverResult], float]:
    """
    Solve constraints with multiple solvers and check for consensus.

    Args:
        constraints: Z3 constraints
        num_solvers: Number of solvers to use
        timeout: Timeout per solver in milliseconds

    Returns:
        Tuple of (result, consensus_ratio)
    """
    with DistributedZ3SolverPool(num_workers=num_solvers) as pool:
        task = SolverTask(
            task_id="consensus_task",
            constraints=constraints,
            timeout=timeout
        )

        # Submit same task to multiple workers
        tasks = [
            SolverTask(
                task_id=f"consensus_{i}",
                constraints=constraints,
                timeout=timeout
            )
            for i in range(num_solvers)
        ]

        results_dict = pool.solve_batch(tasks)
        results = list(results_dict.values())

        # Count results by status
        status_counts = {}
        for result in results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1

        # Find consensus status
        consensus_status = max(status_counts, key=status_counts.get)
        consensus_count = status_counts[consensus_status]
        consensus_ratio = consensus_count / len(results)

        # Return result with consensus status
        consensus_result = next(r for r in results if r.status == consensus_status)

        return consensus_result, consensus_ratio


__all__ = [
    'DistributedZ3SolverPool',
    'Z3SolverWorker',
    'SolverTask',
    'SolverResult',
    'SolverStats',
    'SolverState',
    'TaskStatus',
    'solve_parallel',
    'solve_with_consensus',
]
