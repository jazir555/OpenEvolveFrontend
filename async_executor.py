"""
Async/Concurrent Execution System for OpenEvolve Decomposition Engine

Provides advanced async and concurrent execution patterns:
- Async/await support
- Thread pool execution
- Process pool execution
- Parallel processing
- Batch processing
- Lazy loading
- Future/Promise patterns
- Cancellation support
"""

import asyncio
import concurrent.futures
import threading
import queue
import time
import inspect
from typing import Dict, Any, List, Optional, Callable, TypeVar, Coroutine, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging
import functools

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class TaskResult:
    """Result of an async task"""
    task_id: str
    status: str  # "pending", "running", "completed", "failed", "cancelled"
    result: Any = None
    exception: Optional[Exception] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: float = 0.0
    retries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "result": str(self.result)[:200] if self.result else None,
            "exception": str(self.exception) if self.exception else None,
            "execution_time": self.execution_time,
            "retries": self.retries,
        }


class AsyncTaskExecutor:
    """
    Async task executor using asyncio.

    Features:
    - Async/await support
    - Task scheduling and management
    - Automatic retry logic
    - Timeout handling
    - Cancellation support
    """

    def __init__(self, max_concurrent_tasks: int = 10,
                 default_timeout: float = 300.0,
                 max_retries: int = 3):
        """
        Initialize async task executor.

        Args:
            max_concurrent_tasks: Maximum concurrent tasks
            default_timeout: Default timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_timeout = default_timeout
        self.max_retries = max_retries

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._tasks: Dict[str, TaskResult] = {}
        self._lock = threading.Lock()

        logger.info(f"Async task executor initialized: max_concurrent={max_concurrent_tasks}")

    async def _run_task_async(self, task_id: str, coro: Coroutine,
                              timeout: Optional[float] = None) -> Any:
        """Run a coroutine with timeout"""
        timeout = timeout or self.default_timeout
        return await asyncio.wait_for(coro, timeout=timeout)

    def submit_async(self, task_id: str, coro: Coroutine,
                     timeout: Optional[float] = None) -> str:
        """Submit async task for execution"""
        with self._lock:
            self._tasks[task_id] = TaskResult(
                task_id=task_id,
                status="pending",
                started_at=datetime.now(),
            )

        # Run in existing loop or create new one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run task
        async def run_and_track():
            with self._lock:
                self._tasks[task_id].status = "running"

            start_time = time.perf_counter()

            try:
                result = await self._run_task_async(task_id, coro, timeout)

                execution_time = time.perf_counter() - start_time

                with self._lock:
                    self._tasks[task_id].status = "completed"
                    self._tasks[task_id].result = result
                    self._tasks[task_id].completed_at = datetime.now()
                    self._tasks[task_id].execution_time = execution_time

                return result

            except asyncio.TimeoutError:
                with self._lock:
                    self._tasks[task_id].status = "failed"
                    self._tasks[task_id].exception = TimeoutError("Task timed out")
                    self._tasks[task_id].completed_at = datetime.now()

            except Exception as e:  # TODO: Catch specific exception instead of Exception
                with self._lock:
                    self._tasks[task_id].status = "failed"
                    self._tasks[task_id].exception = e
                    self._tasks[task_id].completed_at = datetime.now()

        # Schedule task
        if loop.is_running():
            asyncio.ensure_future(run_and_track())
        else:
            loop.run_until_complete(run_and_track())

        return task_id

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result (blocks until complete)"""
        start_time = time.time()

        while True:
            with self._lock:
                if task_id in self._tasks:
                    task_result = self._tasks[task_id]
                    if task_result.status in ["completed", "failed", "cancelled"]:
                        if task_result.exception:
                            raise task_result.exception
                        return task_result.result

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")

            time.sleep(0.01)

    def cancel(self, task_id: str) -> bool:
        """Cancel a task"""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].status = "cancelled"
                return True
        return False

    def get_status(self, task_id: str) -> Optional[TaskResult]:
        """Get task status"""
        with self._lock:
            return self._tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, TaskResult]:
        """Get all tasks"""
        with self._lock:
            return dict(self._tasks)


class ThreadPoolExecutor:
    """
    Thread pool executor for concurrent execution.

    Features:
    - Thread pool management
    - Function submission
    - Result retrieval
    - Exception handling
    - Timeout support
    """

    def __init__(self, max_workers: int = None,
                 thread_name_prefix: str = "worker"):
        """
        Initialize thread pool executor.

        Args:
            max_workers: Maximum number of worker threads
            thread_name_prefix: Prefix for thread names
        """
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) * 4)
        self.thread_name_prefix = thread_name_prefix

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=thread_name_prefix
        )

        self._futures: Dict[str, concurrent.futures.Future] = {}
        self._results: Dict[str, TaskResult] = {}
        self._lock = threading.Lock()

        logger.info(f"Thread pool executor initialized: max_workers={self.max_workers}")

    def submit(self, task_id: str, func: Callable, *args, **kwargs) -> str:
        """Submit task to thread pool"""
        with self._lock:
            self._results[task_id] = TaskResult(
                task_id=task_id,
                status="pending",
                started_at=datetime.now(),
            )

        def task_wrapper():
            with self._lock:
                self._results[task_id].status = "running"

            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)

                execution_time = time.perf_counter() - start_time

                with self._lock:
                    self._results[task_id].status = "completed"
                    self._results[task_id].result = result
                    self._results[task_id].completed_at = datetime.now()
                    self._results[task_id].execution_time = execution_time

                return result

            except Exception as e:  # TODO: Catch specific exception instead of Exception
                execution_time = time.perf_counter() - start_time

                with self._lock:
                    self._results[task_id].status = "failed"
                    self._results[task_id].exception = e
                    self._results[task_id].completed_at = datetime.now()
                    self._results[task_id].execution_time = execution_time

                raise

        future = self._executor.submit(task_wrapper)

        with self._lock:
            self._futures[task_id] = future

        return task_id

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result"""
        with self._lock:
            future = self._futures.get(task_id)
            task_result = self._results.get(task_id)

        if not future:
            raise ValueError(f"Task {task_id} not found")

        try:
            future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Timeout waiting for task {task_id}")

        return task_result.result if task_result else None

    def map(self, func: Callable, iterable: List[Any],
            timeout: Optional[float] = None) -> List[Any]:
        """Map function over iterable in parallel"""
        futures = []
        for item in iterable:
            future = self._executor.submit(func, item)
            futures.append(future)

        results = []
        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            try:
                results.append(future.result())
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Task failed: {e}")
                results.append(None)

        return results

    def shutdown(self, wait: bool = True):
        """Shutdown thread pool"""
        self._executor.shutdown(wait=wait)


class ProcessPoolExecutor:
    """
    Process pool executor for CPU-bound tasks.

    Features:
    - Process pool management
    - CPU-bound task parallelization
    - Inter-process communication
    - Result aggregation
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize process pool executor.

        Args:
            max_workers: Maximum number of worker processes
        """
        self.max_workers = max_workers or os.cpu_count() or 1

        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        )

        self._futures: Dict[str, concurrent.futures.Future] = {}
        self._results: Dict[str, TaskResult] = {}
        self._lock = threading.Lock()

        logger.info(f"Process pool executor initialized: max_workers={self.max_workers}")

    def submit(self, task_id: str, func: Callable, *args, **kwargs) -> str:
        """Submit task to process pool"""
        with self._lock:
            self._results[task_id] = TaskResult(
                task_id=task_id,
                status="pending",
                started_at=datetime.now(),
            )

        def task_wrapper():
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)

                execution_time = time.perf_counter() - start_time

                with self._lock:
                    self._results[task_id].status = "completed"
                    self._results[task_id].result = result
                    self._results[task_id].completed_at = datetime.now()
                    self._results[task_id].execution_time = execution_time

                return result

            except Exception as e:  # TODO: Catch specific exception instead of Exception
                execution_time = time.perf_counter() - start_time

                with self._lock:
                    self._results[task_id].status = "failed"
                    self._results[task_id].exception = e
                    self._results[task_id].completed_at = datetime.now()
                    self._results[task_id].execution_time = execution_time

                raise

        future = self._executor.submit(task_wrapper)

        with self._lock:
            self._futures[task_id] = future

        return task_id

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result"""
        with self._lock:
            future = self._futures.get(task_id)
            task_result = self._results.get(task_id)

        if not future:
            raise ValueError(f"Task {task_id} not found")

        try:
            future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Timeout waiting for task {task_id}")

        return task_result.result if task_result else None

    def map(self, func: Callable, iterable: List[Any],
            chunksize: int = 1) -> List[Any]:
        """Map function over iterable using multiple processes"""
        return self._executor.map(func, iterable, chunksize=chunksize)

    def shutdown(self, wait: bool = True):
        """Shutdown process pool"""
        self._executor.shutdown(wait=wait)


class BatchProcessor:
    """
    Batch processing system for efficient bulk operations.

    Features:
    - Batch collection and processing
    - Timeout-based flushing
    - Size-based flushing
    - Parallel batch execution
    """

    def __init__(self, batch_size: int = 100,
                 flush_timeout: float = 5.0,
                 max_parallel_batches: int = 3):
        """
        Initialize batch processor.

        Args:
            batch_size: Maximum batch size
            flush_timeout: Timeout before auto-flush
            max_parallel_batches: Maximum parallel batch executions
        """
        self.batch_size = batch_size
        self.flush_timeout = flush_timeout
        self.max_parallel_batches = max_parallel_batches

        self._batch: List[Any] = []
        self._lock = threading.Lock()
        self._last_flush = time.time()
        self._executor = ThreadPoolExecutor(max_workers=max_parallel_batches)

        logger.info(f"Batch processor initialized: batch_size={batch_size}")

    def add(self, item: Any):
        """Add item to current batch"""
        with self._lock:
            self._batch.append(item)

            # Check if batch is full
            if len(self._batch) >= self.batch_size:
                self._flush_async()

            # Check timeout
            if time.time() - self._last_flush > self.flush_timeout:
                self._flush_async()

    def _flush_async(self):
        """Flush batch asynchronously"""
        if not self._batch:
            return

        batch = self._batch.copy()
        self._batch.clear()
        self._last_flush = time.time()

        # Submit for processing
        task_id = f"batch_{int(time.time() * 1000)}"
        self._executor.submit(task_id, self._process_batch, batch)

    def _process_batch(self, batch: List[Any]) -> int:
        """Process a batch (override in subclass)"""
        logger.info(f"Processing batch of {len(batch)} items")
        # Override with actual processing logic
        return len(batch)

    def flush(self) -> int:
        """Flush current batch synchronously"""
        with self._lock:
            if not self._batch:
                return 0

            batch = self._batch.copy()
            self._batch.clear()
            self._last_flush = time.time()

        return self._process_batch(batch)


class LazyLoader:
    """
    Lazy loading system for deferred computation.

    Features:
    - Deferred computation
    - Result caching
    - Dependency tracking
    - Automatic invalidation
    """

    def __init__(self):
        """Initialize lazy loader"""
        self._values: Dict[str, Any] = {}
        self._computers: Dict[str, Callable] = {}
        self._computed: Dict[str, bool] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._lock = threading.Lock()

    def register(self, name: str, computer: Callable,
                 dependencies: Optional[List[str]] = None):
        """Register a lazy computation"""
        with self._lock:
            self._computers[name] = computer
            self._computed[name] = False
            self._dependencies[name] = dependencies or []

    def get(self, name: str) -> Any:
        """Get lazy-computed value"""
        with self._lock:
            # Check if already computed
            if name in self._values and self._computed.get(name, False):
                return self._values[name]

            # Check dependencies
            for dep in self._dependencies.get(name, []):
                if dep not in self._values or not self._computed.get(dep, False):
                    # Compute dependency first
                    self.get(dep)

            # Compute value
            if name not in self._computers:
                raise ValueError(f"No computer registered for {name}")

            self._values[name] = self._computers[name]()
            self._computed[name] = True

            return self._values[name]

    def invalidate(self, name: Optional[str] = None):
        """Invalidate cached value(s)"""
        with self._lock:
            if name is None:
                # Invalidate all
                self._computed.clear()
                self._values.clear()
            elif name in self._computed:
                self._computed[name] = False

                # Invalidate dependents
                for other_name, deps in self._dependencies.items():
                    if name in deps:
                        self.invalidate(other_name)

    def is_computed(self, name: str) -> bool:
        """Check if value is computed"""
        with self._lock:
            return self._computed.get(name, False)


class ConcurrentExecutor:
    """
    Unified concurrent execution interface.

    Features:
    - Thread pool execution
    - Process pool execution
    - Async execution
    - Batch processing
    - Lazy loading
    """

    def __init__(self, max_threads: Optional[int] = None,
                 max_processes: Optional[int] = None):
        """
        Initialize concurrent executor.

        Args:
            max_threads: Maximum threads for thread pool
            max_processes: Maximum processes for process pool
        """
        self.thread_pool = ThreadPoolExecutor(max_workers=max_threads)
        self.process_pool = ProcessPoolExecutor(max_workers=max_processes)
        self.async_executor = AsyncTaskExecutor()
        self.batch_processor = BatchProcessor()
        self.lazy_loader = LazyLoader()

        logger.info("Concurrent executor initialized")

    def execute_parallel(self, func: Callable, items: List[Any],
                        use_processes: bool = False) -> List[Any]:
        """Execute function on items in parallel"""
        if use_processes:
            return self.process_pool.map(func, items)
        else:
            return self.thread_pool.map(func, items)

    def execute_batch(self, items: List[Any]) -> int:
        """Execute items as batch"""
        for item in items:
            self.batch_processor.add(item)
        return self.batch_processor.flush()

    def execute_async(self, coro: Coroutine) -> Any:
        """Execute async coroutine"""
        task_id = f"async_{int(time.time() * 1000)}"
        self.async_executor.submit_async(task_id, coro)
        return self.async_executor.get_result(task_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            "thread_pool": {
                "pending_tasks": len(self.thread_pool._futures),
            },
            "process_pool": {
                "pending_tasks": len(self.process_pool._futures),
            },
            "async_executor": {
                "pending_tasks": sum(
                    1 for t in self.async_executor._tasks.values()
                    if t.status == "pending"
                ),
            },
            "batch_processor": {
                "current_batch_size": len(self.batch_processor._batch),
            },
        }

    def shutdown(self):
        """Shutdown all executors"""
        self.thread_pool.shutdown()
        self.process_pool.shutdown()


def parallel(num_workers: Optional[int] = None, use_processes: bool = False):
    """
    Decorator for parallel execution.

    Args:
        num_workers: Number of workers
        use_processes: Whether to use processes instead of threads

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(items: List[Any]) -> List[Any]:
            executor = ConcurrentExecutor(max_threads=num_workers)

            try:
                if use_processes:
                    return executor.process_pool.map(func, items)
                else:
                    return executor.thread_pool.map(func, items)
            finally:
                executor.shutdown()

        return wrapper

    return decorator


# Global concurrent executor
_global_executor: Optional[ConcurrentExecutor] = None


def get_executor() -> ConcurrentExecutor:
    """Get global concurrent executor"""
    global _global_executor
    if _global_executor is None:
        _global_executor = ConcurrentExecutor()
    return _global_executor


# Example usage
if __name__ == "__main__":
    import os

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Example 1: Thread pool execution
    def compute_square(n: int) -> int:
        time.sleep(0.1)  # Simulate work
        return n * n

    thread_pool = ThreadPoolExecutor(max_workers=4)

    tasks = []
    for i in range(10):
        task_id = f"square_{i}"
        thread_pool.submit(task_id, compute_square, i)
        tasks.append(task_id)

    results = [thread_pool.get_result(task_id) for task_id in tasks]
    logger.info(f"Thread pool results: {results}")

    # Example 2: Batch processing
    batch_processor = BatchProcessor(batch_size=5)

    for i in range(12):
        batch_processor.add(f"item_{i}")

    batch_processor.flush()

    # Example 3: Lazy loading
    lazy_loader = LazyLoader()

    lazy_loader.register("expensive", lambda: sum(range(1000000)))
    lazy_loader.register("dependent", lambda: lazy_loader.get("expensive") + 1)

    logger.info(f"Lazy value: {lazy_loader.get('expensive')}")

    # Example 4: Parallel decorator
    @parallel(num_workers=4)
    def parallel_sum(items: List[int]) -> int:
        return sum(items)

    result = parallel_sum(list(range(100)))
    logger.info(f"Parallel result: {result}")

    thread_pool.shutdown()
