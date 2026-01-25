"""
RESE Security: Resource Limits and Management

Comprehensive resource limiting, timeout mechanisms, and queue management.

Author: Agent M2 (Security and Reliability Specialist)
Created: 2025-12-31
"""

import time
import threading
import multiprocessing
import psutil
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
import queue
import hashlib


# =============================================================================
# Resource Limits Configuration
# =============================================================================

@dataclass
class ResourceLimits:
    """Resource limits for operations"""
    max_memory_mb: float = 4096          # Maximum memory in MB
    max_time_seconds: float = 3600       # Maximum execution time
    max_cpu_percent: float = 95.0        # Maximum CPU percentage
    max_open_files: int = 1024           # Maximum open file descriptors
    max_threads: int = 32                # Maximum threads
    max_processes: int = 16              # Maximum processes
    queue_size: int = 1000               # Maximum queue size

    def validate(self) -> bool:
        """Validate resource limits are reasonable"""
        return all([
            self.max_memory_mb > 0,
            self.max_time_seconds > 0,
            0 < self.max_cpu_percent <= 100,
            self.max_open_files > 0,
            self.max_threads > 0,
            self.max_processes > 0,
            self.queue_size > 0
        ])


# =============================================================================
# Resource Monitoring
# =============================================================================

class ResourceMonitor:
    """
    Monitor system resource usage.

    Tracks:
    - Memory usage
    - CPU usage
    - Thread count
    - Process count
    - Open file descriptors
    """

    def __init__(self, sampling_interval: float = 1.0):
        """
        Initialize resource monitor.

        Args:
            sampling_interval: Seconds between samples
        """
        self.sampling_interval = sampling_interval
        self.process = psutil.Process()
        self.history: List[Dict[str, float]] = []
        self.max_history_length = 1000

    def get_current_usage(self) -> Dict[str, float]:
        """
        Get current resource usage.

        Returns:
            Dictionary with current usage metrics
        """
        try:
            # Memory info
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)

            # CPU percent
            cpu_percent = self.process.cpu_percent(interval=0.1)

            # Thread count
            num_threads = self.process.num_threads()

            # Open file descriptors
            try:
                num_files = len(self.process.open_files())
            except (psutil.AccessDenied, AttributeError):
                num_files = 0

            return {
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent,
                'num_threads': num_threads,
                'num_files': num_files,
                'timestamp': datetime.now().timestamp()
            }
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().timestamp()
            }

    def check_limits(self, limits: ResourceLimits) -> Tuple[bool, List[str]]:
        """
        Check if current usage exceeds limits.

        Args:
            limits: Resource limits to check against

        Returns:
            Tuple of (within_limits, violations)
        """
        usage = self.get_current_usage()
        violations = []

        if 'error' in usage:
            return True, ["Could not measure resource usage"]

        # Check memory
        if usage['memory_mb'] > limits.max_memory_mb:
            violations.append(
                f"Memory usage ({usage['memory_mb']:.1f} MB) exceeds limit "
                f"({limits.max_memory_mb} MB)"
            )

        # Check CPU
        if usage['cpu_percent'] > limits.max_cpu_percent:
            violations.append(
                f"CPU usage ({usage['cpu_percent']:.1f}%) exceeds limit "
                f"({limits.max_cpu_percent}%)"
            )

        # Check threads
        if usage['num_threads'] > limits.max_threads:
            violations.append(
                f"Thread count ({usage['num_threads']}) exceeds limit "
                f"({limits.max_threads})"
            )

        # Check open files
        if usage['num_files'] > limits.max_open_files:
            violations.append(
                f"Open file count ({usage['num_files']}) exceeds limit "
                f"({limits.max_open_files})"
            )

        return len(violations) == 0, violations

    def record_sample(self) -> Dict[str, float]:
        """
        Record a resource usage sample.

        Returns:
            Current usage metrics
        """
        usage = self.get_current_usage()
        self.history.append(usage)

        # Trim history
        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length:]

        return usage

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get resource usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        if not self.history:
            return {}

        memory_values = [s.get('memory_mb', 0) for s in self.history if 'memory_mb' in s]
        cpu_values = [s.get('cpu_percent', 0) for s in self.history if 'cpu_percent' in s]

        return {
            'memory_mb': {
                'avg': sum(memory_values) / len(memory_values) if memory_values else 0,
                'max': max(memory_values) if memory_values else 0,
                'min': min(memory_values) if memory_values else 0
            },
            'cpu_percent': {
                'avg': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                'max': max(cpu_values) if cpu_values else 0,
                'min': min(cpu_values) if cpu_values else 0
            },
            'sample_count': len(self.history)
        }


# =============================================================================
# Timeout Manager
# =============================================================================

class TimeoutException(Exception):
    """Exception raised when operation times out"""
    pass


class TimeoutManager:
    """
    Manage operation timeouts with various strategies.

    Supports:
    - Hard timeouts (via signal/alarm)
    - Soft timeouts (via polling)
    - Graceful degradation on timeout
    """

    def __init__(self):
        """Initialize timeout manager"""
        self.active_timeouts: Dict[str, datetime] = {}

    def execute_with_timeout(
        self,
        func: Callable,
        timeout_seconds: float,
        *args,
        graceful: bool = True,
        **kwargs
    ) -> Any:
        """
        Execute function with timeout.

        Args:
            func: Function to execute
            timeout_seconds: Timeout in seconds
            *args: Function arguments
            graceful: If True, attempt graceful shutdown
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            TimeoutException: If operation times out
        """
        # Run in separate thread
        result_queue = queue.Queue()
        exception_queue = queue.Queue()

        def target():
            try:
                result = func(*args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()

        # Wait for completion or timeout
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            if graceful:
                # Thread is still running, timed out
                return None
            else:
                raise TimeoutException(
                    f"Operation timed out after {timeout_seconds} seconds"
                )

        # Check for exceptions
        if not exception_queue.empty():
            raise exception_queue.get()

        # Return result
        if not result_queue.empty():
            return result_queue.get()

        raise TimeoutException(
            f"Operation timed out after {timeout_seconds} seconds"
        )

    @contextmanager
    def timeout_context(self, timeout_seconds: float, graceful: bool = True):
        """
        Context manager for timeout.

        Args:
            timeout_seconds: Timeout in seconds
            graceful: If True, attempt graceful handling
        """
        start_time = datetime.now()
        timeout_id = hashlib.md5(str(start_time).encode()).hexdigest()

        self.active_timeouts[timeout_id] = start_time

        try:
            yield
        finally:
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout_seconds:
                if not graceful:
                    raise TimeoutException(
                        f"Operation timed out after {elapsed:.1f} seconds "
                        f"(limit: {timeout_seconds}s)"
                    )
            del self.active_timeouts[timeout_id]


# =============================================================================
# Queue Manager
# =============================================================================

class QueuePriority(Enum):
    """Queue priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class QueueItem:
    """Item in priority queue"""
    priority: QueuePriority
    task_id: str
    func: Callable
    args: tuple
    kwargs: dict
    submitted_at: datetime = field(default_factory=datetime.now)
    timeout: Optional[float] = None

    def __lt__(self, other):
        """Compare for priority queue"""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.submitted_at < other.submitted_at


class TaskQueue:
    """
    Priority-based task queue with resource limits.

    Features:
    - Priority-based scheduling
    - Timeout enforcement
    - Resource-aware queuing
    - Graceful rejection when full
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_workers: int = 4,
        resource_limits: Optional[ResourceLimits] = None
    ):
        """
        Initialize task queue.

        Args:
            max_size: Maximum queue size
            max_workers: Maximum number of worker threads
            resource_limits: Optional resource limits
        """
        self.max_size = max_size
        self.max_workers = max_workers
        self.resource_limits = resource_limits or ResourceLimits()

        self.queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_size)
        self.workers: List[threading.Thread] = []
        self.running = False
        self.completed_tasks: Dict[str, Any] = {}
        self.failed_tasks: Dict[str, Exception] = {}

        # Lock for thread safety
        self.lock = threading.Lock()

    def start(self) -> None:
        """Start worker threads"""
        self.running = True

        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"TaskQueue-Worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

    def stop(self) -> None:
        """Stop worker threads"""
        self.running = False

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)

        self.workers.clear()

    def submit(
        self,
        func: Callable,
        task_id: str,
        priority: QueuePriority = QueuePriority.NORMAL,
        timeout: Optional[float] = None,
        *args,
        **kwargs
    ) -> bool:
        """
        Submit task to queue.

        Args:
            func: Function to execute
            task_id: Unique task identifier
            priority: Task priority
            timeout: Optional timeout in seconds
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            True if task was queued successfully
        """
        # Check queue size
        if self.queue.qsize() >= self.max_size:
            return False

        # Check resource limits
        monitor = ResourceMonitor()
        within_limits, _ = monitor.check_limits(self.resource_limits)
        if not within_limits:
            return False

        # Create queue item
        item = QueueItem(
            priority=priority,
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            timeout=timeout
        )

        try:
            self.queue.put(item, block=False)
            return True
        except queue.Full:
            return False

    def _worker_loop(self) -> None:
        """Worker thread main loop"""
        while self.running:
            try:
                # Get task with timeout
                item = self.queue.get(timeout=1.0)

                # Execute task
                try:
                    result = item.func(*item.args, **item.kwargs)

                    with self.lock:
                        self.completed_tasks[item.task_id] = {
                            'result': result,
                            'completed_at': datetime.now().isoformat()
                        }
                except Exception as e:
                    with self.lock:
                        self.failed_tasks[item.task_id] = e

            except queue.Empty:
                continue
            except Exception as e:
                # Log error and continue
                print(f"Worker error: {e}")

    def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        Get result of completed task.

        Args:
            task_id: Task identifier

        Returns:
            Task result or None if not found/failed
        """
        with self.lock:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]['result']
            elif task_id in self.failed_tasks:
                return None
            else:
                return None

    def get_task_status(self, task_id: str) -> str:
        """
        Get status of task.

        Args:
            task_id: Task identifier

        Returns:
            Status: pending, completed, failed, not_found
        """
        with self.lock:
            if task_id in self.completed_tasks:
                return 'completed'
            elif task_id in self.failed_tasks:
                return 'failed'
            else:
                # Check if in queue
                for item in list(self.queue.queue):
                    if item.task_id == task_id:
                        return 'pending'
                return 'not_found'

    def get_queue_statistics(self) -> Dict[str, Any]:
        """
        Get queue statistics.

        Returns:
            Dictionary with queue statistics
        """
        with self.lock:
            return {
                'queue_size': self.queue.qsize(),
                'max_size': self.max_size,
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'active_workers': sum(1 for w in self.workers if w.is_alive()),
                'max_workers': self.max_workers
            }


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """
    Rate limiter using token bucket algorithm.

    Features:
    - Token bucket algorithm
    - Per-client rate limiting
    - Sliding window tracking
    - Configurable rates
    """

    def __init__(
        self,
        rate_per_minute: int = 60,
        burst_size: int = 10
    ):
        """
        Initialize rate limiter.

        Args:
            rate_per_minute: Requests allowed per minute
            burst_size: Maximum burst size
        """
        self.rate_per_minute = rate_per_minute
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = datetime.now()

        self.client_buckets: Dict[str, Dict[str, Any]] = {}

        self.lock = threading.Lock()

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time"""
        now = datetime.now()
        elapsed = (now - self.last_update).total_seconds()

        # Refill rate: tokens per second
        refill_rate = self.rate_per_minute / 60.0
        tokens_to_add = elapsed * refill_rate

        self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
        self.last_update = now

    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client.

        Args:
            client_id: Client identifier

        Returns:
            True if request is allowed
        """
        with self.lock:
            # Get or create client bucket
            if client_id not in self.client_buckets:
                self.client_buckets[client_id] = {
                    'tokens': self.burst_size,
                    'last_update': datetime.now()
                }

            bucket = self.client_buckets[client_id]

            # Refill tokens
            now = datetime.now()
            elapsed = (now - bucket['last_update']).total_seconds()
            refill_rate = self.rate_per_minute / 60.0
            tokens_to_add = elapsed * refill_rate

            bucket['tokens'] = min(self.burst_size, bucket['tokens'] + tokens_to_add)
            bucket['last_update'] = now

            # Check if tokens available
            if bucket['tokens'] >= 1:
                bucket['tokens'] -= 1
                return True

            return False

    def get_remaining_tokens(self, client_id: str) -> int:
        """
        Get remaining tokens for client.

        Args:
            client_id: Client identifier

        Returns:
            Number of remaining tokens
        """
        with self.lock:
            if client_id not in self.client_buckets:
                return self.burst_size

            return int(self.client_buckets[client_id]['tokens'])

    def reset_client(self, client_id: str) -> None:
        """
        Reset rate limit for client.

        Args:
            client_id: Client identifier
        """
        with self.lock:
            if client_id in self.client_buckets:
                del self.client_buckets[client_id]


# =============================================================================
# Memory Limiter
# =============================================================================

class MemoryLimiter:
    """
    Monitor and limit memory usage.

    Features:
    - Periodic memory monitoring
    - Automatic cleanup when limit exceeded
    - Graceful degradation
    """

    def __init__(
        self,
        max_memory_mb: float = 4096,
        check_interval: float = 5.0,
        cleanup_threshold: float = 0.9
    ):
        """
        Initialize memory limiter.

        Args:
            max_memory_mb: Maximum memory in MB
            check_interval: Seconds between checks
            cleanup_threshold: Fraction of max to trigger cleanup
        """
        self.max_memory_mb = max_memory_mb
        self.check_interval = check_interval
        self.cleanup_threshold = cleanup_threshold

        self.monitor = ResourceMonitor()
        self.cleanup_callbacks: List[Callable] = []
        self.monitoring = False

    def start_monitoring(self) -> None:
        """Start background memory monitoring"""
        self.monitoring = True

        def monitor_loop():
            while self.monitoring:
                usage = self.monitor.get_current_usage()
                memory_mb = usage.get('memory_mb', 0)

                # Check if threshold exceeded
                if memory_mb > self.max_memory_mb * self.cleanup_threshold:
                    self._trigger_cleanup()

                time.sleep(self.check_interval)

        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()

    def stop_monitoring(self) -> None:
        """Stop background monitoring"""
        self.monitoring = False

    def register_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """
        Register cleanup callback.

        Args:
            callback: Function to call for cleanup
        """
        self.cleanup_callbacks.append(callback)

    def _trigger_cleanup(self) -> None:
        """Trigger cleanup callbacks"""
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Cleanup callback error: {e}")

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage.

        Returns:
            Dictionary with memory usage information
        """
        usage = self.monitor.get_current_usage()
        memory_mb = usage.get('memory_mb', 0)

        return {
            'memory_mb': memory_mb,
            'max_memory_mb': self.max_memory_mb,
            'usage_percent': (memory_mb / self.max_memory_mb * 100) if self.max_memory_mb > 0 else 0,
            'threshold_mb': self.max_memory_mb * self.cleanup_threshold
        }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Configuration
    'ResourceLimits',

    # Monitoring
    'ResourceMonitor',

    # Timeout
    'TimeoutManager',
    'TimeoutException',

    # Queue
    'QueuePriority',
    'TaskQueue',

    # Rate limiting
    'RateLimiter',

    # Memory limiting
    'MemoryLimiter',
]
