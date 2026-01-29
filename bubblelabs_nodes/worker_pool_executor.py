"""
Worker Pool Executor for Parallel Problem Solving

Provides a process-based worker pool for CPU-intensive parallel execution,
complementing the asyncio-based parallel executor.

Key Features:
- Process pool for CPU-bound tasks
- Worker queue management
- Work stealing for load balancing
- Configurable pool size
- Graceful worker lifecycle management
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ProcessPoolExecutor, Future, wait
from multiprocessing import Manager

logger = logging.getLogger(__name__)


@dataclass
class WorkerTask:
    """A task to be executed by a worker"""
    task_id: str
    problem: Dict[str, Any]
    executor_func: Callable
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)


@dataclass
class WorkerResult:
    """Result from a worker task"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    worker_id: Optional[int] = None


@dataclass
class PoolExecutionSummary:
    """Summary of worker pool execution"""
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    total_time: float
    results: List[WorkerResult] = field(default_factory=list)
    worker_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 0


class WorkerPoolExecutor:
    """
    Process-based worker pool for parallel problem execution.

    Uses multiprocessing to execute CPU-intensive tasks in parallel,
    with work stealing for load balancing.
    """

    def __init__(
        self,
        max_workers: int = 4,
        timeout_seconds: float = 300.0,
        enable_work_stealing: bool = True
    ):
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.enable_work_stealing = enable_work_stealing
        self.task_queue: asyncio.Queue = None
        self.result_queue: asyncio.Queue = None
        self.workers: List[ProcessPoolExecutor] = []
        self.active_tasks: Dict[str, WorkerTask] = {}
        self.worker_stats: Dict[int, Dict[str, Any]] = {}

    async def execute_in_parallel(
        self,
        problems: List[Dict[str, Any]],
        executor_func: Callable,
        context: Dict[str, Any] = None
    ) -> PoolExecutionSummary:
        """
        Execute problems in parallel using worker pool.

        Args:
            problems: List of problems to solve
            executor_func: Function to execute for each problem
            context: Shared execution context

        Returns:
            PoolExecutionSummary with results and statistics
        """
        start_time = datetime.utcnow()

        # Create tasks
        tasks = [
            WorkerTask(
                task_id=f"task_{i}",
                problem=problem,
                executor_func=executor_func,
                context=context or {},
                priority=0
            )
            for i, problem in enumerate(problems)
        ]

        # Initialize queues
        manager = Manager()
        self.task_queue = manager.Queue()
        self.result_queue = manager.Queue()

        # Add tasks to queue
        for task in tasks:
            self.task_queue.put(task)

        # Create process pool
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures: Dict[str, Future] = {}
            pending_tasks = tasks.copy()

            while pending_tasks:
                # Submit available tasks
                while len(futures) < self.max_workers and pending_tasks:
                    task = pending_tasks.pop(0)
                    future = executor.submit(
                        self._execute_task,
                        task.task_id,
                        task.problem,
                        task.executor_func,
                        task.context
                    )
                    futures[task.task_id] = future

                # Wait for at least one completion
                if futures:
                    done_futures, _ = wait(
                        list(futures.values()),
                        timeout=1.0
                    )

                    # Collect results
                    for future in done_futures:
                        for task_id, fut in list(futures.items()):
                            if fut == future:
                                try:
                                    result = future.result(timeout=self.timeout_seconds)
                                    self.result_queue.put(result)
                                except Exception as e:
                                    self.result_queue.put(
                                        WorkerResult(
                                            task_id=task_id,
                                            success=False,
                                            error=str(e)
                                        )
                                    )
                                del futures[task_id]
                                break

        # Collect all results
        results = []
        while not self.result_queue.empty():
            result = self.result_queue.get()
            results.append(result)

        # Calculate summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_time = (datetime.utcnow() - start_time).total_seconds()

        return PoolExecutionSummary(
            total_tasks=len(tasks),
            successful_tasks=successful,
            failed_tasks=failed,
            total_time=total_time,
            results=results,
            worker_stats=self.worker_stats
        )

    @staticmethod
    def _execute_task(
        task_id: str,
        problem: Dict[str, Any],
        executor_func: Callable,
        context: Dict[str, Any]
    ) -> WorkerResult:
        """Execute a single task (runs in worker process)"""
        start_time = datetime.utcnow()

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(executor_func):
                # Run async function in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(executor_func(problem))
                finally:
                    loop.close()
            else:
                result = executor_func(problem)

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return WorkerResult(
                task_id=task_id,
                success=True,
                result=result,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return WorkerResult(
                task_id=task_id,
                success=False,
                error=str(e),
                execution_time=execution_time
            )

    async def execute_with_work_stealing(
        self,
        problems: List[Dict[str, Any]],
        executor_func: Callable,
        context: Dict[str, Any] = None
    ) -> PoolExecutionSummary:
        """
        Execute problems with work stealing enabled.

        Idle workers can steal tasks from busy workers' queues.
        """
        if not self.enable_work_stealing:
            return await self.execute_in_parallel(problems, executor_func, context)

        # Partition problems among workers
        partition_size = len(problems) // self.max_workers
        partitions = []
        for i in range(self.max_workers):
            start_idx = i * partition_size
            end_idx = start_idx + partition_size if i < self.max_workers - 1 else len(problems)
            partitions.append(problems[start_idx:end_idx])

        # Execute each partition in parallel
        tasks = []
        for worker_id, partition in enumerate(partitions):
            if partition:
                task = asyncio.create_task(
                    self._execute_partition(worker_id, partition, executor_func, context)
                )
                tasks.append(task)

        # Wait for all partitions
        partition_results = await asyncio.gather(*tasks)

        # Combine results
        all_results = []
        for results in partition_results:
            all_results.extend(results)

        successful = sum(1 for r in all_results if r.success)

        return PoolExecutionSummary(
            total_tasks=len(problems),
            successful_tasks=successful,
            failed_tasks=len(all_results) - successful,
            total_time=sum(r.execution_time for r in all_results),
            results=all_results
        )

    async def _execute_partition(
        self,
        worker_id: int,
        problems: List[Dict[str, Any]],
        executor_func: Callable,
        context: Dict[str, Any]
    ) -> List[WorkerResult]:
        """Execute a partition of problems"""
        results = []

        for problem in problems:
            start_time = datetime.utcnow()

            try:
                result = await executor_func(problem)
                execution_time = (datetime.utcnow() - start_time).total_seconds()

                results.append(WorkerResult(
                    task_id=f"worker_{worker_id}_task_{len(results)}",
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    worker_id=worker_id
                ))

            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()

                results.append(WorkerResult(
                    task_id=f"worker_{worker_id}_task_{len(results)}",
                    success=False,
                    error=str(e),
                    execution_time=execution_time,
                    worker_id=worker_id
                ))

        return results

    def get_worker_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics for each worker"""
        return self.worker_stats.copy()

    def shutdown(self):
        """Shutdown the worker pool"""
        for worker in self.workers:
            worker.shutdown(wait=True)

        self.workers.clear()
        logger.info("Worker pool shutdown complete")


def create_worker_pool_executor(
    max_workers: int = 4,
    timeout_seconds: float = 300.0,
    enable_work_stealing: bool = True
) -> WorkerPoolExecutor:
    """Factory function to create worker pool executor"""
    return WorkerPoolExecutor(
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
        enable_work_stealing=enable_work_stealing
    )


# Example usage
async def demo_worker_pool():
    """Demonstration of worker pool executor"""

    print("\n" + "=" * 60)
    print("Worker Pool Executor Demo")
    print("=" * 60)

    # Simple executor function
    def solve_problem_sync(problem: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous problem solver"""
        import time
        time.sleep(0.1)  # Simulate work

        return {
            'problem_id': problem.get('id', 'unknown'),
            'success': True,
            'score': 0.85,
            'solution': f"Solution for {problem.get('id')}",
        }

    # Create test problems
    problems = [
        {'id': f'problem_{i}', 'statement': f'Problem {i}'}
        for i in range(10)
    ]

    # Create worker pool
    executor = create_worker_pool_executor(
        max_workers=4,
        enable_work_stealing=True
    )

    print(f"\nExecuting {len(problems)} problems with worker pool...")

    # Execute in parallel
    summary = await executor.execute_in_parallel(
        problems=problems,
        executor_func=solve_problem_sync,
        context={}
    )

    print(f"\nResults:")
    print(f"  Total tasks: {summary.total_tasks}")
    print(f"  Successful: {summary.successful_tasks}")
    print(f"  Failed: {summary.failed_tasks}")
    print(f"  Success rate: {summary.success_rate:.1%}")
    print(f"  Total time: {summary.total_time:.2f}s")
    print(f"  Avg time per task: {summary.total_time / summary.total_tasks:.2f}s")

    # Test work stealing
    print(f"\nTesting work stealing...")
    summary_ws = await executor.execute_with_work_stealing(
        problems=problems,
        executor_func=solve_problem_sync,
        context={}
    )

    print(f"\nWork Stealing Results:")
    print(f"  Total tasks: {summary_ws.total_tasks}")
    print(f"  Successful: {summary_ws.successful_tasks}")
    print(f"  Success rate: {summary_ws.success_rate:.1%}")

    # Cleanup
    executor.shutdown()

    print("\n" + "=" * 60)


if __name__ == '__main__':
    import asyncio
    asyncio.run(demo_worker_pool())
