"""
Distributed Processing Module

This module enables distributed processing of sub-problems across multiple
workers/nodes for large-scale problem solving.
"""
from __future__ import annotations


import concurrent.futures
from typing import List, Dict, Any, Callable, Optional
import multiprocessing as mp
from queue import Queue
import threading
import time
import logging
from dataclasses import dataclass, field
from enum import Enum

from workflow_structures import SubProblem, DecompositionPlan, WorkflowState, SolutionAttempt

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Status of a worker node."""
    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"
    SHUTDOWN = "shutdown"


@dataclass
class WorkerInfo:
    """Information about a worker node."""
    worker_id: str
    status: WorkerStatus
    current_task: Optional[str] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    last_heartbeat: float = field(default_factory=time.time)
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0


@dataclass
class TaskInfo:
    """Information about a task being processed."""
    task_id: str
    sub_problem_id: str
    worker_id: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3


class SyncManager:
    """Manages synchronization and data consistency across distributed workers."""
    
    def __init__(self):
        """Initialize synchronization manager."""
        self.lock = threading.Lock()
        self.shared_state: Dict[str, Any] = {}
        self.version_counter = 0
    
    def update_shared_state(self, key: str, value: Any) -> None:
        """
        Update shared state with thread safety.
        
        Args:
            key: State key
            value: State value
        """
        with self.lock:
            self.shared_state[key] = value
            self.version_counter += 1
            logger.debug(f"Updated shared state: {key} = {value}")
    
    def get_shared_state(self, key: str) -> Optional[Any]:
        """
        Get value from shared state.
        
        Args:
            key: State key
            
        Returns:
            State value or None if not found
        """
        with self.lock:
            return self.shared_state.get(key)
    
    def get_all_state(self) -> Dict[str, Any]:
        """Get a copy of all shared state."""
        with self.lock:
            return self.shared_state.copy()
    
    def clear_state(self) -> None:
        """Clear all shared state."""
        with self.lock:
            self.shared_state.clear()
            self.version_counter = 0


class WorkerNode:
    """Represents a worker node that executes sub-problem solving."""
    
    def __init__(self, worker_id: str):
        """
        Initialize worker node.
        
        Args:
            worker_id: Unique identifier for this worker
        """
        self.info = WorkerInfo(
            worker_id=worker_id,
            status=WorkerStatus.IDLE
        )
        self.heartbeat_interval = 5.0  # seconds
        self._stop_heartbeat = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
    
    def start_heartbeat(self, callback: Callable[[str], None]) -> None:
        """
        Start sending heartbeats.
        
        Args:
            callback: Function to call with worker_id on each heartbeat
        """
        def heartbeat_loop():
            while not self._stop_heartbeat.is_set():
                self.info.last_heartbeat = time.time()
                callback(self.info.worker_id)
                time.sleep(self.heartbeat_interval)
        
        self._heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
    
    def stop_heartbeat(self) -> None:
        """Stop sending heartbeats."""
        self._stop_heartbeat.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=1.0)
    
    def process_sub_problem(
        self,
        sub_problem: SubProblem,
        solver_function: Callable,
        context: Dict[str, Any]
    ) -> SolutionAttempt:
        """
        Process a sub-problem.
        
        Args:
            sub_problem: Sub-problem to solve
            solver_function: Function to solve the sub-problem
            context: Execution context
            
        Returns:
            Solution attempt
        """
        self.info.status = WorkerStatus.BUSY
        self.info.current_task = sub_problem.id
        start_time = time.time()
        
        try:
            solution = solver_function(sub_problem, context)
            self.info.tasks_completed += 1
            return solution
        except Exception as e:
            self.info.tasks_failed += 1
            logger.error(f"Worker {self.info.worker_id} failed to solve {sub_problem.id}: {e}")
            raise
        finally:
            end_time = time.time()
            self.info.total_processing_time += (end_time - start_time)
            self.info.status = WorkerStatus.IDLE
            self.info.current_task = None
    
    def report_status(self) -> WorkerInfo:
        """Get current worker status."""
        return self.info
    
    def shutdown(self) -> None:
        """Shutdown the worker."""
        self.stop_heartbeat()
        self.info.status = WorkerStatus.SHUTDOWN


class DistributedCoordinator:
    """Coordinates distributed processing with worker management and failure handling."""
    
    def __init__(self, max_workers: int = None):
        """
        Initialize distributed coordinator.
        
        Args:
            max_workers: Maximum number of workers
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.workers: Dict[str, WorkerInfo] = {}
        self.tasks: Dict[str, TaskInfo] = {}
        self.sync_manager = SyncManager()
        self.heartbeat_timeout = 30.0  # seconds
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()
        
        # Initialize workers
        for i in range(self.max_workers):
            worker_id = f"worker_{i}"
            self.workers[worker_id] = WorkerInfo(
                worker_id=worker_id,
                status=WorkerStatus.IDLE
            )
    
    def start_monitoring(self) -> None:
        """Start monitoring worker health."""
        def monitor_loop():
            while not self._stop_monitor.is_set():
                self._check_worker_health()
                time.sleep(5.0)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop monitoring workers."""
        self._stop_monitor.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def _check_worker_health(self) -> None:
        """Check health of all workers and handle failures."""
        current_time = time.time()
        for worker_id, worker_info in self.workers.items():
            if worker_info.status == WorkerStatus.BUSY:
                time_since_heartbeat = current_time - worker_info.last_heartbeat
                if time_since_heartbeat > self.heartbeat_timeout:
                    logger.warning(f"Worker {worker_id} appears to have failed (no heartbeat)")
                    self.handle_worker_failure(worker_id)
    
    def handle_worker_failure(self, worker_id: str) -> None:
        """
        Handle worker failure by reassigning its task.
        
        Args:
            worker_id: ID of the failed worker
        """
        worker_info = self.workers.get(worker_id)
        if not worker_info:
            return
        
        # Mark worker as failed
        worker_info.status = WorkerStatus.FAILED
        
        # Find task assigned to this worker
        failed_task = None
        for task_id, task_info in self.tasks.items():
            if task_info.worker_id == worker_id and task_info.status == "running":
                failed_task = task_info
                break
        
        if failed_task:
            logger.info(f"Reassigning task {failed_task.task_id} from failed worker {worker_id}")
            failed_task.status = "pending"
            failed_task.worker_id = None
            failed_task.retry_count += 1
            
            if failed_task.retry_count >= failed_task.max_retries:
                logger.error(f"Task {failed_task.task_id} exceeded max retries")
                failed_task.status = "failed"
            else:
                # Task will be reassigned by the scheduler
                logger.info(
                    "Task %s queued for retry %s/%s",
                    failed_task.task_id,
                    failed_task.retry_count,
                    failed_task.max_retries
                )
    
    def distribute_sub_problems(
        self,
        sub_problems: List[SubProblem],
        dependencies: Dict[str, List[str]]
    ) -> None:
        """
        Distribute sub-problems to workers based on dependencies.
        
        Args:
            sub_problems: List of sub-problems to distribute
            dependencies: Dependency mapping
        """
        for sp in sub_problems:
            task_id = f"task_{sp.id}"
            self.tasks[task_id] = TaskInfo(
                task_id=task_id,
                sub_problem_id=sp.id
            )
    
    def get_worker_status(self) -> Dict[str, WorkerInfo]:
        """Get status of all workers."""
        return self.workers.copy()
    
    def collect_results(self) -> Dict[str, SolutionAttempt]:
        """Collect results from completed tasks."""
        results = {}
        for task_id, task_info in self.tasks.items():
            if task_info.status == "completed":
                # Results would be stored in sync_manager
                result = self.sync_manager.get_shared_state(f"result_{task_info.sub_problem_id}")
                if result:
                    results[task_info.sub_problem_id] = result
        return results
    
    def shutdown(self) -> None:
        """Shutdown the coordinator."""
        self.stop_monitoring()
        for worker_info in self.workers.values():
            worker_info.status = WorkerStatus.SHUTDOWN


class DistributedProcessor:
    """Manages distributed processing of sub-problems with enhanced failure handling."""
    
    def __init__(self, max_workers: int = None):
        """
        Initialize distributed processor.
        
        Args:
            max_workers: Maximum number of parallel workers (defaults to CPU count)
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.results_queue = Queue()
        self.active_tasks: Dict[str, Any] = {}
        self.coordinator = DistributedCoordinator(max_workers)
        self.coordinator.start_monitoring()
    
    def process_sub_problems_distributed(
        self,
        sub_problems: List[SubProblem],
        solver_function: Callable,
        context: Dict[str, Any]
    ) -> Dict[str, SolutionAttempt]:
        """
        Process multiple sub-problems in parallel across workers with failure handling.
        
        Args:
            sub_problems: List of sub-problems to solve
            solver_function: Function to solve each sub-problem
            context: Context for solving
            
        Returns:
            Dictionary mapping sub-problem IDs to solutions
        """
        solutions = {}
        
        # Distribute tasks to coordinator
        dependencies = {sp.id: sp.dependencies for sp in sub_problems}
        self.coordinator.distribute_sub_problems(sub_problems, dependencies)
        
        # Use ThreadPoolExecutor for better monitoring and failure handling
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all sub-problems
            future_to_sp = {}
            for sp in sub_problems:
                future = executor.submit(self._solve_sub_problem_with_retry, sp, solver_function, context)
                future_to_sp[future] = sp
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_sp):
                sp = future_to_sp[future]
                try:
                    solution = future.result(timeout=300)  # 5 minute timeout per task
                    solutions[sp.id] = solution
                    # Store in sync manager
                    self.coordinator.sync_manager.update_shared_state(f"result_{sp.id}", solution)
                except concurrent.futures.TimeoutError:
                    logger.error(f"Timeout solving sub-problem {sp.id}")
                    solutions[sp.id] = None
                except Exception as e:
                    logger.error(f"Error solving sub-problem {sp.id}: {e}")
                    solutions[sp.id] = None
        
        return solutions
    
    def _solve_sub_problem_with_retry(
        self,
        sub_problem: SubProblem,
        solver_function: Callable,
        context: Dict[str, Any],
        max_retries: int = 3
    ) -> SolutionAttempt:
        """
        Solve a sub-problem with retry logic.
        
        Args:
            sub_problem: Sub-problem to solve
            solver_function: Function to solve the sub-problem
            context: Execution context
            max_retries: Maximum number of retry attempts
            
        Returns:
            Solution attempt
        """
        last_exception = None
        for attempt in range(max_retries):
            try:
                return solver_function(sub_problem, context)
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {sub_problem.id}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # All retries failed
        logger.error(f"All {max_retries} attempts failed for {sub_problem.id}")
        raise last_exception
    
    def _solve_sub_problem_wrapper(
        self,
        sub_problem: SubProblem,
        solver_function: Callable,
        context: Dict[str, Any]
    ) -> SolutionAttempt:
        """Wrapper for solving a sub-problem (must be picklable for multiprocessing)."""
        return solver_function(sub_problem, context)
    
    def process_with_dependency_resolution(
        self,
        plan: DecompositionPlan,
        solver_function: Callable,
        context: Dict[str, Any]
    ) -> Dict[str, SolutionAttempt]:
        """
        Process sub-problems respecting dependencies using parallel execution.
        
        Args:
            plan: Decomposition plan with dependencies
            solver_function: Function to solve each sub-problem
            context: Context for solving
            
        Returns:
            Dictionary mapping sub-problem IDs to solutions
        """
        solutions = {}
        solved_ids = set()
        
        # Build dependency graph
        sub_problems_by_id = {sp.id: sp for sp in plan.sub_problems}
        in_degree = {sp.id: len(sp.dependencies) for sp in plan.sub_problems}
        
        # Process in waves based on dependencies
        while len(solved_ids) < len(plan.sub_problems):
            # Find sub-problems ready to solve (all dependencies met)
            ready_to_solve = [
                sp for sp in plan.sub_problems
                if sp.id not in solved_ids and all(dep in solved_ids for dep in sp.dependencies)
            ]
            
            if not ready_to_solve:
                # No progress possible (circular dependencies or error)
                break
            
            # Solve this wave in parallel
            wave_solutions = self.process_sub_problems_distributed(
                ready_to_solve,
                solver_function,
                context
            )
            
            # Update solutions and solved set
            solutions.update(wave_solutions)
            solved_ids.update(sp.id for sp in ready_to_solve if wave_solutions.get(sp.id))
        
        return solutions
    
    def get_optimal_worker_count(self, num_tasks: int) -> int:
        """
        Calculate optimal number of workers for given task count.
        
        Args:
            num_tasks: Number of tasks to process
            
        Returns:
            Optimal worker count
        """
        # Don't use more workers than tasks
        return min(self.max_workers, num_tasks)


class LoadBalancer:
    """Balances load across multiple workers."""
    
    def __init__(self, num_workers: int):
        """
        Initialize load balancer.
        
        Args:
            num_workers: Number of workers to balance across
        """
        self.num_workers = num_workers
        self.worker_loads: Dict[int, int] = {i: 0 for i in range(num_workers)}
    
    def assign_task(self, task_weight: int = 1) -> int:
        """
        Assign a task to the least loaded worker.
        
        Args:
            task_weight: Weight/complexity of the task
            
        Returns:
            Worker ID to assign task to
        """
        # Find worker with minimum load
        min_worker = min(self.worker_loads.items(), key=lambda x: x[1])
        worker_id = min_worker[0]
        
        # Update load
        self.worker_loads[worker_id] += task_weight
        
        return worker_id
    
    def complete_task(self, worker_id: int, task_weight: int = 1):
        """
        Mark a task as complete and update worker load.
        
        Args:
            worker_id: Worker that completed the task
            task_weight: Weight of the completed task
        """
        self.worker_loads[worker_id] = max(0, self.worker_loads[worker_id] - task_weight)
    
    def get_load_distribution(self) -> Dict[int, int]:
        """Get current load distribution across workers."""
        return self.worker_loads.copy()


class TaskScheduler:
    """Schedules tasks for optimal distributed execution."""
    
    def __init__(self):
        """Initialize task scheduler."""
        self.task_queue: List[Dict[str, Any]] = []
        self.completed_tasks: List[Dict[str, Any]] = []
    
    def schedule_tasks(
        self,
        sub_problems: List[SubProblem],
        num_workers: int
    ) -> List[List[SubProblem]]:
        """
        Schedule sub-problems across workers for optimal execution.
        
        Args:
            sub_problems: List of sub-problems to schedule
            num_workers: Number of available workers
            
        Returns:
            List of sub-problem batches for each worker
        """
        # Sort by complexity (descending) for better load balancing
        sorted_problems = sorted(
            sub_problems,
            key=lambda sp: sp.ai_suggested_complexity_score,
            reverse=True
        )
        
        # Distribute using round-robin with complexity awareness
        worker_batches = [[] for _ in range(num_workers)]
        worker_loads = [0] * num_workers
        
        for sp in sorted_problems:
            # Assign to worker with minimum load
            min_load_worker = worker_loads.index(min(worker_loads))
            worker_batches[min_load_worker].append(sp)
            worker_loads[min_load_worker] += sp.ai_suggested_complexity_score
        
        return worker_batches
    
    def estimate_execution_time(
        self,
        sub_problems: List[SubProblem],
        avg_time_per_complexity: float = 60.0
    ) -> float:
        """
        Estimate total execution time for sub-problems.
        
        Args:
            sub_problems: List of sub-problems
            avg_time_per_complexity: Average time per complexity point (seconds)
            
        Returns:
            Estimated execution time in seconds
        """
        total_complexity = sum(sp.ai_suggested_complexity_score for sp in sub_problems)
        return total_complexity * avg_time_per_complexity


class DistributedWorkflowExecutor:
    """Executes workflows using distributed processing."""
    
    def __init__(self, max_workers: int = None):
        """
        Initialize distributed workflow executor.
        
        Args:
            max_workers: Maximum number of parallel workers
        """
        self.processor = DistributedProcessor(max_workers)
        self.scheduler = TaskScheduler()
        self.load_balancer = LoadBalancer(max_workers or mp.cpu_count())
    
    def execute_workflow_distributed(
        self,
        workflow_state: WorkflowState,
        solver_function: Callable,
        context: Dict[str, Any]
    ) -> Dict[str, SolutionAttempt]:
        """
        Execute workflow using distributed processing.
        
        Args:
            workflow_state: Workflow state
            solver_function: Function to solve sub-problems
            context: Execution context
            
        Returns:
            Dictionary of solutions
        """
        if not workflow_state.decomposition_plan:
            return {}
        
        # Process with dependency resolution
        solutions = self.processor.process_with_dependency_resolution(
            workflow_state.decomposition_plan,
            solver_function,
            context
        )
        
        return solutions
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get statistics about distributed execution."""
        worker_status = self.processor.coordinator.get_worker_status()
        
        return {
            "max_workers": self.processor.max_workers,
            "load_distribution": self.load_balancer.get_load_distribution(),
            "active_tasks": len(self.processor.active_tasks),
            "worker_status": {
                wid: {
                    "status": winfo.status.value,
                    "current_task": winfo.current_task,
                    "tasks_completed": winfo.tasks_completed,
                    "tasks_failed": winfo.tasks_failed,
                    "total_processing_time": winfo.total_processing_time
                }
                for wid, winfo in worker_status.items()
            },
            "sync_state_version": self.processor.coordinator.sync_manager.version_counter
        }
    
    def get_worker_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed status of all workers.
        
        Returns:
            Dictionary mapping worker IDs to their status information
        """
        worker_status = self.processor.coordinator.get_worker_status()
        return {
            wid: {
                "status": winfo.status.value,
                "current_task": winfo.current_task,
                "resource_usage": winfo.resource_usage,
                "last_heartbeat": winfo.last_heartbeat,
                "tasks_completed": winfo.tasks_completed,
                "tasks_failed": winfo.tasks_failed,
                "total_processing_time": winfo.total_processing_time
            }
            for wid, winfo in worker_status.items()
        }
    
    def shutdown(self) -> None:
        """Shutdown the distributed executor."""
        self.processor.coordinator.shutdown()


# Global distributed executor instance
_global_executor: Optional[DistributedWorkflowExecutor] = None


def get_distributed_executor(max_workers: int = None) -> DistributedWorkflowExecutor:
    """Get or create the global distributed executor."""
    global _global_executor
    if _global_executor is None:
        _global_executor = DistributedWorkflowExecutor(max_workers)
    return _global_executor


def enable_distributed_processing(max_workers: int = None):
    """Enable distributed processing with specified worker count."""
    global _global_executor
    _global_executor = DistributedWorkflowExecutor(max_workers)
    return _global_executor


def run_distributed_evolution(content: str, api_key: str, num_workers: int = 4) -> Dict[str, Any]:
    """Run distributed OpenEvolve evolution"""
    try:
        from openevolve_client import OpenEvolveClient
        
        client = OpenEvolveClient(api_key=api_key)
        
        result = client.evolve(
            content=content,
            evolution_mode="standard",
            max_iterations=20,
            population_size=50,
            temperature=0.7,
            distributed=True,
            parallel_evaluations=num_workers
        )
        
        return result
    except Exception as e:
        return {'error': str(e)}
