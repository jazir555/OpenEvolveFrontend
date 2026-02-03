"""
Sovereign-Grade Problem Decomposition System - Parallel Processing
Implements concurrent processing for independent sub-problems and gauntlet executions.
"""

import asyncio
import concurrent.futures
import threading
import multiprocessing
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import time
from functools import partial
import queue
import os

from sovereign_data_models import SubProblem, DecompositionPlan, SolutionAttempt, generate_id
from sovereign_reliability import with_error_handling, ErrorSeverity



@dataclass
class ParallelTaskResult:
    """Result of a parallel task execution."""
    task_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    worker_id: Optional[int] = None


class TaskScheduler:
    """Manages scheduling and execution of tasks with dependency awareness."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize task scheduler.
        
        Args:
            max_workers: Maximum number of concurrent workers (defaults to CPU count)
        """
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.logger = logging.getLogger(__name__)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.task_queue = queue.Queue()
        self.running_tasks = {}
        self.results = {}
    
    def schedule_subproblems_with_dependencies(
        self, 
        sub_problems: List[SubProblem], 
        dependency_graph: Dict[str, List[str]],
        task_func: Callable,
        *args,
        **kwargs
    ) -> List[ParallelTaskResult]:
        """
        Schedule sub-problems for parallel execution respecting dependencies.
        
        Args:
            sub_problems: List of sub-problems to process
            dependency_graph: Dictionary mapping sub-problem ID to list of dependency IDs
            task_func: Function to execute for each sub-problem
            *args: Additional arguments to pass to task function
            **kwargs: Additional keyword arguments to pass to task function
            
        Returns:
            List of task results
        """
        # Group sub-problems by execution level (no dependencies at same level can run in parallel)
        execution_levels = self._build_execution_levels(sub_problems, dependency_graph)
        
        all_results = []
        
        for level_idx, level in enumerate(execution_levels):
            self.logger.info(f"Executing level {level_idx + 1}/{len(execution_levels)} with {len(level)} tasks")
            
            # Execute all tasks in this level in parallel
            level_tasks = []
            for sub_problem in level:
                task_future = self.executor.submit(
                    self._execute_task_with_error_handling,
                    sub_problem,
                    task_func,
                    *args,
                    **kwargs
                )
                level_tasks.append((sub_problem.id, task_future))
            
            # Wait for all tasks in this level to complete
            for sub_id, future in level_tasks:
                result = future.result()
                all_results.append(result)
        
        return all_results
    
    def _build_execution_levels(
        self, 
        sub_problems: List[SubProblem], 
        dependency_graph: Dict[str, List[str]]
    ) -> List[List[SubProblem]]:
        """Build execution levels based on dependencies."""
        # Create a copy of dependencies to track remaining dependencies
        remaining_deps = {sp.id: set(dependency_graph.get(sp.id, [])) for sp in sub_problems}
        
        levels = []
        remaining_problems = set(sp.id for sp in sub_problems)
        
        while remaining_problems:
            # Find all sub-problems with no remaining dependencies
            ready_problems = [
                sp for sp in sub_problems 
                if sp.id in remaining_problems and len(remaining_deps[sp.id]) == 0
            ]
            
            if not ready_problems:
                # Circular dependency detected - break to avoid infinite loop
                self.logger.error("Circular dependency detected in sub-problems")
                break
            
            levels.append(ready_problems)
            
            # Remove these problems from remaining and from other dependencies
            for sp in ready_problems:
                remaining_problems.remove(sp.id)
                
                # Remove this problem as a dependency from others
                for other_id in remaining_problems:
                    remaining_deps[other_id].discard(sp.id)
        
        return levels
    
    def _execute_task_with_error_handling(
        self, 
        sub_problem: SubProblem, 
        task_func: Callable, 
        *args, 
        **kwargs
    ) -> ParallelTaskResult:
        """Execute a task with error handling."""
        start_time = time.time()
        
        try:
            result = task_func(sub_problem, *args, **kwargs)
            execution_time = time.time() - start_time
            
            return ParallelTaskResult(
                task_id=sub_problem.id,
                success=True,
                result=result,
                execution_time=execution_time,
                worker_id=threading.get_ident()
            )
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Task failed for sub-problem {sub_problem.id}: {str(e)}"
            self.logger.error(error_msg)
            
            return ParallelTaskResult(
                task_id=sub_problem.id,
                success=False,
                result=None,
                error=error_msg,
                execution_time=execution_time,
                worker_id=threading.get_ident()
            )
    
    def execute_parallel_tasks(
        self, 
        tasks: List[Tuple[Callable, Tuple, Dict]], 
        timeout: Optional[float] = None
    ) -> List[ParallelTaskResult]:
        """
        Execute a list of tasks in parallel.
        
        Args:
            tasks: List of tuples (function, args, kwargs) to execute
            timeout: Optional timeout in seconds
            
        Returns:
            List of task results
        """
        futures = []
        
        for func, args, kwargs in tasks:
            future = self.executor.submit(
                self._execute_standalone_task,
                func,
                *args,
                **kwargs
            )
            futures.append(future)
        
        results = []
        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            results.append(future.result())
        
        return results
    
    def _execute_standalone_task(
        self, 
        func: Callable, 
        *args, 
        **kwargs
    ) -> ParallelTaskResult:
        """Execute a standalone task with error handling."""
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            task_id = generate_id("task")
            return ParallelTaskResult(
                task_id=task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                worker_id=threading.get_ident()
            )
        except Exception as e:
            execution_time = time.time() - start_time
            task_id = generate_id("task")
            error_msg = f"Task failed: {str(e)}"
            self.logger.error(error_msg)
            
            return ParallelTaskResult(
                task_id=task_id,
                success=False,
                result=None,
                error=error_msg,
                execution_time=execution_time,
                worker_id=threading.get_ident()
            )
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        self.executor.shutdown(wait=wait)


class ParallelDecompositionProcessor:
    """Manages parallel processing of sub-problems during decomposition."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize parallel decomposition processor.
        
        Args:
            max_workers: Maximum number of concurrent workers
        """
        self.scheduler = TaskScheduler(max_workers)
        self.logger = logging.getLogger(__name__)
    
    @with_error_handling(severity=ErrorSeverity.HIGH, fallback=lambda *args, **kwargs: [])
    def process_subproblems_in_parallel(
        self, 
        sub_problems: List[SubProblem], 
        dependency_graph: Dict[str, List[str]], 
        process_func: Callable
    ) -> List[ParallelTaskResult]:
        """
        Process sub-problems in parallel respecting dependencies.
        
        Args:
            sub_problems: List of sub-problems to process
            dependency_graph: Dictionary of dependencies between sub-problems
            process_func: Function to process each sub-problem
            
        Returns:
            List of processing results
        """
        self.logger.info(f"Processing {len(sub_problems)} sub-problems in parallel with dependencies")
        
        results = self.scheduler.schedule_subproblems_with_dependencies(
            sub_problems,
            dependency_graph,
            process_func
        )
        
        successful_count = sum(1 for r in results if r.success)
        self.logger.info(f"Parallel processing completed: {successful_count}/{len(results)} tasks successful")
        
        return results
    
    @with_error_handling(severity=ErrorSeverity.HIGH, fallback=lambda *args, **kwargs: [])
    def execute_gauntlets_in_parallel(
        self, 
        plan: DecompositionPlan, 
        gauntlet_functions: List[Callable]
    ) -> Dict[str, Any]:
        """
        Execute multiple gauntlets in parallel.
        
        Args:
            plan: The decomposition plan to validate
            gauntlet_functions: List of gauntlet functions to execute
            
        Returns:
            Dictionary with results from all gauntlets
        """
        self.logger.info(f"Executing {len(gauntlet_functions)} gauntlets in parallel for plan {plan.id}")
        
        # Create tasks for each gauntlet
        tasks = []
        for gauntlet_func in gauntlet_functions:
            task = (gauntlet_func, (plan,), {})
            tasks.append(task)
        
        results = self.scheduler.execute_parallel_tasks(tasks)
        
        # Organize results
        organized_results = {}
        for i, result in enumerate(results):
            gauntlet_name = f"gauntlet_{i}"
            organized_results[gauntlet_name] = {
                'success': result.success,
                'result': result.result,
                'execution_time': result.execution_time,
                'error': result.error
            }
        
        return organized_results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get parallel processing statistics."""
        return {
            'max_workers': self.scheduler.max_workers,
            'executor_stats': {
                'pending_tasks': self.scheduler.executor._work_queue.qsize() if hasattr(self.scheduler.executor, '_work_queue') else 0
            }
        }


class AsyncSolutionProcessor:
    """Asynchronous solution processor for handling solution attempts."""
    
    def __init__(self, max_concurrent: int = 10):
        """
        Initialize async solution processor.
        
        Args:
            max_concurrent: Maximum number of concurrent solution attempts
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = logging.getLogger(__name__)
    
    async def process_solution_attempts(
        self, 
        solution_attempts: List[SolutionAttempt], 
        process_func: Callable
    ) -> List[ParallelTaskResult]:
        """
        Process solution attempts asynchronously.
        
        Args:
            solution_attempts: List of solution attempts to process
            process_func: Async function to process each solution attempt
            
        Returns:
            List of processing results
        """
        tasks = [
            self._process_with_semaphore(
                attempt, 
                process_func
            ) 
            for attempt in solution_attempts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions and convert to ParallelTaskResult
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ParallelTaskResult(
                    task_id=solution_attempts[i].id,
                    success=False,
                    result=None,
                    error=str(result)
                ))
            elif isinstance(result, ParallelTaskResult):
                final_results.append(result)
            else:
                # If the process_func doesn't return a ParallelTaskResult, wrap it
                final_results.append(ParallelTaskResult(
                    task_id=solution_attempts[i].id,
                    success=True,
                    result=result,
                    error=None
                ))
        
        return final_results
    
    async def _process_with_semaphore(
        self, 
        attempt: SolutionAttempt, 
        process_func: Callable
    ) -> ParallelTaskResult:
        """Process a single solution attempt with concurrency control."""
        async with self.semaphore:
            start_time = time.time()
            
            try:
                result = await process_func(attempt)
                execution_time = time.time() - start_time
                
                return ParallelTaskResult(
                    task_id=attempt.id,
                    success=True,
                    result=result,
                    execution_time=execution_time
                )
            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = f"Solution processing failed for attempt {attempt.id}: {str(e)}"
                
                return ParallelTaskResult(
                    task_id=attempt.id,
                    success=False,
                    result=None,
                    error=error_msg,
                    execution_time=execution_time
                )


class ResourceAwareParallelProcessor:
    """Parallel processor that considers resource constraints."""
    
    def __init__(self, cpu_limit: float = 0.8, memory_limit: float = 0.8):
        """
        Initialize resource-aware parallel processor.
        
        Args:
            cpu_limit: Maximum CPU utilization (0.0 to 1.0)
            memory_limit: Maximum memory utilization (0.0 to 1.0)
        """
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.logger = logging.getLogger(__name__)
        self.active_tasks = 0
        self.max_concurrent_tasks = multiprocessing.cpu_count()
        self.task_lock = threading.Lock()
    
    def execute_with_resource_control(
        self, 
        tasks: List[Tuple[Callable, Tuple, Dict]], 
        timeout: Optional[float] = None
    ) -> List[ParallelTaskResult]:
        """
        Execute tasks with resource utilization monitoring.
        
        Args:
            tasks: List of tasks to execute
            timeout: Optional timeout for execution
            
        Returns:
            List of execution results
        """
        import psutil
        
        # Monitor system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        # Adjust max workers based on current resource usage
        available_cpu = max(1, int(self.max_concurrent_tasks * (1 - cpu_percent / 100 / (1 - self.cpu_limit))))
        available_memory = max(1, int(self.max_concurrent_tasks * (1 - memory_percent / 100 / (1 - self.memory_limit))))
        
        effective_workers = min(available_cpu, available_memory, self.max_concurrent_tasks)
        
        self.logger.info(f"Resource-aware execution: {effective_workers} workers available "
                        f"(CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%)")
        
        # Use ThreadPoolExecutor with calculated worker count
        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = []
            
            for func, args, kwargs in tasks:
                future = executor.submit(
                    self._execute_with_resource_monitoring,
                    func,
                    *args,
                    **kwargs
                )
                futures.append(future)
            
            results = []
            for future in concurrent.futures.as_completed(futures, timeout=timeout):
                results.append(future.result())
            
            return results
    
    def _execute_with_resource_monitoring(
        self, 
        func: Callable, 
        *args, 
        **kwargs
    ) -> ParallelTaskResult:
        """Execute a task with resource monitoring."""
        import psutil
        
        with self.task_lock:
            self.active_tasks += 1
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        execution_time = time.time() - start_time
        memory_used = psutil.Process().memory_info().rss - start_memory
        
        # Log if resource usage is high
        if memory_used > 100 * 1024 * 1024:  # 100 MB
            self.logger.warning(f"High memory usage: {memory_used / (1024*1024):.2f} MB for task")
        
        with self.task_lock:
            self.active_tasks -= 1
        
        task_id = generate_id("resource_task")
        
        return ParallelTaskResult(
            task_id=task_id,
            success=success,
            result=result,
            error=error,
            execution_time=execution_time,
            worker_id=threading.get_ident()
        )


class ParallelWorkflowOrchestrator:
    """Orchestrates parallel execution of decomposition workflows."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize parallel workflow orchestrator.
        
        Args:
            max_workers: Maximum number of concurrent workers
        """
        self.parallel_processor = ParallelDecompositionProcessor(max_workers)
        self.async_processor = AsyncSolutionProcessor()
        self.resource_aware_processor = ResourceAwareParallelProcessor()
        self.logger = logging.getLogger(__name__)
    
    @with_error_handling(severity=ErrorSeverity.HIGH, fallback=lambda *args, **kwargs: {'success': False, 'results': []})
    def execute_parallel_decomposition_workflow(
        self, 
        plans: List[DecompositionPlan], 
        workflow_func: Callable
    ) -> Dict[str, Any]:
        """
        Execute multiple decomposition workflows in parallel.
        
        Args:
            plans: List of decomposition plans to process
            workflow_func: Function to execute for each plan
            
        Returns:
            Dictionary with execution results
        """
        self.logger.info(f"Executing {len(plans)} workflows in parallel")
        
        # Prepare tasks
        tasks = [(workflow_func, (plan,), {}) for plan in plans]
        
        # Execute with resource-aware parallel processing
        results = self.resource_aware_processor.execute_with_resource_control(tasks)
        
        successful_count = sum(1 for r in results if r.success)
        
        return {
            'success': True,
            'total_workflows': len(plans),
            'successful_workflows': successful_count,
            'results': results,
            'success_rate': successful_count / len(plans) if plans else 0
        }
    
    @with_error_handling(severity=ErrorSeverity.HIGH, fallback=lambda *args, **kwargs: {'success': False, 'results': []})
    def execute_parallel_solution_validation(
        self, 
        solution_attempts: List[SolutionAttempt], 
        validation_func: Callable
    ) -> Dict[str, Any]:
        """
        Execute solution validation in parallel.
        
        Args:
            solution_attempts: List of solution attempts to validate
            validation_func: Function to validate each solution
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info(f"Validating {len(solution_attempts)} solutions in parallel")
        
        # Create tasks for each solution attempt
        tasks = [(validation_func, (attempt,), {}) for attempt in solution_attempts]
        
        # Execute with resource control
        results = self.resource_aware_processor.execute_with_resource_control(tasks)
        
        successful_count = sum(1 for r in results if r.success)
        
        return {
            'success': True,
            'total_attempts': len(solution_attempts),
            'validated_attempts': successful_count,
            'results': results,
            'validation_rate': successful_count / len(solution_attempts) if solution_attempts else 0
        }
    
    def get_system_load_info(self) -> Dict[str, Any]:
        """Get system load information for intelligent parallelization."""
        try:
            import psutil
            
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'cpu_count': psutil.cpu_count(),
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None,
                'recommended_workers': max(1, int(psutil.cpu_count() * (1 - psutil.cpu_percent() / 100 / 4)))
            }
        except ImportError:
            # If psutil is not available, return basic info
            return {
                'cpu_count': os.cpu_count(),
                'recommended_workers': os.cpu_count() or 4
            }
    
    def shutdown(self):
        """Shutdown all processors."""
        self.parallel_processor.scheduler.shutdown()


# Global instance for easy access
parallel_processor = ParallelWorkflowOrchestrator()


def integrate_with_system():
    """
    Helper function to integrate parallel processing with existing system components.
    This would typically be called during system initialization.
    """
    from decomposition_engine import DecompositionEngine
    from sovereign_solution_orchestration import SolutionOrchestrator
    from sovereign_gauntlets import GauntletSystem
    
    # Example: Override methods to use parallel processing where appropriate
    def parallel_decompose_with_dependencies(self, plan: DecompositionPlan):
        """Modified decomposition method that uses parallel processing for independent tasks."""
        # Use the orchestrator to execute multiple plans when provided as a list.
        if isinstance(plan, list):
            return parallel_processor.execute_parallel_decomposition_workflow(
                plan,
                self.decompose
            )
        return self.decompose(plan)
    
    # Store original methods for potential restoration
    original_methods = {
        'decomposition_engine_decompose': getattr(DecompositionEngine, 'decompose', None),
        'solution_orchestrator_process': getattr(SolutionOrchestrator, 'integrate_solutions', None),
        'gauntlet_system_run': getattr(GauntletSystem, 'run_decomposition_gauntlets', None)
    }
    
    # The actual integration would involve decorating or replacing specific methods
    # that can benefit from parallel execution
    if not hasattr(DecompositionEngine, "decompose_with_parallel"):
        DecompositionEngine.decompose_with_parallel = parallel_decompose_with_dependencies
    
    return original_methods


def example_usage():
    """Example of how to use the parallel processing system."""
    
    # Example 1: Parallel processing of sub-problems with dependencies
    scheduler = TaskScheduler(max_workers=4)
    
    # Create mock sub-problems with dependencies
    sub_problems = [
        SubProblem(
            id="sp1", 
            parent_id="p1", 
            title="Sub-problem 1", 
            description="First sub-problem", 
            type="ANALYSIS",
            complexity_score=None
        ),
        SubProblem(
            id="sp2", 
            parent_id="p1", 
            title="Sub-problem 2", 
            description="Second sub-problem (depends on 1)", 
            type="ANALYSIS", 
            complexity_score=None
        ),
        SubProblem(
            id="sp3", 
            parent_id="p1", 
            title="Sub-problem 3", 
            description="Third sub-problem (depends on 1)", 
            type="ANALYSIS",
            complexity_score=None
        ),
    ]
    
    dependency_graph = {
        "sp2": ["sp1"],  # sp2 depends on sp1
        "sp3": ["sp1"],  # sp3 depends on sp1
        "sp1": []        # sp1 has no dependencies
    }
    
    def mock_process_function(sub_problem):
        """Mock function to simulate processing a sub-problem."""
        import time
        import random
        
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.5))
        
        return f"Processed {sub_problem.title} successfully"
    
    # Execute with dependencies
    results = scheduler.schedule_subproblems_with_dependencies(
        sub_problems, 
        dependency_graph, 
        mock_process_function
    )
    
    print(f"Execution completed with {len(results)} results")
    for result in results:
        print(f"Task {result.task_id}: {'SUCCESS' if result.success else 'FAILED'} "
              f"in {result.execution_time:.2f}s")
    
    # Example 2: Resource-aware parallel processing
    resource_processor = ResourceAwareParallelProcessor()
    
    def mock_task_function(param):
        """Mock task function."""
        import time
        time.sleep(0.2)  # Simulate work
        return f"Processed {param}"
    
    # Create multiple tasks
    tasks = [(mock_task_function, (f"param_{i}",), {}) for i in range(10)]
    
    resource_results = resource_processor.execute_with_resource_control(tasks)
    
    print(f"\nResource-aware execution completed with {len(resource_results)} results")
    
    # Example 3: Get system load info
    orchestrator = ParallelWorkflowOrchestrator()
    load_info = orchestrator.get_system_load_info()
    
    print(f"\nSystem load info: {load_info}")
    
    # Shutdown
    scheduler.shutdown()
    
    return results, resource_results


if __name__ == "__main__":
    example_usage()
