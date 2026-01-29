"""
Enhanced Gauntlet Solver with Parallel Execution Integration

Provides the main solveProblem function with automatic detection and
execution of parallelizable subproblems.

Key Features:
- Automatic parallelizable subproblem detection
- Conditional parallel vs sequential execution
- Fallback to sequential on error
- Integration with both asyncio and worker pool executors
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

from .parallel_executor import (
    ParallelProblemExecutor,
    ProblemDependencyAnalyzer,
    get_parallel_executor
)
from .worker_pool_executor import (
    WorkerPoolExecutor,
    create_worker_pool_executor
)
from .solution_cache import AtomicSolutionCache, create_solution_cache
from .checkpoint_manager import CheckpointManager, create_checkpoint_manager
from .problem_visualization import visualize_problem, OutputFormat

logger = logging.getLogger(__name__)


class GauntletSolver:
    """
    Enhanced problem solver with parallel execution support.
    """

    def __init__(
        self,
        cache: Optional[AtomicSolutionCache] = None,
        parallel_executor: Optional[ParallelProblemExecutor] = None,
        worker_pool: Optional[WorkerPoolExecutor] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        enable_parallel: bool = True,
        parallel_threshold: int = 3,
        use_worker_pool: bool = False,
        enable_checkpointing: bool = True,
        enable_visualization: bool = False,
        visualization_format: str = 'ascii'
    ):
        self.cache = cache or create_solution_cache()
        self.parallel_executor = parallel_executor or get_parallel_executor()
        self.worker_pool = worker_pool
        self.checkpoint_manager = checkpoint_manager or create_checkpoint_manager()
        self.enable_parallel = enable_parallel
        self.parallel_threshold = parallel_threshold
        self.use_worker_pool = use_worker_pool
        self.enable_checkpointing = enable_checkpointing
        self.enable_visualization = enable_visualization
        self.visualization_format = visualization_format

        # Dependency analyzer for detecting parallelizable problems
        self.dependency_analyzer = ProblemDependencyAnalyzer()

    async def solve_problem(
        self,
        problem: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        force_sequential: bool = False,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Solve a problem with automatic parallel execution detection and checkpointing.

        Args:
            problem: Problem to solve
            context: Execution context
            force_sequential: Force sequential execution
            resume_from_checkpoint: Specific checkpoint ID to resume from

        Returns:
            Solution result
        """
        context = context or {}
        problem_id = problem.get('id', 'unknown')

        # Track decomposition level for checkpointing
        level = context.get('level', 0)
        stage = context.get('stage', 'solving')

        logger.info(f"Solving problem: {problem_id} (level {level})")

        # Visualize initial problem structure
        if self.enable_visualization:
            try:
                viz = visualize_problem(problem, self.visualization_format)
                logger.info(f"\\nInitial Problem Structure:\\n{viz}\\n")
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")

        try:
            # Check if we should resume from checkpoint
            if resume_from_checkpoint and self.enable_checkpointing:
                logger.info(f"Attempting to resume from checkpoint: {resume_from_checkpoint}")
                loaded_state = await self.checkpoint_manager.load_checkpoint(resume_from_checkpoint)
                if loaded_state:
                    logger.info(f"Successfully resumed from checkpoint")
                    context.update(loaded_state.context)
                    # Continue with solving using restored context
                    return await self._solve_with_checkpoint_tracking(
                        problem, context, force_sequential, level, stage, loaded_state
                    )
                else:
                    logger.warning(f"Failed to load checkpoint: {resume_from_checkpoint}")

            # Check for existing checkpoints (auto-resume)
            if self.enable_checkpointing and not resume_from_checkpoint:
                checkpoints = await self.checkpoint_manager.list_checkpoints(problem_id)
                if checkpoints:
                    latest_checkpoint = checkpoints[0]  # Most recent first
                    logger.info(f"Found existing checkpoint: {latest_checkpoint.checkpoint_id}")
                    logger.info("To resume explicitly, pass resume_from_checkpoint parameter")

            # Create initial checkpoint before solving
            if self.enable_checkpointing:
                checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                    problem=problem,
                    context={**context, 'level': level, 'stage': 'starting'},
                    solutions={},
                    level=level,
                    stage='starting'
                )
                if checkpoint_id:
                    logger.info(f"Created initial checkpoint: {checkpoint_id}")
                    context['current_checkpoint_id'] = checkpoint_id

            # Define solve function with cache
            async def solve_with_cache(prob):
                # Detect if problem has parallelizable subproblems
                has_parallel_subproblems = self._detect_parallelizable_subproblems(prob)

                # Decide execution strategy
                if (not force_sequential and
                    self.enable_parallel and
                    has_parallel_subproblems and
                    self._should_use_parallel(prob)):

                    logger.info(f"Using parallel execution for {problem_id}")
                    return await self._solve_parallel(prob, context)
                else:
                    logger.info(f"Using sequential execution for {problem_id}")
                    return await self._solve_sequential(prob, context)

            # Use cache with solve function
            solution = await self.cache.solve(problem, solve_with_cache)
            logger.info(f"Problem {problem_id} solved")

            # Visualize solution structure
            if self.enable_visualization:
                try:
                    viz = visualize_problem(problem, self.visualization_format)
                    logger.info(f"\\nFinal Solution Structure:\\n{viz}\\n")
                except Exception as e:
                    logger.warning(f"Visualization failed: {e}")

            # Create completion checkpoint
            if self.enable_checkpointing:
                checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                    problem=problem,
                    context={**context, 'level': level, 'stage': 'complete'},
                    solutions=solution.get('solutions', {}),
                    level=level,
                    stage='complete'
                )
                if checkpoint_id:
                    logger.info(f"Created completion checkpoint: {checkpoint_id}")

            return solution

        except Exception as e:
            logger.error(f"Error solving problem {problem_id}: {e}")

            # Create error checkpoint for recovery
            if self.enable_checkpointing:
                checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                    problem=problem,
                    context={**context, 'level': level, 'stage': 'error', 'error': str(e)},
                    solutions={},
                    level=level,
                    stage='error'
                )
                if checkpoint_id:
                    logger.info(f"Created error checkpoint: {checkpoint_id}")

            # Fallback to sequential on error
            if not force_sequential:
                logger.info("Falling back to sequential execution")
                return await self._solve_sequential(problem, context)

            raise

    async def _solve_with_checkpoint_tracking(
        self,
        problem: Dict[str, Any],
        context: Dict[str, Any],
        force_sequential: bool,
        level: int,
        stage: str,
        loaded_state
    ) -> Dict[str, Any]:
        """Continue solving from loaded checkpoint state"""
        # Restore solutions from checkpoint
        solutions = loaded_state.solutions

        # Update context with restored state
        context['resumed_from_checkpoint'] = loaded_state.metadata.get('checkpoint_id')

        # Continue solving (either with restored solutions or continue from where we left off)
        has_parallel_subproblems = self._detect_parallelizable_subproblems(problem)

        if (not force_sequential and
            self.enable_parallel and
            has_parallel_subproblems and
            self._should_use_parallel(problem)):

            logger.info(f"Using parallel execution after resume")
            result = await self._solve_parallel(problem, context)
        else:
            logger.info(f"Using sequential execution after resume")
            result = await self._solve_sequential(problem, context)

        return result

    def _detect_parallelizable_subproblems(self, problem: Dict[str, Any]) -> bool:
        """Detect if problem has parallelizable subproblems"""
        subproblems = problem.get('subproblems', [])

        if not subproblems or len(subproblems) < self.parallel_threshold:
            return False

        # Check for dependencies
        independent = self.dependency_analyzer.find_independent_problems(subproblems)

        return len(independent) >= self.parallel_threshold

    def _should_use_parallel(self, problem: Dict[str, Any]) -> bool:
        """Determine if parallel execution should be used"""
        subproblems = problem.get('subproblems', [])

        # Must have enough subproblems
        if len(subproblems) < self.parallel_threshold:
            return False

        # Check if most are independent
        independent = self.dependency_analyzer.find_independent_problems(subproblems)
        independence_ratio = len(independent) / len(subproblems)

        return independence_ratio >= 0.5  # At least 50% independent

    async def _solve_parallel(
        self,
        problem: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Solve problem using parallel execution"""
        subproblems = problem.get('subproblems', [])

        if self.use_worker_pool and self.worker_pool:
            # Use worker pool for CPU-intensive tasks
            summary = await self.worker_pool.execute_in_parallel(
                problems=subproblems,
                executor_func=self._solve_single,
                context=context
            )

            results = [r.result for r in summary.results if r.success]

        else:
            # Use asyncio parallel executor
            summary = await self.parallel_executor.execute_in_parallel(
                problems=subproblems,
                executor_func=self._solve_single,
                context=context
            )

            results = summary.results

        # Combine results
        return self._combine_results(problem, results)

    async def _solve_sequential(
        self,
        problem: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Solve problem sequentially"""
        subproblems = problem.get('subproblems', [])

        if not subproblems:
            # Atomic problem - solve directly
            return await self._solve_single(problem)

        # Solve subproblems sequentially
        results = []
        for subproblem in subproblems:
            result = await self._solve_single(subproblem)
            results.append(result)

        # Combine results
        return self._combine_results(problem, results)

    async def _solve_single(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Solve a single (atomic) problem"""
        # This is where the actual problem solving happens
        # For now, return a mock solution
        return {
            'problem_id': problem.get('id', 'unknown'),
            'success': True,
            'score': 0.85,
            'solution': f"Solution for {problem.get('id')}",
            'timestamp': datetime.utcnow().isoformat(),
        }

    def _combine_results(
        self,
        problem: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine results from subproblems"""
        return {
            'problem_id': problem.get('id', 'unknown'),
            'success': all(r.get('success', False) for r in results),
            'score': sum(r.get('score', 0) for r in results) / len(results) if results else 0,
            'solutions': results,
            'num_solutions': len(results),
            'timestamp': datetime.utcnow().isoformat(),
        }


async def solveProblem(
    problem: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    enable_parallel: bool = True,
    use_worker_pool: bool = False,
    enable_checkpointing: bool = True,
    enable_visualization: bool = False,
    visualization_format: str = 'ascii',
    resume_from_checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main solveProblem function with automatic parallel execution, checkpointing, and visualization.

    Args:
        problem: Problem to solve
        context: Execution context
        enable_parallel: Enable parallel execution
        use_worker_pool: Use worker pool instead of asyncio
        enable_checkpointing: Enable checkpointing for reliability
        enable_visualization: Enable visualization of problem hierarchy
        visualization_format: Format for visualization ('ascii', 'html', 'dot')
        resume_from_checkpoint: Specific checkpoint ID to resume from

    Returns:
        Solution result

    Example:
        >>> problem = {
        ...     'id': 'problem_123',
        ...     'statement': 'Solve this',
        ...     'subproblems': [
        ...         {'id': 'sub_1', 'statement': 'Sub 1'},
        ...         {'id': 'sub_2', 'statement': 'Sub 2'},
        ...         {'id': 'sub_3', 'statement': 'Sub 3'},
        ...     ]
        ... }
        >>> solution = await solveProblem(problem, enable_visualization=True)
        >>> print(solution['success'])
        True
    """
    solver = GauntletSolver(
        enable_parallel=enable_parallel,
        use_worker_pool=use_worker_pool,
        enable_checkpointing=enable_checkpointing,
        enable_visualization=enable_visualization,
        visualization_format=visualization_format
    )

    return await solver.solve_problem(problem, context, resume_from_checkpoint=resume_from_checkpoint)


# Example usage
async def demo_enhanced_solver():
    """Demonstration of enhanced solver"""

    print("\n" + "=" * 60)
    print("Enhanced Gauntlet Solver Demo")
    print("=" * 60)

    # Create problem with subproblems
    problem = {
        'id': 'complex_problem',
        'statement': 'Solve this complex problem',
        'subproblems': [
            {'id': 'sub_1', 'statement': 'Subproblem 1'},
            {'id': 'sub_2', 'statement': 'Subproblem 2'},
            {'id': 'sub_3', 'statement': 'Subproblem 3'},
            {'id': 'sub_4', 'statement': 'Subproblem 4'},
            {'id': 'sub_5', 'statement': 'Subproblem 5'},
        ]
    }

    print("\n1. Solving with parallel execution...")
    solution_parallel = await solveProblem(problem, enable_parallel=True)
    print(f"   Success: {solution_parallel['success']}")
    print(f"   Score: {solution_parallel['score']:.2f}")
    print(f"   Solutions: {solution_parallel['num_solutions']}")

    print("\n2. Solving with sequential execution...")
    solution_sequential = await solveProblem(problem, enable_parallel=False)
    print(f"   Success: {solution_sequential['success']}")
    print(f"   Score: {solution_sequential['score']:.2f}")

    print("\n3. Solving atomic problem...")
    atomic_problem = {'id': 'atomic_1', 'statement': 'Simple problem'}
    atomic_solution = await solveProblem(atomic_problem)
    print(f"   Success: {atomic_solution['success']}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    import asyncio
    asyncio.run(demo_enhanced_solver())
