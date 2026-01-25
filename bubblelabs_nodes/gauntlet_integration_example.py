"""
Complete Gauntlet Integration Example

Demonstrates how to integrate all Phase 1 components:
- Parallel atomic problem solving
- Solution caching
- Problem hierarchy visualization
- Checkpointing and resume

This example shows a complete workflow from problem input to solution.
"""

import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime

from .parallel_executor import (
    ParallelProblemExecutor,
    ProblemDependencyAnalyzer,
    ExecutionResult,
)
from .solution_cache import (
    AtomicSolutionCache,
    create_solution_cache,
)
from .visualization import (
    visualize_problem,
    ProblemTreeBuilder,
)
from .checkpoint_manager import (
    CheckpointManager,
    create_checkpoint_manager,
)
from .gauntlet_pipeline_checkpointed import (
    CheckpointedPipeline,
    PipelineResult,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GauntletSystem:
    """
    Complete Gauntlet system with all Phase 1 enhancements.

    Integrates parallel execution, caching, visualization, and checkpointing
    into a cohesive problem-solving system.
    """

    def __init__(
        self,
        parallel_enabled: bool = True,
        cache_enabled: bool = True,
        checkpointing_enabled: bool = True,
        visualization_enabled: bool = True,
    ):
        # Initialize parallel executor
        self.parallel_enabled = parallel_enabled
        self.parallel_executor = ParallelProblemExecutor(
            config={'max_concurrency': 4, 'timeout': 300}
        ) if parallel_enabled else None

        # Initialize solution cache
        self.cache_enabled = cache_enabled
        self.solution_cache = create_solution_cache(
            config={'max_size': 1000, 'ttl': 3600}
        ) if cache_enabled else None

        # Initialize checkpoint manager
        self.checkpointing_enabled = checkpointing_enabled
        self.checkpoint_manager = create_checkpoint_manager(
            storage_type='file',
            storage_path='./gauntlet_checkpoints',
            compression=False,
        ) if checkpointing_enabled else None

        # Initialize checkpointed pipeline
        self.pipeline = CheckpointedPipeline(
            checkpoint_manager=self.checkpoint_manager,
            auto_checkpoint=checkpointing_enabled,
        ) if checkpointing_enabled else None

        # Initialize visualization
        self.visualization_enabled = visualization_enabled
        self.tree_builder = ProblemTreeBuilder() if visualization_enabled else None

    async def solve_problem(
        self,
        problem: Dict[str, Any],
        use_cache: bool = True,
        use_parallel: bool = True,
        checkpoint_id: str = None,
    ) -> Dict[str, Any]:
        """
        Solve a problem with all enhancements enabled.

        Args:
            problem: Problem to solve
            use_cache: Whether to use solution cache
            use_parallel: Whether to use parallel execution for subproblems
            checkpoint_id: Checkpoint ID to resume from (if any)

        Returns:
            Solution result with metadata
        """
        start_time = datetime.utcnow()
        logger.info(f"{'='*60}")
        logger.info(f"Starting Gauntlet problem solving")
        logger.info(f"Problem: {problem.get('statement', 'Unknown')}")
        logger.info(f"{'='*60}")

        # 1. Check cache first
        if self.cache_enabled and use_cache:
            cached = await self._check_cache(problem)
            if cached:
                logger.info("✅ Cache HIT - returning cached solution")
                return {
                    'solution': cached,
                    'source': 'cache',
                    'execution_time': (datetime.utcnow() - start_time).total_seconds(),
                }

        # 2. Resume from checkpoint if specified
        if checkpoint_id and self.checkpointing_enabled:
            logger.info(f"Resuming from checkpoint: {checkpoint_id}")
            result = await self.pipeline.execute_with_checkpointing(
                problem=problem,
                solve_func=self._solve_internal,
                resume_from_checkpoint=checkpoint_id,
            )
            return self._format_result(result, start_time)

        # 3. Create initial checkpoint
        if self.checkpointing_enabled:
            await self.checkpoint_manager.create_checkpoint(
                problem=problem,
                context={'stage': 'initial'},
                level=0,
                stage='initial'
            )

        # 4. Visualize initial problem hierarchy
        if self.visualization_enabled and self.tree_builder:
            logger.info("Generating problem hierarchy visualization...")
            tree = self.tree_builder.build_tree(problem)
            ascii_tree = visualize_problem(problem, format='ascii')
            logger.info(f"\nProblem Hierarchy:\n{ascii_tree}\n")

        # 5. Solve the problem (with or without parallel execution)
        result = await self._solve_internal(
            problem,
            use_cache=use_cache,
            use_parallel=use_parallel
        )

        # 6. Cache the solution
        if self.cache_enabled and use_cache and result.get('solution'):
            await self._cache_solution(problem, result['solution'])

        # 7. Create final checkpoint
        if self.checkpointing_enabled:
            await self.checkpoint_manager.create_checkpoint(
                problem=problem,
                context={'stage': 'complete'},
                solutions={'solution': result.get('solution')},
                level=0,
                stage='complete'
            )

        # 8. Format and return result
        return self._format_result(result, start_time)

    async def _solve_internal(
        self,
        problem: Dict[str, Any],
        use_cache: bool = True,
        use_parallel: bool = True,
    ) -> Dict[str, Any]:
        """Internal solve method that handles the actual solving logic"""

        # Extract subproblems if present
        subproblems = problem.get('subproblems', [])

        if subproblems and self.parallel_enabled and use_parallel:
            # Use parallel execution
            logger.info(f"Executing {len(subproblems)} subproblems in parallel...")

            async def solve_subproblem(sp):
                # Use cache for each subproblem if enabled
                if self.cache_enabled and use_cache:
                    return await self.solution_cache.solve(
                        sp,
                        self._solve_atomic
                    )
                return await self._solve_atomic(sp)

            summary = await self.parallel_executor.execute_in_parallel(
                problems=subproblems,
                executor_func=solve_subproblem,
                context={},
            )

            logger.info(
                f"Parallel execution complete: "
                f"{summary.successful}/{summary.total_problems} successful, "
                f"{summary.parallel_speedup:.2f}x speedup"
            )

            # Aggregate results
            solutions = {}
            for result in summary.results:
                if result.success and result.result:
                    solutions[result.problem_id] = result.result

            return {
                'solution': self._aggregate_solutions(solutions),
                'subproblem_results': summary,
            }
        else:
            # Sequential execution
            if subproblems:
                logger.info(f"Executing {len(subproblems)} subproblems sequentially...")
                solutions = {}
                for subproblem in subproblems:
                    if self.cache_enabled and use_cache:
                        result = await self.solution_cache.solve(
                            subproblem,
                            self._solve_atomic
                        )
                    else:
                        result = await self._solve_atomic(subproblem)
                    solutions[subproblem.get('id', 'unknown')] = result

                return {
                    'solution': self._aggregate_solutions(solutions),
                }
            else:
                # Atomic problem - solve directly
                solution = await self._solve_atomic(problem)
                return {'solution': solution}

    async def _solve_atomic(self, problem: Dict[str, Any]) -> Any:
        """Solve an atomic problem (lowest level)"""
        # This is where the actual solving happens
        # For this example, we'll just return a mock solution

        logger.info(f"Solving atomic problem: {problem.get('id', 'unknown')}")

        # Simulate some work
        await asyncio.sleep(0.1)

        return f"Solution for: {problem.get('statement', 'problem')}"

    def _aggregate_solutions(self, solutions: Dict[str, Any]) -> Any:
        """Aggregate solutions from subproblems"""
        if not solutions:
            return None

        # For this example, just return the combined solutions
        return {
            'type': 'aggregated',
            'count': len(solutions),
            'solutions': solutions,
        }

    async def _check_cache(self, problem: Dict[str, Any]) -> Any:
        """Check if solution is cached"""
        if not self.solution_cache:
            return None

        cached = await self.solution_cache.get(problem)
        return cached

    async def _cache_solution(self, problem: Dict[str, Any], solution: Any):
        """Cache a solution"""
        if not self.solution_cache:
            return

        # The cache handles storage internally
        # We just need to trigger a solve to cache it
        await self.solution_cache.solve(
            problem,
            lambda p: solution
        )

    def _format_result(self, result: Any, start_time: datetime) -> Dict[str, Any]:
        """Format the result for return"""
        execution_time = (datetime.utcnow() - start_time).total_seconds()

        if isinstance(result, PipelineResult):
            return {
                'success': result.success,
                'solution': result.solution,
                'execution_time': result.execution_time,
                'checkpoints_created': result.checkpoints_created,
                'resumed_from': result.resumed_from,
                'error': result.error,
            }
        elif isinstance(result, dict):
            return {
                'success': True,
                'solution': result.get('solution'),
                'execution_time': execution_time,
                **{k: v for k, v in result.items() if k != 'solution'},
            }
        else:
            return {
                'success': True,
                'solution': result,
                'execution_time': execution_time,
            }

    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        if not self.solution_cache:
            return {'enabled': False}

        return self.solution_cache.get_statistics()

    async def list_checkpoints(self, problem_id: str = None) -> List[Dict[str, Any]]:
        """List available checkpoints"""
        if not self.pipeline:
            return []

        return await self.pipeline.list_available_checkpoints(problem_id)

    async def cleanup_checkpoints(self, problem_id: str, keep_last_n: int = 5) -> int:
        """Clean up old checkpoints"""
        if not self.pipeline:
            return 0

        return await self.pipeline.cleanup_old_checkpoints(problem_id, keep_last_n)


async def demo_complete_gauntlet_system():
    """Demonstration of complete Gauntlet system"""

    # Create Gauntlet system with all features enabled
    gauntlet = GauntletSystem(
        parallel_enabled=True,
        cache_enabled=True,
        checkpointing_enabled=True,
        visualization_enabled=True,
    )

    # Example 1: Simple atomic problem
    print("\n" + "="*60)
    print("Example 1: Simple atomic problem")
    print("="*60)

    problem1 = {
        'id': 'problem_1',
        'statement': 'Build a REST API endpoint',
        'requirements': ['fast', 'secure'],
    }

    result1 = await gauntlet.solve_problem(problem1)
    print(f"Result: {result1['solution']}")
    print(f"Execution time: {result1['execution_time']:.2f}s")

    # Example 2: Problem with subproblems (parallel execution)
    print("\n" + "="*60)
    print("Example 2: Problem with subproblems (parallel)")
    print("="*60)

    problem2 = {
        'id': 'problem_2',
        'statement': 'Build an e-commerce platform',
        'requirements': ['scalable', 'secure'],
        'subproblems': [
            {
                'id': 'subproblem_2_1',
                'statement': 'Design database schema',
                'requirements': ['normalized'],
            },
            {
                'id': 'subproblem_2_2',
                'statement': 'Implement user authentication',
                'requirements': ['oauth2'],
            },
            {
                'id': 'subproblem_2_3',
                'statement': 'Create product catalog',
                'requirements': ['searchable'],
            },
            {
                'id': 'subproblem_2_4',
                'statement': 'Implement shopping cart',
                'requirements': ['persistent'],
            },
        ],
    }

    result2 = await gauntlet.solve_problem(problem2, use_parallel=True)
    print(f"Result: {result2['solution']}")

    if 'subproblem_results' in result2['solution']:
        sp_results = result2['solution']['subproblem_results']
        print(f"\nParallel execution stats:")
        print(f"  - Total problems: {sp_results.total_problems}")
        print(f"  - Successful: {sp_results.successful}")
        print(f"  - Speedup: {sp_results.parallel_speedup:.2f}x")
        print(f"  - Total time: {sp_results.total_time:.2f}s")

    # Example 3: Cache hit (same problem again)
    print("\n" + "="*60)
    print("Example 3: Cache hit (same problem again)")
    print("="*60)

    result3 = await gauntlet.solve_problem(problem1)
    print(f"Result: {result3['solution']}")
    print(f"Source: {result3.get('source', 'computed')}")

    # Check cache statistics
    cache_stats = await gauntlet.get_cache_statistics()
    print(f"\nCache statistics:")
    print(f"  - Enabled: {cache_stats.get('enabled', False)}")
    print(f"  - Hit rate: {cache_stats.get('hit_rate', 0):.1%}")
    print(f"  - Size: {cache_stats.get('size', 0)}")

    # List checkpoints
    checkpoints = await gauntlet.list_checkpoints()
    print(f"\nCheckpoints created: {len(checkpoints)}")
    for cp in checkpoints[:3]:  # Show first 3
        print(f"  - {cp['checkpoint_id']} ({cp['stage']})")

    # Cleanup
    print("\n" + "="*60)
    print("Cleaning up old checkpoints...")
    deleted = await gauntlet.cleanup_checkpoints('problem_2', keep_last_n=2)
    print(f"Deleted {deleted} old checkpoints")

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == '__main__':
    asyncio.run(demo_complete_gauntlet_system())
