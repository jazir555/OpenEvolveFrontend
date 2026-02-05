"""
Gauntlet Pipeline with Checkpointing Integration

Integrates the checkpoint manager into the Gauntlet problem solving
pipeline for automatic checkpoint creation and resume capabilities.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import logging
import asyncio

from .checkpoint_manager import (
    CheckpointManager,
    PipelineState,
    create_checkpoint_manager,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of pipeline execution"""
    success: bool
    solution: Optional[Dict[str, Any]] = None
    checkpoints_created: List[str] = None
    resumed_from: Optional[str] = None
    execution_time: float = 0.0
    error: Optional[str] = None


class CheckpointedPipeline:
    """
    Gauntlet pipeline with automatic checkpointing.

    Creates checkpoints at key stages and can resume from last checkpoint
    if execution is interrupted.
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager = None,
        auto_checkpoint: bool = True,
        checkpoint_on_stages: List[str] = None
    ):
        self.checkpoint_manager = checkpoint_manager or create_checkpoint_manager()
        self.auto_checkpoint = auto_checkpoint
        self.checkpoint_on_stages = checkpoint_on_stages or [
            'decomposition_complete',
            'atomic_solve_complete',
            'reassembly_complete',
            'validation_complete',
        ]

    async def execute_with_checkpointing(
        self,
        problem: Dict[str, Any],
        solve_func: Callable,
        resume_from_checkpoint: str = None,
        context: Dict[str, Any] = None
    ) -> PipelineResult:
        """
        Execute pipeline with automatic checkpointing.

        Args:
            problem: Problem to solve
            solve_func: Async function that solves the problem
            resume_from_checkpoint: Optional checkpoint ID to resume from
            context: Execution context

        Returns:
            PipelineResult with solution and checkpointing info
        """
        from datetime import datetime
        start_time = datetime.utcnow()

        context = context or {}
        checkpoints_created = []

        try:
            # Check if we should resume from a checkpoint
            if resume_from_checkpoint:
                logger.info(f"Attempting to resume from checkpoint: {resume_from_checkpoint}")
                state = await self.checkpoint_manager.load_checkpoint(resume_from_checkpoint)

                if state:
                    logger.info(f"[OK] Resumed from checkpoint: {resume_from_checkpoint}")
                    # Continue from where we left off
                    result = await self._continue_from_checkpoint(state, solve_func)
                    return PipelineResult(
                        success=True,
                        solution=result,
                        checkpoints_created=checkpoints_created,
                        resumed_from=resume_from_checkpoint,
                        execution_time=(datetime.utcnow() - start_time).total_seconds()
                    )
                else:
                    logger.warning(f"Could not load checkpoint: {resume_from_checkpoint}")
                    logger.info("Falling back to fresh execution")

            # Execute pipeline with automatic checkpointing
            result = await self._execute_with_checkpointing(
                problem,
                solve_func,
                context,
                checkpoints_created
            )

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return PipelineResult(
                success=True,
                solution=result,
                checkpoints_created=checkpoints_created,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Pipeline execution failed: {e}")

            return PipelineResult(
                success=False,
                error=str(e),
                checkpoints_created=checkpoints_created,
                execution_time=execution_time
            )

    async def _execute_with_checkpointing(
        self,
        problem: Dict[str, Any],
        solve_func: Callable,
        context: Dict[str, Any],
        checkpoints_created: List[str]
    ) -> Dict[str, Any]:
        """Execute pipeline with automatic checkpoint creation"""

        # Stage 1: Before decomposition
        if self.auto_checkpoint and 'decomposition_complete' in self.checkpoint_on_stages:
            checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                problem=problem,
                context=context,
                level=0,
                stage='before_decomposition'
            )
            if checkpoint_id:
                checkpoints_created.append(checkpoint_id)

        # Execute the solve function (which handles the full pipeline)
        result = await solve_func(problem, context)

        # Stage 2: After solving (if successful)
        if self.auto_checkpoint and result and 'atomic_solve_complete' in self.checkpoint_on_stages:
            checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                problem=problem,
                context=context,
                solutions=result.get('solutions', {}),
                decomposition_tree=result.get('decomposition_tree'),
                execution_status=result.get('execution_status', {}),
                metrics=result.get('metrics', {}),
                level=0,
                stage='after_solving'
            )
            if checkpoint_id:
                checkpoints_created.append(checkpoint_id)

        return result

    async def _continue_from_checkpoint(
        self,
        state: PipelineState,
        solve_func: Callable
    ) -> Dict[str, Any]:
        """Continue execution from a checkpoint"""
        # Restore context and continue solving
        # This is a simplified version - in practice, you'd need more sophisticated
        # logic to determine exactly where to continue from

        # For now, just re-run the solve function with the restored context
        result = await solve_func(state.problem, state.context)

        return result

    async def list_available_checkpoints(self, problem_id: str = None) -> List[Dict[str, Any]]:
        """List available checkpoints for a problem"""
        checkpoints = await self.checkpoint_manager.list_checkpoints(problem_id)

        return [
            {
                'checkpoint_id': c.checkpoint_id,
                'problem_id': c.problem_id,
                'timestamp': c.timestamp.isoformat(),
                'level': c.level,
                'stage': c.stage,
                'state_size': c.state_size,
                'compressed': c.compressed,
                'parent_checkpoint_id': c.parent_checkpoint_id,
            }
            for c in checkpoints
        ]

    async def resume_from_latest_checkpoint(
        self,
        problem_id: str,
        solve_func: Callable
    ) -> Optional[PipelineResult]:
        """
        Resume execution from the latest checkpoint for a problem.

        Args:
            problem_id: Problem ID to find checkpoints for
            solve_func: Solve function to continue with

        Returns:
            PipelineResult if checkpoint found and loaded, None otherwise
        """
        checkpoints = await self.checkpoint_manager.list_checkpoints(problem_id)

        if not checkpoints:
            logger.info(f"No checkpoints found for problem: {problem_id}")
            return None

        # Sort by timestamp, most recent first
        checkpoints.sort(key=lambda c: c.timestamp, reverse=True)

        # Use the most recent checkpoint
        latest_checkpoint = checkpoints[0]

        logger.info(f"Resuming from latest checkpoint: {latest_checkpoint.checkpoint_id}")

        return await self.execute_with_checkpointing(
            problem={'id': problem_id},
            solve_func=solve_func,
            resume_from_checkpoint=latest_checkpoint.checkpoint_id
        )

    async def cleanup_old_checkpoints(
        self,
        problem_id: str,
        keep_last_n: int = 5
    ) -> int:
        """Clean up old checkpoints for a problem"""
        return await self.checkpoint_manager.cleanup_checkpoints(problem_id, keep_last_n)


async def demo_checkpointed_pipeline():
    """Demonstration of checkpointed pipeline usage"""

    # Create a simple solve function
    async def demo_solve_func(problem: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Solving problem: {problem.get('id', 'unknown')}")

        # Simulate some work
        await asyncio.sleep(1)

        return {
            'solution': f"Solution for {problem.get('statement', 'problem')}",
            'metrics': {'execution_time': 1.0},
        }

    # Create checkpointed pipeline
    pipeline = CheckpointedPipeline(
        auto_checkpoint=True,
        checkpoint_on_stages=['before_decomposition', 'after_solving']
    )

    # Execute with checkpointing
    problem = {
        'id': 'test_problem_1',
        'statement': 'Solve this test problem',
        'requirements': ['quality', 'speed'],
    }

    result = await pipeline.execute_with_checkpointing(
        problem=problem,
        solve_func=demo_solve_func
    )

    print(f"Execution successful: {result.success}")
    print(f"Checkpoints created: {result.checkpoints_created}")
    print(f"Execution time: {result.execution_time:.2f}s")

    # List checkpoints
    checkpoints = await pipeline.list_available_checkpoints('test_problem_1')
    print(f"Available checkpoints: {len(checkpoints)}")

    # Clean up
    deleted = await pipeline.cleanup_old_checkpoints('test_problem_1', keep_last_n=2)
    print(f"Deleted {deleted} old checkpoints")


if __name__ == '__main__':
    import asyncio
    asyncio.run(demo_checkpointed_pipeline())
