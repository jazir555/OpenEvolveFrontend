"""ICR Engine - Iterative refinement orchestration.

Coordinates generation, critique, refinement, and judgment in a loop
until quality criteria are met or stopping conditions are reached.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timezone

from integrations.icr.generator import Generator, GenerationResult, GenerationStrategy
from integrations.icr.critic import Critic, CritiqueResult, CritiqueCriteria
from integrations.icr.refiner import Refiner, RefinementStrategy, RefinementTracker
from integrations.icr.judge import Judge, EvaluationResult, Criteria, ComparisonResult

logger = logging.getLogger(__name__)


@dataclass
class IterationResult:
    """Result of a single iteration.
    
    Attributes:
        iteration: Iteration number (1-indexed)
        output: Output at this iteration
        critique: Critique of the output
        evaluation: Quality evaluation
        improvement: Score change from previous
        converged: Whether refinement has converged
        should_continue: Whether to continue iterating
    """
    iteration: int
    output: str
    critique: CritiqueResult
    evaluation: EvaluationResult
    improvement: float
    converged: bool
    should_continue: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "output": self.output,
            "critique": self.critique.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "improvement": self.improvement,
            "converged": self.converged,
            "should_continue": self.should_continue,
            "timestamp": self.timestamp,
        }


@dataclass
class RefinementResult:
    """Final result of iterative refinement.
    
    Attributes:
        final_output: The best output achieved
        iterations: Total number of iterations performed
        improvement_history: Quality scores over iterations
        final_score: Final quality score
        critique_history: Critiques from each iteration
        metadata: Additional result metadata
        convergence_reached: Whether convergence was achieved
        stopped_reason: Why iteration stopped
    """
    final_output: str
    iterations: int
    improvement_history: List[float]
    final_score: float
    critique_history: List[CritiqueResult]
    metadata: Dict[str, Any] = field(default_factory=dict)
    convergence_reached: bool = False
    stopped_reason: str = ""
    iteration_results: List[IterationResult] = field(default_factory=list)
    
    @property
    def total_improvement(self) -> float:
        """Total improvement from first to last iteration."""
        if len(self.improvement_history) >= 2:
            return self.improvement_history[-1] - self.improvement_history[0]
        return 0.0
    
    @property
    def average_improvement_per_iteration(self) -> float:
        """Average improvement per iteration."""
        if self.iterations > 0:
            return self.total_improvement / self.iterations
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_output": self.final_output,
            "iterations": self.iterations,
            "improvement_history": self.improvement_history,
            "final_score": self.final_score,
            "critique_history": [c.to_dict() for c in self.critique_history],
            "metadata": self.metadata,
            "convergence_reached": self.convergence_reached,
            "stopped_reason": self.stopped_reason,
            "total_improvement": self.total_improvement,
            "average_improvement_per_iteration": self.average_improvement_per_iteration,
        }


class ICREngine:
    """ICR Engine - Main orchestrator for iterative refinement.
    
    The ICR Engine coordinates the full refinement loop:
    Generate -> Critique -> Refine -> Judge -> Iterate
    
    It manages convergence detection, early stopping, and tracks
    the history of improvements.
    
    Example:
        >>> engine = ICREngine()
        >>> result = engine.refine(
        ...     prompt="Write a Python docstring",
        ...     max_iterations=5,
        ...     threshold=0.9
        ... )
        >>> print(f"Final score: {result.final_score}")
        >>> print(f"Iterations: {result.iterations}")
    """
    
    def __init__(
        self,
        generator: Optional[Generator] = None,
        critic: Optional[Critic] = None,
        refiner: Optional[Refiner] = None,
        judge: Optional[Judge] = None,
        max_iterations: int = 5,
        quality_threshold: float = 0.9,
        early_stopping: bool = True,
        patience: int = 2,
        min_improvement: float = 0.02,
    ):
        """Initialize the ICR Engine.
        
        Args:
            generator: Content generator (creates default if None)
            critic: Content critic (creates default if None)
            refiner: Content refiner (creates default if None)
            judge: Quality judge (creates default if None)
            max_iterations: Maximum refinement iterations
            quality_threshold: Target quality threshold
            early_stopping: Whether to enable early stopping
            patience: Iterations without improvement before stopping
            min_improvement: Minimum improvement to continue
        """
        self.generator = generator or Generator()
        self.critic = critic or Critic()
        self.refiner = refiner or Refiner()
        self.judge = judge or Judge(default_threshold=quality_threshold)
        
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_improvement = min_improvement
        
        self.tracker = RefinementTracker()
        self._run_count = 0
        
        logger.info(
            f"Initialized ICREngine: max_iterations={max_iterations}, "
            f"threshold={quality_threshold}, early_stopping={early_stopping}"
        )
    
    def refine(
        self,
        prompt: str,
        max_iterations: Optional[int] = None,
        threshold: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        initial_output: Optional[str] = None,
    ) -> RefinementResult:
        """Run iterative refinement on a prompt.
        
        Args:
            prompt: The generation prompt
            max_iterations: Override max iterations
            threshold: Override quality threshold
            context: Additional context for refinement
            initial_output: Skip generation and use this as starting point
            
        Returns:
            RefinementResult with final output and history
        """
        max_iterations = max_iterations or self.max_iterations
        threshold = threshold or self.quality_threshold
        context = context or {}
        
        self._run_count += 1
        start_time = time.time()
        
        logger.info(
            f"Starting ICR refinement run #{self._run_count}",
            extra={
                "correlation_id": context.get("correlation_id"),
                "max_iterations": max_iterations,
                "threshold": threshold,
            },
        )
        
        self.tracker.start_tracking()
        
        # Generate initial output if not provided
        if initial_output is None:
            gen_result = self.generator.generate(prompt, context)
            current_output = gen_result.content
        else:
            current_output = initial_output
            gen_result = None
        
        # Track history
        iteration_results: List[IterationResult] = []
        critique_history: List[CritiqueResult] = []
        improvement_history: List[float] = []
        
        # Initial evaluation
        eval_result = self.judge.evaluate(current_output, context=context)
        improvement_history.append(eval_result.score)
        
        logger.info(f"Initial score: {eval_result.score:.3f}")
        
        # Check if already meets threshold
        if eval_result.score >= threshold:
            logger.info("Initial output already meets threshold")
            return self._create_result(
                current_output,
                0,
                improvement_history,
                critique_history,
                iteration_results,
                "threshold_met_initially",
                start_time,
                gen_result,
                context,
            )
        
        # Iterative refinement loop
        iterations_without_improvement = 0
        last_best_score = eval_result.score
        best_output = current_output
        
        for iteration in range(1, max_iterations + 1):
            iter_result = self.iterate_once(
                current_output,
                iteration,
                threshold,
                context,
            )
            
            iteration_results.append(iter_result)
            critique_history.append(iter_result.critique)
            improvement_history.append(iter_result.evaluation.score)
            self.tracker.record_score(iter_result.evaluation.score)
            
            # Update best output
            if iter_result.evaluation.score > last_best_score:
                last_best_score = iter_result.evaluation.score
                best_output = iter_result.output
                iterations_without_improvement = 0
                logger.info(f"Iteration {iteration}: New best score {last_best_score:.3f}")
            else:
                iterations_without_improvement += 1
                logger.debug(f"Iteration {iteration}: No improvement ({iterations_without_improvement}/{self.patience})")
            
            current_output = iter_result.output
            
            # Check stopping conditions
            if not iter_result.should_continue:
                stopped_reason = "convergence_reached" if iter_result.converged else "threshold_met"
                logger.info(f"Stopping early: {stopped_reason}")
                break
            
            # Check patience
            if self.early_stopping and iterations_without_improvement >= self.patience:
                logger.info(f"Stopping: No improvement for {self.patience} iterations")
                break
            
            # Check threshold
            if iter_result.evaluation.score >= threshold:
                logger.info("Quality threshold reached")
                break
        else:
            logger.info(f"Max iterations ({max_iterations}) reached")
        
        # Determine final stopped reason
        if iter_result.evaluation.score >= threshold:
            stopped_reason = "threshold_met"
        elif iterations_without_improvement >= self.patience:
            stopped_reason = "no_improvement"
        else:
            stopped_reason = "max_iterations"
        
        return self._create_result(
            best_output,
            len(iteration_results),
            improvement_history,
            critique_history,
            iteration_results,
            stopped_reason,
            start_time,
            gen_result,
            context,
        )
    
    def iterate_once(
        self,
        current_output: str,
        iteration: int,
        threshold: float,
        context: Dict[str, Any],
    ) -> IterationResult:
        """Perform a single iteration of the refinement loop.
        
        Args:
            current_output: Current content to refine
            iteration: Current iteration number
            threshold: Quality threshold
            context: Refinement context
            
        Returns:
            IterationResult with iteration details
        """
        logger.debug(f"Iteration {iteration}: Starting")
        
        # Critique current output
        critique = self.critic.critique(current_output, context=context)
        
        # Refine based on critique
        refined = self.refiner.refine(current_output, critique)
        
        # Evaluate refined output
        evaluation = self.judge.evaluate(refined.content, context=context)
        
        # Calculate improvement
        previous_scores = self.tracker.improvement_history
        if previous_scores:
            improvement = evaluation.score - previous_scores[-1]
        else:
            improvement = 0.0
        
        # Check convergence
        converged = self.tracker.has_converged
        
        # Determine if should continue
        should_continue = (
            evaluation.score < threshold and
            not converged and
            improvement > -self.min_improvement  # Allow small regressions
        )
        
        result = IterationResult(
            iteration=iteration,
            output=refined.content,
            critique=critique,
            evaluation=evaluation,
            improvement=improvement,
            converged=converged,
            should_continue=should_continue,
        )
        
        logger.debug(f"Iteration {iteration}: score={evaluation.score:.3f}, improvement={improvement:+.3f}")
        return result
    
    def should_continue(
        self,
        current_score: float,
        iteration: int,
        threshold: Optional[float] = None,
    ) -> bool:
        """Determine if refinement should continue.
        
        Args:
            current_score: Current quality score
            iteration: Current iteration number
            threshold: Quality threshold
            
        Returns:
            True if should continue iterating
        """
        threshold = threshold or self.quality_threshold
        
        if current_score >= threshold:
            return False
        if iteration >= self.max_iterations:
            return False
        if self.tracker.has_converged:
            return False
        return True
    
    def get_best_version(self, history: List[IterationResult]) -> Optional[IterationResult]:
        """Get the best version from iteration history.
        
        Args:
            history: List of iteration results
            
        Returns:
            Best iteration result by score
        """
        if not history:
            return None
        return max(history, key=lambda r: r.evaluation.score)
    
    def _create_result(
        self,
        final_output: str,
        iterations: int,
        improvement_history: List[float],
        critique_history: List[CritiqueResult],
        iteration_results: List[IterationResult],
        stopped_reason: str,
        start_time: float,
        gen_result: Optional[GenerationResult],
        context: Dict[str, Any],
    ) -> RefinementResult:
        """Create final RefinementResult."""
        total_time = time.time() - start_time
        
        return RefinementResult(
            final_output=final_output,
            iterations=iterations,
            improvement_history=improvement_history,
            final_score=improvement_history[-1] if improvement_history else 0.0,
            critique_history=critique_history,
            iteration_results=iteration_results,
            convergence_reached=self.tracker.has_converged,
            stopped_reason=stopped_reason,
            metadata={
                "run_number": self._run_count,
                "total_time": total_time,
                "avg_time_per_iteration": total_time / max(iterations, 1),
                "generator_used": gen_result is not None,
                "tracker_stats": self.tracker.get_stats(),
                **context,
            },
        )
    
    def quick_refine(
        self,
        prompt: str,
        target_iterations: int = 3,
    ) -> str:
        """Quick refinement with minimal configuration.
        
        Args:
            prompt: Generation prompt
            target_iterations: Number of iterations to run
            
        Returns:
            Refined output string
        """
        # Set early_stopping temporarily
        original_early_stopping = self.early_stopping
        self.early_stopping = False
        try:
            result = self.refine(
                prompt=prompt,
                max_iterations=target_iterations,
            )
        finally:
            self.early_stopping = original_early_stopping
        return result.final_output
    
    def batch_refine(
        self,
        prompts: List[str],
        max_iterations: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[RefinementResult]:
        """Refine multiple prompts in batch.
        
        Args:
            prompts: List of generation prompts
            max_iterations: Override max iterations
            threshold: Override quality threshold
            
        Returns:
            List of RefinementResult objects
        """
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Batch processing {i+1}/{len(prompts)}")
            result = self.refine(
                prompt=prompt,
                max_iterations=max_iterations,
                threshold=threshold,
                context={"batch_index": i, "batch_size": len(prompts)},
            )
            results.append(result)
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_runs": self._run_count,
            "max_iterations": self.max_iterations,
            "quality_threshold": self.quality_threshold,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "tracker": self.tracker.get_stats(),
            "generator": self.generator.get_stats(),
            "critic": self.critic.get_stats(),
            "refiner": self.refiner.get_stats(),
            "judge": self.judge.get_stats(),
        }


class ICRError(Exception):
    """Error during ICR operation."""
    pass


# Convenience function for quick usage
def refine_content(
    content: str,
    max_iterations: int = 3,
    threshold: float = 0.85,
) -> str:
    """Quick convenience function to refine content.
    
    Args:
        content: Content to refine
        max_iterations: Maximum iterations
        threshold: Quality threshold
        
    Returns:
        Refined content
    """
    engine = ICREngine(
        max_iterations=max_iterations,
        quality_threshold=threshold,
    )
    result = engine.refine(
        prompt="",  # Not used since we provide initial_output
        initial_output=content,
    )
    return result.final_output
