"""Monitored wrapper for AgnosticPESEngine with callback injection.

This module provides MonitoredAgnosticPES which wraps the original
AgnosticPESEngine to inject monitoring callbacks between iterations.

Since the original engine runs as a black box, this wrapper reimplements
the core evolution loop with callback hooks while reusing all the
language-agnostic analysis and fix generation components.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable

# Import original components
from openevolve_agnostic_pes import (
    AgnosticPESEngine,
    EvolutionResult,
    LanguageDetector,
    UniversalCodeAnalyzer,
    UniversalFixGenerator,
    UniversalTestRunner,
)

from .evolution_callbacks import (
    EvolutionCallback,
    EvolutionContext,
    EvolutionState,
    IterationMetrics,
    CompositeCallback,
    create_standard_callbacks,
)

logger = logging.getLogger(__name__)


@dataclass
class MonitoredEvolutionResult(EvolutionResult):
    """Extended result with monitoring data."""
    
    # Original EvolutionResult fields are inherited
    
    # Additional monitoring data
    stopped_early: bool = False
    stop_reason: Optional[str] = None
    actual_iterations: int = 0
    
    # Metrics history
    metrics_history: List[IterationMetrics] = field(default_factory=list)
    
    # Callback data
    callback_results: Dict[str, Any] = field(default_factory=dict)
    
    # Cost tracking
    total_cost_usd: float = 0.0
    cost_per_iteration: List[float] = field(default_factory=list)


class MonitoredAgnosticPES:
    """Wrapper for AgnosticPESEngine that injects callback monitoring.
    
    This class wraps the original AgnosticPESEngine to add callback hooks
    between iterations. It reuses all the original engine's components
    (language detection, code analysis, fix generation) while adding:
    
    1. Pre/post iteration callbacks
    2. Budget monitoring and enforcement
    3. Convergence detection
    4. Early stopping capability
    
    Usage:
        callbacks = [
            BudgetAwareCallback(max_cost_usd=5.0),
            MonitoringCallback(patience=3)
        ]
        
        engine = MonitoredAgnosticPES(
            max_iterations=10,
            callbacks=callbacks
        )
        
        result = await engine.evolve(code, tests, language)
        
        if result.stopped_early:
            print(f"Stopped: {result.stop_reason}")
    """
    
    def __init__(
        self,
        max_iterations: int = 5,
        callbacks: Optional[List[EvolutionCallback]] = None,
        estimate_cost_per_iteration: Optional[Callable[[], float]] = None,
        **kwargs
    ):
        """Initialize monitored engine.
        
        Args:
            max_iterations: Maximum evolution iterations
            callbacks: List of callbacks to invoke during evolution
            estimate_cost_per_iteration: Function to estimate iteration cost
            **kwargs: Additional arguments (preserved for compatibility)
        """
        self.max_iterations = max_iterations
        self.callbacks = callbacks or []
        self.estimate_cost_fn = estimate_cost_per_iteration
        self.kwargs = kwargs
        
        # Create composite callback for easier management
        if self.callbacks:
            self._composite = CompositeCallback(self.callbacks)
        else:
            self._composite = None
        
        # Evolution context
        self._context: Optional[EvolutionContext] = None
        
        logger.info(f"MonitoredAgnosticPES initialized with {len(self.callbacks)} callbacks")
    
    async def evolve(
        self,
        code: str,
        tests: List[Dict],
        problem_type: str = "general"
    ) -> MonitoredEvolutionResult:
        """Evolve code with callback monitoring.
        
        This method reimplements the core evolution loop from AgnosticPESEngine
        with callback hooks injected at each iteration.
        
        Args:
            code: The source code to evolve
            tests: List of test cases
            problem_type: Type of problem (used for language hint)
        
        Returns:
            MonitoredEvolutionResult with evolution data and monitoring info
        """
        logger.info(f"Starting monitored evolution for {len(tests)} tests")
        
        # Initialize timing
        start_time_ms = int(time.time() * 1000)
        
        # Auto-detect language (same as original)
        language = LanguageDetector.detect(code)
        logger.info(f"Detected language: {language}")
        
        # Analyze code structure (same as original)
        analysis = UniversalCodeAnalyzer.analyze(code)
        logger.info(f"Code analysis: {analysis['functions']} functions found")
        
        # Initialize evolution context
        self._context = EvolutionContext(
            state=EvolutionState.RUNNING,
            current_iteration=0,
            max_iterations=self.max_iterations,
            start_time_ms=start_time_ms,
            problem_type=problem_type,
            language=language
        )
        
        # Notify callbacks: evolution start
        if self._composite:
            try:
                await self._composite.on_evolution_start(
                    self._context, code, tests
                )
            except Exception as e:
                logger.error(f"Callback error in on_evolution_start: {e}")
        
        # Run evolution loop with callback hooks
        result = await self._run_evolution_loop(
            code=code,
            tests=tests,
            language=language,
            analysis=analysis
        )
        
        return result
    
    async def _run_evolution_loop(
        self,
        code: str,
        tests: List[Dict],
        language: str,
        analysis: Dict
    ) -> MonitoredEvolutionResult:
        """Run the evolution loop with callback monitoring.
        
        This reimplements the core loop from AgnosticPESEngine with hooks.
        """
        start_time_ms = self._context.start_time_ms
        
        current_code = code
        fixes_applied = []
        best_code = code
        best_score = 0.0
        metrics_history: List[IterationMetrics] = []
        
        stopped_early = False
        stop_reason = None
        actual_iterations = 0
        total_cost = 0.0
        
        for iteration in range(self.max_iterations):
            actual_iterations = iteration + 1
            self._context.current_iteration = iteration
            
            # === PRE-ITERATION: Check if should stop ===
            if self._composite:
                should_stop, reason = await self._composite.should_stop(
                    self._context,
                    metrics_history[-1] if metrics_history else None
                )
                if should_stop:
                    logger.info(f"Stopping early: {reason}")
                    stopped_early = True
                    stop_reason = reason
                    self._context.stop_requested = True
                    self._context.stop_reason = reason
                    break
            
            # === PRE-ITERATION: Notify callbacks ===
            if self._composite:
                try:
                    await self._composite.on_iteration_start(
                        iteration, self._context
                    )
                except Exception as e:
                    logger.error(f"Callback error in on_iteration_start: {e}")
            
            iteration_start_ms = int(time.time() * 1000)
            
            # === CORE ITERATION LOGIC (from AgnosticPESEngine) ===
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Generate and run tests
            test_wrapper = UniversalTestRunner.generate_test_wrapper(
                current_code, tests, language
            )
            passed, total, failing = UniversalTestRunner.execute(
                test_wrapper, language
            )
            
            # Calculate score
            score = passed / total if total > 0 else 0.0
            logger.info(f"Score: {score:.1%} ({passed}/{total})")
            logger.info(f"Failing: {failing}")
            
            # If all tests pass, we're done
            if score == 1.0:
                best_code = current_code
                best_score = score
                logger.info("All tests passing! Evolution complete.")
                
                # Record final metrics for this iteration
                iteration_metrics = self._create_iteration_metrics(
                    iteration=iteration,
                    score=score,
                    passed=passed,
                    total=total,
                    failing=failing,
                    current_code=current_code,
                    fixes_applied_this_iteration=[],
                    fixes_applied_total=fixes_applied,
                    iteration_start_ms=iteration_start_ms,
                    start_time_ms=start_time_ms
                )
                metrics_history.append(iteration_metrics)
                
                # Post-iteration callback
                if self._composite:
                    try:
                        await self._composite.on_iteration_end(
                            iteration, iteration_metrics, self._context
                        )
                    except Exception as e:
                        logger.error(f"Callback error in on_iteration_end: {e}")
                
                break
            
            # Analyze failures and generate fixes
            fixes_this_iteration = []
            
            for test_name in failing:
                # Find the test case
                test_case = next(
                    (t for t in tests if t.get("name") == test_name), None
                )
                if not test_case:
                    continue
                
                # Analyze failure
                fix_request = UniversalFixGenerator.analyze_failure(
                    test_name,
                    test_case.get("input", {}),
                    test_case.get("expected", {}),
                    current_code,
                    analysis
                )
                
                if fix_request:
                    # Generate fix
                    new_code = UniversalFixGenerator.generate_fix(
                        current_code, analysis, fix_request
                    )
                    
                    if new_code != current_code:
                        # Apply fix
                        current_code = new_code
                        fix_desc = f"{fix_request['strategy']}:{fix_request.get('context', {}).get('value', '')}"
                        fixes_applied.append(fix_desc)
                        fixes_this_iteration.append(fix_desc)
                        logger.info(f"Applied fix: {fix_desc}")
            
            if fixes_this_iteration:
                # Run tests on the fixed code
                final_wrapper = UniversalTestRunner.generate_test_wrapper(
                    current_code, tests, language
                )
                new_passed, new_total, _ = UniversalTestRunner.execute(
                    final_wrapper, language
                )
                new_score = new_passed / new_total if new_total > 0 else 0.0
                
                if new_score > best_score:
                    best_score = new_score
                    best_code = current_code
                    logger.info(f"New best score: {best_score:.1%} ({new_passed}/{new_total})")
                
                if new_score == 1.0:
                    logger.info("All tests passing! Evolution complete.")
                    
                    # Record metrics
                    iteration_metrics = self._create_iteration_metrics(
                        iteration=iteration,
                        score=new_score,
                        passed=new_passed,
                        total=new_total,
                        failing=[],
                        current_code=current_code,
                        fixes_applied_this_iteration=fixes_this_iteration,
                        fixes_applied_total=fixes_applied,
                        iteration_start_ms=iteration_start_ms,
                        start_time_ms=start_time_ms
                    )
                    metrics_history.append(iteration_metrics)
                    
                    # Post-iteration callback
                    if self._composite:
                        try:
                            await self._composite.on_iteration_end(
                                iteration, iteration_metrics, self._context
                            )
                        except Exception as e:
                            logger.error(f"Callback error in on_iteration_end: {e}")
                    
                    break
            else:
                new_score = score
                new_passed = passed
                new_total = total
                logger.info("No more fixes applicable")
                
                # Record metrics anyway for this iteration
                iteration_metrics = self._create_iteration_metrics(
                    iteration=iteration,
                    score=score,
                    passed=passed,
                    total=total,
                    failing=failing,
                    current_code=current_code,
                    fixes_applied_this_iteration=[],
                    fixes_applied_total=fixes_applied,
                    iteration_start_ms=iteration_start_ms,
                    start_time_ms=start_time_ms
                )
                metrics_history.append(iteration_metrics)
                
                # Post-iteration callback
                if self._composite:
                    try:
                        await self._composite.on_iteration_end(
                            iteration, iteration_metrics, self._context
                        )
                    except Exception as e:
                        logger.error(f"Callback error in on_iteration_end: {e}")
                
                # Check if we should continue
                if not fixes_this_iteration and iteration > 0:
                    # No fixes this iteration, check if we should stop
                    if self._composite:
                        should_stop, reason = await self._composite.should_stop(
                            self._context, iteration_metrics
                        )
                        if should_stop:
                            logger.info(f"Stopping: {reason}")
                            stopped_early = True
                            stop_reason = reason
                            break
                
                continue  # Skip to next iteration
            
            # === POST-ITERATION: Record metrics ===
            iteration_metrics = self._create_iteration_metrics(
                iteration=iteration,
                score=new_score,
                passed=new_passed,
                total=new_total,
                failing=[f for f in failing if f not in [t.get("name") for t in tests]],
                current_code=current_code,
                fixes_applied_this_iteration=fixes_this_iteration,
                fixes_applied_total=fixes_applied,
                iteration_start_ms=iteration_start_ms,
                start_time_ms=start_time_ms
            )
            metrics_history.append(iteration_metrics)
            
            # Estimate cost
            if self.estimate_cost_fn:
                cost = self.estimate_cost_fn()
                total_cost += cost
                iteration_metrics.cost_this_iteration = cost
                iteration_metrics.total_cost = total_cost
            
            # === POST-ITERATION: Notify callbacks ===
            if self._composite:
                try:
                    await self._composite.on_iteration_end(
                        iteration, iteration_metrics, self._context
                    )
                except Exception as e:
                    logger.error(f"Callback error in on_iteration_end: {e}")
            
            # === POST-ITERATION: Check for early stopping ===
            if self._composite:
                should_stop, reason = await self._composite.should_stop(
                    self._context, iteration_metrics
                )
                if should_stop:
                    logger.info(f"Stopping after iteration {iteration + 1}: {reason}")
                    stopped_early = True
                    stop_reason = reason
                    break
        
        # Final evaluation
        final_wrapper = UniversalTestRunner.generate_test_wrapper(
            best_code, tests, language
        )
        final_passed, final_total, _ = UniversalTestRunner.execute(
            final_wrapper, language
        )
        final_score = final_passed / final_total if final_total > 0 else 0.0
        
        original_wrapper = UniversalTestRunner.generate_test_wrapper(
            code, tests, language
        )
        original_passed, original_total, _ = UniversalTestRunner.execute(
            original_wrapper, language
        )
        original_score = original_passed / original_total if original_total > 0 else 0.0
        
        # Build result
        result = MonitoredEvolutionResult(
            original_code=code,
            evolved_code=best_code,
            iterations=self.max_iterations,
            fixes_applied=fixes_applied,
            improvement=final_score - original_score,
            final_score=final_score,
            tests_passed=final_passed,
            tests_total=final_total,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            actual_iterations=actual_iterations,
            metrics_history=metrics_history,
            total_cost_usd=total_cost,
            cost_per_iteration=[m.cost_this_iteration for m in metrics_history]
        )
        
        # Notify callbacks: evolution end
        if self._composite:
            try:
                await self._composite.on_evolution_end(
                    self._context,
                    metrics_history[-1] if metrics_history else None,
                    result
                )
            except Exception as e:
                logger.error(f"Callback error in on_evolution_end: {e}")
        
        self._context.state = EvolutionState.COMPLETED
        
        return result
    
    def _create_iteration_metrics(
        self,
        iteration: int,
        score: float,
        passed: int,
        total: int,
        failing: List[str],
        current_code: str,
        fixes_applied_this_iteration: List[str],
        fixes_applied_total: List[str],
        iteration_start_ms: int,
        start_time_ms: int
    ) -> IterationMetrics:
        """Create IterationMetrics from iteration data."""
        iteration_duration = int(time.time() * 1000) - iteration_start_ms
        total_duration = int(time.time() * 1000) - start_time_ms
        
        return IterationMetrics(
            iteration=iteration,
            total_iterations=self.max_iterations,
            best_fitness=score,
            avg_fitness=score,  # Same as best for this engine
            tests_passed=passed,
            tests_total=total,
            failing_tests=failing,
            current_code=current_code,
            fixes_applied_this_iteration=fixes_applied_this_iteration,
            total_fixes_applied=fixes_applied_total.copy(),
            iteration_duration_ms=iteration_duration,
            total_duration_ms=total_duration
        )
    
    def add_callback(self, callback: EvolutionCallback) -> None:
        """Add a callback to the engine."""
        self.callbacks.append(callback)
        if self._composite:
            self._composite.add_callback(callback)
        else:
            self._composite = CompositeCallback(self.callbacks)
    
    def remove_callback(self, callback_name: str) -> bool:
        """Remove a callback by name."""
        if self._composite:
            return self._composite.remove_callback(callback_name)
        return False


# Convenience wrapper for integration with existing code

class CallbackEnabledEngine:
    """Drop-in replacement for AgnosticPESEngine with callback support.
    
    This class provides the same API as AgnosticPESEngine but adds
    optional callback support.
    
    Usage:
        # Without callbacks (same as AgnosticPESEngine)
        engine = CallbackEnabledEngine(max_iterations=5)
        result = await engine.evolve(code, tests)
        
        # With callbacks
        callbacks = [
            BudgetAwareCallback(max_cost_usd=5.0),
            MonitoringCallback(patience=3)
        ]
        engine = CallbackEnabledEngine(max_iterations=10, callbacks=callbacks)
        result = await engine.evolve(code, tests)
        
        if hasattr(result, 'stopped_early') and result.stopped_early:
            print(f"Stopped: {result.stop_reason}")
    """
    
    def __init__(
        self,
        max_iterations: int = 5,
        callbacks: Optional[List[EvolutionCallback]] = None,
        **kwargs
    ):
        """Initialize engine.
        
        Args:
            max_iterations: Maximum evolution iterations
            callbacks: Optional list of callbacks
            **kwargs: Additional arguments passed to underlying engine
        """
        self.max_iterations = max_iterations
        self.callbacks = callbacks
        self.kwargs = kwargs
        
        # Use monitored engine if callbacks provided, else use original
        if callbacks:
            self._engine = MonitoredAgnosticPES(
                max_iterations=max_iterations,
                callbacks=callbacks,
                **kwargs
            )
        else:
            self._engine = AgnosticPESEngine(
                max_iterations=max_iterations,
                **kwargs
            )
    
    async def evolve(
        self,
        code: str,
        tests: List[Dict],
        problem_type: str = "general"
    ) -> EvolutionResult:
        """Evolve code."""
        return await self._engine.evolve(code, tests, problem_type)


# Factory functions

def create_monitored_engine(
    max_iterations: int = 5,
    max_cost_usd: Optional[float] = None,
    patience: int = 3,
    enable_logging: bool = True,
    **kwargs
) -> MonitoredAgnosticPES:
    """Create a monitored engine with standard callbacks.
    
    Args:
        max_iterations: Maximum evolution iterations
        max_cost_usd: Maximum cost budget (enables budget callback)
        patience: Early stopping patience (enables monitoring callback)
        enable_logging: Whether to include logging callback
        **kwargs: Additional arguments
    
    Returns:
        MonitoredAgnosticPES instance
    """
    callbacks: List[EvolutionCallback] = []
    
    if max_cost_usd is not None:
        from .evolution_callbacks import BudgetAwareCallback
        callbacks.append(BudgetAwareCallback(max_cost_usd=max_cost_usd))
    
    if patience > 0:
        from .evolution_callbacks import MonitoringCallback
        callbacks.append(MonitoringCallback(patience=patience))
    
    if enable_logging:
        from .evolution_callbacks import LoggingCallback
        callbacks.append(LoggingCallback())
    
    return MonitoredAgnosticPES(
        max_iterations=max_iterations,
        callbacks=callbacks if callbacks else None,
        **kwargs
    )
