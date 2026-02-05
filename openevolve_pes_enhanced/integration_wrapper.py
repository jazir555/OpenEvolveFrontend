"""Integration wrapper - wraps existing OpenEvolve PES without modifying it.

This is the main entry point for the enhancement layer. It provides:
1. Cost-aware planning before evolution
2. Execution monitoring during evolution  
3. Early stopping with convergence detection
4. Summarization after evolution
5. **Budget enforcement** - stops evolution when budget exceeded

All existing functionality is preserved - this is purely additive.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Union, Tuple

# Initialize logger before use
logger = logging.getLogger(__name__)

# Import existing OpenEvolve components (wrapped, not modified)
try:
    from openevolve_agnostic_pes import AgnosticPESEngine, EvolutionResult
    EXISTING_PES_AVAILABLE = True
except ImportError:
    EXISTING_PES_AVAILABLE = False
    logger.warning("openevolve_agnostic_pes not available - standalone mode")

try:
    from leanaide_pes_handler import LeanPESHandler
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False

# Import our enhancement components
from .cost_optimizer import CostOptimizer, BudgetTracker, CostAwarePlanner
from .execution_monitor import ExecutionMonitor, EarlyStoppingController
from .strategy_enhancer import StrategyEnhancer, StrategyDecision, StrategyType
from .summarization_engine import SummarizationEngine, EvolutionSummary
from .budget_enforcer import BudgetEnforcer, BudgetCheckResult, BudgetEnforcedResult
from .config import PESEnhancedConfig

# Import callback system for iteration hooks
try:
    from .evolution_callbacks import (
        EvolutionCallback,
        BudgetAwareCallback,
        MonitoringCallback,
        LoggingCallback,
        CompositeCallback,
        IterationMetrics,
        EvolutionContext,
    )
    from .monitored_engine import MonitoredAgnosticPES, MonitoredEvolutionResult
    CALLBACKS_AVAILABLE = True
except ImportError:
    CALLBACKS_AVAILABLE = False
    logger.debug("Callback system not available - using legacy budget enforcement")


@dataclass
class EnhancedEvolutionResult:
    """Result with enhancements added."""
    
    # Original result data
    original_result: Any
    
    # Enhancement data
    planning_decision: Optional[StrategyDecision]
    execution_summary: Optional[Any]
    evolution_summary: Optional[EvolutionSummary]
    
    # Cost data
    total_cost_usd: float
    efficiency_gain: float
    evaluations_saved: int
    
    # Performance
    converged: bool
    iterations_to_convergence: Optional[int]
    
    # Status
    stopped_early: bool
    stop_reason: Optional[str]
    
    # Budget enforcement (NEW)
    budget_enforced: bool = False
    budget_percent_used: Optional[float] = None
    iterations_completed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": getattr(self.original_result, 'success', True),
            "best_fitness": getattr(self.original_result, 'best_fitness', 0.0),
            "total_evaluations": getattr(self.original_result, 'total_evaluations', 0),
            "efficiency_gain": self.efficiency_gain,
            "evaluations_saved": self.evaluations_saved,
            "total_cost_usd": self.total_cost_usd,
            "converged": self.converged,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "budget_enforced": self.budget_enforced,
            "budget_percent_used": self.budget_percent_used,
            "iterations_completed": self.iterations_completed,
            "strategy_used": self.planning_decision.strategy.value if self.planning_decision else "unknown",
            "recommendations": self.evolution_summary.recommendations if self.evolution_summary else [],
        }


class PESIntegrationWrapper:
    """Main wrapper that enhances existing OpenEvolve PES.
    
    This class wraps around the existing implementation without
    modifying any of its code. All enhancements are purely additive.
    """
    
    def __init__(self, config: Optional[PESEnhancedConfig] = None):
        """Initialize wrapper.
        
        Args:
            config: Enhancement configuration. If None, uses defaults
                   with all enhancements disabled (preserves existing behavior).
        """
        self.config = config or PESEnhancedConfig()
        
        # Enhancement components
        self.cost_optimizer = CostOptimizer(self.config.cost)
        self.strategy_enhancer = StrategyEnhancer()
        self.summarization = SummarizationEngine()
        
        # Execution monitor (created per-run)
        self.execution_monitor: Optional[ExecutionMonitor] = None
        
        # Budget enforcer (created per-run) - NEW
        self.budget_enforcer: Optional[BudgetEnforcer] = None
        
        logger.info("PES Integration Wrapper initialized "
                   f"(enhancements: cost={self.config.enable_cost_optimization}, "
                   f"stopping={self.config.enable_early_stopping}, "
                   f"planning={self.config.enable_planning}, "
                   f"budget_enforcement={self.config.enable_cost_optimization})")
    
    async def enhance_with_planning(
        self,
        code: str,
        problem_description: str,
        tests: List[Dict],
        language: Optional[str] = None,
        max_cost_usd: Optional[float] = None,
        max_iterations: Optional[int] = None,
        **kwargs
    ) -> EnhancedEvolutionResult:
        """Enhance code with cost-aware planning.
        
        This is the enhanced version of enhance_code() from
        openevolve_pes_integration that adds planning and cost
        optimization.
        
        Args:
            code: Code to enhance
            problem_description: Problem description
            tests: Test cases
            language: Programming language
            max_cost_usd: Maximum budget (optional)
            max_iterations: Maximum iterations (optional)
            **kwargs: Additional arguments passed to original
            
        Returns:
            EnhancedEvolutionResult with cost and efficiency data
        """
        start_time = time.time() * 1000
        
        # === PHASE 1: PLANNING ===
        planning_decision = None
        if self.config.enable_planning:
            budget = {"max_cost_usd": max_cost_usd} if max_cost_usd else None
            
            planning_decision = self.strategy_enhancer.selector.select_strategy(
                problem_description=problem_description,
                code=code,
                language=language,
                max_cost_usd=max_cost_usd or self.config.cost.max_cost_usd,
                max_time_seconds=self.config.cost.max_time_seconds
            )
            
            # Override parameters based on planning
            if max_iterations is None:
                max_iterations = planning_decision.recommended_parameters.get("iterations", 50)
            
            logger.info(f"Planning: selected {planning_decision.strategy.value} strategy")
        
        # === PHASE 2: SETUP COST TRACKING & BUDGET ENFORCEMENT ===
        if self.config.enable_cost_optimization and max_cost_usd:
            self.cost_optimizer.initialize_budget(
                max_cost_usd=max_cost_usd,
                max_tokens=self.config.cost.max_tokens,
                max_time_ms=self.config.cost.max_time_seconds * 1000
            )
            
            # Create budget enforcer - NEW
            self.budget_enforcer = BudgetEnforcer(
                budget_tracker=self.cost_optimizer.budget_tracker,
                warning_threshold=self.config.cost.warning_threshold,
                critical_threshold=self.config.cost.critical_threshold
            )
        
        # === PHASE 3: SETUP EXECUTION MONITORING ===
        if self.config.enable_early_stopping:
            early_stopping = EarlyStoppingController(
                patience=self.config.early_stopping.patience,
                min_improvement=self.config.early_stopping.min_improvement,
                max_evaluations=self.config.early_stopping.max_evaluations,
                max_duration_ms=self.config.early_stopping.max_duration_ms
            )
            
            self.execution_monitor = ExecutionMonitor(
                early_stopping=early_stopping,
                budget_tracker=self.cost_optimizer.budget_tracker if self.config.enable_cost_optimization else None
            )
            self.execution_monitor.start()
            
            # Link budget enforcer to execution monitor - NEW
            if self.budget_enforcer:
                self.budget_enforcer.execution_monitor = self.execution_monitor
        
        # === PHASE 4: RUN EVOLUTION (existing implementation with budget enforcement) ===
        original_result = None
        iterations_completed = 0
        budget_enforced_stop = False
        budget_stop_reason = None
        
        try:
            if EXISTING_PES_AVAILABLE:
                # Use existing AgnosticPESEngine with budget callback
                original_result, iterations_completed = await self._run_with_existing_pes(
                    code=code,
                    tests=tests,
                    language=language,
                    max_iterations=max_iterations,
                    budget_check_callback=self.budget_enforcer.check_budget if self.budget_enforcer else None,
                    **kwargs
                )
            else:
                # Fallback: direct execution
                original_result, iterations_completed = await self._run_fallback(
                    code=code,
                    tests=tests,
                    language=language,
                    max_iterations=max_iterations
                )
        
        except Exception as e:
            logger.error(f"Evolution failed: {e}", exc_info=True)
            if self.config.fallback_on_error:
                logger.warning("Fallback to standard execution")
                original_result = None
            else:
                raise
        
        # Check final budget status
        final_budget_status = None
        if self.budget_enforcer:
            final_budget_status = self.budget_enforcer.get_status()
            if not final_budget_status.can_continue:
                budget_enforced_stop = True
                budget_stop_reason = final_budget_status.reason
                logger.warning(f"Evolution stopped due to budget: {budget_stop_reason}")
        
        # === PHASE 5: SUMMARIZATION ===
        evolution_summary = None
        if self.config.enable_summarization and self.execution_monitor:
            execution_history = [
                {
                    "best_fitness": s.best_fitness,
                    "diversity": s.diversity_score,
                    "evaluations": s.evaluation_count,
                    "timestamp_ms": s.timestamp_ms
                }
                for s in self.execution_monitor.iteration_history
            ]
            
            cost_data = {
                "total_cost_usd": self.cost_optimizer.budget_tracker.cost_used if self.cost_optimizer.budget_tracker else 0.0
            } if self.config.enable_cost_optimization else None
            
            evolution_summary = self.summarization.summarize(
                execution_history=execution_history,
                cost_data=cost_data,
                strategy=planning_decision.strategy.value if planning_decision else "standard",
                problem_type=language or "general"
            )
        
        # === PHASE 6: BUILD RESULT ===
        duration = (time.time() * 1000) - start_time
        
        # Calculate efficiency
        actual_evals = getattr(original_result, 'total_evaluations', 0) if original_result else 0
        if self.config.enable_cost_optimization:
            efficiency = self.cost_optimizer.calculate_efficiency(actual_evals)
            efficiency_gain = efficiency.efficiency_gain
            evals_saved = efficiency.evaluations_saved
        else:
            efficiency_gain = 0.0
            evals_saved = 0
        
        # Get stop reason
        stopped_early = False
        stop_reason = None
        converged = False
        iterations_to_conv = None
        
        if self.execution_monitor and self.execution_monitor.early_stopping:
            status = self.execution_monitor.early_stopping.get_status()
            stopped_early = status["stopped"]
            stop_reason = status["stop_reason"]
            converged = "Converged" in (stop_reason or "")
            if converged and self.execution_monitor.early_stopping.convergence_detector.history:
                iterations_to_conv = len(self.execution_monitor.early_stopping.convergence_detector.history)
        
        # Override with budget stop reason if budget enforced stop
        if budget_enforced_stop:
            stopped_early = True
            stop_reason = budget_stop_reason or "Budget limit reached"
        
        return EnhancedEvolutionResult(
            original_result=original_result,
            planning_decision=planning_decision,
            execution_summary=self.execution_monitor.get_summary() if self.execution_monitor else None,
            evolution_summary=evolution_summary,
            total_cost_usd=self.cost_optimizer.budget_tracker.cost_used if self.cost_optimizer.budget_tracker else 0.0,
            efficiency_gain=efficiency_gain,
            evaluations_saved=evals_saved,
            converged=converged,
            iterations_to_convergence=iterations_to_conv,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            budget_enforced=budget_enforced_stop,
            budget_percent_used=final_budget_status.percent_used if final_budget_status else None,
            iterations_completed=iterations_completed
        )
    
    async def _run_with_existing_pes(
        self,
        code: str,
        tests: List[Dict],
        language: Optional[str],
        max_iterations: int,
        budget_check_callback: Optional[Callable[[], Tuple[bool, str]]] = None,
        callbacks: Optional[List[EvolutionCallback]] = None,
        **kwargs
    ) -> Tuple[Any, int]:
        """Run using existing AgnosticPESEngine with budget enforcement.
        
        This wraps the existing implementation and injects monitoring
        and budget enforcement through callbacks.
        
        Args:
            code: Code to evolve
            tests: Test cases
            language: Programming language
            max_iterations: Maximum iterations
            budget_check_callback: Legacy callback that returns (can_continue, reason)
            callbacks: New-style evolution callbacks for iteration hooks
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (evolution_result, iterations_completed)
        """
        # If new-style callbacks are provided and available, use MonitoredAgnosticPES
        if callbacks and CALLBACKS_AVAILABLE:
            logger.info(f"Using MonitoredAgnosticPES with {len(callbacks)} callbacks")
            return await self._run_with_callbacks(
                code, tests, language, max_iterations, callbacks, **kwargs
            )
        
        # Build callbacks from config if not provided
        if callbacks is None and CALLBACKS_AVAILABLE and self.config:
            callbacks = self._build_callbacks_from_config()
            if callbacks:
                logger.info(f"Using auto-built callbacks: {[c.name for c in callbacks]}")
                return await self._run_with_callbacks(
                    code, tests, language, max_iterations, callbacks, **kwargs
                )
        
        # Fallback: use legacy budget enforcement or original engine
        return await self._run_with_legacy_budget(
            code, tests, language, max_iterations, budget_check_callback, **kwargs
        )
    
    async def _run_with_callbacks(
        self,
        code: str,
        tests: List[Dict],
        language: Optional[str],
        max_iterations: int,
        callbacks: List[EvolutionCallback],
        **kwargs
    ) -> Tuple[Any, int]:
        """Run evolution using MonitoredAgnosticPES with callback support.
        
        This provides true iteration-by-iteration monitoring and control.
        """
        # Create cost estimator from budget tracker if available
        estimate_cost_fn = None
        if self.cost_optimizer and self.cost_optimizer.budget_tracker:
            def estimate_cost():
                # Estimate based on typical API call costs per iteration
                return 0.001  # ~$0.001 per iteration
            estimate_cost_fn = estimate_cost
        
        # Create monitored engine
        engine = MonitoredAgnosticPES(
            max_iterations=max_iterations,
            callbacks=callbacks,
            estimate_cost_per_iteration=estimate_cost_fn,
            **kwargs
        )
        
        # Run evolution
        result = await engine.evolve(code, tests, language or "general")
        
        # Determine iterations completed
        if hasattr(result, 'actual_iterations'):
            iterations_completed = result.actual_iterations
        elif hasattr(result, 'stopped_early') and result.stopped_early:
            # Estimate based on metrics history
            if hasattr(result, 'metrics_history'):
                iterations_completed = len(result.metrics_history)
            else:
                iterations_completed = max_iterations
        else:
            iterations_completed = max_iterations
        
        # Log early stopping
        if hasattr(result, 'stopped_early') and result.stopped_early:
            logger.info(f"Evolution stopped early after {iterations_completed} iterations: {result.stop_reason}")
        
        return result, iterations_completed
    
    def _build_callbacks_from_config(self) -> List[EvolutionCallback]:
        """Build callbacks based on configuration settings."""
        callbacks: List[EvolutionCallback] = []
        
        # Budget-aware callback
        if self.config.enable_cost_optimization and self.cost_optimizer:
            if self.cost_optimizer.budget_tracker:
                budget_tracker = self.cost_optimizer.budget_tracker
                callback = BudgetAwareCallback(
                    max_cost_usd=budget_tracker.max_cost_usd,
                    max_tokens=budget_tracker.max_tokens,
                    max_time_seconds=budget_tracker.max_time_ms / 1000 if budget_tracker.max_time_ms else None
                )
                callbacks.append(callback)
                logger.debug(f"Added BudgetAwareCallback (max_cost=${budget_tracker.max_cost_usd:.2f})")
        
        # Monitoring callback for early stopping
        if self.config.enable_early_stopping and self.execution_monitor:
            if self.execution_monitor.early_stopping:
                es = self.execution_monitor.early_stopping
                callback = MonitoringCallback(
                    patience=es.patience,
                    min_improvement=es.min_improvement,
                    convergence_threshold=0.95
                )
                callbacks.append(callback)
                logger.debug(f"Added MonitoringCallback (patience={es.patience})")
        
        return callbacks
    
    async def _run_with_legacy_budget(
        self,
        code: str,
        tests: List[Dict],
        language: Optional[str],
        max_iterations: int,
        budget_check_callback: Optional[Callable[[], Tuple[bool, str]]] = None,
        **kwargs
    ) -> Tuple[Any, int]:
        """Legacy budget enforcement using periodic asyncio checks.
        
        Used as fallback when callback system is not available.
        Note: This uses asyncio cancellation which is not as clean as
        the callback-based approach in _run_with_callbacks.
        """
        # Create engine
        engine = AgnosticPESEngine(
            max_iterations=max_iterations,
            **kwargs
        )
        
        # Track iterations for partial results
        iterations_completed = 0
        budget_stop_reason = None
        
        # If we have budget enforcement, wrap the evolve method
        if budget_check_callback and self.execution_monitor:
            original_evolve = engine.evolve
            
            async def monitored_evolve(code, tests, problem_type="general"):
                nonlocal iterations_completed, budget_stop_reason
                
                # Check budget before starting
                can_continue, reason = budget_check_callback()
                if not can_continue:
                    logger.warning(f"Budget check failed before evolution: {reason}")
                    budget_stop_reason = reason
                    return None
                
                # Since AgnosticPESEngine.evolve is a black box, we run it
                # but check budget periodically in a background fashion
                import asyncio
                
                # Create evolution task
                evolve_task = asyncio.create_task(
                    original_evolve(code, tests, problem_type)
                )
                
                # Periodically check budget while evolution runs
                check_interval = 0.1  # 100ms
                while not evolve_task.done():
                    # Wait a bit
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(evolve_task),
                            timeout=check_interval
                        )
                    except asyncio.TimeoutError:
                        # Check budget
                        can_continue, reason = budget_check_callback()
                        if not can_continue:
                            logger.warning(f"Budget exceeded during evolution: {reason}")
                            budget_stop_reason = reason
                            # Cancel the evolution task
                            evolve_task.cancel()
                            try:
                                await evolve_task
                            except asyncio.CancelledError:
                                pass
                            break
                
                # Get result if not cancelled
                if evolve_task.done() and not evolve_task.cancelled():
                    try:
                        result = evolve_task.result()
                        iterations_completed = max_iterations
                        return result
                    except Exception as e:
                        logger.error(f"Evolution failed: {e}")
                        raise
                else:
                    return None
            
            result = await monitored_evolve(code, tests, language or "general")
            
        elif budget_check_callback:
            # Just check budget before and after without execution monitor
            can_continue, reason = budget_check_callback()
            if not can_continue:
                logger.warning(f"Budget check failed: {reason}")
                return None, 0
            
            result = await engine.evolve(code, tests, language or "general")
            iterations_completed = max_iterations
            
            # Check budget after
            can_continue, reason = budget_check_callback()
            if not can_continue:
                logger.warning(f"Budget exceeded after evolution: {reason}")
        else:
            # No budget enforcement - original behavior
            result = await engine.evolve(code, tests, language or "general")
            iterations_completed = max_iterations
        
        return result, iterations_completed
    
    async def _run_fallback(
        self,
        code: str,
        tests: List[Dict],
        language: Optional[str],
        max_iterations: int
    ) -> Tuple[Any, int]:
        """Fallback when existing PES not available."""
        logger.warning("Running in fallback mode - limited functionality")
        
        # Simple fallback that just returns the code
        @dataclass
        class SimpleResult:
            code: str
            success: bool = True
            tests_passed: int = 0
            tests_total: int = len(tests)
            iterations: int = 0
            total_evaluations: int = 0
        
        return SimpleResult(code=code), 0
    
    def get_cost_estimate(
        self,
        iterations: int,
        population_size: int,
        problem_complexity: str = "medium"
    ) -> Dict[str, float]:
        """Get cost estimate for a potential run.
        
        This allows users to see costs before running.
        """
        planner = CostAwarePlanner()
        return planner.estimate_cost(iterations, population_size)
    
    def recommend_parameters(
        self,
        problem_description: str,
        max_cost_usd: float = 10.0
    ) -> Dict[str, Any]:
        """Get parameter recommendations for a problem."""
        decision = self.strategy_enhancer.selector.select_strategy(
            problem_description=problem_description,
            code=None,
            language=None,
            max_cost_usd=max_cost_usd
        )
        
        return {
            "strategy": decision.strategy.value,
            "parameters": decision.recommended_parameters,
            "estimated_cost": decision.estimated_cost_usd,
            "estimated_evaluations": decision.estimated_evaluations,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
        }


class EnhancedAgnosticPES:
    """Enhanced version of AgnosticPESEngine.
    
    This provides a drop-in replacement that adds enhancements
    while maintaining the same API.
    """
    
    def __init__(self, max_iterations: int = 10, enable_enhancements: bool = True, **kwargs):
        self.max_iterations = max_iterations
        self.enable_enhancements = enable_enhancements
        
        # Create wrapper if enhancements enabled
        if enable_enhancements:
            config = PESEnhancedConfig.enable_all()
            self.wrapper = PESIntegrationWrapper(config)
        else:
            self.wrapper = None
            # Use original engine directly
            if EXISTING_PES_AVAILABLE:
                self.original_engine = AgnosticPESEngine(max_iterations=max_iterations, **kwargs)
            else:
                self.original_engine = None
    
    async def evolve(self, code: str, tests: List[Dict], problem_type: str = "general"):
        """Evolve code - API compatible with original."""
        if self.enable_enhancements and self.wrapper:
            result = await self.wrapper.enhance_with_planning(
                code=code,
                problem_description=f"Evolve {problem_type} code",
                tests=tests,
                language=problem_type if problem_type != "general" else None,
                max_iterations=self.max_iterations
            )
            return result.original_result
        elif self.original_engine:
            return await self.original_engine.evolve(code, tests, problem_type)
        else:
            raise RuntimeError("No evolution engine available")


class EnhancedLeanHandler:
    """Enhanced Lean 4 handler with cost optimization."""
    
    def __init__(self, enable_enhancements: bool = True):
        self.enable_enhancements = enable_enhancements
        
        if LEAN_AVAILABLE:
            self.original_handler = LeanPESHandler()
        else:
            self.original_handler = None
        
        if enable_enhancements:
            config = PESEnhancedConfig.cost_aware(max_cost_usd=5.0)
            self.wrapper = PESIntegrationWrapper(config)
        else:
            self.wrapper = None
    
    async def complete_proof(
        self,
        theorem_code: str,
        max_cost_usd: float = 5.0
    ) -> Dict[str, Any]:
        """Complete a Lean proof with cost awareness."""
        if self.enable_enhancements and self.wrapper:
            # Use enhanced version
            result = await self.wrapper.enhance_with_planning(
                code=theorem_code,
                problem_description="Complete Lean 4 proof",
                tests=[],  # Lean proofs verified by compiler
                language="lean",
                max_cost_usd=max_cost_usd
            )
            return result.to_dict()
        elif self.original_handler:
            # Use original
            return await self.original_handler.complete_proof(theorem_code)
        else:
            raise RuntimeError("Lean handler not available")


# Convenience functions for easy use

def create_cost_aware_enhancer(max_cost_usd: float = 10.0) -> PESIntegrationWrapper:
    """Create an enhancer focused on cost optimization.
    
    This is the main entry point for users who want to add
    cost awareness to their existing OpenEvolve PES usage.
    
    Example:
        enhancer = create_cost_aware_enhancer(max_cost_usd=5.0)
        result = await enhancer.enhance_with_planning(
            code=my_code,
            problem_description="Optimize sorting",
            tests=my_tests
        )
        print(f"Cost: ${result.total_cost_usd:.2f}, Efficiency: {result.efficiency_gain:.0%}")
    """
    config = PESEnhancedConfig.cost_aware(max_cost_usd=max_cost_usd)
    return PESIntegrationWrapper(config)


def create_fully_enhanced() -> PESIntegrationWrapper:
    """Create an enhancer with all enhancements enabled."""
    config = PESEnhancedConfig.enable_all()
    return PESIntegrationWrapper(config)
