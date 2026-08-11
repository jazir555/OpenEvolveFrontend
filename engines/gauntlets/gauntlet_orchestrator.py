"""
Gauntlet Orchestrator Module

Provides multi-gauntlet orchestration capabilities for the OpenEvolve system.
Supports sequential, parallel, hierarchical, adaptive, and chain execution modes.

Author: OpenEvolve Team
Date: 2026-02-17
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import statistics

from gauntlet_types import (
    GauntletType, GauntletResult, BaseGauntlet,
    AdversarialGauntlet, FormalVerificationGauntlet, StatisticalGauntlet,
    DomainSpecificGauntlet, MultiObjectiveGauntlet, EvolutionaryGauntlet,
    TemporalGauntlet, CrossValidationGauntlet, create_gauntlet
)

logger = logging.getLogger(__name__)


class OrchestrationMode(Enum):
    """Execution modes for gauntlet orchestration."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"
    CHAIN = "chain"


@dataclass
class OrchestrationResult:
    """Result from multi-gauntlet orchestration."""
    mode: OrchestrationMode
    gauntlet_results: List[GauntletResult] = field(default_factory=list)
    overall_score: float = 0.0
    overall_passed: bool = False
    total_execution_time: float = 0.0
    gauntlets_executed: int = 0
    gauntlets_passed: int = 0
    confidence: float = 0.0
    feedback: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode.value,
            "overall_score": self.overall_score,
            "overall_passed": self.overall_passed,
            "total_execution_time": self.total_execution_time,
            "gauntlets_executed": self.gauntlets_executed,
            "gauntlets_passed": self.gauntlets_passed,
            "confidence": self.confidence,
            "feedback": self.feedback,
            "details": self.details,
            "gauntlet_results": [
                {
                    "gauntlet_type": r.gauntlet_type.value,
                    "gauntlet_name": r.gauntlet_name,
                    "passed": r.passed,
                    "score": r.score,
                    "confidence": r.confidence,
                    "execution_time": r.execution_time,
                    "feedback": r.feedback
                }
                for r in self.gauntlet_results
            ]
        }


@dataclass
class HierarchicalLevel:
    """Configuration for a hierarchical gauntlet level."""
    level: int
    gauntlets: List[BaseGauntlet]
    pass_threshold: float
    stop_on_failure: bool = False


class GauntletOrchestrator:
    """
    Orchestrates execution of multiple gauntlets.
    
    Supports 5 execution modes:
    1. Sequential: Run gauntlets one after another
    2. Parallel: Run gauntlets simultaneously
    3. Hierarchical: Multi-level validation
    4. Adaptive: Dynamic gauntlet selection
    5. Chain: Feed output to next gauntlet
    """

    def __init__(self, max_workers: int = 4, timeout: int = 300):
        """
        Initialize gauntlet orchestrator.
        
        Args:
            max_workers: Maximum parallel workers
            timeout: Timeout in seconds for each gauntlet
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self._lock = Lock()
        self.performance_history: List[Dict[str, Any]] = []
        
        logger.info(f"GauntletOrchestrator initialized (max_workers={max_workers}, timeout={timeout}s)")

    def orchestrate(
        self,
        mode: OrchestrationMode,
        gauntlets: List[BaseGauntlet],
        solution: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> OrchestrationResult:
        """
        Orchestrate gauntlet execution.
        
        Args:
            mode: Execution mode
            gauntlets: List of gauntlets to execute
            solution: Solution to evaluate
            context: Additional context
            
        Returns:
            OrchestrationResult with aggregated results
        """
        start_time = time.time()
        context = context or {}
        
        logger.info(f"Starting {mode.value} orchestration with {len(gauntlets)} gauntlets")
        
        try:
            if mode == OrchestrationMode.SEQUENTIAL:
                result = self._run_sequential(gauntlets, solution, context)
            elif mode == OrchestrationMode.PARALLEL:
                result = self._run_parallel(gauntlets, solution, context)
            elif mode == OrchestrationMode.HIERARCHICAL:
                result = self._run_hierarchical(gauntlets, solution, context)
            elif mode == OrchestrationMode.ADAPTIVE:
                result = self._run_adaptive(gauntlets, solution, context)
            elif mode == OrchestrationMode.CHAIN:
                result = self._run_chain(gauntlets, solution, context)
            else:
                raise ValueError(f"Unknown orchestration mode: {mode}")
            
            result.total_execution_time = time.time() - start_time
            
            # Store performance history
            self._store_performance(mode, result)
            
            logger.info(
                f"Orchestration complete: passed={result.overall_passed}, "
                f"score={result.overall_score:.3f}, time={result.total_execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}", exc_info=True)
            return OrchestrationResult(
                mode=mode,
                overall_score=0.0,
                overall_passed=False,
                total_execution_time=time.time() - start_time,
                confidence=0.0,
                feedback=f"Orchestration error: {str(e)}",
                details={"error": str(e)}
            )

    def _run_sequential(
        self,
        gauntlets: List[BaseGauntlet],
        solution: Any,
        context: Dict[str, Any]
    ) -> OrchestrationResult:
        """Run gauntlets sequentially."""
        results = []
        stop_on_failure = context.get("stop_on_failure", False)
        
        for i, gauntlet in enumerate(gauntlets):
            try:
                logger.info(f"Executing gauntlet {i+1}/{len(gauntlets)}: {gauntlet.name}")
                result = gauntlet.execute(solution, context)
                results.append(result)
                
                if stop_on_failure and not result.passed:
                    logger.info(f"Stopping on failure at gauntlet {i+1}")
                    break
                    
            except Exception as e:
                logger.error(f"Gauntlet {gauntlet.name} failed: {e}")
                results.append(self._create_error_result(gauntlet, str(e)))
        
        return self._aggregate_results(OrchestrationMode.SEQUENTIAL, results)

    def _run_parallel(
        self,
        gauntlets: List[BaseGauntlet],
        solution: Any,
        context: Dict[str, Any]
    ) -> OrchestrationResult:
        """Run gauntlets in parallel."""
        results = []
        
        def execute_gauntlet(gauntlet: BaseGauntlet) -> GauntletResult:
            try:
                return gauntlet.execute(solution, context)
            except Exception as e:
                logger.error(f"Parallel gauntlet {gauntlet.name} failed: {e}")
                return self._create_error_result(gauntlet, str(e))
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_gauntlet = {
                executor.submit(execute_gauntlet, g): g
                for g in gauntlets
            }
            
            for future in as_completed(future_to_gauntlet, timeout=self.timeout * len(gauntlets)):
                result = future.result()
                results.append(result)
        
        return self._aggregate_results(OrchestrationMode.PARALLEL, results)

    def _run_hierarchical(
        self,
        gauntlets: List[BaseGauntlet],
        solution: Any,
        context: Dict[str, Any]
    ) -> OrchestrationResult:
        """Run gauntlets in hierarchical levels."""
        # Organize into 3 levels by gauntlet type
        levels = self._organize_hierarchical_levels(gauntlets, context)
        
        all_results = []
        
        for level in levels:
            logger.info(f"Executing hierarchical level {level.level}")
            level_results = []
            
            for gauntlet in level.gauntlets:
                try:
                    result = gauntlet.execute(solution, context)
                    level_results.append(result)
                except Exception as e:
                    logger.error(f"Hierarchical gauntlet {gauntlet.name} failed: {e}")
                    level_results.append(self._create_error_result(gauntlet, str(e)))
            
            # Check level pass threshold
            level_pass_rate = sum(1 for r in level_results if r.passed) / len(level_results)
            
            if level_pass_rate < level.pass_threshold:
                logger.warning(
                    f"Level {level.level} failed: pass_rate={level_pass_rate:.2f} < threshold={level.pass_threshold}"
                )
                if level.stop_on_failure:
                    break
            
            all_results.extend(level_results)
        
        return self._aggregate_results(OrchestrationMode.HIERARCHICAL, all_results)

    def _run_adaptive(
        self,
        gauntlets: List[BaseGauntlet],
        solution: Any,
        context: Dict[str, Any]
    ) -> OrchestrationResult:
        """Run gauntlets adaptively based on solution characteristics."""
        # Select gauntlets based on context
        selected_gauntlets = self._select_adaptive_gauntlets(gauntlets, context)
        
        logger.info(f"Adaptive selection: {len(selected_gauntlets)}/{len(gauntlets)} gauntlets selected")
        
        # Run selected gauntlets sequentially
        return self._run_sequential(selected_gauntlets, solution, context)

    def _run_chain(
        self,
        gauntlets: List[BaseGauntlet],
        solution: Any,
        context: Dict[str, Any]
    ) -> OrchestrationResult:
        """Run gauntlets in chain, feeding output to next."""
        results = []
        current_solution = solution
        
        for i, gauntlet in enumerate(gauntlets):
            try:
                logger.info(f"Chain gauntlet {i+1}/{len(gauntlets)}: {gauntlet.name}")
                result = gauntlet.execute(current_solution, context)
                results.append(result)
                
                # Use result to potentially modify solution for next gauntlet
                if result.passed and result.improvements:
                    # Apply improvements to solution
                    current_solution = self._apply_improvements(current_solution, result.improvements)
                    
            except Exception as e:
                logger.error(f"Chain gauntlet {gauntlet.name} failed: {e}")
                results.append(self._create_error_result(gauntlet, str(e)))
                break
        
        return self._aggregate_results(OrchestrationMode.CHAIN, results)

    def _aggregate_results(
        self,
        mode: OrchestrationMode,
        results: List[GauntletResult]
    ) -> OrchestrationResult:
        """Aggregate individual gauntlet results."""
        if not results:
            return OrchestrationResult(
                mode=mode,
                overall_score=0.0,
                overall_passed=False,
                confidence=0.0,
                feedback="No gauntlets executed"
            )
        
        # Calculate overall score (weighted average)
        scores = [r.score for r in results]
        weights = self._calculate_weights(results, mode)
        
        weighted_score = sum(s * w for s, w in zip(scores, weights))
        overall_score = weighted_score / sum(weights) if sum(weights) > 0 else 0.0
        
        # Determine pass status
        passed_count = sum(1 for r in results if r.passed)
        overall_passed = passed_count == len(results)  # Must pass all
        
        # Calculate confidence
        confidences = [r.confidence for r in results]
        confidence = statistics.mean(confidences) if confidences else 0.0
        
        # Generate feedback
        feedback = self._generate_feedback(results, mode)
        
        return OrchestrationResult(
            mode=mode,
            gauntlet_results=results,
            overall_score=overall_score,
            overall_passed=overall_passed,
            gauntlets_executed=len(results),
            gauntlets_passed=passed_count,
            confidence=confidence,
            feedback=feedback,
            details={
                "score_distribution": {
                    "mean": statistics.mean(scores) if scores else 0.0,
                    "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    "min": min(scores) if scores else 0.0,
                    "max": max(scores) if scores else 0.0
                },
                "pass_rate": passed_count / len(results) if results else 0.0
            }
        )

    def _calculate_weights(
        self,
        results: List[GauntletResult],
        mode: OrchestrationMode
    ) -> List[float]:
        """Calculate weights for aggregating scores."""
        weights = []
        
        for result in results:
            # Base weight
            weight = 1.0
            
            # Boost weight for certain gauntlet types
            if result.gauntlet_type in [GauntletType.FORMAL_VERIFICATION, GauntletType.ADVERSARIAL]:
                weight *= 1.5
            
            # Boost weight for later rounds in sequential mode
            if mode == OrchestrationMode.SEQUENTIAL:
                idx = results.index(result)
                weight *= (1 + 0.1 * idx)
            
            weights.append(weight)
        
        return weights

    def _generate_feedback(
        self,
        results: List[GauntletResult],
        mode: OrchestrationMode
    ) -> str:
        """Generate aggregated feedback."""
        feedback_parts = []
        
        # Summary
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        feedback_parts.append(f"Passed {passed}/{total} gauntlets")
        
        # Failed gauntlets
        failed = [r for r in results if not r.passed]
        if failed:
            feedback_parts.append(f"Failed: {', '.join(r.gauntlet_name for r in failed)}")
        
        # Top improvements
        all_improvements = []
        for r in results:
            all_improvements.extend(r.improvements[:2])  # Top 2 per gauntlet
        
        if all_improvements:
            feedback_parts.append(f"Top improvements: {all_improvements[0]}")
        
        return ". ".join(feedback_parts)

    def _organize_hierarchical_levels(
        self,
        gauntlets: List[BaseGauntlet],
        context: Dict[str, Any]
    ) -> List[HierarchicalLevel]:
        """Organize gauntlets into hierarchical levels."""
        # Level 1: Quick checks (statistical, basic)
        level1 = [g for g in gauntlets if g.gauntlet_type in [
            GauntletType.STATISTICAL, GauntletType.BASIC
        ]]
        
        # Level 2: Domain-specific and multi-objective
        level2 = [g for g in gauntlets if g.gauntlet_type in [
            GauntletType.DOMAIN_PHYSICS, GauntletType.DOMAIN_FINANCE,
            GauntletType.DOMAIN_CHEMISTRY, GauntletType.DOMAIN_ENGINEERING,
            GauntletType.MULTI_OBJECTIVE
        ]]
        
        # Level 3: Heavy validation (formal, adversarial, evolutionary)
        level3 = [g for g in gauntlets if g.gauntlet_type in [
            GauntletType.FORMAL_VERIFICATION, GauntletType.ADVERSARIAL,
            GauntletType.EVOLUTIONARY
        ]]
        
        # Default levels if categorization doesn't fit
        if not level1 and not level2 and not level3:
            level1 = gauntlets[:len(gauntlets)//3]
            level2 = gauntlets[len(gauntlets)//3:2*len(gauntlets)//3]
            level3 = gauntlets[2*len(gauntlets)//3:]
        
        levels = []
        if level1:
            levels.append(HierarchicalLevel(
                level=1,
                gauntlets=level1,
                pass_threshold=context.get("level1_threshold", 0.7),
                stop_on_failure=context.get("level1_stop", False)
            ))
        if level2:
            levels.append(HierarchicalLevel(
                level=2,
                gauntlets=level2,
                pass_threshold=context.get("level2_threshold", 0.75),
                stop_on_failure=context.get("level2_stop", False)
            ))
        if level3:
            levels.append(HierarchicalLevel(
                level=3,
                gauntlets=level3,
                pass_threshold=context.get("level3_threshold", 0.8),
                stop_on_failure=context.get("level3_stop", True)
            ))
        
        return levels

    def _select_adaptive_gauntlets(
        self,
        gauntlets: List[BaseGauntlet],
        context: Dict[str, Any]
    ) -> List[BaseGauntlet]:
        """Select gauntlets adaptively based on context."""
        domain = context.get("domain", "general")
        complexity = context.get("complexity", "medium")
        time_budget = context.get("time_budget", 300)
        
        selected = []
        
        # Always include basic statistical check
        statistical = [g for g in gauntlets if g.gauntlet_type == GauntletType.STATISTICAL]
        if statistical:
            selected.append(statistical[0])
        
        # Domain-specific gauntlets
        domain_map = {
            "physics": GauntletType.DOMAIN_PHYSICS,
            "finance": GauntletType.DOMAIN_FINANCE,
            "chemistry": GauntletType.DOMAIN_CHEMISTRY,
            "engineering": GauntletType.DOMAIN_ENGINEERING,
            "web3": GauntletType.DOMAIN_WEB3,
        }
        
        if domain in domain_map:
            domain_gauntlets = [g for g in gauntlets if g.gauntlet_type == domain_map[domain]]
            selected.extend(domain_gauntlets[:2])
        
        # High complexity: add formal verification and adversarial
        if complexity in ["high", "critical"]:
            formal = [g for g in gauntlets if g.gauntlet_type == GauntletType.FORMAL_VERIFICATION]
            adversarial = [g for g in gauntlets if g.gauntlet_type == GauntletType.ADVERSARIAL]
            selected.extend(formal[:1])
            selected.extend(adversarial[:1])
        
        # Time budget constraints
        if time_budget < 60:
            # Only quick gauntlets
            selected = [g for g in selected if g.gauntlet_type in [
                GauntletType.STATISTICAL, GauntletType.BASIC
            ]]
        
        return selected if selected else gauntlets[:3]  # Default to first 3

    def _apply_improvements(self, solution: Any, improvements: List[str]) -> Any:
        """Apply improvements to solution."""
        # Simple implementation - in practice, this would modify the solution
        logger.info(f"Applying {len(improvements)} improvements to solution")
        return solution

    def _create_error_result(self, gauntlet: BaseGauntlet, error: str) -> GauntletResult:
        """Create an error result."""
        from datetime import datetime
        return GauntletResult(
            gauntlet_type=gauntlet.gauntlet_type,
            gauntlet_name=gauntlet.name,
            solution_id="error",
            passed=False,
            score=0.0,
            confidence=0.0,
            execution_time=0.0,
            timestamp=datetime.now(),
            details={"error": error},
            feedback=f"Execution error: {error}"
        )

    def _store_performance(self, mode: OrchestrationMode, result: OrchestrationResult):
        """Store performance metrics."""
        self.performance_history.append({
            "mode": mode.value,
            "score": result.overall_score,
            "passed": result.overall_passed,
            "execution_time": result.total_execution_time,
            "gauntlets_executed": result.gauntlets_executed
        })
        
        # Keep only last 100 entries
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]


# Convenience functions
def run_sequential_gauntlets(
    gauntlets: List[BaseGauntlet],
    solution: Any,
    context: Optional[Dict[str, Any]] = None,
    stop_on_failure: bool = False
) -> OrchestrationResult:
    """Run gauntlets sequentially."""
    orchestrator = GauntletOrchestrator()
    ctx = context or {}
    ctx["stop_on_failure"] = stop_on_failure
    return orchestrator.orchestrate(OrchestrationMode.SEQUENTIAL, gauntlets, solution, ctx)


def run_parallel_gauntlets(
    gauntlets: List[BaseGauntlet],
    solution: Any,
    context: Optional[Dict[str, Any]] = None,
    max_workers: int = 4
) -> OrchestrationResult:
    """Run gauntlets in parallel."""
    orchestrator = GauntletOrchestrator(max_workers=max_workers)
    return orchestrator.orchestrate(OrchestrationMode.PARALLEL, gauntlets, solution, context or {})


def run_hierarchical_gauntlets(
    gauntlets: List[BaseGauntlet],
    solution: Any,
    context: Optional[Dict[str, Any]] = None
) -> OrchestrationResult:
    """Run gauntlets hierarchically."""
    orchestrator = GauntletOrchestrator()
    return orchestrator.orchestrate(OrchestrationMode.HIERARCHICAL, gauntlets, solution, context or {})


def run_adaptive_gauntlets(
    gauntlets: List[BaseGauntlet],
    solution: Any,
    context: Optional[Dict[str, Any]] = None
) -> OrchestrationResult:
    """Run gauntlets adaptively."""
    orchestrator = GauntletOrchestrator()
    return orchestrator.orchestrate(OrchestrationMode.ADAPTIVE, gauntlets, solution, context or {})


def run_chain_gauntlets(
    gauntlets: List[BaseGauntlet],
    solution: Any,
    context: Optional[Dict[str, Any]] = None
) -> OrchestrationResult:
    """Run gauntlets in chain."""
    orchestrator = GauntletOrchestrator()
    return orchestrator.orchestrate(OrchestrationMode.CHAIN, gauntlets, solution, context or {})


def run_comprehensive_gauntlet_validation(
    solution: Any,
    context: Optional[Dict[str, Any]] = None,
    mode: OrchestrationMode = OrchestrationMode.HIERARCHICAL
) -> OrchestrationResult:
    """
    Run comprehensive gauntlet validation with all available gauntlet types.
    
    Args:
        solution: Solution to validate
        context: Additional context
        mode: Orchestration mode
        
    Returns:
        Comprehensive validation result
    """
    # Create all gauntlet types
    gauntlets = create_all_gauntlets(context)
    
    # Run orchestration
    orchestrator = GauntletOrchestrator()
    return orchestrator.orchestrate(mode, gauntlets, solution, context or {})


def create_all_gauntlets(context: Optional[Dict[str, Any]] = None) -> List[BaseGauntlet]:
    """
    Create instances of all available gauntlet types.
    
    Args:
        context: Optional context for configuration
        
    Returns:
        List of gauntlet instances
    """
    context = context or {}
    gauntlets = []
    
    try:
        # 1. Adversarial Gauntlet
        gauntlets.append(AdversarialGauntlet(
            "comprehensive_adversarial",
            config={"attack_modes": ["systematic", "adversarial"]}
        ))
    except Exception as e:
        logger.warning(f"Failed to create AdversarialGauntlet: {e}")
    
    try:
        # 2. Formal Verification Gauntlet
        gauntlets.append(FormalVerificationGauntlet(
            "comprehensive_formal",
            config={"timeout": 60}
        ))
    except Exception as e:
        logger.warning(f"Failed to create FormalVerificationGauntlet: {e}")
    
    try:
        # 3. Statistical Gauntlet
        gauntlets.append(StatisticalGauntlet(
            "comprehensive_statistical",
            config={"num_samples": 500}
        ))
    except Exception as e:
        logger.warning(f"Failed to create StatisticalGauntlet: {e}")
    
    try:
        # 4. Domain-Specific Gauntlet (based on context)
        domain = context.get("domain", "physics")
        gauntlets.append(DomainSpecificGauntlet(
            domain=domain,
            config={"strictness": "standard"}
        ))
    except Exception as e:
        logger.warning(f"Failed to create DomainSpecificGauntlet: {e}")
    
    try:
        # 5. Multi-Objective Gauntlet
        gauntlets.append(MultiObjectiveGauntlet(
            "comprehensive_multi_objective",
            config={"objectives": ["correctness", "efficiency", "robustness"]}
        ))
    except Exception as e:
        logger.warning(f"Failed to create MultiObjectiveGauntlet: {e}")
    
    try:
        # 6. Evolutionary Gauntlet
        gauntlets.append(EvolutionaryGauntlet(
            "comprehensive_evolutionary",
            config={"population_size": 30, "generations": 5}
        ))
    except Exception as e:
        logger.warning(f"Failed to create EvolutionaryGauntlet: {e}")
    
    try:
        # 7. Temporal Gauntlet
        gauntlets.append(TemporalGauntlet(
            "comprehensive_temporal",
            config={"stability_threshold": 0.1}
        ))
    except Exception as e:
        logger.warning(f"Failed to create TemporalGauntlet: {e}")
    
    try:
        # 8. Cross-Validation Gauntlet
        gauntlets.append(CrossValidationGauntlet(
            "comprehensive_cross_validation",
            config={"k_folds": 5}
        ))
    except Exception as e:
        logger.warning(f"Failed to create CrossValidationGauntlet: {e}")
    
    logger.info(f"Created {len(gauntlets)}/8 gauntlet types")
    return gauntlets


__all__ = [
    'OrchestrationMode',
    'OrchestrationResult',
    'GauntletOrchestrator',
    'run_sequential_gauntlets',
    'run_parallel_gauntlets',
    'run_hierarchical_gauntlets',
    'run_adaptive_gauntlets',
    'run_chain_gauntlets',
    'run_comprehensive_gauntlet_validation',
    'create_all_gauntlets'
]
