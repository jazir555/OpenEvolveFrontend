"""
SubProblemSolver Integration for Adaptive MDAP.

This module provides integration with the existing SubProblemSolver
in the OpenEvolve system, adding adaptive capabilities while maintaining
backward compatibility.
"""

from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

from adaptive_mdap.core.types import SubProblem, SolveStrategy
from adaptive_mdap.classifiers.task_complexity_classifier import TaskComplexityClassifier
from adaptive_mdap.allocators.resource_allocator import AdaptiveMDAPAllocator, AllocationContext
from adaptive_mdap.controllers.execution_controller import AdaptiveExecutionController
from adaptive_mdap.utils.logger import get_logger

logger = get_logger("integrations.subproblem_solver")


@dataclass
class AdaptiveSolveResult:
    """Result of an adaptive solve operation."""
    solution: Any
    success: bool
    complexity_score: float
    strategy_used: str
    n_agents: int
    duration_ms: float
    metadata: Dict[str, Any]


class SubProblemSolverIntegration:
    """
    Integration with SubProblemSolver.
    
    This class wraps the adaptive components and provides a clean interface
    for integration with the existing SubProblemSolver class.
    
    Usage:
        integration = SubProblemSolverIntegration()
        result = integration.solve_adaptive(subproblem)
    """
    
    def __init__(
        self,
        classifier: Optional[TaskComplexityClassifier] = None,
        allocator: Optional[AdaptiveMDAPAllocator] = None,
        enable_adaptive: bool = True,
        classifier_config: Optional[Any] = None,
        allocator_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the integration.
        
        Args:
            classifier: Complexity classifier (creates default if None)
            allocator: Resource allocator (creates default if None)
            enable_adaptive: Whether to enable adaptive allocation
            classifier_config: Configuration for TaskComplexityClassifier
            allocator_config: Configuration for AdaptiveMDAPAllocator
        """
        self.classifier = classifier or TaskComplexityClassifier(config=classifier_config)
        
        if allocator:
            self.allocator = allocator
        else:
            # Create allocator with custom config if provided
            if allocator_config:
                self.allocator = AdaptiveMDAPAllocator(**allocator_config)
            else:
                self.allocator = AdaptiveMDAPAllocator()
                
        self.enable_adaptive = enable_adaptive
        self.controller = AdaptiveExecutionController(
            classifier=self.classifier,
            allocator=self.allocator,
        )
        
        logger.info(f"Initialized SubProblemSolverIntegration (adaptive={enable_adaptive})")
    
    def solve_adaptive(
        self,
        subproblem: SubProblem,
        context: Optional[AllocationContext] = None,
        force_strategy: Optional[SolveStrategy] = None,
    ) -> AdaptiveSolveResult:
        """
        Solve a sub-problem with adaptive resource allocation.
        
        Args:
            subproblem: The sub-problem to solve
            context: Optional allocation context
            force_strategy: Optional strategy override
            
        Returns:
            AdaptiveSolveResult with solution and metadata
        """
        attempt = self.controller.execute_adaptive(
            subproblem=subproblem,
            context=context,
            force_strategy=force_strategy,
        )
        
        return AdaptiveSolveResult(
            solution=attempt.solution,
            success=attempt.status.value == "completed",
            complexity_score=attempt.complexity_score,
            strategy_used=attempt.allocated_strategy,
            n_agents=attempt.n_agents,
            duration_ms=attempt.duration_ms,
            metadata=attempt.metadata,
        )
    
    def get_adaptive_stats(self) -> Dict[str, Any]:
        """Get adaptive allocation statistics."""
        return {
            "classifier": self.classifier.get_cache_stats(),
            "allocator": self.allocator.get_allocation_stats(),
            "controller": self.controller.get_execution_stats(),
        }
    
    def update_allocator_thresholds(self, thresholds: list) -> None:
        """
        Update allocator thresholds.
        
        Args:
            thresholds: New thresholds [t1, t2]
        """
        self.allocator.update_thresholds(thresholds, reason="manual_update")
        logger.info(f"Updated thresholds to {thresholds}")
