"""
Adaptive MDAP Integration for Workflow Engine.

This module provides integration between the Workflow Engine and Adaptive MDAP,
enabling intelligent resource allocation for sub-problem solving.
"""

import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from adaptive_mdap.core.types import SubProblem as AdaptiveSubProblem, SolveStrategy
from adaptive_mdap.classifiers.task_complexity_classifier import TaskComplexityClassifier
from adaptive_mdap.allocators.resource_allocator import AdaptiveMDAPAllocator, AllocationContext
from adaptive_mdap.controllers.execution_controller import AdaptiveExecutionController
from adaptive_mdap.utils.logger import get_logger

logger = get_logger("integrations.workflow_engine")

# Try to import workflow engine types
try:
    from workflow_structures import SubProblem, SolutionAttempt, WorkflowState
    WORKFLOW_TYPES_AVAILABLE = True
except ImportError:
    WORKFLOW_TYPES_AVAILABLE = False
    SubProblem = None
    SolutionAttempt = None
    WorkflowState = None

try:
    from sub_problem_solver import SubProblemSolver
    SUBPROBLEM_SOLVER_AVAILABLE = True
except ImportError:
    SUBPROBLEM_SOLVER_AVAILABLE = False
    SubProblemSolver = None


@dataclass
class AdaptiveWorkflowConfig:
    """Configuration for adaptive workflow integration."""
    enabled: bool = True
    enable_complexity_classification: bool = True
    enable_adaptive_allocation: bool = True
    enable_execution_tracking: bool = True
    default_profile: str = "balanced"
    fallback_on_failure: bool = True


class AdaptiveWorkflowIntegration:
    """
    Integration between Workflow Engine and Adaptive MDAP.
    
    This class provides adaptive resource allocation for workflow execution,
    automatically selecting the optimal solving strategy based on sub-problem
    complexity.
    """
    
    def __init__(
        self,
        config: Optional[AdaptiveWorkflowConfig] = None,
        classifier: Optional[TaskComplexityClassifier] = None,
        allocator: Optional[AdaptiveMDAPAllocator] = None,
        controller: Optional[AdaptiveExecutionController] = None,
    ):
        """
        Initialize adaptive workflow integration.
        
        Args:
            config: Integration configuration
            classifier: Complexity classifier (creates default if None)
            allocator: Resource allocator (creates default if None)
            controller: Execution controller (creates default if None)
        """
        self.config = config or AdaptiveWorkflowConfig()
        
        if not self.config.enabled:
            logger.info("Adaptive workflow integration disabled")
            return
        
        self.classifier = classifier or TaskComplexityClassifier()
        self.allocator = allocator or AdaptiveMDAPAllocator()
        self.controller = controller or AdaptiveExecutionController(
            classifier=self.classifier,
            allocator=self.allocator,
        )
        
        # Statistics
        self._stats = {
            "total_subproblems": 0,
            "classified": 0,
            "adaptively_allocated": 0,
            "fallback_used": 0,
        }
        
        logger.info("Adaptive workflow integration initialized")
    
    def classify_subproblem(
        self,
        sub_problem,
        workflow_context: Optional[Dict[str, Any]] = None
    ) -> Optional[float]:
        """
        Classify sub-problem complexity.
        
        Args:
            sub_problem: SubProblem to classify
            workflow_context: Optional workflow context
            
        Returns:
            Complexity score [0, 1] or None if classification disabled
        """
        if not self.config.enabled or not self.config.enable_complexity_classification:
            return None
        
        try:
            adaptive_sp = self._convert_to_adaptive(sub_problem)
            complexity = self.classifier.compute_complexity(adaptive_sp)
            
            self._stats["classified"] += 1
            
            logger.debug(
                f"Classified {sub_problem.id}: complexity={complexity.overall_score:.3f}"
            )
            
            return complexity.overall_score
            
        except Exception as e:
            logger.warning(f"Failed to classify sub-problem: {e}")
            return None
    
    def allocate_strategy(
        self,
        complexity_score: float,
        workflow_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Allocate solving strategy based on complexity.
        
        Args:
            complexity_score: Complexity score [0, 1]
            workflow_context: Optional workflow context with system load, budget, etc.
            
        Returns:
            Strategy configuration dict
        """
        if not self.config.enabled or not self.config.enable_adaptive_allocation:
            return self._default_strategy()
        
        try:
            # Build allocation context from workflow context
            context = None
            if workflow_context:
                context = AllocationContext(
                    system_load=workflow_context.get("system_load"),
                    budget_remaining=workflow_context.get("budget_remaining"),
                    quality_requirements=workflow_context.get("quality_requirements"),
                )
            
            config = self.allocator.allocate_resources(complexity_score, context)
            
            self._stats["adaptively_allocated"] += 1
            
            return {
                "strategy": config.strategy.value,
                "n_agents": config.n_agents,
                "k_ahead": config.k_ahead,
                "max_retries": config.max_retries,
                "timeout_ms": config.timeout_ms,
                "adaptive": True,
            }
            
        except Exception as e:
            logger.warning(f"Failed to allocate strategy: {e}, using default")
            self._stats["fallback_used"] += 1
            return self._default_strategy()
    
    def get_solver_config(
        self,
        sub_problem,
        workflow_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get complete solver configuration for a sub-problem.
        
        This method:
        1. Classifies the sub-problem complexity
        2. Allocates the optimal strategy
        3. Returns the complete configuration
        
        Args:
            sub_problem: SubProblem to solve
            workflow_context: Optional workflow context
            
        Returns:
            Solver configuration dict with complexity, strategy, and settings
        """
        if not self.config.enabled:
            return self._default_strategy()
        
        self._stats["total_subproblems"] += 1
        
        # Step 1: Classify complexity
        complexity = self.classify_subproblem(sub_problem, workflow_context)
        
        if complexity is None:
            return self._default_strategy()
        
        # Step 2: Allocate strategy
        strategy = self.allocate_strategy(complexity, workflow_context)
        
        return {
            "complexity_score": complexity,
            **strategy,
        }
    
    def create_solver(
        self,
        openevolve_client=None,
        strategy_config: Optional[Dict[str, Any]] = None
    ):
        """
        Create a SubProblemSolver with adaptive configuration.
        
        Args:
            openevolve_client: OpenEvolve client
            strategy_config: Optional strategy configuration from get_solver_config
            
        Returns:
            Configured SubProblemSolver or None if not available
        """
        if not SUBPROBLEM_SOLVER_AVAILABLE:
            logger.warning("SubProblemSolver not available")
            return None
        
        # Map strategy to maker preset
        strategy_map = {
            "direct": "FAST",
            "mdap_light": "BALANCED",
            "mdap_medium": "BALANCED",
            "maker_full": "ZERO_ERROR",
            "maker_ultra": "ZERO_ERROR",
        }
        
        preset = None
        if strategy_config and "strategy" in strategy_config:
            preset = strategy_map.get(strategy_config["strategy"])
        
        return SubProblemSolver(
            openevolve_client=openevolve_client,
            enable_adaptive_allocation=self.config.enabled,
            maker_preset=preset,
        )
    
    def _convert_to_adaptive(self, sub_problem) -> AdaptiveSubProblem:
        """Convert workflow SubProblem to adaptive SubProblem."""
        return AdaptiveSubProblem(
            id=sub_problem.id,
            description=sub_problem.description,
            domain=getattr(sub_problem, "domain", "general"),
            depth=getattr(sub_problem, "depth", 0),
            dependencies=[dep.id for dep in getattr(sub_problem, "dependencies", [])],
            metadata={
                "type": getattr(sub_problem, "type", "implementation"),
                "priority": getattr(sub_problem, "priority", 0),
            },
        )
    
    def _default_strategy(self) -> Dict[str, Any]:
        """Get default strategy configuration."""
        return {
            "strategy": "maker_full",
            "n_agents": 5,
            "k_ahead": 2,
            "max_retries": 3,
            "timeout_ms": 120000,
            "adaptive": False,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            **self._stats,
            "allocator_stats": self.allocator.get_allocation_stats() if self.allocator else {},
        }
    
    def reset_stats(self):
        """Reset integration statistics."""
        self._stats = {
            "total_subproblems": 0,
            "classified": 0,
            "adaptively_allocated": 0,
            "fallback_used": 0,
        }


# Singleton instance
_adaptive_workflow: Optional[AdaptiveWorkflowIntegration] = None


def get_adaptive_workflow(
    config: Optional[AdaptiveWorkflowConfig] = None
) -> AdaptiveWorkflowIntegration:
    """
    Get or create the global adaptive workflow integration.
    
    Args:
        config: Optional configuration (used only on first call)
        
    Returns:
        AdaptiveWorkflowIntegration instance
    """
    global _adaptive_workflow
    
    if _adaptive_workflow is None:
        _adaptive_workflow = AdaptiveWorkflowIntegration(config=config)
    
    return _adaptive_workflow


def configure_adaptive_workflow(config: AdaptiveWorkflowConfig):
    """
    Configure the global adaptive workflow integration.
    
    Args:
        config: Configuration to apply
    """
    global _adaptive_workflow
    _adaptive_workflow = AdaptiveWorkflowIntegration(config=config)
    logger.info("Adaptive workflow integration reconfigured")


def adaptive_solve_subproblem(
    sub_problem,
    openevolve_client=None,
    workflow_context: Optional[Dict[str, Any]] = None
):
    """
    Solve a sub-problem using adaptive resource allocation.
    
    This is a convenience function for one-off adaptive solving.
    
    Args:
        sub_problem: SubProblem to solve
        openevolve_client: OpenEvolve client
        workflow_context: Optional workflow context
        
    Returns:
        SolutionAttempt from the solver
    """
    integration = get_adaptive_workflow()
    
    # Get solver configuration
    config = integration.get_solver_config(sub_problem, workflow_context)
    
    # Create solver
    solver = integration.create_solver(openevolve_client, config)
    
    if solver is None:
        raise RuntimeError("SubProblemSolver not available")
    
    # Solve
    return solver.solve(sub_problem)
