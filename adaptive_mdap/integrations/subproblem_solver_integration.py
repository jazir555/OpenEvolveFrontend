"""
Integration between Adaptive MDAP and SubProblemSolver.

This module provides the bridge between the existing SubProblemSolver
and the new adaptive MDAP allocation system.
"""

import time
import uuid
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass

from adaptive_mdap.core.types import SubProblem as AdaptiveSubProblem, SolveStrategy
from adaptive_mdap.classifiers.task_complexity_classifier import TaskComplexityClassifier
from adaptive_mdap.allocators.resource_allocator import AdaptiveMDAPAllocator, AllocationContext
# Delay import to avoid circular dependency
# from adaptive_mdap.controllers.execution_controller import AdaptiveExecutionController, SolutionAttempt
from adaptive_mdap.utils.logger import get_logger
from adaptive_mdap.utils.metrics import get_metrics

logger = get_logger("integrations.subproblem_solver")

# Try to import existing SubProblem models
try:
    from sovereign_data_models import SubProblem, SubProblemType, ComplexityScore
    SOVEREIGN_MODELS_AVAILABLE = True
except ImportError:
    SOVEREIGN_MODELS_AVAILABLE = False
    SubProblem = None
    SubProblemType = None
    ComplexityScore = None

try:
    from solution_assembler import SolutionAttempt as ExistingSolutionAttempt
    SOLUTION_ASSEMBLER_AVAILABLE = True
except ImportError:
    SOLUTION_ASSEMBLER_AVAILABLE = False
    ExistingSolutionAttempt = None


@dataclass
class AdaptiveSolverConfig:
    """Configuration for adaptive sub-problem solving."""
    enabled: bool = True
    enable_learning: bool = False
    enable_context_aware: bool = False
    default_profile: str = "balanced"
    fallback_on_failure: bool = True
    max_escalation_attempts: int = 3


class AdaptiveSubProblemSolver:
    """
    Enhanced SubProblemSolver with adaptive MDAP integration.
    
    This class wraps or extends the existing SubProblemSolver to add
    adaptive resource allocation capabilities.
    """
    
    def __init__(
        self,
        openevolve_client=None,
        config: Optional[AdaptiveSolverConfig] = None,
        classifier: Optional[TaskComplexityClassifier] = None,
        allocator: Optional[AdaptiveMDAPAllocator] = None,
        controller: Optional["AdaptiveExecutionController"] = None,
    ):
        """
        Initialize the adaptive sub-problem solver.
        
        Args:
            openevolve_client: OpenEvolve client for evolution operations
            config: Adaptive solver configuration
            classifier: Complexity classifier
            allocator: Resource allocator
            controller: Execution controller
        """
        self.openevolve_client = openevolve_client
        self.config = config or AdaptiveSolverConfig()
        
        # Initialize adaptive components
        self.classifier = classifier or TaskComplexityClassifier()
        self.allocator = allocator or AdaptiveMDAPAllocator(
            enable_learning=self.config.enable_learning,
            enable_context_aware=self.config.enable_context_aware,
        )
        self.controller = controller or AdaptiveExecutionController(
            classifier=self.classifier,
            allocator=self.allocator,
        )
        
        # Statistics
        self._stats = {
            "total_solves": 0,
            "adaptive_solves": 0,
            "fallback_solves": 0,
            "failed_solves": 0,
        }
        
        logger.info(f"Initialized AdaptiveSubProblemSolver (enabled={self.config.enabled})")
    
    def solve(
        self,
        sub_problem,
        strategy: Optional[str] = None,
        workflow_id: Optional[str] = None,
        force_adaptive: bool = False,
    ):
        """
        Solve a sub-problem with adaptive resource allocation.
        
        Args:
            sub_problem: SubProblem to solve (sovereign or adaptive type)
            strategy: Explicit strategy to use (bypasses adaptive if provided)
            workflow_id: Optional workflow ID for tracking
            force_adaptive: Force adaptive mode even if strategy provided
            
        Returns:
            SolutionAttempt with results
        """
        self._stats["total_solves"] += 1
        
        # Convert to adaptive sub-problem format if needed
        adaptive_sp = self._convert_subproblem(sub_problem)
        
        # Determine if we should use adaptive allocation
        use_adaptive = self.config.enabled and (
            force_adaptive or strategy is None
        )
        
        if use_adaptive:
            return self._solve_adaptive(
                adaptive_sp,
                workflow_id=workflow_id,
            )
        else:
            # Use explicit strategy
            solve_strategy = self._parse_strategy(strategy) if strategy else SolveStrategy.DIRECT
            return self._solve_with_strategy(
                adaptive_sp,
                solve_strategy,
                workflow_id=workflow_id,
            )
    
    def _solve_adaptive(
        self,
        subproblem: AdaptiveSubProblem,
        workflow_id: Optional[str] = None,
    ) -> "SolutionAttempt":
        """Solve using adaptive resource allocation."""
        self._stats["adaptive_solves"] += 1
        
        try:
            attempt = self.controller.execute_adaptive(
                subproblem=subproblem,
                workflow_id=workflow_id,
                enable_escalation=self.config.fallback_on_failure,
            )
            
            if attempt.status.value == "completed":
                logger.info(f"Adaptive solve succeeded for {subproblem.id}")
            else:
                logger.warning(f"Adaptive solve failed for {subproblem.id}: {attempt.error_message}")
                self._stats["failed_solves"] += 1
            
            return attempt
            
        except Exception as e:
            logger.exception(f"Adaptive solve error for {subproblem.id}: {e}")
            self._stats["failed_solves"] += 1
            
            if self.config.fallback_on_failure:
                return self._fallback_solve(subproblem, workflow_id)
            else:
                raise
    
    def _solve_with_strategy(
        self,
        subproblem: AdaptiveSubProblem,
        strategy: SolveStrategy,
        workflow_id: Optional[str] = None,
    ) -> "SolutionAttempt":
        """Solve with a specific strategy."""
        return self.controller.execute_adaptive(
            subproblem=subproblem,
            workflow_id=workflow_id,
            force_strategy=strategy,
            enable_escalation=False,
        )
    
    def _fallback_solve(
        self,
        subproblem: AdaptiveSubProblem,
        workflow_id: Optional[str] = None,
    ) -> "SolutionAttempt":
        """Fallback to direct solve when adaptive fails."""
        logger.info(f"Falling back to direct solve for {subproblem.id}")
        self._stats["fallback_solves"] += 1
        
        return self.controller.execute_adaptive(
            subproblem=subproblem,
            workflow_id=workflow_id,
            force_strategy=SolveStrategy.DIRECT,
            enable_escalation=False,
        )
    
    def _convert_subproblem(self, sub_problem) -> AdaptiveSubProblem:
        """
        Convert existing SubProblem to adaptive format.
        
        Handles both sovereign_data_models.SubProblem and adaptive_mdap.core.types.SubProblem.
        """
        if isinstance(sub_problem, AdaptiveSubProblem):
            return sub_problem
        
        # Convert from sovereign model
        if SOVEREIGN_MODELS_AVAILABLE and isinstance(sub_problem, SubProblem):
            return AdaptiveSubProblem(
                id=sub_problem.id,
                description=sub_problem.description or sub_problem.title or "",
                domain=self._infer_domain(sub_problem),
                depth=getattr(sub_problem, "depth", 0),
                dependencies=[dep.id for dep in getattr(sub_problem, "dependencies", [])],
                metadata={
                    "type": getattr(sub_problem.type, "value", str(sub_problem.type)) if hasattr(sub_problem, "type") else "unknown",
                    "parent_id": getattr(sub_problem, "parent_id", None),
                    "original_type": "sovereign",
                },
            )
        
        # Convert from dict
        if isinstance(sub_problem, dict):
            return AdaptiveSubProblem(
                id=sub_problem.get("id", str(uuid.uuid4())),
                description=sub_problem.get("description", sub_problem.get("title", "")),
                domain=sub_problem.get("domain", "general"),
                depth=sub_problem.get("depth", 0),
                dependencies=sub_problem.get("dependencies", []),
                metadata=sub_problem.get("metadata", {}),
            )
        
        # Fallback - try to extract attributes
        return AdaptiveSubProblem(
            id=getattr(sub_problem, "id", str(uuid.uuid4())),
            description=getattr(sub_problem, "description", getattr(sub_problem, "title", "")),
            domain=getattr(sub_problem, "domain", "general"),
            depth=getattr(sub_problem, "depth", 0),
            dependencies=list(getattr(sub_problem, "dependencies", [])),
            metadata={"original_type": type(sub_problem).__name__},
        )
    
    def _infer_domain(self, sub_problem) -> str:
        """Infer domain from sub-problem."""
        if hasattr(sub_problem, "type") and sub_problem.type:
            type_val = sub_problem.type
            if isinstance(type_val, Enum):
                return type_val.value.lower()
            return str(type_val).lower()
        
        if hasattr(sub_problem, "domain") and sub_problem.domain:
            return sub_problem.domain
        
        return "general"
    
    def _parse_strategy(self, strategy: str) -> SolveStrategy:
        """Parse strategy string to enum."""
        strategy_map = {
            "direct": SolveStrategy.DIRECT,
            "mdap_light": SolveStrategy.MDAP_LIGHT,
            "mdap_medium": SolveStrategy.MDAP_MEDIUM,
            "maker_full": SolveStrategy.MAKER_FULL,
            "maker_ultra": SolveStrategy.MAKER_ULTRA,
        }
        return strategy_map.get(strategy.lower(), SolveStrategy.DIRECT)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get solver statistics."""
        return {
            **self._stats,
            "allocator_stats": self.allocator.get_allocation_stats(),
            "controller_stats": self.controller.get_execution_stats(),
        }
    
    def update_historical_stats(self, domain: str, success: bool, complexity: float):
        """Update historical statistics for a domain."""
        self.classifier.update_historical_stats(domain, success, complexity)
    
    def get_complexity_distribution(self) -> Dict[str, int]:
        """Get distribution of complexity scores."""
        stats = self.allocator.get_allocation_stats()
        return stats.get("complexity_band_distribution", {})


# Convenience function for quick integration
def create_adaptive_solver(
    openevolve_client=None,
    enabled: bool = True,
    profile: str = "balanced",
) -> AdaptiveSubProblemSolver:
    """
    Create an adaptive sub-problem solver with default configuration.
    
    Args:
        openevolve_client: OpenEvolve client
        enabled: Whether adaptive allocation is enabled
        profile: Configuration profile (conservative, balanced, aggressive)
        
    Returns:
        Configured AdaptiveSubProblemSolver
    """
    from adaptive_mdap.config.profiles import ConfigProfile, get_profile_config
    
    profile_map = {
        "conservative": ConfigProfile.CONSERVATIVE,
        "balanced": ConfigProfile.BALANCED,
        "aggressive": ConfigProfile.AGGRESSIVE,
    }
    
    config = get_profile_config(profile_map.get(profile, ConfigProfile.BALANCED))
    
    adaptive_config = AdaptiveSolverConfig(
        enabled=enabled,
        enable_learning=config.get("allocator", {}).get("enable_learning", False),
        default_profile=profile,
    )
    
    return AdaptiveSubProblemSolver(
        openevolve_client=openevolve_client,
        config=adaptive_config,
    )
