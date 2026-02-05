"""Workflow PES Adapter - Integrates PES Enhanced with Workflow Engine.

This module provides a non-invasive adapter that enables the Workflow Engine
to use PES Enhanced for cost-aware evolution and workflow execution.

The adapter wraps around the existing workflow execution without modifying
workflow_engine.py, allowing users to opt-in to PES Enhanced capabilities.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from enum import Enum
import asyncio

# Import Workflow Engine structures
try:
    from workflow_structures import (
        WorkflowState, SubProblem, DecompositionPlan, 
        Team, GauntletDefinition, SolutionAttempt
    )
    WORKFLOW_STRUCTURES_AVAILABLE = True
except ImportError:
    WORKFLOW_STRUCTURES_AVAILABLE = False
    # Define minimal fallback classes
    @dataclass
    class SubProblem:
        id: str
        description: str
        dependencies: List[str] = field(default_factory=list)
        ai_suggested_complexity_score: int = 5
        metadata: Dict[str, Any] = field(default_factory=dict)
    
    @dataclass  
    class WorkflowState:
        workflow_id: str
        problem_statement: str
        current_stage: str = "initializing"
        status: str = "running"
        decomposition_plan: Optional[Any] = None
        sub_problem_solutions: Dict[str, Any] = field(default_factory=dict)
        solved_sub_problem_ids: Set[str] = field(default_factory=set)
        metadata: Dict[str, Any] = field(default_factory=dict)

# Import PES Enhanced components
from .config import PESEnhancedConfig
from .integration_wrapper import PESIntegrationWrapper, EnhancedEvolutionResult
from .cost_optimizer import BudgetTracker, CostOptimizer, BudgetStatus
from .execution_monitor import ExecutionMonitor, EarlyStoppingController

logger = logging.getLogger(__name__)


class AllocationDecision(Enum):
    """Decision for subproblem resource allocation."""
    FULL_EVOLUTION = "full_evolution"
    LIMITED_EVOLUTION = "limited_evolution"
    SKIP_EVOLUTION = "skip_evolution"
    USE_CACHED = "use_cached"
    DEFER = "defer"


@dataclass
class SubProblemAllocation:
    """Allocation decision for a specific subproblem."""
    sub_problem_id: str
    decision: AllocationDecision
    budget_usd: float
    max_iterations: int
    priority: int  # Lower is higher priority
    estimated_cost: float
    estimated_time_seconds: float
    reason: str


@dataclass
class WorkflowCostMetrics:
    """Cost metrics for a workflow execution."""
    total_cost_usd: float = 0.0
    total_tokens_used: int = 0
    total_time_seconds: float = 0.0
    
    # Breakdown by stage
    decomposition_cost: float = 0.0
    solution_generation_cost: float = 0.0
    verification_cost: float = 0.0
    
    # Per-subproblem costs
    subproblem_costs: Dict[str, float] = field(default_factory=dict)
    
    # Efficiency metrics
    evaluations_saved: int = 0
    efficiency_gain: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cost_usd": self.total_cost_usd,
            "total_tokens_used": self.total_tokens_used,
            "total_time_seconds": self.total_time_seconds,
            "decomposition_cost": self.decomposition_cost,
            "solution_generation_cost": self.solution_generation_cost,
            "verification_cost": self.verification_cost,
            "subproblem_costs": self.subproblem_costs,
            "evaluations_saved": self.evaluations_saved,
            "efficiency_gain": self.efficiency_gain,
        }


class CostAwareWorkflowTracker:
    """Tracks costs throughout workflow execution.
    
    This is a wrapper around BudgetTracker that provides workflow-specific
cost tracking and budget enforcement.
    """
    
    def __init__(self, max_cost_usd: Optional[float] = None):
        self.max_cost_usd = max_cost_usd
        self.budget_tracker = BudgetTracker(max_cost_usd=max_cost_usd) if max_cost_usd else None
        self.metrics = WorkflowCostMetrics()
        self._start_time = time.time()
        self._stage_start_times: Dict[str, float] = {}
        self._stage_costs: Dict[str, float] = {}
        
    def start_stage(self, stage_name: str):
        """Mark the start of a workflow stage."""
        self._stage_start_times[stage_name] = time.time()
        self._stage_costs[stage_name] = 0.0
        
    def end_stage(self, stage_name: str):
        """Mark the end of a workflow stage."""
        if stage_name in self._stage_start_times:
            elapsed = time.time() - self._stage_start_times[stage_name]
            self.metrics.total_time_seconds += elapsed
            
    def record_cost(self, cost_usd: float, stage: Optional[str] = None, 
                   subproblem_id: Optional[str] = None):
        """Record a cost expenditure."""
        self.metrics.total_cost_usd += cost_usd
        
        if stage:
            if stage not in self._stage_costs:
                self._stage_costs[stage] = 0.0
            self._stage_costs[stage] += cost_usd
            
            # Update specific stage metrics
            if stage == "decomposition":
                self.metrics.decomposition_cost += cost_usd
            elif stage == "solution_generation":
                self.metrics.solution_generation_cost += cost_usd
            elif stage == "verification":
                self.metrics.verification_cost += cost_usd
        
        if subproblem_id:
            if subproblem_id not in self.metrics.subproblem_costs:
                self.metrics.subproblem_costs[subproblem_id] = 0.0
            self.metrics.subproblem_costs[subproblem_id] += cost_usd
            
        # Update budget tracker if available
        if self.budget_tracker:
            # Estimate tokens from cost (rough approximation)
            estimated_tokens = int(cost_usd / 0.00003)
            self.budget_tracker.record_tokens(0, estimated_tokens)
            
    def check_budget(self) -> Tuple[bool, Optional[str]]:
        """Check if we should continue based on budget.
        
        Returns:
            (should_continue, reason_if_stopping)
        """
        if not self.max_cost_usd:
            return True, None
        
        # Check cost-based budget
        if self.max_cost_usd > 0:
            cost_pct = self.metrics.total_cost_usd / self.max_cost_usd
            if cost_pct >= 1.0:
                return False, f"exceeded: ${self.metrics.total_cost_usd:.2f} used"
            if cost_pct >= 0.9:
                return False, f"critical: ${self.metrics.total_cost_usd:.2f} used ({cost_pct:.0%})"
        
        # Also check time-based budget
        if self.budget_tracker:
            elapsed_ms = (time.time() * 1000) - self.budget_tracker.start_time
            if self.budget_tracker.max_time_ms > 0:
                time_pct = elapsed_ms / self.budget_tracker.max_time_ms
                if time_pct >= 1.0:
                    return False, f"time exceeded: {elapsed_ms/1000:.0f}s elapsed"
        
        return True, None
        
    def get_status(self) -> Dict[str, Any]:
        """Get current cost status."""
        elapsed = time.time() - self._start_time
        
        # Calculate budget status
        should_stop = False
        budget_status = "ok"
        if self.max_cost_usd and self.max_cost_usd > 0:
            cost_pct = self.metrics.total_cost_usd / self.max_cost_usd
            if cost_pct >= 1.0:
                budget_status = "exceeded"
                should_stop = True
            elif cost_pct >= 0.9:
                budget_status = "critical"
                should_stop = True
            elif cost_pct >= 0.7:
                budget_status = "warning"
        
        status = {
            "total_cost_usd": self.metrics.total_cost_usd,
            "total_time_seconds": elapsed,
            "budget_remaining": self.max_cost_usd - self.metrics.total_cost_usd if self.max_cost_usd else None,
            "budget_pct_used": (self.metrics.total_cost_usd / self.max_cost_usd * 100) if self.max_cost_usd else None,
            "stage_breakdown": self._stage_costs.copy(),
            "budget_status": budget_status,
            "should_stop": should_stop,
        }
        
        return status


class WorkflowPESAdapter:
    """Adapter to integrate PES Enhanced with Workflow Engine.
    
    This adapter provides:
    1. Cost-aware decomposition planning
    2. Budget-aware subproblem allocation
    3. Cost tracking across workflow stages
    4. Budget enforcement and early stopping
    
    Usage:
        adapter = WorkflowPESAdapter(pes_config)
        result = await adapter.execute_workflow_with_pes(workflow_state, max_cost_usd=10.0)
    """
    
    def __init__(self, pes_config: Optional[PESEnhancedConfig] = None):
        """Initialize the adapter.
        
        Args:
            pes_config: PES Enhanced configuration. If None, uses cost-aware defaults.
        """
        self.config = pes_config or PESEnhancedConfig.cost_aware(max_cost_usd=10.0)
        self.pes_wrapper = PESIntegrationWrapper(self.config)
        self.cost_tracker: Optional[CostAwareWorkflowTracker] = None
        
    def initialize_budget(self, max_cost_usd: Optional[float]):
        """Initialize cost tracking for a workflow run."""
        self.cost_tracker = CostAwareWorkflowTracker(max_cost_usd=max_cost_usd)
        if max_cost_usd:
            logger.info(f"Workflow budget initialized: ${max_cost_usd:.2f}")
            
    def check_and_enforce_budget(self) -> Tuple[bool, Optional[str]]:
        """Check budget status and return whether to continue."""
        if not self.cost_tracker:
            return True, None
        return self.cost_tracker.check_budget()
        
    def enhance_decomposition_with_pes(
        self,
        subproblems: List[SubProblem],
        budget_per_problem: float
    ) -> List[Tuple[SubProblem, SubProblemAllocation]]:
        """Apply PES cost-aware allocation to subproblems.
        
        This method analyzes the subproblems and determines:
        - Which subproblems should get full evolution resources
        - Which should get limited resources
        - Which can be skipped or deferred
        - The priority order for processing
        
        Args:
            subproblems: List of subproblems from decomposition
            budget_per_problem: Budget allocated per subproblem
            
        Returns:
            List of (subproblem, allocation_decision) tuples
        """
        allocations = []
        
        # Sort by complexity (highest first for priority)
        sorted_problems = sorted(
            subproblems,
            key=lambda sp: getattr(sp, 'ai_suggested_complexity_score', 5),
            reverse=True
        )
        
        for i, sp in enumerate(sorted_problems):
            complexity = getattr(sp, 'ai_suggested_complexity_score', 5)
            dependencies = getattr(sp, 'dependencies', [])
            
            # Determine allocation based on complexity and budget
            if complexity >= 8:
                decision = AllocationDecision.FULL_EVOLUTION
                budget = budget_per_problem * 1.5
                max_iter = 100
                reason = "High complexity - full resources allocated"
            elif complexity >= 5:
                decision = AllocationDecision.FULL_EVOLUTION
                budget = budget_per_problem
                max_iter = 75
                reason = "Medium complexity - standard resources"
            elif complexity >= 3:
                decision = AllocationDecision.LIMITED_EVOLUTION
                budget = budget_per_problem * 0.5
                max_iter = 50
                reason = "Low complexity - limited resources"
            else:
                decision = AllocationDecision.LIMITED_EVOLUTION
                budget = budget_per_problem * 0.25
                max_iter = 25
                reason = "Very low complexity - minimal resources"
            
            # Check if we have enough budget remaining
            should_continue, stop_reason = self.check_and_enforce_budget()
            if not should_continue:
                decision = AllocationDecision.DEFER
                budget = 0.0
                max_iter = 0
                reason = f"Budget constraint: {stop_reason}"
            
            allocation = SubProblemAllocation(
                sub_problem_id=sp.id if hasattr(sp, 'id') else str(i),
                decision=decision,
                budget_usd=budget,
                max_iterations=max_iter,
                priority=i + 1,
                estimated_cost=budget * 0.8,  # 80% utilization estimate
                estimated_time_seconds=max_iter * 2,  # Rough estimate
                reason=reason
            )
            
            allocations.append((sp, allocation))
            
            # Record estimated cost
            if self.cost_tracker:
                self.cost_tracker.record_cost(
                    allocation.estimated_cost,
                    stage="decomposition",
                    subproblem_id=sp.id if hasattr(sp, 'id') else str(i)
                )
        
        logger.info(f"Allocated resources for {len(allocations)} subproblems")
        return allocations
    
    async def execute_workflow_stage(
        self,
        stage_name: str,
        stage_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute a workflow stage with cost tracking.
        
        Args:
            stage_name: Name of the stage (for tracking)
            stage_func: Function to execute
            *args, **kwargs: Arguments to pass to stage_func
            
        Returns:
            Result from stage_func
        """
        if self.cost_tracker:
            self.cost_tracker.start_stage(stage_name)
            
            # Check budget before starting
            should_continue, stop_reason = self.check_and_enforce_budget()
            if not should_continue:
                logger.warning(f"Budget exceeded before stage {stage_name}: {stop_reason}")
                raise BudgetExceededError(stop_reason)
        
        try:
            start_time = time.time()
            
            # Execute the stage
            if asyncio.iscoroutinefunction(stage_func):
                result = await stage_func(*args, **kwargs)
            else:
                result = stage_func(*args, **kwargs)
            
            # Estimate cost based on time (rough heuristic)
            elapsed = time.time() - start_time
            estimated_cost = self._estimate_cost_from_time(elapsed)
            
            if self.cost_tracker:
                self.cost_tracker.record_cost(estimated_cost, stage=stage_name)
                self.cost_tracker.end_stage(stage_name)
            
            return result
            
        except Exception as e:
            if self.cost_tracker:
                self.cost_tracker.end_stage(stage_name)
            raise
    
    def _estimate_cost_from_time(self, elapsed_seconds: float) -> float:
        """Estimate cost from execution time (rough heuristic)."""
        # Assume $0.01 per second of LLM time on average
        return elapsed_seconds * 0.01
    
    async def execute_workflow_with_pes(
        self,
        workflow_state: WorkflowState,
        max_cost_usd: Optional[float] = None,
        enable_cost_tracking: bool = True,
        original_workflow_func: Optional[Callable] = None,
        **workflow_kwargs
    ) -> WorkflowState:
        """Execute workflow with PES Enhanced cost tracking.
        
        This wraps the workflow execution with PES monitoring, cost tracking,
        and budget enforcement.
        
        Args:
            workflow_state: The workflow state to execute
            max_cost_usd: Maximum budget for this workflow
            enable_cost_tracking: Whether to track costs
            original_workflow_func: Optional original workflow function to wrap
            **workflow_kwargs: Additional arguments for workflow function
            
        Returns:
            Updated workflow state with cost metrics in metadata
        """
        # Initialize budget tracking
        if enable_cost_tracking and max_cost_usd:
            self.initialize_budget(max_cost_usd)
        
        # Enhance decomposition plan if available
        if (hasattr(workflow_state, 'decomposition_plan') and 
            workflow_state.decomposition_plan and
            hasattr(workflow_state.decomposition_plan, 'sub_problems')):
            
            subproblems = workflow_state.decomposition_plan.sub_problems
            if subproblems and max_cost_usd:
                budget_per_problem = max_cost_usd / max(len(subproblems), 1)
                allocations = self.enhance_decomposition_with_pes(
                    subproblems, budget_per_problem
                )
                
                # Store allocations in workflow metadata
                if not hasattr(workflow_state, 'metadata'):
                    workflow_state.metadata = {}
                workflow_state.metadata['pes_allocations'] = [
                    {
                        'sub_problem_id': alloc.sub_problem_id,
                        'decision': alloc.decision.value,
                        'budget_usd': alloc.budget_usd,
                        'max_iterations': alloc.max_iterations,
                        'priority': alloc.priority,
                        'reason': alloc.reason
                    }
                    for sp, alloc in allocations
                ]
        
        # Execute the workflow
        start_time = time.time()
        
        try:
            if original_workflow_func:
                # Wrap the original workflow function
                if asyncio.iscoroutinefunction(original_workflow_func):
                    result = await original_workflow_func(
                        workflow_state, **workflow_kwargs
                    )
                else:
                    result = original_workflow_func(
                        workflow_state, **workflow_kwargs
                    )
            else:
                # Return the state as-is (user will run workflow separately)
                result = workflow_state
            
            # Record final metrics
            if self.cost_tracker:
                elapsed = time.time() - start_time
                self.cost_tracker.metrics.total_time_seconds = elapsed
                
                # Add cost metrics to result metadata
                if hasattr(result, 'metadata'):
                    result.metadata['pes_cost_metrics'] = self.cost_tracker.metrics.to_dict()
                
                logger.info(
                    f"Workflow completed: ${self.cost_tracker.metrics.total_cost_usd:.2f}, "
                    f"{elapsed:.1f}s"
                )
            
            return result
            
        except BudgetExceededError as e:
            logger.error(f"Workflow stopped due to budget: {e}")
            workflow_state.status = "failed_budget_exceeded"
            if hasattr(workflow_state, 'metadata'):
                workflow_state.metadata['budget_error'] = str(e)
                if self.cost_tracker:
                    workflow_state.metadata['pes_cost_metrics'] = self.cost_tracker.metrics.to_dict()
            return workflow_state
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            if self.cost_tracker and hasattr(workflow_state, 'metadata'):
                workflow_state.metadata['pes_cost_metrics'] = self.cost_tracker.metrics.to_dict()
            raise


class BudgetExceededError(Exception):
    """Raised when workflow budget is exceeded."""
    pass


# ============================================================================
# Convenience Functions
# ============================================================================

async def run_sovereign_workflow_with_pes(
    workflow_state: WorkflowState,
    content_analyzer_team: Any,
    planner_team: Any,
    solver_team: Any,
    patcher_team: Any,
    assembler_team: Any,
    sub_problem_red_gauntlet: Any,
    sub_problem_gold_gauntlet: Any,
    final_red_gauntlet: Any,
    final_gold_gauntlet: Any,
    solver_generation_gauntlet: Any,
    max_refinement_loops: int = 3,
    pes_config: Optional[PESEnhancedConfig] = None,
    max_cost_usd: Optional[float] = None,
    enable_cost_tracking: bool = True,
) -> WorkflowState:
    """Enhanced version of run_sovereign_workflow with PES cost tracking.
    
    This is a drop-in replacement for run_sovereign_workflow that adds
    PES Enhanced cost tracking and budget management.
    
    Args:
        workflow_state: The workflow state
        content_analyzer_team: Team for content analysis
        planner_team: Team for decomposition planning
        solver_team: Team for solving subproblems
        patcher_team: Team for patching rejected solutions
        assembler_team: Team for final assembly
        sub_problem_red_gauntlet: Red gauntlet for subproblems
        sub_problem_gold_gauntlet: Gold gauntlet for subproblems
        final_red_gauntlet: Red gauntlet for final solution
        final_gold_gauntlet: Gold gauntlet for final solution
        solver_generation_gauntlet: Gauntlet for solution generation
        max_refinement_loops: Maximum refinement loops
        pes_config: PES Enhanced configuration
        max_cost_usd: Maximum budget in USD
        enable_cost_tracking: Whether to enable cost tracking
        
    Returns:
        Updated workflow state with cost metrics
    """
    # Try to import the original workflow function
    try:
        from workflow_engine import run_sovereign_workflow
    except ImportError:
        raise ImportError(
            "workflow_engine.run_sovereign_workflow not available. "
            "Ensure workflow_engine.py is in the Python path."
        )
    
    # Create adapter
    adapter = WorkflowPESAdapter(pes_config)
    
    # Execute with PES tracking
    return await adapter.execute_workflow_with_pes(
        workflow_state=workflow_state,
        max_cost_usd=max_cost_usd,
        enable_cost_tracking=enable_cost_tracking,
        original_workflow_func=run_sovereign_workflow,
        content_analyzer_team=content_analyzer_team,
        planner_team=planner_team,
        solver_team=solver_team,
        patcher_team=patcher_team,
        assembler_team=assembler_team,
        sub_problem_red_gauntlet=sub_problem_red_gauntlet,
        sub_problem_gold_gauntlet=sub_problem_gold_gauntlet,
        final_red_gauntlet=final_red_gauntlet,
        final_gold_gauntlet=final_gold_gauntlet,
        solver_generation_gauntlet=solver_generation_gauntlet,
        max_refinement_loops=max_refinement_loops,
    )


def create_cost_aware_workflow_config(
    max_cost_usd: float = 10.0,
    enable_early_stopping: bool = True,
    enable_cost_optimization: bool = True
) -> PESEnhancedConfig:
    """Create a PES config optimized for workflow cost tracking.
    
    Args:
        max_cost_usd: Maximum budget in USD
        enable_early_stopping: Whether to enable early stopping
        enable_cost_optimization: Whether to enable cost optimization
        
    Returns:
        Configured PESEnhancedConfig
    """
    config = PESEnhancedConfig.cost_aware(max_cost_usd=max_cost_usd)
    config.enable_early_stopping = enable_early_stopping
    config.enable_cost_optimization = enable_cost_optimization
    config.enable_planning = True
    config.enable_summarization = True
    
    # Adjust sub-configs for workflow use
    config.cost.max_time_seconds = 3600  # 1 hour max
    config.early_stopping.max_duration_ms = 3600000
    config.early_stopping.patience = 3  # Faster convergence
    
    return config


# ============================================================================
# Extensions to WorkflowState for PES tracking
# ============================================================================

class WorkflowStatePESExtension:
    """Mixin class to add PES tracking to WorkflowState.
    
    Usage:
        # Extend workflow state with PES capabilities
        WorkflowStatePESExtension.extend(workflow_state, pes_config)
        
        # Access PES data
        workflow_state.get_pes_config()
        workflow_state.get_cost_tracker()
    """
    
    PES_CONFIG_KEY = "_pes_config"
    COST_TRACKER_KEY = "_cost_tracker"
    BUDGET_REMAINING_KEY = "_budget_remaining"
    
    @classmethod
    def extend(cls, workflow_state: WorkflowState, 
               pes_config: Optional[PESEnhancedConfig] = None):
        """Add PES tracking capabilities to a WorkflowState.
        
        This is non-invasive - it stores data in the workflow_state's
        metadata dict without modifying the class definition.
        """
        if not hasattr(workflow_state, 'metadata'):
            workflow_state.metadata = {}
        
        workflow_state.metadata[cls.PES_CONFIG_KEY] = pes_config
        
        if pes_config and pes_config.cost.max_cost_usd:
            workflow_state.metadata[cls.BUDGET_REMAINING_KEY] = pes_config.cost.max_cost_usd
            
    @classmethod
    def get_pes_config(cls, workflow_state: WorkflowState) -> Optional[PESEnhancedConfig]:
        """Get PES config from workflow state."""
        if hasattr(workflow_state, 'metadata'):
            return workflow_state.metadata.get(cls.PES_CONFIG_KEY)
        return None
    
    @classmethod
    def get_budget_remaining(cls, workflow_state: WorkflowState) -> Optional[float]:
        """Get remaining budget from workflow state."""
        if hasattr(workflow_state, 'metadata'):
            return workflow_state.metadata.get(cls.BUDGET_REMAINING_KEY)
        return None
    
    @classmethod
    def update_budget(cls, workflow_state: WorkflowState, cost_incurred: float):
        """Update remaining budget after incurring a cost."""
        if hasattr(workflow_state, 'metadata'):
            current = workflow_state.metadata.get(cls.BUDGET_REMAINING_KEY)
            if current is not None:
                workflow_state.metadata[cls.BUDGET_REMAINING_KEY] = max(0, current - cost_incurred)


# Export public API
__all__ = [
    # Main adapter
    "WorkflowPESAdapter",
    "run_sovereign_workflow_with_pes",
    
    # Cost tracking
    "CostAwareWorkflowTracker",
    "WorkflowCostMetrics",
    "BudgetExceededError",
    
    # Allocation
    "AllocationDecision",
    "SubProblemAllocation",
    
    # Configuration
    "create_cost_aware_workflow_config",
    "WorkflowStatePESExtension",
]
