"""
PES Executor for OpenEvolve integration.

Executes evolution plans using OpenEvolve core engine with
budget monitoring and adaptive parameter adjustment.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import logging
import time

from .config import UnifiedEvolutionConfig
from .adapter import PESCallbacks, PESOpenEvolveAdapter
from .budget_monitor import BudgetMonitor


logger = logging.getLogger(__name__)


@dataclass
class PhaseResult:
    """Result of executing a plan phase."""
    phase_name: str
    completed: bool
    iterations: int
    final_fitness: float
    budget_consumed: Dict[str, float]
    requires_adaptation: bool = False
    adaptation_reason: Optional[str] = None


@dataclass
class ExecutionState:
    """State of plan execution."""
    current_phase_idx: int = 0
    total_iterations: int = 0
    best_fitness: float = 0.0
    best_solution: Optional[Any] = None
    phase_results: List[PhaseResult] = field(default_factory=list)
    adaptations_made: List[Dict] = field(default_factory=list)
    termination_reason: Optional[str] = None


@dataclass
class ExecutionResult:
    """Complete execution result."""
    state: ExecutionState
    final_solution: Optional[Any]
    cost_summary: Dict[str, float]
    plan_followed: bool
    success: bool


class AdaptationEngine:
    """
    Adapts evolution parameters based on execution progress.
    
    Monitors fitness progression and budget consumption to
    suggest parameter adjustments during evolution.
    """
    
    def __init__(self, config: UnifiedEvolutionConfig):
        self.config = config
        self.adaptation_history = []
    
    def suggest_adjustment(
        self,
        iteration: int,
        current_fitness: float,
        fitness_history: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """
        Suggest parameter adjustment based on progress.
        
        Returns:
            Adjustment dict or None if no adjustment needed
        """
        # Check for stagnation
        if len(fitness_history) >= 10:
            recent = fitness_history[-10:]
            improvement = recent[-1]["best_fitness"] - recent[0]["best_fitness"]
            
            if improvement < 0.01:  # Stagnant
                return self._adapt_for_stagnation(iteration)
        
        # Check for rapid improvement
        if len(fitness_history) >= 5:
            recent = fitness_history[-5:]
            improvement = recent[-1]["best_fitness"] - recent[0]["best_fitness"]
            
            if improvement > 0.1:  # Rapid improvement
                return self._adapt_for_rapid_progress(iteration)
        
        return None
    
    def _adapt_for_stagnation(self, iteration: int) -> Dict[str, Any]:
        """Adapt parameters when evolution stagnates."""
        adjustment = {
            "iteration": iteration,
            "reason": "stagnation",
            "adjustments": {
                "mutation_rate": 1.3,  # Increase by 30%
                "exploration_bonus": 0.15,
                "diversity_maintenance": True
            }
        }
        self.adaptation_history.append(adjustment)
        logger.info(f"Adapting for stagnation at iteration {iteration}")
        return adjustment
    
    def _adapt_for_rapid_progress(self, iteration: int) -> Dict[str, Any]:
        """Adapt parameters when making rapid progress."""
        adjustment = {
            "iteration": iteration,
            "reason": "rapid_progress",
            "adjustments": {
                "mutation_rate": 0.8,  # Deccrease by 20%
                "selection_pressure": 1.5,
                "exploitation_focus": True
            }
        }
        self.adaptation_history.append(adjustment)
        logger.info(f"Adapting for rapid progress at iteration {iteration}")
        return adjustment
    
    def adapt_config(
        self,
        config: Dict[str, Any],
        adjustment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply adjustment to configuration."""
        new_config = config.copy()
        adjustments = adjustment.get("adjustments", {})
        
        for param, value in adjustments.items():
            if param in new_config:
                current = new_config[param]
                if isinstance(current, (int, float)) and isinstance(value, (int, float)):
                    if value < 2.0:  # Assume multiplier
                        new_config[param] = current * value
                    else:
                        new_config[param] = value
                else:
                    new_config[param] = value
        
        return new_config


class PESExecutor:
    """
    Executes evolution according to PES plan.
    
    Responsibilities:
    - Translate PES plan to OpenEvolve parameters
    - Monitor execution against budget
    - Adapt parameters based on intermediate results
    - Coordinate with OpenEvolve evolution engine
    """
    
    def __init__(
        self,
        adapter: PESOpenEvolveAdapter,
        budget_monitor: Optional[BudgetMonitor] = None,
        config: Optional[UnifiedEvolutionConfig] = None
    ):
        self.adapter = adapter
        self.budget_monitor = budget_monitor
        self.config = config or UnifiedEvolutionConfig()
        self.adaptation_engine = AdaptationEngine(self.config)
    
    async def execute_plan(
        self,
        plan: "EvolutionPlan",
        problem: "EvolutionProblem"
    ) -> Dict[str, Any]:
        """
        Execute PES plan for a problem.
        
        Flow:
        1. Configure OpenEvolve from plan
        2. Execute each phase with monitoring
        3. Adapt parameters between phases
        4. Return aggregated results
        """
        logger.info(f"Executing PES plan with {len(plan.phases)} phases")
        
        # Initialize execution state
        state = ExecutionState()
        current_config = self.adapter.translate_plan_to_config(plan)
        
        # Execute phases
        for phase_idx, phase in enumerate(plan.phases):
            logger.info(f"Executing phase: {phase.name}")
            state.current_phase_idx = phase_idx
            
            # Check budget before phase
            if self.budget_monitor and not self.budget_monitor.can_continue():
                logger.warning("Budget exhausted, stopping execution")
                state.termination_reason = "budget_exhausted"
                break
            
            # Apply phase-specific configuration
            phase_config = self._apply_phase_config(current_config, phase)
            
            # Execute phase
            phase_result = await self._execute_phase(
                phase,
                problem,
                phase_config,
                state
            )
            
            state.phase_results.append(phase_result)
            state.total_iterations += phase_result.iterations
            
            # Update best solution
            if phase_result.final_fitness > state.best_fitness:
                state.best_fitness = phase_result.final_fitness
            
            # Check if phase requires adaptation
            if phase_result.requires_adaptation:
                adjustment = self.adaptation_engine.suggest_adjustment(
                    state.total_iterations,
                    state.best_fitness,
                    [{"best_fitness": r.final_fitness} for r in state.phase_results]
                )
                
                if adjustment:
                    current_config = self.adaptation_engine.adapt_config(
                        current_config,
                        adjustment
                    )
                    state.adaptations_made.append(adjustment)
            
            # Check convergence
            if self._check_convergence(phase, phase_result):
                logger.info(f"Phase {phase.name} converged")
                if phase_idx < len(plan.phases) - 1:
                    continue  # Move to next phase
                else:
                    state.termination_reason = "converged"
                    break
        
        # Compile results
        cost_summary = self._compile_cost_summary()
        
        return {
            "success": state.best_fitness > 0.8,
            "best_solution": state.best_solution,
            "final_fitness": state.best_fitness,
            "iterations": state.total_iterations,
            "cost_summary": cost_summary,
            "plan": {
                "phases_executed": len(state.phase_results),
                "phases_planned": len(plan.phases),
                "adaptations": state.adaptations_made
            },
            "adaptations": state.adaptations_made,
            "state": state
        }
    
    async def _execute_phase(
        self,
        phase: "PlanPhase",
        problem: "EvolutionProblem",
        config: Dict[str, Any],
        state: ExecutionState
    ) -> PhaseResult:
        """Execute a single plan phase."""
        
        # Create callbacks for this phase
        callbacks = PESCallbacks(
            budget_monitor=self.budget_monitor,
            adaptation_engine=self.adaptation_engine
        )
        
        # Run evolution for this phase
        phase_iterations = self._calculate_phase_iterations(phase, config)
        config["max_iterations"] = phase_iterations
        
        # Create a temporary plan for this phase
        from .pes_planner import EvolutionPlan, EvolutionMode
        phase_plan = EvolutionPlan(
            recommended_mode=EvolutionMode(config.get("evolution_mode", "standard")),
            reasoning=f"Phase: {phase.name}",
            suggested_parameters=config,
            parameter_reasoning=phase.description,
            budget_allocation=None,  # Use overall budget
            phases=[phase],
            expected_iterations=phase_iterations,
            success_probability=0.8
        )
        
        # Execute
        start_time = time.time()
        result = await self.adapter.run_evolution(problem, phase_plan, callbacks)
        elapsed = time.time() - start_time
        
        # Calculate budget consumed
        budget_consumed = result.get("cost_summary", {})
        if self.budget_monitor:
            budget_consumed = self.budget_monitor.get_summary()["consumed"]
        
        # Determine if adaptation needed
        requires_adaptation = self._check_phase_needs_adaptation(
            phase, result, callbacks.iteration_history
        )
        
        return PhaseResult(
            phase_name=phase.name,
            completed=result.get("success", False),
            iterations=result.get("iterations", 0),
            final_fitness=result.get("final_fitness", 0.0),
            budget_consumed=budget_consumed,
            requires_adaptation=requires_adaptation,
            adaptation_reason="stagnation" if requires_adaptation else None
        )
    
    def _apply_phase_config(
        self,
        base_config: Dict[str, Any],
        phase: "PlanPhase"
    ) -> Dict[str, Any]:
        """Apply phase-specific configuration."""
        config = base_config.copy()
        
        # Override with phase parameters
        for param, value in phase.suggested_parameters.items():
            config[param] = value
        
        return config
    
    def _calculate_phase_iterations(
        self,
        phase: "PlanPhase",
        config: Dict[str, Any]
    ) -> int:
        """Calculate iterations for this phase."""
        total_iterations = config.get("max_iterations", 100)
        
        # Allocate based on phase budget ratio
        return int(total_iterations * phase.budget_allocation)
    
    def _check_phase_needs_adaptation(
        self,
        phase: "PlanPhase",
        result: Dict[str, Any],
        iteration_history: List[Dict]
    ) -> bool:
        """Check if phase requires parameter adaptation."""
        # Check fitness improvement
        if len(iteration_history) >= 10:
            recent = iteration_history[-10:]
            improvement = recent[-1].get("best_fitness", 0) - recent[0].get("best_fitness", 0)
            
            if improvement < phase.convergence_criteria.get("fitness_improvement", 0.01):
                return True
        
        return False
    
    def _check_convergence(
        self,
        phase: "PlanPhase",
        phase_result: PhaseResult
    ) -> bool:
        """Check if phase has converged."""
        criteria = phase.convergence_criteria
        
        if "fitness_improvement" in criteria:
            # Would need history to check properly
            return phase_result.final_fitness > 0.9
        
        if "min_diversity" in criteria:
            # Would need diversity metric
            return phase_result.completed
        
        if "plateau_detection" in criteria:
            return phase_result.final_fitness > 0.95
        
        return phase_result.completed
    
    def _compile_cost_summary(self) -> Dict[str, float]:
        """Compile cost summary from budget monitor."""
        if self.budget_monitor:
            return self.budget_monitor.get_summary()
        
        return {"estimated": True, "cost_usd": 0.0}
