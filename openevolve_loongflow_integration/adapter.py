"""
Adapter between PES planning layer and OpenEvolve evolution engine.

Translates PES plans to OpenEvolve configurations and coordinates execution.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable
import logging

from .config import UnifiedEvolutionConfig


logger = logging.getLogger(__name__)


@dataclass
class OpenEvolveResult:
    """Result from OpenEvolve evolution."""
    success: bool
    best_solution: Optional[Any]
    final_fitness: float
    iterations: int
    population: Optional[List] = None
    archive: Optional[Dict] = None  # For MAP-Elites
    pareto_front: Optional[List] = None  # For NSGA-II
    metadata: Optional[Dict] = None


class PESCallbacks:
    """Callbacks from OpenEvolve to PES layer during evolution."""
    
    def __init__(self, budget_monitor=None, adaptation_engine=None):
        self.budget_monitor = budget_monitor
        self.adaptation_engine = adaptation_engine
        self.iteration_history = []
    
    def on_iteration_complete(
        self,
        iteration: int,
        population: List,
        best_fitness: float,
        cost_breakdown: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Called after each evolution iteration.
        
        Returns:
            Parameter adjustments or None
        """
        from .budget_monitor import CostBreakdown
        
        # Record history
        self.iteration_history.append({
            "iteration": iteration,
            "best_fitness": best_fitness,
            "timestamp": logging.time.time() if hasattr(logging, 'time') else 0
        })
        
        # Update budget monitor
        if self.budget_monitor:
            cost = CostBreakdown(
                tokens_input=cost_breakdown.get("tokens_input", 0),
                tokens_output=cost_breakdown.get("tokens_output", 0),
                api_calls=cost_breakdown.get("api_calls", 0),
                compute_time_seconds=cost_breakdown.get("time_seconds", 0)
            )
            self.budget_monitor.record_spending(cost)
            
            # Check if we should stop
            if not self.budget_monitor.can_continue():
                return {"should_stop": True, "reason": "budget_exhausted"}
        
        # Check for adaptation
        if self.adaptation_engine and iteration % 10 == 0:
            return self.adaptation_engine.suggest_adjustment(
                iteration, best_fitness, self.iteration_history
            )
        
        return None
    
    def on_evaluation_complete(self, individual: Any, fitness: float):
        """Called after each individual evaluation."""
        pass
    
    def on_verification_complete(self, individual: Any, verified: bool):
        """Called after formal verification."""
        pass


class PESOpenEvolveAdapter:
    """
    Adapter between PES planning layer and OpenEvolve evolution engine.
    
    Responsibilities:
    - Translate PES plans to OpenEvolve configurations
    - Route evolution calls to appropriate engines
    - Inject PES guidance into evolution process
    - Collect results for PES summarization
    """
    
    def __init__(self, config: UnifiedEvolutionConfig):
        self.config = config
        self._engines = {}
    
    def _get_engine(self, mode: str):
        """Get or create evolution engine for mode."""
        if mode not in self._engines:
            # This would initialize actual OpenEvolve engines
            # For now, we return a placeholder
            self._engines[mode] = self._create_engine(mode)
        return self._engines[mode]
    
    def _create_engine(self, mode: str):
        """Create an evolution engine for the given mode."""
        # In actual implementation, this would return:
        # - MapElitesEngine for "qd"
        # - NSGA2Engine for "mo"
        # - StandardEvolutionEngine for "standard"
        # - LanguageAgnosticEngine for language-agnostic
        logger.info(f"Creating {mode} evolution engine")
        return MockEvolutionEngine(mode, self.config)
    
    def translate_plan_to_config(
        self,
        plan: "EvolutionPlan"
    ) -> Dict[str, Any]:
        """Translate PES plan to OpenEvolve configuration dictionary."""
        config = self.config.to_openevolve_config()
        
        # Apply suggested parameters from plan
        for param_name, value in plan.suggested_parameters.items():
            config[param_name] = value
        
        # Set mode
        config["evolution_mode"] = plan.recommended_mode.value
        
        # Set budget constraints
        config["cost_limit_usd"] = plan.budget_allocation.evolution_budget.max_cost
        config["token_limit"] = plan.budget_allocation.evolution_budget.max_tokens
        config["api_call_limit"] = plan.budget_allocation.evolution_budget.max_api_calls
        config["max_time"] = plan.budget_allocation.evolution_budget.max_time_seconds
        
        # Enable callbacks for PES integration
        config["enable_pes_callbacks"] = True
        
        return config
    
    async def run_evolution(
        self,
        problem: "EvolutionProblem",
        plan: "EvolutionPlan",
        callbacks: Optional[PESCallbacks] = None
    ) -> Dict[str, Any]:
        """Run evolution with PES guidance."""
        
        # Translate plan to OpenEvolve config
        config = self.translate_plan_to_config(plan)
        
        # Get appropriate engine
        mode = plan.recommended_mode.value
        engine = self._get_engine(mode)
        
        # Run evolution
        logger.info(f"Running {mode} evolution with PES guidance")
        result = await engine.evolve(problem, config, callbacks)
        
        return {
            "success": result.success,
            "best_solution": result.best_solution,
            "final_fitness": result.final_fitness,
            "iterations": result.iterations,
            "cost_summary": self._extract_cost_summary(result),
            "plan": plan,
            "metadata": result.metadata
        }
    
    async def run_standard_evolution(
        self,
        problem: "EvolutionProblem"
    ) -> Dict[str, Any]:
        """Run standard OpenEvolve evolution without PES."""
        config = self.config.to_openevolve_config()
        engine = self._get_engine("standard")
        
        result = await engine.evolve(problem, config, None)
        
        return {
            "success": result.success,
            "best_solution": result.best_solution,
            "final_fitness": result.final_fitness,
            "iterations": result.iterations,
            "cost_summary": self._extract_cost_summary(result),
            "plan": None
        }
    
    async def run_qd_evolution(
        self,
        problem: "EvolutionProblem"
    ) -> Dict[str, Any]:
        """Run Quality Diversity (MAP-Elites) evolution."""
        config = self.config.to_openevolve_config()
        config["evolution_mode"] = "qd"
        engine = self._get_engine("qd")
        
        result = await engine.evolve(problem, config, None)
        
        return {
            "success": result.success,
            "best_solution": result.best_solution,
            "final_fitness": result.final_fitness,
            "iterations": result.iterations,
            "archive": result.archive,
            "cost_summary": self._extract_cost_summary(result)
        }
    
    async def run_mo_evolution(
        self,
        problem: "EvolutionProblem"
    ) -> Dict[str, Any]:
        """Run Multi-Objective (NSGA-II) evolution."""
        config = self.config.to_openevolve_config()
        config["evolution_mode"] = "mo"
        engine = self._get_engine("mo")
        
        result = await engine.evolve(problem, config, None)
        
        return {
            "success": result.success,
            "best_solution": result.best_solution,
            "final_fitness": result.final_fitness,
            "iterations": result.iterations,
            "pareto_front": result.pareto_front,
            "cost_summary": self._extract_cost_summary(result)
        }
    
    def _extract_cost_summary(self, result: OpenEvolveResult) -> Dict[str, float]:
        """Extract cost summary from result."""
        if result.metadata and "cost" in result.metadata:
            return result.metadata["cost"]
        return {"estimated": True, "cost_usd": 0.0}


class MockEvolutionEngine:
    """
    Mock evolution engine for demonstration.
    
    In actual implementation, this would wrap OpenEvolve's
    real evolution engines.
    """
    
    def __init__(self, mode: str, config: UnifiedEvolutionConfig):
        self.mode = mode
        self.config = config
    
    async def evolve(
        self,
        problem: "EvolutionProblem",
        config: Dict[str, Any],
        callbacks: Optional[PESCallbacks]
    ) -> OpenEvolveResult:
        """Simulate evolution."""
        import asyncio
        import random
        
        iterations = config.get("max_iterations", 100)
        population_size = config.get("population_size", 50)
        
        best_fitness = 0.5
        
        for i in range(iterations):
            # Simulate iteration
            await asyncio.sleep(0.01)
            
            # Simulate improvement
            best_fitness = min(1.0, best_fitness + random.uniform(0, 0.01))
            
            # Call callbacks if provided
            if callbacks:
                adjustment = callbacks.on_iteration_complete(
                    i,
                    [],  # population
                    best_fitness,
                    {"tokens_input": 100, "tokens_output": 50, "api_calls": 1, "time_seconds": 0.01}
                )
                
                if adjustment and adjustment.get("should_stop"):
                    logger.info(f"Early stopping at iteration {i}")
                    break
        
        return OpenEvolveResult(
            success=best_fitness > 0.8,
            best_solution={"fitness": best_fitness, "code": "# evolved code"},
            final_fitness=best_fitness,
            iterations=iterations,
            metadata={"cost": {"cost_usd": iterations * 0.01}}
        )


class DirectedEvolutionStrategy:
    """
    Directs evolution using PES planning insights.
    
    Instead of blind mutation, uses plan guidance to:
    - Direct mutations toward promising areas
    - Skip unpromising search regions
    - Adapt mutation rates based on progress
    """
    
    def __init__(self, evolution_plan: "EvolutionPlan"):
        self.plan = evolution_plan
        self.current_phase_idx = 0
    
    def get_current_phase(self) -> Optional["PlanPhase"]:
        """Get current evolution phase."""
        if self.current_phase_idx < len(self.plan.phases):
            return self.plan.phases[self.current_phase_idx]
        return None
    
    def advance_phase(self):
        """Advance to next phase."""
        self.current_phase_idx += 1
        logger.info(f"Advanced to phase {self.current_phase_idx}")
    
    def guided_mutation(
        self,
        individual: Any,
        fitness_history: List[float]
    ) -> Any:
        """Apply mutation guided by PES plan."""
        phase = self.get_current_phase()
        
        if not phase:
            # No guidance, apply standard mutation
            return self._standard_mutation(individual)
        
        # Get phase-specific parameters
        mutation_rate = phase.suggested_parameters.get("mutation_rate", 0.1)
        
        # Apply phase-appropriate mutation
        if phase.name == "exploration":
            return self._exploratory_mutation(individual, mutation_rate)
        elif phase.name == "exploitation":
            return self._exploitative_mutation(individual, mutation_rate)
        else:
            return self._standard_mutation(individual)
    
    def guided_selection(
        self,
        population: List,
        fitness_scores: List[float]
    ) -> List:
        """Select individuals guided by current phase objectives."""
        phase = self.get_current_phase()
        
        if not phase:
            return self._standard_selection(population, fitness_scores)
        
        # Weight by alignment with phase objectives
        weighted_scores = []
        for ind, fit in zip(population, fitness_scores):
            alignment = self._alignment_score(ind, phase.objectives)
            weighted_scores.append((ind, fit * alignment))
        
        # Sort by weighted score
        weighted_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top individuals
        selection_size = len(population) // 2
        return [ind for ind, _ in weighted_scores[:selection_size]]
    
    def _alignment_score(self, individual: Any, objectives: List[str]) -> float:
        """Calculate alignment with phase objectives."""
        # This would calculate how well an individual aligns with objectives
        # For now, return neutral score
        return 1.0
    
    def _standard_mutation(self, individual: Any) -> Any:
        """Apply standard mutation."""
        return individual
    
    def _exploratory_mutation(self, individual: Any, rate: float) -> Any:
        """Apply exploratory (high diversity) mutation."""
        # Higher mutation rate for exploration
        return individual
    
    def _exploitative_mutation(self, individual: Any, rate: float) -> Any:
        """Apply exploitative (local search) mutation."""
        # Lower mutation rate for fine-tuning
        return individual
    
    def _standard_selection(self, population: List, fitness_scores: List[float]) -> List:
        """Apply standard selection."""
        # Tournament selection or similar
        sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        return [ind for ind, _ in sorted_pop[:len(population)//2]]
