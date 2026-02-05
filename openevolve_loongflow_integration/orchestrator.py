"""
Strategy Orchestrator for OpenEvolve + LoongFlow PES Integration.

Coordinates between PES planning layer and OpenEvolve evolution engine.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Dict, Any
import asyncio
import logging

from .config import UnifiedEvolutionConfig, StrategySelectionMode


logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Available evolution strategies."""
    PES_ENHANCED = auto()    # PES-guided evolution
    QD_STANDARD = auto()     # Quality Diversity (MAP-Elites)
    MO_STANDARD = auto()     # Multi-Objective (NSGA-II)
    STANDARD = auto()        # Standard evolution
    ADVERSARIAL = auto()     # Adversarial co-evolution


@dataclass
class EvolutionProblem:
    """Definition of an evolution problem."""
    description: str
    code: Optional[str] = None
    test_cases: Optional[list] = None
    language: str = "python"
    objectives: Optional[list] = None
    constraints: Optional[Dict[str, Any]] = None
    exploration_focus: bool = False


@dataclass
class EvolutionResult:
    """Result of evolution execution."""
    success: bool
    best_solution: Optional[Any]
    final_fitness: float
    iterations: int
    cost_summary: Dict[str, float]
    strategy_used: StrategyType
    execution_time_seconds: float
    
    # PES-specific
    plan_followed: Optional[Dict] = None
    adaptations_made: Optional[list] = None
    summary: Optional[str] = None


class StrategySelector:
    """Selects optimal strategy based on problem characteristics."""
    
    def __init__(self, config: UnifiedEvolutionConfig):
        self.config = config
    
    def select_strategy(self, problem: EvolutionProblem) -> StrategyType:
        """
        Select optimal strategy based on problem analysis.
        
        Rules:
        - Multiple objectives → MO mode
        - Exploration focus → QD mode
        - PES enabled and beneficial → PES_ENHANCED
        - Default → STANDARD
        """
        # Check configuration mode
        if self.config.strategy_selection_mode == StrategySelectionMode.MANUAL:
            return self._manual_strategy()
        elif self.config.strategy_selection_mode == StrategySelectionMode.PES_ONLY:
            return StrategyType.PES_ENHANCED
        
        # Automatic selection
        return self._automatic_selection(problem)
    
    def _automatic_selection(self, problem: EvolutionProblem) -> StrategyType:
        """Automatically select strategy based on problem characteristics."""
        
        # Multi-objective problems
        if problem.objectives and len(problem.objectives) > 1:
            if self.config.pes_config.enable_pes_planning:
                return StrategyType.PES_ENHANCED  # PES can enhance MO
            return StrategyType.MO_STANDARD
        
        # Exploration-focused problems
        if problem.exploration_focus:
            if self.config.pes_config.enable_pes_planning:
                return StrategyType.PES_ENHANCED
            return StrategyType.QD_STANDARD
        
        # Check if PES is enabled and would be beneficial
        if self.config.pes_config.enable_pes_planning:
            # Complex problems benefit from PES
            complexity = self._estimate_complexity(problem)
            if complexity > 0.5:
                return StrategyType.PES_ENHANCED
        
        # Default to standard
        return StrategyType.STANDARD
    
    def _estimate_complexity(self, problem: EvolutionProblem) -> float:
        """Estimate problem complexity (0.0 - 1.0)."""
        complexity = 0.5  # Base complexity
        
        # Longer descriptions tend to be more complex
        if len(problem.description) > 500:
            complexity += 0.1
        
        # Multiple objectives add complexity
        if problem.objectives:
            complexity += min(len(problem.objectives) * 0.1, 0.3)
        
        # Constraints add complexity
        if problem.constraints:
            complexity += min(len(problem.constraints) * 0.05, 0.2)
        
        return min(complexity, 1.0)
    
    def _manual_strategy(self) -> StrategyType:
        """Get manually specified strategy."""
        mode_map = {
            "pes_enhanced": StrategyType.PES_ENHANCED,
            "qd": StrategyType.QD_STANDARD,
            "mo": StrategyType.MO_STANDARD,
            "standard": StrategyType.STANDARD,
            "adversarial": StrategyType.ADVERSARIAL,
        }
        return mode_map.get(self.config.evolution_mode, StrategyType.STANDARD)


class StrategyOrchestrator:
    """
    Central orchestrator for unified evolution system.
    
    Coordinates between:
    - PES planning layer (LoongFlow)
    - OpenEvolve evolution engine
    - Budget monitoring
    - Knowledge extraction
    
    Usage:
        config = UnifiedEvolutionConfig(enable_pes_planning=True)
        orchestrator = StrategyOrchestrator(config)
        result = await orchestrator.evolve(problem)
    """
    
    def __init__(self, config: UnifiedEvolutionConfig):
        self.config = config
        self.strategy_selector = StrategySelector(config)
        self._init_components()
    
    def _init_components(self):
        """Initialize PES and OpenEvolve components."""
        # Lazy initialization to avoid import overhead
        self._pes_planner = None
        self._pes_executor = None
        self._adapter = None
        self._budget_monitor = None
    
    @property
    def pes_planner(self):
        """Lazy-load PES planner."""
        if self._pes_planner is None:
            from .pes_planner import PESPlanner
            self._pes_planner = PESPlanner(self.config)
        return self._pes_planner
    
    @property
    def adapter(self):
        """Lazy-load OpenEvolve adapter."""
        if self._adapter is None:
            from .adapter import PESOpenEvolveAdapter
            self._adapter = PESOpenEvolveAdapter(self.config)
        return self._adapter
    
    async def evolve(self, problem: EvolutionProblem) -> EvolutionResult:
        """
        Execute evolution with optimal strategy.
        
        Flow:
        1. Select strategy based on problem
        2. If PES mode: create plan, execute with adaptation
        3. If standard mode: delegate to OpenEvolve directly
        4. Summarize and return results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 1. Select strategy
            strategy = self.strategy_selector.select_strategy(problem)
            logger.info(f"Selected strategy: {strategy.name}")
            
            # 2. Execute based on strategy
            if strategy == StrategyType.PES_ENHANCED:
                result = await self._run_pes_enhanced(problem)
            elif strategy == StrategyType.QD_STANDARD:
                result = await self._run_standard_qd(problem)
            elif strategy == StrategyType.MO_STANDARD:
                result = await self._run_standard_mo(problem)
            else:
                result = await self._run_standard_evolution(problem)
            
            # 3. Calculate execution time
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return EvolutionResult(
                success=result.get("success", False),
                best_solution=result.get("best_solution"),
                final_fitness=result.get("final_fitness", 0.0),
                iterations=result.get("iterations", 0),
                cost_summary=result.get("cost_summary", {}),
                strategy_used=strategy,
                execution_time_seconds=execution_time,
                plan_followed=result.get("plan"),
                adaptations_made=result.get("adaptations"),
                summary=result.get("summary")
            )
            
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            
            # Fallback to standard mode if configured
            if self.config.fallback_to_standard:
                logger.info("Falling back to standard evolution")
                return await self._run_standard_evolution(problem)
            
            raise
    
    async def _run_pes_enhanced(self, problem: EvolutionProblem) -> Dict:
        """Run PES-enhanced evolution."""
        from .pes_executor import PESExecutor
        from .budget_monitor import BudgetMonitor
        
        # 1. Create plan
        plan = await self.pes_planner.create_plan(problem)
        
        # 2. Initialize budget monitor
        budget_monitor = BudgetMonitor(plan.budget_allocation)
        
        # 3. Create executor
        executor = PESExecutor(
            adapter=self.adapter,
            budget_monitor=budget_monitor,
            config=self.config
        )
        
        # 4. Execute with PES guidance
        return await executor.execute_plan(plan, problem)
    
    async def _run_standard_evolution(self, problem: EvolutionProblem) -> Dict:
        """Run standard OpenEvolve evolution."""
        return await self.adapter.run_standard_evolution(problem)
    
    async def _run_standard_qd(self, problem: EvolutionProblem) -> Dict:
        """Run standard Quality Diversity evolution."""
        return await self.adapter.run_qd_evolution(problem)
    
    async def _run_standard_mo(self, problem: EvolutionProblem) -> Dict:
        """Run standard Multi-Objective evolution."""
        return await self.adapter.run_mo_evolution(problem)
