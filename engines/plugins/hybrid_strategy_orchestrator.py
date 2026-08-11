"""
Hybrid Strategy Orchestrator: Combining OpenEvolve's Strategy Selection with LoongFlow's Execution

This module demonstrates the integration of:
1. OpenEvolve's EnsembleStrategySelector for intelligent strategy selection
2. LoongFlow's PESAgent for Plan-Execute-Summarize execution
3. Cost-aware decision making based on evaluation cost categories
4. Online learning from execution results

Usage:
    orchestrator = HybridStrategyOrchestrator()
    result = await orchestrator.evolve(
        problem_description="Optimize trading strategy for BTC/USD",
        domain="trading",
        constraints={"budget_usd": 100, "max_iterations": 100}
    )
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# OpenEvolve imports
from knowledge_engine.core.strategy_recommender import (
    EnsembleStrategySelector,
    EnsemblePrediction,
    ProblemCharacteristics
)
from adaptive_strategy_selector import (
    StrategyPerformanceTracker,
    AdaptiveWeightCalculator
)
from strategy_templates import StrategyTemplates

# LoongFlow imports
try:
    from loongflow.framework.pes.pes_agent import PESAgent
    from loongflow.framework.pes.context.config import EvolveChainConfig, load_config
    from loongflow.framework.pes.context import Context
    LOONGFLOW_AVAILABLE = True
except ImportError:
    LOONGFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Unified execution result from either system"""
    system_used: str  # "openevolve" or "loongflow"
    mode_used: str  # "pes", "qd", "mo", "adversarial", "standard"
    final_score: float
    iterations: int
    total_cost_usd: float
    total_tokens: int
    duration_seconds: float
    best_solution: Any
    strategy_recommendation: EnsemblePrediction
    metadata: Dict[str, Any]


@dataclass
class CostBudget:
    """Budget constraints for evolution"""
    max_usd: Optional[float] = None
    max_tokens: Optional[int] = None
    max_iterations: Optional[int] = None
    warning_threshold: float = 0.8  # Warn at 80% of budget


class HybridStrategyOrchestrator:
    """
    Orchestrates strategy selection and execution using best of both systems.
    
    Strategy Selection (OpenEvolve):
    - Uses ensemble of 4 methods: rule-based, similarity, trend, ML
    - Considers evaluation cost as primary decision factor
    - Provides confidence intervals for predictions
    - Adapts weights based on historical accuracy
    
    Execution (LoongFlow for PES, OpenEvolve for others):
    - PES mode: Uses LoongFlow's Plan-Execute-Summarize
    - Other modes: Uses OpenEvolve's parameter-rich execution
    
    Cost Tracking (Combined):
    - OpenEvolve's cost categories for strategy selection
    - LoongFlow's token-level tracking for budget monitoring
    """
    
    def __init__(
        self,
        enable_learning: bool = True,
        confidence_level: float = 0.95,
        default_budget: Optional[CostBudget] = None
    ):
        """
        Initialize the hybrid orchestrator.
        
        Args:
            enable_learning: Enable online learning from results
            confidence_level: Confidence level for strategy predictions
            default_budget: Default budget constraints
        """
        self.enable_learning = enable_learning
        self.confidence_level = confidence_level
        self.default_budget = default_budget or CostBudget()
        
        # OpenEvolve components
        self.strategy_selector = EnsembleStrategySelector(
            enable_ml=True,
            enable_loongflow=LOONGFLOW_AVAILABLE,
            learning_enabled=enable_learning
        )
        self.strategy_templates = StrategyTemplates()
        
        # Performance tracking
        self.performance_tracker = StrategyPerformanceTracker()
        self.adaptive_calculator = AdaptiveWeightCalculator(self.performance_tracker)
        
        # Execution history for learning
        self.execution_history: List[ExecutionResult] = []
        
        # Token price lookup (for cost tracking)
        self.token_prices = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4o": {"prompt": 0.005, "completion": 0.015},
            "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
        }
        
        logger.info(f"HybridStrategyOrchestrator initialized (LoongFlow available: {LOONGFLOW_AVAILABLE})")
    
    async def evolve(
        self,
        problem_description: str,
        domain: str = "general",
        constraints: Optional[Dict[str, Any]] = None,
        initial_solution: Optional[str] = None,
        budget: Optional[CostBudget] = None
    ) -> ExecutionResult:
        """
        Main entry point: select strategy and execute evolution.
        
        Args:
            problem_description: Description of the problem to solve
            domain: Problem domain (finance, trading, science, etc.)
            constraints: Additional constraints (objectives, requirements)
            initial_solution: Optional initial solution to start from
            budget: Budget constraints (uses default if not provided)
            
        Returns:
            ExecutionResult with results and metadata
        """
        start_time = datetime.now()
        constraints = constraints or {}
        budget = budget or self.default_budget
        
        logger.info(f"Starting evolution for domain='{domain}' with budget={budget}")
        
        # Phase 1: Strategy Selection (OpenEvolve intelligence)
        logger.info("Phase 1: Strategy selection using ensemble...")
        recommendation = await self._select_strategy(
            problem_description=problem_description,
            domain=domain,
            constraints=constraints
        )
        
        system = recommendation.strategy[0]
        mode = recommendation.strategy[1]
        
        logger.info(f"Selected strategy: {system}/{mode} (confidence: {recommendation.confidence_interval})")
        
        # Phase 2: Execute based on selected strategy
        logger.info("Phase 2: Executing evolution...")
        
        if system == "loongflow" and mode == "pes" and LOONGFLOW_AVAILABLE:
            result = await self._execute_pes(
                problem_description=problem_description,
                recommendation=recommendation,
                initial_solution=initial_solution,
                budget=budget
            )
        else:
            result = await self._execute_openevolve(
                problem_description=problem_description,
                recommendation=recommendation,
                domain=domain,
                constraints=constraints,
                initial_solution=initial_solution,
                budget=budget
            )
        
        # Phase 3: Learn from result (if enabled)
        if self.enable_learning:
            logger.info("Phase 3: Learning from execution...")
            await self._learn_from_result(result, recommendation)
        
        duration = (datetime.now() - start_time).total_seconds()
        result.duration_seconds = duration
        
        # Record execution
        self.execution_history.append(result)
        
        logger.info(f"Evolution complete: score={result.final_score:.3f}, "
                   f"cost=${result.total_cost_usd:.4f}, duration={duration:.1f}s")
        
        return result
    
    async def _select_strategy(
        self,
        problem_description: str,
        domain: str,
        constraints: Dict[str, Any]
    ) -> EnsemblePrediction:
        """
        Use OpenEvolve's ensemble selector to choose optimal strategy.
        
        This considers:
        - Evaluation cost (cheap/moderate/expensive/very_expensive)
        - Multiple objectives
        - Diversity requirements
        - Robustness requirements
        - Domain-specific heuristics
        - Historical performance
        """
        # Add budget constraints to influence selection
        if constraints.get("budget_usd"):
            # Lower budget = prefer PES for expensive evaluations
            constraints["evaluation_cost"] = self._estimate_evaluation_cost(
                domain, constraints.get("time_limit_seconds")
            )
        
        recommendation = await self.strategy_selector.recommend_with_ensemble(
            problem_description=problem_description,
            domain=domain,
            constraints=constraints,
            confidence_level=self.confidence_level,
            enable_loongflow=LOONGFLOW_AVAILABLE
        )
        
        return recommendation
    
    def _estimate_evaluation_cost(
        self,
        domain: str,
        time_limit: Optional[int] = None
    ) -> str:
        """
        Estimate evaluation cost category for strategy selection.
        
        Based on OpenEvolve's cost-aware decision rules:
        - science, pharma: very_expensive
        - finance, trading, engineering: expensive
        - web: cheap
        - others: moderate
        """
        domain_costs = {
            "science": "very_expensive",
            "pharma": "very_expensive",
            "finance": "expensive",
            "trading": "expensive",
            "engineering": "expensive",
            "web": "cheap",
            "web_design": "cheap",
        }
        
        base_cost = domain_costs.get(domain, "moderate")
        
        # Time limit can bump cost category
        if time_limit:
            if time_limit > 600:  # > 10 minutes
                return "very_expensive"
            elif time_limit > 60:  # > 1 minute
                return "expensive"
            elif time_limit < 1:  # < 1 second
                return "cheap"
        
        return base_cost
    
    async def _execute_pes(
        self,
        problem_description: str,
        recommendation: EnsemblePrediction,
        initial_solution: Optional[str],
        budget: CostBudget
    ) -> ExecutionResult:
        """
        Execute using LoongFlow's PES (Plan-Execute-Summarize).
        
        Advantages:
        - Structured reasoning with explicit planning
        - Granular cost tracking per iteration
        - Async task management
        - Graceful interruption handling
        """
        if not LOONGFLOW_AVAILABLE:
            raise RuntimeError("LoongFlow not available but PES strategy selected")
        
        # Create LoongFlow configuration
        config = self._create_pes_config(
            problem_description=problem_description,
            recommendation=recommendation,
            budget=budget
        )
        
        # Initialize PESAgent
        agent = PESAgent(config=config)
        
        # Track costs manually (LoongFlow tracks internally but we need unified view)
        start_tokens = 0  # Would get from agent if exposed
        
        try:
            # Run PES evolution
            final_message = await agent.run()
            
            # Extract results (simplified - actual extraction depends on message format)
            result_data = {
                "final_score": 0.85,  # Would extract from final_message
                "iterations": config.evolve.max_iterations,
                "total_tokens": agent.total_completion_tokens + agent.total_prompt_tokens,
                "total_cost": self._calculate_cost_from_agent(agent),
                "best_solution": final_message.content if hasattr(final_message, 'content') else None
            }
            
        except asyncio.CancelledError:
            logger.warning("PES execution was cancelled")
            raise
        
        return ExecutionResult(
            system_used="loongflow",
            mode_used="pes",
            final_score=result_data["final_score"],
            iterations=result_data["iterations"],
            total_cost_usd=result_data["total_cost"],
            total_tokens=result_data["total_tokens"],
            duration_seconds=0.0,  # Will be set by caller
            best_solution=result_data["best_solution"],
            strategy_recommendation=recommendation,
            metadata={
                "pes_config": config.evolve.task_name,
                "planner_used": config.evolve.planner_name,
                "executor_used": config.evolve.executor_name,
            }
        )
    
    async def _execute_openevolve(
        self,
        problem_description: str,
        recommendation: EnsemblePrediction,
        domain: str,
        constraints: Dict[str, Any],
        initial_solution: Optional[str],
        budget: CostBudget
    ) -> ExecutionResult:
        """
        Execute using OpenEvolve's parameter-rich execution.
        
        Advantages:
        - 272+ parameters for fine-grained control
        - Domain-specific templates
        - Multiple evolution modes (QD, MO, Adversarial, Standard)
        - Extensive configuration options
        """
        # Get appropriate template based on strategy
        template = self._get_template_for_mode(recommendation.strategy[1], domain)
        
        # Create configuration based on recommendation
        config = self._create_openevolve_config(
            recommendation=recommendation,
            template=template,
            budget=budget
        )
        
        # Execute (simplified - actual execution would use evolution.py or similar)
        logger.info(f"Executing OpenEvolve with {len(config)} parameters")
        
        # Simulate execution (replace with actual OpenEvolve execution)
        result = await self._simulate_openevolve_execution(
            config=config,
            problem_description=problem_description,
            budget=budget
        )
        
        return ExecutionResult(
            system_used="openevolve",
            mode_used=recommendation.strategy[1],
            final_score=result["final_score"],
            iterations=result["iterations"],
            total_cost_usd=result["estimated_cost"],
            total_tokens=result["estimated_tokens"],
            duration_seconds=0.0,  # Will be set by caller
            best_solution=result["solution"],
            strategy_recommendation=recommendation,
            metadata={
                "template_used": template.strategy_name if template else None,
                "config_parameters": len(config),
                "domain": domain,
            }
        )
    
    def _create_pes_config(
        self,
        problem_description: str,
        recommendation: EnsemblePrediction,
        budget: CostBudget
    ) -> "EvolveChainConfig":
        """Create LoongFlow configuration from recommendation."""
        # This would create a proper EvolveChainConfig
        # For now, return a placeholder
        config_path = Path("config/pes_default.yaml")
        if config_path.exists():
            return load_config(str(config_path))
        
        # Return minimal config (actual implementation would be more complete)
        return EvolveChainConfig(
            workspace_path="./evolve_output",
            evolve={
                "task": problem_description,
                "max_iterations": budget.max_iterations or 100,
                "target_score": 1.0,
                "concurrency": 5,
            }
        )
    
    def _create_openevolve_config(
        self,
        recommendation: EnsemblePrediction,
        template: Optional[Any],
        budget: CostBudget
    ) -> Dict[str, Any]:
        """Create OpenEvolve configuration from recommendation."""
        config = {
            "evolution_mode": recommendation.strategy[1],
            "max_iterations": budget.max_iterations or 100,
            "population_size": 20,
            "adaptive_parameters": True,
            "early_stopping": True,
        }
        
        # Add mode-specific parameters
        if recommendation.strategy[1] == "qd":
            config.update({
                "feature_dimensions": ["complexity", "diversity"],
                "archive_size": 100,
            })
        elif recommendation.strategy[1] == "mo":
            config.update({
                "objectives": ["score", "efficiency"],
                "pareto_front_size": 50,
            })
        
        # Add template configuration if available
        if template:
            config["system_prompt"] = template.system_prompt
            config["decomposition_criteria"] = template.decomposition_criteria
        
        return config
    
    def _get_template_for_mode(self, mode: str, domain: str) -> Optional[Any]:
        """Get appropriate strategy template for the selected mode."""
        template_map = {
            "pes": None,  # PES doesn't use templates
            "qd": "complexity_based",
            "mo": "priority_based",
            "adversarial": "domain_specific",
            "standard": "domain_specific",
        }
        
        template_name = template_map.get(mode)
        if template_name == "domain_specific":
            return self.strategy_templates.domain_specific_template(domain)
        elif template_name:
            method = getattr(self.strategy_templates, f"{template_name}_template", None)
            if method:
                return method()
        
        return None
    
    def _calculate_cost_from_agent(self, agent: "PESAgent") -> float:
        """Extract cost from LoongFlow agent."""
        completion_cost = (
            agent.total_completion_tokens / 1000
        ) * agent.config.llm_config.completion_token_price
        prompt_cost = (
            agent.total_prompt_tokens / 1000
        ) * agent.config.llm_config.prompt_token_price
        return completion_cost + prompt_cost
    
    async def _simulate_openevolve_execution(
        self,
        config: Dict[str, Any],
        problem_description: str,
        budget: CostBudget
    ) -> Dict[str, Any]:
        """
        Simulate OpenEvolve execution (replace with actual execution).
        
        In production, this would call the actual OpenEvolve evolution engine.
        """
        # Placeholder simulation
        await asyncio.sleep(0.1)  # Simulate work
        
        iterations = min(config.get("max_iterations", 100), budget.max_iterations or 100)
        estimated_tokens = iterations * 500  # Rough estimate
        
        # Estimate cost based on GPT-4 pricing
        cost_per_1k = 0.03  # prompt + completion average
        estimated_cost = (estimated_tokens / 1000) * cost_per_1k
        
        return {
            "final_score": 0.82,  # Simulated
            "iterations": iterations,
            "estimated_cost": estimated_cost,
            "estimated_tokens": estimated_tokens,
            "solution": f"# Solution for: {problem_description[:50]}...",
        }
    
    async def _learn_from_result(
        self,
        result: ExecutionResult,
        recommendation: EnsemblePrediction
    ) -> None:
        """
        Update strategy selector with execution results.
        
        This enables the ensemble to learn which strategies work best
        for different problem types.
        """
        run_result = {
            "run_id": f"run_{datetime.now().isoformat()}",
            "recommendation_id": getattr(recommendation, 'id', None),
            "domain": result.metadata.get("domain", "general"),
            "strategy_used": result.system_used,
            "mode_used": result.mode_used,
            "final_score": result.final_score,
            "iterations": result.iterations,
            "evaluations": result.iterations,  # Approximation
            "diversity_score": 0.5,  # Would calculate from archive
            "evaluation_cost": "moderate",  # Would categorize from actual cost
            "metadata": result.metadata
        }
        
        # Update strategy selector
        await self.strategy_selector.learn_from_run(run_result)
        
        # Update performance tracker
        self.performance_tracker.record_attempt(
            strategy_name=f"{result.system_used}_{result.mode_used}",
            success=result.final_score > 0.7,  # Threshold for success
            quality_score=result.final_score * 100,
            metadata=result.metadata
        )
        
        logger.info(f"Learning recorded: system={result.system_used}, score={result.final_score:.3f}")
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of strategy selection performance."""
        return {
            "total_executions": len(self.execution_history),
            "executions_by_system": self._count_by_system(),
            "average_score": sum(r.final_score for r in self.execution_history) / len(self.execution_history) if self.execution_history else 0,
            "total_cost": sum(r.total_cost_usd for r in self.execution_history),
            "learning_metrics": self.strategy_selector.get_learning_metrics() if hasattr(self.strategy_selector, 'get_learning_metrics') else {},
        }
    
    def _count_by_system(self) -> Dict[str, int]:
        """Count executions by system used."""
        counts = {"openevolve": 0, "loongflow": 0}
        for result in self.execution_history:
            counts[result.system_used] = counts.get(result.system_used, 0) + 1
        return counts


# Example usage
async def main():
    """Example usage of HybridStrategyOrchestrator."""
    logging.basicConfig(level=logging.INFO)
    
    orchestrator = HybridStrategyOrchestrator(
        enable_learning=True,
        confidence_level=0.95,
        default_budget=CostBudget(
            max_usd=50.0,
            max_iterations=50
        )
    )
    
    # Example problem
    problems = [
        {
            "description": "Optimize portfolio allocation for maximum Sharpe ratio",
            "domain": "finance",
            "constraints": {"evaluation_cost": "expensive"}
        },
        {
            "description": "Find diverse trading strategies that work in bull and bear markets",
            "domain": "trading",
            "constraints": {"requires_diversity": True}
        },
        {
            "description": "Generate a simple landing page design",
            "domain": "web",
            "constraints": {"evaluation_cost": "cheap"}
        }
    ]
    
    for problem in problems:
        print(f"\n{'='*60}")
        print(f"Problem: {problem['description'][:50]}...")
        print(f"Domain: {problem['domain']}")
        print(f"{'='*60}")
        
        result = await orchestrator.evolve(
            problem_description=problem["description"],
            domain=problem["domain"],
            constraints=problem["constraints"]
        )
        
        print(f"\nResult:")
        print(f"  System: {result.system_used}")
        print(f"  Mode: {result.mode_used}")
        print(f"  Score: {result.final_score:.3f}")
        print(f"  Cost: ${result.total_cost_usd:.4f}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    summary = orchestrator.get_strategy_summary()
    print(f"Total executions: {summary['total_executions']}")
    print(f"By system: {summary['executions_by_system']}")
    print(f"Average score: {summary['average_score']:.3f}")
    print(f"Total cost: ${summary['total_cost']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
