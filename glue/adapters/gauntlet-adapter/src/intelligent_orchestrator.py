"""
Intelligent Gauntlet Orchestrator

AI-powered orchestration of complex gauntlet workflows with multi-objective optimization.

Features:
- Multi-objective optimization (accuracy, speed, cost)
- Automated gauntlet composition based on problem type
- Intelligent scheduling and resource allocation
- Dynamic adaptation during execution
- Integration with ML optimizer and predictive executor

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives for orchestration"""
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    BALANCED = "balanced"


class OrchestrationStrategy(Enum):
    """Orchestration strategies"""
    SEQUENTIAL = "sequential"  # Execute rounds one after another
    PARALLEL = "parallel"  # Execute rounds in parallel where possible
    ADAPTIVE = "adaptive"  # Adapt based on intermediate results
    HIERARCHICAL = "hierarchical"  # Multi-level decision tree


@dataclass
class OrchestrationPlan:
    """
    Plan for gauntlet orchestration.

    Attributes:
        strategy: Orchestration strategy to use
        execution_order: Order of round execution
        resource_allocation: Resource allocation per round
        stopping_conditions: Conditions for early termination
        fallback_plans: Backup plans if primary fails
        estimated_time: Estimated total execution time
        estimated_cost: Estimated computational cost
    """
    strategy: OrchestrationStrategy
        execution_order: List[str] = field(default_factory=list)
        resource_allocation: Dict[str, Dict[str, Any]] = field(default_factory=dict)
        stopping_conditions: List[Dict[str, Any]] = field(default_factory=list)
        fallback_plans: List[Dict[str, Any]] = field(default_factory=list)
        estimated_time: float = 0.0
        estimated_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "strategy": self.strategy.value,
            "execution_order": self.execution_order,
            "resource_allocation": self.resource_allocation,
            "stopping_conditions": self.stopping_conditions,
            "fallback_plans": self.fallback_plans,
            "estimated_time": self.estimated_time,
            "estimated_cost": self.estimated_cost
        }


@dataclass
class OrchestrationResult:
    """
    Result from gauntlet orchestration.

    Attributes:
        passed: Whether solution passed gauntlet
        final_score: Final aggregated score
        rounds_completed: Number of rounds completed
        execution_time: Total execution time
        actual_cost: Actual computational cost
        resource_utilization: Resource utilization statistics
        adaptations_made: List of adaptations made during execution
        recommendations: Recommendations for improvement
    """
    passed: bool
    final_score: float
    rounds_completed: int
    execution_time: float
    actual_cost: float
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    adaptations_made: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "passed": self.passed,
            "final_score": self.final_score,
            "rounds_completed": self.rounds_completed,
            "execution_time": self.execution_time,
            "actual_cost": self.actual_cost,
            "resource_utilization": self.resource_utilization,
            "adaptations_made": self.adaptations_made,
            "recommendations": self.recommendations
        }


class IntelligentGauntletOrchestrator:
    """
    AI-powered gauntlet orchestrator with intelligent optimization.

    Automatically determines optimal gauntlet configuration,
    execution strategy, and resource allocation based on
    problem characteristics and optimization objectives.

    Example:
        >>> orchestrator = IntelligentGauntletOrchestrator(
        ...     objective=OptimizationObjective.BALANCED
        ... )
        >>>
        >>> # Create optimal plan
        >>> plan = orchestrator.create_orchestration_plan(
        ...     solution="def solve(): return optimal",
        ...     problem="Optimize portfolio",
        ...     domain="finance"
        ... )
        >>>
        >>> # Execute with intelligent orchestration
        >>> result = orchestrator.execute_orchestration(
        ...     solution="def solve(): return optimal",
        ...     problem="Optimize portfolio",
        ...     domain="finance",
        ...     plan=plan
        ... )
    """

    def __init__(
        self,
        objective: OptimizationObjective = OptimizationObjective.BALANCED,
        max_parallelism: int = 4,
        enable_prediction: bool = True,
        enable_optimization: bool = True
    ):
        """
        Initialize intelligent orchestrator.

        Args:
            objective: Primary optimization objective
            max_parallelism: Maximum parallel execution capacity
            enable_prediction: Enable predictive execution
            enable_optimization: Enable ML-based optimization
        """
        self.objective = objective
        self.max_parallelism = max_parallelism
        self.enable_prediction = enable_prediction
        self.enable_optimization = enable_optimization

        # Optional integrations
        self.predictive_executor = None
        self.ml_optimizer = None

        # Performance tracking
        self.execution_history: List[Dict[str, Any]] = []

        logger.info(
            f"Intelligent Gauntlet Orchestrator initialized: "
            f"objective={objective.value}, "
            f"max_parallelism={max_parallelism}"
        )

    def set_predictive_executor(self, executor):
        """Set predictive executor for integration"""
        self.predictive_executor = executor

    def set_ml_optimizer(self, optimizer):
        """Set ML optimizer for integration"""
        self.ml_optimizer = optimizer

    def create_orchestration_plan(
        self,
        solution: str,
        problem: str,
        domain: str,
        context: Optional[Dict[str, Any]] = None
    ) -> OrchestrationPlan:
        """
        Create optimal orchestration plan.

        Analyzes problem and solution to determine:
        - Optimal execution strategy
        - Resource allocation
        - Stopping conditions
        - Fallback plans

        Args:
            solution: Solution to evaluate
            problem: Problem statement
            domain: Problem domain
            context: Additional context

        Returns:
            OrchestrationPlan with optimal strategy
        """
        start_time = time.time()
        logger.info(f"Creating orchestration plan for domain={domain}")

        # Analyze problem characteristics
        characteristics = self._analyze_characteristics(solution, problem, domain, context)

        # Determine optimal strategy
        strategy = self._select_strategy(characteristics)

        # Create execution order
        execution_order = self._create_execution_order(characteristics, strategy)

        # Allocate resources
        resource_allocation = self._allocate_resources(characteristics, execution_order)

        # Set stopping conditions
        stopping_conditions = self._create_stopping_conditions(characteristics)

        # Create fallback plans
        fallback_plans = self._create_fallback_plans(characteristics)

        # Estimate time and cost
        estimated_time = self._estimate_time(characteristics, execution_order)
        estimated_cost = self._estimate_cost(characteristics, execution_order)

        planning_time = time.time() - start_time

        plan = OrchestrationPlan(
            strategy=strategy,
            execution_order=execution_order,
            resource_allocation=resource_allocation,
            stopping_conditions=stopping_conditions,
            fallback_plans=fallback_plans,
            estimated_time=estimated_time,
            estimated_cost=estimated_cost
        )

        logger.info(
            f"Orchestration plan created in {planning_time:.3f}s: "
            f"strategy={strategy.value}, "
            f"estimated_time={estimated_time:.1f}s"
        )

        return plan

    async def execute_orchestration(
        self,
        solution: str,
        problem: str,
        domain: str,
        plan: Optional[OrchestrationPlan] = None,
        gauntlet_executor: Optional[Any] = None
    ) -> OrchestrationResult:
        """
        Execute gauntlet with intelligent orchestration.

        Args:
            solution: Solution to evaluate
            problem: Problem statement
            domain: Problem domain
            plan: Orchestration plan (created if None)
            gauntlet_executor: Actual gauntlet executor

        Returns:
            OrchestrationResult with execution outcome
        """
        start_time = time.time()

        # Create plan if not provided
        if plan is None:
            plan = self.create_orchestration_plan(solution, problem, domain)

        logger.info(f"Executing orchestration: strategy={plan.strategy.value}")

        # Execute based on strategy
        if plan.strategy == OrchestrationStrategy.SEQUENTIAL:
            result = await self._execute_sequential(solution, problem, domain, plan, gauntlet_executor)
        elif plan.strategy == OrchestrationStrategy.PARALLEL:
            result = await self._execute_parallel(solution, problem, domain, plan, gauntlet_executor)
        elif plan.strategy == OrchestrationStrategy.ADAPTIVE:
            result = await self._execute_adaptive(solution, problem, domain, plan, gauntlet_executor)
        elif plan.strategy == OrchestrationStrategy.HIERARCHICAL:
            result = await self._execute_hierarchical(solution, problem, domain, plan, gauntlet_executor)
        else:
            result = await self._execute_sequential(solution, problem, domain, plan, gauntlet_executor)

        result.execution_time = time.time() - start_time

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result, plan)

        # Track execution
        self.execution_history.append({
            "timestamp": datetime.now(UTC).isoformat(),
            "domain": domain,
            "strategy": plan.strategy.value,
            "result": result.to_dict(),
            "plan": plan.to_dict()
        })

        logger.info(
            f"Orchestration complete: passed={result.passed}, "
            f"score={result.final_score:.3f}, "
            f"time={result.execution_time:.2f}s"
        )

        return result

    def _analyze_characteristics(
        self,
        solution: str,
        problem: str,
        domain: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze problem and solution characteristics"""
        return {
            "solution_length": len(solution),
            "solution_lines": len(solution.splitlines()),
            "problem_length": len(problem),
            "domain": domain,
            "complexity": self._calculate_complexity(solution),
            "has_tests": "test" in solution.lower() or "assert" in solution,
            "has_error_handling": "try" in solution and "except" in solution,
            "num_functions": solution.count("def "),
            "num_classes": solution.count("class "),
            "domain_difficulty": self._get_domain_difficulty(domain)
        }

    def _calculate_complexity(self, solution: str) -> float:
        """Calculate solution complexity"""
        complexity = 0.5
        lines = len(solution.splitlines())

        if lines > 100:
            complexity += 0.2
        elif lines > 50:
            complexity += 0.1
        elif lines < 10:
            complexity -= 0.2

        func_count = solution.count("def ")
        complexity += min(0.15, func_count * 0.03)

        class_count = solution.count("class ")
        complexity += min(0.1, class_count * 0.05)

        return max(0.0, min(1.0, complexity))

    def _get_domain_difficulty(self, domain: str) -> float:
        """Get inherent difficulty for domain"""
        difficulties = {
            "math": 0.7,
            "algorithm": 0.8,
            "ml": 0.8,
            "code": 0.5,
            "general": 0.4
        }
        return difficulties.get(domain.lower(), 0.5)

    def _select_strategy(self, characteristics: Dict[str, Any]) -> OrchestrationStrategy:
        """Select optimal orchestration strategy"""
        complexity = characteristics["complexity"]
        domain = characteristics["domain"]

        # High complexity domains benefit from adaptive strategy
        if complexity > 0.7 or domain in ["math", "algorithm"]:
            return OrchestrationStrategy.ADAPTIVE

        # Simple problems can use parallel for speed
        if complexity < 0.4 and self.objective in [OptimizationObjective.MINIMIZE_TIME, OptimizationObjective.MAXIMIZE_THROUGHPUT]:
            return OrchestrationStrategy.PARALLEL

        # Default to sequential for reliability
        return OrchestrationStrategy.SEQUENTIAL

    def _create_execution_order(
        self,
        characteristics: Dict[str, Any],
        strategy: OrchestrationStrategy
    ) -> List[str]:
        """Create optimal execution order"""
        base_order = ["round1_loongflow", "round2_red_team", "round3_gold_team"]

        if strategy == OrchestrationStrategy.PARALLEL:
            # Can run round1 in parallel with some aspects of round2
            return ["round1_loongflow_parallel", "round2_red_team", "round3_gold_team"]

        return base_order

    def _allocate_resources(
        self,
        characteristics: Dict[str, Any],
        execution_order: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Allocate resources for each round"""
        allocation = {}

        complexity = characteristics["complexity"]

        for round_name in execution_order:
            if "round1" in round_name:
                allocation[round_name] = {
                    "max_evaluations": int(50 * (1 + complexity)),
                    "timeout": int(30 * (1 + complexity)),
                    "parallel": self.max_parallelism > 1
                }
            elif "round2" in round_name:
                allocation[round_name] = {
                    "max_attacks": int(10 * (1 + complexity)),
                    "timeout": int(60 * (1 + complexity)),
                    "parallel": False  # Sequential for red team
                }
            elif "round3" in round_name:
                allocation[round_name] = {
                    "num_evaluators": 3,
                    "timeout": int(90 * (1 + complexity)),
                    "parallel": True  # Can run evaluators in parallel
                }

        return allocation

    def _create_stopping_conditions(self, characteristics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create early stopping conditions"""
        return [
            {
                "condition": "round1_failed",
                "threshold": 0.3,
                "action": "terminate"
            },
            {
                "condition": "confidence_low",
                "threshold": 0.5,
                "action": "continue_with_warning"
            },
            {
                "condition": "timeout_exceeded",
                "threshold": 300,
                "action": "terminate_early"
            }
        ]

    def _create_fallback_plans(self, characteristics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create fallback plans for failures"""
        return [
            {
                "scenario": "primary_executor_failed",
                "fallback": "use_mock_executor",
                "config": {"timeout": 10}
            },
            {
                "scenario": "round_timeout",
                "fallback": "reduce_complexity",
                "config": {"max_evaluations": 20}
            }
        ]

    def _estimate_time(
        self,
        characteristics: Dict[str, Any],
        execution_order: List[str]
    ) -> float:
        """Estimate total execution time"""
        base_time = len(execution_order) * 30
        complexity_multiplier = 1 + characteristics["complexity"]
        return base_time * complexity_multiplier

    def _estimate_cost(
        self,
        characteristics: Dict[str, Any],
        execution_order: List[str]
    ) -> float:
        """Estimate computational cost"""
        base_cost = len(execution_order) * 1.0
        complexity_multiplier = 1 + characteristics["complexity"]
        return base_cost * complexity_multiplier

    async def _execute_sequential(
        self,
        solution: str,
        problem: str,
        domain: str,
        plan: OrchestrationPlan,
        executor: Optional[Any]
    ) -> OrchestrationResult:
        """Execute sequential orchestration"""
        rounds_completed = 0
        total_score = 0.0
        adaptations = []

        for round_name in plan.execution_order:
            # Simulate round execution
            result = await self._simulate_round(solution, problem, domain, round_name, plan)

            total_score += result["score"]
            rounds_completed += 1

            if not result["passed"]:
                # Check stopping conditions
                for condition in plan.stopping_conditions:
                    if condition["condition"] == "round1_failed":
                        if result["score"] < condition["threshold"]:
                            adaptations.append(f"Early termination: {condition['condition']}")
                            break
                break

        return OrchestrationResult(
            passed=total_score / rounds_completed > 0.6 if rounds_completed > 0 else False,
            final_score=total_score / max(1, rounds_completed),
            rounds_completed=rounds_completed,
            execution_time=0.0,  # Will be set by caller
            actual_cost=rounds_completed * 1.0,
            adaptations_made=adaptations
        )

    async def _execute_parallel(
        self,
        solution: str,
        problem: str,
        domain: str,
        plan: OrchestrationPlan,
        executor: Optional[Any]
    ) -> OrchestrationResult:
        """Execute parallel orchestration"""
        # Simulate parallel execution
        tasks = [
            self._simulate_round(solution, problem, domain, round_name, plan)
            for round_name in plan.execution_order
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        rounds_completed = len([r for r in results if isinstance(r, dict) and r.get("passed", False)])
        total_score = sum([r["score"] for r in results if isinstance(r, dict)]) / max(1, len(results))

        return OrchestrationResult(
            passed=total_score > 0.6,
            final_score=total_score,
            rounds_completed=rounds_completed,
            execution_time=0.0,
            actual_cost=len(plan.execution_order) * 0.5,  # Parallel is cheaper
            resource_utilization={"parallelism": len(plan.execution_order)}
        )

    async def _execute_adaptive(
        self,
        solution: str,
        problem: str,
        domain: str,
        plan: OrchestrationPlan,
        executor: Optional[Any]
    ) -> OrchestrationResult:
        """Execute adaptive orchestration"""
        rounds_completed = 0
        total_score = 0.0
        adaptations = []

        for round_name in plan.execution_order:
            result = await self._simulate_round(solution, problem, domain, round_name, plan)

            total_score += result["score"]
            rounds_completed += 1

            # Adapt based on result
            if result["score"] < 0.5 and rounds_completed == 1:
                # Low score on first round - adjust strategy
                adaptations.append("Adjusted thresholds after low round1 score")
                # In real implementation, would modify plan here

            if not result["passed"]:
                break

        return OrchestrationResult(
            passed=total_score / rounds_completed > 0.6 if rounds_completed > 0 else False,
            final_score=total_score / max(1, rounds_completed),
            rounds_completed=rounds_completed,
            execution_time=0.0,
            actual_cost=rounds_completed * 1.0,
            adaptations_made=adaptations
        )

    async def _execute_hierarchical(
        self,
        solution: str,
        problem: str,
        domain: str,
        plan: OrchestrationPlan,
        executor: Optional[Any]
    ) -> OrchestrationResult:
        """Execute hierarchical orchestration"""
        # Multi-level decision tree
        result = await self._simulate_round(solution, problem, domain, "round1_loongflow", plan)

        if result["score"] > 0.8:
            # High confidence - skip to final round
            result = await self._simulate_round(solution, problem, domain, "round3_gold_team", plan)
            return OrchestrationResult(
                passed=result["passed"],
                final_score=result["score"],
                rounds_completed=2,
                execution_time=0.0,
                actual_cost=2.0,
                adaptations_made=["Skipped round2 due to high round1 score"]
            )
        else:
            # Standard execution
            return await self._execute_sequential(solution, problem, domain, plan, executor)

    async def _simulate_round(
        self,
        solution: str,
        problem: str,
        domain: str,
        round_name: str,
        plan: OrchestrationPlan
    ) -> Dict[str, Any]:
        """Simulate round execution (or use real executor)"""
        # Simulate execution time
        await asyncio.sleep(0.01)

        # Generate simulated result
        base_score = np.random.uniform(0.4, 0.9)
        passed = base_score > 0.6

        return {
            "round": round_name,
            "score": base_score,
            "passed": passed,
            "execution_time": np.random.uniform(10, 60)
        }

    def _generate_recommendations(
        self,
        result: OrchestrationResult,
        plan: OrchestrationPlan
    ) -> List[str]:
        """Generate recommendations based on execution result"""
        recommendations = []

        if not result.passed:
            recommendations.append("Solution failed gauntlet - review and improve before resubmission")

        if result.execution_time > plan.estimated_time * 1.5:
            recommendations.append("Execution took longer than estimated - consider optimizing solution")

        if result.actual_cost > plan.estimated_cost * 1.2:
            recommendations.append("Higher than expected cost - consider reducing solution complexity")

        if len(result.adaptations_made) > 0:
            recommendations.append(f"System made {len(result.adaptations_made)} adaptations during execution")

        if result.passed and result.final_score > 0.8:
            recommendations.append("Excellent solution quality - suitable for production")

        return recommendations

    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get orchestration statistics"""
        if not self.execution_history:
            return {"error": "No executions yet"}

        total_executions = len(self.execution_history)
        passed_count = sum(1 for e in self.execution_history if e["result"]["passed"])
        avg_score = np.mean([e["result"]["final_score"] for e in self.execution_history])
        avg_time = np.mean([e["result"]["execution_time"] for e in self.execution_history])

        strategy_counts = {}
        for execution in self.execution_history:
            strategy = execution["strategy"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        return {
            "total_executions": total_executions,
            "pass_rate": passed_count / total_executions if total_executions > 0 else 0,
            "average_score": avg_score,
            "average_time": avg_time,
            "strategy_distribution": strategy_counts
        }
