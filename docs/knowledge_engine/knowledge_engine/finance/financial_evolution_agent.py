"""
FinancialEvolutionAgent - LoongFlow-OpenEvolve Bridge for Finance

Combines LoongFlow's PES (Plan-Execute-Summarize) with OpenEvolve's
crisis-aware backtesting to evolve financial strategies that survive
extinction events.

This is the CORE BRIDGE connecting high-level reasoning (LoongFlow)
with low-level evolution (OpenEvolve) for financial applications.
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
import logging

from .schemas import (
    FinancialEvolutionResult,
    GenerationPlan,
    GenerationExecution,
    GenerationSummary,
    Strategy,
    BacktestResult,
    FitnessScore,
    EvolutionObjective,
    EvolutionBudget,
    CrisisType,
    StrategyType,
    MarketConditions
)

from .financial_memory import FinancialEvolutionMemory, CrisisLesson
from .crisis_aware_fitness import CrisisAwareFitness
from .survivorship_backtester import SurvivorshipBacktester


logger = logging.getLogger(__name__)


class FinancialEvolutionAgent:
    """
    Orchestrates financial strategy evolution using LoongFlow PES.

    Flow:
    1. PLAN: LoongFlow analyzes failures, generates hypotheses
    2. EXECUTE: OpenEvolve generates variants, backtests on survivorship-free data
    3. SUMMARIZE: LoongFlow reflects, stores lessons in memory

    This creates a virtuous cycle where strategies evolve to survive
    historical crises and adapt to new market conditions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize FinancialEvolutionAgent.

        Args:
            config: Configuration dictionary
                - loongflow: LoongFlow adapter settings
                - backtester: Backtester settings
                - fitness: Fitness function settings
                - memory: Memory persistence settings
        """
        self.config = config or {}

        # Initialize components
        self.memory = FinancialEvolutionMemory(
            persistence_path=self.config.get("memory", {}).get("persistence_path")
        )

        self.backtester = SurvivorshipBacktester(
            data_source=self.config.get("backtester", {}).get("data_source", "CRSP_SIMULATED"),
            include_delisted=self.config.get("backtester", {}).get("include_delisted", True),
            adjust_for_splits=self.config.get("backtester", {}).get("adjust_for_splits", True),
            adjust_for_dividends=self.config.get("backtester", {}).get("adjust_for_dividends", True)
        )

        # Initialize fitness with crisis periods
        crisis_periods = [
            ("2000-01-01", "2002-12-31", CrisisType.DOTCOM),
            ("2007-09-01", "2009-03-31", CrisisType.GFC),
            ("2020-02-01", "2020-04-30", CrisisType.COVID),
            ("2022-01-01", "2022-12-31", CrisisType.INFLATION)
        ]

        self.fitness = CrisisAwareFitness(
            crisis_periods=crisis_periods,
            memory=self.memory,
            config=self.config.get("fitness", {})
        )

        # Evolution state
        self.generation_count = 0
        self.best_strategies: List[Tuple[Strategy, BacktestResult, FitnessScore]] = []
        self.total_cost = 0.0

        logger.info("FinancialEvolutionAgent initialized")

    async def evolve_strategies(
        self,
        objective: EvolutionObjective,
        budget: EvolutionBudget
    ) -> FinancialEvolutionResult:
        """
        Main evolution loop using LoongFlow PES.

        Args:
            objective: Evolution objective
                - universe: Asset universe
                - crisis_periods: Crisis periods to test
                - survival_constraints: Survival constraints
            budget: Budget constraints
                - iterations: Max generations
                - cost_cap: Maximum LLM cost

        Returns:
            FinancialEvolutionResult with best strategies and lessons learned
        """
        start_time = datetime.utcnow()
        max_iterations = budget.iterations
        cost_cap = budget.cost_cap
        strategies_per_iteration = budget.strategies_per_iteration

        all_best_strategies = []
        all_lessons = []

        logger.info(f"Starting evolution: {max_iterations} iterations, ${cost_cap} budget")

        try:
            for iteration in range(max_iterations):
                self.generation_count = iteration

                # === PLAN PHASE (LoongFlow) ===
                plan = await self._plan_generation(iteration, strategies_per_iteration)

                # Check budget
                if self.total_cost + plan.estimated_cost > cost_cap:
                    logger.info(f"Budget cap reached at iteration {iteration}")
                    break

                # === EXECUTE PHASE (OpenEvolve) ===
                execution = await self._execute_strategies(plan, objective)

                # === SUMMARIZE PHASE (LoongFlow) ===
                summary = await self._summarize_results(execution, objective)

                # Store best strategies
                if execution.best_strategy:
                    all_best_strategies.append(execution.best_strategy)
                    self.best_strategies.append(execution.best_strategy)

                    # Add to memory
                    strategy, result, score = execution.best_strategy
                    self.memory.add_strategy_lineage(
                        parent_id=None,
                        child_id=strategy.strategy_id,
                        strategy_type=strategy.strategy_type,
                        metadata={"fitness": score.total_score, "iteration": iteration}
                    )

                # Store lessons
                all_lessons.extend(summary.lessons)
                for lesson in summary.lessons:
                    self.memory.store_lesson(lesson)

                # Update cost
                self.total_cost += summary.total_cost

                # Early termination if converged
                if summary.converged:
                    logger.info(f"Converged at iteration {iteration}")
                    break

                # Progress logging
                if iteration % 10 == 0:
                    best_fitness = all_best_strategies[-1][2].total_score if all_best_strategies else 0
                    logger.info(
                        f"Iteration {iteration}: "
                        f"Best fitness={best_fitness:.2f}, "
                        f"Cost=${self.total_cost:.2f}"
                    )

        except Exception as e:
            logger.error(f"Evolution failed: {e}", exc_info=True)
            raise

        # Calculate execution time
        execution_time = (datetime.utcnow() - start_time).total_seconds()

        # Extract best strategies
        final_best = [
            s[0] for s in sorted(
                self.best_strategies,
                key=lambda x: x[2].total_score,
                reverse=True
            )[:10]
        ]

        best_fitness = final_best[0].parameters.get("fitness", 0) if final_best else 0

        logger.info(f"Evolution complete: {len(final_best)} strategies, ${self.total_cost:.2f} cost")

        return FinancialEvolutionResult(
            best_strategies=final_best,
            lessons_learned=all_lessons,
            iterations_completed=self.generation_count + 1,
            final_cost=self.total_cost,
            converged=summary.converged if all_lessons else False,
            execution_time_seconds=execution_time,
            best_fitness=best_fitness
        )

    async def _plan_generation(
        self,
        iteration: int,
        n_strategies: int
    ) -> GenerationPlan:
        """
        LoongFlow PLAN phase - analyze failures, generate hypotheses.

        Args:
            iteration: Current iteration number
            n_strategies: Number of strategies to generate

        Returns:
            GenerationPlan with hypotheses and parameters
        """
        if iteration == 0:
            # First generation - broad exploration
            hypotheses = [
                "Momentum with 3-month lookback",
                "Momentum with 6-month lookback",
                "Momentum with 12-month lookback",
                "Mean reversion with 5-day lookback",
                "Mean reversion with 20-day lookback",
                "Value strategy (low P/E)",
                "Growth strategy (high earnings growth)",
                "Low volatility factor",
                "Quality factor combination",
                "Multi-factor momentum + value"
            ]

            parameter_ranges = {
                "lookback": (3, 12),
                "threshold": (0.01, 0.05),
                "alpha": (-0.02, 0.02),
                "beta": (0.5, 1.5)
            }

            rationale = "Initial broad exploration across strategy types"

        else:
            # Subsequent generations - learn from failures
            failures = self.memory.get_recent_failures(n=5)
            niche_representatives = self.memory.get_niche_representatives("crisis_survivors", n=5)

            # Generate targeted hypotheses based on failures
            hypotheses = []
            parameter_ranges = {}

            if failures:
                # Generate hypotheses to address failures
                for failure in failures[:3]:
                    if failure.failure_type == "excessive_drawdown":
                        hypotheses.append(f"Low volatility variant of {failure.strategy_id}")
                        parameter_ranges["beta"] = (0.3, 0.8)
                    elif failure.failure_type == "poor_crisis_survival":
                        hypotheses.append(f"Crisis-hedged {failure.strategy_id}")
                        parameter_ranges["alpha"] = (0.0, 0.05)

            # Add elite strategies from MAP-Elites archive
            for elite in niche_representatives:
                hypotheses.append(f"Variant of successful {elite.strategy_type}")

            parameter_ranges.update({
                "lookback": (3, 12),
                "threshold": (0.01, 0.05),
                "alpha": (-0.01, 0.03),
                "beta": (0.5, 1.2)
            })

            rationale = f"Targeted generation addressing {len(failures)} failure modes"

        # Estimate cost (simplified - in production use actual LLM pricing)
        estimated_cost = 0.5 + (len(hypotheses) * 0.1)

        return GenerationPlan(
            iteration=iteration,
            hypotheses=hypotheses[:n_strategies],
            parameter_ranges=parameter_ranges,
            estimated_cost=estimated_cost,
            rationale=rationale
        )

    async def _execute_strategies(
        self,
        plan: GenerationPlan,
        objective: EvolutionObjective
    ) -> GenerationExecution:
        """
        OpenEvolve EXECUTE phase - generate variants, backtest.

        Args:
            plan: Generation plan
            objective: Evolution objective

        Returns:
            GenerationExecution with results
        """
        strategies = []

        # Generate strategy variants from hypotheses
        for i, hypothesis in enumerate(plan.hypotheses):
            strategy = self._generate_strategy_from_hypothesis(
                hypothesis=hypothesis,
                parameter_ranges=plan.parameter_ranges,
                index=i
            )
            strategies.append(strategy)

        # Run backtests in parallel
        backtest_results = await self.backtester.run_parallel(
            strategies=strategies,
            period=f"{objective.crisis_periods[0][0].split('-')[0]}-01-01:2026-12-31",
            include_delisted=True
        )

        # Evaluate with crisis-aware fitness
        fitness_scores = []
        for result in backtest_results:
            score = self.fitness.evaluate(result)
            fitness_scores.append(score)

        # Rank strategies
        ranked_strategies = sorted(
            zip(strategies, backtest_results, fitness_scores),
            key=lambda x: x[2].total_score,
            reverse=True
        )

        best_strategy = ranked_strategies[0] if ranked_strategies else None
        worst_strategies = ranked_strategies[-5:] if len(ranked_strategies) >= 5 else []

        average_fitness = np.mean([s[2].total_score for s in ranked_strategies])

        return GenerationExecution(
            all_results=ranked_strategies,
            best_strategy=best_strategy,
            worst_strategies=worst_strategies,
            average_fitness=average_fitness,
            total_executed=len(strategies)
        )

    async def _summarize_results(
        self,
        execution: GenerationExecution,
        objective: EvolutionObjective
    ) -> GenerationSummary:
        """
        LoongFlow SUMMARIZE phase - reflect, store insights.

        Args:
            execution: Execution results
            objective: Evolution objective

        Returns:
            GenerationSummary with lessons and insights
        """
        lessons = []
        key_insights = []

        # Analyze failures
        for strategy, result, score in execution.worst_strategies:
            failure_type = self._classify_failure(result, score, objective)

            # Create failure lesson
            for crisis_period in objective.crisis_periods:
                crisis_type = crisis_period[2]

                lesson = self.fitness.update_lesson_from_result(
                    result=result,
                    crisis_type=crisis_type,
                    successful=False
                )

                lessons.append(lesson)

        # Analyze successes
        if execution.best_strategy:
            best_strategy, best_result, best_score = execution.best_strategy

            # Create success lessons for each crisis
            for crisis_period in objective.crisis_periods:
                crisis_type = crisis_period[2]

                # Only create lesson if strategy performed well
                if best_score.total_score > 0:
                    lesson = self.fitness.update_lesson_from_result(
                        result=best_result,
                        crisis_type=crisis_type,
                        successful=True
                    )

                    lessons.append(lesson)

            key_insights.append(
                f"Best strategy: {best_strategy.strategy_type} with "
                f"fitness={best_score.total_score:.2f}"
            )

        # Check convergence
        converged = self._check_convergence(execution)

        # Generate next steps
        next_steps = []
        if not converged:
            if execution.average_fitness < 1.0:
                next_steps.append("Focus on improving base fitness metrics")
            if execution.best_strategy and execution.best_strategy[2].base_score < 0:
                next_steps.append("Address high drawdown in best strategies")
            next_steps.append("Explore new parameter ranges based on failures")

        # Estimate actual cost (simplified)
        total_cost = execution.total_executed * 0.02  # $0.02 per backtest

        return GenerationSummary(
            lessons=lessons,
            converged=converged,
            total_cost=total_cost,
            key_insights=key_insights,
            next_steps=next_steps
        )

    def _generate_strategy_from_hypothesis(
        self,
        hypothesis: str,
        parameter_ranges: Dict[str, Tuple[float, float]],
        index: int
    ) -> Strategy:
        """
        Generate strategy from hypothesis text.

        Args:
            hypothesis: Natural language hypothesis
            parameter_ranges: Valid parameter ranges
            index: Strategy index

        Returns:
            Strategy object
        """
        # Parse hypothesis type
        if "momentum" in hypothesis.lower():
            strategy_type = StrategyType.MOMENTUM
        elif "mean reversion" in hypothesis.lower():
            strategy_type = StrategyType.MEAN_REVERSION
        elif "value" in hypothesis.lower():
            strategy_type = StrategyType.VALUE
        elif "growth" in hypothesis.lower():
            strategy_type = StrategyType.GROWTH
        elif "volatility" in hypothesis.lower():
            strategy_type = StrategyType.FACTOR_COMBINATION
        else:
            strategy_type = StrategyType.FACTOR_COMBINATION

        # Sample parameters from ranges
        parameters = {}
        if "lookback" in parameter_ranges:
            min_val, max_val = parameter_ranges["lookback"]
            parameters["lookback"] = int(np.random.uniform(min_val, max_val))

        if "threshold" in parameter_ranges:
            min_val, max_val = parameter_ranges["threshold"]
            parameters["threshold"] = float(np.random.uniform(min_val, max_val))

        if "alpha" in parameter_ranges:
            min_val, max_val = parameter_ranges["alpha"]
            parameters["alpha"] = float(np.random.uniform(min_val, max_val))

        if "beta" in parameter_ranges:
            min_val, max_val = parameter_ranges["beta"]
            parameters["beta"] = float(np.random.uniform(min_val, max_val))

        return Strategy(
            strategy_id=f"str_{self.generation_count}_{index}_{uuid.uuid4().hex[:8]}",
            strategy_type=strategy_type,
            parameters=parameters,
            description=hypothesis,
            entry_conditions=[],
            exit_conditions=[],
            risk_constraints={"max_position_size": 0.1}
        )

    def _classify_failure(
        self,
        result: BacktestResult,
        score: FitnessScore,
        objective: EvolutionObjective
    ) -> str:
        """Classify why a strategy failed"""
        if result.max_drawdown > objective.survival_constraints.max_drawdown:
            return "excessive_drawdown"
        elif result.final_wealth < objective.survival_constraints.min_equity_final:
            return "poor_returns"
        elif len(result.delistings) > 5:
            return "delisting_risk"
        elif score.base_score < 0:
            return "poor_base_fitness"
        else:
            return "general_failure"

    def _check_convergence(self, execution: GenerationExecution) -> bool:
        """
        Check if evolution has converged.

        Args:
            execution: Execution results

        Returns:
            True if converged
        """
        # Check if we have enough history
        if len(self.best_strategies) < 10:
            return False

        # Check if best fitness has improved in last 5 generations
        recent_best = [
            s[2].total_score
            for s in self.best_strategies[-5:]
        ]

        if len(recent_best) < 5:
            return False

        # Convergence: less than 1% improvement in last 5 generations
        improvement = (max(recent_best) - min(recent_best)) / abs(min(recent_best))
        return improvement < 0.01

    async def evaluate_strategy(
        self,
        strategy: Strategy,
        period: str = "2000-01-01:2026-12-31"
    ) -> Tuple[BacktestResult, FitnessScore]:
        """
        Evaluate a single strategy.

        Args:
            strategy: Strategy to evaluate
            period: Backtest period

        Returns:
            Tuple of (BacktestResult, FitnessScore)
        """
        result = await self.backtester.run(strategy, period)
        score = self.fitness.evaluate(result)

        return result, score
