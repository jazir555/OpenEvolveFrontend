"""
Liquidity Crisis Evolver
Evolve liquidity management strategies for corporate treasury

Author: AI Architecture Team
Date: 2026-01-30
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import asyncio

from .liquidity_calculator import LiquidityCalculator, LiquidityMetrics
from .scenario_generator import LiquidityScenarioGenerator, LiquidityScenario, ScenarioType


@dataclass
class CashFlowProfile:
    """
    Cash flow profile for a company

    Attributes:
        daily_burn_rate: Average daily cash outflow
        volatility_std: Standard deviation of daily flows
        seasonal_patterns: Monthly variations (dict: month -> multiplier)
        capex_schedule: List of large one-time outflows
    """
    daily_burn_rate: float
    volatility_std: float
    seasonal_patterns: Dict[str, float] = field(default_factory=dict)
    capex_schedule: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LiquidityConstraints:
    """
    Liquidity constraints

    Attributes:
        min_liquidity_days: Minimum days of liquidity required
        max_liquidity_cost: Maximum annual drag in bps
        max_drawdown_credit_line: Maximum fraction of credit line usable
        min_liquidity_ratio: Minimum current ratio (assets/liabilities)
    """
    min_liquidity_days: float
    max_liquidity_cost: float
    max_drawdown_credit_line: float = 0.5
    min_liquidity_ratio: float = 2.0


@dataclass
class LiquidityAllocation:
    """
    Liquidity allocation strategy

    Attributes:
        cash: Cash balance
        t_bills: T-bill holdings
        commercial_paper: Commercial paper holdings
        credit_line_total: Total credit line capacity
        credit_line_available: Undrawn credit line
        policy_rules: Optional policy rules (rebalancing triggers, etc.)
    """
    cash: float
    t_bills: float
    commercial_paper: float
    credit_line_total: float
    credit_line_available: float
    policy_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LiquiditySimulationResult:
    """
    Result of simulating a liquidity allocation through a scenario

    Attributes:
        success: Whether allocation survived the scenario
        default_day: Day of default (None if survived)
        min_liquidity_days: Minimum liquidity days during scenario
        max_credit_line_usage: Maximum credit line usage fraction
        final_liquidity_days: Liquidity days at end of scenario
        annual_cost_bps: Annual cost of allocation in bps
        liquidity_history: Daily liquidity days
        credit_usage_history: Daily credit line usage
    """
    success: bool
    default_day: Optional[int]
    min_liquidity_days: float
    max_credit_line_usage: float
    final_liquidity_days: float
    annual_cost_bps: float
    liquidity_history: List[float] = field(default_factory=list)
    credit_usage_history: List[float] = field(default_factory=list)


@dataclass
class LiquidityEvolutionResult:
    """
    Result of liquidity strategy evolution

    Attributes:
        strategy: Best liquidity allocation found
        liquidity_days: Normal liquidity days
        stress_liquidity_days: Minimum liquidity days across all stress scenarios
        annual_cost: Annual cost in bps
        credit_line_usage: Maximum credit line usage
        stress_test_results: Results for each stress scenario
        scenario_names: Names of scenarios tested
        robustness_score: Score indicating strategy robustness
    """
    strategy: LiquidityAllocation
    liquidity_days: float
    stress_liquidity_days: float
    annual_cost: float
    credit_line_usage: float
    stress_test_results: Dict[str, LiquiditySimulationResult]
    scenario_names: List[str]
    robustness_score: float


class LiquidityCrisisEvolver:
    """
    Evolve liquidity management strategies that survive crises

    Objectives:
    - Maintain 90 days liquidity through GFC-like scenarios
    - Minimize cost of liquidity (drag on returns)
    - Avoid emergency borrowing (credit line freezes)

    Assets:
    - Cash (0% yield, immediate liquidity)
    - T-bills (risk-free, 1-7 day liquidity)
    - Commercial paper (higher yield, liquidity risk)
    - Credit lines (contingent, freeze risk)

    Example:
        >>> evolver = LiquidityCrisisEvolver(config={})
        >>> result = await evolver.evolve_liquidity_strategy(
        ...     cash_flow_profile=CashFlowProfile(
        ...         daily_burn_rate=1_000_000,
        ...         volatility_std=200_000
        ...     ),
        ...     constraints=LiquidityConstraints(
        ...         min_liquidity_days=90,
        ...         max_liquidity_cost=50
        ...     )
        ... )
        >>> print(f"Liquidity days: {result.liquidity_days:.1f}")
        >>> print(f"Annual cost: {result.annual_cost:.1f} bps")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize liquidity crisis evolver

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize components
        self.liquidity_calculator = LiquidityCalculator()
        self.scenario_generator = LiquidityScenarioGenerator()

        # Evolution parameters
        self.n_variants = self.config.get('n_variants', 100)
        self.n_top_candidates = self.config.get('n_top_candidates', 10)

    async def evolve_liquidity_strategy(
        self,
        cash_flow_profile: CashFlowProfile,
        constraints: LiquidityConstraints,
        scenarios: Optional[List[LiquidityScenario]] = None
    ) -> LiquidityEvolutionResult:
        """
        Evolve liquidity strategy that survives stress scenarios

        Args:
            cash_flow_profile: Company's cash flow profile
            constraints: Liquidity constraints
            scenarios: Optional list of scenarios (default: all standard scenarios)

        Returns:
            LiquidityEvolutionResult with best strategy found

        Example:
            >>> result = await evolver.evolve_liquidity_strategy(
            ...     cash_flow_profile=CashFlowProfile(
            ...         daily_burn_rate=1_000_000,
            ...         volatility_std=200_000
            ...     ),
            ...     constraints=LiquidityConstraints(
            ...         min_liquidity_days=90,
            ...         max_liquidity_cost=50
            ...     )
            ... )
        """

        # Generate scenarios if not provided
        if scenarios is None:
            scenarios = self.scenario_generator.generate_all_scenarios()

        # Generate initial allocation variants
        allocations = self._generate_allocation_variants(
            cash_flow_profile=cash_flow_profile,
            constraints=constraints,
            n_variants=self.n_variants
        )

        # Evaluate each allocation across all scenarios
        allocation_scores = []
        allocation_results = {}

        for allocation in allocations:
            # Simulate across all scenarios
            scenario_results = {}
            scenario_survived = []

            for scenario in scenarios:
                result = await self._simulate_liquidity(
                    allocation=allocation,
                    scenario=scenario,
                    profile=cash_flow_profile
                )
                scenario_results[scenario.name] = result
                scenario_survived.append(result.success)

            # Calculate score
            score = self._calculate_allocation_score(
                allocation=allocation,
                scenario_results=scenario_results,
                constraints=constraints,
                profile=cash_flow_profile
            )

            allocation_scores.append((allocation, score, scenario_results))
            allocation_results[id(allocation)] = {
                'score': score,
                'scenario_results': scenario_results
            }

        # Rank by score
        allocation_scores.sort(key=lambda x: x[1], reverse=True)

        # Get best allocation
        best_allocation, best_score, best_scenario_results = allocation_scores[0]

        # Calculate comprehensive metrics
        metrics = self.liquidity_calculator.calculate_comprehensive_metrics(
            cash=best_allocation.cash,
            t_bills=best_allocation.t_bills,
            commercial_paper=best_allocation.commercial_paper,
            credit_line_total=best_allocation.credit_line_total,
            credit_line_used=best_allocation.credit_line_total - best_allocation.credit_line_available,
            daily_burn_rate=cash_flow_profile.daily_burn_rate,
            current_assets=sum([
                best_allocation.cash,
                best_allocation.t_bills,
                best_allocation.commercial_paper
            ]),
            current_liabilities=cash_flow_profile.daily_burn_rate * 30  # Approximate
        )

        # Calculate robustness score
        robustness_score = self._calculate_robustness_score(best_scenario_results)

        return LiquidityEvolutionResult(
            strategy=best_allocation,
            liquidity_days=metrics.liquidity_days,
            stress_liquidity_days=metrics.stress_liquidity_days,
            annual_cost=metrics.annual_cost_bps,
            credit_line_usage=1.0 - (best_allocation.credit_line_available / best_allocation.credit_line_total),
            stress_test_results=best_scenario_results,
            scenario_names=[s.name for s in scenarios],
            robustness_score=robustness_score
        )

    def _generate_allocation_variants(
        self,
        cash_flow_profile: CashFlowProfile,
        constraints: LiquidityConstraints,
        n_variants: int = 100
    ) -> List[LiquidityAllocation]:
        """
        Generate allocation variants

        Args:
            cash_flow_profile: Cash flow profile
            constraints: Liquidity constraints
            n_variants: Number of variants to generate

        Returns:
            List of allocation variants
        """

        # Calculate target liquidity needed
        target_liquidity = cash_flow_profile.daily_burn_rate * constraints.min_liquidity_days

        # Credit line capacity (usually 3x daily burn)
        credit_line_total = cash_flow_profile.daily_burn_rate * 90

        allocations = []

        for i in range(n_variants):
            # Generate random allocation weights
            # Use different strategies for different variants

            if i < n_variants // 3:
                # Conservative: More cash, less CP
                cash_weight = np.random.uniform(0.5, 0.8)
                tbill_weight = np.random.uniform(0.2, 0.5)
                cp_weight = 1.0 - cash_weight - tbill_weight

            elif i < 2 * n_variants // 3:
                # Balanced
                cash_weight = np.random.uniform(0.3, 0.5)
                tbill_weight = np.random.uniform(0.3, 0.5)
                cp_weight = 1.0 - cash_weight - tbill_weight

            else:
                # Aggressive: Less cash, more CP
                cash_weight = np.random.uniform(0.2, 0.4)
                tbill_weight = np.random.uniform(0.3, 0.6)
                cp_weight = 1.0 - cash_weight - tbill_weight

            # Ensure non-negative
            cp_weight = max(0, cp_weight)

            # Calculate absolute amounts
            cash = target_liquidity * cash_weight
            t_bills = target_liquidity * tbill_weight
            commercial_paper = target_liquidity * cp_weight

            # Credit line starts fully available
            credit_line_available = credit_line_total

            allocations.append(LiquidityAllocation(
                cash=cash,
                t_bills=t_bills,
                commercial_paper=commercial_paper,
                credit_line_total=credit_line_total,
                credit_line_available=credit_line_available,
                policy_rules={}
            ))

        return allocations

    async def _simulate_liquidity(
        self,
        allocation: LiquidityAllocation,
        scenario: LiquidityScenario,
        profile: CashFlowProfile
    ) -> LiquiditySimulationResult:
        """
        Simulate liquidity through a scenario

        Args:
            allocation: Liquidity allocation
            scenario: Stress scenario
            profile: Cash flow profile

        Returns:
            LiquiditySimulationResult
        """

        # Initialize state
        cash = allocation.cash
        t_bills = allocation.t_bills
        commercial_paper = allocation.commercial_paper
        credit_line_available = allocation.credit_line_available

        liquidity_history = []
        credit_usage_history = []

        # Simulate day by day
        for day in range(scenario.duration_days):
            # Generate cash flow for day
            base_outflow = profile.daily_burn_rate * scenario.daily_outflow[day]

            # Add some randomness
            noise = np.random.normal(0, profile.volatility_std)
            outflow = max(0, base_outflow + noise)

            # Try to fund from most liquid to least liquid

            # 1. Use cash first
            if cash >= outflow:
                cash -= outflow
                outflow = 0
            else:
                outflow -= cash
                cash = 0

            # 2. Sell T-bills (1 day to settle)
            # Can use tomorrow, but we need to fund today
            # So we need to use credit line or CP for today's gap
            if outflow > 0 and t_bills > 0:
                # We'll sell T-bills, but they settle tomorrow
                # For today, we still have a gap
                tbill_sell = min(t_bills, outflow * 1.05)  # Sell 5% extra for settlement
                t_bills -= tbill_sell
                # Cash available tomorrow, not today

            # 3. Use commercial paper (if market not frozen)
            if outflow > 0 and commercial_paper > 0:
                if not scenario.cp_market_frozen[day]:
                    cp_sell = min(commercial_paper, outflow)
                    commercial_paper -= cp_sell
                    cash += cp_sell  # Available today
                    outflow = max(0, outflow - cp_sell)
                else:
                    # CP market frozen - can't sell
                    pass

            # 4. Draw credit line (if not frozen)
            if outflow > 0 and credit_line_available > 0:
                if not scenario.credit_line_frozen[day]:
                    draw = min(credit_line_available, outflow)
                    credit_line_available -= draw
                    cash += draw
                    outflow = max(0, outflow - draw)
                else:
                    # Credit line frozen
                    pass

            # Check if insolvent
            if cash < 0 or outflow > 0:
                # Default!
                return LiquiditySimulationResult(
                    success=False,
                    default_day=day,
                    min_liquidity_days=0,
                    max_credit_line_usage=1.0,
                    final_liquidity_days=0,
                    annual_cost_bps=0,
                    liquidity_history=liquidity_history,
                    credit_usage_history=credit_usage_history
                )

            # Calculate liquidity days
            liquidity_days = self.liquidity_calculator.calculate_liquidity_days(
                cash=cash,
                t_bills=t_bills,
                commercial_paper=commercial_paper,
                credit_line_undrawn=credit_line_available,
                daily_burn_rate=profile.daily_burn_rate,
                stress_mode=True
            )

            liquidity_history.append(liquidity_days)

            # Calculate credit line usage
            credit_usage = 1.0 - (credit_line_available / allocation.credit_line_total)
            credit_usage_history.append(credit_usage)

        # Calculate annual cost
        annual_cost = self.liquidity_calculator.calculate_annual_cost(
            cash=allocation.cash,
            t_bills=allocation.t_bills,
            commercial_paper=allocation.commercial_paper,
            credit_line_total=allocation.credit_line_total,
            credit_line_used=allocation.credit_line_total - allocation.credit_line_available
        )

        return LiquiditySimulationResult(
            success=True,
            default_day=None,
            min_liquidity_days=min(liquidity_history),
            max_credit_line_usage=max(credit_usage_history),
            final_liquidity_days=liquidity_history[-1],
            annual_cost_bps=annual_cost,
            liquidity_history=liquidity_history,
            credit_usage_history=credit_usage_history
        )

    def _calculate_allocation_score(
        self,
        allocation: LiquidityAllocation,
        scenario_results: Dict[str, LiquiditySimulationResult],
        constraints: LiquidityConstraints,
        profile: CashFlowProfile
    ) -> float:
        """
        Calculate score for an allocation

        Args:
            allocation: Liquidity allocation
            scenario_results: Results for each scenario
            constraints: Liquidity constraints
            profile: Cash flow profile

        Returns:
            Score (higher is better)
        """

        # Check if all scenarios survived
        all_survived = all(result.success for result in scenario_results.values())

        if not all_survived:
            return -1000.0  # Heavy penalty for failing any scenario

        # Calculate metrics
        min_liquidity_days = min(result.min_liquidity_days for result in scenario_results.values())
        max_credit_usage = max(result.max_credit_line_usage for result in scenario_results.values())
        avg_annual_cost = np.mean([result.annual_cost_bps for result in scenario_results.values()])

        # Liquidity score (days maintained)
        liquidity_score = min(
            min_liquidity_days / constraints.min_liquidity_days,
            1.0
        )

        # Cost score (lower is better)
        cost_score = 1.0 - (
            avg_annual_cost / constraints.max_liquidity_cost
        )

        # Credit line penalty (avoid if possible)
        credit_penalty = max_credit_usage * 0.3

        # Combined score
        score = (
            liquidity_score * 0.5 +
            cost_score * 0.3 -
            credit_penalty
        )

        # Additional penalty if credit line exceeds limit
        if max_credit_usage > constraints.max_drawdown_credit_line:
            score -= 0.5

        return score

    def _calculate_robustness_score(
        self,
        scenario_results: Dict[str, LiquiditySimulationResult]
    ) -> float:
        """
        Calculate robustness score

        Args:
            scenario_results: Results for each scenario

        Returns:
            Robustness score (0-1, higher is better)
        """

        if not scenario_results:
            return 0.0

        # Check survival rate
        survival_rate = sum(1 for r in scenario_results.values() if r.success) / len(scenario_results)

        # Average minimum liquidity days
        avg_min_liquidity = np.mean([r.min_liquidity_days for r in scenario_results.values()])

        # Robustness score combines survival and liquidity buffer
        robustness = survival_rate * min(avg_min_liquidity / 90.0, 1.0)

        return robustness
