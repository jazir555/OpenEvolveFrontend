"""
Liquidity Calculator
Calculate liquidity metrics and costs for treasury management

Author: AI Architecture Team
Date: 2026-01-30
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class LiquidityMetrics:
    """Liquidity metrics result"""
    liquidity_days: float
    annual_cost_bps: float
    liquidity_ratio: float
    concentration_risk: float
    stress_liquidity_days: float


class LiquidityCalculator:
    """
    Calculate liquidity metrics and costs

    Accounts for:
    - Cash: 100% available immediately
    - T-bills: 95% (5% price volatility haircut)
    - Commercial Paper: 50% (liquidity haircut in stress)
    - Credit Lines: 100% (but may be frozen in crisis)
    """

    # Risk-free rate (approximate current T-bill rate)
    RISK_FREE_RATE = 0.05  # 5% (500 bps)

    # Credit line costs
    COMMITMENT_FEE_BPS = 10  # bps on undrawn amount
    USAGE_RATE_BPS = 50  # bps over SOFR on drawn amount

    # Liquidity haircuts
    TBILL_HAIRCUT = 0.05  # 5% for price volatility
    CP_HAIRCUT_STRESS = 0.5  # 50% in stress conditions
    CP_HAIRCUT_NORMAL = 0.1  # 10% in normal conditions

    def calculate_liquidity_days(
        self,
        cash: float,
        t_bills: float,
        commercial_paper: float,
        credit_line_undrawn: float,
        daily_burn_rate: float,
        stress_mode: bool = True
    ) -> float:
        """
        Calculate days of liquidity.

        Args:
            cash: Cash balance
            t_bills: T-bill holdings
            commercial_paper: Commercial paper holdings
            credit_line_undrawn: Undrawn credit line capacity
            daily_burn_rate: Average daily cash outflow
            stress_mode: If True, use stress haircuts

        Returns:
            Days of liquidity

        Example:
            >>> calculator = LiquidityCalculator()
            >>> days = calculator.calculate_liquidity_days(
            ...     cash=100_000_000,
            ...     t_bills=50_000_000,
            ...     commercial_paper=30_000_000,
            ...     credit_line_undrawn=200_000_000,
            ...     daily_burn_rate=1_000_000
            ... )
            >>> print(f"Days of liquidity: {days:.1f}")
        """

        if daily_burn_rate <= 0:
            raise ValueError("daily_burn_rate must be positive")

        # Select haircuts based on mode
        cp_haircut = self.CP_HAIRCUT_STRESS if stress_mode else self.CP_HAIRCUT_NORMAL

        # Calculate liquid assets (with haircuts)
        liquid_assets = (
            cash +  # 100% available
            t_bills * (1 - self.TBILL_HAIRCUT) +  # 95% available
            commercial_paper * (1 - cp_haircut) +  # Variable
            credit_line_undrawn  # 100% available but may be frozen
        )

        days = liquid_assets / daily_burn_rate
        return days

    def calculate_annual_cost(
        self,
        cash: float,
        t_bills: float,
        commercial_paper: float,
        credit_line_total: float,
        credit_line_used: float,
        tbill_yield: float = 0.05,
        cp_yield: float = 0.06
    ) -> float:
        """
        Calculate annual cost of liquidity in bps.

        Drag from:
        - Cash: 0% yield vs T-bill rate (~400bps opportunity cost)
        - T-bills: T-bill rate (earns rate)
        - CP: CP rate (earns rate, usually higher than T-bills)
        - Credit line: Commitment fee (~10bps) + usage rate (~SOFR + 50)

        Args:
            cash: Cash balance
            t_bills: T-bill holdings
            commercial_paper: Commercial paper holdings
            credit_line_total: Total credit line capacity
            credit_line_used: Used credit line amount
            tbill_yield: Current T-bill yield (default 5%)
            cp_yield: Current commercial paper yield (default 6%)

        Returns:
            Annual cost in basis points (bps)

        Example:
            >>> cost_bps = calculator.calculate_annual_cost(
            ...     cash=100_000_000,
            ...     t_bills=50_000_000,
            ...     commercial_paper=30_000_000,
            ...     credit_line_total=200_000_000,
            ...     credit_line_used=20_000_000
            ... )
            >>> print(f"Annual cost: {cost_bps:.1f} bps")
        """

        # Calculate total liquid assets
        total_assets = cash + t_bills + commercial_paper

        if total_assets <= 0:
            return 0.0

        # Opportunity cost of cash (vs T-bills)
        cash_opportunity_cost = tbill_yield * 10000  # Convert to bps

        # Asset costs (bps)
        asset_cost = (
            (cash / total_assets) * cash_opportunity_cost +
            (t_bills / total_assets) * 0 +  # T-bills earn market rate
            (commercial_paper / total_assets) * ((cp_yield - tbill_yield) * 10000)  # CP earns premium
        )

        # Credit line costs
        if credit_line_total > 0:
            credit_usage_ratio = credit_line_used / credit_line_total

            # Commitment fee on undrawn
            commitment_cost = (1 - credit_usage_ratio) * self.COMMITMENT_FEE_BPS

            # Usage cost on drawn
            usage_cost = credit_usage_ratio * self.USAGE_RATE_BPS

            credit_cost = commitment_cost + usage_cost
        else:
            credit_cost = 0.0

        # Combined cost
        total_cost = asset_cost + credit_cost

        return total_cost

    def calculate_liquidity_ratio(
        self,
        current_assets: float,
        current_liabilities: float
    ) -> float:
        """
        Calculate current ratio (liquidity ratio).

        Args:
            current_assets: Current assets
            current_liabilities: Current liabilities

        Returns:
            Liquidity ratio (>= 2.0 is healthy)

        Example:
            >>> ratio = calculator.calculate_liquidity_ratio(
            ...     current_assets=500_000_000,
            ...     current_liabilities=200_000_000
            ... )
            >>> print(f"Liquidity ratio: {ratio:.2f}")
        """

        if current_liabilities <= 0:
            return float('inf')

        return current_assets / current_liabilities

    def calculate_concentration_risk(
        self,
        allocation: Dict[str, float]
    ) -> float:
        """
        Calculate concentration risk using Herfindahl-Hirschman Index (HHI).

        Args:
            allocation: Dictionary of asset class -> amount

        Returns:
            HHI score (lower is better, max 10,000)

        Example:
            >>> allocation = {
            ...     'cash': 100_000_000,
            ...     't_bills': 50_000_000,
            ...     'commercial_paper': 30_000_000
            ... }
            >>> hhi = calculator.calculate_concentration_risk(allocation)
            >>> print(f"Concentration risk (HHI): {hhi:.0f}")
        """

        total = sum(allocation.values())

        if total <= 0:
            return 0.0

        # Calculate market shares
        shares = [amount / total for amount in allocation.values()]

        # Calculate HHI (sum of squared shares * 10,000)
        hhi = sum(share ** 2 for share in shares) * 10000

        return hhi

    def calculate_stress_liquidity(
        self,
        cash: float,
        t_bills: float,
        commercial_paper: float,
        credit_line_undrawn: float,
        daily_burn_rate: float,
        cp_market_frozen: bool = False,
        credit_line_frozen: bool = False
    ) -> float:
        """
        Calculate liquidity days under stress conditions.

        Args:
            cash: Cash balance
            t_bills: T-bill holdings
            commercial_paper: Commercial paper holdings
            credit_line_undrawn: Undrawn credit line capacity
            daily_burn_rate: Average daily cash outflow
            cp_market_frozen: If True, CP is not liquidatable
            credit_line_frozen: If True, credit line is not accessible

        Returns:
            Days of liquidity under stress

        Example:
            >>> stress_days = calculator.calculate_stress_liquidity(
            ...     cash=100_000_000,
            ...     t_bills=50_000_000,
            ...     commercial_paper=30_000_000,
            ...     credit_line_undrawn=200_000_000,
            ...     daily_burn_rate=1_000_000,
            ...     cp_market_frozen=True,
            ...     credit_line_frozen=True
            ... )
            >>> print(f"Stress liquidity days: {stress_days:.1f}")
        """

        if daily_burn_rate <= 0:
            raise ValueError("daily_burn_rate must be positive")

        # Calculate liquid assets under stress
        liquid_assets = cash  # Always available

        # T-bills: 95% available (5% haircut)
        liquid_assets += t_bills * (1 - self.TBILL_HAIRCUT)

        # Commercial paper: only available if market not frozen
        if not cp_market_frozen:
            liquid_assets += commercial_paper * (1 - self.CP_HAIRCUT_STRESS)
        # else: CP is worth 0 in frozen market

        # Credit line: only available if not frozen
        if not credit_line_frozen:
            liquid_assets += credit_line_undrawn
        # else: Credit line is not accessible

        days = liquid_assets / daily_burn_rate
        return days

    def calculate_comprehensive_metrics(
        self,
        cash: float,
        t_bills: float,
        commercial_paper: float,
        credit_line_total: float,
        credit_line_used: float,
        daily_burn_rate: float,
        current_assets: float,
        current_liabilities: float,
        stress_scenario: Optional[Dict[str, bool]] = None
    ) -> LiquidityMetrics:
        """
        Calculate all liquidity metrics.

        Args:
            cash: Cash balance
            t_bills: T-bill holdings
            commercial_paper: Commercial paper holdings
            credit_line_total: Total credit line capacity
            credit_line_used: Used credit line amount
            daily_burn_rate: Average daily cash outflow
            current_assets: Current assets
            current_liabilities: Current liabilities
            stress_scenario: Optional dict with 'cp_market_frozen' and 'credit_line_frozen'

        Returns:
            LiquidityMetrics object with all calculated metrics

        Example:
            >>> metrics = calculator.calculate_comprehensive_metrics(
            ...     cash=100_000_000,
            ...     t_bills=50_000_000,
            ...     commercial_paper=30_000_000,
            ...     credit_line_total=200_000_000,
            ...     credit_line_used=20_000_000,
            ...     daily_burn_rate=1_000_000,
            ...     current_assets=500_000_000,
            ...     current_liabilities=200_000_000
            ... )
            >>> print(f"Liquidity days: {metrics.liquidity_days:.1f}")
            >>> print(f"Annual cost: {metrics.annual_cost_bps:.1f} bps")
        """

        credit_line_undrawn = credit_line_total - credit_line_used

        # Normal liquidity days
        liquidity_days = self.calculate_liquidity_days(
            cash=cash,
            t_bills=t_bills,
            commercial_paper=commercial_paper,
            credit_line_undrawn=credit_line_undrawn,
            daily_burn_rate=daily_burn_rate,
            stress_mode=False
        )

        # Annual cost
        annual_cost_bps = self.calculate_annual_cost(
            cash=cash,
            t_bills=t_bills,
            commercial_paper=commercial_paper,
            credit_line_total=credit_line_total,
            credit_line_used=credit_line_used
        )

        # Liquidity ratio
        liquidity_ratio = self.calculate_liquidity_ratio(
            current_assets=current_assets,
            current_liabilities=current_liabilities
        )

        # Concentration risk
        allocation = {
            'cash': cash,
            't_bills': t_bills,
            'commercial_paper': commercial_paper
        }
        concentration_risk = self.calculate_concentration_risk(allocation)

        # Stress liquidity days
        if stress_scenario:
            stress_liquidity_days = self.calculate_stress_liquidity(
                cash=cash,
                t_bills=t_bills,
                commercial_paper=commercial_paper,
                credit_line_undrawn=credit_line_undrawn,
                daily_burn_rate=daily_burn_rate,
                cp_market_frozen=stress_scenario.get('cp_market_frozen', False),
                credit_line_frozen=stress_scenario.get('credit_line_frozen', False)
            )
        else:
            # Default stress: both frozen
            stress_liquidity_days = self.calculate_stress_liquidity(
                cash=cash,
                t_bills=t_bills,
                commercial_paper=commercial_paper,
                credit_line_undrawn=credit_line_undrawn,
                daily_burn_rate=daily_burn_rate,
                cp_market_frozen=True,
                credit_line_frozen=True
            )

        return LiquidityMetrics(
            liquidity_days=liquidity_days,
            annual_cost_bps=annual_cost_bps,
            liquidity_ratio=liquidity_ratio,
            concentration_risk=concentration_risk,
            stress_liquidity_days=stress_liquidity_days
        )

    def validate_liquidity_constraints(
        self,
        metrics: LiquidityMetrics,
        min_liquidity_days: float,
        max_cost_bps: float,
        min_liquidity_ratio: float = 2.0
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Validate liquidity metrics against constraints.

        Args:
            metrics: LiquidityMetrics to validate
            min_liquidity_days: Minimum required liquidity days
            max_cost_bps: Maximum acceptable annual cost (bps)
            min_liquidity_ratio: Minimum acceptable liquidity ratio

        Returns:
            (is_valid, validation_details)

        Example:
            >>> is_valid, details = calculator.validate_liquidity_constraints(
            ...     metrics=metrics,
            ...     min_liquidity_days=90,
            ...     max_cost_bps=50
            ... )
            >>> if not is_valid:
            ...     print("Validation failed:", details)
        """

        violations = {}

        # Check liquidity days
        if metrics.liquidity_days < min_liquidity_days:
            violations['liquidity_days'] = {
                'required': min_liquidity_days,
                'actual': metrics.liquidity_days,
                'shortfall': min_liquidity_days - metrics.liquidity_days
            }

        # Check cost
        if metrics.annual_cost_bps > max_cost_bps:
            violations['annual_cost'] = {
                'maximum': max_cost_bps,
                'actual': metrics.annual_cost_bps,
                'excess': metrics.annual_cost_bps - max_cost_bps
            }

        # Check liquidity ratio
        if metrics.liquidity_ratio < min_liquidity_ratio:
            violations['liquidity_ratio'] = {
                'minimum': min_liquidity_ratio,
                'actual': metrics.liquidity_ratio,
                'shortfall': min_liquidity_ratio - metrics.liquidity_ratio
            }

        is_valid = len(violations) == 0

        return is_valid, {
            'violations': violations,
            'metrics': {
                'liquidity_days': metrics.liquidity_days,
                'annual_cost_bps': metrics.annual_cost_bps,
                'liquidity_ratio': metrics.liquidity_ratio,
                'concentration_risk': metrics.concentration_risk,
                'stress_liquidity_days': metrics.stress_liquidity_days
            }
        }
