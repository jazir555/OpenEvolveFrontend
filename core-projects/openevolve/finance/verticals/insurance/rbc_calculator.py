"""
Risk-Based Capital (RBC) Calculator for Insurance

Implements NAIC standard RBC calculation methodology.

The RBC ratio measures an insurance company's capital adequacy by comparing
total adjusted capital to the risk-based capital required to support the
company's risk profile.

RBC Formula:
- RBC Ratio = (Total Adjusted Capital / RBC Required) * 100
- RBC Required = sqrt(C0² + C1² + C2² + C3² + C4² + covariance adjustments)

Where:
- C0: Affiliate risk (subsidiaries and affiliates)
- C1: Fixed income risk (asset risk - bonds)
- C2: Equity risk (common and preferred stock)
- C3: Real estate risk
- C4: Off-balance sheet risk (derivatives, etc.)

Regulatory Requirement: 350% minimum RBC ratio

Author: AI Architecture Team
Date: 2026-01-30
"""

import math
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .models import Portfolio, Bond, CreditRating, RBCCalculationResult


class RBCCalculator:
    """
    Calculate Risk-Based Capital ratios per NAIC standards.

    This calculator implements the NAIC RBC formula for property & casualty
    and life insurance companies. It considers multiple risk categories and
    their correlations.

    Example:
        >>> calculator = RBCCalculator()
        >>> result = calculator.calculate(
        ...     portfolio_value=1_500_000_000,
        ...     liabilities=1_000_000_000,
        ...     portfolio=portfolio
        ... )
        >>> print(f"RBC Ratio: {result.rbc_ratio:.2f}%")
        >>> print(f"Compliant: {result.compliant}")
    """

    # NAIC risk factors by credit rating (simplified)
    RISK_FACTORS = {
        CreditRating.AAA: 0.004,   # 0.4% C1 charge
        CreditRating.AA: 0.006,    # 0.6%
        CreditRating.A: 0.012,     # 1.2%
        CreditRating.BBB: 0.028,   # 2.8%
        CreditRating.BB: 0.080,    # 8.0%
        CreditRating.B: 0.200,     # 20.0%
        CreditRating.CCC: 0.400,   # 40.0%
    }

    # Asset valuation reserve factor (for bonds below par)
    AVR_FACTOR = 1.0

    # Covariance factors for NAIC RBC formula
    COVARIANCE_FACTOR = 0.25

    def __init__(self):
        """Initialize RBC calculator"""
        self.minimum_rbc_ratio = 350.0  # Regulatory minimum

    def calculate(
        self,
        portfolio_value: float,
        liabilities: float,
        portfolio: Optional[Portfolio] = None
    ) -> float:
        """
        Calculate RBC ratio.

        This is a simplified version that returns the ratio as a percentage.
        For detailed breakdown, use calculate_detailed() instead.

        Args:
            portfolio_value: Total market value of portfolio
            liabilities: Total policy liabilities
            portfolio: Portfolio object (optional, for detailed calculation)

        Returns:
            RBC ratio as percentage (e.g., 350 = 350%)

        Example:
            >>> ratio = calculator.calculate(
            ...     portfolio_value=1_500_000_000,
            ...     liabilities=1_000_000_000,
            ...     portfolio=my_portfolio
            ... )
            >>> print(f"RBC: {ratio:.2f}%")
        """
        result = self.calculate_detailed(portfolio_value, liabilities, portfolio)
        return result.rbc_ratio

    def calculate_detailed(
        self,
        portfolio_value: float,
        liabilities: float,
        portfolio: Optional[Portfolio] = None
    ) -> RBCCalculationResult:
        """
        Calculate detailed RBC breakdown.

        Args:
            portfolio_value: Total market value of portfolio
            liabilities: Total policy liabilities
            portfolio: Portfolio object (optional, for detailed calculation)

        Returns:
            RBCCalculationResult with detailed breakdown

        Example:
            >>> result = calculator.calculate_detailed(
            ...     portfolio_value=1_500_000_000,
            ...     liabilities=1_000_000_000,
            ...     portfolio=my_portfolio
            ... )
            >>> print(f"C1 Risk: ${result.c1_risk:,.0f}")
            >>> print(f"Total RBC: ${result.rbc_required:,.0f}")
        """
        # Calculate Total Adjusted Capital (TAC)
        tac = self._calculate_tac(portfolio_value, portfolio)

        # Calculate RBC risk components
        c0 = self._calculate_c0_risk(portfolio)  # Affiliates
        c1 = self._calculate_c1_risk(portfolio)  # Fixed income
        c2 = self._calculate_c2_risk(portfolio)  # Equity
        c3 = self._calculate_c3_risk(portfolio)  # Real estate
        c4 = self._calculate_c4_risk(portfolio)  # Off-balance sheet

        # Calculate RBC required with covariance adjustment
        rbc_required = self._calculate_rbc_with_covariance(c0, c1, c2, c3, c4)

        # Calculate RBC ratio
        rbc_ratio = (tac / rbc_required * 100) if rbc_required > 0 else 0

        # Check compliance
        compliant = rbc_ratio >= self.minimum_rbc_ratio

        # Compile result
        result = RBCCalculationResult(
            tac=tac,
            rbc_required=rbc_required,
            rbc_ratio=rbc_ratio,
            c0_risk=c0,
            c1_risk=c1,
            c2_risk=c2,
            c3_risk=c3,
            c4_risk=c4,
            compliant=compliant,
            details={
                "liabilities": liabilities,
                "portfolio_value": portfolio_value,
                "minimum_rbc_ratio": self.minimum_rbc_ratio,
                "capital_surplus": tac - rbc_required,
                "action_level": self._get_action_level(rbc_ratio)
            }
        )

        return result

    def _calculate_tac(
        self,
        portfolio_value: float,
        portfolio: Optional[Portfolio] = None
    ) -> float:
        """
        Calculate Total Adjusted Capital.

        TAC = Capital and Surplus - Asset Valuation Reserve (AVR)

        The AVR adjusts for assets carried below book value (e.g., bonds
        purchased at a premium that have amortized down).

        Args:
            portfolio_value: Total market value of portfolio
            portfolio: Portfolio object (optional)

        Returns:
            Total Adjusted Capital
        """
        # Start with portfolio value
        capital = portfolio_value

        # Calculate Asset Valuation Reserve (AVR)
        avr = 0.0
        if portfolio:
            for bond in portfolio.bonds:
                if bond.market_value < bond.book_value:
                    # Bond is below book value, add to AVR
                    avr += (bond.book_value - bond.market_value) * self.AVR_FACTOR

        # TAC = Capital - AVR
        tac = capital - avr

        return tac

    def _calculate_c0_risk(self, portfolio: Optional[Portfolio]) -> float:
        """
        Calculate C0 risk - Affiliate risk.

        C0 risk applies to investments in subsidiaries and affiliates.
        For a simple bond portfolio, this is typically zero.

        Args:
            portfolio: Portfolio object

        Returns:
            C0 risk charge
        """
        # For bond portfolios, typically no affiliate risk
        return 0.0

    def _calculate_c1_risk(self, portfolio: Optional[Portfolio]) -> float:
        """
        Calculate C1 risk - Fixed income risk.

        C1 risk is based on credit quality of fixed income investments.
        Higher credit risk = higher C1 charge.

        Risk factors (simplified NAIC):
        - AAA: 0.4%
        - AA: 0.6%
        - A: 1.2%
        - BBB: 2.8%
        - BB: 8.0%
        - B: 20.0%

        Args:
            portfolio: Portfolio object

        Returns:
            C1 risk charge
        """
        if not portfolio or not portfolio.bonds:
            return 0.0

        c1_charge = 0.0

        for bond in portfolio.bonds:
            # Get risk factor for bond's rating
            risk_factor = self.RISK_FACTORS.get(bond.rating, 0.028)  # Default to BBB

            # Calculate charge for this bond
            charge = bond.market_value * risk_factor
            c1_charge += charge

        return c1_charge

    def _calculate_c2_risk(self, portfolio: Optional[Portfolio]) -> float:
        """
        Calculate C2 risk - Equity risk.

        C2 risk applies to common and preferred stocks.
        For bond portfolios, this is typically minimal.

        Args:
            portfolio: Portfolio object

        Returns:
            C2 risk charge
        """
        # Assume 10% allocation to equities for insurance portfolios
        # C2 charge is typically 15-30% of equity value
        if portfolio:
            equity_allocation = portfolio.total_value * 0.10
            return equity_allocation * 0.20  # 20% charge
        return 0.0

    def _calculate_c3_risk(self, portfolio: Optional[Portfolio]) -> float:
        """
        Calculate C3 risk - Real estate risk.

        C3 risk applies to real estate investments.
        For bond portfolios, this is typically zero.

        Args:
            portfolio: Portfolio object

        Returns:
            C3 risk charge
        """
        # Bond portfolios typically don't hold real estate
        return 0.0

    def _calculate_c4_risk(self, portfolio: Optional[Portfolio]) -> float:
        """
        Calculate C4 risk - Off-balance sheet risk.

        C4 risk applies to derivatives, letters of credit, etc.
        For plain vanilla bond portfolios, this is minimal.

        Args:
            portfolio: Portfolio object

        Returns:
            C4 risk charge
        """
        # Assume minimal off-balance sheet exposure
        # for plain vanilla bond portfolios
        return 0.0

    def _calculate_rbc_with_covariance(
        self,
        c0: float,
        c1: float,
        c2: float,
        c3: float,
        c4: float
    ) -> float:
        """
        Calculate total RBC required with covariance adjustment.

        NAIC Formula (simplified):
        RBC = sqrt(C0² + C1² + C2² + C3² + C4² +
                   2 * 0.25 * (C0*C1 + C0*C2 + C0*C3 + C0*C4 +
                               C1*C2 + C1*C3 + C1*C4 +
                               C2*C3 + C2*C4 +
                               C3*C4))

        The covariance factor (0.25) accounts for less-than-perfect
        correlation between risk types.

        Args:
            c0: C0 risk charge
            c1: C1 risk charge
            c2: C2 risk charge
            c3: C3 risk charge
            c4: C4 risk charge

        Returns:
            Total RBC required
        """
        # Squared components
        squared_sum = (c0 ** 2 + c1 ** 2 + c2 ** 2 + c3 ** 2 + c4 ** 2)

        # Covariance adjustments
        covariance = 2 * self.COVARIANCE_FACTOR * (
            c0 * c1 + c0 * c2 + c0 * c3 + c0 * c4 +
            c1 * c2 + c1 * c3 + c1 * c4 +
            c2 * c3 + c2 * c4 +
            c3 * c4
        )

        # Total RBC
        rbc = math.sqrt(squared_sum + covariance)

        return rbc

    def _get_action_level(self, rbc_ratio: float) -> str:
        """
        Get regulatory action level based on RBC ratio.

        NAIC Action Levels:
        - Company Action Level: 200-250%
        - Regulatory Action Level: 150-200%
        - Authorized Control Level: 100-150%
        - Mandatory Control Level: < 100%

        Args:
            rbc_ratio: RBC ratio as percentage

        Returns:
            Action level description
        """
        if rbc_ratio >= 350:
            return "Compliant (No Action Required)"
        elif rbc_ratio >= 250:
            return "Monitoring Zone (250-350%)"
        elif rbc_ratio >= 200:
            return "Company Action Level (200-250%)"
        elif rbc_ratio >= 150:
            return "Regulatory Action Level (150-200%)"
        elif rbc_ratio >= 100:
            return "Authorized Control Level (100-150%)"
        else:
            return "Mandatory Control Level (<100%)"

    def calculate_capital_required(
        self,
        liabilities: float,
        target_rbc_ratio: float = 350.0
    ) -> float:
        """
        Calculate minimum capital required for target RBC ratio.

        Args:
            liabilities: Total policy liabilities
            target_rbc_ratio: Target RBC ratio (default 350%)

        Returns:
            Minimum capital required

        Example:
            >>> capital = calculator.calculate_capital_required(
            ...     liabilities=1_000_000_000,
            ...     target_rbc_ratio=350.0
            ... )
            >>> print(f"Required capital: ${capital:,.0f}")
        """
        # Estimate RBC required as ~10% of liabilities (conservative)
        estimated_rbc = liabilities * 0.10

        # Required capital = RBC * target_ratio / 100
        required_capital = estimated_rbc * target_rbc_ratio / 100

        return required_capital

    def stress_test_rbc(
        self,
        portfolio: Portfolio,
        scenario_shocks: Dict[str, float],
        liabilities: float
    ) -> Dict[str, float]:
        """
        Stress test RBC ratio under adverse scenario.

        Args:
            portfolio: Portfolio to stress test
            scenario_shocks: Dictionary of shocks (e.g., {"corporate_spread": +400})
            liabilities: Policy liabilities

        Returns:
            Dictionary with stressed RBC metrics

        Example:
            >>> result = calculator.stress_test_rbc(
            ...     portfolio=my_portfolio,
            ...     scenario_shocks={"corporate_spread": 400},
            ...     liabilities=1_000_000_000
            ... )
            >>> print(f"Stressed RBC: {result['rbc_ratio']:.2f}%")
        """
        # Apply shocks to portfolio
        stressed_value = portfolio.total_value

        # Apply spread shock
        if "corporate_spread" in scenario_shocks:
            spread_increase = scenario_shocks["corporate_spread"] / 10000
            spread_loss = 0.0
            for bond in portfolio.bonds:
                if bond.sector in ["Corporate", "High Yield"]:
                    spread_loss += bond.market_value * bond.duration * spread_increase
            stressed_value -= spread_loss

        # Calculate stressed RBC
        stressed_result = self.calculate_detailed(
            portfolio_value=stressed_value,
            liabilities=liabilities,
            portfolio=portfolio
        )

        return {
            "rbc_ratio": stressed_result.rbc_ratio,
            "loss": portfolio.total_value - stressed_value,
            "loss_percentage": ((portfolio.total_value - stressed_value) / portfolio.total_value * 100),
            "compliant": stressed_result.compliant,
            "action_level": stressed_result.details["action_level"]
        }
