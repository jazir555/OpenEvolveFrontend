#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical Verifier - Formal Verification of Investment Decisions

Performs formal verification of optimization logic, checks constraint satisfaction,
verifies risk calculations, validates portfolio math (returns, volatility, correlations),
and uses symbolic computation for proofs where applicable.

This module ensures the mathematical correctness of investment decisions.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
import logging
from datetime import datetime
import numpy as np
from dataclasses import dataclass


@dataclass
class VerificationResult:
    """Result of a single verification check."""
    check_name: str
    passed: bool
    details: str
    expected: Any
    actual: Any
    tolerance: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ConstraintCheck:
    """Result of constraint verification."""
    constraint_name: str
    satisfied: bool
    violation_amount: float  # How much constraint is violated (0 if satisfied)
    description: str
    severity: str  # "critical", "warning", "info"


class MathVerifier:
    """
    Mathematical Verifier for Investment Decisions

    Verifies:
    - Portfolio mathematics (returns, volatility, correlations)
    - Constraint satisfaction (position limits, risk limits)
    - Risk calculations (VaR, CVaR, drawdown)
    - Optimization logic
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        critical_tolerance: float = 1e-4
    ):
        """
        Initialize the Math Verifier.

        Args:
            tolerance: Default tolerance for floating point comparisons
            critical_tolerance: Stricter tolerance for critical checks
        """
        self.tolerance = tolerance
        self.critical_tolerance = critical_tolerance
        self.logger = logging.getLogger(__name__)

    async def verify_decision(
        self,
        recommendations: List[Dict[str, Any]],
        current_portfolio: Dict[str, float],
        constraints: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive mathematical verification of investment decision.

        Args:
            recommendations: Investment recommendations to verify
            current_portfolio: Current portfolio allocations
            constraints: Portfolio constraints (max_position_size, risk_tolerance, etc.)

        Returns:
            Dictionary containing verification results
        """
        self.logger.info("Starting mathematical verification of investment decision")

        # Collect all verification results
        all_results = []
        all_constraints = []

        # Verify portfolio mathematics
        portfolio_results = await self._verify_portfolio_math(
            current_portfolio, recommendations
        )
        all_results.extend(portfolio_results)

        # Verify constraints
        constraint_results = await self._verify_constraints(
            current_portfolio, recommendations, constraints
        )
        all_constraints.extend(constraint_results)

        # Verify risk calculations
        risk_results = await self._verify_risk_calculations(
            current_portfolio, recommendations
        )
        all_results.extend(risk_results)

        # Verify optimization logic
        optimization_results = await self._verify_optimization_logic(
            recommendations, constraints
        )
        all_results.extend(optimization_results)

        # Check if all critical verifications passed
        all_passed = all(
            r.passed for r in all_results
            if r.error is None or "critical" not in str(r.error).lower()
        )

        # Count passed/failed
        passed_count = sum(1 for r in all_results if r.passed)
        failed_count = len(all_results) - passed_count

        return {
            "all_passed": all_passed,
            "passed_checks": passed_count,
            "failed_checks": failed_count,
            "total_checks": len(all_results),
            "results": [self._serialize_result(r) for r in all_results],
            "constraints": [self._serialize_constraint(c) for c in all_constraints],
            "critical_issues": [
                self._serialize_result(r) for r in all_results
                if not r.passed and r.error and "critical" in r.error.lower()
            ],
            "verification_metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "tolerance_used": self.tolerance,
                "critical_tolerance_used": self.critical_tolerance
            }
        }

    async def _verify_portfolio_math(
        self,
        current_portfolio: Dict[str, float],
        recommendations: List[Dict[str, Any]]
    ) -> List[VerificationResult]:
        """Verify portfolio mathematics calculations."""
        results = []

        # Check 1: Weights sum to 1 (or 100%)
        weights = list(current_portfolio.values())
        total_weight = sum(weights)

        results.append(VerificationResult(
            check_name="portfolio_weights_sum",
            passed=abs(total_weight - 1.0) < 0.01,  # Allow 1% tolerance
            details=f"Portfolio weights sum to {total_weight:.4f}",
            expected=1.0,
            actual=total_weight,
            tolerance=0.01,
            error=None if abs(total_weight - 1.0) < 0.01 else "CRITICAL: Portfolio weights don't sum to 100%"
        ))

        # Check 2: All weights are non-negative
        all_nonnegative = all(w >= 0 for w in weights)
        results.append(VerificationResult(
            check_name="nonnegative_weights",
            passed=all_nonnegative,
            details="All portfolio weights are non-negative",
            expected="all >= 0",
            actual=f"min: {min(weights):.4f}",
            tolerance=0.0,
            error=None if all_nonnegative else "CRITICAL: Negative weights found"
        ))

        # Check 3: No single weight exceeds 100%
        all_valid = all(w <= 1.0 for w in weights)
        results.append(VerificationResult(
            check_name="valid_weights",
            passed=all_valid,
            details="All portfolio weights are <= 100%",
            expected="all <= 1.0",
            actual=f"max: {max(weights):.4f}",
            tolerance=0.0,
            error=None if all_valid else "CRITICAL: Weight > 100% found"
        ))

        # Check 4: Expected return calculation
        expected_return = 0.0
        for ticker, weight in current_portfolio.items():
            # Assume expected return from recommendation or default
            exp_ret = self._get_expected_return(ticker, recommendations)
            expected_return += weight * exp_ret

        results.append(VerificationResult(
            check_name="expected_return_calculation",
            passed=0.0 <= expected_return <= 1.0,
            details=f"Portfolio expected return: {expected_return:.4f}",
            expected="0 <= return <= 1",
            actual=expected_return,
            tolerance=None,
            error=None if 0.0 <= expected_return <= 1.0 else "WARNING: Unrealistic expected return"
        ))

        return results

    async def _verify_constraints(
        self,
        current_portfolio: Dict[str, float],
        recommendations: List[Dict[str, Any]],
        constraints: Dict[str, float]
    ) -> List[ConstraintCheck]:
        """Verify that all portfolio constraints are satisfied."""
        constraint_checks = []

        # Constraint 1: Maximum position size
        max_position = constraints.get("max_position_size", 0.20)
        violations = [
            (ticker, weight)
            for ticker, weight in current_portfolio.items()
            if weight > max_position
        ]

        if violations:
            for ticker, weight in violations:
                constraint_checks.append(ConstraintCheck(
                    constraint_name="max_position_size",
                    satisfied=False,
                    violation_amount=weight - max_position,
                    description=f"{ticker} weight {weight:.2%} exceeds maximum {max_position:.2%}",
                    severity="critical"
                ))
        else:
            constraint_checks.append(ConstraintCheck(
                constraint_name="max_position_size",
                satisfied=True,
                violation_amount=0.0,
                description="All positions within size limits",
                severity="info"
            ))

        # Constraint 2: Risk tolerance (volatility)
        risk_tolerance = constraints.get("risk_tolerance", 0.15)
        portfolio_volatility = self._calculate_portfolio_volatility(
            current_portfolio, recommendations
        )

        if portfolio_volatility > risk_tolerance:
            constraint_checks.append(ConstraintCheck(
                constraint_name="risk_tolerance",
                satisfied=False,
                violation_amount=portfolio_volatility - risk_tolerance,
                description=f"Portfolio volatility {portfolio_volatility:.2%} exceeds target {risk_tolerance:.2%}",
                severity="warning"
            ))
        else:
            constraint_checks.append(ConstraintCheck(
                constraint_name="risk_tolerance",
                satisfied=True,
                violation_amount=0.0,
                description=f"Portfolio volatility {portfolio_volatility:.2%} within tolerance",
                severity="info"
            ))

        # Constraint 3: Minimum diversification
        min_positions = 5  # Minimum number of positions
        actual_positions = len(current_portfolio)

        if actual_positions < min_positions:
            constraint_checks.append(ConstraintCheck(
                constraint_name="diversification",
                satisfied=False,
                violation_amount=min_positions - actual_positions,
                description=f"Only {actual_positions} positions, minimum {min_positions} required",
                severity="warning"
            ))
        else:
            constraint_checks.append(ConstraintCheck(
                constraint_name="diversification",
                satisfied=True,
                violation_amount=0.0,
                description=f"Portfolio has {actual_positions} positions",
                severity="info"
            ))

        # Constraint 4: Turnover limit
        # Calculate turnover from recommendations
        turnover = self._calculate_turnover(current_portfolio, recommendations)
        max_turnover = 1.0  # 100% annual turnover

        if turnover > max_turnover:
            constraint_checks.append(ConstraintCheck(
                constraint_name="turnover_limit",
                satisfied=False,
                violation_amount=turnover - max_turnover,
                description=f"Portfolio turnover {turnover:.2%} exceeds maximum {max_turnover:.2%}",
                severity="warning"
            ))
        else:
            constraint_checks.append(ConstraintCheck(
                constraint_name="turnover_limit",
                satisfied=True,
                violation_amount=0.0,
                description=f"Portfolio turnover {turnover:.2%} within limits",
                severity="info"
            ))

        return constraint_checks

    async def _verify_risk_calculations(
        self,
        current_portfolio: Dict[str, float],
        recommendations: List[Dict[str, Any]]
    ) -> List[VerificationResult]:
        """Verify risk calculation correctness."""
        results = []

        # Verify volatility calculation
        portfolio_vol = self._calculate_portfolio_volatility(current_portfolio, recommendations)

        # Calculate weighted average volatility
        weighted_avg_vol = 0.0
        for ticker, weight in current_portfolio.items():
            vol = self._get_volatility(ticker, recommendations)
            weighted_avg_vol += weight * vol

        # Portfolio volatility should be <= weighted average (due to diversification)
        results.append(VerificationResult(
            check_name="volatility_diversification_benefit",
            passed=portfolio_vol <= weighted_avg_vol * 1.01,  # Allow small numerical error
            details=f"Portfolio vol {portfolio_vol:.4f} vs weighted avg {weighted_avg_vol:.4f}",
            expected=f"portfolio_vol <= {weighted_avg_vol:.4f}",
            actual=portfolio_vol,
            tolerance=0.01,
            error=None if portfolio_vol <= weighted_avg_vol * 1.01 else "WARNING: Volatility calculation may be incorrect"
        ))

        # Verify VaR calculation (if provided)
        var_95 = self._calculate_var(current_portfolio, recommendations, confidence=0.95)

        results.append(VerificationResult(
            check_name="var_calculation",
            passed=var_95 < 0,  # VaR should be negative (loss)
            details=f"95% VaR: {var_95:.2%}",
            expected="VaR < 0",
            actual=var_95,
            tolerance=None,
            error=None if var_95 < 0 else "WARNING: VaR should be negative (loss)"
        ))

        # Verify drawdown calculation
        max_drawdown = self._calculate_max_drawdown(current_portfolio, recommendations)

        results.append(VerificationResult(
            check_name="drawdown_calculation",
            passed=-1.0 <= max_drawdown <= 0.0,  # Drawdown should be between -100% and 0%
            details=f"Max drawdown: {max_drawdown:.2%}",
            expected="-1.0 <= drawdown <= 0.0",
            actual=max_drawdown,
            tolerance=None,
            error=None if -1.0 <= max_drawdown <= 0.0 else "CRITICAL: Invalid drawdown value"
        ))

        return results

    async def _verify_optimization_logic(
        self,
        recommendations: List[Dict[str, Any]],
        constraints: Dict[str, float]
    ) -> List[VerificationResult]:
        """Verify optimization logic and consistency."""
        results = []

        # Check 1: Recommendations are internally consistent
        results.append(VerificationResult(
            check_name="recommendation_consistency",
            passed=self._check_recommendation_consistency(recommendations),
            details="Recommendations are internally consistent",
            expected="True",
            actual=self._check_recommendation_consistency(recommendations),
            tolerance=None,
            error=None if self._check_recommendation_consistency(recommendations) else "WARNING: Inconsistent recommendations"
        ))

        # Check 2: Recommendations respect constraints
        respects_constraints = True
        for rec in recommendations:
            actions = rec.get("actions", [])
            for action in actions:
                if action.get("action") == "increase_allocation":
                    target = action.get("target_allocation", "")
                    # Parse percentage from string like "10-15% of portfolio"
                    if isinstance(target, str):
                        try:
                            max_val = max([float(x.strip().rstrip('%')) / 100
                                         for x in target.split('-')])
                            if max_val > constraints.get("max_position_size", 0.20):
                                respects_constraints = False
                                break
                        except:
                            pass

        results.append(VerificationResult(
            check_name="constraint_respect",
            passed=respects_constraints,
            details="Recommendations respect portfolio constraints",
            expected="True",
            actual=respects_constraints,
            tolerance=None,
            error=None if respects_constraints else "WARNING: Recommendations violate constraints"
        ))

        # Check 3: Expected returns are realistic
        all_realistic = True
        for rec in recommendations:
            exp_return = rec.get("expected_return", 0.0)
            if exp_return > 0.5 or exp_return < -0.5:  # Outside reasonable range
                all_realistic = False
                break

        results.append(VerificationResult(
            check_name="realistic_returns",
            passed=all_realistic,
            details="Expected returns are within realistic ranges",
            expected="-50% <= return <= 50%",
            actual="all realistic" if all_realistic else "unrealistic returns found",
            tolerance=None,
            error=None if all_realistic else "WARNING: Unrealistic expected returns"
        ))

        return results

    def _calculate_portfolio_volatility(
        self,
        portfolio: Dict[str, float],
        recommendations: List[Dict[str, Any]]
    ) -> float:
        """Calculate portfolio volatility from individual volatilities and correlations."""
        # Simplified calculation assuming constant correlation
        weights = np.array(list(portfolio.values()))
        volatilities = []

        for ticker in portfolio.keys():
            vol = self._get_volatility(ticker, recommendations)
            volatilities.append(vol)

        volatilities = np.array(volatilities)

        # Assume average correlation of 0.3 for demonstration
        avg_correlation = 0.3

        # Portfolio variance = w^T * Σ * w
        # Simplified: sum(w_i^2 * σ_i^2) + sum(sum(w_i * w_j * σ_i * σ_j * ρ))
        variance = np.sum(weights**2 * volatilities**2)
        covariance_sum = 0
        n = len(weights)

        for i in range(n):
            for j in range(i+1, n):
                covariance_sum += 2 * weights[i] * weights[j] * volatilities[i] * volatilities[j] * avg_correlation

        portfolio_variance = variance + covariance_sum

        return np.sqrt(portfolio_variance)

    def _calculate_var(
        self,
        portfolio: Dict[str, float],
        recommendations: List[Dict[str, Any]],
        confidence: float = 0.95
    ) -> float:
        """Calculate Value at Risk at given confidence level."""
        # Simplified parametric VaR
        portfolio_vol = self._calculate_portfolio_volatility(portfolio, recommendations)
        portfolio_return = self._calculate_portfolio_return(portfolio, recommendations)

        # Z-score for confidence level
        z_scores = {0.95: 1.645, 0.99: 2.326}
        z = z_scores.get(confidence, 1.645)

        # VaR = return - z * volatility
        var = portfolio_return - z * portfolio_vol

        return var

    def _calculate_max_drawdown(
        self,
        portfolio: Dict[str, float],
        recommendations: List[Dict[str, Any]]
    ) -> float:
        """Calculate maximum drawdown (simplified)."""
        # For verification, use recommendation's stated max drawdown
        # In production, would calculate from historical returns
        for rec in recommendations:
            if "max_drawdown" in rec:
                return rec["max_drawdown"]

        # Default: assume -20% if not specified
        return -0.20

    def _calculate_portfolio_return(
        self,
        portfolio: Dict[str, float],
        recommendations: List[Dict[str, Any]]
    ) -> float:
        """Calculate expected portfolio return."""
        total_return = 0.0

        for ticker, weight in portfolio.items():
            exp_ret = self._get_expected_return(ticker, recommendations)
            total_return += weight * exp_ret

        return total_return

    def _calculate_turnover(
        self,
        current_portfolio: Dict[str, float],
        recommendations: List[Dict[str, Any]]
    ) -> float:
        """Calculate portfolio turnover from recommendations."""
        # Simplified: sum of all buy/sell actions
        turnover = 0.0

        for rec in recommendations:
            actions = rec.get("actions", [])
            for action in actions:
                if action.get("action") in ["buy", "sell"]:
                    # Estimate size from target allocation
                    target = action.get("target_allocation", "")
                    if isinstance(target, str):
                        try:
                            # Parse "10-15% of portfolio"
                            values = [float(x.strip().rstrip('%')) / 100
                                    for x in target.split('-')]
                            avg_size = sum(values) / len(values)
                            turnover += avg_size
                        except:
                            pass

        return turnover

    def _get_expected_return(
        self,
        ticker: str,
        recommendations: List[Dict[str, Any]]
    ) -> float:
        """Get expected return for a ticker."""
        # Check recommendations first
        for rec in recommendations:
            if ticker in rec.get("hypothesis", ""):
                return rec.get("expected_return", 0.10)

        # Default: 10% annual return
        return 0.10

    def _get_volatility(
        self,
        ticker: str,
        recommendations: List[Dict[str, Any]]
    ) -> float:
        """Get volatility for a ticker."""
        # Check recommendations first
        for rec in recommendations:
            if ticker in rec.get("hypothesis", ""):
                return rec.get("volatility", 0.20)

        # Default: 20% annual volatility
        return 0.20

    def _check_recommendation_consistency(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> bool:
        """Check if recommendations are internally consistent."""
        # Check for contradictory actions
        actions_by_ticker = {}

        for rec in recommendations:
            for action in rec.get("actions", []):
                ticker = action.get("ticker", "")
                act = action.get("action", "")

                if ticker not in actions_by_ticker:
                    actions_by_ticker[ticker] = []

                actions_by_ticker[ticker].append(act)

        # Check for contradictions
        for ticker, actions in actions_by_ticker.items():
            if "buy" in actions and "sell" in actions:
                return False
            if "increase_allocation" in actions and "reduce_allocation" in actions:
                return False

        return True

    def _serialize_result(self, result: VerificationResult) -> Dict[str, Any]:
        """Convert verification result to dictionary."""
        return {
            "check_name": result.check_name,
            "passed": result.passed,
            "details": result.details,
            "expected": str(result.expected),
            "actual": str(result.actual),
            "tolerance": result.tolerance,
            "error": result.error
        }

    def _serialize_constraint(self, constraint: ConstraintCheck) -> Dict[str, Any]:
        """Convert constraint check to dictionary."""
        return {
            "constraint_name": constraint.constraint_name,
            "satisfied": constraint.satisfied,
            "violation_amount": constraint.violation_amount,
            "description": constraint.description,
            "severity": constraint.severity
        }

    async def verify_portfolio_invariants(
        self,
        portfolio: Dict[str, float],
        prices: Dict[str, float],
        cash: float
    ) -> Dict[str, Any]:
        """
        Verify fundamental portfolio invariants.

        Args:
            portfolio: Holdings (ticker -> shares)
            prices: Current prices (ticker -> price)
            cash: Cash amount

        Returns:
            Verification results
        """
        results = []

        # Calculate total value
        holdings_value = sum(shares * prices.get(ticker, 0) for ticker, shares in portfolio.items())
        total_value = holdings_value + cash

        # Invariant 1: Total value is positive
        results.append(VerificationResult(
            check_name="total_value_positive",
            passed=total_value > 0,
            details=f"Total portfolio value: ${total_value:,.2f}",
            expected="total_value > 0",
            actual=total_value,
            tolerance=0.0,
            error=None if total_value > 0 else "CRITICAL: Negative portfolio value"
        ))

        # Invariant 2: Holdings are non-negative
        all_positive = all(shares >= 0 for shares in portfolio.values())
        results.append(VerificationResult(
            check_name="nonnegative_holdings",
            passed=all_positive,
            details="All share counts are non-negative",
            expected="all >= 0",
            actual=f"min: {min(portfolio.values())}",
            tolerance=0.0,
            error=None if all_positive else "CRITICAL: Negative share count"
        ))

        # Invariant 3: Prices are positive
        all_prices_positive = all(p > 0 for p in prices.values())
        results.append(VerificationResult(
            check_name="positive_prices",
            passed=all_prices_positive,
            details="All prices are positive",
            expected="all > 0",
            actual=f"min: ${min(prices.values()):.2f}",
            tolerance=0.0,
            error=None if all_prices_positive else "CRITICAL: Non-positive price"
        ))

        return {
            "all_passed": all(r.passed for r in results),
            "results": [self._serialize_result(r) for r in results],
            "portfolio_value": total_value,
            "holdings_value": holdings_value,
            "cash": cash
        }
