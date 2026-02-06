#!/usr/bin/env python
"""Quick test script to verify insurance test fixes"""

import sys
sys.path.insert(0, 'core-projects/openevolve')

import asyncio
from datetime import datetime
from openevolve.finance.verticals.insurance import (
    InsuranceReserveEvolver,
    PortfolioConstraints,
    Portfolio,
    Bond,
    CreditRating
)

async def test_duration_constraint():
    """Test that fallback portfolio respects duration constraint"""
    print("Testing duration constraint compliance...")

    constraints = PortfolioConstraints(
        max_duration=5.0,
        min_credit_quality="A-",
        max_concentration=0.25,
        min_diversification=25
    )

    evolver = InsuranceReserveEvolver(config={
        "max_iterations": 2,
        "population_size": 2
    })

    # Generate a random portfolio
    portfolio = await evolver._generate_random_portfolio(constraints)

    print(f"  Portfolio duration: {portfolio.duration:.4f}")
    print(f"  Max allowed: {constraints.max_duration}")
    print(f"  Duration OK: {portfolio.duration <= constraints.max_duration + 0.1}")
    print(f"  Number of bonds: {len(portfolio.bonds)}")
    print(f"  Min diversification: {constraints.min_diversification}")
    print(f"  Diversification OK: {len(portfolio.bonds) >= constraints.min_diversification}")

    return portfolio.duration <= constraints.max_duration + 0.1

async def test_numpy_bool():
    """Test that regulatory_compliant is a Python bool"""
    print("\nTesting numpy bool fix...")

    evolver = InsuranceReserveEvolver(config={
        "max_iterations": 2,
        "population_size": 2
    })

    result = await evolver.evolve_reserve_portfolio(
        reserve_requirements={
            "policy_liabilities": 500_000_000,
            "minimum_rbc": 350
        },
        constraints=PortfolioConstraints(
            max_duration=7.0,
            min_credit_quality="BBB-",
            max_concentration=0.5,
            min_diversification=2,
            max_single_bond=0.3,
            liquidity_requirement=0.05
        )
    )

    print(f"  regulatory_compliant value: {result.regulatory_compliant}")
    print(f"  Type: {type(result.regulatory_compliant)}")
    print(f"  Is Python bool: {isinstance(result.regulatory_compliant, bool)}")

    return isinstance(result.regulatory_compliant, bool)

async def main():
    """Run all quick tests"""
    print("=" * 60)
    print("Insurance Test Fixes Verification")
    print("=" * 60)

    test1_pass = await test_duration_constraint()
    test2_pass = await test_numpy_bool()

    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  Duration constraint: {'PASS' if test1_pass else 'FAIL'}")
    print(f"  Numpy bool fix: {'PASS' if test2_pass else 'FAIL'}")
    print("=" * 60)

    return test1_pass and test2_pass

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
