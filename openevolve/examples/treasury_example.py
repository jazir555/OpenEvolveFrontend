"""
Corporate Treasury Example
Demonstrate liquidity management strategy evolution

This example shows how to use the Corporate Treasury vertical to
evolve liquidity management strategies that survive crisis scenarios.

Author: AI Architecture Team
Date: 2026-01-30
"""

import asyncio
from openevolve.finance.verticals.treasury import (
    LiquidityCrisisEvolver,
    CashFlowProfile,
    LiquidityConstraints
)


async def main():
    """Run treasury example"""

    print("=" * 80)
    print("Corporate Treasury Liquidity Management Example")
    print("=" * 80)
    print()

    # Step 1: Define your company's cash flow profile
    print("Step 1: Define Cash Flow Profile")
    print("-" * 80)

    profile = CashFlowProfile(
        daily_burn_rate=5_000_000,  # $5M/day burn rate
        volatility_std=1_000_000,    # ±$1M standard deviation
        seasonal_patterns={
            "q1": 1.1,   # Q1: 10% higher (product launches)
            "q2": 0.95,
            "q3": 0.9,   # Q3: Slow season
            "q4": 1.05
        },
        capex_schedule=[
            {"date": "2026-06-01", "amount": 100_000_000},  # Plant expansion
            {"date": "2026-11-01", "amount": 75_000_000}    # Equipment upgrade
        ]
    )

    print(f"Daily Burn Rate: ${profile.daily_burn_rate:,.0f}")
    print(f"Volatility: ±${profile.volatility_std:,.0f}")
    print(f"Seasonal Patterns: {profile.seasonal_patterns}")
    print(f"Capex Events: {len(profile.capex_schedule)}")
    print()

    # Step 2: Define liquidity constraints
    print("Step 2: Define Liquidity Constraints")
    print("-" * 80)

    constraints = LiquidityConstraints(
        min_liquidity_days=90,       # 90 days minimum
        max_liquidity_cost=75,       # 75 bps max drag
        max_drawdown_credit_line=0.5  # 50% of credit line
    )

    print(f"Minimum Liquidity: {constraints.min_liquidity_days} days")
    print(f"Maximum Cost: {constraints.max_liquidity_cost} bps")
    print(f"Max Credit Line Usage: {constraints.max_drawdown_credit_line:.0%}")
    print()

    # Step 3: Initialize evolver
    print("Step 3: Initialize Evolver")
    print("-" * 80)

    evolver = LiquidityCrisisEvolver(config={
        'n_variants': 100,  # Test 100 different allocations
        'n_top_candidates': 10
    })

    print(f"Number of variants to test: {evolver.n_variants}")
    print(f"Top candidates to track: {evolver.n_top_candidates}")
    print()

    # Step 4: Evolve liquidity strategy
    print("Step 4: Evolve Liquidity Strategy")
    print("-" * 80)
    print("Running evolution... (this may take a few minutes)")
    print()

    result = await evolver.evolve_liquidity_strategy(
        cash_flow_profile=profile,
        constraints=constraints
    )

    # Step 5: Display results
    print("Step 5: Results")
    print("-" * 80)
    print()

    print("Liquidity Metrics:")
    print(f"  Normal Liquidity: {result.liquidity_days:.1f} days")
    print(f"  Stress Liquidity: {result.stress_liquidity_days:.1f} days")
    print(f"  Annual Cost: {result.annual_cost:.1f} bps")
    print(f"  Credit Line Usage: {result.credit_line_usage:.1%}")
    print(f"  Robustness Score: {result.robustness_score:.2f}")
    print()

    print("Optimal Allocation:")
    print(f"  Cash: ${result.strategy.cash:,.0f}")
    print(f"  T-bills: ${result.strategy.t_bills:,.0f}")
    print(f"  Commercial Paper: ${result.strategy.commercial_paper:,.0f}")
    print(f"  Credit Line: ${result.strategy.credit_line_total:,.0f}")
    print()

    print("Stress Test Results:")
    for scenario_name, scenario_result in result.stress_test_results.items():
        status = "[OK] Survived" if scenario_result.success else "[FAIL] Failed"
        print(f"  {scenario_name}: {status}")
        if scenario_result.success:
            print(f"    Min Liquidity: {scenario_result.min_liquidity_days:.1f} days")
            print(f"    Max Credit Usage: {scenario_result.max_credit_line_usage:.1%}")
            print(f"    Final Liquidity: {scenario_result.final_liquidity_days:.1f} days")
        else:
            print(f"    Default Day: {scenario_result.default_day}")
    print()

    # Step 6: Analysis
    print("Step 6: Analysis")
    print("-" * 80)
    print()

    # Calculate key ratios
    total_liquidity = (
        result.strategy.cash +
        result.strategy.t_bills +
        result.strategy.commercial_paper
    )

    cash_ratio = result.strategy.cash / total_liquidity
    tbill_ratio = result.strategy.t_bills / total_liquidity
    cp_ratio = result.strategy.commercial_paper / total_liquidity

    print("Allocation Mix:")
    print(f"  Cash: {cash_ratio:.1%}")
    print(f"  T-bills: {tbill_ratio:.1%}")
    print(f"  Commercial Paper: {cp_ratio:.1%}")
    print()

    # Analyze strategy
    if cash_ratio > 0.5:
        print("Strategy: Conservative")
        print("  - High cash buffer provides immediate liquidity")
        print("  - Lower cost efficiency but higher safety")
    elif cash_ratio > 0.3:
        print("Strategy: Balanced")
        print("  - Good mix of safety and efficiency")
        print("  - T-bills provide core liquidity buffer")
    else:
        print("Strategy: Aggressive")
        print("  - Maximizing yield with lower cash")
        print("  - Higher reliance on credit lines")
    print()

    # Scenario analysis
    survived_scenarios = sum(
        1 for r in result.stress_test_results.values() if r.success
    )

    print(f"Survived {survived_scenarios}/{len(result.stress_test_results)} scenarios")

    if survived_scenarios == len(result.stress_test_results):
        print("  [OK] Strategy survives all stress scenarios")
    elif survived_scenarios >= len(result.stress_test_results) * 0.7:
        print("  [WARN] Strategy survives most scenarios (consider tightening constraints)")
    else:
        print("  [FAIL] Strategy fails too many scenarios (increase liquidity allocation)")
    print()

    # Cost analysis
    if result.annual_cost <= constraints.max_liquidity_cost:
        print(f"  [OK] Annual cost within budget ({result.annual_cost:.1f} <= {constraints.max_liquidity_cost} bps)")
    else:
        print(f"  [WARN] Annual cost exceeds budget ({result.annual_cost:.1f} > {constraints.max_liquidity_cost} bps)")
        print(f"    Excess cost: {result.annual_cost - constraints.max_liquidity_cost:.1f} bps")
    print()

    print("=" * 80)
    print("Example Complete")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
