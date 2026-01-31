"""
Insurance Reserve Evolution Example

Demonstrates evolving insurance reserve portfolios that survive
regulatory stress tests while maintaining RBC ratios.

This example shows:
1. Basic reserve portfolio evolution
2. RBC calculation and analysis
3. Stress testing scenarios
4. Constraint optimization

Author: AI Architecture Team
Date: 2026-01-30
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from openevolve.finance.verticals.insurance import (
    InsuranceReserveEvolver,
    RBCCalculator,
    StressScenarioGenerator,
    PortfolioConstraints,
    Portfolio,
    Bond,
    CreditRating


async def example_1_basic_evolution():
    """Example 1: Basic insurance reserve portfolio evolution"""
    print("=" * 80)
    print("EXAMPLE 1: Basic Insurance Reserve Evolution")
    print("=" * 80)

    # Initialize evolver with conservative config for demo
    evolver = InsuranceReserveEvolver(config={
        "max_iterations": 20,  # Small for demo (use 100+ in production)
        "population_size": 15,
        "mutation_rate": 0.1
    })

    # Define reserve requirements
    reserve_requirements = {
        "policy_liabilities": 1_000_000_000,  # $1B in liabilities
        "minimum_rbc": 350.0  # 350% RBC required
    }

    # Define portfolio constraints
    constraints = PortfolioConstraints(
        max_duration=7.0,
        min_credit_quality="BBB-",
        max_concentration=0.30,
        min_diversification=20,
        max_single_bond=0.05,
        liquidity_requirement=0.10
    )

    print(f"\nReserve Requirements:")
    print(f"  Policy Liabilities: ${reserve_requirements['policy_liabilities']:,.0f}")
    print(f"  Minimum RBC: {reserve_requirements['minimum_rbc']}%")
    print(f"\nPortfolio Constraints:")
    print(f"  Max Duration: {constraints.max_duration} years")
    print(f"  Min Credit Quality: {constraints.min_credit_quality}")
    print(f"  Max Concentration: {constraints.max_concentration*100}%")

    # Evolve portfolio
    print(f"\nEvolving portfolio...")
    result = await evolver.evolve_reserve_portfolio(
        reserve_requirements=reserve_requirements,
        constraints=constraints
    )

    # Display results
    print(f"\n{'='*80}")
    print(f"EVOLUTION RESULTS")
    print(f"{'='*80}")

    print(f"\nEvolved Portfolio:")
    print(f"  Total Value: ${result.portfolio.total_value:,.0f}")
    print(f"  Cash: ${result.portfolio.cash:,.0f}")
    print(f"  Duration: {result.portfolio.duration:.2f} years")
    print(f"  Number of Bonds: {len(result.portfolio.bonds)}")
    print(f"  Credit Quality: {result.portfolio.credit_quality.value}")

    print(f"\nStress Test Results:")
    for scenario_name, stress_result in result.stress_test_results.items():
        print(f"\n  Scenario: {scenario_name}")
        print(f"    Initial Value: ${stress_result.initial_value:,.0f}")
        print(f"    Final Value: ${stress_result.final_value:,.0f}")
        print(f"    Loss: ${stress_result.loss_amount:,.0f} ({stress_result.loss_percentage:.2f}%)")
        print(f"    Initial RBC: {stress_result.rbc_ratio_initial:.2f}%")
        print(f"    Final RBC: {stress_result.rbc_ratio_final:.2f}%")
        print(f"    Breaches RBC: {stress_result.breaches_rbc}")

    print(f"\nOverall Results:")
    print(f"  Minimum RBC: {result.min_rbc_ratio:.2f}%")
    print(f"  Regulatory Compliant: {result.regulatory_compliant}")
    print(f"  Scenarios Tested: {len(result.scenarios_tested)}")
    print(f"  Evolution Iterations: {result.evolution_iterations}")

    return result


async def example_2_rbc_analysis():
    """Example 2: RBC calculation and analysis"""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: RBC Calculation and Analysis")
    print("=" * 80)

    # Create sample portfolio
    bonds = [
        Bond(
            ticker="US10Y",
            rating=CreditRating.AAA,
            par_value=100_000_000,
            market_value=105_000_000,
            book_value=100_000_000,
            duration=6.5,
            convexity=55.0,
            yield_to_maturity=0.042,
            sector="Government",
            coupon_rate=0.040,
            maturity_date=datetime(2035, 1, 1)
        ),
        Bond(
            ticker="CORP_AA",
            rating=CreditRating.AA,
            par_value=75_000_000,
            market_value=77_000_000,
            book_value=75_000_000,
            duration=5.8,
            convexity=48.0,
            yield_to_maturity=0.048,
            sector="Corporate",
            coupon_rate=0.045,
            maturity_date=datetime(2033, 6, 1)
        ),
        Bond(
            ticker="CORP_BBB",
            rating=CreditRating.BBB,
            par_value=50_000_000,
            market_value=49_000_000,
            book_value=50_000_000,
            duration=4.5,
            convexity=38.0,
            yield_to_maturity=0.055,
            sector="Corporate",
            coupon_rate=0.052,
            maturity_date=datetime(2030, 3, 1)
        )
    ]

    portfolio = Portfolio(
        bonds=bonds,
        cash=30_000_000,
        total_value=261_000_000
    )

    print(f"\nSample Portfolio:")
    print(f"  Total Value: ${portfolio.total_value:,.0f}")
    print(f"  Cash: ${portfolio.cash:,.0f}")
    print(f"  Duration: {portfolio.duration:.2f} years")
    print(f"  Number of Bonds: {len(portfolio.bonds)}")

    # Calculate RBC
    calculator = RBCCalculator()
    liabilities = 150_000_000

    result = calculator.calculate_detailed(
        portfolio_value=portfolio.total_value,
        liabilities=liabilities,
        portfolio=portfolio
    )

    print(f"\nRBC Analysis:")
    print(f"  Policy Liabilities: ${liabilities:,.0f}")
    print(f"  Total Adjusted Capital: ${result.tac:,.0f}")
    print(f"  RBC Required: ${result.rbc_required:,.0f}")
    print(f"  RBC Ratio: {result.rbc_ratio:.2f}%")
    print(f"  Status: {result.details['action_level']}")

    print(f"\nRBC Risk Components:")
    print(f"  C0 (Affiliates): ${result.c0_risk:,.0f}")
    print(f"  C1 (Fixed Income): ${result.c1_risk:,.0f}")
    print(f"  C2 (Equity): ${result.c2_risk:,.0f}")
    print(f"  C3 (Real Estate): ${result.c3_risk:,.0f}")
    print(f"  C4 (Off-Balance Sheet): ${result.c4_risk:,.0f}")

    # Calculate capital required
    capital_required = calculator.calculate_capital_required(
        liabilities=liabilities,
        target_rbc_ratio=350.0
    )

    print(f"\nCapital Required for 350% RBC: ${capital_required:,.0f}")
    print(f"Current Capital: ${result.tac:,.0f}")
    print(f"Surplus/Deficit: ${result.tac - capital_required:,.0f}")


async def example_3_stress_testing():
    """Example 3: Stress test scenarios"""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: Stress Testing Scenarios")
    print("=" * 80)

    # Create test portfolio
    bonds = [
        Bond(
            ticker=f"BOND{i}",
            rating=CreditRating.A,
            par_value=50_000_000,
            market_value=52_000_000,
            book_value=50_000_000,
            duration=5.0,
            convexity=45.0,
            yield_to_maturity=0.050,
            sector="Corporate",
            coupon_rate=0.048,
            maturity_date=datetime(2032, 1, 1)
        )
        for i in range(5)
    ]

    portfolio = Portfolio(
        bonds=bonds,
        cash=50_000_000,
        total_value=310_000_000
    )

    print(f"\nTest Portfolio:")
    print(f"  Total Value: ${portfolio.total_value:,.0f}")
    print(f"  Duration: {portfolio.duration:.2f} years")

    # Generate stress scenarios
    generator = StressScenarioGenerator()
    scenarios = generator.generate_all_scenarios()

    print(f"\nStress Scenarios Generated: {len(scenarios)}")

    # Test against scenarios
    calculator = RBCCalculator()
    liabilities = 200_000_000

    print(f"\n{'='*80}")
    print(f"STRESS TEST RESULTS")
    print(f"{'='*80}")

    for scenario in scenarios:
        stress_result = calculator.stress_test_rbc(
            portfolio=portfolio,
            scenario_shocks=scenario.shocks,
            liabilities=liabilities
        )

        print(f"\nScenario: {scenario.name}")
        print(f"  Description: {scenario.description}")
        print(f"  Stressed RBC: {stress_result['rbc_ratio']:.2f}%")
        print(f"  Loss: {stress_result['loss_percentage']:.2f}%")
        print(f"  Compliant: {stress_result['compliant']}")
        print(f"  Action Level: {stress_result.get('action_level', 'N/A')}")


async def example_4_constraint_optimization():
    """Example 4: Optimize under strict constraints"""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 4: Constraint Optimization")
    print("=" * 80)

    evolver = InsuranceReserveEvolver(config={
        "max_iterations": 15,
        "population_size": 12
    })

    # Define increasingly strict constraints
    constraint_levels = [
        ("Relaxed", PortfolioConstraints(
            max_duration=7.0,
            min_credit_quality="BBB-",
            max_concentration=0.40,
            min_diversification=15
        )),
        ("Moderate", PortfolioConstraints(
            max_duration=6.0,
            min_credit_quality="BBB",
            max_concentration=0.30,
            min_diversification=20
        )),
        ("Strict", PortfolioConstraints(
            max_duration=5.0,
            min_credit_quality="A-",
            max_concentration=0.20,
            min_diversification=25
        ))
    ]

    print(f"\nTesting different constraint levels...")

    for level_name, constraints in constraint_levels:
        print(f"\n{'='*80}")
        print(f"Constraint Level: {level_name}")
        print(f"{'='*80}")
        print(f"  Max Duration: {constraints.max_duration} years")
        print(f"  Min Credit: {constraints.min_credit_quality}")
        print(f"  Max Concentration: {constraints.max_concentration*100}%")
        print(f"  Min Diversification: {constraints.min_diversification}")

        try:
            result = await evolver.evolve_reserve_portfolio(
                reserve_requirements={
                    "policy_liabilities": 500_000_000,
                    "minimum_rbc": 350
                },
                constraints=constraints
            )

            print(f"\nResults:")
            print(f"  Portfolio Duration: {result.portfolio.duration:.2f} years")
            print(f"  Number of Bonds: {len(result.portfolio.bonds)}")
            print(f"  Min RBC: {result.min_rbc_ratio:.2f}%")
            print(f"  Compliant: {result.regulatory_compliant}")

            # Check if constraints satisfied
            duration_ok = result.portfolio.duration <= constraints.max_duration + 0.1
            diversification_ok = len(result.portfolio.bonds) >= constraints.min_diversification

            print(f"  Constraints Satisfied: {duration_ok and diversification_ok}")

        except Exception as e:
            print(f"\nError: {e}")
            print(f"  Constraints may be too strict!")


async def main():
    """Run all examples"""
    print("\n")
    print("*" * 80)
    print("INSURANCE RESERVE EVOLUTION EXAMPLES")
    print("LoongFlow-OpenEvolve Finance Platform")
    print("*" * 80)

    # Run examples
    await example_1_basic_evolution()
    await example_2_rbc_analysis()
    await example_3_stress_testing()
    await example_4_constraint_optimization()

    print("\n\n" + "*" * 80)
    print("EXAMPLES COMPLETE")
    print("*" * 80)


if __name__ == "__main__":
    asyncio.run(main())
