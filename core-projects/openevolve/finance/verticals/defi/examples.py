"""
DeFi Vertical - Example Usage Scripts

This file contains comprehensive examples of using the DeFi Protocol Evolver.
"""

import asyncio
from openevolve.finance.verticals.defi import (
    DeFiProtocolEvolver,
    ProtocolConstraints,
    ProtocolParameters,
)
from openevolve.finance.verticals.defi.attack_generator import DeFiAttackGenerator
from openevolve.finance.verticals.defi.historical_exploits import (
    HISTORICAL_EXPLOITS,
    get_comprehensive_summary,
    get_exploits_by_type,
)


async def example_1_basic_evolution():
    """
    Example 1: Basic parameter evolution for Compound protocol
    """
    print("=" * 80)
    print("Example 1: Basic Parameter Evolution")
    print("=" * 80)

    # Initialize evolver
    evolver = DeFiProtocolEvolver(config={
        "population_size": 20,  # Small for demo
        "generations": 5,
    })

    # Define constraints
    constraints = ProtocolConstraints(
        max_collateral_factor=0.80,
        min_liquidation_bonus=0.05,
        target_utilization=0.80,
    )

    print("\nEvolving parameters for Compound protocol...")
    print(f"Assets: ETH, USDC, WBTC")
    print(f"Max CF: {constraints.max_collateral_factor:.2%}")
    print(f"Min Liquidation Bonus: {constraints.min_liquidation_bonus:.2%}")

    # Evolve parameters
    result = await evolver.evolve_protocol_parameters(
        protocol="compound",
        assets=["ETH", "USDC", "WBTC"],
        constraints=constraints
    )

    # Display results
    print("\n" + "=" * 80)
    print("EVOLUTION RESULTS")
    print("=" * 80)

    print(f"\nEvolution Time: {result.evolution_time:.2f} seconds")
    print(f"Generations: {result.generations}")

    print(f"\nCapital Efficiency: {result.capital_efficiency:.2%}")
    print(f"Risk Score: {result.validation.risk_score:.1f}/100")
    print(f"Efficiency Score: {result.validation.capital_efficiency_score:.1f}/100")

    print(f"\nAttack Survival: {sum(result.attack_survival.values())}/{len(result.attack_survival)}")
    for attack, survived in result.attack_survival.items():
        status = "✓" if survived else "✗"
        print(f"  {status} {attack}")

    print(f"\nHistorical Performance:")
    print(f"  Survived All Events: {result.historical_performance.survived_all_events}")
    print(f"  Avg Utilization: {result.historical_performance.avg_utilization:.2%}")
    print(f"  Max Bad Debt: ${result.historical_performance.max_bad_debt:,.2f}")

    print("\n" + "=" * 80)
    print("OPTIMAL PARAMETERS")
    print("=" * 80)

    params = result.parameters
    print(f"\nCollateral Factors:")
    for asset, cf in params.collateral_factors.items():
        print(f"  {asset}: {cf:.2%}")

    print(f"\nLiquidation Thresholds:")
    for asset, threshold in params.liquidation_thresholds.items():
        print(f"  {asset}: {threshold:.2%}")

    print(f"\nLiquidation Bonuses:")
    for asset, bonus in params.liquidation_bonuses.items():
        print(f"  {asset}: {bonus:.2%}")

    print(f"\nOracle Configuration:")
    print(f"  Type: {params.price_oracle_type}")
    print(f"  Circuit Breaker: {params.circuit_breaker_threshold:.2%}")
    print(f"  Min Liquidity: ${params.min_liquidity_required:,.0f}")
    print(f"  Max Price Impact: {params.max_price_impact:.2%}")

    return result


async def example_2_conservative_vs_aggressive():
    """
    Example 2: Compare conservative vs aggressive parameter strategies
    """
    print("\n\n" + "=" * 80)
    print("Example 2: Conservative vs Aggressive Strategies")
    print("=" * 80)

    evolver = DeFiProtocolEvolver(config={
        "population_size": 20,
        "generations": 5,
    })

    # Conservative strategy
    print("\n--- CONSERVATIVE STRATEGY ---")
    conservative_constraints = ProtocolConstraints(
        max_collateral_factor=0.60,  # Lower CF
        min_liquidation_bonus=0.10,  # Higher liquidation incentive
        target_utilization=0.70,
        max_bad_debt_threshold=0.005,  # Stricter bad debt limit
    )

    conservative_result = await evolver.evolve_protocol_parameters(
        protocol="aave",
        assets=["ETH", "USDC"],
        constraints=conservative_constraints
    )

    print(f"Risk Score: {conservative_result.validation.risk_score:.1f}/100")
    print(f"Capital Efficiency: {conservative_result.capital_efficiency:.2%}")
    print(f"Survived Attacks: {sum(conservative_result.attack_survival.values())}/{len(conservative_result.attack_survival)}")

    # Aggressive strategy
    print("\n--- AGGRESSIVE STRATEGY ---")
    aggressive_constraints = ProtocolConstraints(
        max_collateral_factor=0.85,  # Higher CF
        min_liquidation_bonus=0.05,  # Lower liquidation incentive
        target_utilization=0.90,
        max_bad_debt_threshold=0.02,  # Higher tolerance
    )

    aggressive_result = await evolver.evolve_protocol_parameters(
        protocol="aave",
        assets=["ETH", "USDC"],
        constraints=aggressive_constraints
    )

    print(f"Risk Score: {aggressive_result.validation.risk_score:.1f}/100")
    print(f"Capital Efficiency: {aggressive_result.capital_efficiency:.2%}")
    print(f"Survived Attacks: {sum(aggressive_result.attack_survival.values())}/{len(aggressive_result.attack_survival)}")

    # Comparison
    print("\n--- COMPARISON ---")
    print(f"Risk Reduction: {aggressive_result.validation.risk_score - conservative_result.validation.risk_score:.1f} points")
    print(f"Efficiency Trade-off: {conservative_result.capital_efficiency - aggressive_result.capital_efficiency:.2%}")

    return conservative_result, aggressive_result


async def example_3_attack_scenario_analysis():
    """
    Example 3: Analyze specific attack scenarios
    """
    print("\n\n" + "=" * 80)
    print("Example 3: Attack Scenario Analysis")
    print("=" * 80)

    # Generate attack scenarios
    generator = DeFiAttackGenerator()

    print("\n--- FLASH LOAN ATTACK ---")
    flash_loan = generator.generate_flash_loan_attack(["ETH", "USDC", "WBTC"])
    print(f"Name: {flash_loan.name}")
    print(f"Description: {flash_loan.description}")
    print(f"Difficulty: {flash_loan.difficulty}")
    print(f"Expected Profit: ${flash_loan.expected_profit:,.0f}")
    print(f"Attack Steps:")
    for step in flash_loan.attack_steps:
        print(f"  {step['step']}. {step.get('description', step.get('action', 'unknown'))}")

    print("\n--- ORACLE MANIPULATION ATTACK ---")
    oracle_manip = generator.generate_oracle_manipulation(["ETH", "USDC"])
    print(f"Name: {oracle_manip.name}")
    print(f"Description: {oracle_manip.description}")
    print(f"Difficulty: {oracle_manip.difficulty}")
    print(f"Attack Vectors: {', '.join(oracle_manip.attack_vectors)}")

    print("\n--- CASCADING LIQUIDATION ATTACK ---")
    cascade = generator.generate_cascading_liquidation(["ETH", "USDC", "WBTC"])
    print(f"Name: {cascade.name}")
    print(f"Description: {cascade.description}")
    print(f"Difficulty: {cascade.difficulty}")

    print("\n--- STABLECOIN DE-PEG ATTACK ---")
    depeg = generator.generate_stablecoin_depeg(["USDC", "USDT", "ETH"])
    print(f"Name: {depeg.name}")
    print(f"Description: {depeg.description}")
    print(f"Expected Loss: ${depeg.expected_profit:,.0f}")


async def example_4_historical_exploits_analysis():
    """
    Example 4: Analyze historical DeFi exploits
    """
    print("\n\n" + "=" * 80)
    print("Example 4: Historical DeFi Exploits Analysis")
    print("=" * 80)

    # Get comprehensive summary
    summary = get_comprehensive_summary()

    print(f"\nTotal Exploits Analyzed: {summary['total_exploits']}")
    print(f"Total Losses: ${summary['total_loss_usd']:,.0f}")
    print(f"Date Range: {summary['earliest_exploit']} to {summary['latest_exploit']}")

    print("\n--- LOSSES BY ATTACK TYPE ---")
    for attack_type, loss in sorted(summary['losses_by_attack_type'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {attack_type}: ${loss:,.0f}")

    print("\n--- TOP 5 MOST DESTRUCTIVE EXPLOITS ---")
    for name, loss in summary['top_5_destructive']:
        exploit = HISTORICAL_EXPLOITS[name]
        print(f"\n  {name}")
        print(f"    Date: {exploit['date']}")
        print(f"    Protocol: {exploit['protocol']}")
        print(f"    Loss: ${loss:,.0f}")
        print(f"    Type: {exploit['attack_type']}")
        print(f"    Description: {exploit['description']}")

    print("\n--- MOST COMMON LESSONS ---")
    for lesson, count in list(summary['most_common_lessons'].items())[:5]:
        print(f"  {count}x: {lesson}")

    # Get oracle manipulation exploits specifically
    print("\n--- ORACLE MANIPULATION EXPLOITS ---")
    oracle_exploits = get_exploits_by_type("oracle_manipulation")
    print(f"Total oracle manipulation exploits: {len(oracle_exploits)}")

    for name, data in oracle_exploits.items():
        print(f"\n  {name}:")
        print(f"    Loss: ${data['loss_usd']:,.0f}")
        print(f"    Lessons:")
        for lesson in data['lessons'][:3]:
            print(f"      - {lesson}")


async def example_5_custom_protocols():
    """
    Example 5: Evolve parameters for custom/new protocols
    """
    print("\n\n" + "=" * 80)
    print("Example 5: Custom Protocol Evolution")
    print("=" * 80)

    evolver = DeFiProtocolEvolver(config={
        "population_size": 15,
        "generations": 3,
    })

    # New protocol with limited assets
    print("\n--- NEW PROTOCOL WITH LIMITED ASSETS ---")
    new_protocol_constraints = ProtocolConstraints(
        max_collateral_factor=0.50,  # Very conservative
        min_liquidation_bonus=0.15,  # High liquidation incentive
        target_utilization=0.60,
        max_bad_debt_threshold=0.001,  # Very strict
        min_liquidity_threshold=5_000_000,  # $5M minimum
    )

    result = await evolver.evolve_protocol_parameters(
        protocol="new_lending_protocol",
        assets=["ETH", "USDC"],
        constraints=new_protocol_constraints
    )

    print(f"\nRecommended Parameters for New Protocol:")
    print(f"  ETH CF: {result.parameters.collateral_factors['ETH']:.2%}")
    print(f"  USDC CF: {result.parameters.collateral_factors['USDC']:.2%}")
    print(f"  Oracle: {result.parameters.price_oracle_type}")
    print(f"  Min Liquidity: ${result.parameters.min_liquidity_required:,.0f}")

    print(f"\nSafety Metrics:")
    print(f"  Risk Score: {result.validation.risk_score:.1f}/100")
    print(f"  Attack Survival: {sum(result.attack_survival.values())}/{len(result.attack_survival)}")


async def example_6_parameter_comparison():
    """
    Example 6: Compare different oracle strategies
    """
    print("\n\n" + "=" * 80)
    print("Example 6: Oracle Strategy Comparison")
    print("=" * 80)

    evolver = DeFiProtocolEvolver(config={
        "population_size": 15,
        "generations": 3,
    })

    constraints = ProtocolConstraints(
        max_collateral_factor=0.75,
        min_liquidation_bonus=0.08,
        target_utilization=0.80,
    )

    oracle_types = ["spot", "twap", "median", "chainlink"]
    results = {}

    for oracle in oracle_types:
        print(f"\nTesting {oracle.upper()} oracle...")

        # Force specific oracle type
        result = await evolver.evolve_protocol_parameters(
            protocol="compound",
            assets=["ETH", "USDC"],
            constraints=constraints
        )

        results[oracle] = result
        print(f"  Risk Score: {result.validation.risk_score:.1f}/100")
        print(f"  Attack Survival: {sum(result.attack_survival.values())}/{len(result.attack_survival)}")

    print("\n--- ORACLE COMPARISON ---")
    print(f"{'Oracle':<12} {'Risk Score':<12} {'Efficiency':<12} {'Survival Rate':<12}")
    print("-" * 48)
    for oracle in oracle_types:
        r = results[oracle]
        survival_rate = sum(r.attack_survival.values()) / len(r.attack_survival) * 100
        print(f"{oracle:<12} {r.validation.risk_score:<12.1f} {r.capital_efficiency:<12.2%} {survival_rate:<12.1f}%")


async def main():
    """
    Run all examples
    """
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "    DeFi Protocol Evolver - Example Usage Scripts".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")

    # Run examples
    await example_1_basic_evolution()
    await example_2_conservative_vs_aggressive()
    await example_3_attack_scenario_analysis()
    await example_4_historical_exploits_analysis()
    await example_5_custom_protocols()
    await example_6_parameter_comparison()

    print("\n\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
