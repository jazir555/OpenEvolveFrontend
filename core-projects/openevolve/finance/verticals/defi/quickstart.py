"""
DeFi Vertical - Quickstart Guide

This script demonstrates the fastest way to get started with DeFi protocol evolution.
"""

import asyncio


async def quickstart():
    """
    Quickstart example - Evolve parameters for Compound in under 5 minutes
    """

    # Step 1: Import
    from openevolve.finance.verticals.defi import (
        DeFiProtocolEvolver,
        ProtocolConstraints
    )

    print("=" * 80)
    print("DeFi Protocol Evolver - Quickstart")
    print("=" * 80)

    # Step 2: Initialize
    print("\n[1/4] Initializing evolver...")
    evolver = DeFiProtocolEvolver(config={
        "population_size": 50,  # Start with smaller population for speed
        "generations": 25,  # Fewer generations for quick results
    })

    # Step 3: Define constraints
    print("[2/4] Setting protocol constraints...")
    constraints = ProtocolConstraints(
        max_collateral_factor=0.80,  # Max 80% collateral factor
        min_liquidation_bonus=0.05,  # Min 5% liquidation bonus
        target_utilization=0.80,  # Target 80% utilization
    )

    # Step 4: Evolve parameters
    print("[3/4] Evolving parameters (this may take a few minutes)...")
    print("     Testing against 20+ attack scenarios...")
    print("     Simulating 5+ historical events...")

    result = await evolver.evolve_protocol_parameters(
        protocol="compound",
        assets=["ETH", "USDC", "WBTC"],  # Major assets
        constraints=constraints
    )

    # Step 5: View results
    print("\n[4/4] Evolution complete! Results:")
    print("-" * 80)

    # Key metrics
    print(f"\nKey Metrics:")
    print(f"  [OK] Capital Efficiency: {result.capital_efficiency:.2%}")
    print(f"  [OK] Risk Score: {result.validation.risk_score:.1f}/100 (lower is better)")
    print(f"  [OK] Attack Survival: {sum(result.attack_survival.values())}/{len(result.attack_survival)} scenarios")
    print(f"  [OK] Historical Events: Survived {result.historical_performance.total_events} events")

    # Optimal parameters
    print(f"\nOptimal Parameters:")
    print(f"  Collateral Factors:")
    for asset, cf in result.parameters.collateral_factors.items():
        print(f"    {asset}: {cf:.2%}")

    print(f"\n  Oracle Configuration:")
    print(f"    Type: {result.parameters.price_oracle_type}")
    print(f"    Circuit Breaker: {result.parameters.circuit_breaker_threshold:.2%}")
    print(f"    Min Liquidity Required: ${result.parameters.min_liquidity_required:,.0f}")

    # Recommendations
    print(f"\nRecommendations:")
    if result.validation.risk_score < 30:
        print(f"  [OK] Excellent security profile - parameters are conservative")
    elif result.validation.risk_score < 50:
        print(f"  ! Good security profile - monitor in production")
    else:
        print(f"  ⚠ Moderate risk - consider tightening parameters")

    if result.capital_efficiency > 0.75:
        print(f"  [OK] High capital efficiency - good utilization")
    elif result.capital_efficiency > 0.60:
        print(f"  ! Moderate capital efficiency - acceptable")
    else:
        print(f"  ⚠ Low capital efficiency - parameters may be too conservative")

    print("\n" + "=" * 80)
    print("Quickstart complete! See examples.py for more advanced usage.")
    print("=" * 80)

    return result


if __name__ == "__main__":
    asyncio.run(quickstart())
