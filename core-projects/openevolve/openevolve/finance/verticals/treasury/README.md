# Corporate Treasury Vertical

Evolutionary liquidity management for corporate treasury operations.

## Overview

The Corporate Treasury vertical provides tools for optimizing corporate liquidity management. It evolves strategies that balance three competing objectives:

1. **Liquidity Survival**: Maintain sufficient liquidity through crisis scenarios
2. **Cost Minimization**: Minimize the drag on returns from holding liquid assets
3. **Robustness**: Survive a variety of stress scenarios without default

## Quick Start

```python
import asyncio
from openevolve.finance.verticals.treasury import (
    LiquidityCrisisEvolver,
    CashFlowProfile,
    LiquidityConstraints
)

async def main():
    # Define your cash flow profile
    profile = CashFlowProfile(
        daily_burn_rate=1_000_000,  # $1M/day
        volatility_std=200_000,
        seasonal_patterns={"q1": 1.1, "q2": 0.95, "q3": 0.9, "q4": 1.05}
    )

    # Define constraints
    constraints = LiquidityConstraints(
        min_liquidity_days=90,
        max_liquidity_cost=50
    )

    # Evolve strategy
    evolver = LiquidityCrisisEvolver()
    result = await evolver.evolve_liquidity_strategy(
        cash_flow_profile=profile,
        constraints=constraints
    )

    print(f"Liquidity Days: {result.liquidity_days:.1f}")
    print(f"Annual Cost: {result.annual_cost:.1f} bps")
    print(f"Robustness: {result.robustness_score:.2f}")

asyncio.run(main())
```

## Installation

The treasury vertical is included in the main OpenEvolve package:

```bash
pip install openevolve
```

## Components

### LiquidityCrisisEvolver

Main orchestrator for evolving liquidity management strategies.

```python
from openevolve.finance.verticals.treasury import LiquidityCrisisEvolver

evolver = LiquidityCrisisEvolver(config={
    'n_variants': 100  # Number of allocation variants to test
})

result = await evolver.evolve_liquidity_strategy(
    cash_flow_profile=profile,
    constraints=constraints
)
```

### LiquidityCalculator

Calculate liquidity metrics and costs.

```python
from openevolve.finance.verticals.treasury import LiquidityCalculator

calculator = LiquidityCalculator()

# Calculate liquidity days
days = calculator.calculate_liquidity_days(
    cash=100_000_000,
    t_bills=50_000_000,
    commercial_paper=30_000_000,
    credit_line_undrawn=200_000_000,
    daily_burn_rate=1_000_000
)

# Calculate annual cost
cost = calculator.calculate_annual_cost(
    cash=100_000_000,
    t_bills=50_000_000,
    commercial_paper=30_000_000,
    credit_line_total=200_000_000,
    credit_line_used=20_000_000
)
```

### LiquidityScenarioGenerator

Generate realistic stress scenarios based on historical crises.

```python
from openevolve.finance.verticals.treasury import LiquidityScenarioGenerator

generator = LiquidityScenarioGenerator()

# Generate GFC scenario
scenario = generator.generate_gfc_credit_freeze()

# Generate all scenarios
all_scenarios = generator.generate_all_scenarios()

# Create custom scenario
custom = generator.generate_custom_scenario(
    name="custom_stress",
    description="Company-specific stress",
    duration_days=60,
    outflow_pattern="gradual_increase",
    outflow_parameters={'start': 1.0, 'peak': 3.0, 'peak_day': 30}
)
```

## Stress Scenarios

The system includes six built-in stress scenarios:

1. **GFC Credit Freeze** (2008): CP market and credit lines freeze
2. **Supplier Cascade**: Supplier default triggers payment acceleration
3. **CP Market Freeze**: Commercial paper market seizes
4. **Revenue Shock**: Sudden revenue decline with gradual recovery
5. **Capex Surprise**: Urgent unplanned capital expenditure
6. **Combined Stress**: Multiple stressors simultaneously (worst case)

## Example

Run the included example:

```bash
cd openevolve
python examples/treasury_example.py
```

Expected output:

```
================================================================================
Corporate Treasury Liquidity Management Example
================================================================================

Step 1: Define Cash Flow Profile
--------------------------------------------------------------------------------
Daily Burn Rate: $5,000,000
Volatility: ±$1,000,000
Seasonal Patterns: {'q1': 1.1, 'q2': 0.95, 'q3': 0.9, 'q4': 1.05}
Capex Events: 2

Step 2: Define Liquidity Constraints
--------------------------------------------------------------------------------
Minimum Liquidity: 90 days
Maximum Cost: 75 bps
Max Credit Line Usage: 50%

Step 3: Initialize Evolver
--------------------------------------------------------------------------------
Number of variants to test: 100

Step 4: Evolve Liquidity Strategy
--------------------------------------------------------------------------------
Running evolution...

Step 5: Results
--------------------------------------------------------------------------------

Liquidity Metrics:
  Normal Liquidity: 177.7 days
  Stress Liquidity: 80.4 days
  Annual Cost: 306.8 bps
  Robustness Score: 0.54

Optimal Allocation:
  Cash: $259,098,138
  T-bills: $150,637,537
  Commercial Paper: $40,264,325
  Credit Line: $450,000,000

Stress Test Results:
  gfc_credit_freeze: [FAIL] Failed
  supplier_cascade: [OK] Survived
  cp_market_freeze: [OK] Survived
  revenue_shock: [OK] Survived
  capex_surprise: [OK] Survived
  combined_stress: [FAIL] Failed

...
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest openevolve/finance/verticals/treasury/tests/test_liquidity_evolver.py -v

# Run specific test class
pytest openevolve/finance/verticals/treasury/tests/test_liquidity_evolver.py::TestLiquidityCalculator -v

# Run integration test
pytest openevolve/finance/verticals/treasury/tests/test_liquidity_evolver.py::TestIntegration -v -s
```

## Documentation

Comprehensive documentation is available at:

- [TREASURY.md](../../../docs/finance/verticals/TREASURY.md) - Full documentation with case studies and best practices

## Performance

Runtime estimates (6 scenarios):

| Variants | Time |
|----------|------|
| 50 | 2-5 minutes |
| 100 | 5-10 minutes |
| 200 | 10-20 minutes |
| 500 | 30-60 minutes |

## Best Practices

1. **Start with realistic constraints** based on your company's situation
2. **Test multiple scenarios** to ensure robustness
3. **Validate results** by checking robustness scores
4. **Iterate on constraints** to find the optimal trade-off

## License

See the main OpenEvolve LICENSE file.

## Authors

AI Architecture Team

## Version

1.0.0 (2026-01-30)
