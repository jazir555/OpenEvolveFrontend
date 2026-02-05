# Insurance Vertical

Insurance reserve portfolio evolution with regulatory stress testing and Risk-Based Capital (RBC) compliance.

## Overview

The Insurance Vertical provides specialized tools for insurance companies to evolve bond portfolios that:

- Survive regulatory stress tests (2008 GFC, COVID, rate shocks, etc.)
- Maintain RBC ratios ≥350% through crises
- Satisfy portfolio constraints (duration, credit quality, diversification)
- Optimize risk-adjusted returns under regulatory constraints

## Quick Start

```python
import asyncio
from openevolve.finance.verticals.insurance import (
    InsuranceReserveEvolver,
    PortfolioConstraints
)

async def main():
    # Initialize evolver
    evolver = InsuranceReserveEvolver()

    # Evolve portfolio
    result = await evolver.evolve_reserve_portfolio(
        reserve_requirements={
            "policy_liabilities": 1_000_000_000,
            "minimum_rbc": 350
        },
        constraints=PortfolioConstraints(
            max_duration=7.0,
            min_credit_quality="BBB-"
        )
    )

    print(f"Minimum RBC: {result.min_rbc_ratio:.2f}%")
    print(f"Compliant: {result.regulatory_compliant}")

asyncio.run(main())
```

## Components

### 1. InsuranceReserveEvolver

Main evolution engine for insurance reserve portfolios.

```python
evolver = InsuranceReserveEvolver(config={
    "max_iterations": 100,
    "population_size": 50,
    "mutation_rate": 0.1
})

result = await evolver.evolve_reserve_portfolio(
    reserve_requirements={
        "policy_liabilities": 1_000_000_000,
        "minimum_rbc": 350
    },
    constraints=PortfolioConstraints(...)
)
```

**Configuration:**
- `max_iterations`: Maximum evolution iterations (default: 100)
- `population_size`: Population size for evolution (default: 50)
- `mutation_rate`: Mutation rate (default: 0.1)

**Returns:**
- `InsuranceEvolutionResult` with evolved portfolio and stress test results

### 2. RBCCalculator

Calculate Risk-Based Capital per NAIC standards.

```python
from openevolve.finance.verticals.insurance import RBCCalculator

calculator = RBCCalculator()

# Simple calculation
rbc_ratio = calculator.calculate(
    portfolio_value=1_500_000_000,
    liabilities=1_000_000_000,
    portfolio=portfolio
)

# Detailed breakdown
result = calculator.calculate_detailed(
    portfolio_value=1_500_000_000,
    liabilities=1_000_000_000,
    portfolio=portfolio
)

print(f"RBC Ratio: {result.rbc_ratio:.2f}%")
print(f"C1 Risk: ${result.c1_risk:,.0f}")
print(f"Status: {result.details['action_level']}")
```

**Methods:**
- `calculate()`: Quick RBC ratio calculation
- `calculate_detailed()`: Full breakdown with risk components
- `calculate_capital_required()`: Minimum capital for target RBC
- `stress_test_rbc()`: Stress test RBC under adverse scenario

### 3. StressScenarioGenerator

Generate regulatory stress test scenarios.

```python
from openevolve.finance.verticals.insurance import StressScenarioGenerator

generator = StressScenarioGenerator()

# Predefined scenarios
gfc_covid = generator.gfc_plus_covid()
rate_shock = generator.rate_shock_up()
credit_cascade = generator.credit_downgrade_cascade()
mortality = generator.mortality_surge()

# All scenarios
all_scenarios = generator.generate_all_scenarios()

# Custom scenario
custom = generator.generate_custom_scenario(
    name="my_scenario",
    description="Custom stress test",
    equity_shock=-0.25,
    spread_shock_bps=300
)
```

**Available Scenarios:**
- `gfc_plus_covid()`: Compounded 2008 GFC + COVID crisis
- `rate_shock_up()`: +300bps rate shock
- `rate_shock_down()`: -300bps rate shock
- `credit_downgrade_cascade()`: Mass downgrade event
- `mortality_surge()`: 20% excess mortality
- `natural_catastrophe()`: Major natural disaster

## Portfolio Constraints

```python
from openevolve.finance.verticals.insurance import PortfolioConstraints

constraints = PortfolioConstraints(
    max_duration=7.0,           # Maximum portfolio duration
    min_credit_quality="BBB-",  # Minimum credit rating
    max_concentration=0.30,      # Max exposure to any sector
    min_diversification=20,      # Minimum number of bonds
    max_single_bond=0.05,        # Max 5% in any single bond
    liquidity_requirement=0.10   # 10% cash or liquid assets
)
```

## Data Models

### Portfolio

```python
from openevolve.finance.verticals.insurance import Portfolio, Bond, CreditRating

portfolio = Portfolio(
    bonds=[...],
    cash=20_000_000,
    total_value=1_500_000_000
)

# Properties
portfolio.duration        # Portfolio duration
portfolio.credit_quality  # Minimum credit quality
```

### Bond

```python
bond = Bond(
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
)
```

### CreditRating

```python
from openevolve.finance.verticals.insurance import CreditRating

ratings = [
    CreditRating.AAA,
    CreditRating.AA,
    CreditRating.A,
    CreditRating.BBB,
    CreditRating.BB,
    CreditRating.B
]

# Parse from string
rating = CreditRating.from_string("BBB-")
```

## Stress Testing

```python
# Generate scenarios
generator = StressScenarioGenerator()
scenarios = generator.generate_all_scenarios()

# Test portfolio
calculator = RBCCalculator()

for scenario in scenarios:
    result = calculator.stress_test_rbc(
        portfolio=portfolio,
        scenario_shocks=scenario.shocks,
        liabilities=1_000_000_000
    )

    print(f"{scenario.name}: RBC={result['rbc_ratio']:.2f}%")
```

## Examples

See `examples/finance/insurance_example.py` for comprehensive examples:

1. Basic reserve portfolio evolution
2. RBC calculation and analysis
3. Stress testing scenarios
4. Constraint optimization

Run examples:

```bash
python examples/finance/insurance_example.py
```

## Tests

Run test suite:

```bash
# Run all insurance tests
pytest tests/finance/verticals/insurance/ -v

# Run specific test file
pytest tests/finance/verticals/insurance/test_reserve_evolver.py -v

# Run with coverage
pytest tests/finance/verticals/insurance/ --cov=openevolve.finance.verticals.insurance -v
```

## Regulatory Background

### Risk-Based Capital (RBC)

The RBC ratio measures insurance company capital adequacy:

```
RBC Ratio = (Total Adjusted Capital / RBC Required) × 100
```

**Minimum Requirement:** 350%

#### RBC Components

- **C0:** Affiliate risk (subsidiaries)
- **C1:** Fixed income risk (bond credit quality)
- **C2:** Equity risk (stocks)
- **C3:** Real estate risk
- **C4:** Off-balance sheet risk (derivatives)

#### Action Levels

- **350%+:** Compliant (no action required)
- **250-350%:** Monitoring zone
- **200-250%:** Company action level
- **150-200%:** Regulatory action level
- **100-150%:** Authorized control level
- **<100%:** Mandatory control level

## Performance Tips

1. **Start Small:** Use smaller iteration counts for testing
2. **Scale Up:** Increase iterations for production (100-200)
3. **Population Size:** 50-100 provides good diversity
4. **Mutation Rate:** 0.1-0.15 works well for most cases

```python
# Testing config (fast)
config = {
    "max_iterations": 20,
    "population_size": 15
}

# Production config (thorough)
config = {
    "max_iterations": 150,
    "population_size": 75
}
```

## Best Practices

1. **Set Realistic Constraints:**
   - Duration: 5-7 years
   - Credit quality: BBB- minimum
   - Diversification: 20-30 bonds

2. **Test Multiple Scenarios:**
   - Always include historical crises
   - Test both rate shocks (up and down)
   - Add insurance-specific events

3. **Monitor RBC Closely:**
   - Target 350% minimum
   - Aim for 400%+ for safety margin
   - Track C1 risk (bond credit risk)

4. **Regular Re-evolution:**
   - Re-evolve quarterly or semi-annually
   - Update scenarios as market changes
   - Adjust constraints as regulations evolve

## Architecture

The Insurance Vertical uses a hybrid approach:

### LoongFlow Role: Planning

- Analyze historical crises
- Identify risk factor correlations
- Plan realistic compounded scenarios
- Reason about tail dependencies

### OpenEvolve Role: Evolution

- Generate portfolio variants
- Backtest through stress scenarios
- Evolve toward RBC-robust solutions
- Maximize risk-adjusted returns under constraints

## Documentation

Full documentation: `docs/finance/verticals/INSURANCE.md`

Topics covered:
- Regulatory background (NAIC RBC)
- Stress test methodology
- Detailed usage examples
- Case studies
- Performance considerations
- Integration with LoongFlow

## References

- NAIC Risk-Based Capital Handbook
- Insurance Actuarial Standards
- NAIC Annual Statement Instructions

## License

Part of the LoongFlow-OpenEvolve Finance Platform.

See LICENSE file for details.
