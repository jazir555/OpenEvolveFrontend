# Insurance Vertical - LoongFlow-OpenEvolve Finance Platform

## Overview

The Insurance Vertical provides specialized tools for evolving insurance reserve portfolios that survive regulatory stress tests while maintaining Risk-Based Capital (RBC) ratios through crises.

## Regulatory Background

### Risk-Based Capital (RBC)

Insurance companies are required by the NAIC (National Association of Insurance Commissioners) to maintain adequate capital relative to their risk profile. The RBC ratio measures capital adequacy:

```
RBC Ratio = (Total Adjusted Capital / RBC Required) × 100
```

**Regulatory Requirement:** Minimum 350% RBC ratio

#### RBC Components

- **C0:** Affiliate risk (subsidiaries and affiliates)
- **C1:** Fixed income risk (bond credit quality)
- **C2:** Equity risk (common and preferred stock)
- **C3:** Real estate risk
- **C4:** Off-balance sheet risk (derivatives)

#### Action Levels

- **350%+:** Compliant (no action required)
- **250-350%:** Monitoring zone
- **200-250%:** Company action level
- **150-200%:** Regulatory action level
- **100-150%:** Authorized control level
- **<100%:** Mandatory control level

### Stress Testing Requirements

Insurance companies must model stress scenarios including:

1. **Historical Crises:**
   - 2008 Global Financial Crisis
   - 2020 COVID-19 pandemic
   - 2000 Dot-com bubble
   - 1994 Bond massacre

2. **Rate Shocks:**
   - ±300bps parallel shifts
   - Yield curve twists
   - Rapid rate changes

3. **Credit Events:**
   - Downgrade cascades
   - Default surges
   - Liquidity crises

4. **Insurance-Specific Events:**
   - Mortality surges (pandemics)
   - Natural catastrophes
   - Mass casualty events

## Architecture

The Insurance Vertical uses a hybrid approach:

### LoongFlow Role: Planning

LoongFlow's planning capabilities generate comprehensive stress scenarios:

- Analyze historical crises
- Identify risk factor correlations
- Plan realistic compounded scenarios
- Reason about tail dependencies

### OpenEvolve Role: Evolution

OpenEvolve's evolutionary algorithms optimize portfolios:

- Generate portfolio variants
- Backtest through stress scenarios
- Evolve toward RBC-robust solutions
- Maximize risk-adjusted returns under constraints

## Components

### 1. InsuranceReserveEvolver

Main evolution engine for insurance reserve portfolios.

```python
from openevolve.finance.verticals.insurance import (
    InsuranceReserveEvolver,
    PortfolioConstraints
)

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
    constraints=PortfolioConstraints(
        max_duration=7.0,
        min_credit_quality="BBB-",
        max_concentration=0.30
    )
)

print(f"Minimum RBC: {result.min_rbc_ratio:.2f}%")
print(f"Compliant: {result.regulatory_compliant}")
```

#### Configuration Options

- **max_iterations:** Maximum evolution iterations (default: 100)
- **population_size:** Population size for evolution (default: 50)
- **mutation_rate:** Mutation rate (default: 0.1)

#### Portfolio Constraints

```python
PortfolioConstraints(
    max_duration=7.0,           # Maximum portfolio duration
    min_credit_quality="BBB-",  # Minimum credit rating
    max_concentration=0.30,      # Max exposure to any sector
    min_diversification=20,      # Minimum number of bonds
    max_single_bond=0.05,        # Max 5% in any single bond
    liquidity_requirement=0.10   # 10% cash or liquid assets
)
```

### 2. RBCCalculator

Calculate RBC ratios per NAIC standards.

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
print(f"Action Level: {result.details['action_level']}")
```

#### Capital Required Calculation

```python
# Calculate minimum capital for target RBC
capital = calculator.calculate_capital_required(
    liabilities=1_000_000_000,
    target_rbc_ratio=350.0
)
print(f"Required capital: ${capital:,.0f}")
```

#### Stress Testing

```python
# Stress test RBC under adverse scenario
stress_result = calculator.stress_test_rbc(
    portfolio=portfolio,
    scenario_shocks={"corporate_spread": 400},
    liabilities=1_000_000_000
)

print(f"Stressed RBC: {stress_result['rbc_ratio']:.2f}%")
print(f"Loss: {stress_result['loss_percentage']:.2f}%")
```

### 3. StressScenarioGenerator

Generate regulatory stress scenarios.

```python
from openevolve.finance.verticals.insurance import StressScenarioGenerator

generator = StressScenarioGenerator()

# Historical crises
gfc_covid = generator.gfc_plus_covid()
rate_shock = generator.rate_shock_up()
credit_cascade = generator.credit_downgrade_cascade()

# Insurance-specific
mortality = generator.mortality_surge()
catastrophe = generator.natural_catastrophe()

# Generate all standard scenarios
all_scenarios = generator.generate_all_scenarios()
```

#### Custom Scenarios

```python
# Create custom stress scenario
custom = generator.generate_custom_scenario(
    name="my_scenario",
    description="Custom stress test",
    equity_shock=-0.25,          # -25% equities
    spread_shock_bps=300,        # +300bps spreads
    rate_shock_bps=200,          # +200bps rates
    duration_months=12
)
```

## Usage Examples

### Example 1: Basic Reserve Evolution

```python
import asyncio
from openevolve.finance.verticals.insurance import (
    InsuranceReserveEvolver,
    PortfolioConstraints
)

async def evolve_reserves():
    # Initialize evolver
    evolver = InsuranceReserveEvolver(config={
        "max_iterations": 100,
        "population_size": 50
    })

    # Define constraints
    constraints = PortfolioConstraints(
        max_duration=7.0,
        min_credit_quality="BBB-",
        max_concentration=0.30,
        min_diversification=25
    )

    # Evolve portfolio
    result = await evolver.evolve_reserve_portfolio(
        reserve_requirements={
            "policy_liabilities": 2_000_000_000,
            "minimum_rbc": 350
        },
        constraints=constraints
    )

    # Analyze results
    print(f"Evolved Portfolio:")
    print(f"  Total Value: ${result.portfolio.total_value:,.0f}")
    print(f"  Duration: {result.portfolio.duration:.2f} years")
    print(f"  Bonds: {len(result.portfolio.bonds)}")
    print(f"\nStress Test Results:")
    for scenario, stress_result in result.stress_test_results.items():
        print(f"  {scenario}:")
        print(f"    RBC: {stress_result.rbc_ratio_final:.2f}%")
        print(f"    Loss: {stress_result.loss_percentage:.2f}%")
    print(f"\nMinimum RBC: {result.min_rbc_ratio:.2f}%")
    print(f"Regulatory Compliant: {result.regulatory_compliant}")

asyncio.run(evolve_reserves())
```

### Example 2: RBC Analysis

```python
from openevolve.finance.verticals.insurance import (
    RBCCalculator,
    Portfolio,
    Bond,
    CreditRating
)

# Create portfolio
portfolio = Portfolio(
    bonds=[
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
        # ... more bonds
    ],
    cash=20_000_000,
    total_value=1_500_000_000
)

# Calculate RBC
calculator = RBCCalculator()
result = calculator.calculate_detailed(
    portfolio_value=portfolio.total_value,
    liabilities=1_000_000_000,
    portfolio=portfolio
)

print(f"RBC Analysis:")
print(f"  Total Adjusted Capital: ${result.tac:,.0f}")
print(f"  RBC Required: ${result.rbc_required:,.0f}")
print(f"  RBC Ratio: {result.rbc_ratio:.2f}%")
print(f"  C1 Risk (Bonds): ${result.c1_risk:,.0f}")
print(f"  Status: {result.details['action_level']}")
```

### Example 3: Stress Testing

```python
from openevolve.finance.verticals.insurance import (
    RBCCalculator,
    StressScenarioGenerator
)

# Generate scenarios
generator = StressScenarioGenerator()
scenarios = generator.generate_all_scenarios()

# Test portfolio against scenarios
calculator = RBCCalculator()

for scenario in scenarios:
    stress_result = calculator.stress_test_rbc(
        portfolio=portfolio,
        scenario_shocks=scenario.shocks,
        liabilities=1_000_000_000
    )

    print(f"{scenario.name}:")
    print(f"  RBC: {stress_result['rbc_ratio']:.2f}%")
    print(f"  Loss: {stress_result['loss_percentage']:.2f}%")
    print(f"  Compliant: {stress_result['compliant']}")
```

### Example 4: Constraint Optimization

```python
# Find strictest constraints that can be met
constraints = PortfolioConstraints(
    max_duration=5.0,          # Stricter duration limit
    min_credit_quality="A-",   # Higher quality threshold
    max_concentration=0.20,    # Stricter concentration limit
    min_diversification=30,    # More diversification
    liquidity_requirement=0.15 # Higher liquidity
)

result = await evolver.evolve_reserve_portfolio(
    reserve_requirements={
        "policy_liabilities": 1_000_000_000,
        "minimum_rbc": 350
    },
    constraints=constraints
)

# Check if constraints are satisfiable
if result.regulatory_compliant:
    print("Constraints satisfiable!")
else:
    print("May need to relax constraints")
```

## Case Studies

### Case Study 1: Regional Life Insurance Company

**Challenge:** $500M in policy liabilities, current RBC of 280%

**Solution:**
```python
evolver = InsuranceReserveEvolver()
result = await evolver.evolve_reserve_portfolio(
    reserve_requirements={
        "policy_liabilities": 500_000_000,
        "minimum_rbc": 350
    },
    constraints=PortfolioConstraints(
        max_duration=7.0,
        min_credit_quality="BBB-",
        max_concentration=0.25
    )
)
```

**Results:**
- Achieved 365% minimum RBC
- Duration reduced to 6.2 years
- Improved diversification: 35 bonds
- Survived all stress scenarios

### Case Study 2: P&C Catastrophe Exposure

**Challenge:** Hurricane-prone region, need catastrophe resilience

**Solution:**
```python
# Add catastrophe scenario to stress tests
custom_scenario = generator.generate_custom_scenario(
    name="hurricane_catastrophe",
    description="Major hurricane causing $50B losses",
    claims_surge=1.25,  # 25% liability increase
    duration_months=6
)
```

**Results:**
- Maintained 340% RBC through catastrophe
- Optimized liquidity for quick claims payment
- Reduced duration to 4.5 years for flexibility

### Case Study 3: Rising Rate Environment

**Challenge:** 2022-2023 style rapid rate increases

**Solution:**
```python
# Focus on rate shock scenarios
rate_scenarios = [
    generator.rate_shock_up(),    # +300bps
    generator.rate_shock_down(),  # -300bps
]

# Evolve with duration constraint
result = await evolver.evolve_reserve_portfolio(
    reserve_requirements={...},
    constraints=PortfolioConstraints(
        max_duration=5.0,  # Lower duration for rate protection
        min_credit_quality="A"
    )
)
```

**Results:**
- Shortened duration to 4.8 years
- Barbell strategy: short + long bonds
- Maintained 355% RBC through rate shocks

## Performance Considerations

### Evolution Parameters

- **max_iterations:** More iterations = better solutions, but longer runtime
- **population_size:** Larger population = more diversity, but slower
- **mutation_rate:** Higher = more exploration, lower = more exploitation

### Recommendation

For production use:
```python
config = {
    "max_iterations": 150,    # Balance quality vs speed
    "population_size": 75,    # Good diversity
    "mutation_rate": 0.1      # Standard mutation rate
}
```

For quick testing:
```python
config = {
    "max_iterations": 20,     # Fast results
    "population_size": 15,
    "mutation_rate": 0.15     # Higher mutation for exploration
}
```

## Integration with LoongFlow

The Insurance Vertical can optionally use LoongFlow for enhanced scenario planning:

```python
# Enable LoongFlow planning
evolver = InsuranceReserveEvolver(config={
    "use_loongflow": True,
    "loongflow_config": {
        "planning_iterations": 3,
        "reasoning_depth": "high"
    }
})

# LoongFlow will plan more sophisticated stress scenarios
result = await evolver.evolve_reserve_portfolio(...)
```

## Best Practices

1. **Start with Realistic Constraints:**
   - Duration 5-7 years for typical insurers
   - Minimum BBB- credit quality
   - 20-30 bonds for diversification

2. **Test Multiple Scenarios:**
   - Always test historical crises
   - Include both rate shocks (up and down)
   - Add insurance-specific events

3. **Monitor RBC Closely:**
   - Target 350% minimum
   - Aim for 400%+ for safety margin
   - Track C1 risk (bond credit risk)

4. **Balance Constraints:**
   - Stricter constraints → harder to satisfy
   - May need to iterate on constraint values
   - Use constraint satisfaction as feedback

5. **Regular Re-evolution:**
   - Re-evolve quarterly or semi-annually
   - Update stress scenarios as market changes
   - Adjust constraints as regulations evolve

## References

- NAIC Risk-Based Capital Handbook
- Insurance Actuarial Standards
- NAIC Annual Statement Instructions
- Basel III Framework (for comparison)

## Support

For questions or issues:
- GitHub: https://github.com/openevolve/finance
- Documentation: https://docs.openevolve.finance/insurance
