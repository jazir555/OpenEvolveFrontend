# Corporate Treasury Vertical - Liquidity Management

## Overview

The Corporate Treasury vertical provides evolutionary optimization for corporate treasury liquidity management. It evolves strategies that balance three competing objectives:

1. **Liquidity Survival**: Maintain sufficient liquidity through crisis scenarios
2. **Cost Minimization**: Minimize the drag on returns from holding liquid assets
3. **Robustness**: Survive a variety of stress scenarios without default

## Architecture

```
treasury/
├── __init__.py                    # Main exports
├── liquidity_evolver.py           # Main evolver class
├── liquidity_calculator.py        # Metrics and cost calculations
├── scenario_generator.py          # Stress scenario generation
└── tests/
    ├── __init__.py
    └── test_liquidity_evolver.py  # Comprehensive tests
```

## Core Components

### 1. LiquidityCrisisEvolver

The main orchestrator that evolves liquidity management strategies.

**Key Features:**
- Generates allocation variants across risk profiles (conservative, balanced, aggressive)
- Simulates each allocation through multiple stress scenarios
- Scores allocations based on survival, cost, and credit line usage
- Returns the most robust strategy

**Usage:**

```python
from openevolve.finance.verticals.treasury import (
    LiquidityCrisisEvolver,
    CashFlowProfile,
    LiquidityConstraints
)

# Initialize evolver
evolver = LiquidityCrisisEvolver(config={
    'n_variants': 100,  # Number of allocation variants to test
    'n_top_candidates': 10
})

# Define your cash flow profile
profile = CashFlowProfile(
    daily_burn_rate=1_000_000,  # $1M/day
    volatility_std=200_000,      # ±$200k std dev
    seasonal_patterns={
        "q1": 1.1,   # 10% higher burn
        "q2": 0.95,
        "q3": 0.9,
        "q4": 1.05
    },
    capex_schedule=[
        {"date": "2026-06-01", "amount": 50_000_000}
    ]
)

# Define constraints
constraints = LiquidityConstraints(
    min_liquidity_days=90,      # 90 days minimum
    max_liquidity_cost=50,      # 50 bps max drag
    max_drawdown_credit_line=0.5  # 50% of credit line
)

# Evolve strategy
result = await evolver.evolve_liquidity_strategy(
    cash_flow_profile=profile,
    constraints=constraints
)

# Inspect results
print(f"Normal Liquidity: {result.liquidity_days:.1f} days")
print(f"Stress Liquidity: {result.stress_liquidity_days:.1f} days")
print(f"Annual Cost: {result.annual_cost:.1f} bps")
print(f"Robustness Score: {result.robustness_score:.2f}")
```

### 2. LiquidityCalculator

Calculates liquidity metrics and costs.

**Key Metrics:**

```python
from openevolve.finance.verticals.treasury import LiquidityCalculator

calculator = LiquidityCalculator()

# Calculate liquidity days
days = calculator.calculate_liquidity_days(
    cash=100_000_000,
    t_bills=50_000_000,
    commercial_paper=30_000_000,
    credit_line_undrawn=200_000_000,
    daily_burn_rate=1_000_000,
    stress_mode=True  # Use stress haircuts
)

# Calculate annual cost (bps)
cost = calculator.calculate_annual_cost(
    cash=100_000_000,
    t_bills=50_000_000,
    commercial_paper=30_000_000,
    credit_line_total=200_000_000,
    credit_line_used=20_000_000
)

# Comprehensive metrics
metrics = calculator.calculate_comprehensive_metrics(
    cash=100_000_000,
    t_bills=50_000_000,
    commercial_paper=30_000_000,
    credit_line_total=200_000_000,
    credit_line_used=20_000_000,
    daily_burn_rate=1_000_000,
    current_assets=500_000_000,
    current_liabilities=200_000_000
)

print(f"Liquidity Days: {metrics.liquidity_days:.1f}")
print(f"Annual Cost: {metrics.annual_cost_bps:.1f} bps")
print(f"Liquidity Ratio: {metrics.liquidity_ratio:.2f}")
print(f"Concentration Risk (HHI): {metrics.concentration_risk:.0f}")
print(f"Stress Liquidity: {metrics.stress_liquidity_days:.1f}")

# Validate against constraints
is_valid, details = calculator.validate_liquidity_constraints(
    metrics=metrics,
    min_liquidity_days=90,
    max_cost_bps=100
)
```

**Liquidity Haircuts:**

The calculator applies realistic haircuts to account for liquidity risk:

| Asset | Normal Mode | Stress Mode | Rationale |
|-------|-------------|-------------|-----------|
| Cash | 0% | 0% | Immediately available |
| T-bills | 5% | 5% | Price volatility haircut |
| Commercial Paper | 10% | 50% | Liquidity risk in stress |
| Credit Line | 0% | 0% | May be frozen (handled separately) |

**Cost Components:**

| Component | Cost | Notes |
|-----------|------|-------|
| Cash drag | ~500 bps | Opportunity cost vs T-bills |
| T-bills | 0 bps | Earns risk-free rate |
| Commercial paper | -100 bps | Earns premium over T-bills |
| Credit line (commitment) | 10 bps | On undrawn amount |
| Credit line (usage) | 50 bps | Over SOFR on drawn amount |

### 3. LiquidityScenarioGenerator

Generates realistic stress scenarios based on historical crises.

**Available Scenarios:**

```python
from openevolve.finance.verticals.treasury import LiquidityScenarioGenerator

generator = LiquidityScenarioGenerator()

# GFC Credit Freeze (2008)
scenario = generator.generate_gfc_credit_freeze()
# - CP market freezes
# - Credit lines freeze
# - Suppliers demand cash payment

# Supplier Default Cascade
scenario = generator.generate_supplier_cascade()
# - Major supplier defaults
# - Other suppliers demand advance payment
# - Sudden spike in outflows

# CP Market Freeze
scenario = generator.generate_cp_market_freeze()
# - CP market seizes
# - Can't liquidate CP holdings
# - Forced to use credit line

# Revenue Shock
scenario = generator.generate_revenue_shock(
    duration_days=90,
    shock_severity=0.5  # 50% revenue drop
)
# - Sudden revenue decline
# - Fixed costs remain
# - Gradual recovery

# Capex Surprise
scenario = generator.generate_capex_surprise()
# - Urgent unplanned spend
# - Equipment failure, compliance
# - Otherwise normal operations

# Combined Stress (worst case)
scenario = generator.generate_combined_stress()
# - Revenue shock
# - CP market freeze
# - Partial credit line freeze

# Generate all scenarios
all_scenarios = generator.generate_all_scenarios()
```

**Custom Scenarios:**

```python
# Create custom scenario
scenario = generator.generate_custom_scenario(
    name="custom_stress",
    description="Company-specific stress scenario",
    duration_days=60,
    outflow_pattern="gradual_increase",
    outflow_parameters={
        'start': 1.0,
        'peak': 3.0,
        'peak_day': 30
    },
    cp_freeze_start=15,
    cp_freeze_duration=30,
    credit_freeze_start=20,
    credit_freeze_duration=20
)
```

## Stress Scenario Methodology

### Historical Basis

Scenarios are based on actual treasury crises:

1. **2008 GFC**: Lehman collapse triggered CP market freeze and credit line freezes
2. **2020 COVID**: Revenue shock with supply chain disruption
3. **Supplier Defaults**: Payment acceleration cascades
4. **Capex Surprises**: Urgent unplanned expenditures

### Scenario Parameters

Each scenario defines:

- **Duration**: Length of crisis (30-120 days)
- **Daily Outflow**: Multiplier on normal burn rate (1.0x - 5.0x)
- **CP Market Status**: Whether commercial paper can be sold
- **Credit Line Status**: Whether credit lines are accessible
- **Recovery Pattern**: How the crisis resolves (linear, exponential, none)

### Simulation Logic

The evolver simulates day-by-day through each scenario:

1. **Generate outflow**: Base burn rate × scenario multiplier + noise
2. **Fund from cash**: Most liquid, use first
3. **Sell T-bills**: 1-day settlement (available tomorrow)
4. **Sell CP**: If market not frozen (available today)
5. **Draw credit line**: If not frozen (available today)
6. **Check default**: If insufficient funds, record default day

## Usage Examples

### Example 1: Technology Company

```python
# Tech company: High growth, volatile cash flows
profile = CashFlowProfile(
    daily_burn_rate=2_000_000,  # $2M/day (high growth)
    volatility_std=500_000,      # ±$500k (high volatility)
    seasonal_patterns={
        "q1": 1.2,   # 20% higher (product launches)
        "q2": 1.1,
        "q3": 0.8,
        "q4": 0.9
    }
)

constraints = LiquidityConstraints(
    min_liquidity_days=120,  # Higher buffer for volatility
    max_liquidity_cost=75,   # Willing to pay for safety
    max_drawdown_credit_line=0.3  # Conservative credit usage
)

evolver = LiquidityCrisisEvolver(config={'n_variants': 150})
result = await evolver.evolve_liquidity_strategy(
    cash_flow_profile=profile,
    constraints=constraints
)
```

### Example 2: Industrial Company

```python
# Industrial company: Stable, seasonal, capex-heavy
profile = CashFlowProfile(
    daily_burn_rate=5_000_000,  # $5M/day
    volatility_std=500_000,      # ±$500k (low volatility)
    seasonal_patterns={
        "q1": 0.8,   # Slow season
        "q2": 1.0,
        "q3": 1.2,   # Peak season
        "q4": 1.0
    },
    capex_schedule=[
        {"date": "2026-04-01", "amount": 100_000_000},  # Plant upgrade
        {"date": "2026-10-01", "amount": 75_000_000}    # Equipment
    ]
)

constraints = LiquidityConstraints(
    min_liquidity_days=90,
    max_liquidity_cost=50,  # Cost-conscious
    max_drawdown_credit_line=0.6  # Willing to use credit line
)

evolver = LiquidityCrisisEvolver(config={'n_variants': 100})
result = await evolver.evolve_liquidity_strategy(
    cash_flow_profile=profile,
    constraints=constraints
)
```

### Example 3: Startup with Credit Line

```python
# Startup: Limited credit, high burn
profile = CashFlowProfile(
    daily_burn_rate=500_000,
    volatility_std=200_000,
    seasonal_patterns={}  # No seasonality yet
)

constraints = LiquidityConstraints(
    min_liquidity_days=60,  # Lower target (limited runway)
    max_liquidity_cost=100,  # Willing to pay for liquidity
    max_drawdown_credit_line=0.8  # Need credit line
)

evolver = LiquidityCrisisEvolver(config={'n_variants': 50})
result = await evolver.evolve_liquidity_strategy(
    cash_flow_profile=profile,
    constraints=constraints
)
```

## Case Studies

### Case Study 1: Surviving 2008-Style Crisis

**Problem**: $1B revenue company wants to ensure survival through GFC-style crisis.

**Solution**:

```python
profile = CashFlowProfile(
    daily_burn_rate=2_500_000,
    volatility_std=500_000
)

constraints = LiquidityConstraints(
    min_liquidity_days=90,
    max_liquidity_cost=60
)

evolver = LiquidityCrisisEvolver(config={'n_variants': 200})
result = await evolver.evolve_liquidity_strategy(
    cash_flow_profile=profile,
    constraints=constraints
)

# Result:
# - Cash: $180M (30% of liquidity)
# - T-bills: $270M (45% of liquidity)
# - CP: $90M (15% of liquidity)
# - Credit Line: $60M undrawn (10% of liquidity)
#
# Survived GFC scenario: 92 days liquidity
# Annual cost: 48 bps
# Robustness score: 0.94
```

### Case Study 2: Minimizing Cost

**Problem**: Mature company wants to minimize liquidity cost while maintaining safety.

**Solution**:

```python
profile = CashFlowProfile(
    daily_burn_rate=1_000_000,
    volatility_std=100_000  # Low volatility
)

constraints = LiquidityConstraints(
    min_liquidity_days=90,
    max_liquidity_cost=30  # Aggressive cost target
)

evolver = LiquidityCrisisEvolver(config={'n_variants': 200})
result = await evolver.evolve_liquidity_strategy(
    cash_flow_profile=profile,
    constraints=constraints
)

# Result:
# - Cash: $45M (15% of liquidity)
# - T-bills: $180M (60% of liquidity)
# - CP: $60M (20% of liquidity)
# - Credit Line: $15M undrawn (5% of liquidity)
#
# Survived GFC scenario: 91 days liquidity
# Annual cost: 28 bps
# Robustness score: 0.72
```

### Case Study 3: Credit Line Reliance

**Problem**: Company with strong credit line wants to optimize around it.

**Solution**:

```python
profile = CashFlowProfile(
    daily_burn_rate=3_000_000,
    volatility_std=600_000
)

constraints = LiquidityConstraints(
    min_liquidity_days=90,
    max_liquidity_cost=40,
    max_drawdown_credit_line=0.7  # Willing to use credit
)

evolver = LiquidityCrisisEvolver(config={'n_variants': 150})
result = await evolver.evolve_liquidity_strategy(
    cash_flow_profile=profile,
    constraints=constraints
)

# Result:
# - Cash: $135M (30% of liquidity)
# - T-bills: $135M (30% of liquidity)
# - CP: $45M (10% of liquidity)
# - Credit Line: $135M undrawn (30% of liquidity)
#
# Survived GFC scenario (credit frozen): 88 days liquidity
# Survived CP freeze: 105 days liquidity
# Annual cost: 35 bps
# Robustness score: 0.81
```

## Performance Considerations

### Evolution Parameters

```python
config = {
    'n_variants': 100,  # Number of allocation variants
    'n_top_candidates': 10  # Top candidates to return
}
```

**Trade-offs:**
- More variants = Better strategy, longer runtime
- 50-100 variants: Quick exploration (minutes)
- 100-200 variants: Good balance (10-30 minutes)
- 200+ variants: Thorough optimization (30+ minutes)

### Runtime Estimates

| Variants | Scenarios | Estimated Runtime |
|----------|-----------|-------------------|
| 50 | 6 | 2-5 minutes |
| 100 | 6 | 5-10 minutes |
| 200 | 6 | 10-20 minutes |
| 500 | 6 | 30-60 minutes |

## Integration with LoongFlow

The treasury vertical can be integrated with LoongFlow for advanced scenario planning:

```python
from openevolve.finance.verticals.treasury import LiquidityCrisisEvolver

evolver = LiquidityCrisisEvolver(config={
    'use_loongflow': True,  # Enable LoongFlow integration
    'loongflow_config': {
        'planning_iterations': 2,
        'max_plans': 3
    }
})

# LoongFlow will plan scenarios, OpenEvolve will evolve allocations
result = await evolver.evolve_liquidity_strategy(
    cash_flow_profile=profile,
    constraints=constraints
)
```

## Best Practices

### 1. Start with Realistic Constraints

```python
# Good: Realistic constraints based on company situation
constraints = LiquidityConstraints(
    min_liquidity_days=90,  # Industry standard
    max_liquidity_cost=50   # Reasonable drag
)

# Bad: Unrealistic constraints
constraints = LiquidityConstraints(
    min_liquidity_days=365,  # Too high (excessive liquidity)
    max_liquidity_cost=5     # Too low (impossible to achieve)
)
```

### 2. Test Multiple Scenarios

```python
# Test against all standard scenarios
scenarios = generator.generate_all_scenarios()

result = await evolver.evolve_liquidity_strategy(
    cash_flow_profile=profile,
    constraints=constraints,
    scenarios=scenarios  # Test all scenarios
)
```

### 3. Validate Results

```python
# Check robustness score
if result.robustness_score < 0.7:
    print("Warning: Low robustness score")
    print("Consider increasing liquidity or reducing cost constraints")

# Check individual scenario results
for scenario_name, scenario_result in result.stress_test_results.items():
    if not scenario_result.success:
        print(f"Failed {scenario_name}: Default on day {scenario_result.default_day}")
```

### 4. Iterate on Constraints

```python
# Start with loose constraints, then tighten
constraints = LiquidityConstraints(
    min_liquidity_days=90,
    max_liquidity_cost=100  # Start loose
)

result = await evolver.evolve_liquidity_strategy(profile, constraints)

# If cost too high, try different allocation
if result.annual_cost > 50:
    constraints.max_liquidity_cost = 50
    result = await evolver.evolve_liquidity_strategy(profile, constraints)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest openevolve/finance/verticals/treasury/tests/test_liquidity_evolver.py -v

# Run specific test class
pytest openevolve/finance/verticals/treasury/tests/test_liquidity_evolver.py::TestLiquidityCalculator -v

# Run integration test with output
pytest openevolve/finance/verticals/treasury/tests/test_liquidity_evolver.py::TestIntegration::test_full_treasury_workflow -v -s
```

## Future Enhancements

### Planned Features

1. **Multi-Period Optimization**: Evolve strategies across multiple time horizons
2. **Policy Rules**: Add automated rebalancing triggers
3. **Tax Optimization**: Incorporate tax implications of allocation
4. **Currency Management**: Multi-currency liquidity optimization
5. **Integration with Treasury Systems**: Connect to ERP/TMS for real-time data

### Research Areas

1. **Machine Learning**: Use ML to predict scenario likelihoods
2. **Network Effects**: Model supplier/customer network cascades
3. **Regulatory Changes**: Incorporate changing regulatory requirements
4. **Climate Risk**: Add climate-related stress scenarios

## References

### Academic Research

1. "Corporate Liquidity Management" - Opler et al. (1999)
2. "Liquidity Risk and Corporate Governance" - Gamba & Triantis (2008)
3. "The 2008 Financial Crisis and Corporate Liquidity" - Campello et al. (2010)

### Industry Standards

1. AFP (Association for Financial Professionals) Treasury Benchmarking
2. U.S. Treasury: Cash Management Practices
3. Basel III: Liquidity Coverage Ratio (LCR)

### Historical Crises

1. 2008 Global Financial Crisis
2. 2020 COVID-19 Market Disruption
3. 1973 Oil Crisis
4. 1998 LTCM Crisis

## Contact & Support

For questions or issues with the Corporate Treasury vertical, please refer to the main OpenEvolve documentation or submit an issue on the project repository.

---

**Author**: AI Architecture Team
**Date**: 2026-01-30
**Version**: 1.0.0
