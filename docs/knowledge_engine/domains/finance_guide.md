# Finance Domain Guide

**Version:** 1.0
**Last Updated:** January 30, 2026

---

## Table of Contents

- [Domain Overview](#domain-overview)
- [Recommended Approach](#recommended-approach)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Domain Overview

### What Problems Does This Domain Solve?

The finance domain addresses optimization challenges in:

- **Portfolio Management** - Asset allocation, risk management, rebalancing
- **Risk Analysis** - VaR optimization, stress testing, scenario analysis
- **Algorithmic Trading** - Signal optimization, execution strategies
- **Asset Pricing** - Derivative valuation, bond pricing, option strategies
- **Credit Risk** - Default prediction, limit optimization, portfolio selection

### Unique Challenges

#### 1. Expensive Evaluations
Backtesting historical data is computationally expensive:
```python
# Single backtest takes 1-5 minutes
def evaluate_portfolio(portfolio):
    returns = backtest(portfolio, historical_data)
    return calculate_metrics(returns)
```

**Impact:** Limited evaluation budget
**Solution:** Use LoongFlow PES mode (60% fewer backtests)

#### 2. Multiple Conflicting Objectives
Finance problems rarely have a single objective:
```python
objectives = {
    "return": "maximize",
    "risk": "minimize",
    "liquidity": "maximize",
    "diversification": "maximize"
}
```

**Impact:** No single optimal solution
**Solution:** Use Multi-Objective optimization (NSGA-II)

#### 3. Regulatory Constraints
Financial products must comply with regulations:
```python
constraints = {
    "max_position_size": 0.1,  # 10% per position
    "sector_limits": {"tech": 0.3, "healthcare": 0.2},
    "leverage_limit": 2.0
}
```

**Impact:** Feasible region is constrained
**Solution:** Encode constraints in evaluation function

#### 4. Non-Stationarity
Financial markets evolve over time:
```python
# Strategy that worked in 2020 may fail in 2025
train_data = data["2015-2020"]
test_data = data["2021-2025"]
```

**Impact:** Overfitting to historical data
**Solution:** Use adversarial evolution for robustness

### Why Evolutionary Optimization?

#### Traditional Methods
- **Mean-Variance Optimization** (Markowitz)
  - Assumes normal distributions
  - Sensitive to input errors
  - Single-period model

- **Quadratic Programming**
  - Requires convex objective
  - Limited to simple constraints
  - No transaction costs

#### Evolutionary Advantages
- **No distributional assumptions**
- **Handles complex, non-convex objectives**
- **Supports arbitrary constraints**
- **Multi-objective optimization**
- **Adaptive to market regimes**

---

## Recommended Approach

### Best System: LoongFlow PES

**Why?**
- Expensive evaluations (backtesting)
- Need for directed search
- Reasoning improves solutions
- 60% fewer backtests needed

**When to Use:**
- Portfolio optimization
- Risk analysis
- Algorithmic trading
- Limited evaluation budget (<100 backtests)

### Best Mode: PES (Plan-Execute-Summarize)

**Why PES?**
1. **Planning phase** uses financial knowledge
2. **Execute phase** focuses on promising regions
3. **Summarize phase** learns what works
4. **Early stopping** saves evaluations

**Example Plan:**
```python
# LLM-generated plan
plan = """
1. Start with equal-weight portfolio
2. Identify high-return, low-risk assets
3. Increase allocation to best performers
4. Apply sector diversification constraints
5. Optimize position sizes using Kelly criterion
6. Validate with stress tests
"""
```

### Hybrid Approach: PES + Multi-Objective

For complex financial problems with multiple objectives:

```python
# Phase 1: PES for exploration
result1 = await evolve(
    problem="Optimize portfolio allocation",
    domain="finance",
    evolution_mode="pes",
    max_evaluations=30
)

# Phase 2: NSGA-II for Pareto front
result2 = await evolve(
    problem="Refine portfolio for multiple objectives",
    domain="finance",
    evolution_mode="mo",
    objectives=["return", "risk", "liquidity"],
    initial_solutions=result1['archive'],
    max_evaluations=50
)

# Result: Pareto-optimal portfolios
```

---

## Configuration

### Default Configuration

```python
from openevolve.unified import UnifiedEvolutionConfig

finance_config = UnifiedEvolutionConfig(
    # Core
    domain="finance",
    evolution_mode="pes",  # Auto-selected

    # Evaluation budget
    max_evaluations=50,
    max_iterations=30,
    evaluation_timeout=300,  # 5 minutes per backtest

    # PES parameters
    enable_planning=True,
    enable_memory=True,
    early_stopping=True,
    early_stop_threshold=0.9,

    # Financial objectives
    objectives=["return", "risk"],
    objective_weights={"return": 0.7, "risk": 0.3},

    # Constraints
    constraints={
        "max_position_size": 0.1,
        "sector_diversification": True,
        "max_leverage": 2.0,
        "min_liquidity": 0.5
    },

    # Knowledge
    enable_knowledge_engine=True,
    extract_knowledge=True,

    # Gauntlet
    enable_gauntlet=True,
    gauntlet_rounds=["loongflow", "red_team", "gold_team"]
)
```

### Sub-Domain Configurations

#### Portfolio Optimization

```python
portfolio_config = UnifiedEvolutionConfig(
    domain="finance",
    evolution_mode="pes",
    max_evaluations=50,

    # Portfolio-specific
    objectives=["return", "risk", "diversification"],
    constraints={
        "max_position_size": 0.05,  # 5% per position
        "sector_limits": {
            "technology": 0.25,
            "healthcare": 0.20,
            "finance": 0.20,
            "other": 0.35
        },
        "min_positions": 20,
        "max_positions": 50
    },

    # Evaluation
    evaluation_horizon="1year",
    rebalance_frequency="monthly",
    transaction_costs=0.001  # 10 bps
)
```

#### Risk Analysis

```python
risk_config = UnifiedEvolutionConfig(
    domain="finance",
    evolution_mode="adversarial",  # Stress testing
    max_evaluations=100,

    # Risk-specific
    objectives=["var", "cvar", "max_drawdown"],
    constraints={
        "var_limit": 0.05,  # 5% VaR
        "cvar_limit": 0.10,  # 10% CVaR
        "beta_range": [0.8, 1.2]
    },

    # Stress testing
    stress_scenarios=[
        "2008_crisis",
        "covid_crash",
        "inflation_spike",
        "rate_hike"
    ],

    # Adversarial
    adversarial_rounds=20,
    red_team_models=["gpt4", "claude"]
)
```

#### Algorithmic Trading

```python
trading_config = UnifiedEvolutionConfig(
    domain="trading",  # Trading subdomain
    evolution_mode="adversarial",
    max_evaluations=100,

    # Strategy parameters
    strategy_type="momentum",
    lookback_range=[10, 50],
    threshold_range=[0.5, 2.0],

    # Objectives
    objectives=["sharpe_ratio", "max_drawdown", "win_rate"],

    # Constraints
    constraints={
        "max_positions": 10,
        "position_sizing": "kelly",
        "stop_loss": 0.05,
        "take_profit": 0.15
    },

    # Robustness
    enable_gauntlet=True,
    adversarial_market_scenarios=True
)
```

### Key Parameters to Tune

#### 1. Evaluation Budget
```python
# Limited budget (<50 backtests)
max_evaluations=30
evolution_mode="pes"

# Medium budget (50-100 backtests)
max_evaluations=75
evolution_mode="pes"

# Large budget (>100 backtests)
max_evaluations=150
evolution_mode="adversarial"  # More robust
```

#### 2. Planning vs. Exploration
```python
# Exploit known patterns
enable_planning=True
enable_memory=True

# Explore novel strategies
enable_planning=False
mutation_rate=0.2  # Higher mutation
```

#### 3. Risk Tolerance
```python
# Conservative
constraints={"max_drawdown": 0.10}
objective_weights={"return": 0.4, "risk": 0.6}

# Aggressive
constraints={"max_drawdown": 0.30}
objective_weights={"return": 0.8, "risk": 0.2}
```

---

## Evaluation Metrics

### Domain-Specific Metrics

#### Return Metrics
```python
def calculate_return_metrics(portfolio):
    return {
        "total_return": (final_value - initial_value) / initial_value,
        "annual_return": total_return ** (1 / years) - 1,
        "monthly_return": monthly_returns.mean(),
        "daily_return": daily_returns.mean()
    }
```

#### Risk Metrics
```python
def calculate_risk_metrics(portfolio):
    returns = portfolio.returns

    return {
        "volatility": returns.std() * sqrt(252),
        "var_95": returns.quantile(0.05),  # 5% worst case
        "cvar_95": returns[returns <= var_95].mean(),  # Average of worst 5%
        "max_drawdown": max_drawdown(returns),
        "beta": covariance(returns, market) / variance(market)
    }
```

#### Risk-Adjusted Metrics
```python
def calculate_risk_adjusted_metrics(portfolio):
    returns = portfolio.returns
    rf = risk_free_rate

    return {
        "sharpe_ratio": (returns.mean() - rf) / returns.std(),
        "sortino_ratio": (returns.mean() - rf) / downside_deviation(returns),
        "treynor_ratio": (returns.mean() - rf) / beta,
        "information_ratio": alpha / tracking_error
    }
```

#### Portfolio Metrics
```python
def calculate_portfolio_metrics(portfolio):
    return {
        "diversification": herfindahl_index(portfolio.weights),
        "concentration": max_weight(portfolio.weights),
        "turnover": sum(abs(weight_changes)),
        "liquidity": avg_daily_volume / portfolio_value
    }
```

### What Good Targets Look Like

#### Conservative Portfolio
```python
{
    "annual_return": 0.08,  # 8% annual
    "volatility": 0.10,     # 10% volatility
    "sharpe_ratio": 0.8,
    "max_drawdown": -0.12,  # -12% max drawdown
    "var_95": -0.02        # -2% daily VaR
}
```

#### Balanced Portfolio
```python
{
    "annual_return": 0.12,  # 12% annual
    "volatility": 0.15,     # 15% volatility
    "sharpe_ratio": 1.0,
    "max_drawdown": -0.18,  # -18% max drawdown
    "var_95": -0.03        # -3% daily VaR
}
```

#### Aggressive Portfolio
```python
{
    "annual_return": 0.18,  # 18% annual
    "volatility": 0.25,     # 25% volatility
    "sharpe_ratio": 1.2,
    "max_drawdown": -0.30,  # -30% max drawdown
    "var_95": -0.05        # -5% daily VaR
}
```

---

## Examples

### Example 1: Simple Portfolio Optimization

**Problem:** Optimize portfolio allocation for S&P 500 stocks

```python
from openevolve.unified import evolve

problem = """
Optimize portfolio allocation for S&P 500 stocks.

Objective:
- Maximize Sharpe ratio
- Minimize maximum drawdown

Constraints:
- Max position size: 5%
- Min 20 positions
- Max 50 positions
- Sector diversification: max 25% per sector

Data:
- Historical returns: 2015-2024
- Rebalance monthly
- Transaction costs: 10 bps
"""

result = await evolve(
    problem=problem,
    domain="finance",
    max_evaluations=50,
    objectives=["sharpe_ratio", "max_drawdown"],
    constraints={
        "max_position_size": 0.05,
        "min_positions": 20,
        "max_positions": 50
    }
)

print(f"Strategy used: {result['strategy_used']}")  # 'pes'
print(f"Sharpe ratio: {result['objectives']['sharpe_ratio']}")  # 1.35
print(f"Max drawdown: {result['objectives']['max_drawdown']}")  # -15.2%
print(f"Evaluations: {result['evaluations']}")  # 30 (60% fewer)
```

### Example 2: Multi-Objective Portfolio

**Problem:** Balance return, risk, and ESG score

```python
problem = """
Optimize portfolio with ESG constraints.

Objectives:
1. Maximize return
2. Minimize risk
3. Maximize ESG score

Constraints:
- Min ESG score: 70/100
- Max fossil fuel exposure: 5%
- Min 30 positions
"""

result = await evolve(
    problem=problem,
    domain="finance",
    evolution_mode="mo",  # Multi-objective
    max_evaluations=100,
    objectives=["return", "risk", "esg_score"],
    pareto_front_size=50
)

# Get Pareto-optimal portfolios
pareto_solutions = result['pareto_front']

# Plot trade-offs
import matplotlib.pyplot as plt

returns = [s['objectives']['return'] for s in pareto_solutions]
risks = [s['objectives']['risk'] for s in pareto_solutions]
esg_scores = [s['objectives']['esg_score'] for s in pareto_solutions]

plt.scatter(risks, returns, c=esg_scores, cmap='viridis')
plt.xlabel('Risk')
plt.ylabel('Return')
plt.colorbar(label='ESG Score')
plt.title('Pareto Front: Return vs Risk vs ESG')
plt.show()
```

### Example 3: Robust Portfolio with Stress Testing

**Problem:** Design portfolio robust to market crises

```python
problem = """
Design crash-resistant portfolio.

Objectives:
- Maximize return in normal conditions
- Minimize loss during crises

Stress scenarios:
1. 2008 financial crisis
2. 2020 COVID crash
3. 2022 tech bear market
"""

result = await evolve(
    problem=problem,
    domain="finance",
    evolution_mode="adversarial",  # Robustness
    max_evaluations=100,
    adversarial_rounds=20,
    stress_scenarios=["2008_crisis", "covid_crash", "tech_bear_market"],
    objectives=["normal_return", "crisis_return"]
)

print(f"Normal return: {result['objectives']['normal_return']}")  # 12%
print(f"2008 loss: {result['crisis_performance']['2008_crisis']}")  # -18% (vs -40% market)
print(f"COVID loss: {result['crisis_performance']['covid_crash']}")  # -15% (vs -34% market)
print(f"Tech bear loss: {result['crisis_performance']['tech_bear_market']}")  # -12% (vs -30% market)
```

### Example 4: Real-World Case Study

**Problem:** Pension fund liability-driven investment

```python
problem = """
Optimize pension fund portfolio for liability matching.

Liabilities:
- $100M monthly payments for 20 years
- Duration: 12 years
- Convexity requirement: positive

Objectives:
1. Minimize funding ratio variance
2. Maximize excess return
3. Minimize cost of hedging

Constraints:
- Duration match: ±0.5 years
- Convexity: ≥ 0
- Max tracking error: 2%
- Min credit quality: BBB
"""

result = await evolve(
    problem=problem,
    domain="finance",
    evolution_mode="pes",
    max_evaluations=75,
    objectives=["funding_ratio_stability", "excess_return", "hedging_cost"],
    constraints={
        "duration_target": 12,
        "duration_tolerance": 0.5,
        "min_convexity": 0,
        "max_tracking_error": 0.02,
        "min_credit_quality": "BBB"
    }
)

# Result
print(f"Funding ratio volatility: {result['funding_ratio_volatility']}")  # 2.1%
print(f"Excess return: {result['excess_return']}")  # 1.8% over benchmark
print(f"Hedging cost: {result['hedging_cost']}")  # 0.3% annually
print(f"Duration match: {result['duration']}")  # 11.8 years (within tolerance)
```

---

## Best Practices

### 1. Use Realistic Transaction Costs

**Bad:**
```python
# Ignoring transaction costs
portfolio_value = backtest(portfolio)
return portfolio_value
```

**Good:**
```python
# Include transaction costs
portfolio_value = backtest(
    portfolio,
    transaction_cost=0.001,  # 10 bps per trade
    rebalance_cost=0.005,    # 50 bps for rebalancing
    market_impact=True
)
return portfolio_value
```

### 2. Account for Survivor Bias

**Bad:**
```python
# Using current S&P 500 constituents
data = get_current_sp500_prices()
backtest(data, start_year=2000)  # Survivor bias!
```

**Good:**
```python
# Use historical constituent data
data = get_historical_sp500_prices()  # Includes delisted stocks
backtest(data, start_year=2000)  # No survivor bias
```

### 3. Validate on Out-of-Sample Data

**Bad:**
```python
# Train and test on same data
result = evolve(problem, data=train_data)
performance = evaluate(result['best_solution'], data=train_data)
```

**Good:**
```python
# Train-test split
result = evolve(problem, data=train_data)
performance = evaluate(result['best_solution'], data=test_data)

# Walk-forward validation
for fold in folds:
    train = data[fold.train_period]
    test = data[fold.test_period]
    result = evolve(problem, data=train)
    performance = evaluate(result['best_solution'], data=test)
```

### 4. Use Regime-Aware Models

**Bad:**
```python
# Single model for all market conditions
strategy = evolve(problem, data=all_data)
```

**Good:**
```python
# Regime-specific models
bull_market_strategy = evolve(problem, data=bull_market_data)
bear_market_strategy = evolve(problem, data=bear_market_data)
sideways_strategy = evolve(problem, data=sideways_data)

# Use appropriate strategy for current regime
if current_regime == "bull":
    strategy = bull_market_strategy
elif current_regime == "bear":
    strategy = bear_market_strategy
else:
    strategy = sideways_strategy
```

### 5. Implement Proper Risk Limits

**Bad:**
```python
# No risk limits
result = evolve(problem, domain="finance")
```

**Good:**
```python
# Explicit risk limits
result = evolve(
    problem=problem,
    domain="finance",
    constraints={
        "max_var": 0.05,  # 5% daily VaR
        "max_drawdown": 0.20,  # 20% max drawdown
        "max_leverage": 2.0,
        "max_beta": 1.5
    }
)
```

### 6. Account for Market Impact

**Bad:**
```python
# Assume instant execution at mid-price
execution_price = mid_price
```

**Good:**
```python
# Model market impact
execution_price = mid_price + market_impact(
    order_size,
    avg_daily_volume,
    volatility
)
```

### 7. Use Ensemble Methods

**Bad:**
```python
# Single strategy
strategy = evolve(problem, domain="finance")
```

**Good:**
```python
# Ensemble of strategies
strategies = [
    await evolve(problem, domain="finance", random_seed=i)
    for i in range(10)
]

# Combine strategies
ensemble_portfolio = average([s['best_solution'] for s in strategies])
```

### 8. Monitor Regime Changes

**Bad:**
```python
# Static strategy
strategy = evolve(problem, domain="finance")
# Never updated
```

**Good:**
```python
# Dynamic strategy
while True:
    # Check for regime change
    if detect_regime_change():
        # Re-optimize
        strategy = await evolve(
            problem,
            domain="finance",
            data=recent_data
        )

    # Use current strategy
    execute_trades(strategy)
```

### 9. Use Appropriate Benchmarks

**Bad:**
```python
# Compare to S&P 500 for bond portfolio
my_return = 0.05
benchmark_return = 0.10  # S&P 500
print(f"Underperformed by {benchmark_return - my_return}")
```

**Good:**
```python
# Compare to bond index
my_return = 0.05
benchmark_return = 0.03  # Bloomberg Aggregate Bond Index
print(f"Outperformed by {my_return - benchmark_return}")
```

### 10. Document Assumptions

**Bad:**
```python
# No documentation
result = evolve(problem, domain="finance")
```

**Good:**
```python
# Document all assumptions
assumptions = {
    "transaction_costs": 10,  # bps
    "rebalance_frequency": "monthly",
    "data_period": "2015-2024",
    "risk_free_rate": 0.02,
    "tax_rate": 0.0,  # Pre-tax
    "inflation": 0.0  # Nominal returns
}

result = await evolve(
    problem=problem,
    domain="finance",
    metadata={"assumptions": assumptions}
)

# Save assumptions with results
save_results(result, assumptions)
```

---

## Troubleshooting

### Domain-Specific Issues

#### Issue 1: Overfitting to Historical Data

**Symptoms:**
- Great backtest performance
- Poor live performance
- High turnover

**Solutions:**
```python
# 1. Use simpler strategies
config = UnifiedEvolutionConfig(
    complexity_penalty=True,
    max_parameters=10
)

# 2. Cross-validation
result = evolve(
    problem=problem,
    domain="finance",
    validation_method="walk_forward",
    num_folds=5
)

# 3. Regularization
config = UnifiedEvolutionConfig(
    l2_regularization=0.01,
    early_stopping=True
)
```

#### Issue 2: Excessive Turnover

**Symptoms:**
- High transaction costs
- Tax inefficiency
- Slippage impact

**Solutions:**
```python
# 1. Add turnover constraint
result = evolve(
    problem=problem,
    domain="finance",
    constraints={
        "max_annual_turnover": 1.0,  # 100% annual turnover
        "min_holding_period": 30  # 30 days minimum
    }
)

# 2. Transaction cost penalty
config = UnifiedEvolutionConfig(
    transaction_cost_penalty=True,
    transaction_cost=0.001  # 10 bps
)
```

#### Issue 3: Concentration Risk

**Symptoms:**
- Few positions dominate
- Sector concentration
- Idiosyncratic risk

**Solutions:**
```python
# 1. Diversification constraint
result = evolve(
    problem=problem,
    domain="finance",
    constraints={
        "max_position_size": 0.05,  # 5% per position
        "max_sector_exposure": 0.25,  # 25% per sector
        "min_positions": 30
    }
)

# 2. Diversification bonus
config = UnifiedEvolutionConfig(
    diversification_bonus=True,
    herfindahl_target=0.05  # Low concentration
)
```

#### Issue 4: High Correlation with Benchmark

**Symptoms:**
- Low active share
- High tracking error (unexpectedly)
- No alpha

**Solutions:**
```python
# 1. Active constraint
result = evolve(
    problem=problem,
    domain="finance",
    constraints={
        "min_active_share": 0.8,  # 80% active
        "max_tracking_error": 0.02  # 2% tracking error
    }
)

# 2. Alpha objective
result = evolve(
    problem=problem,
    domain="finance",
    objectives=["alpha", "tracking_error"],
    benchmark="S&P 500"
)
```

### When to Ask for Help

#### Consult Knowledge Engine
```python
# Query for similar problems
similar_runs = await query_knowledge(
    query="Portfolio optimization with ESG constraints",
    domain="finance",
    limit=10
)

# Get recommendations
recommendations = await get_strategy_recommendations(
    problem_type="portfolio_optimization",
    constraints={"esg": True}
)
```

#### Check Documentation
- [Unified Evolution Engine Guide](../UNIFIED_EVOLUTION_ENGINE_GUIDE.md)
- [API Reference](../API_REFERENCE.md)
- [Troubleshooting Guide](../TROUBLESHOOTING.md)

#### Community Support
- GitHub Issues
- Stack Overflow tag: `openevolve`
- Discord server

---

**End of Finance Domain Guide**

For more examples, see:
- [Finance Examples](../examples/finance/)
- [Case Studies](../examples/case_studies/)
