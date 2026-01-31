# Trading Domain Guide

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

- **Strategy Development** - Entry/exit rules, signal optimization
- **Parameter Tuning** - Lookback periods, thresholds, position sizing
- **Portfolio Construction** - Multi-strategy portfolios, correlation management
- **Risk Management** - Stop losses, position limits, drawdown controls
- **Execution Optimization** - Order routing, slippage minimization

### Unique Challenges

1. **Regime Changes** - Markets shift between bull/bear/ranging
2. **Overfitting** - Easy to overfit to historical data
3. **Look-Ahead Bias** - Future data leakage in backtests
4. **Transaction Costs** - Can eliminate profits
5. **Market Impact** - Large orders move prices

### Why Evolutionary Optimization?

**Traditional Methods:**
- Technical analysis (subjective)
- Machine learning (static, overfits)
- Linear regression (too simple)

**Evolutionary Advantages:**
- Adapts to regime changes
- Finds non-linear patterns
- Multi-objective optimization
- Robust through adversarial testing

---

## Recommended Approach

### Best System: OpenEvolve

**Why?**
- Need for robustness
- Adversarial co-evolution finds weaknesses
- Quality diversity for strategy variants
- Multi-objective for Sharpe vs drawdown

### Best Mode: Adversarial Co-evolution

**Why Adversarial?**
- Evolve strategies AND market scenarios
- Red team attacks find weaknesses
- More robust to regime changes
- Survives stress testing

### Hybrid: Adversarial + Multi-Objective

```python
# Phase 1: Adversarial for robustness
result1 = await evolve(
    problem="Develop trading strategy",
    domain="trading",
    evolution_mode="adversarial",
    adversarial_rounds=20
)

# Phase 2: Multi-objective for Sharpe/drawdown trade-off
result2 = await evolve(
    problem="Optimize strategy parameters",
    domain="trading",
    evolution_mode="mo",
    objectives=["sharpe_ratio", "max_drawdown", "win_rate"],
    initial_solutions=result1['solutions']
)
```

---

## Configuration

### Default Configuration

```python
from openevolve.unified import UnifiedEvolutionConfig

trading_config = UnifiedEvolutionConfig(
    # Core
    domain="trading",
    evolution_mode="adversarial",

    # Evaluation
    max_evaluations=100,
    adversarial_rounds=20,

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
    market_scenarios=["bull", "bear", "ranging", "crash"]
)
```

### Strategy-Specific Configurations

#### Momentum Strategy

```python
momentum_config = UnifiedEvolutionConfig(
    domain="trading",
    evolution_mode="adversarial",

    # Strategy parameters
    strategy_type="momentum",
    lookback_range=[10, 50],
    entry_threshold_range=[0.5, 2.0],
    exit_threshold_range=[0.3, 1.5],

    # Objectives
    objectives=["sharpe_ratio", "max_drawdown"],

    # Constraints
    constraints={
        "max_positions": 5,
        "position_sizing": "volatility_weighted",
        "min_holding_period": 5  # days
    }
)
```

#### Mean Reversion Strategy

```python
mean_reversion_config = UnifiedEvolutionConfig(
    domain="trading",
    evolution_mode="adversarial",

    # Strategy parameters
    strategy_type="mean_reversion",
    lookback_range=[5, 20],
    z_score_threshold_range=[1.5, 3.0],
    bollinger_period_range=[10, 30],

    # Objectives
    objectives=["sharpe_ratio", "win_rate", "profit_factor"],

    # Constraints
    constraints={
        "max_positions": 10,
        "position_sizing": "fixed_fraction",
        "max_position_size": 0.02
    }
)
```

---

## Evaluation Metrics

### Performance Metrics

```python
def calculate_trading_metrics(equity_curve):
    returns = equity_curve.pct_change().dropna()

    return {
        # Return metrics
        "total_return": (equity_curve[-1] / equity_curve[0]) - 1,
        "annual_return": returns.mean() * 252,

        # Risk metrics
        "volatility": returns.std() * sqrt(252),
        "max_drawdown": max_drawdown(equity_curve),
        "avg_drawdown": avg_drawdown(equity_curve),

        # Risk-adjusted metrics
        "sharpe_ratio": (returns.mean() * 252 - risk_free_rate) / (returns.std() * sqrt(252)),
        "sortino_ratio": (returns.mean() * 252 - risk_free_rate) / downside_std(returns) * sqrt(252),
        "calmar_ratio": annual_return / abs(max_drawdown),

        # Trade metrics
        "win_rate": num_wins / num_trades,
        "profit_factor": gross_profit / gross_loss,
        "avg_win": avg_winning_trade,
        "avg_loss": avg_losing_trade,
        "expectancy": (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    }
```

### What Good Targets Look Like

#### Conservative Strategy
```python
{
    "sharpe_ratio": 1.0,
    "max_drawdown": -0.10,
    "win_rate": 0.55,
    "profit_factor": 2.0
}
```

#### Aggressive Strategy
```python
{
    "sharpe_ratio": 1.5,
    "max_drawdown": -0.25,
    "win_rate": 0.45,
    "profit_factor": 2.5
}
```

---

## Examples

### Example 1: Momentum Strategy

```python
from openevolve.unified import evolve

problem = """
Develop momentum strategy for crypto trading.

Entry: Buy when price > SMA(20) by 2%
Exit: Sell when price < SMA(20) by 1%
Position: Volatility-weighted

Objectives:
- Maximize Sharpe ratio
- Minimize max drawdown
"""

result = await evolve(
    problem=problem,
    domain="trading",
    evolution_mode="adversarial",
    max_evaluations=100,
    objectives=["sharpe_ratio", "max_drawdown"],
    data=bitcoin_hourly_prices
)

print(f"Sharpe: {result['sharpe_ratio']}")  # 1.85
print(f"Max DD: {result['max_drawdown']}")  # -18.3%
print(f"Win rate: {result['win_rate']}")  # 52%
```

### Example 2: Multi-Strategy Portfolio

```python
problem = """
Combine multiple trading strategies.

Strategies:
1. Momentum (crypto)
2. Mean reversion (forex)
3. Breakout (stocks)

Objectives:
- Maximize portfolio Sharpe
- Minimize portfolio drawdown
- Minimize strategy correlation
"""

result = await evolve(
    problem=problem,
    domain="trading",
    evolution_mode="mo",
    max_evaluations=150,
    objectives=["portfolio_sharpe", "portfolio_drawdown", "correlation"],
    strategies=["momentum", "mean_reversion", "breakout"]
)

# Get Pareto-optimal allocations
for solution in result['pareto_front']:
    print(f"Momentum: {solution['momentum_weight']:.2f}")
    print(f"Mean reversion: {solution['mean_reversion_weight']:.2f}")
    print(f"Breakout: {solution['breakout_weight']:.2f}")
    print(f"Sharpe: {solution['portfolio_sharpe']:.2f}")
    print("---")
```

---

## Best Practices

### 1. Avoid Look-Ahead Bias

**Bad:**
```python
# Using future data in calculation
signal = price > price.mean()  # Includes future data!
```

**Good:**
```python
# Rolling mean (only past data)
signal = price > price.rolling(20).mean()
```

### 2. Include Realistic Transaction Costs

**Bad:**
```python
# No transaction costs
profit = final_price - entry_price
```

**Good:**
```python
# Include costs
profit = (final_price - entry_price) - (entry_price * 0.001)  # 10 bps
profit -= slippage_cost
```

### 3. Use Walk-Forward Validation

**Bad:**
```python
# Single train-test split
train = data[:2020]
test = data[2020:]
```

**Good:**
```python
# Walk-forward
for fold in folds:
    train = data[fold.train_start:fold.train_end]
    test = data[fold.test_start:fold.test_end]

    strategy = evolve(problem, data=train)
    performance = test_strategy(strategy, data=test)
```

### 4. Test on Out-of-Sample Data

**Bad:**
```python
# Report in-sample performance
performance = backtest(strategy, data=all_data)
```

**Good:**
```python
# Hold-out test set
train = data[:2023]
test = data[2023:]

strategy = evolve(problem, data=train)
performance = backtest(strategy, data=test)
```

### 5. Account for Market Regimes

**Bad:**
```python
# Single strategy for all conditions
strategy = evolve(problem, data=all_data)
```

**Good:**
```python
# Regime-specific strategies
bull_strategy = evolve(problem, data=bull_periods)
bear_strategy = evolve(problem, data=bear_periods)
sideways_strategy = evolve(problem, data=sideways_periods)
```

### 6. Use Risk Management

**Bad:**
```python
# No stop loss
if entry_signal:
    buy()
```

**Good:**
```python
# With stop loss
if entry_signal:
    buy()
    set_stop_loss(-0.05)  # 5% stop loss
```

### 7. Monitor Strategy Decay

**Bad:**
```python
# Never retrain
strategy = evolve(problem, data=data_2020)
# Use forever
```

**Good:**
```python
# Periodic retraining
if detect_decay():
    strategy = evolve(problem, data=recent_data)
```

---

## Troubleshooting

### Issue 1: Overfitting

**Symptoms:** Great backtest, poor live performance

**Solutions:**
```python
# 1. Simpler strategies
config = UnifiedEvolutionConfig(
    complexity_penalty=True,
    max_parameters=5
)

# 2. Cross-validation
result = evolve(
    problem=problem,
    domain="trading",
    validation_method="walk_forward",
    num_folds=5
)
```

### Issue 2: High Correlation to Benchmark

**Symptoms:** Low alpha, high beta

**Solutions:**
```python
# Add market neutral constraint
result = evolve(
    problem=problem,
    domain="trading",
    constraints={
        "market_neutral": True,
        "max_beta": 0.3,
        "min_active_share": 0.8
    }
)
```

---

**End of Trading Domain Guide**
