# Financial Evolution Platform - LoongFlow x OpenEvolve

## Overview

The Financial Evolution Platform is the CORE BRIDGE connecting **LoongFlow's** high-level PES (Plan-Execute-Summarize) reasoning with **OpenEvolve's** low-level crisis-aware evolution for financial applications.

### Key Innovation

Traditional financial evolution fails because it:
1. **Suffers survivorship bias** - Only tests on currently-traded securities
2. **Ignores extinction events** - Doesn't test against historical crises
3. **Lacks causal learning** - Can't explain *why* strategies work

Our platform solves these by:
- **Survivorship-free backtesting** - Includes delisted securities
- **Crisis-aware fitness** - Explicitly tests on dotcom, GFC, COVID, inflation
- **LoongFlow reflection** - Learns lessons and stores in evolutionary memory

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Financial Evolution Agent                     │
│                  (LoongFlow x OpenEvolve Bridge)                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌───────────┐   ┌─────────────┐   ┌──────────────┐
        │   PLAN    │   │   EXECUTE   │   │   SUMMARIZE  │
        │ (LoongFlow│   │(OpenEvolve) │   │ (LoongFlow)  │
        │  Reasoning)│   │             │   │  Reflection) │
        └───────────┘   └─────────────┘   └──────────────┘
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │Backtester│ │  Fitness │ │  Memory  │
        │(Survivor-│ │(Crisis-  │ │(Hybrid   │
        │ ship-free)│ │  Aware)  │ │ Tree+MAP)│
        └──────────┘ └──────────┘ └──────────┘
```

## Core Components

### 1. FinancialEvolutionAgent

Main orchestrator combining LoongFlow PES with OpenEvolve execution.

**Flow:**

#### PLAN Phase (LoongFlow)
```python
# Analyze failures, generate hypotheses
plan = await agent._plan_generation(iteration, n_strategies)

# Output:
# - hypotheses: List of strategy hypotheses
# - parameter_ranges: Valid parameter ranges
# - estimated_cost: LLM cost estimate
```

#### EXECUTE Phase (OpenEvolve)
```python
# Generate variants, backtest on survivorship-free data
execution = await agent._execute_strategies(plan, objective)

# Output:
# - all_results: Ranked strategies with results
# - best_strategy: Top performer
# - worst_strategies: Failures for learning
```

#### SUMMARIZE Phase (LoongFlow)
```python
# Extract lessons, update memory
summary = await agent._summarize_results(execution, objective)

# Output:
# - lessons: Crisis-specific lessons learned
# - converged: Whether evolution converged
# - next_steps: Recommendations for next generation
```

### 2. SurvivorshipBacktester

Backtester that includes delisted securities to prevent look-ahead bias.

**Features:**
- Includes delisted securities (bankruptcies, below threshold)
- Adjusts for splits and dividends
- Tracks delisting events with impact
- Parallel execution support

**Usage:**
```python
backtester = SurvivorshipBacktester(
    data_source="CRSP_API",
    include_delisted=True
)

result = await backtester.run(
    strategy=momentum_strategy,
    period="2000-01-01:2026-12-31"
)

# Result includes:
# - returns: Time series of returns
# - drawdowns: Drawdown time series
# - delistings: List of delisting events
# - sharpe_ratio: Risk-adjusted return
# - max_drawdown: Maximum loss
```

### 3. CrisisAwareFitness

Fitness function that learns from historical crises.

**Crisis Periods:**
- **Dotcom (2000-2002)**: High volatility, negative drift
- **GFC (2007-2009)**: Extreme volatility, severe losses
- **COVID (2020)**: Sudden crash, rapid recovery
- **Inflation (2022)**: Persistent bear market

**Scoring:**
```python
fitness = CrisisAwareFitness(
    crisis_periods=[...],
    memory=evolution_memory
)

score = fitness.evaluate(backtest_result)

# Score components:
# - base_score: Sharpe, drawdown, wealth
# - learned_boost: LoongFlow-learned heuristics
# - total_score: Combined fitness

# Weights:
# - sharpe_ratio: +2.0
# - max_drawdown: -5.0 (penalty)
# - final_wealth: +3.0
# - crisis_survival: +5.0 (critical)
# - delisting_penalty: -10.0 (severe)
```

### 4. FinancialEvolutionMemory

Hybrid memory combining multiple structures:

#### Evolutionary Tree
Tracks lineage of strategies (parent-child relationships)

#### MAP-Elites Archive
Diverse strategies across niches:
- `high_volatility`: Thrives in volatility
- `low_volatility`: Calm market specialists
- `crisis_survivors`: Survived historical crises
- `bull_market_winners`: Bull market strategies
- `bear_market_winners`: Bear market strategies

#### Crisis-Specific Lessons
Learned lessons per crisis type:
```python
lesson = CrisisLesson(
    crisis=CrisisType.GFC,
    strategy_type=StrategyType.MOMENTUM,
    successful=False,
    lesson="Momentum failed during GFC due to trend reversals",
    feature_importance={"volatility": 0.9, "trend": 0.8},
    boost_amount=-0.5,  # Penalty for similar strategies
    conditions_met={
        "volatility_threshold": 0.25,
        "max_drawdown_threshold": 0.20
    }
)

memory.store_lesson(lesson)

# Later, retrieve relevant lessons:
conditions = MarketConditions(
    volatility=0.30,
    trend="down",
    resembles_crisis=CrisisType.GFC
)

relevant = memory.get_relevant_lessons(conditions)
```

## Usage Examples

### Basic Evolution

```python
from knowledge_engine.finance import (
    FinancialEvolutionAgent,
    EvolutionObjective,
    EvolutionBudget
)

# Initialize agent
agent = FinancialEvolutionAgent(config={
    "backtester": {
        "data_source": "CRSP_API",
        "include_delisted": True
    },
    "fitness": {
        "crisis_weight": 5.0,  # Prioritize crisis survival
        "drawdown_weight": -5.0  # Penalize large losses
    }
})

# Define objective
objective = EvolutionObjective(
    universe="survivorship_free_equities_2000_2026",
    crisis_periods=[
        ("2000-01-01", "2002-12-31", CrisisType.DOTCOM),
        ("2007-09-01", "2009-03-31", CrisisType.GFC),
        ("2020-02-01", "2020-04-30", CrisisType.COVID)
    ],
    survival_constraints={
        "max_drawdown": 0.30,  # Max 30% loss
        "min_equity_final": 1.0,  # At least break even
        "delisting_penalty": -1000
    }
)

# Define budget
budget = EvolutionBudget(
    iterations=500,
    cost_cap=500,  # $500 max LLM cost
    strategies_per_iteration=50
)

# Run evolution
result = await agent.evolve_strategies(
    objective=objective,
    budget=budget
)

# Results
print(f"Best strategies: {len(result.best_strategies)}")
print(f"Lessons learned: {len(result.lessons_learned)}")
print(f"Final cost: ${result.final_cost:.2f}")

# Access best strategy
best = result.best_strategies[0]
print(f"Best strategy: {best.strategy_type}")
print(f"Parameters: {best.parameters}")
```

### Custom Strategy Evaluation

```python
from knowledge_engine.finance import Strategy, StrategyType

# Define custom strategy
momentum_strategy = Strategy(
    strategy_id="momentum_12m",
    strategy_type=StrategyType.MOMENTUM,
    parameters={
        "lookback": 12,  # 12-month momentum
        "alpha": 0.01,   # Base return
        "beta": 1.2      # Market exposure
    },
    description="12-month momentum with market beta",
    entry_conditions=[
        "12-month return > 0",
        "Volume > 20-day average"
    ],
    exit_conditions=[
        "12-month return < 0",
        "Stop loss: -10%"
    ],
    risk_constraints={
        "max_position_size": 0.05,
        "sector_exposure": 0.20
    }
)

# Evaluate
result, score = await agent.evaluate_strategy(
    strategy=momentum_strategy,
    period="2000-01-01:2026-12-31"
)

print(f"Sharpe: {result.sharpe_ratio:.2f}")
print(f"Max DD: {result.max_drawdown:.1%}")
print(f"Fitness: {score.total_score:.2f}")
```

### Memory Analysis

```python
# Get crisis statistics
stats = agent.memory.get_crisis_statistics()

for crisis, data in stats.items():
    print(f"\n{crisis}:")
    print(f"  Total lessons: {data['total_lessons']}")
    print(f"  Success rate: {data['success_rate']:.1%}")
    print(f"  Avg boost: {data['avg_boost']:.3f}")

# Get feature importance
for feature in ["sharpe_ratio", "max_drawdown", "volatility"]:
    avg_imp = agent.memory.get_average_feature_importance(
        feature=feature,
        crisis_type=CrisisType.GFC
    )
    print(f"{feature}: {avg_imp:.2f}")

# Get niche representatives
crisis_survivors = agent.memory.get_niche_representatives(
    niche="crisis_survivors",
    n=5
)

print(f"\nTop 5 crisis survivors:")
for lesson in crisis_survivors:
    print(f"  {lesson.strategy_type}: {lesson.lesson}")
```

## Configuration

### Backtester Configuration

```python
config = {
    "backtester": {
        "data_source": "CRSP_API",  # or "CRSP_SIMULATED"
        "include_delisted": True,   # Critical for survivorship-free
        "adjust_for_splits": True,
        "adjust_for_dividends": True
    }
}
```

### Fitness Configuration

```python
config = {
    "fitness": {
        "sharpe_weight": 2.0,        # Risk-adjusted return importance
        "drawdown_weight": -5.0,     # Drawdown penalty
        "wealth_weight": 3.0,        # Absolute return importance
        "crisis_weight": 5.0,        # Crisis survival (critical!)
        "delisting_weight": -10.0,   # Delisting penalty (severe!)
        "volatility_weight": -1.0,   # Volatility penalty
        "consistency_weight": 1.0,   # Return consistency bonus

        # Crisis-specific multipliers
        "dotcom_multiplier": 1.5,
        "gfc_multiplier": 2.0,       # GFC is most severe
        "covid_multiplier": 1.8,
        "inflation_multiplier": 1.3
    }
}
```

### Memory Configuration

```python
config = {
    "memory": {
        "persistence_path": "/path/to/memory.json",  # Persist to disk
        "max_lessons_per_niche": 10,
        "max_failures_per_type": 100
    }
}
```

## API Reference

### FinancialEvolutionAgent

#### Methods

**`evolve_strategies(objective, budget) -> FinancialEvolutionResult`**
- Main evolution loop
- Returns best strategies and learned lessons

**`evaluate_strategy(strategy, period) -> Tuple[BacktestResult, FitnessScore]`**
- Evaluate single strategy
- Returns backtest result and fitness score

### SurvivorshipBacktester

#### Methods

**`run(strategy, period, include_delisted) -> BacktestResult`**
- Run single backtest
- Includes delisted securities if enabled

**`run_parallel(strategies, period, include_delisted) -> List[BacktestResult]`**
- Run multiple backtests in parallel
- Significantly faster for batch evaluations

### CrisisAwareFitness

#### Methods

**`evaluate(backtest_result, current_conditions) -> FitnessScore`**
- Calculate crisis-aware fitness
- Includes LoongFlow-learned boost

**`update_lesson_from_result(result, crisis_type, successful) -> CrisisLesson`**
- Create lesson from backtest result
- Stores in memory for future use

### FinancialEvolutionMemory

#### Methods

**`store_lesson(lesson) -> None`**
- Store learned lesson
- Updates crisis buckets and MAP-Elites archive

**`get_relevant_lessons(current_conditions) -> List[CrisisLesson]`**
- Retrieve lessons for current market conditions
- Filters by crisis type and market regime

**`get_feature_importance(feature, crisis_type, days_back) -> List[Dict]`**
- Get feature importance history
- Useful for analysis

## Best Practices

### 1. Always Use Survivorship-Free Data

```python
# BAD - survivorship bias
backtester = SurvivorshipBacktester(include_delisted=False)

# GOOD - survivorship-free
backtester = SurvivorshipBacktester(include_delisted=True)
```

### 2. Prioritize Crisis Survival

```python
# Weight crisis survival heavily
config = {
    "fitness": {
        "crisis_weight": 5.0,  # High priority
        "drawdown_weight": -5.0  # Penalize large losses
    }
}
```

### 3. Set Realistic Survival Constraints

```python
objective = EvolutionObjective(
    survival_constraints={
        "max_drawdown": 0.30,      # Acceptable for hedge funds
        "min_equity_final": 1.0,   # At least break even
        "crisis_survival_required": True  # Must survive all crises
    }
)
```

### 4. Monitor Budget

```python
# Set cost cap to prevent runaway LLM costs
budget = EvolutionBudget(
    iterations=500,
    cost_cap=500,  # $500 max
    strategies_per_iteration=50
)

# Check cost during evolution
result = await agent.evolve_strategies(objective, budget)
assert result.final_cost <= budget.cost_cap
```

### 5. Persist Memory

```python
# Save learned lessons to disk
agent = FinancialEvolutionAgent(config={
    "memory": {
        "persistence_path": "/data/financial_memory.json"
    }
})

# Lessons persist across sessions
# Future evolutions benefit from past learnings
```

## Troubleshooting

### Problem: Poor Crisis Survival

**Symptoms:** Strategies fail during crisis periods

**Solutions:**
1. Increase crisis_weight in fitness config
2. Add crisis_survival_required constraint
3. Analyze crisis lessons for patterns
4. Focus on low-volatility strategies

### Problem: Survivorship Bias

**Symptoms:** Unrealistic backtest results

**Solutions:**
1. Ensure include_delisted=True
2. Check delisting events in results
3. Use CRSP_API (not simulated data)
4. Verify delisting penalty is applied

### Problem: Slow Evolution

**Symptoms:** Evolution takes too long

**Solutions:**
1. Reduce strategies_per_iteration
2. Use run_parallel for backtesting
3. Lower iterations budget
4. Use simulated data for testing

### Problem: Overfitting

**Symptoms:** Great backtest, poor live performance

**Solutions:**
1. Reduce parameter complexity
2. Increase regularization in fitness
3. Test on out-of-sample periods
4. Use adversarial testing

## Performance Benchmarks

**Typical Performance:**
- Single backtest: 0.1-1 second (simulated data)
- Parallel backtesting (50 strategies): ~5 seconds
- Evolution (500 iterations): 1-5 hours
- Memory retrieval: <1ms for 10,000 lessons

**Cost Estimates:**
- LLM cost: ~$0.50-2.00 per generation
- Total evolution cost: $100-500 for 500 iterations

## References

- **CRSP Data**: https://www.crsp.org/
- **Survivorship Bias**: "The Surviorship Bias in Hedge Fund Performance"
- **Crisis Testing**: "Stress Testing for Hedge Funds"
- **MAP-Elites**: "Divergent Evolution for Complex Problem Solving"

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.
