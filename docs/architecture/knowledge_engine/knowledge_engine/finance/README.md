# Financial Evolution Bridge

**Production-ready bridge between LoongFlow and OpenEvolve for crisis-surviving financial strategies.**

## What This Does

This module connects two powerful AI systems:

1. **LoongFlow** - High-level reasoning using Plan-Execute-Summarize (PES)
2. **OpenEvolve** - Low-level evolution with crisis-aware backtesting

Together, they evolve financial strategies that:
- ✅ Survive historical crises (dotcom, GFC, COVID, inflation)
- ✅ Avoid survivorship bias (includes delisted securities)
- ✅ Learn from failures (stores lessons in evolutionary memory)
- ✅ Adapt to new conditions (MAP-Elites diverse niches)

## Quick Start

### Installation

```bash
# Already part of knowledge_engine
cd knowledge_engine
pip install -e .
```

### Basic Usage

```python
import asyncio
from knowledge_engine.finance import (
    FinancialEvolutionAgent,
    EvolutionObjective,
    EvolutionBudget,
    CrisisType
)

async def main():
    # Initialize agent
    agent = FinancialEvolutionAgent()

    # Define objective
    objective = EvolutionObjective(
        universe="survivorship_free_equities_2000_2026",
        crisis_periods=[
            ("2007-09-01", "2009-03-31", CrisisType.GFC),
            ("2020-02-01", "2020-04-30", CrisisType.COVID)
        ],
        survival_constraints={
            "max_drawdown": 0.30,
            "min_equity_final": 1.0
        }
    )

    # Define budget
    budget = EvolutionBudget(
        iterations=50,
        cost_cap=100,
        strategies_per_iteration=20
    )

    # Run evolution
    result = await agent.evolve_strategies(objective, budget)

    # Results
    print(f"Best strategies found: {len(result.best_strategies)}")
    print(f"Lessons learned: {len(result.lessons_learned)}")
    print(f"Total cost: ${result.final_cost:.2f}")

    # Access best strategy
    if result.best_strategies:
        best = result.best_strategies[0]
        print(f"\nBest strategy: {best.strategy_type}")
        print(f"Parameters: {best.parameters}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Core Features

### 1. Survivorship-Free Backtesting

Tests on realistic data including:
- Bankruptcies
- Delistings (below threshold)
- M&A removals

```python
from knowledge_engine.finance import SurvivorshipBacktester, Strategy

backtester = SurvivorshipBacktester(include_delisted=True)

result = await backtester.run(
    strategy=your_strategy,
    period="2000-01-01:2026-12-31"
)

# See delisting impacts
print(f"Delistings: {len(result.delistings)}")
print(f"Impact: {sum(d.impact for d in result.delistings)}")
```

### 2. Crisis-Aware Fitness

Explicitly tests on historical crises:

| Crisis | Period | Characteristics |
|--------|--------|-----------------|
| Dotcom | 2000-2002 | High vol, negative drift |
| GFC | 2007-2009 | Extreme vol, severe losses |
| COVID | 2020 | Sudden crash, rapid recovery |
| Inflation | 2022 | Persistent bear market |

```python
from knowledge_engine.finance import CrisisAwareFitness

fitness = CrisisAwareFitness(
    crisis_periods=[
        ("2007-09-01", "2009-03-31", CrisisType.GFC),
        ("2020-02-01", "2020-04-30", CrisisType.COVID)
    ],
    memory=agent.memory
)

score = fitness.evaluate(backtest_result)

# Score includes crisis-survival bonus
print(f"Base score: {score.base_score:.2f}")
print(f"Crisis boost: {score.learned_boost:.2f}")
print(f"Total: {score.total_score:.2f}")
```

### 3. Hybrid Memory System

Combines multiple memory structures:

**Evolutionary Tree** - Lineage tracking
```python
memory.add_strategy_lineage(
    parent_id="strategy_1",
    child_id="strategy_2",
    strategy_type=StrategyType.MOMENTUM
)
```

**MAP-Elites Archive** - Diverse niches
```python
crisis_survivors = memory.get_niche_representatives(
    niche="crisis_survivors",
    n=5
)
```

**Crisis Lessons** - Learned patterns
```python
from knowledge_engine.finance import CrisisLesson, MarketConditions

lesson = CrisisLesson(
    crisis=CrisisType.GFC,
    strategy_type=StrategyType.MOMENTUM,
    successful=False,
    lesson="Momentum fails during trend reversals",
    feature_importance={"volatility": 0.9},
    boost_amount=-0.5,
    conditions_met={"volatility_threshold": 0.25}
)

memory.store_lesson(lesson)

# Retrieve when relevant
conditions = MarketConditions(
    volatility=0.30,
    resembles_crisis=CrisisType.GFC
)

relevant = memory.get_relevant_lessons(conditions)
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│             FinancialEvolutionAgent                      │
│         (LoongFlow x OpenEvolve Bridge)                  │
└──────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   ┌─────────┐   ┌──────────┐   ┌──────────┐
   │  PLAN   │   │ EXECUTE  │   │ SUMMARIZE│
   │LoongFlow│   │OpenEvolve│   │LoongFlow │
   └─────────┘   └──────────┘   └──────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    ┌────────┐   ┌────────┐   ┌────────┐
    │Backtest│   │Fitness │   │ Memory │
    │(CRSP)  │   │(Crisis)│   │(Hybrid)│
    └────────┘   └────────┘   └────────┘
```

## Configuration

### Backtester

```python
config = {
    "backtester": {
        "data_source": "CRSP_API",  # or "CRSP_SIMULATED"
        "include_delisted": True,
        "adjust_for_splits": True,
        "adjust_for_dividends": True
    }
}
```

### Fitness

```python
config = {
    "fitness": {
        "sharpe_weight": 2.0,       # Risk-adjusted returns
        "drawdown_weight": -5.0,    # Drawdown penalty
        "wealth_weight": 3.0,       # Absolute returns
        "crisis_weight": 5.0,       # Crisis survival (critical!)
        "delisting_weight": -10.0   # Delisting penalty
    }
}
```

### Memory

```python
config = {
    "memory": {
        "persistence_path": "/data/financial_memory.json"
    }
}
```

## Examples

### Example 1: Evaluate Single Strategy

```python
from knowledge_engine.finance import Strategy, StrategyType

strategy = Strategy(
    strategy_id="momentum_12m",
    strategy_type=StrategyType.MOMENTUM,
    parameters={
        "lookback": 12,
        "alpha": 0.01,
        "beta": 1.2
    },
    description="12-month momentum strategy"
)

result, score = await agent.evaluate_strategy(strategy)

print(f"Sharpe: {result.sharpe_ratio:.2f}")
print(f"Max DD: {result.max_drawdown:.1%}")
print(f"Fitness: {score.total_score:.2f}")
```

### Example 2: Analyze Lessons

```python
# Get crisis statistics
stats = agent.memory.get_crisis_statistics()

for crisis, data in stats.items():
    print(f"{crisis}:")
    print(f"  Success rate: {data['success_rate']:.1%}")
    print(f"  Lessons: {data['total_lessons']}")
```

### Example 3: Custom Evolution

```python
# Focus on crisis survival
objective = EvolutionObjective(
    universe="equities_2000_2026",
    crisis_periods=[
        ("2000-01-01", "2002-12-31", CrisisType.DOTCOM),
        ("2007-09-01", "2009-03-31", CrisisType.GFC),
        ("2020-02-01", "2020-04-30", CrisisType.COVID),
        ("2022-01-01", "2022-12-31", CrisisType.INFLATION)
    ],
    survival_constraints={
        "max_drawdown": 0.25,  # Stricter
        "min_equity_final": 1.1,  # Must profit
        "crisis_survival_required": True
    }
)

budget = EvolutionBudget(
    iterations=100,
    cost_cap=200,
    strategies_per_iteration=30
)

result = await agent.evolve_strategies(objective, budget)
```

## Testing

```bash
# Run all tests
pytest tests/finance/

# Run specific test
pytest tests/finance/test_financial_evolution.py::TestFinancialEvolutionAgent::test_evolution_loop

# Run with coverage
pytest tests/finance/ --cov=knowledge_engine.finance --cov-report=html
```

## Performance

| Operation | Time | Cost |
|-----------|------|------|
| Single backtest | 0.1-1s | $0 |
| 50 parallel backtests | ~5s | $0 |
| One evolution iteration | ~10s | $0.50-2.00 |
| Full evolution (500 iters) | 1-5 hrs | $100-500 |

## Best Practices

### ✅ DO

- Always use `include_delisted=True`
- Prioritize crisis survival in fitness
- Set realistic survival constraints
- Monitor budget during evolution
- Persist memory to disk

### ❌ DON'T

- Use only currently-traded securities
- Ignore crisis periods
- Set unrealistic return expectations
- Let evolution run without budget cap
- Skip memory persistence

## Troubleshooting

### Poor Crisis Survival

**Symptom:** Strategies fail during crises

**Solution:**
```python
config = {
    "fitness": {
        "crisis_weight": 5.0,  # Increase priority
        "drawdown_weight": -10.0  # Stricter penalty
    }
}
```

### Survivorship Bias

**Symptom:** Unrealistic backtests

**Solution:**
```python
backtester = SurvivorshipBacktester(
    include_delisted=True  # Critical!
)
```

### Overfitting

**Symptom:** Great backtest, poor live

**Solution:**
- Test on out-of-sample periods
- Use adversarial testing
- Simplify strategy parameters

## Documentation

- **Full Guide**: See `docs/finance/LOONGFLOW_OPENEVOLVE_PLATFORM.md`
- **API Reference**: See inline documentation in source files
- **Examples**: See `examples/finance/`

## Citation

If you use this in research, please cite:

```bibtex
@software{financial_evolution_2025,
  title={Financial Evolution Bridge: LoongFlow x OpenEvolve},
  author={OpenEvolve Team},
  year={2025},
  url={https://github.com/openevolve/finance}
}
```

## License

MIT License - See LICENSE file

## Contributing

Contributions welcome! See CONTRIBUTING.md

## Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: support@openevolve.org
