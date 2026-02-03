# Adaptive Trading Strategy Evolution System

A continuous 24/7 autonomous trading strategy research and evolution platform powered by evolutionary algorithms, LLM reasoning, and causal learning.

## Quick Start

```python
import asyncio
from openevolve.agents.trading import TradingEvolver
from datetime import timedelta

async def main():
    # Initialize the evolver
    evolver = TradingEvolver(
        max_variants=10,
        evolution_interval=timedelta(hours=1),
        live_trading_enabled=False  # Start with paper trading
    )

    # Run a single evolution cycle
    state = await evolver.run_evolution_cycle()

    # Get top performing strategies
    top_strategies = await evolver.get_top_strategies(top_n=5)

    for i, strat in enumerate(top_strategies, 1):
        print(f"#{i}: {strat['strategy']['name']}")
        print(f"  Fitness: {strat['fitness']:.3f}")
        print(f"  Sharpe: {strat['performance']['sharpe_ratio']:.2f}")

asyncio.run(main())
```

## Features

- **🔄 Continuous Evolution**: 24/7 autonomous strategy discovery
- **🧠 RLM Strategy Generation**: Reasoning-based strategy ideation
- **⚡ Parallel Testing**: Test multiple variants simultaneously
- **👥 Judge Panel Evaluation**: Multi-perspective strategy assessment
- **🛡️ Adversarial Testing**: Red team testing for robustness
- **🔍 Causal Learning**: Understand what actually drives performance
- **📚 Knowledge Integration**: Persistent learning across runs

## Architecture

```
┌─────────────────────────────────────────────┐
│           TradingEvolver                    │
│        (Main Orchestrator)                  │
└───────────┬─────────────────────────────────┘
            │
    ┌───────┴───────────┐
    │                   │
    ▼                   ▼
┌─────────┐       ┌──────────────┐
│   RLM   │       │  Variant     │
│Generator│──────▶│  Manager     │
└─────────┘       └──────┬───────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    ▼                    ▼                    ▼
┌─────────┐       ┌──────────┐       ┌────────────┐
│  Judge  │       │Adversary │       │  Causal    │
│  Panel  │       │          │       │  Modeler   │
└─────────┘       └──────────┘       └────────────┘
```

## Components

### 1. TradingEvolver
Main orchestrator for continuous evolution cycles.

### 2. RLM Generator
Generates strategies using reasoning:
- Market analysis
- Strategy ideation
- Parameter exploration
- Strategy refinement

### 3. Variant Manager
Manages strategy variants:
- Paper trading
- Performance tracking
- Variant evolution
- Pruning & hybridization

### 4. Judge Panel
Multi-perspective evaluation:
- Risk Manager
- Return Optimizer
- Robustness Expert
- Sustainability Judge
- Implementation Specialist

### 5. Causal Modeler
Learns from outcomes:
- Causal discovery
- Insight extraction
- Performance prediction

### 6. Adversary
Red team testing:
- Failure mode discovery
- Stress testing
- Weakness identification

## Strategy Types

- Momentum
- Mean Reversion
- Statistical Arbitrage
- Market Making
- Trend Following
- Pairs Trading
- Options Strategies
- Machine Learning
- Hybrid (combinations)

## Evolution Cycle

```
┌─────────────┐
│  GENERATE   │  RLM generates new ideas
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   EVOLVE    │  Test and evolve variants
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   SELECT    │  Judge panel selects best
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    LEARN    │  Build causal models
└─────────────┘
```

## Documentation

See [trading_evolver.md](./docs/agents/trading_evolver.md) for comprehensive documentation including:
- Detailed component descriptions
- Usage examples
- Configuration options
- Performance metrics
- Risk management
- Troubleshooting

## Testing

```bash
# Run all tests
pytest openevolve/tests/agents/trading/test_trading_evolver.py -v

# Run specific test class
pytest openevolve/tests/agents/trading/test_trading_evolver.py::TestRLMGenerator -v

# Run with coverage
pytest openevolve/tests/agents/trading/ --cov=openevolve.agents.trading -v
```

## Requirements

- Python 3.10+
- numpy
- asyncio (standard library)
- Optional: LoongFlow (for enhanced evolution)
- Optional: Knowledge Engine (for persistent learning)

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.
