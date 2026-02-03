# Adaptive Trading Strategy Evolution System

## Overview

The Adaptive Trading Strategy Evolution System is a continuous 24/7 autonomous trading strategy research and evolution platform. It uses evolutionary algorithms, LLM reasoning, and causal learning to discover and refine profitable trading strategies.

### Key Features

- **Continuous Evolution**: Runs 24/7 discovering and refining strategies
- **RLM Strategy Generation**: Uses reasoning to generate high-quality strategy ideas
- **Parallel Testing**: Tests multiple strategy variants in parallel
- **Multi-Perspective Evaluation**: Judge panel evaluates from risk, return, robustness, sustainability, and implementation perspectives
- **Adversarial Testing**: Red team testing finds failure modes before live trading
- **Causal Learning**: Distinguishes correlation from causation to understand what actually drives performance
- **Knowledge Integration**: Persistent learning using knowledge engine

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TradingEvolver                           │
│                   Main Orchestrator                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │   RLM   │  │ Variant │  │  Judge  │
    │Generator│  │ Manager │  │  Panel  │
    └────┬────┘  └────┬────┘  └────┬────┘
         │            │             │
         └────────┬───┴─────────────┘
                  ▼
         ┌────────────────┐
         │  Causal Modeler│
         │  & Adversary   │
         └────────────────┘
```

## Components

### 1. TradingEvolver (Main Orchestrator)

The main orchestrator that coordinates all components in continuous evolution cycles.

**Workflow:**
1. **GENERATE Phase**: RLM generates new strategy ideas
2. **EVOLVE Phase**: Variants are tested and evolved
3. **SELECT Phase**: Judge panel selects best strategies
4. **LEARN Phase**: Causal models built from outcomes

**Usage:**
```python
from openevolve.agents.trading import TradingEvolver
from datetime import timedelta

# Initialize evolver
evolver = TradingEvolver(
    knowledge_engine=ke,  # Optional
    max_variants=10,
    max_parallel_variants=3,
    evolution_interval=timedelta(hours=1),
    backtest_days=90,
    live_trading_enabled=False
)

# Start continuous evolution
await evolver.start()

# Or run single cycle
state = await evolver.run_evolution_cycle()

# Get top strategies
top_strategies = await evolver.get_top_strategies(top_n=5)

# Stop continuous evolution
evolver.stop()
```

### 2. RLM Generator

Uses Reasoning via Language Model for strategy ideation.

**Capabilities:**
- Market analysis and opportunity identification
- Strategy hypothesis generation
- Parameter space exploration
- Strategy refinement from feedback
- Strategy combination/hybridization

**Usage:**
```python
from openevolve.agents.trading import RLMGenerator

generator = RLMGenerator(knowledge_engine=ke)

# Generate new strategies
strategies = await generator.generate_strategies(
    market_regime={"regime": "bull", "volatility": "low"},
    num_ideas=5,
    current_state=state
)

# Refine based on feedback
refined = await generator.refine_strategy(
    strategy=strategy,
    performance_feedback={"sharpe_ratio": 1.5, "weaknesses": ["exits too late"]},
    market_regime={"regime": "bull"}
)

# Combine strategies
hybrid = await generator.combine_strategies(
    strategies=[strat1, strat2, strat3],
    performance_data=[perf1, perf2, perf3]
)
```

### 3. Variant Manager

Manages multiple strategy variants in parallel.

**Capabilities:**
- Add strategies and create variants
- Paper trading on historical data
- Variant evolution (mutation, crossover)
- Performance tracking
- Variant pruning
- Hybridization

**Usage:**
```python
from openevolve.agents.trading import VariantManager

manager = VariantManager(
    max_variants=10,
    backtest_days=90
)

# Add strategy
variant = await manager.add_strategy(strategy)

# Paper trade
performance = await manager.paper_trade_variant(
    variant.variant_id,
    days=90
)

# Get top variants
top_variants = await manager.get_top_variants(top_n=5)

# Evolve new variants
children = await manager.evolve_variants(
    parent_variants=top_variants,
    num_children=5
)

# Prune underperformers
await manager.prune_variants(keep_top_n=5)

# Hybridize top performers
hybrid = await manager.hybridize_variants(top_variants[:3])
```

### 4. Judge Panel

Multi-perspective evaluation of strategies.

**Judges:**
1. **Risk Manager**: Focus on drawdown and risk controls
2. **Return Optimizer**: Focus on profitability and Sharpe ratio
3. **Robustness Expert**: Focus on market condition stability
4. **Sustainability Judge**: Focus on long-term viability
5. **Implementation Specialist**: Focus on practical execution

**Usage:**
```python
from openevolve.agents.trading import JudgePanel

panel = JudgePanel(knowledge_engine=ke)

# Evaluate from all perspectives
evaluations = await panel.evaluate_strategy(
    variant=variant,
    performance=performance,
    market_regime={"regime": "bull"}
)

# Aggregate evaluations
aggregate = panel.aggregate_evaluations(evaluations)

print(f"Overall Score: {aggregate['overall_score']:.3f}")
print(f"Consensus: {aggregate['consensus']:.3f}")
print(f"Recommendation: {aggregate['recommendation']}")
print(f"Concerns: {aggregate['concerns']}")
```

### 5. Causal Modeler

Learns causal relationships from strategy outcomes.

**Capabilities:**
- Causal discovery from outcomes
- Correlation vs causation analysis
- Mechanism identification
- Performance prediction
- Insight extraction

**Usage:**
```python
from openevolve.agents.trading import CausalModeler

modeler = CausalModeler(knowledge_engine=ke)

# Learn from outcomes
causal_model = await modeler.learn_from_outcomes(
    strategy=strategy,
    performance_history=history,
    market_context={"regime": "bull"}
)

# Extract insights
insights = await modeler.extract_insights(causal_model)

for insight in insights:
    print(f"{insight['type']}: {insight.get('insight', insight.get('condition'))}")

# Predict performance in new conditions
prediction = await modeler.predict_performance(
    strategy=strategy,
    market_conditions={"regime": "bear"}
)

print(f"Predicted performance: {prediction['predicted_performance']:.3f}")
print(f"Confidence: {prediction['confidence']:.3f}")
```

### 6. Adversary

Red team testing for strategies.

**Capabilities:**
- Failure mode discovery
- Stress testing under adverse conditions
- Assumption challenging
- Counter-strategy generation
- Robustness evaluation

**Usage:**
```python
from openevolve.agents.trading import Adversary

adversary = Adversary(knowledge_engine=ke)

# Test strategy
result = await adversary.test_strategy(
    variant=variant,
    market_conditions=["bull", "bear", "high_volatility", "crisis"]
)

print(f"Robustness: {result['robustness_score']:.3f}")
print(f"Failure Modes: {len(result['failure_modes'])}")
print(f"Recommendations: {result['recommendations']}")

# Find weaknesses
weaknesses = await adversary.find_weaknesses(variant)

# Generate counter-strategy
counter = await adversary.generate_counter_strategy(variant)
```

## Strategy Types Supported

1. **Momentum**: Exploits persistence of price trends
2. **Mean Reversion**: Exploits tendency to return to mean
3. **Statistical Arbitrage**: Exploits statistical mispricings
4. **Market Making**: Provides liquidity
5. **Trend Following**: Identifies and follows major trends
6. **Pairs Trading**: Trades correlated asset pairs
7. **Options Strategies**: Options-based strategies
8. **Machine Learning**: ML-driven strategies
9. **Hybrid**: Combinations of multiple types

## Evolution Cycle Details

### Phase 1: GENERATE

```python
# Identify market regime
market_regime = await evolver._identify_market_regime()

# Generate strategy ideas
new_strategies = await rlm_generator.generate_strategies(
    market_regime=market_regime,
    num_ideas=5
)
```

### Phase 2: EVOLVE

```python
# Add new strategies
for strategy in new_strategies:
    await variant_manager.add_strategy(strategy)

# Evolve using PES (if available) or standard evolution
if evolver.pes_agent:
    evolved = await evolver._evolve_with_pes()
else:
    evolved = await evolver._evolve_standard()

# Paper trade in parallel
tested = await evolver._paper_trade_parallel(evolved)

# Adversarial testing
robust = await evolver._adversarial_test(tested)
```

### Phase 3: SELECT

```python
# Judge panel evaluation
for variant in robust:
    performance = await variant_manager.get_performance(variant.variant_id)
    evaluations = await judge_panel.evaluate_strategy(
        variant, performance, market_regime
    )
    aggregate = judge_panel.aggregate_evaluations(evaluations)

    if aggregate["overall_score"] > 0.7:
        # Select for deployment
        await evolver._deploy_strategy(strategy)
```

### Phase 4: LEARN

```python
# Build causal models
for strategy in selected_strategies:
    history = await variant_manager.get_performance_history(strategy.strategy_id)

    causal_model = await causal_modeler.learn_from_outcomes(
        strategy, history, market_context
    )

    insights = await causal_modeler.extract_insights(causal_model)

    # Store in knowledge
    evolver.state.knowledge_artifacts.extend(insights)
```

## Data Requirements

### Market Data
- Price (OHLCV)
- Volume
- Bid/Ask spreads
- Corporate actions
- Dividends and splits

### Fundamental Data
- Financial statements
- Earnings data
- Valuation metrics
- Industry classifications

### Alternative Data
- News sentiment
- Social media activity
- Web traffic
- Satellite imagery
- Credit card data

### Strategy Performance Data
- Trade execution logs
- P&L tracking
- Slippage analysis
- Market impact

## Performance Metrics

### Return Metrics
- Total Return
- CAGR
- Monthly/Annual returns

### Risk Metrics
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Volatility
- Value at Risk (VaR)
- Conditional VaR

### Trading Metrics
- Win Rate
- Profit Factor
- Average Win/Loss
- Trade Frequency
- Holding Period

### Risk-Adjusted Metrics
- Calmar Ratio
- Information Ratio
- Alpha
- Beta
- Tracking Error

## Risk Management

### Position Sizing
- Maximum position size
- Portfolio concentration limits
- Correlation-adjusted sizing

### Stop Loss
- Hard stop loss
- Trailing stops
- Time-based stops
- Volatility-adjusted stops

### Portfolio Risk
- Maximum portfolio drawdown
- Sector exposure limits
- Market exposure limits
- Leverage limits

### Execution Risk
- Slippage limits
- Market impact controls
- Liquidity requirements

## Configuration

### Environment Variables

```bash
# Knowledge Engine
KNOWLEDGE_ENGINE_ENABLED=true
NEO4J_URI=bolt://localhost:7687
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Trading
LIVE_TRADING_ENABLED=false
MAX_POSITION_SIZE=0.2
STOP_LOSS_PCT=0.05
TAKE_PROFIT_PCT=0.15

# Evolution
MAX_VARIANTS=10
EVOLUTION_INTERVAL_HOURS=1
BACKTEST_DAYS=90

# LoongFlow (optional)
LOONGFLOW_ENABLED=true
LOONGFLOW_API_KEY=your_key
```

### Python Configuration

```python
config = {
    "evolution": {
        "max_variants": 10,
        "max_parallel_variants": 3,
        "evolution_interval": timedelta(hours=1),
        "backtest_days": 90
    },
    "risk": {
        "max_position_size": 0.2,
        "max_drawdown": 0.25,
        "stop_loss_pct": 0.05
    },
    "evaluation": {
        "min_trades": 20,
        "min_sharpe": 0.5,
        "pruning_threshold": 0.5
    }
}
```

## Usage Examples

### Example 1: Single Evolution Cycle

```python
import asyncio
from openevolve.agents.trading import TradingEvolver
from datetime import timedelta

async def main():
    evolver = TradingEvolver(
        max_variants=5,
        evolution_interval=timedelta(minutes=30),
        live_trading_enabled=False
    )

    # Run single cycle
    state = await evolver.run_evolution_cycle()

    print(f"Generation: {state.generation}")
    print(f"Best fitness: {state.best_fitness:.3f}")
    print(f"Population: {len(state.population)}")

    # Get top strategies
    top = await evolver.get_top_strategies(top_n=3)
    for i, strat in enumerate(top, 1):
        print(f"\n#{i}: {strat['strategy']['name']}")
        print(f"  Fitness: {strat['fitness']:.3f}")
        print(f"  Sharpe: {strat['performance']['sharpe_ratio']:.2f}")

asyncio.run(main())
```

### Example 2: Continuous Evolution

```python
import asyncio
from openevolve.agents.trading import TradingEvolver
from datetime import timedelta
from signal import signal, SIGINT

async def main():
    evolver = TradingEvolver(
        max_variants=10,
        evolution_interval=timedelta(hours=1),
        live_trading_enabled=True
    )

    # Handle graceful shutdown
    def shutdown():
        print("\nStopping evolution...")
        evolver.stop()

    signal(SIGINT, lambda s, f: shutdown())

    # Start continuous evolution
    await evolver.start()

asyncio.run(main())
```

### Example 3: Custom Strategy Generation

```python
from openevolve.agents.trading import RLMGenerator, VariantManager, JudgePanel

async def custom_workflow():
    # Initialize components
    generator = RLMGenerator()
    manager = VariantManager()
    panel = JudgePanel()

    # Generate strategies for current regime
    market_regime = {"regime": "bull", "volatility": "low"}

    strategies = await generator.generate_strategies(
        market_regime=market_regime,
        num_ideas=5
    )

    # Test each strategy
    for strategy in strategies:
        variant = await manager.add_strategy(strategy)
        performance = await manager.paper_trade_variant(variant.variant_id)

        # Evaluate
        evaluations = await panel.evaluate_strategy(
            variant, performance, market_regime
        )
        aggregate = panel.aggregate_evaluations(evaluations)

        print(f"{strategy.name}: {aggregate['overall_score']:.3f}")

asyncio.run(custom_workflow())
```

## Best Practices

### 1. Start with Paper Trading
Always test strategies with paper trading before live deployment.

### 2. Use Conservative Risk Limits
Set strict risk limits: max position size, stop loss, max drawdown.

### 3. Monitor Robustness
Pay attention to adversarial testing results and robustness scores.

### 4. Learn from Outcomes
Use causal modeling to understand what actually drives performance.

### 5. Diversify Strategies
Maintain multiple strategies across different types and market regimes.

### 6. Regular Review
Regularly review and prune underperforming strategies.

### 7. Keep Knowledge Persistent
Use knowledge engine to persist learnings across runs.

## Troubleshooting

### Low Strategy Quality
- Increase `backtest_days` for more robust testing
- Adjust `min_trades` threshold
- Check market regime detection
- Review strategy templates

### Slow Convergence
- Increase `max_variants` for more exploration
- Adjust mutation rates
- Enable LoongFlow PES for directed evolution
- Check parameter ranges

### Overfitting
- Reduce parameter count
- Increase validation data
- Use stricter pruning
- Monitor out-of-sample performance

### High Drawdowns
- Tighten stop loss
- Reduce position sizes
- Add diversification
- Implement regime filters

## Performance Optimization

### Parallelization
```python
# Increase parallel variants
evolver = TradingEvolver(
    max_parallel_variants=5  # Test 5 variants at once
)
```

### Selective Evolution
```python
# Only evolve top performers
top_variants = await manager.get_top_variants(top_n=3)
children = await manager.evolve_variants(
    parent_variants=top_variants,
    num_children=5
)
```

### Caching
```python
# Enable performance caching
manager = VariantManager(
    cache_performance=True,
    cache_dir="./performance_cache"
)
```

## Monitoring and Observability

### Logs
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("openevolve.agents.trading")
```

### Metrics Tracking
```python
# Get evolution summary
summary = await evolver.get_evolution_summary()

print(f"Generation: {summary['generation']}")
print(f"Best Strategy: {summary['best_strategy']}")
print(f"Best Fitness: {summary['best_fitness']:.3f}")
print(f"Diversity: {summary['diversity_metrics']}")
```

### Checkpoints
```python
# Save checkpoint
await evolver.save_checkpoint()

# Load checkpoint
await evolver.load_checkpoint("evolution_state_5.json")
```

## Advanced Usage

### Custom Judges
```python
from openevolve.agents.trading.judge_panel import BaseJudge, JudgeEvaluation

class CustomJudge(BaseJudge):
    def __init__(self):
        super().__init__("custom_judge", "custom_perspective")

    async def evaluate(self, variant, performance, market_regime):
        # Custom evaluation logic
        score = self._calculate_custom_score(variant, performance)

        return JudgeEvaluation(
            judge_id=self.judge_id,
            perspective=self.perspective,
            score=score,
            reasoning="Custom reasoning",
            concerns=[],
            recommendations=[]
        )

# Add to panel
panel = JudgePanel()
panel.judges["custom_judge"] = CustomJudge()
```

### Custom Adversarial Scenarios
```python
adversary = Adversary()

adversary.scenarios["custom_scenario"] = {
    "description": "Custom stress test",
    "market_drop": -0.30,
    "volatility_spike": 5.0,
    "custom_param": "value"
}
```

### Integration with Knowledge Engine
```python
from knowledge_engine import KnowledgeEngine

ke = KnowledgeEngine(
    neo4j_uri="bolt://localhost:7687",
    qdrant_host="localhost",
    qdrant_port=6333
)

evolver = TradingEvolver(knowledge_engine=ke)

# Learnings are automatically persisted
await evolver.run_evolution_cycle()
```

## References

- **OpenEvolve**: Evolutionary computation framework
- **LoongFlow**: Plan-Execute-Summarize evolution
- **Knowledge Engine**: Persistent learning and storage
- **Investment Committee Agent**: Related investment decision-making system

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Support

For issues and questions, please open a GitHub issue or contact the maintainers.
