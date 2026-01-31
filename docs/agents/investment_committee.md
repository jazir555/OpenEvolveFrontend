# Autonomous Investment Committee Agent

A sophisticated long-horizon autonomous agent that performs weekly portfolio reviews, continuously learns from outcomes, adapts investment strategies over time, and uses LoongFlow for multi-stage reasoning.

## Overview

The Investment Committee Agent represents a flagship implementation of autonomous financial decision-making, combining advanced AI techniques with rigorous mathematical verification and continuous learning. It orchestrates a complete investment workflow from hypothesis generation through execution and learning.

## Architecture

### Core Components

```
InvestmentCommitteeAgent
├── RLMDecomposer          # Problem decomposition via reasoning
├── ROMATester             # Hypothesis testing against historical data
├── AdversarialTester      # Red team challenge of recommendations
├── MathVerifier           # Formal verification of decisions
└── KnowledgeIntegrator    # Continuous learning from outcomes
```

### Workflow Cycle

The agent executes a complete **weekly review cycle** consisting of four phases:

#### Phase 1: Review
- Gather portfolio state and market data
- Identify significant changes since last review
- Retrieve historical context and lessons learned
- Generate investment hypotheses

#### Phase 2: Analysis
- **RLM Decomposition**: Break down complex problems into sub-problems
- **ROMA Testing**: Test hypotheses against historical data
- **Adversarial Challenge**: Red team against recommendations
- **Mathematical Verification**: Validate all calculations

#### Phase 3: Decision
- Synthesize findings into actionable recommendations
- Document reasoning and evidence
- Create rebalancing plan with specific trades
- Assign confidence scores

#### Phase 4: Learning
- Track outcomes of previous decisions
- Update causal models of market dynamics
- Refine investment heuristics
- Improve prediction accuracy

## Installation

```bash
# Install dependencies
pip install numpy pytest pytest-asyncio

# Optional: Install LoongFlow for advanced reasoning
pip install loongflow
```

## Quick Start

```python
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from openevolve.agents.investment_committee import (
    InvestmentCommitteeAgent,
    PortfolioState
)
from openevolve.agents.investment.rlm_decomposer import RLMDecomposer

# Create portfolio
portfolio = PortfolioState(
    holdings={"AAPL": 100, "MSFT": 50, "GOOGL": 30},
    cash=10000.0,
    total_value=50000.0
)

# Create mock market data provider (replace with real provider)
class MarketDataProvider:
    async def get_current_state(self, tickers):
        # Return current market data
        return {...}

    async def get_historical_data(self, tickers, period="1y"):
        # Return historical data
        return {...}

market_data = MarketDataProvider()

# Initialize agent
agent = InvestmentCommitteeAgent(
    portfolio_state=portfolio,
    market_data_provider=market_data,
    database_path=Path("./investment_db"),
    risk_tolerance=0.15,  # 15% volatility target
    max_position_size=0.20,  # 20% max position
    rebalance_threshold=0.05,  # 5% drift triggers rebalance
    review_frequency_days=7,  # Weekly reviews
    enable_loongflow=True  # Use LoongFlow for advanced reasoning
)

# Run weekly review
async def run_review():
    decision = await agent.weekly_review_cycle()

    print(f"Decision Type: {decision.decision_type}")
    print(f"Confidence: {decision.confidence:.2%}")
    print(f"Reasoning: {decision.reasoning}")

    if decision.actions:
        print("Recommended Actions:")
        for action in decision.actions:
            print(f"  - {action}")

    # Record outcome later
    await agent.record_outcome(
        decision.decision_id,
        actual_outcome="positive return of 5%",
        performance_metrics={"return": 0.05, "volatility": 0.12}
    )

# Run
asyncio.run(run_review())
```

## Configuration Options

### Agent Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `portfolio_state` | PortfolioState | Required | Initial portfolio holdings and cash |
| `market_data_provider` | Any | Required | Provider for market data |
| `database_path` | Path | ./investment_db | Persistent storage location |
| `risk_tolerance` | float | 0.15 | Target max portfolio volatility (15%) |
| `max_position_size` | float | 0.20 | Max allocation to single position (20%) |
| `rebalance_threshold` | float | 0.05 | Allocation drift to trigger rebalance (5%) |
| `review_frequency_days` | int | 7 | Days between reviews |
| `enable_loongflow` | bool | True | Use LoongFlow for multi-stage reasoning |

### RLM Decomposer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_depth` | int | 3 | Maximum decomposition depth |
| `min_importance` | float | 0.3 | Minimum factor importance threshold |

### ROMA Tester Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_observations` | int | 30 | Minimum for statistical significance |
| `confidence_level` | float | 0.95 | Confidence interval level (95%) |

### Math Verifier Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tolerance` | float | 1e-6 | Floating point comparison tolerance |
| `critical_tolerance` | float | 1e-4 | Tolerance for critical checks |

## Data Requirements

### Portfolio Data

- **Holdings**: Dictionary of ticker → shares
- **Cash**: Available cash for trading
- **Total Value**: Current portfolio value
- **Last Rebalance**: Timestamp of last rebalancing

### Market Data

The agent requires the following market data:

#### Current State
```python
{
    "fundamentals": {
        "TICKER": {
            "pe_ratio": float,
            "earnings_growth": float,
            "revenue_growth": float,
            "debt_to_equity": float
        }
    },
    "technical": {
        "market_momentum": float,
        "volatility_regime": str,  # "low", "normal", "high"
        "trend": str  # "bull", "bear", "neutral"
    },
    "macro": {
        "interest_rate": float,
        "inflation": float,
        "gdp_growth": float,
        "unemployment": float
    },
    "sentiment": {
        "market_sentiment": str,  # "positive", "neutral", "negative"
        "news_sentiment": float  # -1.0 to 1.0
    }
}
```

#### Historical Data
```python
{
    "period": str,  # "1mo", "3mo", "6mo", "1y", etc.
    "returns": List[float],  # Historical returns
    "prices": List[float],  # Historical prices
    "volumes": List[float],  # Historical volumes
    "num_observations": int
}
```

## Usage Examples

### Example 1: Single Weekly Review

```python
# Run one review cycle
decision = await agent.weekly_review_cycle()

# Access results
print(f"Decision: {decision.decision_type}")
print(f"Confidence: {decision.confidence:.2%}")
print(f"Expected Outcome: {decision.expected_outcome}")

# View actions
for action in decision.actions:
    print(f"{action['ticker']}: {action['action']}")
    print(f"  Rationale: {action['rationale']}")
```

### Example 2: Multi-Week Operation

```python
# Run for multiple weeks
for week in range(12):
    print(f"Week {week + 1}")

    decision = await agent.weekly_review_cycle()

    # Record outcome at end of week
    if week > 0:
        actual_return = calculate_week_return()
        await agent.record_outcome(
            previous_decision_id,
            f"{'positive' if actual_return > 0 else 'negative'} return of {abs(actual_return):.2%}",
            {"return": actual_return, "volatility": calculate_volatility()}
        )

    previous_decision_id = decision.decision_id

    # Check performance
    summary = agent.get_performance_summary()
    print(f"Total Decisions: {summary['total_decisions']}")
    print(f"Accuracy: {summary['accuracy']:.2%}")
```

### Example 3: Learning from Experience

```python
# Access learned knowledge
knowledge = agent.knowledge_integrator.get_knowledge_summary()

print(f"Total Causal Factors: {knowledge['total_causal_factors']}")
print(f"Total Heuristics: {knowledge['total_heuristics']}")
print(f"Total Lessons: {knowledge['total_lessons']}")

# Get top predictive factors
top_factors = agent.knowledge_integrator.get_top_predictive_factors(n=5)
for factor in top_factors:
    print(f"{factor.name}:")
    print(f"  Predictive Power: {factor.predictive_power:.2%}")
    print(f"  Confidence: {factor.confidence:.2%}")
    print(f"  Sample Count: {factor.sample_count}")

# Get applicable heuristics for current context
context = {"market_momentum": 0.05, "volatility": "low"}
heuristics = agent.knowledge_integrator.get_applicable_heuristics(context)
for heuristic in heuristics:
    print(f"Rule: {heuristic.rule}")
    print(f"Success Rate: {heuristic.success_rate:.2%}")
```

### Example 4: Analyzing Specific Decision

```python
decision = await agent.weekly_review_cycle()

# Access full analysis
metadata = decision.metadata

review_data = metadata["review_data"]
print(f"Portfolio Value: ${review_data['portfolio_value']:,.2f}")
print(f"Market Context: {review_data['market_context']}")

analysis_results = metadata["analysis_results"]

# RLM decomposition
rlm = analysis_results["rlm_decomposition"]
print(f"\nKey Factors:")
for factor in rlm["key_factors"][:3]:
    print(f"  - {factor['name']}: {factor['importance']:.2%}")

# ROMA testing
roma = analysis_results["roma_tests"]
print(f"\nHypotheses Tested: {roma['hypotheses_tested']}")
print(f"Average Confidence: {roma['avg_confidence']:.2%}")

# Adversarial analysis
adversarial = analysis_results["adversarial_analysis"]
print(f"\nConcerns: {len(adversarial['concerns'])}")
for concern in adversarial["concerns"]:
    print(f"  - {concern}")

# Math verification
math_v = analysis_results["math_verification"]
print(f"\nMath Verification: {'PASSED' if math_v['all_passed'] else 'FAILED'}")
print(f"Checks Passed: {math_v['passed_checks']}/{math_v['total_checks']}")
```

## Performance Metrics

The agent tracks multiple performance metrics:

### Decision Metrics
- **Total Decisions**: Number of investment decisions made
- **Decisions with Outcomes**: Number with recorded outcomes
- **Average Confidence**: Mean confidence across decisions
- **Accuracy**: Percentage of correct predictions

### Knowledge Metrics
- **Causal Factors**: Number of identified causal factors
- **Heuristics**: Number of learned rules
- **Lessons Learned**: Number of extracted lessons
- **Scenarios**: Number of stored scenarios

### Risk Metrics
- **Portfolio Volatility**: Actual portfolio volatility
- **Max Drawdown**: Maximum historical drawdown
- **Sharpe Ratio**: Risk-adjusted return
- **Tracking Error**: Deviation from benchmark

## Risk Management

### Built-in Risk Controls

1. **Position Size Limits**: No position exceeds configured maximum
2. **Volatility Target**: Portfolio volatility kept within tolerance
3. **Diversification Requirements**: Minimum number of positions
4. **Turnover Limits**: Maximum portfolio turnover per period
5. **Drawdown Protection**: Monitors and limits maximum drawdown

### Adversarial Testing

The Adversarial Tester actively challenges recommendations:

- Identifies cognitive biases (confirmation bias, overconfidence, etc.)
- Generates counter-arguments
- Finds failure modes
- Stress tests under adverse conditions
- Flags crowded trades and liquidity risks

### Mathematical Verification

All decisions are mathematically verified:

- Portfolio weights sum to 100%
- No negative weights
- Risk calculations validated
- Constraint satisfaction checked
- Optimization logic verified

## Advanced Features

### LoongFlow Integration

When `enable_loongflow=True`, the agent uses LoongFlow's PES (Plan-Execute-Summarize) framework for advanced reasoning:

```python
from loongflow.framework.pes.context import EvolveChainConfig

# Create LoongFlow configuration
config = EvolveChainConfig.from_yaml("investment_config.yaml")

agent = InvestmentCommitteeAgent(
    ...,
    loongflow_config=config,
    enable_loongflow=True
)
```

### Custom Market Data Provider

Implement your own market data provider:

```python
class CustomMarketDataProvider:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = SomeAPIClient(api_key)

    async def get_current_state(self, tickers):
        data = {}
        for ticker in tickers:
            data[ticker] = {
                "pe_ratio": await self.client.get_pe(ticker),
                "earnings_growth": await self.client.get_eg(ticker),
                ...
            }
        return data

    async def get_historical_data(self, tickers, period="1y"):
        returns = []
        for ticker in tickers:
            prices = await self.client.get_historical_prices(ticker, period)
            returns.extend(calculate_returns(prices))
        return {"returns": returns, "num_observations": len(returns)}

# Use custom provider
agent = InvestmentCommitteeAgent(
    ...,
    market_data_provider=CustomMarketDataProvider(api_key="...")
)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/agents/test_investment_committee.py -v

# Run specific test class
pytest tests/agents/test_investment_committee.py::TestSingleWeeklyCycle -v

# Run with coverage
pytest tests/agents/test_investment_committee.py --cov=openevolve.agents.investment_committee
```

## Best Practices

### 1. Data Quality
- Ensure high-quality, timely market data
- Validate data before feeding to agent
- Handle missing or corrupted data gracefully
- Use survivorship-bias-free data for backtesting

### 2. Risk Management
- Start with conservative risk parameters
- Monitor agent decisions closely initially
- Implement position limits and stop losses
- Diversify across strategies and assets

### 3. Continuous Learning
- Record all decision outcomes promptly
- Review learned lessons regularly
- Update causal models with new data
- Refine heuristics based on performance

### 4. Human Oversight
- Use agent as decision support, not replacement
- Review recommendations before execution
- Override when market conditions change
- Maintain accountability for decisions

### 5. Performance Monitoring
- Track accuracy over time
- Monitor for drift in performance
- Compare against benchmarks
- Adjust parameters as needed

## Troubleshooting

### Issue: Low Confidence Scores

**Symptoms**: Agent consistently produces decisions with confidence < 0.5

**Possible Causes**:
- Low quality historical data
- Insufficient sample size
- Conflicting signals
- High market uncertainty

**Solutions**:
- Increase historical data period
- Improve data quality
- Reduce analysis complexity
- Lower confidence thresholds

### Issue: High Volatility

**Symptoms**: Portfolio volatility exceeds risk tolerance

**Possible Causes**:
- Risk parameters too loose
- Concentrated positions
- High market volatility
- Correlation breakdown

**Solutions**:
- Tighten risk tolerance
- Increase diversification
- Reduce position sizes
- Add hedges

### Issue: Poor Learning

**Symptoms**: Agent doesn't improve over time

**Possible Causes**:
- Not recording outcomes
- Outcomes delayed too long
- Insufficient data
- Changing market regimes

**Solutions**:
- Ensure outcomes are recorded
- Record outcomes promptly
- Increase sample size
- Detect regime changes

## Future Enhancements

Planned features for future versions:

1. **Multi-Asset Support**: Expand beyond equities to bonds, commodities, crypto
2. **Options Strategies**: Incorporate options and derivatives
3. **Alternative Data**: Use satellite data, web scraping, etc.
4. **Ensemble Methods**: Combine multiple agent instances
5. **Explainable AI**: Improve decision transparency
6. **Real-Time Monitoring**: Continuous monitoring and alerts
7. **Backtesting Engine**: Comprehensive backtesting framework
8. **Parameter Optimization**: Auto-tune hyperparameters

## References

1. **RLM (Reasoning via Language Model)**
   - Roumeliotis et al., "Language Models are Reasoning Agents", 2024

2. **ROMA (Review-of-Models-Agent)**
   - AI research on model review and synthesis

3. **Portfolio Theory**
   - Markowitz, "Portfolio Selection", 1952
   - Modern Portfolio Theory applications

4. **Behavioral Finance**
   - Kahneman & Tversky, Prospect Theory
   - Cognitive biases in investing

## License

This project is part of OpenEvolve and follows the same license terms.

## Support

For issues, questions, or contributions, please visit the OpenEvolve GitHub repository.
