# Autonomous Investment Committee Agent

A sophisticated AI-powered investment committee that performs autonomous portfolio management, continuous learning, and adaptive strategy optimization.

## Features

### Core Capabilities

- **Autonomous Weekly Reviews**: Performs complete portfolio analysis on a scheduled basis
- **Multi-Stage Reasoning**: Uses LoongFlow's Plan-Execute-Summarize framework
- **Hypothesis Testing**: Tests investment theses against historical data
- **Adversarial Analysis**: Red-teams recommendations to identify weaknesses
- **Mathematical Verification**: Validates all calculations and constraints
- **Continuous Learning**: Learns from outcomes to improve decisions

### Advanced Analysis

- **RLM Decomposition**: Breaks down complex problems using structured reasoning
- **Scenario Analysis**: Tests performance across different market conditions
- **Stress Testing**: Identifies failure modes under adverse conditions
- **Bias Detection**: Identifies cognitive biases in recommendations
- **Causal Modeling**: Builds models of what actually drives outcomes

## Quick Start

```bash
# Run the demo
python examples/investment_committee_demo.py

# Run tests
pytest tests/agents/test_investment_committee.py -v
```

## Project Structure

```
openevolve/agents/
├── investment_committee.py          # Main agent orchestrator
└── investment/
    ├── rlm_decomposer.py             # Problem decomposition
    ├── roma_tester.py                # Hypothesis testing
    ├── adversarial_tester.py         # Red team analysis
    ├── math_verifier.py              # Mathematical verification
    └── knowledge_integrator.py       # Continuous learning

tests/agents/
└── test_investment_committee.py     # Comprehensive tests

docs/agents/
└── investment_committee.md          # Full documentation

examples/
└── investment_committee_demo.py     # Demo script
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│         Investment Committee Agent                      │
│  (Orchestrates Weekly Review Cycle)                     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Review                                         │
│  - Gather portfolio state                               │
│  - Identify changes                                     │
│  - Retrieve historical context                          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 2: Analysis                                       │
│  ┌──────────────────┬──────────────────┬─────────────┐│
│  │ RLM Decomposer   │ ROMA Tester      │ Adversarial ││
│  │                  │                  │ Tester      ││
│  │ • Key factors    │ • Hypothesis     │ • Biases    ││
│  │ • Sub-problems   │   testing        │ • Challenges││
│  │ • Scenarios      │ • Backtesting    │ • Stress    ││
│  └──────────────────┴──────────────────┴─────────────┘│
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Math Verifier                                    │  │
│  │ • Portfolio math verification                    │  │
│  │ • Constraint checking                            │  │
│  │ • Risk calculation validation                    │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 3: Decision                                       │
│  - Synthesize recommendations                           │
│  - Generate specific actions                            │
│  - Assign confidence scores                             │
│  - Document reasoning                                   │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 4: Learning                                       │
│  - Track outcomes                                       │
│  - Update causal models                                 │
│  - Refine heuristics                                    │
│  - Extract lessons                                      │
└─────────────────────────────────────────────────────────┘
```

## Key Components

### 1. InvestmentCommitteeAgent

Main orchestrator that manages the weekly review cycle.

**Key Methods:**
- `weekly_review_cycle()`: Execute complete review
- `record_outcome()`: Track decision outcomes
- `get_performance_summary()`: Get performance metrics

### 2. RLMDecomposer

Decomposes complex investment problems using structured reasoning.

**Outputs:**
- Key factors with importance scores
- Testable hypotheses
- Sub-problems with dependencies
- Alternative scenarios

### 3. ROMATester

Tests investment hypotheses against historical data.

**Features:**
- Backtesting engine
- Scenario analysis
- Stress testing
- Model comparison

### 4. AdversarialTester

Challenges recommendations to find weaknesses.

**Detects:**
- Cognitive biases
- Crowded trades
- Liquidity risks
- Model overfitting

### 5. MathVerifier

Validates all mathematical calculations.

**Checks:**
- Portfolio weights sum to 100%
- Constraint satisfaction
- Risk calculations
- Optimization logic

### 6. KnowledgeIntegrator

Continuously learns from outcomes.

**Maintains:**
- Cusal factor models
- Investment heuristics
- Lessons learned database
- Scenario database

## Usage Example

```python
from openevolve.agents.investment_committee import (
    InvestmentCommitteeAgent,
    PortfolioState
)

# Create portfolio
portfolio = PortfolioState(
    holdings={"AAPL": 100, "MSFT": 50, "GOOGL": 30},
    cash=10000.0,
    total_value=50000.0
)

# Initialize agent
agent = InvestmentCommitteeAgent(
    portfolio_state=portfolio,
    market_data_provider=your_data_provider,
    database_path=Path("./investment_db"),
    risk_tolerance=0.15,
    enable_loongflow=True
)

# Run weekly review
decision = await agent.weekly_review_cycle()

print(f"Decision: {decision.decision_type}")
print(f"Confidence: {decision.confidence:.2%}")
print(f"Actions: {decision.actions}")

# Record outcome
await agent.record_outcome(
    decision.decision_id,
    "positive return of 5%",
    {"return": 0.05, "volatility": 0.12}
)
```

## Testing

The test suite covers:

- ✅ Single weekly cycle execution
- ✅ Multi-week progression
- ✅ Learning from feedback
- ✅ Accuracy of recommendations
- ✅ Robustness to market conditions
- ✅ Integration with all modules
- ✅ Performance metrics
- ✅ Error handling

Run tests:
```bash
pytest tests/agents/test_investment_committee.py -v
```

## Documentation

Full documentation available at:
- [docs/agents/investment_committee.md](docs/agents/investment_committee.md)

Covers:
- Detailed architecture
- Configuration options
- Data requirements
- Usage examples
- Risk management
- Performance metrics
- Best practices
- Troubleshooting

## Data Requirements

### Portfolio Data
- Holdings (ticker → shares)
- Cash balance
- Total value
- Last rebalance timestamp

### Market Data
- Fundamentals (P/E, earnings growth, etc.)
- Technical indicators (momentum, volatility)
- Macro economic data (interest rates, GDP)
- Market sentiment

## Performance

The agent is designed for:
- **Long-horizon operation**: Runs autonomously for months
- **Continuous improvement**: Learns from every decision
- **Robust decision-making**: Multiple validation layers
- **Transparency**: Detailed reasoning and evidence

## Risk Management

Built-in risk controls:
- Position size limits
- Volatility targeting
- Diversification requirements
- Turnover limits
- Drawdown protection

## Future Enhancements

Planned features:
- Multi-asset support (bonds, commodities, crypto)
- Options strategies
- Alternative data integration
- Ensemble methods
- Real-time monitoring
- Comprehensive backtesting

## License

Part of the OpenEvolve project.

## Support

For issues and contributions:
- GitHub: [OpenEvolve Repository]
- Documentation: [Full Docs]
