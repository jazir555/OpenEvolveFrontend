# 3-Round Gauntlet Orchestrator

## Overview

The **3-Round Gauntlet Orchestrator** is a sophisticated progressive evaluation system that combines AI-based quality assessment, adversarial testing, and consensus verification into a unified, configurable framework.

### Key Features

- **Progressive Filtering**: Early termination saves computational resources
- **Weighted Scoring**: Configurable weights for each evaluation round
- **Domain Tuning**: Pre-configured settings for different domains
- **Comprehensive Reporting**: Detailed feedback and analysis
- **Flexible Integration**: Easy to integrate into existing workflows

### Architecture

```
Round 1: LoongFlow AI (Quick Screen)
    ↓ Score ≥ Threshold
Round 2: Red Team (Adversarial Attack)
    ↓ Score ≥ Threshold
Round 3: Gold Team (Consensus Verification)
    ↓
Final Score (Weighted Aggregate)
```

## Quick Start

### Basic Usage

```python
from openevolve.gauntlets.three_round_orchestrator import (
    ThreeRoundGauntletOrchestrator,
    create_balanced_config
)

# Create orchestrator
config = create_balanced_config()
orchestrator = ThreeRoundGauntletOrchestrator(config=config)

# Evaluate solution
result = await orchestrator.run_full_gauntlet(
    solution="def solve(): ...",
    problem="Optimize portfolio allocation",
    domain="finance"
)

print(f"Passed: {result.passed}")
print(f"Score: {result.final_score:.3f}")
print(f"Rounds: {result.rounds_completed}")
```

### Domain-Specific Configuration

```python
from openevolve.gauntlets.three_round_orchestrator import create_domain_config

# Get domain-tuned configuration
config = create_domain_config('finance')  # or 'science', 'web'

orchestrator = ThreeRoundGauntletOrchestrator(config=config)
result = await orchestrator.run_full_gauntlet(...)
```

## Evaluation Rounds

### Round 1: LoongFlow AI Evaluation
- **Weight**: 20%
- **Time**: <30 seconds
- **Purpose**: Fast quality screen
- **Threshold**: 0.5 (configurable)

### Round 2: Red Team Adversarial
- **Weight**: 30%
- **Time**: <2 minutes
- **Purpose**: Robustness testing
- **Threshold**: 0.6 (configurable)

### Round 3: Gold Team Consensus
- **Weight**: 50%
- **Time**: <5 minutes
- **Purpose**: Multi-model verification
- **Threshold**: 0.7 (configurable)

## Configuration

### Strict (High-Stakes)

```python
from openevolve.gauntlets.three_round_orchestrator import create_strict_config

config = create_strict_config()
# Thresholds: 0.7, 0.8, 0.9
# Use for: Finance, Medical, Safety-critical
```

### Lenient (Exploration)

```python
from openevolve.gauntlets.three_round_orchestrator import create_lenient_config

config = create_lenient_config()
# Thresholds: 0.3, 0.5, 0.6
# Use for: Prototyping, R&D, Learning
```

### Balanced (General)

```python
from openevolve.gauntlets.three_round_orchestrator import create_balanced_config

config = create_balanced_config()
# Thresholds: 0.5, 0.6, 0.7
# Use for: General development
```

## Files

- `three_round_orchestrator.py` - Main orchestrator implementation
- `../../tests/gauntlets/test_three_round_orchestrator.py` - Comprehensive test suite
- `../../examples/gauntlet_configs/` - Domain-specific configurations
- `../../docs/knowledge_engine/THREE_ROUND_GAUNTLET.md` - Full documentation

## Examples

### Finance Trading Strategy

```python
from examples.gauntlet_configs.finance_config import get_finance_config

config = get_finance_config('trading')
orchestrator = ThreeRoundGauntletOrchestrator(config=config)

result = await orchestrator.run_full_gauntlet(
    solution=trading_strategy_code,
    problem="Develop momentum-based trading strategy",
    domain="finance"
)
```

### Scientific Experimental Design

```python
from examples.gauntlet_configs.science_config import get_science_config

config = get_science_config('experimental_design')
orchestrator = ThreeRoundGauntletOrchestrator(config=config)

result = await orchestrator.run_full_gauntlet(
    solution=experimental_protocol,
    problem="Design randomized controlled trial",
    domain="science"
)
```

### Web Component

```python
from examples.gauntlet_configs.web_config import get_web_config

config = get_web_config('frontend')
orchestrator = ThreeRoundGauntletOrchestrator(config=config)

result = await orchestrator.run_full_gauntlet(
    solution=react_component,
    problem="Create user profile component",
    domain="web"
)
```

## Evolutionary Integration

See `../../examples/three_round_integration_example.py` for complete example of integrating gauntlet filtering into evolutionary optimization workflows.

## Testing

```bash
# Run tests
pytest tests/gauntlets/test_three_round_orchestrator.py -v

# Run with coverage
pytest tests/gauntlets/test_three_round_orchestrator.py --cov=openevolve/gauntlets --cov-report=html
```

## API Reference

### Classes

- `ThreeRoundGauntletOrchestrator` - Main orchestrator
- `ThreeRoundConfig` - Configuration dataclass
- `FullGauntletResult` - Complete evaluation result
- `Round1Result` - LoongFlow evaluation result
- `Round2Result` - Red Team evaluation result
- `Round3Result` - Gold Team evaluation result

### Factory Functions

- `create_strict_config()` - High-stakes configuration
- `create_lenient_config()` - Exploration configuration
- `create_balanced_config()` - General-purpose configuration
- `create_domain_config(domain)` - Domain-specific configuration

## Documentation

Full documentation available in:
- `../../docs/knowledge_engine/THREE_ROUND_GAUNTLET.md`

## License

Part of OpenEvolve Gauntlet System

## Author

OpenEvolve Gauntlet System

## Version

1.0.0 (2026-01-30)
