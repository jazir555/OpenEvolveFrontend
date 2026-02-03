# Unified Configuration Schema

**OpenEvolve + LoongFlow PES Integration**

A comprehensive, type-safe configuration system that supports all evolutionary optimization modes from both OpenEvolve and LoongFlow PES.

---

## Overview

The Unified Configuration Schema consolidates:
- **OpenEvolve**: 51 actively used parameters (MAP-Elites, Island GA, etc.)
- **LoongFlow PES**: 20+ parameters (Plan-Execute-Summarize)
- **Total**: ~90+ unified parameters with cross-validation

### Supported Modes

1. **PES (Plan-Execute-Summarize)** - LoongFlow's reasoning-guided evolution
2. **QD (Quality-Diversity)** - OpenEvolve's MAP-Elites algorithm
3. **MO (Multi-Objective)** - Pareto optimization (NSGA-II, SPEA2, etc.)
4. **Adversarial** - Co-evolution for robustness testing
5. **Standard** - Traditional evolutionary algorithm
6. **Auto** - Automatic mode selection based on configuration

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install pydantic
```

### Basic Usage

```python
from openevolve.unified import (
    UnifiedEvolutionConfig,
    EvolutionMode,
    PESConfig,
    QDConfig,
    ConfigValidator
)

# Create PES configuration
config = UnifiedEvolutionConfig(
    evolution_mode=EvolutionMode.PES,
    pes=PESConfig(enabled=True),
    llm={
        "models": [{"name": "gpt-4", "weight": 1.0}],
        "temperature": 0.7
    },
    database={
        "num_islands": 5,
        "population_size": 100,
        "enable_memory": True
    },
    max_iterations=100
)

# Validate
validator = ConfigValidator(config)
errors, warnings = validator.validate()

if not errors:
    print("Configuration is valid!")
else:
    for error in errors:
        print(f"ERROR: {error}")
```

---

## Configuration Examples

### PES Mode (LoongFlow) - Mathematical Optimization

```python
config = UnifiedEvolutionConfig(
    evolution_mode=EvolutionMode.PES,
    domain="math",
    pes=PESConfig(
        enabled=True,
        enable_planning=True,
        max_rounds=3
    ),
    llm={
        "models": [
            LLMModelConfig(name="gpt-4", weight=1.0),
            LLMModelConfig(name="claude-3-opus", weight=1.0)
        ],
        "temperature": 0.7
    },
    database={
        "num_islands": 3,
        "population_size": 100,
        "enable_memory": True,
        "adaptive_exploration": True
    },
    evaluator={
        "early_stopping": True,
        "early_stopping_patience": 5
    }
)
```

### QD Mode (OpenEvolve) - Trading Strategy Discovery

```python
config = UnifiedEvolutionConfig(
    evolution_mode=EvolutionMode.QD,
    domain="trading",
    qd=QDConfig(
        enabled=True,
        grid_resolution=10,
        feature_dimensions=["sharpe_ratio", "max_drawdown"],
        archive_size=1000
    ),
    database={
        "population_size": 1000,
        "num_islands": 10,
        "feature_dimensions": ["sharpe_ratio", "max_drawdown"],
        "feature_bins": 10,
        "elite_selection_ratio": 0.1,
        "exploration_ratio": 0.2,
        "exploitation_ratio": 0.7
    },
    llm={
        "models": [LLMModelConfig(name="gpt-4")],
        "temperature": 0.9  # High creativity
    }
)
```

### MO Mode - Portfolio Optimization

```python
config = UnifiedEvolutionConfig(
    evolution_mode=EvolutionMode.MO,
    domain="finance",
    mo=MOConfig(
        enabled=True,
        objectives=["return", "risk", "liquidity"],
        objective_weights={"return": 0.5, "risk": 0.3, "liquidity": 0.2},
        algorithm="nsga2",
        pareto_size=100
    ),
    database={
        "population_size": 500,
        "feature_dimensions": ["return", "risk"]
    }
)
```

---

## Parameter Reference

### Core Parameters (All Modes)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iterations` | int | 10000 | Maximum iterations/generations |
| `checkpoint_interval` | int | 100 | Checkpoint save frequency |
| `random_seed` | int? | 42 | Random seed for reproducibility |
| `time_limit_seconds` | int? | None | Maximum execution time |
| `target_fitness` | float? | None | Stop when fitness reaches target |
| `domain` | DomainType | general | Problem domain |
| `language` | str? | None | Programming language |
| `max_code_length` | int | 10000 | Maximum code length |
| `diff_based_evolution` | bool | True | Use diff-based mutations |

**Count: 9 parameters**

### LLM Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | List[LLMModelConfig] | [] | Model ensemble for mutations |
| `evaluator_models` | List[LLMModelConfig] | [] | Models for evaluation |
| `planner_models` | List[LLMModelConfig] | [] | Models for PES planning |
| `summary_models` | List[LLMModelConfig] | [] | Models for PES summary |
| `temperature` | float | 0.7 | Mutation creativity [0-2] |
| `top_p` | float | 0.95 | Nucleus sampling [0-1] |
| `max_tokens` | int | 4096 | Maximum output tokens |
| `timeout` | int | 60 | Request timeout (seconds) |
| `retries` | int | 3 | Number of retries |
| `retry_delay` | int | 5 | Delay between retries |
| `random_seed` | int? | 42 | LLM sampling seed |
| `reasoning_effort` | str? | None | Reasoning effort (o1 models) |
| `plan_temperature` | float | 0.7 | Planning LLM temperature |
| `summary_temperature` | float | 0.7 | Summary LLM temperature |

**Count: 14 parameters**

### Database / Population Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `population_size` | int | 1000 | Total population size |
| `archive_size` | int | 100 | Elite archive size |
| `num_islands` | int | 5 | Number of islands |
| `elite_selection_ratio` | float | 0.1 | Elite fraction [0-1] |
| `exploration_ratio` | float | 0.2 | Exploration ratio [0-1] |
| `exploitation_ratio` | float | 0.7 | Exploitation ratio [0-1] |
| `feature_dimensions` | List[str] | ["complexity", "diversity"] | MAP-Elites features |
| `feature_bins` | int/Dict | 10 | Bins per dimension |
| `migration_interval` | int | 50 | Generations between migrations |
| `migration_rate` | float | 0.1 | Migration fraction [0-1] |
| `migration_topology` | str | ring | Migration topology |
| `diversity_metric` | str | edit_distance | Diversity metric |
| `diversity_reference_size` | int | 20 | Reference set size |
| `enable_memory` | bool | True | Enable PES memory |
| `memory_path` | str? | None | Path to memory DB |
| `exploration_rate` | float | 0.2 | Base exploration rate |
| `adaptive_exploration` | bool | True | Adaptive exploration |
| `log_prompts` | bool | True | Log prompts |
| `log_artifacts` | bool | True | Log artifacts |

**Count: 18 parameters**

### Evaluator Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | int | 300 | Evaluation timeout (seconds) |
| `max_retries` | int | 3 | Max evaluation retries |
| `cascade_evaluation` | bool | True | Multi-stage cascade |
| `cascade_thresholds` | List[float] | [0.5, 0.75, 0.9] | Cascade thresholds |
| `parallel_evaluations` | int | 4 | Parallel evaluations |
| `parallel_batch_size` | int | 10 | Batch size |
| `use_llm_feedback` | bool | False | Use LLM feedback |
| `llm_feedback_weight` | float | 0.1 | LLM feedback weight |
| `enable_gauntlets` | bool | True | Run gauntlets |
| `gauntlet_strictness` | str | standard | Gauntlet strictness |
| `gauntlet_id` | str? | None | Specific gauntlet ID |
| `enable_artifacts` | bool | True | Enable artifacts |
| `max_artifact_storage` | int | 100MB | Max artifact size |
| `early_stopping` | bool | True | Early stopping |
| `early_stopping_patience` | int | 5 | Early stopping patience |
| `early_stopping_threshold` | float | 0.01 | Improvement threshold |

**Count: 16 parameters**

### PES Configuration (LoongFlow)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | False | Enable PES mode |
| `enable_planning` | bool | True | Enable planning phase |
| `max_plans` | int | 1 | Number of plans |
| `plan_iterations` | int | 1 | Planning iterations |
| `max_rounds` | int | 3 | Max execution rounds |
| `parallel_candidates` | int | 1 | Parallel candidates |
| `enable_summary` | bool | True | Enable summary phase |
| `summary_iterations` | int | 1 | Summary iterations |
| `use_memory` | bool | True | Use memory in planning |
| `memory_top_k` | int | 5 | Top-K solutions to retrieve |

**Count: 10 parameters**

### QD Configuration (OpenEvolve MAP-Elites)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | False | Enable QD mode |
| `grid_resolution` | int | 10 | MAP-Elites resolution |
| `feature_dimensions` | List[str]? | None | Override features |
| `archive_size` | int | 1000 | Archive size |
| `use_cvt_map_elites` | bool | False | Use CVT variant |
| `cvt_samples` | int | 10000 | CVT samples |

**Count: 6 parameters**

### MO Configuration (Multi-Objective)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | False | Enable MO mode |
| `objectives` | List[str]? | None | Objective names |
| `objective_weights` | Dict? | None | Objective weights |
| `algorithm` | str | nsga2 | MO algorithm |
| `pareto_size` | int | 100 | Pareto front size |
| `use_constraint_domination` | bool | True | Constrained domination |

**Count: 6 parameters**

### Adversarial Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | False | Enable adversarial mode |
| `adversarial_rounds` | int | 20 | Number of rounds |
| `red_team_models` | List[str] | ["gpt-4", "claude-3-opus"] | Red team models |
| `blue_team_models` | List[str] | ["gpt-4", "claude-3-opus"] | Blue team models |
| `robustness_threshold` | float | 0.8 | Robustness threshold |

**Count: 5 parameters**

### Knowledge Engine Integration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_knowledge_extraction` | bool | True | Extract learning |
| `enable_strategy_learning` | bool | True | Learn strategies |
| `knowledge_engine_path` | str? | None | KE instance path |

**Count: 3 parameters**

### Output & Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | ./evolution_output | Output directory |
| `verbose` | bool | True | Verbose logging |
| `trace_enabled` | bool | False | Enable evolution trace |

**Count: 3 parameters**

---

## Total Parameter Count

**Grand Total: 102 Parameters (95 unique)**

Breakdown by class:
- UnifiedEvolutionConfig: 26 (includes 7 sub-config refs)
- LLMConfig: 14
- DatabaseConfig: 19
- EvaluatorConfig: 16
- PESConfig: 10
- QDConfig: 6
- MOConfig: 6
- AdversarialConfig: 5

Breakdown by category:
- Core (direct in UnifiedEvolutionConfig): 19
- LLM: 14
- Database: 19
- Evaluator: 16
- PES: 10
- QD: 6
- MO: 6
- Adversarial: 5
- Knowledge Engine: 3 (included in Core)
- Output: 3 (included in Core)
- **TOTAL UNIQUE: 95**

---

## Configuration Mapping

### Convert to LoongFlow PES Format

```python
from openevolve.unified import ConfigMapper

# Create unified config
config = UnifiedEvolutionConfig(evolution_mode=EvolutionMode.PES, ...)

# Convert to PES format
pes_dict = ConfigMapper.to_pes_config(config)

# Use with LoongFlow
# loongflow_agent.run(pes_dict)
```

### Convert to OpenEvolve Format

```python
# Convert to OpenEvolve format
oe_dict = ConfigMapper.to_openevolve_config(config)

# Use with OpenEvolve
# openevolve_controller.run(oe_dict)
```

### Convert from Legacy Formats

```python
# From OpenEvolve dict
oe_dict = {"max_iterations": 1000, "database": {...}}
config = ConfigMapper.from_openevolve_dict(oe_dict)

# From PES dict
pes_dict = {"task": {...}, "evolve": {...}}
config = ConfigMapper.from_pes_dict(pes_dict)
```

---

## Validation

### Basic Validation

```python
from openevolve.unified import ConfigValidator, is_valid_config

# Create validator
validator = ConfigValidator(config)

# Run validation
errors, warnings = validator.validate()

# Check results
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  [{error.severity}] {error.category}: {error.message}")

if warnings:
    print("Warnings:")
    for warning in warnings:
        print(f"  [{warning.severity}] {warning.category}: {warning.message}")

# Quick check
if is_valid_config(config):
    print("Configuration is valid!")
```

### Validation Checks

The validator performs:
1. **Mode compatibility** - Checks if mode matches enabled configs
2. **Parameter conflicts** - Detects conflicting parameters
3. **Resource constraints** - Checks for unrealistic values
4. **LLM configuration** - Validates model setup
5. **Database configuration** - Validates population parameters
6. **Evaluator configuration** - Validates evaluation setup
7. **Domain-specific** - Domain-appropriate recommendations

---

## Domain-Specific Recommendations

### Finance
- Recommended Mode: **MO** (Multi-Objective)
- Objectives: return, risk, liquidity
- Feature dimensions: risk, diversification

### Trading
- Recommended Mode: **QD** (Quality-Diversity)
- Feature dimensions: sharpe_ratio, max_drawdown, win_rate
- High num_islands for regime diversity
- Warning: cascade_evaluation may hide overfitting

### Science
- Recommended Mode: **PES** (Plan-Execute-Summarize)
- Expensive evaluations benefit from guided search
- Domain knowledge integration via planning

### Engineering
- Recommended Mode: **QD** or **PES**
- Feature dimensions: weight, strength, cost
- Long timeouts for FEA/CFD simulations (300-1800s)

### Math
- Recommended Mode: **PES**
- Single objective usually
- Low temperature for precision

### ML (Machine Learning)
- Recommended Mode: **PES**
- Expensive training costs benefit from sample efficiency
- Hyperparameter optimization

---

## Advanced Usage

### Custom LLM Ensemble

```python
config = UnifiedEvolutionConfig(
    llm={
        "models": [
            LLMModelConfig(
                name="gpt-4",
                weight=2.0,  # Higher weight
                temperature=0.8,
                max_tokens=4096
            ),
            LLMModelConfig(
                name="claude-3-opus",
                weight=1.0,
                temperature=0.6,
            )
        ]
    }
)
```

### Adaptive Exploration

```python
config = UnifiedEvolutionConfig(
    database={
        "exploration_rate": 0.2,
        "adaptive_exploration": True,  # Auto-detect local optima
        "enable_memory": True
    }
)
```

### Cascade Evaluation

```python
config = UnifiedEvolutionConfig(
    evaluator={
        "cascade_evaluation": True,
        "cascade_thresholds": [0.3, 0.6, 0.9],
        # Stage 1: Quick test (30% score required)
        # Stage 2: Medium test (60% score required)
        # Stage 3: Full test (90% score required)
    }
)
```

---

## File Structure

```
openevolve/unified/
├── __init__.py              # Main exports
├── config.py                # Configuration classes (90+ params)
├── config_mapper.py         # Format conversion
├── config_validator.py      # Validation logic
├── examples.py              # Usage examples
├── test_config.py           # Test suite
└── README.md                # This file
```

---

## Examples

See `examples.py` for comprehensive examples:
- PES mode for math optimization
- QD mode for trading strategies
- MO mode for portfolio optimization
- Adversarial mode for security testing
- Scientific experiment design
- Auto mode selection
- Domain-specific presets
- Configuration conversion

Run examples:
```bash
python openevolve/unified/examples.py
```

---

## Tests

Run test suite:
```bash
pytest openevolve/unified/test_config.py -v
```

Test coverage:
- Configuration creation
- Auto mode detection
- Validation (all modes)
- Config mapping (all formats)
- Domain-specific validation
- Parameter constraints
- Round-trip conversion
- Full workflows

---

## API Reference

### Classes

- `UnifiedEvolutionConfig` - Main configuration class
- `EvolutionMode` - Enum of evolution modes
- `DomainType` - Enum of problem domains
- `PESConfig` - PES mode configuration
- `QDConfig` - QD mode configuration
- `MOConfig` - Multi-objective configuration
- `AdversarialConfig` - Adversarial configuration
- `LLMModelConfig` - Single LLM configuration
- `LLMConfig` - LLM ensemble configuration
- `DatabaseConfig` - Database configuration
- `EvaluatorConfig` - Evaluator configuration
- `ConfigValidator` - Validation engine
- `ValidationError` - Validation error/warning

### Functions

- `ConfigMapper.to_pes_config()` - Convert to PES format
- `ConfigMapper.to_openevolve_config()` - Convert to OpenEvolve format
- `ConfigMapper.to_qd_config()` - Convert to QD format
- `ConfigMapper.to_mo_config()` - Convert to MO format
- `ConfigMapper.to_adversarial_config()` - Convert to adversarial format
- `ConfigMapper.from_openevolve_dict()` - Convert from OpenEvolve
- `ConfigMapper.from_pes_dict()` - Convert from PES
- `validate_config()` - Validate configuration
- `is_valid_config()` - Check validity

---

## Contributing

When adding new parameters:
1. Add to appropriate config class in `config.py`
2. Add validation in `config_validator.py`
3. Add mapping in `config_mapper.py`
4. Add tests in `test_config.py`
5. Update this README with parameter count

---

## License

MIT License - See OpenEvolve license file

---

## Authors

AI Architecture Team
Date: 2026-01-30

---

**Version: 1.0.0**
**Total Parameters: 90+**
**Supported Modes: 6 (PES, QD, MO, Adversarial, Standard, Auto)**
**Supported Domains: 9 (General, Finance, Trading, Science, Engineering, Pharma, Web, Math, ML)**
