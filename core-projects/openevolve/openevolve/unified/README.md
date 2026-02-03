# Unified Configuration System

Complete unified configuration system for all evolutionary modes.

## Overview

This module provides a **single source of truth** for configuration across:
- **OpenEvolve** (272+ parameters)
- **LoongFlow PES** (~50 parameters)
- **Quality Diversity** (MAP-Elites)
- **Multi-Objective** optimization
- **Adversarial** evolution

**Total Parameters Documented: 322+**

## Quick Start

```python
from openevolve.unified import UnifiedEvolutionConfig, get_finance_config

# Use domain-specific preset
config = get_finance_config()

# Or create custom config
config = UnifiedEvolutionConfig(
    evolution_mode="openevolve",
    max_iterations=100,
    population_size=1000,
)

# Save to file
config.save_yaml("my_config.yaml")

# Load from file
config = UnifiedEvolutionConfig.from_yaml_file("my_config.yaml")

# Validate
from openevolve.unified import ConfigValidator
validator = ConfigValidator(config)
errors, warnings = validator.validate()
print(validator.get_validation_report())
```

## Evolution Modes

### 1. OpenEvolve Mode
Default mode with code evolution, diff-based improvements, and LLM-driven optimization.

```python
config = UnifiedEvolutionConfig(
    evolution_mode="openevolve",
    openevolve=OpenEvolveConfig(
        diff_based_evolution=True,
        max_code_length=10000,
        enable_meta_prompting=True,
    ),
)
```

### 2. PES Mode (Plan-Evolve-Summarize)
LoongFlow's three-phase approach with planning, execution, and summarization.

```python
config = UnifiedEvolutionConfig(
    evolution_mode="pes",
    pes=PESConfig(
        enable_planning=True,
        enable_code_execution=True,
        enable_summary=True,
    ),
)
```

### 3. Quality Diversity Mode
MAP-Elites algorithm for exploring diverse high-quality solutions.

```python
config = UnifiedEvolutionConfig(
    evolution_mode="qd",
    qd=QDConfig(
        enable_map_elites=True,
        grid_resolution=20,
        grid_dimensions=["complexity", "diversity"],
        use_novelty=True,
    ),
)
```

### 4. Multi-Objective Mode
Pareto-based multi-objective optimization (NSGA-II/III).

```python
config = UnifiedEvolutionConfig(
    evolution_mode="mo",
    mo=MOConfig(
        objectives=["accuracy", "efficiency", "cost"],
        use_pareto=True,
        selection_method="nsga2",
    ),
)
```

### 5. Adversarial Mode
Generator-discriminator co-evolution for robustness.

```python
config = UnifiedEvolutionConfig(
    evolution_mode="adversarial",
    adversarial=AdversarialConfig(
        enable_adversarial=True,
        num_adversaries=2,
        adversarial_mode="generator_discriminator",
    ),
)
```

### 6. Hybrid Mode
Combine multiple evolutionary approaches.

```python
config = UnifiedEvolutionConfig(
    evolution_mode="hybrid",
    enable_modes=["openevolve", "qd", "mo"],
)
```

## Domain-Specific Presets

### Finance
```python
config = get_finance_config()
```
- Optimized for: Risk analysis, portfolio optimization
- Characteristics: High precision, multi-objective, conservative

### Trading
```python
config = get_trading_config()
```
- Optimized for: Strategy optimization, signal generation
- Characteristics: Fast iteration, adaptive, high concurrency

### Scientific
```python
config = get_scientific_config()
```
- Optimized for: Parameter tuning, experiment design
- Characteristics: High precision, reproducible, thorough exploration

### Engineering
```python
config = get_engineering_config()
```
- Optimized for: Design optimization, performance tuning
- Characteristics: Practical, resource-constrained, balanced

### Pharmaceutical
```python
config = get_pharmaceutical_config()
```
- Optimized for: Drug discovery, molecular optimization
- Characteristics: Very high precision, safety-critical, extensive validation

### Web Design
```python
config = get_web_design_config()
```
- Optimized for: A/B testing, UX optimization
- Characteristics: Fast iteration, user feedback, visual diversity

## Configuration Mapping

Convert unified config to mode-specific formats:

```python
from openevolve.unified import ConfigMapper

# To OpenEvolve format
oe_config = ConfigMapper.to_openevolve_config(unified_config)

# To PES format
pes_config = ConfigMapper.to_pes_config(unified_config)

# To QD format
qd_config = ConfigMapper.to_qd_config(unified_config)

# From OpenEvolve format
unified_config = ConfigMapper.from_openevolve_config(oe_config_dict)
```

## Validation

Comprehensive validation with detailed error reporting:

```python
from openevolve.unified import ConfigValidator

validator = ConfigValidator(config)

# Quick check
if validator.is_valid():
    print("Config is valid!")

# Detailed validation
errors, warnings = validator.validate()

# Get formatted report
print(validator.get_validation_report())
```

### Validation Checks

1. **Mode Compatibility**: Ensures selected modes work together
2. **Parameter Conflicts**: Detects conflicting parameter values
3. **Resource Constraints**: Checks resource allocation
4. **Feature Dimensions**: Validates MAP-Elites grid configuration
5. **LLM Configuration**: Ensures models are properly configured
6. **Database Configuration**: Validates storage and memory settings
7. **Evaluator Configuration**: Checks evaluation parameters
8. **Mode-Specific Configs**: Validates mode-specific settings

## Serialization

### YAML

```python
# To YAML
yaml_str = config.to_yaml()
config.save_yaml("config.yaml")

# From YAML
config = UnifiedEvolutionConfig.from_yaml(yaml_str)
config = UnifiedEvolutionConfig.from_yaml_file("config.yaml")
```

### JSON

```python
# To JSON
json_str = config.to_json()
config.save_json("config.json")

# From JSON
config = UnifiedEvolutionConfig.from_json(json_str)
config = UnifiedEvolutionConfig.from_json_file("config.json")
```

### Dictionary

```python
# To dict
config_dict = config.to_dict()

# From dict
config = UnifiedEvolutionConfig.from_dict(config_dict)
```

## Parameter Categories

### Common Parameters (29)
Shared by all evolutionary modes:
- Core evolution: `max_iterations`, `random_seed`, `checkpoint_interval`
- Logging: `log_level`, `log_dir`, `log_to_console`, etc.
- Workspace: `workspace_path`, `task_name`, `task_description`
- Concurrency: `concurrency`, `timeout`

### LLM Configuration (26)
- Model ensemble: `models`, `evaluator_models`
- API settings: `api_base`, `api_key`, `model_provider`
- Generation: `temperature`, `top_p`, `max_tokens`, `context_length`
- Request: `timeout`, `retries`, `retry_delay`
- Reasoning: `reasoning_effort`

### Database Configuration (35)
- Storage: `storage_type`, `db_path`, `redis_url`, `output_path`
- Population: `population_size`, `elite_archive_size`, `num_islands`
- Selection: `elite_selection_ratio`, `exploration_rate`, `exploitation_ratio`
- MAP-Elites: `feature_dimensions`, `feature_bins`, `feature_scaling`
- Migration: `migration_interval`, `migration_rate`

### Evaluator Configuration (17)
- General: `timeout`, `max_retries`, `evaluate_code`
- Resources: `memory_limit_mb`, `cpu_limit`
- Strategies: `cascade_evaluation`, `parallel_evaluations`, `distributed`
- LLM Feedback: `use_llm_feedback`, `llm_feedback_weight`
- Artifacts: `enable_artifacts`, `max_artifact_storage`

### PES Configuration (22)
- Planning: `enable_planning`, `planner_type`, `planning_iterations`
- Execution: `executor_type`, `execution_mode`, `enable_code_execution`
- Summarization: `enable_summary`, `summary_type`, `summary_detail_level`
- Memory: `enable_memory`, `memory_type`, `memory_compression`
- Context: `context_window`, `context_compression_threshold`

### Quality Diversity Configuration (18)
- Grid: `grid_resolution`, `grid_dimensions`, `adaptive_grid`
- Archive: `archive_type`, `archive_elitism`, `use_novelty`
- Features: `feature_extraction_method`, `feature_normalization`
- QD: `cvt_samples`, `use_niching`, `niche_radius`

### Multi-Objective Configuration (15)
- Objectives: `objectives`, `objective_weights`, `optimization_direction`
- Pareto: `pareto_archive_size`, `pareto_pruning_method`, `use_hypervolume`
- Selection: `selection_method`, `tournament_size`, `crossover_rate`
- Scalarization: `use_scalarization`, `scalarization_method`, `reference_point`

### Adversarial Configuration (12)
- Setup: `enable_adversarial`, `num_adversaries`, `adversarial_mode`
- Generator/Discriminator: `generator_objective`, `discriminator_objective`
- Coevolution: `use_coevolution`, `coevolution_frequency`, `fitness_sharing`

### OpenEvolve Configuration (48)
- Code Evolution: `diff_based_evolution`, `max_code_length`, `language`
- Prompts: `system_message`, `num_top_programs`, `num_diverse_programs`
- Artifacts: `max_artifact_bytes`, `artifact_security_filter`
- Early Stopping: `early_stopping_patience`, `convergence_threshold`
- Meta-Prompting: `use_meta_prompting`, `meta_prompt_weight`
- Evolution Trace: `evolution_trace_enabled`, `evolution_trace_format`
- Advanced: `use_embedding`, `enable_novelty_search`, `use_crossover`

## Best Practices

### 1. Start with Domain Presets
Always begin with a domain-specific preset, then customize:

```python
config = get_finance_config()
config.common.max_iterations = 1000  # Customize
```

### 2. Validate Before Running
Always validate configuration before evolution:

```python
validator = ConfigValidator(config)
if not validator.is_valid():
    print(validator.get_validation_report())
    # Fix errors before running
```

### 3. Use Appropriate Mode
Choose the right mode for your problem:
- **Single objective**: `openevolve`
- **Multiple objectives**: `mo`
- **Explore diverse solutions**: `qd`
- **Complex planning needed**: `pes`
- **Robustness needed**: `adversarial`
- **Mixed requirements**: `hybrid`

### 4. Tune Population Size
Larger populations = more diversity but slower:

```python
# Fast iteration, less diversity
database=DatabaseConfig(population_size=500)

# Slow iteration, more diversity
database=DatabaseConfig(population_size=2000)
```

### 5. Balance Exploration/Exploitation
Adjust ratios based on problem:

```python
# Conservative (fine-tuning)
DatabaseConfig(
    exploration_rate=0.1,
    exploitation_ratio=0.8,
)

# Aggressive (exploration)
DatabaseConfig(
    exploration_rate=0.4,
    exploitation_ratio=0.5,
)
```

### 6. Set Appropriate Convergence
Early stopping prevents wasted computation:

```python
OpenEvolveConfig(
    early_stopping_patience=50,  # Stop if no improvement for 50 iterations
    convergence_threshold=0.001,  # Minimum improvement to reset patience
)
```

## Examples

### Example 1: Simple Evolution
```python
from openevolve.unified import UnifiedEvolutionConfig

config = UnifiedEvolutionConfig(
    evolution_mode="openevolve",
    common=CommonConfig(
        max_iterations=100,
        task_name="simple_optimization",
    ),
    database=DatabaseConfig(
        population_size=500,
        num_islands=3,
    ),
)
```

### Example 2: Multi-Objective with QD
```python
config = UnifiedEvolutionConfig(
    evolution_mode="hybrid",
    enable_modes=["mo", "qd"],
    mo=MOConfig(
        objectives=["accuracy", "speed", "memory"],
        use_pareto=True,
    ),
    qd=QDConfig(
        enable_map_elites=True,
        grid_dimensions=["accuracy", "speed"],
    ),
)
```

### Example 3: PES with Custom Planning
```python
config = UnifiedEvolutionConfig(
    evolution_mode="pes",
    pes=PESConfig(
        enable_planning=True,
        planner_type="evolve_planner",
        planning_iterations=3,
        use_refinement=True,
        max_refinement_iterations=5,
    ),
)
```

### Example 4: Adversarial Training
```python
config = UnifiedEvolutionConfig(
    evolution_mode="adversarial",
    adversarial=AdversarialConfig(
        enable_adversarial=True,
        num_adversaries=3,
        adversarial_mode="generator_discriminator",
        use_coevolution=True,
        use_arms_race=True,
    ),
)
```

## API Reference

See individual module documentation for complete API reference:

- `config.py`: All configuration classes
- `config_mapper.py`: Conversion utilities
- `config_validator.py`: Validation logic
- `defaults.py`: Domain presets

## Contributing

When adding new parameters:

1. Add to appropriate config class in `config.py`
2. Add mapping logic in `config_mapper.py`
3. Add validation in `config_validator.py`
4. Update this README
5. Update parameter count in docs

## License

Same as OpenEvolve project.
