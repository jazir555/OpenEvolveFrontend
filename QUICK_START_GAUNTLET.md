# Gauntlet System - Quick Start Guide

Get started with the OpenEvolve Gauntlet System in 5 minutes.

## Installation

```bash
# Install dependencies
pip install openevolve-gauntlet

# Or install from source
cd /path/to/openevolve
pip install -e .
```

## Basic Usage

### 1. Simple Gauntlet Execution

```python
from formal_gauntlet_system import GauntletSystem
from sovereign_data_models import GauntletDefinition, GauntletRoundRule

# Create gauntlet system
gauntlet_system = GauntletSystem()

# Create a simple gauntlet
gauntlet = GauntletDefinition(
    gauntlet_id="quick_test",
    name="Quick Test Gauntlet",
    rounds=[
        GauntletRoundRule(
            rule_id="automated_check",
            rule_type="automated",
            description="Automated validation",
            min_score=0.7
        )
    ]
)

# Execute gauntlet
result = gauntlet_system.execute_gauntlet(
    gauntlet=gauntlet,
    solution=solution_attempt,
    sub_problem=sub_problem
)

print(f"Passed: {result.overall_passed}")
print(f"Score: {result.final_score}")
```

### 2. Three-Round Progressive Gauntlet

```python
from core_projects.openevolve.gauntlets.three_round_orchestrator import (
    ThreeRoundGauntletOrchestrator,
    create_balanced_config
)

# Create orchestrator with balanced config
config = create_balanced_config()
orchestrator = ThreeRoundGauntletOrchestrator(config)

# Run complete gauntlet
result = await orchestrator.run_full_gauntlet(
    solution="def solve(): return 42",
    problem="Return the answer to life",
    domain="code"
)

print(f"Passed: {result.passed}")
print(f"Score: {result.final_score:.3f}")
print(f"Rounds: {result.rounds_completed}")
```

### 3. Intelligent Orchestration

```python
from glue.adapters.gauntlet_adapter.src.intelligent_orchestrator import (
    IntelligentGauntletOrchestrator,
    OptimizationObjective
)

# Create intelligent orchestrator
orchestrator = IntelligentGauntletOrchestrator(
    objective=OptimizationObjective.BALANCED
)

# Execute with AI-powered optimization
result = await orchestrator.execute_orchestration(
    solution=your_solution,
    problem=your_problem,
    domain="code"
)

print(f"Strategy: {result.to_dict()}")
```

## Configuration

### Environment Variables

```bash
# OpenEvolve API
export OPENEVOLVE_API_KEY="your-api-key"
export OPENEVOLVE_API_URL="https://api.openevolve.org"

# Gauntlet Settings
export GAUNTLET_DEFAULT_TIMEOUT=300
export GAUNTLET_MAX_PARALLEL=4
export GAUNTLET_LOG_LEVEL="INFO"

# Optional: Custom configuration
export GAUNTLET_CONFIG_PATH="/path/to/config.yaml"
```

### Configuration File (YAML)

```yaml
gauntlet:
  # Round thresholds
  round1_threshold: 0.5
  round2_threshold: 0.6
  round3_threshold: 0.7

  # Execution settings
  execution_order: sequential
  max_parallel_workers: 4
  enable_early_termination: true

  # Quality gates
  min_accuracy: 0.7
  max_failure_rate: 0.2
  max_execution_time: 30

  # Objectives
  primary_objective: balanced  # accuracy, speed, cost, balanced
```

## Common Patterns

### Pre-configured Gauntlets

```python
from formal_gauntlet_system import GauntletTemplates

# Use pre-configured templates
gauntlet = GauntletTemplates.standard_validation_gauntlet()
gauntlet = GauntletTemplates.security_gauntlet()
gauntlet = GauntletTemplates.performance_gauntlet()
gauntlet = GauntletTemplates.research_gauntlet()
```

### Domain-Specific Configuration

```python
from core_projects.openevolve.gauntlets.three_round_orchestrator import (
    create_domain_config
)

# Get domain-specific configuration
config = create_domain_config("finance")  # or "science", "engineering", "web"
orchestrator = ThreeRoundGauntletOrchestrator(config)
```

### Batch Evaluation

```python
solutions = [
    "def solve1(): return 1",
    "def solve2(): return 2",
    "def solve3(): return 3"
]

results = await orchestrator.evaluate_batch(
    solutions=solutions,
    problem="Solve the problem",
    domain="code"
)

for result in results:
    print(f"Score: {result.overall_score:.2f}")
```

## Next Steps

- **Full Documentation**: See `docs/api/gauntlet_api.md`
- **Architecture**: See `docs/architecture/gauntlet-flow.md`
- **Examples**: See `examples/gauntlets/`
- **Testing**: Run `pytest tests/gauntlets/`

## Troubleshooting

### Import Errors

```bash
# Ensure the package is installed
pip install -e /path/to/openevolve

# Check Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/openevolve"
```

### Connection Issues

```python
import os

# Verify API key is set
api_key = os.getenv("OPENEVOLVE_API_KEY")
if not api_key:
    raise ValueError("OPENEVOLVE_API_KEY not set")

# Test connection
from openevolve_client import OpenEvolveClient
client = OpenEvolveClient(api_key=api_key)
```

### Performance Issues

```python
# Enable parallel execution
config = create_balanced_config()
config.enable_parallel_execution = True

# Reduce evaluation count
config.round1_config["max_evaluations"] = 20

# Use early termination
config.enable_early_termination = True
```

## Support

- **Issues**: https://github.com/openevolve/gauntlet-system/issues
- **Discussions**: https://github.com/openevolve/gauntlet-system/discussions
- **Email**: support@openevolve.org
