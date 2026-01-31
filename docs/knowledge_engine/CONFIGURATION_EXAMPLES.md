# Configuration Examples

**Version:** 1.0
**Date:** January 30, 2026
**Status:** Production Ready

20+ working configuration examples for common use cases.

---

## Table of Contents

1. [Basic Examples](#1-basic-examples-1-5)
2. [Intermediate Examples](#2-intermediate-examples-6-10)
3. [Advanced Examples](#3-advanced-examples-11-15)
4. [Domain-Specific Examples](#4-domain-specific-examples-16-20)

---

## 1. Basic Examples

### Example 1: Minimal Configuration (5 lines)

**File:** `evolve.config.yaml`

```yaml
# Minimal configuration - just the essentials
evolution_mode: auto
max_iterations: 100
domain: finance
enable_knowledge_engine: true
log_level: INFO
```

**Use Case:** Getting started quickly

**Python Usage:**
```python
from openevolve import evolve

result = await evolve(
    problem="Optimize portfolio allocation",
    domain="finance"  # Config loaded automatically
)
```

---

### Example 2: Basic Config File (YAML)

**File:** `evolve.config.yaml`

```yaml
# Basic configuration in YAML
# OpenEvolve Configuration File

# Evolutionary parameters
evolution_mode: auto
max_iterations: 100
population_size: 100
convergence_threshold: 0.001

# Problem definition
domain: finance
objectives:
  - return
  - risk

# Resources
max_evaluations: 100
parallel_evaluations: 4

# Logging
log_level: INFO
```

**Use Case:** Standard evolution run

---

### Example 3: Basic Config File (JSON)

**File:** `evolve.config.json`

```json
{
  "evolution_mode": "auto",
  "max_iterations": 100,
  "population_size": 100,
  "convergence_threshold": 0.001,
  "domain": "finance",
  "objectives": ["return", "risk"],
  "max_evaluations": 100,
  "parallel_evaluations": 4,
  "log_level": "INFO"
}
```

**Use Case:** When JSON is preferred (automated generation)

---

### Example 4: Basic Config File (TOML)

**File:** `evolve.config.toml`

```toml
evolution_mode = "auto"
max_iterations = 100
population_size = 100
convergence_threshold = 0.001

domain = "finance"
objectives = ["return", "risk"]

max_evaluations = 100
parallel_evaluations = 4

log_level = "INFO"
```

**Use Case:** When TOML is preferred (more explicit than YAML)

---

### Example 5: Environment Variables Only

**File:** `.env` (don't commit this)

```bash
# Environment variables configuration
EVOLVE_EVOLUTION_MODE=auto
EVOLVE_MAX_ITERATIONS=100
EVOLVE_POPULATION_SIZE=100
EVOLVE_DOMAIN=finance
EVOLVE_OBJECTIVES=return,risk
EVOLVE_MAX_EVALUATIONS=100
EVOLVE_PARALLEL_EVALUATIONS=4
EVOLVE_LOG_LEVEL=INFO

# API configuration
EVOLVE_API_KEY=sk-...
EVOLVE_MODEL=gpt-4
```

**Use Case:** Containerized deployments, secrets

**Load:**
```bash
source .env  # or export EVOLVE_... individually
```

---

## 2. Intermediate Examples

### Example 6: Config File + Environment Variables

**Config File:** `evolve.config.yaml`

```yaml
# Base configuration
evolution_mode: auto
max_iterations: 100
domain: finance
objectives:
  - return
  - risk

# Defaults (can be overridden)
max_evaluations: 100
parallel_evaluations: 4
```

**Environment Variables:**
```bash
export EVOLVE_MAX_EVALUATIONS=200  # Override config
export EVOLVE_PARALLEL_EVALUATIONS=8  # Override config
export EVOLVE_API_KEY="sk-..."  # Add secret
```

**Result:** Merged configuration (env vars override config file)

---

### Example 7: Profile-Based Config

**Profile:** `~/.evolve/profiles/finance_fast.yaml`

```yaml
# Finance fast profile
inherit: dev

overrides:
  domain: finance
  evolution_mode: pes
  max_evaluations: 30
  enable_planning: true
  enable_memory: true
```

**Use Profile:**
```bash
# Method 1: CLI
evolve --profile finance_fast problem="..."

# Method 2: Config file
# evolve.config.yaml
profile: finance_fast
```

---

### Example 8: Preset-Based Config

**Apply Preset:**
```bash
evolve preset apply finance -o evolve.config.yaml
```

**Result:** `evolve.config.yaml`

```yaml
# Generated from finance preset
evolution_mode: pes
max_evaluations: 50
enable_planning: true
enable_memory: true
domain: finance
objectives:
  - return
  - risk
  - sharpe_ratio
constraints:
  max_position_size: 0.1
  sector_diversification: true
enable_gauntlet: true
gauntlet_rounds:
  - loongflow
  - gold_team
enable_knowledge_engine: true
```

---

### Example 9: Runtime Config Update

**Python:**
```python
import asyncio
from openevolve import EvolutionEngine

async def main():
    engine = EvolutionEngine()

    # Start with initial config
    config = {
        'max_iterations': 100,
        'population_size': 100
    }

    # Start evolution in background
    task = asyncio.create_task(
        engine.evolve(problem="...", config=config)
    )

    # Wait a bit, then update config
    await asyncio.sleep(1)

    # Runtime update
    await engine.update_config(max_iterations=200)

    # Get result
    result = await task
    return result

asyncio.run(main())
```

---

### Example 10: Hot-Reload Config

**Python:**
```python
from openevolve import EvolutionEngine

# Enable hot-reload
engine = EvolutionEngine(
    hot_reload=True,
    config_file="evolve.config.yaml"
)

# Start evolution
task = asyncio.create_task(engine.evolve(problem="..."))

# Edit evolve.config.yaml while running
# Changes are automatically reloaded

result = await task
```

**Config File Change:**
```yaml
# Original
max_iterations: 100

# Edit while running
max_iterations: 200  # Automatically reloaded
```

---

## 3. Advanced Examples

### Example 11: Domain-Specific (Finance)

**File:** `evolve.finance.config.yaml`

```yaml
# Finance domain configuration
evolution_mode: pes  # Expensive evaluations
max_evaluations: 50

# Domain
domain: finance
objectives:
  - return
  - risk
  - sharpe_ratio

constraints:
  max_position_size: 0.1
  sector_diversification: true
  max_drawdown: 0.2
  min_liquidity: 0.5

# PES settings
enable_planning: true
enable_memory: true
early_stopping: true
early_stop_threshold: 0.9

# Gauntlet
enable_gauntlet: true
gauntlet_rounds:
  - loongflow  # Quick validation
  - gold_team  # Expert validation

# Knowledge engine
enable_knowledge_engine: true
extract_knowledge: true
use_past_solutions: true

# Resources
parallel_evaluations: 4
evaluation_timeout: 300
```

---

### Example 12: Domain-Specific (Trading)

**File:** `evolve.trading.config.yaml`

```yaml
# Trading domain configuration
evolution_mode: adversarial  # Robustness
max_evaluations: 100

# Domain
domain: trading
objectives:
  - sharpe_ratio
  - max_drawdown
  - win_rate
  - profit_factor

constraints:
  max_positions: 10
  position_sizing: kelly
  stop_loss: 0.05
  take_profit: 0.15

# Adversarial settings
adversarial_rounds: 20
red_team_intensity: high
red_team_strategies:
  - regime_change
  - black_swan
  - high_volatility
  - gap_scenarios

# Gauntlet (critical for trading)
enable_gauntlet: true
gauntlet_rounds:
  - loongflow
  - red_team  # Stress testing
  - gold_team  # Consensus

# Knowledge engine
enable_knowledge_engine: true
market_regime_detection: true
```

---

### Example 13: Multi-Objective Config

**File:** `evolve.mo.config.yaml`

```yaml
# Multi-objective optimization
evolution_mode: mo  # NSGA-II
max_iterations: 100

# Objectives (competing)
objectives:
  - performance
  - cost
  - reliability
  - maintainability

# MO settings
pareto_front_size: 100
crossover_probability: 0.9
mutation_probability: 0.1

# Pareto front analysis
compute_hypervolume: true
compute_spread: true
compute_igd: true
compute_gd: true

# Visualization
generate_pareto_plots: true
plot_objectives:
  - performance vs cost
  - reliability vs maintainability
  - performance vs reliability

# Output all Pareto solutions
save_pareto_front: true
pareto_output_file: pareto_solutions.json
```

---

### Example 14: Expensive Evaluation Config

**File:** `evolve.expensive.config.yaml`

```yaml
# For expensive evaluations (lab experiments, simulations)
evolution_mode: pes  # Reduces evaluations by 60%
max_evaluations: 20  # Minimal budget

# PES settings (maximize sample efficiency)
enable_planning: true
enable_memory: true
early_stopping: true
early_stop_threshold: 0.85
parallel_candidates: 5

# Sequential evaluation (no parallelism)
parallel_evaluations: 1

# Minimal validation
enable_gauntlet: false
enable_knowledge_engine: false

# Save everything
save_intermediate_results: true
checkpoint_interval: 1

# Detailed logging
log_level: DEBUG
log_all_evaluations: true
```

---

### Example 15: Fast Evaluation Config

**File:** `evolve.fast.config.yaml`

```yaml
# For cheap evaluations (simulated users, calculations)
evolution_mode: qd  # Explore diverse solutions
max_evaluations: 500  # Large budget

# QD settings
feature_dimensions:
  - complexity
  - diversity
  - novelty
feature_bins: 20
archive_size: 1000

# Maximum parallelism
parallel_evaluations: 16
evaluation_timeout: 10

# No validation (evaluations are cheap)
enable_gauntlet: false

# Minimal logging
log_level: WARNING
save_intermediate_results: false
```

---

## 4. Domain-Specific Examples

### Example 16: Resource-Constrained Config

**File:** `evolve.constrained.config.yaml`

```yaml
# Limited resources (CPU, memory, API budget)
evolution_mode: pes  # Sample efficient
max_evaluations: 30

# Resource limits
parallel_evaluations: 1  # Sequential
evaluation_timeout: 120
memory_limit_mb: 512
cpu_limit: 1.0

# Minimal population
population_size: 30
num_islands: 1

# Disable resource-intensive features
enable_gauntlet: false
enable_knowledge_engine: false
save_intermediate_results: false

# Minimal logging
log_level: WARNING
```

---

### Example 17: Production Deployment Config

**File:** `evolve.prod.config.yaml`

```yaml
# Production deployment configuration
evolution_mode: auto
max_iterations: 100
max_evaluations: 100

# Resources (controlled)
parallel_evaluations: 4
evaluation_timeout: 300
memory_limit_mb: 4096
cpu_limit: 4.0

# Comprehensive validation
enable_gauntlet: true
gauntlet_rounds:
  - loongflow
  - red_team
  - gold_team

# Knowledge engine (enabled for learning)
enable_knowledge_engine: true
extract_knowledge: true
use_past_solutions: true

# Logging (comprehensive)
log_level: INFO
log_dir: /var/log/evolve
save_intermediate_results: true
checkpoint_interval: 20

# Monitoring
enable_metrics: true
metrics_interval: 10
send_alerts: true
alert_on_failure: true
```

---

### Example 18: Development Config

**File:** `evolve.dev.config.yaml`

```yaml
# Development configuration
evolution_mode: auto
max_iterations: 20  # Quick iteration
max_evaluations: 30

# Small populations
population_size: 50
num_islands: 1

# Fast execution
parallel_evaluations: 2
evaluation_timeout: 60

# Disable validation for speed
enable_gauntlet: false
enable_knowledge_engine: false

# Verbose logging for debugging
log_level: DEBUG
log_dir: ./logs/dev
save_intermediate_results: true
checkpoint_interval: 5

# Reproducible (for debugging)
random_seed: 42
```

---

### Example 19: Custom Profile Config

**Profile:** `~/.evolve/profiles/my_finance.yaml`

```yaml
# Custom finance profile
name: My Finance Profile
description: Optimized for my specific use case
version: 1.0

# Inherit from prod
inherit: prod

# Override for specific needs
overrides:
  domain: finance
  evolution_mode: pes

  # Specific objectives
  objectives:
    - return
    - risk
    - sharpe_ratio
    - sortino_ratio

  # Constraints
  constraints:
    max_position_size: 0.15  # Higher than default
    sector_limits:
      technology: 0.30
      healthcare: 0.25
      finance: 0.20

  # Budget
  max_evaluations: 75

  # Knowledge engine
  enable_knowledge_engine: true
  extract_knowledge: true
  knowledge_sources:
    - past_strategies
    - market_regimes
    - risk_models
```

**Use:**
```bash
evolve --profile my_finance problem="..."
```

---

### Example 20: CLI-Based Config Management

**Scenario:** Complete workflow using only CLI

```bash
# Step 1: Initialize config
evolve config init --preset finance evolve.config.yaml

# Step 2: View config
evolve config show evolve.config.yaml

# Step 3: Validate config
evolve config validate evolve.config.yaml

# Step 4: Edit config (manual or programmatic)
# Add custom settings...

# Step 5: Re-validate
evolve config validate evolve.config.yaml

# Step 6: Test with small run
evolve --config evolve.config.yaml --max-evaluations 5 problem="..."

# Step 7: Full run
evolve --config evolve.config.yaml problem="Optimize portfolio"

# Step 8: Export results
evolve --config evolve.config.yaml \
  problem="..." \
  --output-file results.json
```

---

## Additional Examples

### Example A: Docker Compose Integration

**File:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  evolve:
    image: openevolve:latest
    environment:
      # Configuration via environment
      EVOLVE_EVOLUTION_MODE: pes
      EVOLVE_MAX_EVALUATIONS: 50
      EVOLVE_DOMAIN: finance
      EVOLVE_API_KEY: ${API_KEY}
      EVOLVE_NEO4J_URI: bolt://neo4j:7687
      EVOLVE_QDRANT_HOST: qdrant
    volumes:
      - ./evolve.config.yaml:/app/evolve.config.yaml
      - ./results:/app/results
    depends_on:
      - neo4j
      - qdrant

  neo4j:
    image: neo4j:latest
    environment:
      NEO4J_AUTH: none
    ports:
      - "7474:7474"
      - "7687:7687"

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
```

---

### Example B: Kubernetes ConfigMap

**File:** `k8s-configmap.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: evolve-config
data:
  evolve.config.yaml: |
    evolution_mode: pes
    max_evaluations: 50
    domain: finance
    enable_knowledge_engine: true
    neo4j_uri: bolt://neo4j-service:7687
    qdrant_host: qdrant-service
---
apiVersion: v1
kind: Secret
metadata:
  name: evolve-secrets
type: Opaque
stringData:
  api-key: sk-...
```

---

### Example C: CI/CD Integration

**File:** `.github/workflows/evolve.yml`

```yaml
name: Evolution

on: [push]

jobs:
  evolve:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'

      - name: Install OpenEvolve
        run: pip install openevolve

      - name: Run Evolution
        env:
          EVOLVE_API_KEY: ${{ secrets.API_KEY }}
          EVOLVE_MAX_EVALUATIONS: 50
        run: |
          evolve --profile test \
            problem="Optimize portfolio" \
            --output-file results.json

      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: results
          path: results.json
```

---

**End of Configuration Examples**

For more information:
- [Configuration Guide](CONFIGURATION_GUIDE.md) - Master configuration guide
- [Profile Guide](PROFILE_GUIDE.md) - Profile documentation
- [Preset Catalog](PRESET_CATALOG.md) - Preset documentation
- [Migration Guide](CONFIGURATION_MIGRATION.md) - Migration documentation
