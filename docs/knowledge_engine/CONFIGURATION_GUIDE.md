# Configuration Guide - Complete Guide

**Version:** 1.0
**Date:** January 30, 2026
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#1-overview)
2. [Quick Start](#2-quick-start)
3. [Configuration Hierarchy](#3-configuration-hierarchy)
4. [Configuration Files](#4-configuration-files)
5. [Environment Variables](#5-environment-variables)
6. [Profiles](#6-profiles)
7. [Presets](#7-presets)
8. [Configuration Parameters](#8-configuration-parameter-reference)
9. [Runtime Configuration](#9-runtime-configuration)
10. [CLI Tools](#10-cli-tools)
11. [Best Practices](#11-best-practices)
12. [Troubleshooting](#12-troubleshooting)
13. [Migration Guide](#13-migration-guide)

---

## 1. Overview

### 1.1 What is the Configuration System?

The OpenEvolve Configuration System is a comprehensive, hierarchical configuration management framework designed for evolutionary optimization workflows. It provides:

- **Single Source of Truth:** All 272+ parameters managed in one place
- **Validation:** Type-safe parameter validation with detailed error messages
- **Flexibility:** Multiple configuration sources (files, environment variables, CLI, runtime)
- **Convenience:** Pre-configured profiles and presets for common use cases
- **Runtime Updates:** Dynamic configuration changes without restart

### 1.2 Configuration Hierarchy (7 Levels)

The configuration system uses a 7-level hierarchy with clear precedence rules:

```
Level 1: Runtime Overrides (HIGHEST PRIORITY)
    ↓
Level 2: Environment Variables
    ↓
Level 3: Local Config File (./evolve.config.yaml)
    ↓
Level 4: Profile (dev, test, prod, benchmark)
    ↓
Level 5: User Config (~/.evolve/config.yaml)
    ↓
Level 6: Global Config (/etc/evolve/config.yaml)
    ↓
Level 7: Defaults (LOWEST PRIORITY)
```

**Precedence Rules:**
- Higher levels override lower levels
- Merges are deep (nested structures preserve specificity)
- Lists are replaced, not merged
- Runtime changes take immediate effect

### 1.3 Configuration Sources

| Source | Priority | Format | Use Case |
|--------|----------|--------|----------|
| Runtime API | 1 (Highest) | Python dict | Dynamic adjustments |
| Environment Variables | 2 | `EVOLVE_*` | Container/CI/CD |
| Local Config | 3 | YAML/JSON/TOML | Project-specific settings |
| Profiles | 4 | YAML | Environment configuration |
| User Config | 5 | YAML | Personal preferences |
| Global Config | 6 | YAML | Organization standards |
| Defaults | 7 (Lowest) | Built-in | Fallback values |

### 1.4 When to Use Each Method

#### Use Runtime API When:
- Implementing adaptive algorithms
- Responding to system conditions
- Implementing user preferences in interactive sessions
- A/B testing configuration changes

#### Use Environment Variables When:
- Deploying in containers (Docker, Kubernetes)
- Configuring CI/CD pipelines
- Managing secrets (API keys, tokens)
- Platform-as-Code (Infrastructure as Code)

#### Use Config Files When:
- Project requires version-controlled settings
- Complex nested configurations
- Team collaboration (commit to repo)
- Documentation and reproducibility

#### Use Profiles When:
- Different environments (dev, test, prod)
- Standardized team configurations
- Pre-validated settings
- Quick environment switching

#### Use Presets When:
- Domain-specific optimization (finance, trading, science)
- Use-case patterns (exploration, refinement, robustness)
- Performance tuning (fast, balanced, thorough)
- Getting started quickly

---

## 2. Quick Start

### 2.1 5-Minute Configuration Setup

#### Step 1: Create Basic Config (1 min)

```bash
# Navigate to your project
cd my_project

# Create config file
cat > evolve.config.yaml << 'EOF'
# Evolutionary parameters
evolution_mode: auto
max_iterations: 100
population_size: 100

# Domain
domain: finance
objectives:
  - return
  - risk

# Knowledge engine
enable_knowledge_engine: true
extract_knowledge: true
EOF
```

#### Step 2: Set Environment Variables (1 min)

```bash
# API configuration
export EVOLVE_API_KEY="your-api-key"
export EVOLVE_MODEL="gpt-4"
export EVOLVE_LOG_LEVEL="INFO"

# Resource limits
export EVOLVE_MAX_EVALUATIONS=100
export EVOLVE_PARALLEL_EVALUATIONS=4
```

#### Step 3: First Evolution (3 min)

```python
import asyncio
from openevolve import evolve

async def main():
    # Config loaded automatically from:
    # 1. evolve.config.yaml
    # 2. Environment variables
    # 3. Default values

    result = await evolve(
        problem="Optimize portfolio allocation",
        domain="finance"
    )

    print(f"Best fitness: {result['fitness']}")
    print(f"Evaluations: {result['evaluations']}")
    print(f"Strategy: {result['strategy_used']}")

asyncio.run(main())
```

**Expected Output:**
```
Loaded config from: evolve.config.yaml
Applied 3 environment variable overrides
Using evolution mode: pes (auto-selected)
Knowledge engine: enabled

Best fitness: 0.85
Evaluations: 30 (vs 75 baseline)
Strategy: pes
Improvement: 60% fewer evaluations
```

### 2.2 Basic Configuration File

#### Minimal Configuration (5 lines)

```yaml
# evolve.config.yaml - Minimal configuration

evolution_mode: auto          # Auto-select best mode
max_iterations: 100           # Maximum iterations
domain: finance               # Problem domain
enable_knowledge_engine: true # Enable learning
log_level: INFO              # Logging verbosity
```

#### Recommended Configuration (20 lines)

```yaml
# evolve.config.yaml - Recommended configuration

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
  - sharpe_ratio

# Knowledge engine
enable_knowledge_engine: true
extract_knowledge: true
use_past_solutions: true

# Gauntlet (3-round evaluation)
enable_gauntlet: true
gauntlet_rounds:
  - loongflow
  - red_team
  - gold_team

# Resources
max_evaluations: 100
parallel_evaluations: 4
evaluation_timeout: 300

# Logging
log_level: INFO
save_intermediate_results: true
```

### 2.3 Environment Variables

#### Naming Convention

All environment variables follow the pattern:
```bash
EVOLVE_<SECTION>_<PARAMETER_NAME>
```

#### Common Environment Variables

```bash
# Core parameters
EVOLVE_MAX_ITERATIONS=100
EVOLVE_POPULATION_SIZE=100
EVOLVE_DOMAIN=finance

# API configuration
EVOLVE_API_KEY=sk-...
EVOLVE_API_BASE=https://api.openai.com/v1
EVOLVE_MODEL=gpt-4
EVOLVE_TEMPERATURE=0.7

# Knowledge engine
EVOLVE_ENABLE_KNOWLEDGE_ENGINE=true
EVOLVE_NEO4J_URI=bolt://localhost:7687
EVOLVE_QDRANT_HOST=localhost
EVOLVE_QDRANT_PORT=6333

# Resources
EVOLVE_MAX_EVALUATIONS=100
EVOLVE_PARALLEL_EVALUATIONS=4
EVOLVE_MEMORY_LIMIT_MB=4096

# Logging
EVOLVE_LOG_LEVEL=INFO
EVOLVE_LOG_DIR=./logs
```

### 2.4 First Evolution with Custom Config

#### Config-Driven Evolution

```python
import asyncio
from openevolve import evolve

async def main():
    # All configuration from evolve.config.yaml
    result = await evolve(
        problem="Optimize portfolio allocation",
        domain="finance"
    )

    return result

asyncio.run(main())
```

#### Programmatic Configuration

```python
import asyncio
from openevolve import evolve
from openevolve.config import UnifiedEvolutionConfig

async def main():
    # Create custom config
    config = UnifiedEvolutionConfig(
        evolution_mode="pes",
        max_iterations=50,
        population_size=50,
        domain="finance",
        enable_planning=True,
        enable_memory=True
    )

    # Use custom config
    result = await evolve(
        problem="Optimize portfolio allocation",
        config=config
    )

    return result

asyncio.run(main())
```

#### Hybrid Configuration

```python
import asyncio
from openevolve import evolve

async def main():
    # Config loaded from evolve.config.yaml
    # Runtime overrides:
    result = await evolve(
        problem="Optimize portfolio allocation",
        domain="finance",
        max_iterations=200,  # Override config file
        enable_gauntlet=True  # Override config file
    )

    return result

asyncio.run(main())
```

---

## 3. Configuration Hierarchy

### 3.1 Level 1: Runtime Overrides (Highest Priority)

Runtime overrides have the highest priority and take immediate effect.

#### When to Use
- Dynamic parameter tuning
- Adaptive algorithms
- User interaction
- A/B testing

#### How to Use

```python
# Method 1: Direct parameter override
result = await evolve(
    problem="...",
    max_iterations=200  # Runtime override
)

# Method 2: Runtime update
engine = EvolutionEngine()
await engine.update_config(max_iterations=200)
result = await engine.evolve(problem="...")

# Method 3: Dynamic callback
async def adaptive_config(iteration, history):
    if history.converged():
        return {"early_stopping": True}
    return {}

result = await evolve(
    problem="...",
    adaptive_config=adaptive_config
)
```

### 3.2 Level 2: Environment Variables

Environment variables override config files but are overridden by runtime changes.

#### When to Use
- Containerized deployments
- CI/CD pipelines
- Secret management
- Platform configuration

#### How to Use

```bash
# Set environment variables
export EVOLVE_MAX_ITERATIONS=200
export EVOLVE_API_KEY="sk-..."
export EVOLVE_ENABLE_KNOWLEDGE_ENGINE=true

# Run evolution
python my_evolution.py
```

**Python equivalent:**
```python
import os
os.environ['EVOLVE_MAX_ITERATIONS'] = '200'

from openevolve import evolve
result = await evolve(problem="...")
```

### 3.3 Level 3: Local Config File

Local config files (`./evolve.config.yaml`) provide project-specific configuration.

#### When to Use
- Project-specific settings
- Version-controlled configuration
- Team collaboration
- Reproducible experiments

#### Supported Formats

**YAML (Recommended):**
```yaml
evolution_mode: auto
max_iterations: 100
domain: finance
```

**JSON:**
```json
{
  "evolution_mode": "auto",
  "max_iterations": 100,
  "domain": "finance"
}
```

**TOML:**
```toml
evolution_mode = "auto"
max_iterations = 100
domain = "finance"
```

### 3.4 Level 4: Profiles

Profiles provide pre-configured settings for common environments.

#### Available Profiles

| Profile | Use Case | Key Characteristics |
|---------|----------|---------------------|
| `dev` | Development | Fast iteration, verbose logging, gauntlet disabled |
| `test` | Testing | Deterministic, minimal resources, quick validation |
| `prod` | Production | Optimized, monitored, conservative defaults |
| `benchmark` | Benchmarking | Consistent settings, detailed metrics |

#### How to Use

```bash
# Method 1: CLI
evolve --profile test problem="..."

# Method 2: Config file
# evolve.config.yaml
profile: test

# Method 3: Environment variable
export EVOLVE_PROFILE=test
```

#### Profile Inheritance

Create custom profiles by extending existing ones:

```yaml
# evolve.config.yaml
profile: test  # Base profile

overrides:
  max_iterations: 50  # Override specific values
  log_level: DEBUG
```

### 3.5 Level 5: User Config

User config (`~/.evolve/config.yaml`) stores personal preferences.

#### When to Use
- Personal API keys
- Default preferences
- Local development settings
- Custom presets

#### Example

```yaml
# ~/.evolve/config.yaml
api_key: sk-...
model: gpt-4
log_level: INFO

presets:
  my_finance:
    domain: finance
    evolution_mode: pes
    max_iterations: 50
```

### 3.6 Level 6: Global Config

Global config (`/etc/evolve/config.yaml`) provides organization-wide settings.

#### When to Use
- Team standards
- Organizational policies
- Shared infrastructure
- Compliance requirements

#### Example

```yaml
# /etc/evolve/config.yaml
api_base: https://api.company.com/v1
neo4j_uri: bolt://neo4j.company.com:7687
qdrant_host: qdrant.company.com

security:
  audit_logging: true
  max_evaluations: 1000
  allowed_domains:
    - finance
    - trading
```

### 3.7 Level 7: Defaults

Default values are built into the system and used as fallback.

#### Where to Find Defaults

```python
from openevolve.config import get_defaults

defaults = get_defaults()
print(defaults)
```

**Output:**
```python
{
  'evolution_mode': 'auto',
  'max_iterations': 100,
  'population_size': 100,
  'domain': 'general',
  'log_level': 'INFO',
  ...
}
```

### 3.8 How Overrides Work

#### Merge Strategy

```yaml
# Level 7: Defaults
max_iterations: 100
population_size: 100
feature_dimensions: ['complexity', 'diversity']

# Level 3: Config File
max_iterations: 200
feature_dimensions: ['complexity', 'diversity', 'novelty']

# Level 2: Environment
EVOLVE_POPULATION_SIZE=200

# Result (merged):
max_iterations: 200        # From config file (overrides default)
population_size: 200       # From env var (overrides default)
feature_dimensions:        # From config file (merged)
  - complexity
  - diversity
  - novelty
```

#### Conflict Resolution

**Rule:** Higher priority always wins

```python
# Default: max_iterations=100
# Config file: max_iterations=200
# Environment: EVOLVE_MAX_ITERATIONS=300
# Runtime: max_iterations=400

# Final value: 400 (runtime override)
```

#### List Replacement

Lists are replaced, not merged:

```yaml
# Default
feature_dimensions:
  - complexity
  - diversity

# Config file
feature_dimensions:
  - novelty

# Result: ['novelty'] (replaced, not merged)
```

---

## 4. Configuration Files

### 4.1 YAML Format (Recommended)

YAML is the recommended format for configuration files due to its readability and support for comments.

#### Example

```yaml
# evolve.config.yaml
# OpenEvolve Configuration File
# https://openevolve.dev/docs/configuration

# =============================================================================
# EVOLUTIONARY PARAMETERS
# =============================================================================

evolution_mode: auto          # Auto, pes, qd, mo, adversarial, standard
max_iterations: 100           # Maximum iterations
population_size: 100          # Population size
convergence_threshold: 0.001  # Convergence criterion

# =============================================================================
# PROBLEM DEFINITION
# =============================================================================

domain: finance               # Domain: finance, trading, science, etc.
objectives:                   # Optimization objectives
  - return
  - risk
  - sharpe_ratio

constraints:                  # Problem constraints
  max_position_size: 0.1
  sector_diversification: true

# =============================================================================
# KNOWLEDGE ENGINE
# =============================================================================

enable_knowledge_engine: true # Enable knowledge extraction
extract_knowledge: true       # Extract artifacts from runs
use_past_solutions: true      # Use past solutions for guidance

# Knowledge graph connections
neo4j_uri: bolt://localhost:7687
qdrant_host: localhost
qdrant_port: 6333

# =============================================================================
# GAUNTLET (3-ROUND EVALUATION)
# =============================================================================

enable_gauntlet: true         # Enable 3-round gauntlet
gauntlet_rounds:              # Evaluation rounds
  - loongflow                 # Quick AI evaluation
  - red_team                  # Adversarial testing
  - gold_team                 # Consensus validation

# =============================================================================
# RESOURCES
# =============================================================================

max_evaluations: 100          # Maximum evaluations
parallel_evaluations: 4       # Parallel evaluation workers
evaluation_timeout: 300       # Evaluation timeout (seconds)

memory_limit_mb: 4096         # Memory limit
cpu_limit: null               # CPU limit (null = no limit)

# =============================================================================
# LOGGING
# =============================================================================

log_level: INFO               # DEBUG, INFO, WARNING, ERROR
log_dir: ./logs               # Log directory
save_intermediate_results: true # Save intermediate checkpoints

# =============================================================================
# ADVANCED
# =============================================================================

temperature: 0.7              # LLM temperature
top_p: 0.95                   # LLM top_p
max_tokens: 4096              # LLM max tokens
```

### 4.2 JSON Format

JSON is useful for automated configuration generation and machine readability.

#### Example

```json
{
  "evolution_mode": "auto",
  "max_iterations": 100,
  "population_size": 100,
  "convergence_threshold": 0.001,

  "domain": "finance",
  "objectives": ["return", "risk", "sharpe_ratio"],

  "constraints": {
    "max_position_size": 0.1,
    "sector_diversification": true
  },

  "enable_knowledge_engine": true,
  "extract_knowledge": true,
  "use_past_solutions": true,

  "enable_gauntlet": true,
  "gauntlet_rounds": ["loongflow", "red_team", "gold_team"],

  "max_evaluations": 100,
  "parallel_evaluations": 4,
  "evaluation_timeout": 300,

  "log_level": "INFO",
  "log_dir": "./logs",
  "save_intermediate_results": true
}
```

### 4.3 TOML Format

TOML is more explicit than YAML and suitable for simple configurations.

#### Example

```toml
evolution_mode = "auto"
max_iterations = 100
population_size = 100
convergence_threshold = 0.001

domain = "finance"
objectives = ["return", "risk", "sharpe_ratio"]

[constraints]
max_position_size = 0.1
sector_diversification = true

enable_knowledge_engine = true
extract_knowledge = true
use_past_solutions = true

enable_gauntlet = true
gauntlet_rounds = ["loongflow", "red_team", "gold_team"]

max_evaluations = 100
parallel_evaluations = 4
evaluation_timeout = 300

log_level = "INFO"
log_dir = "./logs"
save_intermediate_results = true
```

### 4.4 File Structure

#### Standard Layout

```
my_project/
├── evolve.config.yaml          # Local config (loaded automatically)
├── evolve.config.test.yaml     # Test environment
├── evolve.config.prod.yaml     # Production environment
├── .env                        # Environment variables (gitignored)
└── .evolve/
    ├── config.yaml             # User config
    └── presets/
        ├── fast.yaml
        ├── balanced.yaml
        └── thorough.yaml
```

#### Config File Selection

The system loads config files in this order:

1. `./evolve.config.yaml` (default)
2. `./evolve.config.{profile}.yaml` (if profile set)
3. `~/.evolve/config.yaml` (user config)
4. `/etc/evolve/config.yaml` (global config)

### 4.5 Supported Data Types

| Type | YAML Example | JSON Example | Description |
|------|--------------|--------------|-------------|
| String | `mode: "auto"` | `"mode": "auto"` | Text value |
| Integer | `count: 100` | `"count": 100` | Whole number |
| Float | `rate: 0.7` | `"rate": 0.7` | Decimal number |
| Boolean | `enabled: true` | `"enabled": true` | True/False |
| List | `items: [a, b]` | `"items": ["a", "b"]` | Array |
| Dict | `params: {a: 1}` | `"params": {"a": 1}` | Object |
| Null | `value: null` | `"value": null` | Null/None |

### 4.6 Comments and Documentation

#### YAML Comments

```yaml
# Single-line comment

evolution_mode: auto  # Inline comment

# Multi-line
# comment
# block
max_iterations: 100
```

#### JSON Comments (Not Supported)

JSON does not support comments. Use documentation fields instead:

```json
{
  "_comment": "OpenEvolve Configuration",
  "evolution_mode": "auto",
  "max_iterations": 100,
  "_documentation": {
    "evolution_mode": "Auto-select best evolutionary mode"
  }
}
```

### 4.7 Validation

#### Automatic Validation

Configuration files are automatically validated on load:

```python
from openevolve.config import load_config

try:
    config = load_config("evolve.config.yaml")
except ConfigurationValidationError as e:
    print(f"Validation errors: {e.errors}")
    print(f"Warnings: {e.warnings}")
```

#### Manual Validation

```bash
# CLI validation
evolve config validate evolve.config.yaml

# Output
✓ Configuration is valid
⚠ Warning: max_iterations is high (200), consider reducing for faster runs
```

#### Common Validation Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `unknown_parameter` | Typo in parameter name | Check parameter name |
| `type_error` | Wrong data type | Use correct type |
| `value_out_of_range` | Value outside valid range | Use value in range |
| `missing_required` | Required parameter missing | Add parameter |

---

## 5. Environment Variables

### 5.1 Naming Convention

All environment variables follow this pattern:

```bash
EVOLVE_<SECTION>_<PARAMETER_NAME>
```

**Examples:**
- `EVOLVE_MAX_ITERATIONS`
- `EVOLVE_API_KEY`
- `EVOLVE_NEO4J_URI`

### 5.2 All Parameters Mapped

#### Core Parameters

| Config Parameter | Environment Variable | Type | Default |
|------------------|---------------------|------|---------|
| `evolution_mode` | `EVOLVE_EVOLUTION_MODE` | string | `auto` |
| `max_iterations` | `EVOLVE_MAX_ITERATIONS` | int | `100` |
| `max_evaluations` | `EVOLVE_MAX_EVALUATIONS` | int | `100` |
| `population_size` | `EVOLVE_POPULATION_SIZE` | int | `100` |
| `convergence_threshold` | `EVOLVE_CONVERGENCE_THRESHOLD` | float | `0.001` |

#### PES Parameters

| Config Parameter | Environment Variable | Type | Default |
|------------------|---------------------|------|---------|
| `enable_planning` | `EVOLVE_ENABLE_PLANNING` | bool | `true` |
| `enable_memory` | `EVOLVE_ENABLE_MEMORY` | bool | `true` |
| `early_stopping` | `EVOLVE_EARLY_STOPPING` | bool | `true` |
| `early_stop_threshold` | `EVOLVE_EARLY_STOP_THRESHOLD` | float | `0.9` |
| `parallel_candidates` | `EVOLVE_PARALLEL_CANDIDATES` | int | `3` |

#### QD Parameters

| Config Parameter | Environment Variable | Type | Default |
|------------------|---------------------|------|---------|
| `feature_dimensions` | `EVOLVE_FEATURE_DIMENSIONS` | list | `["complexity", "diversity"]` |
| `feature_bins` | `EVOLVE_FEATURE_BINS` | int | `10` |
| `archive_size` | `EVOLVE_ARCHIVE_SIZE` | int | `100` |
| `grid_resolution` | `EVOLVE_GRID_RESOLUTION` | int | `10` |

#### Knowledge Engine Parameters

| Config Parameter | Environment Variable | Type | Default |
|------------------|---------------------|------|---------|
| `enable_knowledge_engine` | `EVOLVE_ENABLE_KNOWLEDGE_ENGINE` | bool | `false` |
| `extract_knowledge` | `EVOLVE_EXTRACT_KNOWLEDGE` | bool | `true` |
| `neo4j_uri` | `EVOLVE_NEO4J_URI` | string | `bolt://localhost:7687` |
| `qdrant_host` | `EVOLVE_QDRANT_HOST` | string | `localhost` |
| `qdrant_port` | `EVOLVE_QDRANT_PORT` | int | `6333` |

#### Gauntlet Parameters

| Config Parameter | Environment Variable | Type | Default |
|------------------|---------------------|------|---------|
| `enable_gauntlet` | `EVOLVE_ENABLE_GAUNTLET` | bool | `true` |
| `gauntlet_rounds` | `EVOLVE_GAUNTLET_ROUNDS` | list | `["loongflow", "red_team", "gold_team"]` |

#### Resource Parameters

| Config Parameter | Environment Variable | Type | Default |
|------------------|---------------------|------|---------|
| `parallel_evaluations` | `EVOLVE_PARALLEL_EVALUATIONS` | int | `4` |
| `evaluation_timeout` | `EVOLVE_EVALUATION_TIMEOUT` | int | `300` |
| `memory_limit_mb` | `EVOLVE_MEMORY_LIMIT_MB` | int | `null` |
| `cpu_limit` | `EVOLVE_CPU_LIMIT` | float | `null` |

#### API Parameters

| Config Parameter | Environment Variable | Type | Default |
|------------------|---------------------|------|---------|
| `api_key` | `EVOLVE_API_KEY` | string | `null` |
| `api_base` | `EVOLVE_API_BASE` | string | `https://api.openai.com/v1` |
| `model` | `EVOLVE_MODEL` | string | `gpt-4` |
| `temperature` | `EVOLVE_TEMPERATURE` | float | `0.7` |
| `max_tokens` | `EVOLVE_MAX_TOKENS` | int | `4096` |

### 5.3 Type Conversion

#### Boolean Values

```bash
# True (case-insensitive)
EVOLVE_ENABLE_PLANNING=true
EVOLVE_ENABLE_PLANNING=True
EVOLVE_ENABLE_PLANNING=1
EVOLVE_ENABLE_PLANNING=yes

# False (case-insensitive)
EVOLVE_ENABLE_PLANNING=false
EVOLVE_ENABLE_PLANNING=False
EVOLVE_ENABLE_PLANNING=0
EVOLVE_ENABLE_PLANNING=no
```

#### List Values

```bash
# Comma-separated
EVOLVE_FEATURE_DIMENSIONS=complexity,diversity,novelty

# JSON array
EVOLVE_FEATURE_DIMENSIONS='["complexity", "diversity"]'
```

#### Nested Structures

```bash
# JSON object
EVOLVE_CONSTRAINTS='{"max_position_size": 0.1, "sector_diversification": true}'
```

### 5.4 Best Practices

#### DO:

```bash
# Use uppercase
export EVOLVE_API_KEY="sk-..."

# Use underscores for multi-word names
export EVOLVE_MAX_ITERATIONS=100

# Use quotes for strings with spaces
export EVOLVE_LOG_DIR="/path/to/logs"

# Use .env file for local development
echo "EVOLVE_API_KEY=sk-..." > .env
```

#### DON'T:

```bash
# Don't use lowercase
export evolve_api_key="sk-..."  # Wrong

# Don't use hyphens
export EVOLVE-MAX-ITERATIONS=100  # Wrong

# Don't forget to export
EVOLVE_API_KEY="sk-..."  # Won't work (not exported)
```

---

## 6. Profiles

### 6.1 Available Profiles

#### Development Profile

**Use During:** Active development and testing

**Characteristics:**
- Fast iteration
- Verbose logging
- Gauntlet disabled for speed
- Save intermediate results

**Configuration:**
```yaml
max_iterations: 20
population_size: 10
enable_gauntlet: false
log_level: DEBUG
save_intermediate_results: true
early_stopping: true
```

**When to Use:**
- Developing new features
- Debugging evolution runs
- Quick experimentation
- Local testing

**NOT for:**
- Production runs
- Performance benchmarking
- Final optimization

#### Test Profile

**Use During:** Automated testing

**Characteristics:**
- Deterministic behavior
- Minimal resource usage
- Quick validation
- Reproducible results

**Configuration:**
```yaml
max_iterations: 10
population_size: 20
random_seed: 42
enable_gauntlet: true
gauntlet_rounds:
  - loongflow
log_level: WARNING
parallel_evaluations: 1
```

**When to Use:**
- Unit tests
- Integration tests
- CI/CD pipelines
- Validation experiments

#### Production Profile

**Use During:** Production deployments

**Characteristics:**
- Optimized for performance
- Conservative defaults
- Comprehensive logging
- Resource limits

**Configuration:**
```yaml
max_iterations: 100
population_size: 100
enable_gauntlet: true
gauntlet_rounds:
  - loongflow
  - red_team
  - gold_team
log_level: INFO
save_intermediate_results: true
memory_limit_mb: 4096
cpu_limit: 4.0
enable_knowledge_engine: true
```

**When to Use:**
- Production runs
- Important optimizations
- Client deliverables
- Performance-critical applications

#### Benchmark Profile

**Use During:** Performance benchmarking

**Characteristics:**
- Consistent settings
- Detailed metrics
- Reproducible results
- No adaptive features

**Configuration:**
```yaml
max_iterations: 50
population_size: 50
random_seed: 42
enable_knowledge_engine: false
log_level: INFO
enable_metrics: true
metrics_interval: 1
```

**When to Use:**
- Performance comparisons
- Algorithm research
- Paper experiments
- System validation

### 6.2 Creating Custom Profiles

#### Method 1: Profile File

Create `~/.evolve/profiles/my_profile.yaml`:

```yaml
# My custom profile
max_iterations: 75
population_size: 75
evolution_mode: pes
domain: finance
enable_knowledge_engine: true
```

Use it:
```bash
evolve --profile my_profile problem="..."
```

#### Method 2: Profile Inheritance

```yaml
# ~/.evolve/profiles/my_prod.yaml
inherit: prod  # Inherit from prod profile

overrides:
  max_iterations: 200  # Override specific values
  enable_gauntlet: true
```

### 6.3 Profile Best Practices

#### DO:

1. **Version control profiles** (except secrets)
2. **Document profile purpose**
3. **Use inheritence** to avoid duplication
4. **Test profiles** before deploying
5. **Use descriptive names** (`finance_prod`, `science_fast`)

#### DON'T:

1. **Hardcode secrets** in profiles
2. **Create too many profiles** (keep it simple)
3. **Mix concerns** (separate dev vs prod)
4. **Forget to update** profiles when schema changes

---

## 7. Presets

### 7.1 Performance Presets

#### Fast Preset

**Use When:** You need quick results

**Configuration:**
```yaml
max_iterations: 20
population_size: 50
enable_gauntlet: false
enable_knowledge_engine: false
parallel_evaluations: 8
```

**Trade-offs:**
- ⚡ Fast execution
- ❌ Lower solution quality
- ❌ No validation

#### Balanced Preset

**Use When:** Default choice

**Configuration:**
```yaml
max_iterations: 100
population_size: 100
enable_gauntlet: true
enable_knowledge_engine: true
parallel_evaluations: 4
```

**Trade-offs:**
- ⚖️ Balanced speed/quality
- ✅ Good solution quality
- ✅ Reasonable execution time

#### Thorough Preset

**Use When:** Quality is critical

**Configuration:**
```yaml
max_iterations: 500
population_size: 200
enable_gauntlet: true
enable_knowledge_engine: true
parallel_evaluations: 2
```

**Trade-offs:**
- 🎯 Best solution quality
- ❌ Long execution time
- ✅ Comprehensive validation

#### Budget Preset

**Use When:** Limited API budget

**Configuration:**
```yaml
evolution_mode: pes  # Reduces evaluations by 60%
max_evaluations: 30
enable_planning: true
enable_memory: true
early_stopping: true
parallel_evaluations: 1
```

**Trade-offs:**
- 💰 Minimal API usage
- ✅ Still good quality
- ❌ Slower (sequential)

### 7.2 Domain Presets

#### Finance Preset

```yaml
evolution_mode: pes
max_evaluations: 50
enable_planning: true
enable_memory: true
objectives:
  - return
  - risk
  - sharpe_ratio
constraints:
  max_position_size: 0.1
```

#### Trading Preset

```yaml
evolution_mode: adversarial
max_evaluations: 100
adversarial_rounds: 20
objectives:
  - sharpe_ratio
  - max_drawdown
  - win_rate
enable_gauntlet: true
```

#### Science Preset

```yaml
evolution_mode: qd
max_evaluations: 30
feature_dimensions:
  - yield
  - purity
  - cost
grid_resolution: 10
```

### 7.3 Use Case Presets

#### Exploration Preset

```yaml
evolution_mode: qd
max_iterations: 100
feature_bins: 20
archive_size: 500
```

#### Refinement Preset

```yaml
evolution_mode: pes
max_iterations: 50
enable_planning: true
early_stopping: true
```

#### Robustness Preset

```yaml
evolution_mode: adversarial
adversarial_rounds: 30
enable_gauntlet: true
gauntlet_rounds:
  - red_team
  - gold_team
```

---

## 8. Configuration Parameter Reference

For the complete parameter reference, see [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md).

### 8.1 Evolution Parameters (20)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `evolution_mode` | string | `auto` | - | Evolutionary mode |
| `max_iterations` | int | `100` | 1-10000 | Maximum iterations |
| `max_evaluations` | int | `100` | 1-10000 | Maximum evaluations |
| `population_size` | int | `100` | 10-10000 | Population size |
| `convergence_threshold` | float | `0.001` | 0-1 | Convergence criterion |

### 8.2 PES Parameters (15)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `enable_planning` | bool | `true` | - | Enable PES planning |
| `enable_memory` | bool | `true` | - | Enable PES memory |
| `early_stopping` | bool | `true` | - | Enable early stopping |
| `early_stop_threshold` | float | `0.9` | 0-1 | Early stop threshold |

### 8.3 QD Parameters (12)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `feature_dimensions` | list | `["complexity", "diversity"]` | - | Feature dimensions |
| `feature_bins` | int | `10` | 2-100 | Bins per dimension |
| `archive_size` | int | `100` | 10-10000 | Archive size |

### 8.4 MO Parameters (10)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `pareto_front_size` | int | `100` | 10-1000 | Pareto front size |
| `crossover_probability` | float | `0.9` | 0-1 | Crossover probability |

### 8.5 Adversarial Parameters (8)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `adversarial_rounds` | int | `20` | 1-100 | Adversarial rounds |
| `red_team_intensity` | string | `medium` | - | Red team intensity |

### 8.6 Gauntlet Parameters (12)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `enable_gauntlet` | bool | `true` | - | Enable gauntlet |
| `gauntlet_rounds` | list | `["loongflow", "red_team", "gold_team"]` | - | Gauntlet rounds |

### 8.7 Knowledge Engine Parameters (10)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `enable_knowledge_engine` | bool | `false` | - | Enable knowledge engine |
| `extract_knowledge` | bool | `true` | - | Extract knowledge |
| `neo4j_uri` | string | `bolt://localhost:7687` | - | Neo4j URI |

### 8.8 Domain Parameters (6)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `domain` | string | `general` | - | Problem domain |
| `objectives` | list | `[]` | - | Optimization objectives |
| `constraints` | dict | `{}` | - | Problem constraints |

### 8.9 Resource Parameters (9)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `parallel_evaluations` | int | `4` | 1-64 | Parallel workers |
| `evaluation_timeout` | int | `300` | 10-3600 | Timeout (seconds) |
| `memory_limit_mb` | int | `null` | - | Memory limit |
| `cpu_limit` | float | `null` | - | CPU limit |

---

## 9. Runtime Configuration

### 9.1 Updating Parameters Mid-Evolution

#### Method 1: Direct Update

```python
from openevolve import EvolutionEngine

engine = EvolutionEngine()

# Start evolution
task = asyncio.create_task(engine.evolve(problem="..."))

# Update after 10 iterations
await asyncio.sleep(1)
await engine.update_config(max_iterations=200)

result = await task
```

#### Method 2: Callback-Based

```python
async def adaptive_callback(iteration, history):
    """Adaptive configuration based on progress"""
    if iteration == 10 and not history.converged():
        return {"max_iterations": 200}
    elif iteration == 50 and history.best_fitness() > 0.9:
        return {"early_stopping": True}
    return {}

result = await evolve(
    problem="...",
    adaptive_config=adaptive_callback
)
```

### 9.2 Hot-Reload

#### Enable Hot-Reload

```python
from openevolve import EvolutionEngine

engine = EvolutionEngine(
    hot_reload=True,
    config_file="evolve.config.yaml"
)

# Config changes to evolve.config.yaml
# will be automatically reloaded
```

#### Manual Reload

```python
await engine.reload_config()
```

### 9.3 Dynamic Strategy Switching

```python
async def strategy_switcher(iteration, history):
    """Switch strategies based on progress"""
    if iteration == 20 and history.diversity() < 0.3:
        # Low diversity, switch to QD
        return {"evolution_mode": "qd"}
    elif iteration == 40 and history.converged():
        # Converged, switch to refinement
        return {"evolution_mode": "pes", "max_iterations": 80}
    return {}

result = await evolve(
    problem="...",
    adaptive_config=strategy_switcher
)
```

### 9.4 Adaptive Configuration

```python
from openevolve.adaptive import AdaptiveConfigManager

manager = AdaptiveConfigManager()

# Register adaptive rules
manager.register_rule(
    name="low_diversity_boost",
    condition=lambda hist: hist.diversity() < 0.3,
    action={"mutation_rate": 0.2, "exploration_ratio": 0.5}
)

manager.register_rule(
    name="fast_convergence_early_stop",
    condition=lambda hist: hist.convergence_rate() > 0.1,
    action={"early_stopping": True}
)

# Use with evolution
result = await evolve(
    problem="...",
    adaptive_manager=manager
)
```

### 9.5 Resource-Aware Configuration

```python
from openevolve.monitoring import ResourceMonitor

monitor = ResourceMonitor()

async def resource_aware_callback(resources):
    """Adjust config based on available resources"""
    if resources.memory_usage > 0.9:
        # Near memory limit
        return {
            "population_size": 50,
            "archive_size": 50
        }
    elif resources.cpu_usage < 0.5:
        # CPU available
        return {
            "parallel_evaluations": 8
        }
    return {}

result = await evolve(
    problem="...",
    resource_callback=resource_aware_callback
)
```

---

## 10. CLI Tools

### 10.1 Config Commands

```bash
# Validate configuration
evolve config validate evolve.config.yaml

# Show current configuration
evolve config show

# Show default configuration
evolve config defaults

# Merge configurations
evolve config merge base.yaml override.yaml -o merged.yaml

# Diff configurations
evolve config diff base.yaml updated.yaml
```

### 10.2 Profile Commands

```bash
# List available profiles
evolve profile list

# Show profile details
evolve profile show prod

# Create new profile
evolve profile create my_profile --inherit prod

# Validate profile
evolve profile validate my_profile
```

### 10.3 Preset Commands

```bash
# List available presets
evolve preset list

# Show preset details
evolve preset show finance

# Apply preset
evolve preset apply finance -o evolve.config.yaml

# Create custom preset
evolve preset create my_custom --base balanced
```

### 10.4 Environment Commands

```bash
# Show environment variables
evolve env show

# Export to .env file
evolve env export > .env

# Load from .env file
evolve env load .env

# Validate environment
evolve env validate
```

---

## 11. Best Practices

### 11.1 Configuration Best Practices

#### 1. Use Version Control

```bash
# ✓ Good
git add evolve.config.yaml
git commit -m "Update config for production"

# ✗ Bad
git add .env  # Contains secrets
```

#### 2. Separate Configs by Environment

```
configs/
├── base.yaml
├── development.yaml
├── test.yaml
└── production.yaml
```

#### 3. Use Presets as Starting Points

```python
# ✓ Good
config = PresetLoader.load('balanced')
config.max_iterations = 150

# ✗ Bad
config = from_scratch()
# ... defining 50 parameters manually
```

#### 4. Document Custom Values

```yaml
# evolve.config.yaml
max_iterations: 200  # Increased for complex problem
population_size: 150  # Higher diversity needed
```

#### 5. Validate Before Use

```python
# Always validate
config = load_config("evolve.config.yaml")
assert config.validate()
```

#### 6. Use Environment Variables for Secrets

```bash
# ✓ Good
export EVOLVE_API_KEY="sk-..."

# ✗ Bad
# evolve.config.yaml
api_key: "sk-..."  # Don't commit secrets!
```

#### 7. Keep Configs Simple

```yaml
# ✓ Good
evolution_mode: pes
max_iterations: 100

# ✗ Bad
# Overly complex, hard to understand
evolution_mode:
  type: pes
  planning:
    enabled: true
    memory:
      enabled: true
      ...
```

#### 8. Test Config Changes

```bash
# Validate config
evolve config validate evolve.config.yaml

# Test with small run
evolve --max-iterations 5 problem="..."
```

#### 9. Use Profiles for Consistency

```bash
# ✓ Good
evolve --profile prod problem="..."

# ✗ Bad
evolve --max-iterations 100 --population-size 100 ... problem="..."
```

#### 10. Monitor Resource Usage

```yaml
# Set limits
memory_limit_mb: 4096
cpu_limit: 4.0
max_evaluations: 100
```

### 11.2 Common Patterns

#### Pattern 1: Progressive Refinement

```yaml
# Run 1: Exploration
evolution_mode: qd
max_iterations: 50

# Run 2: Refinement
evolution_mode: pes
initial_solution: <from run 1>
max_iterations: 30
```

#### Pattern 2: Multi-Objective Tuning

```yaml
evolution_mode: mo
objectives:
  - return
  - risk
pareto_front_size: 100
```

#### Pattern 3: Robustness Testing

```yaml
evolution_mode: adversarial
adversarial_rounds: 30
enable_gauntlet: true
```

---

## 12. Troubleshooting

### 12.1 Common Config Issues

#### Issue: Config Not Loading

**Symptoms:**
- Default values used instead of config values
- No error message

**Diagnosis:**
```bash
evolve config show
# Check if your config is listed
```

**Solutions:**
1. Check file location: Must be `./evolve.config.yaml`
2. Check syntax: `evolve config validate evolve.config.yaml`
3. Check filename: Must match exactly (case-sensitive)

#### Issue: Validation Error

**Symptoms:**
- `ConfigurationValidationError` raised
- Parameter values rejected

**Diagnosis:**
```bash
evolve config validate evolve.config.yaml
```

**Solutions:**
1. Check parameter types (int vs float, bool vs string)
2. Check parameter ranges
3. Check for typos in parameter names

#### Issue: Override Not Working

**Symptoms:**
- Environment variable or runtime override ignored

**Diagnosis:**
```bash
evolve env show  # Check env vars
evolve config show  # Check effective config
```

**Solutions:**
1. Check naming: `EVOLVE_MAX_ITERATIONS` not `MAX_ITERATIONS`
2. Check precedence: Runtime > Env > Config
3. Check type: Env vars are strings, converted automatically

#### Issue: Performance Problems

**Symptoms:**
- Evolution slower than expected
- High resource usage

**Diagnosis:**
```bash
# Check resource settings
evolve config show | grep -E "parallel|memory|cpu"
```

**Solutions:**
1. Reduce `parallel_evaluations` if CPU overloaded
2. Reduce `population_size` if memory high
3. Enable `early_stopping` to stop early

### 12.2 Debug Configuration

#### Enable Debug Logging

```yaml
# evolve.config.yaml
log_level: DEBUG
```

#### Trace Configuration Loading

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from openevolve import evolve
result = await evolve(problem="...")
```

#### Show Effective Configuration

```bash
evolve config show --effective
```

---

## 13. Migration Guide

For the complete migration guide, see [CONFIGURATION_MIGRATION.md](CONFIGURATION_MIGRATION.md).

### 13.1 From Old Configuration

#### Before (Old Format)

```python
config = {
    'max_generations': 100,
    'pop_size': 50
}
```

#### After (New Format)

```yaml
# evolve.config.yaml
max_iterations: 100
population_size: 50
```

### 13.2 Breaking Changes

#### Version 1.0 → 2.0

| Old Parameter | New Parameter | Notes |
|--------------|---------------|-------|
| `max_generations` | `max_iterations` | Renamed for clarity |
| `pop_size` | `population_size` | Renamed for clarity |
| `enable_qd` | `evolution_mode: qd` | Unified evolution modes |
| `enable_pes` | `evolution_mode: pes` | Unified evolution modes |

### 13.3 Migration Steps

1. **Backup old config**
2. **Run migration tool** (if available)
3. **Validate new config**
4. **Test with sample problem**
5. **Deploy**

---

**End of Configuration Guide**

For more information:
- [Configuration Reference](CONFIGURATION_REFERENCE.md) - Complete parameter reference
- [Profile Guide](PROFILE_GUIDE.md) - Profile documentation
- [Preset Catalog](PRESET_CATALOG.md) - Preset documentation
- [CLI Reference](CLI_REFERENCE.md) - CLI documentation
- [Configuration Examples](CONFIGURATION_EXAMPLES.md) - Working examples
- [Migration Guide](CONFIGURATION_MIGRATION.md) - Migration documentation
