# Profile Guide

**Version:** 1.0
**Date:** January 30, 2026
**Status:** Production Ready

Complete guide to using and creating configuration profiles for different environments.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Available Profiles](#2-available-profiles)
3. [Using Profiles](#3-using-profiles)
4. [Creating Custom Profiles](#4-creating-custom-profiles)
5. [Profile Inheritance](#5-profile-inheritance)
6. [Profile Best Practices](#6-profile-best-practices)
7. [Profile Examples](#7-profile-examples)

---

## 1. Overview

### What are Profiles?

Profiles are pre-configured settings for common environments and use cases. They provide:
- **Consistency:** Standardized configurations across teams
- **Speed:** Quick environment setup
- **Validation:** Pre-tested parameter combinations
- **Reproducibility:** Version-controlled settings

### Profile Locations

Profiles are loaded from these locations (in order):
1. Built-in profiles (system defaults)
2. `/etc/evolve/profiles/*.yaml` (global profiles)
3. `~/.evolve/profiles/*.yaml` (user profiles)
4. `./evolve.profiles/*.yaml` (project profiles)

### Available Built-in Profiles

| Profile | Description | Use Case |
|---------|-------------|----------|
| `dev` | Development | Active development and testing |
| `test` | Testing | Automated testing and CI/CD |
| `prod` | Production | Production deployments |
| `benchmark` | Benchmarking | Performance comparisons |

---

## 2. Available Profiles

### 2.1 Development Profile

**Profile Name:** `dev`

**Use During:** Active development and testing

**Characteristics:**
- Fast iteration
- Verbose logging
- Gauntlet disabled for speed
- Save intermediate results
- Small populations

**Complete Configuration:**

```yaml
# dev profile configuration
# Location: ~/.evolve/profiles/dev.yaml or built-in

# Evolutionary parameters
evolution_mode: auto
max_iterations: 20
max_evaluations: 30
population_size: 50
convergence_threshold: 0.01

# Resources
parallel_evaluations: 2
evaluation_timeout: 60
memory_limit_mb: 1024
cpu_limit: 2.0

# Gauntlet (disabled for speed)
enable_gauntlet: false

# Knowledge engine (optional)
enable_knowledge_engine: false
extract_knowledge: false

# Logging (verbose for debugging)
log_level: DEBUG
log_dir: ./logs
save_intermediate_results: true
checkpoint_interval: 5

# Early stopping (quick iteration)
early_stopping: true
early_stopping_patience: 3

# Random seed (reproducible for debugging)
random_seed: 42
```

**When to Use:**
- ✅ Developing new features
- ✅ Debugging evolution runs
- ✅ Quick experimentation
- ✅ Local testing
- ✅ Learning the system

**NOT for:**
- ❌ Production runs
- ❌ Performance benchmarking
- ❌ Final optimization
- ❌ Published results

**Advantages:**
- Fast execution (small populations, few iterations)
- Detailed logging for debugging
- Reproducible (fixed seed)
- Saves intermediate results for analysis

**Trade-offs:**
- Lower solution quality (small populations)
- Not suitable for final results
- Gauntlet disabled (no quality validation)

---

### 2.2 Test Profile

**Profile Name:** `test`

**Use During:** Automated testing

**Characteristics:**
- Deterministic behavior
- Minimal resource usage
- Quick validation
- Reproducible results

**Complete Configuration:**

```yaml
# test profile configuration

# Evolutionary parameters
evolution_mode: standard
max_iterations: 10
max_evaluations: 20
population_size: 20
convergence_threshold: 0.01

# Resources
parallel_evaluations: 1  # Sequential for determinism
evaluation_timeout: 30
memory_limit_mb: 512
cpu_limit: 1.0

# Gauntlet (quick validation)
enable_gauntlet: true
gauntlet_rounds:
  - loongflow  # Single round only

# Knowledge engine (disabled for speed)
enable_knowledge_engine: false
extract_knowledge: false

# Logging (minimal)
log_level: WARNING
log_dir: ./logs/test
save_intermediate_results: false

# Deterministic
random_seed: 42
early_stopping: false  # Run full budget

# Performance
checkpoint_interval: 100  # Rare checkpoints
```

**When to Use:**
- ✅ Unit tests
- ✅ Integration tests
- ✅ CI/CD pipelines
- ✅ Validation experiments
- ✅ Smoke tests

**Advantages:**
- Deterministic (fixed seed, sequential)
- Fast execution (small budgets)
- Minimal resource usage
- Quick validation

**Trade-offs:**
- Not suitable for optimization
- Low solution quality
- No exploration of search space

---

### 2.3 Production Profile

**Profile Name:** `prod`

**Use During:** Production deployments

**Characteristics:**
- Optimized for performance
- Conservative defaults
- Comprehensive logging
- Resource limits
- Full validation

**Complete Configuration:**

```yaml
# prod profile configuration

# Evolutionary parameters
evolution_mode: auto
max_iterations: 100
max_evaluations: 100
population_size: 100
convergence_threshold: 0.001

# Resources
parallel_evaluations: 4
evaluation_timeout: 300
memory_limit_mb: 4096
cpu_limit: 4.0

# Gauntlet (comprehensive)
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
log_dir: ./logs/production
save_intermediate_results: true
checkpoint_interval: 20

# Early stopping (safe)
early_stopping: true
early_stopping_patience: 15

# Random seed (different each run)
random_seed: null  # Random for each run
```

**When to Use:**
- ✅ Production runs
- ✅ Important optimizations
- ✅ Client deliverables
- ✅ Performance-critical applications
- ✅ Published results

**Advantages:**
- Balanced configuration
- Full validation (gauntlet)
- Knowledge extraction
- Resource limits
- Comprehensive logging

**Trade-offs:**
- Higher resource usage
- Longer execution time
- More complex setup

---

### 2.4 Benchmark Profile

**Profile Name:** `benchmark`

**Use During:** Performance benchmarking

**Characteristics:**
- Consistent settings
- Detailed metrics
- Reproducible results
- No adaptive features

**Complete Configuration:**

```yaml
# benchmark profile configuration

# Evolutionary parameters
evolution_mode: standard  # Fixed mode for consistency
max_iterations: 50
max_evaluations: 50
population_size: 50
convergence_threshold: 0.001

# Resources
parallel_evaluations: 4
evaluation_timeout: 120
memory_limit_mb: 2048
cpu_limit: null  # No artificial limit

# Gauntlet (disabled for pure algorithm performance)
enable_gauntlet: false

# Knowledge engine (disabled for fair comparison)
enable_knowledge_engine: false
extract_knowledge: false

# Logging (metrics-focused)
log_level: INFO
log_dir: ./logs/benchmark
save_intermediate_results: true
checkpoint_interval: 10

# Deterministic
random_seed: 42
early_stopping: false

# Metrics (detailed)
enable_metrics: true
metrics_interval: 1  # Every iteration
track_diversity: true
track_convergence: true
track_performance: true
```

**When to Use:**
- ✅ Performance comparisons
- ✅ Algorithm research
- ✅ Paper experiments
- ✅ System validation
- ✅ A/B testing

**Advantages:**
- Reproducible (fixed seed)
- Consistent (fixed mode)
- Detailed metrics
- Fair comparisons

**Trade-offs:**
- Not adaptive (fixed settings)
- May not be optimal for specific problems
- No learning from past runs

---

## 3. Using Profiles

### 3.1 Method 1: CLI Flag

```bash
# Use profile via CLI
evolve --profile dev problem="Optimize portfolio"

# Override profile values
evolve --profile prod --max-iterations 200 problem="..."
```

### 3.2 Method 2: Config File

```yaml
# evolve.config.yaml
profile: prod

# Override specific values
overrides:
  max_iterations: 200
  parallel_evaluations: 8
```

### 3.3 Method 3: Environment Variable

```bash
export EVOLVE_PROFILE=prod
evolve problem="..."
```

### 3.4 Method 4: Programmatic

```python
from openevolve import evolve
from openevolve.config import load_profile

# Load profile
profile_config = load_profile('prod')

# Use profile
result = await evolve(
    problem="...",
    config=profile_config
)
```

---

## 4. Creating Custom Profiles

### 4.1 Method 1: Profile File

Create `~/.evolve/profiles/my_profile.yaml`:

```yaml
# My custom profile

# Base configuration
evolution_mode: pes
max_iterations: 75
population_size: 75

# Domain-specific
domain: finance
objectives:
  - return
  - risk

# Knowledge engine
enable_knowledge_engine: true
extract_knowledge: true

# Resources
max_evaluations: 75
parallel_evaluations: 4
```

Use it:
```bash
evolve --profile my_profile problem="..."
```

### 4.2 Method 2: Project-Specific Profile

Create `./evolve.profiles/finance_fast.yaml`:

```yaml
# Finance fast optimization profile

evolution_mode: pes
max_evaluations: 30
enable_planning: true
enable_memory: true
early_stopping: true

domain: finance
objectives:
  - return
  - risk

# Quick evaluation
enable_gauntlet: false
save_intermediate_results: false
```

Use it:
```bash
evolve --profile finance_fast problem="..."
```

### 4.3 Method 3: Profile Template

```yaml
# ~/.evolve/profiles/template.yaml
# Copy this to create new profiles

# Profile metadata
name: "My Profile"
description: "Profile description"
version: "1.0"

# Evolutionary parameters
evolution_mode: auto
max_iterations: 100
max_evaluations: 100
population_size: 100

# Domain
domain: general
objectives: []

# Knowledge engine
enable_knowledge_engine: false
extract_knowledge: false

# Gauntlet
enable_gauntlet: true
gauntlet_rounds:
  - loongflow
  - red_team
  - gold_team

# Resources
parallel_evaluations: 4
evaluation_timeout: 300
memory_limit_mb: 4096
cpu_limit: null

# Logging
log_level: INFO
log_dir: ./logs
save_intermediate_results: true
checkpoint_interval: 20

# Early stopping
early_stopping: true
early_stopping_patience: 10

# Random seed
random_seed: null
```

---

## 5. Profile Inheritance

### 5.1 Basic Inheritance

Create profile that inherits from base:

```yaml
# ~/.evolve/profiles/my_prod.yaml
inherit: prod  # Inherit from prod profile

overrides:
  max_iterations: 200  # Override specific values
  parallel_evaluations: 8
  log_level: DEBUG
```

### 5.2 Multi-Level Inheritance

```yaml
# finance_prod.yaml
inherit: prod

overrides:
  domain: finance
  evolution_mode: pes
  max_evaluations: 50

# trading_prod.yaml
inherit: finance_prod

overrides:
  evolution_mode: adversarial
  adversarial_rounds: 30
```

### 5.3 Profile Composition

```yaml
# my_custom.yaml
inherit:
  - prod  # Base production settings
  - fast  # Apply fast presets

overrides:
  domain: finance
  max_iterations: 75
```

---

## 6. Profile Best Practices

### 6.1 DO:

1. **Use descriptive names**
   ```yaml
   # ✓ Good
   name: finance_fast_budget

   # ✗ Bad
   name: config1
   ```

2. **Document profiles**
   ```yaml
   # Finance Fast Budget Profile
   # Optimized for finance problems with limited API budget
   # Uses PES mode to reduce evaluations by 60%
   ```

3. **Use inheritance**
   ```yaml
   inherit: prod  # Start from proven base
   overrides:
     max_iterations: 200  # Override only what's needed
   ```

4. **Version control profiles**
   ```bash
   git add evolve.profiles/
   git commit -m "Add finance fast profile"
   ```

5. **Test profiles**
   ```bash
   evolve --profile my_profile test --max-iterations 5
   ```

### 6.2 DON'T:

1. **Hardcode secrets**
   ```yaml
   # ✗ Bad
   api_key: sk-...

   # ✓ Good
   # Use environment variable
   export EVOLVE_API_KEY=sk-...
   ```

2. **Create too many profiles**
   - Keep it simple
   - Use inheritance to avoid duplication
   - Delete unused profiles

3. **Forget to document**
   ```yaml
   # ✗ Bad
   max_iterations: 150

   # ✓ Good
   max_iterations: 150  # Increased for complex problem
   ```

4. **Mix concerns**
   ```yaml
   # ✗ Bad
   # Profile mixes dev and prod settings
   max_iterations: 20
   enable_gauntlet: true  # Production feature
   log_level: DEBUG  # Development feature

   # ✓ Good
   # Separate profiles for dev and prod
   ```

5. **Ignore updates**
   - Update profiles when schema changes
   - Test profiles after system updates
   - Document breaking changes

---

## 7. Profile Examples

### 7.1 Finance Development Profile

```yaml
# ~/.evolve/profiles/finance_dev.yaml

name: Finance Development
description: Fast iteration for finance problems
version: 1.0

inherit: dev

overrides:
  domain: finance
  evolution_mode: pes

  # Finance-specific objectives
  objectives:
    - return
    - risk
    - sharpe_ratio

  # Quick iteration
  max_iterations: 15
  max_evaluations: 20

  # Knowledge engine disabled for speed
  enable_knowledge_engine: false

  # Single gauntlet round
  enable_gauntlet: true
  gauntlet_rounds:
    - loongflow
```

### 7.2 Trading Production Profile

```yaml
# ~/.evolve/profiles/trading_prod.yaml

name: Trading Production
description: Robust trading strategy optimization
version: 1.0

inherit: prod

overrides:
  domain: trading
  evolution_mode: adversarial

  # Trading objectives
  objectives:
    - sharpe_ratio
    - max_drawdown
    - win_rate

  # Adversarial for robustness
  adversarial_rounds: 30
  red_team_intensity: high

  # Comprehensive gauntlet
  enable_gauntlet: true
  gauntlet_rounds:
    - loongflow
    - red_team
    - gold_team

  # Knowledge extraction
  enable_knowledge_engine: true
  extract_knowledge: true
```

### 7.3 Science Budget Profile

```yaml
# ~/.evolve/profiles/science_budget.yaml

name: Science Budget
description: Science optimization with limited experiment budget
version: 1.0

inherit: prod

overrides:
  domain: science
  evolution_mode: qd  # Quality-Diversity for exploration

  # Limited budget
  max_evaluations: 20

  # QD settings
  feature_dimensions:
    - yield
    - purity
    - cost
  feature_bins: 5

  # Minimal gauntlet (experiments are expensive)
  enable_gauntlet: true
  gauntlet_rounds:
    - loongflow

  # Save all results
  save_intermediate_results: true
  checkpoint_interval: 1
```

### 7.4 Benchmark Profile

```yaml
# ~/.evolve/profiles/benchmark_strict.yaml

name: Benchmark Strict
description: Strict benchmarking with full reproducibility
version: 1.0

inherit: benchmark

overrides:
  # Fixed mode for fair comparison
  evolution_mode: standard

  # Fixed budget
  max_iterations: 50
  max_evaluations: 50
  population_size: 50

  # Deterministic
  random_seed: 42
  parallel_evaluations: 1

  # No adaptive features
  enable_knowledge_engine: false
  early_stopping: false

  # Detailed metrics
  enable_metrics: true
  metrics_interval: 1
```

---

**End of Profile Guide**

For more information:
- [Configuration Guide](CONFIGURATION_GUIDE.md) - Master configuration guide
- [Preset Catalog](PRESET_CATALOG.md) - Preset documentation
- [Configuration Examples](CONFIGURATION_EXAMPLES.md) - Working examples
