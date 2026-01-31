# Configuration Migration Guide

**Version:** 1.0
**Date:** January 30, 2026
**Status:** Production Ready

Guide for migrating from older configuration formats to the new unified configuration system.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Breaking Changes](#2-breaking-changes)
3. [Migration Steps](#3-migration-steps)
4. [Common Migrations](#4-common-migrations)
5. [Rollback Plan](#5-rollback-plan)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Overview

### What's New in Version 2.0

The new configuration system provides:
- **Unified interface:** Single config for all modules
- **7-level hierarchy:** Flexible configuration sources
- **Profiles & Presets:** Pre-configured settings
- **Runtime updates:** Dynamic configuration changes
- **Better validation:** Type-safe parameter validation
- **CLI tools:** Command-line management

### Migration Path

```
Version 1.0 (Legacy)  →  Version 2.0 (Unified)
     ↓                           ↓
Multiple configs        Single unified config
No validation           Full validation
No runtime updates      Hot-reload & runtime updates
No profiles             4 built-in profiles
No presets              30+ built-in presets
```

---

## 2. Breaking Changes

### 2.1 Parameter Renames

| Old Parameter (v1.0) | New Parameter (v2.0) | Notes |
|---------------------|---------------------|-------|
| `max_generations` | `max_iterations` | More descriptive name |
| `pop_size` | `population_size` | More descriptive name |
| `num_generations` | `max_iterations` | Consolidated naming |
| `enable_qd` | `evolution_mode: qd` | Unified evolution modes |
| `enable_pes` | `evolution_mode: pes` | Unified evolution modes |
| `enable_mo` | `evolution_mode: mo` | Unified evolution modes |
| `enable_adversarial` | `evolution_mode: adversarial` | Unified evolution modes |
| `gauntlet_enabled` | `enable_gauntlet` | Consistent naming |
| `llm_temp` | `temperature` | Shorter, more common |

### 2.2 Structural Changes

#### Before (v1.0)

```python
# Separate config classes
config = EvolutionConfig(
    max_generations=100,
    pop_size=50
)

pes_config = PESConfig(
    enable_planning=True,
    enable_memory=True
)

qd_config = QDConfig(
    feature_dimensions=['complexity', 'diversity'],
    grid_resolution=10
)
```

#### After (v2.0)

```yaml
# Single unified config
evolution_mode: pes  # or qd, mo, etc.
max_iterations: 100
population_size: 50

# PES parameters (if evolution_mode: pes)
enable_planning: true
enable_memory: true

# QD parameters (if evolution_mode: qd)
feature_dimensions:
  - complexity
  - diversity
grid_resolution: 10
```

### 2.3 Default Value Changes

| Parameter | Old Default | New Default | Reason |
|-----------|-------------|-------------|---------|
| `max_iterations` | 50 | 100 | Better default for most problems |
| `population_size` | 50 | 100 | Better exploration |
| `temperature` | 0.5 | 0.7 | More creativity |
| `log_level` | WARNING | INFO | Better visibility |

---

## 3. Migration Steps

### Step 1: Backup Existing Configuration

```bash
# Backup all config files
cp evolve.config.yaml evolve.config.yaml.backup
cp .env .env.backup

# If using multiple configs
mkdir -p backup
cp *.yaml backup/
cp *.json backup/
```

### Step 2: Install New Version

```bash
# Uninstall old version
pip uninstall openevolve -y

# Install new version
pip install openevolve>=2.0.0

# Verify installation
evolve info version
```

### Step 3: Automatic Migration

```bash
# Run migration tool
evolve migrate --from 1.0 --to 2.0

# The tool will:
# - Rename parameters
# - Update structure
# - Add new defaults
# - Validate new config
# - Create backup
```

### Step 4: Manual Migration (If Needed)

#### Option A: Manual Rename

**Old Config:**
```yaml
max_generations: 100
pop_size: 50
enable_qd: true
feature_dimensions:
  - complexity
  - diversity
```

**New Config:**
```yaml
evolution_mode: qd
max_iterations: 100
population_size: 50
feature_dimensions:
  - complexity
  - diversity
```

#### Option B: Use Migration Script

```python
#!/usr/bin/env python3
"""Manual migration script"""

import yaml
from pathlib import Path

# Parameter mapping
PARAMETER_MAP = {
    'max_generations': 'max_iterations',
    'pop_size': 'population_size',
    'num_generations': 'max_iterations',
    'enable_qd': 'evolution_mode',
    'enable_pes': 'evolution_mode',
    'enable_mo': 'evolution_mode',
    'enable_adversarial': 'evolution_mode',
    'gauntlet_enabled': 'enable_gauntlet',
    'llm_temp': 'temperature',
}

def migrate_config(old_config_path: str, new_config_path: str):
    """Migrate old config to new format"""
    with open(old_config_path) as f:
        config = yaml.safe_load(f)

    # Rename parameters
    for old_name, new_name in PARAMETER_MAP.items():
        if old_name in config:
            # Handle evolution_mode specially
            if new_name == 'evolution_mode':
                if config[old_name] is True:
                    # Determine mode from parameter name
                    if old_name == 'enable_qd':
                        config['evolution_mode'] = 'qd'
                    elif old_name == 'enable_pes':
                        config['evolution_mode'] = 'pes'
                    elif old_name == 'enable_mo':
                        config['evolution_mode'] = 'mo'
                    elif old_name == 'enable_adversarial':
                        config['evolution_mode'] = 'adversarial'
                del config[old_name]
            else:
                # Simple rename
                config[new_name] = config.pop(old_name)

    # Add new defaults
    config.setdefault('evolution_mode', 'auto')
    config.setdefault('log_level', 'INFO')

    # Write new config
    with open(new_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Migrated {old_config_path} -> {new_config_path}")

# Usage
migrate_config('evolve.config.yaml.backup', 'evolve.config.yaml')
```

### Step 5: Validate New Configuration

```bash
# Validate config
evolve config validate evolve.config.yaml

# Test with small run
evolve --config evolve.config.yaml --max-evaluations 5 \
  problem="Test problem"
```

### Step 6: Deploy

```bash
# If validation passes, deploy
git add evolve.config.yaml
git commit -m "Migrate to v2.0 configuration"

# Or for production
cp evolve.config.yaml /etc/evolve/config.yaml
```

---

## 4. Common Migrations

### 4.1 Migration from Pure OpenEvolve

#### Before (v1.0)

```python
from openevolve import QDOptimizer, QDConfig

config = QDConfig(
    grid_resolution=10,
    feature_dimensions=["risk", "return"]
)

optimizer = QDOptimizer(config=config)
result = optimizer.run(problem)
```

#### After (v2.0)

**Option A: Use Unified API**
```python
from openevolve import evolve

result = await evolve(
    problem=problem,
    evolution_mode="qd",
    feature_dimensions=["risk", "return"],
    grid_resolution=10
)
```

**Option B: Use Config File**
```yaml
# evolve.config.yaml
evolution_mode: qd
feature_dimensions:
  - risk
  - return
grid_resolution: 10
```

```python
from openevolve import evolve

result = await evolve(problem=problem)
```

### 4.2 Migration from Pure LoongFlow

#### Before (v1.0)

```python
from loongflow.agents.general_agent import PESAgent
from loongflow.config import PESConfig

config = PESConfig(
    max_iterations=50,
    enable_planning=True
)

agent = PESAgent(config=config)
result = agent.run(problem)
```

#### After (v2.0)

**Option A: Use Unified API**
```python
from openevolve import evolve

result = await evolve(
    problem=problem,
    evolution_mode="pes",
    max_iterations=50,
    enable_planning=True
)
```

**Option B: Use Config File**
```yaml
# evolve.config.yaml
evolution_mode: pes
max_iterations: 50
enable_planning: true
```

```python
from openevolve import evolve

result = await evolve(problem=problem)
```

### 4.3 Migration from Multi-Config Setup

#### Before (v1.0)

```python
# Multiple config files
evo_config = EvolutionConfig(max_generations=100)
pes_config = PESConfig(enable_planning=True)
qd_config = QDConfig(grid_resolution=10)

# Manual integration
config = {
    'evolution': evo_config,
    'pes': pes_config,
    'qd': qd_config
}
```

#### After (v2.0)

```yaml
# Single config file
evolution_mode: pes
max_iterations: 100
enable_planning: true
grid_resolution: 10
```

### 4.4 Migration from Environment-Only Config

#### Before (v1.0)

```bash
export OPENEVOLVE_MAX_GENERATIONS=100
export OPENEVOLVE_POP_SIZE=50
export OPENEVOLVE_ENABLE_PES=true
```

#### After (v2.0)

```bash
# New environment variable names
export EVOLVE_MAX_ITERATIONS=100
export EVOLVE_POPULATION_SIZE=50
export EVOLVE_EVOLUTION_MODE=pes
```

**Or use config file:**
```yaml
max_iterations: 100
population_size: 50
evolution_mode: pes
```

### 4.5 Migration from Custom Configuration Class

#### Before (v1.0)

```python
class MyConfig:
    def __init__(self):
        self.max_generations = 100
        self.pop_size = 50
        self.enable_qd = True

config = MyConfig()
```

#### After (v2.0)

**Option A: Use Config File**
```yaml
# evolve.config.yaml
max_iterations: 100
population_size: 50
evolution_mode: qd
```

**Option B: Use Unified Configuration**
```python
from openevolve.config import UnifiedEvolutionConfig

config = UnifiedEvolutionConfig(
    max_iterations=100,
    population_size=50,
    evolution_mode="qd"
)
```

---

## 5. Rollback Plan

### Step 1: Feature Flags

```python
# Add feature flag to your code
import os

USE_UNIFIED_CONFIG = os.getenv('USE_UNIFIED_CONFIG', 'false') == 'true'

if USE_UNIFIED_CONFIG:
    from openevolve import evolve
    result = await evolve(problem=problem)
else:
    from openevolve_v1 import QDOptimizer
    result = QDOptimizer().run(problem)
```

**Enable unified config:**
```bash
export USE_UNIFIED_CONFIG=true
```

### Step 2: Gradual Migration

```python
# Migrate non-critical problems first
NON_CRITICAL_PROBLEMS = [
    "exploration_problem",
    "test_problem"
]

CRITICAL_PROBLEMS = [
    "production_problem"
]

for problem in NON_CRITICAL_PROBLEMS:
    # Use new unified config
    result = await evolve(problem=problem)

for problem in CRITICAL_PROBLEMS:
    # Keep old config for now
    result = old_optimizer.run(problem)
```

### Step 3: A/B Testing

```python
# Test both systems in parallel
import asyncio

async def compare_configs(problem):
    # Old config
    old_result = await asyncio.create_task(
        old_optimizer.run(problem)
    )

    # New config
    new_result = await evolve(
        problem=problem,
        evolution_mode="qd"
    )

    # Compare results
    assert new_result['fitness'] >= old_result['fitness']

    return old_result, new_result
```

### Step 4: Rollback Procedure

```bash
# If issues occur, rollback:
git checkout evolve.config.yaml.backup
cp evolve.config.yaml.backup evolve.config.yaml

# Or reinstall old version
pip uninstall openevolve -y
pip install openevolve==1.0.0
```

---

## 6. Troubleshooting

### Issue 1: Unknown Parameter Error

**Error:**
```
ConfigurationValidationError: Unknown parameter 'max_generations'
```

**Solution:**
Parameter was renamed in v2.0. Use `max_iterations` instead.

```yaml
# Old
max_generations: 100

# New
max_iterations: 100
```

### Issue 2: Evolution Mode Not Recognized

**Error:**
```
ConfigurationValidationError: Invalid value for enable_qd
```

**Solution:**
Evolution modes changed in v2.0. Use `evolution_mode` parameter.

```yaml
# Old
enable_qd: true

# New
evolution_mode: qd
```

### Issue 3: Missing Defaults

**Error:**
```
ConfigurationValidationError: Missing required parameter 'log_level'
```

**Solution:**
Add missing parameter or use defaults.

```yaml
# Add explicitly
log_level: INFO

# Or rely on defaults (remove from config)
```

### Issue 4: Validation Fails

**Error:**
```
ConfigurationValidationError: Value out of range
```

**Solution:**
Check parameter ranges in new version.

```bash
# Show valid range
evolve config param max_iterations

# Update value
max_iterations: 100  # Must be 1-10000
```

### Issue 5: Config Not Loading

**Error:**
Configuration not being applied

**Solution:**
Check config file location and name.

```bash
# Must be exactly:
# - ./evolve.config.yaml (or .json, .toml)
# - Or specify with --config

evolve --config my-config.yaml problem="..."
```

---

## Migration Checklist

### Pre-Migration

- [ ] Backup all configuration files
- [ ] Document current configuration
- [ ] Test migration in development environment
- [ ] Plan rollback strategy

### Migration

- [ ] Install v2.0
- [ ] Run automatic migration tool
- [ ] Manually update any remaining configs
- [ ] Validate new configurations
- [ ] Test with small problems

### Post-Migration

- [ ] Monitor for issues
- [ ] Compare results with old system
- [ ] Update documentation
- [ ] Train team on new system
- [ ] Remove old configs after validation period

---

## Quick Reference

### Migration Commands

```bash
# Automatic migration
evolve migrate --from 1.0 --to 2.0

# Validate migration
evolve config validate evolve.config.yaml

# Test migration
evolve --max-evaluations 5 problem="..."

# Compare configs
evolve config diff old.yaml new.yaml

# Show migration guide
evolve migrate --help
```

### Parameter Mapping

```bash
# Show parameter mapping
evolve migrate --show-mapping

# Check specific parameter
evolve config max_generations --show-mapped
```

---

**End of Migration Guide**

For more information:
- [Configuration Guide](CONFIGURATION_GUIDE.md) - Master configuration guide
- [Configuration Reference](CONFIGURATION_REFERENCE.md) - Complete parameter reference
- [CLI Reference](CLI_REFERENCE.md) - CLI documentation
