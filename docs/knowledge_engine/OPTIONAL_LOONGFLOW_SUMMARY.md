# Optional LoongFlow Documentation Summary

## Overview

This document provides a summary of the comprehensive documentation for optional LoongFlow usage and graceful fallback in the Unified Evolution API.

## Document Suite

### 1. Optional LoongFlow Usage Guide
**File:** `OPTIONAL_LOONGFLOW_GUIDE.md`
**Audience:** Users and developers
**Purpose:** Complete user guide for optional LoongFlow usage

**Contents:**
- Why make LoongFlow optional (6 valid use cases)
- How to disable LoongFlow (5 methods)
- Configuration options explained
- How fallback works (with decision tree)
- Capabilities comparison table
- OpenEvolve-only recommendations by domain
- 7 detailed examples
- Troubleshooting guide (4 common issues)
- Best practices (5 scenarios)
- Migration guide
- FAQ (14 common questions)

**Key Sections:**
```markdown
## Why Make LoongFlow Optional?
1. Dependency Management
2. Cost Optimization
3. Simplicity
4. Testing
5. Compliance
6. Debugging

## How to Disable LoongFlow
### Method 1: Configuration Parameter
### Method 2: Runtime Override
### Method 3: Global Configuration File
### Method 4: Convenience Function
### Method 5: Environment Variable
```

---

### 2. Fallback Mechanism Documentation
**File:** `FALLBACK_DOCUMENTATION.md`
**Audience:** Developers and system architects
**Purpose:** Technical documentation on graceful fallback implementation

**Contents:**
- Architecture diagrams
- Complete decision tree
- Implementation details (code snippets)
- Custom exceptions
- Error recovery strategies
- Logging and monitoring
- Testing strategies (unit, integration, performance)
- Configuration matrix (12 combinations)
- Best practices (5 guidelines)
- Troubleshooting

**Key Sections:**
```markdown
## Architecture
┌─────────────────────────────────────────┐
│         Unified Evolution API           │
└─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌──────────────┐        ┌──────────────┐
│ Use LoongFlow│        │ Use OpenEvolve│
└──────────────┘        └──────────────┘

## Decision Tree
Is enable_loongflow == True?
├─ NO → USE_OPENEVOLVE_ONLY
└─ YES → Check LOONGFLOW_AVAILABLE
    ├─ AVAILABLE → USE_LOONGFLOW
    └─ NOT_AVAILABLE → Check require_loongflow
        ├─ TRUE → RAISE_ERROR
        └─ FALSE → Check loongflow_fallback_enabled
```

---

### 3. Configuration Options Reference
**File:** `CONFIGURATION_OPTIONS.md`
**Audience:** Users and developers
**Purpose:** Complete reference of all LoongFlow-related configuration options

**Contents:**
- Core LoongFlow control parameters (4 options)
- Runtime override
- PES mode configuration
- Configuration combinations (6 scenarios)
- Configuration precedence (5 levels)
- Configuration validation
- Domain-specific examples (7 domains)
- Configuration files (YAML, environment variables)
- Best practices (5 guidelines)
- Migration guide

**Key Sections:**
```markdown
## Core Parameters
- enable_loongflow (bool, default: true)
- loongflow_fallback_enabled (bool, default: true)
- require_loongflow (bool, default: false)
- use_loongflow (runtime override)

## Recommended Configurations
1. Development: Fast Iteration
2. Production: Graceful Degradation
3. Production: Strict LoongFlow Requirement
4. OpenEvolve-Only: No Dependencies
5. Testing: Compare Both Systems
6. Cost-Optimized: Budget-Constrained
```

---

## Quick Reference

### Configuration Decision Matrix

| Scenario | `enable_loongflow` | `loongflow_fallback_enabled` | `require_loongflow` | Result |
|----------|-------------------|------------------------------|-------------------|--------|
| **Development** | `false` | N/A | N/A | OpenEvolve-only (fast) |
| **Production (resilient)** | `true` | `true` | `false` | LoongFlow with fallback |
| **Production (strict)** | `true` | `false` | `true` | Must have LoongFlow |
| **Expensive evals** | `true` | `true` | `false` | Prefer LoongFlow (60% reduction) |
| **Cheap evals** | `false` | N/A | N/A | OpenEvolve is sufficient |
| **Testing** | Test both | N/A | N/A | Compare performance |

### Domain Recommendations

| Domain | LoongFlow | OpenEvolve Mode | Rationale |
|--------|-----------|----------------|-----------|
| **Finance** | ✅ Yes | PES | 60% fewer backtests |
| **Trading** | ❌ No | Adversarial | Robustness to regime changes |
| **Science** | ✅ Yes | PES | 60% fewer experiments |
| **Engineering** | ❌ No | MO | Multi-objective optimization |
| **Pharma** | ❌ No | QD | Chemical space exploration |
| **Web** | ❌ No | Standard | Fast evaluations, GA sufficient |
| **General** | ✅ Yes | AUTO | Auto-select based on problem |

### Key Features

| Feature | With LoongFlow | OpenEvolve Only |
|---------|---------------|----------------|
| **Directed search (PES)** | ✅ Yes | ❌ No |
| **60% fewer evaluations** | ✅ Yes | ❌ No |
| **Planning strategies** | ✅ Yes | ❌ No |
| **Quality Diversity (QD)** | ✅ Yes | ✅ Yes |
| **Multi-Objective (MO)** | ✅ Yes | ✅ Yes |
| **Adversarial testing** | ✅ Yes | ✅ Yes |
| **MAP-Elites archive** | ✅ Yes | ✅ Yes |
| **3-round gauntlet** | ✅ Yes | ✅ Yes |
| **Knowledge extraction** | ✅ Yes | ✅ Yes |

---

## Usage Examples

### Example 1: Quick Start (OpenEvolve-Only)

```python
from openevolve.unified import evolve, evolve_openevolve_only

# Option 1: Convenience function
result = await evolve_openevolve_only(
    problem="Optimize function",
    domain="general"
)

# Option 2: Runtime override
result = await evolve(
    problem="Optimize function",
    domain="general",
    use_loongflow=False
)

# Option 3: Configuration
from openevolve.unified.config import UnifiedEvolutionConfig

config = UnifiedEvolutionConfig(
    enable_loongflow=False
)

result = await evolve(
    problem="Optimize function",
    domain="general",
    config=config
)
```

---

### Example 2: Production (Graceful Fallback)

```python
from openevolve.unified import evolve, UnifiedEvolutionConfig
from openevolve.unified.config import EvolutionMode

config = UnifiedEvolutionConfig(
    # Enable LoongFlow with fallback
    enable_loongflow=True,
    loongflow_fallback_enabled=True,
    require_loongflow=False,

    # Auto-select mode
    evolution_mode=EvolutionMode.AUTO
)

result = await evolve(
    problem="Maximize portfolio Sharpe ratio",
    domain="finance",
    config=config
)

print(f"System: {result.strategy_used.system}")  # 'loongflow' or 'openevolve'
print(f"Mode: {result.strategy_used.mode}")      # 'pes', 'qd', 'mo', etc.
```

---

### Example 3: Strict LoongFlow Requirement

```python
config = UnifiedEvolutionConfig(
    # Require LoongFlow
    enable_loongflow=True,
    require_loongflow=True,
    loongflow_fallback_enabled=False,

    # Force PES mode
    evolution_mode=EvolutionMode.PES
)

try:
    result = await evolve(
        problem="Optimize reaction conditions",
        domain="science",
        config=config
    )
except ImportError:
    print("LoongFlow required but not installed")
    print("Install with: pip install loongflow")
```

---

### Example 4: Compare Both Systems

```python
async def compare_systems(problem: str, domain: str):
    """Compare LoongFlow vs OpenEvolve"""

    # Run with LoongFlow
    result_lf = await evolve(
        problem=problem,
        domain=domain,
        use_loongflow=True
    )

    # Run with OpenEvolve only
    result_oe = await evolve(
        problem=problem,
        domain=domain,
        use_loongflow=False
    )

    # Compare
    print(f"LoongFlow:")
    print(f"  Score: {result_lf.final_score:.4f}")
    print(f"  Evaluations: {result_lf.evaluations}")
    print(f"  Time: {result_lf.total_time:.2f}s")

    print(f"\nOpenEvolve:")
    print(f"  Score: {result_oe.final_score:.4f}")
    print(f"  Evaluations: {result_oe.evaluations}")
    print(f"  Time: {result_oe.total_time:.2f}s")

    return result_lf, result_oe

# Usage
result_lf, result_oe = await compare_systems(
    problem="Maximize Sharpe ratio",
    domain="finance"
)
```

---

## Installation

### With LoongFlow (Recommended for Production)

```bash
# Install OpenEvolve with LoongFlow
pip install openevolve[unified]
pip install loongflow

# Or install from source
pip install git+https://github.com/baidu-baige/LoongFlow.git
```

### Without LoongFlow (OpenEvolve-Only)

```bash
# Install OpenEvolve only
pip install openevolve[unified]

# No LoongFlow dependency needed
```

---

## Configuration Files

### YAML Configuration

```yaml
# evolve.config.yaml
enable_loongflow: true
loongflow_fallback_enabled: true
require_loongflow: false
evolution_mode: AUTO

domain: finance
max_iterations: 100

# PES configuration
pes:
  enabled: true
  enable_planning: true
  enable_memory: true
  max_rounds: 5
```

**Load configuration:**
```python
import yaml

with open("evolve.config.yaml") as f:
    config_dict = yaml.safe_load(f)
    config = UnifiedEvolutionConfig(**config_dict)

result = await evolve(
    problem="Optimize portfolio",
    config=config
)
```

---

### Environment Variables

```bash
# .env file
EVOLVE_ENABLE_LOONGFLOW=true
EVOLVE_LOONGFLOW_FALLBACK_ENABLED=true
EVOLVE_REQUIRE_LOONGFLOW=false
EVOLVE_DOMAIN=finance
EVOLVE_MAX_ITERATIONS=100
```

**Load configuration:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

config = UnifiedEvolutionConfig(
    enable_loongflow=os.getenv('EVOLVE_ENABLE_LOONGFLOW', 'true').lower() == 'true',
    loongflow_fallback_enabled=os.getenv('EVOLVE_LOONGFLOW_FALLBACK_ENABLED', 'true').lower() == 'true',
    max_iterations=int(os.getenv('EVOLVE_MAX_ITERATIONS', '100'))
)
```

---

## Migration Guide

### From LoongFlow-Dependent

**Before:**
```python
from loongflow import LoongFlowEvolve
from loongflow.config import PESConfig

config = PESConfig(enable_planning=True)
evolver = LoongFlowEvolve(config=config)
result = evolver.evolve(problem="...")
```

**After:**
```python
from openevolve.unified import evolve, UnifiedEvolutionConfig
from openevolve.unified.config import EvolutionMode

config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    evolution_mode=EvolutionMode.PES
)

result = await evolve(problem="...", config=config)
```

---

### From OpenEvolve-Only

**Before:**
```python
from openevolve import OpenEvolve
from openevolve.config import QDConfig

config = QDConfig(archive_size=1000)
evolver = OpenEvolve(config=config)
result = evolver.evolve(problem="...")
```

**After:**
```python
from openevolve.unified import evolve, UnifiedEvolutionConfig
from openevolve.unified.config import EvolutionMode

config = UnifiedEvolutionConfig(
    enable_loongflow=False,
    evolution_mode=EvolutionMode.QD
)

result = await evolve(problem="...", config=config)
```

---

## Troubleshooting

### Issue: "LoongFlow not available" Warning

**Cause:** LoongFlow package not installed

**Solutions:**
1. Install LoongFlow: `pip install loongflow`
2. Disable requirement: Set `require_loongflow=False`
3. Allow fallback: Set `loongflow_fallback_enabled=True`

---

### Issue: Poor Performance in OpenEvolve-Only Mode

**Cause:** Problem benefits from LoongFlow's directed search

**Solutions:**
1. Enable LoongFlow: Set `enable_loongflow=True`
2. Increase iterations: Compensate for lack of directed search
3. Use QD mode: Get diversity benefits

---

### Issue: Import Error When Using PES Mode

**Cause:** Trying to use PES without LoongFlow installed

**Solutions:**
1. Install LoongFlow: `pip install loongflow`
2. Use OpenEvolve mode: Set `evolution_mode=EvolutionMode.QD`
3. Allow fallback: Set `loongflow_fallback_enabled=True`

---

## Best Practices

### 1. Development Environment

```python
config = UnifiedEvolutionConfig(
    enable_loongflow=False,  # Fast iteration
    max_iterations=20,
    verbose=True
)
```

### 2. Production Environment

```python
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=True,  # Graceful degradation
    evolution_mode=EvolutionMode.AUTO
)
```

### 3. Testing Environment

```python
# Test both modes
result_lf = await evolve(problem, domain, use_loongflow=True)
result_oe = await evolve(problem, domain, use_loongflow=False)

# Compare results
assert result_lf.final_score > 0
assert result_oe.final_score > 0
```

### 4. Deployment

```bash
# Environment variable for deployment
export EVOLVE_ENABLE_LOONGFLOW=true
```

### 5. Cost Optimization

```python
def get_config(budget: float, eval_cost: float):
    max_evals = budget / eval_cost

    if max_evals < 50:
        # Tight budget, require LoongFlow
        return UnifiedEvolutionConfig(
            enable_loongflow=True,
            require_loongflow=True
        )
    else:
        # Generous budget, OpenEvolve fine
        return UnifiedEvolutionConfig(
            enable_loongflow=False,
            max_iterations=200
        )
```

---

## Key Takeaways

1. **LoongFlow is Optional**: Use it when beneficial, skip when not
2. **Graceful Fallback**: System automatically degrades if LoongFlow unavailable
3. **Configuration Control**: Fine-grained control over LoongFlow usage
4. **OpenEvolve-Only**: Fully functional without LoongFlow
5. **Easy to Switch**: Runtime overrides, environment variables, config files
6. **Performance Awareness**: Use LoongFlow for expensive evaluations

---

## Document Files

1. **OPTIONAL_LOONGFLOW_GUIDE.md** (800+ lines)
   - Complete user guide
   - Examples and use cases
   - Troubleshooting and FAQ

2. **FALLBACK_DOCUMENTATION.md** (600+ lines)
   - Technical implementation
   - Architecture and decision trees
   - Testing and monitoring

3. **CONFIGURATION_OPTIONS.md** (500+ lines)
   - Configuration reference
   - Domain-specific examples
   - Best practices

4. **OPTIONAL_LOONGFLOW_SUMMARY.md** (This document)
   - Quick reference
   - Key points
   - Getting started guide

---

## Next Steps

1. **Read the guides**:
   - Start with `OPTIONAL_LOONGFLOW_GUIDE.md` for user guide
   - Read `FALLBACK_DOCUMENTATION.md` for technical details
   - Reference `CONFIGURATION_OPTIONS.md` for configuration

2. **Try the examples**:
   - Run OpenEvolve-only mode for development
   - Test graceful fallback in production
   - Compare performance between modes

3. **Configure your system**:
   - Choose configuration based on your use case
   - Set up environment variables
   - Create configuration files

4. **Monitor and optimize**:
   - Track fallback frequency
   - Compare performance metrics
   - Adjust configuration based on results

---

## Support

For more information:
- [Optional LoongFlow Usage Guide](./OPTIONAL_LOONGFLOW_GUIDE.md)
- [Fallback Mechanism Documentation](./FALLBACK_DOCUMENTATION.md)
- [Configuration Options Reference](./CONFIGURATION_OPTIONS.md)
- [Unified Evolution API](./UNIFIED_EVOLUTION_API.md)
- [Performance Tuning Guide](./PERFORMANCE_TUNING.md)

---

**Status:** ✅ Complete
**Version:** 1.0
**Last Updated:** January 30, 2026
