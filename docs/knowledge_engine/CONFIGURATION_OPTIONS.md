# Configuration Options Reference

## Overview

Complete reference of all LoongFlow-related configuration options in the Unified Evolution API.

## Configuration Parameters

### Core LoongFlow Control

#### `enable_loongflow`

**Type:** `bool`
**Default:** `true`
**Description:** Enable or disable LoongFlow PES system globally

**Impact:**
- When `false`: Only OpenEvolve modes (QD, MO, Adversarial, Standard) are considered
- When `true`: Both LoongFlow PES and OpenEvolve modes are considered

**Use Cases:**
```python
# Disable LoongFlow (OpenEvolve-only mode)
config = UnifiedEvolutionConfig(
    enable_loongflow=False
)

# Enable LoongFlow (default)
config = UnifiedEvolutionConfig(
    enable_loongflow=True
)
```

**Environment Variable:** `EVOLVE_ENABLE_LOONGFLOW`

**When to use:**
- Set to `false` for development, testing, or when LoongFlow is not available
- Set to `true` for production with expensive evaluations

---

#### `loongflow_fallback_enabled`

**Type:** `bool`
**Default:** `true`
**Description:** Allow fallback to OpenEvolve if LoongFlow is unavailable or fails

**Impact:**
- When `true`: Gracefully degrades to OpenEvolve if LoongFlow unavailable
- When `false`: Raises error if LoongFlow unavailable

**Use Cases:**
```python
# Allow fallback (recommended)
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=True
)

# Disable fallback (strict mode)
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=False
)
```

**Environment Variable:** `EVOLVE_LOONGFLOW_FALLBACK_ENABLED`

**When to use:**
- Set to `true` for production (graceful degradation)
- Set to `false` when LoongFlow is strictly required

---

#### `require_loongflow`

**Type:** `bool`
**Default:** `false`
**Description:** Require LoongFlow to be available. If `true` and LoongFlow is unavailable, raise an error instead of falling back

**Impact:**
- When `true`: System fails fast if LoongFlow not available
- When `false`: System continues with OpenEvolve if LoongFlow unavailable

**Use Cases:**
```python
# Strict requirement (fail if unavailable)
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    require_loongflow=True,
    loongflow_fallback_enabled=False
)

# Optional LoongFlow (graceful fallback)
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    require_loongflow=False,
    loongflow_fallback_enabled=True
)
```

**Environment Variable:** `EVOLVE_REQUIRE_LOONGFLOW`

**When to use:**
- Set to `true` when evaluations are very expensive and LoongFlow is essential
- Set to `false` when OpenEvolve-only mode is acceptable

---

### Runtime Override

#### `use_loongflow` (function parameter)

**Type:** `bool` or `None`
**Default:** `None` (use config value)
**Description:** Override config for a specific evolution run

**Impact:**
- Overrides `enable_loongflow` config setting for a single call
- Does not modify configuration object

**Use Cases:**
```python
# Override to use LoongFlow for this run only
result = await evolve(
    problem="Expensive optimization",
    domain="science",
    use_loongflow=True  # Runtime override
)

# Override to use OpenEvolve for this run only
result = await evolve(
    problem="Quick test",
    domain="web",
    use_loongflow=False  # Runtime override
)
```

**When to use:**
- Temporarily enable/disable LoongFlow for specific runs
- A/B testing between LoongFlow and OpenEvolve
- Quick experiments without changing configuration

---

### PES Mode Configuration

#### `evolution_mode`

**Type:** `EvolutionMode` (enum)
**Default:** `EvolutionMode.AUTO`
**Description:** Evolution mode to use

**Values:**
- `AUTO`: Automatically select best mode (default)
- `PES`: Use LoongFlow PES mode (requires LoongFlow)
- `QD`: Use OpenEvolve Quality-Diversity mode
- `MO`: Use OpenEvolve Multi-Objective mode
- `ADVERSARIAL`: Use OpenEvolve Adversarial mode
- `STANDARD`: Use OpenEvolve Standard GA mode

**Use Cases:**
```python
# Auto-select (recommended)
config = UnifiedEvolutionConfig(
    evolution_mode=EvolutionMode.AUTO
)

# Force PES mode (requires LoongFlow)
config = UnifiedEvolutionConfig(
    evolution_mode=EvolutionMode.PES
)

# Force QD mode (OpenEvolve)
config = UnifiedEvolutionConfig(
    evolution_mode=EvolutionMode.QD
)
```

**When to use:**
- Use `AUTO` for automatic selection based on problem characteristics
- Use `PES` for expensive evaluations where LoongFlow's directed search helps
- Use `QD` for diverse solution archives
- Use `MO` for multi-objective optimization
- Use `ADVERSARIAL` for robustness testing
- Use `STANDARD` for basic genetic algorithm

---

#### `pes` (PESConfig)

**Type:** `PESConfig` (object)
**Default:** `PESConfig()` with defaults
**Description:** PES mode specific configuration (LoongFlow)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable PES mode |
| `enable_planning` | `bool` | `true` | Enable planning phase |
| `enable_memory` | `bool` | `true` | Use evolutionary memory |
| `max_rounds` | `int` | `5` | Maximum PES rounds |
| `planning_iterations` | `int` | `10` | Iterations in planning phase |
| `memory_size` | `int` | `1000` | Memory system size |
| `directed_search_ratio` | `float` | `0.7` | Ratio of directed vs random search |

**Use Cases:**
```python
from openevolve.unified.config import PESConfig

# Full PES configuration
config = UnifiedEvolutionConfig(
    evolution_mode=EvolutionMode.PES,
    pes=PESConfig(
        enabled=True,
        enable_planning=True,
        enable_memory=True,
        max_rounds=5,
        planning_iterations=10,
        memory_size=1000,
        directed_search_ratio=0.7
    )
)

# Minimal PES configuration
config = UnifiedEvolutionConfig(
    evolution_mode=EvolutionMode.PES,
    pes=PESConfig(enabled=True)
)
```

**When to use:**
- When using PES mode (LoongFlow)
- Tune planning and memory for specific problems
- Reduce `max_rounds` for faster iterations
- Increase `memory_size` for complex problems

---

## Configuration Combinations

### Recommended Configurations

#### 1. Development: Fast Iteration

```python
config = UnifiedEvolutionConfig(
    # Disable LoongFlow for speed
    enable_loongflow=False,

    # Quick iterations
    max_iterations=20,
    time_limit_seconds=60,

    # Verbose output
    verbose=True
)
```

**Characteristics:**
- Fast iterations (20 iterations)
- No LoongFlow overhead
- Verbose logging
- OpenEvolve-only mode

---

#### 2. Production: Graceful Degradation

```python
config = UnifiedEvolutionConfig(
    # Enable LoongFlow with fallback
    enable_loongflow=True,
    loongflow_fallback_enabled=True,
    require_loongflow=False,

    # Auto-select mode
    evolution_mode=EvolutionMode.AUTO,

    # Production settings
    max_iterations=100,
    time_limit_seconds=600,
    checkpoint_interval=20
)
```

**Characteristics:**
- LoongFlow enabled for expensive evaluations
- Graceful fallback to OpenEvolve
- Automatic mode selection
- Checkpointing for recovery

---

#### 3. Production: Strict LoongFlow Requirement

```python
config = UnifiedEvolutionConfig(
    # Require LoongFlow
    enable_loongflow=True,
    require_loongflow=True,
    loongflow_fallback_enabled=False,

    # Force PES mode
    evolution_mode=EvolutionMode.PES,
    pes=PESConfig(
        enabled=True,
        enable_planning=True,
        enable_memory=True
    )
)
```

**Characteristics:**
- LoongFlow required (fail if unavailable)
- PES mode for expensive evaluations
- No fallback (fail fast)
- Full planning and memory

---

#### 4. OpenEvolve-Only: No Dependencies

```python
config = UnifiedEvolutionConfig(
    # Disable LoongFlow completely
    enable_loongflow=False,

    # Use QD mode for diversity
    evolution_mode=EvolutionMode.QD,
    qd=QDConfig(
        enabled=True,
        archive_size=2000,
        grid_resolution=[20, 20]
    ),

    # Increase iterations to compensate
    max_iterations=200
)
```

**Characteristics:**
- OpenEvolve-only (no LoongFlow dependency)
- QD mode for diverse solutions
- More iterations to compensate
- Large archive for diversity

---

#### 5. Testing: Compare Both Systems

```python
# Configuration 1: LoongFlow
config_lf = UnifiedEvolutionConfig(
    enable_loongflow=True,
    evolution_mode=EvolutionMode.PES,
    pes=PESConfig(enabled=True)
)

# Configuration 2: OpenEvolve
config_oe = UnifiedEvolutionConfig(
    enable_loongflow=False,
    evolution_mode=EvolutionMode.QD,
    qd=QDConfig(enabled=True)
)

# Run comparison
result_lf = await evolve(problem, domain, config=config_lf)
result_oe = await evolve(problem, domain, config=config_oe)
```

**Characteristics:**
- Two configurations for A/B testing
- Compare LoongFlow vs OpenEvolve
- Same problem, different modes
- Measure performance difference

---

#### 6. Cost-Optimized: Budget-Constrained

```python
def get_config_for_budget(budget: float, eval_cost: float) -> UnifiedEvolutionConfig:
    """Generate config based on budget and evaluation cost"""

    max_evals = int(budget / eval_cost)

    if max_evals < 50:
        # Very tight budget, require LoongFlow
        return UnifiedEvolutionConfig(
            enable_loongflow=True,
            require_loongflow=True,
            evolution_mode=EvolutionMode.PES,
            pes=PESConfig(
                enabled=True,
                max_rounds=3,  # Fewer rounds
                planning_iterations=5
            )
        )

    elif max_evals < 200:
        # Moderate budget, prefer LoongFlow
        return UnifiedEvolutionConfig(
            enable_loongflow=True,
            loongflow_fallback_enabled=True,
            evolution_mode=EvolutionMode.AUTO,
            max_iterations=50
        )

    else:
        # Generous budget, OpenEvolve is fine
        return UnifiedEvolutionConfig(
            enable_loongflow=False,
            evolution_mode=EvolutionMode.QD,
            qd=QDConfig(
                enabled=True,
                archive_size=2000
            ),
            max_iterations=200
        )

# Usage
config = get_config_for_budget(budget=100000, eval_cost=5000)
```

**Characteristics:**
- Adaptive configuration based on budget
- Tight budget → Require LoongFlow
- Moderate budget → Prefer LoongFlow
- Generous budget → OpenEvolve is fine

---

## Configuration Precedence

### Parameter Override Order

```
1. Runtime parameters (highest priority)
   ↓
2. Configuration object
   ↓
3. Environment variables
   ↓
4. Configuration file (YAML)
   ↓
5. Default values (lowest priority)
```

### Example

```python
# Default in code: enable_loongflow=True

# Environment variable
# export EVOLVE_ENABLE_LOONGFLOW=false

# Config file (evolve.config.yaml)
# enable_loongflow: true

# Configuration object
config = UnifiedEvolutionConfig(
    enable_loongflow=False
)

# Runtime parameter (highest priority)
result = await evolve(
    problem="...",
    domain="...",
    use_loongflow=True  # Overrides everything
)

# Result: use_loongflow=True (runtime parameter wins)
```

---

## Configuration Validation

### Automatic Validation

The `UnifiedEvolutionConfig` class performs automatic validation:

```python
from pydantic import ValidationError

try:
    # Invalid: max_iterations must be >= 1
    config = UnifiedEvolutionConfig(
        max_iterations=0  # Invalid
    )
except ValidationError as e:
    print(f"Validation error: {e}")

try:
    # Invalid: selection ratios must sum to <= 1.0
    config = UnifiedEvolutionConfig(
        database=DatabaseConfig(
            elite_selection_ratio=0.5,
            exploration_ratio=0.4,
            exploitation_ratio=0.3  # Sum = 1.2 > 1.0
        )
    )
except ValidationError as e:
    print(f"Validation error: {e}")
```

### Custom Validation Rules

#### Rule 1: PES Mode Requires LoongFlow

```python
@validator('evolution_mode')
def validate_pes_mode(cls, v, values):
    """PES mode requires LoongFlow"""
    if v == EvolutionMode.PES and not LOONGFLOW_AVAILABLE:
        raise ValueError(
            "PES mode requires LoongFlow. "
            "Install with: pip install loongflow"
        )
    return v
```

#### Rule 2: Fallback and Require LoongFlow Conflict

```python
@validator('require_loongflow', pre=True, always=True)
def validate_require_vs_fallback(cls, v, values):
    """require_loongflow=True should disable fallback"""
    if v and values.get('loongflow_fallback_enabled', True):
        logger.warning(
            "require_loongflow=True overrides loongflow_fallback_enabled. "
            "System will fail if LoongFlow unavailable."
        )
    return v
```

---

## Configuration Examples by Domain

### Finance Domain

```python
config = UnifiedEvolutionConfig(
    domain=DomainType.FINANCE,

    # Enable LoongFlow for expensive backtests
    enable_loongflow=True,
    loongflow_fallback_enabled=True,

    # Use PES for 60% reduction in evaluations
    evolution_mode=EvolutionMode.PES,
    pes=PESConfig(
        enabled=True,
        enable_planning=True,
        enable_memory=True,
        max_rounds=5
    ),

    # Limit iterations (backtests are expensive)
    max_iterations=50,

    # Multi-objective (return, risk, liquidity)
    mo=MOConfig(
        enabled=True,
        objectives=['return', 'risk', 'liquidity'],
        pareto_front_size=100
    )
)
```

---

### Trading Domain

```python
config = UnifiedEvolutionConfig(
    domain=DomainType.TRADING,

    # Use OpenEvolve adversarial for robustness
    enable_loongflow=False,  # Adversarial works well without LoongFlow

    # Adversarial mode for robustness to regime changes
    evolution_mode=EvolutionMode.ADVERSARIAL,
    adversarial=AdversarialConfig(
        enabled=True,
        num_adversarial_rounds=3,
        robustness_threshold=0.7
    ),

    # Moderate iterations
    max_iterations=100
)
```

---

### Science Domain

```python
config = UnifiedEvolutionConfig(
    domain=DomainType.SCIENCE,

    # Require LoongFlow (experiments are very expensive)
    enable_loongflow=True,
    require_loongflow=True,
    loongflow_fallback_enabled=False,

    # PES mode for 60% reduction in experiments
    evolution_mode=EvolutionMode.PES,
    pes=PESConfig(
        enabled=True,
        enable_planning=True,
        enable_memory=True,
        max_rounds=3,  # Fewer rounds for very expensive experiments
        planning_iterations=15  # More planning to reduce experiments
    ),

    # Very limited iterations
    max_iterations=30
)
```

---

### Engineering Domain

```python
config = UnifiedEvolutionConfig(
    domain=DomainType.ENGINEERING,

    # OpenEvolve MO for design optimization
    enable_loongflow=False,

    # Multi-objective (weight, strength, cost)
    evolution_mode=EvolutionMode.MO,
    mo=MOConfig(
        enabled=True,
        objectives=['weight', 'strength', 'cost'],
        pareto_front_size=50,
        algorithm='nsga2'
    ),

    # Moderate iterations
    max_iterations=80
)
```

---

### Pharma Domain

```python
config = UnifiedEvolutionConfig(
    domain=DomainType.PHARMA,

    # OpenEvolve QD for chemical space exploration
    enable_loongflow=False,

    # QD mode for diverse molecular structures
    evolution_mode=EvolutionMode.QD,
    qd=QDConfig(
        enabled=True,
        archive_size=5000,  # Large archive for chemical diversity
        grid_resolution=[50, 50],  # High resolution
        feature_dimensions=['molecular_weight', 'polarity', 'solubility']
    ),

    # More iterations for exploration
    max_iterations=150
)
```

---

### Web Domain

```python
config = UnifiedEvolutionConfig(
    domain=DomainType.WEB,

    # OpenEvolve standard (evaluations are cheap)
    enable_loongflow=False,

    # Standard GA is sufficient
    evolution_mode=EvolutionMode.STANDARD,

    # Can do many iterations (fast evaluations)
    max_iterations=300,

    # Large population
    database=DatabaseConfig(
        population_size=200
    )
)
```

---

### General Domain

```python
config = UnifiedEvolutionConfig(
    domain=DomainType.GENERAL,

    # Enable LoongFlow if available
    enable_loongflow=True,
    loongflow_fallback_enabled=True,

    # Auto-select mode
    evolution_mode=EvolutionMode.AUTO,

    # Default iterations
    max_iterations=100
)
```

---

## Configuration Files

### YAML Configuration

```yaml
# evolve.config.yaml

# Core settings
enable_loongflow: true
loongflow_fallback_enabled: true
require_loongflow: false
evolution_mode: AUTO

# Domain
domain: finance

# Iterations
max_iterations: 100
checkpoint_interval: 20

# PES configuration
pes:
  enabled: true
  enable_planning: true
  enable_memory: true
  max_rounds: 5

# Multi-objective
mo:
  enabled: true
  objectives:
    - return
    - risk
    - liquidity
  pareto_front_size: 100

# Output
output_dir: ./evolution_output
verbose: true
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
EVOLVE_OUTPUT_DIR=./evolution_output
EVOLVE_VERBOSE=true
```

**Load configuration:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

config = UnifiedEvolutionConfig(
    enable_loongflow=os.getenv('EVOLVE_ENABLE_LOONGFLOW', 'true').lower() == 'true',
    loongflow_fallback_enabled=os.getenv('EVOLVE_LOONGFLOW_FALLBACK_ENABLED', 'true').lower() == 'true',
    domain=os.getenv('EVOLVE_DOMAIN', 'general'),
    max_iterations=int(os.getenv('EVOLVE_MAX_ITERATIONS', '100')),
    output_dir=os.getenv('EVOLVE_OUTPUT_DIR', './evolution_output'),
    verbose=os.getenv('EVOLVE_VERBOSE', 'true').lower() == 'true'
)
```

---

## Configuration Best Practices

### 1. Use Environment Variables for Deployment

```python
import os

config = UnifiedEvolutionConfig(
    enable_loongflow=os.getenv('EVOLVE_ENABLE_LOONGFLOW', 'true').lower() == 'true',
    loongflow_fallback_enabled=os.getenv('EVOLVE_LOONGFLOW_FALLBACK_ENABLED', 'true').lower() == 'true'
)
```

**Benefits:**
- Easy configuration without code changes
- Environment-specific settings
- Feature flag control

---

### 2. Use Configuration Files for Reproducibility

```yaml
# config.yaml
enable_loongflow: true
evolution_mode: PES
max_iterations: 100
```

**Benefits:**
- Version control for configurations
- Reproducible experiments
- Easy sharing

---

### 3. Validate Configuration Before Use

```python
try:
    config = UnifiedEvolutionConfig(**config_dict)
    # Check LoongFlow availability if required
    if config.require_loongflow and not LOONGFLOW_AVAILABLE:
        raise ValueError("LoongFlow required but not available")
    print("Configuration valid")
except ValidationError as e:
    print(f"Invalid configuration: {e}")
```

**Benefits:**
- Catch errors early
- Clear error messages
- Prevent runtime failures

---

### 4. Document Configuration Decisions

```python
"""
Configuration Rationale:

1. enable_loongflow=True: Science domain has very expensive experiments
2. require_loongflow=True: 60% reduction in experiments is essential
3. loongflow_fallback_enabled=False: System should fail fast if LoongFlow unavailable
4. evolution_mode=PES: Directed search is critical for expensive evaluations
5. max_iterations=30: Limited budget for experiments
"""

config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    require_loongflow=True,
    loongflow_fallback_enabled=False,
    evolution_mode=EvolutionMode.PES,
    max_iterations=30
)
```

**Benefits:**
- Future maintainers understand decisions
- Easy to review and revise
- Knowledge transfer

---

### 5. Use Type Hints and IDE Support

```python
from openevolve.unified.config import (
    UnifiedEvolutionConfig,
    EvolutionMode,
    PESConfig,
    QDConfig
)

config: UnifiedEvolutionConfig = UnifiedEvolutionConfig(
    enable_loongflow=True,
    evolution_mode=EvolutionMode.PES,
    pes=PESConfig(
        enabled=True,
        enable_planning=True
    )
)
```

**Benefits:**
- IDE autocomplete
- Type checking
- Fewer errors

---

## Configuration Migration

### From Old API

**Before (old API):**
```python
from loongflow import LoongFlowEvolve
from loongflow.config import PESConfig

config = PESConfig(
    enable_planning=True,
    enable_memory=True,
    max_rounds=5
)

evolver = LoongFlowEvolve(config=config)
result = evolver.evolve(problem="...")
```

**After (unified API):**
```python
from openevolve.unified import evolve, UnifiedEvolutionConfig
from openevolve.unified.config import EvolutionMode

config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    evolution_mode=EvolutionMode.PES,
    pes=PESConfig(
        enabled=True,
        enable_planning=True,
        enable_memory=True,
        max_rounds=5
    )
)

result = await evolve(problem="...", config=config)
```

---

### From OpenEvolve-Only

**Before (old API):**
```python
from openevolve import OpenEvolve
from openevolve.config import QDConfig

config = QDConfig(
    archive_size=1000,
    grid_resolution=[20, 20]
)

evolver = OpenEvolve(config=config)
result = evolver.evolve(problem="...")
```

**After (unified API):**
```python
from openevolve.unified import evolve, UnifiedEvolutionConfig
from openevolve.unified.config import EvolutionMode

config = UnifiedEvolutionConfig(
    enable_loongflow=False,
    evolution_mode=EvolutionMode.QD,
    qd=QDConfig(
        enabled=True,
        archive_size=1000,
        grid_resolution=[20, 20]
    )
)

result = await evolve(problem="...", config=config)
```

---

## Summary Table

### LoongFlow Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_loongflow` | `bool` | `true` | Enable/disable LoongFlow globally |
| `loongflow_fallback_enabled` | `bool` | `true` | Allow fallback to OpenEvolve |
| `require_loongflow` | `bool` | `false` | Require LoongFlow (fail if unavailable) |
| `use_loongflow` | `bool` | `None` | Runtime override |
| `evolution_mode` | `EvolutionMode` | `AUTO` | Evolution mode (PES, QD, MO, etc.) |
| `pes.enabled` | `bool` | `false` | Enable PES mode |
| `pes.enable_planning` | `bool` | `true` | Enable planning phase |
| `pes.enable_memory` | `bool` | `true` | Enable memory system |
| `pes.max_rounds` | `int` | `5` | Maximum PES rounds |

### Decision Matrix

| Scenario | `enable_loongflow` | `loongflow_fallback_enabled` | `require_loongflow` | Expected Behavior |
|----------|-------------------|------------------------------|-------------------|-------------------|
| **Development** | `false` | N/A | N/A | OpenEvolve-only for speed |
| **Production (resilient)** | `true` | `true` | `false` | Use LoongFlow, fallback to OpenEvolve |
| **Production (strict)** | `true` | `false` | `true` | Must have LoongFlow, fail fast |
| **Testing** | Test both | N/A | N/A | Compare both systems |
| **Expensive evals** | `true` | `true` | `false` | Prefer LoongFlow for efficiency |
| **Cheap evals** | `false` | N/A | N/A | OpenEvolve is sufficient |

---

## Next Steps

1. **Choose configuration** based on your use case
2. **Test both modes** (LoongFlow and OpenEvolve-only)
3. **Monitor performance** (evaluations, solution quality, time)
4. **Adjust settings** based on observed results
5. **Document decisions** for future reference

For more information:
- [Optional LoongFlow Usage Guide](./OPTIONAL_LOONGFLOW_GUIDE.md)
- [Fallback Mechanism Documentation](./FALLBACK_DOCUMENTATION.md)
- [Unified Evolution API](./UNIFIED_EVOLUTION_API.md)
