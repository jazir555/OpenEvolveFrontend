# Optional LoongFlow Configuration Guide

## Overview

The Unified Evolution System now supports **optional LoongFlow integration**. This means you can:

- Use LoongFlow PES mode when available
- Gracefully fallback to OpenEvolve modes when LoongFlow is not installed
- Explicitly require LoongFlow for your workflows
- Use OpenEvolve-only configurations

## Configuration Parameters

### 1. `enable_loongflow` (Default: `True`)

Controls whether LoongFlow PES system can be used.

```python
config = UnifiedEvolutionConfig(
    enable_loongflow=True  # Try to use LoongFlow
)
```

When `False`, only OpenEvolve modes (QD, MO, Standard) will be used.

### 2. `loongflow_fallback_enabled` (Default: `True`)

Allows graceful fallback to OpenEvolve if LoongFlow is unavailable.

```python
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=True  # Fallback if not installed
)
```

When `True` and LoongFlow is not installed, the system will:
- Log a warning message
- Continue with OpenEvolve modes
- Not interrupt your workflow

### 3. `require_loongflow` (Default: `False`)

Requires LoongFlow to be available. Raises an error if not installed.

```python
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    require_loongflow=True  # Error if not installed
)
```

When `True`, the system will:
- Raise `RuntimeError` if LoongFlow is not available
- Prevent accidental execution without LoongFlow
- Ensure PES mode requirements are met

## Usage Patterns

### Pattern 1: Default Behavior (Recommended)

```python
from openevolve.unified.config import UnifiedEvolutionConfig, DomainType

config = UnifiedEvolutionConfig(
    domain=DomainType.FINANCE,
    max_iterations=1000
)

# LoongFlow enabled with fallback
# Will use LoongFlow if available, otherwise OpenEvolve
```

**Best for:** General use cases where you want the best available system.

### Pattern 2: OpenEvolve Only

```python
config = UnifiedEvolutionConfig.openevolve_only(
    max_iterations=500,
    domain=DomainType.TRADING
)

# Explicitly disable LoongFlow
# Use only OpenEvolve modes (QD, MO, Standard)
```

**Best for:** Environments without LoongFlow or when you prefer OpenEvolve.

### Pattern 3: Require LoongFlow

```python
config = UnifiedEvolutionConfig.loongflow_required(
    domain=DomainType.ENGINEERING,
    evolution_mode=EvolutionMode.PES
)

# Must have LoongFlow installed
# Will raise error if not available
```

**Best for:** Production systems that depend on LoongFlow features.

### Pattern 4: Explicit Configuration

```python
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=False,
    require_loongflow=False,
    domain=DomainType.SCIENCE
)

# Enable LoongFlow
# No fallback (use OpenEvolve if LoongFlow unavailable)
# Don't require it (no error if unavailable)
```

**Best for:** Fine-grained control over behavior.

## Helper Methods

### `is_loongflow_enabled()`

Check if LoongFlow is enabled in the configuration.

```python
config = UnifiedEvolutionConfig()
if config.is_loongflow_enabled():
    print("LoongFlow is enabled")
```

### `should_use_loongflow()`

Check if LoongFlow should be used considering availability.

```python
config = UnifiedEvolutionConfig()
if config.should_use_loongflow():
    print("Will use LoongFlow for evolution")
else:
    print("Will use OpenEvolve modes")
```

This method:
- Returns `True` if LoongFlow is enabled and available
- Returns `False` if LoongFlow is disabled or unavailable
- Raises `RuntimeError` if `require_loongflow=True` but unavailable
- Logs warnings when falling back due to unavailability

### `_check_loongflow_availability()`

Check if LoongFlow package is installed.

```python
config = UnifiedEvolutionConfig()
if config._check_loongflow_availability():
    print("LoongFlow package is installed")
else:
    print("LoongFlow package is not available")
```

## Configuration Validation

The system validates configuration consistency:

```python
# This will raise ValueError
config = UnifiedEvolutionConfig(
    enable_loongflow=False,
    require_loongflow=True  # Contradictory!
)
```

**Error:**
```
ValueError: require_loongflow=True but enable_loongflow=False is contradictory.
Either set enable_loongflow=True or require_loongflow=False
```

## Evolution Mode Selection

### PES Mode (LoongFlow)

```python
config = UnifiedEvolutionConfig(
    evolution_mode=EvolutionMode.PES,
    pes=PESConfig(enabled=True),
    enable_loongflow=True,
    require_loongflow=True
)

if config.should_use_loongflow():
    # Will use LoongFlow PES mode
    pass
```

### QD Mode (OpenEvolve)

```python
config = UnifiedEvolutionConfig.openevolve_only(
    evolution_mode=EvolutionMode.QD,
    qd=QDConfig(enabled=True)
)

# Will use OpenEvolve MAP-Elites
```

### Auto Mode

```python
config = UnifiedEvolutionConfig(
    evolution_mode=EvolutionMode.AUTO,
    qd=QDConfig(enabled=True),
    enable_loongflow=False
)

# Will select QD mode (OpenEvolve)
```

## Error Handling

### Graceful Fallback

When `loongflow_fallback_enabled=True`:

```python
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=True
)

if config.should_use_loongflow():
    # Use LoongFlow
    pass
else:
    # Fallback to OpenEvolve
    # Warning logged: "LoongFlow is enabled but not available.
    #                 Falling back to OpenEvolve modes"
    pass
```

### Strict Requirement

When `require_loongflow=True`:

```python
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    require_loongflow=True
)

try:
    if config.should_use_loongflow():
        # Use LoongFlow
        pass
except RuntimeError as e:
    # Handle error: "require_loongflow=True but LoongFlow is not available"
    print(f"Error: {e}")
    print("Please install LoongFlow to continue")
```

## Domain-Specific Configurations

### Finance Domain

```python
config = UnifiedEvolutionConfig.openevolve_only(
    domain=DomainType.FINANCE,
    evolution_mode=EvolutionMode.QD,
    max_iterations=10000
)

# Use OpenEvolve QD for financial optimization
```

### Science Domain

```python
config = UnifiedEvolutionConfig.loongflow_required(
    domain=DomainType.SCIENCE,
    evolution_mode=EvolutionMode.PES,
    pes=PESConfig(
        enabled=True,
        max_rounds=10,
        enable_planning=True
    )
)

# Require LoongFlow PES for scientific discovery
```

### Engineering Domain

```python
config = UnifiedEvolutionConfig(
    domain=DomainType.ENGINEERING,
    enable_loongflow=True,
    loongflow_fallback_enabled=True,
    evolution_mode=EvolutionMode.MO
)

# Try LoongFlow, fallback to OpenEvolve MO
```

## Migration Guide

### From OpenEvolve-Only

```python
# Before (OpenEvolve only)
from openevolve.config import OpenEvolveConfig
config = OpenEvolveConfig(max_iterations=1000)

# After (Unified config with LoongFlow optional)
from openevolve.unified.config import UnifiedEvolutionConfig
config = UnifiedEvolutionConfig.openevolve_only(
    max_iterations=1000
)
```

### From LoongFlow-Only

```python
# Before (LoongFlow only)
from loongflow.config import LoongFlowConfig
config = LoongFlowConfig(max_rounds=5)

# After (Unified config requiring LoongFlow)
from openevolve.unified.config import UnifiedEvolutionConfig
config = UnifiedEvolutionConfig.loongflow_required(
    evolution_mode=EvolutionMode.PES,
    pes=PESConfig(enabled=True, max_rounds=5)
)
```

### From Mixed Usage

```python
# Before (Manual checks)
if loongflow_available:
    config = LoongFlowConfig()
else:
    config = OpenEvolveConfig()

# After (Automatic fallback)
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=True
)
```

## Best Practices

### 1. Development Environment

```python
# Use fallback during development
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=True
)
```

### 2. Production Environment

```python
# Require LoongFlow in production if you depend on it
config = UnifiedEvolutionConfig.loongflow_required(
    domain=DomainType.FINANCE
)
```

### 3. Testing Environment

```python
# Explicitly disable for consistent tests
config = UnifiedEvolutionConfig.openevolve_only(
    max_iterations=100
)
```

### 4. CI/CD Pipeline

```python
# Check availability before execution
config = UnifiedEvolutionConfig()
if config.should_use_loongflow():
    print("Using LoongFlow")
else:
    print("Using OpenEvolve")
```

## Troubleshooting

### Issue: ImportError when using LoongFlow

**Solution:** Install LoongFlow
```bash
pip install loongflow
```

### Issue: RuntimeError "require_loongflow=True but LoongFlow is not available"

**Solution:** Either:
1. Install LoongFlow: `pip install loongflow`
2. Set `require_loongflow=False`
3. Use `openevolve_only()` configuration

### Issue: Unexpected fallback to OpenEvolve

**Solution:** Check settings:
```python
config = UnifiedEvolutionConfig()
print(f"Enabled: {config.enable_loongflow}")
print(f"Available: {config._check_loongflow_availability()}")
print(f"Should use: {config.should_use_loongflow()}")
```

## Performance Considerations

### Availability Checking

The `should_use_loongflow()` method performs an import check:
- **Cost:** Minimal (cached by Python import system)
- **Frequency:** Call once at startup, cache result
- **Recommendation:** Store result in a variable

```python
# Good
use_loongflow = config.should_use_loongflow()
if use_loongflow:
    # Use LoongFlow
    pass

# Avoid repeated calls
if config.should_use_loongflow():  # OK
    pass
if config.should_use_loongflow():  # Unnecessary
    pass
```

## Summary

The optional LoongFlow configuration provides:

✅ **Flexibility:** Use LoongFlow when available, OpenEvolve otherwise
✅ **Control:** Explicitly enable/disable/fallback as needed
✅ **Safety:** Validation prevents contradictory settings
✅ **Convenience:** Helper methods for common patterns
✅ **Compatibility:** Works with existing code

Choose the pattern that best fits your use case!
