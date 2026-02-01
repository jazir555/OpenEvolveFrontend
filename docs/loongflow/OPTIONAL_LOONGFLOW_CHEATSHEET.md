# Optional LoongFlow - Quick Reference

## Quick Start

```python
from openevolve.unified.config import UnifiedEvolutionConfig, DomainType

# Default: LoongFlow enabled with fallback
config = UnifiedEvolutionConfig(domain=DomainType.FINANCE)

# OpenEvolve only
config = UnifiedEvolutionConfig.openevolve_only(domain=DomainType.TRADING)

# Require LoongFlow
config = UnifiedEvolutionConfig.loongflow_required(domain=DomainType.SCIENCE)
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_loongflow` | bool | `True` | Enable LoongFlow PES system |
| `loongflow_fallback_enabled` | bool | `True` | Allow fallback to OpenEvolve |
| `require_loongflow` | bool | `False` | Require LoongFlow (no fallback) |

## Common Patterns

### OpenEvolve Only
```python
config = UnifiedEvolutionConfig.openevolve_only(
    max_iterations=1000
)
```

### LoongFlow Required
```python
config = UnifiedEvolutionConfig.loongflow_required(
    evolution_mode=EvolutionMode.PES
)
```

### With Fallback (Default)
```python
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=True
)
```

### No Fallback
```python
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=False
)
```

## Helper Methods

```python
# Check if enabled
config.is_loongflow_enabled()  # bool

# Check if should use (considers availability)
config.should_use_loongflow()  # bool or raises RuntimeError

# Check if available
config._check_loongflow_availability()  # bool
```

## Validation Rules

✅ **Valid:**
```python
enable_loongflow=True, require_loongflow=True
enable_loongflow=True, require_loongflow=False
enable_loongflow=False, require_loongflow=False
```

❌ **Invalid:**
```python
enable_loongflow=False, require_loongflow=True  # ValueError!
```

## Behavior Matrix

| enable_loongflow | fallback_enabled | require_loongflow | LoongFlow Not Installed |
|------------------|------------------|-------------------|-------------------------|
| True | True | False | ✅ Fallback to OpenEvolve (warning) |
| True | False | False | ✅ Use OpenEvolve (silent) |
| True | False | True | ❌ Raise RuntimeError |
| False | Any | False | ✅ Use OpenEvolve (silent) |

## Evolution Modes

### PES (LoongFlow)
```python
config = UnifiedEvolutionConfig.loongflow_required(
    evolution_mode=EvolutionMode.PES,
    pes=PESConfig(enabled=True)
)
```

### QD (OpenEvolve)
```python
config = UnifiedEvolutionConfig.openevolve_only(
    evolution_mode=EvolutionMode.QD,
    qd=QDConfig(enabled=True)
)
```

### MO (OpenEvolve)
```python
config = UnifiedEvolutionConfig.openevolve_only(
    evolution_mode=EvolutionMode.MO
)
```

## Error Handling

### Graceful Fallback
```python
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=True
)

if config.should_use_loongflow():
    # Use LoongFlow
else:
    # Use OpenEvolve (fallback)
    pass
```

### Strict Requirement
```python
config = UnifiedEvolutionConfig.loongflow_required()

try:
    if config.should_use_loongflow():
        # Use LoongFlow
        pass
except RuntimeError as e:
    print(f"Error: {e}")
    # Install LoongFlow or change config
```

## Domain Examples

```python
# Finance - OpenEvolve QD
config = UnifiedEvolutionConfig.openevolve_only(
    domain=DomainType.FINANCE,
    evolution_mode=EvolutionMode.QD
)

# Science - LoongFlow PES
config = UnifiedEvolutionConfig.loongflow_required(
    domain=DomainType.SCIENCE,
    evolution_mode=EvolutionMode.PES
)

# Trading - With fallback
config = UnifiedEvolutionConfig(
    domain=DomainType.TRADING,
    enable_loongflow=True,
    loongflow_fallback_enabled=True
)
```

## Migration

### From OpenEvolve
```python
# Before
from openevolve.config import OpenEvolveConfig
config = OpenEvolveConfig(max_iterations=1000)

# After
config = UnifiedEvolutionConfig.openevolve_only(
    max_iterations=1000
)
```

### From LoongFlow
```python
# Before
from loongflow.config import LoongFlowConfig
config = LoongFlowConfig(max_rounds=5)

# After
config = UnifiedEvolutionConfig.loongflow_required(
    pes=PESConfig(max_rounds=5)
)
```

## Best Practices

1. **Development:** Use fallback (`loongflow_fallback_enabled=True`)
2. **Production:** Require if needed (`require_loongflow=True`)
3. **Testing:** Disable explicitly (`openevolve_only()`)
4. **CI/CD:** Check availability first

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ImportError | `pip install loongflow` |
| RuntimeError (not available) | Install LoongFlow or set `require_loongflow=False` |
| Unexpected fallback | Check `enable_loongflow` and availability |
| Validation error | Check for contradictory settings |

## Quick Check

```python
# What will be used?
config = UnifiedEvolutionConfig()
if config.should_use_loongflow():
    print("✅ LoongFlow")
else:
    print("✅ OpenEvolve")
```
