# ParameterManager Migration - Quick Reference Guide

**For Developers:** Quick patterns for working with the backward-compatible configuration system

---

## Import Pattern (Add to Top of File)

```python
# Standard backward-compatible import
try:
    from parameter_manager import ParameterManager, ValidationResult
    PARAMETER_MANAGER_AVAILABLE = True
except ImportError:
    PARAMETER_MANAGER_AVAILABLE = False

    # Fallback ValidationResult for when ParameterManager unavailable
    class ValidationResult:
        def __init__(self, valid: bool = True, errors=None, warnings=None):
            self.valid = valid
            self.errors = errors or []
            self.warnings = warnings or []
```

---

## Common Patterns

### 1. Conditional ParameterManager Creation

```python
# ✅ CORRECT - With fallback
if PARAMETER_MANAGER_AVAILABLE:
    param_manager = ParameterManager()
else:
    param_manager = None
    logger.warning("ParameterManager not available - using defaults")

# ❌ WRONG - Will crash if unavailable
param_manager = ParameterManager()
```

### 2. Creating Configuration

```python
# ✅ NEW WAY - Use UnifiedConfiguration (recommended)
from unified_configuration import create_unified_config

unified = create_unified_config({
    'max_iterations': 20,
    'temperature': 0.7
})

# Convert to specific config type
from evolution import EvolutionConfiguration
config = EvolutionConfiguration.from_unified_config(unified)

# ⚠️ OLD WAY - Still works but deprecated
from parameter_manager import ParameterManager
from evolution import EvolutionConfiguration

if PARAMETER_MANAGER_AVAILABLE:
    param_manager = ParameterManager()
    config = EvolutionConfiguration.from_parameter_manager(param_manager, session_state)
```

### 3. Validating Configuration

```python
# ✅ NEW WAY - Validation handles unavailable ParameterManager
validation = config.validate()  # No arguments needed

if not validation.valid:
    print(f"Errors: {validation.errors}")

# ⚠️ OLD WAY - Still works
validation = config.validate(param_manager)
```

### 4. Getting Defaults

```python
# ✅ NEW WAY - UnifiedConfiguration handles defaults
from unified_configuration import create_unified_config

# Empty config gets all defaults
config = create_unified_config()  # Has all 272 params with defaults

# Add specific overrides
config = create_unified_config({'max_iterations': 50})

# ⚠️ OLD WAY - Requires ParameterManager
if PARAMETER_MANAGER_AVAILABLE:
    param_manager = ParameterManager()
    defaults = param_manager.get_defaults()
    defaults.update({'max_iterations': 50})
```

---

## Function Migration Checklist

When migrating a function that uses ParameterManager:

- [ ] Add backward-compatible import at top of file
- [ ] Check if `param_manager = ParameterManager()` exists
- [ ] Wrap in `if PARAMETER_MANAGER_AVAILABLE:`
- [ ] Add fallback logic for when unavailable
- [ ] Update method signatures to accept `Optional[ParameterManager]`
- [ ] Add deprecation notice if replacing old method
- [ ] Test with ParameterManager available
- [ ] Test without ParameterManager (mock import failure)

---

## Testing Patterns

### Test With and Without ParameterManager

```python
import pytest
from unittest.mock import patch

def test_function_with_pm():
    """Test with ParameterManager available"""
    # Normal test
    result = my_function()
    assert result['success']

def test_function_without_pm():
    """Test without ParameterManager"""
    # Mock ParameterManager to be unavailable
    with patch.dict('sys.modules', {'parameter_manager': None}):
        # Reimport to trigger ImportError
        import importlib
        import my_module
        importlib.reload(my_module)

        result = my_module.my_function()
        assert result['success']  # Should work with fallback

# Or use skipif
@pytest.mark.skipif(
    not my_module.PARAMETER_MANAGER_AVAILABLE,
    reason="ParameterManager not available"
)
def test_requires_pm():
    """Test that requires ParameterManager"""
    param_manager = ParameterManager()
    # ... test logic
```

---

## Migration Examples

### Example 1: Simple Function

**Before:**
```python
from parameter_manager import ParameterManager

def my_function():
    param_manager = ParameterManager()
    config = param_manager.get_defaults()
    # ... do work
    return result
```

**After:**
```python
try:
    from parameter_manager import ParameterManager
    PARAMETER_MANAGER_AVAILABLE = True
except ImportError:
    PARAMETER_MANAGER_AVAILABLE = False

def my_function():
    if PARAMETER_MANAGER_AVAILABLE:
        param_manager = ParameterManager()
        config = param_manager.get_defaults()
    else:
        # Fallback
        from unified_configuration import create_unified_config
        unified = create_unified_config()
        config = unified.parameters

    # ... do work (same logic)
    return result
```

### Example 2: Method with ParameterManager Parameter

**Before:**
```python
def validate(self, param_manager: ParameterManager) -> ValidationResult:
    return param_manager.validate(asdict(self))
```

**After:**
```python
def validate(self, param_manager: Optional[ParameterManager] = None) -> ValidationResult:
    """
    Validate configuration.

    Args:
        param_manager: Optional ParameterManager (uses internal if not provided)
    """
    if not PARAMETER_MANAGER_AVAILABLE or not param_manager:
        # Fallback - assume valid
        return ValidationResult(valid=True, errors=[], warnings=[])

    return param_manager.validate(asdict(self))
```

### Example 3: Class Method

**Before:**
```python
@classmethod
def from_parameter_manager(cls, param_manager: ParameterManager, session_state: Dict):
    config = cls()
    defaults = param_manager.get_defaults()
    for key, value in session_state.items():
        setattr(config, key, value)
    return config
```

**After:**
```python
@classmethod
def from_parameter_manager(cls, param_manager: Optional[ParameterManager], session_state: Dict):
    """
    Create configuration from ParameterManager.

    DEPRECATED: Use from_unified_config() instead.
    """
    if not PARAMETER_MANAGER_AVAILABLE or not param_manager:
        logger.warning("ParameterManager not available - creating from session state")
        from unified_configuration import create_unified_config
        unified = create_unified_config(session_state, validate=False)
        return cls.from_unified_config(unified)

    config = cls()
    defaults = param_manager.get_defaults()
    for key, value in session_state.items():
        setattr(config, key, value)
    return config
```

---

## Common Errors and Fixes

### Error: ImportError: No module named 'parameter_manager'

**Cause:** Code doesn't have backward-compatible import

**Fix:**
```python
# Add this to top of file
try:
    from parameter_manager import ParameterManager
    PARAMETER_MANAGER_AVAILABLE = True
except ImportError:
    PARAMETER_MANAGER_AVAILABLE = False

# Then use it
if PARAMETER_MANAGER_AVAILABLE:
    param_manager = ParameterManager()
else:
    # Fallback logic
```

### Error: AttributeError: 'NoneType' object has no attribute 'validate'

**Cause:** Trying to use param_manager when it's None

**Fix:**
```python
# Add check before using
if param_manager:
    result = param_manager.validate(config)
else:
    # Fallback validation
    result = ValidationResult(valid=True)
```

### Error: NameError: name 'ParameterManager' is not defined

**Cause:** Type hint uses ParameterManager but import failed

**Fix:**
```python
from typing import Optional, TYPE_CHECKING

try:
    from parameter_manager import ParameterManager
    PARAMETER_MANAGER_AVAILABLE = True
except ImportError:
    PARAMETER_MANAGER_AVAILABLE = False
    # Only import for type hints
    if TYPE_CHECKING:
        from parameter_manager import ParameterManager

# Use string annotation in function signature
def my_function(manager: Optional['ParameterManager'] = None):
    # ... function body
```

---

## Best Practices

### ✅ DO
- Always use try/except for ParameterManager imports
- Provide sensible fallbacks when unavailable
- Log warnings when falling back
- Use Optional['ParameterManager'] in type hints
- Test both code paths (with and without ParameterManager)
- Document deprecated methods
- Use UnifiedConfiguration for new code

### ❌ DON'T
- Don't assume ParameterManager is always available
- Don't use hard-coded ParameterManager() without checks
- Don't skip fallback logic
- Don't remove old methods without deprecation period
- Don't use ParameterManager in type hints without TYPE_CHECKING
- Don't forget to test graceful degradation

---

## Quick Decision Tree

```
Need to use ParameterManager?
│
├─ Is it for a NEW feature?
│  └─ YES → Use UnifiedConfiguration instead
│         from unified_configuration import create_unified_config
│
├─ Is it existing code?
│  └─ YES → Add backward compatibility
│         try:
│             from parameter_manager import ParameterManager
│             PARAMETER_MANAGER_AVAILABLE = True
│         except ImportError:
│             PARAMETER_MANAGER_AVAILABLE = False
│
├─ Are you creating configuration?
│  └─ YES → Use create_unified_config()
│
├─ Are you validating?
│  └─ YES → Use config.validate() (no arguments)
│
└─ Are you getting defaults?
   └─ YES → Use create_unified_config() (empty params)
```

---

## Configuration Conversion Patterns

### From ParameterManager to UnifiedConfiguration

```python
# OLD
from parameter_manager import ParameterManager
param_manager = ParameterManager()
defaults = param_manager.get_defaults()
defaults['max_iterations'] = 20

# NEW
from unified_configuration import create_unified_config
config = create_unified_config({'max_iterations': 20})
# All other params have defaults already applied
```

### From Session State

```python
# OLD
from parameter_manager import ParameterManager
from evolution import EvolutionConfiguration

param_manager = ParameterManager()
config = EvolutionConfiguration.from_parameter_manager(
    param_manager,
    st.session_state
)

# NEW
from unified_configuration import create_unified_config
from evolution import EvolutionConfiguration

unified = create_unified_config(st.session_state)
config = EvolutionConfiguration.from_unified_config(unified)
```

---

## Status Tracking

### Completed Migrations ✅
- unified_configuration.py
- base_configuration.py
- adversarial.py

### In Progress 🔄
- evolution.py (80% complete)

### TODO 📋
- Complete evolution.py (7 instances remaining)
- Migrate test files (8 files)
- Update documentation

---

## Need Help?

1. **Check the full migration report:** `BATCH_3D_MIGRATION_REPORT.md`
2. **See working examples:** `unified_configuration.py`, `base_configuration.py`
3. **Test your changes:** Run tests with and without ParameterManager
4. **Log issues:** Include the error message and code snippet

---

**Remember:** The goal is backward compatibility. Code should work with OR WITHOUT ParameterManager available.
