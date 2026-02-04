# Graceful Degradation Quick Start Guide

## For Developers Using the Knowledge Engine

### Check What's Available

```python
from knowledge_engine import get_capabilities, print_capability_report

# Get detailed capabilities
capabilities = get_capabilities()

# Print human-readable report
print_capability_report()

# Check specific integration
from knowledge_engine.integrations import DSPY_INTEGRATION_AVAILABLE
if DSPY_INTEGRATION_AVAILABLE:
    # Use DSPy features
    pass
```

### Use Optional Dependencies Safely

```python
from knowledge_engine.optional_imports import import_optional

# Import with silent failure
torch = import_optional(
    'torch',
    'torch',
    'neural network operations',
    'pip install torch',
    fail_silently=True
)

if torch:
    # Use torch
    import torch.nn as nn
else:
    # Use fallback
    print("Using fallback implementation")
```

### Check Integration Availability

```python
from knowledge_engine.integrations import (
    DEEPKE_INTEGRATION_AVAILABLE,
    DSPY_INTEGRATION_AVAILABLE,
    RAGBITS_INTEGRATION_AVAILABLE,
    ACE_INTEGRATION_AVAILABLE,
)

if DEEPKE_INTEGRATION_AVAILABLE:
    from knowledge_engine.integrations import DeepKEIntegration
    ke = DeepKEIntegration()
else:
    print("DeepKE not available, using mock")
```

---

## For Developers Adding New Integrations

### Pattern 1: Add Availability Flag

At the end of your integration file:

```python
# Availability flag
try:
    import external_library
    MY_INTEGRATION_AVAILABLE = True
except ImportError:
    MY_INTEGRATION_AVAILABLE = False
```

### Pattern 2: Export from __init__.py

In `knowledge_engine/integrations/__init__.py`:

```python
# My Integration
try:
    from .my_integration import (
        MyIntegration,
        MY_INTEGRATION_AVAILABLE
    )
except ImportError:
    MY_INTEGRATION_AVAILABLE = False

# Add to __all__
__all__ = [
    # ...
    "MyIntegration",
    "MY_INTEGRATION_AVAILABLE",
]
```

### Pattern 3: Use Optional Imports

```python
from ..optional_imports import import_optional

external_lib = import_optional(
    'external_lib',
    'external-lib',
    'my feature',
    'pip install external-lib',
    fail_silently=True
)

if external_lib:
    # Use the library
    result = external_lib.do_something()
else:
    # Provide fallback
    result = self._fallback_implementation()
```

### Pattern 4: Mock Implementation

```python
class MockExternalLib:
    """Mock implementation when external library unavailable."""
    def __init__(self):
        from ..optional_imports import OptionalDependencyError
        raise OptionalDependencyError(
            package_name='external-lib',
            feature_name='External library features',
            install_command='pip install external-lib'
        )
```

---

## Testing Graceful Degradation

```bash
# Run the graceful degradation test suite
python test_graceful_degradation.py

# Check all optional dependencies
python -c "from knowledge_engine.optional_imports import check_all_optional_dependencies; check_all_optional_dependencies()"

# Print capability report
python -c "from knowledge_engine import print_capability_report; print_capability_report()"
```

---

## Common Patterns

### Pattern: Conditional Feature

```python
def my_feature(self):
    if not MY_INTEGRATION_AVAILABLE:
        logger.warning("Integration not available, using fallback")
        return self._fallback()

    # Use full feature
    return self._full_feature()
```

### Pattern: Lazy Import

```python
class MyIntegration:
    def __init__(self):
        self._external_lib = None

    def _get_external_lib(self):
        if self._external_lib is None:
            self._external_lib = import_optional(
                'external_lib',
                'external-lib',
                'feature',
                'pip install external-lib',
                fail_silently=True
            )
        return self._external_lib

    def use_feature(self):
        lib = self._get_external_lib()
        if lib:
            return lib.do_something()
        return self._fallback()
```

### Pattern: Capability Check with Clear Message

```python
def require_integration(self):
    """Require the integration to be available."""
    if not MY_INTEGRATION_AVAILABLE:
        from ..optional_imports import OptionalDependencyError
        raise OptionalDependencyError(
            package_name='my-integration',
            feature_name='My integration features',
            install_command='pip install my-integration'
        )
```

---

## Environment Variables

```bash
# Disable configuration validation (not recommended)
export KE_VALIDATE_CONFIG=off

# Set to warn mode (default)
export KE_VALIDATE_CONFIG=warn

# Set to strict mode (fail on warnings)
export KE_VALIDATE_CONFIG=strict
```

---

## Troubleshooting

### Integration Not Available

```python
from knowledge_engine import print_capability_report
print_capability_report()

# Check specific integration
from knowledge_engine.integrations import MY_INTEGRATION_AVAILABLE
print(f"My Integration Available: {MY_INTEGRATION_AVAILABLE}")

# Try importing to see error
try:
    import external_lib
except ImportError as e:
    print(f"Error: {e}")
```

### Get Install Instructions

```python
from knowledge_engine.optional_imports import OPTIONAL_DEPENDENCIES

info = OPTIONAL_DEPENDENCIES.get('external_lib')
if info:
    print(f"Package: {info['package']}")
    print(f"Feature: {info['feature']}")
    print(f"Install: {info['install']}")
```

---

## Best Practices

1. **Always check availability flags before using integrations**
2. **Provide helpful fallback implementations**
3. **Log warnings when using fallback behavior**
4. **Use OptionalDependencyError for required features**
5. **Document what happens when integrations are unavailable**
6. **Test with minimal dependencies**

---

## Quick Reference

### Import Modules

```python
# Main imports
from knowledge_engine import get_capabilities, print_capability_report

# Integration flags
from knowledge_engine.integrations import (
    DEEPKE_INTEGRATION_AVAILABLE,
    DSPY_INTEGRATION_AVAILABLE,
    RAGBITS_INTEGRATION_AVAILABLE,
    # ... etc
)

# Optional imports utility
from knowledge_engine.optional_imports import (
    import_optional,
    require_dependency,
    is_available,
    OptionalDependencyError,
)
```

### Check Availability

```python
# Check integration
if DEEPKE_INTEGRATION_AVAILABLE:
    # Use it
    pass

# Check module
if is_available('torch'):
    import torch
    pass

# Get capabilities
caps = get_capabilities()
if 'My Integration' in caps['available']:
    # Use it
    pass
```

### Handle Missing Dependencies

```python
# Silent import
lib = import_optional(..., fail_silently=True)
if lib:
    # Use it

# Fail fast
lib = require_dependency(...)  # Raises if not available

# Check first
if is_available('lib'):
    lib = __import__('lib')
```

---

**For more details, see:** `GRACEFUL_DEGRADATION_REPORT.md`
