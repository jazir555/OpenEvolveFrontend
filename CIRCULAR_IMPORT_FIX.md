# Circular Import Fix Documentation

## Problem Description

The OpenEvolve codebase had a circular import issue involving 4 files:

```
z3_api_server.py
  -> imports z3_leanaide_openevolve_integration.py
     -> imports bubblelabs_integration.py
        -> imports api_server.py (CIRCULAR!)
```

This caused `NameError` and `ImportError` exceptions when trying to import any of these modules.

## Root Cause

The file `bubblelabs_integration.py` was importing `team_manager` and `gauntlet_manager` 
from `api_server.py` at the module level (line 19), creating a circular dependency chain.

Additionally, `bubblelabs_integration.py` was instantiating `BubbleLabsIntegration()` at 
module level, which triggered the import chain during module loading.

## Solution Applied

### 1. Fixed `bubblelabs_integration.py`

**Changes made:**

1. **Converted module-level imports to lazy imports:**
   - Removed: `from api_server import team_manager, gauntlet_manager`
   - Added: `_get_api_server_managers()` function that imports inside the function
   - Added fallback to local `TeamManager()` and `GauntletManager()` if import fails

2. **Converted module-level instantiation to lazy proxy:**
   - Removed: `bubblelabs_integration = BubbleLabsIntegration()` (at module level)
   - Added: `_LazyIntegrationProxy` class that delays instantiation until first access
   - Added: `get_bubblelabs_integration()` function for explicit lazy initialization

3. **Updated `BubbleLabsIntegration.__init__`:**
   - Changed to use `_get_api_server_managers()` for lazy imports

### 2. Fixed `z3_leanaide_openevolve_integration.py`

**Changes made:**

1. **Added fallback for `VerificationStrategy`:**
   - When `z3_leanaide_bridge` import fails, define a local `VerificationStrategy` class
   - This ensures type hints work even when the bridge is not available

### 3. Fixed `collaboration_manager.py`

**Changes made:**
- Added `Any` to typing imports: `from typing import Any, Dict, List, Optional`

### 4. Fixed `export_import_manager.py`

**Changes made:**
- Added `Any` to typing imports: `from typing import Any, Dict, List, Optional`

## Files Modified

1. `bubblelabs_integration.py` - Lazy imports and lazy initialization
2. `z3_leanaide_openevolve_integration.py` - Fallback for VerificationStrategy
3. `collaboration_manager.py` - Added missing `Any` import
4. `export_import_manager.py` - Added missing `Any` import

## Verification

Run the test to verify the fix:

```bash
python test_imports_simple.py
```

Expected output:
```
Testing circular import fix...
============================================================
[PASS] bubblelabs_integration
[PASS] z3_api_server
============================================================
Summary:
  [PASS] bubblelabs_integration
  [PASS] z3_api_server

Circular import fix verified!
```

## Key Techniques Used

### 1. Lazy Import Pattern
```python
# Instead of:
from api_server import team_manager, gauntlet_manager

# Use:
def _get_api_server_managers():
    try:
        from api_server import team_manager, gauntlet_manager
        return team_manager, gauntlet_manager
    except ImportError:
        # Fallback
        return TeamManager(), GauntletManager()
```

### 2. Lazy Module-Level Instance Pattern
```python
# Instead of:
my_instance = MyClass()

# Use:
_instance = None

def get_instance():
    global _instance
    if _instance is None:
        _instance = MyClass()
    return _instance

class _LazyProxy:
    def __getattr__(self, name):
        return getattr(get_instance(), name)

my_instance = _LazyProxy()
```

### 3. Fallback Type Definition Pattern
```python
try:
    from some_module import SomeType
except ImportError:
    class SomeType:
        DEFAULT_VALUE = "default"
```

## Backward Compatibility

The changes maintain backward compatibility:
- Code that imports `bubblelabs_integration` from the module will still work
- The `bubblelabs_integration` variable is now a proxy that lazily initializes
- Direct use of `get_bubblelabs_integration()` is recommended for new code

## Additional Notes

During the fix, several other files were found to have missing type imports:
- `collaboration_manager.py` - Missing `Any`
- `export_import_manager.py` - Missing `Any`
- `validation_manager.py` - Missing `Optional`
- `suggestions.py` - Missing type imports

These are separate issues that should be fixed in a follow-up cleanup task.
