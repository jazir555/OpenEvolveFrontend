# ROMA Integration Refactoring Guide

## Overview

This guide provides instructions for refactoring code that imports directly from `core-projects/ROMA/` to use the canonical adapter instead.

## Current State Analysis

After analyzing 249 files with `roma_dspy` imports:

### ✅ Acceptable (No Refactoring Needed)

**~230 files** are internal ROMA core project files:
- Location: `core-projects/ROMA/src/`
- These can import from each other (internal project imports)
- **Verdict:** Leave as-is

**~8 root-level integration files** use graceful degradation:
- Location: Root level, `knowledge_engine/integrations/`
- Already use try/except with availability flags
- Have fallback mechanisms
- **Verdict:** Acceptable as-is

### ⚠️ Needs Refactoring (Pattern for New Code)

When writing **NEW code** that needs ROMA functionality, use this pattern:

## ✅ NEW PATTERN (Recommended)

### For Direct ROMA Usage

**OLD (violates Air Gap):**
```python
# ❌ VIOLATES Air Gap
from roma_dspy.core.engine.solve import RecursiveSolver, solve

solver = RecursiveSolver(config)
result = solve(goal, max_depth=3)
```

**NEW (Air Gap compliant):**
```python
# ✅ COMPLIANT
from glue.adapters.roma_bridge import get_roma_bridge, solve_with_roma

# Option 1: Using convenience function
result = await solve_with_roma(goal, max_depth=3)

# Option 2: Using bridge directly
bridge = get_roma_bridge()
result = await bridge.execute_task(goal, max_depth=3)
```

## Refactoring Examples

### Example 1: Simple ROMA Solve

**File:** `roma_integration.py` (hypothetical)

**Before:**
```python
from roma_dspy.core.engine.solve import solve, RecursiveSolver

def process_problem(problem: str):
    solver = RecursiveSolver()
    result = solver.solve(problem, max_depth=3)
    return result
```

**After:**
```python
from glue.adapters.roma_bridge import get_roma_bridge
import asyncio

async def process_problem(problem: str):
    bridge = get_roma_bridge()
    result = await bridge.execute_task(problem, max_depth=3)
    return result

# For synchronous contexts:
def process_problem_sync(problem: str):
    return asyncio.run(process_problem(problem))
```

### Example 2: Using Recursive Solver

**Before:**
```python
from roma_dspy.core.engine.solve import RecursiveSolver

class MySolver:
    def __init__(self):
        self.solver = RecursiveSolver()

    def solve(self, goal: str, max_depth: int = 3):
        return self.solver.solve(goal, max_depth=max_depth)
```

**After:**
```python
from glue.adapters.roma_bridge import get_roma_bridge, recursive_solve
import asyncio

class MySolver:
    def __init__(self):
        self.bridge = get_roma_bridge()

    async def solve(self, goal: str, max_depth: int = 3):
        return await self.bridge.execute_task(goal, max_depth=max_depth)

    def solve_sync(self, goal: str, max_depth: int = 3):
        return asyncio.run(self.solve(goal, max_depth))
```

### Example 3: Importing ROMA Components

**Before:**
```python
from roma_dspy.core.modules import Atomizer, Planner
from roma_dspy.config.schemas.root import ROMAConfig

atomizer = Atomizer()
planner = Planner()
config = ROMAConfig()
```

**After:**
```python
from glue.adapters.roma_bridge import get_roma_bridge

# Don't import modules - use ROMA API instead
bridge = get_roma_bridge()

# For configuration, use config profiles
config = {
    'profile': 'general',
    'max_depth': 3,
    'timeout': 30,
}

# ROMA modules are now handled by the API
result = await bridge.execute_task(goal, **config)
```

## Migration Checklist

For each file that directly imports from `core-projects/ROMA/`:

- [ ] Identify ROMA imports
- [ ] Add canonical bridge import
- [ ] Replace direct solver calls with bridge API
- [ ] Update async/await patterns as needed
- [ ] Test refactored code
- [ ] Update imports

## Files That CAN Use Canonical Bridge

These root-level files can be updated to use the canonical bridge:

### 1. roma_matryoshka_integration.py

**Current (line 77):**
```python
from roma_dspy.core.engine.solve import RecursiveSolver
```

**Refactored:**
```python
from glue.adapters.roma_bridge import get_roma_bridge

# In methods:
async def solve_with_matryoshka(goal: str, max_depth: int = 3):
    bridge = get_roma_bridge()
    result = await bridge.execute_task(
        goal,
        max_depth=max_depth,
        execution_method='roma'
    )
    return result
```

### 2. roma_decomposition_hybrid.py

**Current (lines 48-58):**
```python
try:
    # from roma_dspy.core.engine.solve import  # Stubbed
    # from roma_dspy.config.schemas.root import  # Stubbed
    ROMA_AVAILABLE = True
except ImportError:
    ROMA_AVAILABLE = False
```

**Refactored:**
```python
from glue.adapters.roma_bridge import get_roma_bridge, ROMA_AVAILABLE

# Keep availability flag for optional dependency
try:
    bridge = get_roma_bridge()
    # Quick health check
    status = await bridge.get_status()
    ROMA_AVAILABLE = (status.get('status') == 'healthy')
except Exception:
    ROMA_AVAILABLE = False

# Use bridge instead of direct imports
if ROMA_AVAILABLE:
    result = await solve_with_roma(goal)
```

### 3. Knowledge Engine Integration

**File:** `knowledge_engine/integrations/roma_integration.py`

**Add to imports:**
```python
from glue.adapters.roma_bridge import get_roma_bridge, RomaCanonicalBridge
```

**Update ROMAIntegration class:**
```python
class ROMAIntegration:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # ... existing code ...

        # NEW: Use canonical bridge
        try:
            self.roma_bridge = get_roma_bridge()
            self.ROMA_AVAILABLE = True  # Update based on health check
        except Exception as e:
            logger.warning(f"ROMA bridge not available: {e}")
            self.ROMA_AVAILABLE = False
            self.roma_bridge = None
```

## Best Practices

### 1. Always Use Async

The canonical bridge uses async I/O. In synchronous contexts:

```python
import asyncio

def sync_function():
    result = asyncio.run(async_function())
    return result
```

### 2. Handle Availability

```python
from glue.adapters.roma_bridge import ROMA_AVAILABLE, get_roma_bridge

if ROMA_AVAILABLE:
    bridge = get_roma_bridge()
    # Use bridge
else:
    # Use fallback
    logger.warning("ROMA unavailable, using fallback")
```

### 3. Error Handling

```python
try:
    result = await bridge.execute_task(goal)
except requests.exceptions.ConnectionError:
    logger.error("ROMA server unavailable")
    # Use fallback
except requests.exceptions.Timeout:
    logger.error("ROMA request timed out")
    # Retry with shorter timeout
```

### 4. Configuration

```python
# Set environment variables
ROMA_SERVER_URL = os.getenv('ROMA_SERVER_URL', 'http://localhost:8000')
ROMA_API_KEY = os.getenv('ROMA_API_KEY', '')
ROMA_TIMEOUT = int(os.getenv('ROMA_TIMEOUT', '30000'))
```

## Testing Refactored Code

### Unit Test Example

```python
import pytest
from glue.adapters.roma_bridge import RomaCanonicalBridge, reset_roma_bridge

@pytest.fixture
def roma_bridge():
    bridge = RomaCanonical(
        server_url='http://test:8000',
        api_key='test-key'
    )
    return bridge

@pytest.mark.asyncio
async def test_execute_task(roma_bridge):
    result = await roma_bridge.execute_task(
        goal="Test goal",
        max_depth=1
    )
    assert result.execution_id
    assert result.status in ['pending', 'completed']
```

## Summary

- **✅ Glue layer:** Already 100% compliant, no changes needed
- **✅ ROMA core files:** Internal imports acceptable, no changes needed
- **⚠️ Root-level files:** Can use canonical bridge for NEW code
- **📋 Pattern:** Use `get_roma_bridge()` for all new ROMA integrations

**Key Benefit:** Air Gap compliance without breaking existing functionality.
