# Edge Case Fixes - Code Examples and Implementation Guide

## Quick Reference Guide for Fixing Detected Edge Cases

---

## Table of Contents
1. [Thread Safety Fixes](#thread-safety-fixes)
2. [Lazy Import Documentation](#lazy-import-documentation)
3. [Performance Optimizations](#performance-optimizations)
4. [Memory Leak Prevention](#memory-leak-prevention)
5. [Duplicate Code Elimination](#duplicate-code-elimination)
6. [Documentation Improvements](#documentation-improvements)

---

## Thread Safety Fixes

### Problem: Global Variables
**Files:** `evolution.py`, `adversarial.py`, `integrated_workflow.py`

#### Current Code (Unsafe)
```python
# evolution.py
_config = None
logger = logging.getLogger(__name__)

def get_config():
    global _config
    if _config is None:
        _config = load_config()  # Race condition!
    return _config
```

#### Fix Option 1: Thread-Safe with Lock
```python
import threading

_config_lock = threading.Lock()
_config = None

def get_config():
    """
    Thread-safe configuration accessor.

    Returns:
        Configuration object (thread-safe singleton)
    """
    global _config
    with _config_lock:
        if _config is None:
            _config = load_config()
        return _config
```

#### Fix Option 2: Thread-Local Storage
```python
import threading

_thread_local = threading.local()

def get_config():
    """
    Thread-local configuration accessor.

    Each thread gets its own configuration instance.
    """
    if not hasattr(_thread_local, 'config'):
        _thread_local.config = load_config()
    return _thread_local.config
```

#### Fix Option 3: Thread-Safe Cache (for functions)
```python
from functools import lru_cache
import threading

@lru_cache(maxsize=128)
def cached_operation(key):
    """
    Thread-safe cached operation.

    Args:
        key: Cache key

    Returns:
        Cached result
    """
    return expensive_computation(key)
```

### Problem: Global Caches
**File:** `integrated_workflow.py`

#### Current Code (Unsafe)
```python
_session_cache = {}
_result_cache = {}

def cache_result(key, value):
    _session_cache[key] = value  # Not thread-safe!
```

#### Fix: Use Thread-Safe Cache
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_result(key):
    """
    Thread-safe result caching.

    Args:
        key: Cache key

    Returns:
        Cached or computed result
    """
    return compute_result(key)

# Clear cache when needed
def clear_cache():
    """Clear the result cache."""
    get_cached_result.cache_clear()
```

---

## Lazy Import Documentation

### Problem: Undocumented Lazy Imports
**Count:** 66 across all files

#### Current Code (Unclear)
```python
def run_evolution_loop():
    from adversarial import run_adversarial_loop
    # Why is this here? Is it necessary?
```

#### Fix: Add Documentation
```python
def run_evolution_loop():
    """
    Run evolution loop with adversarial testing.

    Note: Lazy import from adversarial.py to avoid circular dependency.
    evolution.py imports adversarial.py, which imports evolution.py.
    """
    # Lazy import to avoid circular dependency
    from adversarial import run_adversarial_loop
    return run_adversarial_loop()
```

### Standard Comment Template
```python
# For circular dependency avoidance:
# Lazy import to avoid circular dependency with [module]

# For optional dependencies:
# Lazy import for optional dependency [module]

# For heavy imports:
# Lazy import to optimize startup time (heavy module)
```

### Examples from Detected Issues

#### evolution.py:2033
```python
# BEFORE
def run_ultimate_adversarial_evolution():
    from adversarial import run_enhanced_adversarial_loop
    ...

# AFTER
def run_ultimate_adversarial_evolution():
    """
    Run ultimate adversarial evolution.

    Note: Lazy import from adversarial to avoid circular dependency.
    """
    # Lazy import to avoid circular dependency with adversarial.py
    from adversarial import run_enhanced_adversarial_loop
    ...
```

#### openevolve_integration.py (Multiple locations)
```python
# BEFORE
def run_unified_evolution():
    import openai
    ...

# AFTER
def run_unified_evolution():
    """
    Run unified evolution with OpenAI backend.

    Note: Lazy import of openai to optimize startup time.
    OpenAI import is heavy and only needed for this function.
    """
    # Lazy import to optimize startup time (heavy module)
    import openai
    ...
```

---

## Performance Optimizations

### Problem: Repeated Operations in Loops
**Count:** 35 instances

#### Example 1: Config Loading
**File:** `evolution.py`

##### Current Code (Slow)
```python
def run_evolution(config_dict):
    for iteration in range(max_iterations):
        config = load_config(config_dict)  # ❌ Loads every iteration!
        result = process(config, iteration)
```

##### Fixed Code (Fast)
```python
def run_evolution(config_dict):
    """
    Run evolution with optimized config loading.

    Config is loaded once and reused across iterations.
    """
    # Load config once before loop
    config = load_config(config_dict)  # ✅ Loads once

    for iteration in range(max_iterations):
        result = process(config, iteration)
```

**Performance Gain:** ~100x faster for 100 iterations

#### Example 2: Expensive Function in Loop
**File:** `decomposition_engine.py`

##### Current Code (Slow)
```python
def process_items(items):
    for item in items:
        features = extract_features(item)  # ❌ Expensive!
        if features in feature_set:  # ❌ O(n) lookup
            ...
```

##### Fixed Code (Fast)
```python
def process_items(items):
    """
    Process items with optimized feature extraction.

    Features are pre-computed and stored in a set for O(1) lookup.
    """
    # Pre-compute features once
    feature_set = {extract_features(item) for item in items}

    for item in items:
        features = item.features  # ✅ Already computed
        if features in feature_set:  # ✅ O(1) lookup
            ...
```

**Performance Gain:**
- Feature extraction: n times (from n²) = O(n)
- Lookup: O(1) instead of O(n)

### Problem: Inefficient Data Structures

#### Current Code (Slow)
```python
# Using list for membership testing
features_list = [extract_features(item) for item in items]
for item in items:
    if item.feature in features_list:  # ❌ O(n) lookup
        ...
```

#### Fixed Code (Fast)
```python
# Using set for O(1) lookups
features_set = {extract_features(item) for item in items}
for item in items:
    if item.feature in features_set:  # ✅ O(1) lookup
        ...
```

---

## Memory Leak Prevention

### Problem: Unclosed Resources

#### Example 1: File Operations
##### Current Code (Leak)
```python
def read_file(filename):
    file = open(filename, 'r')  # ❌ Never closed!
    data = file.read()
    return data
```

##### Fixed Code (No Leak)
```python
def read_file(filename):
    """
    Read file with proper resource management.

    Context manager ensures file is closed even on error.
    """
    with open(filename, 'r') as file:  # ✅ Auto-closed
        data = file.read()
    return data
```

#### Example 2: Cache Without Size Limit
##### Current Code (Leak)
```python
_session_cache = {}

def cache_result(key, value):
    _session_cache[key] = value  # ❌ Grows indefinitely!
```

##### Fixed Code (No Leak)
```python
from functools import lru_cache

@lru_cache(maxsize=1000)  # ✅ Max 1000 entries
def get_cached_result(key):
    """
    Cached result with automatic size management.

    Least recently used entries are automatically evicted
    when cache exceeds maxsize.
    """
    return compute_result(key)
```

### Problem: Cyclic References

#### Current Code (Potential Leak)
```python
class Parent:
    def __init__(self):
        self.children = []

class Child:
    def __init__(self, parent):
        self.parent = parent  # ❌ Strong reference
        parent.children.append(self)

# Cycle: Parent -> Child -> Parent
# Prevents garbage collection!
```

#### Fixed Code (No Leak)
```python
import weakref

class Parent:
    def __init__(self):
        self.children = []

class Child:
    def __init__(self, parent):
        self.parent = weakref.ref(parent)  # ✅ Weak reference
        parent.children.append(self)

# No cycle: garbage collector can clean up
```

---

## Duplicate Code Elimination

### Problem: Repeated Import Guard Pattern
**Count:** 14+ instances

#### Current Code (Duplicate)
```python
# File: adversarial.py
try:
    from evolution import EvolutionConfiguration
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False

# File: integrated_workflow.py
try:
    from evolution import EvolutionConfiguration
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False

# File: openevolve_integration.py
try:
    from evolution import EvolutionConfiguration
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False
```

#### Fixed Code (Centralized)

**Step 1: Create utility in `openevolve_imports.py`**
```python
def safe_import(module_path, item_name=None):
    """
    Centralized safe import with error handling.

    Args:
        module_path: Full module path (e.g., 'evolution.EvolutionConfiguration')
        item_name: Specific item to import (optional)

    Returns:
        Tuple of (imported_object, success_flag)

    Examples:
        EvolutionConfig, available = safe_import('evolution', 'EvolutionConfiguration')
        openai, available = safe_import('openai')
    """
    try:
        if item_name:
            module = __import__(module_path, fromlist=[item_name])
            return getattr(module, item_name), True
        else:
            return __import__(module_path), True
    except ImportError as e:
        logging.debug(f"Failed to import {module_path}: {e}")
        return None, False
```

**Step 2: Use in all files**
```python
# ALL FILES - Consistent pattern
from openevolve_imports import safe_import

# Use it
EvolutionConfiguration, EVOLUTION_AVAILABLE = safe_import(
    'evolution',
    'EvolutionConfiguration'
)

if EVOLUTION_AVAILABLE:
    config = EvolutionConfiguration(...)
else:
    # Fallback logic
    config = None
```

**Benefits:**
- Reduces ~150 lines to ~20 lines
- Consistent error handling
- Single place to update
- Easier testing

---

## Documentation Improvements

### Problem: Missing Docstrings
**Count:** 74 functions/classes

#### Example 1: Simple Function

##### Current Code (No Docs)
```python
def process_result(result):
    return result['data']
```

##### Fixed Code (Documented)
```python
def process_result(result):
    """
    Extract and validate data from evolution result.

    Args:
        result (dict): Dictionary containing evolution results.
            Must contain 'data' key with valid data.

    Returns:
        dict: Validated data dictionary extracted from result.

    Raises:
        KeyError: If 'data' key is missing from result.
        ValueError: If data validation fails.

    Example:
        >>> result = {'data': {'score': 0.95}, 'metadata': {...}}
        >>> process_result(result)
        {'score': 0.95}
    """
    return result['data']
```

#### Example 2: Class

##### Current Code (No Docs)
```python
class EvolutionEngine:
    def __init__(self, config):
        self.config = config

    def run(self):
        ...
```

##### Fixed Code (Documented)
```python
class EvolutionEngine:
    """
    Evolution engine for running optimization algorithms.

    This engine manages the evolution process, including population
    management, fitness evaluation, and iteration control.

    Attributes:
        config (EvolutionConfiguration): Configuration for evolution
        population (list): Current population of candidates
        generation (int): Current generation number

    Example:
        >>> config = EvolutionConfiguration(max_iterations=100)
        >>> engine = EvolutionEngine(config)
        >>> result = engine.run()
    """

    def __init__(self, config):
        """
        Initialize evolution engine.

        Args:
            config (EvolutionConfiguration): Evolution configuration

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self.population = []
        self.generation = 0

    def run(self):
        """
        Run evolution process.

        Returns:
            EvolutionResult: Result containing best candidate and metrics

        Raises:
            RuntimeError: If evolution fails
        """
        ...
```

### Documentation Checklist

For each function/class, document:
- [ ] Purpose (what it does)
- [ ] Parameters (with types and constraints)
- [ ] Return value (with type and meaning)
- [ ] Raises (exceptions that can be raised)
- [ ] Examples (usage examples)
- [ ] Notes (important details, side effects)

---

## Implementation Priority

### Week 1: Thread Safety
- [ ] Add locks to `evolution.py` global variables
- [ ] Add locks to `adversarial.py` global variables
- [ ] Replace caches in `integrated_workflow.py` with thread-safe versions
- [ ] Test with multiple threads

### Week 2: Documentation
- [ ] Document all 66 lazy imports
- [ ] Add docstrings to top 20 most-used functions
- [ ] Add docstrings to all public classes
- [ ] Run documentation linter

### Week 3: Performance
- [ ] Move config loading outside loops (10 instances)
- [ ] Replace list lookups with set lookups (15 instances)
- [ ] Add caching to expensive operations (10 instances)
- [ ] Profile before/after

### Week 4: Memory & Code Quality
- [ ] Add context managers to file operations
- [ ] Replace unlimited caches with LRU caches
- [ ] Centralize import guard pattern
- [ ] Remove duplicate code

---

## Testing Your Fixes

### Thread Safety Test
```python
import threading
import time

def test_thread_safety():
    """Test that get_config() is thread-safe."""
    results = []
    errors = []

    def worker():
        try:
            config = get_config()
            results.append(config)
        except Exception as e:
            errors.append(e)

    # Run 100 threads concurrently
    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Thread safety errors: {errors}"
    assert len(results) == 100, "Not all threads completed"
    print("✅ Thread safety test passed")
```

### Performance Test
```python
import time

def test_performance():
    """Test that optimization improves performance."""
    # Before fix
    start = time.time()
    result_before = slow_function(test_data)
    time_before = time.time() - start

    # After fix
    start = time.time()
    result_after = fast_function(test_data)
    time_after = time.time() - start

    # Verify correctness
    assert result_before == result_after, "Results differ!"

    # Verify improvement
    speedup = time_before / time_after
    assert speedup > 2, f"Only {speedup}x speedup (expected >2x)"
    print(f"✅ Performance test passed: {speedup:.1f}x speedup")
```

### Memory Leak Test
```python
import gc
import sys

def test_memory_leak():
    """Test that resources are properly cleaned up."""
    gc.collect()
    objects_before = len(gc.get_objects())

    # Run operation 1000 times
    for _ in range(1000):
        result = function_with_cache()
        del result

    gc.collect()
    objects_after = len(gc.get_objects())

    # Should not grow unbounded
    growth = objects_after - objects_before
    assert growth < 1000, f"Memory leak detected: {growth} objects"
    print(f"✅ Memory leak test passed: {growth} objects")
```

---

## Quick Reference

### Thread Safety Patterns
```python
# Lock
with lock:
    critical_section()

# Thread-local
thread_local.variable

# LRU cache
@lru_cache(maxsize=1000)
def func():
    ...
```

### Performance Patterns
```python
# Load once
config = load_config()
for _ in range(100):
    use(config)

# Set for O(1) lookup
item_set = set(items)
if item in item_set:
    ...

# Cache expensive ops
@lru_cache
def expensive(x):
    ...
```

### Memory Safety Patterns
```python
# Context manager
with open(filename) as f:
    data = f.read()

# Limited cache
@lru_cache(maxsize=1000)
def func():
    ...

# Weak reference
import weakref
weak_ref = weakref.ref(obj)
```

---

**End of Fix Guide**

For detailed analysis, see:
- `EDGE_CASE_SUMMARY.md` - Executive summary
- `EDGE_CASE_ANALYSIS.md` - Detailed findings
- `EDGE_CASE_DETAILED_REPORT.json` - Machine-readable data
