# Causal-Learn Optionality Verification Report

**Date**: 2026-02-03  
**Status**: ✅ **FULLY OPTIONAL**  
**Test Results**: 7/7 tests passed

---

## Executive Summary

Causal-learn integration is **100% optional**. The system works correctly both:
- **WITH** causal-learn library installed (full functionality)
- **WITHOUT** causal-learn library installed (graceful degradation)

---

## Test Results

```
======================================================================
CAUSAL-LEARN OPTIONALITY TEST SUITE
======================================================================
[PASS]: Integration vs Library         ✅
[PASS]: Imports                        ✅
[PASS]: CausalLearnIntegration         ✅
[PASS]: UnifiedKnowledgeExtractor      ✅
[PASS]: Master Engine                  ✅
[PASS]: KnowledgeOrchestrator          ✅
[PASS]: AdvancedAnalyticsEngine        ✅
----------------------------------------------------------------------
Total: 7 tests
Passed: 7
Failed: 0

[OK] Causal-learn is FULLY OPTIONAL - system works without the library!
     Integration code exists but gracefully handles missing library.
======================================================================
```

---

## Key Design: Integration vs Library

The system correctly distinguishes between two concepts:

| Concept | Description | When Unavailable |
|---------|-------------|------------------|
| **Integration Module** | Python code that provides the interface | N/A - always exists |
| **Causal-Learn Library** | External `causallearn` package | Graceful degradation |

### Integration Module (`causal_learn_integration.py`)
- Always importable (part of OpenEvolve codebase)
- Provides `CausalLearnIntegration` and `CausalDiscoveryEngine` classes
- Checks library availability at runtime via `is_available()` method

### Causal-Learn Library (`causallearn`)
- External dependency (installed via `pip install causal-learn`)
- Imported dynamically within methods, not at module level
- When unavailable, methods return error messages instead of raising exceptions

---

## Implementation Details

### 1. Knowledge Engine Integration

**File**: `knowledge_engine/integrations/causal_learn_integration.py`

```python
class CausalDiscoveryEngine:
    def __init__(self):
        self._causal_learn_available = False
        self._initialize_causal_learn()
    
    def _initialize_causal_learn(self):
        try:
            from causallearn.search.ConstraintBased import PC, FCI
            # ... other imports
            self._causal_learn_available = True
        except ImportError as e:
            print(f"Note: algorithms not available: {e}")
    
    def is_available(self) -> bool:
        return self._causal_learn_available
    
    def discover_causal_structure(self, ...):
        if not self.is_available():
            return {
                'status': 'error',
                'message': 'Causal-learn integration not available',
                'graph': None
            }
        # ... actual implementation
```

**Key Points**:
- ✅ Library imports are inside methods, not at module level
- ✅ `is_available()` method reports library status
- ✅ Methods return error dicts instead of raising exceptions
- ✅ Warning messages inform user of missing library

---

### 2. Master Engine Integration

**File**: `knowledge_engine/master_engine.py`

```python
try:
    from knowledge_engine.integrations.causal_learn_integration import CausalLearnIntegration
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CausalLearnIntegration = None
    CAUSAL_LEARN_AVAILABLE = False

# Component initialization
self.components['causal_learn'] = (
    self._safe_init(CausalLearnIntegration, 'causal_learn') 
    if CAUSAL_LEARN_AVAILABLE 
    else self._create_mock_component('causal_learn')
)
```

**Key Points**:
- ✅ Import wrapped in try/except
- ✅ `CAUSAL_LEARN_AVAILABLE` flag tracks integration availability
- ✅ Falls back to mock component if import fails
- ✅ Component capabilities always registered

---

### 3. Unified Knowledge Extractor

**File**: `knowledge_engine/integrations/unified_knowledge_extraction.py`

```python
try:
    from .causal_learn_integration import CausalDiscoveryEngine
except ImportError:
    CausalDiscoveryEngine = None

# Initialization
if CausalDiscoveryEngine:
    try:
        self.modules['causal_learn'] = CausalDiscoveryEngine()
    except Exception as e:
        print(f"Warning: Could not initialize Causal-Learn: {e}")

# Method implementation
def discover_causal_structure(self, ...):
    if 'causal_learn' not in self.modules:
        return ExtractionResult(
            status='error',
            errors=['Causal-Learn module not available']
        )
    # ... actual implementation
```

**Key Points**:
- ✅ Import wrapped in try/except
- ✅ Module only added if successfully initialized
- ✅ Methods check module availability before use
- ✅ Returns `ExtractionResult` with error status

---

### 4. Knowledge Orchestrator

**File**: `knowledge_engine/orchestration/knowledge_orchestrator.py`

```python
# Component is always defined
ComponentType.CAUSAL_LEARN = "causal_learn"

# Enabled by default, but library availability checked at runtime
default_components = {
    ComponentType.CAUSAL_LEARN: ComponentConfig(
        enabled=True,  # Enabled in config
        required=False
    ),
}

# Pipeline stage
PipelineStage(
    name="discover_causal_structure",
    component=ComponentType.CAUSAL_LEARN,
    enabled=True,
    depends_on=["build_graph"],
    condition="len(get(context, 'graph_nodes', [])) > 2"
)
```

**Key Points**:
- ✅ Component type always defined
- ✅ Pipeline stage always exists
- ✅ Runtime checks handle missing library
- ✅ Stage can be disabled via config if needed

---

### 5. Advanced Analytics Engine

**File**: `knowledge_engine/advanced_analytics_engine.py`

```python
try:
    from .integrations import (
        CausalDiscoveryEngine,
        # ... other imports
    )
    INTEGRATIONS_AVAILABLE = True
except ImportError:
    INTEGRATIONS_AVAILABLE = False
    CausalDiscoveryEngine = None
    # ... other modules set to None

# Initialization
if self.config['causal_learn']['enabled'] and CausalDiscoveryEngine:
    try:
        self.integrations['causal'] = CausalDiscoveryEngine()
    except Exception as e:
        logger.error(f"Failed to initialize Causal-Learn: {e}")
```

**Key Points**:
- ✅ Import wrapped in try/except
- ✅ `INTEGRATIONS_AVAILABLE` flag
- ✅ Checks both config and module availability
- ✅ Logs errors but doesn't crash

---

### 6. Integrations Package

**File**: `knowledge_engine/integrations/__init__.py`

```python
try:
    from .causal_learn_integration import (
        CausalLearnIntegration,
        CausalDiscoveryEngine
    )
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    CausalLearnIntegration = None
    CausalDiscoveryEngine = None

__all__ = [
    # ... other exports
    "CausalLearnIntegration",
    "CausalDiscoveryEngine",
    "CAUSAL_LEARN_AVAILABLE",
]
```

**Key Points**:
- ✅ Optional import with availability flag
- ✅ Exports available even when import fails (as None)
- ✅ Other integrations unaffected

---

## Behavior Matrix

| Scenario | Integration Module | Causal-Learn Library | System Behavior |
|----------|-------------------|---------------------|-----------------|
| Both available | ✅ | ✅ | Full functionality |
| Integration only | ✅ | ❌ | Graceful degradation, informative messages |
| Neither | ❌ | ❌ | Mock components, system continues |
| Library only | ❌ | ✅ | N/A - integration is required |

---

## User Experience

### When Causal-Learn is NOT Installed

1. **Import Warnings** (during initialization):
   ```
   Note: Constraint-based algorithms not available: No module named 'causallearn'
   Note: Score-based algorithms not available: No module named 'causallearn'
   Warning: No causal-learn algorithms could be loaded
   ```

2. **Runtime Behavior**:
   - All causal methods return error status
   - System continues operating with other components
   - No exceptions raised

3. **Example Output**:
   ```python
   >>> integration = CausalLearnIntegration()
   >>> integration.is_available()
   False
   
   >>> result = integration.discover_structure(data, algorithm='pc')
   >>> result
   {'status': 'error', 'message': 'Causal-learn integration not available', 'graph': None}
   ```

### Installing Causal-Learn

```bash
pip install causal-learn
```

After installation:
- No code changes required
- System automatically detects library
- Full causal discovery functionality available

---

## Verification Commands

### Test Without Causal-Learn

```bash
# Ensure causal-learn is not installed
pip uninstall causal-learn -y

# Run optionality test
python test_causal_learn_optional.py
```

Expected: All tests pass ✅

### Test With Causal-Learn

```bash
# Install causal-learn
pip install causal-learn

# Run integration test
python test_causal_learn_full_integration.py
```

Expected: Full functionality available ✅

---

## Files Modified for Optionality

| File | Change |
|------|--------|
| `knowledge_engine/master_engine.py` | Added try/except for import, conditional initialization |
| `knowledge_engine/integrations/__init__.py` | Added optional export with availability flag |
| `knowledge_engine/unified_kg_integration_hub.py` | Added try/except in `_init_causal_learn` method |

**Files Already Optional** (no changes needed):
- `knowledge_engine/integrations/causal_learn_integration.py` - Already uses runtime imports
- `knowledge_engine/integrations/unified_knowledge_extraction.py` - Already has try/except
- `knowledge_engine/advanced_analytics_engine.py` - Already has try/except
- `knowledge_engine/orchestration/knowledge_orchestrator.py` - Already handles missing components

---

## Conclusion

**Causal-learn is fully optional.** The system:

1. ✅ Works without the causal-learn library installed
2. ✅ Provides informative messages about missing library
3. ✅ Continues operating with other knowledge engine components
4. ✅ Automatically enables full functionality when library is installed
5. ✅ No configuration changes required when installing/uninstalling

**Recommendation**: Users can install `causal-learn` when they need causal discovery capabilities, but the system will work correctly without it.
