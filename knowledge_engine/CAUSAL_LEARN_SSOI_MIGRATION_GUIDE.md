# Causal-Learn Integration SSOT Migration Guide

## Overview

This document describes the consolidation of multiple causal-learn integration implementations into a Single Source of Truth (SSOT).

## Problem Statement

Previously, there were **3 separate implementations** of causal-learn integration:

1. **`integrations/causal_learn/`** (87 KB total)
   - `adapter.py` (32 KB) - Main adapter implementing CausalDiscoveryInterface
   - `bridge.py` (25 KB) - Bridge to OpenEvolve systems
   - `config.yaml` (9 KB) - Configuration
   - `__init__.py` (5 KB) - Package exports

2. **`knowledge_engine/integrations/causal_learn_integration.py`** (27 KB)
   - Separate implementation with CausalLearnIntegration and CausalDiscoveryEngine
   - Duplicated functionality from SSOT

3. **`docs/knowledge_engine/knowledge_engine/causal_modeling.py`**
   - Already imports from SSOT (correct pattern)

## Solution: Single Source of Truth (SSOT)

### SSOT Location
```
integrations/causal_learn/          # Single Source of Truth
├── __init__.py                     # Package exports
├── adapter.py                      # CausalLearnAdapter (main implementation)
├── bridge.py                       # CausalDiscoveryBridge
└── config.yaml                     # Configuration

integrations/base/
└── causal_interface.py             # Abstract interface definitions
```

### Wrapper Location (Thin Wrapper/Re-export)
```
knowledge_engine/integrations/
└── causal_learn_integration.py     # Thin wrapper for KE context
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Consumers                                   │
├─────────────────────────────────────────────────────────────────┤
│  Unified Hub    │  Causal Modeling  │  SOP Generator           │
│  (knowledge_    │  (docs/knowledge_ │  (integrations/          │
│   engine)       │   engine)         │   causal_learn/)         │
└────────┬────────┴─────────┬─────────┴──────────┬───────────────┘
         │                  │                    │
         │         ┌────────┴────────────────────┘
         │         │
         ▼         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Thin Wrapper (knowledge_engine/integrations/                   │
│               causal_learn_integration.py)                      │
│  - Re-exports SSOT components                                   │
│  - Provides simplified API for KE context                       │
│  - Maintains backward compatibility                             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  SSOT (integrations/causal_learn/)                              │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Adapter    │  │    Bridge    │  │     Interface        │  │
│  │ CausalLearn  │  │ CausalDisc.  │  │  (base/causal_)      │  │
│  │   Adapter    │  │    Bridge    │  │    interface.py)     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────┘  │
│         │                 │                                     │
│         └─────────────────┘                                     │
│                   │                                             │
│                   ▼                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              causal-learn Library                        │   │
│  │  (core-projects/causal-learn/)                          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Migration Guide

### For New Code

Use the SSOT directly for maximum flexibility:

```python
# Option 1: Use SSOT directly (recommended for advanced use)
from integrations.causal_learn import CausalLearnAdapter, CausalDiscoveryBridge

adapter = CausalLearnAdapter()
await adapter.initialize(config)
result = await adapter.discover_causal_structure(data, method='pc')
```

```python
# Option 2: Use knowledge engine wrapper (simplified API)
from knowledge_engine.integrations.causal_learn_integration import CausalLearnIntegration

causal = CausalLearnIntegration()
await causal.initialize()
result = causal.discover_structure(data, algorithm='pc')
```

```python
# Option 3: Use unified hub (recommended for multi-integration workflows)
from knowledge_engine.unified_kg_integration_hub import UnifiedKGIntegrationHub

hub = UnifiedKGIntegrationHub()
await hub.initialize()
result = await hub.discover_causal_structure(data, algorithm='pc')
```

### For Existing Code

No changes required! The wrapper maintains backward compatibility:

```python
# This still works (backward compatible)
from knowledge_engine.integrations.causal_learn_integration import (
    CausalLearnIntegration,
    CausalDiscoveryEngine
)

integration = CausalLearnIntegration()
result = integration.discover_structure(data, algorithm='pc')
```

## API Comparison

### SSOT (integrations/causal_learn/adapter.py)

```python
class CausalLearnAdapter:
    async def initialize(self, config: Dict[str, Any]) -> bool
    async def discover_causal_structure(self, data, method='pc', **kwargs) -> CausalGraphResult
    async def estimate_causal_effect(self, data, treatment, outcome) -> CausalEffectResult
    async def test_conditional_independence(self, data, x, y, conditioning_set) -> IndependenceTestResult
    async def analyze_confounders(self, data, treatment, outcome) -> ConfounderAnalysisResult
```

### Wrapper (knowledge_engine/integrations/causal_learn_integration.py)

```python
class CausalLearnIntegration:
    def is_available(self) -> bool
    async def initialize(self) -> bool
    def discover_structure(self, data, algorithm='pc', **kwargs) -> Dict[str, Any]
    def get_available_algorithms(self) -> List[str]
    def get_algorithm_info(self, algorithm: str) -> Dict[str, Any]
    def run_independence_test(self, data, x, y, **kwargs) -> Dict[str, Any]
    def get_status(self) -> Dict[str, Any]

class CausalDiscoveryEngine:
    # DEPRECATED: Use CausalLearnIntegration or CausalLearnAdapter
```

## Key Differences

| Feature | SSOT (Adapter) | Wrapper (Integration) |
|---------|---------------|----------------------|
| Return Type | Dataclasses (CausalGraphResult) | Dictionary |
| Async | Full async support | Mixed (sync wrapper) |
| Interface | Implements CausalDiscoveryInterface | Simplified API |
| Flexibility | Maximum | Simplified |
| Best For | Advanced users, custom workflows | Quick usage, hub integration |

## File Changes

### Modified Files

1. **`knowledge_engine/integrations/causal_learn_integration.py`**
   - Rewritten as thin wrapper around SSOT
   - Maintains backward compatibility
   - Delegates all operations to CausalLearnAdapter
   - Deprecated CausalDiscoveryEngine (use CausalLearnIntegration)

2. **`knowledge_engine/unified_kg_integration_hub.py`**
   - Updated `_initialize_causal_learn()` to use SSOT
   - Enhanced `discover_causal_structure()` with more parameters
   - Better error handling and metadata

3. **`knowledge_engine/tests/test_unified_kg_integrations.py`**
   - Added SSOT import tests
   - Added `get_ssot_info()` test
   - Added algorithm enumeration tests

## Configuration

### SSOT Configuration (`integrations/causal_learn/config.yaml`)

```yaml
algorithms:
  default: 'pc'
  pc:
    alpha: 0.05
    indep_test: 'fisherz'
    stable: true
  ges:
    score_func: 'local_score_BIC'
  directlingam:
    bootstrap: false

features:
  causal_discovery: true
  causal_effect_estimation: true
  independence_testing: true
  counterfactual_analysis: false
  intervention_optimization: false

integration:
  auto_start: true
  cache_enabled: true
  cache_ttl: 3600
  fallback_on_error: true
```

### Wrapper Configuration

```python
from knowledge_engine.integrations.causal_learn_integration import CausalLearnIntegration

config = {
    'default_algorithm': 'pc',
    'default_indep_test': 'fisherz',
    'default_alpha': 0.05,
    'cache_enabled': True
}

causal = CausalLearnIntegration(config)
```

## Testing

### Run Tests

```bash
# Test SSOT directly
python -c "from integrations.causal_learn import CausalLearnAdapter; print('SSOT OK')"

# Test wrapper
python -c "from knowledge_engine.integrations.causal_learn_integration import CausalLearnIntegration; print('Wrapper OK')"

# Run full test suite
pytest knowledge_engine/tests/test_unified_kg_integrations.py::TestCausalLearnIntegration -v
```

### Test Coverage

- ✅ SSOT imports
- ✅ Wrapper initialization
- ✅ Causal discovery (PC, GES, FCI, LiNGAM)
- ✅ Independence testing
- ✅ Algorithm enumeration
- ✅ Backward compatibility

## Troubleshooting

### Issue: "SSOT not available"

**Cause**: `integrations/causal_learn/` not in Python path

**Solution**:
```python
import sys
sys.path.insert(0, 'path/to/integrations')
```

### Issue: "causal-learn not available"

**Cause**: causal-learn library not installed

**Solution**:
```bash
pip install causal-learn
```

### Issue: Deprecated warnings for CausalDiscoveryEngine

**Cause**: Using deprecated class

**Solution**: Use `CausalLearnIntegration` or `CausalLearnAdapter` instead

## Benefits of SSOT

1. **Single Implementation**: Only one set of causal discovery logic to maintain
2. **Consistent Interface**: All consumers use the same core implementation
3. **Easier Testing**: One test suite covers all usage patterns
4. **Better Documentation**: Single source for all causal-learn functionality
5. **Reduced Bugs**: No risk of diverging implementations

## Migration Checklist

- [x] Identify duplicate implementations
- [x] Designate SSOT location (`integrations/causal_learn/`)
- [x] Create thin wrapper in knowledge_engine
- [x] Update unified hub to use SSOT
- [x] Maintain backward compatibility
- [x] Update tests
- [x] Create migration guide
- [ ] Deprecate old implementations (future)
- [ ] Update documentation references (future)

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-02 | Initial SSOT implementation in `integrations/causal_learn/` |
| 2.0.0 | 2026-02-03 | Merged duplicate implementations, created thin wrapper |

## Contact

For questions about the SSOT architecture, contact the Knowledge Engine team.
