# Causal-Learn Integration Summary

## What Was Done

This integration updates the knowledge engine's `causal_modeling.py` to use the existing `CausalLearnAdapter` from `integrations/causal_learn/`, following the **Law of the Air Gap** principle.

## Files Updated

### Core Integration
1. **`knowledge_engine/causal_modeling.py`** - Main integration
   - Imports existing `CausalLearnAdapter`
   - Delegates causal discovery to adapter
   - Adds knowledge-engine specific features:
     - Persistent storage (Neo4j, Qdrant)
     - Model versioning
     - Cross-domain learning
     - Counterfactual queries
   - Graceful fallback when causal-learn unavailable

### Schemas
2. **`knowledge_engine/schemas/long_horizon.py`** - Added schema classes
   - `StoredCausalModel` - For persistent storage
   - `CounterfactualResult` - For counterfactual queries

### Configuration
3. **`knowledge_engine/config/causal_config.yaml`** - Configuration file
   - Algorithm selection
   - Independence test configuration
   - Storage settings
   - Performance tuning

4. **`knowledge_engine/config/__init__.py`** - Configuration loader
   - YAML config loading
   - Environment variable overrides
   - Default values

### Testing
5. **`knowledge_engine/tests/test_causal_modeling_integration.py`** - Comprehensive tests
   - Basic functionality tests
   - Causal-learn integration tests
   - Algorithm comparison tests
   - Persistence tests
   - Graceful degradation tests

### Documentation
6. **`knowledge_engine/CAUSAL_MODELING.md`** - Full documentation
   - Architecture overview
   - Usage examples
   - Algorithm reference
   - Troubleshooting guide

7. **`knowledge_engine/examples/causal_modeling_quickstart.py`** - Quickstart example
   - 6 complete examples
   - Synthetic data generation
   - All major features demonstrated

## Key Features

### 1. Delegation to Existing Adapter
```python
# OLD: Reimplemented causal discovery
relationships = await self._discover_causes(data, factors, outcomes)

# NEW: Delegates to CausalLearnAdapter
if self.use_causal_learn and self.adapter:
    relationships = await self._discover_with_causal_learn(
        data, factors, outcomes, method
    )
```

### 2. Graceful Degradation
```python
# Falls back if causal-learn unavailable
if CAUSAL_LEARN_INTEGRATION_AVAILABLE:
    self.adapter = CausalLearnAdapter()
    self.use_causal_learn = True
else:
    self.adapter = None
    self.use_causal_learn = False
    logger.warning("Using fallback implementation")
```

### 3. Knowledge Engine Features
```python
# Persistent storage
await builder.store_model(model, version=1)

# Model updates
updated_model = await builder.update_model(model, new_data)

# Cross-domain learning
suggested = await builder.transfer_causal_knowledge(
    source_domain="finance",
    target_domain="trading"
)
```

## Architecture

```
Knowledge Engine (causal_modeling.py)
    │
    ├─► CausalLearnAdapter (integrations/causal_learn/)
    │       │
    │       └─► causal-learn library (core-projects/)
    │
    ├─► Storage
    │       ├─► Neo4j (graph structure)
    │       └─► Qdrant (similarity search)
    │
    └─► Fallback (correlation-based)
```

## Usage

### Basic
```python
from knowledge_engine.causal_modeling import CausalModelBuilder

builder = CausalModelBuilder(knowledge_engine=ke)
model = await builder.build_model(
    domain="finance",
    outcomes=outcomes,
    method="pc"
)
```

### With Storage
```python
# Store persistently
model_id = await builder.store_model(model, version=1)

# Load later
loaded = await builder.load_model(model_id, domain="finance")
```

### Cross-Domain
```python
# Transfer knowledge
suggested = await builder.transfer_causal_knowledge(
    source_domain="finance",
    target_domain="trading"
)
```

## Configuration

Create `knowledge_engine/config/causal_config.yaml`:

```yaml
use_causal_learn: true
default_algorithm: "pc"
alpha: 0.05
storage:
  neo4j:
    enabled: true
  qdrant:
    enabled: true
```

## Testing

```bash
# Run all tests
pytest knowledge_engine/tests/test_causal_modeling_integration.py -v

# Run examples
python knowledge_engine/examples/causal_modeling_quickstart.py
```

## Compliance with Laws

### ✅ Law of the Air Gap
- Imports from `integrations/causal_learn/`, not `core-projects/`
- No direct dependency on causal-learn source

### ✅ Law of Runtime Truth
- Executes actual causal-learn algorithms
- Doesn't rely on documentation

### ✅ Law of Idempotency
- `store_model()` safe to call multiple times
- `update_model()` idempotent

### ✅ Law of Configuration Explicitness
- All settings in YAML config
- Environment variable overrides supported

### ✅ Law of UTC
- All timestamps in UTC
- Proper ISO-8601 format

## Next Steps

1. **Install causal-learn** (if not already):
   ```bash
   pip install causal-learn
   ```

2. **Run quickstart**:
   ```bash
   python knowledge_engine/examples/causal_modeling_quickstart.py
   ```

3. **Configure storage** (optional):
   - Set up Neo4j for graph storage
   - Set up Qdrant for similarity search

4. **Use in your agents**:
   ```python
   from knowledge_engine.integrations.unified_evolution_integration import (
       UnifiedEvolutionKnowledgeExtractor
   )

   extractor = UnifiedEvolutionKnowledgeExtractor(knowledge_engine=ke)
   causal_model = await extractor.build_causal_model(domain, outcomes)
   ```

## Benefits

1. **Well-Tested Algorithms**: Uses proven causal-learn implementation
2. **No Duplication**: Doesn't reimplement causal discovery
3. **Graceful Degradation**: Works even if causal-learn unavailable
4. **Persistence**: Stores models for reuse
5. **Cross-Domain**: Transfers knowledge across domains
6. **Standards Compliant**: Follows OpenEvolve Constitution

## Support

- Documentation: `knowledge_engine/CAUSAL_MODELING.md`
- Examples: `knowledge_engine/examples/causal_modeling_quickstart.py`
- Tests: `knowledge_engine/tests/test_causal_modeling_integration.py`
- causal-learn docs: https://causal-learn.readthedocs.io/
