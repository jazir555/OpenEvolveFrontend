# Causal Modeling Integration Guide

## Overview

The knowledge engine's causal modeling system integrates the existing `causal-learn` adapter to provide sophisticated causal discovery capabilities from agent outcomes. This integration follows the **Law of the Air Gap** - it delegates causal discovery to the well-tested `CausalLearnAdapter` rather than reimplementing algorithms.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Knowledge Engine                         │
│                  (CausalModelBuilder)                        │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Knowledge Engine Features                           │   │
│  │  • Persistent storage (Neo4j, Qdrant)                 │   │
│  │  • Model versioning                                   │   │
│  │  • Cross-domain learning                              │   │
│  │  • Counterfactual queries                             │   │
│  └───────────────────┬─────────────────────────────────┘   │
│                      │ Delegates to                         │
│  ┌───────────────────▼─────────────────────────────────┐   │
│  │  CausalLearnAdapter (integrations/causal_learn/)    │   │
│  │                                                       │   │
│  │  Algorithms:                                         │   │
│  │  • PC (Peter-Clark)                                  │   │
│  │  • GES (Greedy Equivalence Search)                   │   │
│  │  • FCI (Fast Causal Inference)                       │   │
│  │  • DirectLiNGAM (non-Gaussian)                       │   │
│  │                                                       │   │
│  │  Independence Tests:                                  │   │
│  │  • Fisher Z, Chi-square, G-square, KCI               │   │
│  └───────────────────┬─────────────────────────────────┘   │
│                      │ Uses                                 │
│  ┌───────────────────▼─────────────────────────────────┐   │
│  │  causal-learn Library                                │   │
│  │  (Third-party, in core-projects/)                    │   │
│  └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Key Principles

### 1. Law of the Air Gap
The knowledge engine does NOT import from `core-projects/causal-learn/`. Instead, it imports from `integrations/causal_learn/adapter.py`, which wraps the causal-learn library.

### 2. Law of Runtime Truth
Causal discovery is performed by executing actual causal-learn algorithms, not by reading documentation. The `CausalLearnAdapter` provides verified execution.

### 3. Graceful Degradation
If causal-learn is unavailable, the system falls back to a simplified correlation-based approach with appropriate warnings.

### 4. Law of Idempotency
All operations (building, storing, updating) are safe to run multiple times.

## Usage

### Basic Usage

```python
from knowledge_engine.causal_modeling import CausalModelBuilder

# Initialize builder
builder = CausalModelBuilder(knowledge_engine=ke)

# Build causal model from outcomes
outcomes = [
    {
        "context": {"exploration_rate": 0.5, "population_size": 100},
        "metrics": {"fitness": 0.8, "diversity": 0.6},
        "timestamp": "2026-01-30T12:00:00Z"
    },
    # ... more outcomes
]

model = await builder.build_model(
    domain="finance",
    outcomes=outcomes,
    method="pc"  # Use PC algorithm
)

# Examine discovered relationships
for rel in model.relationships:
    print(f"{rel.cause} -> {rel.effect}")
    print(f"  Strength: {rel.strength:.3f}")
    print(f"  Confidence: {rel.confidence:.3f}")
    print(f"  Mechanism: {rel.mechanism}")
```

### Intervention Prediction

```python
# Predict effect of intervention
prediction = await builder.predict_intervention(
    model=model,
    cause="exploration_rate",
    value=0.8
)

print(f"Intervention: {prediction.intervention}")
print(f"Predicted effect: {prediction.predicted_effect:.3f}")
print(f"Confidence: {prediction.confidence:.3f}")

# Check alternative outcomes
for alt, effect in prediction.alternative_outcomes:
    print(f"  {alt}: {effect:.3f}")
```

### Outcome Explanation

```python
# Explain why an outcome occurred
explanation = await builder.explain_outcome(
    model=model,
    outcome="fitness"
)

print(f"Explaining: {explanation.outcome}")
print(f"Causes: {explanation.causes}")
print(f"Contributions:")
for cause, contrib in explanation.contribution.items():
    print(f"  {cause}: {contrib:.3f}")
print(f"Confidence: {explanation.confidence:.3f}")

# View counterfactuals
print("Counterfactuals:")
for cf in explanation.counterfactuals:
    print(f"  - {cf}")
```

### Persistent Storage

```python
# Store model in knowledge engine
model_id = await builder.store_model(model, version=1)

# Load model later
loaded_model = await builder.load_model(
    model_id=model_id,
    domain="finance"
)
```

### Model Updates

```python
# Update model with new data (incremental learning)
new_outcomes = [
    # ... new observations
]

updated_model = await builder.update_model(
    model=model,
    new_data=new_outcomes
)
```

### Cross-Domain Learning

```python
# Transfer causal knowledge across domains
suggested_rels = await builder.transfer_causal_knowledge(
    source_domain="finance",
    target_domain="trading",
    min_similarity=0.7
)

print("Suggested relationships for trading domain:")
for rel in suggested_rels:
    print(f"  {rel.cause} -> {rel.effect}")
    print(f"    Confidence: {rel.confidence:.3f} (transferred)")
```

## Configuration

Create `knowledge_engine/config/causal_config.yaml`:

```yaml
# Use causal-learn integration
use_causal_learn: true

# Default algorithm
default_algorithm: "pc"

# Algorithm selection based on data size
algorithm_selection:
  small_data: "pc"        # < 100 samples
  medium_data: "ges"      # 100-1000 samples
  large_data: "direct_lingam"  # > 1000 samples

# Independence test
independence_test:
  default: "fisherz"

# Significance level
alpha: 0.05

# Persistence
storage:
  neo4j:
    enabled: true
  qdrant:
    enabled: true
  version_models: true
```

## Algorithms

### PC (Peter-Clark)
**Best for:** Continuous Gaussian data, small to medium datasets

- **Type:** Constraint-based
- **Advantages:** Well-established, interpretable
- **Disadvantages:** Sensitive to independence test errors
- **Parameters:** `alpha` (significance level), `indep_test`

### GES (Greedy Equivalence Search)
**Best for:** Medium to large datasets

- **Type:** Score-based
- **Advantages:** Faster than PC for large data, statistically sound
- **Disadvantages:** May miss some edges
- **Parameters:** `score_func` (BIC, BDeu, CV)

### FCI (Fast Causal Inference)
**Best for:** Data with latent confounders

- **Type:** Constraint-based with latent variables
- **Advantages:** Detects latent confounders (bidirected edges)
- **Disadvantages:** More conservative (fewer directed edges)
- **Parameters:** `alpha`, `indep_test`

### DirectLiNGAM
**Best for:** Non-Gaussian data

- **Type:** Functional causal model
- **Advantages:** Handles non-Gaussian data, provides causal order
- **Disadvantages:** Assumes linear relationships
- **Parameters:** None

## Independence Tests

### Fisher Z
- **Data type:** Continuous Gaussian
- **Use case:** Most continuous data

### Chi-square
- **Data type:** Discrete
- **Use case:** Categorical variables

### G-square
- **Data type:** Discrete
- **Use case:** Alternative to chi-square

### KCI (Kernel-based Conditional Independence)
- **Data type:** Continuous, nonlinear
- **Use case:** Nonlinear relationships

## Integration with Unified Evolution Integration

The `UnifiedEvolutionKnowledgeExtractor` uses causal modeling:

```python
from knowledge_engine.integrations.unified_evolution_integration import (
    UnifiedEvolutionKnowledgeExtractor
)

extractor = UnifiedEvolutionKnowledgeExtractor(knowledge_engine=ke)

# Build causal model from outcomes
causal_model = await extractor.build_causal_model(
    domain="finance",
    outcomes=outcomes
)

# Predict intervention effect
effect = await extractor.predict_intervention_effect(
    domain="finance",
    cause="exploration_rate",
    value=0.8
)

# Explain outcome
explanation = await extractor.explain_outcome(
    domain="finance",
    outcome="fitness"
)
```

## Testing

Run tests with:

```bash
# Run all causal modeling tests
pytest knowledge_engine/tests/test_causal_modeling_integration.py -v

# Run only causal-learn integration tests
pytest knowledge_engine/tests/test_causal_modeling_integration.py::TestCausalLearnIntegration -v

# Run only basic tests (skip causal-learn specific)
pytest knowledge_engine/tests/test_causal_modeling_integration.py::TestCausalModelBuilder -v -m "not integration"
```

## Troubleshooting

### Causal-learn not available

**Warning:**
```
causal-learn integration not available: No module named 'causal_learn'
Using simplified fallback implementation
```

**Solution:**
```bash
pip install causal-learn
```

### No relationships discovered

**Possible causes:**
1. Sample size too small
2. Alpha too strict (try 0.1 instead of 0.05)
3. Variables not causally related
4. Wrong independence test for data type

**Solutions:**
```python
# Try different algorithm
model = await builder.build_model(
    domain="test",
    outcomes=outcomes,
    method="ges"  # Instead of "pc"
)

# Try less strict alpha
model = await builder.build_model(
    domain="test",
    outcomes=outcomes,
    method="pc",
    alpha=0.1  # Instead of 0.05
)

# Try different independence test
model = await builder.build_model(
    domain="test",
    outcomes=outcomes,
    method="pc",
    indep_test="kci"  # For nonlinear relationships
)
```

### Model not persisting

**Check:**
1. Knowledge engine initialized with Neo4j/Qdrant
2. Configuration has storage enabled
3. Database connections working

**Debug:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

builder = CausalModelBuilder(knowledge_engine=ke)
# Check storage availability
print(f"Neo4j: {builder.neo4j}")
print(f"Qdrant: {builder.qdrant}")
```

## Performance Tips

1. **For large datasets (>1000 samples):** Use GES or DirectLiNGAM instead of PC
2. **For categorical data:** Use chi-square or G-square independence test
3. **For nonlinear relationships:** Use KCI independence test
4. **Enable caching:** Reduces repeated computation
5. **Store models:** Avoid re-discovery with persistence

```python
# Fast configuration for large data
builder = CausalModelBuilder(
    discovery_method="ges",  # Faster for large data
    min_confidence=0.6  # Less strict
)

await builder.adapter.initialize({
    'cache_enabled': True,  # Enable caching
    'performance': {
        'max_workers': 4,  # Parallel processing
        'timeout': 600  # 10 minutes
    }
})
```

## Advanced Usage

### Custom Algorithm Parameters

```python
model = await builder.build_model(
    domain="test",
    outcomes=outcomes,
    method="pc",
    alpha=0.01,  # Very strict
    indep_test="fisherz",
    stable=True  # Use stable PC
)
```

### Model Versioning

```python
# Store version 1
v1_id = await builder.store_model(model, version=1)

# Update and store version 2
v2_model = await builder.update_model(model, new_data)
v2_id = await builder.store_model(v2_model, version=2)

# Compare versions
changes = builder._detect_model_changes(model, v2_model)
print(f"Added: {changes['added_relationships']}")
print(f"Removed: {changes['removed_relationships']}")
```

### Counterfactual Analysis

```python
result = await builder.query_counterfactual(
    model=model,
    intervention={"exploration_rate": 0.9},
    outcome="fitness"
)

print(f"Predicted: {result['predicted_value']:.3f}")
print(f"Effect: {result['effect']:.3f}")
print(f"Confidence: {result['confidence']:.3f}")
```

## References

- causal-learn documentation: https://causal-learn.readthedocs.io/
- PC Algorithm: Spirtes, Glymour, Scheines (2000)
- GES Algorithm: Chickering (2002)
- LiNGAM: Shimizu et al. (2006)
- FCI Algorithm: Spirtes et al. (1995)

## Contributing

When adding features:

1. **Use existing adapter:** Don't reimplement causal discovery
2. **Add fallback:** Ensure graceful degradation
3. **Test both paths:** Test with and without causal-learn
4. **Document configuration:** Add options to causal_config.yaml
5. **Follow laws:** Adhere to Constitution principles

## License

This integration follows the same license as the OpenEvolve project.
