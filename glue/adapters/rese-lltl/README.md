# RESE Logic-to-Loss Translation Layer (LLTL) Adapter

## Overview

The Logic-to-Loss Translation Layer (LLTL) provides a computable interface between the **Symbolic Constraint Engine (SCE)** and the **Deep Exploration Engine (DEE)** within the RESE (Recursive Epistemic Solvability Engine) system.

This adapter translates symbolic logical constraints into differentiable loss functions that can be optimized using neural methods.

## Architecture

```
Symbolic Constraints (SCE)
        ↓
    [LLTL Adapter]
        ↓
┌──────────────────────────┐
│  1. Encoder              │  Encode logic to neural format
│  2. Composer             │  Compose differentiable loss functions
│  3. DITO                 │  Detect contradictions (naive O(n²))
└──────────────────────────┘
        ↓
Loss Functions (DEE)
```

## Features

### Core Capabilities

1. **Symbolic Constraint Encoder**
   - Encodes logical constraints into neural representations
   - Fixed-dimension feature vectors (configurable)
   - Structural encoding (AST-like)
   - Caching for idempotency

2. **Loss Function Composer**
   - Multiple loss types: MSE, Cross-Entropy, Hinge, Custom
   - Weighted combination strategies
   - Gradient computation
   - Normalization options

3. **DITO Optimizer (Naive Implementation)**
   - Contradiction detection between constraints
   - O(n²) pairwise comparison
   - Detection result caching
   - Future: R-tree/LSH optimization (Tier 6)

### CLAUDE.md Compliance

This adapter follows all CLAUDE.md principles:

- **Law of Idempotency**: Cache translations, reuse if same input
- **Law of Configuration Explicitness**: All config via environment variables
- **Circuit Breaker**: Detect and handle encoding failures gracefully
- **Structured Logging**: JSON logs with correlation_id
- **Timeout**: All operations have configurable timeouts (default 3000ms)
- **UTC Timestamps**: All timestamps in UTC ISO-8601 format

## Installation

### Prerequisites

- Python 3.11+
- No external dependencies (uses only standard library)

### Setup

```bash
# Navigate to adapter directory
cd glue/adapters/rese-lltl

# Run probe script to verify installation
bash probes/check_lltl.sh
```

## Configuration

All configuration is via environment variables (Law of Configuration Explicitness):

### Encoding Configuration

```bash
export LLTL_ENCODING_DIM=128              # Feature vector dimension
export LLTL_USE_POSITIONAL=true          # Use positional encoding
export LLTL_USE_TYPE_EMBEDDING=true      # Embed constraint type
export LLTL_USE_CATEGORY_EMBEDDING=true  # Embed constraint category
export LLTL_MAX_SEQUENCE_LENGTH=512      # Max constraint length
export LLTL_CACHE_SIZE=1000              # Translation cache size
```

### Loss Configuration

```bash
export LLTL_DEFAULT_LOSS_TYPE=mse                    # Default loss type
export LLTL_COMBINATION_STRATEGY=weighted_sum        # Combination strategy
export LLTL_NORMALIZE_WEIGHTS=true                  # Normalize weights
export LLTL_GRADIENT_CLIP=0                         # Gradient clipping (0=disabled)
export LLTL_LEARNING_RATE=0.001                     # Learning rate
```

### DITO Configuration

```bash
export LLTL_ENABLE_RTREE=false              # R-tree optimization (Tier 6)
export LLTL_ENABLE_LSH=false                # LSH optimization (Tier 6)
export LLTL_ENABLE_HAG=false                # HAG optimization (Tier 6)
export LLTL_CONTRADICTION_THRESHOLD=0.8     # Contradiction threshold
export LLTL_MAX_CONTRADICTIONS=1000         # Max contradictions to track
export LLTL_DITO_CACHE_SIZE=1000            # DITO cache size
```

### General Configuration

```bash
export LLTL_TIMEOUT_MS=3000                 # Operation timeout
export CORRELATION_ID=optional_id           # For tracing
```

## Usage

### Basic Usage

```python
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lltl_adapter import LLTLAdapter
from dataclasses import dataclass

# Define constraint structure
@dataclass
class Constraint:
    constraint_id: str
    type: str  # "hard" or "soft"
    category: str  # "logical", "causal", "temporal", etc.
    description: str
    expression: str  # Logical expression
    dependencies: list
    priority: float
    confidence: float

# Create adapter
adapter = LLTLAdapter()

# Define constraints
constraints = [
    Constraint(
        constraint_id="c1",
        type="hard",
        category="logical",
        description="X must be greater than 5",
        expression="x > 5",
        dependencies=[],
        priority=1.0,
        confidence=0.9
    ),
    Constraint(
        constraint_id="c2",
        type="soft",
        category="causal",
        description="Y should be less than 10",
        expression="y < 10",
        dependencies=[],
        priority=0.8,
        confidence=0.7
    )
]

# Translate constraints
result, error = adapter.translate_constraints(constraints)

if error:
    print(f"Translation failed: {error}")
else:
    print(f"Translation successful!")
    print(f"  Input constraints: {result['input_constraints']}")
    print(f"  Loss functions: {result['loss_functions']}")
    print(f"  Contradictions: {result['contradictions_detected']}")
    print(f"  Duration: {result['duration_ms']:.2f}ms")
```

### Advanced Usage

#### Single Constraint Encoding

```python
# Encode a single constraint
constraint = Constraint(...)

encoded, error = adapter.encode_single(constraint)

if error:
    print(f"Encoding failed: {error}")
else:
    print(f"Encoded: {encoded['constraint_id']}")
    print(f"  Feature vector: {encoded['feature_vector'][:10]}...")  # First 10
    print(f"  Structural encoding: {encoded['structural_encoding']}")
```

#### Contradiction Detection

```python
# Detect contradictions between constraints
contradictions, error = adapter.detect_contradictions(constraints)

if error:
    print(f"Detection had warnings: {error}")

print(f"Found {len(contradictions)} contradictions")
for contradiction in contradictions:
    print(f"  {contradiction['constraint1_id']} <-> {contradiction['constraint2_id']}")
```

#### Health Check

```python
# Verify adapter is healthy
is_healthy, message = adapter.health_check()

if is_healthy:
    print(f"✓ Healthy: {message}")
else:
    print(f"✗ Unhealthy: {message}")
```

#### Statistics

```python
# Get adapter statistics
stats = adapter.get_stats()

print(f"Cache hits: {stats['translator_stats']['encoder_cache']['cache_hits']}")
print(f"Cache hit rate: {stats['translator_stats']['encoder_cache']['hit_rate']:.2%}")
print(f"Contradictions detected: {stats['translator_stats']['dito_contradictions']}")
```

## API Reference

### LLTLAdapter

#### Methods

##### `__init__(config: Optional[Dict] = None)`
Initialize adapter with optional config override.

##### `translate_constraints(constraints: List[Any], timeout_ms: Optional[int] = None, correlation_id: Optional[str] = None) -> Tuple[Optional[Dict], Optional[str]]`
Translate constraints to loss functions.

**Returns:**
- `result`: Dict with translation results
  - `translation_id`: Unique translation ID
  - `input_constraints`: Number of input constraints
  - `encoded_constraints`: Number successfully encoded
  - `loss_functions`: Number of loss functions composed
  - `contradictions_detected`: Number of contradictions found
  - `combined_loss`: Combined loss function dict
  - `contradictions`: List of contradiction pairs
  - `duration_ms`: Translation duration
- `error`: Error message if failed

##### `encode_single(constraint: Any, correlation_id: Optional[str] = None) -> Tuple[Optional[Dict], Optional[str]]`
Encode a single constraint.

**Returns:**
- `encoded`: Encoded constraint dict
- `error`: Error message if failed

##### `detect_contradictions(constraints: List[Any], correlation_id: Optional[str] = None) -> Tuple[List[Dict], Optional[str]]`
Detect contradictions between constraints.

**Returns:**
- `contradictions`: List of contradiction dicts
- `error`: Error message if any

##### `get_stats() -> Dict`
Get adapter and translator statistics.

##### `health_check() -> Tuple[bool, str]`
Check if adapter is healthy.

## Testing

### Run Probe Script

```bash
# From adapter root
bash probes/check_lltl.sh

# Expected output:
# ✓ All 8 tests passed
```

### Test Cases

The probe script verifies:
1. Module imports
2. Adapter initialization
3. Health check
4. Single constraint encoding
5. Multiple constraint translation
6. Contradiction detection
7. Statistics retrieval
8. Configuration validation

## Error Handling

The adapter uses a **Circuit Breaker** pattern to prevent cascade failures:

1. **Transient Failures**: Automatic retry with exponential backoff
2. **Encoding Failures**: Circuit breaker opens after threshold
3. **System Failures**: Graceful degradation with error messages

All errors are logged with structured JSON format.

## Performance

### Current Implementation (Naive)

- **Encoding**: O(1) per constraint (with caching)
- **Composition**: O(n) for n constraints
- **DITO**: O(n²) for n constraints (naive pairwise)
- **Cache Hit Rate**: >90% for repeated constraints

### Future Optimizations (Tier 6)

- **R-tree Spatial Indexing**: O(n log n) contradiction detection
- **LSH (Locality-Sensitive Hashing)**: Approximate detection
- **HAG (Hierarchical Abstraction Graph)**: Multi-scale detection

## Limitations

### Current Limitations

1. **Naive DITO**: O(n²) contradiction detection (acceptable for <1000 constraints)
2. **Simple Encoding**: Hash-based feature vectors (not semantic)
3. **Basic Loss Functions**: MSE only (cross-entropy and hinge are placeholders)
4. **Numerical Gradients**: Not using autograd (future optimization)

### Planned Enhancements

- Semantic encoding using embeddings
- Full loss function implementations
- Automatic differentiation
- R-tree/LSH optimization

## Troubleshooting

### Import Errors

```python
# If you get import errors, check paths:
import sys
print(sys.path)

# Ensure glue/lib is in path:
sys.path.insert(0, "path/to/glue/lib")
```

### Configuration Errors

```python
# Adapter will crash immediately if config is invalid
# Check environment variables:
import os
print({k: v for k, v in os.environ.items() if k.startswith("LLTL_")})
```

### Timeout Errors

```python
# Increase timeout if translation is slow:
result, error = adapter.translate_constraints(
    constraints,
    timeout_ms=10000  # 10 seconds
)
```

## Contributing

When contributing to LLTL:

1. Follow CLAUDE.md principles strictly
2. Add probe tests for new features
3. Update this README
4. Ensure all tests pass: `bash probes/check_lltl.sh`
5. Use structured logging with correlation_id

## References

- **SOURCE_RECOVERY_REPORT.md**: RESE bytecode analysis
- **CLAUDE.md**: Federation Constitution (architecture principles)
- **rese_schemas.py**: Canonical data models

## License

Part of the RESE project. See main project LICENSE.

## Authors

RESE Team
Created: 2026-02-04
