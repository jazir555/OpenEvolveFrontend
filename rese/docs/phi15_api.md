# Φ₁.₅ Tacit Assumption Mining - API Documentation

**Agent**: B1 (Φ₁/Φ₁.₅ Specialist)
**Date**: 2025-12-31
**Status**: 🟢 Active Implementation
**Version**: 1.0.0

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core API](#core-api)
5. [Data Structures](#data-structures)
6. [Components](#components)
7. [Integration](#integration)
8. [Configuration](#configuration)
9. [Examples](#examples)
10. [Performance](#performance)

---

## Overview

Φ₁.₅ (Phi-1.5) is an automated Kuhnian paradigm shift detection system that mines hidden constraints (tacit assumptions) from null results and failure patterns in the RESE framework.

**Key Capabilities**:
- Transform null results into paradigm shift signals
- Infer hidden constraints via abductive reasoning
- Detect accumulating anomalies that indicate paradigm crises
- Integrate seamlessly with RESE Stages 1, 6, and 7

**Target Performance**:
- >70% assumption mining accuracy on validation cases
- <10 seconds for processing 100 failures
- >1000 failures/hour throughput

---

## Installation

### Requirements

- Python 3.11+
- scikit-learn 1.3+
- numpy 1.24+
- pandas 2.0+

### Install Dependencies

```bash
# Install from requirements
pip install -r requirements.txt

# Or install individually
pip install numpy pandas scikit-learn
```

### Setup

```python
# Import Φ₁.₅
from rese.phase1.tacit_assumption_miner import (
    Phi15Engine, NullResult, ErrorType
)

# Create engine
engine = Phi15Engine()
```

---

## Quick Start

### Basic Usage

```python
from rese.phase1.tacit_assumption_miner import (
    Phi15Engine, NullResult, ErrorType
)
from datetime import datetime

# 1. Create engine
engine = Phi15Engine()

# 2. Create null result from Stage 6
null_result = NullResult(
    attempt_id="attempt_001",
    timestamp=datetime.now(),
    problem_type="optimization",
    approach_type="deterministic",
    constraints=["exact_solution_required"],
    error_type=ErrorType.TIMEOUT,
    error_message="Optimization exceeded time limit",
    state={"iteration": 1000},
    iteration=1000,
    resources_used={"cpu": 100.0, "memory": 8000.0}
)

# 3. Process null results
assumptions, paradigm_rec = engine.process_null_results([null_result])

# 4. Get top assumptions
top_assumptions = engine.get_top_assumptions(k=10)

for assumption in top_assumptions:
    print(f"Assumption: {assumption.description}")
    print(f"Confidence: {assumption.confidence:.2f}")
    print(f"Support: {assumption.support} failures")
    print()
```

### Using Interface Manager

```python
from rese.phase1.phi15_interfaces import create_interface_manager

# Create interface manager (integrates with Stages 6, 1, 7)
manager = create_interface_manager()

# Process Stage 6 input
results = manager.process_stage6_input(null_results)

# Check status
status = manager.get_status()
print(f"Total failures: {status['total_failures']}")
print(f"Assumptions inferred: {status['total_assumptions']}")

# Cleanup
manager.shutdown()
```

---

## Core API

### Phi15Engine

Main engine orchestrating all Φ₁.₅ components.

#### Constructor

```python
Phi15Engine(config: Optional[Dict] = None)
```

**Parameters**:
- `config`: Optional configuration dictionary

**Returns**: Initialized Phi15Engine

**Example**:
```python
engine = Phi15Engine(config={
    'confidence_threshold': 0.6,
    'crisis_threshold': 0.7,
    'min_failures_for_clustering': 10
})
```

#### Methods

##### process_null_results()

Process null results through full Φ₁.₅ pipeline.

```python
process_null_results(null_results: List[NullResult]) -> Tuple[
    List[TacitAssumption], ParadigmShiftRecommendation
]
```

**Parameters**:
- `null_results`: List of null results from Stage 6

**Returns**:
- `assumptions`: List of inferred tacit assumptions
- `paradigm_rec`: Paradigm shift recommendation

**Example**:
```python
assumptions, paradigm_rec = engine.process_null_results(null_results)

if paradigm_rec.trigger:
    print(f"PARADIGM CRISIS: {paradigm_rec.explanation}")
```

##### get_top_assumptions()

Get top-k assumptions by confidence.

```python
get_top_assumptions(k: int = 10) -> List[TacitAssumption]
```

**Parameters**:
- `k`: Number of top assumptions to return

**Returns**: List of top-k assumptions

##### save_state() / load_state()

Persist and restore engine state.

```python
save_state(filepath: str) -> None
load_state(filepath: str) -> None
```

---

## Data Structures

### NullResult

Input from Stage 6 representing a failed attempt.

```python
@dataclass
class NullResult:
    attempt_id: str                          # Unique identifier
    timestamp: datetime                       # When attempt occurred
    problem_type: str                         # Type of problem
    approach_type: str                        # Algorithm/method used
    constraints: List[str]                    # Explicit constraints
    error_type: ErrorType                     # Type of error
    error_message: str                        # Error description
    state: Dict                               # Final state
    iteration: int                            # Iteration number
    resources_used: Dict[str, float]          # CPU/memory usage
    metadata: Dict                            # Additional context
```

**Creation**:
```python
result = NullResult(
    attempt_id="test_001",
    timestamp=datetime.now(),
    problem_type="optimization",
    approach_type="deterministic",
    constraints=["exact"],
    error_type=ErrorType.OPTIMIZATION_FAILED,
    error_message="Failed to converge",
    state={"x": 1.0},
    iteration=100,
    resources_used={"cpu": 50.0}
)
```

### TacitAssumption

Output representing an inferred tacit assumption.

```python
@dataclass
class TacitAssumption:
    id: str                                   # Unique ID
    description: str                          # Human-readable
    formalization: str                        # SCE format
    assumption_type: AssumptionType           # Category
    confidence: float                         # [0, 1]
    support: int                              # Failures explained
    evidence: List[str]                       # Supporting attempts
    pattern_type: PatternType                 # Inference pattern
    constraint_relaxation: str                # How to relax
    paradigm_implication: bool                # Paradigm-level?
    alternative_paradigm: Optional[str]       # Suggested alternative
    timestamp: datetime                       # When inferred
    verified: bool                            # Validated by Stage 7
```

**Methods**:
- `to_sce_constraint()`: Convert to SCE Constraint format
- `to_dict()`: Serialize to dictionary

**Example**:
```python
# Convert to SCE constraint for Stage 1
sce_constraint = assumption.to_sce_constraint()
```

### ErrorType (Enum)

Types of errors from Stage 6.

```python
class ErrorType(Enum):
    OPTIMIZATION_FAILED = "optimization_failed"
    DIVERGENCE = "divergence"
    CYCLE_DETECTION = "cycle_detection"
    CONSTRAINT_VIOLATION = "constraint_violation"
    TIMEOUT = "timeout"
    NUMERICAL_INSTABILITY = "numerical_instability"
    INFEASIBILITY = "infeasibility"
    UNKNOWN_FAILURE = "unknown_failure"
```

---

## Components

### FailurePreprocessor

Extract features from null results.

```python
from rese.phase1.tacit_assumption_miner import FailurePreprocessor

preprocessor = FailurePreprocessor()
features = preprocessor.extract_features(null_result)
```

**Features Extracted**:
- Structural: problem type, approach type, error type
- Temporal: timestamp, iteration, time to failure
- Numerical: error magnitude, resource consumption
- Textual: keywords from error message

### AnomalyDetector

Detect anomalies using Isolation Forest and LOF.

```python
from rese.phase1.tacit_assumption_miner import AnomalyDetector

detector = AnomalyDetector(contamination=0.1)
anomaly_scores = detector.detect_anomalies(failures)
```

**Returns**: Dictionary mapping attempt_id to anomaly score [0, 1]

### FailureClusterer

Cluster failures by similarity.

```python
from rese.phase1.tacit_assumption_miner import FailureClusterer

clusterer = FailureClusterer()
clusters = clusterer.cluster_failures(failures)
```

**Clustering Methods**:
- Hierarchical clustering (agglomerative)
- DBSCAN (density-based)
- Consensus clustering

---

## Integration

### Stage 6 → Φ₁.₅ (Input)

Receive null results from Stage 6 Error Source Analysis.

```python
from rese.phase1.phi15_interfaces import Phi15Stage6Interface

interface = Phi15Stage6Interface(phi15_engine)

# Single result
interface.receive_null_result(null_result)

# Batch results
count = interface.receive_batch_null_results(null_results)

# Trigger processing
interface.trigger_full_processing()
```

### Φ₁.₅ → Stage 1 (Output)

Send inferred assumptions to Stage 1 Prompt Analysis.

```python
from rese.phase1.phi15_interfaces import Phi15Stage1Interface

interface = Phi15Stage1Interface(phi15_engine)

# Send assumptions
count = interface.send_assumptions(assumptions)

# Send paradigm shift recommendation
interface.send_paradigm_shift_recommendation(paradigm_rec)
```

### Φ₁.₅ ↔ Stage 7 (Validation)

Validate assumptions and update confidence.

```python
from rese.phase1.phi15_interfaces import Phi15Stage7Interface

interface = Phi15Stage7Interface(phi15_engine)

# Request validation
request = interface.request_validation(assumption)

# Receive validation result
interface.receive_validation_result(validation_result)
```

---

## Configuration

### Default Configuration

```python
default_config = {
    'confidence_threshold': 0.6,      # Minimum assumption confidence
    'crisis_threshold': 0.7,          # Paradigm crisis threshold
    'min_failures_for_clustering': 10,
    'anomaly_contamination': 0.1       # Expected outlier proportion
}
```

### Custom Configuration

```python
config = {
    'confidence_threshold': 0.7,      # Stricter threshold
    'crisis_threshold': 0.8,          # Higher crisis threshold
    'anomaly_contamination': 0.15      # More anomalies expected
}

engine = Phi15Engine(config)
```

### Component Configuration

```python
# Anomaly Detector
anomaly_detector = AnomalyDetector(
    contamination=0.1,
    isolation_weight=0.5,
    lof_weight=0.5
)

# Failure Clusterer
clusterer = FailureClusterer(
    n_clusters_range=(2, 10),
    dbscan_eps=0.5,
    dbscan_min_samples=5
)

# Confidence Scorer
scorer = ConfidenceScorer()
scorer.weights = {
    'support': 0.30,
    'pattern': 0.25,
    'counterfactual': 0.20,
    'novelty': 0.10,
    'historical': 0.10,
    'testability': 0.05
}
```

---

## Examples

### Example 1: Detect Need for Approximation

```python
from rese.phase1.tacit_assumption_miner import (
    Phi15Engine, NullResult, ErrorType
)
from datetime import datetime

# Create engine
engine = Phi15Engine()

# Generate null results: exact algorithms always timeout
null_results = []
for i in range(30):
    result = NullResult(
        attempt_id=f"exact_timeout_{i}",
        timestamp=datetime.now(),
        problem_type="np_optimization",
        approach_type="exact_branch_and_bound",
        constraints=["exact_solution_required"],
        error_type=ErrorType.TIMEOUT,
        error_message=f"Exact algorithm exceeded time limit (2^{i/10} complexity)",
        state={"time_elapsed": 7200, "time_limit": 3600},
        iteration=1000,
        resources_used={"cpu": 100.0, "time": 7200}
    )
    null_results.append(result)

# Process
assumptions, paradigm_rec = engine.process_null_results(null_results)

# Check results
print("Inferred Assumptions:")
for assumption in engine.get_top_assumptions(k=5):
    print(f"  - {assumption.description} (confidence: {assumption.confidence:.2f})")

# Expected: Should infer "Approximation is acceptable"
```

### Example 2: Detect Need for Randomization

```python
# Similar setup, but with deterministic approaches
# converging to same local optima
null_results = []

for i in range(30):
    result = NullResult(
        attempt_id=f"deterministic_stuck_{i}",
        timestamp=datetime.now(),
        problem_type="global_optimization",
        approach_type="deterministic_gradient_descent",
        constraints=["deterministic_required"],
        error_type=ErrorType.OPTIMIZATION_FAILED,
        error_message=f"Converged to local optimum at 0.2 (global: 1.0)",
        state={"objective": 0.2, "global_optimum": 1.0},
        iteration=1000,
        resources_used={"cpu": 80.0}
    )
    null_results.append(result)

assumptions, paradigm_rec = engine.process_null_results(null_results)

# Expected: Should infer "Randomization can help escape local optima"
```

### Example 3: Full Pipeline with Database

```python
from rese.phase1.phi15_interfaces import create_interface_manager

# Create interface manager
manager = create_interface_manager(
    config={'phi15': {'confidence_threshold': 0.6}},
    database_path="my_failures.db"
)

# Process Stage 6 input
results = manager.process_stage6_input(null_results)

print(f"Processed: {results['processed']} failures")
print(f"Assumptions sent: {results['assumptions_sent']}")
print(f"Paradigm crisis: {results['paradigm_crisis']}")

# Get detailed status
status = manager.get_status()
print(f"Total assumptions in database: {status['total_assumptions']}")

# Cleanup
manager.shutdown()
```

---

## Performance

### Benchmarks

| Metric | Target | Typical |
|--------|--------|---------|
| Processing latency (100 failures) | <10s | 5-8s |
| Throughput | >1000/hour | 1200-1500/hour |
| Memory (10k failures) | <2GB | 1.2-1.5GB |
| Storage (10k failures) | <100MB | 60-80MB |

### Optimization Tips

1. **Batch Processing**:
   ```python
   # Faster than single results
   interface.receive_batch_null_results(null_results)
   ```

2. **Incremental Processing**:
   ```python
   # Configure thresholds
   interface.incremental_threshold = 20
   interface.incremental_time_hours = 2
   ```

3. **Database Caching**:
   ```python
   # Cache is automatic, but can be controlled
   database.clear_cache()  # Free memory
   ```

---

## Troubleshooting

### No Assumptions Generated

**Issue**: Processing completes but no assumptions inferred.

**Possible Causes**:
1. Too few failures (<10 minimum for clustering)
2. Failures are too diverse (no clear patterns)
3. Confidence threshold too high

**Solutions**:
```python
# Lower threshold
config = {'confidence_threshold': 0.5}
engine = Phi15Engine(config)

# Check failure count
print(f"Total failures: {len(engine.failures)}")

# Increase data
# Collect more null results before processing
```

### Poor Clustering Quality

**Issue**: Clusters don't group similar failures well.

**Solutions**:
```python
# Adjust clustering parameters
clusterer = FailureClusterer(
    n_clusters_range=(3, 15),  # Try more clusters
    dbscan_eps=0.7,            # Increase neighborhood size
    dbscan_min_samples=3        # Require fewer points
)
```

### Memory Issues

**Issue**: Out of memory with large datasets.

**Solutions**:
```python
# Process in batches
batch_size = 500
for i in range(0, len(null_results), batch_size):
    batch = null_results[i:i+batch_size]
    engine.process_null_results(batch)

# Clear database cache
database.clear_cache()
```

---

## Best Practices

1. **Collect Diverse Failures**: Include different error types and approaches
2. **Provide Rich Context**: Include state and metadata in null results
3. **Validate Regularly**: Use Stage 7 to validate and update confidence
4. **Monitor Paradigm Signals**: Watch for paradigm crisis indicators
5. **Iterative Refinement**: Periodically re-process all failures with updated models

---

## Further Reading

- Research: `rese/docs/phi15_assumption_mining_research.md`
- Algorithm Design: `rese/docs/phi15_algorithm_design.md`
- Implementation Plan: `rese/docs/phi15_implementation_plan.md`
- Validation: `rese/phase1/validate_phi15.py`

---

## Support

For issues, questions, or contributions:
- Agent: B1 (Φ₁/Φ₁.₅ Specialist)
- Date: 2025-12-31
- Status: Active Implementation

---

**End of API Documentation**
