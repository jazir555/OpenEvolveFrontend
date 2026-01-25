# Phase 1 (Epistemic Audit) - Quick Start Guide

## Overview

Phase 1 implements Φ₁-Φ₄ subroutines for the RESE framework, providing:
- **Φ₁.₅:** Tacit Assumption Mining from null results (KEY INNOVATION)
- **Φ₂:** Metacognitive Debiasing for constraint formulation
- Integration with E2E Stages 1, 5, 6, 7

## Quick Test

```python
import sys
sys.path.insert(0, 'rese/phase1')

from tacit_assumption_miner import Phi15Engine, NullResult, ErrorType
from datetime import datetime

# Create engine
engine = Phi15Engine()

# Create sample null results
null_results = [
    NullResult(
        attempt_id=f'test_{i:03d}',
        timestamp=datetime.now(),
        problem_type='optimization',
        approach_type='deterministic',
        constraints=['c1', 'c2'],
        error_type=ErrorType.TIMEOUT if i % 2 == 0 else ErrorType.INFEASIBILITY,
        error_message=f'Failed at iteration {i*100}',
        state={'iter': i*100},
        iteration=i*100,
        resources_used={'cpu': 50.0, 'memory': 1000.0}
    )
    for i in range(30)
]

# Process and mine assumptions
assumptions, paradigm = engine.process_null_results(null_results)

# Get top assumptions
top = engine.get_top_assumptions(k=5)

for i, a in enumerate(top, 1):
    print(f"{i}. {a.description}")
    print(f"   Confidence: {a.confidence:.2f}")
    print(f"   Support: {a.support} failures")
    print(f"   Paradigm shift: {a.paradigm_implication}")
    if a.alternative_paradigm:
        print(f"   Alternative: {a.alternative_paradigm}")
```

## Integration Examples

### Stage 6 → Φ₁.₅: Receive Null Results

```python
from phi15_interfaces import Phi15InterfaceManager

# Create interface manager
manager = Phi15InterfaceManager()

# Process Stage 6 input
result = manager.process_stage6_input(null_results)
print(f"Processed: {result['processed']}")
print(f"Assumptions sent: {result['assumptions_sent']}")
print(f"Paradigm crisis: {result['paradigm_crisis']}")
```

### Φ₂: Detect Cognitive Biases

```python
from cognitive_biases import CognitiveBiasDetector
from symbolic_constraint_engine import Constraint, ConstraintType

detector = CognitiveBiasDetector()

constraints = [
    Constraint(
        id='c1',
        type=ConstraintType.HARD,
        description='This will certainly achieve perfect accuracy',
        formalization='accuracy = 1.0',
        source='user'
    )
]

report = detector.analyze_constraints(constraints)

print(f"Bias score: {report.overall_bias_score:.2f}")
print(f"Detections: {report.total_detections}")

for detection in report.detections:
    print(f"- {detection.bias_type.value}: {detection.suggestion}")
```

### Φ₂ Integration with SCE

```python
from phi2_integration import SCEPhi2Integrator
from symbolic_constraint_engine import SymbolicConstraintEngine

# Create SCE
sce = SymbolicConstraintEngine()

# Create integrator
integrator = SCEPhi2Integrator(sce)

# Add constraint (automatically checked for bias)
constraint = Constraint(
    id='c1',
    type=ConstraintType.HARD,
    description='Clearly the optimal solution',
    formalization='optimal = true',
    source='user'
)

sce.add_constraint(constraint)  # Bias detection happens automatically

# Get biased constraints
biased = integrator.get_biased_constraints(min_severity=Severity.MEDIUM)

for constraint_id, detections in biased.items():
    print(f"{constraint_id}: {len(detections)} biases detected")
```

### Stage 5: Real-time Bias Monitoring

```python
from phi2_integration import Stage5Phi2Monitor

monitor = Stage5Phi2Monitor()

# Monitor generation steps
for step, reasoning in enumerate(generation_steps):
    report = monitor.monitor_generation_step(step, reasoning)

    if monitor.should_intervene(step):
        print(f"Step {step}: BIAS DETECTED - Intervention recommended")
        recommendations = monitor.get_step_recommendations(step)
        for rec in recommendations:
            print(f"  - {rec}")

        # Generate debiased alternatives
        alternatives = monitor.generate_debiased_alternatives(reasoning)
        for alt in alternatives:
            print(f"  Alternative: {alt}")
```

### Database Operations

```python
from failure_database import DatabaseManager
from datetime import datetime, timedelta

# Create database manager
db = DatabaseManager('rese/data/my_failures.db')

# Add failures
db.add_null_results(null_results)

# Get statistics
stats = db.get_statistics()
print(f"Total failures: {stats['total_failures']}")
print(f"Recent assumptions: {stats['recent_assumptions_30d']}")

# Get recent failures
recent = db.db.get_recent_failures(hours=24)

# Get high-confidence assumptions
assumptions = db.db.get_high_confidence_assumptions(min_confidence=0.7)

# Export to JSON
db.export_to_json('failures_export.json')

# Cleanup
db.close()
```

## Validation

Run the validation suite:

```bash
python rese/phase1/validate_phi15.py
```

This will:
1. Generate 4 synthetic test cases with known ground truth
2. Run Φ₁.₅ on each case
3. Measure accuracy against ground truth
4. Generate validation report

**Target:** >70% assumption mining accuracy

## Configuration

### Φ₁.₅ Engine Configuration

```python
config = {
    'confidence_threshold': 0.6,      # Minimum confidence for assumptions
    'crisis_threshold': 0.7,          # Threshold for paradigm crisis
    'min_failures_for_clustering': 10, # Min failures before clustering
    'anomaly_contamination': 0.1      # Expected outlier proportion
}

engine = Phi15Engine(config)
```

### Anomaly Detection Configuration

```python
from tacit_assumption_miner import AnomalyDetector

anomaly_detector = AnomalyDetector(
    contamination=0.1,        # 10% expected outliers
    isolation_weight=0.5,    # Weight for Isolation Forest
    lof_weight=0.5           # Weight for LOF
)
```

### Clustering Configuration

```python
from tacit_assumption_miner import FailureClusterer

clusterer = FailureClusterer(
    n_clusters_range=(2, 10),      # Try 2-10 clusters
    dbscan_eps=0.5,                # DBSCAN epsilon
    dbscan_min_samples=5            # DBSCAN min samples
)
```

### Φ₂ Integration Configuration

```python
from phi2_integration import IntegrationConfig

config = IntegrationConfig(
    auto_check_on_add=True,         # Check bias when adding constraints
    auto_check_on_conflict=True,    # Check bias on conflicts
    bias_threshold=0.5,             # Confidence threshold for alerts
    real_time_monitoring=True,      # Monitor during generation
    max_bias_score=0.7,             # Maximum allowed bias score
    auto_debias=False               # Auto-apply debiasing (experimental)
)
```

## Data Structures

### NullResult (Input from Stage 6)

```python
NullResult(
    attempt_id: str,                    # Unique identifier
    timestamp: datetime,                 # When it occurred
    problem_type: str,                   # Type of problem
    approach_type: str,                  # Approach used
    constraints: List[str],              # Constraints applied
    error_type: ErrorType,               # Type of error
    error_message: str,                  # Error description
    state: Dict[str, Any],               # Final state
    iteration: int,                      # Iteration number
    resources_used: Dict[str, float],    # Resources consumed
    metadata: Dict[str, Any]             # Additional context
)
```

### TacitAssumption (Output from Φ₁.₅)

```python
TacitAssumption(
    id: str,                           # Unique ID
    description: str,                   # Human-readable
    formalization: str,                 # SCE constraint format
    assumption_type: AssumptionType,    # Category
    confidence: float,                  # [0, 1] score
    support: int,                       # Failures explained
    evidence: List[str],                # Supporting failure IDs
    pattern_type: PatternType,          # Pattern type
    constraint_relaxation: str,         # How to relax
    paradigm_implication: bool,         # Suggests paradigm shift?
    alternative_paradigm: Optional[str], # Alternative paradigm
    timestamp: datetime,                # When inferred
    verified: bool                      # Validated by Stage 7?
)
```

### BiasDetection (Output from Φ₂)

```python
BiasDetection(
    bias_type: BiasType,                # Type of bias
    severity: Severity,                 # LOW/MEDIUM/HIGH/CRITICAL
    confidence: float,                  # [0, 1] score
    description: str,                   # Human-readable
    evidence: Dict[str, str],           # Supporting evidence
    suggestion: str,                    # Remediation
    affected_elements: List[str]        # Affected constraint IDs
)
```

## Error Types

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

## Bias Types

```python
class BiasType(Enum):
    CONFIRMATION = "confirmation_bias"
    AVAILABILITY = "availability_bias"
    ANCHORING = "anchoring_bias"
    SUNK_COST = "sunk_cost_fallacy"
    FRAMING = "framing_effect"
    OVERCONFIDENCE = "overconfidence_effect"
    DUNNING_KRUGER = "dunning_kruger_effect"
    AUTHORITY = "authority_bias"
    CLUSTERING = "clustering_illusion"
    TEXAS_SHARPSHOOTER = "texas_sharpshooter_fallacy"
    CAUSAL_OVERSIMPLIFICATION = "causal_oversimplification"
    ILLUSION_OF_CONTROL = "illusion_of_control"
```

## Troubleshooting

### Issue: "sklearn not available" warning
**Solution:** Install scikit-learn: `pip install scikit-learn`
**Fallback:** System will use Z-score based anomaly detection and scipy clustering

### Issue: Low assumption confidence (< 0.6)
**Possible causes:**
1. Not enough failures (need >10 for clustering)
2. High variance in failure patterns
3. Insufficient pattern similarity

**Solution:** Provide more null results with similar characteristics

### Issue: No paradigm crisis detected
**Possible causes:**
1. Crisis threshold too high (default 0.7)
2. Assumptions don't challenge fundamental constraints

**Solution:** Lower crisis_threshold or provide more systematic failures

### Issue: High bias score in all constraints
**Possible causes:**
1. User prompts contain absolutist language
2. Overconfidence in problem formulation

**Solution:** Use debiasing strategies:
```python
from cognitive_biases import DebiasingStrategy

alternatives = DebiasingStrategy.consider_the_opposite(constraint)
challenges = DebiasingStrategy.devils_advocate(constraint)
```

## Performance Tips

1. **Batch processing:** Process null results in batches of 50-100
2. **Database caching:** Use default cache_size (1000) for optimal performance
3. **Incremental processing:** Enable incremental processing for real-time analysis
4. **Feature encoding:** Pre-encode categorical features for faster processing

## Next Steps

1. **Run validation:** `python rese/phase1/validate_phi15.py`
2. **Test with real data:** Replace synthetic null results with actual failures
3. **Tune thresholds:** Adjust confidence and crisis thresholds based on domain
4. **Create seed data:** Add historical paradigm shifts to database
5. **Monitor performance:** Track assumption accuracy over time

## Support

For detailed debugging information, see `DEBUGGING_REPORT.md`

For algorithm details, see individual file docstrings.

---
**Last Updated:** 2025-12-31
**Status:** ✅ All systems operational
