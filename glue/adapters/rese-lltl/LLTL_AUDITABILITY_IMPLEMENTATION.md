# LLTL DEE → SCE Auditability Implementation

## Overview

This document describes the implementation of the DEE → SCE auditability component of the Logic-to-Loss Translation Layer (LLTL). This component enables bidirectional translation between the Symbolic Constraint Engine (SCE) and the Deep Exploration Engine (DEE), completing the LLTL specification from RESE Technical Manual §2.2.

## Architecture

### Components

1. **FormalCommitment**: Dataclass representing a formal propositional commitment
2. **LLTLAdapter**: Extended with DEE → SCE methods
3. **Audit Trail**: Complete tracking of all statistical-to-formal translations

### Data Flow

```
DEE Statistical Result
    ↓
statistical_to_formal()
    ↓
FormalCommitment
    ↓
integrate_into_sce()
    ↓
SCE Logic Graph (with contradiction detection)
```

## Implementation Details

### 1. FormalCommitment Dataclass

Located in `glue/adapters/rese-lltl/src/lltl_adapter.py`

```python
@dataclass
class FormalCommitment:
    """Formal propositional commitment for SCE integration"""
    proposition_id: str
    statement: str  # Formal logical statement
    confidence_threshold: float  # 0-1, minimum confidence to accept
    statistical_evidence: Dict[str, float]
    source_hypothesis: str
    derivation_method: str
    timestamp: str  # UTC ISO-8601
    correlation_id: str
    lean4_theorem: Optional[str] = None
```

**Methods:**
- `to_sce_constraint()`: Convert to SCE constraint format
- `to_dict()`: Serialize to dictionary

### 2. Statistical to Formal Conversion

**Method:** `LLTLAdapter.statistical_to_formal()`

Converts DEE statistical results to Formal Propositional Commitments.

**Input Format:**
```python
statistical_result = {
    'hypothesis_statement': str,      # Required
    'confidence': float,               # Required, 0-1
    'p_value': float,                  # Optional, default 1.0
    'confidence_interval': Tuple[float, float],  # Optional
    'expected_value': float,           # Optional
    'validation_metric': str,          # Optional
    'evidence': List[Dict]             # Optional
}
```

**Processing Steps:**

1. **Extract Statistical Evidence**
   - Confidence score
   - P-value
   - Confidence interval bounds
   - Expected value

2. **Calculate Confidence Threshold**
   ```
   confidence >= 0.95 → threshold = 0.90 (Very high)
   confidence >= 0.80 → threshold = 0.75 (High)
   confidence >= 0.60 → threshold = 0.60 (Moderate)
   confidence < 0.60  → threshold = 0.50 (Low)
   ```

3. **Construct Formal Statement**
   Format: `(H) ∧ (confidence ≥ T) ∧ (p ≤ α) ∧ (CI ∈ [lower, upper]) → Accept(H)`

   Example:
   ```
   (Lattice confinement enables LENR) ∧ (confidence ≥ 0.850) ∧
   (p_value ≤ 0.050) ∧ (CI ∈ [0.780, 0.920]) → Accept(Lattice confinement enables LENR)
   ```

4. **Create FormalCommitment**
   - Store for auditability (Law of Idempotency)
   - Return commitment object

**Output:**
```python
FormalCommitment(
    proposition_id="uuid",
    statement="(...)",
    confidence_threshold=0.75,
    statistical_evidence={...},
    source_hypothesis="hypothesis-1",
    derivation_method="mcts_validation",
    timestamp="2026-02-04T12:00:00Z",
    correlation_id="correlation-123"
)
```

### 3. SCE Integration

**Method:** `LLTLAdapter.integrate_into_sce()`

Integrates formal commitments into the SCE logic graph for contradiction detection.

**Steps:**

1. **Convert to SCE Constraint Format**
   ```python
   sce_constraint = {
       "constraint_id": proposition_id,
       "formal_statement": statement,
       "confidence": confidence_threshold,
       "evidence": statistical_evidence,
       "type": "statistical_commitment"
   }
   ```

2. **Add to SCE Logic Graph**
   - Import SCE Constraint types
   - Create Constraint object
   - Call `sce_engine.add_constraint()`

3. **Detect Contradictions**
   - Call `sce_engine.detect_contradictions()`
   - Log warnings if contradictions found
   - Don't fail integration (contradictions are warnings)

**Error Handling:**
- Missing SCE engine: Return error
- Missing add_constraint: Return error
- Import errors: Use fallback direct integration
- Contradictions: Log warning, don't fail
- Auditability disabled: Skip integration, return success

### 4. Audit Trail

**Methods:**
- `get_audit_trail()`: Get all formal commitments
- `get_commitment(proposition_id)`: Get specific commitment
- `clear_audit_trail()`: Clear all commitments

**Purpose:**
- Complete auditability of DEE → SCE translations
- Replayability (Law of Idempotency)
- Debugging and validation

## Configuration

### Environment Variables

```bash
# Enable/disable DEE → SCE auditability
LLTL_AUDITABILITY_ENABLED=true

# Default confidence threshold
LLTL_CONFIDENCE_THRESHOLD_DEFAULT=0.75

# Statistical significance level (α)
LLTL_SIGNIFICANCE_LEVEL=0.05

# SCE integration timeout (milliseconds)
LLTL_AUDIT_TIMEOUT_MS=5000
```

### Validation

All configuration is validated on startup (Law of Configuration Explicitness):
- Values must be valid types
- Thresholds must be between 0-1
- Timeouts must be positive

## CLAUDE.md Compliance

### Law of Idempotency
- Same statistical result → same formal commitment (deterministic)
- Multiple calls to `integrate_into_sce()` are safe (check before create)
- Audit trail preserves all translations

### Law of Configuration Explicitness
- All configuration via environment variables
- No magic defaults
- Crash on invalid config

### Law of UTC
- All timestamps in UTC ISO-8601 format
- No timezone ambiguity

### Structured Logging
- JSON logs with correlation_id
- All operations logged
- Context: operation, proposition_id, confidence, threshold

### Circuit Breaker
- Timeout on SCE integration
- Graceful degradation if SCE unavailable
- Warning logs for contradictions

### Auditability
- Complete audit trail
- Queryable by proposition_id
- Clear for testing/isolation

## Usage Examples

### Basic Usage

```python
from lltl_adapter import LLTLAdapter, FormalCommitment

# Create adapter
adapter = LLTLAdapter()

# Convert DEE result to formal commitment
statistical_result = {
    'hypothesis_statement': 'Lattice confinement enables LENR',
    'confidence': 0.85,
    'p_value': 0.02,
    'confidence_interval': (0.78, 0.92),
    'expected_value': 0.85
}

commitment, error = adapter.statistical_to_formal(
    statistical_result=statistical_result,
    source_hypothesis='hypothesis-1',
    derivation_method='mcts_validation',
    correlation_id='my-correlation-id'
)

if error:
    print(f"Error: {error}")
else:
    print(f"Created commitment: {commitment.proposition_id}")
    print(f"Statement: {commitment.statement}")
    print(f"Threshold: {commitment.confidence_threshold}")
```

### Integration with SCE

```python
from glue.adapters.rese-sce.src.sce_bridge import SymbolicConstraintEngine

# Create SCE engine
sce = SymbolicConstraintEngine()

# Integrate commitment
success, error = adapter.integrate_into_sce(
    commitment=commitment,
    sce_engine=sce,
    correlation_id='my-correlation-id'
)

if not success:
    print(f"Integration failed: {error}")
else:
    print("Commitment integrated into SCE")
```

### Audit Trail

```python
# Get all commitments
trail = adapter.get_audit_trail()
print(f"Total commitments: {len(trail)}")

# Get specific commitment
commitment = adapter.get_commitment(proposition_id)
if commitment:
    print(f"Found: {commitment.statement}")

# Clear audit trail (for testing)
count = adapter.clear_audit_trail()
print(f"Cleared {count} commitments")
```

### Phase III Integration

```python
# In Phase III executor after MCTS validation
async def execute_phase3():
    # Execute MCTS search
    mcts_result = await mcts_executor.execute_search(...)

    # Convert to formal commitment
    commitment, error = lltlt_adapter.statistical_to_formal(
        statistical_result={
            'hypothesis_statement': mcts_result.best_hypothesis,
            'confidence': mcts_result.confidence,
            'p_value': mcts_result.p_value,
            'confidence_interval': mcts_result.ci,
            'expected_value': mcts_result.expected_value
        },
        source_hypothesis=mcts_result.best_hypothesis_id,
        derivation_method='mcts_validation',
        correlation_id=correlation_id
    )

    # Integrate into SCE
    if commitment:
        success, error = lltlt_adapter.integrate_into_sce(
            commitment=commitment,
            sce_engine=sce_engine,
            correlation_id=correlation_id
        )

        if not success:
            logger.warning(f"SCE integration failed: {error}")
```

## Testing

### Unit Tests

Located in `glue/adapters/rese-lltl/tests/test_dee_to_sce_auditability.py`

Run with:
```bash
cd glue/adapters/rese-lltl
python -m pytest tests/test_dee_to_sce_auditability.py -v
```

### Integration Tests

Located in `glue/adapters/rese-lltl/tests/test_dee_to_sce_simple.py`

Run with:
```bash
cd glue/adapters/rese-lltl
python tests/test_dee_to_sce_simple.py
```

**Test Coverage:**
1. FormalCommitment creation
2. SCE constraint conversion
3. Statistical to formal conversion
4. Confidence threshold calculation
5. Formal statement construction
6. SCE integration
7. Audit trail tracking
8. Idempotency
9. Error handling

## Success Criteria

- [x] Statistical results convert to formal commitments
- [x] Confidence thresholds calculated appropriately
- [x] Formal statements constructed correctly
- [x] SCE integration works
- [x] Contradictions detected
- [x] Audit trail complete and queryable
- [x] All tests passing
- [x] CLAUDE.md compliance verified

## Future Enhancements

1. **Lean 4 Formalization**
   - Add `lean4_theorem` field with actual Lean 4 code
   - Export to `.lean` files
   - Integrate with Lean 4 theorem prover

2. **Advanced Confidence Thresholding**
   - Machine learning-based threshold calibration
   - Domain-specific threshold profiles
   - Adaptive thresholds based on historical accuracy

3. **Batch Integration**
   - Integrate multiple commitments at once
   - Batch contradiction detection
   - Optimized for high-volume scenarios

4. **Export/Import Audit Trail**
   - Export to JSON/CSV
   - Import for replay
   - Cross-session auditability

## References

- RESE Technical Manual §2.2: Logic-to-Loss Translation Layer
- CLAUDE.md: Project Constitution
- glue/adapters/rese-lltl/src/lltl_adapter.py: Implementation
- glue/adapters/rese-sce/src/sce_bridge.py: SCE Bridge

## Authors

RESE Team

## Version History

- 2026-02-04: Initial implementation (DEE → SCE auditability)
- Completes bidirectional LLTL (SCE ↔ DEE)
