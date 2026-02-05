# Tiered Verification System - Complete Guide

Comprehensive documentation for the RESE Tiered Verification System, including architecture, decision trees, performance characteristics, usage examples, and escalation strategies.

## Table of Contents

1. [System Overview](#system-overview)
2. [3-Tier Architecture](#3-tier-architecture)
3. [Decision Trees](#decision-trees)
4. [Performance Characteristics](#performance-characteristics)
5. [Usage Examples](#usage-examples)
6. [Escalation Strategies](#escalation-strategies)
7. [API Reference](#api-reference)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## System Overview

The Tiered Verification System provides a unified API for formal verification across three tiers of increasing rigor. It automatically selects the appropriate solver based on problem complexity and escalates to higher tiers if needed.

### Key Features

- **Adaptive Solver Selection** - Automatically selects best tier
- **Automatic Escalation** - Escalates when lower tiers fail
- **Unified API** - Single interface for all verification types
- **Performance Monitoring** - Tracks solver effectiveness
- **Circuit Breaker** - Graceful failure handling
- **Problem Classification** - Intelligent problem analysis

### Verification Pipeline

```
Problem Input
    ↓
┌─────────────────────┐
│ Problem Classifier  │ → Analyze problem
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Solver Selector    │ → Select initial tier
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Tier Execution     │ → Run verification
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Escalation Check    │ → Should escalate?
└─────────────────────┘
    ↓ (yes)          ↓ (no)
┌─────────────┐   ┌──────────────┐
│ Next Tier   │   │ Return Result│
└─────────────┘   └──────────────┘
```

## 3-Tier Architecture

### Tier 1: Z3 Fast Verification

**Purpose**: Fast constraint satisfaction and contradiction detection

**When to Use**:
- Quick satisfiability checks
- Contradiction detection
- Simple constraint problems
- Fast response needed (<1 second)

**Capabilities**:
- Boolean logic
- Linear arithmetic
- Simple arrays
- Bit vectors
- Optimization problems

**Limitations**:
- No quantifiers (or very shallow nesting)
- Limited nonlinear support
- 0-100 constraints
- Lower confidence (70%)

**Example**:
```python
from glue.adapters.rese_verification.src import TieredVerifier

verifier = TieredVerifier()

# Simple constraint satisfaction
result = verifier.verify(
    "Find x, y such that x > 0 and y > 0 and x + y > 0"
)

# Z3 will solve this in milliseconds
assert result.is_successful()
assert result.successful_tier == VerificationTier.TIER1_Z3
assert result.confidence == 0.7
```

### Tier 2: LeanAide AI-Assisted Proving

**Purpose**: AI-guided theorem proving with autoformalization

**When to Use**:
- Theorem proving with quantifiers
- Natural language to formal proof
- Medium complexity problems
- AI assistance beneficial

**Capabilities**:
- Quantifier reasoning
- Nonlinear arithmetic
- Autoformalization
- Tactic suggestion
- Proof completion

**Limitations**:
- May produce partial proofs
- 100-1000 constraints
- Medium confidence (85%)
- Depends on AI model

**Example**:
```python
from glue.adapters.rese_verification.src import TieredVerifier

verifier = TieredVerifier()

# Theorem proving with quantifiers
result = verifier.verify(
    "For all real numbers x and y, if x > 0 and y > 0 then x + y > 0"
)

# LeanAide will formalize and prove this
assert result.is_successful()
assert result.successful_tier == VerificationTier.TIER2_LEANAIDE
assert result.confidence == 0.85

# Access the proof
if result.tier2_result:
    print(f"Proof script: {result.tier2_result.proof_script}")
    print(f"Tactics used: {result.tier2_result.tactics_used}")
```

### Tier 3: Lean 4 Formal Verification

**Purpose**: Machine-checkable formal proofs

**When to Use**:
- Maximum rigor required
- Complex mathematical reasoning
- Machine-checkable proofs
- Deep quantifier nesting

**Capabilities**:
- Full Lean 4 expressiveness
- Category theory
- Complex analysis
- Dependent type theory
- 100% confidence

**Limitations**:
- No time limit (can be slow)
- Requires Lean expertise
- 1000+ constraints
- Highest overhead

**Example**:
```python
from glue.adapters.rese_verification.src import TieredVerifier, VerificationTier

verifier = TieredVerifier()

# Complex theorem requiring formal verification
result = verifier.verify_with_tier(
    "Prove that the limit of (1 + 1/n)^n as n approaches infinity is e",
    tier=VerificationTier.TIER3_LEAN4
)

# Lean 4 will produce machine-checkable proof
assert result.is_successful()
assert result.confidence == 1.0

# Access Lean 4 code
if result.tier3_result:
    print(f"Theorem name: {result.tier3_result.theorem_name}")
    print(f"Lean 4 code: {result.tier3_result.lean4_code}")
```

## Decision Trees

### Initial Tier Selection

```
Start
  │
  ├─ Problem has quantifiers?
  │   ├─ Yes → Quantifier depth > 2?
  │   │   ├─ Yes → Tier 3 (Lean 4)
  │   │   └─ No → Has nonlinear operations?
  │   │       ├─ Yes → Tier 2 (LeanAide)
  │   │       └─ No → Tier 1 (Z3)
  │   └─ No → Is optimization problem?
  │       ├─ Yes → Tier 1 (Z3)
  │       └─ No → Has nonlinear operations?
  │           ├─ Yes → Tier 2 (LeanAide)
  │           └─ No → Tier 1 (Z3)
```

### Escalation Decision Tree

```
Current Tier Result
  │
  ├─ Status == VERIFIED?
  │   ├─ Yes → SUCCESS (return result)
  │   └─ No → Continue
  │
  ├─ Auto-escalate enabled?
  │   ├─ No → FAILURE (return result)
  │   └─ Yes → Continue
  │
  ├─ Current tier == Tier 3?
  │   ├─ Yes → FAILURE (no more tiers)
  │   └─ No → Continue
  │
  ├─ Execution time > tier timeout?
  │   ├─ Yes → ESCALATE
  │   └─ No → Continue
  │
  ├─ Constraint count > tier limit?
  │   ├─ Yes → ESCALATE
  │   └─ No → Continue
  │
  ├─ Status == UNKNOWN or TIMEOUT?
  │   ├─ Yes → ESCALATE
  │   └─ No → FAILURE (return result)
```

### Solver Strategy Selection

```
User Configuration
  │
  ├─ Strategy == FAST_FIRST?
  │   └─ Start at Tier 1, escalate as needed
  │
  ├─ Strategy == ACCURATE_FIRST?
  │   └─ Start at highest available tier
  │
  ├─ Strategy == PARALLEL?
  │   └─ Run multiple tiers in parallel
  │
  ├─ Strategy == USER_SPECIFIED?
  │   └─ Use user-specified tier
  │
  └─ Strategy == ADAPTIVE?
      └─ Select based on problem classification
```

## Performance Characteristics

### Execution Time

| Tier | Average Time | Max Time | Typical Use |
|------|--------------|----------|-------------|
| 1    | <100ms       | 1s       | Quick checks |
| 2    | <10s         | 1m       | Medium theorems |
| 3    | Variable     | No limit | Complex proofs |

### Constraint Scalability

| Tier | Min Constraints | Max Constraints | Optimal Range |
|------|-----------------|-----------------|---------------|
| 1    | 0              | 100             | 10-50         |
| 2    | 50             | 1000            | 100-500       |
| 3    | 500            | No limit        | 1000+         |

### Confidence Levels

| Tier | Confidence | Justification                              |
|------|------------|--------------------------------------------|
| 1    | 70%        | Fast but less rigorous                      |
| 2    | 85%        | AI-assisted, machine-checked                |
| 3    | 100%       | Machine-checkable, complete mathematical rigor |

### Memory Usage

| Tier | Typical Memory | Peak Memory |
|------|----------------|-------------|
| 1    | <100MB         | 500MB       |
| 2    | <500MB         | 2GB         |
| 3    | <2GB           | 8GB         |

## Usage Examples

### Example 1: Simple Constraint Satisfaction

```python
from glue.adapters.rese_verification.src import TieredVerifier

verifier = TieredVerifier()

# Define constraints
constraints = [
    "x > 0",
    "y > 0",
    "x + y < 10"
]

# Verify constraints are satisfiable
result = verifier.verify(
    "Find x, y satisfying constraints",
    constraints=constraints
)

if result.is_successful():
    print(f"✓ Satisfiable")
    print(f"  Solver: {result.successful_tier.value}")
    print(f"  Confidence: {result.confidence:.1%}")

    # Access model if available
    if result.tier1_result and result.tier1_result.model:
        print(f"  Model: {result.tier1_result.model}")
else:
    print(f"✗ Unsatisfiable or unknown")
```

### Example 2: Contradiction Detection

```python
from glue.adapters.rese_verification.src import TieredVerifier

verifier = TieredVerifier()

# Contradictory constraints
constraints = [
    "x > 10",
    "x < 5"
]

result = verifier.verify(
    "Check for contradictions",
    constraints=constraints
)

if result.tier1_result and result.tier1_result.z3_result == "unsat":
    print("✓ Contradiction detected")
else:
    print("✗ No contradiction found")
```

### Example 3: Theorem Proving with Escalation

```python
from glue.adapters.rese_verification.src import TieredVerifier

verifier = TieredVerifier()

# Theorem that may require escalation
theorem = """
For all natural numbers n,
the sum of the first n natural numbers is n*(n+1)/2
"""

result = verifier.verify(theorem)

print(f"Final status: {result.final_status.value}")
print(f"Successful tier: {result.successful_tier.value if result.successful_tier else 'None'}")
print(f"Escalation path: {[t.value for t in result.escalation_path]}")
print(f"Total time: {result.total_execution_time_ms:.0f}ms")

# Access proof from successful tier
successful_result = result.get_successful_result()
if successful_result and hasattr(successful_result, 'proof_script'):
    print(f"\nProof:\n{successful_result.proof_script}")
```

### Example 4: Parallel Verification

```python
import os
from glue.adapters.rese_verification.src import TieredVerifier, SelectionStrategy

# Configure for parallel execution
os.environ["SELECTION_STRATEGY"] = "parallel"
os.environ["MAX_PARALLEL_SOLVERS"] = "2"

verifier = TieredVerifier()

result = verifier.verify(
    "For all x, P(x) implies Q(x)"
)

# Result combines all parallel verification attempts
print(f"Status: {result.final_status.value}")
print(f"Confidence: {result.confidence:.1%}")
```

### Example 5: Custom Tier Selection

```python
from glue.adapters.rese_verification.src import (
    TieredVerifier,
    VerificationTier,
    SelectionStrategy
)

verifier = TieredVerifier()

# Option 1: Specify tier in metadata
result1 = verifier.verify(
    "Prove theorem",
    metadata={"selection_strategy": "accurate_first"}
)

# Option 2: Use specific tier
result2 = verifier.verify_with_tier(
    "Prove theorem",
    tier=VerificationTier.TIER3_LEAN4
)

# Option 3: Configure strategy globally
result3 = verifier.verify(
    "Prove theorem",
    metadata={"preferred_tier": "tier3_lean4"}
)
```

### Example 6: Performance Monitoring

```python
from glue.adapters.rese_verification.src import TieredVerifier

verifier = TieredVerifier()

# Run multiple verifications
problems = [
    "x > 0",
    "forall x, P(x) -> Q(x)",
    "sum of first n numbers = n*(n+1)/2"
]

for problem in problems:
    verifier.verify(problem)

# Get performance statistics
stats = verifier.selector.get_performance_stats()

print("Performance Summary:")
for tier, metrics in stats.items():
    print(f"\n{tier}:")
    print(f"  Total attempts: {metrics['total_attempts']}")
    print(f"  Success rate: {metrics['success_rate']:.1%}")
    print(f"  Average time: {metrics['average_time_ms']:.0f}ms")
    print(f"  Circuit breaker: {'OPEN' if metrics['circuit_breaker_open'] else 'closed'}")
```

## Escalation Strategies

### Automatic Escalation

By default, the system automatically escalates when:

1. **Timeout** - Current tier exceeds timeout
2. **Unknown Result** - Solver returns "unknown"
3. **Too Complex** - Constraints exceed tier limit
4. **Failure** - Solver fails to solve

**Example**:
```python
# Automatic escalation (default)
result = verifier.verify("complex theorem with quantifiers")

# Escalation path: Tier 1 → Tier 2 → Tier 3
print(result.escalation_path)
# [VerificationTier.TIER1_Z3, VerificationTier.TIER2_LEANAIDE, VerificationTier.TIER3_LEAN4]
```

### Manual Escalation

You can manually control escalation:

```python
# Disable auto-escalation
import os
os.environ["AUTO_ESCALATE"] = "false"

verifier = TieredVerifier()

result = verifier.verify("complex theorem")

# Check if should escalate
if result.tier1_result and result.tier1_result.should_escalate():
    # Manually escalate
    result2 = verifier.escalate_tier(
        result.tier1_result,
        "complex theorem"
    )
```

### Escalation Limits

Prevent escalation beyond certain tier:

```python
# Set max tier to 2 (no Lean 4)
import os
os.environ["MAX_TIER"] = "2"

verifier = TieredVerifier()

# Will only try Tier 1 and Tier 2
result = verifier.verify("complex theorem")
assert result.escalation_path[-1] == VerificationTier.TIER2_LEANAIDE
```

## API Reference

### TieredVerifier

Main verification orchestrator.

#### Constructor

```python
TieredVerifier(config: Optional[TieredVerifierConfig] = None)
```

**Parameters**:
- `config` - Optional configuration (defaults to environment variables)

#### Methods

##### verify()

Main verification entry point with automatic tier selection and escalation.

```python
verify(
    problem: str,
    constraints: Optional[List[Any]] = None,
    variables: Optional[List[Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
) -> UnifiedVerificationResult
```

**Parameters**:
- `problem` - Problem statement (natural language or formal)
- `constraints` - Optional list of constraints
- `variables` - Optional list of variables
- `metadata` - Optional metadata (e.g., selection_strategy, preferred_tier)
- `correlation_id` - Optional correlation ID for tracing

**Returns**: `UnifiedVerificationResult`

##### verify_with_tier()

Verify with a specific tier (no automatic escalation).

```python
verify_with_tier(
    problem: str,
    tier: VerificationTier,
    constraints: Optional[List[Any]] = None,
    variables: Optional[List[Any]] = None,
    correlation_id: Optional[str] = None
) -> Union[Z3VerificationResult, LeanAideVerificationResult, Lean4VerificationResult]
```

**Parameters**:
- `problem` - Problem statement
- `tier` - Specific tier to use
- `constraints` - Optional list of constraints
- `variables` - Optional list of variables
- `correlation_id` - Optional correlation ID

**Returns**: Tier-specific verification result

##### escalate_tier()

Escalate to next tier.

```python
escalate_tier(
    current_result: Union[Z3VerificationResult, LeanAideVerificationResult],
    problem: str,
    constraints: Optional[List[Any]] = None,
    variables: Optional[List[Any]] = None,
    correlation_id: Optional[str] = None
) -> Union[LeanAideVerificationResult, Lean4VerificationResult]
```

**Parameters**:
- `current_result` - Result from current tier
- `problem` - Problem statement
- `constraints` - Optional list of constraints
- `variables` - Optional list of variables
- `correlation_id` - Optional correlation ID

**Returns**: Result from next tier

##### combine_results()

Combine results from multiple tiers.

```python
combine_results(
    results: List[Union[Z3VerificationResult, LeanAideVerificationResult, Lean4VerificationResult]],
    correlation_id: Optional[str] = None
) -> UnifiedVerificationResult
```

**Parameters**:
- `results` - List of tier results
- `correlation_id` - Optional correlation ID

**Returns**: `UnifiedVerificationResult`

### UnifiedVerificationResult

Result object combining all tier results.

#### Attributes

- `correlation_id: str` - Correlation ID for tracing
- `problem_class: ProblemClass` - Problem classification
- `problem_domain: ProblemDomain` - Problem domain
- `tier1_result: Optional[Z3VerificationResult]` - Tier 1 result
- `tier2_result: Optional[LeanAideVerificationResult]` - Tier 2 result
- `tier3_result: Optional[Lean4VerificationResult]` - Tier 3 result
- `final_status: VerificationStatus` - Final verification status
- `successful_tier: Optional[VerificationTier]` - Tier that succeeded
- `confidence: float` - Confidence in result (0.0 to 1.0)
- `escalation_path: List[VerificationTier]` - Tiers tried in order
- `escalation_reasons: List[str]` - Reasons for escalation
- `total_execution_time_ms: float` - Total execution time
- `total_constraints_checked: int` - Total constraints checked

#### Methods

##### is_successful()

```python
is_successful() -> bool
```

Check if verification was successful.

**Returns**: `True` if successful, `False` otherwise

##### get_successful_result()

```python
get_successful_result() -> Optional[Union[Z3VerificationResult, LeanAideVerificationResult, Lean4VerificationResult]]
```

Get the successful tier result.

**Returns**: Successful tier result or `None`

##### get_summary()

```python
get_summary() -> str
```

Get human-readable summary.

**Returns**: Summary string

## Best Practices

### 1. Start with Automatic Selection

Let the system select the appropriate tier:

```python
# Good: Automatic selection
result = verifier.verify("Find x such that x > 0")

# Less ideal: Force specific tier without reason
result = verifier.verify_with_tier("Find x such that x > 0", VerificationTier.TIER3_LEAN4)
```

### 2. Use Correlation IDs for Tracing

Track verification requests:

```python
import uuid

correlation_id = str(uuid.uuid4())
result = verifier.verify("problem", correlation_id=correlation_id)

# Later...
status = verifier.get_verification_status(correlation_id)
```

### 3. Check Confidence Levels

Consider confidence in results:

```python
result = verifier.verify("complex theorem")

if result.is_successful():
    if result.confidence >= 0.9:
        # High confidence - trust result
        use_result(result)
    elif result.confidence >= 0.7:
        # Medium confidence - may want to verify
        verify_manually(result)
    else:
        # Low confidence - escalate or verify
        escalate_or_verify(result)
```

### 4. Monitor Performance

Track solver performance:

```python
# After running multiple verifications
stats = verifier.selector.get_performance_stats()

# Check if any solver needs attention
for tier, metrics in stats.items():
    if metrics['success_rate'] < 0.5:
        print(f"Warning: {tier} has low success rate")
    if metrics['circuit_breaker_open']:
        print(f"Warning: {tier} circuit breaker is open")
```

### 5. Handle Failures Gracefully

Always handle potential failures:

```python
try:
    result = verifier.verify("complex problem")

    if result.is_successful():
        # Process result
        process_result(result)
    else:
        # Handle failure
        handle_failure(result)

except Exception as e:
    # Log error
    logger.error(f"Verification failed: {e}")

    # Fallback or retry
    fallback_or_retry()
```

## Troubleshooting

### Issue: Z3 Not Found

**Symptoms**:
```
FileNotFoundError: Z3 executable not found
```

**Solution**:
```bash
# Install Z3
curl -L https://github.com/Z3Prover/z3/releases/download/z3-4.12.6/z3-4.12.6-x64-glibc-2.35.zip -o z3.zip
unzip z3.zip
sudo mv z3-4.12.6-x64-glibc-2.35/bin/z3 /usr/local/bin/

# Verify installation
z3 --version
```

### Issue: Lean 4 Not Found

**Symptoms**:
```
FileNotFoundError: Lean 4 executable not found
```

**Solution**:
```bash
# Install Lean 4
curl -L https://github.com/leanprover/lean4/releases/download/v4.6.0/lean-4.6.0-linux.tar.gz -o lean4.tar.gz
tar -xzf lean4.tar.gz
sudo mv lean-4.6.0-linux/lean /usr/local/bin/

# Verify installation
lean --version
```

### Issue: Import Errors

**Symptoms**:
```
ModuleNotFoundError: No module named 'verification_result'
```

**Solution**:
```bash
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or install in development mode
pip install -e .
```

### Issue: Circuit Breaker Open

**Symptoms**:
```
Circuit breaker open, refusing request
```

**Solution**:
```python
# Reset circuit breaker
verifier.selector.reset_performance_stats()

# Or adjust threshold
import os
os.environ["Z3_FAILURE_THRESHOLD"] = "10"  # Increase threshold
```

### Issue: Verification Timeout

**Symptoms**:
```
Verification timeout after X ms
```

**Solution**:
```python
# Increase timeout for specific tier
import os
os.environ["TIER2_TIMEOUT_MS"] = "120000"  # 2 minutes

# Or disable auto-escalation
os.environ["AUTO_ESCALATE"] = "false"
```

### Issue: Low Confidence Results

**Symptoms**:
```
Warning: Verification confidence is 70%
```

**Solution**:
```python
# Manually escalate to higher tier
result = verifier.verify_with_tier(
    "complex theorem",
    tier=VerificationTier.TIER3_LEAN4
)

# Or require higher minimum confidence
os.environ["MIN_CONFIDENCE_THRESHOLD"] = "0.9"
```

## Conclusion

The Tiered Verification System provides a powerful, flexible approach to formal verification. By combining the speed of Z3, the AI assistance of LeanAide, and the rigor of Lean 4, it can handle a wide range of verification problems efficiently.

For more information, see:
- `ARCHITECTURE.md` - Detailed architecture documentation
- `README.md` - Installation and quick start guide
- `CLAUDE.md` - Project constitution and principles
