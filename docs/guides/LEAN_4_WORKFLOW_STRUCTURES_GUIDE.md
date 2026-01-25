# Lean 4 / LeanAide Integration Guide for workflow_structures.py

## Overview

This document describes the Lean 4 / LeanAide integration added to `workflow_structures.py`. The integration provides comprehensive support for formal mathematical verification within the Sovereign-Grade Decomposition Workflow.

## Key Features

1. **New Data Structures**: Lean 4 specific dataclasses for proofs, theorems, and verification
2. **Extended Workflow Structures**: Enhanced VerificationReport, SubProblem, and GauntletDefinition with mathematical verification capabilities
3. **JSON Serialization**: Full support for serializing/deserializing Lean 4 structures
4. **Validation Logic**: Built-in validation for mathematical content
5. **Backward Compatibility**: All new fields have sensible defaults, ensuring existing code continues to work

## New Enums

### MathematicalDomain

Defines mathematical domains for classification and verification.

```python
class MathematicalDomain(Enum):
    ALGEBRA = "algebra"
    ANALYSIS = "analysis"
    TOPOLOGY = "topology"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    GEOMETRY = "geometry"
    LOGIC = "logic"
    SET_THEORY = "set_theory"
    CATEGORY_THEORY = "category_theory"
    LINEAR_ALGEBRA = "linear_algebra"
    CALCULUS = "calculus"
    PROBABILITY = "probability"
    GENERAL = "general"
```

### VerificationMethod

Defines available verification methods.

```python
class VerificationMethod(Enum):
    MANUAL = "manual"
    AUTOMATED_TESTING = "automated_testing"
    PEER_REVIEW = "peer_review"
    LEAN4 = "lean4"
    HYBRID = "hybrid"
    STATISTICAL = "statistical"
    CROSS_VALIDATION = "cross_validation"
```

### LeanProofStatus

Defines the status of Lean 4 proof verification.

```python
class LeanProofStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ERROR = "error"
```

## New Dataclasses

### LeanProof

Represents a Lean 4 formal proof with metadata.

**Attributes:**
- `proof_id`: Unique identifier
- `theorem_name`: Name of the theorem
- `lean_code`: Lean 4 proof code
- `natural_language_statement`: Natural language statement
- `proof_status`: Verification status (LeanProofStatus)
- `domain`: Mathematical domain (MathematicalDomain)
- `complexity_score`: Complexity (1-10)
- `proof_steps`: List of proof steps
- `dependencies`: List of dependencies
- `verification_time`: Time for verification
- `elaborated_type`: Elaborated Lean type
- `proof_obligations`: Proof obligations
- `tactics_used`: Lean tactics used
- `metadata`: Additional metadata
- `timestamp`: Creation timestamp

**Methods:**
- `to_dict()`: Convert to dictionary for JSON serialization
- `from_dict(data)`: Create from dictionary
- `validate()`: Validate proof structure and content

**Example:**
```python
from workflow_structures import LeanProof, LeanProofStatus, MathematicalDomain

proof = LeanProof(
    proof_id="proof_001",
    theorem_name="infinitely_many_primes",
    lean_code="theorem infinitely_many_primes : Infinite {p : Nat | Nat.Prime p} := by sorry",
    natural_language_statement="There are infinitely many prime numbers",
    proof_status=LeanProofStatus.PENDING,
    domain=MathematicalDomain.NUMBER_THEORY,
    complexity_score=5,
)

# Convert to JSON
import json
proof_dict = proof.to_dict()
proof_json = json.dumps(proof_dict)

# Validate
errors = proof.validate()
if errors:
    print(f"Validation errors: {errors}")
```

### LeanTheorem

Represents a mathematical theorem with Lean 4 formalization.

**Attributes:**
- `theorem_id`: Unique identifier
- `name`: Theorem name
- `statement`: Natural language statement
- `lean_code`: Lean 4 formal statement
- `domain`: Mathematical domain
- `keywords`: Relevant keywords
- `difficulty`: Difficulty (1-10)
- `is_verified`: Whether verified
- `proof`: Associated LeanProof (optional)
- `related_theorems`: Related theorem IDs
- `references`: Academic references
- `metadata`: Additional metadata

**Example:**
```python
from workflow_structures import LeanTheorem, MathematicalDomain

theorem = LeanTheorem(
    theorem_id="thm_001",
    name="Infinitely Many Primes",
    statement="There are infinitely many prime numbers",
    lean_code="theorem infinitely_many_primes : Infinite {p : Nat | Nat.Prime p}",
    domain=MathematicalDomain.NUMBER_THEORY,
    keywords=["prime", "infinite", "number theory"],
    difficulty=6,
)
```

### LeanVerificationResult

Result of Lean 4 formal verification.

**Attributes:**
- `verification_id`: Unique identifier
- `success`: Whether verification succeeded
- `theorem_id`: ID of theorem verified
- `proof_id`: ID of proof used
- `verification_method`: Method used (VerificationMethod)
- `status`: Detailed status (LeanProofStatus)
- `confidence_score`: Confidence (0-1)
- `verification_time`: Time taken
- `proof_steps`: Steps in proof
- `remaining_obligations`: Unproven obligations
- `errors`: List of errors
- `warnings`: List of warnings
- `server_used`: Whether LeanAide server was used
- `fallback_used`: Whether fallback was used
- `lean_output`: Raw Lean output
- `metadata`: Additional metadata
- `timestamp`: Verification timestamp

**Example:**
```python
from workflow_structures import LeanVerificationResult, VerificationMethod, LeanProofStatus

verification = LeanVerificationResult(
    verification_id="ver_001",
    success=True,
    theorem_id="thm_001",
    proof_id="proof_001",
    verification_method=VerificationMethod.LEAN4,
    status=LeanProofStatus.VERIFIED,
    confidence_score=0.95,
    verification_time=2.5,
)
```

### MathematicalComponent

A mathematical component extracted from a problem or solution.

**Attributes:**
- `component_id`: Unique identifier
- `type`: Type ("theorem", "lemma", "equation", etc.)
- `name`: Component name
- `statement`: Mathematical statement
- `domain`: Mathematical domain
- `complexity`: Complexity (1-10)
- `dependencies`: Dependency IDs
- `formalized`: Whether formalized in Lean
- `lean_code`: Lean code if formalized
- `verification_status`: Verification status
- `metadata`: Additional metadata

**Example:**
```python
from workflow_structures import MathematicalComponent, MathematicalDomain, LeanProofStatus

component = MathematicalComponent(
    component_id="comp_001",
    type="theorem",
    name="Infinitely Many Primes",
    statement="There are infinitely many primes",
    domain=MathematicalDomain.NUMBER_THEORY,
    complexity=5,
    formalized=True,
    lean_code="theorem infinitely_many_primes : Infinite {p : Nat | Nat.Prime p}",
    verification_status=LeanProofStatus.VERIFIED,
)
```

## Extended Structures

### VerificationReport (Extended)

The `VerificationReport` class has been extended with Lean 4 support.

**New Fields:**
- `lean_verification`: Optional LeanVerificationResult
- `verification_method`: VerificationMethod (default: PEER_REVIEW)
- `mathematical_verified`: bool (default: False)
- `formal_proof_available`: bool (default: False)
- `mathematical_confidence`: float (default: 0.0)
- `mathematical_components_verified`: List[str] (default: empty)

**Example:**
```python
from workflow_structures import VerificationReport, LeanVerificationResult, VerificationMethod

# Create with Lean 4 verification
lean_result = LeanVerificationResult(
    verification_id="ver_001",
    success=True,
    theorem_id="thm_001",
)

report = VerificationReport(
    solution_attempt_id="attempt_001",
    gauntlet_name="verification_gauntlet",
    is_approved=True,
    reports_by_judge=[],
    lean_verification=lean_result,
    verification_method=VerificationMethod.LEAN4,
    mathematical_verified=True,
    formal_proof_available=True,
    mathematical_confidence=0.95,
    mathematical_components_verified=["thm_001"],
)
```

### SubProblem (Extended)

The `SubProblem` class has been extended with mathematical components.

**New Fields:**
- `mathematical_components`: List[MathematicalComponent] (default: empty)
- `requires_formal_verification`: bool (default: False)
- `mathematical_domain`: Optional[MathematicalDomain] (default: None)
- `formal_verification_enabled`: bool (default: False)
- `mathematical_properties`: List[str] (default: empty)
- `lean_theorems`: List[LeanTheorem] (default: empty)

**Example:**
```python
from workflow_structures import SubProblem, MathematicalComponent, LeanTheorem, MathematicalDomain

# Create mathematical component
math_component = MathematicalComponent(
    component_id="comp_001",
    type="theorem",
    name="Infinitely Many Primes",
    statement="There are infinitely many primes",
    domain=MathematicalDomain.NUMBER_THEORY,
)

# Create Lean theorem
lean_theorem = LeanTheorem(
    theorem_id="thm_001",
    name="Infinitely Many Primes",
    statement="There are infinitely many primes",
    lean_code="theorem infinitely_many_primes : Infinite {p : Nat | Nat.Prime p}",
    domain=MathematicalDomain.NUMBER_THEORY,
)

# Create SubProblem with mathematical fields
subproblem = SubProblem(
    id="sub_001",
    description="Prove there are infinitely many primes",
    mathematical_components=[math_component],
    requires_formal_verification=True,
    mathematical_domain=MathematicalDomain.NUMBER_THEORY,
    formal_verification_enabled=True,
    mathematical_properties=["infinite", "prime", "existence"],
    lean_theorems=[lean_theorem],
)
```

### GauntletDefinition (Extended)

The `GauntletDefinition` class has been extended with formal verification support.

**New Fields:**
- `formal_verification_enabled`: bool (default: False)
- `verification_methods`: List[VerificationMethod] (default: [PEER_REVIEW])
- `mathematical_requirements`: Dict[str, Any] (default: empty)
- `proof_generation_enabled`: bool (default: False)
- `automatic_formalization`: bool (default: False)
- `formal_verification_threshold`: float (default: 0.9)
- `lean_verification_config`: Dict[str, Any] (default: empty)

**Example:**
```python
from workflow_structures import GauntletDefinition, VerificationMethod

gauntlet = GauntletDefinition(
    name="mathematical_verification_gauntlet",
    team_name="gold_team",
    rounds=[],
    formal_verification_enabled=True,
    verification_methods=[VerificationMethod.LEAN4, VerificationMethod.PEER_REVIEW],
    proof_generation_enabled=True,
    automatic_formalization=True,
    formal_verification_threshold=0.95,
    lean_verification_config={
        "timeout": 300,
        "max_complexity": 8,
    },
)
```

## JSON Serialization

All new dataclasses support JSON serialization through `to_dict()` and `from_dict()` methods.

**Example:**
```python
import json
from workflow_structures import LeanProof

# Create and serialize
proof = LeanProof(
    proof_id="proof_001",
    theorem_name="test_theorem",
    lean_code="theorem test : True := by trivial",
    natural_language_statement="Test theorem",
)

# To JSON
proof_dict = proof.to_dict()
proof_json = json.dumps(proof_dict, indent=2)

# From JSON
loaded_dict = json.loads(proof_json)
loaded_proof = LeanProof.from_dict(loaded_dict)

assert loaded_proof.proof_id == proof.proof_id
```

## Validation

All new dataclasses include validation logic to ensure data integrity.

**Example:**
```python
from workflow_structures import LeanProof

proof = LeanProof(
    proof_id="proof_001",
    theorem_name="test_theorem",
    lean_code="theorem test : True := by trivial",
    natural_language_statement="Test theorem",
)

errors = proof.validate()
if errors:
    print(f"Validation errors: {errors}")
else:
    print("Proof is valid!")
```

## Backward Compatibility

All new fields have sensible default values, ensuring existing code continues to work without modifications.

**Example:**
```python
from workflow_structures import VerificationReport

# Old code still works - new fields use defaults
report = VerificationReport(
    solution_attempt_id="attempt_001",
    gauntlet_name="old_gauntlet",
    is_approved=True,
    reports_by_judge=[],
)

# New fields are set to safe defaults
assert report.lean_verification is None
assert report.verification_method == VerificationMethod.PEER_REVIEW
assert report.mathematical_verified is False
```

## Integration with Existing Workflow

The Lean 4 integration is designed to work seamlessly with the existing workflow engine:

1. **Content Analysis**: Extract mathematical components from problem statements
2. **Decomposition**: Tag subproblems with mathematical domains and verification requirements
3. **Solution Generation**: Generate solutions with Lean 4 formal proofs when enabled
4. **Gauntlet Verification**: Use formal verification as part of the gauntlet process
5. **Knowledge Extraction**: Store verified theorems and proofs for reuse

## Usage Example: Complete Workflow

```python
from workflow_structures import (
    SubProblem, MathematicalComponent, LeanTheorem,
    VerificationReport, LeanVerificationResult,
    MathematicalDomain, VerificationMethod, LeanProofStatus
)

# 1. Create a mathematical subproblem
math_component = MathematicalComponent(
    component_id="comp_001",
    type="theorem",
    name="Infinitely Many Primes",
    statement="There are infinitely many primes",
    domain=MathematicalDomain.NUMBER_THEORY,
)

lean_theorem = LeanTheorem(
    theorem_id="thm_001",
    name="Infinitely Many Primes",
    statement="There are infinitely many primes",
    lean_code="theorem infinitely_many_primes : Infinite {p : Nat | Nat.Prime p}",
    domain=MathematicalDomain.NUMBER_THEORY,
)

subproblem = SubProblem(
    id="sub_001",
    description="Prove there are infinitely many primes",
    mathematical_components=[math_component],
    requires_formal_verification=True,
    mathematical_domain=MathematicalDomain.NUMBER_THEORY,
    formal_verification_enabled=True,
    lean_theorems=[lean_theorem],
)

# 2. Create verification result
verification = LeanVerificationResult(
    verification_id="ver_001",
    success=True,
    theorem_id="thm_001",
    verification_method=VerificationMethod.LEAN4,
    status=LeanProofStatus.VERIFIED,
    confidence_score=0.95,
)

# 3. Create verification report
report = VerificationReport(
    solution_attempt_id="attempt_001",
    gauntlet_name="verification_gauntlet",
    is_approved=True,
    reports_by_judge=[],
    lean_verification=verification,
    verification_method=VerificationMethod.LEAN4,
    mathematical_verified=True,
    formal_proof_available=True,
    mathematical_confidence=0.95,
)

print(f"SubProblem: {subproblem.description}")
print(f"Mathematical Domain: {subproblem.mathematical_domain.value}")
print(f"Requires Formal Verification: {subproblem.requires_formal_verification}")
print(f"Verification Approved: {report.is_approved}")
print(f"Mathematically Verified: {report.mathematical_verified}")
```

## Testing

Run the comprehensive test suite to verify the integration:

```bash
python test_lean_workflow_structures.py
```

This will test:
- All new enums
- All new dataclasses
- JSON serialization/deserialization
- Validation logic
- Extended structures
- Backward compatibility

## Best Practices

1. **Always Use Enums**: Use the provided enums (MathematicalDomain, VerificationMethod, LeanProofStatus) instead of string literals
2. **Validate Input**: Call `validate()` on user-provided data before using it
3. **Check Verification Status**: Always check `proof_status` and `verification_method` before using verification results
4. **Handle Defaults Gracefully**: When working with legacy code, be prepared for new fields to have default values
5. **Use JSON for Storage**: Use `to_dict()` and `from_dict()` for database storage and API responses

## Future Enhancements

Potential future additions:
- Integration with LeanAide MCP tools for automated proof generation
- Support for Lean 4 code highlighting and pretty-printing
- Proof strategy suggestions based on mathematical domain
- Automatic lemma discovery and suggestion
- Integration with Mathlib for theorem lookup
- Parallel verification of multiple components
- Proof cache for frequently used theorems

## References

- [Lean 4 Documentation](https://leanprover.github.io/lean4/doc/)
- [LeanAide Integration](./leanaide_mcp_tools.py)
- [Lean 4 Integration](./lean4_integration.py)
- [Lean 4 Integration Documentation](./Lean_4_Integration_Documentation.md)
