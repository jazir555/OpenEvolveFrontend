# LeanAide Red-Flagging System - Complete Guide

## Overview

The LeanAide Red-Flagging System provides comprehensive quality control for Lean 4 proofs, extending the MDAP red-flagging framework with Lean-specific validation across multiple dimensions:

- **Syntax Validation**: Lean 4 syntax checking
- **Semantic Validation**: Type checking, tactic applicability
- **Structural Validation**: Proof length, circular reasoning detection
- **Quality Validation**: Elegance, clarity, efficiency metrics
- **Verification**: Integration with LeanAide for actual proof verification

## Architecture

```
LeanRedFlagRules
    ↓
LeanRedFlagger
    ↓
LeanProofValidator + LeanProofQualityScorer
    ↓
ValidationResult + LeanQualityScore
```

## Components

### 1. Data Structures

#### `LeanProof`
Represents a Lean 4 proof with metadata:

```python
@dataclass
class LeanProof:
    code: str                      # Full Lean code
    name: str                      # Proof name
    proof_type: LeanProofType      # theorem, lemma, def, etc.
    statement: str                 # Mathematical statement
    tactics: List[str]             # Tactics used
    imports: List[str]             # Required imports
    dependencies: List[str]        # Dependencies
    line_count: int                # Number of lines
    tactic_count: int              # Number of tactics
    has_sorry: bool                # Contains sorry?
    sorry_count: int               # Number of sorries
```

#### `LeanQualityScore`
Multi-dimensional quality score:

```python
@dataclass
class LeanQualityScore:
    overall_score: float           # 0-1
    elegance: float                # Tactic diversity, conciseness
    clarity: float                 # Understandability
    efficiency: float              # Minimal redundancy
    correctness: float             # Verified, no sorries
    flags: List[str]               # Issues found
    suggestions: List[str]         # Improvement suggestions
```

#### `ValidationResult`
Comprehensive validation result:

```python
@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]
    quality_score: LeanQualityScore
    verification_result: VerificationResult
```

### 2. Rules Configuration

#### `LeanRedFlagRules`
Enhanced red-flag rules for Lean 4:

```python
@dataclass
class LeanRedFlagRules(RedFlagRules):
    # Structural limits
    max_proof_length: int = 500           # Max lines
    max_tactic_sequence: int = 100        # Max tactics

    # Completeness
    require_no_sorries: bool = True
    max_sorry_count: int = 0

    # Quality thresholds
    min_elegance_score: float = 0.3
    max_simplification_ratio: float = 0.8

    # Syntax
    require_lean_keywords: bool = True

    # Semantic checking
    check_tactic_applicability: bool = True
    check_imports: bool = True

    # Tactic restrictions
    forbidden_tactics: List[str] = ["admit"]
    required_imports: List[str] = []

    # Complexity
    max_nesting_depth: int = 20
    min_tactic_diversity: int = 2
```

### 3. Main Classes

#### `LeanRedFlagger`
Main red-flagging engine:

```python
class LeanRedFlagger(RedFlagger):
    def __init__(self, rules: LeanRedFlagRules)

    # Main check
    def is_flagged(self, proof: LeanProof) -> Tuple[bool, List[str]]

    # Category checks
    def check_syntax(self, proof: LeanProof) -> List[str]
    def check_semantics(self, proof: LeanProof) -> List[str]
    def check_structure(self, proof: LeanProof) -> List[str]
    def check_quality(self, proof: LeanProof) -> List[str]
    def check_verification(self, proof: LeanProof) -> List[str]

    # Tactic applicability
    def check_tactic_applicability(
        self, tactic: str, state: LeanProofState
    ) -> bool
```

#### `LeanProofValidator`
Comprehensive validation:

```python
class LeanProofValidator:
    def __init__(
        self,
        rules: Optional[LeanRedFlagRules] = None,
        verification_engine: Optional[Lean4VerificationEngine] = None
    )

    def validate_syntax(self, code: str) -> Tuple[bool, List[str]]
    def validate_semantics(self, code: str) -> Tuple[bool, List[str]]
    def validate_structure(self, proof: LeanProof) -> Tuple[bool, List[str]]
    def verify_with_leanaide(self, code: str) -> Tuple[bool, List[str]]
    def full_validation(self, proof: LeanProof) -> ValidationResult
```

#### `LeanProofQualityScorer`
Multi-dimensional quality scoring:

```python
class LeanProofQualityScorer:
    def __init__(self, rules: LeanRedFlagRules)

    def score_proof(self, proof: LeanProof) -> LeanQualityScore
    def score_elegance(self, proof: LeanProof) -> float
    def score_clarity(self, proof: LeanProof) -> float
    def score_efficiency(self, proof: LeanProof) -> float
    def score_correctness(self, proof: LeanProof) -> float
```

## Usage Examples

### Basic Red-Flagging

```python
from leanaide_redflagging import create_lean_red_flagger, parse_lean_code

# Create flagger with custom rules
flagger = create_lean_red_flagger(
    max_proof_length=500,
    require_no_sorries=True,
    min_elegance_score=0.3
)

# Parse proof code
proof = parse_lean_code("""
theorem add_zero (n : Nat) : n + 0 = n := by
  sorry
""")

# Check if flagged
is_flagged, reasons = flagger.is_flagged(proof)

if is_flagged:
    print(f"Proof flagged: {reasons}")
else:
    print("Proof passed all checks")
```

### Comprehensive Validation

```python
from leanaide_redflagging import create_lean_validator

# Create validator
validator = create_lean_validator(
    rules=LeanRedFlagRules(
        require_no_sorries=True,
        min_elegance_score=0.4
    )
)

# Parse proof
proof = parse_lean_code(lean_code)

# Full validation
result = validator.full_validation(proof)

if result.valid:
    print("Proof is valid!")
    print(f"Quality Score: {result.quality_score.overall_score:.2f}")
else:
    print(f"Validation errors: {result.errors}")

# Check individual dimensions
if result.quality_score:
    print(f"Elegance: {result.quality_score.elegance:.2f}")
    print(f"Clarity: {result.quality_score.clarity:.2f}")
    print(f"Efficiency: {result.quality_score.efficiency:.2f}")
    print(f"Correctness: {result.quality_score.correctness:.2f}")
```

### Quality Scoring

```python
from leanaide_redflagging import create_lean_quality_scorer

# Create scorer
scorer = create_lean_quality_scorer()

# Parse proof
proof = parse_lean_code(lean_code)

# Score proof
score = scorer.score_proof(proof)

print(f"Overall: {score.overall_score:.2f}")
print(f"Elegance: {score.elegance:.2f}")
print(f"Clarity: {score.clarity:.2f}")
print(f"Efficiency: {score.efficiency:.2f}")
print(f"Correctness: {score.correctness:.2f}")

# Get improvement suggestions
if score.suggestions:
    print("\nSuggestions:")
    for suggestion in score.suggestions:
        print(f"  - {suggestion}")
```

### Quick Checks

```python
from leanaide_redflagging import (
    quick_red_flag_check,
    comprehensive_validation,
    score_proof_quality
)

# Quick red-flag check
is_flagged, reasons = quick_red_flag_check(lean_code)

# Comprehensive validation
result = comprehensive_validation(lean_code)

# Quality scoring
score = score_proof_quality(lean_code)
```

### Batch Processing

```python
from leanaide_redflagging import create_lean_validator

validator = create_lean_validator()

proofs = [
    proof1_code,
    proof2_code,
    proof3_code,
]

results = []
for proof_code in proofs:
    proof = parse_lean_code(proof_code)
    result = validator.full_validation(proof)
    results.append(result)

# Analyze results
valid_count = sum(1 for r in results if r.valid)
avg_quality = sum(r.quality_score.overall_score for r in results) / len(results)

print(f"Valid: {valid_count}/{len(results)}")
print(f"Average Quality: {avg_quality:.2f}")
```

## Red-Flag Categories

### 1. Syntax Checks

Validates Lean 4 syntax:

- **Malformed tactic syntax**: Tactics with incorrect syntax
- **Missing keywords**: Missing `by`, `:=`, etc.
- **Unmatched parentheses**: Unbalanced brackets/parens
- **Invalid identifiers**: Poorly formed identifiers

Example flags:
```
missing_lean_keywords
unmatched_parentheses
unknown_tactic:foo_tactic:line_42
```

### 2. Semantic Checks

Validates semantic correctness:

- **Tactic applicability**: Tactics applied to appropriate goals
- **Type mismatches**: Type correctness (heuristic)
- **Undefined constants**: References to undefined symbols
- **Missing imports**: Required imports not present

Example flags:
```
forbidden_tactic:admit
repetitive_tactics
simp_without_args
rw_without_args
```

### 3. Structural Checks

Validates proof structure:

- **Proof too long**: Exceeds `max_proof_length`
- **Too many tactics**: Exceeds `max_tactic_sequence`
- **Circular reasoning**: Proof might be circular
- **Excessive nesting**: Too many nesting levels
- **Low tactic diversity**: Not enough variety in tactics

Example flags:
```
proof_too_long:523_lines
too_many_tactics:150_tactics
circular_reasoning
excessive_nesting:25_levels
low_tactic_diversity:2_unique
```

### 4. Quality Checks

Validates proof quality:

- **Too many sorries**: Incomplete proof
- **Excessive simp**: Over-reliance on simplification
- **Automation overuse**: Too much automation
- **Poor naming**: Generic names (theorem1, etc.)

Example flags:
```
contains_sorry:3_instances
excessive_simp:85%_ratio
over_reliance_on_automation
poor_naming_conventions
```

### 5. Verification Checks

Validates with LeanAide:

- **Verification failed**: LeanAide verification fails
- **Elaboration errors**: Type checking errors
- **Remaining goals**: Proof incomplete

Example flags:
```
verification_failed
verification_error:type_mismatch
verification_passed_with_sorry
```

## Quality Dimensions

### Elegance (0-1)
Measures tactic diversity and proof conciseness:

**Factors**:
- Tactic diversity (unique/total ratio)
- Proof length (within bounds)
- Automation vs manual balance
- Creative tactic use

**Scoring**:
- High: Diverse tactics, appropriate length
- Medium: Some repetition or slightly long
- Low: Very repetitive or excessively long

### Clarity (0-1)
Measures understandability:

**Factors**:
- Naming quality
- Use of `have` statements
- Use of clarity tactics (`show`, `suffices`)
- Comments
- Code formatting

**Scoring**:
- High: Good names, well-organized, commented
- Medium: Acceptable naming, some structure
- Low: Generic names, no structure

### Efficiency (0-1)
Measures minimal redundancy:

**Factors**:
- Redundancy detection
- Optimal tactic use
- Unnecessary steps
- Proof length relative to complexity

**Scoring**:
- High: No redundancy, optimal tactics
- Medium: Some redundancy
- Low: Highly redundant patterns

### Correctness (0-1)
Measures proof completeness:

**Factors**:
- Presence of `sorry`/`admit`
- Proof completeness
- Verification status

**Scoring**:
- High: No sorries, verified
- Medium: Some sorries but mostly complete
- Low: Many sorries or incomplete

## Lean 4 Tactics Reference

### Basic Tactics
- `intro`, `intros`, `apply`, `exact`, `refine`
- `by`, `sorry`, `assumption`, `trivial`, `rfl`, `rwa`

### Rewrite Tactics
- `rw`, `rewrite`, `rwa`, `simp`, `dsimp`, `simp_rw`

### Induction Tactics
- `induction`, `cases`, `case`, `rcases`, `obtain`

### Logic Tactics
- `have`, `suffices`, `show`, `calc`, `by_contra`, `by_cases`
- `contrapose`, `refute`, `exfalso`, `absurd`

### Arithmetic Tactics
- `linarith`, `omega`, `ring`, `ring_nf`, `norm_num`
- `norm_cast`, `push_neg`, `positivity`

### Completion Tactics
- `tidy`, `aesop`, `solve_by_elim`, `hint`, `suggest`

### Library Tactics
- `library_search`, `exact?`, `apply?`, `simp?`

### Advanced Tactics
- `wlog`, `generalize`, `specialize`, `transitivity`
- `constructor`, `injection`, `injections`, `subst`

## Integration with LeanAide

The red-flagging system integrates with LeanAide for actual proof verification:

```python
from leanaide_redflagging import create_lean_validator
from lean4_integration import Lean4VerificationEngine, Lean4ServerConfig

# Create verification engine
server_config = Lean4ServerConfig(
    host="localhost",
    port=7654,
    enable_simulation_fallback=True
)

engine = Lean4VerificationEngine(
    server_url="http://localhost:7654",
    server_config=server_config
)

# Create validator with verification
validator = create_lean_validator(
    verification_engine=engine
)

# Validate with actual Lean verification
result = validator.full_validation(proof)

if result.verification_result:
    print(f"Verified: {result.verification_result.success}")
    if not result.verification_result.success:
        print(f"Errors: {result.verification_result.errors}")
```

## Advanced Usage

### Custom Rules

```python
from leanaide_redflagging import LeanRedFlagRules, LeanRedFlagger

# Define custom rules
rules = LeanRedFlagRules(
    max_proof_length=1000,
    max_tactic_sequence=200,
    require_no_sorries=False,  # Allow some sorries
    max_sorry_count=2,
    min_elegance_score=0.5,
    max_simplification_ratio=0.6,
    forbidden_tactics=["admit", "tidy"],  # Forbid specific tactics
    required_imports=["Mathlib"],  # Require Mathlib
    min_tactic_diversity=3
)

flagger = LeanRedFlagger(rules)
```

### Tactic Applicability Checking

```python
from leanaide_redflagging import LeanProofState, create_lean_red_flagger

flagger = create_lean_red_flagger()

# Define proof state
state = LeanProofState(
    goal="n + 0 = n",
    hypotheses=["n : Nat"],
    context={}
)

# Check if tactic is applicable
applicable = flagger.check_tactic_applicability("rw", state)

if not applicable:
    print("Tactic 'rw' may not be applicable to current goal")
```

### Filtering by Quality

```python
from leanaide_redflagging import score_proof_quality, parse_lean_code

proofs = [proof1, proof2, proof3]

high_quality_proofs = []
for proof_code in proofs:
    proof = parse_lean_code(proof_code)
    score = score_proof_quality(proof_code)

    if score.overall_score > 0.7 and score.correctness > 0.8:
        high_quality_proofs.append((proof, score))

print(f"Found {len(high_quality_proofs)} high-quality proofs")
```

## Best Practices

### 1. Choose Appropriate Rules

```python
# For student submissions (strict)
strict_rules = LeanRedFlagRules(
    require_no_sorries=True,
    min_elegance_score=0.5,
    max_simplification_ratio=0.5
)

# For research/development (lenient)
lenient_rules = LeanRedFlagRules(
    require_no_sorries=False,
    max_sorry_count=5,
    min_elegance_score=0.2,
    max_simplification_ratio=0.9
)
```

### 2. Use Category-Specific Checks

```python
flagger = create_lean_red_flagger()
proof = parse_lean_code(lean_code)

# Check specific category
syntax_errors = flagger.check_syntax(proof)
quality_issues = flagger.check_quality(proof)

# Handle differently
if syntax_errors:
    print("Syntax errors must be fixed")
if quality_issues:
    print("Quality suggestions:")
    for issue in quality_issues:
        print(f"  - {issue}")
```

### 3. Provide Actionable Feedback

```python
def provide_feedback(proof_code: str) -> str:
    result = comprehensive_validation(proof_code)

    feedback = []

    if not result.valid:
        feedback.append("❌ Validation Failed")
        feedback.extend([f"  - {e}" for e in result.errors[:5]])
    else:
        feedback.append("✅ Validation Passed")

    if result.quality_score:
        qs = result.quality_score
        feedback.append(f"\nQuality Score: {qs.overall_score:.1%}")
        feedback.append(f"  Elegance: {qs.elegance:.1%}")
        feedback.append(f"  Clarity: {qs.clarity:.1%}")
        feedback.append(f"  Efficiency: {qs.efficiency:.1%}")
        feedback.append(f"  Correctness: {qs.correctness:.1%}")

        if qs.suggestions:
            feedback.append("\n💡 Suggestions:")
            feedback.extend([f"  - {s}" for s in qs.suggestions[:3]])

    return "\n".join(feedback)
```

### 4. Batch Processing with Progress

```python
from tqdm import tqdm

def validate_proofs_batch(proof_codes: List[str]) -> List[ValidationResult]:
    validator = create_lean_validator()
    results = []

    for code in tqdm(proof_codes, desc="Validating proofs"):
        proof = parse_lean_code(code)
        result = validator.full_validation(proof)
        results.append(result)

    return results
```

## Testing

Run the test suite:

```bash
python -m pytest test_leanaide_redflagging.py -v
```

Run specific test categories:

```bash
# Test red-flagging
python -m pytest test_leanaide_redflagging.py::TestLeanRedFlagger -v

# Test quality scoring
python -m pytest test_leanaide_redflagging.py::TestLeanProofQualityScorer -v

# Test integration scenarios
python -m pytest test_leanaide_redflagging.py::TestIntegrationScenarios -v
```

## API Reference

See the module docstring in `leanaide_redflagging.py` for complete API documentation.

## Troubleshooting

### Lean 4 Integration Not Available

If you see warnings about Lean 4 integration not being available:

```python
# The system will fall back to static analysis only
# Verification checks will be skipped
# Syntax/semantic/structural/quality checks still work

# To enable full integration, ensure lean4_integration.py is available
```

### Performance Optimization

For large batches:

```python
# Use stricter rules to fail fast
rules = LeanRedFlagRules(
    max_proof_length=100,  # Fail fast on long proofs
    require_no_sorries=True  # Fail fast on sorries
)

# Or use parallel processing
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(validate, proof_codes))
```

## Summary

The LeanAide Red-Flagging System provides:

1. **Comprehensive Validation**: 5 categories of checks
2. **Quality Scoring**: 4-dimensional quality metrics
3. **LeanAide Integration**: Actual proof verification
4. **Flexible Configuration**: Customizable rules
5. **Actionable Feedback**: Clear flags and suggestions

Use it to:
- Validate student submissions
- Grade proof quality
- Filter proof databases
- Provide feedback on proof style
- Ensure proof completeness
