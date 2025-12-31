# LeanAide Red-Flagging System - Quick Reference

## Quick Start

```python
from leanaide_redflagging import quick_red_flag_check, comprehensive_validation

# Quick check
is_flagged, reasons = quick_red_flag_check(lean_code)
if is_flagged:
    print(f"Flagged: {reasons}")

# Full validation
result = comprehensive_validation(lean_code)
print(f"Valid: {result.valid}")
print(f"Quality: {result.quality_score.overall_score:.2f}")
```

## Factory Functions

| Function | Purpose |
|----------|---------|
| `create_lean_red_flagger()` | Create red-flagger |
| `create_lean_validator()` | Create validator |
| `create_lean_quality_scorer()` | Create scorer |
| `parse_lean_code()` | Parse code to LeanProof |
| `quick_red_flag_check()` | Quick red-flag check |
| `comprehensive_validation()` | Full validation |
| `score_proof_quality()` | Quality scoring |

## Common Rules

```python
# Strict (student submissions)
LeanRedFlagRules(
    require_no_sorries=True,
    min_elegance_score=0.5,
    max_simplification_ratio=0.5
)

# Lenient (development)
LeanRedFlagRules(
    require_no_sorries=False,
    max_sorry_count=5,
    min_elegance_score=0.2
)

# Custom
LeanRedFlagRules(
    max_proof_length=1000,
    forbidden_tactics=["admit", "tidy"],
    required_imports=["Mathlib"]
)
```

## Red-Flag Categories

| Category | Checks | Example Flags |
|----------|--------|---------------|
| **Syntax** | Keywords, parentheses, identifiers | `missing_keywords`, `unmatched_parens` |
| **Semantic** | Tactic applicability, imports | `forbidden_tactic`, `missing_imports` |
| **Structural** | Length, tactics, nesting | `proof_too_long`, `too_many_tactics` |
| **Quality** | Sorries, automation, naming | `contains_sorry`, `poor_naming` |
| **Verification** | LeanAide check | `verification_failed` |

## Quality Dimensions

| Dimension | What It Measures | Good Score |
|-----------|------------------|------------|
| **Elegance** | Tactic diversity, conciseness | > 0.6 |
| **Clarity** | Understandability, naming | > 0.7 |
| **Efficiency** | Minimal redundancy | > 0.6 |
| **Correctness** | No sorries, verified | > 0.8 |

## Lean 4 Tactics

### Common Tactics
```lean
intro, intros, apply, exact, refine  # Basic
rw, rewrite, simp, dsimp             # Rewrite
induction, cases, rcases             # Induction
have, show, suffices, calc           # Structure
linarith, ring, omega                # Arithmetic
```

### Potentially Problematic
```lean
sorry    # Incomplete proof (flagged)
admit    # Alternative to sorry (flagged)
simp     # Overuse indicates lack of understanding
aesop    # Automation - may hide reasoning
tidy     # Automation - may hide reasoning
```

## Usage Patterns

### Validate Single Proof
```python
from leanaide_redflagging import create_lean_validator

validator = create_lean_validator()
proof = parse_lean_code(lean_code)
result = validator.full_validation(proof)

if result.valid:
    print("✅ Valid")
else:
    print("❌ Errors:", result.errors)
```

### Batch Validation
```python
validator = create_lean_validator()
results = []

for code in proof_codes:
    proof = parse_lean_code(code)
    result = validator.full_validation(proof)
    results.append(result)

valid = sum(1 for r in results if r.valid)
print(f"Valid: {valid}/{len(results)}")
```

### Quality Comparison
```python
from leanaide_redflagging import score_proof_quality

score1 = score_proof_quality(proof1_code)
score2 = score_proof_quality(proof2_code)

print(f"Proof 1: {score1.overall_score:.2f}")
print(f"Proof 2: {score2.overall_score:.2f}")

if score1.elegance > score2.elegance:
    print("Proof 1 is more elegant")
```

### Filter by Quality
```python
high_quality = []
for code in proof_codes:
    score = score_proof_quality(code)
    if score.overall_score > 0.7:
        high_quality.append((code, score))
```

### Custom Feedback
```python
def format_feedback(result: ValidationResult) -> str:
    feedback = []
    feedback.append(f"Valid: {result.valid}")
    if result.errors:
        feedback.append(f"Errors: {', '.join(result.errors[:3])}")
    if result.quality_score:
        qs = result.quality_score
        feedback.append(f"Quality: {qs.overall_score:.1%}")
        if qs.suggestions:
            feedback.append(f"Suggestions: {qs.suggestions[0]}")
    return "\n".join(feedback)
```

## Red-Flag Messages

| Message | Meaning | Fix |
|---------|---------|-----|
| `contains_sorry` | Proof has sorry | Replace with actual proof |
| `proof_too_long` | Exceeds max length | Break into lemmas |
| `too_many_tactics` | Exceeds max tactics | Simplify proof |
| `excessive_simp` | Too much simp | Show explicit steps |
| `poor_naming` | Generic name | Use descriptive name |
| `verification_failed` | LeanAide rejected | Fix errors |
| `circular_reasoning` | Possible cycle | Restructure proof |
| `low_tactic_diversity` | Repetitive | Use varied tactics |

## Quality Suggestions

Common improvements:

```python
# Low elegance
suggestions = [
    "Consider using a wider variety of tactics",
    "Proof is quite long - consider breaking into lemmas"
]

# Low clarity
suggestions = [
    "Add intermediate 'have' statements",
    "Consider adding explanatory comments",
    "Use more descriptive naming"
]

# Low efficiency
suggestions = [
    "Look for redundant patterns to simplify",
    "Consider using more concise tactics"
]

# Low correctness
suggestions = [
    "Replace 'sorry' placeholders with actual proofs",
    "Complete incomplete proof steps"
]
```

## Testing

```bash
# Run all tests
python -m pytest test_leanaide_redflagging.py -v

# Run specific tests
python -m pytest test_leanaide_redflagging.py::TestLeanRedFlagger -v
python -m pytest test_leanaide_redflagging.py::TestLeanProofQualityScorer -v

# Run demo
python test_leanaide_redflagging.py
```

## Common Workflows

### Student Submission Check
```python
from leanaide_redflagging import create_lean_validator

validator = create_lean_validator(
    rules=LeanRedFlagRules(
        require_no_sorries=True,
        min_elegance_score=0.3
    )
)

result = validator.full_validation(parse_lean_code(submission))

if result.valid and result.quality_score.overall_score > 0.6:
    print("✅ Accept")
else:
    print("❌ Reject")
    print("Feedback:", result.quality_score.suggestions)
```

### Proof Database Filtering
```python
def filter_high_quality(proofs: List[str]) -> List[str]:
    return [
        p for p in proofs
        if score_proof_quality(p).overall_score > 0.7
    ]
```

### Comparative Analysis
```python
def compare_proofs(proof1: str, proof2: str) -> Dict:
    score1 = score_proof_quality(proof1)
    score2 = score_proof_quality(proof2)

    return {
        "winner": "proof1" if score1.overall_score > score2.overall_score else "proof2",
        "proof1": score1.to_dict(),
        "proof2": score2.to_dict()
    }
```

### Grading Rubric
```python
def grade_proof(proof_code: str) -> str:
    score = score_proof_quality(proof_code)

    if score.correctness < 0.5:
        return "F - Incomplete"
    if score.overall_score > 0.9:
        return "A - Excellent"
    if score.overall_score > 0.8:
        return "B - Good"
    if score.overall_score > 0.7:
        return "C - Acceptable"
    return "D - Needs Improvement"
```

## Tips

1. **Start with quick checks** - Use `quick_red_flag_check()` for fast validation
2. **Adjust rules for context** - Strict for students, lenient for development
3. **Provide suggestions** - Use quality score suggestions for actionable feedback
4. **Batch process** - Validate multiple proofs efficiently
5. **Check dimensions** - Look at individual quality dimensions for specific issues

## Integration

```python
# With LeanAide server
from lean4_integration import Lean4VerificationEngine
from leanaide_redflagging import create_lean_validator

engine = Lean4VerificationEngine(
    server_url="http://localhost:7654",
    server_config=Lean4ServerConfig()
)

validator = create_lean_validator(verification_engine=engine)
result = validator.full_validation(proof)

# Verification included in result
if result.verification_result:
    print(f"Verified: {result.verification_result.success}")
```

## Resources

- Full Guide: `LEANAIDE_REDFLAGGING_GUIDE.md`
- Test Suite: `test_leanaide_redflagging.py`
- Module: `leanaide_redflagging.py`
- Base MDAP: `mdap_engine.py`
- Lean Integration: `lean4_integration.py`
