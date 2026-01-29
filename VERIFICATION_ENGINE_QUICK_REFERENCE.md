# Verification Engine - Quick Reference

## Import

```python
from verification_engine import (
    VerificationEngine,
    VerificationReport,
    SuccessCriterion,
    SolutionQualityMetrics,
    create_default_criteria,
    compare_reports
)
```

## Initialize

```python
# Basic
engine = VerificationEngine()

# With config
engine = VerificationEngine(config={
    'strict_mode': False,
    'min_quality_threshold': 0.6,
    'enable_detailed_logging': True
})
```

## Common Operations

### Verify Solution

```python
# From requirements
criteria = engine.create_success_criteria([
    "Solution must be at least 80% complete",
    "Code must be clear and readable"
])

report = engine.verify_solution(solution, criteria)
print(f"Approved: {report.is_approved}")
print(f"Score: {report.verification_score:.2f}")
```

### Custom Criteria

```python
criteria = [
    SuccessCriterion(
        id="high_quality",
        description="Production quality",
        metric="maintainability",
        threshold=0.85,
        weight=1.5
    )
]
```

### Quality Metrics

```python
metrics = engine.calculate_quality_scores(solution)

# Access individual scores
print(f"Completeness: {metrics.completeness:.2f}")
print(f"Correctness: {metrics.correctness:.2f}")
print(f"Overall: {metrics.overall_score:.2f}")

# Custom weights
metrics.calculate_overall({
    'completeness': 0.4,
    'correctness': 0.6
})
```

### Verification Suite

```python
test_suite = [
    {'metric': 'completeness', 'threshold': 0.7},
    {'metric': 'correctness', 'threshold': 0.6},
    {'metric': 'security', 'threshold': 0.8}
]

report = engine.run_verification_suite(solution, test_suite)
```

## Solution Formats

### Dictionary
```python
solution = {
    'id': 'solution_001',
    'solution_content': 'def solve(): pass'
}
```

### sovereign_data_models
```python
from sovereign_data_models import SolutionAttempt

solution = SolutionAttempt(
    id="test",
    problem_id="problem",
    solution="def solve(): pass",
    score=0.8,
    timestamp=datetime.now()
)
```

### crewai_state_management
```python
from crewai_state_management import SolutionAttempt

solution = SolutionAttempt(
    sub_problem_id="sp_001",
    solution_content="def solve(): pass",
    confidence_score=0.75,
    execution_method="traditional"
)
```

## Report Fields

```python
report.solution_attempt_id    # Solution ID
report.gauntlet_name          # Gauntlet used
report.is_approved            # Pass/fail
report.reports_by_judge       # Judge reports
report.summary                # Text summary
report.quality_metrics        # QualityMetrics object
report.criteria_results       # Dict of criterion results
report.verification_score     # 0.0-1.0 score
report.metadata               # Additional info
```

## Quality Metrics

| Metric | Description |
|--------|-------------|
| completeness | Requirement coverage |
| correctness | Accuracy score |
| efficiency | Performance rating |
| clarity | Readability score |
| maintainability | Maintenance ease |
| scalability | Scaling ability |
| security | Security rating |
| test_coverage | Test percentage |
| overall_score | Weighted average |
| confidence | Assessment confidence |

## Criterion Categories

- `functional` - Functional requirements
- `non_functional` - Performance, scalability
- `security` - Security requirements
- `quality` - Code quality attributes

## Metric Keywords for Auto-Parsing

Requirements containing these keywords auto-detect metrics:

- **completeness**: "complete", "cover"
- **correctness**: "correct", "accurate"
- **security**: "secure", "security"
- **efficiency**: "efficient", "performance"
- **clarity**: "clear", "readable"
- **maintainability**: "maintain"
- **scalability**: "scale", "scalable"
- **test_coverage**: "test"

## Export Reports

```python
# To dictionary
report_dict = report.to_dict()

# To JSON
report_json = report.to_json()

# Save to file
with open('report.json', 'w') as f:
    f.write(report.to_json())
```

## History Management

```python
# Get history
history = engine.get_verification_history()

# Clear history
engine.clear_history()

# Filter history
approved_reports = [r for r in history if r.is_approved]
```

## Compare Reports

```python
comparison = compare_reports(report1, report2)

print(f"Score difference: {comparison['score_difference']:.2f}")
print(f"Approval changed: {comparison['approval_changed']}")
```

## Error Handling

```python
# Try-catch for invalid inputs
try:
    report = engine.verify_solution(solution, criteria)
except ValueError as e:
    print(f"Validation error: {e}")

# Check report for errors
if not report.is_approved:
    print(f"Failed: {report.summary}")
```

## Configuration Examples

```python
# Strict mode
engine = VerificationEngine(config={'strict_mode': True})

# High quality threshold
engine = VerificationEngine(config={'min_quality_threshold': 0.8})

# Minimal logging
engine = VerificationEngine(config={'enable_detailed_logging': False})
```

## Testing

```bash
# Run all tests
python -m pytest test_verification_engine.py -v

# Run specific test
python -m pytest test_verification_engine.py::TestVerificationEngine -v

# With coverage
python -m pytest test_verification_engine.py --cov=verification_engine
```

## Examples

```bash
# Run built-in examples
python verification_engine.py
```

## Common Patterns

### Batch Verification
```python
solutions = [sol1, sol2, sol3]
criteria = engine.create_success_criteria(["Complete solution"])

reports = [
    engine.verify_solution(sol, criteria)
    for sol in solutions
]

approved = [r for r in reports if r.is_approved]
print(f"Approved: {len(approved)}/{len(reports)}")
```

### Custom Quality Threshold
```python
def verify_with_threshold(solution, threshold):
    report = engine.verify_solution(
        solution,
        engine.create_success_criteria(["Complete"])
    )
    return report.verification_score >= threshold
```

### Filter by Quality
```python
def get_high_quality_solutions(solutions, min_score=0.8):
    results = []
    for sol in solutions:
        metrics = engine.calculate_quality_scores(sol)
        if metrics.overall_score >= min_score:
            results.append((sol, metrics))
    return results
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| All criteria failing | Check solution content is not empty |
| Low scores | Review specific metric calculations |
| Integration errors | Verify solution object has required fields |
| Slow verification | Reduce test suite size or disable logging |

## Best Practices

1. ✅ Define clear success criteria upfront
2. ✅ Use appropriate thresholds (0.6-0.8)
3. ✅ Review all quality metrics, not just pass/fail
4. ✅ Store reports for audit trail
5. ✅ Use custom weights for domain-specific needs
6. ✅ Enable logging in production
7. ✅ Handle errors gracefully
8. ✅ Test verification criteria before production

## File Locations

```
verification_engine.py                    # Main implementation
test_verification_engine.py              # Test suite
VERIFICATION_ENGINE_README.md             # Full documentation
VERIFICATION_ENGINE_IMPLEMENTATION_SUMMARY.md  # Implementation details
```

## Support

- See VERIFICATION_ENGINE_README.md for full documentation
- Run examples: `python verification_engine.py`
- Run tests: `python -m pytest test_verification_engine.py -v`
