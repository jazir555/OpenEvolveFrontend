# Verification Engine

Production-ready solution verification and quality assessment system for the OpenEvolve Frontend.

## Overview

The `VerificationEngine` provides comprehensive verification capabilities for solution attempts, including:

- **Success Criteria Validation**: Define and validate against custom success criteria
- **Quality Metrics Calculation**: Multi-dimensional quality assessment (8 dimensions)
- **Verification Reports**: Detailed, production-ready verification reports
- **Test Suite Execution**: Run complete verification suites
- **Integration Support**: Works with `sovereign_data_models` and `crewai_state_management`

## Installation

The verification engine is part of the OpenEvolve Frontend codebase. No additional dependencies required beyond Python 3.8+.

```bash
# Already included in:
# C:\Users\mmeadow\Documents\OpenEvolve\Frontend\verification_engine.py
```

## Quick Start

### Basic Usage

```python
from verification_engine import VerificationEngine

# Initialize engine
engine = VerificationEngine()

# Create a solution
solution = {
    'id': 'solution_001',
    'solution_content': '''
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''
}

# Define success criteria
criteria = engine.create_success_criteria([
    "Solution must be at least 80% complete",
    "Code must be clear and readable"
])

# Verify solution
report = engine.verify_solution(solution, criteria)

# Check results
print(f"Approved: {report.is_approved}")
print(f"Score: {report.verification_score:.2f}")
print(f"Summary: {report.summary}")
```

## Core Components

### 1. VerificationEngine

The main engine for solution verification.

```python
engine = VerificationEngine(config={
    'strict_mode': False,
    'min_quality_threshold': 0.6,
    'enable_detailed_logging': True
})
```

**Configuration Options:**
- `strict_mode`: Enable strict validation (default: False)
- `min_quality_threshold`: Minimum quality score for approval (0.0-1.0)
- `enable_detailed_logging`: Enable verbose logging (default: True)

### 2. SuccessCriterion

Defines a single success criterion for validation.

```python
from verification_engine import SuccessCriterion

criterion = SuccessCriterion(
    id="completeness_check",
    description="Solution must be complete",
    metric="completeness",
    threshold=0.8,
    weight=1.0,
    category="functional"
)
```

**Available Metrics:**
- `completeness`: How completely requirements are addressed
- `correctness`: Accuracy and correctness of solution
- `efficiency`: Performance and resource utilization
- `clarity`: Code readability and documentation
- `maintainability`: Ease of maintenance
- `scalability`: Ability to scale
- `security`: Security considerations
- `test_coverage`: Test coverage percentage

### 3. SolutionQualityMetrics

Comprehensive quality metrics for solutions.

```python
from verification_engine import SolutionQualityMetrics

metrics = SolutionQualityMetrics(
    completeness=0.8,
    correctness=0.9,
    efficiency=0.7,
    clarity=0.85,
    maintainability=0.75,
    scalability=0.6,
    security=0.8,
    test_coverage=0.7
)

# Calculate overall score
metrics.calculate_overall()

print(f"Overall: {metrics.overall_score:.2f}")
```

### 4. VerificationReport

Detailed verification report for a solution.

```python
from verification_engine import VerificationReport

report = VerificationReport(
    solution_attempt_id="solution_001",
    gauntlet_name="test_gauntlet",
    is_approved=True,
    reports_by_judge=[],
    summary="Solution passed all criteria",
    quality_metrics=metrics,
    verification_score=0.85
)

# Export to dict/JSON
report_dict = report.to_dict()
report_json = report.to_json()
```

## API Reference

### VerificationEngine Methods

#### `verify_solution(solution, criteria)`

Verify a solution against success criteria.

**Parameters:**
- `solution`: SolutionAttempt object or dict with solution content
- `criteria`: List of SuccessCriterion objects

**Returns:** `VerificationReport`

**Example:**
```python
report = engine.verify_solution(solution, criteria)
```

#### `create_success_criteria(requirements)`

Create success criteria from requirement strings.

**Parameters:**
- `requirements`: List of requirement descriptions

**Returns:** `List[SuccessCriterion]`

**Example:**
```python
criteria = engine.create_success_criteria([
    "Solution must be at least 90% complete",
    "Code must pass all security checks",
    "Solution must be efficient"
])
```

#### `check_criterion(solution, criterion)`

Check if a solution meets a specific criterion.

**Parameters:**
- `solution`: SolutionAttempt to check
- `criterion`: SuccessCriterion to validate

**Returns:** `bool` (True if passed)

**Example:**
```python
criterion = SuccessCriterion(
    id="test",
    description="Test",
    metric="completeness",
    threshold=0.8
)
passed = engine.check_criterion(solution, criterion)
```

#### `calculate_quality_scores(solution)`

Calculate quality metrics for a solution.

**Parameters:**
- `solution`: SolutionAttempt to analyze

**Returns:** `SolutionQualityMetrics`

**Example:**
```python
metrics = engine.calculate_quality_scores(solution)
print(f"Overall quality: {metrics.overall_score:.2f}")
```

#### `run_verification_suite(solution, test_suite)`

Run a complete verification suite.

**Parameters:**
- `solution`: SolutionAttempt to verify
- `test_suite`: List of test cases or criteria

**Returns:** `VerificationReport`

**Example:**
```python
test_suite = [
    {'metric': 'completeness', 'threshold': 0.7},
    {'metric': 'correctness', 'threshold': 0.6}
]
report = engine.run_verification_suite(solution, test_suite)
```

## Integration Examples

### With sovereign_data_models

```python
from sovereign_data_models import SolutionAttempt
from verification_engine import VerificationEngine

# Create solution using sovereign_data_models
solution = SolutionAttempt(
    id="solution_001",
    problem_id="test_problem",
    solution="def solve(): return True",
    score=0.8,
    timestamp=datetime.now()
)

# Verify
engine = VerificationEngine()
criteria = engine.create_success_criteria(["Solution must be correct"])
report = engine.verify_solution(solution, criteria)
```

### With crewai_state_management

```python
from crewai_state_management import SolutionAttempt
from verification_engine import VerificationEngine

# Create solution using crewai_state_management
solution = SolutionAttempt(
    sub_problem_id="sp_001",
    solution_content="def solve(): return True",
    confidence_score=0.75,
    execution_method="traditional"
)

# Verify
engine = VerificationEngine()
report = engine.verify_solution(solution, engine.create_success_criteria([
    "Complete solution"
]))
```

### With sgd_workflow_orchestrator

```python
from sgd_workflow_orchestrator import SGDWorkflowOrchestrator
from verification_engine import VerificationEngine

orchestrator = SGDWorkflowOrchestrator()
engine = VerificationEngine()

# In workflow execution
def verify_subproblem_solution(solution_attempt):
    criteria = engine.create_success_criteria([
        "Solution must be complete",
        "Solution must be correct"
    ])
    report = engine.verify_solution(solution_attempt, criteria)
    return report
```

## Advanced Usage

### Custom Quality Weights

```python
custom_weights = {
    'completeness': 0.30,
    'correctness': 0.40,
    'efficiency': 0.10,
    'clarity': 0.10,
    'maintainability': 0.05,
    'scalability': 0.05
}

metrics = engine.calculate_quality_scores(solution)
metrics.calculate_overall(custom_weights)
```

### Verification History

```python
# Get verification history
history = engine.get_verification_history()

# Clear history
engine.clear_history()
```

### Report Comparison

```python
from verification_engine import compare_reports

comparison = compare_reports(report1, report2)
print(f"Score difference: {comparison['score_difference']:.2f}")
print(f"Approval changed: {comparison['approval_changed']}")
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest test_verification_engine.py -v

# Run specific test class
python -m pytest test_verification_engine.py::TestVerificationEngine -v

# Run with coverage
python -m pytest test_verification_engine.py --cov=verification_engine --cov-report=html
```

### Test Coverage

The test suite includes:
- **37 unit tests** covering all major functionality
- **Success criterion validation**
- **Quality metrics calculation**
- **Verification report generation**
- **Edge case handling**
- **Integration tests** with related modules
- **Performance tests**

## Examples

Run the built-in examples:

```bash
python verification_engine.py
```

This will demonstrate:
1. Basic verification workflow
2. Verification suite execution
3. Custom criteria with high thresholds

## Error Handling

The verification engine handles errors gracefully:

```python
# Empty solution content -> Returns failed report (doesn't crash)
empty_solution = {'solution_content': ''}
report = engine.verify_solution(empty_solution, criteria)
# report.is_approved will be False

# Invalid criteria -> Raises ValueError
try:
    engine.verify_solution(solution, [])
except ValueError as e:
    print(f"Error: {e}")

# None solution -> Raises ValueError
try:
    engine.verify_solution(None, criteria)
except ValueError as e:
    print(f"Error: {e}")
```

## Performance Considerations

- **Typical verification time**: 10-100ms per solution
- **Memory usage**: Minimal (no caching by default)
- **Scalability**: Tested with 50+ test suites
- **Thread safety**: Not thread-safe (create separate instances per thread)

## Best Practices

1. **Always define clear success criteria** before verification
2. **Use appropriate thresholds** (0.6-0.8 for most cases)
3. **Review quality metrics** in addition to pass/fail status
4. **Store verification reports** for audit trail
5. **Use custom weights** for domain-specific quality assessment
6. **Enable logging** in production for debugging

## Troubleshooting

### Issue: All criteria failing

**Solution**: Check that solution content is not empty and properly formatted.

```python
content = engine._extract_solution_content(solution)
print(f"Content length: {len(content)}")
```

### Issue: Low quality scores

**Solution**: Review the specific metric calculations and adjust thresholds.

```python
metrics = engine.calculate_quality_scores(solution)
print(metrics.to_dict())
```

### Issue: Integration errors

**Solution**: Ensure solution object has required fields.

```python
# For sovereign_data_models
required_fields = ['id', 'solution', 'timestamp']

# For crewai_state_management
required_fields = ['sub_problem_id', 'solution_content', 'execution_method']
```

## Contributing

To extend the verification engine:

1. Add new metric calculators in `_calculate_*` methods
2. Extend SuccessCriterion with new categories
3. Add custom verification logic in `verify_solution`
4. Update tests for new functionality

## License

Part of the OpenEvolve Frontend project. See main LICENSE file for details.

## Support

For issues or questions:
- Check the test suite for usage examples
- Review the inline documentation
- Examine the example functions in `verification_engine.py`

## Version History

- **1.0.0** (2026-01-22): Initial production-ready implementation
  - Full verification engine implementation
  - 8-dimensional quality metrics
  - Comprehensive test suite (37 tests)
  - Integration with sovereign_data_models and crewai_state_management
  - Production-ready error handling and logging
