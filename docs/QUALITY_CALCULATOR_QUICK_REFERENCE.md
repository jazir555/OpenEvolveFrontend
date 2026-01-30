# Quality Calculator - Quick Reference

## Installation

No installation needed - just import:

```python
from quality_calculator import (
    QualityCalculator,
    calculate_quality,
    analyze_code_quality,
    detect_code_smells
)
```

## Quick Start

### 1. Calculate Solution Quality

```python
from quality_calculator import QualityCalculator

calculator = QualityCalculator()

# Calculate all quality metrics
metrics = calculator.calculate_quality(solution, requirements)

# Access individual dimensions
print(f"Correctness: {metrics.correctness:.2%}")
print(f"Completeness: {metrics.completeness:.2%}")
print(f"Efficiency: {metrics.efficiency:.2%}")
print(f"Maintainability: {metrics.maintainability:.2%}")

# Get overall score
overall = calculator.calculate_overall_score(metrics)
```

### 2. Analyze Code Quality

```python
from quality_calculator import analyze_code_quality

analysis = analyze_code_quality(code)

print(f"Complexity: {analysis.complexity_score:.2%}")
print(f"Documentation: {analysis.documentation_score:.2%}")
print(f"Naming: {analysis.naming_score:.2%}")
print(f"Structure: {analysis.structure_score:.2%}")
print(f"Code Smells: {analysis.code_smells}")
```

### 3. Detect Code Smells

```python
from quality_calculator import detect_code_smells

smells = detect_code_smells(bad_code)
for smell in smells:
    print(f"Found: {smell}")
```

## Quality Metrics

### SolutionQualityMetrics

```python
@dataclass
class SolutionQualityMetrics:
    correctness: float      # 0.0-1.0: Requirement satisfaction
    completeness: float     # 0.0-1.0: Component presence
    efficiency: float       # 0.0-1.0: Resource usage
    maintainability: float  # 0.0-1.0: Code quality
```

## Main Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `calculate_quality()` | Full quality assessment | `SolutionQualityMetrics` |
| `calculate_correctness()` | Requirement matching | `float` (0.0-1.0) |
| `calculate_completeness()` | Component presence | `float` (0.0-1.0) |
| `calculate_efficiency()` | Performance analysis | `float` (0.0-1.0) |
| `calculate_maintainability()` | Code quality | `float` (0.0-1.0) |
| `calculate_overall_score()` | Weighted combination | `float` (0.0-1.0) |
| `analyze_code_quality()` | Detailed analysis | `CodeQualityAnalysis` |
| `detect_code_smells()` | Find issues | `List[str]` |

## Custom Weights

```python
# Emphasize correctness over maintainability
custom_weights = {
    "correctness": 0.50,
    "completeness": 0.20,
    "efficiency": 0.20,
    "maintainability": 0.10
}

calculator = QualityCalculator(weights=custom_weights)
```

## Input Formats

The calculator accepts multiple solution formats:

```python
# 1. String
solution = "def foo(): pass"

# 2. Pydantic model (crewai_state_management)
from crewai_state_management import SolutionAttempt
solution = SolutionAttempt(
    sub_problem_id="prob1",
    solution_content="def foo(): pass",
    ...
)

# 3. Dataclass
@dataclass
class SolutionAttempt:
    solution: str
    ...

# 4. Dict
solution = {"solution": "def foo(): pass"}
```

## Code Smells Detected

- Long functions (>2000 chars)
- Too many parameters (>7)
- Magic numbers
- Deep nesting
- Global variables
- Bare except clauses
- Print statements (use logger)
- Empty classes

## Default Weights

```python
DEFAULT_WEIGHTS = {
    "correctness": 0.35,     # Most important
    "completeness": 0.25,    # Second
    "efficiency": 0.20,      # Third
    "maintainability": 0.20  # Fourth
}
```

## Running Tests

```bash
# Unit tests (built-in)
python quality_calculator.py

# Demo examples
python demo_quality_calculator.py

# Integration tests
python test_quality_integration.py
```

## Common Usage Patterns

### Pattern 1: Compare Solutions

```python
calculator = QualityCalculator()

metrics_a = calculator.calculate_quality(solution_a, requirements)
metrics_b = calculator.calculate_quality(solution_b, requirements)

score_a = calculator.calculate_overall_score(metrics_a)
score_b = calculator.calculate_overall_score(metrics_b)

if score_a > score_b:
    print("Solution A is better")
```

### Pattern 2: Quality Gate

```python
calculator = QualityCalculator()
metrics = calculator.calculate_quality(solution, requirements)

if metrics.correctness < 0.8:
    print("Solution does not meet requirements")
    return False

if calculator.calculate_overall_score(metrics) < 0.7:
    print("Overall quality too low")
    return False
```

### Pattern 3: Code Review

```python
analysis = analyze_code_quality(pull_request_code)

if analysis.code_smells:
    print(f"Found {len(analysis.code_smells)} issues:")
    for smell in analysis.code_smells:
        print(f"  - {smell}")

if analysis.suggestions:
    print("\nSuggestions:")
    for suggestion in analysis.suggestions:
        print(f"  - {suggestion}")
```

## Performance Tips

1. **Use Caching**: Same code analyzed only once
2. **Batch Processing**: Analyze multiple solutions together
3. **Custom Weights**: Set once, reuse for multiple calculations
4. **Convenience Functions**: Use `get_quality_calculator()` singleton

## Integration Examples

### With sovereign_data_models

```python
from sovereign_data_models import SolutionAttempt
from quality_calculator import calculate_quality

solution = SolutionAttempt(
    solution_id="sol1",
    plan_id="plan1",
    code=code,
    explanation=explanation,
    metrics=metrics,
    artifacts=[],
    created_at=datetime.now()
)

quality = calculate_quality(solution, requirements)
```

### With crewai_state_management

```python
from crewai_state_management import SolutionAttempt
from quality_calculator import QualityCalculator

calculator = QualityCalculator()

solution = SolutionAttempt(
    sub_problem_id="prob1",
    solution_content=code,
    confidence_score=0.9,
    execution_method=ExecutionMethod.TRADITIONAL
)

metrics = calculator.calculate_quality(solution, requirements)
```

## Key Points

- All metrics normalized to 0.0-1.0
- AST-based analysis (not just regex)
- Graceful error handling
- No external dependencies
- Full type hints
- Production-ready

## Need More?

- **Full Documentation**: See `QUALITY_CALCULATOR_README.md`
- **Implementation Details**: See `QUALITY_CALCULATOR_IMPLEMENTATION_SUMMARY.md`
- **Examples**: Run `python demo_quality_calculator.py`
