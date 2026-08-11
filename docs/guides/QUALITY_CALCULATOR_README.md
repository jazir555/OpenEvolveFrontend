# Quality Calculator Module

## Overview

The `quality_calculator.py` module provides comprehensive quality assessment capabilities for solution attempts, analyzing code across four key dimensions: correctness, completeness, efficiency, and maintainability.

## Features

- **Multi-dimensional Quality Analysis**: Evaluates solutions across 4 quality dimensions
- **AST-based Code Analysis**: Uses Python's Abstract Syntax Tree for deep code inspection
- **Code Smell Detection**: Identifies common code smells and anti-patterns
- **Requirement Validation**: Matches solutions against specified requirements
- **Comprehensive Error Handling**: Gracefully handles edge cases and invalid inputs
- **Full Type Safety**: Complete type hints throughout
- **Production-Ready**: Includes caching, logging, and optimization

## Installation

No external dependencies required beyond Python standard library.

```bash
# No installation needed - just import
from quality_calculator import QualityCalculator, calculate_quality
```

## Quick Start

### Basic Usage

```python
from quality_calculator import QualityCalculator, calculate_quality
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SolutionAttempt:
    id: str
    problem_id: str
    solution: str
    score: float
    timestamp: datetime

# Create calculator
calculator = QualityCalculator()

# Define solution and requirements
solution = SolutionAttempt(
    id="sol1",
    problem_id="prob1",
    solution='''def fibonacci(n):
    """Calculate nth fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)''',
    score=0.5,
    timestamp=datetime.now()
)

requirements = [
    "Implement fibonacci function",
    "Handle base cases",
    "Use recursion"
]

# Calculate quality
metrics = calculator.calculate_quality(solution, requirements)

print(f"Correctness: {metrics.correctness:.2%}")
print(f"Completeness: {metrics.completeness:.2%}")
print(f"Efficiency: {metrics.efficiency:.2%}")
print(f"Maintainability: {metrics.maintainability:.2%}")

# Calculate overall score
overall = calculator.calculate_overall_score(metrics)
print(f"Overall: {overall:.2%}")
```

## API Reference

### Classes

#### `SolutionQualityMetrics`

Data class containing quality metrics for a solution.

**Fields:**
- `correctness` (float): Degree to which solution meets requirements (0.0-1.0)
- `completeness` (float): Extent to which all components are present (0.0-1.0)
- `efficiency` (float): Resource usage and performance quality (0.0-1.0)
- `maintainability` (float): Code quality and readability (0.0-1.0)

**Methods:**
- `to_dict() -> Dict[str, float]`: Convert metrics to dictionary

#### `CodeQualityAnalysis`

Detailed code quality analysis results.

**Fields:**
- `complexity_score` (float): Cyclomatic complexity score
- `documentation_score` (float): Documentation coverage
- `naming_score` (float): Naming convention adherence
- `structure_score` (float): Code structure quality
- `code_smells` (List[str]): Detected code smells
- `suggestions` (List[str]): Improvement suggestions
- `metrics` (Dict[str, Any]): Detailed metrics

#### `RequirementMatch`

Result of requirement matching analysis.

**Fields:**
- `requirement` (str): Original requirement text
- `matched` (bool): Whether requirement was matched
- `confidence` (float): Match confidence (0.0-1.0)
- `evidence` (str): Evidence of match
- `line_numbers` (List[int]): Lines where match found

### Main Class: `QualityCalculator`

#### Constructor

```python
QualityCalculator(weights: Optional[Dict[str, float]] = None)
```

**Parameters:**
- `weights`: Optional custom weights for overall scoring
  - Default: `{"correctness": 0.35, "completeness": 0.25, "efficiency": 0.20, "maintainability": 0.20}`
  - Must sum to 1.0

**Raises:**
- `ValueError`: If weights don't sum to 1.0 or missing required keys

#### Methods

##### `calculate_quality`

```python
calculate_quality(
    solution: Any,
    requirements: List[str]
) -> SolutionQualityMetrics
```

Calculate comprehensive quality metrics.

**Parameters:**
- `solution`: SolutionAttempt object or string containing solution code
- `requirements`: List of requirements to validate against

**Returns:**
- `SolutionQualityMetrics` object

**Example:**
```python
metrics = calculator.calculate_quality(solution, requirements)
```

##### `calculate_correctness`

```python
calculate_correctness(
    solution: Any,
    requirements: List[str]
) -> float
```

Calculate correctness score based on requirement satisfaction.

**Returns:**
- Float between 0.0 and 1.0

##### `calculate_completeness`

```python
calculate_completeness(
    solution: Any,
    requirements: List[str]
) -> float
```

Calculate completeness score based on component presence.

**Returns:**
- Float between 0.0 and 1.0

##### `calculate_efficiency`

```python
calculate_efficiency(solution: Any) -> float
```

Calculate efficiency score based on resource usage patterns.

**Returns:**
- Float between 0.0 and 1.0

##### `calculate_maintainability`

```python
calculate_maintainability(solution: Any) -> float
```

Calculate maintainability score based on code quality.

**Returns:**
- Float between 0.0 and 1.0

##### `calculate_overall_score`

```python
calculate_overall_score(
    metrics: SolutionQualityMetrics,
    weights: Optional[Dict[str, float]] = None
) -> float
```

Calculate overall quality score from component metrics.

**Parameters:**
- `metrics`: SolutionQualityMetrics object
- `weights`: Optional custom weights (overrides instance weights)

**Returns:**
- Float between 0.0 and 1.0

##### `analyze_code_quality`

```python
analyze_code_quality(content: str) -> CodeQualityAnalysis
```

Perform comprehensive code quality analysis.

**Parameters:**
- `content`: Source code string

**Returns:**
- `CodeQualityAnalysis` object with detailed metrics

**Example:**
```python
analysis = calculator.analyze_code_quality(solution_code)
print(f"Complexity: {analysis.complexity_score}")
print(f"Documentation: {analysis.documentation_score}")
print(f"Code Smells: {analysis.code_smells}")
```

##### `detect_code_smells`

```python
detect_code_smells(content: str) -> List[str]
```

Detect code smells using pattern matching and AST analysis.

**Parameters:**
- `content`: Source code string

**Returns:**
- List of code smell descriptions

**Example:**
```python
smells = calculator.detect_code_smells(bad_code)
for smell in smells:
    print(f"Found: {smell}")
```

## Convenience Functions

Top-level functions for quick access:

### `calculate_quality`

```python
calculate_quality(
    solution: Any,
    requirements: List[str],
    weights: Optional[Dict[str, float]] = None
) -> SolutionQualityMetrics
```

Quick quality calculation using singleton calculator.

### `analyze_code_quality`

```python
analyze_code_quality(content: str) -> CodeQualityAnalysis
```

Quick code analysis using singleton calculator.

### `detect_code_smells`

```python
detect_code_smells(content: str) -> List[str]
```

Quick code smell detection using singleton calculator.

### `get_quality_calculator`

```python
get_quality_calculator(
    weights: Optional[Dict[str, float]] = None
) -> QualityCalculator
```

Get singleton calculator instance.

## Quality Dimensions

### Correctness

Measures how well the solution addresses stated requirements.

**Factors:**
- Requirement term matching
- Evidence of implementation
- Confidence scoring

**Example:**
```python
# Good solution addressing requirements
solution = '''
def calculate_sum(numbers):
    """Calculate sum of numbers."""
    return sum(numbers)
'''
# Correctness would be high for "Calculate sum of list"
```

### Completeness

Measures extent to which all necessary components are present.

**Factors:**
- Function/class definitions
- Documentation coverage
- Import organization
- Requirement coverage

**Example:**
```python
# Complete solution
solution = '''
"""
Math utilities module.
"""

from typing import List

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''
# High completeness: has docstrings, imports, type hints
```

### Efficiency

Analyzes resource usage and performance patterns.

**Factors:**
- Time complexity patterns
- Space efficiency
- Algorithmic efficiency
- Resource management (context managers, cleanup)

**Example:**
```python
# Efficient solution
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a  # O(n) - efficient

# Inefficient solution
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)  # O(2^n) - inefficient
```

### Maintainability

Analyzes code quality and readability.

**Factors:**
- Documentation coverage
- Naming conventions
- Code structure
- Code smell detection

**Example:**
```python
# Maintainable solution
def calculate_average(numbers: List[float]) -> float:
    """Calculate arithmetic mean of numbers.

    Args:
        numbers: List of numerical values

    Returns:
        The arithmetic mean
    """
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)
```

## Code Smell Detection

The module detects various code smells:

### Pattern-Based Detection

- **Long Functions**: Functions with excessive length
- **Magic Numbers**: Unnamed numeric literals
- **Deep Nesting**: Excessive indentation
- **Global Variables**: Usage of global state
- **Bare Except**: Generic exception handling
- **Print Debugging**: Using print for debugging

### AST-Based Detection

- **Too Many Parameters**: Functions with 7+ parameters
- **Empty Classes**: Classes without methods
- **Complex Functions**: High cyclomatic complexity

**Example:**
```python
# Code with smells
def BAD(x):
    y = 0
    for i in range(100):
        for j in range(100):
            if x == 42:  # Magic number
                y = y + 1
    print("DEBUG:", y)  # Print debugging
    return y

# Detected smells:
# - Magic number detected
# - Print statement (use logger)
# - Deep nesting detected
```

## Custom Weights

Customize the importance of each quality dimension:

```python
# Emphasize correctness and efficiency
custom_weights = {
    "correctness": 0.50,
    "completeness": 0.15,
    "efficiency": 0.25,
    "maintainability": 0.10
}

calculator = QualityCalculator(weights=custom_weights)
metrics = calculator.calculate_quality(solution, requirements)
overall = calculator.calculate_overall_score(metrics)
```

## Integration with Sovereign System

The module integrates seamlessly with `sovereign_data_models.py`:

```python
from sovereign_data_models import SolutionAttempt
from quality_calculator import calculate_quality

# Using Sovereign SolutionAttempt
solution = SolutionAttempt(
    solution_id="sol1",
    plan_id="plan1",
    code=solution_code,
    explanation="Solution explanation",
    metrics=sovereign_metrics,
    artifacts=[],
    created_at=datetime.now()
)

requirements = [
    "Implement efficient algorithm",
    "Handle edge cases",
    "Include documentation"
]

quality = calculate_quality(solution, requirements)
```

## Error Handling

The module gracefully handles various edge cases:

```python
# Empty solutions
metrics = calculator.calculate_quality(
    SolutionAttempt("", "", "", 0.0, datetime.now()),
    requirements
)
# Returns all zeros

# Solutions with syntax errors
metrics = calculator.calculate_quality(
    SolutionAttempt("", "", "def broken(", 0.0, datetime.now()),
    requirements
)
# Falls back to pattern-based analysis

# Invalid weights
try:
    calculator = QualityCalculator(weights={"wrong": "weights"})
except ValueError as e:
    print(f"Validation error: {e}")
```

## Performance Considerations

### AST Caching

The module caches parsed ASTs for performance:

```python
# Same content parsed only once
analysis1 = calculator.analyze_code_quality(content)
analysis2 = calculator.analyze_code_quality(content)  # Uses cache
```

### Large Files

For large files (>1000 lines), consider:

```python
# Analyze specific sections
section = code.split('\n')[0:100]
analysis = calculator.analyze_code_quality('\n'.join(section))
```

## Examples

See `demo_quality_calculator.py` for comprehensive examples:

1. Basic quality calculation
2. Comparing multiple solutions
3. Detailed code quality analysis
4. Code smell detection
5. Custom weights
6. Convenience functions
7. Edge cases

Run the demo:
```bash
python demo_quality_calculator.py
```

## Testing

Run the built-in unit tests:

```bash
python quality_calculator.py
```

The module includes 11 comprehensive unit tests covering:
- Quality metric calculations
- Code quality analysis
- Code smell detection
- Edge cases
- Weight validation
- Requirement matching

## Best Practices

1. **Define Clear Requirements**: Provide specific, actionable requirements
2. **Use Type Hints**: Improves naming and documentation scores
3. **Add Docstrings**: Improves documentation and completeness scores
4. **Follow PEP 8**: Improves naming and structure scores
5. **Handle Errors**: Improves efficiency and maintainability
6. **Use Context Managers**: Improves resource management scores
7. **Avoid Code Smells**: Directly improves maintainability

## Troubleshooting

### Low Correctness Score

**Issue**: Solution meets requirements but gets low correctness

**Solution**:
- Check that requirement terms appear in solution
- Use descriptive variable/function names
- Add comments explaining implementation

### Low Efficiency Score

**Issue**: Efficient algorithm gets low efficiency score

**Solution**:
- The analyzer uses heuristics - may not catch all optimizations
- Focus on algorithmic patterns (sets, dicts, comprehensions)
- Use context managers for resource management

### AST Parsing Errors

**Issue**: "Failed to parse AST" error

**Solution**:
- Module falls back to pattern-based analysis
- Check Python syntax is valid
- Ensure no encoding issues

## License

Part of the OpenEvolve Frontend project.

## Contributing

When contributing:

1. Add type hints to all functions
2. Update unit tests
3. Add docstrings
4. Handle edge cases
5. Log appropriately

## Changelog

### Version 1.0.0
- Initial release
- Four quality dimensions
- AST-based analysis
- Code smell detection
- Comprehensive testing
- Full type hints
- Production-ready error handling
