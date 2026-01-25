# Quality Control Module

Production-ready code quality analysis and reporting for OpenEvolve Frontend.

## Overview

The `quality_control.py` module provides comprehensive code quality checks including:

- **Code Smell Detection**: Long functions, deep nesting, magic numbers, debug statements
- **Security Scanning**: Hardcoded secrets, SQL injection, eval usage, weak crypto
- **Complexity Analysis**: Cyclomatic complexity, cognitive complexity
- **Code Duplication**: Detection of duplicate code blocks
- **Test Coverage**: Integration with pytest-cov

## Features

### 1. Multi-Language Support

Supports analysis of:
- Python (.py)
- JavaScript (.js, .jsx)
- TypeScript (.ts, .tsx)
- Java (.java)

### 2. Comprehensive Issue Classification

**Severity Levels**:
- CRITICAL: Security vulnerabilities requiring immediate attention
- HIGH: Major code quality issues
- MEDIUM: Moderate issues that should be addressed
- LOW: Minor improvements
- INFO: Informational suggestions

**Issue Types**:
- Code Smells
- Security
- Complexity
- Duplication
- Coverage
- Style

### 3. Configurable Thresholds

All checks are configurable via the `config` parameter:

```python
config = {
    'max_function_length': 50,           # Maximum function length in lines
    'max_nesting_depth': 4,               # Maximum nesting depth
    'max_cyclomatic_complexity': 10,      # Maximum cyclomatic complexity
    'max_cognitive_complexity': 15,       # Maximum cognitive complexity
    'min_coverage': 70.0,                 # Minimum test coverage percentage
    'max_file_length': 500,               # Maximum file length in lines
    'check_code_smells': True,            # Enable/disable code smell checks
    'check_security': True,               # Enable/disable security checks
    'check_complexity': True,             # Enable/disable complexity checks
    'check_duplication': True,            # Enable/disable duplication checks
    'check_coverage': True                # Enable/disable coverage checks
}
```

## Installation

No external dependencies required for basic functionality. Uses Python standard library:

```bash
# No pip install needed - uses only stdlib
python quality_control.py
```

Optional (for coverage analysis):
```bash
pip install pytest pytest-cov
```

## Usage

### Basic Usage

```python
from quality_control import run_quality_checks

# Run all checks with default configuration
result = run_quality_checks(project_root=".")

print(f"Quality Score: {result['quality_score']:.2%}")
print(f"Total Issues: {result['total_issues']}")
```

### Advanced Usage

```python
from quality_control import CodeQualityChecker

# Initialize checker with custom configuration
checker = CodeQualityChecker(
    project_root=".",
    config={
        'max_cyclomatic_complexity': 15,
        'min_coverage': 80.0,
        'check_coverage': False  # Disable for faster execution
    }
)

# Run all checks
report = checker.run_all_checks()

# Access detailed results
print(f"Quality Score: {report.metrics.quality_score:.2%}")
print(f"Files Analyzed: {report.metrics.total_files}")

# Iterate through issues
for issue in report.issues:
    if issue.severity == IssueSeverity.CRITICAL:
        print(f"[CRITICAL] {issue.file_path}:{issue.line_number}")
        print(f"  {issue.message}")
        print(f"  Suggestion: {issue.suggestion}")
```

### Security-Only Checks

```python
result = run_quality_checks(
    project_root=".",
    config={
        'check_code_smells': False,
        'check_complexity': False,
        'check_duplication': False,
        'check_coverage': False,
        'check_security': True
    }
)

print(f"Security Issues: {result['security_issues']}")
```

### Check Specific Paths

```python
result = run_quality_checks(
    project_root=".",
    paths=['src/', 'lib/'],  # Only check these directories
    config={'check_coverage': False}
)
```

## CLI Usage

Run quality checks from command line:

```bash
# Basic usage
python quality_control.py

# With custom thresholds
python quality_control.py --max-complexity 15 --min-coverage 80

# Check specific paths
python quality_control.py --path src/ --path lib/

# Output report to file
python quality_control.py --output report.json

# Only run security checks
python quality_control.py --security-only

# Verbose output
python quality_control.py --verbose
```

## Quality Report Structure

The quality report includes:

```python
{
    'quality_score': 0.85,              # Overall score (0.0 to 1.0)
    'total_issues': 42,                 # Total number of issues
    'critical_issues': 2,               # Critical severity count
    'high_issues': 8,                   # High severity count
    'medium_issues': 20,                # Medium severity count
    'low_issues': 12,                   # Low severity count
    'security_issues': 5,               # Security issue count
    'complexity_issues': 15,            # Complexity issue count
    'duplication_issues': 3,            # Duplication issue count
    'code_smell_issues': 19,            # Code smell count
    'coverage_percent': 75.5,           # Test coverage percentage
    'issues': [...],                    # Detailed issue list
    'correlation_id': 'uuid',           # Unique run identifier
    'timestamp': '2025-01-22T12:00:00Z' # UTC timestamp
}
```

## Issue Details

Each issue includes:

```python
{
    'file_path': 'src/main.py',         # File path
    'line_number': 42,                  # Line number
    'issue_type': 'security',           # Type of issue
    'severity': 'high',                 # Severity level
    'message': 'Hardcoded secret detected',  # Description
    'rule_id': 'HARDCODED_SECRET',      # Rule identifier
    'suggestion': 'Use environment variables',  # Suggested fix
    'context': '...',                   # Code context
    'correlation_id': 'uuid'            # Run identifier
}
```

## CI/CD Integration

### Example: GitHub Actions

```yaml
name: Quality Checks

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Run Quality Checks
        run: |
          python quality_control.py --output report.json
      - name: Check Quality Gate
        run: |
          SCORE=$(python -c "import json; print(json.load(open('report.json'))['quality_score'])")
          if (( $(echo "$SCORE < 0.80" | bc -l) )); then
            echo "Quality score $SCORE is below threshold 0.80"
            exit 1
          fi
```

### Example: Python Script

```python
from quality_control import run_quality_checks

def quality_gate():
    """Enforce quality standards in CI/CD pipeline."""
    result = run_quality_checks(
        project_root=".",
        config={'min_coverage': 80.0}
    )

    # Define quality gates
    MIN_SCORE = 0.80
    MAX_CRITICAL = 0

    # Check quality score
    if result['quality_score'] < MIN_SCORE:
        print(f"FAIL: Quality score {result['quality_score']:.2%} < {MIN_SCORE:.2%}")
        return False

    # Check for critical issues
    if result['critical_issues'] > MAX_CRITICAL:
        print(f"FAIL: {result['critical_issues']} critical issues found")
        return False

    print("PASS: Quality gates satisfied")
    return True

if __name__ == "__main__":
    success = quality_gate()
    exit(0 if success else 1)
```

## API Reference

### `run_quality_checks()`

Main entry point for quality checks.

**Parameters**:
- `project_root` (str): Root directory to analyze (default: ".")
- `config` (dict, optional): Configuration dictionary
- `paths` (list, optional): Specific paths to analyze

**Returns**: Dictionary with quality report data

**Raises**: `QualityCheckError` if checks fail

### `CodeQualityChecker`

Main checker class for advanced usage.

**Constructor Parameters**:
- `project_root` (str): Root directory to analyze
- `config` (dict, optional): Configuration overrides
- `correlation_id` (str, optional): Custom correlation ID

**Methods**:
- `check_code_smells(paths=None)`: Check for code smells
- `check_security_issues(paths=None)`: Check for security vulnerabilities
- `check_complexity(paths=None)`: Check code complexity
- `check_duplication(paths=None)`: Check for duplicate code
- `check_coverage(paths=None)`: Check test coverage
- `run_all_checks(paths=None)`: Run all enabled checks

## Security Checks

The module detects the following security issues:

1. **Hardcoded Secrets**: Passwords, API keys, tokens in code
2. **SQL Injection**: String concatenation in SQL queries
3. **Eval Usage**: Dynamic code execution with eval/exec
4. **Shell Injection**: Command injection vulnerabilities
5. **Unsafe Deserialization**: Pickle/vulnerable deserialization
6. **Weak Cryptography**: MD5, SHA1 usage
7. **XSS Risks**: innerHTML assignments (JavaScript)

## Complexity Metrics

### Cyclomatic Complexity

Measures the number of linearly independent paths through code:

- 1-10: Simple, low risk
- 11-20: Moderate complexity
- 21-50: High complexity, high risk
- 50+: Very high complexity

### Cognitive Complexity

Measures how difficult code is to understand:

- Accounts for nesting levels
- Penalizes breaking control flow
- Reflects human readability

## Code Quality Score

The overall quality score (0.0 to 1.0) is calculated based on:

- Issue type weights (Security = 10.0, Complexity = 2.0, etc.)
- Severity multipliers (Critical = 3.0, High = 2.0, etc.)
- Normalized to 0-100 scale

**Scoring**:
- 0.90-1.00: Excellent quality
- 0.80-0.89: Good quality
- 0.70-0.79: Acceptable quality
- 0.60-0.69: Needs improvement
- Below 0.60: Poor quality

## Testing

Run the unit tests:

```bash
# Run all tests
pytest test_quality_control.py -v

# Run specific test
pytest test_quality_control.py::TestCodeQualityChecker::test_check_security_issues -v

# Run with coverage
pytest test_quality_control.py --cov=quality_control --cov-report=html
```

## Examples

See `quality_control_examples.py` for comprehensive usage examples:

```bash
python quality_control_examples.py
```

Examples include:
1. Basic usage
2. Custom thresholds
3. Security-only checks
4. Specific path checks
5. Detailed report generation
6. Direct checker usage
7. Issue filtering
8. CI/CD integration
9. Incremental checks
10. Exception handling
11. Metrics analysis
12. Language-specific checks

## CLAUDE.md Compliance

This module adheres to the CLAUDE.md Federation Constitution:

- **Law of "Runtime Truth"**: Validates inputs, executes real checks
- **Law of Configuration Explicitness**: All thresholds configurable, validated at startup
- **Observability**: Structured logging with correlation IDs
- **Law of UTC**: All timestamps in UTC ISO-8601 format
- **Law of Idempotency**: Checks are repeatable and deterministic

## Troubleshooting

### Issue: False positives for secrets

**Solution**: Use environment variables or config files. The hardcoded secret detector avoids common patterns like `"${...}"` but may still catch legitimate values.

### Issue: Coverage checks fail

**Solution**: Install pytest-cov: `pip install pytest pytest-cov`

### Issue: Too many issues reported

**Solution**: Adjust thresholds in config or run checks incrementally on changed files only.

### Issue: Slow execution

**Solution**: Disable coverage checks (`'check_coverage': False`) or check specific paths only.

## Contributing

When adding new checks:

1. Add pattern to `SECURITY_PATTERNS` or create new method
2. Add corresponding issue type to `IssueType` enum
3. Update documentation
4. Add unit tests
5. Update `DEFAULT_CONFIG` if adding threshold

## License

Part of the OpenEvolve Frontend project.

## Support

For issues or questions:
- Check `quality_control_examples.py` for usage patterns
- Review `test_quality_control.py` for test cases
- Examine CI/CD integration in `ci_cd_pipeline.py`
