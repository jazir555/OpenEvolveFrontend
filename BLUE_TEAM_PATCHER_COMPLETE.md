# Blue Team Patcher Engine - Complete Implementation Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Advanced Usage](#advanced-usage)
7. [API Reference](#api-reference)
8. [Patch Types](#patch-types)
9. [Testing](#testing)
10. [Examples](#examples)
11. [Integration](#integration)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The **Blue Team Patcher Engine** is a comprehensive automated patching system for OpenEvolve that receives issues from red team validation, categorizes and applies fixes, and validates results. It serves as the defensive counterpart to the red team's offensive testing.

### Key Features

- **15 Patch Types**: Support for security, performance, logic, clarity, structure, documentation, error handling, validation, refactoring, compliance, maintainability, resource management, concurrency, dependencies, and testing patches
- **Intelligent Analysis**: Automatic categorization, complexity estimation, and strategy recommendation
- **Multiple Strategies**: Automatic, semi-automatic, manual, and hybrid patching workflows
- **LLM-Powered**: Uses GPT-4 for intelligent patch generation
- **Validation Built-in**: Checks for regressions, quality improvements, and syntax errors
- **Rollback Support**: Automatic rollback data creation for all patches
- **Comprehensive Reporting**: JSON and Markdown reports with metrics and recommendations

### Workflow

```
Red Team Findings → PatchAnalyzer → PatchApplicationEngine → PatchValidator → PatchReport
                         ↓                ↓                    ↓
                   Categorization    Patch Generation    Validation
                   Prioritization    Application         Quality Check
                   Strategy Rec.     Rollback Data       Regression Check
```

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    BlueTeamPatcherEngine                        │
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │PatchAnalyzer │  │PatchApplication  │  │ PatchValidator  │  │
│  │              │  │Engine            │  │                 │  │
│  │- Categorize  │→ │- Apply Patches   │→ │- Validate       │  │
│  │- Prioritize  │  │- Generate Code   │  │- Check Regress. │  │
│  │- Estimate    │  │- Create Rollback │  │- Quality Check  │  │
│  │  Complexity  │  │- Track Status    │  │- Syntax Check   │  │
│  └──────────────┘  └──────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         ↓                    ↓                     ↓
    PatchAnalysis       PatchResults         ValidationSummary
```

### Data Flow

1. **Input**: Red team findings (`IssueFinding` objects)
2. **Analysis**: `PatchAnalyzer` categorizes and prioritizes issues
3. **Application**: `PatchApplicationEngine` generates and applies patches
4. **Validation**: `PatchValidator` verifies fixes and checks for regressions
5. **Output**: `PatchReport` with comprehensive results and recommendations

---

## Components

### 1. PatchAnalyzer

Analyzes red team findings and recommends optimal patching strategies.

**Key Responsibilities:**
- Categorize issues by type and severity
- Estimate patch complexity (0-1 scale)
- Calculate priority scores (1-10)
- Recommend patching strategy (automatic/manual/semi-automatic/hybrid)
- Generate patch requests with metadata

**Methods:**
- `analyze_findings()`: Main analysis method
- `_categorize_issues()`: Group by category
- `_group_by_severity()`: Group by severity level
- `_estimate_complexity()`: Calculate complexity score
- `_recommend_strategy()`: Suggest best approach

### 2. PatchApplicationEngine

Applies patches using 15+ specialized patch types with LLM-based generation.

**Key Responsibilities:**
- Apply patches using different strategies
- Generate code fixes using LLM
- Create rollback data for all patches
- Track patch status and history
- Support parallel patch application

**Methods:**
- `apply_patches()`: Main application method
- `rollback_patch()`: Rollback a specific patch
- `_generate_and_apply_patch()`: LLM-based generation
- `_generate_diff()`: Create unified diff

**Supported Patch Types:**
1. `SECURITY_PATCH` - Fix security vulnerabilities
2. `PERFORMANCE_OPTIMIZATION` - Improve performance
3. `LOGIC_CORRECTION` - Fix logical errors
4. `CLARITY_IMPROVEMENT` - Enhance code clarity
5. `STRUCTURE_REORGANIZATION` - Reorganize code structure
6. `DOCUMENTATION_ADDITION` - Add documentation
7. `ERROR_HANDLING` - Improve error handling
8. `INPUT_VALIDATION` - Add input validation
9. `CODE_REFACTORING` - Refactor code
10. `COMPLIANCE_FIX` - Fix compliance issues
11. `MAINTAINABILITY_IMPROVEMENT` - Improve maintainability
12. `RESOURCE_MANAGEMENT` - Fix resource leaks
13. `CONCURRENCY_FIX` - Fix concurrency issues
14. `DEPENDENCY_UPDATE` - Update dependencies
15. `TESTING_ENHANCEMENT` - Add/improve tests

### 3. PatchValidator

Validates that patches fix issues and checks for regressions.

**Key Responsibilities:**
- Validate patch success
- Check for regressions
- Verify quality improvements
- Test syntax correctness
- Generate validation scores

**Methods:**
- `validate_patches()`: Main validation method
- `_test_content_changed()`: Verify content modification
- `_test_regressions()`: Check for regressions
- `_test_quality_improvement()`: Compare quality scores
- `_test_syntax()`: Validate syntax

### 4. BlueTeamPatcherEngine

Main orchestrator integrating all components.

**Key Responsibilities:**
- Coordinate patcher workflow
- Generate comprehensive reports
- Export results (JSON/Markdown)
- Manage patch history
- Provide recommendations

**Methods:**
- `run_patcher_workflow()`: Execute complete workflow
- `export_report()`: Export in different formats
- `_generate_report()`: Create comprehensive report

---

## Installation

### Requirements

```
python>=3.8
openai>=1.0.0
pytest>=7.0.0
pytest-cov>=4.0.0
```

### Install Dependencies

```bash
pip install openai pytest pytest-cov
```

### Import

```python
from blue_team_patcher_engine import (
    BlueTeamPatcherEngine,
    PatchAnalyzer,
    PatchApplicationEngine,
    PatchValidator,
    quick_patch,
    PatchType,
    PatchStrategy
)
```

---

## Quick Start

### Basic Usage

```python
from blue_team_patcher_engine import BlueTeamPatcherEngine, PatchStrategy
from red_team import IssueFinding, IssueCategory, SeverityLevel

# 1. Create issue findings from red team
findings = [
    IssueFinding(
        finding_id="F001",
        title="SQL Injection",
        description="User input not sanitized",
        category=IssueCategory.SECURITY_VULNERABILITY,
        severity=SeverityLevel.CRITICAL,
        location="database.py:45",
        confidence=0.95,
        suggested_fix="Use parameterized queries"
    )
]

# 2. Create patcher engine
engine = BlueTeamPatcherEngine(
    api_key="your-api-key",
    model_name="gpt-4o"
)

# 3. Run patcher workflow
report = engine.run_patcher_workflow(
    red_team_findings=findings,
    original_content=code_content,
    content_type="code",
    strategy=PatchStrategy.AUTOMATIC
)

# 4. Get results
print(f"Success rate: {report.summary['success_rate']:.1%}")
print(f"Patches applied: {report.summary['successful_patches']}")

# 5. Export report
json_report = engine.export_report(report, format='json')
md_report = engine.export_report(report, format='markdown')
```

### Quick Patch Function

For simple use cases:

```python
from blue_team_patcher_engine import quick_patch

patched_content, report = quick_patch(
    findings=findings,
    content=original_code,
    api_key="your-api-key",
    content_type="code"
)

print(f"Fixed content:\n{patched_content}")
```

---

## Advanced Usage

### Custom Patch Strategy

```python
from blue_team_patcher_engine import BlueTeamPatcherEngine, PatchStrategy

engine = BlueTeamPatcherEngine(api_key="your-api-key")

# Manual strategy for critical issues
report = engine.run_patcher_workflow(
    red_team_findings=critical_findings,
    original_content=code,
    content_type="code",
    strategy=PatchStrategy.MANUAL  # Generates manual instructions
)

# Hybrid strategy for mixed complexity
report = engine.run_patcher_workflow(
    red_team_findings=findings,
    original_content=code,
    content_type="code",
    strategy=PatchStrategy.HYBRID  # Auto for simple, manual for complex
)
```

### Parallel Patch Application

```python
# Apply up to 5 patches in parallel
report = engine.run_patcher_workflow(
    red_team_findings=findings,
    original_content=code,
    content_type="code",
    strategy=PatchStrategy.AUTOMATIC,
    max_parallel=5
)
```

### Using Individual Components

```python
from blue_team_patcher_engine import PatchAnalyzer, PatchApplicationEngine

# 1. Analyze findings
analyzer = PatchAnalyzer()
analysis = analyzer.analyze_findings(
    findings=red_team_findings,
    original_content=code,
    content_type="code"
)

print(f"Complexity distribution: {analysis.complexity_distribution}")
print(f"Recommended strategy: {analysis.strategy_recommendation}")

# 2. Apply patches
applicator = PatchApplicationEngine(api_key="your-api-key")
results = applicator.apply_patches(
    patch_requests=analysis.recommended_patches,
    strategy=analysis.strategy_recommendation
)

# 3. Rollback if needed
for result in results:
    if not result.success:
        applicator.rollback_patch(result.patch_id)
```

### Custom Validation

```python
from blue_team_patcher_engine import PatchValidator

validator = PatchValidator(
    api_key="your-api-key",
    model_name="gpt-4o"
)

validation = validator.validate_patches(
    patch_results=results,
    original_issues=findings,
    original_content=code
)

print(f"Validation score: {validation['overall_validation_score']:.2%}")
print(f"Regressions: {validation['regressions_detected']}")
```

---

## API Reference

### BlueTeamPatcherEngine

#### Constructor

```python
BlueTeamPatcherEngine(
    api_key: Optional[str] = None,
    api_base: str = "https://api.openai.com/v1",
    model_name: str = "gpt-4o",
    quality_assessment: Optional[QualityAssessmentEngine] = None
)
```

**Parameters:**
- `api_key`: OpenAI API key (required for automatic patching)
- `api_base`: Base URL for API
- `model_name`: Model to use for patch generation
- `quality_assessment`: Optional quality assessment engine

#### Methods

##### run_patcher_workflow()

```python
run_patcher_workflow(
    red_team_findings: List[IssueFinding],
    original_content: str,
    content_type: str = "general",
    strategy: PatchStrategy = PatchStrategy.AUTOMATIC,
    max_parallel: int = 3
) -> PatchReport
```

Execute the complete patcher workflow.

**Returns:** `PatchReport` with comprehensive results

##### export_report()

```python
export_report(report: PatchReport, format: str = "json") -> str
```

Export a report in JSON or Markdown format.

**Parameters:**
- `report`: The patch report to export
- `format`: Either "json" or "markdown"

**Returns:** Formatted report string

### PatchAnalyzer

#### Constructor

```python
PatchAnalyzer(quality_assessment: Optional[QualityAssessmentEngine] = None)
```

#### Methods

##### analyze_findings()

```python
analyze_findings(
    findings: List[IssueFinding],
    original_content: str,
    content_type: str = "general"
) -> PatchAnalysis
```

Analyze findings and generate patch recommendations.

**Returns:** `PatchAnalysis` with categorization and recommendations

### PatchApplicationEngine

#### Constructor

```python
PatchApplicationEngine(
    api_key: Optional[str] = None,
    api_base: str = "https://api.openai.com/v1",
    model_name: str = "gpt-4o",
    quality_assessment: Optional[QualityAssessmentEngine] = None
)
```

#### Methods

##### apply_patches()

```python
apply_patches(
    patch_requests: List[PatchRequest],
    strategy: PatchStrategy = PatchStrategy.AUTOMATIC,
    max_parallel: int = 3,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> List[PatchResult]
```

Apply multiple patches with specified strategy.

**Returns:** List of `PatchResult` objects

##### rollback_patch()

```python
rollback_patch(patch_id: str) -> bool
```

Rollback a specific patch.

**Returns:** `True` if successful, `False` otherwise

### PatchValidator

#### Constructor

```python
PatchValidator(
    quality_assessment: Optional[QualityAssessmentEngine] = None,
    api_key: Optional[str] = None,
    api_base: str = "https://api.openai.com/v1",
    model_name: str = "gpt-4o"
)
```

#### Methods

##### validate_patches()

```python
validate_patches(
    patch_results: List[PatchResult],
    original_issues: List[IssueFinding],
    original_content: str
) -> Dict[str, Any]
```

Validate all applied patches.

**Returns:** Validation summary dictionary

---

## Patch Types

### 1. Security Patch

**Use Case:** Fix security vulnerabilities
```python
PatchType.SECURITY_PATCH
```

**Examples:**
- SQL injection
- XSS vulnerabilities
- Authentication issues
- Authorization flaws

**Generated Fix:**
```python
# Before
query = f"SELECT * FROM users WHERE id = {user_id}"

# After
query = "SELECT * FROM users WHERE id = ?"
cursor.execute(query, (user_id,))
```

### 2. Performance Optimization

**Use Case:** Improve performance
```python
PatchType.PERFORMANCE_OPTIMIZATION
```

**Examples:**
- Inefficient algorithms
- Unnecessary loops
- Missing caching
- Database query optimization

**Generated Fix:**
```python
# Before
results = []
for item in items:
    results.append(process(item))

# After
results = [process(item) for item in items]
```

### 3. Logic Correction

**Use Case:** Fix logical errors
```python
PatchType.LOGIC_CORRECTION
```

**Examples:**
- Wrong operators
- Incorrect conditions
- Logic flow errors

**Generated Fix:**
```python
# Before
if x = y:  # Assignment instead of comparison
    return True

# After
if x == y:  # Proper comparison
    return True
```

### 4. Clarity Improvement

**Use Case:** Enhance code readability
```python
PatchType.CLARITY_IMPROVEMENT
```

**Examples:**
- Poor variable names
- Confusing logic
- Missing comments
- Complex expressions

**Generated Fix:**
```python
# Before
def f(x):
    return x*2+1 if x>0 else x-1

# After
def calculate_value(x: int) -> int:
    """Calculate adjusted value based on input."""
    if x > 0:
        return x * 2 + 1
    else:
        return x - 1
```

### 5. Structure Reorganization

**Use Case:** Reorganize code structure
```python
PatchType.STRUCTURE_REORGANIZATION
```

**Examples:**
- Poor file organization
- Missing separation of concerns
- Circular dependencies

### 6. Documentation Addition

**Use Case:** Add missing documentation
```python
PatchType.DOCUMENTATION_ADDITION
```

**Examples:**
- Missing docstrings
- Undocumented parameters
- Missing README
- No API documentation

**Generated Fix:**
```python
def process_data(data: List[Dict]) -> Dict[str, Any]:
    """
    Process raw data and return aggregated results.

    Args:
        data: List of dictionaries containing raw data

    Returns:
        Dictionary with aggregated results including:
        - 'total': Total count of items
        - 'average': Average value

    Raises:
        ValueError: If data is empty or invalid

    Example:
        >>> process_data([{'value': 10}, {'value': 20}])
        {'total': 2, 'average': 15.0}
    """
    # Implementation...
```

### 7. Error Handling

**Use Case:** Improve error handling
```python
PatchType.ERROR_HANDLING
```

**Examples:**
- Missing try-catch blocks
- Unhandled exceptions
- Poor error messages

**Generated Fix:**
```python
# Before
def divide(a, b):
    return a / b

# After
def divide(a: float, b: float) -> float:
    """Divide two numbers with proper error handling."""
    try:
        return a / b
    except ZeroDivisionError:
        raise ValueError(f"Cannot divide {a} by zero")
    except TypeError as e:
        raise TypeError(f"Both arguments must be numbers: {e}")
```

### 8. Input Validation

**Use Case:** Add input validation
```python
PatchType.INPUT_VALIDATION
```

**Examples:**
- Missing parameter checks
- No type validation
- No range checks

**Generated Fix:**
```python
def calculate_age(birth_year: int) -> int:
    """Calculate age from birth year."""
    current_year = 2024

    # Input validation
    if not isinstance(birth_year, int):
        raise TypeError("birth_year must be an integer")
    if birth_year < 1900 or birth_year > current_year:
        raise ValueError(f"birth_year must be between 1900 and {current_year}")

    return current_year - birth_year
```

### 9. Code Refactoring

**Use Case:** Refactor code quality
```python
PatchType.CODE_REFACTORING
```

**Examples:**
- Code duplication
- Long functions
- Deep nesting
- Magic numbers

### 10. Compliance Fix

**Use Case:** Fix compliance issues
```python
PatchType.COMPLIANCE_FIX
```

**Examples:**
- GDPR violations
- Security standards
- Industry regulations

### 11. Maintainability Improvement

**Use Case:** Improve maintainability
```python
PatchType.MAINTAINABILITY_IMPROVEMENT
```

**Examples:**
- Hard-coded values
- Tight coupling
- Poor modularity

### 12. Resource Management

**Use Case:** Fix resource issues
```python
PatchType.RESOURCE_MANAGEMENT
```

**Examples:**
- Memory leaks
- Unclosed files
- Connection leaks

**Generated Fix:**
```python
# Before
def read_file(filename):
    f = open(filename)
    return f.read()

# After
def read_file(filename: str) -> str:
    """Read file contents with proper resource management."""
    with open(filename, 'r') as f:
        return f.read()
```

### 13. Concurrency Fix

**Use Case:** Fix concurrency issues
```python
PatchType.CONCURRENCY_FIX
```

**Examples:**
- Race conditions
- Missing locks
- Deadlock risks

**Generated Fix:**
```python
from threading import Lock

class Counter:
    def __init__(self):
        self.value = 0
        self.lock = Lock()

    def increment(self):
        """Thread-safe increment."""
        with self.lock:
            self.value += 1
```

### 14. Dependency Update

**Use Case:** Update dependencies
```python
PatchType.DEPENDENCY_UPDATE
```

**Examples:**
- Outdated packages
- Vulnerable dependencies
- Deprecated APIs

### 15. Testing Enhancement

**Use Case:** Add/improve tests
```python
PatchType.TESTING_ENHANCEMENT
```

**Examples:**
- Missing unit tests
- Low code coverage
- No integration tests

**Generated Fix:**
```python
import unittest

class TestDataProcessor(unittest.TestCase):
    def test_process_valid_data(self):
        """Test processing of valid data."""
        result = process_data([{'value': 10}])
        self.assertEqual(result['total'], 1)

    def test_process_empty_data(self):
        """Test that empty data raises appropriate error."""
        with self.assertRaises(ValueError):
            process_data([])

    def test_process_invalid_type(self):
        """Test that invalid data type raises error."""
        with self.assertRaises(TypeError):
            process_data("not a list")
```

---

## Testing

### Run All Tests

```bash
# Run with coverage
pytest test_blue_team_patcher.py -v --cov=blue_team_patcher_engine --cov-report=html

# Run specific test class
pytest test_blue_team_patcher.py::TestPatchAnalyzer -v

# Run specific test
pytest test_blue_team_patcher.py::TestPatchAnalyzer::test_analyze_findings_basic -v
```

### Test Structure

```
test_blue_team_patcher.py
├── TestPatchAnalyzer (7 tests)
│   ├── test_analyze_findings_basic
│   ├── test_categorize_issues
│   ├── test_group_by_severity
│   ├── test_patch_request_generation
│   ├── test_complexity_estimation
│   ├── test_strategy_recommendation
│   └── test_analysis_confidence_calculation
│
├── TestPatchApplicationEngine (10 tests)
│   ├── test_initialization
│   ├── test_patch_handlers_coverage
│   ├── test_generate_patch_id
│   ├── test_create_rollback_data
│   ├── test_extract_patched_content_from_code_block
│   ├── test_extract_patched_content_from_json
│   ├── test_generate_diff
│   ├── test_create_failed_result
│   ├── test_rollback_patch
│   └── test_validate_intermediate_patch
│
├── TestPatchValidator (8 tests)
│   ├── test_initialization
│   ├── test_validate_patches_success
│   ├── test_validate_patches_failure
│   ├── test_content_changed_detection
│   ├── test_regression_detection
│   ├── test_quality_improvement_check
│   ├── test_syntax_check_code
│   └── test_syntax_check_non_code
│
├── TestBlueTeamPatcherEngine (7 tests)
│   ├── test_engine_initialization
│   ├── test_full_workflow_without_api
│   ├── test_manual_strategy_workflow
│   ├── test_report_generation_json
│   ├── test_report_generation_markdown
│   ├── test_recommendations_generation
│   └── test_rollback_log_creation
│
├── TestUtilityFunctions (3 tests)
│   ├── test_quick_patch_function
│   ├── test_patch_type_enum_completeness
│   └── test_patch_strategy_enum_completeness
│
├── TestEdgeCases (5 tests)
│   ├── test_empty_findings_list
│   ├── test_very_long_content
│   ├── test_special_characters_in_content
│   ├── test_unicode_in_findings
│   └── test_mixed_severity_findings
│
└── TestPerformance (2 tests)
    ├── test_analysis_performance
    └── test_patch_generation_caching
```

### Expected Results

```
============================= test session starts =============================
collected 42 items

test_blue_team_patcher.py::TestPatchAnalyzer::test_analyze_findings_basic PASSED
test_blue_team_patcher.py::TestPatchAnalyzer::test_categorize_issues PASSED
...
============================ 42 passed in 5.23s =============================

----------- coverage: platform win32, python 3.9 ----------
Name                                Stmts   Miss  Cover   Missing
-------------------------------------------------------------------------
blue_team_patcher_engine.py          856     45    95%   127-145, 289-301
-------------------------------------------------------------------------
TOTAL                                  856     45    95%
```

---

## Examples

### Example 1: Fix Security Vulnerability

```python
from blue_team_patcher_engine import quick_patch
from red_team import IssueFinding, IssueCategory, SeverityLevel

# Create security issue finding
security_issue = IssueFinding(
    finding_id="SEC001",
    title="SQL Injection Vulnerability",
    description="User input concatenated directly into SQL query",
    category=IssueCategory.SECURITY_VULNERABILITY,
    severity=SeverityLevel.CRITICAL,
    location="user_auth.py:45",
    confidence=0.98,
    suggested_fix="Use parameterized queries"
)

# Original vulnerable code
vulnerable_code = '''
def authenticate_user(username, password):
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    return execute_query(query)
'''

# Apply patch
fixed_code, report = quick_patch(
    findings=[security_issue],
    content=vulnerable_code,
    api_key="your-api-key",
    content_type="code"
)

print("Fixed Code:")
print(fixed_code)
print(f"\nSuccess Rate: {report.summary['success_rate']:.1%}")
```

**Output:**
```python
def authenticate_user(username, password):
    """Authenticate user with parameterized query."""
    query = "SELECT * FROM users WHERE username = ? AND password = ?"
    return execute_query(query, (username, password))

Success Rate: 100.0%
```

### Example 2: Performance Optimization

```python
from blue_team_patcher_engine import BlueTeamPatcherEngine, PatchStrategy

# Create performance issue
performance_issue = IssueFinding(
    finding_id="PERF001",
    title="Inefficient String Concatenation",
    description="String concatenation in loop causes O(n²) complexity",
    category=IssueCategory.PERFORMANCE_PROBLEM,
    severity=SeverityLevel.HIGH,
    location="text_processor.py:78",
    confidence=0.92,
    suggested_fix="Use join() or list comprehension"
)

# Slow code
slow_code = '''
def build_text(items):
    result = ""
    for item in items:
        result += " " + item.upper()
    return result
'''

# Fix with automatic strategy
engine = BlueTeamPatcherEngine(api_key="your-api-key")
report = engine.run_patcher_workflow(
    red_team_findings=[performance_issue],
    original_content=slow_code,
    content_type="code",
    strategy=PatchStrategy.AUTOMATIC
)

# Get optimized code
optimized_code = report.patch_results[0].patched_content
print(optimized_code)
```

**Output:**
```python
def build_text(items):
    """Build text from items efficiently using join."""
    return " ".join(item.upper() for item in items)
```

### Example 3: Manual Patch Instructions

```python
# Complex issue requiring manual intervention
complex_issue = IssueFinding(
    finding_id="COMP001",
    title="Architecture Refactoring Needed",
    description="Monolithic structure needs microservices refactoring",
    category=IssueCategory.STRUCTURAL_FLAW,
    severity=SeverityLevel.HIGH,
    location="app.py",
    confidence=0.85,
    suggested_fix="Refactor into microservices"
)

# Use manual strategy
engine = BlueTeamPatcherEngine(api_key="your-api-key")
report = engine.run_patcher_workflow(
    red_team_findings=[complex_issue],
    original_content=monolithic_code,
    content_type="code",
    strategy=PatchStrategy.MANUAL
)

# Get manual instructions
instructions = report.patch_results[0].diff
print("MANUAL PATCH INSTRUCTIONS:")
print(instructions)
```

**Output:**
```
MANUAL PATCH INSTRUCTIONS:

Step 1: Identify Service Boundaries
- Analyze the monolithic code to identify distinct business capabilities
- Examples: User Service, Order Service, Payment Service
- Create separate modules for each service

Step 2: Extract Services
For each service:
1. Create new service directory: services/{service_name}/
2. Move relevant code to service directory
3. Create API interface: api.py
4. Implement business logic: service.py
5. Add tests: test_service.py

Step 3: Define Service Contracts
- Use protocol buffers or JSON schemas
- Define request/response formats
- Document API endpoints

Step 4: Implement Communication
- Choose communication pattern (REST, gRPC, message queue)
- Add service discovery if needed
- Implement circuit breakers for resilience

Step 5: Data Migration
- Identify shared databases
- Implement data access layer per service
- Plan database migration strategy

Testing Recommendations:
1. Test each service independently
2. Integration tests for service communication
3. Load testing for performance validation
4. Chaos testing for resilience verification

Potential Pitfalls:
- Distributed transactions are complex - avoid if possible
- Network latency can impact performance
- Debugging is harder in distributed systems
```

### Example 4: Batch Processing with Progress Callback

```python
def progress_callback(message, progress):
    """Display progress during patching."""
    print(f"[{progress:.1f}%] {message}")

# Multiple findings
findings = [issue1, issue2, issue3, issue4, issue5]

# Apply patches with progress tracking
engine = BlueTeamPatcherEngine(api_key="your-api-key")
report = engine.run_patcher_workflow(
    red_team_findings=findings,
    original_content=code,
    content_type="code",
    strategy=PatchStrategy.HYBRID,
    max_parallel=3
)

print(f"\nBatch complete: {report.summary['successful_patches']}/{report.summary['total_patches']} successful")
```

### Example 5: Export and Analyze Report

```python
# Run patcher workflow
engine = BlueTeamPatcherEngine(api_key="your-api-key")
report = engine.run_patcher_workflow(
    red_team_findings=findings,
    original_content=code,
    content_type="code"
)

# Export as JSON
json_report = engine.export_report(report, format='json')
with open('patch_report.json', 'w') as f:
    f.write(json_report)

# Export as Markdown
md_report = engine.export_report(report, format='markdown')
with open('patch_report.md', 'w') as f:
    f.write(md_report)

# Analyze metrics
print(f"Total Issues: {report.analysis.total_issues}")
print(f"Complexity Distribution: {report.analysis.complexity_distribution}")
print(f"Average Time per Patch: {report.metrics['avg_time_per_patch']:.2f}s")
print(f"Quality Improvements: {report.metrics['quality_improvements']}")
print(f"Regressions Detected: {report.metrics['regressions_detected']}")

# Print recommendations
print("\nRecommendations:")
for rec in report.recommendations:
    print(f"- {rec}")
```

---

## Integration

### Integration with Red Team

```python
from red_team import RedTeam
from blue_team_patcher_engine import BlueTeamPatcherEngine

# Run red team analysis
red_team = RedTeam()
red_team_assessment = red_team.assess_content(
    content=code,
    content_type="code"
)

# Pass findings to blue team patcher
blue_team = BlueTeamPatcherEngine(api_key="your-api-key")
patch_report = blue_team.run_patcher_workflow(
    red_team_findings=red_team_assessment.findings,
    original_content=code,
    content_type="code"
)

# Full cycle complete
print(f"Red team found {len(red_team_assessment.findings)} issues")
print(f"Blue team fixed {patch_report.summary['successful_patches']} issues")
```

### Integration with Solution Validation Pipeline

```python
from solution_validation_pipeline import SolutionValidationPipeline
from blue_team_patcher_engine import BlueTeamPatcherEngine

# Validate original solution
validator = SolutionValidationPipeline()
original_validation = validator.validate(
    content=code,
    requirements=requirements
)

# Fix issues if validation failed
if not original_validation['passed']:
    # Convert validation errors to findings
    findings = convert_validation_to_findings(original_validation)

    # Apply patches
    patcher = BlueTeamPatcherEngine(api_key="your-api-key")
    patch_report = patcher.run_patcher_workflow(
        red_team_findings=findings,
        original_content=code,
        content_type="code"
    )

    # Validate patched solution
    patched_validation = validator.validate(
        content=patch_report.patch_results[-1].patched_content,
        requirements=requirements
    )

    if patched_validation['passed']:
        print("Solution fixed and validated!")
    else:
        print("Additional fixes needed")
```

### Integration with Blue Team

```python
from blue_team import BlueTeam, BlueTeamAssessment
from blue_team_patcher_engine import BlueTeamPatcherEngine

# Legacy blue team
legacy_blue_team = BlueTeam()
legacy_assessment = legacy_blue_team.apply_fixes(
    content=code,
    issues=findings
)

# New patcher engine for complex fixes
patcher_engine = BlueTeamPatcherEngine(api_key="your-api-key")
enhanced_assessment = patcher_engine.run_patcher_workflow(
    red_team_findings=findings,
    original_content=code,
    content_type="code"
)

# Combine results
combined_fixes = (
    legacy_assessment.applied_fixes +
    enhanced_assessment.patch_results
)
```

---

## Troubleshooting

### Issue: Patches fail to apply

**Symptoms:** High failure rate in patch results

**Solutions:**
1. Check API key is valid
2. Verify model is available
3. Try using `PatchStrategy.SEMI_AUTOMATIC` for more control
4. Review error messages in patch results

```python
# Check for errors
for result in report.patch_results:
    if not result.success:
        print(f"Patch {result.patch_id} failed: {result.error_message}")
```

### Issue: Low validation score

**Symptoms:** Validation score below 0.7

**Solutions:**
1. Review patches for regressions
2. Use manual review for complex patches
3. Apply patches incrementally
4. Check quality assessment configuration

```python
# Investigate low scores
if report.validation_summary['overall_validation_score'] < 0.7:
    print("Issues found:")
    for validation in report.validation_summary['patch_validations']:
        if not validation['passed']:
            print(f"- {validation['patch_id']}: {validation['errors']}")
```

### Issue: Patches introduce regressions

**Symptoms:** `regressions_detected` > 0

**Solutions:**
1. Rollback affected patches
2. Review patch diffs carefully
3. Apply patches one at a time
4. Increase testing before applying

```python
# Rollback patches with regressions
for result in report.patch_results:
    validation = next(
        (v for v in report.validation_summary['patch_validations']
         if v['patch_id'] == result.patch_id),
        None
    )
    if validation and validation['regression_detected']:
        engine.applicator.rollback_patch(result.patch_id)
        print(f"Rolled back {result.patch_id} due to regression")
```

### Issue: Complex patches fail

**Symptoms:** High complexity patches have low success rate

**Solutions:**
1. Use `PatchStrategy.MANUAL` for complex patches
2. Break down complex issues into smaller patches
3. Review manual instructions
4. Consider human intervention

```python
# Separate complex and simple patches
complex_patches = [p for p in analysis.recommended_patches
                   if p.context.get('complexity', 0) >= 0.7]
simple_patches = [p for p in analysis.recommended_patches
                  if p.context.get('complexity', 0) < 0.7]

# Apply simple patches automatically, get instructions for complex
auto_results = engine.applicator.apply_patches(
    simple_patches,
    strategy=PatchStrategy.AUTOMATIC
)
manual_instructions = engine.applicator.apply_patches(
    complex_patches,
    strategy=PatchStrategy.MANUAL
)
```

### Issue: Memory issues with large content

**Symptoms:** Out of memory errors with large files

**Solutions:**
1. Process content in chunks
2. Reduce `max_parallel`
3. Filter findings to most critical
4. Use streaming for large files

```python
# Process large content in chunks
def patch_large_content(findings, content, chunk_size=10000):
    results = []
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i+chunk_size]
        chunk_findings = [f for f in findings
                         if f.location and int(f.location.split(':')[1]) in range(i, i+chunk_size)]

        if chunk_findings:
            report = engine.run_patcher_workflow(
                red_team_findings=chunk_findings,
                original_content=chunk,
                content_type="code"
            )
            results.append(report)

    return results
```

---

## Best Practices

1. **Start with Automatic Strategy**: Use `PatchStrategy.AUTOMATIC` for simple, low-risk patches
2. **Use Manual for Critical Issues**: Apply `PatchStrategy.MANUAL` for security or architectural changes
3. **Review Rollback Data**: Always keep rollback data for critical patches
4. **Validate Incrementally**: Run validation after each batch of patches
5. **Monitor Performance**: Track patch application time and success rates
6. **Test Thoroughly**: Run comprehensive tests after patching
7. **Document Changes**: Keep patch reports for future reference
8. **Handle Edge Cases**: Test patches with edge cases and boundary conditions
9. **Version Control**: Commit changes in manageable chunks
10. **Review Recommendations**: Read and act on validation recommendations

---

## Conclusion

The Blue Team Patcher Engine provides a comprehensive, automated solution for fixing issues identified by red team testing. With 15 patch types, multiple strategies, and built-in validation, it enables efficient and reliable code improvement while maintaining quality and preventing regressions.

For more information, see the API reference or examine the test suite for additional examples.
