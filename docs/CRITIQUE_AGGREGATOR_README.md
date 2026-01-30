# Critique Aggregator - Production-Ready Implementation

**Author:** OpenEvolve Frontend Team
**Created:** 2026-01-22
**License:** MIT
**Status:** Production-Ready

---

## Overview

The Critique Aggregator is a comprehensive system for collecting, aggregating, and analyzing critique reports from multiple judges (AI models, human evaluators, automated tests, security scanners, etc.). It integrates seamlessly with the Sovereign-Grade Decomposition (SGD) workflow orchestrator and OpenEvolve system.

## Features

- **Multi-Judge Aggregation**: Collect critiques from diverse sources
- **Weighted Scoring**: Apply custom weights to different judges or judge types
- **Approval Calculation**: Determine solution approval based on configurable thresholds
- **Comprehensive Summaries**: Generate detailed summaries of all critiques
- **Improvement Extraction**: Consolidate and prioritize improvement suggestions
- **Consensus Measurement**: Quantify agreement among judges
- **Outlier Detection**: Identify and handle outlier opinions
- **Audit Trail**: Export/import functionality for compliance
- **Full Type Hints**: Complete type annotations for IDE support
- **Comprehensive Testing**: 12+ unit tests with high coverage

## Installation

No external dependencies required beyond Python 3.8+. The module uses only standard library modules.

```bash
# Place in your Python path
cp critique_aggregator.py /path/to/your/project/
```

## Quick Start

```python
from critique_aggregator import (
    CritiqueAggregator,
    JudgeReport,
    JudgeType,
    CritiqueSeverity
)

# Create aggregator
aggregator = CritiqueAggregator()

# Create judge reports
judge_reports = [
    JudgeReport(
        judge_name="gpt-4",
        judge_type=JudgeType.AI_MODEL,
        is_approved=True,
        score=0.85,
        feedback="Good solution with minor improvements needed",
        improvements=["Add error handling", "Optimize loops"],
        severity=CritiqueSeverity.MEDIUM
    ),
    # ... more judges
]

# Create critique report
critique_report = aggregator.create_critique_report(
    solution_id="solution_123",
    gauntlet_name="red_team_gauntlet",
    critiques=judge_reports
)

# Access results
print(f"Approved: {critique_report.is_approved}")
print(f"Score: {critique_report.aggregate_score:.2f}")
print(f"Consensus: {critique_report.consensus_score:.2f}")
print(f"Improvements: {critique_report.improvements_needed}")
```

## Data Models

### JudgeReport

Individual critique from a single judge:

```python
@dataclass
class JudgeReport:
    judge_name: str                      # Name/ID of the judge
    judge_type: JudgeType                # Type of judge
    is_approved: bool                    # Approval decision
    score: float                         # Score (0.0 to 1.0)
    feedback: str                        # Detailed feedback
    improvements: List[str]              # List of improvements
    severity: CritiqueSeverity           # Severity level
    confidence: float = 1.0              # Judge confidence (0.0 to 1.0)
    metrics: Dict[str, Any]              # Additional metrics
    timestamp: datetime                  # When critique was generated
    metadata: Dict[str, Any]             # Additional context
```

### CritiqueReport

Aggregated critique from multiple judges:

```python
@dataclass
class CritiqueReport:
    solution_attempt_id: str             # Solution being critiqued
    gauntlet_name: str                   # Gauntlet used
    is_approved: bool                    # Overall approval
    reports_by_judge: List[JudgeReport]  # Individual reports
    summary: str                         # Comprehensive summary
    aggregate_score: float               # Weighted average score
    consensus_score: float               # Agreement measure (0-1)
    improvements_needed: List[str]       # Consolidated improvements
    approval_threshold: float            # Minimum score for approval
    created_at: datetime                 # When report was generated
    metadata: Dict[str, Any]             # Additional metadata
```

### JudgeType

Types of judges supported:

- `AI_MODEL`: AI/ML models (GPT-4, Claude, etc.)
- `HUMAN`: Human evaluators
- `AUTOMATED_TEST`: Test suites (pytest, unittest, etc.)
- `LINTING_TOOL`: Code quality tools (pylint, ESLint, etc.)
- `SECURITY_SCANNER`: Security analyzers (OWASP ZAP, etc.)
- `PERFORMANCE_ANALYZER`: Performance profilers

### CritiqueSeverity

Severity levels for issues:

- `CRITICAL`: Must fix before approval
- `HIGH`: Should fix before production
- `MEDIUM`: Important improvements
- `LOW`: Minor improvements
- `INFO`: Informational only

## Configuration

### AggregationConfig

Customize aggregation behavior:

```python
from critique_aggregator import AggregationConfig, JudgeType

config = AggregationConfig(
    default_approval_threshold=0.7,      # Minimum score for approval
    default_weights={
        JudgeType.HUMAN: 1.0,
        JudgeType.AI_MODEL: 0.9,
        JudgeType.SECURITY_SCANNER: 1.0,
        JudgeType.AUTOMATED_TEST: 0.8
    },
    min_judges_required=2,               # Minimum judges required
    enable_outlier_detection=True,       # Detect outliers
    outlier_std_dev_threshold=2.0,       # Outlier threshold (std devs)
    consensus_algorithm="std_dev",       # "std_dev", "mean_deviation", "pairwise_agreement"
    summary_max_length=2000,             # Max summary length
    extract_improvements=True            # Extract improvements
)

aggregator = CritiqueAggregator(config)
```

## Core Methods

### create_critique_report()

Create a comprehensive critique report:

```python
critique_report = aggregator.create_critique_report(
    solution_id="solution_123",
    gauntlet_name="red_team_gauntlet",
    critiques=judge_reports,
    weights={"gpt-4": 1.0, "human_reviewer": 1.5},  # Optional custom weights
    threshold=0.75  # Optional custom threshold
)
```

### aggregate_judge_reports()

Aggregate reports with weighting:

```python
aggregated = aggregator.aggregate_judge_reports(
    reports=judge_reports,
    weights={"judge_1": 1.0, "judge_2": 0.8}
)
```

### calculate_approval()

Calculate approval status:

```python
is_approved = aggregator.calculate_approval(
    reports=judge_reports,
    threshold=0.7
)
```

**Approval Logic:**
- Weighted average score must meet threshold
- No critical severity issues present
- At least one judge approves

### generate_summary()

Generate comprehensive summary:

```python
summary = aggregator.generate_summary(
    reports=judge_reports,
    max_length=2000  # Optional
)
```

**Summary Includes:**
- Overall score and approval rate
- Critical and high-priority issues
- Detailed judge feedback
- Common themes

### extract_improvements()

Extract and prioritize improvements:

```python
improvements = aggregator.extract_improvements(
    reports=judge_reports,
    max_improvements=20
)
```

**Prioritization:**
1. Grouped by severity
2. Deduplicated
3. Sorted by severity and score impact

### calculate_consensus()

Calculate agreement among judges:

```python
consensus = aggregator.calculate_consensus(reports)
```

**Algorithms:**
- `std_dev`: Based on standard deviation of scores
- `mean_deviation`: Based on mean absolute deviation
- `pairwise_agreement`: Based on pairwise agreement rates

## Integration with SGD Workflow Orchestrator

```python
from critique_aggregator import CritiqueAggregator, JudgeReport, JudgeType
from sgd_workflow_orchestrator import SGDWorkflowOrchestrator

class IntegratedSGDWorkflow:
    def __init__(self):
        self.orchestrator = SGDWorkflowOrchestrator()
        self.critique_aggregator = CritiqueAggregator()

    def evaluate_solution(self, solution_attempt, gauntlet_name):
        """Evaluate a solution using gauntlet judges."""

        # Run gauntlet (this would call OpenEvolve API)
        judge_reports = self._run_gauntlet(solution_attempt, gauntlet_name)

        # Create critique report
        critique_report = self.critique_aggregator.create_critique_report(
            solution_attempt_id=solution_attempt.id,
            gauntlet_name=gauntlet_name,
            critiques=judge_reports
        )

        # Use approval decision
        if critique_report.is_approved:
            print(f"Solution approved with score: {critique_report.aggregate_score:.2f}")
            return True
        else:
            print(f"Solution rejected. Improvements needed:")
            for improvement in critique_report.improvements_needed:
                print(f"  - {improvement}")
            return False
```

## Advanced Usage

### Custom Weights by Judge

```python
weights = {
    "senior_developer": 1.5,    # More weight to senior reviewer
    "gpt-3.5-turbo": 0.7,       # Less weight to smaller model
    "security_scanner": 1.0     # Full weight to security
}

report = aggregator.create_critique_report(
    solution_id="solution_123",
    gauntlet_name="security_gauntlet",
    critiques=judge_reports,
    weights=weights
)
```

### Custom Weights by Judge Type

```python
weights = {
    JudgeType.HUMAN: 1.0,
    JudgeType.AI_MODEL: 0.8,
    JudgeType.SECURITY_SCANNER: 1.2
}
```

### Export and Import

```python
from critique_aggregator import export_critique_report, import_critique_report

# Export to JSON
export_critique_report(critique_report, "audit.json", format="json")

# Export to TXT
export_critique_report(critique_report, "audit.txt", format="txt")

# Import back
restored = import_critique_report("audit.json")
```

### Multi-Round Iteration

```python
iterations = []
for iteration in range(1, 4):
    # Get feedback
    critique_report = aggregator.create_critique_report(
        solution_id=f"solution_v{iteration}",
        gauntlet_name="quality_gauntlet",
        critiques=get_feedback_for_iteration(iteration)
    )

    iterations.append({
        "iteration": iteration,
        "score": critique_report.aggregate_score,
        "approved": critique_report.is_approved,
        "improvements": critique_report.improvements_needed
    })

    if critique_report.is_approved:
        print(f"Solution approved at iteration {iteration}")
        break
```

## Edge Cases Handled

1. **Empty Reports**: Raises `ValueError` with clear message
2. **Single Judge**: Consensus = 1.0 (perfect agreement)
3. **Critical Severity**: Auto-rejects regardless of score
4. **Outlier Detection**: Identifies and excludes outliers (optional)
5. **Invalid Scores**: Validates scores are in [0.0, 1.0]
6. **Invalid Thresholds**: Validates thresholds are in [0.0, 1.0]
7. **Missing Weights**: Falls back to config defaults
8. **Unicode Issues**: Handles international characters properly

## Error Handling

```python
try:
    critique_report = aggregator.create_critique_report(
        solution_id="solution_123",
        gauntlet_name="gauntlet",
        critiques=[]  # Empty list
    )
except ValueError as e:
    print(f"Error: {e}")
    # Output: "Cannot create critique report: no critiques provided"
```

## Testing

Run the included unit tests:

```bash
# Run all tests
python -m unittest critique_aggregator

# Run specific test class
python -m unittest critique_aggregator.TestCritiqueAggregator

# Run with verbose output
python -m unittest critique_aggregator -v
```

**Test Coverage:**
- Basic report creation
- Custom weights
- Approval calculation (unanimous, rejection, critical severity)
- Summary generation
- Improvement extraction
- Consensus calculation
- Serialization (to_dict/from_dict)
- Validation (invalid scores, invalid thresholds)
- Error handling (empty reports)

## Examples

Run the included examples:

```bash
python critique_aggregator_examples.py
```

**Examples Included:**
1. Red Team Gauntlet (security-focused evaluation)
2. Gold Team Verification (quality-focused evaluation)
3. SGD Workflow Integration
4. Multi-Round Iteration (progress tracking)
5. Advanced Configuration (custom weights and settings)
6. Audit Trail (export/import for compliance)

## API Reference

### Classes

- `CritiqueAggregator`: Main aggregation engine
- `JudgeReport`: Individual judge's critique
- `CritiqueReport`: Aggregated critique report
- `AggregationConfig`: Configuration options

### Enums

- `JudgeType`: Types of judges
- `CritiqueSeverity`: Severity levels

### Functions

- `create_sample_judge_reports()`: Generate sample reports for testing
- `export_critique_report()`: Export report to file
- `import_critique_report()`: Import report from file

## Best Practices

1. **Always validate scores**: Ensure scores are in [0.0, 1.0] range
2. **Use appropriate weights**: Human reviewers generally > AI models
3. **Set sensible thresholds**: 0.7 is a good default, adjust based on needs
4. **Enable outlier detection**: For large judge pools (>5)
5. **Export for audit**: Maintain permanent record of evaluations
6. **Handle critical severity**: Always address critical issues first
7. **Track iterations**: Monitor progress across solution iterations

## Performance Considerations

- **Time Complexity**: O(n) for most operations, where n = number of judges
- **Space Complexity**: O(n) for storing judge reports
- **Optimized for**: 1-50 judges typical use case
- **Bottlenecks**: None identified (uses only stdlib)

## Logging

The module uses Python's standard logging:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('critique_aggregator')

# Logs include:
# - Initialization events
# - Aggregation operations
# - Weight applications
# - Outlier detections
# - Approval decisions
```

## Compliance Features

- **Audit Trail**: Export all reports to JSON/TXT
- **Reproducibility**: Deterministic aggregation
- **Transparency**: All weights and decisions logged
- **Traceability**: Timestamps on all reports
- **Verifiability**: Full serialization/deserialization

## Future Enhancements

Potential future improvements:
- Database persistence backend
- Real-time streaming updates
- Machine learning-based consensus
- Multi-language support
- Web dashboard UI
- REST API wrapper

## License

MIT License - See LICENSE file for details

## Support

For issues, questions, or contributions, please contact the OpenEvolve Frontend Team.

## Changelog

### Version 1.0.0 (2026-01-22)
- Initial production-ready release
- Full type hints
- Comprehensive unit tests
- Integration with SGD workflow orchestrator
- Export/import functionality
- Outlier detection
- Multiple consensus algorithms
- Complete documentation
