# Quality Gate System - Complete Implementation Guide

## Overview

The Quality Gate System is a comprehensive quality control framework for OpenEvolve that acts as the final checkpoint before solutions are assembled into the final output. It ensures that only high-quality solutions pass through to the integration phase.

**Architecture:**
```
Blue Team Solutions → Quality Gate → Solution Integration (if PASSED)
                                   ↓
                              Appeal (if FAILED/CONDITIONAL)
```

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Components](#core-components)
4. [Quality Thresholds](#quality-thresholds)
5. [Multi-Stage Validation](#multi-stage-validation)
6. [Consensus Building](#consensus-building)
7. [Appeal Process](#appeal-process)
8. [Integration Examples](#integration-examples)
9. [API Reference](#api-reference)
10. [Testing](#testing)
11. [Performance](#performance)
12. [Best Practices](#best-practices)

---

## Installation

The Quality Gate System is part of the OpenEvolve Evaluator Team. No additional installation is required beyond the base OpenEvolve setup.

```python
from quality_gate_engine import (
    QualityGateEngine,
    MultiStageValidation,
    ConsensusBuilder,
    create_quality_gate
)
```

---

## Quick Start

### Basic Quality Gate Evaluation

```python
from quality_gate_engine import create_quality_gate, ContentType, QualityLevel
from evaluator_team import EvaluatorAssessment, EvaluationScore, EvaluationMetric

# Create quality gate
gate = create_quality_gate()

# Evaluate solution
report = gate.evaluate(
    assessments=evaluator_assessments,
    content_type=ContentType.CODE,
    quality_level=QualityLevel.STANDARD
)

# Check decision
if report.decision == GateDecision.PASS:
    print("Solution passed quality gate!")
    print(f"Overall score: {report.overall_score:.2f}")
else:
    print("Solution failed quality gate")
    print(f"Issues: {report.critical_issues}")
```

### Multi-Stage Validation

```python
from quality_gate_engine import create_multi_stage_validation

# Create validator
validator = create_multi_stage_validation()

# Run full validation
result = validator.validate(
    assessments=assessments,
    content_type=ContentType.TECHNICAL,
    quality_level=QualityLevel.HIGH,
    enable_appeals=True
)

print(f"Final decision: {result.final_decision.value}")
print(f"Total time: {result.total_time:.2f}s")
```

---

## Core Components

### 1. QualityThresholdManager

Manages quality thresholds for different content types and quality levels.

```python
from quality_gate_engine import QualityThresholdManager, ContentType, QualityLevel

manager = QualityThresholdManager()

# Get threshold
threshold = manager.get_threshold(ContentType.CODE, QualityLevel.STANDARD)
print(f"Minimum overall score: {threshold.min_overall_score}")
print(f"Minimum correctness: {threshold.min_correctness}")

# Set custom threshold
from quality_gate_engine import QualityThreshold

custom_threshold = QualityThreshold(
    content_type=ContentType.TECHNICAL,
    quality_level=QualityLevel.EXCEPTIONAL,
    min_overall_score=95.0,
    min_correctness=95.0,
    adaptive_thresholds=True,
    complexity_modifier=2.0
)

manager.set_threshold(custom_threshold)
```

**Features:**
- Pre-configured thresholds for common content types
- Adaptive thresholds based on complexity
- Custom threshold configuration
- Content-type specific requirements (security for code, compliance for legal/medical)

### 2. QualityGateEngine

Main evaluation engine that assesses solutions against thresholds.

```python
from quality_gate_engine import QualityGateEngine

engine = QualityGateEngine()

# Evaluate
report = engine.evaluate(
    assessments=assessments,
    content_type=ContentType.DOCUMENT,
    quality_level=QualityLevel.STANDARD,
    complexity_score=7  # 1-10 scale for adaptive thresholds
)

# Access results
print(f"Decision: {report.decision.value}")
print(f"Overall score: {report.overall_score:.2f}")
print(f"Critical issues: {len(report.critical_issues)}")
print(f"Recommendations: {report.improvement_recommendations}")
```

**Decision Types:**
- `PASS`: Solution meets all quality standards
- `FAIL`: Solution fails to meet quality standards
- `CONDITIONAL_PASS`: Solution passes with minor issues

### 3. MultiStageValidation

Comprehensive validation workflow with multiple stages.

**Stages:**
1. **Pre-evaluation**: Quick sanity checks
2. **Comprehensive evaluation**: Full quality gate assessment
3. **Post-evaluation**: Final verification
4. **Appeal**: Re-evaluation if requested

```python
from quality_gate_engine import MultiStageValidation, QualityGateEngine

gate = QualityGateEngine()
validator = MultiStageValidation(gate)

# Run validation
result = validator.validate(
    assessments=assessments,
    content_type=ContentType.CODE,
    quality_level=QualityLevel.HIGH
)

# Check stage results
for stage_report in result.stage_reports:
    print(f"{stage_report['stage']}: {stage_report['passed']}")

# Submit appeal if needed
if result.final_decision == GateDecision.FAIL:
    appeal = validator.submit_appeal(
        original_result=result,
        appeal_reason="Solution meets practical requirements",
        additional_context="Additional context",
        requested_revaluation=[EvaluationMetric.CORRECTNESS]
    )

    # Process appeal
    appeal_decision = validator.process_appeal(
        appeal=appeal,
        assessments=assessments,
        content_type=ContentType.CODE,
        quality_level=QualityLevel.STANDARD  # Lower threshold for appeal
    )

    print(f"Appeal decision: {appeal_decision.new_decision.value}")
```

### 4. ConsensusBuilder

Aggregates multiple evaluator opinions into a consensus decision.

**Consensus Methods:**
- `MAJORITY_VOTE`: Simple majority
- `WEIGHTED_VOTE`: Weighted by criterion importance
- `EXPERTISE_WEIGHTED`: Weighted by evaluator expertise
- `BAYESIAN_AGGREGATION`: Bayesian belief aggregation
- `MEDIAN`: Median score
- `TRIMMED_MEAN`: Trimmed mean (removes outliers)

```python
from quality_gate_engine import ConsensusBuilder, ConsensusMethod

builder = ConsensusBuilder()

# Build consensus
result = builder.build_consensus(
    assessments=assessments,
    method=ConsensusMethod.WEIGHTED_VOTE,
    criteria=evaluation_criteria
)

print(f"Consensus score: {result.consensus_score:.2f}")
print(f"Agreement level: {result.agreement_level:.2%}")
print(f"Outliers: {result.outlier_evaluators}")
print(f"Confidence: {result.confidence.value}")
```

---

## Quality Thresholds

### Pre-configured Thresholds

The system includes pre-configured thresholds for common content types:

#### Code (Standard Quality)
```python
min_overall_score: 75.0
min_correctness: 75.0
min_completeness: 75.0
min_clarity: 70.0
min_effectiveness: 70.0
min_efficiency: 65.0
min_maintainability: 65.0
min_security: 75.0
```

#### Document (Standard Quality)
```python
min_overall_score: 75.0
min_correctness: 75.0
min_completeness: 75.0
min_clarity: 80.0  # Higher for documents
min_effectiveness: 70.0
min_efficiency: 60.0
min_maintainability: 60.0
```

#### Legal (Standard Quality)
```python
min_overall_score: 80.0  # Higher bar
min_correctness: 85.0
min_completeness: 80.0
min_clarity: 80.0
min_effectiveness: 75.0
min_compliance: 85.0  # Critical for legal
```

#### Medical (Standard Quality)
```python
min_overall_score: 85.0  # Highest bar
min_correctness: 90.0
min_completeness: 85.0
min_clarity: 80.0
min_effectiveness: 80.0
min_compliance: 90.0  # Critical for medical
min_security: 85.0
```

### Adaptive Thresholds

Thresholds can adapt based on problem complexity:

```python
threshold = QualityThreshold(
    content_type=ContentType.CODE,
    quality_level=QualityLevel.STANDARD,
    min_overall_score=75.0,
    adaptive_thresholds=True,
    complexity_modifier=2.0  # Adjust threshold by 2 points per complexity level
)

# For complexity_score=3 (simple): threshold = 75 + (5-3)*2 = 79
# For complexity_score=8 (complex): threshold = 75 + (5-8)*2 = 69
```

---

## Multi-Stage Validation

### Stage 1: Pre-evaluation

Quick sanity checks before full evaluation:
- Minimum evaluator count
- Valid score ranges
- No extreme disagreements
- All assessments have scores

### Stage 2: Comprehensive Evaluation

Full quality gate evaluation:
- Aggregates scores from all evaluators
- Checks against thresholds
- Generates pass/fail decision
- Identifies issues and recommendations

### Stage 3: Post-evaluation

Final verification:
- Decision consistency
- All evaluators considered
- Recommendations generated appropriately

### Stage 4: Appeal (Optional)

Re-evaluation workflow:
- Submit appeal with reasoning
- Provide additional context
- Request specific metric re-evaluation
- Get appeal decision

---

## Consensus Building

### Choosing a Consensus Method

**MAJORITY_VOTE**: Best for quick decisions with many evaluators
```python
result = builder.build_consensus(
    assessments,
    method=ConsensusMethod.MAJORITY_VOTE
)
```

**WEIGHTED_VOTE**: Best when certain metrics are more important
```python
criteria = [
    EvaluationCriterion(metric=EvaluationMetric.SECURITY, weight=0.3, importance="critical"),
    EvaluationCriterion(metric=EvaluationMetric.CORRECTNESS, weight=0.25, importance="critical")
]

result = builder.build_consensus(
    assessments,
    method=ConsensusMethod.WEIGHTED_VOTE,
    criteria=criteria
)
```

**EXPERTISE_WEIGHTED**: Best when evaluators have different expertise levels
```python
result = builder.build_consensus(
    assessments,
    method=ConsensusMethod.EXPERTISE_WEIGHTED
)
```

**BAYESIAN_AGGREGATION**: Best for handling uncertainty
```python
result = builder.build_consensus(
    assessments,
    method=ConsensusMethod.BAYESIAN_AGGREGATION
)
```

**TRIMMED_MEAN**: Best when there might be outliers
```python
result = builder.build_consensus(
    assessments,
    method=ConsensusMethod.TRIMMED_MEAN
)
```

### Agreement Level

The agreement level (0-1 scale) indicates evaluator consensus:
- **0.9-1.0**: Strong consensus
- **0.75-0.9**: Good consensus
- **0.5-0.75**: Moderate consensus
- **0.25-0.5**: Low consensus
- **0.0-0.25**: Very low consensus

---

## Appeal Process

The appeal process allows re-evaluation of failed solutions.

### Submitting an Appeal

```python
from quality_gate_engine import MultiStageValidation

validator = create_multi_stage_validation()

# Run initial validation
result = validator.validate(assessments, quality_level=QualityLevel.HIGH)

# Submit appeal if failed
if result.final_decision == GateDecision.FAIL:
    appeal = validator.submit_appeal(
        original_result=result,
        appeal_reason="Solution meets practical requirements despite theoretical gaps",
        additional_context="Additional context and clarifications",
        requested_revaluation=[
            EvaluationMetric.EFFECTIVENESS,
            EvaluationMetric.EFFICIENCY
        ]
    )
```

### Processing an Appeal

```python
# Process appeal (may use different quality level)
appeal_decision = validator.process_appeal(
    appeal=appeal,
    assessments=assessments,
    content_type=ContentType.CODE,
    quality_level=QualityLevel.STANDARD  # Lower threshold
)

print(f"Appeal decision: {appeal_decision.new_decision.value}")
print(f"Rationale: {appeal_decision.decision_rationale}")
```

---

## Integration Examples

### Integration with Blue Team Solver

```python
from blue_team_solver_engine import SubProblemSolver, SubProblemInput
from quality_gate_engine import create_multi_stage_validation, ContentType
from evaluator_team import EvaluatorTeam

# Create components
solver = SubProblemSolver(config=api_config)
evaluator_team = EvaluatorTeam()
validator = create_multi_stage_validation()

# Solve sub-problem
sub_problem = SubProblemInput(
    id="sub_1",
    description="Implement user authentication",
    complexity_score=7
)

solution_result = solver.solve_sub_problem(**sub_problem.to_dict())

# Evaluate solution
assessments = evaluator_team.evaluate_content(
    content=solution_result.solution,
    content_type="code"
)

# Quality gate check
validation_result = validator.validate(
    assessments=assessments.assessments,
    content_type=ContentType.CODE,
    quality_level=QualityLevel.STANDARD
)

# Use solution if passed
if validation_result.final_decision == GateDecision.PASS:
    print("Solution approved for integration")
else:
    print("Solution needs improvement")
    print(f"Issues: {validation_result.stage_reports[1]['report'].critical_issues}")
```

### Integration with Solution Integration

```python
from solution_integration import SolutionAssembler
from quality_gate_engine import create_quality_gate, GateDecision

# Create components
gate = create_quality_gate()
assembler = SolutionAssembler()

# Evaluate all sub-solutions before assembly
approved_solutions = {}
for sub_id, solution in all_sub_solutions.items():
    assessments = evaluate_solution(solution)  # Your evaluation function

    report = gate.evaluate(
        assessments=assessments,
        content_type=ContentType.CODE,
        quality_level=QualityLevel.STANDARD
    )

    if report.decision in [GateDecision.PASS, GateDecision.CONDITIONAL_PASS]:
        approved_solutions[sub_id] = solution
    else:
        logger.warning(f"Sub-solution {sub_id} failed quality gate: {report.critical_issues}")

# Assemble only approved solutions
if len(approved_solutions) == len(all_sub_solutions):
    integrated = assembler.assemble_solution(decomposition_plan, approved_solutions)
else:
    logger.error(f"Cannot assemble: {len(all_sub_solutions) - len(approved_solutions)} solutions failed quality gate")
```

---

## API Reference

### QualityThresholdManager

```python
class QualityThresholdManager:
    def __init__(self)
    def get_threshold(self, content_type: ContentType, quality_level: QualityLevel) -> Optional[QualityThreshold]
    def set_threshold(self, threshold: QualityThreshold) -> None
    def get_all_thresholds(self) -> List[QualityThreshold]
    def adjust_for_complexity(self, threshold: QualityThreshold, complexity_score: int) -> QualityThreshold
```

### QualityGateEngine

```python
class QualityGateEngine:
    def __init__(self, threshold_manager: Optional[QualityThresholdManager] = None)
    def evaluate(
        self,
        assessments: List[EvaluatorAssessment],
        content_type: ContentType = ContentType.GENERAL,
        quality_level: QualityLevel = QualityLevel.STANDARD,
        complexity_score: int = 5
    ) -> QualityGateReport
    def get_performance_metrics(self) -> Dict[str, Any]
    def reset_performance_metrics(self) -> None
```

### MultiStageValidation

```python
class MultiStageValidation:
    def __init__(self, quality_gate: QualityGateEngine)
    def validate(
        self,
        assessments: List[EvaluatorAssessment],
        content_type: ContentType = ContentType.GENERAL,
        quality_level: QualityLevel = QualityLevel.STANDARD,
        complexity_score: int = 5,
        enable_appeals: bool = True
    ) -> MultiStageValidationResult
    def submit_appeal(
        self,
        original_result: MultiStageValidationResult,
        appeal_reason: str,
        additional_context: str,
        requested_revaluation: List[EvaluationMetric]
    ) -> AppealRequest
    def process_appeal(
        self,
        appeal: AppealRequest,
        assessments: List[EvaluatorAssessment],
        content_type: ContentType,
        quality_level: QualityLevel
    ) -> AppealDecision
```

### ConsensusBuilder

```python
class ConsensusBuilder:
    def __init__(self)
    def build_consensus(
        self,
        assessments: List[EvaluatorAssessment],
        method: ConsensusMethod = ConsensusMethod.WEIGHTED_VOTE,
        criteria: Optional[List[EvaluationCriterion]] = None
    ) -> ConsensusResult
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest test_quality_gate.py -v

# Run with coverage
pytest test_quality_gate.py -v --cov=quality_gate_engine --cov-report=html

# Run specific test class
pytest test_quality_gate.py::TestQualityGateEngine -v

# Run specific test
pytest test_quality_gate.py::TestQualityGateEngine::test_evaluate_passing_solution -v
```

### Test Coverage

The test suite includes 35+ tests covering:
- Threshold management (9 tests)
- Quality gate engine (12 tests)
- Multi-stage validation (8 tests)
- Consensus building (11 tests)
- Integration tests (3 tests)
- Edge cases (5 tests)
- Performance tests (2 tests)

**Target Coverage:** 90%+

### Test Fixtures

```python
@pytest.fixture
def sample_scores() -> List[EvaluationScore]:
    """Sample evaluation scores for testing"""

@pytest.fixture
def sample_assessment(sample_scores) -> EvaluatorAssessment:
    """Sample evaluator assessment"""

@pytest.fixture
def sample_assessments(sample_scores) -> List[EvaluatorAssessment]:
    """Multiple sample assessments with varying quality"""
```

---

## Performance

### Benchmarks

Based on performance tests:

| Operation | Time | Notes |
|-----------|------|-------|
| Single evaluation | < 1s | With 3-5 evaluators |
| 100 evaluations | < 10s | Batch processing |
| Consensus building (all methods) | < 5s | 6 methods on 3 assessments |
| Full multi-stage validation | < 3s | All stages |

### Optimization Tips

1. **Cache threshold lookups**: Thresholds are cached internally
2. **Batch evaluations**: Process multiple solutions together
3. **Use appropriate consensus method**: Some are faster than others
4. **Limit evaluation history**: History is limited to 100 entries automatically

---

## Best Practices

### 1. Choose Appropriate Quality Levels

```python
# Critical systems (medical, legal): Use HIGH or EXCEPTIONAL
validator.validate(assessments, quality_level=QualityLevel.HIGH)

# Standard applications: Use STANDARD
validator.validate(assessments, quality_level=QualityLevel.STANDARD)

# Prototypes/experimental: Use MINIMAL
validator.validate(assessments, quality_level=QualityLevel.MINIMAL)
```

### 2. Enable Adaptive Thresholds for Complex Problems

```python
threshold = QualityThreshold(
    content_type=ContentType.TECHNICAL,
    quality_level=QualityLevel.STANDARD,
    adaptive_thresholds=True,
    complexity_modifier=2.0
)
```

### 3. Use Multiple Consensus Methods for Important Decisions

```python
builder = ConsensusBuilder()

# Try multiple methods
results = {}
for method in ConsensusMethod:
    results[method.value] = builder.build_consensus(assessments, method=method)

# Compare results
for method, result in results.items():
    print(f"{method}: {result.consensus_score:.2f} (agreement: {result.agreement_level:.2%})")
```

### 4. Handle Conditional Passes Appropriately

```python
report = gate.evaluate(assessments)

if report.decision == GateDecision.PASS:
    integrate_solution(solution)
elif report.decision == GateDecision.CONDITIONAL_PASS:
    # Log warnings but still integrate
    logger.warning(f"Integrating with minor issues: {report.minor_issues}")
    integrate_solution(solution)
else:  # FAIL
    # Require fixes before integration
    logger.error(f"Cannot integrate: {report.critical_issues}")
    trigger_improvement_workflow(solution, report.improvement_recommendations)
```

### 5. Track Performance Metrics

```python
engine = QualityGateEngine()

# Run evaluations
for solution in solutions:
    engine.evaluate(evaluate_solution(solution))

# Check performance
metrics = engine.get_performance_metrics()
print(f"Total evaluations: {metrics['total_evaluations']}")
print(f"Pass rate: {metrics['pass_count'] / metrics['total_evaluations']:.2%}")
print(f"Average score: {metrics['average_score']:.2f}")
print(f"Average time: {metrics['average_time']:.2f}s")
```

### 6. Use Appeals Sparingly

Appeals should be reserved for genuine cases of:
- New information not available during initial evaluation
- Practical considerations not captured by quality metrics
- Edge cases where automatic evaluation is insufficient

---

## Troubleshooting

### Common Issues

**Issue**: Solution fails quality gate unexpectedly

**Solution**:
1. Check which metrics are below threshold
2. Review critical and minor issues in the report
3. Consider using a lower quality level for non-critical content
4. Submit appeal if appropriate

**Issue**: Low agreement among evaluators

**Solution**:
1. Check for outlier evaluators
2. Use trimmed_mean consensus method to exclude outliers
3. Review evaluator expertise levels
4. Consider additional evaluation cycles

**Issue**: Performance is slow

**Solution**:
1. Reduce number of evaluators
2. Use simpler consensus methods (median, majority_vote)
3. Cache threshold lookups
4. Batch process multiple solutions

---

## Future Enhancements

Planned features for future versions:

1. **Machine Learning Quality Prediction**: Use ML to predict quality gate outcomes
2. **Automated Improvement Suggestions**: Generate specific improvement recommendations
3. **Continuous Learning**: Learn from past evaluations to improve thresholds
4. **Custom Quality Metrics**: Support for domain-specific quality metrics
5. **Real-time Quality Monitoring**: Continuous quality monitoring during solution development

---

## Support and Contributing

For issues, questions, or contributions:
- GitHub: [OpenEvolve Repository]
- Documentation: [OpenEvolve Docs]
- Issues: [GitHub Issues]

---

## License

This component is part of OpenEvolve and is released under the MIT License.

---

**Version**: 1.0.0
**Last Updated**: 2025-01-04
**Authors**: OpenEvolve Development Team
