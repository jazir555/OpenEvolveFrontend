# LoongFlow Gauntlet Adapter Documentation

## Overview

The LoongFlow Gauntlet Adapter integrates LoongFlow's Plan-Execute-Summarize (PES) evolutionary system as a **Round 1 quick screening evaluator** in the OpenEvolve gauntlet system.

## Architecture

```
Solution Input
    ↓
┌─────────────────────────────────────────┐
│  Round 1: LoongFlow PES Evaluator       │
│  ├─ Quick PES assessment (10-30 sec)    │
│  ├─ Multi-dimensional scoring           │
│  └─ Threshold-based filtering           │
└─────────────────────────────────────────┘
    ↓
Pass? (>0.6 quality & >0.7 confidence)
    ↓ Yes
┌─────────────────────────────────────────┐
│  Round 2: Red Team (Adversarial)        │
│  └─ Attack and vulnerability testing    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Round 3: Gold Team (Consensus)         │
│  └─ Multi-judge verification            │
└─────────────────────────────────────────┘
    ↓
Final Result
```

## Features

### 1. Fast PES-Based Evaluation
- **Target**: <30 seconds per evaluation
- **Typical**: 10-20 seconds for most problems
- **Optimization**: Early stopping on improvement

### 2. Multi-Dimensional Scoring
- **Correctness** (40% weight): Does it solve the problem?
- **Efficiency** (30% weight): Resource usage
- **Robustness** (20% weight): Edge case handling
- **Creativity** (10% weight): Novelty and innovation

### 3. Configurable Thresholds
```python
config = LoongFlowGauntletConfig(
    quality_threshold=0.6,      # Minimum overall score
    confidence_threshold=0.7,   # Minimum confidence
    max_evaluations=50,         # PES iterations
    enable_detailed_feedback=True
)
```

### 4. Detailed Feedback Generation
- Strengths identification
- Weaknesses highlighting
- Actionable suggestions
- Pass/fail recommendation

## Installation

The LoongFlow Gauntlet Adapter is included with OpenEvolve:

```bash
pip install openevolve[gauntlets]
```

## Quick Start

### Basic Usage

```python
from openevolve.gauntlets import (
    LoongFlowGauntletEvaluator,
    LoongFlowGauntletConfig
)

# Create configuration
config = LoongFlowGauntletConfig(
    quality_threshold=0.6,
    confidence_threshold=0.7
)

# Initialize evaluator
evaluator = LoongFlowGauntletEvaluator(config)

# Evaluate a solution
result = await evaluator.evaluate_solution(
    solution="def solve(): return optimal_solution",
    problem="Optimize the circle packing problem",
    domain="math"
)

# Check result
if result.passed:
    print("✅ Passed Round 1 - Proceed to Red Team")
    print(f"Score: {result.overall_score:.1%}")
    print(f"Confidence: {result.confidence:.1%}")
else:
    print("❌ Failed Round 1 - Do not proceed")
    print(result.feedback)
```

### Batch Evaluation

```python
# Evaluate multiple solutions concurrently
solutions = [
    "def solve_v1(): return 1",
    "def solve_v2(): return 2",
    "def solve_v3(): return 3"
]

results = await evaluator.evaluate_batch(
    solutions=solutions,
    problem="Test problem",
    domain="code"
)

# Process results
for i, result in enumerate(results):
    print(f"Solution {i+1}: {result.overall_score:.1%} - {'PASS' if result.passed else 'FAIL'}")
```

## Configuration

### Configuration Options

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `enable_planning` | bool | True | - | Enable PES planning phase |
| `enable_memory` | bool | True | - | Enable PES memory system |
| `early_stopping` | bool | True | - | Enable early stopping on improvement |
| `plan_temperature` | float | 0.7 | 0.0-2.0 | Temperature for planning LLM |
| `summary_temperature` | float | 0.7 | 0.0-2.0 | Temperature for summary LLM |
| `evaluation_timeout` | int | 30 | 5-300 | Timeout per evaluation (seconds) |
| `max_evaluations` | int | 50 | 10-1000 | Maximum PES evaluations |
| `quality_threshold` | float | 0.5 | 0.0-1.0 | Minimum quality to pass |
| `confidence_threshold` | float | 0.6 | 0.0-1.0 | Minimum confidence to pass |
| `enable_detailed_feedback` | bool | True | - | Enable detailed feedback |
| `correctness_weight` | float | 0.4 | 0.0-1.0 | Weight for correctness |
| `efficiency_weight` | float | 0.3 | 0.0-1.0 | Weight for efficiency |
| `robustness_weight` | float | 0.2 | 0.0-1.0 | Weight for robustness |
| `creativity_weight` | float | 0.1 | 0.0-1.0 | Weight for creativity |

**Note**: Scoring weights must sum to 1.0

### Example Configurations

#### Strict Evaluation (High Quality Filter)
```python
config = LoongFlowGauntletConfig(
    quality_threshold=0.8,
    confidence_threshold=0.8,
    correctness_weight=0.5,
    efficiency_weight=0.2,
    robustness_weight=0.2,
    creativity_weight=0.1
)
```

#### Lenient Evaluation (Quick Screen)
```python
config = LoongFlowGauntletConfig(
    quality_threshold=0.4,
    confidence_threshold=0.5,
    max_evaluations=30,  # Fewer evaluations
    evaluation_timeout=20  # Faster
)
```

#### Creativity-Focused Evaluation
```python
config = LoongFlowGauntletConfig(
    quality_threshold=0.6,
    correctness_weight=0.3,
    efficiency_weight=0.2,
    robustness_weight=0.2,
    creativity_weight=0.3  # Higher creativity weight
)
```

## Result Structure

### GauntletEvaluationResult

```python
@dataclass
class GauntletEvaluationResult:
    solution: str                      # Evaluated solution
    passed: bool                       # Pass/fail decision
    overall_score: float               # Overall score (0.0-1.0)
    confidence: float                  # Confidence (0.0-1.0)

    # Dimension scores
    correctness_score: float
    efficiency_score: float
    robustness_score: float
    creativity_score: float

    # PES metrics
    pes_iterations: int
    pes_evaluations: int
    convergence_quality: float

    # Feedback
    feedback: str                      # Formatted feedback text
    strengths: List[str]               # Identified strengths
    weaknesses: List[str]              # Identified weaknesses
    suggestions: List[str]             # Improvement suggestions

    # Metadata
    evaluation_time: float             # Time in seconds
    timestamp: datetime                # When evaluated
    artifacts: Dict[str, Any]          # Additional metadata
```

## Scoring Algorithm

### Overall Score Calculation

```python
overall_score = (
    correctness_score * correctness_weight +
    efficiency_score * efficiency_weight +
    robustness_score * robustness_weight +
    creativity_score * creativity_weight
)
```

### Confidence Calculation

Confidence is based on:
1. **Iterations Performed**: More iterations = higher confidence
2. **Overall Score**: Higher scores = higher confidence
3. **Convergence Quality**: Stable convergence = higher confidence

```python
base_confidence = min(1.0, iterations / 10.0)
score_adjustment = overall_score * 0.2
convergence_adjustment = (convergence_quality - 0.5) * 0.2

confidence = base_confidence + score_adjustment + convergence_adjustment
```

### Threshold Checking

```python
passed = (
    overall_score >= quality_threshold and
    confidence >= confidence_threshold
)
```

## Feedback Generation

### Example Feedback

```
**Overall Score:** 85.0%
**Confidence:** 90.0%

**Score Breakdown:**
- Correctness: 90.0%
- Efficiency: 85.0%
- Robustness: 82.0%
- Creativity: 75.0%

**Strengths:**
✓ Excellent correctness (90.0%)
✓ Highly efficient approach (85.0%)
✓ Very robust solution (82.0%)

**Weaknesses:**
✗ Conventional approach - lacks innovation

**Suggestions:**
1. Consider alternative, more creative approaches

**Recommendation:** ✅ PASS - Proceed to Round 2 (Red Team)
```

## Domain-Specific Usage

### Mathematics Problems

```python
result = await evaluator.evaluate_solution(
    solution="def prove_theorem(): ...",
    problem="Prove the fundamental theorem of calculus",
    domain="math"
)

# Math problems emphasize correctness and creativity
assert result.correctness_score > 0.8
```

### Code Problems

```python
result = await evaluator.evaluate_solution(
    solution="def optimize_algorithm(): ...",
    problem="Optimize sorting algorithm",
    domain="code"
)

# Code problems emphasize efficiency and robustness
assert result.efficiency_score > 0.7
```

### General Problems

```python
result = await evaluator.evaluate_solution(
    solution="solution approach",
    problem="Design a system architecture",
    domain="general"
)

# General problems use balanced scoring
```

## Performance

### Benchmarks

| Metric | Target | Typical |
|--------|--------|---------|
| Single evaluation | <30s | 10-20s |
| Batch (10 solutions) | <5min | 1-3min |
| Memory usage | <500MB | 200-400MB |
| API calls | <50/eval | 20-40/eval |

### Optimization Tips

1. **Enable Early Stopping**: Reduces unnecessary evaluations
2. **Adjust Max Evaluations**: Lower for faster screening
3. **Use Batch Evaluation**: Parallel processing for multiple solutions
4. **Set Appropriate Timeouts**: Prevent hanging on difficult problems

## Integration with Gauntlet System

### 3-Round Gauntlet Flow

```python
from openevolve.gauntlets import MultiRoundGauntletOrchestrator

# Create orchestrator with LoongFlow as Round 1
orchestrator = MultiRoundGauntletOrchestrator(
    round1_evaluator=loongflow_evaluator,
    round2_evaluator=red_team_evaluator,
    round3_evaluator=gold_team_evaluator
)

# Execute complete gauntlet
result = await orchestrator.execute_gauntlet(
    solution=solution,
    problem=problem
)

# Result includes all rounds
print(f"Round 1: {result.round1.passed}")
print(f"Round 2: {result.round2.passed}")
print(f"Round 3: {result.round3.passed}")
print(f"Overall: {result.final_passed}")
```

### Early Exit Optimization

```python
# If Round 1 fails, don't waste resources on Round 2/3
if not result.round1.passed:
    print("Failed quick screen, skip expensive Red Team")
    return result
```

## Error Handling

### Graceful Degradation

```python
# If LoongFlow is unavailable, evaluator uses fallback mode
evaluator = LoongFlowGauntletEvaluator(config)

if not evaluator.is_available():
    print("⚠️  LoongFlow unavailable, using fallback evaluation")

# Fallback provides basic scoring based on heuristics
result = await evaluator.evaluate_solution(...)
```

### Error Recovery

```python
try:
    result = await evaluator.evaluate_solution(...)
except Exception as e:
    # Evaluator catches exceptions and returns failure result
    assert result.passed is False
    assert "error" in result.feedback.lower()
```

## Testing

### Running Tests

```bash
# Run all gauntlet tests
pytest tests/gauntlets/test_loongflow_gauntlet.py -v

# Run specific test
pytest tests/gauntlets/test_loongflow_gauntlet.py::TestLoongFlowGauntletEvaluator::test_evaluate_solution_success -v

# Run with coverage
pytest tests/gauntlets/test_loongflow_gauntlet.py --cov=openevolve.gauntlets --cov-report=html
```

### Test Coverage

- ✅ Configuration validation
- ✅ Result serialization
- ✅ Single solution evaluation
- ✅ Batch evaluation
- ✅ Error handling
- ✅ Threshold checking
- ✅ Score calculation
- ✅ Feedback generation
- ✅ Creativity assessment
- ✅ Performance benchmarks
- ✅ Integration scenarios

## Troubleshooting

### Common Issues

#### 1. LoongFlow Not Available

**Problem**: `LoongFlow not available, using fallback evaluation`

**Solution**:
```bash
# Install LoongFlow
pip install loongflow

# Or use as submodule
git submodule add https://github.com/baidu-baige/LoongFlow.git
```

#### 2. Evaluations Timing Out

**Problem**: Evaluations exceed timeout

**Solution**:
```python
config = LoongFlowGauntletConfig(
    evaluation_timeout=60,  # Increase timeout
    max_evaluations=30      # Reduce evaluations
)
```

#### 3. Too Many Failures

**Problem**: Most solutions fail Round 1

**Solution**:
```python
config = LoongFlowGauntletConfig(
    quality_threshold=0.4,      # Lower threshold
    confidence_threshold=0.5    # Lower confidence
)
```

#### 4. Memory Issues

**Problem**: High memory usage with batch evaluation

**Solution**:
```python
# Process in smaller batches
batch_size = 5
for i in range(0, len(solutions), batch_size):
    batch = solutions[i:i+batch_size]
    results = await evaluator.evaluate_batch(batch, problem, domain)
```

## API Reference

### LoongFlowGauntletEvaluator

```python
class LoongFlowGauntletEvaluator:
    def __init__(self, config: LoongFlowGauntletConfig)
    async def evaluate_solution(
        self,
        solution: str,
        problem: str,
        domain: str = "general",
        **kwargs
    ) -> GauntletEvaluationResult
    async def evaluate_batch(
        self,
        solutions: List[str],
        problem: str,
        domain: str = "general",
        **kwargs
    ) -> List[GauntletEvaluationResult]
    def get_config(self) -> LoongFlowGauntletConfig
    def is_available(self) -> bool
```

### LoongFlowGauntletConfig

```python
class LoongFlowGauntletConfig(BaseModel):
    enable_planning: bool = True
    enable_memory: bool = True
    early_stopping: bool = True
    plan_temperature: float = 0.7
    summary_temperature: float = 0.7
    evaluation_timeout: int = 30
    max_evaluations: int = 50
    quality_threshold: float = 0.5
    confidence_threshold: float = 0.6
    enable_detailed_feedback: bool = True
    correctness_weight: float = 0.4
    efficiency_weight: float = 0.3
    robustness_weight: float = 0.2
    creativity_weight: float = 0.1
```

### GauntletEvaluationResult

```python
@dataclass
class GauntletEvaluationResult:
    solution: str
    passed: bool
    overall_score: float
    confidence: float
    correctness_score: float
    efficiency_score: float
    robustness_score: float
    creativity_score: float
    pes_iterations: int
    pes_evaluations: int
    convergence_quality: float
    feedback: str
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    evaluation_time: float
    timestamp: datetime
    artifacts: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GauntletEvaluationResult"
```

## Best Practices

### 1. Threshold Tuning

Start with lenient thresholds and adjust based on results:

```python
# Initial screening (lenient)
config = LoongFlowGauntletConfig(
    quality_threshold=0.4,
    confidence_threshold=0.5
)

# Production filtering (strict)
config = LoongFlowGauntletConfig(
    quality_threshold=0.7,
    confidence_threshold=0.8
)
```

### 2. Domain-Specific Weights

Adjust weights for different domains:

```python
# Math: Emphasize correctness
math_config = LoongFlowGauntletConfig(
    correctness_weight=0.6,
    efficiency_weight=0.2,
    robustness_weight=0.1,
    creativity_weight=0.1
)

# Engineering: Emphasize robustness
engineering_config = LoongFlowGauntletConfig(
    correctness_weight=0.3,
    efficiency_weight=0.3,
    robustness_weight=0.3,
    creativity_weight=0.1
)
```

### 3. Feedback Utilization

Use feedback for iterative improvement:

```python
result = await evaluator.evaluate_solution(...)

if not result.passed:
    # Address weaknesses
    for weakness in result.weaknesses:
        print(f"Fix: {weakness}")

    # Implement suggestions
    for suggestion in result.suggestions:
        print(f"Suggestion: {suggestion}")

    # Re-evaluate improved solution
    improved_solution = improve_solution(result.solution, result.suggestions)
    new_result = await evaluator.evaluate_solution(improved_solution, ...)
```

### 4. Batch Processing

For multiple solutions, use batch evaluation:

```python
# Bad: Sequential evaluation
for solution in solutions:
    result = await evaluator.evaluate_solution(solution, ...)

# Good: Concurrent batch evaluation
results = await evaluator.evaluate_batch(solutions, ...)
```

## Migration Guide

### From Manual Evaluation

**Before**:
```python
# Manual evaluation
score = manual_evaluate(solution)
if score > 0.6:
    proceed_to_red_team(solution)
```

**After**:
```python
# LoongFlow gauntlet evaluation
result = await evaluator.evaluate_solution(solution, problem, domain)
if result.passed:
    proceed_to_red_team(solution)
```

### From Single-Round Gauntlets

**Before**:
```python
# Only Red Team
red_result = red_team.evaluate(solution)
```

**After**:
```python
# 3-round with LoongFlow screen
r1_result = await loongflow_evaluator.evaluate_solution(solution, ...)
if r1_result.passed:
    red_result = red_team.evaluate(solution)
```

## Future Enhancements

- [ ] LLM-based creativity assessment
- [ ] Domain-specific evaluators
- [ ] Adaptive threshold tuning
- [ ] Cross-solution comparison
- [ ] Learning from historical data
- [ ] Integration with Knowledge Engine

## Contributing

To contribute improvements:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This adapter is part of OpenEvolve and follows the same license.

## Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: Full documentation at `docs/`
