# Solution Integration System - Quick Reference

## Quick Start

```python
from decomposition_engine import DecompositionEngine

# One-line complete workflow
engine = DecompositionEngine()
result = engine.decompose_and_solve(problem)

# Access integrated solution
print(result.integrated_solution.assembled_content)
```

## Assembly Strategies

| Strategy | Best For | Complexity |
|----------|----------|------------|
| `hierarchical` | Complex dependencies | Medium |
| `linear` | Simple, independent | Low |
| `parallel` | Parallelizable tasks | Medium |
| `adaptive` | Unknown structure | Low |

## Conflict Types

| Type | Description | Severity |
|------|-------------|----------|
| `contradiction` | Opposing statements | High |
| `overlap` | Duplicate content | Medium |
| `dependency` | Missing solution | Critical |
| `inconsistency` | Conflicting approaches | Medium |

## Resolution Strategies

| Strategy | Description | Speed |
|----------|-------------|-------|
| `priority` | First wins | Fast |
| `merge` | Combine content | Medium |
| `llm` | AI-mediated | Slow |
| `manual` | Human review | N/A |

## Quality Metrics

All metrics are 0.0-1.0 (higher is better, except conflict_score):

- `completeness_score` - All aspects addressed?
- `consistency_score` - No contradictions?
- `coherence_score` - Good flow?
- `integration_quality` - Fits well?
- `conflict_score` - Lower is better!
- `overall_score` - Weighted average

## Common Patterns

### Basic Assembly
```python
from problem_recomposition import SolutionAssembler

assembler = SolutionAssembler()
integrated = assembler.assemble_solution(plan, solutions, "hierarchical")
```

### Custom Conflict Detection
```python
from problem_recomposition import ConflictDetector

detector = ConflictDetector()
conflicts = detector.detect_conflicts(solutions, sub_problems)
```

### Validate Solution
```python
from problem_recomposition import SolutionValidator

validator = SolutionValidator()
results = validator.validate_solution(integrated, problem)
```

### End-to-End with Custom Strategy
```python
result = engine.decompose_and_solve(
    problem,
    solve_sub_problems=True,
    assemble_solution=True,
    assembly_strategy="adaptive",  # Choose strategy
    validate_solution=True
)
```

## Factory Functions

```python
from problem_recomposition import create_solution_assembler, create_solution_validator

assembler = create_solution_assembler()
validator = create_solution_validator()
```

## Accessing Results

```python
# Integrated content
content = result.integrated_solution.assembled_content

# Quality metrics
quality = result.integrated_solution.quality_metrics
print(f"Overall: {quality.overall_score:.2%}")

# Conflicts
detected = result.integrated_solution.conflicts_detected
resolved = result.integrated_solution.conflicts_resolved

# Validation
for vr in result.integrated_solution.validation_results:
    print(f"{vr.validator}: {vr.passed}")
```

## Troubleshooting

### No conflicts detected?
- Check solutions have actual content
- Verify dependency relationships exist
- Ensure content has meaningful differences

### Low quality scores?
- Check for unresolved conflicts
- Verify all sub-problems have solutions
- Review assembly strategy choice

### Circular dependencies?
- System handles gracefully
- Falls back to linear ordering
- Check dependency graph structure

## Performance Tips

1. Use `hierarchical` for complex problems
2. Use `linear` for simple cases
3. Cache `ConflictDetector` instances
4. Batch validation when possible

## See Also

- `SOLUTION_INTEGRATION_COMPLETE.md` - Full documentation
- `test_problem_recomposition.py` - Usage examples
- `problem_recomposition.py` - API documentation
