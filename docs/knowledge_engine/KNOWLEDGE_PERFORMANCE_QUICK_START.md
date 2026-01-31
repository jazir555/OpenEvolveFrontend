# Quick Start Guide: Knowledge & Performance System

## 30-Second Setup

```python
from knowledge_performance_integration import integrate_with_decomposition

# Enable continuous learning
integration = integrate_with_decomposition(
    problem=your_problem,
    extract_knowledge=True,
    track_performance=True
)

# Use relevant artifacts
relevant_artifacts = integration['relevant_artifacts']
```

## Core Components

### 1. KnowledgeArtifactExtractor
Extracts learning from solved problems.

```python
from knowledge_artifact_extractor import KnowledgeArtifactExtractor

extractor = KnowledgeArtifactExtractor("knowledge_artifacts.json")

# Extract artifacts from completed work
artifacts = extractor.extract_artifacts(plan, solutions, validations)

# Retrieve relevant artifacts for new problem
relevant = extractor.retrieve_relevant_artifacts(problem, domain)
```

### 2. PerformanceMetricsTracker
Tracks system performance at all levels.

```python
from performance_metrics_tracker import PerformanceMetricsTracker

tracker = PerformanceMetricsTracker("performance_metrics.json")

# Record metrics
tracker.record_decomposition_metrics(plan, problem, time_taken)
tracker.record_solution_metrics(sub_problem_id, solution, validation, time)

# Generate reports
report = tracker.generate_performance_report(time_period="month")
```

## Common Operations

### Extract and Use Knowledge

```python
# After solving a problem
artifacts = extractor.extract_artifacts(
    decomposition_plan=plan,
    solutions=solutions_dict,
    validation_results=validations_dict
)

# Next time, use relevant artifacts
relevant = extractor.retrieve_relevant_artifacts(new_problem, "research")
plan.metadata['relevant_artifacts'] = relevant
```

### Track Performance

```python
import time

# Track decomposition
start = time.time()
plan = engine.decompose(problem)
tracker.record_decomposition_metrics(plan, problem, time.time() - start)

# Track solutions
for sub_prob_id, solution in solutions.items():
    tracker.record_solution_metrics(
        sub_prob_id, solution, validation, generation_time
    )
```

### View Performance

```python
# Strategy performance
metrics = tracker.get_strategy_performance("semantic", "research", "analysis")
print(f"Quality: {metrics.avg_quality_score}")
print(f"Trend: {metrics.quality_trend}")

# Team performance
metrics = tracker.get_team_performance("team_alpha")
print(f"Success rate: {metrics.success_rate}")

# Domain performance
metrics = tracker.get_domain_performance("research")
print(f"Common strategies: {metrics.common_strategies}")
```

## Artifact Types

| Type | Purpose | Example |
|------|---------|---------|
| **pattern** | Effective approaches | "Semantic strategy works for research" |
| **anti_pattern** | Approaches to avoid | "Avoid dependency decomposition for simple problems" |
| **best_practice** | Proven techniques | "Use analytical approach for validation sub-problems" |
| **insight** | Domain knowledge | "Research problems typically decompose into 4-6 sub-problems" |

## Performance Metrics

### Levels Tracked
1. **Sub-problem**: Individual solution quality and time
2. **Strategy**: Effectiveness by strategy/domain/problem type
3. **Team**: Performance by team and problem type
4. **Domain**: Domain-specific patterns and metrics
5. **Overall**: System-wide performance indicators

### Key Metrics
- Quality scores (0-1)
- Success rates (% passing validation)
- Time metrics (decomposition, solution, validation)
- Trend directions (improving, stable, declining)

## Integration Example

```python
from knowledge_performance_integration import (
    integrate_with_decomposition,
    record_decomposition_completion,
    extract_and_store_knowledge
)

# 1. Setup
integration = integrate_with_decomposition(
    problem=problem,
    strategy="semantic",
    extract_knowledge=True,
    track_performance=True
)

# 2. Decompose (with timing)
import time
start = time.time()
plan = engine.decompose(problem, strategy="semantic")
record_decomposition_completion(
    plan, problem, time.time() - start,
    integration['performance_tracker']
)

# 3. After solving all sub-problems
artifacts = extract_and_store_knowledge(
    plan, solutions, validations,
    integration['artifact_extractor']
)

print(f"Extracted {len(artifacts)} new artifacts!")
```

## Testing

```bash
# Run all tests
pytest test_knowledge_and_performance.py -v

# Run with coverage
pytest test_knowledge_and_performance.py --cov=. --cov-report=html

# Expected: 29 tests passing
```

## Storage

Files are auto-created in JSON format:
- `knowledge_artifacts.json` - Extracted knowledge
- `performance_metrics.json` - Performance metrics

Both can be backed up, shared, or migrated between systems.

## Key Benefits

✅ **Continuous Learning**: System improves with each problem solved
✅ **Performance Monitoring**: Track quality and efficiency over time
✅ **Pattern Recognition**: Automatically identify what works
✅ **Trend Analysis**: Detect improvements or degradations
✅ **Decision Support**: Data-driven recommendations
✅ **Knowledge Preservation**: Best practices captured and reusable

## Next Steps

1. Enable knowledge extraction in your decomposition workflow
2. Let the system learn from 10-20 solved problems
3. Review extracted artifacts and performance reports
4. Use insights to optimize strategies and team assignments
5. Watch the system improve over time!

---

**Full Documentation**: See `KNOWLEDGE_AND_PERFORMANCE_COMPLETE.md`
