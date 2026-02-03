# LoongFlow PES Knowledge Artifacts Documentation

This document describes the 5 types of knowledge artifacts extracted from LoongFlow Plan-Execute-Summarize (PES) evolutionary runs.

## Overview

The LoongFlow integration extracts knowledge artifacts from each phase of the PES evolutionary algorithm:

```
┌─────────────────────────────────────────────────────────────┐
│                    LoongFlow PES Process                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. PLANNING         →  PlanningStrategyArtifact             │
│     └─ Strategy generation                                    │
│     └─ Optimization approach                                 │
│                                                               │
│  2. EXECUTION        →  ExecutionPatternArtifact             │
│     ├─ Iterative evolution                                    │
│     ├─ Early stopping                                         │
│     └─ Convergence tracking                                   │
│                                                               │
│  3. SUMMARY          →  ReflectionInsightArtifact             │
│     ├─ What worked                                              │
│     ├─ What failed                                             │
│     └─ Recommendations                                        │
│                                                               │
│  4. EVOLUTION        →  EvolutionaryLineageArtifact           │
│     ├─ Ancestry tracking                                       │
│     ├─ Generational history                                   │
│     └─ Mutation patterns                                      │
│                                                               │
│  5. BEST SOLUTION    →  OptimizedSolutionArtifact             │
│     └─ Final optimized code/solution                          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Artifact Types

### 1. PlanningStrategyArtifact

**Description:** Captures the strategic approach generated during the planning phase of PES.

**Source Phase:** Planning (Plan-Execute-Summarize)

**Content Structure:**
```json
{
  "strategy": "Use gradient descent with momentum and adaptive learning rate",
  "success_rate": 0.85,
  "iterations_planned": 50,
  "approach": "hybrid_evolutionary"
}
```

**Metadata Fields:**
- `problem`: Problem description
- `problem_type`: Type classification (e.g., "portfolio_optimization")
- `success_rate`: Predicted success rate (0-1)
- `iterations_planned`: Number of iterations planned
- `planning_approach`: Approach type (e.g., "hybrid_evolutionary", "gradient_based")

**Temporal Metadata:**
- `valid_at`: Timestamp when strategy became valid
- `invalid_at`: Null (still valid)
- `created_at`: Extraction timestamp

**Confidence Score:** 0.8 (high confidence in planning output)

**Use Cases:**
- Strategy retrieval for similar problems
- Planning pattern analysis
- Approach effectiveness comparison

**Example Query:**
```cypher
MATCH (a:KnowledgeArtifact {
  artifact_type: 'planning_strategy',
  source: 'loongflow_pes'
})
WHERE a.metadata.problem_type = 'portfolio_optimization'
AND a.metadata.success_rate > 0.7
RETURN a
ORDER BY a.metadata.success_rate DESC
LIMIT 10
```

---

### 2. ExecutionPatternArtifact

**Description:** Captures execution patterns, efficiency metrics, and early stopping behavior from the execution phase.

**Source Phase:** Execution (Plan-Execute-Summarize)

**Content Structure:**
```json
{
  "early_stopping_events": [15, 25],
  "convergence_rate": 0.95,
  "iterations_to_best": 25,
  "total_evaluations": 30
}
```

**Metadata Fields:**
- `problem`: Problem description
- `problem_type`: Type classification
- `efficiency_gain`: Percentage improvement from early stopping (0-1)
- `time_saved_seconds`: Actual time saved
- `early_stop_count`: Number of early stops triggered

**Temporal Metadata:**
- `valid_at`: Timestamp when pattern was observed
- `invalid_at`: Null (still valid)
- `created_at`: Extraction timestamp

**Confidence Score:** 0.9 (very high confidence in execution metrics)

**Use Cases:**
- Early stopping strategy optimization
- Computational resource planning
- Convergence pattern analysis

**Key Metrics:**
- **Efficiency Gain**: `(planned_evaluations - actual_evaluations) / planned_evaluations`
- **Convergence Rate**: How quickly the algorithm converged
- **Iterations to Best**: When the best solution was found

**Example Query:**
```cypher
MATCH (a:KnowledgeArtifact {
  artifact_type: 'execution_pattern',
  source: 'loongflow_pes'
})
WHERE a.metadata.problem_type = 'scientific'
RETURN AVG(a.metadata.efficiency_gain) as avg_efficiency
```

---

### 3. ReflectionInsightArtifact

**Description:** Captures insights, lessons learned, and recommendations from the summary/reflection phase.

**Source Phase:** Summary (Plan-Execute-Summarize)

**Content Structure:**
```json
{
  "insights": "Momentum helps escape local optima effectively",
  "what_worked": ["momentum", "adaptive_learning_rate", "early_stopping"],
  "what_failed": ["fixed_learning_rate", "large_batch_size"],
  "recommendations": ["Use momentum in future runs"]
}
```

**Metadata Fields:**
- `problem`: Problem description
- `problem_type`: Type classification
- `what_worked`: List of successful techniques
- `what_failed`: List of failed techniques
- `recommendations`: List of recommendations for future runs

**Temporal Metadata:**
- `valid_at`: Timestamp when insights were generated
- `invalid_at`: Null (unless technique is later proven ineffective)
- `created_at`: Extraction timestamp

**Confidence Score:** 0.7 (moderate confidence - insights need validation)

**Use Cases:**
- Technique selection for new problems
- Anti-pattern avoidance
- Best practice identification

**Validation:**
Insights should be validated across multiple runs before high-confidence application:
1. Single run: confidence = 0.7
2. 3+ confirmations: confidence = 0.85
3. 10+ confirmations: confidence = 0.95

**Example Query:**
```cypher
MATCH (a:KnowledgeArtifact {
  artifact_type: 'reflection_insight',
  source: 'loongflow_pes'
})
WHERE 'momentum' IN a.metadata.what_worked
RETURN a.metadata.problem_type, COUNT(*) as usage_count
ORDER BY usage_count DESC
```

---

### 4. EvolutionaryLineageArtifact

**Description:** Captures the complete evolutionary tree, ancestry tracking, and mutation patterns.

**Source Phase:** Evolution (throughout PES execution)

**Content Structure:**
```json
{
  "generations": 10,
  "avg_branching": 2.5,
  "total_mutations": 45,
  "best_lineage": ["root", "gen1", "gen2", "gen3", "best"],
  "mutation_types": {
    "parameter_tweak": 20,
    "structure_change": 15,
    "hybridization": 10
  }
}
```

**Metadata Fields:**
- `problem`: Problem description
- `problem_type`: Type classification
- `generations`: Number of generations evolved
- `branching_factor`: Average offspring per solution
- `total_mutations`: Total mutations applied

**Temporal Metadata:**
- `valid_at`: Timestamp when lineage was recorded
- `invalid_at`: Null (historical record)
- `created_at`: Extraction timestamp

**Confidence Score:** 0.8 (high confidence in lineage tracking)

**Use Cases:**
- Evolutionary dynamics analysis
- Mutation strategy optimization
- Population diversity management

**Lineage Analysis:**
- **Branching Factor**: High branching → more exploration, lower branching → more exploitation
- **Generations**: More generations → deeper search, but higher cost
- **Mutation Distribution**: Reveals which mutation types are most effective

**Example Query:**
```cypher
MATCH (a:KnowledgeArtifact {
  artifact_type: 'evolutionary_lineage',
  source: 'loongflow_pes'
})
WHERE a.metadata.generations > 5
RETURN a.metadata.problem_type,
       AVG(a.metadata.branching_factor) as avg_branching
```

---

### 5. OptimizedSolutionArtifact

**Description:** Captures the best solution found during the evolutionary run, including code and performance metrics.

**Source Phase:** Final Best Solution (after execution)

**Content Structure:**
```python
def optimize_portfolio(weights, returns, risk_tolerance):
    # Momentum-based optimization
    velocity = np.zeros_like(weights)
    momentum = 0.9

    for i in range(100):
        gradient = compute_gradient(weights, returns)
        velocity = momentum * velocity + 0.01 * gradient
        weights = weights - velocity

    return weights
```

**Metadata Fields:**
- `problem`: Problem description
- `problem_type`: Type classification
- `fitness`: Final fitness score (0-1)
- `iteration`: Iteration when best was found
- `improvement_over_baseline`: Percentage improvement over baseline

**Temporal Metadata:**
- `valid_at`: Timestamp when solution was validated
- `invalid_at`: Null (until superseded)
- `created_at`: Extraction timestamp

**Confidence Score:** 0.9 (very high confidence in validated solution)

**Use Cases:**
- Solution retrieval for similar problems
- Baseline comparison
- Solution template/library building

**Solution Quality Metrics:**
- **Fitness**: How well the solution solves the problem
- **Improvement**: How much better than baseline/initial solution
- **Iteration**: Early discovery (low iteration) → efficient search

**Example Query:**
```cypher
MATCH (a:KnowledgeArtifact {
  artifact_type: 'optimized_solution',
  source: 'loongflow_pes'
})
WHERE a.metadata.problem_type = 'portfolio_optimization'
AND a.metadata.fitness > 0.9
RETURN a
ORDER BY a.metadata.fitness DESC
LIMIT 5
```

---

## Temporal Knowledge Graph Structure

All artifacts follow the canonical KnowledgeArtifact structure with temporal metadata:

```python
@dataclass
class KnowledgeArtifact:
    id: str                           # Unique identifier
    content: str                      # Artifact content
    artifact_type: str                # Type (one of 5 above)
    valid_at: datetime                # When knowledge becomes valid
    invalid_at: Optional[datetime]    # When knowledge becomes invalid
    created_at: Optional[datetime]    # When artifact was created
    source: str = "loongflow_pes"     # Source identifier
    metadata: Dict[str, Any]          # Type-specific metadata
    entities: List[str]               # Entities mentioned
    relationships: List[Dict]         # Relationships
    confidence: float = 1.0           # Confidence score
    group_id: Optional[str]           # Related artifacts
```

## Artifact Relationships

Artifacts from the same PES run are related through:

1. **Temporal Sequence:** Plan → Execute → Summarize
2. **Group ID:** All artifacts from same run share `group_id`
3. **Problem Link:** All artifacts reference same problem
4. **Causality:** Execution patterns depend on planning strategy

Example relationship graph:
```
(planning_strategy:1)
       │
       ├─→ (execution_pattern:1) ─┐
       │                          │
       └─→ (reflection_insight:1) ├─→ (optimized_solution:1)
                                  │
                    (evolutionary_lineage:1)
```

## Usage Patterns

### Pattern 1: Strategy Retrieval
```python
# For a new problem, query successful strategies
strategies = await extractor.query_planning_strategies(
    problem_type="portfolio_optimization",
    limit=5,
    min_success_rate=0.8
)

# Apply top strategy to new problem
best_strategy = strategies[0]
```

### Pattern 2: Efficiency Analysis
```python
# Get efficiency metrics for problem type
metrics = await extractor.get_efficiency_metrics(
    problem_type="scientific"
)

# Use metrics to estimate resources needed
estimated_evals = metrics["avg_evaluations_saved"]
estimated_time = estimated_evals * avg_time_per_eval
```

### Pattern 3: Cross-Problem Learning
```python
# Find insights that worked across multiple domains
cross_domain_insights = await ke.query("""
MATCH (a:KnowledgeArtifact {
  artifact_type: 'reflection_insight',
  source: 'loongflow_pes'
})
WHERE size(a.metadata.what_worked) > 3
WITH a.metadata.what_worked as techniques
UNWIND techniques as technique
MATCH (b:KnowledgeArtifact {
  artifact_type: 'reflection_insight',
  source: 'loongflow_pes'
})
WHERE technique IN b.metadata.what_worked
RETURN technique, COUNT(*) as frequency
ORDER BY frequency DESC
LIMIT 10
""")
```

## Confidence Calibration

Artifact confidence scores should be calibrated based on:

1. **Single Run:** Base confidence (0.7-0.9)
2. **Validation Runs:** +0.05 per successful validation
3. **Cross-Problem Success:** +0.1 if works across 3+ problem types
4. **Age Decay:** -0.01 per month (knowledge freshness)
5. **Contradictions:** -0.2 if contradicted by newer artifacts

## Maintenance

### Artifact Validation
```python
# Periodically validate artifacts
async def validate_artifacts():
    old_artifacts = await ke.query("""
        MATCH (a:KnowledgeArtifact {source: 'loongflow_pes'})
        WHERE a.created_at < datetime() - duration('P30D')
        RETURN a
    """)

    for artifact in old_artifacts:
        # Re-test on current problems
        new_fitness = test_solution(artifact.content)
        if new_fitness < artifact.metadata.fitness - 0.1:
            # Mark as invalid
            artifact.invalid_at = datetime.now(timezone.utc)
            await ke.update_artifact(artifact)
```

### Artifact Pruning
```python
# Remove low-confidence, old artifacts
async def prune_artifacts():
    await ke.query("""
        MATCH (a:KnowledgeArtifact {source: 'loongflow_pes'})
        WHERE a.confidence < 0.5
        AND a.created_at < datetime() - duration('P90D')
        DELETE a
    """)
```

## Integration with OpenEvolve

LoongFlow artifacts complement OpenEvolve artifacts:

| LoongFlow Artifact | OpenEvolve Equivalent | Complement |
|---|---|---|
| PlanningStrategy | WorkflowStrategy | Evolutionary vs. Deterministic |
| ExecutionPattern | ExecutionTrace | Early stopping vs. Full execution |
| ReflectionInsight | CritiqueInsight | Algorithm vs. Team insights |
| EvolutionaryLineage | SolutionAncestry | Mutation vs. Human revision |
| OptimizedSolution | BestSolution | Code vs. Workflow |

Both systems can be queried together:
```python
# Get all knowledge for portfolio optimization
loongflow_artifacts = await loongflow_extractor.query_planning_strategies(
    problem_type="portfolio_optimization"
)

openevolve_artifacts = await openevolve_integration.query_workflows(
    domain="finance",
    problem_type="portfolio_optimization"
)

# Merge and rank by confidence
all_knowledge = loongflow_artifacts + openevolve_artifacts
all_knowledge.sort(key=lambda x: x.confidence, reverse=True)
```

## Best Practices

1. **Always Validate:** Test strategies on similar problems before production use
2. **Monitor Confidence:** Track confidence scores and re-validate low-confidence artifacts
3. **Temporal Awareness:** Use `valid_at` and `invalid_at` for point-in-time queries
4. **Cross-Reference:** Query both LoongFlow and OpenEvolve for comprehensive knowledge
5. **Incremental Learning:** Store artifacts from each run to build knowledge base over time
6. **Metadata Enrichment:** Add custom metadata for domain-specific filtering
7. **Regular Maintenance:** Prune outdated artifacts and validate high-value ones

## Future Enhancements

Potential improvements to the artifact system:

1. **Artifact Versioning:** Track evolution of strategies/solutions over time
2. **Cross-Problem Transfer Learning:** Identify transferable patterns across domains
3. **Automated Validation:** Periodically re-test solutions and update confidence
4. **Artifact Fusion:** Combine insights from multiple runs into meta-artifacts
5. **Causal Inference:** Understand why certain strategies work (not just that they work)
6. **Multi-Objective Optimization:** Capture trade-offs between competing objectives
7. **Ensemble Strategies:** Combine multiple strategies for better performance

---

**Document Version:** 1.0
**Last Updated:** 2026-01-30
**Maintained By:** OpenEvolve Knowledge Engine Team
