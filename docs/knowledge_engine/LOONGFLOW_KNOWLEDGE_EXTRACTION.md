# LoongFlow Knowledge Extraction System

## Overview

The LoongFlow Knowledge Extraction System bridges LoongFlow's Plan-Execute-Summarize (PES) evolutionary algorithm with the Knowledge Engine, enabling cross-system learning and knowledge transfer.

**Purpose**: Extract, store, and query knowledge artifacts from LoongFlow PES runs to enable learning across evolutionary optimization sessions.

**Location**: `knowledge_engine/integrations/loongflow_integration.py`

## Architecture

```
LoongFlow PES Run
    ↓
[Plan Phase] → [Execute Phase] → [Summary Phase]
    ↓                ↓                ↓
┌───────────────────────────────────────────────┐
│   LoongFlowKnowledgeExtractor                 │
│   ├── Extract Planning Strategies             │
│   ├── Extract Execution Patterns              │
│   ├── Extract Reflection Insights             │
│   ├── Extract Evolutionary Lineage            │
│   └── Extract Optimized Solutions             │
└───────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────┐
│   Knowledge Artifacts (5 types)                │
│   ├── PlanningStrategyArtifact                │
│   ├── ExecutionPatternArtifact                │
│   ├── ReflectionInsightArtifact               │
│   ├── EvolutionaryLineageArtifact             │
│   └── OptimizedSolutionArtifact               │
└───────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────┐
│   Storage Backends                             │
│   ├── Graphiti (Temporal Knowledge Graph)      │
│   ├── Qdrant (Vector Embeddings)               │
│   ├── Neo4j (Entity Relationships)            │
│   └── MongoDB (Document Archive)               │
└───────────────────────────────────────────────┘
```

## Knowledge Artifacts

### 1. Planning Strategy Artifact

**Captures**: High-level strategic approach from planning phase

**Content**:
- Strategy description
- Reasoning chain
- Action steps
- Expected deliverables
- Success criteria

**Example**:
```python
{
    "strategy": "Use gradient descent with momentum",
    "reasoning": "Momentum helps escape local optima",
    "action_steps": ["Initialize weights", "Compute gradients", "Update"],
    "success_criteria": {"convergence": 0.001},
    "planning_approach": "gradient_based"
}
```

**Use Cases**:
- Reuse successful strategies for similar problems
- Learn which approaches work for specific domains
- Guide planning in future evolutionary runs

### 2. Execution Pattern Artifact

**Captures**: Efficiency metrics and execution patterns

**Content**:
- Early stopping events
- Convergence rate
- Evaluations performed
- Time saved through early stopping
- Parameter tuning trends

**Example**:
```python
{
    "early_stopping_events": [15, 25, 35],
    "convergence_rate": 0.95,
    "iterations_to_best": 25,
    "total_evaluations": 40,
    "efficiency_gain": 0.60,  # 60% fewer evaluations
    "parameter_tuning": {"learning_rate": 0.01, "momentum": 0.9}
}
```

**Use Cases**:
- Validate LoongFlow's 60% efficiency claim
- Identify when early stopping is most effective
- Optimize evaluation budgets

### 3. Reflection Insight Artifact

**Captures**: Learnings from summary/reflection phase

**Content**:
- What worked
- What failed
- Insights and recommendations
- Adaptation patterns

**Example**:
```python
{
    "insights": "Momentum significantly improved convergence",
    "what_worked": ["Gradient descent with momentum", "Adaptive learning rate"],
    "what_failed": ["Pure gradient descent", "Fixed learning rate"],
    "recommendations": ["Always use momentum", "Adapt learning rate"],
    "adaptation_patterns": ["Learning rate decay"]
}
```

**Use Cases**:
- Avoid repeating mistakes
- Accelerate learning curve
- Build domain-specific knowledge

### 4. Evolutionary Lineage Artifact

**Captures**: Evolutionary tree structure and ancestry

**Content**:
- Generation count
- Branching factor
- Parent-child relationships
- Solution provenance

**Example**:
```python
{
    "generations": 10,
    "branching_factor": 2.5,
    "total_mutations": 20,
    "best_path": ["gen_0", "gen_3", "gen_7", "gen_10"],
    "ancestry_tree": {"root": "gen_0", "branches": ["gen_1", "gen_2"]}
}
```

**Use Cases**:
- Track solution evolution over time
- Identify successful evolutionary paths
- Understand population dynamics

### 5. Optimized Solution Artifact

**Captures**: Final best solution found

**Content**:
- Solution code/representation
- Fitness score
- Iteration found
- Improvement over baseline

**Example**:
```python
{
    "solution": "def solve(): return 42",
    "fitness": 0.95,
    "iteration_found": 25,
    "improvement_over_baseline": 0.15,
    "solution_params": {"learning_rate": 0.01, "epochs": 100}
}
```

**Use Cases**:
- Retrieve best solutions for similar problems
- Initialize populations with proven solutions
- Benchmark solution quality

## Usage

### Basic Extraction

```python
from knowledge_engine.integrations.loongflow_integration import (
    LoongFlowKnowledgeExtractor,
    PESRunResults,
)

# Initialize extractor with Knowledge Engine
extractor = LoongFlowKnowledgeExtractor(knowledge_engine=ke)

# Define PES run results
pes_results = PESRunResults(
    plan={
        "strategy": "Use genetic algorithm",
        "reasoning": "Global search needed",
        "action_steps": ["Initialize", "Evolve", "Select"],
        "success_criteria": {"fitness": 0.9},
        "approach": "evolutionary",
        "success_rate": 0.85,
    },
    execution={
        "early_stops": [10, 20],
        "convergence_rate": 0.9,
        "iterations_to_best": 20,
        "total_evaluations": 25,
        "baseline_evaluations": 100,
    },
    summary={
        "insights": "Crossover operator was effective",
        "what_worked": ["Two-point crossover"],
        "what_failed": ["Single-point crossover"],
        "recommendations": ["Use two-point crossover"],
    },
    evolutionary_tree={
        "generations": 5,
        "avg_branching": 2.0,
        "total_mutations": 10,
        "best_path": ["gen_0", "gen_2", "gen_5"],
    },
    best_solution={
        "code": "def optimize(): return best",
        "fitness": 0.92,
        "iteration": 20,
        "improvement": 0.2,
    },
)

# Extract artifacts
artifacts = await extractor.extract_from_pes_run(
    pes_run_results=pes_results,
    problem="Optimize portfolio allocation",
    problem_type="financial_optimization",
    domain="finance",  # Optional - auto-detected if not provided
    run_id="run_001",  # Optional - auto-generated if not provided
)

# Artifacts automatically stored in Knowledge Engine backends
print(f"Extracted {len(artifacts)} artifacts")
```

### Querying Artifacts

```python
# Query successful planning strategies for similar problems
strategies = await extractor.query_planning_strategies(
    problem_type="portfolio_optimization",
    domain="finance",
    limit=10,
    min_success_rate=0.7,
)

for strategy in strategies:
    print(f"Strategy: {strategy['content']['strategy']}")
    print(f"Success Rate: {strategy['metadata']['success_rate']}")
```

### Efficiency Metrics

```python
# Get efficiency metrics for PES on this problem type
metrics = await extractor.get_efficiency_metrics(
    problem_type="portfolio_optimization",
    domain="finance",
)

print(f"Average Efficiency Gain: {metrics['avg_efficiency_gain']}")
print(f"Average Evaluations Saved: {metrics['avg_evaluations_saved']}")
print(f"Total Runs Analyzed: {metrics['total_runs']}")
```

## Domain Detection

The extractor auto-detects domains from problem descriptions:

| Domain | Keywords |
|--------|----------|
| **finance** | portfolio, trading, investment, financial, stock, market |
| **trading** | trading, algorithm, strategy, buy, sell |
| **science** | experiment, scientific, research, hypothesis, lab |
| **mathematics** | equation, prove, theorem, mathematical, optimization |
| **machine_learning** | model, training, neural, ml, deep learning, classifier |
| **engineering** | design, structural, mechanical, civil, engineering |

```python
# Auto-detection examples
domain = extractor._detect_domain(
    "Optimize portfolio allocation for stocks",
    "financial"
)
# Returns: "finance"

domain = extractor._detect_domain(
    "Train neural network for classification",
    "ml_training"
)
# Returns: "machine_learning"
```

## Storage Backend Integration

### Graphiti (Temporal Knowledge Graph)

```python
# Artifacts stored as episodes with temporal metadata
episode = artifact.to_graphiti_episode()

await graphiti.add_episode(
    name=f"{artifact.artifact_type}_{run_id}",
    episode_body=episode,
    reference_datetime=artifact.valid_at,
    valid_from=artifact.valid_at,
)
```

**Benefits**:
- Point-in-time queries
- Temporal provenance
- Contradiction detection
- Timeline reconstruction

### Qdrant (Vector Store)

```python
# Artifacts stored with embeddings for semantic search
payload = artifact.to_qdrant_payload()

await qdrant.upsert(
    collection_name=f"loongflow_{artifact.domain}",
    points=[{
        "id": f"{artifact.artifact_type}_{run_id}",
        "vector": embedding,
        "payload": payload,
    }],
)
```

**Benefits**:
- Semantic similarity search
- Find similar strategies
- Cross-domain pattern matching

### Neo4j (Graph Database)

```python
# Artifacts stored as entities with relationships
query = f"""
MERGE (a:Artifact {{id: '{artifact.artifact_type}_{run_id}'}})
SET a += $artifact_data
MERGE (t:Target {{name: '{problem}'}})
MERGE (a)-[:{rel_type}]->(t)
"""

await neo4j.run(query, artifact_data=artifact.to_dict())
```

**Benefits**:
- Entity-relationship modeling
- Graph traversals
- Lineage tracking

### MongoDB (Document Archive)

```python
# Artifacts stored as raw documents
document = artifact.to_dict()
document["_id"] = f"{artifact.artifact_type}_{run_id}"

await mongodb.insert_one(document)
```

**Benefits**:
- Raw archival
- Flexible querying
- Full-text search

## Advanced Usage

### Extracting Specific Artifacts

```python
# Extract only planning strategies
planning_artifact = await extractor.extract_planning_strategies(
    plan=plan_data,
    problem=problem,
    problem_type=problem_type,
    domain=domain,
    timestamp=datetime.now(timezone.utc),
    run_id=run_id,
)

# Extract only execution patterns
execution_artifact = await extractor.extract_execution_patterns(
    execution=execution_data,
    problem=problem,
    problem_type=problem_type,
    domain=domain,
    timestamp=datetime.now(timezone.utc),
    run_id=run_id,
)
```

### Custom Domain Mapping

```python
# Override auto-detection
artifacts = await extractor.extract_from_pes_run(
    pes_run_results=pes_results,
    problem="Custom problem",
    domain="custom_domain",  # Force specific domain
)
```

### Statistics Tracking

```python
# Get extraction statistics
stats = extractor.get_extraction_stats()
print(stats)
# {
#     "planning_strategy": 15,
#     "execution_pattern": 15,
#     "reflection_insight": 15,
#     "evolutionary_lineage": 15,
#     "optimized_solution": 15
# }

# Reset statistics
extractor.reset_stats()
```

## Error Handling

### Graceful Degradation

The extractor handles missing components gracefully:

```python
# No Knowledge Engine provided
extractor = LoongFlowKnowledgeExtractor(knowledge_engine=None)

# Still extracts artifacts, just doesn't store them
artifacts = await extractor.extract_from_pes_run(...)
# Returns: List of artifacts (not persisted)
```

### Invalid Input Handling

```python
# Invalid input type
artifacts = await extractor.extract_from_pes_run(
    pes_run_results="invalid",  # Wrong type
    problem="Test",
    problem_type="test",
)
# Returns: [] (empty list)
```

### Missing Data Handling

```python
# Incomplete PES results
incomplete_pes = PESRunResults(
    plan={"strategy": "Test"},
    execution={},  # Empty
    summary={},   # Empty
    evolutionary_tree={},  # Empty
    best_solution={},  # Empty
)

# Extracts only from non-empty sections
artifacts = await extractor.extract_from_pes_run(incomplete_pes, ...)
# Returns: List with only planning strategy artifact
```

## Testing

### Running Tests

```bash
# Run all LoongFlow integration tests
pytest tests/knowledge_engine/test_loongflow_integration.py -v

# Run specific test class
pytest tests/knowledge_engine/test_loongflow_integration.py::TestLoongFlowKnowledgeExtractor -v

# Run with coverage
pytest tests/knowledge_engine/test_loongflow_integration.py --cov=knowledge_engine.integrations.loongflow_integration
```

### Test Coverage

The test suite includes:

1. **PESRunResults Tests** (3 tests)
   - `test_to_dict` - Test dictionary conversion
   - `test_from_dict` - Test creation from dictionary
   - Serialization/deserialization

2. **KnowledgeArtifact Tests** (3 tests)
   - `test_to_dict` - Test dictionary conversion
   - `test_to_graphiti_episode` - Test Graphiti format
   - `test_to_qdrant_payload` - Test Qdrant format

3. **Extraction Tests** (8 tests)
   - `test_extract_from_pes_run` - Complete extraction
   - `test_extract_planning_strategies` - Planning extraction
   - `test_extract_execution_patterns` - Execution extraction
   - `test_extract_reflection_insights` - Reflection extraction
   - `test_extract_evolutionary_lineage` - Lineage extraction
   - `test_extract_optimized_solutions` - Solution extraction
   - `test_extract_with_dict_input` - Dict input handling
   - `test_extract_with_missing_data` - Missing data handling

4. **Domain Detection Tests** (4 tests)
   - `test_detect_domain` - Domain auto-detection
   - Finance, ML, Science, General domains

5. **Statistics Tests** (2 tests)
   - `test_get_extraction_stats` - Statistics retrieval
   - `test_reset_stats` - Statistics reset

6. **Storage Backend Tests** (4 tests)
   - `test_store_in_graphiti` - Graphiti storage
   - `test_store_in_neo4j` - Neo4j storage
   - `test_store_in_mongodb` - MongoDB storage
   - `test_full_extraction_with_storage` - End-to-end storage

7. **Query Method Tests** (2 tests)
   - `test_query_planning_strategies` - Strategy querying
   - `test_get_efficiency_metrics` - Efficiency metrics

8. **Edge Case Tests** (5 tests)
   - `test_initialization_without_ke` - No KE handling
   - `test_extract_without_ke` - Extraction without KE
   - `test_extract_with_invalid_input` - Invalid input
   - `test_extract_with_empty_plan` - Empty data
   - `test_domain_detection_with_empty_strings` - Empty strings

9. **Integration Tests** (1 test)
   - `test_end_to_end_extraction_and_storage` - Full workflow

**Total**: 32 comprehensive tests

## Performance Considerations

### Extraction Performance

| Operation | Time (ms) | Notes |
|-----------|-----------|-------|
| Single artifact extraction | < 10 | In-memory processing |
| Full PES run extraction (5 artifacts) | < 50 | All artifacts |
| Storage in all backends | < 100 | Depends on backend latency |
| End-to-end (extract + store) | < 200 | Complete workflow |

### Optimization Tips

1. **Batch Processing**: Extract multiple runs in parallel
```python
import asyncio

runs = [run1, run2, run3, ...]
artifacts = await asyncio.gather(*[
    extractor.extract_from_pes_run(run, ...)
    for run in runs
])
```

2. **Selective Extraction**: Extract only needed artifacts
```python
# Only extract planning strategies (faster)
planning_only = await extractor.extract_planning_strategies(...)
```

3. **Disable Storage**: Extract without persistence for testing
```python
extractor = LoongFlowKnowledgeExtractor(knowledge_engine=None)
```

## Integration with Other Systems

### OpenEvolve Integration

```python
from knowledge_engine.integrations.loongflow_integration import LoongFlowKnowledgeExtractor
from knowledge_engine.integrations.openevolve_integration import OpenEvolveIntegration

# Initialize both extractors
loongflow_extractor = LoongFlowKnowledgeExtractor(knowledge_engine=ke)
openevolve_extractor = OpenEvolveIntegration(knowledge_engine=ke)

# Extract from both systems
lf_artifacts = await loongflow_extractor.extract_from_pes_run(...)
oe_artifacts = await openevolve_extractor.extract_from_workflow(...)

# Knowledge Engine now has unified knowledge
```

### Unified Evolution Integration (Phase 2)

```python
from knowledge_engine.integrations.unified_evolution_integration import (
    UnifiedEvolutionKnowledgeExtractor,
)

unified_extractor = UnifiedEvolutionKnowledgeExtractor(knowledge_engine=ke)

# Extract from both systems and compare
artifacts = await unified_extractor.extract_from_both(
    run_id="unified_run_001",
    openevolve_results=oe_results,
    loongflow_results=lf_results,
    metadata={"domain": "finance"},
)

# Get performance comparison
comparison = artifacts["comparison"]
# {
#     "winner": "loongflow",
#     "improvement": "60%",
#     "reason": "Fewer evaluations with comparable quality"
# }
```

## Troubleshooting

### Common Issues

**Issue**: Artifacts not being stored

**Solution**: Check Knowledge Engine backends are initialized
```python
# Check backend status
extractor = LoongFlowKnowledgeExtractor(knowledge_engine=ke)
print(f"Graphiti: {extractor.graphiti is not None}")
print(f"Qdrant: {extractor.qdrant is not None}")
print(f"Neo4j: {extractor.neo4j is not None}")
```

**Issue**: Domain detection returning "general"

**Solution**: Check problem description includes relevant keywords
```python
# Good: Specific domain keywords
problem = "Optimize portfolio allocation for stocks"

# Bad: Too generic
problem = "Solve this problem"
```

**Issue**: Low confidence scores

**Solution**: Ensure PES results include quality metrics
```python
# Include success_rate in plan
plan["success_rate"] = 0.85

# Include fitness in best_solution
solution["fitness"] = 0.95

# Include what_worked/what_failed in summary
summary["what_worked"] = ["technique_1", "technique_2"]
summary["what_failed"] = ["technique_3"]
```

## Best Practices

### 1. Always Provide Problem Context

```python
# Good: Specific problem description
artifacts = await extractor.extract_from_pes_run(
    pes_run_results=pes_results,
    problem="Optimize portfolio allocation for tech stocks with max Sharpe ratio",
    problem_type="portfolio_optimization",
)

# Bad: Generic description
artifacts = await extractor.extract_from_pes_run(
    pes_run_results=pes_results,
    problem="Optimize something",
    problem_type="generic",
)
```

### 2. Use Consistent Problem Types

```python
# Use consistent problem_type values
PROBLEM_TYPES = [
    "portfolio_optimization",
    "trading_strategy",
    "neural_network_training",
    "algorithm_design",
    "parameter_tuning",
]
```

### 3. Validate PES Results Before Extraction

```python
def validate_pes_results(pes_results):
    """Validate PES results have required fields"""
    required = ["plan", "execution", "summary", "evolutionary_tree", "best_solution"]
    return all(hasattr(pes_results, field) or field in pes_results for field in required)

if validate_pes_results(pes_results):
    artifacts = await extractor.extract_from_pes_run(...)
else:
    logger.error("Invalid PES results")
```

### 4. Monitor Extraction Statistics

```python
# Regularly check extraction stats
stats = extractor.get_extraction_stats()
total_extracted = sum(stats.values())

if total_extracted % 100 == 0:
    logger.info(f"Extracted {total_extracted} artifacts so far")
    # Log breakdown by type
    for artifact_type, count in stats.items():
        logger.info(f"  {artifact_type}: {count}")
```

## Future Enhancements

### Phase 2 Features (Planned)

1. **Cross-System Learning**
   - Transfer learning between OpenEvolve and LoongFlow
   - Strategy recommendation based on historical performance
   - Unified knowledge querying

2. **Advanced Analytics**
   - Evolution trend analysis
   - Success pattern mining
   - Failure mode identification

3. **Real-Time Updates**
   - Live artifact extraction during PES runs
   - Streaming knowledge updates
   - Incremental learning

4. **Multi-Modal Artifacts**
   - Image-based solution representations
   - Code structure analysis
   - Execution trace visualization

## References

- **LoongFlow PES Documentation**: `docs/knowledge_engine/LOONGFLOW_PES_FORENSIC_ANALYSIS.md`
- **Integration Roadmap**: `docs/knowledge_engine/COMPREHENSIVE_INTEGRATION_ROADMAP.md`
- **Knowledge Engine Docs**: `docs/knowledge_engine/comprehensive_documentation.md`
- **Source Code**: `knowledge_engine/integrations/loongflow_integration.py`
- **Tests**: `tests/knowledge_engine/test_loongflow_integration.py`

## Support

For questions or issues:
1. Check troubleshooting section above
2. Review test files for usage examples
3. Examine PES forensic analysis for LoongFlow details
4. Consult Knowledge Engine documentation

---

**Version**: 1.0
**Last Updated**: January 30, 2026
**Status**: Production Ready
