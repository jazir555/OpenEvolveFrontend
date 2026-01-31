# Memory Fusion System Documentation

## Overview

The Memory Fusion System combines evolutionary knowledge from both **OpenEvolve** (Quality Diversity, MAP-Elites) and **LoongFlow** (Plan-Execute-Summarize) systems into a unified knowledge graph that enables cross-system learning and knowledge transfer.

## What is Memory Fusion?

### OpenEvolve Memory
- **Population archives** (MAP-Elites grid)
- **Evolutionary lineage** (parent-child relationships)
- **Fitness history**
- **Elite solutions**
- **Diversity metrics**
- **Convergence data**

### LoongFlow Memory
- **Planning strategies** (strategic reasoning)
- **Execution patterns** (early stopping, efficiency)
- **Reflection insights** (what worked/failed)
- **Summarization episodes** (learning iterations)
- **PES lineage** (plan evolution)
- **Efficiency metrics** (60% sample efficiency)

### Fusion Goal
Combine both into unified knowledge that:
1. Identifies **complementary patterns** (where systems help each other)
2. Detects and **resolves conflicts** (where systems disagree)
3. Enables **cross-pollination** of insights
4. Supports **temporal queries** across systems
5. Creates **unified evolutionary lineage**

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              MEMORY FUSION ENGINE                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  OpenEvolve Memory          LoongFlow Memory            │
│  ├─ Population Archive      ├─ Planning Strategies     │
│  ├─ Evolutionary Lineage    ├─ Execution Patterns      │
│  ├─ Fitness History         ├─ Reflection Insights     │
│  ├─ Elite Solutions         └─ PES Lineage             │
│  └─ Diversity Metrics                                      │
│         │                         │                     │
│         └────────┬────────────────┘                     │
│                  ▼                                      │
│    ┌─────────────────────────┐                        │
│    │  FUSION OPERATIONS      │                        │
│    ├─────────────────────────┤                        │
│    │ 1. Detect Patterns      │                        │
│    │ 2. Detect Conflicts     │                        │
│    │ 3. Resolve Conflicts    │                        │
│    │ 4. Create Lineage       │                        │
│    │ 5. Build Knowledge Graph│                       │
│    │ 6. Find Pollination     │                        │
│    └─────────────────────────┘                        │
│                  │                                      │
│                  ▼                                      │
│    ┌─────────────────────────┐                        │
│    │   FUSED MEMORY OUTPUT   │                        │
│    ├─────────────────────────┤                        │
│    │ • Complementary Patterns│                        │
│    │ • Conflict Resolutions  │                        │
│    │ • Unified Lineage       │                        │
│    │ • Knowledge Graph       │                        │
│    │ • Pollination Opps      │                        │
│    └─────────────────────────┘                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. EvolutionaryMemoryFusion

Main engine orchestrating the fusion process.

```python
from knowledge_engine.integrations.memory_fusion import (
    EvolutionaryMemoryFusion,
    OpenEvolveMemory,
    LoongFlowMemory,
)

# Create fusion engine
fusion = EvolutionaryMemoryFusion(knowledge_engine=ke)

# Prepare memories
oe_memory = OpenEvolveMemory(
    population_archive={...},
    evolutionary_lineage=[...],
    fitness_history=[...],
    elite_solutions=[...],
    diversity_metrics=[...],
    convergence_data={...},
)

lf_memory = LoongFlowMemory(
    planning_strategies=[...],
    execution_patterns=[...],
    reflection_insights=[...],
    summarization_episodes=[...],
    pes_lineage=[...],
    efficiency_metrics={...},
)

# Fuse memories
fused = await fusion.fuse_memories(
    openevolve_memory=oe_memory,
    loongflow_memory=lf_memory,
    domain="finance",
)

# Access results
patterns = fused.complementary_patterns
conflicts = fused.conflicts
lineage = fused.unified_lineage
opportunities = fused.pollination_opportunities
```

### 2. Complementary Pattern Detection

Finds where systems compensate for each other's weaknesses.

#### Pattern Types

**Exploration + Refinement**
- OpenEvolve explores diverse solutions (QD strength)
- LoongFlow refines them efficiently (PES strength)
- Expected improvement: 40%

**Multi-Objective + Directed**
- OpenEvolve handles multiple objectives
- LoongFlow provides directed search
- Expected improvement: 35%

**Global + Local**
- OpenEvolve performs global search
- LoongFlow optimizes locally
- Expected improvement: 30%

**Diversity + Efficiency**
- OpenEvolve maintains diversity
- LoongFlow achieves 60% sample efficiency
- Expected improvement: 45%

**Adversarial + Planning**
- OpenEvolve tests robustness
- LoongFlow plans improvements
- Expected improvement: 38%

```python
patterns = await fusion.detect_complementary_patterns(fused_memory)

for pattern in patterns:
    print(f"Pattern: {pattern.pattern_type}")
    print(f"  OE contribution: {pattern.openevolve_contribution}")
    print(f"  LF contribution: {pattern.loongflow_contribution}")
    print(f"  Improvement: {pattern.expected_improvement:.1%}")
    print(f"  Confidence: {pattern.confidence:.2%}")
```

### 3. Conflict Detection and Resolution

Finds disagreements and determines how to handle them.

#### Conflict Types

**Parameter Value Conflicts**
- Mutation rate disagreements (e.g., 0.1 vs 0.3)
- Population size differences (e.g., 1000 vs 100)
- Resolution: Hybrid approach

**Strategy Effectiveness Conflicts**
- Selection strategy differences
- Evaluation approach disagreements
- Resolution: Favor system with more evidence

**Convergence Criteria Conflicts**
- Threshold differences
- Stopping condition mismatches
- Resolution: Context-dependent

```python
conflicts = await fusion.detect_conflicts(fused_memory)

for conflict in conflicts:
    print(f"Conflict: {conflict.conflict_type}")
    print(f"  OE position: {conflict.openevolve_position}")
    print(f"  LF position: {conflict.loongflow_position}")
    print(f"  Severity: {conflict.severity}")

# Resolve conflicts
resolutions = await fusion.resolve_conflicts(conflicts)

for resolution in resolutions:
    print(f"Strategy: {resolution.resolution_strategy}")
    print(f"  Reasoning: {resolution.reasoning}")
    print(f"  Confidence: {resolution.confidence:.2%}")
```

#### Resolution Strategies

- **FAVOR_OPENEVOLVE**: OE has more evidence
- **FAVOR_LOONGFLOW**: LF has more evidence
- **HYBRID**: Combine both approaches
- **INVESTIGATE**: Need more data
- **MERGE**: Merge strategies

### 4. Unified Lineage

Combines evolutionary trees from both systems.

```python
lineage = fused.unified_lineage

# Trace solution origin
path = lineage.trace_solution_origin(solution_id="oe_gen_5_indiv_3")
for node in path:
    print(f"{node.source_system}: {node.fitness:.3f}")

# Find common ancestors
ancestors = lineage.find_common_ancestors(
    solution1_id="oe_gen_5",
    solution2_id="lf_iter_3"
)

# Get evolutionary path
evolution_path = lineage.get_evolutionary_path(solution_id="oe_gen_10")
```

#### Cross-System Edges

Identifies where knowledge transferred between systems:

```python
for edge in lineage.cross_system_edges:
    print(f"{edge.from_node} → {edge.to_node}")
    print(f"  Type: {edge.transfer_type}")
    print(f"  Improvement: {edge.improvement:+.2%}")
```

### 5. Cross-System Pollination

Transfers knowledge from one system to another.

#### Pollination Opportunities

```python
opportunities = await fusion.enable_cross_system_pollination(fused_memory)

for opp in opportunities:
    print(f"Opportunity: {opp.opportunity_id}")
    print(f"  {opp.source_system} → {opp.target_system}")
    print(f"  Type: {opp.knowledge_type}")
    print(f"  Expected benefit: {opp.expected_benefit:.1%}")
    print(f"  Complexity: {opp.implementation_complexity}")
    print(f"  Description: {opp.description}")
```

#### Applying Pollination

```python
# Apply an opportunity
result = await fusion.apply_pollination(opportunity)

if result.success:
    print(f"Success! Improvement: {result.actual_improvement:.2%}")
    print(f"Side effects: {result.side_effects}")
else:
    print(f"Failed: {result.error_message}")
```

#### Pollination Types

**Strategy Transfer**
- LoongFlow planning → OpenEvolve mutation
- Expected benefit: 25%

**Solution Transfer**
- OpenEvolve elites → LoongFlow initialization
- Expected benefit: 30%

**Parameter Sharing**
- Cross-system configuration sharing
- Expected benefit: 15%

**Pattern Transfer**
- LoongFlow early stopping → OpenEvolve evaluation
- Expected benefit: 35%

### 6. Temporal Queries

Query knowledge from specific time ranges.

```python
# Query last 24 hours
end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(hours=24)

results = await fusion.temporal_query(
    fused_memory=fused,
    query="fitness improvement",
    time_range=(start_time, end_time),
    limit=10,
)

for result in results:
    print(f"[{result['source_system']}] {result['timestamp']}")
    print(f"  Relevance: {result['relevance']:.2%}")
    print(f"  Data: {result['data']}")
```

### 7. Unified Insights

Generates meta-insights from fused memory.

```python
insights = await fusion.get_unified_insights(fused)

print(f"Domain: {insights.domain}")
print(f"Confidence: {insights.confidence:.2%}")

print("\n=== Best Practices ===")
for practice in insights.best_practices:
    print(f"  • {practice}")

print("\n=== Anti-Patterns ===")
for anti in insights.anti_patterns:
    print(f"  ✗ {anti}")

print("\n=== Configuration Recommendations ===")
print(f"OpenEvolve: {insights.recommended_configurations['openevolve']}")
print(f"LoongFlow: {insights.recommended_configurations['loongflow']}")

print("\n=== Performance Comparison ===")
print(f"OE avg fitness: {insights.overall_performance_comparison['openevolve']['avg_fitness']:.3f}")
print(f"LF avg fitness: {insights.overall_performance_comparison['loongflow']['avg_fitness']:.3f}")
```

## Usage Examples

### Example 1: Finance Domain

```python
from knowledge_engine.integrations.memory_fusion import fuse_and_analyze

# Prepare memories from trading strategy optimization
oe_memory = OpenEvolveMemory(
    elite_solutions=[
        {"solution": "strategy_a", "fitness": 0.85, "sharpe_ratio": 1.5},
        {"solution": "strategy_b", "fitness": 0.82, "sharpe_ratio": 1.4},
    ],
    diversity_metrics=[{"diversity": 0.75, "metric": "risk_profile"}],
    metadata={"mutation_rate": 0.1, "population_size": 1000},
)

lf_memory = LoongFlowMemory(
    planning_strategies=[
        {"strategy": "Risk-adjusted optimization", "success_rate": 0.88}
    ],
    efficiency_metrics={"efficiency_gain": 0.60, "avg_evaluations": 100},
    metadata={"mutation_rate": 0.25, "population_size": 100},
)

# Fuse and analyze
fused, insights = await fuse_and_analyze(
    openevolve_memory=oe_memory,
    loongflow_memory=lf_memory,
    domain="finance",
    knowledge_engine=ke,
)

# Apply best pollination opportunity
if fused.pollination_opportunities:
    best_opp = max(
        fused.pollination_opportunities,
        key=lambda o: o.expected_benefit
    )
    result = await fusion.apply_pollination(best_opp)
    print(f"Applied: {result.actual_improvement:.2%} improvement")
```

### Example 2: Scientific Experiments

```python
# Optimize experimental design
oe_memory = OpenEvolveMemory(
    population_archive={
        f"cell_{i}_{j}": {
            "design": f"design_{i}_{j}",
            "cost": 1000 + i * 100,
            "power": 0.8 + j * 0.02
        }
        for i in range(5) for j in range(5)
    },
    diversity_metrics=[{"diversity": 0.85, "metric": "experimental_space"}],
    convergence_data={"avg_evaluations": 500},  # Expensive!
)

lf_memory = LoongFlowMemory(
    planning_strategies=[
        {"strategy": "Sequential design", "success_rate": 0.92}
    ],
    execution_patterns=[
        {"early_stopped": True, "iteration": 15, "savings": "40%"}
    ],
    efficiency_metrics={"efficiency_gain": 0.60, "avg_evaluations": 200},
)

# Fuse
fusion = EvolutionaryMemoryFusion()
fused = await fusion.fuse_memories(
    openevolve_memory=oe_memory,
    loongflow_memory=lf_memory,
    domain="science",
)

# Check for diversity + efficiency pattern
div_eff_patterns = [
    p for p in fused.complementary_patterns
    if p.pattern_type == "diversity_efficiency"
]

if div_eff_patterns:
    pattern = div_eff_patterns[0]
    print(f"Diversity + Efficiency detected!")
    print(f"  Expected improvement: {pattern.expected_improvement:.1%}")
    print(f"  Description: {pattern.synergy_description}")

# Apply early stopping pollination
early_stop_opps = [
    o for o in fused.pollination_opportunities
    if "early_stop" in o.opportunity_id
]

if early_stop_opps:
    result = await fusion.apply_pollination(early_stop_opps[0])
    print(f"Early stopping applied: {result.actual_improvement:.1%} fewer experiments")
```

### Example 3: Temporal Analysis

```python
# Query progress over time
fusion = EvolutionaryMemoryFusion()
fused = await fusion.fuse_memories(oe_memory, lf_memory)

# Query last week
week_ago = datetime.now(timezone.utc) - timedelta(days=7)
now = datetime.now(timezone.utc)

results = await fusion.temporal_query(
    fused_memory=fused,
    query="fitness progress convergence",
    time_range=(week_ago, now),
    limit=20,
)

print("=== Weekly Progress ===")
for result in results:
    print(f"[{result['timestamp']}] {result['source_system']}")
    print(f"  Relevance: {result['relevance']:.2%}")
    if result['source_system'] == 'openevolve':
        print(f"  Fitness: {result['data'].get('fitness', 0):.3f}")
    elif result['source_system'] == 'loongflow':
        print(f"  Summary: {result['data'].get('summary', 'N/A')}")
```

## Data Structures Reference

### FusedMemory

Main output of fusion process.

```python
@dataclass
class FusedMemory:
    openevolve_component: OpenEvolveMemory
    loongflow_component: LoongFlowMemory
    complementary_patterns: List[ComplementaryPattern]
    conflicts: List[MemoryConflict]
    conflict_resolutions: List[ConflictResolution]
    unified_lineage: UnifiedLineage
    unified_knowledge_graph: KnowledgeGraph
    pollination_opportunities: List[PollinationOpportunity]
    applied_pollinations: List[PollinationResult]
    domain: str
    fusion_timestamp: datetime
    fusion_quality_score: float
```

### ComplementaryPattern

Where systems help each other.

```python
@dataclass
class ComplementaryPattern:
    pattern_type: str  # exploration_refinement, diversity_efficiency, etc.
    openevolve_contribution: str
    loongflow_contribution: str
    synergy_description: str
    expected_improvement: float  # % improvement (0.0 to 1.0)
    confidence: float  # 0.0 to 1.0
    evidence: List[str]
```

### MemoryConflict

Where systems disagree.

```python
@dataclass
class MemoryConflict:
    conflict_type: str  # parameter_value, strategy_effectiveness, etc.
    openevolve_position: str
    loongflow_position: str
    severity: str  # low, medium, high
    description: str
    resolution_suggestion: Optional[str]
```

### ConflictResolution

How to resolve a conflict.

```python
@dataclass
class ConflictResolution:
    conflict: MemoryConflict
    resolution_strategy: str  # favor_openevolve, favor_loongflow, hybrid, etc.
    reasoning: str
    confidence: float
    expected_accuracy: float
    implementation: Optional[Dict[str, Any]]
```

### PollinationOpportunity

Knowledge transfer opportunity.

```python
@dataclass
class PollinationOpportunity:
    opportunity_id: str
    source_system: str
    target_system: str
    knowledge_type: str  # strategy, parameter, solution, pattern
    source_knowledge: Any
    expected_benefit: float  # % improvement
    confidence: float
    implementation_complexity: str  # low, medium, high
    description: str
```

## Best Practices

### 1. When to Use Memory Fusion

**Use when:**
- You have runs from both OpenEvolve and LoongFlow
- You want to understand system synergies
- You need to resolve contradictory results
- You want to transfer knowledge between systems
- You're doing temporal analysis

**Don't use when:**
- You only have data from one system
- You need real-time fusion (use batch processing)
- Memory is extremely limited (fusion requires RAM)

### 2. Fusion Quality

The `fusion_quality_score` indicates how good the fusion is:

- **0.9 - 1.0**: Excellent (many patterns, few conflicts)
- **0.7 - 0.9**: Good (some patterns, resolved conflicts)
- **0.5 - 0.7**: Fair (few patterns, some conflicts)
- **< 0.5**: Poor (no patterns, unresolved conflicts)

### 3. Interpreting Patterns

High-confidence patterns (>0.75) with high expected improvement (>0.30) are most reliable.

### 4. Handling Conflicts

- **Low severity**: Accept hybrid resolution
- **Medium severity**: Check which system has more evidence
- **High severity**: Requires investigation before deciding

### 5. Applying Pollination

Start with **low complexity** opportunities first:
- Direct parameter sharing (safest)
- Strategy transfer (medium risk)
- Architecture changes (highest risk, careful implementation)

## Performance Considerations

### Memory Usage

Approximate memory requirements:
- Base fusion: ~50-100 MB
- With full lineage: ~200-500 MB
- With knowledge graph: ~500 MB - 1 GB

### Processing Time

- Pattern detection: 1-5 seconds
- Conflict resolution: < 1 second
- Lineage creation: 1-3 seconds
- Knowledge graph: 2-5 seconds
- **Total**: ~5-15 seconds per fusion

### Optimization Tips

1. **Limit lineage size**: Only include top N generations
2. **Prune artifacts**: Remove old/low-quality artifacts
3. **Use selective fusion**: Only fuse what you need
4. **Batch processing**: Fuse multiple runs together

## Integration with Knowledge Engine

The fusion system integrates with the Knowledge Engine for persistent storage:

```python
from knowledge_engine import KnowledgeEngine

# Create KE instance
ke = KnowledgeEngine()

# Create fusion with KE
fusion = EvolutionaryMemoryFusion(knowledge_engine=ke)

# Fusion results automatically stored
fused = await fusion.fuse_memories(oe_memory, lf_memory, domain="finance")

# Results now in:
# - MongoDB: fused_memories collection
# - Neo4j: Graph entities and relationships
# - Qdrant: Vector embeddings for semantic search
# - Graphiti: Temporal knowledge graph
```

## Troubleshooting

### Issue: No patterns detected

**Cause**: Insufficient diversity or efficiency data

**Solution**:
- Ensure OpenEvolve has diversity metrics
- Ensure LoongFlow has efficiency metrics
- Check data quality and completeness

### Issue: Too many conflicts

**Cause**: Systems have fundamental disagreements

**Solution**:
- Review conflict resolutions
- Investigate high-severity conflicts
- Consider domain-specific configurations

### Issue: Low fusion quality

**Cause**: Poor data quality or incompatible systems

**Solution**:
- Verify input data quality
- Check temporal alignment
- Ensure both systems solved similar problems

## Future Enhancements

Planned features:

1. **Real-time fusion**: Continuous fusion as runs progress
2. **Multi-domain fusion**: Cross-domain pattern transfer
3. **Explainable fusion**: AI explanations for patterns/conflicts
4. **Auto-pollination**: Automatic application of high-confidence opportunities
5. **Fusion learning**: System learns from fusion history

## References

- OpenEvolve Documentation: [link]
- LoongFlow Documentation: [link]
- Integration Roadmap: `COMPREHENSIVE_INTEGRATION_ROADMAP.md`
- OpenEvolve Analysis: `OPENEVOLVE_EVOLUTIONARY_ALGORITHM_FORENSIC_ANALYSIS.md`
- LoongFlow Analysis: `LOONGFLOW_PES_FORENSIC_ANALYSIS.md`

## API Reference

See `knowledge_engine/integrations/memory_fusion.py` for complete API documentation.

### Main Classes

- `EvolutionaryMemoryFusion`: Main fusion engine
- `FusedMemory`: Fusion output
- `ComplementaryPattern`: System synergy
- `MemoryConflict`: System disagreement
- `ConflictResolution`: Conflict solution
- `UnifiedLineage`: Combined evolutionary tree
- `PollinationOpportunity`: Knowledge transfer chance
- `UnifiedInsights`: Meta-insights

### Convenience Functions

- `create_memory_fusion()`: Create fusion engine
- `fuse_and_analyze()`: Fuse and get insights in one call
