# Adaptive MDAP - Quick Start Guide

> **Version**: 1.0.0  
> **Date**: February 2, 2026  
> **Status**: Production Ready

---

## 5-Minute Quick Start

### 1. Verify Installation

```bash
# Check wiring
python check_wiring_complete.py

# Expected output: 40/40 Integration Points passing
```

### 2. Run Demo

```bash
# Run adaptive demo
python demo_mdap_maker.py adaptive

# Or run all demos
python demo_mdap_maker.py all
```

### 3. Use CLI

```bash
# Classify a problem
openevolve adaptive classify \
  --description "Implement user authentication with JWT tokens" \
  --domain security

# Allocate resources
openevolve adaptive allocate 0.65 --profile balanced

# Check status
openevolve adaptive status
```

---

## Python API Quick Start

### Basic Usage

```python
from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
from adaptive_mdap.core.types import SubProblem

# Create a sub-problem
sp = SubProblem(
    id="auth-001",
    description="Implement secure user authentication",
    domain="security",
    depth=2,
    dependencies=[],
    metadata={}
)

# Classify complexity
classifier = TaskComplexityClassifier()
score = classifier.compute_complexity(sp)
print(f"Complexity: {score.overall_score:.3f}")

# Allocate resources
allocator = AdaptiveMDAPAllocator()
config = allocator.allocate_resources(score.overall_score)
print(f"Strategy: {config.strategy.value}")
print(f"Agents: {config.n_agents}")
```

### Workflow Integration

```python
from workflow_engine import get_adaptive_allocation_for_subproblem

# Get allocation for sub-problem
config = get_adaptive_allocation_for_subproblem(sub_problem, workflow_state)

print(f"Strategy: {config['strategy']}")
print(f"Agents: {config['n_agents']}")
print(f"K-Ahead: {config['k_ahead']}")
```

### Team Assignment

```python
from team_assignment_engine import TeamAssignmentEngine

engine = TeamAssignmentEngine(team_manager)
assignment = engine.assign_teams_with_complexity(sub_problem, available_teams)

# Access complexity metadata
complexity = assignment.metadata.get('complexity_score')
recommended_size = assignment.metadata.get('recommended_team_size')
```

### Quality Assessment

```python
from quality_assessment import QualityAssessmentEngine

engine = QualityAssessmentEngine()
result = engine.assess_quality_with_complexity(
    content=content,
    content_type="code",
    use_adaptive_thresholds=True
)

# Access complexity-adjusted score
complexity = result.assessment_metadata.get('complexity_score')
adjusted_score = result.assessment_metadata.get('adjusted_score')
```

---

## Configuration

### Environment Variables

```bash
# Add to .env file
ADAPTIVE_MDAP_ENABLED=true
ADAPTIVE_MDAP_EMBEDDING_MODEL=all-MiniLM-L6-v2
ADAPTIVE_MDAP_ENABLE_LEARNING=false
ADAPTIVE_MDAP_ENABLE_CONTEXT_AWARE=false
```

### Code Configuration

```python
from adaptive_mdap.integrations.workflow_engine_integration import (
    AdaptiveWorkflowIntegration,
    AdaptiveWorkflowConfig
)

config = AdaptiveWorkflowConfig(
    enabled=True,
    enable_learning=False,
    enable_context_aware=False,
    default_profile="balanced"
)

integration = AdaptiveWorkflowIntegration(config)
```

### Evolution Configuration

```python
from evolution import EvolutionConfiguration

config = EvolutionConfiguration(
    enable_adaptive_mdap=True,
    adaptive_mdap_profile="balanced",
    adaptive_mdap_learning=False,
    adaptive_mdap_context_aware=False
)
```

---

## Common Use Cases

### Use Case 1: Simple Task

```python
# Task: "Add two numbers"
sp = SubProblem(
    id="simple-001",
    description="Add two numbers",
    domain="mathematics",
    depth=1,
    dependencies=[],
    metadata={}
)

# Expected: DIRECT strategy (1 agent)
```

### Use Case 2: Medium Complexity

```python
# Task: "Implement BST"
sp = SubProblem(
    id="bst-001",
    description="Implement a binary search tree with insertion and deletion",
    domain="computer_science",
    depth=2,
    dependencies=[],
    metadata={}
)

# Expected: MDAP_MEDIUM strategy (5 agents, k=1)
```

### Use Case 3: High Complexity

```python
# Task: "Design distributed system"
sp = SubProblem(
    id="distributed-001",
    description="Design a distributed consensus protocol with Byzantine fault tolerance",
    domain="distributed_systems",
    depth=4,
    dependencies=[],
    metadata={}
)

# Expected: MAKER_ULTRA strategy (7+ agents, k=3)
```

---

## Monitoring

### Record Metrics

```python
from monitoring_system import (
    record_adaptive_classification,
    record_adaptive_allocation
)

# Record classification
record_adaptive_classification(
    subproblem_id="sp-001",
    complexity_score=0.65,
    latency_ms=45.0,
    success=True
)

# Record allocation
record_adaptive_allocation(
    subproblem_id="sp-001",
    strategy="MDAP_MEDIUM",
    n_agents=5,
    k_ahead=1,
    latency_ms=0.5,
    success=True
)
```

### Get Metrics

```python
from monitoring_system import get_adaptive_metrics

metrics = get_adaptive_metrics()
print(f"Classifications: {metrics['classifications']}")
print(f"Allocations: {metrics['allocations']}")
```

---

## Troubleshooting

### Issue: Adaptive MDAP not available

```
[WARNING] Adaptive MDAP not available - using standard allocation
```

**Solution**: Install dependencies:
```bash
pip install -e .[adaptive]
```

### Issue: Classification timeout

```
[ERROR] Classification timeout after 5000ms
```

**Solution**: Check embedding model cache or reduce text length.

### Issue: High memory usage

```
[WARNING] High memory usage detected
```

**Solution**: Clear embedding cache:
```python
from adaptive_mdap.utils.cache import clear_cache
clear_cache()
```

---

## Best Practices

1. **Start with Balanced Profile**: Use `balanced` profile for initial deployment
2. **Monitor Metrics**: Track classification and allocation latencies
3. **Use Learning Mode**: Enable `enable_learning` after collecting data
4. **Clear Cache**: Clear cache periodically to manage memory
5. **Adjust Thresholds**: Customize thresholds based on your domain

---

## Next Steps

- Read full guide: `ADAPTIVE_MDAP_INTEGRATION_GUIDE.md`
- Check examples: `demo_mdap_maker.py`
- Review API: `docs/adaptive_mdap/API_REFERENCE.md`

---

**Ready to use Adaptive MDAP!** 🚀
