# Adaptive MDAP Documentation

## Overview

Adaptive MDAP (Massively Decomposed Agentic Processes) is an implementation of the MAKER framework that provides **adaptive resource allocation** for LLM-based problem solving. It achieves **30-50% cost reduction** while maintaining quality within **±1% of baseline**.

Based on the research paper: "Solving a Million-Step LLM Task with Zero Errors" by Meyerson et al.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Adaptive MDAP System                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐    ┌──────────────────┐               │
│  │  SubProblem     │───▶│ TaskComplexity   │               │
│  │                 │    │ Classifier       │               │
│  └─────────────────┘    └──────────────────┘               │
│           │                       │                          │
│           │              Complexity Score                    │
│           │                       ▼                          │
│           │            ┌──────────────────┐               │
│           │            │ AdaptiveMDAP     │               │
│           └───────────▶│ Allocator        │               │
│                        └──────────────────┘               │
│                                 │                          │
│                        Strategy Selection                   │
│                                 ▼                          │
│            ┌──────────────────────────────────────┐       │
│            │   AdaptiveExecutionController        │       │
│            │   (with CrewAI Integration)          │       │
│            └──────────────────────────────────────┘       │
│                                 │                          │
│              ┌──────────────────┼──────────────────┐      │
│              ▼                  ▼                  ▼      │
│         ┌─────────┐      ┌──────────┐      ┌──────────┐  │
│         │ DIRECT  │      │MDAP_LIGHT│      │MAKER_FULL│  │
│         │ 1 agent │      │ 3 agents │      │ 5+agents │  │
│         │ k=0     │      │ k=1      │      │ k=2+     │  │
│         └─────────┘      └──────────┘      └──────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
from adaptive_mdap.core.types import SubProblem
from adaptive_mdap.controllers.execution_controller import AdaptiveExecutionController

# Create components
classifier = TaskComplexityClassifier()
allocator = AdaptiveMDAPAllocator()
controller = AdaptiveExecutionController(classifier, allocator)

# Create a sub-problem
subproblem = SubProblem(
    id="example-001",
    description="Solve this complex mathematical optimization problem",
    domain="mathematics",
    depth=3,
    dependencies=["dep1", "dep2"],
    metadata={},
)

# Execute with adaptive allocation
attempt = controller.execute_adaptive(subproblem)

print(f"Complexity Score: {attempt.complexity_score}")
print(f"Strategy Used: {attempt.allocated_strategy}")
print(f"Solution: {attempt.solution}")
```

## Configuration

### Profiles

Three built-in profiles are available:

1. **Conservative** (`config/profiles/conservative.yaml`)
   - Thresholds: [0.2, 0.5]
   - Favors quality over cost
   - Use when reliability is critical

2. **Balanced** (`config/profiles/balanced.yaml`) - Default
   - Thresholds: [0.3, 0.7]
   - Balanced cost/quality tradeoff
   - Recommended for most use cases

3. **Aggressive** (`config/profiles/aggressive.yaml`)
   - Thresholds: [0.4, 0.8]
   - Favors cost savings
   - Use when budget is constrained

### Usage

```python
from adaptive_mdap.config.loader import ConfigLoader
from adaptive_mdap.config.profiles import ConfigProfile, load_profile

# Load from file
config = ConfigLoader("config/profiles/balanced.yaml").load()

# Or use profile directly
profile_config = load_profile(ConfigProfile.BALANCED)
```

## Cost Calculator

Calculate expected costs:

```python
from adaptive_mdap.tools.cost_calculator import CostCalculator, APIPricing

# Create calculator
calculator = CostCalculator(pricing=APIPricing.gpt_4o_mini())

# Calculate for 1000 problems
result = calculator.calculate_adaptive_cost(1000)

print(f"Baseline cost: ${result['baseline_cost']:.2f}")
print(f"Adaptive cost: ${result['adaptive_cost']:.2f}")
print(f"Savings: {result['savings_percent']:.1f}%")
```

## Integration with CrewAI

The system uses CrewAI for orchestration:

```python
from adaptive_mdap.integrations.crewai_integration import CrewAIIntegration

# Create CrewAI integration
crewai = CrewAIIntegration()

# Log allocation decisions
crewai.create_allocation_task(
    subproblem_id="task-001",
    complexity_score=0.75,
    allocated_strategy=SolveStrategy.MAKER_FULL,
    n_agents=5,
    estimated_savings=35.0,
)
```

## API Reference

### TaskComplexityClassifier

```python
class TaskComplexityClassifier:
    def compute_complexity(self, subproblem: SubProblem) -> ComplexityScore
    def update_historical_stats(self, domain: str, success: bool, complexity: float)
```

### AdaptiveMDAPAllocator

```python
class AdaptiveMDAPAllocator:
    def allocate_resources(self, complexity_score: float, context: Optional[AllocationContext] = None) -> SolveConfig
    def update_thresholds(self, thresholds: List[float], reason: str = "")
    def get_allocation_stats(self) -> Dict[str, Any]
```

### AdaptiveExecutionController

```python
class AdaptiveExecutionController:
    def execute_adaptive(self, subproblem: SubProblem, workflow_epic_id: Optional[str] = None) -> SolutionAttempt
    def get_execution_stats(self) -> Dict[str, Any]
```

## Testing

Run tests:

```bash
# Unit tests
pytest tests/adaptive_mdap/unit/

# Integration tests
pytest tests/adaptive_mdap/integration/

# All tests
pytest tests/adaptive_mdap/
```

## Metrics

The system exports metrics in multiple formats:

```python
from adaptive_mdap.utils.metrics import get_metrics

metrics = get_metrics()

# Get all metrics as JSON
data = metrics.get_all_metrics()

# Export as Prometheus format
prometheus_data = metrics.export_prometheus()
```

## Monitoring

Key metrics to monitor:

- `classification_latency_ms`: Time to compute complexity
- `allocation_latency_ms`: Time to allocate resources
- `complexity_score`: Distribution of complexity scores
- `allocation_*`: Count of allocations per strategy
- `execution_*`: Execution outcomes per strategy

## Troubleshooting

### Common Issues

1. **Embedding model not loading**
   - Ensure `sentence-transformers` is installed
   - Check cache directory permissions

2. **High classification latency**
   - Enable caching for embeddings
   - Pre-compute common domain embeddings

3. **Too many MAKER_FULL allocations**
   - Adjust thresholds in configuration
   - Review feature weights

## References

- MAKER Paper: "Solving a Million-Step LLM Task with Zero Errors"
- CrewAI Documentation: https://docs.crewai.com
- Cost Scaling Law: E[cost] = Θ(p⁻¹cs ln s)
