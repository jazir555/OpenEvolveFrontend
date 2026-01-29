# OpenEvolve Decomposition Integration Guide

## Overview

This document describes the comprehensive integration between the **Enhanced Decomposition/Recomposition Systems** and **OpenEvolve**, creating a powerful end-to-end problem-solving pipeline.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPENEVOLVE INTEGRATED PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │   PROBLEM       │───▶│  DECOMPOSITION   │───▶│   SUB-PROBLEMS   │       │
│  │   INPUT         │    │  ENGINE          │    │   (3-10)         │       │
│  └─────────────────┘    └──────────────────┘    └──────────────────┘       │
│                              │                                               │
│                              ▼                                               │
│                    ┌──────────────────┐                                      │
│                    │  OPENEVOLVE      │                                      │
│                    │  PARALLEL        │                                      │
│                    │  EVOLUTION       │                                      │
│                    └──────────────────┘                                      │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │   INTEGRATED    │◀───│  RECOMPOSITION   │◀───│   EVOLVED        │       │
│  │   SOLUTION      │    │  ENGINE          │    │   SOLUTIONS      │       │
│  └─────────────────┘    └──────────────────┘    └──────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Files Created

### Core Files

| File | Size | Description |
|------|------|-------------|
| `enhanced_decomposition_engine.py` | 62KB | Sovereign-grade decomposition with 20+ strategies |
| `enhanced_recomposition_engine.py` | 55KB | Advanced recomposition with conflict detection |
| `decomposition_recomposition_integration.py` | 31KB | Unified pipeline connecting decomposition and recomposition |

### OpenEvolve Integration Files

| File | Size | Description |
|------|------|-------------|
| `openevolve_enhanced_decomposition_integration.py` | 33KB | Core OpenEvolve integration with evolutionary solution generation |
| `openevolve_decomposition_adapter.py` | 23KB | Adapter for existing OpenEvolve API compatibility |
| `test_openevolve_decomposition_integration.py` | 23KB | Comprehensive test suite |
| `demo_openevolve_integration.py` | 20KB | Interactive demonstration script |

## Key Features

### 1. OpenEvolveSolutionSolver

The `OpenEvolveSolutionSolver` class provides:
- **Evolutionary Solution Generation**: Uses OpenEvolve to evolve high-quality solutions
- **Automatic Prompt Generation**: Creates evolution prompts from sub-problem descriptions
- **Quality-Based Evaluation**: Evaluates solutions with customizable evaluators
- **Fallback Mechanisms**: Graceful degradation when OpenEvolve is unavailable

```python
from openevolve_enhanced_decomposition_integration import (
    OpenEvolveSolutionSolver,
    EvolutionConfig
)

solver = OpenEvolveSolutionSolver(
    openevolve_client=client,
    evolution_config=EvolutionConfig(max_iterations=50)
)

solution = solver.solve(sub_problem)
```

### 2. ParallelEvolutionManager

The `ParallelEvolutionManager` enables:
- **Dependency-Aware Parallelism**: Respects sub-problem dependencies
- **Multi-Worker Evolution**: Parallel evolution of independent sub-problems
- **Level-Based Execution**: Groups sub-problems by dependency level

```python
from openevolve_enhanced_decomposition_integration import ParallelEvolutionManager

manager = ParallelEvolutionManager(solver=solver, max_workers=4)
solutions = manager.evolve_all(sub_problems, dependency_graph)
```

### 3. OpenEvolveIntegratedPipeline

The `OpenEvolveIntegratedPipeline` provides:
- **End-to-End Integration**: Problem → Decomposition → Evolution → Assembly
- **Metrics Collection**: Comprehensive timing and quality metrics
- **Configurable Evolution**: Customizable evolution parameters
- **Quality Feedback**: Automatic refinement based on results

```python
from openevolve_enhanced_decomposition_integration import (
    OpenEvolveIntegratedPipeline,
    quick_solve_with_openevolve
)

# Quick solve
result = quick_solve_with_openevolve(
    title="Build API",
    description="Create RESTful API...",
    complexity=7.0
)

# Full pipeline
pipeline = OpenEvolveIntegratedPipeline()
result = pipeline.execute(problem, use_parallel_evolution=True)
```

### 4. OpenEvolveDecompositionAdapter

The `OpenEvolveDecompositionAdapter` provides:
- **API Compatibility**: Works with existing OpenEvolve API
- **Configuration Translation**: Converts between formats
- **Result Conversion**: Transforms results to compatible formats

```python
from openevolve_decomposition_adapter import OpenEvolveDecompositionAdapter

adapter = OpenEvolveDecompositionAdapter(openevolve_api=api)
result = adapter.decompose_and_evolve(
    problem_description="...",
    problem_title="My Problem"
)
```

## Usage Examples

### Basic Usage

```python
from enhanced_decomposition_engine import create_problem_definition
from openevolve_enhanced_decomposition_integration import (
    OpenEvolveIntegratedPipeline,
    EvolutionConfig
)

# Define problem
problem = create_problem_definition(
    title="Build Microservices",
    description="Design distributed system...",
    complexity=8.0
)

# Configure evolution
config = EvolutionConfig(
    max_iterations=50,
    parallel_evolution=True,
    max_workers=4
)

# Execute pipeline
pipeline = OpenEvolveIntegratedPipeline(evolution_config=config)
result = pipeline.execute(problem)

print(f"Quality: {result.overall_quality}")
print(f"Sub-problems: {len(result.sub_solutions)}")
```

### With Existing OpenEvolve Infrastructure

```python
from openevolve_integration import OpenEvolveAPI
from openevolve_decomposition_adapter import (
    OpenEvolveDecompositionAdapter,
    integrate_with_existing_openevolve
)

# Use existing API
api = OpenEvolveAPI(base_url="http://localhost:8000", api_key="...")

# Integrate with decomposition
result = integrate_with_existing_openevolve(
    openevolve_api=api,
    problem_description="...",
    problem_title="My Problem"
)
```

### Strategy Comparison

```python
from openevolve_enhanced_decomposition_integration import (
    compare_strategies_with_openevolve
)

result = compare_strategies_with_openevolve(
    problem,
    strategies=[
        DecompositionStrategy.HIERARCHICAL,
        DecompositionStrategy.FUNCTIONAL,
        DecompositionStrategy.SEMANTIC,
    ]
)

print(f"Best strategy: {result['best_strategy']}")
for r in result['results']:
    print(f"{r['strategy']}: {r['solution_quality']:.2f}")
```

### Metrics Collection

```python
from openevolve_decomposition_adapter import DecompositionMetricsCollector

collector = DecompositionMetricsCollector()

# Collect metrics during execution
collector.collect_decomposition_metrics(plan, decomp_time)
collector.collect_evolution_metrics(sp_id, fitness, iterations, time)
collector.collect_recomposition_metrics(solution, recomp_time)

# Get summary
summary = collector.get_summary()
print(f"Avg Fitness: {summary['avg_fitness']}")
print(f"Total Operations: {summary['total_operations']}")
```

## Decomposition Strategies

The following strategies are available for problem decomposition:

| Strategy | Best For |
|----------|----------|
| `HIERARCHICAL` | Well-structured problems with clear hierarchy |
| `FUNCTIONAL` | System design with distinct capabilities |
| `SEMANTIC` | Complex problems requiring conceptual analysis |
| `TEMPORAL` | Problems with chronological phases |
| `CAUSAL` | Diagnostic or root-cause problems |
| `RISK_BASED` | High-risk projects requiring risk mitigation |
| `COMPLEXITY` | Very complex problems needing cognitive load balancing |
| `DEPENDENCY` | Problems with clear prerequisite relationships |
| `HYBRID` | General-purpose, adaptive strategy |

## Integration Benefits

### 1. Higher Quality Solutions
- Evolutionary optimization improves solution quality
- Multiple iterations refine solutions automatically
- Quality thresholds ensure minimum standards

### 2. Parallel Processing
- Independent sub-problems evolved in parallel
- Dependency-aware execution order
- Reduced total execution time

### 3. Intelligent Conflict Resolution
- Automatic detection of 12+ conflict types
- Multiple resolution strategies
- LLM-mediated resolution for complex conflicts

### 4. Comprehensive Metrics
- Timing metrics for each stage
- Quality metrics for solutions
- Evolution statistics tracking

### 5. Backward Compatibility
- Works with existing OpenEvolve API
- Graceful fallback when OpenEvolve unavailable
- Adapter pattern for easy integration

## Testing

Run the comprehensive test suite:

```bash
python test_openevolve_decomposition_integration.py
```

Run the interactive demo:

```bash
python demo_openevolve_integration.py
```

## Performance Considerations

| Factor | Recommendation |
|--------|----------------|
| Parallel Workers | Set to number of CPU cores |
| Max Iterations | 25-50 for quick results, 100+ for quality |
| Population Size | 50-100 for most problems |
| Sub-Problem Count | 3-10 for optimal balance |

## Troubleshooting

### OpenEvolve Not Available

If OpenEvolve is not available, the system automatically falls back to simulated evolution:

```python
# Fallback is automatic
solver = OpenEvolveSolutionSolver(openevolve_client=None)
solution = solver.solve(sub_problem)  # Uses fallback
```

### Low Solution Quality

Increase evolution iterations or adjust thresholds:

```python
config = EvolutionConfig(
    max_iterations=100,
    min_fitness_threshold=0.8,
    target_fitness=0.95
)
```

### Memory Issues

Reduce parallel workers or population size:

```python
config = EvolutionConfig(
    parallel_evolution=True,
    max_workers=2,  # Reduce from 4
    population_size=50  # Reduce from 100
)
```

## API Reference

### OpenEvolveIntegratedPipeline

```python
class OpenEvolveIntegratedPipeline:
    def __init__(
        self,
        decomposition_engine: Optional[EnhancedDecompositionEngine] = None,
        recomposition_engine: Optional[EnhancedRecompositionEngine] = None,
        openevolve_client: Optional[Any] = None,
        evolution_config: Optional[EvolutionConfig] = None,
        pipeline_config: Optional[PipelineConfig] = None
    )
    
    def execute(
        self,
        problem: ProblemDefinition,
        use_parallel_evolution: bool = True
    ) -> PipelineResult
```

### OpenEvolveSolutionSolver

```python
class OpenEvolveSolutionSolver:
    def __init__(
        self,
        openevolve_client: Optional[Any] = None,
        evolution_config: Optional[EvolutionConfig] = None,
        custom_evaluator: Optional[Callable] = None
    )
    
    def solve(self, sub_problem: SubProblem) -> SubProblemSolution
    def can_solve(self, sub_problem: SubProblem) -> Tuple[bool, float]
```

### EvolutionConfig

```python
@dataclass
class EvolutionConfig:
    max_iterations: int = 50
    population_size: int = 100
    num_islands: int = 3
    min_fitness_threshold: float = 0.7
    target_fitness: float = 0.9
    parallel_evolution: bool = True
    max_workers: int = 4
    temperature: float = 0.7
    max_tokens: int = 4096
```

## Summary

The OpenEvolve integration provides:

1. **20+ Decomposition Strategies** for intelligent problem breakdown
2. **Evolutionary Solution Generation** using OpenEvolve platform
3. **Parallel Processing** for efficient sub-problem evolution
4. **Advanced Conflict Detection** with 12+ conflict types
5. **Comprehensive Metrics** for performance tracking
6. **Backward Compatibility** with existing infrastructure

This integration creates a sovereign-grade problem-solving system that combines the power of intelligent decomposition with evolutionary optimization.
