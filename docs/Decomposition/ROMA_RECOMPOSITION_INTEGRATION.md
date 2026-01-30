# ROMA Solution Recomposition Integration Guide

## Overview

ROMA (Recursive Open Meta-Agent) integration for **solution recomposition** - the process of intelligently assembling sub-solutions back into coherent, integrated solutions.

This guide covers ROMA's role in the recomposition phase of problem-solving, where individual sub-solutions are combined into unified outputs.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Core Concepts](#core-concepts)
- [Configuration Modes](#configuration-modes)
- [Parameter Reference](#parameter-reference)
- [Usage Examples](#usage-examples)
- [Conflict Resolution](#conflict-resolution)
- [Quality Metrics](#quality-metrics)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### The Recomposition Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    SOLUTION ATTEMPTS                         │
│  (Sub-solutions generated for each sub-problem)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  CONFLICT DETECTION                          │
│  - Semantic similarity conflicts                             │
│  - Contradictory requirements                               │
│  - Dependency violations                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  ROMA RECOMPOSITION                          │
│  - Domain-aware context building                            │
│  - Hierarchical solution assembly                          │
│  - LLM-mediated integration                                │
│  - Coherence optimization                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  INTEGRATED SOLUTION                         │
│  - Coherent, unified output                                 │
│  - Quality metrics                                          │
│  - Assembly metadata                                        │
└─────────────────────────────────────────────────────────────┘
```

### ROMA's Role in Recomposition

ROMA enhances traditional recomposition by:

1. **Semantic Integration**: Understanding the meaning and relationships between sub-solutions
2. **Context Awareness**: Leveraging domain knowledge for intelligent assembly
3. **Conflict Resolution**: Using LLM reasoning to resolve detected conflicts
4. **Hierarchical Assembly**: Maintaining problem structure during integration
5. **Coherence Optimization**: Ensuring narrative flow and logical consistency

---

## Core Concepts

### SolutionAttempt vs AssembledSolution

```python
# Individual sub-solution
sub_solution = SolutionAttempt(
    solution_id="sol_1",
    sub_problem_id="sub_1",
    solution_content="Implement JWT authentication...",
    confidence_score=0.9,
)

# Assembled integrated solution
assembled = AssembledSolution(
    assembled_content="# Complete Authentication System\n\n...",
    assembly_strategy="roma",
    quality_metrics=QualityMetrics(
        overall_score=0.85,
        coherence_score=0.88,
        integration_quality=0.82,
        consistency_score=0.90,
    ),
)
```

### Conflict Types

1. **Semantic Conflicts**: Similar but inconsistent terminology
2. **Contradictions**: Directly opposing statements
3. **Dependency Violations**: Required components missing or incorrect
4. **Structural Conflicts**: Incompatible organization or formatting

### Assembly Strategies

```python
# Traditional strategies
assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    assembly_strategy="hierarchical",  # Top-down assembly
)

assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    assembly_strategy="linear",  # Sequential assembly
)

assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    assembly_strategy="parallel",  # Independent assembly
)

# ROMA-enhanced strategy
assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    assembly_strategy="roma",  # LLM-mediated intelligent assembly
    roma_max_depth=2,
    roma_context="Custom domain context...",
)
```

---

## Configuration Modes

### Mode 1: Direct Parameter Configuration

```python
from problem_recomposition import SolutionAssembler

assembler = SolutionAssembler(
    enable_roma=True,
    roma_max_depth=2,
    roma_model="gpt-4o",
    roma_provider="openai",
)

result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    assembly_strategy="roma",
    roma_temperature=0.7,
    roma_max_tokens=4000,
)
```

### Mode 2: Configuration Helper

```python
from problem_recomposition import SolutionAssembler
from roma_recomposition_config import ROMARecompositionConfig, ROMARecompositionPresets

# Use preset
config = ROMARecompositionPresets.balanced()

assembler = SolutionAssembler(
    enable_roma=True,
    roma_max_depth=config.max_depth,
)

result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    **config.to_kwargs(),
)
```

### Mode 3: Recommended Configuration

```python
from roma_recomposition_config import get_recommended_recomposition_config

# Auto-select based on characteristics
config = get_recommended_recomposition_config(
    num_sub_solutions=8,
    num_conflicts=5,
    complexity="high",
    content_type="code",
)

assembler = SolutionAssembler(enable_roma=True, roma_max_depth=config.max_depth)

result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    **config.to_kwargs(),
)
```

---

## Parameter Reference

### Core Recomposition Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_roma` | bool | True | Enable ROMA for recomposition |
| `roma_max_depth` | int | 2 | Maximum recursion depth for assembly |
| `roma_execution_mode` | str | "recursive" | Execution mode: "recursive" or "iterative" |
| `assembly_strategy` | str | "hierarchical" | Strategy to use: "roma", "hierarchical", "linear", "parallel" |

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `roma_provider` | str | None | AI provider: "openai", "anthropic", etc. |
| `roma_model` | str | None | Model name for recomposition |
| `roma_temperature` | float | 0.7 | LLM temperature (0.0-1.0) |
| `roma_max_tokens` | int | 4000 | Maximum tokens to generate |

### Context Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `roma_context` | str | None | Custom context string for recomposition |
| `roma_extra_context` | str | None | Extra context appended to auto-generated |
| `roma_strategy` | str | "chain_of_thought" | ROMA prediction strategy |

### Conflict Resolution Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_conflict_resolution` | bool | True | Enable LLM-mediated conflict resolution |
| `conflict_resolution_fallback` | str | "priority" | Fallback if LLM fails: "priority", "merge", "manual" |

### Tracking Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `track_in_hephaestus` | bool | False | Track recomposition in Hephaestus |

---

## Usage Examples

### Example 1: Basic ROMA Recomposition

```python
from problem_recomposition import SolutionAssembler
from sovereign_data_models import (
    DecompositionPlan,
    SubProblem,
    SolutionAttempt,
    generate_id,
)

# Create assembler with ROMA
assembler = SolutionAssembler(
    enable_roma=True,
    roma_max_depth=2,
)

# Create sub-solutions
sub_solutions = {
    "sub_1": SolutionAttempt(
        solution_id="sol_1",
        sub_problem_id="sub_1",
        solution_content="## User Authentication\n\nImplement JWT tokens...",
        confidence_score=0.9,
    ),
    "sub_2": SolutionAttempt(
        solution_id="sol_2",
        sub_problem_id="sub_2",
        solution_content="## User Profile\n\nProfile management...",
        confidence_score=0.85,
    ),
}

# Create decomposition plan
plan = DecompositionPlan(
    id=generate_id("plan"),
    problem_id=generate_id("problem"),
    problem_statement="Build user management system",
    sub_problems=[
        SubProblem(id="sub_1", description="Authentication", dependencies=[]),
        SubProblem(id="sub_2", description="Profile Management", dependencies=[]),
    ],
)

# Assemble with ROMA
result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    assembly_strategy="roma",
)

print(f"Quality: {result.quality_metrics.overall_score:.2f}")
print(f"Content: {result.assembled_content[:500]}...")
```

### Example 2: Using Configuration Presets

```python
from roma_recomposition_config import ROMARecompositionPresets

# Fast recomposition
config = ROMARecompositionPresets.fast()
result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    **config.to_kwargs(),
)

# High-conflict recomposition
config = ROMARecompositionPresets.high_conflict()
result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    **config.to_kwargs(),
)

# Code-focused recomposition
config = ROMARecompositionPresets.code_focused()
result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    **config.to_kwargs(),
)
```

### Example 3: Custom Domain Context

```python
# Custom domain context
custom_context = """
Domain: E-commerce Platform Architecture
Key Constraints:
- Must support both REST and GraphQL
- REST for simple CRUD operations
- GraphQL for complex queries and dashboard
- Ensure consistent authentication across both APIs
- Performance target: <100ms for 95th percentile
"""

result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    assembly_strategy="roma",
    roma_context=custom_context,
    roma_extra_context="Focus on API consistency and performance.",
)
```

### Example 4: Comparing Assembly Strategies

```python
strategies = ["hierarchical", "linear", "parallel", "roma"]

for strategy in strategies:
    result = assembler.assemble_solution(
        decomposition_plan=plan,
        sub_solutions=sub_solutions,
        assembly_strategy=strategy,
    )

    metrics = result.quality_metrics
    print(f"{strategy}:")
    print(f"  Quality: {metrics.overall_score:.3f}")
    print(f"  Coherence: {metrics.coherence_score:.3f}")
    print(f"  Integration: {metrics.integration_quality:.3f}")
```

---

## Conflict Resolution

### Automatic Conflict Detection

```python
# Conflicts are automatically detected
result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    assembly_strategy="roma",
)

# Access conflict information
num_conflicts = result.metadata.get('num_conflicts', 0)
num_resolved = result.metadata.get('num_resolved', 0)

print(f"Conflicts detected: {num_conflicts}")
print(f"Conflicts resolved: {num_resolved}")
```

### Conflict Resolution Strategies

```python
from roma_recomposition_config import ROMARecompositionConfig

# Priority-based (use highest confidence solution)
config = ROMARecompositionConfig(
    enable_conflict_resolution=True,
    conflict_resolution_fallback="priority",
)

# Merge-based (combine conflicting solutions)
config = ROMARecompositionConfig(
    enable_conflict_resolution=True,
    conflict_resolution_fallback="merge",
)

# Manual (require human intervention)
config = ROMARecompositionConfig(
    enable_conflict_resolution=True,
    conflict_resolution_fallback="manual",
)
```

### Handling Contradictions

```python
# Sub-solutions with contradictions
sub_solutions = {
    "sub_1": SolutionAttempt(
        solution_id="sol_1",
        sub_problem_id="sub_1",
        solution_content="Use PostgreSQL for strong consistency.",
        confidence_score=0.9,
    ),
    "sub_2": SolutionAttempt(
        solution_id="sol_2",
        sub_problem_id="sub_2",
        solution_content="Use MongoDB for flexible schema.",
        confidence_score=0.85,
    ),
}

# ROMA will detect and attempt resolution
config = ROMARecompositionPresets.high_conflict()
result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    **config.to_kwargs(),
)

# Check resolution success
if result.quality_metrics.consistency_score > 0.7:
    print("Successfully resolved contradictions")
else:
    print("Conflicts remain - manual review needed")
```

---

## Quality Metrics

### Available Metrics

```python
result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    assembly_strategy="roma",
)

metrics = result.quality_metrics

# Overall quality score (0-1)
print(f"Overall: {metrics.overall_score:.2f}")

# Coherence: How well components flow together
print(f"Coherence: {metrics.coherence_score:.2f}")

# Integration quality: Technical correctness of integration
print(f"Integration: {metrics.integration_quality:.2f}")

# Consistency: Absence of contradictions
print(f"Consistency: {metrics.consistency_score:.2f}")

# Completeness: Coverage of all requirements
print(f"Completeness: {metrics.completeness_score:.2f}")

# Clarity: Readability and understandability
print(f"Clarity: {metrics.clarity_score:.2f}")
```

### Interpreting Quality Scores

- **0.9-1.0**: Excellent - Production-ready
- **0.7-0.9**: Good - Minor improvements possible
- **0.5-0.7**: Fair - Some issues present
- **0.3-0.5**: Poor - Significant problems
- **0.0-0.3**: Failing - Requires major revision

---

## Best Practices

### 1. Choose the Right Preset

```python
from roma_recomposition_config import ROMARecompositionPresets

# Quick integration for simple problems
config = ROMARecompositionPresets.fast()

# General-purpose recomposition
config = ROMARecompositionPresets.balanced()

# Complex, multi-faceted solutions
config = ROMARecompositionPresets.thorough()

# Solutions with many conflicts
config = ROMARecompositionPresets.high_conflict()

# Software code integration
config = ROMARecompositionPresets.code_focused()

# Documentation and prose
config = ROMARecompositionPresets.documentation_focused()

# Creative, innovative solutions
config = ROMARecompositionPresets.creative()
```

### 2. Use Domain Context

```python
# Provide domain-specific context for better results
domain_context = """
Domain: Financial Trading System
Constraints:
- ACID compliance required
- Sub-millisecond latency
- Real-time risk management
- Audit trail for all transactions
"""

result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    assembly_strategy="roma",
    roma_context=domain_context,
)
```

### 3. Monitor Quality Metrics

```python
result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    assembly_strategy="roma",
)

# Check quality before accepting
if result.quality_metrics.overall_score < 0.7:
    # Try different approach
    result = assembler.assemble_solution(
        decomposition_plan=plan,
        sub_solutions=sub_solutions,
        assembly_strategy="roma",
        roma_max_depth=3,  # Deeper recomposition
        roma_temperature=0.8,  # More creative integration
    )
```

### 4. Handle Errors Gracefully

```python
try:
    result = assembler.assemble_solution(
        decomposition_plan=plan,
        sub_solutions=sub_solutions,
        assembly_strategy="roma",
    )
except Exception as exc:
    logger.error(f"ROMA recomposition failed: {exc}")
    # Fall back to traditional assembly
    result = assembler.assemble_solution(
        decomposition_plan=plan,
        sub_solutions=sub_solutions,
        assembly_strategy="hierarchical",
    )
```

### 5. Validate Before Recomposition

```python
# Ensure sub-solutions exist and are valid
for sub_id, solution in sub_solutions.items():
    if not solution.solution_content:
        raise ValueError(f"Sub-solution {sub_id} is empty")
    if solution.confidence_score < 0.5:
        logger.warning(f"Sub-solution {sub_id} has low confidence")

# Proceed with recomposition
result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    assembly_strategy="roma",
)
```

### 6. Use Appropriate Depth

```python
# More components → deeper assembly
num_components = len(sub_solutions)

if num_components <= 3:
    max_depth = 1
elif num_components <= 7:
    max_depth = 2
else:
    max_depth = 3

result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    assembly_strategy="roma",
    roma_max_depth=max_depth,
)
```

---

## Troubleshooting

### Issue: ROMA Recomposition Fails

**Symptoms**: Exception during ROMA assembly, fallback to traditional method

**Solutions**:

1. **Check ROMA Availability**:
```python
from problem_recomposition import get_roma_recomposition_status

status = get_roma_recomposition_status()
if not status["roma_available"]:
    print("ROMA is not available. Install: pip install roma")
```

2. **Verify API Keys**:
```python
import os
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set"
assert os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not set"
```

3. **Check Model Access**:
```python
# Test model access
assembler = SolutionAssembler(
    enable_roma=True,
    roma_model="gpt-4o",
    roma_provider="openai",
)

# Verify with simple test
test_plan = DecompositionPlan(
    id="test",
    problem_id="test",
    problem_statement="Test",
    sub_problems=[],
)
```

### Issue: Low Quality Scores

**Symptoms**: Quality scores below 0.7, incoherent output

**Solutions**:

1. **Increase Depth**:
```python
result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    assembly_strategy="roma",
    roma_max_depth=3,  # Deeper recomposition
)
```

2. **Adjust Temperature**:
```python
# For code: lower temperature
result = assembler.assemble_solution(
    assembly_strategy="roma",
    roma_temperature=0.3,  # More deterministic
)

# For creative content: higher temperature
result = assembler.assemble_solution(
    assembly_strategy="roma",
    roma_temperature=0.9,  # More creative
)
```

3. **Provide Better Context**:
```python
# Add detailed domain context
result = assembler.assemble_solution(
    assembly_strategy="roma",
    roma_context="Detailed domain context...",
    roma_extra_context="Additional guidance...",
)
```

### Issue: Conflicts Not Resolved

**Symptoms**: High conflict count, low consistency score

**Solutions**:

1. **Enable Conflict Resolution**:
```python
config = ROMARecompositionConfig(
    enable_conflict_resolution=True,
    conflict_resolution_fallback="merge",
)
```

2. **Use High-Conflict Preset**:
```python
config = ROMARecompositionPresets.high_conflict()
result = assembler.assemble_solution(
    assembly_strategy="roma",
    **config.to_kwargs(),
)
```

3. **Manual Review**:
```python
# If automatic resolution fails
if result.quality_metrics.consistency_score < 0.6:
    conflicts = result.metadata.get('conflicts', [])
    print(f"Manual review needed for {len(conflicts)} conflicts")
```

### Issue: Slow Recomposition

**Symptoms**: Long assembly time (>30 seconds)

**Solutions**:

1. **Use Fast Preset**:
```python
config = ROMARecompositionPresets.fast()
result = assembler.assemble_solution(
    assembly_strategy="roma",
    **config.to_kwargs(),
)
```

2. **Reduce Depth**:
```python
result = assembler.assemble_solution(
    assembly_strategy="roma",
    roma_max_depth=1,  # Shallower recomposition
)
```

3. **Use Iterative Mode**:
```python
result = assembler.assemble_solution(
    assembly_strategy="roma",
    roma_execution_mode="iterative",  # Often faster than recursive
)
```

---

## Advanced Features

### Hephaestus Tracking

```python
# Track recomposition in Hephaestus for observability
result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    assembly_strategy="roma",
    track_in_hephaestus=True,
)

# Access Hephaestus metadata
hephaestus_id = result.metadata.get('hephaestus_task_id')
print(f"Tracked in Hephaestus: {hephaestus_id}")
```

### Custom Prediction Strategies

```python
from dspy.primitives import PredictionStrategy

# Chain of Thought (default)
result = assembler.assemble_solution(
    assembly_strategy="roma",
    roma_strategy="chain_of_thought",
)

# ReAct (Reasoning + Acting)
result = assembler.assemble_solution(
    assembly_strategy="roma",
    roma_strategy="react",
)

# Few-shot learning
result = assembler.assemble_solution(
    assembly_strategy="roma",
    roma_strategy="fewshot",
)
```

### Environment-Based Configuration

```python
from roma_recomposition_config import create_recomposition_config_from_env

# Set environment variables
# export ROMA_RECOMPOSITION_MODEL=gpt-4o
# export ROMA_RECOMPOSITION_MAX_DEPTH=2
# export ROMA_RECOMPOSITION_TEMPERATURE=0.7

config = create_recomposition_config_from_env()

assembler = SolutionAssembler(
    enable_roma=True,
    roma_max_depth=config.max_depth,
)

result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    **config.to_kwargs(),
)
```

---

## Integration Examples

### With Problem Decomposition

```python
from problem_decomposition import ProblemDecomposer, DecompositionStrategy
from problem_recomposition import SolutionAssembler

# Decompose problem
decomposer = ProblemDecomposer(auto_create_analyzer=False)
decomposition = decomposer.decompose_content(
    content="Build a microservices architecture",
    strategy=DecompositionStrategy.ROMA,
    max_components=8,
)

# Solve each sub-problem (mock solutions here)
sub_solutions = {}
for component in decomposition.components:
    sub_solutions[component.id] = SolutionAttempt(
        solution_id=f"sol_{component.id}",
        sub_problem_id=component.id,
        solution_content=f"Solution for {component.description}",
        confidence_score=0.85,
    )

# Recompose solutions
assembler = SolutionAssembler(enable_roma=True, roma_max_depth=2)
assembled = assembler.assemble_solution(
    decomposition_plan=decomposition,
    sub_solutions=sub_solutions,
    assembly_strategy="roma",
)

print(f"Final solution quality: {assembled.quality_metrics.overall_score:.2f}")
```

### With Adversarial Testing

```python
from adversarial import AdversarialValidator

# Assemble with ROMA
result = assembler.assemble_solution(
    decomposition_plan=plan,
    sub_solutions=sub_solutions,
    assembly_strategy="roma",
)

# Validate with adversarial testing
validator = AdversarialValidator()
validation_result = validator.validate_solution(
    solution=result.assembled_content,
    problem_statement=plan.problem_statement,
)

if validation_result.passed:
    print("Solution passed adversarial validation")
else:
    print(f"Solution failed: {validation_result.issues}")
```

---

## References

- **ROMA Documentation**: `ROMA/README.md`
- **ROMA Configuration**: `ROMA/config/README.md`
- **Problem Decomposition**: `problem_decomposition.py`
- **Configuration Helper**: `roma_recomposition_config.py`
- **Example Scripts**: `examples/roma_recomposition_examples.py`
- **Decomposition Integration**: `ROMA_PROBLEM_DECOMPOSITION_INTEGRATION.md`

---

## Summary

ROMA recomposition provides intelligent, context-aware solution assembly with:

✅ **Semantic Integration**: Understands meaning and relationships
✅ **Conflict Resolution**: LLM-mediated conflict handling
✅ **Domain Awareness**: Leverages domain knowledge
✅ **Quality Metrics**: Comprehensive quality assessment
✅ **Flexible Configuration**: 7 presets + custom configs
✅ **Robust Fallbacks**: Graceful degradation on errors
✅ **Production Ready**: Tested and documented

For questions or issues, refer to the troubleshooting guide or example scripts.
