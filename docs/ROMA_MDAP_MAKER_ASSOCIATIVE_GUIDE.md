# ROMA-MDAP-MAKER + Associative Recomposition Integration Guide

## Overview

Complete integration of four powerful systems:
- **ROMA** (Recursive Open Meta-Agents) - Hierarchical problem decomposition
- **Associative Recomposition** - Domain-agnostic LLM + algorithmic verification
- **MDAP** (Multi-Agent Debate Protocol) - Multi-agent solution validation
- **MAKER** - Structured workflow orchestration

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 COMPLETE PROBLEM-SOLVING PIPELINE               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  INPUT: Problem Statement                                       │
│    ↓                                                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ PHASE 1: ROMA Hierarchical Decomposition                │  │
│  │   • Analyze problem complexity                              │  │
│  │   • Decompose into hierarchical subtasks                    │  │
│  │   • Identify dependencies                                  │  │
│  │   • Estimate atomic tasks                                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│    ↓                                                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ PHASE 2: Associative Recomposition                        │  │
│  │   • LLM classifies problem domain (not hardcoded)          │  │
│  │   • LLM creates assembly plan (structured JSON)           │  │
│  │   • AgentJSON parses plan (robust parsing)                │  │
│  │   • Algorithmic assembly (verbatim insertion)              │  │
│  │   • Ground truth verification (hash-based)                 │  │
│  │   • LLM judgment (correctness evaluation)                │  │
│  └───────────────────────────────────────────────────────────┘  │
│    ↓                                                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ PHASE 3: MDAP Multi-Agent Validation                      │  │
│  │   • Multiple agents evaluate assembled solution            │  │
│  │   • Each agent votes independently                         │  │
│  │   • Consensus reached (majority voting)                  │  │
│  │   • Aggregate metrics computed                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│    ↓                                                              │
│  OUTPUT: Final Solution with Full Metadata                      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```python
from roma_mdap_maker_associative_integration import (
    solve_with_romamdapmaker_associative
)

# Solve a problem with the complete pipeline
result = solve_with_romamdapmaker_associative(
    problem="Build a user authentication system with JWT tokens",
    context={
        "requirements": ["Secure", "Scalable", "Fast"]
    }
)

# Check result
if result['success']:
    print(f"✓ Success! Confidence: {result['confidence']:.2%}")
    print(f"Solution:\n{result['solution']}")
else:
    print(f"✗ Error: {result['error']}")
```

### Advanced Usage

```python
from roma_mdap_maker_associative_integration import (
    create_romamdapmaker_associative_config,
    ROMAMDAPMakerAssociativeEngine
)

# Create custom configuration
config = create_romamdapmaker_associative_config(
    roma_max_depth_analysis=3,
    roma_max_depth_solving=2,
    mdap_k_ahead=5,  # More agents for higher confidence
    use_associative_recomposition=True,
    enable_ground_truth=True,
    provider="anthropic",
    model="claude-3-5-sonnet-20241022"
)

# Create engine
engine = ROMAMDAPMakerAssociativeEngine(config)

# Solve multiple problems
problems = [
    "Create a REST API for e-commerce",
    "Design a real-time chat system",
    "Build a recommendation engine"
]

for problem in problems:
    result = engine.solve_problem(problem=problem)
    # Process result...
```

## Components

### 1. ROMA Decomposition

**Purpose:** Hierarchical problem decomposition

**Features:**
- Recursive task analysis
- Dependency identification
- Atomic task detection
- DAG-based decomposition

**Output:**
```python
{
    'roma_hierarchy': {...},
    'roma_dag': {...},
    'max_depth': 3,
    'sub_solutions': [
        {'id': 'sol_1', 'description': '...', 'solution_content': '...'},
        {'id': 'sol_2', 'description': '...', 'solution_content': '...'}
    ],
    'total_atomic_tasks': 5
}
```

### 2. Associative Recomposition

**Purpose:** Domain-agnostic recomposition with LLM + algorithmic verification

**Features:**
- LLM classifies domain (not hardcoded)
- Structured JSON assembly plan
- AgentJSON robust parsing
- Algorithmic assembly (verbatim)
- Ground truth verification
- LLM judgment

**Output:**
```python
{
    'assembled_solution': 'Complete assembled text...',
    'metadata': {
        'classification': {
            'domain': 'software_development',
            'solution_type': 'code',
            'field': 'web security',
            'complexity': 'medium'
        },
        'plan': {...},
        'judgment': {
            'is_correct': True,
            'quality_score': 0.90
        }
    }
}
```

### 3. MDAP Validation

**Purpose:** Multi-agent solution validation

**Features:**
- Multiple agents evaluate independently
- Consensus via majority voting
- Aggregate metrics computed
- Red-flag detection

**Output:**
```python
{
    'confidence': 0.92,
    'validation_details': {...},
    'error_rate': 0.0,
    'red_flags': 0,
    'validated': True
}
```

## Configuration

### ROMA Settings

```python
config = create_romamdapmaker_associative_config(
    # ROMA decomposition settings
    roma_max_depth_analysis=3,    # Max depth for ROMA analysis
    roma_max_depth_solving=2,     # Max depth for ROMA solving
    roma_execution_mode="recursive",  # "recursive" or "event_driven"
)
```

### MDAP/MAKER Settings

```python
config = create_romamdapmaker_associative_config(
    # MDAP/MAKER settings
    mdap_k_ahead=3,               # Voting threshold (first-to-ahead-by-k)
    mdap_max_samples=100,         # Max samples per voting round
    mdap_enable_red_flagging=True, # Enable content validation
)
```

### Associative Settings

```python
config = create_romamdapmaker_associative_config(
    # Associative recomposition settings
    use_associative_recomposition=True,  # Use associative system
    associative_max_retries=3,           # Retry attempts
    associative_use_agentjson=True,      # Use AgentJSON parsing
)
```

### Ground Truth Settings

```python
config = create_romamdapmaker_associative_config(
    # Ground truth settings
    enable_ground_truth=True,  # Enable verification
    ground_truth_storage_path="roma_mdap_maker_ground_truth.json"
)
```

## Workflow Stages

### Stage 1: ROMA Decomposition

```python
result['roma_decomposition'] = {
    'description': 'Main problem',
    'subtasks': [...]
}

result['num_sub_solutions'] = 5
result['roma_depth'] = 3
result['total_atomic_tasks'] = 12
```

### Stage 2: Associative Recomposition

```python
result['domain_classification'] = {
    'domain': 'software_development',
    'solution_type': 'code',
    'field': 'web security',
    'complexity': 'medium'
}

result['assembly_plan'] = {
    'instructions': [...],
    'success_criteria': [...]
}

result['recomposition_metadata'] = {
    'judgment': {
        'is_correct': True,
        'quality_score': 0.90
    },
    'verification_results': {
        'sol_1': (True, "Content preserved"),
        'sol_2': (True, "Content preserved")
    }
}
```

### Stage 3: MDAP Validation

```python
result['mdap_validation'] = {
    'confidence': 0.92,
    'error_rate': 0.0,
    'red_flags': 0,
    'validated': True
}
```

## Error Handling

### Fallback Behavior

```
IF ROMA unavailable:
    → Use simple decomposition (single task)
    → Warning logged

IF Associative unavailable:
    → Use simple concatenation assembly
    → Warning logged

IF MDAP unavailable:
    → Skip validation phase
    → Use default confidence (0.8)
    → Warning logged
```

### Retry Logic

```python
# Associative recomposition has built-in retry
assembled, metadata = associative_recomposer.recompose_with_verification(
    sub_solutions=solutions,
    max_retries=3,
    llm_call_fn=llm
)

# Each attempt:
# - Gets new assembly plan from LLM
# - Verifies algorithmically
# - Evaluates with LLM judgment
# - Provides feedback on failure
```

## Examples

### Example 1: Simple Problem

```python
from roma_mdap_maker_associative_integration import (
    solve_with_romamdapmaker_associative
)

result = solve_with_romamdapmaker_associative(
    problem="Create a todo list app with add, edit, delete features"
)

print(f"Success: {result['success']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Solution:\n{result['solution']}")
```

### Example 2: Complex Problem

```python
result = solve_with_romamdapmaker_associative(
    problem="Design a scalable e-commerce recommendation system",
    context={
        "constraints": [
            "Handle 1M+ products",
            "Real-time response < 100ms",
            "A/B testing framework"
        ],
        "tech_stack": ["Python", "Redis", "PostgreSQL"]
    }
)

# Analyze result
if result['success']:
    print(f"Sub-solutions: {result['num_sub_solutions']}")
    print(f"Atomic Tasks: {result['num_atomic_tasks']}")
    print(f"Time Breakdown:")
    print(f"  ROMA: {result['decomposition_time']:.2f}s")
    print(f"  Associative: {result['recomposition_time']:.2f}s")
    print(f"  MDAP: {result['validation_time']:.2f}s")
```

### Example 3: With Custom Configuration

```python
from roma_mdap_maker_associative_integration import (
    create_romamdapmaker_associative_config,
    ROMAMDAPMakerAssociativeEngine
)

# High-accuracy configuration
config = create_romamdapmaker_associative_config(
    roma_max_depth_analysis=4,  # Deeper analysis
    roma_max_depth_solving=3,
    mdap_k_ahead=7,  # More agents for higher confidence
    mdap_max_samples=200,  # More samples
    use_associative_recomposition=True,
    enable_ground_truth=True
)

engine = ROMAMDAPMakerAssociativeEngine(config)

# Solve critical problem
result = engine.solve_problem(
    problem="Design nuclear reactor control system",
    context={
        "criticality": "high",
        "safety_requirements": ["redundancy", "fail-safe"]
    }
)

# Check validation
if result['mdap_validation']['validated']:
    print("✓ Solution validated by multiple agents")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Red Flags: {result['mdap_validation']['red_flags']}")
```

## Metrics

### Execution Metrics

```python
engine = ROMAMDAPMakerAssociativeEngine(config)

# ... solve problems ...

metrics = engine.get_metrics()

print(f"Problems Solved: {metrics['total_problems_solved']}")
print(f"Avg Confidence: {metrics['avg_confidence']:.2%}")
print(f"Successful Recompositions: {metrics['successful_recompositions']}")
print(f"Failed Recompositions: {metrics['failed_recompositions']}")
print(f"Total Time: {sum([metrics['total_decomposition_time'],
                      metrics['total_recomposition_time'],
                      metrics['total_validation_time']]):.2f}s")
```

### Per-Result Metrics

```python
result = engine.solve_problem(problem=problem)

print(f"Decomposition Time: {result['decomposition_time']:.2f}s")
print(f"Recomposition Time: {result['recomposition_time']:.2f}s")
print(f"Validation Time: {result['validation_time']:.2f}s")
print(f"Total Time: {result['total_time']:.2f}s")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Error-Free: {result['error_free']}")
```

## Comparison: With vs Without Integration

| Aspect | ROMA Only | Associative Only | Full Integration |
|--------|-----------|-----------------|------------------|
| Decomposition | Hierarchical | Flat | Hierarchical |
| Recombination | Manual | Domain-agnostic LLM | Domain-agnostic LLM |
| Validation | None | LLM judgment | Multi-agent + LLM |
| Verification | None | Algorithmic | Algorithmic + Multi-agent |
| Confidence | Medium | Medium-High | **Very High** |
| Robustness | Medium | High | **Maximum** |
| Scalability | Good | Good | **Excellent** |

## Best Practices

### 1. Tune Depth Based on Complexity

```python
def estimate_complexity(problem: str) -> int:
    """Estimate problem complexity"""
    if len(problem) < 200:
        return 1  # Simple
    elif len(problem) < 500:
        return 2  # Medium
    else:
        return 3  # Complex

complexity = estimate_complexity(problem)

config = create_romamdapmaker_associative_config(
    roma_max_depth_analysis=complexity + 1,
    mdap_k_ahead=complexity + 2
)
```

### 2. Use Ground Truth for Critical Problems

```python
# For safety-critical systems, always enable ground truth
config = create_romamdapmaker_associative_config(
    enable_ground_truth=True,
    mdap_enable_red_flagging=True
)
```

### 3. Monitor Metrics

```python
engine = ROMAMDAPMakerAssociativeEngine(config)

# Track metrics over time
for problem in problems:
    result = engine.solve_problem(problem=problem)

    # Check if metrics are degrading
    if result['confidence'] < 0.7:
        logger.warning(f"Low confidence: {result['confidence']}")
        # Increase k-value or adjust configuration
```

### 4. Handle Errors Gracefully

```python
result = solve_with_romamdapmaker_associative(
    problem=problem,
    context=context
)

if result.get('error'):
    if result['phase'] == 'roma_decomposition':
        # Try with simpler decomposition
        logger.error("ROMA failed, using fallback")
    elif result['phase'] == 'associative_recomposition':
        # Try with simple assembly
        logger.error("Recomposition failed, using fallback")
    elif result['phase'] == 'mdap_validation':
        # Accept without validation
        logger.warning("MDAP validation failed, accepting result")
```

## Files

- `roma_mdap_maker_associative_integration.py` - Main integration
- `roma_mdap_maker_engine.py` - ROMA-MDAP-MAKER engine
- `associative_recomposition.py` - Associative recomposition
- `ground_truth_store.py` - Ground truth verification
- `examples/roma_mdap_maker_associative_example.py` - Working examples
- `ROMA_MDAP_MAKER_ASSOCIATIVE_GUIDE.md` - This guide

## Summary

The ROMA-MDAP-MAKER + Associative Recomposition system provides:

✅ **Hierarchical decomposition** - ROMA breaks down complex problems
✅ **Domain-agnostic recomposition** - LLM classifies and assembles
✅ **Multi-agent validation** - MDAP ensures quality
✅ **Algorithmic verification** - Ground truth prevents content loss
✅ **Complete pipeline** - End-to-end problem solving
✅ **High confidence** - Multiple layers of validation
✅ **Production-ready** - Robust error handling and fallbacks

This is the most comprehensive problem-solving system, combining the best of all approaches!
