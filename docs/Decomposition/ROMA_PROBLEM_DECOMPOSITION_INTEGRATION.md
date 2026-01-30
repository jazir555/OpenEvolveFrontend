# ROMA Integration into Problem Decomposition - Quick Reference

## Overview

ROMA (Recursive Open Meta-Agent) has been fully integrated into the `problem_decomposition.py` workflow, providing intelligent hierarchical decomposition capabilities.

## Architecture

```
ProblemDecomposer
├── ROMA DSPy Integration (Direct Library)
│   ├── Atomizer (determines atomicity)
│   ├── Planner (creates subtask plans)
│   └── Fractal Decomposition (recursive)
├── ROMA MCP Integration (via MCP Tools)
│   └── Fallback when DSPy unavailable
└── Semantic Decomposition (Fallback)
```

## Configuration Modes

### Mode 1: Profile-Based Configuration

Load from named ROMA profiles (e.g., 'crypto_agent'):

```python
decomposer = ProblemDecomposer()
result = decomposer.decompose_content(
    content="Complex problem to solve",
    strategy=DecompositionStrategy.ROMA,
    roma_profile='crypto_agent',  # Named profile
)
```

### Mode 2: Config File-Based Configuration

Load from custom YAML config file:

```python
result = decomposer.decompose_content(
    content="Complex problem to solve",
    strategy=DecompositionStrategy.ROMA,
    roma_config_path='/path/to/roma_config.yaml',
)
```

### Mode 3: Direct Model Configuration

Specify models and strategies directly:

```python
result = decomposer.decompose_content(
    content="Complex problem to solve",
    strategy=DecompositionStrategy.ROMA,
    roma_model='gpt-4o',
    roma_prediction_strategy=PredictionStrategy.CHAIN_OF_THOUGHT,
)
```

## ROMA-Specific Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `roma_fractal` | bool | True | Use fractal recursive decomposition |
| `roma_allow_small_components` | bool | True | Allow components below min_component_size |
| `roma_max_depth` | int | 3 | Maximum recursion depth |
| `roma_max_nodes` | int | max_components * 4 | Maximum nodes to create |
| `roma_include_non_leaf` | bool | False | Include intermediate plan nodes |

### Model Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `roma_model` | str | Model for both atomizer and planner |
| `roma_model_config` | dict | Model configuration dict |
| `roma_prediction_strategy` | Any | Prediction strategy for both |
| `roma_atomizer_model` | str | Model for atomizer only |
| `roma_atomizer_model_config` | dict | Atomizer model configuration |
| `roma_planner_model` | str | Model for planner only |
| `roma_planner_model_config` | dict | Planner model configuration |
| `roma_atomizer_prediction_strategy` | Any | Atomizer prediction strategy |
| `roma_planner_prediction_strategy` | Any | Planner prediction strategy |

### Context Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `roma_context` | str | None | Custom context string (overrides auto-generation) |
| `roma_extra_context` | str | None | Extra context appended to auto-generated context |
| `use_problem_analyzer` | bool | True | Use ProblemAnalyzer for domain context |

### Config File Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `roma_profile` | str | Named profile from ROMA config |
| `roma_config_path` | str | Path to ROMA YAML config file |
| `roma_overrides` | list[str] | Config override strings |

## Usage Examples

### Example 1: Basic ROMA Decomposition

```python
from problem_decomposition import ProblemDecomposer, DecompositionStrategy

# Create decomposer
decomposer = ProblemDecomposer()

# Decompose with ROMA (default strategy)
result = decomposer.decompose_content(
    content="Implement a secure authentication system with OAuth2 support",
    max_components=10,
    min_component_size=50,
)

# Access components
for component in result.components:
    print(f"Component: {component.title}")
    print(f"Type: {component.component_type}")
    print(f"Complexity: {component.complexity_score}")
    print(f"Dependencies: {component.dependencies}")
    print()
```

### Example 2: ROMA with Custom Model

```python
result = decomposer.decompose_content(
    content="Design a microservices architecture for e-commerce",
    strategy=DecompositionStrategy.ROMA,
    roma_model='claude-3-5-sonnet-20241022',
    roma_prediction_strategy=PredictionStrategy.REACT,
    roma_max_depth=4,
    roma_max_nodes=50,
)
```

### Example 3: ROMA with Domain Context

```python
from problem_analyzer import ProblemAnalyzer

# Create decomposer with problem analyzer
analyzer = ProblemAnalyzer()
decomposer = ProblemDecomposer(problem_analyzer=analyzer)

# Decompose with domain-aware ROMA
result = decomposer.decompose_content(
    content="Implement a blockchain-based voting system",
    strategy=DecompositionStrategy.ROMA,
    use_problem_analyzer=True,  # Use ProblemAnalyzer for domain context
    roma_max_depth=3,
)

# Access domain context used by ROMA
if result.metadata.get('domain_context'):
    domain = result.metadata['domain_context']
    print(f"Domain: {domain.get('domain')}")
    print(f"Key Concepts: {domain.get('key_concepts')}")
```

### Example 4: ROMA Fractal Decomposition

```python
result = decomposer.decompose_content(
    content="Build a complete CI/CD pipeline with monitoring",
    strategy=DecompositionStrategy.ROMA,
    roma_fractal=True,  # Use fractal decomposition
    roma_max_depth=3,
    roma_max_nodes=100,
    roma_include_non_leaf=False,  # Only include leaf nodes
)

# Check if ROMA was used successfully
if decomposer.last_roma_error:
    print(f"ROMA error (fallback used): {decomposer.last_roma_error}")
else:
    print("ROMA decomposition successful")
```

### Example 5: ROMA with Profile-Based Config

```python
result = decomposer.decompose_content(
    content="Develop a cryptocurrency trading bot",
    strategy=DecompositionStrategy.ROMA,
    roma_profile='crypto_agent',  # Use named profile
    roma_overrides=[
        'atomizer.model=gpt-4o',
        'planner.max_depth=4',
    ],
)
```

## Component Metadata

Each ROMA-generated component includes rich metadata:

```python
component = result.components[0]

# Basic info
print(component.id)          # roma_1_1
print(component.title)       # Subtask with goal
print(component.content)     # Goal description
print(component.complexity_score)

# Dependencies
print(component.dependencies)  # List of component IDs

# ROMA-specific metadata
metadata = component.metadata
print(metadata.get('roma_task_type'))     # Task type
print(metadata.get('roma_source'))        # 'atomizer', 'planner', etc.
print(metadata.get('roma_depth'))         # Depth in decomposition tree
print(metadata.get('roma_is_atomic'))     # Whether ROMA marked as atomic
print(metadata.get('roma_node_kind'))     # 'leaf' or 'plan'
print(metadata.get('roma_children'))      # Child node IDs (if plan node)
```

## Error Handling and Fallbacks

ROMA integration includes robust fallback logic:

1. **ROMA DSPy** → Direct library usage (primary)
2. **ROMA MCP** → MCP tools fallback (secondary)
3. **Semantic** → Basic decomposition fallback (tertiary)

```python
result = decomposer.decompose_content(
    content="Some problem",
    strategy=DecompositionStrategy.ROMA,
)

# Check if ROMA was used
if decomposer.last_roma_error:
    print(f"ROMA failed: {decomposer.last_roma_error}")
    print(f"Fallback strategy used: {result.metadata.get('fallback_strategy')}")
```

## Utility Functions

### Check ROMA Status

```python
from problem_decomposition import get_roma_integration_status

status = get_roma_integration_status()
print(f"ROMA DSPy Available: {status['roma_dspy_available']}")
print(f"ROMA MCP Available: {status['roma_mcp_available']}")
print(f"ROMA Available: {status['roma_available']}")
print(f"Recommended: {status['recommendation']}")
```

### Get Recommended Strategy

```python
from problem_decomposition import get_recommended_strategy

strategy = get_recommended_strategy(
    content="Complex problem with many functions",
    prefer_roma=True,
)
print(f"Recommended strategy: {strategy.value}")
```

## Integration with ProblemAnalyzer

ROMA decomposition leverages ProblemAnalyzer for domain context:

```python
from problem_analyzer import ProblemAnalyzer
from problem_decomposition import ProblemDecomposer

# Create problem analyzer
analyzer = ProblemAnalyzer()

# Create decomposer with analyzer
decomposer = ProblemDecomposer(problem_analyzer=analyzer)

# Decompose with domain-aware ROMA
result = decomposer.decompose_content(
    content="Design a distributed consensus algorithm",
    use_problem_analyzer=True,  # Enable domain analysis
)

# Access extracted domain context
domain_context = decomposer.last_domain_context
print(f"Domain: {domain_context.get('domain')}")
print(f"Problem Type: {domain_context.get('problem_type')}")
print(f"Key Concepts: {domain_context.get('key_concepts')}")
print(f"Complexity: {domain_context.get('complexity')}")
```

## Decomposition Quality Metrics

Each decomposition result includes quality metrics:

```python
result = decomposer.decompose_content(content="...")

# Overall quality score
print(f"Quality Score: {result.quality_score}")

# Component statistics
print(f"Components: {len(result.components)}")
print(f"Avg Size: {result.metadata.get('avg_component_size')}")

# Complexity distribution
complexity_dist = result.metadata.get('complexity_distribution', {})
print(f"Min Complexity: {complexity_dist.get('min')}")
print(f"Max Complexity: {complexity_dist.get('max')}")
print(f"Avg Complexity: {complexity_dist.get('avg')}")
print(f"High Complexity Count: {complexity_dist.get('high_complexity_count')}")
```

## Reassembly Instructions

Decomposition results include reassembly instructions:

```python
result = decomposer.decompose_content(content="...")

# Assembly order (topological sort)
assembly_order = result.reassembly_instructions['assembly_order']
print(f"Assembly Order: {assembly_order}")

# Merge strategies
merge_strategies = result.reassembly_instructions['merge_strategies']
for component_id, strategy in merge_strategies.items():
    print(f"{component_id}: {strategy}")

# Validation checks
validation_checks = result.reassembly_instructions['validation_checks']
for check in validation_checks:
    print(f"Check: {check}")
```

## Best Practices

1. **Use ROMA for Complex Content**: ROMA excels at decomposing complex, hierarchical problems
2. **Provide Domain Context**: Enable ProblemAnalyzer for domain-aware decomposition
3. **Adjust Depth**: Use `roma_max_depth` to control decomposition granularity
4. **Leverage Profiles**: Use named profiles for consistent agent behavior
5. **Monitor Quality**: Check quality scores and adjust parameters accordingly
6. **Handle Fallbacks**: Always check `last_roma_error` for fallback conditions
7. **Use Metadata**: Leverage component metadata for advanced workflows

## Troubleshooting

### ROMA Not Available

```python
status = get_roma_integration_status()
if not status['roma_available']:
    print("ROMA not available. Install roma_dspy:")
    print("  pip install roma_dspy")
```

### Decomposition Falls Back to Semantic

```python
result = decomposer.decompose_content(content="...", strategy=DecompositionStrategy.ROMA)
if decomposer.last_roma_error:
    print(f"ROMA failed: {decomposer.last_roma_error}")
    print("Check ROMA configuration and model availability")
```

### Too Many/Too Few Components

```python
# Adjust component limits
result = decomposer.decompose_content(
    content="...",
    max_components=20,  # Increase for more components
    min_component_size=100,  # Increase to filter small components
    roma_max_depth=4,  # Increase for deeper decomposition
    roma_max_nodes=200,  # Increase for more nodes
)
```

## References

- ROMA Documentation: `ROMA/README.md`
- ROMA Configuration: `ROMA/config/README.md`
- ROMA Quickstart: `ROMA/docs/QUICKSTART.md`
- Problem Analyzer: `problem_analyzer.py`
- MCP Integration: `roma_mcp_tools.py`
