# ROMA Reliability Adapter

**Wrapper that adds LMQL constraints and Guardrails validation to ROMA without modifying ROMA core code.**

## Architecture Principle: AIR GAP

This adapter follows the **AIR GAP PRINCIPLE** from the OpenEvolve constitution:

1. **NO IMPORTS from ROMA core source files** - Only uses public MCP tools
2. **NO MODIFICATIONS to ROMA core files** - ROMA remains read-only
3. **Wrapper Pattern** - All reliability logic lives in the adapter, not ROMA
4. **Graceful Degradation** - Works even if LMQL/Guardrails unavailable

### Architecture Flow

```
ROMA Core (READ ONLY)
    ↓
MCP Tools Interface (solve_with_roma, analyze_with_roma)
    ↓
Reliability Adapter (LMQL constraints + Guardrails validation)
    ↓
Unified Bridge
```

## Features

### Layer 1: Input Validation (Guardrails)
- Validates input task length
- Checks for toxic language
- Ensures input safety before processing

### Layer 2: Pre-Generation Constraints (LMQL)
- **Depth Constraint**: Limits decomposition depth (1-N levels)
- **Subtask Count**: Limits number of subtasks generated
- **Token Limit**: Constrains subtask description length
- **JSON Format**: Enforces structured JSON output
- **Custom Constraints**: Support for custom LMQL constraints

### Layer 3: ROMA Execution (via MCP Tools)
- Calls ROMA via public MCP tool interface
- No direct imports from ROMA core
- Supports all ROMA execution modes (recursive, event-driven)

### Layer 4: Output Validation (Guardrails)
- Validates output structure (JSON)
- Checks decomposition depth
- Applies remediation if validation fails
- Logs all validation results

## Installation

```bash
# Located at: reliability-plugin/adapters/roma/
# Automatically imported by unified bridge
```

## Quick Start

### Basic Usage

```python
from reliability_plugin.adapters.roma import (
    RomaReliabilityAdapter,
    solve_with_constraints
)

# Method 1: Use convenience function
result = solve_with_constraints(
    task="Solve the traveling salesman problem",
    max_depth=3,
    constraints={
        "max_depth": 3,
        "max_subtasks": 10,
        "subtask_token_limit": 500
    }
)

if result.success:
    print(f"Solution: {result.result}")
    print(f"Layers used: {result.layers_used}")
else:
    print(f"Error: {result.error}")
    print(f"Validation failures: {result.validation_failures}")
```

### Advanced Usage

```python
from reliability_plugin.adapters.roma import RomaReliabilityAdapter

# Create adapter with custom configuration
adapter = RomaReliabilityAdapter()

# Solve with detailed constraints
result = adapter.solve_with_constraints(
    task="Design a microservices architecture for e-commerce",
    max_depth=4,
    execution_mode="event_driven",
    enable_checkpoints=True,
    constraints={
        "max_depth": 4,
        "max_subtasks": 15,
        "subtask_token_limit": 750,
        "require_json": True
    },
    provider="openai",
    model="gpt-4"
)

# Check result
if result.success:
    # Access ROMA result
    roma_result = result.result
    print(f"Status: {roma_result.get('status')}")
    print(f"Result: {roma_result.get('result')}")

    # Check constraint violations
    if result.has_violations():
        print(f"Violations: {result.constraint_violations}")

    # Check validation failures
    if result.has_validation_failures():
        print(f"Validation issues: {result.validation_failures}")

    # Check if remediation was applied
    if result.was_remediated():
        print(f"Remediations: {result.remediation_applied}")
```

## Configuration

### Environment Variables

```bash
# ROMA Adapter Settings
export ROMA_ADAPTER_ENABLED=true
export ROMA_LMQL_ENABLED=true
export ROMA_GUARDRAILS_ENABLED=true
export ROMA_MAX_DEPTH=3
export ROMA_EXECUTION_MODE=recursive
export ROMA_CHECKPOINTS=true
export ROMA_FALLBACK=true
export ROMA_MAX_RETRIES=3
```

### Programmatic Configuration

```python
from reliability_plugin.adapters.roma.config import (
    RomaAdapterConfig,
    set_config,
    create_constraints
)

# Create custom configuration
config = RomaAdapterConfig(
    enabled=True,
    lmql_enabled=True,
    guardrails_enabled=True,
    max_depth_default=4,
    execution_mode_default="event_driven",
    constraint_defaults={
        "max_depth": 4,
        "max_subtasks": 15,
        "subtask_token_limit": 750
    }
)

# Validate and set
errors = config.validate()
if not errors:
    set_config(config)
```

## Constraint Builder

Use the fluent constraint builder for creating complex constraints:

```python
from reliability_plugin.adapters.roma.config import create_constraints

# Build constraints
constraints = create_constraints() \
    .with_max_depth(3) \
    .with_max_subtasks(10) \
    .with_subtask_token_limit(500) \
    .require_json() \
    .build()

# Use constraints
result = solve_with_constraints(
    task="Analyze the system architecture",
    max_depth=3,
    constraints=constraints
)
```

## API Reference

### RomaReliabilityAdapter

Main adapter class that wraps ROMA with reliability layers.

#### Methods

##### `solve_with_constraints()`

```python
def solve_with_constraints(
    task: str,
    max_depth: int = 3,
    constraints: Optional[Dict[str, Any]] = None,
    execution_mode: str = "recursive",
    enable_checkpoints: bool = True,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> RomaSolutionResult
```

**Parameters:**
- `task`: The task to solve
- `max_depth`: Maximum decomposition depth (default: 3)
- `constraints`: Optional LMQL constraints dict
- `execution_mode`: "recursive" or "event_driven"
- `enable_checkpoints`: Enable ROMA checkpoint/recovery
- `provider`: LLM provider (openai, anthropic, google, openrouter)
- `model`: Model name
- `api_key`: API key for provider

**Returns:** `RomaSolutionResult`

##### `analyze_with_constraints()`

```python
def analyze_with_constraints(
    task: str,
    analysis_type: str = "decomposition",
    max_depth: int = 3,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> RomaAnalysisResult
```

**Parameters:**
- `task`: Problem statement to analyze
- `analysis_type`: "decomposition", "complexity", or "dependencies"
- `max_depth`: Maximum decomposition depth
- `provider`: LLM provider
- `model`: Model name

**Returns:** `RomaAnalysisResult`

##### `verify_with_constraints()`

```python
def verify_with_constraints(
    solution: str,
    original_task: str,
    verification_criteria: Optional[List[str]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> RomaSolutionResult
```

**Parameters:**
- `solution`: The solution to verify
- `original_task`: The original task/problem
- `verification_criteria`: List of criteria to verify
- `provider`: LLM provider
- `model`: Model name

**Returns:** `RomaSolutionResult`

##### `critique_with_constraints()`

```python
def critique_with_constraints(
    solution: str,
    original_task: str,
    critique_focus: str = "comprehensive",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> RomaSolutionResult
```

**Parameters:**
- `solution`: The solution to critique
- `original_task`: The original task
- `critique_focus`: "comprehensive", "security", "performance", "correctness"
- `provider`: LLM provider
- `model`: Model name

**Returns:** `RomaSolutionResult`

### Result Types

#### RomaSolutionResult

```python
@dataclass
class RomaSolutionResult:
    success: bool
    result: Optional[Dict[str, Any]] = None
    task: Optional[str] = None
    error: Optional[str] = None
    layers_used: List[str] = []
    constraint_violations: List[str] = []
    validation_failures: List[Dict[str, Any]] = []
    remediation_applied: List[str] = []
    correlation_id: str = ""
    metadata: Dict[str, Any] = {}
```

**Methods:**
- `to_dict()`: Convert to dictionary for JSON serialization
- `has_violations()`: Check if constraint violations occurred
- `has_validation_failures()`: Check if validation failures occurred
- `was_remediated()`: Check if remediations were applied

#### RomaAnalysisResult

```python
@dataclass
class RomaAnalysisResult:
    success: bool
    analysis: Optional[Dict[str, Any]] = None
    task: Optional[str] = None
    error: Optional[str] = None
    layers_used: List[str] = []
    validation_failures: List[Dict[str, Any]] = []
    correlation_id: str = ""
    metadata: Dict[str, Any] = {}
```

## Examples

### Example 1: Basic Problem Solving

```python
from reliability_plugin.adapters.roma import solve_with_constraints

result = solve_with_constraints(
    task="Implement a REST API for user management",
    max_depth=3
)

if result.success:
    print(f"Solution generated using layers: {result.layers_used}")
    print(f"ROMA result: {result.result}")
```

### Example 2: Constrained Decomposition

```python
from reliability_plugin.adapters.roma import solve_with_constraints

result = solve_with_constraints(
    task="Design a database schema for a multi-tenant SaaS application",
    max_depth=4,
    constraints={
        "max_depth": 4,
        "max_subtasks": 12,
        "subtask_token_limit": 600,
        "require_json": True
    }
)

# Check for issues
if result.has_violations():
    print(f"Constraint violations: {result.constraint_violations}")

if result.has_validation_failures():
    print(f"Validation failures: {result.validation_failures}")
```

### Example 3: Analysis Mode

```python
from reliability_plugin.adapters.roma import RomaReliabilityAdapter

adapter = RomaReliabilityAdapter()

result = adapter.analyze_with_constraints(
    task="Optimize this SQL query for performance",
    analysis_type="decomposition",
    max_depth=2
)

if result.success:
    print(f"Analysis: {result.analysis}")
```

### Example 4: Verification and Critique

```python
from reliability_plugin.adapters.roma import RomaReliabilityAdapter

adapter = RomaReliabilityAdapter()

# Verify a solution
solution = "Use database indexes on foreign keys"
result = adapter.verify_with_constraints(
    solution=solution,
    original_task="Optimize SQL query performance",
    verification_criteria=["correctness", "completeness"]
)

# Critique from Red Team perspective
critique = adapter.critique_with_constraints(
    solution=solution,
    original_task="Optimize SQL query performance",
    critique_focus="security"
)
```

### Example 5: Health Check

```python
from reliability_plugin.adapters.roma import RomaReliabilityAdapter

adapter = RomaReliabilityAdapter()

# Get status
status = adapter.get_status()
print(f"ROMA available: {status['roma_available']}")
print(f"LMQL enabled: {status['lmql_available']}")
print(f"Guardrails enabled: {status['guardrails_available']}")

# Comprehensive health check
health = adapter.health_check()
print(f"Adapter healthy: {health['adapter_healthy']}")
print(f"Component status: {health['components']}")
```

## Error Handling

The adapter provides comprehensive error handling and graceful degradation:

```python
result = solve_with_constraints(task="Some task")

if not result.success:
    # Check error type
    if "Input validation failed" in result.error:
        print("Input doesn't meet safety requirements")
    elif "ROMA not available" in result.error:
        print("ROMA core is not available")
    elif "LMQL" in result.error:
        print("Constraint engine failed, but ROMA may have completed")
    else:
        print(f"Unexpected error: {result.error}")

# Check for partial success
if result.success and result.has_validation_failures():
    print("Task completed but with validation warnings")
```

## Logging

The adapter uses structured JSON logging with correlation IDs:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# All adapter operations log with correlation_id
result = solve_with_constraints(task="Test task")
print(f"Correlation ID: {result.correlation_id}")
```

## Troubleshooting

### ROMA Not Available

```python
status = adapter.get_status()
if not status['roma_available']:
    print("ROMA MCP tools not installed")
    print("Install from: ROMA/src/roma_dspy/")
```

### LMQL Not Available

```python
health = adapter.health_check()
if not health['components']['lmql']['healthy']:
    print("LMQL not available - constraints will not be applied")
    print("Install: pip install lmql")
```

### Guardrails Not Available

```python
health = adapter.health_check()
if not health['components']['guardrails']['healthy']:
    print("Guardrails not available - validation will be skipped")
    print("Install: pip install guardrails-ai")
```

## Integration with Unified Bridge

```python
from reliability.unified_bridge import UnifiedBridge

bridge = UnifiedBridge()

# ROMA adapter is automatically registered
result = bridge.process_with_reliability(
    task="Solve a problem",
    engine="roma",
    constraints={"max_depth": 3}
)
```

## File Structure

```
reliability-plugin/adapters/roma/
├── __init__.py                  # Package exports
├── roma_reliability_adapter.py  # Main adapter implementation
├── config.py                    # Configuration management
└── README.md                    # This file
```

## Contributing

When modifying this adapter:

1. **Never import from ROMA core source files**
2. **Only use ROMA MCP tools as public interface**
3. **Maintain graceful degradation** - adapter should work even if layers unavailable
4. **Add comprehensive logging** with correlation IDs
5. **Document all constraint types** in this README

## License

MIT License - See OpenEvolve project license

## Authors

OpenEvolve Team

## Version

1.0.0
