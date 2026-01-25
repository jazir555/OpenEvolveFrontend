# MDAP Reliability Adapter

A production-ready adapter that adds Guardrails validation to MDAP voting **WITHOUT modifying MDAP core code**.

## Architecture

### Air Gap Principle

This adapter strictly follows the Air Gap principle:

- **NO imports from MDAP core source files** - Only imports MDAP MCP tools (public interface)
- **NO modifications to MDAP core files** - All validation logic lives in the adapter
- **Read-Only Core** - MDAP core files remain completely untouched
- **MCP Tool Interface** - Uses existing `solve_with_roma_mdap_maker` interface

### Wrapper Pattern

```
[User Request]
       ↓
[MDAP Reliability Adapter]
       ↓
┌──────────────────────────┐
│  Layer 1: Input Validation│ ← Guardrails
├──────────────────────────┤
│  Layer 2: MDAP Core Call  │ ← Unmodified (via MCP)
├──────────────────────────┤
│  Layer 3: Output Validation│ ← Guardrails
└──────────────────────────┘
       ↓
[Validated Result]
```

## Features

### 1. Input Validation

- Task format validation
- Parameter range checking (mdap_k_ahead: 2-20)
- Injection attack prevention
- Guardrails validation

### 2. Vote-Level Validation

- Individual vote validation during MDAP execution
- JSON structure validation
- Required field checking
- Malicious pattern detection
- Automatic remediation

### 3. Output Validation

- Result structure validation
- Vote consistency checking
- Statistics tracking
- Comprehensive logging

### 4. Graceful Degradation

- Works even if Guardrails unavailable
- Fallback validation logic
- Clear error messages
- Statistics tracking

## Installation

The adapter is part of the reliability plugin:

```bash
# No additional installation needed
# Reliability plugin is part of OpenEvolve Frontend
```

## Usage

### Basic Usage

```python
from reliability_plugin.adapters.mdap import MDAPReliabilityAdapter

# Create adapter
adapter = MDAPReliabilityAdapter()

# Solve with validation
result = adapter.solve_with_validation(
    task="What is 2 + 2?",
    mdap_k_ahead=5,
    validators=["vote_format", "json_structure", "required_fields"]
)

# Check result
if result.success:
    print(f"Solution: {result.result}")
    print(f"Statistics: {result.statistics}")
else:
    print(f"Error: {result.error}")
    print(f"Failures: {result.validation_failures}")
```

### Vote Validation

```python
from reliability_plugin.adapters.mdap import MDAPReliabilityAdapter

adapter = MDAPReliabilityAdapter()

# Validate individual vote
vote = {"decision": "APPROVE", "confidence": 0.9}

validation = adapter.verify_vote(
    vote=vote,
    validators=["vote_format", "json_structure"]
)

if validation.is_valid:
    print(f"Valid vote: {validation.vote}")
else:
    print(f"Invalid vote: {validation.failures}")
    if validation.remediated:
        print(f"Remediated to: {validation.vote}")
```

### Convenience Function

```python
from reliability_plugin.adapters.mdap import solve_with_guardrails

# One-off solve
result = solve_with_guardrails(
    task="Solve this problem",
    mdap_k_ahead=3,
    validators=["vote_format", "json_structure"]
)

if result.success:
    print(f"Success: {result.result}")
```

## Configuration

### Environment Variables

```bash
# Enable/disable Guardrails
export GUARDRAILS_ENABLED=true

# Specify validators
export GUARDRAILS_VALIDATORS="vote_format,json_structure,required_fields"

# Set remediation strategy
export GUARDRAILS_ON_FAIL="fix"

# Set max retries
export GUARDRAILS_MAX_RETRIES="3"

# Set timeout
export GUARDRAILS_TIMEOUT="30"
```

### Runtime Configuration

```python
from reliability.config import update_config

# Update Guardrails configuration
update_config({
    "guardrails": {
        "enabled": True,
        "validators": ["vote_format", "json_structure"],
        "on_fail": "fix",
        "max_retries": 5
    }
})
```

## API Reference

### MDAPReliabilityAdapter

#### `__init__(config: Optional[ReliabilityConfig] = None)`

Initialize adapter with optional configuration.

#### `solve_with_validation(task, mdap_k_ahead=5, validators=None, **kwargs) -> MDAPSolveResult`

Solve task using MDAP with Guardrails validation.

**Parameters:**
- `task` (str): The task to solve
- `mdap_k_ahead` (int): Number of agents for voting (2-20)
- `validators` (List[str]): List of validators to apply
- `**kwargs`: Additional arguments for MDAP

**Returns:**
- `MDAPSolveResult`: Result with validation outcome

#### `verify_vote(vote, validators=None, correlation_id=None) -> VoteValidationResult`

Validate an individual MDAP vote.

**Parameters:**
- `vote` (Any): The vote to validate
- `validators` (List[str]): List of validators to apply
- `correlation_id` (str): Optional correlation ID for logging

**Returns:**
- `VoteValidationResult`: Validation result

#### `get_status() -> Dict[str, Any]`

Get adapter status and health.

**Returns:**
- `Dict`: Status information

#### `get_statistics() -> Dict[str, int]`

Get adapter statistics.

**Returns:**
- `Dict`: Statistics counters

### Data Classes

#### `MDAPSolveResult`

```python
@dataclass
class MDAPSolveResult:
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    statistics: Dict[str, int] = field(default_factory=dict)
    validation_failures: List[str] = field(default_factory=list)
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
```

#### `VoteValidationResult`

```python
@dataclass
class VoteValidationResult:
    is_valid: bool
    vote: Any
    failures: List[str] = field(default_factory=list)
    remediated: bool = False
    original_vote: Any = None
    validator_name: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
```

## Validators

### Available Validators

The adapter uses Guardrails validators from the pre-configured library:

- **vote_format**: Validates MDAP vote is exactly two words
- **vote_json**: Validates MDAP vote is valid JSON
- **vote_id**: Validates MDAP vote ID format (e.g., A01, B23)
- **vote_decision**: Validates MDAP vote decision is in allowed list
- **json_structure**: Validates JSON structure
- **required_fields**: Validates required fields are present
- **malicious_patterns**: Detects malicious patterns
- **toxic_language**: Detects toxic language
- **pii_filter**: Detects and redacts PII

### Custom Validators

```python
from reliability.guardrails_adapter import GuardrailsAdapter

# Create adapter
base_adapter = GuardrailsAdapter()

# Register custom validator
base_adapter.register_validator(
    "custom_vote_validator",
    RegexMatch,
    regex=r"^[A-Z]{2}\d{4}$",
    on_fail="fix"
)

# Use with MDAP adapter
mdap_adapter = MDAPReliabilityAdapter()
result = mdap_adapter.solve_with_validation(
    task="Solve this",
    validators=["custom_vote_validator"]
)
```

## Error Handling

### Graceful Degradation

The adapter degrades gracefully when components are unavailable:

1. **Guardrails Unavailable**
   - Falls back to basic validation
   - Continues execution with warnings
   - Logs degraded mode

2. **MDAP Unavailable**
   - Returns error with clear message
   - Does not crash
   - Logs error details

3. **Validation Failures**
   - Attempts remediation (if configured)
   - Returns detailed failure information
   - Continues with partial results if possible

### Error Messages

```python
# Input validation error
{
    "success": False,
    "error": "Input validation failed: mdap_k_ahead must be 2-20, got 25",
    "validation_failures": ["mdap_k_ahead out of range"]
}

# MDAP execution error
{
    "success": False,
    "error": "MDAP MCP tools not available"
}

# Output validation error (with remediation)
{
    "success": True,
    "result": {...},
    "statistics": {
        "remediated_votes": 1
    },
    "validation_failures": ["Invalid JSON structure"]
}
```

## Logging

The adapter uses structured JSON logging:

```python
{
    "timestamp": "2025-01-10T12:00:00.000Z",
    "level": "INFO",
    "logger": "mdap_reliability_adapter",
    "message": "Starting MDAP solve with validation",
    "correlation_id": "mdap_1704883200.123",
    "task_length": 100,
    "mdap_k_ahead": 5,
    "validators": ["vote_format", "json_structure"]
}
```

## Testing

### Unit Tests

```python
import pytest
from reliability_plugin.adapters.mdap import MDAPReliabilityAdapter

def test_adapter_initialization():
    adapter = MDAPReliabilityAdapter()
    assert adapter is not None
    status = adapter.get_status()
    assert "mdap_available" in status

def test_solve_with_validation():
    adapter = MDAPReliabilityAdapter()
    result = adapter.solve_with_validation(
        task="Test task",
        mdap_k_ahead=3
    )
    assert result is not None
    assert isinstance(result.success, bool)

def test_vote_validation():
    adapter = MDAPReliabilityAdapter()
    vote = {"decision": "APPROVE"}
    validation = adapter.verify_vote(vote)
    assert validation is not None
    assert isinstance(validation.is_valid, bool)
```

### Integration Tests

```python
def test_full_workflow():
    adapter = MDAPReliabilityAdapter()

    # Check status
    status = adapter.get_status()
    assert status["mdap_available"]

    # Solve task
    result = adapter.solve_with_validation(
        task="What is 2 + 2?",
        mdap_k_ahead=3,
        validators=["vote_format", "json_structure"]
    )

    # Check result
    assert result.success
    assert result.result is not None

    # Check statistics
    assert result.statistics["total_votes"] > 0
```

## Performance

### Overhead

- **Input Validation**: ~10-50ms
- **Output Validation**: ~20-100ms
- **Vote Validation**: ~5-20ms per vote

### Optimization Tips

1. **Enable Caching**: Reduce repeated validations
2. **Adjust Validators**: Use only necessary validators
3. **Parallel Validation**: Enable parallel validation in Guardrails
4. **Timeout Configuration**: Set appropriate timeouts

## Troubleshooting

### MDAP Tools Not Available

```python
# Check status
status = adapter.get_status()
if not status["mdap_available"]:
    print("MDAP MCP tools not found")
    print("Install: Check roma_mdap_maker_mcp_tools.py")
```

### Guardrails Not Available

```python
# Check status
status = adapter.get_status()
if not status["guardrails_available"]:
    print("Guardrails not installed")
    print("Install: pip install guardrails-ai")
```

### Validation Failures

```python
# Check failures
result = adapter.solve_with_validation(...)
if result.validation_failures:
    print("Validation failures:")
    for failure in result.validation_failures:
        print(f"  - {failure}")
```

## Contributing

When contributing to this adapter:

1. **Maintain Air Gap**: Never import from MDAP core files
2. **Use MCP Tools**: Only use public MCP tool interfaces
3. **Add Tests**: Include unit and integration tests
4. **Update Docs**: Update README and docstrings
5. **Follow Patterns**: Follow existing code patterns

## License

MIT License - See LICENSE file for details

## Authors

OpenEvolve Team

## Version

1.0.0
