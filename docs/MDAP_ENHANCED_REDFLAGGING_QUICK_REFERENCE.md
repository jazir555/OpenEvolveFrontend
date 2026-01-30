# MDAP Enhanced Red Flagging - Quick Reference

## Quick Start

```python
from reliability_plugin.adapters.mdap.mdap_reliability_adapter import solve_with_redflagging

# Solve with enhanced red flagging
result = solve_with_redflagging(
    task="Your task here",
    mdap_k_ahead=5
)

if result["success"]:
    print(f"Result: {result['result']}")
```

## Main API Methods

### 1. `solve_with_enhanced_redflagging()` - Primary Method
```python
adapter = MDAPReliabilityAdapter()
result = adapter.solve_with_enhanced_redflagging(
    task="Solve this problem",
    mdap_k_ahead=5,
    use_lmql_constraints=True,
    use_enhanced_validation=True
)
```

### 2. `solve_with_redflagging()` - Convenience Function
```python
from reliability_plugin.adapters.mdap.mdap_reliability_adapter import solve_with_redflagging

result = solve_with_redflagging(
    task="Solve this problem",
    mdap_k_ahead=5
)
```

## Return Value Structure

```python
{
    "success": bool,                      # Was the solve successful?
    "result": Any,                        # The actual result
    "task": str,                          # Original task
    "red_flags": List[Dict],              # Any red flags detected
    "red_flag_count": int,                # Number of red flags
    "layers_used": List[str],             # Which validation layers were used
    "flagging_statistics": Dict,          # Red flagging statistics
    "metadata": {
        "method": str,                    # "enhanced_redflagging"
        "lmql_constraints_used": int,     # Number of LMQL constraints
        "validation_enabled": bool,       # Was validation enabled?
        "correlation_id": str             # For tracking
    }
}
```

## Red Flag Severity Levels

- **CRITICAL**: Result fails immediately
- **HIGH**: Result fails immediately
- **MEDIUM**: Warning only, result passes
- **LOW**: Informational only

## Configuration

```python
from types import SimpleNamespace

config = SimpleNamespace(
    lmql_enabled=False,                      # LMQL pre-generation
    enhanced_redflagging_enabled=True        # Enhanced validation
)

adapter = MDAPReliabilityAdapter(config=config)
```

## Status Checking

```python
adapter = create_mdap_adapter()
status = adapter.get_status()

# Check availability
print(f"Enhanced Red Flagging: {status['enhanced_redflagging_available']}")
print(f"LMQL: {status['lmql_available']}")

# Check layers
print(f"Layers: {list(status['layers'].keys())}")
```

## Statistics

```python
stats = adapter.get_statistics()

print(f"Enhanced red flagging used: {stats['enhanced_redflagging_used']}")
print(f"Red flags detected: {stats['red_flags_detected']}")
```

## Common Use Cases

### Basic Usage
```python
result = solve_with_redflagging(
    task="What is 2 + 2?",
    mdap_k_ahead=3
)
```

### With LMQL Constraints
```python
result = adapter.solve_with_enhanced_redflagging(
    task="Generate a response",
    mdap_k_ahead=5,
    use_lmql_constraints=True
)
```

### Disable Validation
```python
result = adapter.solve_with_enhanced_redflagging(
    task="Quick task",
    mdap_k_ahead=3,
    use_enhanced_validation=False
)
```

### With Schema Validation
```python
result = adapter.solve_with_enhanced_redflagging(
    task="Generate structured data",
    mdap_k_ahead=7,
    schema=output_schema
)
```

## Error Handling

```python
result = adapter.solve_with_enhanced_redflagging(task="...")

if not result["success"]:
    print(f"Error: {result.get('error')}")
    if result["red_flags"]:
        print(f"Red flags: {result['red_flags']}")
```

## Checking Red Flags

```python
if result["red_flags"]:
    print(f"Found {result['red_flag_count']} red flags:")
    for flag in result["red_flags"]:
        print(f"  - {flag['type']}: {flag['message']} (severity: {flag['severity']})")
```

## Layers Used

```python
print(f"Validation layers used: {result['layers_used']}")
# Output: ['lmql_pre_generation', 'mdap_core', 'enhanced_redflagging']
```

## Statistics Access

```python
print(f"Flagging stats: {result['flagging_statistics']}")
```

## Default Red Flag Rules

- **Token limit**: 750 tokens
- **Character limit**: 6000 characters
- **Confidence threshold**: 0.5
- **Toxicity threshold**: 0.8
- **PII detection**: Strict mode
- **Format**: JSON required
- **Validators**:
  - toxic_language
  - pii_filter
  - secrets_detection
  - malicious_patterns
  - injection_check
  - json_structure

## Backward Compatibility

All existing methods still work:
```python
# Original methods (unchanged)
adapter = MDAPReliabilityAdapter()
result = adapter.solve_with_validation(task="...", mdap_k_ahead=5)
result = adapter.solve_with_core_integration(task="...", mdap_k_ahead=5)
result = adapter.solve_with_mcp_tools(task="...", mdap_k_ahead=5)
```

## Performance Tips

1. **Disable LMQL** if not needed: `use_lmql_constraints=False`
2. **Lower k_ahead** for faster results: `mdap_k_ahead=3`
3. **Reuse adapter** instance across multiple calls
4. **Disable validation** for trusted content: `use_enhanced_validation=False`

## Troubleshooting

### Enhanced Red Flagging Unavailable
```python
if not status['enhanced_redflagging_available']:
    print("Enhanced red flagging not installed")
    print("Falling back to standard methods")
```

### LMQL Not Working
```python
if not status['lmql_available']:
    print("LMQL adapter not installed")
    print("Pre-generation constraints disabled")
```

### Too Many Red Flags
```python
# Adjust rules
adapter = create_mdap_adapter()
rules = adapter._create_default_redflag_rules()
rules.toxicity_threshold = 0.9  # More lenient
```

## Import Options

```python
# Option 1: Import class
from reliability_plugin.adapters.mdap.mdap_reliability_adapter import MDAPReliabilityAdapter

# Option 2: Import factory function
from reliability_plugin.adapters.mdap.mdap_reliability_adapter import create_mdap_adapter

# Option 3: Import convenience function
from reliability_plugin.adapters.mdap.mdap_reliability_adapter import solve_with_redflagging

# Option 4: Import all
from reliability_plugin.adapters.mdap.mdap_reliability_adapter import *
```

## Key Points

- ✅ Fully backward compatible
- ✅ Graceful fallback if unavailable
- ✅ Multi-layered validation
- ✅ Detailed red flag reporting
- ✅ Statistics tracking
- ✅ Production ready

## Documentation

- **Complete Guide**: `MDAP_ENHANCED_REDFLAGGING_INTEGRATION.md`
- **Status Document**: `MDAP_ENHANCED_REDFLAGGING_UPDATE_COMPLETE.md`
- **Test Suite**: `test_mdap_enhanced_integration.py`

## Support

For issues or questions:
1. Check the complete integration guide
2. Run the test suite to verify installation
3. Check adapter status with `get_status()`
4. Review statistics with `get_statistics()`
