# MDAP Enhanced Red Flagging Integration Summary

## Overview

The MDAP reliability adapter has been successfully updated to integrate the enhanced red flagging system. This integration provides multi-layered validation and security for MDAP operations.

## Integration Location

**File:** `reliability-plugin/adapters/mdap/mdap_reliability_adapter.py`

## Key Components Added

### 1. Enhanced Red Flagging Imports

```python
from reliability.enhanced_redflagger import (
    EnhancedRedFlagger,
    EnhancedRedFlagRules,
    RedFlag,
    RedFlagSeverity,
    create_enhanced_redflagger
)
```

### 2. Initialization in `__init__` Method

- Added `enhanced_redflagger` attribute
- Added `enhanced_redflagging_enabled` flag
- Integrated LMQL adapter support
- Added new statistics tracking:
  - `enhanced_redflagging_used`
  - `red_flags_detected`

### 3. Core Methods Added

#### `_create_enhanced_redflagger()`
Creates and configures the enhanced red flagger with:
- LMQL adapter integration
- Guardrails adapter linkage
- Default red flag rules
- Configuration binding

#### `_create_default_redflag_rules()`
Creates default enhanced red flag rules with:
- Token limits (750 tokens, 6000 characters)
- Confidence thresholds (0.5)
- LMQL constraint configuration
- Guardrails validators:
  - toxic_language
  - pii_filter
  - secrets_detection
  - malicious_patterns
  - injection_check
  - json_structure
- Forbidden keywords list
- Format requirements (JSON)
- Security thresholds (toxicity 0.8, PII strict)

#### `solve_with_enhanced_redflagging()`
Main method for solving with enhanced red flagging. Implements multi-layered validation:

**Layer 1: Pre-generation (LMQL Constraints)**
- Generates LMQL constraints before execution
- Prevents flagged content from being generated
- Configurable via `use_lmql_constraints` parameter

**Layer 2: During Execution (Core Integration)**
- Executes MDAP solve with enhanced red flagger
- Applies LMQL constraints to orchestrator
- Validates votes during execution
- Falls back to MCP tools if core unavailable

**Layer 3: Post-generation (Comprehensive Validation)**
- Validates final result with enhanced red flagging
- Checks for red flags with severity categorization
- Marks results as failed if critical/high flags detected
- Returns detailed red flag information

#### `_solve_with_core_redflagging()`
Helper method that:
- Integrates enhanced red flagger with MDAP core
- Creates RedFlagger with enhanced rules
- Applies LMQL constraints to orchestrator
- Executes MDAP task with full validation

#### `_convert_to_dict_result()`
Converts MDAPSolveResult to dictionary with additional red flagging fields

#### `_extract_statistics()`
Extracts statistics from MDAP results

### 4. Convenience Functions

#### `solve_with_redflagging()`
One-off function for enhanced red flagging:
```python
result = solve_with_redflagging(
    task="Solve this problem",
    mdap_k_ahead=5,
    use_lmql_constraints=True,
    use_enhanced_validation=True
)
```

### 5. Updated Methods

#### `get_status()`
Enhanced to include:
- `enhanced_redflagging_available` status
- `lmql_available` status
- Enhanced red flagging layer status
- Configuration flags for both features

#### `reset_statistics()`
Updated to include new statistics fields

## Return Value Structure

The `solve_with_enhanced_redflagging()` method returns:

```python
{
    "success": bool,                      # Overall success status
    "result": Any,                        # MDAP result
    "task": str,                          # Original task
    "red_flags": List[Dict],              # Detected red flags
    "red_flag_count": int,                # Number of red flags
    "layers_used": List[str],             # Validation layers used
    "flagging_statistics": Dict,          # Red flagging statistics
    "metadata": {
        "method": "enhanced_redflagging",
        "lmql_constraints_used": int,     # Number of LMQL constraints
        "validation_enabled": bool,       # Whether validation was enabled
        "correlation_id": str             # Correlation ID for tracking
    }
}
```

## Red Flag Severity Handling

The integration supports three severity levels:
- **CRITICAL**: Immediately fails the result
- **HIGH**: Immediately fails the result
- **MEDIUM**: Logged but doesn't fail result
- **LOW**: Informational only

## Configuration

Enhanced red flagging can be configured via:
```python
config = SimpleNamespace(
    lmql_enabled=False,                      # Enable LMQL pre-generation
    enhanced_redflagging_enabled=True        # Enable enhanced red flagging
)
```

## Backward Compatibility

The integration maintains full backward compatibility:
- Existing methods (`solve_with_validation`, `solve_with_core_integration`) unchanged
- Enhanced red flagging is opt-in via new methods
- Graceful fallback if enhanced red flagging unavailable
- No breaking changes to existing API

## Error Handling

Comprehensive error handling includes:
- Graceful degradation if enhanced red flagging unavailable
- Fallback to standard core integration
- Detailed error logging with correlation IDs
- Exception handling at all layers

## Statistics Tracking

New statistics track:
- `enhanced_redflagging_used`: Number of solves using enhanced red flagging
- `red_flags_detected`: Total red flags detected across all solves

## Usage Examples

### Basic Usage
```python
adapter = MDAPReliabilityAdapter()
result = adapter.solve_with_enhanced_redflagging(
    task="Solve this complex problem",
    mdap_k_ahead=7
)
if result["success"]:
    print(f"Solution: {result['result']}")
if result["red_flags"]:
    print(f"Warnings: {result['red_flags']}")
```

### With LMQL Constraints
```python
result = adapter.solve_with_enhanced_redflagging(
    task="Generate a response",
    mdap_k_ahead=5,
    use_lmql_constraints=True,       # Enable pre-generation constraints
    use_enhanced_validation=True     # Enable post-generation validation
)
```

### Convenience Function
```python
from mdap_reliability_adapter import solve_with_redflagging

result = solve_with_redflagging(
    task="What is the capital of France?",
    mdap_k_ahead=3
)
```

## Dependencies

Required for enhanced red flagging:
- `reliability.enhanced_redflagger` module
- `reliability.lmql_adapter` (optional, for LMQL constraints)
- `reliability.guardrails_adapter` (for validation)

## Testing

The integration includes comprehensive error handling and should work with:
- MDAP core available (primary method)
- MCP tools fallback (secondary method)
- Enhanced red flagging available (best validation)
- Enhanced red flagging unavailable (graceful degradation)

## File Statistics

- **Lines Added**: ~400
- **Methods Added**: 7
- **Convenience Functions**: 1
- **Configuration Options**: 2 (lmql_enabled, enhanced_redflagging_enabled)

## Summary

This integration provides MDAP with enterprise-grade validation and security through:
1. **Pre-generation constraints** (LMQL)
2. **During-execution validation** (Enhanced red flagger)
3. **Post-generation comprehensive checking** (Red flag detection)

The multi-layered approach ensures safety at every stage of MDAP execution while maintaining backward compatibility and graceful degradation.
