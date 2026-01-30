# CrewAI Zero-Error Workflow - Quick Reference

## Overview

The `crewai_zero_error_workflow.py` module provides comprehensive zero-error workflow orchestration for CrewAI with automatic error prevention, detection, and correction capabilities.

## Key Components

### 1. Core Classes

#### ZeroErrorWorkflow
Main orchestrator for executing workflows with zero-error handling.

**Key Features:**
- Multi-phase execution (Initialization → Validation → Preparation → Execution → Verification → Completion)
- Automatic error detection and classification
- Retry logic with exponential backoff
- Rollback on critical failures
- Comprehensive error reporting

**Usage:**
```python
from crewai_zero_error_workflow import (
    ZeroErrorWorkflow,
    create_workflow_definition,
    execute_workflow_zero_error
)

# Define workflow
workflow_def = create_workflow_definition(
    name="my_workflow",
    description="Example workflow",
    steps=[
        {
            "name": "validate",
            "action": "validation",
            "validations": ["check_complete"]
        },
        {
            "name": "process",
            "action": "data_processing",
            "operation": "transform",
            "source": "input"
        }
    ],
    input_schema={
        "type": "object",
        "properties": {
            "data": {"type": "array"}
        },
        "required": ["data"]
    },
    timeout_seconds=300,
    max_retries=3
)

# Execute workflow
result = await execute_workflow_zero_error(
    workflow_definition=workflow_def,
    inputs={"data": [1, 2, 3]},
    enable_auto_correction=True,
    strict_mode=False
)

print(f"Status: {result.status.value}")
print(f"Success Rate: {result.success_rate:.1f}%")
print(f"Errors: {len(result.errors)}")
```

### 2. Error Handling

#### Error Types
- **WorkflowImportError**: Missing dependencies
- **WorkflowConfigurationError**: Invalid workflow setup
- **WorkflowValidationError**: Input/state validation failures
- **WorkflowExecutionError**: Runtime execution failures
- **WorkflowTimeoutError**: Timeout exceeded
- **WorkflowResourceError**: Resource unavailable

#### Error Categories
- IMPORT: Missing modules
- CONFIGURATION: Setup issues
- VALIDATION: Input/output validation failures
- EXECUTION: Runtime errors
- TIMEOUT: Time limit exceeded
- RESOURCE: Missing resources
- DEPENDENCY: Failed dependencies
- LOGIC: Logical errors

#### Error Severity Levels
- CRITICAL: System cannot continue
- HIGH: Major issue compromising results
- MEDIUM: Recoverable with impact
- LOW: Minor workaround available
- INFO: Informational only

### 3. Workflow Phases

1. **INITIALIZATION**
   - Validate workflow definition
   - Check CrewAI availability
   - Initialize state management

2. **VALIDATION**
   - Validate inputs against schema
   - Auto-correct type mismatches
   - Check required fields

3. **PREPARATION**
   - Prepare execution context
   - Generate input hash
   - Setup CrewAI components

4. **EXECUTION**
   - Execute each step with retries
   - Monitor for errors
   - Apply corrections

5. **VERIFICATION**
   - Verify all steps completed
   - Validate outputs
   - Check success criteria

6. **COMPLETION**
   - Generate final results
   - Calculate success metrics
   - Create error reports

7. **ROLLBACK** (on failure)
   - Reverse completed steps
   - Restore state
   - Clean up resources

### 4. Step Types

#### validation
```python
{
    "name": "validate_data",
    "action": "validation",
    "validations": ["check_completeness", "check_accuracy"]
}
```

#### python_function
```python
{
    "name": "transform_data",
    "action": "python_function",
    "function": "transform_function",
    "parameters": {"mode": "strict"}
}
```

#### data_processing
```python
{
    "name": "process_data",
    "action": "data_processing",
    "operation": "transform",
    "source": "input"
}
```

#### crewai_crew
```python
{
    "name": "ai_analysis",
    "action": "crewai_crew",
    "task": {...},
    "agents": [...],
    "expected_output": "Analysis result"
}
```

### 5. Integration with ClaudeMiro Bridge

```python
from crewai_zero_error_workflow import ClaudeMiroWorkflowBridge

# Create workflow from bridge config
bridge_config = {
    "name": "claudiomiro_workflow",
    "tasks": [...],
    "input_schema": {...},
    "output_schema": {...}
}

result = await ClaudeMiroWorkflowBridge.execute_bridge_workflow(
    bridge_config=bridge_config,
    inputs={"data": [...]},
    state_manager=state_manager,
    enable_auto_correction=True
)
```

### 6. Error Correction Strategies

The module provides automatic correction for:

- **Missing Dependencies**: Suggests pip install commands
- **Type Mismatches**: Attempts type conversion
- **Timeout Issues**: Suggests timeout increases
- **Missing Environment Variables**: Suggests values
- **Invalid Parameters**: Suggests corrections

### 7. Error Reporting

```python
# Generate comprehensive error report
orchestrator = ZeroErrorWorkflow(definition=workflow_def)
await orchestrator.execute(inputs={"data": [...]})

report = orchestrator.generate_error_report()
print(json.dumps(report, indent=2))

# Report includes:
# - Total error count
# - Errors by category
# - Errors by severity
# - Auto-corrected errors
# - Detailed error records
```

### 8. Configuration Options

```python
ZeroErrorWorkflow(
    definition=workflow_def,
    crewai_state_manager=None,          # Optional state manager
    enable_auto_correction=True,         # Enable auto-correction
    strict_mode=True,                    # Fail fast on errors
    log_all_steps=True                   # Log all executions
)
```

### 9. Workflow Definition Options

```python
create_workflow_definition(
    name="workflow_name",
    steps=[...],
    description="Workflow description",
    version="1.0.0",
    input_schema={...},                  # JSON schema for inputs
    output_schema={...},                 # JSON schema for outputs
    validation_rules=[...],              # Additional validation rules
    timeout_seconds=300,                 # Execution timeout
    max_retries=3,                       # Retry attempts per step
    critical=True,                       # Critical workflow (rollback on failure)
    rollback_on_failure=True             # Enable rollback
)
```

## Best Practices

1. **Always define input/output schemas** for proper validation
2. **Enable auto-correction** in development for easier debugging
3. **Use strict_mode=True** in production for fail-fast behavior
4. **Set appropriate timeouts** based on expected execution time
5. **Configure max_retries** based on network reliability
6. **Mark critical workflows** to enable rollback on failure
7. **Review error reports** to identify recurring issues
8. **Test workflows** with various input combinations

## Testing

Run unit tests:
```bash
python -m unittest crewai_zero_error_workflow
```

Run example:
```bash
python crewai_zero_error_workflow.py
```

## Requirements

- Python 3.8+
- asyncio
- logging
- dataclasses
- typing
- unittest (for tests)
- Optional: crewai (for AI workflow execution)

## File Location

`C:\Users\mmeadow\Documents\OpenEvolve\Frontend\crewai_zero_error_workflow.py`

## Integration

This module integrates with:
- `claudiomiro_crewai_bridge.py` - Via ClaudeMiroWorkflowBridge class
- `crewai_state_management.py` - Via optional state_manager parameter
- CrewAI framework - Via lazy loading and crewai_crew steps

## Error Recovery Flow

```
Error Detected
    ↓
Categorize Error
    ↓
Assess Severity
    ↓
Attempt Auto-Correction (if enabled)
    ↓
Correction Successful?
    ↓ YES
    Retry Phase/Step
    ↓ NO
    Log Error
    ↓
    Check Strict Mode
    ↓ YES      ↓ NO
    Raise      Continue
    Error      Execution
```

## Support

For issues or questions:
1. Check error reports for detailed information
2. Review logs for execution traces
3. Enable debug logging for more details
4. Run unit tests to verify installation
