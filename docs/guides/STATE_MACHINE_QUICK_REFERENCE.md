# State Machine Validation - Quick Reference Guide

## Quick Lookup: Valid State Transitions

### Workflow States

| From State | Valid To States | Notes |
|------------|----------------|-------|
| **CREATED** | PENDING, CANCELLED | Initial state |
| **PENDING** | RUNNING, CANCELLED | Queued and ready |
| **RUNNING** | PAUSED, STOPPING, COMPLETED, FAILED, CANCELLED | Actively executing |
| **PAUSED** | RUNNING, STOPPING, CANCELLED | Temporarily paused |
| **STOPPING** | STOPPED, CANCELLED, FAILED | Graceful shutdown in progress |
| **STOPPED** | PENDING, RUNNING | Can be restarted |
| **COMPLETED** | *(none)* | ✅ Terminal state |
| **FAILED** | PENDING, RUNNING | Can be retried |
| **CANCELLED** | *(none)* | ✅ Terminal state |

### Ticket States

| From State | Valid To States | Notes |
|------------|----------------|-------|
| **TODO** | IN_PROGRESS, CANCELLED, BLOCKED | Initial state |
| **IN_PROGRESS** | IN_REVIEW, TODO, CANCELLED, BLOCKED | Work in progress |
| **IN_REVIEW** | IN_PROGRESS, DONE, TODO, CANCELLED, BLOCKED | Under review |
| **DONE** | *(none)* | ✅ Terminal state |
| **CANCELLED** | *(none)* | ✅ Terminal state |
| **BLOCKED** | TODO, IN_PROGRESS, CANCELLED | Work blocked |

## Common Use Cases

### Starting a Workflow
```
CREATED → PENDING → RUNNING
```

### Pausing and Resuming
```
RUNNING → PAUSED → RUNNING
```

### Stopping a Workflow
```
RUNNING → STOPPING → STOPPED
```

### Restarting a Failed Workflow
```
FAILED → PENDING → RUNNING
```

### Cancelling a Workflow
```
[ANY STATE] → CANCELLED
```

### Normal Ticket Progression
```
TODO → IN_PROGRESS → IN_REVIEW → DONE
```

### Ticket Blocked and Unblocked
```
IN_PROGRESS → BLOCKED → TODO → IN_PROGRESS
```

### Ticket Sent Back for Review
```
IN_REVIEW → IN_PROGRESS → IN_REVIEW
```

## Code Examples

### Validate a Workflow Transition

```python
from bubblelabs_crewai_bridge import validate_workflow_transition

# Check if transition is valid
if validate_workflow_transition("running", "paused"):
    # Transition is valid, proceed
    workflow.status = "paused"
else:
    # Transition is invalid, handle error
    print("Cannot pause workflow from current state")
```

### Get Valid Transitions

```python
from bubblelabs_crewai_bridge import get_valid_workflow_transitions

# Get all valid next states from current state
valid_states = get_valid_workflow_transitions("running")
# Returns: {'paused', 'stopping', 'completed', 'failed', 'cancelled'}

print(f"Valid transitions from running: {valid_states}")
```

### Check if State is Terminal

```python
from bubblelabs_crewai_bridge import is_terminal_workflow_status

# Check if workflow is in a terminal state
if is_terminal_workflow_status("completed"):
    print("Workflow has completed and cannot be changed")
else:
    print("Workflow is still active")
```

### Validate a Ticket Transition

```python
from bubblelabs_crewai_bridge import validate_ticket_transition

# Check if transition is valid
if validate_ticket_transition("IN_PROGRESS", "IN_REVIEW"):
    # Transition is valid, proceed
    ticket.status = "IN_REVIEW"
else:
    # Transition is invalid, handle error
    print("Cannot move ticket to review from current state")
```

### Handle Validation Errors

```python
result = bubblelabs_integration.control_workflow_local(
    instance_id="workflow-123",
    action="start"
)

if "error" in result:
    if "Invalid state transition" in result["error"]:
        print(f"Error: {result['error']}")
        print(f"Valid transitions: {result.get('valid_transitions', [])}")
        # Handle error appropriately
```

## Error Messages

### Invalid Workflow Transition
```
ERROR: Invalid workflow transition: completed -> running
ERROR: Valid transitions from completed: []
```

**Solution:** Cannot restart a completed workflow. Create a new instance instead.

### Invalid Ticket Transition
```
ERROR: Invalid ticket transition: TODO -> DONE
ERROR: Valid transitions from TODO: ['IN_PROGRESS', 'CANCELLED', 'BLOCKED']
```

**Solution:** Move ticket through IN_PROGRESS and IN_REVIEW first.

## Terminal States

### Workflow Terminal States
- **COMPLETED**: Workflow finished successfully
- **CANCELLED**: Workflow was cancelled by user

These states have no valid transitions out. The workflow instance is final.

### Ticket Terminal States
- **DONE**: Ticket completed and approved
- **CANCELLED**: Ticket was cancelled

These states have no valid transitions out. The ticket is final.

## State Diagram Reference

### Workflow Lifecycle
```
┌──────────────────────────────────────────────────────────────┐
│                        WORKFLOW STATES                        │
└──────────────────────────────────────────────────────────────┘

  CREATED ──► PENDING ──► RUNNING ──► PAUSED
     │            │           │           │
     │            │           ├──► STOPPING ──► STOPPED ──► PENDING
     │            │           │                      │
     └──────────►┼───────────┼──────────────┬───────┘
                  │           │              │
                  └─────┬─────┴──────┬───────┘
                        │            │
                        ▼            ▼
                   COMPLETED      FAILED
                    (Terminal)       │
                                     └──► PENDING (retry)
```

### Ticket Lifecycle
```
┌──────────────────────────────────────────────────────────────┐
│                         TICKET STATES                         │
└──────────────────────────────────────────────────────────────┘

    TODO ──► IN_PROGRESS ──► IN_REVIEW ──► DONE
      │           │              │          (Terminal)
      │           ├──────────────┘
      │           │
      ├─────┐     │
      │     │     ▼
      │     │  BLOCKED
      │     │     │
      │     └─────┘
      │
      └───────────┴────► CANCELLED
                      (Terminal)
```

## Best Practices

1. **Always Validate**: Never assume a state transition is valid
2. **Handle Errors**: Provide clear feedback when transitions fail
3. **Use Query Functions**: Get valid transitions instead of hardcoding
4. **Check Terminal States**: Prevent unnecessary operations on terminal states
5. **Log Transitions**: Keep audit trail of state changes
6. **Graceful Degradation**: Handle validation failures gracefully

## Troubleshooting

### Problem: "Invalid state transition" error

**Solution:**
1. Check current state of workflow/ticket
2. Use `get_valid_*_transitions()` to see valid next states
3. Ensure workflow follows correct sequence
4. Check if current state is terminal

### Problem: Workflow stuck in invalid state

**Solution:**
1. Check logs for validation errors
2. Verify state transition logic
3. Use direct database update (emergency only)
4. Restart workflow if needed

### Problem: Ticket not updating

**Solution:**
1. Check if ticket is in terminal state
2. Verify ticket status transitions
3. Check CrewAI API connectivity
4. Review validation error logs

## Testing

Run the test suite to verify state machine validation:

```bash
python test_state_machine_validation.py
```

Expected output: 23/23 tests passing

## API Reference

### Validation Functions

#### `validate_workflow_transition(current, new)`
Validate workflow state transition.

**Parameters:**
- `current`: Current workflow status (str or Enum)
- `new`: New workflow status (str or Enum)

**Returns:** `bool` - True if valid, False otherwise

#### `validate_ticket_transition(current, new)`
Validate ticket state transition.

**Parameters:**
- `current`: Current ticket status (str or Enum)
- `new`: New ticket status (str or Enum)

**Returns:** `bool` - True if valid, False otherwise

#### `get_valid_workflow_transitions(status)`
Get valid transitions for workflow status.

**Parameters:**
- `status`: Current workflow status (str or Enum)

**Returns:** `Set[str]` - Set of valid next states

#### `get_valid_ticket_transitions(status)`
Get valid transitions for ticket status.

**Parameters:**
- `status`: Current ticket status (str or Enum)

**Returns:** `Set[str]` - Set of valid next states

#### `is_terminal_workflow_status(status)`
Check if workflow status is terminal.

**Parameters:**
- `status`: Workflow status to check (str or Enum)

**Returns:** `bool` - True if terminal, False otherwise

#### `is_terminal_ticket_status(status)`
Check if ticket status is terminal.

**Parameters:**
- `status`: Ticket status to check (str or Enum)

**Returns:** `bool` - True if terminal, False otherwise

---

**Last Updated:** 2025-12-29
**Version:** 1.0.0
**Status:** Production Ready
