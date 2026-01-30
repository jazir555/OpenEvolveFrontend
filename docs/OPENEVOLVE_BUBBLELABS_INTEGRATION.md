# OpenEvolve + BubbleLabs Integration Guide

**Date:** 2025-12-30
**Status:** ✅ **COMPLETE - PRODUCTION READY**

---

## Overview

This integration enables comprehensive workflow management by connecting **OpenEvolve** workflows with **BubbleLabs** visual workflow designer. It provides:

- ✅ Visual workflow creation and editing
- ✅ Real-time workflow execution monitoring
- ✅ Workflow parameter management
- ✅ Analytics and performance tracking
- ✅ State machine validation for workflow transitions
- ✅ Hephaestus integration for project management
- ✅ MCP tools for external agent control

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BubbleLabs UI                             │
│  (Visual Workflow Designer, Parameter Controls, Monitoring)  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         OpenEvolveWorkflowManager                            │
│  - Creates workflows from templates                          │
│  - Manages workflow execution                                │
│  - Tracks state and metrics                                  │
│  - Integrates with analytics and Hephaestus                 │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐    ┌──────────┐   ┌──────────┐
    │BubbleLabs│    │Analytics │   │Hephaestus│
    │Integration│  │  Database│   │  Bridge   │
    └─────────┘    └──────────┘   └──────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                  OpenEvolve Workflows                        │
│  (sovereign_decomposition, evolutionary_optimization, etc.)  │
└─────────────────────────────────────────────────────────────┘
```

---

## Files Created

1. **openevolve_workflow_manager.py** (~1000 lines)
   - Main workflow manager class
   - Template-based workflow creation
   - Workflow execution engine
   - State management and monitoring
   - Integration with analytics and Hephaestus

2. **openevolve_workflow_mcp_tools.py** (~600 lines)
   - MCP tools for external control
   - Thread-safe singleton manager
   - 9 MCP tool functions
   - Complete parameter validation

3. **OPENEVOLVE_BUBBLELABS_INTEGRATION.md** (this file)
   - Complete usage guide
   - API reference
   - Examples

---

## Quick Start

### 1. Basic Workflow Creation

```python
from openevolve_workflow_manager import OpenEvolveWorkflowManager, WorkflowTemplate

# Initialize manager
manager = OpenEvolveWorkflowManager(
    analytics_db_path='openevolve_analytics.db',
    enable_hephaestus=True
)

# Create workflow from template
workflow_id = manager.create_workflow_from_template(
    template=WorkflowTemplate.SOVEREIGN_DECOMPOSITION,
    name="My First Workflow",
    description="Solves complex problems using decomposition",
    parameters={
        'max_refinement_loops': 5,
        'team_size': 3
    }
)

print(f"Created workflow: {workflow_id}")
```

### 2. Execute Workflow

```python
# Execute with problem statement
result = manager.execute_workflow(
    workflow_id=workflow_id,
    problem_statement="How can we optimize distributed system performance?"
)

if result.success:
    print(f"Solution: {result.result}")
    print(f"Execution time: {result.execution_time}s")
    print(f"Tokens used: {result.tokens_used}")
else:
    print(f"Error: {result.error}")
```

### 3. Monitor Workflow

```python
# Get workflow status
status = manager.get_workflow_status(workflow_id)
print(f"Status: {status['status']}")
print(f"Progress: {status['progress']*100}%")
print(f"Current node: {status['current_node']}")

# Get analytics metrics
metrics = manager.get_workflow_metrics(workflow_id)
print(f"Metrics: {metrics}")
```

### 4. Control Workflow

```python
# Pause workflow
manager.pause_workflow(workflow_id)

# Resume workflow
manager.resume_workflow(workflow_id)

# Cancel workflow
manager.cancel_workflow(workflow_id)
```

---

## Workflow Templates

### Available Templates

| Template | Description | Use Case |
|----------|-------------|----------|
| `sovereign_decomposition` | Decompose problems into sub-problems, solve in parallel, assemble solution | Complex multi-faceted problems |
| `evolutionary_optimization` | Use evolutionary algorithms to iteratively optimize solutions | Optimization problems |
| `adversarial_testing` | Red teams attack solutions, blue teams defend | Robustness testing |
| `multi_team_gauntlet` | Run solutions through verification gauntlet | High-stakes solutions |
| `hybrid_decomposition` | Combine multiple decomposition methods | Very complex problems |

### Template Parameters

#### Sovereign Decomposition
```python
{
    'max_refinement_loops': 3,      # Max refinement iterations
    'team_size': 3,                  # Teams per role
    'use_gauntlet': True,            # Use gauntlet verification
    'decomposition_depth': 3         # Decomposition depth
}
```

#### Evolutionary Optimization
```python
{
    'max_iterations': 10,            # Max evolution iterations
    'population_size': 20,           # Population size
    'mutation_rate': 0.1,            # Mutation probability
    'crossover_rate': 0.8,           # Crossover probability
    'selection_method': 'tournament'  # Selection method
}
```

#### Adversarial Testing
```python
{
    'num_red_teams': 2,              # Number of red teams
    'num_blue_teams': 3,             # Number of blue teams
    'adversarial_rounds': 5,         # Adversarial rounds
    'confidence_threshold': 0.8      # Min confidence threshold
}
```

---

## MCP Tools

### Available MCP Tools

1. **create_openevolve_workflow** - Create workflow from template
2. **execute_openevolve_workflow** - Execute workflow
3. **get_openevolve_workflow_status** - Get workflow status
4. **get_openevolve_workflow_metrics** - Get workflow metrics
5. **list_openevolve_workflows** - List all workflows
6. **pause_openevolve_workflow** - Pause workflow
7. **resume_openevolve_workflow** - Resume workflow
8. **cancel_openevolve_workflow** - Cancel workflow
9. **get_workflow_templates** - Get template information

### MCP Tool Examples

```python
from openevolve_workflow_mcp_tools import (
    create_openevolve_workflow,
    execute_openevolve_workflow,
    get_openevolve_workflow_status
)

# Create workflow
result = create_openevolve_workflow(
    name="Optimization Workflow",
    template="evolutionary_optimization",
    description="Optimize system parameters",
    parameters='{"max_iterations": 15, "population_size": 30}'
)

workflow_id = result['workflow_id']

# Execute workflow
result = execute_openevolve_workflow(
    workflow_id=workflow_id,
    problem_statement="Optimize database query performance",
    wait_for_completion=True
)

# Check status
status = get_openevolve_workflow_status(workflow_id)
print(status)
```

---

## State Machine Validation

All workflow state transitions are validated using the state machine in `bubblelabs_hephaestus_bridge.py`:

### Valid States
- `created` - Workflow definition created
- `pending` - Workflow queued for execution
- `running` - Workflow currently executing
- `paused` - Workflow temporarily paused
- `stopping` - Workflow in process of stopping
- `stopped` - Workflow stopped (can be restarted)
- `completed` - Workflow finished successfully (terminal)
- `failed` - Workflow failed (can be retried)
- `cancelled` - Workflow cancelled by user (terminal)

### Valid Transitions

```
created → pending → running → completed
                  ↓         ↓
                paused    failed
                  ↓         ↓
               stopped ←────┘
                  ↓
               cancelled (terminal)
```

---

## Analytics Integration

### Enable Analytics

```python
manager = OpenEvolveWorkflowManager(
    analytics_db_path='openevolve_analytics.db'
)
```

### Analytics Tracked

- Workflow execution time
- Token usage per workflow
- Node execution metrics
- Provider costs (OpenAI, Anthropic, etc.)
- Success/failure rates
- Performance trends over time

### Query Analytics

```python
from bubblelabs_analytics import BubbleLabsAnalytics

analytics = BubbleLabsAnalytics('openevolve_analytics.db')

# Get workflow analytics
workflow_analytics = analytics.get_workflow_analytics(workflow_id)

# Get provider metrics
provider_metrics = analytics.get_provider_metrics(workflow_id)

# Get node-level metrics
node_metrics = analytics.get_node_metrics(workflow_id)
```

---

## Hephaestus Integration

### Enable Project Management

```python
from bubblelabs_hephaestus_bridge import BubbleLabsTicketConfig

hephaestus_config = BubbleLabsTicketConfig(
    auto_create_tickets=True,
    auto_update_progress=True,
    auto_close_on_completion=True,
    ticket_prefix="OE-",
    ticket_type="story",
    default_labels=["openevolve", "workflow"]
)

manager = OpenEvolveWorkflowManager(
    enable_hephaestus=True,
    hephaestus_config=hephaestus_config
)
```

### Hephaestus Features

- Automatic ticket creation when workflow starts
- Ticket status updates as workflow progresses
- Automatic ticket closure on completion
- Workflow-to-ticket mapping stored in database

---

## Event Callbacks

### Register Event Handlers

```python
def on_workflow_completed(data):
    print(f"Workflow {data['workflow_id']} completed!")
    print(f"Status: {data['status']}")
    print(f"Result: {data['result']}")

manager.register_event_callback('workflow_completed', on_workflow_completed)

# Available events:
# - workflow_completed
# - workflow_paused
# - workflow_resumed
# - workflow_cancelled
```

---

## Advanced Usage

### Custom Workflow Creation

```python
# Define custom nodes
nodes = [
    {
        'id': 'analyze',
        'type': 'processNode',
        'position': {'x': 300, 'y': 100},
        'data': {'label': 'Analyze Problem'}
    },
    {
        'id': 'optimize',
        'type': 'processNode',
        'position': {'x': 500, 'y': 100},
        'data': {'label': 'Optimize Solution'}
    }
]

# Define edges
edges = [
    {'id': 'e1', 'source': 'start', 'target': 'analyze'},
    {'id': 'e2', 'source': 'analyze', 'target': 'optimize'},
    {'id': 'e3', 'source': 'optimize', 'target': 'end'}
]

# Create custom workflow
workflow_id = manager.create_custom_workflow(
    name="Custom Optimization",
    description="Custom optimization workflow",
    workflow_type="evolution",
    nodes=nodes,
    edges=edges,
    parameters={'max_iterations': 20}
)
```

### Asynchronous Execution

```python
def completion_callback(result):
    print(f"Async execution completed: {result.status}")

# Execute asynchronously
instance_id = manager.execute_workflow_async(
    workflow_id=workflow_id,
    problem_statement="Solve complex problem",
    callback=completion_callback
)

print(f"Started async execution: {instance_id}")
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENEVOLVE_ANALYTICS_DB` | Path to analytics database | `openevolve_workflow_analytics.db` |
| `ENABLE_HEPHAESTUS` | Enable Hephaestus integration | `false` |

---

## Testing

### Run Tests

```bash
# Test workflow manager
python -m pytest test_openevolve_workflow_manager.py -v

# Test MCP tools
python -m pytest test_openevolve_mcp_tools.py -v

# Test integration
python -m pytest test_openevolve_bubblelabs_integration.py -v
```

---

## Troubleshooting

### Issue: Workflow not found

**Solution:** Ensure workflow was created successfully before executing

```python
# List workflows to verify
workflows = manager.list_workflows()
print([wf['id'] for wf in workflows])
```

### Issue: State transition not allowed

**Solution:** Check current status before attempting transition

```python
status = manager.get_workflow_status(workflow_id)
print(f"Current status: {status['status']}")

# Verify transition is valid
if status['status'] == 'running':
    manager.pause_workflow(workflow_id)  # Valid
else:
    print("Cannot pause - not in running state")
```

### Issue: Analytics not working

**Solution:** Ensure analytics database path is provided

```python
manager = OpenEvolveWorkflowManager(
    analytics_db_path='openevolve_analytics.db'  # Required for analytics
)
```

---

## Best Practices

1. **Always validate workflow exists before executing**
   ```python
   if manager.get_workflow_status(workflow_id):
       manager.execute_workflow(workflow_id, problem)
   ```

2. **Use async execution for long-running workflows**
   ```python
   manager.execute_workflow_async(workflow_id, problem)
   ```

3. **Enable analytics for production workflows**
   ```python
   manager = OpenEvolveWorkflowManager(analytics_db_path='analytics.db')
   ```

4. **Handle workflow events with callbacks**
   ```python
   manager.register_event_callback('workflow_completed', handler)
   ```

5. **Check state before transitions**
   ```python
   if validate_workflow_transition(current_status, new_status):
       # Transition is valid
   ```

---

## API Reference

### OpenEvolveWorkflowManager

#### Methods

- `create_workflow_from_template(template, name, description, parameters)` - Create workflow
- `create_custom_workflow(name, description, workflow_type, nodes, edges, parameters)` - Custom workflow
- `execute_workflow(workflow_id, problem_statement, **kwargs)` - Execute workflow
- `execute_workflow_async(workflow_id, problem_statement, callback, **kwargs)` - Async execute
- `get_workflow_status(workflow_id)` - Get status
- `get_workflow_metrics(workflow_id)` - Get metrics
- `list_workflows()` - List all workflows
- `pause_workflow(workflow_id)` - Pause workflow
- `resume_workflow(workflow_id)` - Resume workflow
- `cancel_workflow(workflow_id)` - Cancel workflow
- `register_event_callback(event_type, callback)` - Register callback

---

## Conclusion

The OpenEvolve + BubbleLabs integration provides comprehensive workflow management with:

✅ Visual workflow creation and editing
✅ Real-time execution monitoring
✅ State machine validation
✅ Analytics tracking
✅ Hephaestus integration
✅ MCP tools for external control
✅ Template-based workflows
✅ Custom workflow support

**Ready for production use.**

---

**Integration Date:** 2025-12-30
**Status:** ✅ **COMPLETE - PRODUCTION READY**

---

*End of Integration Guide*
