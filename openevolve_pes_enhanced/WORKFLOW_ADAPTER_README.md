# Workflow PES Adapter

This adapter integrates PES Enhanced with the Workflow Engine, enabling cost-aware evolution and budget tracking for workflow execution.

## Overview

The `WorkflowPESAdapter` provides a **non-invasive** way to add PES Enhanced capabilities to workflow execution:

- **Cost-aware decomposition**: Allocate resources based on subproblem complexity
- **Budget tracking**: Monitor costs across workflow stages
- **Budget enforcement**: Stop workflow if budget exceeded
- **Drop-in replacement**: Use `run_sovereign_workflow_with_pes` instead of `run_sovereign_workflow`

## Quick Start

### Basic Usage

```python
from openevolve_pes_enhanced import (
    run_sovereign_workflow_with_pes,
    PESEnhancedConfig
)

# Create PES config
pes_config = PESEnhancedConfig.cost_aware(max_cost_usd=10.0)

# Use drop-in replacement for run_sovereign_workflow
result = await run_sovereign_workflow_with_pes(
    workflow_state=workflow_state,
    content_analyzer_team=content_analyzer_team,
    planner_team=planner_team,
    solver_team=solver_team,
    patcher_team=patcher_team,
    assembler_team=assembler_team,
    sub_problem_red_gauntlet=sub_problem_red_gauntlet,
    sub_problem_gold_gauntlet=sub_problem_gold_gauntlet,
    final_red_gauntlet=final_red_gauntlet,
    final_gold_gauntlet=final_gold_gauntlet,
    solver_generation_gauntlet=solver_generation_gauntlet,
    pes_config=pes_config,
    max_cost_usd=10.0  # Enable cost tracking
)

# Access cost metrics
if 'pes_cost_metrics' in result.metadata:
    metrics = result.metadata['pes_cost_metrics']
    print(f"Total cost: ${metrics['total_cost_usd']:.2f}")
    print(f"Evaluations saved: {metrics['evaluations_saved']}")
```

### Advanced Usage with Adapter Class

```python
from openevolve_pes_enhanced.workflow_adapter import WorkflowPESAdapter

# Create adapter
adapter = WorkflowPESAdapter(pes_config)

# Enhance decomposition with cost-aware allocation
allocations = adapter.enhance_decomposition_with_pes(
    subproblems=workflow_state.decomposition_plan.sub_problems,
    budget_per_problem=2.50
)

# Execute with PES tracking
result = await adapter.execute_workflow_with_pes(
    workflow_state=workflow_state,
    max_cost_usd=10.0,
    enable_cost_tracking=True,
    original_workflow_func=run_sovereign_workflow,
    # ... other workflow args
)
```

## Components

### WorkflowPESAdapter

Main adapter class that wraps workflow execution with PES Enhanced capabilities.

```python
class WorkflowPESAdapter:
    def __init__(self, pes_config: Optional[PESEnhancedConfig] = None)
    
    async def execute_workflow_with_pes(...)
    
    def enhance_decomposition_with_pes(
        self,
        subproblems: List[SubProblem],
        budget_per_problem: float
    ) -> List[Tuple[SubProblem, SubProblemAllocation]]
```

### CostAwareWorkflowTracker

Tracks costs throughout workflow execution.

```python
tracker = CostAwareWorkflowTracker(max_cost_usd=10.0)
tracker.start_stage("decomposition")
tracker.record_cost(1.50, stage="decomposition")
should_continue, reason = tracker.check_budget()
```

### AllocationDecision

Decision types for subproblem resource allocation:

- `FULL_EVOLUTION`: Full resources for complex subproblems
- `LIMITED_EVOLUTION`: Reduced resources for simpler subproblems
- `SKIP_EVOLUTION`: Skip expensive evolution
- `USE_CACHED`: Use cached solution if available
- `DEFER`: Defer to later (budget constraints)

## Configuration

### PESEnhancedConfig for Workflows

```python
from openevolve_pes_enhanced import create_cost_aware_workflow_config

config = create_cost_aware_workflow_config(
    max_cost_usd=20.0,
    enable_early_stopping=True,
    enable_cost_optimization=True
)
```

Or manually configure:

```python
from openevolve_pes_enhanced import PESEnhancedConfig

config = PESEnhancedConfig()
config.enable_cost_optimization = True
config.enable_early_stopping = True
config.cost.max_cost_usd = 20.0
config.cost.max_time_seconds = 3600
config.early_stopping.patience = 3
```

## Integration Points

The adapter hooks into workflow execution at these points:

1. **Decomposition Phase**: Allocates budget per subproblem based on complexity
2. **Solution Generation**: Wraps evolution calls with cost tracking
3. **Stage Transitions**: Tracks costs across workflow stages
4. **Budget Enforcement**: Stops workflow if budget exceeded

## Cost Metrics

The adapter tracks:

- `total_cost_usd`: Total cost of workflow execution
- `decomposition_cost`: Cost of decomposition stage
- `solution_generation_cost`: Cost of generating solutions
- `verification_cost`: Cost of verification/gauntlets
- `subproblem_costs`: Per-subproblem cost breakdown
- `evaluations_saved`: Efficiency gains from PES

Access via:

```python
metrics = result.metadata.get('pes_cost_metrics', {})
```

## Examples

See `examples/workflow_example.py` for complete examples:

1. **Basic adapter usage** with cost tracking
2. **Full workflow execution** with PES
3. **Budget enforcement** demonstration
4. **Stage-by-stage cost tracking**
5. **Integration patterns** with existing code

Run examples:

```bash
cd c:\Users\mmeadow\Documents\OpenEvolve\Frontend
set PYTHONPATH=.
python openevolve_pes_enhanced/examples/workflow_example.py
```

## API Reference

### run_sovereign_workflow_with_pes

Drop-in replacement for `run_sovereign_workflow` with PES cost tracking.

```python
async def run_sovereign_workflow_with_pes(
    workflow_state: WorkflowState,
    content_analyzer_team: Team,
    planner_team: Team,
    solver_team: Team,
    patcher_team: Team,
    assembler_team: Team,
    sub_problem_red_gauntlet: GauntletDefinition,
    sub_problem_gold_gauntlet: GauntletDefinition,
    final_red_gauntlet: GauntletDefinition,
    final_gold_gauntlet: GauntletDefinition,
    solver_generation_gauntlet: GauntletDefinition,
    max_refinement_loops: int = 3,
    pes_config: Optional[PESEnhancedConfig] = None,
    max_cost_usd: Optional[float] = None,
    enable_cost_tracking: bool = True,
) -> WorkflowState
```

### WorkflowStatePESExtension

Mixin for adding PES data to WorkflowState without modifying the class:

```python
from openevolve_pes_enhanced import WorkflowStatePESExtension

# Extend workflow state
WorkflowStatePESExtension.extend(workflow_state, pes_config)

# Access PES data
config = WorkflowStatePESExtension.get_pes_config(workflow_state)
budget = WorkflowStatePESExtension.get_budget_remaining(workflow_state)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Workflow Engine                          │
│              (workflow_engine.py - unchanged)               │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ run_sovereign_workflow()
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              WorkflowPESAdapter                             │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ CostAwareWorkflowTracker │  │ PESIntegrationWrapper      │
│  │ - Budget tracking        │  │ - Strategy selection       │
│  │ - Stage monitoring       │  │ - Cost optimization        │
│  │ - Enforcement            │  │ - Early stopping           │
│  └─────────────────┘  └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## Notes

- The adapter is **non-invasive**: `workflow_engine.py` is not modified
- Cost tracking is **opt-in**: Set `max_cost_usd` to enable
- Backward compatible: Without `max_cost_usd`, behavior matches original
- Budget enforcement can be disabled by not setting `max_cost_usd`
