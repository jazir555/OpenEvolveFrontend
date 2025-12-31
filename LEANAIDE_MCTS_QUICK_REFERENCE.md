# LeanAide MCTS Workflow Integration - Quick Reference

## Quick Start

```python
from leanaide_mcts_workflow import (
    MCTSWorkflowIntegrator,
    MCTSWorkflowConfig,
    MCTSStrategy,
    solve_with_mcts_approach
)
import asyncio

# Basic usage
async def main():
    config = MCTSWorkflowConfig(
        lean_mcts_enabled=True,
        lean_mcts_strategy=MCTSStrategy.ADAPTIVE
    )

    integrator = MCTSWorkflowIntegrator(config=config)
    solution = await integrator.solve_with_mcts(sub_problem)

asyncio.run(main())
```

## Main Classes

### MCTSWorkflowIntegrator
**Purpose**: Main integration class for MCTS in workflow

**Key Methods**:
- `solve_with_mcts(subproblem, config)` - Solve with MCTS
- `mcts_stage3a(subproblem)` - Stage 3A: Initial proof search
- `mcts_stage3b(solution)` - Stage 3B: Proof refinement
- `analyze_search_space(subproblem)` - Check if MCTS applicable
- `configure_mcts_from_workflow(state)` - Extract config from state

### MCTSSubProblemSolver
**Purpose**: Specialized solver using MCTS

**Key Methods**:
- `can_solve_with_mcts(subproblem)` - Detect applicability
- `solve_with_mcts(subproblem)` - Solve sub-problem
- `extract_proof_from_mcts(result)` - Convert to LeanProof
- `create_solution_attempt(proof, subproblem)` - Create SolutionAttempt

### MCTSProofRefiner
**Purpose**: Refine existing proofs with MCTS

**Key Methods**:
- `refine_proof(proof, iterations)` - Refine proof
- `expand_partial_proof(proof)` - Expand partial proof
- `initialize_tree_from_proof(proof)` - Initialize tree from proof

### MCTSWorkflowMonitor
**Purpose**: Real-time monitoring of MCTS progress

**Key Methods**:
- `start_monitoring(mcts, sub_problem_id)` - Start monitoring
- `update_progress(...)` - Update statistics
- `get_progress(sub_problem_id)` - Get progress dict
- `get_statistics(sub_problem_id)` - Get detailed stats
- `should_early_terminate(sub_problem_id)` - Check termination

## Configuration

### MCTSWorkflowConfig

```python
config = MCTSWorkflowConfig(
    # Enablement
    lean_mcts_enabled=True,
    lean_mcts_strategy=MCTSStrategy.ADAPTIVE,

    # Core parameters
    lean_mcts_iterations=1000,
    lean_mcts_time_budget=300.0,
    lean_mcts_c_param=1.414,
    lean_mcts_rollout_policy=MCTSRolloutType.LEANAIDE,
    lean_mcts_parallel_simulations=4,

    # Refinement
    lean_mcts_refinement_iterations=100,
    lean_mcts_refinement_depth=20,

    # Verification
    lean_mcts_verification_confidence=0.8,

    # Fallback
    lean_mcts_fallback_to_evolution=True,
    lean_mcts_fallback_to_standard=True,

    # Integration
    lean_mcts_auto_detect_applicable=True,
    lean_mcts_store_patterns=True
)
```

## WorkflowState Integration

### Add to WorkflowState

```python
from leanaide_mcts_workflow import add_mcts_config_to_workflow_state

workflow_state = add_mcts_config_to_workflow_state(
    workflow_state,
    config
)
```

### Extract from WorkflowState

```python
from leanaide_mcts_workflow import extract_mcts_config_from_workflow_state

config = extract_mcts_config_from_workflow_state(workflow_state)
```

## WorkflowState Parameters

```python
workflow_state.openevolve_parameters = {
    "lean_mcts_enabled": True,
    "lean_mcts_strategy": "adaptive",
    "lean_mcts_iterations": 1000,
    "lean_mcts_time_budget": 300.0,
    "lean_mcts_c_param": 1.414,
    "lean_mcts_rollout_policy": "leanaide",
    "lean_mcts_parallel_simulations": 4
}
```

## Stage Integration

### Stage 3A: Initial Proof Search

```python
from leanaide_mcts_workflow import MCTSWorkflowIntegrator

integrator = MCTSWorkflowIntegrator(config=config, workflow_state=state)
solution = await integrator.mcts_stage3a(sub_problem)
```

### Stage 3B: Proof Refinement

```python
refined_solution = await integrator.mcts_stage3b(solution)
```

### Stage 3C: Verification

```python
from leanaide_mcts_workflow import verify_sub_problem_with_leanaide_mcts

verification = await verify_sub_problem_with_leanaide_mcts(
    sub_problem,
    solution_attempt,
    workflow_state
)
```

## Strategies

### MCTSStrategy Options

- `STANDARD` - Basic MCTS
- `UCT` - UCB1 for Trees
- `THOMPSON_SAMPLING` - Thompson sampling
- `HYBRID_EVOLUTION` - MCTS + Genetic Algorithm
- `HYBRID_ADVERSARIAL` - MCTS + Adversarial
- `ADAPTIVE` - Adaptive strategy selection (recommended)

### When to Use Each

| Strategy | Best For | Performance |
|----------|----------|-------------|
| STANDARD | General use | Medium |
| HYBRID_EVOLUTION | Complex proofs | High |
| HYBRID_ADVERSARIAL | Robustness | High |
| ADAPTIVE | Unknown/Variable | Optimal |

## Search Space Analysis

```python
search_space = integrator.analyze_search_space(sub_problem)

# Check results
print(f"Branching factor: {search_space.branching_factor}")
print(f"Estimated depth: {search_space.estimated_depth}")
print(f"Has heuristics: {search_space.has_heuristics}")
print(f"Applicability score: {search_space.calculate_applicability_score():.2f}")
print(f"Is applicable: {search_space.is_applicable}")
```

## Monitoring Example

```python
from leanaide_mcts_workflow import MCTSWorkflowMonitor

monitor = MCTSWorkflowMonitor(config)
monitor.start_monitoring(mcts, "sp_001")

# During search
monitor.update_progress(
    sub_problem_id="sp_001",
    iteration=100,
    best_score=0.85,
    current_best_proof="...",
    tree_size=500,
    nodes_explored=250
)

# Check termination
if monitor.should_early_terminate("sp_001"):
    # Stop early
    pass

# Get statistics
stats = monitor.get_statistics("sp_001")
progress = monitor.get_progress("sp_001")
```

## Convenience Functions

```python
from leanaide_mcts_workflow import solve_with_mcts_approach

# Quick solve
solution = await solve_with_mcts_approach(
    sub_problem,
    workflow_state,
    config
)
```

## Error Handling

```python
try:
    solution = await integrator.solve_with_mcts(sub_problem)
except Exception as e:
    logger.error(f"MCTS failed: {e}")

    # Automatic fallback if enabled
    if config.lean_mcts_fallback_to_evolution:
        # Falls back to evolution
        pass
```

## Common Patterns

### Adaptive MCTS with Fallback

```python
config = MCTSWorkflowConfig(
    lean_mcts_strategy=MCTSStrategy.ADAPTIVE,
    lean_mcts_fallback_to_evolution=True,
    lean_mcts_fallback_to_standard=True
)

integrator = MCTSWorkflowIntegrator(config=config)
solution = await integrator.solve_with_mcts(sub_problem)
# Automatically selects best strategy or falls back
```

### MCTS + Evolution Hybrid

```python
config = MCTSWorkflowConfig(
    lean_mcts_strategy=MCTSStrategy.HYBRID_EVOLUTION,
    lean_mcts_iterations=500,
    lean_mcts_evolution_generations=20
)

# MCTS finds promising regions, evolution refines them
solution = await integrator.solve_with_mcts(sub_problem)
```

### Proof Refinement

```python
from leanaide_mcts_workflow import MCTSProofRefiner

refiner = MCTSProofRefiner(config)
refined_proof = await refiner.refine_proof(
    proof,
    iterations=100
)
```

## Quick Decision Tree

```
Is proof search space large?
├─ Yes → Use MCTS
│  ├─ High branching factor → Pure MCTS
│  ├─ Medium branching factor → Hybrid MCTS + Evolution
│  └─ Unknown → Adaptive MCTS
└─ No → Use standard approach
```

## Performance Tuning

### Speed vs Quality

**Faster** (Lower quality):
- `lean_mcts_iterations = 100`
- `lean_mcts_time_budget = 30.0`
- `lean_mcts_parallel_simulations = 8`

**Balanced**:
- `lean_mcts_iterations = 500`
- `lean_mcts_time_budget = 120.0`
- `lean_mcts_parallel_simulations = 4`

**Thorough** (Higher quality):
- `lean_mcts_iterations = 1000`
- `lean_mcts_time_budget = 300.0`
- `lean_mcts_parallel_simulations = 2`

### Exploration vs Exploitation

**More Exploration** (find diverse proofs):
- `lean_mcts_c_param = 2.0`

**Balanced**:
- `lean_mcts_c_param = 1.414` (√2)

**More Exploitation** (refine known good paths):
- `lean_mcts_c_param = 1.0`

## Key Benefits

1. ✅ Seamless workflow integration
2. ✅ Automatic strategy selection
3. ✅ Real-time monitoring
4. ✅ Robust fallback mechanisms
5. ✅ Knowledge learning via ACE
6. ✅ Hybrid capabilities
7. ✅ Production-ready error handling

## File Locations

- **Main Integration**: `leanaide_mcts_workflow.py`
- **MCTS Strategies**: `leanaide_mcts_strategies.py`
- **Documentation**: `LEANAIDE_MCTS_WORKFLOW_INTEGRATION.md`
- **This Guide**: `LEANAIDE_MCTS_QUICK_REFERENCE.md`

## Support

For issues or questions:
1. Check main documentation
2. Review example usage in `leanaide_mcts_workflow.py`
3. Verify LeanAide availability
4. Check WorkflowState configuration
