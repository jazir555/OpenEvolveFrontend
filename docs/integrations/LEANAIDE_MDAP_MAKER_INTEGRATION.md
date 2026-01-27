# LeanAide MDAP/MAKER Workflow Integration Guide

## Overview

This document describes the comprehensive LeanAide MDAP/MAKER workflow integration created at `leanaide_mdap_workflow.py`.

## Architecture

### MDAP (Multi-Strategy Decision Aggregation Protocol)

MDAP uses multiple agents to generate proofs using different strategies, then aggregates to select the best:

```
Multiple Agent Generation (MAG)
├── Direct Prover Agent
├── Inductive Prover Agent
├── Constructive Prover Agent
└── Decomposition Prover Agent
         ↓
    Strategy Pool
         ↓
    Aggregation (Weighted/Majority/Borda)
         ↓
    Best Strategy → LeanAide Verification
```

### MAKER (Multi-Agent voting for KEeping Reliability)

MAKER constructs proofs step-by-step with tactic voting:

```
Task Decomposition
     ↓
Tactic Voting (First-to-Ahead-by-K)
     ↓
Red-Flagging (Error Detection)
     ↓
Final Proof → LeanAide Verification
```

### Hybrid Approach

Combines both methods:
- **MDAP → MAKER**: MDAP generates candidates, MAKER refines
- **MAKER → MDAP**: MAKER constructs, MDAP validates
- **Adaptive**: Automatically selects based on problem complexity

## Components

### 1. LeanMDAPWorkflowIntegrator

Main class for MDAP integration with workflow.

**Key Methods:**
- `solve_subproblem_with_mdap(sub_problem)`: Solve using MDAP
- `mdap_stage3a(sub_problem, workflow_state)`: Stage 3A integration
- `mdap_stage3b(solution, workflow_state)`: Stage 3B refinement
- `configure_mdap_from_workflow(state)`: Extract config from workflow state

**Configuration (LeanMDAPConfig):**
```python
@dataclass
class LeanMDAPConfig:
    enabled: bool = True
    agents: List[str] = ["direct_prover", "inductive_prover", ...]
    parallel_agents: int = 4
    voting_strategy: str = "weighted_confidence"  # or "majority", "borda"
    k_ahead: int = 3
    min_consensus: float = 0.6
    verify_strategies: bool = True
    confidence_threshold: float = 0.7
    fallback_to_evolution: bool = True
    hephaestus_enabled: bool = False
```

**MDAP Strategies:**
- `DIRECT`: Direct theorem proving
- `INDUCTION`: Mathematical induction
- `CONSTRUCTIVE`: Constructive proofs
- `DECOMPOSITION`: Lemma-based decomposition
- `FORWARD`: Forward reasoning
- `BACKWARD`: Backward reasoning (proof by contradiction)
- `HYBRID`: Mixed strategies

### 2. LeanMakerWorkflowIntegrator

Main class for MAKER integration with workflow.

**Key Methods:**
- `solve_with_maker_voting(sub_problem)`: Solve using MAKER
- `maker_stage3a(sub_problem, workflow_state)`: Stage 3A integration
- `maker_refinement_stage3b(solution, workflow_state)`: Stage 3B refinement
- `configure_maker_from_workflow(state)`: Extract config from workflow state

**Configuration (LeanMakerConfig):**
```python
@dataclass
class LeanMakerConfig:
    enabled: bool = True
    mode: MAKERMode = MAKERMode.RECURSIVE
    k_min: int = 2
    k_max: int = 5
    max_votes: int = 100
    enable_first_to_ahead: bool = True
    enable_red_flagging: bool = True
    max_token_length: int = 750
    max_steps: int = 1000
    verify_each_step: bool = False
    verify_final: bool = True
```

**MAKER Tactics:**
- `INTRO`: Introduce hypothesis
- `APPLY`: Apply theorem/lemma
- `REWRITE`: Rewrite using equality
- `SIMP`: Simplify
- `ASSUME`: Assume hypothesis
- `HAVE`: Introduce intermediate fact
- `CALC`: Calculate chain
- `LINARITH`: Linear arithmetic
- `RING`: Ring tactics
- `EXACT`: Exact term
- `REFINE`: Refine with holes

### 3. LeanHybridIntegrator

Hybrid MDAP + MAKER integration.

**Key Methods:**
- `solve_with_mdap_then_maker(sub_problem)`: MDAP generates, MAKER refines
- `solve_with_maker_and_mdap(sub_problem)`: MAKER constructs, MDAP validates
- `adaptive_hybrid_solve(sub_problem)`: Auto-select based on complexity

**Adaptive Logic:**
- Complexity ≥ 8 or effort ≥ 20: Use MDAP (multiple strategies)
- Complexity ≤ 3 or effort ≤ 5: Use MAKER (step-by-step)
- Otherwise: Try MAKER, fall back to MDAP

### 4. LeanMDAPMonitor

Real-time monitoring of MDAP execution.

**Key Methods:**
- `start_monitoring(task)`: Start monitoring a task
- `get_progress()`: Get current progress
- `get_agent_status()`: Get status of all agents
- `get_voting_statistics()`: Get MAKER voting stats

## Workflow Stage Integration

### Stage 3A: Initial Proof Generation

**MDAP Approach:**
```python
from leanaide_mdap_workflow import LeanMDAPWorkflowIntegrator

# Configure from workflow state
integrator = LeanMDAPWorkflowIntegrator(
    config=mdap_config,
    workflow_state=workflow_state
)

# Solve sub-problem
solution = await integrator.mdap_stage3a(sub_problem, workflow_state)
```

**MAKER Approach:**
```python
from leanaide_mdap_workflow import LeanMakerWorkflowIntegrator

integrator = LeanMakerWorkflowIntegrator(
    config=maker_config,
    workflow_state=workflow_state
)

solution = await integrator.maker_stage3a(sub_problem, workflow_state)
```

**Convenience Function:**
```python
from leanaide_mdap_workflow import solve_with_lean_mdap_maker

# Auto-select mode
solution = await solve_with_lean_mdap_maker(
    sub_problem, workflow_state, team, mode="auto"
)

# Or specify mode
solution = await solve_with_lean_mdap_maker(
    sub_problem, workflow_state, team, mode="mdap"  # or "maker" or "hybrid"
)
```

### Stage 3B: Proof Refinement

```python
# MDAP refinement
refined_solution = await integrator.mdap_stage3b(
    solution, workflow_state
)

# MAKER refinement
refined_solution = await maker_integrator.maker_refinement_stage3b(
    solution, workflow_state
)
```

### Stage 3C: Verification

```python
# MDAP/MAKER solutions include verification metadata
if solution.openevolve_metrics:
    verification = solution.openevolve_metrics.get("maker_verification")
    if verification:
        print(f"Verification: {verification['success']}")
        print(f"Confidence: {verification['confidence']}")
```

### Stage 5: Final Verification

```python
# Use LeanAide verification for MDAP/MAKER proofs
from leanaide_workflow_integration import verify_sub_problem_with_leanaide

verification_report = await verify_sub_problem_with_leanaide(
    sub_problem, solution, workflow_state
)
```

## Configuration Integration

### Adding to WorkflowState

```python
from leanaide_mdap_workflow import (
    add_mdap_maker_config_to_workflow_state,
    LeanMDAPConfig,
    LeanMakerConfig
)

# Create configs
mdap_config = LeanMDAPConfig(
    enabled=True,
    parallel_agents=4,
    voting_strategy="weighted_confidence",
    k_ahead=3
)

maker_config = LeanMakerConfig(
    enabled=True,
    k_min=2,
    k_max=5,
    max_votes=100
)

# Add to workflow state
workflow_state = add_mdap_maker_config_to_workflow_state(
    workflow_state,
    mdap_config=mdap_config,
    maker_config=maker_config
)
```

### WorkflowState Parameters

The following parameters are added to `workflow_state.openevolve_parameters`:

**MDAP Parameters:**
- `lean_mdap_enabled`: bool
- `lean_mdap_agents`: List[str]
- `lean_mdap_parallel_agents`: int
- `lean_mdap_voting_strategy`: str
- `lean_mdap_k_ahead`: int
- `lean_mdap_verify`: bool
- `lean_mdap_confidence_threshold`: float

**MAKER Parameters:**
- `lean_maker_enabled`: bool
- `lean_maker_k_min`: int
- `lean_maker_k_max`: int
- `lean_maker_max_votes`: int
- `lean_maker_red_flagging`: bool
- `lean_maker_verify`: bool

## When to Use MDAP vs MAKER

### Use MDAP for:

1. **Complete proof generation from scratch**
   - When you have no clear starting point
   - Want to explore multiple approaches

2. **Multiple strategies in parallel**
   - Complex theorems with multiple proof paths
   - When you want to compare different approaches

3. **Complex theorems requiring decomposition**
   - Theorems that need lemma decomposition
   - Multi-step proofs with clear structure

4. **Exploration**
   - Research phase exploring proof strategies
   - When unsure which approach will work

### Use MAKER for:

1. **Step-by-step tactic selection**
   - When the proof path is relatively clear
   - Incremental proof construction

2. **Fine-grained proof construction**
   - Need control over each tactic
   - Want voting on each step

3. **Good applicable tactics available**
   - When you have a good set of candidate tactics
   - Tactic voting is meaningful

4. **Voting-based decision making**
   - When you want confidence in each step
   - Red-flagging is valuable

### Use Hybrid for:

1. **Best of both approaches**
   - MDAP generates candidate proofs
   - MAKER refines through voting

2. **Adaptive solving**
   - Let the system decide based on complexity
   - Automatic fallback strategies

## Fallback Strategies

### MDAP Fallback Chain:

1. **MDAP fails** → Try single strategy (direct/inductive)
2. **Single strategy fails** → Fall back to evolution
3. **Evolution fails** → Fall back to standard approach

### MAKER Fallback Chain:

1. **MAKER fails** → Try direct tactic selection
2. **Direct fails** → Fall back to MDAP
3. **MDAP fails** → Fall back to standard approach

### Hybrid Fallback Chain:

1. **MAKER fails** → Try MDAP
2. **MDAP fails** → Try evolution
3. **All fail** → Standard approach

## Integration with Existing Components

### LeanAide Client

```python
# Verification is handled through LeanAideWorkflowIntegrator
if LEANAIDE_AVAILABLE:
    verification = await leanaide_integrator.verify_sub_problem_solution(
        sub_problem_id=sub_problem.id,
        problem_statement=task.theorem_statement,
        solution_content=lean_code,
        verification_requirements=sub_problem.solution_requirements
    )
```

### Hephaestus Tracking

```python
# Track MDAP tickets if enabled
if config.hephaestus_enabled and HEPHAESTUS_AVAILABLE:
    hephaestus_client = HephaestusClient()
    # Create ticket for each MDAP task
    # Update status as tasks complete
```

### Knowledge Engine Storage

```python
# Store MDAP results in ACE knowledge base
if ACE_AVAILABLE:
    ace_manager = ACEKnowledgeManager()
    artifact = {
        "type": "mdap_result",
        "sub_problem_id": sub_problem.id,
        "best_strategy": best_result.strategy_type.value,
        "confidence": best_result.confidence,
        "lean_code": best_result.lean_code
    }
    ace_manager.store_artifact(artifact)
```

## Error Handling

### Timeout Handling

```python
# Each MDAP agent has timeout
agent_timeout: float = 120.0

# Tasks that timeout return partial results
# Partial results are included in aggregation
```

### Agent Failure Handling

```python
# Failed agents are tracked in agent_status
agent_status[agent_id] = {
    "status": "failed",
    "error": str(e),
    "execution_time": time.time() - start_time
}

# Other agents continue
# Partial results still aggregated
```

### Voting Failure Handling

```python
# If voting fails (e.g., no consensus):
# 1. Fall back to weighted confidence
# 2. If still fails, select highest confidence
# 3. If all fail, use direct strategy
```

## Logging

MDAP/MAKER progress is logged at each step:

```python
logger.info(f"Solving sub-problem {sub_problem.id} with MDAP")
logger.info(f"Executing {len(tasks)} MDAP tasks in parallel")
logger.info(f"Selected MDAP strategy: {best.strategy_type.value}")
logger.info(f"MAKER Stage 3A: Generating proof for {sub_problem.id}")
```

Progress tracking is available through `LeanMDAPMonitor`.

## Example Usage

### Basic MDAP Usage

```python
import asyncio
from leanaide_mdap_workflow import (
    LeanMDAPWorkflowIntegrator,
    LeanMDAPConfig
)

async def solve_with_mdap():
    # Configure MDAP
    config = LeanMDAPConfig(
        enabled=True,
        parallel_agents=4,
        voting_strategy="weighted_confidence",
        k_ahead=3,
        verify_strategies=True
    )

    # Create integrator
    integrator = LeanMDAPWorkflowIntegrator(config=config)

    # Solve sub-problem
    solution = await integrator.solve_subproblem_with_mdap(sub_problem)

    print(f"Proof: {solution.content}")
    print(f"Status: {solution.status}")
    print(f"Strategy: {solution.openevolve_metrics['mdap_strategy']}")

asyncio.run(solve_with_mdap())
```

### Basic MAKER Usage

```python
from leanaide_mdap_workflow import (
    LeanMakerWorkflowIntegrator,
    LeanMakerConfig
)

async def solve_with_maker():
    # Configure MAKER
    config = LeanMakerConfig(
        enabled=True,
        k_min=2,
        k_max=5,
        max_votes=100,
        enable_red_flagging=True,
        verify_final=True
    )

    # Create integrator
    integrator = LeanMakerWorkflowIntegrator(config=config)

    # Solve sub-problem
    solution = await integrator.solve_with_maker_voting(sub_problem)

    print(f"Proof: {solution.content}")
    print(f"Status: {solution.status}")

asyncio.run(solve_with_maker())
```

### Hybrid Usage

```python
from leanaide_mdap_workflow import LeanHybridIntegrator

async def solve_with_hybrid():
    # Create hybrid integrator
    integrator = LeanHybridIntegrator(
        mdap_config=mdap_config,
        maker_config=maker_config
    )

    # Adaptive solve
    solution = await integrator.adaptive_hybrid_solve(sub_problem)

    print(f"Proof: {solution.content}")
    print(f"Approach: {solution.solution_approach}")

asyncio.run(solve_with_hybrid())
```

### Monitoring Progress

```python
from leanaide_mdap_workflow import LeanMDAPMonitor, LeanMDAPTask

# Create monitor
monitor = LeanMDAPMonitor(mdap_integrator)

# Start monitoring
task = LeanMDAPTask(
    task_id="task_001",
    sub_problem_id="sp_001",
    theorem_statement="forall n, n + 0 = n",
    proof_goal="prove_zero_add",
    context={},
    strategy_type=MDAPStrategyType.DIRECT,
    agent_id="direct_prover"
)

monitor.start_monitoring(task)

# Check progress
progress = monitor.get_progress()
print(f"Elapsed: {progress['elapsed_time']:.2f}s")
print(f"Agent status: {progress['agent_status']}")
```

## Availability Flags

Check component availability:

```python
from leanaide_mdap_workflow import (
    LEANAIDE_AVAILABLE,
    MDAP_AVAILABLE,
    MAKER_AVAILABLE,
    WORKFLOW_AVAILABLE,
    HEPHAESTUS_AVAILABLE,
    ACE_AVAILABLE
)

print(f"LeanAide: {LEANAIDE_AVAILABLE}")
print(f"MDAP: {MDAP_AVAILABLE}")
print(f"MAKER: {MAKER_AVAILABLE}")
print(f"Workflow: {WORKFLOW_AVAILABLE}")
print(f"Hephaestus: {HEPHAESTUS_AVAILABLE}")
print(f"ACE: {ACE_AVAILABLE}")
```

## Summary

The LeanAide MDAP/MAKER workflow integration provides:

1. **Multi-agent proof generation** (MDAP)
   - Multiple strategies in parallel
   - Voting-based aggregation
   - LeanAide verification

2. **Step-by-step proof construction** (MAKER)
   - Tactic voting
   - First-to-ahead-by-K error correction
   - Red-flagging

3. **Hybrid approach**
   - Adaptive strategy selection
   - Best of both methods
   - Automatic fallback

4. **Seamless workflow integration**
   - Stage 3A/B/C integration
   - WorkflowState configuration
   - Progress monitoring

5. **Robust error handling**
   - Timeout handling
   - Agent failure handling
   - Multiple fallback strategies

6. **Integration with existing components**
   - LeanAide client for verification
   - Hephaestus for tracking
   - Knowledge Engine for storage

This integration adds powerful Lean theorem proving capabilities to the OpenEvolve decomposition workflow while maintaining backward compatibility and graceful fallbacks.
