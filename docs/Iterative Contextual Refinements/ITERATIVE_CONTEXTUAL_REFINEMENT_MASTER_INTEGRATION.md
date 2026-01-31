# Iterative Contextual Refinement - Master Integration Guide

This document provides a comprehensive guide for integrating iterative contextual refinements into the OpenEvolve workflow system. It synthesizes findings from the codebase and provides integration patterns for all major workflow components.

## Table of Contents

1. [Overview](#overview)
2. [Core Refinement Architecture](#core-refinement-architecture)
3. [Integration Points](#integration-points)
4. [Decomposition Workflow Integration](#decomposition-workflow-integration)
5. [Adaptive Maker Integration](#adaptive-maker-integration)
6. [Deterministic LLM Integration](#deterministic-llm-integration)
7. [Gauntlet System Integration](#gauntlet-system-integration)
8. [Best Practices](#best-practices)
9. [Configuration Reference](#configuration-reference)

---

## Overview

### What is Iterative Contextual Refinement?

Iterative contextual refinement is a systematic approach to continuously improving solutions through feedback loops that leverage contextual information from previous iterations. Unlike single-pass generation, iterative refinement:

- **Builds on partial successes**: Each iteration learns from prior attempts
- **Adapts to feedback**: Critiques and verification results drive improvements
- **Converges toward quality**: Multiple passes eliminate edge cases and improve robustness

### Key Concepts

| Concept | Description | Default Value |
|---------|-------------|---------------|
| `refinement_loop_count` | Current iteration number in the refinement loop | 0 |
| `max_refinement_loops` | Maximum allowed iterations before manual intervention | 3 |
| `auto_approval_enabled` | Whether to auto-approve solutions meeting quality thresholds | False |

---

## Core Refinement Architecture

### RefinementCoordinator

The [`RefinementCoordinator`](sovereign_refinement.py:61) class is the central orchestrator for iterative refinement. It bridges the backend state with the **Iterative Studio** frontend.

### Mapping: Backend to Frontend (Iterative Studio)

| Backend Component | Studio Mode | Integration Implementation |
| :--- | :--- | :--- |
| `RefinementCoordinator` | **Refine Mode** | Pipeline-based iterative loops in `WebsiteLogic.ts`. |
| `ComprehensiveRefinementEngine` | **Deepthink Mode** | The 3-phase correction loop in `DeepthinkCore.ts`. |
| `GauntletSystem` | **Red Team Filter** | Consolidated adversarial analysis in `runConsolidatedRedTeamAnalysis`. |
| `MemoryAgentHistoryManager` | **Contextual Mode** | 10-turn condensation logic in `ContextualCore.ts`. |
| `DeterministicRefinementEngine` | **Agentic Mode** | LangChain structured message history in `AgenticCoreLangchain.ts`. |

### Refinement Loop Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    REFINEMENT LOOP                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Generate  │───▶│   Critique  │───▶│   Analyze   │         │
│  │   Solution  │    │   (Red Team)│    │   Feedback  │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         ▲                                       │              │
│         │                                       ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │    Apply    │◀───│    Fix      │◀───│   Plan      │         │
│  │   Solution  │    │   (Blue Team)│    │   Improvements│      │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                                       │              │
│         └───────────────────────────────────────┘              │
│                    (Loop until approved or                     │
│                     max_refinement_loops reached)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Integration Points

### 1. Workflow State Integration

The workflow state tracks refinement progress through:

```python
@dataclass
class WorkflowState:
    # ... other fields ...
    final_solution: Optional[SolutionAttempt] = None
    refinement_loop_count: int = 0
    final_gold_gauntlet: Optional[GauntletDefinition] = None
    max_refinement_loops: int = 3  # Max iterations for the self-healing loop
    all_critique_reports: List[CritiqueReport] = dataclasses.field(default_factory=list)
```

### 2. Quality Gates

Each refinement iteration passes through quality gates:

```python
def check_quality_gates(workflow_state: WorkflowState) -> QualityGateResult:
    """Evaluate if current solution meets quality thresholds."""
    if workflow_state.refinement_loop_count > workflow_state.max_refinement_loops:
        return QualityGateResult(
            passed=False,
            reason="Max refinement loops exceeded",
            requires_manual_intervention=True
        )
    # Additional quality checks...
```

### 3. Analytics Integration

Refinement metrics are tracked for optimization:

```python
def track_refinement_metrics(workflow_state: WorkflowState) -> RefinementMetrics:
    return {
        "refinement_loop_count": workflow_state.refinement_loop_count,
        "max_refinement_loops": workflow_state.max_refinement_loops,
        "refinement_effective": workflow_state.refinement_loop_count < workflow_state.max_refinement_loops,
        "verification_reports_count": len(workflow_state.all_verification_reports),
    }
```

---

## Decomposition Workflow Integration

### Adding Refinement to Decomposition Stages

The decomposition workflow supports iterative refinement at multiple stages:

#### Stage 1: Initial Decomposition

```python
async def initial_decomposition_stage(
    problem_statement: str,
    max_refinement_loops: int = 3
) -> DecompositionPlan:
    """
    Initial decomposition with iterative refinement support.
    
    Refinement triggers:
    - Missing dependencies detected
    - Overlapping sub-problems
    - Insufficient granularity
    """
    plan = await generate_initial_plan(problem_statement)
    
    for i in range(max_refinement_loops):
        feedback = await analyze_decomposition_quality(plan)
        if feedback.is_sufficient:
            break
        plan = await refine_plan(plan, feedback)
        
    return plan
```

#### Stage 2: Sub-Problem Refinement

```python
async def refine_sub_problems(
    plan: DecompositionPlan,
    context: RefinementContext
) -> RefinedPlan:
    """
    Refine individual sub-problems based on:
    - Dependency validation
    - Complexity assessment
    - Solution feasibility analysis
    """
```

### Integration with Existing Decomposition Engines

The following engines support iterative refinement:

| Engine | Refinement Support | Integration Point |
|--------|-------------------|-------------------|
| `decomposition_engine.py` | Full | `refine_decomposition()` |
| `comprehensive_decomposition_engine.py` | Full | Multi-stage refinement |
| `decomposition_engine_lean_enhanced.py` | Partial | Hill climbing mode |

---

## Adaptive Maker Integration

### MDAP Integration with Refinement

The Adaptive Maker leverages iterative refinement through the MDAP (Multi-Dimensional Adaptive Processing) framework:

```python
class AdaptiveMakerRefinement:
    """Integrates iterative refinement into Adaptive Maker workflow."""
    
    def __init__(self, mdap_engine, llm_provider):
        self.mdap_engine = mdap_engine
        self.llm_provider = llm_provider
        self.refinement_strategies = {
            "gradient_descent": self._gradient_descent_refine,
            "simulated_annealing": self._annealing_refine,
            "hill_climbing": self._hill_climb_refine,
        }
    
    async def adaptive_refine(
        self,
        solution: SolutionAttempt,
        gauntlets: List[GauntletDefinition],
        strategy: str = "adaptive"
    ) -> RefinedSolution:
        """Apply adaptive refinement based on gauntlet feedback."""
```

### Refinement Strategies

| Strategy | Use Case | Characteristics |
|----------|----------|-----------------|
| `gradient_descent` | Local optimization | Fast convergence, may get stuck in local optima |
| `simulated_annealing` | Global search | Slower, escapes local optima |
| `hill_climbing` | Simple improvements | Fast, minimal exploration |
| `adaptive` | Dynamic selection | Automatically selects best strategy |

### Integration with CrewAI Teams

The Adaptive Maker uses CrewAI agents for refinement:

```python
# From crewai_mdap_maker_engine.py
REFINEMENT_AGENT_PROMPT = """You are an expert at improving solutions through iterative refinement.
You focus on:
- Analyzing critique feedback systematically
- Identifying root causes of failures
- Implementing targeted improvements
- Validating fixes don't introduce regressions
"""
```

---

## Deterministic LLM Integration

### Achieving Determinism in Refinement

To ensure reproducible refinement behavior:

```python
class DeterministicRefinementEngine:
    """Provides deterministic iterative refinement using seeded LLM calls."""
    
    def __init__(self, llm_adapter, seed: int = 42):
        self.llm_adapter = llm_adapter
        self.seed = seed
        self.refinement_trace = []
    
    async def deterministic_refine(
        self,
        solution: SolutionAttempt,
        context: DeterministicContext
    ) -> DeterministicRefinedSolution:
        """
        Perform refinement with deterministic outputs.
        
        Key techniques:
        1. Seeded random number generation
        2. Consistent prompt templates
        3. Deterministic sampling parameters
        """
        # Set deterministic seed
        set_seed(self.seed)
        
        # Track all decisions for reproducibility
        self.refinement_trace.append({
            "step": len(self.refinement_trace),
            "action": "analyze_feedback",
            "seed": self.seed,
            "deterministic_hash": self._compute_deterministic_hash()
        })
```

### Checkpoint System

```python
@dataclass
class RefinementCheckpoint:
    """Snapshot of refinement state for reproducibility."""
    iteration: int
    solution_hash: str
    feedback_hash: str
    action_taken: str
    deterministic_seed: int

def save_checkpoint(
    workflow_state: WorkflowState,
    checkpoint_path: str
) -> RefinementCheckpoint:
    """Save refinement checkpoint for reproducibility."""
```

---

## Gauntlet System Integration

### Gauntlet Roles in Refinement

The gauntlet system provides structured feedback for refinement:

| Gauntlet | Role in Refinement | Timing |
|----------|-------------------|--------|
| `SolverGenerationGauntlet` | Solution quality validation | Each iteration |
| `FinalVerificationGauntlet` | Final solution approval | End of refinement |
| `RedTeamGauntlet` | Identify weaknesses | After each solution |
| `BlueTeamGauntlet` | Propose fixes | When Red Team finds issues |
| `GoldGauntlet` | Holistic evaluation | Final approval |

### Integration Pattern

```python
def run_gauntlet_refinement(
    solution: SolutionAttempt,
    gauntlets: GauntletPipeline,
    max_loops: int = 3
) -> RefinementResult:
    """
    Run gauntlet-driven refinement loop.
    
    Args:
        solution: Initial solution to refine
        gauntlets: Pipeline of gauntlets for evaluation
        max_loops: Maximum refinement iterations
    
    Returns:
        RefinementResult with final solution and all critique reports
    """
    current_solution = solution
    all_reports = []
    
    for i in range(max_loops):
        # Run gauntlet evaluation
        report = gauntlets.evaluate(current_solution)
        all_reports.append(report)
        
        if report.passed_all_checks:
            return RefinementResult(
                solution=current_solution,
                reports=all_reports,
                iterations_needed=i + 1
            )
        
        # Apply fixes based on critique
        current_solution = apply_critique_fixes(current_solution, report)
    
    return RefinementResult(
        solution=current_solution,
        reports=all_reports,
        iterations_needed=max_loops,
        incomplete=True
    )
```

---

## Best Practices

### 1. Set Appropriate Max Loops

```python
# Good: Reasonable default with user override
config.max_refinement_loops = 3  # Default

# Context-specific adjustment
if problem.complexity > Complexity.HIGH:
    config.max_refinement_loops = 5
elif problem.complexity == Complexity.TRIVIAL:
    config.max_refinement_loops = 1
```

### 2. Track Refinement Patterns

```python
def analyze_refinement_patterns(workflow_history: List[WorkflowState]):
    """Identify patterns in refinement behavior."""
    avg_loops = statistics.mean([
        w.refinement_loop_count for w in workflow_history
    ])
    
    if avg_loops > 3:
        return OptimizationRecommendation(
            type="reduce_refinement_loops",
            suggestion="Review initial solution quality to reduce need for refinements"
        )
```

### 3. Early Termination Criteria

```python
def should_terminate_refinement(
    workflow_state: WorkflowState,
    last_feedback: CritiqueReport
) -> bool:
    """Determine if refinement should terminate early."""
    # Quality threshold met
    if last_feedback.overall_score >= 0.95:
        return True
    
    # Diminishing returns
    if workflow_state.refinement_loop_count > 0:
        improvement = last_feedback.overall_score - \
            workflow_state.previous_score
        if improvement < 0.01:  # Less than 1% improvement
            return True
    
    # Max loops reached
    if workflow_state.refinement_loop_count >= \
       workflow_state.max_refinement_loops:
        return True
    
    return False
```

### 4. Logging and Debugging

```python
import structlog

logger = structlog.get_logger(__name__)

async def tracked_refinement(
    workflow_state: WorkflowState,
    ...
):
    logger.info(
        "refinement_started",
        iteration=workflow_state.refinement_loop_count,
        max_loops=workflow_state.max_refinement_loops
    )
    
    try:
        result = await perform_refinement(...)
        
        logger.info(
            "refinement_completed",
            iterations=workflow_state.refinement_loop_count,
            success=result.success
        )
        
        return result
    except Exception as e:
        logger.error("refinement_failed", error=str(e))
        raise
```

---

## Configuration Reference

### Full Configuration Schema

```yaml
refinement:
  # Maximum iterations
  max_refinement_loops: 3
  
  # Auto-approval settings
  auto_approval:
    enabled: false
    quality_threshold: 0.95
    require_consensus: true
  
  # Refinement strategies
  strategies:
    default: adaptive
    available:
      - gradient_descent
      - simulated_annealing
      - hill_climbing
      - adaptive
  
  # Gauntlet integration
  gauntlets:
    enabled: true
    pipeline:
      - solver_generation
      - red_team
      - blue_team
      - final_verification
  
  # Determinism settings
  determinism:
    enabled: false
    seed: 42
    checkpoint_interval: 1
  
  # Analytics and monitoring
  analytics:
    track_patterns: true
    log_each_iteration: true
    generate_report: true
```

---

## See Also

- [Decomposition Workflow](docs/Decomposition/Decomposition_Workflow.md)
- [Adaptive Maker Integration Guide](docs/Adaptive%20Maker/ADAPTIVE_MAKER_INTEGRATION_GUIDE.md)
- [Deterministic LLM Integration Master Guide](docs/determinism/DETERMINISTIC_LLM_INTEGRATION_MASTER_GUIDE.md)
- [Gauntlet System Documentation](docs/gauntlets/GAUNTLET_SYSTEM_DOCUMENTATION.md)
- [Sovereign Refinement System](sovereign_refinement.py)
- [Workflow Engine](workflow_engine.py)
