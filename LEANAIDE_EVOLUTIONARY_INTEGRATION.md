# LeanAide Evolutionary Workflow Integration

Comprehensive integration of evolutionary LeanAide capabilities with the OpenEvolve decomposition workflow.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Components](#components)
- [Workflow Stage Integration](#workflow-stage-integration)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Error Handling](#error-handling)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Overview

This module provides seamless integration of evolutionary proof generation, adversarial critique, and self-play learning into the existing OpenEvolve decomposition workflow without breaking changes. It automatically detects mathematical sub-problems and applies appropriate evolutionary strategies.

### Key Capabilities

- **Automatic Mathematical Detection**: Identifies mathematical sub-problems and classifies them by domain
- **Multiple Evolution Strategies**: Evolution, Adversarial, Self-Play, and Hybrid approaches
- **Workflow Stage Integration**: Integrates with Stages 3A, 3B, 3C, and 5
- **Graceful Fallback**: Falls back to standard approach when evolution isn't applicable
- **Progress Tracking**: Real-time monitoring of evolutionary progress
- **Knowledge Storage**: ACE integration for storing learned patterns

## Features

### 1. LeanEvolutionaryWorkflowStage

Main integration class that wraps evolutionary LeanAide for workflow use.

**Capabilities:**
- Detects mathematical sub-problems automatically
- Solves using evolutionary approaches
- Integrates with workflow stages 3A/B/C and 5
- Tracks evolutionary progress
- Stores evolved proofs in knowledge base

### 2. LeanEvolutionarySubProblemSolver

Specialized solver for mathematical sub-problems using evolution.

**Capabilities:**
- Solves mathematical sub-problems with evolution
- Tracks evolutionary progress per sub-problem
- Returns evolved solutions with metadata
- Caches solved sub-problems

### 3. LeanEvolutionaryReassembler

Reassembles evolved sub-proofs into complete proof.

**Capabilities:**
- Validates dependencies between sub-proofs
- Checks consistency across evolved components
- Optimizes final proof
- Handles name conflicts and circular dependencies

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenEvolve Workflow                       │
│                  (workflow_engine.py)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ Integration Points
                         │
┌────────────────────────▼────────────────────────────────────┐
│          LeanEvolutionaryWorkflowStage                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Mathematical Problem Detector                     │    │
│  │  - Classifies problems by domain                   │    │
│  │  - Estimates mathematical confidence               │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Evolution Engine Selector                         │    │
│  │  - Pure Evolution (Genetic Algorithm)              │    │
│  │  - Adversarial (Red vs Blue Team)                  │    │
│  │  - Self-Play (Reinforcement Learning)              │    │
│  │  - Hybrid (Multi-strategy)                         │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Progress Tracker                                  │    │
│  │  - Real-time monitoring                            │    │
│  │  - Convergence detection                           │    │
│  │  - Statistics collection                           │    │
│  └────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
┌────────▼────────┐ ┌───▼────┐ ┌──────▼──────────┐
│  LeanAide       │ │  ACE   │ │  Hephaestus     │
│  Evolution      │ │  KM    │ │  Tracking       │
└─────────────────┘ └────────┘ └─────────────────┘
```

## Components

### LeanEvolutionaryWorkflowStage

```python
class LeanEvolutionaryWorkflowStage:
    """Main integration class for evolutionary LeanAide in workflow stages."""

    def __init__(
        self,
        config: Optional[EvolutionaryConfig] = None,
        workflow_state: Optional[WorkflowState] = None
    )

    # Mathematical detection
    def is_mathematical_subproblem(
        self, sub_problem: SubProblem
    ) -> Tuple[bool, float, Optional[MathematicalDomain]]

    # Stage 3A: Generate proofs using evolution
    async def solve_subproblem_evolutionary(
        self, sub_problem: SubProblem, workflow_state: WorkflowState
    ) -> SolutionAttempt

    # Stage 3B: Evolve through adversarial critique
    async def adversarial_evolution_stage3b(
        self, solution: SolutionAttempt, workflow_state: WorkflowState
    ) -> SolutionAttempt

    # Stage 3C: Verify using evolved proofs
    async def verify_evolved_proof_stage3c(
        self, solution: SolutionAttempt, workflow_state: WorkflowState
    ) -> VerificationReport

    # Stage 5: Final evolutionary verification
    async def evolutionary_final_verification_stage5(
        self, solution: SolutionAttempt, workflow_state: WorkflowState
    ) -> VerificationReport
```

### LeanEvolutionarySubProblemSolver

```python
class LeanEvolutionarySubProblemSolver:
    """Solves mathematical sub-problems using evolutionary approaches."""

    async def solve(
        self, sub_problem: SubProblem, workflow_state: WorkflowState
    ) -> SolutionAttempt

    def get_solution_metadata(
        self, sub_problem_id: str
    ) -> Optional[Dict[str, Any]]
```

### LeanEvolutionaryReassembler

```python
class LeanEvolutionaryReassembler:
    """Reassembles evolved sub-proofs into complete proof."""

    async def reassemble(
        self,
        sub_problem_solutions: Dict[str, SolutionAttempt],
        workflow_state: WorkflowState
    ) -> SolutionAttempt
```

## Workflow Stage Integration

### Stage 3A: Solution Generation

Evolutionary proof generation using genetic algorithms:

```python
# In workflow_stage_functions.py

async def execute_stage3a_solution_loop(workflow_state: WorkflowState):
    """Execute Stage 3A: Solution Loop with evolutionary integration."""

    # Create evolutionary stage
    config = extract_evolutionary_config_from_workflow_state(workflow_state)
    evolutionary_stage = LeanEvolutionaryWorkflowStage(config, workflow_state)

    for sub_problem in workflow_state.decomposition_plan.sub_problems:
        # Check if mathematical
        is_math, confidence, domain = evolutionary_stage.is_mathematical_subproblem(
            sub_problem
        )

        if is_math:
            # Use evolutionary approach
            solution = await evolutionary_stage.solve_subproblem_evolutionary(
                sub_problem, workflow_state
            )
        else:
            # Use standard approach
            solution = await solve_standard(sub_problem, workflow_state)

        workflow_state.sub_problem_solutions[sub_problem.id] = solution
```

### Stage 3B: Adversarial Critique

Evolution through adversarial competition:

```python
# In workflow_stage_functions.py

async def execute_stage3b_adversarial_critique(workflow_state: WorkflowState):
    """Execute Stage 3B: Adversarial evolution."""

    evolutionary_stage = LeanEvolutionaryWorkflowStage(config, workflow_state)

    for sp_id, solution in workflow_state.sub_problem_solutions.items():
        # Apply adversarial evolution
        evolved = await evolutionary_stage.adversarial_evolution_stage3b(
            solution, workflow_state
        )

        workflow_state.sub_problem_solutions[sp_id] = evolved
```

### Stage 3C: Gold Team Verification

Formal verification using LeanAide:

```python
# In workflow_stage_functions.py

async def execute_stage3c_verification(workflow_state: WorkflowState):
    """Execute Stage 3C: Verification with LeanAide."""

    evolutionary_stage = LeanEvolutionaryWorkflowStage(config, workflow_state)

    for sp_id, solution in workflow_state.sub_problem_solutions.items():
        # Verify with LeanAide
        verification_report = await evolutionary_stage.verify_evolved_proof_stage3c(
            solution, workflow_state
        )

        solution.verification_reports.append(verification_report)
```

### Stage 5: Final Verification

Comprehensive final verification:

```python
# In workflow_stage_functions.py

async def execute_stage5_final_verification(workflow_state: WorkflowState):
    """Execute Stage 5: Final verification with evolutionary components."""

    evolutionary_stage = LeanEvolutionaryWorkflowStage(config, workflow_state)

    # Reassemble solutions
    reassembler = LeanEvolutionaryReassembler(evolutionary_stage)
    final_solution = await reassembler.reassemble(
        workflow_state.sub_problem_solutions,
        workflow_state
    )

    # Verify final solution
    final_report = await evolutionary_stage.evolutionary_final_verification_stage5(
        final_solution, workflow_state
    )

    workflow_state.final_solution = final_solution
```

## Configuration

### EvolutionaryConfig

Complete configuration for evolutionary integration:

```python
@dataclass
class EvolutionaryConfig:
    # Evolution enablement
    lean_evolution_enabled: bool = True
    lean_evolution_strategy: EvolutionStrategy = EvolutionStrategy.HYBRID

    # Evolution parameters
    lean_evolution_generations: int = 50
    lean_evolution_population_size: int = 20
    lean_evolution_mutation_rate: float = 0.1
    lean_evolution_crossover_rate: float = 0.8
    lean_evolution_elitism_ratio: float = 0.1

    # Adversarial parameters
    lean_adversarial_rounds: int = 10
    lean_adversarial_convergence_threshold: float = 0.95

    # Self-play parameters
    lean_self_play_games: int = 20
    lean_self_play_exploration_rate: float = 0.3

    # Verification parameters
    lean_verification_confidence_threshold: float = 0.7
    lean_verification_timeout: float = 300.0

    # Fallback behavior
    lean_fallback_to_standard: bool = True
    lean_timeout_handling: str = "partial"  # "partial", "skip", "wait"

    # Integration settings
    lean_auto_detect_mathematical: bool = True
    lean_store_evolved_proofs: bool = True
    lean_track_evolution_statistics: bool = True

    # Hephaestus integration
    hephaestus_enabled: bool = False
    hephaestus_timeout: float = 600.0

    # ACE integration
    ace_learning_enabled: bool = True
    ace_store_patterns: bool = True
```

### Adding Configuration to WorkflowState

```python
from leanaide_evolutionary_workflow import (
    add_evolutionary_config_to_workflow_state,
    extract_evolutionary_config_from_workflow_state
)

# Add configuration
workflow_state = add_evolutionary_config_to_workflow_state(
    workflow_state,
    evolutionary_config
)

# Extract configuration
config = extract_evolutionary_config_from_workflow_state(workflow_state)
```

### Evolution Strategies

```python
class EvolutionStrategy(Enum):
    STANDARD = "standard"              # Non-evolutionary approach
    EVOLUTION = "evolution"            # Pure genetic algorithm evolution
    ADVERSARIAL = "adversarial"        # Red team vs Blue team competition
    SELF_PLAY = "self_play"           # Self-play reinforcement learning
    HYBRID = "hybrid"                  # Combine multiple strategies
```

### Mathematical Domains

```python
class MathematicalDomain(Enum):
    ALGEBRA = "algebra"
    ANALYSIS = "analysis"
    COMBINATORICS = "combinatorics"
    GEOMETRY = "geometry"
    NUMBER_THEORY = "number_theory"
    TOPOLOGY = "topology"
    LOGIC = "logic"
    COMPUTABILITY = "computability"
    COMPLEXITY = "complexity"
    GENERAL = "general"
```

## Usage Examples

### Basic Usage

```python
from leanaide_evolutionary_workflow import (
    LeanEvolutionaryWorkflowStage,
    EvolutionaryConfig,
    EvolutionStrategy
)

# Create configuration
config = EvolutionaryConfig(
    lean_evolution_enabled=True,
    lean_evolution_strategy=EvolutionStrategy.HYBRID,
    lean_evolution_generations=50,
    lean_evolution_population_size=20
)

# Create workflow stage
stage = LeanEvolutionaryWorkflowStage(config, workflow_state)

# Solve a sub-problem
solution = await stage.solve_subproblem_evolutionary(
    sub_problem, workflow_state
)
```

### Advanced Usage

```python
from leanaide_evolutionary_workflow import (
    LeanEvolutionarySubProblemSolver,
    LeanEvolutionaryReassembler
)

# Create specialized solver
solver = LeanEvolutionarySubProblemSolver(stage, config)

# Solve each sub-problem
for sub_problem in decomposition_plan.sub_problems:
    solution = await solver.solve(sub_problem, workflow_state)
    workflow_state.sub_problem_solutions[sub_problem.id] = solution

# Reassemble into final proof
reassembler = LeanEvolutionaryReassembler(stage)
final_proof = await reassembler.reassemble(
    workflow_state.sub_problem_solutions,
    workflow_state
)
```

### Integration with Existing Workflow

```python
from leanaide_evolutionary_workflow import (
    verify_sub_problem_with_leanaide_evolutionary,
    verify_final_solution_with_leanaide_evolutionary
)

# In Stage 3C
verification_report = await verify_sub_problem_with_leanaide_evolutionary(
    sub_problem,
    solution_attempt,
    workflow_state
)

# In Stage 5
final_report = await verify_final_solution_with_leanaide_evolutionary(
    integrated_solution,
    workflow_state
)
```

## API Reference

### Main Classes

#### `LeanEvolutionaryWorkflowStage`

Main integration class for evolutionary LeanAide.

**Constructor:**
```python
LeanEvolutionaryWorkflowStage(
    config: Optional[EvolutionaryConfig] = None,
    workflow_state: Optional[WorkflowState] = None
)
```

**Methods:**

- `is_mathematical_subproblem(sub_problem) -> Tuple[bool, float, Optional[MathematicalDomain]]`
- `solve_subproblem_evolutionary(sub_problem, workflow_state) -> SolutionAttempt`
- `evolve_solution_stage3a(solution, workflow_state) -> SolutionAttempt`
- `adversarial_evolution_stage3b(solution, workflow_state) -> SolutionAttempt`
- `verify_evolved_proof_stage3c(solution, workflow_state) -> VerificationReport`
- `evolutionary_final_verification_stage5(solution, workflow_state) -> VerificationReport`
- `get_progress(sub_problem_id) -> Optional[EvolutionaryProgress]`
- `get_statistics() -> Dict[str, Any]`

#### `LeanEvolutionarySubProblemSolver`

Specialized solver for mathematical sub-problems.

**Constructor:**
```python
LeanEvolutionarySubProblemSolver(
    workflow_stage: LeanEvolutionaryWorkflowStage,
    config: Optional[EvolutionaryConfig] = None
)
```

**Methods:**

- `solve(sub_problem, workflow_state) -> SolutionAttempt`
- `get_solution_metadata(sub_problem_id) -> Optional[Dict[str, Any]]`

#### `LeanEvolutionaryReassembler`

Reassembles evolved sub-proofs.

**Constructor:**
```python
LeanEvolutionaryReassembler(
    workflow_stage: LeanEvolutionaryWorkflowStage
)
```

**Methods:**

- `reassemble(sub_problem_solutions, workflow_state) -> SolutionAttempt`

### Convenience Functions

#### `add_evolutionary_config_to_workflow_state`

Add configuration to workflow state.

```python
def add_evolutionary_config_to_workflow_state(
    workflow_state: WorkflowState,
    config: EvolutionaryConfig
) -> WorkflowState
```

#### `extract_evolutionary_config_from_workflow_state`

Extract configuration from workflow state.

```python
def extract_evolutionary_config_from_workflow_state(
    workflow_state: WorkflowState
) -> EvolutionaryConfig
```

#### `is_subproblem_mathematical`

Check if sub-problem is mathematical.

```python
def is_subproblem_mathematical(
    sub_problem: SubProblem,
    workflow_stage: LeanEvolutionaryWorkflowStage
) -> Tuple[bool, float]
```

#### `solve_with_evolutionary_approach`

Convenience solve function.

```python
async def solve_with_evolutionary_approach(
    sub_problem: SubProblem,
    workflow_state: WorkflowState,
    config: Optional[EvolutionaryConfig] = None
) -> SolutionAttempt
```

## Error Handling

### Graceful Fallback

The system gracefully falls back to standard approach when:

1. Evolution is not applicable (non-mathematical problem)
2. Evolution components are not available
3. Evolution fails with errors
4. Timeout occurs

```python
# Fallback behavior is automatic
config = EvolutionaryConfig(
    lean_fallback_to_standard=True,  # Enable fallback
    lean_timeout_handling="partial"  # Return partial results on timeout
)
```

### Error Recovery

```python
try:
    solution = await stage.solve_subproblem_evolutionary(
        sub_problem, workflow_state
    )
except Exception as e:
    # Automatic fallback if enabled
    # Or handle error manually
    logger.error(f"Evolution failed: {e}")
    solution = await standard_solver.solve(sub_problem)
```

### Partial Results

When timeout occurs:

```python
config = EvolutionaryConfig(
    lean_timeout_handling="partial"  # Return best partial result
)

# Progress is tracked and can be retrieved
progress = stage.get_progress(sub_problem_id)
if progress and progress.status != "completed":
    # Use partial result
    partial_solution = progress.current_best
```

## Testing

### Running Tests

```bash
# Run all tests
pytest test_leanaide_evolutionary_workflow.py -v

# Run specific test class
pytest test_leanaide_evolutionary_workflow.py::TestEvolutionaryWorkflowStage -v

# Run with coverage
pytest test_leanaide_evolutionary_workflow.py --cov=leanaide_evolutionary_workflow
```

### Test Coverage

The test suite includes:

- Configuration tests
- Workflow stage tests
- Sub-problem solver tests
- Reassembler tests
- Integration function tests
- Error handling tests
- Stage integration tests (3A, 3B, 3C, 5)

### Example Test

```python
import pytest
from leanaide_evolutionary_workflow import (
    LeanEvolutionaryWorkflowStage,
    EvolutionaryConfig,
    EvolutionStrategy
)

@pytest.mark.asyncio
async def test_solve_subproblem_evolutionary():
    """Test evolutionary sub-problem solving."""

    config = EvolutionaryConfig(
        lean_evolution_strategy=EvolutionStrategy.EVOLUTION
    )

    stage = LeanEvolutionaryWorkflowStage(config)
    solution = await stage.solve_subproblem_evolutionary(
        sub_problem, workflow_state
    )

    assert solution is not None
    assert solution.sub_problem_id == sub_problem.id
```

## Troubleshooting

### Common Issues

#### 1. LeanAide Not Available

**Problem:** Components not available

```python
LEANAIDE_AVAILABLE = False
EVOLUTION_AVAILABLE = False
```

**Solution:** Install required dependencies

```bash
pip install leanaide-client
```

#### 2. Non-Mathematical Problem Detected

**Problem:** Problem classified as non-mathematical

```python
is_math, confidence, domain = stage.is_mathematical_subproblem(sub_problem)
# is_math = False
```

**Solution:**
- Verify problem description contains mathematical keywords
- Adjust confidence threshold if needed
- Manually override classification

```python
config = EvolutionaryConfig(
    lean_auto_detect_mathematical=False  # Disable auto-detection
)
```

#### 3. Evolution Timeout

**Problem:** Evolution takes too long

**Solution:**
- Reduce generations or population size
- Enable timeout handling
- Use faster strategy

```python
config = EvolutionaryConfig(
    lean_evolution_generations=20,  # Reduce from 50
    lean_evolution_population_size=10,  # Reduce from 20
    lean_verification_timeout=60.0,  # Reduce timeout
    lean_timeout_handling="partial"  # Return partial results
)
```

#### 4. Memory Issues

**Problem:** High memory usage during evolution

**Solution:**
- Reduce population size
- Enable periodic checkpointing
- Use memory-efficient strategy

```python
config = EvolutionaryConfig(
    lean_evolution_population_size=10,  # Smaller population
    lean_evolution_elitism_ratio=0.05,  # Keep fewer elites
)
```

### Debug Mode

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("leanaide_evolutionary_workflow")
logger.setLevel(logging.DEBUG)
```

### Performance Optimization

1. **Use Hybrid Strategy**: Falls back to faster methods when evolution stalls
2. **Enable Caching**: Reuse evolved solutions
3. **Adjust Parameters**: Tune for specific problem types
4. **Monitor Progress**: Use progress tracking to detect issues early

## Integration Checklist

When integrating with existing workflow:

- [ ] Import module: `from leanaide_evolutionary_workflow import ...`
- [ ] Add configuration to workflow state
- [ ] Create `LeanEvolutionaryWorkflowStage` instance
- [ ] Integrate Stage 3A: `solve_subproblem_evolutionary()`
- [ ] Integrate Stage 3B: `adversarial_evolution_stage3b()`
- [ ] Integrate Stage 3C: `verify_evolved_proof_stage3c()`
- [ ] Integrate Stage 5: `evolutionary_final_verification_stage5()`
- [ ] Add progress tracking
- [ ] Configure error handling
- [ ] Enable knowledge storage
- [ ] Add logging
- [ ] Write tests
- [ ] Document integration

## Additional Resources

- [LeanAide Documentation](./LeanAide/)
- [Workflow Documentation](./Decomposition_Workflow.md)
- [Integration Guide](./OPENEREVOLVE_INTEGRATION_GUIDE.md)
- [API Reference](./OPENEVOLVE_API_REFERENCE.md)

## Support

For issues or questions:

1. Check troubleshooting section
2. Review test examples
3. Enable debug logging
4. Check component availability flags
5. Verify configuration

## License

This integration is part of OpenEvolve and follows the same license terms.

---

**Author:** OpenEvolve
**Created:** 2025-12-30
**Version:** 1.0.0
