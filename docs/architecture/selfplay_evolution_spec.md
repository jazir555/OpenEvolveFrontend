# Self-Play Evolution Implementation Specification (PSV: Propose, Solve, Verify)

## Overview
This specification outlines the implementation of PSV (Propose, Solve, Verify) self-play functionality in the OpenEvolve framework. PSV is a self-play algorithm for code generation that leverages formal verification to provide reliable correctness signals, preventing error propagation and reward hacking that occurs with unit test-based approaches.

## Core Components

### 1. Proposer Model (Pϕt)
- **Purpose**: Generate formal specifications (problems) for the solver to work on
- **Input**: Current solver performance metrics, difficulty assessments, and historical data
- **Output**: New formal specifications in Verus/Rust format
- **Features**:
  - Difficulty-aware proposal based on solver pass rates
  - In-context learning using historical examples
  - Diversity through dynamic prompt refreshment
  - Coverage of different problem types (Easy, Medium, Hard, Impossible)

### 2. Solver Model (Sθt)
- **Purpose**: Generate code solutions that meet formal specifications
- **Input**: Formal specifications (function signatures, preconditions, postconditions)
- **Output**: Verified code implementations with proof annotations
- **Features**:
  - Rejection fine-tuning (RFT) on verified solutions only
  - Temperature-based sampling (0.8 recommended)
  - Integration with existing evolution loop infrastructure

### 3. Formal Verifier
- **Purpose**: Provide sound correctness verification for generated solutions
- **Input**: Specification + generated code
- **Output**: Binary verification result (pass/fail)
- **Features**:
  - Integration with Verus framework for Rust verification
  - Sound verification (if accepted, guaranteed correct for all inputs)
  - Specification validation for proposed problems

## Implementation Architecture

### New Classes and Functions

#### SelfPlayEvolutionManager
```python
class SelfPlayEvolutionManager:
    def __init__(self, config: EvolutionConfiguration):
        # Initialize proposer, solver, and verifier components
        # Load seed specifications dataset
        # Set up training loop parameters
    
    def run_selfplay_iteration(self, iteration: int) -> Dict[str, Any]:
        # Execute one iteration of PSV: Propose -> Solve -> Verify -> Train
        pass
    
    def propose_new_specifications(self, target_difficulty: str) -> List[Specification]:
        # Generate new formal specifications based on current solver performance
        pass
    
    def solve_specifications(self, specifications: List[Specification]) -> List[SolverResult]:
        # Generate solutions for given specifications
        pass
    
    def verify_solutions(self, solutions: List[SolverResult]) -> List[VerificationResult]:
        # Verify solutions against specifications using formal verification
        pass
    
    def update_solver_model(self, verified_solutions: List[VerificationResult]):
        # Perform rejection fine-tuning on verified solutions only
        pass
    
    def update_proposer_model(self, verification_results: List[VerificationResult]):
        # Update proposer based on solver performance and difficulty feedback
        pass
```

#### Specification Class
```python
@dataclass
class Specification:
    id: str
    content: str  # Formal specification in Verus/Rust format
    difficulty: str  # Easy, Medium, Hard, Impossible
    category: str  # Function type, algorithm class, etc.
    metadata: Dict[str, Any]  # Additional metadata
```

#### SolverResult Class
```python
@dataclass
class SolverResult:
    specification_id: str
    solution_code: str  # Generated implementation
    proof_annotations: str  # Loop invariants, etc.
    success: bool
    confidence: float
    execution_time: float
```

#### VerificationResult Class
```python
@dataclass
class VerificationResult:
    specification_id: str
    solution_code: str
    verified: bool
    verification_log: str
    error_details: Optional[str]
    confidence: float
```

## Integration with Existing Evolution.py

### New Evolution Modes
Add new evolution modes to `EvolutionConfiguration`:
- `selfplay_propose`: For specification generation
- `selfplay_solve`: For solution generation  
- `selfplay_verify`: For formal verification
- `selfplay_full`: Complete PSV loop

### Enhanced Configuration Parameters
Add PSV-specific parameters to `EvolutionConfiguration`:
```python
# Self-Play Parameters
selfplay_iterations: int = 10
selfplay_question_budget: int = 1000
selfplay_difficulty_thresholds: Dict[str, float] = None  # Easy, Medium thresholds
selfplay_proposal_examples: int = 12
selfplay_verification_enabled: bool = True
selfplay_formal_verification_backend: str = "verus"  # verus, dafny, etc.
selfplay_solver_temperature: float = 0.8
selfplay_solver_attempts_per_spec: int = 10
```

### New Functions in evolution.py
```python
def run_selfplay_evolution(
    content: str,
    content_type: str = "verified_code",
    config: Optional[EvolutionConfiguration] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main entry point for PSV self-play evolution
    """
    pass

def run_propose_phase(
    manager: SelfPlayEvolutionManager,
    config: EvolutionConfiguration
) -> List[Specification]:
    """
    Execute the propose phase of PSV
    """
    pass

def run_solve_phase(
    specifications: List[Specification],
    config: EvolutionConfiguration
) -> List[SolverResult]:
    """
    Execute the solve phase of PSV
    """
    pass

def run_verify_phase(
    solutions: List[SolverResult],
    config: EvolutionConfiguration
) -> List[VerificationResult]:
    """
    Execute the verify phase of PSV
    """
    pass

def integrate_with_formal_verification(
    code: str,
    specification: str,
    backend: str = "verus"
) -> bool:
    """
    Interface with formal verification backends
    """
    pass
```

## Self-Play Loop Implementation

### Main PSV Algorithm
```python
def run_psv_algorithm(
    seed_specifications: List[Specification],
    solver_model: Any,
    proposer_model: Any,
    config: EvolutionConfiguration
) -> Dict[str, Any]:
    """
    Implements the core PSV algorithm:
    1. Initialize with seed specifications
    2. For each iteration:
       a. Solve current specifications
       b. Verify solutions
       c. Update solver with verified solutions
       d. Update proposer with difficulty feedback
       e. Generate new specifications
    3. Return final trained model and metrics
    """
    pass
```

### Difficulty-Aware Proposal
The proposer should adapt specification difficulty based on solver performance:
- **Easy**: Pass rate ≥ threshold_E (e.g., 0.8)
- **Medium**: threshold_M ≤ pass rate < threshold_E (e.g., 0.2 ≤ pass rate < 0.8)  
- **Hard**: 0 < pass rate < threshold_M (e.g., 0 < pass rate < 0.2)
- **Impossible**: Pass rate = 0

### Rejection Fine-Tuning (RFT)
- Only train solver on verified solutions (verification = True)
- Use cross-entropy loss on verified solution tokens
- Maintain base model initialization for stable training

## Formal Verification Integration

### Verus Backend Integration
- Interface with Verus verification framework
- Support for Rust code verification
- Handling of proof annotations and loop invariants
- Error reporting and debugging support

### Verification Process
1. Parse generated code and specification
2. Run formal verification using SMT solver
3. Return binary result with detailed logs
4. Handle verification timeouts and errors gracefully

## Metrics and Monitoring

### Self-Play Metrics
- Solver pass rates per difficulty level
- Specification diversity metrics
- Verification success rates
- Training convergence indicators
- Performance scaling with iterations

### Evolution Metrics Integration
- Integrate PSV metrics with existing evolution metrics
- Track improvement over iterations
- Monitor for error propagation prevention

## Error Handling and Safety

### Verification Safety
- Ensure only formally verified solutions are used for training
- Prevent incorrect solutions from entering training loop
- Handle verification failures gracefully

### Training Stability
- Monitor for model degradation
- Implement early stopping based on verification metrics
- Maintain diversity in generated specifications

## Testing and Validation

### Unit Tests
- Test each PSV component (Propose, Solve, Verify)
- Validate formal verification integration
- Test difficulty-aware proposal logic

### Integration Tests
- End-to-end PSV loop validation
- Verification accuracy testing
- Performance comparison with baseline methods

## Deployment Considerations

### Resource Requirements
- Formal verification can be computationally expensive
- Plan for adequate compute resources
- Consider verification timeouts and fallbacks

### Scalability
- Support for distributed verification
- Batch processing of specifications
- Efficient storage of verification results

## Future Extensions

### Multi-Modal Self-Play
- Extend beyond code generation to other domains
- Support for different formal verification backends
- Integration with existing OpenEvolve features

### Advanced Proposer Techniques
- Learning to propose based on solver improvement
- Curriculum learning approaches
- Adversarial specification generation