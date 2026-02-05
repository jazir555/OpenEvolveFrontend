# LeanAide-RESE Workflow Architecture

## Overview

The LeanAide-RESE workflow adapter integrates LeanAide's AI-powered theorem proving capabilities with RESE's 4-phase pipeline, creating a comprehensive system for formal verification and mathematical reasoning.

## Architecture Principles

Following CLAUDE.md laws:

1. **Law of Air Gap**: No imports from `core-projects/` directory
2. **Law of Runtime Truth**: Verify via execution, not documentation
3. **Law of Untouchable DB**: Read-only database access
4. **Law of Idempotency**: All operations safe to replay
5. **Law of Configuration Explicitness**: All config via environment variables
6. **Law of UTC**: All timestamps in UTC

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LeanAide-RESE Workflow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │ Phase I:     │      │ Phase II:    │      │ Phase III:   │  │
│  │ Epistemic    │ ───▶ │ Isomorphic   │ ───▶ │ MCTS         │  │
│  │ Audit        │      │ Mapping      │      │ Refinement   │  │
│  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘  │
│         │                     │                      │          │
│         ▼                     ▼                      ▼          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Autoformalization Service                     │  │
│  │  - Natural language → Lean 4                              │  │
│  │  - Domain detection                                        │  │
│  │  - Theorem naming                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                     │                      │          │
│         ▼                     ▼                      ▼          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 Proof Search Service                       │  │
│  │  - MCTS-guided search                                     │  │
│  │  - Z3-LeanAide hybrid                                     │  │
│  │  - Auto tactics                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────┐                                               │
│  │ Phase IV:    │ ◀── All phases converge                     │
│  │ Architectural│                                               │
│  │ Synthesis    │                                               │
│  └──────────────┘                                               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│  LeanAide       │         │  Z3 Bridge      │
│  Server         │         │  (Optional)     │
│  (localhost:    │         │                 │
│   7654)         │         │                 │
└─────────────────┘         └─────────────────┘
```

## Component Design

### 1. Autoformalization Service

**Purpose**: Translate natural language to Lean 4 code

**Key Features**:
- Domain detection (arithmetic, logic, graph theory, etc.)
- Theorem name generation
- Phase-specific formalization strategies
- Fallback to template-based generation

**API**:
```python
async def autoformalize_phase_i(
    constraint_text: str,
    constraint_type: str = "logical",
    correlation_id: Optional[str] = None
) -> AutoformalizationResult

async def autoformalize_phase_ii(
    mapping_description: str,
    source_domain: str,
    target_domain: str,
    correlation_id: Optional[str] = None
) -> AutoformalizationResult

async def autoformalize_phase_iii(
    hypothesis_text: str,
    hypothesis_type: str = "causal",
    correlation_id: Optional[str] = None
) -> AutoformalizationResult

async def autoformalize_phase_iv(
    model_description: str,
    efficacy_claim: str,
    correlation_id: Optional[str] = None
) -> AutoformalizationResult
```

### 2. Proof Search Service

**Purpose**: Find proofs using AI-guided strategies

**Key Features**:
- MCTS-guided proof search
- Z3-LeanAide hybrid verification
- Auto tactics
- Counterexample generation

**API**:
```python
async def search_phase_i(
    lean_code: str,
    constraint_type: str = "logical",
    strategy: ProofStrategy = ProofStrategy.Z3_LEAN_HYBRID,
    correlation_id: Optional[str] = None
) -> ProofSearchResult

async def search_phase_ii(
    lean_code: str,
    isomorphism_type: str = "structural",
    correlation_id: Optional[str] = None
) -> ProofSearchResult

async def search_phase_iii(
    lean_code: str,
    hypothesis: Optional[Hypothesis] = None,
    correlation_id: Optional[str] = None
) -> ProofSearchResult

async def search_phase_iv(
    lean_code: str,
    efficacy_claim: str = "",
    correlation_id: Optional[str] = None
) -> ProofSearchResult
```

### 3. Workflow Orchestrator

**Purpose**: Coordinate all phases and services

**Key Features**:
- Problem classification
- Adaptive solver selection
- Stage-aware processing
- Comprehensive error handling

**API**:
```python
async def execute(
    problem_statement: str,
    context: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
) -> WorkflowResult
```

## Phase-Specific Integrations

### Phase I: Epistemic Audit

**Autoformalization**:
- Extract constraints from natural language
- Autoformalize to Lean 4 propositions
- Detect domain automatically

**Proof Search**:
- Verify constraints using Z3-LeanAide hybrid
- Detect contradictions
- Generate counterexamples

**Example**:
```python
result = await service.autoformalize_phase_i(
    constraint_text="All prime numbers greater than 2 are odd",
    constraint_type="arithmetic"
)
```

### Phase II: Isomorphic Mapping

**Autoformalization**:
- Build isomorphic mapping statements
- Generate structural correspondence theorems
- Support category theory formalization

**Proof Search**:
- Verify isomorphisms
- MCTS-guided search for complex proofs
- Validate mechanistic isomorphisms

**Example**:
```python
result = await service.autoformalize_phase_ii(
    mapping_description="Structure-preserving bijection",
    source_domain="natural_numbers",
    target_domain="integers"
)
```

### Phase III: MCTS Refinement

**Autoformalization**:
- Formalize hypotheses
- Generate testable statements
- Support causal and structural hypotheses

**Proof Search**:
- MCTS-guided proof search
- AI-guided tactic selection
- Hypothesis verification

**Example**:
```python
result = await service.autoformalize_phase_iii(
    hypothesis_text="If x > 0 and y > 0, then x + y > 0",
    hypothesis_type="causal"
)
```

### Phase IV: Architectural Synthesis

**Autoformalization**:
- Formalize predictive models
- Generate efficacy claim theorems
- Support optimization problems

**Proof Search**:
- Verify efficacy claims
- Generate formal proofs
- Mathematical validation

**Example**:
```python
result = await service.autoformalization_phase_iv(
    model_description="Linear regression with squared error",
    efficacy_claim="Model converges to true values"
)
```

## Problem Classification

The workflow classifies problems to select appropriate solvers:

**Problem Types**:
- `CONSTRAINT_VERIFICATION`: Verify logical constraints
- `THEOREM_PROVING`: Prove mathematical theorems
- `ISOMORPHISM_DETECTION`: Find isomorphic mappings
- `OPTIMIZATION`: Solve optimization problems
- `HYPOTHESIS_TESTING`: Test hypotheses
- `MODEL_VALIDATION`: Validate predictive models

**Mathematical Domains**:
- Arithmetic
- Algebra
- Logic
- Set Theory
- Calculus
- Graph Theory
- Probability
- Topology
- Category Theory

**Solver Selection**:
- `Z3`: SMT solving
- `LEANAIDE`: AI-assisted proving
- `LEAN4`: Formal verification
- `HYBRID_Z3_LEANAIDE`: Combined approach
- `HYBRID_ALL`: All solvers

## Configuration

All configuration via environment variables (Law of Configuration Explicitness):

```bash
# LeanAide Server
LEANAIDE_HOST=localhost
LEANAIDE_PORT=7654
LEANAIDE_TIMEOUT_MS=30000

# Autoformalization
LEANAIDE_CONFIDENCE_THRESHOLD=0.7
LEANAIDE_MAX_ALTERNATIVES=3

# Proof Search
PROOF_SEARCH_TIMEOUT_MS=60000
PROOF_SEARCH_MAX_DEPTH=100
PROOF_SEARCH_MCTS_ITERATIONS=1000
PROOF_SEARCH_ENABLE_Z3=true
PROOF_SEARCH_CONFIDENCE_THRESHOLD=0.8

# Workflow
WORKFLOW_PHASE_I_TIMEOUT_MS=60000
WORKFLOW_PHASE_II_TIMEOUT_MS=90000
WORKFLOW_PHASE_III_TIMEOUT_MS=120000
WORKFLOW_PHASE_IV_TIMEOUT_MS=90000
WORKFLOW_TIMEOUT_MS=600000
WORKFLOW_MAX_RETRIES=3

# Correlation Tracking
CORRELATION_ID=<uuid>
```

## Failure Handling

Following CLAUDE.md resilience patterns:

### Transient Failures
- **Strategy**: Exponential backoff with jitter
- **Example**: Network timeout to LeanAide server
- **Action**: Retry with increasing delay

### Logic Failures
- **Strategy**: Dead Letter Queue (DLQ)
- **Example**: Invalid constraint syntax
- **Action**: Log to DLQ, continue processing

### System Failures
- **Strategy**: Circuit Breaker
- **Example**: LeanAide server down
- **Action**: Stop calling, use fallback mode

## Data Flow

```
Problem Statement
        │
        ▼
Problem Classification
        │
        ├─▶ Problem Type
        ├─▶ Mathematical Domain
        └─▶ Recommended Solver
        │
        ▼
Phase I: Epistemic Audit
        │
        ├─▶ Extract Constraints
        ├─▶ Autoformalize
        └─▶ Search Proofs
        │
        ▼
Phase II: Isomorphic Mapping
        │
        ├─▶ Identify Domains
        ├─▶ Autoformalize Mappings
        └─▶ Verify Isomorphisms
        │
        ▼
Phase III: MCTS Refinement
        │
        ├─▶ Generate Hypotheses
        ├─▶ Autoformalize
        └─▶ MCTS Proof Search
        │
        ▼
Phase IV: Architectural Synthesis
        │
        ├─▶ Build Predictive Model
        ├─▶ Autoformalize Efficacy
        └─▶ Verify Claims
        │
        ▼
Workflow Result
        │
        ├─▶ All Phase Results
        ├─▶ Summary Statistics
        └─▶ Final Status
```

## Testing

Probe script verifies functionality:

```bash
./probes/check_leanaide_workflow.sh
```

Tests include:
- Python availability
- Dependency imports
- Service loading
- Phase I integration
- Configuration loading

## Performance Considerations

- **Parallel Processing**: Batch autoformalization and proof search
- **Caching**: Lean code generation results
- **Timeouts**: Per-phase and overall timeouts
- **Circuit Breakers**: Prevent cascading failures
- **Retry Logic**: Exponential backoff for transient failures

## Security

- **Input Validation**: All inputs sanitized
- **No Code Execution**: No eval/exec on user input
- **Dependency Isolation**: Air gap from core-projects
- **Containerization**: Docker isolation
- **Non-root User**: Runs as non-root in container

## Future Enhancements

1. **Phase Extensions**: Add more phase-specific strategies
2. **Tactic Library**: Expand auto tactics
3. **Parallel Phases**: Run independent phases in parallel
4. **Distributed Processing**: Support multi-node deployment
5. **Real Collaboration**: Multi-user proof editing
6. **Visualization**: Interactive proof tree display

## References

- CLAUDE.md: Project constitution and principles
- LeanAide Documentation: AI theorem proving
- RESE Documentation: 4-phase pipeline
- Z3 Documentation: SMT solving
- Lean 4 Documentation: Interactive theorem proving
