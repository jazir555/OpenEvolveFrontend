# RESE-LeanAide Workflow Integration Guide

Complete guide for integrating LeanAide's AI-powered theorem proving with RESE's 4-phase pipeline.

## Table of Contents

1. [Overview](#overview)
2. [Phase I Integration](#phase-i-integration)
3. [Phase II Integration](#phase-ii-integration)
4. [Phase III Integration](#phase-iii-integration)
5. [Phase IV Integration](#phase-iv-integration)
6. [Workflow Patterns](#workflow-patterns)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Overview

The RESE-LeanAide workflow integration enables formal verification and mathematical reasoning across all 4 RESE phases:

```
Problem Statement
        │
        ▼
┌───────────────────────────────────────┐
│  Problem Classification               │
│  - Type: Constraint/Theorem/Optimize  │
│  - Domain: Arithmetic/Logic/Graph...  │
│  - Solver: Z3/LeanAide/Hybrid         │
└───────────────────────────────────────┘
        │
        ├─────────────────────────────────────┐
        │                                     │
        ▼                                     ▼
┌──────────────┐                    ┌──────────────┐
│ Phase I      │                    │ Autoformal.  │
│ Epistemic    │──▶ Constraints ───▶│ Service      │
│ Audit        │                    └──────────────┘
└──────────────┘                            │
        │                                     │
        ▼                                     ▼
┌──────────────┐                    ┌──────────────┐
│ Phase II     │                    │ Proof Search │
│ Isomorphic   │──▶ Isomorphisms ──▶│ Service      │
│ Mapping      │                    └──────────────┘
└──────────────┘                            │
        │                                     │
        ▼                                     ▼
┌──────────────┐                    ┌──────────────┐
│ Phase III    │                    │ MCTS-Guided  │
│ MCTS         │──▶ Hypotheses ────▶│ Search       │
│ Refinement   │                    └──────────────┘
└──────────────┘
        │
        ▼
┌──────────────┐
│ Phase IV     │
│ Architectural│──▶ Predictive Model ──▶ Formal Proof
│ Synthesis    │
└──────────────┘
```

## Phase I Integration

### Epistemic Audit with Autoformalization

**Purpose**: Formalize natural language constraints and verify them.

#### Usage Pattern

```python
from src.autoformalization_service import AutoformalizationService
from src.proof_search_service import ProofSearchService

# Initialize services
auto_service = AutoformalizationService()
proof_service = ProofSearchService()

# Step 1: Extract constraints from problem
constraints = [
    "All prime numbers greater than 2 are odd",
    "For all x, if x > 0 then x + 1 > 0",
    "The sum of two positive numbers is positive"
]

# Step 2: Autoformalize each constraint
for constraint in constraints:
    auto_result = await auto_service.autoformalize_phase_i(
        constraint_text=constraint,
        constraint_type="logical",
        correlation_id="audit-001"
    )

    print(f"Theorem: {auto_result.lean_theorem_name}")
    print(f"Lean code:\n{auto_result.lean_code}")

    # Step 3: Search for proof
    proof_result = await proof_service.search_phase_i(
        lean_code=auto_result.lean_code,
        constraint_type="logical",
        strategy=ProofStrategy.Z3_LEAN_HYBRID,
        correlation_id="audit-001"
    )

    if proof_result.proof_found:
        print(f"✓ Constraint verified")
    else:
        print(f"✗ Constraint not verified")
        if proof_result.counterexample:
            print(f"  Counterexample: {proof_result.counterexample}")
```

#### Output Example

```
Theorem: prime_numbers_greater_than_two_are_odd_constraint
Lean code:
import Mathlib

theorem prime_numbers_greater_than_two_are_odd_constraint :
  ∀ (p : Nat), p > 1 → Prime p → Odd p := by
  sorry

✓ Constraint verified
Confidence: 0.85
```

#### Tacit Assumption Mining

```python
# Mine implicit assumptions from constraints
assumptions = await auto_service.autoformalize_phase_i(
    constraint_text="If a function is continuous and differentiable, it is smooth",
    constraint_type="logical"
)

# The autoformalizer will detect:
# - Assumption: Domain is Real numbers
# - Assumption: Standard definition of smoothness
# - Assumption: Function maps ℝ → ℝ
```

#### Contradiction Detection

```python
# Detect contradictions using Z3
proof_result = await proof_service.search_phase_i(
    lean_code="""
theorem contradiction_test :
  (∀ x : Nat, x > 0) ∧ (∃ y : Nat, y = 0) := by
    sorry
""",
    strategy=ProofStrategy.Z3_LEAN_HYBRID
)

if proof_result.counterexample:
    print("Contradiction detected!")
    print(f"Counterexample: {proof_result.counterexample}")
```

## Phase II Integration

### Isomorphic Mapping with Formal Verification

**Purpose**: Formalize and verify isomorphic mappings between domains.

#### Usage Pattern

```python
# Step 1: Identify domains
domains = ["natural_numbers", "integers", "rational_numbers"]

# Step 2: Autoformalize isomorphic mappings
for source, target in itertools.combinations(domains, 2):
    auto_result = await auto_service.autoformalize_phase_ii(
        mapping_description="Canonical embedding preserving structure",
        source_domain=source,
        target_domain=target,
        correlation_id="isomorphism-001"
    )

    print(f"Mapping: {source} → {target}")
    print(f"Theorem: {auto_result.lean_theorem_name}")

    # Step 3: Verify isomorphism
    proof_result = await proof_service.search_phase_ii(
        lean_code=auto_result.lean_code,
        isomorphism_type="structural",
        correlation_id="isomorphism-001"
    )

    if proof_result.proof_found:
        print(f"✓ Isomorphism verified")
        print(f"Confidence: {proof_result.confidence}")
```

#### Functional Dependency Graph (FDG) Construction

```python
# Autoformalize FDG from domain description
fdg_description = """
Functional dependencies in the domain:
- f: X → Y depends on parameter α
- g: Y → Z depends on parameter β
- Composition g∘f: X → Z
"""

auto_result = await auto_service.autoformalize_phase_ii(
    mapping_description=fdg_description,
    source_domain="abstract_algebra",
    target_domain="category_theory",
    correlation_id="fdg-001"
)

# Generated Lean code will include:
# - Structure definitions
# - Morphism preservation
# - Functor laws
```

#### Mechanistic Isomorphism Validation

```python
# Validate I_mech (mechanistic isomorphism) score
proof_result = await proof_service.search_phase_ii(
    lean_code="""
theorem mechanistic_isomorphism :
  ∀ (φ : A → B),
    IsBijection φ →
    PreservesStructure φ →
    PreservesCausality φ := by
      sorry
""",
    isomorphism_type="mechanistic"
)

if proof_result.proof_found:
    # High I_mech score indicates strong mechanistic isomorphism
    print(f"I_mech score: {proof_result.confidence}")
```

## Phase III Integration

### MCTS Refinement with AI-Guided Search

**Purpose**: Test hypotheses using MCTS-guided proof search.

#### Usage Pattern

```python
# Step 1: Generate hypotheses
hypotheses = [
    "If the system is linear, then superposition holds",
    "If the system is causal, then no future input affects past output",
    "If the system is stable, then bounded input produces bounded output"
]

# Step 2: Autoformalize hypotheses
for hypothesis_text in hypotheses:
    auto_result = await auto_service.autoformalize_phase_iii(
        hypothesis_text=hypothesis_text,
        hypothesis_type="causal",
        correlation_id="mcts-001"
    )

    print(f"Hypothesis: {hypothesis_text}")
    print(f"Theorem: {auto_result.lean_theorem_name}")

    # Step 3: MCTS-guided proof search
    proof_result = await proof_service.search_phase_iii(
        lean_code=auto_result.lean_code,
        correlation_id="mcts-001"
    )

    print(f"Proof found: {proof_result.proof_found}")
    print(f"Nodes explored: {proof_result.search_nodes_explored}")
    print(f"Search depth: {proof_result.search_depth}")

    if proof_result.proof_found:
        print(f"✓ Hypothesis confirmed")
        print(f"Proof script:\n{proof_result.proof_script}")
    else:
        print(f"✗ Hypothesis not confirmed")
        print(f"Confidence: {proof_result.confidence}")
```

#### AI-Guided Tactic Selection

```python
# MCTS automatically selects optimal tactics
proof_result = await proof_service.search_phase_iii(
    lean_code="""
theorem complex_hypothesis (X Y Z : Type) (f : X → Y) (g : Y → Z) :
  Bijective f → Bijective g → Bijective (g ∘ f) := by
    sorry
""",
    correlation_id="tactic-selection"
)

# Tactics used by MCTS
for tactic in proof_result.tactics_used:
    print(f"Tactic: {tactic.name}")
    print(f"Confidence: {tactic.confidence}")
    print(f"Explanation: {tactic.explanation}")
```

#### Anomaly Detection

```python
# Detect anomalies in hypotheses
proof_result = await proof_service.search_phase_iii(
    lean_code="""
theorem anomaly_detection :
  ∀ (x : ℝ), x > 0 → x + x > 2*x := by
    sorry
""",
    correlation_id="anomaly-001"
)

# If proof fails with counterexample, anomaly detected
if proof_result.counterexample:
    print("Anomaly detected!")
    print(f"Counterexample: {proof_result.counterexample}")
```

## Phase IV Integration

### Architectural Synthesis with Formal Proofs

**Purpose**: Formally verify predictive models and efficacy claims.

#### Usage Pattern

```python
# Step 1: Define predictive model
model_description = """
Linear regression model:
- y = β₀ + β₁x + ε
- where ε ~ N(0, σ²)
- Ordinary Least Squares estimation
"""

# Step 2: Define efficacy claim
efficacy_claim = """
As n → ∞, the estimated parameters converge to true values:
- β̂₀ → β₀ almost surely
- β̂₁ → β₁ almost surely
"""

# Step 3: Autoformalize efficacy claim
auto_result = await auto_service.autoformalize_phase_iv(
    model_description=model_description,
    efficacy_claim=efficacy_claim,
    correlation_id="synthesis-001"
)

print(f"Theorem: {auto_result.lean_theorem_name}")
print(f"Lean code:\n{auto_result.lean_code}")

# Step 4: Prove efficacy claim
proof_result = await proof_service.search_phase_iv(
    lean_code=auto_result.lean_code,
    efficacy_claim=efficacy_claim,
    correlation_id="synthesis-001"
)

if proof_result.proof_found:
    print(f"✓ Efficacy claim verified")
    print(f"Confidence: {proof_result.confidence}")
else:
    print(f"✗ Efficacy claim not verified")
```

#### Mathematical Validation

```python
# Validate paradigm transformation mathematically
auto_result = await auto_service.autoformalize_phase_iv(
    model_description="Bayesian inference framework",
    efficacy_claim="Posterior converges to true parameter as data increases",
    correlation_id="validation-001"
)

proof_result = await proof_service.search_phase_iv(
    lean_code=auto_result.lean_code,
    efficacy_claim="Posterior consistency holds",
    correlation_id="validation-001"
)

if proof_result.proof_found:
    print("Paradigm transformation mathematically validated")
```

## Workflow Patterns

### Pattern 1: Sequential Phase Execution

```python
from src.leanaide_rese_workflow import LeanAideRESEWorkflow

workflow = LeanAideRESEWorkflow()

result = await workflow.execute(
    problem_statement="Prove that the sum of two even numbers is even",
    context={
        "domain": "number_theory",
        "difficulty": "easy"
    }
)

# Access phase results
for phase_name, phase_result in result.phase_results.items():
    print(f"{phase_name}: {phase_result.status}")
    print(f"  Autoformalizations: {len(phase_result.autoformalization_results)}")
    print(f"  Proofs found: {len(phase_result.proof_search_results)}")
```

### Pattern 2: Batch Processing

```python
# Process multiple constraints in parallel
constraints = [f"Constraint {i}" for i in range(10)]

results = await auto_service.batch_autoformalize(
    items=[{"text": c, "type": "logical"} for c in constraints],
    phase=AutoformalizationPhase.PHASE_I_EPISTEMIC_AUDIT,
    correlation_id="batch-001"
)

successful = sum(1 for r in results if r.success)
print(f"Successfully formalized: {successful}/{len(results)}")
```

### Pattern 3: Iterative Refinement

```python
# Iteratively refine hypothesis based on proof results
hypothesis = "Initial hypothesis text"

for iteration in range(5):
    auto_result = await auto_service.autoformalize_phase_iii(
        hypothesis_text=hypothesis,
        correlation_id=f"iter-{iteration}"
    )

    proof_result = await proof_service.search_phase_iii(
        lean_code=auto_result.lean_code,
        correlation_id=f"iter-{iteration}"
    )

    if proof_result.proof_found:
        print(f"Hypothesis verified in iteration {iteration}")
        break
    else:
        # Refine hypothesis based on feedback
        hypothesis = refine_hypothesis(hypothesis, proof_result)
```

### Pattern 4: Adaptive Solver Selection

```python
# Let workflow classify and select solver automatically
result = await workflow.execute(
    problem_statement="Find isomorphic mapping between sets"
)

classification = result.problem_classification
print(f"Problem type: {classification.problem_type}")
print(f"Domain: {classification.mathematical_domain}")
print(f"Recommended solver: {classification.recommended_solver}")
```

## Best Practices

### 1. Always Use Correlation IDs

```python
# Good
result = await service.autoformalize_phase_i(
    constraint_text="...",
    correlation_id=str(uuid.uuid4())
)

# Bad
result = await service.autoformalize_phase_i(
    constraint_text="..."
)
```

### 2. Handle Errors Gracefully

```python
try:
    result = await workflow.execute(problem_statement)
    if result.overall_status == "completed":
        # Process results
        pass
    else:
        # Handle partial results
        for phase_result in result.phase_results.values():
            if phase_result.errors:
                logger.error(f"Phase errors: {phase_result.errors}")
except Exception as e:
    logger.error(f"Workflow failed: {e}")
```

### 3. Configure Timeouts Appropriately

```python
# For quick checks
config = WorkflowConfig(
    phase_i_timeout_ms=5000,
    phase_ii_timeout_ms=10000,
    phase_iii_timeout_ms=15000,
    phase_iv_timeout_ms=10000
)

# For complex problems
config = WorkflowConfig(
    phase_i_timeout_ms=60000,
    phase_ii_timeout_ms=120000,
    phase_iii_timeout_ms=300000,
    phase_iv_timeout_ms=180000
)
```

### 4. Use Caching for Repeated Problems

```python
# Enable caching
config = WorkflowConfig(enable_caching=True)

# Subsequent calls with same problem will use cache
result1 = await workflow.execute(problem_statement)
result2 = await workflow.execute(problem_statement)  # From cache
```

### 5. Monitor Confidence Scores

```python
result = await service.autoformalize_phase_i(constraint_text)

if result.confidence < 0.7:
    logger.warning(f"Low confidence: {result.confidence}")
    # Consider manual review or alternative approach
```

## Troubleshooting

### Problem: LeanAide Server Not Available

**Symptoms**: Timeout errors, connection refused

**Solution**:
1. Check if LeanAide server is running: `curl http://localhost:7654`
2. Adapter will use simulation mode automatically
3. For full functionality, start LeanAide server

### Problem: Low Confidence Scores

**Symptoms**: Autoformalization confidence < 0.7

**Solution**:
1. Refine natural language input
2. Use more precise mathematical terminology
3. Provide context through additional parameters

### Problem: Proof Search Timeout

**Symptoms**: Phase timeout, no proof found

**Solution**:
1. Increase timeout values
2. Simplify theorem statement
3. Use different proof strategy
4. Break complex proof into lemmas

### Problem: Import Errors

**Symptoms**: ModuleNotFoundError, ImportError

**Solution**:
1. Install dependencies: `pip install -r requirements.txt`
2. Check Python path includes adapter directory
3. Ensure working directory is correct

## Advanced Topics

### Custom Proof Strategies

```python
class CustomProofStrategy(ProofStrategy):
    CUSTOM = "custom"

proof_result = await proof_service.search_phase_i(
    lean_code=code,
    strategy=CustomProofStrategy.CUSTOM
)
```

### Domain-Specific Autoformalization

```python
# Register custom domain detector
def detect_custom_domain(text: str) -> FormalizationDomain:
    if "manifold" in text.lower():
        return FormalizationDomain.TOPOLOGY
    # ... custom logic

# Inject into service
auto_service._detect_domain = detect_custom_domain
```

### Extending Phase Executors

```python
class CustomPhaseExecutor(PhaseExecutor):
    phase_name = "custom_phase"

    async def execute(self, input_data, correlation_id):
        # Custom phase logic
        pass

# Register with workflow
workflow.custom_phase = CustomPhaseExecutor(config, event_bus)
```

## References

- [CLAUDE.md](../../CLAUDE.md) - Project principles
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture
- [README.md](../README.md) - Usage guide
- LeanAide Documentation
- RESE Documentation
- Lean 4 Documentation
