# OpenEvolve-Math Integration Guide

## Overview

This guide describes how to integrate **OpenEvolve bubbles** (problem-solving workflow) with **Mathematical Verification bubbles** (Lean 4 + Z3) to create coherent, end-to-end mathematical workflows.

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OpenEvolve-Math Integrated System                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────┐ │
│  │   OpenEvolve Layer  │      │   Integration Layer │      │  Math Layer │ │
│  │                     │      │                     │      │             │ │
│  │ • DecompositionNode │◄────►│ • OpenEvolveMath    │◄────►│ • Lean      │ │
│  │ • AssemblyNode      │      │   BridgeNode        │      │   Nodes     │ │
│  │ • SolutionNode      │      │                     │      │ • Z3 Nodes  │ │
│  │ • SubproblemNode    │◄────►│ • MathWorkflow      │◄────►│ • Math      │ │
│  │ • VerificationNode  │      │   OrchestratorNode  │      │   Pipeline  │ │
│  │ • Knowledge* Nodes  │◄────►│                     │◄────►│ • Support   │ │
│  └─────────────────────┘      └─────────────────────┘      │   Nodes     │ │
│           ▲                            ▲                   └─────────────┘ │
│           │                            │                          ▲        │
│           └────────────────────────────┴──────────────────────────┘        │
│                              Coherent Workflows                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Integration Nodes

### 1. OpenEvolveMathBridgeNode
**Purpose:** Bridge between OpenEvolve and Math Verification bubbles

**Key Operations:**
- `route_to_verification` - Route problem to appropriate verifier
- `convert_problem` - Convert OpenEvolve format to math format
- `integrate_result` - Convert verification result to OpenEvolve format
- `classify_and_route` - Classify then route to verifier
- `verify_subproblem` - Verify decomposed subproblem
- `batch_verify` - Verify multiple subproblems
- `formalize_solution` - Formalize OpenEvolve solution

**Usage:**
```python
# Route problem to verification
bridge = OpenEvolveMathBridgeNode(config={
    "operation": "route_to_verification",
    "preferred_verifier": "auto",
    "cross_verify": True
})
result = bridge.execute({
    "problem": {"id": "p1", "description": "Prove that n + 0 = n"}
}, context)
```

---

### 2. MathWorkflowOrchestratorNode
**Purpose:** Orchestrate coherent multi-step workflows

**Pre-built Templates:**
1. **formalize_and_verify** - Convert NL to formal proof
2. **decompose_and_verify** - Decompose and verify parts
3. **evolve_solution** - Evolve with continuous verification
4. **conjecture_to_theorem** - Convert conjecture to theorem
5. **counterexample_search** - Search before proving
6. **proof_optimization** - Optimize existing proof
7. **complete_verification** - End-to-end pipeline

**Usage:**
```python
# Execute formalization workflow
orchestrator = MathWorkflowOrchestratorNode(config={
    "operation": "execute_template",
    "template": "formalize_and_verify"
})
result = orchestrator.execute({
    "input_data": {"text": "For all natural numbers n, n + 0 = n"}
}, context)
```

---

## Coherent Workflow Examples

### Workflow 1: Problem Formalization and Verification

```
[OpenEvolve Problem]
    ↓
[DecompositionNode] → Break into subproblems
    ↓
[OpenEvolveMathBridgeNode] → Convert to math format
    ├─► classify_and_route
    ├─► convert_problem
    └─► route_to_verification
    ↓
[MathProblemClassificationNode] → Classify domain/type
    ↓
[LeanAutoformalizationNode] → Convert NL to Lean
    ↓
[MathTacticRecommendationNode] → Recommend tactics
    ↓
[LeanProofCheckingNode] → Verify proof
    ↓
[OpenEvolveMathBridgeNode] → Convert back
    ├─► integrate_result
    ↓
[AssemblyNode] → Assemble verified solution
    ↓
[MathVerificationDashboardNode] → Generate report
```

**Code Example:**
```python
# Complete workflow execution
workflow = [
    ("DecompositionNode", {"operation": "decompose"}),
    ("OpenEvolveMathBridgeNode", {"operation": "convert_problem"}),
    ("MathProblemClassificationNode", {"operation": "classify"}),
    ("LeanAutoformalizationNode", {"operation": "autoformalize"}),
    ("MathTacticRecommendationNode", {"operation": "recommend"}),
    ("LeanProofCheckingNode", {"operation": "check_proof"}),
    ("OpenEvolveMathBridgeNode", {"operation": "integrate_result"}),
    ("AssemblyNode", {"operation": "assemble"}),
    ("MathVerificationDashboardNode", {"operation": "generate_report"})
]

orchestrator = MathWorkflowOrchestratorNode(config={
    "operation": "custom_workflow",
    "custom_steps": workflow
})
```

---

### Workflow 2: Decomposed Subproblem Verification

```
[Complex Theorem]
    ↓
[DecompositionNode] → Break into lemmas
    ├─► Lemma 1: Base case
    ├─► Lemma 2: Inductive step
    └─► Lemma 3: Main result
    ↓
[OpenEvolveMathBridgeNode] → batch_verify
    ↓
For each lemma:
    ├─► [MathProblemClassificationNode] → Classify
    ├─► [LeanAutoformalizationNode] → Formalize
    ├─► [LeanProofCheckingNode] → Verify
    └─► [MathProofSimplificationNode] → Simplify
    ↓
[AssemblyNode] → Compose verified lemmas
    ↓
[MathVerificationPipelineNode] → Cross-verify
    ↓
[SolutionNode] → Final verified solution
```

---

### Workflow 3: Evolution with Verification

```
[Initial Solution]
    ↓
[KnowledgeEvolutionNode] → Evolve solution
    ↓
[OpenEvolveMathBridgeNode] → verify_subproblem
    ├─► Check each generation
    ↓
[Z3ConstraintSolvingNode] → Quick check
    ├─► If SAT, continue
    ├─► If UNSAT, reject
    ↓
[MathCounterexampleNode] → Verify no counterexamples
    ↓
[LeanProofCheckingNode] → Formal verification
    ↓
[MathProofSimplificationNode] → Optimize final
    ↓
[Verified Solution]
```

---

### Workflow 4: Conjecture to Theorem

```
[Mathematical Conjecture]
    ↓
[MathConjectureNode] → Generate from patterns
    ├─► test_conjecture
    ├─► rank_conjectures
    ↓
[MathCounterexampleNode] → Find counterexamples
    ├─► If found: refine conjecture
    ├─► If not: proceed to proof
    ↓
[MathProblemClassificationNode] → Classify
    ↓
[LeanAutoformalizationNode] → Formalize
    ↓
[MathLibrarySearchNode] → Find related theorems
    ↓
[MathTacticRecommendationNode] → Get tactics
    ↓
[MathInductionHelperNode] → Setup proof (if needed)
    ↓
[LeanProofCheckingNode] → Verify proof
    ↓
[MathVerificationDashboardNode] → Document theorem
```

---

### Workflow 5: Counterexample-Driven Development

```
[Conjectured Property]
    ↓
[MathProblemClassificationNode] → Quick classify
    ↓
[MathCounterexampleNode] → Search
    ├─► find_counterexample
    ├─► find_all (small cases)
    ├─► analyze_failure
    ↓
If counterexamples found:
    ├─► [MathConjectureNode] → suggest_fix
    ├─► Refine conjecture
    └─► Loop back
    ↓
If no counterexamples:
    ├─► [Z3ConstraintSolvingNode] → check_sat
    ├─► [Z3TheoremProvingNode] → attempt_proof
    └─► [LeanProofCheckingNode] → formal_verify
    ↓
[Verified Theorem]
```

---

## Data Flow Between Layers

### OpenEvolve → Math Conversion

```python
# OpenEvolve Problem Format
openevolve_problem = {
    "id": "problem_001",
    "title": "Sum of Evens",
    "description": "Prove that the sum of two even numbers is even",
    "type": "theorem",
    "constraints": ["x, y are even integers"],
    "requirements": ["x + y is even"],
    "difficulty": "intermediate"
}

# Bridge converts to Math Format
math_problem = {
    "statement": "For all even integers x and y, x + y is even",
    "domain": "number_theory",
    "problem_type": "theorem",
    "verification_target": "formal_proof"
}
```

### Math → OpenEvolve Conversion

```python
# Math Verification Result
math_result = {
    "status": "verified",
    "proof": "theorem sum_even...",
    "system": "lean",
    "confidence": 0.95
}

# Bridge converts to OpenEvolve Format
openevolve_result = {
    "problem_id": "problem_001",
    "verification_status": "verified",
    "formal_proof": "theorem sum_even...",
    "quality_score": 0.95,
    "reliability": "high"
}
```

---

## Integration Best Practices

### 1. Classification First
Always classify problems before routing:
```python
# Good: Classify then route
classifier = MathProblemClassificationNode()
classification = classifier.execute({"problem": problem}, context)

# Route based on classification
if classification["domain"] == "number_theory":
    verifier = "lean"
elif classification["type"] == "constraint":
    verifier = "z3"
```

### 2. Progressive Verification
Start with quick checks, progress to formal:
```python
workflow = [
    ("MathCounterexampleNode", "find_counterexample"),  # Quick
    ("Z3ConstraintSolvingNode", "check_sat"),           # Medium
    ("LeanProofCheckingNode", "check_proof")            # Thorough
]
```

### 3. Cross-Verification
Use both Lean and Z3 when possible:
```python
bridge = OpenEvolveMathBridgeNode(config={
    "cross_verify": True,  # Enable cross-verification
    "preferred_verifier": "lean"
})
```

### 4. Error Handling
Handle integration failures gracefully:
```python
try:
    result = bridge.execute(inputs, context)
except NodeExecutionError as e:
    # Fallback to simpler verification
    result = z3_node.execute(inputs, context)
```

---

## Complete Integration Example

```python
# Full integration example
from bubblelabs_nodes.openevolve_math_bridge_node import OpenEvolveMathBridgeNode
from bubblelabs_nodes.math_workflow_orchestrator_node import MathWorkflowOrchestratorNode

# 1. Define OpenEvolve problem
problem = {
    "id": "theorem_001",
    "title": "Commutativity of Addition",
    "description": "Prove that addition of natural numbers is commutative",
    "type": "theorem",
    "difficulty": "elementary"
}

# 2. Use orchestrator for complete workflow
orchestrator = MathWorkflowOrchestratorNode(config={
    "operation": "formalize_and_verify"
})

result = orchestrator.execute({
    "input_data": problem,
    "stop_on_error": True,
    "collect_metrics": True
}, context)

# 3. Access results
print(f"Workflow completed: {result['steps_executed']} steps")
print(f"Template used: {result['template']}")
print(f"Final output: {result['final_output']}")
```

---

## Workflow Templates Reference

| Template | Description | Use Case |
|----------|-------------|----------|
| **formalize_and_verify** | NL → Formal proof | New theorems |
| **decompose_and_verify** | Break & verify parts | Complex proofs |
| **evolve_solution** | Evolve with verification | Optimization |
| **conjecture_to_theorem** | Pattern → Proof | Discovery |
| **counterexample_search** | Search before prove | Validation |
| **proof_optimization** | Simplify existing | Refinement |
| **complete_verification** | End-to-end | Production |

---

## Summary

The OpenEvolve-Math integration provides:

✅ **Seamless Data Flow** between problem-solving and verification  
✅ **Pre-built Workflows** for common mathematical tasks  
✅ **Flexible Routing** to appropriate verification systems  
✅ **Cross-Verification** using both Lean and Z3  
✅ **Progressive Verification** from quick checks to formal proofs  
✅ **Coherent Pipelines** that combine the best of both worlds  

**Total Integrated Bubbles:** 50+ (33 OpenEvolve + 17 Math Verification + Bridge nodes)
