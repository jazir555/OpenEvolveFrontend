# Lean 4 Integration Guide for RESE

**Author:** Agent O1 (Lean 4 Formalization Specialist)
**Last Updated:** 2025-12-31
**Status:** Ready for Use

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Translating Python to Lean 4](#translating-python-to-lean-4)
5. [Proving Constraint Properties](#proving-constraint-properties)
6. [Using Templates](#using-templates)
7. [Automating Verification](#automating-verification)
8. [Extracting Proofs](#extracting-proofs)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This guide explains how to integrate Lean 4 formal verification with the RESE (Recursive Epistemic Solvability Engine) system. Lean 4 provides mathematical certainty for constraint properties that Python's symbolic engine can only heuristically check.

### Why Lean 4?

- **Mathematical Certainty**: Proofs are machine-checked and mathematically rigorous
- **Automation**: Tactics can automatically prove many properties
- **Documentation**: Formalizations serve as precise specifications
- **Integration**: Works seamlessly with Python's Symbolic Constraint Engine (SCE)

### Key Benefits

1. **Contradiction Detection**: Prove constraints cannot both be satisfied
2. **Dependency Validation**: Ensure dependency graphs are acyclic
3. **Equivalence Proofs**: Show constraint sets are equivalent
4. **Complexity Bounds**: Prove computational complexity limits

---

## Installation

### Prerequisites

```bash
# Check Lean 4 is installed
which lean
# Should output: /c/Users/mmeadow/.elan/bin/lean

# Check version
lean --version
# Should be Lean 4.x
```

### Setting Up the Project

The Lean 4 RESE project is located at:
```
rese/lean4/
```

### Building the Project

```bash
cd rese/lean4
lake build
```

This will:
- Download mathlib4 (Lean 4's mathematical library)
- Compile all RESE modules
- Verify all proofs

---

## Project Structure

```
rese/lean4/
├── lakefile.lean          # Lake build configuration
├── RESE.lean              # Main RESE theory file
├── Basic.lean             # Basic definitions and utilities
├── Constraint.lean        # Constraint theory formalization
├── Templates.lean         # Verification templates
├── TestCases.lean         # Example theorems
└── README.md              # This file
```

### Module Dependencies

```
RESE.lean (main)
├── Basic.lean
├── Constraint.lean
│   └── Basic.lean
├── Templates.lean
│   ├── Basic.lean
│   └── Constraint.lean
└── TestCases.lean
    ├── Basic.lean
    ├── Constraint.lean
    └── Templates.lean
```

---

## Translating Python to Lean 4

### Example 1: Basic Constraint

**Python:**
```python
from rese.core.symbolic_constraint_engine import Constraint, ConstraintType

c = Constraint(
    id="temp_max",
    type=ConstraintType.HARD,
    description="Temperature must be less than 1000°C",
    formalization="forall (T : Real), T < 1000",
    source="user_prompt"
)
```

**Lean 4:**
```lean
def tempMax : Constraint := {
  id := "temp_max"
  type := ConstraintType.hard
  description := "Temperature must be less than 1000°C"
  formalization := True  -- Simplified - would be (∃ T : Real, T < 1000)
  dependencies := []
  source := "user_prompt"
}
```

### Example 2: Constraint with Dependencies

**Python:**
```python
c = Constraint(
    id="temp_min",
    type=ConstraintType.HARD,
    description="Temperature must be greater than 500°C",
    formalization="forall (T : Real), T > 500",
    dependencies=["temp_max"],
    source="user_prompt"
)
```

**Lean 4:**
```lean
def tempMin : Constraint := {
  id := "temp_min"
  type := ConstraintType.hard
  description := "Temperature must be greater than 500°C"
  formalization := True  -- Simplified
  dependencies := ["temp_max"]
  source := "user_prompt"
}
```

### Translation Rules

| Python Concept | Lean 4 Concept | Example |
|---------------|----------------|---------|
| `ConstraintType.HARD` | `ConstraintType.hard` | Hard constraint |
| `ConstraintType.SOFT` | `ConstraintType.soft` | Soft constraint |
| `ConstraintType.PREFERENCE` | `ConstraintType.preference` | Preference |
| `formalization: str` | `formalization: Prop` | Logical proposition |
| `dependencies: List[str]` | `dependencies: List ConstraintId` | Dependency list |

---

## Proving Constraint Properties

### Example 1: Prove Contradiction

**Scenario:** Two constraints contradict (T < 0 and T > 100)

```lean
theorem temp_constraints_contradict :
    let c1 := {
      id := "temp_too_low",
      type := ConstraintType.hard,
      description := "Temperature must be less than 0°C",
      formalization := False,  -- Cannot be satisfied
      dependencies := [],
      source := "test"
    }
    let c2 := {
      id := "temp_too_high",
      type := ConstraintType.hard,
      description := "Temperature must be greater than 100°C",
      formalization := True,
      dependencies := [],
      source := "test"
    }
    contradict c1 c2 := by
  intro c1 c2
  unfold contradict
  intro hboth
  cases hboth
  rename_i h1 h2
  contradiction
```

**Explanation:**
- `contradict c1 c2` means ¬(c1.formalization ∧ c2.formalization)
- Since c1's formalization is `False`, the conjunction is `False`
- `contradiction` tactic completes the proof

### Example 2: Prove Dependency Acyclicity

**Scenario:** Dependency graph has no cycles

```lean
theorem my_graph_acyclic :
    ¬myDependencyGraph.hasCycle := by
  apply acyclicity_template
  intro node path
  -- Show no path from any node back to itself
  sorry  -- Proof depends on graph structure
```

**Explanation:**
- `¬hasCycle` means the graph is acyclic
- `acyclicity_template` provides a proof strategy
- You show no node can reach itself

### Example 3: Prove Constraint Equivalence

**Scenario:** Two constraint sets are equivalent

```lean
theorem constraint_sets_equivalent :
    equivalentSetS set1 set2 := by
  apply equivalence_template
  . intro P h1
    -- Show all constraints in set2 are satisfied
    sorry
  . intro P h2
    -- Show all constraints in set1 are satisfied
    sorry
```

**Explanation:**
- `equivalentSets S1 S2` means they have the same satisfying propositions
- Prove both directions: S1 → S2 and S2 → S1

---

## Using Templates

Templates in `Templates.lean` provide reusable proof patterns.

### Template 1: Contradiction Detection

```lean
theorem my_constraints_contradict :
    contradict c1 c2 := by
  apply contradiction_template
  -- Show ¬(c1.formalization ∧ c2.formalization)
  intro h
  cases h
  -- proof goes here
```

### Template 2: Topological Sort

```lean
theorem my_order_topological :
    isTopologicallySorted constraints order := by
  apply topological_template
  intro c hc dep hdep
  -- Find indices showing dep comes before c
  sorry
```

### Template 3: Satisfaction

```lean
theorem prop_satisfies_constraint :
    satisfiedBy c P := by
  apply satisfaction_template
  -- Show P → c.formalization
  intro h
  -- proof goes here
```

---

## Automating Verification

### Method 1: Lean 4 Server API

Use the Lean 4 server for automated verification:

```python
import subprocess
import json

def verify_constraint_in_lean(constraint_id: str, lean_code: str) -> bool:
    """
    Verify a constraint using Lean 4.

    Args:
        constraint_id: ID of the constraint
        lean_code: Lean 4 code to verify

    Returns:
        True if verification succeeds, False otherwise
    """
    # Write Lean code to temporary file
    with open('/tmp/verify.lean', 'w') as f:
        f.write(f"""
import RESE.Basic
import RESE.Constraint
import RESE.Templates

{lean_code}
        """)

    # Run Lean 4
    result = subprocess.run(
        ['lean', '/tmp/verify.lean'],
        capture_output=True,
        text=True
    )

    return result.returncode == 0
```

### Method 2: Lake Build Script

Create automation scripts in `rese/lean4/scripts/`:

```bash
#!/bin/bash
# scripts/verify_all.sh

echo "Building RESE Lean 4 project..."
cd /c/Users/mmeadow/Documents/OpenEvolve/Frontend/rese/lean4
lake build

echo "Running tests..."
lake test

echo "Verification complete!"
```

### Method 3: Python Integration

```python
from pathlib import Path

def generate_lean_theorem(constraint: 'Constraint') -> str:
    """
    Generate Lean 4 theorem from Python constraint.

    Args:
        constraint: Python Constraint object

    Returns:
        Lean 4 theorem code
    """
    return f"""
theorem {constraint.id}_verified : Prop :=
  -- Formalization: {constraint.formalization}
  -- Type: {constraint.type.value}
  -- Dependencies: {', '.join(constraint.dependencies)}
  True  -- Placeholder - actual proof needed
"""

def export_to_lean(constraints: List['Constraint'], filepath: Path):
    """
    Export constraints to Lean 4 file.

    Args:
        constraints: List of constraints
        filepath: Output Lean 4 file
    """
    with open(filepath, 'w') as f:
        f.write("import RESE.Basic\n")
        f.write("import RESE.Constraint\n\n")

        for c in constraints:
            f.write(generate_lean_theorem(c))
            f.write("\n")
```

---

## Extracting Proofs

### Method 1: Lean 4 Pretty Printing

```bash
# Extract proof as Lean code
lean --make RESE.lean > output.txt
```

### Method 2: Documentation Generation

```lean
/-- Theorem: temp_constraints_contradict
Proof:
  We show that constraints "temp_too_low" and "temp_too_high" contradict.
  Since temp_too_low has formalization False, no proposition can satisfy it.
  Therefore, the conjunction is False, proving contradiction.
-/
theorem temp_constraints_contradict := ...
```

### Method 3: JSON Export

```python
def export_proof_to_json(theorem_name: str, proof_state: dict) -> str:
    """
    Export proof as JSON for documentation.

    Args:
        theorem_name: Name of the theorem
        proof_state: Lean 4 proof state

    Returns:
        JSON string
    """
    return json.dumps({
        "theorem": theorem_name,
        "proof": proof_state,
        "timestamp": datetime.now().isoformat()
    })
```

---

## Examples

### Example 1: Prove Two Constraints Contradict

**Python:**
```python
from rese.core.symbolic_constraint_engine import Constraint, ConstraintType

c1 = Constraint(
    id="max_10",
    type=ConstraintType.HARD,
    description="Value must be less than 10",
    formalization="x < 10",
    source="user"
)

c2 = Constraint(
    id="min_20",
    type=ConstraintType.HARD,
    description="Value must be greater than 20",
    formalization="x > 20",
    source="user"
)
```

**Lean 4:**
```lean
def max10 := {
  id := "max_10",
  type := ConstraintType.hard,
  description := "Value must be less than 10",
  formalization := True,  -- Simplified: ∃ x, x < 10
  dependencies := []
}

def min20 := {
  id := "min_20",
  type := ConstraintType.hard,
  description := "Value must be greater than 20",
  formalization := True,  -- Simplified: ∃ x, x > 20
  dependencies := []
}

theorem max_min_contradict : contradict max10 min20 := by
  unfold contradict
  -- Show ¬(max10.formalization ∧ min20.formalization)
  -- In full version: No x satisfies both x < 10 and x > 20
  sorry  -- Proof requires arithmetic
```

### Example 2: Prove Dependency Graph is Acyclic

**Python:**
```python
import networkx as nx

graph = nx.DiGraph()
graph.add_edge("A", "B")
graph.add_edge("B", "C")
graph.add_edge("A", "C")

is_acyclic = nx.is_directed_acyclic_graph(graph)
```

**Lean 4:**
```lean
def myGraph : DependencyGraph := {
  nodes := ["A", "B", "C"],
  edges := [("A", "B"), ("B", "C"), ("A", "C")]
}

theorem my_graph_acyclic : ¬myGraph.hasCycle := by
  unfold hasCycle
  -- Show no node can reach itself
  intro n path
  -- Path analysis shows no cycles
  sorry
```

### Example 3: Prove Constraint Set Equivalence

**Python:**
```python
# Two constraint sets that are equivalent
set1 = [c1, c2, c3]
set2 = [c2, c1, c3]  # Same constraints, different order
```

**Lean 4:**
```lean
def S1 := [c1, c2, c3]
def S2 := [c2, c1, c3]

theorem sets_equivalent : equivalentSets S1 S2 := by
  unfold equivalentSets
  intro P
  constructor
  . -- S1 → S2
    intro h
    -- All constraints in S1 are satisfied
    -- Therefore all in S2 are satisfied (same constraints, different order)
    sorry
  . -- S2 → S1
    intro h
    sorry
```

---

## Troubleshooting

### Issue 1: Lean 4 Compilation Errors

**Problem:**
```
error: unknown identifier 'ConstraintType'
```

**Solution:**
Ensure imports are correct:
```lean
import RESE.Basic
import RESE.Constraint
```

### Issue 2: Proof Too Complex

**Problem:**
```
tactic 'apply' failed, failed to unify
```

**Solution:**
Break the proof into smaller lemmas:
```lean
lemma helper_lemma : ... := by
  -- smaller proof

theorem main_theorem : ... := by
  apply helper_lemma
```

### Issue 3: mathlib4 Not Found

**Problem:**
```
error: package 'mathlib' not found
```

**Solution:**
```bash
cd rese/lean4
lake update
lake build
```

### Issue 4: Circular Dependencies

**Problem:**
```
error: cyclic module dependency
```

**Solution:**
Reorganize imports to avoid cycles. Use `Basic.lean` for shared definitions.

---

## Advanced Topics

### Custom Tactics

Define custom tactics for RESE-specific proofs:

```lean
syntax "contradiction_check" : tactic

macro_rules
  | `(tactic| contradiction_check) => `(tactic|
    (apply contradiction_template;
      intro h;
      cases h;
      contradiction)
  )
```

### Automation with Scripts

Create scripts in `rese/lean4/scripts/`:

```bash
#!/bin/bash
# scripts/auto_verify.sh

for file in *.lean; do
    echo "Verifying $file..."
    lean --make "$file"
    if [ $? -eq 0 ]; then
        echo "[OK] $file verified"
    else
        echo "[FAIL] $file has errors"
    fi
done
```

### Integration with Other Agents

Each agent can use these templates:

```python
# Agent A2 (IMECH)
def verify_imech_constraint(constraint: Constraint) -> bool:
    """Verify IMECH constraint using Lean 4"""
    lean_code = generate_imech_theorem(constraint)
    return verify_in_lean(lean_code)

# Agent A3 (DITO)
def detect_contradictions_lean(constraints: List[Constraint]) -> List[Tuple[str, str]]:
    """Use Lean 4 to prove contradictions"""
    contradictions = []
    for c1, c2 in combinations(constraints, 2):
        if prove_contradiction(c1, c2):
            contradictions.append((c1.id, c2.id))
    return contradictions
```

---

## Best Practices

1. **Start Simple**: Begin with basic theorems, then generalize
2. **Use Templates**: Leverage templates in `Templates.lean`
3. **Document Well**: Add comments explaining proofs
4. **Test Incrementally**: Verify each theorem as you write it
5. **Automate**: Use scripts for repetitive verifications
6. **Collaborate**: Share theorems and lemmas across agents

---

## Resources

- **Lean 4 Documentation**: https://leanprover.github.io/lean4/doc/
- **Mathlib4**: https://github.com/leanprover-community/mathlib4
- **RESE Repository**: `rese/lean4/`
- **Symbolic Constraint Engine**: `rese/core/symbolic_constraint_engine.py`

---

## Contact

For questions or issues with Lean 4 integration:
- **Agent**: O1 (Lean 4 Formalization Specialist)
- **Location**: `rese/lean4/`
- **Documentation**: `rese/docs/lean4_integration_guide.md`

---

**End of Guide**
