# Symbolic Constraint Engine (SCE) Integration Guide

**Target Audience**: Developers integrating SCE into RESE modules
**Author**: Agent A1
**Status**: Active Implementation
**Version**: 1.0.0

---

## Table of Contents

1. [Overview](#overview)
2. [Integration Architecture](#integration-architecture)
3. [Stage-Specific Integration](#stage-specific-integration)
4. [Dependency Management](#dependency-management)
5. [Common Integration Patterns](#common-integration-patterns)
6. [API Usage Examples](#api-usage-examples)
7. [Testing and Validation](#testing-and-validation)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The Symbolic Constraint Engine (SCE) is the foundational module for the RESE system. All other modules interact with SCE to manage, validate, and enforce constraints throughout the RESE pipeline.

### What is SCE?

SCE provides:
- **Formal constraint storage** with Lean 4 formalizations
- **Dependency tracking** via directed acyclic graph (DAG)
- **Conflict detection** (basic version, enhanced by DITO)
- **Type system** (Hard, Soft, Preference constraints)
- **Verification tracking** for Lean 4 theorems

### Who Uses SCE?

| Stage | Module | Interaction Type |
|-------|--------|------------------|
| Stage 1 | Input Processing | Extract and create constraints |
| Stage 5 | Epistemic Audit (Φ) | Validate constraint consistency |
| Stage 6 | Isomorphic Resonance (Ψ) | Transform constraint structures |
| Stage 7 | Monte Carlo Refinement (Γ) | Optimize constraint satisfaction |
| Stage 8 | Architectural Synthesis (Δ) | Assemble constraint-based solutions |

---

## Integration Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RESE Pipeline                         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Symbolic Constraint Engine (SCE)            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Constraints  │→ │ Dependencies │→ │ Conflicts    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────┼─────────────────┐
        │                 │                 │
    ┌───▼────┐       ┌───▼────┐       ┌───▼────┐
    │  LLTL  │       │  DITO  │       │ Lean 4 │
    └────────┘       └────────┘       └────────┘
```

### Module Interactions

```
Stage 1 (Input Processing)
    ↓ creates constraints
SCE
    ↓ validates
Stage 5 (Epistemic Audit)
    ↓ transforms
Stage 6 (Isomorphic Resonance)
    ↓ optimizes
Stage 7 (Monte Carlo Refinement)
    ↓ assembles
Stage 8 (Architectural Synthesis)
```

---

## Stage-Specific Integration

### Stage 1: Input Processing

**Purpose**: Extract constraints from user prompts and system requirements

**Integration Pattern**: Constraint Creation

```python
from rese.core.symbolic_constraint_engine import (
    SymbolicConstraintEngine,
    Constraint,
    ConstraintType
)

class InputProcessor:
    def __init__(self, sce: SymbolicConstraintEngine):
        self.sce = sce

    def extract_constraints(self, user_prompt: str) -> List[Constraint]:
        """
        Extract constraints from natural language prompt

        Args:
            user_prompt: User's problem description

        Returns:
            List of extracted constraints
        """
        constraints = []

        # Example: Extract "must be less than" constraints
        if "less than" in user_prompt:
            constraint = Constraint(
                id="extracted_1",
                type=ConstraintType.HARD,
                description=self._extract_description(user_prompt),
                formalization=self._generate_formalization(user_prompt),
                source="user_prompt"
            )
            constraints.append(constraint)

        # Add all to SCE
        for constraint in constraints:
            self.sce.add_constraint(constraint)

        return constraints

    def _extract_description(self, prompt: str) -> str:
        """Extract human-readable description"""
        # NLP processing here
        return "Extracted constraint"

    def _generate_formalization(self, prompt: str) -> str:
        """Generate Lean 4 formalization"""
        # Formalization logic here
        return "forall (x : Type), x < limit"
```

**Key Points**:
- Always validate constraints before adding
- Use `source="user_prompt"` for user-specified constraints
- Use `source="system_inferred"` for derived constraints
- Set dependencies if constraints are related

---

### Stage 5: Epistemic Audit (Φ₁, Φ₁.₅, Φ₂, Φ₃)

**Purpose**: Validate constraint consistency, detect contradictions, mine assumptions

**Integration Pattern**: Constraint Validation

```python
from rese.core.symbolic_constraint_engine import ConstraintType

class EpistemicAuditor:
    def __init__(self, sce: SymbolicConstraintEngine):
        self.sce = sce

    def audit_constraints(self) -> Dict[str, any]:
        """
        Perform epistemic audit on constraints

        Returns:
            Audit results with conflicts, assumptions, biases
        """
        results = {
            "conflicts": [],
            "assumptions": [],
            "biases": [],
            "validation_status": "unknown"
        }

        # 1. Detect conflicts
        conflicts = self.sce.detect_conflicts()
        results["conflicts"] = conflicts

        # 2. Validate dependencies
        if not self.sce.validate_dependencies():
            results["validation_status"] = "circular_dependencies"
            return results

        # 3. Mine tacit assumptions (Φ₁.₅)
        assumptions = self._mine_assumptions()
        results["assumptions"] = assumptions

        # 4. Check for cognitive biases (Φ₂)
        biases = self._detect_biases()
        results["biases"] = biases

        results["validation_status"] = "valid"
        return results

    def _mine_assumptions(self) -> List[Constraint]:
        """
        Mine tacit assumptions from constraints

        Φ₁.₅: Tacit Assumption Mining
        """
        assumptions = []

        # Look for implicit assumptions
        for constraint in self.sce.get_all_constraints():
            if "implicitly" in constraint.description.lower():
                assumption = Constraint(
                    id=f"assumption_{constraint.id}",
                    type=ConstraintType.SOFT,
                    description=f"Assumption: {constraint.description}",
                    formalization=f"assumption_{constraint.formalization}",
                    source="assumption_mining",
                    dependencies=[constraint.id]
                )
                assumptions.append(assumption)
                self.sce.add_constraint(assumption)

        return assumptions

    def _detect_biases(self) -> List[str]:
        """
        Detect cognitive biases in constraints

        Φ₂: Metacognitive Debiasing
        """
        biases = []

        # Check for common biases
        all_constraints = self.sce.get_all_constraints()

        # Confirmation bias: over-reliance on preferred outcomes
        preference_count = len(self.sce.get_constraints_by_type(
            ConstraintType.PREFERENCE
        ))
        if preference_count > len(all_constraints) * 0.5:
            biases.append("confirmation_bias")

        return biases
```

**Key Points**:
- Use `detect_conflicts()` for basic contradiction detection
- Use `validate_dependencies()` to check for circular dependencies
- Create new constraints for discovered assumptions
- Use SOFT type for assumptions and biases

---

### Stage 6: Isomorphic Resonance (Ψ₁, Ψ₂, Ψ₃)

**Purpose**: Transform constraint structures, map ontologies, invert constraints

**Integration Pattern**: Constraint Transformation

```python
class IsomorphicResonanceProcessor:
    def __init__(self, sce: SymbolicConstraintEngine):
        self.sce = sce

    def transform_constraints(self, target_ontology: str) -> List[Constraint]:
        """
        Transform constraints to target ontology

        Ψ₂: Ontology Mapping

        Args:
            target_ontology: Target ontology to map to

        Returns:
            Transformed constraints
        """
        transformed = []

        for constraint in self.sce.get_all_constraints():
            # Transform formalization to target ontology
            new_formalization = self._map_ontology(
                constraint.formalization,
                target_ontology
            )

            new_constraint = Constraint(
                id=f"{constraint.id}_mapped",
                type=constraint.type,
                description=f"Mapped: {constraint.description}",
                formalization=new_formalization,
                source="ontology_mapping",
                dependencies=[constraint.id]
            )

            transformed.append(new_constraint)
            self.sce.add_constraint(new_constraint)

        return transformed

    def invert_constraints(self, constraint_ids: List[str]) -> List[Constraint]:
        """
        Invert constraints for complexity reduction

        Ψ₃: Constraint Inversion

        Args:
            constraint_ids: Constraints to invert

        Returns:
            Inverted constraints
        """
        inverted = []

        for constraint_id in constraint_ids:
            constraint = self.sce.get_constraint(constraint_id)
            if not constraint:
                continue

            # Invert the formalization
            inverted_formalization = self._invert_formalization(
                constraint.formalization
            )

            inverted_constraint = Constraint(
                id=f"{constraint_id}_inverted",
                type=constraint.type,
                description=f"Inverted: {constraint.description}",
                formalization=inverted_formalization,
                source="constraint_inversion",
                dependencies=[constraint_id]
            )

            inverted.append(inverted_constraint)
            self.sce.add_constraint(inverted_constraint)

        return inverted

    def _map_ontology(self, formalization: str, target: str) -> str:
        """Map formalization to target ontology"""
        # Ontology mapping logic here
        return f"{target}_{formalization}"

    def _invert_formalization(self, formalization: str) -> str:
        """Invert constraint formalization"""
        # Inversion logic here
        return f"not ({formalization})"
```

**Key Points**:
- Always set dependencies on original constraints
- Use `source` to track transformation origin
- Keep both original and transformed constraints
- Use topological sort to process in correct order

---

### Stage 7: Monte Carlo Refinement (Γ₁, Γ₂, Γ₃)

**Purpose**: Optimize constraint satisfaction, search for optimal solutions

**Integration Pattern**: Constraint Optimization

```python
class MonteCarloRefiner:
    def __init__(self, sce: SymbolicConstraintEngine):
        self.sce = sce

    def optimize_satisfaction(self, iterations: int = 1000) -> Dict[str, any]:
        """
        Optimize constraint satisfaction using MCTS

        Γ₂: Monte Carlo Tree Search

        Args:
            iterations: Number of MCTS iterations

        Returns:
            Optimization results
        """
        results = {
            "best_solution": None,
            "satisfaction_rate": 0.0,
            "violations": []
        }

        # Get all hard constraints (must satisfy)
        hard_constraints = self.sce.get_constraints_by_type(
            ConstraintType.HARD
        )

        # Run MCTS
        for i in range(iterations):
            # Sample constraint satisfaction
            solution = self._sample_solution()

            # Check violations
            violations = self._check_violations(solution, hard_constraints)

            # Track best
            if len(violations) < len(results["violations"]):
                results["best_solution"] = solution
                results["violations"] = violations

        # Calculate satisfaction rate
        total_hard = len(hard_constraints)
        satisfied = total_hard - len(results["violations"])
        results["satisfaction_rate"] = satisfied / total_hard if total_hard > 0 else 0

        return results

    def detect_anomalies(self) -> List[Dict[str, any]]:
        """
        Detect anomalies in constraint satisfaction

        Γ₁: ACI (Automated Constraint Inference) Analyzer

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Analyze constraint statistics
        stats = self.sce.get_statistics()

        # Check for unusual patterns
        if stats["conflicts"] > 10:
            anomalies.append({
                "type": "high_conflict_count",
                "severity": "high",
                "description": f"Too many conflicts: {stats['conflicts']}"
            })

        # Check for isolated constraints (no dependencies)
        for constraint in self.sce.get_all_constraints():
            deps = self.sce.get_dependencies(constraint.id)
            dependents = self.sce.get_dependents(constraint.id)

            if len(deps) == 0 and len(dependents) == 0:
                anomalies.append({
                    "type": "isolated_constraint",
                    "severity": "low",
                    "constraint_id": constraint.id,
                    "description": "Constraint has no dependencies or dependents"
                })

        return anomalies

    def _sample_solution(self) -> Dict[str, any]:
        """Sample a potential solution"""
        # MCTS sampling logic here
        return {}

    def _check_violations(self, solution: Dict, constraints: List) -> List:
        """Check constraint violations"""
        # Violation checking logic here
        return []
```

**Key Points**:
- Prioritize HARD constraints over SOFT/PREFERENCE
- Use `get_constraints_by_type()` to filter by priority
- Track satisfaction rates for optimization
- Detect anomalies in constraint patterns

---

### Stage 8: Architectural Synthesis (Δ₁, Δ₂, Δ₃)

**Purpose**: Assemble constraint-based solutions, validate ACI reduction

**Integration Pattern**: Constraint Assembly

```python
class ArchitectureSynthesizer:
    def __init__(self, sce: SymbolicConstraintEngine):
        self.sce = sce

    def assemble_architecture(self) -> Dict[str, any]:
        """
        Assemble architecture from constraints

        Δ₁: Architecture Assembly

        Returns:
            Assembled architecture
        """
        architecture = {
            "components": [],
            "constraints": [],
            "satisfaction_status": {}
        }

        # Get constraints in topological order
        try:
            constraint_order = self.sce.topological_sort()
        except ValueError:
            # Circular dependencies - need to resolve
            constraint_order = self._resolve_circular_dependencies()

        # Assemble components based on constraints
        for constraint_id in constraint_order:
            constraint = self.sce.get_constraint(constraint_id)

            component = self._constraint_to_component(constraint)
            architecture["components"].append(component)
            architecture["constraints"].append(constraint)

        return architecture

    def validate_aci_reduction(self, before_sce: SymbolicConstraintEngine,
                               after_sce: SymbolicConstraintEngine) -> Dict[str, any]:
        """
        Validate ACI (Automated Constraint Inference) reduction

        Δ₃: ACI Reduction Validator

        Args:
            before_sce: SCE before transformation
            after_sce: SCE after transformation

        Returns:
            Validation results
        """
        results = {
            "aci_reduced": False,
            "reduction_amount": 0,
            "preserved_correctness": False
        }

        # Calculate ACI before
        before_stats = before_sce.get_statistics()
        before_aci = self._calculate_aci(before_sce)

        # Calculate ACI after
        after_stats = after_sce.get_statistics()
        after_aci = self._calculate_aci(after_sce)

        # Check reduction
        reduction = before_aci - after_aci
        results["reduction_amount"] = reduction

        if reduction > 0:
            results["aci_reduced"] = True

        # Validate correctness preservation
        if after_stats["conflicts"] == 0:
            results["preserved_correctness"] = True

        return results

    def _constraint_to_component(self, constraint: Constraint) -> Dict:
        """Convert constraint to architecture component"""
        return {
            "id": constraint.id,
            "type": constraint.type.value,
            "description": constraint.description,
            "formalization": constraint.formalization
        }

    def _calculate_aci(self, sce: SymbolicConstraintEngine) -> float:
        """Calculate ACI (Automated Constraint Inference) metric"""
        stats = sce.get_statistics()
        # ACI calculation logic here
        return float(stats["total_constraints"])

    def _resolve_circular_dependencies(self) -> List[str]:
        """Resolve circular dependencies"""
        # Cycle resolution logic here
        return []
```

**Key Points**:
- Use `topological_sort()` for processing order
- Compare SCE snapshots for validation
- Track ACI reduction metrics
- Validate correctness preservation

---

## Dependency Management

### Adding Dependencies

```python
# Create dependent constraint
dependent = Constraint(
    id="derived",
    type=ConstraintType.HARD,
    description="Derived constraint",
    formalization="derived_property",
    source="inferred",
    dependencies=["base", "auxiliary"]  # Must exist!
)

# Add after dependencies exist
sce.add_constraint(dependent)
```

### Checking Dependencies

```python
# Get what a constraint depends on
deps = sce.get_dependencies("derived")
for dep in deps:
    print(f"Depends on: {dep.id}")

# Get what depends on a constraint
dependents = sce.get_dependents("base")
for dependent in dependents:
    print(f"{dependent.id} depends on this")
```

### Validating Dependencies

```python
# Check for circular dependencies
if not sce.validate_dependencies():
    print("Circular dependency detected!")

# Get topological order
try:
    order = sce.topological_sort()
    print(f"Processing order: {order}")
except ValueError:
    print("Cannot sort - circular dependencies!")
```

---

## Common Integration Patterns

### Pattern 1: Constraint Pipeline

```python
def process_constraints(user_prompt: str):
    """Full constraint processing pipeline"""

    # 1. Create SCE
    sce = SymbolicConstraintEngine()

    # 2. Extract constraints (Stage 1)
    processor = InputProcessor(sce)
    constraints = processor.extract_constraints(user_prompt)

    # 3. Audit constraints (Stage 5)
    auditor = EpistemicAuditor(sce)
    audit_results = auditor.audit_constraints()

    if audit_results["validation_status"] != "valid":
        print("Constraints invalid!")
        return None

    # 4. Transform constraints (Stage 6)
    transformer = IsomorphicResonanceProcessor(sce)
    transformed = transformer.transform_constraints("target_ontology")

    # 5. Optimize satisfaction (Stage 7)
    refiner = MonteCarloRefiner(sce)
    results = refiner.optimize_satisfaction()

    # 6. Assemble architecture (Stage 8)
    synthesizer = ArchitectureSynthesizer(sce)
    architecture = synthesizer.assemble_architecture()

    return architecture
```

---

### Pattern 2: Incremental Constraint Addition

```python
# Add base constraints first
base = Constraint(id="base", ...)
sce.add_constraint(base)

# Then add derived constraints
derived = Constraint(id="derived", dependencies=["base"], ...)
sce.add_constraint(derived)

# Validate after each addition
if not sce.validate_dependencies():
    print("Invalid dependency structure!")
```

---

### Pattern 3: Conflict Resolution

```python
# Detect conflicts
conflicts = sce.detect_conflicts()

for id1, id2, reason in conflicts:
    print(f"Conflict: {id1} <-> {id2}: {reason}")

    # Resolve conflict (example: remove higher ID)
    c1 = sce.get_constraint(id1)
    c2 = sce.get_constraint(id2)

    # Resolution logic here
    # (e.g., remove lower priority constraint)
```

---

### Pattern 4: Constraint Serialization

```python
# Export to dict for serialization
def serialize_constraint(constraint: Constraint) -> Dict:
    return {
        "id": constraint.id,
        "type": constraint.type.value,
        "description": constraint.description,
        "formalization": constraint.formalization,
        "source": constraint.source,
        "dependencies": constraint.dependencies,
        "verified": constraint.verified,
        "lean_theorem": constraint.lean_theorem
    }

# Serialize all constraints
all_constraints = sce.get_all_constraints()
serialized = [serialize_constraint(c) for c in all_constraints]

# Save to file
import json
with open("constraints.json", "w") as f:
    json.dump(serialized, f, indent=2)
```

---

## API Usage Examples

### Example 1: Create SCE and Add Constraints

```python
from rese.core.symbolic_constraint_engine import (
    SymbolicConstraintEngine,
    Constraint,
    ConstraintType
)

# Create engine
sce = SymbolicConstraintEngine()

# Add constraints
c1 = Constraint(
    id="temp_max",
    type=ConstraintType.HARD,
    description="Temperature must be < 1000°C",
    formalization="forall (T : Temperature), T < 1000",
    source="user_prompt"
)

c2 = Constraint(
    id="temp_min",
    type=ConstraintType.HARD,
    description="Temperature must be > 500°C",
    formalization="forall (T : Temperature), T > 500",
    source="user_prompt",
    dependencies=["temp_max"]
)

sce.add_constraint(c1)
sce.add_constraint(c2)

# Get statistics
stats = sce.get_statistics()
print(f"Total constraints: {stats['total_constraints']}")
```

---

### Example 2: Query Constraints by Type

```python
# Get all hard constraints
hard_constraints = sce.get_constraints_by_type(ConstraintType.HARD)
print(f"Hard constraints: {len(hard_constraints)}")

# Get all soft constraints
soft_constraints = sce.get_constraints_by_type(ConstraintType.SOFT)
print(f"Soft constraints: {len(soft_constraints)}")

# Get all preferences
preferences = sce.get_constraints_by_type(ConstraintType.PREFERENCE)
print(f"Preferences: {len(preferences)}")
```

---

### Example 3: Dependency Analysis

```python
# Get dependencies
deps = sce.get_dependencies("temp_min")
print(f"temp_min depends on: {[d.id for d in deps]}")

# Get dependents
dependents = sce.get_dependents("temp_max")
print(f"Constraints depending on temp_max: {[d.id for d in dependents]}")

# Validate dependencies
if sce.validate_dependencies():
    print("Dependency graph is valid (acyclic)")
else:
    print("Circular dependencies detected!")

# Get topological order
order = sce.topological_sort()
print(f"Processing order: {order}")
```

---

### Example 4: Conflict Detection

```python
# Detect conflicts
conflicts = sce.detect_conflicts()

if conflicts:
    print(f"Found {len(conflicts)} conflicts:")
    for id1, id2, reason in conflicts:
        c1 = sce.get_constraint(id1)
        c2 = sce.get_constraint(id2)
        print(f"  {id1} <-> {id2}:")
        print(f"    {c1.description}")
        print(f"    {c2.description}")
        print(f"    Reason: {reason}")
else:
    print("No conflicts detected")
```

---

### Example 5: Export and Import

```python
from pathlib import Path

# Export dependency graph to DOT
dot_data = sce.export_to_dot(Path("constraints.dot"))

# Save constraints to JSON
import json

all_constraints = sce.get_all_constraints()
data = [
    {
        "id": c.id,
        "type": c.type.value,
        "description": c.description,
        "formalization": c.formalization,
        "source": c.source,
        "dependencies": c.dependencies,
        "verified": c.verified,
        "lean_theorem": c.lean_theorem
    }
    for c in all_constraints
]

with open("constraints.json", "w") as f:
    json.dump(data, f, indent=2)

# Import constraints back
new_sce = SymbolicConstraintEngine()
for item in data:
    constraint = Constraint(
        id=item["id"],
        type=ConstraintType(item["type"]),
        description=item["description"],
        formalization=item["formalization"],
        source=item["source"],
        dependencies=item["dependencies"],
        verified=item["verified"],
        lean_theorem=item["lean_theorem"]
    )
    new_sce.add_constraint(constraint)
```

---

## Testing and Validation

### Unit Testing SCE Integration

```python
import pytest
from rese.core.symbolic_constraint_engine import (
    SymbolicConstraintEngine,
    Constraint,
    ConstraintType
)

def test_integration_basic():
    """Test basic SCE integration"""
    sce = SymbolicConstraintEngine()

    # Add constraint
    c = Constraint(
        id="test",
        type=ConstraintType.HARD,
        description="Test",
        formalization="test",
        source="test"
    )
    sce.add_constraint(c)

    # Verify
    assert sce.get_constraint("test") is not None
    assert len(sce.get_all_constraints()) == 1

def test_integration_dependencies():
    """Test dependency management"""
    sce = SymbolicConstraintEngine()

    # Add base constraint
    base = Constraint(
        id="base",
        type=ConstraintType.HARD,
        description="Base",
        formalization="base",
        source="test"
    )
    sce.add_constraint(base)

    # Add dependent constraint
    dependent = Constraint(
        id="dependent",
        type=ConstraintType.HARD,
        description="Dependent",
        formalization="dependent",
        source="test",
        dependencies=["base"]
    )
    sce.add_constraint(dependent)

    # Verify dependencies
    deps = sce.get_dependencies("dependent")
    assert len(deps) == 1
    assert deps[0].id == "base"

    # Validate
    assert sce.validate_dependencies() is True
```

### Integration Testing with Stages

```python
def test_stage_1_integration():
    """Test Stage 1 (Input Processing) integration"""
    sce = SymbolicConstraintEngine()
    processor = InputProcessor(sce)

    user_prompt = "Temperature must be less than 1000°C"
    constraints = processor.extract_constraints(user_prompt)

    assert len(constraints) > 0
    assert len(sce.get_all_constraints()) > 0

def test_stage_5_integration():
    """Test Stage 5 (Epistemic Audit) integration"""
    sce = SymbolicConstraintEngine()
    # Add constraints...

    auditor = EpistemicAuditor(sce)
    results = auditor.audit_constraints()

    assert "validation_status" in results
    assert "conflicts" in results

def test_stage_6_integration():
    """Test Stage 6 (Isomorphic Resonance) integration"""
    sce = SymbolicConstraintEngine()
    # Add constraints...

    transformer = IsomorphicResonanceProcessor(sce)
    transformed = transformer.transform_constraints("target_ontology")

    assert len(transformed) > 0
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Circular Dependencies

**Symptom**: `ValueError: Cannot topologically sort graph with cycles`

**Cause**: Constraint A depends on B, B depends on A (directly or indirectly)

**Solution**:
1. Detect the cycle
2. Break the cycle by removing a dependency
3. Re-structure constraints

```python
# Detect cycle
if not sce.validate_dependencies():
    print("Circular dependency detected!")

    # Find and break cycle
    # (implementation-specific)
```

---

#### Issue 2: Constraint Already Exists

**Symptom**: `ValueError: Constraint X already exists`

**Cause**: Adding constraint with duplicate ID

**Solution**:
```python
# Check if constraint exists first
if sce.get_constraint(constraint_id) is None:
    sce.add_constraint(constraint)
else:
    print(f"Constraint {constraint_id} already exists")
    # Update existing constraint instead
```

---

#### Issue 3: Non-existent Dependency

**Symptom**: `ValueError: Constraint X depends on non-existent Y`

**Cause**: Referencing dependency that hasn't been added yet

**Solution**:
```python
# Add dependencies first
for dep_id in constraint.dependencies:
    if sce.get_constraint(dep_id) is None:
        print(f"Dependency {dep_id} not found, skipping constraint")
        continue

# Then add constraint
sce.add_constraint(constraint)
```

---

#### Issue 4: Too Many Conflicts

**Symptom**: `detect_conflicts()` returns many conflicts

**Cause**: Inconsistent constraint requirements

**Solution**:
1. Review conflicting constraints
2. Prioritize constraints (HARD vs SOFT)
3. Remove or modify lower-priority constraints

```python
conflicts = sce.detect_conflicts()

# Sort by priority
prioritized_conflicts = []
for id1, id2, reason in conflicts:
    c1 = sce.get_constraint(id1)
    c2 = sce.get_constraint(id2)

    # Prioritize keeping HARD constraints
    if c1.is_hard() and not c2.is_hard():
        prioritized_conflicts.append((id1, id2, "keep_id1"))
    elif c2.is_hard() and not c1.is_hard():
        prioritized_conflicts.append((id1, id2, "keep_id2"))
    else:
        prioritized_conflicts.append((id1, id2, "manual_review"))
```

---

## Best Practices

### 1. Always Validate Before Adding

```python
def safe_add_constraint(sce: SymbolicConstraintEngine, constraint: Constraint):
    """Safely add constraint with validation"""
    # Check if already exists
    if sce.get_constraint(constraint.id) is not None:
        print(f"Warning: {constraint.id} already exists")
        return False

    # Check dependencies exist
    for dep_id in constraint.dependencies:
        if sce.get_constraint(dep_id) is None:
            print(f"Error: Dependency {dep_id} not found")
            return False

    # Add constraint
    sce.add_constraint(constraint)
    return True
```

---

### 2. Use Topological Sort for Processing

```python
def process_constraints_in_order(sce: SymbolicConstraintEngine):
    """Process constraints in dependency order"""
    try:
        order = sce.topological_sort()
    except ValueError:
        print("Cannot process - circular dependencies")
        return

    for constraint_id in order:
        constraint = sce.get_constraint(constraint_id)
        # Process constraint
        print(f"Processing: {constraint.id}")
```

---

### 3. Track Constraint Source

```python
# Always set source field
constraint = Constraint(
    id="user_constraint",
    type=ConstraintType.HARD,
    description="User requirement",
    formalization="user_requirement_formal",
    source="user_prompt"  # Important for tracking!
)

# Different sources for different origins
sources = [
    "user_prompt",        # From user input
    "system_inferred",    # Inferred by system
    "assumption_mining",  # From Φ₁.₅
    "ontology_mapping",   # From Ψ₂
    "constraint_inversion",  # From Ψ₃
    "manual_review"       # From manual review
]
```

---

### 4. Use Appropriate Constraint Types

```python
# HARD: Must satisfy (blocking)
hard_constraint = Constraint(
    id="safety_limit",
    type=ConstraintType.HARD,
    description="Temperature must not exceed safety limit",
    formalization="T < T_max",
    source="safety_requirement"
)

# SOFT: Prefer to satisfy (optimization)
soft_constraint = Constraint(
    id="performance_target",
    type=ConstraintType.SOFT,
    description="System should respond within 100ms",
    formalization="response_time < 100ms preferred",
    source="performance_requirement"
)

# PREFERENCE: Nice to have (guidance)
preference = Constraint(
    id="ui_style",
    type=ConstraintType.PREFERENCE,
    description="UI should use modern design",
    formalization="ui_style = modern preferred",
    source="design_guideline"
)
```

---

## See Also

- [SCE API Documentation](../api/sce_api.md)
- [Performance Tests](../../tests/test_core/test_sce_performance.py)
- [Unit Tests](../../tests/test_core/test_symbolic_constraint_engine.py)
- [RESE Architecture](../architecture/RESE_ARCHITECTURE.md)
- [Integration Testing Guide](../testing/INTEGRATION_TESTING.md)

---

**Last Updated**: 2025-12-31
**Author**: Agent A1
**Status**: Active Implementation
