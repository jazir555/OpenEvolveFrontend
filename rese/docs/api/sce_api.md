# Symbolic Constraint Engine (SCE) API Documentation

**Module**: `rese.core.symbolic_constraint_engine`
**Author**: Agent A1
**Status**: Active Implementation
**Version**: 1.0.0

---

## Table of Contents

1. [Overview](#overview)
2. [Core Classes](#core-classes)
3. [API Reference](#api-reference)
4. [Usage Examples](#usage-examples)
5. [Lean 4 Integration](#lean-4-integration)
6. [Error Handling](#error-handling)
7. [Performance Considerations](#performance-considerations)

---

## Overview

The Symbolic Constraint Engine (SCE) is the foundational module for the RESE system. It provides formal constraint management with dependency tracking, conflict detection, and Lean 4 verification capabilities.

### Key Features

- **Formal Constraint Representation**: Constraints stored with Lean 4 formalizations
- **Dependency Management**: Directed acyclic graph (DAG) for constraint dependencies
- **Conflict Detection**: Basic contradiction detection (will be enhanced by DITO)
- **Type System**: Hard, Soft, and Preference constraint types
- **Verification Tracking**: Track Lean 4 theorem verification status

### Dependencies

```python
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum
from pathlib import Path
```

---

## Core Classes

### 1. ConstraintType Enum

```python
class ConstraintType(Enum):
    """Types of constraints in the RESE system"""
    HARD = "hard"           # Must satisfy (blocking)
    SOFT = "soft"           # Prefer to satisfy (optimization)
    PREFERENCE = "preference"  # Nice to have (guidance)
```

**Values**:
- `HARD`: Must be satisfied. Violations block operation.
- `SOFT`: Should be satisfied. Used for optimization.
- `PREFERENCE`: Optional guidance. Not enforced.

---

### 2. Constraint Dataclass

```python
@dataclass
class Constraint:
    """
    A formal constraint in the RESE system.

    Attributes:
        id: Unique identifier for this constraint
        type: Constraint type (HARD, SOFT, PREFERENCE)
        description: Human-readable description
        formalization: Lean 4 representation
        source: Where this constraint came from
        dependencies: List of constraint IDs this constraint depends on
        verified: Whether this constraint has been verified in Lean 4
        lean_theorem: Optional Lean 4 theorem proving this constraint
    """
```

#### Attributes

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | `str` | Yes | Unique constraint identifier |
| `type` | `ConstraintType` | Yes | Type of constraint |
| `description` | `str` | Yes | Human-readable description |
| `formalization` | `str` | Yes | Lean 4 formal representation |
| `source` | `str` | Yes | Origin (user_prompt, system, inferred, etc.) |
| `dependencies` | `List[str]` | No | List of constraint IDs this depends on |
| `verified` | `bool` | No | Lean 4 verification status |
| `lean_theorem` | `Optional[str]` | No | Lean 4 theorem if verified |

#### Methods

**`is_hard() -> bool`**
- Returns: `True` if constraint is HARD type
- Example:
  ```python
  if constraint.is_hard():
      print("This constraint must be satisfied")
  ```

**`is_verified() -> bool`**
- Returns: `True` if constraint has Lean 4 theorem
- Example:
  ```python
  if constraint.is_verified():
      print(f"Verified with theorem: {constraint.lean_theorem}")
  ```

**`__hash__() -> int`**
- Makes constraint hashable for use in sets
- Example:
  ```python
  constraint_set = {constraint1, constraint2}
  ```

**`__eq__(other) -> bool`**
- Constraint equality based on ID
- Example:
  ```python
  if constraint1 == constraint2:
      print("Same constraint ID")
  ```

---

### 3. SymbolicConstraintEngine Class

```python
class SymbolicConstraintEngine:
    """
    Manages constraints and their dependencies.

    The SCE is the foundation for all RESE phases. It provides:
    - Constraint storage and retrieval
    - Dependency tracking via directed graph
    - Contradiction detection
    - Constraint satisfaction checking
    """
```

#### Constructor

```python
def __init__(self):
    """Initialize an empty constraint engine"""
```

**Example**:
```python
sce = SymbolicConstraintEngine()
```

---

## API Reference

### Constraint Management

#### `add_constraint(constraint: Constraint) -> None`

Add a constraint to the engine.

**Parameters**:
- `constraint` (Constraint): Constraint to add

**Raises**:
- `ValueError`: If constraint ID already exists
- `ValueError`: If dependency refers to non-existent constraint

**Example**:
```python
c = Constraint(
    id="temp_limit",
    type=ConstraintType.HARD,
    description="Temperature must be less than 1000°C",
    formalization="forall (T : Temperature), T < 1000",
    source="user_prompt"
)
sce.add_constraint(c)
```

---

#### `get_constraint(constraint_id: str) -> Optional[Constraint]`

Retrieve a constraint by ID.

**Parameters**:
- `constraint_id` (str): ID of constraint to retrieve

**Returns**:
- `Constraint` if found, `None` otherwise

**Example**:
```python
constraint = sce.get_constraint("temp_limit")
if constraint:
    print(f"Found: {constraint.description}")
```

---

#### `get_all_constraints() -> List[Constraint]`

Get all constraints in the system.

**Returns**:
- `List[Constraint]`: All constraints

**Example**:
```python
all_constraints = sce.get_all_constraints()
for c in all_constraints:
    print(f"{c.id}: {c.description}")
```

---

#### `get_constraints_by_type(constraint_type: ConstraintType) -> List[Constraint]`

Get all constraints of a specific type.

**Parameters**:
- `constraint_type` (ConstraintType): Type to filter by

**Returns**:
- `List[Constraint]`: Constraints of specified type

**Example**:
```python
hard_constraints = sce.get_constraints_by_type(ConstraintType.HARD)
print(f"Found {len(hard_constraints)} hard constraints")
```

---

### Dependency Management

#### `get_dependencies(constraint_id: str) -> List[Constraint]`

Get all dependencies for a constraint.

**Parameters**:
- `constraint_id` (str): ID of constraint

**Returns**:
- `List[Constraint]`: Constraints that this constraint depends on

**Example**:
```python
deps = sce.get_dependencies("derived_constraint")
for dep in deps:
    print(f"Depends on: {dep.id}")
```

---

#### `get_dependents(constraint_id: str) -> List[Constraint]`

Get all constraints that depend on this constraint.

**Parameters**:
- `constraint_id` (str): ID of constraint

**Returns**:
- `List[Constraint]`: Constraints that depend on this one

**Example**:
```python
dependents = sce.get_dependents("base_constraint")
for dep in dependents:
    print(f"{dep.id} depends on this")
```

---

#### `validate_dependencies() -> bool`

Validate that all dependencies are satisfied (acyclic graph).

**Returns**:
- `bool`: True if dependency graph is acyclic, False otherwise

**Example**:
```python
if sce.validate_dependencies():
    print("Dependency graph is valid")
else:
    print("Circular dependency detected!")
```

---

#### `topological_sort() -> List[str]`

Get constraints in topological order (dependencies before dependents).

**Returns**:
- `List[str]`: Constraint IDs in topological order

**Raises**:
- `ValueError`: If graph has cycles

**Example**:
```python
try:
    sorted_ids = sce.topological_sort()
    for constraint_id in sorted_ids:
        print(f"Process: {constraint_id}")
except ValueError as e:
    print(f"Cannot sort: {e}")
```

---

### Conflict Detection

#### `detect_conflicts() -> List[Tuple[str, str, str]]`

Detect conflicting constraints.

**Returns**:
- `List[Tuple[str, str, str]]`: List of tuples (id1, id2, reason)

**Note**: This is a basic implementation. The full version will use DITO.

**Example**:
```python
conflicts = sce.detect_conflicts()
if conflicts:
    for id1, id2, reason in conflicts:
        print(f"Conflict: {id1} <-> {id2}: {reason}")
else:
    print("No conflicts detected")
```

---

### Statistics and Analysis

#### `get_statistics() -> Dict[str, int]`

Get statistics about the constraint system.

**Returns**:
- `Dict[str, int]`: Dictionary with statistics:
  - `total_constraints`: Total number of constraints
  - `hard_constraints`: Number of hard constraints
  - `soft_constraints`: Number of soft constraints
  - `preference_constraints`: Number of preference constraints
  - `verified_constraints`: Number of verified constraints
  - `conflicts`: Number of detected conflicts
  - `dependencies`: Number of dependency edges

**Example**:
```python
stats = sce.get_statistics()
print(f"Total: {stats['total_constraints']}")
print(f"Hard: {stats['hard_constraints']}")
print(f"Conflicts: {stats['conflicts']}")
```

---

#### `export_to_dot(filepath: Optional[Path] = None) -> str`

Export dependency graph to DOT format.

**Parameters**:
- `filepath` (Optional[Path]): Optional filepath to save DOT file

**Returns**:
- `str`: DOT format string

**Example**:
```python
# Get DOT string
dot_data = sce.export_to_dot()

# Save to file
sce.export_to_dot(Path("constraints.dot"))
```

---

### Convenience Functions

#### `create_constraint_from_dict(data: Dict) -> Constraint`

Create a Constraint from a dictionary.

**Parameters**:
- `data` (Dict): Dictionary with constraint fields

**Returns**:
- `Constraint`: Constraint instance

**Example**:
```python
data = {
    "id": "test",
    "type": "hard",
    "description": "Test constraint",
    "formalization": "test",
    "source": "test"
}
constraint = create_constraint_from_dict(data)
```

---

## Usage Examples

### Example 1: Basic Constraint Management

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
    id="max_temp",
    type=ConstraintType.HARD,
    description="Temperature must be less than 1000°C",
    formalization="forall (T : Temperature), T < 1000",
    source="user_prompt"
)

c2 = Constraint(
    id="min_temp",
    type=ConstraintType.HARD,
    description="Temperature must be greater than 500°C",
    formalization="forall (T : Temperature), T > 500",
    source="user_prompt",
    dependencies=["max_temp"]
)

sce.add_constraint(c1)
sce.add_constraint(c2)

# Display statistics
stats = sce.get_statistics()
print(f"Total constraints: {stats['total_constraints']}")
```

---

### Example 2: Dependency Chain

```python
# Create chain: base -> derived1 -> derived2
base = Constraint(
    id="base",
    type=ConstraintType.HARD,
    description="Base constraint",
    formalization="base_property",
    source="system"
)

derived1 = Constraint(
    id="derived1",
    type=ConstraintType.HARD,
    description="Derived from base",
    formalization="derived1_property",
    source="inferred",
    dependencies=["base"]
)

derived2 = Constraint(
    id="derived2",
    type=ConstraintType.HARD,
    description="Derived from derived1",
    formalization="derived2_property",
    source="inferred",
    dependencies=["derived1"]
)

sce.add_constraint(base)
sce.add_constraint(derived1)
sce.add_constraint(derived2)

# Get topological order
sorted_ids = sce.topological_sort()
print(f"Processing order: {sorted_ids}")
# Output: ['base', 'derived1', 'derived2']
```

---

### Example 3: Conflict Detection

```python
# Add conflicting constraints
c1 = Constraint(
    id="must_enable",
    type=ConstraintType.HARD,
    description="Feature must be enabled",
    formalization="feature_enabled = true",
    source="user_prompt"
)

c2 = Constraint(
    id="must_disable",
    type=ConstraintType.HARD,
    description="Feature must be disabled",
    formalization="feature_enabled = false",
    source="user_prompt"
)

sce.add_constraint(c1)
sce.add_constraint(c2)

# Detect conflicts
conflicts = sce.detect_conflicts()
for id1, id2, reason in conflicts:
    print(f"Conflict between {id1} and {id2}: {reason}")
```

---

### Example 4: Verified Constraints

```python
# Create verified constraint
verified_c = Constraint(
    id="verified_temp_limit",
    type=ConstraintType.HARD,
    description="Temperature limit (verified)",
    formalization="forall (T : Temperature), T < 1000",
    source="user_prompt",
    verified=True,
    lean_theorem="""
theorem temperature_limit_valid :
  forall (T : Temperature), T.value < 1000 :=
by
  -- Lean 4 proof here
  sorry
"""
)

sce.add_constraint(verified_c)

# Check verification status
if verified_c.is_verified():
    print(f"Verified with theorem: {verified_c.lean_theorem}")
```

---

## Lean 4 Integration

### Formal Verification Workflow

1. **Create Constraint with Formalization**:
```python
constraint = Constraint(
    id="temp_limit",
    type=ConstraintType.HARD,
    description="Temperature must be less than 1000°C",
    formalization="forall (T : Temperature), T < 1000",
    source="user_prompt"
)
```

2. **Generate Lean 4 Theorem** (manual or automated):
```lean
theorem temperature_limit :
  forall (T : Temperature), T.value < 1000 := by
  -- Proof goes here
  sorry
```

3. **Mark as Verified**:
```python
constraint.verified = True
constraint.lean_theorem = """
theorem temperature_limit :
  forall (T : Temperature), T.value < 1000 := by
  sorry
"""
```

4. **Use in System**:
```python
if constraint.is_verified():
    print("Constraint formally verified in Lean 4")
```

### Lean 4 Integration Notes

**Current State**:
- SCE stores Lean 4 formalizations and theorems
- Verification tracking only (no automated proving)

**Future Enhancements** (Agent O1):
- Automated Lean 4 code generation
- Interactive theorem proving integration
- Proof assistant API calls
- Verification pipeline automation

**Formalization Template**:
```python
formalization_template = """
theorem {constraint_id} :
  {formal_statement} := by
  {proof}
"""
```

---

## Error Handling

### Common Exceptions

#### 1. ValueError: Empty ID
```python
try:
    c = Constraint(
        id="",  # Empty!
        type=ConstraintType.HARD,
        description="Test",
        formalization="test",
        source="test"
    )
except ValueError as e:
    print(f"Error: {e}")
# Output: Error: Constraint must have a non-empty ID
```

---

#### 2. ValueError: Duplicate Constraint
```python
try:
    sce.add_constraint(c1)
    sce.add_constraint(c1)  # Duplicate!
except ValueError as e:
    print(f"Error: {e}")
# Output: Error: Constraint c1 already exists
```

---

#### 3. ValueError: Non-existent Dependency
```python
try:
    c = Constraint(
        id="orphan",
        type=ConstraintType.HARD,
        description="Orphan",
        formalization="test",
        source="test",
        dependencies=["nonexistent"]  # Doesn't exist!
    )
    sce.add_constraint(c)
except ValueError as e:
    print(f"Error: {e}")
# Output: Error: Constraint orphan depends on non-existent nonexistent
```

---

#### 4. ValueError: Cyclic Dependencies
```python
try:
    # After creating cycle manually
    sorted_ids = sce.topological_sort()
except ValueError as e:
    print(f"Error: {e}")
# Output: Error: Cannot topologically sort graph with cycles
```

---

## Performance Considerations

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `add_constraint` | O(1) | Graph insertion |
| `get_constraint` | O(1) | Dictionary lookup |
| `get_all_constraints` | O(n) | Returns all n constraints |
| `get_constraints_by_type` | O(n) | Filters all n constraints |
| `get_dependencies` | O(d) | d = number of dependencies |
| `get_dependents` | O(d) | d = number of dependents |
| `detect_conflicts` | O(n²) | Checks all pairs |
| `validate_dependencies` | O(V + E) | V = vertices, E = edges |
| `topological_sort` | O(V + E) | V = vertices, E = edges |

### Space Complexity

- **Constraints**: O(n) where n = number of constraints
- **Dependency Graph**: O(V + E) where V = constraints, E = dependencies
- **Contradiction Cache**: O(c²) where c = number of constraint pairs checked

### Optimization Tips

1. **Batch Additions**: Add multiple constraints before conflict detection
2. **Selective Queries**: Use `get_constraints_by_type()` instead of filtering manually
3. **Cache Results**: Store `get_all_constraints()` results if used multiple times
4. **Dependency Pruning**: Remove unused dependencies to reduce graph size

### Scalability

Current implementation tested with:
- **1000+ constraints**: Supported
- **100+ dependency levels**: Supported
- **Performance**: See `test_sce_performance.py` for benchmarks

---

## Integration Points

### Used By:

1. **Stage 1 (Input Processing)**: Extract constraints from user prompts
2. **Stage 5 (Epistemic Audit)**: Validate constraint consistency
3. **Stage 6 (Isomorphic Resonance)**: Transform constraint structures
4. **Stage 7 (Monte Carlo Refinement)**: Optimize constraint satisfaction

### Dependencies:

- **NetworkX**: Graph algorithms
- **Lean 4** (future): Formal verification

---

## See Also

- [SCE Integration Guide](../developer_guides/sce_integration.md)
- [Performance Tests](../../tests/test_core/test_sce_performance.py)
- [Unit Tests](../../tests/test_core/test_symbolic_constraint_engine.py)
- [RESE Architecture](../architecture/RESE_ARCHITECTURE.md)

---

**Last Updated**: 2025-12-31
**Author**: Agent A1
**Status**: Active Implementation
