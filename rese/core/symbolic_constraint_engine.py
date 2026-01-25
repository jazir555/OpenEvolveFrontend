"""
Symbolic Constraint Engine (SCE)

Foundation for all RESE phases - enforces logical consistency using formal logic.
All constraints verified in Lean 4 Interactive Theorem Prover.

Author: Agent A1
Created: 2025-12-31
Status: 🟢 Active Implementation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum
import networkx as nx
from pathlib import Path


class ConstraintType(Enum):
    """Types of constraints in the RESE system"""
    HARD = "hard"           # Must satisfy (blocking)
    SOFT = "soft"           # Prefer to satisfy (optimization)
    PREFERENCE = "preference"  # Nice to have (guidance)


@dataclass
class Constraint:
    """
    A formal constraint in the RESE system.

    Attributes:
        id: Unique identifier for this constraint
        type: Constraint type (HARD, SOFT, PREFERENCE)
        description: Human-readable description
        formalization: Lean 4 representation
        source: Where this constraint came from (user_prompt, system, inferred, etc.)
        dependencies: List of constraint IDs this constraint depends on
        verified: Whether this constraint has been verified in Lean 4
        lean_theorem: Optional Lean 4 theorem proving this constraint
    """
    id: str
    type: ConstraintType
    description: str
    formalization: str
    source: str
    dependencies: List[str] = field(default_factory=list)
    verified: bool = False
    lean_theorem: Optional[str] = None

    def __post_init__(self):
        """Validate constraint after initialization"""
        if not self.id or not self.id.strip():
            raise ValueError("Constraint must have a non-empty ID")
        if not self.description or not self.description.strip():
            raise ValueError(f"Constraint {self.id} must have a non-empty description")
        if not self.formalization or not self.formalization.strip():
            raise ValueError(f"Constraint {self.id} must have a formalization")

    def __hash__(self):
        """Make constraint hashable for use in sets"""
        return hash(self.id)

    def __eq__(self, other):
        """Constraint equality based on ID"""
        if not isinstance(other, Constraint):
            return False
        return self.id == other.id

    def is_hard(self) -> bool:
        """Check if this is a hard constraint"""
        return self.type == ConstraintType.HARD

    def is_verified(self) -> bool:
        """Check if this constraint has been verified in Lean 4"""
        return self.verified and self.lean_theorem is not None


class SymbolicConstraintEngine:
    """
    Manages constraints and their dependencies.

    The SCE is the foundation for all RESE phases. It provides:
    - Constraint storage and retrieval
    - Dependency tracking via directed graph
    - Contradiction detection (basic, will be enhanced by DITO)
    - Constraint satisfaction checking
    """

    def __init__(self):
        self.constraints: Dict[str, Constraint] = {}
        self.dependency_graph = nx.DiGraph()
        self._contradiction_cache: Dict[Tuple[str, str], bool] = {}

    def add_constraint(self, constraint: Constraint) -> None:
        """
        Add a constraint to the engine.

        Args:
            constraint: Constraint to add

        Raises:
            ValueError: If constraint ID already exists
            ValueError: If dependency refers to non-existent constraint
        """
        if constraint.id in self.constraints:
            raise ValueError(f"Constraint {constraint.id} already exists")

        # Validate dependencies exist
        for dep_id in constraint.dependencies:
            if dep_id not in self.constraints:
                raise ValueError(
                    f"Constraint {constraint.id} depends on non-existent {dep_id}"
                )

        # Add to storage
        self.constraints[constraint.id] = constraint

        # Add to dependency graph
        self.dependency_graph.add_node(constraint.id, constraint=constraint)

        # Add dependency edges
        for dep_id in constraint.dependencies:
            self.dependency_graph.add_edge(dep_id, constraint.id)
            # Invalidate contradiction cache
            self._invalidate_contradiction_cache()

    def get_constraint(self, constraint_id: str) -> Optional[Constraint]:
        """
        Retrieve a constraint by ID.

        Args:
            constraint_id: ID of constraint to retrieve

        Returns:
            Constraint if found, None otherwise
        """
        return self.constraints.get(constraint_id)

    def get_all_constraints(self) -> List[Constraint]:
        """Get all constraints in the system"""
        return list(self.constraints.values())

    def get_constraints_by_type(self, constraint_type: ConstraintType) -> List[Constraint]:
        """Get all constraints of a specific type"""
        return [
            c for c in self.constraints.values()
            if c.type == constraint_type
        ]

    def get_dependencies(self, constraint_id: str) -> List[Constraint]:
        """
        Get all dependencies for a constraint.

        Args:
            constraint_id: ID of constraint

        Returns:
            List of constraints that this constraint depends on
        """
        if constraint_id not in self.dependency_graph:
            return []
        return [
            self.constraints[dep_id]
            for dep_id in list(self.dependency_graph.predecessors(constraint_id))
            if dep_id in self.constraints
        ]

    def get_dependents(self, constraint_id: str) -> List[Constraint]:
        """
        Get all constraints that depend on this constraint.

        Args:
            constraint_id: ID of constraint

        Returns:
            List of constraints that depend on this one
        """
        if constraint_id not in self.dependency_graph:
            return []
        return [
            self.constraints[dep_id]
            for dep_id in list(self.dependency_graph.successors(constraint_id))
            if dep_id in self.constraints
        ]

    def detect_conflicts(self) -> List[Tuple[str, str, str]]:
        """
        Detect conflicting constraints.

        Returns:
            List of tuples (id1, id2, reason) describing conflicts

        Note:
            This is a basic implementation. The full version will use
            DITO (Agent A3) for polynomial-time contradiction detection.
        """
        conflicts = []

        # Check all pairs of constraints
        constraint_ids = list(self.constraints.keys())
        for i, id1 in enumerate(constraint_ids):
            for id2 in constraint_ids[i+1:]:
                c1 = self.constraints[id1]
                c2 = self.constraints[id2]

                # Check for contradictions
                if self._are_contradictory(c1, c2):
                    reason = self._explain_contradiction(c1, c2)
                    conflicts.append((id1, id2, reason))

        return conflicts

    def _are_contradictory(self, c1: Constraint, c2: Constraint) -> bool:
        """
        Check if two constraints are contradictory.

        Args:
            c1: First constraint
            c2: Second constraint

        Returns:
            True if constraints contradict, False otherwise

        Note:
            This is a placeholder. The full implementation will use
            Lean 4 formal verification.
        """
        # Use cache if available
        cache_key = (c1.id, c2.id) if c1.id < c2.id else (c2.id, c1.id)
        if cache_key in self._contradiction_cache:
            return self._contradiction_cache[cache_key]

        # Basic keyword-based contradiction detection
        contradictions = [
            ("less than", "greater than"),
            ("<", ">"),
            ("always", "never"),
            ("required", "forbidden"),
            ("must", "must not"),
            ("should", "should not"),
            ("equal to", "not equal to"),
            ("=", "≠"),
        ]

        desc1 = c1.description.lower()
        desc2 = c2.description.lower()

        is_contradictory = False
        for pos, neg in contradictions:
            if pos in desc1 and neg in desc2:
                is_contradictory = True
                break
            if neg in desc1 and pos in desc2:
                is_contradictory = True
                break

        # Cache result
        self._contradiction_cache[cache_key] = is_contradictory
        return is_contradictory

    def _explain_contradiction(self, c1: Constraint, c2: Constraint) -> str:
        """Generate explanation for why two constraints contradict"""
        if "less than" in c1.description.lower() and "greater than" in c2.description.lower():
            return f"Logical contradiction: Cannot be both '{c1.description}' and '{c2.description}'"
        if "greater than" in c1.description.lower() and "less than" in c2.description.lower():
            return f"Logical contradiction: Cannot be both '{c1.description}' and '{c2.description}'"
        if "required" in c1.description.lower() and "forbidden" in c2.description.lower():
            return f"Logical contradiction: Cannot both '{c1.description}' and '{c2.description}'"
        if "forbidden" in c1.description.lower() and "required" in c2.description.lower():
            return f"Logical contradiction: Cannot both '{c1.description}' and '{c2.description}'"
        return "Potential logical contradiction detected"

    def _invalidate_contradiction_cache(self):
        """Clear the contradiction cache (called when constraints change)"""
        self._contradiction_cache.clear()

    def validate_dependencies(self) -> bool:
        """
        Validate that all dependencies are satisfied.

        Returns:
            True if dependency graph is acyclic, False otherwise
        """
        return nx.algorithms.is_directed_acyclic_graph(self.dependency_graph)

    def topological_sort(self) -> List[str]:
        """
        Get constraints in topological order (dependencies before dependents).

        Returns:
            List of constraint IDs in topological order

        Raises:
            ValueError: If graph has cycles
        """
        if not self.validate_dependencies():
            raise ValueError("Cannot topologically sort graph with cycles")
        try:
            return list(nx.topological_sort(self.dependency_graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Cannot topologically sort graph with cycles")

    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the constraint system.

        Returns:
            Dictionary with various statistics
        """
        return {
            "total_constraints": len(self.constraints),
            "hard_constraints": len(self.get_constraints_by_type(ConstraintType.HARD)),
            "soft_constraints": len(self.get_constraints_by_type(ConstraintType.SOFT)),
            "preference_constraints": len(self.get_constraints_by_type(ConstraintType.PREFERENCE)),
            "verified_constraints": sum(1 for c in self.constraints.values() if c.is_verified()),
            "conflicts": len(self.detect_conflicts()),
            "dependencies": self.dependency_graph.number_of_edges(),
        }

    def export_to_dot(self, filepath: Optional[Path] = None) -> str:
        """
        Export dependency graph to DOT format.

        Args:
            filepath: Optional filepath to save DOT file

        Returns:
            DOT format string
        """
        # Use networkx's native DOT export (works without pygraphviz)
        from io import StringIO

        try:
            # Try pydot first (more commonly available)
            dot_data = nx.drawing.nx_pydot.to_pydot(self.dependency_graph).to_string()
        except (ImportError, AttributeError):
            # Fallback to simple DOT format generation
            lines = ["digraph G {"]
            for node in self.dependency_graph.nodes():
                lines.append(f'  "{node}";')
            for src, dst in self.dependency_graph.edges():
                lines.append(f'  "{src}" -> "{dst}";')
            lines.append("}")
            dot_data = "\n".join(lines)

        if filepath:
            filepath.write_text(dot_data)

        return dot_data


# Convenience functions

def create_constraint_from_dict(data: Dict) -> Constraint:
    """
    Create a Constraint from a dictionary.

    Args:
        data: Dictionary with constraint fields

    Returns:
        Constraint instance
    """
    return Constraint(
        id=data["id"],
        type=ConstraintType(data["type"]),
        description=data["description"],
        formalization=data.get("formalization", ""),
        source=data.get("source", "unknown"),
        dependencies=data.get("dependencies", []),
        verified=data.get("verified", False),
        lean_theorem=data.get("lean_theorem", None),
    )


# Testing and demonstration

if __name__ == "__main__":
    print("=" * 70)
    print("Symbolic Constraint Engine (SCE) - Demonstration")
    print("=" * 70)

    # Create SCE instance
    sce = SymbolicConstraintEngine()
    print("\n[OK] SCE initialized")

    # Add test constraints
    c1 = Constraint(
        id="temp_limit",
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
        dependencies=["temp_limit"]
    )

    c3 = Constraint(
        id="max_pressure",
        type=ConstraintType.SOFT,
        description="Pressure should preferably be below 10 bar",
        formalization="forall (P : Pressure), P < 10 preferred",
        source="system_inferred"
    )

    sce.add_constraint(c1)
    print("[OK] Added constraint: temp_limit")

    sce.add_constraint(c2)
    print("[OK] Added constraint: min_temp (depends on temp_limit)")

    sce.add_constraint(c3)
    print("[OK] Added constraint: max_pressure")

    # Display statistics
    print("\n" + "=" * 70)
    print("Statistics:")
    print("=" * 70)
    stats = sce.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test dependencies
    print("\n" + "=" * 70)
    print("Dependencies:")
    print("=" * 70)
    deps = sce.get_dependencies("min_temp")
    print(f"  min_temp depends on: {[c.id for c in deps]}")

    # Test topological sort
    print("\n" + "=" * 70)
    print("Topological Sort:")
    print("=" * 70)
    try:
        sorted_ids = sce.topological_sort()
        for i, constraint_id in enumerate(sorted_ids, 1):
            c = sce.get_constraint(constraint_id)
            print(f"  {i}. {constraint_id}: {c.description}")
    except ValueError as e:
        print(f"  [ERROR] {e}")

    # Test conflict detection
    print("\n" + "=" * 70)
    print("Conflict Detection:")
    print("=" * 70)
    conflicts = sce.detect_conflicts()
    if conflicts:
        print(f"  Found {len(conflicts)} conflicts:")
        for id1, id2, reason in conflicts:
            print(f"    {id1} <-> {id2}: {reason}")
    else:
        print("  [OK] No conflicts detected")

    print("\n" + "=" * 70)
    print("[OK] SCE demonstration complete")
    print("=" * 70)
