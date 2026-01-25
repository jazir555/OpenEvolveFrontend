"""
Example 2: Symbolic Constraint Engine (SCE) - Basic Usage

This example demonstrates how to use the SCE directly to manage constraints.
"""

import sys
sys.path.insert(0, r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend')

from core.symbolic_constraint_engine import (
    SymbolicConstraintEngine,
    Constraint,
    ConstraintType
)

def main():
    print("=" * 60)
    print("Example 2: Symbolic Constraint Engine - Basic Usage")
    print("=" * 60)
    print()

    # Create constraint engine
    sce = SymbolicConstraintEngine()
    print("Created Symbolic Constraint Engine")
    print()

    # Add constraints
    print("Adding Constraints:")
    print("-" * 60)

    constraints = [
        Constraint(
            id="c1",
            type=ConstraintType.HARD,
            description="All variables must be positive",
            formalization="∀ x ∈ variables: x > 0",
            source="user"
        ),
        Constraint(
            id="c2",
            type=ConstraintType.HARD,
            description="Variables must be less than 100",
            formalization="∀ x ∈ variables: x < 100",
            source="user"
        ),
        Constraint(
            id="c3",
            type=ConstraintType.SOFT,
            description="Prefer values near 50",
            formalization="minimize |x - 50|",
            source="preference"
        )
    ]

    for constraint in constraints:
        sce.add_constraint(constraint)
        print(f"Added: {constraint.id} - {constraint.description}")
        print(f"  Type: {constraint.type.value}")
        print(f"  Formalization: {constraint.formalization}")
        print()

    # Get all constraints
    print("-" * 60)
    print(f"Total Constraints: {len(sce.get_all_constraints())}")
    print()

    # Detect conflicts
    print("Detecting Conflicts:")
    print("-" * 60)
    conflicts = sce.detect_conflicts()

    if conflicts:
        print(f"Found {len(conflicts)} conflicts:")
        for c1, c2 in conflicts:
            print(f"  - {c1} vs {c2}")
    else:
        print("No conflicts detected!")
    print()

    # Get execution order
    print("Execution Order:")
    print("-" * 60)
    order = sce.get_execution_order()
    print(f"Topological sort: {' → '.join(order)}")
    print()

    # Validate all constraints
    print("Validation:")
    print("-" * 60)
    validation = sce.validate()
    print(f"Valid: {validation['is_valid']}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    print()

    # Statistics
    print("Statistics:")
    print("-" * 60)
    print(f"Total constraints: {len(sce.get_all_constraints())}")
    print(f"Hard constraints: {sum(1 for c in sce.get_all_constraints() if c.is_hard())}")
    print(f"Soft constraints: {sum(1 for c in sce.get_all_constraints() if not c.is_hard())}")
    print(f"Verified constraints: {sum(1 for c in sce.get_all_constraints() if c.is_verified())}")
    print()

    print("=" * 60)
    print("Example 2 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
