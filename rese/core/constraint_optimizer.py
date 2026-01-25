"""
Constraint Optimizer for Symbolic Constraint Engine

Provides constraint satisfaction solving using Z3 SMT solver.
Implements constraint prioritization and conflict resolution.

Author: Agent A1
Created: 2025-12-31
Status: Active Implementation
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import re

try:
    from z3 import (
        Solver, Real, Bool, Int, ArithRef, BoolRef,
        sat, unsat, unknown, And, Or, Not, Implies
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    # Create dummy classes for type hints when Z3 is not available
    Solver = None
    sat = None
    unsat = None

from .symbolic_constraint_engine import (
    Constraint,
    ConstraintType,
    SymbolicConstraintEngine
)


class ResolutionStrategy(Enum):
    """Strategies for resolving constraint conflicts"""
    PRIORITY_BASED = "priority"  # Keep higher priority constraints
    SATISFIABILITY = "satisfiability"  # Maximize satisfiable constraints
    MINIMAL_REMOVAL = "minimal"  # Remove minimum constraints to satisfy
    WEIGHTED = "weighted"  # Use constraint weights


@dataclass
class OptimizationResult:
    """
    Result of constraint optimization.

    Attributes:
        satisfiable: Whether constraints are satisfiable
        solution: Variable assignments (if satisfiable)
        unsatisfied_constraints: Constraints that couldn't be satisfied
        removed_constraints: Constraints removed during resolution
        strategy: Strategy used for resolution
    """
    satisfiable: bool
    solution: Dict[str, float]  # Variable name -> value
    unsatisfied_constraints: List[str]  # Constraint IDs
    removed_constraints: List[str]  # Constraint IDs
    strategy: ResolutionStrategy
    solver_time_ms: float  # Time taken by solver

    def __post_init__(self):
        if self.solution is None:
            self.solution = {}
        if self.unsatisfied_constraints is None:
            self.unsatisfied_constraints = []
        if self.removed_constraints is None:
            self.removed_constraints = []


class ConstraintOptimizer:
    """
    Optimizes and solves constraints using Z3 SMT solver.

    Features:
    - Constraint satisfaction solving
    - Conflict detection and resolution
    - Constraint prioritization
    - Solution extraction
    """

    def __init__(self, sce: Optional[SymbolicConstraintEngine] = None):
        """
        Initialize constraint optimizer.

        Args:
            sce: Optional existing constraint engine
        """
        self.sce = sce or SymbolicConstraintEngine()

        if not Z3_AVAILABLE:
            print("[WARNING] Z3 not available. Install with: pip install z3-solver")

        self._var_counter = 0

    def check_satisfiability(self, constraints: Optional[List[Constraint]] = None) -> Tuple[bool, str]:
        """
        Check if constraints are satisfiable.

        Args:
            constraints: Optional list of constraints (uses all if None)

        Returns:
            Tuple of (is_satisfiable, message)
        """
        if not Z3_AVAILABLE:
            return False, "Z3 solver not available"

        if constraints is None:
            constraints = self.sce.get_all_constraints()

        if not constraints:
            return True, "No constraints to check"

        try:
            # Create Z3 solver
            solver = Solver()

            # Add hard constraints only
            for constraint in constraints:
                if constraint.is_hard():
                    z3_constraint = self._constraint_to_z3(constraint)
                    if z3_constraint is not None:
                        solver.add(z3_constraint)

            # Check satisfiability
            result = solver.check()

            if result == sat:
                return True, "Constraints are satisfiable"
            elif result == unsat:
                return False, "Constraints are unsatisfiable"
            else:
                return False, "Could not determine satisfiability"

        except Exception as e:
            return False, f"Error checking satisfiability: {str(e)}"

    def find_solution(
        self,
        constraints: Optional[List[Constraint]] = None
    ) -> OptimizationResult:
        """
        Find a solution satisfying all constraints.

        Args:
            constraints: Optional list of constraints

        Returns:
            OptimizationResult with solution
        """
        if not Z3_AVAILABLE:
            return OptimizationResult(
                satisfiable=False,
                solution={},
                unsatisfied_constraints=[c.id for c in (constraints or self.sce.get_all_constraints())],
                removed_constraints=[],
                strategy=ResolutionStrategy.SATISFIABILITY,
                solver_time_ms=0.0
            )

        if constraints is None:
            constraints = self.sce.get_all_constraints()

        import time
        start_time = time.time()

        try:
            solver = Solver()

            # Add all constraints (prioritize hard > soft > preference)
            for constraint in constraints:
                z3_constraint = self._constraint_to_z3(constraint)
                if z3_constraint is not None:
                    solver.add(z3_constraint)

            result = solver.check()

            if result == sat:
                # Extract solution
                model = solver.model()
                solution = self._extract_solution(model)

                return OptimizationResult(
                    satisfiable=True,
                    solution=solution,
                    unsatisfied_constraints=[],
                    removed_constraints=[],
                    strategy=ResolutionStrategy.SATISFIABILITY,
                    solver_time_ms=(time.time() - start_time) * 1000
                )
            else:
                # Unsat - try to resolve conflicts
                return self._resolve_conflicts(
                    constraints,
                    ResolutionStrategy.PRIORITY_BASED,
                    start_time
                )

        except Exception as e:
            return OptimizationResult(
                satisfiable=False,
                solution={},
                unsatisfied_constraints=[c.id for c in constraints],
                removed_constraints=[],
                strategy=ResolutionStrategy.SATISFIABILITY,
                solver_time_ms=(time.time() - start_time) * 1000
            )

    def _constraint_to_z3(self, constraint: Constraint) -> Optional[BoolRef]:
        """
        Convert a constraint to Z3 BoolRef.

        Args:
            constraint: Python constraint

        Returns:
            Z3 BoolRef or None if conversion fails
        """
        try:
            formal = constraint.formalization

            # Extract variable names
            variables = self._extract_variables_from_formal(formal)

            # Create Z3 variables
            var_map = {}
            for var in variables:
                var_map[var] = Real(var)

            # Parse formalization and create Z3 expression
            expr = self._parse_formalization(formal, var_map)

            return expr

        except Exception as e:
            # If parsing fails, return None
            return None

    def _extract_variables_from_formal(self, formal: str) -> List[str]:
        """
        Extract variable names from formal constraint.

        Args:
            formal: Formal constraint string

        Returns:
            List of variable names
        """
        # Look for patterns like "forall T : Real" or "∀ T : Real"
        pattern = r'(?:forall|∀)\s+(\w+)\s*:\s*\w+'
        matches = re.findall(pattern, formal)

        if matches:
            return list(set(matches))

        # If no explicit quantifier, look for single letters
        pattern = r'\b([a-zA-Z])\b'
        all_letters = re.findall(pattern, formal)

        # Filter out common non-variable letters
        non_vars = {'a', 'I', 'x'}  # Common words
        variables = [l for l in all_letters if l not in non_vars]

        return list(set(variables))

    def _parse_formalization(self, formal: str, var_map: Dict[str, ArithRef]) -> Optional[BoolRef]:
        """
        Parse formal constraint string into Z3 expression.

        Args:
            formal: Formal constraint string
            var_map: Variable name -> Z3 variable mapping

        Returns:
            Z3 BoolRef or None if parsing fails
        """
        # Simple parser for common patterns

        # Pattern 1: "T < 1000"
        match = re.match(r'(\w+)\s*<\s*(\d+(?:\.\d+)?)', formal)
        if match:
            var = var_map.get(match.group(1))
            if var:
                value = float(match.group(2))
                return var < value

        # Pattern 2: "T > 500"
        match = re.match(r'(\w+)\s*>\s*(\d+(?:\.\d+)?)', formal)
        if match:
            var = var_map.get(match.group(1))
            if var:
                value = float(match.group(2))
                return var > value

        # Pattern 3: "T <= 1000" or "T ≤ 1000"
        match = re.match(r'(\w+)\s*[≤<=]\s*(\d+(?:\.\d+)?)', formal)
        if match:
            var = var_map.get(match.group(1))
            if var:
                value = float(match.group(2))
                return var <= value

        # Pattern 4: "T >= 500" or "T ≥ 500"
        match = re.match(r'(\w+)\s*[≥>=]\s*(\d+(?:\.\d+)?)', formal)
        if match:
            var = var_map.get(match.group(1))
            if var:
                value = float(match.group(2))
                return var >= value

        # Pattern 5: "T = 100"
        match = re.match(r'(\w+)\s*=\s*(\d+(?:\.\d+)?)', formal)
        if match:
            var = var_map.get(match.group(1))
            if var:
                value = float(match.group(2))
                return var == value

        # Pattern 6: "forall T : Real, T < 1000"
        match = re.match(r'(?:forall|∀)\s+(\w+)\s*:\s*\w+,\s*(.+)', formal)
        if match:
            var_name = match.group(1)
            expr = match.group(2)
            var = var_map.get(var_name)
            if var:
                # Recursively parse the expression
                return self._parse_formalization(expr, var_map)

        return None

    def _extract_solution(self, model) -> Dict[str, float]:
        """
        Extract variable assignments from Z3 model.

        Args:
            model: Z3 model

        Returns:
            Dictionary of variable -> value
        """
        solution = {}

        try:
            for decl in model:
                name = decl.name()
                value = model[decl]

                # Convert to float if possible
                try:
                    if hasattr(value, 'as_decimal'):
                        # Handle rational numbers
                        solution[name] = float(value.as_decimal(10))
                    elif hasattr(value, 'as_long'):
                        solution[name] = float(value.as_long())
                    else:
                        solution[name] = float(str(value))
                except:
                    solution[name] = str(value)

        except Exception as e:
            pass

        return solution

    def _resolve_conflicts(
        self,
        constraints: List[Constraint],
        strategy: ResolutionStrategy,
        start_time: float
    ) -> OptimizationResult:
        """
        Resolve constraint conflicts using specified strategy.

        Args:
            constraints: List of constraints
            strategy: Resolution strategy
            start_time: Start time for timing

        Returns:
            OptimizationResult with resolution
        """
        import time

        if strategy == ResolutionStrategy.PRIORITY_BASED:
            return self._resolve_by_priority(constraints, start_time)
        elif strategy == ResolutionStrategy.MINIMAL_REMOVAL:
            return self._resolve_by_minimal_removal(constraints, start_time)
        else:
            return self._resolve_by_priority(constraints, start_time)

    def _resolve_by_priority(
        self,
        constraints: List[Constraint],
        start_time: float
    ) -> OptimizationResult:
        """
        Resolve conflicts by keeping higher priority constraints.

        Priority: HARD > SOFT > PREFERENCE

        Args:
            constraints: List of constraints
            start_time: Start time

        Returns:
            OptimizationResult
        """
        import time

        # Sort by priority
        sorted_constraints = sorted(
            constraints,
            key=lambda c: ({
                ConstraintType.HARD: 0,
                ConstraintType.SOFT: 1,
                ConstraintType.PREFERENCE: 2
            }[c.type])
        )

        removed = []

        # Try to satisfy constraints in priority order
        for i in range(len(sorted_constraints)):
            current_constraints = sorted_constraints[i:]

            satisfiable, _ = self.check_satisfiability(current_constraints)

            if satisfiable:
                # Found satisfiable subset
                unsatisfied = [c.id for c in sorted_constraints[:i]]
                return OptimizationResult(
                    satisfiable=True,
                    solution={},  # Could extract actual solution
                    unsatisfied_constraints=unsatisfied,
                    removed_constraints=removed,
                    strategy=ResolutionStrategy.PRIORITY_BASED,
                    solver_time_ms=(time.time() - start_time) * 1000
                )

        # No satisfiable subset found
        return OptimizationResult(
            satisfiable=False,
            solution={},
            unsatisfied_constraints=[c.id for c in constraints],
            removed_constraints=[],
            strategy=ResolutionStrategy.PRIORITY_BASED,
            solver_time_ms=(time.time() - start_time) * 1000
        )

    def _resolve_by_minimal_removal(
        self,
        constraints: List[Constraint],
        start_time: float
    ) -> OptimizationResult:
        """
        Resolve conflicts by removing minimal number of constraints.

        Args:
            constraints: List of constraints
            start_time: Start time

        Returns:
            OptimizationResult
        """
        import time

        # Try removing one constraint at a time
        for i in range(len(constraints)):
            test_constraints = constraints[:i] + constraints[i+1:]

            satisfiable, _ = self.check_satisfiability(test_constraints)

            if satisfiable:
                removed = [constraints[i].id]
                return OptimizationResult(
                    satisfiable=True,
                    solution={},
                    unsatisfied_constraints=[],
                    removed_constraints=removed,
                    strategy=ResolutionStrategy.MINIMAL_REMOVAL,
                    solver_time_ms=(time.time() - start_time) * 1000
                )

        # Try removing two constraints
        for i in range(len(constraints)):
            for j in range(i+1, len(constraints)):
                test_constraints = [
                    c for k, c in enumerate(constraints)
                    if k != i and k != j
                ]

                satisfiable, _ = self.check_satisfiability(test_constraints)

                if satisfiable:
                    removed = [constraints[i].id, constraints[j].id]
                    return OptimizationResult(
                        satisfiable=True,
                        solution={},
                        unsatisfied_constraints=[],
                        removed_constraints=removed,
                        strategy=ResolutionStrategy.MINIMAL_REMOVAL,
                        solver_time_ms=(time.time() - start_time) * 1000
                    )

        # Couldn't find solution
        return OptimizationResult(
            satisfiable=False,
            solution={},
            unsatisfied_constraints=[c.id for c in constraints],
            removed_constraints=[],
            strategy=ResolutionStrategy.MINIMAL_REMOVAL,
            solver_time_ms=(time.time() - start_time) * 1000
        )

    def prioritize_constraints(self) -> List[Tuple[str, float]]:
        """
        Calculate priority scores for all constraints.

        Returns:
            List of (constraint_id, priority_score) tuples

        Priority factors:
        - Type (HARD=3, SOFT=2, PREFERENCE=1)
        - Dependencies (more dependencies = higher priority)
        - Verification status (verified = higher priority)
        """
        constraints = self.sce.get_all_constraints()
        priorities = []

        for constraint in constraints:
            # Base priority from type
            type_score = {
                ConstraintType.HARD: 3.0,
                ConstraintType.SOFT: 2.0,
                ConstraintType.PREFERENCE: 1.0
            }[constraint.type]

            # Boost for dependencies
            dependency_boost = len(constraint.dependencies) * 0.1

            # Boost for verification
            verification_boost = 0.5 if constraint.is_verified() else 0.0

            # Calculate total priority
            priority = type_score + dependency_boost + verification_boost

            priorities.append((constraint.id, priority))

        # Sort by priority (descending)
        priorities.sort(key=lambda x: x[1], reverse=True)

        return priorities

    def get_statistics(self) -> Dict[str, any]:
        """Get optimizer statistics"""
        stats = self.sce.get_statistics()

        # Add optimizer-specific stats
        stats["z3_available"] = Z3_AVAILABLE
        stats["priorities"] = self.prioritize_constraints()

        return stats


# Convenience functions

def optimize_constraints(constraints: List[Constraint]) -> OptimizationResult:
    """
    Optimize a list of constraints (convenience function).

    Args:
        constraints: List of constraints to optimize

    Returns:
        OptimizationResult
    """
    sce = SymbolicConstraintEngine()
    for c in constraints:
        try:
            sce.add_constraint(c)
        except ValueError:
            pass

    optimizer = ConstraintOptimizer(sce)
    return optimizer.find_solution()


# Testing and demonstration

if __name__ == "__main__":
    print("=" * 70)
    print("Constraint Optimizer - Demonstration")
    print("=" * 70)

    from symbolic_constraint_engine import SymbolicConstraintEngine

    # Create test constraints
    sce = SymbolicConstraintEngine()

    c1 = Constraint(
        id="temp_low",
        type=ConstraintType.HARD,
        description="Temperature must be greater than 0",
        formalization="T > 0",
        source="test"
    )

    c2 = Constraint(
        id="temp_high",
        type=ConstraintType.HARD,
        description="Temperature must be less than 1000",
        formalization="T < 1000",
        source="test"
    )

    c3 = Constraint(
        id="temp_optimal",
        type=ConstraintType.SOFT,
        description="Temperature should be around 500",
        formalization="T = 500",
        source="test"
    )

    sce.add_constraint(c1)
    sce.add_constraint(c2)
    sce.add_constraint(c3)

    # Create optimizer
    optimizer = ConstraintOptimizer(sce)
    print("\n[OK] Constraint Optimizer initialized")

    # Check satisfiability
    print("\n" + "=" * 70)
    print("Satisfiability Check:")
    print("=" * 70)
    satisfiable, message = optimizer.check_satisfiability()
    print(f"Satisfiable: {satisfiable}")
    print(f"Message: {message}")

    # Find solution
    print("\n" + "=" * 70)
    print("Finding Solution:")
    print("=" * 70)
    result = optimizer.find_solution()

    print(f"Satisfiable: {result.satisfiable}")
    print(f"Strategy: {result.strategy.value}")
    print(f"Solver time: {result.solver_time_ms:.2f}ms")

    if result.solution:
        print("\nSolution:")
        for var, value in result.solution.items():
            print(f"  {var} = {value}")

    if result.unsatisfied_constraints:
        print("\nUnsatisfied constraints:")
        for c_id in result.unsatisfied_constraints:
            print(f"  - {c_id}")

    if result.removed_constraints:
        print("\nRemoved constraints:")
        for c_id in result.removed_constraints:
            print(f"  - {c_id}")

    # Show priorities
    print("\n" + "=" * 70)
    print("Constraint Priorities:")
    print("=" * 70)
    priorities = optimizer.prioritize_constraints()
    for c_id, priority in priorities:
        print(f"  {c_id}: {priority:.2f}")

    # Statistics
    print("\n" + "=" * 70)
    print("Statistics:")
    print("=" * 70)
    stats = optimizer.get_statistics()
    for key, value in stats.items():
        if key != "priorities":
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("[OK] Constraint Optimizer demonstration complete")
    print("=" * 70)
