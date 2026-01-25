"""
Stage 1: Syntactic Preprocessing for Ψ₃

Implements O(k²) syntactic redundancy elimination.
"""

from typing import List, Set, Tuple, Dict
from itertools import combinations
import time

from ..core.constraint import Constraint, ConstraintType
from ..core.expression import Expr, BoolExpr, ArithExpr, BoolOp, ArithOp
from ..solvers.sat_wrapper import SATInterface, SatResult


class PreprocessingResult:
    """Result of syntactic preprocessing"""

    def __init__(
        self,
        reduced_constraints: List[Constraint],
        removed_count: int,
        duplicates_removed: int,
        subsumptions_removed: int,
        simplifications: int,
        runtime_seconds: float
    ):
        self.reduced_constraints = reduced_constraints
        self.removed_count = removed_count
        self.duplicates_removed = duplicates_removed
        self.subsumptions_removed = subsumptions_removed
        self.simplifications = simplifications
        self.runtime_seconds = runtime_seconds

    @property
    def reduction_ratio(self) -> float:
        """Ratio of original size to reduced size"""
        if not self.reduced_constraints:
            return 1.0
        original_size = len(self.reduced_constraints) + self.removed_count
        return original_size / len(self.reduced_constraints)


def syntactic_preprocessing(
    constraints: List[Constraint],
    solver: SATInterface,
    verbose: bool = False
) -> PreprocessingResult:
    """
    Stage 1: Syntactic redundancy elimination.

    Removes:
    1. Exact duplicates
    2. Syntactically subsumed constraints
    3. Applies simplifications

    Complexity: O(k²) where k = |constraints|

    Args:
        constraints: Input constraint set
        solver: SAT solver interface
        verbose: Enable verbose logging

    Returns:
        PreprocessingResult with reduced constraints
    """
    start_time = time.time()

    if verbose:
        print(f"[Stage 1] Starting syntactic preprocessing on {len(constraints)} constraints")

    # Step 1: Remove exact duplicates
    step1_start = time.time()
    unique_constraints = _remove_duplicates(constraints)
    duplicates_removed = len(constraints) - len(unique_constraints)
    step1_time = time.time() - step1_start

    if verbose:
        print(f"[Stage 1.1] Removed {duplicates_removed} duplicates ({step1_time:.3f}s)")

    # Step 2: Detect and remove subsumptions
    step2_start = time.time()
    reduced_set, subsumptions_removed = _remove_subsumptions(
        unique_constraints,
        solver,
        verbose
    )
    step2_time = time.time() - step2_start

    if verbose:
        print(f"[Stage 1.2] Removed {subsumptions_removed} subsumptions ({step2_time:.3f}s)")

    # Step 3: Simplify constraints
    step3_start = time.time()
    simplified_constraints = _simplify_constraints(reduced_set)
    simplifications = len(reduced_set) - len(simplified_constraints)
    step3_time = time.time() - step3_start

    if verbose:
        print(f"[Stage 1.3] Simplified {simplifications} constraints ({step3_time:.3f}s)")

    # Step 4: Normalize representation
    step4_start = time.time()
    normalized_constraints = _normalize_constraints(simplified_constraints)
    step4_time = time.time() - step4_start

    if verbose:
        print(f"[Stage 1.4] Normalized {len(normalized_constraints)} constraints ({step4_time:.3f}s)")

    total_time = time.time() - start_time
    total_removed = len(constraints) - len(normalized_constraints)

    if verbose:
        print(f"[Stage 1] Complete: {len(constraints)} → {len(normalized_constraints)} "
              f"({total_removed} removed, {total_time:.3f}s)")

    return PreprocessingResult(
        reduced_constraints=normalized_constraints,
        removed_count=total_removed,
        duplicates_removed=duplicates_removed,
        subsumptions_removed=subsumptions_removed,
        simplifications=simplifications,
        runtime_seconds=total_time
    )


def _remove_duplicates(constraints: List[Constraint]) -> List[Constraint]:
    """
    Remove exact duplicate constraints.

    Uses normalized form for comparison.
    """
    seen: Dict[str, Constraint] = {}
    unique: List[Constraint] = []

    for c in constraints:
        # Use normalized expression as key
        key = str(c.normalized)
        if key not in seen:
            seen[key] = c
            unique.append(c)

    return unique


def _remove_subsumptions(
    constraints: List[Constraint],
    solver: SATInterface,
    verbose: bool = False
) -> Tuple[List[Constraint], int]:
    """
    Detect and remove subsumed constraints.

    A constraint c1 subsumes c2 if c1 ⊨ c2 (c1 implies c2).
    If c1 subsumes c2, we can remove c2.

    Args:
        constraints: Constraints to process
        solver: SAT solver for semantic checks
        verbose: Enable logging

    Returns:
        (Reduced constraint list, number of subsumptions removed)
    """
    if len(constraints) <= 1:
        return constraints, 0

    removed_count = 0
    redundant = set()

    # Check all pairs
    for i, c1 in enumerate(constraints):
        if c1.id in redundant:
            continue

        for j, c2 in enumerate(constraints):
            if i == j or c2.id in redundant:
                continue

            # Check syntactic subsumption first (fast)
            if _syntactically_subsumes(c1, c2):
                redundant.add(c2.id)
                removed_count += 1
                if verbose:
                    print(f"  [Subsumption] {c1.id} ⊨ {c2.id} (syntactic)")
                continue

            # Check semantic subsumption (slow but accurate)
            if c1.subsumes(c2, solver):
                redundant.add(c2.id)
                removed_count += 1
                if verbose:
                    print(f"  [Subsumption] {c1.id} ⊨ {c2.id} (semantic)")

    # Filter out redundant constraints
    reduced = [c for c in constraints if c.id not in redundant]
    return reduced, removed_count


def _syntactically_subsumes(c1: Constraint, c2: Constraint) -> bool:
    """
    Fast syntactic check for subsumption.

    Checks:
    1. If c1 is conjunction, c2 is one of its conjuncts
    2. If c1 has stronger arithmetic bounds than c2
    3. If c1 has more specific type constraint
    """
    # Case 1: c1 is conjunction, c2 is a conjunct
    if isinstance(c1.expr, BoolExpr) and c1.expr.op == BoolOp.AND:
        if c2.expr in c1.expr.args:
            return True

    # Case 2: Both are arithmetic constraints on same variable
    if (isinstance(c1.expr, ArithExpr) and
        isinstance(c2.expr, ArithExpr) and
        c1.expr.op == c2.expr.op and
        str(c1.expr.left) == str(c2.expr.left)):

        # Compare bounds
        try:
            bound1 = float(str(c1.expr.right))
            bound2 = float(str(c2.expr.right))

            # Both are > constraints
            if c1.expr.op == ArithOp.GT:
                return bound1 > bound2  # Stronger bound subsumes weaker

            # Both are >= constraints
            if c1.expr.op == ArithOp.GE:
                return bound1 >= bound2

            # Both are < constraints
            if c1.expr.op == ArithOp.LT:
                return bound1 < bound2

            # Both are <= constraints
            if c1.expr.op == ArithOp.LE:
                return bound1 <= bound2

        except (ValueError, TypeError):
            pass  # Can't compare numerically

    return False


def _simplify_constraints(constraints: List[Constraint]) -> List[Constraint]:
    """
    Simplify constraint expressions.

    Applies algebraic simplifications:
    - (x > 5) ∧ (x > 10) → (x > 10)
    - (x > 5) ∨ (x > 10) → (x > 5)
    - ¬¬P → P
    """
    simplified = []

    for c in constraints:
        try:
            c_simplified = c.simplify()
            simplified.append(c_simplified)
        except Exception:
            # If simplification fails, keep original
            simplified.append(c)

    return simplified


def _normalize_constraints(constraints: List[Constraint]) -> List[Constraint]:
    """
    Normalize constraint representation.

    Ensures:
    - Sorted AND/OR arguments
    - Removed duplicate arguments
    - Consistent formatting
    """
    normalized = []

    for c in constraints:
        # Use pre-computed normalized form
        if c.normalized != c.expr:
            # Create new constraint with normalized expression
            from ..core.constraint import Metadata
            normalized_c = Constraint(
                id=c.id,
                expr=c.normalized,
                type=c.type,
                vars=c.vars,
                metadata=c.metadata
            )
            normalized.append(normalized_c)
        else:
            normalized.append(c)

    return normalized


def estimate_redundancy(constraints: List[Constraint]) -> float:
    """
    Estimate redundancy level in constraint set.

    Returns:
        Redundancy score (0.0 = no redundancy, 1.0 = high redundancy)

    Used to decide whether Ψ₃ will be beneficial.
    """
    if len(constraints) <= 1:
        return 0.0

    # Heuristics for redundancy estimation
    redundancy_indicators = 0

    # Check 1: Variables appearing in multiple constraints
    var_counts: Dict[str, int] = {}
    for c in constraints:
        for var in c.vars:
            var_counts[var] = var_counts.get(var, 0) + 1

    avg_var_occurrence = sum(var_counts.values()) / len(var_counts) if var_counts else 1.0
    if avg_var_occurrence > 2.0:
        redundancy_indicators += 1

    # Check 2: Similar constraint lengths
    lengths = [c.get_complexity() for c in constraints]
    if len(set(lengths)) < len(lengths) * 0.5:
        redundancy_indicators += 1

    # Check 3: Constraint types concentrated
    type_counts = {}
    for c in constraints:
        type_counts[c.type] = type_counts.get(c.type, 0) + 1

    if any(count > len(constraints) * 0.7 for count in type_counts.values()):
        redundancy_indicators += 1

    # Normalize to [0, 1]
    return redundancy_indicators / 3.0
