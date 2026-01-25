"""
Core Constraint Data Structures for Ψ₃

Defines immutable constraint representations with cached computations.
"""

from dataclasses import dataclass, field
from typing import Set, Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum
from datetime import datetime
import hashlib

# Forward references to avoid circular imports
if TYPE_CHECKING:
    from .expression import Expr


class ConstraintType(Enum):
    """Types of constraints in the Ψ₃ system"""
    BOOL = "bool"           # Boolean expression
    ARITH = "arith"         # Arithmetic expression
    QUANT = "quant"         # Quantified expression
    TYPE = "type"           # Type constraint


@dataclass
class Metadata:
    """
    Constraint metadata tracking provenance and properties.

    Attributes:
        source: Origin of constraint (user, derived, system, etc.)
        priority: Importance level (1-10, higher = more important)
        confidence: Trust level (0.0-1.0)
        dependencies: IDs of constraints this constraint implies
        verified: Formal verification status
        timestamp: Creation time
        tags: User-defined tags for organization
    """
    source: str
    priority: int = field(default=5)
    confidence: float = field(default=1.0)
    dependencies: List[int] = field(default_factory=list)
    verified: bool = field(default=False)
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Validate metadata fields"""
        if not 1 <= self.priority <= 10:
            raise ValueError(f"Priority must be 1-10, got {self.priority}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")


@dataclass(frozen=True)
class Constraint:
    """
    Immutable constraint representation.

    This is the core data structure for Ψ₃. Constraints are immutable and hashable,
    making them suitable for use in sets and as dictionary keys.

    Attributes:
        id: Unique identifier for this constraint
        expr: Logical expression (AST)
        type: Constraint classification
        vars: Free variables in the constraint
        metadata: Provenance information

    Properties:
        hash: Pre-computed hash for fast equality checks
        normalized: Normalized form of the expression
    """
    id: int
    expr: Any  # Will be 'Expr' at runtime
    type: ConstraintType
    vars: frozenset[str]
    metadata: Metadata

    # Cached fields (computed on construction)
    hash: int = field(init=False, repr=False)
    _normalized: Optional[Any] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        """Compute cached fields"""
        # Compute hash from expression
        expr_str = str(self.expr)
        hash_value = hashlib.sha256(expr_str.encode()).hexdigest()
        object.__setattr__(self, 'hash', hash((self.id, hash_value)))

        # Normalize expression
        object.__setattr__(self, '_normalized', self._normalize_expr(self.expr))

    def _normalize_expr(self, expr: Any) -> Any:
        """
        Normalize expression to canonical form.

        This helps with equivalence checking by ensuring
        structurally equivalent expressions have the same representation.
        """
        from .expression import BoolExpr, BoolOp

        if isinstance(expr, BoolExpr):
            # Sort AND and OR arguments (commutative)
            if expr.op in (BoolOp.AND, BoolOp.OR):
                # Recursively normalize arguments
                normalized_args = [self._normalize_expr(arg) for arg in expr.args]
                # Sort for canonical form
                sorted_args = sorted(normalized_args, key=str)
                # Remove duplicates
                unique_args = []
                seen = set()
                for arg in sorted_args:
                    arg_str = str(arg)
                    if arg_str not in seen:
                        seen.add(arg_str)
                        unique_args.append(arg)
                return BoolExpr(expr.op, unique_args)

        return expr

    @property
    def normalized(self) -> Any:
        """Get normalized form of the constraint's expression"""
        return self._normalized

    def subsumes(self, other: 'Constraint', solver: Any) -> bool:
        """
        Check if self ⊨ other (self implies other).

        Uses SAT solver to check implication:
            self ⊨ other  iff  UNSAT(self ∧ ¬other)

        Args:
            other: Constraint to check implication for
            solver: SAT solver interface

        Returns:
            True if self implies other, False otherwise
        """
        from .expression import Not, And

        # Quick check: if normalized forms are equal, they're equivalent
        if self.normalized == other.normalized:
            return True

        # Use SAT solver for semantic check
        negation = And(self.expr, Not(other.expr))
        result = solver.check_sat(negation)
        return result == SatResult.UNSATISFIABLE

    def is_equivalent(self, other: 'Constraint', solver: Any) -> bool:
        """
        Check if self ≡ other (mutual implication).

        Args:
            other: Constraint to check equivalence with
            solver: SAT solver interface

        Returns:
            True if constraints are equivalent, False otherwise
        """
        return self.subsumes(other, solver) and other.subsumes(self, solver)

    def simplify(self) -> 'Constraint':
        """
        Simplify constraint expression.

        Returns:
            New constraint with simplified expression
        """
        simplified_expr = self._simplify_expr(self.expr)
        return Constraint(
            id=self.id,
            expr=simplified_expr,
            type=self.type,
            vars=self.vars,
            metadata=self.metadata
        )

    def _simplify_expr(self, expr: Any) -> Any:
        """
        Simplify an expression using algebraic rules.

        Examples:
            - (x > 5) ∧ (x > 10) → (x > 10)
            - (x > 5) ∨ (x > 10) → (x > 5)
            - ¬(¬P) → P
        """
        from .expression import BoolExpr, ArithExpr, BoolOp, ArithOp, Not

        if isinstance(expr, BoolExpr):
            # Double negation elimination
            if expr.op == BoolOp.NOT and isinstance(expr.args[0], BoolExpr):
                if expr.args[0].op == BoolOp.NOT:
                    return self._simplify_expr(expr.args[0].args[0])

            # Simplify AND/OR arguments recursively
            simplified_args = [self._simplify_expr(arg) for arg in expr.args]

            # Apply domain-specific simplifications
            if expr.op == BoolOp.AND and len(simplified_args) == 2:
                # Check for arithmetic constraints that can be combined
                if (isinstance(simplified_args[0], ArithExpr) and
                    isinstance(simplified_args[1], ArithExpr)):
                    return self._simplify_arith_conjunction(
                        simplified_args[0],
                        simplified_args[1]
                    )

            return BoolExpr(expr.op, simplified_args)

        return expr

    def _simplify_arith_conjunction(
        self,
        left: Any,
        right: Any
    ) -> Any:
        """
        Simplify conjunction of arithmetic constraints.

        Example: (x > 5) ∧ (x > 10) → (x > 10)
        """
        # Check if both are comparisons of the same variable
        if (isinstance(left.left, type(right.left)) and
            str(left.left) == str(right.left)):

            # Both are > constraints
            if left.op == ArithOp.GT and right.op == ArithOp.GT:
                # Keep the stronger one
                return left if str(left.right) > str(right.right) else right

            # Both are >= constraints
            if left.op == ArithOp.GE and right.op == ArithOp.GE:
                return left if str(left.right) > str(right.right) else right

            # Similar simplifications for <, <=, etc.
            # (omitted for brevity)

        # Can't simplify, return conjunction
        from .expression import BoolExpr, BoolOp
        return BoolExpr(BoolOp.AND, [left, right])

    def get_complexity(self) -> int:
        """
        Estimate complexity of constraint.

        Returns:
            Complexity score (higher = more complex)
        """
        return self._count_nodes(self.expr)

    def _count_nodes(self, expr: Any) -> int:
        """Count nodes in expression tree"""
        from .expression import BoolExpr

        count = 1
        if isinstance(expr, BoolExpr):
            for arg in expr.args:
                count += self._count_nodes(arg)
        return count

    def to_lean4(self) -> str:
        """
        Convert constraint to Lean 4 syntax.

        Returns:
            Lean 4 expression string
        """
        return self._expr_to_lean4(self.expr)

    def _expr_to_lean4(self, expr: Any) -> str:
        """Recursively convert expression to Lean 4"""
        from .expression import BoolExpr, ArithExpr, QuantExpr, BoolOp, ArithOp, Quantifier

        if isinstance(expr, BoolExpr):
            match expr.op:
                case BoolOp.AND:
                    inner = ' '.join([self._expr_to_lean4(a) for a in expr.args])
                    return f"(And {inner})"
                case BoolOp.OR:
                    inner = ' '.join([self._expr_to_lean4(a) for a in expr.args])
                    return f"(Or {inner})"
                case BoolOp.NOT:
                    return f"(Not {self._expr_to_lean4(expr.args[0])})"
                case BoolOp.IMPLIES:
                    left = self._expr_to_lean4(expr.args[0])
                    right = self._expr_to_lean4(expr.args[1])
                    return f"(Imp {left} {right})"
                case _:
                    return f"?{expr.op}"

        elif isinstance(expr, ArithExpr):
            left = self._expr_to_lean4(expr.left)
            right = self._expr_to_lean4(expr.right)
            match expr.op:
                case ArithOp.LT:
                    return f"(Lt {left} {right})"
                case ArithOp.LE:
                    return f"(Le {left} {right})"
                case ArithOp.GT:
                    return f"(Gt {left} {right})"
                case ArithOp.GE:
                    return f"(Ge {left} {right})"
                case ArithOp.EQ:
                    return f"(Eq {left} {right})"
                case ArithOp.NE:
                    return f"(Ne {left} {right})"
                case _:
                    return f"?{expr.op}"

        elif isinstance(expr, QuantExpr):
            var_str = ' '.join(expr.vars)
            body = self._expr_to_lean4(expr.body)
            match expr.quant:
                case Quantifier.FORALL:
                    return f"(forall [{var_str}], {body})"
                case Quantifier.EXISTS:
                    return f"(exists [{var_str}], {body})"

        # Fallback for other expressions
        return str(expr)


# Forward reference for SAT solver
class SatResult(Enum):
    SATISFIABLE = 1
    UNSATISFIABLE = 0
    UNKNOWN = -1


class SATInterface:
    """Base class for SAT solver interfaces (to avoid circular imports)"""
    def check_sat(self, expr: Any) -> SatResult:
        """Check satisfiability of expression"""
        raise NotImplementedError
