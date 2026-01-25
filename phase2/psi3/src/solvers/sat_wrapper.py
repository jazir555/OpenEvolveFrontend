"""
SAT Solver Wrapper for Ψ₃

Provides interface to Z3 SMT solver for implication checking and satisfiability.
"""

from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import subprocess
import json

try:
    from z3 import *
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    print("[WARNING] Z3 not available. Install with: pip install z3-solver")


from ..core.expression import Expr, BoolExpr, ArithExpr, QuantExpr, BoolOp, ArithOp, Quantifier, Variable, Constant


class SatResult(Enum):
    """Result of SAT solver query"""
    SATISFIABLE = 1
    UNSATISFIABLE = 0
    UNKNOWN = -1


class SATInterface:
    """
    Interface to SAT/SMT solvers.

    Primary backend is Z3, with fallback to other solvers possible.
    """

    def __init__(self, solver_type: str = "z3", timeout: float = 10.0):
        """
        Initialize SAT solver interface.

        Args:
            solver_type: Type of solver ("z3", "kissat", "cadical")
            timeout: Timeout per query in seconds
        """
        if not Z3_AVAILABLE and solver_type == "z3":
            raise ImportError(
                "Z3 is not installed. Install with: pip install z3-solver"
            )

        self.solver_type = solver_type
        self.timeout = timeout
        self.solver = self._init_solver()
        self._cache: Dict[str, SatResult] = {}

    def _init_solver(self):
        """Initialize solver backend"""
        if self.solver_type == "z3":
            solver = Solver()
            solver.set("timeout", int(self.timeout * 1000))
            return solver
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")

    def check_implication(
        self,
        antecedent: Expr,
        consequent: Expr
    ) -> bool:
        """
        Check if antecedent ⊨ consequent (antecedent implies consequent).

        Method: Check UNSAT(antecedent ∧ ¬consequent)

        Args:
            antecedent: Antecedent expression
            consequent: Consequent expression

        Returns:
            True if implication holds, False otherwise
        """
        # Build formula: antecedent ∧ ¬consequent
        negation = And(
            self._expr_to_z3(antecedent),
            Not(self._expr_to_z3(consequent))
        )

        # Check if unsatisfiable
        result = self._check_sat_z3(negation)
        return result == SatResult.UNSATISFIABLE

    def check_equivalence(self, expr1: Expr, expr2: Expr) -> bool:
        """
        Check if expr1 ≡ expr2 (mutual implication).

        Args:
            expr1: First expression
            expr2: Second expression

        Returns:
            True if equivalent, False otherwise
        """
        return (self.check_implication(expr1, expr2) and
                self.check_implication(expr2, expr1))

    def check_sat(self, expr: Expr) -> SatResult:
        """
        Check satisfiability of expression.

        Args:
            expr: Expression to check

        Returns:
            SatResult indicating satisfiability
        """
        # Check cache
        cache_key = str(expr)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Convert to Z3 and check
        z3_expr = self._expr_to_z3(expr)
        result = self._check_sat_z3(z3_expr)

        # Cache result
        self._cache[cache_key] = result
        return result

    def find_model(self, constraints: List[Expr]) -> Optional[Dict[str, Any]]:
        """
        Find satisfying assignment for constraints.

        Args:
            constraints: List of constraints to satisfy

        Returns:
            Dictionary mapping variables to values if satisfiable, None otherwise
        """
        if self.solver_type != "z3":
            raise NotImplementedError("Model finding only supported for Z3")

        self.solver.push()

        # Add constraints
        for c in constraints:
            self.solver.add(self._expr_to_z3(c))

        # Check satisfiability
        result = self.solver.check()

        if result == sat:
            model = self.solver.model()
            assignment = self._extract_assignment(model)
            self.solver.pop()
            return assignment
        else:
            self.solver.pop()
            return None

    def _check_sat_z3(self, z3_expr) -> SatResult:
        """
        Check Z3 expression satisfiability.

        Args:
            z3_expr: Z3 expression

        Returns:
            SatResult
        """
        self.solver.push()
        self.solver.add(z3_expr)
        result = self.solver.check()
        self.solver.pop()

        if result == sat:
            return SatResult.SATISFIABLE
        elif result == unsat:
            return SatResult.UNSATISFIABLE
        else:
            return SatResult.UNKNOWN

    def _expr_to_z3(self, expr: Expr):
        """
        Convert internal Expr to Z3 expression.

        Args:
            expr: Internal expression

        Returns:
            Z3 expression
        """
        if not Z3_AVAILABLE:
            raise RuntimeError("Z3 not available")

        if isinstance(expr, BoolExpr):
            match expr.op:
                case BoolOp.AND:
                    return And(*[self._expr_to_z3(a) for a in expr.args])
                case BoolOp.OR:
                    return Or(*[self._expr_to_z3(a) for a in expr.args])
                case BoolOp.NOT:
                    return Not(self._expr_to_z3(expr.args[0]))
                case BoolOp.IMPLIES:
                    return Implies(
                        self._expr_to_z3(expr.args[0]),
                        self._expr_to_z3(expr.args[1])
                    )
                case BoolOp.IFF:
                    # X ↔ Y = (X → Y) ∧ (Y → X)
                    left = self._expr_to_z3(expr.args[0])
                    right = self._expr_to_z3(expr.args[1])
                    return And(Implies(left, right), Implies(right, left))

        elif isinstance(expr, ArithExpr):
            left = self._expr_to_z3(expr.left)
            right = self._expr_to_z3(expr.right)
            match expr.op:
                case ArithOp.LT:
                    return left < right
                case ArithOp.LE:
                    return left <= right
                case ArithOp.GT:
                    return left > right
                case ArithOp.GE:
                    return left >= right
                case ArithOp.EQ:
                    return left == right
                case ArithOp.NE:
                    return left != right

        elif isinstance(expr, QuantExpr):
            # Convert quantified variables to Z3 constants
            vars = []
            for var_name in expr.vars:
                # Create Z3 variable (default to Int for arithmetic)
                z3_var = Int(var_name)
                vars.append(z3_var)

            # Convert body with variable substitution
            body_z3 = self._expr_to_z3(expr.body)

            # Apply quantification
            match expr.quant:
                case Quantifier.FORALL:
                    return ForAll(vars, body_z3)
                case Quantifier.EXISTS:
                    return Exists(vars, body_z3)

        elif isinstance(expr, Variable):
            # Return Z3 variable (default to Int)
            return Int(expr.name)

        elif isinstance(expr, Constant):
            # Return Z3 constant
            if isinstance(expr.value, bool):
                return BoolVal(expr.value)
            elif isinstance(expr.value, int):
                return IntVal(expr.value)
            elif isinstance(expr.value, float):
                return RealVal(expr.value)
            elif isinstance(expr.value, str):
                return StringVal(expr.value)
            else:
                raise ValueError(f"Unsupported constant type: {type(expr.value)}")

        else:
            raise ValueError(f"Unsupported expression type: {type(expr)}")

    def _extract_assignment(self, model: ModelRef) -> Dict[str, Any]:
        """
        Extract variable assignment from Z3 model.

        Args:
            model: Z3 model

        Returns:
            Dictionary mapping variable names to values
        """
        assignment = {}
        for decl in model:
            name = decl.name()
            value = model[decl]

            # Convert Z3 value to Python value
            if isinstance(value, BoolRef):
                assignment[name] = bool(value)
            elif isinstance(value, IntNumRef):
                assignment[name] = value.as_long()
            elif isinstance(value, RatNumRef):
                assignment[name] = float(value.numerator.as_long() /
                                       value.denominator.as_long())
            else:
                assignment[name] = str(value)

        return assignment

    def clear_cache(self):
        """Clear satisfiability cache"""
        self._cache.clear()

    def reset(self):
        """Reset solver state"""
        self.solver.reset()
        self.clear_cache()


# Convenience functions

def check_implication_batch(
    implications: List[Tuple[Expr, Expr]],
    timeout: float = 10.0
) -> List[bool]:
    """
    Check multiple implications efficiently.

    Args:
        implications: List of (antecedent, consequent) pairs
        timeout: Timeout per query

    Returns:
        List of bool results (True if implication holds)
    """
    solver = SATInterface(timeout=timeout)
    results = []

    for antecedent, consequent in implications:
        result = solver.check_implication(antecedent, consequent)
        results.append(result)

    return results


def find_counterexample(
    antecedent: Expr,
    consequent: Expr,
    timeout: float = 10.0
) -> Optional[Dict[str, Any]]:
    """
    Find counterexample to implication antecedent ⊨ consequent.

    Returns model where antecedent is true but consequent is false.

    Args:
        antecedent: Antecedent expression
        consequent: Consequent expression
        timeout: Timeout in seconds

    Returns:
        Assignment if counterexample exists, None otherwise
    """
    from ..core.expression import Not, And

    # Build formula: antecedent ∧ ¬consequent
    negation = And(antecedent, Not(consequent))

    solver = SATInterface(timeout=timeout)
    model = solver.find_model([negation])

    return model  # None if unsatisfiable (implication holds)
