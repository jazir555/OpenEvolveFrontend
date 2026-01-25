"""
Expression AST for Ψ₃ Constraints

Defines the abstract syntax tree for logical expressions.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Set
from enum import Enum, auto


class Expr(ABC):
    """
    Base expression class.

    All expressions in Ψ₃ inherit from this abstract base class.
    Expressions are immutable and implement value semantics.
    """

    @abstractmethod
    def __str__(self) -> str:
        """String representation of expression"""

    @abstractmethod
    def __hash__(self) -> int:
        """Hash for caching and equality checks"""

    @abstractmethod
    def __eq__(self, other) -> bool:
        """Structural equality"""

    @abstractmethod
    def get_free_vars(self) -> Set[str]:
        """Get set of free variables in expression"""

    def substitute(self, var: str, replacement: 'Expr') -> 'Expr':
        """
        Substitute variable with expression.

        Args:
            var: Variable name to replace
            replacement: Expression to substitute

        Returns:
            New expression with substitution applied
        """
        raise NotImplementedError


class Variable(Expr):
    """
    Variable expression.

    Represents a free variable in constraints.
    """

    def __init__(self, name: str):
        if not name or not name.strip():
            raise ValueError("Variable name must be non-empty")
        self.name = name.strip()

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(("VAR", self.name))

    def __eq__(self, other) -> bool:
        return isinstance(other, Variable) and self.name == other.name

    def get_free_vars(self) -> Set[str]:
        return {self.name}

    def substitute(self, var: str, replacement: Expr) -> Expr:
        if self.name == var:
            return replacement
        return self


class Constant(Expr):
    """
    Constant expression.

    Represents a constant value (integer, float, string, etc.).
    """

    def __init__(self, value: Union[int, float, str, bool]):
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    def __hash__(self) -> int:
        return hash(("CONST", type(self.value), self.value))

    def __eq__(self, other) -> bool:
        return (isinstance(other, Constant) and
                type(self.value) == type(other.value) and
                self.value == other.value)

    def get_free_vars(self) -> Set[str]:
        return set()

    def substitute(self, var: str, replacement: Expr) -> Expr:
        return self


class BoolOp(Enum):
    """Boolean operators"""
    AND = auto()
    OR = auto()
    NOT = auto()
    IMPLIES = auto()
    IFF = auto()


class BoolExpr(Expr):
    """
    Boolean expression.

    Represents logical combinations of sub-expressions.
    """

    def __init__(self, op: BoolOp, args: List[Expr]):
        if op == BoolOp.NOT and len(args) != 1:
            raise ValueError("NOT operator requires exactly one argument")
        if op not in (BoolOp.NOT,) and len(args) < 2:
            raise ValueError(f"{op} operator requires at least two arguments")

        self.op = op
        self.args = args

    def __str__(self) -> str:
        match self.op:
            case BoolOp.AND:
                inner = ' ∧ '.join(str(a) for a in self.args)
                return f"({inner})"
            case BoolOp.OR:
                inner = ' ∨ '.join(str(a) for a in self.args)
                return f"({inner})"
            case BoolOp.NOT:
                return f"¬{self.args[0]}"
            case BoolOp.IMPLIES:
                return f"({self.args[0]} → {self.args[1]})"
            case BoolOp.IFF:
                return f"({self.args[0]} ↔ {self.args[1]})"

    def __hash__(self) -> int:
        return hash((self.op, tuple(self.args)))

    def __eq__(self, other) -> bool:
        return (isinstance(other, BoolExpr) and
                self.op == other.op and
                self.args == other.args)

    def get_free_vars(self) -> Set[str]:
        vars: Set[str] = set()
        for arg in self.args:
            vars.update(arg.get_free_vars())
        return vars

    def substitute(self, var: str, replacement: Expr) -> Expr:
        new_args = [arg.substitute(var, replacement) for arg in self.args]
        return BoolExpr(self.op, new_args)


class ArithOp(Enum):
    """Arithmetic/comparison operators"""
    LT = auto()   # Less than
    LE = auto()   # Less than or equal
    GT = auto()   # Greater than
    GE = auto()   # Greater than or equal
    EQ = auto()   # Equal
    NE = auto()   # Not equal


class ArithExpr(Expr):
    """
    Arithmetic expression.

    Represents comparisons and arithmetic operations.
    """

    def __init__(self, op: ArithOp, left: Expr, right: Expr):
        self.op = op
        self.left = left
        self.right = right

    def __str__(self) -> str:
        match self.op:
            case ArithOp.LT:
                return f"({self.left} < {self.right})"
            case ArithOp.LE:
                return f"({self.left} ≤ {self.right})"
            case ArithOp.GT:
                return f"({self.left} > {self.right})"
            case ArithOp.GE:
                return f"({self.left} ≥ {self.right})"
            case ArithOp.EQ:
                return f"({self.left} = {self.right})"
            case ArithOp.NE:
                return f"({self.left} ≠ {self.right})"

    def __hash__(self) -> int:
        return hash((self.op, self.left, self.right))

    def __eq__(self, other) -> bool:
        return (isinstance(other, ArithExpr) and
                self.op == other.op and
                self.left == other.left and
                self.right == other.right)

    def get_free_vars(self) -> Set[str]:
        vars: Set[str] = set()
        vars.update(self.left.get_free_vars())
        vars.update(self.right.get_free_vars())
        return vars

    def substitute(self, var: str, replacement: Expr) -> Expr:
        new_left = self.left.substitute(var, replacement)
        new_right = self.right.substitute(var, replacement)
        return ArithExpr(self.op, new_left, new_right)


class Quantifier(Enum):
    """Quantifiers"""
    FORALL = auto()   # Universal quantification
    EXISTS = auto()   # Existential quantification


class QuantExpr(Expr):
    """
    Quantified expression.

    Represents universally or existentially quantified formulas.
    """

    def __init__(self, quant: Quantifier, vars: List[str], body: Expr):
        if not vars:
            raise ValueError("Quantified expression must have at least one variable")

        self.quant = quant
        self.vars = vars
        self.body = body

        # Check that variables don't appear free in body (shadowing)
        body_vars = body.get_free_vars()
        for v in vars:
            if v not in body_vars:
                raise ValueError(f"Variable '{v}' does not appear in body")

    def __str__(self) -> str:
        var_str = ', '.join(self.vars)
        match self.quant:
            case Quantifier.FORALL:
                return f"∀{var_str}. {self.body}"
            case Quantifier.EXISTS:
                return f"∃{var_str}. {self.body}"

    def __hash__(self) -> int:
        return hash((self.quant, tuple(self.vars), self.body))

    def __eq__(self, other) -> bool:
        return (isinstance(other, QuantExpr) and
                self.quant == other.quant and
                self.vars == other.vars and
                self.body == other.body)

    def get_free_vars(self) -> Set[str]:
        # Quantified variables are bound, not free
        body_vars = self.body.get_free_vars()
        return body_vars - set(self.vars)

    def substitute(self, var: str, replacement: Expr) -> Expr:
        # Don't substitute bound variables
        if var in self.vars:
            return self

        # Substitute in body
        new_body = self.body.substitute(var, replacement)
        return QuantExpr(self.quant, self.vars, new_body)


# Convenience functions for building expressions

def Var(name: str) -> Variable:
    """Create a variable"""
    return Variable(name)


def Const(value: Union[int, float, str, bool]) -> Constant:
    """Create a constant"""
    return Constant(value)


def And(*args: Expr) -> BoolExpr:
    """Create conjunction"""
    return BoolExpr(BoolOp.AND, list(args))


def Or(*args: Expr) -> BoolExpr:
    """Create disjunction"""
    return BoolExpr(BoolOp.OR, list(args))


def Not(arg: Expr) -> BoolExpr:
    """Create negation"""
    return BoolExpr(BoolOp.NOT, [arg])


def Implies(left: Expr, right: Expr) -> BoolExpr:
    """Create implication"""
    return BoolExpr(BoolOp.IMPLIES, [left, right])


def Iff(left: Expr, right: Expr) -> BoolExpr:
    """Create equivalence"""
    return BoolExpr(BoolOp.IFF, [left, right])


def Lt(left: Expr, right: Expr) -> ArithExpr:
    """Create less-than comparison"""
    return ArithExpr(ArithOp.LT, left, right)


def Le(left: Expr, right: Expr) -> ArithExpr:
    """Create less-than-or-equal comparison"""
    return ArithExpr(ArithOp.LE, left, right)


def Gt(left: Expr, right: Expr) -> ArithExpr:
    """Create greater-than comparison"""
    return ArithExpr(ArithOp.GT, left, right)


def Ge(left: Expr, right: Expr) -> ArithExpr:
    """Create greater-than-or-equal comparison"""
    return ArithExpr(ArithOp.GE, left, right)


def Eq(left: Expr, right: Expr) -> ArithExpr:
    """Create equality comparison"""
    return ArithExpr(ArithOp.EQ, left, right)


def Ne(left: Expr, right: Expr) -> ArithExpr:
    """Create inequality comparison"""
    return ArithExpr(ArithOp.NE, left, right)


def Forall(vars: List[str], body: Expr) -> QuantExpr:
    """Create universal quantification"""
    return QuantExpr(Quantifier.FORALL, vars, body)


def Exists(vars: List[str], body: Expr) -> QuantExpr:
    """Create existential quantification"""
    return QuantExpr(Quantifier.EXISTS, vars, body)
