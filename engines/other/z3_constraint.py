"""z3_constraint - canonical Z3 constraint representation.

Flat-script module providing the shared ``Z3Constraint`` / ``Z3Config`` definitions
that the ``engines/`` decomposition, validation and quality-gate scripts expect via
``from z3_constraint import Z3Constraint, Z3Config``.

The real ``z3`` package is used when installed (``Z3Constraint.to_z3()`` builds a
genuine Z3 expression and ``check_sat()`` calls a real solver). When ``z3`` is
absent the module falls back to a self-contained constraint representation with
``parse``/``serialize`` plus a restricted AST evaluator, so downstream importers
still load and produce useful results.
"""

from __future__ import annotations

import ast
import json
import logging
import operator
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


try:  # pragma: no cover - depends on environment
    import z3  # type: ignore

    Z3_AVAILABLE = True
except Exception:  # noqa: BLE001 - optional dependency
    z3 = None  # type: ignore
    Z3_AVAILABLE = False


def is_z3_available() -> bool:
    """True when the real ``z3`` package can be used."""
    return Z3_AVAILABLE


class ConstraintKind(str, Enum):
    """Sort of the constraint / of its declared variables."""

    BOOLEAN = "boolean"
    INTEGER = "integer"
    REAL = "real"
    STRING = "string"
    BITVECTOR = "bitvector"


class SatStatus(str, Enum):
    """Outcome of a satisfiability check."""

    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Z3Config:
    """Solver configuration.

    Accepts ``timeout`` in seconds (the form used across the engines) and mirrors
    it into ``timeout_ms`` for the Z3 API, so either field may be supplied.
    """

    timeout: float = 5.0
    timeout_ms: Optional[int] = None
    random_seed: int = 0
    proof_generation: bool = False
    model_generation: bool = True
    unsat_core: bool = False
    # When True (default) use the internal evaluator if z3 is unavailable.
    allow_fallback: bool = True
    max_constraints: int = 10_000
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timeout_ms is None:
            self.timeout_ms = int(max(0.0, float(self.timeout)) * 1000)
        else:
            self.timeout_ms = int(self.timeout_ms)
            self.timeout = self.timeout_ms / 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeout": self.timeout,
            "timeout_ms": self.timeout_ms,
            "random_seed": self.random_seed,
            "proof_generation": self.proof_generation,
            "model_generation": self.model_generation,
            "unsat_core": self.unsat_core,
            "allow_fallback": self.allow_fallback,
            "z3_available": Z3_AVAILABLE,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Z3Config":
        known = {
            "timeout", "timeout_ms", "random_seed", "proof_generation",
            "model_generation", "unsat_core", "allow_fallback", "max_constraints",
        }
        kwargs = {k: v for k, v in (data or {}).items() if k in known}
        extra = {k: v for k, v in (data or {}).items() if k not in known}
        cfg = cls(**kwargs)
        cfg.extra.update(extra)
        return cfg

    def apply_to_solver(self, solver: Any) -> Any:  # pragma: no cover - needs z3
        """Push this configuration onto a live ``z3.Solver``."""
        if not Z3_AVAILABLE or solver is None:
            return solver
        try:
            solver.set("timeout", int(self.timeout_ms or 0))
            solver.set("random_seed", int(self.random_seed))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not apply Z3 config: %s", exc)
        return solver


# ---------------------------------------------------------------------------
# Restricted evaluator used by the no-z3 fallback
# ---------------------------------------------------------------------------
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_VAR_RE = re.compile(r"\b(?!and\b|or\b|not\b|True\b|False\b)[A-Za-z_][A-Za-z_0-9]*\b")


class ConstraintSyntaxError(ValueError):
    """Raised when an expression cannot be parsed."""


def _eval_node(node: ast.AST, env: Dict[str, Any]) -> Any:
    """Evaluate a whitelisted AST node against ``env``."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ConstraintSyntaxError(f"unbound variable: {node.id}")
    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand, env)
        if isinstance(node.op, ast.Not):
            return not val
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return +val
        raise ConstraintSyntaxError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        fn = _BIN_OPS.get(type(node.op))
        if fn is None:
            raise ConstraintSyntaxError("unsupported binary operator")
        return fn(_eval_node(node.left, env), _eval_node(node.right, env))
    if isinstance(node, ast.BoolOp):
        vals = [_eval_node(v, env) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(vals)
        return any(vals)
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env)
        for op, comparator in zip(node.ops, node.comparators):
            fn = _CMP_OPS.get(type(op))
            if fn is None:
                raise ConstraintSyntaxError("unsupported comparison")
            right = _eval_node(comparator, env)
            if not fn(left, right):
                return False
            left = right
        return True
    raise ConstraintSyntaxError(f"unsupported expression node: {type(node).__name__}")


def _normalize(expression: str) -> str:
    """Normalize common SMT-ish / Lean-ish syntax into Python-evaluable text."""
    expr = str(expression).strip()
    # Strip a single wrapping pair of SMT parens: "(not x)" -> "not x"
    replacements = [
        ("&&", " and "), ("||", " or "), ("!=", "__NE__"),
        ("¬", " not "), ("∧", " and "), ("∨", " or "),
        ("<=", "__LE__"), (">=", "__GE__"), ("==", "__EQ__"),
    ]
    for old, new in replacements:
        expr = expr.replace(old, new)
    # A single '=' that is not part of a compound operator becomes '=='.
    expr = re.sub(r"(?<![<>=!_])=(?!=)", "__EQ__", expr)
    for token, real in (
        ("__NE__", "!="), ("__LE__", "<="), ("__GE__", ">="), ("__EQ__", "=="),
    ):
        expr = expr.replace(token, real)
    expr = re.sub(r"\btrue\b", "True", expr)
    expr = re.sub(r"\bfalse\b", "False", expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    return expr


# ---------------------------------------------------------------------------
# Variable
# ---------------------------------------------------------------------------
@dataclass
class Z3Variable:
    """A declared variable with an optional numeric domain."""

    name: str
    kind: ConstraintKind = ConstraintKind.INTEGER
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    def to_z3(self) -> Any:  # pragma: no cover - needs z3
        if not Z3_AVAILABLE:
            return None
        if self.kind is ConstraintKind.BOOLEAN:
            return z3.Bool(self.name)
        if self.kind is ConstraintKind.REAL:
            return z3.Real(self.name)
        if self.kind is ConstraintKind.STRING:
            return z3.String(self.name)
        if self.kind is ConstraintKind.BITVECTOR:
            return z3.BitVec(self.name, 32)
        return z3.Int(self.name)

    def bound_terms(self, term: Any = None) -> List[Any]:  # pragma: no cover - needs z3
        """Z3 assertions expressing this variable's declared numeric domain."""
        if not Z3_AVAILABLE or self.kind in (ConstraintKind.BOOLEAN, ConstraintKind.STRING):
            return []
        var = term if term is not None else self.to_z3()
        if var is None:
            return []
        terms = []
        if self.lower_bound is not None:
            terms.append(var >= self.lower_bound)
        if self.upper_bound is not None:
            terms.append(var <= self.upper_bound)
        return terms

    def default_value(self) -> Any:
        """A concrete value inside the declared domain (fallback search seed)."""
        if self.kind is ConstraintKind.BOOLEAN:
            return False
        if self.kind is ConstraintKind.STRING:
            return ""
        lo = self.lower_bound if self.lower_bound is not None else 0
        if self.upper_bound is not None and lo > self.upper_bound:
            lo = self.upper_bound
        return int(lo) if self.kind is ConstraintKind.INTEGER else float(lo)

    def candidates(self, limit: int = 12) -> List[Any]:
        """Small deterministic candidate set used by the fallback search."""
        if self.kind is ConstraintKind.BOOLEAN:
            return [False, True]
        if self.kind is ConstraintKind.STRING:
            return ["", "a"]
        lo = int(self.lower_bound) if self.lower_bound is not None else 0
        hi = int(self.upper_bound) if self.upper_bound is not None else lo + limit - 1
        if hi < lo:
            lo, hi = hi, lo
        span = list(range(lo, min(hi, lo + limit - 1) + 1))
        return span or [lo]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }


# ---------------------------------------------------------------------------
# Constraint
# ---------------------------------------------------------------------------
@dataclass
class Z3Constraint:
    """A single logical constraint.

    ``expression`` is stored verbatim; ``normalized`` holds the Python-evaluable
    form used by the fallback evaluator. ``to_z3()`` produces a real Z3 term when
    the package is installed.
    """

    expression: str
    kind: ConstraintKind = ConstraintKind.BOOLEAN
    variables: List[Z3Variable] = field(default_factory=list)
    label: Optional[str] = None
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.expression = str(self.expression).strip()
        if isinstance(self.kind, str):
            try:
                self.kind = ConstraintKind(self.kind.lower())
            except ValueError:
                self.kind = ConstraintKind.BOOLEAN
        # Allow bare variable names to be passed in.
        self.variables = [
            v if isinstance(v, Z3Variable) else Z3Variable(name=str(v))
            for v in self.variables
        ]
        if not self.variables:
            self.variables = [
                Z3Variable(name=n, kind=self._infer_kind())
                for n in self.free_variables()
            ]

    def _infer_kind(self) -> ConstraintKind:
        return ConstraintKind.INTEGER if self.kind is ConstraintKind.BOOLEAN else self.kind

    # -- parsing ---------------------------------------------------------
    @property
    def normalized(self) -> str:
        """The expression rewritten into Python-evaluable syntax."""
        return _normalize(self.expression)

    def free_variables(self) -> List[str]:
        """Identifiers appearing in the expression, in first-seen order."""
        seen: List[str] = []
        for match in _VAR_RE.finditer(self.normalized):
            name = match.group(0)
            if name not in seen:
                seen.append(name)
        return seen

    @classmethod
    def parse(
        cls,
        text: str,
        kind: ConstraintKind = ConstraintKind.BOOLEAN,
        label: Optional[str] = None,
    ) -> "Z3Constraint":
        """Build a constraint from ``text``, validating that it is parseable."""
        constraint = cls(expression=text, kind=kind, label=label)
        constraint.validate()
        return constraint

    def validate(self) -> bool:
        """Raise :class:`ConstraintSyntaxError` when the expression is malformed."""
        if not self.expression:
            raise ConstraintSyntaxError("empty constraint expression")
        try:
            ast.parse(self.normalized, mode="eval")
        except SyntaxError as exc:
            raise ConstraintSyntaxError(
                f"cannot parse constraint {self.expression!r}: {exc}"
            ) from exc
        return True

    def is_valid(self) -> bool:
        """Non-raising form of :meth:`validate`."""
        try:
            return self.validate()
        except ConstraintSyntaxError:
            return False

    # -- evaluation ------------------------------------------------------
    def evaluate(self, assignment: Optional[Dict[str, Any]] = None) -> bool:
        """Evaluate the constraint under ``assignment``.

        Missing variables fall back to their declared default, so evaluation is
        always total and deterministic.
        """
        env: Dict[str, Any] = {v.name: v.default_value() for v in self.variables}
        env.update(assignment or {})
        tree = ast.parse(self.normalized, mode="eval")
        return bool(_eval_node(tree, env))

    def to_z3(self, declarations: Optional[Dict[str, Any]] = None) -> Any:
        """Build the corresponding Z3 term, or ``None`` without ``z3``."""
        if not Z3_AVAILABLE:  # pragma: no cover - exercised without z3
            return None
        decls = dict(declarations or {})
        for var in self.variables:
            decls.setdefault(var.name, var.to_z3())
        try:  # pragma: no cover - needs z3
            return eval(  # noqa: S307 - restricted, internal expressions only
                self.normalized,
                {"__builtins__": {}, "And": z3.And, "Or": z3.Or, "Not": z3.Not},
                decls,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not build z3 term for %r: %s", self.expression, exc)
            return None

    # -- serialization ---------------------------------------------------
    def serialize(self) -> str:
        """Serialize to a compact, stable JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def deserialize(cls, payload: str) -> "Z3Constraint":
        return cls.from_dict(json.loads(payload))

    def to_smt2(self) -> str:
        """Render as an SMT-LIB2 assertion (best effort, prefix form)."""
        expr = self.normalized
        for py, smt in ((" and ", " "), (" or ", " ")):
            expr = expr.replace(py, smt)
        return f"(assert {self.expression})" if "(" in self.expression else f"(assert {expr})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expression": self.expression,
            "kind": self.kind.value,
            "variables": [v.to_dict() for v in self.variables],
            "label": self.label,
            "weight": self.weight,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Z3Constraint":
        return cls(
            expression=data.get("expression", ""),
            kind=data.get("kind", ConstraintKind.BOOLEAN),
            variables=[
                Z3Variable(
                    name=v.get("name", ""),
                    kind=ConstraintKind(v.get("kind", "integer")),
                    lower_bound=v.get("lower_bound"),
                    upper_bound=v.get("upper_bound"),
                )
                for v in (data.get("variables") or [])
                if isinstance(v, dict)
            ],
            label=data.get("label"),
            weight=float(data.get("weight", 1.0)),
            metadata=dict(data.get("metadata") or {}),
        )

    def __str__(self) -> str:  # pragma: no cover - trivial
        prefix = f"{self.label}: " if self.label else ""
        return f"{prefix}{self.expression}"


# ---------------------------------------------------------------------------
# Convenience solving helper
# ---------------------------------------------------------------------------
def check_sat(
    constraints: Iterable[Z3Constraint],
    config: Optional[Z3Config] = None,
) -> Tuple[SatStatus, Dict[str, Any]]:
    """Check a conjunction of constraints.

    Uses real Z3 when available; otherwise performs a bounded, deterministic
    search over each variable's candidate domain. Returns ``(status, model)``.
    """
    cfg = config or Z3Config()
    items = [c for c in constraints if c is not None]
    if not items:
        return SatStatus.SAT, {}

    if Z3_AVAILABLE:  # pragma: no cover - needs z3
        try:
            solver = cfg.apply_to_solver(z3.Solver())
            decls: Dict[str, Any] = {}
            bounded: set = set()
            for c in items:
                for v in c.variables:
                    decls.setdefault(v.name, v.to_z3())
                    # Enforce each variable's declared domain exactly once.
                    if v.name not in bounded:
                        bounded.add(v.name)
                        for term in v.bound_terms(decls[v.name]):
                            solver.add(term)
            terms = []
            unconvertible = False
            for c in items:
                term = c.to_z3(decls)
                if term is None:
                    unconvertible = True
                    break
                terms.append(term)
            # Only answer from Z3 when the whole conjunction was expressible;
            # a partial conjunction could report a false "sat".
            if not unconvertible and terms:
                for term in terms:
                    solver.add(term)
                res = solver.check()
                if res == z3.sat:
                    model = solver.model()
                    return SatStatus.SAT, {
                        d.name(): str(model[d]) for d in model.decls()
                    }
                if res == z3.unsat:
                    return SatStatus.UNSAT, {}
                return SatStatus.UNKNOWN, {}
            logger.debug("Not all constraints expressible in z3; using fallback")
        except Exception as exc:  # noqa: BLE001
            logger.warning("z3 check failed, using fallback: %s", exc)

    if not cfg.allow_fallback:
        return SatStatus.UNKNOWN, {}
    return _check_sat_fallback(items, cfg)


def _check_sat_fallback(
    items: List[Z3Constraint], cfg: Z3Config
) -> Tuple[SatStatus, Dict[str, Any]]:
    """Bounded product search over declared variable candidates."""
    import itertools

    variables: Dict[str, Z3Variable] = {}
    for c in items:
        for v in c.variables:
            variables.setdefault(v.name, v)
    names = sorted(variables)
    if not names:
        try:
            ok = all(c.evaluate({}) for c in items)
        except (ConstraintSyntaxError, Exception):  # noqa: BLE001
            return SatStatus.UNKNOWN, {}
        return (SatStatus.SAT if ok else SatStatus.UNSAT), {}

    domains = [variables[n].candidates() for n in names]
    budget = 20_000
    deadline = time.time() + max(0.05, float(cfg.timeout))
    combos = itertools.product(*domains)
    tried = 0
    for combo in combos:
        tried += 1
        if tried > budget or time.time() > deadline:
            return SatStatus.UNKNOWN, {}
        assignment = dict(zip(names, combo))
        try:
            if all(c.evaluate(assignment) for c in items):
                return SatStatus.SAT, assignment
        except ConstraintSyntaxError:
            return SatStatus.UNKNOWN, {}
        except Exception:  # noqa: BLE001 - e.g. division by zero in a candidate
            continue
    return SatStatus.UNSAT, {}


__all__ = [
    "Z3Constraint",
    "Z3Config",
    "Z3Variable",
    "ConstraintKind",
    "SatStatus",
    "ConstraintSyntaxError",
    "check_sat",
    "is_z3_available",
    "Z3_AVAILABLE",
]
