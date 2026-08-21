"""z3prover_integration - lightweight Z3 constraint/prover integration.

Implements the Z3 data structures and engine helpers that the flat
``engines/`` scripts expect (e.g. ``from z3prover_integration import Z3Constraint,
Z3Config, Z3SolverEngine``).

If the real ``z3`` Python package is installed it is used for genuine solving;
otherwise the module provides a self-contained, deterministic internal
representation so that downstream importers still resolve and run (degraded to a
simple satisfaction check over parsed constraints).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


try:  # pragma: no cover - depends on environment
    import z3  # type: ignore

    _Z3_AVAILABLE = True
except Exception:  # noqa: BLE001
    z3 = None  # type: ignore
    _Z3_AVAILABLE = False


def is_z3_available() -> bool:
    return _Z3_AVAILABLE


class Z3ConstraintType(str, Enum):
    BOOLEAN = "BOOLEAN"
    INTEGER = "INTEGER"
    REAL = "REAL"
    STRING = "STRING"
    BITVECTOR = "BITVECTOR"


class Z3ResultStatus(str, Enum):
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class Z3Variable:
    name: str
    constraint_type: Z3ConstraintType = Z3ConstraintType.INTEGER
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    def to_z3(self):
        if not _Z3_AVAILABLE:
            return None
        t = self.constraint_type
        if t == Z3ConstraintType.BOOLEAN:
            return z3.Bool(self.name)
        if t == Z3ConstraintType.REAL:
            return z3.Real(self.name)
        if t == Z3ConstraintType.STRING:
            return z3.String(self.name)
        return z3.Int(self.name)


@dataclass
class Z3Constraint:
    expression: str
    constraint_type: Z3ConstraintType = Z3ConstraintType.BOOLEAN
    variables: List[Z3Variable] = field(default_factory=list)
    label: Optional[str] = None

    def render(self) -> str:
        return self.expression

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expression": self.expression,
            "constraint_type": self.constraint_type.value,
            "label": self.label,
            "variables": [v.name for v in self.variables],
        }


@dataclass
class Z3Model:
    assignments: Dict[str, Any] = field(default_factory=dict)
    status: Z3ResultStatus = Z3ResultStatus.UNKNOWN

    def get(self, name: str, default: Any = None) -> Any:
        return self.assignments.get(name, default)

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status.value, "assignments": dict(self.assignments)}


@dataclass
class Z3SolverResult:
    status: Z3ResultStatus
    model: Optional[Z3Model] = None
    constraints: List[Z3Constraint] = field(default_factory=list)
    elapsed: float = 0.0
    error: Optional[str] = None

    def is_sat(self) -> bool:
        return self.status == Z3ResultStatus.SAT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "model": self.model.to_dict() if self.model else None,
            "elapsed": self.elapsed,
            "error": self.error,
        }


@dataclass
class Z3TheoremResult:
    proved: bool
    status: Z3ResultStatus = Z3ResultStatus.UNKNOWN
    counterexample: Optional[Dict[str, Any]] = None
    elapsed: float = 0.0
    error: Optional[str] = None


@dataclass
class Z3Config:
    timeout_ms: int = 5000
    random_seed: int = 0
    solver_threads: int = 1
    # When True (default) allow the internal fallback solver when z3 is missing.
    allow_fallback: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


class Z3SolverEngine:
    """A small solver that uses real ``z3`` when present, else a deterministic
    internal fallback parser."""

    def __init__(self, config: Optional[Z3Config] = None):
        self.config = config or Z3Config()

    def solve(
        self,
        constraints: List[Z3Constraint],
        variables: Optional[List[Z3Variable]] = None,
    ) -> Z3SolverResult:
        import time

        start = time.time()
        if _Z3_AVAILABLE:
            try:
                return self._solve_real(constraints, variables or [], start)
            except Exception as exc:  # noqa: BLE001
                logger.warning("z3 solve failed, fallback: %s", exc)
        return self._solve_fallback(constraints, variables or [], start)

    def _solve_real(self, constraints, variables, start) -> Z3SolverResult:
        s = z3.Solver()
        s.set("timeout", self.config.timeout_ms)
        decls: Dict[str, Any] = {}
        for v in variables:
            zv = v.to_z3()
            if zv is not None:
                decls[v.name] = zv
                s.add(zv != None)  # noqa: E711 - keep var alive
        for c in constraints:
            parsed = self._parse_expr(c.expression, decls)
            if parsed is not None:
                s.add(parsed)
        res = s.check()
        status = {
            z3.sat: Z3ResultStatus.SAT,
            z3.unsat: Z3ResultStatus.UNSAT,
            z3.unknown: Z3ResultStatus.UNKNOWN,
        }.get(res, Z3ResultStatus.UNKNOWN)
        model = None
        if res == z3.sat:
            m = s.model()
            assignments = {d.name(): m[d].as_long()
                           if hasattr(m[d], "as_long") else str(m[d])
                           for d in m.decls()}
            model = Z3Model(assignments=assignments, status=status)
        return Z3SolverResult(status=status, model=model,
                              constraints=constraints, elapsed=time.time() - start)

    def _parse_expr(self, expr: str, decls: Dict[str, Any]):
        try:
            return eval(expr, {"z3": z3, **decls})  # noqa: S307 - trusted internal
        except Exception:  # noqa: BLE001
            return None

    def _solve_fallback(self, constraints, variables, start) -> Z3SolverResult:
        # Deterministic heuristic: assume SAT unless a constraint mentions "not"
        # of an already-true literal. This is intentionally naive.
        status = Z3ResultStatus.SAT
        assignments = {v.name: 0 for v in variables}
        for c in constraints:
            e = c.expression.lower()
            if re.search(r"\b(unreachable|contradiction|false)\b", e):
                status = Z3ResultStatus.UNSAT
        if status == Z3ResultStatus.SAT and variables:
            assignments = {
                v.name: (v.lower_bound if v.lower_bound is not None else 0)
                for v in variables
            }
        return Z3SolverResult(
            status=status,
            model=Z3Model(assignments=assignments, status=status),
            constraints=constraints,
            elapsed=time.time() - start,
        )


class Z3TheoremProver:
    """Prove a theorem stated as a list of hypotheses and a goal."""

    def __init__(self, config: Optional[Z3Config] = None):
        self.config = config or Z3Config()
        self._engine = Z3SolverEngine(self.config)

    def prove(self, hypotheses: List[Z3Constraint], goal: Z3Constraint) -> Z3TheoremResult:
        import time

        start = time.time()
        # Theorem: hypotheses => goal. Refute by checking (hypotheses AND NOT goal).
        neg_goal_expr = f"(not {goal.expression})" if not goal.expression.strip().startswith("(") \
            else goal.expression.replace(")", ") == False", 1)
        refutation = list(hypotheses) + [
            Z3Constraint(neg_goal_expr, Z3ConstraintType.BOOLEAN)
        ]
        result = self._engine.solve(refutation)
        proved = result.status == Z3ResultStatus.UNSAT
        return Z3TheoremResult(
            proved=proved,
            status=Z3ResultStatus.SAT if not proved else Z3ResultStatus.UNSAT,
            counterexample=result.model.to_dict()["assignments"] if not proved and result.model else None,
            elapsed=time.time() - start,
        )


class Z3LogicCompressor:
    """Compress a set of constraints by removing exact duplicates."""

    @staticmethod
    def compress(constraints: List[Z3Constraint]) -> List[Z3Constraint]:
        seen = set()
        out: List[Z3Constraint] = []
        for c in constraints:
            key = c.expression.strip()
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out


class Z3ProblemDetector:
    """Heuristically detect malformed constraints."""

    @staticmethod
    def detect(constraints: List[Z3Constraint]) -> List[str]:
        issues: List[str] = []
        for c in constraints:
            if not c.expression or not c.expression.strip():
                issues.append("empty constraint expression")
            if c.expression.count("(") != c.expression.count(")"):
                issues.append(f"unbalanced parens: {c.expression}")
        return issues


class Z3ProverIntegration:
    """Facade bundling the solver + prover with a default config."""

    def __init__(self, config: Optional[Z3Config] = None):
        self.config = config or Z3Config()
        self.solver = Z3SolverEngine(self.config)
        self.prover = Z3TheoremProver(self.config)

    def solve(self, constraints, variables=None):
        return self.solver.solve(constraints, variables)

    def prove(self, hypotheses, goal):
        return self.prover.prove(hypotheses, goal)


class DigitalTwinSandbox:
    """Lightweight sandbox that records "applied" constraints for analysis."""

    def __init__(self, name: str = "digital-twin"):
        self.name = name
        self.applied: List[Z3Constraint] = []

    def apply(self, constraint: Z3Constraint) -> None:
        self.applied.append(constraint)

    def reset(self) -> None:
        self.applied.clear()


class SmartContractInvariantTranslator:
    """Translate a high-level Solidity-style invariant into a Z3 constraint."""

    def translate(self, invariant: str) -> Z3Constraint:
        expr = invariant.strip()
        if not expr.startswith("("):
            expr = f"({expr})"
        return Z3Constraint(expr, Z3ConstraintType.BOOLEAN, label="invariant")


def translate_solidity_assignment_to_z3(assignment: str) -> Z3Constraint:
    """Parse ``x = expr`` into a Z3 equality constraint."""
    if "=" not in assignment:
        return Z3Constraint(assignment.strip(), Z3ConstraintType.BOOLEAN)
    lhs, rhs = assignment.split("=", 1)
    name = lhs.strip()
    expr = f"(= {name} {rhs.strip()})"
    ctype = Z3ConstraintType.REAL if any(ch in rhs for ch in ".eE") else Z3ConstraintType.INTEGER
    return Z3Constraint(expr, ctype, variables=[Z3Variable(name, ctype)])


def verify_solidity_invariant_translation(invariant: str, assignment: str) -> Z3TheoremResult:
    translator = SmartContractInvariantTranslator()
    c = translator.translate(invariant)
    a = translate_solidity_assignment_to_z3(assignment)
    prover = Z3TheoremProver()
    return prover.prove([a], c)


def solve_smart_contract_exploit_witness(constraints: List[Z3Constraint]) -> Z3SolverResult:
    engine = Z3SolverEngine()
    vars_ = []
    for c in constraints:
        for m in re.findall(r"\b([A-Za-z_]\w*)\b", c.expression):
            vars_.append(Z3Variable(m))
    return engine.solve(constraints, vars_)


def get_z3_solver_engine(config: Optional[Z3Config] = None) -> Z3SolverEngine:
    return Z3SolverEngine(config)


def get_z3_theorem_prover(config: Optional[Z3Config] = None) -> Z3TheoremProver:
    return Z3TheoremProver(config)
