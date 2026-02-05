"""
RESE-Z3 Bridge Canonical Schema

Defines the canonical data models for Z3 interactions following the
Anti-Corruption Layer pattern from CLAUDE.md.

This schema provides:
1. Unified interface for all RESE phases
2. Transformation between canonical and Z3 formats
3. Validation and type safety
4. Serialization/deserialization

Author: RESE Team
Created: 2026-02-04
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union


# =============================================================================
# ENUMS
# =============================================================================

class Z3ResultStatus(Enum):
    """Z3 solver result status"""
    SAT = "sat"           # Satisfiable
    UNSAT = "unsat"       # Unsatisfiable
    UNKNOWN = "unknown"   # Unknown
    ERROR = "error"       # Error occurred
    TIMEOUT = "timeout"   # Timeout


class ConstraintType(Enum):
    """Types of constraints"""
    BOOLEAN = "boolean"
    INTEGER = "integer"
    REAL = "real"
    BIT_VECTOR = "bit_vector"
    ARRAY = "array"
    STRING = "string"


class ProblemType(Enum):
    """Types of problems for Z3"""
    CONSTRAINT_SAT = "constraint_sat"      # Constraint satisfaction
    OPTIMIZATION = "optimization"          # Optimization problem
    THEOREM_PROVING = "theorem_proving"    # Theorem proving
    CONTRADICTION_DETECTION = "contradiction_detection"  # Detect contradictions


# =============================================================================
# CANONICAL SOLVER REQUEST
# =============================================================================

@dataclass
class CanonicalVariable:
    """Variable definition in canonical format"""
    name: str
    var_type: ConstraintType
    bounds: Optional[tuple[Optional[float], Optional[float]]] = None
    bit_width: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "var_type": self.var_type.value,
            "bounds": self.bounds,
            "bit_width": self.bit_width,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CanonicalVariable':
        return cls(
            name=data["name"],
            var_type=ConstraintType(data["var_type"]),
            bounds=data.get("bounds"),
            bit_width=data.get("bit_width"),
        )


@dataclass
class CanonicalConstraint:
    """Constraint definition in canonical format"""
    expression: str
    constraint_type: ConstraintType
    description: Optional[str] = None
    constraint_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expression": self.expression,
            "constraint_type": self.constraint_type.value,
            "description": self.description,
            "constraint_id": self.constraint_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CanonicalConstraint':
        return cls(
            expression=data["expression"],
            constraint_type=ConstraintType(data["constraint_type"]),
            description=data.get("description"),
            constraint_id=data.get("constraint_id"),
        )


@dataclass
class CanonicalSolverRequest:
    """
    Canonical solver request format

    Law of Configuration Explicitness: timeout_ms is mandatory
    Law of UTC: timestamp is UTC ISO-8601
    """
    problem: str  # SMT-LIB2 or natural language description
    problem_type: ProblemType
    variables: List[CanonicalVariable] = field(default_factory=list)
    constraints: List[CanonicalConstraint] = field(default_factory=list)
    timeout_ms: int = 30000  # MANDATORY (Law of Configuration Explicitness)
    tactics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "problem_type": self.problem_type.value,
            "variables": [v.to_dict() for v in self.variables],
            "constraints": [c.to_dict() for c in self.constraints],
            "timeout_ms": self.timeout_ms,
            "tactics": self.tactics,
            "metadata": self.metadata,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CanonicalSolverRequest':
        return cls(
            problem=data["problem"],
            problem_type=ProblemType(data["problem_type"]),
            variables=[CanonicalVariable.from_dict(v) for v in data.get("variables", [])],
            constraints=[CanonicalConstraint.from_dict(c) for c in data.get("constraints", [])],
            timeout_ms=data.get("timeout_ms", 30000),
            tactics=data.get("tactics", []),
            metadata=data.get("metadata", {}),
            correlation_id=data.get("correlation_id"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class CanonicalModel:
    """Z3 model in canonical format"""
    assignments: Dict[str, Any]
    objective_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignments": self.assignments,
            "objective_value": self.objective_value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CanonicalModel':
        return cls(
            assignments=data["assignments"],
            objective_value=data.get("objective_value"),
        )


@dataclass
class CanonicalSolverResponse:
    """
    Canonical solver response format

    Law of UTC: timestamp is UTC ISO-8601
    """
    result: Z3ResultStatus
    model: Optional[CanonicalModel] = None
    proof: Optional[str] = None
    reason: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result": self.result.value,
            "model": self.model.to_dict() if self.model else None,
            "proof": self.proof,
            "reason": self.reason,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CanonicalSolverResponse':
        model_data = data.get("model")
        model = CanonicalModel.from_dict(model_data) if model_data else None

        return cls(
            result=Z3ResultStatus(data["result"]),
            model=model,
            proof=data.get("proof"),
            reason=data.get("reason"),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            metadata=data.get("metadata", {}),
            correlation_id=data.get("correlation_id"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            errors=data.get("errors", []),
        )


# =============================================================================
# CANONICAL THEOREM REQUEST/RESPONSE
# =============================================================================

@dataclass
class CanonicalTheoremRequest:
    """
    Canonical theorem proving request

    Used for formal verification and theorem proving
    """
    theorem_statement: str  # SMT-LIB2 or natural language
    assumptions: List[str] = field(default_factory=list)
    variables: Dict[str, str] = field(default_factory=dict)  # name -> type
    timeout_ms: int = 30000  # MANDATORY
    proof_generation: bool = True
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theorem_statement": self.theorem_statement,
            "assumptions": self.assumptions,
            "variables": self.variables,
            "timeout_ms": self.timeout_ms,
            "proof_generation": self.proof_generation,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CanonicalTheoremRequest':
        return cls(
            theorem_statement=data["theorem_statement"],
            assumptions=data.get("assumptions", []),
            variables=data.get("variables", {}),
            timeout_ms=data.get("timeout_ms", 30000),
            proof_generation=data.get("proof_generation", True),
            correlation_id=data.get("correlation_id"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class CanonicalTheoremResponse:
    """
    Canonical theorem proving response
    """
    proven: bool
    proof: Optional[str] = None
    counterexample: Optional[Dict[str, Any]] = None
    tactic_used: Optional[str] = None
    execution_time_ms: float = 0.0
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proven": self.proven,
            "proof": self.proof,
            "counterexample": self.counterexample,
            "tactic_used": self.tactic_used,
            "execution_time_ms": self.execution_time_ms,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CanonicalTheoremResponse':
        return cls(
            proven=data["proven"],
            proof=data.get("proof"),
            counterexample=data.get("counterexample"),
            tactic_used=data.get("tactic_used"),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            correlation_id=data.get("correlation_id"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            errors=data.get("errors", []),
        )


# =============================================================================
# TRANSFORMATION FUNCTIONS
# =============================================================================

def canonical_to_z3_request(canonical: CanonicalSolverRequest) -> Dict[str, Any]:
    """
    Transform canonical request to Z3 API format

    Anti-Corruption Layer: Translate canonical format to Z3-specific format
    """
    # Convert variables to Z3 format
    z3_variables = []
    for var in canonical.variables:
        type_map = {
            ConstraintType.BOOLEAN: "Bool",
            ConstraintType.INTEGER: "Int",
            ConstraintType.REAL: "Real",
            ConstraintType.BIT_VECTOR: f"(_ BitVec {var.bit_width or 32})",
            ConstraintType.ARRAY: "(Array Int Int)",
            ConstraintType.STRING: "String",
        }
        z3_type = type_map.get(var.var_type, "Int")

        z3_variables.append({
            "name": var.name,
            "type": z3_type,
            "bounds": var.bounds,
        })

    # Convert constraints to Z3 format
    z3_constraints = []
    for constraint in canonical.constraints:
        z3_constraints.append({
            "expression": constraint.expression,
            "description": constraint.description,
        })

    return {
        "problem": canonical.problem,
        "problem_type": canonical.problem_type.value,
        "variables": z3_variables,
        "constraints": z3_constraints,
        "timeout_ms": canonical.timeout_ms,
        "tactics": canonical.tactics,
        "metadata": canonical.metadata,
        "correlation_id": canonical.correlation_id,
    }


def z3_to_canonical_response(
    z3_response: Dict[str, Any],
    correlation_id: Optional[str] = None
) -> CanonicalSolverResponse:
    """
    Transform Z3 response to canonical format

    Anti-Corruption Layer: Translate Z3 format to canonical format
    """
    # Parse result status
    result_str = z3_response.get("status", "unknown").lower()
    if result_str == "sat":
        result = Z3ResultStatus.SAT
    elif result_str == "unsat":
        result = Z3ResultStatus.UNSAT
    elif result_str == "unknown":
        result = Z3ResultStatus.UNKNOWN
    else:
        result = Z3ResultStatus.ERROR

    # Parse model if present
    model = None
    if "model" in z3_response and z3_response["model"]:
        model_data = z3_response["model"]
        if isinstance(model_data, dict):
            assignments = model_data.get("assignments", model_data)
            model = CanonicalModel(assignments=assignments)

    return CanonicalSolverResponse(
        result=result,
        model=model,
        proof=z3_response.get("proof"),
        reason=z3_response.get("reason"),
        execution_time_ms=z3_response.get("execution_time", z3_response.get("time", 0.0)),
        metadata=z3_response.get("metadata", {}),
        correlation_id=correlation_id or z3_response.get("correlation_id"),
        timestamp=datetime.now(timezone.utc).isoformat(),
        errors=z3_response.get("errors", []),
    )


def canonical_to_smtlib(canonical: CanonicalSolverRequest) -> str:
    """
    Convert canonical request to SMT-LIB2 format

    Used when Z3 is not directly available
    """
    lines = [
        "; Generated by RESE-Z3 Bridge",
        f"; Correlation ID: {canonical.correlation_id}",
        f"; Problem Type: {canonical.problem_type.value}",
        "",
        "(set-logic ALL)",
        "(set-option :produce-models true)",
        "(set-option :produce-proofs true)",
    ]

    # Declare variables
    for var in canonical.variables:
        type_map = {
            ConstraintType.BOOLEAN: "Bool",
            ConstraintType.INTEGER: "Int",
            ConstraintType.REAL: "Real",
            ConstraintType.BIT_VECTOR: f"(_ BitVec {var.bit_width or 32})",
            ConstraintType.ARRAY: "(Array Int Int)",
            ConstraintType.STRING: "String",
        }
        z3_type = type_map.get(var.var_type, "Int")
        lines.append(f"(declare-fun {var.name} () {z3_type})")

    # Add constraints
    for constraint in canonical.constraints:
        if constraint.description:
            lines.append(f"; {constraint.description}")
        lines.append(f"(assert {constraint.expression})")

    # Add problem if it's a full SMT-LIB script
    if canonical.problem and not canonical.problem.startswith(";"):
        if "(check-sat)" not in canonical.problem:
            lines.append(canonical.problem)

    # Check satisfiability
    lines.append("(check-sat)")

    # Get model if satisfiable
    lines.append("(get-model)")

    return "\n".join(lines)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_solver_request(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate canonical solver request

    Returns:
        (is_valid, error_message)
    """
    try:
        # Check required fields
        if "problem" not in data:
            return False, "Missing required field: problem"

        if "problem_type" not in data:
            return False, "Missing required field: problem_type"

        # Validate problem_type
        try:
            ProblemType(data["problem_type"])
        except ValueError:
            return False, f"Invalid problem_type: {data['problem_type']}"

        # Validate timeout_ms (mandatory)
        timeout_ms = data.get("timeout_ms", 0)
        if not isinstance(timeout_ms, int) or timeout_ms <= 0:
            return False, "timeout_ms must be a positive integer"

        if timeout_ms > 300000:  # 5 minutes max
            return False, "timeout_ms cannot exceed 300000ms (5 minutes)"

        # Validate variables if present
        if "variables" in data:
            for var in data["variables"]:
                if "name" not in var or "var_type" not in var:
                    return False, "Variable must have name and var_type"

        # Validate constraints if present
        if "constraints" in data:
            for constraint in data["constraints"]:
                if "expression" not in constraint:
                    return False, "Constraint must have expression"

        return True, None

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_theorem_request(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate canonical theorem request

    Returns:
        (is_valid, error_message)
    """
    try:
        # Check required fields
        if "theorem_statement" not in data:
            return False, "Missing required field: theorem_statement"

        # Validate timeout_ms (mandatory)
        timeout_ms = data.get("timeout_ms", 0)
        if not isinstance(timeout_ms, int) or timeout_ms <= 0:
            return False, "timeout_ms must be a positive integer"

        return True, None

    except Exception as e:
        return False, f"Validation error: {str(e)}"


# =============================================================================
# LEANAIDE INTEGRATION SCHEMAS
# =============================================================================

class LeanAideTaskType(Enum):
    """LeanAide task types"""
    TRANSLATE_THM = "translate_thm"
    TRANSLATE_THM_DETAILED = "translate_thm_detailed"
    TRANSLATE_DEF = "translate_def"
    THEOREM_DOC = "theorem_doc"
    DEF_DOC = "def_doc"
    THEOREM_NAME = "theorem_name"
    PROVE_FOR_FORMALIZATION = "prove_for_formalization"
    JSON_STRUCTURED = "json_structured"
    LEAN_FROM_JSON_STRUCTURED = "lean_from_json_structured"
    ELABORATE = "elaborate"
    MATH_QUERY = "math_query"


@dataclass
class LeanAideAutoformalizeRequest:
    """
    Request for autoformalization (natural language to Lean 4)

    Law of Configuration Explicitness: timeout_ms is mandatory
    Law of UTC: timestamp is UTC ISO-8601
    """
    natural_language: str  # Natural language theorem/definition
    task_type: LeanAideTaskType = LeanAideTaskType.TRANSLATE_THM_DETAILED
    theorem_name: Optional[str] = None
    timeout_ms: int = 30000  # MANDATORY
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "natural_language": self.natural_language,
            "task_type": self.task_type.value,
            "theorem_name": self.theorem_name,
            "timeout_ms": self.timeout_ms,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LeanAideAutoformalizeRequest':
        return cls(
            natural_language=data["natural_language"],
            task_type=LeanAideTaskType(data.get("task_type", LeanAideTaskType.TRANSLATE_THM_DETAILED.value)),
            theorem_name=data.get("theorem_name"),
            timeout_ms=data.get("timeout_ms", 30000),
            correlation_id=data.get("correlation_id"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LeanAideAutoformalizeResponse:
    """
    Response from autoformalization

    Contains Lean 4 code generated from natural language
    """
    success: bool
    lean_code: Optional[str] = None
    theorem_name: Optional[str] = None
    theorem_type: Optional[str] = None
    proof_sketch: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "lean_code": self.lean_code,
            "theorem_name": self.theorem_name,
            "theorem_type": self.theorem_type,
            "proof_sketch": self.proof_sketch,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LeanAideAutoformalizeResponse':
        return cls(
            success=data["success"],
            lean_code=data.get("lean_code"),
            theorem_name=data.get("theorem_name"),
            theorem_type=data.get("theorem_type"),
            proof_sketch=data.get("proof_sketch"),
            error=data.get("error"),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            correlation_id=data.get("correlation_id"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LeanAideProveRequest:
    """
    Request for AI-powered theorem proving

    Law of Configuration Explicitness: timeout_ms is mandatory
    Law of UTC: timestamp is UTC ISO-8601
    """
    theorem_text: str  # Natural language theorem
    theorem_code: Optional[str] = None  # Lean 4 code (if already formalized)
    theorem_statement: Optional[str] = None  # Elaborated theorem type
    generate_proof: bool = True
    timeout_ms: int = 60000  # MANDATORY
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theorem_text": self.theorem_text,
            "theorem_code": self.theorem_code,
            "theorem_statement": self.theorem_statement,
            "generate_proof": self.generate_proof,
            "timeout_ms": self.timeout_ms,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LeanAideProveRequest':
        return cls(
            theorem_text=data["theorem_text"],
            theorem_code=data.get("theorem_code"),
            theorem_statement=data.get("theorem_statement"),
            generate_proof=data.get("generate_proof", True),
            timeout_ms=data.get("timeout_ms", 60000),
            correlation_id=data.get("correlation_id"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LeanAideProveResponse:
    """
    Response from AI-powered theorem proving
    """
    success: bool
    proof: Optional[str] = None
    tactics_used: List[str] = field(default_factory=list)
    proof_script: Optional[str] = None
    counterexample: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "proof": self.proof,
            "tactics_used": self.tactics_used,
            "proof_script": self.proof_script,
            "counterexample": self.counterexample,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LeanAideProveResponse':
        return cls(
            success=data["success"],
            proof=data.get("proof"),
            tactics_used=data.get("tactics_used", []),
            proof_script=data.get("proof_script"),
            counterexample=data.get("counterexample"),
            error=data.get("error"),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            correlation_id=data.get("correlation_id"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Z3ToLeanTranslationRequest:
    """
    Request for Z3 to Lean 4 translation

    Law of Configuration Explicitness: timeout_ms is mandatory
    Law of UTC: timestamp is UTC ISO-8601
    """
    smtlib_content: str  # SMT-LIB2 content
    constraint_type: ConstraintType = ConstraintType.BOOLEAN
    generate_proof: bool = False
    timeout_ms: int = 30000  # MANDATORY
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "smtlib_content": self.smtlib_content,
            "constraint_type": self.constraint_type.value,
            "generate_proof": self.generate_proof,
            "timeout_ms": self.timeout_ms,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Z3ToLeanTranslationRequest':
        return cls(
            smtlib_content=data["smtlib_content"],
            constraint_type=ConstraintType(data.get("constraint_type", ConstraintType.BOOLEAN.value)),
            generate_proof=data.get("generate_proof", False),
            timeout_ms=data.get("timeout_ms", 30000),
            correlation_id=data.get("correlation_id"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Z3ToLeanTranslationResponse:
    """
    Response from Z3 to Lean 4 translation
    """
    success: bool
    lean_code: Optional[str] = None
    theorem_statement: Optional[str] = None
    variables: List[str] = field(default_factory=list)
    translated_constraints: List[str] = field(default_factory=list)
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "lean_code": self.lean_code,
            "theorem_statement": self.theorem_statement,
            "variables": self.variables,
            "translated_constraints": self.translated_constraints,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Z3ToLeanTranslationResponse':
        return cls(
            success=data["success"],
            lean_code=data.get("lean_code"),
            theorem_statement=data.get("theorem_statement"),
            variables=data.get("variables", []),
            translated_constraints=data.get("translated_constraints", []),
            error=data.get("error"),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            correlation_id=data.get("correlation_id"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LeanAideTacticSuggestionRequest:
    """
    Request for AI-powered tactic suggestions

    Law of Configuration Explicitness: timeout_ms is mandatory
    Law of UTC: timestamp is UTC ISO-8601
    """
    goal_state: str  # Current goal state in Lean 4
    context: Optional[str] = None  # Additional context
    num_suggestions: int = 3
    timeout_ms: int = 15000  # MANDATORY
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_state": self.goal_state,
            "context": self.context,
            "num_suggestions": self.num_suggestions,
            "timeout_ms": self.timeout_ms,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LeanAideTacticSuggestionRequest':
        return cls(
            goal_state=data["goal_state"],
            context=data.get("context"),
            num_suggestions=data.get("num_suggestions", 3),
            timeout_ms=data.get("timeout_ms", 15000),
            correlation_id=data.get("correlation_id"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LeanAideTacticSuggestion:
    """Individual tactic suggestion"""
    tactic: str
    description: Optional[str] = None
    confidence: float = 0.0
    reasoning: Optional[str] = None


@dataclass
class LeanAideTacticSuggestionResponse:
    """
    Response with tactic suggestions
    """
    success: bool
    suggestions: List[LeanAideTacticSuggestion] = field(default_factory=list)
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "suggestions": [s.__dict__ for s in self.suggestions],
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LeanAideTacticSuggestionResponse':
        suggestions = []
        for sugg_data in data.get("suggestions", []):
            suggestions.append(LeanAideTacticSuggestion(
                tactic=sugg_data["tactic"],
                description=sugg_data.get("description"),
                confidence=sugg_data.get("confidence", 0.0),
                reasoning=sugg_data.get("reasoning"),
            ))

        return cls(
            success=data["success"],
            suggestions=suggestions,
            error=data.get("error"),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            correlation_id=data.get("correlation_id"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# LEANAIDE VALIDATION FUNCTIONS
# =============================================================================

def validate_autoformalize_request(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate autoformalization request

    Returns:
        (is_valid, error_message)
    """
    try:
        # Check required fields
        if "natural_language" not in data:
            return False, "Missing required field: natural_language"

        # Validate timeout_ms (mandatory)
        timeout_ms = data.get("timeout_ms", 0)
        if not isinstance(timeout_ms, int) or timeout_ms <= 0:
            return False, "timeout_ms must be a positive integer"

        # Validate task_type if present
        if "task_type" in data:
            try:
                LeanAideTaskType(data["task_type"])
            except ValueError:
                return False, f"Invalid task_type: {data['task_type']}"

        return True, None

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_prove_request(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate AI-powered proving request

    Returns:
        (is_valid, error_message)
    """
    try:
        # Check required fields
        if "theorem_text" not in data:
            return False, "Missing required field: theorem_text"

        # Validate timeout_ms (mandatory)
        timeout_ms = data.get("timeout_ms", 0)
        if not isinstance(timeout_ms, int) or timeout_ms <= 0:
            return False, "timeout_ms must be a positive integer"

        return True, None

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_translation_request(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate Z3 to Lean translation request

    Returns:
        (is_valid, error_message)
    """
    try:
        # Check required fields
        if "smtlib_content" not in data:
            return False, "Missing required field: smtlib_content"

        # Validate timeout_ms (mandatory)
        timeout_ms = data.get("timeout_ms", 0)
        if not isinstance(timeout_ms, int) or timeout_ms <= 0:
            return False, "timeout_ms must be a positive integer"

        # Validate constraint_type if present
        if "constraint_type" in data:
            try:
                ConstraintType(data["constraint_type"])
            except ValueError:
                return False, f"Invalid constraint_type: {data['constraint_type']}"

        return True, None

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_tactic_suggestion_request(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate tactic suggestion request

    Returns:
        (is_valid, error_message)
    """
    try:
        # Check required fields
        if "goal_state" not in data:
            return False, "Missing required field: goal_state"

        # Validate timeout_ms (mandatory)
        timeout_ms = data.get("timeout_ms", 0)
        if not isinstance(timeout_ms, int) or timeout_ms <= 0:
            return False, "timeout_ms must be a positive integer"

        # Validate num_suggestions if present
        if "num_suggestions" in data:
            num_suggestions = data["num_suggestions"]
            if not isinstance(num_suggestions, int) or num_suggestions < 1 or num_suggestions > 10:
                return False, "num_suggestions must be between 1 and 10"

        return True, None

    except Exception as e:
        return False, f"Validation error: {str(e)}"
