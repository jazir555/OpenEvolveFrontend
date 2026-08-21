"""sop_parameter - canonical SOP parameter specification.

Flat-script module providing the shared ``SOPParameter`` definition that the
``engines/`` SOP, planner and end-to-end-invention scripts expect via
``from sop_parameter import SOPParameter``.

An ``SOPParameter`` is a validated parameter slot in a Standard Operating
Procedure: it carries a ``name``, a declared ``type``, a ``value``, a
``constraints`` mapping and a ``validate()`` method that checks the value against
those constraints. Pure-Python, no external dependencies.

Supported constraint keys::

    min, max            numeric (inclusive) bounds
    exclusive_min/max   numeric (exclusive) bounds
    allowed             iterable of permitted values (enumeration)
    forbidden           iterable of rejected values
    pattern             regex the string form must fully match
    min_length/max_length   length bounds for str/list values
    required            when True, value must not be None/empty
    tolerance           +/- band around `nominal` (defaults to the value)
    nominal             target value for the tolerance check
    step                value must be an integral number of steps from min/0
    unit                expected unit label (checked against `self.unit`)
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ParameterType(str, Enum):
    """Declared type of an SOP parameter."""

    FLOAT = "float"
    INTEGER = "integer"
    STRING = "string"
    BOOLEAN = "boolean"
    ENUM = "enum"
    DURATION = "duration"
    LIST = "list"
    ANY = "any"

    @classmethod
    def coerce(cls, value: Any) -> "ParameterType":
        """Lenient lookup, accepting python types, aliases and member names."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.ANY
        if isinstance(value, type):
            return {
                float: cls.FLOAT, int: cls.INTEGER, str: cls.STRING,
                bool: cls.BOOLEAN, list: cls.LIST, tuple: cls.LIST,
            }.get(value, cls.ANY)
        text = str(value).strip().lower()
        aliases = {
            "num": cls.FLOAT, "number": cls.FLOAT, "double": cls.FLOAT, "real": cls.FLOAT,
            "int": cls.INTEGER, "integer": cls.INTEGER, "count": cls.INTEGER,
            "str": cls.STRING, "text": cls.STRING,
            "bool": cls.BOOLEAN, "flag": cls.BOOLEAN,
            "choice": cls.ENUM, "categorical": cls.ENUM,
            "time": cls.DURATION, "seconds": cls.DURATION,
            "array": cls.LIST, "sequence": cls.LIST,
        }
        for member in cls:
            if text in (member.value, member.name.lower()):
                return member
        return aliases.get(text, cls.ANY)


@dataclass
class ValidationResult:
    """Outcome of validating a parameter."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    parameter: Optional[str] = None

    def __bool__(self) -> bool:
        return self.valid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "parameter": self.parameter,
        }


_NUMERIC_TYPES = (ParameterType.FLOAT, ParameterType.INTEGER, ParameterType.DURATION)

# Sentinel distinguishing "argument omitted" from an explicit ``None``.
_MISSING = object()


@dataclass
class SOPParameter:
    """A validated parameter specification inside an SOP."""

    name: str
    type: ParameterType = ParameterType.ANY
    value: Any = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    unit: str = ""
    description: str = ""
    validation_method: str = ""
    critical: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        self.type = ParameterType.coerce(self.type)
        self.constraints = dict(self.constraints or {})
        if self.type is ParameterType.ANY and self.value is not None:
            self.type = ParameterType.coerce(type(self.value))
        # An `allowed` constraint implies an enumeration.
        if "allowed" in self.constraints and self.type in (ParameterType.ANY, ParameterType.STRING):
            self.type = ParameterType.ENUM
        if not self.unit and self.constraints.get("unit"):
            self.unit = str(self.constraints["unit"])

    # -- typing ----------------------------------------------------------
    @property
    def param_type(self) -> ParameterType:
        """Alias for :attr:`type` (avoids shadowing the builtin at call sites)."""
        return self.type

    def coerce_value(self, value: Any = _MISSING) -> Any:
        """Return ``value`` converted to the declared type, or unchanged on failure."""
        raw = self.value if value is _MISSING else value
        if raw is None:
            return None
        try:
            if self.type in (ParameterType.FLOAT, ParameterType.DURATION):
                return float(raw)
            if self.type is ParameterType.INTEGER:
                return int(raw)
            if self.type is ParameterType.STRING:
                return str(raw)
            if self.type is ParameterType.BOOLEAN:
                if isinstance(raw, str):
                    return raw.strip().lower() in ("1", "true", "yes", "y", "on")
                return bool(raw)
            if self.type is ParameterType.LIST:
                return list(raw) if isinstance(raw, (list, tuple, set)) else [raw]
        except (TypeError, ValueError):
            return raw
        return raw

    def _type_matches(self, value: Any) -> bool:
        if self.type in (ParameterType.ANY, ParameterType.ENUM):
            return True
        if self.type is ParameterType.BOOLEAN:
            return isinstance(value, bool)
        if self.type is ParameterType.INTEGER:
            return isinstance(value, int) and not isinstance(value, bool)
        if self.type in (ParameterType.FLOAT, ParameterType.DURATION):
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if self.type is ParameterType.STRING:
            return isinstance(value, str)
        if self.type is ParameterType.LIST:
            return isinstance(value, (list, tuple))
        return True

    # -- validation ------------------------------------------------------
    def validate(self, value: Any = None, strict: bool = False) -> ValidationResult:
        """Validate ``value`` (or ``self.value``) against the constraints.

        Set ``strict`` to reject values that need type coercion. Returns a
        :class:`ValidationResult`, which is falsy when validation fails.
        """
        raw = self.value if value is None else value
        errors: List[str] = []
        warnings: List[str] = []
        c = self.constraints

        if not self.name:
            errors.append("parameter has no name")

        # Required / presence.
        required = bool(c.get("required", self.critical))
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            if required:
                errors.append(f"{self.name or 'parameter'} is required but has no value")
            return ValidationResult(not errors, errors, warnings, self.name or None)

        # Type check (with coercion unless strict).
        if not self._type_matches(raw):
            coerced = self.coerce_value(raw)
            if strict or not self._type_matches(coerced):
                errors.append(
                    f"{self.name}: expected {self.type.value}, got {type(raw).__name__}"
                )
            else:
                warnings.append(f"{self.name}: coerced to {self.type.value}")
                raw = coerced

        # Enumeration membership.
        allowed = c.get("allowed")
        if allowed is not None and raw not in list(allowed):
            errors.append(f"{self.name}: {raw!r} not in allowed values {list(allowed)!r}")
        forbidden = c.get("forbidden")
        if forbidden is not None and raw in list(forbidden):
            errors.append(f"{self.name}: {raw!r} is forbidden")

        # Numeric bounds / tolerance / step.
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            errors.extend(self._check_numeric(float(raw)))
            if not math.isfinite(float(raw)):
                errors.append(f"{self.name}: value must be finite")

        # String / sequence checks.
        if isinstance(raw, str):
            pattern = c.get("pattern")
            if pattern and not re.fullmatch(str(pattern), raw):
                errors.append(f"{self.name}: {raw!r} does not match pattern {pattern!r}")
        if isinstance(raw, (str, list, tuple)):
            length = len(raw)
            if c.get("min_length") is not None and length < int(c["min_length"]):
                errors.append(f"{self.name}: length {length} < min_length {c['min_length']}")
            if c.get("max_length") is not None and length > int(c["max_length"]):
                errors.append(f"{self.name}: length {length} > max_length {c['max_length']}")

        # Unit agreement.
        expected_unit = c.get("unit")
        if expected_unit and self.unit and str(expected_unit) != str(self.unit):
            warnings.append(
                f"{self.name}: unit {self.unit!r} differs from expected {expected_unit!r}"
            )

        if self.critical and not self.validation_method:
            warnings.append(f"{self.name}: critical parameter has no validation_method")

        return ValidationResult(not errors, errors, warnings, self.name or None)

    def _check_numeric(self, num: float) -> List[str]:
        """Numeric bound, tolerance and step checks."""
        errors: List[str] = []
        c = self.constraints
        if c.get("min") is not None and num < float(c["min"]):
            errors.append(f"{self.name}: {num} < min {c['min']}")
        if c.get("max") is not None and num > float(c["max"]):
            errors.append(f"{self.name}: {num} > max {c['max']}")
        if c.get("exclusive_min") is not None and num <= float(c["exclusive_min"]):
            errors.append(f"{self.name}: {num} must be > {c['exclusive_min']}")
        if c.get("exclusive_max") is not None and num >= float(c["exclusive_max"]):
            errors.append(f"{self.name}: {num} must be < {c['exclusive_max']}")

        tolerance = c.get("tolerance")
        if tolerance is not None:
            nominal = float(c.get("nominal", self.value if isinstance(self.value, (int, float)) else num))
            if abs(num - nominal) > abs(float(tolerance)):
                errors.append(
                    f"{self.name}: {num} deviates from {nominal} by more than "
                    f"+/-{tolerance}"
                )

        step = c.get("step")
        if step:
            base = float(c.get("min", 0.0))
            steps = (num - base) / float(step)
            if abs(steps - round(steps)) > 1e-9:
                errors.append(f"{self.name}: {num} is not a multiple of step {step} from {base}")
        return errors

    def is_valid(self, value: Any = None) -> bool:
        """Boolean form of :meth:`validate`."""
        return bool(self.validate(value))

    def clamp(self, value: Any = None) -> Any:
        """Clamp a numeric value into the ``min``/``max`` window."""
        raw = self.coerce_value(self.value if value is None else value)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            return raw
        low, high = self.constraints.get("min"), self.constraints.get("max")
        if low is not None:
            raw = max(float(low), float(raw))
        if high is not None:
            raw = min(float(high), float(raw))
        return int(raw) if self.type is ParameterType.INTEGER else raw

    def set_value(self, value: Any, validate: bool = True) -> ValidationResult:
        """Assign ``value``, optionally rejecting it when invalid."""
        result = self.validate(value)
        if result.valid or not validate:
            self.value = self.coerce_value(value)
        return result

    # -- rendering / serialization ---------------------------------------
    def format_spec(self) -> str:
        """Human-readable specification line, e.g. ``temp: 20 C +/- 0.5``."""
        parts = [f"{self.name}:"]
        parts.append("<unset>" if self.value is None else f"{self.value}")
        if self.unit:
            parts.append(self.unit)
        tolerance = self.constraints.get("tolerance")
        if tolerance is not None:
            parts.append(f"+/- {tolerance}")
        bounds = []
        if self.constraints.get("min") is not None:
            bounds.append(f"min {self.constraints['min']}")
        if self.constraints.get("max") is not None:
            bounds.append(f"max {self.constraints['max']}")
        if bounds:
            parts.append(f"[{', '.join(bounds)}]")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "constraints": dict(self.constraints),
            "unit": self.unit,
            "description": self.description,
            "validation_method": self.validation_method,
            "critical": self.critical,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SOPParameter":
        return cls(
            name=data.get("name", ""),
            type=data.get("type", ParameterType.ANY),
            value=data.get("value"),
            constraints=dict(data.get("constraints") or {}),
            unit=data.get("unit", "") or "",
            description=data.get("description", "") or "",
            validation_method=data.get("validation_method", "") or "",
            critical=bool(data.get("critical", False)),
            metadata=dict(data.get("metadata") or {}),
        )

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.format_spec()


def validate_all(parameters: Iterable[SOPParameter]) -> Tuple[bool, Dict[str, ValidationResult]]:
    """Validate many parameters, returning ``(all_ok, {name: result})``."""
    results = {p.name: p.validate() for p in parameters}
    return all(bool(r) for r in results.values()), results


__all__ = [
    "SOPParameter",
    "ParameterType",
    "ValidationResult",
    "validate_all",
]
