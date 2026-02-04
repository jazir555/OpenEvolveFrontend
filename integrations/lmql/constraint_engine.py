"""Constraint evaluation engine for LMQL-style constraints.

Python implementation of LMQL constraint checking.
Provides constraint parsing, evaluation, and optimization.

Architecture: SSOT (Single Source of Truth)
- Primary implementation in integrations/lmql/
- Knowledge Engine uses these via wrapper

Author: OpenEvolve
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# Configure structured logging
logger = logging.getLogger(__name__)

# Type variable
T = TypeVar("T")


# =============================================================================
# CONSTRAINT TYPES
# =============================================================================


class ConstraintOperator(Enum):
    """Operators supported in constraints."""
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    IN = "in"
    NOT_IN = "not in"
    CONTAINS = "contains"
    MATCHES = "matches"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"


class ConstraintType(Enum):
    """Types of constraints."""
    LENGTH = "length"
    TYPE = "type"
    REGEX = "regex"
    RANGE = "range"
    ENUM = "enum"
    CUSTOM = "custom"
    STOP_AT = "stop_at"
    STOPS_BEFORE = "stops_before"
    AND = "and"
    OR = "or"
    NOT = "not"


# =============================================================================
# CONSTRAINT RESULTS
# =============================================================================


@dataclass
class ConstraintEvaluationResult:
    """Result of constraint evaluation.
    
    Attributes:
        satisfied: Whether the constraint is satisfied
        value: The value that was evaluated
        constraint: The constraint that was evaluated
        error_message: Error message if constraint not satisfied
        metadata: Additional evaluation metadata
        timestamp: UTC timestamp of evaluation
    """
    satisfied: bool
    value: Any = None
    constraint: Optional["Constraint"] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class BatchEvaluationResult:
    """Result of batch constraint evaluation."""
    all_satisfied: bool
    results: List[ConstraintEvaluationResult] = field(default_factory=list)
    satisfied_count: int = 0
    failed_count: int = 0
    total_count: int = 0


# =============================================================================
# BASE CONSTRAINT CLASS
# =============================================================================


class Constraint(ABC):
    """Abstract base class for constraints."""
    
    @abstractmethod
    def evaluate(self, value: Any) -> ConstraintEvaluationResult:
        """Evaluate the constraint against a value."""
        pass
        
    @abstractmethod
    def to_lmql_syntax(self) -> str:
        """Convert to LMQL syntax string."""
        pass
        
    @abstractmethod
    def get_type(self) -> ConstraintType:
        """Get the constraint type."""
        pass


# =============================================================================
# CONCRETE CONSTRAINT IMPLEMENTATIONS
# =============================================================================


@dataclass
class LengthConstraint(Constraint):
    """Constraint on string/list length.
    
    Examples:
        >>> LengthConstraint(min=1, max=100).evaluate("hello")
        >>> LengthConstraint(min=1).evaluate([1, 2, 3])
    """
    min: Optional[int] = None
    max: Optional[int] = None
    exact: Optional[int] = None
    description: Optional[str] = None
    
    def evaluate(self, value: Any) -> ConstraintEvaluationResult:
        """Evaluate length constraint."""
        try:
            length = len(value) if value is not None else 0
        except TypeError:
            return ConstraintEvaluationResult(
                satisfied=False,
                value=value,
                constraint=self,
                error_message=f"Value {value} does not support len()"
            )
            
        errors = []
        
        if self.exact is not None and length != self.exact:
            errors.append(f"Length {length} != {self.exact}")
            
        if self.min is not None and length < self.min:
            errors.append(f"Length {length} < minimum {self.min}")
            
        if self.max is not None and length > self.max:
            errors.append(f"Length {length} > maximum {self.max}")
            
        return ConstraintEvaluationResult(
            satisfied=len(errors) == 0,
            value=value,
            constraint=self,
            error_message="; ".join(errors) if errors else None,
            metadata={"length": length}
        )
        
    def to_lmql_syntax(self) -> str:
        """Convert to LMQL syntax."""
        if self.exact is not None:
            return f"len(VALUE) == {self.exact}"
        parts = []
        if self.min is not None:
            parts.append(f"len(VALUE) >= {self.min}")
        if self.max is not None:
            parts.append(f"len(VALUE) <= {self.max}")
        return " and ".join(parts) if parts else ""
        
    def get_type(self) -> ConstraintType:
        return ConstraintType.LENGTH


@dataclass
class TypeConstraint(Constraint):
    """Constraint on value type.
    
    Examples:
        >>> TypeConstraint(allowed_types=["str", "int"]).evaluate("hello")
        >>> TypeConstraint(allowed_types=["list"]).evaluate([1, 2, 3])
    """
    allowed_types: List[str]
    description: Optional[str] = None
    
    TYPE_MAP = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "none": type(None),
    }
    
    def evaluate(self, value: Any) -> ConstraintEvaluationResult:
        """Evaluate type constraint."""
        actual_type = type(value).__name__
        
        # Handle None specially
        if value is None and "none" in self.allowed_types:
            return ConstraintEvaluationResult(
                satisfied=True,
                value=value,
                constraint=self,
                metadata={"type": actual_type}
            )
            
        # Check if type is allowed
        type_allowed = False
        for allowed in self.allowed_types:
            if allowed in self.TYPE_MAP:
                if isinstance(value, self.TYPE_MAP[allowed]):
                    type_allowed = True
                    break
            elif actual_type == allowed:
                type_allowed = True
                break
                
        return ConstraintEvaluationResult(
            satisfied=type_allowed,
            value=value,
            constraint=self,
            error_message=None if type_allowed else f"Type {actual_type} not in {self.allowed_types}",
            metadata={"type": actual_type}
        )
        
    def to_lmql_syntax(self) -> str:
        """Convert to LMQL syntax."""
        types_str = ", ".join(f"'{t}'" for t in self.allowed_types)
        return f"type(VALUE) in [{types_str}]"
        
    def get_type(self) -> ConstraintType:
        return ConstraintType.TYPE


@dataclass
class RegexConstraint(Constraint):
    """Constraint matching regex pattern.
    
    Examples:
        >>> RegexConstraint(pattern=r"\\d{4}-\\d{2}-\\d{2}").evaluate("2024-01-15")
    """
    pattern: str
    flags: int = 0
    description: Optional[str] = None
    _compiled: Optional[Pattern] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Compile regex pattern."""
        try:
            self._compiled = re.compile(self.pattern, self.flags)
        except re.error as e:
            logger.error(f"Invalid regex pattern: {self.pattern}, error: {e}")
            self._compiled = None
            
    def evaluate(self, value: Any) -> ConstraintEvaluationResult:
        """Evaluate regex constraint."""
        if self._compiled is None:
            return ConstraintEvaluationResult(
                satisfied=False,
                value=value,
                constraint=self,
                error_message=f"Invalid regex pattern: {self.pattern}"
            )
            
        str_value = str(value)
        match = self._compiled.match(str_value)
        
        return ConstraintEvaluationResult(
            satisfied=match is not None,
            value=value,
            constraint=self,
            error_message=None if match else f"Value '{str_value}' does not match pattern '{self.pattern}'",
            metadata={"matched": match is not None, "groups": match.groups() if match else None}
        )
        
    def to_lmql_syntax(self) -> str:
        """Convert to LMQL syntax."""
        return f'REGEX(VALUE, r"{self.pattern}")'
        
    def get_type(self) -> ConstraintType:
        return ConstraintType.REGEX


@dataclass
class RangeConstraint(Constraint):
    """Constraint on numeric range.
    
    Examples:
        >>> RangeConstraint(min=0, max=100).evaluate(50)
        >>> RangeConstraint(min=0.0, max=1.0).evaluate(0.5)
    """
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None
    inclusive_min: bool = True
    inclusive_max: bool = True
    description: Optional[str] = None
    
    def evaluate(self, value: Any) -> ConstraintEvaluationResult:
        """Evaluate range constraint."""
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            return ConstraintEvaluationResult(
                satisfied=False,
                value=value,
                constraint=self,
                error_message=f"Value '{value}' is not numeric"
            )
            
        errors = []
        
        if self.min is not None:
            if self.inclusive_min and num_value < self.min:
                errors.append(f"Value {num_value} < minimum {self.min}")
            elif not self.inclusive_min and num_value <= self.min:
                errors.append(f"Value {num_value} <= minimum {self.min}")
                
        if self.max is not None:
            if self.inclusive_max and num_value > self.max:
                errors.append(f"Value {num_value} > maximum {self.max}")
            elif not self.inclusive_max and num_value >= self.max:
                errors.append(f"Value {num_value} >= maximum {self.max}")
                
        return ConstraintEvaluationResult(
            satisfied=len(errors) == 0,
            value=value,
            constraint=self,
            error_message="; ".join(errors) if errors else None,
            metadata={"numeric_value": num_value}
        )
        
    def to_lmql_syntax(self) -> str:
        """Convert to LMQL syntax."""
        parts = []
        if self.min is not None:
            op = ">=" if self.inclusive_min else ">"
            parts.append(f"VALUE {op} {self.min}")
        if self.max is not None:
            op = "<=" if self.inclusive_max else "<"
            parts.append(f"VALUE {op} {self.max}")
        return " and ".join(parts) if parts else ""
        
    def get_type(self) -> ConstraintType:
        return ConstraintType.RANGE


@dataclass
class EnumConstraint(Constraint):
    """Constraint restricting to enum values.
    
    Examples:
        >>> EnumConstraint(values=["yes", "no"]).evaluate("yes")
    """
    values: List[str]
    case_sensitive: bool = True
    description: Optional[str] = None
    
    def evaluate(self, value: Any) -> ConstraintEvaluationResult:
        """Evaluate enum constraint."""
        str_value = str(value)
        
        if self.case_sensitive:
            is_allowed = str_value in self.values
        else:
            is_allowed = str_value.lower() in [v.lower() for v in self.values]
            
        return ConstraintEvaluationResult(
            satisfied=is_allowed,
            value=value,
            constraint=self,
            error_message=None if is_allowed else f"Value '{str_value}' not in {self.values}",
            metadata={"allowed_values": self.values}
        )
        
    def to_lmql_syntax(self) -> str:
        """Convert to LMQL syntax."""
        values_str = ", ".join(f"'{v}'" for v in self.values)
        return f"VALUE in [{values_str}]"
        
    def get_type(self) -> ConstraintType:
        return ConstraintType.ENUM


@dataclass
class CustomConstraint(Constraint):
    """Custom constraint with user-provided predicate.
    
    Examples:
        >>> CustomConstraint(predicate=lambda x: x % 2 == 0).evaluate(4)
    """
    predicate: Callable[[Any], bool]
    predicate_name: str = "custom"
    description: Optional[str] = None
    
    def evaluate(self, value: Any) -> ConstraintEvaluationResult:
        """Evaluate custom constraint."""
        try:
            satisfied = self.predicate(value)
        except Exception as e:
            return ConstraintEvaluationResult(
                satisfied=False,
                value=value,
                constraint=self,
                error_message=f"Custom predicate error: {e}"
            )
            
        return ConstraintEvaluationResult(
            satisfied=satisfied,
            value=value,
            constraint=self,
            error_message=None if satisfied else f"Custom predicate '{self.predicate_name}' returned False",
            metadata={"predicate": self.predicate_name}
        )
        
    def to_lmql_syntax(self) -> str:
        """Convert to LMQL syntax."""
        return f"CUSTOM(VALUE, '{self.predicate_name}')"
        
    def get_type(self) -> ConstraintType:
        return ConstraintType.CUSTOM


@dataclass
class StopAtConstraint(Constraint):
    """Constraint to stop generation at specific sequence."""
    sequence: str
    description: Optional[str] = None
    
    def evaluate(self, value: Any) -> ConstraintEvaluationResult:
        """Check if value stops at the sequence."""
        str_value = str(value)
        # In generation context, this ensures the sequence is included
        # For validation, we check it ends with or contains the stop
        satisfied = self.sequence in str_value or str_value.endswith(self.sequence.rstrip())
        
        return ConstraintEvaluationResult(
            satisfied=satisfied,
            value=value,
            constraint=self,
            error_message=None if satisfied else f"Value does not stop at '{self.sequence}'",
            metadata={"stop_sequence": self.sequence}
        )
        
    def to_lmql_syntax(self) -> str:
        """Convert to LMQL syntax."""
        return f"STOPS_AT(VALUE, '{self.sequence}')"
        
    def get_type(self) -> ConstraintType:
        return ConstraintType.STOP_AT


@dataclass
class CompositeConstraint(Constraint):
    """Composite constraint combining multiple constraints."""
    constraints: List[Constraint]
    operator: ConstraintType  # AND or OR
    description: Optional[str] = None
    
    def evaluate(self, value: Any) -> ConstraintEvaluationResult:
        """Evaluate composite constraint."""
        results = [c.evaluate(value) for c in self.constraints]
        
        if self.operator == ConstraintType.AND:
            satisfied = all(r.satisfied for r in results)
        elif self.operator == ConstraintType.OR:
            satisfied = any(r.satisfied for r in results)
        else:
            satisfied = False
            
        failed = [r for r in results if not r.satisfied]
        
        return ConstraintEvaluationResult(
            satisfied=satisfied,
            value=value,
            constraint=self,
            error_message="; ".join(r.error_message for r in failed if r.error_message) if failed else None,
            metadata={"sub_results": results, "operator": self.operator.value}
        )
        
    def to_lmql_syntax(self) -> str:
        """Convert to LMQL syntax."""
        parts = [c.to_lmql_syntax() for c in self.constraints if c.to_lmql_syntax()]
        op = " and " if self.operator == ConstraintType.AND else " or "
        return f"({op.join(parts)})"
        
    def get_type(self) -> ConstraintType:
        return self.operator


# =============================================================================
# CONSTRAINT EVALUATOR
# =============================================================================


class ConstraintEvaluator:
    """Main constraint evaluation engine.
    
    Provides methods for evaluating various constraint types,
    batch evaluation, and constraint parsing.
    
    Example:
        >>> evaluator = ConstraintEvaluator()
        >>> result = evaluator.evaluate_length("hello", min=1, max=100)
        >>> results = evaluator.evaluate_all("hello", constraints)
    """
    
    def __init__(self):
        self._constraints: List[Constraint] = []
        self._metrics = {
            "evaluations": 0,
            "satisfied": 0,
            "failed": 0,
            "errors": 0,
        }
        
    def evaluate_length(
        self,
        value: Any,
        min: Optional[int] = None,
        max: Optional[int] = None,
        exact: Optional[int] = None
    ) -> ConstraintEvaluationResult:
        """Evaluate length constraint.
        
        Args:
            value: Value to check (must support len())
            min: Minimum length
            max: Maximum length
            exact: Exact length required
            
        Returns:
            Constraint evaluation result
        """
        constraint = LengthConstraint(min=min, max=max, exact=exact)
        return self._track(constraint.evaluate(value))
        
    def evaluate_type(
        self,
        value: Any,
        allowed_types: List[str]
    ) -> ConstraintEvaluationResult:
        """Evaluate type constraint.
        
        Args:
            value: Value to check
            allowed_types: List of allowed type names
            
        Returns:
            Constraint evaluation result
        """
        constraint = TypeConstraint(allowed_types=allowed_types)
        return self._track(constraint.evaluate(value))
        
    def evaluate_regex(
        self,
        value: Any,
        pattern: str,
        flags: int = 0
    ) -> ConstraintEvaluationResult:
        """Evaluate regex constraint.
        
        Args:
            value: Value to check
            pattern: Regex pattern
            flags: Regex flags
            
        Returns:
            Constraint evaluation result
        """
        constraint = RegexConstraint(pattern=pattern, flags=flags)
        return self._track(constraint.evaluate(value))
        
    def evaluate_range(
        self,
        value: Any,
        min: Optional[Union[int, float]] = None,
        max: Optional[Union[int, float]] = None,
        inclusive_min: bool = True,
        inclusive_max: bool = True
    ) -> ConstraintEvaluationResult:
        """Evaluate range constraint.
        
        Args:
            value: Numeric value to check
            min: Minimum value
            max: Maximum value
            inclusive_min: Whether min is inclusive
            inclusive_max: Whether max is inclusive
            
        Returns:
            Constraint evaluation result
        """
        constraint = RangeConstraint(
            min=min,
            max=max,
            inclusive_min=inclusive_min,
            inclusive_max=inclusive_max
        )
        return self._track(constraint.evaluate(value))
        
    def evaluate_enum(
        self,
        value: Any,
        values: List[str],
        case_sensitive: bool = True
    ) -> ConstraintEvaluationResult:
        """Evaluate enum constraint.
        
        Args:
            value: Value to check
            values: Allowed values
            case_sensitive: Whether comparison is case-sensitive
            
        Returns:
            Constraint evaluation result
        """
        constraint = EnumConstraint(values=values, case_sensitive=case_sensitive)
        return self._track(constraint.evaluate(value))
        
    def evaluate_custom(
        self,
        value: Any,
        predicate: Callable[[Any], bool],
        predicate_name: str = "custom"
    ) -> ConstraintEvaluationResult:
        """Evaluate custom constraint.
        
        Args:
            value: Value to check
            predicate: Function returning bool
            predicate_name: Name for error messages
            
        Returns:
            Constraint evaluation result
        """
        constraint = CustomConstraint(predicate=predicate, predicate_name=predicate_name)
        return self._track(constraint.evaluate(value))
        
    def evaluate_all(
        self,
        value: Any,
        constraints: List[Constraint]
    ) -> BatchEvaluationResult:
        """Evaluate multiple constraints.
        
        Args:
            value: Value to check
            constraints: List of constraints
            
        Returns:
            Batch evaluation result
        """
        results = []
        satisfied_count = 0
        failed_count = 0
        
        for constraint in constraints:
            result = constraint.evaluate(value)
            results.append(result)
            if result.satisfied:
                satisfied_count += 1
            else:
                failed_count += 1
            self._metrics["evaluations"] += 1
                
        return BatchEvaluationResult(
            all_satisfied=failed_count == 0,
            results=results,
            satisfied_count=satisfied_count,
            failed_count=failed_count,
            total_count=len(constraints)
        )
        
    def evaluate_and(
        self,
        value: Any,
        constraints: List[Constraint]
    ) -> ConstraintEvaluationResult:
        """Evaluate AND composite constraint."""
        composite = CompositeConstraint(
            constraints=constraints,
            operator=ConstraintType.AND
        )
        return self._track(composite.evaluate(value))
        
    def evaluate_or(
        self,
        value: Any,
        constraints: List[Constraint]
    ) -> ConstraintEvaluationResult:
        """Evaluate OR composite constraint."""
        composite = CompositeConstraint(
            constraints=constraints,
            operator=ConstraintType.OR
        )
        return self._track(composite.evaluate(value))
        
    def _track(self, result: ConstraintEvaluationResult) -> ConstraintEvaluationResult:
        """Track evaluation metrics."""
        self._metrics["evaluations"] += 1
        if result.satisfied:
            self._metrics["satisfied"] += 1
        else:
            self._metrics["failed"] += 1
        if result.error_message and "error" in result.error_message.lower():
            self._metrics["errors"] += 1
        return result
        
    def get_metrics(self) -> Dict[str, int]:
        """Get evaluation metrics."""
        return self._metrics.copy()
        
    def reset_metrics(self) -> None:
        """Reset evaluation metrics."""
        self._metrics = {
            "evaluations": 0,
            "satisfied": 0,
            "failed": 0,
            "errors": 0,
        }


# =============================================================================
# CONSTRAINT PARSER
# =============================================================================


class ConstraintParser:
    """Parser for LMQL-style constraint syntax.
    
    Parses LMQL WHERE clauses into Constraint objects.
    
    Example:
        >>> parser = ConstraintParser()
        >>> constraints = parser.parse("WHERE len(x) > 0 AND x in ['a', 'b']")
    """
    
    def __init__(self):
        self._patterns = {
            'length': re.compile(r'len\((\w+)\)\s*([<>=!]+)\s*(\d+)'),
            'length_range': re.compile(r'len\((\w+)\)\s*in\s*\[(\d+),\s*(\d+)\]'),
            'type': re.compile(r'type\((\w+)\)\s*in\s*\[(.+?)\]'),
            'regex': re.compile(r'REGEX\((\w+),\s*r["\'](.+?)["\']\)'),
            'range': re.compile(r'(\w+)\s*([<>=!]+)\s*([\d.]+)'),
            'enum': re.compile(r'(\w+)\s+in\s+\[(.+?)\]'),
            'stop_at': re.compile(r'STOPS_AT\((\w+),\s*["\'](.+?)["\']\)'),
        }
        
    def parse(self, lmql_where_clause: str) -> List[Constraint]:
        """Parse LMQL WHERE clause into constraints.
        
        Args:
            lmql_where_clause: WHERE clause from LMQL query
            
        Returns:
            List of parsed constraints
        """
        constraints = []
        
        # Remove WHERE keyword
        clause = lmql_where_clause.replace("WHERE", "").strip()
        
        # Split by AND/OR (simple approach)
        parts = [p.strip() for p in re.split(r'\s+(?:AND|and|&&)\s+', clause)]
        
        for part in parts:
            constraint = self._parse_single(part)
            if constraint:
                constraints.append(constraint)
                
        return constraints
        
    def _parse_single(self, part: str) -> Optional[Constraint]:
        """Parse a single constraint."""
        # Try length constraint
        match = self._patterns['length'].match(part)
        if match:
            var, op, val = match.groups()
            val = int(val)
            if op == '>':
                return LengthConstraint(min=val + 1)
            elif op == '>=':
                return LengthConstraint(min=val)
            elif op == '<':
                return LengthConstraint(max=val - 1)
            elif op == '<=':
                return LengthConstraint(max=val)
            elif op in ('==', '='):
                return LengthConstraint(exact=val)
                
        # Try length range
        match = self._patterns['length_range'].match(part)
        if match:
            var, min_val, max_val = match.groups()
            return LengthConstraint(min=int(min_val), max=int(max_val))
            
        # Try regex constraint
        match = self._patterns['regex'].match(part)
        if match:
            var, pattern = match.groups()
            return RegexConstraint(pattern=pattern)
            
        # Try range constraint
        match = self._patterns['range'].match(part)
        if match:
            var, op, val = match.groups()
            val = float(val)
            if op == '>':
                return RangeConstraint(min=val, inclusive_min=False)
            elif op == '>=':
                return RangeConstraint(min=val)
            elif op == '<':
                return RangeConstraint(max=val, inclusive_max=False)
            elif op == '<=':
                return RangeConstraint(max=val)
                
        # Try enum constraint
        match = self._patterns['enum'].match(part)
        if match:
            var, values_str = match.groups()
            values = [v.strip().strip("'\"") for v in values_str.split(',')]
            return EnumConstraint(values=values)
            
        # Try stop_at constraint
        match = self._patterns['stop_at'].match(part)
        if match:
            var, sequence = match.groups()
            return StopAtConstraint(sequence=sequence)
            
        return None
        
    def parse_from_lmql(self, lmql_query: str) -> Dict[str, List[Constraint]]:
        """Parse full LMQL query and extract constraints by variable.
        
        Args:
            lmql_query: Full LMQL query string
            
        Returns:
            Dictionary mapping variable names to their constraints
        """
        constraints_by_var: Dict[str, List[Constraint]] = {}
        
        # Extract WHERE clause
        where_match = re.search(r'WHERE\s+(.+?)(?:RETURN|DISTRIBUTION|$)', lmql_query, re.DOTALL)
        if not where_match:
            return constraints_by_var
            
        where_clause = where_match.group(1)
        
        # Split into individual constraints
        constraint_parts = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)
        
        for part in constraint_parts:
            # Try to identify the variable
            var_match = re.search(r'(\w+)', part)
            if var_match:
                var = var_match.group(1)
                constraint = self._parse_single(part.strip())
                if constraint:
                    if var not in constraints_by_var:
                        constraints_by_var[var] = []
                    constraints_by_var[var].append(constraint)
                    
        return constraints_by_var


# =============================================================================
# OPTIMIZATION HINTS
# =============================================================================


class ConstraintOptimizer:
    """Optimizer for constraint evaluation.
    
    Provides optimization hints and strategies for common constraint patterns.
    """
    
    @staticmethod
    def optimize_constraints(constraints: List[Constraint]) -> List[Constraint]:
        """Optimize constraint order for evaluation.
        
        Orders constraints by computational cost:
        1. Type checks (fastest)
        2. Length checks
        3. Enum checks
        4. Range checks
        5. Regex checks (slowest)
        
        Args:
            constraints: List of constraints
            
        Returns:
            Optimized constraint order
        """
        priority_map = {
            ConstraintType.TYPE: 0,
            ConstraintType.LENGTH: 1,
            ConstraintType.ENUM: 2,
            ConstraintType.RANGE: 3,
            ConstraintType.REGEX: 4,
            ConstraintType.CUSTOM: 5,
            ConstraintType.STOP_AT: 6,
        }
        
        return sorted(constraints, key=lambda c: priority_map.get(c.get_type(), 99))
        
    @staticmethod
    def get_optimization_hints(constraint: Constraint) -> Dict[str, Any]:
        """Get optimization hints for a constraint.
        
        Args:
            constraint: Constraint to analyze
            
        Returns:
            Dictionary of optimization hints
        """
        hints = {
            "can_batch": False,
            "can_short_circuit": False,
            "estimated_cost": "medium",
        }
        
        if isinstance(constraint, TypeConstraint):
            hints["can_batch"] = True
            hints["estimated_cost"] = "low"
            hints["can_short_circuit"] = True
        elif isinstance(constraint, LengthConstraint):
            hints["can_batch"] = True
            hints["estimated_cost"] = "low"
        elif isinstance(constraint, EnumConstraint):
            hints["can_batch"] = True
            hints["estimated_cost"] = "low"
            if len(constraint.values) <= 3:
                hints["can_short_circuit"] = True
        elif isinstance(constraint, RegexConstraint):
            hints["estimated_cost"] = "high"
            if constraint._compiled:
                hints["pattern_complexity"] = len(constraint.pattern)
        elif isinstance(constraint, RangeConstraint):
            hints["can_batch"] = True
            hints["estimated_cost"] = "low"
            
        return hints


# =============================================================================
# DEFAULT INSTANCES
# =============================================================================

_default_evaluator: Optional[ConstraintEvaluator] = None
_default_parser: Optional[ConstraintParser] = None
_default_optimizer: Optional[ConstraintOptimizer] = None


def get_default_evaluator() -> ConstraintEvaluator:
    """Get default constraint evaluator."""
    global _default_evaluator
    if _default_evaluator is None:
        _default_evaluator = ConstraintEvaluator()
    return _default_evaluator


def get_default_parser() -> ConstraintParser:
    """Get default constraint parser."""
    global _default_parser
    if _default_parser is None:
        _default_parser = ConstraintParser()
    return _default_parser


def get_default_optimizer() -> ConstraintOptimizer:
    """Get default constraint optimizer."""
    global _default_optimizer
    if _default_optimizer is None:
        _default_optimizer = ConstraintOptimizer()
    return _default_optimizer


def reset_defaults() -> None:
    """Reset default instances."""
    global _default_evaluator, _default_parser, _default_optimizer
    _default_evaluator = None
    _default_parser = None
    _default_optimizer = None


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "ConstraintOperator",
    "ConstraintType",
    # Data classes
    "ConstraintEvaluationResult",
    "BatchEvaluationResult",
    # Constraint classes
    "Constraint",
    "LengthConstraint",
    "TypeConstraint",
    "RegexConstraint",
    "RangeConstraint",
    "EnumConstraint",
    "CustomConstraint",
    "StopAtConstraint",
    "CompositeConstraint",
    # Engine classes
    "ConstraintEvaluator",
    "ConstraintParser",
    "ConstraintOptimizer",
    # Functions
    "get_default_evaluator",
    "get_default_parser",
    "get_default_optimizer",
    "reset_defaults",
]


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    evaluator = ConstraintEvaluator()
    
    # Test length constraint
    print("\nLength Constraints:")
    result = evaluator.evaluate_length("hello", min=1, max=10)
    print(f"  'hello' (min=1, max=10): {result.satisfied}")
    
    result = evaluator.evaluate_length("hello", min=10)
    print(f"  'hello' (min=10): {result.satisfied}, error: {result.error_message}")
    
    # Test regex constraint
    print("\nRegex Constraints:")
    result = evaluator.evaluate_regex("2024-01-15", r"\d{4}-\d{2}-\d{2}")
    print(f"  '2024-01-15' matches date pattern: {result.satisfied}")
    
    # Test range constraint
    print("\nRange Constraints:")
    result = evaluator.evaluate_range(50, min=0, max=100)
    print(f"  50 in [0, 100]: {result.satisfied}")
    
    result = evaluator.evaluate_range(150, min=0, max=100)
    print(f"  150 in [0, 100]: {result.satisfied}")
    
    # Test enum constraint
    print("\nEnum Constraints:")
    result = evaluator.evaluate_enum("yes", ["yes", "no"])
    print(f"  'yes' in ['yes', 'no']: {result.satisfied}")
    
    # Test parser
    print("\nConstraint Parsing:")
    parser = ConstraintParser()
    constraints = parser.parse("WHERE len(x) > 0 AND x in ['a', 'b'] AND REGEX(x, r'[a-z]+')")
    print(f"  Parsed {len(constraints)} constraints from LMQL")
    for c in constraints:
        print(f"    - {c.get_type().value}")
        
    # Test batch evaluation
    print("\nBatch Evaluation:")
    constraints = [
        LengthConstraint(min=1, max=100),
        RegexConstraint(pattern=r"^[a-z]+$"),
    ]
    batch_result = evaluator.evaluate_all("hello", constraints)
    print(f"  All constraints satisfied: {batch_result.all_satisfied}")
    print(f"  Satisfied: {batch_result.satisfied_count}, Failed: {batch_result.failed_count}")
