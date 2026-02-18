"""
Symbolic Constraint Engine for OpenEvolve Knowledge Engine.

This module provides symbolic reasoning and constraint solving capabilities
for formal verification and validation of knowledge artifacts.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import re
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Structures
# ============================================================================

class ConstraintType(Enum):
    """Types of symbolic constraints"""
    REQUIRED = "required"           # Must be satisfied
    OPTIONAL = "optional"           # May be satisfied
    PREFERRED = "preferred"         # Should be satisfied if possible
    FORBIDDEN = "forbidden"         # Must NOT be satisfied


class ConstraintStatus(Enum):
    """Status of constraint satisfaction"""
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"
    PENDING = "pending"


@dataclass
class Constraint:
    """
    A symbolic constraint on knowledge artifacts.

    Attributes:
        id: Unique identifier
        type: Constraint type
        expression: Symbolic expression (e.g., "x > 5", "A AND B")
        description: Human-readable description
        variables: Variables in the constraint
        status: Current satisfaction status
        metadata: Additional metadata
    """
    id: str
    type: ConstraintType
    expression: str
    description: str
    variables: Set[str] = field(default_factory=set)
    status: ConstraintStatus = ConstraintStatus.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Extract variables from expression after initialization."""
        if not self.variables:
            self.variables = self._extract_variables()

    def _extract_variables(self) -> Set[str]:
        """Extract variable names from the expression."""
        # Simple regex to find potential variables
        # Matches words that aren't operators or keywords
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        tokens = set(re.findall(pattern, self.expression))

        # Filter out common operators and keywords
        keywords = {'AND', 'OR', 'NOT', 'TRUE', 'FALSE', 'NULL',
                   'IF', 'THEN', 'ELSE', 'FOR', 'WHILE', 'IN', 'IS'}

        return tokens - keywords


@dataclass
class ConstraintViolation:
    """
    A violation of a constraint.

    Attributes:
        constraint_id: ID of violated constraint
        violation_type: Type of violation
        message: Description of the violation
        severity: Severity level (1-10)
        timestamp: When violation was detected
        context: Additional context
    """
    constraint_id: str
    violation_type: str
    message: str
    severity: int = 5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SatisfactionResult:
    """
    Result of constraint satisfaction checking.

    Attributes:
        is_satisfied: Whether all constraints are satisfied
        satisfied_constraints: IDs of satisfied constraints
        violated_constraints: IDs of violated constraints
        violations: Details of violations
        timestamp: When check was performed
    """
    is_satisfied: bool
    satisfied_constraints: Set[str] = field(default_factory=set)
    violated_constraints: Set[str] = field(default_factory=set)
    violations: List[ConstraintViolation] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# Main Engine
# ============================================================================

class SymbolicConstraintEngine:
    """
    Symbolic constraint engine for knowledge validation.

    Provides:
    - Constraint definition and management
    - Satisfaction checking
    - Violation detection
    - Formal verification helpers
    """

    def __init__(self):
        """Initialize the constraint engine."""
        self.constraints: Dict[str, Constraint] = {}
        self.constraint_dependencies: Dict[str, Set[str]] = {}
        self.evaluation_count = 0
        self.violation_count = 0

    def add_constraint(self, constraint: Constraint) -> bool:
        """
        Add a constraint to the engine.

        Args:
            constraint: Constraint to add

        Returns:
            True if added successfully
        """
        if constraint.id in self.constraints:
            logger.warning(f"Constraint {constraint.id} already exists")
            return False

        self.constraints[constraint.id] = constraint
        logger.info({
            "msg": "Constraint added",
            "constraint_id": constraint.id,
            "type": constraint.type.value,
            "expression": constraint.expression
        })
        return True

    def remove_constraint(self, constraint_id: str) -> bool:
        """
        Remove a constraint from the engine.

        Args:
            constraint_id: ID of constraint to remove

        Returns:
            True if removed successfully
        """
        if constraint_id not in self.constraints:
            logger.warning(f"Constraint {constraint_id} not found")
            return False

        del self.constraints[constraint_id]
        if constraint_id in self.constraint_dependencies:
            del self.constraint_dependencies[constraint_id]

        logger.info({
            "msg": "Constraint removed",
            "constraint_id": constraint_id
        })
        return True

    def check_satisfaction(
        self,
        context: Dict[str, Any],
        constraint_ids: Optional[List[str]] = None
    ) -> SatisfactionResult:
        """
        Check constraint satisfaction.

        Args:
            context: Variable bindings to check against
            constraint_ids: Optional list of constraints to check
                       (checks all if None)

        Returns:
            SatisfactionResult with details
        """
        self.evaluation_count += 1

        # Determine which constraints to check
        if constraint_ids is None:
            constraints_to_check = list(self.constraints.values())
        else:
            constraints_to_check = [
                self.constraints[cid] for cid in constraint_ids
                if cid in self.constraints
            ]

        satisfied = set()
        violated = set()
        violations = []

        for constraint in constraints_to_check:
            result = self._evaluate_constraint(constraint, context)

            if result:
                satisfied.add(constraint.id)
                constraint.status = ConstraintStatus.SATISFIED
            else:
                violated.add(constraint.id)
                constraint.status = ConstraintStatus.VIOLATED
                self.violation_count += 1

                # Create violation record
                violations.append(ConstraintViolation(
                    constraint_id=constraint.id,
                    violation_type=constraint.type.value,
                    message=f"Constraint '{constraint.expression}' not satisfied",
                    severity=5 if constraint.type == ConstraintType.REQUIRED else 3,
                    context={"constraint": constraint.expression, "context": context}
                ))

        is_satisfied = len(violated) == 0

        logger.info({
            "msg": "Satisfaction check completed",
            "is_satisfied": is_satisfied,
            "satisfied_count": len(satisfied),
            "violated_count": len(violated)
        })

        return SatisfactionResult(
            is_satisfied=is_satisfied,
            satisfied_constraints=satisfied,
            violated_constraints=violated,
            violations=violations
        )

    def _evaluate_constraint(
        self,
        constraint: Constraint,
        context: Dict[str, Any]
    ) -> bool:
        """
        Evaluate a single constraint.

        Args:
            constraint: Constraint to evaluate
            context: Variable bindings

        Returns:
            True if constraint is satisfied
        """
        try:
            expression = constraint.expression

            # For forbidden constraints, check if expression is false
            if constraint.type == ConstraintType.FORBIDDEN:
                return not self._evaluate_expression(expression, context)

            # For required/preferred/optional, check if expression is true
            return self._evaluate_expression(expression, context)

        except Exception as e:
            logger.error({
                "msg": "Constraint evaluation failed",
                "constraint_id": constraint.id,
                "error": str(e)
            })
            return False

    def _evaluate_expression(
        self,
        expression: str,
        context: Dict[str, Any]
    ) -> bool:
        """
        Evaluate a symbolic expression.

        This is a simplified implementation. A full system would use
        a proper symbolic evaluation library like z3 or sympy.

        Args:
            expression: Expression to evaluate
            context: Variable bindings

        Returns:
            Boolean result
        """
        # Simple variable substitution
        result_expr = expression
        for var, value in context.items():
            if isinstance(value, bool):
                result_expr = result_expr.replace(var, str(value))
            elif isinstance(value, (int, float)):
                result_expr = result_expr.replace(var, str(value))
            elif isinstance(value, str):
                result_expr = result_expr.replace(var, f'"{value}"')

        # Handle simple comparisons
        # This is very basic - a real implementation would use proper parsing
        try:
            # Check for existence: "var"
            if result_expr.strip() in context:
                return bool(context[result_expr.strip()])

            # Check for simple comparison: "var == value"
            if '==' in result_expr:
                parts = result_expr.split('==')
                if len(parts) == 2:
                    left = parts[0].strip().strip('"\'')
                    right = parts[1].strip().strip('"\'')
                    return left == right

            # Check for boolean keywords
            if result_expr.upper() == 'TRUE':
                return True
            if result_expr.upper() == 'FALSE':
                return False

            # Default: try to eval as Python (use with caution!)
            # In production, use a proper expression evaluator
            return bool(eval(result_expr, {"__builtins__": {}}, context))

        except Exception:
            # If we can't evaluate, default to False
            return False

    def find_violations(
        self,
        context: Dict[str, Any]
    ) -> List[ConstraintViolation]:
        """
        Find all constraint violations in the context.

        Args:
            context: Variable bindings to check

        Returns:
            List of constraint violations
        """
        result = self.check_satisfaction(context)
        return result.violations

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_constraints": len(self.constraints),
            "evaluation_count": self.evaluation_count,
            "violation_count": self.violation_count,
            "violation_rate": (
                self.violation_count / self.evaluation_count
                if self.evaluation_count > 0 else 0.0
            ),
            "constraints_by_type": self._count_by_type()
        }

    def _count_by_type(self) -> Dict[str, int]:
        """Count constraints by type."""
        counts = {}
        for constraint in self.constraints.values():
            ctype = constraint.type.value
            counts[ctype] = counts.get(ctype, 0) + 1
        return counts


# ============================================================================
# Convenience Functions
# ============================================================================

def create_constraint(
    id: str,
    expression: str,
    type: ConstraintType = ConstraintType.REQUIRED,
    description: str = ""
) -> Constraint:
    """
    Convenience function to create a constraint.

    Args:
        id: Unique identifier
        expression: Symbolic expression
        type: Constraint type
        description: Human-readable description

    Returns:
        Constraint object
    """
    return Constraint(
        id=id,
        type=type,
        expression=expression,
        description=description or f"Constraint {id}"
    )


def validate_knowledge(
    knowledge: Dict[str, Any],
    constraints: List[Constraint]
) -> SatisfactionResult:
    """
    Validate knowledge against a list of constraints.

    Args:
        knowledge: Knowledge to validate
        constraints: Constraints to check

    Returns:
        SatisfactionResult
    """
    engine = SymbolicConstraintEngine()

    for constraint in constraints:
        engine.add_constraint(constraint)

    return engine.check_satisfaction(knowledge)


# Export all components
__all__ = [
    'ConstraintType',
    'ConstraintStatus',
    'Constraint',
    'ConstraintViolation',
    'SatisfactionResult',
    'SymbolicConstraintEngine',
    'create_constraint',
    'validate_knowledge'
]
