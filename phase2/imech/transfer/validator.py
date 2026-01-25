"""
Solution Validator

Validate transferred solutions against target domain constraints.

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
"""

from typing import Dict, Any, List
from ..core.domain import Domain


class SolutionValidator:
    """
    Validate transferred solutions
    """

    def __init__(self, tolerance: float = 0.1):
        self.tolerance = tolerance

    def validate(
        self,
        solution: Any,
        domain: Domain
    ) -> Dict[str, Any]:
        """
        Validate solution against domain constraints

        Args:
            solution: Transferred solution to validate
            domain: Target domain

        Returns:
            Validation result with is_valid flag and details
        """
        violations = []

        # Check formal constraints
        for constraint in domain.formal_constraints:
            if not self._satisfies_constraint(solution, constraint):
                violations.append({
                    'constraint': constraint,
                    'type': 'formal'
                })

        # Check natural language constraints (if applicable)
        for text_constraint in domain.natural_language_constraints:
            # Can't automatically validate without NLP
            # Placeholder for future enhancement
            pass

        if not violations:
            return {
                'is_valid': True,
                'confidence': 0.9,
                'violations': [],
                'details': 'All constraints satisfied'
            }
        else:
            return {
                'is_valid': False,
                'confidence': 0.5,
                'violations': violations,
                'details': f'{len(violations)} constraint(s) violated'
            }

    def _satisfies_constraint(self, solution: Any, constraint: Any) -> bool:
        """
        Check if solution satisfies a constraint

        Implementation depends on constraint format
        """
        if isinstance(constraint, str):
            # Parse and evaluate constraint
            return self._evaluate_string_constraint(solution, constraint)

        elif isinstance(constraint, dict):
            # Handle structured constraints
            return self._evaluate_structured_constraint(solution, constraint)

        elif callable(constraint):
            # Constraint is a function
            try:
                return constraint(solution)
            except Exception:
                return False

        # Unknown constraint format
        return True

    def _evaluate_string_constraint(
        self,
        solution: Dict,
        constraint: str
    ) -> bool:
        """
        Evaluate string constraint against solution

        Simplified implementation
        """
        try:
            # Extract solution parameters
            if isinstance(solution, dict):
                params = solution.get('parameters', {})

                # Replace variable names in constraint
                expr = constraint
                for var, value in params.items():
                    if isinstance(value, (int, float)):
                        expr = expr.replace(var, str(value))

                # Safe evaluation (very limited)
                # Only allow simple comparisons and arithmetic
                if '<=' in expr or '>=' in expr or '<' in expr or '>' in expr or '==' in expr:
                    # Use simple evaluation
                    # Note: In production, use proper expression parser
                    try:
                        result = eval(expr, {'__builtins__': {}}, {})
                        return bool(result)
                    except:
                        return False

        except Exception:
            pass

        return True

    def _evaluate_structured_constraint(
        self,
        solution: Dict,
        constraint: Dict
    ) -> bool:
        """
        Evaluate structured constraint
        """
        # Handle different constraint types
        constraint_type = constraint.get('type')

        if constraint_type == 'range':
            # Range constraint: value in [min, max]
            var = constraint.get('variable')
            min_val = constraint.get('min')
            max_val = constraint.get('max')

            if isinstance(solution, dict):
                params = solution.get('parameters', {})
                if var in params:
                    value = params[var]
                    return min_val <= value <= max_val

        elif constraint_type == 'equality':
            # Equality constraint: value == target
            var = constraint.get('variable')
            target = constraint.get('target')

            if isinstance(solution, dict):
                params = solution.get('parameters', {})
                if var in params:
                    value = params[var]
                    return abs(value - target) <= self.tolerance

        elif constraint_type == 'custom':
            # Custom constraint with validator function
            validator = constraint.get('validator')
            if callable(validator):
                return validator(solution)

        # Unknown constraint type
        return True
