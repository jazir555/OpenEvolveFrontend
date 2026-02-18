"""
Z3 Canonicalizer - Complete Implementation

Provides canonicalization and normalization of Z3 expressions and constraints.

Features:
- Expression normalization
- Variable renaming
- Constraint simplification
- SMT-LIB format conversion
- Canonization for better solver performance

Author: OpenEvolve Team
Date: 2026-02-17
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)

# Z3 imports
try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    z3 = None


class CanonicalizationRule(Enum):
    """Types of canonicalization rules."""
    VARIABLE_RENAMING = "variable_renaming"
    SIMPLIFICATION = "simplification"
    FLATTENING = "flattening"
    SORT_CANONICALIZATION = "sort_canonicalization"
    NNF = "nnf"  # Negation Normal Form
    DNF = "dnf"  # Disjunctive Normal Form
    CNF = "cnf"  # Conjunctive Normal Form


@dataclass
class CanonicalizationResult:
    """Result of canonicalization."""
    original: str
    canonical: str
    rules_applied: List[CanonicalizationRule] = field(default_factory=list)
    variable_map: Dict[str, str] = field(default_factory=dict)
    simplifications: int = 0
    success: bool = True
    error: Optional[str] = None


class Z3Canonicalizer:
    """
    Z3 Expression Canonicalizer.

    Normalizes and simplifies Z3 expressions for:
    - Better solver performance
    - Consistent comparison
    - Easier debugging
    - Proof extraction
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the canonicalizer.

        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.enable_variable_renaming = self.config.get("enable_variable_renaming", True)
        self.enable_simplification = self.config.get("enable_simplification", True)
        self.enable_flattening = self.config.get("enable_flattening", True)
        self._var_counter = 0
        self._var_map: Dict[str, str] = {}

    def canonicalize(self, expression: str, rules: Optional[List[CanonicalizationRule]] = None) -> CanonicalizationResult:
        """
        Canonicalize a Z3 expression.

        Args:
            expression: Input expression (Z3 SMT-LIB or Python-like)
            rules: Optional list of specific rules to apply

        Returns:
            Canonicalization result
        """
        try:
            rules = rules or [
                CanonicalizationRule.VARIABLE_RENAMING,
                CanonicalizationRule.SIMPLIFICATION,
                CanonicalizationRule.FLATTENING
            ]

            canonical = expression
            applied_rules = []
            simplifications = 0

            # Apply variable renaming
            if CanonicalizationRule.VARIABLE_RENAMING in rules and self.enable_variable_renaming:
                canonical, var_map = self._rename_variables(canonical)
                applied_rules.append(CanonicalizationRule.VARIABLE_RENAMING)

            # Apply simplification
            if CanonicalizationRule.SIMPLIFICATION in rules and self.enable_simplification:
                canonical, simps = self._simplify(canonical)
                simplifications = simps
                applied_rules.append(CanonicalizationRule.SIMPLIFICATION)

            # Apply flattening
            if CanonicalizationRule.FLATTENING in rules and self.enable_flattening:
                canonical = self._flatten(canonical)
                applied_rules.append(CanonicalizationRule.FLATTENING)

            return CanonicalizationResult(
                original=expression,
                canonical=canonical,
                rules_applied=applied_rules,
                variable_map=self._var_map.copy(),
                simplifications=simplifications,
                success=True
            )

        except Exception as e:
            logger.error(f"Canonicalization failed: {e}")
            return CanonicalizationResult(
                original=expression,
                canonical=expression,
                success=False,
                error=str(e)
            )

    def _rename_variables(self, expression: str) -> Tuple[str, Dict[str, str]]:
        """
        Rename variables to canonical form.

        Args:
            expression: Input expression

        Returns:
            Tuple of (canonicalized expression, variable map)
        """
        # Find all variables
        # Variables are typically alphanumeric strings starting with letter
        # Exclude keywords
        keywords = {
            'and', 'or', 'not', 'implies', 'ite', 'forall', 'exists',
            'true', 'false', 'sat', 'unsat', 'unknown', 'let', 'assert',
            'check-sat', 'get-model', 'push', 'pop', 'define-fun',
            'declare-fun', 'Int', 'Real', 'Bool', 'BitVec', 'Array'
        }

        # Extract variables using regex
        pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        potential_vars = set(re.findall(pattern, expression))
        variables = [v for v in potential_vars if v not in keywords and not v[0].isdigit()]

        # Create mapping to canonical names
        var_map = {}
        canonical_expr = expression

        for var in sorted(variables):
            if var not in self._var_map:
                self._var_map[var] = f"v{self._var_counter}"
                self._var_counter += 1

            canonical_var = self._var_map[var]
            var_map[var] = canonical_var

            # Replace in expression (word boundaries only)
            pattern = r'\b' + re.escape(var) + r'\b'
            canonical_expr = re.sub(pattern, canonical_var, canonical_expr)

        return canonical_expr, var_map

    def _simplify(self, expression: str) -> Tuple[str, int]:
        """
        Simplify expression by applying algebraic rules.

        Args:
            expression: Input expression

        Returns:
            Tuple of (simplified expression, number of simplifications)
        """
        simplifications = 0
        simplified = expression

        # Apply simplification rules
        rules = [
            (r'\btrue\s+and\s+(.+?)\b', r'\1'),  # true and x -> x
            (r'\b(.+?)\s+and\s+true\b', r'\1'),  # x and true -> x
            (r'\bfalse\s+and\s+(.+?)\b', 'false'),   # false and x -> false
            (r'\b(.+?)\s+and\s+false\b', 'false'),  # x and false -> false
            (r'\btrue\s+or\s+(.+?)\b', 'true'),     # true or x -> true
            (r'\b(.+?)\s+or\s+true\b', 'true'),      # x or true -> true
            (r'\bfalse\s+or\s+(.+?)\b', r'\1'),      # false or x -> x
            (r'\b(.+?)\s+or\s+false\b', r'\1'),      # x or false -> x
            (r'\bnot\s+not\s+(.+?)\b', r'\1'),        # not not x -> x
            (r'\((.+?)\)', r'\1'),                    # (x) -> x
        ]

        for pattern, replacement in rules:
            old = simplified
            simplified = re.sub(pattern, replacement, simplified)
            if simplified != old:
                simplifications += 1

        return simplified, simplifications

    def _flatten(self, expression: str) -> str:
        """
        Flatten nested expressions.

        Args:
            expression: Input expression

        Returns:
            Flattened expression
        """
        # Flatten nested conjunctions and disjunctions
        # (and (and a b) c) -> (and a b c)
        # (or (or a b) c) -> (or a b c)

        flattened = expression

        # Flatten conjunctions
        while True:
            old = flattened
            # Match (and (and X Y) Z) -> (and X Y Z)
            flattened = re.sub(
                r'\(and\s*\(and\s+(.+?)\)\s*(.+?)\)',
                r'(and \1 \2)',
                flattened
            )
            if old == flattened:
                break

        # Flatten disjunctions
        while True:
            old = flattened
            # Match (or (or X Y) Z) -> (or X Y Z)
            flattened = re.sub(
                r'\(or\s*\(or\s+(.+?)\)\s*(.+?)\)',
                r'(or \1 \2)',
                flattened
            )
            if old == flattened:
                break

        return flattened

    def canonicalize_constraint_set(self, constraints: List[str]) -> List[str]:
        """
        Canonicalize a set of constraints together.

        This ensures consistent variable naming across all constraints.

        Args:
            constraints: List of constraint expressions

        Returns:
            List of canonicalized constraints
        """
        # Reset state for new set
        self._var_counter = 0
        self._var_map.clear()

        canonicalized = []
        for constraint in constraints:
            result = self.canonicalize(constraint)
            if result.success:
                canonicalized.append(result.canonical)
            else:
                canonicalized.append(constraint)  # Keep original on failure

        return canonicalized

    def to_smtlib(self, expression: str) -> str:
        """
        Convert expression to SMT-LIB format.

        Args:
            expression: Input expression (Python-like or Z3 Python)

        Returns:
            SMT-LIB format string
        """
        # If already in SMT-LIB format, return as-is
        if expression.strip().startswith('('):
            return expression

        # Convert Python-like syntax to SMT-LIB
        smtlib = expression

        # Convert operators
        operator_map = {
            '==': '=',
            '!=': 'distinct',
            '<=': '<=',
            '>=': '>=',
            '<': '<',
            '>': '>',
            '&&': 'and',
            '||': 'or',
            '!': 'not',
            'True': 'true',
            'False': 'false'
        }

        for python_op, smtlib_op in operator_map.items():
            smtlib = smtlib.replace(python_op, smtlib_op)

        # Wrap in assert if not already
        if not smtlib.strip().startswith('(assert '):
            smtlib = f"(assert {smtlib})"

        return smtlib

    def from_z3_ast(self, z3_ast) -> str:
        """
        Convert Z3 AST to canonical string representation.

        Args:
            z3_ast: Z3 expression AST node

        Returns:
            Canonical string representation
        """
        if not Z3_AVAILABLE:
            return str(z3_ast)

        try:
            # Use Z3's sexpr conversion
            return str(z3_ast.sexpr())
        except Exception as e:
            logger.debug(f"Failed to convert AST: {e}")
            return str(z3_ast)


def canonicalize_expression(
    expression: str,
    rules: Optional[List[CanonicalizationRule]] = None
) -> str:
    """
    Convenience function to canonicalize an expression.

    Args:
        expression: Input expression
        rules: Optional canonicalization rules

    Returns:
        Canonicalized expression
    """
    canonicalizer = Z3Canonicalizer()
    result = canonicalizer.canonicalize(expression, rules)
    return result.canonical if result.success else expression


def canonicalize_constraints(constraints: List[str]) -> List[str]:
    """
    Convenience function to canonicalize a set of constraints.

    Args:
        constraints: List of constraint expressions

    Returns:
        List of canonicalized constraints
    """
    canonicalizer = Z3Canonicalizer()
    return canonicalizer.canonicalize_constraint_set(constraints)
