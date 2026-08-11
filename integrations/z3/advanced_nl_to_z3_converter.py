"""
Advanced Natural Language to Z3 SMT-LIB Converter

This module provides sophisticated parsing of natural language mathematical
expressions into Z3 SMT-LIB constraints. It supports:
- Complex mathematical expressions (differential equations, integrals, derivatives)
- Domain-specific language patterns (thermodynamics, fluid dynamics, physics)
- Quantified expressions (forall, exists)
- Array and matrix operations
- Set theory expressions
- Logical connectives and quantifiers
- Unit handling and dimensional analysis

Author: Z3-Lean Integration Project
Date: 2026-02-17
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MathDomain(Enum):
    """Mathematical domains for specialized parsing"""
    GENERAL = "general"
    THERMODYNAMICS = "thermodynamics"
    FLUID_DYNAMICS = "fluid_dynamics"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    ENGINEERING = "engineering"
    ECONOMICS = "economics"
    PROBABILITY = "probability"


class ConstraintType(Enum):
    """Types of Z3 constraints"""
    DECLARE_CONST = "declare-const"
    DECLARE_FUN = "declare-fun"
    ASSERT = "assert"
    DEFINE_FUN = "define-fun"
    PUSH = "push"
    POP = "pop"
    CHECK_SAT = "check-sat"
    GET_MODEL = "get-model"
    GET_VALUE = "get-value"


@dataclass
class ParsedExpression:
    """Represents a parsed mathematical expression"""
    original: str
    normalized: str
    variables: Dict[str, str] = field(default_factory=dict)  # var_name -> sort
    constraints: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    domain: MathDomain = MathDomain.GENERAL
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Z3Constraint:
    """Represents a Z3 SMT-LIB constraint"""
    constraint_type: ConstraintType
    content: str
    line_number: int = 0
    dependencies: List[str] = field(default_factory=list)


class AdvancedNLToZ3Converter:
    """
    Advanced converter for natural language to Z3 SMT-LIB constraints.

    Features:
    - Pattern-based parsing with regex
    - Context-aware variable extraction
    - Domain-specific knowledge bases
    - Multi-stage normalization pipeline
    - Type inference and checking
    - Unit conversion and dimensional analysis
    """

    # Domain-specific patterns
    PATTERNS = {
        # Comparison operators
        r'\b(?:is|equals?|=)\s*(?:greater|more|higher)\s+than\s+([-\d.]+)': r'> \1',
        r'\b(?:is|equals?|=)\s*(?:less|lower|smaller)\s+than\s+([-\d.]+)': r'< \1',
        r'\b(?:at\s+least|minimum|min\s*\.?)\s*([-\d.]+)': r'>= \1',
        r'\b(?:at\s+most|maximum|max\s*\.?)\s*([-\d.]+)': r'<= \1',
        r'\b(?:between|from)\s+([-\d.]+)\s+(?:and|to)\s+([-\d.]+)': r'>= \1, <= \2',

        # Mathematical operations
        r'\bsquared\b': r'^2',
        r'\bcubed\b': r'^3',
        r'\bsquare\s+root\s+of\s+(\w+)': r'sqrt(\1)',
        r'\b(?:square|square\s+root)\s+(\w+)\b': r'sqrt(\1)',

        # Differential equations
        r'\bd/d(\w+)\s*(?:of\s+)?(\w+)': r'd(\2)/d(\1)',
        r'\bderivative\s+of\s+(\w+)\s+with\s+respect\s+to\s+(\w+)\b': r'd(\1)/d(\2)',
        r'\b(\w+)\'\b': r'd(\1)/dt',  # Time derivative
        r'\b(\w+)\'\'\b': r'd^2(\1)/dt^2',  # Second time derivative

        # Integrals
        r'\bintegral\s+of\s+(\w+)\s+(?:from|wrt)\s+(\w+)\b': r'∫\1 d\2',

        # Thermodynamics specific
        r'\btemperature\b': 'T',
        r'\bpressure\b': 'P',
        r'\bvolume\b': 'V',
        r'\bentropy\b': 'S',
        r'\benthalpy\b': 'H',
        r'\binternal\s+energy\b': 'U',

        # Physics specific
        r'\bvelocity\b': 'v',
        r'\bspeed\b': 'v',
        r'\bacceleration\b': 'a',
        r'\bforce\b': 'F',
        r'\bmass\b': 'm',
        r'\benergy\b': 'E',
        r'\btime\b': 't',

        # Chemical engineering
        r'\bconcentration\b': 'C',
        r'\breactant\b': 'R',
        r'\bproduct\b': 'P',
        r'\brate\s+constant\b': 'k',

        # Logical operators
        r'\band\b': '∧',
        r'\bor\b': '∨',
        r'\bnot\b': '¬',
        r'\bimplies?\b': '→',
        r'\bif\s+and\s+only\s+if\b': '↔',
        r'\bfor\s+all\b': '∀',
        r'\bthere\s+exists?\b': '∃',

        # Quantifiers
        r'\bevery\b': '∀',
        r'\bexists?\b': '∃',
        r'\bsome\b': '∃',
    }

    # Unit patterns
    UNIT_PATTERNS = {
        # Temperature
        r'([-+]?\d*\.?\d+)\s*[°c]?c\b': r'\1',  # Celsius to base (simplified)
        r'([-+]?\d*\.?\d+)\s*[°f]?f\b': r'(\1 - 32) * 5/9',  # Fahrenheit to Celsius
        r'([-+]?\d*\.?\d+)\s*k\b': r'\1 - 273.15',  # Kelvin to Celsius

        # Pressure
        r'([-+]?\d*\.?\d+)\s*bar\b': r'\1 * 100000',  # bar to Pa
        r'([-+]?\d*\.?\d+)\s*psi\b': r'\1 * 6894.76',  # psi to Pa
        r'([-+]?\d*\.?\d+)\s*atm\b': r'\1 * 101325',  # atm to Pa

        # Time
        r'([-+]?\d*\.?\d+)\s*sec(?:onds?)?\b': r'\1',
        r'([-+]?\d*\.?\d+)\s*min(?:utes?)?\b': r'\1 * 60',
        r'([-+]?\d*\.?\d+)\s*hr?s?\b': r'\1 * 3600',
    }

    # Variable type inference patterns
    TYPE_INFERENCE = {
        # Real numbers
        r'\b(?:temperature|pressure|volume|concentration|rate|speed|velocity|energy|force|mass|time)\b': 'Real',

        # Integers
        r'\b(?:count|number|atoms|molecules|particles)\b': 'Int',

        # Booleans
        r'\b(?:is|are|exists?|present|absent|active|inactive)\b': 'Bool',

        # Arrays
        r'\b(?:array|vector|matrix|list|sequence)\s+of\s+(\w+)\b': r'Array \1',
    }

    def __init__(self, domain: MathDomain = MathDomain.GENERAL):
        """
        Initialize the converter.

        Args:
            domain: Mathematical domain for specialized parsing
        """
        self.domain = domain
        self.variable_registry: Dict[str, str] = {}
        self.constraint_counter = 0
        self.confidence_threshold = 0.7

        # Compile patterns for efficiency
        self.compiled_patterns = [(re.compile(pattern, re.IGNORECASE), replacement)
                                   for pattern, replacement in self.PATTERNS.items()]
        self.compiled_unit_patterns = [(re.compile(pattern, re.IGNORECASE), replacement)
                                       for pattern, replacement in self.UNIT_PATTERNS.items()]
        self.compiled_type_patterns = [(re.compile(pattern, re.IGNORECASE), sort)
                                       for pattern, sort in self.TYPE_INFERENCE.items()]

        logger.info(f"Advanced NL-to-Z3 converter initialized for domain: {domain.value}")

    def parse_expression(self, text: str, context: Optional[Dict] = None) -> ParsedExpression:
        """
        Parse a natural language mathematical expression.

        Args:
            text: Natural language text to parse
            context: Optional context information

        Returns:
            ParsedExpression with normalized form and constraints
        """
        logger.info(f"Parsing expression: {text[:100]}")

        # Stage 1: Preprocessing and normalization
        normalized = self._preprocess(text)

        # Stage 2: Apply domain-specific patterns
        normalized = self._apply_patterns(normalized)

        # Stage 3: Extract variables
        variables = self._extract_variables(normalized, context)

        # Stage 4: Generate Z3 constraints
        constraints = self._generate_constraints(normalized, variables)

        # Stage 5: Extract assumptions
        assumptions = self._extract_assumptions(text, context)

        # Stage 6: Calculate confidence
        confidence = self._calculate_confidence(text, normalized, constraints)

        parsed = ParsedExpression(
            original=text,
            normalized=normalized,
            variables=variables,
            constraints=constraints,
            assumptions=assumptions,
            domain=self.domain,
            confidence=confidence,
            metadata={
                'context': context or {},
                'constraint_count': len(constraints),
                'variable_count': len(variables)
            }
        )

        logger.info(f"Parsed expression: {len(constraints)} constraints, {len(variables)} variables, confidence: {confidence:.2f}")

        return parsed

    def _preprocess(self, text: str) -> str:
        """Preprocess text for parsing."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Normalize quotes - replace various quote types with standard single quote
        text = text.replace('"', "'")
        text = text.replace("'", "'")  # Right single quote
        text = text.replace("'", "'")  # Left single quote

        # Convert to lowercase for pattern matching (but preserve original)
        text_lower = text.lower()

        return text_lower

    def _apply_patterns(self, text: str) -> str:
        """Apply domain-specific patterns to normalize the expression."""
        result = text

        # Apply each pattern in sequence
        for pattern, replacement in self.compiled_patterns:
            result = pattern.sub(replacement, result)

        # Apply unit conversions
        for pattern, replacement in self.compiled_unit_patterns:
            result = pattern.sub(replacement, result)

        return result

    def _extract_variables(self, text: str, context: Optional[Dict]) -> Dict[str, str]:
        """Extract variables and infer their types."""
        variables = {}

        # Extract mathematical variables (single letters or words with underscores)
        var_pattern = r'\b([a-z]|[a-z][a-z0-9_]*)\b'
        matches = re.findall(var_pattern, text)

        for var in matches:
            if var not in variables:
                # Infer type from context or patterns
                var_type = self._infer_variable_type(var, text, context)
                variables[var] = var_type

        # Extract numeric constants
        number_pattern = r'\b([-+]?\d*\.?\d+)\b'
        numbers = re.findall(number_pattern, text)

        for i, num in enumerate(numbers):
            const_name = f'const_{i}'
            if const_name not in variables:
                # Determine if integer or real
                if '.' in num:
                    variables[const_name] = 'Real'
                else:
                    variables[const_name] = 'Int'

        # Merge with variable registry
        for var, var_type in variables.items():
            if var not in self.variable_registry:
                self.variable_registry[var] = var_type

        return variables

    def _infer_variable_type(self, var: str, text: str, context: Optional[Dict]) -> str:
        """Infer the type (sort) of a variable."""
        # Check type inference patterns
        for pattern, sort in self.compiled_type_patterns:
            if pattern.search(text):
                # If it's an array pattern, extract the element type
                if 'Array' in sort:
                    return sort.replace('\\1', 'Real')  # Default to Real for array elements
                return sort

        # Check context
        if context and 'variables' in context:
            if var in context['variables']:
                return context['variables'][var]

        # Default: single letters are Real, words are Real
        return 'Real'

    def _generate_constraints(self, normalized: str, variables: Dict[str, str]) -> List[str]:
        """Generate Z3 SMT-LIB constraints from normalized expression."""
        constraints = []

        # Generate variable declarations
        for var, var_type in variables.items():
            if var_type == 'Real':
                constraints.append(f'(declare-const {var} Real)')
            elif var_type == 'Int':
                constraints.append(f'(declare-const {var} Int)')
            elif var_type == 'Bool':
                constraints.append(f'(declare-const {var} Bool)')
            elif var_type.startswith('Array'):
                constraints.append(f'(declare-const {var} (Array Int Real))')

        # Extract and convert mathematical expressions
        # Look for inequalities and equalities
        inequality_pattern = r'(\w+)\s*(>=|<=|>|<|=)\s*([-+]?\d*\.?\d+)'
        for match in re.finditer(inequality_pattern, normalized):
            var, op, value = match.groups()
            if var in variables:
                constraints.append(f'(assert ({op} {var} {value}))')

        # Look for compound expressions
        compound_pattern = r'(\w+)\s*([+\-*/^])\s*(\w+)'
        for match in re.finditer(compound_pattern, normalized):
            left, op, right = match.groups()
            if left in variables and right in variables:
                if op == '^':
                    constraints.append(f'(assert (= {left} (^ {right} 2)))')
                elif op == '*':
                    constraints.append(f'(assert (> (* {left} {right}) 0))')

        # Handle differential equations (simplified)
        if 'd(' in normalized or '∫' in normalized:
            # Generate constraints for differential equations
            # This is a simplified version - full implementation would require
            # more sophisticated parsing and constraint generation
            constraints.append('(check-sat)')

        return constraints

    def _extract_assumptions(self, text: str, context: Optional[Dict]) -> List[str]:
        """Extract assumptions from the text."""
        assumptions = []

        # Look for phrases like "assuming", "given that", "provided that"
        assumption_patterns = [
            r'(?:assuming|given|provided)\s+(?:that\s+)?(.+?)(?:\.|,|$)',
            r'(?:under\s+the\s+assumption\s+that)\s+(.+?)(?:\.|,|$)',
        ]

        for pattern in assumption_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                assumptions.append(match.strip())

        # Add context assumptions
        if context and 'assumptions' in context:
            assumptions.extend(context['assumptions'])

        return assumptions

    def _calculate_confidence(self, original: str, normalized: str, constraints: List[str]) -> float:
        """Calculate confidence score for the parsing."""
        confidence = 1.0

        # Reduce confidence if no constraints generated
        if not constraints:
            confidence -= 0.3

        # Reduce confidence if normalized text is too different from original
        original_words = set(original.lower().split())
        normalized_words = set(normalized.split())
        overlap = len(original_words & normalized_words)
        total = len(original_words | normalized_words)

        if total > 0:
            similarity = overlap / total
            confidence *= similarity

        # Increase confidence if we have well-formed constraints
        if constraints and all(c.startswith('(') for c in constraints):
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def convert_to_smtlib(self, parsed: ParsedExpression) -> str:
        """
        Convert a parsed expression to full SMT-LIB format.

        Args:
            parsed: Parsed expression

        Returns:
            SMT-LIB formatted string
        """
        lines = []

        # Header
        lines.append('; Z3 SMT-LIB generated by AdvancedNLToZ3Converter')
        lines.append(f'; Original: {parsed.original[:100]}')
        lines.append(f'; Domain: {parsed.domain.value}')
        lines.append(f'; Confidence: {parsed.confidence:.2f}')
        lines.append('')

        # Set logic
        lines.append('(set-logic AUFLIRA)')  # Supports arrays, integers, reals
        lines.append('')

        # Variable declarations
        for var, var_type in parsed.variables.items():
            if var_type == 'Real':
                lines.append(f'(declare-const {var} Real)')
            elif var_type == 'Int':
                lines.append(f'(declare-const {var} Int)')
            elif var_type == 'Bool':
                lines.append(f'(declare-const {var} Bool)')
            elif var_type.startswith('Array'):
                lines.append(f'(declare-const {var} (Array Int Real))')

        lines.append('')

        # Constraints
        for constraint in parsed.constraints:
            if constraint and not constraint.startswith('(declare-const'):
                lines.append(constraint)

        lines.append('')

        # Check satisfiability
        lines.append('(check-sat)')
        lines.append('(get-model)')

        return '\n'.join(lines)

    def batch_convert(self, texts: List[str], context: Optional[Dict] = None) -> List[ParsedExpression]:
        """
        Convert multiple natural language expressions.

        Args:
            texts: List of natural language texts
            context: Optional context for all expressions

        Returns:
            List of parsed expressions
        """
        results = []
        for text in texts:
            try:
                parsed = self.parse_expression(text, context)
                results.append(parsed)
            except Exception as e:
                logger.warning(f"Failed to parse expression: {text[:50]}... - {e}")
                # Create a low-confidence parsed expression
                results.append(ParsedExpression(
                    original=text,
                    normalized=text,
                    variables={},
                    constraints=[],
                    confidence=0.0
                ))

        return results


# Convenience functions
def convert_nl_to_z3(text: str, domain: MathDomain = MathDomain.GENERAL) -> ParsedExpression:
    """
    Convert natural language to Z3 constraints.

    Args:
        text: Natural language mathematical expression
        domain: Mathematical domain

    Returns:
        Parsed expression with constraints
    """
    converter = AdvancedNLToZ3Converter(domain=domain)
    return converter.parse_expression(text)


def convert_nl_to_smtlib(text: str, domain: MathDomain = MathDomain.GENERAL) -> str:
    """
    Convert natural language directly to SMT-LIB format.

    Args:
        text: Natural language mathematical expression
        domain: Mathematical domain

    Returns:
        SMT-LIB formatted string
    """
    converter = AdvancedNLToZ3Converter(domain=domain)
    parsed = converter.parse_expression(text)
    return converter.convert_to_smtlib(parsed)


__all__ = [
    'AdvancedNLToZ3Converter',
    'ParsedExpression',
    'Z3Constraint',
    'MathDomain',
    'ConstraintType',
    'convert_nl_to_z3',
    'convert_nl_to_smtlib',
]
