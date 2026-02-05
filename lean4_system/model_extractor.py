"""
Mathematical Model Extractor for Lean4 Formal Verification

This module extracts mathematical models and proof obligations from Python code
for formal verification in Lean4.
"""

import ast
import re
from typing import List, Optional
from .lean4_data_models import ProofObligation


class MathematicalModelExtractor:
    """
    Extracts mathematical models and proof obligations from code.

    This class analyzes Python code to identify functions and properties
    that need formal verification, generating appropriate proof obligations.
    """

    def __init__(self):
        """Initialize the MathematicalModelExtractor."""
        self.function_counter = 0

    def extract(self, solution_content: str, properties: List[str]) -> List[ProofObligation]:
        """
        Extract proof obligations from solution content.

        Args:
            solution_content: Python code to analyze
            properties: List of properties to verify (e.g., ['correctness', 'termination'])

        Returns:
            List of ProofObligation objects
        """
        if not solution_content or not solution_content.strip():
            # Return default obligation for empty content
            return [ProofObligation(
                name="overall_content_correctness",
                statement="content_is_correct",
                property_type="correctness"
            )]

        # Try to parse as Python code
        try:
            tree = ast.parse(solution_content)
            functions = self._extract_functions(tree)

            if not functions:
                # No functions found, create overall obligation
                return [ProofObligation(
                    name="overall_content_correctness",
                    statement="content_is_correct",
                    property_type="correctness"
                )]

            # Create obligations for each function and property combination
            obligations = []
            for func_name in functions:
                for prop in properties:
                    self.function_counter += 1
                    obligation = ProofObligation(
                        name=f"{prop}_of_{func_name}_{self.function_counter - 1}",
                        statement=f"Formal verification of {prop} for function {func_name}",
                        property_type=prop,
                        function_name=func_name,
                        metadata={'original_function': func_name}
                    )
                    obligations.append(obligation)

            return obligations

        except SyntaxError:
            # If parsing fails, create overall obligation
            return [ProofObligation(
                name="overall_content_correctness",
                statement="content_is_correct",
                property_type="correctness"
            )]

    def _extract_functions(self, tree: ast.AST) -> List[str]:
        """
        Extract function names from AST.

        Args:
            tree: AST tree

        Returns:
            List of function names
        """
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                functions.append(node.name)
        return functions

    def extract_specifications(self, code: str) -> List[str]:
        """
        Extract formal specifications from code comments or docstrings.

        Args:
            code: Source code

        Returns:
            List of specification strings
        """
        specs = []
        # Look for specification patterns in comments
        spec_patterns = [
            r'#\s*SPEC:\s*(.*)',
            r'#\s*REQUIRES:\s*(.*)',
            r'#\s*ENSURES:\s*(.*)',
        ]

        for pattern in spec_patterns:
            matches = re.findall(pattern, code, re.MULTILINE)
            specs.extend(matches)

        return specs

    def identify_loop_invariants(self, code: str) -> List[str]:
        """
        Identify potential loop invariants in code.

        Args:
            code: Source code

        Returns:
            List of loop invariant suggestions
        """
        invariants = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    invariants.append(f"Loop invariant for loop at line {node.lineno}")
                elif isinstance(node, ast.While):
                    invariants.append(f"Loop invariant for while loop at line {node.lineno}")
        except SyntaxError:
            pass
        return invariants
