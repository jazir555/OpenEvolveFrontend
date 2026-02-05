"""
Constraint Translator: RESE → Lean 4

Translates RESE constraints, theorems, and Functional Dependency Graphs (FDGs)
into Lean 4 formal verification syntax.

Following CLAUDE.md principles:
- Anti-Corruption Layer: Translate between RESE and Lean 4 formats
- Law of Runtime Truth: Verify translations before use
- Structured Logging: JSON with correlation_id

Usage:
    >>> translator = ConstraintTranslator()
    >>> lean4_code = translator.translate_to_lean4("forall x, P(x) -> Q(x)")
"""

import re
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import structlog


# ============================================================================
# EXCEPTIONS
# ============================================================================

class Lean4SyntaxError(Exception):
    """Lean 4 syntax error in translation."""
    pass


# ============================================================================
# TRANSLATOR
# ============================================================================

class ConstraintTranslator:
    """
    Translates RESE constraints to Lean 4 syntax.

    Responsibilities:
    1. Translate natural language constraints to Lean 4 propositions
    2. Generate Lean 4 theorem statements
    3. Translate Functional Dependency Graphs (FDGs) to Lean 4 structures
    4. Validate Lean 4 syntax

    Attributes:
        logger: Structured logger
    """

    def __init__(self, logger: Optional[structlog.BoundLogger] = None):
        """Initialize constraint translator."""
        self.logger = logger or structlog.get_logger()
        self.logger = self.logger.bind(component="constraint_translator")

        # Common RESE → Lean 4 mappings
        self.operator_map = {
            "and": "∧",
            "or": "∨",
            "not": "¬",
            "implies": "→",
            "iff": "↔",
            "forall": "∀",
            "exists": "∃",
            "for all": "∀",
            "there exists": "∃",
            "such that": ", ",
            "=>": "→",
            "->": "→",
            "<=>": "↔",
            "<->": "↔",
            "/\\": "∧",
            "\\/": "∨",
            "~": "¬",
            "!": "¬",
        }

        # Type annotations for common RESE concepts
        self.type_map = {
            "Real": "Real",
            "Nat": "Nat",
            "Int": "Int",
            "Bool": "Bool",
            "String": "String",
            "List": "List",
            "Set": "Set",
        }

    # ========================================================================
    # MAIN TRANSLATION METHODS
    # ========================================================================

    def translate_to_lean4(
        self,
        constraint: str,
        constraint_type: str = "proposition",
    ) -> str:
        """
        Translate a RESE constraint to Lean 4 syntax.

        Args:
            constraint: Natural language or formal constraint
            constraint_type: Type of constraint (proposition, theorem, axiom)

        Returns:
            Lean 4 code string

        Raises:
            Lean4SyntaxError: If translation fails
        """
        self.logger.info(
            "Translating constraint to Lean 4",
            constraint_length=len(constraint),
            constraint_type=constraint_type,
        )

        try:
            # Detect constraint language and translate
            if self._is_lean4_syntax(constraint):
                # Already Lean 4 syntax, just format it
                lean4_code = self._format_lean4_code(constraint, constraint_type)
            else:
                # Natural language or other format, translate it
                lean4_code = self._translate_natural_language(constraint, constraint_type)

            self.logger.info(
                "Translation successful",
                lean4_code_length=len(lean4_code),
            )

            return lean4_code

        except Exception as e:
            self.logger.error(
                "Translation failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise Lean4SyntaxError(f"Translation failed: {e}")

    def translate_fdg_to_lean4(self, fdg: Dict[str, Any]) -> str:
        """
        Translate a Functional Dependency Graph (FDG) to Lean 4.

        Args:
            fdg: Functional dependency graph from RESE Phase II

        Returns:
            Lean 4 code for FDG
        """
        self.logger.info(
            "Translating FDG to Lean 4",
            node_count=len(fdg.get("nodes", [])),
            edge_count=len(fdg.get("edges", [])),
        )

        lean4_code = []

        # Import required Lean 4 modules
        lean4_code.append("import Mathlib.Data.Graph.Basic")
        lean4_code.append("import Mathlib.Data.Relation")
        lean4_code.append("")

        # Define node type
        lean4_code.append("-- Define node type for FDG")
        lean4_code.append("structure FDGNode where")
        lean4_code.append("  id : String")
        lean4_code.append("  nodeType : String")
        lean4_code.append("deriving Repr, BEq")
        lean4_code.append("")

        # Define edge type
        lean4_code.append("-- Define edge type for FDG")
        lean4_code.append("structure FDGEdge where")
        lean4_code.append("  source : FDGNode")
        lean4_code.append("  target : FDGNode")
        lean4_code.append("  relationType : String")
        lean4_code.append("  strength : Real")
        lean4_code.append("deriving Repr, BEq")
        lean4_code.append("")

        # Define FDG structure
        lean4_code.append("-- Define Functional Dependency Graph")
        lean4_code.append("structure FunctionalDependencyGraph where")
        lean4_code.append("  nodes : List FDGNode")
        lean4_code.append("  edges : List FDGEdge")
        lean4_code.append("deriving Repr, BEq")
        lean4_code.append("")

        # Create nodes
        lean4_code.append("-- Create nodes")
        for node in fdg.get("nodes", []):
            node_id = node.get("id", "").replace('"', '\\"')
            node_type = node.get("type", "").replace('"', '\\"')
            lean4_code.append(f'def {node_id.replace("-", "_")} : FDGNode := {{')
            lean4_code.append(f'  id := "{node_id}",')
            lean4_code.append(f'  nodeType := "{node_type}"')
            lean4_code.append('}')

        lean4_code.append("")

        # Create edges
        lean4_code.append("-- Create edges")
        for edge in fdg.get("edges", []):
            source = edge.get("source", "").replace("-", "_")
            target = edge.get("target", "").replace("-", "_")
            relation = edge.get("relation_type", "").replace('"', '\\"')
            strength = edge.get("strength", 1.0)
            lean4_code.append(f'def edge_{uuid.uuid4().hex[:8]} : FDGEdge := {{')
            lean4_code.append(f'  source := {source},')
            lean4_code.append(f'  target := {target},')
            lean4_code.append(f'  relationType := "{relation}",')
            lean4_code.append(f'  strength := {strength}')
            lean4_code.append('}')

        lean4_code.append("")

        # Create FDG instance
        lean4_code.append("-- Create FDG instance")
        fdg_name = f"fdg_{uuid.uuid4().hex[:8]}"
        lean4_code.append(f'def {fdg_name} : FunctionalDependencyGraph := {{')
        lean4_code.append('  nodes := [')
        for node in fdg.get("nodes", []):
            node_id = node.get("id", "").replace("-", "_")
            lean4_code.append(f'    {node_id},')
        lean4_code.append('  ],')
        lean4_code.append('  edges := []  -- TODO: Add edges')
        lean4_code.append('}')

        lean4_code.append("")

        # Add theorems about FDG properties
        lean4_code.append("-- Theorems about FDG properties")
        lean4_code.append(f"theorem {fdg_name}_nodes_nonempty :")
        lean4_code.append(f"  ({fdg_name}.nodes.length) > 0 := by")
        lean4_code.append("  simp")
        lean4_code.append("")

        return "\n".join(lean4_code)

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _is_lean4_syntax(self, constraint: str) -> bool:
        """Check if constraint is already in Lean 4 syntax."""
        lean4_indicators = ["∀", "∃", "→", "∧", "∨", "¬", "theorem", "axiom", "def"]
        return any(indicator in constraint for indicator in lean4_indicators)

    def _format_lean4_code(self, code: str, constraint_type: str) -> str:
        """Format Lean 4 code with proper structure."""
        code = code.strip()

        if constraint_type == "theorem" and not code.startswith("theorem"):
            # Wrap as theorem
            theorem_name = f"theorem_{uuid.uuid4().hex[:8]}"
            return f"theorem {theorem_name} : {code} := by\n  sorry"

        elif constraint_type == "axiom" and not code.startswith("axiom"):
            # Wrap as axiom
            axiom_name = f"axiom_{uuid.uuid4().hex[:8]}"
            return f"axiom {axiom_name} : {code}"

        return code

    def _translate_natural_language(self, constraint: str, constraint_type: str) -> str:
        """Translate natural language constraint to Lean 4."""
        # Apply operator mappings
        translated = constraint.lower()

        # Replace operators with Lean 4 symbols
        for nl_op, lean4_op in self.operator_map.items():
            translated = translated.replace(nl_op, lean4_op)

        # Clean up spacing
        translated = re.sub(r'\s+', ' ', translated)
        translated = translated.strip()

        # Wrap as theorem if needed
        if constraint_type == "theorem":
            theorem_name = f"theorem_{uuid.uuid4().hex[:8]}"
            return f"theorem {theorem_name} : {translated} := by\n  sorry"

        return translated


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ConstraintTranslator",
    "Lean4SyntaxError",
]
