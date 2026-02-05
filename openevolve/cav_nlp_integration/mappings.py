"""
CAV-NLP Integration Mappings Module

This module preserves type and operator mappings from the Z3-Lean bridge
while adding CAV-NLP canonicalization rules for mathematical equivalence
validation.

Author: OpenEvolve Team
"""

# =============================================================================
# Z3 to Lean Type Mappings
# =============================================================================

Z3_TO_LEAN_TYPES = {
    "Bool": "Prop",
    "Int": "ℤ",
    "Real": "ℝ",
    "Array": "Array"
}

LEAN_TO_Z3_TYPES = {
    "Prop": "Bool",
    "ℤ": "Int",
    "ℝ": "Real",
    "Int": "Int",
    "Real": "Real",
    "Bool": "Bool"
}

# =============================================================================
# Operator Mappings
# =============================================================================

Z3_TO_LEAN_OPERATORS = {
    "And": "∧",
    "Or": "∨",
    "Not": "¬",
    "Implies": "->",
    "Eq": "=",
    "Lt": "<",
    "Le": "≤",
    "Gt": ">",
    "Ge": "≥",
    "Add": "+",
    "Sub": "-",
    "Mul": "*",
    "Div": "/",
    "Mod": "%",
    "Neg": "-"
}

LEAN_TO_Z3_OPERATORS = {v: k for k, v in Z3_TO_LEAN_OPERATORS.items()}

# =============================================================================
# Tactic Selection Map
# =============================================================================

CONSTRAINT_TYPE_TACTICS = {
    "boolean": ["tauto"],
    "arithmetic": ["linarith"],
    "nonlinear": ["nlinarith", "ring_nf"],
    "array": ["simp", "aesop"],
    "quantified": ["intro", "simp"],
    "bitvector": ["bv_decide", "bv_normalize"]
}

# =============================================================================
# CAV-NLP Canonicalization Rules
# =============================================================================

# Mathematical equivalence rules that CAV-NLP validates with Z3
CANONICALIZATION_RULES = {
    "commutativity_add": "x + y == y + x",
    "commutativity_mul": "x * y == y * x",
    "associativity_add": "(x + y) + z == x + (y + z)",
    "associativity_mul": "(x * y) * z == x * (y * z)",
    "distributivity": "x * (y + z) == x * y + x * z",
    "de_morgan_and": "¬(A ∧ B) == ¬A ∨ ¬B",
    "de_morgan_or": "¬(A ∨ B) == ¬A ∧ ¬B",
    "double_negation": "¬¬A == A",
    "idempotent_and": "A ∧ A == A",
    "idempotent_or": "A ∨ A == A"
}

# Order of normalization for canonical form
CANONICALIZATION_ORDER = [
    "eliminate_implications",
    "push_negations",
    "distribute_and_over_or",
    "sort_operands_commutative",
    "normalize_arithmetic"
]

# =============================================================================
# Import Requirements Map
# =============================================================================

LEAN_IMPORTS_BY_TYPE = {
    "boolean": ["import Mathlib"],
    "arithmetic": ["import Mathlib"],
    "nonlinear": ["import Mathlib", "open Real"],
    "array": ["import Mathlib", "import Std.Data.Array"],
    "quantified": ["import Mathlib"],
    "bitvector": ["import Mathlib", "import Std.Data.BitVec"]
}
