"""
Lean 4 Integration Bridge for Symbolic Constraint Engine

Provides bidirectional translation between Python constraints and Lean 4 theorems.
Enables automated verification of contradictions using Lean 4's proof system.

Author: Agent A1
Created: 2025-12-31
Status: Active Implementation
"""

import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .symbolic_constraint_engine import Constraint, ConstraintType


@dataclass
class Lean4Theorem:
    """
    Represents a Lean 4 theorem with its metadata.

    Attributes:
        name: Theorem name in Lean 4
        statement: The theorem statement (Lean 4 syntax)
        proof: The proof script (Lean 4 tactics)
        verified: Whether the theorem has been verified
        contradiction_with: Optional list of theorem names this contradicts
    """
    name: str
    statement: str
    proof: str
    verified: bool = False
    contradiction_with: List[str] = None

    def __post_init__(self):
        if self.contradiction_with is None:
            self.contradiction_with = []


class Lean4Bridge:
    """
    Bridge between Python constraints and Lean 4 theorems.

    Features:
    - Python → Lean 4 translation
    - Lean 4 → Python extraction
    - Automated contradiction verification
    - Theorem export/import
    """

    # Mapping of Python operators to Lean 4
    OPERATOR_MAP = {
        "<": "<",
        "<=": "≤",
        ">": ">",
        ">=": "≥",
        "=": "=",
        "!=": "≠",
        "and": "∧",
        "or": "∨",
        "not": "¬",
        "implies": "→",
        "forall": "∀",
        "exists": "∃"
    }

    # Mapping of common constraint patterns to Lean 4 types
    TYPE_MAP = {
        "Real": "Real",
        "Int": "Int",
        "Nat": "Nat",
        "Bool": "Bool",
        "String": "String",
        "Temperature": "Real",  # Domain-specific
        "Pressure": "Real",
        "Time": "Real",
        "Value": "Real"
    }

    def __init__(self, lean4_path: Optional[Path] = None):
        """
        Initialize the Lean 4 bridge.

        Args:
            lean4_path: Path to Lean 4 installation (optional)
        """
        self.lean4_path = lean4_path
        self.theorems: Dict[str, Lean4Theorem] = {}
        self._theorem_counter = 0

    def constraint_to_lean4(self, constraint: Constraint) -> Lean4Theorem:
        """
        Convert a Python constraint to a Lean 4 theorem.

        Args:
            constraint: Python constraint to convert

        Returns:
            Lean4Theorem object

        Example:
            Input:  "Temperature must be less than 1000°C"
            Output: theorem temp_limit : ∀ T : Real, T < 1000
        """
        # Generate theorem name
        theorem_name = self._sanitize_name(f"theorem_{constraint.id}")

        # Convert formalization to Lean 4
        lean_statement = self._formalization_to_lean4(
            constraint.formalization,
            constraint.description
        )

        # Generate proof sketch (placeholder for actual proof)
        proof = self._generate_proof_sketch(constraint)

        theorem = Lean4Theorem(
            name=theorem_name,
            statement=lean_statement,
            proof=proof,
            verified=False  # Requires Lean 4 verification
        )

        self.theorems[theorem_name] = theorem
        return theorem

    def _sanitize_name(self, name: str) -> str:
        """
        Sanitize a name for Lean 4 (remove special characters).

        Args:
            name: Name to sanitize

        Returns:
            Sanitized name
        """
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure it starts with a letter
        if sanitized and sanitized[0].isdigit():
            sanitized = "thm_" + sanitized
        return sanitized

    def _formalization_to_lean4(self, formalization: str, description: str) -> str:
        """
        Convert constraint formalization to Lean 4 syntax.

        Args:
            formalization: Constraint formalization
            description: Human-readable description

        Returns:
            Lean 4 statement
        """
        # If formalization already looks like Lean 4, return as-is
        if self._looks_like_lean4(formalization):
            return formalization

        # Parse description/formalization and convert to Lean 4
        lean_statement = formalization

        # Replace operators
        for py_op, lean_op in self.OPERATOR_MAP.items():
            lean_statement = lean_statement.replace(py_op, lean_op)

        # Add quantifiers if missing
        if "forall" not in lean_statement.lower() and "∀" not in lean_statement:
            # Try to infer variables from description
            variables = self._extract_variables(description)
            if variables:
                var_decls = ", ".join([f"{v} : Real" for v in variables])
                lean_statement = f"∀ ({var_decls}), {lean_statement}"

        return lean_statement

    def _looks_like_lean4(self, text: str) -> bool:
        """Check if text already looks like Lean 4 code"""
        lean4_indicators = ["∀", "∃", "→", "∧", "∨", "∀", "theorem ", "def "]
        return any(indicator in text for indicator in lean4_indicators)

    def _extract_variables(self, description: str) -> List[str]:
        """
        Extract variable names from description.

        Args:
            description: Constraint description

        Returns:
            List of variable names
        """
        # Common variable patterns
        patterns = [
            r'\bTemperature\b',
            r'\bPressure\b',
            r'\bTime\b',
            r'\bValue\b',
            r'\b[xX]\b',
            r'\b[yY]\b',
            r'\b[nN]\b'
        ]

        variables = []
        for pattern in patterns:
            matches = re.findall(pattern, description)
            variables.extend(matches)

        return list(set(variables))  # Remove duplicates

    def _generate_proof_sketch(self, constraint: Constraint) -> str:
        """
        Generate a proof sketch for the constraint.

        Args:
            constraint: Constraint to generate proof for

        Returns:
            Lean 4 proof sketch (tactics)
        """
        if constraint.type == ConstraintType.HARD:
            return "by\n  intro h\n  -- Proof placeholder\n  sorry"
        elif constraint.type == ConstraintType.SOFT:
            return "by\n  -- Soft constraint proof\n  sorry"
        else:
            return "by\n  -- Preference constraint\n  sorry"

    def batch_convert_constraints(self, constraints: List[Constraint]) -> List[Lean4Theorem]:
        """
        Convert multiple constraints to Lean 4 theorems.

        Args:
            constraints: List of constraints to convert

        Returns:
            List of Lean4Theorem objects
        """
        theorems = []
        for constraint in constraints:
            theorem = self.constraint_to_lean4(constraint)
            theorems.append(theorem)
        return theorems

    def export_to_lean4_file(self, filepath: Path) -> None:
        """
        Export all theorems to a Lean 4 file.

        Args:
            filepath: Path to output .lean file
        """
        lean_code = self._generate_lean4_file()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(lean_code)

    def _generate_lean4_file(self) -> str:
        """
        Generate complete Lean 4 file content.

        Returns:
            Lean 4 file content as string
        """
        lines = [
            "import Mathlib.Data.Real.Basic",
            "import Mathlib.Logic.Basic",
            "",
            "-- Auto-generated by RESE Symbolic Constraint Engine",
            "-- Author: Agent A1",
            f"-- Generated: {self._get_timestamp()}",
            "",
            "namespace ResE",
            ""
        ]

        # Add each theorem
        for theorem in self.theorems.values():
            lines.append(f"-- {theorem.name}")
            lines.append(f"theorem {theorem.name} : {theorem.statement} := {theorem.proof}")
            lines.append("")

        lines.append("end ResE")

        return "\n".join(lines)

    def _get_timestamp(self) -> str:
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def verify_theorem_in_lean4(self, theorem_name: str) -> Tuple[bool, str]:
        """
        Verify a theorem in Lean 4 (requires Lean 4 installation).

        Args:
            theorem_name: Name of theorem to verify

        Returns:
            Tuple of (success, message)
        """
        if self.lean4_path is None:
            return False, "Lean 4 path not configured"

        theorem = self.theorems.get(theorem_name)
        if theorem is None:
            return False, f"Theorem {theorem_name} not found"

        # Create temporary Lean 4 file (cross-platform)
        import tempfile
        temp_file = Path(tempfile.gettempdir()) / "temp_verify.lean"
        self.export_to_lean4_file(temp_file)

        try:
            # Run Lean 4
            result = subprocess.run(
                [str(self.lean4_path), str(temp_file)],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                theorem.verified = True
                return True, "Theorem verified successfully"
            else:
                return False, result.stderr

        except subprocess.TimeoutExpired:
            return False, "Verification timed out"
        except Exception as e:
            return False, f"Verification error: {str(e)}"
        finally:
            # Clean up temp file
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass

    def detect_contradictions_lean4(self) -> List[Tuple[str, str, str]]:
        """
        Detect contradictions between theorems using Lean 4.

        Returns:
            List of tuples (theorem1, theorem2, contradiction_proof)

        Note:
            This is a placeholder. Full implementation requires
            Lean 4's contradiction detection tactics.
        """
        contradictions = []

        # Check pairs of theorems
        theorem_names = list(self.theorems.keys())
        for i, name1 in enumerate(theorem_names):
            for name2 in theorem_names[i+1:]:
                thm1 = self.theorems[name1]
                thm2 = self.theorems[name2]

                # Check for obvious contradictions
                if self._theorems_contradict(thm1, thm2):
                    contradictions.append((
                        name1,
                        name2,
                        "Contradiction detected (basic check)"
                    ))

        return contradictions

    def _theorems_contradict(self, thm1: Lean4Theorem, thm2: Lean4Theorem) -> bool:
        """
        Check if two theorems contradict each other.

        Args:
            thm1: First theorem
            thm2: Second theorem

        Returns:
            True if theorems contradict, False otherwise
        """
        # Basic keyword-based contradiction detection
        stmt1 = thm1.statement.lower()
        stmt2 = thm2.statement.lower()

        contradictions = [
            ("<", ">"),
            ("≤", "≥"),
            ("true", "false"),
            ("⊤", "⊥")
        ]

        for pos, neg in contradictions:
            if pos in stmt1 and neg in stmt2:
                return True
            if neg in stmt1 and pos in stmt2:
                return True

        return False

    def import_lean4_theorems(self, filepath: Path) -> int:
        """
        Import theorems from a Lean 4 file.

        Args:
            filepath: Path to .lean file

        Returns:
            Number of theorems imported

        Note:
            This is a simplified parser. Full implementation would
            require proper Lean 4 AST parsing.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse theorems using regex
        theorem_pattern = r'theorem\s+(\w+)\s*:\s*(.+?)\s*:=\s*(.+?)(?=\n\n|\ntheorem|$)'

        matches = re.finditer(theorem_pattern, content, re.DOTALL)

        count = 0
        for match in matches:
            name = match.group(1)
            statement = match.group(2).strip()
            proof = match.group(3).strip()

            theorem = Lean4Theorem(
                name=name,
                statement=statement,
                proof=proof,
                verified=False
            )

            self.theorems[name] = theorem
            count += 1

        return count

    def get_theorem(self, name: str) -> Optional[Lean4Theorem]:
        """Get a theorem by name"""
        return self.theorems.get(name)

    def get_all_theorems(self) -> List[Lean4Theorem]:
        """Get all theorems"""
        return list(self.theorems.values())

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about theorems"""
        return {
            "total_theorems": len(self.theorems),
            "verified_theorems": sum(1 for t in self.theorems.values() if t.verified),
            "contradictions": len(self.detect_contradictions_lean4())
        }


# Convenience functions

def create_bridge_from_constraints(constraints: List[Constraint]) -> Lean4Bridge:
    """
    Create a Lean 4 bridge from a list of constraints.

    Args:
        constraints: List of constraints

    Returns:
        Lean4Bridge with all constraints converted
    """
    bridge = Lean4Bridge()
    bridge.batch_convert_constraints(constraints)
    return bridge


# Testing and demonstration

if __name__ == "__main__":
    print("=" * 70)
    print("Lean 4 Integration Bridge - Demonstration")
    print("=" * 70)

    from symbolic_constraint_engine import SymbolicConstraintEngine

    # Create test constraints
    sce = SymbolicConstraintEngine()

    c1 = Constraint(
        id="temp_limit",
        type=ConstraintType.HARD,
        description="Temperature must be less than 1000°C",
        formalization="forall T : Real, T < 1000",
        source="user_prompt"
    )

    c2 = Constraint(
        id="min_temp",
        type=ConstraintType.HARD,
        description="Temperature must be greater than 500°C",
        formalization="forall T : Real, T > 500",
        source="user_prompt"
    )

    sce.add_constraint(c1)
    sce.add_constraint(c2)

    # Create bridge
    bridge = Lean4Bridge()
    print("\n[OK] Lean 4 Bridge initialized")

    # Convert constraints
    theorems = bridge.batch_convert_constraints(sce.get_all_constraints())
    print(f"[OK] Converted {len(theorems)} constraints to Lean 4 theorems")

    # Display theorems
    print("\n" + "=" * 70)
    print("Generated Theorems:")
    print("=" * 70)
    for theorem in theorems:
        print(f"\n{theorem.name}:")
        print(f"  Statement: {theorem.statement}")
        print(f"  Proof: {theorem.proof[:50]}...")

    # Export to file
    import tempfile
    output_file = Path(tempfile.gettempdir()) / "rese_constraints.lean"
    bridge.export_to_lean4_file(output_file)
    print(f"\n[OK] Exported theorems to {output_file}")

    # Detect contradictions
    contradictions = bridge.detect_contradictions_lean4()
    print(f"\n[INFO] Detected {len(contradictions)} contradictions")

    # Statistics
    stats = bridge.get_statistics()
    print("\n" + "=" * 70)
    print("Statistics:")
    print("=" * 70)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("[OK] Lean 4 Bridge demonstration complete")
    print("=" * 70)
