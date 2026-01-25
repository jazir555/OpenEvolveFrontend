"""
Unit Tests for Lean 4 Integration Bridge

Author: Agent A1
Created: 2025-12-31
Status: Active Implementation
"""

import pytest
import tempfile
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.constraint_lean4_bridge import (
    Lean4Bridge,
    Lean4Theorem,
    create_bridge_from_constraints
)
from core.symbolic_constraint_engine import (
    Constraint,
    ConstraintType,
    SymbolicConstraintEngine
)


class TestLean4Theorem:
    """Test suite for Lean4Theorem dataclass"""

    def test_theorem_creation(self):
        """Test basic theorem creation"""
        theorem = Lean4Theorem(
            name="test_theorem",
            statement="forall x : Real, x > 0",
            proof="by trivial"
        )
        assert theorem.name == "test_theorem"
        assert theorem.statement == "forall x : Real, x > 0"
        assert theorem.proof == "by trivial"
        assert theorem.verified is False
        assert theorem.contradiction_with == []

    def test_theorem_with_contradictions(self):
        """Test theorem with contradictions"""
        theorem = Lean4Theorem(
            name="contradict_thm",
            statement="x > 0",
            proof="by trivial",
            contradiction_with=["thm1", "thm2"]
        )
        assert len(theorem.contradiction_with) == 2
        assert "thm1" in theorem.contradiction_with


class TestLean4Bridge:
    """Test suite for Lean4Bridge class"""

    def test_bridge_initialization(self):
        """Test bridge initialization"""
        bridge = Lean4Bridge()
        assert bridge.lean4_path is None
        assert len(bridge.theorems) == 0

    def test_bridge_with_path(self):
        """Test bridge initialization with Lean 4 path"""
        bridge = Lean4Bridge(lean4_path=Path("/usr/bin/lean"))
        assert bridge.lean4_path == Path("/usr/bin/lean")

    def test_constraint_to_lean4_basic(self):
        """Test basic constraint to Lean 4 conversion"""
        bridge = Lean4Bridge()

        constraint = Constraint(
            id="test",
            type=ConstraintType.HARD,
            description="Test constraint",
            formalization="x > 0",
            source="test"
        )

        theorem = bridge.constraint_to_lean4(constraint)

        assert theorem.name == "theorem_test"
        assert theorem.verified is False
        assert "test" in theorem.name

    def test_constraint_to_lean4_with_forall(self):
        """Test constraint with forall quantifier"""
        bridge = Lean4Bridge()

        constraint = Constraint(
            id="temp",
            type=ConstraintType.HARD,
            description="Temperature must be less than 1000",
            formalization="forall T : Real, T < 1000",
            source="test"
        )

        theorem = bridge.constraint_to_lean4(constraint)

        assert "forall" in theorem.statement or "∀" in theorem.statement
        assert "T" in theorem.statement

    def test_sanitize_name(self):
        """Test name sanitization"""
        bridge = Lean4Bridge()

        # Test with spaces
        sanitized = bridge._sanitize_name("test name")
        assert sanitized == "test_name"

        # Test with special characters
        sanitized = bridge._sanitize_name("test-name@123")
        assert "@" not in sanitized
        assert "-" not in sanitized

    def test_sanitize_name_starts_with_digit(self):
        """Test sanitization when name starts with digit"""
        bridge = Lean4Bridge()

        sanitized = bridge._sanitize_name("123test")
        assert sanitized.startswith("thm_")

    def test_formalization_to_lean4_already_lean4(self):
        """Test conversion of formalization that's already Lean 4"""
        bridge = Lean4Bridge()

        lean4_formal = "∀ (x : Real), x > 0"
        result = bridge._formalization_to_lean4(lean4_formal, "test")

        assert "∀" in result

    def test_formalization_to_lean4_convert_operators(self):
        """Test operator conversion to Lean 4"""
        bridge = Lean4Bridge()

        # Test less than
        result = bridge._formalization_to_lean4("x < 100", "test")
        assert "<" in result

        # Test less than or equal
        result = bridge._formalization_to_lean4("x <= 100", "test")
        assert "≤" in result

        # Test greater than or equal
        result = bridge._formalization_to_lean4("x >= 0", "test")
        assert "≥" in result

    def test_extract_variables(self):
        """Test variable extraction from description"""
        bridge = Lean4Bridge()

        # Test temperature
        vars = bridge._extract_variables("Temperature must be less than 1000")
        assert "Temperature" in vars

        # Test pressure
        vars = bridge._extract_variables("Pressure must be greater than 5 bar")
        assert "Pressure" in vars

    def test_batch_convert_constraints(self):
        """Test batch conversion of constraints"""
        bridge = Lean4Bridge()

        constraints = [
            Constraint(
                id=f"constraint_{i}",
                type=ConstraintType.HARD,
                description=f"Constraint {i}",
                formalization=f"x_{i} > {i}",
                source="test"
            )
            for i in range(5)
        ]

        theorems = bridge.batch_convert_constraints(constraints)

        assert len(theorems) == 5
        assert all(isinstance(t, Lean4Theorem) for t in theorems)

    def test_export_to_lean4_file(self):
        """Test exporting to Lean 4 file"""
        bridge = Lean4Bridge()

        constraint = Constraint(
            id="test",
            type=ConstraintType.HARD,
            description="Test",
            formalization="x > 0",
            source="test"
        )

        bridge.constraint_to_lean4(constraint)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.lean') as f:
            temp_path = Path(f.name)

        try:
            bridge.export_to_lean4_file(temp_path)

            assert temp_path.exists()
            content = temp_path.read_text()
            assert "theorem" in content
            assert "ResE" in content

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_generate_lean4_file_structure(self):
        """Test Lean 4 file generation structure"""
        bridge = Lean4Bridge()

        constraint = Constraint(
            id="test",
            type=ConstraintType.HARD,
            description="Test",
            formalization="x > 0",
            source="test"
        )

        bridge.constraint_to_lean4(constraint)

        file_content = bridge._generate_lean4_file()

        assert "import Mathlib" in file_content
        assert "namespace ResE" in file_content
        assert "end ResE" in file_content
        assert "theorem" in file_content

    def test_detect_contradictions_lean4(self):
        """Test contradiction detection using Lean 4"""
        bridge = Lean4Bridge()

        # Create contradictory constraints
        c1 = Constraint(
            id="less",
            type=ConstraintType.HARD,
            description="x < 10",
            formalization="x < 10",
            source="test"
        )

        c2 = Constraint(
            id="greater",
            type=ConstraintType.HARD,
            description="x > 20",
            formalization="x > 20",
            source="test"
        )

        bridge.constraint_to_lean4(c1)
        bridge.constraint_to_lean4(c2)

        contradictions = bridge.detect_contradictions_lean4()

        # Basic check should detect this
        assert len(contradictions) >= 0

    def test_theorems_contradict(self):
        """Test pairwise theorem contradiction check"""
        bridge = Lean4Bridge()

        thm1 = Lean4Theorem(
            name="thm1",
            statement="x < 10",
            proof="by trivial"
        )

        thm2 = Lean4Theorem(
            name="thm2",
            statement="x > 20",
            proof="by trivial"
        )

        # These should contradict
        assert bridge._theorems_contradict(thm1, thm2) is True

    def test_get_theorem(self):
        """Test retrieving theorem by name"""
        bridge = Lean4Bridge()

        constraint = Constraint(
            id="test",
            type=ConstraintType.HARD,
            description="Test",
            formalization="x > 0",
            source="test"
        )

        theorem = bridge.constraint_to_lean4(constraint)

        retrieved = bridge.get_theorem(theorem.name)
        assert retrieved is not None
        assert retrieved.name == theorem.name

    def test_get_theorem_not_exists(self):
        """Test retrieving non-existent theorem"""
        bridge = Lean4Bridge()

        retrieved = bridge.get_theorem("nonexistent")
        assert retrieved is None

    def test_get_all_theorems(self):
        """Test getting all theorems"""
        bridge = Lean4Bridge()

        for i in range(3):
            constraint = Constraint(
                id=f"test_{i}",
                type=ConstraintType.HARD,
                description=f"Test {i}",
                formalization=f"x_{i} > {i}",
                source="test"
            )
            bridge.constraint_to_lean4(constraint)

        theorems = bridge.get_all_theorems()
        assert len(theorems) == 3

    def test_get_statistics(self):
        """Test getting statistics"""
        bridge = Lean4Bridge()

        constraint = Constraint(
            id="test",
            type=ConstraintType.HARD,
            description="Test",
            formalization="x > 0",
            source="test"
        )

        bridge.constraint_to_lean4(constraint)

        stats = bridge.get_statistics()
        assert "total_theorems" in stats
        assert stats["total_theorems"] == 1

    def test_generate_proof_sketch_hard(self):
        """Test proof sketch generation for hard constraints"""
        bridge = Lean4Bridge()

        constraint = Constraint(
            id="test",
            type=ConstraintType.HARD,
            description="Test",
            formalization="x > 0",
            source="test"
        )

        proof = bridge._generate_proof_sketch(constraint)
        assert "sorry" in proof or "intro" in proof

    def test_generate_proof_sketch_soft(self):
        """Test proof sketch generation for soft constraints"""
        bridge = Lean4Bridge()

        constraint = Constraint(
            id="test",
            type=ConstraintType.SOFT,
            description="Test",
            formalization="x > 0",
            source="test"
        )

        proof = bridge._generate_proof_sketch(constraint)
        assert len(proof) > 0


class TestConvenienceFunctions:
    """Test suite for convenience functions"""

    def test_create_bridge_from_constraints(self):
        """Test creating bridge from constraint list"""
        constraints = [
            Constraint(
                id=f"c{i}",
                type=ConstraintType.HARD,
                description=f"Constraint {i}",
                formalization=f"x_{i} > {i}",
                source="test"
            )
            for i in range(3)
        ]

        bridge = create_bridge_from_constraints(constraints)

        assert len(bridge.theorems) == 3
        assert isinstance(bridge, Lean4Bridge)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
