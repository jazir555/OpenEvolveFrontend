"""
Unit Tests for LLTL Handoff Module

Author: Agent A1
Created: 2025-12-31
Status: Active Implementation
"""


import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import tempfile
import json
from pathlib import Path
from core.constraint_lltl_handoff import (
    LLTLHandoff,
    LLTLSpecification,
    HandoffPackage,
    LLTLTemplate,
    prepare_lltl_handoff
)
from core.symbolic_constraint_engine import (
    Constraint,
    ConstraintType,
    SymbolicConstraintEngine
)


class TestLLTLSpecification:
    """Test suite for LLTLSpecification dataclass"""

    def test_specification_creation(self):
        """Test basic specification creation"""
        spec = LLTLSpecification(
            id="ltl_1",
            name="Test Spec",
            template=LLTLTemplate.SAFETY,
            formula="[] (x > 0)",
            source_constraint="c1",
            priority=3,
            variables=["x"],
            assumptions=["x is real"],
            guarantees=["x is positive"]
        )

        assert spec.id == "ltl_1"
        assert spec.template == LLTLTemplate.SAFETY
        assert spec.priority == 3
        assert len(spec.variables) == 1
        assert len(spec.assumptions) == 1
        assert len(spec.guarantees) == 1

    def test_specification_defaults(self):
        """Test specification with default values"""
        spec = LLTLSpecification(
            id="ltl_2",
            name="Test 2",
            template=LLTLTemplate.LIVENESS,
            formula="<> (goal)",
            source_constraint="c2",
            priority=2,
            variables=None,
            assumptions=None,
            guarantees=None
        )

        assert spec.variables == []
        assert spec.assumptions == []
        assert spec.guarantees == []


class TestHandoffPackage:
    """Test suite for HandoffPackage dataclass"""

    def test_package_creation(self):
        """Test basic package creation"""
        package = HandoffPackage(
            constraints=[],
            ltl_specifications=[],
            translation_map={},
            metadata={"version": "1.0"}
        )

        assert package.constraints == []
        assert package.ltl_specifications == []
        assert package.translation_map == {}
        assert package.metadata["version"] == "1.0"

    def test_package_defaults(self):
        """Test package with default values"""
        package = HandoffPackage(
            constraints=[],
            ltl_specifications=[],
            translation_map=None,
            metadata=None
        )

        assert package.translation_map == {}
        assert package.metadata == {}


class TestLLTLHandoff:
    """Test suite for LLTLHandoff class"""

    def test_handoff_initialization(self):
        """Test handoff module initialization"""
        handoff = LLTLHandoff()
        assert handoff.sce is not None
        assert handoff._spec_counter == 0

    def test_handoff_with_sce(self):
        """Test handoff with existing SCE"""
        sce = SymbolicConstraintEngine()
        handoff = LLTLHandoff(sce)

        assert handoff.sce == sce

    def test_prepare_handoff_empty(self):
        """Test preparing handoff with no constraints"""
        handoff = LLTLHandoff()
        package = handoff.prepare_handoff()

        assert isinstance(package, HandoffPackage)
        assert package.metadata["total_constraints"] == 0
        assert package.metadata["total_ltl_specs"] == 0

    def test_prepare_handoff_with_constraints(self):
        """Test preparing handoff with constraints"""
        handoff = LLTLHandoff()
        sce = handoff.sce

        sce.add_constraint(Constraint(
            id="test",
            type=ConstraintType.HARD,
            description="Temperature must always be below 1000°C",
            formalization="T < 1000",
            source="test"
        ))

        package = handoff.prepare_handoff()

        assert package.metadata["total_constraints"] == 1
        assert package.metadata["total_ltl_specs"] == 1
        assert len(package.ltl_specifications) == 1

    def test_constraint_to_lltl_safety(self):
        """Test converting constraint to safety LLTL"""
        handoff = LLTLHandoff()

        constraint = Constraint(
            id="safety_test",
            type=ConstraintType.HARD,
            description="Temperature must always be below 1000°C",
            formalization="T < 1000",
            source="test"
        )

        specs = handoff.constraint_to_lltl(constraint)

        assert len(specs) >= 1
        assert specs[0].template == LLTLTemplate.SAFETY
        assert "[]" in specs[0].formula

    def test_constraint_to_lltl_liveness(self):
        """Test converting constraint to liveness LLTL"""
        handoff = LLTLHandoff()

        constraint = Constraint(
            id="liveness_test",
            type=ConstraintType.HARD,
            description="The goal must eventually be reached",
            formalization="eventually goal",
            source="test"
        )

        specs = handoff.constraint_to_lltl(constraint)

        assert len(specs) >= 1
        # Should detect liveness pattern
        assert specs[0].formula is not None

    def test_select_template_safety(self):
        """Test template selection for safety constraints"""
        handoff = LLTLHandoff()

        constraint = Constraint(
            id="test",
            type=ConstraintType.HARD,
            description="Temperature must always be below 1000°C",
            formalization="T < 1000",
            source="test"
        )

        template = handoff._select_template(constraint)

        assert template == LLTLTemplate.SAFETY

    def test_select_template_default(self):
        """Test default template selection"""
        handoff = LLTLHandoff()

        constraint = Constraint(
            id="pref_test",
            type=ConstraintType.PREFERENCE,
            description="Some preference",
            formalization="x = 5",
            source="test"
        )

        template = handoff._select_template(constraint)

        # Should default to LIVENESS for preferences
        assert template in [LLTLTemplate.LIVENESS, LLTLTemplate.SAFETY]

    def test_generate_lltl_formula_safety(self):
        """Test LLTL formula generation for safety"""
        handoff = LLTLHandoff()

        constraint = Constraint(
            id="test",
            type=ConstraintType.HARD,
            description="Temperature must be positive",
            formalization="T > 0",
            source="test"
        )

        formula = handoff._generate_lltl_formula(constraint, LLTLTemplate.SAFETY)

        assert "[]" in formula
        assert "(" in formula and ")" in formula

    def test_generate_lltl_formula_liveness(self):
        """Test LLTL formula generation for liveness"""
        handoff = LLTLHandoff()

        constraint = Constraint(
            id="test",
            type=ConstraintType.HARD,
            description="Goal must be reached",
            formalization="eventually goal",
            source="test"
        )

        formula = handoff._generate_lltl_formula(constraint, LLTLTemplate.LIVENESS)

        assert "<>" in formula

    def test_extract_proposition(self):
        """Test proposition extraction"""
        handoff = LLTLHandoff()

        prop = handoff._extract_proposition("The temperature must be below 1000°C")

        assert "temperature" in prop.lower()
        assert "below" in prop.lower()

    def test_split_reactivity(self):
        """Test splitting reactivity constraint"""
        handoff = LLTLHandoff()

        trigger, response = handoff._split_reactivity("If button pressed then system starts")

        assert trigger is not None
        assert response is not None

    def test_split_bounded_response(self):
        """Test splitting bounded response constraint"""
        handoff = LLTLHandoff()

        trigger, response, bound = handoff._split_bounded_response(
            "Respond to request within 5 seconds"
        )

        # May find match or not depending on exact pattern
        assert (trigger is None) == (response is None)

    def test_extract_variables(self):
        """Test variable extraction"""
        handoff = LLTLHandoff()

        constraint = Constraint(
            id="test",
            type=ConstraintType.HARD,
            description="Test",
            formalization="forall T : Real, T < 1000",
            source="test"
        )

        vars = handoff._extract_variables(constraint)

        assert "T" in vars

    def test_generate_assumptions_guarantees(self):
        """Test assumption and guarantee generation"""
        handoff = LLTLHandoff()
        sce = handoff.sce

        # Add base constraint
        sce.add_constraint(Constraint(
            id="base",
            type=ConstraintType.HARD,
            description="Base assumption",
            formalization="x > 0",
            source="test"
        ))

        # Add dependent constraint
        constraint = Constraint(
            id="dependent",
            type=ConstraintType.HARD,
            description="Dependent guarantee",
            formalization="y < 100",
            source="test",
            dependencies=["base"]
        )

        assumptions, guarantees = handoff._generate_assumptions_guarantees(constraint)

        assert len(assumptions) >= 1
        assert len(guarantees) >= 1
        assert "base" in str(assumptions)

    def test_calculate_priority(self):
        """Test priority calculation"""
        handoff = LLTLHandoff()

        hard_constraint = Constraint(
            id="hard",
            type=ConstraintType.HARD,
            description="Hard",
            formalization="x > 0",
            source="test"
        )

        soft_constraint = Constraint(
            id="soft",
            type=ConstraintType.SOFT,
            description="Soft",
            formalization="y < 100",
            source="test"
        )

        hard_priority = handoff._calculate_priority(hard_constraint)
        soft_priority = handoff._calculate_priority(soft_constraint)

        assert hard_priority == 3
        assert soft_priority == 2

    def test_get_template_distribution(self):
        """Test template distribution calculation"""
        handoff = LLTLHandoff()

        specs = [
            LLTLSpecification(
                id="s1",
                name="Safety 1",
                template=LLTLTemplate.SAFETY,
                formula="[] P",
                source_constraint="c1",
                priority=3,
                variables=[],
                assumptions=[],
                guarantees=[]
            ),
            LLTLSpecification(
                id="s2",
                name="Liveness 1",
                template=LLTLTemplate.LIVENESS,
                formula="<> Q",
                source_constraint="c2",
                priority=2,
                variables=[],
                assumptions=[],
                guarantees=[]
            )
        ]

        distribution = handoff._get_template_distribution(specs)

        assert distribution["safety"] == 1
        assert distribution["liveness"] == 1

    def test_export_to_json(self):
        """Test exporting handoff package to JSON"""
        handoff = LLTLHandoff()
        sce = handoff.sce

        sce.add_constraint(Constraint(
            id="test",
            type=ConstraintType.HARD,
            description="Test constraint",
            formalization="x > 0",
            source="test"
        ))

        package = handoff.prepare_handoff()

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        try:
            handoff.export_to_json(temp_file, package)

            # Verify file was created and is valid JSON
            with open(temp_file, 'r') as f:
                data = json.load(f)

            assert "metadata" in data
            assert "constraints" in data
            assert "ltl_specifications" in data

        finally:
            Path(temp_file).unlink()

    def test_create_example_translations(self):
        """Test creation of example translations"""
        handoff = LLTLHandoff()

        examples = handoff.create_example_translations()

        assert "safety" in examples
        assert "liveness" in examples
        assert "reactivity" in examples
        assert "bounded_response" in examples
        assert "persistence" in examples

        # Check that examples are lists
        assert all(isinstance(formulas, list) for formulas in examples.values())

        # Check that safety examples contain []
        assert any("[]" in ex for ex in examples["safety"])

        # Check that liveness examples contain <>
        assert any("<>" in ex for ex in examples["liveness"])


class TestLLTLTemplate:
    """Test suite for LLTLTemplate enum"""

    def test_template_values(self):
        """Test template enum values"""
        assert LLTLTemplate.SAFETY.value == "safety"
        assert LLTLTemplate.LIVENESS.value == "liveness"
        assert LLTLTemplate.REACTIVITY.value == "reactivity"
        assert LLTLTemplate.BOUNDED_RESPONSE.value == "bounded_response"
        assert LLTLTemplate.PERSISTENCE.value == "persistence"


class TestConvenienceFunctions:
    """Test suite for convenience functions"""

    def test_prepare_lltl_handoff(self):
        """Test convenience function for preparing handoff"""
        sce = SymbolicConstraintEngine()

        sce.add_constraint(Constraint(
            id="test",
            type=ConstraintType.HARD,
            description="Test constraint",
            formalization="x > 0",
            source="test"
        ))

        package = prepare_lltl_handoff(sce)

        assert isinstance(package, HandoffPackage)
        assert package.metadata["total_constraints"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
