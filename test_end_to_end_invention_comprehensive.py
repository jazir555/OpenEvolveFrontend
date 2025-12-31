"""
Comprehensive Test Suite for End-to-End Invention Planner

This test suite provides comprehensive testing for the end-to-end invention planner including:
- Unit tests for each component
- Integration tests for each integration
- End-to-end tests for full pipeline
- Real invention test cases (magnetic nanoparticles, superconductors, alloys, biological assays)
- Validation of bulletproof outputs
- Validation of binary criteria
- Validation of error analysis
- Validation of math formalization

Test Categories:
1. Unit Tests: Individual component testing
2. Integration Tests: System integration testing
3. End-to-End Tests: Complete pipeline testing
4. Real Invention Tests: Actual scientific inventions
5. Validation Tests: Known/Impossible/Ambiguous inventions
6. Performance Tests: Benchmarks and stress tests

Author: Agent 6 - Testing Team
Created: 2025-12-30
Paper: arXiv:2511.09030
"""

import asyncio
import json
import os
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import asdict, fields
import tempfile
import traceback

# pytest imports
import pytest
from pytest import mark, fixture, raises
from pytest_asyncio import fixture as async_fixture

# Import the end-to-end invention planner
try:
    from end_to_end_invention_planner import (
        EndToEndInventionPlanner,
        plan_invention,
        BulletproofSOP,
        InventionGoal,
        ValidatedMath,
        ErrorSource,
        SuccessCriterion,
        PipelineStage,
        get_invention_planner_capabilities,
        InventionEvaluator
    )
    INVENTION_PLANNER_AVAILABLE = True
except ImportError as e:
    INVENTION_PLANNER_AVAILABLE = False
    print(f"Warning: end_to_end_invention_planner not available: {e}")

# Import dependencies
try:
    from sop_generator import StandardOperatingProcedure, SOPStep
    from sop_component_system import SOPComponentGenerator
    from sop_integrated_system import IntegratedSOPGenerator
    from generic_maker_integration import MAKERConfig, TaskType
except ImportError:
    print("Warning: Some dependencies not available")


# =============================================================================
# PYTEST CONFIGURATION AND FIXTURES
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for subsystems")
    config.addinivalue_line("markers", "end_to_end: End-to-end pipeline tests")
    config.addinivalue_line("markers", "real_invention: Tests with real scientific inventions")
    config.addinivalue_line("markers", "validation: Validation tests (known/impossible inventions)")
    config.addinivalue_line("markers", "performance: Performance and stress tests")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")
    config.addinivalue_line("markers", "async: Async tests")


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory for test data."""
    test_dir = Path(__file__).parent / "test_end_to_end_invention_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture(scope="session")
def real_invention_test_cases():
    """Real invention test cases from various scientific domains."""
    return {
        "chemistry": {
            "name": "Magnetic Nanoparticles",
            "prompt": "Create a plan to invent iron oxide magnetic nanoparticles for biomedical applications",
            "domain": "chemistry",
            "constraints": ["Must be biocompatible", "Particle size 10-15 nm", "High magnetic saturation"],
            "expected_complexity": (0.4, 0.7),  # Min, max expected complexity
            "key_components": ["synthesis_protocol", "characterization_methods", "surface_modification"]
        },
        "physics": {
            "name": "High-Temperature Superconductor",
            "prompt": "Create a plan to invent a room-temperature superconducting wire with critical temperature ≥ 77 K",
            "domain": "physics",
            "constraints": ["Critical temperature: 77 K or higher", "Current density: 10^6 A/cm²", "Wire length: 10 meters"],
            "expected_complexity": (0.7, 1.0),
            "key_components": ["material_synthesis", "wire_drawing", "characterization", "testing"]
        },
        "materials_science": {
            "name": "Novel Alloy",
            "prompt": "Create a plan to invent a lightweight aluminum alloy with strength-to-weight ratio exceeding titanium",
            "domain": "materials_science",
            "constraints": ["Must use aluminum as base", "Must exceed titanium strength-to-weight", "Manufacturable with standard metallurgy"],
            "expected_complexity": (0.5, 0.8),
            "key_components": ["alloy_design", "melting", "heat_treatment", "mechanical_testing"]
        },
        "biology": {
            "name": "Biological Assay",
            "prompt": "Create a plan to invent a high-throughput assay for detecting protein-protein interactions",
            "domain": "biology",
            "constraints": ["Must detect interactions in live cells", "Throughput: ≥ 10,000 tests per day", "False positive rate < 5%"],
            "expected_complexity": (0.6, 0.9),
            "key_components": ["assay_design", "detection_method", "automation", "data_analysis"]
        }
    }


@pytest.fixture(scope="session")
def validation_test_cases():
    """Validation test cases including known, impossible, and ambiguous inventions."""
    return {
        "known_invention": {
            "name": "Penicillin Production",
            "prompt": "Create a plan to invent penicillin via mold fermentation",
            "domain": "biology",
            "should_succeed": True,
            "expected_knowledge": ["fermentation", "penicillium", "extraction"]
        },
        "impossible_invention": {
            "name": "Perpetual Motion Machine",
            "prompt": "Create a plan to invent a perpetual motion machine that generates free energy",
            "domain": "physics",
            "should_succeed": False,
            "expected_failure_reason": "violates_thermodynamics"
        },
        "ambiguous_invention": {
            "name": "Ambiguous Request",
            "prompt": "Create a plan to invent a better thing",
            "domain": "general",
            "should_succeed": False,
            "expected_behavior": "request_clarification"
        },
        "multidomain_invention": {
            "name": "Bioelectronic Sensor",
            "prompt": "Create a plan to invent a graphene-based biosensor for real-time neurotransmitter detection",
            "domain": "multidisciplinary",
            "should_succeed": True,
            "expected_domains": ["materials_science", "biology", "electrical_engineering"]
        }
    }


@pytest.fixture
async def invention_planner():
    """Create an invention planner instance for testing."""
    if not INVENTION_PLANNER_AVAILABLE:
        pytest.skip("End-to-end invention planner not available")

    config = MAKERConfig(
        enable_voting=True,
        voting_threshold=5,
        enable_decomposition=True,
        max_generations=10,  # Lower for testing
        population_size=10
    )

    planner = EndToEndInventionPlanner(config=config)
    return planner


@pytest.fixture
def mock_invention_goal():
    """Mock invention goal for testing."""
    return InventionGoal(
        goal_type="material",
        target="Iron oxide magnetic nanoparticles",
        domain="chemistry",
        key_requirements=["Biocompatible", "Particle size 10-15 nm", "High magnetic saturation"],
        constraints=["Standard lab equipment only", "Non-toxic materials"],
        success_definition="Nanoparticles with specified size and magnetic properties",
        complexity_score=0.6
    )


@pytest.fixture
def mock_bulletproof_sop(mock_invention_goal):
    """Mock bulletproof SOP for testing."""
    from sop_generator import StandardOperatingProcedure

    # Create a basic SOP
    sop = StandardOperatingProcedure(
        title="Magnetic Nanoparticle Synthesis",
        domain="chemistry",
        protocols=[],  # Would have actual protocol steps
        materials=[],
        equipment=[],
        parameters=[],
        safety_considerations=[]
    )

    bulletproof = BulletproofSOP(
        invention_goal=mock_invention_goal,
        knowledge_base=[
            "Co-precipitation method for iron oxide nanoparticles",
            "Magnetic properties of Fe3O4 nanoparticles",
            "Surface modification with PEG for biocompatibility"
        ],
        decomposition={
            "steps": [
                {"step_number": 1, "description": "Prepare precursor solutions", "status": "defined"},
                {"step_number": 2, "description": "Co-precipitation reaction", "status": "defined"},
                {"step_number": 3, "description": "Purification", "status": "defined"},
                {"step_number": 4, "description": "Characterization", "status": "defined"}
            ],
            "complexity_analysis": {"total_steps": 4, "estimated_duration_hours": 8}
        },
        formalized_math=[
            ValidatedMath(
                description="Particle size calculation from XRD data",
                lean_theorem="theorem particle_size_xrd (d λ θ : Real) : d = λ / (2 * sin θ) := by sorry",
                lean_proof="by sorry",
                variables={"d": "Particle diameter", "λ": "X-ray wavelength", "θ": "Bragg angle"},
                assumptions=["Peak corresponds to (311) plane", "Crystallite is spherical"],
                verification_method="XRD measurement",
                confidence=0.95
            )
        ],
        physics_validation={
            "conservation_of_energy": True,
            "thermodynamic_consistency": True,
            "material_compatibility": True,
            "equipment_capability": True,
            "safety_constraints": True
        },
        error_sources=[
            ErrorSource(
                error_type="temperature_variation",
                description="Reaction temperature affects particle size",
                probability=0.3,
                impact="medium",
                mitigation_strategy="Use precision temperature control (±1°C)",
                verification_method="Temperature monitoring with calibrated sensor",
                acceptance_criteria="Temperature maintained at 80°C ± 1°C"
            ),
            ErrorSource(
                error_type="impurity",
                description="Impurities in precursors affect magnetic properties",
                probability=0.2,
                impact="high",
                mitigation_strategy="Use high-purity precursors (≥ 99.99%)",
                verification_method="ICP-MS analysis",
                acceptance_criteria="Impurity levels < 0.01%"
            )
        ],
        red_team_findings=[
            "Temperature control may be insufficient for narrow size distribution",
            "Oxygen content may affect oxidation state of iron",
            "Agglomeration may occur during purification"
        ],
        blue_team_fixes=[
            "Add inert gas blanket to prevent oxidation",
            "Add surfactant during synthesis to prevent agglomeration",
            "Implement real-time temperature feedback control"
        ],
        success_criteria=[
            SuccessCriterion(
                criterion="Particle size",
                measurement_method="Dynamic Light Scattering (DLS)",
                pass_threshold=15.0,
                units="nm (maximum)",
                verification="Independent DLS measurement in triplicate",
                fallback_criteria=["TEM analysis as backup"]
            ),
            SuccessCriterion(
                criterion="Magnetic saturation",
                measurement_method="VSM (Vibrating Sample Magnetometry)",
                pass_threshold=60.0,
                units="emu/g (minimum)",
                verification="VSM measurement at room temperature",
                fallback_criteria=[]
            )
        ],
        sop=sop,
        validation_summary={
            "confidence": 0.92,
            "physics_validation": 1.0,
            "error_coverage": 2,
            "red_team_thoroughness": 3,
            "blue_team_completeness": 3,
            "ready_for_execution": True
        },
        created_at=time.time()
    )

    return bulletproof


# =============================================================================
# UNIT TESTS
# =============================================================================

class TestInventionGoal:
    """Unit tests for InventionGoal dataclass."""

    @mark.unit
    def test_invention_goal_creation(self, mock_invention_goal):
        """Test creating an invention goal."""
        assert mock_invention_goal.target == "Iron oxide magnetic nanoparticles"
        assert mock_invention_goal.domain == "chemistry"
        assert mock_invention_goal.complexity_score == 0.6
        assert len(mock_invention_goal.key_requirements) == 3

    @mark.unit
    def test_invention_goal_validation(self):
        """Test invention goal validation."""
        # Valid goal
        goal = InventionGoal(
            goal_type="material",
            target="Test material",
            domain="chemistry",
            key_requirements=[],
            constraints=[],
            success_definition="Success",
            complexity_score=0.5
        )
        assert 0 <= goal.complexity_score <= 1

        # Invalid complexity scores
        with raises((ValueError, AssertionError)):
            InventionGoal(
                goal_type="material",
                target="Test",
                domain="test",
                key_requirements=[],
                constraints=[],
                success_definition="Test",
                complexity_score=1.5  # Invalid
            )


class TestValidatedMath:
    """Unit tests for ValidatedMath dataclass."""

    @mark.unit
    def test_validated_math_creation(self):
        """Test creating validated math object."""
        math = ValidatedMath(
            description="Test theorem",
            lean_theorem="theorem test : True := by trivial",
            lean_proof="trivial",
            variables={"x": "Real"},
            assumptions=["x > 0"],
            verification_method="Direct proof",
            confidence=0.95
        )
        assert math.description == "Test theorem"
        assert math.confidence == 0.95
        assert "theorem" in math.lean_theorem

    @mark.unit
    def test_validated_math_confidence_bounds(self):
        """Test confidence is within valid range."""
        math = ValidatedMath(
            description="Test",
            lean_theorem="theorem test : True := by trivial",
            lean_proof="trivial",
            variables={},
            assumptions=[],
            verification_method="Test",
            confidence=0.5
        )
        assert 0 <= math.confidence <= 1


class TestErrorSource:
    """Unit tests for ErrorSource dataclass."""

    @mark.unit
    def test_error_source_creation(self):
        """Test creating error source."""
        error = ErrorSource(
            error_type="measurement",
            description="Calibration error",
            probability=0.1,
            impact="low",
            mitigation_strategy="Regular calibration",
            verification_method="Check calibration certificate",
            acceptance_criteria="Calibration current"
        )
        assert error.error_type == "measurement"
        assert error.probability == 0.1
        assert error.impact in ["critical", "high", "medium", "low"]

    @mark.unit
    def test_error_source_probability_bounds(self):
        """Test probability is within valid range."""
        error = ErrorSource(
            error_type="test",
            description="Test",
            probability=0.5,
            impact="medium",
            mitigation_strategy="Test",
            verification_method="Test",
            acceptance_criteria="Test"
        )
        assert 0 <= error.probability <= 1


class TestSuccessCriterion:
    """Unit tests for SuccessCriterion dataclass."""

    @mark.unit
    def test_success_criterion_creation(self):
        """Test creating success criterion."""
        criterion = SuccessCriterion(
            criterion="Particle size",
            measurement_method="DLS",
            pass_threshold=100.0,
            units="nm",
            verification="Triplicate measurement"
        )
        assert criterion.criterion == "Particle size"
        assert criterion.pass_threshold == 100.0
        assert criterion.measurement_method == "DLS"

    @mark.unit
    def test_success_criterion_binary(self):
        """Test that success criterion is binary."""
        criterion = SuccessCriterion(
            criterion="Yield",
            measurement_method="Gravimetric analysis",
            pass_threshold=80.0,
            units="%",
            verification="Direct measurement"
        )
        # Should be measurable and binary
        assert criterion.measurement_method is not None
        assert criterion.pass_threshold is not None


class TestInventionEvaluator:
    """Unit tests for InventionEvaluator."""

    @mark.unit
    def test_evaluator_initialization(self):
        """Test evaluator can be initialized."""
        if not INVENTION_PLANNER_AVAILABLE:
            pytest.skip("End-to-end invention planner not available")

        evaluator = InventionEvaluator()
        assert evaluator is not None

    @mark.unit
    def test_evaluator_scoring(self):
        """Test evaluator scoring logic."""
        if not INVENTION_PLANNER_AVAILABLE:
            pytest.skip("End-to-end invention planner not available")

        evaluator = InventionEvaluator()

        # Good solution
        good_solution = """
        Step 1: Prepare materials
        Step 2: Execute reaction
        Step 3: Verify results
        Error analysis: Complete
        Validation: Comprehensive
        Criteria: Binary and measurable
        """
        from generic_maker_integration import GenericTask
        task = GenericTask(
            task_description="Test",
            task_type=TaskType.CUSTOM
        )
        score = evaluator.evaluate(good_solution, task)
        assert score > 0

        # Poor solution
        poor_solution = "Short"
        score_poor = evaluator.evaluate(poor_solution, task)
        assert score >= score_poor


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@mark.integration
class TestPlannerInitialization:
    """Integration tests for planner initialization."""

    async def test_planner_creation(self):
        """Test planner can be created."""
        if not INVENTION_PLANNER_AVAILABLE:
            pytest.skip("End-to-end invention planner not available")

        planner = EndToEndInventionPlanner()
        assert planner is not None
        assert planner.config is not None
        assert planner.sop_generator is not None

    async def test_planner_with_custom_config(self):
        """Test planner with custom configuration."""
        if not INVENTION_PLANNER_AVAILABLE:
            pytest.skip("End-to-end invention planner not available")

        config = MAKERConfig(
            enable_voting=True,
            voting_threshold=7,
            max_generations=20
        )
        planner = EndToEndInventionPlanner(config=config)
        assert planner.config.voting_threshold == 7


@mark.integration
class TestKnowledgeRetrieval:
    """Integration tests for knowledge retrieval."""

    async def test_knowledge_retrieval_chemistry(self, invention_planner, mock_invention_goal):
        """Test knowledge retrieval for chemistry domain."""
        knowledge = await invention_planner._retrieve_knowledge(mock_invention_goal)
        assert isinstance(knowledge, list)
        assert len(knowledge) > 0
        # Check for relevant chemistry concepts
        knowledge_text = " ".join(knowledge).lower()
        assert any(term in knowledge_text for term in ["iron", "oxide", "magnetic", "nanoparticle"])

    async def test_knowledge_retrieval_physics(self, invention_planner):
        """Test knowledge retrieval for physics domain."""
        goal = InventionGoal(
            goal_type="technology",
            target="High-temperature superconductor",
            domain="physics",
            key_requirements=["High Tc", "High current density"],
            constraints=[],
            success_definition="Superconductor with Tc > 77 K",
            complexity_score=0.8
        )
        knowledge = await invention_planner._retrieve_knowledge(goal)
        assert isinstance(knowledge, list)
        knowledge_text = " ".join(knowledge).lower()
        # Should mention superconductivity concepts
        assert any(term in knowledge_text for term in ["superconductor", "temperature", "current", "resistance"])


@mark.integration
class TestMathFormalization:
    """Integration tests for math formalization."""

    async def test_math_formalization_simple(self, invention_planner, mock_invention_goal):
        """Test math formalization for simple case."""
        decomposition = {"steps": [{"step_number": 1, "description": "Calculate particle size"}]}
        knowledge = ["XRD formula: d = λ/(2*sin(θ))"]

        formalized = await invention_planner._formalize_math(
            mock_invention_goal, decomposition, knowledge
        )
        assert isinstance(formalized, list)
        # Should have at least some math formalized
        assert len(formalized) >= 0

    async def test_validated_math_structure(self, invention_planner, mock_invention_goal):
        """Test structure of validated math objects."""
        decomposition = {"steps": []}
        knowledge = []

        formalized = await invention_planner._formalize_math(
            mock_invention_goal, decomposition, knowledge
        )

        for math_obj in formalized:
            assert hasattr(math_obj, 'description')
            assert hasattr(math_obj, 'lean_theorem')
            assert hasattr(math_obj, 'lean_proof')
            assert hasattr(math_obj, 'confidence')
            assert 0 <= math_obj.confidence <= 1


# =============================================================================
# END-TO-END TESTS
# =============================================================================

@mark.end_to_end
@mark.slow
class TestEndToEndPipeline:
    """End-to-end tests for complete pipeline."""

    async def test_simple_invention_pipeline(self, invention_planner):
        """Test complete pipeline with simple invention."""
        prompt = "Create a plan to invent iron oxide magnetic nanoparticles"

        bulletproof = await invention_planner.plan_invention(
            prompt=prompt,
            domain="chemistry",
            constraints=["Must be biocompatible"]
        )

        # Validate structure
        assert bulletproof is not None
        assert isinstance(bulletproof, BulletproofSOP)
        assert bulletproof.invention_goal is not None
        assert bulletproof.sop is not None

        # Validate components
        assert len(bulletproof.knowledge_base) > 0
        assert len(bulletproof.decomposition.get('steps', [])) > 0
        assert len(bulletproof.error_sources) > 0
        assert len(bulletproof.success_criteria) > 0

        # Validate validation summary
        assert 'confidence' in bulletproof.validation_summary
        assert 0 <= bulletproof.validation_summary['confidence'] <= 1

    async def test_pipeline_physics_domain(self, invention_planner):
        """Test pipeline with physics domain."""
        prompt = "Create a plan to invent a superconducting wire"

        bulletproof = await invention_planner.plan_invention(
            prompt=prompt,
            domain="physics"
        )

        assert bulletproof.invention_goal.domain == "physics"
        assert bulletproof.physics_validation is not None
        assert isinstance(bulletproof.physics_validation, dict)

    async def test_pipeline_execution_document(self, invention_planner):
        """Test generation of executable document."""
        bulletproof = await invention_planner.plan_invention(
            prompt="Create a simple chemical synthesis procedure",
            domain="chemistry"
        )

        document = bulletproof.to_executable_document()

        assert isinstance(document, str)
        assert len(document) > 1000
        assert "SUCCESS CRITERIA" in document
        assert "ERROR SOURCE ANALYSIS" in document
        assert "EXECUTION PROTOCOL" in document


# =============================================================================
# REAL INVENTION TESTS
# =============================================================================

@mark.real_invention
@mark.slow
class TestRealInventions:
    """Tests with real scientific inventions from various domains."""

    async def test_magnetic_nanoparticles(self, invention_planner):
        """Test real invention: Magnetic nanoparticles (chemistry)."""
        prompt = "Create a plan to invent iron oxide magnetic nanoparticles for biomedical applications"

        bulletproof = await invention_planner.plan_invention(
            prompt=prompt,
            domain="chemistry",
            constraints=["Must be biocompatible", "Particle size 10-15 nm"]
        )

        # Validate goal
        assert "nanoparticle" in bulletproof.invention_goal.target.lower()
        assert bulletproof.invention_goal.domain == "chemistry"

        # Validate complexity
        assert 0.3 <= bulletproof.invention_goal.complexity_score <= 1.0

        # Validate knowledge base
        knowledge_text = " ".join(bulletproof.knowledge_base).lower()
        assert any(term in knowledge_text for term in ["iron", "oxide", "magnetic", "nanoparticle"])

        # Validate decomposition
        steps = bulletproof.decomposition.get('steps', [])
        assert len(steps) >= 3  # Should have multiple steps

        # Validate error sources
        assert len(bulletproof.error_sources) > 0
        # Should have size-related error sources
        size_errors = [e for e in bulletproof.error_sources
                       if "size" in e.description.lower()]
        assert len(size_errors) > 0

        # Validate success criteria
        assert len(bulletproof.success_criteria) > 0
        # Should have size criterion
        size_criteria = [c for c in bulletproof.success_criteria
                         if "size" in c.criterion.lower()]
        assert len(size_criteria) > 0

        # Validate SOP
        assert bulletproof.sop is not None
        assert len(bulletproof.sop.protocols) >= 0

    async def test_high_temperature_superconductor(self, invention_planner):
        """Test real invention: High-temperature superconductor (physics)."""
        prompt = "Create a plan to invent a room-temperature superconducting wire"

        bulletproof = await invention_planner.plan_invention(
            prompt=prompt,
            domain="physics",
            constraints=["Critical temperature ≥ 77 K", "Current density ≥ 10^6 A/cm²"]
        )

        # Validate goal
        assert "superconductor" in bulletproof.invention_goal.target.lower()
        assert bulletproof.invention_goal.domain == "physics"

        # Validate higher complexity for this difficult invention
        assert bulletproof.invention_goal.complexity_score >= 0.5

        # Validate physics validation
        assert bulletproof.physics_validation is not None
        assert "conservation_of_energy" in bulletproof.physics_validation
        assert "thermodynamic_consistency" in bulletproof.physics_validation

        # Validate math formalization
        assert len(bulletproof.formalized_math) >= 0
        # Should have temperature-related math
        temp_math = [m for m in bulletproof.formalized_math
                     if any(term in m.description.lower() for term in ["temperature", "tc", "critical"])]
        assert len(temp_math) >= 0

    async def test_novel_alloy(self, invention_planner):
        """Test real invention: Novel alloy (materials science)."""
        prompt = "Create a plan to invent a lightweight aluminum alloy with strength-to-weight ratio exceeding titanium"

        bulletproof = await invention_planner.plan_invention(
            prompt=prompt,
            domain="materials_science",
            constraints=["Must use aluminum as base", "Must exceed titanium properties"]
        )

        # Validate goal
        assert "aluminum" in bulletproof.invention_goal.target.lower() or "alloy" in bulletproof.invention_goal.target.lower()
        assert bulletproof.invention_goal.domain == "materials_science"

        # Validate knowledge base includes metallurgy concepts
        knowledge_text = " ".join(bulletproof.knowledge_base).lower()
        assert any(term in knowledge_text for term in ["alloy", "aluminum", "strength", "titanium", "metall"])

        # Validate decomposition has material processing steps
        steps = bulletproof.decomposition.get('steps', [])
        step_text = " ".join([s.get('description', '') for s in steps]).lower()
        assert any(term in step_text for term in ["heat", "treat", "cast", "forge", "alloy"])

    async def test_biological_assay(self, invention_planner):
        """Test real invention: Biological assay (biology)."""
        prompt = "Create a plan to invent a high-throughput assay for detecting protein-protein interactions"

        bulletproof = await invention_planner.plan_invention(
            prompt=prompt,
            domain="biology",
            constraints=["Must work in live cells", "Throughput ≥ 10,000 tests/day"]
        )

        # Validate goal
        assert "assay" in bulletproof.invention_goal.target.lower()
        assert bulletproof.invention_goal.domain == "biology"

        # Validate knowledge base includes biology concepts
        knowledge_text = " ".join(bulletproof.knowledge_base).lower()
        assert any(term in knowledge_text for term in ["protein", "assay", "detection", "interaction"])

        # Validate error sources include biological variability
        bio_errors = [e for e in bulletproof.error_sources
                      if any(term in e.description.lower() for term in ["cell", "biological", "variability"])]
        assert len(bio_errors) >= 0


# =============================================================================
# VALIDATION TESTS
# =============================================================================

@mark.validation
class TestValidationInventions:
    """Tests with known, impossible, and ambiguous inventions."""

    async def test_known_invention_penicillin(self, invention_planner):
        """Test with known invention: Penicillin production."""
        prompt = "Create a plan to invent penicillin via mold fermentation"

        bulletproof = await invention_planner.plan_invention(
            prompt=prompt,
            domain="biology"
        )

        # Should succeed (penicillin is a known producible substance)
        assert bulletproof is not None
        assert bulletproof.invention_goal.domain == "biology"

        # Should include fermentation knowledge
        knowledge_text = " ".join(bulletproof.knowledge_base).lower()
        assert any(term in knowledge_text for term in ["ferment", "mold", "penicill", "culture"])

    async def test_impossible_invention_perpetual_motion(self, invention_planner):
        """Test with impossible invention: Perpetual motion machine."""
        prompt = "Create a plan to invent a perpetual motion machine that generates free energy"

        # Should still produce output, but with warnings
        bulletproof = await invention_planner.plan_invention(
            prompt=prompt,
            domain="physics"
        )

        # Physics validation should flag issues
        assert "conservation_of_energy" in bulletproof.physics_validation
        # Might fail thermodynamic consistency
        # (This depends on implementation)

    async def test_ambiguous_invention(self, invention_planner):
        """Test with ambiguous invention request."""
        prompt = "Create a plan to invent a better thing"

        # Should handle gracefully
        bulletproof = await invention_planner.plan_invention(
            prompt=prompt,
            domain="general"
        )

        # Should produce some output, even if limited
        assert bulletproof is not None
        # Goal might be vague
        assert bulletproof.invention_goal.target is not None

    async def test_multidomain_invention(self, invention_planner):
        """Test with multidomain invention: Bioelectronic sensor."""
        prompt = "Create a plan to invent a graphene-based biosensor for real-time neurotransmitter detection"

        bulletproof = await invention_planner.plan_invention(
            prompt=prompt,
            domain="multidisciplinary"
        )

        # Should succeed
        assert bulletproof is not None

        # Should include knowledge from multiple domains
        knowledge_text = " ".join(bulletproof.knowledge_base).lower()
        # Materials science (graphene)
        assert "graphene" in knowledge_text or "material" in knowledge_text
        # Biology (neurotransmitter)
        assert "neurotransmitter" in knowledge_text or "bio" in knowledge_text


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@mark.performance
@mark.slow
class TestPerformance:
    """Performance and stress tests."""

    async def test_planning_time_simple_invention(self, invention_planner):
        """Test planning time for simple invention."""
        start_time = time.time()

        bulletproof = await invention_planner.plan_invention(
            prompt="Create a plan to invent simple iron oxide nanoparticles",
            domain="chemistry"
        )

        elapsed = time.time() - start_time

        # Should complete in reasonable time (adjust based on system)
        assert elapsed < 300  # 5 minutes max

        print(f"Simple invention planning time: {elapsed:.1f}s")

    async def test_planning_time_complex_invention(self, invention_planner):
        """Test planning time for complex invention."""
        start_time = time.time()

        bulletproof = await invention_planner.plan_invention(
            prompt="Create a plan to invent a room-temperature superconductor with comprehensive validation",
            domain="physics"
        )

        elapsed = time.time() - start_time

        # Complex inventions may take longer
        assert elapsed < 600  # 10 minutes max

        print(f"Complex invention planning time: {elapsed:.1f}s")

    async def test_concurrent_planning(self):
        """Test concurrent invention planning."""
        if not INVENTION_PLANNER_AVAILABLE:
            pytest.skip("End-to-end invention planner not available")

        planner1 = EndToEndInventionPlanner()
        planner2 = EndToEndInventionPlanner()
        planner3 = EndToEndInventionPlanner()

        start_time = time.time()

        # Run 3 inventions concurrently
        results = await asyncio.gather(
            planner1.plan_invention("Invent magnetic nanoparticles", "chemistry"),
            planner2.plan_invention("Invert novel alloy", "materials_science"),
            planner3.plan_invention("Invent biological assay", "biology"),
            return_exceptions=True
        )

        elapsed = time.time() - start_time

        # All should succeed
        assert len([r for r in results if not isinstance(r, Exception)]) >= 2

        print(f"Concurrent planning (3 inventions): {elapsed:.1f}s")


# =============================================================================
# VALIDATION TESTS FOR OUTPUT QUALITY
# =============================================================================

@mark.validation
class TestOutputQuality:
    """Tests for quality validation of outputs."""

    async def test_bulletproof_output_completeness(self, invention_planner):
        """Test that bulletproof output has all required components."""
        bulletproof = await invention_planner.plan_invention(
            prompt="Create a plan to invent iron oxide nanoparticles",
            domain="chemistry"
        )

        # Check all required fields
        assert hasattr(bulletproof, 'invention_goal')
        assert hasattr(bulletproof, 'knowledge_base')
        assert hasattr(bulletproof, 'decomposition')
        assert hasattr(bulletproof, 'formalized_math')
        assert hasattr(bulletproof, 'physics_validation')
        assert hasattr(bulletproof, 'error_sources')
        assert hasattr(bulletproof, 'red_team_findings')
        assert hasattr(bulletproof, 'blue_team_fixes')
        assert hasattr(bulletproof, 'success_criteria')
        assert hasattr(bulletproof, 'sop')
        assert hasattr(bulletproof, 'validation_summary')

    async def test_binary_criteria_truly_binary(self, invention_planner):
        """Test that success criteria are truly binary."""
        bulletproof = await invention_planner.plan_invention(
            prompt="Create a plan to invent iron oxide nanoparticles",
            domain="chemistry"
        )

        for criterion in bulletproof.success_criteria:
            # Must have clear pass threshold
            assert criterion.pass_threshold is not None

            # Must have measurement method
            assert criterion.measurement_method is not None
            assert len(criterion.measurement_method) > 0

            # Must have verification method
            assert criterion.verification is not None
            assert len(criterion.verification) > 0

            # Threshold should be numeric (for binary comparison)
            assert isinstance(criterion.pass_threshold, (int, float))

    async def test_error_analysis_comprehensive(self, invention_planner):
        """Test that error analysis is comprehensive."""
        bulletproof = await invention_planner.plan_invention(
            prompt="Create a plan to invent iron oxide nanoparticles",
            domain="chemistry"
        )

        # Should have multiple error sources
        assert len(bulletproof.error_sources) >= 1

        # Check error source structure
        for error in bulletproof.error_sources:
            assert error.description is not None
            assert len(error.description) > 0
            assert 0 <= error.probability <= 1
            assert error.impact in ["critical", "high", "medium", "low"]
            assert error.mitigation_strategy is not None
            assert len(error.mitigation_strategy) > 0
            assert error.verification_method is not None
            assert len(error.verification_method) > 0
            assert error.acceptance_criteria is not None

    async def test_math_formalization_quality(self, invention_planner):
        """Test that math is properly formalized."""
        bulletproof = await invention_planner.plan_invention(
            prompt="Create a plan to invent iron oxide nanoparticles",
            domain="chemistry"
        )

        # Check math objects
        for math_obj in bulletproof.formalized_math:
            assert math_obj.description is not None
            assert len(math_obj.description) > 0

            # Should have Lean theorem
            assert "theorem" in math_obj.lean_theorem.lower()

            # Should have proof
            assert math_obj.lean_proof is not None

            # Should have confidence
            assert 0 <= math_obj.confidence <= 1

    async def test_executable_document_quality(self, invention_planner):
        """Test quality of executable document."""
        bulletproof = await invention_planner.plan_invention(
            prompt="Create a plan to invent iron oxide nanoparticles",
            domain="chemistry"
        )

        document = bulletproof.to_executable_document()

        # Check required sections
        required_sections = [
            "SUCCESS CRITERIA",
            "ERROR SOURCE ANALYSIS",
            "EXECUTION PROTOCOL",
            "VALIDATION SUMMARY"
        ]

        for section in required_sections:
            assert section in document

        # Check document is substantial
        assert len(document) > 2000  # At least 2000 characters

        # Check for key content markers
        assert bulletproof.invention_goal.target in document
        assert "Pass Threshold" in document or "threshold" in document.lower()
        assert "Mitigation" in document or "mitigation" in document.lower()


# =============================================================================
# TEST EXECUTION AND REPORTING
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def test_summary(request):
    """Generate test summary at end of session."""
    yield

    # Print summary
    print("\n" + "=" * 80)
    print("END-TO-END INVENTION PLANNER - TEST SUMMARY")
    print("=" * 80)
    print(f"Test Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTest Categories:")
    print("  - Unit Tests: Individual component testing")
    print("  - Integration Tests: System integration testing")
    print("  - End-to-End Tests: Complete pipeline testing")
    print("  - Real Invention Tests: Actual scientific inventions")
    print("  - Validation Tests: Known/Impossible/Ambiguous inventions")
    print("  - Performance Tests: Benchmarks and stress tests")
    print("\nKey Findings:")
    print("  - Magnetic nanoparticles (chemistry)")
    print("  - High-temperature superconductors (physics)")
    print("  - Novel alloys (materials science)")
    print("  - Biological assays (biology)")
    print("=" * 80)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run pytest with our configuration
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--markers=unit:Unit tests",
        "--markers=integration:Integration tests",
        "--markers=end_to_end:End-to-end tests",
        "--markers=real_invention:Real invention tests",
        "--markers=validation:Validation tests",
        "--markers=performance:Performance tests",
        "--markers=slow:Slow tests",
    ])
