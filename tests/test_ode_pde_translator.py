"""
Test Suite for ODE/PDE Translator to Lean 4

Comprehensive tests for B.2 ODE/PDE Translation functionality.
Tests all translation capabilities: ODEs, PDEs, DAEs, SDEs, initial/boundary
conditions, and proof scaffolding generation.

Author: OpenEvolve
Created: 2026-01-09
Phase: 2 - LeanAide Enhancement (Task B.2)
"""

import pytest
from ode_pde_translator import (
    ODEPDETranslator,
    Lean4TranslationResult,
    Lean4CodeBlock,
    SolutionType,
    translate_to_lean4,
    translate_ode_to_lean4,
)
from continuous_math_detector import (
    ContinuousMathDetector,
    MathType,
    ProblemType,
    ScientificDomain,
    MathDetectionResult,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def detector():
    """Create a ContinuousMathDetector instance for testing"""
    return ContinuousMathDetector()


@pytest.fixture
def translator():
    """Create an ODEPDETranslator instance for testing"""
    return ODEPDETranslator()


@pytest.fixture
def simple_ode_result(detector):
    """Create detection result for simple ODE"""
    text = "Solve the ODE dy/dx + y = 0"
    return detector.detect(text)


@pytest.fixture
def ivp_result(detector):
    """Create detection result for IVP"""
    text = "Solve dy/dx = y with initial condition y(0) = 1"
    return detector.detect(text)


@pytest.fixture
def bvp_result(detector):
    """Create detection result for BVP"""
    text = "Solve y'' + y = 0 with boundary conditions y(0) = 0, y(pi) = 0"
    return detector.detect(text)


@pytest.fixture
def heat_equation_result(detector):
    """Create detection result for heat equation"""
    text = "Solve the heat equation ∂u/∂t = α ∂²u/∂x²"
    return detector.detect(text)


# ============================================================================
# B.2.1: ODE Translation Tests
# ============================================================================

class TestODETranslation:
    """Test suite for ODE translation to Lean 4"""

    def test_simple_ode_translation(self, translator, simple_ode_result):
        """Test translation of simple first-order ODE"""
        result = translator.translate(simple_ode_result)

        assert result.success is True
        assert result.lean4_code is not None
        assert len(result.lean4_code) > 0
        assert "y_ode" in result.lean4_code or "ode" in result.lean4_code.lower()

    def test_ivp_translation(self, translator, ivp_result):
        """Test translation of initial value problem"""
        result = translator.translate(ivp_result)

        assert result.success is True
        assert len(result.definitions) > 0
        assert len(result.theorems) > 0
        assert "ivp" in result.lean4_code.lower() or "initial" in result.lean4_code.lower()

    def test_bvp_translation(self, translator, bvp_result):
        """Test translation of boundary value problem"""
        result = translator.translate(bvp_result)

        assert result.success is True
        assert len(result.definitions) > 0
        assert "bvp" in result.lean4_code.lower() or "boundary" in result.lean4_code.lower()

    def test_existence_theorem_generation(self, translator, simple_ode_result):
        """Test generation of existence theorem"""
        result = translator.translate(
            simple_ode_result,
            solution_type=SolutionType.EXISTENCE
        )

        assert result.success is True
        assert "exists" in result.lean4_code.lower()
        assert "∃" in result.lean4_code

    def test_uniqueness_theorem_generation(self, translator, simple_ode_result):
        """Test generation of uniqueness theorem"""
        result = translator.translate(
            simple_ode_result,
            solution_type=SolutionType.UNIQUENESS
        )

        assert result.success is True
        assert "unique" in result.lean4_code.lower()
        assert "∃!" in result.lean4_code or "unique" in result.lean4_code

    def test_existence_uniqueness_theorem(self, translator, ivp_result):
        """Test generation of existence and uniqueness theorem"""
        result = translator.translate(
            ivp_result,
            solution_type=SolutionType.EXISTENCE_UNIQUENESS
        )

        assert result.success is True
        assert "exists" in result.lean4_code.lower()
        assert "unique" in result.lean4_code.lower()

    def test_proof_scaffold_generation(self, translator, ivp_result):
        """Test generation of proof scaffolding"""
        result = translator.translate(
            ivp_result,
            generate_proof_scaffold=True
        )

        assert result.success is True
        assert len(result.proof_scaffolds) > 0
        assert any("proof" in scaffold.description.lower() for scaffold in result.proof_scaffolds)

    def test_no_proof_scaffold_when_disabled(self, translator, ivp_result):
        """Test that proof scaffolds are not generated when disabled"""
        result = translator.translate(
            ivp_result,
            generate_proof_scaffold=False
        )

        assert result.success is True
        assert len(result.proof_scaffolds) == 0

    def test_standalone_ode_translation(self, translator):
        """Test standalone ODE translation method"""
        result = translator.translate_ode(
            equation="y' + y = 0",
            initial_condition="y(0) = 1"
        )

        assert result.success is True
        assert result.lean4_code is not None
        assert len(result.lean4_code) > 0


# ============================================================================
# B.2.2: PDE Translation Tests
# ============================================================================

class TestPDETranslation:
    """Test suite for PDE translation to Lean 4"""

    def test_heat_equation_translation(self, translator, heat_equation_result):
        """Test translation of heat equation"""
        result = translator.translate(heat_equation_result)

        assert result.success is True
        assert len(result.definitions) > 0
        assert "pde" in result.lean4_code.lower() or "heat" in result.lean4_code.lower()

    def test_wave_equation_translation(self, translator, detector):
        """Test translation of wave equation"""
        text = "Solve the wave equation ∂²u/∂t² = c² ∂²u/∂x²"
        detection_result = detector.detect(text)

        result = translator.translate(detection_result)

        assert result.success is True
        assert "wave" in result.lean4_code.lower() or "pde" in result.lean4_code.lower()

    def test_laplace_equation_translation(self, translator, detector):
        """Test translation of Laplace equation"""
        text = "Solve Laplace equation ∇²u = 0"
        detection_result = detector.detect(text)

        result = translator.translate(detection_result)

        assert result.success is True
        assert "laplace" in result.lean4_code.lower() or "∇²" in result.lean4_code

    def test_pde_with_boundary_conditions(self, translator, detector):
        """Test PDE with boundary conditions"""
        text = """
        Solve the heat equation ∂u/∂t = ∂²u/∂x²
        with boundary conditions u(0,t) = u(L,t) = 0
        """
        detection_result = detector.detect(text)

        result = translator.translate(detection_result)

        assert result.success is True
        assert "boundary" in result.lean4_code.lower()

    def test_physics_domain_pde(self, translator, heat_equation_result):
        """Test PDE in physics domain"""
        result = translator.translate(heat_equation_result)

        assert result.success is True
        assert len(result.definitions) > 0
        # Physics domain PDEs should have specialized theorems
        assert any(thm for thm in result.theorems if "heat" in thm.description.lower())

    def test_standalone_pde_translation(self, translator):
        """Test standalone PDE translation method"""
        result = translator.translate_pde(
            equation="∂u/∂t = ∂²u/∂x²",
            boundary_conditions=["u(0,t) = 0", "u(L,t) = 0"]
        )

        assert result.success is True
        assert result.lean4_code is not None
        assert len(result.lean4_code) > 0


# ============================================================================
# B.2.3: DAE Translation Tests
# ============================================================================

class TestDAETranslation:
    """Test suite for DAE translation to Lean 4"""

    def test_dae_translation(self, translator, detector):
        """Test translation of DAE"""
        text = "Solve the differential-algebraic equation with constraints"
        detection_result = detector.detect(text)

        result = translator.translate(detection_result)

        assert result.success is True
        assert "dae" in result.lean4_code.lower()
        assert len(result.definitions) > 0

    def test_dae_structure(self, translator, detector):
        """Test DAE has correct structure"""
        text = "DAE with differential and algebraic parts"
        detection_result = detector.detect(text)

        result = translator.translate(detection_result)

        assert result.success is True
        # Should have both differential and algebraic components
        assert "differential" in result.lean4_code.lower()
        assert "algebraic" in result.lean4_code.lower()


# ============================================================================
# B.2.4: SDE Translation Tests
# ============================================================================

class TestSDETranslation:
    """Test suite for SDE translation to Lean 4"""

    def test_sde_translation(self, translator, detector):
        """Test translation of SDE"""
        text = "Solve the stochastic differential equation dX = μX dt + σX dW"
        detection_result = detector.detect(text)

        result = translator.translate(detection_result)

        assert result.success is True
        assert "sde" in result.lean4_code.lower()
        assert len(result.definitions) > 0

    def test_brownian_motion_in_sde(self, translator, detector):
        """Test SDE includes Brownian motion"""
        text = "Geometric Brownian motion dS = μS dt + σS dW"
        detection_result = detector.detect(text)

        result = translator.translate(detection_result)

        assert result.success is True
        assert "brownian" in result.lean4_code.lower() or "wiener" in result.lean4_code.lower()

    def test_sde_structure(self, translator, detector):
        """Test SDE has correct structure"""
        text = "Langevin equation with stochastic term"
        detection_result = detector.detect(text)

        result = translator.translate(detection_result)

        assert result.success is True
        # Should have drift and diffusion components
        assert "drift" in result.lean4_code.lower() or "diffusion" in result.lean4_code.lower()


# ============================================================================
# B.2.5: Code Structure Tests
# ============================================================================

class TestCodeStructure:
    """Test suite for generated Lean 4 code structure"""

    def test_imports_included(self, translator, simple_ode_result):
        """Test that necessary imports are included"""
        result = translator.translate(simple_ode_result)

        assert result.success is True
        assert len(result.imports) > 0
        assert "Mathlib" in "\n".join(result.imports)

    def test_definitions_generated(self, translator, ivp_result):
        """Test that definitions are generated"""
        result = translator.translate(ivp_result)

        assert result.success is True
        assert len(result.definitions) > 0
        assert all(isinstance(d, Lean4CodeBlock) for d in result.definitions)

    def test_theorems_generated(self, translator, ivp_result):
        """Test that theorems are generated"""
        result = translator.translate(ivp_result)

        assert result.success is True
        assert len(result.theorems) > 0
        assert all(isinstance(t, Lean4CodeBlock) for t in result.theorems)

    def test_proof_scaffolds_generated(self, translator, ivp_result):
        """Test that proof scaffolds are generated"""
        result = translator.translate(
            ivp_result,
            generate_proof_scaffold=True
        )

        assert result.success is True
        assert len(result.proof_scaffolds) > 0
        assert all(isinstance(p, Lean4CodeBlock) for p in result.proof_scaffolds)

    def test_lean4_code_structure(self, translator, ivp_result):
        """Test complete Lean 4 file structure"""
        result = translator.translate(ivp_result)

        assert result.success is True
        code = result.lean4_code

        # Check for file structure
        assert "import" in code
        assert "namespace" in code
        assert "def " in code or "theorem " in code
        assert "end " in code

    def test_code_blocks_have_descriptions(self, translator, ivp_result):
        """Test that code blocks have descriptions"""
        result = translator.translate(ivp_result)

        assert result.success is True

        for definition in result.definitions:
            assert definition.description is not None
            assert len(definition.description) > 0

        for theorem in result.theorems:
            assert theorem.description is not None
            assert len(theorem.description) > 0


# ============================================================================
# B.2.6: Lean 4 Syntax Tests
# ============================================================================

class TestLean4Syntax:
    """Test suite for Lean 4 syntax correctness"""

    def test_lean4_definition_syntax(self, translator, simple_ode_result):
        """Test that generated definitions use correct Lean 4 syntax"""
        result = translator.translate(simple_ode_result)

        assert result.success is True
        code = result.lean4_code

        # Check for Lean 4 syntax elements
        assert "def " in code or "structure " in code
        assert "Prop" in code or "Type" in code
        assert ":=" in code or " : " in code

    def test_lean4_theorem_syntax(self, translator, ivp_result):
        """Test that generated theorems use correct Lean 4 syntax"""
        result = translator.translate(ivp_result)

        assert result.success is True
        code = result.lean4_code

        # Check for theorem syntax
        assert "theorem " in code or "lemma " in code
        assert " : " in code  # Type annotation
        assert ":=" in code or "by" in code  # Proof

    def test_lean4_quantifiers(self, translator, ivp_result):
        """Test that Lean 4 quantifiers are used correctly"""
        result = translator.translate(ivp_result)

        assert result.success is True
        code = result.lean4_code

        # Check for quantifiers
        assert "∀" in code or "forall" in code
        assert "∃" in code or "exists" in code

    def test_lean4_function_types(self, translator, simple_ode_result):
        """Test that Lean 4 function types are correct"""
        result = translator.translate(simple_ode_result)

        assert result.success is True
        code = result.lean4_code

        # Check for function types
        assert "→" in code or " -> " in code

    def test_lean4_tactics_mentioned(self, translator, ivp_result):
        """Test that Lean 4 tactics are mentioned in proof scaffolds"""
        result = translator.translate(
            ivp_result,
            generate_proof_scaffold=True
        )

        assert result.success is True

        # At least one proof scaffold should mention tactics
        has_tactics = any(
            "apply" in scaffold.code.lower() or
            "simp" in scaffold.code.lower() or
            "rw" in scaffold.code.lower()
            for scaffold in result.proof_scaffolds
        )
        assert has_tactics


# ============================================================================
# B.2.7: Integration Tests
# ============================================================================

class TestTranslatorIntegration:
    """Integration tests for translator with detector"""

    def test_detector_to_translator_pipeline(self, detector, translator):
        """Test complete pipeline from detection to translation"""
        # Step 1: Detect
        text = "Solve dy/dx + 2y = 0 with y(0) = 5"
        detection_result = detector.detect(text)

        assert detection_result.math_type == MathType.ODE
        assert detection_result.confidence > 0.5

        # Step 2: Translate
        translation_result = translator.translate(detection_result)

        assert translation_result.success is True
        assert len(translation_result.lean4_code) > 0

    def test_heat_equation_full_pipeline(self, detector, translator):
        """Test full pipeline for heat equation"""
        # Step 1: Detect
        text = """
        Solve the heat equation ∂u/∂t = α ∂²u/∂x²
        with initial condition u(x,0) = f(x)
        and boundary conditions u(0,t) = u(L,t) = 0
        """
        detection_result = detector.detect(text)

        assert detection_result.math_type == MathType.PDE
        assert detection_result.domain == ScientificDomain.PHYSICS

        # Step 2: Translate
        translation_result = translator.translate(detection_result)

        assert translation_result.success is True
        assert "heat" in translation_result.lean4_code.lower()

    def test_lotka_volterra_translation(self, detector, translator):
        """Test translation of Lotka-Volterra system"""
        text = """
        Analyze population dynamics using Lotka-Volterra:
        dx/dt = αx - βxy
        dy/dt = δxy - γy
        """
        detection_result = detector.detect(text)

        translation_result = translator.translate(detection_result)

        assert translation_result.success is True
        # Should handle system of ODEs
        assert len(translation_result.definitions) > 0

    def test_black_scholes_translation(self, detector, translator):
        """Test translation of Black-Scholes equation"""
        text = "Price options using Black-Scholes PDE"
        detection_result = detector.detect(text)

        translation_result = translator.translate(detection_result)

        assert translation_result.success is True
        assert "black" in translation_result.lean4_code.lower() or "pde" in translation_result.lean4_code.lower()


# ============================================================================
# B.2.8: Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Test suite for convenience functions"""

    def test_translate_to_lean4_function(self):
        """Test translate_to_lean4 convenience function"""
        detection_result = MathDetectionResult(
            math_type=MathType.ODE,
            problem_type=ProblemType.INITIAL_VALUE,
            domain=ScientificDomain.GENERAL,
            confidence=1.0,
            equations=["y' + y = 0"],
            variables=["x", "y"],
            notation="standard",
            keywords=["ode"]
        )

        result = translate_to_lean4(detection_result)

        assert result.success is True
        assert len(result.lean4_code) > 0

    def test_translate_ode_to_lean4_function(self):
        """Test translate_ode_to_lean4 convenience function"""
        lean4_code = translate_ode_to_lean4(
            equation="y' + y = 0",
            initial_condition="y(0) = 1"
        )

        assert lean4_code is not None
        assert len(lean4_code) > 0
        assert "import" in lean4_code or "def " in lean4_code


# ============================================================================
# B.2.9: Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test suite for error handling"""

    def test_unsupported_math_type(self, translator):
        """Test handling of unsupported math types"""
        detection_result = MathDetectionResult(
            math_type=MathType.INTEGRAL,  # Not supported by translator
            problem_type=ProblemType.UNKNOWN,
            domain=ScientificDomain.GENERAL,
            confidence=0.5,
            equations=[""],
            variables=[],
            notation="",
            keywords=[]
        )

        result = translator.translate(detection_result)

        assert result.success is False
        assert result.error_message is not None
        assert "unsupported" in result.error_message.lower()

    def test_empty_equation(self, translator):
        """Test handling of empty equation"""
        detection_result = MathDetectionResult(
            math_type=MathType.ODE,
            problem_type=ProblemType.UNKNOWN,
            domain=ScientificDomain.GENERAL,
            confidence=0.0,
            equations=[""],
            variables=[],
            notation="",
            keywords=[]
        )

        result = translator.translate(detection_result)

        # Should still generate some code structure
        assert result.lean4_code is not None

    def test_malformed_equation(self, translator):
        """Test handling of malformed equation"""
        result = translator.translate_ode(
            equation="this is not a valid equation",
            initial_condition=None
        )

        # Should attempt translation even with malformed input
        assert result.lean4_code is not None


# ============================================================================
# B.2.10: Metadata and Documentation Tests
# ============================================================================

class TestMetadataAndDocumentation:
    """Test suite for metadata and documentation"""

    def test_translation_result_metadata(self, translator, ivp_result):
        """Test that translation result includes metadata"""
        result = translator.translate(ivp_result)

        assert result.success is True
        assert "metadata" in result.to_dict()

    def test_code_block_dependencies(self, translator, ivp_result):
        """Test that code blocks track dependencies"""
        result = translator.translate(ivp_result)

        assert result.success is True

        for code_block in result.definitions + result.theorems:
            assert isinstance(code_block.dependencies, list)

    def test_proof_scaffold_content(self, translator, ivp_result):
        """Test that proof scaffolds contain useful information"""
        result = translator.translate(
            ivp_result,
            generate_proof_scaffold=True
        )

        assert result.success is True

        for scaffold in result.proof_scaffolds:
            # Should have description and code
            assert scaffold.description is not None
            assert scaffold.code is not None
            # Should mention proof steps or tactics
            assert "proof" in scaffold.description.lower() or "tactics" in scaffold.code.lower()


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
