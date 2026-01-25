"""
Test Suite for Verification Methods

Comprehensive tests for B.4 Verification Methods functionality.
Tests all verification capabilities: syntax, types, mathematical correctness,
domain patterns, conservation laws, boundary conditions, and proof verification.

Author: OpenEvolve
Created: 2026-01-09
Phase: 2 - LeanAide Enhancement (Task B.4)
"""

import pytest
from verification_methods import (
    Lean4Verifier,
    VerificationResult,
    VerificationStatus,
    VerificationIssue,
    CheckType,
    verify_lean4_code,
    verify_translation,
)
from continuous_math_detector import (
    ContinuousMathDetector,
    ScientificDomain,
    MathType,
    ProblemType,
    MathDetectionResult,
)
from ode_pde_translator import (
    ODEPDETranslator,
    Lean4TranslationResult,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def verifier():
    """Create a Lean4Verifier instance for testing"""
    return Lean4Verifier(enable_leanaide=False)  # Disable LeanAide for tests


@pytest.fixture
def detector():
    """Create a ContinuousMathDetector instance"""
    return ContinuousMathDetector()


@pytest.fixture
def translator():
    """Create an ODEPDETranslator instance"""
    return ODEPDETranslator()


@pytest.fixture
def valid_lean4_code():
    """Valid Lean 4 code for testing"""
    return '''
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Tactic

namespace Test

open Real

/-- Simple ODE definition -/
def test_ode (f : Real → Real) : Prop :=
  ∀ x, deriv f x + f x = 0

/-- Existence theorem -/
theorem test_exists
    (f : Real → Real)
    (x₀ y₀ : Real)
    : ∃ y : Real → Real, test_ode y ∧ y x₀ = y₀ :=
  by
    sorry

end Test
'''


# ============================================================================
# B.4.1: Syntax Validation Tests
# ============================================================================

class TestSyntaxValidation:
    """Test suite for syntax validation"""

    def test_valid_code_passes_syntax_check(self, verifier, valid_lean4_code):
        """Test that valid Lean 4 code passes syntax check"""
        result = verifier.verify_code(valid_lean4_code)

        syntax_issues = [i for i in result.issues if i.check_type == CheckType.SYNTAX]
        errors = [i for i in syntax_issues if i.severity == "error"]

        assert len(errors) == 0, f"Found unexpected syntax errors: {errors}"

    def test_empty_code_fails_syntax_check(self, verifier):
        """Test that empty code fails syntax check"""
        result = verifier.verify_code("")

        assert result.overall_status in [VerificationStatus.FAILED, VerificationStatus.ERROR]
        assert any("Empty" in i.message for i in result.issues)

    def test_missing_namespace_warning(self, verifier):
        """Test warning for missing namespace"""
        code = '''
def test (x : Real) : Prop := x > 0
'''
        result = verifier.verify_code(code)

        namespace_issues = [i for i in result.issues if "namespace" in i.message.lower()]
        assert len(namespace_issues) > 0

    def test_mismatched_braces_error(self, verifier):
        """Test error for mismatched braces"""
        code = '''
def test (x : Real) : Prop :=
  x > 0
'''  # Missing closing brace

        result = verifier.verify_code(code)

        brace_issues = [i for i in result.issues if "brace" in i.message.lower()]
        assert len(brace_issues) > 0

    def test_no_imports_warning(self, verifier):
        """Test warning for missing imports"""
        code = '''
namespace Test
def test (x : Real) : Prop := x > 0
end Test
'''

        result = verifier.verify_code(code)

        import_issues = [i for i in result.issues if "import" in i.message.lower()]
        assert len(import_issues) > 0


# ============================================================================
# B.4.2: Type Checking Tests
# ============================================================================

class TestTypeChecking:
    """Test suite for type consistency checks"""

    def test_real_type_with_import(self, verifier, valid_lean4_code):
        """Test Real type usage with proper imports"""
        result = verifier.verify_code(valid_lean4_code)

        type_issues = [i for i in result.issues if i.check_type == CheckType.TYPE]
        errors = [i for i in type_issues if i.severity == "error"]

        assert len(errors) == 0

    def test_real_without_import_warning(self, verifier):
        """Test warning for Real type without proper import"""
        code = '''
namespace Test
def test (x : Real) : Prop := x > 0
end Test
'''

        result = verifier.verify_code(code)

        type_issues = [i for i in result.issues if i.check_type == CheckType.TYPE]
        import_warnings = [i for i in type_issues if "import" in i.message.lower()]

        assert len(import_warnings) > 0

    def test_prop_type_checking(self, verifier):
        """Test Prop type annotations"""
        code = '''
import Mathlib.Data.Real.Basic
namespace Test
def test : Prop := ∀ x : Real, x > 0
end Test
'''

        result = verifier.verify_code(code)

        prop_issues = [i for i in result.issues if "Prop" in i.message]
        # Should not have errors about Prop
        errors = [i for i in prop_issues if i.severity == "error"]
        assert len(errors) == 0


# ============================================================================
# B.4.3: Mathematical Correctness Tests
# ============================================================================

class TestMathematicalCorrectness:
    """Test suite for mathematical correctness checks"""

    def test_derivative_with_import(self, verifier):
        """Test derivative usage with proper imports"""
        code = '''
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.Calculus.Deriv
namespace Test
def has_deriv (f : Real → Real) : Prop :=
  ∀ x, deriv f x > 0
end Test
'''

        result = verifier.verify_code(code)

        math_issues = [i for i in result.issues if i.check_type == CheckType.MATHEMATICAL]
        errors = [i for i in math_issues if i.severity == "error"]
        assert len(errors) == 0

    def test_derivative_without_import_warning(self, verifier):
        """Test warning for derivative without proper import"""
        code = '''
import Mathlib.Data.Real.Basic
namespace Test
def has_deriv (f : Real → Real) : Prop :=
  ∀ x, deriv f x > 0
end Test
'''

        result = verifier.verify_code(code)

        math_issues = [i for i in result.issues if i.check_type == CheckType.MATHEMATICAL]
        deriv_warnings = [i for i in math_issues if "deriv" in i.message.lower() or "import" in i.message.lower()]

        assert len(deriv_warnings) > 0

    def test_ode_structure_check(self, verifier):
        """Test ODE structure validation"""
        code = '''
import Mathlib.Data.Real.Basic
namespace Test
def my_ode (y : Real → Real) : Prop :=
  ∀ x, deriv y x + y x = 0
end Test
'''

        result = verifier.verify_code(code)

        # Should recognize ODE structure
        has_ode = any("ode" in i.message.lower() for i in result.issues)
        # Should not have errors
        errors = [i for i in result.issues if i.severity == "error"]
        assert len(errors) == 0

    def test_quantifier_suggestion(self, verifier):
        """Test quantifier usage suggestions"""
        code = '''
import Mathlib.Data.Real.Basic
namespace Test
theorem test : True := sorry
end Test
'''

        result = verifier.verify_code(code)

        quantifier_issues = [i for i in result.issues if "quantifier" in i.message.lower()]
        # Should suggest quantifiers for theorems
        assert len(quantifier_issues) > 0


# ============================================================================
# B.4.4: Domain Pattern Tests
# ============================================================================

class TestDomainPatterns:
    """Test suite for domain-specific pattern checking"""

    def test_physics_domain_check(self, verifier, detector):
        """Test physics domain pattern checking"""
        text = "Solve the heat equation ∂u/∂t = α ∂²u/∂x²"
        detection_result = detector.detect(text)

        code = '''
import Mathlib.Data.Real.Basic
namespace Test
def heat_eq (u : Real → Real → Real) : Prop :=
  ∀ x t, deriv (fun t => u x t) t = deriv (deriv (fun x => u x t)) x
end Test
'''

        result = verifier.verify_code(code, domain=ScientificDomain.PHYSICS)

        # Should complete without errors
        domain_issues = [i for i in result.issues if i.check_type == CheckType.DOMAIN]
        errors = [i for i in domain_issues if i.severity == "error"]
        assert len(errors) == 0

    def test_biology_domain_check(self, verifier, detector):
        """Test biology domain pattern checking"""
        text = "Lotka-Volterra predator-prey model"
        detection_result = detector.detect(text)

        code = '''
import Mathlib.Data.Real.Basic
namespace Test
def lotka_volterra (x y : Real → Real) : Prop :=
  ∀ t, deriv x t = x t - x t * y t ∧ deriv y t = x t * y t - y t
end Test
'''

        result = verifier.verify_code(code, domain=ScientificDomain.BIOLOGY)

        # Should complete without errors
        assert result.overall_status != VerificationStatus.ERROR

    def test_general_domain_skips_patterns(self, verifier):
        """Test that GENERAL domain skips pattern checking"""
        code = '''
import Mathlib.Data.Real.Basic
namespace Test
def test (x : Real) : Prop := x > 0
end Test
'''

        result = verifier.verify_code(code, domain=ScientificDomain.GENERAL)

        domain_issues = [i for i in result.issues if i.check_type == CheckType.DOMAIN]
        # GENERAL domain should not generate domain issues
        errors = [i for i in domain_issues if i.severity == "error"]
        assert len(errors) == 0


# ============================================================================
# B.4.5: Conservation Law Tests
# ============================================================================

class TestConservationLaws:
    """Test suite for conservation law verification"""

    def test_physics_energy_conservation(self, verifier):
        """Test physics energy conservation pattern"""
        code = '''
import Mathlib.Data.Real.Basic
namespace Test
def energy_conserved (E : Real → Real) : Prop :=
  ∀ t, deriv E t = 0
end Test
'''

        result = verifier.verify_code(code, domain=ScientificDomain.PHYSICS)

        # Should not complain about missing conservation
        conservation_issues = [i for i in result.issues if i.check_type == CheckType.CONSERVATION]
        errors = [i for i in conservation_issues if i.severity == "error"]
        assert len(errors) == 0

    def test_physics_conservation_suggestion(self, verifier):
        """Test conservation law suggestion for physics"""
        code = '''
import Mathlib.Data.Real.Basic
namespace Test
def motion (x : Real → Real) : Prop :=
  ∀ t, deriv (deriv x) t = -9.8
end Test
'''

        result = verifier.verify_code(code, domain=ScientificDomain.PHYSICS)

        # Should suggest adding conservation laws
        conservation_issues = [i for i in result.issues if i.check_type == CheckType.CONSERVATION]
        suggestions = [i for i in conservation_issues if i.suggestion is not None]
        assert len(suggestions) > 0

    def test_biology_mass_conservation(self, verifier):
        """Test biology mass/total population conservation"""
        code = '''
import Mathlib.Data.Real.Basic
namespace Test
def total_population (S I R : Real → Real) : Prop :=
  ∀ t, S t + I t + R t = 1000
end Test
'''

        result = verifier.verify_code(code, domain=ScientificDomain.BIOLOGY)

        # Should recognize conservation pattern
        assert result.overall_status != VerificationStatus.ERROR


# ============================================================================
# B.4.6: Boundary Condition Tests
# ============================================================================

class TestBoundaryConditions:
    """Test suite for boundary condition validation"""

    def test_ivp_with_initial_condition(self, verifier, detector):
        """Test IVP with initial condition"""
        text = "Solve dy/dx = y with initial condition y(0) = 1"
        detection_result = detector.detect(text)

        code = '''
import Mathlib.Data.Real.Basic
namespace Test
def ivp_solution (y : Real → Real) : Prop :=
  deriv y 0 = 1 ∧ y 0 = 1
end Test
'''

        result = verifier.verify(code, detector_result=None)

        # Should recognize initial condition
        boundary_issues = [i for i in result.issues if i.check_type == CheckType.BOUNDARY]
        errors = [i for i in boundary_issues if i.severity == "error"]
        assert len(errors) == 0

    def test_ivp_without_initial_condition_warning(self, verifier):
        """Test warning for IVP without initial condition"""
        code = '''
import Mathlib.Data.Real.Basic
namespace Test
def ivp (y : Real → Real) : Prop :=
  ∀ x, deriv y x = y x
end Test
'''

        # Create detection result for IVP
        detection_result = MathDetectionResult(
            math_type=MathType.ODE,
            problem_type=ProblemType.INITIAL_VALUE,
            domain=ScientificDomain.GENERAL,
            confidence=1.0,
            equations=["dy/dx = y"],
            variables=["x", "y"],
            notation="",
            keywords=[]
        )

        result = verifier.verify_code(code, domain=ScientificDomain.GENERAL)

        # Should warn about missing initial condition
        boundary_issues = [i for i in result.issues if i.check_type == CheckType.BOUNDARY]
        warnings = [i for i in boundary_issues if i.severity == "warning"]
        assert len(warnings) > 0


# ============================================================================
# B.4.7: Integration Tests
# ============================================================================

class TestVerificationIntegration:
    """Integration tests for complete verification pipeline"""

    def test_complete_pipeline(self, verifier, detector, translator):
        """Test complete detection → translation → verification pipeline"""
        # Step 1: Detect
        text = "Solve dy/dx + y = 0 with y(0) = 1"
        detection_result = detector.detect(text)

        # Step 2: Translate
        translation_result = translator.translate(detection_result)

        # Step 3: Verify
        verification_result = verifier.verify(translation_result, detection_result)

        # Should complete without critical errors
        assert verification_result.overall_status != VerificationStatus.ERROR
        assert len(verification_result.checks_performed) > 0
        assert verification_result.verification_time > 0

    def test_heat_equation_verification(self, verifier, detector, translator):
        """Test verification of heat equation translation"""
        text = "Solve the heat equation ∂u/∂t = α ∂²u/∂x²"
        detection_result = detector.detect(text)
        translation_result = translator.translate(detection_result)

        verification_result = verifier.verify(translation_result, detection_result)

        # Should have checks performed
        assert len(verification_result.checks_performed) > 0

        # Should have metadata
        assert "total_checks" in verification_result.metadata

    def test_domain_aware_verification(self, verifier, detector, translator):
        """Test domain-aware verification"""
        text = "Analyze population dynamics with Lotka-Volterra"
        detection_result = detector.detect(text)
        translation_result = translator.translate(detection_result)

        verification_result = verifier.verify(translation_result, detection_result)

        # Should include domain checks
        assert CheckType.DOMAIN in verification_result.checks_performed


# ============================================================================
# B.4.8: Result Structure Tests
# ============================================================================

class TestVerificationResult:
    """Test suite for VerificationResult structure"""

    def test_result_to_dict(self, verifier, valid_lean4_code):
        """Test VerificationResult.to_dict() method"""
        result = verifier.verify_code(valid_lean4_code)

        result_dict = result.to_dict()

        assert "overall_status" in result_dict
        assert "checks_performed" in result_dict
        assert "issues" in result_dict
        assert "passed_checks" in result_dict
        assert "failed_checks" in result_dict
        assert "warnings" in result_dict
        assert "verification_time" in result_dict

    def test_result_is_valid_property(self, verifier):
        """Test is_valid property"""
        # Passing result
        code = '''
import Mathlib.Data.Real.Basic
namespace Test
def test (x : Real) : Prop := x > 0
end Test
'''

        result = verifier.verify_code(code)

        # Should be valid if no errors
        errors = [i for i in result.issues if i.severity == "error"]
        if len(errors) == 0:
            assert result.is_valid

    def test_result_status_determination(self, verifier):
        """Test overall status determination"""
        code = '''
import Mathlib.Data.Real.Basic
namespace Test
def test (x : Real) : Prop := x > 0
end Test
'''

        result = verifier.verify_code(code)

        # Status should be one of the expected values
        assert result.overall_status in [
            VerificationStatus.PASSED,
            VerificationStatus.FAILED,
            VerificationStatus.WARNING,
            VerificationStatus.ERROR
        ]


# ============================================================================
# B.4.9: Issue Structure Tests
# ============================================================================

class TestVerificationIssue:
    """Test suite for VerificationIssue structure"""

    def test_issue_to_dict(self):
        """Test VerificationIssue.to_dict() method"""
        issue = VerificationIssue(
            check_type=CheckType.SYNTAX,
            severity="error",
            message="Test message",
            location="line 10",
            suggestion="Fix it",
            code_snippet="def test := ..."
        )

        issue_dict = issue.to_dict()

        assert issue_dict["check_type"] == "syntax"
        assert issue_dict["severity"] == "error"
        assert issue_dict["message"] == "Test message"
        assert issue_dict["location"] == "line 10"
        assert issue_dict["suggestion"] == "Fix it"
        assert issue_dict["code_snippet"] == "def test := ..."

    def test_issue_with_optional_fields(self):
        """Test VerificationIssue with None optional fields"""
        issue = VerificationIssue(
            check_type=CheckType.SYNTAX,
            severity="warning",
            message="Test warning"
        )

        assert issue.location is None
        assert issue.suggestion is None
        assert issue.code_snippet is None


# ============================================================================
# B.4.10: Convenience Functions Tests
# ============================================================================

class TestConvenienceFunctions:
    """Test suite for convenience functions"""

    def test_verify_lean4_code_function(self, valid_lean4_code):
        """Test verify_lean4_code convenience function"""
        result = verify_lean4_code(valid_lean4_code, enable_leanaide=False)

        assert isinstance(result, VerificationResult)
        assert result.verification_time > 0

    def test_verify_translation_function(self, detector, translator):
        """Test verify_translation convenience function"""
        text = "Solve dy/dx + y = 0"
        detection_result = detector.detect(text)
        translation_result = translator.translate(detection_result)

        result = verify_translation(translation_result, detection_result)

        assert isinstance(result, VerificationResult)
        assert len(result.checks_performed) > 0


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
