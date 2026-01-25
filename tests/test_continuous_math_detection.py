"""
Test Suite for Continuous Mathematics Detector

Comprehensive tests for B.1 Continuous Math Detection functionality.
Tests all detection capabilities: ODEs, PDEs, DAEs, SDEs, integrals, and derivatives.

Author: OpenEvolve
Created: 2026-01-09
Phase: 2 - LeanAide Enhancement (Task B.1)
"""

import pytest
from continuous_math_detector import (
    ContinuousMathDetector,
    MathType,
    ProblemType,
    ScientificDomain,
    MathDetectionResult,
    detect_continuous_math,
    is_ode,
    is_pde,
    is_dae,
    is_sde,
    is_integral,
    is_derivative,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def detector():
    """Create a ContinuousMathDetector instance for testing"""
    return ContinuousMathDetector()


# ============================================================================
# B.1.1: ODE Detection Tests
# ============================================================================

class TestODEDetection:
    """Test suite for ODE (Ordinary Differential Equation) detection"""

    def test_simple_ode_detection(self, detector):
        """Test detection of simple first-order ODE"""
        text = "Solve the ODE dy/dx = x^2 + y"
        result = detector.detect_ode(text)

        assert result.math_type == MathType.ODE
        assert result.confidence > 0.7
        assert "dy/dx" in result.equations or "x" in result.variables

    def test_second_order_ode_detection(self, detector):
        """Test detection of second-order ODE"""
        text = "Find solution to y'' + 4y' + 4y = 0"
        result = detector.detect_ode(text)

        assert result.math_type == MathType.ODE
        assert result.confidence > 0.7

    def test_ivp_detection(self, detector):
        """Test detection of Initial Value Problem"""
        text = "Solve dy/dx = y with initial condition y(0) = 1"
        result = detector.detect_ode(text)

        assert result.math_type == MathType.ODE
        assert result.problem_type == ProblemType.INITIAL_VALUE
        assert result.confidence > 0.7

    def test_bvp_detection(self, detector):
        """Test detection of Boundary Value Problem"""
        text = "Solve y'' + y = 0 with boundary conditions y(0) = 0, y(pi) = 0"
        result = detector.detect_ode(text)

        assert result.math_type == MathType.ODE
        assert result.problem_type == ProblemType.BOUNDARY_VALUE

    def test_named_ode_detection(self, detector):
        """Test detection of named ODEs"""
        test_cases = [
            "Bessel equation",
            "Legendre equation",
            "Airy equation",
            "van der Pol equation"
        ]

        for text in test_cases:
            result = detector.detect_ode(text)
            assert result.math_type == MathType.ODE, f"Failed to detect: {text}"
            assert result.confidence > 0.5

    def test_convenience_is_ode_function(self):
        """Test convenience function is_ode()"""
        assert is_ode("dy/dx = x + y") == True
        assert is_ode("Solve the ODE") == True
        assert is_ode("integral of x") == False


# ============================================================================
# B.1.2: PDE Detection Tests
# ============================================================================

class TestPDEDetection:
    """Test suite for PDE (Partial Differential Equation) detection"""

    def test_heat_equation_detection(self, detector):
        """Test detection of heat equation"""
        text = "Solve the heat equation ∂u/∂t = ∂²u/∂x²"
        result = detector.detect_pde(text)

        assert result.math_type == MathType.PDE
        assert result.confidence > 0.7

    def test_wave_equation_detection(self, detector):
        """Test detection of wave equation"""
        text = "Find solution to wave equation ∂²u/∂t² = c² ∂²u/∂x²"
        result = detector.detect_pde(text)

        assert result.math_type == MathType.PDE
        assert result.confidence > 0.7

    def test_laplace_equation_detection(self, detector):
        """Test detection of Laplace equation"""
        text = "Solve Laplace equation ∇²u = 0"
        result = detector.detect_pde(text)

        assert result.math_type == MathType.PDE

    def test_schrodinger_equation_detection(self, detector):
        """Test detection of Schrödinger equation"""
        text = "Time-independent Schrödinger equation"
        result = detector.detect_pde(text)

        assert result.math_type == MathType.PDE
        assert result.confidence > 0.5

    def test_navier_stokes_detection(self, detector):
        """Test detection of Navier-Stokes equation"""
        text = "Navier-Stokes equation for fluid flow"
        result = detector.detect_pde(text)

        assert result.math_type == MathType.PDE
        assert result.domain == ScientificDomain.PHYSICS

    def test_convenience_is_pde_function(self):
        """Test convenience function is_pde()"""
        assert is_pde("∂u/∂t = ∂²u/∂x²") == True
        assert is_pde("heat equation") == True
        assert is_pde("dy/dx = x") == False


# ============================================================================
# B.1.3: DAE Detection Tests
# ============================================================================

class TestDAEDetection:
    """Test suite for DAE (Differential-Algebraic Equation) detection"""

    def test_dae_detection(self, detector):
        """Test detection of DAE"""
        text = "Solve the differential-algebraic equation with algebraic constraints"
        result = detector.detect_dae(text)

        assert result.math_type == MathType.DAE
        assert result.confidence > 0.5

    def test_index_1_dae(self, detector):
        """Test detection of index-1 DAE"""
        text = "DAE with index 1 and mass matrix"
        result = detector.detect_dae(text)

        assert result.math_type == MathType.DAE

    def test_convenience_is_dae_function(self):
        """Test convenience function is_dae()"""
        assert is_dae("differential-algebraic equation") == True
        assert is_dae("DAE with constraints") == True
        assert is_dae("simple ODE") == False


# ============================================================================
# B.1.4: SDE Detection Tests
# ============================================================================

class TestSDEDetection:
    """Test suite for SDE (Stochastic Differential Equation) detection"""

    def test_sde_detection(self, detector):
        """Test detection of SDE"""
        text = "Solve the stochastic differential equation dX = μX dt + σX dW"
        result = detector.detect_sde(text)

        assert result.math_type == MathType.SDE
        assert result.confidence > 0.7

    def test_brownian_motion_detection(self, detector):
        """Test detection of Brownian motion SDE"""
        text = "Geometric Brownian motion for stock prices"
        result = detector.detect_sde(text)

        assert result.math_type == MathType.SDE
        assert result.domain == ScientificDomain.ECONOMICS

    def test_langevin_equation_detection(self, detector):
        """Test detection of Langevin equation"""
        text = "Langevin equation with Wiener process"
        result = detector.detect_sde(text)

        assert result.math_type == MathType.SDE

    def test_convenience_is_sde_function(self):
        """Test convenience function is_sde()"""
        assert is_sde("stochastic differential equation") == True
        assert is_sde("dW = Wiener process") == True
        assert is_sde("dy/dx = f(x)") == False


# ============================================================================
# B.1.5: Integral Detection Tests
# ============================================================================

class TestIntegralDetection:
    """Test suite for integral detection"""

    def test_simple_integral_detection(self, detector):
        """Test detection of simple integral"""
        text = "Calculate the integral of x^2 from 0 to 1"
        result = detector.detect_integral(text)

        assert result.math_type == MathType.INTEGRAL
        assert result.confidence > 0.7

    def test_latex_integral_detection(self, detector):
        """Test detection of LaTeX integral notation"""
        text = "Evaluate \\int_{0}^{1} x^2 dx"
        result = detector.detect_integral(text)

        assert result.math_type == MathType.INTEGRAL
        assert result.confidence > 0.7

    def test_definite_integral_detection(self, detector):
        """Test detection of definite integral"""
        text = "Compute the definite integral"
        result = detector.detect_integral(text)

        assert result.math_type == MathType.INTEGRAL

    def test_double_integral_detection(self, detector):
        """Test detection of double integral"""
        text = "Evaluate the double integral over region R"
        result = detector.detect_integral(text)

        assert result.math_type == MathType.INTEGRAL

    def test_convenience_is_integral_function(self):
        """Test convenience function is_integral()"""
        assert is_integral("integral of x^2") == True
        assert is_integral("\\int x dx") == True
        assert is_integral("derivative") == False


# ============================================================================
# B.1.6: Derivative Detection Tests
# ============================================================================

class TestDerivativeDetection:
    """Test suite for derivative detection"""

    def test_simple_derivative_detection(self, detector):
        """Test detection of simple derivative"""
        text = "Find the derivative of f(x) = x^3"
        result = detector.detect_derivative(text)

        assert result.math_type == MathType.DERIVATIVE
        assert result.confidence > 0.7

    def test_prime_notation_detection(self, detector):
        """Test detection of prime notation"""
        text = "Calculate f'(x) for f(x) = x^2"
        result = detector.detect_derivative(text)

        assert result.math_type == MathType.DERIVATIVE

    def test_partial_derivative_detection(self, detector):
        """Test detection of partial derivative"""
        text = "Find ∂f/∂x for f(x,y) = x^2 + y^2"
        result = detector.detect_derivative(text)

        assert result.math_type == MathType.DERIVATIVE

    def test_rate_of_change_detection(self, detector):
        """Test detection of rate of change"""
        text = "Calculate the rate of change of velocity"
        result = detector.detect_derivative(text)

        assert result.math_type == MathType.DERIVATIVE

    def test_convenience_is_derivative_function(self):
        """Test convenience function is_derivative()"""
        assert is_derivative("derivative of x^2") == True
        assert is_derivative("dy/dx") == True
        assert is_derivative("integral") == False


# ============================================================================
# B.1.7: Pattern Matching Tests
# ============================================================================

class TestPatternMatching:
    """Test suite for pattern matching capabilities"""

    def test_latex_notation_detection(self, detector):
        """Test LaTeX notation recognition"""
        text = "Solve $y' + y = 0$ with initial condition"
        result = detector.detect(text)

        assert result.notation == "LaTeX"
        assert len(result.equations) > 0

    def test_sympy_notation_detection(self, detector):
        """Test SymPy notation recognition"""
        text = "Solve using symbols and Derivative"
        result = detector.detect(text)

        assert result.notation == "SymPy"

    def test_variable_extraction(self, detector):
        """Test variable extraction"""
        text = "Solve for x and t in the equation"
        result = detector.detect(text)

        assert "x" in result.variables or "t" in result.variables

    def test_equation_extraction(self, detector):
        """Test equation extraction"""
        text = "Given dy/dx = x + y and y(0) = 1, find y(x)"
        result = detector.detect(text)

        assert len(result.equations) > 0


# ============================================================================
# B.1.8: Domain Classification Tests
# ============================================================================

class TestDomainClassification:
    """Test suite for scientific domain classification"""

    def test_physics_domain_detection(self, detector):
        """Test physics domain detection"""
        text = "Newton's second law with force and momentum"
        result = detector.detect(text)

        assert result.domain == ScientificDomain.PHYSICS
        assert result.confidence > 0.3

    def test_chemistry_domain_detection(self, detector):
        """Test chemistry domain detection"""
        text = "Chemical reaction kinetics and concentration"
        result = detector.detect(text)

        assert result.domain == ScientificDomain.CHEMISTRY

    def test_biology_domain_detection(self, detector):
        """Test biology domain detection"""
        text = "Population dynamics and predator-prey model"
        result = detector.detect(text)

        assert result.domain == ScientificDomain.BIOLOGY

    def test_engineering_domain_detection(self, detector):
        """Test engineering domain detection"""
        text = "Control system with feedback and stability"
        result = detector.detect(text)

        assert result.domain == ScientificDomain.ENGINEERING

    def test_economics_domain_detection(self, detector):
        """Test economics domain detection"""
        text = "Stock price and Black-Scholes option pricing"
        result = detector.detect(text)

        assert result.domain == ScientificDomain.ECONOMICS


# ============================================================================
# B.1.9: Problem Type Classification Tests
# ============================================================================

class TestProblemTypeClassification:
    """Test suite for problem type classification"""

    def test_initial_value_problem(self, detector):
        """Test IVP classification"""
        text = "Solve with initial condition y(0) = 1"
        result = detector.detect(text)

        assert result.problem_type == ProblemType.INITIAL_VALUE

    def test_boundary_value_problem(self, detector):
        """Test BVP classification"""
        text = "Solve with boundary conditions at x=0 and x=L"
        result = detector.detect(text)

        assert result.problem_type == ProblemType.BOUNDARY_VALUE

    def test_eigenvalue_problem(self, detector):
        """Test eigenvalue problem classification"""
        text = "Find eigenvalues and eigenfunctions"
        result = detector.detect(text)

        assert result.problem_type == ProblemType.EIGENVALUE

    def test_control_problem(self, detector):
        """Test control problem classification"""
        text = "Design a feedback controller to stabilize the system"
        result = detector.detect(text)

        assert result.problem_type == ProblemType.CONTROL

    def test_optimization_problem(self, detector):
        """Test optimization problem classification"""
        text = "Minimize the cost function"
        result = detector.detect(text)

        assert result.problem_type == ProblemType.OPTIMIZATION


# ============================================================================
# B.1.10: Integration Tests
# ============================================================================

class TestContinuousMathIntegration:
    """Integration tests for continuous math detection"""

    def test_heat_equation_full_analysis(self, detector):
        """Test full analysis of heat equation problem"""
        text = """
        Solve the heat equation ∂u/∂t = α ∂²u/∂x² for 0 < x < L
        with initial condition u(x,0) = f(x)
        and boundary conditions u(0,t) = u(L,t) = 0
        """

        result = detector.detect(text)

        assert result.math_type == MathType.PDE
        assert result.problem_type == ProblemType.INITIAL_BOUNDARY_VALUE
        assert result.domain == ScientificDomain.PHYSICS
        assert result.confidence > 0.7

    def test_lotka_volterra_equations(self, detector):
        """Test detection of Lotka-Volterra predator-prey model"""
        text = """
        Analyze the population dynamics using Lotka-Volterra equations:
        dx/dt = αx - βxy
        dy/dt = δxy - γy
        """

        result = detector.detect(text)

        assert result.math_type in [MathType.ODE, MathType.PDE]
        assert result.domain == ScientificDomain.BIOLOGY
        assert "x" in result.variables and "y" in result.variables

    def test_black_scholes_equation(self, detector):
        """Test detection of Black-Scholes equation"""
        text = "Price options using Black-Scholes model with volatility"
        result = detector.detect(text)

        assert result.math_type == MathType.PDE  # Black-Scholes is a PDE
        assert result.domain == ScientificDomain.ECONOMICS

    def test_bernoulli_differential_equation(self, detector):
        """Test detection of Bernoulli equation"""
        text = "Solve dy/dx + P(x)y = Q(x)y^n"
        result = detector.detect(text)

        assert result.math_type == MathType.ODE

    def test_calculus_problem_sequence(self, detector):
        """Test detection of multiple calculus operations"""
        text = """
        First find the derivative of f(x) = x^3,
        then calculate the integral from 0 to 1,
        and finally evaluate the limit as x approaches 0.
        """

        result = detector.detect(text)

        # Should detect multiple math types
        assert result.math_type in [MathType.DERIVATIVE, MathType.INTEGRAL, MathType.LIMIT]
        assert len(result.keywords) >= 2


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
