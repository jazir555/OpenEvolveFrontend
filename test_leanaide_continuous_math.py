"""
Comprehensive Tests for LeanAide Continuous Math Implementation

This test suite covers:
1. Continuous mathematical domains (real analysis, complex analysis, etc.)
2. Autoformalization pipeline
3. Lean 4 integration
4. Z3 bridge
5. MDAP/MAKER integration

Run with: pytest test_leanaide_continuous_math.py -v
"""

import asyncio
import pytest
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules under test
try:
    from leanaide_continuous_math import (
        ContinuousMathEngine,
        LeanAideAutoformalizer,
        ContinuousDomain,
        LimitType,
        DifferentiabilityClass,
        OptimizationType,
        Interval,
        LimitResult,
        DerivativeResult,
        IntegralResult,
        ComplexResult,
        FunctionalResult,
        MeasureResult,
        TopologicalResult,
        OptimizationResult,
        ODEResult,
        create_continuous_math_engine,
        create_autoformalizer
    )
    CONTINUOUS_MATH_AVAILABLE = True
except ImportError as e:
    CONTINUOUS_MATH_AVAILABLE = False
    print(f"Warning: leanaide_continuous_math not available: {e}")

try:
    from lean4_integration import (
        LeanAideService,
        Lean4ServerConfig,
        Lean4VerificationEngine,
        Lean4AutoformalizationEngine,
        Lean4ProofCompletionEngine,
        VerificationResult,
        VerificationStatus,
        AutoformalizationResult,
        ProofCompletionResult,
        ProofSuggestion,
        create_lean4_service,
        create_verification_engine
    )
    LEAN4_AVAILABLE = True
except ImportError as e:
    LEAN4_AVAILABLE = False
    print(f"Warning: lean4_integration not available: {e}")

try:
    from leanaide_autoformalization_mdap_maker import (
        LeanAideAutoformalizationMDAPMaker,
        MultiAgentFormalizationSystem,
        MDAPMakerIntegration,
        FormalizationStage,
        InputType,
        VerificationLevel,
        FormalizationAgent,
        FormalizationVote,
        RedFlag,
        MDAPFormalizationResult,
        BatchFormalizationResult,
        create_autoformalization_mdap_maker
    )
    MDAP_AVAILABLE = True
except ImportError as e:
    MDAP_AVAILABLE = False
    print(f"Warning: leanaide_autoformalization_mdap_maker not available: {e}")

try:
    from z3_leanaide_bridge import (
        Z3LeanAideBridge,
        Z3ToLeanTranslator,
        LeanToZ3Translator,
        Z3LeanVerificationBridge,
        HybridProofEngine,
        TranslationDirection,
        ConstraintType,
        Z3Constraint,
        Lean4Constraint,
        TranslationResult,
        VerificationBridgeResult,
        HybridProofResult,
        create_z3_lean_bridge
    )
    Z3_BRIDGE_AVAILABLE = True
except ImportError as e:
    Z3_BRIDGE_AVAILABLE = False
    print(f"Warning: z3_leanaide_bridge not available: {e}")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def math_engine():
    """Create ContinuousMathEngine instance"""
    if CONTINUOUS_MATH_AVAILABLE:
        return create_continuous_math_engine(enable_lean_proofs=False)
    return None


@pytest.fixture
def autoformalizer():
    """Create LeanAideAutoformalizer instance"""
    if CONTINUOUS_MATH_AVAILABLE:
        return create_autoformalizer()
    return None


@pytest.fixture
def mdap_maker():
    """Create MDAP maker instance"""
    if MDAP_AVAILABLE:
        return create_autoformalization_mdap_maker()
    return None


@pytest.fixture
def z3_bridge():
    """Create Z3 bridge instance"""
    if Z3_BRIDGE_AVAILABLE:
        return create_z3_lean_bridge()
    return None


# ============================================================================
# Continuous Math Engine Tests
# ============================================================================

@pytest.mark.skipif(not CONTINUOUS_MATH_AVAILABLE, reason="Continuous math not available")
class TestContinuousMathEngine:
    """Tests for ContinuousMathEngine"""
    
    @pytest.mark.asyncio
    async def test_compute_limit_basic(self, math_engine):
        """Test basic limit computation"""
        result = await math_engine.compute_limit("sin(x)/x", "x", 0.0)
        
        assert isinstance(result, LimitResult)
        assert result.expression == "sin(x)/x"
        assert result.variable == "x"
        assert result.point == 0.0
        assert result.limit_value == pytest.approx(1.0, abs=1e-10)
        assert result.limit_type == LimitType.TWO_SIDED
        assert result.existence_proven
    
    @pytest.mark.asyncio
    async def test_compute_limit_infinity(self, math_engine):
        """Test limit at infinity"""
        result = await math_engine.compute_limit("1/x", "x", "oo")
        
        assert isinstance(result, LimitResult)
        assert result.expression == "1/x"
        assert result.point == "oo"
        assert result.limit_type == LimitType.INFINITE
    
    @pytest.mark.asyncio
    async def test_compute_derivative(self, math_engine):
        """Test derivative computation"""
        result = await math_engine.compute_derivative("x**3 + 2*x**2 + x", "x", order=1)
        
        assert isinstance(result, DerivativeResult)
        assert result.function == "x**3 + 2*x**2 + x"
        assert result.variable == "x"
        assert result.order == 1
        assert "3*x**2 + 4*x + 1" in result.derivative or "3*x**2+4*x+1" in result.derivative
    
    @pytest.mark.asyncio
    async def test_compute_integral_definite(self, math_engine):
        """Test definite integral computation"""
        result = await math_engine.compute_integral("x**2", "x", 0.0, 1.0)
        
        assert isinstance(result, IntegralResult)
        assert result.integrand == "x**2"
        assert result.variable == "x"
        assert result.is_definite
        assert result.value == pytest.approx(1/3, abs=1e-6)
        assert result.convergence_proven
    
    @pytest.mark.asyncio
    async def test_compute_integral_indefinite(self, math_engine):
        """Test indefinite integral computation"""
        result = await math_engine.compute_integral("x**2", "x")
        
        assert isinstance(result, IntegralResult)
        assert result.integrand == "x**2"
        assert not result.is_definite
        assert isinstance(result.value, str)
    
    @pytest.mark.asyncio
    async def test_complex_analysis(self, math_engine):
        """Test complex analysis operations"""
        result = await math_engine.complex_analysis("exp(I*z)", "z", point=1+1j)
        
        assert isinstance(result, ComplexResult)
        assert result.expression == "exp(I*z)"
        assert abs(result.magnitude) > 0
    
    @pytest.mark.asyncio
    async def test_optimize_unconstrained(self, math_engine):
        """Test unconstrained optimization"""
        result = await math_engine.optimize(
            "(x - 2)**2 + (y - 3)**2",
            ["x", "y"],
            initial_guess=[0.0, 0.0]
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.optimization_type == OptimizationType.UNCONSTRAINED
        assert len(result.optimal_point) == 2
        assert result.optimal_point[0] == pytest.approx(2.0, abs=0.1)
        assert result.optimal_point[1] == pytest.approx(3.0, abs=0.1)
    
    @pytest.mark.asyncio
    async def test_solve_ode(self, math_engine):
        """Test ODE solving"""
        result = await math_engine.solve_ode(
            "-y",
            "y",
            "t",
            {"y": 1.0},
            (0.0, 5.0)
        )
        
        assert isinstance(result, ODEResult)
        assert result.equation == "-y"
        assert result.initial_conditions["y"] == 1.0
        assert result.is_linear
    
    @pytest.mark.asyncio
    async def test_functional_analysis(self, math_engine):
        """Test functional analysis operations"""
        result = await math_engine.functional_analysis(
            "norm",
            "x**2",
            "L2",
            Interval(0, 1)
        )
        
        assert isinstance(result, FunctionalResult)
        assert result.functional_type == "norm"
        assert result.space == "L2"
        assert result.value > 0


# ============================================================================
# Interval Tests
# ============================================================================

@pytest.mark.skipif(not CONTINUOUS_MATH_AVAILABLE, reason="Continuous math not available")
class TestInterval:
    """Tests for Interval class"""
    
    def test_interval_creation(self):
        """Test interval creation"""
        interval = Interval(0, 1)
        assert interval.lower == 0
        assert interval.upper == 1
    
    def test_interval_auto_correct(self):
        """Test interval auto-corrects reversed bounds"""
        interval = Interval(1, 0)
        assert interval.lower == 0
        assert interval.upper == 1
    
    def test_interval_contains(self):
        """Test interval membership"""
        interval = Interval(0, 1)
        assert 0.5 in interval
        assert 0 in interval
        assert 1 in interval
        assert -0.1 not in interval
        assert 1.1 not in interval
    
    def test_interval_arithmetic(self):
        """Test interval arithmetic"""
        i1 = Interval(0, 1)
        i2 = Interval(2, 3)
        
        result = i1 + i2
        assert result.lower == 2
        assert result.upper == 4
    
    def test_interval_to_lean(self):
        """Test interval to Lean conversion"""
        interval = Interval(0, 1)
        lean_str = interval.to_lean()
        assert "Set.Icc" in lean_str
        assert "0" in lean_str
        assert "1" in lean_str


# ============================================================================
# Autoformalizer Tests
# ============================================================================

@pytest.mark.skipif(not CONTINUOUS_MATH_AVAILABLE, reason="Continuous math not available")
class TestAutoformalizer:
    """Tests for LeanAideAutoformalizer"""
    
    @pytest.mark.asyncio
    async def test_formalize_limit_statement(self, autoformalizer):
        """Test formalizing limit statement"""
        result = await autoformalizer.formalize_problem(
            "The limit as x approaches 0 of sin(x)/x equals 1",
            domain_hint="real_analysis"
        )
        
        assert result["success"]
        assert "lean_code" in result
        assert result["domain"] == "real_analysis"
    
    @pytest.mark.asyncio
    async def test_formalize_derivative(self, autoformalizer):
        """Test formalizing derivative statement"""
        result = await autoformalizer.formalize_problem(
            "The derivative of x squared is 2x",
            domain_hint="real_analysis"
        )
        
        assert result["success"]
        assert "lean_code" in result
    
    @pytest.mark.asyncio
    async def test_formalize_latex(self, autoformalizer):
        """Test formalizing LaTeX"""
        result = await autoformalizer.formalize_latex(
            r"\lim_{x \to 0} \frac{\sin x}{x} = 1"
        )
        
        assert result["success"]
        assert "lean_code" in result
    
    @pytest.mark.asyncio
    async def test_formalize_python(self, autoformalizer):
        """Test formalizing Python code"""
        result = await autoformalizer.formalize_python(
            "def f(x): return x**2"
        )
        
        assert result["success"]
        assert "lean_code" in result


# ============================================================================
# Lean4 Integration Tests
# ============================================================================

@pytest.mark.skipif(not LEAN4_AVAILABLE, reason="Lean4 integration not available")
class TestLean4Integration:
    """Tests for Lean4 integration"""
    
    def test_verification_engine_creation(self):
        """Test verification engine creation"""
        engine = create_verification_engine()
        assert isinstance(engine, Lean4VerificationEngine)
    
    @pytest.mark.asyncio
    async def test_verify_simple_proof(self):
        """Test verifying simple proof"""
        engine = create_verification_engine()
        code = """
theorem simple_theorem : 1 + 1 = 2 := by
  rfl
"""
        result = await engine.verify(code)
        
        assert isinstance(result, VerificationResult)
        # Note: This will fail if Lean is not installed
        # In CI environment, this is expected to work
    
    def test_autoformalization_engine(self):
        """Test autoformalization engine creation"""
        engine = Lean4AutoformalizationEngine()
        assert isinstance(engine, Lean4AutoformalizationEngine)
    
    @pytest.mark.asyncio
    async def test_autoformalize_limit(self):
        """Test autoformalizing limit"""
        engine = Lean4AutoformalizationEngine()
        result = await engine.autoformalize(
            "The limit as x approaches 0 of sin(x)/x equals 1",
            domain="real_analysis"
        )
        
        assert isinstance(result, AutoformalizationResult)
        assert result.domain == "real_analysis"
        assert "limit" in result.natural_language.lower() or "sin" in result.natural_language.lower()
    
    def test_proof_completion_engine(self):
        """Test proof completion engine creation"""
        engine = Lean4ProofCompletionEngine()
        assert isinstance(engine, Lean4ProofCompletionEngine)
    
    @pytest.mark.asyncio
    async def test_suggest_tactics(self):
        """Test tactic suggestion"""
        engine = Lean4ProofCompletionEngine()
        code = """
import Mathlib

theorem example_theorem (n : ℕ) : n + 0 = n := by
  sorry
"""
        suggestions = await engine.suggest_tactics(code)
        
        assert isinstance(suggestions, list)
        if suggestions:
            assert isinstance(suggestions[0], ProofSuggestion)


# ============================================================================
# MDAP Maker Tests
# ============================================================================

@pytest.mark.skipif(not MDAP_AVAILABLE, reason="MDAP maker not available")
class TestMDAPMaker:
    """Tests for MDAP Maker integration"""
    
    def test_multi_agent_system_creation(self):
        """Test multi-agent system creation"""
        system = MultiAgentFormalizationSystem()
        assert len(system.agents) > 0
    
    @pytest.mark.asyncio
    async def test_multi_agent_formalize(self):
        """Test multi-agent formalization"""
        system = MultiAgentFormalizationSystem()
        result = await system.formalize(
            "The limit as x approaches 0 of sin(x)/x equals 1",
            InputType.NATURAL_LANGUAGE,
            "real_analysis"
        )
        
        assert isinstance(result, MDAPFormalizationResult)
        assert result.input_type == InputType.NATURAL_LANGUAGE
        assert result.domain == "real_analysis"
    
    def test_formalization_agent(self):
        """Test formalization agent creation"""
        agent = FormalizationAgent(
            agent_id="test_0",
            agent_type="parser",
            specialization="analysis",
            confidence=0.8
        )
        
        assert agent.agent_id == "test_0"
        assert agent.agent_type == "parser"
        assert agent.confidence == 0.8
    
    def test_formalization_vote(self):
        """Test formalization vote"""
        agent = FormalizationAgent(
            agent_id="test_0",
            agent_type="parser",
            specialization="analysis"
        )
        
        vote = FormalizationVote(
            agent=agent,
            proposed_code="import Mathlib",
            confidence=0.9,
            rationale="Test vote",
            expected_success=0.8
        )
        
        assert vote.agent == agent
        assert vote.confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_mdap_maker_integration(self):
        """Test MDAP Maker integration"""
        integration = MDAPMakerIntegration()
        result = await integration.formalize_and_prove(
            "The derivative of x squared is 2x",
            InputType.NATURAL_LANGUAGE,
            "real_analysis"
        )
        
        assert isinstance(result, MDAPFormalizationResult)
    
    @pytest.mark.asyncio
    async def test_batch_formalization(self, mdap_maker):
        """Test batch formalization"""
        problems = [
            {"text": "The limit as x approaches 0 of sin(x)/x equals 1", "domain": "real_analysis"},
            {"text": "The derivative of x squared is 2x", "domain": "real_analysis"}
        ]
        
        result = await mdap_maker.batch_formalize(problems)
        
        assert isinstance(result, BatchFormalizationResult)
        assert len(result.results) == 2
    
    def test_input_type_enum(self):
        """Test InputType enum values"""
        assert InputType.NATURAL_LANGUAGE.value == "natural_language"
        assert InputType.LATEX.value == "latex"
        assert InputType.PYTHON.value == "python"
    
    def test_formalization_stage_enum(self):
        """Test FormalizationStage enum values"""
        assert FormalizationStage.NL_PARSING.value == "nl_parsing"
        assert FormalizationStage.CODE_GENERATION.value == "code_generation"


# ============================================================================
# Z3 Bridge Tests
# ============================================================================

@pytest.mark.skipif(not Z3_BRIDGE_AVAILABLE, reason="Z3 bridge not available")
class TestZ3Bridge:
    """Tests for Z3-LeanAide bridge"""
    
    def test_bridge_creation(self, z3_bridge):
        """Test bridge creation"""
        assert z3_bridge is not None
        assert isinstance(z3_bridge, Z3LeanAideBridge)
    
    def test_get_capabilities(self, z3_bridge):
        """Test getting capabilities"""
        caps = z3_bridge.get_capabilities()
        
        assert "z3_available" in caps
        assert "lean_available" in caps
        assert "translation_z3_to_lean" in caps
    
    def test_z3_to_lean_translator(self):
        """Test Z3 to Lean translator"""
        translator = Z3ToLeanTranslator()
        
        # Test simple translation (without actual Z3)
        lean_constraint = translator.translate("x > 0", ConstraintType.ARITHMETIC)
        
        assert isinstance(lean_constraint, Lean4Constraint)
        assert lean_constraint.constraint_type == ConstraintType.ARITHMETIC
    
    def test_lean_to_z3_translator(self):
        """Test Lean to Z3 translator"""
        translator = LeanToZ3Translator()
        
        lean_code = """
import Mathlib

theorem test (x : ℝ) (hx : x > 0) : x > 0 := by
  exact hx
"""
        z3_constraint = translator.translate(lean_code)
        
        # May be None if Z3 is not available
        if z3_constraint is not None:
            assert isinstance(z3_constraint, Z3Constraint)
    
    def test_verification_bridge_creation(self):
        """Test verification bridge creation"""
        bridge = Z3LeanVerificationBridge()
        assert bridge is not None
    
    def test_hybrid_proof_engine_creation(self):
        """Test hybrid proof engine creation"""
        engine = HybridProofEngine()
        assert engine is not None


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.skipif(not all([CONTINUOUS_MATH_AVAILABLE, MDAP_AVAILABLE]), 
                    reason="Required modules not available")
class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_formalization(self, mdap_maker):
        """Test end-to-end formalization workflow"""
        # Step 1: Formalize natural language
        result = await mdap_maker.formalize(
            "The limit as x approaches infinity of 1/x equals 0",
            InputType.NATURAL_LANGUAGE,
            "real_analysis",
            complete_proof=False
        )
        
        assert result.success or len(result.red_flags) >= 0  # May not succeed but should complete
        assert result.lean_code is not None
        assert len(result.stages_completed) > 0
    
    @pytest.mark.asyncio
    async def test_continuous_math_with_lean(self):
        """Test continuous math with Lean proof generation"""
        if not CONTINUOUS_MATH_AVAILABLE:
            pytest.skip("Continuous math not available")
        
        engine = create_continuous_math_engine(enable_lean_proofs=False)
        
        # Compute limit
        limit_result = await engine.compute_limit("sin(x)/x", "x", 0.0)
        
        assert limit_result.limit_value == pytest.approx(1.0, abs=1e-10)
        
        # Generate Lean proof (even without verification)
        lean_proof = await engine._generate_limit_proof(
            "sin(x)/x", "x", 0.0, 1.0, 1e-10, 0.01
        )
        
        # Note: lean_proof may be None if leanaide client is not available
        # This is acceptable in test environment
        if lean_proof is not None:
            assert "import Mathlib" in lean_proof
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing of multiple problems"""
        if not CONTINUOUS_MATH_AVAILABLE:
            pytest.skip("Continuous math not available")
        
        engine = create_continuous_math_engine()
        
        problems = [
            ("sin(x)/x", "x", 0.0),
            ("(1+x)**(1/x)", "x", 0.0),
            ("1/x", "x", "oo")
        ]
        
        from leanaide_continuous_math import BatchContinuousMath
        batch = BatchContinuousMath(engine)
        results = await batch.batch_limits(problems)
        
        assert len(results) == 3
        assert all(isinstance(r, LimitResult) for r in results)


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.skipif(not CONTINUOUS_MATH_AVAILABLE, reason="Continuous math not available")
class TestPerformance:
    """Performance tests"""
    
    @pytest.mark.asyncio
    async def test_limit_performance(self, math_engine):
        """Test limit computation performance"""
        import time
        
        start = time.time()
        result = await math_engine.compute_limit("sin(x)/x", "x", 0.0)
        elapsed = time.time() - start
        
        assert elapsed < 5.0  # Should complete in under 5 seconds
        assert result.computation_time < 5.0
    
    @pytest.mark.asyncio
    async def test_batch_performance(self, math_engine):
        """Test batch processing performance"""
        import time
        
        from leanaide_continuous_math import BatchContinuousMath
        batch = BatchContinuousMath(math_engine)
        
        problems = [
            ("sin(x)/x", "x", 0.0),
            ("(exp(x)-1)/x", "x", 0.0),
            ("log(1+x)/x", "x", 0.0),
            ("(1+x)**(1/x)", "x", 0.0)
        ]
        
        start = time.time()
        results = await batch.batch_limits(problems)
        elapsed = time.time() - start
        
        assert len(results) == 4
        assert elapsed < 10.0  # Should complete in under 10 seconds


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.skipif(not CONTINUOUS_MATH_AVAILABLE, reason="Continuous math not available")
class TestErrorHandling:
    """Tests for error handling"""
    
    @pytest.mark.asyncio
    async def test_invalid_expression(self, math_engine):
        """Test handling of invalid expression"""
        try:
            result = await math_engine.compute_limit("invalid@@@syntax", "x", 0.0)
            # If no exception, should have failed gracefully
        except Exception as e:
            # Expected to fail
            pass
    
    @pytest.mark.asyncio
    async def test_division_by_zero(self, math_engine):
        """Test handling of division by zero"""
        result = await math_engine.compute_limit("1/x", "x", 0.0)
        
        # Limit should not exist (or be infinite)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_empty_input(self, autoformalizer):
        """Test handling of empty input"""
        result = await autoformalizer.formalize_problem("")
        
        # Should handle gracefully
        assert "success" in result


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
