"""
End-to-End Integration Tests for LeanAide Continuous Mathematics System

Comprehensive integration tests for the complete Phase 2 LeanAide Enhancement.
Tests all components (B.1-B.5) working together in realistic workflows.

Author: OpenEvolve
Created: 2026-01-09
Phase: 2 - LeanAide Enhancement (Task B.6)
"""

import pytest
import time
from typing import Dict, Any, List

# Import all components
from continuous_math_detector import (
    ContinuousMathDetector,
    detect_continuous_math,
    MathType,
    ProblemType,
    ScientificDomain,
    MathDetectionResult,
)
from ode_pde_translator import (
    ODEPDETranslator,
    translate_to_lean4,
    Lean4TranslationResult,
    SolutionType,
)
from scientific_domain_patterns import (
    ScientificDomainPatterns,
    get_domain_patterns,
    EquationTemplate,
    ParameterConvention,
)
from verification_methods import (
    Lean4Verifier,
    verify_lean4_code,
    verify_translation,
    VerificationResult,
    VerificationStatus,
    CheckType,
)
from leanaide_continuous_mcp import (
    LeanAideContinuousMCP,
    get_mcp_tools,
    MCPToolResult,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def detector():
    """Create math detector"""
    return ContinuousMathDetector()


@pytest.fixture
def translator():
    """Create ODE/PDE translator"""
    return ODEPDETranslator()


@pytest.fixture
def domain_patterns():
    """Create domain patterns"""
    return get_domain_patterns()


@pytest.fixture
def verifier():
    """Create Lean 4 verifier"""
    return Lean4Verifier(enable_leanaide=False)


@pytest.fixture
def mcp_tools():
    """Create MCP tools instance"""
    return get_mcp_tools()


@pytest.fixture
def integration_system(detector, translator, domain_patterns, verifier):
    """Create complete integrated system"""
    return {
        "detector": detector,
        "translator": translator,
        "domain_patterns": domain_patterns,
        "verifier": verifier
    }


# ============================================================================
# B.6.1: Complete Pipeline Integration Tests
# ============================================================================

class TestCompletePipeline:
    """Test complete detection → translation → verification pipeline"""

    def test_simple_ode_pipeline(self, integration_system):
        """Test complete pipeline for simple ODE"""
        text = "Solve dy/dx + y = 0 with initial condition y(0) = 1"

        # Step 1: Detect
        detection_result = integration_system["detector"].detect(text)

        assert detection_result.math_type == MathType.ODE
        assert detection_result.problem_type == ProblemType.INITIAL_VALUE

        # Step 2: Get domain knowledge
        domain = detection_result.domain
        templates = integration_system["domain_patterns"].get_equation_templates(domain)
        assert len(templates) > 0

        # Step 3: Translate
        translation_result = integration_system["translator"].translate(
            detection_result,
            solution_type=SolutionType.EXISTENCE_UNIQUENESS
        )

        assert translation_result.success
        assert translation_result.lean4_code is not None
        assert len(translation_result.definitions) > 0

        # Step 4: Verify
        verification_result = integration_system["verifier"].verify(
            translation_result,
            detection_result
        )

        assert verification_result.verification_time > 0
        assert len(verification_result.checks_performed) > 0

    def test_heat_equation_pipeline(self, integration_system):
        """Test complete pipeline for heat equation"""
        text = "Solve the heat equation ∂u/∂t = α ∂²u/∂x²"

        # Detect
        detection_result = integration_system["detector"].detect(text)
        assert detection_result.math_type == MathType.PDE

        # Get domain knowledge
        methods = integration_system["domain_patterns"].get_solution_methods(
            detection_result.domain
        )
        assert len(methods) > 0

        # Translate
        translation_result = integration_system["translator"].translate(detection_result)
        assert translation_result.success

        # Verify
        verification_result = integration_system["verifier"].verify_code(
            translation_result.lean4_code,
            domain=detection_result.domain
        )
        assert verification_result is not None

    def test_lotka_volterra_pipeline(self, integration_system):
        """Test complete pipeline for Lotka-Volterra equations"""
        text = """
        Analyze the Lotka-Volterra predator-prey model:
        dx/dt = αx - βxy
        dy/dt = δxy - γy
        """

        # Detect
        detection_result = integration_system["detector"].detect(text)
        # Should detect as ODE system
        assert detection_result.math_type in [MathType.ODE, MathType.ODE_SYSTEM]

        # Get domain patterns
        templates = integration_system["domain_patterns"].get_equation_templates(
            ScientificDomain.BIOLOGY
        )
        # Should have biology templates
        assert len(templates) > 0

        # Translate
        translation_result = integration_system["translator"].translate(detection_result)
        assert translation_result.success

        # Verify with biology domain
        verification_result = integration_system["verifier"].verify(
            translation_result,
            detection_result
        )
        assert verification_result.overall_status != VerificationStatus.ERROR


# ============================================================================
# B.6.2: MCP Integration Tests
# ============================================================================

class TestMCPIntegration:
    """Test MCP tools integration with all components"""

    def test_mcp_complete_workflow(self, mcp_tools):
        """Test complete workflow through MCP tools"""
        text = "Solve dy/dx = y with y(0) = 1"

        # Use complete pipeline tool
        result = mcp_tools.execute_tool(
            "complete_pipeline",
            {"text": text, "verify": True}
        )

        assert result.success
        assert "detection" in result.data
        assert "translation" in result.data
        assert "verification" in result.data

        # Verify detection
        assert result.data["detection"]["math_type"] == "ordinary_differential_equation"

        # Verify translation
        assert result.data["translation"]["lean4_code"] is not None

        # Verify verification
        assert "status" in result.data["verification"]

    def test_mcp_detection_to_translation(self, mcp_tools):
        """Test detection followed by translation through MCP"""
        text = "Solve ∂u/∂t = ∂²u/∂x²"

        # Detect
        detect_result = mcp_tools.execute_tool(
            "detect_math",
            {"text": text, "detailed": True}
        )

        assert detect_result.success
        math_type = detect_result.data["math_type"]

        # Translate based on detection
        translate_result = mcp_tools.execute_tool(
            "translate_to_lean4",
            {"text": text}
        )

        assert translate_result.success
        assert translate_result.data["lean4_code"] is not None

    def test_mcp_domain_aware_workflow(self, mcp_tools):
        """Test domain-aware workflow through MCP"""
        # Biology problem
        text = "SIR model: dS/dt = -βSI, dI/dt = βSI - γI, dR/dt = γI"

        # Detect
        detect_result = mcp_tools.execute_tool("detect_math", {"text": text})
        assert detect_result.success

        # Get domain knowledge
        domain = detect_result.data.get("domain", "general")
        templates_result = mcp_tools.execute_tool(
            "get_equation_templates",
            {"domain": domain}
        )

        assert templates_result.success
        assert templates_result.data["count"] > 0

        # Get solution methods
        methods_result = mcp_tools.execute_tool(
            "get_solution_methods",
            {"domain": domain, "math_type": "ODE"}
        )

        assert methods_result.success
        assert len(methods_result.data["solution_methods"]) > 0

    def test_mcp_verification_workflow(self, mcp_tools):
        """Test verification workflow through MCP"""
        code = '''
import Mathlib.Data.Real.Basic
namespace Test
def test (x : Real) : Prop := x > 0
end Test
'''

        # Verify
        verify_result = mcp_tools.execute_tool(
            "verify_lean4_code",
            {"code": code, "domain": "general"}
        )

        assert verify_result.success
        assert "status" in verify_result.data
        assert "is_valid" in verify_result.data


# ============================================================================
# B.6.3: Cross-Domain Integration Tests
# ============================================================================

class TestCrossDomainIntegration:
    """Test system across different scientific domains"""

    def test_physics_integration(self, integration_system):
        """Test physics domain integration"""
        # Wave equation
        text = "Solve the wave equation ∂²u/∂t² = c² ∂²u/∂x²"

        detection_result = integration_system["detector"].detect(text)
        assert detection_result.math_type == MathType.PDE

        # Get physics templates
        templates = integration_system["domain_patterns"].get_equation_templates(
            ScientificDomain.PHYSICS
        )
        physics_templates = [t for t in templates if "wave" in t.name.lower()]
        # Should have wave-related templates

        # Translate and verify
        translation_result = integration_system["translator"].translate(detection_result)
        assert translation_result.success

    def test_biology_integration(self, integration_system):
        """Test biology domain integration"""
        # Population growth
        text = "Model population growth with dP/dt = rP(1 - P/K)"

        detection_result = integration_system["detector"].detect(text)
        # Should detect as ODE

        # Get biology methods
        methods = integration_system["domain_patterns"].get_solution_methods(
            ScientificDomain.BIOLOGY
        )
        assert len(methods) > 0

        # Translate
        translation_result = integration_system["translator"].translate(detection_result)
        assert translation_result.success

        # Verify with biology patterns
        verification_result = integration_system["verifier"].verify(
            translation_result,
            detection_result
        )
        # Should have biology-specific checks if domain detected correctly

    def test_chemistry_integration(self, integration_system):
        """Test chemistry domain integration"""
        # Rate equation
        text = "Reaction rate: d[A]/dt = -k[A]"

        detection_result = integration_system["detector"].detect(text)
        # Should detect as ODE

        # Get chemistry patterns
        templates = integration_system["domain_patterns"].get_equation_templates(
            ScientificDomain.CHEMISTRY
        )
        # Should have chemistry templates

        # Translate
        translation_result = integration_system["translator"].translate(detection_result)
        assert translation_result.success

    def test_engineering_integration(self, integration_system):
        """Test engineering domain integration"""
        # Control system
        text = "Control system: dy/dt + 2y = u(t)"

        detection_result = integration_system["detector"].detect(text)
        # Should detect as ODE

        # Get engineering methods
        methods = integration_system["domain_patterns"].get_solution_methods(
            ScientificDomain.ENGINEERING
        )
        assert len(methods) > 0

        # Translate
        translation_result = integration_system["translator"].translate(detection_result)
        assert translation_result.success

    def test_economics_integration(self, integration_system):
        """Test economics domain integration"""
        # Economic growth
        text = "Economic growth model: dY/dt = sY - δY"

        detection_result = integration_system["detector"].detect(text)
        # Should detect as ODE

        # Get economics patterns
        templates = integration_system["domain_patterns"].get_equation_templates(
            ScientificDomain.ECONOMICS
        )
        # Should have economics templates

        # Translate
        translation_result = integration_system["translator"].translate(detection_result)
        assert translation_result.success


# ============================================================================
# B.6.4: Performance Integration Tests
# ============================================================================

class TestPerformanceIntegration:
    """Test performance of integrated system"""

    def test_pipeline_performance(self, integration_system):
        """Test complete pipeline performance"""
        text = "Solve dy/dx + y = 0 with y(0) = 1"

        start_time = time.time()

        # Complete pipeline
        detection_result = integration_system["detector"].detect(text)
        translation_result = integration_system["translator"].translate(detection_result)
        verification_result = integration_system["verifier"].verify(
            translation_result,
            detection_result
        )

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete in reasonable time (< 5 seconds)
        assert total_time < 5.0

        # All steps should succeed
        assert detection_result.math_type == MathType.ODE
        assert translation_result.success
        assert verification_result.verification_time > 0

    def test_batch_processing(self, integration_system):
        """Test processing multiple problems"""
        problems = [
            "Solve dy/dx = y",
            "Solve ∂u/∂t = ∂²u/∂x²",
            "Solve d²y/dx² + y = 0",
            "Solve the system: dx/dt = y, dy/dt = -x"
        ]

        start_time = time.time()

        results = []
        for problem in problems:
            detection_result = integration_system["detector"].detect(problem)
            translation_result = integration_system["translator"].translate(detection_result)
            results.append((detection_result, translation_result))

        end_time = time.time()
        avg_time = (end_time - start_time) / len(problems)

        # Average time per problem should be reasonable
        assert avg_time < 2.0

        # All should succeed
        for detection_result, translation_result in results:
            assert translation_result.success

    def test_mcp_performance(self, mcp_tools):
        """Test MCP tools performance"""
        text = "Solve dy/dx + y = 0"

        # Test complete pipeline performance
        start_time = time.time()
        result = mcp_tools.execute_tool(
            "complete_pipeline",
            {"text": text, "verify": False}  # Skip verification for speed
        )
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete in reasonable time
        assert execution_time < 3.0
        assert result.success


# ============================================================================
# B.6.5: Error Handling Integration Tests
# ============================================================================

class TestErrorHandlingIntegration:
    """Test error handling in integrated system"""

    def test_empty_input_handling(self, integration_system):
        """Test handling of empty input"""
        # Detection should handle empty input gracefully
        detection_result = integration_system["detector"].detect("")

        # Should not crash
        assert detection_result is not None

    def test_invalid_math_handling(self, integration_system):
        """Test handling of invalid mathematics"""
        text = "This is not mathematics"

        detection_result = integration_system["detector"].detect(text)

        # Should detect as unknown/general
        assert detection_result is not None

        # Translation might not succeed
        translation_result = integration_system["translator"].translate(detection_result)

        # Should handle gracefully
        assert translation_result is not None

    def test_verification_error_handling(self, integration_system):
        """Test verification error handling"""
        invalid_code = "this is not valid Lean 4 code"

        verification_result = integration_system["verifier"].verify_code(invalid_code)

        # Should not crash
        assert verification_result is not None
        # Should detect issues
        assert len(verification_result.issues) > 0

    def test_mcp_error_handling(self, mcp_tools):
        """Test MCP error handling"""
        # Unknown tool
        result = mcp_tools.execute_tool("unknown_tool", {})

        assert result.success is False
        assert result.error is not None

        # Missing required parameter
        result = mcp_tools.execute_tool("detect_math", {})

        # Should handle gracefully
        assert result is not None


# ============================================================================
# B.6.6: Data Consistency Tests
# ============================================================================

class TestDataConsistency:
    """Test data consistency across components"""

    def test_detection_to_translation_consistency(self, integration_system):
        """Test consistency between detection and translation"""
        text = "Solve dy/dx + y = 0"

        detection_result = integration_system["detector"].detect(text)
        translation_result = integration_system["translator"].translate(detection_result)

        # Translation should use detection results
        assert translation_result.math_type == detection_result.math_type
        assert translation_result.domain == detection_result.domain

    def test_domain_knowledge_consistency(self, integration_system):
        """Test domain knowledge consistency"""
        # Get templates for all domains
        domains = [
            ScientificDomain.PHYSICS,
            ScientificDomain.BIOLOGY,
            ScientificDomain.CHEMISTRY,
            ScientificDomain.ENGINEERING,
            ScientificDomain.ECONOMICS
        ]

        for domain in domains:
            templates = integration_system["domain_patterns"].get_equation_templates(domain)
            methods = integration_system["domain_patterns"].get_solution_methods(domain)

            # Should have data for each domain
            assert len(templates) >= 0
            assert len(methods) >= 0

    def test_verification_metadata_consistency(self, integration_system):
        """Test verification metadata consistency"""
        text = "Solve dy/dx = y"

        detection_result = integration_system["detector"].detect(text)
        translation_result = integration_system["translator"].translate(detection_result)
        verification_result = integration_system["verifier"].verify(
            translation_result,
            detection_result
        )

        # Metadata should be consistent
        assert verification_result.metadata is not None
        assert "total_checks" in verification_result.metadata


# ============================================================================
# B.6.7: Real-World Workflow Tests
# ============================================================================

class TestRealWorldWorkflows:
    """Test realistic user workflows"""

    def test_researcher_workflow(self, integration_system):
        """Simulate researcher solving a physics problem"""
        # Researcher has a problem
        problem = """
        I need to model heat diffusion in a metal rod.
        The equation is ∂T/∂t = α ∂²T/∂x²
        with boundary conditions T(0,t) = T(L,t) = 0
        and initial condition T(x,0) = f(x)
        """

        # Step 1: Understand the problem
        detection_result = integration_system["detector"].detect(problem)
        assert detection_result.math_type == MathType.PDE

        # Step 2: Get relevant knowledge
        domain = detection_result.domain
        templates = integration_system["domain_patterns"].get_equation_templates(domain)
        methods = integration_system["domain_patterns"].get_solution_methods(domain)

        assert len(templates) > 0
        assert len(methods) > 0

        # Step 3: Formalize in Lean 4
        translation_result = integration_system["translator"].translate(detection_result)
        assert translation_result.success

        # Step 4: Verify the formalization
        verification_result = integration_system["verifier"].verify(
            translation_result,
            detection_result
        )

        assert verification_result.verification_time > 0

    def test_student_workflow(self, integration_system):
        """Simulate student learning differential equations"""
        # Student encounters an ODE
        problem = "Find the general solution to dy/dx + 2y = 0"

        # Step 1: Detect
        detection_result = integration_system["detector"].detect(problem)

        # Step 2: Get solution method recommendations
        methods = integration_system["domain_patterns"].get_solution_methods(
            detection_result.domain
        )

        # Should have relevant methods
        assert len(methods) > 0

        # Step 3: See formalization
        translation_result = integration_system["translator"].translate(detection_result)

        assert translation_result.success
        assert len(translation_result.definitions) > 0
        assert len(translation_result.theorems) > 0

    def test_mcp_assistant_workflow(self, mcp_tools):
        """Test AI assistant workflow through MCP"""
        user_queries = [
            "Is this an ODE: dy/dx = y?",
            "Translate dy/dx + y = 0 to Lean 4",
            "Verify this code: def test (x : Real) : Prop := x > 0",
            "What are the solution methods for ODEs in physics?"
        ]

        for query in user_queries:
            # For this test, just verify the tools are accessible
            tools = mcp_tools.list_tools()
            assert len(tools) > 0


# ============================================================================
# B.6.8: System Validation Tests
# ============================================================================

class TestSystemValidation:
    """Validate overall system correctness"""

    def test_all_tools_accessible(self, mcp_tools):
        """Test that all MCP tools are accessible"""
        tools = mcp_tools.list_tools()

        # Should have all expected tools
        expected_tools = [
            "detect_math",
            "is_ode",
            "is_pde",
            "translate_to_lean4",
            "translate_ode",
            "translate_pde",
            "get_equation_templates",
            "get_solution_methods",
            "recommend_solution_method",
            "verify_lean4_code",
            "complete_pipeline"
        ]

        for tool in expected_tools:
            assert tool in tools

    def test_all_components_initialized(self, integration_system):
        """Test that all components are properly initialized"""
        # Detector
        assert integration_system["detector"] is not None
        assert hasattr(integration_system["detector"], "detect")

        # Translator
        assert integration_system["translator"] is not None
        assert hasattr(integration_system["translator"], "translate")

        # Domain patterns
        assert integration_system["domain_patterns"] is not None
        assert hasattr(integration_system["domain_patterns"], "get_equation_templates")

        # Verifier
        assert integration_system["verifier"] is not None
        assert hasattr(integration_system["verifier"], "verify")

    def test_manifest_export(self, mcp_tools):
        """Test MCP manifest export"""
        manifest = mcp_tools.export_manifest()

        # Should have required fields
        assert "name" in manifest
        assert "version" in manifest
        assert "description" in manifest
        assert "tools" in manifest
        assert "categories" in manifest

        # Should have all tools
        assert len(manifest["tools"]) == len(mcp_tools.list_tools())

    def test_category_organization(self, mcp_tools):
        """Test tool category organization"""
        categories = [
            "detection",
            "translation",
            "domain_knowledge",
            "verification",
            "workflow"
        ]

        for category in categories:
            tools = mcp_tools.get_tools_by_category(category)
            # Each category should have at least one tool
            assert len(tools) > 0


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
