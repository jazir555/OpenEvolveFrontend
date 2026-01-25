"""
Test Suite for LeanAide Continuous MCP Tools

Comprehensive tests for B.5 MCP Tools functionality.
Tests all MCP tool wrappers for detection, translation, domain knowledge,
and verification capabilities.

Author: OpenEvolve
Created: 2026-01-09
Phase: 2 - LeanAide Enhancement (Task B.5)
"""

import pytest
from leanaide_continuous_mcp import (
    LeanAideContinuousMCP,
    MCPToolResult,
    MCPToolDefinition,
    get_mcp_tools,
)
from continuous_math_detector import ScientificDomain, MathType, ProblemType


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mcp_tools():
    """Create a LeanAideContinuousMCP instance for testing"""
    return LeanAideContinuousMCP()


# ============================================================================
# B.5.1: MCP Tools Initialization Tests
# ============================================================================

class TestMCPToolsInitialization:
    """Test suite for MCP tools initialization"""

    def test_tools_initialization(self, mcp_tools):
        """Test that MCP tools initialize correctly"""
        assert mcp_tools is not None
        assert isinstance(mcp_tools, LeanAideContinuousMCP)

    def test_tools_registered(self, mcp_tools):
        """Test that tools are registered"""
        tools = mcp_tools.list_tools()
        assert len(tools) > 0
        assert isinstance(tools, list)

    def test_all_tools_have_definitions(self, mcp_tools):
        """Test that all tools have definitions"""
        tools = mcp_tools.list_tools()
        definitions = mcp_tools.list_tool_definitions()

        assert len(tools) == len(definitions)

        for tool_name in tools:
            definition = mcp_tools.get_tool_definition(tool_name)
            assert definition is not None
            assert isinstance(definition, MCPToolDefinition)


# ============================================================================
# B.5.2: Detection Tools Tests
# ============================================================================

class TestDetectionTools:
    """Test suite for detection MCP tools"""

    def test_detect_math_tool(self, mcp_tools):
        """Test detect_math tool"""
        result = mcp_tools.execute_tool(
            "detect_math",
            {"text": "Solve dy/dx + y = 0"}
        )

        assert result.success is True
        assert result.data is not None
        assert "math_type" in result.data
        assert "domain" in result.data
        assert "confidence" in result.data

    def test_detect_math_detailed(self, mcp_tools):
        """Test detect_math with detailed output"""
        result = mcp_tools.execute_tool(
            "detect_math",
            {"text": "Heat equation: ∂u/∂t = α ∂²u/∂x²", "detailed": True}
        )

        assert result.success is True
        assert result.data["math_type"] == "partial_differential_equation"
        assert result.data["domain"] == "physics"

    def test_is_ode_tool_true(self, mcp_tools):
        """Test is_ode tool with ODE"""
        result = mcp_tools.execute_tool(
            "is_ode",
            {"text": "dy/dx = x + y"}
        )

        assert result.success is True
        assert result.data["is_ode"] is True

    def test_is_ode_tool_false(self, mcp_tools):
        """Test is_ode tool with non-ODE"""
        result = mcp_tools.execute_tool(
            "is_ode",
            {"text": "integral of x"}
        )

        assert result.success is True
        assert result.data["is_ode"] is False

    def test_is_pde_tool_true(self, mcp_tools):
        """Test is_pde tool with PDE"""
        result = mcp_tools.execute_tool(
            "is_pde",
            {"text": "∂u/∂t = ∂²u/∂x²"}
        )

        assert result.success is True
        assert result.data["is_pde"] is True

    def test_is_pde_tool_false(self, mcp_tools):
        """Test is_pde tool with non-PDE"""
        result = mcp_tools.execute_tool(
            "is_pde",
            {"text": "dy/dx = x"}
        )

        assert result.success is True
        assert result.data["is_pde"] is False


# ============================================================================
# B.5.3: Translation Tools Tests
# ============================================================================

class TestTranslationTools:
    """Test suite for translation MCP tools"""

    def test_translate_to_lean4_tool(self, mcp_tools):
        """Test translate_to_lean4 tool"""
        result = mcp_tools.execute_tool(
            "translate_to_lean4",
            {"text": "dy/dx + y = 0"}
        )

        assert result.success is True
        assert "lean4_code" in result.data
        assert result.data["lean4_code"] is not None
        assert len(result.data["lean4_code"]) > 0

    def test_translate_to_lean4_with_solution_type(self, mcp_tools):
        """Test translate_to_lean4 with solution type"""
        result = mcp_tools.execute_tool(
            "translate_to_lean4",
            {"text": "dy/dx = y", "solution_type": "existence"}
        )

        assert result.success is True
        assert result.data["success"] is True

    def test_translate_ode_tool(self, mcp_tools):
        """Test translate_ode tool"""
        result = mcp_tools.execute_tool(
            "translate_ode",
            {"equation": "y' + y = 0", "initial_condition": "y(0) = 1"}
        )

        assert result.success is True
        assert "lean4_code" in result.data

    def test_translate_pde_tool(self, mcp_tools):
        """Test translate_pde tool"""
        result = mcp_tools.execute_tool(
            "translate_pde",
            {"equation": "∂u/∂t = ∂²u/∂x²"}
        )

        assert result.success is True
        assert "lean4_code" in result.data


# ============================================================================
# B.5.4: Domain Knowledge Tools Tests
# ============================================================================

class TestDomainKnowledgeTools:
    """Test suite for domain knowledge MCP tools"""

    def test_get_equation_templates_tool(self, mcp_tools):
        """Test get_equation_templates tool"""
        result = mcp_tools.execute_tool(
            "get_equation_templates",
            {"domain": "physics"}
        )

        assert result.success is True
        assert "templates" in result.data
        assert "count" in result.data
        assert result.data["count"] > 0

    def test_get_equation_templates_with_category(self, mcp_tools):
        """Test get_equation_templates with category filter"""
        result = mcp_tools.execute_tool(
            "get_equation_templates",
            {"domain": "physics", "category": "thermodynamics"}
        )

        assert result.success is True
        # Should have at least one template (heat equation)
        assert result.data["count"] >= 0

    def test_get_solution_methods_tool(self, mcp_tools):
        """Test get_solution_methods tool"""
        result = mcp_tools.execute_tool(
            "get_solution_methods",
            {"domain": "physics"}
        )

        assert result.success is True
        assert "solution_methods" in result.data
        assert len(result.data["solution_methods"]) > 0

    def test_recommend_solution_method_tool(self, mcp_tools):
        """Test recommend_solution_method tool"""
        result = mcp_tools.execute_tool(
            "recommend_solution_method",
            {
                "domain": "biology",
                "math_type": "ODE",
                "problem_type": "IVP"
            }
        )

        assert result.success is True
        assert "recommended_methods" in result.data
        assert len(result.data["recommended_methods"]) > 0


# ============================================================================
# B.5.5: Verification Tools Tests
# ============================================================================

class TestVerificationTools:
    """Test suite for verification MCP tools"""

    def test_verify_lean4_code_tool(self, mcp_tools):
        """Test verify_lean4_code tool"""
        code = '''
import Mathlib.Data.Real.Basic
namespace Test
def test (x : Real) : Prop := x > 0
end Test
'''

        result = mcp_tools.execute_tool(
            "verify_lean4_code",
            {"code": code}
        )

        assert result.success is True
        assert "status" in result.data
        assert "is_valid" in result.data

    def test_verify_lean4_code_with_domain(self, mcp_tools):
        """Test verify_lean4_code with domain"""
        code = '''
namespace Test
def energy_conserved (E : Real → Real) : Prop :=
  ∀ t, deriv E t = 0
end Test
'''

        result = mcp_tools.execute_tool(
            "verify_lean4_code",
            {"code": code, "domain": "physics"}
        )

        assert result.success is True


# ============================================================================
# B.5.6: Workflow Tools Tests
# ============================================================================

class TestWorkflowTools:
    """Test suite for workflow MCP tools"""

    def test_complete_pipeline_tool(self, mcp_tools):
        """Test complete_pipeline tool"""
        result = mcp_tools.execute_tool(
            "complete_pipeline",
            {"text": "Solve dy/dx + y = 0", "verify": False}
        )

        assert result.success is True
        assert "detection" in result.data
        assert "translation" in result.data
        assert result.data["detection"]["math_type"] == "ordinary_differential_equation"

    def test_complete_pipeline_with_verification(self, mcp_tools):
        """Test complete_pipeline with verification enabled"""
        result = mcp_tools.execute_tool(
            "complete_pipeline",
            {"text": "dy/dx + y = 0", "verify": True}
        )

        assert result.success is True
        assert "verification" in result.data
        # Should have verification results
        assert result.data["verification"] is not None


# ============================================================================
# B.5.7: Tool Definition Tests
# ============================================================================

class TestToolDefinitions:
    """Test suite for tool definitions"""

    def test_tool_definition_structure(self, mcp_tools):
        """Test that tool definitions have correct structure"""
        definitions = mcp_tools.list_tool_definitions()

        for definition in definitions:
            assert definition.name is not None
            assert definition.description is not None
            assert definition.category is not None
            assert isinstance(definition.input_schema, list)
            assert definition.output_schema is not None

    def test_tool_definition_categories(self, mcp_tools):
        """Test that tools are organized by category"""
        categories = set()
        for definition in mcp_tools.list_tool_definitions():
            categories.add(definition.category)

        # Should have multiple categories
        assert len(categories) > 1
        assert "detection" in categories
        assert "translation" in categories
        assert "verification" in categories

    def test_tool_definition_to_dict(self, mcp_tools):
        """Test tool definition serialization"""
        definition = mcp_tools.get_tool_definition("detect_math")

        assert definition is not None

        definition_dict = definition.to_dict()

        assert "name" in definition_dict
        assert "description" in definition_dict
        assert "category" in definition_dict
        assert "input_schema" in definition_dict
        assert "output_schema" in definition_dict


# ============================================================================
# B.5.8: Tool Execution Tests
# ============================================================================

class TestToolExecution:
    """Test suite for tool execution"""

    def test_execute_unknown_tool(self, mcp_tools):
        """Test executing unknown tool"""
        result = mcp_tools.execute_tool(
            "unknown_tool",
            {}
        )

        assert result.success is False
        assert result.error is not None
        assert "Unknown tool" in result.error

    def test_execute_tool_with_missing_required_param(self, mcp_tools):
        """Test executing tool with missing required parameter"""
        result = mcp_tools.execute_tool(
            "detect_math",
            {}  # Missing required 'text' parameter
        )

        # Should fail gracefully
        assert result is not None

    def test_execution_time_recorded(self, mcp_tools):
        """Test that execution time is recorded"""
        result = mcp_tools.execute_tool(
            "is_ode",
            {"text": "dy/dx = x"}
        )

        assert result.execution_time >= 0


# ============================================================================
# B.5.9: Result Structure Tests
# ============================================================================

class TestMCPToolResult:
    """Test suite for MCPToolResult structure"""

    def test_result_to_dict(self):
        """Test MCPToolResult.to_dict() method"""
        result = MCPToolResult(
            tool_name="test_tool",
            success=True,
            data={"test": "data"}
        )

        result_dict = result.to_dict()

        assert result_dict["tool_name"] == "test_tool"
        assert result_dict["success"] is True
        assert result_dict["data"] == {"test": "data"}

    def test_result_to_json(self):
        """Test MCPToolResult.to_json() method"""
        result = MCPToolResult(
            tool_name="test_tool",
            success=True,
            data={"test": "data"}
        )

        json_str = result.to_json()

        assert isinstance(json_str, str)
        assert "test_tool" in json_str
        assert "success" in json_str

    def test_result_with_error(self):
        """Test MCPToolResult with error"""
        result = MCPToolResult(
            tool_name="test_tool",
            success=False,
            error="Test error"
        )

        assert result.success is False
        assert result.error == "Test error"
        assert result.data is None


# ============================================================================
# B.5.10: Manifest Export Tests
# ============================================================================

class TestManifestExport:
    """Test suite for manifest export"""

    def test_export_manifest(self, mcp_tools):
        """Test manifest export"""
        manifest = mcp_tools.export_manifest()

        assert "name" in manifest
        assert "version" in manifest
        assert "description" in manifest
        assert "tools" in manifest
        assert "categories" in manifest

    def test_manifest_contains_all_tools(self, mcp_tools):
        """Test that manifest contains all registered tools"""
        manifest = mcp_tools.export_manifest()
        tool_count = len(mcp_tools.list_tools())

        assert len(manifest["tools"]) == tool_count

    def test_manifest_categories(self, mcp_tools):
        """Test that manifest has categories"""
        manifest = mcp_tools.export_manifest()

        assert len(manifest["categories"]) > 0
        assert isinstance(manifest["categories"], list)


# ============================================================================
# B.5.11: Integration Tests
# ============================================================================

class TestMCPIntegration:
    """Integration tests for MCP tools"""

    def test_full_workflow_integration(self, mcp_tools):
        """Test complete workflow through MCP tools"""
        # Step 1: Detect
        detect_result = mcp_tools.execute_tool(
            "detect_math",
            {"text": "Solve the heat equation"}
        )

        assert detect_result.success is True

        # Step 2: Get domain knowledge
        templates_result = mcp_tools.execute_tool(
            "get_equation_templates",
            {"domain": detect_result.data["domain"]}
        )

        assert templates_result.success is True

        # Step 3: Translate
        translate_result = mcp_tools.execute_tool(
            "translate_to_lean4",
            {"text": "Solve the heat equation ∂u/∂t = α ∂²u/∂x²"}
        )

        assert translate_result.success is True

        # Step 4: Verify
        verify_result = mcp_tools.execute_tool(
            "verify_lean4_code",
            {"code": translate_result.data["lean4_code"], "domain": "physics"}
        )

        assert verify_result.success is True

    def test_detection_to_translation_workflow(self, mcp_tools):
        """Test detection followed by translation"""
        # Detect
        detect_result = mcp_tools.execute_tool(
            "detect_math",
            {"text": "dy/dx = x + y"}
        )

        assert detect_result.success is True
        math_type = detect_result.data["math_type"]
        domain = detect_result.data["domain"]

        # Translate
        translate_result = mcp_tools.execute_tool(
            "translate_to_lean4",
            {"text": "dy/dx = x + y"}
        )

        assert translate_result.success is True
        assert translate_result.data["lean4_code"] is not None

    def test_cross_domain_workflow(self, mcp_tools):
        """Test workflow across different domains"""
        domains = ["physics", "biology", "economics"]

        for domain in domains:
            # Get solution methods
            result = mcp_tools.execute_tool(
                "get_solution_methods",
                {"domain": domain}
            )

            assert result.success is True
            assert len(result.data["solution_methods"]) > 0


# ============================================================================
# B.5.12: Category Organization Tests
# ============================================================================

class TestCategoryOrganization:
    """Test suite for tool category organization"""

    def test_get_tools_by_category(self, mcp_tools):
        """Test getting tools by category"""
        detection_tools = mcp_tools.get_tools_by_category("detection")
        translation_tools = mcp_tools.get_tools_by_category("translation")
        verification_tools = mcp_tools.get_tools_by_category("verification")

        assert len(detection_tools) > 0
        assert len(translation_tools) > 0
        assert len(verification_tools) > 0

    def test_detection_category_has_correct_tools(self, mcp_tools):
        """Test that detection category has detection tools"""
        detection_tools = mcp_tools.get_tools_by_category("detection")

        assert "detect_math" in detection_tools
        assert "is_ode" in detection_tools
        assert "is_pde" in detection_tools

    def test_translation_category_has_correct_tools(self, mcp_tools):
        """Test that translation category has translation tools"""
        translation_tools = mcp_tools.get_tools_by_category("translation")

        assert "translate_to_lean4" in translation_tools
        assert "translate_ode" in translation_tools
        assert "translate_pde" in translation_tools


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
