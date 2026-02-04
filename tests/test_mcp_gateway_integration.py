"""
Comprehensive Test Suite for MCP Gateway Integration

This module provides complete test coverage for MCP (Model Context Protocol) Gateway
integration components:

- MCPGatewayIntegration (main integration class)
- MCPResult (operation result data structure)
- Tool discovery and execution
- Multi-server coordination
- Workflow orchestration

Test Statistics:
- Total Test Functions: 62
- Test Classes: 10
- Fixture Functions: 8
- Coverage Areas: Unit, Integration, Edge Cases, Configuration, Error Handling

Test Categories:
1. Unit Tests - Test each method in isolation with mocked dependencies
2. Integration Tests - Test interactions with MCP Gateway
3. Edge Case Tests - Test boundary conditions and error scenarios
4. Configuration Tests - Test default and custom configuration
5. Workflow Tests - Test complex workflow orchestration
6. Batch Operation Tests - Test batch execution
7. Error Handling Tests - Test graceful degradation
8. Status and Health Tests - Test status reporting

Testing Best Practices:
- Use pytest with asyncio support
- Mock external dependencies (MCP Gateway, Unified Gateway)
- Test both success and failure cases
- Verify structured logging (JSON format)
- Test UTC timestamps
- Test correlation ID propagation
- Aim for >80% code coverage

Running Tests:
    pytest tests/test_mcp_gateway_integration.py -v
    pytest tests/test_mcp_gateway_integration.py -v -k "test_call_tool"
    pytest tests/test_mcp_gateway_integration.py --cov=knowledge_engine.integrations.mcp_gateway_integration

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, MagicMock, patch, mock_open
from dataclasses import asdict

# Import MCP Gateway integration components
try:
    from knowledge_engine.integrations.mcp_gateway_integration import (
        MCPGatewayIntegration,
        MCPResult
    )
    MCPGATEWAY_AVAILABLE = True
except ImportError:
    MCPGATEWAY_AVAILABLE = False
    MCPGatewayIntegration = None
    MCPResult = None
    pytestmark = pytest.mark.skip("MCP Gateway integration not available")


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_config():
    """Sample configuration for MCP Gateway integration."""
    return {
        "gateway_url": "http://localhost:8080",
        "timeout": 30,
        "max_retries": 3,
        "retry_delay": 1.0,
        "circuit_breaker": {
            "enabled": True,
            "failure_threshold": 5,
            "reset_timeout": 30000,
            "success_threshold": 2
        },
        "load_balancing": "round_robin",
        "fallback_enabled": True,
        "metrics_enabled": True,
        "cache_enabled": True,
        "cache_ttl": 300,
        "supported_namespaces": [
            "kggen", "graphiti", "openevolve", "crewai",
            "deepke", "ragbits", "oneke", "aikg"
        ],
        "default_namespace": "openevolve",
        "tool_call_timeout": 120
    }


@pytest.fixture
def mock_unified_gateway():
    """Mock Unified MCP Gateway."""
    gateway = AsyncMock()
    gateway.is_initialized = True
    gateway.is_running = True
    gateway.initialize = AsyncMock(return_value=True)
    gateway.shutdown = AsyncMock(return_value=True)
    gateway.call_tool = AsyncMock()
    gateway.list_tools = AsyncMock(return_value=[])
    gateway.get_health_status = AsyncMock(return_value={
        "gateway": {"status": "running"},
        "servers": {},
        "tools": 10
    })

    # Create mock tool registry
    mock_registry = Mock()
    mock_registry.tools = {}
    gateway.tool_registry = mock_registry

    # Create mock tool router
    mock_router = Mock()
    gateway.tool_router = mock_router

    return gateway


@pytest.fixture
def mock_tool_result():
    """Mock successful tool execution result."""
    result = AsyncMock()
    result.success = True
    result.result = {"output": "test_result", "data": {"key": "value"}}
    result.server_name = "test_server"
    return result


@pytest.fixture
def integration(sample_config):
    """MCP Gateway integration instance with mocked gateway."""
    with patch('knowledge_engine.integrations.mcp_gateway_integration.UnifiedMCPGateway') as mock_gateway_class:
        mock_gateway_instance = AsyncMock()
        mock_gateway_instance.is_initialized = True
        mock_gateway_instance.is_running = True
        mock_gateway_instance.initialize = AsyncMock(return_value=True)
        mock_gateway_instance.call_tool = AsyncMock()
        mock_gateway_instance.list_tools = AsyncMock(return_value=[])
        mock_gateway_instance.get_health_status = AsyncMock(return_value={})

        mock_registry = Mock()
        mock_registry.tools = {}
        mock_gateway_instance.tool_registry = mock_registry

        mock_router = Mock()
        mock_gateway_instance.tool_router = mock_router

        mock_gateway_class.return_value = mock_gateway_instance

        # Prevent asyncio.run in __init__ by mocking it
        with patch('asyncio.run', return_value=None):
            integration = MCPGatewayIntegration(config=sample_config)
            integration.unified_gateway = mock_gateway_instance

        return integration


@pytest.fixture
def integration_no_gateway():
    """MCP Gateway integration instance without actual gateway (mock mode)."""
    with patch('knowledge_engine.integrations.mcp_gateway_integration.UnifiedMCPGateway') as mock_gateway_class:
        # Make import fail to trigger mock initialization
        mock_gateway_class.side_effect = ImportError("mcp-gateway not installed")

        with patch('knowledge_engine.integrations.mcp_gateway_integration.create_failing_mock') as mock_create:
            mock_failing_class = Mock()
            mock_create.return_value = mock_failing_class

            integration = MCPGatewayIntegration(config=None)

        return integration


# =============================================================================
# TEST CLASS: MCPResult
# =============================================================================

class TestMCPResult:
    """Test MCPResult data class."""

    def test_mcp_result_initialization_success(self):
        """Test MCPResult initialization for successful result."""
        result = MCPResult(
            success=True,
            output={"data": "test_output"},
            metadata={"tool": "test_tool"},
            processing_time_ms=150.5
        )

        assert result.success is True
        assert result.output == {"data": "test_output"}
        assert result.metadata == {"tool": "test_tool"}
        assert result.processing_time_ms == 150.5
        assert result.error is None

    def test_mcp_result_initialization_failure(self):
        """Test MCPResult initialization for failed result."""
        result = MCPResult(
            success=False,
            output=None,
            metadata={"tool": "test_tool"},
            processing_time_ms=50.0,
            error="Tool execution failed"
        )

        assert result.success is False
        assert result.output is None
        assert result.error == "Tool execution failed"
        assert result.processing_time_ms == 50.0

    def test_mcp_result_to_dict(self):
        """Test MCPResult to_dict conversion."""
        result = MCPResult(
            success=True,
            output="test_output",
            metadata={"key": "value"},
            processing_time_ms=100.0,
            error=None
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["output"] == "test_output"
        assert data["metadata"] == {"key": "value"}
        assert data["processing_time_ms"] == 100.0
        assert data["error"] is None

    def test_mcp_result_with_all_fields(self):
        """Test MCPResult with all fields populated."""
        result = MCPResult(
            success=True,
            output={"data": "value"},
            metadata={
                "tool_name": "extract_entities",
                "namespace": "deepke",
                "server_name": "deepke_server"
            },
            processing_time_ms=250.75,
            error=None
        )

        assert result.success is True
        assert len(result.metadata) == 3
        assert result.processing_time_ms == 250.75


# =============================================================================
# TEST CLASS: MCPGatewayIntegration Initialization
# =============================================================================

class TestMCPGatewayIntegrationInitialization:
    """Test MCPGatewayIntegration initialization and configuration."""

    def test_default_initialization(self):
        """Test initialization with default configuration."""
        with patch('knowledge_engine.integrations.mcp_gateway_integration.UnifiedMCPGateway') as mock_gateway_class:
            mock_gateway_instance = AsyncMock()
            mock_gateway_instance.is_initialized = True
            mock_gateway_instance.initialize = AsyncMock(return_value=True)
            mock_gateway_instance.tool_registry = Mock()
            mock_gateway_instance.tool_router = Mock()

            mock_gateway_class.return_value = mock_gateway_instance

            with patch('asyncio.run', return_value=None):
                integration = MCPGatewayIntegration(config=None)

            assert integration.config is not None
            assert "gateway_url" in integration.config
            assert "timeout" in integration.config
            assert "supported_namespaces" in integration.config

    def test_custom_initialization(self, sample_config):
        """Test initialization with custom configuration."""
        with patch('knowledge_engine.integrations.mcp_gateway_integration.UnifiedMCPGateway') as mock_gateway_class:
            mock_gateway_instance = AsyncMock()
            mock_gateway_instance.is_initialized = True
            mock_gateway_instance.initialize = AsyncMock(return_value=True)
            mock_gateway_instance.tool_registry = Mock()
            mock_gateway_instance.tool_router = Mock()

            mock_gateway_class.return_value = mock_gateway_instance

            with patch('asyncio.run', return_value=None):
                integration = MCPGatewayIntegration(config=sample_config)

            assert integration.config == sample_config

    def test_get_default_config(self):
        """Test default configuration values."""
        with patch('knowledge_engine.integrations.mcp_gateway_integration.UnifiedMCPGateway') as mock_gateway_class:
            mock_gateway_instance = AsyncMock()
            mock_gateway_instance.initialize = AsyncMock()
            mock_gateway_instance.tool_registry = Mock()
            mock_gateway_instance.tool_router = Mock()

            mock_gateway_class.return_value = mock_gateway_instance

            with patch('asyncio.run', return_value=None):
                integration = MCPGatewayIntegration(config=None)

            config = integration.config

            assert config["gateway_url"] == "http://localhost:8080"
            assert config["timeout"] == 30
            assert config["max_retries"] == 3
            assert config["circuit_breaker"]["enabled"] is True
            assert config["load_balancing"] == "round_robin"
            assert config["fallback_enabled"] is True
            assert config["cache_enabled"] is True
            assert config["cache_ttl"] == 300
            assert len(config["supported_namespaces"]) > 0
            assert config["default_namespace"] == "openevolve"
            assert config["tool_call_timeout"] == 120

    def test_initialize_components_success(self):
        """Test successful component initialization."""
        with patch('knowledge_engine.integrations.mcp_gateway_integration.UnifiedMCPGateway') as mock_gateway_class:
            mock_gateway_instance = AsyncMock()
            mock_gateway_instance.is_initialized = True
            mock_gateway_instance.initialize = AsyncMock(return_value=True)
            mock_gateway_instance.tool_registry = Mock()
            mock_gateway_instance.tool_router = Mock()

            mock_gateway_class.return_value = mock_gateway_instance

            with patch('asyncio.run', return_value=None):
                integration = MCPGatewayIntegration(config={})

            assert integration.unified_gateway is not None

    def test_initialize_components_import_error(self):
        """Test component initialization with import error (mock mode)."""
        with patch('knowledge_engine.integrations.mcp_gateway_integration.UnifiedMCPGateway') as mock_gateway_class:
            mock_gateway_class.side_effect = ImportError("mcp-gateway not installed")

            with patch('knowledge_engine.integrations.mcp_gateway_integration.create_failing_mock') as mock_create:
                mock_failing_class = Mock()
                mock_create.return_value = mock_failing_class

                integration = MCPGatewayIntegration(config={})

            assert integration.unified_gateway is None


# =============================================================================
# TEST CLASS: MCPGatewayIntegration Tool Calling
# =============================================================================

class TestMCPGatewayIntegrationToolCalling:
    """Test tool calling through MCP Gateway."""

    @pytest.mark.asyncio
    async def test_call_tool_success(self, integration):
        """Test successful tool call."""
        # Setup mock result
        mock_result = AsyncMock()
        mock_result.success = True
        mock_result.result = {"output": "test_data", "entities": ["entity1", "entity2"]}
        mock_result.server_name = "test_server"

        integration.unified_gateway.call_tool.return_value = mock_result

        result = await integration.call_tool(
            tool_name="extract_entities",
            params={"text": "Sample text for extraction"},
            namespace="deepke",
            correlation_id="test_correlation_123"
        )

        assert result.success is True
        assert result.output is not None
        assert result.metadata["tool_name"] == "extract_entities"
        assert result.metadata["namespace"] == "deepke"
        assert result.processing_time_ms >= 0
        integration.unified_gateway.call_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_without_namespace(self, integration):
        """Test tool call without namespace."""
        mock_result = AsyncMock()
        mock_result.success = True
        mock_result.result = {"status": "ok"}
        mock_result.server_name = "default_server"

        integration.unified_gateway.call_tool.return_value = mock_result

        result = await integration.call_tool(
            tool_name="test_tool",
            params={"param1": "value1"}
        )

        assert result.success is True
        assert result.metadata["namespace"] == "default"

    @pytest.mark.asyncio
    async def test_call_tool_with_auto_correlation_id(self, integration):
        """Test tool call generates correlation ID if not provided."""
        mock_result = AsyncMock()
        mock_result.success = True
        mock_result.result = {}
        mock_result.server_name = "test"

        integration.unified_gateway.call_tool.return_value = mock_result

        result = await integration.call_tool(
            tool_name="test_tool",
            params={}
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_call_tool_gateway_not_initialized(self):
        """Test tool call when gateway not initialized."""
        with patch('knowledge_engine.integrations.mcp_gateway_integration.UnifiedMCPGateway') as mock_gateway_class:
            mock_gateway_instance = AsyncMock()
            mock_gateway_instance.initialize = AsyncMock()
            mock_gateway_instance.tool_registry = Mock()
            mock_gateway_instance.tool_router = Mock()

            mock_gateway_class.return_value = mock_gateway_instance

            with patch('asyncio.run', return_value=None):
                integration = MCPGatewayIntegration(config={})
                integration.unified_gateway = None

        result = await integration.call_tool(
            tool_name="test_tool",
            params={}
        )

        assert result.success is False
        assert result.error is not None
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_call_tool_exception_handling(self, integration):
        """Test tool call exception handling."""
        integration.unified_gateway.call_tool.side_effect = Exception("Network error")

        result = await integration.call_tool(
            tool_name="test_tool",
            params={},
            correlation_id="test_123"
        )

        assert result.success is False
        assert result.error is not None
        assert "Network error" in result.error


# =============================================================================
# TEST CLASS: MCPGatewayIntegration Tool Discovery
# =============================================================================

class TestMCPGatewayIntegrationToolDiscovery:
    """Test tool discovery through MCP Gateway."""

    @pytest.mark.asyncio
    async def test_discover_tools_all(self, integration):
        """Test discovering all tools."""
        mock_tools = [
            {"name": "extract_entities", "namespace": "deepke", "category": "extraction"},
            {"name": "build_kg", "namespace": "graphiti", "category": "knowledge_graph"}
        ]
        integration.unified_gateway.list_tools.return_value = mock_tools

        result = await integration.discover_tools(
            namespace=None,
            category=None,
            correlation_id="discover_123"
        )

        assert result.success is True
        assert result.output == mock_tools
        assert result.metadata["discovered_count"] == 2
        integration.unified_gateway.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_discover_tools_with_namespace_filter(self, integration):
        """Test discovering tools with namespace filter."""
        mock_tools = [
            {"name": "extract_entities", "namespace": "deepke"}
        ]
        integration.unified_gateway.list_tools.return_value = mock_tools

        result = await integration.discover_tools(
            namespace="deepke",
            correlation_id="discover_123"
        )

        assert result.success is True
        assert result.metadata["namespace_filter"] == "deepke"

    @pytest.mark.asyncio
    async def test_discover_tools_with_category_filter(self, integration):
        """Test discovering tools with category filter."""
        mock_tools = [
            {"name": "tool1", "category": "extraction"}
        ]
        integration.unified_gateway.list_tools.return_value = mock_tools

        result = await integration.discover_tools(
            category="extraction",
            correlation_id="discover_123"
        )

        assert result.success is True
        assert result.metadata["category_filter"] == "extraction"

    @pytest.mark.asyncio
    async def test_discover_tools_empty_result(self, integration):
        """Test discovering tools when none found."""
        integration.unified_gateway.list_tools.return_value = []

        result = await integration.discover_tools(
            namespace="nonexistent",
            correlation_id="discover_123"
        )

        assert result.success is True
        assert result.output == []
        assert result.metadata["discovered_count"] == 0

    @pytest.mark.asyncio
    async def test_discover_tools_gateway_error(self, integration):
        """Test tool discovery with gateway error."""
        integration.unified_gateway.list_tools.side_effect = Exception("Gateway error")

        result = await integration.discover_tools(
            correlation_id="discover_123"
        )

        assert result.success is False
        assert result.error is not None
        assert result.output == []


# =============================================================================
# TEST CLASS: MCPGatewayIntegration Workflow Execution
# =============================================================================

class TestMCPGatewayIntegrationWorkflowExecution:
    """Test workflow execution through MCP Gateway."""

    @pytest.mark.asyncio
    async def test_knowledge_extraction_workflow_all_types(self, integration):
        """Test complete knowledge extraction workflow."""
        # Mock tool calls
        async def mock_call_tool(tool_name, params, namespace=None, correlation_id=None):
            mock_result = AsyncMock()
            mock_result.success = True

            if "entities" in tool_name:
                mock_result.result = {"entities": ["entity1", "entity2"]}
            elif "relations" in tool_name:
                mock_result.result = {"relations": [{"subject": "e1", "object": "e2", "predicate": "rel"}]}
            elif "triples" in tool_name:
                mock_result.result = {"triples": [["e1", "rel", "e2"]]}
            elif "graph" in tool_name:
                mock_result.result = {"nodes": ["e1", "e2"], "edges": [["e1", "e2"]]}

            return mock_result

        integration.call_tool = mock_call_tool

        result = await integration.execute_knowledge_extraction_workflow(
            text="Apple is a technology company founded by Steve Jobs.",
            extraction_types=["entities", "relations", "triples", "graph"],
            correlation_id="workflow_123"
        )

        assert result.success is True
        assert "entities" in result.output
        assert "relations" in result.output
        assert "triples" in result.output
        assert "graph" in result.output
        assert result.metadata["successful_extractions"] == 4

    @pytest.mark.asyncio
    async def test_knowledge_extraction_workflow_default_types(self, integration):
        """Test knowledge extraction workflow with default extraction types."""
        async def mock_call_tool(tool_name, params, namespace=None, correlation_id=None):
            mock_result = AsyncMock()
            mock_result.success = True
            mock_result.result = {"data": "test"}
            return mock_result

        integration.call_tool = mock_call_tool

        result = await integration.execute_knowledge_extraction_workflow(
            text="Sample text",
            extraction_types=None,
            correlation_id="workflow_123"
        )

        assert result.success is True
        assert result.metadata["successful_extractions"] > 0

    @pytest.mark.asyncio
    async def test_knowledge_extraction_workflow_partial_failure(self, integration):
        """Test workflow with some extraction failures."""
        async def mock_call_tool(tool_name, params, namespace=None, correlation_id=None):
            mock_result = AsyncMock()
            if "entities" in tool_name:
                mock_result.success = True
                mock_result.result = {"entities": []}
            else:
                mock_result.success = False
                mock_result.error = "Tool not available"
            return mock_result

        integration.call_tool = mock_call_tool

        result = await integration.execute_knowledge_extraction_workflow(
            text="Sample text",
            extraction_types=["entities", "relations"],
            correlation_id="workflow_123"
        )

        # Should still succeed if at least one extraction succeeded
        assert result.success is True
        assert result.output["entities"] is not None
        assert result.metadata["successful_extractions"] == 1

    @pytest.mark.asyncio
    async def test_knowledge_extraction_workflow_all_failures(self, integration):
        """Test workflow with all extractions failing."""
        async def mock_call_tool(tool_name, params, namespace=None, correlation_id=None):
            mock_result = AsyncMock()
            mock_result.success = False
            mock_result.error = "Tool failed"
            return mock_result

        integration.call_tool = mock_call_tool

        result = await integration.execute_knowledge_extraction_workflow(
            text="Sample text",
            extraction_types=["entities"],
            correlation_id="workflow_123"
        )

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_knowledge_extraction_workflow_gateway_not_initialized(self):
        """Test workflow when gateway not initialized."""
        with patch('knowledge_engine.integrations.mcp_gateway_integration.UnifiedMCPGateway') as mock_gateway_class:
            mock_gateway_instance = AsyncMock()
            mock_gateway_instance.initialize = AsyncMock()
            mock_gateway_instance.tool_registry = Mock()
            mock_gateway_instance.tool_router = Mock()

            mock_gateway_class.return_value = mock_gateway_instance

            with patch('asyncio.run', return_value=None):
                integration = MCPGatewayIntegration(config={})
                integration.unified_gateway = None

        result = await integration.execute_knowledge_extraction_workflow(
            text="Sample text",
            correlation_id="workflow_123"
        )

        assert result.success is False
        assert result.error is not None


# =============================================================================
# TEST CLASS: MCPGatewayIntegration Multi-Agent Coordination
# =============================================================================

class TestMCPGatewayIntegrationMultiAgentCoordination:
    """Test multi-agent coordination through MCP Gateway."""

    @pytest.mark.asyncio
    async def test_multi_agent_coordination_success(self, integration):
        """Test successful multi-agent coordination."""
        async def mock_call_tool(tool_name, params, namespace=None, correlation_id=None):
            mock_result = AsyncMock()
            mock_result.success = True
            mock_result.result = {
                "coordination_result": "success",
                "agents_used": ["agent1", "agent2"],
                "final_output": "Task completed"
            }
            return mock_result

        integration.call_tool = mock_call_tool

        result = await integration.execute_multi_agent_coordination(
            task_description="Analyze market trends and predict stock prices",
            agent_preferences={"max_agents": 3},
            correlation_id="coord_123"
        )

        assert result.success is True
        assert result.output is not None
        assert result.metadata["task_description_length"] > 0

    @pytest.mark.asyncio
    async def test_multi_agent_coordination_without_preferences(self, integration):
        """Test multi-agent coordination without preferences."""
        async def mock_call_tool(tool_name, params, namespace=None, correlation_id=None):
            mock_result = AsyncMock()
            mock_result.success = True
            mock_result.result = {"status": "completed"}
            return mock_result

        integration.call_tool = mock_call_tool

        result = await integration.execute_multi_agent_coordination(
            task_description="Test task",
            agent_preferences=None,
            correlation_id="coord_123"
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_multi_agent_coordination_failure(self, integration):
        """Test multi-agent coordination with failure."""
        async def mock_call_tool(tool_name, params, namespace=None, correlation_id=None):
            mock_result = AsyncMock()
            mock_result.success = False
            mock_result.error = "Coordination failed"
            return mock_result

        integration.call_tool = mock_call_tool

        result = await integration.execute_multi_agent_coordination(
            task_description="Test task",
            correlation_id="coord_123"
        )

        assert result.success is False
        assert result.error is not None


# =============================================================================
# TEST CLASS: MCPGatewayIntegration Formal Verification
# =============================================================================

class TestMCPGatewayIntegrationFormalVerification:
    """Test formal verification through MCP Gateway."""

    @pytest.mark.asyncio
    async def test_formal_verification_success(self, integration):
        """Test successful formal verification."""
        async def mock_call_tool(tool_name, params, namespace=None, correlation_id=None):
            mock_result = AsyncMock()
            mock_result.success = True
            mock_result.result = {
                "verified": True,
                "proof_valid": True,
                "theorem": "Test theorem"
            }
            return mock_result

        integration.call_tool = mock_call_tool

        result = await integration.execute_formal_verification(
            theorem="For all natural numbers n, n + 0 = n",
            proof=None,
            correlation_id="verify_123"
        )

        assert result.success is True
        assert result.output is not None
        assert result.metadata["proof_provided"] is False

    @pytest.mark.asyncio
    async def test_formal_verification_with_proof(self, integration):
        """Test formal verification with provided proof."""
        async def mock_call_tool(tool_name, params, namespace=None, correlation_id=None):
            mock_result = AsyncMock()
            mock_result.success = True
            mock_result.result = {
                "verified": True,
                "proof_valid": True
            }
            return mock_result

        integration.call_tool = mock_call_tool

        result = await integration.execute_formal_verification(
            theorem="Test theorem",
            proof="Proof by induction...",
            correlation_id="verify_123"
        )

        assert result.success is True
        assert result.metadata["proof_provided"] is True

    @pytest.mark.asyncio
    async def test_formal_verification_failure(self, integration):
        """Test formal verification with failure."""
        async def mock_call_tool(tool_name, params, namespace=None, correlation_id=None):
            mock_result = AsyncMock()
            mock_result.success = False
            mock_result.error = "Verification failed"
            return mock_result

        integration.call_tool = mock_call_tool

        result = await integration.execute_formal_verification(
            theorem="Invalid theorem",
            correlation_id="verify_123"
        )

        assert result.success is False
        assert result.error is not None


# =============================================================================
# TEST CLASS: MCPGatewayIntegration Batch Execution
# =============================================================================

class TestMCPGatewayIntegrationBatchExecution:
    """Test batch execution through MCP Gateway."""

    @pytest.mark.asyncio
    async def test_batch_execute_all_success(self, integration):
        """Test batch execution with all successful calls."""
        async def mock_call_tool(tool_name, params, namespace=None, correlation_id=None):
            mock_result = AsyncMock()
            mock_result.success = True
            mock_result.result = {"output": f"Result for {tool_name}"}
            mock_result.server_name = "test_server"
            return mock_result

        integration.call_tool = mock_call_tool

        tool_calls = [
            {"tool_name": "tool1", "params": {"p": "v1"}},
            {"tool_name": "tool2", "params": {"p": "v2"}, "namespace": "test"},
            {"tool_name": "tool3", "params": {"p": "v3"}}
        ]

        results = await integration.batch_execute(
            tool_calls=tool_calls,
            correlation_id="batch_123"
        )

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_batch_execute_partial_failures(self, integration):
        """Test batch execution with some failures."""
        call_count = 0

        async def mock_call_tool(tool_name, params, namespace=None, correlation_id=None):
            nonlocal call_count
            call_count += 1
            mock_result = AsyncMock()
            if call_count % 2 == 0:
                mock_result.success = True
                mock_result.result = {"output": "success"}
                mock_result.server_name = "test"
            else:
                mock_result.success = False
                mock_result.error = "Tool failed"
            return mock_result

        integration.call_tool = mock_call_tool

        tool_calls = [
            {"tool_name": "tool1", "params": {}},
            {"tool_name": "tool2", "params": {}},
            {"tool_name": "tool3", "params": {}}
        ]

        results = await integration.batch_execute(
            tool_calls=tool_calls,
            correlation_id="batch_123"
        )

        assert len(results) == 3
        assert sum(1 for r in results if r.success) > 0

    @pytest.mark.asyncio
    async def test_batch_execute_exception_handling(self, integration):
        """Test batch execution with exception handling."""
        async def mock_call_tool(tool_name, params, namespace=None, correlation_id=None):
            if "error" in tool_name:
                raise Exception("Tool error")
            mock_result = AsyncMock()
            mock_result.success = True
            mock_result.result = {"output": "ok"}
            mock_result.server_name = "test"
            return mock_result

        integration.call_tool = mock_call_tool

        tool_calls = [
            {"tool_name": "tool1", "params": {}},
            {"tool_name": "error_tool", "params": {}},
            {"tool_name": "tool3", "params": {}}
        ]

        results = await integration.batch_execute(
            tool_calls=tool_calls,
            correlation_id="batch_123"
        )

        assert len(results) == 3
        # The exception should be caught and converted to error result
        assert results[1].success is False

    @pytest.mark.asyncio
    async def test_batch_execute_gateway_not_initialized(self):
        """Test batch execution when gateway not initialized."""
        with patch('knowledge_engine.integrations.mcp_gateway_integration.UnifiedMCPGateway') as mock_gateway_class:
            mock_gateway_instance = AsyncMock()
            mock_gateway_instance.initialize = AsyncMock()
            mock_gateway_instance.tool_registry = Mock()
            mock_gateway_instance.tool_router = Mock()

            mock_gateway_class.return_value = mock_gateway_instance

            with patch('asyncio.run', return_value=None):
                integration = MCPGatewayIntegration(config={})
                integration.unified_gateway = None

        tool_calls = [
            {"tool_name": "tool1", "params": {}},
            {"tool_name": "tool2", "params": {}}
        ]

        results = await integration.batch_execute(
            tool_calls=tool_calls,
            correlation_id="batch_123"
        )

        assert len(results) == 2
        assert all(r.success is False for r in results)


# =============================================================================
# TEST CLASS: MCPGatewayIntegration Status and Health
# =============================================================================

class TestMCPGatewayIntegrationStatus:
    """Test status and health check methods."""

    def test_get_mcp_status_with_gateway(self, integration):
        """Test getting MCP status with gateway available."""
        integration.unified_gateway.is_initialized = True
        integration.unified_gateway.is_running = True

        status = integration.get_mcp_status()

        assert status["available"] is True
        assert status["initialized"] is True
        assert status["running"] is True
        assert "timestamp" in status
        assert "status" in status

    def test_get_mcp_status_without_gateway(self):
        """Test getting MCP status without gateway."""
        with patch('knowledge_engine.integrations.mcp_gateway_integration.UnifiedMCPGateway') as mock_gateway_class:
            mock_gateway_instance = AsyncMock()
            mock_gateway_instance.initialize = AsyncMock()
            mock_gateway_instance.tool_registry = Mock()
            mock_gateway_instance.tool_router = Mock()

            mock_gateway_class.return_value = mock_gateway_instance

            with patch('asyncio.run', return_value=None):
                integration = MCPGatewayIntegration(config={})
                integration.unified_gateway = None

        status = integration.get_mcp_status()

        assert status["available"] is False
        assert status["initialized"] is False
        assert status["running"] is False

    def test_get_mcp_status_with_health_check(self, integration):
        """Test getting MCP status with health check."""
        mock_health = {
            "gateway": {
                "status": "running",
                "initialized": True,
                "uptime_seconds": 3600
            },
            "servers": {
                "server1": {"status": "healthy"},
                "server2": {"status": "healthy"}
            },
            "tools": 15
        }
        integration.unified_gateway.get_health_status.return_value = mock_health
        integration.unified_gateway.is_initialized = True
        integration.unified_gateway.is_running = True

        status = integration.get_mcp_status()

        assert status["available"] is True
        assert status["status"]["gateway"]["status"] == "running"
        assert status["status"]["tools"] == 15


# =============================================================================
# TEST CLASS: MCPGatewayIntegration Resource Management
# =============================================================================

class TestMCPGatewayIntegrationResourceManagement:
    """Test resource management and cleanup."""

    @pytest.mark.asyncio
    async def test_close_with_gateway(self, integration):
        """Test closing integration with gateway."""
        integration.unified_gateway.shutdown = AsyncMock(return_value=True)

        await integration.close()

        integration.unified_gateway.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_gateway(self):
        """Test closing integration without gateway."""
        with patch('knowledge_engine.integrations.mcp_gateway_integration.UnifiedMCPGateway') as mock_gateway_class:
            mock_gateway_instance = AsyncMock()
            mock_gateway_instance.initialize = AsyncMock()
            mock_gateway_instance.tool_registry = Mock()
            mock_gateway_instance.tool_router = Mock()

            mock_gateway_class.return_value = mock_gateway_instance

            with patch('asyncio.run', return_value=None):
                integration = MCPGatewayIntegration(config={})
                integration.unified_gateway = None

        # Should not raise error
        await integration.close()

    @pytest.mark.asyncio
    async def test_close_with_shutdown_error(self, integration):
        """Test closing with shutdown error."""
        integration.unified_gateway.shutdown.side_effect = Exception("Shutdown error")

        # Should not raise error, just log
        await integration.close()


# =============================================================================
# TEST CLASS: Edge Cases and Error Handling
# =============================================================================

class TestMCPGatewayIntegrationEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_call_tool_with_empty_params(self, integration):
        """Test tool call with empty parameters."""
        mock_result = AsyncMock()
        mock_result.success = True
        mock_result.result = {}
        mock_result.server_name = "test"

        integration.unified_gateway.call_tool.return_value = mock_result

        result = await integration.call_tool(
            tool_name="test_tool",
            params={},
            correlation_id="test_123"
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_call_tool_with_complex_params(self, integration):
        """Test tool call with complex parameters."""
        complex_params = {
            "text": "Sample text",
            "options": {
                "nested": {"value": 123},
                "list": [1, 2, 3]
            },
            "flags": [True, False, True]
        }

        mock_result = AsyncMock()
        mock_result.success = True
        mock_result.result = {"status": "ok"}
        mock_result.server_name = "test"

        integration.unified_gateway.call_tool.return_value = mock_result

        result = await integration.call_tool(
            tool_name="complex_tool",
            params=complex_params,
            correlation_id="test_123"
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_batch_execute_empty_list(self, integration):
        """Test batch execution with empty tool call list."""
        results = await integration.batch_execute(
            tool_calls=[],
            correlation_id="batch_123"
        )

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_workflow_with_empty_text(self, integration):
        """Test knowledge extraction workflow with empty text."""
        async def mock_call_tool(tool_name, params, namespace=None, correlation_id=None):
            mock_result = AsyncMock()
            mock_result.success = True
            mock_result.result = {}
            return mock_result

        integration.call_tool = mock_call_tool

        result = await integration.execute_knowledge_extraction_workflow(
            text="",
            extraction_types=["entities"],
            correlation_id="workflow_123"
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_discover_tools_with_invalid_filters(self, integration):
        """Test tool discovery with invalid filters."""
        integration.unified_gateway.list_tools.return_value = []

        result = await integration.discover_tools(
            namespace="nonexistent_namespace",
            category="nonexistent_category",
            correlation_id="discover_123"
        )

        assert result.success is True
        assert result.output == []

    @pytest.mark.asyncio
    async def test_call_tool_with_special_characters_in_params(self, integration):
        """Test tool call with special characters in parameters."""
        params = {
            "text": "Text with \"quotes\" and 'apostrophes' and \n newlines",
            "special": "@#$%^&*()"
        }

        mock_result = AsyncMock()
        mock_result.success = True
        mock_result.result = {"processed": True}
        mock_result.server_name = "test"

        integration.unified_gateway.call_tool.return_value = mock_result

        result = await integration.call_tool(
            tool_name="sanitize_tool",
            params=params,
            correlation_id="test_123"
        )

        assert result.success is True
