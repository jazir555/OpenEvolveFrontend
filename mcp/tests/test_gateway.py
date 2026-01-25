"""
Comprehensive Test Suite for Unified MCP Gateway.

Tests cover:
- Tool registration and discovery
- Tool routing and execution
- Circuit breaking
- Fallback and retry logic
- Analytics tracking
- Gateway initialization
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

import sys
from pathlib import Path

# Add parent directory to path
frontend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(frontend_path))

from mcp.gateway.models import (
    ToolDefinition,
    ServerConfig,
    ToolCategory,
    ServerStatus,
    ToolCallResult,
)
from mcp.gateway.tool_registry import ToolRegistry
from mcp.gateway.tool_router import ToolRouter, CircuitBreaker
from mcp.gateway.unified_mcp_gateway import UnifiedMCPGateway
from mcp.gateway.analytics import MCPGatewayAnalytics


# ============================================================================
# Tool Registry Tests
# ============================================================================

class TestToolRegistry:
    """Test suite for ToolRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        return ToolRegistry()

    @pytest.fixture
    def sample_tool(self):
        """Create a sample tool definition."""
        return ToolDefinition(
            name="test_tool",
            description="A test tool",
            namespace="test",
            server_name="test_server",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                }
            },
            category=ToolCategory.UTILITIES,
        )

    def test_register_tool(self, registry, sample_tool):
        """Test tool registration."""
        result = registry.register_tool(sample_tool)
        assert result is True

        tool = registry.get_tool("test_tool", "test")
        assert tool is not None
        assert tool.name == "test_tool"
        assert tool.namespace == "test"

    def test_get_tool_not_found(self, registry):
        """Test getting non-existent tool."""
        tool = registry.get_tool("nonexistent")
        assert tool is None

    def test_list_tools_by_namespace(self, registry, sample_tool):
        """Test listing tools by namespace."""
        registry.register_tool(sample_tool)

        tools = registry.list_tools(namespace="test")
        assert len(tools) == 1
        assert tools[0].name == "test_tool"

    def test_list_tools_by_category(self, registry, sample_tool):
        """Test listing tools by category."""
        registry.register_tool(sample_tool)

        tools = registry.list_tools(category=ToolCategory.UTILITIES)
        assert len(tools) >= 1

    def test_deprecate_tool(self, registry, sample_tool):
        """Test tool deprecation."""
        registry.register_tool(sample_tool)

        result = registry.deprecate_tool("test/test_tool", "test/new_tool")
        assert result is True

        tool = registry.get_tool("test_tool", "test")
        assert tool.deprecated is True

    def test_search_tools(self, registry, sample_tool):
        """Test tool search."""
        registry.register_tool(sample_tool)

        results = registry.search_tools("test")
        assert len(results) >= 1

        # Should find by name
        results = registry.search_tools("test_tool")
        assert len(results) >= 1

    def test_get_tool_count(self, registry, sample_tool):
        """Test tool count statistics."""
        registry.register_tool(sample_tool)

        stats = registry.get_tool_count()
        assert stats["total_tools"] >= 1
        assert stats["namespaces"] >= 1


# ============================================================================
# Circuit Breaker Tests
# ============================================================================

class TestCircuitBreaker:
    """Test suite for CircuitBreaker."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker with low threshold."""
        return CircuitBreaker(threshold=3, timeout=60)

    def test_initial_state(self, circuit_breaker):
        """Test initial circuit breaker state."""
        state = circuit_breaker.get_state("test_server")
        assert state.failure_count == 0
        assert state.is_open is False

    def test_record_failure(self, circuit_breaker):
        """Test recording failures."""
        circuit_breaker.record_failure("test_server")
        state = circuit_breaker.get_state("test_server")
        assert state.failure_count == 1

    def test_circuit_opens_after_threshold(self, circuit_breaker):
        """Test circuit opens after threshold failures."""
        # Record failures up to threshold
        for _ in range(3):
            circuit_breaker.record_failure("test_server")

        # Circuit should be open
        assert circuit_breaker.is_open("test_server") is True

    def test_circuit_resets(self, circuit_breaker):
        """Test circuit reset."""
        # Open the circuit
        for _ in range(3):
            circuit_breaker.record_failure("test_server")

        assert circuit_breaker.is_open("test_server") is True

        # Reset
        circuit_breaker.reset("test_server")

        assert circuit_breaker.is_open("test_server") is False
        state = circuit_breaker.get_state("test_server")
        assert state.failure_count == 0

    def test_record_success_resets_failure_count(self, circuit_breaker):
        """Test that success resets failure count."""
        # Record some failures
        circuit_breaker.record_failure("test_server")
        circuit_breaker.record_failure("test_server")

        # Record success
        circuit_breaker.record_success("test_server")

        state = circuit_breaker.get_state("test_server")
        assert state.failure_count == 0
        assert state.is_open is False


# ============================================================================
# Tool Router Tests
# ============================================================================

class TestToolRouter:
    """Test suite for ToolRouter."""

    @pytest.fixture
    def registry(self):
        """Create a registry with sample tools."""
        reg = ToolRegistry()
        tool = ToolDefinition(
            name="test_tool",
            description="Test tool",
            namespace="test",
            server_name="test_server",
            parameters={},
            category=ToolCategory.UTILITIES,
        )
        reg.register_tool(tool)
        return reg

    @pytest.fixture
    def router(self, registry):
        """Create a router with registry."""
        return ToolRouter(
            registry=registry,
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=60,
        )

    def test_register_server(self, router):
        """Test server registration."""
        router.register_server("test_server", "http://localhost:8001")

        assert "test_server" in router.server_urls
        assert router.server_urls["test_server"] == "http://localhost:8001"

    def test_route_tool(self, router):
        """Test tool routing."""
        router.register_server("test_server", "http://localhost:8001")

        destination = router.route("test_tool", "test")

        assert destination is not None
        assert destination.server_name == "test_server"
        assert destination.tool_name == "test_tool"

    def test_route_tool_not_found(self, router):
        """Test routing non-existent tool."""
        destination = router.route("nonexistent_tool")
        assert destination is None

    def test_execute_with_retry_success(self, router):
        """Test successful execution with retry."""
        router.register_server("test_server", "http://localhost:8001")

        async def mock_execute(server_url, tool_name, params):
            return {"result": "success"}

        destination = router.route("test_tool", "test")

        result = asyncio.run(router.execute_with_retry(
            destination,
            {},
            mock_execute
        ))

        assert result.success is True
        assert result.result == {"result": "success"}

    def test_get_healthy_servers(self, router):
        """Test getting healthy servers."""
        router.register_server("server1", "http://localhost:8001")
        router.register_server("server2", "http://localhost:8002")

        router.update_server_status("server1", ServerStatus.ONLINE)
        router.update_server_status("server2", ServerStatus.OFFLINE)

        healthy = router.get_healthy_servers()
        assert "server1" in healthy
        assert "server2" not in healthy


# ============================================================================
# Analytics Tests
# ============================================================================

class TestAnalytics:
    """Test suite for MCPGatewayAnalytics."""

    @pytest.fixture
    def analytics(self):
        """Create an analytics instance."""
        return MCPGatewayAnalytics(retention_days=30)

    @pytest.fixture
    def sample_result(self):
        """Create a sample tool call result."""
        return ToolCallResult(
            success=True,
            tool_name="test_tool",
            namespace="test",
            server_name="test_server",
            result={"output": "test"},
            execution_time=0.5,
        )

    @pytest.mark.asyncio
    async def test_track_tool_call(self, analytics, sample_result):
        """Test tracking tool calls."""
        await analytics.track_tool_call(sample_result)

        key = "test/test_tool"
        assert key in analytics.tool_metrics

        metrics = analytics.tool_metrics[key]
        assert metrics.total_calls == 1
        assert metrics.successful_calls == 1

    @pytest.mark.asyncio
    async def test_get_popular_tools(self, analytics):
        """Test getting popular tools."""
        # Create multiple results
        for i in range(10):
            result = ToolCallResult(
                success=True,
                tool_name="popular_tool",
                namespace="test",
                server_name="test_server",
                result={},
                execution_time=0.1,
            )
            await analytics.track_tool_call(result)

        popular = analytics.get_popular_tools(limit=5)
        assert len(popular) >= 1
        assert popular[0]["tool_name"] == "test/popular_tool"

    @pytest.mark.asyncio
    async def test_get_tool_success_rate(self, analytics):
        """Test getting tool success rate."""
        # Track successful call
        result1 = ToolCallResult(
            success=True,
            tool_name="test_tool",
            namespace="test",
            server_name="test_server",
            result={},
            execution_time=0.1,
        )
        await analytics.track_tool_call(result1)

        # Track failed call
        result2 = ToolCallResult(
            success=False,
            tool_name="test_tool",
            namespace="test",
            server_name="test_server",
            error="Test error",
            execution_time=0.1,
        )
        await analytics.track_tool_call(result2)

        success_rate = analytics.get_tool_success_rate("test_tool", "test")
        assert success_rate == 0.5

    def test_cleanup_old_data(self, analytics):
        """Test cleanup of old analytics data."""
        # Add old data
        old_time = datetime.utcnow() - timedelta(days=35)
        analytics.calls_over_time[old_time] = 100

        # Cleanup
        analytics.cleanup_old_data()

        # Old data should be removed
        assert old_time not in analytics.calls_over_time


# ============================================================================
# Gateway Tests
# ============================================================================

class TestUnifiedMCPGateway:
    """Test suite for UnifiedMCPGateway."""

    @pytest.fixture
    async def gateway(self):
        """Create and initialize a gateway."""
        gw = UnifiedMCPGateway()
        await gw.initialize()
        yield gw
        await gw.shutdown()

    @pytest.mark.asyncio
    async def test_initialization(self, gateway):
        """Test gateway initialization."""
        assert gateway.is_initialized is True
        assert gateway.tool_registry is not None
        assert gateway.tool_router is not None

    @pytest.mark.asyncio
    async def test_list_tools(self, gateway):
        """Test listing tools."""
        tools = await gateway.list_tools()
        assert isinstance(tools, list)

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self, gateway):
        """Test calling non-existent tool."""
        result = await gateway.call_tool("nonexistent_tool", {})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_get_health_status(self, gateway):
        """Test getting health status."""
        health = await gateway.get_health_status()
        assert "gateway" in health
        assert "servers" in health
        assert "tools" in health


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the gateway system."""

    @pytest.mark.asyncio
    async def test_end_to_end_tool_call(self):
        """Test complete tool call flow."""
        # Create gateway
        gateway = UnifiedMCPGateway()
        await gateway.initialize()

        # Mock tool execution
        async def mock_execute(server_url, tool_name, params):
            return {"mock_result": "success"}

        # Get a tool
        tools = await gateway.list_tools()
        if tools:
            tool_name = tools[0]["name"]

            # Call the tool (will fail if no servers, but tests the flow)
            result = await gateway.call_tool(tool_name, {"test": "value"})
            assert result is not None

        await gateway.shutdown()

    @pytest.mark.asyncio
    async def test_gateway_with_circuit_breaker(self):
        """Test gateway with circuit breaker behavior."""
        gateway = UnifiedMCPGateway()
        await gateway.initialize()

        # Simulate failures
        router = gateway.tool_router

        # Record failures
        for _ in range(10):
            router.circuit_breaker.record_failure("test_server")

        # Circuit should be open
        assert router.circuit_breaker.is_open("test_server") is True

        await gateway.shutdown()


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
