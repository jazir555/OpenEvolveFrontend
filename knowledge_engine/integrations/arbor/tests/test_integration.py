"""
Integration tests for Arbor integration

These tests require a running Arbor server.
Set RUN_ARBOR_INTEGRATION_TESTS=1 to enable.

Following CLAUDE.md principles:
- END-TO-END: Test full integration flow
- REAL DEPENDENCIES: Use actual Arbor server
"""

import os
import pytest
import asyncio

# Skip all integration tests unless explicitly enabled
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_ARBOR_INTEGRATION_TESTS") != "1",
    reason="Set RUN_ARBOR_INTEGRATION_TESTS=1 to run integration tests"
)

# Import after skip check to avoid import errors
from knowledge_engine.integrations.arbor import (
    ArborClient,
    ArborConfig,
    ArborConnectionConfig,
    ArborGraphAdapter,
    ArborSchemaMapper,
    ArborMCPBridge
)


@pytest.fixture(scope="module")
def arbor_config():
    """Create configuration for integration tests."""
    ws_url = os.getenv("ARBOR_WS_URL", "ws://localhost:7433")
    return ArborConfig(
        connection=ArborConnectionConfig(
            ws_url=ws_url,
            connection_timeout=10.0,
            request_timeout=30.0
        ),
        debug=True
    )


@pytest.fixture(scope="module")
async def connected_client(arbor_config):
    """Create and connect Arbor client."""
    client = ArborClient(arbor_config)
    
    try:
        connected = await client.connect()
        assert connected, "Failed to connect to Arbor server"
        yield client
    finally:
        await client.disconnect()


class TestArborServerConnection:
    """Test basic server connectivity."""
    
    @pytest.mark.asyncio
    async def test_connect_to_server(self, arbor_config):
        """Test connecting to Arbor server."""
        client = ArborClient(arbor_config)
        
        try:
            result = await client.connect()
            assert result is True
            assert client.is_connected
        finally:
            await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_stats(self, connected_client):
        """Test getting server statistics."""
        stats = await connected_client.get_stats()
        
        assert isinstance(stats, dict)
        # Arbor should return basic stats
        assert "node_count" in stats or "files" in stats
    
    @pytest.mark.asyncio
    async def test_disconnect_gracefully(self, arbor_config):
        """Test graceful disconnection."""
        client = ArborClient(arbor_config)
        
        await client.connect()
        assert client.is_connected
        
        await client.disconnect()
        assert not client.is_connected


class TestGraphQueries:
    """Test graph query operations."""
    
    @pytest.mark.asyncio
    async def test_query_graph_basic(self, connected_client):
        """Test basic graph query."""
        # Query for all functions (should work on any indexed codebase)
        result = await connected_client.query_graph("FIND function LIMIT 5")
        
        assert result is not None
        assert isinstance(result.nodes, list)
    
    @pytest.mark.asyncio
    async def test_find_node(self, connected_client):
        """Test finding a specific node."""
        # First get some node name from the graph
        all_nodes = await connected_client.query_graph("FIND * LIMIT 1")
        
        if all_nodes.nodes:
            node_name = all_nodes.nodes[0]["name"]
            found = await connected_client.find_node(node_name)
            
            if found:
                assert found["name"] == node_name
    
    @pytest.mark.asyncio
    async def test_find_node_not_found(self, connected_client):
        """Test finding non-existent node."""
        result = await connected_client.find_node("__nonexistent_node_xyz__")
        assert result is None


class TestSchemaMapping:
    """Test schema mapping with real data."""
    
    @pytest.mark.asyncio
    async def test_convert_real_nodes(self, connected_client):
        """Test converting real nodes from server."""
        mapper = ArborSchemaMapper(storage_prefix="test")
        
        # Get real nodes
        result = await connected_client.query_graph("FIND function LIMIT 3")
        
        for arbor_node in result.nodes:
            entity = mapper.convert_arbor_node(arbor_node)
            
            assert entity.entity_id.startswith("test:")
            assert entity.name == arbor_node["name"]
            assert entity.entity_type.startswith("code_")
    
    @pytest.mark.asyncio
    async def test_convert_real_edges(self, connected_client):
        """Test converting real edges from server."""
        mapper = ArborSchemaMapper(storage_prefix="test")
        
        # Get edges (requires nodes with relationships)
        result = await connected_client.query_graph(
            "FIND edge WHERE kind = 'calls' LIMIT 3"
        )
        
        for arbor_edge in result.edges:
            relationship = mapper.convert_arbor_edge(arbor_edge)
            
            assert relationship.source_id.startswith("test:")
            assert relationship.target_id.startswith("test:")
            assert relationship.relationship_type.startswith("code_")


class TestMCPBridge:
    """Test MCP bridge with real server."""
    
    @pytest.mark.asyncio
    async def test_mcp_find_definition(self, connected_client):
        """Test MCP find_definition tool."""
        bridge = ArborMCPBridge(connected_client)
        
        # Get a real function name first
        result = await connected_client.query_graph("FIND function LIMIT 1")
        
        if result.nodes:
            func_name = result.nodes[0]["name"]
            
            tool_result = await bridge.execute_tool(
                "arbor_find_definition",
                {"symbol": func_name}
            )
            
            assert isinstance(tool_result.success, bool)
            if tool_result.success:
                assert "name" in tool_result.data
    
    @pytest.mark.asyncio
    async def test_mcp_search(self, connected_client):
        """Test MCP search tool."""
        bridge = ArborMCPBridge(connected_client)
        
        tool_result = await bridge.execute_tool(
            "arbor_search",
            {"query": "main", "max_results": 5}
        )
        
        assert tool_result.success is True
        assert "matches" in tool_result.data
        assert "total_count" in tool_result.data
    
    @pytest.mark.asyncio
    async def test_mcp_get_context(self, connected_client):
        """Test MCP get_context tool."""
        bridge = ArborMCPBridge(connected_client)
        
        # Get a node with context
        result = await connected_client.query_graph("FIND function LIMIT 1")
        
        if result.nodes:
            node_name = result.nodes[0]["name"]
            
            tool_result = await bridge.execute_tool(
                "arbor_get_context",
                {"symbol": node_name, "depth": 1}
            )
            
            assert isinstance(tool_result.success, bool)


class TestGraphAdapter:
    """Test graph adapter with real server."""
    
    @pytest.mark.asyncio
    async def test_export_and_merge(self, connected_client):
        """Test exporting graph and merging into mock KG."""
        from unittest.mock import Mock, AsyncMock
        
        # Export full graph
        graph_data = await connected_client.export_graph()
        
        assert "nodes" in graph_data or "version" in graph_data
        
        # Create mock knowledge graph
        kg = Mock()
        kg.add_entity_async = AsyncMock(return_value=True)
        kg.add_relationship_async = AsyncMock(return_value=True)
        kg.get_entity_async = AsyncMock(return_value=None)
        
        adapter = ArborGraphAdapter(kg, connected_client)
        
        # Merge (limited size for test)
        small_graph = {
            "nodes": graph_data.get("nodes", [])[:10],
            "edges": graph_data.get("edges", [])[:10]
        }
        
        result = await adapter.merge_arbor_graph(small_graph)
        
        assert isinstance(result.success, bool)


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, arbor_config):
        """Test complete workflow from connection to MCP query."""
        # 1. Connect
        client = ArborClient(arbor_config)
        await client.connect()
        
        try:
            # 2. Get stats
            stats = await client.get_stats()
            assert stats is not None
            
            # 3. Query graph
            result = await client.query_graph("FIND * LIMIT 5")
            assert result is not None
            
            # 4. Use MCP bridge
            bridge = ArborMCPBridge(client)
            
            # 5. Execute tool
            tool_result = await bridge.execute_tool(
                "arbor_search",
                {"query": "test", "max_results": 3}
            )
            
            assert tool_result is not None
            
        finally:
            await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_reconnection(self, arbor_config):
        """Test reconnection after disconnection."""
        client = ArborClient(arbor_config)
        
        # Connect
        await client.connect()
        assert client.is_connected
        
        # Disconnect
        await client.disconnect()
        assert not client.is_connected
        
        # Reconnect
        await client.connect()
        assert client.is_connected
        
        # Should still work
        stats = await client.get_stats()
        assert stats is not None
        
        await client.disconnect()
