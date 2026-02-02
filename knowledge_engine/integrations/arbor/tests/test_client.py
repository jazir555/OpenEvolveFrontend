"""
Tests for Arbor Client

Following CLAUDE.md principles:
- CONTRACT TESTS: Verify API contracts
- ISOLATION: Mock external dependencies
"""

import asyncio
import json
import os
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Skip all tests if websockets not available
try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    pytest.skip("websockets not available", allow_module_level=True)

from knowledge_engine.integrations.arbor import (
    ArborClient,
    ArborConfig,
    ArborConnectionConfig,
    QueryResult,
    IndexingResult,
    CodePath,
    ImpactAnalysis
)
from knowledge_engine.integrations.arbor.exceptions import (
    ArborConnectionError,
    ArborNotConnectedError,
    ArborQueryError,
    ArborTimeoutError
)


class TestArborClient:
    """Test suite for ArborClient."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ArborConfig(
            connection=ArborConnectionConfig(
                ws_url="ws://localhost:7433",
                connection_timeout=5.0,
                request_timeout=5.0
            )
        )
    
    @pytest.fixture
    def client(self, config):
        """Create test client."""
        return ArborClient(config)
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.config.connection.ws_url == "ws://localhost:7433"
        assert not client.is_connected
        assert client._reconnect_count == 0
    
    @pytest.mark.asyncio
    async def test_connect_success(self, client):
        """Test successful connection."""
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[asyncio.CancelledError()])  # Stop receive loop
        
        with patch('websockets.connect', return_value=mock_ws):
            result = await client.connect()
            
        assert result is True
        assert client.is_connected
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, client):
        """Test connection failure."""
        with patch('websockets.connect', side_effect=ConnectionRefusedError()):
            with pytest.raises(ArborConnectionError) as exc_info:
                await client.connect()
        
        assert "ws://localhost:7433" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_disconnect(self, client):
        """Test disconnection."""
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[asyncio.CancelledError()])
        
        with patch('websockets.connect', return_value=mock_ws):
            await client.connect()
            assert client.is_connected
            
            await client.disconnect()
            assert not client.is_connected
    
    @pytest.mark.asyncio
    async def test_query_graph_success(self, client):
        """Test successful graph query."""
        mock_ws = AsyncMock()
        
        # Mock the receive to return a proper response
        response = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "nodes": [
                    {"id": "1", "name": "authenticate", "kind": "function"}
                ],
                "edges": [],
                "execution_time_ms": 10.5,
                "total_count": 1
            }
        }
        
        async def mock_recv():
            # Return response then cancel
            return json.dumps(response)
        
        mock_ws.recv = mock_recv
        mock_ws.send = AsyncMock()
        
        with patch('websockets.connect', return_value=mock_ws):
            await client.connect()
            
            # Override the request ID matching
            with patch.object(client, '_pending_requests', {}):
                # Manually simulate the response handling
                future = asyncio.get_event_loop().create_future()
                future.set_result(response)
                client._pending_requests["test-id"] = future
                
                result = await client.query_graph("FIND function WHERE name = 'authenticate'")
        
        assert isinstance(result, QueryResult)
        assert len(result.nodes) == 1
        assert result.nodes[0]["name"] == "authenticate"
    
    @pytest.mark.asyncio
    async def test_query_not_connected(self, client):
        """Test query when not connected."""
        with pytest.raises(ArborNotConnectedError):
            await client.query_graph("FIND function")
    
    @pytest.mark.asyncio
    async def test_find_node_success(self, client):
        """Test finding a node by name."""
        mock_ws = AsyncMock()
        
        response = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "nodes": [
                    {"id": "1", "name": "AuthController", "kind": "class"}
                ],
                "total_count": 1
            }
        }
        
        async def mock_recv():
            return json.dumps(response)
        
        mock_ws.recv = mock_recv
        mock_ws.send = AsyncMock()
        
        with patch('websockets.connect', return_value=mock_ws):
            await client.connect()
            
            with patch.object(client, '_send_request', return_value=response["result"]):
                result = await client.find_node("AuthController")
        
        assert result is not None
        assert result["name"] == "AuthController"
    
    @pytest.mark.asyncio
    async def test_find_path_success(self, client):
        """Test finding path between nodes."""
        path_result = {
            "found": True,
            "start": {"id": "1", "name": "login"},
            "end": {"id": "5", "name": "find_user"},
            "path": [
                {"id": "1", "name": "login"},
                {"id": "2", "name": "authenticate"},
                {"id": "5", "name": "find_user"}
            ],
            "distance": 2
        }
        
        with patch.object(client, '_send_request', return_value=path_result):
            result = await client.find_path("login", "find_user")
        
        assert isinstance(result, CodePath)
        assert result.distance == 2
        assert len(result.path) == 3
    
    @pytest.mark.asyncio
    async def test_find_path_not_found(self, client):
        """Test finding path when no path exists."""
        with patch.object(client, '_send_request', return_value={"found": False}):
            result = await client.find_path("a", "b")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_callers(self, client):
        """Test getting callers of a function."""
        callers_result = {
            "callers": [
                {"id": "1", "name": "login", "kind": "function"},
                {"id": "2", "name": "refresh_token", "kind": "function"}
            ]
        }
        
        with patch.object(client, '_send_request', return_value=callers_result):
            result = await client.get_callers("authenticate")
        
        assert len(result) == 2
        assert result[0]["name"] == "login"
    
    @pytest.mark.asyncio
    async def test_analyze_impact(self, client):
        """Test impact analysis."""
        impact_result = {
            "target": {"id": "1", "name": "validateUser"},
            "direct": [{"id": "2", "name": "login"}],
            "transitive": [{"id": "3", "name": "refresh"}],
            "total_affected": 2,
            "files": ["src/auth.ts", "src/login.ts"]
        }
        
        with patch.object(client, '_send_request', return_value=impact_result):
            result = await client.analyze_impact("validateUser", "rename")
        
        assert isinstance(result, ImpactAnalysis)
        assert result.change_type == "rename"
        assert result.total_affected == 2
        assert len(result.files_to_modify) == 2
    
    @pytest.mark.asyncio
    async def test_index_codebase(self, client):
        """Test codebase indexing."""
        index_result = {
            "success": True,
            "files_indexed": 150,
            "nodes_created": 1200,
            "edges_created": 3500,
            "errors": []
        }
        
        with patch.object(client, '_send_request', return_value=index_result):
            result = await client.index_codebase("/path/to/code")
        
        assert isinstance(result, IndexingResult)
        assert result.success
        assert result.files_indexed == 150
        assert result.nodes_created == 1200
    
    @pytest.mark.asyncio
    async def test_get_context(self, client):
        """Test getting context around a node."""
        context_result = {
            "nodes": [
                {"id": "1", "name": "AuthController"},
                {"id": "2", "name": "TokenMiddleware"},
                {"id": "3", "name": "UserService"}
            ],
            "edges": [
                {"from": "2", "to": "1", "kind": "calls"}
            ],
            "total_count": 3
        }
        
        with patch.object(client, '_send_request', return_value=context_result):
            result = await client.get_context("AuthController", depth=2)
        
        assert isinstance(result, QueryResult)
        assert len(result.nodes) == 3
        assert len(result.edges) == 1
    
    @pytest.mark.asyncio
    async def test_export_graph(self, client):
        """Test exporting full graph."""
        export_result = {
            "version": "1.0",
            "nodes": [{"id": "1"}],
            "edges": [{"from": "1", "to": "2"}]
        }
        
        with patch.object(client, '_send_request', return_value=export_result):
            result = await client.export_graph()
        
        assert result["version"] == "1.0"
        assert len(result["nodes"]) == 1


class TestArborConfig:
    """Test suite for ArborConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ArborConfig()
        
        assert config.enabled is True
        assert config.connection.ws_url == "ws://localhost:7433"
        assert config.connection.reconnect_interval == 5.0
        assert config.sync.mode == "realtime"
        assert "python" in config.indexing.languages
    
    def test_config_from_env(self, monkeypatch):
        """Test loading configuration from environment."""
        monkeypatch.setenv("ARBOR_WS_URL", "ws://arbor.example.com:7433")
        monkeypatch.setenv("ARBOR_ENABLED", "false")
        monkeypatch.setenv("ARBOR_SYNC_MODE", "batch")
        
        config = ArborConfig.from_env()
        
        assert config.connection.ws_url == "ws://arbor.example.com:7433"
        assert config.enabled is False
        assert config.sync.mode == "batch"
    
    def test_config_validation(self):
        """Test configuration validation."""
        from knowledge_engine.integrations.arbor.config import ArborConnectionConfig
        
        with pytest.raises(ValueError, match="ws_url must start with"):
            ArborConnectionConfig(ws_url="http://invalid")
        
        with pytest.raises(ValueError, match="reconnect_interval must be"):
            ArborConnectionConfig(reconnect_interval=-1)
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = ArborConfig()
        data = config.to_dict()
        
        assert data["enabled"] == config.enabled
        assert data["connection"]["ws_url"] == config.connection.ws_url
        assert "sync" in data
        assert "indexing" in data


@pytest.mark.asyncio
class TestArborClientIntegration:
    """
    Integration tests that require a real Arbor server.
    
    These are skipped by default. Set RUN_ARBOR_INTEGRATION_TESTS=1 to run.
    """
    
    @pytest.fixture(scope="class")
    def run_integration_tests(self):
        """Check if integration tests should run."""
        import os
        return os.getenv("RUN_ARBOR_INTEGRATION_TESTS") == "1"
    
    @pytest.mark.skipif(
        not os.getenv("RUN_ARBOR_INTEGRATION_TESTS"),
        reason="Set RUN_ARBOR_INTEGRATION_TESTS=1 to run"
    )
    async def test_real_connection(self):
        """Test connection to real Arbor server."""
        config = ArborConfig(
            connection=ArborConnectionConfig(ws_url="ws://localhost:7433")
        )
        client = ArborClient(config)
        
        try:
            result = await client.connect()
            assert result is True
            assert client.is_connected
            
            # Try a simple query
            stats = await client.get_stats()
            assert isinstance(stats, dict)
            
        finally:
            await client.disconnect()
