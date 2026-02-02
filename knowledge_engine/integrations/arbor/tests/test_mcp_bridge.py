"""
Tests for Arbor MCP Bridge

Following CLAUDE.md principles:
- CONTRACT TESTS: Verify tool contracts
- ISOLATION: Mock Arbor client
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch

from knowledge_engine.integrations.arbor import (
    ArborMCPBridge,
    ArborClient,
    ArborConfig,
    ArborMCPConfig,
    ToolResult
)
from knowledge_engine.integrations.arbor.mcp_bridge import ToolResult


class TestToolResult:
    """Test suite for ToolResult dataclass."""
    
    def test_basic_creation(self):
        """Test creating ToolResult."""
        result = ToolResult(
            success=True,
            data={"key": "value"},
            message="Operation completed"
        )
        
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.message == "Operation completed"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ToolResult(
            success=False,
            data={"error": "not found"},
            message="Failed"
        )
        
        data = result.to_dict()
        
        assert data["success"] is False
        assert data["data"] == {"error": "not found"}
        assert data["message"] == "Failed"


class TestArborMCPBridge:
    """Test suite for ArborMCPBridge."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock Arbor client."""
        client = Mock(spec=ArborClient)
        client.config = ArborConfig()
        return client
    
    @pytest.fixture
    def bridge(self, mock_client):
        """Create test bridge with all tools enabled."""
        config = ArborMCPConfig(
            enabled=True,
            tools=[
                "arbor_find_definition",
                "arbor_get_callers",
                "arbor_get_callees",
                "arbor_find_path",
                "arbor_analyze_impact",
                "arbor_get_context",
                "arbor_search"
            ],
            max_results=50,
            max_context_depth=3
        )
        return ArborMCPBridge(mock_client, config)
    
    def test_initialization(self, mock_client):
        """Test bridge initialization."""
        bridge = ArborMCPBridge(mock_client)
        
        assert bridge.client == mock_client
        assert bridge.config.enabled is True
        assert len(bridge._tools) == 7  # All default tools
    
    def test_initialization_partial_tools(self, mock_client):
        """Test initialization with partial tool list."""
        config = ArborMCPConfig(
            enabled=True,
            tools=["arbor_find_definition", "arbor_search"]
        )
        bridge = ArborMCPBridge(mock_client, config)
        
        assert len(bridge._tools) == 2
        assert "arbor_find_definition" in bridge._tools
        assert "arbor_search" in bridge._tools
        assert "arbor_get_callers" not in bridge._tools
    
    def test_get_available_tools(self, bridge):
        """Test getting tool definitions."""
        tools = bridge.get_available_tools()
        
        assert len(tools) == 7
        
        # Check tool structure
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert tool["parameters"]["type"] == "object"
    
    def test_get_available_tools_filtered(self, mock_client):
        """Test getting tools with filtering."""
        config = ArborMCPConfig(
            enabled=True,
            tools=["arbor_find_definition"]
        )
        bridge = ArborMCPBridge(mock_client, config)
        
        tools = bridge.get_available_tools()
        
        assert len(tools) == 1
        assert tools[0]["name"] == "arbor_find_definition"


class TestMCPToolExecution:
    """Test suite for MCP tool execution."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock Arbor client."""
        client = Mock(spec=ArborClient)
        client.config = ArborConfig()
        return client
    
    @pytest.fixture
    def bridge(self, mock_client):
        """Create test bridge."""
        config = ArborMCPConfig(
            enabled=True,
            tools=[
                "arbor_find_definition",
                "arbor_get_callers",
                "arbor_get_callees",
                "arbor_find_path",
                "arbor_analyze_impact",
                "arbor_get_context",
                "arbor_search"
            ],
            max_results=50,
            max_context_depth=3
        )
        return ArborMCPBridge(mock_client, config)
    
    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, bridge):
        """Test executing unknown tool."""
        result = await bridge.execute_tool("unknown_tool", {})
        
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "Unknown tool" in result.message
    
    @pytest.mark.asyncio
    async def test_execute_tool_exception(self, bridge, mock_client):
        """Test tool execution with exception."""
        mock_client.find_node = AsyncMock(side_effect=Exception("DB error"))
        
        result = await bridge.execute_tool(
            "arbor_find_definition",
            {"symbol": "test"}
        )
        
        assert result.success is False
        assert "DB error" in result.message


class TestToolFindDefinition:
    """Test suite for arbor_find_definition tool."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock Arbor client."""
        client = Mock(spec=ArborClient)
        client.config = ArborConfig()
        return client
    
    @pytest.fixture
    def bridge(self, mock_client):
        """Create test bridge."""
        return ArborMCPBridge(mock_client, ArborMCPConfig())
    
    @pytest.mark.asyncio
    async def test_find_definition_success(self, bridge, mock_client):
        """Test successful definition lookup."""
        mock_client.find_node = AsyncMock(return_value={
            "id": "func_1",
            "name": "authenticate",
            "kind": "function",
            "file": "/src/auth.py",
            "lineStart": 10,
            "lineEnd": 25,
            "signature": "def authenticate(user, password)",
            "docstring": "Authenticate a user"
        })
        
        result = await bridge.tool_find_definition(
            symbol="authenticate",
            file="/src/auth.py"
        )
        
        assert result.success is True
        assert result.data["name"] == "authenticate"
        assert result.data["kind"] == "function"
        assert result.data["file"] == "/src/auth.py"
        assert result.data["location"]["line_start"] == 10
        assert result.data["signature"] == "def authenticate(user, password)"
        assert "Found function 'authenticate'" in result.message
    
    @pytest.mark.asyncio
    async def test_find_definition_not_found(self, bridge, mock_client):
        """Test definition not found."""
        mock_client.find_node = AsyncMock(return_value=None)
        
        result = await bridge.tool_find_definition(symbol="nonexistent")
        
        assert result.success is False
        assert "not found" in result.message
    
    @pytest.mark.asyncio
    async def test_find_definition_with_kind_filter(self, bridge, mock_client):
        """Test definition lookup with kind filter."""
        mock_client.find_node = AsyncMock(return_value={
            "id": "class_1",
            "name": "User",
            "kind": "class"
        })
        
        result = await bridge.tool_find_definition(
            symbol="User",
            kind="class"
        )
        
        mock_client.find_node.assert_called_with("User", kind="class")
        assert result.success is True


class TestToolGetCallers:
    """Test suite for arbor_get_callers tool."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock Arbor client."""
        client = Mock(spec=ArborClient)
        client.config = ArborConfig()
        return client
    
    @pytest.fixture
    def bridge(self, mock_client):
        """Create test bridge."""
        config = ArborMCPConfig(max_results=10)
        return ArborMCPBridge(mock_client, config)
    
    @pytest.mark.asyncio
    async def test_get_callers_success(self, bridge, mock_client):
        """Test successful callers lookup."""
        mock_client.get_callers = AsyncMock(return_value=[
            {"id": "1", "name": "login", "kind": "function", "file": "/src/auth.py", "lineStart": 10},
            {"id": "2", "name": "refresh", "kind": "method", "file": "/src/auth.py", "lineStart": 20}
        ])
        
        result = await bridge.tool_get_callers(function_name="authenticate")
        
        assert result.success is True
        assert result.data["function"] == "authenticate"
        assert result.data["total_count"] == 2
        assert len(result.data["callers"]) == 2
        assert result.data["callers"][0]["name"] == "login"
    
    @pytest.mark.asyncio
    async def test_get_callers_empty(self, bridge, mock_client):
        """Test callers lookup with no results."""
        mock_client.get_callers = AsyncMock(return_value=[])
        
        result = await bridge.tool_get_callers(function_name="unused_func")
        
        assert result.success is True
        assert result.data["total_count"] == 0
        assert result.data["callers"] == []


class TestToolGetCallees:
    """Test suite for arbor_get_callees tool."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock Arbor client."""
        client = Mock(spec=ArborClient)
        client.config = ArborConfig()
        return client
    
    @pytest.fixture
    def bridge(self, mock_client):
        """Create test bridge."""
        config = ArborMCPConfig(max_results=10)
        return ArborMCPBridge(mock_client, config)
    
    @pytest.mark.asyncio
    async def test_get_callees_success(self, bridge, mock_client):
        """Test successful callees lookup."""
        mock_client.get_callees = AsyncMock(return_value=[
            {"id": "1", "name": "validate", "kind": "function", "file": "/src/utils.py"},
            {"id": "2", "name": "hash_password", "kind": "function", "file": "/src/crypto.py"}
        ])
        
        result = await bridge.tool_get_callees(function_name="authenticate")
        
        assert result.success is True
        assert result.data["function"] == "authenticate"
        assert result.data["total_count"] == 2
        assert len(result.data["callees"]) == 2


class TestToolFindPath:
    """Test suite for arbor_find_path tool."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock Arbor client."""
        client = Mock(spec=ArborClient)
        client.config = ArborConfig()
        return client
    
    @pytest.fixture
    def bridge(self, mock_client):
        """Create test bridge."""
        return ArborMCPBridge(mock_client, ArborMCPConfig())
    
    @pytest.mark.asyncio
    async def test_find_path_success(self, bridge, mock_client):
        """Test successful path finding."""
        from knowledge_engine.integrations.arbor import CodePath
        
        mock_client.find_path = AsyncMock(return_value=CodePath(
            start_node={"id": "1", "name": "login"},
            end_node={"id": "3", "name": "find_user"},
            path=[
                {"id": "1", "name": "login", "kind": "function"},
                {"id": "2", "name": "authenticate", "kind": "function"},
                {"id": "3", "name": "find_user", "kind": "function"}
            ],
            distance=2
        ))
        
        result = await bridge.tool_find_path(start="login", end="find_user")
        
        assert result.success is True
        assert result.data["start"] == "login"
        assert result.data["end"] == "find_user"
        assert result.data["distance"] == 2
        assert len(result.data["path"]) == 3
        assert "login -> authenticate -> find_user" in result.message
    
    @pytest.mark.asyncio
    async def test_find_path_not_found(self, bridge, mock_client):
        """Test path not found."""
        mock_client.find_path = AsyncMock(return_value=None)
        
        result = await bridge.tool_find_path(start="a", end="b")
        
        assert result.success is False
        assert "No path found" in result.message


class TestToolAnalyzeImpact:
    """Test suite for arbor_analyze_impact tool."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock Arbor client."""
        client = Mock(spec=ArborClient)
        client.config = ArborConfig()
        return client
    
    @pytest.fixture
    def bridge(self, mock_client):
        """Create test bridge."""
        return ArborMCPBridge(mock_client, ArborMCPConfig())
    
    @pytest.mark.asyncio
    async def test_analyze_impact_success(self, bridge, mock_client):
        """Test successful impact analysis."""
        from knowledge_engine.integrations.arbor import ImpactAnalysis
        
        mock_client.analyze_impact = AsyncMock(return_value=ImpactAnalysis(
            target_node={"id": "1", "name": "UserService"},
            change_type="modify",
            direct_impacts=[
                {"id": "2", "name": "AuthController", "kind": "class"},
                {"id": "3", "name": "UserController", "kind": "class"}
            ],
            transitive_impacts=[
                {"id": "4", "name": "API", "kind": "module"}
            ],
            total_affected=3,
            files_to_modify=["/src/auth.py", "/src/user.py"]
        ))
        
        result = await bridge.tool_analyze_impact(
            symbol="UserService",
            change_type="modify"
        )
        
        assert result.success is True
        assert result.data["target"] == "UserService"
        assert result.data["change_type"] == "modify"
        assert result.data["total_affected"] == 3
        assert len(result.data["direct_impacts"]) == 2
        assert len(result.data["files_to_modify"]) == 2
        assert "affects 3 components" in result.message


class TestToolGetContext:
    """Test suite for arbor_get_context tool."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock Arbor client."""
        client = Mock(spec=ArborClient)
        client.config = ArborConfig()
        return client
    
    @pytest.fixture
    def bridge(self, mock_client):
        """Create test bridge."""
        config = ArborMCPConfig(max_results=10, max_context_depth=3)
        return ArborMCPBridge(mock_client, config)
    
    @pytest.mark.asyncio
    async def test_get_context_success(self, bridge, mock_client):
        """Test successful context retrieval."""
        from knowledge_engine.integrations.arbor import QueryResult
        
        mock_client.find_node = AsyncMock(return_value={
            "id": "1",
            "name": "AuthController",
            "kind": "class",
            "signature": "class AuthController",
            "docstring": "Handles authentication"
        })
        
        mock_client.get_context = AsyncMock(return_value=QueryResult(
            query="context",
            nodes=[
                {"id": "1", "name": "AuthController", "kind": "class"},
                {"id": "2", "name": "TokenMiddleware", "kind": "class"},
                {"id": "3", "name": "UserService", "kind": "class"}
            ],
            edges=[],
            total_count=3
        ))
        
        result = await bridge.tool_get_context(symbol="AuthController", depth=2)
        
        assert result.success is True
        assert result.data["symbol"] == "AuthController"
        assert result.data["kind"] == "class"
        assert result.data["signature"] == "class AuthController"
        assert result.data["total_related"] == 2  # 3 nodes - 1 central
        assert len(result.data["related_components"]) == 2
    
    @pytest.mark.asyncio
    async def test_get_context_not_found(self, bridge, mock_client):
        """Test context for non-existent symbol."""
        mock_client.find_node = AsyncMock(return_value=None)
        
        result = await bridge.tool_get_context(symbol="nonexistent")
        
        assert result.success is False
        assert "not found" in result.message
    
    @pytest.mark.asyncio
    async def test_get_context_depth_clamping(self, bridge, mock_client):
        """Test depth parameter clamping."""
        mock_client.find_node = AsyncMock(return_value={"id": "1", "name": "test", "kind": "function"})
        mock_client.get_context = AsyncMock(return_value={"nodes": [], "edges": []})
        
        # Test depth > max
        await bridge.tool_get_context(symbol="test", depth=10)
        
        # Should be clamped to max_context_depth (3)
        mock_client.get_context.assert_called_with("1", depth=3, include_edges=True)


class TestToolSearch:
    """Test suite for arbor_search tool."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock Arbor client."""
        client = Mock(spec=ArborClient)
        client.config = ArborConfig()
        return client
    
    @pytest.fixture
    def bridge(self, mock_client):
        """Create test bridge."""
        config = ArborMCPConfig(max_results=10)
        return ArborMCPBridge(mock_client, config)
    
    @pytest.mark.asyncio
    async def test_search_success(self, bridge, mock_client):
        """Test successful search."""
        from knowledge_engine.integrations.arbor import QueryResult
        
        mock_client.query_graph = AsyncMock(return_value=QueryResult(
            query='FIND * WHERE name CONTAINS "auth"',
            nodes=[
                {"id": "1", "name": "authenticate", "kind": "function", "file": "/src/auth.py", "lineStart": 10},
                {"id": "2", "name": "AuthController", "kind": "class", "file": "/src/auth.py", "lineStart": 20}
            ],
            edges=[],
            total_count=2
        ))
        
        result = await bridge.tool_search(query="auth", max_results=5)
        
        assert result.success is True
        assert result.data["query"] == "auth"
        assert result.data["total_count"] == 2
        assert len(result.data["matches"]) == 2
        assert result.data["matches"][0]["name"] == "authenticate"
    
    @pytest.mark.asyncio
    async def test_search_with_kind_filter(self, bridge, mock_client):
        """Test search with kind filter."""
        from knowledge_engine.integrations.arbor import QueryResult
        
        mock_client.query_graph = AsyncMock(return_value=QueryResult(
            query='FIND * WHERE name CONTAINS "Auth" AND kind = "class"',
            nodes=[
                {"id": "1", "name": "AuthController", "kind": "class"}
            ],
            edges=[],
            total_count=1
        ))
        
        result = await bridge.tool_search(query="Auth", kind="class")
        
        assert result.success is True
        # Verify the query includes kind filter
        call_args = mock_client.query_graph.call_args
        assert 'kind = "class"' in call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_search_empty_results(self, bridge, mock_client):
        """Test search with no results."""
        from knowledge_engine.integrations.arbor import QueryResult
        
        mock_client.query_graph = AsyncMock(return_value=QueryResult(
            query='FIND * WHERE name CONTAINS "xyz"',
            nodes=[],
            edges=[],
            total_count=0
        ))
        
        result = await bridge.tool_search(query="xyz")
        
        assert result.success is True
        assert result.data["total_count"] == 0
        assert result.data["matches"] == []
