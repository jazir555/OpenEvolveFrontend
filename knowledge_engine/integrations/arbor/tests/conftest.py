"""
Shared fixtures for Arbor integration tests

Following CLAUDE.md principles:
- REUSABILITY: Shared fixtures across tests
- ISOLATION: Fresh instances per test
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

# Check if websockets is available
try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_arbor_node():
    """Sample Arbor node for testing."""
    return {
        "id": "func_001",
        "name": "authenticate",
        "kind": "function",
        "file": "/src/auth.py",
        "lineStart": 10,
        "lineEnd": 25,
        "signature": "def authenticate(user: str, password: str) -> bool",
        "visibility": "public",
        "qualifiedName": "auth.authenticate",
        "docstring": "Authenticate a user with username and password.",
        "centrality": 0.85
    }


@pytest.fixture
def sample_arbor_class_node():
    """Sample Arbor class node for testing."""
    return {
        "id": "class_001",
        "name": "AuthController",
        "kind": "class",
        "file": "/src/controllers.py",
        "lineStart": 15,
        "lineEnd": 100,
        "signature": "class AuthController(BaseController)",
        "visibility": "public",
        "qualifiedName": "controllers.AuthController"
    }


@pytest.fixture
def sample_arbor_edge():
    """Sample Arbor edge for testing."""
    return {
        "from": "func_001",
        "to": "func_002",
        "kind": "calls",
        "location": {
            "line": 20,
            "column": 12
        }
    }


@pytest.fixture
def sample_arbor_graph():
    """Sample complete Arbor graph for testing."""
    return {
        "version": "1.0",
        "projectRoot": "/home/user/myproject",
        "timestamp": "2024-01-15T10:30:00Z",
        "stats": {
            "node_count": 3,
            "edge_count": 2,
            "file_count": 1
        },
        "nodes": [
            {
                "id": "func_001",
                "name": "main",
                "kind": "function",
                "file": "/src/main.py",
                "lineStart": 1,
                "lineEnd": 10
            },
            {
                "id": "func_002",
                "name": "helper",
                "kind": "function",
                "file": "/src/main.py",
                "lineStart": 12,
                "lineEnd": 20
            },
            {
                "id": "class_001",
                "name": "MyClass",
                "kind": "class",
                "file": "/src/main.py",
                "lineStart": 22,
                "lineEnd": 50
            }
        ],
        "edges": [
            {
                "from": "func_001",
                "to": "func_002",
                "kind": "calls"
            },
            {
                "from": "func_002",
                "to": "class_001",
                "kind": "uses_type"
            }
        ]
    }


@pytest.fixture
def mock_knowledge_graph():
    """Create mock Knowledge Engine graph."""
    kg = Mock()
    kg.add_entity_async = AsyncMock(return_value=True)
    kg.add_relationship_async = AsyncMock(return_value=True)
    kg.get_entity_async = AsyncMock(return_value=None)
    kg._entities = {}
    return kg


@pytest.fixture
def mock_arbor_client():
    """Create mock Arbor client."""
    from knowledge_engine.integrations.arbor import ArborConfig
    
    client = Mock()
    client.config = ArborConfig()
    client.is_connected = True
    client._reconnect_count = 0
    
    # Mock async methods
    client.connect = AsyncMock(return_value=True)
    client.disconnect = AsyncMock()
    client.query_graph = AsyncMock()
    client.find_node = AsyncMock()
    client.find_path = AsyncMock()
    client.get_callers = AsyncMock(return_value=[])
    client.get_callees = AsyncMock(return_value=[])
    client.analyze_impact = AsyncMock()
    client.get_context = AsyncMock()
    client.get_stats = AsyncMock(return_value={
        "node_count": 100,
        "edge_count": 250,
        "file_count": 20
    })
    client.export_graph = AsyncMock()
    client.index_codebase = AsyncMock()
    
    return client


@pytest.fixture
def arbor_config():
    """Create test Arbor configuration."""
    from knowledge_engine.integrations.arbor import (
        ArborConfig,
        ArborConnectionConfig,
        ArborSyncConfig
    )
    
    return ArborConfig(
        enabled=True,
        connection=ArborConnectionConfig(
            ws_url="ws://localhost:7433",
            connection_timeout=5.0,
            request_timeout=5.0,
            reconnect_interval=1.0,
            max_reconnects=3
        ),
        sync=ArborSyncConfig(
            mode="manual",
            batch_size=100
        ),
        debug=True
    )


@pytest.fixture
def schema_mapper():
    """Create test schema mapper."""
    from knowledge_engine.integrations.arbor import ArborSchemaMapper
    return ArborSchemaMapper(storage_prefix="arbor")


# Skip marker for tests requiring websockets
requires_websockets = pytest.mark.skipif(
    not HAS_WEBSOCKETS,
    reason="websockets package not available"
)

# Skip marker for integration tests
integration_test = pytest.mark.skipif(
    True,  # Set to False to run integration tests
    reason="Integration tests disabled by default. Set to False to enable."
)
