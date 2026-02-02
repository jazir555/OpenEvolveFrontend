"""
Arbor Integration for OpenEvolve Knowledge Engine

This module provides integration with Arbor (https://github.com/Anandb71/arbor),
a Rust-based code graph intelligence layer that uses Tree-sitter to parse
codebases into navigable graphs.

The integration enables the Knowledge Engine to:
- Ingest code structure graphs from Arbor
- Query code relationships (calls, imports, inheritance)
- Provide AI agents with precise code navigation
- Support refactoring impact analysis

Usage:
    from knowledge_engine.integrations.arbor import ArborClient, ArborConfig
    
    config = ArborConfig.from_env()
    client = ArborClient(config.connection)
    await client.connect()
    
    # Query the code graph
    result = await client.query_graph("FIND function WHERE name = 'authenticate'")
    
    # Use MCP tools
    from knowledge_engine.integrations.arbor import ArborMCPBridge
    mcp = ArborMCPBridge(client)
    result = await mcp.execute_tool("arbor_find_definition", {"symbol": "MyClass"})

Configuration:
    Environment variables:
    - ARBOR_WS_URL: WebSocket URL (default: ws://localhost:7433)
    - ARBOR_ENABLED: Enable/disable integration (default: true)
    - ARBOR_DEBUG: Enable debug logging (default: false)

Phases:
    Phase 1 (Foundation): Client, config, health checks ✓
    Phase 2 (Graph Bridge): Adapter, import, sync ✓
    Phase 3 (Intelligence): Query, context, impact ✓
    Phase 4 (MCP): Tools for AI agents ✓
    Phase 5 (Viz): Visualizer integration (future)
"""

__version__ = "1.0.0"
__author__ = "OpenEvolve"

# Core components (Phase 1)
from .config import (
    ArborConfig,
    ArborConnectionConfig,
    ArborSyncConfig,
    ArborIndexingConfig,
    ArborMCPConfig
)
from .client import ArborClient
from .health import ArborHealthChecker, HealthStatus
from .exceptions import (
    ArborError,
    ArborConnectionError,
    ArborQueryError,
    ArborTimeoutError,
    ArborNotConnectedError,
    ArborSchemaError,
    ArborSyncError,
    ArborMCPError,
)

# Graph adapter (Phase 2)
from .schema_mapping import (
    ArborSchemaMapper,
    convert_arbor_node,
    convert_arbor_edge,
    ARBOR_KIND_TO_ENTITY_TYPE,
    ARBOR_EDGE_TO_RELATIONSHIP_TYPE
)
from .graph_adapter import (
    ArborGraphAdapter,
    MergeResult,
    GraphDelta
)

# MCP bridge (Phase 4)
from .mcp_bridge import (
    ArborMCPBridge,
    ToolResult
)

__all__ = [
    # Version
    "__version__",
    
    # Config
    "ArborConfig",
    "ArborConnectionConfig",
    "ArborSyncConfig",
    "ArborIndexingConfig",
    "ArborMCPConfig",
    
    # Client
    "ArborClient",
    
    # Graph Adapter (Phase 2)
    "ArborSchemaMapper",
    "ArborGraphAdapter",
    "MergeResult",
    "GraphDelta",
    "convert_arbor_node",
    "convert_arbor_edge",
    "ARBOR_KIND_TO_ENTITY_TYPE",
    "ARBOR_EDGE_TO_RELATIONSHIP_TYPE",
    
    # MCP Bridge (Phase 4)
    "ArborMCPBridge",
    "ToolResult",
    
    # Health
    "ArborHealthChecker",
    "HealthStatus",
    
    # Exceptions
    "ArborError",
    "ArborConnectionError",
    "ArborQueryError",
    "ArborTimeoutError",
    "ArborNotConnectedError",
    "ArborSchemaError",
    "ArborSyncError",
    "ArborMCPError",
]
