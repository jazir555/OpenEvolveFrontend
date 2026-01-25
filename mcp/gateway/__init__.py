"""
Unified MCP Gateway Package.

This package provides a centralized gateway for coordinating tools from
multiple MCP servers (kg-gen, Graphiti, OpenEvolve, etc.).
"""

from .unified_mcp_gateway import UnifiedMCPGateway
from .tool_registry import ToolRegistry
from .tool_router import ToolRouter, CircuitBreaker, LoadBalancer
from .analytics import MCPGatewayAnalytics
from .models import (
    GatewayConfig,
    ServerConfig,
    ToolDefinition,
    ToolCallResult,
    ServerStatus,
    ToolCategory,
)

__all__ = [
    "UnifiedMCPGateway",
    "ToolRegistry",
    "ToolRouter",
    "CircuitBreaker",
    "LoadBalancer",
    "MCPGatewayAnalytics",
    "GatewayConfig",
    "ServerConfig",
    "ToolDefinition",
    "ToolCallResult",
    "ServerStatus",
    "ToolCategory",
]

__version__ = "1.0.0"
