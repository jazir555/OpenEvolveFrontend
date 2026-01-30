"""
Unified MCP Gateway - Main Gateway Class.

This module provides the central gateway that coordinates tools from multiple
MCP servers (kg-gen, Graphiti, OpenEvolve, etc.) into a single namespace.
"""

import logging
import asyncio
import yaml
import os
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

from .models import (
    GatewayConfig,
    ServerConfig,
    ToolDefinition,
    ToolCallResult,
    ServerStatus,
    ToolCategory,
)
from .tool_registry import ToolRegistry
from .tool_router import ToolRouter

logger = logging.getLogger(__name__)


class UnifiedMCPGateway:
    """
    Unified MCP gateway that routes tool calls to multiple MCP servers.

    Features:
    - Tool registry from all servers
    - Automatic tool routing
    - Response aggregation
    - Error handling with fallback
    - Circuit breaking
    - Load balancing
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the unified gateway.

        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        self.tool_registry = ToolRegistry(self.config.to_dict()["tool_registry"])
        self.tool_router = ToolRouter(
            registry=self.tool_registry,
            circuit_breaker_threshold=self.config.circuit_breaker_threshold,
            circuit_breaker_timeout=self.config.circuit_breaker_timeout,
            load_balancing=self.config.load_balancing,
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
        )

        # Server management
        self.servers: Dict[str, ServerConfig] = {}
        self.server_clients: Dict[str, Any] = {}  # Server connection clients

        # Analytics
        self.tool_call_stats: Dict[str, Dict] = {}

        # Gateway status
        self.is_initialized = False
        self.is_running = False
        self._start_time: Optional[float] = None

        logger.info("UnifiedMCPGateway initialized")

    def _load_config(self, config_path: Optional[str]) -> GatewayConfig:
        """Load configuration from file or defaults."""
        default_config_path = Path(__file__).parent.parent / "config" / "gateway.yaml"

        if config_path is None and default_config_path.exists():
            config_path = str(default_config_path)

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f)

                # Extract gateway config
                gateway_cfg = config_data.get("gateway", {})
                registry_cfg = config_data.get("tool_registry", {})
                routing_cfg = config_data.get("routing", {})
                monitoring_cfg = config_data.get("monitoring", {})
                cache_cfg = config_data.get("cache", {})

                return GatewayConfig(
                    host=gateway_cfg.get("host", "0.0.0.0"),
                    port=gateway_cfg.get("port", 8080),
                    log_level=gateway_cfg.get("log_level", "INFO"),
                    max_workers=gateway_cfg.get("max_workers", 10),
                    request_timeout=gateway_cfg.get("request_timeout", 120),
                    enable_cors=gateway_cfg.get("enable_cors", True),
                    # Registry
                    categorization_enabled=registry_cfg.get("categorization_enabled", True),
                    versioning_enabled=registry_cfg.get("versioning_enabled", True),
                    deprecation_grace_period=registry_cfg.get("deprecation_grace_period", 30),
                    cache_ttl=registry_cfg.get("cache_ttl", 300),
                    # Routing
                    load_balancing=routing_cfg.get("load_balancing", "round_robin"),
                    circuit_breaker_threshold=routing_cfg.get("circuit_breaker_threshold", 5),
                    circuit_breaker_timeout=routing_cfg.get("circuit_breaker_timeout", 60),
                    fallback_enabled=routing_cfg.get("fallback_enabled", True),
                    max_retries=routing_cfg.get("max_retries", 3),
                    retry_delay=routing_cfg.get("retry_delay", 1),
                    # Monitoring
                    metrics_enabled=monitoring_cfg.get("metrics_enabled", True),
                    log_tool_calls=monitoring_cfg.get("log_tool_calls", True),
                    alert_on_failures=monitoring_cfg.get("alert_on_failures", True),
                    analytics_retention_days=monitoring_cfg.get("analytics_retention_days", 30),
                    performance_tracking=monitoring_cfg.get("performance_tracking", True),
                    # Cache
                    cache_enabled=cache_cfg.get("enabled", True),
                    cache_backend=cache_cfg.get("backend", "memory"),
                    cache_max_size=cache_cfg.get("max_size", 1000),
                )

            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}, using defaults")

        # Return default configuration
        return GatewayConfig(
            host="0.0.0.0",
            port=8080,
            log_level="INFO",
            max_workers=10,
            request_timeout=120,
            enable_cors=True,
            categorization_enabled=True,
            versioning_enabled=True,
            deprecation_grace_period=30,
            load_balancing="round_robin",
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=60,
            fallback_enabled=True,
            max_retries=3,
            retry_delay=1,
            metrics_enabled=True,
            log_tool_calls=True,
            alert_on_failures=True,
            analytics_retention_days=30,
            performance_tracking=True,
            cache_enabled=True,
            cache_backend="memory",
            cache_ttl=300,
            cache_max_size=1000,
        )

    async def initialize(self):
        """
        Initialize the gateway and connect to all servers.

        This method should be called before using the gateway.
        """
        if self.is_initialized:
            logger.warning("Gateway already initialized")
            return

        logger.info("Initializing Unified MCP Gateway...")

        # Load server configurations
        await self._load_servers()

        # Connect to servers
        for server_name, server_config in self.servers.items():
            if server_config.enabled:
                await self._connect_server(server_name)

        # Register tools from all servers
        await self._register_all_tools()

        self.is_initialized = True
        self._start_time = time.time()
        logger.info("Unified MCP Gateway initialized successfully")

    async def _load_servers(self):
        """Load server configurations."""
        # Default servers - in production, load from config
        default_servers = {
            "kggen": {
                "url": "http://localhost:8001",
                "timeout": 30,
                "namespace": "kggen",
                "description": "kg-gen knowledge graph memory",
            },
            "graphiti": {
                "url": "http://localhost:8002",
                "timeout": 30,
                "namespace": "graphiti",
                "description": "Graphiti knowledge graph",
            },
            "openevolve": {
                "url": "http://localhost:8003",
                "timeout": 30,
                "namespace": "openevolve",
                "description": "OpenEvolve evolution",
            },
        }

        for name, config in default_servers.items():
            server = ServerConfig(
                name=name,
                url=config["url"],
                timeout=config["timeout"],
                namespace=config["namespace"],
                description=config["description"],
                enabled=True,
                status=ServerStatus.OFFLINE,
            )
            self.servers[name] = server
            self.tool_router.register_server(name, server.url)

    async def _connect_server(self, server_name: str):
        """Connect to an MCP server."""
        server = self.servers.get(server_name)
        if not server:
            logger.error(f"Unknown server: {server_name}")
            return

        try:
            # In a real implementation, this would establish a connection
            # For now, just mark as online
            server.status = ServerStatus.ONLINE
            server.last_health_check = datetime.utcnow()
            logger.info(f"Connected to server: {server_name}")

        except Exception as e:
            logger.error(f"Failed to connect to {server_name}: {e}")
            server.status = ServerStatus.ERROR

    async def _register_all_tools(self):
        """Register tools from all connected servers."""
        logger.info("Registering tools from all servers...")

        for server_name, server in self.servers.items():
            if server.status == ServerStatus.ONLINE:
                await self._register_server_tools(server_name)

    async def _register_server_tools(self, server_name: str):
        """Register tools from a specific server."""
        server = self.servers.get(server_name)
        if not server:
            return

        try:
            # In a real implementation, this would query the server for its tools
            # For now, add placeholder tools
            if server_name == "kggen":
                tools = [
                    ToolDefinition(
                        name="add_memories",
                        description="Extract and store memories from text",
                        namespace=server.namespace,
                        server_name=server_name,
                        parameters={
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "Text to extract memories from"}
                            },
                            "required": ["text"],
                        },
                        category=ToolCategory.KNOWLEDGE,
                    ),
                    ToolDefinition(
                        name="retrieve_relevant_memories",
                        description="Retrieve relevant memories for a query",
                        namespace=server.namespace,
                        server_name=server_name,
                        parameters={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Query to find memories for"}
                            },
                            "required": ["query"],
                        },
                        category=ToolCategory.KNOWLEDGE,
                    ),
                ]
                self.tool_registry.register_tools_batch(tools)

        except Exception as e:
            logger.error(f"Failed to register tools from {server_name}: {e}")

    async def list_tools(
        self, namespace: str = "", category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all available tools.

        Args:
            namespace: Optional namespace filter
            category: Optional category filter

        Returns:
            List of tool definitions
        """
        tools = self.tool_registry.list_tools(
            namespace=namespace or None,
            category=ToolCategory(category) if category else None,
        )

        return [tool.to_dict() for tool in tools]

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> ToolCallResult:
        """
        Execute a tool call through the gateway.

        Args:
            tool_name: Full tool name (namespace/tool_name) or just tool_name
            params: Tool parameters

        Returns:
            ToolCallResult with execution result
        """
        if not self.is_initialized:
            return ToolCallResult(
                success=False,
                tool_name=tool_name,
                namespace="",
                server_name="",
                error="Gateway not initialized",
            )

        try:
            # Parse tool name
            namespace = None
            if "/" in tool_name:
                namespace, tool_name = tool_name.split("/", 1)

            # Route the call
            destination = self.tool_router.route(tool_name, namespace)

            if not destination:
                return ToolCallResult(
                    success=False,
                    tool_name=tool_name,
                    namespace=namespace or "",
                    server_name="",
                    error=f"Tool not found or no available servers: {tool_name}",
                )

            # Execute with retry and fallback
            result = await self.tool_router.execute_with_fallback(
                tool_name=tool_name,
                params=params,
                execute_func=self._execute_tool_call,
                namespace=namespace,
            )

            # Track analytics
            if self.config.metrics_enabled:
                await self._track_tool_call(result)

            return result

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return ToolCallResult(
                success=False,
                tool_name=tool_name,
                namespace=namespace or "",
                server_name="",
                error=str(e),
            )

    async def _execute_tool_call(
        self, server_url: str, tool_name: str, params: Dict[str, Any]
    ) -> Any:
        """
        Execute a tool call on a specific server.

        Args:
            server_url: URL of the server
            tool_name: Name of the tool
            params: Tool parameters

        Returns:
            Tool execution result
        """
        # In a real implementation, this would make an HTTP request to the server
        # For now, return a placeholder result
        await asyncio.sleep(0.1)  # Simulate network delay
        return {"result": f"Executed {tool_name} with params {params}"}

    async def _track_tool_call(self, result: ToolCallResult):
        """Track tool call analytics."""
        key = f"{result.namespace}/{result.tool_name}"

        if key not in self.tool_call_stats:
            self.tool_call_stats[key] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_execution_time": 0.0,
                "last_called": None,
            }

        stats = self.tool_call_stats[key]
        stats["total_calls"] += 1
        stats["total_execution_time"] += result.execution_time
        stats["last_called"] = result.timestamp

        if result.success:
            stats["successful_calls"] += 1
        else:
            stats["failed_calls"] += 1

    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the gateway and all servers.

        Returns:
            Health status information
        """
        # Calculate actual uptime
        uptime_seconds = 0
        if self._start_time is not None:
            uptime_seconds = int(time.time() - self._start_time)
        
        return {
            "gateway": {
                "status": "running" if self.is_running else "initialized",
                "initialized": self.is_initialized,
                "uptime_seconds": uptime_seconds,
            },
            "servers": {
                name: {
                    "status": server.status.value,
                    "url": server.url,
                    "enabled": server.enabled,
                    "last_health_check": server.last_health_check.isoformat() if server.last_health_check else None,
                }
                for name, server in self.servers.items()
            },
            "router": self.tool_router.get_router_stats(),
            "tools": self.tool_registry.get_tool_count(),
        }

    async def shutdown(self):
        """Shutdown the gateway and close all connections."""
        logger.info("Shutting down Unified MCP Gateway...")

        # Close server connections
        for server_name in list(self.server_clients.keys()):
            try:
                await self._disconnect_server(server_name)
            except Exception as e:
                logger.error(f"Error disconnecting {server_name}: {e}")

        self.is_running = False
        logger.info("Unified MCP Gateway shut down")

    async def _disconnect_server(self, server_name: str):
        """Disconnect from a server."""
        server = self.servers.get(server_name)
        if server:
            server.status = ServerStatus.OFFLINE
            if server_name in self.server_clients:
                del self.server_clients[server_name]
            logger.info(f"Disconnected from server: {server_name}")
