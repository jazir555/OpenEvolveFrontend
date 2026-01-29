"""
Tool Registry for the Unified MCP Gateway.

This module manages the registration, categorization, and discovery of tools
from all connected MCP servers.
"""

import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict

from .models import ToolDefinition, ToolCategory, ServerConfig

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for all MCP tools from all servers.

    Features:
    - Tool registration and discovery
    - Tool categorization
    - Tool versioning
    - Tool deprecation handling
    - Namespace management
    """

    def __init__(self, config=None):
        """
        Initialize the tool registry.

        Args:
            config: Optional configuration for the registry
        """
        self.tools: Dict[str, ToolDefinition] = {}  # Full tool name -> definition
        self.tools_by_namespace: Dict[str, Dict[str, ToolDefinition]] = defaultdict(dict)
        self.tools_by_category: Dict[ToolCategory, List[str]] = defaultdict(list)
        self.tools_by_server: Dict[str, Set[str]] = defaultdict(set)
        self.versions: Dict[str, str] = {}  # Tool name -> version
        self.deprecated_tools: Dict[str, str] = {}  # Tool name -> replacement tool

        # Configuration
        self.categorization_enabled = True
        self.versioning_enabled = True
        self.deprecation_grace_period = timedelta(days=30)

        if config:
            self.categorization_enabled = config.get("categorization_enabled", True)
            self.versioning_enabled = config.get("versioning_enabled", True)
            grace_period = config.get("deprecation_grace_period", 30)
            self.deprecation_grace_period = timedelta(days=grace_period)

        logger.info("ToolRegistry initialized")

    def register_tool(self, tool: ToolDefinition) -> bool:
        """
        Register a tool from a server.

        Args:
            tool: ToolDefinition to register

        Returns:
            True if registered successfully
        """
        try:
            # Create full tool name with namespace
            full_name = f"{tool.namespace}/{tool.name}"

            # Check if tool already exists
            if full_name in self.tools:
                existing = self.tools[full_name]
                if self.versioning_enabled:
                    # Compare versions
                    if tool.version > existing.version:
                        logger.info(f"Updating tool {full_name} from version {existing.version} to {tool.version}")
                    else:
                        logger.warning(f"Tool {full_name} version {tool.version} is older than existing {existing.version}")
                        return False
                else:
                    logger.warning(f"Tool {full_name} already registered, overwriting")

            # Register tool
            self.tools[full_name] = tool
            self.tools_by_namespace[tool.namespace][tool.name] = tool
            self.tools_by_server[tool.server_name].add(full_name)

            # Categorize
            if self.categorization_enabled:
                if tool.category not in self.tools_by_category[tool.category]:
                    self.tools_by_category[tool.category].append(full_name)
                elif full_name not in self.tools_by_category[tool.category]:
                    self.tools_by_category[tool.category].append(full_name)

            # Track version
            if self.versioning_enabled:
                self.versions[full_name] = tool.version

            # Track deprecation
            if tool.deprecated:
                self.deprecated_tools[full_name] = tool.deprecation_replacement or ""

            logger.debug(f"Registered tool: {full_name} (version {tool.version})")
            return True

        except Exception as e:
            logger.error(f"Failed to register tool {tool.name}: {e}")
            return False

    def register_tools_batch(self, tools: List[ToolDefinition]) -> int:
        """
        Register multiple tools at once.

        Args:
            tools: List of ToolDefinitions to register

        Returns:
            Number of tools successfully registered
        """
        registered = 0
        for tool in tools:
            if self.register_tool(tool):
                registered += 1
        logger.info(f"Registered {registered}/{len(tools)} tools in batch")
        return registered

    def get_tool(self, tool_name: str, namespace: Optional[str] = None) -> Optional[ToolDefinition]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool (with or without namespace)
            namespace: Optional namespace to search in

        Returns:
            ToolDefinition if found, None otherwise
        """
        # Try full name first
        if tool_name in self.tools:
            return self.tools[tool_name]

        # Try with namespace
        if namespace:
            full_name = f"{namespace}/{tool_name}"
            if full_name in self.tools:
                return self.tools[full_name]

        # Search all namespaces
        for ns, tools in self.tools_by_namespace.items():
            if tool_name in tools:
                return tools[tool_name]

        return None

    def list_tools(
        self,
        namespace: Optional[str] = None,
        category: Optional[ToolCategory] = None,
        server_name: Optional[str] = None,
        include_deprecated: bool = False,
    ) -> List[ToolDefinition]:
        """
        List tools with optional filtering.

        Args:
            namespace: Filter by namespace
            category: Filter by category
            server_name: Filter by server name
            include_deprecated: Include deprecated tools

        Returns:
            List of ToolDefinitions
        """
        tools = []

        if namespace:
            # Filter by namespace
            if namespace in self.tools_by_namespace:
                tools = list(self.tools_by_namespace[namespace].values())
        elif category:
            # Filter by category
            tool_names = self.tools_by_category.get(category, [])
            tools = [self.tools[name] for name in tool_names if name in self.tools]
        elif server_name:
            # Filter by server
            tool_names = self.tools_by_server.get(server_name, set())
            tools = [self.tools[name] for name in tool_names if name in self.tools]
        else:
            # All tools
            tools = list(self.tools.values())

        # Filter deprecated
        if not include_deprecated:
            tools = [t for t in tools if not t.deprecated]

        return tools

    def list_tools_by_category(self, category: ToolCategory) -> List[ToolDefinition]:
        """
        List all tools in a category.

        Args:
            category: ToolCategory to filter by

        Returns:
            List of ToolDefinitions in the category
        """
        tool_names = self.tools_by_category.get(category, [])
        return [self.tools[name] for name in tool_names if name in self.tools]

    def list_namespaces(self) -> List[str]:
        """
        List all registered namespaces.

        Returns:
            List of namespace names
        """
        return list(self.tools_by_namespace.keys())

    def list_servers(self) -> List[str]:
        """
        List all server names that have registered tools.

        Returns:
            List of server names
        """
        return list(self.tools_by_server.keys())

    def deprecate_tool(self, tool_name: str, replacement: str, grace_period_days: Optional[int] = None) -> bool:
        """
        Deprecate a tool.

        Args:
            tool_name: Full name of the tool to deprecate
            replacement: Replacement tool name
            grace_period_days: Optional custom grace period

        Returns:
            True if deprecated successfully
        """
        if tool_name not in self.tools:
            logger.warning(f"Cannot deprecate unknown tool: {tool_name}")
            return False

        tool = self.tools[tool_name]
        tool.deprecated = True
        tool.deprecation_replacement = replacement

        self.deprecated_tools[tool_name] = replacement

        logger.info(f"Deprecated tool: {tool_name} -> {replacement}")
        return True

    def get_deprecated_tools(self) -> Dict[str, str]:
        """
        Get all deprecated tools and their replacements.

        Returns:
            Dict mapping deprecated tool names to replacements
        """
        return self.deprecated_tools.copy()

    def search_tools(self, query: str, limit: int = 10) -> List[ToolDefinition]:
        """
        Search for tools by name or description.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of matching ToolDefinitions
        """
        query_lower = query.lower()
        results = []

        for tool in self.tools.values():
            # Skip deprecated
            if tool.deprecated:
                continue

            # Search in name and description
            if (query_lower in tool.name.lower() or
                query_lower in tool.description.lower() or
                any(query_lower in tag.lower() for tag in tool.tags)):
                results.append(tool)

        # Sort by relevance (exact name match first)
        results.sort(key=lambda t: 0 if query_lower == t.name.lower() else 1)

        return results[:limit]

    def get_tool_count(self) -> Dict[str, int]:
        """
        Get statistics about registered tools.

        Returns:
            Dict with tool counts
        """
        return {
            "total_tools": len(self.tools),
            "namespaces": len(self.tools_by_namespace),
            "categories": len(self.tools_by_category),
            "servers": len(self.tools_by_server),
            "deprecated": len(self.deprecated_tools),
        }

    def export_registry(self) -> Dict[str, any]:
        """
        Export the entire registry for backup/analysis.

        Returns:
            Dict representation of the registry
        """
        return {
            "tools": {name: tool.to_dict() for name, tool in self.tools.items()},
            "namespaces": list(self.tools_by_namespace.keys()),
            "categories": {cat.value: tools for cat, tools in self.tools_by_category.items()},
            "servers": {server: list(tools) for server, tools in self.tools_by_server.items()},
            "versions": self.versions,
            "deprecated": self.deprecated_tools,
            "stats": self.get_tool_count(),
        }

    def clear_namespace(self, namespace: str) -> int:
        """
        Remove all tools from a namespace.

        Args:
            namespace: Namespace to clear

        Returns:
            Number of tools removed
        """
        if namespace not in self.tools_by_namespace:
            return 0

        count = 0
        for tool_name, tool in list(self.tools_by_namespace[namespace].items()):
            full_name = f"{namespace}/{tool_name}"
            if full_name in self.tools:
                del self.tools[full_name]
                count += 1

        # Clear namespace
        del self.tools_by_namespace[namespace]

        logger.info(f"Cleared namespace {namespace} ({count} tools)")
        return count

    def unregister_server(self, server_name: str) -> int:
        """
        Remove all tools from a server.

        Args:
            server_name: Server to remove

        Returns:
            Number of tools removed
        """
        if server_name not in self.tools_by_server:
            return 0

        count = 0
        tool_names = list(self.tools_by_server[server_name])

        for tool_name in tool_names:
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                # Remove from namespace
                if tool.namespace in self.tools_by_namespace:
                    if tool.name in self.tools_by_namespace[tool.namespace]:
                        del self.tools_by_namespace[tool.namespace][tool.name]
                # Remove from tools
                del self.tools[tool_name]
                count += 1

        # Clear server
        del self.tools_by_server[server_name]

        logger.info(f"Unregistered server {server_name} ({count} tools)")
        return count
