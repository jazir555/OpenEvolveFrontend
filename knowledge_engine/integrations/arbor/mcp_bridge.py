"""
MCP Bridge: Expose Arbor Capabilities to AI Agents

Provides Model Context Protocol (MCP) tools for AI agents to:
- Navigate code graphs
- Analyze impact of changes
- Find paths between components
- Get contextual code information

Following CLAUDE.md principles:
- ZERO TRUST: Validate all inputs
- STRUCTURED LOGGING: Track tool usage
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from .client import ArborClient
from .config import ArborMCPConfig
from .exceptions import ArborError, ArborQueryError

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from MCP tool execution."""
    
    success: bool
    data: Dict[str, Any]
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MCP response."""
        return {
            "success": self.success,
            "data": self.data,
            "message": self.message
        }


class ArborMCPBridge:
    """
    Bridge to expose Arbor capabilities through Model Context Protocol.
    
    This class registers tools that AI agents can use to:
    - Find code definitions
    - Trace call graphs
    - Analyze refactoring impact
    - Get contextual code information
    
    Example:
        bridge = ArborMCPBridge(arbor_client)
        
        # Agent asks: "Where is the authenticate function?"
        result = await bridge.tool_find_definition(
            symbol="authenticate",
            file="src/auth.py"
        )
    """
    
    def __init__(
        self,
        arbor_client: ArborClient,
        config: Optional[ArborMCPConfig] = None
    ):
        """
        Initialize MCP bridge.
        
        Args:
            arbor_client: Connected Arbor client
            config: MCP configuration
        """
        self.client = arbor_client
        self.config = config or ArborMCPConfig()
        self._tools: Dict[str, Callable] = {}
        
        # Register tools
        self._register_tools()
        
        logger.info({
            "msg": "ArborMCPBridge initialized",
            "enabled_tools": len(self.config.tools)
        })
    
    def _register_tools(self) -> None:
        """Register available MCP tools."""
        tool_mapping = {
            "arbor_find_definition": self.tool_find_definition,
            "arbor_get_callers": self.tool_get_callers,
            "arbor_get_callees": self.tool_get_callees,
            "arbor_find_path": self.tool_find_path,
            "arbor_analyze_impact": self.tool_analyze_impact,
            "arbor_get_context": self.tool_get_context,
            "arbor_search": self.tool_search,
        }
        
        # Only register enabled tools
        for tool_name in self.config.tools:
            if tool_name in tool_mapping:
                self._tools[tool_name] = tool_mapping[tool_name]
            else:
                logger.warning(f"Unknown tool requested: {tool_name}")
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available tools for MCP registration.
        
        Returns:
            List of tool definitions
        """
        tool_definitions = {
            "arbor_find_definition": {
                "name": "arbor_find_definition",
                "description": "Find the definition of a symbol (function, class, variable) in the codebase",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Name of the symbol to find"
                        },
                        "file": {
                            "type": "string",
                            "description": "Optional file path for context (helps disambiguate)"
                        },
                        "kind": {
                            "type": "string",
                            "description": "Optional kind filter (function, class, etc.)"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            "arbor_get_callers": {
                "name": "arbor_get_callers",
                "description": "Get all functions that call the specified function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "description": "Name of the function"
                        }
                    },
                    "required": ["function_name"]
                }
            },
            "arbor_get_callees": {
                "name": "arbor_get_callees",
                "description": "Get all functions called by the specified function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "description": "Name of the function"
                        }
                    },
                    "required": ["function_name"]
                }
            },
            "arbor_find_path": {
                "name": "arbor_find_path",
                "description": "Find the logic flow (call path) between two code components",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start": {
                            "type": "string",
                            "description": "Starting symbol (e.g., 'AuthController.login')"
                        },
                        "end": {
                            "type": "string",
                            "description": "Ending symbol (e.g., 'UserRepository.find')"
                        }
                    },
                    "required": ["start", "end"]
                }
            },
            "arbor_analyze_impact": {
                "name": "arbor_analyze_impact",
                "description": "Analyze what code would be affected by changing a symbol (rename, modify, delete)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Symbol to analyze"
                        },
                        "change_type": {
                            "type": "string",
                            "enum": ["rename", "modify", "delete"],
                            "description": "Type of change being considered"
                        }
                    },
                    "required": ["symbol", "change_type"]
                }
            },
            "arbor_get_context": {
                "name": "arbor_get_context",
                "description": "Get relevant code context around a symbol for understanding",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Symbol to get context for"
                        },
                        "depth": {
                            "type": "integer",
                            "default": 2,
                            "description": "How many relationship hops to include (1-3)"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            "arbor_search": {
                "name": "arbor_search",
                "description": "Search for code by name, signature, or docstring content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "kind": {
                            "type": "string",
                            "description": "Optional kind filter (function, class, etc.)"
                        },
                        "max_results": {
                            "type": "integer",
                            "default": 10,
                            "description": "Maximum number of results"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
        
        return [
            tool_definitions[name]
            for name in self._tools.keys()
            if name in tool_definitions
        ]
    
    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        """
        Execute an MCP tool.
        
        Args:
            tool_name: Name of the tool to execute
            params: Tool parameters
            
        Returns:
            ToolResult with execution results
        """
        if tool_name not in self._tools:
            return ToolResult(
                success=False,
                data={},
                message=f"Unknown tool: {tool_name}"
            )
        
        try:
            tool_func = self._tools[tool_name]
            result = await tool_func(**params)
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"Tool execution failed: {str(e)}"
            )
    
    # ========================================================================
    # MCP Tool Implementations
    # ========================================================================
    
    async def tool_find_definition(
        self,
        symbol: str,
        file: Optional[str] = None,
        kind: Optional[str] = None
    ) -> ToolResult:
        """
        MCP Tool: Find the definition of a symbol.
        
        Args:
            symbol: Symbol name to find
            file: Optional file context for disambiguation
            kind: Optional kind filter
        """
        try:
            logger.info({
                "msg": "MCP tool: find_definition",
                "symbol": symbol,
                "file": file
            })
            
            # Search for the symbol
            node = await self.client.find_node(symbol, kind=kind)
            
            if not node:
                return ToolResult(
                    success=False,
                    data={},
                    message=f"Symbol '{symbol}' not found in codebase"
                )
            
            # Format result
            result = {
                "name": node["name"],
                "kind": node["kind"],
                "file": node.get("file"),
                "location": {
                    "line_start": node.get("lineStart"),
                    "line_end": node.get("lineEnd")
                },
                "signature": node.get("signature"),
                "docstring": node.get("docstring")
            }
            
            return ToolResult(
                success=True,
                data=result,
                message=f"Found {node['kind']} '{symbol}' at {node.get('file', 'unknown')}:{node.get('lineStart', '?')}"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data={},
                message=f"Failed to find definition: {str(e)}"
            )
    
    async def tool_get_callers(self, function_name: str) -> ToolResult:
        """
        MCP Tool: Get all functions that call the specified function.
        
        Args:
            function_name: Name of the function
        """
        try:
            logger.info({
                "msg": "MCP tool: get_callers",
                "function": function_name
            })
            
            callers = await self.client.get_callers(function_name)
            
            # Format results
            formatted_callers = [
                {
                    "name": c["name"],
                    "kind": c["kind"],
                    "file": c.get("file"),
                    "line": c.get("lineStart")
                }
                for c in callers[:self.config.max_results]
            ]
            
            return ToolResult(
                success=True,
                data={
                    "function": function_name,
                    "callers": formatted_callers,
                    "total_count": len(callers)
                },
                message=f"Found {len(callers)} callers of '{function_name}'"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data={},
                message=f"Failed to get callers: {str(e)}"
            )
    
    async def tool_get_callees(self, function_name: str) -> ToolResult:
        """
        MCP Tool: Get all functions called by the specified function.
        
        Args:
            function_name: Name of the function
        """
        try:
            logger.info({
                "msg": "MCP tool: get_callees",
                "function": function_name
            })
            
            callees = await self.client.get_callees(function_name)
            
            formatted_callees = [
                {
                    "name": c["name"],
                    "kind": c["kind"],
                    "file": c.get("file")
                }
                for c in callees[:self.config.max_results]
            ]
            
            return ToolResult(
                success=True,
                data={
                    "function": function_name,
                    "callees": formatted_callees,
                    "total_count": len(callees)
                },
                message=f"Found {len(callees)} functions called by '{function_name}'"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data={},
                message=f"Failed to get callees: {str(e)}"
            )
    
    async def tool_find_path(self, start: str, end: str) -> ToolResult:
        """
        MCP Tool: Find the logic flow between two components.
        
        Args:
            start: Starting symbol
            end: Ending symbol
        """
        try:
            logger.info({
                "msg": "MCP tool: find_path",
                "start": start,
                "end": end
            })
            
            path = await self.client.find_path(start, end)
            
            if not path:
                return ToolResult(
                    success=False,
                    data={},
                    message=f"No path found from '{start}' to '{end}'"
                )
            
            # Format path
            path_steps = [
                {
                    "name": node["name"],
                    "kind": node["kind"]
                }
                for node in path.path
            ]
            
            return ToolResult(
                success=True,
                data={
                    "start": start,
                    "end": end,
                    "distance": path.distance,
                    "path": path_steps
                },
                message=f"Found path: {' -> '.join(s['name'] for s in path_steps)}"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data={},
                message=f"Failed to find path: {str(e)}"
            )
    
    async def tool_analyze_impact(
        self,
        symbol: str,
        change_type: str
    ) -> ToolResult:
        """
        MCP Tool: Analyze impact of changing a symbol.
        
        Args:
            symbol: Symbol to analyze
            change_type: Type of change (rename, modify, delete)
        """
        try:
            logger.info({
                "msg": "MCP tool: analyze_impact",
                "symbol": symbol,
                "change_type": change_type
            })
            
            impact = await self.client.analyze_impact(symbol, change_type)
            
            # Format impact report
            data = {
                "target": impact.target_node.get("name"),
                "change_type": change_type,
                "total_affected": impact.total_affected,
                "direct_impacts": [
                    {"name": n["name"], "kind": n["kind"]}
                    for n in impact.direct_impacts[:10]
                ],
                "transitive_impacts_count": len(impact.transitive_impacts),
                "files_to_modify": impact.files_to_modify[:10]
            }
            
            message = (
                f"Changing '{symbol}' affects {impact.total_affected} components "
                f"in {len(impact.files_to_modify)} files"
            )
            
            return ToolResult(
                success=True,
                data=data,
                message=message
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data={},
                message=f"Failed to analyze impact: {str(e)}"
            )
    
    async def tool_get_context(
        self,
        symbol: str,
        depth: int = 2
    ) -> ToolResult:
        """
        MCP Tool: Get contextual code information.
        
        Args:
            symbol: Symbol to get context for
            depth: Relationship depth (1-3)
        """
        try:
            # Clamp depth to valid range
            depth = max(1, min(depth, self.config.max_context_depth))
            
            logger.info({
                "msg": "MCP tool: get_context",
                "symbol": symbol,
                "depth": depth
            })
            
            # Find the node first
            node = await self.client.find_node(symbol)
            if not node:
                return ToolResult(
                    success=False,
                    data={},
                    message=f"Symbol '{symbol}' not found"
                )
            
            # Get context
            context = await self.client.get_context(node["id"], depth=depth)
            
            # Format context
            related = [
                {
                    "name": n["name"],
                    "kind": n["kind"],
                    "relationship": "related"
                }
                for n in context.nodes[:self.config.max_results]
                if n["id"] != node["id"]
            ]
            
            return ToolResult(
                success=True,
                data={
                    "symbol": symbol,
                    "kind": node["kind"],
                    "signature": node.get("signature"),
                    "docstring": node.get("docstring"),
                    "related_components": related,
                    "total_related": len(context.nodes) - 1
                },
                message=f"Found context for '{symbol}' with {len(related)} related components"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data={},
                message=f"Failed to get context: {str(e)}"
            )
    
    async def tool_search(
        self,
        query: str,
        kind: Optional[str] = None,
        max_results: int = 10
    ) -> ToolResult:
        """
        MCP Tool: Search for code.
        
        Args:
            query: Search query
            kind: Optional kind filter
            max_results: Maximum results to return
        """
        try:
            logger.info({
                "msg": "MCP tool: search",
                "query": query,
                "kind": kind
            })
            
            # Build ArborQL query
            arbor_query = f'FIND * WHERE name CONTAINS "{query}"'
            if kind:
                arbor_query += f' AND kind = "{kind}"'
            
            # Execute query
            result = await self.client.query_graph(arbor_query)
            
            # Format results
            matches = [
                {
                    "name": n["name"],
                    "kind": n["kind"],
                    "file": n.get("file"),
                    "line": n.get("lineStart"),
                    "signature": n.get("signature")
                }
                for n in result.nodes[:max_results]
            ]
            
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "matches": matches,
                    "total_count": result.total_count
                },
                message=f"Found {result.total_count} matches for '{query}'"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data={},
                message=f"Search failed: {str(e)}"
            )
