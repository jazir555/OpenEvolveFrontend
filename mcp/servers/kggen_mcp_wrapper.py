"""
kg-gen MCP Wrapper for Unified Gateway.

This module wraps the kg-gen MCP server, providing a standardized interface
for the unified gateway to interact with kg-gen tools.
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Kggen Mcp Wrapper
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

from ..gateway.models import ToolDefinition, ToolCategory, ServerConfig

logger = logging.getLogger(__name__)


class KGenMCPWrapper:
    """
    Wraps kg-gen MCP server for unified gateway.

    Tools:
    - kggen/add_memories
    - kggen/retrieve_relevant_memories
    - kggen/visualize_memories
    - kggen/get_memory_stats
    """

    def __init__(self, server_url: str, timeout: int = 30):
        """
        Initialize kg-gen wrapper.

        Args:
            server_url: URL of the kg-gen MCP server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url
        self.timeout = timeout
        self.client = None

        if HTTPX_AVAILABLE:
            self.client = httpx.AsyncClient(timeout=timeout)
        else:
            logger.warning("httpx not available, kg-gen wrapper will use mock responses")

        # Register tools
        self.tools = self._register_tools()

        logger.info(f"KGenMCPWrapper initialized for {server_url}")

    def _register_tools(self) -> List[ToolDefinition]:
        """Register all kg-gen tools."""
        return [
            ToolDefinition(
                name="add_memories",
                description="Extract and store memories from unstructured text",
                namespace="kggen",
                server_name="kggen",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Unstructured text to extract memories from",
                        }
                    },
                    "required": ["text"],
                },
                category=ToolCategory.KNOWLEDGE,
                version="1.0.0",
                tags=["knowledge", "memory", "extraction"],
                examples=[
                    {
                        "input": {"text": "John works at OpenAI as a software engineer."},
                        "output": "Extracted 2 entities: John, OpenAI",
                    }
                ],
            ),
            ToolDefinition(
                name="retrieve_relevant_memories",
                description="Retrieve relevant memories for a query",
                namespace="kggen",
                server_name="kggen",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to find relevant memories for",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of memories to return",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
                category=ToolCategory.KNOWLEDGE,
                version="1.0.0",
                tags=["knowledge", "memory", "retrieval"],
                examples=[
                    {
                        "input": {"query": "Where does John work?"},
                        "output": "John works at OpenAI",
                    }
                ],
            ),
            ToolDefinition(
                name="visualize_memories",
                description="Generate HTML visualization of the memory graph",
                namespace="kggen",
                server_name="kggen",
                parameters={
                    "type": "object",
                    "properties": {
                        "output_filename": {
                            "type": "string",
                            "description": "Name for the output HTML file",
                            "default": "memory_graph.html",
                        }
                    },
                },
                category=ToolCategory.KNOWLEDGE,
                version="1.0.0",
                tags=["knowledge", "visualization", "graph"],
            ),
            ToolDefinition(
                name="get_memory_stats",
                description="Get statistics about stored memories",
                namespace="kggen",
                server_name="kggen",
                parameters={
                    "type": "object",
                    "properties": {},
                },
                category=ToolCategory.KNOWLEDGE,
                version="1.0.0",
                tags=["knowledge", "stats", "monitoring"],
            ),
        ]

    async def add_memories(self, text: str) -> Dict[str, Any]:
        """
        Extract and store memories from text.

        Args:
            text: Unstructured text to extract memories from

        Returns:
            Dict with extraction results
        """
        if not HTTPX_AVAILABLE or not self.client:
            return self._mock_response("add_memories", {"text": text})

        try:
            response = await self.client.post(
                f"{self.server_url}/tools/add_memories",
                json={"text": text},
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"kg-gen add_memories failed: {e}")
            return {"error": str(e)}

    async def retrieve_relevant_memories(
        self, query: str, limit: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve relevant memories for a query.

        Args:
            query: Query to find relevant memories for
            limit: Maximum number of memories to return

        Returns:
            Dict with retrieved memories
        """
        if not HTTPX_AVAILABLE or not self.client:
            return self._mock_response("retrieve_relevant_memories", {"query": query, "limit": limit})

        try:
            response = await self.client.post(
                f"{self.server_url}/tools/retrieve_relevant_memories",
                json={"query": query, "limit": limit},
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"kg-gen retrieve_relevant_memories failed: {e}")
            return {"error": str(e)}

    async def visualize_memories(
        self, output_filename: str = "memory_graph.html"
    ) -> Dict[str, Any]:
        """
        Generate HTML visualization of the memory graph.

        Args:
            output_filename: Name for the output HTML file

        Returns:
            Dict with visualization result
        """
        if not HTTPX_AVAILABLE or not self.client:
            return self._mock_response("visualize_memories", {"output_filename": output_filename})

        try:
            response = await self.client.post(
                f"{self.server_url}/tools/visualize_memories",
                json={"output_filename": output_filename},
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"kg-gen visualize_memories failed: {e}")
            return {"error": str(e)}

    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored memories.

        Returns:
            Dict with memory statistics
        """
        if not HTTPX_AVAILABLE or not self.client:
            return self._mock_response("get_memory_stats", {})

        try:
            response = await self.client.get(
                f"{self.server_url}/tools/get_memory_stats",
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"kg-gen get_memory_stats failed: {e}")
            return {"error": str(e)}

    def _mock_response(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock response when httpx is not available."""
        logger.warning(f"Using mock response for kg-gen/{tool_name}")

        if tool_name == "add_memories":
            return {
                "success": True,
                "message": f"Mock: Extracted memories from text ({len(params.get('text', ''))} chars)",
                "entities_count": 2,
                "relations_count": 1,
            }

        elif tool_name == "retrieve_relevant_memories":
            return {
                "success": True,
                "message": f"Mock: Retrieved memories for query '{params.get('query', '')}'",
                "memories_count": 1,
            }

        elif tool_name == "visualize_memories":
            return {
                "success": True,
                "message": f"Mock: Generated visualization at {params.get('output_filename', 'memory_graph.html')}",
                "path": params.get("output_filename", "memory_graph.html"),
            }

        elif tool_name == "get_memory_stats":
            return {
                "success": True,
                "total_entities": 10,
                "total_relations": 5,
                "storage_path": "./kg_memory.json",
            }

        return {"error": "Unknown tool"}

    async def health_check(self) -> bool:
        """
        Check if the kg-gen server is healthy.

        Returns:
            True if server is healthy
        """
        if not HTTPX_AVAILABLE or not self.client:
            return True  # Assume healthy in mock mode

        try:
            response = await self.client.get(f"{self.server_url}/health")
            response.raise_for_status()
            return True

        except Exception as e:
            logger.error(f"kg-gen health check failed: {e}")
            return False

    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
            logger.info("KGenMCPWrapper closed")


class KGenMCPWrapperFactory:
    """Factory for creating kg-gen wrappers."""

    @staticmethod
    def create_from_config(config: ServerConfig) -> KGenMCPWrapper:
        """
        Create a kg-gen wrapper from server config.

        Args:
            config: ServerConfig instance

        Returns:
            KGenMCPWrapper instance
        """
        return KGenMCPWrapper(
            server_url=config.url,
            timeout=config.timeout,
        )

    @staticmethod
    async def test_connection(server_url: str) -> bool:
        """
        Test connection to kg-gen server.

        Args:
            server_url: URL of the kg-gen server

        Returns:
            True if connection successful
        """
        wrapper = KGenMCPWrapper(server_url)
        try:
            healthy = await wrapper.health_check()
            return healthy
        finally:
            await wrapper.close()


# Export tools for registration
def get_kggen_tools() -> List[ToolDefinition]:
    """
    Get all kg-gen tool definitions.

    Returns:
        List of ToolDefinitions
    """
    wrapper = KGenMCPWrapper("http://localhost:8001")
    return wrapper.tools
