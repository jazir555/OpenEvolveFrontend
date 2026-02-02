"""
Arbor WebSocket Client

Provides asynchronous communication with Arbor server via WebSocket.
Handles connection management, reconnection, and protocol messaging.

Following CLAUDE.md principles:
- ZERO TRUST: Validate all responses
- RUNTIME TRUTH: Track connection state explicitly
- IDEMPOTENCY: Safe to reconnect multiple times
- STRUCTURED LOGGING: JSON logs with correlation IDs
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable, Awaitable, Union

logger = logging.getLogger(__name__)

# Check if websockets is available
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets package not available. Install with: pip install websockets")

from .config import ArborConfig
from .exceptions import (
    ArborConnectionError,
    ArborNotConnectedError,
    ArborQueryError,
    ArborTimeoutError
)


@dataclass
class QueryResult:
    """Result from an Arbor graph query."""
    
    query: str
    """The query that was executed."""
    
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    """Nodes returned by the query."""
    
    edges: List[Dict[str, Any]] = field(default_factory=list)
    """Edges returned by the query."""
    
    execution_time_ms: float = 0.0
    """Query execution time."""
    
    total_count: int = 0
    """Total result count (for pagination)."""
    
    def __bool__(self) -> bool:
        """Return True if result has nodes or edges."""
        return bool(self.nodes or self.edges)


@dataclass
class IndexingResult:
    """Result from codebase indexing operation."""
    
    success: bool = False
    """Whether indexing completed successfully."""
    
    files_indexed: int = 0
    """Number of files processed."""
    
    nodes_created: int = 0
    """Number of graph nodes created."""
    
    edges_created: int = 0
    """Number of graph edges created."""
    
    errors: List[str] = field(default_factory=list)
    """Any errors during indexing."""
    
    duration_seconds: float = 0.0
    """Total indexing duration."""


@dataclass
class CodePath:
    """Represents a path through the code graph."""
    
    start_node: Dict[str, Any]
    """Starting node."""
    
    end_node: Dict[str, Any]
    """Ending node."""
    
    path: List[Dict[str, Any]] = field(default_factory=list)
    """Nodes in the path (including start and end)."""
    
    edges: List[Dict[str, Any]] = field(default_factory=list)
    """Edges traversed."""
    
    distance: int = 0
    """Number of hops."""


@dataclass
class ImpactAnalysis:
    """Result of impact analysis for a code change."""
    
    target_node: Dict[str, Any]
    """The node being changed."""
    
    change_type: str
    """Type of change (rename, modify, delete)."""
    
    direct_impacts: List[Dict[str, Any]] = field(default_factory=list)
    """Direct dependents (1 hop)."""
    
    transitive_impacts: List[Dict[str, Any]] = field(default_factory=list)
    """Transitive dependents (2+ hops)."""
    
    total_affected: int = 0
    """Total number of affected nodes."""
    
    files_to_modify: List[str] = field(default_factory=list)
    """List of files that would need changes."""


class ArborClient:
    """
    WebSocket client for Arbor code graph server.
    
    Provides methods to:
    - Connect/disconnect from Arbor server
    - Query the code graph
    - Find paths between code entities
    - Analyze impact of changes
    - Subscribe to real-time updates
    
    Example:
        config = ArborConfig(ws_url="ws://localhost:7433")
        client = ArborClient(config)
        
        await client.connect()
        
        # Find all functions that call 'authenticate'
        result = await client.get_callers("authenticate")
        
        # Find path from API to database
        path = await client.find_path("AuthController.login", "UserRepository.find")
        
        await client.disconnect()
    """
    
    def __init__(self, config: Optional[ArborConfig] = None):
        """
        Initialize Arbor client.
        
        Args:
            config: Arbor configuration. Uses defaults if not provided.
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError(
                "websockets package is required. "
                "Install with: pip install websockets"
            )
        
        self.config = config or ArborConfig()
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._connection_lock = asyncio.Lock()
        self._reconnect_count = 0
        self._message_handlers: Dict[str, Callable] = {}
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._correlation_id = str(uuid.uuid4())
        
        logger.info({
            "msg": "ArborClient initialized",
            "ws_url": self.config.connection.ws_url,
            "correlation_id": self._correlation_id
        })
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected to Arbor server."""
        return self._connected and self._ws is not None
    
    async def connect(self) -> bool:
        """
        Connect to Arbor server.
        
        Returns:
            True if connection successful
            
        Raises:
            ArborConnectionError: If connection fails
        """
        async with self._connection_lock:
            if self._connected:
                return True
            
            try:
                logger.info({
                    "msg": "Connecting to Arbor server",
                    "ws_url": self.config.connection.ws_url
                })
                
                self._ws = await websockets.connect(
                    self.config.connection.ws_url,
                    ping_interval=self.config.connection.heartbeat_interval,
                    ping_timeout=self.config.connection.connection_timeout
                )
                
                self._connected = True
                self._reconnect_count = 0
                
                # Start receive loop
                self._receive_task = asyncio.create_task(self._receive_loop())
                
                # Start heartbeat
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                
                logger.info({
                    "msg": "Connected to Arbor server",
                    "ws_url": self.config.connection.ws_url
                })
                
                return True
                
            except Exception as e:
                raise ArborConnectionError(
                    ws_url=self.config.connection.ws_url,
                    cause=e
                )
    
    async def disconnect(self) -> None:
        """Disconnect from Arbor server."""
        async with self._connection_lock:
            if not self._connected:
                return
            
            self._connected = False
            
            # Cancel tasks
            if self._receive_task:
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass
            
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # Close websocket
            if self._ws:
                await self._ws.close()
                self._ws = None
            
            # Clear pending requests
            for future in self._pending_requests.values():
                if not future.done():
                    future.cancel()
            self._pending_requests.clear()
            
            logger.info("Disconnected from Arbor server")
    
    async def _receive_loop(self) -> None:
        """Main receive loop for WebSocket messages."""
        try:
            while self._connected and self._ws:
                try:
                    message = await self._ws.recv()
                    await self._handle_message(message)
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Arbor WebSocket connection closed")
                    break
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Receive loop error: {e}")
        finally:
            if self._connected:
                # Connection closed unexpectedly
                self._connected = False
                await self._attempt_reconnect()
    
    async def _handle_message(self, message: Union[str, bytes]) -> None:
        """Handle incoming WebSocket message."""
        try:
            if isinstance(message, bytes):
                message = message.decode('utf-8')
            
            data = json.loads(message)
            
            # Check if this is a response to a pending request
            msg_id = data.get('id')
            if msg_id and msg_id in self._pending_requests:
                future = self._pending_requests.pop(msg_id)
                if not future.done():
                    future.set_result(data)
                return
            
            # Handle server-initiated messages (e.g., file change events)
            msg_type = data.get('type', 'unknown')
            handler = self._message_handlers.get(msg_type)
            if handler:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Message handler error: {e}")
            else:
                logger.debug(f"Unhandled message type: {msg_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Send a request to Arbor server and wait for response.
        
        Args:
            method: RPC method name
            params: Method parameters
            timeout: Request timeout (uses config default if not set)
            
        Returns:
            Response data
            
        Raises:
            ArborNotConnectedError: If not connected
            ArborTimeoutError: If request times out
            ArborQueryError: If server returns error
        """
        if not self.is_connected:
            raise ArborNotConnectedError(method)
        
        msg_id = str(uuid.uuid4())
        request = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
            "params": params or {}
        }
        
        # Create future for response
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[msg_id] = future
        
        try:
            # Send request
            await self._ws.send(json.dumps(request))
            
            # Wait for response
            timeout_val = timeout or self.config.connection.request_timeout
            response = await asyncio.wait_for(future, timeout=timeout_val)
            
            # Check for error
            if "error" in response:
                error = response["error"]
                raise ArborQueryError(
                    query=method,
                    error_code=error.get("code"),
                    message=error.get("message", "Unknown error")
                )
            
            return response.get("result", {})
            
        except asyncio.TimeoutError:
            self._pending_requests.pop(msg_id, None)
            raise ArborTimeoutError(method, timeout_val)
        except Exception:
            self._pending_requests.pop(msg_id, None)
            raise
    
    async def _attempt_reconnect(self) -> None:
        """Attempt to reconnect to Arbor server."""
        if self._reconnect_count >= self.config.connection.max_reconnects:
            logger.error("Max reconnection attempts reached")
            return
        
        self._reconnect_count += 1
        
        logger.info({
            "msg": "Attempting to reconnect to Arbor",
            "attempt": self._reconnect_count,
            "max_attempts": self.config.connection.max_reconnects
        })
        
        await asyncio.sleep(self.config.connection.reconnect_interval)
        
        try:
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat pings."""
        try:
            while self._connected:
                await asyncio.sleep(self.config.connection.heartbeat_interval)
                
                if self.is_connected:
                    try:
                        # Send ping (Arbor protocol specific)
                        await self._ws.send(json.dumps({"type": "ping"}))
                    except Exception as e:
                        logger.warning(f"Heartbeat failed: {e}")
        except asyncio.CancelledError:
            pass
    
    # =========================================================================
    # Public API Methods
    # =========================================================================
    
    async def index_codebase(self, path: str) -> IndexingResult:
        """
        Trigger full codebase indexing.
        
        Args:
            path: Root path of codebase to index
            
        Returns:
            IndexingResult with statistics
        """
        import time
        start_time = time.time()
        
        result = await self._send_request(
            "index",
            {
                "path": path,
                "languages": self.config.indexing.languages,
                "exclude_patterns": self.config.indexing.exclude_patterns
            }
        )
        
        duration = time.time() - start_time
        
        return IndexingResult(
            success=result.get("success", False),
            files_indexed=result.get("files_indexed", 0),
            nodes_created=result.get("nodes_created", 0),
            edges_created=result.get("edges_created", 0),
            errors=result.get("errors", []),
            duration_seconds=duration
        )
    
    async def query_graph(self, query: str) -> QueryResult:
        """
        Execute ArborQL query against the graph.
        
        Args:
            query: ArborQL query string
            
        Returns:
            QueryResult with nodes and edges
        """
        result = await self._send_request(
            "query",
            {"query": query}
        )
        
        return QueryResult(
            query=query,
            nodes=result.get("nodes", []),
            edges=result.get("edges", []),
            execution_time_ms=result.get("execution_time_ms", 0.0),
            total_count=result.get("total_count", 0)
        )
    
    async def find_node(self, name: str, kind: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Find a node by name.
        
        Args:
            name: Node name to search for
            kind: Optional node kind filter
            
        Returns:
            Node dictionary or None if not found
        """
        query = f'FIND * WHERE name = "{name}"'
        if kind:
            query += f' AND kind = "{kind}"'
        
        result = await self.query_graph(query)
        return result.nodes[0] if result.nodes else None
    
    async def find_path(self, start: str, end: str) -> Optional[CodePath]:
        """
        Find path between two nodes using A* algorithm.
        
        Args:
            start: Starting node identifier
            end: Ending node identifier
            
        Returns:
            CodePath or None if no path exists
        """
        result = await self._send_request(
            "find_path",
            {"start": start, "end": end}
        )
        
        if not result.get("found"):
            return None
        
        return CodePath(
            start_node=result["start"],
            end_node=result["end"],
            path=result.get("path", []),
            edges=result.get("edges", []),
            distance=result.get("distance", 0)
        )
    
    async def get_callers(self, function_name: str) -> List[Dict[str, Any]]:
        """
        Get all functions that call the specified function.
        
        Args:
            function_name: Name of the function
            
        Returns:
            List of calling functions
        """
        result = await self._send_request(
            "get_callers",
            {"function": function_name}
        )
        return result.get("callers", [])
    
    async def get_callees(self, function_name: str) -> List[Dict[str, Any]]:
        """
        Get all functions called by the specified function.
        
        Args:
            function_name: Name of the function
            
        Returns:
            List of called functions
        """
        result = await self._send_request(
            "get_callees",
            {"function": function_name}
        )
        return result.get("callees", [])
    
    async def analyze_impact(self, symbol: str, change_type: str = "modify") -> ImpactAnalysis:
        """
        Analyze the impact of modifying a symbol.
        
        Args:
            symbol: Symbol to analyze
            change_type: Type of change (rename, modify, delete)
            
        Returns:
            ImpactAnalysis with affected nodes
        """
        result = await self._send_request(
            "analyze_impact",
            {"symbol": symbol, "change_type": change_type}
        )
        
        return ImpactAnalysis(
            target_node=result.get("target", {}),
            change_type=change_type,
            direct_impacts=result.get("direct", []),
            transitive_impacts=result.get("transitive", []),
            total_affected=result.get("total_affected", 0),
            files_to_modify=result.get("files", [])
        )
    
    async def get_context(
        self,
        node_id: str,
        depth: int = 2,
        include_edges: bool = True
    ) -> QueryResult:
        """
        Get contextual subgraph around a node.
        
        Args:
            node_id: Center node ID
            depth: How many hops to include
            include_edges: Include connecting edges
            
        Returns:
            QueryResult with subgraph
        """
        result = await self._send_request(
            "get_context",
            {
                "node_id": node_id,
                "depth": depth,
                "include_edges": include_edges
            }
        )
        
        return QueryResult(
            query=f"context({node_id}, depth={depth})",
            nodes=result.get("nodes", []),
            edges=result.get("edges", []),
            total_count=result.get("total_count", 0)
        )
    
    async def export_graph(self) -> Dict[str, Any]:
        """
        Export full graph as JSON.
        
        Returns:
            Complete graph data
        """
        return await self._send_request("export_graph", {})
    
    async def subscribe_changes(self, callback: Callable[[Dict], Awaitable[None]]) -> None:
        """
        Subscribe to real-time graph changes.
        
        Args:
            callback: Async function to call on changes
        """
        self._message_handlers["graph_change"] = callback
        await self._send_request("subscribe_changes", {})
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get graph statistics.
        
        Returns:
            Statistics dictionary
        """
        return await self._send_request("get_stats", {})
