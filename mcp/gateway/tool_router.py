"""
Tool Router for the Unified MCP Gateway.

This module handles routing of tool calls to appropriate MCP servers,
including load balancing, circuit breaking, and fallback logic.
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Tool Router
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
import time
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from collections import deque

from .models import RouteDestination, ToolCallResult, CircuitBreakerState, ServerStatus
from .tool_registry import ToolRegistry, ToolDefinition

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Circuit breaker for preventing cascading failures.

    Opens after threshold failures and closes after timeout.
    """

    def __init__(self, threshold: int = 5, timeout: int = 60):
        """
        Initialize circuit breaker.

        Args:
            threshold: Number of failures before opening
            timeout: Seconds to wait before attempting to close
        """
        self.threshold = threshold
        self.timeout = timeout
        self.states: Dict[str, CircuitBreakerState] = {}

    def get_state(self, server_name: str) -> CircuitBreakerState:
        """Get or create circuit breaker state for a server."""
        if server_name not in self.states:
            self.states[server_name] = CircuitBreakerState(server_name=server_name)
        return self.states[server_name]

    def is_open(self, server_name: str) -> bool:
        """
        Check if circuit is open for a server.

        Args:
            server_name: Name of the server

        Returns:
            True if circuit is open
        """
        state = self.get_state(server_name)

        if not state.is_open:
            return False

        # Check if timeout has passed
        if state.last_failure_time:
            time_since_failure = (datetime.utcnow() - state.last_failure_time).total_seconds()
            if time_since_failure > self.timeout:
                # Try to close circuit
                logger.info(f"Circuit breaker timeout for {server_name}, attempting to close")
                state.reset()
                return False

        return True

    def record_success(self, server_name: str):
        """Record a successful call."""
        state = self.get_state(server_name)
        state.record_success()
        logger.debug(f"Circuit breaker recorded success for {server_name}")

    def record_failure(self, server_name: str):
        """Record a failed call."""
        state = self.get_state(server_name)
        state.record_failure()

        if state.failure_count >= self.threshold:
            state.is_open = True
            logger.warning(
                f"Circuit breaker OPENED for {server_name} "
                f"({state.failure_count} failures)"
            )

    def reset(self, server_name: str):
        """Reset circuit breaker for a server."""
        if server_name in self.states:
            self.states[server_name].reset()
            logger.info(f"Circuit breaker reset for {server_name}")

    def get_all_states(self) -> Dict[str, Dict]:
        """Get all circuit breaker states."""
        return {name: state.to_dict() for name, state in self.states.items()}


class LoadBalancer:
    """
    Load balancer for distributing requests across servers.
    """

    def __init__(self, strategy: str = "round_robin"):
        """
        Initialize load balancer.

        Args:
            strategy: Load balancing strategy (round_robin, least_connections, random)
        """
        self.strategy = strategy
        self.round_robin_index: Dict[str, int] = {}
        self.connections: Dict[str, int] = {}

    def select_server(self, servers: List[str], server_name: str = None) -> Optional[str]:
        """
        Select a server using the configured strategy.

        Args:
            servers: List of available server names
            server_name: Optional preferred server (overrides strategy)

        Returns:
            Selected server name
        """
        if not servers:
            return None

        # If preferred server is available and healthy, use it
        if server_name and server_name in servers:
            return server_name

        if self.strategy == "round_robin":
            return self._round_robin_select(servers)
        elif self.strategy == "least_connections":
            return self._least_connections_select(servers)
        elif self.strategy == "random":
            import random
            return random.choice(servers)
        else:
            logger.warning(f"Unknown load balancing strategy: {self.strategy}, using round_robin")
            return self._round_robin_select(servers)

    def _round_robin_select(self, servers: List[str]) -> str:
        """Select server using round-robin."""
        # Create a key for this set of servers
        key = ",".join(sorted(servers))

        if key not in self.round_robin_index:
            self.round_robin_index[key] = 0

        index = self.round_robin_index[key] % len(servers)
        self.round_robin_index[key] += 1

        return servers[index]

    def _least_connections_select(self, servers: List[str]) -> str:
        """Select server with least active connections."""
        # Find server with minimum connections
        min_server = servers[0]
        min_conns = self.connections.get(min_server, 0)

        for server in servers[1:]:
            conns = self.connections.get(server, 0)
            if conns < min_conns:
                min_server = server
                min_conns = conns

        return min_server

    def increment_connections(self, server_name: str):
        """Increment connection count for a server."""
        self.connections[server_name] = self.connections.get(server_name, 0) + 1

    def decrement_connections(self, server_name: str):
        """Decrement connection count for a server."""
        if server_name in self.connections:
            self.connections[server_name] = max(0, self.connections[server_name] - 1)


class ToolRouter:
    """
    Routes tool calls to appropriate MCP servers.

    Features:
    - Pattern-based routing
    - Load balancing
    - Circuit breaking
    - Fallback chains
    - Retry logic
    """

    def __init__(
        self,
        registry: ToolRegistry,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 60,
        load_balancing: str = "round_robin",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the tool router.

        Args:
            registry: ToolRegistry instance
            circuit_breaker_threshold: Failures before opening circuit
            circuit_breaker_timeout: Seconds before retrying open circuit
            load_balancing: Load balancing strategy
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.registry = registry
        self.circuit_breaker = CircuitBreaker(circuit_breaker_threshold, circuit_breaker_timeout)
        self.load_balancer = LoadBalancer(load_balancing)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Server status tracking
        self.server_status: Dict[str, ServerStatus] = {}
        self.server_urls: Dict[str, str] = {}

        logger.info("ToolRouter initialized")

    def register_server(self, name: str, url: str, status: ServerStatus = ServerStatus.OFFLINE):
        """
        Register a server with the router.

        Args:
            name: Server name
            url: Server URL
            status: Initial server status
        """
        self.server_urls[name] = url
        self.server_status[name] = status
        logger.info(f"Registered server {name} at {url} with status {status.value}")

    def update_server_status(self, name: str, status: ServerStatus):
        """
        Update status of a server.

        Args:
            name: Server name
            status: New server status
        """
        if name in self.server_status:
            self.server_status[name] = status
            logger.info(f"Server {name} status updated to {status.value}")

    def get_healthy_servers(self) -> List[str]:
        """
        Get list of healthy servers.

        Returns:
            List of server names that are online
        """
        return [
            name
            for name, status in self.server_status.items()
            if status == ServerStatus.ONLINE
            and not self.circuit_breaker.is_open(name)
        ]

    def route(
        self,
        tool_name: str,
        namespace: Optional[str] = None,
    ) -> Optional[RouteDestination]:
        """
        Determine which server should handle the tool.

        Args:
            tool_name: Name of the tool to route
            namespace: Optional namespace hint

        Returns:
            RouteDestination if tool found, None otherwise
        """
        # Get tool definition
        tool = self.registry.get_tool(tool_name, namespace)
        if not tool:
            logger.warning(f"Tool not found: {tool_name} (namespace: {namespace})")
            return None

        # Get server URL
        server_url = self.server_urls.get(tool.server_name)
        if not server_url:
            logger.error(f"No URL registered for server: {tool.server_name}")
            return None

        # Check circuit breaker
        if self.circuit_breaker.is_open(tool.server_name):
            logger.warning(f"Circuit breaker open for {tool.server_name}")
            return None

        # Create route destination
        destination = RouteDestination(
            server_name=tool.server_name,
            server_url=server_url,
            namespace=tool.namespace,
            tool_name=tool.name,
            priority=0,
        )

        # Add fallback servers (other servers with same tool)
        # This is a simple fallback strategy
        fallback_servers = []
        for server in self.get_healthy_servers():
            if server != tool.server_name:
                fallback_servers.append(server)

        destination.fallback_servers = fallback_servers[:3]  # Max 3 fallbacks

        return destination

    async def execute_with_retry(
        self,
        destination: RouteDestination,
        params: dict,
        execute_func: Callable,
    ) -> ToolCallResult:
        """
        Execute tool call with retry logic.

        Args:
            destination: RouteDestination
            params: Tool parameters
            execute_func: Async function to execute the tool call

        Returns:
            ToolCallResult
        """
        last_error = None
        attempt = 0

        servers_to_try = [destination.server_name] + destination.fallback_servers

        for server_name in servers_to_try:
            if attempt >= self.max_retries:
                break

            # Check circuit breaker
            if self.circuit_breaker.is_open(server_name):
                logger.warning(f"Skipping {server_name} due to open circuit breaker")
                continue

            # Get server URL
            server_url = self.server_urls.get(server_name)
            if not server_url:
                logger.error(f"No URL for server: {server_name}")
                continue

            # Track connections
            self.load_balancer.increment_connections(server_name)

            try:
                start_time = time.time()

                # Execute the tool call
                result = await execute_func(server_url, destination.tool_name, params)

                execution_time = time.time() - start_time

                # Record success
                self.circuit_breaker.record_success(server_name)

                return ToolCallResult(
                    success=True,
                    tool_name=destination.tool_name,
                    namespace=destination.namespace,
                    server_name=server_name,
                    result=result,
                    execution_time=execution_time,
                )

            except Exception as e:
                last_error = e
                attempt += 1

                logger.error(
                    f"Tool call failed on {server_name} (attempt {attempt}/{self.max_retries}): {e}"
                )

                # Record failure
                self.circuit_breaker.record_failure(server_name)

                # Wait before retry
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)

            finally:
                self.load_balancer.decrement_connections(server_name)

        # All attempts failed
        return ToolCallResult(
            success=False,
            tool_name=destination.tool_name,
            namespace=destination.namespace,
            server_name=destination.server_name,
            error=str(last_error) if last_error else "Unknown error",
        )

    async def execute_with_fallback(
        self,
        tool_name: str,
        params: dict,
        execute_func: Callable,
        namespace: Optional[str] = None,
        fallback_chain: Optional[List[str]] = None,
    ) -> ToolCallResult:
        """
        Execute with fallback servers.

        Args:
            tool_name: Name of the tool
            params: Tool parameters
            execute_func: Async function to execute the tool call
            namespace: Optional namespace
            fallback_chain: Ordered list of fallback server names

        Returns:
            ToolCallResult
        """
        # Route to primary server
        destination = self.route(tool_name, namespace)

        if not destination:
            return ToolCallResult(
                success=False,
                tool_name=tool_name,
                namespace=namespace or "",
                server_name="",
                error="Could not route tool call",
            )

        # Override fallback chain if provided
        if fallback_chain:
            destination.fallback_servers = fallback_chain

        # Execute with retry
        return await self.execute_with_retry(destination, params, execute_func)

    def get_router_stats(self) -> Dict[str, any]:
        """
        Get router statistics.

        Returns:
            Dict with router statistics
        """
        return {
            "server_status": {name: status.value for name, status in self.server_status.items()},
            "circuit_breakers": self.circuit_breaker.get_all_states(),
            "active_connections": self.load_balancer.connections.copy(),
            "healthy_servers": self.get_healthy_servers(),
        }
