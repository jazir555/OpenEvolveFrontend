"""
OpenEvolve gRPC Client for Python

Python client for connecting to the OpenEvolve gRPC server.
Used for testing and direct Python-to-Python communication.

The synchronous gRPC stubs are wrapped in an async-friendly API: unary calls are
dispatched to a thread executor so they never block the event loop, and streaming
responses are pumped from a worker thread into the calling coroutine.
"""

import asyncio
import functools
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

import grpc

# Generated protobuf/gRPC stubs (run `python scripts/generate.py`).
# Support both package-relative and flat execution.
try:
    from .generated import common_pb2, nodes_pb2, nodes_pb2_grpc, health_pb2, health_pb2_grpc
    from . import proto_mapping as pm
except ImportError:  # running flat, from inside the `python/` directory
    from generated import common_pb2, nodes_pb2, nodes_pb2_grpc, health_pb2, health_pb2_grpc
    import proto_mapping as pm

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRequest:
    """Request to execute a node"""
    node_type: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    options: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.options is None:
            self.options = {}


@dataclass
class ExecutionProgress:
    """Progress update from execution"""
    percent: int
    message: str
    stage: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionResult:
    """Result of node execution"""
    execution_id: str
    state: str  # 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    progress: Optional[ExecutionProgress] = None
    metrics: Optional[Dict[str, Any]] = None


@dataclass
class NodeInfo:
    """Information about a node type"""
    node_id: str
    node_type: str
    display_name: str
    description: str
    icon: str
    category: str
    version: str
    tags: List[str] = field(default_factory=list)
    capabilities: Optional[Dict[str, Any]] = None
    parameter_schema: Optional[Dict[str, Any]] = None


@dataclass
class GRPCClientConfig:
    """Configuration for gRPC client"""
    host: str = "localhost"
    port: int = 50051
    secure: bool = False
    credentials: Optional[grpc.ChannelCredentials] = None

    # Connection pooling
    pool_size: int = 5

    # Retry configuration
    max_retries: int = 3
    retry_delay_ms: int = 1000

    # Timeouts
    default_timeout_ms: int = 60000
    connect_timeout_ms: int = 10000

    # Keepalive
    keepalive_time_ms: int = 10000
    keepalive_timeout_ms: int = 5000

    # Compression
    compression: Optional[grpc.Compression] = None


class OpenEvolveGRPCClient:
    """
    Python gRPC client for OpenEvolve.

    Provides methods to interact with the OpenEvolve gRPC server,
    including node execution with streaming support.
    """

    def __init__(self, config: Optional[GRPCClientConfig] = None):
        self.config = config or GRPCClientConfig()
        self.channel: Optional[grpc.Channel] = None
        self.stub: Optional[nodes_pb2_grpc.NodeRegistryStub] = None
        self.health_stub: Optional[health_pb2_grpc.HealthStub] = None
        self._connected = False

    @property
    def _timeout(self) -> float:
        return self.config.default_timeout_ms / 1000

    async def connect(self) -> None:
        """Connect to the gRPC server"""
        if self._connected:
            return

        target = f"{self.config.host}:{self.config.port}"
        options = [
            ('grpc.keepalive_time_ms', self.config.keepalive_time_ms),
            ('grpc.keepalive_timeout_ms', self.config.keepalive_timeout_ms),
        ]

        if self.config.secure:
            credentials = self.config.credentials or grpc.ssl_channel_credentials()
            self.channel = grpc.secure_channel(target, credentials, options=options)
        else:
            self.channel = grpc.insecure_channel(target, options=options)

        # Wait for the channel to actually become ready before returning.
        try:
            await asyncio.wait_for(
                self._wait_for_ready(),
                timeout=self.config.connect_timeout_ms / 1000
            )
        except (asyncio.TimeoutError, grpc.FutureTimeoutError):
            self.channel.close()
            self.channel = None
            raise ConnectionError(f"Could not connect to {target} within timeout")

        self.stub = nodes_pb2_grpc.NodeRegistryStub(self.channel)
        self.health_stub = health_pb2_grpc.HealthStub(self.channel)

        self._connected = True
        logger.info(f"Connected to OpenEvolve gRPC server at {target}")

    async def _wait_for_ready(self) -> None:
        """Wait for the underlying HTTP/2 channel to be ready."""
        timeout = self.config.connect_timeout_ms / 1000
        await asyncio.get_event_loop().run_in_executor(
            None,
            functools.partial(grpc.channel_ready_future(self.channel).result, timeout=timeout),
        )

    async def close(self) -> None:
        """Close the connection"""
        if self.channel:
            self.channel.close()
            self.channel = None
        self._connected = False
        self.stub = None
        self.health_stub = None
        logger.info("Disconnected from OpenEvolve gRPC server")

    def _require_connected(self) -> None:
        if not self._connected or self.stub is None:
            raise RuntimeError("Client not connected")

    async def _call(self, method: Callable, request) -> Any:
        """Run a blocking unary stub call on the executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(method, request, timeout=self._timeout),
        )

    async def list_nodes(self, category: Optional[str] = None) -> List[NodeInfo]:
        """
        List all available nodes.

        Args:
            category: Optional category filter (e.g. "analysis", "utility")

        Returns:
            List of node information
        """
        self._require_connected()

        request = nodes_pb2.ListNodesRequest(
            metadata=pm.make_request_metadata(),
            category=pm.category_to_enum(category),
        )
        response = await self._call(self.stub.ListNodes, request)
        return [self._map_node_info(n) for n in response.nodes]

    async def get_node_schema(self, node_type: str) -> NodeInfo:
        """
        Get schema information for a specific node.

        Args:
            node_type: Type of node to get schema for

        Returns:
            Node information including parameter schema
        """
        self._require_connected()

        request = nodes_pb2.GetNodeSchemaRequest(
            metadata=pm.make_request_metadata(),
            node_type=pm.node_type_to_enum(node_type),
        )
        response = await self._call(self.stub.GetNodeSchema, request)
        return self._map_node_info(response.node_info)

    async def execute_node(self, request: ExecutionRequest) -> ExecutionResult:
        """
        Execute a node synchronously.

        Args:
            request: Execution request with node type and inputs

        Returns:
            Execution result
        """
        self._require_connected()

        grpc_request = self._create_execution_request(request)
        response = await self._call(self.stub.ExecuteNode, grpc_request)
        return self._map_execution_response(response)

    async def execute_node_streaming(
        self,
        request: ExecutionRequest,
        progress_callback: Callable[[ExecutionProgress], None]
    ) -> ExecutionResult:
        """
        Execute a node with streaming progress updates.

        Args:
            request: Execution request
            progress_callback: Callback for progress updates

        Returns:
            Final execution result
        """
        self._require_connected()

        grpc_request = self._create_execution_request(request)
        loop = asyncio.get_event_loop()
        queue: "asyncio.Queue" = asyncio.Queue()
        _SENTINEL = object()

        def pump():
            # Runs on a worker thread; the blocking stream iterator is drained
            # here and handed back to the event loop one update at a time.
            try:
                for update in self.stub.ExecuteNodeStreaming(
                    grpc_request, timeout=self._timeout
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, ("update", update))
            except Exception as e:  # noqa: BLE001 - forwarded to the coroutine
                loop.call_soon_threadsafe(queue.put_nowait, ("error", e))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", _SENTINEL))

        worker = loop.run_in_executor(None, pump)

        last_update = None
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "done":
                    break
                if kind == "error":
                    raise payload
                last_update = payload
                if payload.HasField("progress"):
                    progress_callback(self._map_progress(payload.progress))
        finally:
            await worker

        if last_update is None:
            return ExecutionResult(execution_id="", state="UNKNOWN")
        return self._map_execution_update(last_update)

    async def execute_batch(
        self,
        requests: List[ExecutionRequest],
        parallel: bool = True,
        max_concurrency: int = 0,
    ) -> List[ExecutionResult]:
        """Execute multiple nodes in a single batch call."""
        self._require_connected()

        batch = nodes_pb2.BatchExecutionRequest(
            metadata=pm.make_request_metadata(),
            requests=[self._create_execution_request(r) for r in requests],
            parallel=parallel,
            max_concurrency=max_concurrency,
        )
        response = await self._call(self.stub.ExecuteBatch, batch)
        return [self._map_execution_response(r) for r in response.responses]

    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel a running execution.

        Args:
            execution_id: ID of execution to cancel

        Returns:
            True if cancelled successfully
        """
        self._require_connected()

        request = nodes_pb2.CancelExecutionRequest(
            metadata=pm.make_request_metadata(),
            execution_id=execution_id,
        )
        response = await self._call(self.stub.CancelExecution, request)
        return response.success

    async def get_execution_status(self, execution_id: str) -> ExecutionResult:
        """
        Get status of an execution.

        Args:
            execution_id: ID of execution to check

        Returns:
            Current execution status
        """
        self._require_connected()

        request = nodes_pb2.GetExecutionStatusRequest(
            metadata=pm.make_request_metadata(),
            execution_id=execution_id,
        )
        response = await self._call(self.stub.GetExecutionStatus, request)
        return ExecutionResult(
            execution_id=response.execution_id,
            state=pm.execution_state_name(response.state),
            result=pm.struct_to_dict(response.result) if response.HasField("result") else None,
            error=self._map_error(response.error) if response.HasField("error") else None,
        )

    async def check_health(self) -> Dict[str, Any]:
        """
        Check server health.

        Returns:
            Health status information
        """
        self._require_connected()

        request = health_pb2.HealthCheckRequest(service="")
        response = await self._call(self.health_stub.Check, request)
        status_name = health_pb2.HealthCheckResponse.ServingStatus.Name(response.status)
        return {
            "status": status_name,
            "serving": response.status == health_pb2.HealthCheckResponse.SERVING,
        }

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------
    def _create_execution_request(self, request: ExecutionRequest) -> nodes_pb2.NodeExecutionRequest:
        """Convert an ExecutionRequest dataclass to the protobuf message."""
        options_msg = None
        if request.options:
            options_msg = nodes_pb2.ExecutionOptions(
                timeout_seconds=int(request.options.get("timeout_seconds", 0)),
                enable_streaming=bool(request.options.get("enable_streaming", False)),
                enable_checkpointing=bool(request.options.get("enable_checkpointing", False)),
                max_retries=int(request.options.get("max_retries", 0)),
                execution_priority=str(request.options.get("execution_priority", "")),
            )

        return nodes_pb2.NodeExecutionRequest(
            metadata=pm.make_request_metadata(),
            node_id=request.node_type,
            node_type=pm.node_type_to_enum(request.node_type),
            config=pm.dict_to_struct(request.config),
            inputs=pm.dict_to_struct(request.inputs),
            options=options_msg,
        )

    def _map_node_info(self, proto_node: nodes_pb2.NodeInfo) -> NodeInfo:
        """Map a protobuf NodeInfo to the Python dataclass."""
        capabilities = None
        if proto_node.HasField("capabilities"):
            c = proto_node.capabilities
            capabilities = {
                "supports_streaming": c.supports_streaming,
                "supports_cancellation": c.supports_cancellation,
                "supports_progress": c.supports_progress,
                "supports_checkpointing": c.supports_checkpointing,
                "supports_parallel_execution": c.supports_parallel_execution,
                "max_timeout_seconds": c.max_timeout_seconds,
            }
        return NodeInfo(
            node_id=proto_node.node_id,
            node_type=pm.enum_to_node_type(proto_node.node_type) or proto_node.node_id,
            display_name=proto_node.display_name,
            description=proto_node.description,
            icon=proto_node.icon,
            category=pm.enum_to_category(proto_node.category),
            version=proto_node.version,
            tags=list(proto_node.tags),
            capabilities=capabilities,
            parameter_schema=pm.struct_to_dict(proto_node.parameter_schema)
            if proto_node.HasField("parameter_schema") else None,
        )

    def _map_execution_response(self, response: nodes_pb2.NodeExecutionResponse) -> ExecutionResult:
        return ExecutionResult(
            execution_id=response.execution_id,
            state=pm.execution_state_name(response.state),
            result=pm.struct_to_dict(response.result) if response.HasField("result") else None,
            error=self._map_error(response.error) if response.HasField("error") else None,
            metrics=pm.struct_to_dict(response.execution_metrics)
            if response.HasField("execution_metrics") else None,
        )

    def _map_execution_update(self, update: nodes_pb2.ExecutionUpdate) -> ExecutionResult:
        return ExecutionResult(
            execution_id=update.execution_id,
            state=pm.execution_state_name(update.state),
            result=pm.struct_to_dict(update.partial_result)
            if update.HasField("partial_result") else None,
            error=self._map_error(update.error) if update.HasField("error") else None,
            progress=self._map_progress(update.progress) if update.HasField("progress") else None,
        )

    def _map_progress(self, proto_progress: common_pb2.Progress) -> ExecutionProgress:
        return ExecutionProgress(
            percent=proto_progress.percent,
            message=proto_progress.message,
            stage=proto_progress.stage or None,
        )

    def _map_error(self, proto_error: common_pb2.ErrorDetails) -> Dict[str, Any]:
        return {
            "error_code": proto_error.error_code,
            "message": proto_error.message,
            "stack_trace": proto_error.stack_trace,
            "retryable": proto_error.retryable,
        }


# Convenience functions

def create_grpc_client(
    host: str = "localhost",
    port: int = 50051,
    **kwargs
) -> OpenEvolveGRPCClient:
    """
    Create a gRPC client with the given configuration.

    Args:
        host: Server host
        port: Server port
        **kwargs: Additional configuration options

    Returns:
        Configured gRPC client
    """
    config = GRPCClientConfig(host=host, port=port, **kwargs)
    return OpenEvolveGRPCClient(config)


async def quick_execute(
    node_type: str,
    inputs: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    host: str = "localhost",
    port: int = 50051
) -> ExecutionResult:
    """
    Quick execute a node without managing client lifecycle.

    Args:
        node_type: Type of node to execute
        inputs: Input data
        config: Optional node configuration
        host: Server host
        port: Server port

    Returns:
        Execution result
    """
    client = create_grpc_client(host=host, port=port)

    try:
        await client.connect()

        request = ExecutionRequest(
            node_type=node_type,
            inputs=inputs,
            config=config or {}
        )

        return await client.execute_node(request)
    finally:
        await client.close()


# Example usage
if __name__ == "__main__":
    async def main():
        # Create and connect client
        client = create_grpc_client()
        await client.connect()

        try:
            # List nodes
            nodes = await client.list_nodes()
            print(f"Available nodes: {len(nodes)}")
            for node in nodes:
                print(f"  - {node.node_id}: {node.display_name} ({node.category})")

            # Execute with streaming
            request = ExecutionRequest(
                node_type="decomposition",
                inputs={"problem_statement": "Design a scalable system and test it"}
            )

            def on_progress(progress: ExecutionProgress):
                print(f"{progress.percent}%: {progress.message}")

            result = await client.execute_node_streaming(request, on_progress)
            print(f"Result: {result}")

        finally:
            await client.close()

    # Run example
    asyncio.run(main())
