"""
OpenEvolve gRPC Server

High-performance gRPC server that wraps the existing bubblelabs_nodes Python implementations.
Provides streaming support, health checks, and efficient binary serialization.
"""

import logging
import os
import queue
import signal
import sys
import threading
import time
import traceback
import uuid
from concurrent import futures
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type

import grpc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Generated protobuf modules (run `python scripts/generate.py` to (re)create them).
# Support both package-relative and flat imports so the module works whether it is
# imported as `openevolve_grpc.python.server` or run directly from `python/`.
try:
    from .generated import common_pb2, health_pb2, health_pb2_grpc, nodes_pb2, nodes_pb2_grpc
    from . import proto_mapping as pm
    from .local_nodes import NodeRegistry as LocalNodeRegistry
except ImportError:  # running flat, from inside the `python/` directory
    from generated import common_pb2, health_pb2, health_pb2_grpc, nodes_pb2, nodes_pb2_grpc
    import proto_mapping as pm
    from local_nodes import NodeRegistry as LocalNodeRegistry

# Repo root, used to locate the real `bubblelabs_nodes` package:
# python/ -> openevolve-grpc/ -> integrations/ -> BubbleLab/ -> core-projects/ -> <repo root>
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')
)

# Keep the last N finished executions so GetExecutionStatus still answers after
# an execution completes.
EXECUTION_HISTORY_LIMIT = 256


@dataclass
class ServerConfig:
    """Configuration for the gRPC server"""
    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 10
    max_concurrent_rpcs: int = 100
    compression: grpc.Compression = grpc.Compression.Gzip
    enable_reflection: bool = True
    keepalive_time_ms: int = 10000
    keepalive_timeout_ms: int = 5000

    # Feature flags
    enable_health_check: bool = True
    enable_metrics: bool = True
    enable_streaming: bool = True
    enable_cancellation: bool = True

    # Node loading. The real `bubblelabs_nodes` package imports the whole
    # openevolve stack, so it is opt-in; local seed nodes are always available.
    use_real_nodes: bool = False
    default_timeout_seconds: int = 300


@dataclass
class ExecutionContext:
    """Context for node execution"""
    execution_id: str
    node_type: str
    start_time: float = field(default_factory=time.time)
    cancelled: bool = False
    progress_callbacks: List[Callable] = field(default_factory=list)
    state: int = common_pb2.EXECUTION_STATE_PENDING
    result: Optional[Dict] = None
    error: Optional[str] = None
    completed_time: Optional[float] = None

    def is_cancelled(self) -> bool:
        return self.cancelled

    def cancel(self):
        self.cancelled = True
        self.state = common_pb2.EXECUTION_STATE_CANCELLED
        for callback in self.progress_callbacks:
            try:
                callback({"percent": 0, "message": "cancelled"})
            except Exception:
                pass


class ExecutionManager:
    """
    Manages active executions for cancellation and monitoring.

    Thread-based rather than asyncio-based: the gRPC sync server dispatches each
    RPC onto a thread-pool worker, so an asyncio.Lock here would either be used
    from several short-lived event loops or block the servicer needlessly.
    """

    def __init__(self, history_limit: int = EXECUTION_HISTORY_LIMIT):
        self._executions: Dict[str, ExecutionContext] = {}
        self._history: Dict[str, ExecutionContext] = {}
        self._history_limit = history_limit
        self._lock = threading.RLock()

    def create_execution(self, execution_id: str, node_type: str) -> ExecutionContext:
        with self._lock:
            ctx = ExecutionContext(execution_id=execution_id, node_type=node_type)
            ctx.state = common_pb2.EXECUTION_STATE_RUNNING
            self._executions[execution_id] = ctx
            return ctx

    def get_execution(self, execution_id: str) -> Optional[ExecutionContext]:
        with self._lock:
            return self._executions.get(execution_id) or self._history.get(execution_id)

    def cancel_execution(self, execution_id: str) -> bool:
        with self._lock:
            ctx = self._executions.get(execution_id)
            if ctx:
                ctx.cancel()
                return True
            return False

    def complete_execution(
        self,
        execution_id: str,
        state: Optional[int] = None,
        result: Optional[Dict] = None,
        error: Optional[str] = None,
    ):
        with self._lock:
            ctx = self._executions.pop(execution_id, None)
            if not ctx:
                return
            ctx.completed_time = time.time()
            if state is not None:
                ctx.state = state
            elif ctx.state == common_pb2.EXECUTION_STATE_RUNNING:
                ctx.state = common_pb2.EXECUTION_STATE_COMPLETED
            if result is not None:
                ctx.result = result
            if error is not None:
                ctx.error = error
            self._history[execution_id] = ctx
            while len(self._history) > self._history_limit:
                self._history.pop(next(iter(self._history)))

    def list_executions(self) -> List[ExecutionContext]:
        with self._lock:
            return list(self._executions.values())


class WorkflowStateWrapper:
    """
    Bridges gRPC execution context to the `WorkflowState` object that
    bubblelabs_nodes' `execute(inputs, context)` expects.
    """

    def __init__(self, exec_ctx: Optional[ExecutionContext]):
        self.exec_ctx = exec_ctx
        self.artifacts: Dict[str, Any] = {}
        self.errors: List[Dict] = []

    def generate_execution_id(self) -> str:
        return self.exec_ctx.execution_id if self.exec_ctx else str(time.time())

    def add_artifact(self, name: str, data: Dict):
        self.artifacts[name] = data

    def add_error(self, error: Dict):
        self.errors.append(error)

    def update_progress(self, progress: int, message: str = ""):
        if not self.exec_ctx:
            return
        for callback in self.exec_ctx.progress_callbacks:
            try:
                callback({"percent": progress, "message": message})
            except Exception:
                logger.debug("Progress callback failed", exc_info=True)


class NodeAdapter:
    """
    Adapter that wraps existing bubblelabs_nodes to work with gRPC.
    This bridges the old Python API with the new gRPC interface.
    """

    def __init__(self, use_real_nodes: Optional[bool] = None):
        if use_real_nodes is None:
            use_real_nodes = os.getenv('OPENEVOLVE_USE_REAL_NODES', '').lower() in ('1', 'true', 'yes')
        self.use_real_nodes = use_real_nodes
        self._node_registry: Dict[str, Type] = {}
        self._import_nodes()

    def _import_nodes(self):
        """Populate the registry from local seed nodes and, optionally, bubblelabs_nodes."""
        # Local seed nodes are dependency-free and always present.
        self._node_registry = LocalNodeRegistry.list_nodes()
        logger.info(f"Loaded {len(self._node_registry)} local seed nodes")

        if not self.use_real_nodes:
            return

        try:
            # The real package lives at the monorepo root; put the root (not the
            # package dir) on sys.path so `import bubblelabs_nodes` resolves.
            if REPO_ROOT not in sys.path:
                sys.path.insert(0, REPO_ROOT)

            from bubblelabs_nodes import NodeRegistry  # type: ignore

            real_nodes = NodeRegistry.list_nodes()
            self._node_registry.update(real_nodes)
            logger.info(f"Loaded {len(real_nodes)} nodes from bubblelabs_nodes")
        except ImportError as e:
            logger.warning(f"Could not import NodeRegistry: {e}")
        except Exception as e:
            logger.error(f"Error importing nodes: {e}")

    def get_node(self, node_type: str, config: Optional[Dict] = None):
        """Get a node instance by type"""
        node_class = self._node_registry.get(node_type)
        if not node_class:
            raise ValueError(f"Unknown node type: {node_type}")
        return node_class(config or {})

    def list_nodes(self) -> Dict[str, Type]:
        """List all available node types"""
        return self._node_registry.copy()

    def execute_node(
        self,
        node_type: str,
        inputs: Dict,
        config: Optional[Dict] = None,
        ctx: Optional[ExecutionContext] = None
    ) -> Dict:
        """
        Execute a node with the given inputs.

        This method wraps the existing bubblelabs_nodes API and provides
        integration with the new gRPC execution context.
        """
        node = self.get_node(node_type, config)
        workflow_state = WorkflowStateWrapper(ctx)

        # Check for cancellation before execution
        if ctx and ctx.is_cancelled():
            raise ExecutionCancelled("Execution was cancelled")

        # Execute the node (bubblelabs_nodes' API is synchronous)
        try:
            if hasattr(node, 'execute_safe'):
                result = node.execute_safe(inputs, workflow_state)
            elif hasattr(node, 'execute'):
                result = node.execute(inputs, workflow_state)
            else:
                raise RuntimeError(f"Node {node_type} has no execute method")

            # Check for cancellation after execution
            if ctx and ctx.is_cancelled():
                raise ExecutionCancelled("Execution was cancelled")

            return {
                "result": result,
                "artifacts": workflow_state.artifacts,
                "errors": workflow_state.errors
            }

        except ExecutionCancelled:
            raise
        except Exception as e:
            logger.error(f"Node execution failed: {e}")
            raise


class ExecutionCancelled(Exception):
    """Raised when an execution is cancelled by the client or a Cancel RPC."""


class HealthServicer(health_pb2_grpc.HealthServicer):
    """
    Health checking service for gRPC.

    Backed by this repo's own `health.proto`, which declares the canonical
    `grpc.health.v1` contract. Using the generated stubs (rather than the
    optional `grpcio-health-checking` package) keeps the dependency set small
    and avoids registering two copies of `grpc.health.v1` in the descriptor pool.
    """

    def __init__(self, node_adapter: Optional[NodeAdapter] = None):
        self.node_adapter = node_adapter
        self._status = health_pb2.HealthCheckResponse.SERVING

    def Check(self, request, context):
        """Standard gRPC health check"""
        return health_pb2.HealthCheckResponse(status=self._status)

    def Watch(self, request, context):
        """Streaming health check"""
        while context.is_active():
            yield health_pb2.HealthCheckResponse(status=self._status)
            time.sleep(5)

    def set_status(self, status):
        if isinstance(status, str):
            status = health_pb2.HealthCheckResponse.ServingStatus.Value(status)
        self._status = status


class OpenEvolveServicer(nodes_pb2_grpc.NodeRegistryServicer):
    """
    Main gRPC service implementation.
    Implements the NodeRegistry service defined in the proto files.
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self.node_adapter = NodeAdapter(use_real_nodes=config.use_real_nodes)
        self.execution_manager = ExecutionManager()
        self.version = "2.0.0-grpc"
        self.start_time = time.time()

    # =========================================================================
    # Helpers
    # =========================================================================

    def _response_metadata(self, request, started: float) -> common_pb2.ResponseMetadata:
        """Build response metadata, echoing the caller's request/correlation ids."""
        request_id = ""
        correlation_id = ""
        if request is not None and request.HasField("metadata"):
            request_id = request.metadata.request_id
            correlation_id = request.metadata.correlation_id
        return pm.make_response_metadata(
            request_id=request_id,
            correlation_id=correlation_id,
            processing_time_ms=int((time.time() - started) * 1000),
            server_version=self.version,
        )

    def _execution_id(self, request) -> str:
        """Use the caller's request id when present, else mint one."""
        if request is not None and request.HasField("metadata") and request.metadata.request_id:
            return request.metadata.request_id
        return f"exec_{uuid.uuid4().hex}"

    def _resolve_node_type(self, request) -> str:
        """
        Resolve the registry key for an execution request.

        `node_id` (free-form string) wins over the `NodeType` enum so nodes that
        predate the enum are still addressable.
        """
        node_type = getattr(request, "node_id", "") or ""
        if not node_type:
            node_type = pm.enum_to_node_type(getattr(request, "node_type", 0))
        return node_type

    def _capabilities(self, node) -> nodes_pb2.NodeCapabilities:
        return nodes_pb2.NodeCapabilities(
            supports_streaming=self.config.enable_streaming,
            supports_cancellation=self.config.enable_cancellation,
            supports_progress=hasattr(node, 'execute'),
            supports_checkpointing=hasattr(node, 'checkpoint'),
            supports_parallel_execution=True,
            max_timeout_seconds=self.config.default_timeout_seconds,
        )

    def _node_info(self, node_type: str, node_class: Type) -> Optional[nodes_pb2.NodeInfo]:
        """Build a NodeInfo message by introspecting a node class/instance."""
        try:
            node = node_class({})
        except Exception as e:
            logger.warning(f"Error getting info for {node_type}: {e}")
            return None

        schema: Dict[str, Any] = {}
        if hasattr(node, 'get_parameter_schema'):
            try:
                schema = node.get_parameter_schema() or {}
            except Exception as e:
                logger.warning(f"Could not get schema for {node_type}: {e}")

        return nodes_pb2.NodeInfo(
            node_id=node_type,
            node_type=pm.node_type_to_enum(node_type),
            category=pm.category_to_enum(getattr(node, 'CATEGORY', 'general')),
            display_name=getattr(node, 'DISPLAY_NAME', node_type),
            description=getattr(node, 'DESCRIPTION', ''),
            icon=getattr(node, 'ICON', 'default'),
            version=getattr(node, 'VERSION', '1.0.0'),
            tags=list(getattr(node, 'TAGS', []) or []),
            capabilities=self._capabilities(node),
            parameter_schema=pm.dict_to_struct(schema),
        )

    # =========================================================================
    # Node Registry Service Methods
    # =========================================================================

    def ListNodes(self, request, context):
        """List all available nodes"""
        started = time.time()

        try:
            nodes = self.node_adapter.list_nodes()

            node_infos: List[nodes_pb2.NodeInfo] = []
            for node_type, node_class in sorted(nodes.items()):
                info = self._node_info(node_type, node_class)
                if info is not None:
                    node_infos.append(info)

            # Filters
            if request.category:
                node_infos = [n for n in node_infos if n.category == request.category]

            if request.search_query:
                needle = request.search_query.lower()
                node_infos = [
                    n for n in node_infos
                    if needle in n.node_id.lower()
                    or needle in n.display_name.lower()
                    or needle in n.description.lower()
                ]

            if request.tags:
                wanted = set(request.tags)
                node_infos = [n for n in node_infos if wanted.issubset(set(n.tags))]

            total_count = len(node_infos)

            # Pagination (page is 1-based; 0 means "first page")
            page = request.pagination.page if request.HasField("pagination") else 0
            page_size = request.pagination.page_size if request.HasField("pagination") else 0
            has_next = False
            if page_size > 0:
                page_index = max(page - 1, 0)
                start = page_index * page_size
                node_infos = node_infos[start:start + page_size]
                has_next = start + page_size < total_count

            return nodes_pb2.ListNodesResponse(
                metadata=self._response_metadata(request, started),
                nodes=node_infos,
                pagination=common_pb2.PaginationInfo(
                    total_count=total_count,
                    page=page,
                    page_size=page_size,
                    has_next=has_next,
                ),
            )

        except Exception as e:
            logger.error(f"ListNodes failed: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def GetNodeSchema(self, request, context):
        """Get schema for a specific node"""
        started = time.time()
        node_type = pm.enum_to_node_type(request.node_type)

        if not node_type:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "node_type must be a known NodeType value",
            )

        node_class = self.node_adapter.list_nodes().get(node_type)
        if node_class is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown node type: {node_type}")

        info = self._node_info(node_type, node_class)
        if info is None:
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"Could not introspect node type: {node_type}",
            )

        return nodes_pb2.GetNodeSchemaResponse(
            metadata=self._response_metadata(request, started),
            node_info=info,
        )

    def ExecuteNode(self, request, context):
        """Execute a node synchronously"""
        started = time.time()
        execution_id = self._execution_id(request)
        node_type = self._resolve_node_type(request)

        if node_type not in self.node_adapter.list_nodes():
            context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown node type: {node_type}")

        exec_ctx = self.execution_manager.create_execution(execution_id, node_type)

        inputs = pm.struct_to_dict(request.inputs) if request.HasField("inputs") else {}
        config = pm.struct_to_dict(request.config) if request.HasField("config") else {}

        try:
            result = self.node_adapter.execute_node(node_type, inputs, config, exec_ctx)
            self.execution_manager.complete_execution(
                execution_id,
                state=common_pb2.EXECUTION_STATE_COMPLETED,
                result=result,
            )

            return nodes_pb2.NodeExecutionResponse(
                metadata=self._response_metadata(request, started),
                execution_id=execution_id,
                state=common_pb2.EXECUTION_STATE_COMPLETED,
                result=pm.dict_to_struct(result.get("result", {})),
                final_progress=pm.progress(100, "Complete", "completed"),
                execution_metrics=pm.dict_to_struct({
                    "processing_time_ms": int((time.time() - started) * 1000),
                    "artifact_count": len(result.get("artifacts", {})),
                }),
            )

        except ExecutionCancelled:
            self.execution_manager.complete_execution(
                execution_id, state=common_pb2.EXECUTION_STATE_CANCELLED
            )
            context.abort(grpc.StatusCode.CANCELLED, "Execution was cancelled")

        except Exception as e:
            self.execution_manager.complete_execution(
                execution_id,
                state=common_pb2.EXECUTION_STATE_FAILED,
                error=str(e),
            )
            logger.error(f"ExecuteNode failed: {e}")
            # Node-level failures are a normal outcome, not a transport error:
            # report them in the response so callers get the error details.
            return nodes_pb2.NodeExecutionResponse(
                metadata=self._response_metadata(request, started),
                execution_id=execution_id,
                state=common_pb2.EXECUTION_STATE_FAILED,
                error=pm.error_details(
                    error_code=type(e).__name__,
                    message=str(e),
                    stack_trace=traceback.format_exc(),
                    retryable=False,
                ),
            )

    def ExecuteNodeStreaming(self, request, context) -> Iterator[nodes_pb2.ExecutionUpdate]:
        """Execute a node with streaming progress updates"""
        execution_id = self._execution_id(request)
        node_type = self._resolve_node_type(request)

        if node_type not in self.node_adapter.list_nodes():
            context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown node type: {node_type}")

        exec_ctx = self.execution_manager.create_execution(execution_id, node_type)

        # Node code runs on a worker thread; progress hops back over a queue.
        progress_queue: "queue.Queue[Dict]" = queue.Queue()
        exec_ctx.progress_callbacks.append(progress_queue.put)

        inputs = pm.struct_to_dict(request.inputs) if request.HasField("inputs") else {}
        config = pm.struct_to_dict(request.config) if request.HasField("config") else {}

        result_holder: List[Optional[Dict]] = [None]
        error_holder: List[Optional[BaseException]] = [None]

        def run():
            try:
                result_holder[0] = self.node_adapter.execute_node(
                    node_type, inputs, config, exec_ctx
                )
            except BaseException as e:  # noqa: BLE001 - surfaced to the stream below
                error_holder[0] = e

        yield nodes_pb2.ExecutionUpdate(
            execution_id=execution_id,
            state=common_pb2.EXECUTION_STATE_RUNNING,
            progress=pm.progress(0, "Starting...", "starting"),
        )

        worker = threading.Thread(target=run, name=f"exec-{execution_id}", daemon=True)
        worker.start()

        try:
            while True:
                if not context.is_active():
                    # Client disconnected: cancel and stop streaming.
                    self.execution_manager.cancel_execution(execution_id)
                    return
                try:
                    update = progress_queue.get(timeout=0.05)
                except queue.Empty:
                    if not worker.is_alive():
                        break
                    continue

                yield nodes_pb2.ExecutionUpdate(
                    execution_id=execution_id,
                    state=common_pb2.EXECUTION_STATE_RUNNING,
                    progress=pm.progress(
                        update.get("percent", 0),
                        update.get("message", ""),
                        update.get("stage", "running"),
                    ),
                )

            worker.join()

            # Drain any progress published between the last poll and completion.
            while not progress_queue.empty():
                update = progress_queue.get_nowait()
                yield nodes_pb2.ExecutionUpdate(
                    execution_id=execution_id,
                    state=common_pb2.EXECUTION_STATE_RUNNING,
                    progress=pm.progress(
                        update.get("percent", 0),
                        update.get("message", ""),
                        update.get("stage", "running"),
                    ),
                )

            error = error_holder[0]
            if isinstance(error, ExecutionCancelled):
                self.execution_manager.complete_execution(
                    execution_id, state=common_pb2.EXECUTION_STATE_CANCELLED
                )
                yield nodes_pb2.ExecutionUpdate(
                    execution_id=execution_id,
                    state=common_pb2.EXECUTION_STATE_CANCELLED,
                    progress=pm.progress(0, "Cancelled", "cancelled"),
                )
                return

            if error is not None:
                self.execution_manager.complete_execution(
                    execution_id,
                    state=common_pb2.EXECUTION_STATE_FAILED,
                    error=str(error),
                )
                yield nodes_pb2.ExecutionUpdate(
                    execution_id=execution_id,
                    state=common_pb2.EXECUTION_STATE_FAILED,
                    error=pm.error_details(
                        error_code=type(error).__name__,
                        message=str(error),
                        retryable=False,
                    ),
                )
                return

            result = result_holder[0] or {}
            self.execution_manager.complete_execution(
                execution_id,
                state=common_pb2.EXECUTION_STATE_COMPLETED,
                result=result,
            )
            yield nodes_pb2.ExecutionUpdate(
                execution_id=execution_id,
                state=common_pb2.EXECUTION_STATE_COMPLETED,
                progress=pm.progress(100, "Complete", "completed"),
                partial_result=pm.dict_to_struct(result.get("result", {})),
            )

        finally:
            if exec_ctx.progress_callbacks:
                try:
                    exec_ctx.progress_callbacks.remove(progress_queue.put)
                except ValueError:
                    pass
            self.execution_manager.complete_execution(execution_id)

    def ExecuteBatch(self, request, context):
        """Execute several node requests, optionally in parallel"""
        started = time.time()
        responses: List[nodes_pb2.NodeExecutionResponse] = []

        def run_one(sub_request) -> nodes_pb2.NodeExecutionResponse:
            try:
                return self.ExecuteNode(sub_request, _BatchContext(context))
            except _BatchAbort as abort:
                # One bad sub-request must not fail the whole batch.
                return nodes_pb2.NodeExecutionResponse(
                    metadata=self._response_metadata(sub_request, started),
                    execution_id=self._execution_id(sub_request),
                    state=common_pb2.EXECUTION_STATE_FAILED,
                    error=pm.error_details(
                        error_code=str(abort.code),
                        message=abort.details,
                        retryable=False,
                    ),
                )

        if request.parallel and len(request.requests) > 1:
            max_workers = request.max_concurrency or min(len(request.requests), self.config.max_workers)
            with futures.ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
                responses = list(pool.map(run_one, request.requests))
        else:
            responses = [run_one(r) for r in request.requests]

        succeeded = sum(1 for r in responses if r.state == common_pb2.EXECUTION_STATE_COMPLETED)

        return nodes_pb2.BatchExecutionResponse(
            metadata=self._response_metadata(request, started),
            responses=responses,
            succeeded=succeeded,
            failed=len(responses) - succeeded,
            total=len(responses),
        )

    def GetExecutionStatus(self, request, context):
        """Get status of an execution"""
        started = time.time()
        execution_id = request.execution_id

        exec_ctx = self.execution_manager.get_execution(execution_id)
        if not exec_ctx:
            context.abort(
                grpc.StatusCode.NOT_FOUND, f"Execution {execution_id} not found"
            )

        end = exec_ctx.completed_time or time.time()
        response = nodes_pb2.GetExecutionStatusResponse(
            metadata=self._response_metadata(request, started),
            execution_id=execution_id,
            state=exec_ctx.state,
            elapsed_seconds=int(end - exec_ctx.start_time),
        )
        response.started_at.FromSeconds(int(exec_ctx.start_time))
        if exec_ctx.completed_time:
            response.completed_at.FromSeconds(int(exec_ctx.completed_time))
        if exec_ctx.result:
            response.result.CopyFrom(pm.dict_to_struct(exec_ctx.result.get("result", {})))
        if exec_ctx.error:
            response.error.CopyFrom(
                pm.error_details("EXECUTION_ERROR", exec_ctx.error)
            )
        return response

    def CancelExecution(self, request, context):
        """Cancel a running execution"""
        started = time.time()
        success = self.execution_manager.cancel_execution(request.execution_id)

        return nodes_pb2.CancelExecutionResponse(
            metadata=self._response_metadata(request, started),
            success=success,
            message="Cancelled" if success else "Execution not found",
            final_state=(
                common_pb2.EXECUTION_STATE_CANCELLED if success
                else common_pb2.EXECUTION_STATE_UNSPECIFIED
            ),
        )


class _BatchContext:
    """
    Minimal servicer context used for the sub-requests of ExecuteBatch.

    `ExecuteNode` may call `context.abort()`, which would tear down the whole
    batch RPC. This wrapper turns an abort into an exception local to the one
    sub-request so the remaining requests still run.
    """

    def __init__(self, parent):
        self._parent = parent

    def abort(self, code, details):
        raise _BatchAbort(code, details)

    def is_active(self) -> bool:
        return self._parent.is_active()

    def __getattr__(self, name):
        return getattr(self._parent, name)


class _BatchAbort(Exception):
    def __init__(self, code, details):
        self.code = code
        self.details = details
        super().__init__(details)


class OpenEvolveGRPCServer:
    """
    Main gRPC server class.
    Manages the server lifecycle and all services.
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        self.server: Optional[grpc.Server] = None
        self.servicer = OpenEvolveServicer(self.config)
        self.health_servicer: Optional[HealthServicer] = None
        self.bound_port: Optional[int] = None
        self._shutdown_event = threading.Event()

    def _create_server(self) -> grpc.Server:
        """Create the gRPC server with all configurations"""

        # Server options
        options = [
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.keepalive_time_ms', self.config.keepalive_time_ms),
            ('grpc.keepalive_timeout_ms', self.config.keepalive_timeout_ms),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.http2.min_ping_interval_without_data_ms', 5000),
        ]

        # Create server
        server = grpc.server(
            thread_pool=futures.ThreadPoolExecutor(max_workers=self.config.max_workers),
            options=options,
            compression=self.config.compression,
            maximum_concurrent_rpcs=self.config.max_concurrent_rpcs
        )

        return server

    def _register_services(self) -> List[str]:
        """Register all servicers, returning the served service names."""
        service_names: List[str] = []

        # Application service
        nodes_pb2_grpc.add_NodeRegistryServicer_to_server(self.servicer, self.server)
        service_names.append(
            nodes_pb2.DESCRIPTOR.services_by_name['NodeRegistry'].full_name
        )
        logger.info(
            "NodeRegistry service registered with "
            f"{len(self.servicer.node_adapter.list_nodes())} nodes"
        )

        # Health check service (from this repo's health.proto)
        if self.config.enable_health_check:
            self.health_servicer = HealthServicer(self.servicer.node_adapter)
            health_pb2_grpc.add_HealthServicer_to_server(self.health_servicer, self.server)
            service_names.append(
                health_pb2.DESCRIPTOR.services_by_name['Health'].full_name
            )
            logger.info("Health check service enabled")

        # Reflection service (optional dependency)
        if self.config.enable_reflection:
            try:
                from grpc_reflection.v1alpha import reflection

                reflection.enable_server_reflection(
                    service_names + [reflection.SERVICE_NAME], self.server
                )
                logger.info("Reflection service enabled")
            except ImportError:
                logger.warning(
                    "grpc_reflection not available, skipping reflection "
                    "(pip install grpcio-reflection)"
                )

        return service_names

    def start(self, block: bool = True) -> int:
        """
        Start the gRPC server.

        Args:
            block: when True (default) run until a shutdown signal arrives.
                   Pass False to start in the background (used by tests).

        Returns:
            The port the server is actually listening on. Useful when
            `config.port` is 0, which asks the OS for a free port.
        """
        logger.info(f"Starting OpenEvolve gRPC Server v{self.servicer.version}")

        self.server = self._create_server()
        self._register_services()

        # Bind to port (port 0 -> OS-assigned ephemeral port)
        address = f"{self.config.host}:{self.config.port}"
        self.bound_port = self.server.add_insecure_port(address)
        if self.bound_port == 0:
            raise RuntimeError(f"Could not bind gRPC server to {address}")

        # Start server
        self.server.start()
        logger.info(f"gRPC Server started on {self.config.host}:{self.bound_port}")

        if not block:
            return self.bound_port

        # Setup signal handlers
        self._setup_signal_handlers()

        try:
            self._wait_for_shutdown()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            self.stop()

        return self.bound_port

    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown"""

        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}")
            self._shutdown_event.set()

        # Signal handlers can only be installed from the main thread.
        if threading.current_thread() is not threading.main_thread():
            logger.debug("Not on main thread, skipping signal handlers")
            return

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if hasattr(signal, 'SIGQUIT'):
            signal.signal(signal.SIGQUIT, signal_handler)

    def _wait_for_shutdown(self):
        """Wait for shutdown signal"""
        while not self._shutdown_event.wait(timeout=0.5):
            pass

    def stop(self, grace_period: float = 5.0):
        """Stop the gRPC server gracefully"""
        logger.info("Stopping gRPC server...")
        self._shutdown_event.set()

        if self.server:
            self.server.stop(grace_period).wait(grace_period + 1)
            self.server = None
            logger.info("gRPC server stopped")


def main():
    """Main entry point"""

    # Load configuration from environment
    config = ServerConfig(
        host=os.getenv('GRPC_HOST', '0.0.0.0'),
        port=int(os.getenv('GRPC_PORT', '50051')),
        max_workers=int(os.getenv('GRPC_MAX_WORKERS', '10')),
        enable_reflection=os.getenv('GRPC_ENABLE_REFLECTION', 'true').lower() == 'true',
        enable_health_check=os.getenv('GRPC_ENABLE_HEALTH', 'true').lower() == 'true',
        use_real_nodes=os.getenv('OPENEVOLVE_USE_REAL_NODES', 'false').lower() in ('1', 'true', 'yes'),
    )

    server = OpenEvolveGRPCServer(config)
    server.start()


if __name__ == '__main__':
    main()
