"""
OpenEvolve gRPC Client for Python

Python client for connecting to the OpenEvolve gRPC server.
Used for testing and direct Python-to-Python communication.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, AsyncIterator
from datetime import datetime

import grpc

# These would be generated from protobuf
# from generated import nodes_pb2, nodes_pb2_grpc

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
    compression: Optional[int] = None


class OpenEvolveGRPCClient:
    """
    Python gRPC client for OpenEvolve.
    
    Provides methods to interact with the OpenEvolve gRPC server,
    including node execution with streaming support.
    """
    
    def __init__(self, config: Optional[GRPCClientConfig] = None):
        self.config = config or GRPCClientConfig()
        self.channel: Optional[grpc.Channel] = None
        self.stub: Optional[Any] = None  # Would be nodes_pb2_grpc.NodeRegistryStub
        self._connected = False
        
    async def connect(self) -> None:
        """Connect to the gRPC server"""
        if self._connected:
            return
            
        target = f"{self.config.host}:{self.config.port}"
        
        # Create channel credentials
        if self.config.secure:
            credentials = self.config.credentials or grpc.ssl_channel_credentials()
            self.channel = grpc.secure_channel(target, credentials)
        else:
            self.channel = grpc.insecure_channel(target)
        
        # Wait for connection
        try:
            await asyncio.wait_for(
                self._wait_for_ready(),
                timeout=self.config.connect_timeout_ms / 1000
            )
        except asyncio.TimeoutError:
            raise ConnectionError(f"Could not connect to {target} within timeout")
        
        # Create stub (would use generated code)
        # self.stub = nodes_pb2_grpc.NodeRegistryStub(self.channel)
        
        self._connected = True
        logger.info(f"Connected to OpenEvolve gRPC server at {target}")
    
    async def _wait_for_ready(self) -> None:
        """Wait for channel to be ready"""
        # In real implementation, would use channel readiness
        await asyncio.sleep(0.1)  # Stub implementation
    
    async def close(self) -> None:
        """Close the connection"""
        if self.channel:
            self.channel.close()
            self.channel = None
        self._connected = False
        self.stub = None
        logger.info("Disconnected from OpenEvolve gRPC server")
    
    async def list_nodes(self, category: Optional[str] = None) -> List[NodeInfo]:
        """
        List all available nodes.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of node information
        """
        if not self._connected:
            raise RuntimeError("Client not connected")
        
        # In real implementation, would call gRPC
        # request = nodes_pb2.ListNodesRequest(category=category)
        # response = await self.stub.ListNodes(request)
        # return [self._map_node_info(n) for n in response.nodes]
        
        # Stub implementation
        return []
    
    async def get_node_schema(self, node_type: str) -> NodeInfo:
        """
        Get schema information for a specific node.
        
        Args:
            node_type: Type of node to get schema for
            
        Returns:
            Node information including parameter schema
        """
        if not self._connected:
            raise RuntimeError("Client not connected")
        
        # In real implementation, would call gRPC
        # request = nodes_pb2.GetNodeSchemaRequest(node_type=node_type)
        # response = await self.stub.GetNodeSchema(request)
        # return self._map_node_info(response.node_info)
        
        # Stub implementation
        return NodeInfo(
            node_id=node_type,
            node_type=node_type,
            display_name=node_type,
            description="",
            icon="default",
            category="general",
            version="1.0.0"
        )
    
    async def execute_node(self, request: ExecutionRequest) -> ExecutionResult:
        """
        Execute a node synchronously.
        
        Args:
            request: Execution request with node type and inputs
            
        Returns:
            Execution result
        """
        if not self._connected:
            raise RuntimeError("Client not connected")
        
        # In real implementation, would call gRPC
        # grpc_request = self._create_execution_request(request)
        # response = await self.stub.ExecuteNode(grpc_request)
        # return self._map_execution_result(response)
        
        # Stub implementation
        return ExecutionResult(
            execution_id=f"exec_{datetime.now().timestamp()}",
            state="COMPLETED",
            result={}
        )
    
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
        if not self._connected:
            raise RuntimeError("Client not connected")
        
        # In real implementation, would call gRPC streaming
        # grpc_request = self._create_execution_request(request)
        # async for update in self.stub.ExecuteNodeStreaming(grpc_request):
        #     progress = self._map_progress(update.progress)
        #     progress_callback(progress)
        # return self._map_execution_result(update)
        
        # Stub implementation - simulate progress
        for i in range(0, 101, 10):
            progress = ExecutionProgress(
                percent=i,
                message=f"Processing... {i}%",
                stage="running"
            )
            progress_callback(progress)
            await asyncio.sleep(0.1)
        
        return ExecutionResult(
            execution_id=f"exec_{datetime.now().timestamp()}",
            state="COMPLETED",
            result={},
            progress=ExecutionProgress(percent=100, message="Complete")
        )
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel a running execution.
        
        Args:
            execution_id: ID of execution to cancel
            
        Returns:
            True if cancelled successfully
        """
        if not self._connected:
            raise RuntimeError("Client not connected")
        
        # In real implementation, would call gRPC
        # request = nodes_pb2.CancelExecutionRequest(execution_id=execution_id)
        # response = await self.stub.CancelExecution(request)
        # return response.success
        
        # Stub implementation
        return True
    
    async def get_execution_status(self, execution_id: str) -> ExecutionResult:
        """
        Get status of an execution.
        
        Args:
            execution_id: ID of execution to check
            
        Returns:
            Current execution status
        """
        if not self._connected:
            raise RuntimeError("Client not connected")
        
        # In real implementation, would call gRPC
        # request = nodes_pb2.GetExecutionStatusRequest(execution_id=execution_id)
        # response = await self.stub.GetExecutionStatus(request)
        # return self._map_execution_result(response)
        
        # Stub implementation
        return ExecutionResult(
            execution_id=execution_id,
            state="COMPLETED",
            result={}
        )
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Check server health.
        
        Returns:
            Health status information
        """
        if not self._connected:
            raise RuntimeError("Client not connected")
        
        # In real implementation, would use gRPC health check
        # stub = health_pb2_grpc.HealthStub(self.channel)
        # response = await stub.Check(health_pb2.HealthCheckRequest())
        # return {"status": response.status, "serving": response.status == 1}
        
        # Stub implementation
        return {"status": "SERVING", "serving": True}
    
    # Helper methods for mapping (would use actual protobuf types)
    def _create_execution_request(self, request: ExecutionRequest) -> Any:
        """Convert ExecutionRequest to gRPC request"""
        # Would create nodes_pb2.NodeExecutionRequest
        pass
    
    def _map_node_info(self, proto_node: Any) -> NodeInfo:
        """Map protobuf NodeInfo to Python dataclass"""
        # Would extract fields from proto message
        pass
    
    def _map_execution_result(self, proto_result: Any) -> ExecutionResult:
        """Map protobuf ExecutionResult to Python dataclass"""
        # Would extract fields from proto message
        pass
    
    def _map_progress(self, proto_progress: Any) -> ExecutionProgress:
        """Map protobuf Progress to Python dataclass"""
        # Would extract fields from proto message
        pass


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
            
            # Execute with streaming
            request = ExecutionRequest(
                node_type="decomposition",
                inputs={"problem_statement": "Design a scalable system"}
            )
            
            def on_progress(progress: ExecutionProgress):
                print(f"{progress.percent}%: {progress.message}")
            
            result = await client.execute_node_streaming(request, on_progress)
            print(f"Result: {result}")
            
        finally:
            await client.close()
    
    # Run example
    asyncio.run(main())
