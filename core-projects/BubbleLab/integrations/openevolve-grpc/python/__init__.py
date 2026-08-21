"""
OpenEvolve gRPC integration (Python).

Exposes the gRPC server, the Python client, and the service mesh helpers.
The generated protobuf stubs live in the `generated` subpackage; regenerate them
with `python scripts/generate.py`.
"""

from .client import (
    ExecutionProgress,
    ExecutionRequest,
    ExecutionResult,
    GRPCClientConfig,
    NodeInfo,
    OpenEvolveGRPCClient,
    create_grpc_client,
    quick_execute,
)
from .server import (
    ExecutionManager,
    NodeAdapter,
    OpenEvolveGRPCServer,
    OpenEvolveServicer,
    ServerConfig,
)
from .service_mesh import Endpoint, LoadBalancer, ServiceMesh

__all__ = [
    # Client
    'OpenEvolveGRPCClient',
    'GRPCClientConfig',
    'ExecutionRequest',
    'ExecutionResult',
    'ExecutionProgress',
    'NodeInfo',
    'create_grpc_client',
    'quick_execute',
    # Server
    'OpenEvolveGRPCServer',
    'OpenEvolveServicer',
    'ServerConfig',
    'NodeAdapter',
    'ExecutionManager',
    # Service mesh
    'ServiceMesh',
    'LoadBalancer',
    'Endpoint',
]

# `rest_bridge` pulls in fastapi/uvicorn, which are optional for gRPC-only use.
try:  # pragma: no cover - optional dependency
    from .rest_bridge import RESTToGRPCBridge  # noqa: F401

    __all__.append('RESTToGRPCBridge')
except ImportError:  # pragma: no cover
    pass
