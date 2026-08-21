"""
Shared protobuf <-> Python mapping helpers for the OpenEvolve gRPC integration.

Both ``server.py`` and ``client.py`` need to move between protobuf messages and
plain Python dicts, and between the string node identifiers used by
``bubblelabs_nodes`` (``"causal_analysis"``) and the ``NodeType`` enum declared in
``nodes.proto`` (``NODE_TYPE_CAUSAL_ANALYSIS``). Keeping that logic in one place
guarantees client and server agree on the wire contract.
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Dict, Optional

from google.protobuf import json_format, struct_pb2, timestamp_pb2

try:  # imported as part of a package (e.g. `from .proto_mapping import ...`)
    from .generated import common_pb2, nodes_pb2
except ImportError:  # running flat, from inside the `python/` directory
    from generated import common_pb2, nodes_pb2

NODE_TYPE_PREFIX = "NODE_TYPE_"
NODE_CATEGORY_PREFIX = "NODE_CATEGORY_"
EXECUTION_STATE_PREFIX = "EXECUTION_STATE_"


# ---------------------------------------------------------------------------
# Struct <-> dict
# ---------------------------------------------------------------------------

def _jsonify(value: Any) -> Any:
    """
    Coerce arbitrary Python values into something ``google.protobuf.Struct`` can
    hold. Node implementations return whatever they like (sets, dataclasses,
    exceptions, NaN); refusing to serialize those would surface as an opaque
    INTERNAL error, so unsupported values degrade to their ``repr``/``str``.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        # Struct is JSON-backed: NaN/Infinity are not representable.
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return str(value)
        return value
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonify(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _jsonify(value.to_dict())
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return str(value)


def dict_to_struct(data: Optional[Dict[str, Any]]) -> struct_pb2.Struct:
    """Convert a Python dict to a protobuf Struct (lossy for exotic types)."""
    struct = struct_pb2.Struct()
    if data:
        json_format.ParseDict(_jsonify(data), struct)
    return struct


def struct_to_dict(struct: Optional[struct_pb2.Struct]) -> Dict[str, Any]:
    """Convert a protobuf Struct to a plain Python dict."""
    if struct is None:
        return {}
    return json_format.MessageToDict(struct)


# ---------------------------------------------------------------------------
# Enum mapping
# ---------------------------------------------------------------------------

def node_type_to_enum(node_type: str) -> int:
    """
    Map a registry key (``"causal_analysis"``) to ``NodeType``.

    Unknown node types map to ``NODE_TYPE_UNSPECIFIED`` rather than raising:
    the registry is the source of truth for which nodes exist, and the proto
    enum only enumerates the ones known when the schema was written.
    """
    if not node_type:
        return nodes_pb2.NODE_TYPE_UNSPECIFIED
    try:
        return nodes_pb2.NodeType.Value(NODE_TYPE_PREFIX + node_type.upper())
    except ValueError:
        return nodes_pb2.NODE_TYPE_UNSPECIFIED


def enum_to_node_type(value: int) -> str:
    """Map a ``NodeType`` enum value back to its registry key."""
    if not value:
        return ""
    try:
        name = nodes_pb2.NodeType.Name(value)
    except ValueError:
        return ""
    if name.startswith(NODE_TYPE_PREFIX):
        name = name[len(NODE_TYPE_PREFIX):]
    return name.lower()


def category_to_enum(category: Optional[str]) -> int:
    """Map a node ``CATEGORY`` string to ``NodeCategory``."""
    if not category:
        return nodes_pb2.NODE_CATEGORY_UNSPECIFIED
    try:
        return nodes_pb2.NodeCategory.Value(NODE_CATEGORY_PREFIX + category.upper())
    except ValueError:
        return nodes_pb2.NODE_CATEGORY_UNSPECIFIED


def enum_to_category(value: int) -> str:
    """Map a ``NodeCategory`` enum value back to a lowercase category string."""
    if not value:
        return ""
    try:
        name = nodes_pb2.NodeCategory.Name(value)
    except ValueError:
        return ""
    if name.startswith(NODE_CATEGORY_PREFIX):
        name = name[len(NODE_CATEGORY_PREFIX):]
    return name.lower()


def execution_state_name(value: int) -> str:
    """
    Map ``ExecutionState`` to the short names the Python client exposes
    (``"COMPLETED"`` rather than ``"EXECUTION_STATE_COMPLETED"``).
    """
    try:
        name = common_pb2.ExecutionState.Name(value)
    except ValueError:
        return "UNKNOWN"
    if name.startswith(EXECUTION_STATE_PREFIX):
        name = name[len(EXECUTION_STATE_PREFIX):]
    return name or "UNKNOWN"


def execution_state_value(name: str) -> int:
    """Inverse of :func:`execution_state_name`."""
    if not name:
        return common_pb2.EXECUTION_STATE_UNSPECIFIED
    candidate = name if name.startswith(EXECUTION_STATE_PREFIX) else EXECUTION_STATE_PREFIX + name
    try:
        return common_pb2.ExecutionState.Value(candidate.upper())
    except ValueError:
        return common_pb2.EXECUTION_STATE_UNSPECIFIED


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def now_timestamp() -> timestamp_pb2.Timestamp:
    ts = timestamp_pb2.Timestamp()
    ts.GetCurrentTime()
    return ts


def make_request_metadata(
    request_id: Optional[str] = None,
    correlation_id: str = "",
    client_version: str = "",
) -> common_pb2.RequestMetadata:
    """Build request metadata, generating a request id when not supplied."""
    return common_pb2.RequestMetadata(
        request_id=request_id or f"req_{uuid.uuid4().hex}",
        correlation_id=correlation_id,
        timestamp=now_timestamp(),
        client_version=client_version,
    )


def make_response_metadata(
    request_id: str = "",
    processing_time_ms: int = 0,
    server_version: str = "",
    correlation_id: str = "",
) -> common_pb2.ResponseMetadata:
    """Build response metadata for a servicer reply."""
    return common_pb2.ResponseMetadata(
        request_id=request_id,
        correlation_id=correlation_id,
        timestamp=now_timestamp(),
        processing_time_ms=processing_time_ms,
        server_version=server_version,
    )


def error_details(
    error_code: str,
    message: str,
    stack_trace: str = "",
    retryable: bool = False,
) -> common_pb2.ErrorDetails:
    return common_pb2.ErrorDetails(
        error_code=error_code,
        message=message,
        stack_trace=stack_trace,
        retryable=retryable,
    )


def progress(percent: int, message: str = "", stage: str = "") -> common_pb2.Progress:
    return common_pb2.Progress(
        percent=max(0, min(100, int(percent))),
        message=message,
        stage=stage,
        timestamp=now_timestamp(),
    )
