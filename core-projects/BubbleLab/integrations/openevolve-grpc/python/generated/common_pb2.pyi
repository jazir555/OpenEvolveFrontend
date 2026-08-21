from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import any_pb2 as _any_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HealthStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    HEALTH_STATUS_UNSPECIFIED: _ClassVar[HealthStatus]
    HEALTH_STATUS_HEALTHY: _ClassVar[HealthStatus]
    HEALTH_STATUS_DEGRADED: _ClassVar[HealthStatus]
    HEALTH_STATUS_UNHEALTHY: _ClassVar[HealthStatus]
    HEALTH_STATUS_UNKNOWN: _ClassVar[HealthStatus]

class ExecutionState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EXECUTION_STATE_UNSPECIFIED: _ClassVar[ExecutionState]
    EXECUTION_STATE_PENDING: _ClassVar[ExecutionState]
    EXECUTION_STATE_RUNNING: _ClassVar[ExecutionState]
    EXECUTION_STATE_PAUSED: _ClassVar[ExecutionState]
    EXECUTION_STATE_COMPLETED: _ClassVar[ExecutionState]
    EXECUTION_STATE_FAILED: _ClassVar[ExecutionState]
    EXECUTION_STATE_CANCELLED: _ClassVar[ExecutionState]
    EXECUTION_STATE_TIMEOUT: _ClassVar[ExecutionState]
HEALTH_STATUS_UNSPECIFIED: HealthStatus
HEALTH_STATUS_HEALTHY: HealthStatus
HEALTH_STATUS_DEGRADED: HealthStatus
HEALTH_STATUS_UNHEALTHY: HealthStatus
HEALTH_STATUS_UNKNOWN: HealthStatus
EXECUTION_STATE_UNSPECIFIED: ExecutionState
EXECUTION_STATE_PENDING: ExecutionState
EXECUTION_STATE_RUNNING: ExecutionState
EXECUTION_STATE_PAUSED: ExecutionState
EXECUTION_STATE_COMPLETED: ExecutionState
EXECUTION_STATE_FAILED: ExecutionState
EXECUTION_STATE_CANCELLED: ExecutionState
EXECUTION_STATE_TIMEOUT: ExecutionState

class RequestMetadata(_message.Message):
    __slots__ = ("request_id", "correlation_id", "timestamp", "client_version", "headers")
    class HeadersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    CORRELATION_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    CLIENT_VERSION_FIELD_NUMBER: _ClassVar[int]
    HEADERS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    correlation_id: str
    timestamp: _timestamp_pb2.Timestamp
    client_version: str
    headers: _containers.ScalarMap[str, str]
    def __init__(self, request_id: _Optional[str] = ..., correlation_id: _Optional[str] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., client_version: _Optional[str] = ..., headers: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ResponseMetadata(_message.Message):
    __slots__ = ("request_id", "correlation_id", "timestamp", "processing_time_ms", "server_version")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    CORRELATION_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    SERVER_VERSION_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    correlation_id: str
    timestamp: _timestamp_pb2.Timestamp
    processing_time_ms: int
    server_version: str
    def __init__(self, request_id: _Optional[str] = ..., correlation_id: _Optional[str] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., processing_time_ms: _Optional[int] = ..., server_version: _Optional[str] = ...) -> None: ...

class Pagination(_message.Message):
    __slots__ = ("page", "page_size", "cursor")
    PAGE_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    CURSOR_FIELD_NUMBER: _ClassVar[int]
    page: int
    page_size: int
    cursor: str
    def __init__(self, page: _Optional[int] = ..., page_size: _Optional[int] = ..., cursor: _Optional[str] = ...) -> None: ...

class PaginationInfo(_message.Message):
    __slots__ = ("total_count", "page", "page_size", "has_next", "next_cursor")
    TOTAL_COUNT_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    HAS_NEXT_FIELD_NUMBER: _ClassVar[int]
    NEXT_CURSOR_FIELD_NUMBER: _ClassVar[int]
    total_count: int
    page: int
    page_size: int
    has_next: bool
    next_cursor: str
    def __init__(self, total_count: _Optional[int] = ..., page: _Optional[int] = ..., page_size: _Optional[int] = ..., has_next: bool = ..., next_cursor: _Optional[str] = ...) -> None: ...

class ServiceHealth(_message.Message):
    __slots__ = ("service_name", "status", "message", "last_check", "response_time_ms", "metrics")
    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    SERVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    LAST_CHECK_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    service_name: str
    status: HealthStatus
    message: str
    last_check: _timestamp_pb2.Timestamp
    response_time_ms: int
    metrics: _containers.MessageMap[str, _struct_pb2.Value]
    def __init__(self, service_name: _Optional[str] = ..., status: _Optional[_Union[HealthStatus, str]] = ..., message: _Optional[str] = ..., last_check: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., response_time_ms: _Optional[int] = ..., metrics: _Optional[_Mapping[str, _struct_pb2.Value]] = ...) -> None: ...

class Progress(_message.Message):
    __slots__ = ("percent", "message", "stage", "timestamp", "metrics")
    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    PERCENT_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    percent: int
    message: str
    stage: str
    timestamp: _timestamp_pb2.Timestamp
    metrics: _containers.MessageMap[str, _struct_pb2.Value]
    def __init__(self, percent: _Optional[int] = ..., message: _Optional[str] = ..., stage: _Optional[str] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., metrics: _Optional[_Mapping[str, _struct_pb2.Value]] = ...) -> None: ...

class ErrorDetails(_message.Message):
    __slots__ = ("error_code", "message", "stack_trace", "context", "retryable", "retry_after_seconds")
    class ContextEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    STACK_TRACE_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    RETRYABLE_FIELD_NUMBER: _ClassVar[int]
    RETRY_AFTER_SECONDS_FIELD_NUMBER: _ClassVar[int]
    error_code: str
    message: str
    stack_trace: str
    context: _containers.MessageMap[str, _struct_pb2.Value]
    retryable: bool
    retry_after_seconds: int
    def __init__(self, error_code: _Optional[str] = ..., message: _Optional[str] = ..., stack_trace: _Optional[str] = ..., context: _Optional[_Mapping[str, _struct_pb2.Value]] = ..., retryable: bool = ..., retry_after_seconds: _Optional[int] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class VersionInfo(_message.Message):
    __slots__ = ("version", "commit_hash", "build_date", "dependencies")
    class DependenciesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    VERSION_FIELD_NUMBER: _ClassVar[int]
    COMMIT_HASH_FIELD_NUMBER: _ClassVar[int]
    BUILD_DATE_FIELD_NUMBER: _ClassVar[int]
    DEPENDENCIES_FIELD_NUMBER: _ClassVar[int]
    version: str
    commit_hash: str
    build_date: str
    dependencies: _containers.ScalarMap[str, str]
    def __init__(self, version: _Optional[str] = ..., commit_hash: _Optional[str] = ..., build_date: _Optional[str] = ..., dependencies: _Optional[_Mapping[str, str]] = ...) -> None: ...
