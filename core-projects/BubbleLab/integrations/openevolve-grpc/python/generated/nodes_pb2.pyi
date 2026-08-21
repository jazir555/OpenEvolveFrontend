import common_pb2 as _common_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class NodeType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NODE_TYPE_UNSPECIFIED: _ClassVar[NodeType]
    NODE_TYPE_DECOMPOSITION: _ClassVar[NodeType]
    NODE_TYPE_PROBLEM_ANALYZER: _ClassVar[NodeType]
    NODE_TYPE_COMPLEXITY_ANALYZER: _ClassVar[NodeType]
    NODE_TYPE_KNOWLEDGE_QUERY: _ClassVar[NodeType]
    NODE_TYPE_KNOWLEDGE_EXTRACTION: _ClassVar[NodeType]
    NODE_TYPE_KNOWLEDGE_INTEGRATION: _ClassVar[NodeType]
    NODE_TYPE_KNOWLEDGE_REASONING: _ClassVar[NodeType]
    NODE_TYPE_KNOWLEDGE_VALIDATION: _ClassVar[NodeType]
    NODE_TYPE_KNOWLEDGE_ENRICHMENT: _ClassVar[NodeType]
    NODE_TYPE_KNOWLEDGE_FEDERATION: _ClassVar[NodeType]
    NODE_TYPE_KNOWLEDGE_LEARNING: _ClassVar[NodeType]
    NODE_TYPE_KNOWLEDGE_EVOLUTION: _ClassVar[NodeType]
    NODE_TYPE_KNOWLEDGE_ALERTING: _ClassVar[NodeType]
    NODE_TYPE_KNOWLEDGE_ANALYTICS: _ClassVar[NodeType]
    NODE_TYPE_KNOWLEDGE_SUMMARIZATION: _ClassVar[NodeType]
    NODE_TYPE_KNOWLEDGE_IMPORT_EXPORT: _ClassVar[NodeType]
    NODE_TYPE_TEMPORAL_KNOWLEDGE: _ClassVar[NodeType]
    NODE_TYPE_ENTITY_PROFILE: _ClassVar[NodeType]
    NODE_TYPE_MATH_PROOF_CHECKING: _ClassVar[NodeType]
    NODE_TYPE_MATH_CONJECTURE: _ClassVar[NodeType]
    NODE_TYPE_MATH_COUNTEREXAMPLE: _ClassVar[NodeType]
    NODE_TYPE_MATH_EQUIVALENCE: _ClassVar[NodeType]
    NODE_TYPE_MATH_INDUCTION: _ClassVar[NodeType]
    NODE_TYPE_MATH_LIBRARY_SEARCH: _ClassVar[NodeType]
    NODE_TYPE_MATH_PROBLEM_CLASSIFICATION: _ClassVar[NodeType]
    NODE_TYPE_MATH_PROOF_COMPLETION: _ClassVar[NodeType]
    NODE_TYPE_MATH_PROOF_SIMPLIFICATION: _ClassVar[NodeType]
    NODE_TYPE_MATH_TACTIC_RECOMMENDATION: _ClassVar[NodeType]
    NODE_TYPE_LEAN_AUTOFORMALIZATION: _ClassVar[NodeType]
    NODE_TYPE_Z3_CONSTRAINT_SOLVING: _ClassVar[NodeType]
    NODE_TYPE_Z3_THEOREM_PROVING: _ClassVar[NodeType]
    NODE_TYPE_MATH_WORKFLOW_ORCHESTRATOR: _ClassVar[NodeType]
    NODE_TYPE_MATH_VERIFICATION_PIPELINE: _ClassVar[NodeType]
    NODE_TYPE_MATH_VERIFICATION_DASHBOARD: _ClassVar[NodeType]
    NODE_TYPE_GAUNTLET: _ClassVar[NodeType]
    NODE_TYPE_FORMAL_GAUNTLET: _ClassVar[NodeType]
    NODE_TYPE_QUALITY_ASSURANCE: _ClassVar[NodeType]
    NODE_TYPE_CONTRADICTION_DETECTION: _ClassVar[NodeType]
    NODE_TYPE_BIAS_DETECTION: _ClassVar[NodeType]
    NODE_TYPE_WORKFLOW_ORCHESTRATION: _ClassVar[NodeType]
    NODE_TYPE_SOLUTION: _ClassVar[NodeType]
    NODE_TYPE_SUBPROBLEM: _ClassVar[NodeType]
    NODE_TYPE_OUTPUT: _ClassVar[NodeType]
    NODE_TYPE_ASSEMBLY: _ClassVar[NodeType]
    NODE_TYPE_BACKUP_RECOVERY: _ClassVar[NodeType]
    NODE_TYPE_VERSION_CONTROL: _ClassVar[NodeType]
    NODE_TYPE_EXPLAINABILITY: _ClassVar[NodeType]
    NODE_TYPE_UNCERTAINTY_QUANTIFICATION: _ClassVar[NodeType]
    NODE_TYPE_PATTERN_MINING: _ClassVar[NodeType]
    NODE_TYPE_RECOMMENDATION_ENGINE: _ClassVar[NodeType]
    NODE_TYPE_CAUSAL_ANALYSIS: _ClassVar[NodeType]
    NODE_TYPE_CHANGE_DETECTION: _ClassVar[NodeType]
    NODE_TYPE_SECURITY_COMPLIANCE: _ClassVar[NodeType]
    NODE_TYPE_NATURAL_LANGUAGE_INTERFACE: _ClassVar[NodeType]
    NODE_TYPE_STREAMING_INGESTION: _ClassVar[NodeType]
    NODE_TYPE_DEDUPLICATION: _ClassVar[NodeType]
    NODE_TYPE_SEMANTIC_SEARCH: _ClassVar[NodeType]

class NodeCategory(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NODE_CATEGORY_UNSPECIFIED: _ClassVar[NodeCategory]
    NODE_CATEGORY_ANALYSIS: _ClassVar[NodeCategory]
    NODE_CATEGORY_KNOWLEDGE: _ClassVar[NodeCategory]
    NODE_CATEGORY_MATH: _ClassVar[NodeCategory]
    NODE_CATEGORY_VERIFICATION: _ClassVar[NodeCategory]
    NODE_CATEGORY_QUALITY: _ClassVar[NodeCategory]
    NODE_CATEGORY_WORKFLOW: _ClassVar[NodeCategory]
    NODE_CATEGORY_UTILITY: _ClassVar[NodeCategory]
NODE_TYPE_UNSPECIFIED: NodeType
NODE_TYPE_DECOMPOSITION: NodeType
NODE_TYPE_PROBLEM_ANALYZER: NodeType
NODE_TYPE_COMPLEXITY_ANALYZER: NodeType
NODE_TYPE_KNOWLEDGE_QUERY: NodeType
NODE_TYPE_KNOWLEDGE_EXTRACTION: NodeType
NODE_TYPE_KNOWLEDGE_INTEGRATION: NodeType
NODE_TYPE_KNOWLEDGE_REASONING: NodeType
NODE_TYPE_KNOWLEDGE_VALIDATION: NodeType
NODE_TYPE_KNOWLEDGE_ENRICHMENT: NodeType
NODE_TYPE_KNOWLEDGE_FEDERATION: NodeType
NODE_TYPE_KNOWLEDGE_LEARNING: NodeType
NODE_TYPE_KNOWLEDGE_EVOLUTION: NodeType
NODE_TYPE_KNOWLEDGE_ALERTING: NodeType
NODE_TYPE_KNOWLEDGE_ANALYTICS: NodeType
NODE_TYPE_KNOWLEDGE_SUMMARIZATION: NodeType
NODE_TYPE_KNOWLEDGE_IMPORT_EXPORT: NodeType
NODE_TYPE_TEMPORAL_KNOWLEDGE: NodeType
NODE_TYPE_ENTITY_PROFILE: NodeType
NODE_TYPE_MATH_PROOF_CHECKING: NodeType
NODE_TYPE_MATH_CONJECTURE: NodeType
NODE_TYPE_MATH_COUNTEREXAMPLE: NodeType
NODE_TYPE_MATH_EQUIVALENCE: NodeType
NODE_TYPE_MATH_INDUCTION: NodeType
NODE_TYPE_MATH_LIBRARY_SEARCH: NodeType
NODE_TYPE_MATH_PROBLEM_CLASSIFICATION: NodeType
NODE_TYPE_MATH_PROOF_COMPLETION: NodeType
NODE_TYPE_MATH_PROOF_SIMPLIFICATION: NodeType
NODE_TYPE_MATH_TACTIC_RECOMMENDATION: NodeType
NODE_TYPE_LEAN_AUTOFORMALIZATION: NodeType
NODE_TYPE_Z3_CONSTRAINT_SOLVING: NodeType
NODE_TYPE_Z3_THEOREM_PROVING: NodeType
NODE_TYPE_MATH_WORKFLOW_ORCHESTRATOR: NodeType
NODE_TYPE_MATH_VERIFICATION_PIPELINE: NodeType
NODE_TYPE_MATH_VERIFICATION_DASHBOARD: NodeType
NODE_TYPE_GAUNTLET: NodeType
NODE_TYPE_FORMAL_GAUNTLET: NodeType
NODE_TYPE_QUALITY_ASSURANCE: NodeType
NODE_TYPE_CONTRADICTION_DETECTION: NodeType
NODE_TYPE_BIAS_DETECTION: NodeType
NODE_TYPE_WORKFLOW_ORCHESTRATION: NodeType
NODE_TYPE_SOLUTION: NodeType
NODE_TYPE_SUBPROBLEM: NodeType
NODE_TYPE_OUTPUT: NodeType
NODE_TYPE_ASSEMBLY: NodeType
NODE_TYPE_BACKUP_RECOVERY: NodeType
NODE_TYPE_VERSION_CONTROL: NodeType
NODE_TYPE_EXPLAINABILITY: NodeType
NODE_TYPE_UNCERTAINTY_QUANTIFICATION: NodeType
NODE_TYPE_PATTERN_MINING: NodeType
NODE_TYPE_RECOMMENDATION_ENGINE: NodeType
NODE_TYPE_CAUSAL_ANALYSIS: NodeType
NODE_TYPE_CHANGE_DETECTION: NodeType
NODE_TYPE_SECURITY_COMPLIANCE: NodeType
NODE_TYPE_NATURAL_LANGUAGE_INTERFACE: NodeType
NODE_TYPE_STREAMING_INGESTION: NodeType
NODE_TYPE_DEDUPLICATION: NodeType
NODE_TYPE_SEMANTIC_SEARCH: NodeType
NODE_CATEGORY_UNSPECIFIED: NodeCategory
NODE_CATEGORY_ANALYSIS: NodeCategory
NODE_CATEGORY_KNOWLEDGE: NodeCategory
NODE_CATEGORY_MATH: NodeCategory
NODE_CATEGORY_VERIFICATION: NodeCategory
NODE_CATEGORY_QUALITY: NodeCategory
NODE_CATEGORY_WORKFLOW: NodeCategory
NODE_CATEGORY_UTILITY: NodeCategory

class NodeCapabilities(_message.Message):
    __slots__ = ("supports_streaming", "supports_cancellation", "supports_progress", "supports_checkpointing", "supports_parallel_execution", "max_timeout_seconds", "required_resources")
    SUPPORTS_STREAMING_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_CANCELLATION_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_PROGRESS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_CHECKPOINTING_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_PARALLEL_EXECUTION_FIELD_NUMBER: _ClassVar[int]
    MAX_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_RESOURCES_FIELD_NUMBER: _ClassVar[int]
    supports_streaming: bool
    supports_cancellation: bool
    supports_progress: bool
    supports_checkpointing: bool
    supports_parallel_execution: bool
    max_timeout_seconds: int
    required_resources: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, supports_streaming: bool = ..., supports_cancellation: bool = ..., supports_progress: bool = ..., supports_checkpointing: bool = ..., supports_parallel_execution: bool = ..., max_timeout_seconds: _Optional[int] = ..., required_resources: _Optional[_Iterable[str]] = ...) -> None: ...

class NodeInfo(_message.Message):
    __slots__ = ("node_id", "node_type", "category", "display_name", "description", "icon", "version", "tags", "capabilities", "parameter_schema", "input_schema", "output_schema", "example_inputs", "documentation_urls")
    class DocumentationUrlsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_TYPE_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    ICON_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    INPUT_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    EXAMPLE_INPUTS_FIELD_NUMBER: _ClassVar[int]
    DOCUMENTATION_URLS_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    node_type: NodeType
    category: NodeCategory
    display_name: str
    description: str
    icon: str
    version: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    capabilities: NodeCapabilities
    parameter_schema: _struct_pb2.Struct
    input_schema: _struct_pb2.Struct
    output_schema: _struct_pb2.Struct
    example_inputs: _containers.RepeatedScalarFieldContainer[str]
    documentation_urls: _containers.ScalarMap[str, str]
    def __init__(self, node_id: _Optional[str] = ..., node_type: _Optional[_Union[NodeType, str]] = ..., category: _Optional[_Union[NodeCategory, str]] = ..., display_name: _Optional[str] = ..., description: _Optional[str] = ..., icon: _Optional[str] = ..., version: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., capabilities: _Optional[_Union[NodeCapabilities, _Mapping]] = ..., parameter_schema: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., input_schema: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., output_schema: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., example_inputs: _Optional[_Iterable[str]] = ..., documentation_urls: _Optional[_Mapping[str, str]] = ...) -> None: ...

class NodeExecutionRequest(_message.Message):
    __slots__ = ("metadata", "node_id", "node_type", "config", "inputs", "options")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_TYPE_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    metadata: _common_pb2.RequestMetadata
    node_id: str
    node_type: NodeType
    config: _struct_pb2.Struct
    inputs: _struct_pb2.Struct
    options: ExecutionOptions
    def __init__(self, metadata: _Optional[_Union[_common_pb2.RequestMetadata, _Mapping]] = ..., node_id: _Optional[str] = ..., node_type: _Optional[_Union[NodeType, str]] = ..., config: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., inputs: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., options: _Optional[_Union[ExecutionOptions, _Mapping]] = ...) -> None: ...

class ExecutionOptions(_message.Message):
    __slots__ = ("timeout_seconds", "enable_streaming", "enable_checkpointing", "checkpoint_id", "max_retries", "execution_priority", "labels", "resource_limits")
    class LabelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    ENABLE_STREAMING_FIELD_NUMBER: _ClassVar[int]
    ENABLE_CHECKPOINTING_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_ID_FIELD_NUMBER: _ClassVar[int]
    MAX_RETRIES_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_PRIORITY_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_LIMITS_FIELD_NUMBER: _ClassVar[int]
    timeout_seconds: int
    enable_streaming: bool
    enable_checkpointing: bool
    checkpoint_id: str
    max_retries: int
    execution_priority: str
    labels: _containers.ScalarMap[str, str]
    resource_limits: _struct_pb2.Struct
    def __init__(self, timeout_seconds: _Optional[int] = ..., enable_streaming: bool = ..., enable_checkpointing: bool = ..., checkpoint_id: _Optional[str] = ..., max_retries: _Optional[int] = ..., execution_priority: _Optional[str] = ..., labels: _Optional[_Mapping[str, str]] = ..., resource_limits: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class NodeExecutionResponse(_message.Message):
    __slots__ = ("metadata", "execution_id", "state", "result", "error", "final_progress", "checkpoint_id", "execution_metrics")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    FINAL_PROGRESS_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_METRICS_FIELD_NUMBER: _ClassVar[int]
    metadata: _common_pb2.ResponseMetadata
    execution_id: str
    state: _common_pb2.ExecutionState
    result: _struct_pb2.Struct
    error: _common_pb2.ErrorDetails
    final_progress: _common_pb2.Progress
    checkpoint_id: str
    execution_metrics: _struct_pb2.Struct
    def __init__(self, metadata: _Optional[_Union[_common_pb2.ResponseMetadata, _Mapping]] = ..., execution_id: _Optional[str] = ..., state: _Optional[_Union[_common_pb2.ExecutionState, str]] = ..., result: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., error: _Optional[_Union[_common_pb2.ErrorDetails, _Mapping]] = ..., final_progress: _Optional[_Union[_common_pb2.Progress, _Mapping]] = ..., checkpoint_id: _Optional[str] = ..., execution_metrics: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class ExecutionUpdate(_message.Message):
    __slots__ = ("execution_id", "state", "progress", "partial_result", "error", "metrics")
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    PARTIAL_RESULT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    execution_id: str
    state: _common_pb2.ExecutionState
    progress: _common_pb2.Progress
    partial_result: _struct_pb2.Struct
    error: _common_pb2.ErrorDetails
    metrics: _struct_pb2.Struct
    def __init__(self, execution_id: _Optional[str] = ..., state: _Optional[_Union[_common_pb2.ExecutionState, str]] = ..., progress: _Optional[_Union[_common_pb2.Progress, _Mapping]] = ..., partial_result: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., error: _Optional[_Union[_common_pb2.ErrorDetails, _Mapping]] = ..., metrics: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class CancelExecutionRequest(_message.Message):
    __slots__ = ("metadata", "execution_id", "reason")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    metadata: _common_pb2.RequestMetadata
    execution_id: str
    reason: str
    def __init__(self, metadata: _Optional[_Union[_common_pb2.RequestMetadata, _Mapping]] = ..., execution_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class CancelExecutionResponse(_message.Message):
    __slots__ = ("metadata", "success", "message", "final_state")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    FINAL_STATE_FIELD_NUMBER: _ClassVar[int]
    metadata: _common_pb2.ResponseMetadata
    success: bool
    message: str
    final_state: _common_pb2.ExecutionState
    def __init__(self, metadata: _Optional[_Union[_common_pb2.ResponseMetadata, _Mapping]] = ..., success: bool = ..., message: _Optional[str] = ..., final_state: _Optional[_Union[_common_pb2.ExecutionState, str]] = ...) -> None: ...

class GetExecutionStatusRequest(_message.Message):
    __slots__ = ("metadata", "execution_id")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    metadata: _common_pb2.RequestMetadata
    execution_id: str
    def __init__(self, metadata: _Optional[_Union[_common_pb2.RequestMetadata, _Mapping]] = ..., execution_id: _Optional[str] = ...) -> None: ...

class GetExecutionStatusResponse(_message.Message):
    __slots__ = ("metadata", "execution_id", "state", "current_progress", "result", "error", "started_at", "completed_at", "elapsed_seconds")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_PROGRESS_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_AT_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_SECONDS_FIELD_NUMBER: _ClassVar[int]
    metadata: _common_pb2.ResponseMetadata
    execution_id: str
    state: _common_pb2.ExecutionState
    current_progress: _common_pb2.Progress
    result: _struct_pb2.Struct
    error: _common_pb2.ErrorDetails
    started_at: _timestamp_pb2.Timestamp
    completed_at: _timestamp_pb2.Timestamp
    elapsed_seconds: int
    def __init__(self, metadata: _Optional[_Union[_common_pb2.ResponseMetadata, _Mapping]] = ..., execution_id: _Optional[str] = ..., state: _Optional[_Union[_common_pb2.ExecutionState, str]] = ..., current_progress: _Optional[_Union[_common_pb2.Progress, _Mapping]] = ..., result: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., error: _Optional[_Union[_common_pb2.ErrorDetails, _Mapping]] = ..., started_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., completed_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., elapsed_seconds: _Optional[int] = ...) -> None: ...

class ListNodesRequest(_message.Message):
    __slots__ = ("metadata", "category", "search_query", "tags", "pagination")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    SEARCH_QUERY_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    PAGINATION_FIELD_NUMBER: _ClassVar[int]
    metadata: _common_pb2.RequestMetadata
    category: NodeCategory
    search_query: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    pagination: _common_pb2.Pagination
    def __init__(self, metadata: _Optional[_Union[_common_pb2.RequestMetadata, _Mapping]] = ..., category: _Optional[_Union[NodeCategory, str]] = ..., search_query: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., pagination: _Optional[_Union[_common_pb2.Pagination, _Mapping]] = ...) -> None: ...

class ListNodesResponse(_message.Message):
    __slots__ = ("metadata", "nodes", "pagination")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    NODES_FIELD_NUMBER: _ClassVar[int]
    PAGINATION_FIELD_NUMBER: _ClassVar[int]
    metadata: _common_pb2.ResponseMetadata
    nodes: _containers.RepeatedCompositeFieldContainer[NodeInfo]
    pagination: _common_pb2.PaginationInfo
    def __init__(self, metadata: _Optional[_Union[_common_pb2.ResponseMetadata, _Mapping]] = ..., nodes: _Optional[_Iterable[_Union[NodeInfo, _Mapping]]] = ..., pagination: _Optional[_Union[_common_pb2.PaginationInfo, _Mapping]] = ...) -> None: ...

class GetNodeSchemaRequest(_message.Message):
    __slots__ = ("metadata", "node_type")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    NODE_TYPE_FIELD_NUMBER: _ClassVar[int]
    metadata: _common_pb2.RequestMetadata
    node_type: NodeType
    def __init__(self, metadata: _Optional[_Union[_common_pb2.RequestMetadata, _Mapping]] = ..., node_type: _Optional[_Union[NodeType, str]] = ...) -> None: ...

class GetNodeSchemaResponse(_message.Message):
    __slots__ = ("metadata", "node_info")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    NODE_INFO_FIELD_NUMBER: _ClassVar[int]
    metadata: _common_pb2.ResponseMetadata
    node_info: NodeInfo
    def __init__(self, metadata: _Optional[_Union[_common_pb2.ResponseMetadata, _Mapping]] = ..., node_info: _Optional[_Union[NodeInfo, _Mapping]] = ...) -> None: ...

class BatchExecutionRequest(_message.Message):
    __slots__ = ("metadata", "requests", "parallel", "max_concurrency")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    REQUESTS_FIELD_NUMBER: _ClassVar[int]
    PARALLEL_FIELD_NUMBER: _ClassVar[int]
    MAX_CONCURRENCY_FIELD_NUMBER: _ClassVar[int]
    metadata: _common_pb2.RequestMetadata
    requests: _containers.RepeatedCompositeFieldContainer[NodeExecutionRequest]
    parallel: bool
    max_concurrency: int
    def __init__(self, metadata: _Optional[_Union[_common_pb2.RequestMetadata, _Mapping]] = ..., requests: _Optional[_Iterable[_Union[NodeExecutionRequest, _Mapping]]] = ..., parallel: bool = ..., max_concurrency: _Optional[int] = ...) -> None: ...

class BatchExecutionResponse(_message.Message):
    __slots__ = ("metadata", "responses", "succeeded", "failed", "total")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    RESPONSES_FIELD_NUMBER: _ClassVar[int]
    SUCCEEDED_FIELD_NUMBER: _ClassVar[int]
    FAILED_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    metadata: _common_pb2.ResponseMetadata
    responses: _containers.RepeatedCompositeFieldContainer[NodeExecutionResponse]
    succeeded: int
    failed: int
    total: int
    def __init__(self, metadata: _Optional[_Union[_common_pb2.ResponseMetadata, _Mapping]] = ..., responses: _Optional[_Iterable[_Union[NodeExecutionResponse, _Mapping]]] = ..., succeeded: _Optional[int] = ..., failed: _Optional[int] = ..., total: _Optional[int] = ...) -> None: ...
