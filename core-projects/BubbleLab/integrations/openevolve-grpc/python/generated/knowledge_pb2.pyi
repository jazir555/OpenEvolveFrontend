import common_pb2 as _common_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DocumentType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DOCUMENT_TYPE_UNSPECIFIED: _ClassVar[DocumentType]
    DOCUMENT_TYPE_TEXT: _ClassVar[DocumentType]
    DOCUMENT_TYPE_MARKDOWN: _ClassVar[DocumentType]
    DOCUMENT_TYPE_JSON: _ClassVar[DocumentType]
    DOCUMENT_TYPE_CODE: _ClassVar[DocumentType]
    DOCUMENT_TYPE_PDF: _ClassVar[DocumentType]
    DOCUMENT_TYPE_WEBPAGE: _ClassVar[DocumentType]
    DOCUMENT_TYPE_DATABASE_RECORD: _ClassVar[DocumentType]
DOCUMENT_TYPE_UNSPECIFIED: DocumentType
DOCUMENT_TYPE_TEXT: DocumentType
DOCUMENT_TYPE_MARKDOWN: DocumentType
DOCUMENT_TYPE_JSON: DocumentType
DOCUMENT_TYPE_CODE: DocumentType
DOCUMENT_TYPE_PDF: DocumentType
DOCUMENT_TYPE_WEBPAGE: DocumentType
DOCUMENT_TYPE_DATABASE_RECORD: DocumentType

class KnowledgeDocument(_message.Message):
    __slots__ = ("id", "title", "content", "document_type", "tags", "metadata", "source", "author", "created_at", "updated_at", "relevance_score", "embedding")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    AUTHOR_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    RELEVANCE_SCORE_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    id: str
    title: str
    content: str
    document_type: DocumentType
    tags: _containers.RepeatedScalarFieldContainer[str]
    metadata: _containers.ScalarMap[str, str]
    source: str
    author: str
    created_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    relevance_score: float
    embedding: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, id: _Optional[str] = ..., title: _Optional[str] = ..., content: _Optional[str] = ..., document_type: _Optional[_Union[DocumentType, str]] = ..., tags: _Optional[_Iterable[str]] = ..., metadata: _Optional[_Mapping[str, str]] = ..., source: _Optional[str] = ..., author: _Optional[str] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., relevance_score: _Optional[float] = ..., embedding: _Optional[_Iterable[float]] = ...) -> None: ...

class KnowledgeQuery(_message.Message):
    __slots__ = ("query_text", "filters", "limit", "min_relevance", "search_type", "parameters")
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    QUERY_TEXT_FIELD_NUMBER: _ClassVar[int]
    FILTERS_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    MIN_RELEVANCE_FIELD_NUMBER: _ClassVar[int]
    SEARCH_TYPE_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    query_text: str
    filters: _containers.RepeatedScalarFieldContainer[str]
    limit: int
    min_relevance: float
    search_type: str
    parameters: _containers.MessageMap[str, _struct_pb2.Value]
    def __init__(self, query_text: _Optional[str] = ..., filters: _Optional[_Iterable[str]] = ..., limit: _Optional[int] = ..., min_relevance: _Optional[float] = ..., search_type: _Optional[str] = ..., parameters: _Optional[_Mapping[str, _struct_pb2.Value]] = ...) -> None: ...

class KnowledgeQueryResult(_message.Message):
    __slots__ = ("documents", "total_count", "query_time_ms", "facets")
    class FacetsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_COUNT_FIELD_NUMBER: _ClassVar[int]
    QUERY_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    FACETS_FIELD_NUMBER: _ClassVar[int]
    documents: _containers.RepeatedCompositeFieldContainer[KnowledgeDocument]
    total_count: int
    query_time_ms: float
    facets: _containers.MessageMap[str, _struct_pb2.Value]
    def __init__(self, documents: _Optional[_Iterable[_Union[KnowledgeDocument, _Mapping]]] = ..., total_count: _Optional[int] = ..., query_time_ms: _Optional[float] = ..., facets: _Optional[_Mapping[str, _struct_pb2.Value]] = ...) -> None: ...

class KnowledgeExtractionRequest(_message.Message):
    __slots__ = ("source_content", "source_type", "entity_types", "relation_types", "extract_hierarchy", "options")
    SOURCE_CONTENT_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    ENTITY_TYPES_FIELD_NUMBER: _ClassVar[int]
    RELATION_TYPES_FIELD_NUMBER: _ClassVar[int]
    EXTRACT_HIERARCHY_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    source_content: str
    source_type: str
    entity_types: _containers.RepeatedScalarFieldContainer[str]
    relation_types: _containers.RepeatedScalarFieldContainer[str]
    extract_hierarchy: bool
    options: _struct_pb2.Struct
    def __init__(self, source_content: _Optional[str] = ..., source_type: _Optional[str] = ..., entity_types: _Optional[_Iterable[str]] = ..., relation_types: _Optional[_Iterable[str]] = ..., extract_hierarchy: bool = ..., options: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class KnowledgeExtractionResult(_message.Message):
    __slots__ = ("entities", "relations", "hierarchy", "extracted_document", "statistics")
    ENTITIES_FIELD_NUMBER: _ClassVar[int]
    RELATIONS_FIELD_NUMBER: _ClassVar[int]
    HIERARCHY_FIELD_NUMBER: _ClassVar[int]
    EXTRACTED_DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    STATISTICS_FIELD_NUMBER: _ClassVar[int]
    entities: _containers.RepeatedCompositeFieldContainer[ExtractedEntity]
    relations: _containers.RepeatedCompositeFieldContainer[ExtractedRelation]
    hierarchy: ExtractedHierarchy
    extracted_document: KnowledgeDocument
    statistics: _struct_pb2.Struct
    def __init__(self, entities: _Optional[_Iterable[_Union[ExtractedEntity, _Mapping]]] = ..., relations: _Optional[_Iterable[_Union[ExtractedRelation, _Mapping]]] = ..., hierarchy: _Optional[_Union[ExtractedHierarchy, _Mapping]] = ..., extracted_document: _Optional[_Union[KnowledgeDocument, _Mapping]] = ..., statistics: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class ExtractedEntity(_message.Message):
    __slots__ = ("id", "name", "entity_type", "aliases", "attributes", "confidence", "source_locations")
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    ALIASES_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_LOCATIONS_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    entity_type: str
    aliases: _containers.RepeatedScalarFieldContainer[str]
    attributes: _containers.MessageMap[str, _struct_pb2.Value]
    confidence: float
    source_locations: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., entity_type: _Optional[str] = ..., aliases: _Optional[_Iterable[str]] = ..., attributes: _Optional[_Mapping[str, _struct_pb2.Value]] = ..., confidence: _Optional[float] = ..., source_locations: _Optional[_Iterable[str]] = ...) -> None: ...

class ExtractedRelation(_message.Message):
    __slots__ = ("id", "source_entity_id", "target_entity_id", "relation_type", "attributes", "confidence")
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ENTITY_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_ENTITY_ID_FIELD_NUMBER: _ClassVar[int]
    RELATION_TYPE_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    attributes: _containers.MessageMap[str, _struct_pb2.Value]
    confidence: float
    def __init__(self, id: _Optional[str] = ..., source_entity_id: _Optional[str] = ..., target_entity_id: _Optional[str] = ..., relation_type: _Optional[str] = ..., attributes: _Optional[_Mapping[str, _struct_pb2.Value]] = ..., confidence: _Optional[float] = ...) -> None: ...

class StringList(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, values: _Optional[_Iterable[str]] = ...) -> None: ...

class ExtractedHierarchy(_message.Message):
    __slots__ = ("root_id", "parent_child_map", "max_depth")
    class ParentChildMapEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: StringList
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[StringList, _Mapping]] = ...) -> None: ...
    ROOT_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_CHILD_MAP_FIELD_NUMBER: _ClassVar[int]
    MAX_DEPTH_FIELD_NUMBER: _ClassVar[int]
    root_id: str
    parent_child_map: _containers.MessageMap[str, StringList]
    max_depth: int
    def __init__(self, root_id: _Optional[str] = ..., parent_child_map: _Optional[_Mapping[str, StringList]] = ..., max_depth: _Optional[int] = ...) -> None: ...

class KnowledgeReasoningRequest(_message.Message):
    __slots__ = ("query", "context_document_ids", "reasoning_type", "max_hops", "explain_reasoning")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_DOCUMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    REASONING_TYPE_FIELD_NUMBER: _ClassVar[int]
    MAX_HOPS_FIELD_NUMBER: _ClassVar[int]
    EXPLAIN_REASONING_FIELD_NUMBER: _ClassVar[int]
    query: str
    context_document_ids: _containers.RepeatedScalarFieldContainer[str]
    reasoning_type: str
    max_hops: int
    explain_reasoning: bool
    def __init__(self, query: _Optional[str] = ..., context_document_ids: _Optional[_Iterable[str]] = ..., reasoning_type: _Optional[str] = ..., max_hops: _Optional[int] = ..., explain_reasoning: bool = ...) -> None: ...

class KnowledgeReasoningResult(_message.Message):
    __slots__ = ("answer", "confidence", "steps", "supporting_documents", "is_complete")
    ANSWER_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    STEPS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTING_DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    IS_COMPLETE_FIELD_NUMBER: _ClassVar[int]
    answer: str
    confidence: float
    steps: _containers.RepeatedCompositeFieldContainer[ReasoningStep]
    supporting_documents: _containers.RepeatedScalarFieldContainer[str]
    is_complete: bool
    def __init__(self, answer: _Optional[str] = ..., confidence: _Optional[float] = ..., steps: _Optional[_Iterable[_Union[ReasoningStep, _Mapping]]] = ..., supporting_documents: _Optional[_Iterable[str]] = ..., is_complete: bool = ...) -> None: ...

class ReasoningStep(_message.Message):
    __slots__ = ("step_number", "description", "premise", "conclusion", "rule_applied", "confidence")
    STEP_NUMBER_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    PREMISE_FIELD_NUMBER: _ClassVar[int]
    CONCLUSION_FIELD_NUMBER: _ClassVar[int]
    RULE_APPLIED_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    step_number: int
    description: str
    premise: str
    conclusion: str
    rule_applied: str
    confidence: float
    def __init__(self, step_number: _Optional[int] = ..., description: _Optional[str] = ..., premise: _Optional[str] = ..., conclusion: _Optional[str] = ..., rule_applied: _Optional[str] = ..., confidence: _Optional[float] = ...) -> None: ...

class KnowledgeFederationRequest(_message.Message):
    __slots__ = ("source_endpoints", "query", "merge_strategy")
    SOURCE_ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    MERGE_STRATEGY_FIELD_NUMBER: _ClassVar[int]
    source_endpoints: _containers.RepeatedScalarFieldContainer[str]
    query: KnowledgeQuery
    merge_strategy: str
    def __init__(self, source_endpoints: _Optional[_Iterable[str]] = ..., query: _Optional[_Union[KnowledgeQuery, _Mapping]] = ..., merge_strategy: _Optional[str] = ...) -> None: ...

class KnowledgeFederationResult(_message.Message):
    __slots__ = ("source_results", "merged_results", "source_stats")
    class SourceStatsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    SOURCE_RESULTS_FIELD_NUMBER: _ClassVar[int]
    MERGED_RESULTS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_STATS_FIELD_NUMBER: _ClassVar[int]
    source_results: _containers.RepeatedCompositeFieldContainer[FederatedSourceResults]
    merged_results: KnowledgeQueryResult
    source_stats: _containers.MessageMap[str, _struct_pb2.Value]
    def __init__(self, source_results: _Optional[_Iterable[_Union[FederatedSourceResults, _Mapping]]] = ..., merged_results: _Optional[_Union[KnowledgeQueryResult, _Mapping]] = ..., source_stats: _Optional[_Mapping[str, _struct_pb2.Value]] = ...) -> None: ...

class FederatedSourceResults(_message.Message):
    __slots__ = ("source_id", "source_name", "results", "response_time_ms", "success", "error_message")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_NAME_FIELD_NUMBER: _ClassVar[int]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    source_name: str
    results: KnowledgeQueryResult
    response_time_ms: int
    success: bool
    error_message: str
    def __init__(self, source_id: _Optional[str] = ..., source_name: _Optional[str] = ..., results: _Optional[_Union[KnowledgeQueryResult, _Mapping]] = ..., response_time_ms: _Optional[int] = ..., success: bool = ..., error_message: _Optional[str] = ...) -> None: ...

class KnowledgeGraphNode(_message.Message):
    __slots__ = ("id", "label", "node_type", "properties")
    class PropertiesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    NODE_TYPE_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    id: str
    label: str
    node_type: str
    properties: _containers.MessageMap[str, _struct_pb2.Value]
    def __init__(self, id: _Optional[str] = ..., label: _Optional[str] = ..., node_type: _Optional[str] = ..., properties: _Optional[_Mapping[str, _struct_pb2.Value]] = ...) -> None: ...

class KnowledgeGraphEdge(_message.Message):
    __slots__ = ("id", "source_id", "target_id", "relation_type", "properties", "weight")
    class PropertiesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_ID_FIELD_NUMBER: _ClassVar[int]
    RELATION_TYPE_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    id: str
    source_id: str
    target_id: str
    relation_type: str
    properties: _containers.MessageMap[str, _struct_pb2.Value]
    weight: float
    def __init__(self, id: _Optional[str] = ..., source_id: _Optional[str] = ..., target_id: _Optional[str] = ..., relation_type: _Optional[str] = ..., properties: _Optional[_Mapping[str, _struct_pb2.Value]] = ..., weight: _Optional[float] = ...) -> None: ...

class KnowledgeGraph(_message.Message):
    __slots__ = ("nodes", "edges", "statistics")
    class StatisticsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    NODES_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    STATISTICS_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[KnowledgeGraphNode]
    edges: _containers.RepeatedCompositeFieldContainer[KnowledgeGraphEdge]
    statistics: _containers.MessageMap[str, _struct_pb2.Value]
    def __init__(self, nodes: _Optional[_Iterable[_Union[KnowledgeGraphNode, _Mapping]]] = ..., edges: _Optional[_Iterable[_Union[KnowledgeGraphEdge, _Mapping]]] = ..., statistics: _Optional[_Mapping[str, _struct_pb2.Value]] = ...) -> None: ...
