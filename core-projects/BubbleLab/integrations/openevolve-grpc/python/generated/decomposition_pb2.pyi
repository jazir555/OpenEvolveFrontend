import common_pb2 as _common_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DecompositionStrategy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DECOMPOSITION_STRATEGY_UNSPECIFIED: _ClassVar[DecompositionStrategy]
    DECOMPOSITION_STRATEGY_SEMANTIC: _ClassVar[DecompositionStrategy]
    DECOMPOSITION_STRATEGY_DEPENDENCY: _ClassVar[DecompositionStrategy]
    DECOMPOSITION_STRATEGY_COMPLEXITY: _ClassVar[DecompositionStrategy]
    DECOMPOSITION_STRATEGY_HYBRID: _ClassVar[DecompositionStrategy]
    DECOMPOSITION_STRATEGY_RESEARCH: _ClassVar[DecompositionStrategy]
    DECOMPOSITION_STRATEGY_FUNCTIONAL: _ClassVar[DecompositionStrategy]
    DECOMPOSITION_STRATEGY_TEMPORAL: _ClassVar[DecompositionStrategy]
    DECOMPOSITION_STRATEGY_RISK_BASED: _ClassVar[DecompositionStrategy]
    DECOMPOSITION_STRATEGY_VALUE_BASED: _ClassVar[DecompositionStrategy]
    DECOMPOSITION_STRATEGY_TECHNICAL_DEPENDENCY: _ClassVar[DecompositionStrategy]

class ProblemType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROBLEM_TYPE_UNSPECIFIED: _ClassVar[ProblemType]
    PROBLEM_TYPE_RESEARCH: _ClassVar[ProblemType]
    PROBLEM_TYPE_ENGINEERING: _ClassVar[ProblemType]
    PROBLEM_TYPE_ANALYTICAL: _ClassVar[ProblemType]
    PROBLEM_TYPE_CREATIVE: _ClassVar[ProblemType]
    PROBLEM_TYPE_OPTIMIZATION: _ClassVar[ProblemType]
    PROBLEM_TYPE_VERIFICATION: _ClassVar[ProblemType]
    PROBLEM_TYPE_IMPLEMENTATION: _ClassVar[ProblemType]
    PROBLEM_TYPE_EXPLORATORY: _ClassVar[ProblemType]
DECOMPOSITION_STRATEGY_UNSPECIFIED: DecompositionStrategy
DECOMPOSITION_STRATEGY_SEMANTIC: DecompositionStrategy
DECOMPOSITION_STRATEGY_DEPENDENCY: DecompositionStrategy
DECOMPOSITION_STRATEGY_COMPLEXITY: DecompositionStrategy
DECOMPOSITION_STRATEGY_HYBRID: DecompositionStrategy
DECOMPOSITION_STRATEGY_RESEARCH: DecompositionStrategy
DECOMPOSITION_STRATEGY_FUNCTIONAL: DecompositionStrategy
DECOMPOSITION_STRATEGY_TEMPORAL: DecompositionStrategy
DECOMPOSITION_STRATEGY_RISK_BASED: DecompositionStrategy
DECOMPOSITION_STRATEGY_VALUE_BASED: DecompositionStrategy
DECOMPOSITION_STRATEGY_TECHNICAL_DEPENDENCY: DecompositionStrategy
PROBLEM_TYPE_UNSPECIFIED: ProblemType
PROBLEM_TYPE_RESEARCH: ProblemType
PROBLEM_TYPE_ENGINEERING: ProblemType
PROBLEM_TYPE_ANALYTICAL: ProblemType
PROBLEM_TYPE_CREATIVE: ProblemType
PROBLEM_TYPE_OPTIMIZATION: ProblemType
PROBLEM_TYPE_VERIFICATION: ProblemType
PROBLEM_TYPE_IMPLEMENTATION: ProblemType
PROBLEM_TYPE_EXPLORATORY: ProblemType

class ComplexityScore(_message.Message):
    __slots__ = ("cognitive_load", "technical_difficulty", "domain_knowledge_required", "uncertainty_level", "overall_score")
    COGNITIVE_LOAD_FIELD_NUMBER: _ClassVar[int]
    TECHNICAL_DIFFICULTY_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_KNOWLEDGE_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    UNCERTAINTY_LEVEL_FIELD_NUMBER: _ClassVar[int]
    OVERALL_SCORE_FIELD_NUMBER: _ClassVar[int]
    cognitive_load: int
    technical_difficulty: int
    domain_knowledge_required: int
    uncertainty_level: int
    overall_score: int
    def __init__(self, cognitive_load: _Optional[int] = ..., technical_difficulty: _Optional[int] = ..., domain_knowledge_required: _Optional[int] = ..., uncertainty_level: _Optional[int] = ..., overall_score: _Optional[int] = ...) -> None: ...

class Constraint(_message.Message):
    __slots__ = ("id", "description", "constraint_type", "parameters")
    ID_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINT_TYPE_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    id: str
    description: str
    constraint_type: str
    parameters: _struct_pb2.Struct
    def __init__(self, id: _Optional[str] = ..., description: _Optional[str] = ..., constraint_type: _Optional[str] = ..., parameters: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class DomainContext(_message.Message):
    __slots__ = ("primary_domain", "related_domains", "key_terms", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    PRIMARY_DOMAIN_FIELD_NUMBER: _ClassVar[int]
    RELATED_DOMAINS_FIELD_NUMBER: _ClassVar[int]
    KEY_TERMS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    primary_domain: str
    related_domains: _containers.RepeatedScalarFieldContainer[str]
    key_terms: _containers.RepeatedScalarFieldContainer[str]
    metadata: _containers.MessageMap[str, _struct_pb2.Value]
    def __init__(self, primary_domain: _Optional[str] = ..., related_domains: _Optional[_Iterable[str]] = ..., key_terms: _Optional[_Iterable[str]] = ..., metadata: _Optional[_Mapping[str, _struct_pb2.Value]] = ...) -> None: ...

class ProblemDefinition(_message.Message):
    __slots__ = ("id", "title", "description", "problem_type", "complexity", "domain", "constraints", "success_criteria", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    PROBLEM_TYPE_FIELD_NUMBER: _ClassVar[int]
    COMPLEXITY_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_CRITERIA_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    id: str
    title: str
    description: str
    problem_type: ProblemType
    complexity: ComplexityScore
    domain: DomainContext
    constraints: _containers.RepeatedCompositeFieldContainer[Constraint]
    success_criteria: _containers.RepeatedScalarFieldContainer[str]
    metadata: _containers.MessageMap[str, _struct_pb2.Value]
    def __init__(self, id: _Optional[str] = ..., title: _Optional[str] = ..., description: _Optional[str] = ..., problem_type: _Optional[_Union[ProblemType, str]] = ..., complexity: _Optional[_Union[ComplexityScore, _Mapping]] = ..., domain: _Optional[_Union[DomainContext, _Mapping]] = ..., constraints: _Optional[_Iterable[_Union[Constraint, _Mapping]]] = ..., success_criteria: _Optional[_Iterable[str]] = ..., metadata: _Optional[_Mapping[str, _struct_pb2.Value]] = ...) -> None: ...

class SubProblem(_message.Message):
    __slots__ = ("id", "parent_id", "title", "description", "depth", "order_index", "problem_type", "complexity", "dependencies", "tags", "assigned_teams", "metadata", "estimated_effort", "priority", "status")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    DEPTH_FIELD_NUMBER: _ClassVar[int]
    ORDER_INDEX_FIELD_NUMBER: _ClassVar[int]
    PROBLEM_TYPE_FIELD_NUMBER: _ClassVar[int]
    COMPLEXITY_FIELD_NUMBER: _ClassVar[int]
    DEPENDENCIES_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    ASSIGNED_TEAMS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_EFFORT_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    id: str
    parent_id: str
    title: str
    description: str
    depth: int
    order_index: int
    problem_type: ProblemType
    complexity: ComplexityScore
    dependencies: _containers.RepeatedScalarFieldContainer[str]
    tags: _containers.RepeatedScalarFieldContainer[str]
    assigned_teams: _containers.RepeatedScalarFieldContainer[str]
    metadata: _containers.MessageMap[str, _struct_pb2.Value]
    estimated_effort: str
    priority: int
    status: str
    def __init__(self, id: _Optional[str] = ..., parent_id: _Optional[str] = ..., title: _Optional[str] = ..., description: _Optional[str] = ..., depth: _Optional[int] = ..., order_index: _Optional[int] = ..., problem_type: _Optional[_Union[ProblemType, str]] = ..., complexity: _Optional[_Union[ComplexityScore, _Mapping]] = ..., dependencies: _Optional[_Iterable[str]] = ..., tags: _Optional[_Iterable[str]] = ..., assigned_teams: _Optional[_Iterable[str]] = ..., metadata: _Optional[_Mapping[str, _struct_pb2.Value]] = ..., estimated_effort: _Optional[str] = ..., priority: _Optional[int] = ..., status: _Optional[str] = ...) -> None: ...

class QualityScores(_message.Message):
    __slots__ = ("coherence", "independence", "completeness", "granularity", "traceability", "overall_score", "improvement_suggestions")
    class ImprovementSuggestionsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    COHERENCE_FIELD_NUMBER: _ClassVar[int]
    INDEPENDENCE_FIELD_NUMBER: _ClassVar[int]
    COMPLETENESS_FIELD_NUMBER: _ClassVar[int]
    GRANULARITY_FIELD_NUMBER: _ClassVar[int]
    TRACEABILITY_FIELD_NUMBER: _ClassVar[int]
    OVERALL_SCORE_FIELD_NUMBER: _ClassVar[int]
    IMPROVEMENT_SUGGESTIONS_FIELD_NUMBER: _ClassVar[int]
    coherence: float
    independence: float
    completeness: float
    granularity: float
    traceability: float
    overall_score: float
    improvement_suggestions: _containers.ScalarMap[str, str]
    def __init__(self, coherence: _Optional[float] = ..., independence: _Optional[float] = ..., completeness: _Optional[float] = ..., granularity: _Optional[float] = ..., traceability: _Optional[float] = ..., overall_score: _Optional[float] = ..., improvement_suggestions: _Optional[_Mapping[str, str]] = ...) -> None: ...

class DecompositionResult(_message.Message):
    __slots__ = ("decomposition_id", "original_problem", "subproblems", "strategy_used", "quality", "execution_metrics", "summary")
    DECOMPOSITION_ID_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_PROBLEM_FIELD_NUMBER: _ClassVar[int]
    SUBPROBLEMS_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_USED_FIELD_NUMBER: _ClassVar[int]
    QUALITY_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_METRICS_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    decomposition_id: str
    original_problem: ProblemDefinition
    subproblems: _containers.RepeatedCompositeFieldContainer[SubProblem]
    strategy_used: DecompositionStrategy
    quality: QualityScores
    execution_metrics: _struct_pb2.Struct
    summary: str
    def __init__(self, decomposition_id: _Optional[str] = ..., original_problem: _Optional[_Union[ProblemDefinition, _Mapping]] = ..., subproblems: _Optional[_Iterable[_Union[SubProblem, _Mapping]]] = ..., strategy_used: _Optional[_Union[DecompositionStrategy, str]] = ..., quality: _Optional[_Union[QualityScores, _Mapping]] = ..., execution_metrics: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., summary: _Optional[str] = ...) -> None: ...

class DecompositionRequest(_message.Message):
    __slots__ = ("problem", "strategy", "auto_select_strategy", "max_depth", "max_subproblems", "enable_team_assignment", "enable_mdap", "options")
    PROBLEM_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    AUTO_SELECT_STRATEGY_FIELD_NUMBER: _ClassVar[int]
    MAX_DEPTH_FIELD_NUMBER: _ClassVar[int]
    MAX_SUBPROBLEMS_FIELD_NUMBER: _ClassVar[int]
    ENABLE_TEAM_ASSIGNMENT_FIELD_NUMBER: _ClassVar[int]
    ENABLE_MDAP_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    problem: ProblemDefinition
    strategy: DecompositionStrategy
    auto_select_strategy: bool
    max_depth: int
    max_subproblems: int
    enable_team_assignment: bool
    enable_mdap: bool
    options: _struct_pb2.Struct
    def __init__(self, problem: _Optional[_Union[ProblemDefinition, _Mapping]] = ..., strategy: _Optional[_Union[DecompositionStrategy, str]] = ..., auto_select_strategy: bool = ..., max_depth: _Optional[int] = ..., max_subproblems: _Optional[int] = ..., enable_team_assignment: bool = ..., enable_mdap: bool = ..., options: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class RecompositionRequest(_message.Message):
    __slots__ = ("decomposition_id", "subproblems", "validate_completeness", "check_dependencies")
    DECOMPOSITION_ID_FIELD_NUMBER: _ClassVar[int]
    SUBPROBLEMS_FIELD_NUMBER: _ClassVar[int]
    VALIDATE_COMPLETENESS_FIELD_NUMBER: _ClassVar[int]
    CHECK_DEPENDENCIES_FIELD_NUMBER: _ClassVar[int]
    decomposition_id: str
    subproblems: _containers.RepeatedCompositeFieldContainer[SubProblem]
    validate_completeness: bool
    check_dependencies: bool
    def __init__(self, decomposition_id: _Optional[str] = ..., subproblems: _Optional[_Iterable[_Union[SubProblem, _Mapping]]] = ..., validate_completeness: bool = ..., check_dependencies: bool = ...) -> None: ...

class RecompositionResult(_message.Message):
    __slots__ = ("solution_id", "synthesized_solution", "component_solutions", "is_complete", "missing_components", "dependency_issues", "quality")
    SOLUTION_ID_FIELD_NUMBER: _ClassVar[int]
    SYNTHESIZED_SOLUTION_FIELD_NUMBER: _ClassVar[int]
    COMPONENT_SOLUTIONS_FIELD_NUMBER: _ClassVar[int]
    IS_COMPLETE_FIELD_NUMBER: _ClassVar[int]
    MISSING_COMPONENTS_FIELD_NUMBER: _ClassVar[int]
    DEPENDENCY_ISSUES_FIELD_NUMBER: _ClassVar[int]
    QUALITY_FIELD_NUMBER: _ClassVar[int]
    solution_id: str
    synthesized_solution: str
    component_solutions: _containers.RepeatedScalarFieldContainer[str]
    is_complete: bool
    missing_components: _containers.RepeatedScalarFieldContainer[str]
    dependency_issues: _containers.RepeatedScalarFieldContainer[str]
    quality: QualityScores
    def __init__(self, solution_id: _Optional[str] = ..., synthesized_solution: _Optional[str] = ..., component_solutions: _Optional[_Iterable[str]] = ..., is_complete: bool = ..., missing_components: _Optional[_Iterable[str]] = ..., dependency_issues: _Optional[_Iterable[str]] = ..., quality: _Optional[_Union[QualityScores, _Mapping]] = ...) -> None: ...

class StrategyRecommendation(_message.Message):
    __slots__ = ("recommended_strategy", "confidence", "all_scores", "reasoning")
    class AllScoresEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    RECOMMENDED_STRATEGY_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    ALL_SCORES_FIELD_NUMBER: _ClassVar[int]
    REASONING_FIELD_NUMBER: _ClassVar[int]
    recommended_strategy: DecompositionStrategy
    confidence: float
    all_scores: _containers.ScalarMap[str, float]
    reasoning: str
    def __init__(self, recommended_strategy: _Optional[_Union[DecompositionStrategy, str]] = ..., confidence: _Optional[float] = ..., all_scores: _Optional[_Mapping[str, float]] = ..., reasoning: _Optional[str] = ...) -> None: ...
