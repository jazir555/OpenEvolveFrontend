import common_pb2 as _common_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ProofStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROOF_STATUS_UNSPECIFIED: _ClassVar[ProofStatus]
    PROOF_STATUS_PROVED: _ClassVar[ProofStatus]
    PROOF_STATUS_DISPROVED: _ClassVar[ProofStatus]
    PROOF_STATUS_UNKNOWN: _ClassVar[ProofStatus]
    PROOF_STATUS_IN_PROGRESS: _ClassVar[ProofStatus]
    PROOF_STATUS_TIMEOUT: _ClassVar[ProofStatus]
    PROOF_STATUS_ERROR: _ClassVar[ProofStatus]
PROOF_STATUS_UNSPECIFIED: ProofStatus
PROOF_STATUS_PROVED: ProofStatus
PROOF_STATUS_DISPROVED: ProofStatus
PROOF_STATUS_UNKNOWN: ProofStatus
PROOF_STATUS_IN_PROGRESS: ProofStatus
PROOF_STATUS_TIMEOUT: ProofStatus
PROOF_STATUS_ERROR: ProofStatus

class LeanProof(_message.Message):
    __slots__ = ("theorem_name", "statement", "proof_code", "status", "tactics_used", "proof_length", "elaboration_time_ms", "error_message")
    THEOREM_NAME_FIELD_NUMBER: _ClassVar[int]
    STATEMENT_FIELD_NUMBER: _ClassVar[int]
    PROOF_CODE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TACTICS_USED_FIELD_NUMBER: _ClassVar[int]
    PROOF_LENGTH_FIELD_NUMBER: _ClassVar[int]
    ELABORATION_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    theorem_name: str
    statement: str
    proof_code: str
    status: ProofStatus
    tactics_used: _containers.RepeatedScalarFieldContainer[str]
    proof_length: int
    elaboration_time_ms: int
    error_message: str
    def __init__(self, theorem_name: _Optional[str] = ..., statement: _Optional[str] = ..., proof_code: _Optional[str] = ..., status: _Optional[_Union[ProofStatus, str]] = ..., tactics_used: _Optional[_Iterable[str]] = ..., proof_length: _Optional[int] = ..., elaboration_time_ms: _Optional[int] = ..., error_message: _Optional[str] = ...) -> None: ...

class Z3Result(_message.Message):
    __slots__ = ("status", "model", "proof", "statistics", "solving_time_ms", "error_message")
    class StatisticsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    PROOF_FIELD_NUMBER: _ClassVar[int]
    STATISTICS_FIELD_NUMBER: _ClassVar[int]
    SOLVING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    status: ProofStatus
    model: str
    proof: str
    statistics: _containers.MessageMap[str, _struct_pb2.Value]
    solving_time_ms: int
    error_message: str
    def __init__(self, status: _Optional[_Union[ProofStatus, str]] = ..., model: _Optional[str] = ..., proof: _Optional[str] = ..., statistics: _Optional[_Mapping[str, _struct_pb2.Value]] = ..., solving_time_ms: _Optional[int] = ..., error_message: _Optional[str] = ...) -> None: ...

class MathProblemClassification(_message.Message):
    __slots__ = ("problem_type", "relevant_tactics", "suggested_libraries", "estimated_difficulty", "features")
    class FeaturesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    PROBLEM_TYPE_FIELD_NUMBER: _ClassVar[int]
    RELEVANT_TACTICS_FIELD_NUMBER: _ClassVar[int]
    SUGGESTED_LIBRARIES_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_DIFFICULTY_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    problem_type: str
    relevant_tactics: _containers.RepeatedScalarFieldContainer[str]
    suggested_libraries: _containers.RepeatedScalarFieldContainer[str]
    estimated_difficulty: float
    features: _containers.MessageMap[str, _struct_pb2.Value]
    def __init__(self, problem_type: _Optional[str] = ..., relevant_tactics: _Optional[_Iterable[str]] = ..., suggested_libraries: _Optional[_Iterable[str]] = ..., estimated_difficulty: _Optional[float] = ..., features: _Optional[_Mapping[str, _struct_pb2.Value]] = ...) -> None: ...

class AutoformalizationRequest(_message.Message):
    __slots__ = ("natural_language_statement", "target_language", "domain_hint", "max_attempts", "include_explanation")
    NATURAL_LANGUAGE_STATEMENT_FIELD_NUMBER: _ClassVar[int]
    TARGET_LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_HINT_FIELD_NUMBER: _ClassVar[int]
    MAX_ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_EXPLANATION_FIELD_NUMBER: _ClassVar[int]
    natural_language_statement: str
    target_language: str
    domain_hint: str
    max_attempts: int
    include_explanation: bool
    def __init__(self, natural_language_statement: _Optional[str] = ..., target_language: _Optional[str] = ..., domain_hint: _Optional[str] = ..., max_attempts: _Optional[int] = ..., include_explanation: bool = ...) -> None: ...

class AutoformalizationResult(_message.Message):
    __slots__ = ("formal_statement", "explanation", "is_well_formed", "warnings", "attempts_made", "confidence")
    FORMAL_STATEMENT_FIELD_NUMBER: _ClassVar[int]
    EXPLANATION_FIELD_NUMBER: _ClassVar[int]
    IS_WELL_FORMED_FIELD_NUMBER: _ClassVar[int]
    WARNINGS_FIELD_NUMBER: _ClassVar[int]
    ATTEMPTS_MADE_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    formal_statement: str
    explanation: str
    is_well_formed: bool
    warnings: _containers.RepeatedScalarFieldContainer[str]
    attempts_made: int
    confidence: float
    def __init__(self, formal_statement: _Optional[str] = ..., explanation: _Optional[str] = ..., is_well_formed: bool = ..., warnings: _Optional[_Iterable[str]] = ..., attempts_made: _Optional[int] = ..., confidence: _Optional[float] = ...) -> None: ...

class ProofCheckRequest(_message.Message):
    __slots__ = ("theorem_statement", "proof_code", "language", "check_completeness", "timeout_seconds")
    THEOREM_STATEMENT_FIELD_NUMBER: _ClassVar[int]
    PROOF_CODE_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    CHECK_COMPLETENESS_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    theorem_statement: str
    proof_code: str
    language: str
    check_completeness: bool
    timeout_seconds: int
    def __init__(self, theorem_statement: _Optional[str] = ..., proof_code: _Optional[str] = ..., language: _Optional[str] = ..., check_completeness: bool = ..., timeout_seconds: _Optional[int] = ...) -> None: ...

class ProofCheckResult(_message.Message):
    __slots__ = ("status", "is_valid", "errors", "warnings", "elaborated_code", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    STATUS_FIELD_NUMBER: _ClassVar[int]
    IS_VALID_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    WARNINGS_FIELD_NUMBER: _ClassVar[int]
    ELABORATED_CODE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    status: ProofStatus
    is_valid: bool
    errors: _containers.RepeatedCompositeFieldContainer[ProofError]
    warnings: _containers.RepeatedCompositeFieldContainer[ProofWarning]
    elaborated_code: str
    metadata: _containers.MessageMap[str, _struct_pb2.Value]
    def __init__(self, status: _Optional[_Union[ProofStatus, str]] = ..., is_valid: bool = ..., errors: _Optional[_Iterable[_Union[ProofError, _Mapping]]] = ..., warnings: _Optional[_Iterable[_Union[ProofWarning, _Mapping]]] = ..., elaborated_code: _Optional[str] = ..., metadata: _Optional[_Mapping[str, _struct_pb2.Value]] = ...) -> None: ...

class ProofError(_message.Message):
    __slots__ = ("line", "column", "message", "error_type", "suggestion")
    LINE_FIELD_NUMBER: _ClassVar[int]
    COLUMN_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ERROR_TYPE_FIELD_NUMBER: _ClassVar[int]
    SUGGESTION_FIELD_NUMBER: _ClassVar[int]
    line: int
    column: int
    message: str
    error_type: str
    suggestion: str
    def __init__(self, line: _Optional[int] = ..., column: _Optional[int] = ..., message: _Optional[str] = ..., error_type: _Optional[str] = ..., suggestion: _Optional[str] = ...) -> None: ...

class ProofWarning(_message.Message):
    __slots__ = ("line", "message", "warning_type")
    LINE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    WARNING_TYPE_FIELD_NUMBER: _ClassVar[int]
    line: int
    message: str
    warning_type: str
    def __init__(self, line: _Optional[int] = ..., message: _Optional[str] = ..., warning_type: _Optional[str] = ...) -> None: ...

class TacticRecommendationRequest(_message.Message):
    __slots__ = ("theorem_statement", "current_proof_state", "goal", "num_recommendations", "context")
    THEOREM_STATEMENT_FIELD_NUMBER: _ClassVar[int]
    CURRENT_PROOF_STATE_FIELD_NUMBER: _ClassVar[int]
    GOAL_FIELD_NUMBER: _ClassVar[int]
    NUM_RECOMMENDATIONS_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    theorem_statement: str
    current_proof_state: str
    goal: str
    num_recommendations: int
    context: str
    def __init__(self, theorem_statement: _Optional[str] = ..., current_proof_state: _Optional[str] = ..., goal: _Optional[str] = ..., num_recommendations: _Optional[int] = ..., context: _Optional[str] = ...) -> None: ...

class TacticRecommendation(_message.Message):
    __slots__ = ("tactic", "confidence", "explanation", "expected_outcome", "typical_usage_count")
    TACTIC_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    EXPLANATION_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_OUTCOME_FIELD_NUMBER: _ClassVar[int]
    TYPICAL_USAGE_COUNT_FIELD_NUMBER: _ClassVar[int]
    tactic: str
    confidence: float
    explanation: str
    expected_outcome: str
    typical_usage_count: int
    def __init__(self, tactic: _Optional[str] = ..., confidence: _Optional[float] = ..., explanation: _Optional[str] = ..., expected_outcome: _Optional[str] = ..., typical_usage_count: _Optional[int] = ...) -> None: ...

class TacticRecommendationResult(_message.Message):
    __slots__ = ("recommendations", "reasoning")
    RECOMMENDATIONS_FIELD_NUMBER: _ClassVar[int]
    REASONING_FIELD_NUMBER: _ClassVar[int]
    recommendations: _containers.RepeatedCompositeFieldContainer[TacticRecommendation]
    reasoning: str
    def __init__(self, recommendations: _Optional[_Iterable[_Union[TacticRecommendation, _Mapping]]] = ..., reasoning: _Optional[str] = ...) -> None: ...

class LibrarySearchRequest(_message.Message):
    __slots__ = ("query", "language", "filter_libraries", "max_results")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    FILTER_LIBRARIES_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    query: str
    language: str
    filter_libraries: _containers.RepeatedScalarFieldContainer[str]
    max_results: int
    def __init__(self, query: _Optional[str] = ..., language: _Optional[str] = ..., filter_libraries: _Optional[_Iterable[str]] = ..., max_results: _Optional[int] = ...) -> None: ...

class LibrarySearchResult(_message.Message):
    __slots__ = ("items", "total_matches")
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MATCHES_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[LibraryItem]
    total_matches: int
    def __init__(self, items: _Optional[_Iterable[_Union[LibraryItem, _Mapping]]] = ..., total_matches: _Optional[int] = ...) -> None: ...

class LibraryItem(_message.Message):
    __slots__ = ("name", "signature", "documentation", "library", "module", "relevance_score", "tags")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    DOCUMENTATION_FIELD_NUMBER: _ClassVar[int]
    LIBRARY_FIELD_NUMBER: _ClassVar[int]
    MODULE_FIELD_NUMBER: _ClassVar[int]
    RELEVANCE_SCORE_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    name: str
    signature: str
    documentation: str
    library: str
    module: str
    relevance_score: float
    tags: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: _Optional[str] = ..., signature: _Optional[str] = ..., documentation: _Optional[str] = ..., library: _Optional[str] = ..., module: _Optional[str] = ..., relevance_score: _Optional[float] = ..., tags: _Optional[_Iterable[str]] = ...) -> None: ...

class ConjectureRequest(_message.Message):
    __slots__ = ("problem_statement", "known_theorems", "target_property", "max_conjectures")
    PROBLEM_STATEMENT_FIELD_NUMBER: _ClassVar[int]
    KNOWN_THEOREMS_FIELD_NUMBER: _ClassVar[int]
    TARGET_PROPERTY_FIELD_NUMBER: _ClassVar[int]
    MAX_CONJECTURES_FIELD_NUMBER: _ClassVar[int]
    problem_statement: str
    known_theorems: _containers.RepeatedScalarFieldContainer[str]
    target_property: str
    max_conjectures: int
    def __init__(self, problem_statement: _Optional[str] = ..., known_theorems: _Optional[_Iterable[str]] = ..., target_property: _Optional[str] = ..., max_conjectures: _Optional[int] = ...) -> None: ...

class ConjectureResult(_message.Message):
    __slots__ = ("conjectures", "reasoning")
    CONJECTURES_FIELD_NUMBER: _ClassVar[int]
    REASONING_FIELD_NUMBER: _ClassVar[int]
    conjectures: _containers.RepeatedCompositeFieldContainer[Conjecture]
    reasoning: str
    def __init__(self, conjectures: _Optional[_Iterable[_Union[Conjecture, _Mapping]]] = ..., reasoning: _Optional[str] = ...) -> None: ...

class Conjecture(_message.Message):
    __slots__ = ("statement", "plausibility", "justification", "supporting_evidence", "counter_examples")
    STATEMENT_FIELD_NUMBER: _ClassVar[int]
    PLAUSIBILITY_FIELD_NUMBER: _ClassVar[int]
    JUSTIFICATION_FIELD_NUMBER: _ClassVar[int]
    SUPPORTING_EVIDENCE_FIELD_NUMBER: _ClassVar[int]
    COUNTER_EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    statement: str
    plausibility: float
    justification: str
    supporting_evidence: _containers.RepeatedScalarFieldContainer[str]
    counter_examples: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, statement: _Optional[str] = ..., plausibility: _Optional[float] = ..., justification: _Optional[str] = ..., supporting_evidence: _Optional[_Iterable[str]] = ..., counter_examples: _Optional[_Iterable[str]] = ...) -> None: ...

class CounterexampleRequest(_message.Message):
    __slots__ = ("statement", "constraints", "max_counterexamples", "search_depth")
    STATEMENT_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    MAX_COUNTEREXAMPLES_FIELD_NUMBER: _ClassVar[int]
    SEARCH_DEPTH_FIELD_NUMBER: _ClassVar[int]
    statement: str
    constraints: _containers.RepeatedScalarFieldContainer[str]
    max_counterexamples: int
    search_depth: int
    def __init__(self, statement: _Optional[str] = ..., constraints: _Optional[_Iterable[str]] = ..., max_counterexamples: _Optional[int] = ..., search_depth: _Optional[int] = ...) -> None: ...

class CounterexampleResult(_message.Message):
    __slots__ = ("counterexamples", "exhaustive_search", "search_space_size")
    COUNTEREXAMPLES_FIELD_NUMBER: _ClassVar[int]
    EXHAUSTIVE_SEARCH_FIELD_NUMBER: _ClassVar[int]
    SEARCH_SPACE_SIZE_FIELD_NUMBER: _ClassVar[int]
    counterexamples: _containers.RepeatedCompositeFieldContainer[Counterexample]
    exhaustive_search: bool
    search_space_size: int
    def __init__(self, counterexamples: _Optional[_Iterable[_Union[Counterexample, _Mapping]]] = ..., exhaustive_search: bool = ..., search_space_size: _Optional[int] = ...) -> None: ...

class Counterexample(_message.Message):
    __slots__ = ("values", "explanation", "is_minimal")
    class ValuesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    VALUES_FIELD_NUMBER: _ClassVar[int]
    EXPLANATION_FIELD_NUMBER: _ClassVar[int]
    IS_MINIMAL_FIELD_NUMBER: _ClassVar[int]
    values: _containers.MessageMap[str, _struct_pb2.Value]
    explanation: str
    is_minimal: bool
    def __init__(self, values: _Optional[_Mapping[str, _struct_pb2.Value]] = ..., explanation: _Optional[str] = ..., is_minimal: bool = ...) -> None: ...

class ProofSimplificationRequest(_message.Message):
    __slots__ = ("proof_code", "language", "aggressive", "target_length")
    PROOF_CODE_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    AGGRESSIVE_FIELD_NUMBER: _ClassVar[int]
    TARGET_LENGTH_FIELD_NUMBER: _ClassVar[int]
    proof_code: str
    language: str
    aggressive: bool
    target_length: int
    def __init__(self, proof_code: _Optional[str] = ..., language: _Optional[str] = ..., aggressive: bool = ..., target_length: _Optional[int] = ...) -> None: ...

class ProofSimplificationResult(_message.Message):
    __slots__ = ("simplified_proof", "original_length", "simplified_length", "compression_ratio", "transformations_applied", "preserves_semantics")
    SIMPLIFIED_PROOF_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_LENGTH_FIELD_NUMBER: _ClassVar[int]
    SIMPLIFIED_LENGTH_FIELD_NUMBER: _ClassVar[int]
    COMPRESSION_RATIO_FIELD_NUMBER: _ClassVar[int]
    TRANSFORMATIONS_APPLIED_FIELD_NUMBER: _ClassVar[int]
    PRESERVES_SEMANTICS_FIELD_NUMBER: _ClassVar[int]
    simplified_proof: str
    original_length: int
    simplified_length: int
    compression_ratio: float
    transformations_applied: _containers.RepeatedScalarFieldContainer[str]
    preserves_semantics: bool
    def __init__(self, simplified_proof: _Optional[str] = ..., original_length: _Optional[int] = ..., simplified_length: _Optional[int] = ..., compression_ratio: _Optional[float] = ..., transformations_applied: _Optional[_Iterable[str]] = ..., preserves_semantics: bool = ...) -> None: ...

class InductionHelperRequest(_message.Message):
    __slots__ = ("property", "induction_variable", "base_case", "inductive_step_template")
    PROPERTY_FIELD_NUMBER: _ClassVar[int]
    INDUCTION_VARIABLE_FIELD_NUMBER: _ClassVar[int]
    BASE_CASE_FIELD_NUMBER: _ClassVar[int]
    INDUCTIVE_STEP_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    property: str
    induction_variable: str
    base_case: str
    inductive_step_template: str
    def __init__(self, property: _Optional[str] = ..., induction_variable: _Optional[str] = ..., base_case: _Optional[str] = ..., inductive_step_template: _Optional[str] = ...) -> None: ...

class InductionHelperResult(_message.Message):
    __slots__ = ("base_case_proof", "inductive_hypothesis", "inductive_step_proof", "suggested_lemmas")
    BASE_CASE_PROOF_FIELD_NUMBER: _ClassVar[int]
    INDUCTIVE_HYPOTHESIS_FIELD_NUMBER: _ClassVar[int]
    INDUCTIVE_STEP_PROOF_FIELD_NUMBER: _ClassVar[int]
    SUGGESTED_LEMMAS_FIELD_NUMBER: _ClassVar[int]
    base_case_proof: str
    inductive_hypothesis: str
    inductive_step_proof: str
    suggested_lemmas: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, base_case_proof: _Optional[str] = ..., inductive_hypothesis: _Optional[str] = ..., inductive_step_proof: _Optional[str] = ..., suggested_lemmas: _Optional[_Iterable[str]] = ...) -> None: ...

class Z3ConstraintRequest(_message.Message):
    __slots__ = ("constraints", "variables", "objective", "timeout_seconds")
    CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    VARIABLES_FIELD_NUMBER: _ClassVar[int]
    OBJECTIVE_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    constraints: _containers.RepeatedScalarFieldContainer[str]
    variables: _containers.RepeatedScalarFieldContainer[str]
    objective: str
    timeout_seconds: int
    def __init__(self, constraints: _Optional[_Iterable[str]] = ..., variables: _Optional[_Iterable[str]] = ..., objective: _Optional[str] = ..., timeout_seconds: _Optional[int] = ...) -> None: ...

class Z3TheoremRequest(_message.Message):
    __slots__ = ("theorem", "axioms", "definitions", "timeout_seconds", "generate_proof")
    THEOREM_FIELD_NUMBER: _ClassVar[int]
    AXIOMS_FIELD_NUMBER: _ClassVar[int]
    DEFINITIONS_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    GENERATE_PROOF_FIELD_NUMBER: _ClassVar[int]
    theorem: str
    axioms: _containers.RepeatedScalarFieldContainer[str]
    definitions: _containers.RepeatedScalarFieldContainer[str]
    timeout_seconds: int
    generate_proof: bool
    def __init__(self, theorem: _Optional[str] = ..., axioms: _Optional[_Iterable[str]] = ..., definitions: _Optional[_Iterable[str]] = ..., timeout_seconds: _Optional[int] = ..., generate_proof: bool = ...) -> None: ...

class EquivalenceRequest(_message.Message):
    __slots__ = ("expression1", "expression2", "language", "assumptions")
    EXPRESSION1_FIELD_NUMBER: _ClassVar[int]
    EXPRESSION2_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    ASSUMPTIONS_FIELD_NUMBER: _ClassVar[int]
    expression1: str
    expression2: str
    language: str
    assumptions: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, expression1: _Optional[str] = ..., expression2: _Optional[str] = ..., language: _Optional[str] = ..., assumptions: _Optional[_Iterable[str]] = ...) -> None: ...

class EquivalenceResult(_message.Message):
    __slots__ = ("are_equivalent", "proof", "counterexample", "statistics")
    class StatisticsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    ARE_EQUIVALENT_FIELD_NUMBER: _ClassVar[int]
    PROOF_FIELD_NUMBER: _ClassVar[int]
    COUNTEREXAMPLE_FIELD_NUMBER: _ClassVar[int]
    STATISTICS_FIELD_NUMBER: _ClassVar[int]
    are_equivalent: bool
    proof: str
    counterexample: str
    statistics: _containers.MessageMap[str, _struct_pb2.Value]
    def __init__(self, are_equivalent: bool = ..., proof: _Optional[str] = ..., counterexample: _Optional[str] = ..., statistics: _Optional[_Mapping[str, _struct_pb2.Value]] = ...) -> None: ...
