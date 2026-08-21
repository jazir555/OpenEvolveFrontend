import common_pb2 as _common_pb2
import decomposition_pb2 as _decomposition_pb2
import math_pb2 as _math_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GauntletType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    GAUNTLET_TYPE_UNSPECIFIED: _ClassVar[GauntletType]
    GAUNTLET_TYPE_FORMAL: _ClassVar[GauntletType]
    GAUNTLET_TYPE_ADVERSARIAL: _ClassVar[GauntletType]
    GAUNTLET_TYPE_EDGE_CASE: _ClassVar[GauntletType]
    GAUNTLET_TYPE_STRESS: _ClassVar[GauntletType]
    GAUNTLET_TYPE_COMPREHENSIVE: _ClassVar[GauntletType]

class AttackVector(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ATTACK_VECTOR_UNSPECIFIED: _ClassVar[AttackVector]
    ATTACK_VECTOR_EDGE_CASES: _ClassVar[AttackVector]
    ATTACK_VECTOR_ASSUMPTIONS: _ClassVar[AttackVector]
    ATTACK_VECTOR_BOUNDARIES: _ClassVar[AttackVector]
    ATTACK_VECTOR_CONTRADICTIONS: _ClassVar[AttackVector]
    ATTACK_VECTOR_AMBIGUITY: _ClassVar[AttackVector]
    ATTACK_VECTOR_SCALE: _ClassVar[AttackVector]
    ATTACK_VECTOR_COMPOSITION: _ClassVar[AttackVector]
GAUNTLET_TYPE_UNSPECIFIED: GauntletType
GAUNTLET_TYPE_FORMAL: GauntletType
GAUNTLET_TYPE_ADVERSARIAL: GauntletType
GAUNTLET_TYPE_EDGE_CASE: GauntletType
GAUNTLET_TYPE_STRESS: GauntletType
GAUNTLET_TYPE_COMPREHENSIVE: GauntletType
ATTACK_VECTOR_UNSPECIFIED: AttackVector
ATTACK_VECTOR_EDGE_CASES: AttackVector
ATTACK_VECTOR_ASSUMPTIONS: AttackVector
ATTACK_VECTOR_BOUNDARIES: AttackVector
ATTACK_VECTOR_CONTRADICTIONS: AttackVector
ATTACK_VECTOR_AMBIGUITY: AttackVector
ATTACK_VECTOR_SCALE: AttackVector
ATTACK_VECTOR_COMPOSITION: AttackVector

class GauntletChallenge(_message.Message):
    __slots__ = ("id", "name", "description", "type", "attack_vectors", "parameters", "difficulty", "tags")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    ATTACK_VECTORS_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    DIFFICULTY_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    description: str
    type: GauntletType
    attack_vectors: _containers.RepeatedScalarFieldContainer[AttackVector]
    parameters: _struct_pb2.Struct
    difficulty: int
    tags: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., type: _Optional[_Union[GauntletType, str]] = ..., attack_vectors: _Optional[_Iterable[_Union[AttackVector, str]]] = ..., parameters: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., difficulty: _Optional[int] = ..., tags: _Optional[_Iterable[str]] = ...) -> None: ...

class GauntletTarget(_message.Message):
    __slots__ = ("problem", "subproblem", "proof", "solution_code", "generic_target")
    PROBLEM_FIELD_NUMBER: _ClassVar[int]
    SUBPROBLEM_FIELD_NUMBER: _ClassVar[int]
    PROOF_FIELD_NUMBER: _ClassVar[int]
    SOLUTION_CODE_FIELD_NUMBER: _ClassVar[int]
    GENERIC_TARGET_FIELD_NUMBER: _ClassVar[int]
    problem: _decomposition_pb2.ProblemDefinition
    subproblem: _decomposition_pb2.SubProblem
    proof: _math_pb2.LeanProof
    solution_code: str
    generic_target: _struct_pb2.Struct
    def __init__(self, problem: _Optional[_Union[_decomposition_pb2.ProblemDefinition, _Mapping]] = ..., subproblem: _Optional[_Union[_decomposition_pb2.SubProblem, _Mapping]] = ..., proof: _Optional[_Union[_math_pb2.LeanProof, _Mapping]] = ..., solution_code: _Optional[str] = ..., generic_target: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class GauntletRequest(_message.Message):
    __slots__ = ("gauntlet_id", "target", "config", "enable_red_team", "enable_blue_team", "max_iterations", "timeout_seconds")
    GAUNTLET_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    ENABLE_RED_TEAM_FIELD_NUMBER: _ClassVar[int]
    ENABLE_BLUE_TEAM_FIELD_NUMBER: _ClassVar[int]
    MAX_ITERATIONS_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    gauntlet_id: str
    target: GauntletTarget
    config: _struct_pb2.Struct
    enable_red_team: bool
    enable_blue_team: bool
    max_iterations: int
    timeout_seconds: int
    def __init__(self, gauntlet_id: _Optional[str] = ..., target: _Optional[_Union[GauntletTarget, _Mapping]] = ..., config: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., enable_red_team: bool = ..., enable_blue_team: bool = ..., max_iterations: _Optional[int] = ..., timeout_seconds: _Optional[int] = ...) -> None: ...

class GauntletResult(_message.Message):
    __slots__ = ("execution_id", "status", "challenge_results", "red_team_report", "blue_team_report", "overall_score", "metadata")
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CHALLENGE_RESULTS_FIELD_NUMBER: _ClassVar[int]
    RED_TEAM_REPORT_FIELD_NUMBER: _ClassVar[int]
    BLUE_TEAM_REPORT_FIELD_NUMBER: _ClassVar[int]
    OVERALL_SCORE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    execution_id: str
    status: GauntletStatus
    challenge_results: _containers.RepeatedCompositeFieldContainer[ChallengeResult]
    red_team_report: RedTeamReport
    blue_team_report: BlueTeamReport
    overall_score: OverallScore
    metadata: _struct_pb2.Struct
    def __init__(self, execution_id: _Optional[str] = ..., status: _Optional[_Union[GauntletStatus, _Mapping]] = ..., challenge_results: _Optional[_Iterable[_Union[ChallengeResult, _Mapping]]] = ..., red_team_report: _Optional[_Union[RedTeamReport, _Mapping]] = ..., blue_team_report: _Optional[_Union[BlueTeamReport, _Mapping]] = ..., overall_score: _Optional[_Union[OverallScore, _Mapping]] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class GauntletStatus(_message.Message):
    __slots__ = ("state", "current_iteration", "total_iterations", "current_phase", "started_at", "completed_at", "elapsed_seconds")
    STATE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_ITERATION_FIELD_NUMBER: _ClassVar[int]
    TOTAL_ITERATIONS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_PHASE_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_AT_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_SECONDS_FIELD_NUMBER: _ClassVar[int]
    state: _common_pb2.ExecutionState
    current_iteration: int
    total_iterations: int
    current_phase: str
    started_at: _timestamp_pb2.Timestamp
    completed_at: _timestamp_pb2.Timestamp
    elapsed_seconds: int
    def __init__(self, state: _Optional[_Union[_common_pb2.ExecutionState, str]] = ..., current_iteration: _Optional[int] = ..., total_iterations: _Optional[int] = ..., current_phase: _Optional[str] = ..., started_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., completed_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., elapsed_seconds: _Optional[int] = ...) -> None: ...

class ChallengeResult(_message.Message):
    __slots__ = ("challenge_id", "challenge_name", "passed", "score", "findings", "execution_time_ms", "details")
    CHALLENGE_ID_FIELD_NUMBER: _ClassVar[int]
    CHALLENGE_NAME_FIELD_NUMBER: _ClassVar[int]
    PASSED_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    FINDINGS_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    challenge_id: str
    challenge_name: str
    passed: bool
    score: float
    findings: _containers.RepeatedCompositeFieldContainer[Finding]
    execution_time_ms: int
    details: _struct_pb2.Struct
    def __init__(self, challenge_id: _Optional[str] = ..., challenge_name: _Optional[str] = ..., passed: bool = ..., score: _Optional[float] = ..., findings: _Optional[_Iterable[_Union[Finding, _Mapping]]] = ..., execution_time_ms: _Optional[int] = ..., details: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class Finding(_message.Message):
    __slots__ = ("id", "type", "severity", "title", "description", "location", "recommendation", "evidence")
    ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    RECOMMENDATION_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_FIELD_NUMBER: _ClassVar[int]
    id: str
    type: str
    severity: str
    title: str
    description: str
    location: str
    recommendation: str
    evidence: _struct_pb2.Struct
    def __init__(self, id: _Optional[str] = ..., type: _Optional[str] = ..., severity: _Optional[str] = ..., title: _Optional[str] = ..., description: _Optional[str] = ..., location: _Optional[str] = ..., recommendation: _Optional[str] = ..., evidence: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class RedTeamReport(_message.Message):
    __slots__ = ("total_attacks", "successful_attacks", "success_rate", "attacks", "attack_vector_effectiveness", "vulnerabilities_found")
    class AttackVectorEffectivenessEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    TOTAL_ATTACKS_FIELD_NUMBER: _ClassVar[int]
    SUCCESSFUL_ATTACKS_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_RATE_FIELD_NUMBER: _ClassVar[int]
    ATTACKS_FIELD_NUMBER: _ClassVar[int]
    ATTACK_VECTOR_EFFECTIVENESS_FIELD_NUMBER: _ClassVar[int]
    VULNERABILITIES_FOUND_FIELD_NUMBER: _ClassVar[int]
    total_attacks: int
    successful_attacks: int
    success_rate: float
    attacks: _containers.RepeatedCompositeFieldContainer[AttackResult]
    attack_vector_effectiveness: _containers.ScalarMap[str, float]
    vulnerabilities_found: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, total_attacks: _Optional[int] = ..., successful_attacks: _Optional[int] = ..., success_rate: _Optional[float] = ..., attacks: _Optional[_Iterable[_Union[AttackResult, _Mapping]]] = ..., attack_vector_effectiveness: _Optional[_Mapping[str, float]] = ..., vulnerabilities_found: _Optional[_Iterable[str]] = ...) -> None: ...

class AttackResult(_message.Message):
    __slots__ = ("attack_id", "vector", "description", "succeeded", "target_component", "severity_score", "payload", "result_description")
    ATTACK_ID_FIELD_NUMBER: _ClassVar[int]
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SUCCEEDED_FIELD_NUMBER: _ClassVar[int]
    TARGET_COMPONENT_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_SCORE_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    RESULT_DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    attack_id: str
    vector: AttackVector
    description: str
    succeeded: bool
    target_component: str
    severity_score: float
    payload: _struct_pb2.Struct
    result_description: str
    def __init__(self, attack_id: _Optional[str] = ..., vector: _Optional[_Union[AttackVector, str]] = ..., description: _Optional[str] = ..., succeeded: bool = ..., target_component: _Optional[str] = ..., severity_score: _Optional[float] = ..., payload: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., result_description: _Optional[str] = ...) -> None: ...

class BlueTeamReport(_message.Message):
    __slots__ = ("total_defenses", "successful_defenses", "defense_rate", "defenses", "defense_strategy_effectiveness", "improvements_made")
    class DefenseStrategyEffectivenessEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    TOTAL_DEFENSES_FIELD_NUMBER: _ClassVar[int]
    SUCCESSFUL_DEFENSES_FIELD_NUMBER: _ClassVar[int]
    DEFENSE_RATE_FIELD_NUMBER: _ClassVar[int]
    DEFENSES_FIELD_NUMBER: _ClassVar[int]
    DEFENSE_STRATEGY_EFFECTIVENESS_FIELD_NUMBER: _ClassVar[int]
    IMPROVEMENTS_MADE_FIELD_NUMBER: _ClassVar[int]
    total_defenses: int
    successful_defenses: int
    defense_rate: float
    defenses: _containers.RepeatedCompositeFieldContainer[DefenseResult]
    defense_strategy_effectiveness: _containers.ScalarMap[str, float]
    improvements_made: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, total_defenses: _Optional[int] = ..., successful_defenses: _Optional[int] = ..., defense_rate: _Optional[float] = ..., defenses: _Optional[_Iterable[_Union[DefenseResult, _Mapping]]] = ..., defense_strategy_effectiveness: _Optional[_Mapping[str, float]] = ..., improvements_made: _Optional[_Iterable[str]] = ...) -> None: ...

class DefenseResult(_message.Message):
    __slots__ = ("defense_id", "strategy", "description", "succeeded", "countered_attack", "effectiveness_score", "modifications")
    DEFENSE_ID_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SUCCEEDED_FIELD_NUMBER: _ClassVar[int]
    COUNTERED_ATTACK_FIELD_NUMBER: _ClassVar[int]
    EFFECTIVENESS_SCORE_FIELD_NUMBER: _ClassVar[int]
    MODIFICATIONS_FIELD_NUMBER: _ClassVar[int]
    defense_id: str
    strategy: str
    description: str
    succeeded: bool
    countered_attack: str
    effectiveness_score: float
    modifications: _struct_pb2.Struct
    def __init__(self, defense_id: _Optional[str] = ..., strategy: _Optional[str] = ..., description: _Optional[str] = ..., succeeded: bool = ..., countered_attack: _Optional[str] = ..., effectiveness_score: _Optional[float] = ..., modifications: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class OverallScore(_message.Message):
    __slots__ = ("total_score", "robustness", "correctness", "completeness", "performance", "grade", "summary")
    TOTAL_SCORE_FIELD_NUMBER: _ClassVar[int]
    ROBUSTNESS_FIELD_NUMBER: _ClassVar[int]
    CORRECTNESS_FIELD_NUMBER: _ClassVar[int]
    COMPLETENESS_FIELD_NUMBER: _ClassVar[int]
    PERFORMANCE_FIELD_NUMBER: _ClassVar[int]
    GRADE_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    total_score: float
    robustness: float
    correctness: float
    completeness: float
    performance: float
    grade: str
    summary: str
    def __init__(self, total_score: _Optional[float] = ..., robustness: _Optional[float] = ..., correctness: _Optional[float] = ..., completeness: _Optional[float] = ..., performance: _Optional[float] = ..., grade: _Optional[str] = ..., summary: _Optional[str] = ...) -> None: ...

class GauntletUpdate(_message.Message):
    __slots__ = ("execution_id", "status", "current_challenge", "progress", "new_findings", "phase_message")
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_CHALLENGE_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    NEW_FINDINGS_FIELD_NUMBER: _ClassVar[int]
    PHASE_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    execution_id: str
    status: GauntletStatus
    current_challenge: ChallengeResult
    progress: _common_pb2.Progress
    new_findings: _containers.RepeatedCompositeFieldContainer[Finding]
    phase_message: str
    def __init__(self, execution_id: _Optional[str] = ..., status: _Optional[_Union[GauntletStatus, _Mapping]] = ..., current_challenge: _Optional[_Union[ChallengeResult, _Mapping]] = ..., progress: _Optional[_Union[_common_pb2.Progress, _Mapping]] = ..., new_findings: _Optional[_Iterable[_Union[Finding, _Mapping]]] = ..., phase_message: _Optional[str] = ...) -> None: ...

class CreateGauntletRequest(_message.Message):
    __slots__ = ("name", "description", "type", "challenge_templates", "custom_parameters")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    CHALLENGE_TEMPLATES_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    type: GauntletType
    challenge_templates: _containers.RepeatedScalarFieldContainer[str]
    custom_parameters: _struct_pb2.Struct
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ..., type: _Optional[_Union[GauntletType, str]] = ..., challenge_templates: _Optional[_Iterable[str]] = ..., custom_parameters: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class CreateGauntletResponse(_message.Message):
    __slots__ = ("gauntlet_id", "challenge", "success", "message")
    GAUNTLET_ID_FIELD_NUMBER: _ClassVar[int]
    CHALLENGE_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    gauntlet_id: str
    challenge: GauntletChallenge
    success: bool
    message: str
    def __init__(self, gauntlet_id: _Optional[str] = ..., challenge: _Optional[_Union[GauntletChallenge, _Mapping]] = ..., success: bool = ..., message: _Optional[str] = ...) -> None: ...

class ListGauntletsRequest(_message.Message):
    __slots__ = ("type", "tags", "min_difficulty", "max_difficulty")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    MIN_DIFFICULTY_FIELD_NUMBER: _ClassVar[int]
    MAX_DIFFICULTY_FIELD_NUMBER: _ClassVar[int]
    type: GauntletType
    tags: _containers.RepeatedScalarFieldContainer[str]
    min_difficulty: int
    max_difficulty: int
    def __init__(self, type: _Optional[_Union[GauntletType, str]] = ..., tags: _Optional[_Iterable[str]] = ..., min_difficulty: _Optional[int] = ..., max_difficulty: _Optional[int] = ...) -> None: ...

class ListGauntletsResponse(_message.Message):
    __slots__ = ("gauntlets",)
    GAUNTLETS_FIELD_NUMBER: _ClassVar[int]
    gauntlets: _containers.RepeatedCompositeFieldContainer[GauntletChallenge]
    def __init__(self, gauntlets: _Optional[_Iterable[_Union[GauntletChallenge, _Mapping]]] = ...) -> None: ...
