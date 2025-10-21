import dataclasses
from typing import List, Dict, Any, Optional, Literal, Set
import time

# --- Core Data Structures ---

@dataclasses.dataclass
class ModelConfig:
    """Configuration for a single AI model within a team.

    Attributes:
        model_id (str): Unique identifier for the AI model (e.g., "gpt-4o", "claude-3-opus").
        api_key (str): API key for authentication with the model provider.
        api_base (str): Base URL for the API endpoint (defaults to OpenAI's).
        temperature (float): Controls randomness in model outputs (0.0-2.0).
        top_p (float): Nucleus sampling parameter (0.0-1.0).
        max_tokens (int): Maximum number of tokens to generate.
        frequency_penalty (float): Penalizes new tokens based on their existing frequency in the text so far.
        presence_penalty (float): Penalizes new tokens based on whether they appear in the text so far.
        seed (Optional[int]): Seed for reproducible sampling.
        n (Optional[int]): Number of chat completion choices to generate for each input message.
        logit_bias (Optional[Dict[int, int]]): Modify the likelihood of specified tokens appearing in the completion.
        reasoning_effort (Optional[str]): The reasoning effort to apply for the model (e.g., 'low', 'medium', 'high').
        stop_sequences (Optional[List[str]]): Up to 4 sequences where the API will stop generating further tokens.
        logprobs (Optional[bool]): Whether to return log probabilities of the output tokens or not.
        top_logprobs (Optional[int]): An integer between 0 and 5 specifying the number of most likely tokens to return at each token position.
        response_format (Optional[Dict[str, str]]): An object specifying the format that the model must output.
        stream (Optional[bool]): If set, partial message deltas will be sent, like in ChatGPT.
        user (Optional[str]): A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
        max_retries (int): Maximum number of retries for API calls.
        timeout (int): Timeout for API calls in seconds.
        organization (Optional[str]): For OpenAI, the organization ID.
        response_model (Optional[str]): For structured output, a Pydantic model or similar (string representation).
        tools (Optional[List[Dict[str, Any]]]): For function calling, a list of tool definitions.
        tool_choice (Optional[Any]): For function calling, control over tool usage (e.g., "auto", "none", {"type": "function", "function": {"name": "my_function"}}).
        system_fingerprint (Optional[str]): For OpenAI, a unique identifier for the model's configuration.
        deployment_id (Optional[str]): For Azure OpenAI, the deployment name.
        encoding_format (Optional[str]): For some models, the encoding format for output (e.g., "base64").
        max_input_tokens (Optional[int]): Maximum number of input tokens.
        stop_token (Optional[str]): A single stop token (alternative to stop_sequences).
        best_of (Optional[int]): Generates best_of completions on the server side and returns the "best".
        logprobs_offset (Optional[int]): Offset for logprobs.
        suffix (Optional[str]): A suffix that will be appended to the end of the generated text.
        presence_penalty_range (Optional[List[float]]): Range for presence penalty.
        frequency_penalty_range (Optional[List[float]]): Range for frequency penalty.
        stop_token_id (Optional[int]): For models that use token IDs for stopping.
        response_json_format (Optional[bool]): If the response should be in JSON format.
        max_output_tokens (Optional[int]): Maximum number of output tokens.
        stream_options (Optional[Dict[str, Any]]): For more granular control over streaming.
        logprobs_type (Optional[str]): To specify the type of log probabilities.
        top_k (Optional[int]): Another common sampling parameter.
        repetition_penalty (Optional[float]): To penalize repeated tokens.
        length_penalty (Optional[float]): To control the length of generated sequences.
        early_stopping (Optional[bool]): For beam search.
        num_beams (Optional[int]): For beam search.
        do_sample (Optional[bool]): To enable/disable sampling.
        temperature_fallback (Optional[float]): A fallback temperature.
        top_p_fallback (Optional[float]): A fallback top_p.
        max_time (Optional[int]): Maximum time to generate a response.
        return_full_text (Optional[bool]): Whether to return the full text or just the generated part.
        tokenizer_config (Optional[Dict[str, Any]]): For tokenizer-specific settings.
        model_kwargs (Optional[Dict[str, Any]]): For any other model-specific keyword arguments.
    """
    model_id: str
    api_key: str
    api_base: str = "https://api.openai.com/v1"
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 4096
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    n: Optional[int] = 1
    logit_bias: Optional[Dict[int, int]] = None
    reasoning_effort: Optional[str] = None
    stop_sequences: Optional[List[str]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    response_format: Optional[Dict[str, str]] = None
    stream: Optional[bool] = None
    user: Optional[str] = None
    max_retries: int = 5
    timeout: int = 120
    organization: Optional[str] = None
    response_model: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    system_fingerprint: Optional[str] = None
    deployment_id: Optional[str] = None
    encoding_format: Optional[str] = None
    max_input_tokens: Optional[int] = None
    stop_token: Optional[str] = None
    best_of: Optional[int] = None
    logprobs_offset: Optional[int] = None
    suffix: Optional[str] = None
    presence_penalty_range: Optional[List[float]] = None
    frequency_penalty_range: Optional[List[float]] = None
    stop_token_id: Optional[int] = None
    response_json_format: Optional[bool] = None
    max_output_tokens: Optional[int] = None
    stream_options: Optional[Dict[str, Any]] = None
    logprobs_type: Optional[str] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    early_stopping: Optional[bool] = None
    num_beams: Optional[int] = None
    do_sample: Optional[bool] = None
    temperature_fallback: Optional[float] = None
    top_p_fallback: Optional[float] = None
    max_time: Optional[int] = None
    return_full_text: Optional[bool] = None
    tokenizer_config: Optional[Dict[str, Any]] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    # Domain specialization for the model
    domain_specialization: Optional[List[str]] = None
    # Problem type specialization for the model
    problem_type_specialization: Optional[List[str]] = None
    # Performance metrics for the model
    performance_metrics: Optional[Dict[str, float]] = None
    # Cost per token for the model
    cost_per_token: Optional[float] = None

@dataclasses.dataclass
class Team:
    """
    A user-defined group of AI models assigned to a specific role within the workflow.

    Attributes:
        name (str): A unique name for the team.
        role (Literal["Blue", "Red", "Gold"]): Specifies the team's primary function (e.g., creation, critique, evaluation).
        members (List[ModelConfig]): A list of `ModelConfig` objects defining the AI models that comprise this team.
        description (Optional[str]): An optional human-readable description of the team's purpose or specialization.
        content_analysis_system_prompt (Optional[str]): System prompt for content analysis if this team is used for it.
        content_analysis_user_prompt_template (Optional[str]): User prompt template for content analysis if this team is used for it.
        decomposition_system_prompt (Optional[str]): System prompt for decomposition if this team is used for it.
        decomposition_user_prompt_template (Optional[str]): User prompt template for decomposition if this team is used for it.
        solver_system_prompt (Optional[str]): System prompt for solvers if this team is used for generating solutions.
        solver_user_prompt_template (Optional[str]): User prompt template for solvers if this team is used for generating solutions.
        patcher_system_prompt (Optional[str]): System prompt for patchers if this team is used for fixing rejected solutions.
        patcher_user_prompt_template (Optional[str]): User prompt template for patchers if this team is used for fixing rejected solutions.
        assembler_system_prompt (Optional[str]): System prompt for assemblers if this team is used for reassembling the final solution.
        assembler_user_prompt_template (Optional[str]): User prompt template for assemblers if this team is used for reassembling the final solution.
        red_team_system_prompt (Optional[str]): System prompt for Red Teams when performing critiques.
        red_team_user_prompt_template (Optional[str]): User prompt template for Red Teams when performing critiques.
        gold_team_system_prompt (Optional[str]): System prompt for Gold Teams when performing verifications.
        gold_team_user_prompt_template (Optional[str]): User prompt template for Gold Teams when performing verifications.
    """
    name: str
    role: Literal["Blue", "Red", "Gold"]
    members: List[ModelConfig]
    description: Optional[str] = None
    content_analysis_system_prompt: Optional[str] = None
    content_analysis_user_prompt_template: Optional[str] = None
    decomposition_system_prompt: Optional[str] = None
    decomposition_user_prompt_template: Optional[str] = None
    solver_system_prompt: Optional[str] = None
    solver_user_prompt_template: Optional[str] = None
    patcher_system_prompt: Optional[str] = None
    patcher_user_prompt_template: Optional[str] = None
    assembler_system_prompt: Optional[str] = None
    assembler_user_prompt_template: Optional[str] = None
    red_team_system_prompt: Optional[str] = None
    red_team_user_prompt_template: Optional[str] = None
    gold_team_system_prompt: Optional[str] = None
    gold_team_user_prompt_template: Optional[str] = None
    # Sub-role for the team (e.g., "Planner", "Solver", "Patcher" for Blue teams)
    sub_role: Optional[str] = None
    # Domain specialization for the team
    domain_specialization: Optional[List[str]] = None
    # Problem type specialization for the team
    problem_type_specialization: Optional[List[str]] = None
    # Performance metrics for the team
    performance_metrics: Optional[Dict[str, float]] = None
    # Team configuration parameters
    team_config: Optional[Dict[str, Any]] = None

@dataclasses.dataclass
class GauntletRoundRule:
    """Defines the specific rules and criteria for a single round within a Gauntlet."""
    round_number: int
    quorum_required_approvals: int
    quorum_from_panel_size: int
    min_overall_confidence: float = 0.0
    max_score_variance: Optional[float] = None
    per_judge_requirements: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)
    collaboration_mode: Literal["independent", "share_previous_feedback"] = "independent"
    # Time constraints for this round
    time_limit_seconds: Optional[int] = None
    # Resource constraints for this round
    max_api_calls: Optional[int] = None
    max_tokens: Optional[int] = None
    # Adaptive rules for this round
    adaptive_rules: Optional[Dict[str, Any]] = None

@dataclasses.dataclass
class GauntletDefinition:
    """A programmable, multi-round process that a piece of content must pass to be approved."""
    name: str
    team_name: str
    rounds: List[GauntletRoundRule]
    description: Optional[str] = None
    attack_modes: List[str] = dataclasses.field(default_factory=list)
    generation_mode: Literal["single_candidate", "multi_candidate_peer_review", "evolutionary", "hybrid"] = "single_candidate"
    # Gauntlet type
    gauntlet_type: Literal["standard", "adaptive", "hierarchical", "competitive", "collaborative"] = "standard"
    # Performance metrics for the gauntlet
    performance_metrics: Optional[Dict[str, float]] = None
    # Gauntlet configuration parameters
    gauntlet_config: Optional[Dict[str, Any]] = None

@dataclasses.dataclass
class SubProblem:
    """Represents a single sub-problem in the decomposition plan, with its own configurations."""
    id: str
    description: str
    dependencies: List[str] = dataclasses.field(default_factory=list)
    ai_suggested_evolution_mode: str = "standard"
    ai_suggested_complexity_score: int = 5
    ai_suggested_evaluation_prompt: str = ""
    content_type: str = "text_general" # New: Content type for the sub-problem's solution
    solver_team_name: str = ""
    red_team_gauntlet_name: Optional[str] = None
    gold_team_gauntlet_name: str = ""
    solver_generation_gauntlet_name: Optional[str] = None
    patcher_team_name: str = ""  # Name of the Blue Team assigned to patch solutions for this sub-problem
    evolution_params: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # AI suggestions (additional fields from document)
    ai_suggested_team_assignment: Optional[str] = None
    ai_suggested_gauntlet_assignment: Optional[Dict[str, str]] = None
    estimated_resources: Optional[Dict[str, Any]] = None
    potential_approaches: Optional[List[str]] = None
    # Sub-problem status
    status: Literal["pending", "in_progress", "solved", "failed", "requires_rework"] = "pending"
    # Solution attempts for this sub-problem
    solution_attempts: List['SolutionAttempt'] = dataclasses.field(default_factory=list)
    # Performance metrics for this sub-problem
    performance_metrics: Optional[Dict[str, float]] = None

@dataclasses.dataclass
class DecompositionPlan:
    """The overall plan for decomposing and solving a complex problem, including global configurations."""
    problem_statement: str
    analyzed_context: Dict[str, Any]
    sub_problems: List[SubProblem]
    max_refinement_loops: int = 3
    auto_approval_enabled: bool = False
    auto_approval_criteria: Optional[Dict[str, Any]] = None
    # Resource limits for the workflow
    resource_limits: Optional[Dict[str, Any]] = None
    # Parallel processing configuration
    parallel_processing_enabled: bool = False
    max_parallel_sub_problems: int = 1
    # Learning configuration
    learning_enabled: bool = False
    learning_config: Optional[Dict[str, Any]] = None
    # Teams and Gauntlets for final stages
    content_analyzer_team_name: str = ""
    planner_team_name: str = ""
    assembler_team_name: str = ""
    final_red_team_gauntlet_name: Optional[str] = None
    final_gold_team_gauntlet_name: str = ""

@dataclasses.dataclass
class SolutionAttempt:
    """Represents a candidate solution for a sub-problem or the final solution at a given point in time."""
    sub_problem_id: str
    content: str
    generated_by_model: str
    timestamp: float
    history: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    # Solution metadata
    solution_type: Optional[str] = None
    solution_approach: Optional[str] = None
    # Solution quality metrics
    quality_metrics: Optional[Dict[str, float]] = None
    # Resource usage for this solution attempt
    resource_usage: Optional[Dict[str, Any]] = None
    # Solution status
    status: Literal["generated", "critiqued", "verified", "rejected", "patched"] = "generated"
    # Related critiques and verifications
    critique_reports: List['CritiqueReport'] = dataclasses.field(default_factory=list)
    verification_reports: List['VerificationReport'] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class CritiqueReport:
    """Report generated by a Red Team Gauntlet, detailing identified flaws and overall robustness."""
    solution_attempt_id: str
    gauntlet_name: str
    is_approved: bool
    reports_by_judge: List[Dict[str, Any]]
    summary: str = ""
    # Critique metadata
    critique_timestamp: float = dataclasses.field(default_factory=time.time)
    # Critique metrics
    overall_score: float = 0.0
    flaw_severity_scores: Dict[str, float] = dataclasses.field(default_factory=dict)
    # Identified flaws
    identified_flaws: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    # Suggested improvements
    suggested_improvements: List[str] = dataclasses.field(default_factory=list)
    # Resource usage for this critique
    resource_usage: Optional[Dict[str, Any]] = None

@dataclasses.dataclass
class VerificationReport:
    """Report generated by a Gold Team Gauntlet, detailing verification results and confidence."""
    solution_attempt_id: str
    gauntlet_name: str
    is_approved: bool
    reports_by_judge: List[Dict[str, Any]]
    average_score: float = 0.0
    score_variance: float = 0.0
    summary: str = ""
    # Verification metadata
    verification_timestamp: float = dataclasses.field(default_factory=time.time)
    # Verification metrics
    dimension_scores: Dict[str, float] = dataclasses.field(default_factory=dict)
    # Verification criteria
    criteria_met: List[str] = dataclasses.field(default_factory=list)
    criteria_not_met: List[str] = dataclasses.field(default_factory=list)
    # Targeted feedback
    targeted_feedback: Optional[str] = None
    # Resource usage for this verification
    resource_usage: Optional[Dict[str, Any]] = None

# --- Knowledge Management ---

@dataclasses.dataclass
class KnowledgeArtifact:
    """Represents a piece of knowledge extracted from a workflow execution."""
    id: str  # Unique identifier for this knowledge artifact
    artifact_type: Literal["solution_pattern", "problem_solution_mapping", "critique_insight", "team_performance", "gauntlet_effectiveness"]
    content: Dict[str, Any]  # Content of the knowledge artifact
    source_workflow_id: str  # ID of the workflow this artifact was extracted from
    extraction_timestamp: float = dataclasses.field(default_factory=time.time)
    # Artifact metadata
    domain: Optional[str] = None
    problem_type: Optional[str] = None
    # Artifact usage metrics
    usage_count: int = 0
    effectiveness_score: float = 0.0
    # Artifact relationships
    related_artifacts: List[str] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class PerformanceMetrics:
    """Represents performance metrics for a team, gauntlet, or workflow."""
    entity_type: Literal["team", "gauntlet", "workflow"]
    entity_id: str
    metrics: Dict[str, float]
    timestamp: float = dataclasses.field(default_factory=time.time)
    # Metrics metadata
    domain: Optional[str] = None
    problem_type: Optional[str] = None
    # Metrics context
    context: Optional[Dict[str, Any]] = None

# --- Workflow State Management ---

@dataclasses.dataclass
class WorkflowState:
    """Manages the state of an active Sovereign-Grade Decomposition Workflow."""
    workflow_id: str
    workflow_type: Any
    problem_statement: str
    current_stage: str
    current_sub_problem_id: Optional[str] = None
    current_gauntlet_name: Optional[str] = None
    status: str = "running"
    progress: float = 0.0
    start_time: float = dataclasses.field(default_factory=time.time)
    end_time: Optional[float] = None
    decomposition_plan: Optional[DecompositionPlan] = None
    sub_problem_solutions: Dict[str, SolutionAttempt] = dataclasses.field(default_factory=dict)
    solved_sub_problem_ids: Set[str] = dataclasses.field(default_factory=set)
    rejected_sub_problems: Dict[str, Any] = dataclasses.field(default_factory=dict)
    final_solution: Optional[SolutionAttempt] = None
    refinement_loop_count: int = 0
    
    # Store the specific teams and gauntlets used for THIS workflow run.
    # This ensures consistency even if global definitions in TeamManager/GauntletManager change later.
    content_analyzer_team: Optional[Team] = None
    planner_team: Optional[Team] = None
    solver_team: Optional[Team] = None
    patcher_team: Optional[Team] = None
    assembler_team: Optional[Team] = None
    sub_problem_red_gauntlet: Optional[GauntletDefinition] = None
    sub_problem_gold_gauntlet: Optional[GauntletDefinition] = None
    final_red_gauntlet: Optional[GauntletDefinition] = None
    final_gold_gauntlet: Optional[GauntletDefinition] = None
    max_refinement_loops: int = 3 # Max iterations for the self-healing loop
    all_critique_reports: List[CritiqueReport] = dataclasses.field(default_factory=list)
    all_verification_reports: List[VerificationReport] = dataclasses.field(default_factory=list)
    # Resource usage for the workflow
    resource_usage: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # Performance metrics for the workflow
    performance_metrics: Dict[str, float] = dataclasses.field(default_factory=dict)
    # Knowledge artifacts extracted from the workflow
    knowledge_artifacts: List[KnowledgeArtifact] = dataclasses.field(default_factory=list)

    # OpenEvolve Parameters (User-configurable via UI) - Complete set
    max_iterations: int = 100
    population_size: int = 50
    num_islands: int = 5
    migration_interval: int = 10
    migration_rate: float = 0.1
    archive_size: int = 10
    elite_ratio: float = 0.1
    exploration_ratio: float = 0.7
    exploitation_ratio: float = 0.3
    checkpoint_interval: int = 10
    feature_dimensions: List[str] = dataclasses.field(default_factory=lambda: ["complexity", "diversity"])
    feature_bins: int = 10
    diversity_metric: str = "edit_distance"
    enable_artifacts: bool = True
    cascade_evaluation: bool = False
    cascade_thresholds: List[float] = dataclasses.field(default_factory=lambda: [0.5, 0.75, 0.9])
    use_llm_feedback: bool = True
    llm_feedback_weight: float = 0.5
    parallel_evaluations: int = 4
    distributed: bool = False
    template_dir: str = "./templates"
    num_top_programs: int = 5
    num_diverse_programs: int = 5
    use_template_stochasticity: bool = True
    template_variations: Dict[str, List[str]] = dataclasses.field(default_factory=dict)
    use_meta_prompting: bool = True
    meta_prompt_weight: float = 0.5
    include_artifacts: bool = True
    max_artifact_bytes: int = 1048576 # 1 MB
    artifact_security_filter: bool = True
    early_stopping_patience: int = 10
    convergence_threshold: float = 0.01
    early_stopping_metric: str = "fitness"
    memory_limit_mb: int = 2048
    cpu_limit: float = 1.0
    random_seed: Optional[int] = None
    db_path: str = "./openevolve.db"
    in_memory: bool = False
    diff_based_evolution: bool = False
    max_code_length: int = 10000
    evolution_trace_enabled: bool = False
    evolution_trace_format: str = "json"
    evolution_trace_include_code: bool = False
    evolution_trace_include_prompts: bool = False
    evolution_trace_output_path: str = "./evolution_trace"
    evolution_trace_buffer_size: int = 100
    evolution_trace_compress: bool = False
    log_level: str = "INFO"
    log_dir: str = "./logs"
    api_timeout: int = 60
    api_retries: int = 3
    api_retry_delay: float = 1.0
    artifact_size_threshold: int = 5242880 # 5 MB
    cleanup_old_artifacts: bool = True
    artifact_retention_days: int = 7
    diversity_reference_size: int = 100
    max_retries_eval: int = 3
    evaluator_timeout: int = 30
    evaluator_models: Optional[List[Dict[str, Any]]] = None
    
    # Advanced OpenEvolve research features
    double_selection: bool = False
    adaptive_feature_dimensions: bool = False
    test_time_compute: bool = False
    optillm_integration: bool = False
    plugin_system: bool = False
    hardware_optimization: bool = False
    multi_strategy_sampling: bool = False
    ring_topology: bool = False
    controlled_gene_flow: bool = False
    auto_diff: bool = False
    symbolic_execution: bool = False
    coevolutionary_approach: bool = False
    
    # Additional LLM parameters for comprehensive control
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    reasoning_effort: Optional[str] = None
    system_message: str = ""
    evaluator_system_message: str = ""
    
    # Quality Diversity specific parameters
    qd_algorithm: str = "map_elites"  # map_elites, cvt_map_elites, novelty_search
    qd_selection_pressure: float = 0.8
    qd_mutation_rate: float = 0.1
    qd_crossover_rate: float = 0.7
    qd_novelty_threshold: float = 0.1
    qd_archive_threshold: float = 0.5
    
    # Multi-objective specific parameters
    mo_algorithm: str = "nsga2"  # nsga2, spea2, moea_d
    mo_crossover_prob: float = 0.9
    mo_mutation_prob: float = 0.1
    mo_tournament_size: int = 3
    mo_reference_point: Optional[List[float]] = None
    
    # Adversarial specific parameters
    adversarial_rounds: int = 5
    adversarial_attack_budget: int = 10
    adversarial_defense_budget: int = 10
    adversarial_success_threshold: float = 0.8
    
    # Prompt optimization specific parameters
    prompt_max_length: int = 2000
    prompt_min_length: int = 50
    prompt_optimization_target: str = "performance"  # performance, brevity, clarity
    prompt_evaluation_samples: int = 10
    
    # Code evolution specific parameters
    code_language: str = "python"
    code_style_guide: str = "pep8"
    code_complexity_limit: int = 15
    code_test_coverage_target: float = 0.8
    code_security_scan: bool = True
    
    # Document evolution specific parameters
    document_max_words: int = 5000
    document_min_words: int = 100
    document_readability_target: str = "college"  # elementary, middle, high_school, college, graduate
    document_tone: str = "professional"  # casual, professional, academic, technical
    document_format: str = "markdown"  # markdown, html, latex, plain_text
