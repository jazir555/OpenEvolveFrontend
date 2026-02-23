export type TeamRole = "Blue" | "Red" | "Gold";
export type TeamType = "standard" | "swarm" | "sovereign";
export interface ModelConfig {
    model_id: string;
    api_key: string;
    api_base?: string;
    temperature?: number;
    top_p?: number;
    max_tokens?: number;
    frequency_penalty?: number;
    presence_penalty?: number;
    seed?: number | null;
    n?: number | null;
    logit_bias?: Record<number, number> | null;
    reasoning_effort?: string | null;
    stop_sequences?: string[] | null;
    logprobs?: boolean | null;
    top_logprobs?: number | null;
    response_format?: Record<string, string> | null;
    stream?: boolean | null;
    user?: string | null;
    max_retries?: number;
    timeout?: number;
    organization?: string | null;
    response_model?: string | null;
    tools?: Array<Record<string, unknown>> | null;
    tool_choice?: unknown | null;
    system_fingerprint?: string | null;
    deployment_id?: string | null;
    encoding_format?: string | null;
    max_input_tokens?: number | null;
    stop_token?: string | null;
    best_of?: number | null;
    logprobs_offset?: number | null;
    suffix?: string | null;
    presence_penalty_range?: number[] | null;
    frequency_penalty_range?: number[] | null;
    stop_token_id?: number | null;
    response_json_format?: boolean | null;
    max_output_tokens?: number | null;
    stream_options?: Record<string, unknown> | null;
    logprobs_type?: string | null;
    top_k?: number | null;
    repetition_penalty?: number | null;
    length_penalty?: number | null;
    early_stopping?: boolean | null;
    num_beams?: number | null;
    do_sample?: boolean | null;
    temperature_fallback?: number | null;
    top_p_fallback?: number | null;
    max_time?: number | null;
    return_full_text?: boolean | null;
    tokenizer_config?: Record<string, unknown> | null;
    model_kwargs?: Record<string, unknown> | null;
    domain_specialization?: string[] | null;
    problem_type_specialization?: string[] | null;
    performance_metrics?: Record<string, number> | null;
    cost_per_token?: number | null;
}
export interface Team {
    name: string;
    role: TeamRole;
    members: ModelConfig[];
    tenant_id?: string | null;
    description?: string | null;
    content_analysis_system_prompt?: string | null;
    content_analysis_user_prompt_template?: string | null;
    decomposition_system_prompt?: string | null;
    decomposition_user_prompt_template?: string | null;
    solver_system_prompt?: string | null;
    solver_user_prompt_template?: string | null;
    patcher_system_prompt?: string | null;
    patcher_user_prompt_template?: string | null;
    assembler_system_prompt?: string | null;
    assembler_user_prompt_template?: string | null;
    red_team_system_prompt?: string | null;
    red_team_user_prompt_template?: string | null;
    gold_team_system_prompt?: string | null;
    gold_team_user_prompt_template?: string | null;
    sub_role?: string | null;
    domain_specialization?: string[] | null;
    problem_type_specialization?: string[] | null;
    performance_metrics?: Record<string, number> | null;
    team_config?: Record<string, unknown> | null;
    openevolve_metrics?: Array<Record<string, unknown>> | null;
    team_type?: TeamType;
}
export interface TeamSummary {
    name: string;
    role: TeamRole;
    description?: string | null;
    member_count: number;
}
export type CollaborationMode = "independent" | "share_previous_feedback";
export type VotingStrategy = "fixed_quorum" | "first_to_ahead_by_k";
export interface GauntletRoundRule {
    round_number: number;
    quorum_required_approvals: number;
    quorum_from_panel_size: number;
    min_overall_confidence?: number;
    max_score_variance?: number | null;
    per_judge_requirements?: Record<string, Record<string, unknown>>;
    collaboration_mode?: CollaborationMode;
    time_limit_seconds?: number | null;
    max_api_calls?: number | null;
    max_tokens?: number | null;
    adaptive_rules?: Record<string, unknown> | null;
    voting_strategy?: VotingStrategy;
    margin_k?: number | null;
    max_dynamic_votes?: number | null;
    required_mathematical_properties?: string[];
    proof_obligation_threshold?: number;
    mathematical_complexity_level?: number;
    proof_generation_enabled?: boolean;
    proof_verification_enabled?: boolean;
    mathematical_approach?: string;
    verification_timeout?: number;
    proof_storage_enabled?: boolean;
    mathematical_quality_threshold?: number;
}
export type GenerationMode = "single_candidate" | "multi_candidate_peer_review" | "evolutionary" | "hybrid";
export type GauntletType = "standard" | "adaptive" | "hierarchical" | "competitive" | "collaborative";
export interface GauntletDefinition {
    name: string;
    team_name: string;
    rounds: GauntletRoundRule[];
    tenant_id?: string | null;
    description?: string | null;
    attack_modes?: string[];
    generation_mode?: GenerationMode;
    gauntlet_type?: GauntletType;
    performance_metrics?: Record<string, number> | null;
    gauntlet_config?: Record<string, unknown> | null;
    red_flags?: Record<string, unknown>;
    formal_verification_enabled?: boolean;
    verification_methods?: string[];
    mathematical_requirements?: Record<string, unknown>;
    proof_generation_enabled?: boolean;
    automatic_formalization?: boolean;
    formal_verification_threshold?: number;
    lean_verification_config?: Record<string, unknown>;
}
export interface GauntletSummary {
    name: string;
    team_name: string;
    description?: string | null;
    round_count: number;
}
export interface WorkflowSummary {
    workflow_id: string;
    status: string;
    current_stage: string;
    progress: number;
}
export interface WorkflowDetail {
    workflow_id: string;
    problem_statement: string;
    status: string;
    current_stage: string;
    progress: number;
    start_time: number;
    end_time?: number | null;
    refinement_loop_count: number;
    solved_sub_problems: number;
    total_sub_problems: number;
}
export interface WorkflowCreateRequest {
    problem_statement: string;
    content_analyzer_team: string;
    planner_team: string;
    solver_team: string;
    patcher_team: string;
    assembler_team: string;
    sub_problem_red_gauntlet: string;
    sub_problem_gold_gauntlet: string;
    final_red_gauntlet: string;
    final_gold_gauntlet: string;
    solver_generation_gauntlet: string;
    max_refinement_loops?: number;
    mdap_enabled?: boolean;
    mdap_config?: Record<string, unknown>;
    maker_enabled?: boolean;
    maker_config?: Record<string, unknown>;
}
export interface WorkflowCreateResponse {
    workflow_id: string;
    status: string;
    current_stage: string;
    progress: number;
    created_at: string;
}
export interface WorkflowResults {
    workflow_id: string;
    problem_statement: string;
    status: string;
    final_solution?: {
        content: string;
        generated_by?: string;
        timestamp?: string;
    } | null;
    sub_problem_solutions: Record<string, {
        content: string;
        generated_by?: string;
        timestamp?: string;
    }>;
    execution_time?: number | null;
    refinement_loops: number;
}
export interface StatisticsSummary {
    total_workflows: number;
    completed: number;
    failed: number;
    running: number;
    total_teams: number;
    total_gauntlets: number;
}
export interface AuditLogEntry {
    timestamp?: string;
    user?: string;
    operation?: string;
    resource?: string;
    resource_id?: string;
    success?: boolean;
    details?: Record<string, unknown>;
}
export interface IcrOverview {
    icr_enabled: boolean;
    total_patterns: number;
    overall_success_rate: number;
    active_components: number;
    total_refinements: number;
}
export type IcrComponents = Record<string, {
    total_patterns?: number;
    overall_pass_rate?: number;
    active?: boolean;
    recent_pass_rate?: number;
    recent_fail_rate?: number;
    top_failure_modes?: string[];
}>;
export interface IcrRefinements {
    events: Array<Record<string, unknown>>;
    total_count: number;
}
export type TaskStatus = "To Do" | "In Progress" | "On Hold" | "Completed";
export interface TaskItem {
    id: string;
    title: string;
    description?: string | null;
    assignee?: string | null;
    status: TaskStatus;
    due_date?: string | null;
    created_at: string;
}
export interface AdaptiveMdapDashboard {
    generated_at?: string;
    summary?: Record<string, number>;
    execution?: Record<string, Record<string, number>>;
    costs?: Record<string, Record<string, number>>;
    allocations?: Record<string, number>;
    complexity_distribution?: Record<string, number>;
}
export interface AdaptiveMdapProfiles {
    profiles: Record<string, string>;
    default: string;
}
export interface KnowledgeArtifact {
    id: string;
    artifact_type: string;
    content: string | Record<string, unknown>;
    source_workflow_id: string;
    extraction_timestamp: string | number;
    domain?: string | null;
    problem_type?: string | null;
    usage_count: number;
    effectiveness_score: number;
    related_artifacts?: string[];
}
export interface KnowledgeGraphNode {
    id: string;
    type?: string;
    domain?: string | null;
    usage?: number;
}
export interface KnowledgeGraphEdge {
    source: string;
    target: string;
}
export interface KnowledgeGraph {
    nodes: KnowledgeGraphNode[];
    edges: KnowledgeGraphEdge[];
}
export interface KnowledgeStats {
    total_artifacts: number;
    total_usage: number;
    average_effectiveness: number;
    by_type: Record<string, number>;
}
export interface PerformanceMetric {
    entity_type: string;
    entity_id: string;
    metrics: Record<string, number>;
    timestamp?: string | null;
    domain?: string | null;
    problem_type?: string | null;
    context?: Record<string, unknown> | null;
}
export interface AnalyticsKnowledgeStats {
    total_artifacts: number;
    total_usage: number;
    avg_effectiveness: number;
    artifact_type_distribution: Record<string, number>;
    domain_distribution: Record<string, number>;
    top_used_artifacts: Array<Record<string, unknown>>;
    top_effective_artifacts: Array<Record<string, unknown>>;
}
export interface MonitoringDashboardMetrics {
    timestamp: string;
    system: {
        system?: Record<string, number>;
    };
    health: {
        status: string;
        healthy: boolean;
        timestamp: string;
        uptime_seconds: number;
    };
    workflow: Record<string, number>;
    recent_metrics: Record<string, Record<string, unknown>>;
}
export interface MonitoringAlert {
    name?: string;
    metric_name?: string;
    condition?: string;
    threshold?: number;
    description?: string;
    active?: boolean;
    triggered?: boolean;
    latest_value?: number;
}
export interface MonitoringMetric {
    name?: string;
    value?: number;
    type?: string;
    labels?: Record<string, unknown>;
    timestamp?: string | null;
    description?: string | null;
}
export interface MonitoringService {
    name: string;
    status?: string;
    healthy?: boolean;
    execution_time?: number;
    timestamp?: string;
    error?: string | null;
}
export interface MonitoringLogEntry {
    source: string;
    line: string;
}
export interface AnalyticsWorkflowMetric {
    timestamp?: string;
    workflow_id: string;
    status?: string;
    progress?: number;
    best_fitness?: number | null;
    avg_fitness?: number | null;
    diversity?: number | null;
    tokens_used?: number | null;
    execution_time?: number | null;
    memory_usage?: number | null;
    cpu_usage?: number | null;
    population_size?: number | null;
    generation?: number | null;
    metrics?: Record<string, unknown>;
}
export interface WorkflowTelemetry {
    workflow_id: string;
    workflow_type?: string;
    status?: string;
    current_stage?: string;
    progress?: number;
    start_time?: number;
    end_time?: number | null;
    execution_time_seconds?: number | null;
    refinement_loop_count?: number;
    resource_usage?: Record<string, unknown>;
    performance_metrics?: Record<string, unknown>;
    openevolve_metrics?: Record<string, unknown>;
    crewai_workflow_id?: string | null;
    gauntlet_summary?: {
        critique_total: number;
        critique_approved: number;
        critique_avg_score: number;
        verification_total: number;
        verification_approved: number;
        verification_avg_score: number;
    };
}
export interface CrewAIWorkflowSummary {
    workflow_id: string;
    problem_statement?: string;
    phase?: number;
    status?: string;
    execution_method?: string;
    created_at?: string;
    updated_at?: string;
    has_decomposition_plan?: boolean;
    num_sub_solutions?: number;
    num_critiques?: number;
    num_verification_results?: number;
    has_reassembly_result?: boolean;
    has_final_validation?: boolean;
}
export interface CrewAIWorkflowTicket {
    id: string;
    title?: string;
    description?: string;
    status?: string;
    assigned_agent_id?: string | null;
    created_at?: string | number;
    updated_at?: string | number;
    sub_problem_id?: string;
    dependencies?: string[];
    priority?: number | null;
}
export interface WorkflowSubProblem {
    id: string;
    description: string;
    dependencies: string[];
    ai_suggested_evolution_mode?: string;
    ai_suggested_complexity_score?: number;
    ai_suggested_evaluation_prompt?: string;
    content_type?: string;
    solver_team_name?: string;
    red_team_gauntlet_name?: string | null;
    gold_team_gauntlet_name?: string | null;
    solver_generation_gauntlet_name?: string | null;
    patcher_team_name?: string | null;
    evolution_params?: Record<string, unknown>;
    status?: string;
    atomic_mode?: boolean;
    decomposition_depth?: number;
    acceptance_criteria?: string[];
    solution_requirements?: Record<string, unknown>;
    specific_constraints?: string[];
    dependency_outputs?: Record<string, unknown>;
    metadata?: Record<string, unknown>;
}
export interface WorkflowDecompositionPlan {
    problem_statement: string;
    analyzed_context: Record<string, unknown>;
    sub_problems: WorkflowSubProblem[];
    max_refinement_loops?: number;
    auto_approval_enabled?: boolean;
    auto_approval_criteria?: Record<string, unknown> | null;
    mdap_enabled?: boolean;
    mdap_config?: Record<string, unknown>;
    maker_enabled?: boolean;
    maker_config?: Record<string, unknown>;
    resource_limits?: Record<string, unknown> | null;
    parallel_processing_enabled?: boolean;
    max_parallel_sub_problems?: number;
    learning_enabled?: boolean;
    learning_config?: Record<string, unknown> | null;
    content_analyzer_team_name?: string;
    planner_team_name?: string;
    assembler_team_name?: string;
    final_red_team_gauntlet_name?: string | null;
    final_gold_team_gauntlet_name?: string | null;
    metadata?: Record<string, unknown>;
}
export interface WorkflowDependencyGraph {
    edges: Record<string, string[]>;
    execution_order?: string[];
}
export interface WorkflowPlanResponse {
    workflow_id: string;
    plan: WorkflowDecompositionPlan;
    dependency_graph: WorkflowDependencyGraph;
}
export interface SovereignSubProblem {
    id: string;
    parent_id: string;
    title: string;
    description: string;
    type: string;
    complexity_score?: Record<string, unknown>;
    dependencies?: string[];
    success_criteria?: Array<Record<string, unknown>>;
    validation_gauntlet?: string;
    assigned_team?: string | null;
    estimated_effort?: number;
    priority?: number;
    status?: string;
    created_at?: string;
    updated_at?: string;
    metadata?: Record<string, unknown>;
}
export interface DependencyGraph {
    nodes: Record<string, SovereignSubProblem>;
    edges: Record<string, string[]>;
    critical_path?: string[];
    parallel_groups?: string[][];
    execution_order?: string[];
    metadata?: Record<string, unknown>;
}
export interface SovereignPlan {
    id: string;
    problem_id: string;
    strategy: string;
    sub_problems: SovereignSubProblem[];
    dependency_graph?: DependencyGraph | null;
    validation_checkpoints?: Array<Record<string, unknown>>;
    quality_scores?: Record<string, unknown> | null;
    confidence_level?: number;
    created_by?: string;
    approved_by?: string | null;
    status?: string;
    created_at?: string;
    updated_at?: string;
    metadata?: Record<string, unknown>;
    error_message?: string | null;
}
export interface KnowledgeRecommendations {
    recommended_approaches: Array<Record<string, unknown>>;
    similar_problems: Array<Record<string, unknown>>;
    team_recommendations: Array<Record<string, unknown>>;
    gauntlet_recommendations: Array<Record<string, unknown>>;
}
export type PromptMap = Record<string, string>;
export interface ContentTemplate {
    name: string;
    content: string;
    source?: "builtin" | "custom";
}
export interface ProtocolValidationResult {
    valid: boolean;
    score: number;
    errors: string[];
    warnings: string[];
    suggestions: string[];
    [key: string]: unknown;
}
export type AutoApprovalAction = "approve" | "reject" | "escalate";
export interface AutoApprovalCondition {
    field: string;
    operator: string;
    value: string | number;
    logical_op?: "AND" | "OR";
}
export interface AutoApprovalRule {
    name: string;
    priority: number;
    action: AutoApprovalAction;
    enabled: boolean;
    conditions: AutoApprovalCondition[];
    created_at?: string;
}
export interface AutoApprovalConfig {
    enabled: boolean;
    rules: AutoApprovalRule[];
}
export interface AutoApprovalTestResult {
    rule_name: string;
    action: AutoApprovalAction;
    matched: boolean;
}
export interface AutoApprovalAuditEntry {
    timestamp: string;
    rule_name: string;
    action: AutoApprovalAction;
    matched: boolean;
    plan: Record<string, unknown>;
}
export interface WorkflowTemplate {
    id: string;
    name: string;
    description?: string;
    version?: string;
    config: Record<string, unknown>;
    usage_count?: number;
    created_at?: string;
    updated_at?: string;
    tags?: string[];
}
export interface ProviderSummary {
    id: string;
    name: string;
    api_base?: string | null;
    models_endpoint?: string | null;
    default_model?: string | null;
}
export interface VersionEntry {
    id: string;
    name: string;
    timestamp: string;
    protocol_text: string;
    comment?: string;
    author?: string;
    complexity_metrics?: Record<string, unknown>;
    structure_analysis?: Record<string, unknown>;
    branch_from?: string;
    branch_name?: string;
}
export interface VersionCompareResult {
    version1: string;
    version2: string;
    chars_added: number;
    chars_removed: number;
    total_chars_change: number;
    complexity_diff?: Record<string, unknown>;
    error?: string;
}
export interface ValidationRule {
    max_length?: number;
    min_length?: number;
    required_keywords?: string[];
    forbidden_patterns?: string[];
    required_sections?: string[];
}
export interface ValidationRuleResult {
    valid: boolean;
    errors: string[];
    warnings: string[];
    suggestions: string[];
    rule_name: string;
    rule_config: ValidationRule;
}
export interface ValidationRunResult {
    content_length: number;
    validations: Record<string, ValidationRuleResult>;
    overall_result: boolean;
    error_count: number;
    warning_count: number;
    suggestion_count: number;
}
export interface ComplianceCheckResult extends ValidationRuleResult {
}
export interface WorkflowDefinitionSummary {
    id: string;
    name: string;
    description?: string;
    workflow_type: string;
    created_at?: string;
}
export interface WorkflowDefinitionDetail extends WorkflowDefinitionSummary {
    parameters?: Record<string, unknown>;
}
export interface WorkflowInstanceSummary {
    instance_id: string;
    workflow_type?: string;
    status: string;
    current_stage?: string;
    problem_statement?: string;
    start_time?: number | null;
    end_time?: number | null;
    progress?: number;
}
export interface WorkflowInstanceStatus {
    instance_id: string;
    status: string;
    current_stage?: string;
    progress?: number;
    start_time?: number | null;
    end_time?: number | null;
    execution_time?: number | null;
    error_message?: string | null;
}
export interface WorkflowInstanceDetail {
    status: WorkflowInstanceStatus;
    parameters: Record<string, unknown>;
}
export interface ParameterDefinition {
    name: string;
    type: string;
    default: unknown;
    description: string;
    category: string;
    min_value?: number | null;
    max_value?: number | null;
    options?: string[] | null;
    required?: boolean;
}
export interface ParameterValidationResult {
    valid: boolean;
    errors: string[];
    warnings: string[];
}
export declare const createDefaultModelConfig: () => ModelConfig;
export interface EvaluatorListResponse {
    evaluators: Record<string, string>;
}
export interface EvaluatorUploadResponse {
    evaluator_id: string;
}
export interface WorkflowPlanUpdateRequest {
    sub_problems: WorkflowSubProblem[];
    max_refinement_loops?: number;
    auto_approval_enabled?: boolean;
    auto_approval_criteria?: Record<string, unknown> | null;
    mdap_enabled?: boolean;
    mdap_config?: Record<string, unknown>;
    maker_enabled?: boolean;
    maker_config?: Record<string, unknown>;
    resource_limits?: Record<string, unknown> | null;
    parallel_processing_enabled?: boolean;
    max_parallel_sub_problems?: number;
    learning_enabled?: boolean;
    learning_config?: Record<string, unknown> | null;
    content_analyzer_team_name?: string;
    planner_team_name?: string;
    assembler_team_name?: string;
    final_red_team_gauntlet_name?: string | null;
    final_gold_team_gauntlet_name?: string | null;
    metadata?: Record<string, unknown>;
}
export interface ResourceUsageSummary {
    api_calls?: number;
    tokens_used?: number;
    estimated_cost?: number;
    execution_time_seconds?: number;
    memory_usage_mb?: number;
    limits?: Record<string, unknown>;
    component_breakdown?: Record<string, Record<string, unknown>>;
}
export interface WorkflowResourceUsageResponse {
    workflow_id: string;
    resource_usage: ResourceUsageSummary;
}
export interface WorkflowResourceOptimizationResponse {
    workflow_id: string;
    suggestions: Record<string, unknown>;
}
export interface IntegratedWorkflowRequest {
    current_content: string;
    content_type?: string;
    api_key: string;
    base_url?: string;
    red_team_models: string[];
    blue_team_models: string[];
    evaluator_models: string[];
    max_iterations?: number;
    adversarial_iterations?: number;
    evolution_iterations?: number;
    evaluation_iterations?: number;
    system_prompt: string;
    evaluator_system_prompt: string;
    temperature?: number;
    top_p?: number;
    frequency_penalty?: number;
    presence_penalty?: number;
    max_tokens?: number;
    seed?: number | null;
    rotation_strategy?: string;
    red_team_sample_size?: number;
    blue_team_sample_size?: number;
    evaluator_sample_size?: number;
    confidence_threshold?: number;
    evaluator_threshold?: number;
    evaluator_consecutive_rounds?: number;
    compliance_requirements?: string;
    enable_data_augmentation?: boolean;
    augmentation_model_id?: string | null;
    augmentation_temperature?: number;
    enable_human_feedback?: boolean;
    multi_objective_optimization?: boolean;
    feature_dimensions?: string[] | null;
    feature_bins?: number | null;
    elite_ratio?: number;
    exploration_ratio?: number;
    exploitation_ratio?: number;
    archive_size?: number;
    checkpoint_interval?: number;
    keyword_analysis_enabled?: boolean;
    keywords_to_target?: string[] | null;
    keyword_penalty_weight?: number;
}
export interface ModelOrchestrationModel {
    name: string;
    role: string;
    weight: number;
    api_base?: string;
}
export interface ModelOrchestrationListResponse {
    models: ModelOrchestrationModel[];
    metrics: Record<string, unknown>;
    selection_strategies: string[];
}
export interface ModelOrchestrationRegisterRequest {
    model_name: string;
    role: string;
    weight?: number;
    api_key?: string;
    api_base?: string;
    temperature?: number;
    top_p?: number;
    max_tokens?: number;
    frequency_penalty?: number;
    presence_penalty?: number;
}
export interface ModelOrchestrationRegisterResponse {
    message: string;
    model_name: string;
}
export interface ModelOrchestrationEnsembleRequest {
    role: string;
    messages: Array<Record<string, string>>;
    selection_strategy?: string;
    temperature?: number;
    max_tokens?: number;
    num_responses?: number;
}
export interface ModelOrchestrationEnsembleResponse {
    responses: Array<Record<string, unknown>>;
}
export interface BubbleLabsStatusResponse {
    total_components: number;
    available_components: number;
    components: Record<string, Record<string, unknown>>;
}
export interface BubbleLabsInitializeResponse {
    [key: string]: Record<string, unknown>;
}
export interface BubbleLabsActionResponse {
    success?: boolean;
    error?: string;
    [key: string]: unknown;
}
export declare const createDefaultTeam: () => Team;
export declare const createDefaultGauntletRound: (roundNumber: number) => GauntletRoundRule;
export declare const createDefaultGauntlet: () => GauntletDefinition;
export interface MakerToolDefinition {
    tool_id: string;
    name: string;
    description: string;
    version: string;
    status: string;
    maker_mode: string;
    config: Record<string, unknown>;
    prompt_template?: string | null;
    system_prompt?: string | null;
    expected_schema?: Record<string, unknown> | null;
    created_at?: string;
    created_by?: string;
    test_results?: Record<string, unknown> | null;
    usage_count?: number;
    metadata?: Record<string, unknown>;
}
export interface MakerToolListResponse {
    tools: MakerToolDefinition[];
}
export interface MakerToolResponse {
    tool: MakerToolDefinition;
}
export interface MakerExecutionResult {
    tool_id: string;
    execution_id: string;
    input_data: Record<string, unknown>;
    output_data: unknown;
    execution_time: number;
    success: boolean;
    error_message?: string | null;
    metrics?: Record<string, unknown> | null;
    CREWAI_ticket_id?: string | null;
    timestamp?: string;
}
export interface MakerExecutionResponse {
    result: MakerExecutionResult;
}
export interface MakerDelegation {
    delegation_id: string;
    task_id: string;
    title: string;
    description: string;
    status: string;
    delegation_type: string;
    tool_id?: string | null;
    workflow_epic_id?: string | null;
    assigned_to?: string | null;
    created_at?: string;
    updated_at?: string;
    completed_at?: string | null;
    result?: Record<string, unknown> | null;
    metadata?: Record<string, unknown> | null;
}
export interface MakerDelegationListResponse {
    delegations: MakerDelegation[];
}
export interface KnowledgeExplorerQueryResponse {
    results: Record<string, unknown>;
    history: Array<Record<string, unknown>>;
}
export interface KnowledgeExplorerExtractResponse {
    results: Record<string, unknown>;
}
export interface KnowledgeExplorerHistoryResponse {
    history: Array<Record<string, unknown>>;
}
export interface LeanAideStatusResponse {
    leanaide_available: boolean;
    mcts_available: boolean;
    mdap_available: boolean;
    lean4_available: boolean;
    mcts_enabled: boolean;
    mdap_enabled: boolean;
    lean4_enabled: boolean;
    server: string;
    active_trees: number;
    active_proofs: number;
    execution_history_count: number;
    server_status?: Record<string, unknown> | null;
}
export interface LeanAideExecuteResponse {
    result: Record<string, unknown>;
}
export interface LeanAideTreeListResponse {
    tree_ids: string[];
}
export interface LeanAideTreeResponse {
    tree: Record<string, unknown>;
}
export interface LeanAideProofListResponse {
    proof_ids: string[];
}
export interface LeanAideProofResponse {
    proof: Record<string, unknown>;
}
export interface EvolutionRunResponse {
    run_id: string;
    status: string;
}
export interface EvolutionRunStatus {
    run_id: string;
    status: string;
    created_at?: string;
    started_at?: string | null;
    completed_at?: string | null;
    logs?: string[];
    result?: Record<string, unknown> | null;
    error?: string | null;
}
export interface EvolutionRunListResponse {
    runs: Array<Pick<EvolutionRunStatus, "run_id" | "status" | "created_at" | "started_at" | "completed_at">>;
}
export interface AdversarialRunResponse {
    run_id: string;
    status: string;
}
export interface AdversarialRunStatus {
    run_id: string;
    status: string;
    created_at?: string;
    started_at?: string | null;
    completed_at?: string | null;
    logs?: string[];
    result?: Record<string, unknown> | null;
    error?: string | null;
}
export interface AdversarialRunListResponse {
    runs: Array<Pick<AdversarialRunStatus, "run_id" | "status" | "created_at" | "started_at" | "completed_at">>;
}
//# sourceMappingURL=types.d.ts.map