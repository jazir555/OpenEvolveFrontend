/**
 * OpenEvolve Main Orchestration Adapter
 *
 * This is the primary orchestration adapter that coordinates all integrated
 * systems within the OpenEvolve federation. It serves as the central hub for:
 * - Multi-adapter coordination (Z3, LeanAide, RAGBits, Vector DB, etc.)
 * - Workflow orchestration across multiple systems
 * - Knowledge aggregation from all sources
 * - Event bus integration for pub/sub patterns
 * - Circuit breaker and retry logic for resilience
 * - Canonical schema enforcement (Anti-Corruption Layer)
 *
 * Environment Variables:
 *   OPENEVOLVE_API_URL - Base URL of the OpenEvolve API (required, no default)
 *   TIMEOUT_MS - Request timeout in milliseconds (required, no default)
 *   EVENT_BUS_URL - Event bus URL for pub/sub (optional)
 *   LOG_LEVEL - Logging level (default: info)
 */
/**
 * Model configuration for AI team members
 */
export interface ModelConfig {
    model_id: string;
    api_key: string;
    api_base: string;
    temperature: number;
    top_p: number;
    max_tokens: number;
    frequency_penalty: number;
    presence_penalty: number;
    [key: string]: any;
}
/**
 * Team definition (Blue/Red/Gold)
 */
export interface Team {
    name: string;
    role: 'Blue' | 'Red' | 'Gold';
    members: ModelConfig[];
    tenant_id?: string;
    description?: string;
    sub_role?: string;
    domain_specialization?: string[];
    problem_type_specialization?: string[];
    performance_metrics?: Record<string, number>;
    team_config?: Record<string, any>;
    openevolve_metrics?: Record<string, any>[];
}
/**
 * Gauntlet round rules
 */
export interface GauntletRoundRule {
    round_number: number;
    quorum_required_approvals: number;
    quorum_from_panel_size: number;
    min_overall_confidence: number;
    max_score_variance?: number;
    per_judge_requirements?: Record<string, any>;
    collaboration_mode?: 'independent' | 'share_previous_feedback';
    time_limit_seconds?: number;
    max_api_calls?: number;
    max_tokens?: number;
    adaptive_rules?: Record<string, any>;
}
/**
 * Gauntlet definition
 */
export interface Gauntlet {
    name: string;
    team_name: string;
    rounds: GauntletRoundRule[];
    tenant_id?: string;
    description?: string;
    attack_modes?: string[];
    generation_mode?: 'single_candidate' | 'multi_candidate_peer_review' | 'evolutionary' | 'hybrid';
    gauntlet_type?: 'standard' | 'adaptive' | 'hierarchical' | 'competitive' | 'collaborative';
    performance_metrics?: Record<string, number>;
    gauntlet_config?: Record<string, any>;
}
/**
 * Sub-problem in decomposition
 */
export interface SubProblem {
    id: string;
    description: string;
    dependencies: string[];
    ai_suggested_evolution_mode?: string;
    ai_suggested_complexity_score?: number;
    ai_suggested_evaluation_prompt?: string;
    content_type?: string;
    solver_team_name: string;
    red_team_gauntlet_name?: string;
    gold_team_gauntlet_name: string;
    solver_generation_gauntlet_name?: string;
    patcher_team_name?: string;
    evolution_params?: Record<string, any>;
    ai_suggested_team_assignment?: string;
    ai_suggested_gauntlet_assignment?: Record<string, string>;
    estimated_resources?: Record<string, any>;
    potential_approaches?: string[];
    status?: 'pending' | 'in_progress' | 'solved' | 'failed' | 'requires_rework';
    solution_attempts?: SolutionAttempt[];
    performance_metrics?: Record<string, number>;
    openevolve_metrics?: Record<string, any>;
}
/**
 * Solution attempt
 */
export interface SolutionAttempt {
    sub_problem_id: string;
    content: string;
    generated_by_model: string;
    timestamp: number;
    history?: Record<string, any>[];
    solution_type?: string;
    solution_approach?: string;
    quality_metrics?: Record<string, number>;
    resource_usage?: Record<string, any>;
    status?: 'generated' | 'critiqued' | 'verified' | 'rejected' | 'patched';
    critique_reports?: CritiqueReport[];
    verification_reports?: VerificationReport[];
    openevolve_metrics?: Record<string, any>;
}
/**
 * Critique report from Red Team
 */
export interface CritiqueReport {
    solution_attempt_id: string;
    gauntlet_name: string;
    is_approved: boolean;
    reports_by_judge: Record<string, any>[];
    summary?: string;
    critique_timestamp?: number;
    overall_score?: number;
    flaw_severity_scores?: Record<string, number>;
    identified_flaws?: Record<string, any>[];
    suggested_improvements?: string[];
    resource_usage?: Record<string, any>;
}
/**
 * Verification report from Gold Team
 */
export interface VerificationReport {
    solution_attempt_id: string;
    gauntlet_name: string;
    is_approved: boolean;
    reports_by_judge: Record<string, any>[];
    average_score?: number;
    score_variance?: number;
    summary?: string;
    verification_timestamp?: number;
    dimension_scores?: Record<string, number>;
    criteria_met?: string[];
    criteria_not_met?: string[];
    targeted_feedback?: string;
    resource_usage?: Record<string, any>;
}
/**
 * Workflow definition
 */
export interface WorkflowDefinition {
    workflow_id: string;
    name: string;
    description?: string;
    problem_statement: string;
    max_refinement_loops: number;
    auto_approval_enabled: boolean;
    auto_approval_criteria?: Record<string, any>;
    mdap_enabled?: boolean;
    mdap_config?: Record<string, any>;
    maker_enabled?: boolean;
    maker_config?: Record<string, any>;
    resource_limits?: Record<string, any>;
    parallel_processing_enabled?: boolean;
    max_parallel_sub_problems?: number;
    learning_enabled?: boolean;
    learning_config?: Record<string, any>;
    content_analyzer_team_name?: string;
    planner_team_name?: string;
    assembler_team_name?: string;
    final_red_team_gauntlet_name?: string;
    final_gold_team_gauntlet_name?: string;
    sub_problems: SubProblem[];
}
/**
 * Workflow state
 */
export interface WorkflowState {
    workflow_id: string;
    workflow_type?: any;
    problem_statement: string;
    current_stage: string;
    tenant_id?: string;
    current_sub_problem_id?: string;
    current_gauntlet_name?: string;
    status: string;
    progress: number;
    start_time: string;
    end_time?: string;
    decomposition_plan?: WorkflowDefinition;
    sub_problem_solutions?: Record<string, SolutionAttempt>;
    solved_sub_problem_ids?: Set<string>;
    rejected_sub_problems?: Record<string, any>;
    final_solution?: SolutionAttempt;
    refinement_loop_count?: number;
    content_analyzer_team?: Team;
    planner_team?: Team;
    solver_team?: Team;
    patcher_team?: Team;
    assembler_team?: Team;
    sub_problem_red_gauntlet?: Gauntlet;
    sub_problem_gold_gauntlet?: Gauntlet;
    final_red_gauntlet?: Gauntlet;
    final_gold_gauntlet?: Gauntlet;
    max_refinement_loops?: number;
    all_critique_reports?: CritiqueReport[];
    all_verification_reports?: VerificationReport[];
    resource_usage?: Record<string, any>;
    performance_metrics?: Record<string, number>;
    knowledge_artifacts?: KnowledgeArtifact[];
    openevolve_metrics?: Record<string, any>;
    [key: string]: any;
}
/**
 * Knowledge artifact
 */
export interface KnowledgeArtifact {
    id: string;
    artifact_type: 'solution_pattern' | 'problem_solution_mapping' | 'critique_insight' | 'team_performance' | 'gauntlet_effectiveness';
    content: Record<string, any>;
    source_workflow_id: string;
    extraction_timestamp: number;
    domain?: string;
    problem_type?: string;
    usage_count?: number;
    effectiveness_score?: number;
    related_artifacts?: string[];
}
/**
 * Integration health status
 */
export interface IntegrationHealth {
    name: string;
    status: 'healthy' | 'unhealthy' | 'unknown';
    latency_ms?: number;
    last_check?: string;
    error_message?: string;
}
interface CircuitBreakerConfig {
    failureThreshold: number;
    successThreshold: number;
    timeout: number;
    monitorPeriod: number;
}
interface RetryConfig {
    maxRetries: number;
    baseDelay: number;
    maxDelay: number;
    jitter: boolean;
}
export interface LogContext {
    correlation_id: string;
    source_service: string;
    target_service?: string;
    [key: string]: any;
}
export declare class StructuredLogger {
    private readonly serviceName;
    private readonly logLevel;
    constructor(serviceName: string, logLevel?: string);
    private log;
    info(message: string, context?: Record<string, any>): void;
    warn(message: string, context?: Record<string, any>): void;
    error(message: string, context?: Record<string, any>): void;
    debug(message: string, context?: Record<string, any>): void;
}
export interface OpenEvolveAdapterConfig {
    api_url: string;
    timeout_ms: number;
    event_bus_url?: string;
    log_level?: string;
    circuit_breaker?: Partial<CircuitBreakerConfig>;
    retry?: Partial<RetryConfig>;
}
export declare class OpenEvolveAdapter {
    private readonly api;
    private readonly logger;
    private readonly circuitBreakers;
    private readonly retryConfig;
    private readonly correlationId;
    private z3CircuitBreaker;
    private leanaideCircuitBreaker;
    private ragbitsCircuitBreaker;
    private vectordbCircuitBreaker;
    private graphitiCircuitBreaker;
    private karateclubCircuitBreaker;
    constructor(config: OpenEvolveAdapterConfig);
    healthCheck(): Promise<{
        status: string;
        timestamp: string;
        integrations: IntegrationHealth[];
    }>;
    private checkIntegrationHealth;
    createTeam(team: Team): Promise<{
        message: string;
        team_name: string;
    }>;
    getTeams(): Promise<Team[]>;
    getTeam(name: string): Promise<Team>;
    updateTeam(name: string, team: Team): Promise<{
        message: string;
        team_name: string;
    }>;
    deleteTeam(name: string): Promise<{
        message: string;
        team_name: string;
    }>;
    createGauntlet(gauntlet: Gauntlet): Promise<{
        message: string;
        gauntlet_name: string;
    }>;
    getGauntlets(): Promise<Gauntlet[]>;
    getGauntlet(name: string): Promise<Gauntlet>;
    deleteGauntlet(name: string): Promise<{
        message: string;
        gauntlet_name: string;
    }>;
    createWorkflow(workflow: WorkflowDefinition): Promise<{
        message: string;
        workflow_id: string;
    }>;
    getWorkflows(): Promise<WorkflowState[]>;
    getWorkflowStatus(workflowId: string): Promise<WorkflowState>;
    deleteWorkflow(workflowId: string): Promise<{
        message: string;
        workflow_id: string;
    }>;
    getIntegrationHealth(): Promise<{
        integrations: IntegrationHealth[];
    }>;
    getAvailableAdapters(): Promise<{
        name: string;
        type: string;
        status: string;
    }[]>;
}
export declare function createOpenEvolveAdapter(config: OpenEvolveAdapterConfig): OpenEvolveAdapter;
export default OpenEvolveAdapter;
//# sourceMappingURL=adapter.d.ts.map