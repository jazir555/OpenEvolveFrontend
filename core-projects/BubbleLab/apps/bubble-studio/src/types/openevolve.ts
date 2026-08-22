/**
 * OpenEvolve Type Definitions
 *
 * Type definitions for OpenEvolve workflow parameters and responses
 * Matches Pydantic models in BubbleLab/services/openevolve-api/models/__init__.py
 */

// ==================== Evolution Parameters ====================

/**
 * Evolution workflow parameters
 */
export interface EvolutionParameters {
  max_iterations: number;        // Default: 100, Range: 1-200
  population_size: number;        // Default: 50, Range: 1-100
  temperature: number;            // Default: 0.7, Range: 0.0-2.0
  top_p: number;                  // Default: 1.0, Range: 0.0-1.0
  max_tokens: number;             // Default: 4096, Range: 1-100000
  frequency_penalty: number;      // Default: 0.0, Range: -2.0 to 2.0
  presence_penalty: number;       // Default: 0.0, Range: -2.0 to 2.0
  seed: number;                   // Default: 42, Range: -1 to 999999 (-1 for random)
}

// ==================== Adversarial Parameters ====================

/**
 * Adversarial testing workflow parameters
 */
export interface AdversarialParameters {
  test_cases: string[];           // Test cases for adversarial evaluation
  attack_types: string[];         // Types of attacks to test (default: ["fuzzing", "prompt_injection", "code_injection"])
  rounds: number;                 // Number of testing rounds, Default: 3, Range: 1-10
}

// ==================== Sovereign Parameters ====================

/**
 * Sovereign decomposition workflow parameters
 */
export interface SovereignParameters {
  decomposition_depth: number;    // Max depth of problem decomposition, Default: 3, Range: 1-10
  parallel_subproblems: number;   // Number of sub-problems to solve in parallel, Default: 5, Range: 1-20
  verification_strictness: 'lenient' | 'standard' | 'strict';  // Default: "standard"
}

// ==================== Workflow Types ====================

export type WorkflowType = 'evolution' | 'adversarial' | 'sovereign';
export type WorkflowStatus =
  | 'created'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'draft'
  | 'ready'
  | 'archived';
export type ExecutionStatus = 'queued' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';

// ==================== Workflow Metadata ====================

export interface WorkflowMetadata {
  mdap_enabled?: boolean;
  maker_enabled?: boolean;
  maker_config?: Record<string, unknown>;
  adaptive_config?: Record<string, unknown>;
  evolution_params?: Record<string, unknown>;
  performance_params?: Record<string, unknown>;
}

// ==================== Common Interfaces ====================

/**
 * Base workflow interface
 */
export interface WorkflowBase {
  name: string;
  description: string;
  workflow_type: WorkflowType;
  problem_statement?: string;
  content_type?: string;
  teams?: string[];
  gauntlets?: string[];
  metadata?: WorkflowMetadata | null;
  parameters?: Record<string, unknown>;
}

/**
 * Workflow creation request
 */
export interface WorkflowCreate extends WorkflowBase {
  parameters?: Record<string, unknown>;
}

/**
 * Workflow update request
 */
export interface WorkflowUpdate {
  name?: string;
  description?: string;
  problem_statement?: string;
  content_type?: string;
  teams?: string[];
  gauntlets?: string[];
  metadata?: WorkflowMetadata | null;
  parameters?: Record<string, unknown>;
}

/**
 * Complete workflow response
 */
export interface WorkflowResponse extends WorkflowBase {
  id: string;
  parameters: Record<string, unknown>;
  status: WorkflowStatus;
  created_at: string;             // ISO 8601 datetime
  updated_at: string;             // ISO 8601 datetime
  started_at?: string | null;
  completed_at?: string | null;
  user_id?: string;
  tenant_id?: string;
}

/**
 * Workflow list response
 */
export interface WorkflowListResponse {
  workflows: WorkflowResponse[];
  total: number;
  page: number;
  page_size: number;
}

// ==================== Execution Types ====================

/**
 * Inputs for workflow execution
 */
export interface WorkflowInputs {
  problem_statement: string;
  context?: string;
}

/**
 * Execution response
 */
export interface ExecutionResponse {
  execution_id: string;
  workflow_id: string;
  status: ExecutionStatus;
  progress: number;               // Range: 0.0 to 1.0
  started_at?: string;
  completed_at?: string;
  result?: Record<string, unknown>;
  error?: string;
  name?: string;
  real_engine?: boolean;
  real_engine_available?: boolean;
  best_score?: number;
  result_summary?: string;
}

/**
 * Execution logs response
 */
export interface ExecutionLogsResponse {
  logs: Array<Record<string, unknown>>;
  total: number;
  since?: string;
}

// ==================== Team Types ====================

/**
 * Team member definition
 */
export interface TeamMember {
  id?: string;
  name: string;
  role: string;
  model: string;
  temperature: number;
  max_tokens: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  max_iterations?: number;
}

/**
 * Team creation request
 */
export interface TeamCreate {
  name: string;
  description: string;
  members: TeamMember[];
}

/**
 * Team response
 */
export interface TeamResponse {
  id: string;
  name: string;
  description: string;
  members: TeamMember[];
  created_at: string;
}

/**
 * Team list response
 */
export interface TeamListResponse {
  teams: TeamResponse[];
  total: number;
}

// ==================== Gauntlet Types ====================

/**
 * Gauntlet round definition
 */
export interface GauntletRound {
  name: string;
  quorum_threshold: number;       // Range: 0.0 to 1.0
  confidence_threshold: number;   // Range: 0.0 to 1.0
  evaluation_type: string;
}

/**
 * Gauntlet creation request
 */
export interface GauntletCreate {
  name: string;
  description: string;
  rounds: GauntletRound[];
}

/**
 * Gauntlet response
 */
export interface GauntletResponse {
  id: string;
  name: string;
  description: string;
  rounds: GauntletRound[];
  created_at: string;
}

/**
 * Gauntlet list response
 */
export interface GauntletListResponse {
  gauntlets: GauntletResponse[];
  total: number;
}

// ==================== Health & Status ====================

/**
 * Health check response
 */
export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  features: {
    evolution: boolean;
    adversarial: boolean;
    sovereign: boolean;
  };
}

// ==================== Default Values ====================

/**
 * Default evolution parameters
 */
export const DEFAULT_EVOLUTION_PARAMETERS: EvolutionParameters = {
  max_iterations: 100,
  population_size: 50,
  temperature: 0.7,
  top_p: 1.0,
  max_tokens: 4096,
  frequency_penalty: 0.0,
  presence_penalty: 0.0,
  seed: 42,
};

/**
 * Default adversarial parameters
 */
export const DEFAULT_ADVERSARIAL_PARAMETERS: AdversarialParameters = {
  test_cases: [],
  attack_types: ['fuzzing', 'prompt_injection', 'code_injection'],
  rounds: 3,
};

/**
 * Default sovereign parameters
 */
export const DEFAULT_SOVEREIGN_PARAMETERS: SovereignParameters = {
  decomposition_depth: 3,
  parallel_subproblems: 5,
  verification_strictness: 'standard',
};

// ============================================================================
// Decomposition / Adversarial Workflow Surface
// ============================================================================
//
// The types below mirror the canonical OpenEvolve SDK (`src/lib/types.ts` in the
// converted BubbleLab SDK) 1:1 so that the BubbleLab app speaks the same wire
// contract as `engines/other/api_server.py`.
//
// NOTE: these coexist with the older service-oriented types above
// (`TeamCreate` / `TeamResponse` / `GauntletRound` / `GauntletCreate` /
// `GauntletResponse` / `WorkflowResponse`), which model the legacy
// `openevolve-api` microservice rather than the current FastAPI backend. New UI
// should prefer the canonical types in this section.

// ==================== Teams (canonical) ====================

export type TeamRole = 'Blue' | 'Red' | 'Gold';
export type TeamType = 'standard' | 'swarm' | 'sovereign';

/**
 * Full model configuration for a single team member (canonical `ModelConfig`).
 */
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

/**
 * Full team definition — the body accepted by `POST /teams` and `PUT /teams/{name}`
 * and the payload returned by `GET /teams/{name}`.
 */
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

/**
 * Condensed team row returned by `GET /teams`.
 */
export interface TeamSummary {
  id: string;
  name: string;
  role: TeamRole;
  description?: string | null;
  member_count: number;
}

// ==================== Gauntlets (canonical) ====================

export type CollaborationMode = 'independent' | 'share_previous_feedback';
export type VotingStrategy = 'fixed_quorum' | 'first_to_ahead_by_k';

/**
 * A single adversarial gauntlet round rule.
 */
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

export type GenerationMode =
  | 'single_candidate'
  | 'multi_candidate_peer_review'
  | 'evolutionary'
  | 'hybrid';

export type GauntletType =
  | 'standard'
  | 'adaptive'
  | 'hierarchical'
  | 'competitive'
  | 'collaborative';

/**
 * Full gauntlet definition — the body accepted by `POST /gauntlets` and
 * `PUT /gauntlets/{name}` and the payload returned by `GET /gauntlets/{name}`.
 */
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

/**
 * Condensed gauntlet row returned by `GET /gauntlets`.
 */
export interface GauntletSummary {
  id: string;
  name: string;
  team_name: string;
  description?: string | null;
  round_count: number;
}

// ==================== Decomposition Workflows (canonical) ====================

/**
 * Condensed workflow row returned by `GET /workflows`.
 */
export interface WorkflowSummary {
  workflow_id: string;
  status: string;
  current_stage: string;
  progress: number;
}

/**
 * Full workflow detail returned by `GET /workflows/{id}`.
 */
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

/**
 * Body accepted by `POST /workflows` (decomposition workflow creation).
 */
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

/**
 * Response returned by `POST /workflows`.
 */
export interface WorkflowCreateResponse {
  workflow_id: string;
  status: string;
  current_stage: string;
  progress: number;
  created_at: string;
}

/**
 * Response returned by `GET /api/workflows/{id}/engine/results` (engine-backed
 * decomposition-workflow run results). The shape is intentionally permissive:
 * different engine versions expose `final_solution`, `sub_problems`,
 * `statistics`, and `error` with varying nested structures.
 */
export interface WorkflowEngineResults {
  workflow_id?: string;
  status?: string;
  final_solution?: unknown;
  sub_problems?: unknown;
  statistics?: unknown;
  error?: string;
  [key: string]: unknown;
}

/**
 * Response returned by `GET /workflows/{id}/results`.
 */
export interface WorkflowResults {
  workflow_id: string;
  problem_statement: string;
  status: string;
  final_solution?: {
    content: string;
    generated_by?: string;
    timestamp?: string;
  } | null;
  sub_problem_solutions: Record<
    string,
    {
      content: string;
      generated_by?: string;
      timestamp?: string;
    }
  >;
  execution_time?: number | null;
  refinement_loops: number;
}

// ==================== Decomposition Plan ====================

/**
 * A single sub-problem inside a workflow decomposition plan.
 */
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

/**
 * The decomposition plan for a workflow.
 */
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

/**
 * Sub-problem dependency edges plus the derived topological execution order.
 */
export interface WorkflowDependencyGraph {
  edges: Record<string, string[]>;
  execution_order?: string[];
}

/**
 * Response returned by `GET /workflows/{id}/decomposition-plan`.
 */
export interface WorkflowPlanResponse {
  workflow_id: string;
  plan: WorkflowDecompositionPlan;
  dependency_graph: WorkflowDependencyGraph;
}

/**
 * Body accepted by `PUT /workflows/{id}/decomposition-plan`.
 */
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

// ==================== Evaluators ====================

/**
 * Response returned by `GET /evaluators` — a map of evaluator id to source code.
 */
export interface EvaluatorListResponse {
  evaluators: Record<string, string>;
}

/**
 * Response returned by `POST /evaluators`.
 */
export interface EvaluatorUploadResponse {
  evaluator_id: string;
}

// ==================== Execution Records (canonical) ====================

/**
 * Execution compatibility surface (`/executions`).
 *
 * Mirrors the in-memory records returned by the FastAPI compatibility endpoints
 * used by the BubbleLab execution controls.
 *
 * NOTE: this is the raw wire record. `ExecutionResponse` above is the richer
 * normalized shape that `openevolveApi` exposes to existing callers (it maps
 * `id` -> `execution_id`).
 */
export interface ExecutionRecord {
  id: string;
  name: string;
  status: string;
  workflow_id?: string | null;
  created_at: string;
  updated_at: string;
}

/**
 * Response returned by `GET /executions`.
 */
export interface ExecutionListResponse {
  executions: ExecutionRecord[];
  total: number;
  limit?: number;
  offset?: number;
}

/**
 * Body accepted by `POST /executions`.
 */
export interface ExecutionCreateRequest {
  name?: string;
  workflow_id?: string;
  [key: string]: unknown;
}

// ============================================================================
// Monitoring / Analytics / Knowledge / CrewAI / LeanAide / Version-Control /
// Validation / Parameters / Integrated-run Surface
// ============================================================================
//
// These types mirror the canonical OpenEvolve SDK (`src/lib/types.ts`) 1:1 so
// the BubbleLab app speaks the same wire contract as `engines/other/api_server.py`
// for the Monitoring, Analytics, Knowledge Base, CrewAI, LeanAide,
// Version-Control, Validation, Parameters and Integrated-run feature areas.

// ==================== Monitoring ====================

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

// ==================== Analytics ====================

export interface StatisticsSummary {
  total_workflows: number;
  completed: number;
  failed: number;
  running: number;
  total_teams: number;
  total_gauntlets: number;
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

export interface AnalyticsKnowledgeStats {
  total_artifacts: number;
  total_usage: number;
  avg_effectiveness: number;
  artifact_type_distribution: Record<string, number>;
  domain_distribution: Record<string, number>;
  top_used_artifacts: Array<Record<string, unknown>>;
  top_effective_artifacts: Array<Record<string, unknown>>;
}

// ==================== Knowledge Base ====================

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

export interface KnowledgeRecommendations {
  recommended_approaches: Array<Record<string, unknown>>;
  similar_problems: Array<Record<string, unknown>>;
  team_recommendations: Array<Record<string, unknown>>;
  gauntlet_recommendations: Array<Record<string, unknown>>;
}

// ==================== CrewAI ====================

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

// ==================== LeanAide ====================

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

// ==================== Version Control ====================

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

// ==================== Validation ====================

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

export interface ComplianceCheckResult extends ValidationRuleResult {}

// ==================== Parameters ====================

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

// ==================== Integrated Workflow ====================

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

// ==================== Security / API Keys (RBAC) ====================

/**
 * Body for `POST /api/security/api-keys`.
 *
 * `username` is required (1..256 chars). `roles` are RBAC role names
 * (e.g. `admin`, `user`, `readonly`). Requires an admin key on the wire.
 */
export interface ApiKeyCreateRequest {
  username: string;
  email?: string | null;
  roles: string[];
}

/**
 * Response of `POST /api/security/api-keys`.
 *
 * `api_key` is the RAW secret and is returned exactly ONCE by the engine — it is
 * never included in `GET /api/security/api-keys`. Never log or persist it
 * anywhere except (optionally) the caller's active-key storage.
 */
export interface ApiKeyCreateResponse {
  key_id: string;
  api_key: string;
  user_id?: string;
  username?: string;
  roles?: string[];
  warning?: string;
}

/** One entry of `GET /api/security/api-keys` (never contains the raw secret). */
export interface ApiKeyListItem {
  key_id: string;
  username?: string;
  roles?: string[];
  permissions?: string[];
  created_at?: string;
  created_by?: string;
  revoked?: boolean;
  user_id?: string;
}

export interface ApiKeyListResponse {
  api_keys: ApiKeyListItem[];
  /** `false` when the engine has no RBAC subsystem loaded. */
  rbac_available: boolean;
}

/** Response of `DELETE /api/security/api-keys/{key_id}`. */
export interface RevokeApiKeyResponse {
  status: string;
  key_id: string;
}

export interface SecurityRole {
  name: string;
  description?: string;
  permissions: string[];
}

/** Body for `POST /api/security/roles`. */
export interface SecurityRoleCreateRequest {
  name: string;
  description?: string;
  permissions: string[];
}

export interface RolesResponse {
  roles: SecurityRole[];
  rbac_available: boolean;
}

/**
 * Audit log entries are engine-defined dicts. The commonly present keys are
 * `timestamp`, `operation`, `resource`, `resource_id` and `success`, but the
 * shape is intentionally loose so unknown fields survive.
 */
export type AuditLogEntry = Record<string, unknown>;

export interface AuditLogsResponse {
  audit_logs: AuditLogEntry[];
  source: 'rbac' | 'in_memory';
}

// ==================== Canonical Factory Functions ====================

/**
 * Default model configuration for a new team member.
 */
export const createDefaultModelConfig = (): ModelConfig => ({
  model_id: '',
  api_key: '',
  api_base: 'https://api.openai.com/v1',
  temperature: 0.7,
  top_p: 1.0,
  max_tokens: 4096,
  frequency_penalty: 0.0,
  presence_penalty: 0.0,
  seed: null,
  n: 1,
  logit_bias: null,
  reasoning_effort: null,
  stop_sequences: null,
  logprobs: null,
  top_logprobs: null,
  response_format: null,
  stream: null,
  user: null,
  max_retries: 5,
  timeout: 120,
  organization: null,
  response_model: null,
  tools: null,
  tool_choice: null,
  system_fingerprint: null,
  deployment_id: null,
  encoding_format: null,
  max_input_tokens: null,
  stop_token: null,
  best_of: null,
  logprobs_offset: null,
  suffix: null,
  presence_penalty_range: null,
  frequency_penalty_range: null,
  stop_token_id: null,
  response_json_format: null,
  max_output_tokens: null,
  stream_options: null,
  logprobs_type: null,
  top_k: null,
  repetition_penalty: null,
  length_penalty: null,
  early_stopping: null,
  num_beams: null,
  do_sample: null,
  temperature_fallback: null,
  top_p_fallback: null,
  max_time: null,
  return_full_text: null,
  tokenizer_config: null,
  model_kwargs: null,
  domain_specialization: null,
  problem_type_specialization: null,
  performance_metrics: null,
  cost_per_token: null,
});

/**
 * Default team definition (one blank Blue-team member).
 */
export const createDefaultTeam = (): Team => ({
  name: '',
  role: 'Blue',
  members: [createDefaultModelConfig()],
  description: '',
  content_analysis_system_prompt: '',
  content_analysis_user_prompt_template: '',
  decomposition_system_prompt: '',
  decomposition_user_prompt_template: '',
  solver_system_prompt: '',
  solver_user_prompt_template: '',
  patcher_system_prompt: '',
  patcher_user_prompt_template: '',
  assembler_system_prompt: '',
  assembler_user_prompt_template: '',
  red_team_system_prompt: '',
  red_team_user_prompt_template: '',
  gold_team_system_prompt: '',
  gold_team_user_prompt_template: '',
  sub_role: '',
  team_type: 'standard',
});

/**
 * Default gauntlet round rule for the given round number.
 */
export const createDefaultGauntletRound = (roundNumber: number): GauntletRoundRule => ({
  round_number: roundNumber,
  quorum_required_approvals: 1,
  quorum_from_panel_size: 1,
  min_overall_confidence: 0.5,
  max_score_variance: null,
  per_judge_requirements: {},
  collaboration_mode: 'independent',
  time_limit_seconds: null,
  max_api_calls: null,
  max_tokens: null,
  adaptive_rules: null,
  voting_strategy: 'fixed_quorum',
  margin_k: null,
  max_dynamic_votes: 100,
  required_mathematical_properties: [],
  proof_obligation_threshold: 0.0,
  mathematical_complexity_level: 1,
  proof_generation_enabled: false,
  proof_verification_enabled: false,
  mathematical_approach: 'direct_proof',
  verification_timeout: 300,
  proof_storage_enabled: false,
  mathematical_quality_threshold: 0.0,
});

/**
 * Default gauntlet definition (single round).
 */
export const createDefaultGauntlet = (): GauntletDefinition => ({
  name: '',
  team_name: '',
  rounds: [createDefaultGauntletRound(1)],
  description: '',
  attack_modes: [],
  generation_mode: 'single_candidate',
  gauntlet_type: 'standard',
  red_flags: {
    max_token_length: 2000,
    strict_format_adherence: true,
    forbidden_phrases: ['I apologize', "I'm confused", 'As an AI'],
  },
  formal_verification_enabled: false,
  verification_methods: ['peer_review'],
  mathematical_requirements: {},
  proof_generation_enabled: false,
  automatic_formalization: false,
  formal_verification_threshold: 0.9,
  lean_verification_config: {},
});
