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

export type GenerationMode =
  | "single_candidate"
  | "multi_candidate_peer_review"
  | "evolutionary"
  | "hybrid";

export type GauntletType =
  | "standard"
  | "adaptive"
  | "hierarchical"
  | "competitive"
  | "collaborative";

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

export type IcrComponents = Record<
  string,
  {
    total_patterns?: number;
    overall_pass_rate?: number;
    active?: boolean;
    recent_pass_rate?: number;
    recent_fail_rate?: number;
    top_failure_modes?: string[];
  }
>;

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

export interface KnowledgeRecommendations {
  recommended_approaches: Array<Record<string, unknown>>;
  similar_problems: Array<Record<string, unknown>>;
  team_recommendations: Array<Record<string, unknown>>;
  gauntlet_recommendations: Array<Record<string, unknown>>;
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

export const createDefaultModelConfig = (): ModelConfig => ({
  model_id: "",
  api_key: "",
  api_base: "https://api.openai.com/v1",
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

export const createDefaultTeam = (): Team => ({
  name: "",
  role: "Blue",
  members: [createDefaultModelConfig()],
  description: "",
  content_analysis_system_prompt: "",
  content_analysis_user_prompt_template: "",
  decomposition_system_prompt: "",
  decomposition_user_prompt_template: "",
  solver_system_prompt: "",
  solver_user_prompt_template: "",
  patcher_system_prompt: "",
  patcher_user_prompt_template: "",
  assembler_system_prompt: "",
  assembler_user_prompt_template: "",
  red_team_system_prompt: "",
  red_team_user_prompt_template: "",
  gold_team_system_prompt: "",
  gold_team_user_prompt_template: "",
  sub_role: "",
  team_type: "standard",
});

export const createDefaultGauntletRound = (roundNumber: number): GauntletRoundRule => ({
  round_number: roundNumber,
  quorum_required_approvals: 1,
  quorum_from_panel_size: 1,
  min_overall_confidence: 0.5,
  max_score_variance: null,
  per_judge_requirements: {},
  collaboration_mode: "independent",
  time_limit_seconds: null,
  max_api_calls: null,
  max_tokens: null,
  adaptive_rules: null,
  voting_strategy: "fixed_quorum",
  margin_k: null,
  max_dynamic_votes: 100,
  required_mathematical_properties: [],
  proof_obligation_threshold: 0.0,
  mathematical_complexity_level: 1,
  proof_generation_enabled: false,
  proof_verification_enabled: false,
  mathematical_approach: "direct_proof",
  verification_timeout: 300,
  proof_storage_enabled: false,
  mathematical_quality_threshold: 0.0,
});

export const createDefaultGauntlet = (): GauntletDefinition => ({
  name: "",
  team_name: "",
  rounds: [createDefaultGauntletRound(1)],
  description: "",
  attack_modes: [],
  generation_mode: "single_candidate",
  gauntlet_type: "standard",
  red_flags: {
    max_token_length: 2000,
    strict_format_adherence: true,
    forbidden_phrases: ["I apologize", "I'm confused", "As an AI"],
  },
  formal_verification_enabled: false,
  verification_methods: ["peer_review"],
  mathematical_requirements: {},
  proof_generation_enabled: false,
  automatic_formalization: false,
  formal_verification_threshold: 0.9,
  lean_verification_config: {},
});
