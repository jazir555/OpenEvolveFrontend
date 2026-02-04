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
