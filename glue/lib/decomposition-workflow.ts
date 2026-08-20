/**
 * Decomposition Workflow Client (OpenEvolve integration)
 *
 * A self-contained, typed, resilient client for the OpenEvolve
 * decomposition / adversarial workflow backend (teams, gauntlets,
 * evaluators, workflows + decomposition-plan).
 *
 * It re-implements locally only the contract types it needs (no cross-package
 * coupling) and wraps every request in Glue's resilience primitives:
 *   - CircuitBreaker (stop hammering a dead service)
 *   - retryWithBackoff (transient failure recovery)
 *   - apiLogger (structured JSON logging with correlation ids)
 *
 * Routes are unprefixed on the backend; BubbleLab strips the leading `/api`.
 * This client uses whatever `baseUrl` you configure (e.g. `/api`).
 */

import { CircuitBreaker, CircuitState, CircuitBreakerOpenError } from './circuitBreaker';
import { retryWithBackoff, RetryConfig } from './retry';
import { apiLogger, LogContext } from './structuredLogger';

/* ------------------------------------------------------------------ *
 * Local contract types (mirrored from bubblelab types, not imported)
 * ------------------------------------------------------------------ */

export type TeamRole = 'Blue' | 'Red' | 'Gold';
export type TeamType = 'standard' | 'swarm' | 'sovereign';

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

export type CollaborationMode = 'independent' | 'share_previous_feedback';
export type VotingStrategy = 'fixed_quorum' | 'first_to_ahead_by_k';

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

export type GauntletType = 'standard' | 'adaptive' | 'hierarchical' | 'competitive' | 'collaborative';

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

export interface EvaluatorListResponse {
  evaluators: Record<string, string>;
}

export interface EvaluatorUploadResponse {
  evaluator_id: string;
}

/* ------------------------------------------------------------------ *
 * Client configuration
 * ------------------------------------------------------------------ */

export interface ApiConfig {
  baseUrl: string;
  apiKey: string;
  timeout?: number;
  /** Override circuit breaker tuning. */
  circuitBreaker?: {
    failureThreshold?: number;
    successThreshold?: number;
    timeoutMs?: number;
    monitoringPeriodMs?: number;
  };
  /** Override retry tuning applied to every request. */
  retry?: RetryConfig;
}

/* ------------------------------------------------------------------ *
 * Errors
 * ------------------------------------------------------------------ */

export class ApiError extends Error {
  readonly status: number;
  readonly body: string;
  readonly correlationId?: string;
  constructor(message: string, status: number, body: string, correlationId?: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
    this.correlationId = correlationId;
  }
}

/* ------------------------------------------------------------------ *
 * Client
 * ------------------------------------------------------------------ */

export interface DecompositionWorkflowClient {
  listTeams(): Promise<Team[]>;
  getTeam(name: string): Promise<Team>;
  createTeam(team: Team): Promise<Team>;
  updateTeam(name: string, team: Team): Promise<Team>;
  deleteTeam(name: string): Promise<void>;
  listGauntlets(): Promise<GauntletDefinition[]>;
  getGauntlet(name: string): Promise<GauntletDefinition>;
  createGauntlet(gauntlet: GauntletDefinition): Promise<GauntletDefinition>;
  updateGauntlet(name: string, gauntlet: GauntletDefinition): Promise<GauntletDefinition>;
  deleteGauntlet(name: string): Promise<void>;
  listEvaluators(): Promise<EvaluatorListResponse>;
  uploadEvaluator(name: string, code: string): Promise<EvaluatorUploadResponse>;
  deleteEvaluator(id: string): Promise<void>;
  listWorkflows(): Promise<WorkflowSummary[]>;
  getWorkflow(id: string): Promise<WorkflowDetail>;
  createWorkflow(request: WorkflowCreateRequest): Promise<WorkflowCreateResponse>;
  pauseWorkflow(id: string): Promise<void>;
  resumeWorkflow(id: string): Promise<void>;
  deleteWorkflow(id: string): Promise<void>;
  getWorkflowResults(id: string): Promise<WorkflowResults>;
  getWorkflowPlan(id: string): Promise<WorkflowPlanResponse>;
  updateWorkflowPlan(id: string, request: WorkflowPlanUpdateRequest): Promise<WorkflowPlanResponse>;
}

function generateCorrelationId(): string {
  return `dwf-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

export class DecompositionWorkflowClientImpl implements DecompositionWorkflowClient {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly timeout: number;
  private readonly circuitBreaker: CircuitBreaker;
  private readonly retryConfig: RetryConfig;
  private readonly defaultContext: LogContext;

  constructor(config: ApiConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, '');
    this.apiKey = config.apiKey;
    this.timeout = config.timeout ?? 30000;
    this.circuitBreaker = new CircuitBreaker('openevolve-decomposition-workflow', {
      failureThreshold: config.circuitBreaker?.failureThreshold ?? 5,
      successThreshold: config.circuitBreaker?.successThreshold ?? 2,
      timeoutMs: config.circuitBreaker?.timeoutMs ?? 60000,
      monitoringPeriodMs: config.circuitBreaker?.monitoringPeriodMs ?? 10000,
    });
    this.retryConfig = config.retry ?? { max_retries: 3 };
    this.defaultContext = {
      source_service: 'glue-decomposition-workflow',
      target_service: 'openevolve-backend',
    };
  }

  /**
   * Core request wrapper: correlation id + API key header, circuit breaker,
   * retry with backoff, structured logging, and non-2xx -> ApiError.
   */
  private async request<T>(
    method: string,
    path: string,
    options: {
      body?: unknown;
      query?: Record<string, string | number | boolean | undefined>;
      expectEmpty?: boolean;
    } = {}
  ): Promise<T> {
    const correlationId = generateCorrelationId();
    const context: LogContext = {
      ...this.defaultContext,
      correlation_id: correlationId,
      operation: `${method} ${path}`,
    };

    const url = this.buildUrl(path, options.query);

    const doFetch = async (): Promise<T> => {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeout);
      try {
        apiLogger.info(`${method} ${path}`, context);
        const response = await fetch(url, {
          method,
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': this.apiKey,
            'X-Correlation-ID': correlationId,
          },
          body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
          signal: controller.signal,
        });

        if (!response.ok) {
          const text = await response.text().catch(() => '');
          apiLogger.error(`Request failed with status ${response.status}`, undefined, {
            ...context,
            status: response.status,
          });
          throw new ApiError(
            `Request ${method} ${path} failed with status ${response.status}`,
            response.status,
            text,
            correlationId
          );
        }

        if (options.expectEmpty) {
          return undefined as T;
        }

        const contentType = response.headers.get('content-type') || '';
        if (!contentType.includes('application/json')) {
          return (await response.text()) as unknown as T;
        }
        return (await response.json()) as T;
      } finally {
        clearTimeout(timer);
      }
    };

    try {
      return await this.circuitBreaker.execute(
        () => retryWithBackoff(doFetch, this.retryConfig),
        context
      );
    } catch (err) {
      if (err instanceof CircuitBreakerOpenError) {
        apiLogger.error('Circuit breaker open; request rejected', err, context);
      }
      throw err;
    }
  }

  private buildUrl(path: string, query?: Record<string, string | number | boolean | undefined>): string {
    const base = `${this.baseUrl}${path.startsWith('/') ? path : `/${path}`}`;
    if (!query) {
      return base;
    }
    const params = new URLSearchParams();
    Object.entries(query).forEach(([key, value]) => {
      if (value !== undefined) {
        params.append(key, String(value));
      }
    });
    const qs = params.toString();
    return qs ? `${base}?${qs}` : base;
  }

  /* ----------------------------- Teams ----------------------------- */

  listTeams(): Promise<Team[]> {
    return this.request<Team[]>('GET', '/teams');
  }

  getTeam(name: string): Promise<Team> {
    return this.request<Team>('GET', `/teams/${encodeURIComponent(name)}`);
  }

  createTeam(team: Team): Promise<Team> {
    return this.request<Team>('POST', '/teams', { body: team });
  }

  updateTeam(name: string, team: Team): Promise<Team> {
    return this.request<Team>('PUT', `/teams/${encodeURIComponent(name)}`, { body: team });
  }

  deleteTeam(name: string): Promise<void> {
    return this.request<void>('DELETE', `/teams/${encodeURIComponent(name)}`, { expectEmpty: true });
  }

  /* --------------------------- Gauntlets --------------------------- */

  listGauntlets(): Promise<GauntletDefinition[]> {
    return this.request<GauntletDefinition[]>('GET', '/gauntlets');
  }

  getGauntlet(name: string): Promise<GauntletDefinition> {
    return this.request<GauntletDefinition>('GET', `/gauntlets/${encodeURIComponent(name)}`);
  }

  createGauntlet(gauntlet: GauntletDefinition): Promise<GauntletDefinition> {
    return this.request<GauntletDefinition>('POST', '/gauntlets', { body: gauntlet });
  }

  updateGauntlet(name: string, gauntlet: GauntletDefinition): Promise<GauntletDefinition> {
    return this.request<GauntletDefinition>('PUT', `/gauntlets/${encodeURIComponent(name)}`, { body: gauntlet });
  }

  deleteGauntlet(name: string): Promise<void> {
    return this.request<void>('DELETE', `/gauntlets/${encodeURIComponent(name)}`, { expectEmpty: true });
  }

  /* --------------------------- Evaluators -------------------------- */

  listEvaluators(): Promise<EvaluatorListResponse> {
    return this.request<EvaluatorListResponse>('GET', '/evaluators');
  }

  uploadEvaluator(name: string, code: string): Promise<EvaluatorUploadResponse> {
    return this.request<EvaluatorUploadResponse>('POST', '/evaluators', { body: { name, code } });
  }

  deleteEvaluator(id: string): Promise<void> {
    return this.request<void>('DELETE', `/evaluators/${encodeURIComponent(id)}`, { expectEmpty: true });
  }

  /* --------------------------- Workflows --------------------------- */

  listWorkflows(): Promise<WorkflowSummary[]> {
    return this.request<WorkflowSummary[]>('GET', '/workflows');
  }

  getWorkflow(id: string): Promise<WorkflowDetail> {
    return this.request<WorkflowDetail>('GET', `/workflows/${encodeURIComponent(id)}`);
  }

  createWorkflow(request: WorkflowCreateRequest): Promise<WorkflowCreateResponse> {
    return this.request<WorkflowCreateResponse>('POST', '/workflows', { body: request });
  }

  pauseWorkflow(id: string): Promise<void> {
    return this.request<void>('POST', `/workflows/${encodeURIComponent(id)}/pause`, { expectEmpty: true });
  }

  resumeWorkflow(id: string): Promise<void> {
    return this.request<void>('POST', `/workflows/${encodeURIComponent(id)}/resume`, { expectEmpty: true });
  }

  deleteWorkflow(id: string): Promise<void> {
    return this.request<void>('DELETE', `/workflows/${encodeURIComponent(id)}`, { expectEmpty: true });
  }

  getWorkflowResults(id: string): Promise<WorkflowResults> {
    return this.request<WorkflowResults>('GET', `/workflows/${encodeURIComponent(id)}/results`);
  }

  getWorkflowPlan(id: string): Promise<WorkflowPlanResponse> {
    return this.request<WorkflowPlanResponse>('GET', `/workflows/${encodeURIComponent(id)}/decomposition-plan`);
  }

  updateWorkflowPlan(id: string, request: WorkflowPlanUpdateRequest): Promise<WorkflowPlanResponse> {
    return this.request<WorkflowPlanResponse>('PUT', `/workflows/${encodeURIComponent(id)}/decomposition-plan`, { body: request });
  }
}

/**
 * Factory for the typed, resilient decomposition-workflow client.
 */
export function createDecompositionWorkflowClient(config: ApiConfig): DecompositionWorkflowClient {
  return new DecompositionWorkflowClientImpl(config);
}

export { CircuitState, CircuitBreakerOpenError };
export type { LogContext, RetryConfig };
