/**
 * OpenEvolve API Client
 *
 * Connects BubbleLab frontend to the new OpenEvolve FastAPI service
 * Replaces the old Streamlit-based integration
 *
 * Architecture: Service-Oriented Architecture
 * - Eliminates parameter sync issues
 * - Clean REST API interface
 * - Full OpenAPI documentation at /docs
 *
 * PATH CONTRACT: the OpenEvolve backend is `services/openevolve-api` (FastAPI). Its
 * routers are mounted ALREADY prefixed (`/api/workflows`, `/api/teams`, `/api/gauntlets`,
 * `/api/executions`, `/api/monitoring`, ...). There is NO `rewrite_api_prefix`
 * middleware: the `/api/...` paths used below ARE the canonical, final routes and are
 * sent as-is (no upstream path rewriting). The unprefixed route groups this service
 * mounts are `/health`, `/icr`, `/determinism`, `/bubblelabs`, and `/stream/...`.
 *
 * CONTROL PLANE: `services/openevolve-api/api/bubblelabs_control.py` is mounted at
 * `/bubblelabs` and implements the `/bubblelabs/control/*`,
 * `/bubblelabs/workflow-definitions*`, and `/bubblelabs/workflow-instances*` routes
 * this client calls (catalog/discover/execute, definitions CRUD, and full instance
 * lifecycle). The BubbleLab Hono proxy now forwards `/*` upstream, so the UI reaches
 * the control plane directly. These are NOT client-vs-backend gaps.
 *
 * A separate library server (`core-projects/openevolve/openevolve/server_stdlib.py`)
 * also exists and exposes `/api/v1/...` routes that wrap the real engine. This client
 * targets the FastAPI service directly via `OPENEVOLVE_API_BASE_URL` (default
 * http://localhost:8000). The BubbleLab Hono proxy
 * (`apps/bubblelab-api/src/routes/openevolve.ts`) can mediate these calls, but the
 * contract (already-prefixed `/api/...`) is unchanged.
 *
 * SECOND CLIENT: `src/lib/api-client.ts` is a separate fetch-based client used by
 * `use-workflows-api.ts`, `use-teams-api.ts`, `use-gauntlets-api.ts` and
 * `SettingsPanel.tsx`. Keep the two in sync: teams/gauntlets are addressed by NAME,
 * and neither client may call `PUT /workflows/{id}`, `POST /workflows/{id}/start`,
 * `POST /workflows/{id}/stop` or any `/settings` route, because the backend has none.
 *
 * @see BubbleLab/services/openevolve-api/README.md
 */

import { ApiClient, ApiClientConfig } from '@/lib/api';
import { OPENEVOLVE_API_BASE_URL } from '@/env';
import { logger } from '@/utils/logger';
import type {
  EvolutionParameters,
  AdversarialParameters,
  SovereignParameters,
  Team,
  TeamSummary,
  GauntletDefinition,
  GauntletSummary,
  WorkflowSummary,
  WorkflowDetail,
  WorkflowCreateRequest,
  WorkflowCreateResponse,
  WorkflowResults,
  WorkflowPlanResponse,
  WorkflowPlanUpdateRequest,
  EvaluatorListResponse,
  EvaluatorUploadResponse,
  ExecutionRecord,
  ExecutionListResponse,
  ExecutionCreateRequest,
  StatisticsSummary,
  PerformanceMetric,
  AnalyticsWorkflowMetric,
  AnalyticsKnowledgeStats,
  MonitoringDashboardMetrics,
  MonitoringAlert,
  MonitoringMetric,
  MonitoringService,
  MonitoringLogEntry,
  KnowledgeArtifact,
  KnowledgeGraph,
  KnowledgeStats,
  KnowledgeRecommendations,
  CrewAIWorkflowSummary,
  CrewAIWorkflowTicket,
  LeanAideStatusResponse,
  LeanAideExecuteResponse,
  LeanAideTreeListResponse,
  LeanAideTreeResponse,
  LeanAideProofListResponse,
  LeanAideProofResponse,
  VersionEntry,
  VersionCompareResult,
  ValidationRule,
  ValidationRunResult,
  ComplianceCheckResult,
  ParameterDefinition,
  ParameterValidationResult,
  IntegratedWorkflowRequest,
} from '@/types/openevolve';

// Re-export the canonical decomposition/adversarial types so UI components can
// import both the client and its contract from a single module.
export type {
  ModelConfig,
  Team,
  TeamRole,
  TeamType,
  TeamSummary,
  GauntletDefinition,
  GauntletSummary,
  GauntletRoundRule,
  GauntletType,
  GenerationMode,
  CollaborationMode,
  VotingStrategy,
  WorkflowSummary,
  WorkflowDetail,
  WorkflowCreateRequest,
  WorkflowCreateResponse,
  WorkflowResults,
  WorkflowSubProblem,
  WorkflowDecompositionPlan,
  WorkflowDependencyGraph,
  WorkflowPlanResponse,
  WorkflowPlanUpdateRequest,
  EvaluatorListResponse,
  EvaluatorUploadResponse,
  ExecutionRecord,
  ExecutionListResponse,
  ExecutionCreateRequest,
} from '@/types/openevolve';

// Create ApiClient with retry and timeout configuration
const openevolveClientConfig: ApiClientConfig = {
  baseURL: OPENEVOLVE_API_BASE_URL,
  timeout: 60000,      // 60 seconds (evolution can take longer)
  enableRetry: true,   // Enable retry logic
  maxRetries: 3,       // Maximum 3 retries
  retryDelay: 2000,    // Base delay 2 seconds
};

const openevolveApiClient = new ApiClient(openevolveClientConfig);

// ==================== Types ====================

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

export interface WorkflowCreate {
  name: string;
  description: string;
  workflow_type: WorkflowType;
  problem_statement?: string;
  content_type?: string;
  teams?: string[];
  gauntlets?: string[];
  metadata?: Record<string, unknown>;
  parameters?: Record<string, unknown>;
}

export interface WorkflowResponse {
  id: string;
  name: string;
  description: string;
  workflow_type: WorkflowType;
  problem_statement?: string;
  content_type?: string;
  teams?: string[];
  gauntlets?: string[];
  metadata?: Record<string, unknown>;
  parameters: Record<string, unknown>;
  status: WorkflowStatus;
  created_at: string;
  updated_at: string;
  started_at?: string;
  completed_at?: string;
  user_id?: string;
  tenant_id?: string;
}

export interface WorkflowListResponse {
  workflows: WorkflowResponse[];
  total: number;
  // NOTE: the backend `GET /workflows` handler returns only `{ workflows, total }`.
  // It does not paginate, so these are optional rather than guaranteed.
  page?: number;
  page_size?: number;
}

export interface WorkflowInputs {
  problem_statement: string;
  context?: string;
}

export interface ExecutionResponse {
  execution_id: string;
  workflow_id: string;
  status: ExecutionStatus;
  progress: number;
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

export interface ExecutionLogsResponse {
  logs: Array<Record<string, unknown>>;
  // The backend `GET /executions/{id}/logs` handler returns
  // `{ execution_id, logs, since }` and does not include a `total`.
  execution_id?: string;
  total?: number;
  since?: string;
}

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

export interface TeamCreate {
  name: string;
  description: string;
  members: TeamMember[];
}

export interface TeamResponse {
  id: string;
  name: string;
  description: string;
  members: TeamMember[];
  created_at: string;
}

export interface GauntletRound {
  name: string;
  quorum_threshold: number;
  confidence_threshold: number;
  evaluation_type: string;
}

export interface GauntletCreate {
  name: string;
  description: string;
  rounds: GauntletRound[];
}

export interface GauntletResponse {
  id: string;
  name: string;
  description: string;
  rounds: GauntletRound[];
  created_at: string;
}

export interface BubbleLabsControlCatalogResponse {
  success: boolean;
  components: Record<string, string[]>;
  auto_discovery?: {
    enabled?: boolean;
    summary?: Record<string, unknown>;
    components?: Record<string, string[]>;
  };
}

export interface BubbleLabsControlDiscoveryResponse {
  success: boolean;
  discovered_components?: number;
  discovered_actions?: number;
  scanned_paths?: string[];
  indexed_components?: number;
  [key: string]: unknown;
}

export interface BubbleLabsControlExecuteResponse {
  success: boolean;
  component?: string;
  action?: string;
  result?: Record<string, unknown>;
  error?: string;
  [key: string]: unknown;
}

export interface BubbleLabsWorkflowDefinitionSummary {
  id: string;
  name: string;
  description: string;
  workflow_type: string;
  created_at: number | string;
}

export interface BubbleLabsWorkflowDefinitionDetail extends BubbleLabsWorkflowDefinitionSummary {
  parameters: Record<string, unknown>;
  nodes: Array<Record<string, unknown>>;
  edges: Array<Record<string, unknown>>;
}

export interface BubbleLabsWorkflowInstanceSummary {
  instance_id: string;
  workflow_type: string;
  status: string;
  current_stage: string;
  problem_statement: string;
  start_time?: number;
  end_time?: number;
  progress?: number;
}

export interface BubbleLabsWorkflowInstanceStatus {
  instance_id: string;
  status: string;
  current_stage: string;
  progress: number;
  start_time?: number;
  end_time?: number;
  execution_time?: number;
  error_message?: string | null;
}

export interface BubbleLabsWorkflowInstanceDetail {
  status: BubbleLabsWorkflowInstanceStatus;
  parameters: Record<string, unknown>;
}

// ==================== Backend Response Normalization ====================

/**
 * The OpenEvolve backend (services/openevolve-api, FastAPI) uses different identifier
 * field names on the wire than this client's public contract:
 *
 *  - Workflow endpoints return `workflow_id` (see the `WorkflowResponse` pydantic
 *    model), but this client and its consumers read `.id`.
 *  - Execution endpoints return `id` (see `POST /executions`), but this client and
 *    its consumers read `.execution_id`.
 *
 * Without normalization, `workflow.id` / `execution.execution_id` silently evaluate
 * to `undefined`, which then gets interpolated into follow-up request URLs. These
 * helpers map the wire format onto the declared contract at the client boundary so
 * that consumers (and the declared TypeScript types) stay correct.
 */
type RawWorkflowResponse = Partial<WorkflowResponse> & {
  workflow_id?: string;
  current_stage?: string;
  progress?: number;
};

const normalizeWorkflow = (raw: RawWorkflowResponse): WorkflowResponse => ({
  ...(raw as WorkflowResponse),
  id: raw.id ?? raw.workflow_id ?? '',
});

type RawExecutionResponse = Partial<ExecutionResponse> & { id?: string };

const normalizeExecution = (raw: RawExecutionResponse): ExecutionResponse => ({
  ...(raw as ExecutionResponse),
  execution_id: raw.execution_id ?? raw.id ?? '',
  progress: raw.progress ?? 0,
});

type RawExecutionListResponse = {
  executions?: RawExecutionResponse[];
  total?: number;
};

const normalizeExecutionList = (raw: RawExecutionListResponse) => ({
  executions: (raw.executions ?? []).map(normalizeExecution),
  total: raw.total ?? 0,
});

// ==================== API Client ====================

/**
 * OpenEvolve API Client
 *
 * Provides full access to OpenEvolve workflow execution capabilities
 */
export const openevolveApi = {
  // ==================== Health & Info ====================

  /**
   * Check API health
   */
  health: async (): Promise<{
    status: string;
    service: string;
    version: string;
    features: {
      evolution: boolean;
      adversarial: boolean;
      sovereign: boolean;
    };
  }> => {
    logger.debug({
      msg: 'Checking OpenEvolve API health',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get('/health');
  },

  // ==================== Workflows ====================

  /**
   * Create a new workflow
   */
  createWorkflow: async (workflow: WorkflowCreate): Promise<WorkflowResponse> => {
    logger.debug({
      msg: 'Creating workflow',
      component: 'openevolveApi',
      workflow_type: workflow.workflow_type,
      name: workflow.name,
    });

    return normalizeWorkflow(
      await openevolveApiClient.post<RawWorkflowResponse>('/api/workflows', workflow)
    );
  },

  /**
   * Get all workflows
   */
  listWorkflows: async (page = 1, pageSize = 10): Promise<WorkflowListResponse> => {
    logger.debug({
      msg: 'Listing workflows',
      component: 'openevolveApi',
      page,
      page_size: pageSize,
    });

    const response = await openevolveApiClient.get<{
      workflows?: RawWorkflowResponse[];
      total?: number;
      page?: number;
      page_size?: number;
    }>(`/api/workflows?page=${page}&page_size=${pageSize}`);

    return {
      ...response,
      workflows: (response.workflows ?? []).map(normalizeWorkflow),
      total: response.total ?? 0,
    };
  },

  /**
   * Get a specific workflow
   */
  getWorkflow: async (workflowId: string): Promise<WorkflowResponse> => {
    logger.debug({
      msg: 'Getting workflow',
      component: 'openevolveApi',
      workflow_id: workflowId,
    });

    return normalizeWorkflow(
      await openevolveApiClient.get<RawWorkflowResponse>(
        `/api/workflows/${encodeURIComponent(workflowId)}`
      )
    );
  },

  /**
   * Update a workflow — NOT SUPPORTED BY THE BACKEND.
   *
   * `services/openevolve-api` exposes no `PUT /workflows/{workflow_id}` route:
   * the workflow routes are `POST /workflows` (create), `GET /workflows` (list),
   * `GET /workflows/{id}`, `POST /workflows/{id}/pause`, `POST /workflows/{id}/resume`,
   * `GET /workflows/{id}/results` and `DELETE /workflows/{id}`. The only mutable part
   * of a workflow is its decomposition plan
   * (`PUT /workflows/{workflow_id}/decomposition-plan`), exposed here as
   * `updateWorkflowPlan`.
   *
   * Rather than firing a request that can only 404/405, this rejects locally with an
   * actionable message. The method is kept (not deleted) so existing imports and the
   * client's public surface stay stable.
   */
  updateWorkflow: async (
    workflowId: string,
    updates: Partial<Pick<WorkflowCreate, 'name' | 'description' | 'parameters'>>
  ): Promise<WorkflowResponse> => {
    logger.warn({
      msg: 'Workflow update is not supported by the OpenEvolve backend; no request sent',
      component: 'openevolveApi',
      workflow_id: workflowId,
      attempted_fields: Object.keys(updates ?? {}).join(','),
    });

    throw new Error(
      `Workflow update is not supported by the OpenEvolve backend (no PUT /workflows/${workflowId} route). ` +
        'Use openevolveApi.updateWorkflowPlan() to edit the decomposition plan, or ' +
        'openevolveApi.createDecompositionWorkflow() to create a replacement workflow.'
    );
  },

  /**
   * Delete a workflow
   */
  deleteWorkflow: async (workflowId: string): Promise<{ message: string }> => {
    logger.debug({
      msg: 'Deleting workflow',
      component: 'openevolveApi',
      workflow_id: workflowId,
    });

    return openevolveApiClient.delete<{ message: string }>(
      `/api/workflows/${encodeURIComponent(workflowId)}`
    );
  },

  // ==================== Execution ====================

  /**
   * Execute a workflow
   */
  executeWorkflow: async (
    workflowId: string,
    inputs: WorkflowInputs
  ): Promise<ExecutionResponse> => {
    logger.info({
      msg: 'Executing workflow',
      component: 'openevolveApi',
      workflow_id: workflowId,
      problem_statement_length: inputs.problem_statement.length,
    });

    return normalizeExecution(
      await openevolveApiClient.post<RawExecutionResponse>(`/api/executions`, {
        workflow_id: workflowId,
        ...inputs,
      })
    );
  },

  /**
   * Get execution status
   */
  getExecutionStatus: async (executionId: string): Promise<ExecutionResponse> => {
    logger.debug({
      msg: 'Getting execution status',
      component: 'openevolveApi',
      execution_id: executionId,
    });

    return normalizeExecution(
      await openevolveApiClient.get<RawExecutionResponse>(
        `/api/executions/${encodeURIComponent(executionId)}`
      )
    );
  },

  /**
   * Get a single execution by ID
   */
  getExecution: async (executionId: string): Promise<ExecutionResponse> => {
    return normalizeExecution(
      await openevolveApiClient.get<RawExecutionResponse>(
        `/api/executions/${encodeURIComponent(executionId)}`
      )
    );
  },

  /**
   * List all executions with optional limit/offset
   */
  listExecutions: async (
    params?: { limit?: number; offset?: number }
  ): Promise<{ executions: ExecutionResponse[]; total: number }> => {
    const search = new URLSearchParams();
    if (params?.limit !== undefined) search.set('limit', String(params.limit));
    if (params?.offset !== undefined) search.set('offset', String(params.offset));
    const suffix = search.toString() ? `?${search.toString()}` : '';
    return normalizeExecutionList(
      await openevolveApiClient.get<RawExecutionListResponse>(`/api/executions${suffix}`)
    );
  },

  /**
   * Pause an execution
   */
  pauseExecution: async (executionId: string): Promise<ExecutionResponse> => {
    logger.info({
      msg: 'Pausing execution',
      component: 'openevolveApi',
      execution_id: executionId,
    });

    return normalizeExecution(
      await openevolveApiClient.post<RawExecutionResponse>(
        `/api/executions/${encodeURIComponent(executionId)}/pause`,
        {}
      )
    );
  },

  /**
   * Resume a paused execution
   */
  resumeExecution: async (executionId: string): Promise<ExecutionResponse> => {
    logger.info({
      msg: 'Resuming execution',
      component: 'openevolveApi',
      execution_id: executionId,
    });

    return normalizeExecution(
      await openevolveApiClient.post<RawExecutionResponse>(
        `/api/executions/${encodeURIComponent(executionId)}/resume`,
        {}
      )
    );
  },

  /**
   * Cancel an execution
   */
  cancelExecution: async (executionId: string): Promise<ExecutionResponse> => {
    logger.info({
      msg: 'Cancelling execution',
      component: 'openevolveApi',
      execution_id: executionId,
    });

    return normalizeExecution(
      await openevolveApiClient.post<RawExecutionResponse>(
        `/api/executions/${encodeURIComponent(executionId)}/cancel`,
        {}
      )
    );
  },

  /**
   * Get execution logs
   */
  getExecutionLogs: async (
    executionId: string,
    since?: string
  ): Promise<ExecutionLogsResponse> => {
    logger.debug({
      msg: 'Getting execution logs',
      component: 'openevolveApi',
      execution_id: executionId,
    });

    const basePath = `/api/executions/${encodeURIComponent(executionId)}/logs`;
    const url = since ? `${basePath}?since=${encodeURIComponent(since)}` : basePath;

    return openevolveApiClient.get<ExecutionLogsResponse>(url);
  },

  // ==================== Teams ====================

  /**
   * Create a team
   */
  createTeam: async (team: TeamCreate): Promise<TeamResponse> => {
    logger.debug({
      msg: 'Creating team',
      component: 'openevolveApi',
      name: team.name,
      member_count: team.members.length,
    });

    return openevolveApiClient.post<TeamResponse>('/api/teams', team);
  },

  /**
   * Get all teams
   */
  listTeams: async (): Promise<{ teams: TeamResponse[]; total: number }> => {
    logger.debug({
      msg: 'Listing teams',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<{ teams: TeamResponse[]; total: number }>('/api/teams');
  },

  /**
   * Get a specific team
   *
   * NOTE: the backend route is `GET /teams/{team_name}` — it looks teams up by
   * name, not by an opaque id, so `teamId` must be the team's name.
   */
  getTeam: async (teamId: string): Promise<TeamResponse> => {
    logger.debug({
      msg: 'Getting team',
      component: 'openevolveApi',
      team_id: teamId,
    });

    return openevolveApiClient.get<TeamResponse>(
      `/api/teams/${encodeURIComponent(teamId)}`
    );
  },

  // ==================== Teams (canonical decomposition surface) ====================
  //
  // The three methods above (`createTeam` / `listTeams` / `getTeam`) model the
  // legacy `openevolve-api` microservice shapes. The methods below mirror the
  // canonical SDK and the current FastAPI backend (`GET/POST/PUT/DELETE /teams`).

  /**
   * List teams as canonical summaries (`GET /teams`).
   */
  listTeamSummaries: async (): Promise<{ teams: TeamSummary[]; total: number }> => {
    logger.debug({
      msg: 'Listing team summaries',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<{ teams: TeamSummary[]; total: number }>('/api/teams');
  },

  /**
   * Get a full team definition by name (`GET /teams/{team_name}`).
   *
   * NOTE: the backend projects each member down to
   * `{ model_id, temperature, max_tokens }`, so `api_key` and the other
   * `ModelConfig` fields are absent on responses even though the canonical
   * `Team` type declares them for request bodies.
   */
  getTeamDefinition: async (teamName: string): Promise<Team> => {
    logger.debug({
      msg: 'Getting team definition',
      component: 'openevolveApi',
      team_name: teamName,
    });

    return openevolveApiClient.get<Team>(`/api/teams/${encodeURIComponent(teamName)}`);
  },

  /**
   * Create a team from a canonical `Team` definition (`POST /teams`).
   */
  createTeamDefinition: async (
    team: Team
  ): Promise<{ message: string; team_name: string }> => {
    logger.debug({
      msg: 'Creating team definition',
      component: 'openevolveApi',
      name: team.name,
      role: team.role,
      member_count: team.members.length,
    });

    return openevolveApiClient.post<{ message: string; team_name: string }>(
      '/api/teams',
      team
    );
  },

  /**
   * Update an existing team (`PUT /teams/{team_name}`).
   */
  updateTeam: async (
    teamName: string,
    team: Team
  ): Promise<{ message: string; team_name: string }> => {
    logger.debug({
      msg: 'Updating team',
      component: 'openevolveApi',
      team_name: teamName,
      member_count: team.members.length,
    });

    return openevolveApiClient.put<{ message: string; team_name: string }>(
      `/api/teams/${encodeURIComponent(teamName)}`,
      team
    );
  },

  /**
   * Delete a team (`DELETE /teams/{team_name}`, requires ADMIN role).
   *
   * NOTE: the backend returns `{ message, team_name }`; the canonical SDK still
   * declares `{ success: boolean }` for this route (stale).
   */
  deleteTeam: async (teamName: string): Promise<{ message: string; team_name: string }> => {
    logger.info({
      msg: 'Deleting team',
      component: 'openevolveApi',
      team_name: teamName,
    });

    return openevolveApiClient.delete<{ message: string; team_name: string }>(
      `/api/teams/${encodeURIComponent(teamName)}`
    );
  },

  // ==================== Gauntlets ====================

  /**
   * Create a gauntlet
   */
  createGauntlet: async (gauntlet: GauntletCreate): Promise<GauntletResponse> => {
    logger.debug({
      msg: 'Creating gauntlet',
      component: 'openevolveApi',
      name: gauntlet.name,
      rounds_count: gauntlet.rounds.length,
    });

    return openevolveApiClient.post<GauntletResponse>('/api/gauntlets', gauntlet);
  },

  /**
   * Get all gauntlets
   */
  listGauntlets: async (): Promise<{ gauntlets: GauntletResponse[]; total: number }> => {
    logger.debug({
      msg: 'Listing gauntlets',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<{ gauntlets: GauntletResponse[]; total: number }>(
      '/api/gauntlets'
    );
  },

  /**
   * Get a specific gauntlet
   *
   * NOTE: the backend route is `GET /gauntlets/{gauntlet_name}` — it looks gauntlets
   * up by name, not by an opaque id, so `gauntletId` must be the gauntlet's name.
   */
  getGauntlet: async (gauntletId: string): Promise<GauntletResponse> => {
    logger.debug({
      msg: 'Getting gauntlet',
      component: 'openevolveApi',
      gauntlet_id: gauntletId,
    });

    return openevolveApiClient.get<GauntletResponse>(
      `/api/gauntlets/${encodeURIComponent(gauntletId)}`
    );
  },

  // ============ Gauntlets (canonical adversarial surface) ============
  //
  // As with teams, `createGauntlet` / `listGauntlets` / `getGauntlet` above use
  // the legacy microservice shapes. The methods below mirror the canonical SDK
  // and the current FastAPI backend (`GET/POST/PUT/DELETE /gauntlets`).

  /**
   * List gauntlets as canonical summaries (`GET /gauntlets`).
   */
  listGauntletSummaries: async (): Promise<{
    gauntlets: GauntletSummary[];
    total: number;
  }> => {
    logger.debug({
      msg: 'Listing gauntlet summaries',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<{ gauntlets: GauntletSummary[]; total: number }>(
      '/api/gauntlets'
    );
  },

  /**
   * Get a full gauntlet definition by name (`GET /gauntlets/{gauntlet_name}`).
   *
   * NOTE: the backend projects each round down to
   * `{ round_number, quorum_required_approvals, quorum_from_panel_size, min_overall_confidence }`,
   * so the remaining `GauntletRoundRule` fields are absent on responses even
   * though the canonical type declares them for request bodies.
   */
  getGauntletDefinition: async (gauntletName: string): Promise<GauntletDefinition> => {
    logger.debug({
      msg: 'Getting gauntlet definition',
      component: 'openevolveApi',
      gauntlet_name: gauntletName,
    });

    return openevolveApiClient.get<GauntletDefinition>(
      `/api/gauntlets/${encodeURIComponent(gauntletName)}`
    );
  },

  /**
   * Create a gauntlet from a canonical definition (`POST /gauntlets`).
   */
  createGauntletDefinition: async (
    gauntlet: GauntletDefinition
  ): Promise<{ message: string; gauntlet_name: string }> => {
    logger.debug({
      msg: 'Creating gauntlet definition',
      component: 'openevolveApi',
      name: gauntlet.name,
      team_name: gauntlet.team_name,
      rounds_count: gauntlet.rounds.length,
    });

    return openevolveApiClient.post<{ message: string; gauntlet_name: string }>(
      '/api/gauntlets',
      gauntlet
    );
  },

  /**
   * Update an existing gauntlet (`PUT /gauntlets/{gauntlet_name}`).
   */
  updateGauntlet: async (
    gauntletName: string,
    gauntlet: GauntletDefinition
  ): Promise<{ message: string; gauntlet_name: string }> => {
    logger.debug({
      msg: 'Updating gauntlet',
      component: 'openevolveApi',
      gauntlet_name: gauntletName,
      rounds_count: gauntlet.rounds.length,
    });

    return openevolveApiClient.put<{ message: string; gauntlet_name: string }>(
      `/api/gauntlets/${encodeURIComponent(gauntletName)}`,
      gauntlet
    );
  },

  /**
   * Delete a gauntlet (`DELETE /gauntlets/{gauntlet_name}`, requires ADMIN role).
   *
   * NOTE: the backend returns `{ message, gauntlet_name }`; the canonical SDK
   * still declares `{ success: boolean }` for this route (stale).
   */
  deleteGauntlet: async (
    gauntletName: string
  ): Promise<{ message: string; gauntlet_name: string }> => {
    logger.info({
      msg: 'Deleting gauntlet',
      component: 'openevolveApi',
      gauntlet_name: gauntletName,
    });

    return openevolveApiClient.delete<{ message: string; gauntlet_name: string }>(
      `/api/gauntlets/${encodeURIComponent(gauntletName)}`
    );
  },

  // ==================== Evaluators ====================

  /**
   * List custom evaluators (`GET /evaluators`).
   *
   * Returns a map of evaluator id -> evaluator source code.
   */
  listEvaluators: async (): Promise<EvaluatorListResponse> => {
    logger.debug({
      msg: 'Listing evaluators',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<EvaluatorListResponse>('/api/evaluators');
  },

  /**
   * Upload a custom evaluator (`POST /evaluators`).
   */
  uploadEvaluator: async (payload: { code: string }): Promise<EvaluatorUploadResponse> => {
    logger.info({
      msg: 'Uploading evaluator',
      component: 'openevolveApi',
      code_length: payload.code.length,
    });

    return openevolveApiClient.post<EvaluatorUploadResponse>('/api/evaluators', payload);
  },

  /**
   * Delete a custom evaluator (`DELETE /evaluators/{evaluator_id}`, requires ADMIN role).
   */
  deleteEvaluator: async (
    evaluatorId: string
  ): Promise<{ success: boolean; evaluator_id: string }> => {
    logger.info({
      msg: 'Deleting evaluator',
      component: 'openevolveApi',
      evaluator_id: evaluatorId,
    });

    return openevolveApiClient.delete<{ success: boolean; evaluator_id: string }>(
      `/api/evaluators/${encodeURIComponent(evaluatorId)}`
    );
  },

  // ============ Decomposition Workflows (canonical surface) ============
  //
  // `createWorkflow` / `listWorkflows` / `getWorkflow` / `updateWorkflow` /
  // `deleteWorkflow` above use the legacy microservice shapes (and normalize
  // `workflow_id` -> `id`). The methods below mirror the canonical SDK and the
  // current FastAPI decomposition workflow routes.

  /**
   * List workflows as canonical summaries (`GET /workflows`).
   */
  listWorkflowSummaries: async (): Promise<{
    workflows: WorkflowSummary[];
    total: number;
  }> => {
    logger.debug({
      msg: 'Listing workflow summaries',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<{ workflows: WorkflowSummary[]; total: number }>(
      '/api/workflows'
    );
  },

  /**
   * Get canonical workflow detail (`GET /workflows/{workflow_id}`).
   */
  getWorkflowDetail: async (workflowId: string): Promise<WorkflowDetail> => {
    logger.debug({
      msg: 'Getting workflow detail',
      component: 'openevolveApi',
      workflow_id: workflowId,
    });

    return openevolveApiClient.get<WorkflowDetail>(
      `/api/workflows/${encodeURIComponent(workflowId)}`
    );
  },

  /**
   * Create a decomposition workflow (`POST /workflows`).
   *
   * Requires all five team names and all five gauntlet names to already exist.
   */
  createDecompositionWorkflow: async (
    payload: WorkflowCreateRequest
  ): Promise<WorkflowCreateResponse> => {
    logger.info({
      msg: 'Creating decomposition workflow',
      component: 'openevolveApi',
      problem_statement_length: payload.problem_statement.length,
      solver_team: payload.solver_team,
    });

    return openevolveApiClient.post<WorkflowCreateResponse>('/api/workflows', payload);
  },

  /**
   * Pause a running workflow (`POST /workflows/{workflow_id}/pause`).
   */
  pauseWorkflow: async (
    workflowId: string
  ): Promise<{ message: string; workflow_id: string; status: string }> => {
    logger.info({
      msg: 'Pausing workflow',
      component: 'openevolveApi',
      workflow_id: workflowId,
    });

    return openevolveApiClient.post<{
      message: string;
      workflow_id: string;
      status: string;
    }>(`/api/workflows/${encodeURIComponent(workflowId)}/pause`, {});
  },

  /**
   * Resume a paused workflow (`POST /workflows/{workflow_id}/resume`).
   */
  resumeWorkflow: async (
    workflowId: string
  ): Promise<{ message: string; workflow_id: string; status: string }> => {
    logger.info({
      msg: 'Resuming workflow',
      component: 'openevolveApi',
      workflow_id: workflowId,
    });

    return openevolveApiClient.post<{
      message: string;
      workflow_id: string;
      status: string;
    }>(`/api/workflows/${encodeURIComponent(workflowId)}/resume`, {});
  },

  /**
   * Get final and per-sub-problem results (`GET /workflows/{workflow_id}/results`).
   */
  getWorkflowResults: async (workflowId: string): Promise<WorkflowResults> => {
    logger.debug({
      msg: 'Getting workflow results',
      component: 'openevolveApi',
      workflow_id: workflowId,
    });

    return openevolveApiClient.get<WorkflowResults>(
      `/api/workflows/${encodeURIComponent(workflowId)}/results`
    );
  },

  /**
   * Get the decomposition plan and dependency graph
   * (`GET /workflows/{workflow_id}/decomposition-plan`).
   */
  getWorkflowPlan: async (workflowId: string): Promise<WorkflowPlanResponse> => {
    logger.debug({
      msg: 'Getting workflow decomposition plan',
      component: 'openevolveApi',
      workflow_id: workflowId,
    });

    return openevolveApiClient.get<WorkflowPlanResponse>(
      `/api/workflows/${encodeURIComponent(workflowId)}/decomposition-plan`
    );
  },

  /**
   * Update the decomposition plan / sub-problems
   * (`PUT /workflows/{workflow_id}/decomposition-plan`).
   *
   * NOTE: the backend responds with `{ message, execution_order }` (the freshly
   * computed topological order), not the full plan. Re-fetch via
   * `getWorkflowPlan` if the updated plan is needed.
   */
  updateWorkflowPlan: async (
    workflowId: string,
    payload: WorkflowPlanUpdateRequest
  ): Promise<{ message: string; execution_order: string[] }> => {
    logger.info({
      msg: 'Updating workflow decomposition plan',
      component: 'openevolveApi',
      workflow_id: workflowId,
      sub_problem_count: payload.sub_problems.length,
    });

    return openevolveApiClient.put<{ message: string; execution_order: string[] }>(
      `/api/workflows/${encodeURIComponent(workflowId)}/decomposition-plan`,
      payload
    );
  },

  // ============ Execution Records (canonical surface) ============

  /**
   * Create an execution record directly (`POST /executions`).
   *
   * Lower-level counterpart to `executeWorkflow`: returns the raw
   * `ExecutionRecord` (with `id`) instead of the normalized `ExecutionResponse`.
   */
  createExecution: async (payload: ExecutionCreateRequest): Promise<ExecutionRecord> => {
    logger.info({
      msg: 'Creating execution record',
      component: 'openevolveApi',
      workflow_id: payload.workflow_id,
    });

    return openevolveApiClient.post<ExecutionRecord>('/api/executions', payload);
  },

  /**
   * List raw execution records (`GET /executions`).
   *
   * Canonical counterpart to `listExecutions`, which normalizes records into
   * `ExecutionResponse`.
   */
  listExecutionRecords: async (params?: {
    limit?: number;
    offset?: number;
  }): Promise<ExecutionListResponse> => {
    const search = new URLSearchParams();
    if (params?.limit !== undefined) search.set('limit', String(params.limit));
    if (params?.offset !== undefined) search.set('offset', String(params.offset));
    const suffix = search.toString() ? `?${search.toString()}` : '';

    logger.debug({
      msg: 'Listing execution records',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<ExecutionListResponse>(`/api/executions${suffix}`);
  },

  // ==================== BubbleLabs Control Plane ====================

  /**
   * Get unified BubbleLabs/OpenEvolve control catalog.
   */
  getControlCatalog: async (): Promise<BubbleLabsControlCatalogResponse> => {
    logger.debug({
      msg: 'Fetching BubbleLabs control catalog',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<BubbleLabsControlCatalogResponse>(
      '/bubblelabs/control/catalog'
    );
  },

  /**
   * Refresh auto-discovered control components.
   */
  discoverControlComponents: async (
    force = false
  ): Promise<BubbleLabsControlDiscoveryResponse> => {
    logger.info({
      msg: 'Refreshing BubbleLabs control discovery',
      component: 'openevolveApi',
      force,
    });

    return openevolveApiClient.post<BubbleLabsControlDiscoveryResponse>(
      '/bubblelabs/control/discover',
      { force }
    );
  },

  /**
   * Execute a control action for a component.
   */
  executeControlAction: async (
    component: string,
    action: string,
    payload: Record<string, unknown> = {}
  ): Promise<BubbleLabsControlExecuteResponse> => {
    logger.info({
      msg: 'Executing BubbleLabs control action',
      component: 'openevolveApi',
      control_component: component,
      control_action: action,
    });

    return openevolveApiClient.post<BubbleLabsControlExecuteResponse>(
      '/bubblelabs/control/execute',
      {
        component,
        action,
        payload,
      }
    );
  },

  // ==================== BubbleLabs Workflow Lifecycle ====================

  listBubblelabsWorkflowDefinitions: async (): Promise<{
    definitions: BubbleLabsWorkflowDefinitionSummary[];
  }> => {
    logger.debug({
      msg: 'Listing BubbleLabs workflow definitions',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get('/bubblelabs/workflow-definitions');
  },

  getBubblelabsWorkflowDefinition: async (
    definitionId: string
  ): Promise<BubbleLabsWorkflowDefinitionDetail> => {
    logger.debug({
      msg: 'Getting BubbleLabs workflow definition',
      component: 'openevolveApi',
      definition_id: definitionId,
    });

    return openevolveApiClient.get(
      `/bubblelabs/workflow-definitions/${encodeURIComponent(definitionId)}`
    );
  },

  createBubblelabsWorkflowDefinition: async (payload: {
    name: string;
    description: string;
    workflow_type: string;
    parameters: Record<string, unknown>;
  }): Promise<{ definition_id: string }> => {
    logger.info({
      msg: 'Creating BubbleLabs workflow definition',
      component: 'openevolveApi',
      workflow_type: payload.workflow_type,
      name: payload.name,
    });

    return openevolveApiClient.post('/bubblelabs/workflow-definitions', payload);
  },

  listBubblelabsWorkflowInstances: async (): Promise<{
    instances: BubbleLabsWorkflowInstanceSummary[];
  }> => {
    logger.debug({
      msg: 'Listing BubbleLabs workflow instances',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get('/bubblelabs/workflow-instances');
  },

  getBubblelabsWorkflowInstance: async (
    instanceId: string
  ): Promise<BubbleLabsWorkflowInstanceDetail> => {
    logger.debug({
      msg: 'Getting BubbleLabs workflow instance',
      component: 'openevolveApi',
      instance_id: instanceId,
    });

    return openevolveApiClient.get(
      `/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}`
    );
  },

  createBubblelabsWorkflowInstance: async (payload: {
    definition_id: string;
    instance_name: string;
    inputs: Record<string, unknown>;
    parameters?: Record<string, unknown>;
  }): Promise<{ instance_id: string }> => {
    logger.info({
      msg: 'Creating BubbleLabs workflow instance',
      component: 'openevolveApi',
      definition_id: payload.definition_id,
      instance_name: payload.instance_name,
    });

    return openevolveApiClient.post('/bubblelabs/workflow-instances', payload);
  },

  syncBubblelabsWorkflowInstanceParameters: async (
    instanceId: string,
    payload: { parameters: Record<string, unknown> }
  ): Promise<{ message: string; instance_id: string; updated_count: number }> => {
    logger.info({
      msg: 'Syncing BubbleLabs workflow instance parameters',
      component: 'openevolveApi',
      instance_id: instanceId,
    });

    return openevolveApiClient.post(
      `/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/parameters`,
      payload
    );
  },

  startBubblelabsWorkflowInstance: async (
    instanceId: string
  ): Promise<Record<string, unknown>> => {
    return openevolveApiClient.post(
      `/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/start`,
      {}
    );
  },

  pauseBubblelabsWorkflowInstance: async (
    instanceId: string
  ): Promise<Record<string, unknown>> => {
    return openevolveApiClient.post(
      `/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/pause`,
      {}
    );
  },

  resumeBubblelabsWorkflowInstance: async (
    instanceId: string
  ): Promise<Record<string, unknown>> => {
    return openevolveApiClient.post(
      `/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/resume`,
      {}
    );
  },

  stopBubblelabsWorkflowInstance: async (
    instanceId: string
  ): Promise<Record<string, unknown>> => {
    return openevolveApiClient.post(
      `/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/stop`,
      {}
    );
  },

  cancelBubblelabsWorkflowInstance: async (
    instanceId: string
  ): Promise<Record<string, unknown>> => {
    return openevolveApiClient.post(
      `/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/cancel`,
      {}
    );
  },

  restartBubblelabsWorkflowInstance: async (
    instanceId: string
  ): Promise<Record<string, unknown>> => {
    return openevolveApiClient.post(
      `/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/restart`,
      {}
    );
  },

  deleteBubblelabsWorkflowInstance: async (
    instanceId: string
  ): Promise<Record<string, unknown>> => {
    return openevolveApiClient.delete(
      `/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}`
    );
  },

  // ==================== Monitoring ====================

  /**
   * Get the monitoring dashboard snapshot (`GET /monitoring/dashboard`).
   */
  getMonitoringDashboard: async (): Promise<MonitoringDashboardMetrics> => {
    logger.debug({
      msg: 'Getting monitoring dashboard',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<MonitoringDashboardMetrics>('/api/monitoring/dashboard');
  },

  /**
   * Get active monitoring alerts (`GET /monitoring/alerts`).
   */
  getMonitoringAlerts: async (): Promise<{ alerts: MonitoringAlert[] }> => {
    logger.debug({
      msg: 'Getting monitoring alerts',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<{ alerts: MonitoringAlert[] }>('/api/monitoring/alerts');
  },

  /**
   * Get monitored service health (`GET /monitoring/services`).
   */
  getMonitoringServices: async (): Promise<{
    services: MonitoringService[];
    timestamp?: string;
  }> => {
    logger.debug({
      msg: 'Getting monitoring services',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<{
      services: MonitoringService[];
      timestamp?: string;
    }>('/api/monitoring/services');
  },

  /**
   * Get monitoring logs (`GET /monitoring/logs`).
   */
  getMonitoringLogs: async (
    limit = 200,
    source?: string
  ): Promise<{ entries: MonitoringLogEntry[]; total: number }> => {
    const search = new URLSearchParams();
    if (limit) search.set('limit', String(limit));
    if (source) search.set('source', source);
    const suffix = search.toString() ? `?${search.toString()}` : '';

    logger.debug({
      msg: 'Getting monitoring logs',
      component: 'openevolveApi',
      limit,
      source,
    });

    return openevolveApiClient.get<{ entries: MonitoringLogEntry[]; total: number }>(
      `/api/monitoring/logs${suffix}`
    );
  },

  /**
   * Query monitoring metrics (`GET /monitoring/metrics`).
   */
  getMonitoringMetrics: async (params: {
    name?: string;
    start_time?: string;
    end_time?: string;
  }): Promise<{ metrics: MonitoringMetric[] }> => {
    const search = new URLSearchParams();
    if (params.name) search.set('name', params.name);
    if (params.start_time) search.set('start_time', params.start_time);
    if (params.end_time) search.set('end_time', params.end_time);
    const suffix = search.toString() ? `?${search.toString()}` : '';

    logger.debug({
      msg: 'Getting monitoring metrics',
      component: 'openevolveApi',
      name: params.name,
    });

    return openevolveApiClient.get<{ metrics: MonitoringMetric[] }>(
      `/api/monitoring/metrics${suffix}`
    );
  },

  /**
   * Get raw monitoring health (`GET /monitoring/health`).
   */
  getMonitoringHealth: async (): Promise<Record<string, unknown>> => {
    logger.debug({
      msg: 'Getting monitoring health',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<Record<string, unknown>>('/api/monitoring/health');
  },

  // ==================== Analytics ====================

  /**
   * Get aggregate statistics (`GET /statistics`).
   */
  getStatistics: async (): Promise<StatisticsSummary> => {
    logger.debug({
      msg: 'Getting statistics',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<StatisticsSummary>('/api/statistics');
  },

  /**
   * Get performance metrics (`GET /analytics/performance-metrics`).
   */
  getPerformanceMetrics: async (
    entityType?: string,
    limit = 200
  ): Promise<{ metrics: PerformanceMetric[]; total: number }> => {
    const search = new URLSearchParams();
    if (entityType) search.set('entity_type', entityType);
    if (limit) search.set('limit', String(limit));
    const suffix = search.toString() ? `?${search.toString()}` : '';

    logger.debug({
      msg: 'Getting performance metrics',
      component: 'openevolveApi',
      entity_type: entityType,
      limit,
    });

    return openevolveApiClient.get<{ metrics: PerformanceMetric[]; total: number }>(
      `/api/analytics/performance-metrics${suffix}`
    );
  },

  /**
   * Get knowledge analytics stats (`GET /analytics/knowledge-stats`).
   */
  getAnalyticsKnowledgeStats: async (): Promise<AnalyticsKnowledgeStats> => {
    logger.debug({
      msg: 'Getting analytics knowledge stats',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<AnalyticsKnowledgeStats>('/api/analytics/knowledge-stats');
  },

  /**
   * Get workflow analytics metrics (`GET /analytics/workflow-metrics`).
   */
  getWorkflowMetrics: async (): Promise<{
    metrics: AnalyticsWorkflowMetric[];
    total: number;
  }> => {
    logger.debug({
      msg: 'Getting workflow metrics',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<{
      metrics: AnalyticsWorkflowMetric[];
      total: number;
    }>('/api/analytics/workflow-metrics');
  },

  // ==================== Knowledge Base ====================

  /**
   * List knowledge artifacts (`GET /knowledge/artifacts`).
   */
  listKnowledgeArtifacts: async (): Promise<{ artifacts: KnowledgeArtifact[] }> => {
    logger.debug({
      msg: 'Listing knowledge artifacts',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<{ artifacts: KnowledgeArtifact[] }>('/api/knowledge/artifacts');
  },

  /**
   * Get a single knowledge artifact (`GET /knowledge/artifacts/{artifact_id}`).
   */
  getKnowledgeArtifact: async (artifactId: string): Promise<KnowledgeArtifact> => {
    logger.debug({
      msg: 'Getting knowledge artifact',
      component: 'openevolveApi',
      artifact_id: artifactId,
    });

    return openevolveApiClient.get<KnowledgeArtifact>(
      `/api/knowledge/artifacts/${encodeURIComponent(artifactId)}`
    );
  },

  /**
   * Create a knowledge artifact (`POST /knowledge/artifacts`).
   */
  createKnowledgeArtifact: async (
    payload: Record<string, unknown>
  ): Promise<KnowledgeArtifact> => {
    logger.info({
      msg: 'Creating knowledge artifact',
      component: 'openevolveApi',
      artifact_type: payload.artifact_type,
    });

    return openevolveApiClient.post<KnowledgeArtifact>('/api/knowledge/artifacts', payload);
  },

  /**
   * Delete a knowledge artifact (`DELETE /knowledge/artifacts/{artifact_id}`).
   */
  deleteKnowledgeArtifact: async (
    artifactId: string
  ): Promise<{ success: boolean }> => {
    logger.info({
      msg: 'Deleting knowledge artifact',
      component: 'openevolveApi',
      artifact_id: artifactId,
    });

    return openevolveApiClient.delete<{ success: boolean }>(
      `/api/knowledge/artifacts/${encodeURIComponent(artifactId)}`
    );
  },

  /**
   * Search knowledge base (`POST /knowledge/search`).
   */
  searchKnowledge: async (
    payload: Record<string, unknown>
  ): Promise<{ results: KnowledgeArtifact[] }> => {
    logger.debug({
      msg: 'Searching knowledge',
      component: 'openevolveApi',
    });

    return openevolveApiClient.post<{ results: KnowledgeArtifact[] }>(
      '/api/knowledge/search',
      payload
    );
  },

  /**
   * Get the knowledge graph (`GET /knowledge/graph`).
   */
  getKnowledgeGraph: async (): Promise<KnowledgeGraph> => {
    logger.debug({
      msg: 'Getting knowledge graph',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<KnowledgeGraph>('/api/knowledge/graph');
  },

  /**
   * Get knowledge base stats (`GET /knowledge/stats`).
   */
  getKnowledgeStats: async (): Promise<KnowledgeStats> => {
    logger.debug({
      msg: 'Getting knowledge stats',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<KnowledgeStats>('/api/knowledge/stats');
  },

  /**
   * Get knowledge recommendations (`POST /knowledge/recommendations`).
   */
  getKnowledgeRecommendations: async (
    payload: Record<string, unknown>
  ): Promise<KnowledgeRecommendations> => {
    logger.debug({
      msg: 'Getting knowledge recommendations',
      component: 'openevolveApi',
    });

    return openevolveApiClient.post<KnowledgeRecommendations>(
      '/api/knowledge/recommendations',
      payload
    );
  },

  /**
   * Export the knowledge base (`GET /knowledge/export`).
   */
  exportKnowledgeBase: async (): Promise<Record<string, unknown>> => {
    logger.info({
      msg: 'Exporting knowledge base',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<Record<string, unknown>>('/api/knowledge/export');
  },

  /**
   * Import the knowledge base (`POST /knowledge/import`).
   */
  importKnowledgeBase: async (
    payload: Record<string, unknown>
  ): Promise<{ success: boolean }> => {
    logger.info({
      msg: 'Importing knowledge base',
      component: 'openevolveApi',
    });

    return openevolveApiClient.post<{ success: boolean }>('/api/knowledge/import', payload);
  },

  // ==================== CrewAI ====================

  /**
   * List CrewAI workflows (`GET /crewai/workflows`).
   */
  listCrewaiWorkflows: async (): Promise<{
    workflows: CrewAIWorkflowSummary[];
    total: number;
  }> => {
    logger.debug({
      msg: 'Listing CrewAI workflows',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<{
      workflows: CrewAIWorkflowSummary[];
      total: number;
    }>('/api/crewai/workflows');
  },

  /**
   * Get a CrewAI workflow (`GET /crewai/workflows/{workflow_id}`).
   */
  getCrewaiWorkflow: async (workflowId: string): Promise<Record<string, unknown>> => {
    logger.debug({
      msg: 'Getting CrewAI workflow',
      component: 'openevolveApi',
      workflow_id: workflowId,
    });

    return openevolveApiClient.get<Record<string, unknown>>(
      `/api/crewai/workflows/${encodeURIComponent(workflowId)}`
    );
  },

  /**
   * Get CrewAI workflow tickets (`GET /crewai/workflows/{workflow_id}/tickets`).
   */
  getCrewaiWorkflowTickets: async (
    workflowId: string
  ): Promise<{
    tickets: CrewAIWorkflowTicket[];
    total: number;
    status_breakdown?: Record<string, number>;
  }> => {
    logger.debug({
      msg: 'Getting CrewAI workflow tickets',
      component: 'openevolveApi',
      workflow_id: workflowId,
    });

    return openevolveApiClient.get<{
      tickets: CrewAIWorkflowTicket[];
      total: number;
      status_breakdown?: Record<string, number>;
    }>(`/api/crewai/workflows/${encodeURIComponent(workflowId)}/tickets`);
  },

  // ==================== LeanAide (BubbleLabs integration) ====================

  /**
   * Get LeanAide status (`GET /bubblelabs/leanaide/status`).
   */
  bubblelabsLeanAideStatus: async (): Promise<LeanAideStatusResponse> => {
    logger.debug({
      msg: 'Getting LeanAide status',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<LeanAideStatusResponse>('/api/bubblelabs/leanaide/status');
  },

  /**
   * Execute a LeanAide task (`POST /bubblelabs/leanaide/execute`).
   */
  bubblelabsLeanAideExecute: async (payload: {
    task_type: string;
    payload: Record<string, unknown>;
  }): Promise<LeanAideExecuteResponse> => {
    logger.info({
      msg: 'Executing LeanAide task',
      component: 'openevolveApi',
      task_type: payload.task_type,
    });

    return openevolveApiClient.post<LeanAideExecuteResponse>(
      '/api/bubblelabs/leanaide/execute',
      payload
    );
  },

  /**
   * List LeanAide proof trees (`GET /bubblelabs/leanaide/trees`).
   */
  bubblelabsLeanAideTrees: async (): Promise<LeanAideTreeListResponse> => {
    logger.debug({
      msg: 'Listing LeanAide trees',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<LeanAideTreeListResponse>('/api/bubblelabs/leanaide/trees');
  },

  /**
   * Get a LeanAide proof tree (`GET /bubblelabs/leanaide/trees/{tree_id}`).
   */
  bubblelabsLeanAideTree: async (treeId: string): Promise<LeanAideTreeResponse> => {
    logger.debug({
      msg: 'Getting LeanAide tree',
      component: 'openevolveApi',
      tree_id: treeId,
    });

    return openevolveApiClient.get<LeanAideTreeResponse>(
      `/api/bubblelabs/leanaide/trees/${encodeURIComponent(treeId)}`
    );
  },

  /**
   * List LeanAide proofs (`GET /bubblelabs/leanaide/proofs`).
   */
  bubblelabsLeanAideProofs: async (): Promise<LeanAideProofListResponse> => {
    logger.debug({
      msg: 'Listing LeanAide proofs',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<LeanAideProofListResponse>('/api/bubblelabs/leanaide/proofs');
  },

  /**
   * Get a LeanAide proof (`GET /bubblelabs/leanaide/proofs/{proof_id}`).
   */
  bubblelabsLeanAideProof: async (proofId: string): Promise<LeanAideProofResponse> => {
    logger.debug({
      msg: 'Getting LeanAide proof',
      component: 'openevolveApi',
      proof_id: proofId,
    });

    return openevolveApiClient.get<LeanAideProofResponse>(
      `/api/bubblelabs/leanaide/proofs/${encodeURIComponent(proofId)}`
    );
  },

  /**
   * Prove a theorem via LeanAide (`POST /bubblelabs/leanaide/prove`).
   */
  bubblelabsLeanAideProve: async (payload: {
    theorem: string;
  }): Promise<Record<string, unknown>> => {
    logger.info({
      msg: 'Proving theorem via LeanAide',
      component: 'openevolveApi',
    });

    return openevolveApiClient.post<Record<string, unknown>>(
      '/api/bubblelabs/leanaide/prove',
      payload
    );
  },

  // ==================== Version Control ====================

  /**
   * List protocol versions (`GET /version-control/versions`).
   */
  listVersions: async (): Promise<{
    versions: VersionEntry[];
    current_version_id?: string | null;
  }> => {
    logger.debug({
      msg: 'Listing versions',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<{
      versions: VersionEntry[];
      current_version_id?: string | null;
    }>('/api/version-control/versions');
  },

  /**
   * Get a protocol version (`GET /version-control/versions/{version_id}`).
   */
  getVersion: async (versionId: string): Promise<VersionEntry> => {
    logger.debug({
      msg: 'Getting version',
      component: 'openevolveApi',
      version_id: versionId,
    });

    return openevolveApiClient.get<VersionEntry>(
      `/api/version-control/versions/${encodeURIComponent(versionId)}`
    );
  },

  /**
   * Get the currently loaded version (`GET /version-control/current`).
   */
  getCurrentVersion: async (): Promise<{ current: VersionEntry | null }> => {
    logger.debug({
      msg: 'Getting current version',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<{ current: VersionEntry | null }>(
      '/api/version-control/current'
    );
  },

  /**
   * Create a protocol version (`POST /version-control/versions`).
   */
  createVersion: async (payload: {
    protocol_text: string;
    version_name?: string;
    comment?: string;
    author?: string;
  }): Promise<{ version_id: string; version: VersionEntry }> => {
    logger.info({
      msg: 'Creating version',
      component: 'openevolveApi',
      version_name: payload.version_name,
    });

    return openevolveApiClient.post<{ version_id: string; version: VersionEntry }>(
      '/api/version-control/versions',
      payload
    );
  },

  /**
   * Load a protocol version (`POST /version-control/versions/{version_id}/load`).
   */
  loadVersion: async (
    versionId: string
  ): Promise<{ loaded: boolean; current: VersionEntry | null }> => {
    logger.info({
      msg: 'Loading version',
      component: 'openevolveApi',
      version_id: versionId,
    });

    return openevolveApiClient.post<{ loaded: boolean; current: VersionEntry | null }>(
      `/api/version-control/versions/${encodeURIComponent(versionId)}/load`,
      {}
    );
  },

  /**
   * Branch a protocol version (`POST /version-control/versions/{version_id}/branch`).
   */
  branchVersion: async (
    versionId: string,
    payload: { new_version_name: string }
  ): Promise<{ version_id: string; version: VersionEntry }> => {
    logger.info({
      msg: 'Branching version',
      component: 'openevolveApi',
      version_id: versionId,
      new_version_name: payload.new_version_name,
    });

    return openevolveApiClient.post<{ version_id: string; version: VersionEntry }>(
      `/api/version-control/versions/${encodeURIComponent(versionId)}/branch`,
      payload
    );
  },

  /**
   * Compare two protocol versions (`POST /version-control/compare`).
   */
  compareVersions: async (payload: {
    version_id_1: string;
    version_id_2: string;
  }): Promise<VersionCompareResult> => {
    logger.debug({
      msg: 'Comparing versions',
      component: 'openevolveApi',
      version_id_1: payload.version_id_1,
      version_id_2: payload.version_id_2,
    });

    return openevolveApiClient.post<VersionCompareResult>(
      '/api/version-control/compare',
      payload
    );
  },

  /**
   * Delete a protocol version (`DELETE /version-control/versions/{version_id}`).
   */
  deleteVersion: async (versionId: string): Promise<{ deleted: boolean }> => {
    logger.info({
      msg: 'Deleting version',
      component: 'openevolveApi',
      version_id: versionId,
    });

    return openevolveApiClient.delete<{ deleted: boolean }>(
      `/api/version-control/versions/${encodeURIComponent(versionId)}`
    );
  },

  // ==================== Validation ====================

  /**
   * List validation rules (`GET /validation/rules`).
   */
  listValidationRules: async (): Promise<{
    rules: Record<string, ValidationRule>;
    rule_names: string[];
  }> => {
    logger.debug({
      msg: 'Listing validation rules',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<{
      rules: Record<string, ValidationRule>;
      rule_names: string[];
    }>('/api/validation/rules');
  },

  /**
   * Get a validation rule (`GET /validation/rules/{rule_name}`).
   */
  getValidationRule: async (
    ruleName: string
  ): Promise<{ name: string; rule: ValidationRule }> => {
    logger.debug({
      msg: 'Getting validation rule',
      component: 'openevolveApi',
      rule_name: ruleName,
    });

    return openevolveApiClient.get<{ name: string; rule: ValidationRule }>(
      `/api/validation/rules/${encodeURIComponent(ruleName)}`
    );
  },

  /**
   * Create a validation rule (`POST /validation/rules`).
   */
  createValidationRule: async (payload: {
    name: string;
    max_length?: number | null;
    min_length?: number | null;
    required_keywords?: string[];
    forbidden_patterns?: string[];
    required_sections?: string[];
  }): Promise<{ created: boolean; rule_name: string; rule: ValidationRule }> => {
    logger.info({
      msg: 'Creating validation rule',
      component: 'openevolveApi',
      name: payload.name,
    });

    return openevolveApiClient.post<{
      created: boolean;
      rule_name: string;
      rule: ValidationRule;
    }>('/api/validation/rules', payload);
  },

  /**
   * Update a validation rule (`PUT /validation/rules/{rule_name}`).
   */
  updateValidationRule: async (
    ruleName: string,
    payload: {
      name?: string;
      max_length?: number | null;
      min_length?: number | null;
      required_keywords?: string[] | null;
      forbidden_patterns?: string[] | null;
      required_sections?: string[] | null;
    }
  ): Promise<{ updated: boolean; rule_name: string; rule: ValidationRule }> => {
    logger.info({
      msg: 'Updating validation rule',
      component: 'openevolveApi',
      rule_name: ruleName,
    });

    return openevolveApiClient.put<{
      updated: boolean;
      rule_name: string;
      rule: ValidationRule;
    }>(`/api/validation/rules/${encodeURIComponent(ruleName)}`, payload);
  },

  /**
   * Delete a validation rule (`DELETE /validation/rules/{rule_name}`).
   */
  deleteValidationRule: async (
    ruleName: string
  ): Promise<{ deleted: boolean; rule_name: string }> => {
    logger.info({
      msg: 'Deleting validation rule',
      component: 'openevolveApi',
      rule_name: ruleName,
    });

    return openevolveApiClient.delete<{ deleted: boolean; rule_name: string }>(
      `/api/validation/rules/${encodeURIComponent(ruleName)}`
    );
  },

  /**
   * Run validation against content (`POST /validation/run`).
   */
  runValidation: async (
    payload: { content: string; rule_names: string[] }
  ): Promise<ValidationRunResult> => {
    logger.info({
      msg: 'Running validation',
      component: 'openevolveApi',
      rule_names: payload.rule_names,
    });

    return openevolveApiClient.post<ValidationRunResult>('/api/validation/run', payload);
  },

  /**
   * Run a compliance check (`POST /validation/compliance`).
   */
  runComplianceCheck: async (
    payload: { content: string; framework?: string }
  ): Promise<ComplianceCheckResult> => {
    logger.info({
      msg: 'Running compliance check',
      component: 'openevolveApi',
      framework: payload.framework,
    });

    return openevolveApiClient.post<ComplianceCheckResult>(
      '/api/validation/compliance',
      payload
    );
  },

  // ==================== Parameters ====================

  /**
   * Get the full parameter schema (`GET /parameters/schema`).
   */
  getParameterSchema: async (): Promise<{ parameters: ParameterDefinition[] }> => {
    logger.debug({
      msg: 'Getting parameter schema',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<{ parameters: ParameterDefinition[] }>(
      '/api/parameters/schema'
    );
  },

  /**
   * Get default parameter values (`GET /parameters/defaults`).
   */
  getParameterDefaults: async (): Promise<Record<string, unknown>> => {
    logger.debug({
      msg: 'Getting parameter defaults',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<Record<string, unknown>>('/api/parameters/defaults');
  },

  /**
   * Get parameter categories (`GET /parameters/categories`).
   */
  getParameterCategories: async (): Promise<{ categories: string[] }> => {
    logger.debug({
      msg: 'Getting parameter categories',
      component: 'openevolveApi',
    });

    return openevolveApiClient.get<{ categories: string[] }>('/api/parameters/categories');
  },

  /**
   * Validate parameter values (`POST /parameters/validate`).
   */
  validateParameters: async (
    payload: Record<string, unknown>
  ): Promise<ParameterValidationResult> => {
    logger.debug({
      msg: 'Validating parameters',
      component: 'openevolveApi',
    });

    return openevolveApiClient.post<ParameterValidationResult>(
      '/api/parameters/validate',
      payload
    );
  },

  // ==================== Integrated Run ====================

  /**
   * Run an integrated workflow (`POST /integrated/run`).
   */
  runIntegratedWorkflow: async (
    payload: IntegratedWorkflowRequest
  ): Promise<Record<string, unknown>> => {
    logger.info({
      msg: 'Running integrated workflow',
      component: 'openevolveApi',
      content_type: payload.content_type,
      max_iterations: payload.max_iterations,
    });

    return openevolveApiClient.post<Record<string, unknown>>('/api/integrated/run', payload);
  },
};

// ==================== Convenience Functions ====================

/**
 * Quick evolution workflow execution
 *
 * Creates an evolution workflow and executes it in one call
 */
export const executeEvolution = async (
  problemStatement: string,
  parameters?: Partial<EvolutionParameters>,
  context?: string
): Promise<ExecutionResponse> => {
  logger.info({
    msg: 'Executing quick evolution',
    component: 'openevolveApi',
    problem_statement_length: problemStatement.length,
  });

  // Create workflow
  const workflow = await openevolveApi.createWorkflow({
    name: 'Quick Evolution',
    description: 'Auto-generated evolution workflow',
    workflow_type: 'evolution',
    parameters: parameters || {},
  });

  // Execute workflow
  return openevolveApi.executeWorkflow(workflow.id, {
    problem_statement: problemStatement,
    context,
  });
};

/**
 * Quick adversarial workflow execution
 */
export const executeAdversarial = async (
  problemStatement: string,
  parameters?: Partial<AdversarialParameters>,
  context?: string
): Promise<ExecutionResponse> => {
  logger.info({
    msg: 'Executing quick adversarial',
    component: 'openevolveApi',
    problem_statement_length: problemStatement.length,
  });

  const workflow = await openevolveApi.createWorkflow({
    name: 'Quick Adversarial',
    description: 'Auto-generated adversarial workflow',
    workflow_type: 'adversarial',
    parameters: parameters || {},
  });

  return openevolveApi.executeWorkflow(workflow.id, {
    problem_statement: problemStatement,
    context,
  });
};

/**
 * Quick sovereign workflow execution
 */
export const executeSovereign = async (
  problemStatement: string,
  parameters?: Partial<SovereignParameters>,
  context?: string
): Promise<ExecutionResponse> => {
  logger.info({
    msg: 'Executing quick sovereign',
    component: 'openevolveApi',
    problem_statement_length: problemStatement.length,
  });

  const workflow = await openevolveApi.createWorkflow({
    name: 'Quick Sovereign',
    description: 'Auto-generated sovereign workflow',
    workflow_type: 'sovereign',
    parameters: parameters || {},
  });

  return openevolveApi.executeWorkflow(workflow.id, {
    problem_statement: problemStatement,
    context,
  });
};
