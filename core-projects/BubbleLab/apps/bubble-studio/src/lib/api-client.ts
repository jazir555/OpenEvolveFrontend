/**
 * OpenEvolve API Client
 * Handles all HTTP requests to the backend API
 *
 * Backend contract (`engines/other/api_server.py`):
 * - An `@app.middleware("http")` hook (`rewrite_api_prefix`) strips a leading `/api`,
 *   so the `/api/...` paths below reach the canonical unprefixed routes
 *   (`/api/workflows` -> `/workflows`, `/api/teams` -> `/teams`, ...).
 * - Existing routes used here: `GET|POST /workflows`, `GET|DELETE /workflows/{id}`,
 *   `POST /workflows/{id}/pause`, `POST /workflows/{id}/resume`,
 *   `GET /workflows/{id}/results`, `GET|POST /teams`, `GET|PUT|DELETE /teams/{name}`,
 *   `GET|POST /gauntlets`, `GET|PUT|DELETE /gauntlets/{name}`, `GET /health`.
 * - Routes that DO NOT exist (see `UNSUPPORTED_ENDPOINTS` below): `PUT /workflows/{id}`,
 *   `POST /workflows/{id}/start`, `POST /workflows/{id}/stop` and every `/settings/*`
 *   endpoint. The corresponding methods are kept for API stability but fail fast
 *   locally instead of issuing a request that can only 404.
 *
 * NOTE: `/teams/{...}` and `/gauntlets/{...}` are keyed by NAME on the backend
 * (`GET /teams/{team_name}`), matching `services/openevolveApi.ts`. The `teamId` /
 * `gauntletId` parameter names below are historical: pass the name.
 */

import {
  Workflow,
  Team,
  Gauntlet,
  CreateWorkflowRequest,
  UpdateWorkflowRequest,
  CreateTeamRequest,
  UpdateTeamRequest,
  CreateGauntletRequest,
  UpdateGauntletRequest,
  LLMConfig,
  ICRConfig,
  DeterminismDefaults,
  DecompositionDefaults,
  ExecutionResult,
  WorkflowListResponse,
  TeamListResponse,
  GauntletListResponse,
  ApiResponse,
  ListQueryParams,
} from '../types/api';
import { OPENEVOLVE_API_BASE_URL, OPENEVOLVE_API_KEY } from '../env';

// ============================================================================
// Configuration
// ============================================================================

const API_BASE_URL =
  OPENEVOLVE_API_BASE_URL ||
  import.meta.env.VITE_API_BASE_URL ||
  'http://localhost:8000';
const API_TIMEOUT = 30000; // 30 seconds

const resolveOpenEvolveApiKey = (): string | undefined => {
  if (OPENEVOLVE_API_KEY && OPENEVOLVE_API_KEY.trim().length > 0) {
    return OPENEVOLVE_API_KEY.trim();
  }
  try {
    const stored = globalThis.localStorage?.getItem('openevolve_api_key');
    if (stored && stored.trim().length > 0) {
      return stored.trim();
    }
  } catch {
    // ignore localStorage access errors
  }
  return undefined;
};

// ============================================================================
// Unsupported backend endpoints
// ============================================================================

/**
 * Endpoints this client used to call that have NO route in
 * `engines/other/api_server.py`. Kept as documentation for the guards below.
 */
export const UNSUPPORTED_ENDPOINTS = [
  'PUT /api/workflows/{id}',
  'POST /api/workflows/{id}/start',
  'POST /api/workflows/{id}/stop',
  'GET|PUT /api/settings/llm',
  'GET|PUT /api/settings/icr',
  'GET|PUT /api/settings/determinism',
  'GET|PUT /api/settings/decomposition',
] as const;

/** `false` while the backend exposes no `/settings/*` routes. */
export const SETTINGS_API_AVAILABLE = false;

/** `false` while the backend exposes no `POST /workflows/{id}/start` route. */
export const WORKFLOW_START_SUPPORTED = false;

/** `false` while the backend exposes no `POST /workflows/{id}/stop` route. */
export const WORKFLOW_STOP_SUPPORTED = false;

/** `false` while the backend exposes no `PUT /workflows/{id}` route. */
export const WORKFLOW_UPDATE_SUPPORTED = false;

export const SETTINGS_UNAVAILABLE_MESSAGE =
  'Server-side settings are not available: this OpenEvolve backend exposes no /api/settings/* endpoints. ' +
  'Values are kept locally in this browser only.';

export const WORKFLOW_START_UNAVAILABLE_MESSAGE =
  'Start is not supported by the backend: there is no POST /api/workflows/{id}/start route. ' +
  'Creating a workflow (POST /api/workflows) registers it, and a run is launched through ' +
  'POST /api/executions (see openevolveApi.executeWorkflow).';

export const WORKFLOW_STOP_UNAVAILABLE_MESSAGE =
  'Stop is not supported by the backend: there is no POST /api/workflows/{id}/stop route. ' +
  'Use pause/resume on the workflow, or cancel the run via POST /api/executions/{id}/cancel ' +
  '(see openevolveApi.cancelExecution).';

/**
 * Thrown instead of performing a request against a route the backend does not have,
 * so callers get an immediate, actionable failure rather than an opaque 404.
 */
export class UnsupportedEndpointError extends Error {
  readonly endpoint: string;

  constructor(endpoint: string, message: string) {
    super(message);
    this.name = 'UnsupportedEndpointError';
    this.endpoint = endpoint;
  }
}

// ============================================================================
// API Client Class
// ============================================================================

class ApiClient {
  private baseURL: string;
  private timeout: number;
  private defaultHeaders: Record<string, string>;

  constructor(baseURL: string, timeout: number = API_TIMEOUT) {
    this.baseURL = baseURL;
    this.timeout = timeout;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
  }

  /**
   * Set authentication token
   */
  setAuthToken(token: string) {
    this.defaultHeaders['Authorization'] = `Bearer ${token}`;
  }

  /**
   * Clear authentication token
   */
  clearAuthToken() {
    delete this.defaultHeaders['Authorization'];
  }

  /**
   * Make an HTTP request
   */
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    const apiKey = resolveOpenEvolveApiKey();
    const config: RequestInit = {
      ...options,
      headers: {
        ...this.defaultHeaders,
        ...(apiKey ? { 'X-API-Key': apiKey } : {}),
        ...options.headers,
      },
    };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        ...config,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const error = await response.json().catch(() => ({
          error: response.statusText,
        }));
        throw new Error(error.error || error.detail || 'API request failed');
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error('Request timeout');
        }
        throw error;
      }

      throw new Error('Unknown error occurred');
    }
  }

  /**
   * GET request
   */
  async get<T>(endpoint: string, params?: Record<string, any>): Promise<T> {
    const queryString = params
      ? `?${new URLSearchParams(params).toString()}`
      : '';
    return this.request<T>(`${endpoint}${queryString}`, {
      method: 'GET',
    });
  }

  /**
   * POST request
   */
  async post<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  /**
   * PUT request
   */
  async put<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  /**
   * PATCH request
   */
  async patch<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PATCH',
      body: JSON.stringify(data),
    });
  }

  /**
   * DELETE request
   */
  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'DELETE',
    });
  }

  // ========================================================================
  // Health Check
  // ========================================================================

  async healthCheck() {
    return this.get('/health');
  }

  // ========================================================================
  // Workflows
  // ========================================================================

  async getWorkflows(params?: ListQueryParams) {
    return this.get<WorkflowListResponse>('/api/workflows', params);
  }

  async getWorkflow(workflowId: string) {
    return this.get<Workflow>(`/api/workflows/${workflowId}`);
  }

  async createWorkflow(data: CreateWorkflowRequest) {
    return this.post<Workflow>('/api/workflows', data);
  }

  /**
   * NOT SUPPORTED: the backend has no `PUT /workflows/{workflow_id}` route.
   * Only the decomposition plan is mutable
   * (`PUT /workflows/{id}/decomposition-plan`, see `openevolveApi.updateWorkflowPlan`).
   */
  async updateWorkflow(
    workflowId: string,
    _data: UpdateWorkflowRequest
  ): Promise<Workflow> {
    throw new UnsupportedEndpointError(
      `PUT /api/workflows/${workflowId}`,
      'Workflow update is not supported by the backend: there is no PUT /api/workflows/{id} route. ' +
        'Edit the decomposition plan via openevolveApi.updateWorkflowPlan() instead.'
    );
  }

  async deleteWorkflow(workflowId: string) {
    return this.delete<{ message: string }>(`/api/workflows/${workflowId}`);
  }

  /**
   * NOT SUPPORTED: the backend has no `POST /workflows/{workflow_id}/start` route.
   *
   * `POST /workflows` only registers a workflow (status `created`, stage
   * `INITIALIZING`) and does not begin execution, so it is not a drop-in
   * replacement for starting an existing workflow id. Runs are launched through
   * `POST /executions` (`openevolveApi.executeWorkflow`), which returns an
   * execution record rather than a `Workflow`.
   */
  async startWorkflow(workflowId: string): Promise<Workflow> {
    throw new UnsupportedEndpointError(
      `POST /api/workflows/${workflowId}/start`,
      WORKFLOW_START_UNAVAILABLE_MESSAGE
    );
  }

  async pauseWorkflow(workflowId: string) {
    return this.post<Workflow>(`/api/workflows/${workflowId}/pause`);
  }

  async resumeWorkflow(workflowId: string) {
    return this.post<Workflow>(`/api/workflows/${workflowId}/resume`);
  }

  /**
   * NOT SUPPORTED: the backend has no `POST /workflows/{workflow_id}/stop` route.
   * Pause/resume exist; a running execution can be cancelled through
   * `POST /executions/{id}/cancel` (`openevolveApi.cancelExecution`).
   */
  async stopWorkflow(workflowId: string): Promise<Workflow> {
    throw new UnsupportedEndpointError(
      `POST /api/workflows/${workflowId}/stop`,
      WORKFLOW_STOP_UNAVAILABLE_MESSAGE
    );
  }

  async getWorkflowResults(workflowId: string) {
    return this.get<ExecutionResult>(`/api/workflows/${workflowId}/results`);
  }

  // ========================================================================
  // Teams
  // ========================================================================

  async getTeams(params?: ListQueryParams) {
    return this.get<TeamListResponse>('/api/teams', params);
  }

  async getTeam(teamId: string) {
    return this.get<Team>(`/api/teams/${teamId}`);
  }

  async createTeam(data: CreateTeamRequest) {
    return this.post<Team>('/api/teams', data);
  }

  async updateTeam(teamId: string, data: UpdateTeamRequest) {
    return this.put<Team>(`/api/teams/${teamId}`, data);
  }

  async deleteTeam(teamId: string) {
    return this.delete<{ message: string }>(`/api/teams/${teamId}`);
  }

  // ========================================================================
  // Gauntlets
  // ========================================================================

  async getGauntlets(params?: ListQueryParams) {
    return this.get<GauntletListResponse>('/api/gauntlets', params);
  }

  async getGauntlet(gauntletId: string) {
    return this.get<Gauntlet>(`/api/gauntlets/${gauntletId}`);
  }

  async createGauntlet(data: CreateGauntletRequest) {
    return this.post<Gauntlet>('/api/gauntlets', data);
  }

  async updateGauntlet(gauntletId: string, data: UpdateGauntletRequest) {
    return this.put<Gauntlet>(`/api/gauntlets/${gauntletId}`, data);
  }

  async deleteGauntlet(gauntletId: string) {
    return this.delete<{ message: string }>(`/api/gauntlets/${gauntletId}`);
  }

  // ========================================================================
  // Settings — NOT SUPPORTED BY THE BACKEND
  //
  // `engines/other/api_server.py` defines no `/settings/*` routes (and the
  // `/api` prefix middleware only rewrites paths, it does not add handlers), so
  // every method below would 404. They are retained so callers keep compiling,
  // but they fail fast with `UnsupportedEndpointError` instead of hitting the
  // network. Consumers should check `SETTINGS_API_AVAILABLE` first — see
  // `components/settings/SettingsPanel.tsx`, which keeps these settings in the
  // local config store and shows a "not available" note.
  // ========================================================================

  async getLLMConfig(): Promise<LLMConfig> {
    throw new UnsupportedEndpointError(
      'GET /api/settings/llm',
      SETTINGS_UNAVAILABLE_MESSAGE
    );
  }

  async updateLLMConfig(_data: Partial<LLMConfig>): Promise<LLMConfig> {
    throw new UnsupportedEndpointError(
      'PUT /api/settings/llm',
      SETTINGS_UNAVAILABLE_MESSAGE
    );
  }

  async getICRConfig(): Promise<ICRConfig> {
    throw new UnsupportedEndpointError(
      'GET /api/settings/icr',
      SETTINGS_UNAVAILABLE_MESSAGE
    );
  }

  async updateICRConfig(_data: Partial<ICRConfig>): Promise<ICRConfig> {
    throw new UnsupportedEndpointError(
      'PUT /api/settings/icr',
      SETTINGS_UNAVAILABLE_MESSAGE
    );
  }

  async getDeterminismDefaults(): Promise<DeterminismDefaults> {
    throw new UnsupportedEndpointError(
      'GET /api/settings/determinism',
      SETTINGS_UNAVAILABLE_MESSAGE
    );
  }

  async updateDeterminismDefaults(
    _data: Partial<DeterminismDefaults>
  ): Promise<DeterminismDefaults> {
    throw new UnsupportedEndpointError(
      'PUT /api/settings/determinism',
      SETTINGS_UNAVAILABLE_MESSAGE
    );
  }

  async getDecompositionDefaults(): Promise<DecompositionDefaults> {
    throw new UnsupportedEndpointError(
      'GET /api/settings/decomposition',
      SETTINGS_UNAVAILABLE_MESSAGE
    );
  }

  async updateDecompositionDefaults(
    _data: Partial<DecompositionDefaults>
  ): Promise<DecompositionDefaults> {
    throw new UnsupportedEndpointError(
      'PUT /api/settings/decomposition',
      SETTINGS_UNAVAILABLE_MESSAGE
    );
  }
}

// ============================================================================
// Export singleton instance
// ============================================================================

export const apiClient = new ApiClient(API_BASE_URL);

// ============================================================================
// Utility functions
// ============================================================================

/**
 * Set auth token from Clerk
 */
export const setAuthToken = (token: string) => {
  apiClient.setAuthToken(token);
};

/**
 * Clear auth token
 */
export const clearAuthToken = () => {
  apiClient.clearAuthToken();
};

export default apiClient;
