import type {
  Team,
  TeamSummary,
  GauntletDefinition,
  GauntletSummary,
  WorkflowSummary,
  WorkflowDetail,
  WorkflowCreateRequest,
  WorkflowCreateResponse,
  WorkflowResults,
} from "./types";
import type {
  AuditLogEntry,
  StatisticsSummary,
  AdaptiveMdapDashboard,
  AdaptiveMdapProfiles,
} from "./types";
import type { IcrOverview, IcrComponents, IcrRefinements } from "./types";

export interface ApiConfig {
  baseUrl?: string;
  apiKey?: string;
}

const resolveBaseUrl = (override?: string) => {
  if (override) {
    return override;
  }
  const fromWindow = (globalThis as any)?.OPENEVOLVE_API_BASE as string | undefined;
  if (fromWindow) {
    return fromWindow;
  }
  return "";
};

const resolveApiKey = (override?: string) => {
  if (override) {
    return override;
  }
  try {
    const stored = globalThis?.localStorage?.getItem("openevolve_api_key");
    if (stored) {
      return stored;
    }
  } catch (_) {
    // ignore storage errors
  }
  return "";
};

const buildHeaders = (apiKey?: string) => {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
  return headers;
};

async function request<T>(
  path: string,
  options: RequestInit = {},
  config: ApiConfig = {},
): Promise<T> {
  const baseUrl = resolveBaseUrl(config.baseUrl);
  const apiKey = resolveApiKey(config.apiKey);
  const url = `${baseUrl}${path}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      ...buildHeaders(apiKey),
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const openevolveApi = {
  listTeams: (config?: ApiConfig) =>
    request<{ teams: TeamSummary[]; total: number }>("/teams", {}, config),
  getTeam: (teamName: string, config?: ApiConfig) =>
    request<Team>(`/teams/${encodeURIComponent(teamName)}`, {}, config),
  createTeam: (team: Team, config?: ApiConfig) =>
    request<{ message: string; team_name: string }>(
      "/teams",
      { method: "POST", body: JSON.stringify(team) },
      config,
    ),
  updateTeam: (teamName: string, team: Team, config?: ApiConfig) =>
    request<{ message: string; team_name: string }>(
      `/teams/${encodeURIComponent(teamName)}`,
      { method: "PUT", body: JSON.stringify(team) },
      config,
    ),
  deleteTeam: (teamName: string, config?: ApiConfig) =>
    request<{ success: boolean }>(
      `/teams/${encodeURIComponent(teamName)}`,
      { method: "DELETE" },
      config,
    ),
  listGauntlets: (config?: ApiConfig) =>
    request<{ gauntlets: GauntletSummary[]; total: number }>("/gauntlets", {}, config),
  getGauntlet: (name: string, config?: ApiConfig) =>
    request<GauntletDefinition>(`/gauntlets/${encodeURIComponent(name)}`, {}, config),
  createGauntlet: (gauntlet: GauntletDefinition, config?: ApiConfig) =>
    request<{ message: string; gauntlet_name: string }>(
      "/gauntlets",
      { method: "POST", body: JSON.stringify(gauntlet) },
      config,
    ),
  updateGauntlet: (name: string, gauntlet: GauntletDefinition, config?: ApiConfig) =>
    request<{ message: string; gauntlet_name: string }>(
      `/gauntlets/${encodeURIComponent(name)}`,
      { method: "PUT", body: JSON.stringify(gauntlet) },
      config,
    ),
  deleteGauntlet: (name: string, config?: ApiConfig) =>
    request<{ success: boolean }>(
      `/gauntlets/${encodeURIComponent(name)}`,
      { method: "DELETE" },
      config,
    ),
  listWorkflows: (config?: ApiConfig) =>
    request<{ workflows: WorkflowSummary[]; total: number }>("/workflows", {}, config),
  getWorkflow: (workflowId: string, config?: ApiConfig) =>
    request<WorkflowDetail>(`/workflows/${encodeURIComponent(workflowId)}`, {}, config),
  createWorkflow: (payload: WorkflowCreateRequest, config?: ApiConfig) =>
    request<WorkflowCreateResponse>(
      "/workflows",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  pauseWorkflow: (workflowId: string, config?: ApiConfig) =>
    request<{ message: string; workflow_id: string; status: string }>(
      `/workflows/${encodeURIComponent(workflowId)}/pause`,
      { method: "POST" },
      config,
    ),
  resumeWorkflow: (workflowId: string, config?: ApiConfig) =>
    request<{ message: string; workflow_id: string; status: string }>(
      `/workflows/${encodeURIComponent(workflowId)}/resume`,
      { method: "POST" },
      config,
    ),
  deleteWorkflow: (workflowId: string, config?: ApiConfig) =>
    request<{ message: string; workflow_id: string }>(
      `/workflows/${encodeURIComponent(workflowId)}`,
      { method: "DELETE" },
      config,
    ),
  getWorkflowResults: (workflowId: string, config?: ApiConfig) =>
    request<WorkflowResults>(
      `/workflows/${encodeURIComponent(workflowId)}/results`,
      {},
      config,
    ),
  getStatistics: (config?: ApiConfig) => request<StatisticsSummary>("/statistics", {}, config),
  listAuditLogs: (limit = 200, config?: ApiConfig) =>
    request<{ logs: AuditLogEntry[]; total: number }>(`/audit/logs?limit=${limit}`, {}, config),
  getIcrOverview: (config?: ApiConfig) => request<IcrOverview>("/icr/analytics/overview", {}, config),
  getIcrComponents: (config?: ApiConfig) =>
    request<IcrComponents>("/icr/analytics/components", {}, config),
  getIcrRefinements: (config?: ApiConfig) =>
    request<IcrRefinements>("/icr/analytics/refinements", {}, config),
  getAdaptiveMdapHealth: (config?: ApiConfig) =>
    request<{ status: string; version?: string; details?: any }>("/adaptive-mdap/health", {}, config),
  getAdaptiveMdapDashboard: (config?: ApiConfig) =>
    request<AdaptiveMdapDashboard>("/adaptive-mdap/dashboard", {}, config),
  getAdaptiveMdapProfiles: (config?: ApiConfig) =>
    request<AdaptiveMdapProfiles>("/adaptive-mdap/profiles", {}, config),
  getAdaptiveMdapProfileConfig: (profileName: string, config?: ApiConfig) =>
    request<Record<string, unknown>>(
      `/adaptive-mdap/profiles/${encodeURIComponent(profileName)}`,
      {},
      config,
    ),
  calculateAdaptiveMdapCost: (
    payload: { num_problems: number; workload_distribution?: Record<string, number>; model?: string },
    config?: ApiConfig,
  ) =>
    request<Record<string, unknown>>(
      "/adaptive-mdap/cost",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  classifyAdaptiveMdapComplexity: (
    payload: {
      description: string;
      domain?: string;
      depth?: number;
      dependencies?: string[];
      constraints?: string[];
      success_criteria?: string[];
      context?: Record<string, unknown>;
    },
    config?: ApiConfig,
  ) =>
    request<Record<string, unknown>>(
      "/adaptive-mdap/complexity",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  allocateAdaptiveMdapResources: (
    payload: { complexity_score: number; context?: Record<string, unknown> },
    config?: ApiConfig,
  ) =>
    request<Record<string, unknown>>(
      "/adaptive-mdap/allocate",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  getHealth: (config?: ApiConfig) => request<Record<string, unknown>>("/health", {}, config),
};
