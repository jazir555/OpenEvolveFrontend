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
import type {
  KnowledgeArtifact,
  KnowledgeGraph,
  KnowledgeStats,
  KnowledgeRecommendations,
  PromptMap,
  ContentTemplate,
  ProtocolValidationResult,
  AutoApprovalConfig,
  AutoApprovalTestResult,
  AutoApprovalAuditEntry,
  WorkflowTemplate,
  ProviderSummary,
  ParameterDefinition,
  ParameterValidationResult,
} from "./types";
import type {
  PerformanceMetric,
  AnalyticsWorkflowMetric,
  AnalyticsKnowledgeStats,
  MonitoringDashboardMetrics,
  MonitoringAlert,
  MonitoringMetric,
  MonitoringService,
  MonitoringLogEntry,
  WorkflowTelemetry,
  CrewAIWorkflowSummary,
  CrewAIWorkflowTicket,
  WorkflowPlanResponse,
  SovereignPlan,
} from "./types";

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
  try {
    const stored = globalThis?.localStorage?.getItem("openevolve_api_base");
    if (stored) {
      return stored;
    }
  } catch (_) {
    // ignore storage errors
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
  getPerformanceMetrics: (entityType?: string, limit = 200, config?: ApiConfig) => {
    const params = new URLSearchParams();
    if (entityType) {
      params.set("entity_type", entityType);
    }
    if (limit) {
      params.set("limit", String(limit));
    }
    const suffix = params.toString() ? `?${params.toString()}` : "";
    return request<{ metrics: PerformanceMetric[]; total: number }>(
      `/analytics/performance-metrics${suffix}`,
      {},
      config,
    );
  },
  getAnalyticsKnowledgeStats: (config?: ApiConfig) =>
    request<AnalyticsKnowledgeStats>("/analytics/knowledge-stats", {}, config),
  getWorkflowPlan: (workflowId: string, config?: ApiConfig) =>
    request<WorkflowPlanResponse>(
      `/workflows/${encodeURIComponent(workflowId)}/decomposition-plan`,
      {},
      config,
    ),
  getWorkflowTelemetry: (workflowId: string, config?: ApiConfig) =>
    request<WorkflowTelemetry>(
      `/workflows/${encodeURIComponent(workflowId)}/telemetry`,
      {},
      config,
    ),
  getWorkflowMetrics: (config?: ApiConfig) =>
    request<{ metrics: AnalyticsWorkflowMetric[]; total: number }>(
      "/analytics/workflow-metrics",
      {},
      config,
    ),
  listSovereignPlans: (config?: ApiConfig) =>
    request<{ plans: SovereignPlan[] }>("/sovereign/plans", {}, config),
  getMonitoringDashboard: (config?: ApiConfig) =>
    request<MonitoringDashboardMetrics>("/monitoring/dashboard", {}, config),
  getMonitoringAlerts: (config?: ApiConfig) =>
    request<{ alerts: MonitoringAlert[] }>("/monitoring/alerts", {}, config),
  getMonitoringServices: (config?: ApiConfig) =>
    request<{ services: MonitoringService[]; timestamp?: string }>(
      "/monitoring/services",
      {},
      config,
    ),
  getMonitoringLogs: (limit = 200, source?: string, config?: ApiConfig) => {
    const params = new URLSearchParams();
    if (limit) params.set("limit", String(limit));
    if (source) params.set("source", source);
    const suffix = params.toString() ? `?${params.toString()}` : "";
    return request<{ entries: MonitoringLogEntry[]; total: number }>(
      `/monitoring/logs${suffix}`,
      {},
      config,
    );
  },
  getMonitoringMetrics: (
    params: { name?: string; start_time?: string; end_time?: string },
    config?: ApiConfig,
  ) => {
    const search = new URLSearchParams();
    if (params.name) search.set("name", params.name);
    if (params.start_time) search.set("start_time", params.start_time);
    if (params.end_time) search.set("end_time", params.end_time);
    const suffix = search.toString() ? `?${search.toString()}` : "";
    return request<{ metrics: MonitoringMetric[] }>(`/monitoring/metrics${suffix}`, {}, config);
  },
  getMonitoringHealth: (config?: ApiConfig) =>
    request<Record<string, unknown>>("/monitoring/health", {}, config),
  listCrewaiWorkflows: (config?: ApiConfig) =>
    request<{ workflows: CrewAIWorkflowSummary[]; total: number }>("/crewai/workflows", {}, config),
  getCrewaiWorkflow: (workflowId: string, config?: ApiConfig) =>
    request<Record<string, unknown>>(
      `/crewai/workflows/${encodeURIComponent(workflowId)}`,
      {},
      config,
    ),
  getCrewaiWorkflowTickets: (workflowId: string, config?: ApiConfig) =>
    request<{ tickets: CrewAIWorkflowTicket[]; total: number; status_breakdown?: Record<string, number> }>(
      `/crewai/workflows/${encodeURIComponent(workflowId)}/tickets`,
      {},
      config,
    ),
  listPrompts: (config?: ApiConfig) =>
    request<{ prompts: PromptMap }>("/prompts", {}, config),
  savePrompt: (payload: { name: string; content: string }, config?: ApiConfig) =>
    request<{ success: boolean; name: string }>(
      "/prompts",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  deletePrompt: (promptName: string, config?: ApiConfig) =>
    request<{ success: boolean }>(
      `/prompts/${encodeURIComponent(promptName)}`,
      { method: "DELETE" },
      config,
    ),
  listContentTemplates: (config?: ApiConfig) =>
    request<{ templates: string[] }>("/content/templates", {}, config),
  getContentTemplate: (templateName: string, config?: ApiConfig) =>
    request<ContentTemplate>(
      `/content/templates/${encodeURIComponent(templateName)}`,
      {},
      config,
    ),
  createContentTemplate: (payload: { name: string; content: string }, config?: ApiConfig) =>
    request<{ template: Record<string, unknown> }>(
      "/content/templates",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  validateProtocol: (payload: { protocol_text: string; validation_type?: string }, config?: ApiConfig) =>
    request<ProtocolValidationResult>(
      "/content/validate",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
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

  // Knowledge Base
  listKnowledgeArtifacts: (config?: ApiConfig) =>
    request<{ artifacts: KnowledgeArtifact[] }>("/knowledge/artifacts", {}, config),
  getKnowledgeArtifact: (artifactId: string, config?: ApiConfig) =>
    request<KnowledgeArtifact>(`/knowledge/artifacts/${encodeURIComponent(artifactId)}`, {}, config),
  createKnowledgeArtifact: (payload: Record<string, unknown>, config?: ApiConfig) =>
    request<KnowledgeArtifact>(
      "/knowledge/artifacts",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  deleteKnowledgeArtifact: (artifactId: string, config?: ApiConfig) =>
    request<{ success: boolean }>(
      `/knowledge/artifacts/${encodeURIComponent(artifactId)}`,
      { method: "DELETE" },
      config,
    ),
  searchKnowledge: (payload: Record<string, unknown>, config?: ApiConfig) =>
    request<{ results: KnowledgeArtifact[] }>(
      "/knowledge/search",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  getKnowledgeGraph: (config?: ApiConfig) =>
    request<KnowledgeGraph>("/knowledge/graph", {}, config),
  getKnowledgeStats: (config?: ApiConfig) => request<KnowledgeStats>("/knowledge/stats", {}, config),
  getKnowledgeRecommendations: (payload: Record<string, unknown>, config?: ApiConfig) =>
    request<KnowledgeRecommendations>(
      "/knowledge/recommendations",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  exportKnowledgeBase: (config?: ApiConfig) =>
    request<Record<string, unknown>>("/knowledge/export", {}, config),
  importKnowledgeBase: (payload: Record<string, unknown>, config?: ApiConfig) =>
    request<{ success: boolean }>(
      "/knowledge/import",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),

  // Auto-Approval
  getAutoApprovalConfig: (config?: ApiConfig) =>
    request<AutoApprovalConfig>("/auto-approval/config", {}, config),
  updateAutoApprovalConfig: (payload: AutoApprovalConfig, config?: ApiConfig) =>
    request<AutoApprovalConfig>(
      "/auto-approval/config",
      { method: "PUT", body: JSON.stringify(payload) },
      config,
    ),
  testAutoApproval: (payload: Record<string, unknown>, config?: ApiConfig) =>
    request<{ results: AutoApprovalTestResult[] }>(
      "/auto-approval/test",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  getAutoApprovalAudit: (config?: ApiConfig) =>
    request<{ logs: AutoApprovalAuditEntry[] }>(
      "/auto-approval/audit",
      {},
      config,
    ),

  // Workflow Templates
  listWorkflowTemplates: (config?: ApiConfig) =>
    request<{ templates: WorkflowTemplate[] }>("/workflow-templates", {}, config),
  createWorkflowTemplate: (payload: Record<string, unknown>, config?: ApiConfig) =>
    request<WorkflowTemplate>(
      "/workflow-templates",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  updateWorkflowTemplate: (templateId: string, payload: Record<string, unknown>, config?: ApiConfig) =>
    request<WorkflowTemplate>(
      `/workflow-templates/${encodeURIComponent(templateId)}`,
      { method: "PUT", body: JSON.stringify(payload) },
      config,
    ),
  deleteWorkflowTemplate: (templateId: string, config?: ApiConfig) =>
    request<{ success: boolean }>(
      `/workflow-templates/${encodeURIComponent(templateId)}`,
      { method: "DELETE" },
      config,
    ),
  exportWorkflowTemplates: (config?: ApiConfig) =>
    request<Record<string, unknown>>("/workflow-templates/export", {}, config),
  importWorkflowTemplates: (payload: Record<string, unknown>, config?: ApiConfig) =>
    request<{ success: boolean }>(
      "/workflow-templates/import",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),

  // Providers and parameters
  listProviders: (config?: ApiConfig) =>
    request<{ providers: ProviderSummary[] }>("/providers", {}, config),
  getProviderModels: (providerId: string, apiKey?: string, config?: ApiConfig) =>
    request<{ models: string[] }>(
      `/providers/${encodeURIComponent(providerId)}/models`,
      {
        method: "POST",
        body: JSON.stringify({ api_key: apiKey }),
      },
      config,
    ),
  getParameterSchema: (config?: ApiConfig) =>
    request<{ parameters: ParameterDefinition[] }>("/parameters/schema", {}, config),
  getParameterDefaults: (config?: ApiConfig) =>
    request<Record<string, unknown>>("/parameters/defaults", {}, config),
  validateParameters: (payload: Record<string, unknown>, config?: ApiConfig) =>
    request<ParameterValidationResult>(
      "/parameters/validate",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  getParameterCategories: (config?: ApiConfig) =>
    request<{ categories: string[] }>("/parameters/categories", {}, config),

  // Sovereign dashboard
  getSovereignHealth: (config?: ApiConfig) =>
    request<Record<string, unknown>>("/sovereign/health", {}, config),
  listSovereignProblems: (config?: ApiConfig) =>
    request<{ problems: Record<string, unknown>[] }>("/sovereign/problems", {}, config),
  listSovereignPlans: (config?: ApiConfig) =>
    request<{ plans: Record<string, unknown>[] }>("/sovereign/plans", {}, config),

  // Suggestions
  getContentSuggestions: (payload: Record<string, unknown>, config?: ApiConfig) =>
    request<{ suggestions: string[] }>(
      "/suggestions/content",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  getContentClassification: (payload: Record<string, unknown>, config?: ApiConfig) =>
    request<Record<string, unknown>>(
      "/suggestions/classification",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  getSecuritySuggestions: (payload: Record<string, unknown>, config?: ApiConfig) =>
    request<{ vulnerabilities: string[] }>(
      "/suggestions/security",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  getImprovementPotential: (payload: Record<string, unknown>, config?: ApiConfig) =>
    request<{ score: number }>(
      "/suggestions/improvement",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
};
