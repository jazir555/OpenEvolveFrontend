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
import type {
  EvaluatorListResponse,
  EvaluatorUploadResponse,
  WorkflowPlanUpdateRequest,
  WorkflowResourceUsageResponse,
  WorkflowResourceOptimizationResponse,
  IntegratedWorkflowRequest,
  ModelOrchestrationListResponse,
  ModelOrchestrationRegisterRequest,
  ModelOrchestrationRegisterResponse,
  ModelOrchestrationEnsembleRequest,
  ModelOrchestrationEnsembleResponse,
  BubbleLabsStatusResponse,
  BubbleLabsInitializeResponse,
  BubbleLabsActionResponse,
  VersionEntry,
  VersionCompareResult,
  ValidationRule,
  ValidationRunResult,
  ComplianceCheckResult,
  WorkflowDefinitionSummary,
  WorkflowDefinitionDetail,
  WorkflowInstanceSummary,
  WorkflowInstanceDetail,
  MakerToolListResponse,
  MakerToolResponse,
  MakerExecutionResponse,
  MakerDelegationListResponse,
  KnowledgeExplorerQueryResponse,
  KnowledgeExplorerExtractResponse,
  KnowledgeExplorerHistoryResponse,
  LeanAideStatusResponse,
  LeanAideExecuteResponse,
  LeanAideTreeListResponse,
  LeanAideTreeResponse,
  LeanAideProofListResponse,
  LeanAideProofResponse,
  EvolutionRunResponse,
  EvolutionRunStatus,
  EvolutionRunListResponse,
  AdversarialRunResponse,
  AdversarialRunStatus,
  AdversarialRunListResponse,
} from "./types";

import { apiLogger, LogContext } from '../../../glue/lib/structuredLogger';
import { retryWithBackoff, RetryConfig } from '../../../glue/lib/retry';
import { CircuitBreaker, CircuitState } from '../../../glue/lib/circuit-breaker';

export interface ApiConfig {
  baseUrl?: string;
  apiKey?: string;
  timeout?: number; // MANDATORY per Law of Configuration Explicitness
}

// Correlation ID generator for request tracking
const generateCorrelationId = (): string => {
  return `api-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

// Default timeout - no magic defaults, but using process.env if available
const DEFAULT_TIMEOUT = typeof process !== 'undefined' && process.env?.DEFAULT_REQUEST_TIMEOUT
  ? parseInt(process.env.DEFAULT_REQUEST_TIMEOUT, 10)
  : 30000;

// Create circuit breaker for OpenEvolve API calls
const openevolveCircuitBreaker = new CircuitBreaker({
  threshold: 5,           // Trip after 5 consecutive failures
  timeout_ms: 60000,      // Stay open for 1 minute
  reset_timeout_ms: 10000, // Test recovery after 10 seconds
  onStateChange: (oldState, newState) => {
    apiLogger.warn('Circuit breaker state changed', {
      old_state: oldState,
      new_state: newState,
      target_service: 'openevolve-api'
    });
  }
});

const resolveBaseUrl = (override?: string): string => {
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
  } catch (error) {
    apiLogger.warn('Failed to access localStorage for api_base', {
      error: error instanceof Error ? error.message : String(error)
    });
  }

  // Law of Configuration Explicitness: No magic defaults
  // If no baseUrl is found, this will fail loudly
  throw new Error(
    'OpenEvolve API base URL not configured. ' +
    'Set OPENEVOLVE_API_BASE environment variable or provide via config.'
  );
};

const resolveApiKey = (override?: string): string => {
  if (override) {
    return override;
  }
  try {
    const stored = globalThis?.localStorage?.getItem("openevolve_api_key");
    if (stored) {
      return stored;
    }
  } catch (error) {
    apiLogger.warn('Failed to access localStorage for api_key', {
      error: error instanceof Error ? error.message : String(error)
    });
  }

  // Law of Configuration Explicitness: No magic defaults
  throw new Error(
    'OpenEvolve API key not configured. ' +
    'Set OPENEVOLVE_API_KEY environment variable or provide via config.'
  );
};

const buildHeaders = (apiKey?: string, correlationId?: string): Record<string, string> => {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
  // Add correlation ID header for distributed tracing
  if (correlationId) {
    headers["X-Correlation-ID"] = correlationId;
  }
  return headers;
};

async function requestFormData<T>(
  path: string,
  formData: FormData,
  config: ApiConfig = {},
): Promise<T> {
  const baseUrl = resolveBaseUrl(config.baseUrl);
  const apiKey = resolveApiKey(config.apiKey);
  const headers: Record<string, string> = {};
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
  const response = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers,
    body: formData,
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

async function request<T>(
  path: string,
  options: RequestInit = {},
  config: ApiConfig = {},
): Promise<T> {
  const correlationId = generateCorrelationId();
  const context: LogContext = {
    correlation_id: correlationId,
    source_service: 'frontend',
    target_service: 'openevolve-api',
    path
  };

  const startTime = Date.now();

  // Retry configuration - Law of Configuration Explicitness
  const retryConfig: RetryConfig = {
    max_retries: typeof process !== 'undefined' && process.env?.MAX_RETRIES
      ? parseInt(process.env.MAX_RETRIES, 10)
      : 3
  };

  // Wrap with circuit breaker and retry logic
  return openevolveCircuitBreaker.execute(async () => {
    return retryWithBackoff(async () => {
      try {
        const baseUrl = resolveBaseUrl(config.baseUrl);
        const apiKey = resolveApiKey(config.apiKey);
        const timeout = config.timeout || DEFAULT_TIMEOUT;

        const url = `${baseUrl}${path}`;

        apiLogger.info('API request initiated', {
          ...context,
          method: options.method || 'GET',
          timeout
        });

        // Create abort controller for timeout - MANDATORY per Law 3.2
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        try {
          const response = await fetch(url, {
            ...options,
            headers: {
              ...buildHeaders(apiKey, correlationId),
              ...(options.headers || {}),
            },
            signal: controller.signal,
          });

          clearTimeout(timeoutId);

          const duration = Date.now() - startTime;

          if (!response.ok) {
            const text = await response.text();
            apiLogger.error('API request failed', new Error(text), {
              ...context,
              status: response.status,
              status_text: response.statusText,
              duration_ms: duration
            });
            throw new Error(text || `Request failed with status ${response.status}`);
          }

          apiLogger.info('API request successful', {
            ...context,
            status: response.status,
            duration_ms: duration
          });

          return response.json() as Promise<T>;
        } catch (fetchError) {
          clearTimeout(timeoutId);

          if (fetchError instanceof Error && fetchError.name === 'AbortError') {
            apiLogger.error('API request timeout', new Error(`Request exceeded ${timeout}ms`), context);
            throw new Error(`Request timeout after ${timeout}ms`);
          }

          throw fetchError;
        }
      } catch (error) {
        const duration = Date.now() - startTime;
        apiLogger.error('API request error', error as Error, {
          ...context,
          duration_ms: duration,
          error_type: error instanceof Error ? error.constructor.name : 'Unknown'
        });
        throw error;
      }
    }, retryConfig);
  });
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

  // Version control
  listVersions: (config?: ApiConfig) =>
    request<{ versions: VersionEntry[]; current_version_id?: string | null }>(
      "/version-control/versions",
      {},
      config,
    ),
  getVersion: (versionId: string, config?: ApiConfig) =>
    request<VersionEntry>(`/version-control/versions/${encodeURIComponent(versionId)}`, {}, config),
  getCurrentVersion: (config?: ApiConfig) =>
    request<{ current: VersionEntry | null }>("/version-control/current", {}, config),
  createVersion: (
    payload: { protocol_text: string; version_name?: string; comment?: string; author?: string },
    config?: ApiConfig,
  ) =>
    request<{ version_id: string; version: VersionEntry }>(
      "/version-control/versions",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  loadVersion: (versionId: string, config?: ApiConfig) =>
    request<{ loaded: boolean; current: VersionEntry | null }>(
      `/version-control/versions/${encodeURIComponent(versionId)}/load`,
      { method: "POST" },
      config,
    ),
  branchVersion: (versionId: string, payload: { new_version_name: string }, config?: ApiConfig) =>
    request<{ version_id: string; version: VersionEntry }>(
      `/version-control/versions/${encodeURIComponent(versionId)}/branch`,
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  compareVersions: (payload: { version_id_1: string; version_id_2: string }, config?: ApiConfig) =>
    request<VersionCompareResult>(
      "/version-control/compare",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  deleteVersion: (versionId: string, config?: ApiConfig) =>
    request<{ deleted: boolean }>(
      `/version-control/versions/${encodeURIComponent(versionId)}`,
      { method: "DELETE" },
      config,
    ),

  // Validation manager
  listValidationRules: (config?: ApiConfig) =>
    request<{ rules: Record<string, ValidationRule>; rule_names: string[] }>(
      "/validation/rules",
      {},
      config,
    ),
  getValidationRule: (ruleName: string, config?: ApiConfig) =>
    request<{ name: string; rule: ValidationRule }>(
      `/validation/rules/${encodeURIComponent(ruleName)}`,
      {},
      config,
    ),
  createValidationRule: (
    payload: {
      name: string;
      max_length?: number | null;
      min_length?: number | null;
      required_keywords?: string[];
      forbidden_patterns?: string[];
      required_sections?: string[];
    },
    config?: ApiConfig,
  ) =>
    request<{ created: boolean; rule_name: string; rule: ValidationRule }>(
      "/validation/rules",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  updateValidationRule: (
    ruleName: string,
    payload: {
      name?: string;
      max_length?: number | null;
      min_length?: number | null;
      required_keywords?: string[] | null;
      forbidden_patterns?: string[] | null;
      required_sections?: string[] | null;
    },
    config?: ApiConfig,
  ) =>
    request<{ updated: boolean; rule_name: string; rule: ValidationRule }>(
      `/validation/rules/${encodeURIComponent(ruleName)}`,
      { method: "PUT", body: JSON.stringify(payload) },
      config,
    ),
  deleteValidationRule: (ruleName: string, config?: ApiConfig) =>
    request<{ deleted: boolean; rule_name: string }>(
      `/validation/rules/${encodeURIComponent(ruleName)}`,
      { method: "DELETE" },
      config,
    ),
  runValidation: (payload: { content: string; rule_names: string[] }, config?: ApiConfig) =>
    request<ValidationRunResult>(
      "/validation/run",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  runComplianceCheck: (payload: { content: string; framework?: string }, config?: ApiConfig) =>
    request<ComplianceCheckResult>(
      "/validation/compliance",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),

  // BubbleLabs workflow lifecycle
  listWorkflowDefinitions: (config?: ApiConfig) =>
    request<{ definitions: WorkflowDefinitionSummary[] }>(
      "/bubblelabs/workflow-definitions",
      {},
      config,
    ),
  getWorkflowDefinition: (definitionId: string, config?: ApiConfig) =>
    request<WorkflowDefinitionDetail>(
      `/bubblelabs/workflow-definitions/${encodeURIComponent(definitionId)}`,
      {},
      config,
    ),
  createWorkflowDefinition: (
    payload: {
      name: string;
      description: string;
      workflow_type: string;
      parameters: Record<string, unknown>;
    },
    config?: ApiConfig,
  ) =>
    request<{ definition_id: string }>(
      "/bubblelabs/workflow-definitions",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  listWorkflowInstances: (config?: ApiConfig) =>
    request<{ instances: WorkflowInstanceSummary[] }>(
      "/bubblelabs/workflow-instances",
      {},
      config,
    ),
  getWorkflowInstance: (instanceId: string, config?: ApiConfig) =>
    request<WorkflowInstanceDetail>(
      `/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}`,
      {},
      config,
    ),
  createWorkflowInstance: (
    payload: { definition_id: string; instance_name: string; inputs: Record<string, unknown> },
    config?: ApiConfig,
  ) =>
    request<{ instance_id: string }>(
      "/bubblelabs/workflow-instances",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  startWorkflowInstance: (instanceId: string, config?: ApiConfig) =>
    request<Record<string, unknown>>(
      `/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/start`,
      { method: "POST" },
      config,
    ),
  pauseWorkflowInstance: (instanceId: string, config?: ApiConfig) =>
    request<Record<string, unknown>>(
      `/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/pause`,
      { method: "POST" },
      config,
    ),
  resumeWorkflowInstance: (instanceId: string, config?: ApiConfig) =>
    request<Record<string, unknown>>(
      `/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/resume`,
      { method: "POST" },
      config,
    ),
  stopWorkflowInstance: (instanceId: string, config?: ApiConfig) =>
    request<Record<string, unknown>>(
      `/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/stop`,
      { method: "POST" },
      config,
    ),
  cancelWorkflowInstance: (instanceId: string, config?: ApiConfig) =>
    request<Record<string, unknown>>(
      `/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/cancel`,
      { method: "POST" },
      config,
    ),
  restartWorkflowInstance: (instanceId: string, config?: ApiConfig) =>
    request<Record<string, unknown>>(
      `/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/restart`,
      { method: "POST" },
      config,
    ),
  deleteWorkflowInstance: (instanceId: string, config?: ApiConfig) =>
    request<Record<string, unknown>>(
      `/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}`,
      { method: "DELETE" },
      config,
    ),

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

  // Evaluators
  listEvaluators: (config?: ApiConfig) => request<EvaluatorListResponse>("/evaluators", {}, config),
  uploadEvaluator: (payload: { code: string }, config?: ApiConfig) =>
    request<EvaluatorUploadResponse>(
      "/evaluators",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  deleteEvaluator: (evaluatorId: string, config?: ApiConfig) =>
    request<{ success: boolean; evaluator_id: string }>(
      `/evaluators/${encodeURIComponent(evaluatorId)}`,
      { method: "DELETE" },
      config,
    ),

  // Decomposition plan updates
  updateWorkflowPlan: (workflowId: string, payload: WorkflowPlanUpdateRequest, config?: ApiConfig) =>
    request<{ message: string; execution_order: string[] }>(
      `/workflows/${encodeURIComponent(workflowId)}/decomposition-plan`,
      { method: "PUT", body: JSON.stringify(payload) },
      config,
    ),
  getWorkflowResourceUsage: (workflowId: string, config?: ApiConfig) =>
    request<WorkflowResourceUsageResponse>(
      `/workflows/${encodeURIComponent(workflowId)}/resource-usage`,
      {},
      config,
    ),
  optimizeWorkflowResources: (workflowId: string, config?: ApiConfig) =>
    request<WorkflowResourceOptimizationResponse>(
      `/workflows/${encodeURIComponent(workflowId)}/resource-optimization`,
      { method: "POST" },
      config,
    ),

  // Integrated workflow
  runIntegratedWorkflow: (payload: IntegratedWorkflowRequest, config?: ApiConfig) =>
    request<Record<string, unknown>>(
      "/integrated/run",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),

  // Model orchestration
  listOrchestrationModels: (config?: ApiConfig) =>
    request<ModelOrchestrationListResponse>("/orchestration/models", {}, config),
  registerOrchestrationModel: (payload: ModelOrchestrationRegisterRequest, config?: ApiConfig) =>
    request<ModelOrchestrationRegisterResponse>(
      "/orchestration/models",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  executeOrchestrationEnsemble: (payload: ModelOrchestrationEnsembleRequest, config?: ApiConfig) =>
    request<ModelOrchestrationEnsembleResponse>(
      "/orchestration/ensemble",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),

  // BubbleLabs integration
  getBubblelabsStatus: (config?: ApiConfig) =>
    request<BubbleLabsStatusResponse>("/bubblelabs/status", {}, config),
  initializeBubblelabs: (config?: ApiConfig) =>
    request<BubbleLabsInitializeResponse>("/bubblelabs/initialize", { method: "POST" }, config),
  bubblelabsAceSkillbook: (payload: { name: string; skills: Array<Record<string, unknown>> }, config?: ApiConfig) =>
    request<BubbleLabsActionResponse>(
      "/bubblelabs/ace/skillbook",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  bubblelabsAcePatterns: (payload: { workflow_results: Array<Record<string, unknown>> }, config?: ApiConfig) =>
    request<BubbleLabsActionResponse>(
      "/bubblelabs/ace/patterns",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  bubblelabsZ3Solve: (payload: { variables: Array<Record<string, unknown>>; constraints: string[] }, config?: ApiConfig) =>
    request<BubbleLabsActionResponse>(
      "/bubblelabs/z3/solve",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  bubblelabsZ3Prove: (payload: { theorem: string }, config?: ApiConfig) =>
    request<BubbleLabsActionResponse>(
      "/bubblelabs/z3/prove",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  bubblelabsRomaAnalyze: (payload: { problem: string; max_depth?: number }, config?: ApiConfig) =>
    request<BubbleLabsActionResponse>(
      "/bubblelabs/roma/analyze",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  bubblelabsRomaConfig: (payload: { config: Record<string, unknown> }, config?: ApiConfig) =>
    request<BubbleLabsActionResponse>(
      "/bubblelabs/roma/config",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  bubblelabsKnowledgeStore: (payload: { artifact: Record<string, unknown> }, config?: ApiConfig) =>
    request<BubbleLabsActionResponse>(
      "/bubblelabs/knowledge/store",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  bubblelabsKnowledgeQuery: (payload: { query: string }, config?: ApiConfig) =>
    request<BubbleLabsActionResponse>(
      "/bubblelabs/knowledge/query",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  bubblelabsAnalyticsTrack: (payload: { workflow_id: string; metrics: Record<string, unknown> }, config?: ApiConfig) =>
    request<BubbleLabsActionResponse>(
      "/bubblelabs/analytics/track",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  bubblelabsAnalyticsDashboard: (config?: ApiConfig) =>
    request<BubbleLabsActionResponse>("/bubblelabs/analytics/dashboard", {}, config),
  bubblelabsLeanAideProve: (payload: { theorem: string }, config?: ApiConfig) =>
    request<BubbleLabsActionResponse>(
      "/bubblelabs/leanaide/prove",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),

  // Maker integration
  getMakerStatus: (config?: ApiConfig) => request<{ available: boolean }>("/maker/status", {}, config),
  listMakerTools: (params?: { status?: string; maker_mode?: string; search?: string }, config?: ApiConfig) => {
    const search = new URLSearchParams();
    if (params?.status) search.set("status", params.status);
    if (params?.maker_mode) search.set("maker_mode", params.maker_mode);
    if (params?.search) search.set("search", params.search);
    const suffix = search.toString() ? `?${search.toString()}` : "";
    return request<MakerToolListResponse>(`/maker/tools${suffix}`, {}, config);
  },
  getMakerTool: (toolId: string, config?: ApiConfig) =>
    request<MakerToolResponse>(`/maker/tools/${encodeURIComponent(toolId)}`, {}, config),
  createMakerTool: (
    payload: {
      name: string;
      description: string;
      task: string;
      maker_mode?: string;
      k_ahead?: number;
      max_depth?: number;
      context?: Record<string, unknown>;
      prompt_template?: string;
      system_prompt?: string;
      expected_schema?: Record<string, unknown>;
      metadata?: Record<string, unknown>;
    },
    config?: ApiConfig,
  ) =>
    request<MakerToolResponse>(
      "/maker/tools",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  testMakerTool: (
    toolId: string,
    payload: { input_data: Record<string, unknown>; delegate_to_crewai?: boolean },
    config?: ApiConfig,
  ) =>
    request<MakerExecutionResponse>(
      `/maker/tools/${encodeURIComponent(toolId)}/test`,
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  validateMakerTool: (toolId: string, config?: ApiConfig) =>
    request<{ status: string }>(
      `/maker/tools/${encodeURIComponent(toolId)}/validate`,
      { method: "POST" },
      config,
    ),
  executeMakerTool: (
    toolId: string,
    payload: { input_data: Record<string, unknown>; delegate_to_crewai?: boolean },
    config?: ApiConfig,
  ) =>
    request<MakerExecutionResponse>(
      `/maker/tools/${encodeURIComponent(toolId)}/execute`,
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  listMakerDelegations: (params?: { status?: string; delegation_type?: string }, config?: ApiConfig) => {
    const search = new URLSearchParams();
    if (params?.status) search.set("status", params.status);
    if (params?.delegation_type) search.set("delegation_type", params.delegation_type);
    const suffix = search.toString() ? `?${search.toString()}` : "";
    return request<MakerDelegationListResponse>(`/maker/delegations${suffix}`, {}, config);
  },
  syncMakerDelegations: (config?: ApiConfig) =>
    request<{ synced: number }>("/maker/delegations/sync", { method: "POST" }, config),

  // Knowledge Explorer
  bubblelabsKnowledgeStatus: (config?: ApiConfig) =>
    request<{ initialized: boolean; query_history_count: number }>(
      "/bubblelabs/knowledge/status",
      {},
      config,
    ),
  bubblelabsKnowledgeQueryAdvanced: (
    payload: { query: string; sources?: string[]; bedrock_kb_id?: string; index_path?: string },
    config?: ApiConfig,
  ) =>
    request<KnowledgeExplorerQueryResponse>(
      "/bubblelabs/knowledge/query-advanced",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  bubblelabsKnowledgeQueryHistory: (config?: ApiConfig) =>
    request<KnowledgeExplorerHistoryResponse>("/bubblelabs/knowledge/query-history", {}, config),
  bubblelabsKnowledgeExtract: (
    payload: { source_type: string; source_value: string; extraction_config?: Record<string, unknown> },
    config?: ApiConfig,
  ) =>
    request<KnowledgeExplorerExtractResponse>(
      "/bubblelabs/knowledge/extract",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  bubblelabsKnowledgeExtractFile: (
    file: File,
    extractionConfig?: Record<string, unknown>,
    config?: ApiConfig,
  ) => {
    const formData = new FormData();
    formData.append("file", file);
    if (extractionConfig) {
      formData.append("extraction_config", JSON.stringify(extractionConfig));
    }
    return requestFormData<KnowledgeExplorerExtractResponse>(
      "/bubblelabs/knowledge/extract-file",
      formData,
      config,
    );
  },

  // LeanAide
  bubblelabsLeanAideStatus: (config?: ApiConfig) =>
    request<LeanAideStatusResponse>("/bubblelabs/leanaide/status", {}, config),
  bubblelabsLeanAideExecute: (
    payload: { task_type: string; payload: Record<string, unknown> },
    config?: ApiConfig,
  ) =>
    request<LeanAideExecuteResponse>(
      "/bubblelabs/leanaide/execute",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  bubblelabsLeanAideTrees: (config?: ApiConfig) =>
    request<LeanAideTreeListResponse>("/bubblelabs/leanaide/trees", {}, config),
  bubblelabsLeanAideTree: (treeId: string, config?: ApiConfig) =>
    request<LeanAideTreeResponse>(`/bubblelabs/leanaide/trees/${encodeURIComponent(treeId)}`, {}, config),
  bubblelabsLeanAideProofs: (config?: ApiConfig) =>
    request<LeanAideProofListResponse>("/bubblelabs/leanaide/proofs", {}, config),
  bubblelabsLeanAideProof: (proofId: string, config?: ApiConfig) =>
    request<LeanAideProofResponse>(`/bubblelabs/leanaide/proofs/${encodeURIComponent(proofId)}`, {}, config),

  // Evolution and adversarial runs
  startEvolutionRun: (payload: {
    content: string;
    content_type?: string;
    evolution_mode?: string;
    parameters?: Record<string, unknown>;
    gauntlet_name?: string;
    use_decomposition?: boolean;
  }, config?: ApiConfig) =>
    request<EvolutionRunResponse>(
      "/evolution/runs",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  listEvolutionRuns: (config?: ApiConfig) =>
    request<EvolutionRunListResponse>("/evolution/runs", {}, config),
  getEvolutionRun: (runId: string, config?: ApiConfig) =>
    request<EvolutionRunStatus>(`/evolution/runs/${encodeURIComponent(runId)}`, {}, config),
  stopEvolutionRun: (runId: string, config?: ApiConfig) =>
    request<{ status: string }>(`/evolution/runs/${encodeURIComponent(runId)}/stop`, { method: "POST" }, config),

  startAdversarialRun: (payload: {
    content: string;
    content_type?: string;
    parameters?: Record<string, unknown>;
    use_decomposition?: boolean;
  }, config?: ApiConfig) =>
    request<AdversarialRunResponse>(
      "/adversarial/runs",
      { method: "POST", body: JSON.stringify(payload) },
      config,
    ),
  listAdversarialRuns: (config?: ApiConfig) =>
    request<AdversarialRunListResponse>("/adversarial/runs", {}, config),
  getAdversarialRun: (runId: string, config?: ApiConfig) =>
    request<AdversarialRunStatus>(`/adversarial/runs/${encodeURIComponent(runId)}`, {}, config),
  stopAdversarialRun: (runId: string, config?: ApiConfig) =>
    request<{ status: string }>(`/adversarial/runs/${encodeURIComponent(runId)}/stop`, { method: "POST" }, config),
};
