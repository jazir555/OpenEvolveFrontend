"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.OpenEvolveClient = exports.openevolveApi = void 0;
const structuredLogger_1 = require("../../../../lib/structuredLogger");
const retry_1 = require("../../../../lib/retry");
const circuit_breaker_1 = require("../../../../lib/circuit-breaker");
// Correlation ID generator for request tracking
const generateCorrelationId = () => {
    return `api-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};
// Default timeout - no magic defaults, but using process.env if available
const DEFAULT_TIMEOUT = typeof process !== 'undefined' && process.env?.DEFAULT_REQUEST_TIMEOUT
    ? parseInt(process.env.DEFAULT_REQUEST_TIMEOUT, 10)
    : 30000;
// Create circuit breaker for OpenEvolve API calls
const openevolveCircuitBreaker = new circuit_breaker_1.CircuitBreaker({
    threshold: 5, // Trip after 5 consecutive failures
    timeout_ms: 60000, // Stay open for 1 minute
    reset_timeout_ms: 10000, // Test recovery after 10 seconds
    onStateChange: (oldState, newState) => {
        structuredLogger_1.apiLogger.warn('Circuit breaker state changed', {
            old_state: oldState,
            new_state: newState,
            target_service: 'openevolve-api'
        });
    }
});
const resolveBaseUrl = (override) => {
    if (override) {
        return override;
    }
    const fromWindow = globalThis?.OPENEVOLVE_API_BASE;
    if (fromWindow) {
        return fromWindow;
    }
    try {
        const stored = globalThis?.localStorage?.getItem("openevolve_api_base");
        if (stored) {
            return stored;
        }
    }
    catch (error) {
        structuredLogger_1.apiLogger.warn('Failed to access localStorage for api_base', {
            error: error instanceof Error ? error.message : String(error)
        });
    }
    // Law of Configuration Explicitness: No magic defaults
    // If no baseUrl is found, this will fail loudly
    throw new Error('OpenEvolve API base URL not configured. ' +
        'Set OPENEVOLVE_API_BASE environment variable or provide via config.');
};
const resolveApiKey = (override) => {
    if (override) {
        return override;
    }
    try {
        const stored = globalThis?.localStorage?.getItem("openevolve_api_key");
        if (stored) {
            return stored;
        }
    }
    catch (error) {
        structuredLogger_1.apiLogger.warn('Failed to access localStorage for api_key', {
            error: error instanceof Error ? error.message : String(error)
        });
    }
    // Law of Configuration Explicitness: No magic defaults
    throw new Error('OpenEvolve API key not configured. ' +
        'Set OPENEVOLVE_API_KEY environment variable or provide via config.');
};
const buildHeaders = (apiKey, correlationId) => {
    const headers = {
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
async function requestFormData(path, formData, config = {}) {
    const baseUrl = resolveBaseUrl(config.baseUrl);
    const apiKey = resolveApiKey(config.apiKey);
    const headers = {};
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
    return (await response.json());
}
async function request(path, options = {}, config = {}) {
    const correlationId = generateCorrelationId();
    const context = {
        correlation_id: correlationId,
        source_service: 'frontend',
        target_service: 'openevolve-api',
        path
    };
    const startTime = Date.now();
    // Retry configuration - Law of Configuration Explicitness
    const retryConfig = {
        max_retries: typeof process !== 'undefined' && process.env?.MAX_RETRIES
            ? parseInt(process.env.MAX_RETRIES, 10)
            : 3
    };
    // Wrap with circuit breaker and retry logic
    return openevolveCircuitBreaker.execute(async () => {
        return (0, retry_1.retryWithBackoff)(async () => {
            try {
                const baseUrl = resolveBaseUrl(config.baseUrl);
                const apiKey = resolveApiKey(config.apiKey);
                const timeout = config.timeout || DEFAULT_TIMEOUT;
                const url = `${baseUrl}${path}`;
                structuredLogger_1.apiLogger.info('API request initiated', {
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
                        structuredLogger_1.apiLogger.error('API request failed', new Error(text), {
                            ...context,
                            status: response.status,
                            status_text: response.statusText,
                            duration_ms: duration
                        });
                        throw new Error(text || `Request failed with status ${response.status}`);
                    }
                    structuredLogger_1.apiLogger.info('API request successful', {
                        ...context,
                        status: response.status,
                        duration_ms: duration
                    });
                    return response.json();
                }
                catch (fetchError) {
                    clearTimeout(timeoutId);
                    if (fetchError instanceof Error && fetchError.name === 'AbortError') {
                        structuredLogger_1.apiLogger.error('API request timeout', new Error(`Request exceeded ${timeout}ms`), context);
                        throw new Error(`Request timeout after ${timeout}ms`);
                    }
                    throw fetchError;
                }
            }
            catch (error) {
                const duration = Date.now() - startTime;
                structuredLogger_1.apiLogger.error('API request error', error, {
                    ...context,
                    duration_ms: duration,
                    error_type: error instanceof Error ? error.constructor.name : 'Unknown'
                });
                throw error;
            }
        }, retryConfig);
    });
}
exports.openevolveApi = {
    listTeams: (config) => request("/teams", {}, config),
    getTeam: (teamName, config) => request(`/teams/${encodeURIComponent(teamName)}`, {}, config),
    createTeam: (team, config) => request("/teams", { method: "POST", body: JSON.stringify(team) }, config),
    updateTeam: (teamName, team, config) => request(`/teams/${encodeURIComponent(teamName)}`, { method: "PUT", body: JSON.stringify(team) }, config),
    deleteTeam: (teamName, config) => request(`/teams/${encodeURIComponent(teamName)}`, { method: "DELETE" }, config),
    listGauntlets: (config) => request("/gauntlets", {}, config),
    getGauntlet: (name, config) => request(`/gauntlets/${encodeURIComponent(name)}`, {}, config),
    createGauntlet: (gauntlet, config) => request("/gauntlets", { method: "POST", body: JSON.stringify(gauntlet) }, config),
    updateGauntlet: (name, gauntlet, config) => request(`/gauntlets/${encodeURIComponent(name)}`, { method: "PUT", body: JSON.stringify(gauntlet) }, config),
    deleteGauntlet: (name, config) => request(`/gauntlets/${encodeURIComponent(name)}`, { method: "DELETE" }, config),
    listWorkflows: (config) => request("/workflows", {}, config),
    getWorkflow: (workflowId, config) => request(`/workflows/${encodeURIComponent(workflowId)}`, {}, config),
    createWorkflow: (payload, config) => request("/workflows", { method: "POST", body: JSON.stringify(payload) }, config),
    pauseWorkflow: (workflowId, config) => request(`/workflows/${encodeURIComponent(workflowId)}/pause`, { method: "POST" }, config),
    resumeWorkflow: (workflowId, config) => request(`/workflows/${encodeURIComponent(workflowId)}/resume`, { method: "POST" }, config),
    deleteWorkflow: (workflowId, config) => request(`/workflows/${encodeURIComponent(workflowId)}`, { method: "DELETE" }, config),
    getWorkflowResults: (workflowId, config) => request(`/workflows/${encodeURIComponent(workflowId)}/results`, {}, config),
    getStatistics: (config) => request("/statistics", {}, config),
    getPerformanceMetrics: (entityType, limit = 200, config) => {
        const params = new URLSearchParams();
        if (entityType) {
            params.set("entity_type", entityType);
        }
        if (limit) {
            params.set("limit", String(limit));
        }
        const suffix = params.toString() ? `?${params.toString()}` : "";
        return request(`/analytics/performance-metrics${suffix}`, {}, config);
    },
    getAnalyticsKnowledgeStats: (config) => request("/analytics/knowledge-stats", {}, config),
    getWorkflowPlan: (workflowId, config) => request(`/workflows/${encodeURIComponent(workflowId)}/decomposition-plan`, {}, config),
    getWorkflowTelemetry: (workflowId, config) => request(`/workflows/${encodeURIComponent(workflowId)}/telemetry`, {}, config),
    getWorkflowMetrics: (config) => request("/analytics/workflow-metrics", {}, config),
    listSovereignPlans: (config) => request("/sovereign/plans", {}, config),
    getMonitoringDashboard: (config) => request("/monitoring/dashboard", {}, config),
    getMonitoringAlerts: (config) => request("/monitoring/alerts", {}, config),
    getMonitoringServices: (config) => request("/monitoring/services", {}, config),
    getMonitoringLogs: (limit = 200, source, config) => {
        const params = new URLSearchParams();
        if (limit)
            params.set("limit", String(limit));
        if (source)
            params.set("source", source);
        const suffix = params.toString() ? `?${params.toString()}` : "";
        return request(`/monitoring/logs${suffix}`, {}, config);
    },
    getMonitoringMetrics: (params, config) => {
        const search = new URLSearchParams();
        if (params.name)
            search.set("name", params.name);
        if (params.start_time)
            search.set("start_time", params.start_time);
        if (params.end_time)
            search.set("end_time", params.end_time);
        const suffix = search.toString() ? `?${search.toString()}` : "";
        return request(`/monitoring/metrics${suffix}`, {}, config);
    },
    getMonitoringHealth: (config) => request("/monitoring/health", {}, config),
    listCrewaiWorkflows: (config) => request("/crewai/workflows", {}, config),
    getCrewaiWorkflow: (workflowId, config) => request(`/crewai/workflows/${encodeURIComponent(workflowId)}`, {}, config),
    getCrewaiWorkflowTickets: (workflowId, config) => request(`/crewai/workflows/${encodeURIComponent(workflowId)}/tickets`, {}, config),
    listPrompts: (config) => request("/prompts", {}, config),
    savePrompt: (payload, config) => request("/prompts", { method: "POST", body: JSON.stringify(payload) }, config),
    deletePrompt: (promptName, config) => request(`/prompts/${encodeURIComponent(promptName)}`, { method: "DELETE" }, config),
    listContentTemplates: (config) => request("/content/templates", {}, config),
    getContentTemplate: (templateName, config) => request(`/content/templates/${encodeURIComponent(templateName)}`, {}, config),
    createContentTemplate: (payload, config) => request("/content/templates", { method: "POST", body: JSON.stringify(payload) }, config),
    validateProtocol: (payload, config) => request("/content/validate", { method: "POST", body: JSON.stringify(payload) }, config),
    listAuditLogs: (limit = 200, config) => request(`/audit/logs?limit=${limit}`, {}, config),
    getIcrOverview: (config) => request("/icr/analytics/overview", {}, config),
    getIcrComponents: (config) => request("/icr/analytics/components", {}, config),
    getIcrRefinements: (config) => request("/icr/analytics/refinements", {}, config),
    getAdaptiveMdapHealth: (config) => request("/adaptive-mdap/health", {}, config),
    getAdaptiveMdapDashboard: (config) => request("/adaptive-mdap/dashboard", {}, config),
    getAdaptiveMdapProfiles: (config) => request("/adaptive-mdap/profiles", {}, config),
    getAdaptiveMdapProfileConfig: (profileName, config) => request(`/adaptive-mdap/profiles/${encodeURIComponent(profileName)}`, {}, config),
    calculateAdaptiveMdapCost: (payload, config) => request("/adaptive-mdap/cost", { method: "POST", body: JSON.stringify(payload) }, config),
    classifyAdaptiveMdapComplexity: (payload, config) => request("/adaptive-mdap/complexity", { method: "POST", body: JSON.stringify(payload) }, config),
    allocateAdaptiveMdapResources: (payload, config) => request("/adaptive-mdap/allocate", { method: "POST", body: JSON.stringify(payload) }, config),
    getHealth: (config) => request("/health", {}, config),
    // Knowledge Base
    listKnowledgeArtifacts: (config) => request("/knowledge/artifacts", {}, config),
    getKnowledgeArtifact: (artifactId, config) => request(`/knowledge/artifacts/${encodeURIComponent(artifactId)}`, {}, config),
    createKnowledgeArtifact: (payload, config) => request("/knowledge/artifacts", { method: "POST", body: JSON.stringify(payload) }, config),
    deleteKnowledgeArtifact: (artifactId, config) => request(`/knowledge/artifacts/${encodeURIComponent(artifactId)}`, { method: "DELETE" }, config),
    searchKnowledge: (payload, config) => request("/knowledge/search", { method: "POST", body: JSON.stringify(payload) }, config),
    getKnowledgeGraph: (config) => request("/knowledge/graph", {}, config),
    getKnowledgeStats: (config) => request("/knowledge/stats", {}, config),
    getKnowledgeRecommendations: (payload, config) => request("/knowledge/recommendations", { method: "POST", body: JSON.stringify(payload) }, config),
    exportKnowledgeBase: (config) => request("/knowledge/export", {}, config),
    importKnowledgeBase: (payload, config) => request("/knowledge/import", { method: "POST", body: JSON.stringify(payload) }, config),
    // Auto-Approval
    getAutoApprovalConfig: (config) => request("/auto-approval/config", {}, config),
    updateAutoApprovalConfig: (payload, config) => request("/auto-approval/config", { method: "PUT", body: JSON.stringify(payload) }, config),
    testAutoApproval: (payload, config) => request("/auto-approval/test", { method: "POST", body: JSON.stringify(payload) }, config),
    getAutoApprovalAudit: (config) => request("/auto-approval/audit", {}, config),
    // Workflow Templates
    listWorkflowTemplates: (config) => request("/workflow-templates", {}, config),
    createWorkflowTemplate: (payload, config) => request("/workflow-templates", { method: "POST", body: JSON.stringify(payload) }, config),
    updateWorkflowTemplate: (templateId, payload, config) => request(`/workflow-templates/${encodeURIComponent(templateId)}`, { method: "PUT", body: JSON.stringify(payload) }, config),
    deleteWorkflowTemplate: (templateId, config) => request(`/workflow-templates/${encodeURIComponent(templateId)}`, { method: "DELETE" }, config),
    exportWorkflowTemplates: (config) => request("/workflow-templates/export", {}, config),
    importWorkflowTemplates: (payload, config) => request("/workflow-templates/import", { method: "POST", body: JSON.stringify(payload) }, config),
    // Providers and parameters
    listProviders: (config) => request("/providers", {}, config),
    getProviderModels: (providerId, apiKey, config) => request(`/providers/${encodeURIComponent(providerId)}/models`, {
        method: "POST",
        body: JSON.stringify({ api_key: apiKey }),
    }, config),
    getParameterSchema: (config) => request("/parameters/schema", {}, config),
    getParameterDefaults: (config) => request("/parameters/defaults", {}, config),
    validateParameters: (payload, config) => request("/parameters/validate", { method: "POST", body: JSON.stringify(payload) }, config),
    getParameterCategories: (config) => request("/parameters/categories", {}, config),
    // Version control
    listVersions: (config) => request("/version-control/versions", {}, config),
    getVersion: (versionId, config) => request(`/version-control/versions/${encodeURIComponent(versionId)}`, {}, config),
    getCurrentVersion: (config) => request("/version-control/current", {}, config),
    createVersion: (payload, config) => request("/version-control/versions", { method: "POST", body: JSON.stringify(payload) }, config),
    loadVersion: (versionId, config) => request(`/version-control/versions/${encodeURIComponent(versionId)}/load`, { method: "POST" }, config),
    branchVersion: (versionId, payload, config) => request(`/version-control/versions/${encodeURIComponent(versionId)}/branch`, { method: "POST", body: JSON.stringify(payload) }, config),
    compareVersions: (payload, config) => request("/version-control/compare", { method: "POST", body: JSON.stringify(payload) }, config),
    deleteVersion: (versionId, config) => request(`/version-control/versions/${encodeURIComponent(versionId)}`, { method: "DELETE" }, config),
    // Validation manager
    listValidationRules: (config) => request("/validation/rules", {}, config),
    getValidationRule: (ruleName, config) => request(`/validation/rules/${encodeURIComponent(ruleName)}`, {}, config),
    createValidationRule: (payload, config) => request("/validation/rules", { method: "POST", body: JSON.stringify(payload) }, config),
    updateValidationRule: (ruleName, payload, config) => request(`/validation/rules/${encodeURIComponent(ruleName)}`, { method: "PUT", body: JSON.stringify(payload) }, config),
    deleteValidationRule: (ruleName, config) => request(`/validation/rules/${encodeURIComponent(ruleName)}`, { method: "DELETE" }, config),
    runValidation: (payload, config) => request("/validation/run", { method: "POST", body: JSON.stringify(payload) }, config),
    runComplianceCheck: (payload, config) => request("/validation/compliance", { method: "POST", body: JSON.stringify(payload) }, config),
    // BubbleLabs workflow lifecycle
    listWorkflowDefinitions: (config) => request("/bubblelabs/workflow-definitions", {}, config),
    getWorkflowDefinition: (definitionId, config) => request(`/bubblelabs/workflow-definitions/${encodeURIComponent(definitionId)}`, {}, config),
    createWorkflowDefinition: (payload, config) => request("/bubblelabs/workflow-definitions", { method: "POST", body: JSON.stringify(payload) }, config),
    listWorkflowInstances: (config) => request("/bubblelabs/workflow-instances", {}, config),
    getWorkflowInstance: (instanceId, config) => request(`/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}`, {}, config),
    createWorkflowInstance: (payload, config) => request("/bubblelabs/workflow-instances", { method: "POST", body: JSON.stringify(payload) }, config),
    syncWorkflowInstanceParameters: (instanceId, payload, config) => request(`/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/parameters`, { method: "POST", body: JSON.stringify(payload) }, config),
    startWorkflowInstance: (instanceId, config) => request(`/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/start`, { method: "POST" }, config),
    pauseWorkflowInstance: (instanceId, config) => request(`/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/pause`, { method: "POST" }, config),
    resumeWorkflowInstance: (instanceId, config) => request(`/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/resume`, { method: "POST" }, config),
    stopWorkflowInstance: (instanceId, config) => request(`/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/stop`, { method: "POST" }, config),
    cancelWorkflowInstance: (instanceId, config) => request(`/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/cancel`, { method: "POST" }, config),
    restartWorkflowInstance: (instanceId, config) => request(`/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}/restart`, { method: "POST" }, config),
    deleteWorkflowInstance: (instanceId, config) => request(`/bubblelabs/workflow-instances/${encodeURIComponent(instanceId)}`, { method: "DELETE" }, config),
    // Sovereign dashboard
    getSovereignHealth: (config) => request("/sovereign/health", {}, config),
    listSovereignProblems: (config) => request("/sovereign/problems", {}, config),
    // Suggestions
    getContentSuggestions: (payload, config) => request("/suggestions/content", { method: "POST", body: JSON.stringify(payload) }, config),
    getContentClassification: (payload, config) => request("/suggestions/classification", { method: "POST", body: JSON.stringify(payload) }, config),
    getSecuritySuggestions: (payload, config) => request("/suggestions/security", { method: "POST", body: JSON.stringify(payload) }, config),
    getImprovementPotential: (payload, config) => request("/suggestions/improvement", { method: "POST", body: JSON.stringify(payload) }, config),
    // Evaluators
    listEvaluators: (config) => request("/evaluators", {}, config),
    uploadEvaluator: (payload, config) => request("/evaluators", { method: "POST", body: JSON.stringify(payload) }, config),
    deleteEvaluator: (evaluatorId, config) => request(`/evaluators/${encodeURIComponent(evaluatorId)}`, { method: "DELETE" }, config),
    // Decomposition plan updates
    updateWorkflowPlan: (workflowId, payload, config) => request(`/workflows/${encodeURIComponent(workflowId)}/decomposition-plan`, { method: "PUT", body: JSON.stringify(payload) }, config),
    getWorkflowResourceUsage: (workflowId, config) => request(`/workflows/${encodeURIComponent(workflowId)}/resource-usage`, {}, config),
    optimizeWorkflowResources: (workflowId, config) => request(`/workflows/${encodeURIComponent(workflowId)}/resource-optimization`, { method: "POST" }, config),
    // Integrated workflow
    runIntegratedWorkflow: (payload, config) => request("/integrated/run", { method: "POST", body: JSON.stringify(payload) }, config),
    // Model orchestration
    listOrchestrationModels: (config) => request("/orchestration/models", {}, config),
    registerOrchestrationModel: (payload, config) => request("/orchestration/models", { method: "POST", body: JSON.stringify(payload) }, config),
    executeOrchestrationEnsemble: (payload, config) => request("/orchestration/ensemble", { method: "POST", body: JSON.stringify(payload) }, config),
    // BubbleLabs integration
    getBubblelabsStatus: (config) => request("/bubblelabs/status", {}, config),
    initializeBubblelabs: (config) => request("/bubblelabs/initialize", { method: "POST" }, config),
    bubblelabsControlCatalog: (config) => request("/bubblelabs/control/catalog", {}, config),
    bubblelabsControlDiscover: (payload = {}, config) => request("/bubblelabs/control/discover", { method: "POST", body: JSON.stringify(payload) }, config),
    bubblelabsControlExecute: (payload, config) => request("/bubblelabs/control/execute", { method: "POST", body: JSON.stringify(payload) }, config),
    bubblelabsAceSkillbook: (payload, config) => request("/bubblelabs/ace/skillbook", { method: "POST", body: JSON.stringify(payload) }, config),
    bubblelabsAcePatterns: (payload, config) => request("/bubblelabs/ace/patterns", { method: "POST", body: JSON.stringify(payload) }, config),
    bubblelabsZ3Solve: (payload, config) => request("/bubblelabs/z3/solve", { method: "POST", body: JSON.stringify(payload) }, config),
    bubblelabsZ3Prove: (payload, config) => request("/bubblelabs/z3/prove", { method: "POST", body: JSON.stringify(payload) }, config),
    bubblelabsRomaAnalyze: (payload, config) => request("/bubblelabs/roma/analyze", { method: "POST", body: JSON.stringify(payload) }, config),
    bubblelabsRomaConfig: (payload, config) => request("/bubblelabs/roma/config", { method: "POST", body: JSON.stringify(payload) }, config),
    bubblelabsKnowledgeStore: (payload, config) => request("/bubblelabs/knowledge/store", { method: "POST", body: JSON.stringify(payload) }, config),
    bubblelabsKnowledgeQuery: (payload, config) => request("/bubblelabs/knowledge/query", { method: "POST", body: JSON.stringify(payload) }, config),
    bubblelabsAnalyticsTrack: (payload, config) => request("/bubblelabs/analytics/track", { method: "POST", body: JSON.stringify(payload) }, config),
    bubblelabsAnalyticsDashboard: (config) => request("/bubblelabs/analytics/dashboard", {}, config),
    bubblelabsLeanAideProve: (payload, config) => request("/bubblelabs/leanaide/prove", { method: "POST", body: JSON.stringify(payload) }, config),
    // Maker integration
    getMakerStatus: (config) => request("/maker/status", {}, config),
    listMakerTools: (params, config) => {
        const search = new URLSearchParams();
        if (params?.status)
            search.set("status", params.status);
        if (params?.maker_mode)
            search.set("maker_mode", params.maker_mode);
        if (params?.search)
            search.set("search", params.search);
        const suffix = search.toString() ? `?${search.toString()}` : "";
        return request(`/maker/tools${suffix}`, {}, config);
    },
    getMakerTool: (toolId, config) => request(`/maker/tools/${encodeURIComponent(toolId)}`, {}, config),
    createMakerTool: (payload, config) => request("/maker/tools", { method: "POST", body: JSON.stringify(payload) }, config),
    testMakerTool: (toolId, payload, config) => request(`/maker/tools/${encodeURIComponent(toolId)}/test`, { method: "POST", body: JSON.stringify(payload) }, config),
    validateMakerTool: (toolId, config) => request(`/maker/tools/${encodeURIComponent(toolId)}/validate`, { method: "POST" }, config),
    executeMakerTool: (toolId, payload, config) => request(`/maker/tools/${encodeURIComponent(toolId)}/execute`, { method: "POST", body: JSON.stringify(payload) }, config),
    listMakerDelegations: (params, config) => {
        const search = new URLSearchParams();
        if (params?.status)
            search.set("status", params.status);
        if (params?.delegation_type)
            search.set("delegation_type", params.delegation_type);
        const suffix = search.toString() ? `?${search.toString()}` : "";
        return request(`/maker/delegations${suffix}`, {}, config);
    },
    syncMakerDelegations: (config) => request("/maker/delegations/sync", { method: "POST" }, config),
    // Knowledge Explorer
    bubblelabsKnowledgeStatus: (config) => request("/bubblelabs/knowledge/status", {}, config),
    bubblelabsKnowledgeQueryAdvanced: (payload, config) => request("/bubblelabs/knowledge/query-advanced", { method: "POST", body: JSON.stringify(payload) }, config),
    bubblelabsKnowledgeQueryHistory: (config) => request("/bubblelabs/knowledge/query-history", {}, config),
    bubblelabsKnowledgeExtract: (payload, config) => request("/bubblelabs/knowledge/extract", { method: "POST", body: JSON.stringify(payload) }, config),
    bubblelabsKnowledgeExtractFile: (file, extractionConfig, config) => {
        const formData = new FormData();
        formData.append("file", file);
        if (extractionConfig) {
            formData.append("extraction_config", JSON.stringify(extractionConfig));
        }
        return requestFormData("/bubblelabs/knowledge/extract-file", formData, config);
    },
    // LeanAide
    bubblelabsLeanAideStatus: (config) => request("/bubblelabs/leanaide/status", {}, config),
    bubblelabsLeanAideExecute: (payload, config) => request("/bubblelabs/leanaide/execute", { method: "POST", body: JSON.stringify(payload) }, config),
    bubblelabsLeanAideTrees: (config) => request("/bubblelabs/leanaide/trees", {}, config),
    bubblelabsLeanAideTree: (treeId, config) => request(`/bubblelabs/leanaide/trees/${encodeURIComponent(treeId)}`, {}, config),
    bubblelabsLeanAideProofs: (config) => request("/bubblelabs/leanaide/proofs", {}, config),
    bubblelabsLeanAideProof: (proofId, config) => request(`/bubblelabs/leanaide/proofs/${encodeURIComponent(proofId)}`, {}, config),
    // Evolution and adversarial runs
    startEvolutionRun: (payload, config) => request("/evolution/runs", { method: "POST", body: JSON.stringify(payload) }, config),
    listEvolutionRuns: (config) => request("/evolution/runs", {}, config),
    getEvolutionRun: (runId, config) => request(`/evolution/runs/${encodeURIComponent(runId)}`, {}, config),
    stopEvolutionRun: (runId, config) => request(`/evolution/runs/${encodeURIComponent(runId)}/stop`, { method: "POST" }, config),
    startAdversarialRun: (payload, config) => request("/adversarial/runs", { method: "POST", body: JSON.stringify(payload) }, config),
    listAdversarialRuns: (config) => request("/adversarial/runs", {}, config),
    getAdversarialRun: (runId, config) => request(`/adversarial/runs/${encodeURIComponent(runId)}`, {}, config),
    stopAdversarialRun: (runId, config) => request(`/adversarial/runs/${encodeURIComponent(runId)}/stop`, { method: "POST" }, config),
    // Gauntlet execution endpoints
    executeGauntlet: (gauntletName, payload, config) => request(`/gauntlets/${encodeURIComponent(gauntletName)}/execute`, { method: "POST", body: JSON.stringify(payload) }, config),
    getGauntletExecutionStatus: (executionId, config) => request(`/gauntlets/executions/${encodeURIComponent(executionId)}/status`, {}, config),
    listGauntletExecutions: (gauntletName, config) => {
        const params = new URLSearchParams();
        if (gauntletName)
            params.set("gauntlet_name", gauntletName);
        const suffix = params.toString() ? `?${params.toString()}` : "";
        return request(`/gauntlets/executions${suffix}`, {}, config);
    },
    // Decomposition execution endpoints
    executeDecomposition: (workflowId, payload, config) => request(`/workflows/${encodeURIComponent(workflowId)}/execute-decomposition`, { method: "POST", body: JSON.stringify(payload) }, config),
    getDecompositionExecutionStatus: (executionId, config) => request(`/decomposition/executions/${encodeURIComponent(executionId)}/status`, {}, config),
    listDecompositionExecutions: (workflowId, config) => {
        const params = new URLSearchParams();
        if (workflowId)
            params.set("workflow_id", workflowId);
        const suffix = params.toString() ? `?${params.toString()}` : "";
        return request(`/decomposition/executions${suffix}`, {}, config);
    },
    // Workflow template execution endpoints
    executeWorkflowTemplate: (templateId, payload, config) => request(`/workflow-templates/${encodeURIComponent(templateId)}/execute`, { method: "POST", body: JSON.stringify(payload) }, config),
    getWorkflowTemplateExecutionStatus: (executionId, config) => request(`/workflow-templates/executions/${encodeURIComponent(executionId)}/status`, {}, config),
    stopWorkflowTemplateExecution: (executionId, config) => request(`/workflow-templates/executions/${encodeURIComponent(executionId)}/stop`, { method: "POST" }, config),
    // Unified execution status endpoint
    getExecutionStatus: (executionType, executionId, config) => {
        switch (executionType) {
            case 'gauntlet':
                return request(`/gauntlets/executions/${encodeURIComponent(executionId)}/status`, {}, config);
            case 'decomposition':
                return request(`/decomposition/executions/${encodeURIComponent(executionId)}/status`, {}, config);
            case 'workflow-template':
                return request(`/workflow-templates/executions/${encodeURIComponent(executionId)}/status`, {}, config);
            default:
                throw new Error(`Unknown execution type: ${executionType}`);
        }
    },
};
/**
 * Backward-compatible client wrapper for contract tests and legacy consumers.
 * New code should prefer direct `openevolveApi.*` function calls.
 */
class OpenEvolveClient {
    constructor(config = {}) {
        this.config = config;
    }
    health() {
        return exports.openevolveApi.getHealth(this.config);
    }
    listTeams() {
        return exports.openevolveApi.listTeams(this.config);
    }
    listWorkflows() {
        return exports.openevolveApi.listWorkflows(this.config);
    }
    listGauntlets() {
        return exports.openevolveApi.listGauntlets(this.config);
    }
    controlCatalog() {
        return exports.openevolveApi.bubblelabsControlCatalog(this.config);
    }
}
exports.OpenEvolveClient = OpenEvolveClient;
//# sourceMappingURL=openevolveApi.js.map