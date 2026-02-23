import type { Team, TeamSummary, GauntletDefinition, GauntletSummary, WorkflowSummary, WorkflowDetail, WorkflowCreateRequest, WorkflowCreateResponse, WorkflowResults } from "./types";
import type { AuditLogEntry, StatisticsSummary, AdaptiveMdapDashboard, AdaptiveMdapProfiles } from "./types";
import type { IcrOverview, IcrComponents, IcrRefinements } from "./types";
import type { KnowledgeArtifact, KnowledgeGraph, KnowledgeStats, KnowledgeRecommendations, PromptMap, ContentTemplate, ProtocolValidationResult, AutoApprovalConfig, AutoApprovalTestResult, AutoApprovalAuditEntry, WorkflowTemplate, ProviderSummary, ParameterDefinition, ParameterValidationResult } from "./types";
import type { PerformanceMetric, AnalyticsWorkflowMetric, AnalyticsKnowledgeStats, MonitoringDashboardMetrics, MonitoringAlert, MonitoringMetric, MonitoringService, MonitoringLogEntry, WorkflowTelemetry, CrewAIWorkflowSummary, CrewAIWorkflowTicket, WorkflowPlanResponse, SovereignPlan } from "./types";
import type { EvaluatorListResponse, EvaluatorUploadResponse, WorkflowPlanUpdateRequest, WorkflowResourceUsageResponse, WorkflowResourceOptimizationResponse, IntegratedWorkflowRequest, ModelOrchestrationListResponse, ModelOrchestrationRegisterRequest, ModelOrchestrationRegisterResponse, ModelOrchestrationEnsembleRequest, ModelOrchestrationEnsembleResponse, BubbleLabsStatusResponse, BubbleLabsInitializeResponse, BubbleLabsActionResponse, BubbleLabsControlCatalogResponse, BubbleLabsControlDiscoverResponse, BubbleLabsControlExecuteResponse, VersionEntry, VersionCompareResult, ValidationRule, ValidationRunResult, ComplianceCheckResult, WorkflowDefinitionSummary, WorkflowDefinitionDetail, WorkflowInstanceSummary, WorkflowInstanceDetail, MakerToolListResponse, MakerToolResponse, MakerExecutionResponse, MakerDelegationListResponse, KnowledgeExplorerQueryResponse, KnowledgeExplorerExtractResponse, KnowledgeExplorerHistoryResponse, LeanAideStatusResponse, LeanAideExecuteResponse, LeanAideTreeListResponse, LeanAideTreeResponse, LeanAideProofListResponse, LeanAideProofResponse, EvolutionRunResponse, EvolutionRunStatus, EvolutionRunListResponse, AdversarialRunResponse, AdversarialRunStatus, AdversarialRunListResponse } from "./types";
export interface ApiConfig {
    baseUrl?: string;
    apiKey?: string;
    timeout?: number;
}
export declare const openevolveApi: {
    listTeams: (config?: ApiConfig) => Promise<{
        teams: TeamSummary[];
        total: number;
    }>;
    getTeam: (teamName: string, config?: ApiConfig) => Promise<Team>;
    createTeam: (team: Team, config?: ApiConfig) => Promise<{
        message: string;
        team_name: string;
    }>;
    updateTeam: (teamName: string, team: Team, config?: ApiConfig) => Promise<{
        message: string;
        team_name: string;
    }>;
    deleteTeam: (teamName: string, config?: ApiConfig) => Promise<{
        success: boolean;
    }>;
    listGauntlets: (config?: ApiConfig) => Promise<{
        gauntlets: GauntletSummary[];
        total: number;
    }>;
    getGauntlet: (name: string, config?: ApiConfig) => Promise<GauntletDefinition>;
    createGauntlet: (gauntlet: GauntletDefinition, config?: ApiConfig) => Promise<{
        message: string;
        gauntlet_name: string;
    }>;
    updateGauntlet: (name: string, gauntlet: GauntletDefinition, config?: ApiConfig) => Promise<{
        message: string;
        gauntlet_name: string;
    }>;
    deleteGauntlet: (name: string, config?: ApiConfig) => Promise<{
        success: boolean;
    }>;
    listWorkflows: (config?: ApiConfig) => Promise<{
        workflows: WorkflowSummary[];
        total: number;
    }>;
    getWorkflow: (workflowId: string, config?: ApiConfig) => Promise<WorkflowDetail>;
    createWorkflow: (payload: WorkflowCreateRequest, config?: ApiConfig) => Promise<WorkflowCreateResponse>;
    pauseWorkflow: (workflowId: string, config?: ApiConfig) => Promise<{
        message: string;
        workflow_id: string;
        status: string;
    }>;
    resumeWorkflow: (workflowId: string, config?: ApiConfig) => Promise<{
        message: string;
        workflow_id: string;
        status: string;
    }>;
    deleteWorkflow: (workflowId: string, config?: ApiConfig) => Promise<{
        message: string;
        workflow_id: string;
    }>;
    getWorkflowResults: (workflowId: string, config?: ApiConfig) => Promise<WorkflowResults>;
    getStatistics: (config?: ApiConfig) => Promise<StatisticsSummary>;
    getPerformanceMetrics: (entityType?: string, limit?: number, config?: ApiConfig) => Promise<{
        metrics: PerformanceMetric[];
        total: number;
    }>;
    getAnalyticsKnowledgeStats: (config?: ApiConfig) => Promise<AnalyticsKnowledgeStats>;
    getWorkflowPlan: (workflowId: string, config?: ApiConfig) => Promise<WorkflowPlanResponse>;
    getWorkflowTelemetry: (workflowId: string, config?: ApiConfig) => Promise<WorkflowTelemetry>;
    getWorkflowMetrics: (config?: ApiConfig) => Promise<{
        metrics: AnalyticsWorkflowMetric[];
        total: number;
    }>;
    listSovereignPlans: (config?: ApiConfig) => Promise<{
        plans: SovereignPlan[];
    }>;
    getMonitoringDashboard: (config?: ApiConfig) => Promise<MonitoringDashboardMetrics>;
    getMonitoringAlerts: (config?: ApiConfig) => Promise<{
        alerts: MonitoringAlert[];
    }>;
    getMonitoringServices: (config?: ApiConfig) => Promise<{
        services: MonitoringService[];
        timestamp?: string;
    }>;
    getMonitoringLogs: (limit?: number, source?: string, config?: ApiConfig) => Promise<{
        entries: MonitoringLogEntry[];
        total: number;
    }>;
    getMonitoringMetrics: (params: {
        name?: string;
        start_time?: string;
        end_time?: string;
    }, config?: ApiConfig) => Promise<{
        metrics: MonitoringMetric[];
    }>;
    getMonitoringHealth: (config?: ApiConfig) => Promise<Record<string, unknown>>;
    listCrewaiWorkflows: (config?: ApiConfig) => Promise<{
        workflows: CrewAIWorkflowSummary[];
        total: number;
    }>;
    getCrewaiWorkflow: (workflowId: string, config?: ApiConfig) => Promise<Record<string, unknown>>;
    getCrewaiWorkflowTickets: (workflowId: string, config?: ApiConfig) => Promise<{
        tickets: CrewAIWorkflowTicket[];
        total: number;
        status_breakdown?: Record<string, number>;
    }>;
    listPrompts: (config?: ApiConfig) => Promise<{
        prompts: PromptMap;
    }>;
    savePrompt: (payload: {
        name: string;
        content: string;
    }, config?: ApiConfig) => Promise<{
        success: boolean;
        name: string;
    }>;
    deletePrompt: (promptName: string, config?: ApiConfig) => Promise<{
        success: boolean;
    }>;
    listContentTemplates: (config?: ApiConfig) => Promise<{
        templates: string[];
    }>;
    getContentTemplate: (templateName: string, config?: ApiConfig) => Promise<ContentTemplate>;
    createContentTemplate: (payload: {
        name: string;
        content: string;
    }, config?: ApiConfig) => Promise<{
        template: Record<string, unknown>;
    }>;
    validateProtocol: (payload: {
        protocol_text: string;
        validation_type?: string;
    }, config?: ApiConfig) => Promise<ProtocolValidationResult>;
    listAuditLogs: (limit?: number, config?: ApiConfig) => Promise<{
        logs: AuditLogEntry[];
        total: number;
    }>;
    getIcrOverview: (config?: ApiConfig) => Promise<IcrOverview>;
    getIcrComponents: (config?: ApiConfig) => Promise<IcrComponents>;
    getIcrRefinements: (config?: ApiConfig) => Promise<IcrRefinements>;
    getAdaptiveMdapHealth: (config?: ApiConfig) => Promise<{
        status: string;
        version?: string;
        details?: any;
    }>;
    getAdaptiveMdapDashboard: (config?: ApiConfig) => Promise<AdaptiveMdapDashboard>;
    getAdaptiveMdapProfiles: (config?: ApiConfig) => Promise<AdaptiveMdapProfiles>;
    getAdaptiveMdapProfileConfig: (profileName: string, config?: ApiConfig) => Promise<Record<string, unknown>>;
    calculateAdaptiveMdapCost: (payload: {
        num_problems: number;
        workload_distribution?: Record<string, number>;
        model?: string;
    }, config?: ApiConfig) => Promise<Record<string, unknown>>;
    classifyAdaptiveMdapComplexity: (payload: {
        description: string;
        domain?: string;
        depth?: number;
        dependencies?: string[];
        constraints?: string[];
        success_criteria?: string[];
        context?: Record<string, unknown>;
    }, config?: ApiConfig) => Promise<Record<string, unknown>>;
    allocateAdaptiveMdapResources: (payload: {
        complexity_score: number;
        context?: Record<string, unknown>;
    }, config?: ApiConfig) => Promise<Record<string, unknown>>;
    getHealth: (config?: ApiConfig) => Promise<Record<string, unknown>>;
    listKnowledgeArtifacts: (config?: ApiConfig) => Promise<{
        artifacts: KnowledgeArtifact[];
    }>;
    getKnowledgeArtifact: (artifactId: string, config?: ApiConfig) => Promise<KnowledgeArtifact>;
    createKnowledgeArtifact: (payload: Record<string, unknown>, config?: ApiConfig) => Promise<KnowledgeArtifact>;
    deleteKnowledgeArtifact: (artifactId: string, config?: ApiConfig) => Promise<{
        success: boolean;
    }>;
    searchKnowledge: (payload: Record<string, unknown>, config?: ApiConfig) => Promise<{
        results: KnowledgeArtifact[];
    }>;
    getKnowledgeGraph: (config?: ApiConfig) => Promise<KnowledgeGraph>;
    getKnowledgeStats: (config?: ApiConfig) => Promise<KnowledgeStats>;
    getKnowledgeRecommendations: (payload: Record<string, unknown>, config?: ApiConfig) => Promise<KnowledgeRecommendations>;
    exportKnowledgeBase: (config?: ApiConfig) => Promise<Record<string, unknown>>;
    importKnowledgeBase: (payload: Record<string, unknown>, config?: ApiConfig) => Promise<{
        success: boolean;
    }>;
    getAutoApprovalConfig: (config?: ApiConfig) => Promise<AutoApprovalConfig>;
    updateAutoApprovalConfig: (payload: AutoApprovalConfig, config?: ApiConfig) => Promise<AutoApprovalConfig>;
    testAutoApproval: (payload: Record<string, unknown>, config?: ApiConfig) => Promise<{
        results: AutoApprovalTestResult[];
    }>;
    getAutoApprovalAudit: (config?: ApiConfig) => Promise<{
        logs: AutoApprovalAuditEntry[];
    }>;
    listWorkflowTemplates: (config?: ApiConfig) => Promise<{
        templates: WorkflowTemplate[];
    }>;
    createWorkflowTemplate: (payload: Record<string, unknown>, config?: ApiConfig) => Promise<WorkflowTemplate>;
    updateWorkflowTemplate: (templateId: string, payload: Record<string, unknown>, config?: ApiConfig) => Promise<WorkflowTemplate>;
    deleteWorkflowTemplate: (templateId: string, config?: ApiConfig) => Promise<{
        success: boolean;
    }>;
    exportWorkflowTemplates: (config?: ApiConfig) => Promise<Record<string, unknown>>;
    importWorkflowTemplates: (payload: Record<string, unknown>, config?: ApiConfig) => Promise<{
        success: boolean;
    }>;
    listProviders: (config?: ApiConfig) => Promise<{
        providers: ProviderSummary[];
    }>;
    getProviderModels: (providerId: string, apiKey?: string, config?: ApiConfig) => Promise<{
        models: string[];
    }>;
    getParameterSchema: (config?: ApiConfig) => Promise<{
        parameters: ParameterDefinition[];
    }>;
    getParameterDefaults: (config?: ApiConfig) => Promise<Record<string, unknown>>;
    validateParameters: (payload: Record<string, unknown>, config?: ApiConfig) => Promise<ParameterValidationResult>;
    getParameterCategories: (config?: ApiConfig) => Promise<{
        categories: string[];
    }>;
    listVersions: (config?: ApiConfig) => Promise<{
        versions: VersionEntry[];
        current_version_id?: string | null;
    }>;
    getVersion: (versionId: string, config?: ApiConfig) => Promise<VersionEntry>;
    getCurrentVersion: (config?: ApiConfig) => Promise<{
        current: VersionEntry | null;
    }>;
    createVersion: (payload: {
        protocol_text: string;
        version_name?: string;
        comment?: string;
        author?: string;
    }, config?: ApiConfig) => Promise<{
        version_id: string;
        version: VersionEntry;
    }>;
    loadVersion: (versionId: string, config?: ApiConfig) => Promise<{
        loaded: boolean;
        current: VersionEntry | null;
    }>;
    branchVersion: (versionId: string, payload: {
        new_version_name: string;
    }, config?: ApiConfig) => Promise<{
        version_id: string;
        version: VersionEntry;
    }>;
    compareVersions: (payload: {
        version_id_1: string;
        version_id_2: string;
    }, config?: ApiConfig) => Promise<VersionCompareResult>;
    deleteVersion: (versionId: string, config?: ApiConfig) => Promise<{
        deleted: boolean;
    }>;
    listValidationRules: (config?: ApiConfig) => Promise<{
        rules: Record<string, ValidationRule>;
        rule_names: string[];
    }>;
    getValidationRule: (ruleName: string, config?: ApiConfig) => Promise<{
        name: string;
        rule: ValidationRule;
    }>;
    createValidationRule: (payload: {
        name: string;
        max_length?: number | null;
        min_length?: number | null;
        required_keywords?: string[];
        forbidden_patterns?: string[];
        required_sections?: string[];
    }, config?: ApiConfig) => Promise<{
        created: boolean;
        rule_name: string;
        rule: ValidationRule;
    }>;
    updateValidationRule: (ruleName: string, payload: {
        name?: string;
        max_length?: number | null;
        min_length?: number | null;
        required_keywords?: string[] | null;
        forbidden_patterns?: string[] | null;
        required_sections?: string[] | null;
    }, config?: ApiConfig) => Promise<{
        updated: boolean;
        rule_name: string;
        rule: ValidationRule;
    }>;
    deleteValidationRule: (ruleName: string, config?: ApiConfig) => Promise<{
        deleted: boolean;
        rule_name: string;
    }>;
    runValidation: (payload: {
        content: string;
        rule_names: string[];
    }, config?: ApiConfig) => Promise<ValidationRunResult>;
    runComplianceCheck: (payload: {
        content: string;
        framework?: string;
    }, config?: ApiConfig) => Promise<ComplianceCheckResult>;
    listWorkflowDefinitions: (config?: ApiConfig) => Promise<{
        definitions: WorkflowDefinitionSummary[];
    }>;
    getWorkflowDefinition: (definitionId: string, config?: ApiConfig) => Promise<WorkflowDefinitionDetail>;
    createWorkflowDefinition: (payload: {
        name: string;
        description: string;
        workflow_type: string;
        parameters: Record<string, unknown>;
    }, config?: ApiConfig) => Promise<{
        definition_id: string;
    }>;
    listWorkflowInstances: (config?: ApiConfig) => Promise<{
        instances: WorkflowInstanceSummary[];
    }>;
    getWorkflowInstance: (instanceId: string, config?: ApiConfig) => Promise<WorkflowInstanceDetail>;
    createWorkflowInstance: (payload: {
        definition_id: string;
        instance_name: string;
        inputs: Record<string, unknown>;
    }, config?: ApiConfig) => Promise<{
        instance_id: string;
    }>;
    syncWorkflowInstanceParameters: (instanceId: string, payload: {
        parameters: Record<string, unknown>;
    }, config?: ApiConfig) => Promise<Record<string, unknown>>;
    startWorkflowInstance: (instanceId: string, config?: ApiConfig) => Promise<Record<string, unknown>>;
    pauseWorkflowInstance: (instanceId: string, config?: ApiConfig) => Promise<Record<string, unknown>>;
    resumeWorkflowInstance: (instanceId: string, config?: ApiConfig) => Promise<Record<string, unknown>>;
    stopWorkflowInstance: (instanceId: string, config?: ApiConfig) => Promise<Record<string, unknown>>;
    cancelWorkflowInstance: (instanceId: string, config?: ApiConfig) => Promise<Record<string, unknown>>;
    restartWorkflowInstance: (instanceId: string, config?: ApiConfig) => Promise<Record<string, unknown>>;
    deleteWorkflowInstance: (instanceId: string, config?: ApiConfig) => Promise<Record<string, unknown>>;
    getSovereignHealth: (config?: ApiConfig) => Promise<Record<string, unknown>>;
    listSovereignProblems: (config?: ApiConfig) => Promise<{
        problems: Record<string, unknown>[];
    }>;
    getContentSuggestions: (payload: Record<string, unknown>, config?: ApiConfig) => Promise<{
        suggestions: string[];
    }>;
    getContentClassification: (payload: Record<string, unknown>, config?: ApiConfig) => Promise<Record<string, unknown>>;
    getSecuritySuggestions: (payload: Record<string, unknown>, config?: ApiConfig) => Promise<{
        vulnerabilities: string[];
    }>;
    getImprovementPotential: (payload: Record<string, unknown>, config?: ApiConfig) => Promise<{
        score: number;
    }>;
    listEvaluators: (config?: ApiConfig) => Promise<EvaluatorListResponse>;
    uploadEvaluator: (payload: {
        code: string;
    }, config?: ApiConfig) => Promise<EvaluatorUploadResponse>;
    deleteEvaluator: (evaluatorId: string, config?: ApiConfig) => Promise<{
        success: boolean;
        evaluator_id: string;
    }>;
    updateWorkflowPlan: (workflowId: string, payload: WorkflowPlanUpdateRequest, config?: ApiConfig) => Promise<{
        message: string;
        execution_order: string[];
    }>;
    getWorkflowResourceUsage: (workflowId: string, config?: ApiConfig) => Promise<WorkflowResourceUsageResponse>;
    optimizeWorkflowResources: (workflowId: string, config?: ApiConfig) => Promise<WorkflowResourceOptimizationResponse>;
    runIntegratedWorkflow: (payload: IntegratedWorkflowRequest, config?: ApiConfig) => Promise<Record<string, unknown>>;
    listOrchestrationModels: (config?: ApiConfig) => Promise<ModelOrchestrationListResponse>;
    registerOrchestrationModel: (payload: ModelOrchestrationRegisterRequest, config?: ApiConfig) => Promise<ModelOrchestrationRegisterResponse>;
    executeOrchestrationEnsemble: (payload: ModelOrchestrationEnsembleRequest, config?: ApiConfig) => Promise<ModelOrchestrationEnsembleResponse>;
    getBubblelabsStatus: (config?: ApiConfig) => Promise<BubbleLabsStatusResponse>;
    initializeBubblelabs: (config?: ApiConfig) => Promise<BubbleLabsInitializeResponse>;
    bubblelabsControlCatalog: (config?: ApiConfig) => Promise<BubbleLabsControlCatalogResponse>;
    bubblelabsControlDiscover: (payload?: {
        force?: boolean;
    }, config?: ApiConfig) => Promise<BubbleLabsControlDiscoverResponse>;
    bubblelabsControlExecute: (payload: {
        component: string;
        action: string;
        payload?: Record<string, unknown>;
    }, config?: ApiConfig) => Promise<BubbleLabsControlExecuteResponse>;
    bubblelabsAceSkillbook: (payload: {
        name: string;
        skills: Array<Record<string, unknown>>;
    }, config?: ApiConfig) => Promise<BubbleLabsActionResponse>;
    bubblelabsAcePatterns: (payload: {
        workflow_results: Array<Record<string, unknown>>;
    }, config?: ApiConfig) => Promise<BubbleLabsActionResponse>;
    bubblelabsZ3Solve: (payload: {
        variables: Array<Record<string, unknown>>;
        constraints: string[];
    }, config?: ApiConfig) => Promise<BubbleLabsActionResponse>;
    bubblelabsZ3Prove: (payload: {
        theorem: string;
    }, config?: ApiConfig) => Promise<BubbleLabsActionResponse>;
    bubblelabsRomaAnalyze: (payload: {
        problem: string;
        max_depth?: number;
    }, config?: ApiConfig) => Promise<BubbleLabsActionResponse>;
    bubblelabsRomaConfig: (payload: {
        config: Record<string, unknown>;
    }, config?: ApiConfig) => Promise<BubbleLabsActionResponse>;
    bubblelabsKnowledgeStore: (payload: {
        artifact: Record<string, unknown>;
    }, config?: ApiConfig) => Promise<BubbleLabsActionResponse>;
    bubblelabsKnowledgeQuery: (payload: {
        query: string;
    }, config?: ApiConfig) => Promise<BubbleLabsActionResponse>;
    bubblelabsAnalyticsTrack: (payload: {
        workflow_id: string;
        metrics: Record<string, unknown>;
    }, config?: ApiConfig) => Promise<BubbleLabsActionResponse>;
    bubblelabsAnalyticsDashboard: (config?: ApiConfig) => Promise<BubbleLabsActionResponse>;
    bubblelabsLeanAideProve: (payload: {
        theorem: string;
    }, config?: ApiConfig) => Promise<BubbleLabsActionResponse>;
    getMakerStatus: (config?: ApiConfig) => Promise<{
        available: boolean;
    }>;
    listMakerTools: (params?: {
        status?: string;
        maker_mode?: string;
        search?: string;
    }, config?: ApiConfig) => Promise<MakerToolListResponse>;
    getMakerTool: (toolId: string, config?: ApiConfig) => Promise<MakerToolResponse>;
    createMakerTool: (payload: {
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
    }, config?: ApiConfig) => Promise<MakerToolResponse>;
    testMakerTool: (toolId: string, payload: {
        input_data: Record<string, unknown>;
        delegate_to_crewai?: boolean;
    }, config?: ApiConfig) => Promise<MakerExecutionResponse>;
    validateMakerTool: (toolId: string, config?: ApiConfig) => Promise<{
        status: string;
    }>;
    executeMakerTool: (toolId: string, payload: {
        input_data: Record<string, unknown>;
        delegate_to_crewai?: boolean;
    }, config?: ApiConfig) => Promise<MakerExecutionResponse>;
    listMakerDelegations: (params?: {
        status?: string;
        delegation_type?: string;
    }, config?: ApiConfig) => Promise<MakerDelegationListResponse>;
    syncMakerDelegations: (config?: ApiConfig) => Promise<{
        synced: number;
    }>;
    bubblelabsKnowledgeStatus: (config?: ApiConfig) => Promise<{
        initialized: boolean;
        query_history_count: number;
    }>;
    bubblelabsKnowledgeQueryAdvanced: (payload: {
        query: string;
        sources?: string[];
        bedrock_kb_id?: string;
        index_path?: string;
    }, config?: ApiConfig) => Promise<KnowledgeExplorerQueryResponse>;
    bubblelabsKnowledgeQueryHistory: (config?: ApiConfig) => Promise<KnowledgeExplorerHistoryResponse>;
    bubblelabsKnowledgeExtract: (payload: {
        source_type: string;
        source_value: string;
        extraction_config?: Record<string, unknown>;
    }, config?: ApiConfig) => Promise<KnowledgeExplorerExtractResponse>;
    bubblelabsKnowledgeExtractFile: (file: File, extractionConfig?: Record<string, unknown>, config?: ApiConfig) => Promise<KnowledgeExplorerExtractResponse>;
    bubblelabsLeanAideStatus: (config?: ApiConfig) => Promise<LeanAideStatusResponse>;
    bubblelabsLeanAideExecute: (payload: {
        task_type: string;
        payload: Record<string, unknown>;
    }, config?: ApiConfig) => Promise<LeanAideExecuteResponse>;
    bubblelabsLeanAideTrees: (config?: ApiConfig) => Promise<LeanAideTreeListResponse>;
    bubblelabsLeanAideTree: (treeId: string, config?: ApiConfig) => Promise<LeanAideTreeResponse>;
    bubblelabsLeanAideProofs: (config?: ApiConfig) => Promise<LeanAideProofListResponse>;
    bubblelabsLeanAideProof: (proofId: string, config?: ApiConfig) => Promise<LeanAideProofResponse>;
    startEvolutionRun: (payload: {
        content: string;
        content_type?: string;
        evolution_mode?: string;
        parameters?: Record<string, unknown>;
        gauntlet_name?: string;
        use_decomposition?: boolean;
    }, config?: ApiConfig) => Promise<EvolutionRunResponse>;
    listEvolutionRuns: (config?: ApiConfig) => Promise<EvolutionRunListResponse>;
    getEvolutionRun: (runId: string, config?: ApiConfig) => Promise<EvolutionRunStatus>;
    stopEvolutionRun: (runId: string, config?: ApiConfig) => Promise<{
        status: string;
    }>;
    startAdversarialRun: (payload: {
        content: string;
        content_type?: string;
        parameters?: Record<string, unknown>;
        use_decomposition?: boolean;
    }, config?: ApiConfig) => Promise<AdversarialRunResponse>;
    listAdversarialRuns: (config?: ApiConfig) => Promise<AdversarialRunListResponse>;
    getAdversarialRun: (runId: string, config?: ApiConfig) => Promise<AdversarialRunStatus>;
    stopAdversarialRun: (runId: string, config?: ApiConfig) => Promise<{
        status: string;
    }>;
    executeGauntlet: (gauntletName: string, payload: {
        content: string;
        content_type?: string;
        evolution_mode?: string;
        parameters?: Record<string, unknown>;
    }, config?: ApiConfig) => Promise<EvolutionRunResponse>;
    getGauntletExecutionStatus: (executionId: string, config?: ApiConfig) => Promise<EvolutionRunStatus>;
    listGauntletExecutions: (gauntletName?: string, config?: ApiConfig) => Promise<{
        executions: Array<EvolutionRunStatus>;
    }>;
    executeDecomposition: (workflowId: string, payload: {
        problem_statement: string;
        content_type?: string;
        decomposition_method?: string;
        granularity?: string;
        max_depth?: number;
        max_sub_problems?: number;
        parameters?: Record<string, unknown>;
    }, config?: ApiConfig) => Promise<{
        execution_id: string;
        status: string;
    }>;
    getDecompositionExecutionStatus: (executionId: string, config?: ApiConfig) => Promise<{
        execution_id: string;
        status: string;
        sub_problems_completed: number;
        sub_problems_total: number;
        current_sub_problem?: string;
        results?: Record<string, unknown>;
    }>;
    listDecompositionExecutions: (workflowId?: string, config?: ApiConfig) => Promise<{
        executions: Array<{
            execution_id: string;
            status: string;
            created_at: string;
        }>;
    }>;
    executeWorkflowTemplate: (templateId: string, payload: {
        parameters: Record<string, unknown>;
        callback_url?: string;
    }, config?: ApiConfig) => Promise<{
        execution_id: string;
        status: string;
        template_id: string;
    }>;
    getWorkflowTemplateExecutionStatus: (executionId: string, config?: ApiConfig) => Promise<{
        execution_id: string;
        status: string;
        template_id: string;
        current_step?: string;
        completed_steps: string[];
        results?: Record<string, unknown>;
        error?: string;
    }>;
    stopWorkflowTemplateExecution: (executionId: string, config?: ApiConfig) => Promise<{
        status: string;
    }>;
    getExecutionStatus: (executionType: "gauntlet" | "decomposition" | "workflow-template", executionId: string, config?: ApiConfig) => Promise<EvolutionRunStatus> | Promise<{
        execution_id: string;
        status: string;
        sub_problems_completed: number;
        sub_problems_total: number;
    }> | Promise<{
        execution_id: string;
        status: string;
        current_step?: string;
        completed_steps: string[];
    }>;
};
/**
 * Backward-compatible client wrapper for contract tests and legacy consumers.
 * New code should prefer direct `openevolveApi.*` function calls.
 */
export declare class OpenEvolveClient {
    private readonly config;
    constructor(config?: ApiConfig);
    health(): Promise<Record<string, unknown>>;
    listTeams(): Promise<{
        teams: TeamSummary[];
        total: number;
    }>;
    listWorkflows(): Promise<{
        workflows: WorkflowSummary[];
        total: number;
    }>;
    listGauntlets(): Promise<{
        gauntlets: GauntletSummary[];
        total: number;
    }>;
    controlCatalog(): Promise<BubbleLabsControlCatalogResponse>;
}
//# sourceMappingURL=openevolveApi.d.ts.map