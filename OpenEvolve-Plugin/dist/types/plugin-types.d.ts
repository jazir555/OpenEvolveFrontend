/**
 * OpenEvolve BubbleLabs Plugin - TypeScript Interfaces
 *
 * This file contains all TypeScript interfaces, types, and constants for the OpenEvolve plugin.
 * The interfaces follow the same pattern as other BubbleLabs plugins (LeanAIDE, ClaudieMiro, Datapizza, ROMA).
 *
 * This plugin integrates the complete OpenEvolve system including:
 * - Evolution functionality (genetic algorithms, evolutionary optimization)
 * - Adversarial functionality (red/blue team testing, quality improvement)
 * - Decomposition functionality (problem decomposition, task breakdown)
 * - Full integration with ROMA, MDAP/MAKER, and MCP systems
 */
/**
 * OpenEvolve Plugin Metadata
 */
export interface OpenEvolvePluginMetadata {
    name: string;
    version: string;
    description: string;
    author: string;
    license: string;
    repository?: string;
    documentation?: string;
}
/**
 * OpenEvolve Execution Status
 */
export type OpenEvolveExecutionStatus = 'initializing' | 'idle' | 'configuring' | 'executing' | 'paused' | 'completed' | 'failed' | 'cancelled';
/**
 * OpenEvolve Module Types (Core OpenEvolve Architecture)
 */
export type OpenEvolveModuleType = 'evolution' | 'adversarial' | 'decomposition' | 'integration';
/**
 * Evolution Strategy Types
 */
export type EvolutionStrategy = 'standard' | 'genetic_algorithm' | 'quality_diversity' | 'novelty_search' | 'multi_objective' | 'adaptive' | 'hybrid';
/**
 * Adversarial Strategy Types
 */
export type AdversarialStrategy = 'red_blue_team' | 'multi_agent' | 'self_play' | 'co_evolution' | 'competitive' | 'cooperative';
/**
 * Decomposition Strategy Types
 */
export type DecompositionStrategy = 'semantic' | 'hierarchical' | 'functional' | 'modular' | 'temporal' | 'hybrid';
/**
 * Evolution Configuration Interface
 * Comprehensive configuration for evolutionary algorithms and optimization
 */
export interface EvolutionConfig {
    evolutionMode: EvolutionStrategy;
    maxIterations: number;
    populationSize: number;
    temperature: number;
    maxTokens: number;
    seed?: number | null;
    earlyStopping: boolean;
    convergenceThreshold: number;
    fitnessFunction: string;
    selectionPressure: number;
    mutationRate: number;
    crossoverRate: number;
    elitism: boolean;
    diversityMaintenance: boolean;
    adaptiveParameters: boolean;
    reasoningEffort: string;
    language: string;
    fileSuffix: string;
    apiKey: string;
    apiBase: string;
    modelId: string;
    backupModels: string[] | null;
    timeout: number;
    maxRetries: number;
    retryDelay: number;
    rateLimit: number;
    concurrentRequests: number;
    modelRotation: boolean;
    topP: number;
    frequencyPenalty: number;
    presencePenalty: number;
    n: number;
    logitBias: Record<string, number> | null;
    stopSequences: string[] | null;
    logprobs: boolean;
    topLogprobs: number;
    responseFormat: string;
    featureDimensions: string[] | null;
    featureBins: number;
    archiveSize: number;
    noveltyThreshold: number;
    mctsEnabled: boolean;
    mctsIterations: number;
    explorationWeight: number;
    rolloutDepth: number;
    treeReuse: boolean;
    mdapMakerEnabled: boolean;
    mdapMakerAutoSelect: boolean;
    mdapMakerMaxDepth: number;
    mdapMakerKAhead: number;
    mdapMakerRedFlagging: boolean;
    mdapMakerAdaptiveK: boolean;
    mdapMakerProvider: string;
    mdapMakerModel: string;
    mdapMakerAutoSelectionKeywords: string[];
}
/**
 * Adversarial Configuration Interface
 * Comprehensive configuration for adversarial testing and improvement
 */
export interface AdversarialConfig {
    adversarialMode: AdversarialStrategy;
    redTeamSize: number;
    blueTeamSize: number;
    evaluatorTeamSize: number;
    maxRounds: number;
    critiqueDepth: string;
    improvementDepth: string;
    evaluationStrictness: string;
    contentType: string;
    requirements: string[];
    complianceStandards: string[];
    maxContentLength: number;
    minContentLength: number;
    allowContentGeneration: boolean;
    allowContentModification: boolean;
    allowStructuralChanges: boolean;
    preserveOriginalFunctionality: boolean;
    focusAreas: string[];
    excludeAreas: string[];
    redTeamAggressiveness: number;
    blueTeamCreativity: number;
    evaluatorRigor: number;
    teamDiversity: number;
    collaborationLevel: number;
    parallelExecution: boolean;
    maxParallelTasks: number;
    timeoutSeconds: number;
    maxRetries: number;
    fallbackStrategy: string;
    qualityThreshold: number;
    improvementThreshold: number;
    acceptanceThreshold: number;
    metricWeights: Record<string, number>;
    mdapMakerEnabled: boolean;
    mdapMakerAutoSelect: boolean;
    mdapMakerMaxDepth: number;
    mdapMakerKAhead: number;
    mdapMakerRedFlagging: boolean;
    mdapMakerAdaptiveK: boolean;
    mdapMakerProvider: string;
    mdapMakerModel: string;
    mdapMakerAutoSelectionKeywords: string[];
}
/**
 * Decomposition Configuration Interface
 * Comprehensive configuration for problem decomposition
 */
export interface DecompositionConfig {
    decompositionStrategy: DecompositionStrategy;
    maxSubProblems: number;
    minSubProblemSize: number;
    maxSubProblemSize: number;
    granularityLevel: string;
    dependencyAnalysis: boolean;
    complexityAnalysis: boolean;
    feasibilityAnalysis: boolean;
    validationRequired: boolean;
    successCriteriaRequired: boolean;
    dependencyGraphRequired: boolean;
    semanticAnalysisEnabled: boolean;
    hierarchicalDepth: number;
    functionalGranularity: string;
    modularIndependence: number;
    temporalGranularity: string;
    qualityThreshold: number;
    completenessThreshold: number;
    clarityThreshold: number;
    feasibilityThreshold: number;
    validationThreshold: number;
    maxIterations: number;
    timeoutSeconds: number;
    maxRetries: number;
    fallbackStrategy: string;
    parallelProcessing: boolean;
    maxParallelTasks: number;
    knowledgeBaseEnabled: boolean;
    knowledgeBaseSources: string[];
    contextAnalysisEnabled: boolean;
    domainSpecificAnalysis: boolean;
    mdapMakerEnabled: boolean;
    mdapMakerAutoSelect: boolean;
    mdapMakerMaxDepth: number;
    mdapMakerKAhead: number;
    mdapMakerRedFlagging: boolean;
    mdapMakerAdaptiveK: boolean;
    mdapMakerProvider: string;
    mdapMakerModel: string;
    mdapMakerAutoSelectionKeywords: string[];
}
/**
 * OpenEvolve Execution Statistics
 */
export interface OpenEvolveExecutionStatistics {
    executionId: string;
    startTime: string;
    endTime: string | null;
    durationMs: number | null;
    status: OpenEvolveExecutionStatus;
    module: OpenEvolveModuleType;
    strategy: string;
    iterations: number;
    successRate: number;
    errorCount: number;
    warningCount: number;
    tokensUsed: number;
    apiCalls: number;
    cacheHits: number;
    cacheMisses: number;
    performanceScore: number;
    qualityScore: number;
    improvementScore: number;
    complexityReduction: number;
    errorMessages: string[];
    warningMessages: string[];
}
/**
 * OpenEvolve Execution Result
 */
export interface OpenEvolveExecutionResult {
    executionId: string;
    status: OpenEvolveExecutionStatus;
    module: OpenEvolveModuleType;
    input: any;
    output: any;
    statistics: OpenEvolveExecutionStatistics;
    error?: Error | null;
    timestamp: string;
}
/**
 * OpenEvolve Plugin State
 */
export interface OpenEvolvePluginState {
    initialized: boolean;
    metadata: OpenEvolvePluginMetadata;
    status: OpenEvolveExecutionStatus;
    currentExecutionId: string | null;
    executionHistory: OpenEvolveExecutionResult[];
    statistics: OpenEvolveExecutionStatistics[];
    evolutionConfig: EvolutionConfig;
    adversarialConfig: AdversarialConfig;
    decompositionConfig: DecompositionConfig;
    defaultExecutionMethod: string;
    mdapMaker?: {
        enabled: boolean;
        autoSelect: boolean;
        maxDepth: number;
        kAhead: number;
        redFlagging: boolean;
        adaptiveK: boolean;
        provider: string;
        model: string;
        autoSelectionKeywords: string[];
    };
}
/**
 * OpenEvolve Execution Options
 */
export interface OpenEvolveExecutionOptions {
    executionMethod?: string;
    evolutionConfig?: Partial<EvolutionConfig>;
    adversarialConfig?: Partial<AdversarialConfig>;
    decompositionConfig?: Partial<DecompositionConfig>;
    mdapMakerConfig?: {
        enabled?: boolean;
        autoSelect?: boolean;
        maxDepth?: number;
        kAhead?: number;
        redFlagging?: boolean;
        adaptiveK?: boolean;
        provider?: string;
        model?: string;
        autoSelectionKeywords?: string[];
    };
    timeout?: number;
    maxRetries?: number;
    cacheTTL?: number;
}
/**
 * OpenEvolve Plugin Interface
 */
export interface OpenEvolvePlugin {
    getMetadata(): OpenEvolvePluginMetadata;
    getState(): OpenEvolvePluginState;
    initialize(config?: Partial<OpenEvolvePluginState>): Promise<void>;
    updateConfig(config: Partial<OpenEvolvePluginState>): Promise<void>;
    resetConfig(): Promise<void>;
    getConfig(): OpenEvolvePluginState;
    executeEvolution(goal: string, options?: OpenEvolveExecutionOptions): Promise<OpenEvolveExecutionResult>;
    executeAdversarial(content: string, options?: OpenEvolveExecutionOptions): Promise<OpenEvolveExecutionResult>;
    executeDecomposition(problem: string, options?: OpenEvolveExecutionOptions): Promise<OpenEvolveExecutionResult>;
    executeIntegrated(goal: string, options?: OpenEvolveExecutionOptions): Promise<OpenEvolveExecutionResult>;
    getExecution(executionId: string): Promise<OpenEvolveExecutionResult | null>;
    getExecutionHistory(): Promise<OpenEvolveExecutionResult[]>;
    getStatistics(): Promise<OpenEvolveExecutionStatistics[]>;
    cancelExecution(executionId: string): Promise<boolean>;
    clearHistory(): Promise<void>;
    shouldUseMdapMakerForGoal(goal: string): boolean;
    getMdapMakerConfig(): any | null;
    validateConfig(): Promise<{
        valid: boolean;
        errors: string[];
    }>;
    getAvailableStrategies(): {
        evolution: EvolutionStrategy[];
        adversarial: AdversarialStrategy[];
        decomposition: DecompositionStrategy[];
    };
}
/**
 * OpenEvolve Plugin Constants
 */
export declare const OPENEVOLVE_PLUGIN_CONSTANTS: {
    PLUGIN_NAME: string;
    PLUGIN_VERSION: string;
    PLUGIN_DESCRIPTION: string;
    PLUGIN_AUTHOR: string;
    PLUGIN_LICENSE: string;
    DEFAULT_EVOLUTION_MODE: EvolutionStrategy;
    DEFAULT_ADVERSARIAL_MODE: AdversarialStrategy;
    DEFAULT_DECOMPOSITION_STRATEGY: DecompositionStrategy;
    DEFAULT_MAX_ITERATIONS: number;
    DEFAULT_POPULATION_SIZE: number;
    DEFAULT_TEMPERATURE: number;
    DEFAULT_MAX_TOKENS: number;
    DEFAULT_MODEL_ID: string;
    DEFAULT_API_BASE: string;
    DEFAULT_TIMEOUT: number;
    DEFAULT_MAX_RETRIES: number;
    DEFAULT_RETRY_DELAY: number;
    DEFAULT_MDAP_MAKER_ENABLED: boolean;
    DEFAULT_MDAP_MAKER_AUTO_SELECT: boolean;
    DEFAULT_MDAP_MAKER_MAX_DEPTH: number;
    DEFAULT_MDAP_MAKER_K_AHEAD: number;
    DEFAULT_MDAP_MAKER_RED_FLAGGING: boolean;
    DEFAULT_MDAP_MAKER_ADAPTIVE_K: boolean;
    DEFAULT_MDAP_MAKER_PROVIDER: string;
    DEFAULT_MDAP_MAKER_MODEL: string;
    DEFAULT_MDAP_MAKER_KEYWORDS: string[];
    EXECUTION_METHODS: string[];
    DEFAULT_EXECUTION_METHOD: string;
    EVOLUTION_STRATEGIES: string[];
    ADVERSARIAL_STRATEGIES: string[];
    DECOMPOSITION_STRATEGIES: string[];
};
/**
 * Default OpenEvolve Configuration
 */
export declare const DEFAULT_OPENEVOLVE_CONFIG: OpenEvolvePluginState;
