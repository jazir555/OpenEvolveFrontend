"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.DEFAULT_OPENEVOLVE_CONFIG = exports.OPENEVOLVE_PLUGIN_CONSTANTS = void 0;
/**
 * OpenEvolve Plugin Constants
 */
exports.OPENEVOLVE_PLUGIN_CONSTANTS = {
    PLUGIN_NAME: 'OpenEvolve BubbleLabs Plugin',
    PLUGIN_VERSION: '1.0.0',
    PLUGIN_DESCRIPTION: 'Comprehensive OpenEvolve system integration for BubbleLabs',
    PLUGIN_AUTHOR: 'OpenEvolve Team',
    PLUGIN_LICENSE: 'MIT',
    // Default Configuration Values
    DEFAULT_EVOLUTION_MODE: 'standard',
    DEFAULT_ADVERSARIAL_MODE: 'red_blue_team',
    DEFAULT_DECOMPOSITION_STRATEGY: 'semantic',
    DEFAULT_MAX_ITERATIONS: 10,
    DEFAULT_POPULATION_SIZE: 20,
    DEFAULT_TEMPERATURE: 0.7,
    DEFAULT_MAX_TOKENS: 2048,
    DEFAULT_MODEL_ID: 'gpt-4',
    DEFAULT_API_BASE: 'https://api.openai.com/v1',
    DEFAULT_TIMEOUT: 30,
    DEFAULT_MAX_RETRIES: 3,
    DEFAULT_RETRY_DELAY: 1.0,
    // MDAP/MAKER Defaults
    DEFAULT_MDAP_MAKER_ENABLED: false,
    DEFAULT_MDAP_MAKER_AUTO_SELECT: true,
    DEFAULT_MDAP_MAKER_MAX_DEPTH: 5,
    DEFAULT_MDAP_MAKER_K_AHEAD: 3,
    DEFAULT_MDAP_MAKER_RED_FLAGGING: true,
    DEFAULT_MDAP_MAKER_ADAPTIVE_K: true,
    DEFAULT_MDAP_MAKER_PROVIDER: 'openai',
    DEFAULT_MDAP_MAKER_MODEL: 'gpt-4',
    DEFAULT_MDAP_MAKER_KEYWORDS: [
        'critical', 'important', 'high priority', 'mission critical',
        'production', 'deployment', 'security', 'sensitive'
    ],
    // Execution Methods
    EXECUTION_METHODS: ['auto', 'evolution', 'adversarial', 'decomposition', 'roma_mdap_maker'],
    DEFAULT_EXECUTION_METHOD: 'auto',
    // Strategy Options
    EVOLUTION_STRATEGIES: [
        'standard', 'genetic_algorithm', 'quality_diversity',
        'novelty_search', 'multi_objective', 'adaptive', 'hybrid'
    ],
    ADVERSARIAL_STRATEGIES: [
        'red_blue_team', 'multi_agent', 'self_play',
        'co_evolution', 'competitive', 'cooperative'
    ],
    DECOMPOSITION_STRATEGIES: [
        'semantic', 'hierarchical', 'functional',
        'modular', 'temporal', 'hybrid'
    ],
};
/**
 * Default OpenEvolve Configuration
 */
exports.DEFAULT_OPENEVOLVE_CONFIG = {
    initialized: false,
    metadata: {
        name: exports.OPENEVOLVE_PLUGIN_CONSTANTS.PLUGIN_NAME,
        version: exports.OPENEVOLVE_PLUGIN_CONSTANTS.PLUGIN_VERSION,
        description: exports.OPENEVOLVE_PLUGIN_CONSTANTS.PLUGIN_DESCRIPTION,
        author: exports.OPENEVOLVE_PLUGIN_CONSTANTS.PLUGIN_AUTHOR,
        license: exports.OPENEVOLVE_PLUGIN_CONSTANTS.PLUGIN_LICENSE,
    },
    status: 'idle',
    currentExecutionId: null,
    executionHistory: [],
    statistics: [],
    evolutionConfig: {
        // Core Evolution Parameters
        evolutionMode: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_EVOLUTION_MODE,
        maxIterations: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MAX_ITERATIONS,
        populationSize: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_POPULATION_SIZE,
        temperature: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_TEMPERATURE,
        maxTokens: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MAX_TOKENS,
        seed: null,
        earlyStopping: false,
        convergenceThreshold: 0.001,
        fitnessFunction: 'default',
        selectionPressure: 1.0,
        mutationRate: 0.1,
        crossoverRate: 0.8,
        elitism: true,
        diversityMaintenance: true,
        adaptiveParameters: false,
        reasoningEffort: 'medium',
        language: 'python',
        fileSuffix: '.py',
        // Model Configuration Parameters
        apiKey: '',
        apiBase: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_API_BASE,
        modelId: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MODEL_ID,
        backupModels: null,
        timeout: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_TIMEOUT,
        maxRetries: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MAX_RETRIES,
        retryDelay: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_RETRY_DELAY,
        rateLimit: 60,
        concurrentRequests: 5,
        modelRotation: false,
        topP: 1.0,
        frequencyPenalty: 0.0,
        presencePenalty: 0.0,
        n: 1,
        logitBias: null,
        stopSequences: null,
        logprobs: false,
        topLogprobs: 0,
        responseFormat: 'text',
        // Quality Diversity Parameters
        featureDimensions: null,
        featureBins: 10,
        archiveSize: 100,
        noveltyThreshold: 0.1,
        // Evolutionary MCTS Parameters
        mctsEnabled: false,
        mctsIterations: 100,
        explorationWeight: 1.4,
        rolloutDepth: 5,
        treeReuse: true,
        // MDAP/MAKER Integration
        mdapMakerEnabled: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_ENABLED,
        mdapMakerAutoSelect: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_AUTO_SELECT,
        mdapMakerMaxDepth: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_MAX_DEPTH,
        mdapMakerKAhead: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_K_AHEAD,
        mdapMakerRedFlagging: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_RED_FLAGGING,
        mdapMakerAdaptiveK: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_ADAPTIVE_K,
        mdapMakerProvider: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_PROVIDER,
        mdapMakerModel: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_MODEL,
        mdapMakerAutoSelectionKeywords: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_KEYWORDS,
    },
    adversarialConfig: {
        // Core Adversarial Parameters
        adversarialMode: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_ADVERSARIAL_MODE,
        redTeamSize: 3,
        blueTeamSize: 3,
        evaluatorTeamSize: 1,
        maxRounds: 5,
        critiqueDepth: 'medium',
        improvementDepth: 'medium',
        evaluationStrictness: 'medium',
        contentType: 'code',
        requirements: [],
        complianceStandards: [],
        maxContentLength: 5000,
        minContentLength: 50,
        allowContentGeneration: true,
        allowContentModification: true,
        allowStructuralChanges: true,
        preserveOriginalFunctionality: true,
        focusAreas: [],
        excludeAreas: [],
        // Team Configuration
        redTeamAggressiveness: 0.7,
        blueTeamCreativity: 0.8,
        evaluatorRigor: 0.9,
        teamDiversity: 0.5,
        collaborationLevel: 0.3,
        // Execution Parameters
        parallelExecution: false,
        maxParallelTasks: 3,
        timeoutSeconds: 300,
        maxRetries: 3,
        fallbackStrategy: 'default',
        // Quality Metrics
        qualityThreshold: 0.7,
        improvementThreshold: 0.2,
        acceptanceThreshold: 0.8,
        metricWeights: {},
        // MDAP/MAKER Integration
        mdapMakerEnabled: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_ENABLED,
        mdapMakerAutoSelect: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_AUTO_SELECT,
        mdapMakerMaxDepth: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_MAX_DEPTH,
        mdapMakerKAhead: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_K_AHEAD,
        mdapMakerRedFlagging: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_RED_FLAGGING,
        mdapMakerAdaptiveK: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_ADAPTIVE_K,
        mdapMakerProvider: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_PROVIDER,
        mdapMakerModel: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_MODEL,
        mdapMakerAutoSelectionKeywords: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_KEYWORDS,
    },
    decompositionConfig: {
        // Core Decomposition Parameters
        decompositionStrategy: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_DECOMPOSITION_STRATEGY,
        maxSubProblems: 10,
        minSubProblemSize: 50,
        maxSubProblemSize: 500,
        granularityLevel: 'medium',
        dependencyAnalysis: true,
        complexityAnalysis: true,
        feasibilityAnalysis: true,
        validationRequired: true,
        successCriteriaRequired: true,
        dependencyGraphRequired: true,
        // Strategy-Specific Parameters
        semanticAnalysisEnabled: true,
        hierarchicalDepth: 3,
        functionalGranularity: 'medium',
        modularIndependence: 0.7,
        temporalGranularity: 'medium',
        // Quality Parameters
        qualityThreshold: 0.7,
        completenessThreshold: 0.8,
        clarityThreshold: 0.7,
        feasibilityThreshold: 0.7,
        validationThreshold: 0.8,
        // Execution Parameters
        maxIterations: 5,
        timeoutSeconds: 120,
        maxRetries: 3,
        fallbackStrategy: 'default',
        parallelProcessing: false,
        maxParallelTasks: 3,
        // Knowledge Integration
        knowledgeBaseEnabled: false,
        knowledgeBaseSources: [],
        contextAnalysisEnabled: true,
        domainSpecificAnalysis: true,
        // MDAP/MAKER Integration
        mdapMakerEnabled: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_ENABLED,
        mdapMakerAutoSelect: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_AUTO_SELECT,
        mdapMakerMaxDepth: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_MAX_DEPTH,
        mdapMakerKAhead: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_K_AHEAD,
        mdapMakerRedFlagging: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_RED_FLAGGING,
        mdapMakerAdaptiveK: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_ADAPTIVE_K,
        mdapMakerProvider: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_PROVIDER,
        mdapMakerModel: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_MODEL,
        mdapMakerAutoSelectionKeywords: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_KEYWORDS,
    },
    defaultExecutionMethod: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_EXECUTION_METHOD,
    mdapMaker: {
        enabled: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_ENABLED,
        autoSelect: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_AUTO_SELECT,
        maxDepth: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_MAX_DEPTH,
        kAhead: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_K_AHEAD,
        redFlagging: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_RED_FLAGGING,
        adaptiveK: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_ADAPTIVE_K,
        provider: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_PROVIDER,
        model: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_MODEL,
        autoSelectionKeywords: exports.OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_KEYWORDS,
    },
};
//# sourceMappingURL=plugin-types.js.map