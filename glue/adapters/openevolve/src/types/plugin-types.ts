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

// Import necessary types from React and other libraries
type ReactNode = any;
type ReactElement = any;
type Dispatch = any;
type SetStateAction = any;

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
export type OpenEvolveExecutionStatus =
  | 'initializing'
  | 'idle'
  | 'configuring'
  | 'executing'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled';

/**
 * OpenEvolve Module Types (Core OpenEvolve Architecture)
 */
export type OpenEvolveModuleType =
  | 'evolution'
  | 'adversarial'
  | 'decomposition'
  | 'integration';

/**
 * Evolution Strategy Types
 */
export type EvolutionStrategy =
  | 'standard'
  | 'genetic_algorithm'
  | 'quality_diversity'
  | 'novelty_search'
  | 'multi_objective'
  | 'adaptive'
  | 'hybrid';

/**
 * Adversarial Strategy Types
 */
export type AdversarialStrategy =
  | 'red_blue_team'
  | 'multi_agent'
  | 'self_play'
  | 'co_evolution'
  | 'competitive'
  | 'cooperative';

/**
 * Decomposition Strategy Types
 */
export type DecompositionStrategy =
  | 'semantic'
  | 'hierarchical'
  | 'functional'
  | 'modular'
  | 'temporal'
  | 'hybrid';

/**
 * Evolution Configuration Interface
 * Comprehensive configuration for evolutionary algorithms and optimization
 */
export interface EvolutionConfig {
  // Core Evolution Parameters
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

  // Model Configuration Parameters
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

  // Quality Diversity Parameters
  featureDimensions: string[] | null;
  featureBins: number;
  archiveSize: number;
  noveltyThreshold: number;

  // Evolutionary MCTS Parameters
  mctsEnabled: boolean;
  mctsIterations: number;
  explorationWeight: number;
  rolloutDepth: number;
  treeReuse: boolean;

  // MDAP/MAKER Integration
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
  // Core Adversarial Parameters
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

  // Team Configuration
  redTeamAggressiveness: number;
  blueTeamCreativity: number;
  evaluatorRigor: number;
  teamDiversity: number;
  collaborationLevel: number;

  // Execution Parameters
  parallelExecution: boolean;
  maxParallelTasks: number;
  timeoutSeconds: number;
  maxRetries: number;
  fallbackStrategy: string;

  // Quality Metrics
  qualityThreshold: number;
  improvementThreshold: number;
  acceptanceThreshold: number;
  metricWeights: Record<string, number>;

  // MDAP/MAKER Integration
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
  // Core Decomposition Parameters
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

  // Strategy-Specific Parameters
  semanticAnalysisEnabled: boolean;
  hierarchicalDepth: number;
  functionalGranularity: string;
  modularIndependence: number;
  temporalGranularity: string;

  // Quality Parameters
  qualityThreshold: number;
  completenessThreshold: number;
  clarityThreshold: number;
  feasibilityThreshold: number;
  validationThreshold: number;

  // Execution Parameters
  maxIterations: number;
  timeoutSeconds: number;
  maxRetries: number;
  fallbackStrategy: string;
  parallelProcessing: boolean;
  maxParallelTasks: number;

  // Knowledge Integration
  knowledgeBaseEnabled: boolean;
  knowledgeBaseSources: string[];
  contextAnalysisEnabled: boolean;
  domainSpecificAnalysis: boolean;

  // MDAP/MAKER Integration
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
  // Metadata and Initialization
  getMetadata(): OpenEvolvePluginMetadata;
  getState(): OpenEvolvePluginState;
  initialize(config?: Partial<OpenEvolvePluginState>): Promise<void>;

  // Configuration Management
  updateConfig(config: Partial<OpenEvolvePluginState>): Promise<void>;
  resetConfig(): Promise<void>;
  getConfig(): OpenEvolvePluginState;

  // Evolution Functionality
  executeEvolution(
    goal: string,
    options?: OpenEvolveExecutionOptions
  ): Promise<OpenEvolveExecutionResult>;

  // Adversarial Functionality
  executeAdversarial(
    content: string,
    options?: OpenEvolveExecutionOptions
  ): Promise<OpenEvolveExecutionResult>;

  // Decomposition Functionality
  executeDecomposition(
    problem: string,
    options?: OpenEvolveExecutionOptions
  ): Promise<OpenEvolveExecutionResult>;

  // Integrated Execution
  executeIntegrated(
    goal: string,
    options?: OpenEvolveExecutionOptions
  ): Promise<OpenEvolveExecutionResult>;

  // Execution Management
  getExecution(executionId: string): Promise<OpenEvolveExecutionResult | null>;
  getExecutionHistory(): Promise<OpenEvolveExecutionResult[]>;
  getStatistics(): Promise<OpenEvolveExecutionStatistics[]>;
  cancelExecution(executionId: string): Promise<boolean>;
  clearHistory(): Promise<void>;

  // MDAP/MAKER Integration
  shouldUseMdapMakerForGoal(goal: string): boolean;
  getMdapMakerConfig(): any | null;

  // Utility Methods
  validateConfig(): Promise<{ valid: boolean; errors: string[] }>;
  getAvailableStrategies(): {
    evolution: EvolutionStrategy[];
    adversarial: AdversarialStrategy[];
    decomposition: DecompositionStrategy[];
  };
}

/**
 * OpenEvolve Plugin Constants
 */
export const OPENEVOLVE_PLUGIN_CONSTANTS = {
  PLUGIN_NAME: 'OpenEvolve BubbleLabs Plugin',
  PLUGIN_VERSION: '1.0.0',
  PLUGIN_DESCRIPTION: 'Comprehensive OpenEvolve system integration for BubbleLabs',
  PLUGIN_AUTHOR: 'OpenEvolve Team',
  PLUGIN_LICENSE: 'MIT',

  // Default Configuration Values
  DEFAULT_EVOLUTION_MODE: 'standard' as EvolutionStrategy,
  DEFAULT_ADVERSARIAL_MODE: 'red_blue_team' as AdversarialStrategy,
  DEFAULT_DECOMPOSITION_STRATEGY: 'semantic' as DecompositionStrategy,
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
export const DEFAULT_OPENEVOLVE_CONFIG: OpenEvolvePluginState = {
  initialized: false,
  metadata: {
    name: OPENEVOLVE_PLUGIN_CONSTANTS.PLUGIN_NAME,
    version: OPENEVOLVE_PLUGIN_CONSTANTS.PLUGIN_VERSION,
    description: OPENEVOLVE_PLUGIN_CONSTANTS.PLUGIN_DESCRIPTION,
    author: OPENEVOLVE_PLUGIN_CONSTANTS.PLUGIN_AUTHOR,
    license: OPENEVOLVE_PLUGIN_CONSTANTS.PLUGIN_LICENSE,
  },
  status: 'idle',
  currentExecutionId: null,
  executionHistory: [],
  statistics: [],
  evolutionConfig: {
    // Core Evolution Parameters
    evolutionMode: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_EVOLUTION_MODE,
    maxIterations: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MAX_ITERATIONS,
    populationSize: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_POPULATION_SIZE,
    temperature: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_TEMPERATURE,
    maxTokens: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MAX_TOKENS,
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
    apiBase: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_API_BASE,
    modelId: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MODEL_ID,
    backupModels: null,
    timeout: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_TIMEOUT,
    maxRetries: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MAX_RETRIES,
    retryDelay: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_RETRY_DELAY,
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
    mdapMakerEnabled: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_ENABLED,
    mdapMakerAutoSelect: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_AUTO_SELECT,
    mdapMakerMaxDepth: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_MAX_DEPTH,
    mdapMakerKAhead: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_K_AHEAD,
    mdapMakerRedFlagging: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_RED_FLAGGING,
    mdapMakerAdaptiveK: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_ADAPTIVE_K,
    mdapMakerProvider: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_PROVIDER,
    mdapMakerModel: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_MODEL,
    mdapMakerAutoSelectionKeywords: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_KEYWORDS,
  },
  adversarialConfig: {
    // Core Adversarial Parameters
    adversarialMode: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_ADVERSARIAL_MODE,
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
    mdapMakerEnabled: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_ENABLED,
    mdapMakerAutoSelect: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_AUTO_SELECT,
    mdapMakerMaxDepth: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_MAX_DEPTH,
    mdapMakerKAhead: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_K_AHEAD,
    mdapMakerRedFlagging: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_RED_FLAGGING,
    mdapMakerAdaptiveK: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_ADAPTIVE_K,
    mdapMakerProvider: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_PROVIDER,
    mdapMakerModel: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_MODEL,
    mdapMakerAutoSelectionKeywords: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_KEYWORDS,
  },
  decompositionConfig: {
    // Core Decomposition Parameters
    decompositionStrategy: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_DECOMPOSITION_STRATEGY,
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
    mdapMakerEnabled: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_ENABLED,
    mdapMakerAutoSelect: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_AUTO_SELECT,
    mdapMakerMaxDepth: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_MAX_DEPTH,
    mdapMakerKAhead: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_K_AHEAD,
    mdapMakerRedFlagging: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_RED_FLAGGING,
    mdapMakerAdaptiveK: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_ADAPTIVE_K,
    mdapMakerProvider: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_PROVIDER,
    mdapMakerModel: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_MODEL,
    mdapMakerAutoSelectionKeywords: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_KEYWORDS,
  },
  defaultExecutionMethod: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_EXECUTION_METHOD,
  mdapMaker: {
    enabled: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_ENABLED,
    autoSelect: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_AUTO_SELECT,
    maxDepth: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_MAX_DEPTH,
    kAhead: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_K_AHEAD,
    redFlagging: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_RED_FLAGGING,
    adaptiveK: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_ADAPTIVE_K,
    provider: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_PROVIDER,
    model: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_MODEL,
    autoSelectionKeywords: OPENEVOLVE_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_KEYWORDS,
  },
};
