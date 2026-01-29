/**
 * ROMA BubbleLabs Plugin - TypeScript Interfaces
 * 
 * This file contains all TypeScript interfaces, types, and constants for the ROMA plugin.
 * The interfaces follow the same pattern as other BubbleLabs plugins (LeanAIDE, ClaudieMiro, Datapizza).
 */

// Import necessary types from React and other libraries
type ReactNode = any;
type ReactElement = any;
type Dispatch = any;
type SetStateAction = any;

/**
 * ROMA Plugin Metadata
 */
export interface RomaPluginMetadata {
  name: string;
  version: string;
  description: string;
  author: string;
  license: string;
  repository?: string;
  documentation?: string;
}

/**
 * ROMA Execution Status
 */
export type RomaExecutionStatus =
  | 'initializing'
  | 'idle'
  | 'configuring'
  | 'executing'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled';

/**
 * ROMA Module Types (Core ROMA Architecture)
 */
export type RomaModuleType =
  | 'atomizer'
  | 'planner'
  | 'executor'
  | 'aggregator'
  | 'verifier';

/**
 * ROMA Task Types (MECE Framework)
 */
export type RomaTaskType =
  | 'retrieve'
  | 'write'
  | 'think'
  | 'code_interpret'
  | 'image_generation';

/**
 * ROMA Prediction Strategies
 */
export type RomaPredictionStrategy =
  | 'predict'
  | 'chain_of_thought'
  | 'react'
  | 'code_act'
  | 'best_of_n'
  | 'refine'
  | 'parallel'
  | 'majority';

/**
 * ROMA MCP Server Configuration
 */
export interface RomaMcpServerConfig {
  server_name: string;
  server_type: 'http' | 'stdio';
  url?: string; // For HTTP servers
  command?: string; // For stdio servers
  args?: string[]; // For stdio servers
  headers?: Record<string, string>; // For HTTP servers
  env?: Record<string, string>; // Environment variables
  use_storage?: boolean;
  storage_threshold_kb?: number;
  enabled?: boolean;
}

/**
 * ROMA Toolkit Configuration
 */
export interface RomaToolkitConfig {
  class_name: string;
  enabled: boolean;
  toolkit_config?: Record<string, any>;
  include_tools?: string[];
  exclude_tools?: string[];
}

/**
 * ROMA Agent Configuration
 */
export interface RomaAgentConfig {
  llm?: {
    model?: string;
    temperature?: number;
    cache?: boolean;
    max_tokens?: number;
  };
  prediction_strategy?: RomaPredictionStrategy;
  toolkits?: RomaToolkitConfig[];
  context_defaults?: Record<string, any>;
}

/**
 * ROMA MDAP/MAKER Configuration
 */
export interface RomaMdapMakerConfig {
  enabled?: boolean;
  autoSelect?: boolean;
  maxDepth?: number;
  kAhead?: number;
  enableRedFlagging?: boolean;
  enableAdaptiveK?: boolean;
  provider?: string;
  model?: string;
  autoSelectionKeywords?: string[];
}

/**
 * ROMA Execution Method
 */
export type RomaExecutionMethod =
  | 'traditional'
  | 'claudiomiro'
  | 'datapizza'
  | 'roma'
  | 'hybrid'
  | 'roma_mdap_maker'
  | 'auto';

/**
 * ROMA Plugin Configuration
 */
export interface RomaPluginConfig {
  serverUrl?: string;
  apiKey?: string;
  defaultProfile?: string;
  maxDepth?: number;
  timeout?: number;
  cacheTTL?: number;
  enableObservability?: boolean;
  enableStorage?: boolean;
  storageBasePath?: string;
  defaultExecutionMethod?: RomaExecutionMethod;
  mdapMaker?: RomaMdapMakerConfig;
  agents?: {
    atomizer?: RomaAgentConfig;
    planner?: RomaAgentConfig;
    executor?: RomaAgentConfig;
    aggregator?: RomaAgentConfig;
    verifier?: RomaAgentConfig;
  };
  mcpServers?: RomaMcpServerConfig[];
  debugMode?: boolean;
}

/**
 * ROMA Execution Statistics
 */
export interface RomaExecutionStatistics {
  totalExecutions: number;
  successfulExecutions: number;
  failedExecutions: number;
  averageExecutionTime: number;
  totalExecutionTime: number;
  lastExecutionTime?: number;
  lastExecutionStatus?: RomaExecutionStatus;
}

/**
 * ROMA Execution Result
 */
export interface RomaExecutionResult {
  executionId: string;
  goal: string;
  status: RomaExecutionStatus;
  result?: any;
  error?: string;
  statistics?: {
    executionTime: number;
    subtasksCreated: number;
    subtasksCompleted: number;
    toolsUsed: string[];
    modulesUsed: RomaModuleType[];
  };
  timestamp: number;
}

/**
 * ROMA Plugin State
 */
export interface RomaPluginState extends RomaPluginConfig {
  status: RomaExecutionStatus;
  executionHistory: RomaExecutionResult[];
  statistics: RomaExecutionStatistics;
  currentExecution?: RomaExecutionResult;
  isInitialized: boolean;
  initializationError?: string;
}

/**
 * ROMA Execution Options
 */
export interface RomaExecutionOptions {
  maxDepth?: number;
  timeout?: number;
  profile?: string;
  useCache?: boolean;
  debug?: boolean;
  executionMethod?: RomaExecutionMethod;
  mdapMakerConfig?: Partial<RomaMdapMakerConfig>;
}

/**
 * ROMA Subtask
 */
export interface RomaSubtask {
  subtaskId: string;
  goal: string;
  taskType: RomaTaskType;
  dependencies?: string[];
  status: RomaExecutionStatus;
  result?: any;
  error?: string;
  executionTime?: number;
}

/**
 * ROMA Execution Plan
 */
export interface RomaExecutionPlan {
  executionId: string;
  originalGoal: string;
  subtasks: RomaSubtask[];
  dependenciesGraph?: Record<string, string[]>;
  createdAt: number;
  status: RomaExecutionStatus;
}

/**
 * ROMA Plugin Interface
 */
export interface RomaPlugin {
  metadata: RomaPluginMetadata;
  
  /**
   * Initialize the ROMA plugin
   * @param config Plugin configuration
   */
  initialize: (config?: Partial<RomaPluginConfig>) => Promise<void>;
  
  /**
   * Update plugin configuration
   * @param configUpdate Configuration updates
   */
  updateConfig: (configUpdate: Partial<RomaPluginConfig>) => Promise<void>;
  
  /**
   * Execute a task using ROMA
   * @param goal The task goal
   * @param options Execution options
   */
  executeTask: (goal: string, options?: RomaExecutionOptions) => Promise<RomaExecutionResult>;
  
  /**
   * Get current plugin state
   */
  getState: () => RomaPluginState;
  
  /**
   * Get execution history
   * @param limit Maximum number of results
   */
  getExecutionHistory: (limit?: number) => RomaExecutionResult[];
  
  /**
   * Get execution by ID
   * @param executionId Execution ID
   */
  getExecution: (executionId: string) => RomaExecutionResult | undefined;
  
  /**
   * Cancel current execution
   */
  cancelExecution: () => Promise<void>;
  
  /**
   * Clear execution history
   */
  clearHistory: () => Promise<void>;
  
  /**
   * Reset plugin state
   */
  reset: () => Promise<void>;
  
  /**
   * Get available MCP servers
   */
  getAvailableMcps: () => RomaMcpServerConfig[];
  
  /**
   * Add MCP server configuration
   * @param mcpConfig MCP server configuration
   */
  addMcpServer: (mcpConfig: RomaMcpServerConfig) => Promise<void>;
  
  /**
   * Remove MCP server
   * @param serverName Server name
   */
  removeMcpServer: (serverName: string) => Promise<void>;
  
  /**
   * Get available toolkits
   */
  getAvailableToolkits: () => RomaToolkitConfig[];
  
  /**
   * Add toolkit configuration
   * @param toolkitConfig Toolkit configuration
   */
  addToolkit: (toolkitConfig: RomaToolkitConfig) => Promise<void>;
  
  /**
   * Remove toolkit
   * @param toolkitName Toolkit name
   */
  removeToolkit: (toolkitName: string) => Promise<void>;
  
  /**
   * Get plugin statistics
   */
  getStatistics: () => RomaExecutionStatistics;
  
  /**
   * Export plugin state
   */
  exportState: () => RomaPluginState;
  
  /**
   * Import plugin state
   * @param state Plugin state to import
   */
  importState: (state: RomaPluginState) => Promise<void>;
  
  /**
   * Check if plugin is ready
   */
  isReady: () => boolean;
  
  /**
   * Get plugin version
   */
  getVersion: () => string;
  
  /**
   * Get plugin metadata
   */
  getMetadata: () => RomaPluginMetadata;
}

/**
 * ROMA Plugin Constants
 */
export const ROMA_PLUGIN_CONSTANTS = {
  DEFAULT_SERVER_URL: 'http://localhost:8000',
  DEFAULT_API_KEY: '',
  DEFAULT_PROFILE: 'general',
  DEFAULT_MAX_DEPTH: 3,
  DEFAULT_TIMEOUT: 30000,
  DEFAULT_CACHE_TTL: 3600000,
  DEFAULT_STORAGE_PATH: './roma-storage',
  DEFAULT_EXECUTION_METHOD: 'auto' as RomaExecutionMethod,
  DEFAULT_MDAP_MAKER_CONFIG: {
    enabled: true,
    autoSelect: true,
    maxDepth: 2,
    kAhead: 3,
    enableRedFlagging: true,
    enableAdaptiveK: true,
    provider: 'openai',
    model: 'gpt-4o-mini',
    autoSelectionKeywords: [
      'critical',
      'zero error',
      'flawless',
      'perfect',
      'mission-critical',
      'safety-critical',
      'high-reliability'
    ]
  } as RomaMdapMakerConfig,
  SUPPORTED_STRATEGIES: [
    'predict',
    'chain_of_thought',
    'react',
    'code_act',
    'best_of_n',
    'refine',
    'parallel',
    'majority'
  ] as const,
  SUPPORTED_MODULES: [
    'atomizer',
    'planner',
    'executor',
    'aggregator',
    'verifier'
  ] as const,
  SUPPORTED_TASK_TYPES: [
    'retrieve',
    'write',
    'think',
    'code_interpret',
    'image_generation'
  ] as const,
  SUPPORTED_EXECUTION_METHODS: [
    'traditional',
    'claudiomiro',
    'datapizza',
    'roma',
    'hybrid',
    'roma_mdap_maker',
    'auto'
  ] as const,
  DEFAULT_AGENT_CONFIG: {
    llm: {
      model: 'openrouter/google/gemini-2.5-flash',
      temperature: 0.6,
      cache: true
    },
    prediction_strategy: 'chain_of_thought' as RomaPredictionStrategy,
    toolkits: [],
    context_defaults: {}
  }
};

/**
 * ROMA Plugin Default Configuration
 */
export const DEFAULT_ROMA_CONFIG: RomaPluginConfig = {
  serverUrl: ROMA_PLUGIN_CONSTANTS.DEFAULT_SERVER_URL,
  apiKey: ROMA_PLUGIN_CONSTANTS.DEFAULT_API_KEY,
  defaultProfile: ROMA_PLUGIN_CONSTANTS.DEFAULT_PROFILE,
  maxDepth: ROMA_PLUGIN_CONSTANTS.DEFAULT_MAX_DEPTH,
  timeout: ROMA_PLUGIN_CONSTANTS.DEFAULT_TIMEOUT,
  cacheTTL: ROMA_PLUGIN_CONSTANTS.DEFAULT_CACHE_TTL,
  enableObservability: false,
  enableStorage: false,
  storageBasePath: ROMA_PLUGIN_CONSTANTS.DEFAULT_STORAGE_PATH,
  defaultExecutionMethod: ROMA_PLUGIN_CONSTANTS.DEFAULT_EXECUTION_METHOD,
  mdapMaker: ROMA_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_CONFIG,
  agents: {
    atomizer: ROMA_PLUGIN_CONSTANTS.DEFAULT_AGENT_CONFIG,
    planner: ROMA_PLUGIN_CONSTANTS.DEFAULT_AGENT_CONFIG,
    executor: ROMA_PLUGIN_CONSTANTS.DEFAULT_AGENT_CONFIG,
    aggregator: ROMA_PLUGIN_CONSTANTS.DEFAULT_AGENT_CONFIG,
    verifier: ROMA_PLUGIN_CONSTANTS.DEFAULT_AGENT_CONFIG
  },
  mcpServers: [],
  debugMode: false
};

/**
 * ROMA Plugin Error Types
 */
export class RomaPluginError extends Error {
  constructor(message: string, public code: string, public details?: any) {
    super(message);
    this.name = 'RomaPluginError';
  }
}

/**
 * ROMA Plugin Events
 */
export type RomaPluginEvent =
  | { type: 'initialized'; config: RomaPluginConfig }
  | { type: 'config_updated'; config: RomaPluginConfig }
  | { type: 'execution_started'; executionId: string; goal: string }
  | { type: 'execution_completed'; executionId: string; result: RomaExecutionResult }
  | { type: 'execution_failed'; executionId: string; error: string }
  | { type: 'execution_cancelled'; executionId: string }
  | { type: 'status_changed'; status: RomaExecutionStatus }
  | { type: 'error'; error: RomaPluginError };

/**
 * ROMA Plugin Event Handler
 */
export type RomaPluginEventHandler = (event: RomaPluginEvent) => void;

/**
 * ROMA Plugin React Props
 */
export interface RomaPluginProps {
  plugin: RomaPlugin;
  children?: ReactNode;
}

/**
 * ROMA Config Panel Props
 */
export interface RomaConfigPanelProps {
  plugin: RomaPlugin;
  onConfigChange?: (config: RomaPluginConfig) => void;
  onClose?: () => void;
}

/**
 * ROMA Execution Panel Props
 */
export interface RomaExecutionPanelProps {
  plugin: RomaPlugin;
  executionId?: string;
  onClose?: () => void;
}

/**
 * ROMA Status Indicator Props
 */
export interface RomaStatusIndicatorProps {
  status: RomaExecutionStatus;
  size?: 'sm' | 'md' | 'lg';
}

/**
 * ROMA Module Selector Props
 */
export interface RomaModuleSelectorProps {
  selectedModule: RomaModuleType;
  onModuleChange: (module: RomaModuleType) => void;
  disabled?: boolean;
}

/**
 * ROMA Strategy Selector Props
 */
export interface RomaStrategySelectorProps {
  selectedStrategy: RomaPredictionStrategy;
  onStrategyChange: (strategy: RomaPredictionStrategy) => void;
  disabled?: boolean;
}

/**
 * ROMA MCP Server Selector Props
 */
export interface RomaMcpServerSelectorProps {
  servers: RomaMcpServerConfig[];
  selectedServer?: string;
  onServerSelect: (serverName: string) => void;
  onAddServer: () => void;
  onRemoveServer: (serverName: string) => void;
}

/**
 * ROMA Toolkit Selector Props
 */
export interface RomaToolkitSelectorProps {
  toolkits: RomaToolkitConfig[];
  selectedToolkit?: string;
  onToolkitSelect: (toolkitName: string) => void;
  onAddToolkit: () => void;
  onRemoveToolkit: (toolkitName: string) => void;
}

/**
 * ROMA Execution History Props
 */
export interface RomaExecutionHistoryProps {
  executions: RomaExecutionResult[];
  onExecutionSelect: (executionId: string) => void;
  onClearHistory: () => void;
  limit?: number;
}

/**
 * ROMA Execution Detail Props
 */
export interface RomaExecutionDetailProps {
  execution: RomaExecutionResult;
  onClose: () => void;
}

/**
 * ROMA Statistics Display Props
 */
export interface RomaStatisticsDisplayProps {
  statistics: RomaExecutionStatistics;
  onReset: () => void;
}

/**
 * ROMA Plugin Context Type
 */
export interface RomaPluginContextType {
  plugin: RomaPlugin;
  config: RomaPluginConfig;
  state: RomaPluginState;
  updateConfig: (configUpdate: Partial<RomaPluginConfig>) => Promise<void>;
  executeTask: (goal: string, options?: RomaExecutionOptions) => Promise<RomaExecutionResult>;
  cancelExecution: () => Promise<void>;
  resetPlugin: () => Promise<void>;
}

/**
 * ROMA Plugin Hook Return Type
 */
export interface RomaPluginHookReturn {
  plugin: RomaPlugin;
  config: RomaPluginConfig;
  state: RomaPluginState;
  isReady: boolean;
  isExecuting: boolean;
  error?: string;
  executeTask: (goal: string, options?: RomaExecutionOptions) => Promise<RomaExecutionResult>;
  cancelExecution: () => Promise<void>;
  updateConfig: (configUpdate: Partial<RomaPluginConfig>) => Promise<void>;
  reset: () => Promise<void>;
}

/**
 * ROMA Client Configuration
 */
export interface RomaClientConfig {
  baseUrl: string;
  apiKey?: string;
  timeout?: number;
  headers?: Record<string, string>;
}

/**
 * ROMA Client Interface
 */
export interface RomaClient {
  config: RomaClientConfig;
  
  executeTask(goal: string, options?: RomaExecutionOptions): Promise<RomaExecutionResult>;
  getExecution(executionId: string): Promise<RomaExecutionResult>;
  getExecutionHistory(limit?: number): Promise<RomaExecutionResult[]>;
  cancelExecution(executionId: string): Promise<void>;
  getStatus(): Promise<{ status: RomaExecutionStatus }>;
  getStatistics(): Promise<RomaExecutionStatistics>;
  getAvailableMcps(): Promise<RomaMcpServerConfig[]>;
  addMcpServer(mcpConfig: RomaMcpServerConfig): Promise<void>;
  removeMcpServer(serverName: string): Promise<void>;
  getAvailableToolkits(): Promise<RomaToolkitConfig[]>;
  addToolkit(toolkitConfig: RomaToolkitConfig): Promise<void>;
  removeToolkit(toolkitName: string): Promise<void>;
  getProfiles(): Promise<string[]>;
  getProfileConfig(profileName: string): Promise<Partial<RomaPluginConfig>>;
}

/**
 * ROMA Service Interface
 */
export interface RomaService {
  client: RomaClient;
  
  executeTaskWithRetry(goal: string, options?: RomaExecutionOptions, retries?: number): Promise<RomaExecutionResult>;
  executeTaskWithCache(goal: string, options?: RomaExecutionOptions): Promise<RomaExecutionResult>;
  getCachedExecution(goal: string): RomaExecutionResult | undefined;
  clearCache(): void;
  validateExecutionResult(result: RomaExecutionResult): boolean;
  formatExecutionResult(result: RomaExecutionResult): string;
  getExecutionPlan(executionId: string): Promise<RomaExecutionPlan>;
  analyzeExecutionPerformance(executionId: string): Promise<Record<string, any>>;
}

/**
 * ROMA Plugin Factory Options
 */
export interface RomaPluginFactoryOptions {
  initialConfig?: Partial<RomaPluginConfig>;
  clientConfig?: Partial<RomaClientConfig>;
  debugMode?: boolean;
  autoInitialize?: boolean;
}

/**
 * ROMA Plugin Factory Return
 */
export interface RomaPluginFactoryReturn {
  plugin: RomaPlugin;
  client: RomaClient;
  service: RomaService;
  state: RomaPluginState;
  initialize: () => Promise<void>;
  destroy: () => Promise<void>;
}

// Export all types for easy import
export {
  RomaPluginMetadata,
  RomaExecutionStatus,
  RomaModuleType,
  RomaTaskType,
  RomaPredictionStrategy,
  RomaExecutionMethod,
  RomaMdapMakerConfig,
  RomaMcpServerConfig,
  RomaToolkitConfig,
  RomaAgentConfig,
  RomaPluginConfig,
  RomaExecutionStatistics,
  RomaExecutionResult,
  RomaPluginState,
  RomaExecutionOptions,
  RomaSubtask,
  RomaExecutionPlan,
  RomaPlugin,
  RomaPluginError,
  RomaPluginEvent,
  RomaPluginEventHandler,
  RomaPluginProps,
  RomaConfigPanelProps,
  RomaExecutionPanelProps,
  RomaStatusIndicatorProps,
  RomaModuleSelectorProps,
  RomaStrategySelectorProps,
  RomaMcpServerSelectorProps,
  RomaToolkitSelectorProps,
  RomaExecutionHistoryProps,
  RomaExecutionDetailProps,
  RomaStatisticsDisplayProps,
  RomaPluginContextType,
  RomaPluginHookReturn,
  RomaClientConfig,
  RomaClient,
  RomaService,
  RomaPluginFactoryOptions,
  RomaPluginFactoryReturn,
  ROMA_PLUGIN_CONSTANTS,
  DEFAULT_ROMA_CONFIG
};