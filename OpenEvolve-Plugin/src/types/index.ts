// @ts-nocheck
/**
 * Type definitions for @openevolve/bubblelab-plugin
 *
 * This file contains all TypeScript interfaces and types used throughout
 * the OpenEvolve BubbleLab plugin.
 */

// ============================================================================
// CORE NODE TYPES
// ============================================================================

/**
 * Base configuration for all OpenEvolve nodes
 */
export interface OpenEvolveNodeConfig extends Record<string, unknown> {
  id?: string;
  name?: string;
  enabled?: boolean;
  debug?: boolean;
  timeout?: number;
}

/**
 * Base data structure for OpenEvolve nodes
 */
export interface OpenEvolveNodeData extends OpenEvolveNodeConfig {
  type: 'evolution' | 'adversarial' | 'decomposition' | 'knowledge' | 'leanaide' | 'crewai' | 'mdap' | 'maker' | 'researchQuest' | 'pyGraphistry';
  label: string;
}

// ============================================================================
// EVOLUTION TYPES
// ============================================================================

/**
 * Evolution node configuration
 */
export interface EvolutionConfig {
  generations: number;
  populationSize: number;
  mutationRate: number;
  crossoverRate: number;
  selectionMethod: 'tournament' | 'roulette' | 'rank' | 'steady-state';
  elitismCount: number;
  tournamentSize?: number;
  mutationStrategy?: 'gaussian' | 'uniform' | 'adaptive';
  crossoverStrategy?: 'single-point' | 'two-point' | 'uniform' | 'arithmetic';
}

/**
 * Evolution node data structure
 */
export interface EvolutionNodeData extends OpenEvolveNodeData {
  type: 'evolution';
  config: EvolutionConfig;
}

/**
 * Evolution result
 */
export interface EvolutionResult {
  success: boolean;
  generation: number;
  bestFitness: number;
  averageFitness: number;
  bestIndividual: any;
  population: any[];
  convergenceMetrics: {
    generationalDistance: number;
    spread: number;
    hyperVolume: number;
  };
  executionTime: number;
}

/**
 * Evolution strategy
 */
export type EvolutionStrategy =
  | 'generational'
  | 'steady-state'
  | 'island'
  | 'cellular'
  | 'coevolutionary';

// ============================================================================
// ADVERSARIAL TYPES
// ============================================================================

/**
 * Adversarial node configuration
 */
export interface AdversarialConfig {
  enabled: boolean;
  attackStrategy: AttackStrategy;
  numExamples: number;
  strength: number;
  stepSize: number;
  numSteps: number;
  targeted?: boolean;
  targetClass?: number;
  norm: 'L1' | 'L2' | 'Linf';
  defenseStrategies: string[];
}

/**
 * Adversarial node data structure
 */
export interface AdversarialNodeData extends OpenEvolveNodeData {
  type: 'adversarial';
  config: AdversarialConfig;
}

/**
 * Adversarial attack result
 */
export interface AdversarialResult {
  success: boolean;
  adversarialExamples: any[];
  attackSuccessRate: number;
  averagePerturbation: number;
  robustnessMetrics: {
    empiricalRobustness: number;
    certifiedRobustness?: number;
  };
  executionTime: number;
}

/**
 * Attack strategy types
 */
export type AttackStrategy =
  | 'fgsm'
  | 'pgd'
  | 'cw'
  | 'deepfool'
  | 'boundary'
  | 'spatial'
  | 'universal'
  | 'one-pixel';

// ============================================================================
// DECOMPOSITION TYPES
// ============================================================================

/**
 * Decomposition node configuration
 */
export interface DecompositionConfig {
  strategy: DecompositionStrategy;
  maxDepth: number;
  recursionDepthLimit?: number;
  pruningThreshold: number;
  granularity: 'coarse' | 'medium' | 'fine';
  parallelDecomposition: boolean;
  maxSubtasks?: number;
  maxSubProblems?: number;
  dependencyAnalysis: boolean;
  constraintPropagation: boolean;
}

/**
 * Decomposition node data structure
 */
export interface DecompositionNodeData extends OpenEvolveNodeData {
  type: 'decomposition';
  config: DecompositionConfig;

  // Additional properties for DecompositionNodeComponent
  subProblems?: SubProblem[];
  dependencyGraph?: DependencyInfo;
  qualityScore?: number;
  complexity?: number;
  completeness?: number;
}

/**
 * Sub-problem in decomposition
 */
export interface SubProblem {
  id: string;
  title: string;
  description: string;
  status: 'pending' | 'in_progress' | 'completed' | 'blocked';
  complexity: number;
  dependencies: string[];
}

/**
 * Dependency information
 */
export interface DependencyInfo {
  totalDependencies: number;
  criticalPath: number;
  circularDeps: number;
}

/**
 * Solution node data structure
 */
export interface SolutionNodeData extends OpenEvolveNodeData {
  type: 'solution';
  config?: SolutionConfig;

  // Additional properties for SolutionNodeComponent
  currentStrategy?: string;
  availableStrategies?: string[];
  qualityScore?: number;
  confidence?: number;
  iterations?: number;
  alternativeSolutions?: AlternativeSolution[];
  metrics?: SolutionMetrics;
}

/**
 * Alternative solution
 */
export interface AlternativeSolution {
  id: string;
  name: string;
  score: number;
  confidence: number;
  strategy: string;
}

/**
 * Solution metrics
 */
export interface SolutionMetrics {
  executionTime: number;
  convergence: number;
  qualityScore: number;
  resourceUsage: number;
  diversity?: number;
  efficiency?: number;
}

/**
 * Verification node data structure
 */
export interface VerificationNodeData extends OpenEvolveNodeData {
  type: 'verification';
  config?: VerificationConfig;

  // Additional properties for VerificationNodeComponent
  verificationStatus?: 'pass' | 'fail' | 'warning' | 'pending';
  verificationScore?: number;
  qualityMetrics?: VerificationQualityMetrics;
  requirements?: VerificationRequirement[];
  checksPerformed?: number;
  checksPassed?: number;
  checksFailed?: number;
}

/**
 * Verification quality metrics
 */
export interface VerificationQualityMetrics {
  accuracy: number;
  completeness: number;
  consistency: number;
  performance: number;
  security: number;
}

/**
 * Verification requirement
 */
export interface VerificationRequirement {
  id: string;
  name: string;
  status: 'pass' | 'fail' | 'warning' | 'skipped';
  description: string;
  category: string;
}

/**
 * Decomposition result
 */
export interface DecompositionResult {
  success: boolean;
  decomposition: DecompositionTree;
  numSubtasks: number;
  maxDepth: number;
  executionTime: number;
  qualityMetrics: {
    coherence: number;
    completeness: number;
    independence: number;
  };
}

/**
 * Decomposition tree structure
 */
export interface DecompositionTree {
  id: string;
  task: string;
  description?: string;
  children?: DecompositionTree[];
  dependencies?: string[];
  constraints?: string[];
  metadata?: {
    complexity?: number;
    estimatedTime?: number;
    resources?: string[];
  };
}

/**
 * Decomposition strategy
 */
export type DecompositionStrategy =
  | 'hierarchical'
  | 'flat'
  | 'adaptive'
  | 'goal-oriented'
  | 'constraint-based'
  | 'knowledge-guided';

// ============================================================================
// INTEGRATION TYPES
// ============================================================================

/**
 * Main integration configuration
 */
export interface IntegrationConfig {
  knowledgeEngine: KnowledgeEngineConfig;
  leanaide: LeanAIDEConfig;
  crewai: CrewAIConfig;
}

/**
 * Knowledge engine configuration
 */
export interface KnowledgeEngineConfig {
  enabled: boolean;
  endpoint: string;
  timeout: number;
  maxRetries: number;
  cacheEnabled: boolean;
  cacheTTL: number;
  graphType: 'neo4j' | 'networkx' | 'custom';
}

/**
 * LeanAIDE configuration
 */
export interface LeanAIDEConfig {
  enabled: boolean;
  endpoint: string;
  timeout: number;
  maxRetries: number;
  formalizationStrategy: 'automatic' | 'interactive' | 'hybrid';
  verificationEnabled: boolean;
  leanVersion: string;
}

/**
 * CrewAI configuration
 */
export interface CrewAIConfig {
  enabled: boolean;
  endpoint: string;
  timeout: number;
  maxRetries: number;
  delegationStrategy: 'automatic' | 'manual' | 'hybrid';
  orchestrationMode: 'centralized' | 'distributed';
  maxConcurrentTasks: number;
}

/**
 * Integration result
 */
export interface IntegrationResult {
  success: boolean;
  data?: any;
  error?: string;
  metadata?: {
    executionTime: number;
    endpoint: string;
    timestamp: number;
  };
}

// ============================================================================
// PLUGIN TYPES
// ============================================================================

/**
 * Main plugin interface
 */
export interface OpenEvolvePlugin {
  id: string;
  version: string;
  config: OpenEvolveNodeConfig;

  // Lifecycle methods
  initialize(): Promise<void>;
  destroy(): void;

  // Configuration methods
  updateConfig(config: Partial<OpenEvolveNodeConfig>): void;
  getConfig(): OpenEvolveNodeConfig;
  validateConfig(config: OpenEvolveNodeConfig): boolean;

  // State management
  getState(): PluginState;
  subscribe(listener: (state: PluginState) => void): () => void;

  // Actions
  actions: PluginActions;
}

/**
 * Plugin context for nodes
 */
export interface PluginContext {
  plugin: OpenEvolvePlugin;
  nodeId: string;
  nodeData: OpenEvolveNodeData;
}

/**
 * Plugin state
 */
export interface PluginState {
  initialized: boolean;
  activeNodes: string[];
  config: OpenEvolveNodeConfig;
  integrations: {
    knowledgeEngine: boolean;
    leanaide: boolean;
    crewai: boolean;
  };
  statistics: {
    totalEvolutions: number;
    totalAdversarialAttacks: number;
    totalDecompositions: number;
    successfulExecutions: number;
    failedExecutions: number;
    averageExecutionTime: number;
  };
}

/**
 * Plugin actions
 */
export interface PluginActions {
  // Evolution actions
  runEvolution: (config: EvolutionConfig) => Promise<EvolutionResult>;

  // Adversarial actions
  runAdversarial: (config: AdversarialConfig) => Promise<AdversarialResult>;

  // Decomposition actions
  runDecomposition: (config: DecompositionConfig) => Promise<DecompositionResult>;

  // Integration actions
  queryKnowledgeEngine: (query: any) => Promise<IntegrationResult>;
  runLeanAIDE: (task: any) => Promise<IntegrationResult>;
  delegateToCrewAI: (task: any) => Promise<IntegrationResult>;

  // Utility actions
  resetStatistics: () => void;
  exportState: () => string;
  importState: (state: string) => void;
}

// ============================================================================
// NODE TYPES FOR BUBBLELAB
// ============================================================================

/**
 * Node connection types
 */
export interface NodeConnection {
  id: string;
  sourceNodeId: string;
  targetNodeId: string;
  sourceHandle?: string;
  targetHandle?: string;
  type?: 'default' | 'evolution' | 'adversarial' | 'decomposition';
}

/**
 * Node position
 */
export interface NodePosition {
  x: number;
  y: number;
}

/**
 * Complete node definition
 */
export interface NodeDefinition {
  id: string;
  type: string;
  position: NodePosition;
  data: OpenEvolveNodeData;
}

// ============================================================================
// UTILITY TYPES
// ============================================================================

/**
 * Deep partial type for nested objects
 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/**
 * Event types for plugin
 */
export type PluginEvent =
  | { type: 'initialized'; timestamp: number }
  | { type: 'config-updated'; config: OpenEvolveNodeConfig; timestamp: number }
  | { type: 'evolution-started'; nodeId: string; timestamp: number }
  | { type: 'evolution-completed'; nodeId: string; result: EvolutionResult; timestamp: number }
  | { type: 'adversarial-started'; nodeId: string; timestamp: number }
  | { type: 'adversarial-completed'; nodeId: string; result: AdversarialResult; timestamp: number }
  | { type: 'decomposition-started'; nodeId: string; timestamp: number }
  | { type: 'decomposition-completed'; nodeId: string; result: DecompositionResult; timestamp: number }
  | { type: 'error'; nodeId: string; error: Error; timestamp: number };

/**
 * Event listener
 */
export type EventListener = (event: PluginEvent) => void;

// ============================================================================
// PYGRAPHISTRY TYPES
// ============================================================================

/**
 * PyGraphistry visualization layout types
 */
export type PyGraphistryLayout = 'force_directed' | 'circular' | 'hierarchical';

/**
 * PyGraphistry clustering methods
 */
export type PyGraphistryClusteringMethod = 'dbscan' | 'kmeans';

/**
 * Graph node interface
 */
export interface GraphNode {
  id: string;
  label?: string;
  type?: string;
  [key: string]: string | number | boolean | undefined;
}

/**
 * Graph edge interface
 */
export interface GraphEdge {
  source: string;
  target: string;
  type?: string;
  weight?: number;
  [key: string]: string | number | boolean | undefined;
}

/**
 * PyGraphistry node configuration
 */
export interface PyGraphistryNodeConfig extends NodeConfig {
  layout?: PyGraphistryLayout;
  clustering?: boolean;
  clusteringMethod?: PyGraphistryClusteringMethod;
  enableGPUAcceleration?: boolean;
  apiKey?: string;
  serverUrl?: string;
  enableBackendExecution?: boolean;
  backendUrl?: string;
}

/**
 * PyGraphistry node data structure
 */
export interface PyGraphistryNodeData extends OpenEvolveNodeData {
  type: 'pyGraphistry';
  config: PyGraphistryNodeConfig;
  nodes?: GraphNode[];
  edges?: GraphEdge[];
  visualizationUrl?: string;
  layout?: PyGraphistryLayout;
  clustering?: boolean;
}

/**
 * PyGraphistry result
 */
export interface PyGraphistryResult {
  success: boolean;
  visualizationUrl?: string;
  message: string;
  metadata: {
    executionTime: number;
    backendUsed: boolean;
    nodesProcessed: number;
    edgesProcessed: number;
    layoutUsed: PyGraphistryLayout;
    clusteringApplied: boolean;
    note?: string;
  };
  error?: string;
}

// ============================================================================
// RESEARCH QUEST TYPES
// ============================================================================

/**
 * Research Quest node configuration
 */
export interface ResearchQuestConfig {
  enableMultiLayer?: boolean;
  maxHypotheses?: number;
  enableCausalInference?: boolean;
  enableTemporalAnalysis?: boolean;
  enableBiasAssessment?: boolean;
  enableFalsificationChecks?: boolean;
  enableImpactScoring?: boolean;
  enableInterdisciplinaryBridges?: boolean;
  enableKnowledgeGapDetection?: boolean;
  enableProbabilisticConfidence?: boolean;
  enableGraphRestructuring?: boolean;
  enableTopologyAnalysis?: boolean;
  enableInformationTheoryMetrics?: boolean;
  enableAttributionTracking?: boolean;
  enableStatisticalPowerAnalysis?: boolean;
  enableMultiScaleAnalysis?: boolean;
  enableCostEstimation?: boolean;
  enableSelfAudit?: boolean;
  enableEvidenceIntegration?: boolean;
  enablePruningMerging?: boolean;
  enableSubgraphExtraction?: boolean;
  enableReflection?: boolean;
  enableComposition?: boolean;
  enableHypothesisGeneration?: boolean;
  enableTaskDecomposition?: boolean;
  enableInitialization?: boolean;
  enableBackendExecution?: boolean;
  backendUrl?: string;
  parameters?: ResearchQuestParameters;
}

/**
 * Research Quest parameters (P1.0-P1.29)
 */
export interface ResearchQuestParameters {
  P1_0?: boolean;
  P1_1?: boolean;
  P1_2?: boolean;
  P1_3?: boolean;
  P1_4?: boolean;
  P1_5?: boolean;
  P1_6?: boolean;
  P1_7?: boolean;
  P1_8?: boolean;
  P1_9?: boolean;
  P1_10?: boolean;
  P1_11?: boolean;
  P1_12?: boolean;
  P1_13?: boolean;
  P1_14?: boolean;
  P1_15?: boolean;
  P1_16?: boolean;
  P1_17?: boolean;
  P1_18?: boolean;
  P1_19?: boolean;
  P1_20?: boolean;
  P1_21?: boolean;
  P1_22?: boolean;
  P1_23?: boolean;
  P1_24?: boolean;
  P1_25?: boolean;
  P1_26?: boolean;
  P1_27?: boolean;
  P1_28?: boolean;
  P1_29?: boolean;
}

/**
 * Research Quest node data structure
 */
export interface ResearchQuestNodeData extends OpenEvolveNodeData {
  type: 'researchQuest';
  config: ResearchQuestConfig;
  stage?: 'initialization' | 'decomposition' | 'hypothesis_planning' | 'evidence_integration' | 'pruning_merging' | 'subgraph_extraction' | 'composition' | 'reflection';
  taskDescription?: string;
  graphSummary?: any;
  reasoningTrace?: any;
  topologyInsights?: any;
}

/**
 * Research Quest result
 */
export interface ResearchQuestResult {
  success: boolean;
  node_id?: string;
  message: string;
  current_stage: number;
  stage_name: string;
  dimension_nodes?: string[];
  hypothesis_nodes?: string[];
  active_parameters?: string[];
  warnings?: string[];
  errors?: string[];
  graph_summary?: any;
  reasoning_trace?: any;
  topology_insights?: any;
  export_data?: string;
  partial_success?: boolean;
  recovery_attempted?: boolean;
  metadata?: {
    executionTime: number;
    stageExecuted: string;
    parametersEnabled?: ResearchQuestParameters;
    backendUsed: boolean;
    note?: string;
  };
}
