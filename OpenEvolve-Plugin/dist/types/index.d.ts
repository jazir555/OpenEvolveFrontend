/**
 * Type definitions for @openevolve/bubblelab-plugin
 *
 * This file contains all TypeScript interfaces and types used throughout
 * the OpenEvolve BubbleLab plugin.
 */
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
    type: 'evolution' | 'adversarial' | 'decomposition' | 'knowledge' | 'leanaide' | 'hephaestus' | 'mdap' | 'maker';
    label: string;
}
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
export type EvolutionStrategy = 'generational' | 'steady-state' | 'island' | 'cellular' | 'coevolutionary';
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
export type AttackStrategy = 'fgsm' | 'pgd' | 'cw' | 'deepfool' | 'boundary' | 'spatial' | 'universal' | 'one-pixel';
/**
 * Decomposition node configuration
 */
export interface DecompositionConfig {
    strategy: DecompositionStrategy;
    maxDepth: number;
    pruningThreshold: number;
    granularity: 'coarse' | 'medium' | 'fine';
    parallelDecomposition: boolean;
    maxSubtasks?: number;
    dependencyAnalysis: boolean;
    constraintPropagation: boolean;
}
/**
 * Decomposition node data structure
 */
export interface DecompositionNodeData extends OpenEvolveNodeData {
    type: 'decomposition';
    config: DecompositionConfig;
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
export type DecompositionStrategy = 'hierarchical' | 'flat' | 'adaptive' | 'goal-oriented' | 'constraint-based' | 'knowledge-guided';
/**
 * Main integration configuration
 */
export interface IntegrationConfig {
    knowledgeEngine: KnowledgeEngineConfig;
    leanaide: LeanAIDEConfig;
    hephaestus: HephaestusConfig;
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
 * Hephaestus configuration
 */
export interface HephaestusConfig {
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
/**
 * Main plugin interface
 */
export interface OpenEvolvePlugin {
    id: string;
    version: string;
    config: OpenEvolveNodeConfig;
    initialize(): Promise<void>;
    destroy(): void;
    updateConfig(config: Partial<OpenEvolveNodeConfig>): void;
    getConfig(): OpenEvolveNodeConfig;
    validateConfig(config: OpenEvolveNodeConfig): boolean;
    getState(): PluginState;
    subscribe(listener: (state: PluginState) => void): () => void;
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
        hephaestus: boolean;
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
    runEvolution: (config: EvolutionConfig) => Promise<EvolutionResult>;
    runAdversarial: (config: AdversarialConfig) => Promise<AdversarialResult>;
    runDecomposition: (config: DecompositionConfig) => Promise<DecompositionResult>;
    queryKnowledgeEngine: (query: any) => Promise<IntegrationResult>;
    runLeanAIDE: (task: any) => Promise<IntegrationResult>;
    delegateToHephaestus: (task: any) => Promise<IntegrationResult>;
    resetStatistics: () => void;
    exportState: () => string;
    importState: (state: string) => void;
}
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
/**
 * Deep partial type for nested objects
 */
export type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};
/**
 * Event types for plugin
 */
export type PluginEvent = {
    type: 'initialized';
    timestamp: number;
} | {
    type: 'config-updated';
    config: OpenEvolveNodeConfig;
    timestamp: number;
} | {
    type: 'evolution-started';
    nodeId: string;
    timestamp: number;
} | {
    type: 'evolution-completed';
    nodeId: string;
    result: EvolutionResult;
    timestamp: number;
} | {
    type: 'adversarial-started';
    nodeId: string;
    timestamp: number;
} | {
    type: 'adversarial-completed';
    nodeId: string;
    result: AdversarialResult;
    timestamp: number;
} | {
    type: 'decomposition-started';
    nodeId: string;
    timestamp: number;
} | {
    type: 'decomposition-completed';
    nodeId: string;
    result: DecompositionResult;
    timestamp: number;
} | {
    type: 'error';
    nodeId: string;
    error: Error;
    timestamp: number;
};
/**
 * Event listener
 */
export type EventListener = (event: PluginEvent) => void;
