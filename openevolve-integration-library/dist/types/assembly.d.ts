import { ExecutionConfig } from './common';
export interface AssemblyInputs {
    operation: 'assemble' | 'integrate' | 'optimize';
    input: AssemblyInput | IntegrationInput | OptimizationInput;
    config?: ExecutionConfig;
}
export interface AssemblyInput {
    solutions: Solution[];
    strategy: AssemblyStrategy;
    options?: AssemblyOptions;
}
export interface Solution {
    id: string;
    type: string;
    data: any;
    metadata: SolutionMetadata;
    dependencies?: string[];
    priority?: number;
}
export interface SolutionMetadata {
    source: string;
    created: string;
    version: string;
}
export type AssemblyStrategy = 'sequential' | 'parallel' | 'hierarchical' | 'priority-based' | 'dependency-driven' | 'custom';
export interface AssemblyOptions {
    mergeStrategy?: 'override' | 'merge' | 'concat' | 'custom';
    conflictResolution?: 'first' | 'last' | 'highest-priority' | 'manual';
    optimize?: boolean;
    validate?: boolean;
    generateDocs?: boolean;
    mergeRules?: MergeRule[];
}
export interface MergeRule {
    pattern: string;
    strategy: 'override' | 'merge' | 'concat' | 'transform';
    transform?: string;
}
export interface IntegrationInput {
    assembledSolution: any;
    targetSystem: {
        type: string;
        configuration: Record<string, any>;
        endpoints?: string[];
    };
    options?: {
        test?: boolean;
        deploy?: boolean;
        rollback?: boolean;
    };
}
export interface OptimizationInput {
    solution: any;
    objectives: OptimizationObjective[];
    constraints?: OptimizationConstraint[];
    options?: {
        algorithm?: 'gradient-descent' | 'genetic' | 'simulated-annealing' | 'custom';
        maxIterations?: number;
        targetScore?: number;
    };
}
export interface OptimizationObjective {
    name: string;
    type: 'maximize' | 'minimize' | 'target';
    target?: number;
    weight: number;
    metric: string;
}
export interface OptimizationConstraint {
    name: string;
    type: 'equality' | 'inequality' | 'boundary';
    value: number;
    parameter: string;
}
export interface AssemblyResult {
    assembledSolution: any;
    status: 'success' | 'partial' | 'failed';
}
export interface AssemblyStatistics {
    solutionsAssembled: number;
    conflictsResolved: number;
    assemblyTime: number;
}
export interface AssemblyConflict {
    type: 'naming' | 'dependency' | 'logic' | 'custom';
    description: string;
}
export interface AssemblyMetadata {
    timestamp: string;
    strategy: AssemblyStrategy;
}
export interface IntegrationResult {
    integratedSolution: any;
    status: 'success' | 'failed' | 'partial';
    tests?: IntegrationTest[];
    metadata: IntegrationMetadata;
}
export interface IntegrationTest {
    name: string;
    status: 'passed' | 'failed' | 'skipped';
    result?: any;
    error?: string;
    executionTime: number;
}
export interface IntegrationMetadata {
    timestamp: string;
    targetSystem: string;
    integrationTime: number;
    deploymentStatus?: 'deployed' | 'pending' | 'failed';
}
export interface OptimizationResult {
    optimizedSolution: any;
    status: 'success' | 'failed' | 'converged';
    improvement: {
        before: number;
        after: number;
        percentage: number;
    };
    iterations: number;
    convergence?: number[];
    metadata: OptimizationMetadata;
}
export interface OptimizationMetadata {
    timestamp: string;
    algorithm: string;
    optimizationTime: number;
    objectivesAchieved: string[];
}
export interface AssemblyExecutionResult {
    type: 'assemble' | 'integrate' | 'optimize';
    result: AssemblyResult | IntegrationResult | OptimizationResult;
    metadata: {
        executionTime: number;
        timestamp: string;
        apiVersion: string;
    };
}
//# sourceMappingURL=assembly.d.ts.map