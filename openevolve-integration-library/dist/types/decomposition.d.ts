import { ExecutionConfig } from './common';
export interface DecompositionInputs {
    operation: 'decompose' | 'subproblems' | 'dependencies';
    input: DecompositionInput | SubProblemsInput | DependencyInput;
    config?: ExecutionConfig;
}
export interface DecompositionInput {
    problem: string;
    strategy: DecompositionStrategy;
    options?: DecompositionOptions;
}
export type DecompositionStrategy = 'hierarchical' | 'temporal' | 'functional' | 'data-flow' | 'goal-oriented' | 'constraint-based' | 'hybrid';
export interface DecompositionOptions {
    maxDepth?: number;
    granularity?: 'coarse' | 'medium' | 'fine';
    preserveDependencies?: boolean;
}
export interface DecompositionConstraint {
    type: 'ordering' | 'resource' | 'precedence' | 'custom';
    description: string;
    parameters?: Record<string, any>;
}
export interface SubProblemsInput {
    planId: string;
    filter?: {
        depth?: number;
        status?: 'pending' | 'in-progress' | 'completed' | 'blocked';
        priority?: 'low' | 'medium' | 'high' | 'critical';
    };
}
export interface DependencyInput {
    planId: string;
    options?: {
        transitive?: boolean;
    };
}
export interface DecompositionResult {
    planId: string;
    problem: string;
    strategy: DecompositionStrategy;
}
export interface SubProblem {
    id: string;
    parentId: string | null;
    title: string;
    description: string;
    depth: number;
    priority: 'low' | 'medium' | 'high' | 'critical';
    complexity: number;
    estimatedEffort?: {
        hours: number;
        confidence: {
            min: number;
            max: number;
        };
    };
    dependencies: string[];
    inputs?: string[];
    outputs?: string[];
    status?: 'pending' | 'in-progress' | 'completed' | 'blocked';
    assignedTo?: string;
    tags?: string[];
    metadata?: Record<string, any>;
}
export interface DependencyGraph {
    nodes: string[];
    edges: DependencyEdge[];
}
export interface DependencyEdge {
    from: string;
    to: string;
    type: 'data' | 'control' | 'resource';
    strength: number;
}
export interface ExecutionOrder {
    level: number;
    subProblems: string[];
}
export interface DecompositionMetadata {
    created: string;
    decompositionTime: number;
}
export interface SubProblemsListResult {
    planId: string;
    subProblems: SubProblem[];
    total: number;
    filter?: any;
}
export interface DependencyGraphResult {
    planId: string;
    graph: DependencyGraph;
}
export interface GraphMetrics {
    nodeCount: number;
    edgeCount: number;
}
export interface DecompositionExecutionResult {
    type: 'decompose' | 'subproblems' | 'dependencies';
    result: DecompositionResult | SubProblemsListResult | DependencyGraphResult;
    metadata: {
        executionTime: number;
        timestamp: string;
        apiVersion: string;
    };
}
//# sourceMappingURL=decomposition.d.ts.map