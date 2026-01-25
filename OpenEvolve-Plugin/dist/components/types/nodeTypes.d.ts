import { Node } from '@xyflow/react';
/**
 * Node Status - Represents the current execution state
 */
export type NodeStatus = 'idle' | 'running' | 'completed' | 'error' | 'paused';
/**
 * Node Types - Different OpenEvolve node types
 */
export type NodeType = 'openevolve' | 'decomposition' | 'solution' | 'verification';
/**
 * Node Configuration - Generic configuration object
 */
export interface NodeConfig {
    [key: string]: any;
}
/**
 * Node Result - Execution result data
 */
export interface NodeResult {
    success?: boolean;
    score?: number;
    confidence?: number;
    iterations?: number;
    duration?: number;
    error?: string;
    [key: string]: any;
}
/**
 * Base OpenEvolve Node Data Interface
 * This is the core data structure for all OpenEvolve nodes
 */
export interface OpenEvolveNodeData extends Record<string, unknown> {
    id: string;
    type: NodeType;
    displayName: string;
    description?: string;
    status: NodeStatus;
    progress?: number;
    config?: NodeConfig;
    parameters?: Record<string, any>;
    results?: NodeResult;
    onParameterChange?: (name: string, value: any) => void;
    onExecute?: () => void;
}
/**
 * Full React Flow Node type for OpenEvolve
 */
export type OpenEvolveFlowNode = Node<OpenEvolveNodeData>;
/**
 * Extended types for specific nodes (re-exported for convenience)
 */
export interface BaseOpenEvolveNodeProps {
    data: OpenEvolveNodeData;
    selected?: boolean;
}
/**
 * Sub-problem type for Decomposition Node
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
 * Dependency information for Decomposition Node
 */
export interface DependencyInfo {
    totalDependencies: number;
    criticalPath: number;
    circularDeps: number;
}
/**
 * Alternative solution type for Solution Node
 */
export interface AlternativeSolution {
    id: string;
    name: string;
    score: number;
    confidence: number;
    strategy: string;
}
/**
 * Solution metrics for Solution Node
 */
export interface SolutionMetrics {
    executionTime: number;
    convergence: number;
    diversity: number;
    efficiency: number;
}
/**
 * Quality metrics for Verification Node
 */
export interface QualityMetrics {
    accuracy: number;
    completeness: number;
    consistency: number;
    performance: number;
    security: number;
}
/**
 * Requirement type for Verification Node
 */
export interface Requirement {
    id: string;
    name: string;
    status: 'pass' | 'fail' | 'warning' | 'skipped';
    description: string;
    category: string;
}
/**
 * Preset configurations for common node types
 */
export declare const NODE_PRESETS: {
    readonly decomposition: {
        readonly type: "decomposition";
        readonly status: "idle";
        readonly displayName: "Decomposition";
        readonly description: "Break down complex problems into manageable sub-problems";
    };
    readonly solution: {
        readonly type: "solution";
        readonly status: "idle";
        readonly displayName: "Solution Generator";
        readonly description: "Generate and optimize solutions using evolutionary algorithms";
    };
    readonly verification: {
        readonly type: "verification";
        readonly status: "idle";
        readonly displayName: "Verification";
        readonly description: "Validate solutions against requirements and quality metrics";
    };
};
/**
 * Helper function to create a new OpenEvolve node
 */
export declare function createOpenEvolveNode(type: NodeType, overrides?: Partial<OpenEvolveNodeData>): OpenEvolveNodeData;
/**
 * Helper function to create initial React Flow node
 */
export declare function createFlowNode(type: NodeType, position: {
    x: number;
    y: number;
}, data?: Partial<OpenEvolveNodeData>): OpenEvolveFlowNode;
