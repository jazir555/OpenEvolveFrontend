import { OpenEvolveBaseNode, NodeInputs, NodeResult, ExecutionContext, ValidationError, ParameterSchema } from './OpenEvolveBaseNode';
/**
 * Decomposition strategy types
 */
export type DecompositionStrategy = 'semantic' | 'dependency' | 'complexity' | 'hybrid' | 'research';
/**
 * Sub-problem interface
 */
export interface SubProblem {
    id: string;
    title: string;
    description: string;
    complexity: number;
    estimated_time: number;
    dependencies: string[];
    success_criteria: any[];
    type: string;
    status: string;
}
/**
 * Dependency graph interface
 */
export interface DependencyGraph {
    nodes: string[];
    edges: any[];
    execution_order: string[];
}
/**
 * Decomposition node configuration
 */
export interface DecompositionNodeConfig {
    strategy?: DecompositionStrategy;
    maxSubProblems?: number;
    qualityThreshold?: number;
    enableBackendExecution?: boolean;
    backendUrl?: string;
}
/**
 * Decomposition result interface
 */
export interface DecompositionResult {
    sub_problems: SubProblem[];
    decomposition_tree: DependencyGraph;
    complexity_metrics: {
        overall_score: number;
        meets_thresholds: boolean;
        confidence: number;
    };
    estimated_time: number;
    method_used: string;
    total_sub_problems: number;
    confidence: number;
    validation_checkpoints: number;
    plan_id: string;
    problem_id: string;
}
/**
 * Problem Decomposition Node (Integration Library Version)
 *
 * This node uses the OpenEvolve Integration Library to delegate decomposition
 * to the Python backend. The Python backend uses the existing DecompositionEngine
 * from decomposition_engine.py.
 *
 * Benefits of this approach:
 * - Reuses existing Python implementation
 * - No need to duplicate logic in TypeScript
 * - Consistent behavior across all clients
 * - Easy to update Python backend without changing frontend
 */
export declare class DecompositionNode extends OpenEvolveBaseNode {
    static readonly DISPLAY_NAME = "Problem Decomposition";
    static readonly DESCRIPTION = "Break down complex problems using Python DecompositionEngine via integration library";
    static readonly ICON = "decomposition";
    static readonly CATEGORY = "analysis";
    static readonly VERSION = "2.0.0";
    private client;
    constructor(id: string, config?: DecompositionNodeConfig);
    /**
     * Execute problem decomposition using the integration library
     *
     * @param inputs - Must contain 'problem_statement' string
     * @param context - Execution context
     * @returns Promise resolving to decomposition result
     */
    execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult>;
    /**
     * Execute decomposition using Python backend via integration library
     */
    private executeWithBackend;
    /**
     * Execute decomposition locally (fallback/simplified version)
     * This is used when backend is unavailable or for testing
     */
    private executeLocally;
    /**
     * Assess priority from complexity score
     */
    private assessPriorityFromComplexity;
    /**
     * Validate input data
     *
     * @param inputs - Input data to validate
     * @returns Array of validation errors
     */
    validateInputs(inputs: NodeInputs): ValidationError[];
    /**
     * Get JSON Schema for configuration parameters
     *
     * @returns Parameter schema
     */
    getParameterSchema(): ParameterSchema;
    /**
     * Cleanup when node is destroyed
     */
    destroy(): void;
}
export default DecompositionNode;
