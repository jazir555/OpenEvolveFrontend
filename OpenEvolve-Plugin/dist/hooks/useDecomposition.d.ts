/**
 * Decomposition parameters
 */
export interface DecompositionParams {
    problem_statement: string;
    decomposition_strategy?: 'hierarchical' | 'functional' | 'data_flow' | 'hybrid';
    max_depth?: number;
    include_dependencies?: boolean;
    include_subtasks?: boolean;
}
/**
 * Decomposition result
 */
export interface DecompositionResult {
    decomposition_id: string;
    problem_statement: string;
    decomposition_tree: DecompositionNode;
    tasks: DecompositionTask[];
    dependencies: Dependency[];
    execution_plan: ExecutionStep[];
    metadata: {
        total_tasks: number;
        estimated_duration: number;
        complexity_score: number;
        created_at: string;
    };
}
/**
 * Decomposition node in tree
 */
export interface DecompositionNode {
    id: string;
    title: string;
    description: string;
    level: number;
    children: DecompositionNode[];
    metadata?: Record<string, any>;
}
/**
 * Individual task
 */
export interface DecompositionTask {
    task_id: string;
    title: string;
    description: string;
    parent_id?: string;
    dependencies: string[];
    status: 'pending' | 'in_progress' | 'completed' | 'blocked';
    estimated_effort: number;
    priority: 'low' | 'medium' | 'high' | 'critical';
    assignee?: string;
}
/**
 * Task dependency
 */
export interface Dependency {
    from: string;
    to: string;
    type: 'sequential' | 'parallel' | 'conditional';
}
/**
 * Execution step
 */
export interface ExecutionStep {
    step_id: string;
    task_id: string;
    order: number;
    can_start_in_parallel: boolean;
}
/**
 * Decomposition state
 */
export interface DecompositionState {
    data: DecompositionResult | null;
    loading: boolean;
    error: Error | null;
    progress: number;
}
/**
 * Custom hook for problem decomposition
 * Breaks down complex problems into manageable subtasks
 */
export declare function useDecomposition(decompositionId?: string): {
    execute: (params: DecompositionParams) => Promise<DecompositionResult | null>;
    getStatus: () => Promise<DecompositionResult | null>;
    getResults: () => DecompositionResult | null;
    cancel: () => Promise<void>;
    updateTaskStatus: (taskId: string, status: DecompositionTask["status"]) => Promise<void>;
    getExecutionPlan: () => ExecutionStep[];
    getTaskById: (taskId: string) => DecompositionTask | null;
    getTasksByStatus: (status: DecompositionTask["status"]) => DecompositionTask[];
    reset: () => void;
    data: DecompositionResult | null;
    loading: boolean;
    error: Error | null;
    progress: number;
};
/**
 * Decomposition history hook
 */
export declare function useDecompositionHistory(params?: {
    limit?: number;
    offset?: number;
}): {
    refetch: () => Promise<void>;
    data: DecompositionResult[] | null;
    loading: boolean;
    error: Error | null;
};
/**
 * Decomposition templates hook
 */
export declare function useDecompositionTemplates(): {
    refetch: () => Promise<void>;
    data: Array<{
        id: string;
        name: string;
        description: string;
        strategy: string;
    }> | null;
    loading: boolean;
    error: Error | null;
};
