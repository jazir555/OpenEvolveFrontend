/**
 * Mock OpenEvolve Adapter for Testing
 *
 * Provides a mock implementation of the OpenEvolve adapter interface
 * for testing purposes without requiring actual OpenEvolve service.
 */
export interface MockOpenEvolveConfig {
    mockOptimized?: {
        solution: any;
        fitness: number;
    };
    mockGenerations?: number;
    mockPopulationSize?: number;
    mockConverged?: boolean;
    mockTimeout?: boolean;
    mockError?: boolean;
}
export interface WorkflowDefinition {
    workflow_id: string;
    name: string;
    description: string;
    problem_statement: string;
    max_refinement_loops: number;
    auto_approval_enabled: boolean;
    sub_problems: Array<{
        id: string;
        description: string;
        dependencies: string[];
        solver_team_name: string;
        gold_team_gauntlet_name: string;
    }>;
}
export interface WorkflowResponse {
    workflow_id: string;
    status: string;
    created_at: string;
}
export interface WorkflowState {
    status: 'running' | 'completed' | 'failed';
    final_solution?: {
        quality_metrics?: {
            score: number;
        };
    };
}
export interface OptimizationResult {
    best_fitness: number;
    generations: number;
    population_size: number;
    final_population: Array<{
        solution: any;
        fitness: number;
    }>;
    optimization_history: Array<{
        generation: number;
        best_fitness: number;
    }>;
}
/**
 * Create a mock OpenEvolve adapter
 */
export declare function createMockOpenEvolveAdapter(config?: MockOpenEvolveConfig): {
    /**
     * Create an optimization workflow
     */
    createWorkflow(workflow: WorkflowDefinition): Promise<WorkflowResponse>;
    /**
     * Get workflow status
     */
    getWorkflowStatus(workflowId: string): Promise<WorkflowState>;
    /**
     * Optimize a solution using evolutionary algorithms
     */
    optimize(params: {
        solution: any;
        generations?: number;
        population_size?: number;
        mutation_rate?: number;
        crossover_rate?: number;
    }): Promise<OptimizationResult>;
    /**
     * Validate a formula against constraints
     */
    validate(formula: any, constraints: any): Promise<{
        valid: boolean;
        satisfied: boolean;
        errors?: string[];
    }>;
};
/**
 * Create a mock OpenEvolve adapter with realistic delays
 */
export declare function createRealisticMockOpenEvolveAdapter(config?: MockOpenEvolveConfig): {
    /**
     * Create an optimization workflow
     */
    createWorkflow(workflow: WorkflowDefinition): Promise<WorkflowResponse>;
    /**
     * Get workflow status
     */
    getWorkflowStatus(workflowId: string): Promise<WorkflowState>;
    /**
     * Optimize a solution using evolutionary algorithms
     */
    optimize(params: {
        solution: any;
        generations?: number;
        population_size?: number;
        mutation_rate?: number;
        crossover_rate?: number;
    }): Promise<OptimizationResult>;
    /**
     * Validate a formula against constraints
     */
    validate(formula: any, constraints: any): Promise<{
        valid: boolean;
        satisfied: boolean;
        errors?: string[];
    }>;
};
//# sourceMappingURL=openevolve-mock.d.ts.map