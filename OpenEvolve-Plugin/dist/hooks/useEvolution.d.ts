import { WorkflowExecution } from '../stores';
/**
 * Evolution parameters
 */
export interface EvolutionParams {
    content: string;
    mode: 'standard' | 'quality_diversity' | 'island_model';
    parameters: {
        max_iterations: number;
        population_size: number;
        temperature: number;
        top_p: number;
    };
    models: Array<{
        provider: string;
        model: string;
        api_key: string;
    }>;
}
/**
 * Evolution state
 */
export interface EvolutionState {
    data: WorkflowExecution | null;
    loading: boolean;
    error: Error | null;
    progress: number;
}
/**
 * Custom hook for genetic algorithm evolution
 * Manages evolutionary optimization workflows with real-time updates
 */
export declare function useEvolution(evolutionId?: string): {
    execute: (params: EvolutionParams) => Promise<void>;
    getStatus: () => Promise<WorkflowExecution | null>;
    getResults: () => WorkflowExecution | null;
    cancel: () => Promise<void>;
    pause: () => Promise<void>;
    resume: () => Promise<void>;
    reset: () => void;
    data: WorkflowExecution | null;
    loading: boolean;
    error: Error | null;
    progress: number;
};
/**
 * Evolution list hook
 */
export declare function useEvolutions(params?: {
    status?: string;
    limit?: number;
    offset?: number;
}): {
    refetch: () => Promise<void>;
    data: WorkflowExecution[] | null;
    loading: boolean;
    error: Error | null;
};
