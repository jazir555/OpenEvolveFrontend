/**
 * Workflow status types
 */
export type WorkflowStatus = 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'stopped';
/**
 * Workflow configuration
 */
export interface WorkflowConfig {
    mode: 'standard' | 'quality_diversity' | 'island_model';
    max_iterations: number;
    population_size: number;
    temperature: number;
    top_p: number;
    mutation_rate: number;
    crossover_rate: number;
}
/**
 * Model configuration
 */
export interface ModelConfig {
    provider: string;
    model: string;
    api_key?: string;
    api_base?: string;
}
/**
 * Individual in population
 */
export interface Individual {
    id: string;
    fitness: number;
    content: string;
    generation: number;
}
/**
 * Evolution progress
 */
export interface EvolutionProgress {
    current_iteration: number;
    max_iterations: number;
    percentage: number;
}
/**
 * Workflow execution state
 */
export interface WorkflowExecution {
    evolution_id: string;
    status: WorkflowStatus;
    progress: EvolutionProgress;
    population: Individual[];
    best_individual: Individual | null;
    metrics: {
        average_fitness: number;
        diversity_score: number;
        convergence_rate: number;
    };
    started_at: string;
    updated_at: string;
    websocket_url?: string;
}
/**
 * Workflow state interface
 */
interface WorkflowState {
    currentWorkflow: WorkflowExecution | null;
    workflows: WorkflowExecution[];
    config: WorkflowConfig;
    models: ModelConfig[];
    initialContent: string;
    isLoading: boolean;
    error: string | null;
    activeTab: string;
    setCurrentWorkflow: (workflow: WorkflowExecution | null) => void;
    setWorkflows: (workflows: WorkflowExecution[]) => void;
    addWorkflow: (workflow: WorkflowExecution) => void;
    updateWorkflow: (id: string, updates: Partial<WorkflowExecution>) => void;
    removeWorkflow: (id: string) => void;
    setConfig: (config: Partial<WorkflowConfig>) => void;
    resetConfig: () => void;
    setModels: (models: ModelConfig[]) => void;
    addModel: (model: ModelConfig) => void;
    removeModel: (index: number) => void;
    setInitialContent: (content: string) => void;
    setLoading: (loading: boolean) => void;
    setError: (error: string | null) => void;
    setActiveTab: (tab: string) => void;
    clearCurrentWorkflow: () => void;
    reset: () => void;
}
/**
 * Workflow store
 */
export declare const useWorkflowStore: import('zustand').UseBoundStore<Omit<Omit<import('zustand').StoreApi<WorkflowState>, "setState"> & {
    setState<A extends string | {
        type: string;
    }>(partial: WorkflowState | Partial<WorkflowState> | ((state: WorkflowState) => WorkflowState | Partial<WorkflowState>), replace?: boolean, action?: A): void;
}, "persist"> & {
    persist: {
        setOptions: (options: Partial<import('zustand/middleware').PersistOptions<WorkflowState, {
            config: WorkflowConfig;
            models: ModelConfig[];
            activeTab: string;
        }>>) => void;
        clearStorage: () => void;
        rehydrate: () => Promise<void> | void;
        hasHydrated: () => boolean;
        onHydrate: (fn: (state: WorkflowState) => void) => () => void;
        onFinishHydration: (fn: (state: WorkflowState) => void) => () => void;
        getOptions: () => Partial<import('zustand/middleware').PersistOptions<WorkflowState, {
            config: WorkflowConfig;
            models: ModelConfig[];
            activeTab: string;
        }>>;
    };
}>;
export {};
