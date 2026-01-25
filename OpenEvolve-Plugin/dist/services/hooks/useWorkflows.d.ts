/**
 * Workflow operations hook
 */
export declare function useWorkflows(): {
    workflows: import('../../stores/workflowStore').WorkflowExecution[];
    isLoading: boolean;
    error: Error;
    startWorkflow: import('@tanstack/react-query').UseMutateAsyncFunction<{
        evolution_id: string;
        status: string;
        created_at: string;
        websocket_url: string;
    }, Error, {
        content: string;
        mode: "standard" | "quality_diversity" | "island_model";
        parameters: any;
        models: any[];
    }, unknown>;
    pauseWorkflow: import('@tanstack/react-query').UseMutateAsyncFunction<{
        evolution_id: string;
        status: string;
        paused_at: string;
    }, Error, string, unknown>;
    resumeWorkflow: import('@tanstack/react-query').UseMutateAsyncFunction<{
        evolution_id: string;
        status: string;
        resumed_at: string;
    }, Error, string, unknown>;
    stopWorkflow: import('@tanstack/react-query').UseMutateAsyncFunction<{
        evolution_id: string;
        status: string;
        stopped_at: string;
        final_results: any;
    }, Error, string, unknown>;
    deleteWorkflow: import('@tanstack/react-query').UseMutateAsyncFunction<string, Error, string, unknown>;
    isStarting: boolean;
    isPausing: boolean;
    isResuming: boolean;
    isStopping: boolean;
    isDeleting: boolean;
};
/**
 * Single workflow hook
 */
export declare function useWorkflow(evolutionId?: string): {
    workflow: import('../../stores/workflowStore').WorkflowExecution;
    isLoading: boolean;
    error: Error;
    pause: () => Promise<void>;
    resume: () => Promise<void>;
    stop: () => Promise<void>;
    remove: () => Promise<void>;
    refetch: () => Promise<void>;
};
/**
 * Workflow configuration hook
 */
export declare function useWorkflowConfig(): {
    config: import('../../stores/workflowStore').WorkflowConfig;
    updateConfig: (updates: Partial<import('../../stores/workflowStore').WorkflowConfig>) => void;
    reset: () => void;
};
/**
 * Workflow models hook
 */
export declare function useWorkflowModels(): {
    models: import('../../stores/workflowStore').ModelConfig[];
    updateModels: (newModels: import('../../stores/workflowStore').ModelConfig[]) => void;
    add: (model: any) => void;
    remove: (index: number) => void;
};
/**
 * Integrated workflow hook
 */
export declare function useIntegratedWorkflow(): {
    startWorkflow: import('@tanstack/react-query').UseMutateAsyncFunction<{
        workflow_id: string;
        status: string;
        current_stage: string;
        websocket_url: string;
    }, Error, {
        problem_statement: string;
        workflow_template?: string;
        parameters?: Record<string, any>;
    }, unknown>;
    getWorkflowStatus: (workflowId: string) => Promise<{
        workflow_id: string;
        status: string;
        current_stage: string;
        stages: Array<{
            stage: string;
            status: string;
            result?: any;
            progress?: number;
        }>;
    }>;
    isStarting: boolean;
};
