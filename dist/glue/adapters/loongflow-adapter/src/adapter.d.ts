/**
 * LoongFlow Adapter
 *
 * This adapter integrates the LoongFlow PES (Plan-Execute-Summary) evolutionary
 * AI framework into the OpenEvolve federation.
 *
 * Architecture:
 * - LoongFlow is a Python library, not an HTTP API
 * - This adapter communicates with a Python sidecar service via HTTP
 * - The sidecar runs LoongFlow and exposes REST endpoints
 *
 * Environment Variables (Law of Configuration Explicitness):
 *   LOONGFLOW_API_URL - Base URL of LoongFlow sidecar (required)
 *   LOONGFLOW_TIMEOUT_MS - Request timeout in ms (default: 30000)
 *   LOONGFLOW_MAX_RETRIES - Max retry attempts (default: 3)
 *   LOG_LEVEL - Logging level (default: info)
 *
 * Following Federation Constitution:
 * - Law of Air Gap: No imports from core-projects/LoongFlow
 * - Law of Runtime Truth: All operations verified via probes
 * - Law of Idempotency: All operations safe to retry
 * - Law of UTC: All timestamps in UTC ISO-8601
 * - Law of Configuration Explicitness: Required env vars crash service
 * - Observability: Structured JSON logging with correlation_id
 */
/**
 * PES Agent configuration
 */
export interface PESAgentConfig {
    task: string;
    max_iterations: number;
    target_score: number;
    concurrency: number;
    workspace_path?: string;
    initial_code?: string;
    initial_score?: number;
    initial_evaluation?: string;
    checkpoint_path?: string;
    metadata?: Record<string, any>;
}
/**
 * PES Agent state
 */
export interface PESAgentState {
    agent_id: string;
    status: 'idle' | 'running' | 'interrupted' | 'completed' | 'failed';
    current_iteration: number;
    max_iterations: number;
    target_score: number;
    best_score: number;
    start_time: string;
    end_time?: string;
    completion_count: number;
    total_prompt_tokens: number;
    total_completion_tokens: number;
    total_cost: number;
}
/**
 * Solution from LoongFlow evolutionary database
 */
export interface Solution {
    solution_id: string;
    solution: string;
    evaluation: string;
    score: number;
    parent_id?: string;
    island_id: number;
    iteration: number;
    generate_plan: string;
    summary: string;
    created_at: string;
}
/**
 * Evolutionary database status
 */
export interface DatabaseStatus {
    global_status: {
        current_iteration: number;
        best_score: number;
        total_solutions: number;
    };
    island_status?: Record<number, {
        best_score: number;
        total_solutions: number;
    }>;
}
/**
 * Checkpoint information
 */
export interface CheckpointInfo {
    checkpoint_path: string;
    tag: string;
    created_at: string;
    iteration: number;
    completion_count: number;
}
/**
 * Problem submission request
 */
export interface SubmitProblemRequest {
    task: string;
    max_iterations?: number;
    target_score?: number;
    concurrency?: number;
    initial_code?: string;
    initial_score?: number;
    initial_evaluation?: string;
    workspace_path?: string;
    metadata?: Record<string, any>;
}
/**
 * Problem submission response
 */
export interface SubmitProblemResponse {
    agent_id: string;
    status: string;
    message: string;
}
/**
 * Execution result
 */
export interface ExecutionResult {
    agent_id: string;
    status: string;
    final_solution?: string;
    final_score?: number;
    best_solutions?: Solution[];
    total_iterations: number;
    total_tokens: number;
    total_cost: number;
    was_interrupted: boolean;
    start_time: string;
    end_time: string;
}
export interface LoongFlowAdapterConfig {
    api_url: string;
    timeout_ms: number;
    max_retries?: number;
    log_level?: string;
    circuit_breaker?: Partial<{
        threshold: number;
        timeout_ms: number;
        reset_timeout_ms: number;
    }>;
}
export declare class LoongFlowAdapter {
    private readonly api;
    private readonly logger;
    private readonly circuitBreaker;
    private readonly correlationId;
    constructor(config: LoongFlowAdapterConfig);
    /**
     * Check if LoongFlow sidecar is healthy
     */
    healthCheck(): Promise<{
        status: string;
        timestamp: string;
        version?: string;
    }>;
    /**
     * Submit a problem to the PES Agent for evolution
     * This is idempotent - submitting the same task_id will return the existing agent
     */
    submitProblem(request: SubmitProblemRequest): Promise<SubmitProblemResponse>;
    /**
     * Get the current state of a PES Agent
     */
    getAgentState(agentId: string): Promise<PESAgentState>;
    /**
     * Interrupt a running PES Agent
     * This is idempotent - interrupting an already stopped agent is a no-op
     */
    interruptAgent(agentId: string): Promise<{
        message: string;
    }>;
    /**
     * Get the final execution result of a PES Agent
     */
    getExecutionResult(agentId: string): Promise<ExecutionResult>;
    /**
     * Sample a solution from the evolutionary database
     */
    sampleSolution(islandId?: number): Promise<Solution | {}>;
    /**
     * Add a solution to the evolutionary database
     * This is idempotent if solution_id is the same
     */
    addSolution(solution: Omit<Solution, 'solution_id' | 'created_at'>): Promise<string>;
    /**
     * Update a solution in the evolutionary database
     */
    updateSolution(solutionId: string, updates: Partial<Solution>): Promise<string>;
    /**
     * Get the best solutions from the evolutionary database
     */
    getBestSolutions(islandId?: number, topK?: number): Promise<Solution[]>;
    /**
     * Get database status
     */
    getDatabaseStatus(islandId?: number): Promise<DatabaseStatus>;
    /**
     * Save a checkpoint of the current evolutionary state
     */
    saveCheckpoint(checkpointPath: string, tag: string): Promise<CheckpointInfo>;
    /**
     * Load a checkpoint
     */
    loadCheckpoint(checkpointPath: string): Promise<{
        message: string;
    }>;
    /**
     * List available checkpoints
     */
    listCheckpoints(checkpointPath: string): Promise<CheckpointInfo[]>;
    /**
     * Get circuit breaker state (for monitoring)
     */
    getCircuitBreakerState(): import("../../../lib/circuit-breaker").CircuitBreakerStats;
    /**
     * Manually reset circuit breaker (for recovery)
     */
    resetCircuitBreaker(): void;
}
export declare function createLoongFlowAdapter(config: LoongFlowAdapterConfig): LoongFlowAdapter;
export default LoongFlowAdapter;
//# sourceMappingURL=adapter.d.ts.map