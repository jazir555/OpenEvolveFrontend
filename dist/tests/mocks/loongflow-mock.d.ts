/**
 * Mock LoongFlow Adapter for Testing
 *
 * Provides a mock implementation of the LoongFlow adapter interface
 * for testing purposes without requiring actual LoongFlow service.
 */
import { LoongFlowSolution } from '../../glue/schemas/loongflow-canonical';
export interface MockLoongFlowConfig {
    mockSolution?: Partial<LoongFlowSolution>;
    mockBestSolutions?: LoongFlowSolution[];
    mockLowConfidence?: boolean;
    mockTimeout?: boolean;
    mockError?: boolean;
}
export interface AgentSubmitResponse {
    agent_id: string;
    status: string;
    submitted_at: string;
}
export interface AgentState {
    status: 'idle' | 'planning' | 'executing' | 'summarizing' | 'evolving' | 'completed' | 'failed';
    current_iteration: number;
    best_score: number;
    total_cost: number;
}
export interface ExecutionResult {
    final_solution: string;
    final_score: number;
    was_interrupted: boolean;
    total_iterations: number;
    best_solutions?: LoongFlowSolution[];
    start_time: string;
    end_time: string;
}
/**
 * Create a mock LoongFlow adapter
 */
export declare function createMockLoongFlowAdapter(config?: MockLoongFlowConfig): {
    /**
     * Submit a problem for execution
     */
    submitProblem(params: {
        task: string;
        max_iterations?: number;
        target_score?: number;
        concurrency?: number;
        metadata?: Record<string, any>;
    }): Promise<AgentSubmitResponse>;
    /**
     * Get current agent state
     */
    getAgentState(agentId: string): Promise<AgentState>;
    /**
     * Get execution result
     */
    getExecutionResult(agentId: string): Promise<ExecutionResult>;
    /**
     * Get best solutions from evolutionary database
     */
    getBestSolutions(islandId?: number, topK?: number): Promise<LoongFlowSolution[]>;
    /**
     * Get specific solution by ID
     */
    getSolution(solutionId: string): Promise<LoongFlowSolution | null>;
};
/**
 * Create a mock LoongFlow adapter with realistic delays
 */
export declare function createRealisticMockLoongFlowAdapter(config?: MockLoongFlowConfig): {
    /**
     * Submit a problem for execution
     */
    submitProblem(params: {
        task: string;
        max_iterations?: number;
        target_score?: number;
        concurrency?: number;
        metadata?: Record<string, any>;
    }): Promise<AgentSubmitResponse>;
    /**
     * Get current agent state
     */
    getAgentState(agentId: string): Promise<AgentState>;
    /**
     * Get execution result
     */
    getExecutionResult(agentId: string): Promise<ExecutionResult>;
    /**
     * Get best solutions from evolutionary database
     */
    getBestSolutions(islandId?: number, topK?: number): Promise<LoongFlowSolution[]>;
    /**
     * Get specific solution by ID
     */
    getSolution(solutionId: string): Promise<LoongFlowSolution | null>;
};
//# sourceMappingURL=loongflow-mock.d.ts.map