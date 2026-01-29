/**
 * ROMA (Reasoning and Multi-Agent) Hook
 *
 * Provides ROMA reasoning functionality
 *
 * @module hooks
 * @version 1.0.0
 */
export interface ROMARequest {
    task: string;
    reasoningMode: 'collaborative' | 'adversarial' | 'debate' | 'consensus' | 'hierarchical';
    agentCount: number;
    agentRoles: string[];
    rounds: number;
    confidenceThreshold: number;
    includeReasoningTrace: boolean;
    enableVoting: boolean;
}
export interface ROMAResponse {
    solution: string;
    confidence: number;
    rounds: number;
    reasoningTrace?: any[];
    agentVotes?: any[];
    subtasks?: any[];
    consensusReached: boolean;
    qualityMetrics: {
        coherence: number;
        completeness: number;
        validity: number;
    };
    executionTime: number;
}
export interface ROMAStatus {
    isRunning: boolean;
    progress: number;
    currentRound: number;
    message: string;
}
/**
 * ROMA hook
 */
export declare function useROMA(): {
    execute: (request: ROMARequest) => Promise<ROMAResponse>;
    reset: () => void;
    status: ROMAStatus;
    result: ROMAResponse;
    error: Error;
};
