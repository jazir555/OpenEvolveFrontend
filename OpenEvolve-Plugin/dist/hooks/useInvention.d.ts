/**
 * End-to-End Invention Planner Hook
 *
 * Provides invention planning functionality
 *
 * @module hooks
 * @version 1.0.0
 */
export interface InventionRequest {
    goal: string;
    domain: 'technology' | 'hardware' | 'business' | 'process' | 'scientific' | 'creative';
    innovativeness: number;
    planningStages: string[];
    constraints?: string;
    targetAudience?: string;
    includePriorArt: boolean;
    includeFeasibility: boolean;
    includeRoadmap: boolean;
    detailLevel: 'overview' | 'detailed' | 'comprehensive';
}
export interface InventionResponse {
    plan: any;
    priorArt?: any;
    feasibility?: any;
    roadmap?: any;
    leanProofs?: any[];
    errorAnalysis: any;
    redTeamResults?: any;
    blueTeamResults?: any;
    successCriteria: any[];
    executionTime: number;
    qualityAssessment: {
        innovation: number;
        feasibility: number;
        clarity: number;
        completeness: number;
    };
}
export interface InventionStatus {
    isRunning: boolean;
    progress: number;
    currentStage: string;
    message: string;
}
/**
 * Invention planner hook
 */
export declare function useInvention(): {
    createPlan: (request: InventionRequest) => Promise<InventionResponse>;
    reset: () => void;
    status: InventionStatus;
    result: InventionResponse;
    error: Error;
};
