/**
 * useAnalytics Hook
 * Stub implementation - TODO: Implement actual analytics functionality
 */
export interface AnalyticsData {
    totalEvolutions: number;
    evolutionsChange: number;
    successRate: number;
    avgExecutionTime: number;
    successRateChange?: number;
    avgDuration?: number;
    durationChange?: number;
    activeWorkflows?: number;
    workflowsChange?: number;
    performanceOverTime?: Array<{
        timestamp: string;
        value: number;
    }>;
    artifacts?: Array<any>;
}
export declare function useAnalytics(dateRange: string): import('@tanstack/react-query').UseQueryResult<AnalyticsData, Error>;
