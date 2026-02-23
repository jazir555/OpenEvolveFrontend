/**
 * ROMA Service Layer
 *
 * This service provides business logic and additional functionality on top of the ROMA client.
 * It includes caching, retry logic, validation, and performance analysis.
 */
import { RomaService as RomaServiceInterface, RomaClient as RomaClientInterface, RomaExecutionResult, RomaExecutionOptions } from '../types/plugin-types';
/**
 * ROMA Service Implementation
 */
export declare class RomaService implements RomaServiceInterface {
    client: RomaClientInterface;
    private executionCache;
    private cacheTTL;
    /**
     * Create ROMA Service
     * @param client ROMA client instance
     */
    constructor(client: RomaClientInterface);
    /**
     * Initialize the service
     */
    initialize(): Promise<void>;
    /**
     * Execute task with retry logic
     * @param goal Task goal
     * @param options Execution options
     * @param retries Number of retries (default: 3)
     */
    executeTaskWithRetry(goal: string, options?: RomaExecutionOptions, retries?: number): Promise<RomaExecutionResult>;
    /**
     * Execute task with caching
     * @param goal Task goal
     * @param options Execution options
     */
    executeTaskWithCache(goal: string, options?: RomaExecutionOptions): Promise<RomaExecutionResult>;
    /**
     * Get cached execution result
     * @param goal Task goal
     */
    getCachedExecution(goal: string): RomaExecutionResult | undefined;
    /**
     * Cache execution result
     * @param goal Task goal
     * @param result Execution result
     */
    cacheExecutionResult(goal: string, result: RomaExecutionResult): void;
    /**
     * Clear cache
     */
    clearCache(): void;
    /**
     * Set cache TTL
     * @param ttl Cache time-to-live in milliseconds
     */
    setCacheTTL(ttl: number): void;
    /**
     * Validate execution result
     * @param result Execution result to validate
     */
    validateExecutionResult(result: RomaExecutionResult): boolean;
    /**
     * Format execution result for display
     * @param result Execution result
     */
    formatExecutionResult(result: RomaExecutionResult): string;
    /**
     * Get execution plan
     * Retrieves the detailed execution plan for a specific execution from ROMA API.
     * Includes subtasks, dependencies graph, and execution metadata.
     *
     * @param executionId - Execution ID to retrieve plan for
     * @returns Execution plan with subtasks and dependencies
     * @throws Error if execution not found or API call fails
     */
    getExecutionPlan(executionId: string): Promise<any>;
    /**
     * Analyze execution performance
     * Provides detailed performance analysis for a specific execution.
     * Includes timing metrics, tool usage, module efficiency, and overall score.
     *
     * @param executionId - Execution ID to analyze
     * @returns Performance metrics including timing, usage, and efficiency score
     * @throws Error if execution not found or API call fails
     */
    analyzeExecutionPerformance(executionId: string): Promise<Record<string, any>>;
    /**
     * Get cache statistics
     */
    getCacheStatistics(): {
        size: number;
        hitRate: number;
    };
    /**
     * Check if result is cached
     * @param goal Task goal
     */
    isResultCached(goal: string): boolean;
    /**
     * Get all cached execution goals
     */
    getCachedExecutionGoals(): string[];
}
//# sourceMappingURL=RomaService.d.ts.map