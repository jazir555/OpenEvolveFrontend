/**
 * ROMA Service Layer
 * 
 * This service provides business logic and additional functionality on top of the ROMA client.
 * It includes caching, retry logic, validation, and performance analysis.
 */

import { RomaService, RomaClient, RomaExecutionResult, RomaExecutionOptions } from '../types/plugin-types';

/**
 * ROMA Service Implementation
 */
export class RomaService implements RomaService {
  public client: RomaClient;
  private executionCache: Map<string, RomaExecutionResult>;
  private cacheTTL: number;

  /**
   * Create ROMA Service
   * @param client ROMA client instance
   */
  constructor(client: RomaClient) {
    this.client = client;
    this.executionCache = new Map();
    this.cacheTTL = 3600000; // 1 hour default
  }

  /**
   * Initialize the service
   */
  public async initialize(): Promise<void> {
    // Service initialization logic
    console.log('ROMA service initialized');
  }

  /**
   * Execute task with retry logic
   * @param goal Task goal
   * @param options Execution options
   * @param retries Number of retries (default: 3)
   */
  public async executeTaskWithRetry(
    goal: string,
    options?: RomaExecutionOptions,
    retries: number = 3
  ): Promise<RomaExecutionResult> {
    let lastError: Error | undefined;

    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        const result = await this.client.executeTask(goal, options);
        
        // Validate the result
        if (this.validateExecutionResult(result)) {
          return result;
        } else {
          throw new Error('Invalid execution result');
        }
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        console.warn(`ROMA execution attempt ${attempt} failed: ${lastError.message}`);

        // Don't retry on the last attempt
        if (attempt === retries) {
          break;
        }

        // Exponential backoff
        const delay = Math.min(1000 * Math.pow(2, attempt), 5000);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    throw lastError || new Error('ROMA execution failed after retries');
  }

  /**
   * Execute task with caching
   * @param goal Task goal
   * @param options Execution options
   */
  public async executeTaskWithCache(
    goal: string,
    options?: RomaExecutionOptions
  ): Promise<RomaExecutionResult> {
    // Check cache first
    const cachedResult = this.getCachedExecution(goal);
    if (cachedResult) {
      console.log(`ROMA cache hit for goal: ${goal}`);
      return cachedResult;
    }

    // Execute task
    const result = await this.executeTaskWithRetry(goal, options);

    // Cache the result
    this.cacheExecutionResult(goal, result);

    return result;
  }

  /**
   * Get cached execution result
   * @param goal Task goal
   */
  public getCachedExecution(goal: string): RomaExecutionResult | undefined {
    const cachedResult = this.executionCache.get(goal);
    
    if (!cachedResult) {
      return undefined;
    }

    // Check if cache is still valid
    const cacheAge = Date.now() - cachedResult.timestamp;
    if (cacheAge > this.cacheTTL) {
      this.executionCache.delete(goal);
      return undefined;
    }

    return cachedResult;
  }

  /**
   * Cache execution result
   * @param goal Task goal
   * @param result Execution result
   */
  public cacheExecutionResult(goal: string, result: RomaExecutionResult): void {
    this.executionCache.set(goal, {
      ...result,
      timestamp: Date.now()
    });
  }

  /**
   * Clear cache
   */
  public clearCache(): void {
    this.executionCache.clear();
    console.log('ROMA execution cache cleared');
  }

  /**
   * Set cache TTL
   * @param ttl Cache time-to-live in milliseconds
   */
  public setCacheTTL(ttl: number): void {
    this.cacheTTL = ttl;
    console.log(`ROMA cache TTL set to ${ttl}ms`);
  }

  /**
   * Validate execution result
   * @param result Execution result to validate
   */
  public validateExecutionResult(result: RomaExecutionResult): boolean {
    // Basic validation
    if (!result || !result.executionId || !result.goal || !result.status) {
      console.error('Invalid ROMA execution result: missing required fields');
      return false;
    }

    // Check status
    const validStatuses: RomaExecutionResult['status'][] = [
      'initializing', 'idle', 'configuring', 'executing', 'paused',
      'completed', 'failed', 'cancelled'
    ];

    if (!validStatuses.includes(result.status)) {
      console.error(`Invalid ROMA execution status: ${result.status}`);
      return false;
    }

    // Check statistics
    if (result.statistics && (
      result.statistics.executionTime < 0 ||
      result.statistics.subtasksCreated < 0 ||
      result.statistics.subtasksCompleted < 0
    )) {
      console.error('Invalid ROMA execution statistics: negative values');
      return false;
    }

    console.log('ROMA execution result validation passed');
    return true;
  }

  /**
   * Format execution result for display
   * @param result Execution result
   */
  public formatExecutionResult(result: RomaExecutionResult): string {
    if (result.status === 'failed') {
      return `Execution failed: ${result.error || 'Unknown error'}`;
    }

    if (result.status === 'cancelled') {
      return 'Execution was cancelled';
    }

    if (!result.result) {
      return 'No result available';
    }

    // Format based on result type
    if (typeof result.result === 'string') {
      return result.result;
    }

    if (typeof result.result === 'object') {
      try {
        return JSON.stringify(result.result, null, 2);
      } catch (error) {
        return '[Object result]';
      }
    }

    return String(result.result);
  }

  /**
   * Get execution plan
   * Retrieves the detailed execution plan for a specific execution from ROMA API.
   * Includes subtasks, dependencies graph, and execution metadata.
   *
   * @param executionId - Execution ID to retrieve plan for
   * @returns Execution plan with subtasks and dependencies
   * @throws Error if execution not found or API call fails
   */
  public async getExecutionPlan(executionId: string): Promise<any> {
    try {
      console.log(`Getting execution plan for ${executionId}`);
      
      // Get execution details from ROMA API
      const execution = await this.client.getExecution(executionId);
      
      if (!execution) {
        throw new Error(`Execution ${executionId} not found`);
      }

      // Build execution plan from execution data
      // In a real implementation, this would call ROMA's /executions/{id}/plan endpoint
      // For now, we construct a plan from available execution data
      
      const plan = {
        executionId,
        originalGoal: execution.goal,
        subtasks: execution.statistics?.subtasksCreated || [],
        dependenciesGraph: execution.result?.dependenciesGraph || {},
        createdAt: execution.timestamp,
        status: execution.status,
        modules: execution.statistics?.modulesUsed || [],
        tools: execution.statistics?.toolsUsed || []
      };

      console.log('Execution plan retrieved:', plan);
      return plan;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error retrieving execution plan';
      console.error('Failed to get execution plan:', error);
      throw new Error(errorMessage);
    }
  }

  /**
   * Analyze execution performance
   * Provides detailed performance analysis for a specific execution.
   * Includes timing metrics, tool usage, module efficiency, and overall score.
   *
   * @param executionId - Execution ID to analyze
   * @returns Performance metrics including timing, usage, and efficiency score
   * @throws Error if execution not found or API call fails
   */
  public async analyzeExecutionPerformance(executionId: string): Promise<Record<string, any>> {
    try {
      console.log(`Analyzing performance for execution ${executionId}`);
      
      // Get execution details from ROMA API
      const execution = await this.client.getExecution(executionId);
      
      if (!execution) {
        throw new Error(`Execution ${executionId} not found`);
      }

      const stats = execution.statistics;
      if (!stats) {
        throw new Error(`No statistics available for execution ${executionId}`);
      }

      // Calculate performance metrics
      const totalTime = stats.executionTime || 0;
      const subtasksCreated = stats.subtasksCreated || 0;
      const subtasksCompleted = stats.subtasksCompleted || 0;
      const toolsUsed = stats.toolsUsed || [];
      const modulesUsed = stats.modulesUsed || [];

      // Calculate average subtask time
      const averageSubtaskTime = subtasksCompleted > 0
        ? totalTime / subtasksCompleted
        : 0;

      // Calculate completion rate
      const completionRate = subtasksCreated > 0
        ? (subtasksCompleted / subtasksCreated) * 100
        : 100;

      // Calculate tool usage frequency
      const toolUsage: Record<string, number> = {};
      toolsUsed.forEach(tool => {
        toolUsage[tool] = (toolUsage[tool] || 0) + 1;
      });

      // Calculate module usage frequency
      const moduleUsage: Record<string, number> = {};
      modulesUsed.forEach(module => {
        moduleUsage[module] = (moduleUsage[module] || 0) + 1;
      });

      // Calculate efficiency score based on completion rate and time
      // Higher completion rate and lower time = better efficiency
      const efficiencyScore = completionRate > 0
        ? Math.min(1, (completionRate / 100) * (1 - (totalTime / 60000))) // Normalize time (60s = 1.0)
        : 0;

      const performance = {
        executionId,
        totalTime,
        averageSubtaskTime,
        subtasksCreated,
        subtasksCompleted,
        completionRate: `${completionRate.toFixed(1)}%`,
        toolUsage,
        moduleUsage,
        efficiencyScore: efficiencyScore.toFixed(3),
        toolsUsed: toolsUsed.length,
        modulesUsed: modulesUsed.length,
        timestamp: Date.now()
      };

      console.log('Performance analysis completed:', performance);
      return performance;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error analyzing performance';
      console.error('Failed to analyze performance:', error);
      throw new Error(errorMessage);
    }
  }

  /**
   * Get cache statistics
   */
  public getCacheStatistics(): { size: number; hitRate: number } {
    return {
      size: this.executionCache.size,
      hitRate: 0 // Would need to track hits/misses for accurate rate
    };
  }

  /**
   * Check if result is cached
   * @param goal Task goal
   */
  public isResultCached(goal: string): boolean {
    return this.executionCache.has(goal);
  }

  /**
   * Get all cached execution goals
   */
  public getCachedExecutionGoals(): string[] {
    return Array.from(this.executionCache.keys());
  }
}