/**
 * Advanced Utilities for OpenEvolve Enhanced Plugin
 * Provides sophisticated utility functions for performance, security, monitoring, and integration
 */
/**
 * Performance Benchmarking Utility
 * Measures and compares execution times for different approaches
 */
export declare class PerformanceBenchmark {
    private results;
    private maxResults;
    private enableMemoryTracking;
    constructor(options?: {
        maxResults?: number;
        enableMemoryTracking?: boolean;
    });
    /**
     * Measure execution time and memory usage of a function
     */
    measure<T>(name: string, fn: () => Promise<T>): Promise<{
        result: T;
        metrics: {
            executionTime: number;
            memoryUsage: number;
        };
    }>;
    /**
     * Compare multiple approaches with optional parallel execution
     */
    compare<T>(name: string, approaches: Array<{
        name: string;
        fn: () => Promise<T>;
    }>, options?: {
        parallel?: boolean;
        maxConcurrent?: number;
    }): Promise<{
        winner: string;
        results: Array<{
            name: string;
            executionTime: number;
            memoryUsage: number;
            result: T;
            error?: string;
        }>;
    }>;
    /**
     * Get all benchmark results
     */
    getResults(): typeof this.results;
    /**
     * Clear benchmark results
     */
    clearResults(): void;
    /**
     * Calculate performance improvement percentage
     */
    calculateImprovement(baselineTime: number, optimizedTime: number): number;
    /**
     * Get performance statistics and insights
     */
    getPerformanceStatistics(): {
        totalBenchmarks: number;
        averageExecutionTime: number;
        averageMemoryUsage: number;
        fastestBenchmark?: {
            name: string;
            executionTime: number;
        };
        slowestBenchmark?: {
            name: string;
            executionTime: number;
        };
        memoryIntensiveBenchmark?: {
            name: string;
            memoryUsage: number;
        };
    };
    /**
     * Get memory usage with fallback for different environments
     */
    getMemoryUsage(): {
        used: number;
        total: number;
        free: number;
        percentage: number;
    };
    /**
     * Clear all benchmark results to free memory
     */
    clearAllResults(): void;
    /**
     * Get results within a specific time window
     */
    getRecentResults(timeWindow?: number): typeof this.results;
    /**
     * Export results to JSON for analysis
     */
    exportResultsToJson(pretty?: boolean): string;
    /**
     * Import results from JSON
     */
    importResultsFromJson(json: string): void;
}
/**
 * Security Utility Functions
 */
export declare class SecurityUtils {
    /**
     * Generate secure random token
     */
    static generateSecureToken(length?: number): string;
    /**
     * Hash data using SHA-256 (browser implementation)
     */
    static hashData(data: string): Promise<string>;
    /**
     * Simple hash function (fallback)
     */
    private static simpleHash;
    /**
     * Validate password strength
     */
    static validatePasswordStrength(password: string): {
        score: number;
        feedback: string[];
        isStrong: boolean;
    };
    /**
     * Generate JWT-like token (simplified)
     */
    static generateJwtLikeToken(payload: Record<string, any>, secret: string, expiresIn?: number): string;
    /**
     * Validate JWT-like token (simplified)
     */
    static validateJwtLikeToken(token: string, secret: string): {
        valid: boolean;
        payload?: Record<string, any>;
        error?: string;
    };
}
/**
 * Monitoring and Alerting Utilities
 */
export declare class MonitoringUtils {
    private alerts;
    /**
     * Check metric against threshold and create alert if needed
     */
    checkMetricAndAlert(metric: string, value: number, threshold: number, severity?: 'low' | 'medium' | 'high' | 'critical'): boolean;
    /**
     * Resolve an alert
     */
    resolveAlert(alertId: string): boolean;
    /**
     * Get active alerts
     */
    getActiveAlerts(): typeof this.alerts;
    /**
     * Get alert history
     */
    getAlertHistory(): typeof this.alerts;
    /**
     * Clear resolved alerts
     */
    clearResolvedAlerts(): void;
    /**
     * Calculate alert frequency
     */
    getAlertFrequency(metric: string, timeWindow?: number): number;
    /**
     * Check if system is in alert storm
     */
    isAlertStorm(timeWindow?: number, threshold?: number): boolean;
}
/**
 * Integration and API Utilities
 */
export declare class IntegrationUtils {
    private apiCache;
    private cacheStats;
    private maxCacheSize;
    constructor(options?: {
        maxCacheSize?: number;
    });
    /**
     * Smart API caller with caching and retry logic
     */
    smartApiCall(url: string, options?: {
        method?: string;
        headers?: Record<string, string>;
        body?: any;
        cacheTtl?: number;
        maxRetries?: number;
        retryDelay?: number;
        timeout?: number;
        useExponentialBackoff?: boolean;
    }): Promise<{
        success: boolean;
        data?: any;
        error?: string;
        fromCache?: boolean;
        retryCount: number;
    }>;
    /**
     * Simulate API call (replace with actual fetch/axios in production)
     */
    private simulateApiCall;
    /**
     * Clear API cache
     */
    clearApiCache(): void;
    /**
     * Get cache statistics
     */
    getCacheStats(): {
        size: number;
        hitRate: number;
    };
    /**
     * Batch API calls with rate limiting
     */
    batchApiCalls(calls: Array<{
        url: string;
        options?: any;
    }>, batchSize?: number, delayBetweenBatches?: number): Promise<Array<{
        url: string;
        success: boolean;
        data?: any;
        error?: string;
    }>>;
}
/**
 * Advanced Error Analysis Utilities
 */
export declare class ErrorAnalysisUtils {
    private errorHistory;
    /**
     * Analyze error and add to history
     */
    analyzeError(error: unknown, context?: string, severity?: 'low' | 'medium' | 'high' | 'critical', metadata?: Record<string, any>): string;
    /**
     * Get error patterns and trends
     */
    getErrorPatterns(timeWindow?: number): {
        byType: Record<string, number>;
        byContext: Record<string, number>;
        bySeverity: Record<string, number>;
        total: number;
    };
    /**
     * Get most common errors
     */
    getMostCommonErrors(limit?: number): Array<{
        errorType: string;
        count: number;
        percentage: number;
    }>;
    /**
     * Detect error trends (increasing frequency)
     */
    detectErrorTrends(errorType: string, timeWindows?: number[]): {
        isTrending: boolean;
        trendData: Array<{
            windowMs: number;
            count: number;
            rate: number;
        }>;
    };
    /**
     * Get error history
     */
    getErrorHistory(limit?: number): typeof this.errorHistory;
    /**
     * Clear error history
     */
    clearErrorHistory(): void;
    /**
     * Get error severity distribution
     */
    getSeverityDistribution(): Record<'low' | 'medium' | 'high' | 'critical', number>;
}
/**
 * Configuration Management Utilities
 */
export declare class ConfigUtils {
    /**
     * Deep merge configuration objects
     */
    static deepMerge<T>(target: T, source: Partial<T>): T;
    /**
     * Validate configuration against schema
     */
    static validateConfig(config: any, schema: any): {
        valid: boolean;
        errors: string[];
    };
    /**
     * Generate configuration diff
     */
    static generateConfigDiff(oldConfig: any, newConfig: any): Array<{
        path: string;
        oldValue: any;
        newValue: any;
        changeType: 'added' | 'removed' | 'changed';
    }>;
    /**
     * Sanitize configuration (remove sensitive data)
     */
    static sanitizeConfig(config: any, sensitiveKeys?: string[]): any;
    /**
     * Export configuration to JSON
     */
    static exportConfigToJson(config: any, pretty?: boolean): string;
    /**
     * Import configuration from JSON
     */
    static importConfigFromJson(json: string): any;
}
