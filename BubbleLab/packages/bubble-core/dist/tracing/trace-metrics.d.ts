/**
 * Trace Metrics - Performance analysis and metrics from traces
 *
 * This module provides utilities for analyzing trace data to extract
 * performance metrics, identify bottlenecks, and generate insights.
 */
import type { TraceMetrics as TraceMetricsType, PerformanceAnalysis } from './types.js';
/**
 * Performance metrics analyzer
 */
export declare class TraceMetricsAnalyzer {
    private logger;
    private operations;
    private errors;
    /**
     * Record an operation duration
     */
    recordOperation(name: string, duration: number): void;
    /**
     * Record an error
     */
    recordError(operation: string): void;
    /**
     * Get metrics for all operations
     */
    getMetrics(): TraceMetricsType;
    /**
     * Get metrics for a specific operation
     */
    getOperationMetrics(name: string): {
        count: number;
        avgDuration: number;
        p95Duration: number;
        p99Duration: number;
        errorRate: number;
    } | null;
    /**
     * Analyze performance and provide recommendations
     */
    analyzePerformance(): PerformanceAnalysis;
    /**
     * Identify performance bottlenecks
     */
    private identifyBottlenecks;
    /**
     * Get suggested action for a bottleneck
     */
    private getSuggestedAction;
    /**
     * Generate performance recommendations
     */
    private generateRecommendations;
    /**
     * Identify critical path in operations
     */
    private identifyCriticalPath;
    /**
     * Get slowest operations
     */
    private getSlowestOperations;
    /**
     * Calculate average
     */
    private average;
    /**
     * Calculate percentile
     */
    private percentile;
    /**
     * Calculate throughput (operations per second)
     */
    private calculateThroughput;
    /**
     * Clear all recorded metrics
     */
    clear(): void;
}
/**
 * Analyze performance from recorded metrics
 */
export declare function analyzePerformance(): PerformanceAnalysis;
/**
 * Create a performance tracking span
 */
export declare function trackPerformance(operationName: string, fn: () => Promise<void>): Promise<void>;
//# sourceMappingURL=trace-metrics.d.ts.map