/**
 * PERFORMANCE MONITORING UTILITIES
 *
 * Comprehensive performance monitoring for BubbleLab bubbles.
 * Tracks memory usage, resource leaks, and operation timing.
 *
 * Usage:
 * ```typescript
 * import { PerformanceMonitor } from './utils/performance-monitor.js';
 *
 * // Monitor a function
 * const result = await PerformanceMonitor.measure('my-operation', async () => {
 *   return await someOperation();
 * });
 *
 * // Log current memory usage
 * PerformanceMonitor.logMemoryUsage('startup');
 *
 * // Start monitoring session
 * const session = PerformanceMonitor.startSession('load-test');
 * // ... run operations ...
 * session.end();
 * ```
 */
export interface MemoryUsage {
    heapUsed: number;
    heapTotal: number;
    external: number;
    rss: number;
    arrayBuffers: number;
}
export interface PerformanceMetrics {
    context: string;
    duration: number;
    memoryDelta: number;
    success: boolean;
    error?: string;
    timestamp: string;
}
export interface SessionMetrics {
    sessionId: string;
    startTime: number;
    endTime?: number;
    operations: PerformanceMetrics[];
    peakMemory: number;
    totalOperations: number;
    failedOperations: number;
}
/**
 * Performance monitoring utilities
 */
export declare class PerformanceMonitor {
    private static eventEmitter;
    private static activeSessions;
    private static metricsHistory;
    private static readonly MAX_HISTORY_SIZE;
    /**
     * Log current memory usage
     */
    static logMemoryUsage(context: string): MemoryUsage;
    /**
     * Measure performance of an async operation
     */
    static measure<T>(context: string, fn: () => Promise<T>, metadata?: Record<string, unknown>): Promise<T>;
    /**
     * Start a monitoring session
     */
    static startSession(sessionId: string): SessionMonitor;
    /**
     * End a monitoring session and get report
     */
    static endSession(sessionId: string): SessionReport;
    /**
     * Get performance summary for a context
     */
    static getSummary(context: string): PerformanceSummary;
    /**
     * Detect potential memory leaks
     */
    static detectMemoryLeaks(thresholdMB?: number): LeakReport;
    /**
     * Calculate trend direction (-1 to 1)
     */
    private static calculateTrend;
    /**
     * Add metrics to history with size limit
     */
    private static addToHistory;
    /**
     * Clear all metrics history
     */
    static clearHistory(): void;
    /**
     * Subscribe to performance events
     */
    static on(event: string, listener: (...args: unknown[]) => void): void;
    /**
     * Unsubscribe from performance events
     */
    static off(event: string, listener: (...args: unknown[]) => void): void;
}
/**
 * Session monitor for tracking a group of operations
 */
export declare class SessionMonitor {
    private sessionId;
    constructor(sessionId: string);
    /**
     * Track an operation within this session
     */
    track<T>(context: string, fn: () => Promise<T>, metadata?: Record<string, unknown>): Promise<T>;
    /**
     * End the session and get report
     */
    end(): SessionReport;
}
/**
 * Type definitions for reports
 */
export interface PerformanceSummary {
    context: string;
    totalOperations: number;
    successRate: number;
    avgDuration: number;
    avgMemoryDelta: number;
    peakMemory: number;
}
export interface SessionReport {
    sessionId: string;
    duration: number;
    totalOperations: number;
    failedOperations: number;
    successRate: number;
    avgDuration: number;
    peakMemory: number;
    memoryDelta: number;
    operations: PerformanceMetrics[];
}
export interface LeakReport {
    potentialLeak: boolean;
    avgMemoryDeltaPerOp: number;
    trend: number;
    recommendation: string;
}
/**
 * Example usage and testing
 */
export declare function exampleUsage(): Promise<void>;
//# sourceMappingURL=performance-monitor.d.ts.map