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

import { EventEmitter } from 'events';

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
export class PerformanceMonitor {
  private static eventEmitter = new EventEmitter();
  private static activeSessions = new Map<string, SessionMetrics>();
  private static metricsHistory: PerformanceMetrics[] = [];
  private static readonly MAX_HISTORY_SIZE = 10000;

  /**
   * Log current memory usage
   */
  static logMemoryUsage(context: string): MemoryUsage {
    const usage = process.memoryUsage();
    const formatted: MemoryUsage = {
      heapUsed: Math.round(usage.heapUsed / 1024 / 1024),
      heapTotal: Math.round(usage.heapTotal / 1024 / 1024),
      external: Math.round((usage as any).external / 1024 / 1024),
      rss: Math.round(usage.rss / 1024 / 1024),
      arrayBuffers: Math.round((usage as any).arrayBuffers / 1024 / 1024),
    };

    console.log({
      context,
      timestamp: new Date().toISOString(),
      memory: formatted,
    });

    // Emit event for monitoring systems
    this.eventEmitter.emit('memory-usage', { context, memory: formatted });

    return formatted;
  }

  /**
   * Measure performance of an async operation
   */
  static async measure<T>(
    context: string,
    fn: () => Promise<T>,
    metadata?: Record<string, unknown>
  ): Promise<T> {
    const startTime = Date.now();
    const startMemory = process.memoryUsage().heapUsed;

    console.log(`[PerformanceMonitor] Starting: ${context}`);

    try {
      const result = await fn();

      const duration = Date.now() - startTime;
      const memoryDelta = process.memoryUsage().heapUsed - startMemory;

      const metrics: PerformanceMetrics = {
        context,
        duration,
        memoryDelta: Math.round(memoryDelta / 1024 / 1024),
        success: true,
        timestamp: new Date().toISOString(),
      };

      console.log({
        message: `✓ ${context} completed`,
        duration: `${duration}ms`,
        memoryDelta: `${metrics.memoryDelta}MB`,
        metadata,
      });

      // Add to history
      this.addToHistory(metrics);

      // Emit event
      this.eventEmitter.emit('operation-complete', { metrics, metadata });

      return result;
    } catch (error) {
      const duration = Date.now() - startTime;
      const memoryDelta = process.memoryUsage().heapUsed - startMemory;

      const metrics: PerformanceMetrics = {
        context,
        duration,
        memoryDelta: Math.round(memoryDelta / 1024 / 1024),
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString(),
      };

      console.error({
        message: `✗ ${context} failed`,
        duration: `${duration}ms`,
        memoryDelta: `${metrics.memoryDelta}MB`,
        error: metrics.error,
        metadata,
      });

      // Add to history
      this.addToHistory(metrics);

      // Emit event
      this.eventEmitter.emit('operation-failed', { metrics, metadata });

      throw error;
    }
  }

  /**
   * Start a monitoring session
   */
  static startSession(sessionId: string): SessionMonitor {
    const session: SessionMetrics = {
      sessionId,
      startTime: Date.now(),
      operations: [],
      peakMemory: process.memoryUsage().heapUsed,
      totalOperations: 0,
      failedOperations: 0,
    };

    this.activeSessions.set(sessionId, session);
    console.log(`[PerformanceMonitor] Session started: ${sessionId}`);

    return new SessionMonitor(sessionId);
  }

  /**
   * End a monitoring session and get report
   */
  static endSession(sessionId: string): SessionReport {
    const session = this.activeSessions.get(sessionId);
    if (!session) {
      throw new Error(`Session not found: ${sessionId}`);
    }

    session.endTime = Date.now();
    const duration = session.endTime - session.startTime;

    const report: SessionReport = {
      sessionId,
      duration,
      totalOperations: session.totalOperations,
      failedOperations: session.failedOperations,
      successRate: ((session.totalOperations - session.failedOperations) / session.totalOperations) * 100,
      avgDuration: session.operations.reduce((sum, op) => sum + op.duration, 0) / session.operations.length,
      peakMemory: Math.round(session.peakMemory / 1024 / 1024),
      memoryDelta: Math.round((process.memoryUsage().heapUsed - session.startTime) / 1024 / 1024),
      operations: session.operations,
    };

    console.log(`[PerformanceMonitor] Session ended: ${sessionId}`, report);

    // Remove from active sessions
    this.activeSessions.delete(sessionId);

    // Emit event
    this.eventEmitter.emit('session-ended', { sessionId, report });

    return report;
  }

  /**
   * Get performance summary for a context
   */
  static getSummary(context: string): PerformanceSummary {
    const contextMetrics = this.metricsHistory.filter((m) => m.context === context);

    if (contextMetrics.length === 0) {
      return {
        context,
        totalOperations: 0,
        successRate: 0,
        avgDuration: 0,
        avgMemoryDelta: 0,
        peakMemory: 0,
      };
    }

    const successful = contextMetrics.filter((m) => m.success);
    const totalDuration = contextMetrics.reduce((sum, m) => sum + m.duration, 0);
    const totalMemoryDelta = contextMetrics.reduce((sum, m) => sum + m.memoryDelta, 0);

    return {
      context,
      totalOperations: contextMetrics.length,
      successRate: (successful.length / contextMetrics.length) * 100,
      avgDuration: totalDuration / contextMetrics.length,
      avgMemoryDelta: totalMemoryDelta / contextMetrics.length,
      peakMemory: Math.max(...contextMetrics.map((m) => m.memoryDelta)),
    };
  }

  /**
   * Detect potential memory leaks
   */
  static detectMemoryLeaks(thresholdMB: number = 100): LeakReport {
    const recentMetrics = this.metricsHistory.slice(-100); // Last 100 operations
    const memoryDeltas = recentMetrics.map((m) => m.memoryDelta);

    const avgDelta = memoryDeltas.reduce((sum, d) => sum + d, 0) / memoryDeltas.length;
    const increasingTrend = this.calculateTrend(memoryDeltas);

    const potentialLeak = avgDelta > thresholdMB || increasingTrend > 0.5;

    return {
      potentialLeak,
      avgMemoryDeltaPerOp: avgDelta,
      trend: increasingTrend,
      recommendation: potentialLeak
        ? 'WARNING: Potential memory leak detected. Memory is increasing over time.'
        : 'OK: No memory leak detected.',
    };
  }

  /**
   * Calculate trend direction (-1 to 1)
   */
  private static calculateTrend(values: number[]): number {
    if (values.length < 2) return 0;

    let increases = 0;
    for (let i = 1; i < values.length; i++) {
      if (values[i] > values[i - 1]) increases++;
    }

    return increases / (values.length - 1);
  }

  /**
   * Add metrics to history with size limit
   */
  private static addToHistory(metrics: PerformanceMetrics): void {
    this.metricsHistory.push(metrics);

    // Enforce max history size
    if (this.metricsHistory.length > this.MAX_HISTORY_SIZE) {
      // Remove oldest entries
      const excess = this.metricsHistory.length - this.MAX_HISTORY_SIZE;
      this.metricsHistory.splice(0, excess);
    }
  }

  /**
   * Clear all metrics history
   */
  static clearHistory(): void {
    this.metricsHistory = [];
    console.log('[PerformanceMonitor] History cleared');
  }

  /**
   * Subscribe to performance events
   */
  static on(event: string, listener: (...args: unknown[]) => void): void {
    this.eventEmitter.on(event, listener);
  }

  /**
   * Unsubscribe from performance events
   */
  static off(event: string, listener: (...args: unknown[]) => void): void {
    this.eventEmitter.off(event, listener);
  }
}

/**
 * Session monitor for tracking a group of operations
 */
export class SessionMonitor {
  constructor(private sessionId: string) {}

  /**
   * Track an operation within this session
   */
  async track<T>(
    context: string,
    fn: () => Promise<T>,
    metadata?: Record<string, unknown>
  ): Promise<T> {
    const session = PerformanceMonitor['activeSessions'].get(this.sessionId);
    if (!session) {
      throw new Error(`Session not found: ${this.sessionId}`);
    }

    const startTime = Date.now();
    const startMemory = process.memoryUsage().heapUsed;

    try {
      const result = await fn();

      const duration = Date.now() - startTime;
      const memoryDelta = process.memoryUsage().heapUsed - startMemory;

      const metrics: PerformanceMetrics = {
        context,
        duration,
        memoryDelta: Math.round(memoryDelta / 1024 / 1024),
        success: true,
        timestamp: new Date().toISOString(),
      };

      session.operations.push(metrics);
      session.totalOperations++;
      session.peakMemory = Math.max(session.peakMemory, process.memoryUsage().heapUsed);

      return result;
    } catch (error) {
      const duration = Date.now() - startTime;
      const memoryDelta = process.memoryUsage().heapUsed - startMemory;

      const metrics: PerformanceMetrics = {
        context,
        duration,
        memoryDelta: Math.round(memoryDelta / 1024 / 1024),
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString(),
      };

      session.operations.push(metrics);
      session.totalOperations++;
      session.failedOperations++;

      throw error;
    }
  }

  /**
   * End the session and get report
   */
  end(): SessionReport {
    return PerformanceMonitor.endSession(this.sessionId);
  }
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
export async function exampleUsage(): Promise<void> {
  // Monitor memory usage at startup
  PerformanceMonitor.logMemoryUsage('startup');

  // Measure a single operation
  const result = await PerformanceMonitor.measure('data-fetch', async () => {
    // Simulate async operation
    await new Promise((resolve) => setTimeout(resolve, 100));
    return { data: 'some result' };
  });

  // Start a session for multiple operations
  const session = PerformanceMonitor.startSession('load-test');

  await session.track('operation-1', async () => {
    await new Promise((resolve) => setTimeout(resolve, 50));
    return 'result-1';
  });

  await session.track('operation-2', async () => {
    await new Promise((resolve) => setTimeout(resolve, 75));
    return 'result-2';
  });

  // End session and get report
  const report = session.end();
  console.log('Session Report:', report);

  // Check for memory leaks
  const leakReport = PerformanceMonitor.detectMemoryLeaks(100);
  console.log('Memory Leak Report:', leakReport);
}
