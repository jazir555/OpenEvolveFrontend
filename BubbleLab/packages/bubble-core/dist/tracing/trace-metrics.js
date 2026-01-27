/**
 * Trace Metrics - Performance analysis and metrics from traces
 *
 * This module provides utilities for analyzing trace data to extract
 * performance metrics, identify bottlenecks, and generate insights.
 */
import { TraceLogger } from './trace-logger.js';
/**
 * Performance metrics analyzer
 */
export class TraceMetricsAnalyzer {
    logger = new TraceLogger();
    operations = new Map();
    errors = new Map();
    /**
     * Record an operation duration
     */
    recordOperation(name, duration) {
        if (!this.operations.has(name)) {
            this.operations.set(name, []);
        }
        this.operations.get(name).push(duration);
        this.logger.debug('Recorded operation', {
            name,
            duration,
            count: this.operations.get(name).length,
        });
    }
    /**
     * Record an error
     */
    recordError(operation) {
        const count = this.errors.get(operation) || 0;
        this.errors.set(operation, count + 1);
        this.logger.debug('Recorded error', {
            operation,
            count: count + 1,
        });
    }
    /**
     * Get metrics for all operations
     */
    getMetrics() {
        const durations = Array.from(this.operations.values()).flat();
        const errorCount = Array.from(this.errors.values()).reduce((sum, count) => sum + count, 0);
        const totalOperations = durations.length;
        return {
            totalTraces: totalOperations,
            totalSpans: totalOperations,
            errorRate: totalOperations > 0 ? errorCount / totalOperations : 0,
            avgDuration: durations.length > 0 ? this.average(durations) : 0,
            p50Duration: durations.length > 0 ? this.percentile(durations, 50) : 0,
            p95Duration: durations.length > 0 ? this.percentile(durations, 95) : 0,
            p99Duration: durations.length > 0 ? this.percentile(durations, 99) : 0,
            throughput: this.calculateThroughput(durations),
            slowestOperations: this.getSlowestOperations(10),
        };
    }
    /**
     * Get metrics for a specific operation
     */
    getOperationMetrics(name) {
        const durations = this.operations.get(name);
        if (!durations || durations.length === 0) {
            return null;
        }
        const errorCount = this.errors.get(name) || 0;
        return {
            count: durations.length,
            avgDuration: this.average(durations),
            p95Duration: this.percentile(durations, 95),
            p99Duration: this.percentile(durations, 99),
            errorRate: errorCount / durations.length,
        };
    }
    /**
     * Analyze performance and provide recommendations
     */
    analyzePerformance() {
        const metrics = this.getMetrics();
        return {
            metrics,
            bottlenecks: this.identifyBottlenecks(),
            recommendations: this.generateRecommendations(),
            criticalPath: this.identifyCriticalPath(),
        };
    }
    /**
     * Identify performance bottlenecks
     */
    identifyBottlenecks() {
        const bottlenecks = [];
        const avgDuration = this.getMetrics().avgDuration;
        for (const [name, durations] of this.operations.entries()) {
            const avgOpDuration = this.average(durations);
            const errorRate = (this.errors.get(name) || 0) / durations.length;
            let impact = 'low';
            if (avgOpDuration > avgDuration * 3) {
                impact = 'critical';
            }
            else if (avgOpDuration > avgDuration * 2) {
                impact = 'high';
            }
            else if (avgOpDuration > avgDuration * 1.5) {
                impact = 'medium';
            }
            if (impact !== 'low') {
                bottlenecks.push({
                    operation: name,
                    avgDuration: avgOpDuration,
                    impact,
                    frequency: durations.length,
                    suggestedAction: this.getSuggestedAction(name, impact),
                });
            }
        }
        return bottlenecks.sort((a, b) => b.avgDuration - a.avgDuration);
    }
    /**
     * Get suggested action for a bottleneck
     */
    getSuggestedAction(operation, impact) {
        if (operation.includes('database') || operation.includes('db')) {
            return impact === 'critical'
                ? 'Consider adding database indexes or optimizing queries'
                : 'Review query performance and add caching if applicable';
        }
        if (operation.includes('http') || operation.includes('api')) {
            return impact === 'critical'
                ? 'Implement request caching and consider rate limiting'
                : 'Review API call patterns and implement connection pooling';
        }
        if (operation.includes('ai') || operation.includes('llm')) {
            return 'Consider implementing response caching or using smaller models';
        }
        return 'Review operation implementation and consider optimization strategies';
    }
    /**
     * Generate performance recommendations
     */
    generateRecommendations() {
        const recommendations = [];
        const metrics = this.getMetrics();
        if (metrics.p95Duration > 5000) {
            recommendations.push({
                type: 'optimization',
                priority: 'high',
                description: 'P95 latency exceeds 5 seconds. Consider optimizing slow operations.',
                expectedImpact: '20-30% reduction in latency',
            });
        }
        if (metrics.errorRate > 0.05) {
            recommendations.push({
                type: 'error-handling',
                priority: 'critical',
                description: 'Error rate exceeds 5%. Review error handling and implement retry logic.',
                expectedImpact: 'Improved reliability and user experience',
            });
        }
        const bottlenecks = this.identifyBottlenecks();
        if (bottlenecks.length > 0) {
            const criticalBottlenecks = bottlenecks.filter((b) => b.impact === 'critical');
            if (criticalBottlenecks.length > 0) {
                recommendations.push({
                    type: 'optimization',
                    priority: 'critical',
                    description: `${criticalBottlenecks.length} critical bottlenecks identified. Focus optimization efforts here.`,
                    expectedImpact: 'Significant performance improvement',
                });
            }
        }
        return recommendations;
    }
    /**
     * Identify critical path in operations
     */
    identifyCriticalPath() {
        // This is a simplified version. A full implementation would analyze
        // the span hierarchy to determine actual critical paths.
        const paths = [];
        const avgDuration = this.getMetrics().avgDuration;
        for (const [name, durations] of this.operations.entries()) {
            const totalDuration = this.average(durations);
            const percentageOfTotal = avgDuration > 0 ? (totalDuration / avgDuration) * 100 : 0;
            if (percentageOfTotal > 20) {
                paths.push({
                    operations: [name],
                    totalDuration,
                    percentageOfTotal,
                    optimizationPotential: Math.min(percentageOfTotal - 20, 50),
                });
            }
        }
        return paths.sort((a, b) => b.totalDuration - a.totalDuration);
    }
    /**
     * Get slowest operations
     */
    getSlowestOperations(limit) {
        const operations = [];
        for (const [name, durations] of this.operations.entries()) {
            const avgDuration = this.average(durations);
            const maxDuration = Math.max(...durations);
            operations.push({
                name,
                duration: maxDuration,
                timestamp: new Date(),
            });
        }
        return operations
            .sort((a, b) => b.duration - a.duration)
            .slice(0, limit);
    }
    /**
     * Calculate average
     */
    average(values) {
        if (values.length === 0)
            return 0;
        return values.reduce((sum, val) => sum + val, 0) / values.length;
    }
    /**
     * Calculate percentile
     */
    percentile(values, p) {
        if (values.length === 0)
            return 0;
        const sorted = [...values].sort((a, b) => a - b);
        const index = Math.ceil((p / 100) * sorted.length) - 1;
        return sorted[Math.max(0, index)];
    }
    /**
     * Calculate throughput (operations per second)
     */
    calculateThroughput(durations) {
        if (durations.length === 0)
            return 0;
        const totalTime = durations.reduce((sum, val) => sum + val, 0);
        const avgDuration = totalTime / durations.length;
        return avgDuration > 0 ? 1000 / avgDuration : 0;
    }
    /**
     * Clear all recorded metrics
     */
    clear() {
        this.operations.clear();
        this.errors.clear();
    }
}
/**
 * Analyze performance from recorded metrics
 */
export function analyzePerformance() {
    const metrics = new TraceMetricsAnalyzer();
    return metrics.analyzePerformance();
}
/**
 * Create a performance tracking span
 */
export async function trackPerformance(operationName, fn) {
    const metrics = new TraceMetricsAnalyzer();
    const startTime = Date.now();
    try {
        await fn();
        const duration = Date.now() - startTime;
        metrics.recordOperation(operationName, duration);
    }
    catch (error) {
        metrics.recordError(operationName);
        throw error;
    }
}
//# sourceMappingURL=trace-metrics.js.map