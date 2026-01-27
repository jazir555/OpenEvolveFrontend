/**
 * Trace Alerts - Alerting based on trace metrics
 *
 * This module provides alerting capabilities based on trace data,
 * allowing you to set up rules and thresholds for monitoring.
 */
import { TraceMetricsAnalyzer } from './trace-metrics.js';
import { TraceLogger } from './trace-logger.js';
/**
 * Alert manager for trace-based alerting
 */
export class TraceAlertManager {
    rules = new Map();
    metrics;
    alertHistory = [];
    logger = new TraceLogger();
    notificationCallbacks = new Map();
    constructor(metrics) {
        this.metrics = metrics || new TraceMetricsAnalyzer();
    }
    /**
     * Add a new alert rule
     */
    addRule(rule) {
        this.rules.set(rule.id, rule);
        this.logger.info('Added alert rule', {
            id: rule.id,
            name: rule.name,
            severity: rule.severity,
        });
    }
    /**
     * Remove an alert rule
     */
    removeRule(ruleId) {
        this.rules.delete(ruleId);
        this.logger.info('Removed alert rule', { ruleId });
    }
    /**
     * Get all rules
     */
    getRules() {
        return Array.from(this.rules.values());
    }
    /**
     * Get a specific rule
     */
    getRule(ruleId) {
        return this.rules.get(ruleId);
    }
    /**
     * Enable/disable a rule
     */
    setRuleEnabled(ruleId, enabled) {
        const rule = this.rules.get(ruleId);
        if (rule) {
            rule.enabled = enabled;
            this.logger.info('Updated rule enabled state', {
                ruleId,
                enabled,
            });
        }
    }
    /**
     * Register a notification callback
     */
    registerNotificationCallback(channel, callback) {
        this.notificationCallbacks.set(channel, callback);
        this.logger.info('Registered notification callback', { channel });
    }
    /**
     * Evaluate all enabled rules
     */
    evaluateRules() {
        const triggers = [];
        const metrics = this.metrics.getMetrics();
        for (const rule of this.rules.values()) {
            if (!rule.enabled) {
                continue;
            }
            const trigger = this.evaluateRule(rule, metrics);
            if (trigger) {
                triggers.push(trigger);
                this.alertHistory.push(trigger);
                this.sendNotifications(trigger);
            }
        }
        return triggers;
    }
    /**
     * Evaluate a single rule
     */
    evaluateRule(rule, metrics) {
        let shouldTrigger = false;
        let actualValue = 0;
        switch (rule.condition.metric) {
            case 'latency':
                actualValue = this.getLatencyMetric(rule.condition, metrics);
                shouldTrigger = actualValue > rule.threshold;
                break;
            case 'error_rate':
                actualValue = metrics.errorRate * 100;
                shouldTrigger = actualValue > rule.threshold;
                break;
            case 'throughput':
                actualValue = metrics.throughput;
                shouldTrigger = actualValue < rule.threshold;
                break;
            case 'missing_spans':
                actualValue = metrics.totalSpans;
                shouldTrigger = actualValue === 0;
                break;
        }
        if (shouldTrigger) {
            return {
                rule,
                actualValue,
                threshold: rule.threshold,
                timestamp: new Date(),
                affectedOperations: this.getAffectedOperations(rule),
                context: this.buildAlertContext(rule, metrics),
            };
        }
        return null;
    }
    /**
     * Get latency metric based on aggregation function
     */
    getLatencyMetric(condition, metrics) {
        switch (condition.aggregation) {
            case 'avg':
                return metrics.avgDuration;
            case 'p95':
                return metrics.p95Duration;
            case 'p99':
                return metrics.p99Duration;
            case 'max':
                return Math.max(...metrics.slowestOperations.map((op) => op.duration));
            default:
                return metrics.avgDuration;
        }
    }
    /**
     * Get operations affected by the alert
     */
    getAffectedOperations(rule) {
        const operations = [];
        if (rule.condition.filters) {
            for (const [key, value] of Object.entries(rule.condition.filters)) {
                if (key === 'bubble.name') {
                    operations.push(value);
                }
            }
        }
        return operations;
    }
    /**
     * Build additional context for the alert
     */
    buildAlertContext(rule, metrics) {
        return {
            totalTraces: metrics.totalTraces,
            totalSpans: metrics.totalSpans,
            errorRate: metrics.errorRate,
            avgDuration: metrics.avgDuration,
            p95Duration: metrics.p95Duration,
            throughput: metrics.throughput,
        };
    }
    /**
     * Send notifications for an alert
     */
    sendNotifications(alert) {
        this.logger.warn('Alert triggered', {
            ruleId: alert.rule.id,
            ruleName: alert.rule.name,
            severity: alert.rule.severity,
            actualValue: alert.actualValue,
            threshold: alert.threshold,
        });
        for (const channel of alert.rule.notificationChannels) {
            const callback = this.notificationCallbacks.get(channel);
            if (callback) {
                try {
                    callback(alert);
                }
                catch (error) {
                    this.logger.error('Failed to send notification', error, {
                        channel,
                        ruleId: alert.rule.id,
                    });
                }
            }
        }
    }
    /**
     * Get alert history
     */
    getAlertHistory(limit) {
        if (limit) {
            return this.alertHistory.slice(-limit);
        }
        return [...this.alertHistory];
    }
    /**
     * Clear alert history
     */
    clearAlertHistory() {
        this.alertHistory = [];
    }
    /**
     * Get alerts for a specific time range
     */
    getAlertsInTimeRange(startTime, endTime) {
        return this.alertHistory.filter((alert) => alert.timestamp >= startTime && alert.timestamp <= endTime);
    }
    /**
     * Get active alerts (recently triggered)
     */
    getActiveAlerts(withinLastSeconds = 300) {
        const cutoff = new Date(Date.now() - withinLastSeconds * 1000);
        return this.alertHistory.filter((alert) => alert.timestamp >= cutoff);
    }
}
/**
 * Predefined alert rules for common scenarios
 */
export class CommonAlertRules {
    /**
     * Alert on high P95 latency
     */
    static highP95Latency(thresholdMs = 30000) {
        return {
            id: 'high-p95-latency',
            name: 'High P95 Latency',
            description: `P95 latency exceeds ${thresholdMs}ms`,
            condition: {
                metric: 'latency',
                aggregation: 'p95',
            },
            threshold: thresholdMs,
            timeWindowSeconds: 300,
            severity: 'warning',
            enabled: true,
            notificationChannels: ['default'],
        };
    }
    /**
     * Alert on high error rate
     */
    static highErrorRate(thresholdPercent = 5) {
        return {
            id: 'high-error-rate',
            name: 'High Error Rate',
            description: `Error rate exceeds ${thresholdPercent}%`,
            condition: {
                metric: 'error_rate',
                aggregation: 'avg',
            },
            threshold: thresholdPercent,
            timeWindowSeconds: 300,
            severity: 'error',
            enabled: true,
            notificationChannels: ['default'],
        };
    }
    /**
     * Alert on missing spans (no traces)
     */
    static missingSpans(timeWindowSeconds = 300) {
        return {
            id: 'missing-spans',
            name: 'Missing Spans',
            description: 'No traces received within time window',
            condition: {
                metric: 'missing_spans',
                aggregation: 'avg',
            },
            threshold: 1,
            timeWindowSeconds,
            severity: 'critical',
            enabled: true,
            notificationChannels: ['default'],
        };
    }
    /**
     * Alert on slow operations
     */
    static slowOperations(thresholdMs = 10000) {
        return {
            id: 'slow-operations',
            name: 'Slow Operations',
            description: `Operations exceed ${thresholdMs}ms`,
            condition: {
                metric: 'latency',
                aggregation: 'p99',
            },
            threshold: thresholdMs,
            timeWindowSeconds: 600,
            severity: 'warning',
            enabled: true,
            notificationChannels: ['default'],
        };
    }
    /**
     * Alert on low throughput
     */
    static lowThroughput(thresholdOpsPerSecond = 0.1) {
        return {
            id: 'low-throughput',
            name: 'Low Throughput',
            description: `Throughput falls below ${thresholdOpsPerSecond} ops/sec`,
            condition: {
                metric: 'throughput',
                aggregation: 'avg',
            },
            threshold: thresholdOpsPerSecond,
            timeWindowSeconds: 600,
            severity: 'info',
            enabled: true,
            notificationChannels: ['default'],
        };
    }
}
//# sourceMappingURL=trace-alerts.js.map