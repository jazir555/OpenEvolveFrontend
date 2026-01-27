/**
 * Trace Alerts - Alerting based on trace metrics
 *
 * This module provides alerting capabilities based on trace data,
 * allowing you to set up rules and thresholds for monitoring.
 */
import type { AlertRule, AlertTrigger } from './types.js';
import { TraceMetricsAnalyzer } from './trace-metrics.js';
/**
 * Alert manager for trace-based alerting
 */
export declare class TraceAlertManager {
    private rules;
    private metrics;
    private alertHistory;
    private logger;
    private notificationCallbacks;
    constructor(metrics?: TraceMetricsAnalyzer);
    /**
     * Add a new alert rule
     */
    addRule(rule: AlertRule): void;
    /**
     * Remove an alert rule
     */
    removeRule(ruleId: string): void;
    /**
     * Get all rules
     */
    getRules(): AlertRule[];
    /**
     * Get a specific rule
     */
    getRule(ruleId: string): AlertRule | undefined;
    /**
     * Enable/disable a rule
     */
    setRuleEnabled(ruleId: string, enabled: boolean): void;
    /**
     * Register a notification callback
     */
    registerNotificationCallback(channel: string, callback: (alert: AlertTrigger) => void): void;
    /**
     * Evaluate all enabled rules
     */
    evaluateRules(): AlertTrigger[];
    /**
     * Evaluate a single rule
     */
    private evaluateRule;
    /**
     * Get latency metric based on aggregation function
     */
    private getLatencyMetric;
    /**
     * Get operations affected by the alert
     */
    private getAffectedOperations;
    /**
     * Build additional context for the alert
     */
    private buildAlertContext;
    /**
     * Send notifications for an alert
     */
    private sendNotifications;
    /**
     * Get alert history
     */
    getAlertHistory(limit?: number): AlertTrigger[];
    /**
     * Clear alert history
     */
    clearAlertHistory(): void;
    /**
     * Get alerts for a specific time range
     */
    getAlertsInTimeRange(startTime: Date, endTime: Date): AlertTrigger[];
    /**
     * Get active alerts (recently triggered)
     */
    getActiveAlerts(withinLastSeconds?: number): AlertTrigger[];
}
/**
 * Predefined alert rules for common scenarios
 */
export declare class CommonAlertRules {
    /**
     * Alert on high P95 latency
     */
    static highP95Latency(thresholdMs?: number): AlertRule;
    /**
     * Alert on high error rate
     */
    static highErrorRate(thresholdPercent?: number): AlertRule;
    /**
     * Alert on missing spans (no traces)
     */
    static missingSpans(timeWindowSeconds?: number): AlertRule;
    /**
     * Alert on slow operations
     */
    static slowOperations(thresholdMs?: number): AlertRule;
    /**
     * Alert on low throughput
     */
    static lowThroughput(thresholdOpsPerSecond?: number): AlertRule;
}
/**
 * Alert rule type (re-export for convenience)
 */
export type { AlertRule };
//# sourceMappingURL=trace-alerts.d.ts.map