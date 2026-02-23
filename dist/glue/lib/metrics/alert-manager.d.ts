/**
 * Alert Manager
 *
 * Follows the Federation Constitution:
 * - Law of Configuration Explicitness: Alert config via environment variables
 * - Failure Management: Alert on thresholds and anomalies
 * - Observability: Alert notifications and history
 *
 * Features:
 * - Alert rule definitions
 * - Threshold monitoring
 * - Alert notifications (webhook, email)
 * - Alert aggregation
 * - Alert history
 */
export type AlertSeverity = 'info' | 'warning' | 'error' | 'critical';
export interface AlertRule {
    id: string;
    name: string;
    description: string;
    severity: AlertSeverity;
    condition: AlertCondition;
    notifications: NotificationChannel[];
    cooldown?: number;
    enabled: boolean;
}
export interface AlertCondition {
    type: 'threshold' | 'rate' | 'pattern' | 'custom';
    metric?: string;
    operator?: '>' | '<' | '>=' | '<=' | '==' | '!=';
    threshold?: number;
    window?: number;
    eval: (data: any) => boolean;
}
export interface NotificationChannel {
    type: 'webhook' | 'email' | 'slack' | 'log';
    config: Record<string, any>;
}
export interface Alert {
    id: string;
    rule_id: string;
    rule_name: string;
    severity: AlertSeverity;
    message: string;
    timestamp: string;
    data: Record<string, any>;
    status: 'firing' | 'resolved' | 'acknowledged';
    labels: Record<string, string>;
}
export interface AlertHistory {
    alert: Alert;
    triggered_at: string;
    resolved_at?: string;
    duration_ms?: number;
    notification_sent: boolean;
}
/**
 * Alert Manager class
 *
 * Manages alert rules, evaluation, and notifications
 */
export declare class AlertManager {
    private rules;
    private activeAlerts;
    private alertHistory;
    private lastTriggered;
    private serviceName;
    constructor(serviceName: string);
    /**
     * Register an alert rule
     */
    registerRule(rule: AlertRule): void;
    /**
     * Unregister an alert rule
     */
    unregisterRule(ruleId: string): void;
    /**
     * Update existing rule
     */
    updateRule(ruleId: string, updates: Partial<AlertRule>): void;
    /**
     * Evaluate all rules against data
     */
    evaluateRules(data: any): Promise<Alert[]>;
    /**
     * Trigger an alert
     */
    private triggerAlert;
    /**
     * Generate alert message from rule and data
     */
    private generateAlertMessage;
    /**
     * Extract metric value from data
     */
    private extractMetricValue;
    /**
     * Check if rule is in cooldown period
     */
    private isInCooldown;
    /**
     * Send alert notifications
     */
    private sendNotifications;
    /**
     * Send webhook notification
     */
    private sendWebhookNotification;
    /**
     * Send email notification
     */
    private sendEmailNotification;
    /**
     * Send Slack notification
     */
    private sendSlackNotification;
    /**
     * Log notification
     */
    private logNotification;
    /**
     * Acknowledge an alert
     */
    acknowledgeAlert(alertId: string): void;
    /**
     * Resolve an alert
     */
    resolveAlert(alertId: string): void;
    /**
     * Get active alerts
     */
    getActiveAlerts(filter?: {
        severity?: AlertSeverity;
    }): Alert[];
    /**
     * Get alert history
     */
    getAlertHistory(limit?: number): AlertHistory[];
    /**
     * Get rule statistics
     */
    getRuleStats(): Array<{
        rule_id: string;
        rule_name: string;
        triggered_count: number;
        active_count: number;
    }>;
    /**
     * Clear resolved alerts
     */
    clearResolvedAlerts(): void;
}
/**
 * Predefined alert rules
 */
export declare class AlertRulePresets {
    /**
     * High error rate alert
     */
    static highErrorRate(options: {
        threshold: number;
        window: number;
        notifications: NotificationChannel[];
    }): AlertRule;
    /**
     * Circuit breaker open alert
     */
    static circuitBreakerOpen(options: {
        service: string;
        notifications: NotificationChannel[];
    }): AlertRule;
    /**
     * High latency alert
     */
    static highLatency(options: {
        threshold: number;
        notifications: NotificationChannel[];
    }): AlertRule;
    /**
     * Service unhealthy alert
     */
    static serviceUnhealthy(options: {
        service: string;
        notifications: NotificationChannel[];
    }): AlertRule;
}
/**
 * Get or create global alert manager
 */
export declare function getAlertManager(serviceName?: string): AlertManager;
/**
 * Reset global alert manager (useful for testing)
 */
export declare function resetAlertManager(): void;
/**
 * Example usage:
 *
 * ```typescript
 * import { getAlertManager, AlertRulePresets } from './alert-manager';
 *
 * const alertManager = getAlertManager('crm-adapter');
 *
 * // Register predefined rules
 * alertManager.registerRule(AlertRulePresets.highErrorRate({
 *   threshold: 5, // 5%
 *   window: 60000, // 1 minute
 *   notifications: [
 *     { type: 'log', config: {} },
 *     { type: 'webhook', config: { url: 'https://hooks.example.com/alert' } },
 *   ],
 * }));
 *
 * alertManager.registerRule(AlertRulePresets.circuitBreakerOpen({
 *   service: 'crm-adapter',
 *   notifications: [
 *     { type: 'slack', config: { webhook_url: 'https://hooks.slack.com/...' } },
 *   ],
 * }));
 *
 * // Register custom rule
 * alertManager.registerRule({
 *   id: 'custom-rule',
 *   name: 'Custom Rule',
 *   description: 'Monitors custom condition',
 *   severity: 'warning',
 *   condition: {
 *     type: 'custom',
 *     eval: (data) => {
 *       return data.custom_metric > 100;
 *     },
 *   },
 *   notifications: [
 *     { type: 'log', config: {} },
 *   ],
 *   cooldown: 30000,
 *   enabled: true,
 * });
 *
 * // Evaluate rules
 * const alerts = await alertManager.evaluateRules({
 *   error_rate: 7,
 *   latency_p95: 2500,
 *   health_status: 'healthy',
 * });
 *
 * // Get active alerts
 * const activeAlerts = alertManager.getActiveAlerts({ severity: 'critical' });
 *
 * // Acknowledge alert
 * alertManager.acknowledgeAlert(alerts[0].id);
 *
 * // Resolve alert
 * alertManager.resolveAlert(alerts[0].id);
 * ```
 */
//# sourceMappingURL=alert-manager.d.ts.map