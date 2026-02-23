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
export declare class AlertManager {
    private rules;
    private activeAlerts;
    private alertHistory;
    private lastTriggered;
    private serviceName;
    constructor(serviceName: string);
    registerRule(rule: AlertRule): void;
    unregisterRule(ruleId: string): void;
    updateRule(ruleId: string, updates: Partial<AlertRule>): void;
    evaluateRules(data: any): Promise<Alert[]>;
    private triggerAlert;
    private generateAlertMessage;
    private extractMetricValue;
    private isInCooldown;
    private sendNotifications;
    private sendWebhookNotification;
    private sendEmailNotification;
    private sendSlackNotification;
    private logNotification;
    acknowledgeAlert(alertId: string): void;
    resolveAlert(alertId: string): void;
    getActiveAlerts(filter?: {
        severity?: AlertSeverity;
    }): Alert[];
    getAlertHistory(limit?: number): AlertHistory[];
    getRuleStats(): Array<{
        rule_id: string;
        rule_name: string;
        triggered_count: number;
        active_count: number;
    }>;
    clearResolvedAlerts(): void;
}
export declare class AlertRulePresets {
    static highErrorRate(options: {
        threshold: number;
        window: number;
        notifications: NotificationChannel[];
    }): AlertRule;
    static circuitBreakerOpen(options: {
        service: string;
        notifications: NotificationChannel[];
    }): AlertRule;
    static highLatency(options: {
        threshold: number;
        notifications: NotificationChannel[];
    }): AlertRule;
    static serviceUnhealthy(options: {
        service: string;
        notifications: NotificationChannel[];
    }): AlertRule;
}
export declare function getAlertManager(serviceName?: string): AlertManager;
export declare function resetAlertManager(): void;
//# sourceMappingURL=alert-manager.d.ts.map