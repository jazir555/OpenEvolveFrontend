"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.AlertRulePresets = exports.AlertManager = void 0;
exports.getAlertManager = getAlertManager;
exports.resetAlertManager = resetAlertManager;
const logger_1 = require("../logger");
class AlertManager {
    constructor(serviceName) {
        this.serviceName = serviceName;
        this.rules = new Map();
        this.activeAlerts = new Map();
        this.alertHistory = [];
        this.lastTriggered = new Map();
        logger_1.logger.info('Alert manager initialized', {
            service: serviceName,
        });
    }
    registerRule(rule) {
        this.rules.set(rule.id, rule);
        logger_1.logger.info('Alert rule registered', {
            service: this.serviceName,
            rule_id: rule.id,
            rule_name: rule.name,
            severity: rule.severity,
        });
    }
    unregisterRule(ruleId) {
        this.rules.delete(ruleId);
        logger_1.logger.info('Alert rule unregistered', {
            service: this.serviceName,
            rule_id: ruleId,
        });
    }
    updateRule(ruleId, updates) {
        const rule = this.rules.get(ruleId);
        if (!rule) {
            throw new Error(`Alert rule '${ruleId}' not found`);
        }
        const updatedRule = { ...rule, ...updates };
        this.rules.set(ruleId, updatedRule);
        logger_1.logger.info('Alert rule updated', {
            service: this.serviceName,
            rule_id: ruleId,
        });
    }
    async evaluateRules(data) {
        const triggeredAlerts = [];
        for (const [ruleId, rule] of this.rules.entries()) {
            if (!rule.enabled) {
                continue;
            }
            if (this.isInCooldown(ruleId, rule.cooldown)) {
                continue;
            }
            const triggered = rule.condition.eval(data);
            if (triggered) {
                const alert = await this.triggerAlert(rule, data);
                triggeredAlerts.push(alert);
                this.lastTriggered.set(ruleId, Date.now());
            }
        }
        return triggeredAlerts;
    }
    async triggerAlert(rule, data) {
        const alertId = `${rule.id}-${Date.now()}`;
        const alert = {
            id: alertId,
            rule_id: rule.id,
            rule_name: rule.name,
            severity: rule.severity,
            message: this.generateAlertMessage(rule, data),
            timestamp: new Date().toISOString(),
            data,
            status: 'firing',
            labels: {
                service: this.serviceName,
                rule_id: rule.id,
                severity: rule.severity,
            },
        };
        this.activeAlerts.set(alertId, alert);
        logger_1.logger.warn('Alert triggered', {
            alert_id: alertId,
            rule_id: rule.id,
            severity: rule.severity,
            message: alert.message,
        });
        await this.sendNotifications(alert, rule.notifications);
        this.alertHistory.push({
            alert: { ...alert },
            triggered_at: alert.timestamp,
            notification_sent: true,
        });
        return alert;
    }
    generateAlertMessage(rule, data) {
        if (rule.condition.type === 'threshold' && rule.condition.metric) {
            const value = this.extractMetricValue(data, rule.condition.metric);
            return `${rule.name}: ${rule.condition.metric} is ${value} (threshold: ${rule.condition.operator} ${rule.condition.threshold})`;
        }
        return rule.description || rule.name;
    }
    extractMetricValue(data, metricPath) {
        const parts = metricPath.split('.');
        let value = data;
        for (const part of parts) {
            value = value?.[part];
        }
        return typeof value === 'number' ? value : 0;
    }
    isInCooldown(ruleId, cooldown) {
        if (!cooldown) {
            return false;
        }
        const lastTrigger = this.lastTriggered.get(ruleId);
        if (!lastTrigger) {
            return false;
        }
        return Date.now() - lastTrigger < cooldown;
    }
    async sendNotifications(alert, channels) {
        for (const channel of channels) {
            try {
                switch (channel.type) {
                    case 'webhook':
                        await this.sendWebhookNotification(alert, channel.config);
                        break;
                    case 'email':
                        await this.sendEmailNotification(alert, channel.config);
                        break;
                    case 'slack':
                        await this.sendSlackNotification(alert, channel.config);
                        break;
                    case 'log':
                        this.logNotification(alert);
                        break;
                }
                logger_1.logger.info('Alert notification sent', {
                    alert_id: alert.id,
                    channel_type: channel.type,
                });
            }
            catch (error) {
                logger_1.logger.error('Failed to send alert notification', error, {
                    alert_id: alert.id,
                    channel_type: channel.type,
                });
            }
        }
    }
    async sendWebhookNotification(alert, config) {
        const url = config.url;
        if (!url) {
            throw new Error('Webhook URL not configured');
        }
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...config.headers,
            },
            body: JSON.stringify({
                alert,
                service: this.serviceName,
                timestamp: new Date().toISOString(),
            }),
        });
        if (!response.ok) {
            throw new Error(`Webhook returned ${response.status}`);
        }
    }
    async sendEmailNotification(alert, config) {
        logger_1.logger.info('Email notification would be sent', {
            alert_id: alert.id,
            to: config.to,
            subject: `[${alert.severity.toUpperCase()}] ${alert.rule_name}`,
        });
    }
    async sendSlackNotification(alert, config) {
        const webhookUrl = config.webhook_url;
        if (!webhookUrl) {
            throw new Error('Slack webhook URL not configured');
        }
        const color = {
            info: '#36a64f',
            warning: '#ff9900',
            error: '#ff0000',
            critical: '#000000',
        }[alert.severity];
        const response = await fetch(webhookUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                attachments: [
                    {
                        color,
                        title: `[${alert.severity.toUpperCase()}] ${alert.rule_name}`,
                        text: alert.message,
                        fields: [
                            {
                                title: 'Service',
                                value: this.serviceName,
                                short: true,
                            },
                            {
                                title: 'Severity',
                                value: alert.severity,
                                short: true,
                            },
                            {
                                title: 'Timestamp',
                                value: alert.timestamp,
                                short: true,
                            },
                        ],
                        footer: 'OpenEvolve Alert Manager',
                    },
                ],
            }),
        });
        if (!response.ok) {
            throw new Error(`Slack webhook returned ${response.status}`);
        }
    }
    logNotification(alert) {
        const logMethod = {
            info: logger_1.logger.info.bind(logger_1.logger),
            warning: logger_1.logger.warn.bind(logger_1.logger),
            error: logger_1.logger.error.bind(logger_1.logger),
            critical: logger_1.logger.error.bind(logger_1.logger),
        }[alert.severity];
        logMethod(`ALERT: [${alert.severity.toUpperCase()}] ${alert.rule_name}`, {
            alert_id: alert.id,
            message: alert.message,
            data: alert.data,
        });
    }
    acknowledgeAlert(alertId) {
        const alert = this.activeAlerts.get(alertId);
        if (!alert) {
            throw new Error(`Alert '${alertId}' not found`);
        }
        alert.status = 'acknowledged';
        logger_1.logger.info('Alert acknowledged', {
            alert_id: alertId,
        });
    }
    resolveAlert(alertId) {
        const alert = this.activeAlerts.get(alertId);
        if (!alert) {
            throw new Error(`Alert '${alertId}' not found`);
        }
        alert.status = 'resolved';
        const historyEntry = this.alertHistory.find((h) => h.alert.id === alertId);
        if (historyEntry) {
            historyEntry.resolved_at = new Date().toISOString();
            historyEntry.duration_ms = Date.now() - new Date(alert.timestamp).getTime();
        }
        logger_1.logger.info('Alert resolved', {
            alert_id: alertId,
            duration_ms: historyEntry?.duration_ms,
        });
    }
    getActiveAlerts(filter) {
        let alerts = Array.from(this.activeAlerts.values());
        if (filter?.severity) {
            alerts = alerts.filter((a) => a.severity === filter.severity);
        }
        return alerts;
    }
    getAlertHistory(limit) {
        let history = [...this.alertHistory].reverse();
        if (limit) {
            history = history.slice(0, limit);
        }
        return history;
    }
    getRuleStats() {
        const stats = new Map();
        for (const entry of this.alertHistory) {
            const ruleId = entry.alert.rule_id;
            const stat = stats.get(ruleId) || { rule_name: entry.alert.rule_name, triggered_count: 0, active_count: 0 };
            stat.triggered_count++;
            stats.set(ruleId, stat);
        }
        for (const alert of this.activeAlerts.values()) {
            const ruleId = alert.rule_id;
            const stat = stats.get(ruleId) || { rule_name: alert.rule_name, triggered_count: 0, active_count: 0 };
            stat.active_count++;
            stats.set(ruleId, stat);
        }
        return Array.from(stats.entries()).map(([rule_id, stat]) => ({
            rule_id,
            ...stat,
        }));
    }
    clearResolvedAlerts() {
        const before = this.activeAlerts.size;
        for (const [alertId, alert] of this.activeAlerts.entries()) {
            if (alert.status === 'resolved') {
                this.activeAlerts.delete(alertId);
            }
        }
        const cleared = before - this.activeAlerts.size;
        logger_1.logger.info('Cleared resolved alerts', {
            count: cleared,
        });
    }
}
exports.AlertManager = AlertManager;
class AlertRulePresets {
    static highErrorRate(options) {
        return {
            id: 'high-error-rate',
            name: 'High Error Rate',
            description: `Error rate exceeds ${options.threshold}% over ${options.window}ms`,
            severity: 'error',
            condition: {
                type: 'threshold',
                metric: 'error_rate',
                operator: '>',
                threshold: options.threshold,
                window: options.window,
                eval: (data) => {
                    const errorRate = data.error_rate || 0;
                    return errorRate > options.threshold;
                },
            },
            notifications: options.notifications,
            cooldown: 60000,
            enabled: true,
        };
    }
    static circuitBreakerOpen(options) {
        return {
            id: `circuit-breaker-open-${options.service}`,
            name: `Circuit Breaker Open: ${options.service}`,
            description: `Circuit breaker has opened for ${options.service}`,
            severity: 'critical',
            condition: {
                type: 'pattern',
                eval: (data) => {
                    return data.circuit_state === 'open';
                },
            },
            notifications: options.notifications,
            cooldown: 300000,
            enabled: true,
        };
    }
    static highLatency(options) {
        return {
            id: 'high-latency',
            name: 'High Latency',
            description: `Request latency exceeds ${options.threshold}ms`,
            severity: 'warning',
            condition: {
                type: 'threshold',
                metric: 'latency_p95',
                operator: '>',
                threshold: options.threshold,
                eval: (data) => {
                    const latency = data.latency_p95 || data.latency_avg || 0;
                    return latency > options.threshold;
                },
            },
            notifications: options.notifications,
            cooldown: 30000,
            enabled: true,
        };
    }
    static serviceUnhealthy(options) {
        return {
            id: `service-unhealthy-${options.service}`,
            name: `Service Unhealthy: ${options.service}`,
            description: `Health check failed for ${options.service}`,
            severity: 'critical',
            condition: {
                type: 'pattern',
                eval: (data) => {
                    return data.health_status === 'unhealthy';
                },
            },
            notifications: options.notifications,
            cooldown: 60000,
            enabled: true,
        };
    }
}
exports.AlertRulePresets = AlertRulePresets;
let globalAlertManager = null;
function getAlertManager(serviceName) {
    if (!globalAlertManager) {
        const name = serviceName || process.env.SERVICE_NAME || 'unknown-service';
        globalAlertManager = new AlertManager(name);
    }
    return globalAlertManager;
}
function resetAlertManager() {
    globalAlertManager = null;
}
//# sourceMappingURL=alert-manager.js.map