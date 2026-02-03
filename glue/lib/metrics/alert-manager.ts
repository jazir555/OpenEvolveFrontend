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

import { logger, LoggerContext } from '../logger';

export type AlertSeverity = 'info' | 'warning' | 'error' | 'critical';

export interface AlertRule {
  id: string;
  name: string;
  description: string;
  severity: AlertSeverity;
  condition: AlertCondition;
  notifications: NotificationChannel[];
  cooldown?: number; // Cooldown period in ms
  enabled: boolean;
}

export interface AlertCondition {
  type: 'threshold' | 'rate' | 'pattern' | 'custom';
  metric?: string;
  operator?: '>' | '<' | '>=' | '<=' | '==' | '!=';
  threshold?: number;
  window?: number; // Time window in ms
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
  timestamp: string; // ISO-8601 UTC
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
export class AlertManager {
  private rules: Map<string, AlertRule>;
  private activeAlerts: Map<string, Alert>;
  private alertHistory: AlertHistory[];
  private lastTriggered: Map<string, number>;
  private serviceName: string;

  constructor(serviceName: string) {
    this.serviceName = serviceName;
    this.rules = new Map();
    this.activeAlerts = new Map();
    this.alertHistory = [];
    this.lastTriggered = new Map();

    logger.info('Alert manager initialized', {
      service: serviceName,
    });
  }

  /**
   * Register an alert rule
   */
  registerRule(rule: AlertRule): void {
    this.rules.set(rule.id, rule);

    logger.info('Alert rule registered', {
      service: this.serviceName,
      rule_id: rule.id,
      rule_name: rule.name,
      severity: rule.severity,
    });
  }

  /**
   * Unregister an alert rule
   */
  unregisterRule(ruleId: string): void {
    this.rules.delete(ruleId);

    logger.info('Alert rule unregistered', {
      service: this.serviceName,
      rule_id: ruleId,
    });
  }

  /**
   * Update existing rule
   */
  updateRule(ruleId: string, updates: Partial<AlertRule>): void {
    const rule = this.rules.get(ruleId);

    if (!rule) {
      throw new Error(`Alert rule '${ruleId}' not found`);
    }

    const updatedRule = { ...rule, ...updates };
    this.rules.set(ruleId, updatedRule);

    logger.info('Alert rule updated', {
      service: this.serviceName,
      rule_id: ruleId,
    });
  }

  /**
   * Evaluate all rules against data
   */
  async evaluateRules(data: any): Promise<Alert[]> {
    const triggeredAlerts: Alert[] = [];

    for (const [ruleId, rule] of this.rules.entries()) {
      if (!rule.enabled) {
        continue;
      }

      // Check cooldown
      if (this.isInCooldown(ruleId, rule.cooldown)) {
        continue;
      }

      // Evaluate condition
      const triggered = rule.condition.eval(data);

      if (triggered) {
        const alert = await this.triggerAlert(rule, data);
        triggeredAlerts.push(alert);
        this.lastTriggered.set(ruleId, Date.now());
      }
    }

    return triggeredAlerts;
  }

  /**
   * Trigger an alert
   */
  private async triggerAlert(rule: AlertRule, data: any): Promise<Alert> {
    const alertId = `${rule.id}-${Date.now()}`;

    const alert: Alert = {
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

    logger.warn('Alert triggered', {
      alert_id: alertId,
      rule_id: rule.id,
      severity: rule.severity,
      message: alert.message,
    });

    // Send notifications
    await this.sendNotifications(alert, rule.notifications);

    // Add to history
    this.alertHistory.push({
      alert: { ...alert },
      triggered_at: alert.timestamp,
      notification_sent: true,
    });

    return alert;
  }

  /**
   * Generate alert message from rule and data
   */
  private generateAlertMessage(rule: AlertRule, data: any): string {
    if (rule.condition.type === 'threshold' && rule.condition.metric) {
      const value = this.extractMetricValue(data, rule.condition.metric!);
      return `${rule.name}: ${rule.condition.metric} is ${value} (threshold: ${rule.condition.operator} ${rule.condition.threshold})`;
    }

    return rule.description || rule.name;
  }

  /**
   * Extract metric value from data
   */
  private extractMetricValue(data: any, metricPath: string): number {
    const parts = metricPath.split('.');
    let value = data;

    for (const part of parts) {
      value = value?.[part];
    }

    return typeof value === 'number' ? value : 0;
  }

  /**
   * Check if rule is in cooldown period
   */
  private isInCooldown(ruleId: string, cooldown?: number): boolean {
    if (!cooldown) {
      return false;
    }

    const lastTrigger = this.lastTriggered.get(ruleId);
    if (!lastTrigger) {
      return false;
    }

    return Date.now() - lastTrigger < cooldown;
  }

  /**
   * Send alert notifications
   */
  private async sendNotifications(alert: Alert, channels: NotificationChannel[]): Promise<void> {
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

        logger.info('Alert notification sent', {
          alert_id: alert.id,
          channel_type: channel.type,
        });
      } catch (error) {
        logger.error('Failed to send alert notification', error as Error, {
          alert_id: alert.id,
          channel_type: channel.type,
        });
      }
    }
  }

  /**
   * Send webhook notification
   */
  private async sendWebhookNotification(alert: Alert, config: Record<string, any>): Promise<void> {
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

  /**
   * Send email notification
   */
  private async sendEmailNotification(alert: Alert, config: Record<string, any>): Promise<void> {
    // Email sending would be implemented with an email service
    logger.info('Email notification would be sent', {
      alert_id: alert.id,
      to: config.to,
      subject: `[${alert.severity.toUpperCase()}] ${alert.rule_name}`,
    });
  }

  /**
   * Send Slack notification
   */
  private async sendSlackNotification(alert: Alert, config: Record<string, any>): Promise<void> {
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

  /**
   * Log notification
   */
  private logNotification(alert: Alert): void {
    const logMethod = {
      info: logger.info.bind(logger),
      warning: logger.warn.bind(logger),
      error: logger.error.bind(logger),
      critical: logger.error.bind(logger),
    }[alert.severity];

    logMethod(`ALERT: [${alert.severity.toUpperCase()}] ${alert.rule_name}`, {
      alert_id: alert.id,
      message: alert.message,
      data: alert.data,
    });
  }

  /**
   * Acknowledge an alert
   */
  acknowledgeAlert(alertId: string): void {
    const alert = this.activeAlerts.get(alertId);

    if (!alert) {
      throw new Error(`Alert '${alertId}' not found`);
    }

    alert.status = 'acknowledged';

    logger.info('Alert acknowledged', {
      alert_id: alertId,
    });
  }

  /**
   * Resolve an alert
   */
  resolveAlert(alertId: string): void {
    const alert = this.activeAlerts.get(alertId);

    if (!alert) {
      throw new Error(`Alert '${alertId}' not found`);
    }

    alert.status = 'resolved';

    // Update history
    const historyEntry = this.alertHistory.find((h) => h.alert.id === alertId);
    if (historyEntry) {
      historyEntry.resolved_at = new Date().toISOString();
      historyEntry.duration_ms = Date.now() - new Date(alert.timestamp).getTime();
    }

    logger.info('Alert resolved', {
      alert_id: alertId,
      duration_ms: historyEntry?.duration_ms,
    });
  }

  /**
   * Get active alerts
   */
  getActiveAlerts(filter?: { severity?: AlertSeverity }): Alert[] {
    let alerts = Array.from(this.activeAlerts.values());

    if (filter?.severity) {
      alerts = alerts.filter((a) => a.severity === filter.severity);
    }

    return alerts;
  }

  /**
   * Get alert history
   */
  getAlertHistory(limit?: number): AlertHistory[] {
    let history = [...this.alertHistory].reverse();

    if (limit) {
      history = history.slice(0, limit);
    }

    return history;
  }

  /**
   * Get rule statistics
   */
  getRuleStats(): Array<{
    rule_id: string;
    rule_name: string;
    triggered_count: number;
    active_count: number;
  }> {
    const stats = new Map<string, { rule_name: string; triggered_count: number; active_count: number }>();

    // Count from history
    for (const entry of this.alertHistory) {
      const ruleId = entry.alert.rule_id;
      const stat = stats.get(ruleId) || { rule_name: entry.alert.rule_name, triggered_count: 0, active_count: 0 };
      stat.triggered_count++;
      stats.set(ruleId, stat);
    }

    // Count active
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

  /**
   * Clear resolved alerts
   */
  clearResolvedAlerts(): void {
    const before = this.activeAlerts.size;

    for (const [alertId, alert] of this.activeAlerts.entries()) {
      if (alert.status === 'resolved') {
        this.activeAlerts.delete(alertId);
      }
    }

    const cleared = before - this.activeAlerts.size;

    logger.info('Cleared resolved alerts', {
      count: cleared,
    });
  }
}

/**
 * Predefined alert rules
 */
export class AlertRulePresets {
  /**
   * High error rate alert
   */
  static highErrorRate(options: {
    threshold: number; // percentage
    window: number; // time window in ms
    notifications: NotificationChannel[];
  }): AlertRule {
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
      cooldown: 60000, // 1 minute cooldown
      enabled: true,
    };
  }

  /**
   * Circuit breaker open alert
   */
  static circuitBreakerOpen(options: {
    service: string;
    notifications: NotificationChannel[];
  }): AlertRule {
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
      cooldown: 300000, // 5 minute cooldown
      enabled: true,
    };
  }

  /**
   * High latency alert
   */
  static highLatency(options: {
    threshold: number; // milliseconds
    notifications: NotificationChannel[];
  }): AlertRule {
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
      cooldown: 30000, // 30 second cooldown
      enabled: true,
    };
  }

  /**
   * Service unhealthy alert
   */
  static serviceUnhealthy(options: {
    service: string;
    notifications: NotificationChannel[];
  }): AlertRule {
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
      cooldown: 60000, // 1 minute cooldown
      enabled: true,
    };
  }
}

/**
 * Global alert manager instance
 */
let globalAlertManager: AlertManager | null = null;

/**
 * Get or create global alert manager
 */
export function getAlertManager(serviceName?: string): AlertManager {
  if (!globalAlertManager) {
    const name = serviceName || process.env.SERVICE_NAME || 'unknown-service';
    globalAlertManager = new AlertManager(name);
  }
  return globalAlertManager;
}

/**
 * Reset global alert manager (useful for testing)
 */
export function resetAlertManager(): void {
  globalAlertManager = null;
}

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
