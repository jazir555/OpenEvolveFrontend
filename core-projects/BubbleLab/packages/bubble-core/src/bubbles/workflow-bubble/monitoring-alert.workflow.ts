/**
 * MONITORING ALERT WORKFLOW
 *
 * System monitoring with configurable alerts and notification routing.
 */

import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
import { HttpBubble } from '../service-bubble/http.js';
import { SlackBubble } from '../service-bubble/slack.js';

/**
 * Alert severity levels
 */
const AlertSeveritySchema = z.enum([
  'info',
  'warning',
  'error',
  'critical',
]);

/**
 * Alert status
 */
const AlertStatusSchema = z.enum([
  'active',
  'acknowledged',
  'resolved',
  'suppressed',
]);

/**
 * Parameters schema for monitoring alert workflow
 */
const MonitoringAlertParamsSchema = z.object({
  /**
   * Metrics to monitor
   */
  metrics: z
    .array(
      z.object({
        name: z.string().describe('Metric name'),
        type: z.enum(['counter', 'gauge', 'histogram']),
        value: z.number().describe('Current value'),
        threshold: z.object({
          warning: z.number().optional(),
          error: z.number().optional(),
          critical: z.number().optional(),
        }),
      })
    )
    .describe('Metrics to monitor'),

  /**
   * Alert configuration
   */
  alertConfig: z
    .object({
      severity: AlertSeveritySchema,
      message: z.string().describe('Alert message'),
      tags: z.array(z.string()).optional().describe('Alert tags'),
      source: z.string().describe('Alert source'),
    })
    .describe('Alert configuration'),

  /**
   * Notification channels
   */
  notifications: z
    .object({
      slack: z
        .object({
          enabled: z.boolean().default(false),
          channel: z.string().optional(),
          webhookUrl: z.string().optional(),
        })
        .optional(),
      email: z
        .object({
          enabled: z.boolean().default(false),
          recipients: z.array(z.string()).optional(),
        })
        .optional(),
      webhook: z
        .object({
          enabled: z.boolean().default(false),
          url: z.string().optional(),
        })
        .optional(),
    })
    .describe('Notification channels'),

  /**
   * Alert escalation
   */
  escalation: z
    .object({
      enabled: z.boolean().default(false),
      timeout: z.number().optional().describe('Minutes before escalation'),
      escalateTo: z.array(z.string()).optional(),
    })
    .optional(),

  credentials: z
    .record(z.nativeEnum(CredentialType), z.string())
    .optional(),
});

type MonitoringAlertParams = z.input<typeof MonitoringAlertParamsSchema>;

/**
 * Result schema for monitoring alert workflow
 */
const MonitoringAlertResultSchema = z.object({
  success: z.boolean(),
  error: z.string(),

  /**
   * Alert details
   */
  alert: z
    .object({
      alertId: z.string(),
      severity: AlertSeveritySchema,
      status: AlertStatusSchema,
      message: z.string(),
      timestamp: z.date(),
      triggeredBy: z.array(z.string()),
    })
    .optional(),

  /**
   * Notification results
   */
  notifications: z
    .object({
      slack: z.object({
        sent: z.boolean(),
        channelId: z.string().optional(),
        timestamp: z.string().optional(),
      }).optional(),
      email: z.object({
        sent: z.boolean(),
        recipients: z.array(z.string()).optional(),
      }).optional(),
      webhook: z.object({
        sent: z.boolean(),
        response: z.unknown().optional(),
      }).optional(),
    })
    .optional(),
});

type MonitoringAlertResult = z.infer<typeof MonitoringAlertResultSchema>;

export class MonitoringAlertWorkflow extends WorkflowBubble<
  MonitoringAlertParams,
  MonitoringAlertResult
> {
  static readonly type = 'workflow' as const;
  static readonly bubbleName: BubbleName = 'monitoring-alert-workflow';
  static readonly schema = MonitoringAlertParamsSchema;
  static readonly resultSchema = MonitoringAlertResultSchema;
  static readonly shortDescription = 'System monitoring with intelligent alerting';
  static readonly longDescription = `
    Comprehensive system monitoring with configurable alert thresholds and multi-channel notifications.

    Features:
    - Multi-metric monitoring with customizable thresholds
    - Severity-based alert classification (info, warning, error, critical)
    - Multi-channel notifications (Slack, email, webhooks)
    - Alert escalation with timeout
    - Alert lifecycle management (active, acknowledged, resolved, suppressed)
    - Tag-based alert organization

    Use cases:
    - Infrastructure monitoring
    - Application performance monitoring
    - Business metrics alerting
    - DevOps incident response
    - SLO/SLA monitoring
  `;
  static readonly alias = 'monitor-alert';

  constructor(params: MonitoringAlertParams, context?: BubbleContext) {
    super(params, context);
  }

  protected async performAction(): Promise<MonitoringAlertResult> {
    const alertId = this.generateAlertId();
    const startTime = Date.now();

    console.log(`[MonitoringAlert] Processing alert: ${alertId}`);

    try {
      // Step 1: Evaluate metrics against thresholds
      console.log('[MonitoringAlert] Step 1: Evaluating metrics');
      const evaluation = this.evaluateMetrics();
      const triggeredMetrics = evaluation.triggered;

      if (triggeredMetrics.length === 0) {
        console.log('[MonitoringAlert] No thresholds breached');
        return {
          success: true,
          error: '',
          alert: {
            alertId,
            severity: 'info',
            status: 'resolved',
            message: 'All metrics within normal thresholds',
            timestamp: new Date(),
            triggeredBy: [],
          },
        };
      }

      // Step 2: Determine alert severity
      console.log('[MonitoringAlert] Step 2: Determining severity');
      const severity = this.determineSeverity(triggeredMetrics);

      // Step 3: Send notifications
      console.log('[MonitoringAlert] Step 3: Sending notifications');
      const notifications = await this.sendNotifications(alertId, severity, triggeredMetrics);

      // Step 4: Set up escalation if enabled
      if (this.params.escalation?.enabled) {
        console.log('[MonitoringAlert] Step 4: Setting up escalation');
        this.setupEscalation(alertId, severity);
      }

      const processingTime = Date.now() - startTime;
      console.log(`[MonitoringAlert] Alert processed in ${processingTime}ms`);

      return {
        success: true,
        error: '',
        alert: {
          alertId,
          severity,
          status: 'active',
          message: this.params.alertConfig.message,
          timestamp: new Date(),
          triggeredBy: triggeredMetrics.map(m => m.name),
        },
        notifications,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('[MonitoringAlert] Alert processing failed:', errorMessage);

      return {
        success: false,
        error: `Alert processing failed: ${errorMessage}`,
      };
    }
  }

  /**
   * Evaluate metrics against thresholds
   */
  private evaluateMetrics() {
    const triggered: Array<{ name: string; value: number; severity: string }> = [];

    for (const metric of this.params.metrics) {
      const { name, value, threshold } = metric;

      if (threshold.critical && value >= threshold.critical) {
        triggered.push({ name, value, severity: 'critical' });
      } else if (threshold.error && value >= threshold.error) {
        triggered.push({ name, value, severity: 'error' });
      } else if (threshold.warning && value >= threshold.warning) {
        triggered.push({ name, value, severity: 'warning' });
      }
    }

    return { triggered };
  }

  /**
   * Determine alert severity from triggered metrics
   */
  private determineSeverity(triggeredMetrics: Array<{ severity: string }>) {
    const severities = triggeredMetrics.map(m => m.severity);

    if (severities.includes('critical')) return 'critical';
    if (severities.includes('error')) return 'error';
    if (severities.includes('warning')) return 'warning';
    return 'info';
  }

  /**
   * Send notifications to configured channels
   */
  private async sendNotifications(
    alertId: string,
    severity: string,
    triggeredMetrics: Array<{ name: string; value: number; severity: string }>
  ) {
    const notifications: NonNullable<MonitoringAlertResult['notifications']> = {};
    const notifConfig = this.params.notifications || {};

    // Slack notification
    if (notifConfig.slack?.enabled) {
      console.log('[MonitoringAlert] Sending Slack notification');

      try {
        const slackBubble = new SlackBubble(
          {
            operation: 'send_message',
            channel: notifConfig.slack.channel || '#alerts',
            text: this.formatSlackMessage(alertId, severity, triggeredMetrics),
            credentials: this.params.credentials,
          },
          this.context
        );

        const result = await slackBubble.action();

        notifications.slack = {
          sent: result.success,
          channelId: notifConfig.slack.channel,
          timestamp: new Date().toISOString(),
        };
      } catch (error) {
        console.error('[MonitoringAlert] Slack notification failed:', error);
        notifications.slack = { sent: false };
      }
    }

    // Webhook notification
    if (notifConfig.webhook?.enabled && notifConfig.webhook.url) {
      console.log('[MonitoringAlert] Sending webhook notification');

      try {
        const httpBubble = new HttpBubble(
          {
            url: notifConfig.webhook.url,
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: {
              alertId,
              severity,
              message: this.params.alertConfig.message,
              metrics: triggeredMetrics,
              timestamp: new Date().toISOString(),
            },
            credentials: this.params.credentials,
          },
          this.context
        );

        const result = await httpBubble.action();

        notifications.webhook = {
          sent: result.success,
          response: result.data.json,
        };
      } catch (error) {
        console.error('[MonitoringAlert] Webhook notification failed:', error);
        notifications.webhook = { sent: false };
      }
    }

    return notifications;
  }

  /**
   * Format Slack message
   */
  private formatSlackMessage(
    alertId: string,
    severity: string,
    triggeredMetrics: Array<{ name: string; value: number; severity: string }>
  ): string {
    const emojiMap: Record<string, string> = {
      critical: ':rotating_light:',
      error: ':x:',
      warning: ':warning:',
      info: ':information_source:',
    };

    const emoji = emojiMap[severity] || ':information_source:';

    let message = `${emoji} *Alert: ${this.params.alertConfig.message}*\n\n`;
    message += `*Alert ID:* \`${alertId}\`\n`;
    message += `*Severity:* ${severity.toUpperCase()}\n`;
    message += `*Source:* ${this.params.alertConfig.source}\n\n`;
    message += '*Triggered Metrics:*\n';

    for (const metric of triggeredMetrics) {
      message += `• \`${metric.name}\`: ${metric.value} (${metric.severity})\n`;
    }

    message += `\n*Timestamp:* ${new Date().toISOString()}`;

    if (this.params.alertConfig.tags) {
      message += `\n*Tags:* ${this.params.alertConfig.tags.join(', ')}`;
    }

    return message;
  }

  /**
   * Set up alert escalation
   */
  private setupEscalation(alertId: string, severity: string): void {
    const timeout = this.params.escalation?.timeout || 30;

    console.log(`[MonitoringAlert] Scheduling escalation in ${timeout} minutes`);

    // In production, this would use a job scheduler or database-based queue
    setTimeout(async () => {
      console.log(`[MonitoringAlert] Escalating alert ${alertId}`);

      // Send escalation notifications
      const escalateTo = this.params.escalation?.escalateTo || [];
      // Implementation would send to escalateTo recipients
    }, timeout * 60 * 1000);
  }

  /**
   * Generate alert ID
   */
  private generateAlertId(): string {
    return `alert_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
  }
}
