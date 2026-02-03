/**
 * Resource Scaling Automation
 * Purpose: Auto-scale resources based on metrics and load
 * Category: Infrastructure Automation
 * Event Type: schedule/cron
 * Schedule: */10 * * * * (Every 10 minutes)
 *
 * Required Credentials:
 * - KUBERNETES_CONFIG: Kubernetes cluster configuration
 * - PROMETHEUS_URL: Prometheus metrics endpoint
 * - API_KEY: API key for authentication (required)
 * - SLACK_WEBHOOK_URL: Slack webhook for notifications (optional)
 *
 * Security Fixes Applied (Wave 5):
 * - Environment variable validation at startup
 * - API key authentication
 * - Rate limiting
 * - Input validation for all user inputs
 * - Error message sanitization
 * - Structured logging with correlation IDs
 * - URL validation for all endpoints
 */

import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  SlackBubble,
  type CronEvent
} from '@bubblelab/bubble-core';
import {
  validateEnvironment,
  authenticateRequest,
  requireAuthentication,
  RateLimiter,
  InputValidator,
  sanitizeError,
  StructuredLogger,
  generateCorrelationId,
  SecuritySchemas,
} from '../security-utils';

interface ResourceMetrics {
  cpuUsage: number;
  memoryUsage: number;
  requestCount: number;
  responseTime: number;
}

interface ScalingDecision {
  service: string;
  action: 'scale_up' | 'scale_down' | 'no_action';
  currentReplicas: number;
  targetReplicas: number;
  reason: string;
}

interface ScalingResult {
  timestamp: string;
  services: ScalingDecision[];
  totalScaled: number;
  correlationId: string;
}

// Security: Environment variable validation
validateEnvironment({
  required: ['KUBERNETES_API', 'KUBERNETES_TOKEN', 'PROMETHEUS_URL', 'API_KEY'],
  schemas: {
    API_KEY: SecuritySchemas.apiKey,
    KUBERNETES_API: SecuritySchemas.url,
    PROMETHEUS_URL: SecuritySchemas.url,
  },
});

export class ResourceScalingAutomation extends BubbleFlow<'schedule/cron'> {
  readonly cronSchedule = '*/10 * * * *';
  readonly name = 'Resource Scaling Automation';
  readonly description = 'Auto-scale resources based on metrics and load';

  private logger = new StructuredLogger('resource-scaling-automation');
  private rateLimiter = new RateLimiter({
    maxRequests: 6, // 6 scaling operations per minute (one every 10 seconds)
    windowMs: 60000, // 1 minute
  });

  async handle(payload: CronEvent): Promise<ScalingResult> {
    // Security: Generate correlation ID for tracing
    const correlationId = generateCorrelationId();
    this.logger = this.logger.child({ correlationId });

    const timestamp = new Date().toISOString();
    const scalingDecisions: ScalingDecision[] = [];

    // Security: Rate limiting check
    if (!this.rateLimiter.checkLimit(correlationId)) {
      this.logger.warn({
        msg: 'Rate limit exceeded for scaling operations',
      });
      return {
        timestamp,
        services: [],
        totalScaled: 0,
        correlationId,
      };
    }

    // Security: API key authentication
    const authContext = authenticateRequest(
      payload.headers?.['x-api-key'],
      process.env.API_KEY,
      { correlationId, ip: payload.headers?.['x-forwarded-for'] }
    );
    requireAuthentication(authContext);

    this.logger.info({
      msg: 'Starting resource scaling check',
      timestamp,
    });

    // Services to monitor and scale
    const services = [
      { name: 'openevolve-api', minReplicas: 2, maxReplicas: 10, targetCPU: 70, targetMemory: 80 },
      { name: 'openevolve-worker', minReplicas: 1, maxReplicas: 5, targetCPU: 75, targetMemory: 85 },
      { name: 'qdrant', minReplicas: 1, maxReplicas: 3, targetCPU: 80, targetMemory: 85 },
    ];

    for (const serviceConfig of services) {
      try {
        const validatedServiceName = InputValidator.validateServiceName(serviceConfig.name);

        // Step 1: Get current metrics from Prometheus
        const metricsQuery = InputValidator.sanitizeString(
          `avg(rate(container_cpu_usage_seconds_total{pod=~"${validatedServiceName}-.*"}[5m])) * 100`,
          500
        );

        const validatedPrometheusUrl = InputValidator.validateUrl(process.env.PROMETHEUS_URL);

        const cpuMetrics = new HttpBubble({
          url: InputValidator.validateUrl(`${validatedPrometheusUrl}/api/v1/query`),
          method: 'GET',
          params: {
            query: metricsQuery,
          },
          timeout: 10000,
        });

        const cpuResult = await cpuMetrics.action();
        const cpuUsage = InputValidator.sanitizeNumber(parseFloat(cpuResult.data.data.result[0]?.value[1]) || 0, 0, 100);

        // Get memory metrics
        const memoryQuery = InputValidator.sanitizeString(
          `avg(container_memory_usage_bytes{pod=~"${validatedServiceName}-.*"} / container_spec_memory_limit_bytes{pod=~"${validatedServiceName}-.*"}) * 100`,
          500
        );

        const memoryMetrics = new HttpBubble({
          url: InputValidator.validateUrl(`${validatedPrometheusUrl}/api/v1/query`),
          method: 'GET',
          params: {
            query: memoryQuery,
          },
          timeout: 10000,
        });

        const memoryResult = await memoryMetrics.action();
        const memoryUsage = InputValidator.sanitizeNumber(parseFloat(memoryResult.data.data.result[0]?.value[1]) || 0, 0, 100);

        // Get request rate
        const requestQuery = InputValidator.sanitizeString(
          `sum(rate(http_requests_total{service="${validatedServiceName}"}[5m]))`,
          500
        );

        const requestMetrics = new HttpBubble({
          url: InputValidator.validateUrl(`${validatedPrometheusUrl}/api/v1/query`),
          method: 'GET',
          params: {
            query: requestQuery,
          },
          timeout: 10000,
        });

        const requestResult = await requestMetrics.action();
        const requestCount = InputValidator.sanitizeNumber(parseFloat(requestResult.data.data.result[0]?.value[1]) || 0);

        // Step 2: Get current replica count
        const validatedKubernetesApi = InputValidator.validateUrl(process.env.KUBERNETES_API);

        const getReplicas = new HttpBubble({
          url: InputValidator.validateUrl(`${validatedKubernetesApi}/apis/apps/v1/namespaces/default/deployments/${validatedServiceName}`),
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${process.env.KUBERNETES_TOKEN}`,
          },
          timeout: 10000,
        });

        const replicaResult = await getReplicas.action();
        const currentReplicas = InputValidator.sanitizeNumber(replicaResult.data.spec.replicas, 1, 100);

        // Step 3: Make scaling decision
        const metrics: ResourceMetrics = {
          cpuUsage,
          memoryUsage,
          requestCount,
          responseTime: 0, // Could fetch from Prometheus
        };

        let action: 'scale_up' | 'scale_down' | 'no_action' = 'no_action';
        let targetReplicas = currentReplicas;
        let reason = '';

        // Scale up if CPU or memory is high
        if (metrics.cpuUsage > serviceConfig.targetCPU || metrics.memoryUsage > serviceConfig.targetMemory) {
          action = 'scale_up';
          targetReplicas = Math.min(currentReplicas + 1, serviceConfig.maxReplicas);
          reason = `High CPU (${cpuUsage.toFixed(1)}%) or Memory (${memoryUsage.toFixed(1)}%)`;
        }
        // Scale down if resources are underutilized
        else if (metrics.cpuUsage < 30 && metrics.memoryUsage < 50 && currentReplicas > serviceConfig.minReplicas) {
          action = 'scale_down';
          targetReplicas = Math.max(currentReplicas - 1, serviceConfig.minReplicas);
          reason = `Low utilization - CPU: ${cpuUsage.toFixed(1)}%, Memory: ${memoryUsage.toFixed(1)}%`;
        }

        // Step 4: Execute scaling
        if (action !== 'no_action' && targetReplicas !== currentReplicas) {
          const scaleDeployment = new HttpBubble({
            url: InputValidator.validateUrl(`${validatedKubernetesApi}/apis/apps/v1/namespaces/default/deployments/${validatedServiceName}/scale`),
            method: 'PATCH',
            headers: {
              'Authorization': `Bearer ${process.env.KUBERNETES_TOKEN}`,
              'Content-Type': 'application/merge-patch+json',
            },
            body: {
              spec: {
                replicas: targetReplicas,
              },
            },
            timeout: 30000,
          });

          await scaleDeployment.action();

          scalingDecisions.push({
            service: validatedServiceName,
            action,
            currentReplicas,
            targetReplicas,
            reason: InputValidator.sanitizeString(reason, 500),
          });

          this.logger.info({
            msg: 'Scaled service',
            service: validatedServiceName,
            action,
            currentReplicas,
            targetReplicas,
            reason,
          });
        }

      } catch (error) {
        const sanitizedError = sanitizeError(error);
        this.logger.error({
          msg: 'Error scaling service',
          service: serviceConfig.name,
        }, error);
      }
    }

    const result: ScalingResult = {
      timestamp,
      services: scalingDecisions,
      totalScaled: scalingDecisions.length,
      correlationId,
    };

    // Step 5: Send notification if scaling occurred
    if (result.totalScaled > 0 && process.env.SLACK_WEBHOOK_URL) {
      try {
        const message = InputValidator.sanitizeString(`
📊 Resource Scaling Report

Timestamp: ${timestamp}
Services Scaled: ${result.totalScaled}

${scalingDecisions.map(decision => `
${decision.action === 'scale_up' ? '⬆️' : '⬇️'} ${decision.service}
  ${decision.currentReplicas} → ${decision.targetReplicas} replicas
  Reason: ${decision.reason}
`).join('\n')}
        `.trim(), 10000);

        const slack = new SlackBubble({
          webhookUrl: process.env.SLACK_WEBHOOK_URL,
          message,
        });

        await slack.action();
      } catch (slackError) {
        this.logger.warn({
          msg: 'Slack notification failed',
        }, slackError);
      }
    }

    this.logger.info({
      msg: 'Scaling check completed',
      totalScaled: result.totalScaled,
    });

    return result;
  }
}

export default ResourceScalingAutomation;
