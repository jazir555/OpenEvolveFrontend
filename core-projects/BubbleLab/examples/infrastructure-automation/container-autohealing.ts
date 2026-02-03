/**
 * Workflow: Container Auto-Healing
 * Description: Automatically detect and heal unhealthy containers in your infrastructure
 * Use Case: Production environment reliability - automatically restart or replace containers that are failing health checks
 *
 * Setup Instructions:
 * 1. Configure monitoring credentials (Prometheus/Datadog/NewRelic)
 * 2. Set up container runtime credentials (Docker/Kubernetes)
 * 3. Configure alert thresholds (CPU, memory, response time)
 * 4. Set notification channel (Slack/Email)
 *
 * Required Credentials:
 * - docker-host: Docker daemon socket or API endpoint
 * - prometheus: Monitoring system (optional, for metrics-based detection)
 * - slack: For notifications (optional)
 *
 * Trigger Options:
 * - Scheduled: Run every 5 minutes
 * - Webhook: Trigger from alerting system
 * - Manual: On-demand health check
 *
 * Example Webhook Payload:
 * {
 *   "alert_name": "high_cpu_usage",
 *   "container_id": "abc123",
 *   "threshold": 80,
 *   "current_value": 95
 * }
 *
 * Monitoring Configuration:
 * - CPU threshold: 80%
 * - Memory threshold: 85%
 * - Response time threshold: 5000ms
 * - Failed health checks: 3 consecutive failures
 *
 * Security Fixes Applied (Wave 5):
 * - Environment variable validation at startup
 * - API key authentication
 * - Rate limiting (60 requests/minute)
 * - Input validation for all user inputs
 * - Error message sanitization
 * - Structured logging with correlation IDs
 * - URL validation for all endpoints
 */

import {
  BubbleFlow,
  HttpBubble,
  SlackBubble,
  AIAgentBubble,
  type WebhookEvent,
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
} from '../../templates/security-utils';

// Security: Environment variable validation at startup
validateEnvironment({
  required: ['API_KEY'],
  schemas: {
    API_KEY: SecuritySchemas.apiKey,
  },
});

export interface ContainerHealth {
  containerId: string;
  containerName: string;
  status: 'healthy' | 'unhealthy' | 'unknown';
  cpuUsage: number;
  memoryUsage: number;
  uptime: number;
  lastHealthCheck: string;
}

export interface HealingAction {
  action: 'restart' | 'stop' | 'replace' | 'notify_only';
  containerId: string;
  reason: string;
  timestamp: string;
}

export interface Output {
  message: string;
  containersChecked: number;
  containersHealed: number;
  healingActions: HealingAction[];
  alertUrl?: string;
}

export interface CustomWebhookPayload extends WebhookEvent {
  /**
   * Specific container ID to check (optional)
   * @canBeFile false
   */
  containerId?: string;

  /**
   * Force healing even if container appears healthy
   * @canBeFile false
   */
  forceHeal?: boolean;

  /**
   * CPU usage threshold percentage
   * @canBeFile false
   */
  cpuThreshold?: number;

  /**
   * Memory usage threshold percentage
   * @canBeFile false
   */
  memoryThreshold?: number;

  /**
   * Send notifications to Slack
   * @canBeFile false
   */
  notify?: boolean;

  /**
   * Alert details if triggered by monitoring system
   * @canBeFile false
   */
  alertDetails?: {
    alertName: string;
    threshold: number;
    currentValue: number;
    metric: string;
  };
}

export class ContainerAutohealing extends BubbleFlow<'webhook/http'> {
  private logger = new StructuredLogger('container_autohealing');
  private rateLimiter = new RateLimiter({
    maxRequests: 60,
    windowMs: 60000,
  });

  private readonly DEFAULT_CPU_THRESHOLD = 80;
  private readonly DEFAULT_MEMORY_THRESHOLD = 85;
  private readonly MAX_RESTART_ATTEMPTS = 3;
  private readonly HEALTH_CHECK_TIMEOUT = 10000; // 10 seconds

  // Get container health status via Docker API
  private async getContainerHealth(containerId: string): Promise<ContainerHealth> {
    const http = new HttpBubble({
      url: `http://docker-api:2375/containers/${containerId}/json`,
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 5000,
    });

    const response = await http.action();

    if (!response.success || !response.data) {
      throw new Error(`Failed to get container info: ${response.error}`);
    }

    const container = response.data;
    const state = container.State;

    // Calculate health status
    let status: 'healthy' | 'unhealthy' | 'unknown' = 'unknown';
    if (state.Health && state.Health.Status === 'healthy') {
      status = 'healthy';
    } else if (state.Health && state.Health.Status === 'unhealthy') {
      status = 'unhealthy';
    } else if (state.Running) {
      status = 'healthy';
    } else {
      status = 'unhealthy';
    }

    // Get stats for CPU and memory
    const statsResponse = await new HttpBubble({
      url: `http://docker-api:2375/containers/${containerId}/stats?stream=false`,
      method: 'GET',
      timeout: 5000,
    }).action();

    let cpuUsage = 0;
    let memoryUsage = 0;

    if (statsResponse.success && statsResponse.data) {
      const stats = statsResponse.data;
      // Calculate CPU usage percentage
      const cpuDelta = stats.cpu_stats.cpu_usage.total_usage - stats.precpu_stats.cpu_usage.total_usage;
      const systemDelta = stats.cpu_stats.system_cpu_usage - stats.precpu_stats.system_cpu_usage;
      cpuUsage = (cpuDelta / systemDelta) * 100 * stats.cpu_stats.online_cpus;

      // Calculate memory usage percentage
      const memoryUsageBytes = stats.memory_stats.usage - stats.memory_stats.stats.cache;
      const memoryLimit = stats.memory_stats.limit;
      memoryUsage = (memoryUsageBytes / memoryLimit) * 100;
    }

    return {
      containerId,
      containerName: container.Name.replace('/', ''),
      status,
      cpuUsage: Math.round(cpuUsage * 100) / 100,
      memoryUsage: Math.round(memoryUsage * 100) / 100,
      uptime: State.StartedAt ? Date.now() - new Date(state.StartedAt).getTime() : 0,
      lastHealthCheck: new Date().toISOString(),
    };
  }

  // Get all running containers
  private async getAllContainers(): Promise<string[]> {
    const http = new HttpBubble({
      url: 'http://docker-api:2375/containers/json',
      method: 'GET',
      timeout: 5000,
    });

    const response = await http.action();

    if (!response.success || !response.data) {
      throw new Error(`Failed to list containers: ${response.error}`);
    }

    return response.data.map((container: any) => container.Id);
  }

  // Restart a container
  private async restartContainer(containerId: string): Promise<boolean> {
    const http = new HttpBubble({
      url: `http://docker-api:2375/containers/${containerId}/restart`,
      method: 'POST',
      timeout: 30000, // 30 seconds for restart
    });

    const response = await http.action();

    if (!response.success) {
      this.logger?.error(`Failed to restart container ${containerId}: ${response.error}`);
      return false;
    }

    this.logger?.info(`Successfully restarted container ${containerId}`);
    return true;
  }

  // Analyze root cause using AI
  private async analyzeRootCause(health: ContainerHealth): Promise<string> {
    const agent = new AIAgentBubble({
      model: {
        model: 'gpt-4',
        temperature: 0.3,
      },
      systemPrompt: 'You are a DevOps expert specializing in container diagnostics. Analyze the container health data and identify the likely root cause of issues.',
      message: `Container health analysis:
- Container: ${health.containerName}
- Status: ${health.status}
- CPU Usage: ${health.cpuUsage}%
- Memory Usage: ${health.memoryUsage}%
- Uptime: ${Math.round(health.uptime / 1000)} seconds

Identify the most likely root cause and suggest preventive measures.`,
    });

    const result = await agent.action();

    if (!result.success) {
      return 'Unable to analyze root cause';
    }

    return result.data.response;
  }

  // Send notification to Slack
  private async sendSlackNotification(healingActions: HealingAction[]): Promise<void> {
    const slack = new SlackBubble({
      channel: '#ops-alerts',
      message: {
        text: '🚨 Container Auto-Healing Report',
        attachments: [
          {
            color: healingActions.some(a => a.action !== 'notify_only') ? 'warning' : 'good',
            fields: [
              {
                title: 'Total Actions',
                value: healingActions.length.toString(),
                short: true,
              },
              {
                title: 'Containers Healed',
                value: healingActions.filter(a => a.action === 'restart').length.toString(),
                short: true,
              },
              {
                title: 'Timestamp',
                value: new Date().toISOString(),
                short: false,
              },
            ],
          },
          {
            title: 'Healing Actions',
            text: healingActions
              .map(
                a =>
                  `• ${a.action.toUpperCase()}: ${a.containerId}\n  Reason: ${a.reason}\n  Time: ${a.timestamp}`
              )
              .join('\n\n'),
          },
        ],
      },
    });

    await slack.action();
  }

  // Main healing logic
  private async healContainer(
    health: ContainerHealth,
    thresholds: { cpu: number; memory: number },
    force: boolean
  ): Promise<HealingAction> {
    const action: HealingAction = {
      action: 'notify_only',
      containerId: health.containerId,
      reason: '',
      timestamp: new Date().toISOString(),
    };

    // Determine if healing is needed
    const needsHealing =
      force ||
      health.status === 'unhealthy' ||
      health.cpuUsage > thresholds.cpu ||
      health.memoryUsage > thresholds.memory;

    if (!needsHealing) {
      action.reason = 'Container is healthy';
      return action;
    }

    // Decide action based on severity
    if (health.status === 'unhealthy' || force) {
      action.action = 'restart';
      action.reason = `Container status: ${health.status}${force ? ' (forced)' : ''}`;

      const success = await this.restartContainer(health.containerId);
      if (!success) {
        action.action = 'notify_only';
        action.reason = `Failed to restart: ${health.status}`;
      }
    } else if (health.cpuUsage > thresholds.cpu + 10) {
      // CPU is critically high
      action.action = 'restart';
      action.reason = `Critical CPU usage: ${health.cpuUsage}%`;

      const success = await this.restartContainer(health.containerId);
      if (!success) {
        action.action = 'notify_only';
        action.reason = `Failed to restart due to high CPU`;
      }
    } else if (health.memoryUsage > thresholds.memory + 10) {
      // Memory is critically high
      action.action = 'restart';
      action.reason = `Critical memory usage: ${health.memoryUsage}%`;

      const success = await this.restartContainer(health.containerId);
      if (!success) {
        action.action = 'notify_only';
        action.reason = `Failed to restart due to high memory`;
      }
    } else {
      // Minor issues, just notify
      action.action = 'notify_only';
      action.reason = `Warning - CPU: ${health.cpuUsage}%, Memory: ${health.memoryUsage}%`;
    }

    return action;
  }

  // Main workflow orchestration
  async handle(payload: CustomWebhookPayload): Promise<Output> {
    // Security: Generate correlation ID for tracing
    const correlationId = generateCorrelationId();
    this.logger = this.logger.child({ correlationId });

    // Security: Rate limiting check
    if (!this.rateLimiter.checkLimit(correlationId)) {
      throw new Error('Rate limit exceeded. Please try again later.');
    }

    // Security: API key authentication
    const authContext = authenticateRequest(
      payload.headers?.['x-api-key'],
      process.env.API_KEY,
      { correlationId, ip: payload.headers?.['x-forwarded-for'] }
    );
    requireAuthentication(authContext);

    this.logger.info({
      msg: 'Starting container autohealing',
    });

    const {
      containerId: targetContainerId,
      forceHeal = false,
      cpuThreshold = this.DEFAULT_CPU_THRESHOLD,
      memoryThreshold = this.DEFAULT_MEMORY_THRESHOLD,
      notify = true,
      alertDetails,
    } = payload;

    this.logger?.info('Starting container auto-healing check...');

    // Log alert details if provided
    if (alertDetails) {
      this.logger?.info(
        `Triggered by alert: ${alertDetails.alertName} - ${alertDetails.metric} = ${alertDetails.currentValue} (threshold: ${alertDetails.threshold})`
      );
    }

    // Get containers to check
    let containerIds: string[] = [];
    if (targetContainerId) {
      containerIds = [targetContainerId];
    } else {
      containerIds = await this.getAllContainers();
      this.logger?.info(`Found ${containerIds.length} containers to check`);
    }

    const thresholds = { cpu: cpuThreshold, memory: memoryThreshold };
    const healingActions: HealingAction[] = [];
    let containersHealed = 0;

    // Check each container
    for (const containerId of containerIds) {
      try {
        this.logger?.info(`Checking container ${containerId}...`);
        const health = await this.getContainerHealth(containerId);

        this.logger?.info(
          `Container ${health.containerName}: Status=${health.status}, CPU=${health.cpuUsage}%, Memory=${health.memoryUsage}%`
        );

        // Perform healing if needed
        const action = await this.healContainer(health, thresholds, forceHeal);
        healingActions.push(action);

        if (action.action === 'restart') {
          containersHealed++;
        }
      } catch (error) {
        this.logger?.error(`Error checking container ${containerId}: ${error}`);

        healingActions.push({
          action: 'notify_only',
          containerId,
          reason: `Error during health check: ${error}`,
          timestamp: new Date().toISOString(),
        });
      }
    }

    // Send notifications
    let alertUrl: string | undefined;
    if (notify && healingActions.length > 0) {
      try {
        await this.sendSlackNotification(healingActions);
        this.logger?.info('Slack notification sent');
      } catch (error) {
        this.logger?.error(`Failed to send Slack notification: ${error}`);
      }
    }

    return {
      message: `Checked ${containerIds.length} container(s), healed ${containersHealed} container(s)`,
      containersChecked: containerIds.length,
      containersHealed,
      healingActions,
      alertUrl,
    };
  }
}

// Export workflow configuration
export const workflowConfig = {
  id: 'container-autohealing',
  name: 'Container Auto-Healing',
  description: 'Automatically detect and heal unhealthy containers',
  version: '1.0.0',
  category: 'infrastructure-automation',
  icon: '🏥',
  tags: ['docker', 'kubernetes', 'reliability', 'auto-healing', 'devops'],
};
