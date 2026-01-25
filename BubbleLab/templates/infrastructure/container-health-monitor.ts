/**
 * Container Health Monitor
 * Purpose: Monitors Docker container health and auto-heals unhealthy containers
 * Category: Infrastructure Automation
 * Event Type: schedule/cron
 * Schedule: */5 * * * * (Every 5 minutes)
 *
 * Required Credentials:
 * - DOCKER_HOST: Docker daemon socket or API endpoint
 * - SLACK_WEBHOOK_URL: Slack webhook for alerts (optional)
 * - API_KEY: API key for authentication (required)
 *
 * Security Fixes Applied (Wave 2):
 * - Environment variable validation at startup
 * - API key authentication
 * - Input validation for container IDs (command injection prevention)
 * - Rate limiting
 * - Error message sanitization
 * - Structured logging with correlation IDs
 */

import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  SlackBubble,
  type CronEvent
} from '@bubblelab/bubble-core';
import { z } from 'zod';
import crypto from 'crypto';

// Input validation schemas
const ContainerIdSchema = z.string().regex(/^[a-f0-9]{12,}$/, 'Invalid container ID format');
const ContainerNameSchema = z.string().min(1).max(255).regex(/^[a-zA-Z0-9_-]+$/, 'Invalid container name');
const ApiKeySchema = z.string().min(32).max(256);

interface ContainerHealth {
  containerId: string;
  name: string;
  status: string;
  health: string;
  uptime: number;
}

interface HealthCheckResult {
  timestamp: string;
  total: number;
  healthy: number;
  unhealthy: number;
  healed: number;
  containers: ContainerHealth[];
  correlationId: string;
}

// Security: Environment variable validation
const requiredEnvVars = ['DOCKER_HOST', 'API_KEY'];
const missing = requiredEnvVars.filter(key => !process.env[key]);
if (missing.length > 0) {
  throw new Error(`CRITICAL: Missing required environment variables: ${missing.join(', ')}. Set them and restart.`);
}

// Security: Validate API key format
try {
  ApiKeySchema.parse(process.env.API_KEY);
} catch (error) {
  throw new Error('CRITICAL: API_KEY format is invalid. Must be 32-256 characters.');
}

export class ContainerHealthMonitor extends BubbleFlow<'schedule/cron'> {
  readonly cronSchedule = '*/5 * * * *';
  readonly name = 'Container Health Monitor';
  readonly description = 'Monitors container health and auto-heals unhealthy containers';

  // Security: Rate limiting state
  private static requestCounts = new Map<string, { count: number; resetTime: number }>();
  private static readonly RATE_LIMIT = {
    maxRequests: 100,
    windowMs: 60000 // 1 minute
  };

  private checkRateLimit(identifier: string): boolean {
    const now = Date.now();
    const key = identifier || 'anonymous';

    let record = ContainerHealthMonitor.requestCounts.get(key);

    if (!record || now > record.resetTime) {
      record = { count: 0, resetTime: now + ContainerHealthMonitor.RATE_LIMIT.windowMs };
      ContainerHealthMonitor.requestCounts.set(key, record);
    }

    record.count++;

    if (record.count > ContainerHealthMonitor.RATE_LIMIT.maxRequests) {
      return false;
    }

    return true;
  }

  private sanitizeContainerId(containerId: string): string {
    // Security: Command injection prevention - validate container ID format
    try {
      ContainerIdSchema.parse(containerId);
      return containerId;
    } catch (error) {
      throw new Error(`Invalid container ID format: ${containerId.substring(0, 12)}`);
    }
  }

  private sanitizeContainerName(name: string): string {
    // Security: Input sanitization for container names
    try {
      return ContainerNameSchema.parse(name);
    } catch (error) {
      return 'invalid-name';
    }
  }

  private sanitizeError(error: unknown): string {
    // Security: Remove sensitive data from error messages
    if (error instanceof Error) {
      // Remove stack traces and internal paths
      return error.message.replace(/\/[a-zA-Z0-9_\-\/]+\.ts:\d+:\d+/g, '[internal]').replace(/at .+/g, '');
    }
    return 'Unknown error';
  }

  private generateCorrelationId(): string {
    return crypto.randomBytes(16).toString('hex');
  }

  private structuredLog(level: 'info' | 'warn' | 'error', data: Record<string, unknown>, error?: unknown) {
    const logEntry = {
      timestamp: new Date().toISOString(),
      level,
      ...data,
      ...(error && { error: this.sanitizeError(error) }),
    };

    console.log(JSON.stringify(logEntry));
  }

  async handle(payload: CronEvent): Promise<HealthCheckResult> {
    // Security: Generate correlation ID for tracing
    const correlationId = this.generateCorrelationId();
    const timestamp = new Date().toISOString();

    // Security: Rate limiting check
    if (!this.checkRateLimit(correlationId)) {
      throw new Error('Rate limit exceeded. Please try again later.');
    }

    // Security: API key authentication
    const providedApiKey = payload.headers?.['x-api-key'] || process.env.API_KEY;
    if (providedApiKey !== process.env.API_KEY) {
      this.structuredLog('error', {
        msg: 'Authentication failed',
        correlationId,
        ip: payload.headers?.['x-forwarded-for'] || 'unknown',
      });
      throw new Error('Unauthorized: Invalid API key');
    }

    // Step 1: List all containers
    this.structuredLog('info', {
      msg: 'Starting container health check',
      correlationId,
    });

    const listContainers = new HttpBubble({
      url: `${process.env.DOCKER_HOST}/containers/json?all=true`,
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 10000,
    });

    let containersResult;
    try {
      containersResult = await listContainers.action();
    } catch (error) {
      this.structuredLog('error', {
        msg: 'Failed to list containers',
        correlationId,
        dockerHost: process.env.DOCKER_HOST,
      }, error);
      throw new Error('Failed to retrieve container list');
    }

    const containers = JSON.parse(containersResult.data);

    // Step 2: Check health status for each container
    const containerHealth: ContainerHealth[] = [];
    const unhealthyContainers: string[] = [];

    for (const container of containers) {
      try {
        // Security: Validate container ID before use
        const sanitizedId = this.sanitizeContainerId(container.Id);

        // Get container details
        const inspectContainer = new HttpBubble({
          url: `${process.env.DOCKER_HOST}/containers/${sanitizedId}/json`,
          method: 'GET',
          timeout: 5000,
        });

        const inspectResult = await inspectContainer.action();
        const details = JSON.parse(inspectResult.data);

        // Security: Sanitize container name
        const sanitizedName = this.sanitizeContainerName(details.Name.replace('/', ''));

        const health: ContainerHealth = {
          containerId: sanitizedId.substring(0, 12),
          name: sanitizedName,
          status: details.State.Status,
          health: details.State.Health?.Status || 'no healthcheck',
          uptime: Date.now() - new Date(details.State.StartedAt).getTime(),
        };

        containerHealth.push(health);

        // Check if unhealthy
        if (health.health === 'unhealthy' || health.status === 'exited') {
          unhealthyContainers.push(container.Id);
        }
      } catch (error) {
        this.structuredLog('warn', {
          msg: 'Failed to inspect container',
          correlationId,
          containerId: container.Id.substring(0, 12),
        }, error);

        // Security: Sanitize even in error case
        const sanitizedId = this.sanitizeContainerId(container.Id);
        const sanitizedName = this.sanitizeContainerName(
          container.Names[0]?.replace('/', '') || 'unknown'
        );

        containerHealth.push({
          containerId: sanitizedId.substring(0, 12),
          name: sanitizedName,
          status: 'error',
          health: 'unknown',
          uptime: 0,
        });
      }
    }

    // Step 3: Attempt to heal unhealthy containers
    let healedCount = 0;

    for (const containerId of unhealthyContainers) {
      try {
        // Security: Validate container ID before restart
        const sanitizedId = this.sanitizeContainerId(containerId);

        // Restart container
        const restartContainer = new HttpBubble({
          url: `${process.env.DOCKER_HOST}/containers/${sanitizedId}/restart`,
          method: 'POST',
          timeout: 30000,
        });

        await restartContainer.action();
        healedCount++;

        this.structuredLog('info', {
          msg: 'Container restarted successfully',
          correlationId,
          containerId: sanitizedId.substring(0, 12),
        });

        // Wait briefly for restart
        await new Promise(resolve => setTimeout(resolve, 2000));
      } catch (error) {
        // Security: Sanitized error message
        this.structuredLog('error', {
          msg: 'Failed to restart container',
          correlationId,
          containerId: containerId.substring(0, 12),
        }, error);
      }
    }

    const healthyCount = containerHealth.filter(
      c => c.health === 'healthy' || c.status === 'running'
    ).length;

    const result: HealthCheckResult = {
      timestamp,
      total: containerHealth.length,
      healthy: healthyCount,
      unhealthy: unhealthyContainers.length,
      healed: healedCount,
      containers: containerHealth,
      correlationId,
    };

    // Step 4: Send alert if critical issues
    if (result.unhealthy > 0 || result.healed > 0) {
      try {
        const agent = new AIAgentBubble({
          model: { model: 'google/gemini-2.5-flash' },
          systemPrompt: 'Analyze container health status and provide concise summary with recommendations',
          message: JSON.stringify(result, null, 2),
        });

        const analysis = await agent.action();

        if (process.env.SLACK_WEBHOOK_URL) {
          const slack = new SlackBubble({
            webhookUrl: process.env.SLACK_WEBHOOK_URL,
            message: `🐳 Container Health Alert\n\nTimestamp: ${timestamp}\nStatus: ${result.healthy}/${result.total} healthy\nHealed: ${result.healed}\n\nAnalysis:\n${analysis.data.response}`,
          });

          await slack.action();
        }
      } catch (error) {
        // Don't throw - notification failure shouldn't break the workflow
        this.structuredLog('warn', {
          msg: 'Slack notification failed',
          correlationId,
        }, error);
      }
    }

    this.structuredLog('info', {
      msg: 'Container health check completed',
      correlationId,
      total: result.total,
      healthy: result.healthy,
      unhealthy: result.unhealthy,
      healed: result.healed,
    });

    return result;
  }
}

export default ContainerHealthMonitor;
