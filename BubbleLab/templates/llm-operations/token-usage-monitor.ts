/**
 * Token Usage Monitor
 * Purpose: Monitor and analyze LLM token usage and costs
 * Category: LLM Operations
 * Event Type: schedule/cron
 * Schedule: 0 * * * * (Every hour)
 *
 * Required Credentials:
 * - POSTGRES_CONNECTION_STRING: Database with token usage logs
 * - SLACK_WEBHOOK_URL: For alerts (optional)
 * - API_KEY: API key for authentication (required)
 *
 * Security Fixes Applied (Wave 2):
 * - Environment variable validation at startup
 * - API key authentication
 * - Input validation for all user inputs (Zod schemas)
 * - Rate limiting
 * - SQL injection prevention with parameterized queries
 * - Error message sanitization
 * - Structured logging with correlation IDs
 */

import {
  BubbleFlow,
  PostgreSQLBubble,
  AIAgentBubble,
  SlackBubble,
  type CronEvent
} from '@bubblelab/bubble-core';
import { z } from 'zod';
import {
  validateEnvironment,
  authenticateRequest,
  requireAuthentication,
  RateLimiter,
  InputValidator,
  sanitizeError,
  StructuredLogger,
  generateCorrelationId,
  buildParameterizedQuery,
} from '../security-utils';

// Input validation schemas
const ApiKeySchema = z.string().min(32).max(256);
const ModelSchema = z.string().min(1).max(100);
const WorkflowSchema = z.string().min(1).max(255);

interface TokenUsage {
  timestamp: string;
  model: string;
  workflow: string;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  cost: number;
}

interface UsageSummary {
  period: string;
  totalTokens: number;
  totalCost: number;
  byModel: Record<string, { tokens: number; cost: number }>;
  byWorkflow: Record<string, { tokens: number; cost: number }>;
  projections: {
    daily: number;
    monthly: number;
  };
}

interface MonitoringResult {
  timestamp: string;
  summary: UsageSummary;
  alerts: string[];
  recommendations: string[];
  correlationId: string;
}

// Security: Environment variable validation
validateEnvironment({
  required: ['POSTGRES_CONNECTION_STRING', 'API_KEY'],
  optional: ['SLACK_WEBHOOK_URL', 'OPENAI_API_KEY'],
  schemas: {
    API_KEY: ApiKeySchema,
  },
});

export class TokenUsageMonitor extends BubbleFlow<'schedule/cron'> {
  readonly cronSchedule = '0 * * * *';
  readonly name = 'Token Usage Monitor';
  readonly description = 'Monitor and analyze LLM token usage and costs';

  private logger = new StructuredLogger('token-usage-monitor');
  private rateLimiter = new RateLimiter({
    maxRequests: 24, // Once per hour is expected
    windowMs: 3600000, // 1 hour
  });

  async handle(payload: CronEvent): Promise<MonitoringResult> {
    // Security: Generate correlation ID for tracing
    const correlationId = generateCorrelationId();
    this.logger = this.logger.child({ correlationId });

    const timestamp = new Date().toISOString();
    const oneHourAgo = new Date(Date.now() - 3600000).toISOString();
    const oneDayAgo = new Date(Date.now() - 86400000).toISOString();

    // Security: Rate limiting check
    if (!this.rateLimiter.checkLimit(correlationId)) {
      this.logger.warn({
        msg: 'Rate limit exceeded',
      });
      throw new Error('Rate limit exceeded. Maximum 24 checks per hour.');
    }

    // Security: API key authentication
    const authContext = authenticateRequest(
      payload.headers?.['x-api-key'],
      process.env.API_KEY,
      { correlationId, ip: payload.headers?.['x-forwarded-for'] }
    );
    requireAuthentication(authContext);

    this.logger.info({
      msg: 'Starting token usage monitoring',
    });

    // Step 1: Get token usage for the last hour
    // Security: SQL injection prevention - use parameterized query
    const recentUsageQuery = buildParameterizedQuery(
      `
        SELECT
          timestamp,
          model,
          workflow,
          prompt_tokens,
          completion_tokens,
          total_tokens,
          cost
        FROM token_usage_log
        WHERE timestamp > $1
        ORDER BY timestamp DESC
      `,
      [oneHourAgo]
    );

    let recentResult;
    try {
      const recentUsage = new PostgreSQLBubble({
        connectionString: process.env.POSTGRES_CONNECTION_STRING,
        query: recentUsageQuery.query,
        params: recentUsageQuery.params,
      });

      recentResult = await recentUsage.action();
    } catch (error) {
      this.logger.error({
        msg: 'Failed to collect recent token usage',
      }, error);
      throw new Error('Failed to retrieve token usage data');
    }

    const recentUsageData: TokenUsage[] = recentResult.data.rows || [];

    // Step 2: Get token usage for the last day (for projections)
    // Security: SQL injection prevention - use parameterized query
    const dailyUsageQuery = buildParameterizedQuery(
      recentUsageQuery.query,
      [oneDayAgo]
    );

    let dailyResult;
    try {
      const dailyUsage = new PostgreSQLBubble({
        connectionString: process.env.POSTGRES_CONNECTION_STRING,
        query: dailyUsageQuery.query,
        params: dailyUsageQuery.params,
      });

      dailyResult = await dailyUsage.action();
    } catch (error) {
      this.logger.warn({
        msg: 'Failed to collect daily token usage',
      }, error);
      // Don't throw - continue with empty data
      dailyResult = { data: { rows: [] } };
    }

    const dailyUsageData: TokenUsage[] = dailyResult.data.rows || [];

    // Step 3: Calculate summary
    const summary = this.calculateSummary(recentUsageData, dailyUsageData);

    // Step 4: Generate alerts
    const alerts = this.generateAlerts(summary);

    // Step 5: Generate recommendations with AI
    const recommendations = await this.generateRecommendations(summary);

    const result: MonitoringResult = {
      timestamp,
      summary,
      alerts,
      recommendations,
      correlationId,
    };

    // Step 6: Send alerts if needed
    if (alerts.length > 0 && process.env.SLACK_WEBHOOK_URL) {
      const message = InputValidator.sanitizeString(`
💰 Token Usage Alert

Period: Last Hour
Total Tokens: ${summary.totalTokens.toLocaleString()}
Total Cost: $${summary.totalCost.toFixed(2)}

Alerts:
${alerts.map(a => `  ⚠️ ${a}`).join('\n')}

Recommendations:
${recommendations.map(r => `  💡 ${r}`).join('\n')}
      `.trim(), 10000);

      try {
        const slack = new SlackBubble({
          webhookUrl: process.env.SLACK_WEBHOOK_URL,
          message,
        });

        await slack.action();
      } catch (error) {
        // Don't throw - notification failure shouldn't break the workflow
        this.logger.warn({
          msg: 'Slack notification failed',
        }, error);
      }
    }

    this.logger.info({
      msg: 'Token usage monitoring completed',
      totalTokens: summary.totalTokens,
      totalCost: summary.totalCost,
      alertCount: alerts.length,
    });

    return result;
  }

  private calculateSummary(recent: TokenUsage[], daily: TokenUsage[]): UsageSummary {
    const totalTokens = recent.reduce((sum, u) => sum + u.totalTokens, 0);
    const totalCost = recent.reduce((sum, u) => sum + u.cost, 0);

    // Group by model (with validation)
    const byModel: Record<string, { tokens: number; cost: number }> = {};

    for (const usage of recent) {
      try {
        // Validate model name
        ModelSchema.parse(usage.model);

        if (!byModel[usage.model]) {
          byModel[usage.model] = { tokens: 0, cost: 0 };
        }
        byModel[usage.model].tokens += usage.totalTokens;
        byModel[usage.model].cost += usage.cost;
      } catch {
        // Skip invalid model names
        continue;
      }
    }

    // Group by workflow (with validation)
    const byWorkflow: Record<string, { tokens: number; cost: number }> = {};

    for (const usage of recent) {
      try {
        // Validate workflow name
        WorkflowSchema.parse(usage.workflow);

        if (!byWorkflow[usage.workflow]) {
          byWorkflow[usage.workflow] = { tokens: 0, cost: 0 };
        }
        byWorkflow[usage.workflow].tokens += usage.totalTokens;
        byWorkflow[usage.workflow].cost += usage.cost;
      } catch {
        // Skip invalid workflow names
        continue;
      }
    }

    // Calculate projections
    const dailyTotal = daily.reduce((sum, u) => sum + u.totalTokens, 0);
    const dailyCost = daily.reduce((sum, u) => sum + u.cost, 0);

    return {
      period: 'last_hour',
      totalTokens: InputValidator.sanitizeNumber(totalTokens, 0, Number.MAX_SAFE_INTEGER),
      totalCost: InputValidator.sanitizeNumber(totalCost, 0, Number.MAX_SAFE_INTEGER),
      byModel,
      byWorkflow,
      projections: {
        daily: InputValidator.sanitizeNumber(dailyTotal, 0, Number.MAX_SAFE_INTEGER),
        monthly: InputValidator.sanitizeNumber(dailyTotal * 30, 0, Number.MAX_SAFE_INTEGER),
      },
    };
  }

  private generateAlerts(summary: UsageSummary): string[] {
    const alerts: string[] = [];

    // Alert on high cost
    if (summary.totalCost > 10) {
      alerts.push(`High hourly cost: $${summary.totalCost.toFixed(2)}`);
    }

    // Alert on high token usage
    if (summary.totalTokens > 1000000) {
      alerts.push(`High token usage: ${(summary.totalTokens / 1000000).toFixed(2)}M tokens`);
    }

    // Alert on expensive models
    for (const [model, data] of Object.entries(summary.byModel)) {
      if (data.cost > 5) {
        alerts.push(`Expensive model ${model}: $${data.cost.toFixed(2)}`);
      }
    }

    // Alert on projection
    if (summary.projections.monthly > 50000000) {
      alerts.push(`Projected monthly usage: ${(summary.projections.monthly / 1000000).toFixed(2)}M tokens`);
    }

    return alerts.map(a => InputValidator.sanitizeString(a, 500));
  }

  private async generateRecommendations(summary: UsageSummary): Promise<string[]> {
    const agent = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: 'Analyze token usage data and provide cost optimization recommendations',
      message: InputValidator.sanitizeString(`
Token Usage Summary:
${JSON.stringify(summary, null, 2)}

Provide 3-5 specific recommendations to reduce costs or optimize usage.
Respond with a JSON array of strings.
      `.trim(), 10000),
    });

    const result = await agent.action();

    try {
      const recommendations = JSON.parse(result.data.response);
      return recommendations.map((r: string) => InputValidator.sanitizeString(r, 500));
    } catch (parseError) {
      // Fallback recommendations
      const recommendations: string[] = [];

      // Find most expensive model
      const mostExpensiveModel = Object.entries(summary.byModel)
        .sort((a, b) => b[1].cost - a[1].cost)[0];

      if (mostExpensiveModel) {
        recommendations.push(
          `Consider switching from ${mostExpensiveModel[0]} to a more cost-effective model`
        );
      }

      // Find most expensive workflow
      const mostExpensiveWorkflow = Object.entries(summary.byWorkflow)
        .sort((a, b) => b[1].cost - a[1].cost)[0];

      if (mostExpensiveWorkflow) {
        recommendations.push(
          `Review workflow "${mostExpensiveWorkflow[0]}" for optimization opportunities`
        );
      }

      recommendations.push('Implement response caching to reduce redundant API calls');
      recommendations.push('Use smaller models for simple tasks');

      return recommendations;
    }
  }
}

export default TokenUsageMonitor;
