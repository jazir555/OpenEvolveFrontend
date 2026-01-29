/**
 * Distributed Tracing Analyzer
 * Purpose: Analyze distributed traces to identify performance bottlenecks
 * Category: Infrastructure Automation
 * Event Type: schedule/cron
 * Schedule: */15 * * * * (Every 15 minutes)
 *
 * Required Credentials:
 * - JAEGER_API: Jaeger tracing API endpoint
 * - POSTGRES_CONNECTION_STRING: To store analysis results
 * - API_KEY: API key for authentication (required)
 * - SLACK_WEBHOOK_URL: For alerts (optional)
 *
 * Security Fixes Applied (Wave 5):
 * - Environment variable validation at startup
 * - API key authentication
 * - Rate limiting
 * - Input validation for all user inputs
 * - SQL injection prevention with parameterized queries
 * - Error message sanitization
 * - Structured logging with correlation IDs
 * - URL validation for all endpoints
 */

import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  SlackBubble,
  PostgreSQLBubble,
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
  buildParameterizedQuery,
  SecuritySchemas,
} from '../security-utils';

interface TraceSpan {
  traceID: string;
  spanID: string;
  operationName: string;
  service: string;
  duration: number;
  tags: Record<string, string>;
  logs: any[];
}

interface TraceAnalysis {
  timestamp: string;
  totalTraces: number;
  slowTraces: number;
  errorTraces: number;
  bottlenecks: Array<{
    service: string;
    operation: string;
    avgDuration: number;
    p95Duration: number;
    errorRate: number;
  }>;
  recommendations: string[];
  correlationId: string;
}

// Security: Environment variable validation
validateEnvironment({
  required: ['JAEGER_API', 'POSTGRES_CONNECTION_STRING', 'API_KEY'],
  optional: ['SLACK_WEBHOOK_URL'],
  schemas: {
    API_KEY: SecuritySchemas.apiKey,
    JAEGER_API: SecuritySchemas.url,
  },
});

export class DistributedTracingAnalyzer extends BubbleFlow<'schedule/cron'> {
  readonly cronSchedule = '*/15 * * * *';
  readonly name = 'Distributed Tracing Analyzer';
  readonly description = 'Analyze distributed traces for performance bottlenecks';

  private logger = new StructuredLogger('distributed-tracing-analyzer');
  private rateLimiter = new RateLimiter({
    maxRequests: 4, // 4 analyses per minute (one every 15 seconds)
    windowMs: 60000, // 1 minute
  });

  async handle(payload: CronEvent): Promise<TraceAnalysis> {
    // Security: Generate correlation ID for tracing
    const correlationId = generateCorrelationId();
    this.logger = this.logger.child({ correlationId });

    const timestamp = new Date().toISOString();
    const lookbackMinutes = 15;
    const startTime = new Date(Date.now() - lookbackMinutes * 60 * 1000).toISOString();

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
      msg: 'Starting trace analysis',
      lookbackMinutes,
    });

    // Step 1: Query Jaeger for traces
    const validatedJaegerApi = InputValidator.validateUrl(process.env.JAEGER_API);

    const getTraces = new HttpBubble({
      url: InputValidator.validateUrl(`${validatedJaegerApi}/api/traces?start=${startTime}&end=${timestamp}&limit=1000`),
      method: 'GET',
      timeout: 30000,
    });

    const tracesResponse = await getTraces.action();
    const traces = tracesResponse.data.data || [];

    // Step 2: Analyze traces
    let slowTraces = 0;
    let errorTraces = 0;
    const serviceMetrics = new Map<string, {
      durations: number[];
      errors: number;
      total: number;
    }>();

    for (const trace of traces) {
      const traceID = InputValidator.sanitizeString(trace.traceID, 64);
      const spans = trace.spans || [];

      let hasError = false;
      let maxDuration = 0;

      for (const span of spans) {
        const service = InputValidator.validateServiceName(span.process.serviceName);
        const duration = InputValidator.sanitizeNumber(span.duration, 0);
        const hasErrorTag = span.tags?.some((t: any) => t.key === 'error' && t.value === true);

        if (!serviceMetrics.has(service)) {
          serviceMetrics.set(service, {
            durations: [],
            errors: 0,
            total: 0,
          });
        }

        const metrics = serviceMetrics.get(service)!;
        metrics.durations.push(duration);
        metrics.total++;

        if (hasErrorTag) {
          metrics.errors++;
          hasError = true;
        }

        maxDuration = Math.max(maxDuration, duration);
      }

      if (hasError) {
        errorTraces++;
      }

      if (maxDuration > 5000000) { // > 5 seconds
        slowTraces++;
      }
    }

    // Step 3: Calculate bottlenecks
    const bottlenecks: TraceAnalysis['bottlenecks'] = [];

    for (const [service, metrics] of serviceMetrics.entries()) {
      const avgDuration = metrics.durations.reduce((a, b) => a + b, 0) / metrics.durations.length;
      const sortedDurations = metrics.durations.sort((a, b) => a - b);
      const p95Index = Math.floor(sortedDurations.length * 0.95);
      const p95Duration = sortedDurations[p95Index] || 0;
      const errorRate = InputValidator.sanitizeNumber((metrics.errors / metrics.total) * 100, 0, 100);

      bottlenecks.push({
        service,
        operation: 'all',
        avgDuration,
        p95Duration,
        errorRate,
      });
    }

    // Sort by p95 duration descending
    bottlenecks.sort((a, b) => b.p95Duration - a.p95Duration);

    // Step 4: Generate recommendations with AI
    const recommendations = await this.generateRecommendations(bottlenecks, traces);

    const result: TraceAnalysis = {
      timestamp,
      totalTraces: traces.length,
      slowTraces,
      errorTraces,
      bottlenecks: bottlenecks.slice(0, 10), // Top 10
      recommendations,
      correlationId,
    };

    // Step 5: Store analysis
    // Security: SQL injection prevention - use parameterized query
    const storeAnalysisQuery = buildParameterizedQuery(
      `
        INSERT INTO trace_analysis (
          timestamp, total_traces, slow_traces, error_traces, bottlenecks, recommendations
        )
        VALUES ($1, $2, $3, $4, $5, $6)
      `,
      [
        timestamp,
        result.totalTraces,
        result.slowTraces,
        result.errorTraces,
        JSON.stringify(result.bottlenecks),
        JSON.stringify(result.recommendations),
      ]
    );

    try {
      const storeAnalysis = new PostgreSQLBubble({
        connectionString: process.env.POSTGRES_CONNECTION_STRING,
        query: storeAnalysisQuery.query,
        params: storeAnalysisQuery.params,
      });

      await storeAnalysis.action();
    } catch (error) {
      this.logger.error({
        msg: 'Failed to store trace analysis',
      }, error);
      throw new Error('Failed to store trace analysis in database');
    }

    // Step 6: Send alerts for critical issues
    if (errorTraces > 10 || slowTraces > 20) {
      if (process.env.SLACK_WEBHOOK_URL) {
        try {
          const message = InputValidator.sanitizeString(`
🔍 Distributed Tracing Alert

Period: Last ${lookbackMinutes} minutes
Total Traces: ${result.totalTraces}
Slow Traces: ${result.slowTraces} ⚠️
Error Traces: ${result.errorTraces} 🚨

Top Bottlenecks:
${result.bottlenecks.slice(0, 5).map(b => `
${b.service}
  Avg: ${(b.avgDuration / 1000000).toFixed(2)}s
  P95: ${(b.p95Duration / 1000000).toFixed(2)}s
  Errors: ${b.errorRate.toFixed(1)}%
`).join('\n')}

Recommendations:
${result.recommendations.map(r => `• ${InputValidator.sanitizeString(r, 200)}`).join('\n')}
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
    }

    this.logger.info({
      msg: 'Trace analysis completed',
      totalTraces: result.totalTraces,
      slowTraces: result.slowTraces,
      errorTraces: result.errorTraces,
    });

    return result;
  }

  private async generateRecommendations(
    bottlenecks: TraceAnalysis['bottlenecks'],
    traces: any[]
  ): Promise<string[]> {
    const agent = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: 'Analyze performance bottlenecks and provide actionable recommendations',
      message: InputValidator.sanitizeString(`
Performance Bottlenecks:
${JSON.stringify(bottlenecks.slice(0, 5), null, 2)}

Provide 3-5 specific recommendations to improve performance.
Return as JSON array of strings.
      `.trim(), 10000),
    });

    const result = await agent.action();

    try {
      return JSON.parse(result.data.response);
    } catch (parseError) {
      // Fallback recommendations
      const recommendations: string[] = [];

      const slowest = bottlenecks[0];
      if (slowest && slowest.avgDuration > 1000000) {
        recommendations.push(`Optimize ${slowest.service}: average latency is ${(slowest.avgDuration / 1000000).toFixed(2)}s`);
      }

      const highErrorRate = bottlenecks.find(b => b.errorRate > 5);
      if (highErrorRate) {
        recommendations.push(`Fix errors in ${highErrorRate.service}: error rate is ${highErrorRate.errorRate.toFixed(1)}%`);
      }

      recommendations.push('Consider implementing caching for frequently accessed data');
      recommendations.push('Review database query performance and add indexes');

      return recommendations;
    }
  }
}

export default DistributedTracingAnalyzer;
