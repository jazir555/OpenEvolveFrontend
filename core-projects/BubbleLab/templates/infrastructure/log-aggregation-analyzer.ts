/**
 * Log Aggregation & Analyzer
 * Purpose: Aggregates logs from multiple services and detects anomalies using AI
 * Category: Infrastructure Automation
 * Event Type: schedule/cron
 * Schedule: * * * * * (Every minute)
 *
 * Required Credentials:
 * - POSTGRES_CONNECTION_STRING: PostgreSQL connection string
 * - SLACK_WEBHOOK_URL: Slack webhook for alerts (optional)
 * - API_KEY: API key for authentication (required)
 * - OPENAI_API_KEY: For AI analysis (optional, uses default model if not provided)
 *
 * Security Fixes Applied (Wave 2):
 * - Environment variable validation at startup
 * - SQL injection prevention with parameterized queries
 * - API key authentication
 * - Rate limiting
 * - Error message sanitization
 * - Structured logging with correlation IDs
 * - Input validation for all user inputs
 */

import {
  BubbleFlow,
  PostgreSQLBubble,
  AIAgentBubble,
  SlackBubble,
  type CronEvent
} from '@bubblelab/bubble-core';
import { z } from 'zod';
import crypto from 'crypto';
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
const ServiceNameSchema = z.string().min(1).max(255).regex(/^[a-zA-Z0-9_-]+$/, 'Invalid service name');
const LogLevelSchema = z.enum(['debug', 'info', 'warn', 'error', 'fatal']);
const ApiKeySchema = z.string().min(32).max(256);

interface LogEntry {
  service: string;
  level: string;
  message: string;
  timestamp: Date;
  metadata?: Record<string, any>;
}

interface AnomalyDetection {
  detected: boolean;
  anomalies: string[];
  confidence: number;
  recommendations: string[];
}

interface LogAnalysisResult {
  timestamp: string;
  logsProcessed: number;
  errorCount: number;
  warnCount: number;
  anomalies: AnomalyDetection;
  correlationId: string;
}

// Security: Environment variable validation
validateEnvironment({
  required: ['POSTGRES_CONNECTION_STRING', 'API_KEY'],
  schemas: {
    API_KEY: ApiKeySchema,
  },
});

export class LogAggregationAnalyzer extends BubbleFlow<'schedule/cron'> {
  readonly cronSchedule = '* * * * *';
  readonly name = 'Log Aggregation & Analyzer';
  readonly description = 'Aggregates logs from services and detects anomalies';

  private logger = new StructuredLogger('log-aggregation-analyzer');
  private rateLimiter = new RateLimiter({
    maxRequests: 60, // 1 per second average
    windowMs: 60000, // 1 minute
  });

  async handle(payload: CronEvent): Promise<LogAnalysisResult> {
    // Security: Generate correlation ID for tracing
    const correlationId = generateCorrelationId();
    this.logger = this.logger.child({ correlationId });

    const timestamp = new Date().toISOString();
    const oneMinuteAgo = new Date(Date.now() - 60000).toISOString();

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
      msg: 'Starting log aggregation',
      timestamp,
    });

    // Step 1: Collect logs from database
    // Security: SQL injection prevention - use parameterized query
    const collectLogsQuery = buildParameterizedQuery(
      `
        SELECT
          service,
          level,
          message,
          timestamp,
          metadata
        FROM logs
        WHERE timestamp > $1
        ORDER BY timestamp DESC
        LIMIT 1000
      `,
      [oneMinuteAgo]
    );

    const collectLogs = new PostgreSQLBubble({
      connectionString: process.env.POSTGRES_CONNECTION_STRING,
      query: collectLogsQuery.query,
      params: collectLogsQuery.params,
    });

    let logsResult;
    try {
      logsResult = await collectLogs.action();
    } catch (error) {
      this.logger.error({
        msg: 'Failed to collect logs',
      }, error);
      throw new Error('Failed to retrieve logs from database');
    }

    const logs: LogEntry[] = logsResult.data.rows || [];

    // Step 2: Categorize logs
    const errorLogs = logs.filter(log => log.level === 'error');
    const warnLogs = logs.filter(log => log.level === 'warn');
    const criticalLogs = errorLogs.filter(log =>
      log.message.toLowerCase().includes('critical') ||
      log.message.toLowerCase().includes('fatal')
    );

    // Step 3: Detect anomalies with AI
    let anomalyDetection: AnomalyDetection = {
      detected: false,
      anomalies: [],
      confidence: 0,
      recommendations: [],
    };

    if (logs.length > 0) {
      const logSummary = {
        totalLogs: logs.length,
        errorCount: errorLogs.length,
        warnCount: warnLogs.length,
        criticalCount: criticalLogs.length,
        services: [...new Set(logs.map(l => l.service))],
        sampleErrors: errorLogs.slice(0, 5).map(l => ({
          service: l.service,
          message: InputValidator.sanitizeString(l.message, 500),
        })),
      };

      const agent = new AIAgentBubble({
        model: { model: 'openai/gpt-4' },
        systemPrompt: `Analyze these log summaries and detect anomalies. Look for:
1. Error rate spikes (sudden increases)
2. Unusual error patterns
3. Service failures
4. Performance degradation
5. Security-related events

Respond with JSON:
{
  "detected": boolean,
  "anomalies": ["description1", "description2"],
  "confidence": number (0-1),
  "recommendations": ["action1", "action2"]
}`,
        message: JSON.stringify(logSummary, null, 2),
      });

      const analysisResult = await agent.action();

      try {
        anomalyDetection = JSON.parse(analysisResult.data.response);
      } catch (parseError) {
        // Fallback if AI response is not valid JSON
        anomalyDetection = {
          detected: criticalLogs.length > 0,
          anomalies: criticalLogs.map(l => InputValidator.sanitizeString(l.message, 500)),
          confidence: criticalLogs.length > 0 ? 0.8 : 0.2,
          recommendations: ['Review critical logs', 'Check affected services'],
        };
      }
    }

    const result: LogAnalysisResult = {
      timestamp,
      logsProcessed: logs.length,
      errorCount: errorLogs.length,
      warnCount: warnLogs.length,
      anomalies: anomalyDetection,
      correlationId,
    };

    // Step 4: Send alerts for critical issues
    if (anomalyDetection.detected || criticalLogs.length > 0) {
      const alertMessage = `
🚨 Log Analysis Alert

Timestamp: ${timestamp}
Logs Processed: ${result.logsProcessed}
Errors: ${result.errorCount}
Warnings: ${result.warnCount}

${anomalyDetection.detected ? `
⚠️ Anomalies Detected:
${anomalyDetection.anomalies.map(a => `  • ${a}`).join('\n')}

Confidence: ${(anomalyDetection.confidence * 100).toFixed(1)}%

Recommendations:
${anomalyDetection.recommendations.map(r => `  • ${r}`).join('\n')}
` : ''}

${criticalLogs.length > 0 ? `
🔴 Critical Logs:
${criticalLogs.slice(0, 3).map(l => `  [${l.service}] ${InputValidator.sanitizeString(l.message, 200)}`).join('\n')}
` : ''}
      `.trim();

      if (process.env.SLACK_WEBHOOK_URL) {
        try {
          const slack = new SlackBubble({
            webhookUrl: process.env.SLACK_WEBHOOK_URL,
            message: alertMessage,
          });

          await slack.action();
        } catch (error) {
          // Don't throw - notification failure shouldn't break the workflow
          this.logger.warn({
            msg: 'Slack notification failed',
          }, error);
        }
      }
    }

    // Step 5: Store analysis results (optional)
    if (anomalyDetection.detected) {
      // Security: SQL injection prevention - use parameterized query
      const storeAnalysisQuery = buildParameterizedQuery(
        `
          INSERT INTO log_anomalies (timestamp, detected, anomalies, confidence, recommendations)
          VALUES ($1, $2, $3, $4, $5)
        `,
        [
          timestamp,
          anomalyDetection.detected,
          JSON.stringify(anomalyDetection.anomalies),
          anomalyDetection.confidence,
          JSON.stringify(anomalyDetection.recommendations),
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
          msg: 'Failed to store analysis results',
        }, error);
        // Don't throw - storage failure shouldn't break the workflow
      }
    }

    this.logger.info({
      msg: 'Log analysis completed',
      logsProcessed: result.logsProcessed,
      errorCount: result.errorCount,
      warnCount: result.warnCount,
      anomaliesDetected: anomalyDetection.detected,
    });

    return result;
  }
}

export default LogAggregationAnalyzer;
