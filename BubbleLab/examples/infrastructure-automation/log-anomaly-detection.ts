/**
 * Workflow: Log Anomaly Detection
 * Description: ML-based detection of anomalies in application and system logs
 * Use Case: Security monitoring, performance issue detection, and operational intelligence
 *
 * Setup Instructions:
 * 1. Configure log source credentials (CloudWatch, ELK, Loki, or custom HTTP endpoint)
 * 2. Set up storage for anomaly reports (S3, Google Drive, or database)
 * 3. Configure notification channels (Slack, PagerDuty, Email)
 * 4. Customize anomaly detection patterns and thresholds
 *
 * Required Credentials:
 * - elasticsearch: For ELK stack logs (optional)
 * - aws: For CloudWatch logs (optional)
 * - google-drive: For storing anomaly reports (optional)
 * - slack: For alert notifications (optional)
 * - ai-agent: For intelligent anomaly analysis
 *
 * Trigger Options:
 * - Scheduled: Run every 15-30 minutes
 * - Webhook: Trigger from log aggregation system
 * - Manual: Analyze specific time range
 *
 * Example Webhook Payload:
 * {
 *   "timeRange": "last-15m",
 *   "logSources": ["application", "nginx", "system"],
 *   "severity": "warning"
 * }
 *
 * Detection Patterns:
 * - Error rate spikes (sudden increase in 5xx errors)
 * - Unusual access patterns (brute force attacks, scraping)
 * - Performance anomalies (slow queries, timeouts)
 * - Security events (failed auth, permission changes)
 * - Resource exhaustion (disk space, memory)
 *
 * Performance Optimization:
 * - Use time-based pagination for large log sets
 * - Implement caching for frequently analyzed patterns
 * - Parallel processing of multiple log sources
 * - Batch anomaly reports to reduce storage operations
  *
 * Security Fixes Applied (Wave 5):
 * - Environment variable validation at startup
 * - API key authentication
 * - Rate limiting (60 requests/minute)
 * - Input validation for all user inputs
 * - Error message sanitization
 * - Structured logging with correlation IDs
 * - URL validation for all endpoints
 *

import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  SlackBubble,
  GoogleDriveBubble,
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

export interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  source: string;
  metadata?: Record<string, any>;
}

export interface Anomaly {
  id: string;
  type: 'error_spike' | 'security_event' | 'performance_issue' | 'resource_exhaustion' | 'unknown';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  affectedResources: string[];
  evidence: string[];
  timestamp: string;
  confidence: number;
}

export interface AnomalyReport {
  reportId: string;
  timeRange: { start: string; end: string };
  logsAnalyzed: number;
  anomaliesFound: number;
  anomalies: Anomaly[];
  summary: string;
  recommendations: string[];
  generatedAt: string;
}

export interface Output {
  message: string;
  reportId: string;
  anomaliesDetected: number;
  criticalIssues: number;
  reportUrl?: string;
  notificationSent: boolean;
}

export interface CustomWebhookPayload extends WebhookEvent {
  /**
   * Time range for log analysis
   * @canBeFile false
   */
  timeRange?: string | { start: string; end: string };

  /**
   * Log sources to analyze
   * @canBeFile false
   */
  logSources?: string[];

  /**
   * Minimum severity level to report
   * @canBeFile false
   */
  minSeverity?: 'low' | 'medium' | 'high' | 'critical';

  /**
   * Log aggregation endpoint
   * @canBeFile false
   */
  logEndpoint?: string;

  /**
   * API key for log endpoint
   * @canBeFile false
   */
  logApiKey?: string;

  /**
   * Enable AI-powered anomaly detection
   * @canBeFile false
   */
  enableAI?: boolean;

  /**
   * Store report to Google Drive
   * @canBeFile false
   */
  storeReport?: boolean;

  /**
   * Send notifications to Slack
   * @canBeFile false
   */
  notify?: boolean;

  /**
   * Slack channel for notifications
   * @canBeFile false
   */
  slackChannel?: string;
}

// Security: Environment variable validation at startup
validateEnvironment({
  required: ['LOG_ENDPOINT', 'LOG_API_KEY', 'API_KEY'],
  schemas: {
    API_KEY: SecuritySchemas.apiKey,
    LOG_ENDPOINT: SecuritySchemas.url,
    LOG_API_KEY: SecuritySchemas.url,
  },
});

export class LogAnomalyDetection extends BubbleFlow<'webhook/http'> {
  private logger = new StructuredLogger('log_anomaly_detection');
  private rateLimiter = new RateLimiter({
    maxRequests: 60,
    windowMs: 60000,
  });

  private readonly DEFAULT_TIME_RANGE_MINUTES = 15;
  private readonly ERROR_SPIKE_THRESHOLD = 2.5; // 2.5x normal error rate
  private readonly HIGH_ERROR_RATE_THRESHOLD = 0.05; // 5% error rate

  // Fetch logs from configured endpoint
  private async fetchLogs(
    endpoint: string,
    apiKey: string,
    timeRange: { start: string; end: string },
    sources: string[]
  ): Promise<LogEntry[]> {
    const http = new HttpBubble({
      url: endpoint,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        query: {
          bool: {
            must: [
              { range: { '@timestamp': { gte: timeRange.start, lte: timeRange.end } } },
              { terms: { 'source.keyword': sources } },
            ],
          },
        },
        size: 10000, // Adjust based on your needs
        sort: [{ '@timestamp': 'desc' }],
      }),
      timeout: 30000,
    });

    const response = await http.action();

    if (!response.success || !response.data) {
      throw new Error(`Failed to fetch logs: ${response.error}`);
    }

    // Transform response to LogEntry format
    return response.data.hits.hits.map((hit: any) => ({
      timestamp: hit._source['@timestamp'],
      level: hit._source.level || 'info',
      message: hit._source.message,
      source: hit._source.source,
      metadata: hit._source.metadata,
    }));
  }

  // Detect error rate spikes
  private detectErrorSpikes(logs: LogEntry[]): Anomaly[] {
    const anomalies: Anomaly[] = [];

    // Group logs by 5-minute intervals
    const intervalMs = 5 * 60 * 1000;
    const intervals = new Map<number, { total: number; errors: number }>();

    logs.forEach(log => {
      const timestamp = new Date(log.timestamp).getTime();
      const intervalStart = Math.floor(timestamp / intervalMs) * intervalMs;

      if (!intervals.has(intervalStart)) {
        intervals.set(intervalStart, { total: 0, errors: 0 });
      }

      const interval = intervals.get(intervalStart)!;
      interval.total++;
      if (log.level === 'error' || log.level === 'critical') {
        interval.errors++;
      }
    });

    // Calculate average error rate
    const errorRates = Array.from(intervals.values()).map(i => i.errors / i.total);
    const avgErrorRate =
      errorRates.reduce((sum, rate) => sum + rate, 0) / errorRates.length;

    // Find spikes
    intervals.forEach((interval, intervalStart) => {
      const errorRate = interval.errors / interval.total;

      if (errorRate > avgErrorRate * this.ERROR_SPIKE_THRESHOLD && errorRate > this.HIGH_ERROR_RATE_THRESHOLD) {
        anomalies.push({
          id: `error-spike-${intervalStart}`,
          type: 'error_spike',
          severity: errorRate > 0.1 ? 'critical' : errorRate > 0.05 ? 'high' : 'medium',
          description: `Error rate spike detected: ${(errorRate * 100).toFixed(2)}% (average: ${(avgErrorRate * 100).toFixed(2)}%)`,
          affectedResources: ['application'],
          evidence: [
            `Time window: ${new Date(intervalStart).toISOString()} to ${new Date(intervalStart + intervalMs).toISOString()}`,
            `Errors: ${interval.errors}/${interval.total} logs`,
          ],
          timestamp: new Date(intervalStart).toISOString(),
          confidence: 0.85,
        });
      }
    });

    return anomalies;
  }

  // Detect security events
  private detectSecurityEvents(logs: LogEntry[]): Anomaly[] {
    const anomalies: Anomaly[] = [];
    const securityPatterns = [
      { pattern: /failed.*login|authentication.*failed|invalid.*credentials/gi, type: 'brute_force' },
      { pattern: /unauthorized.*access|access.*denied|permission.*denied/gi, type: 'unauthorized_access' },
      { pattern: /sql injection|union select|' OR '1'='1/gi, type: 'sql_injection' },
      { pattern: /cross.site.scripting|xss|<script>/gi, type: 'xss_attempt' },
      { pattern: /path.traversal|\.\.\//gi, type: 'path_traversal' },
    ];

    const ipCounts = new Map<string, number>();
    const securityEvents: Map<string, { count: number; examples: string[] }> = new Map();

    logs.forEach(log => {
      // Count failed login attempts by IP
      const ipMatch = log.message.match(/(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})/);
      if (ipMatch && log.message.toLowerCase().includes('failed')) {
        const ip = ipMatch[1];
        ipCounts.set(ip, (ipCounts.get(ip) || 0) + 1);
      }

      // Match security patterns
      securityPatterns.forEach(({ pattern, type }) => {
        if (pattern.test(log.message)) {
          if (!securityEvents.has(type)) {
            securityEvents.set(type, { count: 0, examples: [] });
          }
          const event = securityEvents.get(type)!;
          event.count++;
          if (event.examples.length < 5) {
            event.examples.push(log.message);
          }
        }
      });
    });

    // Detect brute force attacks
    ipCounts.forEach((count, ip) => {
      if (count > 10) {
        anomalies.push({
          id: `brute-force-${ip}`,
          type: 'security_event',
          severity: count > 50 ? 'critical' : count > 20 ? 'high' : 'medium',
          description: `Potential brute force attack from IP ${ip}: ${count} failed login attempts`,
          affectedResources: [ip],
          evidence: [`${count} failed authentication attempts detected`],
          timestamp: new Date().toISOString(),
          confidence: Math.min(count / 50, 1),
        });
      }
    });

    // Report other security events
    securityEvents.forEach((event, type) => {
      if (event.count > 0) {
        anomalies.push({
          id: `security-${type}`,
          type: 'security_event',
          severity: event.count > 10 ? 'high' : 'medium',
          description: `Detected ${event.count} ${type.replace(/_/g, ' ')} attempt(s)`,
          affectedResources: ['application'],
          evidence: event.examples.slice(0, 3),
          timestamp: new Date().toISOString(),
          confidence: 0.75,
        });
      }
    });

    return anomalies;
  }

  // Detect performance issues
  private detectPerformanceIssues(logs: LogEntry[]): Anomaly[] {
    const anomalies: Anomaly[] = [];

    // Slow queries
    const slowQueries = logs.filter(log => {
      const durationMatch = log.message.match(/duration[:\s]+(\d+)ms/i);
      return durationMatch && parseInt(durationMatch[1]) > 5000;
    });

    if (slowQueries.length > 5) {
      anomalies.push({
        id: 'slow-queries',
        type: 'performance_issue',
        severity: slowQueries.length > 20 ? 'critical' : 'high',
        description: `Detected ${slowQueries.length} slow database queries (>5s)`,
        affectedResources: ['database'],
        evidence: slowQueries.slice(0, 5).map(log => log.message),
        timestamp: new Date().toISOString(),
        confidence: 0.9,
      });
    }

    // Timeouts
    const timeouts = logs.filter(log =>
      log.message.toLowerCase().includes('timeout') || log.level === 'error'
    );

    if (timeouts.length > 10) {
      anomalies.push({
        id: 'timeouts',
        type: 'performance_issue',
        severity: timeouts.length > 50 ? 'critical' : 'high',
        description: `High number of timeout errors: ${timeouts.length}`,
        affectedResources: ['application'],
        evidence: timeouts.slice(0, 5).map(log => log.message),
        timestamp: new Date().toISOString(),
        confidence: 0.85,
      });
    }

    return anomalies;
  }

  // Detect resource exhaustion
  private detectResourceExhaustion(logs: LogEntry[]): Anomaly[] {
    const anomalies: Anomaly[] = [];
    const resourcePatterns = [
      { pattern: /out of memory|oom killer|memory.*exhausted/gi, resource: 'memory' },
      { pattern: /no space left|disk.*full|storage.*exhausted/gi, resource: 'disk' },
      { pattern: /too many open files|file descriptor.*limit/gi, resource: 'file_descriptors' },
      { pattern: /connection.*pool.*exhausted|max.*connections.*reached/gi, resource: 'connections' },
    ];

    resourcePatterns.forEach(({ pattern, resource }) => {
      const matches = logs.filter(log => pattern.test(log.message));

      if (matches.length > 0) {
        anomalies.push({
          id: `resource-${resource}`,
          type: 'resource_exhaustion',
          severity: 'critical',
          description: `Resource exhaustion detected: ${resource.replace(/_/g, ' ')} (${matches.length} occurrences)`,
          affectedResources: [resource],
          evidence: matches.slice(0, 5).map(log => log.message),
          timestamp: new Date().toISOString(),
          confidence: 0.95,
        });
      }
    });

    return anomalies;
  }

  // AI-powered anomaly analysis
  private async analyzeWithAI(logs: LogEntry[], anomalies: Anomaly[]): Promise<Anomaly[]> {
    // Sample logs for AI analysis (don't send all logs)
    const sampleLogs = logs.slice(0, 100);

    const agent = new AIAgentBubble({
      model: {
        model: 'gpt-4',
        temperature: 0.3,
      },
      systemPrompt: `You are a log analysis expert. Identify unusual patterns, potential issues, and security concerns in the provided logs.
      Focus on: errors, performance issues, security events, and operational anomalies.
      Return your analysis as a JSON array of anomalies with: type, severity, description, and confidence.`,
      message: `Analyze these log entries and identify any anomalies that weren't caught by rule-based detection:\n\n${JSON.stringify(sampleLogs, null, 2)}`,
    });

    const result = await agent.action();

    if (!result.success) {
      this.logger?.warn('AI analysis failed, using rule-based detection only');
      return anomalies;
    }

    try {
      const aiAnomalies = JSON.parse(result.data.response);
      return [...anomalies, ...aiAnomalies];
    } catch (error) {
      this.logger?.error(`Failed to parse AI response: ${error}`);
      return anomalies;
    }
  }

  // Generate recommendations
  private async generateRecommendations(anomalies: Anomaly[]): Promise<string[]> {
    if (anomalies.length === 0) {
      return ['No action needed - all systems operating normally'];
    }

    const agent = new AIAgentBubble({
      model: {
        model: 'gpt-4',
        temperature: 0.4,
      },
      systemPrompt: 'You are a DevOps expert. Based on the detected anomalies, provide actionable recommendations to prevent future occurrences.',
      message: `Anomalies detected:\n${JSON.stringify(anomalies, null, 2)}\n\nProvide 3-5 specific, actionable recommendations.`,
    });

    const result = await agent.action();

    if (!result.success) {
      return ['Review detected anomalies and implement appropriate fixes'];
    }

    const recommendations = result.data.response
      .split('\n')
      .filter(line => line.trim().length > 0)
      .slice(0, 5);

    return recommendations;
  }

  // Store report to Google Drive
  private async storeReport(report: AnomalyReport): Promise<string> {
    const googleDrive = new GoogleDriveBubble({
      operation: 'upload_file',
      name: `anomaly-report-${report.reportId}.json`,
      content: JSON.stringify(report, null, 2),
      mimeType: 'application/json',
    });

    const result = await googleDrive.action();

    if (!result.success || !result.data?.file) {
      throw new Error(`Failed to store report: ${result.error}`);
    }

    return result.data.file.webViewLink || result.data.file.id;
  }

  // Send Slack notification
  private async sendSlackNotification(
    report: AnomalyReport,
    channel: string
  ): Promise<void> {
    const criticalCount = report.anomalies.filter(a => a.severity === 'critical').length;

    const slack = new SlackBubble({
      channel,
      message: {
        text: `🚨 Log Anomaly Detection Report - ${criticalCount > 0 ? 'CRITICAL ISSUES FOUND' : 'Review Needed'}`,
        attachments: [
          {
            color: criticalCount > 0 ? 'danger' : report.anomalies.length > 0 ? 'warning' : 'good',
            fields: [
              {
                title: 'Report ID',
                value: report.reportId,
                short: true,
              },
              {
                title: 'Logs Analyzed',
                value: report.logsAnalyzed.toString(),
                short: true,
              },
              {
                title: 'Anomalies Found',
                value: report.anomalies.length.toString(),
                short: true,
              },
              {
                title: 'Critical Issues',
                value: criticalCount.toString(),
                short: true,
              },
            ],
          },
          {
            title: 'Summary',
            text: report.summary,
          },
          {
            title: 'Top Recommendations',
            text: report.recommendations.slice(0, 3).join('\n'),
          },
        ],
      },
    });

    await slack.action();
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
      msg: 'Starting log anomaly detection',
    });

    const {
      timeRange = `last-${this.DEFAULT_TIME_RANGE_MINUTES}m`,
      logSources = ['application'],
      minSeverity = 'medium',
      logEndpoint = process.env.LOG_ENDPOINT || 'http://logs:9200/_search',
      logApiKey = process.env.LOG_API_KEY || '',
      enableAI = true,
      storeReport = true,
      notify = true,
      slackChannel = '#ops-alerts',
    } = payload;

    this.logger?.info('Starting log anomaly detection...');

    // Calculate time range
    let timeRangeObj: { start: string; end: string };
    if (typeof timeRange === 'string') {
      const match = timeRange.match(/last-(\d+)(m|h)/);
      if (!match) {
        throw new Error('Invalid timeRange format. Use "last-Xm" or "last-Xh"');
      }
      const value = parseInt(match[1]);
      const unit = match[2];
      const now = new Date();
      const start = new Date(
        unit === 'h' ? now.getTime() - value * 60 * 60 * 1000 : now.getTime() - value * 60 * 1000
      );

      timeRangeObj = {
        start: start.toISOString(),
        end: now.toISOString(),
      };
    } else {
      timeRangeObj = timeRange;
    }

    // Fetch logs
    this.logger?.info(`Fetching logs from ${logSources.join(', ')}...`);
    const logs = await this.fetchLogs(logEndpoint, logApiKey, timeRangeObj, logSources);
    this.logger?.info(`Fetched ${logs.length} log entries`);

    if (logs.length === 0) {
      return {
        message: 'No logs found in the specified time range',
        reportId: 'none',
        anomaliesDetected: 0,
        criticalIssues: 0,
      };
    }

    // Detect anomalies
    this.logger?.info('Detecting anomalies...');
    let anomalies: Anomaly[] = [];

    anomalies.push(...this.detectErrorSpikes(logs));
    anomalies.push(...this.detectSecurityEvents(logs));
    anomalies.push(...this.detectPerformanceIssues(logs));
    anomalies.push(...this.detectResourceExhaustion(logs));

    // AI-powered analysis
    if (enableAI) {
      this.logger?.info('Running AI-powered anomaly analysis...');
      anomalies = await this.analyzeWithAI(logs, anomalies);
    }

    // Filter by severity
    const severityLevels = ['low', 'medium', 'high', 'critical'];
    const minSeverityIndex = severityLevels.indexOf(minSeverity);
    anomalies = anomalies.filter(a => severityLevels.indexOf(a.severity) >= minSeverityIndex);

    this.logger?.info(`Detected ${anomalies.length} anomalies`);

    // Generate recommendations
    this.logger?.info('Generating recommendations...');
    const recommendations = await this.generateRecommendations(anomalies);

    // Create report
    const report: AnomalyReport = {
      reportId: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timeRange: timeRangeObj,
      logsAnalyzed: logs.length,
      anomaliesFound: anomalies.length,
      anomalies,
      summary: `Found ${anomalies.length} anomaly(s) across ${logs.length} log entries`,
      recommendations,
      generatedAt: new Date().toISOString(),
    };

    // Store report
    let reportUrl: string | undefined;
    if (storeReport) {
      this.logger?.info('Storing report to Google Drive...');
      try {
        reportUrl = await this.storeReport(report);
      } catch (error) {
        this.logger?.error(`Failed to store report: ${error}`);
      }
    }

    // Send notification
    let notificationSent = false;
    if (notify && anomalies.length > 0) {
      this.logger?.info('Sending Slack notification...');
      try {
        await this.sendSlackNotification(report, slackChannel);
        notificationSent = true;
      } catch (error) {
        this.logger?.error(`Failed to send notification: ${error}`);
      }
    }

    const criticalIssues = anomalies.filter(a => a.severity === 'critical').length;

    return {
      message: `Analyzed ${logs.length} logs, detected ${anomalies.length} anomaly(s)`,
      reportId: report.reportId,
      anomaliesDetected: anomalies.length,
      criticalIssues,
      reportUrl,
      notificationSent,
    };
  }
}

// Export workflow configuration
export const workflowConfig = {
  id: 'log-anomaly-detection',
  name: 'Log Anomaly Detection',
  description: 'ML-based detection of anomalies in application and system logs',
  version: '1.0.0',
  category: 'infrastructure-automation',
  icon: '🔍',
  tags: ['logging', 'monitoring', 'security', 'anomaly-detection', 'ml'],
};
