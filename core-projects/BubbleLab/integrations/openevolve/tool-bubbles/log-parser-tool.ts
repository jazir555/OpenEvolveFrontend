/**
 * Log Parser Tool Bubble
 *
 * Parses and analyzes OpenEvolve service logs with support for
 * multiple formats, error detection, and pattern recognition.
 */

import { z } from 'zod';
import { HttpBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';

const LogFormatSchema = z.enum([
  'json',
  'apache',
  'nginx',
  'syslog',
  'common',
  'combined',
  'auto',
]);

const LogLevelSchema = z.enum([
  'all',
  'debug',
  'info',
  'warn',
  'error',
  'fatal',
  'trace',
]);

const LogParserParamsSchema = z.object({
  operation: z.enum(['parse', 'analyze', 'filter', 'aggregate', 'export']).describe('Parser operation'),

  // Input
  logData: z.string().optional().describe('Raw log data to parse'),
  logFile: z.string().optional().describe('Path to log file'),
  logUrl: z.string().url().optional().describe('URL to fetch logs from'),

  // Parsing options
  format: LogFormatSchema.default('auto').describe('Log format'),
  encoding: z.string().default('utf-8').describe('Log file encoding'),
  multiline: z.boolean().default(false).describe('Enable multiline log parsing'),

  // Filtering
  filterLevel: LogLevelSchema.default('all').describe('Filter by log level'),
  filterPattern: z.string().optional().describe('Regex pattern to filter logs'),
  filterTimeRange: z.object({
    start: z.string().optional().describe('Start time (ISO 8601)'),
    end: z.string().optional().describe('End time (ISO 8601)'),
  }).optional().describe('Time range filter'),

  // Analysis
  detectAnomalies: z.boolean().default(false).describe('Detect anomalies in logs'),
  extractMetrics: z.boolean().default(true).describe('Extract metrics from logs'),
  identifyErrors: z.boolean().default(true).describe('Identify error patterns'),

  // Aggregation
  groupBy: z.array(z.string()).optional().describe('Fields to group by'),
  aggregations: z.array(z.object({
    field: z.string(),
    operation: z.enum(['count', 'sum', 'avg', 'min', 'max', 'distinct']),
  })).optional().describe('Aggregations to perform'),

  // Export
  exportFormat: z.enum(['json', 'csv', 'html', 'markdown']).default('json'),
  includeRaw: z.boolean().default(false).describe('Include raw logs in output'),
});

type LogParserParamsInput = z.input<typeof LogParserParamsSchema>;
type LogParserParams = z.output<typeof LogParserParamsSchema>;

const LogEntrySchema = z.object({
  timestamp: z.string(),
  level: z.string(),
  message: z.string(),
  service: z.string().optional(),
  correlationId: z.string().optional(),
  metadata: z.record(z.unknown()).optional(),
  raw: z.string().optional(),
});

const LogParserResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  entries: z.array(LogEntrySchema).optional().describe('Parsed log entries'),
  count: z.number().optional().describe('Total log entries'),
  errorCount: z.number().optional().describe('Number of error entries'),
  metrics: z.record(z.number()).optional().describe('Extracted metrics'),
  anomalies: z.array(z.object({
    type: z.string(),
    description: z.string(),
    timestamp: z.string(),
    severity: z.string(),
  })).optional().describe('Detected anomalies'),
  errors: z.array(z.object({
    pattern: z.string(),
    count: z.number(),
    samples: z.array(z.string()),
  })).optional().describe('Identified error patterns'),
  aggregations: z.record(z.unknown()).optional().describe('Aggregated results'),
  export: z.string().optional().describe('Exported data'),
  summary: z.object({
    timeRange: z.tuple([z.string(), z.string()]),
    levelCounts: z.record(z.number()),
    topErrors: z.array(z.object({
      error: z.string(),
      count: z.number(),
    })),
    services: z.array(z.string()).optional(),
  }).optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type LogParserResult = z.output<typeof LogParserResultSchema>;

export class LogParserTool {
  private http: HttpBubble;
  private params: LogParserParams;
  private context?: BubbleContext;

  constructor(params: LogParserParamsInput, context?: BubbleContext) {
    this.params = LogParserParamsSchema.parse(params);
    this.context = context;

    this.http = new HttpBubble({
      url: 'http://localhost:8000',
      method: 'GET',
      timeout: 30000,
    }, context);
  }

  private async loadLogs(): Promise<string> {
    if (this.params.logData) {
      return this.params.logData;
    }

    if (this.params.logUrl) {
      const response = await fetch(this.params.logUrl);
      return await response.text();
    }

    if (this.params.logFile) {
      const response = await fetch(`http://localhost:8000/api/logs/file`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file: this.params.logFile }),
      });
      const data = await response.json();
      return data.content;
    }

    throw new Error('Either logData, logFile, or logUrl must be provided');
  }

  private parseJsonLine(line: string): any {
    try {
      return JSON.parse(line);
    } catch {
      return { message: line };
    }
  }

  private detectLogLevel(entry: any): string {
    if (entry.level) return entry.level;
    if (entry.severity) return entry.severity;

    const message = entry.message || '';
    const lower = message.toLowerCase();

    if (lower.includes('error') || lower.includes('err')) return 'error';
    if (lower.includes('warn')) return 'warn';
    if (lower.includes('fatal') || lower.includes('crit')) return 'fatal';
    if (lower.includes('debug')) return 'debug';
    if (lower.includes('trace')) return 'trace';
    if (lower.includes('info')) return 'info';

    return 'info';
  }

  private extractTimestamp(entry: any): string {
    if (entry.timestamp) return entry.timestamp;
    if (entry.time) return entry.time;
    if (entry.date) return entry.date;
    if (entry['@timestamp']) return entry['@timestamp'];

    return new Date().toISOString();
  }

  public async parse(): Promise<LogParserResult> {
    const startTime = Date.now();

    try {
      const logContent = await this.loadLogs();
      const lines = logContent.split('\n').filter(line => line.trim());

      const entries: any[] = [];
      let errorCount = 0;

      for (const line of lines) {
        let entry: any;

        if (this.params.format === 'json') {
          entry = this.parseJsonLine(line);
        } else {
          entry = { message: line };
        }

        entry.level = this.detectLogLevel(entry);
        entry.timestamp = this.extractTimestamp(entry);

        if (entry.level === 'error' || entry.level === 'fatal') {
          errorCount++;
        }

        entries.push(entry);
      }

      const timing = Date.now() - startTime;

      return {
        success: true,
        operation: 'parse',
        entries,
        count: entries.length,
        errorCount,
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'parse',
        error: errorMessage,
        timing,
      };
    }
  }

  public async analyze(): Promise<LogParserResult> {
    const parseResult = await this.parse();

    if (!parseResult.success || !parseResult.entries) {
      return parseResult;
    }

    const startTime = Date.now();
    const entries = parseResult.entries;

    // Calculate metrics
    const metrics: Record<string, number> = {};
    const levelCounts: Record<string, number> = {};
    const errorPatterns: Record<string, { count: number; samples: string[] }> = {};
    const services = new Set<string>();

    for (const entry of entries) {
      // Count by level
      const level = entry.level || 'info';
      levelCounts[level] = (levelCounts[level] || 0) + 1;

      // Track services
      if (entry.service) {
        services.add(entry.service);
      }

      // Identify error patterns
      if (entry.level === 'error' || entry.level === 'fatal') {
        const message = entry.message || '';
        const pattern = message.substring(0, 100); // First 100 chars

        if (!errorPatterns[pattern]) {
          errorPatterns[pattern] = { count: 0, samples: [] };
        }
        errorPatterns[pattern].count++;
        if (errorPatterns[pattern].samples.length < 5) {
          errorPatterns[pattern].samples.push(message);
        }
      }
    }

    metrics.totalEntries = entries.length;
    metrics.errorCount = levelCounts.error || 0;
    metrics.fatalCount = levelCounts.fatal || 0;
    metrics.warnCount = levelCounts.warn || 0;
    metrics.infoCount = levelCounts.info || 0;
    metrics.debugCount = levelCounts.debug || 0;

    // Top errors
    const topErrors = Object.entries(errorPatterns)
      .sort(([, a], [, b]) => b.count - a.count)
      .slice(0, 10)
      .map(([error, data]) => ({
        error,
        count: data.count,
      }));

    const timing = Date.now() - startTime + (parseResult.timing || 0);

    return {
      ...parseResult,
      operation: 'analyze',
      metrics,
      summary: {
        timeRange: [
          entries[0]?.timestamp || new Date().toISOString(),
          entries[entries.length - 1]?.timestamp || new Date().toISOString(),
        ],
        levelCounts,
        topErrors,
        services: Array.from(services),
      },
      errors: Object.entries(errorPatterns).map(([pattern, data]) => ({
        pattern,
        count: data.count,
        samples: data.samples,
      })),
      timing,
    };
  }

  public async action(): Promise<LogParserResult> {
    switch (this.params.operation) {
      case 'parse':
        return this.parse();
      case 'analyze':
        return this.analyze();
      default:
        return {
          success: false,
          operation: this.params.operation,
          error: `Unknown operation: ${this.params.operation}`,
          timing: 0,
        };
    }
  }
}

export default LogParserTool;
