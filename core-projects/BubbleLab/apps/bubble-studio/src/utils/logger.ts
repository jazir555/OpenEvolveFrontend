/**
 * Structured Logger Utility
 *
 * Implements CLAUDE.md observability requirements:
 * - JSON Lines format (jsonl)
 * - Context: correlation_id, source_service, target_service
 * - Consistent logging across the application
 *
 * @example
 * ```typescript
 * logger.info({
 *   msg: 'Making API request',
 *   correlation_id: correlationId,
 *   endpoint: '/api/users',
 *   method: 'GET'
 * });
 * ```
 */

export interface LogContext {
  msg: string;
  correlation_id?: string;
  source_service?: string;
  target_service?: string;
  endpoint?: string;
  method?: string;
  status?: number | string;
  error?: string;
  [key: string]: unknown;
}

export class Logger {
  private sourceService: string;

  constructor(sourceService: string) {
    this.sourceService = sourceService;
  }

  /**
   * Log informational message
   */
  info(context: LogContext): void {
    this.log('info', context);
  }

  /**
   * Log warning message
   */
  warn(context: LogContext): void {
    this.log('warn', context);
  }

  /**
   * Log error message
   */
  error(context: LogContext): void {
    this.log('error', context);
  }

  /**
   * Log debug message (only in development)
   */
  debug(context: LogContext): void {
    if (process.env.NODE_ENV === 'development') {
      this.log('debug', context);
    }
  }

  private log(level: string, context: LogContext): void {
    const logEntry = {
      level,
      timestamp: new Date().toISOString(),
      source_service: context.source_service || this.sourceService,
      ...context,
    };

    // Output in JSON Lines format for parsing
    const logLine = JSON.stringify(logEntry);

    switch (level) {
      case 'error':
        console.error(logLine);
        break;
      case 'warn':
        console.warn(logLine);
        break;
      case 'debug':
        console.debug(logLine);
        break;
      default:
        console.log(logLine);
    }
  }
}

/**
 * Default logger instance for bubble-studio
 */
export const logger = new Logger('bubble-studio');

/**
 * Create a logger with a specific source service
 */
export function createLogger(sourceService: string): Logger {
  return new Logger(sourceService);
}
