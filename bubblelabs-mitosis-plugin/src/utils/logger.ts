/**
 * Enhanced logging utility for the Mitosis Plugin
 * Provides comprehensive logging with different log levels and debugging capabilities
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LoggerConfig {
  level: LogLevel;
  enabled: boolean;
  prefix?: string;
}

class MitosisLogger {
  private config: LoggerConfig;
  private readonly logLevels: Record<LogLevel, number> = {
    debug: 0,
    info: 1,
    warn: 2,
    error: 3
  };

  constructor(config?: Partial<LoggerConfig>) {
    this.config = {
      level: config?.level || 'info',
      enabled: config?.enabled !== undefined ? config.enabled : true,
      prefix: config?.prefix || 'MitosisPlugin'
    };
  }

  private shouldLog(level: LogLevel): boolean {
    if (!this.config.enabled) {
      return false;
    }
    
    const currentLevel = this.logLevels[this.config.level];
    const messageLevel = this.logLevels[level];
    
    return messageLevel >= currentLevel;
  }

  private formatMessage(level: LogLevel, message: string, ...args: unknown[]): string {
    try {
      const timestamp = new Date().toISOString();
      const prefix = this.config.prefix ? `[${this.config.prefix}]` : '';
      return `${timestamp} ${prefix} [${level.toUpperCase()}] ${message}`;
    } catch (formatError) {
      // Fallback if formatting fails
      return `[LOG_ERROR] ${message}`;
    }
  }

  debug(message: string, ...args: unknown[]): void {
    try {
      if (this.shouldLog('debug')) {
        if (typeof console.debug === 'function') {
          console.debug(this.formatMessage('debug', message, ...args), ...args);
        } else {
          // Fallback to console.log if debug is not available
          console.log(this.formatMessage('debug', message, ...args), ...args);
        }
      }
    } catch (logError) {
      // If logging fails, at least try to output the raw message
      try {
        console.log(`[LOG_ERROR] Debug: ${message}`, ...args);
      } catch (fallbackError) {
        // If all logging fails, silently continue
      }
    }
  }

  info(message: string, ...args: unknown[]): void {
    try {
      if (this.shouldLog('info')) {
        if (typeof console.info === 'function') {
          console.info(this.formatMessage('info', message, ...args), ...args);
        } else {
          // Fallback to console.log if info is not available
          console.log(this.formatMessage('info', message, ...args), ...args);
        }
      }
    } catch (logError) {
      // If logging fails, at least try to output the raw message
      try {
        console.log(`[LOG_ERROR] Info: ${message}`, ...args);
      } catch (fallbackError) {
        // If all logging fails, silently continue
      }
    }
  }

  warn(message: string, ...args: unknown[]): void {
    try {
      if (this.shouldLog('warn')) {
        if (typeof console.warn === 'function') {
          console.warn(this.formatMessage('warn', message, ...args), ...args);
        } else {
          // Fallback to console.log if warn is not available
          console.log(this.formatMessage('warn', message, ...args), ...args);
        }
      }
    } catch (logError) {
      // If logging fails, at least try to output the raw message
      try {
        console.log(`[LOG_ERROR] Warn: ${message}`, ...args);
      } catch (fallbackError) {
        // If all logging fails, silently continue
      }
    }
  }

  error(message: string, ...args: unknown[]): void {
    try {
      if (this.shouldLog('error')) {
        if (typeof console.error === 'function') {
          console.error(this.formatMessage('error', message, ...args), ...args);
        } else {
          // Fallback to console.log if error is not available
          console.log(this.formatMessage('error', message, ...args), ...args);
        }
      }
    } catch (logError) {
      // If logging fails, at least try to output the raw message
      try {
        console.log(`[LOG_ERROR] Error: ${message}`, ...args);
      } catch (fallbackError) {
        // If all logging fails, silently continue
        // This ensures errors in logging don't break the application
      }
    }
  }

  setLevel(level: LogLevel): void {
    this.config.level = level;
  }

  setEnabled(enabled: boolean): void {
    this.config.enabled = enabled;
  }
}

// Create a singleton logger instance
export const logger = new MitosisLogger();

// Export the logger class for creating custom loggers if needed
export { MitosisLogger };