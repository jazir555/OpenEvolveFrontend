/**
 * Comprehensive Error Logging and Reporting Utility
 * Provides centralized error logging with multiple destinations and reporting mechanisms
 */

import { toast } from 'react-toastify';

// Define error severity levels
export type ErrorSeverity = 'debug' | 'info' | 'warn' | 'error' | 'critical';

// Define error context interface
export interface ErrorContext {
  component?: string;
  function?: string;
  userId?: string;
  sessionId?: string;
  url?: string;
  userAgent?: string;
  timestamp?: Date;
  additionalData?: Record<string, any>;
}

// Define error report interface
export interface ErrorReport {
  id: string;
  message: string;
  stack?: string;
  severity: ErrorSeverity;
  context: ErrorContext;
  timestamp: Date;
  handled: boolean;
}

// Define error logger configuration
export interface ErrorLoggerConfig {
  logToConsole: boolean;
  logToFile: boolean;
  logToRemote: boolean;
  remoteEndpoint?: string;
  maxFileSize?: number; // in KB
  maxFiles?: number;
  reportCriticalOnly: boolean;
  enableToastNotifications: boolean;
  toastDuration?: number;
}

// Default configuration
const DEFAULT_CONFIG: ErrorLoggerConfig = {
  logToConsole: true,
  logToFile: false,
  logToRemote: false,
  remoteEndpoint: '/api/logs',
  maxFileSize: 1024, // 1MB
  maxFiles: 5,
  reportCriticalOnly: false,
  enableToastNotifications: true,
  toastDuration: 5000,
};

/**
 * Error Logger Class
 * Provides centralized error logging with multiple destinations
 */
export class ErrorLogger {
  private config: ErrorLoggerConfig;
  private reports: ErrorReport[] = [];
  private readonly MAX_REPORTS = 1000; // Maximum number of reports to keep in memory

  constructor(config?: Partial<ErrorLoggerConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Log an error with context and severity
   */
  logError(
    error: Error | string,
    severity: ErrorSeverity = 'error',
    context: ErrorContext = {}
  ): void {
    try {
      const message = typeof error === 'string' ? error : error.message;
      const stack = typeof error !== 'string' ? error.stack : undefined;
      
      // Skip logging if it's below critical threshold and we're only reporting critical errors
      if (this.config.reportCriticalOnly && !['critical', 'error'].includes(severity)) {
        return;
      }

      const report: ErrorReport = {
        id: this.generateErrorId(),
        message,
        stack,
        severity,
        context: {
          ...context,
          timestamp: context.timestamp || new Date(),
          userAgent: context.userAgent || (typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown'),
          url: context.url || (typeof window !== 'undefined' ? window.location.href : 'unknown'),
        },
        timestamp: new Date(),
        handled: true,
      };

      // Add to internal reports
      this.addToReports(report);

      // Log to console if enabled
      if (this.config.logToConsole) {
        this.logToConsole(report);
      }

      // Show toast notification if enabled
      if (this.config.enableToastNotifications) {
        this.showToastNotification(report);
      }

      // Log to file if enabled
      if (this.config.logToFile) {
        this.logToFile(report);
      }

      // Send to remote endpoint if enabled
      if (this.config.logToRemote && this.config.remoteEndpoint) {
        this.logToRemote(report);
      }
    } catch (loggingError) {
      // If logging itself fails, at least log to console
      console.error('Failed to log error:', loggingError);
      console.error('Original error:', error);
    }
  }

  /**
   * Log an unhandled error (for use with window.onerror, etc.)
   */
  logUnhandledError(
    message: string,
    source?: string,
    lineno?: number,
    colno?: number,
    error?: Error
  ): void {
    const fullMessage = error ? `${message}: ${error.message}` : message;
    const context: ErrorContext = {
      component: 'Global',
      function: 'UnhandledError',
      url: source,
      additionalData: {
        source,
        lineno,
        colno,
        originalError: error?.message,
      },
    };

    this.logError(fullMessage, 'error', context);
  }

  /**
   * Get recent error reports
   */
  getRecentReports(count: number = 10): ErrorReport[] {
    return [...this.reports].slice(-count).reverse();
  }

  /**
   * Get error reports by severity
   */
  getReportsBySeverity(severity: ErrorSeverity): ErrorReport[] {
    return this.reports.filter(report => report.severity === severity);
  }

  /**
   * Get error count by severity
   */
  getErrorCountBySeverity(severity: ErrorSeverity): number {
    return this.reports.filter(report => report.severity === severity).length;
  }

  /**
   * Clear all error reports
   */
  clearReports(): void {
    this.reports = [];
  }

  /**
   * Update logger configuration
   */
  updateConfig(config: Partial<ErrorLoggerConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Generate a unique error ID
   */
  private generateErrorId(): string {
    return `err_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Add report to internal storage
   */
  private addToReports(report: ErrorReport): void {
    this.reports.push(report);
    
    // Keep only the most recent reports
    if (this.reports.length > this.MAX_REPORTS) {
      this.reports = this.reports.slice(-this.MAX_REPORTS);
    }
  }

  /**
   * Log to console with appropriate level
   */
  private logToConsole(report: ErrorReport): void {
    const logMessage = `[${report.severity.toUpperCase()}] ${report.message}`;
    const logContext = {
      id: report.id,
      context: report.context,
      timestamp: report.timestamp.toISOString(),
    };

    switch (report.severity) {
      case 'debug':
        console.debug(logMessage, logContext);
        break;
      case 'info':
        console.info(logMessage, logContext);
        break;
      case 'warn':
        console.warn(logMessage, logContext);
        if (report.stack) console.warn('Stack trace:', report.stack);
        break;
      case 'error':
        console.error(logMessage, logContext);
        if (report.stack) console.error('Stack trace:', report.stack);
        break;
      case 'critical':
        console.error(`CRITICAL: ${logMessage}`, logContext);
        if (report.stack) console.error('Stack trace:', report.stack);
        break;
    }
  }

  /**
   * Show toast notification for errors
   */
  private showToastNotification(report: ErrorReport): void {
    const message = `${report.severity.toUpperCase()}: ${report.message}`;
    
    switch (report.severity) {
      case 'debug':
      case 'info':
        toast.info(message, { autoClose: this.config.toastDuration });
        break;
      case 'warn':
        toast.warn(message, { autoClose: this.config.toastDuration });
        break;
      case 'error':
        toast.error(message, { autoClose: this.config.toastDuration });
        break;
      case 'critical':
        toast.error(`🚨 CRITICAL: ${report.message}`, { 
          autoClose: this.config.toastDuration,
          className: 'critical-error-toast'
        });
        break;
    }
  }

  /**
   * Log to file (simulated - in browser this would use localStorage or IndexedDB)
   */
  private logToFile(report: ErrorReport): void {
    try {
      // In a real implementation, this would write to a file
      // For browser environments, we'll use localStorage as a simulation
      if (typeof window !== 'undefined' && window.localStorage) {
        const key = 'openevolve_error_log';
        const existingLogs = JSON.parse(window.localStorage.getItem(key) || '[]');
        
        // Add new report
        existingLogs.push({
          ...report,
          timestamp: report.timestamp.toISOString(),
          context: {
            ...report.context,
            timestamp: report.context.timestamp?.toISOString(),
          }
        });
        
        // Keep only recent logs to prevent storage overflow
        const recentLogs = existingLogs.slice(-(this.config.maxFiles || 5) * 20); // 20 per file estimate
        
        window.localStorage.setItem(key, JSON.stringify(recentLogs));
      }
    } catch (fileError) {
      console.error('Failed to log to file/storage:', fileError);
    }
  }

  /**
   * Send error report to remote endpoint
   */
  private async logToRemote(report: ErrorReport): Promise<void> {
    if (!this.config.remoteEndpoint) return;

    try {
      const response = await fetch(this.config.remoteEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...report,
          timestamp: report.timestamp.toISOString(),
          context: {
            ...report.context,
            timestamp: report.context.timestamp?.toISOString(),
          }
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (remoteError) {
      console.error('Failed to send error to remote endpoint:', remoteError);
    }
  }

  /**
   * Capture and log promise rejections
   */
  captureUnhandledRejection(event: PromiseRejectionEvent): void {
    this.logError(
      event.reason instanceof Error ? event.reason : String(event.reason),
      'error',
      { component: 'Promise', function: 'UnhandledRejection' }
    );
  }

  /**
   * Capture and log uncaught exceptions
   */
  captureUncaughtException(event: ErrorEvent): void {
    this.logError(
      event.error instanceof Error ? event.error : String(event.error),
      'critical',
      {
        component: 'Global',
        function: 'UncaughtException',
        url: event.filename,
        additionalData: {
          lineno: event.lineno,
          colno: event.colno,
        },
      }
    );
  }
}

// Create a singleton instance
const errorLogger = new ErrorLogger();

// Set up global error handlers if in browser environment
if (typeof window !== 'undefined') {
  window.addEventListener('error', (event) => {
    errorLogger.captureUncaughtException(event);
  });

  window.addEventListener('unhandledrejection', (event) => {
    errorLogger.captureUnhandledRejection(event);
  });
}

export default errorLogger;