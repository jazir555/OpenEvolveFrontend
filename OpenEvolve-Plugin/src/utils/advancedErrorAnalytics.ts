/**
 * Advanced Error Analytics
 * Provides comprehensive error tracking, analysis, and visualization
 */

import { errorLogger, ComprehensiveErrorHandler } from '@/utils';
import { ErrorReport } from '@/utils/errorLogging';

/**
 * Error analytics configuration
 */
export interface ErrorAnalyticsConfig {
  /**
   * Whether to collect error analytics
   * @default true
   */
  enabled: boolean;

  /**
   * Sample rate for error collection (0-1)
   * @default 1
   */
  sampleRate: number;

  /**
   * Maximum number of errors to store
   * @default 1000
   */
  maxErrors: number;

  /**
   * Whether to send analytics to remote endpoint
   * @default false
   */
  sendToRemote: boolean;

  /**
   * Remote endpoint for analytics
   * @default '/api/analytics/errors'
   */
  remoteEndpoint: string;

  /**
   * Interval for sending analytics (in ms)
   * @default 300000 (5 minutes)
   */
  sendInterval: number;
}

/**
 * Error trend data
 */
export interface ErrorTrend {
  timestamp: Date;
  count: number;
  severity: 'debug' | 'info' | 'warn' | 'error' | 'critical';
  category?: string;
}

/**
 * Error analytics summary
 */
export interface ErrorAnalyticsSummary {
  totalErrors: number;
  errorsBySeverity: Record<string, number>;
  errorsByCategory: Record<string, number>;
  errorRatePerMinute: number;
  topErrorMessages: Array<{ message: string; count: number }>;
  errorTrends: ErrorTrend[];
  recoveryRate: number;
  averageResolutionTime: number; // in milliseconds
}

/**
 * Advanced Error Analytics Class
 */
export class AdvancedErrorAnalytics {
  private config: ErrorAnalyticsConfig;
  private trends: ErrorTrend[] = [];
  private startTime: Date;
  private errorCounts: Map<string, number> = new Map();
  private errorMessages: Map<string, number> = new Map();
  private errorResolutionTimes: number[] = [];
  private sendIntervalId: NodeJS.Timeout | null = null;
  private comprehensiveErrorHandler: ComprehensiveErrorHandler;

  constructor(config?: Partial<ErrorAnalyticsConfig>) {
    this.config = {
      enabled: true,
      sampleRate: 1,
      maxErrors: 1000,
      sendToRemote: false,
      remoteEndpoint: '/api/analytics/errors',
      sendInterval: 300000, // 5 minutes
      ...config,
    };

    this.startTime = new Date();
    this.comprehensiveErrorHandler = new ComprehensiveErrorHandler();

    if (this.config.enabled) {
      this.startTracking();
    }
  }

  /**
   * Start tracking errors
   */
  private startTracking(): void {
    // Listen for new error reports
    // Note: This is a simplified implementation
    // In a real app, you'd integrate with your error logging system
    // Development logging only - removed in production
  }

  /**
   * Track an error
   */
  trackError(error: Error | string, severity: 'debug' | 'info' | 'warn' | 'error' | 'critical' = 'error', category?: string): void {
    if (!this.config.enabled) return;

    // Sample the error based on sample rate
    if (Math.random() > this.config.sampleRate) return;

    const message = typeof error === 'string' ? error : error.message;
    const timestamp = new Date();

    // Update counts
    const countKey = `${severity}:${category || 'uncategorized'}`;
    this.errorCounts.set(countKey, (this.errorCounts.get(countKey) || 0) + 1);

    // Update message counts
    this.errorMessages.set(message, (this.errorMessages.get(message) || 0) + 1);

    // Add to trends
    this.trends.push({
      timestamp,
      count: 1,
      severity,
      category,
    });

    // Keep trends within max size
    if (this.trends.length > this.config.maxErrors) {
      this.trends = this.trends.slice(-this.config.maxErrors);
    }
  }

  /**
   * Track error resolution time
   */
  trackResolutionTime(timeMs: number): void {
    if (!this.config.enabled) return;

    this.errorResolutionTimes.push(timeMs);

    // Keep resolution times within max size
    if (this.errorResolutionTimes.length > this.config.maxErrors) {
      this.errorResolutionTimes = this.errorResolutionTimes.slice(-this.config.maxErrors);
    }
  }

  /**
   * Get error analytics summary
   */
  getSummary(): ErrorAnalyticsSummary {
    const now = new Date();
    const timeElapsedMinutes = (now.getTime() - this.startTime.getTime()) / (1000 * 60);
    const totalErrors = Array.from(this.errorCounts.values()).reduce((sum, count) => sum + count, 0);

    // Errors by severity
    const errorsBySeverity: Record<string, number> = {};
    for (const [key, count] of this.errorCounts.entries()) {
      const [severity] = key.split(':');
      errorsBySeverity[severity] = (errorsBySeverity[severity] || 0) + count;
    }

    // Errors by category
    const errorsByCategory: Record<string, number> = {};
    for (const [key, count] of this.errorCounts.entries()) {
      const [, category] = key.split(':');
      errorsByCategory[category] = (errorsByCategory[category] || 0) + count;
    }

    // Top error messages
    const topErrorMessages = Array.from(this.errorMessages.entries())
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10)
      .map(([message, count]) => ({ message, count }));

    // Recovery rate (simplified calculation)
    const recoveryRate = this.comprehensiveErrorHandler.getErrorStatistics().recoverySuccessRate;

    // Average resolution time
    const averageResolutionTime = this.errorResolutionTimes.length > 0
      ? this.errorResolutionTimes.reduce((sum, time) => sum + time, 0) / this.errorResolutionTimes.length
      : 0;

    return {
      totalErrors,
      errorsBySeverity,
      errorsByCategory,
      errorRatePerMinute: timeElapsedMinutes > 0 ? totalErrors / timeElapsedMinutes : 0,
      topErrorMessages,
      errorTrends: [...this.trends],
      recoveryRate,
      averageResolutionTime,
    };
  }

  /**
   * Get error trends for a specific time period
   */
  getTrends(hoursBack: number = 24): ErrorTrend[] {
    const cutoffTime = new Date(Date.now() - hoursBack * 60 * 60 * 1000);
    return this.trends.filter(trend => trend.timestamp > cutoffTime);
  }

  /**
   * Get error distribution by severity
   */
  getSeverityDistribution(): Record<string, number> {
    const distribution: Record<string, number> = {};
    for (const [key, count] of this.errorCounts.entries()) {
      const [severity] = key.split(':');
      distribution[severity] = (distribution[severity] || 0) + count;
    }
    return distribution;
  }

  /**
   * Get error distribution by category
   */
  getCategoryDistribution(): Record<string, number> {
    const distribution: Record<string, number> = {};
    for (const [key, count] of this.errorCounts.entries()) {
      const [, category] = key.split(':');
      distribution[category] = (distribution[category] || 0) + count;
    }
    return distribution;
  }

  /**
   * Get top recurring errors
   */
  getTopRecurringErrors(limit: number = 10): Array<{ message: string; count: number }> {
    return Array.from(this.errorMessages.entries())
      .sort(([, a], [, b]) => b - a)
      .slice(0, limit)
      .map(([message, count]) => ({ message, count }));
  }

  /**
   * Get error rate over time
   */
  getErrorRateOverTime(windowHours: number = 1): Array<{ time: Date; ratePerHour: number }> {
    const intervals: Array<{ time: Date; ratePerHour: number }> = [];
    const now = new Date();
    const startTime = new Date(now.getTime() - windowHours * 60 * 60 * 1000);

    // Group trends by hour
    const hourlyGroups = new Map<string, number>();
    for (const trend of this.trends) {
      if (trend.timestamp < startTime) continue;

      const hourKey = new Date(
        Date.UTC(
          trend.timestamp.getFullYear(),
          trend.timestamp.getMonth(),
          trend.timestamp.getDate(),
          trend.timestamp.getHours()
        )
      ).toISOString();

      hourlyGroups.set(hourKey, (hourlyGroups.get(hourKey) || 0) + trend.count);
    }

    // Convert to array
    for (const [hourKey, count] of hourlyGroups.entries()) {
      intervals.push({
        time: new Date(hourKey),
        ratePerHour: count,
      });
    }

    return intervals.sort((a, b) => a.time.getTime() - b.time.getTime());
  }

  /**
   * Start sending analytics to remote endpoint
   */
  startSendingToRemote(): void {
    if (!this.config.sendToRemote) return;

    this.sendIntervalId = setInterval(() => {
      this.sendAnalyticsToRemote();
    }, this.config.sendInterval);
  }

  /**
   * Stop sending analytics to remote endpoint
   */
  stopSendingToRemote(): void {
    if (this.sendIntervalId) {
      clearInterval(this.sendIntervalId);
      this.sendIntervalId = null;
    }
  }

  /**
   * Send analytics to remote endpoint
   */
  private async sendAnalyticsToRemote(): Promise<void> {
    if (!this.config.sendToRemote) return;

    try {
      const summary = this.getSummary();
      const payload = {
        timestamp: new Date().toISOString(),
        summary,
        config: this.config,
      };

      await fetch(this.config.remoteEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      // Successfully sent - no logging needed
    } catch (error) {
      errorLogger.logError('Failed to send analytics to remote endpoint', 'error', {
        component: 'AdvancedErrorAnalytics',
        function: 'sendAnalyticsToRemote',
        additionalData: { error }
      });
    }
  }

  /**
   * Reset analytics
   */
  reset(): void {
    this.trends = [];
    this.errorCounts.clear();
    this.errorMessages.clear();
    this.errorResolutionTimes = [];
    this.startTime = new Date();
  }

  /**
   * Enable analytics tracking
   */
  enable(): void {
    this.config.enabled = true;
  }

  /**
   * Disable analytics tracking
   */
  disable(): void {
    this.config.enabled = false;
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<ErrorAnalyticsConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  getConfig(): ErrorAnalyticsConfig {
    return { ...this.config };
  }

  /**
   * Destroy the analytics instance
   */
  destroy(): void {
    this.stopSendingToRemote();
    this.reset();
  }
}

/**
 * Global error analytics instance
 */
export const errorAnalytics = new AdvancedErrorAnalytics();

/**
 * Hook-like function for React components to access error analytics
 */
export function useErrorAnalytics(): AdvancedErrorAnalytics {
  return errorAnalytics;
}

/**
 * Function to track an error with analytics
 */
export function trackErrorWithAnalytics(
  error: Error | string,
  severity: 'debug' | 'info' | 'warn' | 'error' | 'critical' = 'error',
  category?: string
): void {
  errorAnalytics.trackError(error, severity, category);
}

/**
 * Function to track error resolution time
 */
export function trackErrorResolutionTime(timeMs: number): void {
  errorAnalytics.trackResolutionTime(timeMs);
}

// Initialize error analytics when module loads
if (typeof window !== 'undefined') {
  // Start sending to remote if configured
  if (errorAnalytics.getConfig().sendToRemote) {
    errorAnalytics.startSendingToRemote();
  }
}

// Example usage:
/*
// Track an error
trackErrorWithAnalytics(new Error('Something went wrong'), 'error', 'api');

// Track resolution time
trackErrorResolutionTime(1500); // 1.5 seconds

// Get analytics summary
const summary = errorAnalytics.getSummary();
// Use errorLogger instead of console.log for analytics summary

// Get trends for the last 6 hours
const trends = errorAnalytics.getTrends(6);
// Use errorLogger instead of console.log for trends
*/