// @ts-nocheck
import { v4 as uuidv4 } from 'uuid';
import { toast } from 'react-toastify';
import errorLogger from './errorLogging';

/**
 * Advanced Utilities for OpenEvolve Enhanced Plugin
 * Provides sophisticated utility functions for performance, security, monitoring, and integration
 */

// Performance Utilities
/**
 * Performance Benchmarking Utility
 * Measures and compares execution times for different approaches
 */
export class PerformanceBenchmark {
  private results: Array<{
    id: string;
    name: string;
    executionTime: number;
    memoryUsage: number;
    timestamp: number;
  }> = [];
  
  private maxResults: number = 1000; // Limit memory usage
  private enableMemoryTracking: boolean = true;

  constructor(options: { maxResults?: number; enableMemoryTracking?: boolean } = {}) {
    try {
      this.maxResults = options.maxResults || 1000;
      this.enableMemoryTracking = options.enableMemoryTracking !== false; // Default to true
    } catch (error) {
      console.error('Error initializing PerformanceBenchmark:', error);
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'PerformanceBenchmark', function: 'constructor', additionalData: { options } }
      );
      // Set defaults in case of error
      this.maxResults = 1000;
      this.enableMemoryTracking = true;
    }
  }

  /**
   * Measure execution time and memory usage of a function
   */
  async measure<T>(name: string, fn: () => Promise<T>): Promise<{ result: T; metrics: { executionTime: number; memoryUsage: number } }> {
    try {
      const startTime = performance.now();
      const startMemory = this.enableMemoryTracking ? this.getMemoryUsage() : { used: 0, total: 0, free: 0, percentage: 0 };

      try {
        const result = await fn();
        const endTime = performance.now();
        const endMemory = this.enableMemoryTracking ? this.getMemoryUsage() : { used: 0, total: 0, free: 0, percentage: 0 };

        const executionTime = endTime - startTime;
        const memoryUsage = this.enableMemoryTracking ? (endMemory.used - startMemory.used) : 0;

        // Add result and enforce limit
        this.results.push({
          id: uuidv4(),
          name,
          executionTime,
          memoryUsage,
          timestamp: Date.now(),
        });

        // Keep only the most recent results to limit memory usage
        if (this.results.length > this.maxResults) {
          this.results = this.results.slice(-this.maxResults);
        }

        return {
          result,
          metrics: { executionTime, memoryUsage },
        };
      } catch (fnError) {
        errorLogger.logError(
          fnError instanceof Error ? fnError : new Error(String(fnError)),
          'error',
          { component: 'PerformanceBenchmark', function: 'measure', additionalData: { name } }
        );
        toast.error(`Benchmark failed for ${name}: ${fnError instanceof Error ? fnError.message : String(fnError)}`);
        throw fnError;
      }
    } catch (outerError) {
      errorLogger.logError(
        outerError instanceof Error ? outerError : new Error(String(outerError)),
        'error',
        { component: 'PerformanceBenchmark', function: 'measure', additionalData: { name } }
      );
      throw outerError;
    }
  }

  /**
   * Compare multiple approaches with optional parallel execution
   */
  async compare<T>(name: string, approaches: Array<{ name: string; fn: () => Promise<T> }>, 
                  options: { parallel?: boolean; maxConcurrent?: number } = {}): Promise<{
    winner: string;
    results: Array<{ name: string; executionTime: number; memoryUsage: number; result: T; error?: string }>;
  }> {
    const { parallel = false, maxConcurrent = 3 } = options;
    const comparisonResults = [];

    if (parallel && approaches.length > 1) {
      // Execute approaches in parallel with concurrency limit
      const batches = [];
      for (let i = 0; i < approaches.length; i += maxConcurrent) {
        batches.push(approaches.slice(i, i + maxConcurrent));
      }

      for (const batch of batches) {
        const batchPromises = batch.map(approach => 
          this.measure(approach.name, approach.fn)
            .then(({ result, metrics }) => ({ ...metrics, name: approach.name, result }))
            .catch(error => ({
              name: approach.name,
              executionTime: -1,
              memoryUsage: -1,
              result: null,
              error: error instanceof Error ? error.message : String(error),
            }))
        );

        const batchResults = await Promise.all(batchPromises);
        comparisonResults.push(...batchResults);
      }
    } else {
      // Sequential execution
      for (const approach of approaches) {
        try {
          const { result, metrics } = await this.measure(approach.name, approach.fn);
          comparisonResults.push({ ...metrics, name: approach.name, result });
        } catch (error) {
          comparisonResults.push({
            name: approach.name,
            executionTime: -1,
            memoryUsage: -1,
            result: null,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      }
    }

    // Find the fastest approach with lowest memory usage
    const validResults = comparisonResults.filter(r => r.executionTime >= 0);
    const winner = validResults.length > 0
      ? validResults.reduce((best, current) => 
          current.executionTime < best.executionTime || 
          (current.executionTime === best.executionTime && current.memoryUsage < best.memoryUsage)
            ? current
            : best
        ).name
      : 'None';

    return { winner, results: comparisonResults };
  }


  /**
   * Get all benchmark results
   */
  getResults(): typeof this.results {
    return [...this.results];
  }

  /**
   * Clear benchmark results
   */
  clearResults(): void {
    this.results = [];
  }

  /**
   * Calculate performance improvement percentage
   */
  calculateImprovement(baselineTime: number, optimizedTime: number): number {
    if (baselineTime <= 0 || optimizedTime < 0) return 0;
    return ((baselineTime - optimizedTime) / baselineTime) * 100;
  }

  /**
   * Get performance statistics and insights
   */
  getPerformanceStatistics(): {
    totalBenchmarks: number;
    averageExecutionTime: number;
    averageMemoryUsage: number;
    fastestBenchmark?: { name: string; executionTime: number };
    slowestBenchmark?: { name: string; executionTime: number };
    memoryIntensiveBenchmark?: { name: string; memoryUsage: number };
  } {
    if (this.results.length === 0) {
      return {
        totalBenchmarks: 0,
        averageExecutionTime: 0,
        averageMemoryUsage: 0,
      };
    }

    const validResults = this.results.filter(r => r.executionTime >= 0);
    
    const totalExecutionTime = validResults.reduce((sum, r) => sum + r.executionTime, 0);
    const totalMemoryUsage = validResults.reduce((sum, r) => sum + r.memoryUsage, 0);

    // Find fastest and slowest benchmarks
    const fastest = validResults.reduce((fastest, current) => 
      current.executionTime < fastest.executionTime ? current : fastest
    );
    const slowest = validResults.reduce((slowest, current) => 
      current.executionTime > slowest.executionTime ? current : slowest
    );
    const memoryIntensive = validResults.reduce((intensive, current) => 
      current.memoryUsage > intensive.memoryUsage ? current : intensive
    );

    return {
      totalBenchmarks: this.results.length,
      averageExecutionTime: totalExecutionTime / validResults.length,
      averageMemoryUsage: totalMemoryUsage / validResults.length,
      fastestBenchmark: fastest ? { name: fastest.name, executionTime: fastest.executionTime } : undefined,
      slowestBenchmark: slowest ? { name: slowest.name, executionTime: slowest.executionTime } : undefined,
      memoryIntensiveBenchmark: memoryIntensive ? { name: memoryIntensive.name, memoryUsage: memoryIntensive.memoryUsage } : undefined,
    };
  }

  /**
   * Get memory usage with fallback for different environments
   */
  getMemoryUsage(): { used: number; total: number; free: number; percentage: number } {
    // Optimized memory usage tracking with environment detection
    if (typeof performance !== 'undefined' && performance.memory) {
      // Browser environment with performance.memory
      const memory = performance.memory;
      const used = memory.usedJSHeapSize / 1024 / 1024; // MB
      const total = memory.jsHeapSizeLimit / 1024 / 1024; // MB
      const free = total - used;
      const percentage = (used / total) * 100;
      return { used, total, free, percentage };
    } else if (typeof process !== 'undefined' && process.memoryUsage) {
      // Node.js environment
      const memoryUsage = process.memoryUsage();
      const used = memoryUsage.heapUsed / 1024 / 1024; // MB
      const total = memoryUsage.heapTotal / 1024 / 1024; // MB
      const free = total - used;
      const percentage = (used / total) * 100;
      return { used, total, free, percentage };
    } else {
      // Fallback for environments without memory APIs
      return { used: 0, total: 1024, free: 1024, percentage: 0 };
    }
  }

  /**
   * Clear all benchmark results to free memory
   */
  clearAllResults(): void {
    this.results = [];
  }

  /**
   * Get results within a specific time window
   */
  getRecentResults(timeWindow: number = 3600000): typeof this.results {
    const now = Date.now();
    return this.results.filter(result => now - result.timestamp <= timeWindow);
  }

  /**
   * Export results to JSON for analysis
   */
  exportResultsToJson(pretty: boolean = true): string {
    return pretty 
      ? JSON.stringify(this.results, null, 2)
      : JSON.stringify(this.results);
  }

  /**
   * Import results from JSON
   */
  importResultsFromJson(json: string): void {
    try {
      const imported = JSON.parse(json);
      if (Array.isArray(imported)) {
        // Validate and filter imported results
        const validResults = imported.filter(item => 
          item.id && item.name && typeof item.executionTime === 'number'
        );
        
        // Merge with existing results, respecting maxResults limit
        this.results = [...this.results, ...validResults].slice(-this.maxResults);
      }
    } catch (error) {
      toast.error(`Failed to import benchmark results: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
}

// Security Utilities
/**
 * Security Utility Functions
 */
export class SecurityUtils {
  /**
   * Generate secure random token
   */
  static generateSecureToken(length: number = 32): string {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let token = '';

    if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
      const values = new Uint32Array(length);
      crypto.getRandomValues(values);
      
      for (let i = 0; i < length; i++) {
        token += chars[values[i] % chars.length];
      }
    } else {
      // Fallback for environments without crypto
      for (let i = 0; i < length; i++) {
        token += chars[Math.floor(Math.random() * chars.length)];
      }
    }

    return token;
  }

  /**
   * Hash data using SHA-256 (browser implementation)
   */
  static async hashData(data: string): Promise<string> {
    if (typeof crypto !== 'undefined' && crypto.subtle) {
      const encoder = new TextEncoder();
      const dataBuffer = encoder.encode(data);
      const hashBuffer = await crypto.subtle.digest('SHA-256', dataBuffer);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    }

    // Fallback for environments without crypto.subtle
    return this.simpleHash(data);
  }

  /**
   * Simple hash function (fallback)
   */
  private static simpleHash(data: string): string {
    let hash = 0;
    for (let i = 0; i < data.length; i++) {
      const char = data.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return hash.toString(16);
  }

  /**
   * Validate password strength
   */
  static validatePasswordStrength(password: string): {
    score: number;
    feedback: string[];
    isStrong: boolean;
  } {
    let score = 0;
    const feedback: string[] = [];

    // Length check
    if (password.length >= 12) {
      score += 2;
    } else if (password.length >= 8) {
      score += 1;
      feedback.push('Password should be at least 12 characters');
    } else {
      feedback.push('Password is too short (minimum 8 characters)');
    }

    // Character variety
    if (/[A-Z]/.test(password)) score += 1;
    else feedback.push('Add uppercase letters');

    if (/[a-z]/.test(password)) score += 1;
    else feedback.push('Add lowercase letters');

    if (/[0-9]/.test(password)) score += 1;
    else feedback.push('Add numbers');

    if (/[^A-Za-z0-9]/.test(password)) score += 2;
    else feedback.push('Add special characters');

    // Common patterns
    const commonPatterns = ['password', '123456', 'qwerty', 'admin', 'welcome'];
    if (commonPatterns.some(pattern => password.toLowerCase().includes(pattern))) {
      score -= 2;
      feedback.push('Avoid common password patterns');
    }

    // Repeated characters
    if (/(.)\1{2,}/.test(password)) {
      score -= 1;
      feedback.push('Avoid repeated characters');
    }

    const isStrong = score >= 6;

    return { score, feedback, isStrong };
  }

  /**
   * Generate JWT-like token (simplified)
   */
  static generateJwtLikeToken(payload: Record<string, any>, secret: string, expiresIn: number = 3600): string {
    const header = JSON.stringify({ alg: 'HS256', typ: 'JWT' });
    const encodedPayload = JSON.stringify({
      ...payload,
      iat: Math.floor(Date.now() / 1000),
      exp: Math.floor(Date.now() / 1000) + expiresIn,
    });

    const base64UrlEncode = (str: string): string => {
      return btoa(str)
        .replace(/\+/g, '-')
        .replace(/\//g, '_')
        .replace(/=+$/, '');
    };

    const signature = this.simpleHash(header + '.' + encodedPayload + secret);

    return [header, encodedPayload, signature]
      .map(base64UrlEncode)
      .join('.');
  }

  /**
   * Validate JWT-like token (simplified)
   */
  static validateJwtLikeToken(token: string, secret: string): {
    valid: boolean;
    payload?: Record<string, any>;
    error?: string;
  } {
    try {
      const parts = token.split('.');
      if (parts.length !== 3) {
        return { valid: false, error: 'Invalid token format' };
      }

      const [header, payload, signature] = parts;
      const expectedSignature = this.simpleHash(header + '.' + payload + secret);

      if (signature !== expectedSignature) {
        return { valid: false, error: 'Invalid signature' };
      }

      try {
        const decodedPayload = JSON.parse(atob(payload.replace(/-/g, '+').replace(/_/g, '/')));
        
        // Check expiration
        if (decodedPayload.exp && decodedPayload.exp < Math.floor(Date.now() / 1000)) {
          return { valid: false, error: 'Token expired' };
        }

        return { valid: true, payload: decodedPayload };
      } catch (e) {
        return { valid: false, error: 'Invalid payload' };
      }
    } catch (error) {
      return { valid: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }
}

// Monitoring Utilities
/**
 * Monitoring and Alerting Utilities
 */
export class MonitoringUtils {
  private alerts: Array<{
    id: string;
    metric: string;
    value: number;
    threshold: number;
    severity: 'low' | 'medium' | 'high' | 'critical';
    timestamp: number;
    resolved: boolean;
  }> = [];

  /**
   * Check metric against threshold and create alert if needed
   */
  checkMetricAndAlert(
    metric: string,
    value: number,
    threshold: number,
    severity: 'low' | 'medium' | 'high' | 'critical' = 'medium'
  ): boolean {
    if (value > threshold) {
      const alertId = uuidv4();
      const alert = {
        id: alertId,
        metric,
        value,
        threshold,
        severity,
        timestamp: Date.now(),
        resolved: false,
      };

      this.alerts.push(alert);
      
      // Trigger appropriate notification based on severity
      switch (severity) {
        case 'critical':
          toast.error(`🚨 CRITICAL ALERT: ${metric} = ${value} (threshold: ${threshold})`);
          break;
        case 'high':
          toast.error(`⚠️ HIGH ALERT: ${metric} = ${value} (threshold: ${threshold})`);
          break;
        case 'medium':
          toast.warn(`⚠️ MEDIUM ALERT: ${metric} = ${value} (threshold: ${threshold})`);
          break;
        case 'low':
          toast.info(`ℹ️ LOW ALERT: ${metric} = ${value} (threshold: ${threshold})`);
          break;
      }

      return true;
    }
    return false;
  }

  /**
   * Resolve an alert
   */
  resolveAlert(alertId: string): boolean {
    const alertIndex = this.alerts.findIndex(alert => alert.id === alertId);
    
    if (alertIndex !== -1) {
      this.alerts[alertIndex].resolved = true;
      toast.success(`✅ Alert resolved: ${this.alerts[alertIndex].metric}`);
      return true;
    }
    
    return false;
  }

  /**
   * Get active alerts
   */
  getActiveAlerts(): typeof this.alerts {
    return this.alerts.filter(alert => !alert.resolved);
  }

  /**
   * Get alert history
   */
  getAlertHistory(): typeof this.alerts {
    return [...this.alerts];
  }

  /**
   * Clear resolved alerts
   */
  clearResolvedAlerts(): void {
    this.alerts = this.alerts.filter(alert => !alert.resolved);
  }

  /**
   * Calculate alert frequency
   */
  getAlertFrequency(metric: string, timeWindow: number = 3600000): number {
    const now = Date.now();
    return this.alerts.filter(alert => 
      alert.metric === metric && 
      alert.timestamp >= now - timeWindow
    ).length;
  }

  /**
   * Check if system is in alert storm
   */
  isAlertStorm(timeWindow: number = 300000, threshold: number = 10): boolean {
    const now = Date.now();
    const recentAlerts = this.alerts.filter(alert => 
      alert.timestamp >= now - timeWindow
    );
    
    return recentAlerts.length >= threshold;
  }
}

// Integration Utilities
/**
 * Integration and API Utilities
 */
export class IntegrationUtils {
  private apiCache: Map<string, { data: any; timestamp: number; ttl: number }> = new Map();
  private cacheStats: { hits: number; misses: number; lastCleared: number } = { hits: 0, misses: 0, lastCleared: Date.now() };
  private maxCacheSize: number = 100; // Limit cache size to prevent memory issues

  constructor(options: { maxCacheSize?: number } = {}) {
    this.maxCacheSize = options.maxCacheSize || 100;
  }

  /**
   * Smart API caller with caching and retry logic
   */
  async smartApiCall(
    url: string,
    options: {
      method?: string;
      headers?: Record<string, string>;
      body?: any;
      cacheTtl?: number;
      maxRetries?: number;
      retryDelay?: number;
      timeout?: number;
      useExponentialBackoff?: boolean;
    } = {}
  ): Promise<{ 
    success: boolean; 
    data?: any; 
    error?: string; 
    fromCache?: boolean; 
    retryCount: number; 
  }> {
    const {
      method = 'GET',
      headers = { 'Content-Type': 'application/json' },
      body,
      cacheTtl = 300000, // 5 minutes default
      maxRetries = 3,
      retryDelay = 1000,
      timeout = 10000,
      useExponentialBackoff = true,
    } = options;

    const cacheKey = `${method}:${url}:${body ? JSON.stringify(body) : ''}`;

    // Check cache first for GET requests
    if (method === 'GET' && cacheTtl > 0) {
      const cached = this.apiCache.get(cacheKey);
      if (cached && Date.now() - cached.timestamp < cached.ttl) {
        this.cacheStats.hits++;
        return { 
          success: true, 
          data: cached.data, 
          fromCache: true, 
          retryCount: 0 
        };
      }
      this.cacheStats.misses++;
    }

    let retryCount = 0;
    let lastError: string | null = null;
    let currentRetryDelay = retryDelay;

    const startTime = Date.now();

    while (retryCount <= maxRetries) {
      // Check timeout
      if (Date.now() - startTime > timeout) {
        lastError = `Request timed out after ${timeout}ms`;
        break;
      }

      try {
        // Simulate API call (in real implementation, use fetch or axios)
        const response = await this.simulateApiCall(url, { method, headers, body });

        // Cache successful GET responses
        if (method === 'GET' && cacheTtl > 0) {
          // Enforce cache size limit
          if (this.apiCache.size >= this.maxCacheSize) {
            this.cleanupCache();
          }
          
          this.apiCache.set(cacheKey, { 
            data: response.data, 
            timestamp: Date.now(), 
            ttl: cacheTtl 
          });
        }

        return { 
          success: true, 
          data: response.data, 
          fromCache: false, 
          retryCount 
        };
      } catch (error) {
        lastError = error instanceof Error ? error.message : String(error);
        retryCount++;

        if (retryCount <= maxRetries) {
          // Exponential backoff for retries
          if (useExponentialBackoff) {
            currentRetryDelay = retryDelay * Math.pow(2, retryCount - 1);
            // Cap the maximum delay
            currentRetryDelay = Math.min(currentRetryDelay, 10000);
          }

          toast.warn(`API call failed (attempt ${retryCount}/${maxRetries}), retrying in ${currentRetryDelay}ms...`);
          await new Promise(resolve => setTimeout(resolve, currentRetryDelay));
        }
      }
    }

    return { 
      success: false, 
      error: lastError || 'Unknown error', 
      retryCount: maxRetries 
    };
  }

  /**
   * Simulate API call (replace with actual fetch/axios in production)
   */
  private async simulateApiCall(
    url: string,
    options: { method: string; headers: Record<string, string>; body?: any }
  ): Promise<{ data: any }> {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 500));

    // Simulate different responses based on URL
    if (url.includes('error')) {
      throw new Error('Simulated API error');
    }

    if (url.includes('slow')) {
      await new Promise(resolve => setTimeout(resolve, 2000));
    }

    return { 
      data: {
        url,
        method: options.method,
        timestamp: Date.now(),
        success: true,
      } 
    };
  }

  /**
   * Clear API cache
   */
  clearApiCache(): void {
    this.apiCache.clear();
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { size: number; hitRate: number } {
    return {
      size: this.apiCache.size,
      hitRate: 0 // Would need to track hits/misses in real implementation
    };
  }

  /**
   * Batch API calls with rate limiting
   */
  async batchApiCalls(
    calls: Array<{ url: string; options?: any }>,
    batchSize: number = 5,
    delayBetweenBatches: number = 1000
  ): Promise<Array<{ 
    url: string; 
    success: boolean; 
    data?: any; 
    error?: string; 
  }>> {
    const results: Array<{ 
      url: string; 
      success: boolean; 
      data?: any; 
      error?: string; 
    }> = [];

    for (let i = 0; i < calls.length; i += batchSize) {
      const batch = calls.slice(i, i + batchSize);
      const batchPromises = batch.map(call => 
        this.smartApiCall(call.url, call.options)
          .then(result => ({
            url: call.url,
            success: result.success,
            data: result.data,
            error: result.error
          }))
          .catch(error => ({
            url: call.url,
            success: false,
            error: error instanceof Error ? error.message : String(error)
          }))
      );

      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults);

      if (i + batchSize < calls.length) {
        await new Promise(resolve => setTimeout(resolve, delayBetweenBatches));
      }
    }

    return results;
  }
}

// Error Analysis Utilities
/**
 * Advanced Error Analysis Utilities
 */
export class ErrorAnalysisUtils {
  private errorHistory: Array<{
    id: string;
    errorType: string;
    errorMessage: string;
    context: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    timestamp: number;
    stackTrace?: string;
    metadata?: Record<string, any>;
  }> = [];

  /**
   * Analyze error and add to history
   */
  analyzeError(
    error: unknown,
    context: string = 'unknown',
    severity: 'low' | 'medium' | 'high' | 'critical' = 'medium',
    metadata: Record<string, any> = {}
  ): string {
    const errorId = uuidv4();
    const timestamp = Date.now();

    let errorType = 'UnknownError';
    let errorMessage = 'Unknown error';
    let stackTrace: string | undefined;

    if (error instanceof Error) {
      errorType = error.name;
      errorMessage = error.message;
      stackTrace = error.stack;
    } else if (typeof error === 'string') {
      errorMessage = error;
    } else if (error && typeof error === 'object') {
      errorType = 'ObjectError';
      errorMessage = JSON.stringify(error);
    }

    const errorRecord = {
      id: errorId,
      errorType,
      errorMessage,
      context,
      severity,
      timestamp,
      stackTrace,
      metadata,
    };

    this.errorHistory.push(errorRecord);

    // Log error based on severity using error logger
    const severityMap: Record<string, 'critical' | 'error' | 'warn' | 'info'> = {
      'critical': 'critical',
      'high': 'error',
      'medium': 'warn',
      'low': 'info'
    };

    errorLogger.logError(errorMessage, severityMap[severity] || 'error', {
      component: 'ErrorAnalysisUtils',
      function: 'trackError',
      additionalData: { errorRecord, severity }
    });

    // Show toast notification
    switch (severity) {
      case 'critical':
        toast.error(`🚨 CRITICAL: ${errorMessage}`);
        break;
      case 'high':
        toast.error(`❌ HIGH: ${errorMessage}`);
        break;
      case 'medium':
        toast.warn(`⚠️ MEDIUM: ${errorMessage}`);
        break;
      case 'low':
        toast.info(`ℹ️ LOW: ${errorMessage}`);
        break;
    }

    return errorId;
  }

  /**
   * Get error patterns and trends
   */
  getErrorPatterns(timeWindow: number = 86400000): {
    byType: Record<string, number>;
    byContext: Record<string, number>;
    bySeverity: Record<string, number>;
    total: number;
  } {
    const now = Date.now();
    const recentErrors = this.errorHistory.filter(error => 
      error.timestamp >= now - timeWindow
    );

    const byType: Record<string, number> = {};
    const byContext: Record<string, number> = {};
    const bySeverity: Record<string, number> = {};

    recentErrors.forEach(error => {
      byType[error.errorType] = (byType[error.errorType] || 0) + 1;
      byContext[error.context] = (byContext[error.context] || 0) + 1;
      bySeverity[error.severity] = (bySeverity[error.severity] || 0) + 1;
    });

    return {
      byType,
      byContext,
      bySeverity,
      total: recentErrors.length,
    };
  }

  /**
   * Get most common errors
   */
  getMostCommonErrors(limit: number = 5): Array<{ 
    errorType: string; 
    count: number; 
    percentage: number; 
  }> {
    const patterns = this.getErrorPatterns();
    const sorted = Object.entries(patterns.byType)
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit);

    return sorted.map(([errorType, count]) => ({
      errorType,
      count,
      percentage: (count / patterns.total) * 100,
    }));
  }

  /**
   * Detect error trends (increasing frequency)
   */
  detectErrorTrends(
    errorType: string,
    timeWindows: number[] = [3600000, 86400000, 604800000]
  ): { 
    isTrending: boolean; 
    trendData: Array<{ windowMs: number; count: number; rate: number }>; 
  } {
    const now = Date.now();
    const trendData = timeWindows.map(windowMs => {
      const count = this.errorHistory.filter(error => 
        error.errorType === errorType && 
        error.timestamp >= now - windowMs
      ).length;

      const rate = count / (windowMs / 3600000); // errors per hour
      return { windowMs, count, rate };
    });

    // Check if error rate is increasing
    const isTrending = trendData.length >= 2 && 
      trendData[0].rate > trendData[1].rate * 1.5;

    return { isTrending, trendData };
  }

  /**
   * Get error history
   */
  getErrorHistory(limit: number = 100): typeof this.errorHistory {
    return [...this.errorHistory].slice(-limit);
  }

  /**
   * Clear error history
   */
  clearErrorHistory(): void {
    this.errorHistory = [];
  }

  /**
   * Get error severity distribution
   */
  getSeverityDistribution(): Record<'low' | 'medium' | 'high' | 'critical', number> {
    const distribution: Record<'low' | 'medium' | 'high' | 'critical', number> = {
      low: 0,
      medium: 0,
      high: 0,
      critical: 0,
    };

    this.errorHistory.forEach(error => {
      distribution[error.severity]++;
    });

    return distribution;
  }
}

// Configuration Utilities
/**
 * Configuration Management Utilities
 */
export class ConfigUtils {
  /**
   * Deep merge configuration objects
   */
  static deepMerge<T>(target: T, source: Partial<T>): T {
    if (typeof target !== 'object' || target === null) return source as T;
    if (typeof source !== 'object' || source === null) return target;

    const output = { ...target };

    for (const key in source) {
      if (source.hasOwnProperty(key)) {
        if (typeof source[key] === 'object' && source[key] !== null && 
            typeof target[key] === 'object' && target[key] !== null) {
          output[key] = this.deepMerge(target[key], source[key]);
        } else {
          output[key] = source[key];
        }
      }
    }

    return output;
  }

  /**
   * Validate configuration against schema
   */
  static validateConfig(config: any, schema: any): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Simple validation - in real implementation use a proper validation library
    for (const key in schema) {
      if (schema.hasOwnProperty(key)) {
        const schemaDef = schema[key];

        if (schemaDef.required && (config[key] === undefined || config[key] === null)) {
          errors.push(`Missing required field: ${key}`);
        }

        if (config[key] !== undefined && schemaDef.type) {
          const actualType = typeof config[key];
          if (actualType !== schemaDef.type) {
            errors.push(`Invalid type for ${key}: expected ${schemaDef.type}, got ${actualType}`);
          }
        }

        if (config[key] !== undefined && schemaDef.min !== undefined && config[key] < schemaDef.min) {
          errors.push(`${key} must be at least ${schemaDef.min}, got ${config[key]}`);
        }

        if (config[key] !== undefined && schemaDef.max !== undefined && config[key] > schemaDef.max) {
          errors.push(`${key} must be at most ${schemaDef.max}, got ${config[key]}`);
        }
      }
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }

  /**
   * Generate configuration diff
   */
  static generateConfigDiff(oldConfig: any, newConfig: any): Array<{ 
    path: string; 
    oldValue: any; 
    newValue: any; 
    changeType: 'added' | 'removed' | 'changed';
  }> {
    const diff: Array<{ 
      path: string; 
      oldValue: any; 
      newValue: any; 
      changeType: 'added' | 'removed' | 'changed';
    }> = [];

    const findDiff = (path: string, oldObj: any, newObj: any) => {
      for (const key in newObj) {
        if (!oldObj.hasOwnProperty(key)) {
          diff.push({
            path: path ? `${path}.${key}` : key,
            oldValue: undefined,
            newValue: newObj[key],
            changeType: 'added',
          });
        }
      }

      for (const key in oldObj) {
        if (!newObj.hasOwnProperty(key)) {
          diff.push({
            path: path ? `${path}.${key}` : key,
            oldValue: oldObj[key],
            newValue: undefined,
            changeType: 'removed',
          });
        }
      }

      for (const key in oldObj) {
        if (newObj.hasOwnProperty(key)) {
          const oldValue = oldObj[key];
          const newValue = newObj[key];

          if (typeof oldValue === 'object' && oldValue !== null &&
              typeof newValue === 'object' && newValue !== null) {
            findDiff(path ? `${path}.${key}` : key, oldValue, newValue);
          } else if (oldValue !== newValue) {
            diff.push({
              path: path ? `${path}.${key}` : key,
              oldValue,
              newValue,
              changeType: 'changed',
            });
          }
        }
      }
    };

    findDiff('', oldConfig, newConfig);
    return diff;
  }

  /**
   * Sanitize configuration (remove sensitive data)
   */
  static sanitizeConfig(config: any, sensitiveKeys: string[] = ['password', 'secret', 'token', 'apiKey']): any {
    if (typeof config !== 'object' || config === null) return config;

    const sanitized = { ...config };

    for (const key in sanitized) {
      if (sensitiveKeys.some(sensitiveKey => 
          key.toLowerCase().includes(sensitiveKey.toLowerCase()))) {
        sanitized[key] = '***REDACTED***';
      } else if (typeof sanitized[key] === 'object') {
        sanitized[key] = this.sanitizeConfig(sanitized[key], sensitiveKeys);
      }
    }

    return sanitized;
  }

  /**
   * Export configuration to JSON
   */
  static exportConfigToJson(config: any, pretty: boolean = true): string {
    return pretty 
      ? JSON.stringify(config, null, 2)
      : JSON.stringify(config);
  }

  /**
   * Import configuration from JSON
   */
  static importConfigFromJson(json: string): any {
    try {
      return JSON.parse(json);
    } catch (error) {
      toast.error(`Failed to parse JSON: ${error instanceof Error ? error.message : String(error)}`);
      return null;
    }
  }
}