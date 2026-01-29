/**
 * Performance Monitoring Service
 * Monitors application performance and integrates with error handling
 */

import { errorLogger } from './errorLogging';
import { ErrorCategory } from './EnhancedErrorReporting';

// Define performance metric types
export interface PerformanceMetrics {
  navigationStart: number;
  domContentLoaded: number;
  loadComplete: number;
  memoryUsage?: number;
  cpuUsage?: number;
  jsHeapSizeLimit?: number;
  totalJSHeapSize?: number;
  usedJSHeapSize?: number;
  firstPaint?: number;
  firstContentfulPaint?: number;
  largestContentfulPaint?: number;
  cumulativeLayoutShift?: number;
  interactionToNextPaint?: number;
  resourceLoadTimes?: ResourceLoadMetrics[];
  customMetrics?: Record<string, number>;
}

// Define resource load metrics
export interface ResourceLoadMetrics {
  name: string;
  entryType: string;
  startTime: number;
  duration: number;
  transferSize?: number;
  decodedBodySize?: number;
  nextHopProtocol?: string;
}

// Define performance threshold configuration
export interface PerformanceThresholds {
  slowOperationThreshold: number; // ms
  highMemoryUsageThreshold: number; // percentage
  slowResourceLoadThreshold: number; // ms
  highCLSthreshold: number; // Cumulative Layout Shift
  slowLCPthreshold: number; // Largest Contentful Paint in ms
  slowFCPthreshold: number; // First Contentful Paint in ms
}

// Default thresholds
export const DEFAULT_PERFORMANCE_THRESHOLDS: PerformanceThresholds = {
  slowOperationThreshold: 3000, // 3 seconds
  highMemoryUsageThreshold: 80, // 80%
  slowResourceLoadThreshold: 5000, // 5 seconds
  highCLSthreshold: 0.1, // 0.1 is considered poor
  slowLCPthreshold: 2500, // 2.5 seconds
  slowFCPthreshold: 1800, // 1.8 seconds
};

// Performance monitoring options
export interface PerformanceMonitoringOptions {
  enabled: boolean;
  captureLongTasks: boolean;
  captureLayoutShifts: boolean;
  captureResourceLoads: boolean;
  captureNavigation: boolean;
  capturePaint: boolean;
  reportPerformanceIssues: boolean;
  thresholds: PerformanceThresholds;
}

/**
 * Performance Monitoring Service
 * Monitors application performance and integrates with error handling
 */
export class PerformanceMonitoringService {
  private static instance: PerformanceMonitoringService;
  private observer: PerformanceObserver | null = null;
  private longTaskObserver: PerformanceObserver | null = null;
  private layoutShiftObserver: PerformanceObserver | null = null;
  private resourceObserver: PerformanceObserver | null = null;
  private navigationObserver: PerformanceObserver | null = null;
  private paintObserver: PerformanceObserver | null = null;
  private metrics: PerformanceMetrics = {} as PerformanceMetrics;
  private options: PerformanceMonitoringOptions;
  private performanceIssueCallbacks: Array<(metric: string, value: number, threshold: number) => void> = [];

  private constructor(options?: Partial<PerformanceMonitoringOptions>) {
    this.options = {
      enabled: true,
      captureLongTasks: true,
      captureLayoutShifts: true,
      captureResourceLoads: true,
      captureNavigation: true,
      capturePaint: true,
      reportPerformanceIssues: true,
      thresholds: DEFAULT_PERFORMANCE_THRESHOLDS,
      ...options
    };

    if (this.options.enabled) {
      this.setupPerformanceMonitoring();
    }
  }

  /**
   * Get singleton instance
   */
  static getInstance(options?: Partial<PerformanceMonitoringOptions>): PerformanceMonitoringService {
    if (!PerformanceMonitoringService.instance) {
      PerformanceMonitoringService.instance = new PerformanceMonitoringService(options);
    }
    return PerformanceMonitoringService.instance;
  }

  /**
   * Setup performance monitoring observers
   */
  private setupPerformanceMonitoring(): void {
    if (typeof PerformanceObserver === 'undefined') {
      console.warn('Performance Observer API not supported in this environment');
      return;
    }

    // Observe paint metrics
    if (this.options.capturePaint) {
      this.paintObserver = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          if (entry.name === 'first-paint') {
            this.metrics.firstPaint = entry.startTime;
            this.checkPerformanceThreshold('firstPaint', entry.startTime, this.options.thresholds.slowFCPthreshold);
          } else if (entry.name === 'first-contentful-paint') {
            this.metrics.firstContentfulPaint = entry.startTime;
            this.checkPerformanceThreshold('firstContentfulPaint', entry.startTime, this.options.thresholds.slowFCPthreshold);
          }
        });
      });

      try {
        this.paintObserver.observe({ entryTypes: ['paint'] });
      } catch (e) {
        console.warn('Could not observe paint metrics:', e);
      }
    }

    // Observe largest contentful paint
    if (this.options.capturePaint) {
      const lcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        if (entries.length > 0) {
          const lastEntry = entries[entries.length - 1] as PerformanceEntry;
          this.metrics.largestContentfulPaint = lastEntry.startTime;
          this.checkPerformanceThreshold('largestContentfulPaint', lastEntry.startTime, this.options.thresholds.slowLCPthreshold);
        }
      });

      try {
        lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });
      } catch (e) {
        console.warn('Could not observe LCP metrics:', e);
      }
    }

    // Observe cumulative layout shift
    if (this.options.captureLayoutShifts) {
      this.layoutShiftObserver = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          if ((entry as any).name === 'cumulative-layout-shift') {
            this.metrics.cumulativeLayoutShift = (entry as any).value;
            this.checkPerformanceThreshold('cumulativeLayoutShift', (entry as any).value, this.options.thresholds.highCLSthreshold);
          }
        });
      });

      try {
        this.layoutShiftObserver.observe({ entryTypes: ['layout-shift'] });
      } catch (e) {
        console.warn('Could not observe layout shift metrics:', e);
      }
    }

    // Observe long tasks
    if (this.options.captureLongTasks) {
      this.longTaskObserver = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          // Log long tasks as potential performance issues
          if (entry.duration > 50) { // Tasks longer than 50ms are considered long
            this.reportPerformanceIssue('longTask', entry.duration, 50);
          }
        });
      });

      try {
        this.longTaskObserver.observe({ entryTypes: ['longtask'] });
      } catch (e) {
        console.warn('Could not observe long tasks:', e);
      }
    }

    // Observe resource loads
    if (this.options.captureResourceLoads) {
      this.resourceObserver = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          const resourceMetric: ResourceLoadMetrics = {
            name: entry.name,
            entryType: entry.entryType,
            startTime: entry.startTime,
            duration: entry.duration,
          };

          // Add additional properties if available
          if ('transferSize' in entry) {
            (resourceMetric as any).transferSize = (entry as any).transferSize;
          }
          if ('decodedBodySize' in entry) {
            (resourceMetric as any).decodedBodySize = (entry as any).decodedBodySize;
          }
          if ('nextHopProtocol' in entry) {
            (resourceMetric as any).nextHopProtocol = (entry as any).nextHopProtocol;
          }

          if (!this.metrics.resourceLoadTimes) {
            this.metrics.resourceLoadTimes = [];
          }
          this.metrics.resourceLoadTimes.push(resourceMetric);

          // Check if resource load is too slow
          if (entry.duration > this.options.thresholds.slowResourceLoadThreshold) {
            this.reportPerformanceIssue('slowResourceLoad', entry.duration, this.options.thresholds.slowResourceLoadThreshold);
          }
        });
      });

      try {
        this.resourceObserver.observe({ entryTypes: ['resource', 'navigation'] });
      } catch (e) {
        console.warn('Could not observe resource load metrics:', e);
      }
    }

    // Observe navigation
    if (this.options.captureNavigation) {
      this.navigationObserver = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          if (entry.entryType === 'navigation') {
            const navEntry = entry as PerformanceNavigationTiming;
            this.metrics.navigationStart = navEntry.navigationStart;
            this.metrics.domContentLoaded = navEntry.domContentLoadedEventEnd - navEntry.navigationStart;
            this.metrics.loadComplete = navEntry.loadEventEnd - navEntry.navigationStart;
          }
        });
      });

      try {
        this.navigationObserver.observe({ entryTypes: ['navigation'] });
      } catch (e) {
        console.warn('Could not observe navigation metrics:', e);
      }
    }

    // Capture memory info if available
    if ((performance as any).memory) {
      const mem = (performance as any).memory;
      this.metrics.memoryUsage = mem.usedJSHeapSize / mem.jsHeapSizeLimit * 100;
      this.metrics.jsHeapSizeLimit = mem.jsHeapSizeLimit;
      this.metrics.totalJSHeapSize = mem.totalJSHeapSize;
      this.metrics.usedJSHeapSize = mem.usedJSHeapSize;

      // Check if memory usage is too high
      if (mem.usedJSHeapSize / mem.jsHeapSizeLimit * 100 > this.options.thresholds.highMemoryUsageThreshold) {
        this.reportPerformanceIssue('highMemoryUsage', mem.usedJSHeapSize / mem.jsHeapSizeLimit * 100, this.options.thresholds.highMemoryUsageThreshold);
      }
    }
  }

  /**
   * Measure execution time of a function
   */
  async measureFunction<T>(fn: () => Promise<T> | T, name: string): Promise<{ result: T; duration: number }> {
    const start = performance.now();
    let result;

    try {
      result = await Promise.resolve(fn());
    } finally {
      const end = performance.now();
      const duration = end - start;

      // Check if function execution was too slow
      if (duration > this.options.thresholds.slowOperationThreshold) {
        this.reportPerformanceIssue('slowOperation', duration, this.options.thresholds.slowOperationThreshold);
      }

      // Add to custom metrics
      if (!this.metrics.customMetrics) {
        this.metrics.customMetrics = {};
      }
      this.metrics.customMetrics[name] = duration;

      return { result, duration };
    }
  }

  /**
   * Measure execution time of an async operation
   */
  async measureAsyncOperation<T>(
    operation: () => Promise<T>,
    name: string,
    options?: { 
      reportSlowOperations?: boolean; 
      threshold?: number;
    }
  ): Promise<{ result: T; duration: number }> {
    const start = performance.now();
    let result;

    try {
      result = await operation();
    } finally {
      const end = performance.now();
      const duration = end - start;

      const threshold = options?.threshold ?? this.options.thresholds.slowOperationThreshold;
      
      // Check if operation was too slow
      if (options?.reportSlowOperations !== false && duration > threshold) {
        this.reportPerformanceIssue(name, duration, threshold);
      }

      // Add to custom metrics
      if (!this.metrics.customMetrics) {
        this.metrics.customMetrics = {};
      }
      this.metrics.customMetrics[name] = duration;

      return { result, duration };
    }
  }

  /**
   * Check if a metric exceeds its threshold
   */
  private checkPerformanceThreshold(metric: string, value: number, threshold: number): void {
    if (value > threshold) {
      this.reportPerformanceIssue(metric, value, threshold);
    }
  }

  /**
   * Report a performance issue
   */
  private reportPerformanceIssue(metric: string, value: number, threshold: number): void {
    if (!this.options.reportPerformanceIssues) {
      return;
    }

    const message = `Performance issue detected: ${metric} (${value.toFixed(2)}ms) exceeded threshold (${threshold}ms)`;
    
    // Log the performance issue
    errorLogger.logError(new Error(message), 'warn', {
      component: 'PerformanceMonitoring',
      function: 'reportPerformanceIssue',
      additionalData: {
        metric,
        value,
        threshold,
        timestamp: Date.now()
      }
    });

    // Call any registered callbacks
    this.performanceIssueCallbacks.forEach(callback => {
      try {
        callback(metric, value, threshold);
      } catch (e) {
        console.error('Error in performance issue callback:', e);
      }
    });
  }

  /**
   * Add a callback for performance issues
   */
  addPerformanceIssueCallback(callback: (metric: string, value: number, threshold: number) => void): void {
    this.performanceIssueCallbacks.push(callback);
  }

  /**
   * Remove a callback for performance issues
   */
  removePerformanceIssueCallback(callback: (metric: string, value: number, threshold: number) => void): boolean {
    const index = this.performanceIssueCallbacks.indexOf(callback);
    if (index !== -1) {
      this.performanceIssueCallbacks.splice(index, 1);
      return true;
    }
    return false;
  }

  /**
   * Get current performance metrics
   */
  getMetrics(): PerformanceMetrics {
    return { ...this.metrics };
  }

  /**
   * Get a specific metric value
   */
  getMetric(name: keyof PerformanceMetrics): number | ResourceLoadMetrics[] | Record<string, number> | undefined {
    return this.metrics[name];
  }

  /**
   * Clear all metrics
   */
  clearMetrics(): void {
    this.metrics = {} as PerformanceMetrics;
  }

  /**
   * Update monitoring options
   */
  updateOptions(options: Partial<PerformanceMonitoringOptions>): void {
    this.options = { ...this.options, ...options };
  }

  /**
   * Enable performance monitoring
   */
  enable(): void {
    if (!this.options.enabled) {
      this.options.enabled = true;
      this.setupPerformanceMonitoring();
    }
  }

  /**
   * Disable performance monitoring
   */
  disable(): void {
    this.options.enabled = false;
    this.disconnectObservers();
  }

  /**
   * Disconnect all performance observers
   */
  private disconnectObservers(): void {
    if (this.observer) {
      this.observer.disconnect();
      this.observer = null;
    }
    if (this.longTaskObserver) {
      this.longTaskObserver.disconnect();
      this.longTaskObserver = null;
    }
    if (this.layoutShiftObserver) {
      this.layoutShiftObserver.disconnect();
      this.layoutShiftObserver = null;
    }
    if (this.resourceObserver) {
      this.resourceObserver.disconnect();
      this.resourceObserver = null;
    }
    if (this.navigationObserver) {
      this.navigationObserver.disconnect();
      this.navigationObserver = null;
    }
    if (this.paintObserver) {
      this.paintObserver.disconnect();
      this.paintObserver = null;
    }
  }

  /**
   * Destroy the service and clean up
   */
  destroy(): void {
    this.disconnectObservers();
    this.performanceIssueCallbacks = [];
  }

  /**
   * Get performance score based on metrics
   */
  getPerformanceScore(): number {
    let score = 100;

    // Deduct points for performance issues
    if (this.metrics.largestContentfulPaint && this.metrics.largestContentfulPaint > this.options.thresholds.slowLCPthreshold) {
      score -= 20;
    }

    if (this.metrics.firstContentfulPaint && this.metrics.firstContentfulPaint > this.options.thresholds.slowFCPthreshold) {
      score -= 15;
    }

    if (this.metrics.cumulativeLayoutShift && this.metrics.cumulativeLayoutShift > this.options.thresholds.highCLSthreshold) {
      score -= 25;
    }

    if (this.metrics.memoryUsage && this.metrics.memoryUsage > this.options.thresholds.highMemoryUsageThreshold) {
      score -= 10;
    }

    // Cap the score between 0 and 100
    return Math.max(0, Math.min(100, score));
  }

  /**
   * Check if performance is good
   */
  isPerformanceGood(): boolean {
    return this.getPerformanceScore() >= 80;
  }

  /**
   * Check if performance is poor
   */
  isPerformancePoor(): boolean {
    return this.getPerformanceScore() < 50;
  }
}

// Create a singleton instance
export const performanceMonitoringService = PerformanceMonitoringService.getInstance();

/**
 * Helper function to measure function execution time
 */
export async function measureFunctionPerformance<T>(fn: () => Promise<T> | T, name: string): Promise<{ result: T; duration: number }> {
  return performanceMonitoringService.measureFunction(fn, name);
}

/**
 * Helper function to measure async operation performance
 */
export async function measureAsyncOperationPerformance<T>(
  operation: () => Promise<T>,
  name: string,
  options?: { reportSlowOperations?: boolean; threshold?: number }
): Promise<{ result: T; duration: number }> {
  return performanceMonitoringService.measureAsyncOperation(operation, name, options);
}

/**
 * Helper function to get performance metrics
 */
export function getPerformanceMetrics(): PerformanceMetrics {
  return performanceMonitoringService.getMetrics();
}

/**
 * Helper function to get performance score
 */
export function getPerformanceScore(): number {
  return performanceMonitoringService.getPerformanceScore();
}