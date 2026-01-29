/**
 * Enhanced Error Context and Metadata Collection
 * Provides comprehensive context gathering and metadata enrichment for errors
 */

import { ErrorContext as BasicErrorContext } from './errorLogging';

// Define extended error context with additional metadata
export interface ExtendedErrorContext extends BasicErrorContext {
  // Basic context (inherited)
  component?: string;
  function?: string;
  operation?: string;
  userId?: string;
  sessionId?: string;
  url?: string;
  userAgent?: string;
  timestamp?: Date;
  additionalData?: Record<string, any>;
  
  // Enhanced context fields
  stackTrace?: string;
  environment?: string;
  version?: string;
  browserInfo?: BrowserInfo;
  deviceInfo?: DeviceInfo;
  networkInfo?: NetworkInfo;
  performanceMetrics?: PerformanceMetrics;
  userActionTrail?: UserAction[];
  relatedComponents?: string[];
  previousErrors?: ErrorSummary[];
  correlationId?: string;
  requestId?: string;
  sessionData?: any;
  applicationState?: any;
  systemResources?: SystemResources;
  customTags?: string[];
}

// Browser information interface
export interface BrowserInfo {
  name: string;
  version: string;
  engine: string;
  engineVersion: string;
  userAgent: string;
  cookiesEnabled: boolean;
  localStorageEnabled: boolean;
  sessionStorageEnabled: boolean;
}

// Device information interface
export interface DeviceInfo {
  type: 'mobile' | 'tablet' | 'desktop' | 'unknown';
  manufacturer: string;
  model: string;
  os: string;
  osVersion: string;
  screenResolution: string;
  pixelRatio: number;
  hardwareConcurrency: number;
}

// Network information interface
export interface NetworkInfo {
  type: string; // 'wifi', 'cellular', 'ethernet', etc.
  effectiveType: string; // 'slow-2g', '2g', '3g', '4g', '5g'
  downlink: number; // Download speed in Mbps
  rtt: number; // Round-trip time in milliseconds
  saveData: boolean; // Whether data saving mode is enabled
}

// Performance metrics interface
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
}

// User action interface
export interface UserAction {
  timestamp: number;
  action: string;
  element?: string;
  details?: any;
}

// Error summary interface
export interface ErrorSummary {
  timestamp: number;
  message: string;
  type: string;
  component?: string;
}

// System resources interface
export interface SystemResources {
  memory: {
    used: number;
    total: number;
    available: number;
  };
  cpu: {
    usage: number;
    cores: number;
  };
  storage: {
    used: number;
    total: number;
    available: number;
  };
}

/**
 * Enhanced Error Context Manager
 * Collects and manages comprehensive error context and metadata
 */
export class EnhancedErrorContextManager {
  private static instance: EnhancedErrorContextManager;
  private contextCache: Map<string, ExtendedErrorContext> = new Map();
  private correlationId: string;
  private sessionStartTime: number;
  private userActionTrail: UserAction[] = [];
  private previousErrors: ErrorSummary[] = [];
  private performanceObserver: PerformanceObserver | null = null;

  private constructor() {
    this.correlationId = this.generateCorrelationId();
    this.sessionStartTime = Date.now();
    this.setupPerformanceMonitoring();
    this.setupUserActionTracking();
  }

  /**
   * Get singleton instance
   */
  static getInstance(): EnhancedErrorContextManager {
    if (!EnhancedErrorContextManager.instance) {
      EnhancedErrorContextManager.instance = new EnhancedErrorContextManager();
    }
    return EnhancedErrorContextManager.instance;
  }

  /**
   * Create enhanced error context
   */
  async createEnhancedContext(
    basicContext: BasicErrorContext = {},
    additionalMetadata: Record<string, any> = {}
  ): Promise<ExtendedErrorContext> {
    const context: ExtendedErrorContext = {
      ...basicContext,
      timestamp: basicContext.timestamp || new Date(),
      stackTrace: this.getCurrentStackTrace(),
      environment: this.getEnvironment(),
      version: this.getVersion(),
      browserInfo: await this.getBrowserInfo(),
      deviceInfo: await this.getDeviceInfo(),
      networkInfo: await this.getNetworkInfo(),
      performanceMetrics: await this.getPerformanceMetrics(),
      userActionTrail: [...this.userActionTrail],
      previousErrors: [...this.previousErrors],
      correlationId: this.correlationId,
      requestId: this.generateRequestId(),
      sessionData: this.getSessionData(),
      applicationState: this.getApplicationState(),
      systemResources: await this.getSystemResources(),
      customTags: additionalMetadata.tags || [],
      additionalData: {
        ...basicContext.additionalData,
        ...additionalMetadata,
        errorCollectionTime: Date.now(),
      }
    };

    // Cache the context
    const cacheKey = this.generateContextCacheKey(context);
    this.contextCache.set(cacheKey, context);

    // Limit cache size
    if (this.contextCache.size > 100) {
      const firstKey = this.contextCache.keys().next().value;
      this.contextCache.delete(firstKey);
    }

    return context;
  }

  /**
   * Add user action to trail
   */
  addUserAction(action: string, element?: string, details?: any): void {
    this.userActionTrail.push({
      timestamp: Date.now(),
      action,
      element,
      details
    });

    // Limit trail size
    if (this.userActionTrail.length > 50) {
      this.userActionTrail = this.userActionTrail.slice(-50);
    }
  }

  /**
   * Record error for context
   */
  recordError(error: any, component?: string): void {
    this.previousErrors.push({
      timestamp: Date.now(),
      message: error.message || String(error),
      type: error.constructor?.name || typeof error,
      component
    });

    // Limit error history
    if (this.previousErrors.length > 20) {
      this.previousErrors = this.previousErrors.slice(-20);
    }
  }

  /**
   * Get current stack trace
   */
  private getCurrentStackTrace(): string {
    try {
      throw new Error();
    } catch (e) {
      return e.stack || '';
    }
  }

  /**
   * Get environment information
   */
  private getEnvironment(): string {
    if (typeof window !== 'undefined') {
      if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        return 'development';
      } else if (window.location.hostname.includes('staging') || window.location.hostname.includes('test')) {
        return 'staging';
      } else {
        return 'production';
      }
    }
    return 'unknown';
  }

  /**
   * Get application version
   */
  private getVersion(): string {
    // In a real app, this would come from package.json or build process
    return typeof window !== 'undefined' 
      ? (window as any).__OPENEVOLVE_VERSION__ || '1.0.0' 
      : '1.0.0';
  }

  /**
   * Get browser information
   */
  private async getBrowserInfo(): Promise<BrowserInfo> {
    if (typeof navigator === 'undefined') {
      return {
        name: 'unknown',
        version: 'unknown',
        engine: 'unknown',
        engineVersion: 'unknown',
        userAgent: 'unknown',
        cookiesEnabled: false,
        localStorageEnabled: false,
        sessionStorageEnabled: false
      };
    }

    // Detect browser
    let name = 'unknown';
    let version = 'unknown';
    const userAgent = navigator.userAgent;

    if (userAgent.includes('Chrome')) {
      name = 'Chrome';
      const match = userAgent.match(/Chrome\/(\d+\.\d+)/);
      if (match) version = match[1];
    } else if (userAgent.includes('Firefox')) {
      name = 'Firefox';
      const match = userAgent.match(/Firefox\/(\d+\.\d+)/);
      if (match) version = match[1];
    } else if (userAgent.includes('Safari')) {
      name = 'Safari';
      const match = userAgent.match(/Version\/(\d+\.\d+)/);
      if (match) version = match[1];
    } else if (userAgent.includes('Edge')) {
      name = 'Edge';
      const match = userAgent.match(/Edg\/(\d+\.\d+)/);
      if (match) version = match[1];
    }

    // Detect engine
    let engine = 'unknown';
    let engineVersion = 'unknown';
    if (userAgent.includes('Gecko')) {
      engine = 'Gecko';
      const match = userAgent.match(/rv:(\d+\.\d+)/);
      if (match) engineVersion = match[1];
    } else if (userAgent.includes('WebKit')) {
      engine = 'WebKit';
      const match = userAgent.match(/WebKit\/(\d+\.\d+)/);
      if (match) engineVersion = match[1];
    } else if (userAgent.includes('Blink')) {
      engine = 'Blink';
      // Version usually matches Chrome version for Blink
      engineVersion = version;
    }

    return {
      name,
      version,
      engine,
      engineVersion,
      userAgent,
      cookiesEnabled: navigator.cookieEnabled,
      localStorageEnabled: typeof Storage !== 'undefined',
      sessionStorageEnabled: typeof Storage !== 'undefined'
    };
  }

  /**
   * Get device information
   */
  private async getDeviceInfo(): Promise<DeviceInfo> {
    if (typeof navigator === 'undefined' || typeof screen === 'undefined') {
      return {
        type: 'unknown',
        manufacturer: 'unknown',
        model: 'unknown',
        os: 'unknown',
        osVersion: 'unknown',
        screenResolution: 'unknown',
        pixelRatio: 1,
        hardwareConcurrency: 1
      };
    }

    // Detect device type
    let type: 'mobile' | 'tablet' | 'desktop' | 'unknown' = 'unknown';
    const width = screen.width;
    const height = screen.height;
    const userAgent = navigator.userAgent.toLowerCase();

    if (/(android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini)/i.test(userAgent)) {
      type = width <= 768 ? 'mobile' : 'tablet';
    } else {
      type = 'desktop';
    }

    // Detect OS
    let os = 'unknown';
    let osVersion = 'unknown';
    
    if (userAgent.includes('win')) {
      os = 'Windows';
      // Could detect version but keeping it simple
    } else if (userAgent.includes('mac')) {
      os = 'MacOS';
    } else if (userAgent.includes('linux')) {
      os = 'Linux';
    } else if (userAgent.includes('android')) {
      os = 'Android';
    } else if (userAgent.includes('ios') || userAgent.includes('iphone') || userAgent.includes('ipad')) {
      os = 'iOS';
    }

    return {
      type,
      manufacturer: 'unknown', // Would require more sophisticated detection
      model: 'unknown', // Would require more sophisticated detection
      os,
      osVersion,
      screenResolution: `${width}x${height}`,
      pixelRatio: window.devicePixelRatio || 1,
      hardwareConcurrency: navigator.hardwareConcurrency || 1
    };
  }

  /**
   * Get network information
   */
  private async getNetworkInfo(): Promise<NetworkInfo> {
    if (typeof navigator === 'undefined') {
      return {
        type: 'unknown',
        effectiveType: 'unknown',
        downlink: 0,
        rtt: 0,
        saveData: false
      };
    }

    // Use the Network Information API if available
    const connection = (navigator as any).connection || 
                      (navigator as any).mozConnection || 
                      (navigator as any).webkitConnection;

    if (connection) {
      return {
        type: connection.type || 'unknown',
        effectiveType: connection.effectiveType || 'unknown',
        downlink: connection.downlink || 0,
        rtt: connection.rtt || 0,
        saveData: connection.saveData || false
      };
    }

    // Fallback to basic information
    return {
      type: 'unknown',
      effectiveType: 'unknown',
      downlink: 0,
      rtt: 0,
      saveData: false
    };
  }

  /**
   * Get performance metrics
   */
  private async getPerformanceMetrics(): Promise<PerformanceMetrics> {
    if (typeof performance === 'undefined') {
      return {
        navigationStart: 0,
        domContentLoaded: 0,
        loadComplete: 0
      };
    }

    const timing = performance.timing;
    const navStart = timing.navigationStart;

    const metrics: PerformanceMetrics = {
      navigationStart: 0,
      domContentLoaded: 0,
      loadComplete: 0
    };

    if (timing) {
      metrics.navigationStart = 0; // Relative to navigation start
      metrics.domContentLoaded = timing.domContentLoadedEventEnd - navStart;
      metrics.loadComplete = timing.loadEventEnd - navStart;
    }

    // Add memory info if available
    if ((performance as any).memory) {
      const mem = (performance as any).memory;
      metrics.memoryUsage = mem.usedJSHeapSize;
      metrics.jsHeapSizeLimit = mem.jsHeapSizeLimit;
      metrics.totalJSHeapSize = mem.totalJSHeapSize;
      metrics.usedJSHeapSize = mem.usedJSHeapSize;
    }

    // Add paint timing if available
    const entries = performance.getEntriesByType('paint');
    entries.forEach(entry => {
      if (entry.name === 'first-paint') {
        metrics.firstPaint = entry.startTime;
      } else if (entry.name === 'first-contentful-paint') {
        metrics.firstContentfulPaint = entry.startTime;
      }
    });

    // Add LCP if available
    const lcpEntries = performance.getEntriesByName('largest-contentful-paint');
    if (lcpEntries.length > 0) {
      metrics.largestContentfulPaint = (lcpEntries[lcpEntries.length - 1] as any).startTime;
    }

    return metrics;
  }

  /**
   * Get session data
   */
  private getSessionData(): any {
    if (typeof window === 'undefined') {
      return null;
    }

    // Collect relevant session data
    return {
      sessionAge: Date.now() - this.sessionStartTime,
      url: window.location.href,
      referrer: document.referrer,
      title: document.title,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      language: navigator.language || 'unknown'
    };
  }

  /**
   * Get application state
   */
  private getApplicationState(): any {
    // In a real app, this would collect state from your state management solution
    // (Redux, Zustand, etc.) or other application state
    if (typeof window !== 'undefined' && (window as any).__OPENEVOLVE_STATE__) {
      return (window as any).__OPENEVOLVE_STATE__;
    }
    
    return null;
  }

  /**
   * Get system resources
   */
  private async getSystemResources(): Promise<SystemResources> {
    // Browser APIs for system resources are limited, but we can get some info
    return {
      memory: {
        used: 0, // Would need special APIs that aren't widely available
        total: 0,
        available: 0
      },
      cpu: {
        usage: 0, // Would need special APIs that aren't widely available
        cores: navigator.hardwareConcurrency || 1
      },
      storage: {
        used: 0, // Would need special APIs
        total: 0,
        available: 0
      }
    };
  }

  /**
   * Setup performance monitoring
   */
  private setupPerformanceMonitoring(): void {
    if (typeof PerformanceObserver !== 'undefined') {
      this.performanceObserver = new PerformanceObserver((list) => {
        // Process performance entries as they become available
        list.getEntries().forEach((entry) => {
          // Could store performance data for context
        });
      });

      try {
        this.performanceObserver.observe({ entryTypes: ['measure', 'navigation', 'paint', 'largest-contentful-paint'] });
      } catch (e) {
        // Some browsers might not support certain entry types
        console.warn('Performance observer setup partially failed:', e);
      }
    }
  }

  /**
   * Setup user action tracking
   */
  private setupUserActionTracking(): void {
    if (typeof window !== 'undefined') {
      // Track clicks
      window.addEventListener('click', (e) => {
        this.addUserAction('click', (e.target as Element).tagName, {
          x: e.clientX,
          y: e.clientY,
          targetId: (e.target as Element).id,
          targetClass: (e.target as Element).className
        });
      });

      // Track form submissions
      window.addEventListener('submit', (e) => {
        this.addUserAction('form_submit', (e.target as Element).tagName, {
          targetId: (e.target as Element).id,
          targetForm: (e.target as HTMLFormElement).name
        });
      });

      // Track keyboard events
      window.addEventListener('keydown', (e) => {
        this.addUserAction('keydown', e.key, {
          keyCode: e.keyCode,
          ctrlKey: e.ctrlKey,
          shiftKey: e.shiftKey,
          altKey: e.altKey
        });
      });
    }
  }

  /**
   * Generate correlation ID
   */
  private generateCorrelationId(): string {
    return `corr_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Generate request ID
   */
  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Generate context cache key
   */
  private generateContextCacheKey(context: ExtendedErrorContext): string {
    return `${context.component || 'unknown'}_${context.function || 'unknown'}_${Date.now()}`;
  }

  /**
   * Get context from cache
   */
  getContextFromCache(key: string): ExtendedErrorContext | undefined {
    return this.contextCache.get(key);
  }

  /**
   * Clear context cache
   */
  clearContextCache(): void {
    this.contextCache.clear();
  }

  /**
   * Get current correlation ID
   */
  getCurrentCorrelationId(): string {
    return this.correlationId;
  }

  /**
   * Get user action trail
   */
  getUserActionTrail(): UserAction[] {
    return [...this.userActionTrail];
  }

  /**
   * Get previous errors
   */
  getPreviousErrors(): ErrorSummary[] {
    return [...this.previousErrors];
  }

  /**
   * Destroy the instance (cleanup)
   */
  destroy(): void {
    if (this.performanceObserver) {
      this.performanceObserver.disconnect();
      this.performanceObserver = null;
    }
  }
}

// Create a singleton instance
export const enhancedErrorContextManager = EnhancedErrorContextManager.getInstance();

/**
 * Helper function to create enhanced error context
 */
export async function createEnhancedErrorContext(
  basicContext: BasicErrorContext = {},
  additionalMetadata: Record<string, any> = {}
): Promise<ExtendedErrorContext> {
  return enhancedErrorContextManager.createEnhancedContext(basicContext, additionalMetadata);
}

/**
 * Helper function to record an error for context
 */
export function recordErrorForContext(error: any, component?: string): void {
  enhancedErrorContextManager.recordError(error, component);
}

/**
 * Helper function to add user action to trail
 */
export function addUserActionToContext(action: string, element?: string, details?: any): void {
  enhancedErrorContextManager.addUserAction(action, element, details);
}