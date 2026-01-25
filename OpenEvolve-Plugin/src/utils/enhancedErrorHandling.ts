import { v4 as uuidv4 } from 'uuid';
import { toast } from 'react-toastify';

/**
 * Enhanced Error Handling System
 * Provides sophisticated error classification, recovery strategies, and reporting
 */

// Error Classification System
/**
 * Advanced Error Classifier with ML-like pattern matching
 */
export class AdvancedErrorClassifier {
  private errorPatterns: Map<string, {
    pattern: RegExp | ((error: any) => boolean);
    category: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    description: string;
    recoverySuggestions: string[];
  }> = new Map();

  constructor() {
    // Initialize with common error patterns
    this.initializeDefaultPatterns();
  }

  /**
   * Initialize default error patterns
   */
  private initializeDefaultPatterns(): void {
    // Network errors
    this.addPattern('network_error', {
      pattern: (error: any) => 
        error.message?.includes('network') || 
        error.message?.includes('connection') ||
        error.message?.includes('timeout') ||
        error.code === 'ECONNABORTED' ||
        error.code === 'ENOTFOUND',
      category: 'network_error',
      severity: 'high',
      description: 'Network connectivity issues',
      recoverySuggestions: [
        'Check internet connection',
        'Verify API endpoint availability',
        'Implement retry logic with exponential backoff',
        'Check firewall and network settings',
      ],
    });

    // Authentication errors
    this.addPattern('authentication_error', {
      pattern: (error: any) => 
        error.message?.includes('authentication') ||
        error.message?.includes('unauthorized') ||
        error.message?.includes('401') ||
        error.message?.includes('token') ||
        error.code === 401,
      category: 'authentication_error',
      severity: 'high',
      description: 'Authentication or authorization failures',
      recoverySuggestions: [
        'Verify credentials',
        'Check token expiration',
        'Renew authentication token',
        'Verify user permissions',
      ],
    });

    // Validation errors
    this.addPattern('validation_error', {
      pattern: (error: any) => 
        error.message?.includes('validation') ||
        error.message?.includes('invalid') ||
        error.message?.includes('required') ||
        error.message?.includes('format') ||
        error.code === 422,
      category: 'validation_error',
      severity: 'medium',
      description: 'Input validation failures',
      recoverySuggestions: [
        'Check input data format',
        'Validate required fields',
        'Review data constraints',
        'Provide user-friendly error messages',
      ],
    });

    // Rate limiting errors
    this.addPattern('rate_limit_error', {
      pattern: (error: any) => 
        error.message?.includes('rate limit') ||
        error.message?.includes('too many requests') ||
        error.message?.includes('429') ||
        error.code === 429,
      category: 'rate_limit_error',
      severity: 'medium',
      description: 'API rate limiting exceeded',
      recoverySuggestions: [
        'Implement exponential backoff',
        'Check rate limit policies',
        'Optimize API calls',
        'Consider upgrading API plan',
      ],
    });

    // Database errors
    this.addPattern('database_error', {
      pattern: (error: any) => 
        error.message?.includes('database') ||
        error.message?.includes('query') ||
        error.message?.includes('connection') ||
        error.message?.includes('SQL') ||
        error.message?.includes('Mongo') ||
        error.message?.includes('collection'),
      category: 'database_error',
      severity: 'critical',
      description: 'Database operation failures',
      recoverySuggestions: [
        'Check database connection',
        'Verify query syntax',
        'Review database logs',
        'Implement transaction rollback',
      ],
    });

    // Memory errors
    this.addPattern('memory_error', {
      pattern: (error: any) => 
        error.message?.includes('memory') ||
        error.message?.includes('heap') ||
        error.message?.includes('out of memory') ||
        error.message?.includes('allocation'),
      category: 'memory_error',
      severity: 'critical',
      description: 'Memory allocation failures',
      recoverySuggestions: [
        'Increase memory limits',
        'Optimize memory usage',
        'Implement garbage collection',
        'Review memory-intensive operations',
      ],
    });

    // Timeout errors
    this.addPattern('timeout_error', {
      pattern: (error: any) => 
        error.message?.includes('timeout') ||
        error.message?.includes('timed out') ||
        error.code === 'ETIMEDOUT',
      category: 'timeout_error',
      severity: 'high',
      description: 'Operation timeout',
      recoverySuggestions: [
        'Increase timeout values',
        'Optimize slow operations',
        'Implement retry logic',
        'Check network latency',
      ],
    });

    // Configuration errors
    this.addPattern('configuration_error', {
      pattern: (error: any) => 
        error.message?.includes('configuration') ||
        error.message?.includes('config') ||
        error.message?.includes('setting') ||
        error.message?.includes('environment'),
      category: 'configuration_error',
      severity: 'high',
      description: 'Configuration issues',
      recoverySuggestions: [
        'Verify configuration files',
        'Check environment variables',
        'Review default settings',
        'Validate configuration schema',
      ],
    });
  }

  /**
   * Add custom error pattern
   */
  addPattern(
    id: string,
    pattern: {
      pattern: RegExp | ((error: any) => boolean);
      category: string;
      severity: 'low' | 'medium' | 'high' | 'critical';
      description: string;
      recoverySuggestions: string[];
    }
  ): void {
    this.errorPatterns.set(id, pattern);
  }

  /**
   * Remove error pattern
   */
  removePattern(id: string): boolean {
    return this.errorPatterns.delete(id);
  }

  /**
   * Classify error using pattern matching
   */
  classifyError(error: any): {
    category: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    description: string;
    recoverySuggestions: string[];
    confidence: number;
  } {
    let bestMatch: {
      category: string;
      severity: 'low' | 'medium' | 'high' | 'critical';
      description: string;
      recoverySuggestions: string[];
      confidence: number;
    } = {
      category: 'general_error',
      severity: 'medium',
      description: 'General error',
      recoverySuggestions: ['Check error details', 'Review logs', 'Contact support'],
      confidence: 0.5,
    };

    this.errorPatterns.forEach(pattern => {
      let matches = false;

      if (pattern.pattern instanceof RegExp) {
        matches = pattern.pattern.test(error.message || error.toString());
      } else if (typeof pattern.pattern === 'function') {
        matches = pattern.pattern(error);
      }

      if (matches) {
        // Calculate confidence based on pattern specificity
        const confidence = this.calculateConfidence(pattern, error);
        
        if (confidence > bestMatch.confidence) {
          bestMatch = {
            category: pattern.category,
            severity: pattern.severity,
            description: pattern.description,
            recoverySuggestions: pattern.recoverySuggestions,
            confidence,
          };
        }
      }
    });

    return bestMatch;
  }

  /**
   * Calculate confidence score for pattern match
   */
  private calculateConfidence(pattern: any, error: any): number {
    // Simple confidence calculation based on pattern type and match quality
    if (pattern.pattern instanceof RegExp) {
      const message = error.message || error.toString();
      const matches = message.match(pattern.pattern);
      return matches ? matches.length / 10 : 0.1;
    }

    return 0.8; // Function patterns get higher confidence
  }

  /**
   * Get all error patterns
   */
  getAllPatterns(): Map<string, any> {
    return new Map(this.errorPatterns);
  }
}

// Error Recovery System
/**
 * Advanced Error Recovery System
 */
export class AdvancedErrorRecovery {
  private recoveryStrategies: Map<string, {
    strategy: (error: any, context: any) => Promise<boolean>;
    applicableCategories: string[];
    description: string;
  }> = new Map();

  constructor() {
    this.initializeDefaultStrategies();
  }

  /**
   * Initialize default recovery strategies
   */
  private initializeDefaultStrategies(): void {
    // Retry strategy
    this.addStrategy('retry', {
      strategy: async (error: any, context: any = {}) => {
        const maxRetries = context.maxRetries || 3;
        const retryDelay = context.retryDelay || 1000;
        const currentRetry = context.currentRetry || 0;

        if (currentRetry < maxRetries) {
          toast.info(`Attempting retry ${currentRetry + 1}/${maxRetries}...`);
          await new Promise(resolve => setTimeout(resolve, retryDelay));
          return true; // Indicate that retry should be attempted
        }

        return false; // No more retries
      },
      applicableCategories: ['network_error', 'timeout_error', 'rate_limit_error'],
      description: 'Retry the failed operation with exponential backoff',
    });

    // Fallback strategy
    this.addStrategy('fallback', {
      strategy: async (error: any, context: any = {}) => {
        if (context.fallbackFunction && typeof context.fallbackFunction === 'function') {
          try {
            toast.info('Attempting fallback operation...');
            await context.fallbackFunction();
            return true;
          } catch (fallbackError) {
            toast.error('Fallback operation failed');
            return false;
          }
        }
        return false;
      },
      applicableCategories: ['network_error', 'database_error', 'api_error'],
      description: 'Execute fallback operation when primary fails',
    });

    // Cache strategy
    this.addStrategy('cache', {
      strategy: async (error: any, context: any = {}) => {
        if (context.cache && context.cacheKey) {
          const cachedData = context.cache.get(context.cacheKey);
          if (cachedData) {
            toast.info('Using cached data as fallback');
            return true;
          }
        }
        return false;
      },
      applicableCategories: ['network_error', 'api_error', 'database_error'],
      description: 'Use cached data when fresh data is unavailable',
    });

    // Circuit breaker strategy
    this.addStrategy('circuit_breaker', {
      strategy: async (error: any, context: any = {}) => {
        // Simple circuit breaker implementation
        const failureThreshold = context.failureThreshold || 5;
        const resetTimeout = context.resetTimeout || 30000;

        if (!context.circuitState) {
          context.circuitState = { failures: 0, lastFailure: 0, isOpen: false };
        }

        const now = Date.now();

        if (context.circuitState.isOpen) {
          if (now - context.circuitState.lastFailure > resetTimeout) {
            // Reset circuit after timeout
            context.circuitState.failures = 0;
            context.circuitState.isOpen = false;
            toast.info('Circuit breaker reset');
            return true; // Allow retry
          } else {
            toast.warn('Circuit breaker is open - operation skipped');
            return false; // Don't retry
          }
        }

        // Increment failure count
        context.circuitState.failures++;
        context.circuitState.lastFailure = now;

        if (context.circuitState.failures >= failureThreshold) {
          context.circuitState.isOpen = true;
          toast.error('Circuit breaker opened due to repeated failures');
        }

        return true; // Allow retry
      },
      applicableCategories: ['network_error', 'database_error', 'api_error'],
      description: 'Circuit breaker pattern to prevent cascading failures',
    });

    // Graceful degradation strategy
    this.addStrategy('graceful_degradation', {
      strategy: async (error: any, context: any = {}) => {
        if (context.degradedFunction && typeof context.degradedFunction === 'function') {
          try {
            toast.info('Switching to degraded functionality');
            await context.degradedFunction();
            return true;
          } catch (degradedError) {
            toast.error('Degraded functionality also failed');
            return false;
          }
        }
        return false;
      },
      applicableCategories: ['network_error', 'database_error', 'api_error', 'performance_error'],
      description: 'Switch to degraded functionality when full functionality fails',
    });
  }

  /**
   * Add custom recovery strategy
   */
  addStrategy(
    id: string,
    strategy: {
      strategy: (error: any, context: any) => Promise<boolean>;
      applicableCategories: string[];
      description: string;
    }
  ): void {
    this.recoveryStrategies.set(id, strategy);
  }

  /**
   * Remove recovery strategy
   */
  removeStrategy(id: string): boolean {
    return this.recoveryStrategies.delete(id);
  }

  /**
   * Attempt error recovery using appropriate strategies
   */
  async attemptRecovery(
    error: any,
    classification: {
      category: string;
      severity: 'low' | 'medium' | 'high' | 'critical';
    },
    context: any = {}
  ): Promise<{
    success: boolean;
    strategiesAttempted: string[];
    finalError?: any;
  }> {
    const strategiesAttempted: string[] = [];
    let finalError = error;

    // Get applicable strategies for this error category
    const applicableStrategies = Array.from(this.recoveryStrategies.entries())
      .filter(([_, strategy]) => 
        strategy.applicableCategories.includes(classification.category) ||
        strategy.applicableCategories.includes('*')
      );

    // Sort by most specific to least specific
    applicableStrategies.sort((a, b) => {
      const aSpecific: number = a[1].applicableCategories.includes(classification.category) ? 1 : 0;
      const bSpecific: number = b[1].applicableCategories.includes(classification.category) ? 1 : 0;
      return bSpecific - aSpecific;
    });

    // Attempt each strategy in order
    for (const [strategyId, strategy] of applicableStrategies) {
      try {
        strategiesAttempted.push(strategyId);
        toast.info(`Attempting recovery strategy: ${strategy.description}`);

        const recoveryContext = { ...context, error, classification };
        const shouldContinue = await strategy.strategy(error, recoveryContext);

        if (shouldContinue) {
          // Strategy succeeded or indicated to continue
          return { success: true, strategiesAttempted };
        }
      } catch (strategyError) {
        toast.error(`Recovery strategy failed: ${strategy.description}`);
        finalError = strategyError;
        strategiesAttempted.push(`${strategyId}_failed`);
      }
    }

    return { 
      success: false, 
      strategiesAttempted, 
      finalError 
    };
  }

  /**
   * Get recovery strategy suggestions for error category
   */
  getRecoverySuggestions(category: string): string[] {
    const suggestions: string[] = [];

    this.recoveryStrategies.forEach(strategy => {
      if (strategy.applicableCategories.includes(category) || 
          strategy.applicableCategories.includes('*')) {
        suggestions.push(strategy.description);
      }
    });

    return suggestions;
  }

  /**
   * Get all recovery strategies
   */
  getAllStrategies(): Map<string, any> {
    return new Map(this.recoveryStrategies);
  }
}

// Error Reporting System
/**
 * Advanced Error Reporting System
 */
export class AdvancedErrorReporter {
  private errorQueue: Array<{
    errorId: string;
    errorData: any;
    timestamp: number;
    retries: number;
  }> = [];

  private reportingDestinations: Map<string, {
    reporter: (errorData: any) => Promise<boolean>;
    enabled: boolean;
    maxRetries: number;
  }> = new Map();

  constructor() {
    this.initializeDefaultDestinations();
  }

  /**
   * Initialize default reporting destinations
   */
  private initializeDefaultDestinations(): void {
    // Console reporter
    this.addDestination('console', {
      reporter: async (errorData: any) => {
        console.error('Error Report:', errorData);
        return true;
      },
      enabled: true,
      maxRetries: 0,
    });

    // Mock API reporter
    this.addDestination('api', {
      reporter: async (errorData: any) => {
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 500));
        console.log('Reported to API:', errorData);
        return true;
      },
      enabled: false, // Disabled by default
      maxRetries: 3,
    });

    // Mock email reporter
    this.addDestination('email', {
      reporter: async (errorData: any) => {
        // Simulate email sending
        await new Promise(resolve => setTimeout(resolve, 1000));
        console.log('Sent error email:', errorData);
        return true;
      },
      enabled: false, // Disabled by default
      maxRetries: 2,
    });

    // Mock database reporter
    this.addDestination('database', {
      reporter: async (errorData: any) => {
        // Simulate database storage
        await new Promise(resolve => setTimeout(resolve, 300));
        console.log('Stored error in database:', errorData);
        return true;
      },
      enabled: true,
      maxRetries: 3,
    });
  }

  /**
   * Add reporting destination
   */
  addDestination(
    id: string,
    destination: {
      reporter: (errorData: any) => Promise<boolean>;
      enabled: boolean;
      maxRetries: number;
    }
  ): void {
    this.reportingDestinations.set(id, destination);
  }

  /**
   * Remove reporting destination
   */
  removeDestination(id: string): boolean {
    return this.reportingDestinations.delete(id);
  }

  /**
   * Enable/disable reporting destination
   */
  setDestinationEnabled(id: string, enabled: boolean): boolean {
    const destination = this.reportingDestinations.get(id);
    if (destination) {
      destination.enabled = enabled;
      return true;
    }
    return false;
  }

  /**
   * Queue error for reporting
   */
  queueError(errorData: any): string {
    const errorId = uuidv4();
    
    this.errorQueue.push({
      errorId,
      errorData,
      timestamp: Date.now(),
      retries: 0,
    });

    toast.info(`Error queued for reporting (ID: ${errorId})`);
    return errorId;
  }

  /**
   * Process error queue
   */
  async processQueue(): Promise<{
    successCount: number;
    failureCount: number;
    totalProcessed: number;
  }> {
    let successCount = 0;
    let failureCount = 0;

    while (this.errorQueue.length > 0) {
      const queueItem = this.errorQueue[0];
      const { errorId, errorData, retries } = queueItem;

      try {
        const results = await this.reportToDestinations(errorData);
        
        if (results.some(r => r.success)) {
          // At least one destination succeeded
          this.errorQueue.shift();
          successCount++;
          toast.success(`Error reported successfully (ID: ${errorId})`);
        } else {
          // All destinations failed
          if (retries >= this.getMaxRetriesForError(errorData)) {
            this.errorQueue.shift();
            failureCount++;
            toast.error(`Failed to report error after max retries (ID: ${errorId})`);
          } else {
            // Increment retries and try again later
            queueItem.retries++;
            toast.warn(`Retrying error report (attempt ${queueItem.retries})`);
            break; // Process next item in queue
          }
        }
      } catch (error) {
        toast.error(`Error processing queue: ${error instanceof Error ? error.message : String(error)}`);
        break;
      }
    }

    return {
      successCount,
      failureCount,
      totalProcessed: successCount + failureCount,
    };
  }

  /**
   * Report error to all enabled destinations
   */
  private async reportToDestinations(errorData: any): Promise<Array<{ 
    destination: string; 
    success: boolean; 
    error?: string; 
  }>> {
    const results: Array<{ destination: string; success: boolean; error?: string }> = [];

    for (const [destinationId, destination] of this.reportingDestinations) {
      if (destination.enabled) {
        try {
          const success = await destination.reporter(errorData);
          results.push({ destination: destinationId, success });
          
          if (success) {
            toast.success(`Reported to ${destinationId} successfully`);
          } else {
            toast.warn(`Failed to report to ${destinationId}`);
          }
        } catch (error) {
          results.push({
            destination: destinationId,
            success: false,
            error: error instanceof Error ? error.message : String(error),
          });
          toast.error(`Error reporting to ${destinationId}: ${error instanceof Error ? error.message : String(error)}`);
        }
      }
    }

    return results;
  }

  /**
   * Get max retries for error based on destinations
   */
  private getMaxRetriesForError(errorData: any): number {
    let maxRetries = 0;

    this.reportingDestinations.forEach(destination => {
      if (destination.enabled && destination.maxRetries > maxRetries) {
        maxRetries = destination.maxRetries;
      }
    });

    return maxRetries;
  }

  /**
   * Get queue status
   */
  getQueueStatus(): {
    queueSize: number;
    oldestErrorTimestamp?: number;
    destinations: Array<{ id: string; enabled: boolean; maxRetries: number }>;
  } {
    return {
      queueSize: this.errorQueue.length,
      oldestErrorTimestamp: this.errorQueue.length > 0 ? this.errorQueue[0].timestamp : undefined,
      destinations: Array.from(this.reportingDestinations.entries()).map(([id, dest]) => ({
        id,
        enabled: dest.enabled,
        maxRetries: dest.maxRetries,
      })),
    };
  }

  /**
   * Clear error queue
   */
  clearQueue(): void {
    this.errorQueue = [];
    toast.info('Error queue cleared');
  }

  /**
   * Get all reporting destinations
   */
  getAllDestinations(): Map<string, any> {
    return new Map(this.reportingDestinations);
  }
}

// Comprehensive Error Handling System
/**
 * Comprehensive Error Handling System
 * Combines classification, recovery, and reporting
 */
export class ComprehensiveErrorHandler {
  private classifier: AdvancedErrorClassifier;
  private recovery: AdvancedErrorRecovery;
  private reporter: AdvancedErrorReporter;
  private errorHistory: Array<{
    errorId: string;
    originalError: any;
    classification: any;
    recoveryAttempt: any;
    reportingStatus: any;
    timestamp: number;
  }> = [];

  constructor() {
    this.classifier = new AdvancedErrorClassifier();
    this.recovery = new AdvancedErrorRecovery();
    this.reporter = new AdvancedErrorReporter();
  }

  /**
   * Handle error with comprehensive processing
   */
  async handleError(
    error: any,
    context: any = {},
    options: {
      autoRecover?: boolean;
      autoReport?: boolean;
      metadata?: Record<string, any>;
    } = {}
  ): Promise<{
    errorId: string;
    classification: any;
    recoveryResult?: any;
    reportingResult?: any;
  }> {
    const { autoRecover = true, autoReport = true, metadata = {} } = options;
    const errorId = uuidv4();
    const timestamp = Date.now();

    // Step 1: Classify the error
    const classification = this.classifier.classifyError(error);

    // Step 2: Attempt recovery if enabled
    let recoveryResult: any = null;
    if (autoRecover) {
      try {
        recoveryResult = await this.recovery.attemptRecovery(error, classification, {
          ...context,
          errorId,
          classification,
          metadata,
        });
      } catch (recoveryError) {
        console.error('Recovery failed:', recoveryError);
        recoveryResult = { 
          success: false, 
          strategiesAttempted: ['recovery_failed'],
          finalError: recoveryError,
        };
      }
    }

    // Step 3: Report error if enabled
    let reportingResult: any = null;
    if (autoReport) {
      try {
        const errorData = this.prepareErrorData(error, classification, recoveryResult, context, metadata);
        this.reporter.queueError(errorData);
        
        // Process queue immediately for critical errors
        if (classification.severity === 'critical') {
          reportingResult = await this.reporter.processQueue();
        } else {
          reportingResult = { queued: true };
        }
      } catch (reportingError) {
        console.error('Reporting failed:', reportingError);
        reportingResult = { success: false, error: reportingError };
      }
    }

    // Step 4: Store error in history
    this.errorHistory.push({
      errorId,
      originalError: error,
      classification,
      recoveryAttempt: recoveryResult,
      reportingStatus: reportingResult,
      timestamp,
    });

    // Step 5: Log and notify based on severity
    this.notifyError(classification, errorId);

    return {
      errorId,
      classification,
      recoveryResult,
      reportingResult,
    };
  }

  /**
   * Prepare error data for reporting
   */
  private prepareErrorData(
    error: any,
    classification: any,
    recoveryResult: any,
    context: any,
    metadata: any
  ): any {
    return {
      errorId: uuidv4(),
      timestamp: Date.now(),
      errorType: classification.category,
      errorMessage: error.message || String(error),
      stackTrace: error.stack || null,
      severity: classification.severity,
      context: JSON.stringify(context),
      classification: JSON.stringify(classification),
      recoveryAttempt: recoveryResult ? JSON.stringify({
        success: recoveryResult.success,
        strategies: recoveryResult.strategiesAttempted,
      }) : null,
      metadata: JSON.stringify(metadata),
      userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown',
      platform: typeof process !== 'undefined' ? process.platform : 'unknown',
    };
  }

  /**
   * Notify error based on severity
   */
  private notifyError(classification: any, errorId: string): void {
    const message = `Error: ${classification.description} (${classification.category})`;

    switch (classification.severity) {
      case 'critical':
        toast.error(`🚨 CRITICAL: ${message} (ID: ${errorId})`);
        console.error('🚨 CRITICAL ERROR:', { classification, errorId });
        break;
      case 'high':
        toast.error(`❌ HIGH: ${message} (ID: ${errorId})`);
        console.error('❌ HIGH ERROR:', { classification, errorId });
        break;
      case 'medium':
        toast.warn(`⚠️ MEDIUM: ${message} (ID: ${errorId})`);
        console.warn('⚠️ MEDIUM ERROR:', { classification, errorId });
        break;
      case 'low':
        toast.info(`ℹ️ LOW: ${message} (ID: ${errorId})`);
        console.log('ℹ️ LOW ERROR:', { classification, errorId });
        break;
    }
  }

  /**
   * Process error queue
   */
  async processErrorQueue(): Promise<any> {
    return this.reporter.processQueue();
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
   * Get error statistics
   */
  getErrorStatistics(): {
    totalErrors: number;
    bySeverity: Record<string, number>;
    byCategory: Record<string, number>;
    recoverySuccessRate: number;
    reportingSuccessRate: number;
  } {
    const bySeverity: Record<string, number> = { low: 0, medium: 0, high: 0, critical: 0 };
    const byCategory: Record<string, number> = {};
    let successfulRecoveries = 0;
    let successfulReports = 0;

    this.errorHistory.forEach(error => {
      bySeverity[error.classification.severity]++;
      byCategory[error.classification.category] = (byCategory[error.classification.category] || 0) + 1;

      if (error.recoveryAttempt?.success) successfulRecoveries++;
      if (error.reportingStatus?.successCount) successfulReports++;
    });

    return {
      totalErrors: this.errorHistory.length,
      bySeverity,
      byCategory,
      recoverySuccessRate: this.errorHistory.length > 0 
        ? (successfulRecoveries / this.errorHistory.length) * 100
        : 0,
      reportingSuccessRate: this.errorHistory.length > 0
        ? (successfulReports / this.errorHistory.length) * 100
        : 0,
    };
  }

  /**
   * Get queue status
   */
  getQueueStatus(): any {
    return this.reporter.getQueueStatus();
  }

  /**
   * Add custom error pattern
   */
  addErrorPattern(id: string, pattern: any): void {
    this.classifier.addPattern(id, pattern);
  }

  /**
   * Add custom recovery strategy
   */
  addRecoveryStrategy(id: string, strategy: any): void {
    this.recovery.addStrategy(id, strategy);
  }

  /**
   * Add reporting destination
   */
  addReportingDestination(id: string, destination: any): void {
    this.reporter.addDestination(id, destination);
  }

  /**
   * Get all components
   */
  getComponents(): {
    classifier: AdvancedErrorClassifier;
    recovery: AdvancedErrorRecovery;
    reporter: AdvancedErrorReporter;
  } {
    return {
      classifier: this.classifier,
      recovery: this.recovery,
      reporter: this.reporter,
    };
  }
}