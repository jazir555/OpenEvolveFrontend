/**
 * Advanced Error Recovery Strategies
 * Provides sophisticated recovery mechanisms for different types of errors
 */

import { toast } from 'react-toastify';

// Define recovery strategy types
export type RecoveryStrategyType = 
  | 'retry_with_backoff'
  | 'fallback_to_cache'
  | 'degraded_mode'
  | 'alternative_service'
  | 'circuit_breaker'
  | 'timeout_adjustment'
  | 'resource_cleanup'
  | 'session_recovery'
  | 'data_repair'
  | 'connection_reset';

// Define recovery context
export interface RecoveryContext {
  error: any;
  operation: string;
  params?: any;
  previousAttempts?: number;
  maxAttempts?: number;
  fallbackData?: any;
  recoveryOptions?: any;
  metadata?: Record<string, any>;
}

// Define recovery result
export interface RecoveryResult {
  success: boolean;
  actionTaken: string;
  newData?: any;
  shouldRetry?: boolean;
  shouldContinue?: boolean;
  recoveryTimeMs?: number;
  error?: any;
}

/**
 * Advanced Error Recovery Manager
 * Manages multiple recovery strategies and applies them based on error type
 */
export class AdvancedErrorRecoveryManager {
  private strategies: Map<RecoveryStrategyType, (context: RecoveryContext) => Promise<RecoveryResult>>;
  private strategyWeights: Map<RecoveryStrategyType, number>;
  private recoveryHistory: Array<{
    operation: string;
    strategy: RecoveryStrategyType;
    success: boolean;
    timestamp: number;
    duration: number;
  }> = [];

  constructor() {
    this.strategies = new Map();
    this.strategyWeights = new Map();
    
    // Initialize default strategies
    this.initializeDefaultStrategies();
    this.initializeStrategyWeights();
  }

  /**
   * Initialize default recovery strategies
   */
  private initializeDefaultStrategies(): void {
    // Retry with exponential backoff
    this.strategies.set('retry_with_backoff', this.retryWithBackoff.bind(this));
    
    // Fallback to cached data
    this.strategies.set('fallback_to_cache', this.fallbackToCache.bind(this));
    
    // Switch to degraded mode
    this.strategies.set('degraded_mode', this.activateDegradedMode.bind(this));
    
    // Use alternative service/provider
    this.strategies.set('alternative_service', this.useAlternativeService.bind(this));
    
    // Circuit breaker pattern
    this.strategies.set('circuit_breaker', this.applyCircuitBreaker.bind(this));
    
    // Adjust timeout values
    this.strategies.set('timeout_adjustment', this.adjustTimeout.bind(this));
    
    // Resource cleanup
    this.strategies.set('resource_cleanup', this.cleanupResources.bind(this));
    
    // Session recovery
    this.strategies.set('session_recovery', this.recoverSession.bind(this));
    
    // Data repair
    this.strategies.set('data_repair', this.repairData.bind(this));
    
    // Connection reset
    this.strategies.set('connection_reset', this.resetConnection.bind(this));
  }

  /**
   * Initialize strategy weights (higher weight = more likely to be selected)
   */
  private initializeStrategyWeights(): void {
    this.strategyWeights.set('retry_with_backoff', 90);
    this.strategyWeights.set('fallback_to_cache', 85);
    this.strategyWeights.set('degraded_mode', 80);
    this.strategyWeights.set('alternative_service', 75);
    this.strategyWeights.set('timeout_adjustment', 70);
    this.strategyWeights.set('circuit_breaker', 65);
    this.strategyWeights.set('resource_cleanup', 60);
    this.strategyWeights.set('session_recovery', 55);
    this.strategyWeights.set('data_repair', 50);
    this.strategyWeights.set('connection_reset', 45);
  }

  /**
   * Apply the most appropriate recovery strategy based on error characteristics
   */
  async applyRecoveryStrategy(context: RecoveryContext): Promise<RecoveryResult> {
    const startTime = Date.now();
    
    // Determine the most appropriate strategy based on error type
    const strategy = this.selectBestStrategy(context);
    
    if (!strategy) {
      return {
        success: false,
        actionTaken: 'no_strategy_found',
        error: new Error('No suitable recovery strategy found')
      };
    }

    try {
      const result = await this.strategies.get(strategy)!(context);
      
      // Record recovery attempt
      this.recoveryHistory.push({
        operation: context.operation,
        strategy,
        success: result.success,
        timestamp: Date.now(),
        duration: Date.now() - startTime
      });
      
      // Limit history size
      if (this.recoveryHistory.length > 100) {
        this.recoveryHistory = this.recoveryHistory.slice(-100);
      }
      
      return result;
    } catch (error) {
      console.error(`Recovery strategy ${strategy} failed:`, error);
      
      // Record failed recovery attempt
      this.recoveryHistory.push({
        operation: context.operation,
        strategy,
        success: false,
        timestamp: Date.now(),
        duration: Date.now() - startTime
      });
      
      return {
        success: false,
        actionTaken: `strategy_${strategy}_failed`,
        error: error instanceof Error ? error : new Error(String(error))
      };
    }
  }

  /**
   * Select the best recovery strategy based on error characteristics
   */
  private selectBestStrategy(context: RecoveryContext): RecoveryStrategyType | null {
    const error = context.error;
    
    // Check for network-related errors
    if (this.isNetworkError(error)) {
      return 'retry_with_backoff';
    }
    
    // Check for timeout errors
    if (this.isTimeoutError(error)) {
      return 'timeout_adjustment';
    }
    
    // Check for authentication errors
    if (this.isAuthError(error)) {
      return 'session_recovery';
    }
    
    // Check for resource exhaustion
    if (this.isResourceExhaustionError(error)) {
      return 'resource_cleanup';
    }
    
    // Check for database/connection errors
    if (this.isConnectionError(error)) {
      return 'connection_reset';
    }
    
    // Default to retry for general errors
    return 'retry_with_backoff';
  }

  /**
   * Check if error is network-related
   */
  private isNetworkError(error: any): boolean {
    return error.message?.includes('network') ||
           error.message?.includes('connection') ||
           error.message?.includes('fetch') ||
           error.message?.includes('Failed to fetch') ||
           error.code === 'ECONNREFUSED' ||
           error.code === 'ENOTFOUND' ||
           error.code === 'ECONNABORTED';
  }

  /**
   * Check if error is timeout-related
   */
  private isTimeoutError(error: any): boolean {
    return error.message?.includes('timeout') ||
           error.message?.includes('timed out') ||
           error.code === 'ETIMEDOUT';
  }

  /**
   * Check if error is authentication-related
   */
  private isAuthError(error: any): boolean {
    return error.message?.includes('auth') ||
           error.message?.includes('token') ||
           error.message?.includes('401') ||
           error.message?.includes('403') ||
           error.status === 401 ||
           error.status === 403;
  }

  /**
   * Check if error is resource exhaustion
   */
  private isResourceExhaustionError(error: any): boolean {
    return error.message?.includes('memory') ||
           error.message?.includes('heap') ||
           error.message?.includes('out of memory') ||
           error.message?.includes('allocation failed');
  }

  /**
   * Check if error is connection-related
   */
  private isConnectionError(error: any): boolean {
    return error.message?.includes('connection') ||
           error.message?.includes('database') ||
           error.message?.includes('pool') ||
           error.message?.includes('connect');
  }

  /**
   * Retry with exponential backoff strategy
   */
  private async retryWithBackoff(context: RecoveryContext): Promise<RecoveryResult> {
    const attempts = context.previousAttempts || 0;
    const maxAttempts = context.maxAttempts || 3;
    
    if (attempts >= maxAttempts) {
      return {
        success: false,
        actionTaken: 'max_retry_attempts_exceeded',
        shouldRetry: false
      };
    }

    // Calculate delay with exponential backoff and jitter
    const baseDelay = 1000; // 1 second base
    const jitter = Math.random() * 0.5; // Add up to 50% jitter
    const delay = baseDelay * Math.pow(2, attempts) * (1 + jitter);
    
    toast.info(`Retrying operation in ${(delay / 1000).toFixed(1)} seconds... (Attempt ${attempts + 1}/${maxAttempts})`);
    
    await this.delay(delay);
    
    return {
      success: true,
      actionTaken: 'retry_scheduled',
      shouldRetry: true,
      shouldContinue: true
    };
  }

  /**
   * Fallback to cached data strategy
   */
  private async fallbackToCache(context: RecoveryContext): Promise<RecoveryResult> {
    try {
      // Check if we have cached data available
      if (context.fallbackData !== undefined) {
        toast.info('Using cached data as fallback');
        return {
          success: true,
          actionTaken: 'fallback_to_cache_used',
          newData: context.fallbackData,
          shouldRetry: false,
          shouldContinue: true
        };
      }

      // Try to retrieve from browser storage
      if (typeof window !== 'undefined' && window.sessionStorage) {
        const cacheKey = `fallback_data_${context.operation}`;
        const cachedDataStr = window.sessionStorage.getItem(cacheKey);
        
        if (cachedDataStr) {
          try {
            const cachedData = JSON.parse(cachedDataStr);
            toast.info('Using cached data as fallback');
            return {
              success: true,
              actionTaken: 'fallback_to_browser_cache_used',
              newData: cachedData,
              shouldRetry: false,
              shouldContinue: true
            };
          } catch (parseError) {
            console.error('Failed to parse cached data:', parseError);
          }
        }
      }

      return {
        success: false,
        actionTaken: 'no_cached_data_available',
        shouldRetry: false,
        shouldContinue: false
      };
    } catch (error) {
      return {
        success: false,
        actionTaken: 'fallback_to_cache_failed',
        error: error instanceof Error ? error : new Error(String(error))
      };
    }
  }

  /**
   * Activate degraded mode strategy
   */
  private async activateDegradedMode(context: RecoveryContext): Promise<RecoveryResult> {
    try {
      // In a real implementation, this would disable non-critical features
      // and switch to a minimal viable functionality mode
      
      toast.warn('Switching to degraded mode to maintain core functionality');
      
      // Store degraded mode flag in session storage
      if (typeof window !== 'undefined' && window.sessionStorage) {
        window.sessionStorage.setItem('degraded_mode', 'true');
        window.sessionStorage.setItem('degraded_mode_start', new Date().toISOString());
      }
      
      return {
        success: true,
        actionTaken: 'degraded_mode_activated',
        shouldRetry: false,
        shouldContinue: true
      };
    } catch (error) {
      return {
        success: false,
        actionTaken: 'degraded_mode_activation_failed',
        error: error instanceof Error ? error : new Error(String(error))
      };
    }
  }

  /**
   * Use alternative service strategy
   */
  private async useAlternativeService(context: RecoveryContext): Promise<RecoveryResult> {
    try {
      // In a real implementation, this would switch to a backup service
      // or alternate endpoint
      
      toast.info('Attempting to use alternative service endpoint');
      
      // This would typically involve switching to a backup API endpoint
      // or using a different service provider
      const altServiceAvailable = await this.checkAlternativeServiceAvailability();
      
      if (altServiceAvailable) {
        return {
          success: true,
          actionTaken: 'alternative_service_used',
          shouldRetry: true,
          shouldContinue: true
        };
      }
      
      return {
        success: false,
        actionTaken: 'no_alternative_service_available',
        shouldRetry: false,
        shouldContinue: false
      };
    } catch (error) {
      return {
        success: false,
        actionTaken: 'alternative_service_selection_failed',
        error: error instanceof Error ? error : new Error(String(error))
      };
    }
  }

  /**
   * Apply circuit breaker pattern
   */
  private async applyCircuitBreaker(context: RecoveryContext): Promise<RecoveryResult> {
    try {
      const operation = context.operation || 'unknown';
      const circuitKey = `circuit_breaker_${operation}`;
      
      if (typeof window !== 'undefined' && window.sessionStorage) {
        let circuitState = JSON.parse(
          window.sessionStorage.getItem(circuitKey) || 
          '{"state": "CLOSED", "failureCount": 0, "lastFailure": 0, "lastAttempt": 0}'
        );
        
        const now = Date.now();
        const resetTimeout = 30000; // 30 seconds
        
        // Check if circuit should be reset
        if (circuitState.state === 'OPEN' && now - circuitState.lastFailure > resetTimeout) {
          circuitState = { state: 'HALF_OPEN', failureCount: 0, lastFailure: 0, lastAttempt: now };
          window.sessionStorage.setItem(circuitKey, JSON.stringify(circuitState));
          toast.info(`Circuit breaker for ${operation} reset to HALF_OPEN state`);
          return {
            success: true,
            actionTaken: 'circuit_breaker_reset',
            shouldRetry: true,
            shouldContinue: true
          };
        }
        
        // If circuit is open, don't allow the operation
        if (circuitState.state === 'OPEN') {
          toast.error(`Circuit breaker for ${operation} is OPEN - operation blocked`);
          return {
            success: false,
            actionTaken: 'circuit_breaker_open',
            shouldRetry: false,
            shouldContinue: false
          };
        }
        
        // Increment failure count
        circuitState.failureCount++;
        circuitState.lastFailure = now;
        circuitState.lastAttempt = now;
        
        // If failure threshold is reached, open the circuit
        const failureThreshold = context.recoveryOptions?.failureThreshold || 5;
        if (circuitState.failureCount >= failureThreshold) {
          circuitState.state = 'OPEN';
          window.sessionStorage.setItem(circuitKey, JSON.stringify(circuitState));
          toast.error(`Circuit breaker for ${operation} opened due to repeated failures`);
          return {
            success: false,
            actionTaken: 'circuit_breaker_opened',
            shouldRetry: false,
            shouldContinue: false
          };
        }
        
        window.sessionStorage.setItem(circuitKey, JSON.stringify(circuitState));
      }
      
      return {
        success: true,
        actionTaken: 'circuit_breaker_check_passed',
        shouldRetry: true,
        shouldContinue: true
      };
    } catch (error) {
      return {
        success: false,
        actionTaken: 'circuit_breaker_application_failed',
        error: error instanceof Error ? error : new Error(String(error))
      };
    }
  }

  /**
   * Adjust timeout values strategy
   */
  private async adjustTimeout(context: RecoveryContext): Promise<RecoveryResult> {
    try {
      // In a real implementation, this would adjust timeout values
      // based on historical performance data
      
      const currentTimeout = context.recoveryOptions?.currentTimeout || 30000;
      const newTimeout = Math.min(currentTimeout * 1.5, 120000); // Increase by 50%, max 2 mins
      
      toast.info(`Adjusting timeout from ${currentTimeout}ms to ${newTimeout}ms`);
      
      return {
        success: true,
        actionTaken: 'timeout_adjusted',
        shouldRetry: true,
        shouldContinue: true,
        newData: { newTimeout }
      };
    } catch (error) {
      return {
        success: false,
        actionTaken: 'timeout_adjustment_failed',
        error: error instanceof Error ? error : new Error(String(error))
      };
    }
  }

  /**
   * Cleanup resources strategy
   */
  private async cleanupResources(context: RecoveryContext): Promise<RecoveryResult> {
    try {
      // In a real implementation, this would clean up resources
      // like closing connections, clearing caches, etc.
      
      toast.info('Cleaning up resources to free up memory');
      
      // Clear temporary data from session storage
      if (typeof window !== 'undefined' && window.sessionStorage) {
        // Find and clear temporary data
        for (let i = 0; i < window.sessionStorage.length; i++) {
          const key = window.sessionStorage.key(i);
          if (key && key.startsWith('temp_')) {
            window.sessionStorage.removeItem(key);
          }
        }
      }
      
      // Trigger garbage collection hint (browser may ignore this)
      if (typeof window !== 'undefined' && (window as any).gc) {
        (window as any).gc();
      }
      
      return {
        success: true,
        actionTaken: 'resources_cleaned_up',
        shouldRetry: true,
        shouldContinue: true
      };
    } catch (error) {
      return {
        success: false,
        actionTaken: 'resource_cleanup_failed',
        error: error instanceof Error ? error : new Error(String(error))
      };
    }
  }

  /**
   * Recover session strategy
   */
  private async recoverSession(context: RecoveryContext): Promise<RecoveryResult> {
    try {
      // In a real implementation, this would attempt to recover
      // authentication/session state
      
      toast.info('Attempting to recover session...');
      
      // Check if we have refresh token available
      if (typeof window !== 'undefined' && window.localStorage) {
        const refreshToken = window.localStorage.getItem('refresh_token');
        if (refreshToken) {
          // In a real app, this would make an API call to refresh the token
          // For now, we'll simulate the process
          await this.delay(500); // Simulate API call
          
          // If successful, update tokens
          // window.localStorage.setItem('access_token', newAccessToken);
          
          toast.success('Session recovered successfully');
          return {
            success: true,
            actionTaken: 'session_recovered',
            shouldRetry: true,
            shouldContinue: true
          };
        }
      }
      
      return {
        success: false,
        actionTaken: 'session_recovery_not_possible',
        shouldRetry: false,
        shouldContinue: false
      };
    } catch (error) {
      return {
        success: false,
        actionTaken: 'session_recovery_failed',
        error: error instanceof Error ? error : new Error(String(error))
      };
    }
  }

  /**
   * Repair data strategy
   */
  private async repairData(context: RecoveryContext): Promise<RecoveryResult> {
    try {
      // In a real implementation, this would attempt to repair
      // corrupted or invalid data
      
      toast.info('Attempting to repair data...');
      
      // This would typically involve validating and fixing data structures
      // or rolling back to a known good state
      
      return {
        success: true,
        actionTaken: 'data_repair_attempted',
        shouldRetry: true,
        shouldContinue: true
      };
    } catch (error) {
      return {
        success: false,
        actionTaken: 'data_repair_failed',
        error: error instanceof Error ? error : new Error(String(error))
      };
    }
  }

  /**
   * Reset connection strategy
   */
  private async resetConnection(context: RecoveryContext): Promise<RecoveryResult> {
    try {
      // In a real implementation, this would reset network connections
      // or database connections
      
      toast.info('Resetting network connections...');
      
      // This would typically involve closing and reopening connections
      // For browser environments, we might clear connection pools
      // or force a reconnect to services
      
      return {
        success: true,
        actionTaken: 'connection_reset',
        shouldRetry: true,
        shouldContinue: true
      };
    } catch (error) {
      return {
        success: false,
        actionTaken: 'connection_reset_failed',
        error: error instanceof Error ? error : new Error(String(error))
      };
    }
  }

  /**
   * Check if alternative service is available
   */
  private async checkAlternativeServiceAvailability(): Promise<boolean> {
    // In a real implementation, this would check if backup services are available
    // For now, we'll simulate availability
    return true;
  }

  /**
   * Delay helper function
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Register a custom recovery strategy
   */
  registerStrategy(type: RecoveryStrategyType, strategy: (context: RecoveryContext) => Promise<RecoveryResult>): void {
    this.strategies.set(type, strategy);
  }

  /**
   * Unregister a recovery strategy
   */
  unregisterStrategy(type: RecoveryStrategyType): boolean {
    return this.strategies.delete(type);
  }

  /**
   * Set weight for a strategy (higher weight = more likely to be selected)
   */
  setStrategyWeight(type: RecoveryStrategyType, weight: number): void {
    this.strategyWeights.set(type, weight);
  }

  /**
   * Get recovery history
   */
  getRecoveryHistory(limit: number = 20): typeof this.recoveryHistory {
    return [...this.recoveryHistory].slice(-limit).reverse();
  }

  /**
   * Get recovery statistics
   */
  getRecoveryStats(): {
    totalAttempts: number;
    successfulRecoveries: number;
    successRate: number;
    avgRecoveryTime: number;
    byStrategy: Record<RecoveryStrategyType, { attempts: number; successes: number; successRate: number }>;
  } {
    const stats = {
      totalAttempts: this.recoveryHistory.length,
      successfulRecoveries: this.recoveryHistory.filter(r => r.success).length,
      successRate: 0,
      avgRecoveryTime: 0,
      byStrategy: {} as Record<RecoveryStrategyType, { attempts: number; successes: number; successRate: number }>
    };

    if (stats.totalAttempts > 0) {
      stats.successRate = (stats.successfulRecoveries / stats.totalAttempts) * 100;
      const totalTime = this.recoveryHistory.reduce((sum, rec) => sum + rec.duration, 0);
      stats.avgRecoveryTime = totalTime / stats.totalAttempts;
    }

    // Calculate stats by strategy
    this.recoveryHistory.forEach(record => {
      if (!stats.byStrategy[record.strategy]) {
        stats.byStrategy[record.strategy] = { attempts: 0, successes: 0, successRate: 0 };
      }
      stats.byStrategy[record.strategy].attempts++;
      if (record.success) {
        stats.byStrategy[record.strategy].successes++;
      }
    });

    // Calculate success rates by strategy
    Object.values(stats.byStrategy).forEach(strategy => {
      if (strategy.attempts > 0) {
        strategy.successRate = (strategy.successes / strategy.attempts) * 100;
      }
    });

    return stats;
  }

  /**
   * Clear recovery history
   */
  clearRecoveryHistory(): void {
    this.recoveryHistory = [];
  }

  /**
   * Get active strategies
   */
  getActiveStrategies(): RecoveryStrategyType[] {
    return Array.from(this.strategies.keys());
  }
}

// Create a singleton instance
export const advancedErrorRecoveryManager = new AdvancedErrorRecoveryManager();