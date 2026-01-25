/**
 * Error Handling Configuration
 * Provides centralized configuration for error handling systems
 */

// Define error handling configuration options
export interface ErrorHandlingConfig {
  // Global settings
  enabled: boolean;
  logLevel: 'debug' | 'info' | 'warn' | 'error' | 'critical';
  enableUserNotifications: boolean;
  enableErrorReporting: boolean;
  enableRecovery: boolean;
  enableCaching: boolean;
  enableCircuitBreaker: boolean;
  
  // Retry settings
  defaultMaxRetries: number;
  defaultRetryDelay: number; // in ms
  maxRetryDelay: number; // in ms (for exponential backoff)
  retryJitter: boolean; // Add randomization to delay
  
  // Circuit breaker settings
  circuitBreakerEnabled: boolean;
  circuitBreakerFailureThreshold: number; // Number of failures before opening
  circuitBreakerTimeout: number; // Time in ms before attempting reset
  circuitBreakerFailureRateThreshold: number; // Percentage threshold
  
  // Cache settings
  cacheEnabled: boolean;
  defaultCacheExpiry: number; // in ms
  maxCacheSize: number; // Maximum number of cached items
  
  // Reporting settings
  reportTimeout: number; // Time to wait for error reports
  maxConcurrentReports: number; // Max number of concurrent error reports
  reportBatchSize: number; // Number of reports to batch together
  
  // Performance settings
  performanceMonitoringEnabled: boolean;
  slowOperationThreshold: number; // Time in ms to consider an operation slow
  memoryUsageThreshold: number; // Percentage of memory usage to trigger warnings
  
  // Security settings
  sanitizeErrorMessages: boolean; // Remove sensitive data from error messages
  enableSecurityScanning: boolean; // Scan errors for security issues
  
  // Custom strategies
  customRecoveryStrategies: string[];
  customFallbackStrategies: string[];
  
  // UI settings
  toastPosition: 'top-right' | 'top-center' | 'top-left' | 'bottom-right' | 'bottom-center' | 'bottom-left';
  toastAutoClose: number; // in ms
  toastCloseButton: boolean;
  toastProgressBar: boolean;
  
  // Environment-specific settings
  development: ErrorHandlingEnvConfig;
  staging: ErrorHandlingEnvConfig;
  production: ErrorHandlingEnvConfig;
}

// Define environment-specific configuration
export interface ErrorHandlingEnvConfig {
  enabled: boolean;
  logLevel: 'debug' | 'info' | 'warn' | 'error' | 'critical';
  enableUserNotifications: boolean;
  enableErrorReporting: boolean;
  reportToExternalServices: boolean;
  logToConsole: boolean;
  logToFile: boolean;
}

// Default configuration values
export const DEFAULT_ERROR_HANDLING_CONFIG: ErrorHandlingConfig = {
  enabled: true,
  logLevel: 'error',
  enableUserNotifications: true,
  enableErrorReporting: true,
  enableRecovery: true,
  enableCaching: true,
  enableCircuitBreaker: true,
  
  defaultMaxRetries: 3,
  defaultRetryDelay: 1000,
  maxRetryDelay: 30000,
  retryJitter: true,
  
  circuitBreakerEnabled: true,
  circuitBreakerFailureThreshold: 5,
  circuitBreakerTimeout: 30000,
  circuitBreakerFailureRateThreshold: 50,
  
  cacheEnabled: true,
  defaultCacheExpiry: 300000, // 5 minutes
  maxCacheSize: 1000,
  
  reportTimeout: 10000,
  maxConcurrentReports: 5,
  reportBatchSize: 10,
  
  performanceMonitoringEnabled: true,
  slowOperationThreshold: 5000,
  memoryUsageThreshold: 80,
  
  sanitizeErrorMessages: true,
  enableSecurityScanning: true,
  
  customRecoveryStrategies: [],
  customFallbackStrategies: [],
  
  toastPosition: 'top-right',
  toastAutoClose: 5000,
  toastCloseButton: true,
  toastProgressBar: true,
  
  development: {
    enabled: true,
    logLevel: 'debug',
    enableUserNotifications: true,
    enableErrorReporting: false, // Don't report dev errors to external services
    reportToExternalServices: false,
    logToConsole: true,
    logToFile: true
  },
  
  staging: {
    enabled: true,
    logLevel: 'warn',
    enableUserNotifications: true,
    enableErrorReporting: true,
    reportToExternalServices: true,
    logToConsole: true,
    logToFile: false
  },
  
  production: {
    enabled: true,
    logLevel: 'error',
    enableUserNotifications: true,
    enableErrorReporting: true,
    reportToExternalServices: true,
    logToConsole: false,
    logToFile: false
  }
};

/**
 * Error Handling Configuration Manager
 * Manages configuration for error handling systems
 */
export class ErrorHandlingConfigManager {
  private static instance: ErrorHandlingConfigManager;
  private config: ErrorHandlingConfig;
  private originalConfig: ErrorHandlingConfig;

  private constructor() {
    this.config = { ...DEFAULT_ERROR_HANDLING_CONFIG };
    this.originalConfig = { ...DEFAULT_ERROR_HANDLING_CONFIG };
    this.applyEnvironmentConfig();
  }

  /**
   * Get singleton instance
   */
  static getInstance(): ErrorHandlingConfigManager {
    if (!ErrorHandlingConfigManager.instance) {
      ErrorHandlingConfigManager.instance = new ErrorHandlingConfigManager();
    }
    return ErrorHandlingConfigManager.instance;
  }

  /**
   * Apply configuration based on current environment
   */
  private applyEnvironmentConfig(): void {
    const env = this.getCurrentEnvironment();
    const envConfig = this.config[env];
    
    if (envConfig) {
      // Override global settings with environment-specific settings
      this.config.enabled = envConfig.enabled;
      this.config.logLevel = envConfig.logLevel;
      this.config.enableUserNotifications = envConfig.enableUserNotifications;
      this.config.enableErrorReporting = envConfig.enableErrorReporting;
    }
  }

  /**
   * Get current environment
   */
  private getCurrentEnvironment(): 'development' | 'staging' | 'production' {
    if (typeof window !== 'undefined') {
      if (window.location.hostname === 'localhost' || 
          window.location.hostname === '127.0.0.1' || 
          window.location.port) {
        return 'development';
      } else if (window.location.hostname.includes('staging') || 
                 window.location.hostname.includes('test')) {
        return 'staging';
      } else {
        return 'production';
      }
    }
    
    // For Node.js environments
    const nodeEnv = process.env.NODE_ENV;
    if (nodeEnv === 'development' || nodeEnv === 'dev') {
      return 'development';
    } else if (nodeEnv === 'staging' || nodeEnv === 'test') {
      return 'staging';
    } else {
      return 'production';
    }
  }

  /**
   * Get current configuration
   */
  getConfig(): ErrorHandlingConfig {
    return { ...this.config };
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<ErrorHandlingConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Update environment-specific configuration
   */
  updateEnvironmentConfig(
    environment: 'development' | 'staging' | 'production',
    envConfig: Partial<ErrorHandlingEnvConfig>
  ): void {
    (this.config[environment] as any) = { 
      ...this.config[environment], 
      ...envConfig 
    };
  }

  /**
   * Reset to default configuration
   */
  resetToDefault(): void {
    this.config = { ...this.originalConfig };
    this.applyEnvironmentConfig();
  }

  /**
   * Check if a log level should be processed
   */
  shouldLog(level: 'debug' | 'info' | 'warn' | 'error' | 'critical'): boolean {
    const levels = ['debug', 'info', 'warn', 'error', 'critical'];
    const currentLevelIndex = levels.indexOf(this.config.logLevel);
    const messageLevelIndex = levels.indexOf(level);
    
    return messageLevelIndex >= currentLevelIndex;
  }

  /**
   * Check if user notifications are enabled
   */
  areUserNotificationsEnabled(): boolean {
    return this.config.enableUserNotifications;
  }

  /**
   * Check if error reporting is enabled
   */
  isReportingEnabled(): boolean {
    return this.config.enableErrorReporting;
  }

  /**
   * Check if recovery is enabled
   */
  isRecoveryEnabled(): boolean {
    return this.config.enableRecovery;
  }

  /**
   * Check if caching is enabled
   */
  isCachingEnabled(): boolean {
    return this.config.enableCaching;
  }

  /**
   * Check if circuit breaker is enabled
   */
  isCircuitBreakerEnabled(): boolean {
    return this.config.enableCircuitBreaker && this.config.circuitBreakerEnabled;
  }

  /**
   * Get retry settings
   */
  getRetrySettings() {
    return {
      maxRetries: this.config.defaultMaxRetries,
      baseDelay: this.config.defaultRetryDelay,
      maxDelay: this.config.maxRetryDelay,
      useJitter: this.config.retryJitter
    };
  }

  /**
   * Get circuit breaker settings
   */
  getCircuitBreakerSettings() {
    return {
      enabled: this.config.circuitBreakerEnabled,
      failureThreshold: this.config.circuitBreakerFailureThreshold,
      timeout: this.config.circuitBreakerTimeout,
      failureRateThreshold: this.config.circuitBreakerFailureRateThreshold
    };
  }

  /**
   * Get cache settings
   */
  getCacheSettings() {
    return {
      enabled: this.config.cacheEnabled,
      defaultExpiry: this.config.defaultCacheExpiry,
      maxCacheSize: this.config.maxCacheSize
    };
  }

  /**
   * Get toast settings
   */
  getToastSettings() {
    return {
      position: this.config.toastPosition,
      autoClose: this.config.toastAutoClose,
      closeButton: this.config.toastCloseButton,
      progressBar: this.config.toastProgressBar
    };
  }

  /**
   * Get performance monitoring settings
   */
  getPerformanceSettings() {
    return {
      enabled: this.config.performanceMonitoringEnabled,
      slowOpThreshold: this.config.slowOperationThreshold,
      memoryThreshold: this.config.memoryUsageThreshold
    };
  }

  /**
   * Get security settings
   */
  getSecuritySettings() {
    return {
      sanitizeMessages: this.config.sanitizeErrorMessages,
      enableScanning: this.config.enableSecurityScanning
    };
  }
}

// Create a singleton instance
export const errorHandlingConfigManager = ErrorHandlingConfigManager.getInstance();

/**
 * Helper function to get current configuration
 */
export function getErrorHandlingConfig(): ErrorHandlingConfig {
  return errorHandlingConfigManager.getConfig();
}

/**
 * Helper function to update configuration
 */
export function updateErrorHandlingConfig(newConfig: Partial<ErrorHandlingConfig>): void {
  errorHandlingConfigManager.updateConfig(newConfig);
}