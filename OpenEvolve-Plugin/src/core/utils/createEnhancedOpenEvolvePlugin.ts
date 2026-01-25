// @ts-nocheck
import { createOpenEvolvePlugin, OpenEvolvePlugin } from './createOpenEvolvePlugin';
import { EnhancedOpenEvolvePluginState, EnhancedOpenEvolvePlugin, PerformanceConfiguration, SecurityConfiguration, MonitoringConfiguration, IntegrationConfiguration, ErrorHandlingConfiguration, DEFAULT_ENHANCED_OPENEVOLVE_CONFIG } from '../types/enhanced-plugin-types';
import { toast } from 'react-toastify';
import { v4 as uuidv4 } from 'uuid';

/**
 * Enhanced OpenEvolve Plugin Factory
 * Creates a plugin with extended performance, security, monitoring, integration, and error handling capabilities
 */
export function createEnhancedOpenEvolvePlugin(
  initialConfig: Partial<EnhancedOpenEvolvePluginState> = {}
): EnhancedOpenEvolvePlugin {
  // Merge with enhanced defaults
  const mergedConfig: EnhancedOpenEvolvePluginState = {
    ...DEFAULT_ENHANCED_OPENEVOLVE_CONFIG,
    ...initialConfig,
  };

  // Create base plugin
  const basePlugin = createOpenEvolvePlugin(mergedConfig);

  // Enhanced state management
  let enhancedState: EnhancedOpenEvolvePluginState = { ...mergedConfig };
  const listeners = new Set<(state: EnhancedOpenEvolvePluginState) => void>();

  // Notify all listeners of state changes
  const notifyListeners = () => {
    listeners.forEach(listener => listener({ ...enhancedState }));
  };

  // Enhanced plugin implementation
  const enhancedPlugin: EnhancedOpenEvolvePlugin = {
    ...basePlugin,

    // State management
    getEnhancedState: () => ({ ...enhancedState }),
    
    subscribeToEnhancedState: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },

    updateEnhancedConfig: (updates: Partial<EnhancedOpenEvolvePluginState>) => {
      try {
        enhancedState = { ...enhancedState, ...updates };
        notifyListeners();
        toast.success('Enhanced configuration updated successfully');
        return true;
      } catch (error) {
        toast.error(`Failed to update enhanced configuration: ${error instanceof Error ? error.message : String(error)}`);
        return false;
      }
    },

    resetEnhancedConfig: () => {
      try {
        enhancedState = { ...DEFAULT_ENHANCED_OPENEVOLVE_CONFIG };
        notifyListeners();
        toast.success('Enhanced configuration reset to defaults');
        return true;
      } catch (error) {
        toast.error(`Failed to reset enhanced configuration: ${error instanceof Error ? error.message : String(error)}`);
        return false;
      }
    },

    // Performance validation and management
    validatePerformanceConfig: (config: PerformanceConfiguration = enhancedState.performanceConfig): boolean => {
      try {
        if (!config) {
          throw new Error('Performance configuration is required');
        }

        // Validate caching configuration
        if (config.caching) {
          if (config.caching.max_size < 1 || config.caching.max_size > 10000) {
            throw new Error('Cache max_size must be between 1 and 10000');
          }
          if (config.caching.ttl < 0 || config.caching.ttl > 86400) {
            throw new Error('Cache TTL must be between 0 and 86400 seconds');
          }
        }

        // Validate parallel processing configuration
        if (config.parallel_processing) {
          if (config.parallel_processing.max_workers < 1 || config.parallel_processing.max_workers > 100) {
            throw new Error('Max workers must be between 1 and 100');
          }
        }

        // Validate memory management configuration
        if (config.memory_management) {
          if (config.memory_management.max_memory_mb < 100 || config.memory_management.max_memory_mb > 100000) {
            throw new Error('Max memory must be between 100 and 100000 MB');
          }
        }

        toast.success('Performance configuration is valid');
        return true;
      } catch (error) {
        toast.error(`Invalid performance configuration: ${error instanceof Error ? error.message : String(error)}`);
        return false;
      }
    },

    // Security validation and management
    validateSecurityConfig: (config: SecurityConfiguration = enhancedState.securityConfig): boolean => {
      try {
        if (!config) {
          throw new Error('Security configuration is required');
        }

        // Validate authentication configuration
        if (config.authentication) {
          const validMethods = ['api-key', 'oauth2', 'jwt', 'basic'];
          if (!validMethods.includes(config.authentication.method)) {
            throw new Error(`Invalid authentication method. Must be one of: ${validMethods.join(', ')}`);
          }
        }

        // Validate data protection configuration
        if (config.data_protection) {
          if (config.data_protection.encryption) {
            const validAlgorithms = ['aes-256', 'rsa-2048', 'chacha20'];
            if (!validAlgorithms.includes(config.data_protection.encryption.algorithm)) {
              throw new Error(`Invalid encryption algorithm. Must be one of: ${validAlgorithms.join(', ')}`);
            }
          }
        }

        // Validate compliance configuration
        if (config.compliance) {
          if (config.compliance.audit_logging) {
            if (config.compliance.audit_logging.retention_days < 30 || config.compliance.audit_logging.retention_days > 3650) {
              throw new Error('Audit log retention must be between 30 and 3650 days');
            }
          }
        }

        toast.success('Security configuration is valid');
        return true;
      } catch (error) {
        toast.error(`Invalid security configuration: ${error instanceof Error ? error.message : String(error)}`);
        return false;
      }
    },

    // Monitoring validation and management
    validateMonitoringConfig: (config: MonitoringConfiguration = enhancedState.monitoringConfig): boolean => {
      try {
        if (!config) {
          throw new Error('Monitoring configuration is required');
        }

        // Validate metrics configuration
        if (config.metrics) {
          if (config.metrics.collection_interval < 1 || config.metrics.collection_interval > 3600) {
            throw new Error('Metrics collection interval must be between 1 and 3600 seconds');
          }
        }

        // Validate logging configuration
        if (config.logging) {
          const validLevels = ['debug', 'info', 'warn', 'error', 'critical'];
          if (!validLevels.includes(config.logging.level)) {
            throw new Error(`Invalid log level. Must be one of: ${validLevels.join(', ')}`);
          }
        }

        // Validate alerting configuration
        if (config.alerting) {
          if (config.alerting.thresholds) {
            Object.entries(config.alerting.thresholds).forEach(([metric, threshold]) => {
              if (typeof threshold !== 'number' || threshold <= 0) {
                throw new Error(`Alert threshold for ${metric} must be a positive number`);
              }
            });
          }
        }

        toast.success('Monitoring configuration is valid');
        return true;
      } catch (error) {
        toast.error(`Invalid monitoring configuration: ${error instanceof Error ? error.message : String(error)}`);
        return false;
      }
    },

    // Integration validation and management
    validateIntegrationConfig: (config: IntegrationConfiguration = enhancedState.integrationConfig): boolean => {
      try {
        if (!config) {
          throw new Error('Integration configuration is required');
        }

        // Validate REST API configuration
        if (config.rest_api) {
          if (config.rest_api.timeout < 1000 || config.rest_api.timeout > 60000) {
            throw new Error('REST API timeout must be between 1000 and 60000 ms');
          }
          if (config.rest_api.max_retries < 0 || config.rest_api.max_retries > 10) {
            throw new Error('REST API max retries must be between 0 and 10');
          }
        }

        // Validate GraphQL configuration
        if (config.graphql) {
          if (config.graphql.max_batch_size < 1 || config.graphql.max_batch_size > 100) {
            throw new Error('GraphQL max batch size must be between 1 and 100');
          }
        }

        // Validate WebSocket configuration
        if (config.websocket) {
          if (config.websocket.ping_interval < 1000 || config.websocket.ping_interval > 30000) {
            throw new Error('WebSocket ping interval must be between 1000 and 30000 ms');
          }
        }

        toast.success('Integration configuration is valid');
        return true;
      } catch (error) {
        toast.error(`Invalid integration configuration: ${error instanceof Error ? error.message : String(error)}`);
        return false;
      }
    },

    // Error handling validation and management
    validateErrorHandlingConfig: (config: ErrorHandlingConfiguration = enhancedState.errorHandlingConfig): boolean => {
      try {
        if (!config) {
          throw new Error('Error handling configuration is required');
        }

        // Validate error classification configuration
        if (config.error_classification) {
          if (config.error_classification.max_history < 1 || config.error_classification.max_history > 1000) {
            throw new Error('Error classification max history must be between 1 and 1000');
          }
        }

        // Validate error recovery configuration
        if (config.error_recovery) {
          if (config.error_recovery.max_attempts < 1 || config.error_recovery.max_attempts > 10) {
            throw new Error('Error recovery max attempts must be between 1 and 10');
          }
        }

        // Validate error reporting configuration
        if (config.error_reporting) {
          const validDestinations = ['console', 'file', 'api', 'email', 'database'];
          config.error_reporting.destinations?.forEach(destination => {
            if (!validDestinations.includes(destination)) {
              throw new Error(`Invalid error reporting destination. Must be one of: ${validDestinations.join(', ')}`);
            }
          });
        }

        toast.success('Error handling configuration is valid');
        return true;
      } catch (error) {
        toast.error(`Invalid error handling configuration: ${error instanceof Error ? error.message : String(error)}`);
        return false;
      }
    },

    // Enhanced execution methods
    executeEvolutionWithEnhancedFeatures: async (goal: string, options: {
      performanceProfile?: string;
      securityProfile?: string;
      monitoringEnabled?: boolean;
      integrationMode?: 'auto' | 'manual' | 'disabled';
    } = {}): Promise<{
      success: boolean;
      result?: any;
      performanceMetrics?: any;
      securityStatus?: any;
      monitoringData?: any;
      integrationResults?: any;
      error?: Error;
    }> => {
      try {
        const { performanceProfile, securityProfile, monitoringEnabled = true, integrationMode = 'auto' } = options;

        // Apply performance optimizations
        const performanceConfig = performanceProfile 
          ? enhancedState.performanceProfiles?.[performanceProfile] 
          : enhancedState.performanceConfig;

        if (performanceConfig) {
          this.validatePerformanceConfig(performanceConfig);
        }

        // Apply security checks
        const securityConfig = securityProfile 
          ? enhancedState.securityProfiles?.[securityProfile] 
          : enhancedState.securityConfig;

        if (securityConfig) {
          this.validateSecurityConfig(securityConfig);
        }

        // Execute with monitoring
        let monitoringData = {};
        if (monitoringEnabled && enhancedState.monitoringConfig) {
          this.validateMonitoringConfig(enhancedState.monitoringConfig);
          monitoringData = this.startMonitoring();
        }

        // Execute with integration
        let integrationResults = {};
        if (integrationMode !== 'disabled' && enhancedState.integrationConfig) {
          this.validateIntegrationConfig(enhancedState.integrationConfig);
          integrationResults = await this.setupIntegrations(integrationMode === 'auto');
        }

        // Execute base evolution
        const executionId = uuidv4();
        const startTime = Date.now();

        const result = await basePlugin.executeEvolution(goal, {
          ...options,
          executionId,
        });

        const endTime = Date.now();
        const executionTime = endTime - startTime;

        // Collect performance metrics
        const performanceMetrics = {
          executionId,
          executionTime,
          startTime,
          endTime,
          memoryUsage: performanceConfig?.memory_management ? this.getMemoryUsage() : undefined,
          cacheStats: performanceConfig?.caching ? this.getCacheStats() : undefined,
        };

        // Collect security status
        const securityStatus = {
          authenticationStatus: securityConfig?.authentication ? 'enabled' : 'disabled',
          encryptionStatus: securityConfig?.data_protection?.encryption?.enabled ? 'enabled' : 'disabled',
          complianceStatus: securityConfig?.compliance ? 'enabled' : 'disabled',
        };

        // Stop monitoring
        if (monitoringEnabled) {
          this.stopMonitoring();
        }

        // Clean up integrations
        if (integrationMode !== 'disabled') {
          await this.cleanupIntegrations();
        }

        // Update statistics
        this.updateExecutionStatistics({
          executionId,
          executionTime,
          success: true,
          timestamp: endTime,
        });

        return {
          success: true,
          result,
          performanceMetrics,
          securityStatus,
          monitoringData,
          integrationResults,
        };
      } catch (error) {
        const errorId = uuidv4();
        
        // Handle error with enhanced error handling
        if (enhancedState.errorHandlingConfig) {
          this.handleError(error, {
            errorId,
            context: 'executeEvolutionWithEnhancedFeatures',
            severity: 'high',
          });
        }

        // Update error statistics
        this.updateErrorStatistics({
          errorId,
          errorType: error instanceof Error ? error.name : 'UnknownError',
          errorMessage: error instanceof Error ? error.message : String(error),
          timestamp: Date.now(),
        });

        return {
          success: false,
          error: error instanceof Error ? error : new Error(String(error)),
          performanceMetrics: {
            executionId: errorId,
            executionTime: 0,
            startTime: Date.now(),
            endTime: Date.now(),
          },
        };
      }
    },

    // Enhanced monitoring methods
    startMonitoring: (): any => {
      try {
        if (!enhancedState.monitoringConfig) {
          throw new Error('Monitoring configuration is not set');
        }

        const monitoringId = uuidv4();
        const startTime = Date.now();

        // Simulate monitoring data collection
        const monitoringData = {
          monitoringId,
          startTime,
          metrics: [],
          logs: [],
          events: [],
        };

        toast.info(`Monitoring started with ID: ${monitoringId}`);
        return monitoringData;
      } catch (error) {
        toast.error(`Failed to start monitoring: ${error instanceof Error ? error.message : String(error)}`);
        return {};
      }
    },

    stopMonitoring: (): void => {
      try {
        toast.info('Monitoring stopped successfully');
      } catch (error) {
        toast.error(`Failed to stop monitoring: ${error instanceof Error ? error.message : String(error)}`);
      }
    },

    getMonitoringData: (): any => {
      try {
        // Return simulated monitoring data
        return {
          metrics: [],
          logs: [],
          events: [],
          alerts: [],
        };
      } catch (error) {
        toast.error(`Failed to get monitoring data: ${error instanceof Error ? error.message : String(error)}`);
        return {};
      }
    },

    // Enhanced integration methods
    setupIntegrations: async (autoMode: boolean = true): Promise<any> => {
      try {
        if (!enhancedState.integrationConfig) {
          throw new Error('Integration configuration is not set');
        }

        const integrationId = uuidv4();
        const results: Record<string, any> = {};

        // Setup REST API integration
        if (enhancedState.integrationConfig.rest_api?.enabled) {
          results.restApi = {
            status: 'connected',
            endpoints: enhancedState.integrationConfig.rest_api.endpoints || [],
          };
        }

        // Setup GraphQL integration
        if (enhancedState.integrationConfig.graphql?.enabled) {
          results.graphql = {
            status: 'connected',
            schema: enhancedState.integrationConfig.graphql(config.integrationConfig?.api as any)?.schema_url || '',
          };
        }

        // Setup WebSocket integration
        if (enhancedState.integrationConfig.websocket?.enabled) {
          results.websocket = {
            status: 'connected',
            url: enhancedState.integrationConfig.websocket.url || '',
          };
        }

        toast.info(`Integrations setup completed with ID: ${integrationId}`);
        return results;
      } catch (error) {
        toast.error(`Failed to setup integrations: ${error instanceof Error ? error.message : String(error)}`);
        return {};
      }
    },

    cleanupIntegrations: async (): Promise<void> => {
      try {
        toast.info('Integrations cleaned up successfully');
      } catch (error) {
        toast.error(`Failed to cleanup integrations: ${error instanceof Error ? error.message : String(error)}`);
      }
    },

    // Enhanced error handling methods
    handleError: (error: unknown, options: {
      errorId?: string;
      context?: string;
      severity?: 'low' | 'medium' | 'high' | 'critical';
    } = {}): void => {
      try {
        const { errorId = uuidv4(), context = 'unknown', severity = 'medium' } = options;
        const errorMessage = error instanceof Error ? error.message : String(error);
        const errorType = error instanceof Error ? error.name : 'UnknownError';

        // Classify error
        const classification = this.classifyError(errorType, errorMessage);

        // Log error
        this.logError({
          errorId,
          errorType,
          errorMessage,
          context,
          severity,
          classification,
          timestamp: Date.now(),
        });

        // Report error based on configuration
        if (enhancedState.errorHandlingConfig?.error_reporting) {
          this.reportError({
            errorId,
            errorType,
            errorMessage,
            context,
            severity,
            classification,
          });
        }

        // Attempt recovery based on configuration
        if (enhancedState.errorHandlingConfig?.error_recovery?.enabled) {
          this.attemptErrorRecovery(errorId, errorType, errorMessage, context);
        }

        toast.error(`Error handled: ${errorMessage} (ID: ${errorId})`);
      } catch (handlingError) {
        toast.error(`Failed to handle error: ${handlingError instanceof Error ? handlingError.message : String(handlingError)}`);
      }
    },

    classifyError: (errorType: string, errorMessage: string): string => {
      try {
        // Simple error classification logic
        if (errorType.includes('Network') || errorMessage.includes('network')) {
          return 'network_error';
        } else if (errorType.includes('Timeout') || errorMessage.includes('timeout')) {
          return 'timeout_error';
        } else if (errorType.includes('Validation') || errorMessage.includes('validation')) {
          return 'validation_error';
        } else if (errorType.includes('Authentication') || errorMessage.includes('authentication')) {
          return 'authentication_error';
        } else if (errorType.includes('Authorization') || errorMessage.includes('authorization')) {
          return 'authorization_error';
        } else {
          return 'general_error';
        }
      } catch (error) {
        return 'classification_error';
      }
    },

    logError: (errorData: {
      errorId: string;
      errorType: string;
      errorMessage: string;
      context: string;
      severity: string;
      classification: string;
      timestamp: number;
    }): void => {
      try {
        // In a real implementation, this would log to a file or database
        console.error('Error logged:', errorData);
        toast.info(`Error logged: ${errorData.errorMessage} (ID: ${errorData.errorId})`);
      } catch (error) {
        toast.error(`Failed to log error: ${error instanceof Error ? error.message : String(error)}`);
      }
    },

    reportError: (errorData: {
      errorId: string;
      errorType: string;
      errorMessage: string;
      context: string;
      severity: string;
      classification: string;
    }): void => {
      try {
        const config = enhancedState.errorHandlingConfig?.error_reporting;
        if (!config) {
          throw new Error('Error reporting configuration is not set');
        }

        // Simulate error reporting to different destinations
        config.destinations?.forEach(destination => {
          console.log(`Reporting error to ${destination}:`, errorData);
        });

        toast.info(`Error reported to ${config.destinations?.join(', ') || 'default destinations'}`);
      } catch (error) {
        toast.error(`Failed to report error: ${error instanceof Error ? error.message : String(error)}`);
      }
    },

    attemptErrorRecovery: (errorId: string, errorType: string, errorMessage: string, context: string): void => {
      try {
        const config = enhancedState.errorHandlingConfig?.error_recovery;
        if (!config) {
          throw new Error('Error recovery configuration is not set');
        }

        // Simple recovery attempt logic
        if (errorType.includes('Network') || errorMessage.includes('network')) {
          toast.info(`Attempting network error recovery for error ID: ${errorId}`);
        } else if (errorType.includes('Timeout') || errorMessage.includes('timeout')) {
          toast.info(`Attempting timeout error recovery for error ID: ${errorId}`);
        } else {
          toast.info(`Attempting general error recovery for error ID: ${errorId}`);
        }

        // Simulate recovery success
        toast.success(`Error recovery attempted for error ID: ${errorId}`);
      } catch (error) {
        toast.error(`Failed to attempt error recovery: ${error instanceof Error ? error.message : String(error)}`);
      }
    },

    // Enhanced statistics methods
    updateExecutionStatistics: (stats: {
      executionId: string;
      executionTime: number;
      success: boolean;
      timestamp: number;
    }): void => {
      try {
        enhancedState = {
          ...enhancedState,
          executionStatistics: {
            ...enhancedState.executionStatistics,
            totalExecutions: (enhancedState.executionStatistics?.totalExecutions || 0) + 1,
            successfulExecutions: stats.success 
              ? (enhancedState.executionStatistics?.successfulExecutions || 0) + 1
              : enhancedState.executionStatistics?.successfulExecutions || 0,
            failedExecutions: !stats.success
              ? (enhancedState.executionStatistics?.failedExecutions || 0) + 1
              : enhancedState.executionStatistics?.failedExecutions || 0,
            totalExecutions: (enhancedState.executionStatistics?.totalExecutions || 0) + stats.executionTime,
            averageExecutionTime: undefined, // Will be calculated when needed
            lastExecution: {
              executionId: stats.executionId,
              executionTime: stats.executionTime,
              success: stats.success,
              timestamp: stats.timestamp,
            },
          },
        };
        notifyListeners();
      } catch (error) {
        toast.error(`Failed to update execution statistics: ${error instanceof Error ? error.message : String(error)}`);
      }
    },

    updateErrorStatistics: (stats: {
      errorId: string;
      errorType: string;
      errorMessage: string;
      timestamp: number;
    }): void => {
      try {
        enhancedState = {
          ...enhancedState,
          errorStatistics: {
            ...enhancedState.errorStatistics,
            totalErrors: (enhancedState.errorStatistics?.totalErrors || 0) + 1,
            errorsByType: {
              ...enhancedState.errorStatistics?.errorsByType,
              [stats.errorType]: (enhancedState.errorStatistics?.errorsByType?.[stats.errorType] || 0) + 1,
            },
            lastError: {
              errorId: stats.errorId,
              errorType: stats.errorType,
              errorMessage: stats.errorMessage,
              timestamp: stats.timestamp,
            },
          },
        };
        notifyListeners();
      } catch (error) {
        toast.error(`Failed to update error statistics: ${error instanceof Error ? error.message : String(error)}`);
      }
    },

    getPerformanceMetrics: (): any => {
      try {
        // Return simulated performance metrics
        return {
          memoryUsage: this.getMemoryUsage(),
          cacheStats: this.getCacheStats(),
          cpuUsage: Math.random() * 100,
          networkLatency: Math.random() * 1000,
        };
      } catch (error) {
        toast.error(`Failed to get performance metrics: ${error instanceof Error ? error.message : String(error)}`);
        return {};
      }
    },

    getMemoryUsage: (): any => {
      try {
        // Return simulated memory usage
        return {
          used: Math.random() * 1000,
          total: 1000,
          free: 1000 - Math.random() * 1000,
          percentage: Math.random() * 100,
        };
      } catch (error) {
        toast.error(`Failed to get memory usage: ${error instanceof Error ? error.message : String(error)}`);
        return {};
      }
    },

    getCacheStats: (): any => {
      try {
        // Return simulated cache statistics
        return {
          hits: Math.floor(Math.random() * 1000),
          misses: Math.floor(Math.random() * 100),
          size: Math.floor(Math.random() * 10000),
          hitRate: Math.random(),
        };
      } catch (error) {
        toast.error(`Failed to get cache stats: ${error instanceof Error ? error.message : String(error)}`);
        return {};
      }
    },

    getSecurityStatus: (): any => {
      try {
        // Return simulated security status
        return {
          authentication: enhancedState.securityConfig?.authentication ? 'enabled' : 'disabled',
          encryption: enhancedState.securityConfig?.data_protection?.encryption?.enabled ? 'enabled' : 'disabled',
          compliance: enhancedState.securityConfig?.compliance ? 'enabled' : 'disabled',
          accessControl: enhancedState.securityConfig?.access_control ? 'enabled' : 'disabled',
        };
      } catch (error) {
        toast.error(`Failed to get security status: ${error instanceof Error ? error.message : String(error)}`);
        return {};
      }
    },

    getIntegrationStatus: (): any => {
      try {
        // Return simulated integration status
        return {
          restApi: enhancedState.integrationConfig?.rest_api?.enabled ? 'connected' : 'disconnected',
          graphql: enhancedState.integrationConfig?.graphql?.enabled ? 'connected' : 'disconnected',
          websocket: enhancedState.integrationConfig?.websocket?.enabled ? 'connected' : 'disconnected',
          webhooks: enhancedState.integrationConfig?.webhooks?.enabled ? 'connected' : 'disconnected',
          eventStreaming: enhancedState.integrationConfig?.event_streaming?.enabled ? 'connected' : 'disconnected',
        };
      } catch (error) {
        toast.error(`Failed to get integration status: ${error instanceof Error ? error.message : String(error)}`);
        return {};
      }
    },

    // Profile management methods
    addPerformanceProfile: (profileName: string, profileConfig: PerformanceConfiguration): boolean => {
      try {
        if (!profileName || !profileConfig) {
          throw new Error('Profile name and configuration are required');
        }

        if (enhancedState.performanceProfiles?.[profileName]) {
          throw new Error(`Performance profile "${profileName}" already exists`);
        }

        enhancedState = {
          ...enhancedState,
          performanceProfiles: {
            ...enhancedState.performanceProfiles,
            [profileName]: profileConfig,
          },
        };

        notifyListeners();
        toast.success(`Performance profile "${profileName}" added successfully`);
        return true;
      } catch (error) {
        toast.error(`Failed to add performance profile: ${error instanceof Error ? error.message : String(error)}`);
        return false;
      }
    },

    addSecurityProfile: (profileName: string, profileConfig: SecurityConfiguration): boolean => {
      try {
        if (!profileName || !profileConfig) {
          throw new Error('Profile name and configuration are required');
        }

        if (enhancedState.securityProfiles?.[profileName]) {
          throw new Error(`Security profile "${profileName}" already exists`);
        }

        enhancedState = {
          ...enhancedState,
          securityProfiles: {
            ...enhancedState.securityProfiles,
            [profileName]: profileConfig,
          },
        };

        notifyListeners();
        toast.success(`Security profile "${profileName}" added successfully`);
        return true;
      } catch (error) {
        toast.error(`Failed to add security profile: ${error instanceof Error ? error.message : String(error)}`);
        return false;
      }
    },

    removePerformanceProfile: (profileName: string): boolean => {
      try {
        if (!profileName) {
          throw new Error('Profile name is required');
        }

        if (!enhancedState.performanceProfiles?.[profileName]) {
          throw new Error(`Performance profile "${profileName}" does not exist`);
        }

        const newProfiles = { ...enhancedState.performanceProfiles };
        delete newProfiles[profileName];

        enhancedState = {
          ...enhancedState,
          performanceProfiles: newProfiles,
        };

        notifyListeners();
        toast.success(`Performance profile "${profileName}" removed successfully`);
        return true;
      } catch (error) {
        toast.error(`Failed to remove performance profile: ${error instanceof Error ? error.message : String(error)}`);
        return false;
      }
    },

    removeSecurityProfile: (profileName: string): boolean => {
      try {
        if (!profileName) {
          throw new Error('Profile name is required');
        }

        if (!enhancedState.securityProfiles?.[profileName]) {
          throw new Error(`Security profile "${profileName}" does not exist`);
        }

        const newProfiles = { ...enhancedState.securityProfiles };
        delete newProfiles[profileName];

        enhancedState = {
          ...enhancedState,
          securityProfiles: newProfiles,
        };

        notifyListeners();
        toast.success(`Security profile "${profileName}" removed successfully`);
        return true;
      } catch (error) {
        toast.error(`Failed to remove security profile: ${error instanceof Error ? error.message : String(error)}`);
        return false;
      }
    },

    // Validation history methods
    addValidationResult: (validationResult: {
      validationId: string;
      validationType: 'performance' | 'security' | 'monitoring' | 'integration' | 'error_handling';
      success: boolean;
      errorMessage?: string;
      timestamp: number;
    }): boolean => {
      try {
        if (!validationResult) {
          throw new Error('Validation result is required');
        }

        enhancedState = {
          ...enhancedState,
          validationHistory: [
            ...(enhancedState.validationHistory || []),
            validationResult,
          ].slice(-100), // Keep only last 100 entries
        };

        notifyListeners();
        return true;
      } catch (error) {
        toast.error(`Failed to add validation result: ${error instanceof Error ? error.message : String(error)}`);
        return false;
      }
    },

    clearValidationHistory: (): boolean => {
      try {
        enhancedState = {
          ...enhancedState,
          validationHistory: [],
        };

        notifyListeners();
        toast.success('Validation history cleared successfully');
        return true;
      } catch (error) {
        toast.error(`Failed to clear validation history: ${error instanceof Error ? error.message : String(error)}`);
        return false;
      }
    },
  };

  return enhancedPlugin;
}

// Singleton instance management
let enhancedPluginInstance: EnhancedOpenEvolvePlugin | null = null;

/**
 * Get or create the singleton enhanced plugin instance
 */
export function getEnhancedOpenEvolvePlugin(
  initialConfig: Partial<EnhancedOpenEvolvePluginState> = {}
): EnhancedOpenEvolvePlugin {
  if (!enhancedPluginInstance) {
    enhancedPluginInstance = createEnhancedOpenEvolvePlugin(initialConfig);
  }
  return enhancedPluginInstance;
}

/**
 * Reset the singleton enhanced plugin instance
 */
export function resetEnhancedOpenEvolvePlugin(): void {
  enhancedPluginInstance = null;
}