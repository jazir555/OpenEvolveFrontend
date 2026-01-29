/**
 * Utils Module - Utility Functions Export
 *
 * Exports all utility functions, helpers, and factory methods
 * for the OpenEvolve BubbleLab plugin.
 *
 * @module utils
 * @version 1.0.0
 */

// ==========================================================================
// Export Plugin Creation Utilities
// ==========================================================================

export {
  createEnhancedOpenEvolvePlugin,
  getEnhancedOpenEvolvePlugin,
  resetEnhancedOpenEvolvePlugin,
} from './createEnhancedOpenEvolvePlugin';

export {
  createOpenEvolvePlugin,
} from './createOpenEvolvePlugin';

// ==========================================================================
// Export Advanced Utilities
// ==========================================================================

export {
  PerformanceBenchmark,
  SecurityUtils,
  MonitoringUtils,
  IntegrationUtils,
  ErrorAnalysisUtils,
  ConfigUtils,
} from './advancedUtilities';

// ==========================================================================
// Export Error Handling
// ==========================================================================

export {
  AdvancedErrorClassifier,
  AdvancedErrorRecovery,
  AdvancedErrorReporter,
  ComprehensiveErrorHandler,
} from './enhancedErrorHandling';

// ==========================================================================
// Export Data Serialization Utilities
// ==========================================================================

export {
  safeStringify,
  safeParse,
  serializeWithCircularRef,
  safeClone,
  serializeToUrlParams,
  deserializeFromUrlParams,
  serializeToFormData,
  serializeForStorage,
  deserializeFromStorage,
  createSerializer,
  safeSerializer,
} from './dataSerialization';

// ==========================================================================
// Export Safe Effect Utilities
// ==========================================================================

export {
  useSafeEffect,
  useSafeLayoutEffect,
  useSafeEffectWithRetry,
  useSafeEffectOnce,
  useSafeAsyncEffect,
  useSafePromiseEffect,
} from './safeEffects';

// ==========================================================================
// Export Safe Context Utilities
// ==========================================================================

export {
  createSafeContext,
  createSafeConsumer,
  useSafeContextHook,
  SafeContextProvider,
  SafeContextConsumer,
} from './safeContext';

// ==========================================================================
// Export Safe Routing Utilities
// ==========================================================================

export {
  createSafeNavigate,
  handleRouteChange,
  withRouteGuard,
  safeRouteLoader,
  safeRouteAction,
  safeMatchRoutes,
  safeParseRouteParams,
  SafeRouteStateManager,
  safeWatchLocation,
} from './safeRouting';

// ==========================================================================
// Export Safe Asset Loading Utilities
// ==========================================================================

export {
  safeLoadImage,
  safeLoadScript,
  safeLoadStylesheet,
  SafeAssetPreloader,
  safeLoadAssetWithRetry,
  safeLoadAssetBundle,
} from './safeAssetLoading';

// ==========================================================================
// Export Safe Store Subscription Utilities
// ==========================================================================

export {
  safeSubscribe,
  safeSubscribeWithSelectorErrorHandling,
  safeSubscribeWithRetry,
  safeGetState,
  safeSetState,
  safeBatchUpdate,
  withSafeStoreMiddleware,
  SafeStoreSubscriptionManager,
} from './safeStoreSubscriptions';

// ==========================================================================
// Export Safe Event Listener Utilities
// ==========================================================================

export {
  addSafeEventListener,
  addSafeWindowEventListener,
  addSafeDocumentEventListener,
  SafeEventListenerManager,
  safeDispatchEvent,
  createSafeEventHandler,
  addSafeEventListenerWithRetry,
  addSafeEventListenerWithTimeout,
  addSafeDebouncedEventListener,
} from './safeEventListeners';

// ==========================================================================
// Export Safe Timer Utilities
// ==========================================================================

export {
  safeSetTimeout,
  safeClearTimeout,
  safeSetInterval,
  safeClearInterval,
  SafeTimerManager,
  safeSetTimeoutWithRetry,
  safeDelay,
  createCancellableTimer,
  createCancellableInterval,
  safeExecuteWithTimeout,
} from './safeTimers';

// ==========================================================================
// Export Safe Promise Chain Utilities
// ==========================================================================

export {
  safePromise,
  safeResolve,
  safeReject,
  safeChain,
  safePromiseAll,
  safePromiseAllSettled,
  safePromiseRace,
  safeAsync,
  safePromiseWithTimeout,
  safeRetry,
  safeWaterfall,
} from './safePromiseChains';

// ==========================================================================
// Export Safe Memory Management Utilities
// ==========================================================================

export {
  safeCleanup,
  SafeResourceManager,
  SafeMemoryMonitor,
  SafeObjectPool,
  SafeMemoryLeakDetector,
  safeGarbageCollect,
  safeGetMemoryUsage,
} from './safeMemoryManagement';

// ==========================================================================
// Export Error Handling
// ==========================================================================

export {
  AdvancedErrorClassifier,
  AdvancedErrorRecovery,
  AdvancedErrorReporter,
  ComprehensiveErrorHandler,
} from './enhancedErrorHandling';

export {
  gracefulErrorHandler,
  withGracefulErrorHandling,
  useGracefulErrorHandler,
} from './gracefulErrorHandler';

export {
  createErrorHandlingMiddleware,
  createApiErrorHandlingMiddleware,
  createAsyncOperationMiddleware,
  createComponentErrorHandlingMiddleware,
  createCircuitBreakerMiddleware,
  createCachingMiddleware,
  composeMiddleware,
  defaultErrorHandlingMiddleware,
  type MiddlewareFunction,
  type ErrorHandlingMiddlewareConfig
} from './errorHandlingMiddleware';

export {
  HandleError,
  HandleGetterError,
  HandleClassErrors,
  HandleAsyncError,
  HandleApiError,
  HandleFormError,
  HandleDataFetch,
  type ErrorHandlingDecoratorOptions
} from './errorHandlingDecorators';

export {
  safeAsyncOperation,
  safeSyncOperation,
  safeGet,
  safeJsonParse,
  safeJsonStringify,
  safeTimeout,
  safeInterval,
  safePromiseChain,
  safeEventHandler,
  safeEffect,
  safeReducer,
  safeContextConsumer,
  safeStorage,
  safeSetStorage,
  safeFetch,
  safeArrayOperation,
  safeMap,
  safeAsyncMap,
} from './safeOperations';

export {
  ApiErrorHandler,
  HttpClient,
  createHttpClient,
  apiClient,
  ApiInterceptor,
  withApiErrorHandling,
} from './ApiErrorHandlingMiddleware';

export {
  MockServiceWorker,
  mockServiceWorker,
  type ServiceWorkerMessageType,
  type ServiceWorkerMessage,
  type ServiceWorkerResponse
} from './mockServiceWorker';

export {
  AdvancedErrorAnalytics,
  errorAnalytics,
  useErrorAnalytics,
  trackErrorWithAnalytics,
  trackErrorResolutionTime,
  type ErrorAnalyticsConfig,
  type ErrorTrend,
  type ErrorAnalyticsSummary
} from './advancedErrorAnalytics';

// ==========================================================================
// Export Error Logging
// ==========================================================================

export {
  ErrorLogger,
  ErrorSeverity,
  ErrorContext,
  ErrorReport,
  ErrorLoggerConfig,
} from './errorLogging';

export { default as errorLogger } from './errorLogging';

// ==========================================================================
// Export Node Factory Utilities
// ==========================================================================

/**
 * Create a workflow node from configuration
 *
 * @param type - Node type (must be registered)
 * @param id - Unique node identifier
 * @param config - Node configuration
 * @returns Node instance or null
 *
 * @example
 * ```typescript
 * import { createNode } from '@openevolve/bubblelab-plugin/utils';
 *
 * const node = createNode('Decomposition', 'my-node', {
 *   config: { strategy: 'semantic' }
 * });
 * ```
 */
export function createNode(
  type: string,
  id?: string,
  config?: any
): any {
  // Dynamic import to avoid circular dependencies
  const { NodeRegistry } = require('../nodes/registry');
  return NodeRegistry.create(type, id, config);
}


/**
 * Get metadata for a node type
 *
 * @param type - Node type
 * @returns Node metadata or null
 */
export function getNodeMetadata(type: string): any {
  const { NodeRegistry } = require('../nodes/registry');
  return NodeRegistry.getMetadata(type);
}

/**
 * List all available node types
 *
 * @returns Array of node metadata
 */
export function listAvailableNodes(): any[] {
  const { NodeRegistry } = require('../nodes/registry');
  return NodeRegistry.listAll();
}




/**
 * Get all node categories
 *
 * @returns Array of category names
 */
export function getNodeCategories(): string[] {
  const { NodeRegistry } = require('../nodes/registry');
  return NodeRegistry.getCategories();
}

// ==========================================================================
// Export Configuration Helpers
// ==========================================================================

/**
 * Create default evolution configuration
 *
 * @returns Default evolution config
 */
export function createDefaultEvolutionConfig(): any {
  return {
    generations: 10,
    populationSize: 50,
    mutationRate: 0.1,
    crossoverRate: 0.8,
    selectionMethod: 'tournament',
    elitismCount: 2,
    tournamentSize: 3,
    mutationStrategy: 'gaussian',
    crossoverStrategy: 'uniform',
  };
}

/**
 * Create default adversarial configuration
 *
 * @returns Default adversarial config
 */
export function createDefaultAdversarialConfig(): any {
  return {
    enabled: true,
    attackStrategy: 'fgsm',
    numExamples: 10,
    strength: 0.3,
    stepSize: 0.1,
    numSteps: 10,
    norm: 'Linf',
    defenseStrategies: [],
  };
}

/**
 * Create default decomposition configuration
 *
 * @returns Default decomposition config
 */
export function createDefaultDecompositionConfig(): any {
  return {
    strategy: 'hierarchical',
    maxDepth: 3,
    recursionDepthLimit: 1,
    pruningThreshold: 0.5,
    granularity: 'medium',
    parallelDecomposition: false,
    maxSubtasks: 10,
    maxSubProblems: 3,
    dependencyAnalysis: true,
    constraintPropagation: true,
  };
}

// ==========================================================================
// Export Validation Utilities
// ==========================================================================

/**
 * Validate evolution configuration
 *
 * @param config - Configuration to validate
 * @returns Validation result with errors
 */
export function validateEvolutionConfig(config: any): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  if (!config.generations || config.generations < 1) {
    errors.push('Generations must be at least 1');
  }

  if (!config.populationSize || config.populationSize < 2) {
    errors.push('Population size must be at least 2');
  }

  if (config.mutationRate < 0 || config.mutationRate > 1) {
    errors.push('Mutation rate must be between 0 and 1');
  }

  if (config.crossoverRate < 0 || config.crossoverRate > 1) {
    errors.push('Crossover rate must be between 0 and 1');
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Validate adversarial configuration
 *
 * @param config - Configuration to validate
 * @returns Validation result with errors
 */
export function validateAdversarialConfig(config: any): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  if (!config.attackStrategy) {
    errors.push('Attack strategy is required');
  }

  if (!config.numExamples || config.numExamples < 1) {
    errors.push('Number of examples must be at least 1');
  }

  if (config.strength < 0 || config.strength > 1) {
    errors.push('Strength must be between 0 and 1');
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Validate decomposition configuration
 *
 * @param config - Configuration to validate
 * @returns Validation result with errors
 */
export function validateDecompositionConfig(config: any): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  if (!config.strategy) {
    errors.push('Decomposition strategy is required');
  }

  if (!config.maxDepth || config.maxDepth < 1) {
    errors.push('Max depth must be at least 1');
  }
  if (config.recursionDepthLimit !== undefined && config.recursionDepthLimit < 0) {
    errors.push('Recursion depth limit must be 0 (unlimited) or greater');
  }
  if (config.maxSubProblems !== undefined && config.maxSubProblems < 0) {
    errors.push('Max sub-problems must be 0 (unlimited) or greater');
  }

  if (config.pruningThreshold < 0 || config.pruningThreshold > 1) {
    errors.push('Pruning threshold must be between 0 and 1');
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

// ==========================================================================
// Export Logging Utilities
// ==========================================================================

/**
 * Create a logger with context
 *
 * @param context - Logging context
 * @returns Logger object with debug, info, warn, error methods
 */
export function createLogger(context: string): {
  debug: (message: string, meta?: any) => void;
  info: (message: string, meta?: any) => void;
  warn: (message: string, meta?: any) => void;
  error: (message: string, meta?: any) => void;
} {
  return {
    debug: (message: string, meta?: any) => {
      if (process.env.NODE_ENV === 'development') {
        console.debug(`[${context}] DEBUG:`, message, meta || '');
      }
    },
    info: (message: string, meta?: any) => {
      console.info(`[${context}] INFO:`, message, meta || '');
    },
    warn: (message: string, meta?: any) => {
      // Use error logger for warnings
      try {
        import('./errorLogging').then(({ default: errorLogger }) => {
          errorLogger.logError(message, 'warn', {
            component: context,
            additionalData: meta
          });
        });
      } catch (e) {
        console.warn(`[${context}] WARN:`, message, meta || '');
      }
    },
    error: (message: string, meta?: any) => {
      // Use error logger for errors
      try {
        import('./errorLogging').then(({ default: errorLogger }) => {
          errorLogger.logError(message, 'error', {
            component: context,
            additionalData: meta
          });
        });
      } catch (e) {
        console.error(`[${context}] ERROR:`, message, meta || '');
      }
    },
  };
}

// ==========================================================================
// Note: All utilities are exported as named exports above
// ==========================================================================
