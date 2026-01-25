/**
 * Utils Module - Utility Functions Export
 *
 * Exports all utility functions, helpers, and factory methods
 * for the OpenEvolve BubbleLab plugin.
 *
 * @module utils
 * @version 1.0.0
 */
export { createEnhancedOpenEvolvePlugin, getEnhancedOpenEvolvePlugin, resetEnhancedOpenEvolvePlugin, } from './createEnhancedOpenEvolvePlugin';
export { createOpenEvolvePlugin, } from './createOpenEvolvePlugin';
export { PerformanceBenchmark, SecurityUtils, MonitoringUtils, IntegrationUtils, ErrorAnalysisUtils, ConfigUtils, } from './advancedUtilities';
export { AdvancedErrorClassifier, AdvancedErrorRecovery, AdvancedErrorReporter, ComprehensiveErrorHandler, } from './enhancedErrorHandling';
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
export declare function createNode(type: string, id?: string, config?: any): any;
/**
 * Get metadata for a node type
 *
 * @param type - Node type
 * @returns Node metadata or null
 */
export declare function getNodeMetadata(type: string): any;
/**
 * List all available node types
 *
 * @returns Array of node metadata
 */
export declare function listAvailableNodes(): any[];
/**
 * Get all node categories
 *
 * @returns Array of category names
 */
export declare function getNodeCategories(): string[];
/**
 * Create default evolution configuration
 *
 * @returns Default evolution config
 */
export declare function createDefaultEvolutionConfig(): any;
/**
 * Create default adversarial configuration
 *
 * @returns Default adversarial config
 */
export declare function createDefaultAdversarialConfig(): any;
/**
 * Create default decomposition configuration
 *
 * @returns Default decomposition config
 */
export declare function createDefaultDecompositionConfig(): any;
/**
 * Validate evolution configuration
 *
 * @param config - Configuration to validate
 * @returns Validation result with errors
 */
export declare function validateEvolutionConfig(config: any): {
    valid: boolean;
    errors: string[];
};
/**
 * Validate adversarial configuration
 *
 * @param config - Configuration to validate
 * @returns Validation result with errors
 */
export declare function validateAdversarialConfig(config: any): {
    valid: boolean;
    errors: string[];
};
/**
 * Validate decomposition configuration
 *
 * @param config - Configuration to validate
 * @returns Validation result with errors
 */
export declare function validateDecompositionConfig(config: any): {
    valid: boolean;
    errors: string[];
};
/**
 * Create a logger with context
 *
 * @param context - Logging context
 * @returns Logger object with debug, info, warn, error methods
 */
export declare function createLogger(context: string): {
    debug: (message: string, meta?: any) => void;
    info: (message: string, meta?: any) => void;
    warn: (message: string, meta?: any) => void;
    error: (message: string, meta?: any) => void;
};
