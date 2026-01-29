import { createEnhancedOpenEvolvePlugin, getEnhancedOpenEvolvePlugin, resetEnhancedOpenEvolvePlugin } from './createEnhancedOpenEvolvePlugin';
import { createOpenEvolvePlugin } from './createOpenEvolvePlugin';
import { PerformanceBenchmark, SecurityUtils, MonitoringUtils, IntegrationUtils, ErrorAnalysisUtils, ConfigUtils } from './advancedUtilities';
import { AdvancedErrorClassifier, AdvancedErrorRecovery, AdvancedErrorReporter, ComprehensiveErrorHandler } from './enhancedErrorHandling';
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
 * Create a node from a configuration object
 *
 * @param config - Configuration object with type field
 * @returns Node instance or null
 *
 * @example
 * ```typescript
 * import { createNodeFromConfig } from '@openevolve/bubblelab-plugin/utils';
 *
 * const node = createNodeFromConfig({
 *   type: 'Decomposition',
 *   id: 'node-1',
 *   config: { strategy: 'semantic' }
 * });
 * ```
 */
export declare function createNodeFromConfig(config: any): any;
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
 * Validate node configuration
 *
 * @param type - Node type
 * @param config - Configuration to validate
 * @returns Validation result
 */
export declare function validateNodeConfig(type: string, config: Record<string, any>): any;
/**
 * Search for nodes by query
 *
 * @param query - Search query
 * @returns Array of matching nodes
 */
export declare function searchNodes(query: string): any[];
/**
 * Get nodes by category
 *
 * @param category - Category name
 * @returns Array of nodes in category
 */
export declare function getNodesByCategory(category: string): any[];
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
/**
 * Default export - All utilities
 */
declare const _default: {
    createEnhancedOpenEvolvePlugin: typeof createEnhancedOpenEvolvePlugin;
    getEnhancedOpenEvolvePlugin: typeof getEnhancedOpenEvolvePlugin;
    resetEnhancedOpenEvolvePlugin: typeof resetEnhancedOpenEvolvePlugin;
    createOpenEvolvePlugin: typeof createOpenEvolvePlugin;
    getOpenEvolvePlugin: any;
    resetOpenEvolvePlugin: any;
    createNode: typeof createNode;
    createNodeFromConfig: typeof createNodeFromConfig;
    getNodeMetadata: typeof getNodeMetadata;
    listAvailableNodes: typeof listAvailableNodes;
    validateNodeConfig: typeof validateNodeConfig;
    searchNodes: typeof searchNodes;
    getNodesByCategory: typeof getNodesByCategory;
    getNodeCategories: typeof getNodeCategories;
    createDefaultEvolutionConfig: typeof createDefaultEvolutionConfig;
    createDefaultAdversarialConfig: typeof createDefaultAdversarialConfig;
    createDefaultDecompositionConfig: typeof createDefaultDecompositionConfig;
    validateEvolutionConfig: typeof validateEvolutionConfig;
    validateAdversarialConfig: typeof validateAdversarialConfig;
    validateDecompositionConfig: typeof validateDecompositionConfig;
    createLogger: typeof createLogger;
    PerformanceBenchmark: typeof PerformanceBenchmark;
    SecurityUtils: typeof SecurityUtils;
    MonitoringUtils: typeof MonitoringUtils;
    IntegrationUtils: typeof IntegrationUtils;
    ErrorAnalysisUtils: typeof ErrorAnalysisUtils;
    ConfigUtils: typeof ConfigUtils;
    AdvancedErrorClassifier: typeof AdvancedErrorClassifier;
    AdvancedErrorRecovery: typeof AdvancedErrorRecovery;
    AdvancedErrorReporter: typeof AdvancedErrorReporter;
    ComprehensiveErrorHandler: typeof ComprehensiveErrorHandler;
};
export default _default;
