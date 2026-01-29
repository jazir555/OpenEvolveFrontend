// @ts-nocheck
/**
 * Utils Module - Utility Functions Export
 *
 * Exports all utility functions, helpers, and factory methods
 * for the OpenEvolve BubbleLab plugin.
 *
 * @module utils
 * @version 1.0.0
 */

import {
  createEnhancedOpenEvolvePlugin,
  getEnhancedOpenEvolvePlugin,
  resetEnhancedOpenEvolvePlugin,
} from './createEnhancedOpenEvolvePlugin';

import {
  createOpenEvolvePlugin,
//   getOpenEvolvePlugin,
//   resetOpenEvolvePlugin,
} from './createOpenEvolvePlugin';

import {
  PerformanceBenchmark,
  SecurityUtils,
  MonitoringUtils,
  IntegrationUtils,
  ErrorAnalysisUtils,
  ConfigUtils,
} from './advancedUtilities';

import {
  AdvancedErrorClassifier,
  AdvancedErrorRecovery,
  AdvancedErrorReporter,
  ComprehensiveErrorHandler,
} from './enhancedErrorHandling';

import { createNodeFromConfig as _createNodeFromConfig } from '../nodes/registry';

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
export function createNodeFromConfig(config: any): any {
  // Dynamic import to avoid circular dependencies
  const { createNodeFromConfig: factory } = require('../nodes/registry');
  return factory(config);
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
 * Validate node configuration
 *
 * @param type - Node type
 * @param config - Configuration to validate
 * @returns Validation result
 */
export function validateNodeConfig(
  type: string,
  config: Record<string, any>
): any {
  const { NodeRegistry } = require('../nodes/registry');
  return NodeRegistry.validateConfig(type, config);
}

/**
 * Search for nodes by query
 *
 * @param query - Search query
 * @returns Array of matching nodes
 */
export function searchNodes(query: string): any[] {
  const { NodeRegistry } = require('../nodes/registry');
  return NodeRegistry.search(query);
}

/**
 * Get nodes by category
 *
 * @param category - Category name
 * @returns Array of nodes in category
 */
export function getNodesByCategory(category: string): any[] {
  const { NodeRegistry } = require('../nodes/registry');
  return NodeRegistry.getByCategory(category);
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
      console.warn(`[${context}] WARN:`, message, meta || '');
    },
    error: (message: string, meta?: any) => {
      console.error(`[${context}] ERROR:`, message, meta || '');
    },
  };
}

// ==========================================================================
// Default Export
// ==========================================================================

/**
 * Default export - All utilities
 */
export default {
  // Plugin creation
  createEnhancedOpenEvolvePlugin,
  getEnhancedOpenEvolvePlugin,
  resetEnhancedOpenEvolvePlugin,
  createOpenEvolvePlugin,
  getOpenEvolvePlugin,
  resetOpenEvolvePlugin,

  // Node factory
  createNode,
  createNodeFromConfig,
  getNodeMetadata,
  listAvailableNodes,
  validateNodeConfig,
  searchNodes,
  getNodesByCategory,
  getNodeCategories,

  // Configuration helpers
  createDefaultEvolutionConfig,
  createDefaultAdversarialConfig,
  createDefaultDecompositionConfig,

  // Validation
  validateEvolutionConfig,
  validateAdversarialConfig,
  validateDecompositionConfig,

  // Logging
  createLogger,

  // Advanced utilities
  PerformanceBenchmark,
  SecurityUtils,
  MonitoringUtils,
  IntegrationUtils,
  ErrorAnalysisUtils,
  ConfigUtils,

  // Error handling
  AdvancedErrorClassifier,
  AdvancedErrorRecovery,
  AdvancedErrorReporter,
  ComprehensiveErrorHandler,
};
