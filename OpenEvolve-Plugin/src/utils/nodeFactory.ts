// @ts-nocheck
/**
 * Node Factory Utility
 *
 * Provides convenient factory functions for creating and managing
 * OpenEvolve workflow nodes with validation and error handling.
 *
 * @module nodeFactory
 * @version 1.0.0
 */

import { v4 as uuidv4 } from 'uuid';
import {
  OpenEvolveBaseNode,
  NodeConfig,
  NodeInputs,
  ExecutionContext,
} from '../nodes/BaseNode';
import {
  NodeRegistry,
  NodeClass,
  NodeMetadata,
  NodeCreationConfig,
  ValidationResult,
} from '../nodes/registry';

/**
 * Node configuration interface for factory
 */
export interface NodeConfigInput {
  /** Node type (must be registered) */
  type: string;

  /** Unique node ID (auto-generated if not provided) */
  id?: string;

  /** Configuration values */
  config?: Record<string, any>;

  /** Initial input values */
  inputs?: Record<string, any>;

  /** Additional metadata */
  metadata?: Record<string, any>;
}

/**
 * Batch node creation result
 */
export interface BatchCreationResult {
  /** Successfully created nodes */
  created: Array<{
    id: string;
    type: string;
    node: OpenEvolveBaseNode;
  }>;

  /** Failed creations */
  failed: Array<{
    type: string;
    id: string;
    error: string;
  }>;

  /** Total count */
  total: number;

  /** Success count */
  successCount: number;

  /** Failure count */
  failureCount: number;
}

/**
 * Create a workflow node
 *
 * Factory function for creating node instances with automatic
 * ID generation, validation, and error handling.
 *
 * @param type - Node type (must be registered in NodeRegistry)
 * @param id - Unique node identifier (auto-generated if not provided)
 * @param config - Optional node configuration
 * @returns Node instance
 * @throws {Error} If node type is not registered
 * @throws {Error} If node creation fails
 *
 * @example
 * ```typescript
 * import { createNode } from '@openevolve/bubblelab-plugin/utils/nodeFactory';
 *
 * const node = createNode('Decomposition', 'my-node', {
 *   config: { strategy: 'semantic' }
 * });
 *
 * // Auto-generate ID
 * const node2 = createNode('Decomposition');
 * ```
 */
export function createNode(
  type: string,
  id?: string,
  config?: NodeConfig
): OpenEvolveBaseNode {
  // Check if node type is registered
  if (!NodeRegistry.has(type)) {
    throw new Error(
      `Node type '${type}' is not registered. ` +
      `Available types: ${NodeRegistry.listAll().map(n => n.type).join(', ')}`
    );
  }

  // Generate ID if not provided
  const nodeId = id || uuidv4();

  // Create node
  const node = NodeRegistry.create(type, nodeId, config);

  if (!node) {
    throw new Error(`Failed to create node of type '${type}'`);
  }

  return node;
}

/**
 * Create a node from configuration object
 *
 * Factory function that accepts a configuration object
 * with a 'type' field for flexible node creation.
 *
 * @param config - Configuration object with type field
 * @returns Node instance
 * @throws {Error} If config.type is not registered
 * @throws {Error} If node creation fails
 *
 * @example
 * ```typescript
 * import { createNodeFromConfig } from '@openevolve/bubblelab-plugin/utils/nodeFactory';
 *
 * const node = createNodeFromConfig({
 *   type: 'Decomposition',
 *   id: 'my-node',
 *   config: { strategy: 'semantic' },
 *   inputs: { problem: 'Solve X' }
 * });
 * ```
 */
export function createNodeFromConfig(config: NodeConfigInput): OpenEvolveBaseNode {
  const { type, id, ...rest } = config;

  if (!type) {
    throw new Error('Configuration object must include a "type" field');
  }

  return createNode(type, id, rest as NodeConfig);
}

/**
 * Get metadata for a node type
 *
 * Returns comprehensive metadata about a registered node type
 * including inputs, outputs, configuration schema, and more.
 *
 * @param type - Node type
 * @returns Node metadata or null if not found
 *
 * @example
 * ```typescript
 * import { getNodeMetadata } from '@openevolve/bubblelab-plugin/utils/nodeFactory';
 *
 * const metadata = getNodeMetadata('Decomposition');
 * if (metadata) {
 *   console.log(metadata.displayName); // "Problem Decomposition"
 *   console.log(metadata.inputs); // Array of input definitions
 *   console.log(metadata.outputs); // Array of output definitions
 * }
 * ```
 */
export function getNodeMetadata(type: string): NodeMetadata | null {
  return NodeRegistry.getMetadata(type);
}

/**
 * List all available node types
 *
 * Returns an array of all registered node types with their metadata,
 * sorted by category and display name.
 *
 * @returns Array of node metadata
 *
 * @example
 * ```typescript
 * import { listAvailableNodes } from '@openevolve/bubblelab-plugin/utils/nodeFactory';
 *
 * const nodes = listAvailableNodes();
 * nodes.forEach(({ type, metadata }) => {
 *   console.log(`${metadata.displayName} (${type})`);
 *   console.log(`  Category: ${metadata.category}`);
 *   console.log(`  Description: ${metadata.description}`);
 * });
 * ```
 */
export function listAvailableNodes(): NodeMetadata[] {
  const nodes = NodeRegistry.listAll();
  return nodes.map(n => n.metadata);
}

/**
 * Validate node configuration
 *
 * Validates a configuration object against a node type's schema.
 * Returns detailed validation errors and warnings.
 *
 * @param type - Node type
 * @param config - Configuration to validate
 * @returns Validation result with errors and warnings
 *
 * @example
 * ```typescript
 * import { validateNodeConfig } from '@openevolve/bubblelab-plugin/utils/nodeFactory';
 *
 * const result = validateNodeConfig('Decomposition', {
 *   strategy: 'invalid',
 *   maxSubProblems: -1
 * });
 *
 * if (!result.valid) {
 *   console.error('Validation errors:');
 *   result.errors.forEach(err => console.error(`  ${err.field}: ${err.message}`));
 * }
 *
 * if (result.warnings.length > 0) {
 *   console.warn('Warnings:');
 *   result.warnings.forEach(warn => console.warn(`  ${warn.field}: ${warn.message}`));
 * }
 * ```
 */
export function validateNodeConfig(
  type: string,
  config: Record<string, any>
): ValidationResult {
  return NodeRegistry.validateConfig(type, config);
}

/**
 * Create multiple nodes in batch
 *
 * Creates multiple nodes from configuration objects,
 * collecting successes and failures separately.
 *
 * @param configs - Array of node configurations
 * @returns Batch creation result with successes and failures
 *
 * @example
 * ```typescript
 * import { createNodeBatch } from '@openevolve/bubblelab-plugin/utils/nodeFactory';
 *
 * const result = createNodeBatch([
 *   { type: 'Decomposition', config: { strategy: 'semantic' } },
 *   { type: 'Solution', config: { iterations: 10 } },
 *   { type: 'Verification' }
 * ]);
 *
 * console.log(`Created ${result.successCount} nodes`);
 * console.log(`Failed ${result.failureCount} nodes`);
 *
 * result.created.forEach(({ id, type }) => {
 *   console.log(`  Created ${type}: ${id}`);
 * });
 * ```
 */
export function createNodeBatch(configs: NodeConfigInput[]): BatchCreationResult {
  const result: BatchCreationResult = {
    created: [],
    failed: [],
    total: configs.length,
    successCount: 0,
    failureCount: 0,
  };

  for (const config of configs) {
    try {
      const node = createNodeFromConfig(config);
      result.created.push({
        id: node.getId(),
        type: config.type,
        node,
      });
      result.successCount++;
    } catch (error) {
      result.failed.push({
        type: config.type,
        id: config.id || 'unknown',
        error: error instanceof Error ? error.message : String(error),
      });
      result.failureCount++;
    }
  }

  return result;
}

/**
 * Create a workflow from node definitions
 *
 * Creates a complete workflow with multiple nodes and returns
 * a map of node IDs to node instances for easy access.
 *
 * @param nodes - Array of node configurations
 * @returns Map of node ID to node instance
 * @throws {Error} If any node creation fails
 *
 * @example
 * ```typescript
 * import { createWorkflow } from '@openevolve/bubblelab-plugin/utils/nodeFactory';
 *
 * const workflow = createWorkflow([
 *   { type: 'Decomposition', id: 'decomp-1' },
 *   { type: 'Solution', id: 'solution-1' },
 *   { type: 'Verification', id: 'verify-1' }
 * ]);
 *
 * const decomposeNode = workflow.get('decomp-1');
 * const solutionNode = workflow.get('solution-1');
 * const verifyNode = workflow.get('verify-1');
 * ```
 */
export function createWorkflow(
  nodes: NodeConfigInput[]
): Map<string, OpenEvolveBaseNode> {
  const workflow = new Map<string, OpenEvolveBaseNode>();

  for (const config of nodes) {
    const node = createNodeFromConfig(config);
    workflow.set(node.getId(), node);
  }

  return workflow;
}

/**
 * Clone a node instance
 *
 * Creates a deep copy of a node with a new ID.
 *
 * @param node - Node to clone
 * @param newId - New node ID (auto-generated if not provided)
 * @returns Cloned node instance
 *
 * @example
 * ```typescript
 * import { cloneNode } from '@openevolve/bubblelab-plugin/utils/nodeFactory';
 *
 * const original = createNode('Decomposition', 'original');
 * const clone = cloneNode(original, 'clone');
 *
 * // Clone has same config but different ID
 * console.log(clone.getId()); // 'clone'
 * console.log(clone.getConfig()); // Same as original
 * ```
 */
export function cloneNode(node: OpenEvolveBaseNode, newId?: string): OpenEvolveBaseNode {
  const metadata = node.getMetadata ? node.getMetadata() : null;
  const type = metadata ? metadata.type : node.constructor.name;

  const clonedConfig = node.getConfig();
  const clonedId = newId || uuidv4();

  const clonedNode = createNode(type, clonedId, clonedConfig);

  return clonedNode;
}

/**
 * Get all nodes by category
 *
 * Returns all nodes of a specific category.
 *
 * @param category - Category name
 * @returns Array of node metadata in the category
 *
 * @example
 * ```typescript
 * import { getNodesByCategory } from '@openevolve/bubblelab-plugin/utils/nodeFactory';
 *
 * const analysisNodes = getNodesByCategory('analysis');
 * console.log(`Found ${analysisNodes.length} analysis nodes`);
 * ```
 */
export function getNodesByCategory(category: string): NodeMetadata[] {
  const nodes = NodeRegistry.getByCategory(category);
  return nodes.map(n => n.metadata);
}

/**
 * Search for nodes
 *
 * Searches for nodes by type, display name, description, or category.
 *
 * @param query - Search query
 * @returns Array of matching node metadata
 *
 * @example
 * ```typescript
 * import { searchNodes } from '@openevolve/bubblelab-plugin/utils/nodeFactory';
 *
 * const results = searchNodes('decomposition');
 * results.forEach(metadata => {
 *   console.log(`${metadata.displayName}: ${metadata.description}`);
 * });
 * ```
 */
export function searchNodes(query: string): NodeMetadata[] {
  const results = NodeRegistry.search(query);
  return results.map(r => r.metadata);
}

/**
 * Get all node categories
 *
 * Returns a sorted array of all unique category names.
 *
 * @returns Array of category names
 *
 * @example
 * ```typescript
 * import { getNodeCategories } from '@openevolve/bubblelab-plugin/utils/nodeFactory';
 *
 * const categories = getNodeCategories();
 * categories.forEach(category => {
 *   console.log(category);
 * });
 * ```
 */
export function getNodeCategories(): string[] {
  return NodeRegistry.getCategories();
}

/**
 * Validate a node instance
 *
 * Performs runtime validation of a node instance to ensure
 * it implements required methods and properties.
 *
 * @param node - Node instance to validate
 * @returns True if node is valid
 *
 * @example
 * ```typescript
 * import { isNodeValid, createNode } from '@openevolve/bubblelab-plugin/utils/nodeFactory';
 *
 * const node = createNode('Decomposition');
 * if (isNodeValid(node)) {
 *   console.log('Node is valid');
 * } else {
 *   console.error('Node is missing required methods');
 * }
 * ```
 */
export function isNodeValid(node: any): node is OpenEvolveBaseNode {
  return (
    node &&
    typeof node.execute === 'function' &&
    typeof node.validateInputs === 'function' &&
    typeof node.getParameterSchema === 'function' &&
    typeof node.getId === 'function' &&
    typeof node.getConfig === 'function'
  );
}

/**
 * Export node registry state
 *
 * Exports the current state of the node registry for debugging
 * or serialization purposes.
 *
 * @returns Registry state as JSON-serializable object
 *
 * @example
 * ```typescript
 * import { exportRegistryState } from '@openevolve/bubblelab-plugin/utils/nodeFactory';
 *
 * const state = exportRegistryState();
 * console.log(JSON.stringify(state, null, 2));
 * ```
 */
export function exportRegistryState(): Record<string, any> {
  return NodeRegistry.exportState();
}

/**
 * Get registry statistics
 *
 * Returns statistics about registered nodes.
 *
 * @returns Registry statistics
 *
 * @example
 * ```typescript
 * import { getRegistryStats } from '@openevolve/bubblelab-plugin/utils/nodeFactory';
 *
 * const stats = getRegistryStats();
 * console.log(`Total nodes: ${stats.totalNodes}`);
 * console.log(`Nodes by category:`, stats.nodesByCategory);
 * ```
 */
export function getRegistryStats(): Record<string, any> {
  return NodeRegistry.getStats();
}

/**
 * Default export - All factory functions
 */
export default {
  createNode,
  createNodeFromConfig,
  getNodeMetadata,
  listAvailableNodes,
  validateNodeConfig,
  createNodeBatch,
  createWorkflow,
  cloneNode,
  getNodesByCategory,
  searchNodes,
  getNodeCategories,
  isNodeValid,
  exportRegistryState,
  getRegistryStats,
};
