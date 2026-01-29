/**
 * Node Registry System
 *
 * Centralized registry for all OpenEvolve workflow nodes.
 * Provides type-safe node registration, creation, and metadata queries.
 *
 * @module registry
 * @version 1.0.0
 */

import { v4 as uuidv4 } from 'uuid';
import { OpenEvolveBaseNode, NodeConfig, NodeInputs, ExecutionContext } from './BaseNode';

/**
 * Node class interface for registration
 *
 * Defines the contract for node classes that can be registered.
 * All node classes must extend OpenEvolveBaseNode and provide static metadata.
 */
export interface NodeClass {
  /**
   * Create a new node instance
   *
   * @param id - Unique node identifier
   * @param config - Optional node configuration
   * @returns New node instance
   */
  new (id: string, config?: NodeConfig): OpenEvolveBaseNode;

  /**
   * Human-readable display name
   */
  DISPLAY_NAME: string;

  /**
   * Node description
   */
  DESCRIPTION: string;

  /**
   * Icon for UI display (emoji or icon name)
   */
  ICON: string;

  /**
   * Node category (e.g., 'transform', 'output', 'logic')
   */
  CATEGORY: string;

  /**
   * Node version
   */
  VERSION: string;
}

/**
 * Node metadata interface
 *
 * Extends base class metadata with additional runtime information.
 */
export interface NodeMetadata {
  /** Unique type identifier */
  type: string;

  /** Display name */
  displayName: string;

  /** Description */
  description: string;

  /** Icon */
  icon: string;

  /** Category */
  category: string;

  /** Version */
  version: string;

  /** Input port definitions */
  inputs: Array<{
    name: string;
    type: string;
    required: boolean;
    description: string;
    defaultValue?: any;
  }>;

  /** Output port definitions */
  outputs: Array<{
    name: string;
    type: string;
    description: string;
  }>;

  /** Configuration schema */
  config?: Record<string, {
    type: string;
    required: boolean;
    default?: any;
    description: string;
    enum?: any[];
    min?: number;
    max?: number;
  }>;

  /** When the node was registered */
  registeredAt?: string;

  /** Registration source (for debugging) */
  source?: string;
}

/**
 * Node configuration for creation
 */
export interface NodeCreationConfig {
  /** Unique node ID (auto-generated if not provided) */
  id?: string;

  /** Initial configuration values */
  config?: Record<string, any>;

  /** Initial input values */
  inputs?: Record<string, any>;

  /** Metadata tags */
  metadata?: Record<string, any>;
}

/**
 * Validation result for node configuration
 */
export interface ValidationResult {
  /** Whether validation passed */
  valid: boolean;

  /** Validation errors (if any) */
  errors: Array<{
    field: string;
    message: string;
    code?: string;
  }>;

  /** Validation warnings (non-critical) */
  warnings: Array<{
    field: string;
    message: string;
    code?: string;
  }>;
}

/**
 * Registration options
 */
export interface RegistrationOptions {
  /**
   * Allow overwriting existing node type
   *
   * @default false
   */
  overwrite?: boolean;

  /**
   * Registration source for debugging
   */
  source?: string;

  /**
   * Validate node class before registration
   *
   * @default true
   */
  validate?: boolean;

  /**
   * Skip duplicate check (allow multiple registrations)
   *
   * @default false
   */
  allowDuplicates?: boolean;
}

/**
 * Registry statistics
 */
export interface RegistryStats {
  /** Total number of registered nodes */
  totalNodes: number;

  /** Number of nodes per category */
  nodesByCategory: Record<string, number>;

  /** Most recently registered node type */
  lastRegistered?: string;

  /** Registry initialization time */
  initializedAt: string;
}

/**
 * Singleton Node Registry
 *
 * Manages registration and creation of all workflow nodes.
 * Uses singleton pattern to ensure a single source of truth.
 *
 * @example
 * ```typescript
 * // Register a node
 * NodeRegistry.register('MyNode', MyNodeClass);
 *
 * // Create a node instance
 * const node = NodeRegistry.create('MyNode', 'node-123', { config: { ... } });
 *
 * // List all nodes
 * const nodes = NodeRegistry.listAll();
 *
 * // Get metadata
 * const metadata = NodeRegistry.getMetadata('MyNode');
 * ```
 */
export class NodeRegistry {
  /**
   * Internal storage for registered node classes
   */
  private static nodes: Map<string, NodeClass> = new Map();

  /**
   * Metadata cache for quick lookups
   */
  private static metadataCache: Map<string, NodeMetadata> = new Map();

  /**
   * Registry initialization timestamp
   */
  private static initializedAt: string = new Date().toISOString();

  /**
   * Registration history for debugging
   */
  private static registrationHistory: Array<{
    type: string;
    timestamp: string;
    source?: string;
    action: 'register' | 'unregister';
  }> = [];

  /**
   * Private constructor to enforce singleton
   */
  private constructor() {}

  /**
   * Register a node class
   *
   * Validates and registers a node class for later instantiation.
   * Prevents duplicate registration unless options specify otherwise.
   *
   * @param type - Unique type identifier for the node
   * @param nodeClass - Node class to register
   * @param options - Registration options
   * @throws {Error} If node type already registered and overwrite is false
   * @throws {Error} If node class validation fails
   *
   * @example
   * ```typescript
   * NodeRegistry.register('Decomposition', DecompositionNode, {
   *   source: 'internal',
   *   validate: true
   * });
   * ```
   */
  static register(
    type: string,
    nodeClass: NodeClass,
    options: RegistrationOptions = {}
  ): void {
    const {
      overwrite = false,
      source,
      validate = true,
      allowDuplicates = false,
    } = options;

    // Check for existing registration
    if (this.nodes.has(type)) {
      if (!overwrite && !allowDuplicates) {
        throw new Error(
          `Node type '${type}' is already registered. ` +
          `Use overwrite option to replace it, or ensure you're not registering twice.`
        );
      }

      if (!overwrite && allowDuplicates) {
        // Silently skip duplicate registration
        return;
      }
    }

    // Validate node class if requested
    if (validate) {
      this.validateNodeClass(type, nodeClass);
    }

    // Register the node
    this.nodes.set(type, nodeClass);

    // Cache metadata
    const metadata = this.extractMetadata(type, nodeClass, source);
    this.metadataCache.set(type, metadata);

    // Record registration history
    this.registrationHistory.push({
      type,
      timestamp: new Date().toISOString(),
      source,
      action: overwrite && this.nodes.has(type) ? 'register' : 'register',
    });

    // Log registration (in debug mode)
    if (process.env.NODE_ENV === 'development') {
      console.log(`[NodeRegistry] Registered node type: ${type}`, {
        displayName: metadata.displayName,
        category: metadata.category,
        version: metadata.version,
        source,
      });
    }
  }

  /**
   * Unregister a node type
   *
   * Removes a node type from the registry.
   *
   * @param type - Node type to unregister
   * @returns True if node was unregistered, false if not found
   *
   * @example
   * ```typescript
   * NodeRegistry.unregister('ObsoleteNode');
   * ```
   */
  static unregister(type: string): boolean {
    const existed = this.nodes.delete(type);
    this.metadataCache.delete(type);

    this.registrationHistory.push({
      type,
      timestamp: new Date().toISOString(),
      action: 'unregister',
    });

    if (existed && process.env.NODE_ENV === 'development') {
      console.log(`[NodeRegistry] Unregistered node type: ${type}`);
    }

    return existed;
  }

  /**
   * Get a registered node class
   *
   * @param type - Node type to retrieve
   * @returns Node class or undefined if not found
   *
   * @example
   * ```typescript
   * const NodeClass = NodeRegistry.get('Decomposition');
   * if (NodeClass) {
   *   // Use the class
   * }
   * ```
   */
  static get(type: string): NodeClass | undefined {
    return this.nodes.get(type);
  }

  /**
   * Check if a node type is registered
   *
   * @param type - Node type to check
   * @returns True if registered
   */
  static has(type: string): boolean {
    return this.nodes.has(type);
  }

  /**
   * Create a node instance
   *
   * Instantiates a node from its registered type with optional configuration.
   *
   * @param type - Node type to create
   * @param id - Unique node identifier (auto-generated if not provided)
   * @param config - Optional node configuration
   * @returns Node instance or null if type not found
   * @throws {Error} If node creation fails
   *
   * @example
   * ```typescript
   * const node = NodeRegistry.create(
   *   'Decomposition',
   *   'decomp-123',
   *   {
   *     config: { strategy: 'semantic' },
   *     inputs: { problem: 'Solve X' }
   *   }
   * );
   * ```
   */
  static create(
    type: string,
    id?: string,
    config: NodeCreationConfig = {}
  ): OpenEvolveBaseNode | null {
    const NodeClass = this.nodes.get(type);

    if (!NodeClass) {
      console.error(`[NodeRegistry] Node type '${type}' not found`);
      return null;
    }

    try {
      // Generate ID if not provided
      const nodeId = id || uuidv4();

      // Create instance
      const node = new NodeClass(nodeId, config);

      // Log creation
      if (process.env.NODE_ENV === 'development') {
        console.log(`[NodeRegistry] Created node instance: ${type}`, {
          id: nodeId,
          config: Object.keys(config.config || {}),
        });
      }

      return node;
    } catch (error) {
      console.error(`[NodeRegistry] Failed to create node: ${type}`, {
        error: error instanceof Error ? error.message : String(error),
        id,
        config,
      });
      throw error;
    }
  }

  /**
   * List all registered nodes
   *
   * @returns Array of node types with metadata
   *
   * @example
   * ```typescript
   * const nodes = NodeRegistry.listAll();
   * nodes.forEach(node => {
   *   console.log(`${node.displayName}: ${node.description}`);
   * });
   * ```
   */
  static listAll(): Array<{ type: string; metadata: NodeMetadata }> {
    const result: Array<{ type: string; metadata: NodeMetadata }> = [];

    for (const [type, nodeClass] of this.nodes.entries()) {
      const metadata = this.metadataCache.get(type);
      if (metadata) {
        result.push({ type, metadata });
      }
    }

    // Sort by category and display name
    result.sort((a, b) => {
      if (a.metadata.category !== b.metadata.category) {
        return a.metadata.category.localeCompare(b.metadata.category);
      }
      return a.metadata.displayName.localeCompare(b.metadata.displayName);
    });

    return result;
  }

  /**
   * Get metadata for a node type
   *
   * @param type - Node type
   * @returns Node metadata or null if not found
   *
   * @example
   * ```typescript
   * const metadata = NodeRegistry.getMetadata('Decomposition');
   * console.log(metadata.displayName); // "Decomposition"
   * ```
   */
  static getMetadata(type: string): NodeMetadata | null {
    return this.metadataCache.get(type) || null;
  }

  /**
   * Get all nodes by category
   *
   * @param category - Category to filter by
   * @returns Array of node types in the category
   *
   * @example
   * ```typescript
   * const transformNodes = NodeRegistry.getByCategory('transform');
   * ```
   */
  static getByCategory(category: string): Array<{ type: string; metadata: NodeMetadata }> {
    return this.listAll().filter(node => node.metadata.category === category);
  }

  /**
   * Get all categories
   *
   * @returns Array of unique category names
   */
  static getCategories(): string[] {
    const categories = new Set<string>();
    for (const metadata of this.metadataCache.values()) {
      categories.add(metadata.category);
    }
    return Array.from(categories).sort();
  }

  /**
   * Search nodes by query
   *
   * Searches display names, descriptions, and categories.
   *
   * @param query - Search query
   * @returns Array of matching nodes
   *
   * @example
   * ```typescript
   * const results = NodeRegistry.search('decomposition');
   * ```
   */
  static search(query: string): Array<{ type: string; metadata: NodeMetadata }> {
    const lowerQuery = query.toLowerCase();

    return this.listAll().filter(node => {
      const { type, metadata } = node;
      return (
        type.toLowerCase().includes(lowerQuery) ||
        metadata.displayName.toLowerCase().includes(lowerQuery) ||
        metadata.description.toLowerCase().includes(lowerQuery) ||
        metadata.category.toLowerCase().includes(lowerQuery)
      );
    });
  }

  /**
   * Validate node configuration against schema
   *
   * @param type - Node type
   * @param config - Configuration to validate
   * @returns Validation result
   *
   * @example
   * ```typescript
   * const result = NodeRegistry.validateConfig('Decomposition', {
   *   strategy: 'semantic',
   *   maxDepth: 5
   * });
   * if (!result.valid) {
   *   console.error(result.errors);
   * }
   * ```
   */
  static validateConfig(type: string, config: Record<string, any>): ValidationResult {
    const errors: ValidationResult['errors'] = [];
    const warnings: ValidationResult['warnings'] = [];

    const metadata = this.metadataCache.get(type);
    if (!metadata) {
      errors.push({
        field: 'type',
        message: `Node type '${type}' not found in registry`,
        code: 'NODE_TYPE_NOT_FOUND',
      });
      return { valid: false, errors, warnings };
    }

    const schema = metadata.config;
    if (!schema) {
      // No schema to validate against
      return { valid: true, errors, warnings };
    }

    // Validate each config field
    for (const [field, fieldSchema] of Object.entries(schema)) {
      const value = config[field];

      // Check required fields
      if (fieldSchema.required && (value === undefined || value === null)) {
        errors.push({
          field,
          message: `Required configuration field '${field}' is missing`,
          code: 'MISSING_REQUIRED_FIELD',
        });
        continue;
      }

      // Skip type validation if value is undefined and not required
      if (value === undefined || value === null) {
        continue;
      }

      // Type validation
      const actualType = Array.isArray(value) ? 'array' : typeof value;
      const expectedType = fieldSchema.type;

      if (actualType !== expectedType) {
        errors.push({
          field,
          message: `Field '${field}' must be of type ${expectedType}, got ${actualType}`,
          code: 'TYPE_MISMATCH',
        });
        continue;
      }

      // Enum validation
      if (fieldSchema.enum && !fieldSchema.enum.includes(value)) {
        errors.push({
          field,
          message: `Field '${field}' must be one of: ${fieldSchema.enum.join(', ')}`,
          code: 'INVALID_ENUM_VALUE',
        });
      }

      // Range validation for numbers
      if (expectedType === 'number') {
        if (fieldSchema.min !== undefined && value < fieldSchema.min) {
          errors.push({
            field,
            message: `Field '${field}' must be at least ${fieldSchema.min}`,
            code: 'VALUE_TOO_SMALL',
          });
        }
        if (fieldSchema.max !== undefined && value > fieldSchema.max) {
          errors.push({
            field,
            message: `Field '${field}' must be at most ${fieldSchema.max}`,
            code: 'VALUE_TOO_LARGE',
          });
        }
      }
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
    };
  }

  /**
   * Get registry statistics
   *
   * @returns Registry statistics
   */
  static getStats(): RegistryStats {
    const nodesByCategory: Record<string, number> = {};

    for (const metadata of this.metadataCache.values()) {
      nodesByCategory[metadata.category] = (nodesByCategory[metadata.category] || 0) + 1;
    }

    const lastEntry = this.registrationHistory[this.registrationHistory.length - 1];

    return {
      totalNodes: this.nodes.size,
      nodesByCategory,
      lastRegistered: lastEntry?.type,
      initializedAt: this.initializedAt,
    };
  }

  /**
   * Clear all registered nodes
   *
   * Primarily used for testing purposes.
   */
  static clear(): void {
    this.nodes.clear();
    this.metadataCache.clear();
    this.registrationHistory = [];
    this.initializedAt = new Date().toISOString();
  }

  /**
   * Export registry state
   *
   * @returns JSON-serializable registry state
   */
  static exportState(): Record<string, any> {
    return {
      nodes: Array.from(this.nodes.keys()),
      metadata: Array.from(this.metadataCache.entries()),
      stats: this.getStats(),
      history: this.registrationHistory,
    };
  }

  // ==========================================================================
  // Private Helper Methods
  // ==========================================================================

  /**
   * Validate a node class before registration
   */
  private static validateNodeClass(type: string, nodeClass: NodeClass): void {
    // Check for required static properties
    const requiredProps = ['DISPLAY_NAME', 'DESCRIPTION', 'ICON', 'CATEGORY', 'VERSION'];
    const missingProps = requiredProps.filter(prop => !(prop in nodeClass));

    if (missingProps.length > 0) {
      throw new Error(
        `Node class '${type}' is missing required static properties: ${missingProps.join(', ')}`
      );
    }

    // Check if it's a constructor
    if (typeof nodeClass !== 'function') {
      throw new Error(`Node class '${type}' must be a constructor function`);
    }

    // Check if it extends OpenEvolveBaseNode
    // Note: This is a runtime check, compile-time check would be better
    const prototype = nodeClass.prototype;
    if (!prototype || !prototype.execute || !prototype.validateInputs) {
      throw new Error(
        `Node class '${type}' must extend OpenEvolveBaseNode and implement required methods`
      );
    }

    // Validate static property types
    if (typeof nodeClass.DISPLAY_NAME !== 'string' || !nodeClass.DISPLAY_NAME.trim()) {
      throw new Error(`Node class '${type}' must have a non-empty DISPLAY_NAME string`);
    }

    if (typeof nodeClass.VERSION !== 'string') {
      throw new Error(`Node class '${type}' must have a VERSION string`);
    }

    if (typeof nodeClass.CATEGORY !== 'string' || !nodeClass.CATEGORY.trim()) {
      throw new Error(`Node class '${type}' must have a non-empty CATEGORY string`);
    }
  }

  /**
   * Extract metadata from a node class
   */
  private static extractMetadata(
    type: string,
    nodeClass: NodeClass,
    source?: string
  ): NodeMetadata {
    return {
      type,
      displayName: nodeClass.DISPLAY_NAME,
      description: nodeClass.DESCRIPTION,
      icon: nodeClass.ICON,
      category: nodeClass.CATEGORY,
      version: nodeClass.VERSION,
      inputs: [], // Would need to be populated by node class
      outputs: [], // Would need to be populated by node class
      config: {}, // Would need to be populated by node class
      registeredAt: new Date().toISOString(),
      source,
    };
  }
}

/**
 * Convenience function to register multiple nodes at once
 *
 * @param nodes - Object mapping node types to node classes
 * @param options - Registration options to apply to all nodes
 *
 * @example
 * ```typescript
 * registerNodes({
 *   Decomposition: DecompositionNode,
 *   Solution: SolutionNode,
 *   Verification: VerificationNode,
 * }, { source: 'internal' });
 * ```
 */
export function registerNodes(
  nodes: Record<string, NodeClass>,
  options?: RegistrationOptions
): void {
  for (const [type, nodeClass] of Object.entries(nodes)) {
    try {
      NodeRegistry.register(type, nodeClass, options);
    } catch (error) {
      console.error(`Failed to register node type '${type}':`, error);
      // Continue registering other nodes
    }
  }
}

/**
 * Convenience function to create a node from configuration object
 *
 * @param config - Node configuration object with 'type' field
 * @returns Node instance or null
 *
 * @example
 * ```typescript
 * const node = createNodeFromConfig({
 *   type: 'Decomposition',
 *   id: 'decomp-123',
 *   config: { strategy: 'semantic' }
 * });
 * ```
 */
export function createNodeFromConfig(config: NodeCreationConfig & { type: string }): OpenEvolveBaseNode | null {
  const { type, id, ...rest } = config;
  return NodeRegistry.create(type, id, rest);
}
