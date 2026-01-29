import { OpenEvolveBaseNode, NodeConfig } from './BaseNode';
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
export declare class NodeRegistry {
    /**
     * Internal storage for registered node classes
     */
    private static nodes;
    /**
     * Metadata cache for quick lookups
     */
    private static metadataCache;
    /**
     * Registry initialization timestamp
     */
    private static initializedAt;
    /**
     * Registration history for debugging
     */
    private static registrationHistory;
    /**
     * Private constructor to enforce singleton
     */
    private constructor();
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
    static register(type: string, nodeClass: NodeClass, options?: RegistrationOptions): void;
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
    static unregister(type: string): boolean;
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
    static get(type: string): NodeClass | undefined;
    /**
     * Check if a node type is registered
     *
     * @param type - Node type to check
     * @returns True if registered
     */
    static has(type: string): boolean;
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
    static create(type: string, id?: string, config?: NodeCreationConfig): OpenEvolveBaseNode | null;
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
    static listAll(): Array<{
        type: string;
        metadata: NodeMetadata;
    }>;
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
    static getMetadata(type: string): NodeMetadata | null;
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
    static getByCategory(category: string): Array<{
        type: string;
        metadata: NodeMetadata;
    }>;
    /**
     * Get all categories
     *
     * @returns Array of unique category names
     */
    static getCategories(): string[];
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
    static search(query: string): Array<{
        type: string;
        metadata: NodeMetadata;
    }>;
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
    static validateConfig(type: string, config: Record<string, any>): ValidationResult;
    /**
     * Get registry statistics
     *
     * @returns Registry statistics
     */
    static getStats(): RegistryStats;
    /**
     * Clear all registered nodes
     *
     * Primarily used for testing purposes.
     */
    static clear(): void;
    /**
     * Export registry state
     *
     * @returns JSON-serializable registry state
     */
    static exportState(): Record<string, any>;
    /**
     * Validate a node class before registration
     */
    private static validateNodeClass;
    /**
     * Extract metadata from a node class
     */
    private static extractMetadata;
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
export declare function registerNodes(nodes: Record<string, NodeClass>, options?: RegistrationOptions): void;
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
export declare function createNodeFromConfig(config: NodeCreationConfig & {
    type: string;
}): OpenEvolveBaseNode | null;
