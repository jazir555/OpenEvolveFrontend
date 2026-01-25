/**
 * OpenEvolve Base Node Implementation
 *
 * This file provides the abstract base class for all OpenEvolve workflow nodes.
 * All specific node implementations should extend this class and implement
 * the required abstract methods.
 *
 * @module BaseNode
 * @version 1.0.0
 */
/**
 * Node execution status tracking
 */
export declare enum NodeStatus {
    /** Node is idle and ready to execute */
    IDLE = "idle",
    /** Node is currently executing */
    RUNNING = "running",
    /** Node executed successfully */
    COMPLETED = "completed",
    /** Node execution failed */
    FAILED = "failed",
    /** Node execution was cancelled */
    CANCELLED = "cancelled",
    /** Node is paused */
    PAUSED = "paused"
}
/**
 * Node configuration interface
 */
export interface NodeConfig {
    /** Enable debug logging for this node */
    debug?: boolean;
    /** Maximum execution time in milliseconds */
    timeout?: number;
    /** Number of retry attempts on failure */
    retryAttempts?: number;
    /** Delay between retries in milliseconds */
    retryDelay?: number;
    /** Enable performance metrics collection */
    enableMetrics?: boolean;
    /** Custom node-specific configuration */
    [key: string]: any;
}
/**
 * Node inputs interface
 */
export interface NodeInputs {
    /** Input data for the node */
    [key: string]: any;
}
/**
 * Node execution result interface
 */
export interface NodeResult {
    /** Success status */
    success: boolean;
    /** Output data from the node */
    outputs: Record<string, any>;
    /** Execution metrics */
    metrics: NodeMetrics;
    /** Error information if execution failed */
    error?: ErrorDetails;
    /** Additional metadata */
    metadata?: Record<string, any>;
}
/**
 * Error details interface
 */
export interface ErrorDetails {
    /** Error message */
    message: string;
    /** Error code */
    code?: string;
    /** Stack trace */
    stack?: string;
    /** Additional error context */
    context?: Record<string, any>;
}
/**
 * Execution context interface
 */
export interface ExecutionContext {
    /** Unique execution ID */
    executionId: string;
    /** Workflow ID */
    workflowId: string;
    /** Current step number */
    stepNumber: number;
    /** Total steps in workflow */
    totalSteps: number;
    /** Start timestamp of workflow execution */
    workflowStartTime: number;
    /** Logging function */
    log: (level: 'debug' | 'info' | 'warn' | 'error', message: string, meta?: any) => void;
    /** Store artifacts from execution */
    storeArtifact: (artifactId: string, data: any) => void;
    /** Retrieve artifacts */
    getArtifact: (artifactId: string) => any;
    /** Get shared context data */
    getContext: (key: string) => any;
    /** Set shared context data */
    setContext: (key: string, value: any) => void;
    /** Check if execution should continue */
    isCancelled: () => boolean;
    /** Additional context data */
    [key: string]: any;
}
/**
 * Parameter schema interface
 */
export interface ParameterSchema {
    /** Parameter name */
    name: string;
    /** Parameter type */
    type: 'string' | 'number' | 'boolean' | 'array' | 'object' | 'enum';
    /** Whether parameter is required */
    required: boolean;
    /** Default value */
    default?: any;
    /** Description of parameter */
    description?: string;
    /** Valid values for enum type */
    enumValues?: any[];
    /** Minimum value for numbers */
    min?: number;
    /** Maximum value for numbers */
    max?: number;
    /** Validation regex for strings */
    pattern?: string;
    /** Nested schema for objects/arrays */
    schema?: ParameterSchema[];
}
/**
 * Validation error interface
 */
export interface ValidationError {
    /** Field that failed validation */
    field: string;
    /** Error message */
    message: string;
    /** Error code */
    code?: string;
}
/**
 * Node execution metrics
 */
export interface NodeMetrics {
    /** Execution time in milliseconds */
    executionTime: number;
    /** Memory used in bytes (if available) */
    memoryUsed?: number;
    /** Custom metrics specific to the node */
    customMetrics?: Record<string, number>;
    /** Start timestamp */
    startTime: number;
    /** End timestamp */
    endTime: number;
}
/**
 * Node Execution Error Class
 *
 * Custom error class for node execution failures with detailed context
 */
export declare class NodeExecutionError extends Error {
    /** Name of the node that threw the error */
    nodeName: string;
    /** Unique error code */
    errorCode: string;
    /** Additional error details */
    details: Record<string, any>;
    /** Timestamp when error occurred */
    timestamp: string;
    /** Original error if this wraps another error */
    originalError?: Error;
    /**
     * Create a new NodeExecutionError
     *
     * @param nodeName - Name of the node
     * @param message - Error message
     * @param errorCode - Error code (e.g., 'VALIDATION_ERROR', 'TIMEOUT')
     * @param details - Additional error context
     * @param originalError - Original error if wrapping
     * @param timestamp - Error timestamp (defaults to now)
     */
    constructor(nodeName: string, message: string, errorCode?: string, details?: Record<string, any>, originalError?: Error, timestamp?: string);
    /**
     * Convert error to JSON-serializable object
     */
    toJSON(): Record<string, any>;
    /**
     * Convert error to string representation
     */
    toString(): string;
}
/**
 * OpenEvolve Base Node Abstract Class
 *
 * All workflow nodes must extend this class and implement the abstract methods.
 * Provides common functionality for lifecycle management, error handling,
 * validation, and execution tracking.
 */
export declare abstract class OpenEvolveBaseNode {
    /** Unique node identifier */
    protected id: string;
    /** Node configuration */
    protected config: NodeConfig;
    /** Current execution status */
    protected status: NodeStatus;
    /** Execution metrics storage */
    protected metrics: Map<string, number>;
    /** Number of execution attempts */
    protected executionAttempts: number;
    /**
     * Create a new base node instance
     *
     * @param id - Unique node identifier
     * @param config - Node configuration (merged with defaults)
     */
    constructor(id: string, config?: NodeConfig);
    /**
     * Execute the node's primary logic
     *
     * This is the main method that subclasses must implement to define
     * the node's behavior. It receives inputs and context, and should
     * return outputs and metrics.
     *
     * @param inputs - Input data for the node
     * @param context - Execution context with logging and utilities
     * @returns Promise resolving to node execution result
     */
    abstract execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult>;
    /**
     * Validate input data before execution
     *
     * Subclasses implement this to validate that inputs meet requirements
     * before execution begins. Return empty array if validation passes.
     *
     * @param inputs - Input data to validate
     * @returns Array of validation errors (empty if valid)
     */
    abstract validateInputs(inputs: NodeInputs): ValidationError[];
    /**
     * Get the parameter schema for this node
     *
     * Returns the schema defining what parameters this node accepts,
     * their types, requirements, and constraints.
     *
     * @returns Parameter schema definition
     */
    abstract getParameterSchema(): ParameterSchema[];
    /**
     * Hook called before execution starts
     *
     * Override this to perform setup, initialization, or pre-execution
     * validation. Throwing here will prevent execution.
     *
     * @param inputs - Input data for the node
     * @param context - Execution context
     */
    protected beforeExecute(inputs: NodeInputs, context: ExecutionContext): Promise<void>;
    /**
     * Hook called after successful execution
     *
     * Override this to perform cleanup, post-processing, or result
     * enrichment after successful execution.
     *
     * @param result - Result from execute()
     * @param context - Execution context
     */
    protected afterExecute(result: NodeResult, context: ExecutionContext): Promise<void>;
    /**
     * Hook called when execution encounters an error
     *
     * Override this to perform error logging, recovery attempts, or
     * cleanup after failed execution.
     *
     * @param error - The error that occurred
     * @param context - Execution context
     */
    protected onError(error: Error, context: ExecutionContext): Promise<void>;
    /**
     * Execute the node with comprehensive error handling
     *
     * Wraps execute() with try-catch, lifecycle hooks, validation,
     * timeout handling, and retry logic.
     *
     * @param inputs - Input data for the node
     * @param context - Execution context
     * @returns Promise resolving to node execution result
     */
    executeSafe(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult>;
    /**
     * Validate the node's configuration
     *
     * @returns Array of validation errors (empty if valid)
     */
    validateConfig(): ValidationError[];
    /**
     * Get current node status
     */
    getStatus(): NodeStatus;
    /**
     * Get node ID
     */
    getId(): string;
    /**
     * Get node configuration
     */
    getConfig(): NodeConfig;
    /**
     * Update node configuration
     *
     * @param config - Partial configuration to update
     */
    updateConfig(config: Partial<NodeConfig>): void;
    /**
     * Get execution metrics
     *
     * @returns Map of metric names to values
     */
    getMetrics(): Map<string, number>;
    /**
     * Reset node state
     */
    reset(): void;
    /**
     * Get human-readable display name
     *
     * @returns Display name for the node
     */
    getDisplayName(): string;
    /**
     * Get node description
     *
     * @returns Description of what the node does
     */
    getDescription(): string;
    /**
     * Get icon for UI display
     *
     * @returns Icon name or emoji
     */
    getIcon(): string;
    /**
     * Get node category
     *
     * @returns Category name (e.g., 'transform', 'output', 'logic')
     */
    getCategory(): string;
    /**
     * Get node version
     *
     * @returns Version string
     */
    getVersion(): string;
    /**
     * Get node metadata
     *
     * Returns comprehensive metadata about this node instance, including
     * type information, display properties, and input/output schemas.
     *
     * @returns Node metadata object
     */
    getMetadata(): {
        type: string;
        displayName: string;
        description: string;
        icon: string;
        category: string;
        version: string;
        inputs: ParameterSchema[];
        outputs: any;
    };
    /**
     * Execute with timeout and retry logic
     *
     * @param inputs - Input data
     * @param context - Execution context
     * @returns Promise resolving to execution result
     */
    protected executeWithTimeout(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult>;
    /**
     * Create a timeout promise
     *
     * @param timeoutMs - Timeout in milliseconds
     * @returns Promise that rejects after timeout
     */
    protected createTimeoutPromise(timeoutMs: number): Promise<never>;
    /**
     * Delay execution for specified milliseconds
     *
     * @param ms - Milliseconds to delay
     * @returns Promise that resolves after delay
     */
    protected delay(ms: number): Promise<void>;
    /**
     * Record a custom metric
     *
     * @param name - Metric name
     * @param value - Metric value
     */
    protected recordMetric(name: string, value: number): void;
    /**
     * Create a standard node result
     *
     * @param outputs - Output data
     * @param startTime - Execution start time
     * @param metadata - Optional metadata
     * @returns Node result object
     */
    protected createResult(outputs: Record<string, any>, startTime: number, metadata?: Record<string, any>): NodeResult;
}
/**
 * Example: Transform Node Implementation
 *
 * This example shows how to extend OpenEvolveBaseNode to create
 * a specific node implementation.
 *
 * @example
 * ```typescript
 * export class TransformNode extends OpenEvolveBaseNode {
 *   constructor(id: string, config: NodeConfig = {}) {
 *     super(id, config);
 *   }
 *
 *   async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
 *     const startTime = Date.now();
 *
 *     // Get transformation function from config
 *     const transformFn = this.config.transform;
 *
 *     // Apply transformation
 *     const outputs = {
 *       result: transformFn(inputs.data),
 *     };
 *
 *     return this.createResult(outputs, startTime);
 *   }
 *
 *   validateInputs(inputs: NodeInputs): ValidationError[] {
 *     const errors: ValidationError[] = [];
 *
 *     if (!inputs.data) {
 *       errors.push({
 *         field: 'data',
 *         message: 'Input data is required',
 *         code: 'MISSING_DATA',
 *       });
 *     }
 *
 *     return errors;
 *   }
 *
 *   getParameterSchema(): ParameterSchema[] {
 *     return [
 *       {
 *         name: 'data',
 *         type: 'any',
 *         required: true,
 *         description: 'Data to transform',
 *       },
 *     ];
 *   }
 *
 *   getDisplayName(): string {
 *     return 'Transform';
 *   }
 *
 *   getDescription(): string {
 *     return 'Transform input data using a configured function';
 *   }
 *
 *   getIcon(): string {
 *     return '🔄';
 *   }
 *
 *   getCategory(): string {
 *     return 'transform';
 *   }
 * }
 * ```
 */
