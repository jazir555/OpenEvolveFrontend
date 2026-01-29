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

import { ValidationError as EnhancedValidationError } from '../types/enhanced-plugin-types';
import { errorLogger } from '../utils';

/**
 * Node execution status tracking
 */
export enum NodeStatus {
  /** Node is idle and ready to execute */
  IDLE = 'idle',
  /** Node is currently executing */
  RUNNING = 'running',
  /** Node executed successfully */
  COMPLETED = 'completed',
  /** Node execution failed */
  FAILED = 'failed',
  /** Node execution was cancelled */
  CANCELLED = 'cancelled',
  /** Node is paused */
  PAUSED = 'paused',
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
 * Default node configuration
 */
const DEFAULT_NODE_CONFIG: NodeConfig = {
  debug: false,
  timeout: 300000, // 5 minutes
  retryAttempts: 0,
  retryDelay: 1000,
  enableMetrics: true,
};

/**
 * Node Execution Error Class
 *
 * Custom error class for node execution failures with detailed context
 */
export class NodeExecutionError extends Error {
  /** Name of the node that threw the error */
  public nodeName: string;

  /** Unique error code */
  public errorCode: string;

  /** Additional error details */
  public details: Record<string, any>;

  /** Timestamp when error occurred */
  public timestamp: string;

  /** Original error if this wraps another error */
  public originalError?: Error;

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
  constructor(
    nodeName: string,
    message: string,
    errorCode: string = 'NODE_ERROR',
    details: Record<string, any> = {},
    originalError?: Error,
    timestamp: string = new Date().toISOString()
  ) {
    super(`[${nodeName}] ${message}`);
    this.name = 'NodeExecutionError';
    this.nodeName = nodeName;
    this.errorCode = errorCode;
    this.details = details;
    this.originalError = originalError;
    this.timestamp = timestamp;

    // Maintain proper stack trace
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, NodeExecutionError);
    }
  }

  /**
   * Convert error to JSON-serializable object
   */
  toJSON(): Record<string, any> {
    return {
      nodeName: this.nodeName,
      errorCode: this.errorCode,
      message: this.message,
      details: this.details,
      timestamp: this.timestamp,
      stack: this.stack,
      originalError: this.originalError ? {
        message: this.originalError.message,
        stack: this.originalError.stack,
      } : undefined,
    };
  }

  /**
   * Convert error to string representation
   */
  toString(): string {
    return JSON.stringify(this.toJSON(), null, 2);
  }
}

/**
 * OpenEvolve Base Node Abstract Class
 *
 * All workflow nodes must extend this class and implement the abstract methods.
 * Provides common functionality for lifecycle management, error handling,
 * validation, and execution tracking.
 */
export abstract class OpenEvolveBaseNode {
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
  constructor(id: string, config: NodeConfig = {}) {
    try {
      this.id = id;
      this.config = { ...DEFAULT_NODE_CONFIG, ...config };
      this.status = NodeStatus.IDLE;
      this.metrics = new Map();
      this.executionAttempts = 0;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'constructor', additionalData: { id } }
      );
      // Set default values in case of constructor failure
      this.id = id || 'unknown';
      this.config = DEFAULT_NODE_CONFIG;
      this.status = NodeStatus.FAILED;
      this.metrics = new Map();
      this.executionAttempts = 0;
    }
  }

  // ==========================================================================
  // Abstract Methods - Must be implemented by subclasses
  // ==========================================================================

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

  // ==========================================================================
  // Lifecycle Hooks - Override in subclass as needed
  // ==========================================================================

  /**
   * Hook called before execution starts
   *
   * Override this to perform setup, initialization, or pre-execution
   * validation. Throwing here will prevent execution.
   *
   * @param inputs - Input data for the node
   * @param context - Execution context
   */
  protected async beforeExecute(inputs: NodeInputs, context: ExecutionContext): Promise<void> {
    try {
      if (this.config.debug) {
        context.log('debug', `[${this.getDisplayName()}] Before execute hook called`, {
          nodeId: this.id,
          inputs: Object.keys(inputs),
        });
      }
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'beforeExecute', additionalData: { nodeId: this.id } }
      );
      // Don't throw here to avoid preventing execution
    }
  }

  /**
   * Hook called after successful execution
   *
   * Override this to perform cleanup, post-processing, or result
   * enrichment after successful execution.
   *
   * @param result - Result from execute()
   * @param context - Execution context
   */
  protected async afterExecute(result: NodeResult, context: ExecutionContext): Promise<void> {
    try {
      if (this.config.debug) {
        context.log('debug', `[${this.getDisplayName()}] After execute hook called`, {
          nodeId: this.id,
          success: result.success,
          executionTime: result.metrics.executionTime,
        });
      }

      // Store result as artifact
      context.storeArtifact(`${this.id}_result`, result);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'afterExecute', additionalData: { nodeId: this.id } }
      );
      // Don't throw here to avoid interfering with execution flow
    }
  }

  /**
   * Hook called when execution encounters an error
   *
   * Override this to perform error logging, recovery attempts, or
   * cleanup after failed execution.
   *
   * @param error - The error that occurred
   * @param context - Execution context
   */
  protected async onError(error: Error, context: ExecutionContext): Promise<void> {
    try {
      if (this.config.debug) {
        context.log('error', `[${this.getDisplayName()}] Error hook called`, {
          nodeId: this.id,
          error: error.message,
          stack: error.stack,
        });
      }

      // Store error as artifact
      context.storeArtifact(`${this.id}_error`, {
        message: error.message,
        stack: error.stack,
        timestamp: new Date().toISOString(),
      });
    } catch (errorHookError) {
      errorLogger.logError(
        errorHookError instanceof Error ? errorHookError : new Error(String(errorHookError)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'onError', additionalData: { nodeId: this.id } }
      );
      // Don't throw here to avoid masking the original error
    }
  }

  // ==========================================================================
  // Public Methods
  // ==========================================================================

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
  async executeSafe(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    let startTime = Date.now();
    try {
      this.status = NodeStatus.RUNNING;

      // Check if execution is cancelled
      if (context.isCancelled()) {
        this.status = NodeStatus.CANCELLED;
        throw new NodeExecutionError(
          this.getDisplayName(),
          'Execution cancelled before start',
          'CANCELLED',
          { nodeId: this.id }
        );
      }

      // Validate inputs
      const validationErrors = this.validateInputs(inputs);
      if (validationErrors.length > 0) {
        throw new NodeExecutionError(
          this.getDisplayName(),
          `Input validation failed: ${validationErrors.map(e => e.message).join(', ')}`,
          'VALIDATION_ERROR',
          { validationErrors }
        );
      }

      // Validate config
      const configErrors = this.validateConfig();
      if (configErrors.length > 0) {
        throw new NodeExecutionError(
          this.getDisplayName(),
          `Configuration validation failed: ${configErrors.map(e => e.message).join(', ')}`,
          'CONFIG_ERROR',
          { configErrors }
        );
      }

      // Call beforeExecute hook
      await this.beforeExecute(inputs, context);

      // Execute with timeout
      const result = await this.executeWithTimeout(inputs, context);

      // Call afterExecute hook
      await this.afterExecute(result, context);

      this.status = NodeStatus.COMPLETED;
      return result;

    } catch (error) {
      this.status = NodeStatus.FAILED;
      const nodeError = error instanceof NodeExecutionError
        ? error
        : new NodeExecutionError(
            this.getDisplayName(),
            error instanceof Error ? error.message : String(error),
            'EXECUTION_ERROR',
            {},
            error instanceof Error ? error : undefined
          );

      // Call onError hook
      await this.onError(nodeError, context);

      // Return failure result
      return {
        success: false,
        outputs: {},
        metrics: {
          executionTime: Date.now() - startTime,
          startTime,
          endTime: Date.now(),
        },
        error: {
          message: nodeError.message,
          code: nodeError.errorCode,
          stack: nodeError.stack,
          context: nodeError.details,
        },
      };
    }
  }

  /**
   * Validate the node's configuration
   *
   * @returns Array of validation errors (empty if valid)
   */
  validateConfig(): ValidationError[] {
    try {
      const errors: ValidationError[] = [];

      // Validate timeout
      if (this.config.timeout !== undefined && this.config.timeout <= 0) {
        errors.push({
          field: 'timeout',
          message: 'Timeout must be greater than 0',
          code: 'INVALID_TIMEOUT',
        });
      }

      // Validate retry attempts
      if (this.config.retryAttempts !== undefined && this.config.retryAttempts < 0) {
        errors.push({
          field: 'retryAttempts',
          message: 'Retry attempts cannot be negative',
          code: 'INVALID_RETRY_ATTEMPTS',
        });
      }

      // Validate retry delay
      if (this.config.retryDelay !== undefined && this.config.retryDelay < 0) {
        errors.push({
          field: 'retryDelay',
          message: 'Retry delay cannot be negative',
          code: 'INVALID_RETRY_DELAY',
        });
      }

      return errors;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'validateConfig', additionalData: { nodeId: this.id } }
      );
      return [{
        field: 'config',
        message: 'Configuration validation failed',
        code: 'CONFIG_VALIDATION_ERROR',
      }];
    }
  }

  /**
   * Get current node status
   */
  getStatus(): NodeStatus {
    try {
      return this.status;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'getStatus', additionalData: { nodeId: this.id } }
      );
      return NodeStatus.FAILED;
    }
  }

  /**
   * Get node ID
   */
  getId(): string {
    try {
      return this.id;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'getId', additionalData: { nodeId: this.id } }
      );
      return 'unknown';
    }
  }

  /**
   * Get node configuration
   */
  getConfig(): NodeConfig {
    try {
      return { ...this.config };
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'getConfig', additionalData: { nodeId: this.id } }
      );
      return DEFAULT_NODE_CONFIG;
    }
  }

  /**
   * Update node configuration
   *
   * @param config - Partial configuration to update
   */
  updateConfig(config: Partial<NodeConfig>): void {
    try {
      this.config = { ...this.config, ...config };
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'updateConfig', additionalData: { nodeId: this.id, config } }
      );
    }
  }

  /**
   * Get execution metrics
   *
   * @returns Map of metric names to values
   */
  getMetrics(): Map<string, number> {
    try {
      return new Map(this.metrics);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'getMetrics', additionalData: { nodeId: this.id } }
      );
      return new Map();
    }
  }

  /**
   * Reset node state
   */
  reset(): void {
    try {
      this.status = NodeStatus.IDLE;
      this.metrics.clear();
      this.executionAttempts = 0;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'reset', additionalData: { nodeId: this.id } }
      );
    }
  }

  // ==========================================================================
  // Metadata Methods - Override in subclass
  // ==========================================================================

  /**
   * Get human-readable display name
   *
   * @returns Display name for the node
   */
  getDisplayName(): string {
    try {
      return this.constructor.name.replace('Node', '');
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'getDisplayName', additionalData: { nodeId: this.id } }
      );
      return 'Unknown Node';
    }
  }

  /**
   * Get node description
   *
   * @returns Description of what the node does
   */
  getDescription(): string {
    try {
      return 'An OpenEvolve workflow node';
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'getDescription', additionalData: { nodeId: this.id } }
      );
      return 'Unknown node description';
    }
  }

  /**
   * Get icon for UI display
   *
   * @returns Icon name or emoji
   */
  getIcon(): string {
    try {
      return '⚙️';
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'getIcon', additionalData: { nodeId: this.id } }
      );
      return '❓';
    }
  }

  /**
   * Get node category
   *
   * @returns Category name (e.g., 'transform', 'output', 'logic')
   */
  getCategory(): string {
    try {
      return 'general';
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'getCategory', additionalData: { nodeId: this.id } }
      );
      return 'unknown';
    }
  }

  /**
   * Get node version
   *
   * @returns Version string
   */
  getVersion(): string {
    try {
      return '1.0.0';
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'getVersion', additionalData: { nodeId: this.id } }
      );
      return 'unknown';
    }
  }

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
  } {
    try {
      return {
        type: this.constructor.name,
        displayName: this.getDisplayName(),
        description: this.getDescription(),
        icon: this.getIcon(),
        category: this.getCategory(),
        version: this.getVersion(),
        inputs: this.getParameterSchema(),
        outputs: {
          type: 'object',
          properties: {},
        },
      };
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'getMetadata', additionalData: { nodeId: this.id } }
      );
      return {
        type: 'UnknownNode',
        displayName: 'Unknown',
        description: 'Error retrieving node metadata',
        icon: '❓',
        category: 'unknown',
        version: 'unknown',
        inputs: [],
        outputs: { type: 'object', properties: {} },
      };
    }
  }

  // ==========================================================================
  // Private/Protected Helper Methods
  // ==========================================================================

  /**
   * Execute with timeout and retry logic
   *
   * @param inputs - Input data
   * @param context - Execution context
   * @returns Promise resolving to execution result
   */
  protected async executeWithTimeout(
    inputs: NodeInputs,
    context: ExecutionContext
  ): Promise<NodeResult> {
    try {
      const maxAttempts = (this.config.retryAttempts ?? 0) + 1;
      let lastError: Error | null = null;

      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        this.executionAttempts = attempt + 1;

        try {
          // Execute with timeout
          const result = await Promise.race([
            this.execute(inputs, context),
            this.createTimeoutPromise(this.config.timeout ?? DEFAULT_NODE_CONFIG.timeout),
          ]);

          return result;

        } catch (error) {
          lastError = error instanceof Error ? error : new Error(String(error));

          // If this is the last attempt, don't retry
          if (attempt < maxAttempts - 1) {
            context.log('warn', `[${this.getDisplayName()}] Execution failed, retrying...`, {
              attempt: attempt + 1,
              maxAttempts,
              error: lastError.message,
            });

            // Wait before retry
            await this.delay(this.config.retryDelay ?? DEFAULT_NODE_CONFIG.retryDelay);
          }
        }
      }

      // All retries exhausted
      throw lastError;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'executeWithTimeout', additionalData: { nodeId: this.id } }
      );
      throw error;
    }
  }

  /**
   * Create a timeout promise
   *
   * @param timeoutMs - Timeout in milliseconds
   * @returns Promise that rejects after timeout
   */
  protected createTimeoutPromise(timeoutMs: number): Promise<never> {
    try {
      return new Promise((_, reject) => {
        setTimeout(() => {
          reject(new NodeExecutionError(
            this.getDisplayName(),
            `Execution timeout after ${timeoutMs}ms`,
            'TIMEOUT',
            { timeout: timeoutMs }
          ));
        }, timeoutMs);
      });
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'createTimeoutPromise', additionalData: { nodeId: this.id, timeoutMs } }
      );
      throw error;
    }
  }

  /**
   * Delay execution for specified milliseconds
   *
   * @param ms - Milliseconds to delay
   * @returns Promise that resolves after delay
   */
  protected delay(ms: number): Promise<void> {
    try {
      return new Promise(resolve => setTimeout(resolve, ms));
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'delay', additionalData: { nodeId: this.id, ms } }
      );
      // Return a resolved promise to avoid breaking execution flow
      return Promise.resolve();
    }
  }

  /**
   * Record a custom metric
   *
   * @param name - Metric name
   * @param value - Metric value
   */
  protected recordMetric(name: string, value: number): void {
    try {
      this.metrics.set(name, value);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'recordMetric', additionalData: { nodeId: this.id, name, value } }
      );
    }
  }

  /**
   * Create a standard node result
   *
   * @param outputs - Output data
   * @param startTime - Execution start time
   * @param metadata - Optional metadata
   * @returns Node result object
   */
  protected createResult(
    outputs: Record<string, any>,
    startTime: number,
    metadata?: Record<string, any>
  ): NodeResult {
    try {
      const endTime = Date.now();

      return {
        success: true,
        outputs,
        metrics: {
          executionTime: endTime - startTime,
          startTime,
          endTime,
          customMetrics: this.config.enableMetrics ? Object.fromEntries(this.metrics) : undefined,
        },
        metadata,
      };
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'OpenEvolveBaseNode', function: 'createResult', additionalData: { nodeId: this.id } }
      );
      return {
        success: false,
        outputs: {},
        metrics: {
          executionTime: Date.now() - startTime,
          startTime,
          endTime: Date.now(),
        },
        error: {
          message: error instanceof Error ? error.message : String(error),
          code: 'CREATE_RESULT_ERROR',
        },
      };
    }
  }
}

// ============================================================================
// Example Subclass Implementation
// ============================================================================

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
