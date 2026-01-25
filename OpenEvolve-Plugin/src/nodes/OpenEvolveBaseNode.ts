/**
 * OpenEvolve Base Node
 *
 * Abstract base class for all OpenEvolve workflow nodes.
 * Provides common functionality and enforces consistent node interface.
 *
 * @module nodes
 */

export interface NodeInputs {
  [key: string]: any;
}

export interface NodeResult {
  success: boolean;
  data?: any;
  error?: string;
  metadata?: {
    executionTime: number;
    timestamp: Date;
    nodeId: string;
    [key: string]: any;
  };
}

export interface ExecutionContext {
  workflowId?: string;
  executionId?: string;
  userId?: string;
  timestamp: Date;
  environment: 'development' | 'staging' | 'production';
  [key: string]: any;
}

export interface ValidationError {
  field: string;
  message: string;
  severity: 'error' | 'warning';
}

export interface ParameterSchema {
  type: 'object';
  properties: Record<string, {
    type: string;
    description: string;
    required?: boolean;
    default?: any;
    enum?: any[];
    minimum?: number;
    maximum?: number;
    pattern?: string;
  }>;
  required?: string[];
}

export interface NodeConfig {
  [key: string]: any;
}

/**
 * Abstract base class for OpenEvolve nodes
 *
 * All workflow nodes must extend this class and implement its abstract methods.
 */
export abstract class OpenEvolveBaseNode {
  /** Unique identifier for this node instance */
  readonly id: string;

  /** Node configuration */
  protected config: NodeConfig;

  /** Execution history for this node */
  protected executionHistory: NodeResult[];

  /**
   * Creates a new OpenEvolve node instance
   *
   * @param id - Unique identifier for this node
   * @param config - Node configuration object
   */
  constructor(id: string, config: NodeConfig = {}) {
    this.id = id;
    this.config = config;
    this.executionHistory = [];
  }

  /**
   * Display name for this node type
   */
  static readonly DISPLAY_NAME: string;

  /**
   * Detailed description of what this node does
   */
  static readonly DESCRIPTION: string;

  /**
   * Icon identifier for UI rendering
   */
  static readonly ICON: string;

  /**
   * Node category for organization
   */
  static readonly CATEGORY: string;

  /**
   * Node version
   */
  static readonly VERSION: string;

  /**
   * Execute the node's primary logic
   *
   * @param inputs - Input data from previous nodes or user input
   * @param context - Execution context with metadata
   * @returns Promise resolving to execution result
   */
  abstract execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult>;

  /**
   * Validate input data before execution
   *
   * @param inputs - Input data to validate
   * @returns Array of validation errors (empty if valid)
   */
  abstract validateInputs(inputs: NodeInputs): ValidationError[];

  /**
   * Get JSON Schema for node configuration parameters
   *
   * @returns Parameter schema object
   */
  abstract getParameterSchema(): ParameterSchema;

  /**
   * Get current node configuration
   */
  getConfig(): NodeConfig {
    return { ...this.config };
  }

  /**
   * Update node configuration
   *
   * @param config - New configuration values
   */
  setConfig(config: Partial<NodeConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get execution history
   *
   * @returns Array of historical execution results
   */
  getExecutionHistory(): NodeResult[] {
    return [...this.executionHistory];
  }

  /**
   * Clear execution history
   */
  clearHistory(): void {
    this.executionHistory = [];
  }

  /**
   * Record execution result in history
   *
   * @param result - Result to record
   */
  protected recordExecution(result: NodeResult): void {
    this.executionHistory.push(result);
    // Keep only last 100 executions
    if (this.executionHistory.length > 100) {
      this.executionHistory = this.executionHistory.slice(-100);
    }
  }

  /**
   * Create a successful result object
   *
   * @param data - Result data
   * @param metadata - Optional metadata
   * @returns Success result object
   */
  protected createSuccessResult(data: any, metadata?: any): NodeResult {
    return {
      success: true,
      data,
      metadata: {
        executionTime: Date.now(),
        timestamp: new Date(),
        nodeId: this.id,
        ...metadata
      }
    };
  }

  /**
   * Create an error result object
   *
   * @param error - Error message or Error object
   * @param metadata - Optional metadata
   * @returns Error result object
   */
  protected createErrorResult(error: string | Error, metadata?: any): NodeResult {
    const errorMessage = error instanceof Error ? error.message : error;
    return {
      success: false,
      error: errorMessage,
      metadata: {
        executionTime: Date.now(),
        timestamp: new Date(),
        nodeId: this.id,
        ...metadata
      }
    };
  }

  /**
   * Execute with automatic history recording and error handling
   *
   * @param inputs - Input data
   * @param context - Execution context
   * @returns Promise resolving to execution result
   */
  async executeWithHistory(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    const startTime = Date.now();

    try {
      // Validate inputs
      const validationErrors = this.validateInputs(inputs);
      if (validationErrors.length > 0) {
        const result = this.createErrorResult(
          `Validation failed: ${validationErrors.map(e => e.message).join(', ')}`,
          { validationErrors }
        );
        this.recordExecution(result);
        return result;
      }

      // Execute node logic
      const result = await this.execute(inputs, context);

      // Add execution time if not already present
      if (result.metadata && !result.metadata.executionTime) {
        result.metadata.executionTime = Date.now() - startTime;
      }

      // Record execution
      this.recordExecution(result);

      return result;
    } catch (error) {
      const result = this.createErrorResult(
        error instanceof Error ? error : String(error),
        { executionTime: Date.now() - startTime }
      );
      this.recordExecution(result);
      return result;
    }
  }

  /**
   * Get static metadata about this node type
   *
   * @returns Object containing display name, description, icon, category, and version
   */
  static getMetadata() {
    return {
      displayName: this.DISPLAY_NAME,
      description: this.DESCRIPTION,
      icon: this.ICON,
      category: this.CATEGORY,
      version: this.VERSION
    };
  }

  /**
   * Get instance metadata about this node
   *
   * @returns Object containing type, display name, description, icon, category, version, inputs, and outputs
   */
  getMetadata(): {
    type: string;
    displayName: string;
    description: string;
    icon: string;
    category: string;
    version: string;
    inputs: any;
    outputs: any;
  } {
    const constructor = this.constructor as any;
    return {
      type: constructor.name,
      displayName: constructor.DISPLAY_NAME || 'Unknown',
      description: constructor.DESCRIPTION || '',
      icon: constructor.ICON || 'default',
      category: constructor.CATEGORY || 'general',
      version: constructor.VERSION || '1.0.0',
      inputs: this.getParameterSchema(),
      outputs: {
        type: 'object',
        properties: {},
      },
    };
  }
}
