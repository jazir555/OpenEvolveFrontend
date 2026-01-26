/**
 * Hephaestus Node
 *
 * Code generation bridge node for integration with Hephaestus.
 * Facilitates cross-service code generation and execution delegation.
 *
 * @module nodes
 */

import {
  OpenEvolveBaseNode,
  NodeInputs,
  NodeResult,
  ExecutionContext,
  ValidationError,
  ParameterSchema
} from './OpenEvolveBaseNode';
import { apiClient } from '@/services/api';

/**
 * Hephaestus task types
 */
export type HephaestusTaskType = 'generate' | 'execute' | 'delegate' | 'optimize';

/**
 * Code language types
 */
export type CodeLanguage = 'python' | 'javascript' | 'typescript' | 'java' | 'cpp' | 'go' | 'rust';

/**
 * Hephaestus node configuration
 */
export interface HephaestusNodeConfig {
  taskType?: HephaestusTaskType;
  language?: CodeLanguage;
  enableExecution?: boolean;
  enableOptimization?: boolean;
  timeoutMs?: number;
}

/**
 * Code generation result
 */
export interface CodeGenerationResult {
  code: string;
  language: CodeLanguage;
  quality: number;
  dependencies: string[];
  documentation: string;
  tests?: string;
}

/**
 * Code execution result
 */
export interface CodeExecutionResult {
  success: boolean;
  output: string;
  error?: string;
  executionTime: number;
  memory: number;
  cpu: number;
}

/**
 * Delegation result
 */
export interface DelegationResult {
  delegatedTo: string;
  taskId: string;
  status: string;
  result?: any;
  metadata: {
    delegatedAt: Date;
    completedAt?: Date;
    executionTime: number;
  };
}

/**
 * Optimization result
 */
export interface OptimizationResult {
  originalCode: string;
  optimizedCode: string;
  improvements: Array<{
    type: string;
    description: string;
    impact: string;
  }>;
  performance: {
    speedup: number;
    memoryReduction: number;
    timeComplexity: string;
    spaceComplexity: string;
  };
}

/**
 * Hephaestus result
 */
export interface HephaestusResult {
  taskId: string;
  taskType: HephaestusTaskType;
  language: CodeLanguage;
  input: string;
  output: CodeGenerationResult | CodeExecutionResult | DelegationResult | OptimizationResult;
  metadata: {
    executedAt: Date;
    executionTime: number;
    parameters: {
      language: CodeLanguage;
      enableExecution: boolean;
      enableOptimization: boolean;
    };
  };
}

/**
 * Hephaestus Node
 *
 * Bridges to Hephaestus for code generation and execution.
 * Supports delegation, optimization, and cross-service integration.
 */
export class HephaestusNode extends OpenEvolveBaseNode {
  static readonly DISPLAY_NAME = 'Hephaestus Bridge';
  static readonly DESCRIPTION = 'Code generation and execution bridge with delegation and optimization capabilities';
  static readonly ICON = 'hephaestus';
  static readonly CATEGORY = 'integration';
  static readonly VERSION = '1.0.0';

  constructor(id: string, config: HephaestusNodeConfig = {}) {
    super(id, {
      taskType: 'generate',
      language: 'python',
      enableExecution: false,
      enableOptimization: false,
      timeoutMs: 60000, // 1 minute
      ...config
    });
  }

  /**
   * Execute Hephaestus task
   *
   * @param inputs - Must contain 'input' and optionally 'taskType'
   * @param context - Execution context
   * @returns Promise resolving to Hephaestus result
   */
  async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    try {
      const startTime = Date.now();

      // Extract inputs
      const input = inputs.input as string;
      const taskType = (inputs.taskType as HephaestusTaskType) || (this.config.taskType as HephaestusTaskType);
      const language = (inputs.language as CodeLanguage) || (this.config.language as CodeLanguage);
      const code = inputs.code as string | undefined;
      const delegateTo = inputs.delegateTo as string | undefined;

      // Validate required inputs
      if (!input || input.trim().length === 0) {
        return this.createErrorResult('Input is required and cannot be empty');
      }

      context.updateProgress(10, `Preparing ${taskType} task`);

      let result: CodeGenerationResult | CodeExecutionResult | DelegationResult | OptimizationResult;

      // Execute based on task type
      switch (taskType) {
        case 'generate':
          result = await this.generateCode(input, language, context);
          break;

        case 'execute':
          if (!code) {
            return this.createErrorResult('Code is required for execution');
          }
          result = await this.executeCode(code, language, context);
          break;

        case 'delegate':
          if (!delegateTo) {
            return this.createErrorResult('Delegate target is required for delegation');
          }
          result = await this.delegateTask(input, delegateTo, context);
          break;

        case 'optimize':
          if (!code) {
            return this.createErrorResult('Code is required for optimization');
          }
          result = await this.optimizeCode(code, language, context);
          break;

        default:
          return this.createErrorResult(`Unknown task type: ${taskType}`);
      }

      const executionTime = Date.now() - startTime;

      const hephaestusResult: HephaestusResult = {
        taskId: `task-${Date.now()}`,
        taskType,
        language,
        input,
        output: result,
        metadata: {
          executedAt: new Date(),
          executionTime,
          parameters: {
            language,
            enableExecution: this.config.enableExecution as boolean,
            enableOptimization: this.config.enableOptimization as boolean
          }
        }
      };

      context.updateProgress(100, `${taskType} task complete`);

      return this.createSuccessResult(hephaestusResult);

    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Unknown error during Hephaestus execution'
      );
    }
  }

  /**
   * Generate code from natural language
   *
   * @param input - Natural language description
   * @param language - Target programming language
   * @param context - Execution context
   * @returns Promise resolving to generated code
   */
  private async generateCode(
    input: string,
    language: CodeLanguage,
    context: ExecutionContext
  ): Promise<CodeGenerationResult> {
    context.updateProgress(30, `Generating ${language} code`);

    const response = await apiClient.post<any>('/hephaestus/generate', {
      description: input,
      language,
      include_tests: true,
      include_docs: true
    });

    context.updateProgress(80, 'Code generated, post-processing');

    return {
      code: response.code || '',
      language,
      quality: response.quality || 0.8,
      dependencies: response.dependencies || [],
      documentation: response.documentation || '',
      tests: response.tests
    };
  }

  /**
   * Execute code
   *
   * @param code - Code to execute
   * @param language - Programming language
   * @param context - Execution context
   * @returns Promise resolving to execution result
   */
  private async executeCode(
    code: string,
    language: CodeLanguage,
    context: ExecutionContext
  ): Promise<CodeExecutionResult> {
    context.updateProgress(30, `Executing ${language} code`);

    const response = await apiClient.post<any>('/hephaestus/execute', {
      code,
      language,
      timeout: this.config.timeoutMs
    });

    context.updateProgress(100, 'Execution complete');

    return {
      success: response.success || false,
      output: response.output || '',
      error: response.error,
      executionTime: response.execution_time || 0,
      memory: response.memory || 0,
      cpu: response.cpu || 0
    };
  }

  /**
   * Delegate task to another service
   *
   * @param input - Task input
   * @param delegateTo - Target service
   * @param context - Execution context
   * @returns Promise resolving to delegation result
   */
  private async delegateTask(
    input: string,
    delegateTo: string,
    context: ExecutionContext
  ): Promise<DelegationResult> {
    context.updateProgress(30, `Delegating to ${delegateTo}`);

    const response = await apiClient.post<any>('/hephaestus/delegate', {
      task: input,
      target_service: delegateTo
    });

    const taskId = response.task_id || `delegate-${Date.now()}`;

    // Monitor delegation progress
    const result = await this.monitorDelegation(taskId, context);

    return {
      delegatedTo: delegateTo,
      taskId,
      status: result.status,
      result: result.output,
      metadata: {
        delegatedAt: new Date(result.created_at || Date.now()),
        completedAt: result.completed_at ? new Date(result.completed_at) : undefined,
        executionTime: result.execution_time || 0
      }
    };
  }

  /**
   * Optimize code
   *
   * @param code - Code to optimize
   * @param language - Programming language
   * @param context - Execution context
   * @returns Promise resolving to optimization result
   */
  private async optimizeCode(
    code: string,
    language: CodeLanguage,
    context: ExecutionContext
  ): Promise<OptimizationResult> {
    context.updateProgress(30, `Optimizing ${language} code`);

    const response = await apiClient.post<any>('/hephaestus/optimize', {
      code,
      language
    });

    context.updateProgress(100, 'Optimization complete');

    return {
      originalCode: code,
      optimizedCode: response.optimized_code || code,
      improvements: response.improvements || [],
      performance: {
        speedup: response.performance?.speedup || 1.0,
        memoryReduction: response.performance?.memory_reduction || 0,
        timeComplexity: response.performance?.time_complexity || 'Unknown',
        spaceComplexity: response.performance?.space_complexity || 'Unknown'
      }
    };
  }

  /**
   * Monitor delegation progress
   *
   * @param taskId - Task ID to monitor
   * @param context - Execution context
   * @returns Promise resolving to delegation status
   */
  private async monitorDelegation(taskId: string, context: ExecutionContext): Promise<any> {
    const maxAttempts = 60; // 5 minutes with 5 second intervals
    let attempts = 0;
    const startTime = Date.now();
    const timeoutMs = this.config.timeoutMs as number;

    while (attempts < maxAttempts) {
      // Check timeout
      if (Date.now() - startTime > timeoutMs) {
        throw new Error('Delegation monitoring timeout exceeded');
      }

      try {
        const status = await apiClient.get<any>(`/hephaestus/delegation/${taskId}`);

        const progress = Math.min(30 + (attempts / maxAttempts) * 70, 95);
        context.updateProgress(progress, `Delegation status: ${status.status}`);

        // Check if delegation is complete
        if (status.status === 'completed' || status.status === 'failed') {
          return status;
        }

        // Wait before next poll
        await new Promise(resolve => setTimeout(resolve, 5000));
        attempts++;

      } catch (error) {
        // If polling fails, wait and retry
        await new Promise(resolve => setTimeout(resolve, 5000));
        attempts++;
      }
    }

    throw new Error('Delegation did not complete within the expected time');
  }

  /**
   * Validate input data
   *
   * @param inputs - Input data to validate
   * @returns Array of validation errors
   */
  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];

    if (!inputs.input) {
      errors.push({
        field: 'input',
        message: 'Input is required',
        severity: 'error'
      });
    }

    if (inputs.input && typeof inputs.input !== 'string') {
      errors.push({
        field: 'input',
        message: 'Input must be a string',
        severity: 'error'
      });
    }

    // Validate task type
    if (inputs.taskType && typeof inputs.taskType === 'string') {
      const validTypes = ['generate', 'execute', 'delegate', 'optimize'];
      if (!validTypes.includes(inputs.taskType)) {
        errors.push({
          field: 'taskType',
          message: `Task type must be one of: ${validTypes.join(', ')}`,
          severity: 'error'
        });
      }
    }

    // Validate language
    if (inputs.language && typeof inputs.language === 'string') {
      const validLanguages = ['python', 'javascript', 'typescript', 'java', 'cpp', 'go', 'rust'];
      if (!validLanguages.includes(inputs.language)) {
        errors.push({
          field: 'language',
          message: `Language must be one of: ${validLanguages.join(', ')}`,
          severity: 'error'
        });
      }
    }

    // Validate code requirement for execute and optimize tasks
    const taskType = inputs.taskType as HephaestusTaskType || this.config.taskType as HephaestusTaskType;
    if ((taskType === 'execute' || taskType === 'optimize') && !inputs.code) {
      errors.push({
        field: 'code',
        message: `Code is required for ${taskType} task`,
        severity: 'error'
      });
    }

    // Validate delegateTo requirement for delegate task
    if (taskType === 'delegate' && !inputs.delegateTo) {
      errors.push({
        field: 'delegateTo',
        message: 'Delegate target is required for delegation task',
        severity: 'error'
      });
    }

    return errors;
  }

  /**
   * Get JSON Schema for configuration parameters
   *
   * @returns Parameter schema
   */
  getParameterSchema(): ParameterSchema {
    return {
      type: 'object',
      properties: {
        taskType: {
          type: 'string',
          description: 'Type of Hephaestus task to execute',
          enum: ['generate', 'execute', 'delegate', 'optimize'],
          default: 'generate'
        },
        language: {
          type: 'string',
          description: 'Programming language',
          enum: ['python', 'javascript', 'typescript', 'java', 'cpp', 'go', 'rust'],
          default: 'python'
        },
        enableExecution: {
          type: 'boolean',
          description: 'Enable code execution after generation',
          default: false
        },
        enableOptimization: {
          type: 'boolean',
          description: 'Enable code optimization',
          default: false
        },
        timeoutMs: {
          type: 'number',
          description: 'Timeout for code execution in milliseconds',
          minimum: 1000,
          maximum: 300000,
          default: 60000
        }
      },
      required: []
    };
  }

  /**
   * Get supported languages
   *
   * @returns Array of supported languages
   */
  getSupportedLanguages(): CodeLanguage[] {
    return ['python', 'javascript', 'typescript', 'java', 'cpp', 'go', 'rust'];
  }

  /**
   * Get available services for delegation
   *
   * @returns Promise resolving to available services
   */
  async getAvailableServices(): Promise<NodeResult> {
    try {
      const response = await apiClient.get<any>('/hephaestus/services');
      return this.createSuccessResult({ services: response.services || [] });
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to get available services'
      );
    }
  }

  /**
   * Get code quality metrics
   *
   * @param code - Code to analyze
   * @param language - Programming language
   * @returns Promise resolving to quality metrics
   */
  async getQualityMetrics(code: string, language: CodeLanguage): Promise<NodeResult> {
    try {
      const response = await apiClient.post<any>('/hephaestus/quality', {
        code,
        language
      });
      return this.createSuccessResult(response);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to get quality metrics'
      );
    }
  }
}

export default HephaestusNode;
