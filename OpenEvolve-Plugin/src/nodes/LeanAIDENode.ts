// @ts-nocheck
/**
 * LeanAide Node
 *
 * Lean 4 formal verification node for mathematical proofs.
 * Integrates with LeanAide for proof generation and verification.
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
import { leanaideApi, knowledgeApi } from '@/services/api/endpoints';

/**
 * Proof task types
 */
export type ProofTaskType = 'generate' | 'verify' | 'repair' | 'synthesis';

/**
 * LeanAide node configuration
 */
export interface LeanAideNodeConfig {
  taskType?: ProofTaskType;
  model?: string;
  temperature?: number;
  enableAutoRepair?: boolean;
  maxRepairAttempts?: number;
}

/**
 * Lean code output
 */
export interface LeanCodeOutput {
  code: string;
  theorem: string;
  proof: string;
  tactics: string[];
  verified: boolean;
  errors: string[];
  warnings: string[];
}

/**
 * Verification result
 */
export interface VerificationResult {
  verified: boolean;
  errors: Array<{
    line: number;
    column: number;
    message: string;
    severity: 'error' | 'warning';
  }>;
  warnings: string[];
  tacticsUsed: string[];
  proofSteps: number;
}

/**
 * Proof repair result
 */
export interface ProofRepairResult {
  originalCode: string;
  repairedCode: string;
  repairs: Array<{
    line: number;
    issue: string;
    fix: string;
  }>;
  verified: boolean;
  attempts: number;
}

/**
 * LeanAide result
 */
export interface LeanAideResult {
  taskId: string;
  taskType: ProofTaskType;
  theorem: string;
  output: LeanCodeOutput | VerificationResult | ProofRepairResult;
  model: string;
  metadata: {
    executedAt: Date;
    executionTime: number;
    temperature: number;
    parameters: {
      model: string;
      temperature: number;
      enableAutoRepair: boolean;
    };
  };
}

/**
 * LeanAide Node
 *
 * Generates and verifies Lean 4 formal proofs.
 * Provides automated proof repair and synthesis capabilities.
 */
export class LeanAideNode extends OpenEvolveBaseNode {
  static readonly DISPLAY_NAME = 'LeanAide Verification';
  static readonly DESCRIPTION = 'Lean 4 formal proof generation and verification with automated repair';
  static readonly ICON = 'leanaide';
  static readonly CATEGORY = 'verification';
  static readonly VERSION = '1.0.0';

  constructor(id: string, config: LeanAideNodeConfig = {}) {
    super(id, {
      taskType: 'generate',
      model: 'gpt-4',
      temperature: 0.3,
      enableAutoRepair: true,
      maxRepairAttempts: 3,
      ...config
    });
  }

  /**
   * Execute LeanAide task
   *
   * @param inputs - Must contain 'theorem' and optionally 'proof_attempt'
   * @param context - Execution context
   * @returns Promise resolving to LeanAide result
   */
  async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    try {
      const startTime = Date.now();

      // Extract inputs
      const taskType = (inputs.taskType as ProofTaskType) || (this.config.taskType as ProofTaskType);
      const theorem = inputs.theorem as string;
      const proofAttempt = inputs.proof_attempt as string | undefined;
      const model = (inputs.model as string) || this.config.model as string;
      const temperature = (inputs.temperature as number) || this.config.temperature as number;

      // Validate required inputs
      if (!theorem || theorem.trim().length === 0) {
        return this.createErrorResult('Theorem is required and cannot be empty');
      }

      context.updateProgress(10, `Preparing ${taskType} task`);

      let result: LeanCodeOutput | VerificationResult | ProofRepairResult;

      // Execute based on task type
      switch (taskType) {
        case 'generate':
          result = await this.generateProof(theorem, model, temperature, context);
          break;

        case 'verify':
          if (!proofAttempt) {
            return this.createErrorResult('Proof attempt is required for verification');
          }
          result = await this.verifyProof(proofAttempt, context);
          break;

        case 'repair':
          if (!proofAttempt) {
            return this.createErrorResult('Proof attempt is required for repair');
          }
          result = await this.repairProof(theorem, proofAttempt, model, temperature, context);
          break;

        case 'synthesis':
          result = await this.synthesizeProof(theorem, model, temperature, context);
          break;

        default:
          return this.createErrorResult(`Unknown task type: ${taskType}`);
      }

      const executionTime = Date.now() - startTime;

      const leanAideResult: LeanAideResult = {
        taskId: `task-${Date.now()}`,
        taskType,
        theorem,
        output: result,
        model,
        metadata: {
          executedAt: new Date(),
          executionTime,
          temperature,
          parameters: {
            model,
            temperature,
            enableAutoRepair: this.config.enableAutoRepair as boolean
          }
        }
      };

      context.updateProgress(100, `${taskType} task complete`);

      return this.createSuccessResult(leanAideResult);

    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Unknown error during LeanAide execution'
      );
    }
  }

  /**
   * Generate Lean 4 proof
   *
   * @param theorem - Theorem statement
   * @param model - Model to use
   * @param temperature - Temperature for generation
   * @param context - Execution context
   * @returns Promise resolving to generated proof
   */
  private async generateProof(
    theorem: string,
    model: string,
    temperature: number,
    context: ExecutionContext
  ): Promise<LeanCodeOutput> {
    context.updateProgress(20, 'Searching for relevant Lean 4 theorems');
    
    let theoremContext = '';
    try {
      const relatedTheorems = await knowledgeApi.getLean4Theorems();
      if (relatedTheorems && relatedTheorems.length > 0) {
        // Simple heuristic: find theorems mentioned in the statement
        // For now, just take top 3 as context
        theoremContext = relatedTheorems.slice(0, 3).map(t => `${t.name}: ${t.statement}`).join('\n');
      }
    } catch (e) {
      console.warn('Failed to fetch theorem context', e);
    }

    context.updateProgress(30, 'Generating Lean 4 proof');

    const response = await leanaideApi.generateProof({
      theorem: theoremContext ? `${theorem}\n\nContext:\n${theoremContext}` : theorem,
      model,
      temperature
    });

    context.updateProgress(80, 'Proof generated, post-processing');

    // If auto-repair is enabled and verification failed, attempt repair
    if (this.config.enableAutoRepair && !response.verified && response.errors.length > 0) {
      context.updateProgress(85, 'Verification failed, attempting auto-repair');
      return await this.attemptAutoRepair(theorem, response.code, model, temperature, context);
    }

    return response;
  }

  /**
   * Verify Lean 4 proof
   *
   * @param code - Lean 4 code to verify
   * @param context - Execution context
   * @returns Promise resolving to verification result
   */
  private async verifyProof(code: string, context: ExecutionContext): Promise<VerificationResult> {
    context.updateProgress(30, 'Verifying Lean 4 proof');

    const response = await leanaideApi.verifyProof(code);

    context.updateProgress(100, 'Verification complete');

    return response;
  }

  /**
   * Repair Lean 4 proof
   *
   * @param theorem - Theorem statement
   * @param proofAttempt - Proof attempt to repair
   * @param model - Model to use
   * @param temperature - Temperature for generation
   * @param context - Execution context
   * @returns Promise resolving to repair result
   */
  private async repairProof(
    theorem: string,
    proofAttempt: string,
    model: string,
    temperature: number,
    context: ExecutionContext
  ): Promise<ProofRepairResult> {
    context.updateProgress(30, 'Repairing Lean 4 proof');

    // Verify first to identify errors
    const verification = await this.verifyProof(proofAttempt, context);

    if (verification.verified) {
      // Proof is already valid
      return {
        originalCode: proofAttempt,
        repairedCode: proofAttempt,
        repairs: [],
        verified: true,
        attempts: 1
      };
    }

    // Attempt repair
    let currentCode = proofAttempt;
    const repairs: Array<{ line: number; issue: string; fix: string }> = [];
    const maxAttempts = this.config.maxRepairAttempts as number;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      context.updateProgress(
        40 + (attempt / maxAttempts) * 50,
        `Repair attempt ${attempt + 1}/${maxAttempts}`
      );

      // Generate new proof based on errors
      const response = await leanaideApi.generateProof({
        theorem,
        model,
        temperature
      });

      // Track repairs
      verification.errors.forEach(error => {
        repairs.push({
          line: error.line,
          issue: error.message,
          fix: 'Regenerated proof section'
        });
      });

      currentCode = response.code;

      // Verify repaired code
      const newVerification = await this.verifyProof(currentCode, context);
      if (newVerification.verified) {
        context.updateProgress(100, 'Proof repaired successfully');
        return {
          originalCode: proofAttempt,
          repairedCode: currentCode,
          repairs,
          verified: true,
          attempts: attempt + 1
        };
      }
    }

    context.updateProgress(100, 'Max repair attempts reached');

    return {
      originalCode: proofAttempt,
      repairedCode: currentCode,
      repairs,
      verified: false,
      attempts: maxAttempts
    };
  }

  /**
   * Synthesize proof from natural language
   *
   * @param theorem - Theorem statement in natural language
   * @param model - Model to use
   * @param temperature - Temperature for generation
   * @param context - Execution context
   * @returns Promise resolving to synthesized proof
   */
  private async synthesizeProof(
    theorem: string,
    model: string,
    temperature: number,
    context: ExecutionContext
  ): Promise<LeanCodeOutput> {
    context.updateProgress(30, 'Synthesizing proof from natural language');

    // For synthesis, we use the same generate proof endpoint
    // but with a higher temperature for more creativity
    const response = await leanaideApi.generateProof({
      theorem,
      model,
      temperature: Math.min(temperature + 0.2, 1.0)
    });

    context.updateProgress(100, 'Proof synthesis complete');

    return response;
  }

  /**
   * Attempt automatic repair of generated proof
   *
   * @param theorem - Theorem statement
   * @param code - Generated code with errors
   * @param model - Model to use
   * @param temperature - Temperature for generation
   * @param context - Execution context
   * @returns Promise resolving to repaired code
   */
  private async attemptAutoRepair(
    theorem: string,
    code: string,
    model: string,
    temperature: number,
    context: ExecutionContext
  ): Promise<LeanCodeOutput> {
    const maxAttempts = 2;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        // Regenerate with slightly higher temperature
        const response = await leanaideApi.generateProof({
          theorem,
          model,
          temperature: Math.min(temperature + 0.1, 1.0)
        });

        if (response.verified) {
          context.updateProgress(95, 'Auto-repair successful');
          return response;
        }
      } catch (error) {
        console.warn(`Auto-repair attempt ${attempt + 1} failed:`, error);
      }
    }

    // Return original if repair failed
    context.updateProgress(95, 'Auto-repair unsuccessful, returning original');
    return await leanaideApi.generateProof({ theorem, model, temperature });
  }

  /**
   * Validate input data
   *
   * @param inputs - Input data to validate
   * @returns Array of validation errors
   */
  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];

    if (!inputs.theorem) {
      errors.push({
        field: 'theorem',
        message: 'Theorem is required',
        severity: 'error'
      });
    }

    if (inputs.theorem && typeof inputs.theorem !== 'string') {
      errors.push({
        field: 'theorem',
        message: 'Theorem must be a string',
        severity: 'error'
      });
    }

    // Validate task type
    if (inputs.taskType && typeof inputs.taskType === 'string') {
      const validTypes = ['generate', 'verify', 'repair', 'synthesis'];
      if (!validTypes.includes(inputs.taskType)) {
        errors.push({
          field: 'taskType',
          message: `Task type must be one of: ${validTypes.join(', ')}`,
          severity: 'error'
        });
      }
    }

    // Validate proof attempt for verify and repair tasks
    const taskType = inputs.taskType as ProofTaskType || this.config.taskType as ProofTaskType;
    if ((taskType === 'verify' || taskType === 'repair') && !inputs.proof_attempt) {
      errors.push({
        field: 'proof_attempt',
        message: `Proof attempt is required for ${taskType} task`,
        severity: 'error'
      });
    }

    // Validate temperature
    if (inputs.temperature && typeof inputs.temperature === 'number') {
      if (inputs.temperature < 0 || inputs.temperature > 1) {
        errors.push({
          field: 'temperature',
          message: 'Temperature must be between 0 and 1',
          severity: 'error'
        });
      }
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
          description: 'Type of LeanAide task to execute',
          enum: ['generate', 'verify', 'repair', 'synthesis'],
          default: 'generate'
        },
        model: {
          type: 'string',
          description: 'Model to use for proof generation',
          default: 'gpt-4'
        },
        temperature: {
          type: 'number',
          description: 'Temperature for proof generation (0-1)',
          minimum: 0,
          maximum: 1,
          default: 0.3
        },
        enableAutoRepair: {
          type: 'boolean',
          description: 'Enable automatic proof repair on verification failure',
          default: true
        },
        maxRepairAttempts: {
          type: 'number',
          description: 'Maximum number of repair attempts',
          minimum: 1,
          maximum: 10,
          default: 3
        }
      },
      required: []
    };
  }

  /**
   * Get supported models
   *
   * @returns Promise resolving to list of supported models
   */
  async getSupportedModels(): Promise<NodeResult> {
    try {
      const models = await leanaideApi.getModels();
      return this.createSuccessResult({ models });
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to get supported models'
      );
    }
  }

  /**
   * Run benchmark
   *
   * @param dataset - Benchmark dataset
   * @param model - Model to use
   * @param evaluator - Evaluator to use
   * @returns Promise resolving to benchmark result
   */
  async runBenchmark(
    dataset: any[],
    model: string,
    evaluator: string
  ): Promise<NodeResult> {
    try {
      const response = await leanaideApi.runBenchmark({ dataset, model, evaluator });
      return this.createSuccessResult(response);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to run benchmark'
      );
    }
  }

  /**
   * Get benchmark results
   *
   * @param benchmarkId - Benchmark ID
   * @returns Promise resolving to benchmark results
   */
  async getBenchmarkResults(benchmarkId: string): Promise<NodeResult> {
    try {
      const results = await leanaideApi.getBenchmarkResults(benchmarkId);
      return this.createSuccessResult({ benchmarkId, results });
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to get benchmark results'
      );
    }
  }
}

export default LeanAideNode;
