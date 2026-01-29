// @ts-nocheck
/**
 * Decomposition Node - Integration Library Version
 *
 * This node uses the OpenEvolve Integration Library to communicate with
 * the Python backend, which executes the actual decomposition logic.
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
import { useAuthStore } from '@/stores/authStore';

/**
 * Decomposition strategy types
 */
export type DecompositionStrategy = 'semantic' | 'dependency' | 'complexity' | 'hybrid' | 'research';

/**
 * Sub-problem interface
 */
export interface SubProblem {
  id: string;
  title: string;
  description: string;
  complexity: number;
  estimated_time: number;
  dependencies: string[];
  success_criteria: any[];
  type: string;
  status: string;
}

/**
 * Dependency graph interface
 */
export interface DependencyGraph {
  nodes: string[];
  edges: any[];
  execution_order: string[];
}

/**
 * Decomposition node configuration
 */
export interface DecompositionNodeConfig {
  strategy?: DecompositionStrategy;
  maxSubProblems?: number;
  recursionDepthLimit?: number;
  qualityThreshold?: number;
  enableBackendExecution?: boolean;
  backendUrl?: string;
}

/**
 * Decomposition result interface
 */
export interface DecompositionResult {
  sub_problems: SubProblem[];
  decomposition_tree: DependencyGraph;
  complexity_metrics: {
    overall_score: number;
    meets_thresholds: boolean;
    confidence: number;
  };
  estimated_time: number;
  method_used: string;
  total_sub_problems: number;
  confidence: number;
  validation_checkpoints: number;
  plan_id: string;
  problem_id: string;
}

/**
 * Problem Decomposition Node (Integration Library Version)
 *
 * This node uses the OpenEvolve Integration Library to delegate decomposition
 * to the Python backend. The Python backend uses the existing DecompositionEngine
 * from decomposition_engine.py.
 *
 * Benefits of this approach:
 * - Reuses existing Python implementation
 * - No need to duplicate logic in TypeScript
 * - Consistent behavior across all clients
 * - Easy to update Python backend without changing frontend
 */
export class DecompositionNode extends OpenEvolveBaseNode {
  static readonly DISPLAY_NAME = 'Problem Decomposition';
  static readonly DESCRIPTION = 'Break down complex problems using Python DecompositionEngine via integration library';
  static readonly ICON = 'decomposition';
  static readonly CATEGORY = 'analysis';
  static readonly VERSION = '2.0.0';

  constructor(id: string, config: DecompositionNodeConfig = {}) {
    super(id, {
      strategy: 'hybrid',
      maxSubProblems: 3,
      recursionDepthLimit: 1,
      qualityThreshold: 0.7,
      enableBackendExecution: true,
      backendUrl: 'http://localhost:8000',
      ...config
    });

  }

  /**
   * Execute problem decomposition using the integration library
   *
   * @param inputs - Must contain 'problem_statement' string
   * @param context - Execution context
   * @returns Promise resolving to decomposition result
   */
  async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    try {
      const startTime = Date.now();

      // Extract inputs
      const problemStatement = inputs.problem_statement || inputs.problem as string;
      const method = (inputs.method as DecompositionStrategy) || (this.config.strategy as DecompositionStrategy);
      const requirements = inputs.requirements as Record<string, any> | undefined;
      const constraints = inputs.constraints as Record<string, any> | undefined;

      // Validate that we have a problem to decompose
      if (!problemStatement || problemStatement.trim().length === 0) {
        return this.createErrorResult('Problem statement is required and cannot be empty');
      }

      context.updateProgress(10, 'Validating inputs');

      // Use integration library to call Python backend
      if (this.config.enableBackendExecution) {
        return await this.executeWithBackend(problemStatement, method, requirements, constraints, context);
      } else {
        return await this.executeLocally(problemStatement, method, requirements, context);
      }

    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Unknown error during decomposition'
      );
    }
  }

  /**
   * Execute decomposition using Python backend via integration library
   */
  private async executeWithBackend(
    problemStatement: string,
    method: DecompositionStrategy,
    requirements: Record<string, any> | undefined,
    constraints: Record<string, any> | undefined,
    context: ExecutionContext
  ): Promise<NodeResult> {
    try {
      context.updateProgress(20, 'Connecting to backend');

      // Prepare request for backend
      const backendInputs: Record<string, any> = {
        problem_statement: problemStatement,
        method: method,
        max_subproblems: this.config.maxSubProblems,
        recursion_depth_limit: this.config.recursionDepthLimit,
      };

      if (requirements) {
        backendInputs.requirements = requirements;
      }

      if (constraints) {
        backendInputs.constraints = constraints;
      }

      context.updateProgress(30, 'Executing decomposition on backend');

      const result: DecompositionResult = await this.postToBackend(
        '/decomposition/analyze',
        backendInputs
      );

      context.updateProgress(100, `Decomposition complete: ${result.total_sub_problems} sub-problems`);

      // Transform backend result to match expected output format
      const transformedResult = {
        subProblems: result.sub_problems.map(sp => ({
          id: sp.id,
          title: sp.title,
          description: sp.description,
          complexity: sp.complexity,
          estimatedEffort: sp.estimated_time,
          dependencies: sp.dependencies,
          requirements: sp.success_criteria || [],
          priority: this.assessPriorityFromComplexity(sp.complexity),
          status: sp.status as any,
          metadata: {
            parentProblem: problemStatement.substring(0, 100),
            decompositionStrategy: result.method_used as DecompositionStrategy,
            qualityScore: result.complexity_metrics.confidence
          }
        })),
        dependencyGraph: result.decomposition_tree,
        qualityMetrics: {
          completeness: result.complexity_metrics.overall_score,
          clarity: result.complexity_metrics.overall_score * 0.9,
          feasibility: result.complexity_metrics.overall_score * 0.95,
          overall: result.complexity_metrics.overall_score
        },
        metadata: {
          originalProblem: problemStatement,
          strategyUsed: result.method_used as DecompositionStrategy,
          executionTime: 0, // Backend tracks this
          subProblemCount: result.total_sub_problems,
          totalEstimatedEffort: result.estimated_time,
          planId: result.plan_id,
          problemId: result.problem_id,
          confidence: result.confidence
        }
      };

      return this.createSuccessResult(transformedResult);

    } catch (error) {
      // If backend call fails, fall back to local execution
      console.warn('Backend execution failed, falling back to local:', error);
      context.updateProgress(20, 'Backend unavailable, using local execution');
      return this.executeLocally(problemStatement, method, requirements, context);
    }
  }

  /**
   * Execute decomposition locally (fallback/simplified version)
   * This is used when backend is unavailable or for testing
   */
  private async executeLocally(
    problemStatement: string,
    method: DecompositionStrategy,
    requirements: Record<string, any> | undefined,
    context: ExecutionContext
  ): Promise<NodeResult> {
    context.updateProgress(40, 'Performing local decomposition');

    // Simple local decomposition logic
    const sentences = problemStatement.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const subProblems = sentences.map((sentence, index) => ({
      id: `sub-problem-${index + 1}`,
      title: `Component ${index + 1}`,
      description: sentence.trim(),
      complexity: 0.5,
      estimatedEffort: 10,
      dependencies: index > 0 ? [`sub-problem-${index}`] : [],
      requirements: requirements ? Object.values(requirements) : [],
      priority: index === 0 ? 'high' as const : 'medium' as const,
      status: 'pending' as const,
      metadata: {
        parentProblem: problemStatement.substring(0, 100),
        decompositionStrategy: method
      }
    }));

    context.updateProgress(100, 'Local decomposition complete');

    return this.createSuccessResult({
      subProblems,
      dependencyGraph: {
        nodes: subProblems.map(sp => sp.id),
        edges: [],
        levels: [subProblems.map(sp => sp.id)],
        criticalPath: []
      },
      qualityMetrics: {
        completeness: 0.7,
        clarity: 0.7,
        feasibility: 0.7,
        overall: 0.7
      },
      metadata: {
        originalProblem: problemStatement,
        strategyUsed: method,
        executionTime: 0,
        subProblemCount: subProblems.length,
        totalEstimatedEffort: subProblems.reduce((sum, sp) => sum + sp.estimatedEffort, 0),
        note: 'Executed locally (backend unavailable)'
      }
    });
  }

  /**
   * Assess priority from complexity score
   */
  private assessPriorityFromComplexity(complexity: number): 'high' | 'medium' | 'low' {
    if (complexity > 0.7) return 'high';
    if (complexity > 0.4) return 'medium';
    return 'low';
  }

  /**
   * Validate input data
   *
   * @param inputs - Input data to validate
   * @returns Array of validation errors
   */
  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];

    // Check for required problem field (can be 'problem' or 'problem_statement')
    const problem = inputs.problem_statement || inputs.problem;

    if (!problem) {
      errors.push({
        field: 'problem_statement',
        message: 'Problem statement is required',
        severity: 'error'
      });
    }

    if (problem && typeof problem !== 'string') {
      errors.push({
        field: 'problem_statement',
        message: 'Problem must be a string',
        severity: 'error'
      });
    }

    if (problem && problem.length < 20) {
      errors.push({
        field: 'problem_statement',
        message: 'Problem statement is too short (minimum 20 characters)',
        severity: 'warning'
      });
    }

    // Validate optional fields
    if (inputs.requirements && typeof inputs.requirements !== 'object') {
      errors.push({
        field: 'requirements',
        message: 'Requirements must be an object',
        severity: 'error'
      });
    }

    if (inputs.constraints && typeof inputs.constraints !== 'object') {
      errors.push({
        field: 'constraints',
        message: 'Constraints must be an object',
        severity: 'error'
      });
    }

    // Validate method if provided
    if (inputs.method && typeof inputs.method === 'string') {
      const validMethods = ['semantic', 'dependency', 'complexity', 'hybrid', 'research'];
      if (!validMethods.includes(inputs.method)) {
        errors.push({
          field: 'method',
          message: `Method must be one of: ${validMethods.join(', ')}`,
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
        strategy: {
          type: 'string',
          description: 'Decomposition strategy to use',
          enum: ['semantic', 'dependency', 'complexity', 'hybrid', 'research'],
          default: 'hybrid'
        },
        maxSubProblems: {
          type: 'number',
          description: 'Maximum number of sub-problems to generate',
          minimum: 0,
          maximum: 50,
          default: 3
        },
        recursionDepthLimit: {
          type: 'number',
          description: '0 = unlimited recursion depth',
          minimum: 0,
          maximum: 10,
          default: 1
        },
        qualityThreshold: {
          type: 'number',
          description: 'Minimum quality score (0-1) for sub-problems',
          minimum: 0,
          maximum: 1,
          default: 0.7
        },
        enableBackendExecution: {
          type: 'boolean',
          description: 'Use Python backend via integration library',
          default: true
        },
        backendUrl: {
          type: 'string',
          description: 'URL of the Python backend API',
          default: 'http://localhost:8000'
        }
      },
      required: []
    };
  }

  /**
   * Cleanup when node is destroyed
   */
  private async postToBackend(endpoint: string, payload: Record<string, any>): Promise<DecompositionResult> {
    const backendUrl = (this.config.backendUrl as string | undefined) || '';

    if (!backendUrl) {
      return apiClient.post<DecompositionResult>(endpoint, payload);
    }

    const url = new URL(endpoint, backendUrl).toString();
    const token = useAuthStore.getState().token;
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({}));
      throw new Error(errorPayload?.error?.message || response.statusText);
    }

    return response.json();
  }

  destroy(): void {
    // No-op for now; backend connections are stateless HTTP calls.
  }
}

export default DecompositionNode;
