/**
 * SubProblem Node
 *
 * Handles solving or planning for an individual sub-problem produced by
 * a decomposition step. Supports custom solver callbacks and structured
 * plan generation as a fallback.
 *
 * @module nodes
 */

import {
  OpenEvolveBaseNode,
  NodeInputs,
  NodeResult,
  ExecutionContext,
  ValidationError,
  ParameterSchema,
} from './OpenEvolveBaseNode';

export interface SubProblemNodeConfig {
  maxAttempts?: number;
  qualityThreshold?: number;
  useAdversarialTesting?: boolean;
  useEvolutionaryOptimization?: boolean;
}

export interface SubProblemPlanStep {
  id: string;
  description: string;
  status: 'pending' | 'completed';
  rationale?: string;
}

export interface SubProblemResult {
  subProblemId: string;
  title: string;
  summary: string;
  solution: string;
  plan: SubProblemPlanStep[];
  qualityScore: number;
  attempts: number;
  metadata: {
    solvedAt: Date;
    executionTime: number;
    usedCustomSolver: boolean;
  };
}

export class SubProblemNode extends OpenEvolveBaseNode {
  static readonly DISPLAY_NAME = 'Sub-Problem Solver';
  static readonly DESCRIPTION = 'Solve or plan an individual sub-problem with optional custom solver callbacks';
  static readonly ICON = 'subproblem';
  static readonly CATEGORY = 'analysis';
  static readonly VERSION = '1.0.0';

  constructor(id: string, config: SubProblemNodeConfig = {}) {
    super(id, {
      maxAttempts: 3,
      qualityThreshold: 0.6,
      useAdversarialTesting: false,
      useEvolutionaryOptimization: false,
      ...config,
    });
  }

  async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    try {
      const startTime = Date.now();
      const subProblem = inputs.subProblem || {};
      const title =
        subProblem.title ||
        (inputs.title as string) ||
        'Untitled sub-problem';
      const description =
        subProblem.description ||
        (inputs.description as string) ||
        (inputs.problem_statement as string) ||
        '';
      const subProblemId =
        subProblem.id || (inputs.subProblemId as string) || `subproblem-${Date.now()}`;

      if (!description.trim()) {
        return this.createErrorResult('Sub-problem description is required');
      }

      context.updateProgress(15, 'Preparing sub-problem analysis');

      const solver = inputs.solveFn as
        | ((payload: { title: string; description: string; inputs: NodeInputs }) => Promise<string> | string)
        | undefined;

      let solution = '';
      let usedCustomSolver = false;

      if (solver) {
        usedCustomSolver = true;
        const result = await solver({ title, description, inputs });
        solution = typeof result === 'string' ? result : JSON.stringify(result);
      } else if (typeof inputs.solution === 'string') {
        solution = inputs.solution as string;
      } else {
        solution = this.generateSolution(description);
      }

      context.updateProgress(60, 'Building execution plan');

      const plan = this.buildPlan(description);
      const qualityScore = this.estimateQuality(solution, plan);
      const executionTime = Date.now() - startTime;

      const result: SubProblemResult = {
        subProblemId,
        title,
        summary: description.substring(0, 160),
        solution,
        plan,
        qualityScore,
        attempts: 1,
        metadata: {
          solvedAt: new Date(),
          executionTime,
          usedCustomSolver,
        },
      };

      context.updateProgress(100, 'Sub-problem solution complete');
      return this.createSuccessResult(result);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Unknown error during sub-problem solving'
      );
    }
  }

  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];
    const description =
      inputs?.subProblem?.description ||
      inputs.description ||
      inputs.problem_statement;

    if (!description) {
      errors.push({
        field: 'description',
        message: 'Sub-problem description is required',
        severity: 'error',
      });
    }

    if (description && typeof description !== 'string') {
      errors.push({
        field: 'description',
        message: 'Description must be a string',
        severity: 'error',
      });
    }

    return errors;
  }

  getParameterSchema(): ParameterSchema {
    return {
      type: 'object',
      properties: {
        maxAttempts: {
          type: 'number',
          description: 'Maximum solution attempts for the sub-problem',
          minimum: 1,
          maximum: 10,
          default: 3,
        },
        qualityThreshold: {
          type: 'number',
          description: 'Minimum quality score required for acceptance (0-1)',
          minimum: 0,
          maximum: 1,
          default: 0.6,
        },
        useAdversarialTesting: {
          type: 'boolean',
          description: 'Enable adversarial evaluation of the solution',
          default: false,
        },
        useEvolutionaryOptimization: {
          type: 'boolean',
          description: 'Enable evolutionary optimization for solution refinement',
          default: false,
        },
      },
      required: [],
    };
  }

  private buildPlan(description: string): SubProblemPlanStep[] {
    const sentences = description
      .split(/[.!?]+/)
      .map((segment) => segment.trim())
      .filter(Boolean);

    if (sentences.length === 0) {
      return [
        {
          id: 'step-1',
          description: 'Clarify sub-problem requirements and expected output',
          status: 'pending',
        },
      ];
    }

    return sentences.map((sentence, index) => ({
      id: `step-${index + 1}`,
      description: sentence,
      status: 'pending',
      rationale: index === 0 ? 'Establish the core objective' : undefined,
    }));
  }

  private generateSolution(description: string): string {
    const summary = description.length > 180 ? `${description.slice(0, 180)}...` : description;
    return [
      'Proposed Solution:',
      `- Objective: ${summary}`,
      '- Approach: Break down the objective into actionable steps, validate assumptions, and iterate.',
      '- Deliverable: A verified outcome aligned with the sub-problem constraints.',
    ].join('\n');
  }

  private estimateQuality(solution: string, plan: SubProblemPlanStep[]): number {
    const solutionScore = Math.min(solution.length / 500, 1);
    const planScore = Math.min(plan.length / 6, 1);
    return Number((0.6 * solutionScore + 0.4 * planScore).toFixed(2));
  }
}

export default SubProblemNode;
