/**
 * Gauntlet Node
 *
 * Runs a multi-stage validation pipeline against a candidate output.
 * Supports custom validators and weighted scoring.
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

export interface GauntletStageConfig {
  name: string;
  criteria: Array<{
    id: string;
    description?: string;
    weight?: number;
    validator?: string;
  }>;
  minScore?: number;
}

export interface GauntletNodeConfig {
  stages?: number;
  stageConfigs?: GauntletStageConfig[];
  progressiveValidation?: boolean;
  minStageScore?: number;
  stopOnFailure?: boolean;
  detailedReports?: boolean;
  strictness?: 'lenient' | 'medium' | 'strict';
}

export interface CriteriaResult {
  criterion: string;
  passed: boolean;
  score: number;
  details?: string;
  skipped?: boolean;
}

export interface StageResult {
  stage: string;
  score: number;
  passed: boolean;
  criteriaResults: CriteriaResult[];
  warnings: string[];
}

export interface GauntletResult {
  passed: boolean;
  score: number;
  stages: StageResult[];
  warnings: string[];
  metadata: {
    executedAt: Date;
    executionTime: number;
    strictness: string;
  };
}

export class GauntletNode extends OpenEvolveBaseNode {
  static readonly DISPLAY_NAME = 'Validation Gauntlet';
  static readonly DESCRIPTION = 'Run multi-stage validation on candidate outputs with weighted scoring';
  static readonly ICON = 'gauntlet';
  static readonly CATEGORY = 'verification';
  static readonly VERSION = '1.0.0';

  constructor(id: string, config: GauntletNodeConfig = {}) {
    super(id, {
      stages: 1,
      stageConfigs: [],
      progressiveValidation: true,
      minStageScore: 0.7,
      stopOnFailure: true,
      detailedReports: true,
      strictness: 'medium',
      ...config,
    });
  }

  async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    try {
      const startTime = Date.now();
      const candidate = inputs.candidate ?? inputs.outputs ?? inputs.data;
      const validators = (inputs.validators || {}) as Record<
        string,
        (candidate: any) => boolean | { passed: boolean; score?: number; details?: string }
      >;

      const stageConfigs = (inputs.stageConfigs as GauntletStageConfig[] | undefined) ||
        (this.config.stageConfigs as GauntletStageConfig[]);

      const stages = stageConfigs.length
        ? stageConfigs
        : [
            {
              name: 'default',
              criteria: [
                { id: 'structure', description: 'Output structure is valid', weight: 0.5 },
                { id: 'content', description: 'Output content meets expectations', weight: 0.5 },
              ],
            },
          ];

      if (!candidate) {
        return this.createErrorResult('Candidate output is required for validation');
      }

      context.updateProgress(10, 'Starting validation gauntlet');

      const stageResults: StageResult[] = [];
      const warnings: string[] = [];
      let overallScore = 0;
      let overallPassed = true;

      for (let index = 0; index < stages.length; index++) {
        const stage = stages[index];
        context.updateProgress(
          15 + (index / stages.length) * 70,
          `Running stage: ${stage.name}`
        );

        const criteriaResults: CriteriaResult[] = [];
        const stageWarnings: string[] = [];

        for (const criterion of stage.criteria) {
          const validator = criterion.validator ? validators[criterion.validator] : validators[criterion.id];
          if (!validator) {
            criteriaResults.push({
              criterion: criterion.id,
              passed: true,
              score: 0,
              skipped: true,
              details: 'No validator supplied; criterion skipped',
            });
            stageWarnings.push(`No validator provided for criterion '${criterion.id}'`);
            continue;
          }

          const outcome = validator(candidate);
          const normalized =
            typeof outcome === 'boolean'
              ? { passed: outcome, score: outcome ? 1 : 0 }
              : {
                  passed: outcome.passed,
                  score: outcome.score ?? (outcome.passed ? 1 : 0),
                  details: outcome.details,
                };

          criteriaResults.push({
            criterion: criterion.id,
            passed: normalized.passed,
            score: normalized.score ?? 0,
            details: normalized.details,
          });
        }

        const stageScore = this.calculateStageScore(criteriaResults, stage.criteria);
        const minScore = stage.minScore ?? (this.config.minStageScore as number);
        const stagePassed = stageScore >= minScore;

        const stageResult: StageResult = {
          stage: stage.name,
          score: Number(stageScore.toFixed(2)),
          passed: stagePassed,
          criteriaResults,
          warnings: stageWarnings,
        };

        stageResults.push(stageResult);
        warnings.push(...stageWarnings);
        overallScore += stageScore;
        overallPassed = overallPassed && stagePassed;

        if (!stagePassed && (this.config.stopOnFailure as boolean)) {
          warnings.push(`Stopping after stage '${stage.name}' due to failure`);
          break;
        }
      }

      const normalizedScore =
        stageResults.length > 0 ? overallScore / stageResults.length : 0;

      const result: GauntletResult = {
        passed: overallPassed,
        score: Number(normalizedScore.toFixed(2)),
        stages: stageResults,
        warnings,
        metadata: {
          executedAt: new Date(),
          executionTime: Date.now() - startTime,
          strictness: this.config.strictness as string,
        },
      };

      context.updateProgress(100, 'Validation gauntlet complete');
      return this.createSuccessResult(result);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Unknown error during gauntlet validation'
      );
    }
  }

  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];

    if (!inputs.candidate && !inputs.outputs && !inputs.data) {
      errors.push({
        field: 'candidate',
        message: 'Candidate output is required for validation',
        severity: 'error',
      });
    }

    return errors;
  }

  getParameterSchema(): ParameterSchema {
    return {
      type: 'object',
      properties: {
        stages: {
          type: 'number',
          description: 'Number of validation stages',
          minimum: 1,
          maximum: 10,
          default: 1,
        },
        minStageScore: {
          type: 'number',
          description: 'Minimum score required per stage (0-1)',
          minimum: 0,
          maximum: 1,
          default: 0.7,
        },
        stopOnFailure: {
          type: 'boolean',
          description: 'Stop validation when a stage fails',
          default: true,
        },
        strictness: {
          type: 'string',
          description: 'Validation strictness level',
          enum: ['lenient', 'medium', 'strict'],
          default: 'medium',
        },
      },
      required: [],
    };
  }

  private calculateStageScore(criteriaResults: CriteriaResult[], criteriaConfig: GauntletStageConfig['criteria']): number {
    if (!criteriaResults.length) {
      return 0;
    }

    const weights = criteriaConfig.map((criterion) => criterion.weight ?? 1);
    const totalWeight = weights.reduce((sum, weight) => sum + weight, 0) || 1;

    return criteriaResults.reduce((sum, result, index) => {
      if (result.skipped) {
        return sum;
      }
      const weight = weights[index] || 1;
      return sum + result.score * weight;
    }, 0) / totalWeight;
  }
}

export default GauntletNode;
