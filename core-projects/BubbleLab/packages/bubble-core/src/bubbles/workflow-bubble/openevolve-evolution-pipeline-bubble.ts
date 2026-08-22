import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';
import { BubbleFactory } from '../../bubble-factory.js';

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  return (envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000').replace(/\/$/, '');
};

const EvolutionPipelineParamsSchema = z
  .object({
    problem_statement: z.string().min(1),
    iterations: z.number().int().min(1).max(10000).default(100),
    population_size: z.number().int().min(1).max(1000).default(50),
    workflow_type: z.string().default('evolution'),
    evolved_code: z.string().optional(),
    target_system: z.string().optional(),
    target_path: z.string().optional(),
    base_url: z.string().url().default(resolveBaseUrl()),
    auth_token: z.string().optional(),
    auto_deploy: z.boolean().default(true),
  })
  .passthrough();

type EvolutionPipelineParams = z.input<typeof EvolutionPipelineParamsSchema> & ServiceBubbleParams;

const EvolutionPipelineResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z
    .object({
      trigger: z.unknown().optional(),
      validation: z.unknown().optional(),
      application: z.unknown().optional(),
    })
    .optional(),
  error: z.string().optional(),
  timing: z.object({ total: z.number() }),
});

type EvolutionPipelineResult = z.output<typeof EvolutionPipelineResultSchema> & BubbleOperationResult;

export class OpenEvolveEvolutionPipelineBubble extends WorkflowBubble<
  EvolutionPipelineParams,
  EvolutionPipelineResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-evolution-pipeline' as BubbleName;
  static readonly type = 'workflow' as const;
  static readonly schema = EvolutionPipelineParamsSchema;
  static readonly resultSchema = EvolutionPipelineResultSchema;
  static readonly shortDescription = 'OpenEvolve evolution pipeline (trigger -> validate -> apply)';
  static readonly longDescription = `
    End-to-end evolution workflow: triggers an OpenEvolve evolution, validates the
    evolved code with Z3 + LeanAide, then applies the validated code.
  `;
  static readonly alias = 'openevolve-evolution-pipeline';

  constructor(params: EvolutionPipelineParams, context?: BubbleContext) {
    super(params, context);
  }

  protected async performAction(): Promise<EvolutionPipelineResult> {
    const startTime = Date.now();
    try {
      const factory = await BubbleFactory.getInstance();

      const triggerRes = await factory
        .createBubble('openevolve-evolution-trigger', {
          operation: 'create_and_run',
          problem_statement: this.params.problem_statement,
          iterations: this.params.iterations,
          population_size: this.params.population_size,
          workflow_type: this.params.workflow_type,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();
      if (!triggerRes.success) {
        return {
          success: false,
          operation: 'pipeline',
          error: `trigger failed: ${triggerRes.error}`,
          timing: { total: Date.now() - startTime },
        };
      }
      const triggerData = (triggerRes.data ?? {}) as any;

      const code =
        this.params.evolved_code ?? JSON.stringify(triggerData.raw ?? triggerData);

      const validationRes = await factory
        .createBubble('openevolve-evolution-validation', {
          operation: 'validate',
          code,
          run_z3: true,
          run_leanaide: true,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();

      const applicationRes = await factory
        .createBubble('openevolve-evolution-application', {
          operation: 'apply',
          workflow_id: triggerData.workflow_id,
          evolved_code: code,
          language: 'json',
          target_system: this.params.target_system,
          target_path: this.params.target_path,
          auto_deploy: this.params.auto_deploy,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();

      return {
        success: true,
        operation: 'pipeline',
        data: {
          trigger: triggerRes.data,
          validation: validationRes.data,
          application: applicationRes.data,
        },
        timing: { total: Date.now() - startTime },
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: 'pipeline',
        error: message,
        timing: { total: Date.now() - startTime },
      };
    }
  }
}

export default OpenEvolveEvolutionPipelineBubble;
