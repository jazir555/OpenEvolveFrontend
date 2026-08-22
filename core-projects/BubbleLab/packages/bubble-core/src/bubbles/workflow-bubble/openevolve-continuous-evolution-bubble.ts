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

const ContinuousEvolutionParamsSchema = z
  .object({
    problem_statement: z.string().min(1),
    iterations: z.number().int().min(1).max(10000).default(50),
    population_size: z.number().int().min(1).max(1000).default(30),
    workflow_type: z.string().default('evolution'),
    metrics: z
      .array(
        z.object({
          name: z.string(),
          value: z.union([z.number(), z.string(), z.boolean()]),
        })
      )
      .optional(),
    base_url: z.string().url().default(resolveBaseUrl()),
    auth_token: z.string().optional(),
  })
  .passthrough();

type ContinuousEvolutionParams = z.input<typeof ContinuousEvolutionParamsSchema> & ServiceBubbleParams;

const ContinuousEvolutionResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z
    .object({
      trigger: z.unknown().optional(),
      validation: z.unknown().optional(),
      metrics: z.unknown().optional(),
    })
    .optional(),
  error: z.string().optional(),
  timing: z.object({ total: z.number() }),
});

type ContinuousEvolutionResult = z.output<typeof ContinuousEvolutionResultSchema> & BubbleOperationResult;

export class OpenEvolveContinuousEvolutionBubble extends WorkflowBubble<
  ContinuousEvolutionParams,
  ContinuousEvolutionResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-continuous-evolution' as BubbleName;
  static readonly type = 'workflow' as const;
  static readonly schema = ContinuousEvolutionParamsSchema;
  static readonly resultSchema = ContinuousEvolutionResultSchema;
  static readonly shortDescription = 'OpenEvolve continuous evolution (trigger -> validate -> metrics)';
  static readonly longDescription = `
    Scheduled-style continuous evolution: triggers an OpenEvolve evolution, runs
    faster validation, then collects & aggregates metrics via the metrics collector.
  `;
  static readonly alias = 'openevolve-continuous-evolution';

  constructor(params: ContinuousEvolutionParams, context?: BubbleContext) {
    super(params, context);
  }

  protected async performAction(): Promise<ContinuousEvolutionResult> {
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
          operation: 'continuous',
          error: `trigger failed: ${triggerRes.error}`,
          timing: { total: Date.now() - startTime },
        };
      }
      const triggerData = (triggerRes.data ?? {}) as any;
      const code = JSON.stringify(triggerData.raw ?? triggerData);

      const validationRes = await factory
        .createBubble('openevolve-evolution-validation', {
          operation: 'validate',
          code,
          run_z3: true,
          run_leanaide: false,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();

      const metricsRes = await factory
        .createBubble('openevolve-metrics-collector', {
          operation: 'collect',
          metrics: this.params.metrics,
          workflow_id: triggerData.workflow_id,
          fetch_analytics: true,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();

      return {
        success: true,
        operation: 'continuous',
        data: {
          trigger: triggerRes.data,
          validation: validationRes.data,
          metrics: metricsRes.data,
        },
        timing: { total: Date.now() - startTime },
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: 'continuous',
        error: message,
        timing: { total: Date.now() - startTime },
      };
    }
  }
}

export default OpenEvolveContinuousEvolutionBubble;
