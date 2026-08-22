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

const AdaptiveEvolutionParamsSchema = z
  .object({
    problem_statement: z.string().min(1),
    learn_from_history: z.boolean().default(true),
    iterations: z.number().int().min(1).max(10000).default(100),
    population_size: z.number().int().min(1).max(1000).default(50),
    workflow_type: z.string().default('evolution'),
    base_url: z.string().url().default(resolveBaseUrl()),
    auth_token: z.string().optional(),
  })
  .passthrough();

type AdaptiveEvolutionParams = z.input<typeof AdaptiveEvolutionParamsSchema> & ServiceBubbleParams;

const AdaptiveEvolutionResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z
    .object({
      retrieval: z.unknown().optional(),
      trigger: z.unknown().optional(),
      validation: z.unknown().optional(),
      capture: z.unknown().optional(),
    })
    .optional(),
  error: z.string().optional(),
  timing: z.object({ total: z.number() }),
});

type AdaptiveEvolutionResult = z.output<typeof AdaptiveEvolutionResultSchema> & BubbleOperationResult;

export class OpenEvolveAdaptiveEvolutionBubble extends WorkflowBubble<
  AdaptiveEvolutionParams,
  AdaptiveEvolutionResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-adaptive-evolution' as BubbleName;
  static readonly type = 'workflow' as const;
  static readonly schema = AdaptiveEvolutionParamsSchema;
  static readonly resultSchema = AdaptiveEvolutionResultSchema;
  static readonly shortDescription = 'OpenEvolve adaptive evolution (retrieve -> trigger -> validate -> capture)';
  static readonly longDescription = `
    Knowledge-aware evolution: retrieves prior learnings from the knowledge store,
    triggers an evolution, validates it, then captures the result back into the
    knowledge store for future runs.
  `;
  static readonly alias = 'openevolve-adaptive-evolution';

  constructor(params: AdaptiveEvolutionParams, context?: BubbleContext) {
    super(params, context);
  }

  protected async performAction(): Promise<AdaptiveEvolutionResult> {
    const startTime = Date.now();
    try {
      const factory = await BubbleFactory.getInstance();

      const retrievalRes = this.params.learn_from_history
        ? await factory
            .createBubble('openevolve-knowledge-retrieval', {
              operation: 'retrieve',
              query: this.params.problem_statement,
              base_url: this.params.base_url,
              auth_token: this.params.auth_token,
            })
            .action()
        : { success: true, data: undefined as unknown };

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
          operation: 'adaptive',
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
          run_leanaide: true,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();

      const captureRes = await factory
        .createBubble('openevolve-knowledge-capture', {
          operation: 'capture',
          content: code,
          artifact_type: 'learning',
          source_workflow_id: triggerData.workflow_id,
          problem_type: this.params.workflow_type,
          effectiveness_score: validationRes.data
            ? ((validationRes.data as any).valid ? 1 : 0)
            : 0,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();

      return {
        success: true,
        operation: 'adaptive',
        data: {
          retrieval: retrievalRes.data,
          trigger: triggerRes.data,
          validation: validationRes.data,
          capture: captureRes.data,
        },
        timing: { total: Date.now() - startTime },
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: 'adaptive',
        error: message,
        timing: { total: Date.now() - startTime },
      };
    }
  }
}

export default OpenEvolveAdaptiveEvolutionBubble;
