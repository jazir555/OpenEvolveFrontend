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

const OpenEvolveAdversarialToEvolutionParamsSchema = z
  .object({
    problem_statement: z.string().optional(),
    run_id: z.string().optional(),
    config: z.record(z.unknown()).optional(),
    evolved_code: z.string().optional(),
    base_url: z.string().url().default(resolveBaseUrl()),
    auth_token: z.string().optional(),
  })
  .passthrough();

type OpenEvolveAdversarialToEvolutionParams = z.input<
  typeof OpenEvolveAdversarialToEvolutionParamsSchema
> &
  ServiceBubbleParams;

const OpenEvolveAdversarialToEvolutionResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  steps: z.array(z.unknown()).optional(),
  error: z.string().optional(),
  timing: z.object({ total: z.number() }),
});

type OpenEvolveAdversarialToEvolutionResult = z.output<
  typeof OpenEvolveAdversarialToEvolutionResultSchema
> &
  BubbleOperationResult;

export class OpenEvolveAdversarialToEvolutionBubble extends WorkflowBubble<
  OpenEvolveAdversarialToEvolutionParams,
  OpenEvolveAdversarialToEvolutionResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-adversarial-to-evolution' as BubbleName;
  static readonly type = 'workflow' as const;
  static readonly schema = OpenEvolveAdversarialToEvolutionParamsSchema;
  static readonly resultSchema = OpenEvolveAdversarialToEvolutionResultSchema;
  static readonly shortDescription = 'OpenEvolve adversarial -> evolution workflow';
  static readonly longDescription = `
    Runs an OpenEvolve adversarial run, validates the adversarial findings with
    evolution validation, then captures the resulting knowledge.
  `;
  static readonly alias = 'openevolve-adversarial-to-evolution';

  constructor(
    params: OpenEvolveAdversarialToEvolutionParams,
    context?: BubbleContext
  ) {
    super(params, context);
  }

  protected async performAction(): Promise<OpenEvolveAdversarialToEvolutionResult> {
    const startTime = Date.now();
    const steps: unknown[] = [];
    try {
      const factory = await BubbleFactory.getInstance();

      const runRes = await factory
        .createBubble('openevolve-adversarial-run' as unknown as BubbleName, {
          operation: 'start',
          problem_statement: this.params.problem_statement,
          config: this.params.config,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();
      steps.push({ step: 'adversarial-run', result: runRes.data });
      if (!runRes.success) {
        return this.fail(`adversarial-run failed: ${runRes.error}`, steps, startTime);
      }
      const runData = (runRes.data ?? {}) as any;
      const runId = this.params.run_id ?? runData?.run_id ?? runData?.id;
      const code = this.params.evolved_code ?? JSON.stringify(runData ?? {});

      const validationRes = await factory
        .createBubble('openevolve-evolution-validation' as unknown as BubbleName, {
          operation: 'validate',
          code,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();
      steps.push({ step: 'evolution-validation', result: validationRes.data });
      if (!validationRes.success) {
        return this.fail(`evolution-validation failed: ${validationRes.error}`, steps, startTime);
      }

      const captureRes = await factory
        .createBubble('openevolve-knowledge-capture' as unknown as BubbleName, {
          operation: 'capture',
          run_id: runId,
          source: 'adversarial',
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();
      steps.push({ step: 'knowledge-capture', result: captureRes.data });

      return {
        success: true,
        operation: 'adversarial-to-evolution',
        data: {
          run: runRes.data,
          validation: validationRes.data,
          knowledge: captureRes.data,
        },
        steps,
        timing: { total: Date.now() - startTime },
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: 'adversarial-to-evolution',
        error: message,
        steps,
        timing: { total: Date.now() - startTime },
      };
    }
  }

  private fail(error: string, steps: unknown[], startTime: number): OpenEvolveAdversarialToEvolutionResult {
    return {
      success: false,
      operation: 'adversarial-to-evolution',
      error,
      steps,
      timing: { total: Date.now() - startTime },
    };
  }
}

export default OpenEvolveAdversarialToEvolutionBubble;
