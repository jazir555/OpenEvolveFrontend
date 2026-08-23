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

const SovereignPipelineParamsSchema = z
  .object({
    problem_statement: z.string().min(1),
    title: z.string().optional(),
    strategy: z.string().optional(),
    base_url: z.string().url().default(resolveBaseUrl()),
    auth_token: z.string().optional(),
  })
  .passthrough();

type SovereignPipelineParams = z.input<typeof SovereignPipelineParamsSchema> & ServiceBubbleParams;

const SovereignPipelineResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z
    .object({
      status: z.unknown().optional(),
      run: z.unknown().optional(),
      problems: z.unknown().optional(),
    })
    .optional(),
  error: z.string().optional(),
  timing: z.object({ total: z.number() }),
});

type SovereignPipelineResult = z.output<typeof SovereignPipelineResultSchema> & BubbleOperationResult;

export class OpenEvolveSovereignPipelineBubble extends WorkflowBubble<
  SovereignPipelineParams,
  SovereignPipelineResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'sovereign-pipeline' as BubbleName;
  static readonly type = 'workflow' as const;
  static readonly schema = SovereignPipelineParamsSchema;
  static readonly resultSchema = SovereignPipelineResultSchema;
  static readonly shortDescription =
    'OpenEvolve sovereign pipeline (status -> run -> problems)';
  static readonly longDescription = `
    End-to-end Sovereign workflow: checks subsystem health, analyzes and persists
    the problem, then lists the stored problems.
  `;
  static readonly alias = 'sovereign-pipeline';

  constructor(params: SovereignPipelineParams, context?: BubbleContext) {
    super(params, context);
  }

  protected async performAction(): Promise<SovereignPipelineResult> {
    const startTime = Date.now();
    try {
      const factory = await BubbleFactory.getInstance();

      const statusRes = await factory
        .createBubble('sovereign-status', {
          operation: 'status',
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();

      const runRes = await factory
        .createBubble('sovereign-run', {
          operation: 'run',
          problem_statement: this.params.problem_statement,
          ...(this.params.title !== undefined ? { title: this.params.title } : {}),
          ...(this.params.strategy !== undefined
            ? { strategy: this.params.strategy }
            : {}),
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();
      if (!runRes.success) {
        return {
          success: false,
          operation: 'pipeline',
          error: `run failed: ${runRes.error}`,
          timing: { total: Date.now() - startTime },
        };
      }

      const problemsRes = await factory
        .createBubble('sovereign-problems', {
          operation: 'problems',
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();

      return {
        success: true,
        operation: 'pipeline',
        data: {
          status: statusRes.data,
          run: runRes.data,
          problems: problemsRes.data,
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

export default OpenEvolveSovereignPipelineBubble;
