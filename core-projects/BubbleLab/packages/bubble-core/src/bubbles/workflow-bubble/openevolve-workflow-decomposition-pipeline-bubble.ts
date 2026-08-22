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

const OpenEvolveWorkflowDecompositionPipelineParamsSchema = z
  .object({
    problem_statement: z.string().optional(),
    plan_id: z.string().optional(),
    workflow_id: z.string().optional(),
    base_url: z.string().url().default(resolveBaseUrl()),
    auth_token: z.string().optional(),
  })
  .passthrough();

type OpenEvolveWorkflowDecompositionPipelineParams = z.input<
  typeof OpenEvolveWorkflowDecompositionPipelineParamsSchema
> &
  ServiceBubbleParams;

const OpenEvolveWorkflowDecompositionPipelineResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  steps: z.array(z.unknown()).optional(),
  error: z.string().optional(),
  timing: z.object({ total: z.number() }),
});

type OpenEvolveWorkflowDecompositionPipelineResult = z.output<
  typeof OpenEvolveWorkflowDecompositionPipelineResultSchema
> &
  BubbleOperationResult;

export class OpenEvolveWorkflowDecompositionPipelineBubble extends WorkflowBubble<
  OpenEvolveWorkflowDecompositionPipelineParams,
  OpenEvolveWorkflowDecompositionPipelineResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-workflow-decomposition-pipeline' as BubbleName;
  static readonly type = 'workflow' as const;
  static readonly schema = OpenEvolveWorkflowDecompositionPipelineParamsSchema;
  static readonly resultSchema = OpenEvolveWorkflowDecompositionPipelineResultSchema;
  static readonly shortDescription = 'OpenEvolve workflow decomposition pipeline';
  static readonly longDescription = `
    Decomposes a problem into a plan, creates a workflow from it, runs the
    workflow, collects decomposition results, then captures knowledge.
  `;
  static readonly alias = 'openevolve-workflow-decomposition-pipeline';

  constructor(
    params: OpenEvolveWorkflowDecompositionPipelineParams,
    context?: BubbleContext
  ) {
    super(params, context);
  }

  protected async performAction(): Promise<OpenEvolveWorkflowDecompositionPipelineResult> {
    const startTime = Date.now();
    const steps: unknown[] = [];
    try {
      const factory = await BubbleFactory.getInstance();

      const planRes = await factory
        .createBubble('openevolve-decomposition' as unknown as BubbleName, {
          operation: 'plan',
          problem_statement: this.params.problem_statement,
          plan_id: this.params.plan_id,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();
      steps.push({ step: 'decomposition-plan', result: planRes.data });
      if (!planRes.success) {
        return this.fail(`decomposition-plan failed: ${planRes.error}`, steps, startTime);
      }
      const planData = (planRes.data ?? {}) as any;
      const workflowId = this.params.workflow_id ?? planData?.workflow_id ?? planData?.id;

      const createRes = await factory
        .createBubble('openevolve-workflow' as unknown as BubbleName, {
          operation: 'create',
          workflow_id: workflowId,
          plan: planData,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();
      steps.push({ step: 'workflow-create', result: createRes.data });
      if (!createRes.success) {
        return this.fail(`workflow-create failed: ${createRes.error}`, steps, startTime);
      }

      const runRes = await factory
        .createBubble('openevolve-workflow' as unknown as BubbleName, {
          operation: 'run',
          workflow_id: workflowId,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();
      steps.push({ step: 'workflow-run', result: runRes.data });
      if (!runRes.success) {
        return this.fail(`workflow-run failed: ${runRes.error}`, steps, startTime);
      }

      const resultsRes = await factory
        .createBubble('openevolve-decomposition-results' as unknown as BubbleName, {
          operation: 'get',
          workflow_id: workflowId,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();
      steps.push({ step: 'decomposition-results', result: resultsRes.data });

      const knowledgeRes = await factory
        .createBubble('openevolve-knowledge-capture' as unknown as BubbleName, {
          operation: 'capture',
          workflow_id: workflowId,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();
      steps.push({ step: 'knowledge-capture', result: knowledgeRes.data });

      return {
        success: true,
        operation: 'workflow-decomposition-pipeline',
        data: {
          plan: planRes.data,
          create: createRes.data,
          run: runRes.data,
          results: resultsRes.data,
          knowledge: knowledgeRes.data,
        },
        steps,
        timing: { total: Date.now() - startTime },
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: 'workflow-decomposition-pipeline',
        error: message,
        steps,
        timing: { total: Date.now() - startTime },
      };
    }
  }

  private fail(error: string, steps: unknown[], startTime: number): OpenEvolveWorkflowDecompositionPipelineResult {
    return {
      success: false,
      operation: 'workflow-decomposition-pipeline',
      error,
      steps,
      timing: { total: Date.now() - startTime },
    };
  }
}

export default OpenEvolveWorkflowDecompositionPipelineBubble;
