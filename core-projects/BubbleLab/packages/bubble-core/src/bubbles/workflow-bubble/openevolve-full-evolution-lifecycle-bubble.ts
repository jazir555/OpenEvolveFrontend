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

const OpenEvolveFullEvolutionLifecycleParamsSchema = z
  .object({
    problem_statement: z.string().optional(),
    workflow_type: z.string().default('evolution'),
    evolved_code: z.string().optional(),
    target_system: z.string().optional(),
    target_path: z.string().optional(),
    run_id: z.string().optional(),
    base_url: z.string().url().default(resolveBaseUrl()),
    auth_token: z.string().optional(),
    auto_deploy: z.boolean().default(true),
  })
  .passthrough();

type OpenEvolveFullEvolutionLifecycleParams = z.input<
  typeof OpenEvolveFullEvolutionLifecycleParamsSchema
> &
  ServiceBubbleParams;

const OpenEvolveFullEvolutionLifecycleResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  steps: z.array(z.unknown()).optional(),
  error: z.string().optional(),
  timing: z.object({ total: z.number() }),
});

type OpenEvolveFullEvolutionLifecycleResult = z.output<
  typeof OpenEvolveFullEvolutionLifecycleResultSchema
> &
  BubbleOperationResult;

export class OpenEvolveFullEvolutionLifecycleBubble extends WorkflowBubble<
  OpenEvolveFullEvolutionLifecycleParams,
  OpenEvolveFullEvolutionLifecycleResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-full-evolution-lifecycle' as BubbleName;
  static readonly type = 'workflow' as const;
  static readonly schema = OpenEvolveFullEvolutionLifecycleParamsSchema;
  static readonly resultSchema = OpenEvolveFullEvolutionLifecycleResultSchema;
  static readonly shortDescription = 'OpenEvolve full evolution lifecycle workflow';
  static readonly longDescription = `
    Complete evolution lifecycle: trigger an evolution, validate it, apply it,
    collect metrics, and capture knowledge.
  `;
  static readonly alias = 'openevolve-full-evolution-lifecycle';

  constructor(
    params: OpenEvolveFullEvolutionLifecycleParams,
    context?: BubbleContext
  ) {
    super(params, context);
  }

  protected async performAction(): Promise<OpenEvolveFullEvolutionLifecycleResult> {
    const startTime = Date.now();
    const steps: unknown[] = [];
    try {
      const factory = await BubbleFactory.getInstance();

      const triggerRes = await factory
        .createBubble('openevolve-evolution-trigger' as unknown as BubbleName, {
          operation: 'create_and_run',
          problem_statement: this.params.problem_statement,
          workflow_type: this.params.workflow_type,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();
      steps.push({ step: 'evolution-trigger', result: triggerRes.data });
      if (!triggerRes.success) {
        return this.fail(`evolution-trigger failed: ${triggerRes.error}`, steps, startTime);
      }
      const triggerData = (triggerRes.data ?? {}) as any;
      const code = this.params.evolved_code ?? JSON.stringify(triggerData.raw ?? triggerData);

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

      const applicationRes = await factory
        .createBubble('openevolve-evolution-application' as unknown as BubbleName, {
          operation: 'apply',
          workflow_id: triggerData.workflow_id,
          evolved_code: code,
          target_system: this.params.target_system,
          target_path: this.params.target_path,
          auto_deploy: this.params.auto_deploy,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();
      steps.push({ step: 'evolution-application', result: applicationRes.data });
      if (!applicationRes.success) {
        return this.fail(`evolution-application failed: ${applicationRes.error}`, steps, startTime);
      }

      const metricsRes = await factory
        .createBubble('openevolve-metrics-collector' as unknown as BubbleName, {
          operation: 'collect',
          workflow_id: triggerData.workflow_id,
          run_id: this.params.run_id,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();
      steps.push({ step: 'metrics-collector', result: metricsRes.data });

      const knowledgeRes = await factory
        .createBubble('openevolve-knowledge-capture' as unknown as BubbleName, {
          operation: 'capture',
          workflow_id: triggerData.workflow_id,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();
      steps.push({ step: 'knowledge-capture', result: knowledgeRes.data });

      return {
        success: true,
        operation: 'full-evolution-lifecycle',
        data: {
          trigger: triggerRes.data,
          validation: validationRes.data,
          application: applicationRes.data,
          metrics: metricsRes.data,
          knowledge: knowledgeRes.data,
        },
        steps,
        timing: { total: Date.now() - startTime },
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: 'full-evolution-lifecycle',
        error: message,
        steps,
        timing: { total: Date.now() - startTime },
      };
    }
  }

  private fail(error: string, steps: unknown[], startTime: number): OpenEvolveFullEvolutionLifecycleResult {
    return {
      success: false,
      operation: 'full-evolution-lifecycle',
      error,
      steps,
      timing: { total: Date.now() - startTime },
    };
  }
}

export default OpenEvolveFullEvolutionLifecycleBubble;
