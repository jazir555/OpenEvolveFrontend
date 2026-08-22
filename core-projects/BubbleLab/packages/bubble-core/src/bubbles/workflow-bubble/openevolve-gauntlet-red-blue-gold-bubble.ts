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

const OpenEvolveGauntletRedBlueGoldParamsSchema = z
  .object({
    gauntlet_name: z.string().min(1),
    config: z.record(z.unknown()).optional(),
    rounds: z.number().int().min(1).optional(),
    difficulty: z.string().optional(),
    execution_id: z.string().optional(),
    base_url: z.string().url().default(resolveBaseUrl()),
    auth_token: z.string().optional(),
  })
  .passthrough();

type OpenEvolveGauntletRedBlueGoldParams = z.input<
  typeof OpenEvolveGauntletRedBlueGoldParamsSchema
> &
  ServiceBubbleParams;

const OpenEvolveGauntletRedBlueGoldResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  steps: z.array(z.unknown()).optional(),
  error: z.string().optional(),
  timing: z.object({ total: z.number() }),
});

type OpenEvolveGauntletRedBlueGoldResult = z.output<
  typeof OpenEvolveGauntletRedBlueGoldResultSchema
> &
  BubbleOperationResult;

export class OpenEvolveGauntletRedBlueGoldBubble extends WorkflowBubble<
  OpenEvolveGauntletRedBlueGoldParams,
  OpenEvolveGauntletRedBlueGoldResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-gauntlet-red-blue-gold' as BubbleName;
  static readonly type = 'workflow' as const;
  static readonly schema = OpenEvolveGauntletRedBlueGoldParamsSchema;
  static readonly resultSchema = OpenEvolveGauntletRedBlueGoldResultSchema;
  static readonly shortDescription = 'OpenEvolve gauntlet red/blue/gold workflow';
  static readonly longDescription = `
    Runs an OpenEvolve gauntlet (red), analyzes with ACE tools (blue), gold-verifies
    the result with evolution validation, then reports execution status.
  `;
  static readonly alias = 'openevolve-gauntlet-red-blue-gold';

  constructor(
    params: OpenEvolveGauntletRedBlueGoldParams,
    context?: BubbleContext
  ) {
    super(params, context);
  }

  protected async performAction(): Promise<OpenEvolveGauntletRedBlueGoldResult> {
    const startTime = Date.now();
    const steps: unknown[] = [];
    try {
      const factory = await BubbleFactory.getInstance();

      const executeRes = await factory
        .createBubble('openevolve-gauntlet-execute' as unknown as BubbleName, {
          operation: 'execute',
          gauntlet_name: this.params.gauntlet_name,
          config: this.params.config,
          rounds: this.params.rounds,
          difficulty: this.params.difficulty,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();
      steps.push({ step: 'gauntlet-execute', result: executeRes.data });
      if (!executeRes.success) {
        return this.fail(`gauntlet-execute failed: ${executeRes.error}`, steps, startTime);
      }
      const execData = (executeRes.data ?? {}) as any;
      const executionId = this.params.execution_id ?? execData?.execution_id ?? execData?.id;

      const aceRes = await factory
        .createBubble('openevolve-ace-tools' as unknown as BubbleName, {
          operation: 'analyze',
          execution_id: executionId,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();
      steps.push({ step: 'ace-tools', result: aceRes.data });
      if (!aceRes.success) {
        return this.fail(`ace-tools failed: ${aceRes.error}`, steps, startTime);
      }

      const goldRes = await factory
        .createBubble('openevolve-evolution-validation' as unknown as BubbleName, {
          operation: 'validate',
          code: JSON.stringify(aceRes.data ?? {}),
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();
      steps.push({ step: 'evolution-validation-gold', result: goldRes.data });
      if (!goldRes.success) {
        return this.fail(`gold verification failed: ${goldRes.error}`, steps, startTime);
      }

      const statusRes = await factory
        .createBubble('openevolve-gauntlet-execution-status' as unknown as BubbleName, {
          operation: 'status',
          execution_id: executionId,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();
      steps.push({ step: 'gauntlet-execution-status', result: statusRes.data });

      return {
        success: true,
        operation: 'gauntlet-red-blue-gold',
        data: {
          execute: executeRes.data,
          ace: aceRes.data,
          gold: goldRes.data,
          status: statusRes.data,
        },
        steps,
        timing: { total: Date.now() - startTime },
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: 'gauntlet-red-blue-gold',
        error: message,
        steps,
        timing: { total: Date.now() - startTime },
      };
    }
  }

  private fail(error: string, steps: unknown[], startTime: number): OpenEvolveGauntletRedBlueGoldResult {
    return {
      success: false,
      operation: 'gauntlet-red-blue-gold',
      error,
      steps,
      timing: { total: Date.now() - startTime },
    };
  }
}

export default OpenEvolveGauntletRedBlueGoldBubble;
