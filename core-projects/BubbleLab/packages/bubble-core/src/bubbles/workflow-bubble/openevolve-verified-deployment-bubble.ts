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

const OpenEvolveVerifiedDeploymentParamsSchema = z
  .object({
    evolved_code: z.string().optional(),
    workflow_id: z.string().optional(),
    target_system: z.string().optional(),
    target_path: z.string().optional(),
    auto_deploy: z.boolean().default(true),
    base_url: z.string().url().default(resolveBaseUrl()),
    auth_token: z.string().optional(),
  })
  .passthrough();

type OpenEvolveVerifiedDeploymentParams = z.input<
  typeof OpenEvolveVerifiedDeploymentParamsSchema
> &
  ServiceBubbleParams;

const OpenEvolveVerifiedDeploymentResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  steps: z.array(z.unknown()).optional(),
  error: z.string().optional(),
  timing: z.object({ total: z.number() }),
});

type OpenEvolveVerifiedDeploymentResult = z.output<
  typeof OpenEvolveVerifiedDeploymentResultSchema
> &
  BubbleOperationResult;

export class OpenEvolveVerifiedDeploymentBubble extends WorkflowBubble<
  OpenEvolveVerifiedDeploymentParams,
  OpenEvolveVerifiedDeploymentResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-verified-deployment' as BubbleName;
  static readonly type = 'workflow' as const;
  static readonly schema = OpenEvolveVerifiedDeploymentParamsSchema;
  static readonly resultSchema = OpenEvolveVerifiedDeploymentResultSchema;
  static readonly shortDescription = 'OpenEvolve verified deployment workflow';
  static readonly longDescription = `
    Validates evolved code, applies it, then runs a security check before
    declaring the deployment verified.
  `;
  static readonly alias = 'openevolve-verified-deployment';

  constructor(params: OpenEvolveVerifiedDeploymentParams, context?: BubbleContext) {
    super(params, context);
  }

  protected async performAction(): Promise<OpenEvolveVerifiedDeploymentResult> {
    const startTime = Date.now();
    const steps: unknown[] = [];
    try {
      const factory = await BubbleFactory.getInstance();

      const validationRes = await factory
        .createBubble('openevolve-evolution-validation' as unknown as BubbleName, {
          operation: 'validate',
          code: this.params.evolved_code,
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
          workflow_id: this.params.workflow_id,
          evolved_code: this.params.evolved_code,
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

      const securityRes = await factory
        .createBubble('openevolve-security' as unknown as BubbleName, {
          operation: 'scan',
          workflow_id: this.params.workflow_id,
          target_system: this.params.target_system,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        })
        .action();
      steps.push({ step: 'security', result: securityRes.data });

      return {
        success: true,
        operation: 'verified-deployment',
        data: {
          validation: validationRes.data,
          application: applicationRes.data,
          security: securityRes.data,
        },
        steps,
        timing: { total: Date.now() - startTime },
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: 'verified-deployment',
        error: message,
        steps,
        timing: { total: Date.now() - startTime },
      };
    }
  }

  private fail(error: string, steps: unknown[], startTime: number): OpenEvolveVerifiedDeploymentResult {
    return {
      success: false,
      operation: 'verified-deployment',
      error,
      steps,
      timing: { total: Date.now() - startTime },
    };
  }
}

export default OpenEvolveVerifiedDeploymentBubble;
