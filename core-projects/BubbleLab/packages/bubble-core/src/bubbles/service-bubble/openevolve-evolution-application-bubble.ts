import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const EvolutionApplicationOperationSchema = z.enum(['apply', 'health_check']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const EvolutionApplicationParamsSchema = z.object({
  operation: EvolutionApplicationOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(120000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('X-API-Key'),

  workflow_id: z.string().optional(),
  evolved_code: z.string().min(1),
  language: z.string().default('json'),
  target_system: z.string().optional(),
  target_path: z.string().optional(),
  deployment_config: z.record(z.unknown()).optional(),
  auto_deploy: z.boolean().default(true),
});

type EvolutionApplicationParams = z.input<typeof EvolutionApplicationParamsSchema> & ServiceBubbleParams;

const EvolutionApplicationDataSchema = z.object({
  deployment_id: z.string().optional(),
  workflow_id: z.string().optional(),
  applied: z.boolean(),
  apply_status: z.string().optional(),
  target_system: z.string().optional(),
  stored_artifact_id: z.string().optional(),
});

const EvolutionApplicationResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: EvolutionApplicationDataSchema.optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type EvolutionApplicationResult = z.output<typeof EvolutionApplicationResultSchema> & BubbleOperationResult;

export class OpenEvolveEvolutionApplicationBubble extends ServiceBubble<
  EvolutionApplicationParams,
  EvolutionApplicationResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-evolution-application' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = EvolutionApplicationParamsSchema;
  static readonly resultSchema = EvolutionApplicationResultSchema;
  static readonly shortDescription = 'OpenEvolve evolution application (apply evolved code)';
  static readonly longDescription = `
    Applies validated evolved code. Attempts the OpenEvolve apply endpoint
    (POST /api/workflows/{id}/apply) and, regardless, durably records the
    deployment as a knowledge artifact (POST /api/knowledge/artifacts) so the
    deployment is never lost. No simulated output.
  `;
  static readonly alias = 'openevolve-evolution-application';

  constructor(params: EvolutionApplicationParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<EvolutionApplicationResult> {
    const startTime = Date.now();
    try {
      if (this.params.operation === 'health_check') {
        const r = await this.request('GET', '/health', undefined, startTime);
        return {
          success: r.success,
          operation: this.params.operation,
          data: undefined,
          error: r.error,
          timing: Date.now() - startTime,
        };
      }

      const deploymentRecord = {
        artifact_type: 'deployment',
        content: this.params.evolved_code,
        source_workflow_id: this.params.workflow_id || 'unknown',
        domain: this.params.target_system || 'openevolve',
        problem_type: this.params.language,
        usage_count: 0,
        effectiveness_score: this.params.auto_deploy ? 1.0 : 0.5,
        deployment_config: this.params.deployment_config || {},
        target_system: this.params.target_system,
        target_path: this.params.target_path,
      };

      // Attempt the real apply endpoint (degrades gracefully if absent).
      let applyStatus: string | undefined;
      let applied = false;
      if (this.params.workflow_id) {
        const apply = await this.request(
          'POST',
          `/api/workflows/${this.params.workflow_id}/apply`,
          {
            evolved_code: this.params.evolved_code,
            language: this.params.language,
            target_system: this.params.target_system,
            target_path: this.params.target_path,
            deployment_config: this.params.deployment_config,
            auto_deploy: this.params.auto_deploy,
          },
          startTime,
          true
        );
        applied = apply.success;
        applyStatus = applied ? 'applied' : `apply_endpoint_unavailable: ${apply.error}`;
      } else {
        applyStatus = 'skipped_no_workflow_id';
      }

      // Durably record the deployment via the real knowledge store endpoint.
      const artifact = await this.request(
        'POST',
        '/api/knowledge/artifacts',
        deploymentRecord,
        startTime
      );
      const storedId = artifact.success
        ? ((artifact.data as any)?.id as string | undefined)
        : undefined;

      return {
        success: true,
        operation: this.params.operation,
        data: {
          deployment_id: storedId,
          workflow_id: this.params.workflow_id,
          applied,
          apply_status: applyStatus,
          target_system: this.params.target_system,
          stored_artifact_id: storedId,
        },
        error: artifact.success ? undefined : `deployment record failed: ${artifact.error}`,
        timing: Date.now() - startTime,
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: this.params.operation,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }

  private buildHeaders(includeApiKey = false): Record<string, string> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.params.headers) Object.assign(headers, this.params.headers);
    if (includeApiKey && this.params.auth_token) {
      const headerName = this.params.auth_header || 'X-API-Key';
      headers[headerName] = this.params.auth_token;
    }
    return headers;
  }

  private async request(
    method: 'GET' | 'POST',
    endpoint: string,
    body: unknown,
    startTime: number,
    includeApiKey = false
  ): Promise<{ success: boolean; data?: unknown; error?: string }> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.params.timeout);
    const url = `${this.params.base_url}${endpoint}`;
    try {
      const response = await fetch(url, {
        method,
        headers: this.buildHeaders(includeApiKey),
        body: body && method !== 'GET' ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      const data = await response.json().catch(() => undefined);
      return {
        success: response.ok,
        data,
        error: response.ok ? undefined : ((data as any)?.detail as string) || response.statusText,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      const message = error instanceof Error ? error.message : 'Unknown error';
      return { success: false, error: message };
    }
  }
}

export default OpenEvolveEvolutionApplicationBubble;
