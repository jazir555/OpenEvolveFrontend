import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const EvolutionTriggerOperationSchema = z.enum(['create_and_run', 'health_check']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const EvolutionTriggerParamsSchema = z.object({
  operation: EvolutionTriggerOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('X-API-Key'),

  problem_statement: z.string().min(1),
  workflow_type: z.string().default('evolution'),
  description: z.string().optional(),
  iterations: z.number().int().min(1).max(10000).default(100),
  population_size: z.number().int().min(1).max(1000).default(50),
  teams: z.array(z.string()).optional(),
  gauntlets: z.array(z.string()).optional(),
  config: z.record(z.unknown()).optional(),
});

type EvolutionTriggerParams = z.input<typeof EvolutionTriggerParamsSchema> & ServiceBubbleParams;

const EvolutionTriggerDataSchema = z.object({
  workflow_id: z.string().optional(),
  status: z.string().optional(),
  engine_workflow_id: z.string().optional(),
  raw: z.unknown().optional(),
});

const EvolutionTriggerResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: EvolutionTriggerDataSchema.optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type EvolutionTriggerResult = z.output<typeof EvolutionTriggerResultSchema> & BubbleOperationResult;

export class OpenEvolveEvolutionTriggerBubble extends ServiceBubble<
  EvolutionTriggerParams,
  EvolutionTriggerResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-evolution-trigger' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = EvolutionTriggerParamsSchema;
  static readonly resultSchema = EvolutionTriggerResultSchema;
  static readonly shortDescription = 'OpenEvolve evolution trigger (create + run workflow)';
  static readonly longDescription = `
    Triggers an OpenEvolve evolution workflow via the OpenEvolve API:
    POST /api/workflows (create) then POST /api/workflows/{id}/run (with X-API-Key header).
  `;
  static readonly alias = 'openevolve-evolution-trigger';

  constructor(params: EvolutionTriggerParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<EvolutionTriggerResult> {
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

      const createBody = {
        name: `Evolution: ${this.params.problem_statement.slice(0, 50)}`,
        description: this.params.description || this.params.problem_statement,
        problem_statement: this.params.problem_statement,
        workflow_type: this.params.workflow_type,
        parameters: {
          iterations: this.params.iterations,
          populationSize: this.params.population_size,
          ...(this.params.config || {}),
        },
        teams: this.params.teams,
        gauntlets: this.params.gauntlets,
      };
      const created = await this.request('POST', '/api/workflows', createBody, startTime);
      if (!created.success || !created.data) {
        return {
          success: false,
          operation: this.params.operation,
          error: created.error || 'Workflow creation failed',
          timing: Date.now() - startTime,
        };
      }
      const workflowId =
        (created.data as any).id ?? (created.data as any).workflow_id ?? undefined;
      if (!workflowId) {
        return {
          success: false,
          operation: this.params.operation,
          error: 'Workflow creation did not return an id',
          data: { workflow_id: undefined, status: 'created_no_id', engine_workflow_id: undefined, raw: created.data },
          timing: Date.now() - startTime,
        };
      }

      const runBody = { config: this.params.config || {} };
      const run = await this.request(
        'POST',
        `/api/workflows/${workflowId}/run`,
        runBody,
        startTime,
        true
      );

      const engineWorkflowId = run.success
        ? ((run.data as any)?.workflow_id ?? (run.data as any)?.engine_workflow_id ?? undefined)
        : undefined;

      return {
        success: true,
        operation: this.params.operation,
        data: {
          workflow_id: workflowId,
          status: run.success ? 'running' : 'run_failed',
          engine_workflow_id: engineWorkflowId,
          raw: run.success ? run.data : { error: run.error },
        },
        error: run.success ? undefined : run.error,
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

export default OpenEvolveEvolutionTriggerBubble;
