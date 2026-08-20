import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const ExecutionOperationSchema = z.enum([
  'start',
  'status',
  'pause',
  'resume',
  'cancel',
  'logs',
  'list',
  'health',
]);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const BaseParamsSchema = z.object({
  operation: ExecutionOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().int().min(1000).max(300000).default(60000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),
});

const StartExecutionSchema = BaseParamsSchema.extend({
  operation: z.literal('start'),
  workflow_id: z.string().min(1),
  problem_statement: z.string().optional(),
  context: z.string().optional(),
});

const StatusExecutionSchema = BaseParamsSchema.extend({
  operation: z.literal('status'),
  execution_id: z.string().min(1),
});

const PauseExecutionSchema = BaseParamsSchema.extend({
  operation: z.literal('pause'),
  execution_id: z.string().min(1),
});

const ResumeExecutionSchema = BaseParamsSchema.extend({
  operation: z.literal('resume'),
  execution_id: z.string().min(1),
});

const CancelExecutionSchema = BaseParamsSchema.extend({
  operation: z.literal('cancel'),
  execution_id: z.string().min(1),
});

const LogsExecutionSchema = BaseParamsSchema.extend({
  operation: z.literal('logs'),
  execution_id: z.string().min(1),
  since: z.string().optional(),
});

const ListExecutionSchema = BaseParamsSchema.extend({
  operation: z.literal('list'),
  workflow_id: z.string().min(1),
  limit: z.number().int().min(1).max(100).optional(),
});

const HealthSchema = BaseParamsSchema.extend({
  operation: z.literal('health'),
});

const OpenEvolveExecutionParamsSchema = z.discriminatedUnion('operation', [
  StartExecutionSchema,
  StatusExecutionSchema,
  PauseExecutionSchema,
  ResumeExecutionSchema,
  CancelExecutionSchema,
  LogsExecutionSchema,
  ListExecutionSchema,
  HealthSchema,
]);

type OpenEvolveExecutionParams = z.input<typeof OpenEvolveExecutionParamsSchema>;

const OpenEvolveExecutionResultSchema = z.object({
  success: z.boolean(),
  status: z.number().optional(),
  operation: z.string(),
  workflow_id: z.string().optional(),
  execution_id: z.string().optional(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type OpenEvolveExecutionResult = z.output<typeof OpenEvolveExecutionResultSchema>;

export class OpenEvolveExecutionBubble extends ServiceBubble<
  OpenEvolveExecutionParams,
  OpenEvolveExecutionResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-execution' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = OpenEvolveExecutionParamsSchema;
  static readonly resultSchema = OpenEvolveExecutionResultSchema;
  static readonly shortDescription =
    'OpenEvolve execution control (start, pause, logs, status)';
  static readonly longDescription = `
    OpenEvolve execution bubble for managing workflow runs.

    Operations:
    - start: start a workflow execution
    - status: get execution status
    - pause/resume/cancel: control execution lifecycle
    - logs: fetch execution logs
    - list: list executions for a workflow
  `;
  static readonly alias = 'openevolve-execution';

  constructor(
    params: OpenEvolveExecutionParams,
    context?: BubbleContext,
    instanceId?: string
  ) {
    super(params, context, instanceId);
  }

  protected chooseCredential(): string | undefined {
    return undefined;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<OpenEvolveExecutionResult> {
    const startTime = Date.now();
    const params = this.params;

    try {
      switch (params.operation) {
        case 'start':
          return await this.request(
            'POST',
            '/api/executions',
            {
              workflow_id: params.workflow_id,
              problem_statement: params.problem_statement,
              context: params.context,
            },
            startTime
          );
        case 'status':
          return await this.request(
            'GET',
            `/api/executions/${params.execution_id}`,
            undefined,
            startTime
          );
        case 'pause':
          return await this.request(
            'POST',
            `/api/executions/${params.execution_id}/pause`,
            {},
            startTime
          );
        case 'resume':
          return await this.request(
            'POST',
            `/api/executions/${params.execution_id}/resume`,
            {},
            startTime
          );
        case 'cancel':
          return await this.request(
            'POST',
            `/api/executions/${params.execution_id}/cancel`,
            {},
            startTime
          );
        case 'logs': {
          const query = new URLSearchParams();
          if (params.since) query.set('since', params.since);
          const endpoint = query.toString()
            ? `/api/executions/${params.execution_id}/logs?${query.toString()}`
            : `/api/executions/${params.execution_id}/logs`;
          return await this.request('GET', endpoint, undefined, startTime);
        }
        case 'list': {
          const query = new URLSearchParams();
          if (params.limit) query.set('limit', String(params.limit));
          const endpoint = query.toString()
            ? `/api/executions/workflows/${params.workflow_id}/executions?${query.toString()}`
            : `/api/executions/workflows/${params.workflow_id}/executions`;
          return await this.request('GET', endpoint, undefined, startTime);
        }
        case 'health':
          return await this.request('GET', '/health', undefined, startTime);
        default:
          return {
            success: false,
            status: 400,
            operation: params.operation,
            error: `Unsupported operation: ${params.operation}`,
            timing: Date.now() - startTime,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        status: 500,
        operation: params.operation,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }

  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.params.headers) {
      Object.assign(headers, this.params.headers);
    }
    if (this.params.auth_token) {
      const headerName = this.params.auth_header || 'Authorization';
      const token = this.params.auth_token;
      if (headerName.toLowerCase() === 'authorization' && !token.startsWith('Bearer ')) {
        headers[headerName] = `Bearer ${token}`;
      } else {
        headers[headerName] = token;
      }
    }
    return headers;
  }

  private async request(
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    endpoint: string,
    body: unknown,
    startTime: number
  ): Promise<OpenEvolveExecutionResult> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.params.timeout);
    const url = `${this.params.base_url}${endpoint}`;

    try {
      const response = await fetch(url, {
        method,
        headers: this.buildHeaders(),
        body: body !== undefined && method !== 'GET' ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      const text = await response.text();
      let data: unknown = undefined;
      if (text) {
        try {
          data = JSON.parse(text);
        } catch {
          data = text;
        }
      }

      return {
        success: response.ok,
        status: response.status,
        operation: this.params.operation,
        workflow_id:
          (this.params as any).workflow_id ||
          (typeof data === 'object' && data && 'workflow_id' in data
            ? String((data as any).workflow_id)
            : undefined),
        execution_id:
          (this.params as any).execution_id ||
          (typeof data === 'object' && data && 'execution_id' in data
            ? String((data as any).execution_id)
            : undefined),
        data,
        error: response.ok
          ? undefined
          : typeof data === 'object' && data && 'detail' in data
            ? String((data as any).detail)
            : response.statusText,
        timing: Date.now() - startTime,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        status: 0,
        operation: this.params.operation,
        execution_id: (this.params as any).execution_id,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }
}

export default OpenEvolveExecutionBubble;
