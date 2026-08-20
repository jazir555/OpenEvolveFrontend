import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';
import {
  OpenEvolveParametersSchema,
  OpenEvolveFlatParametersSchema,
  type OpenEvolveParameters,
  type OpenEvolveFlatParameters,
} from '../../integrations/openevolve/parameter-schema.js';

const WorkflowTypeSchema = z.enum(['evolution', 'adversarial', 'sovereign']);

const WorkflowOperationSchema = z.enum([
  'create',
  'list',
  'get',
  'update',
  'delete',
  'start',
  'pause',
  'resume',
  'stop',
  'results',
  'health',
]);

const WorkflowMetadataSchema = z
  .object({
    mdap_enabled: z.boolean().optional(),
    maker_enabled: z.boolean().optional(),
    maker_config: z.record(z.unknown()).optional(),
    adaptive_config: z.record(z.unknown()).optional(),
    evolution_params: OpenEvolveParametersSchema.optional(),
    performance_params: z.record(z.unknown()).optional(),
  })
  .partial()
  .passthrough();

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const BaseParamsSchema = z.object({
  operation: WorkflowOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().int().min(1000).max(300000).default(60000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),
});

const CreateWorkflowSchema = BaseParamsSchema.extend({
  operation: z.literal('create'),
  name: z.string().min(1),
  description: z.string().optional(),
  problem_statement: z.string().optional(),
  content_type: z.string().optional(),
  teams: z.array(z.string()).optional(),
  gauntlets: z.array(z.string()).optional(),
  workflow_type: WorkflowTypeSchema.default('sovereign'),
  metadata: WorkflowMetadataSchema.optional(),
  parameters: OpenEvolveParametersSchema.optional(),
  parameters_flat: OpenEvolveFlatParametersSchema.optional(),
});

const ListWorkflowSchema = BaseParamsSchema.extend({
  operation: z.literal('list'),
  page: z.number().int().min(1).optional(),
  page_size: z.number().int().min(1).max(100).optional(),
  workflow_type: WorkflowTypeSchema.optional(),
  status: z.string().optional(),
});

const GetWorkflowSchema = BaseParamsSchema.extend({
  operation: z.literal('get'),
  workflow_id: z.string().min(1),
});

const UpdateWorkflowSchema = BaseParamsSchema.extend({
  operation: z.literal('update'),
  workflow_id: z.string().min(1),
  name: z.string().min(1).optional(),
  description: z.string().optional(),
  problem_statement: z.string().optional(),
  content_type: z.string().optional(),
  teams: z.array(z.string()).optional(),
  gauntlets: z.array(z.string()).optional(),
  metadata: WorkflowMetadataSchema.optional(),
  parameters: OpenEvolveParametersSchema.optional(),
  parameters_flat: OpenEvolveFlatParametersSchema.optional(),
});

const DeleteWorkflowSchema = BaseParamsSchema.extend({
  operation: z.literal('delete'),
  workflow_id: z.string().min(1),
});

const ExecutionInputSchema = z.object({
  problem_statement: z.string().optional(),
  context: z.string().optional(),
});

const StartWorkflowSchema = BaseParamsSchema.extend({
  operation: z.literal('start'),
  workflow_id: z.string().min(1),
  inputs: ExecutionInputSchema.optional(),
});

const PauseWorkflowSchema = BaseParamsSchema.extend({
  operation: z.literal('pause'),
  workflow_id: z.string().min(1),
});

const ResumeWorkflowSchema = BaseParamsSchema.extend({
  operation: z.literal('resume'),
  workflow_id: z.string().min(1),
});

const StopWorkflowSchema = BaseParamsSchema.extend({
  operation: z.literal('stop'),
  workflow_id: z.string().min(1),
});

const ResultsWorkflowSchema = BaseParamsSchema.extend({
  operation: z.literal('results'),
  workflow_id: z.string().min(1),
});

const HealthWorkflowSchema = BaseParamsSchema.extend({
  operation: z.literal('health'),
});

const OpenEvolveWorkflowParamsSchema = z.discriminatedUnion('operation', [
  CreateWorkflowSchema,
  ListWorkflowSchema,
  GetWorkflowSchema,
  UpdateWorkflowSchema,
  DeleteWorkflowSchema,
  StartWorkflowSchema,
  PauseWorkflowSchema,
  ResumeWorkflowSchema,
  StopWorkflowSchema,
  ResultsWorkflowSchema,
  HealthWorkflowSchema,
]);

type OpenEvolveWorkflowParams = z.input<typeof OpenEvolveWorkflowParamsSchema> & ServiceBubbleParams;

const OpenEvolveWorkflowResultSchema = z.object({
  success: z.boolean(),
  status: z.number().optional(),
  operation: z.string(),
  workflow_id: z.string().optional(),
  execution_id: z.string().optional(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type OpenEvolveWorkflowResult = z.output<typeof OpenEvolveWorkflowResultSchema> & BubbleOperationResult;

const flattenNestedParameters = (
  nested: Record<string, unknown>
): Record<string, unknown> => {
  const flat: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(nested)) {
    if (value === undefined) continue;
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      for (const [innerKey, innerValue] of Object.entries(
        value as Record<string, unknown>
      )) {
        if (innerValue !== undefined) {
          flat[innerKey] = innerValue;
        }
      }
    } else {
      flat[key] = value;
    }
  }
  return flat;
};

const flattenFlatParameters = (
  flat: OpenEvolveFlatParameters | undefined
): Record<string, unknown> => {
  const output: Record<string, unknown> = {};
  if (!flat) return output;
  for (const [key, value] of Object.entries(flat)) {
    if (value === undefined) continue;
    const parts = key.split('.');
    const paramName = parts.length > 1 ? parts[1] : parts[0];
    output[paramName] = value;
  }
  return output;
};

export class OpenEvolveWorkflowBubble extends ServiceBubble<
  OpenEvolveWorkflowParams,
  OpenEvolveWorkflowResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-workflow' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = OpenEvolveWorkflowParamsSchema;
  static readonly resultSchema = OpenEvolveWorkflowResultSchema;
  static readonly shortDescription =
    'OpenEvolve workflow orchestration (create, execute, manage)';
  static readonly longDescription = `
    OpenEvolve workflow bubble for evolutionary, adversarial, and sovereign runs.

    Features:
    - Create/update workflows with full OpenEvolve parameter definitions
    - Start/pause/resume/stop executions
    - Fetch workflow results
    - Health checks and listing

    Parameters support:
    - Nested parameters (category-based)
    - Flat parameters (category.param) for UI knobs
  `;
  static readonly alias = 'openevolve-workflow';

  constructor(
    params: OpenEvolveWorkflowParams,
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

  protected async performAction(): Promise<OpenEvolveWorkflowResult> {
    const startTime = Date.now();
    const params = this.params;
    const op: string = params.operation;
    try {
      switch (params.operation) {
        case 'create':
          return await this.handleCreate(startTime);
        case 'list':
          return await this.handleList(startTime);
        case 'get':
          return await this.request('GET', `/api/workflows/${params.workflow_id}`, undefined, startTime);
        case 'update':
          return await this.handleUpdate(startTime);
        case 'delete':
          return await this.request('DELETE', `/api/workflows/${params.workflow_id}`, undefined, startTime);
        case 'start':
          return await this.request(
            'POST',
            `/api/workflows/${params.workflow_id}/start`,
            params.inputs ?? {},
            startTime
          );
        case 'pause':
          return await this.request(
            'POST',
            `/api/workflows/${params.workflow_id}/pause`,
            {},
            startTime
          );
        case 'resume':
          return await this.request(
            'POST',
            `/api/workflows/${params.workflow_id}/resume`,
            {},
            startTime
          );
        case 'stop':
          return await this.request(
            'POST',
            `/api/workflows/${params.workflow_id}/stop`,
            {},
            startTime
          );
        case 'results':
          return await this.request(
            'GET',
            `/api/workflows/${params.workflow_id}/results`,
            undefined,
            startTime
          );
        case 'health':
          return await this.request('GET', '/health', undefined, startTime);
        default:
          return {
            success: false,
            status: 400,
            operation: op,
            error: `Unsupported operation: ${op}`,
            timing: Date.now() - startTime,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        status: 500,
        operation: op,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }

  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if ((this.params as any).headers) {
      Object.assign(headers, (this.params as any).headers);
    }
    if ((this.params as any).auth_token) {
      const headerName = (this.params as any).auth_header || 'Authorization';
      const token = (this.params as any).auth_token;
      if (headerName.toLowerCase() === 'authorization' && !token.startsWith('Bearer ')) {
        headers[headerName] = `Bearer ${token}`;
      } else {
        headers[headerName] = token;
      }
    }
    return headers;
  }

  private buildParameters():
    | Record<string, unknown>
    | undefined {
    const nested = (this.params as any).parameters as OpenEvolveParameters | undefined;
    const flat = (this.params as any).parameters_flat as OpenEvolveFlatParameters | undefined;
    if (!nested && !flat) return undefined;
    const flattenedNested = nested ? flattenNestedParameters(nested as Record<string, unknown>) : {};
    const flattenedFlat = flattenFlatParameters(flat);
    return { ...flattenedNested, ...flattenedFlat };
  }

  private async handleCreate(startTime: number): Promise<OpenEvolveWorkflowResult> {
    const payload = {
      name: (this.params as any).name,
      description: (this.params as any).description,
      problem_statement: (this.params as any).problem_statement,
      content_type: (this.params as any).content_type,
      teams: (this.params as any).teams,
      gauntlets: (this.params as any).gauntlets,
      workflow_type: (this.params as any).workflow_type,
      metadata: (this.params as any).metadata,
      parameters: this.buildParameters(),
    };
    return this.request('POST', '/api/workflows', payload, startTime);
  }

  private async handleUpdate(startTime: number): Promise<OpenEvolveWorkflowResult> {
    const payload: Record<string, unknown> = {};
    const params = this.params as any;
    const op: string = params.operation;

    if (params.name !== undefined) payload.name = params.name;
    if (params.description !== undefined) payload.description = params.description;
    if (params.problem_statement !== undefined) payload.problem_statement = params.problem_statement;
    if (params.content_type !== undefined) payload.content_type = params.content_type;
    if (params.teams !== undefined) payload.teams = params.teams;
    if (params.gauntlets !== undefined) payload.gauntlets = params.gauntlets;
    if (params.metadata !== undefined) payload.metadata = params.metadata;

    const mergedParameters = this.buildParameters();
    if (mergedParameters) payload.parameters = mergedParameters;

    return this.request('PUT', `/api/workflows/${params.workflow_id}`, payload, startTime);
  }

  private async handleList(startTime: number): Promise<OpenEvolveWorkflowResult> {
    const params = this.params as any;
    const op: string = params.operation;
    const query = new URLSearchParams();
    if (params.page) query.set('page', String(params.page));
    if (params.page_size) query.set('page_size', String(params.page_size));
    if (params.workflow_type) query.set('workflow_type', params.workflow_type);
    if (params.status) query.set('status', params.status);

    const endpoint = query.toString()
      ? `/api/workflows?${query.toString()}`
      : '/api/workflows';
    return this.request('GET', endpoint, undefined, startTime);
  }

  private async request(
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    endpoint: string,
    body: unknown,
    startTime: number
  ): Promise<OpenEvolveWorkflowResult> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), (this.params as any).timeout);
    const url = `${(this.params as any).base_url}${endpoint}`;

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
        operation: (((this.params as any).operation as string) as string),
        workflow_id: (this.params as any).workflow_id,
        execution_id:
          typeof data === 'object' && data && 'execution_id' in data
            ? String((data as any).execution_id)
            : undefined,
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
        operation: (((this.params as any).operation as string) as string),
        workflow_id: (this.params as any).workflow_id,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }
}

export default OpenEvolveWorkflowBubble;
