import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const GauntletOperationSchema = z.enum(['create', 'list', 'get', 'update', 'delete', 'health']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const BaseParamsSchema = z.object({
  operation: GauntletOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().int().min(1000).max(300000).default(60000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),
});

const GauntletRoundSchema = z.object({
  id: z.string().optional(),
  name: z.string().min(1),
  quorum_threshold: z.number().min(0).max(1),
  confidence_threshold: z.number().min(0).max(1),
  evaluation_type: z.string().min(1),
  required_consensus: z.boolean().optional(),
  max_iterations: z.number().int().min(1).optional(),
});

const CreateGauntletSchema = BaseParamsSchema.extend({
  operation: z.literal('create'),
  name: z.string().min(1),
  description: z.string().optional(),
  rounds: z.array(GauntletRoundSchema).min(1),
});

const ListGauntletSchema = BaseParamsSchema.extend({
  operation: z.literal('list'),
});

const GetGauntletSchema = BaseParamsSchema.extend({
  operation: z.literal('get'),
  gauntlet_id: z.string().min(1),
});

const UpdateGauntletSchema = BaseParamsSchema.extend({
  operation: z.literal('update'),
  gauntlet_id: z.string().min(1),
  name: z.string().min(1).optional(),
  description: z.string().optional(),
  rounds: z.array(GauntletRoundSchema).optional(),
});

const DeleteGauntletSchema = BaseParamsSchema.extend({
  operation: z.literal('delete'),
  gauntlet_id: z.string().min(1),
});

const HealthSchema = BaseParamsSchema.extend({
  operation: z.literal('health'),
});

const OpenEvolveGauntletParamsSchema = z.discriminatedUnion('operation', [
  CreateGauntletSchema,
  ListGauntletSchema,
  GetGauntletSchema,
  UpdateGauntletSchema,
  DeleteGauntletSchema,
  HealthSchema,
]);

type OpenEvolveGauntletParams = z.input<typeof OpenEvolveGauntletParamsSchema> & ServiceBubbleParams;

const OpenEvolveGauntletResultSchema = z.object({
  success: z.boolean(),
  status: z.number().optional(),
  operation: z.string(),
  gauntlet_id: z.string().optional(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type OpenEvolveGauntletResult = z.output<typeof OpenEvolveGauntletResultSchema> & BubbleOperationResult;

export class OpenEvolveGauntletBubble extends ServiceBubble<
  OpenEvolveGauntletParams,
  OpenEvolveGauntletResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-gauntlet' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = OpenEvolveGauntletParamsSchema;
  static readonly resultSchema = OpenEvolveGauntletResultSchema;
  static readonly shortDescription =
    'OpenEvolve gauntlet management (create, update, list)';
  static readonly longDescription = `
    OpenEvolve gauntlet bubble for managing evaluation gauntlets.
  `;
  static readonly alias = 'openevolve-gauntlet';

  constructor(
    params: OpenEvolveGauntletParams,
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

  protected async performAction(): Promise<OpenEvolveGauntletResult> {
    const startTime = Date.now();
    const params = this.params;
    const op: string = params.operation;

    try {
      switch (params.operation) {
        case 'create':
          return await this.request(
            'POST',
            '/api/gauntlets',
            {
              name: params.name,
              description: params.description,
              rounds: params.rounds,
            },
            startTime
          );
        case 'list':
          return await this.request('GET', '/api/gauntlets', undefined, startTime);
        case 'get':
          return await this.request(
            'GET',
            `/api/gauntlets/${params.gauntlet_id}`,
            undefined,
            startTime
          );
        case 'update':
          return await this.request(
            'PUT',
            `/api/gauntlets/${params.gauntlet_id}`,
            {
              name: params.name,
              description: params.description,
              rounds: params.rounds,
            },
            startTime
          );
        case 'delete':
          return await this.request(
            'DELETE',
            `/api/gauntlets/${params.gauntlet_id}`,
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
  ): Promise<OpenEvolveGauntletResult> {
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
        operation: ((this.params.operation as string) as string),
        gauntlet_id:
          (this.params as any).gauntlet_id ||
          (typeof data === 'object' && data && 'id' in data
            ? String((data as any).id)
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
        operation: ((this.params.operation as string) as string),
        gauntlet_id: (this.params as any).gauntlet_id,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }
}

export default OpenEvolveGauntletBubble;
