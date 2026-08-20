import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const TeamOperationSchema = z.enum(['create', 'list', 'get', 'update', 'delete', 'health']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const BaseParamsSchema = z.object({
  operation: TeamOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().int().min(1000).max(300000).default(60000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),
});

const TeamMemberSchema = z.object({
  id: z.string().optional(),
  name: z.string().min(1),
  role: z.string().min(1),
  model: z.string().min(1),
  temperature: z.number().min(0).max(2).default(0.7),
  max_tokens: z.number().int().min(1).default(4096),
  top_p: z.number().min(0).max(1).optional(),
  frequency_penalty: z.number().min(-2).max(2).optional(),
  presence_penalty: z.number().min(-2).max(2).optional(),
  max_iterations: z.number().int().min(1).optional(),
});

const CreateTeamSchema = BaseParamsSchema.extend({
  operation: z.literal('create'),
  name: z.string().min(1),
  description: z.string().optional(),
  members: z.array(TeamMemberSchema).min(1),
});

const ListTeamSchema = BaseParamsSchema.extend({
  operation: z.literal('list'),
});

const GetTeamSchema = BaseParamsSchema.extend({
  operation: z.literal('get'),
  team_id: z.string().min(1),
});

const UpdateTeamSchema = BaseParamsSchema.extend({
  operation: z.literal('update'),
  team_id: z.string().min(1),
  name: z.string().min(1).optional(),
  description: z.string().optional(),
  members: z.array(TeamMemberSchema).optional(),
});

const DeleteTeamSchema = BaseParamsSchema.extend({
  operation: z.literal('delete'),
  team_id: z.string().min(1),
});

const HealthSchema = BaseParamsSchema.extend({
  operation: z.literal('health'),
});

const OpenEvolveTeamParamsSchema = z.discriminatedUnion('operation', [
  CreateTeamSchema,
  ListTeamSchema,
  GetTeamSchema,
  UpdateTeamSchema,
  DeleteTeamSchema,
  HealthSchema,
]);

type OpenEvolveTeamParams = z.input<typeof OpenEvolveTeamParamsSchema>;

const OpenEvolveTeamResultSchema = z.object({
  success: z.boolean(),
  status: z.number().optional(),
  operation: z.string(),
  team_id: z.string().optional(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type OpenEvolveTeamResult = z.output<typeof OpenEvolveTeamResultSchema>;

export class OpenEvolveTeamBubble extends ServiceBubble<
  OpenEvolveTeamParams,
  OpenEvolveTeamResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-team' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = OpenEvolveTeamParamsSchema;
  static readonly resultSchema = OpenEvolveTeamResultSchema;
  static readonly shortDescription = 'OpenEvolve team management (create, update, list)';
  static readonly longDescription = `
    OpenEvolve team bubble for managing AI agent teams.

    Supports CRUD operations against the OpenEvolve API.
  `;
  static readonly alias = 'openevolve-team';

  constructor(
    params: OpenEvolveTeamParams,
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

  protected async performAction(): Promise<OpenEvolveTeamResult> {
    const startTime = Date.now();
    const params = this.params;

    try {
      switch (params.operation) {
        case 'create':
          return await this.request(
            'POST',
            '/api/teams',
            {
              name: params.name,
              description: params.description,
              members: params.members,
            },
            startTime
          );
        case 'list':
          return await this.request('GET', '/api/teams', undefined, startTime);
        case 'get':
          return await this.request(
            'GET',
            `/api/teams/${params.team_id}`,
            undefined,
            startTime
          );
        case 'update':
          return await this.request(
            'PUT',
            `/api/teams/${params.team_id}`,
            {
              name: params.name,
              description: params.description,
              members: params.members,
            },
            startTime
          );
        case 'delete':
          return await this.request(
            'DELETE',
            `/api/teams/${params.team_id}`,
            undefined,
            startTime
          );
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
  ): Promise<OpenEvolveTeamResult> {
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
        team_id:
          (this.params as any).team_id ||
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
        operation: this.params.operation,
        team_id: (this.params as any).team_id,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }
}

export default OpenEvolveTeamBubble;
