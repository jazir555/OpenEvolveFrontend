import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const TeamMembersOperationSchema = z.enum(['add', 'remove']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const TeamMembersParamsSchema = z.object({
  operation: TeamMembersOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  team_id: z.string().optional(),
  member_id: z.string().optional(),
  user_id: z.string().optional(),
});

type TeamMembersParams = z.input<typeof TeamMembersParamsSchema> & ServiceBubbleParams;

const TeamMembersResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type TeamMembersResult = z.output<typeof TeamMembersResultSchema> & BubbleOperationResult;

export class OpenEvolveTeamMembersBubble extends ServiceBubble<
  TeamMembersParams,
  TeamMembersResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-team-members' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = TeamMembersParamsSchema;
  static readonly resultSchema = TeamMembersResultSchema;
  static readonly shortDescription = 'OpenEvolve team members management bubble';
  static readonly longDescription = `
    Manage team members within OpenEvolve enhanced teams.
  `;
  static readonly alias = 'openevolve-team-members';

  constructor(params: TeamMembersParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<TeamMembersResult> {
    const startTime = Date.now();
    const op = (this.params.operation as string) as string;
    try {
      switch (op) {
        case 'add':
          return await this.request(
            'POST',
            `/api/teams-enhanced/teams/${this.requireParam('team_id')}/members`,
            { user_id: this.requireParam('user_id') },
            startTime
          );
        case 'remove':
          return await this.request(
            'DELETE',
            `/api/teams-enhanced/teams/${this.requireParam('team_id')}/members/${this.requireParam('member_id')}`,
            undefined,
            startTime
          );
        default:
          return {
            success: false,
            operation: op,
            error: `Unsupported operation: ${op}`,
            timing: Date.now() - startTime,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: op,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }

  private requireParam(key: 'team_id' | 'member_id' | 'user_id'): string {
    const value = (this.params as any)[key];
    if (!value) {
      throw new Error(`${key} is required for ${((this.params.operation as string) as string)}`);
    }
    return value;
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
      headers[headerName] =
        headerName.toLowerCase() === 'authorization' &&
        !this.params.auth_token.startsWith('Bearer ')
          ? `Bearer ${this.params.auth_token}`
          : this.params.auth_token;
    }
    return headers;
  }

  private async request(
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    endpoint: string,
    body: unknown,
    startTime: number
  ): Promise<TeamMembersResult> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.params.timeout);
    const url = `${this.params.base_url}${endpoint}`;

    try {
      const response = await fetch(url, {
        method,
        headers: this.buildHeaders(),
        body: body && method !== 'GET' && method !== 'DELETE' ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      const data = await response.json().catch(() => undefined);

      return {
        success: response.ok,
        operation: ((this.params.operation as string) as string),
        data,
        error: response.ok ? undefined : (data as any)?.error || response.statusText,
        timing: Date.now() - startTime,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: ((this.params.operation as string) as string),
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }
}

export default OpenEvolveTeamMembersBubble;
