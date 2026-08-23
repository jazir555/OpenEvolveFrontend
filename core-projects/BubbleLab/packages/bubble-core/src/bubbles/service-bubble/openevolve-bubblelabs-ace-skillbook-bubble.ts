import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const BubblelabsAceSkillbookOperationSchema = z.enum([
  'skillbook',
]);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const BubblelabsAceSkillbookParamsSchema = z.object({
  operation: BubblelabsAceSkillbookOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),
  options: z.record(z.unknown()).optional(),
});

type BubblelabsAceSkillbookParams = z.input<typeof BubblelabsAceSkillbookParamsSchema> & ServiceBubbleParams;

const BubblelabsAceSkillbookResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type BubblelabsAceSkillbookResult = z.output<typeof BubblelabsAceSkillbookResultSchema> & BubbleOperationResult;

export class OpenEvolveBubblelabsAceSkillbookBubble extends ServiceBubble<
  BubblelabsAceSkillbookParams,
  BubblelabsAceSkillbookResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-bubblelabs-ace-skillbook' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = BubblelabsAceSkillbookParamsSchema;
  static readonly resultSchema = BubblelabsAceSkillbookResultSchema;
  static readonly shortDescription = 'BubbleLabs ACE skillbook bubble';
  static readonly longDescription = `
    Generate or query the ACE skillbook.
  `;
  static readonly alias = 'openevolve-bubblelabs-ace-skillbook';

  constructor(params: BubblelabsAceSkillbookParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<BubblelabsAceSkillbookResult> {
    const startTime = Date.now();
    try {
      switch (((this.params.operation as string) as string)) {
        case 'skillbook':
          return await this.request('POST', `/bubblelabs/ace/skillbook`, this.params.options, startTime);
        default:
          return {
            success: false,
            operation: ((this.params.operation as string) as string),
            error: `Unsupported operation: ${(this.params.operation as string)}`,
            timing: Date.now() - startTime,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: ((this.params.operation as string) as string),
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }

  private requireParam(key: 'id'): string {
    const value = (this.params as any)[key];
    if (!value) {
      throw new Error(`${key} is required for ${(this.params.operation as string)}`);
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
    method: 'GET' | 'POST',
    endpoint: string,
    body: unknown,
    startTime: number
  ): Promise<BubblelabsAceSkillbookResult> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.params.timeout);
    const url = `${this.params.base_url}${endpoint}`;

    try {
      const response = await fetch(url, {
        method,
        headers: this.buildHeaders(),
        body: body && method !== 'GET' ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      const data = await response.json().catch(() => undefined);

      return {
        success: response.ok,
        operation: ((this.params.operation as string) as string),
        data,
        error: response.ok ? undefined : data?.error || response.statusText,
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

export default OpenEvolveBubblelabsAceSkillbookBubble;
