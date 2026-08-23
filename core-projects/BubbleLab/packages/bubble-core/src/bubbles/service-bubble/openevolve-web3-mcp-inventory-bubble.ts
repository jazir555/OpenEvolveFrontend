import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const OpenEvolveWeb3McpInventoryParamsSchema = z.object({
  operation: z.enum(['mcp-inventory']),
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),
});

type OpenEvolveWeb3McpInventoryParams = z.input<
  typeof OpenEvolveWeb3McpInventoryParamsSchema
> &
  ServiceBubbleParams;

const OpenEvolveWeb3McpInventoryResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type OpenEvolveWeb3McpInventoryResult = z.output<
  typeof OpenEvolveWeb3McpInventoryResultSchema
> &
  BubbleOperationResult;

export class OpenEvolveWeb3McpInventoryBubble extends ServiceBubble<
  OpenEvolveWeb3McpInventoryParams,
  OpenEvolveWeb3McpInventoryResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'web3-mcp-inventory' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = OpenEvolveWeb3McpInventoryParamsSchema;
  static readonly resultSchema = OpenEvolveWeb3McpInventoryResultSchema;
  static readonly shortDescription =
    'OpenEvolve web3-mcp-inventory service bubble';
  static readonly longDescription = `
    Returns the inventory of web3 MCP tools available for the audit service.
  `;
  static readonly alias = 'web3-mcp-inventory';

  constructor(
    params: OpenEvolveWeb3McpInventoryParams,
    context?: BubbleContext
  ) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<OpenEvolveWeb3McpInventoryResult> {
    const startTime = Date.now();
    return this.request(
      'GET',
      '/web3/mcp-tool-inventory',
      undefined,
      startTime
    );
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
  ): Promise<OpenEvolveWeb3McpInventoryResult> {
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
        operation: this.params.operation,
        data,
        error: response.ok ? undefined : data?.error || response.statusText,
        timing: Date.now() - startTime,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: this.params.operation,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }
}

export default OpenEvolveWeb3McpInventoryBubble;
