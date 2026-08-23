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

const KnowledgeExplorerQueryParamsSchema = z.object({
  operation: z.enum(['query_advanced']),
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  query: z.string(),
  filters: z.record(z.unknown()).optional(),
  limit: z.number().optional(),
  top_k: z.number().optional(),
});

type KnowledgeExplorerQueryParams = z.input<
  typeof KnowledgeExplorerQueryParamsSchema
> &
  ServiceBubbleParams;

const KnowledgeExplorerQueryResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type KnowledgeExplorerQueryResult = z.output<
  typeof KnowledgeExplorerQueryResultSchema
> &
  BubbleOperationResult;

export class OpenEvolveKnowledgeExplorerQueryBubble extends ServiceBubble<
  KnowledgeExplorerQueryParams,
  KnowledgeExplorerQueryResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName =
    'knowledge-explorer-query' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = KnowledgeExplorerQueryParamsSchema;
  static readonly resultSchema = KnowledgeExplorerQueryResultSchema;
  static readonly shortDescription =
    'OpenEvolve Knowledge Explorer advanced query bubble';
  static readonly longDescription = `
    Runs an advanced query against the OpenEvolve Knowledge Explorer.
  `;
  static readonly alias = 'openevolve-knowledge-explorer-query';

  constructor(
    params: KnowledgeExplorerQueryParams,
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

  protected async performAction(): Promise<KnowledgeExplorerQueryResult> {
    const startTime = Date.now();
    return this.request(
      'POST',
      '/bubblelabs/knowledge/query-advanced',
      {
        query: this.params.query,
        filters: this.params.filters,
        limit: this.params.limit,
        top_k: this.params.top_k,
      },
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
  ): Promise<KnowledgeExplorerQueryResult> {
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

export default OpenEvolveKnowledgeExplorerQueryBubble;
