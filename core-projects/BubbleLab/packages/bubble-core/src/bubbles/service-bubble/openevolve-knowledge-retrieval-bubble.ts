import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const KnowledgeRetrievalOperationSchema = z.enum(['retrieve', 'health_check']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const KnowledgeRetrievalParamsSchema = z.object({
  operation: KnowledgeRetrievalOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(60000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('X-API-Key'),

  query: z.string().optional(),
  limit: z.number().int().min(1).max(1000).default(10),
  artifact_type: z.string().optional(),
});

type KnowledgeRetrievalParams = z.input<typeof KnowledgeRetrievalParamsSchema> & ServiceBubbleParams;

const KnowledgeRetrievalDataSchema = z.object({
  items: z.array(z.unknown()),
  source: z.string(),
  query: z.string().optional(),
});

const KnowledgeRetrievalResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: KnowledgeRetrievalDataSchema.optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type KnowledgeRetrievalResult = z.output<typeof KnowledgeRetrievalResultSchema> & BubbleOperationResult;

export class OpenEvolveKnowledgeRetrievalBubble extends ServiceBubble<
  KnowledgeRetrievalParams,
  KnowledgeRetrievalResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-knowledge-retrieval' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = KnowledgeRetrievalParamsSchema;
  static readonly resultSchema = KnowledgeRetrievalResultSchema;
  static readonly shortDescription = 'OpenEvolve knowledge retrieval';
  static readonly longDescription = `
    Retrieves learned knowledge from the OpenEvolve knowledge store.
    Uses POST /api/knowledge/search when a query is supplied, otherwise
    GET /api/knowledge/artifacts. Real endpoints only.
  `;
  static readonly alias = 'openevolve-knowledge-retrieval';

  constructor(params: KnowledgeRetrievalParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<KnowledgeRetrievalResult> {
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

      let items: unknown[] = [];
      let source = 'none';

      if (this.params.query && this.params.query.trim().length > 0) {
        const search = await this.request(
          'POST',
          '/api/knowledge/search',
          { query: this.params.query, limit: this.params.limit },
          startTime,
          true
        );
        if (search.success) {
          items =
            (search.data as any)?.results ??
            (search.data as any)?.items ??
            (Array.isArray(search.data) ? search.data : []);
          source = 'search';
        }
      }

      if (items.length === 0) {
        const list = await this.request(
          'GET',
          this.params.artifact_type
            ? `/api/knowledge/artifacts?artifact_type=${encodeURIComponent(this.params.artifact_type)}`
            : '/api/knowledge/artifacts',
          undefined,
          startTime,
          true
        );
        if (list.success) {
          items = Array.isArray(list.data) ? list.data : ((list.data as any)?.items ?? []);
          source = 'artifacts';
        }
      }

      return {
        success: true,
        operation: this.params.operation,
        data: { items, source, query: this.params.query },
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

export default OpenEvolveKnowledgeRetrievalBubble;
