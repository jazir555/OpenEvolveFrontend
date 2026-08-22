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

const OpenEvolveKnowledgeEmbedParamsSchema = z.object({
  operation: z.string().default('embed'),
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  text: z.string().optional(),
  documents: z.array(z.unknown()).optional(),
});

type OpenEvolveKnowledgeEmbedParams = z.input<
  typeof OpenEvolveKnowledgeEmbedParamsSchema
> &
  ServiceBubbleParams;

const OpenEvolveKnowledgeEmbedResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type OpenEvolveKnowledgeEmbedResult =
  z.output<typeof OpenEvolveKnowledgeEmbedResultSchema> & BubbleOperationResult;

export class OpenEvolveKnowledgeEmbedBubble extends ServiceBubble<
  OpenEvolveKnowledgeEmbedParams,
  OpenEvolveKnowledgeEmbedResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-knowledge-embed' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = OpenEvolveKnowledgeEmbedParamsSchema;
  static readonly resultSchema = OpenEvolveKnowledgeEmbedResultSchema;
  static readonly shortDescription = 'OpenEvolve knowledge embed bubble';
  static readonly longDescription = `
    Generate embeddings for a text snippet or a set of documents.
  `;
  static readonly alias = 'openevolve-knowledge-embed';

  constructor(params: OpenEvolveKnowledgeEmbedParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<OpenEvolveKnowledgeEmbedResult> {
    const startTime = Date.now();
    return this.request(
      'POST',
      '/api/knowledge/embed',
      { text: this.params.text, documents: this.params.documents },
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
    method: 'GET' | 'POST' | 'DELETE',
    endpoint: string,
    body: unknown,
    startTime: number
  ): Promise<OpenEvolveKnowledgeEmbedResult> {
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

export default OpenEvolveKnowledgeEmbedBubble;
