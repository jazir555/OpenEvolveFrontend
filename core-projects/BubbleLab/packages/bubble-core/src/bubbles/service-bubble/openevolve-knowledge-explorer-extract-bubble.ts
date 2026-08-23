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

const KnowledgeExplorerExtractParamsSchema = z.object({
  operation: z.enum(['extract']),
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  text: z.string().optional(),
  source: z.string().optional(),
  format: z.string().optional(),
});

type KnowledgeExplorerExtractParams = z.input<
  typeof KnowledgeExplorerExtractParamsSchema
> &
  ServiceBubbleParams;

const KnowledgeExplorerExtractResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type KnowledgeExplorerExtractResult = z.output<
  typeof KnowledgeExplorerExtractResultSchema
> &
  BubbleOperationResult;

export class OpenEvolveKnowledgeExplorerExtractBubble extends ServiceBubble<
  KnowledgeExplorerExtractParams,
  KnowledgeExplorerExtractResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName =
    'openevolve-knowledge-explorer-extract' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = KnowledgeExplorerExtractParamsSchema;
  static readonly resultSchema = KnowledgeExplorerExtractResultSchema;
  static readonly shortDescription =
    'OpenEvolve Knowledge Explorer extract bubble';
  static readonly longDescription = `
    Extracts structured knowledge from text via the OpenEvolve Knowledge Explorer.
  `;
  static readonly alias = 'openevolve-knowledge-explorer-extract';

  constructor(
    params: KnowledgeExplorerExtractParams,
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

  protected async performAction(): Promise<KnowledgeExplorerExtractResult> {
    const startTime = Date.now();
    return this.request(
      'POST',
      '/bubblelabs/knowledge/extract',
      {
        text: this.params.text,
        source: this.params.source,
        format: this.params.format,
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
  ): Promise<KnowledgeExplorerExtractResult> {
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

export default OpenEvolveKnowledgeExplorerExtractBubble;
