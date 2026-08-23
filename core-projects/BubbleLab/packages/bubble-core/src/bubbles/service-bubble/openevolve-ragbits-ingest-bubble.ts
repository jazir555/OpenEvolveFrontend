import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const OperationSchema = z.enum(['ingest']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const RagbitsIngestParamsSchema = z.object({
  operation: OperationSchema.default('ingest'),
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  documents: z.array(z.record(z.unknown())).optional(),
  paths: z.array(z.string()).optional(),
  urls: z.array(z.string()).optional(),
  collection: z.string().optional(),
  chunk_size: z.number().optional(),
  chunk_overlap: z.number().optional(),
  metadata: z.record(z.unknown()).optional(),
  options: z.record(z.unknown()).optional(),
});

type RagbitsIngestParams = z.input<typeof RagbitsIngestParamsSchema> &
  ServiceBubbleParams;

const RagbitsIngestResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type RagbitsIngestResult = z.output<typeof RagbitsIngestResultSchema> &
  BubbleOperationResult;

export class OpenEvolveRagbitsIngestBubble extends ServiceBubble<
  RagbitsIngestParams,
  RagbitsIngestResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName =
    'openevolve-ragbits-ingest' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = RagbitsIngestParamsSchema;
  static readonly resultSchema = RagbitsIngestResultSchema;
  static readonly shortDescription =
    'OpenEvolve RAGBits document ingestion bubble';
  static readonly longDescription = `
    Ingests documents, files, or URLs into a RAGBits collection
    (POST /openevolve/ragbits/ingest) with chunking and metadata controls.
  `;
  static readonly alias = 'openevolve-ragbits-ingest';

  constructor(params: RagbitsIngestParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<RagbitsIngestResult> {
    const startTime = Date.now();
    try {
      switch (this.params.operation as string) {
        case 'ingest': {
          const hasSource =
            (this.params.documents && this.params.documents.length > 0) ||
            (this.params.paths && this.params.paths.length > 0) ||
            (this.params.urls && this.params.urls.length > 0);
          if (!hasSource) {
            throw new Error(
              'documents, paths, or urls is required for ingest'
            );
          }
          return await this.request(
            'POST',
            '/openevolve/ragbits/ingest',
            {
              documents: this.params.documents,
              paths: this.params.paths,
              urls: this.params.urls,
              collection: this.params.collection,
              chunk_size: this.params.chunk_size,
              chunk_overlap: this.params.chunk_overlap,
              metadata: this.params.metadata,
              options: this.params.options,
            },
            startTime
          );
        }
        default:
          return {
            success: false,
            operation: this.params.operation as string,
            error: `Unsupported operation: ${this.params.operation as string}`,
            timing: Date.now() - startTime,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: this.params.operation as string,
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
  ): Promise<RagbitsIngestResult> {
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
        operation: this.params.operation as string,
        data,
        error: response.ok ? undefined : data?.error || response.statusText,
        timing: Date.now() - startTime,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: this.params.operation as string,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }
}

export default OpenEvolveRagbitsIngestBubble;
