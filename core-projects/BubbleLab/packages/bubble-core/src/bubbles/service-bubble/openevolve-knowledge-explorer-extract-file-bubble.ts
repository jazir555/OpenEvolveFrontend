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

const KnowledgeExplorerExtractFileParamsSchema = z.object({
  operation: z.enum(['extract_file']),
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  file_path: z.string().optional(),
  contents: z.string().optional(),
});

type KnowledgeExplorerExtractFileParams = z.input<
  typeof KnowledgeExplorerExtractFileParamsSchema
> &
  ServiceBubbleParams;

const KnowledgeExplorerExtractFileResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type KnowledgeExplorerExtractFileResult = z.output<
  typeof KnowledgeExplorerExtractFileResultSchema
> &
  BubbleOperationResult;

export class OpenEvolveKnowledgeExplorerExtractFileBubble extends ServiceBubble<
  KnowledgeExplorerExtractFileParams,
  KnowledgeExplorerExtractFileResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName =
    'openevolve-knowledge-explorer-extract-file' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = KnowledgeExplorerExtractFileParamsSchema;
  static readonly resultSchema = KnowledgeExplorerExtractFileResultSchema;
  static readonly shortDescription =
    'OpenEvolve Knowledge Explorer extract-file bubble';
  static readonly longDescription = `
    Extracts structured knowledge from a file (JSON body with file_path/contents)
    via the OpenEvolve Knowledge Explorer.
  `;
  static readonly alias = 'openevolve-knowledge-explorer-extract-file';

  constructor(
    params: KnowledgeExplorerExtractFileParams,
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

  protected async performAction(): Promise<KnowledgeExplorerExtractFileResult> {
    const startTime = Date.now();
    return this.request(
      'POST',
      '/bubblelabs/knowledge/extract-file',
      {
        file_path: this.params.file_path,
        contents: this.params.contents,
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
  ): Promise<KnowledgeExplorerExtractFileResult> {
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

export default OpenEvolveKnowledgeExplorerExtractFileBubble;
