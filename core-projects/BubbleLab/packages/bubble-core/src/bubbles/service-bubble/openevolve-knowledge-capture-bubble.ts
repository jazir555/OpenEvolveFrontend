import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const KnowledgeCaptureOperationSchema = z.enum(['capture', 'health_check']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const KnowledgeCaptureParamsSchema = z.object({
  operation: KnowledgeCaptureOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(60000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('X-API-Key'),

  content: z.string().min(1),
  artifact_type: z.string().default('learning'),
  source_workflow_id: z.string().optional(),
  domain: z.string().optional(),
  problem_type: z.string().optional(),
  effectiveness_score: z.number().min(0).max(1).default(0),
  related_artifacts: z.array(z.string()).optional(),
});

type KnowledgeCaptureParams = z.input<typeof KnowledgeCaptureParamsSchema> & ServiceBubbleParams;

const KnowledgeCaptureDataSchema = z.object({
  id: z.string().optional(),
  artifact_type: z.string(),
  source_workflow_id: z.string().optional(),
  effectiveness_score: z.number(),
});

const KnowledgeCaptureResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: KnowledgeCaptureDataSchema.optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type KnowledgeCaptureResult = z.output<typeof KnowledgeCaptureResultSchema> & BubbleOperationResult;

export class OpenEvolveKnowledgeCaptureBubble extends ServiceBubble<
  KnowledgeCaptureParams,
  KnowledgeCaptureResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-knowledge-capture' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = KnowledgeCaptureParamsSchema;
  static readonly resultSchema = KnowledgeCaptureResultSchema;
  static readonly shortDescription = 'OpenEvolve knowledge capture';
  static readonly longDescription = `
    Persists learned results into the OpenEvolve knowledge store via the real
    POST /api/knowledge/artifacts endpoint and returns the stored artifact id.
  `;
  static readonly alias = 'openevolve-knowledge-capture';

  constructor(params: KnowledgeCaptureParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<KnowledgeCaptureResult> {
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

      const payload = {
        artifact_type: this.params.artifact_type,
        content: this.params.content,
        source_workflow_id: this.params.source_workflow_id || '',
        domain: this.params.domain,
        problem_type: this.params.problem_type,
        effectiveness_score: this.params.effectiveness_score,
        related_artifacts: this.params.related_artifacts || [],
      };

      const res = await this.request('POST', '/api/knowledge/artifacts', payload, startTime, true);
      const id = res.success ? ((res.data as any)?.id as string | undefined) : undefined;

      return {
        success: res.success,
        operation: this.params.operation,
        data: {
          id,
          artifact_type: this.params.artifact_type ?? 'learning',
          source_workflow_id: this.params.source_workflow_id,
          effectiveness_score: this.params.effectiveness_score ?? 0,
        },
        error: res.success ? undefined : res.error,
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

export default OpenEvolveKnowledgeCaptureBubble;
