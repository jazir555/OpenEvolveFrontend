import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const OperationSchema = z.enum(['visualize']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const PygraphistryParamsSchema = z.object({
  operation: OperationSchema.default('visualize'),
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(600000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  nodes: z.array(z.record(z.unknown())).optional(),
  edges: z.array(z.record(z.unknown())).optional(),
  graph: z.record(z.unknown()).optional(),
  graph_id: z.string().optional(),
  title: z.string().optional(),
  layout: z.string().optional(),
  node_id_field: z.string().optional(),
  edge_source_field: z.string().optional(),
  edge_target_field: z.string().optional(),
  options: z.record(z.unknown()).optional(),
});

type PygraphistryParams = z.input<typeof PygraphistryParamsSchema> &
  ServiceBubbleParams;

const PygraphistryResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type PygraphistryResult = z.output<typeof PygraphistryResultSchema> &
  BubbleOperationResult;

export class OpenEvolvePygraphistryBubble extends ServiceBubble<
  PygraphistryParams,
  PygraphistryResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName =
    'pygraphistry' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = PygraphistryParamsSchema;
  static readonly resultSchema = PygraphistryResultSchema;
  static readonly shortDescription =
    'OpenEvolve PyGraphistry graph visualization bubble';
  static readonly longDescription = `
    Builds a PyGraphistry visualization from nodes/edges or a prepared graph payload
    (POST /api/openevolve/visualize/pygraphistry) and returns the plot URL/metadata.
  `;
  static readonly alias = 'openevolve-pygraphistry';

  constructor(params: PygraphistryParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<PygraphistryResult> {
    const startTime = Date.now();
    try {
      switch (this.params.operation as string) {
        case 'visualize': {
          const hasGraph =
            (this.params.edges && this.params.edges.length > 0) ||
            (this.params.nodes && this.params.nodes.length > 0) ||
            this.params.graph !== undefined ||
            this.params.graph_id !== undefined;
          if (!hasGraph) {
            throw new Error(
              'nodes, edges, graph, or graph_id is required for visualize'
            );
          }
          return await this.request(
            'POST',
            '/api/openevolve/visualize/pygraphistry',
            {
              nodes: this.params.nodes,
              edges: this.params.edges,
              graph: this.params.graph,
              graph_id: this.params.graph_id,
              title: this.params.title,
              layout: this.params.layout,
              node_id_field: this.params.node_id_field,
              edge_source_field: this.params.edge_source_field,
              edge_target_field: this.params.edge_target_field,
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
  ): Promise<PygraphistryResult> {
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

export default OpenEvolvePygraphistryBubble;
