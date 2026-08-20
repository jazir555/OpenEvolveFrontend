import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const IcrOperationSchema = z.enum([
  'emit_refinement',
  'poll_refinement',
  'request_reward_calibration',
  'next_reward_calibration',
  'respond_reward_calibration',
  'get_reward_calibration_response',
  'post_heatmap_snapshot',
  'health',
]);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_ICR_API_URL ||
        process.env.ICR_API_BASE_URL ||
        process.env.OPENEVOLVE_API_URL ||
        process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const BaseParamsSchema = z.object({
  operation: IcrOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().int().min(1000).max(300000).default(60000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),
});

const EmitRefinementSchema = BaseParamsSchema.extend({
  operation: z.literal('emit_refinement'),
  reason: z.string().optional(),
  overall_score: z.number().min(0).max(1).optional(),
  weaknesses: z.array(z.string()).optional(),
  friction_points: z.array(z.string()).optional(),
  auto_refine: z.boolean().optional(),
});

const PollRefinementSchema = BaseParamsSchema.extend({
  operation: z.literal('poll_refinement'),
  limit: z.number().int().min(1).max(50).optional(),
});

const RewardCalibrationRequestSchema = BaseParamsSchema.extend({
  operation: z.literal('request_reward_calibration'),
  request_id: z.string().optional(),
  option_a: z.string().min(1),
  option_b: z.string().min(1),
  confidence: z.number().min(0).max(1).optional(),
  prompt: z.string().optional(),
});

const RewardCalibrationNextSchema = BaseParamsSchema.extend({
  operation: z.literal('next_reward_calibration'),
});

const RewardCalibrationRespondSchema = BaseParamsSchema.extend({
  operation: z.literal('respond_reward_calibration'),
  request_id: z.string().optional(),
  choice: z.string().min(1),
});

const RewardCalibrationResponseSchema = BaseParamsSchema.extend({
  operation: z.literal('get_reward_calibration_response'),
  request_id: z.string().min(1),
});

const HeatmapPointSchema = z.object({
  x: z.number().min(0).max(1),
  y: z.number().min(0).max(1),
  intensity: z.number().optional(),
  dwellMs: z.number().optional(),
  timestamp: z.number().optional(),
  type: z.string().optional(),
});

const HeatmapSnapshotSchema = BaseParamsSchema.extend({
  operation: z.literal('post_heatmap_snapshot'),
  snapshot_id: z.string().optional(),
  timestamp: z.number().optional(),
  screen_html: z.string().min(1),
  heatmap_data_url: z.string().optional(),
  composite_data_url: z.string().optional(),
  points: z.array(HeatmapPointSchema).optional(),
  manual_code_delta: z.number().optional(),
  context_text: z.string().optional(),
  auto_refine: z.boolean().optional(),
});

const HealthSchema = BaseParamsSchema.extend({
  operation: z.literal('health'),
});

const OpenEvolveIcrParamsSchema = z.discriminatedUnion('operation', [
  EmitRefinementSchema,
  PollRefinementSchema,
  RewardCalibrationRequestSchema,
  RewardCalibrationNextSchema,
  RewardCalibrationRespondSchema,
  RewardCalibrationResponseSchema,
  HeatmapSnapshotSchema,
  HealthSchema,
]);

type OpenEvolveIcrParams = z.input<typeof OpenEvolveIcrParamsSchema> & ServiceBubbleParams;

const OpenEvolveIcrResultSchema = z.object({
  success: z.boolean(),
  status: z.number().optional(),
  operation: z.string(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type OpenEvolveIcrResult = z.output<typeof OpenEvolveIcrResultSchema> & BubbleOperationResult;

export class OpenEvolveIcrBubble extends ServiceBubble<
  OpenEvolveIcrParams,
  OpenEvolveIcrResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-icr' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = OpenEvolveIcrParamsSchema;
  static readonly resultSchema = OpenEvolveIcrResultSchema;
  static readonly shortDescription = 'OpenEvolve ICR event + calibration bridge';
  static readonly longDescription = `
    OpenEvolve ICR bridge for refinement events, reward calibration, and heatmap snapshots.

    Operations:
    - emit_refinement: enqueue a refinement-needed event
    - poll_refinement: consume queued refinement events
    - request_reward_calibration: enqueue preference query
    - next_reward_calibration: consume next preference query
    - respond_reward_calibration: submit sovereign preference
    - get_reward_calibration_response: fetch response by request_id
    - post_heatmap_snapshot: send UI heatmap/composite snapshot
  `;
  static readonly alias = 'openevolve-icr';

  constructor(
    params: OpenEvolveIcrParams,
    context?: BubbleContext,
    instanceId?: string
  ) {
    super(params, context, instanceId);
  }

  protected chooseCredential(): string | undefined {
    return undefined;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<OpenEvolveIcrResult> {
    const startTime = Date.now();
    const params = this.params;
    const op: string = params.operation;

    try {
      switch (params.operation) {
        case 'emit_refinement':
          return await this.request(
            'POST',
            '/icr/events/refinement-needed',
            {
              reason: params.reason,
              overall_score: params.overall_score,
              weaknesses: params.weaknesses,
              friction_points: params.friction_points,
              auto_refine: params.auto_refine,
            },
            startTime
          );
        case 'poll_refinement': {
          const limit = params.limit ?? 5;
          return await this.request(
            'GET',
            `/icr/events/refinement-needed?limit=${limit}`,
            undefined,
            startTime
          );
        }
        case 'request_reward_calibration':
          return await this.request(
            'POST',
            '/icr/reward-calibration/request',
            {
              request_id: params.request_id,
              option_a: params.option_a,
              option_b: params.option_b,
              confidence: params.confidence,
              prompt: params.prompt,
            },
            startTime
          );
        case 'next_reward_calibration':
          return await this.request(
            'GET',
            '/icr/reward-calibration/next',
            undefined,
            startTime
          );
        case 'respond_reward_calibration':
          return await this.request(
            'POST',
            '/icr/reward-calibration/respond',
            {
              request_id: params.request_id,
              choice: params.choice,
            },
            startTime
          );
        case 'get_reward_calibration_response':
          return await this.request(
            'GET',
            `/icr/reward-calibration/response/${params.request_id}`,
            undefined,
            startTime
          );
        case 'post_heatmap_snapshot':
          return await this.request(
            'POST',
            '/icr/heatmap/snapshot',
            {
              snapshot_id: params.snapshot_id,
              timestamp: params.timestamp,
              screen_html: params.screen_html,
              heatmap_data_url: params.heatmap_data_url,
              composite_data_url: params.composite_data_url,
              points: params.points,
              manual_code_delta: params.manual_code_delta,
              context_text: params.context_text,
              auto_refine: params.auto_refine,
            },
            startTime
          );
        case 'health':
          return await this.request('GET', '/health', undefined, startTime);
        default:
          return {
            success: false,
            status: 400,
            operation: op,
            error: `Unsupported operation: ${op}`,
            timing: Date.now() - startTime,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        status: 500,
        operation: op,
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
      const token = this.params.auth_token;
      if (headerName.toLowerCase() === 'authorization' && !token.startsWith('Bearer ')) {
        headers[headerName] = `Bearer ${token}`;
      } else {
        headers[headerName] = token;
      }
    }
    return headers;
  }

  private async request(
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    endpoint: string,
    body: unknown,
    startTime: number
  ): Promise<OpenEvolveIcrResult> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.params.timeout);
    const url = `${this.params.base_url}${endpoint}`;

    try {
      const response = await fetch(url, {
        method,
        headers: this.buildHeaders(),
        body: body !== undefined && method !== 'GET' ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      const text = await response.text();
      let data: unknown = undefined;
      if (text) {
        try {
          data = JSON.parse(text);
        } catch {
          data = text;
        }
      }

      return {
        success: response.ok,
        status: response.status,
        operation: ((this.params.operation as string) as string),
        data,
        error: response.ok
          ? undefined
          : typeof data === 'object' && data && 'detail' in data
            ? String((data as any).detail)
            : response.statusText,
        timing: Date.now() - startTime,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        status: 0,
        operation: ((this.params.operation as string) as string),
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }
}

export default OpenEvolveIcrBubble;
