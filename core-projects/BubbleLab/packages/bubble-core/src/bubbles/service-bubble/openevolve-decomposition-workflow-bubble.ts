import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const WorkflowOperationSchema = z.enum([
  'create',
  'list',
  'get',
  'pause',
  'resume',
  'results',
  'delete',
  'health',
]);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_DECOMPOSITION_WORKFLOW_API_URL ||
        process.env.OPENEVOLVE_API_URL ||
        process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const BaseParamsSchema = z.object({
  operation: WorkflowOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().int().min(1000).max(300000).default(60000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),
});

const CreateWorkflowSchema = BaseParamsSchema.extend({
  operation: z.literal('create'),
  problem_statement: z.string().min(10),
  content_analyzer_team: z.string().min(1),
  planner_team: z.string().min(1),
  solver_team: z.string().min(1),
  patcher_team: z.string().min(1),
  assembler_team: z.string().min(1),
  sub_problem_red_gauntlet: z.string().min(1),
  sub_problem_gold_gauntlet: z.string().min(1),
  final_red_gauntlet: z.string().min(1),
  final_gold_gauntlet: z.string().min(1),
  solver_generation_gauntlet: z.string().min(1),
  max_refinement_loops: z.number().int().min(1).max(10).default(3),
  mdap_enabled: z.boolean().optional(),
  mdap_config: z.record(z.unknown()).optional(),
  maker_enabled: z.boolean().optional(),
  maker_config: z.record(z.unknown()).optional(),
});

const GetWorkflowSchema = BaseParamsSchema.extend({
  operation: z.literal('get'),
  workflow_id: z.string().min(1),
});

const PauseWorkflowSchema = BaseParamsSchema.extend({
  operation: z.literal('pause'),
  workflow_id: z.string().min(1),
});

const ResumeWorkflowSchema = BaseParamsSchema.extend({
  operation: z.literal('resume'),
  workflow_id: z.string().min(1),
});

const ResultsWorkflowSchema = BaseParamsSchema.extend({
  operation: z.literal('results'),
  workflow_id: z.string().min(1),
});

const DeleteWorkflowSchema = BaseParamsSchema.extend({
  operation: z.literal('delete'),
  workflow_id: z.string().min(1),
});

const ListWorkflowSchema = BaseParamsSchema.extend({
  operation: z.literal('list'),
});

const HealthSchema = BaseParamsSchema.extend({
  operation: z.literal('health'),
});

const OpenEvolveDecompositionWorkflowParamsSchema = z.discriminatedUnion('operation', [
  CreateWorkflowSchema,
  ListWorkflowSchema,
  GetWorkflowSchema,
  PauseWorkflowSchema,
  ResumeWorkflowSchema,
  ResultsWorkflowSchema,
  DeleteWorkflowSchema,
  HealthSchema,
]);

type OpenEvolveDecompositionWorkflowParams = z.input<
  typeof OpenEvolveDecompositionWorkflowParamsSchema
>;

const OpenEvolveDecompositionWorkflowResultSchema = z.object({
  success: z.boolean(),
  status: z.number().optional(),
  operation: z.string(),
  workflow_id: z.string().optional(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type OpenEvolveDecompositionWorkflowResult = z.output<
  typeof OpenEvolveDecompositionWorkflowResultSchema
>;

export class OpenEvolveDecompositionWorkflowBubble extends ServiceBubble<
  OpenEvolveDecompositionWorkflowParams,
  OpenEvolveDecompositionWorkflowResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-decomposition-workflow' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = OpenEvolveDecompositionWorkflowParamsSchema;
  static readonly resultSchema = OpenEvolveDecompositionWorkflowResultSchema;
  static readonly shortDescription = 'OpenEvolve sovereign decomposition workflows (MDAP/MAKER)';
  static readonly longDescription = `
    OpenEvolve decomposition workflow bubble for sovereign decomposition workflows.

    Operations:
    - create/list/get/pause/resume/results/delete
    - Supports MDAP/MAKER toggles and configs
  `;
  static readonly alias = 'openevolve-decomposition-workflow';

  constructor(
    params: OpenEvolveDecompositionWorkflowParams,
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

  protected async performAction(): Promise<OpenEvolveDecompositionWorkflowResult> {
    const startTime = Date.now();
    const params = this.params;

    try {
      switch (params.operation) {
        case 'create':
          return await this.request('POST', '/workflows', {
            problem_statement: params.problem_statement,
            content_analyzer_team: params.content_analyzer_team,
            planner_team: params.planner_team,
            solver_team: params.solver_team,
            patcher_team: params.patcher_team,
            assembler_team: params.assembler_team,
            sub_problem_red_gauntlet: params.sub_problem_red_gauntlet,
            sub_problem_gold_gauntlet: params.sub_problem_gold_gauntlet,
            final_red_gauntlet: params.final_red_gauntlet,
            final_gold_gauntlet: params.final_gold_gauntlet,
            solver_generation_gauntlet: params.solver_generation_gauntlet,
            max_refinement_loops: params.max_refinement_loops,
            mdap_enabled: params.mdap_enabled,
            mdap_config: params.mdap_config,
            maker_enabled: params.maker_enabled,
            maker_config: params.maker_config,
          }, startTime);
        case 'list':
          return await this.request('GET', '/workflows', undefined, startTime);
        case 'get':
          return await this.request('GET', `/workflows/${params.workflow_id}`, undefined, startTime);
        case 'pause':
          return await this.request('POST', `/workflows/${params.workflow_id}/pause`, {}, startTime);
        case 'resume':
          return await this.request('POST', `/workflows/${params.workflow_id}/resume`, {}, startTime);
        case 'results':
          return await this.request('GET', `/workflows/${params.workflow_id}/results`, undefined, startTime);
        case 'delete':
          return await this.request('DELETE', `/workflows/${params.workflow_id}`, undefined, startTime);
        case 'health':
          return await this.request('GET', '/health', undefined, startTime);
        default:
          return {
            success: false,
            status: 400,
            operation: params.operation,
            error: `Unsupported operation: ${params.operation}`,
            timing: Date.now() - startTime,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        status: 500,
        operation: params.operation,
        workflow_id: (this.params as any).workflow_id,
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
  ): Promise<OpenEvolveDecompositionWorkflowResult> {
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
        operation: this.params.operation,
        workflow_id: (this.params as any).workflow_id,
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
        operation: this.params.operation,
        workflow_id: (this.params as any).workflow_id,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }
}

export default OpenEvolveDecompositionWorkflowBubble;
