import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const CrewAIOperationSchema = z.enum([
  'health_check',
  'get_capabilities',
  'execute_workflow',
  'execute_phase',
  'get_status',
  'get_results',
  'list_workflows',
  'delegate_task',
]);

const CrewAIExecutionMethodSchema = z.enum([
  'traditional',
  'roma',
  'roma_mdap_maker',
  'claudiomiro',
  'datapizza',
  'hybrid',
  'auto',
]);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const CrewAIParamsSchema = z.object({
  operation: CrewAIOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(300000).default(60000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  workflow_id: z.string().optional(),
  problem_statement: z.string().optional(),
  execution_method: CrewAIExecutionMethodSchema.default('auto'),

  phase_number: z.number().min(1).max(6).optional(),
  phase_input: z.record(z.unknown()).optional(),

  task_name: z.string().optional(),
  task_description: z.string().optional(),
  team_name: z.string().optional(),

  parameters: z.record(z.unknown()).optional(),
});

type CrewAIParams = z.input<typeof CrewAIParamsSchema> & ServiceBubbleParams;

const CrewAIResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  workflow_id: z.string().optional(),
  status: z.string().optional(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type CrewAIResult = z.output<typeof CrewAIResultSchema> & BubbleOperationResult;

export class OpenEvolveCrewAIBubble extends ServiceBubble<
  CrewAIParams,
  CrewAIResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'openevolve-crewai' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = CrewAIParamsSchema;
  static readonly resultSchema = CrewAIResultSchema;
  static readonly shortDescription = 'OpenEvolve CrewAI orchestration bubble';
  static readonly longDescription = `
    CrewAI orchestration bubble for OpenEvolve workflows.
  `;
  static readonly alias = 'openevolve-crewai';

  constructor(params: CrewAIParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<CrewAIResult> {
    const startTime = Date.now();
    try {
      switch (((this.params.operation as string) as string)) {
        case 'health_check':
          return await this.request('GET', '/api/crewai/health', undefined, startTime);
        case 'get_capabilities':
          return await this.request('GET', '/api/crewai/capabilities', undefined, startTime);
        case 'execute_workflow':
          return await this.request('POST', '/api/crewai/workflows', {
            problem_statement: this.params.problem_statement,
            execution_method: this.params.execution_method,
            parameters: this.params.parameters,
          }, startTime);
        case 'execute_phase':
          return await this.request('POST', '/api/crewai/phases', {
            workflow_id: this.params.workflow_id,
            phase_number: this.params.phase_number,
            phase_input: this.params.phase_input,
            parameters: this.params.parameters,
          }, startTime);
        case 'get_status':
          return await this.request('GET', `/api/crewai/workflows/${this.requireWorkflowId()}/status`, undefined, startTime);
        case 'get_results':
          return await this.request('GET', `/api/crewai/workflows/${this.requireWorkflowId()}/results`, undefined, startTime);
        case 'list_workflows':
          return await this.request('GET', '/api/crewai/workflows', undefined, startTime);
        case 'delegate_task':
          return await this.request('POST', '/api/crewai/tasks', {
            task_name: this.params.task_name,
            task_description: this.requireTaskDescription(),
            team_name: this.params.team_name,
            context: this.params.parameters,
          }, startTime);
        default:
          return {
            success: false,
            operation: ((this.params.operation as string) as string),
            error: `Unsupported operation: ${((this.params.operation as string) as string)}`,
            timing: Date.now() - startTime,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: ((this.params.operation as string) as string),
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }

  private requireWorkflowId(): string {
    if (!this.params.workflow_id) {
      throw new Error('workflow_id is required');
    }
    return this.params.workflow_id;
  }

  private requireTaskDescription(): string {
    if (!this.params.task_description) {
      throw new Error('task_description is required for delegate_task');
    }
    return this.params.task_description;
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
  ): Promise<CrewAIResult> {
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
        workflow_id: data?.workflow_id || this.params.workflow_id,
        status: data?.status,
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

export default OpenEvolveCrewAIBubble;
