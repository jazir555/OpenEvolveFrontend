import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const WorkflowSystemSchema = z.enum([
  'decomposition',
  'evolutionary',
  'mdap_maker',
  'adversarial',
  'integrated',
]);

const WorkflowOperationSchema = z.enum([
  'start_workflow',
  'stop_workflow',
  'pause_workflow',
  'resume_workflow',
  'get_status',
  'get_results',
  'health_check',
  'configure',
  'batch_execute',
  'chain_workflows',
]);

const WorkflowParamsSchema = z.object({
  operation: WorkflowOperationSchema,
  system: WorkflowSystemSchema,

  base_url: z.string().url(),
  timeout: z.number().min(1000).max(300000).default(60000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  workflow_id: z.string().optional(),
  workflow_name: z.string().optional(),
  workflow_definition: z.record(z.unknown()).optional(),

  problem_statement: z.string().optional(),
  decomposition_strategy: z
    .enum(['hierarchical', 'parallel', 'adaptive'])
    .default('adaptive'),
  max_depth: z.number().min(1).max(10).default(5),

  population_size: z.number().min(10).max(1000).default(100),
  generations: z.number().min(1).max(1000).default(50),
  mutation_rate: z.number().min(0).max(1).default(0.1),
  crossover_rate: z.number().min(0).max(1).default(0.8),
  selection_method: z
    .enum(['tournament', 'roulette', 'rank'])
    .default('tournament'),
  fitness_function: z.string().optional(),

  mdap_config: z
    .object({
      enable_mcts: z.boolean().default(true),
      enable_adversarial: z.boolean().default(true),
      enable_evolution: z.boolean().default(true),
      iterations: z.number().default(100),
    })
    .optional(),

  maker_config: z.record(z.unknown()).optional(),
  adaptive_config: z.record(z.unknown()).optional(),

  test_types: z
    .array(z.enum(['red_team', 'blue_team', 'purple_team', 'automated']))
    .default(['automated']),
  attack_vectors: z.array(z.string()).optional(),
  defense_mechanisms: z.array(z.string()).optional(),
  test_duration: z.number().default(300),

  async_execution: z.boolean().default(false),
  callback_url: z.string().url().optional(),
  parameters: z.record(z.unknown()).optional(),
});

type WorkflowParams = z.input<typeof WorkflowParamsSchema>;

const WorkflowResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  system: z.string(),
  workflow_id: z.string().optional(),
  status: z.string().optional(),
  results: z.record(z.unknown()).optional(),
  metrics: z.record(z.unknown()).optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type WorkflowResult = z.output<typeof WorkflowResultSchema>;

export class OpenEvolveWorkflowOrchestratorBubble extends ServiceBubble<
  WorkflowParams,
  WorkflowResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName =
    'openevolve-workflow-orchestrator' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = WorkflowParamsSchema;
  static readonly resultSchema = WorkflowResultSchema;
  static readonly shortDescription =
    'OpenEvolve workflow orchestrator (decomposition, evolution, MDAP, adversarial)';
  static readonly longDescription = `
    Orchestrates OpenEvolve workflows across decomposition, evolutionary search,
    MDAP maker, and adversarial testing systems.
  `;
  static readonly alias = 'openevolve-workflow-orchestrator';

  constructor(params: WorkflowParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<WorkflowResult> {
    const startTime = Date.now();
    try {
      switch (this.params.operation) {
        case 'start_workflow':
          return await this.startWorkflow(startTime);
        case 'stop_workflow':
          return await this.simpleRequest('POST', `/api/workflows/${this.requireWorkflowId()}/stop`, startTime);
        case 'pause_workflow':
          return await this.simpleRequest('POST', `/api/workflows/${this.requireWorkflowId()}/pause`, startTime);
        case 'resume_workflow':
          return await this.simpleRequest('POST', `/api/workflows/${this.requireWorkflowId()}/resume`, startTime);
        case 'get_status':
          return await this.simpleRequest('GET', `/api/workflows/${this.requireWorkflowId()}/status`, startTime);
        case 'get_results':
          return await this.simpleRequest('GET', `/api/workflows/${this.requireWorkflowId()}/results`, startTime);
        case 'health_check':
          return await this.simpleRequest('GET', '/api/workflows/health', startTime);
        case 'configure':
          return await this.configureWorkflow(startTime);
        case 'batch_execute':
          return await this.batchExecute(startTime);
        case 'chain_workflows':
          return await this.chainWorkflows(startTime);
        default:
          return {
            success: false,
            operation: this.params.operation,
            system: this.params.system,
            error: `Unsupported operation: ${this.params.operation}`,
            timing: Date.now() - startTime,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: this.params.operation,
        system: this.params.system,
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
      headers[headerName] =
        headerName.toLowerCase() === 'authorization' && !token.startsWith('Bearer ')
          ? `Bearer ${token}`
          : token;
    }
    return headers;
  }

  private buildUrl(endpoint: string): string {
    return `${this.params.base_url.replace(/\/$/, '')}${endpoint}`;
  }

  private async request(
    method: 'GET' | 'POST',
    endpoint: string,
    body: unknown,
    startTime: number
  ): Promise<WorkflowResult> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.params.timeout);

    try {
      const response = await fetch(this.buildUrl(endpoint), {
        method,
        headers: this.buildHeaders(),
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      const data = await response.json().catch(() => ({}));

      return {
        success: response.ok,
        operation: this.params.operation,
        system: this.params.system,
        workflow_id: data.workflow_id || this.params.workflow_id,
        status: data.status,
        results: data.results,
        metrics: data.metrics,
        error: response.ok ? undefined : data.error || response.statusText,
        timing: Date.now() - startTime,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: this.params.operation,
        system: this.params.system,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }

  private async simpleRequest(
    method: 'GET' | 'POST',
    endpoint: string,
    startTime: number
  ): Promise<WorkflowResult> {
    return this.request(method, endpoint, undefined, startTime);
  }

  private async startWorkflow(startTime: number): Promise<WorkflowResult> {
    switch (this.params.system) {
      case 'decomposition':
        return this.request(
          'POST',
          '/api/workflows/decomposition/start',
          {
            problem_statement: this.params.problem_statement,
            strategy: this.params.decomposition_strategy,
            max_depth: this.params.max_depth,
            parameters: {
              ...this.params.parameters,
              maker: this.params.maker_config,
              adaptive: this.params.adaptive_config,
            },
            async: this.params.async_execution,
            callback_url: this.params.callback_url,
          },
          startTime
        );
      case 'evolutionary':
        return this.request(
          'POST',
          '/api/workflows/evolutionary/start',
          {
            population_size: this.params.population_size,
            generations: this.params.generations,
            mutation_rate: this.params.mutation_rate,
            crossover_rate: this.params.crossover_rate,
            selection_method: this.params.selection_method,
            fitness_function: this.params.fitness_function,
            parameters: {
              ...this.params.parameters,
              maker: this.params.maker_config,
              adaptive: this.params.adaptive_config,
            },
            async: this.params.async_execution,
            callback_url: this.params.callback_url,
          },
          startTime
        );
      case 'mdap_maker':
        return this.request(
          'POST',
          '/api/workflows/mdap/start',
          {
            config: this.params.mdap_config,
            parameters: {
              ...this.params.parameters,
              maker: this.params.maker_config,
              adaptive: this.params.adaptive_config,
            },
            async: this.params.async_execution,
            callback_url: this.params.callback_url,
          },
          startTime
        );
      case 'adversarial':
        return this.request(
          'POST',
          '/api/workflows/adversarial/start',
          {
            test_types: this.params.test_types,
            attack_vectors: this.params.attack_vectors,
            defense_mechanisms: this.params.defense_mechanisms,
            test_duration: this.params.test_duration,
            parameters: this.params.parameters,
            async: this.params.async_execution,
            callback_url: this.params.callback_url,
          },
          startTime
        );
      default:
        return {
          success: false,
          operation: this.params.operation,
          system: this.params.system,
          error: `Unsupported workflow system: ${this.params.system}`,
          timing: Date.now() - startTime,
        };
    }
  }

  private async configureWorkflow(startTime: number): Promise<WorkflowResult> {
    if (!this.params.workflow_definition) {
      throw new Error('workflow_definition is required');
    }

    return this.request(
      'POST',
      '/api/workflows/configure',
      {
        workflow_name: this.params.workflow_name,
        definition: this.params.workflow_definition,
      },
      startTime
    );
  }

  private async batchExecute(startTime: number): Promise<WorkflowResult> {
    const workflows = (this.params.workflow_definition as any)?.workflows;
    if (!Array.isArray(workflows)) {
      throw new Error('workflow_definition.workflows array is required');
    }

    return this.request(
      'POST',
      '/api/workflows/batch',
      {
        workflows,
        parameters: this.params.parameters,
      },
      startTime
    );
  }

  private async chainWorkflows(startTime: number): Promise<WorkflowResult> {
    const chain = (this.params.workflow_definition as any)?.chain;
    if (!Array.isArray(chain)) {
      throw new Error('workflow_definition.chain array is required');
    }

    return this.request(
      'POST',
      '/api/workflows/chain',
      {
        chain,
        parameters: this.params.parameters,
      },
      startTime
    );
  }
}

export default OpenEvolveWorkflowOrchestratorBubble;
