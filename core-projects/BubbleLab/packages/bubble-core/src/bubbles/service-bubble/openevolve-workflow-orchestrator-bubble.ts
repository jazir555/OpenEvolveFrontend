import { z } from 'zod';
import type {
  BubbleOperationResult,
  BubbleResult,
} from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
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

const DEFAULT_BASE_URL = 'http://localhost:8000';

/**
 * Resolve the OpenEvolve server base URL.
 *
 * Precedence: explicit `baseUrl` param > env
 * (`OPENEVOLVE_API_URL` / `OPENEVOLVE_BASE_URL` / `OPENEVOLVE_API_BASE_URL`) >
 * `http://localhost:8000`. Resolved lazily (per request) so tests can point the
 * bubble at a live or dead server via env at runtime.
 */
const resolveBaseUrl = (explicit?: string): string => {
  const fromEnv =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL ||
        process.env.OPENEVOLVE_BASE_URL ||
        process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base =
    (explicit && explicit.trim().length > 0 ? explicit : undefined) ||
    (fromEnv.trim().length > 0 ? fromEnv : undefined) ||
    DEFAULT_BASE_URL;
  return base.replace(/\/+$/, '');
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const WorkflowParamsSchema = z.object({
  operation: WorkflowOperationSchema,
  system: WorkflowSystemSchema,

  baseUrl: z.string().url().optional(),
  timeout: z.number().min(1000).max(300000).default(60000),
  headers: z.record(z.string()).optional(),
  authToken: z.string().optional(),
  authHeader: z.string().default('Authorization'),

  workflowId: z.string().optional(),
  workflowName: z.string().optional(),
  workflowDefinition: z.record(z.unknown()).optional(),

  problemStatement: z.string().optional(),

  // Optional real-LLM wiring. When omitted, the OpenEvolve server defaults to
  // its offline deterministic mock backend. Supplying `llmApiKey` + `llmModel`
  // makes the bridge attempt a live model run.
  llmApiKey: z.string().optional(),
  llmModel: z.string().optional(),
  llmApiBase: z.string().optional(),
  llmProvider: z.string().optional(),

  decompositionStrategy: z
    .enum(['hierarchical', 'parallel', 'adaptive'])
    .default('adaptive'),
  maxDepth: z.number().min(1).max(10).default(5),

  populationSize: z.number().min(1).max(1000).default(100),
  generations: z.number().min(1).max(1000).default(50),
  mutationRate: z.number().min(0).max(1).default(0.1),
  crossoverRate: z.number().min(0).max(1).default(0.8),
  selectionMethod: z
    .enum(['tournament', 'roulette', 'rank'])
    .default('tournament'),
  fitnessFunction: z.string().optional(),

  mdapConfig: z
    .object({
      enableMcts: z.boolean().default(true),
      enableAdversarial: z.boolean().default(true),
      enableEvolution: z.boolean().default(true),
      iterations: z.number().default(100),
    })
    .optional(),

  makerConfig: z.record(z.unknown()).optional(),
  adaptiveConfig: z.record(z.unknown()).optional(),

  testTypes: z
    .array(z.enum(['red_team', 'blue_team', 'purple_team', 'automated']))
    .default(['automated']),
  attackVectors: z.array(z.string()).optional(),
  defenseMechanisms: z.array(z.string()).optional(),
  testDuration: z.number().default(300),

  asyncExecution: z.boolean().default(false),
  callbackUrl: z.string().url().optional(),
  parameters: z.record(z.unknown()).optional(),
});

type WorkflowParams = z.input<typeof WorkflowParamsSchema> & ServiceBubbleParams;

const WorkflowResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  system: z.string(),
  workflowId: z.string().optional(),
  status: z.string().optional(),
  // `result` mirrors GET /api/v1/runs/{run_id}.result (best_score, best_code, metrics, ...)
  result: z.record(z.unknown()).optional(),
  results: z.record(z.unknown()).optional(),
  metrics: z.record(z.unknown()).optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type WorkflowResult = z.output<typeof WorkflowResultSchema> &
  BubbleOperationResult;

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

    Targets the OpenEvolve server /api/v1/* contract:
      - GET  {baseUrl}/api/v1/health                -> health_check
      - POST {baseUrl}/api/v1/workflows/orchestrate -> start_workflow (returns workflowId)
      - GET  {baseUrl}/api/v1/runs/{workflowId}     -> get_status / get_results
  `;
  static readonly alias = 'openevolve-workflow-orchestrator';

  constructor(params: WorkflowParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.authToken;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  /**
   * Return the operation payload flattened onto the standard bubble envelope so
   * callers can read `workflowId` / `status` / `result` directly off `action()`
   * while `data` / `executionId` / `timestamp` remain available.
   */
  public async action(): Promise<
    BubbleResult<WorkflowResult> & WorkflowResult
  > {
    const envelope = await super.action();
    const payload = (envelope.data ?? {}) as WorkflowResult;
    return {
      ...envelope,
      ...payload,
      data: payload,
    } as BubbleResult<WorkflowResult> & WorkflowResult;
  }

  protected async performAction(): Promise<WorkflowResult> {
    const startTime = Date.now();
    try {
      switch (this.params.operation as string) {
        case 'start_workflow':
          return await this.startWorkflow(startTime);
        case 'get_status':
        case 'get_results':
          return await this.simpleRequest(
            'GET',
            `/api/v1/runs/${this.requireWorkflowId()}`,
            startTime
          );
        case 'health_check':
          return await this.simpleRequest('GET', '/api/v1/health', startTime);
        case 'stop_workflow':
          return await this.simpleRequest(
            'POST',
            `/api/v1/workflows/${this.requireWorkflowId()}/stop`,
            startTime
          );
        case 'pause_workflow':
          return await this.simpleRequest(
            'POST',
            `/api/v1/workflows/${this.requireWorkflowId()}/pause`,
            startTime
          );
        case 'resume_workflow':
          return await this.simpleRequest(
            'POST',
            `/api/v1/workflows/${this.requireWorkflowId()}/resume`,
            startTime
          );
        case 'configure':
          return await this.configureWorkflow(startTime);
        case 'batch_execute':
          return await this.batchExecute(startTime);
        case 'chain_workflows':
          return await this.chainWorkflows(startTime);
        default:
          return {
            success: false,
            operation: this.params.operation as string,
            system: this.params.system,
            error: `Unsupported operation: ${this.params.operation as string}`,
            timing: Date.now() - startTime,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: this.params.operation as string,
        system: this.params.system,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }

  private requireWorkflowId(): string {
    if (!this.params.workflowId) {
      throw new Error('workflowId is required');
    }
    return this.params.workflowId;
  }

  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.params.headers) {
      Object.assign(headers, this.params.headers);
    }
    if (this.params.authToken) {
      const headerName = this.params.authHeader || 'Authorization';
      const token = this.params.authToken;
      headers[headerName] =
        headerName.toLowerCase() === 'authorization' &&
        !token.startsWith('Bearer ')
          ? `Bearer ${token}`
          : token;
    }
    return headers;
  }

  private buildUrl(endpoint: string): string {
    return `${resolveBaseUrl(this.params.baseUrl)}${endpoint}`;
  }

  private async request(
    method: 'GET' | 'POST',
    endpoint: string,
    body: unknown,
    startTime: number
  ): Promise<WorkflowResult> {
    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(),
      this.params.timeout ?? 60000
    );

    try {
      const response = await fetch(this.buildUrl(endpoint), {
        method,
        headers: this.buildHeaders(),
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      const data = (await response
        .json()
        .catch(() => ({}))) as Record<string, unknown>;

      // POST /api/v1/workflows/orchestrate -> { workflowId, status }
      // GET  /api/v1/runs/{run_id}         -> { run_id, status, result, error }
      const returnedId =
        typeof data.workflowId === 'string'
          ? data.workflowId
          : typeof data.workflow_id === 'string'
            ? data.workflow_id
            : typeof data.run_id === 'string'
              ? data.run_id
              : undefined;

      const responseError =
        typeof data.error === 'string' && data.error.length > 0
          ? data.error
          : undefined;

      return {
        success: response.ok,
        operation: this.params.operation as string,
        system: this.params.system,
        workflowId: returnedId ?? this.params.workflowId,
        status: typeof data.status === 'string' ? data.status : undefined,
        result: isRecord(data.result) ? data.result : undefined,
        results: isRecord(data.results) ? data.results : undefined,
        metrics: isRecord(data.metrics) ? data.metrics : undefined,
        error: response.ok
          ? undefined
          : responseError ||
            `${response.status} ${response.statusText || 'request failed'}`,
        timing: Date.now() - startTime,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: this.params.operation as string,
        system: this.params.system,
        workflowId: this.params.workflowId,
        error: `OpenEvolve request failed (${method} ${endpoint}): ${message}`,
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

  /**
   * POST {baseUrl}/api/v1/workflows/orchestrate
   *
   * The server accepts camelCase `system` / `problemStatement` / `generations` /
   * `populationSize` and (optionally) an `llm` object holding real-model config.
   * Replies 202 `{ workflowId, status: 'running' }`.
   */
  private async startWorkflow(startTime: number): Promise<WorkflowResult> {
    if (
      !this.params.problemStatement ||
      this.params.problemStatement.trim().length === 0
    ) {
      return {
        success: false,
        operation: this.params.operation as string,
        system: this.params.system,
        error: 'problemStatement is required to start an OpenEvolve workflow',
        timing: Date.now() - startTime,
      };
    }

    // Only include `llm` when at least a model is named; the server treats a
    // missing/absent `llm` as a request for the offline mock backend.
    const llm: Record<string, string> = {};
    if (this.params.llmModel) {
      llm.name = this.params.llmModel;
    }
    if (this.params.llmApiKey) {
      llm.api_key = this.params.llmApiKey;
    }
    if (this.params.llmApiBase) {
      llm.api_base = this.params.llmApiBase;
    }
    if (this.params.llmProvider) {
      llm.provider = this.params.llmProvider;
    }

    return this.request(
      'POST',
      '/api/v1/workflows/orchestrate',
      {
        system: this.params.system,
        problemStatement: this.params.problemStatement,
        generations: this.params.generations,
        populationSize: this.params.populationSize,
        ...(this.params.parameters
          ? { parameters: this.params.parameters }
          : {}),
        ...(Object.keys(llm).length > 0 ? { llm } : {}),
      },
      startTime
    );
  }

  private async configureWorkflow(startTime: number): Promise<WorkflowResult> {
    if (!this.params.workflowDefinition) {
      throw new Error('workflowDefinition is required');
    }

    return this.request(
      'POST',
      '/api/v1/workflows/configure',
      {
        workflowName: this.params.workflowName,
        definition: this.params.workflowDefinition,
      },
      startTime
    );
  }

  private async batchExecute(startTime: number): Promise<WorkflowResult> {
    const workflows = this.params.workflowDefinition?.workflows;
    if (!Array.isArray(workflows)) {
      throw new Error('workflowDefinition.workflows array is required');
    }

    return this.request(
      'POST',
      '/api/v1/workflows/batch',
      {
        workflows,
        parameters: this.params.parameters,
      },
      startTime
    );
  }

  private async chainWorkflows(startTime: number): Promise<WorkflowResult> {
    const chain = this.params.workflowDefinition?.chain;
    if (!Array.isArray(chain)) {
      throw new Error('workflowDefinition.chain array is required');
    }

    return this.request(
      'POST',
      '/api/v1/workflows/chain',
      {
        chain,
        parameters: this.params.parameters,
      },
      startTime
    );
  }
}

export default OpenEvolveWorkflowOrchestratorBubble;
