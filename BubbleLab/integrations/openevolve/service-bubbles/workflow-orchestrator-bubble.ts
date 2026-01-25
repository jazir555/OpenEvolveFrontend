/**
 * Workflow Orchestrator Service Bubble
 *
 * Orchestrates OpenEvolve workflow systems including Decomposition Engine,
 * Evolutionary Optimization, MDAP Maker, and Adversarial Testing.
 */

import { z } from 'zod';
import { HttpBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';

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
  operation: WorkflowOperationSchema.describe('Workflow operation'),
  system: WorkflowSystemSchema.describe('Workflow system to use'),

  // Workflow configuration
  workflowId: z.string().optional().describe('Workflow ID'),
  workflowName: z.string().optional().describe('Workflow name'),
  workflowDefinition: z.record(z.unknown()).optional().describe('Workflow definition'),

  // Decomposition engine
  problemStatement: z.string().optional().describe('Problem to decompose'),
  decompositionStrategy: z.enum(['hierarchical', 'parallel', 'adaptive']).default('adaptive'),
  maxDepth: z.number().min(1).max(10).default(5).describe('Maximum decomposition depth'),

  // Evolutionary optimization
  populationSize: z.number().min(10).max(1000).default(100),
  generations: z.number().min(1).max(1000).default(50),
  mutationRate: z.number().min(0).max(1).default(0.1),
  crossoverRate: z.number().min(0).max(1).default(0.8),
  selectionMethod: z.enum(['tournament', 'roulette', 'rank']).default('tournament'),
  fitnessFunction: z.string().optional().describe('Fitness function to use'),

  // MDAP maker
  mdapConfig: z.object({
    enableMcts: z.boolean().default(true),
    enableAdversarial: z.boolean().default(true),
    enableEvolution: z.boolean().default(true),
    iterations: z.number().default(100),
  }).optional(),

  // Adversarial testing
  testTypes: z.array(z.enum(['red_team', 'blue_team', 'purple_team', 'automated'])).default(['automated']),
  attackVectors: z.array(z.string()).optional().describe('Attack vectors to test'),
  defenseMechanisms: z.array(z.string()).optional().describe('Defense mechanisms to test'),
  testDuration: z.number().default(300).describe('Test duration in seconds'),

  // Common parameters
  baseUrl: z.string().url().default('http://localhost:8000'),
  timeout: z.number().min(1000).max(300000).default(60000),
  asyncExecution: z.boolean().default(false).describe('Execute workflow asynchronously'),
  callbackUrl: z.string().url().optional().describe('Webhook URL for async completion'),
  parameters: z.record(z.unknown()).optional().describe('Additional workflow parameters'),
});

type WorkflowParamsInput = z.input<typeof WorkflowParamsSchema>;
type WorkflowParams = z.output<typeof WorkflowParamsSchema>;

const WorkflowResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  system: z.string(),
  workflowId: z.string().optional(),
  status: z.enum(['pending', 'running', 'paused', 'completed', 'failed', 'cancelled']).optional(),
  results: z.record(z.unknown()).optional(),
  metrics: z.object({
    executionTime: z.number().optional(),
    progress: z.number().optional(),
    generation: z.number().optional(),
    bestFitness: z.number().optional(),
    testsPassed: z.number().optional(),
    testsFailed: z.number().optional(),
  }).optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type WorkflowResult = z.output<typeof WorkflowResultSchema>;

export class WorkflowOrchestratorBubble {
  private http: HttpBubble;
  private params: WorkflowParams;
  private context?: BubbleContext;

  constructor(params: WorkflowParamsInput, context?: BubbleContext) {
    this.params = WorkflowParamsSchema.parse(params);
    this.context = context;

    this.http = new HttpBubble({
      url: this.params.baseUrl,
      method: 'GET',
      timeout: this.params.timeout,
    }, context);
  }

  private buildUrl(endpoint: string): string {
    return `${this.params.baseUrl}${endpoint}`;
  }

  private async request(method: string, endpoint: string, body?: unknown): Promise<WorkflowResult> {
    const startTime = Date.now();

    try {
      const options: RequestInit = {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
      };

      if (body) {
        options.body = JSON.stringify(body);
      }

      const response = await fetch(this.buildUrl(endpoint), options);
      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: this.params.operation,
        system: this.params.system,
        workflowId: data.workflow_id || this.params.workflowId,
        status: data.status,
        results: data.results,
        metrics: data.metrics,
        error: response.ok ? undefined : data.error || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: this.params.operation,
        system: this.params.system,
        error: errorMessage,
        timing,
      };
    }
  }

  public async startDecompositionWorkflow(): Promise<WorkflowResult> {
    return this.request('POST', '/api/workflows/decomposition/start', {
      problem_statement: this.params.problemStatement,
      strategy: this.params.decompositionStrategy,
      max_depth: this.params.maxDepth,
      parameters: this.params.parameters,
      async: this.params.asyncExecution,
      callback_url: this.params.callbackUrl,
    });
  }

  public async startEvolutionaryWorkflow(): Promise<WorkflowResult> {
    return this.request('POST', '/api/workflows/evolutionary/start', {
      population_size: this.params.populationSize,
      generations: this.params.generations,
      mutation_rate: this.params.mutationRate,
      crossover_rate: this.params.crossoverRate,
      selection_method: this.params.selectionMethod,
      fitness_function: this.params.fitnessFunction,
      parameters: this.params.parameters,
      async: this.params.asyncExecution,
      callback_url: this.params.callbackUrl,
    });
  }

  public async startMdapMakerWorkflow(): Promise<WorkflowResult> {
    return this.request('POST', '/api/workflows/mdap/start', {
      config: this.params.mdapConfig,
      parameters: this.params.parameters,
      async: this.params.asyncExecution,
      callback_url: this.params.callbackUrl,
    });
  }

  public async startAdversarialWorkflow(): Promise<WorkflowResult> {
    return this.request('POST', '/api/workflows/adversarial/start', {
      test_types: this.params.testTypes,
      attack_vectors: this.params.attackVectors,
      defense_mechanisms: this.params.defenseMechanisms,
      test_duration: this.params.testDuration,
      parameters: this.params.parameters,
      async: this.params.asyncExecution,
      callback_url: this.params.callbackUrl,
    });
  }

  public async startWorkflow(): Promise<WorkflowResult> {
    switch (this.params.system) {
      case 'decomposition':
        return this.startDecompositionWorkflow();
      case 'evolutionary':
        return this.startEvolutionaryWorkflow();
      case 'mdap_maker':
        return this.startMdapMakerWorkflow();
      case 'adversarial':
        return this.startAdversarialWorkflow();
      default:
        return {
          success: false,
          operation: 'start_workflow',
          system: this.params.system,
          error: `Unknown workflow system: ${this.params.system}`,
          timing: 0,
        };
    }
  }

  public async stopWorkflow(): Promise<WorkflowResult> {
    if (!this.params.workflowId) {
      throw new Error('workflowId is required for stop_workflow operation');
    }

    return this.request('POST', `/api/workflows/${this.params.workflowId}/stop`);
  }

  public async getStatus(): Promise<WorkflowResult> {
    if (!this.params.workflowId) {
      throw new Error('workflowId is required for get_status operation');
    }

    return this.request('GET', `/api/workflows/${this.params.workflowId}/status`);
  }

  public async getResults(): Promise<WorkflowResult> {
    if (!this.params.workflowId) {
      throw new Error('workflowId is required for get_results operation');
    }

    return this.request('GET', `/api/workflows/${this.params.workflowId}/results`);
  }

  public async healthCheck(): Promise<WorkflowResult> {
    return this.request('GET', '/api/workflows/health');
  }

  public async configure(): Promise<WorkflowResult> {
    if (!this.params.workflowDefinition) {
      throw new Error('workflowDefinition is required for configure operation');
    }

    return this.request('POST', '/api/workflows/configure', {
      workflow_name: this.params.workflowName,
      definition: this.params.workflowDefinition,
    });
  }

  public async batchExecute(): Promise<WorkflowResult> {
    if (!this.params.workflowDefinition || !Array.isArray(this.params.workflowDefinition.workflows)) {
      throw new Error('workflowDefinition with workflows array is required for batch_execute');
    }

    return this.request('POST', '/api/workflows/batch', {
      workflows: this.params.workflowDefinition.workflows,
      parameters: this.params.parameters,
    });
  }

  public async chainWorkflows(): Promise<WorkflowResult> {
    if (!this.params.workflowDefinition || !Array.isArray(this.params.workflowDefinition.chain)) {
      throw new Error('workflowDefinition with chain array is required for chain_workflows');
    }

    return this.request('POST', '/api/workflows/chain', {
      chain: this.params.workflowDefinition.chain,
      parameters: this.params.parameters,
    });
  }

  public async action(): Promise<WorkflowResult> {
    switch (this.params.operation) {
      case 'start_workflow':
        return this.startWorkflow();
      case 'stop_workflow':
        return this.stopWorkflow();
      case 'get_status':
        return this.getStatus();
      case 'get_results':
        return this.getResults();
      case 'health_check':
        return this.healthCheck();
      case 'configure':
        return this.configure();
      case 'batch_execute':
        return this.batchExecute();
      case 'chain_workflows':
        return this.chainWorkflows();
      default:
        return {
          success: false,
          operation: this.params.operation,
          system: this.params.system,
          error: `Unknown operation: ${this.params.operation}`,
          timing: 0,
        };
    }
  }
}

export default WorkflowOrchestratorBubble;
