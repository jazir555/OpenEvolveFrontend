/**
 * CrewAI Orchestration Service Bubble
 *
 * Provides orchestration hooks for CrewAI-based workflows in OpenEvolve.
 * Replaces Hephaestus with CrewAI as the orchestration layer.
 */

import { z } from 'zod';
import { ServiceBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';
import { ResilienceWrapper, DEFAULT_RESILIENCE_CONFIG } from '../adapters/resilience';

// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================

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

const CrewAIParamsSchema = z.object({
  operation: CrewAIOperationSchema.describe('CrewAI operation'),

  baseUrl: z.string().url().default('http://localhost:8000')
    .describe('CrewAI service URL'),
  apiKey: z.string().optional().describe('Optional API key for authentication'),
  timeout: z.number().min(1000).max(300000).default(60000),

  workflowId: z.string().optional().describe('Workflow ID'),
  problemStatement: z.string().optional().describe('Problem statement to solve'),
  executionMethod: CrewAIExecutionMethodSchema.default('auto'),

  phaseNumber: z.number().min(1).max(6).optional().describe('Phase number'),
  phaseInput: z.record(z.unknown()).optional().describe('Phase input payload'),

  taskName: z.string().optional().describe('Task name for delegation'),
  taskDescription: z.string().optional().describe('Task description'),
  teamName: z.string().optional().describe('Team name for delegation'),

  parameters: z.record(z.unknown()).optional().describe('Additional parameters'),
});

export type CrewAIParamsInput = z.input<typeof CrewAIParamsSchema>;
export type CrewAIParams = z.output<typeof CrewAIParamsSchema>;

// ============================================================================
// RESULT SCHEMA
// ============================================================================

const CrewAIResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  workflowId: z.string().optional(),
  status: z.string().optional(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

export type CrewAIResult = z.output<typeof CrewAIResultSchema>;

// ============================================================================
// CREWAI BUBBLE
// ============================================================================

export class CrewAIBubble extends ServiceBubble<CrewAIParams, CrewAIResult> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName = 'crewai' as const;
  static readonly type = 'service' as const;
  static readonly schema = CrewAIParamsSchema;
  static readonly resultSchema = CrewAIResultSchema;
  static readonly credentialType = 'crewai_api_key' as const;

  static readonly shortDescription = 'CrewAI orchestration layer for OpenEvolve workflows';
  static readonly longDescription = `
    CrewAI orchestration bubble for OpenEvolve workflows.

    Features:
    - Execute full workflows and individual phases
    - Retrieve workflow status and results
    - Delegate tasks to CrewAI teams
    - Health checks and capability discovery
  `;

  private resilience: ResilienceWrapper;

  constructor(params: CrewAIParamsInput, context?: BubbleContext) {
    super(params, context);
    this.resilience = new ResilienceWrapper('crewai', DEFAULT_RESILIENCE_CONFIG);
  }

  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.params.apiKey) {
      headers['Authorization'] = `Bearer ${this.params.apiKey}`;
    }
    return headers;
  }

  private buildUrl(endpoint: string): string {
    const base = this.params.baseUrl.endsWith('/')
      ? this.params.baseUrl.slice(0, -1)
      : this.params.baseUrl;
    return `${base}${endpoint}`;
  }

  private async makeRequest(
    method: string,
    endpoint: string,
    body?: unknown
  ): Promise<Response> {
    return fetch(this.buildUrl(endpoint), {
      method,
      headers: this.buildHeaders(),
      body: body ? JSON.stringify(body) : undefined,
    });
  }

  private async healthCheck(): Promise<CrewAIResult> {
    const startTime = Date.now();
    try {
      const response = await this.resilience.execute(
        'crewai-health',
        () => this.makeRequest('GET', '/api/crewai/health'),
        { operation: 'health_check' }
      );
      const timing = Date.now() - startTime;
      const data = await response.json();
      return {
        success: response.ok,
        operation: 'health_check',
        data,
        error: response.ok ? undefined : data.error || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      return {
        success: false,
        operation: 'health_check',
        error: error instanceof Error ? error.message : 'Unknown error',
        timing,
      };
    }
  }

  private async getCapabilities(): Promise<CrewAIResult> {
    const startTime = Date.now();
    try {
      const response = await this.resilience.execute(
        'crewai-capabilities',
        () => this.makeRequest('GET', '/api/crewai/capabilities'),
        { operation: 'get_capabilities' }
      );
      const timing = Date.now() - startTime;
      const data = await response.json();
      return {
        success: response.ok,
        operation: 'get_capabilities',
        data,
        error: response.ok ? undefined : data.error || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      return {
        success: false,
        operation: 'get_capabilities',
        error: error instanceof Error ? error.message : 'Unknown error',
        timing,
      };
    }
  }

  private async executeWorkflow(): Promise<CrewAIResult> {
    const startTime = Date.now();
    try {
      const response = await this.resilience.execute(
        'crewai-execute-workflow',
        () => this.makeRequest('POST', '/api/crewai/workflows', {
          problem_statement: this.params.problemStatement,
          execution_method: this.params.executionMethod,
          parameters: this.params.parameters,
        }),
        { operation: 'execute_workflow' }
      );
      const timing = Date.now() - startTime;
      const data = await response.json();
      return {
        success: response.ok,
        operation: 'execute_workflow',
        workflowId: data.workflow_id || data.workflowId,
        status: data.status,
        data,
        error: response.ok ? undefined : data.error || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      return {
        success: false,
        operation: 'execute_workflow',
        error: error instanceof Error ? error.message : 'Unknown error',
        timing,
      };
    }
  }

  private async executePhase(): Promise<CrewAIResult> {
    if (!this.params.workflowId || !this.params.phaseNumber) {
      throw new Error('workflowId and phaseNumber are required for execute_phase');
    }

    const startTime = Date.now();
    try {
      const response = await this.resilience.execute(
        `crewai-phase-${this.params.workflowId}-${this.params.phaseNumber}`,
        () => this.makeRequest(
          'POST',
          `/api/crewai/workflows/${this.params.workflowId}/phases/${this.params.phaseNumber}`,
          {
            phase_input: this.params.phaseInput,
            parameters: this.params.parameters,
          }
        ),
        { operation: 'execute_phase' }
      );
      const timing = Date.now() - startTime;
      const data = await response.json();
      return {
        success: response.ok,
        operation: 'execute_phase',
        workflowId: this.params.workflowId,
        status: data.status,
        data,
        error: response.ok ? undefined : data.error || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      return {
        success: false,
        operation: 'execute_phase',
        workflowId: this.params.workflowId,
        error: error instanceof Error ? error.message : 'Unknown error',
        timing,
      };
    }
  }

  private async getStatus(): Promise<CrewAIResult> {
    if (!this.params.workflowId) {
      throw new Error('workflowId is required for get_status');
    }
    const startTime = Date.now();
    try {
      const response = await this.resilience.execute(
        `crewai-status-${this.params.workflowId}`,
        () => this.makeRequest('GET', `/api/crewai/workflows/${this.params.workflowId}/status`),
        { operation: 'get_status' }
      );
      const timing = Date.now() - startTime;
      const data = await response.json();
      return {
        success: response.ok,
        operation: 'get_status',
        workflowId: this.params.workflowId,
        status: data.status,
        data,
        error: response.ok ? undefined : data.error || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      return {
        success: false,
        operation: 'get_status',
        workflowId: this.params.workflowId,
        error: error instanceof Error ? error.message : 'Unknown error',
        timing,
      };
    }
  }

  private async getResults(): Promise<CrewAIResult> {
    if (!this.params.workflowId) {
      throw new Error('workflowId is required for get_results');
    }
    const startTime = Date.now();
    try {
      const response = await this.resilience.execute(
        `crewai-results-${this.params.workflowId}`,
        () => this.makeRequest('GET', `/api/crewai/workflows/${this.params.workflowId}/results`),
        { operation: 'get_results' }
      );
      const timing = Date.now() - startTime;
      const data = await response.json();
      return {
        success: response.ok,
        operation: 'get_results',
        workflowId: this.params.workflowId,
        status: data.status,
        data,
        error: response.ok ? undefined : data.error || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      return {
        success: false,
        operation: 'get_results',
        workflowId: this.params.workflowId,
        error: error instanceof Error ? error.message : 'Unknown error',
        timing,
      };
    }
  }

  private async listWorkflows(): Promise<CrewAIResult> {
    const startTime = Date.now();
    try {
      const response = await this.resilience.execute(
        'crewai-list-workflows',
        () => this.makeRequest('GET', '/api/crewai/workflows'),
        { operation: 'list_workflows' }
      );
      const timing = Date.now() - startTime;
      const data = await response.json();
      return {
        success: response.ok,
        operation: 'list_workflows',
        data,
        error: response.ok ? undefined : data.error || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      return {
        success: false,
        operation: 'list_workflows',
        error: error instanceof Error ? error.message : 'Unknown error',
        timing,
      };
    }
  }

  private async delegateTask(): Promise<CrewAIResult> {
    if (!this.params.taskDescription) {
      throw new Error('taskDescription is required for delegate_task');
    }
    const startTime = Date.now();
    try {
      const response = await this.resilience.execute(
        'crewai-delegate-task',
        () => this.makeRequest('POST', '/api/crewai/tasks', {
          task_name: this.params.taskName,
          task_description: this.params.taskDescription,
          team_name: this.params.teamName,
          context: this.params.parameters,
        }),
        { operation: 'delegate_task' }
      );
      const timing = Date.now() - startTime;
      const data = await response.json();
      return {
        success: response.ok,
        operation: 'delegate_task',
        workflowId: data.workflow_id || data.workflowId,
        status: data.status,
        data,
        error: response.ok ? undefined : data.error || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      return {
        success: false,
        operation: 'delegate_task',
        error: error instanceof Error ? error.message : 'Unknown error',
        timing,
      };
    }
  }

  async action(): Promise<CrewAIResult> {
    switch (this.params.operation) {
      case 'health_check':
        return this.healthCheck();
      case 'get_capabilities':
        return this.getCapabilities();
      case 'execute_workflow':
        return this.executeWorkflow();
      case 'execute_phase':
        return this.executePhase();
      case 'get_status':
        return this.getStatus();
      case 'get_results':
        return this.getResults();
      case 'list_workflows':
        return this.listWorkflows();
      case 'delegate_task':
        return this.delegateTask();
      default:
        return {
          success: false,
          operation: this.params.operation,
          error: `Unknown operation: ${this.params.operation}`,
          timing: 0,
        };
    }
  }
}

export default CrewAIBubble;
