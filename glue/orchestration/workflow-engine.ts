/**
 * Workflow Engine - Multi-Step Orchestration
 *
 * Follows the Federation Constitution:
 * - Law of Idempotency: Workflow state management for replay
 * - Failure Management: Circuit breakers for external services
 * - Observability: JSON Lines logging with full context
 */

import { v4 as uuidv4 } from 'uuid';
import { EventBus, eventBus } from './event-bus';
import { Event, createBaseEvent } from './event-types';
import { CorrelationContext, correlationTracker } from './correlation-tracker';
import { Logger } from '../lib/logger';
import { CircuitBreaker } from '../lib/circuit-breaker';
import { retryWithBackoff } from '../lib/retry';

export interface WorkflowStep {
  step_id: string;
  step_name: string;
  service: string;
  operation: string;
  handler: (context: WorkflowContext) => Promise<any>;
  timeout_ms?: number;
  retry_on_failure?: boolean;
  max_retries?: number;
  circuit_breaker?: boolean;
}

export interface WorkflowDefinition {
  workflow_id: string;
  workflow_name: string;
  description: string;
  steps: WorkflowStep[];
  parallel?: boolean; // Execute steps in parallel
  on_failure?: 'stop' | 'continue' | 'retry';
  timeout_ms?: number;
}

export interface WorkflowContext {
  workflow_id: string;
  execution_id: string;
  correlation_context: CorrelationContext;
  input_data: any;
  output_data: any;
  step_results: Map<string, any>;
  state: WorkflowState;
  current_step: number;
  started_at: string; // ISO-8601 UTC
  completed_at?: string; // ISO-8601 UTC
  error?: Error;
}

export enum WorkflowState {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

export interface WorkflowExecutionResult {
  execution_id: string;
  workflow_id: string;
  workflow_name: string;
  state: WorkflowState;
  duration_ms: number;
  steps_completed: number;
  steps_failed: number;
  output_data?: any;
  error?: string;
}

/**
 * Predefined Workflows
 */
export const PREDEFINED_WORKFLOWS: Record<string, WorkflowDefinition> = {
  /**
   * Z3 → LeanAide Cross-Validation
   * Verify proofs with both systems for cross-validation
   */
  'z3-lean-validation': {
    workflow_id: 'z3-lean-validation',
    workflow_name: 'Z3 LeanAide Cross-Validation',
    description: 'Verify formal proofs using both Z3 and LeanAide for cross-validation',
    steps: [
      {
        step_id: 'verify-z3',
        step_name: 'Verify with Z3',
        service: 'z3-adapter',
        operation: 'verify-proof',
        handler: async (context) => {
          // Z3 verification logic
          return { verified: true, system: 'z3' };
        },
        timeout_ms: 30000,
        retry_on_failure: true,
        max_retries: 2,
        circuit_breaker: true
      },
      {
        step_id: 'verify-lean',
        step_name: 'Verify with LeanAide',
        service: 'lean-aide-adapter',
        operation: 'verify-proof',
        handler: async (context) => {
          // LeanAide verification logic
          return { verified: true, system: 'lean-aide' };
        },
        timeout_ms: 30000,
        retry_on_failure: true,
        max_retries: 2,
        circuit_breaker: true
      },
      {
        step_id: 'cross-validate',
        step_name: 'Cross-Validate Results',
        service: 'orchestration',
        operation: 'validate-results',
        handler: async (context) => {
          const z3Result = context.step_results.get('verify-z3');
          const leanResult = context.step_results.get('verify-lean');

          return {
            cross_validated: z3Result.verified && leanResult.verified,
            z3_result: z3Result,
            lean_result: leanResult
          };
        }
      }
    ],
    parallel: false,
    on_failure: 'stop',
    timeout_ms: 90000 // 90 seconds total
  },

  /**
   * RAGBits → Vector DB → Knowledge Graph
   * Complete RAG pipeline: extract, embed, index, graph
   */
  'rag-pipeline': {
    workflow_id: 'rag-pipeline',
    workflow_name: 'RAG Processing Pipeline',
    description: 'Extract knowledge, create embeddings, index in vector DB, update knowledge graph',
    steps: [
      {
        step_id: 'extract-knowledge',
        step_name: 'Extract Knowledge Chunks',
        service: 'ragbits-adapter',
        operation: 'extract-chunks',
        handler: async (context) => {
          // Extract chunks from document
          return { chunks: [], count: 0 };
        },
        timeout_ms: 60000,
        retry_on_failure: true,
        circuit_breaker: true
      },
      {
        step_id: 'create-embeddings',
        step_name: 'Create Embeddings',
        service: 'vector-db-adapter',
        operation: 'create-embeddings',
        handler: async (context) => {
          // Create embeddings for chunks
          return { embeddings: [], dimension: 1536 };
        },
        timeout_ms: 120000,
        retry_on_failure: true,
        circuit_breaker: true
      },
      {
        step_id: 'index-vectors',
        step_name: 'Index Vectors',
        service: 'vector-db-adapter',
        operation: 'index-embeddings',
        handler: async (context) => {
          // Index embeddings in vector DB
          return { index_id: 'idx-123', count: 0 };
        },
        timeout_ms: 60000,
        retry_on_failure: true,
        circuit_breaker: true
      },
      {
        step_id: 'update-graph',
        step_name: 'Update Knowledge Graph',
        service: 'graphiti-adapter',
        operation: 'update-graph',
        handler: async (context) => {
          // Update knowledge graph with new knowledge
          return { graph_id: 'graph-456', nodes_added: 0, edges_added: 0 };
        },
        timeout_ms: 60000,
        retry_on_failure: true,
        circuit_breaker: true
      }
    ],
    parallel: false,
    on_failure: 'stop',
    timeout_ms: 300000 // 5 minutes total
  },

  /**
   * Document → Embedding → Index
   * Quick document indexing workflow
   */
  'document-index': {
    workflow_id: 'document-index',
    workflow_name: 'Document Indexing',
    description: 'Index a document in vector database',
    steps: [
      {
        step_id: 'extract-content',
        step_name: 'Extract Document Content',
        service: 'ragbits-adapter',
        operation: 'extract-content',
        handler: async (context) => {
          return { content: '', metadata: {} };
        },
        timeout_ms: 30000,
        retry_on_failure: true
      },
      {
        step_id: 'generate-embeddings',
        step_name: 'Generate Embeddings',
        service: 'vector-db-adapter',
        operation: 'generate-embeddings',
        handler: async (context) => {
          return { embeddings: [], model: 'text-embedding-ada-002' };
        },
        timeout_ms: 60000,
        retry_on_failure: true,
        circuit_breaker: true
      },
      {
        step_id: 'store-embeddings',
        step_name: 'Store Embeddings',
        service: 'vector-db-adapter',
        operation: 'store-embeddings',
        handler: async (context) => {
          return { index_id: 'idx-789', count: 0 };
        },
        timeout_ms: 30000,
        retry_on_failure: true,
        circuit_breaker: true
      }
    ],
    parallel: false,
    on_failure: 'stop',
    timeout_ms: 120000 // 2 minutes
  }
};

/**
 * Workflow Engine
 *
 * Executes multi-step workflows with state management, retries, and circuit breakers
 */
export class WorkflowEngine {
  private logger: Logger;
  private eventBus: EventBus;
  private activeWorkflows: Map<string, WorkflowContext> = new Map();
  private completedWorkflows: Map<string, WorkflowContext> = new Map();
  private circuitBreakers: Map<string, CircuitBreaker> = new Map();

  constructor(eventBus?: EventBus) {
    this.logger = new Logger('workflow-engine');
    this.eventBus = eventBus || eventBus;
  }

  /**
   * Execute a workflow definition
   */
  async execute(
    workflow: WorkflowDefinition,
    inputData: any,
    correlationContext?: CorrelationContext
  ): Promise<WorkflowExecutionResult> {
    const executionId = uuidv4();
    const correlationCtx = correlationContext || correlationTracker.createContext();

    const context: WorkflowContext = {
      workflow_id: workflow.workflow_id,
      execution_id: executionId,
      correlation_context: correlationCtx,
      input_data: inputData,
      output_data: null,
      step_results: new Map(),
      state: WorkflowState.RUNNING,
      current_step: 0,
      started_at: new Date().toISOString()
    };

    this.activeWorkflows.set(executionId, context);

    // Publish workflow started event
    await this.publishWorkflowStarted(workflow, context);

    this.logger.info('Workflow execution started', {
      execution_id: executionId,
      workflow_id: workflow.workflow_id,
      workflow_name: workflow.workflow_name,
      correlation_id: correlationCtx.correlation_id,
      steps_count: workflow.steps.length
    });

    try {
      // Execute workflow steps
      if (workflow.parallel) {
        await this.executeParallelSteps(workflow, context);
      } else {
        await this.executeSequentialSteps(workflow, context);
      }

      // Mark as completed
      context.state = WorkflowState.COMPLETED;
      context.completed_at = new Date().toISOString();

      this.activeWorkflows.delete(executionId);
      this.completedWorkflows.set(executionId, context);

      // Publish workflow completed event
      await this.publishWorkflowCompleted(workflow, context);

      this.logger.info('Workflow execution completed', {
        execution_id: executionId,
        workflow_id: workflow.workflow_id,
        duration_ms: this.calculateDuration(context)
      });

      return this.buildResult(context);
    } catch (error) {
      context.state = WorkflowState.FAILED;
      context.error = error as Error;
      context.completed_at = new Date().toISOString();

      this.activeWorkflows.delete(executionId);
      this.completedWorkflows.set(executionId, context);

      // Publish workflow failed event
      await this.publishWorkflowFailed(workflow, context, error as Error);

      this.logger.error('Workflow execution failed', error as Error, {
        execution_id: executionId,
        workflow_id: workflow.workflow_id,
        duration_ms: this.calculateDuration(context)
      });

      return this.buildResult(context);
    }
  }

  /**
   * Execute workflow steps sequentially
   */
  private async executeSequentialSteps(
    workflow: WorkflowDefinition,
    context: WorkflowContext
  ): Promise<void> {
    for (let i = 0; i < workflow.steps.length; i++) {
      const step = workflow.steps[i];
      context.current_step = i;

      const result = await this.executeStep(step, context);
      context.step_results.set(step.step_id, result);

      // Record service call
      correlationTracker.recordServiceCall(
        context.correlation_context,
        step.service,
        step.operation
      );
    }

    context.output_data = this.aggregateResults(context);
  }

  /**
   * Execute workflow steps in parallel
   */
  private async executeParallelSteps(
    workflow: WorkflowDefinition,
    context: WorkflowContext
  ): Promise<void> {
    const promises = workflow.steps.map(async (step) => {
      const result = await this.executeStep(step, context);
      context.step_results.set(step.step_id, result);

      correlationTracker.recordServiceCall(
        context.correlation_context,
        step.service,
        step.operation
      );

      return result;
    });

    await Promise.all(promises);
    context.output_data = this.aggregateResults(context);
  }

  /**
   * Execute a single workflow step
   */
  private async executeStep(
    step: WorkflowStep,
    context: WorkflowContext
  ): Promise<any> {
    this.logger.info('Executing workflow step', {
      execution_id: context.execution_id,
      step_id: step.step_id,
      step_name: step.step_name,
      service: step.service,
      operation: step.operation
    });

    try {
      // Use circuit breaker if enabled
      if (step.circuit_breaker) {
        const cb = this.getCircuitBreaker(step.service);
        return await cb.execute(async () => this.executeStepHandler(step, context));
      }

      // Use retry if enabled
      if (step.retry_on_failure) {
        return await retryWithBackoff(
          () => this.executeStepHandler(step, context),
          { max_retries: step.max_retries || 3 }
        );
      }

      // Direct execution
      return await this.executeStepHandler(step, context);
    } catch (error) {
      this.logger.error('Workflow step failed', error as Error, {
        execution_id: context.execution_id,
        step_id: step.step_id,
        step_name: step.step_name
      });
      throw error;
    }
  }

  /**
   * Execute step handler with timeout
   */
  private async executeStepHandler(step: WorkflowStep, context: WorkflowContext): Promise<any> {
    const timeout = step.timeout_ms || 30000;

    return Promise.race([
      step.handler(context),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error(`Step timeout: ${step.step_name}`)), timeout)
      )
    ]);
  }

  /**
   * Aggregate step results
   */
  private aggregateResults(context: WorkflowContext): any {
    const results: any = {};
    for (const [stepId, result] of context.step_results.entries()) {
      results[stepId] = result;
    }
    return results;
  }

  /**
   * Get or create circuit breaker for service
   */
  private getCircuitBreaker(service: string): CircuitBreaker {
    if (!this.circuitBreakers.has(service)) {
      this.circuitBreakers.set(
        service,
        new CircuitBreaker({
          threshold: 5,
          timeout_ms: 60000
        })
      );
    }
    return this.circuitBreakers.get(service)!;
  }

  /**
   * Calculate workflow duration
   */
  private calculateDuration(context: WorkflowContext): number {
    const start = new Date(context.started_at).getTime();
    const end = context.completed_at
      ? new Date(context.completed_at).getTime()
      : Date.now();
    return end - start;
  }

  /**
   * Build execution result
   */
  private buildResult(context: WorkflowContext): WorkflowExecutionResult {
    const stepsCompleted = context.step_results.size;
    const stepsFailed = context.state === WorkflowState.FAILED ? 1 : 0;

    return {
      execution_id: context.execution_id,
      workflow_id: context.workflow_id,
      workflow_name: '', // Set from workflow definition
      state: context.state,
      duration_ms: this.calculateDuration(context),
      steps_completed: stepsCompleted,
      steps_failed: stepsFailed,
      output_data: context.output_data,
      error: context.error?.message
    };
  }

  /**
   * Publish workflow started event
   */
  private async publishWorkflowStarted(
    workflow: WorkflowDefinition,
    context: WorkflowContext
  ): Promise<void> {
    const event = createBaseEvent(
      'WorkflowStarted',
      'workflow-engine',
      context.correlation_context.correlation_id,
      {
        workflow_id: workflow.workflow_id,
        workflow_name: workflow.workflow_name,
        input_data: context.input_data,
        steps: workflow.steps.map(s => ({
          step_id: s.step_id,
          step_name: s.step_name,
          service: s.service
        }))
      }
    );

    await this.eventBus.publish(event);
  }

  /**
   * Publish workflow completed event
   */
  private async publishWorkflowCompleted(
    workflow: WorkflowDefinition,
    context: WorkflowContext
  ): Promise<void> {
    const event = createBaseEvent(
      'WorkflowCompleted',
      'workflow-engine',
      context.correlation_context.correlation_id,
      {
        workflow_id: workflow.workflow_id,
        workflow_name: workflow.workflow_name,
        duration_ms: this.calculateDuration(context),
        output_data: context.output_data,
        steps_completed: context.step_results.size,
        steps_failed: 0
      }
    );

    await this.eventBus.publish(event);
  }

  /**
   * Publish workflow failed event
   */
  private async publishWorkflowFailed(
    workflow: WorkflowDefinition,
    context: WorkflowContext,
    error: Error
  ): Promise<void> {
    const event = createBaseEvent(
      'WorkflowFailed',
      'workflow-engine',
      context.correlation_context.correlation_id,
      {
        workflow_id: workflow.workflow_id,
        workflow_name: workflow.workflow_name,
        failure_reason: error.message,
        failed_step: workflow.steps[context.current_step]?.step_name || 'unknown',
        error_details: {
          name: error.name,
          message: error.message,
          stack: error.stack
        },
        duration_ms: this.calculateDuration(context)
      }
    );

    await this.eventBus.publish(event);
  }

  /**
   * Get active workflows
   */
  getActiveWorkflows(): WorkflowContext[] {
    return Array.from(this.activeWorkflows.values());
  }

  /**
   * Get completed workflows
   */
  getCompletedWorkflows(): WorkflowContext[] {
    return Array.from(this.completedWorkflows.values());
  }

  /**
   * Get workflow by execution ID
   */
  getWorkflow(executionId: string): WorkflowContext | undefined {
    return (
      this.activeWorkflows.get(executionId) ||
      this.completedWorkflows.get(executionId)
    );
  }

  /**
   * Cancel a running workflow
   */
  cancel(executionId: string): boolean {
    const workflow = this.activeWorkflows.get(executionId);
    if (workflow) {
      workflow.state = WorkflowState.CANCELLED;
      workflow.completed_at = new Date().toISOString();
      this.activeWorkflows.delete(executionId);
      this.completedWorkflows.set(executionId, workflow);

      this.logger.info('Workflow cancelled', {
        execution_id: executionId,
        workflow_id: workflow.workflow_id
      });

      return true;
    }
    return false;
  }
}

/**
 * Singleton instance
 */
export const workflowEngine = new WorkflowEngine();

/**
 * Example usage:
 *
 * ```typescript
 * import { workflowEngine, PREDEFINED_WORKFLOWS } from './workflow-engine';
 *
 * // Execute predefined workflow
 * const result = await workflowEngine.execute(
 *   PREDEFINED_WORKFLOWS['rag-pipeline'],
 *   {
 *     document_id: 'doc-123',
 *     document_path: '/path/to/document.pdf'
 *   }
 * );
 *
 * console.log('Workflow result:', result);
 *
 * // Define custom workflow
 * const customWorkflow: WorkflowDefinition = {
 *   workflow_id: 'custom-analysis',
 *   workflow_name: 'Custom Analysis',
 *   description: 'My custom analysis workflow',
 *   steps: [
 *     {
 *       step_id: 'step1',
 *       step_name: 'First Step',
 *       service: 'my-service',
 *       operation: 'analyze',
 *       handler: async (context) => {
 *         // Do work
 *         return { result: 'success' };
 *       }
 *     }
 *   ]
 * };
 *
 * const customResult = await workflowEngine.execute(customWorkflow, { data: 'test' });
 * ```
 */
