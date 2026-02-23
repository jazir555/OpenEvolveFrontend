/**
 * Workflow Engine - Multi-Step Orchestration
 *
 * Follows the Federation Constitution:
 * - Law of Idempotency: Workflow state management for replay
 * - Failure Management: Circuit breakers for external services
 * - Observability: JSON Lines logging with full context
 */
import { EventBus } from './event-bus';
import { CorrelationContext } from './correlation-tracker';
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
    parallel?: boolean;
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
    started_at: string;
    completed_at?: string;
    error?: Error;
}
export declare enum WorkflowState {
    PENDING = "pending",
    RUNNING = "running",
    COMPLETED = "completed",
    FAILED = "failed",
    CANCELLED = "cancelled"
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
export declare const PREDEFINED_WORKFLOWS: Record<string, WorkflowDefinition>;
/**
 * Workflow Engine
 *
 * Executes multi-step workflows with state management, retries, and circuit breakers
 */
export declare class WorkflowEngine {
    private logger;
    private eventBus;
    private activeWorkflows;
    private completedWorkflows;
    private circuitBreakers;
    constructor(eventBus?: EventBus);
    /**
     * Execute a workflow definition
     */
    execute(workflow: WorkflowDefinition, inputData: any, correlationContext?: CorrelationContext): Promise<WorkflowExecutionResult>;
    /**
     * Execute workflow steps sequentially
     */
    private executeSequentialSteps;
    /**
     * Execute workflow steps in parallel
     */
    private executeParallelSteps;
    /**
     * Execute a single workflow step
     */
    private executeStep;
    /**
     * Execute step handler with timeout
     */
    private executeStepHandler;
    /**
     * Aggregate step results
     */
    private aggregateResults;
    /**
     * Get or create circuit breaker for service
     */
    private getCircuitBreaker;
    /**
     * Calculate workflow duration
     */
    private calculateDuration;
    /**
     * Build execution result
     */
    private buildResult;
    /**
     * Publish workflow started event
     */
    private publishWorkflowStarted;
    /**
     * Publish workflow completed event
     */
    private publishWorkflowCompleted;
    /**
     * Publish workflow failed event
     */
    private publishWorkflowFailed;
    /**
     * Get active workflows
     */
    getActiveWorkflows(): WorkflowContext[];
    /**
     * Get completed workflows
     */
    getCompletedWorkflows(): WorkflowContext[];
    /**
     * Get workflow by execution ID
     */
    getWorkflow(executionId: string): WorkflowContext | undefined;
    /**
     * Cancel a running workflow
     */
    cancel(executionId: string): boolean;
}
/**
 * Singleton instance
 */
export declare const workflowEngine: WorkflowEngine;
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
//# sourceMappingURL=workflow-engine.d.ts.map