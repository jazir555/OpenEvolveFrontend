/**
 * Workflow Orchestrator
 *
 * Executes complex multi-step workflows across multiple plugins.
 * Supports parallel execution, conditional branching, and error handling.
 *
 * Workflow Definition:
 * {
 *   id: string;
 *   name: string;
 *   description: string;
 *   steps: WorkflowStep[];
 *   onError?: 'stop' | 'continue' | 'retry';
 *   maxRetries?: number;
 * }
 *
 * Workflow Step:
 * {
 *   id: string;
 *   name: string;
 *   plugin: string;
 *   action: string;
 *   input: Record<string, unknown>;
 *   outputMapping?: Record<string, string>;
 *   condition?: (context: WorkflowContext) => boolean;
 *   retryOnFailure?: boolean;
 *   timeout?: number;
 * }
 */
import type { PluginRegistry } from './plugin-registry';
export interface WorkflowStep {
    id: string;
    name: string;
    description?: string;
    plugin: string;
    action: string;
    input: Record<string, unknown>;
    outputMapping?: Record<string, string>;
    condition?: (context: WorkflowContext) => boolean;
    retryOnFailure?: boolean;
    timeout?: number;
    dependsOn?: string[];
}
export interface WorkflowDefinition {
    id: string;
    name: string;
    description?: string;
    version?: string;
    steps: WorkflowStep[];
    onError?: 'stop' | 'continue' | 'retry';
    maxRetries?: number;
    timeout?: number;
    metadata?: Record<string, unknown>;
}
export interface WorkflowContext {
    workflowId: string;
    executionId: string;
    input: Record<string, unknown>;
    output: Record<string, unknown>;
    variables: Record<string, unknown>;
    stepResults: Map<string, unknown>;
    startTime: number;
    currentStep?: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
}
export interface WorkflowExecutionResult {
    executionId: string;
    workflowId: string;
    status: 'completed' | 'failed' | 'partial';
    duration: number;
    results: Record<string, unknown>;
    stepResults: Map<string, unknown>;
    errors: Array<{
        stepId: string;
        error: string;
    }>;
}
interface StepExecutionResult {
    stepId: string;
    success: boolean;
    output?: unknown;
    error?: string;
    duration: number;
}
declare class WorkflowOrchestrator {
    private registry;
    private activeWorkflows;
    private correlationContext;
    private monitor;
    private eventIntegration;
    constructor(registry?: PluginRegistry);
    /**
     * Execute a workflow
     */
    executeWorkflow(workflow: WorkflowDefinition, input?: Record<string, unknown>): Promise<WorkflowExecutionResult>;
    /**
     * Execute workflow steps in dependency order
     */
    private executeSteps;
    /**
     * Execute a single step
     */
    private executeStep;
    /**
     * Resolve input variables from context
     */
    private resolveInput;
    /**
     * Map step outputs to workflow output
     */
    private mapOutputs;
    /**
     * Topological sort for dependency execution order
     */
    private topologicalSort;
    /**
     * Get active workflow executions
     */
    getActiveWorkflows(): WorkflowContext[];
    /**
     * Cancel a workflow execution
     */
    cancelWorkflow(executionId: string): Promise<boolean>;
    /**
     * Validate workflow definition
     */
    validateWorkflow(workflow: WorkflowDefinition): {
        valid: boolean;
        errors: string[];
    };
}
/**
 * Get or create the global workflow orchestrator
 */
export declare function getWorkflowOrchestrator(registry?: PluginRegistry): WorkflowOrchestrator;
export { WorkflowOrchestrator };
export type { StepExecutionResult };
//# sourceMappingURL=workflow-orchestrator.d.ts.map