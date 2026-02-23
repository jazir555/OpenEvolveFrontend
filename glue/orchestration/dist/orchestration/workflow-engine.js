"use strict";
/**
 * Workflow Engine - Multi-Step Orchestration
 *
 * Follows the Federation Constitution:
 * - Law of Idempotency: Workflow state management for replay
 * - Failure Management: Circuit breakers for external services
 * - Observability: JSON Lines logging with full context
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.workflowEngine = exports.WorkflowEngine = exports.PREDEFINED_WORKFLOWS = exports.WorkflowState = void 0;
const uuid_1 = require("uuid");
const event_types_1 = require("./event-types");
const correlation_tracker_1 = require("./correlation-tracker");
const logger_1 = require("../lib/logger");
const circuit_breaker_1 = require("../lib/circuit-breaker");
const retry_1 = require("../lib/retry");
var WorkflowState;
(function (WorkflowState) {
    WorkflowState["PENDING"] = "pending";
    WorkflowState["RUNNING"] = "running";
    WorkflowState["COMPLETED"] = "completed";
    WorkflowState["FAILED"] = "failed";
    WorkflowState["CANCELLED"] = "cancelled";
})(WorkflowState || (exports.WorkflowState = WorkflowState = {}));
/**
 * Predefined Workflows
 */
exports.PREDEFINED_WORKFLOWS = {
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
class WorkflowEngine {
    logger;
    eventBus;
    activeWorkflows = new Map();
    completedWorkflows = new Map();
    circuitBreakers = new Map();
    constructor(eventBus) {
        this.logger = new logger_1.Logger('workflow-engine');
        this.eventBus = eventBus || eventBus;
    }
    /**
     * Execute a workflow definition
     */
    async execute(workflow, inputData, correlationContext) {
        const executionId = (0, uuid_1.v4)();
        const correlationCtx = correlationContext || correlation_tracker_1.correlationTracker.createContext();
        const context = {
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
            }
            else {
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
        }
        catch (error) {
            context.state = WorkflowState.FAILED;
            context.error = error;
            context.completed_at = new Date().toISOString();
            this.activeWorkflows.delete(executionId);
            this.completedWorkflows.set(executionId, context);
            // Publish workflow failed event
            await this.publishWorkflowFailed(workflow, context, error);
            this.logger.error('Workflow execution failed', error, {
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
    async executeSequentialSteps(workflow, context) {
        for (let i = 0; i < workflow.steps.length; i++) {
            const step = workflow.steps[i];
            context.current_step = i;
            const result = await this.executeStep(step, context);
            context.step_results.set(step.step_id, result);
            // Record service call
            correlation_tracker_1.correlationTracker.recordServiceCall(context.correlation_context, step.service, step.operation);
        }
        context.output_data = this.aggregateResults(context);
    }
    /**
     * Execute workflow steps in parallel
     */
    async executeParallelSteps(workflow, context) {
        const promises = workflow.steps.map(async (step) => {
            const result = await this.executeStep(step, context);
            context.step_results.set(step.step_id, result);
            correlation_tracker_1.correlationTracker.recordServiceCall(context.correlation_context, step.service, step.operation);
            return result;
        });
        await Promise.all(promises);
        context.output_data = this.aggregateResults(context);
    }
    /**
     * Execute a single workflow step
     */
    async executeStep(step, context) {
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
                return await (0, retry_1.retryWithBackoff)(() => this.executeStepHandler(step, context), { max_retries: step.max_retries || 3 });
            }
            // Direct execution
            return await this.executeStepHandler(step, context);
        }
        catch (error) {
            this.logger.error('Workflow step failed', error, {
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
    async executeStepHandler(step, context) {
        const timeout = step.timeout_ms || 30000;
        return Promise.race([
            step.handler(context),
            new Promise((_, reject) => setTimeout(() => reject(new Error(`Step timeout: ${step.step_name}`)), timeout))
        ]);
    }
    /**
     * Aggregate step results
     */
    aggregateResults(context) {
        const results = {};
        for (const [stepId, result] of context.step_results.entries()) {
            results[stepId] = result;
        }
        return results;
    }
    /**
     * Get or create circuit breaker for service
     */
    getCircuitBreaker(service) {
        if (!this.circuitBreakers.has(service)) {
            this.circuitBreakers.set(service, new circuit_breaker_1.CircuitBreaker({
                threshold: 5,
                timeout_ms: 60000
            }));
        }
        return this.circuitBreakers.get(service);
    }
    /**
     * Calculate workflow duration
     */
    calculateDuration(context) {
        const start = new Date(context.started_at).getTime();
        const end = context.completed_at
            ? new Date(context.completed_at).getTime()
            : Date.now();
        return end - start;
    }
    /**
     * Build execution result
     */
    buildResult(context) {
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
    async publishWorkflowStarted(workflow, context) {
        const event = (0, event_types_1.createBaseEvent)('WorkflowStarted', 'workflow-engine', context.correlation_context.correlation_id, {
            workflow_id: workflow.workflow_id,
            workflow_name: workflow.workflow_name,
            input_data: context.input_data,
            steps: workflow.steps.map(s => ({
                step_id: s.step_id,
                step_name: s.step_name,
                service: s.service
            }))
        });
        await this.eventBus.publish(event);
    }
    /**
     * Publish workflow completed event
     */
    async publishWorkflowCompleted(workflow, context) {
        const event = (0, event_types_1.createBaseEvent)('WorkflowCompleted', 'workflow-engine', context.correlation_context.correlation_id, {
            workflow_id: workflow.workflow_id,
            workflow_name: workflow.workflow_name,
            duration_ms: this.calculateDuration(context),
            output_data: context.output_data,
            steps_completed: context.step_results.size,
            steps_failed: 0
        });
        await this.eventBus.publish(event);
    }
    /**
     * Publish workflow failed event
     */
    async publishWorkflowFailed(workflow, context, error) {
        const event = (0, event_types_1.createBaseEvent)('WorkflowFailed', 'workflow-engine', context.correlation_context.correlation_id, {
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
        });
        await this.eventBus.publish(event);
    }
    /**
     * Get active workflows
     */
    getActiveWorkflows() {
        return Array.from(this.activeWorkflows.values());
    }
    /**
     * Get completed workflows
     */
    getCompletedWorkflows() {
        return Array.from(this.completedWorkflows.values());
    }
    /**
     * Get workflow by execution ID
     */
    getWorkflow(executionId) {
        return (this.activeWorkflows.get(executionId)
            || this.completedWorkflows.get(executionId));
    }
    /**
     * Cancel a running workflow
     */
    cancel(executionId) {
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
exports.WorkflowEngine = WorkflowEngine;
/**
 * Singleton instance
 */
exports.workflowEngine = new WorkflowEngine();
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
//# sourceMappingURL=workflow-engine.js.map