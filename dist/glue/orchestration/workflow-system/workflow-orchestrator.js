"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.WorkflowOrchestrator = void 0;
exports.getWorkflowOrchestrator = getWorkflowOrchestrator;
const structured_logger_1 = require("../../../glue/lib/structured-logger");
const retry_1 = require("../../../glue/lib/retry");
const plugin_registry_1 = require("./plugin-registry");
const workflow_monitoring_1 = require("./workflow-monitoring");
const plugin_events_1 = require("./plugin-events");
class WorkflowOrchestrator {
    constructor(registry) {
        this.activeWorkflows = new Map();
        this.monitor = (0, workflow_monitoring_1.getWorkflowMonitor)();
        this.eventIntegration = (0, plugin_events_1.getPluginEventIntegration)();
        this.registry = registry || (0, plugin_registry_1.getPluginRegistry)();
        this.correlationContext = {
            correlation_id: `workflow-orchestrator-${Date.now()}`,
            source_service: 'workflow-orchestrator',
            target_service: 'plugins'
        };
        structured_logger_1.apiLogger.info('Workflow Orchestrator initialized', this.correlationContext);
    }
    /**
     * Execute a workflow
     */
    async executeWorkflow(workflow, input = {}) {
        const executionId = `exec-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        const startTime = Date.now();
        structured_logger_1.apiLogger.info('Executing workflow', {
            ...this.correlationContext,
            workflow_id: workflow.id,
            workflow_name: workflow.name,
            execution_id: executionId
        });
        // Initialize workflow context
        const context = {
            workflowId: workflow.id,
            executionId,
            input,
            output: {},
            variables: {},
            stepResults: new Map(),
            startTime,
            status: 'running'
        };
        this.activeWorkflows.set(executionId, context);
        // Record workflow start in monitor
        this.monitor.recordWorkflowStart(context);
        // Emit workflow started event
        await this.eventIntegration.emitWorkflowStarted(workflow.id, executionId, input);
        try {
            // Execute steps in dependency order
            const stepResults = await this.executeSteps(workflow, context);
            // Map final outputs
            context.output = this.mapOutputs(workflow, context);
            context.status = 'completed';
            const duration = Date.now() - startTime;
            structured_logger_1.apiLogger.info('Workflow completed successfully', {
                ...this.correlationContext,
                workflow_id: workflow.id,
                execution_id: executionId,
                duration_ms: duration
            });
            const result = {
                executionId,
                workflowId: workflow.id,
                status: 'completed',
                duration,
                results: context.output,
                stepResults,
                errors: []
            };
            // Record completion in monitor
            this.monitor.recordWorkflowCompletion(context, result);
            // Emit workflow completed event
            await this.eventIntegration.emitWorkflowCompleted(result);
            return result;
        }
        catch (error) {
            context.status = 'failed';
            const duration = Date.now() - startTime;
            const errorMessage = error instanceof Error ? error.message : String(error);
            structured_logger_1.apiLogger.error('Workflow execution failed', error, {
                ...this.correlationContext,
                workflow_id: workflow.id,
                execution_id: executionId,
                duration_ms: duration
            });
            const result = {
                executionId,
                workflowId: workflow.id,
                status: 'failed',
                duration,
                results: context.output,
                stepResults: context.stepResults,
                errors: [{ stepId: context.currentStep || 'unknown', error: errorMessage }]
            };
            // Record completion in monitor
            this.monitor.recordWorkflowCompletion(context, result);
            // Emit workflow failed event
            await this.eventIntegration.emitWorkflowFailed(workflow.id, executionId, errorMessage);
            return result;
        }
        finally {
            this.activeWorkflows.delete(executionId);
        }
    }
    /**
     * Execute workflow steps in dependency order
     */
    async executeSteps(workflow, context) {
        const results = new Map();
        const errors = [];
        // Build dependency graph and determine execution order
        const executionOrder = this.topologicalSort(workflow.steps);
        for (const step of executionOrder) {
            // Check if step should be executed
            if (step.condition && !step.condition(context)) {
                structured_logger_1.apiLogger.info('Skipping step due to condition', {
                    ...this.correlationContext,
                    step_id: step.id,
                    step_name: step.name
                });
                continue;
            }
            // Check dependencies
            if (step.dependsOn) {
                const dependenciesMet = step.dependsOn.every(depId => results.has(depId));
                if (!dependenciesMet) {
                    structured_logger_1.apiLogger.warn('Step dependencies not met, skipping', {
                        ...this.correlationContext,
                        step_id: step.id,
                        dependencies: step.dependsOn
                    });
                    continue;
                }
            }
            context.currentStep = step.id;
            try {
                const result = await this.executeStep(step, context, workflow);
                results.set(step.id, result.output);
                context.stepResults.set(step.id, result);
            }
            catch (error) {
                const errorMessage = error instanceof Error ? error.message : String(error);
                errors.push({ stepId: step.id, error: errorMessage });
                structured_logger_1.apiLogger.error('Step execution failed', error, {
                    ...this.correlationContext,
                    step_id: step.id,
                    step_name: step.name
                });
                // Handle error based on workflow configuration
                if (workflow.onError === 'stop') {
                    throw error;
                }
                else if (workflow.onError === 'retry' && step.retryOnFailure) {
                    const maxRetries = workflow.maxRetries || 3;
                    const retryConfig = { max_retries: maxRetries };
                    const result = await (0, retry_1.retryWithBackoff)(async () => {
                        return await this.executeStep(step, context, workflow);
                    }, retryConfig);
                    results.set(step.id, result.output);
                    context.stepResults.set(step.id, result);
                }
                // If onError === 'continue', just continue to next step
            }
        }
        return results;
    }
    /**
     * Execute a single step
     */
    async executeStep(step, context, workflow) {
        const startTime = Date.now();
        let retryCount = 0;
        structured_logger_1.apiLogger.info('Executing step', {
            ...this.correlationContext,
            step_id: step.id,
            step_name: step.name,
            plugin: step.plugin,
            action: step.action
        });
        // Get plugin
        const plugin = this.registry.getPlugin(step.plugin);
        if (!plugin) {
            throw new Error(`Plugin ${step.plugin} not found`);
        }
        // Resolve input variables
        const resolvedInput = this.resolveInput(step.input, context);
        try {
            // Execute through registry for circuit breaker protection
            const output = await this.registry.executePlugin(step.plugin, async () => {
                // Dynamically call the plugin action
                const pluginAny = plugin;
                if (typeof pluginAny[step.action] !== 'function') {
                    throw new Error(`Plugin ${step.plugin} does not have action ${step.action}`);
                }
                return await pluginAny[step.action](resolvedInput);
            });
            const endTime = Date.now();
            const duration = endTime - startTime;
            // Record step execution in monitor
            this.monitor.recordStepExecution(context, step.id, step.name, step.plugin, step.action, startTime, endTime, true, undefined, retryCount);
            structured_logger_1.apiLogger.info('Step completed', {
                ...this.correlationContext,
                step_id: step.id,
                duration_ms: duration
            });
            return {
                stepId: step.id,
                success: true,
                output,
                duration
            };
        }
        catch (error) {
            const endTime = Date.now();
            const duration = endTime - startTime;
            const errorMessage = error instanceof Error ? error.message : String(error);
            // Record failed step execution in monitor
            this.monitor.recordStepExecution(context, step.id, step.name, step.plugin, step.action, startTime, endTime, false, errorMessage, retryCount);
            throw error;
        }
    }
    /**
     * Resolve input variables from context
     */
    resolveInput(input, context) {
        const resolved = {};
        for (const [key, value] of Object.entries(input)) {
            if (typeof value === 'string' && value.startsWith('$')) {
                // Reference to context variable
                const varName = value.slice(1);
                resolved[key] = context.variables[varName] ?? context.input[varName];
            }
            else if (typeof value === 'object' && value !== null) {
                // Recursively resolve nested objects
                resolved[key] = this.resolveInput(value, context);
            }
            else {
                resolved[key] = value;
            }
        }
        return resolved;
    }
    /**
     * Map step outputs to workflow output
     */
    mapOutputs(workflow, context) {
        const output = {};
        for (const step of workflow.steps) {
            if (step.outputMapping) {
                const stepResult = context.stepResults.get(step.id);
                if (stepResult) {
                    for (const [sourceKey, targetKey] of Object.entries(step.outputMapping)) {
                        const resultAny = stepResult;
                        if (resultAny.output && sourceKey in resultAny.output) {
                            output[targetKey] = resultAny.output[sourceKey];
                        }
                    }
                }
            }
        }
        return output;
    }
    /**
     * Topological sort for dependency execution order
     */
    topologicalSort(steps) {
        const sorted = [];
        const visited = new Set();
        const visiting = new Set();
        const visit = (stepId) => {
            if (visited.has(stepId))
                return;
            if (visiting.has(stepId)) {
                throw new Error(`Circular dependency detected involving step ${stepId}`);
            }
            visiting.add(stepId);
            const step = steps.find(s => s.id === stepId);
            if (step?.dependsOn) {
                for (const depId of step.dependsOn) {
                    visit(depId);
                }
            }
            visiting.delete(stepId);
            visited.add(stepId);
            const stepToAdd = steps.find(s => s.id === stepId);
            if (stepToAdd) {
                sorted.push(stepToAdd);
            }
        };
        for (const step of steps) {
            visit(step.id);
        }
        return sorted;
    }
    /**
     * Get active workflow executions
     */
    getActiveWorkflows() {
        return Array.from(this.activeWorkflows.values());
    }
    /**
     * Cancel a workflow execution
     */
    async cancelWorkflow(executionId) {
        const context = this.activeWorkflows.get(executionId);
        if (!context) {
            return false;
        }
        context.status = 'cancelled';
        this.activeWorkflows.delete(executionId);
        structured_logger_1.apiLogger.info('Workflow cancelled', {
            ...this.correlationContext,
            execution_id: executionId
        });
        return true;
    }
    /**
     * Validate workflow definition
     */
    validateWorkflow(workflow) {
        const errors = [];
        // Check required fields
        if (!workflow.id) {
            errors.push('Workflow ID is required');
        }
        if (!workflow.name) {
            errors.push('Workflow name is required');
        }
        if (!workflow.steps || workflow.steps.length === 0) {
            errors.push('Workflow must have at least one step');
        }
        // Check step definitions
        const stepIds = new Set();
        for (const step of workflow.steps) {
            if (!step.id) {
                errors.push(`Step missing ID`);
            }
            else if (stepIds.has(step.id)) {
                errors.push(`Duplicate step ID: ${step.id}`);
            }
            else {
                stepIds.add(step.id);
            }
            if (!step.plugin) {
                errors.push(`Step ${step.id}: missing plugin`);
            }
            if (!step.action) {
                errors.push(`Step ${step.id}: missing action`);
            }
            // Check dependencies exist
            if (step.dependsOn) {
                for (const depId of step.dependsOn) {
                    if (!stepIds.has(depId) && depId !== step.id) {
                        // Dependency will be checked after all steps are added
                    }
                }
            }
        }
        // Check dependencies refer to valid steps
        for (const step of workflow.steps) {
            if (step.dependsOn) {
                for (const depId of step.dependsOn) {
                    if (!stepIds.has(depId)) {
                        errors.push(`Step ${step.id}: dependency ${depId} does not exist`);
                    }
                }
            }
        }
        return {
            valid: errors.length === 0,
            errors
        };
    }
}
exports.WorkflowOrchestrator = WorkflowOrchestrator;
// Global singleton instance
let globalOrchestrator = null;
/**
 * Get or create the global workflow orchestrator
 */
function getWorkflowOrchestrator(registry) {
    if (!globalOrchestrator) {
        globalOrchestrator = new WorkflowOrchestrator(registry);
    }
    return globalOrchestrator;
}
//# sourceMappingURL=workflow-orchestrator.js.map