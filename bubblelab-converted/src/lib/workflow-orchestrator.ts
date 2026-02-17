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

import { apiLogger, LogContext } from '../../../glue/lib/structuredLogger';
import { retryWithBackoff, RetryConfig } from '../../../glue/lib/retry';
import { getPluginRegistry, type PluginInterface } from './plugin-registry';
import type { PluginRegistry } from './plugin-registry';
import { getWorkflowMonitor } from './workflow-monitoring';
import { getPluginEventIntegration } from './plugin-events';

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
  dependsOn?: string[]; // Step IDs that must complete first
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
  errors: Array<{ stepId: string; error: string }>;
}

interface StepExecutionResult {
  stepId: string;
  success: boolean;
  output?: unknown;
  error?: string;
  duration: number;
}

class WorkflowOrchestrator {
  private registry: PluginRegistry;
  private activeWorkflows = new Map<string, WorkflowContext>();
  private correlationContext: LogContext;
  private monitor = getWorkflowMonitor();
  private eventIntegration = getPluginEventIntegration();

  constructor(registry?: PluginRegistry) {
    this.registry = registry || getPluginRegistry();
    this.correlationContext = {
      correlation_id: `workflow-orchestrator-${Date.now()}`,
      source_service: 'workflow-orchestrator',
      target_service: 'plugins'
    };

    apiLogger.info('Workflow Orchestrator initialized', this.correlationContext);
  }

  /**
   * Execute a workflow
   */
  async executeWorkflow(
    workflow: WorkflowDefinition,
    input: Record<string, unknown> = {}
  ): Promise<WorkflowExecutionResult> {
    const executionId = `exec-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const startTime = Date.now();

    apiLogger.info('Executing workflow', {
      ...this.correlationContext,
      workflow_id: workflow.id,
      workflow_name: workflow.name,
      execution_id: executionId
    });

    // Initialize workflow context
    const context: WorkflowContext = {
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
      const stepExecution = await this.executeSteps(workflow, context);
      const stepResults = stepExecution.results;
      const stepErrors = stepExecution.errors;

      // Map final outputs
      context.output = this.mapOutputs(workflow, context);

      context.status = 'completed';
      const duration = Date.now() - startTime;

      apiLogger.info('Workflow completed successfully', {
        ...this.correlationContext,
        workflow_id: workflow.id,
        execution_id: executionId,
        duration_ms: duration
      });

      const result = {
        executionId,
        workflowId: workflow.id,
        status: 'completed' as const,
        duration,
        results: context.output,
        stepResults,
        errors: stepErrors
      };

      // Record completion in monitor
      this.monitor.recordWorkflowCompletion(context, result);

      // Emit workflow completed event
      await this.eventIntegration.emitWorkflowCompleted(result);

      return result;
    } catch (error) {
      context.status = 'failed';
      const duration = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : String(error);
      const shouldRethrow = errorMessage.includes('Circular dependency');

      apiLogger.error('Workflow execution failed', error as Error, {
        ...this.correlationContext,
        workflow_id: workflow.id,
        execution_id: executionId,
        duration_ms: duration
      });

      const result = {
        executionId,
        workflowId: workflow.id,
        status: 'failed' as const,
        duration,
        results: context.output,
        stepResults: context.stepResults,
        errors: [{ stepId: context.currentStep || 'unknown', error: errorMessage }]
      };

      // Record completion in monitor
      this.monitor.recordWorkflowCompletion(context, result);

      // Emit workflow failed event
      await this.eventIntegration.emitWorkflowFailed(workflow.id, executionId, errorMessage);

      if (shouldRethrow) {
        throw error;
      }

      return result;
    } finally {
      this.activeWorkflows.delete(executionId);
    }
  }

  /**
   * Execute workflow steps in dependency order
   */
  private async executeSteps(
    workflow: WorkflowDefinition,
    context: WorkflowContext
  ): Promise<{
    results: Map<string, unknown>;
    errors: Array<{ stepId: string; error: string }>;
  }> {
    const results = new Map<string, unknown>();
    const errors: Array<{ stepId: string; error: string }> = [];

    // Build dependency graph and determine execution order
    const executionOrder = this.topologicalSort(workflow.steps);

    for (const step of executionOrder) {
      // Check if step should be executed
      if (step.condition && !step.condition(context)) {
        apiLogger.info('Skipping step due to condition', {
          ...this.correlationContext,
          step_id: step.id,
          step_name: step.name
        });
        continue;
      }

      // Check dependencies
      if (step.dependsOn) {
        const dependenciesMet = step.dependsOn.every(depId =>
          results.has(depId)
        );

        if (!dependenciesMet) {
          apiLogger.warn('Step dependencies not met, skipping', {
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
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        errors.push({ stepId: step.id, error: errorMessage });

        apiLogger.error('Step execution failed', error as Error, {
          ...this.correlationContext,
          step_id: step.id,
          step_name: step.name
        });

        // Handle error based on workflow configuration
        if (workflow.onError === 'stop') {
          throw error;
        } else if (workflow.onError === 'retry' && step.retryOnFailure) {
          const maxRetries = workflow.maxRetries || 3;
          const retryConfig: RetryConfig = { max_retries: maxRetries };

          const result = await retryWithBackoff(async () => {
            return await this.executeStep(step, context, workflow);
          }, retryConfig);

          results.set(step.id, result.output);
          context.stepResults.set(step.id, result);
        }
        // If onError === 'continue', just continue to next step
      }
    }

    return {
      results,
      errors,
    };
  }

  /**
   * Execute a single step
   */
  private async executeStep(
    step: WorkflowStep,
    context: WorkflowContext,
    workflow: WorkflowDefinition
  ): Promise<StepExecutionResult> {
    const startTime = Date.now();
    let retryCount = 0;

    apiLogger.info('Executing step', {
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
      const output = await this.registry.executePlugin(
        step.plugin,
        async () => {
          // Dynamically call the plugin action
          const pluginAny = plugin as any;
          if (typeof pluginAny[step.action] !== 'function') {
            throw new Error(`Plugin ${step.plugin} does not have action ${step.action}`);
          }

          return await pluginAny[step.action](resolvedInput);
        }
      );

      const endTime = Date.now();
      const duration = endTime - startTime;

      // Record step execution in monitor
      this.monitor.recordStepExecution(
        context,
        step.id,
        step.name,
        step.plugin,
        step.action,
        startTime,
        endTime,
        true,
        undefined,
        retryCount
      );

      apiLogger.info('Step completed', {
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
    } catch (error) {
      const endTime = Date.now();
      const duration = endTime - startTime;
      const errorMessage = error instanceof Error ? error.message : String(error);

      // Record failed step execution in monitor
      this.monitor.recordStepExecution(
        context,
        step.id,
        step.name,
        step.plugin,
        step.action,
        startTime,
        endTime,
        false,
        errorMessage,
        retryCount
      );

      throw error;
    }
  }

  /**
   * Resolve input variables from context
   */
  private resolveInput(
    input: Record<string, unknown>,
    context: WorkflowContext
  ): Record<string, unknown> {
    const resolved: Record<string, unknown> = {};

    for (const [key, value] of Object.entries(input)) {
      if (typeof value === 'string' && value.startsWith('$')) {
        // Reference to context variable
        const varName = value.slice(1);
        resolved[key] = context.variables[varName] ?? context.input[varName];
      } else if (typeof value === 'object' && value !== null) {
        // Recursively resolve nested objects
        resolved[key] = this.resolveInput(value as Record<string, unknown>, context);
      } else {
        resolved[key] = value;
      }
    }

    return resolved;
  }

  /**
   * Map step outputs to workflow output
   */
  private mapOutputs(
    workflow: WorkflowDefinition,
    context: WorkflowContext
  ): Record<string, unknown> {
    const output: Record<string, unknown> = {};

    for (const step of workflow.steps) {
      if (step.outputMapping) {
        const stepResult = context.stepResults.get(step.id);
        if (stepResult) {
          for (const [sourceKey, targetKey] of Object.entries(step.outputMapping)) {
            const resultAny = stepResult as any;
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
  private topologicalSort(steps: WorkflowStep[]): WorkflowStep[] {
    const sorted: WorkflowStep[] = [];
    const visited = new Set<string>();
    const visiting = new Set<string>();

    const visit = (stepId: string) => {
      if (visited.has(stepId)) return;
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
  getActiveWorkflows(): WorkflowContext[] {
    return Array.from(this.activeWorkflows.values());
  }

  /**
   * Cancel a workflow execution
   */
  async cancelWorkflow(executionId: string): Promise<boolean> {
    const context = this.activeWorkflows.get(executionId);
    if (!context) {
      return false;
    }

    context.status = 'cancelled';
    this.activeWorkflows.delete(executionId);

    apiLogger.info('Workflow cancelled', {
      ...this.correlationContext,
      execution_id: executionId
    });

    return true;
  }

  /**
   * Validate workflow definition
   */
  validateWorkflow(workflow: WorkflowDefinition): {
    valid: boolean;
    errors: string[];
  } {
    const errors: string[] = [];

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
    const stepIds = new Set<string>();
    for (const step of workflow.steps) {
      if (!step.id) {
        errors.push(`Step missing ID`);
      } else if (stepIds.has(step.id)) {
        errors.push(`Duplicate step ID: ${step.id}`);
      } else {
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

// Global singleton instance
let globalOrchestrator: WorkflowOrchestrator | null = null;

/**
 * Get or create the global workflow orchestrator
 */
export function getWorkflowOrchestrator(registry?: PluginRegistry): WorkflowOrchestrator {
  if (!globalOrchestrator) {
    globalOrchestrator = new WorkflowOrchestrator(registry);
  }
  return globalOrchestrator;
}

export { WorkflowOrchestrator };
export type { StepExecutionResult };
