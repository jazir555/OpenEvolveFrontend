// @ts-nocheck
/**
 * MDAP Node
 *
 * Multi-Domain Agent Planner node for complex problem solving.
 * Coordinates multiple specialized agents to solve domain-specific problems.
 *
 * @module nodes
 */

import {
  OpenEvolveBaseNode,
  NodeInputs,
  NodeResult,
  ExecutionContext,
  ValidationError,
  ParameterSchema
} from './OpenEvolveBaseNode';
import { apiClient } from '@/services/api';

/**
 * MDAP planning strategies
 */
export type MDAPStrategy = 'sequential' | 'parallel' | 'hierarchical' | 'adaptive';

/**
 * Agent domains
 */
export type AgentDomain =
  | 'analysis'
  | 'design'
  | 'implementation'
  | 'testing'
  | 'optimization'
  | 'documentation';

/**
 * MDAP node configuration
 */
export interface MDAPNodeConfig {
  strategy?: MDAPStrategy;
  domains?: AgentDomain[];
  enableAgentCollaboration?: boolean;
  enableKnowledgeSharing?: boolean;
  maxIterations?: number;
}

/**
 * Agent task
 */
export interface AgentTask {
  taskId: string;
  domain: AgentDomain;
  description: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  result?: any;
  dependencies: string[];
  assignedAgent: string;
  startTime?: Date;
  endTime?: Date;
}

/**
 * Agent collaboration
 */
export interface AgentCollaboration {
  fromAgent: string;
  toAgent: string;
  domain: AgentDomain;
  message: string;
  sharedKnowledge: any;
  timestamp: Date;
}

/**
 * MDAP plan
 */
export interface MDAPPlan {
  planId: string;
  problem: string;
  strategy: MDAPStrategy;
  domains: AgentDomain[];
  tasks: AgentTask[];
  collaborations: AgentCollaboration[];
  executionOrder: string[][];
  estimatedDuration: number;
}

/**
 * MDAP execution result
 */
export interface MDAPExecutionResult {
  planId: string;
  problem: string;
  strategy: MDAPStrategy;
  status: 'in_progress' | 'completed' | 'failed';
  tasks: AgentTask[];
  collaborations: AgentCollaboration[];
  finalResult?: any;
  metrics: {
    totalTasks: number;
    completedTasks: number;
    failedTasks: number;
    avgTaskDuration: number;
    totalExecutionTime: number;
    collaborationCount: number;
    knowledgeShared: number;
  };
  metadata: {
    startedAt: Date;
    completedAt?: Date;
    executionTime: number;
    parameters: {
      strategy: MDAPStrategy;
      domains: AgentDomain[];
      enableAgentCollaboration: boolean;
      enableKnowledgeSharing: boolean;
    };
  };
}

/**
 * MDAP Node
 *
 * Plans and executes complex multi-domain problem solving.
 * Coordinates specialized agents with collaboration and knowledge sharing.
 */
export class MDAPNode extends OpenEvolveBaseNode {
  static readonly DISPLAY_NAME = 'Multi-Domain Agent Planner';
  static readonly DESCRIPTION = 'Coordinate multiple specialized agents for complex problem solving with collaboration';
  static readonly ICON = 'mdap';
  static readonly CATEGORY = 'planning';
  static readonly VERSION = '1.0.0';

  constructor(id: string, config: MDAPNodeConfig = {}) {
    super(id, {
      strategy: 'adaptive',
      domains: ['analysis', 'design', 'implementation', 'testing'],
      enableAgentCollaboration: true,
      enableKnowledgeSharing: true,
      maxIterations: 5,
      ...config
    });
  }

  /**
   * Execute MDAP planning and execution
   *
   * @param inputs - Must contain 'problem' statement
   * @param context - Execution context
   * @returns Promise resolving to MDAP execution result
   */
  async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    try {
      const startTime = Date.now();

      // Extract inputs
      const problem = inputs.problem as string;
      const strategy = (inputs.strategy as MDAPStrategy) || (this.config.strategy as MDAPStrategy);
      const domains = (inputs.domains as AgentDomain[]) || this.config.domains as AgentDomain[];
      const constraints = inputs.constraints as Record<string, any> | undefined;
      const requirements = inputs.requirements as string[] | undefined;

      // Validate required inputs
      if (!problem || problem.trim().length === 0) {
        return this.createErrorResult('Problem statement is required and cannot be empty');
      }

      context.updateProgress(10, 'Analyzing problem and creating execution plan');

      // Step 1: Create execution plan
      const plan = await this.createPlan(problem, strategy, domains, requirements, constraints, context);

      context.updateProgress(30, 'Plan created, executing tasks');

      // Step 2: Execute plan
      const result = await this.executePlan(plan, context);

      const executionTime = Date.now() - startTime;

      // Step 3: Calculate metrics
      const metrics = this.calculateMetrics(result);

      const executionResult: MDAPExecutionResult = {
        planId: plan.planId,
        problem,
        strategy,
        status: result.status,
        tasks: result.tasks,
        collaborations: result.collaborations,
        finalResult: result.finalResult,
        metrics,
        metadata: {
          startedAt: new Date(startTime),
          completedAt: result.status === 'completed' ? new Date() : undefined,
          executionTime,
          parameters: {
            strategy,
            domains,
            enableAgentCollaboration: this.config.enableAgentCollaboration as boolean,
            enableKnowledgeSharing: this.config.enableKnowledgeSharing as boolean
          }
        }
      };

      context.updateProgress(100, `MDAP execution complete: ${metrics.completedTasks}/${metrics.totalTasks} tasks completed`);

      return this.createSuccessResult(executionResult);

    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Unknown error during MDAP execution'
      );
    }
  }

  /**
   * Create execution plan
   *
   * @param problem - Problem statement
   * @param strategy - Planning strategy
   * @param domains - Agent domains to involve
   * @param requirements - Optional requirements
   * @param constraints - Optional constraints
   * @param context - Execution context
   * @returns Promise resolving to execution plan
   */
  private async createPlan(
    problem: string,
    strategy: MDAPStrategy,
    domains: AgentDomain[],
    requirements?: string[],
    constraints?: Record<string, any>,
    context?: ExecutionContext
  ): Promise<MDAPPlan> {
    const response = await apiClient.post<any>('/mdap/plan', {
      problem,
      strategy,
      domains,
      requirements,
      constraints
    });

    // Transform response to plan format
    const tasks: AgentTask[] = (response.tasks || []).map((t: any) => ({
      taskId: t.task_id || `task-${Date.now()}`,
      domain: t.domain,
      description: t.description || '',
      status: 'pending',
      dependencies: t.dependencies || [],
      assignedAgent: t.assigned_agent || 'auto'
    }));

    const collaborations: AgentCollaboration[] = (response.collaborations || []).map((c: any) => ({
      fromAgent: c.from_agent,
      toAgent: c.to_agent,
      domain: c.domain,
      message: c.message || '',
      sharedKnowledge: c.knowledge,
      timestamp: new Date(c.timestamp || Date.now())
    }));

    return {
      planId: response.plan_id || `plan-${Date.now()}`,
      problem,
      strategy,
      domains,
      tasks,
      collaborations,
      executionOrder: response.execution_order || [],
      estimatedDuration: response.estimated_duration || 0
    };
  }

  /**
   * Execute plan
   *
   * @param plan - Execution plan
   * @param context - Execution context
   * @returns Promise resolving to execution result
   */
  private async executePlan(
    plan: MDAPPlan,
    context: ExecutionContext
  ): Promise<any> {
    const response = await apiClient.post<any>('/mdap/execute', {
      plan_id: plan.planId,
      enable_collaboration: this.config.enableAgentCollaboration,
      enable_knowledge_sharing: this.config.enableKnowledgeSharing,
      max_iterations: this.config.maxIterations
    });

    // Monitor execution
    return await this.monitorExecution(plan.planId, context);
  }

  /**
   * Monitor execution progress
   *
   * @param planId - Plan ID to monitor
   * @param context - Execution context
   * @returns Promise resolving to execution status
   */
  private async monitorExecution(planId: string, context: ExecutionContext): Promise<any> {
    const maxAttempts = 60; // 5 minutes with 5 second intervals
    let attempts = 0;
    const startTime = Date.now();
    const timeoutMs = 300000; // 5 minutes

    while (attempts < maxAttempts) {
      // Check timeout
      if (Date.now() - startTime > timeoutMs) {
        throw new Error('MDAP execution monitoring timeout exceeded');
      }

      try {
        const status = await apiClient.get<any>(`/mdap/execution/${planId}`);

        // Update progress
        const progress = status.current_task && status.total_tasks
          ? (status.current_task / status.total_tasks) * 70 + 30
          : 30 + (attempts / maxAttempts) * 70;

        context.updateProgress(
          Math.min(progress, 95),
          `Task ${status.current_task || 0}/${status.total_tasks || '?'} - ${status.current_domain || 'executing'}`
        );

        // Check if execution is complete
        if (status.status === 'completed' || status.status === 'failed') {
          return status;
        }

        // Wait before next poll
        await new Promise(resolve => setTimeout(resolve, 5000));
        attempts++;

      } catch (error) {
        // If polling fails, wait and retry
        await new Promise(resolve => setTimeout(resolve, 5000));
        attempts++;
      }
    }

    throw new Error('MDAP execution did not complete within the expected time');
  }

  /**
   * Calculate execution metrics
   *
   * @param result - Execution result
   * @returns Calculated metrics
   */
  private calculateMetrics(result: any): MDAPExecutionResult['metrics'] {
    const tasks = result.tasks || [];
    const completedTasks = tasks.filter((t: AgentTask) => t.status === 'completed');
    const failedTasks = tasks.filter((t: AgentTask) => t.status === 'failed');

    // Calculate average task duration
    const taskDurations = completedTasks
      .filter((t: AgentTask) => t.startTime && t.endTime)
      .map((t: AgentTask) => {
        const start = new Date(t.startTime!).getTime();
        const end = new Date(t.endTime!).getTime();
        return end - start;
      });

    const avgTaskDuration = taskDurations.length > 0
      ? taskDurations.reduce((sum, d) => sum + d, 0) / taskDurations.length
      : 0;

    const totalExecutionTime = result.completed_at
      ? new Date(result.completed_at).getTime() - new Date(result.started_at).getTime()
      : 0;

    return {
      totalTasks: tasks.length,
      completedTasks: completedTasks.length,
      failedTasks: failedTasks.length,
      avgTaskDuration,
      totalExecutionTime,
      collaborationCount: result.collaborations?.length || 0,
      knowledgeShared: result.knowledge_shared || 0
    };
  }

  /**
   * Validate input data
   *
   * @param inputs - Input data to validate
   * @returns Array of validation errors
   */
  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];

    if (!inputs.problem) {
      errors.push({
        field: 'problem',
        message: 'Problem statement is required',
        severity: 'error'
      });
    }

    if (inputs.problem && typeof inputs.problem !== 'string') {
      errors.push({
        field: 'problem',
        message: 'Problem must be a string',
        severity: 'error'
      });
    }

    if (inputs.problem && inputs.problem.length < 20) {
      errors.push({
        field: 'problem',
        message: 'Problem statement is too short for meaningful planning (minimum 20 characters)',
        severity: 'warning'
      });
    }

    // Validate strategy
    if (inputs.strategy && typeof inputs.strategy === 'string') {
      const validStrategies = ['sequential', 'parallel', 'hierarchical', 'adaptive'];
      if (!validStrategies.includes(inputs.strategy)) {
        errors.push({
          field: 'strategy',
          message: `Strategy must be one of: ${validStrategies.join(', ')}`,
          severity: 'error'
        });
      }
    }

    // Validate domains
    if (inputs.domains && Array.isArray(inputs.domains)) {
      const validDomains = ['analysis', 'design', 'implementation', 'testing', 'optimization', 'documentation'];
      const invalidDomains = inputs.domains.filter(d => !validDomains.includes(d));
      if (invalidDomains.length > 0) {
        errors.push({
          field: 'domains',
          message: `Invalid domains: ${invalidDomains.join(', ')}`,
          severity: 'error'
        });
      }
    }

    return errors;
  }

  /**
   * Get JSON Schema for configuration parameters
   *
   * @returns Parameter schema
   */
  getParameterSchema(): ParameterSchema {
    return {
      type: 'object',
      properties: {
        strategy: {
          type: 'string',
          description: 'Planning and execution strategy',
          enum: ['sequential', 'parallel', 'hierarchical', 'adaptive'],
          default: 'adaptive'
        },
        domains: {
          type: 'array',
          description: 'Agent domains to involve in problem solving',
          items: {
            type: 'string',
            enum: ['analysis', 'design', 'implementation', 'testing', 'optimization', 'documentation']
          },
          default: ['analysis', 'design', 'implementation', 'testing']
        },
        enableAgentCollaboration: {
          type: 'boolean',
          description: 'Enable collaboration between agents',
          default: true
        },
        enableKnowledgeSharing: {
          type: 'boolean',
          description: 'Enable knowledge sharing between agents',
          default: true
        },
        maxIterations: {
          type: 'number',
          description: 'Maximum iterations for adaptive planning',
          minimum: 1,
          maximum: 20,
          default: 5
        }
      },
      required: []
    };
  }

  /**
   * Get available agent domains
   *
   * @returns Array of available domains
   */
  getAvailableDomains(): AgentDomain[] {
    return ['analysis', 'design', 'implementation', 'testing', 'optimization', 'documentation'];
  }

  /**
   * Get available strategies
   *
   * @returns Array of available strategies
   */
  getAvailableStrategies(): MDAPStrategy[] {
    return ['sequential', 'parallel', 'hierarchical', 'adaptive'];
  }

  /**
   * Get execution status
   *
   * @param planId - Plan ID
   * @returns Promise resolving to execution status
   */
  async getExecutionStatus(planId: string): Promise<NodeResult> {
    try {
      const status = await apiClient.get<any>(`/mdap/execution/${planId}`);
      return this.createSuccessResult(status);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to get execution status'
      );
    }
  }

  /**
   * Cancel execution
   *
   * @param planId - Plan ID to cancel
   * @returns Promise resolving to cancellation result
   */
  async cancelExecution(planId: string): Promise<NodeResult> {
    try {
      const response = await apiClient.post<any>(`/mdap/execution/${planId}/cancel`, {});
      return this.createSuccessResult(response);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to cancel execution'
      );
    }
  }

  /**
   * Get execution history
   *
   * @param params - Query parameters
   * @returns Promise resolving to execution history
   */
  async getExecutionHistory(params?: {
    limit?: number;
    offset?: number;
    status?: string;
  }): Promise<NodeResult> {
    try {
      const response = await apiClient.get<any>('/mdap/history', params);
      return this.createSuccessResult(response);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to get execution history'
      );
    }
  }
}

export default MDAPNode;
