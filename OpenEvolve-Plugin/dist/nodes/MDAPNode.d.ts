import { OpenEvolveBaseNode, NodeInputs, NodeResult, ExecutionContext, ValidationError, ParameterSchema } from './OpenEvolveBaseNode';
/**
 * MDAP planning strategies
 */
export type MDAPStrategy = 'sequential' | 'parallel' | 'hierarchical' | 'adaptive';
/**
 * Agent domains
 */
export type AgentDomain = 'analysis' | 'design' | 'implementation' | 'testing' | 'optimization' | 'documentation';
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
export declare class MDAPNode extends OpenEvolveBaseNode {
    static readonly DISPLAY_NAME = "Multi-Domain Agent Planner";
    static readonly DESCRIPTION = "Coordinate multiple specialized agents for complex problem solving with collaboration";
    static readonly ICON = "mdap";
    static readonly CATEGORY = "planning";
    static readonly VERSION = "1.0.0";
    constructor(id: string, config?: MDAPNodeConfig);
    /**
     * Execute MDAP planning and execution
     *
     * @param inputs - Must contain 'problem' statement
     * @param context - Execution context
     * @returns Promise resolving to MDAP execution result
     */
    execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult>;
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
    private createPlan;
    /**
     * Execute plan
     *
     * @param plan - Execution plan
     * @param context - Execution context
     * @returns Promise resolving to execution result
     */
    private executePlan;
    /**
     * Monitor execution progress
     *
     * @param planId - Plan ID to monitor
     * @param context - Execution context
     * @returns Promise resolving to execution status
     */
    private monitorExecution;
    /**
     * Calculate execution metrics
     *
     * @param result - Execution result
     * @returns Calculated metrics
     */
    private calculateMetrics;
    /**
     * Validate input data
     *
     * @param inputs - Input data to validate
     * @returns Array of validation errors
     */
    validateInputs(inputs: NodeInputs): ValidationError[];
    /**
     * Get JSON Schema for configuration parameters
     *
     * @returns Parameter schema
     */
    getParameterSchema(): ParameterSchema;
    /**
     * Get available agent domains
     *
     * @returns Array of available domains
     */
    getAvailableDomains(): AgentDomain[];
    /**
     * Get available strategies
     *
     * @returns Array of available strategies
     */
    getAvailableStrategies(): MDAPStrategy[];
    /**
     * Get execution status
     *
     * @param planId - Plan ID
     * @returns Promise resolving to execution status
     */
    getExecutionStatus(planId: string): Promise<NodeResult>;
    /**
     * Cancel execution
     *
     * @param planId - Plan ID to cancel
     * @returns Promise resolving to cancellation result
     */
    cancelExecution(planId: string): Promise<NodeResult>;
    /**
     * Get execution history
     *
     * @param params - Query parameters
     * @returns Promise resolving to execution history
     */
    getExecutionHistory(params?: {
        limit?: number;
        offset?: number;
        status?: string;
    }): Promise<NodeResult>;
}
export default MDAPNode;
