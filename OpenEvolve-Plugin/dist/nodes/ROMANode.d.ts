import { OpenEvolveBaseNode } from './OpenEvolveBaseNode';
import { NodeConfig, NodeResult, ExecutionContext } from './BaseNode';
/**
 * ROMA reasoning modes
 */
export type ROMAMode = 'collaborative' | 'adversarial' | 'debate' | 'consensus' | 'hierarchical';
/**
 * ROMA agent roles
 */
export type AgentRole = 'analyst' | 'critic' | 'synthesizer' | 'validator' | 'explorer' | 'integrator';
/**
 * ROMA reasoning result
 */
export interface ROMAResult {
    /** Solution or answer */
    solution: string;
    /** Confidence score (0-1) */
    confidence: number;
    /** Number of reasoning rounds */
    rounds: number;
    /** Reasoning trace if enabled */
    reasoningTrace?: ReasoningStep[];
    /** Agent votes if enabled */
    agentVotes?: AgentVote[];
    /** Subtasks if decomposition occurred */
    subtasks?: ROMASubtask[];
    /** Execution time in seconds */
    executionTime: number;
    /** Final consensus status */
    consensusReached: boolean;
    /** Quality metrics */
    qualityMetrics: {
        coherence: number;
        completeness: number;
        validity: number;
    };
}
/**
 * Reasoning step in trace
 */
export interface ReasoningStep {
    /** Step number */
    step: number;
    /** Agent role */
    agentRole: AgentRole;
    /** Reasoning content */
    reasoning: string;
    /** Confidence in this step */
    confidence: number;
    /** Timestamp */
    timestamp: Date;
}
/**
 * Agent vote
 */
export interface AgentVote {
    /** Agent role */
    agentRole: AgentRole;
    /** Vote/position */
    vote: string;
    /** Reasoning */
    reasoning: string;
    /** Confidence */
    confidence: number;
}
/**
 * ROMA subtask
 */
export interface ROMASubtask {
    /** Subtask ID */
    id: string;
    /** Subtask description */
    task: string;
    /** Assigned agent */
    agentRole: AgentRole;
    /** Result */
    result?: string;
    /** Status */
    status: 'pending' | 'in_progress' | 'complete' | 'failed';
}
/**
 * ROMA node configuration
 */
export interface ROMANodeConfig extends NodeConfig {
    /** Complex reasoning task */
    task: string;
    /** Reasoning mode */
    reasoningMode: ROMAMode;
    /** Number of agents */
    agentCount: number;
    /** Agent roles */
    agentRoles: AgentRole[];
    /** Number of reasoning rounds */
    rounds: number;
    /** Confidence threshold for consensus */
    confidenceThreshold: number;
    /** Include detailed reasoning trace */
    includeReasoningTrace: boolean;
    /** Enable agent voting */
    enableVoting: boolean;
}
/**
 * ROMA Node class
 */
export declare class ROMANode extends OpenEvolveBaseNode {
    /**
     * Node type identifier
     */
    static readonly NODE_TYPE = "ROMA";
    /**
     * Node display name
     */
    static readonly DISPLAY_NAME = "ROMA Reasoning";
    /**
     * Node category
     */
    static readonly CATEGORY = "reasoning";
    /**
     * Node icon
     */
    static readonly ICON = "\uD83E\uDDE0";
    constructor(config: ROMANodeConfig);
    /**
     * Get parameter schema
     */
    getParameterSchema(): ({
        name: string;
        type: string;
        label: string;
        description: string;
        required: boolean;
        multiline: boolean;
        defaultValue?: undefined;
        options?: undefined;
        min?: undefined;
        max?: undefined;
        step?: undefined;
    } | {
        name: string;
        type: string;
        label: string;
        description: string;
        required: boolean;
        defaultValue: string;
        options: {
            value: string;
            label: string;
        }[];
        multiline?: undefined;
        min?: undefined;
        max?: undefined;
        step?: undefined;
    } | {
        name: string;
        type: string;
        label: string;
        defaultValue: number;
        min: number;
        max: number;
        description?: undefined;
        required?: undefined;
        multiline?: undefined;
        options?: undefined;
        step?: undefined;
    } | {
        name: string;
        type: string;
        label: string;
        description: string;
        options: {
            value: string;
            label: string;
        }[];
        required?: undefined;
        multiline?: undefined;
        defaultValue?: undefined;
        min?: undefined;
        max?: undefined;
        step?: undefined;
    } | {
        name: string;
        type: string;
        label: string;
        description: string;
        defaultValue: number;
        min: number;
        max: number;
        required?: undefined;
        multiline?: undefined;
        options?: undefined;
        step?: undefined;
    } | {
        name: string;
        type: string;
        label: string;
        description: string;
        defaultValue: number;
        min: number;
        max: number;
        step: number;
        required?: undefined;
        multiline?: undefined;
        options?: undefined;
    } | {
        name: string;
        type: string;
        label: string;
        description: string;
        defaultValue: boolean;
        required?: undefined;
        multiline?: undefined;
        options?: undefined;
        min?: undefined;
        max?: undefined;
        step?: undefined;
    } | {
        name: string;
        type: string;
        label: string;
        defaultValue: boolean;
        description?: undefined;
        required?: undefined;
        multiline?: undefined;
        options?: undefined;
        min?: undefined;
        max?: undefined;
        step?: undefined;
    })[];
    /**
     * Validate inputs
     */
    protected validate(inputs: any, context: ExecutionContext): Promise<void>;
    /**
     * Execute ROMA reasoning
     */
    protected execute(inputs: any, context: ExecutionContext): Promise<NodeResult>;
    /**
     * Validate inputs
     */
    validateInputs(inputs: NodeInputs): ValidationError[];
    /**
     * Get display name
     */
    getDisplayName(): string;
    /**
     * Get icon
     */
    getIcon(): string;
    /**
     * Get category
     */
    getCategory(): string;
    /**
     * Get version
     */
    getVersion(): string;
}
