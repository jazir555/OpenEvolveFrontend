// @ts-nocheck
/**
 * ROMA (Reasoning and Multi-Agent) Node
 *
 * Implements the ROMA framework for recursive hierarchical decomposition
 * and multi-agent reasoning tasks.
 *
 * ROMA Architecture:
 * - Atomizer: Decides if task needs planning
 * - Planner: Breaks tasks into subtasks
 * - Executor: Handles atomic tasks
 * - Aggregator: Integrates results
 *
 * @module nodes
 * @version 1.0.0
 */

import { OpenEvolveBaseNode } from './OpenEvolveBaseNode';
import type { NodeConfig, NodeResult, ExecutionContext } from './BaseNode';

/**
 * ROMA reasoning modes
 */
export type ROMAMode =
  | 'collaborative'      // Collaborative reasoning
  | 'adversarial'        // Adversarial reasoning
  | 'debate'             // Debate-based reasoning
  | 'consensus'          // Consensus building
  | 'hierarchical';      // Hierarchical reasoning

/**
 * ROMA agent roles
 */
export type AgentRole =
  | 'analyst'            // Analyzes the problem
  | 'critic'             // Critiques solutions
  | 'synthesizer'        // Synthesizes perspectives
  | 'validator'          // Validates results
  | 'explorer'           // Explores alternatives
  | 'integrator';        // Integrates findings

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
export class ROMANode extends OpenEvolveBaseNode {
  /**
   * Node type identifier
   */
  static readonly NODE_TYPE = 'ROMA';

  /**
   * Node display name
   */
  static readonly DISPLAY_NAME = 'ROMA Reasoning';

  /**
   * Node category
   */
  static readonly CATEGORY = 'reasoning';

  /**
   * Node icon
   */
  static readonly ICON = '🧠';

  constructor(config: ROMANodeConfig) {
    super(config);
    this.config = config;
  }

  /**
   * Get parameter schema
   */
  getParameterSchema() {
    return [
      {
        name: 'task',
        type: 'textarea',
        label: 'Task Description',
        description: 'The complex reasoning task',
        required: true,
        multiline: true,
      },
      {
        name: 'reasoningMode',
        type: 'select',
        label: 'Reasoning Mode',
        description: 'Type of reasoning to apply',
        required: true,
        defaultValue: 'collaborative',
        options: [
          { value: 'collaborative', label: 'Collaborative Reasoning' },
          { value: 'adversarial', label: 'Adversarial Reasoning' },
          { value: 'debate', label: 'Debate-Based' },
          { value: 'consensus', label: 'Consensus Building' },
          { value: 'hierarchical', label: 'Hierarchical Reasoning' },
        ],
      },
      {
        name: 'agentCount',
        type: 'number',
        label: 'Number of Agents',
        defaultValue: 3,
        min: 2,
        max: 7,
      },
      {
        name: 'agentRoles',
        type: 'multiselect',
        label: 'Agent Roles',
        description: 'Roles for reasoning agents',
        options: [
          { value: 'analyst', label: 'Analyst' },
          { value: 'critic', label: 'Critic' },
          { value: 'synthesizer', label: 'Synthesizer' },
          { value: 'validator', label: 'Validator' },
          { value: 'explorer', label: 'Explorer' },
          { value: 'integrator', label: 'Integrator' },
        ],
      },
      {
        name: 'rounds',
        type: 'number',
        label: 'Reasoning Rounds',
        description: 'Number of reasoning iterations',
        defaultValue: 3,
        min: 1,
        max: 10,
      },
      {
        name: 'confidenceThreshold',
        type: 'slider',
        label: 'Confidence Threshold',
        description: 'Minimum confidence for consensus (0.0 - 1.0)',
        defaultValue: 0.7,
        min: 0,
        max: 1,
        step: 0.1,
      },
      {
        name: 'includeReasoningTrace',
        type: 'boolean',
        label: 'Include Reasoning Trace',
        description: 'Show detailed reasoning process',
        defaultValue: true,
      },
      {
        name: 'enableVoting',
        type: 'boolean',
        label: 'Enable Agent Voting',
        defaultValue: true,
      },
    ];
  }

  /**
   * Validate inputs
   */
  protected async validate(inputs: any, context: ExecutionContext): Promise<void> {
    const config = this.config as ROMANodeConfig;

    if (!config.task || config.task.trim().length === 0) {
      throw new Error('Task description is required');
    }

    if (config.agentCount < 2 || config.agentCount > 7) {
      throw new Error('Agent count must be between 2 and 7');
    }

    if (config.rounds < 1 || config.rounds > 10) {
      throw new Error('Rounds must be between 1 and 10');
    }

    if (config.confidenceThreshold < 0 || config.confidenceThreshold > 1) {
      throw new Error('Confidence threshold must be between 0 and 1');
    }
  }

  /**
   * Execute ROMA reasoning
   */
  protected async execute(inputs: any, context: ExecutionContext): Promise<NodeResult> {
    const config = this.config as ROMANodeConfig;
    const startTime = Date.now();

    // Report progress
    this.reportProgress(0, 'Initializing ROMA reasoning...');

    try {
      // Call ROMA API
      const response = await fetch(`${context.apiUrl}/api/openevolve/roma/solve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${context.apiKey}`,
        },
        body: JSON.stringify({
          task: config.task,
          reasoning_mode: config.reasoningMode,
          agent_count: config.agentCount,
          agent_roles: config.agentRoles,
          rounds: config.rounds,
          confidence_threshold: config.confidenceThreshold,
          include_reasoning_trace: config.includeReasoningTrace,
          enable_voting: config.enableVoting,
        }),
      });

      if (!response.ok) {
        throw new Error(`ROMA API error: ${response.statusText}`);
      }

      const data = await response.json();

      // Report completion
      this.reportProgress(100, 'ROMA reasoning complete');

      const executionTime = (Date.now() - startTime) / 1000;

      return {
        success: true,
        data: {
          solution: data.solution || '',
          confidence: data.confidence || 0,
          rounds: data.rounds || config.rounds,
          reasoningTrace: data.reasoning_trace,
          agentVotes: data.agent_votes,
          subtasks: data.subtasks,
          executionTime,
          consensusReached: data.consensus_reached || false,
          qualityMetrics: data.quality_metrics || {
            coherence: 0,
            completeness: 0,
            validity: 0,
          },
        } as ROMAResult,
        metrics: {
          executionTime,
          rounds: data.rounds || config.rounds,
          consensusReached: data.consensus_reached || false,
        },
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      throw new Error(`ROMA execution failed: ${errorMessage}`);
    }
  }

  /**
   * Validate inputs
   */
  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];

    if (!inputs.task && !this.config.task) {
      errors.push({
        field: 'task',
        message: 'Task is required',
        severity: 'error'
      });
    }

    if (inputs.agentCount && (typeof inputs.agentCount !== 'number' || inputs.agentCount < 1)) {
      errors.push({
        field: 'agentCount',
        message: 'Agent count must be a positive number',
        severity: 'error'
      });
    }

    if (inputs.rounds && (typeof inputs.rounds !== 'number' || inputs.rounds < 1)) {
      errors.push({
        field: 'rounds',
        message: 'Rounds must be a positive number',
        severity: 'error'
      });
    }

    return errors;
  }

  /**
   * Get display name
   */
  getDisplayName(): string {
    return ROMANode.DISPLAY_NAME;
  }

  /**
   * Get icon
   */
  getIcon(): string {
    return ROMANode.ICON;
  }

  /**
   * Get category
   */
  getCategory(): string {
    return ROMANode.CATEGORY;
  }

  /**
   * Get version
   */
  getVersion(): string {
    return '1.0.0';
  }
}
