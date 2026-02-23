/**
 * ROMA Workflow Templates
 *
 * Dedicated workflow templates for ROMA integration in the orchestration layer.
 * These templates leverage ROMA's unique capabilities for hierarchical problem solving.
 */

import { z } from 'zod';

// ============================================================================
// ROMA WORKFLOW SCHEMAS
// ============================================================================

/**
 * ROMA Decomposition Workflow Schema
 */
export const RomaDecompositionWorkflowSchema = z.object({
  workflow_type: z.literal('roma_decomposition'),
  goal: z.string().min(1),
  max_depth: z.number().int().min(1).max(10).default(3),
  execution_method: z.enum(['traditional', 'claudiomiro', 'datapizza', 'roma', 'hybrid']).default('roma'),
  enable_verification: z.boolean().default(true),
  enable_checkpointing: z.boolean().default(true),
  timeout_ms: z.number().int().positive().default(300000),
});

/**
 * ROMA MDAP/MAKER Workflow Schema
 */
export const RomaMdapMakerWorkflowSchema = z.object({
  workflow_type: z.literal('roma_mdap_maker'),
  goal: z.string().min(1),
  max_depth: z.number().int().min(1).max(5).default(2),
  k_ahead: z.number().int().min(1).max(10).default(3),
  enable_red_flagging: z.boolean().default(true),
  enable_adaptive_k: z.boolean().default(true),
  zero_error_mode: z.boolean().default(true),
  provider: z.string().default('openai'),
  model: z.string().default('gpt-4o-mini'),
  auto_selection_keywords: z.array(z.string()).default([
    'critical', 'zero error', 'flawless', 'perfect', 'exact',
    'precision', 'accuracy', 'formal', 'verification', 'proven'
  ]),
  timeout_ms: z.number().int().positive().default(300000),
});

/**
 * ROMA Multi-Agent Workflow Schema
 */
export const RomaMultiAgentWorkflowSchema = z.object({
  workflow_type: z.literal('roma_multi_agent'),
  goal: z.string().min(1),
  prediction_strategies: z.array(z.enum([
    'predict', 'chain_of_thought', 'react', 'code_act',
    'best_of_n', 'refine', 'parallel', 'majority'
  ])).default(['chain_of_thought']),
  num_agents: z.number().int().min(2).max(7).default(3),
  reasoning_rounds: z.number().int().min(1).max(10).default(3),
  confidence_threshold: z.number().min(0).max(1).default(0.7),
  enable_voting: z.boolean().default(true),
  timeout_ms: z.number().int().positive().default(300000),
});

/**
 * ROMA Hybrid Workflow Schema
 */
export const RomaHybridWorkflowSchema = z.object({
  workflow_type: z.literal('roma_hybrid'),
  goal: z.string().min(1),
  execution_phases: z.array(z.enum([
    'atomize', 'plan', 'execute', 'aggregate', 'verify'
  ])).default(['atomize', 'plan', 'execute', 'aggregate', 'verify']),
  enable_openevolve_integration: z.boolean().default(true),
  enable_team_qa: z.boolean().default(true),
  max_depth: z.number().int().min(1).max(5).default(2),
  timeout_ms: z.number().int().positive().default(300000),
});

// ============================================================================
// WORKFLOW TEMPLATES
// ============================================================================

/**
 * ROMA Decomposition Workflow
 *
 * Full hierarchical decomposition execution:
 * Atomize → Plan → Execute → Aggregate → Verify
 */
export const ROMA_DECOMPOSITION_WORKFLOW = {
  name: 'roma_decomposition',
  description: 'ROMA hierarchical problem decomposition with full pipeline execution',
  version: '1.0.0',
  schema: RomaDecompositionWorkflowSchema,
  steps: [
    {
      name: 'atomize',
      description: 'Decompose problem into atomic sub-problems',
      action: 'roma.atomize',
      inputs: {
        goal: '$.goal',
        max_depth: '$.max_depth',
      },
      outputs: {
        decomposition: '$.atomize_result',
        sub_problems: '$.sub_problems',
      },
    },
    {
      name: 'plan',
      description: 'Create execution plan for sub-problems',
      action: 'roma.plan',
      inputs: {
        decomposition: '$.atomize_result',
      },
      outputs: {
        execution_plan: '$.plan_result',
      },
    },
    {
      name: 'execute',
      description: 'Execute atomic sub-problems',
      action: 'roma.execute_subtasks',
      inputs: {
        execution_plan: '$.plan_result',
        execution_method: '$.execution_method',
      },
      outputs: {
        subtask_results: '$.execution_results',
      },
    },
    {
      name: 'aggregate',
      description: 'Aggregate sub-task solutions',
      action: 'roma.aggregate',
      inputs: {
        subtask_results: '$.execution_results',
      },
      outputs: {
        aggregated_solution: '$.aggregated_result',
      },
    },
    {
      name: 'verify',
      description: 'Verify solution meets requirements',
      action: 'roma.verify',
      inputs: {
        aggregated_solution: '$.aggregated_result',
        enable_verification: '$.enable_verification',
      },
      outputs: {
        verification_result: '$.verification_result',
        final_solution: '$.final_result',
      },
    },
  ],
  error_handling: {
    retry_policy: {
      max_retries: 3,
      backoff_ms: 1000,
      exponential: true,
    },
    dead_letter_queue: true,
  },
};

/**
 * ROMA MDAP/MAKER Workflow
 *
 * Zero-error execution with k-ahead planning and adaptive strategies
 */
export const ROMA_MDAP_MAKER_WORKFLOW = {
  name: 'roma_mdap_maker',
  description: 'ROMA MDAP/MAKER zero-error execution with adaptive planning',
  version: '1.0.0',
  schema: RomaMdapMakerWorkflowSchema,
  steps: [
    {
      name: 'auto_select',
      description: 'Auto-select if task requires zero-error mode',
      action: 'roma.mdap_auto_select',
      inputs: {
        goal: '$.goal',
        keywords: '$.auto_selection_keywords',
      },
      outputs: {
        use_mdap_maker: '$.use_mdap',
        confidence: '$.selection_confidence',
      },
      condition: '$.use_mdap == true',
    },
    {
      name: 'k_ahead_planning',
      description: 'K-ahead planning with look-ahead',
      action: 'roma.mdap_k_ahead_planning',
      inputs: {
        goal: '$.goal',
        k_ahead: '$.k_ahead',
        max_depth: '$.max_depth',
      },
      outputs: {
        execution_plan: '$.k_ahead_plan',
        red_flags: '$.identified_red_flags',
      },
    },
    {
      name: 'adaptive_k',
      description: 'Adaptive K adjustment based on complexity',
      action: 'roma.mdap_adaptive_k',
      inputs: {
        execution_plan: '$.k_ahead_plan',
        enable_adaptive_k: '$.enable_adaptive_k',
      },
      outputs: {
        adjusted_plan: '$.adjusted_plan',
        final_k: '$.final_k_value',
      },
    },
    {
      name: 'hierarchical_voting',
      description: 'Hierarchical voting across ROMA hierarchy',
      action: 'roma.mdap_hierarchical_voting',
      inputs: {
        execution_plan: '$.adjusted_plan',
        provider: '$.provider',
        model: '$.model',
      },
      outputs: {
        voted_solution: '$.voted_result',
        confidence_scores: '$.confidences',
      },
    },
    {
      name: 'red_flag_verification',
      description: 'Red flagging and verification',
      action: 'roma.mdap_red_flag_verification',
      inputs: {
        voted_solution: '$.voted_result',
        enable_red_flagging: '$.enable_red_flagging',
        red_flags: '$.identified_red_flags',
      },
      outputs: {
        verification_result: '$.verification',
        final_solution: '$.final_result',
      },
    },
    {
      name: 'zero_error_validation',
      description: 'Zero-error validation loop',
      action: 'roma.mdap_zero_error_validation',
      inputs: {
        final_solution: '$.final_result',
        zero_error_mode: '$.zero_error_mode',
      },
      outputs: {
        is_zero_error: '$.zero_error_achieved',
        validation_report: '$.report',
      },
    },
  ],
  error_handling: {
    retry_policy: {
      max_retries: 5,
      backoff_ms: 2000,
      exponential: true,
    },
    dead_letter_queue: true,
  },
};

/**
 * ROMA Multi-Agent Workflow
 *
 * Multi-agent reasoning with prediction strategies
 */
export const ROMA_MULTI_AGENT_WORKFLOW = {
  name: 'roma_multi_agent',
  description: 'ROMA multi-agent reasoning with prediction strategy orchestration',
  version: '1.0.0',
  schema: RomaMultiAgentWorkflowSchema,
  steps: [
    {
      name: 'agent_initialization',
      description: 'Initialize multiple reasoning agents',
      action: 'roma.initialize_agents',
      inputs: {
        num_agents: '$.num_agents',
        prediction_strategies: '$.prediction_strategies',
      },
      outputs: {
        agents: '$.initialized_agents',
      },
    },
    {
      name: 'parallel_reasoning',
      description: 'Execute reasoning in parallel across agents',
      action: 'roma.parallel_reasoning',
      inputs: {
        goal: '$.goal',
        agents: '$.initialized_agents',
        reasoning_rounds: '$.reasoning_rounds',
      },
      outputs: {
        agent_results: '$.parallel_results',
      },
    },
    {
      name: 'voting_aggregation',
      description: 'Aggregate results with voting mechanism',
      action: 'roma.voting_aggregation',
      inputs: {
        agent_results: '$.parallel_results',
        enable_voting: '$.enable_voting',
      },
      outputs: {
        voted_result: '$.voted_solution',
        vote_distribution: '$.votes',
      },
    },
    {
      name: 'confidence_filtering',
      description: 'Filter results by confidence threshold',
      action: 'roma.confidence_filter',
      inputs: {
        voted_result: '$.voted_solution',
        confidence_threshold: '$.confidence_threshold',
      },
      outputs: {
        filtered_result: '$.final_result',
        confidence_met: '$.confidence_ok',
      },
    },
    {
      name: 'quality_metrics',
      description: 'Calculate quality metrics for execution',
      action: 'roma.quality_metrics',
      inputs: {
        agent_results: '$.parallel_results',
        final_result: '$.final_result',
      },
      outputs: {
        metrics: '$.quality_metrics',
        efficiency_score: '$.efficiency',
      },
    },
  ],
  error_handling: {
    retry_policy: {
      max_retries: 3,
      backoff_ms: 1500,
      exponential: true,
    },
    dead_letter_queue: true,
  },
};

/**
 * ROMA Hybrid Problem Solving Workflow
 *
 * Combines ROMA with OpenEvolve for hybrid problem solving
 */
export const ROMA_HYBRID_WORKFLOW = {
  name: 'roma_hybrid',
  description: 'ROMA + OpenEvolve hybrid problem solving',
  version: '1.0.0',
  schema: RomaHybridWorkflowSchema,
  steps: [
    {
      name: 'roma_decomposition',
      description: 'ROMA decomposes problem into sub-problems',
      action: 'roma.decompose',
      inputs: {
        goal: '$.goal',
        max_depth: '$.max_depth',
      },
      outputs: {
        decomposition: '$.roma_decomposition',
        sub_problems: '$.sub_problems',
      },
    },
    {
      name: 'openevolve_analysis',
      description: 'OpenEvolve analyzes each sub-problem',
      action: 'openevolve.analyze_batch',
      inputs: {
        sub_problems: '$.sub_problems',
      },
      condition: '$.enable_openevolve_integration == true',
      outputs: {
        analysis_results: '$.openevolve_analyses',
      },
    },
    {
      name: 'team_qa',
      description: 'Team-based quality assurance',
      action: 'roma.team_qa',
      inputs: {
        sub_problems: '$.sub_problems',
        openevolve_analyses: '$.openevolve_analyses',
      },
      condition: '$.enable_team_qa == true',
      outputs: {
        qa_results: '$.qa_results',
        approved_sub_problems: '$.approved_problems',
      },
    },
    {
      name: 'roma_execution',
      description: 'Execute approved sub-problems',
      action: 'roma.execute_subtasks',
      inputs: {
        approved_sub_problems: '$.approved_problems',
        execution_phases: '$.execution_phases',
      },
      outputs: {
        subtask_results: '$.execution_results',
      },
    },
    {
      name: 'hybrid_aggregation',
      description: 'Aggregate with hybrid approach',
      action: 'roma.hybrid_aggregate',
      inputs: {
        subtask_results: '$.execution_results',
        openevolve_analyses: '$.openevolve_analyses',
        qa_results: '$.qa_results',
      },
      outputs: {
        final_solution: '$.aggregated_solution',
      },
    },
    {
      name: 'hybrid_verification',
      description: 'Hybrid verification approach',
      action: 'roma.hybrid_verify',
      inputs: {
        final_solution: '$.aggregated_solution',
        enable_openevolve_integration: '$.enable_openevolve_integration',
      },
      outputs: {
        verification_result: '$.verification',
        final_result: '$.final_result',
      },
    },
  ],
  error_handling: {
    retry_policy: {
      max_retries: 4,
      backoff_ms: 1000,
      exponential: true,
    },
    dead_letter_queue: true,
  },
};

// ============================================================================
// WORKFLOW REGISTRY
// ============================================================================

/**
 * ROMA Workflow Templates Registry
 */
export const ROMA_WORKFLOW_TEMPLATES = {
  ROMA_DECOMPOSITION_WORKFLOW,
  ROMA_MDAP_MAKER_WORKFLOW,
  ROMA_MULTI_AGENT_WORKFLOW,
  ROMA_HYBRID_WORKFLOW,
};

/**
 * Get workflow template by name
 */
export function getRomaWorkflowTemplate(name: string) {
  const templateKey = Object.keys(ROMA_WORKFLOW_TEMPLATES).find(
    (key) => ROMA_WORKFLOW_TEMPLATES[key as keyof typeof ROMA_WORKFLOW_TEMPLATES].name === name
  );

  if (!templateKey) {
    throw new Error(`Unknown ROMA workflow template: ${name}`);
  }

  return ROMA_WORKFLOW_TEMPLATES[templateKey as keyof typeof ROMA_WORKFLOW_TEMPLATES];
}

/**
 * List all ROMA workflow templates
 */
export function listRomaWorkflowTemplates() {
  return Object.values(ROMA_WORKFLOW_TEMPLATES).map((template) => ({
    name: template.name,
    description: template.description,
    version: template.version,
  }));
}

/**
 * Validate workflow input against schema
 */
export function validateRomaWorkflowInput(
  workflowName: string,
  input: unknown
): { isValid: boolean; errors: string[] } {
  try {
    const template = getRomaWorkflowTemplate(workflowName);
    const result = template.schema.safeParse(input);

    if (result.success) {
      return { isValid: true, errors: [] };
    }

    const errors = result.error.errors.map(
      (err) => `${err.path.join('.')}: ${err.message}`
    );

    return { isValid: false, errors };
  } catch (error) {
    return {
      isValid: false,
      errors: [`Unknown workflow template: ${workflowName}`],
    };
  }
}
