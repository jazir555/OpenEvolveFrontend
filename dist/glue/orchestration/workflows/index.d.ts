/**
 * Hybrid Orchestration Workflows
 *
 * This module exports all hybrid workflows that orchestrate LoongFlow (PES)
 * and OpenEvolve (Evolution) in powerful hybrid patterns for advanced AI problem-solving.
 *
 * Workflow Types:
 * - PES_EVOLUTION: Sequential PES → Evolution → Summary
 * - KNOWLEDGE_EXTRACTION: Extract knowledge → formulate new problems
 * - ADAPTIVE_EXECUTION: Dynamically switch paradigms based on triggers
 * - MULTI_STAGE_REASONING: Complex multi-system reasoning with validation
 *
 * Following Federation Constitution:
 * - Law of Configuration Explicitness: All config via ENV vars
 * - Law of Idempotency: All workflows safe to retry
 * - Observability: Structured logging with correlation IDs
 */
export { PESEvolutionWorkflow, PESEvolutionWorkflowConfig, PESEvolutionInput, WorkflowCheckpoint, createPESEvolutionWorkflow, } from './pes-evolution-workflow';
export { KnowledgeExtractionWorkflow, KnowledgeExtractionWorkflowConfig, KnowledgeExtractionInput, SolutionPattern, FormulatedProblem, createKnowledgeExtractionWorkflow, } from './knowledge-extraction-workflow';
export { AdaptiveExecutionWorkflow, AdaptiveExecutionWorkflowConfig, AdaptiveExecutionInput, ParadigmExecutionState, TriggerEvaluation, createAdaptiveExecutionWorkflow, } from './adaptive-execution-workflow';
export { MultiStageReasoningWorkflow, MultiStageReasoningWorkflowConfig, MultiStageReasoningInput, ValidationResult, RefinementResult, StageResult, createMultiStageReasoningWorkflow, } from './multi-stage-reasoning-workflow';
/**
 * Available workflow types
 */
export declare const WORKFLOWS: {
    readonly PES_EVOLUTION: "pes-evolution";
    readonly KNOWLEDGE_EXTRACTION: "knowledge-extraction";
    readonly ADAPTIVE_EXECUTION: "adaptive-execution";
    readonly MULTI_STAGE_REASONING: "multi-stage-reasoning";
};
export type WorkflowType = typeof WORKFLOWS[keyof typeof WORKFLOWS];
/**
 * Workflow interface
 */
export interface Workflow {
    execute(input: any, correlationContext?: any): Promise<any>;
}
/**
 * Workflow factory function
 *
 * Creates workflow instances based on type
 *
 * @param type - Workflow type
 * @param config - Workflow configuration
 * @returns Workflow instance
 */
export declare function createWorkflow(type: WorkflowType, config: any): Workflow;
/**
 * Get all available workflow types
 */
export declare function getAvailableWorkflowTypes(): WorkflowType[];
/**
 * Validate workflow configuration
 */
export declare function validateWorkflowConfig(type: WorkflowType, config: any): {
    valid: boolean;
    errors?: string[];
};
/**
 * Workflow metadata
 */
export declare const WORKFLOW_METADATA: {
    readonly "pes-evolution": {
        readonly name: "PES-Evolution Hybrid Workflow";
        readonly description: "Sequential PES → Evolution → Summary for enhanced problem-solving";
        readonly required_adapters: readonly ["loongflow", "openevolve"];
        readonly optional_adapters: readonly [];
        readonly estimated_duration_ms: 300000;
        readonly timeout_ms: 600000;
    };
    readonly "knowledge-extraction": {
        readonly name: "Knowledge Extraction Workflow";
        readonly description: "Extract knowledge from solutions and formulate new problems";
        readonly required_adapters: readonly ["loongflow"];
        readonly optional_adapters: readonly ["graphiti", "vectordb"];
        readonly estimated_duration_ms: 60000;
        readonly timeout_ms: 120000;
    };
    readonly "adaptive-execution": {
        readonly name: "Adaptive Execution Workflow";
        readonly description: "Dynamically switch between PES and Evolution based on triggers";
        readonly required_adapters: readonly ["loongflow", "openevolve"];
        readonly optional_adapters: readonly [];
        readonly estimated_duration_ms: 600000;
        readonly timeout_ms: 900000;
    };
    readonly "multi-stage-reasoning": {
        readonly name: "Multi-Stage Reasoning Workflow";
        readonly description: "Complex multi-system reasoning with Plan, Optimize, Validate, Refine, Summarize";
        readonly required_adapters: readonly ["loongflow", "openevolve"];
        readonly optional_adapters: readonly ["z3", "lean-aide"];
        readonly estimated_duration_ms: 600000;
        readonly timeout_ms: 1200000;
    };
};
/**
 * Get workflow metadata
 */
export declare function getWorkflowMetadata(type: WorkflowType): {
    readonly name: "PES-Evolution Hybrid Workflow";
    readonly description: "Sequential PES → Evolution → Summary for enhanced problem-solving";
    readonly required_adapters: readonly ["loongflow", "openevolve"];
    readonly optional_adapters: readonly [];
    readonly estimated_duration_ms: 300000;
    readonly timeout_ms: 600000;
} | {
    readonly name: "Knowledge Extraction Workflow";
    readonly description: "Extract knowledge from solutions and formulate new problems";
    readonly required_adapters: readonly ["loongflow"];
    readonly optional_adapters: readonly ["graphiti", "vectordb"];
    readonly estimated_duration_ms: 60000;
    readonly timeout_ms: 120000;
} | {
    readonly name: "Adaptive Execution Workflow";
    readonly description: "Dynamically switch between PES and Evolution based on triggers";
    readonly required_adapters: readonly ["loongflow", "openevolve"];
    readonly optional_adapters: readonly [];
    readonly estimated_duration_ms: 600000;
    readonly timeout_ms: 900000;
} | {
    readonly name: "Multi-Stage Reasoning Workflow";
    readonly description: "Complex multi-system reasoning with Plan, Optimize, Validate, Refine, Summarize";
    readonly required_adapters: readonly ["loongflow", "openevolve"];
    readonly optional_adapters: readonly ["z3", "lean-aide"];
    readonly estimated_duration_ms: 600000;
    readonly timeout_ms: 1200000;
};
/**
 * Get all workflow metadata
 */
export declare function getAllWorkflowMetadata(): {
    readonly "pes-evolution": {
        readonly name: "PES-Evolution Hybrid Workflow";
        readonly description: "Sequential PES → Evolution → Summary for enhanced problem-solving";
        readonly required_adapters: readonly ["loongflow", "openevolve"];
        readonly optional_adapters: readonly [];
        readonly estimated_duration_ms: 300000;
        readonly timeout_ms: 600000;
    };
    readonly "knowledge-extraction": {
        readonly name: "Knowledge Extraction Workflow";
        readonly description: "Extract knowledge from solutions and formulate new problems";
        readonly required_adapters: readonly ["loongflow"];
        readonly optional_adapters: readonly ["graphiti", "vectordb"];
        readonly estimated_duration_ms: 60000;
        readonly timeout_ms: 120000;
    };
    readonly "adaptive-execution": {
        readonly name: "Adaptive Execution Workflow";
        readonly description: "Dynamically switch between PES and Evolution based on triggers";
        readonly required_adapters: readonly ["loongflow", "openevolve"];
        readonly optional_adapters: readonly [];
        readonly estimated_duration_ms: 600000;
        readonly timeout_ms: 900000;
    };
    readonly "multi-stage-reasoning": {
        readonly name: "Multi-Stage Reasoning Workflow";
        readonly description: "Complex multi-system reasoning with Plan, Optimize, Validate, Refine, Summarize";
        readonly required_adapters: readonly ["loongflow", "openevolve"];
        readonly optional_adapters: readonly ["z3", "lean-aide"];
        readonly estimated_duration_ms: 600000;
        readonly timeout_ms: 1200000;
    };
};
export type { Problem, ExecutionPlan, ExecutionResult, Summary, } from '../../schemas/pes-canonical';
export type { HybridTask, HybridExecutionResult, EvolutionaryKnowledge, EvolutionConfig, EvolutionResult, IntegrationMetrics, AdaptiveTrigger, } from '../../schemas/hybrid-pes-evolution-canonical';
/**
 * Example usage:
 *
 * ```typescript
 * import {
 *   createWorkflow,
 *   WORKFLOWS,
 *   createPESEvolutionWorkflow,
 *   validateWorkflowConfig,
 *   getWorkflowMetadata,
 * } from './workflows';
 *
 * import { LoongFlowAdapter } from '../../adapters/loongflow-adapter/src/adapter';
 * import { OpenEvolveAdapter } from '../../adapters/openevolve-adapter/src/adapter';
 *
 * // Create adapters
 * const loongflowAdapter = new LoongFlowAdapter({
 *   api_url: process.env.LOONGFLOW_API_URL!,
 *   timeout_ms: 30000,
 * });
 *
 * const openevolveAdapter = new OpenEvolveAdapter({
 *   api_url: process.env.OPENEVOLVE_API_URL!,
 *   timeout_ms: 30000,
 * });
 *
 * // Create PES-Evolution workflow using factory
 * const pesEvolutionWorkflow = createWorkflow(
 *   WORKFLOWS.PES_EVOLUTION,
 *   {
 *     loongflowAdapter,
 *     openevolveAdapter,
 *     checkpoints_enabled: true,
 *   }
 * );
 *
 * // Or create directly
 * const directWorkflow = createPESEvolutionWorkflow({
 *   loongflowAdapter,
 *   openevolveAdapter,
 * });
 *
 * // Validate configuration
 * const validation = validateWorkflowConfig(
 *   WORKFLOWS.PES_EVOLUTION,
 *   { loongflowAdapter, openevolveAdapter }
 * );
 *
 * if (!validation.valid) {
 *   console.error('Invalid config:', validation.errors);
 * }
 *
 * // Get workflow metadata
 * const metadata = getWorkflowMetadata(WORKFLOWS.PES_EVOLUTION);
 * console.log('Workflow:', metadata.name);
 * console.log('Description:', metadata.description);
 * console.log('Required adapters:', metadata.required_adapters);
 *
 * // Execute workflow
 * const result = await pesEvolutionWorkflow.execute(
 *   {
 *     problem: {
 *       id: 'problem-123',
 *       type: 'optimization',
 *       description: 'Optimize neural network hyperparameters',
 *       context: { dataset: 'MNIST' },
 *       constraints: ['max_layers <= 10'],
 *       success_criteria: ['accuracy > 0.95'],
 *       created_at: new Date().toISOString(),
 *     },
 *     pes_config: {
 *       max_iterations: 10,
 *       target_score: 0.9,
 *     },
 *     evolution_config: {
 *       generations: 10,
 *       population_size: 100,
 *     },
 *   }
 * );
 *
 * console.log('Workflow result:', result);
 * ```
 */
//# sourceMappingURL=index.d.ts.map