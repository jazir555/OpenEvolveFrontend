/**
 * Multi-Stage Reasoning Workflow
 *
 * Orchestrates complex reasoning across multiple systems:
 * 1. Plan (LoongFlow)
 * 2. Optimize (OpenEvolve)
 * 3. Validate (Z3 or LeanAide)
 * 4. Refine (OpenEvolve)
 * 5. Summarize (LoongFlow)
 * 6. Extract knowledge
 *
 * This workflow enables advanced multi-system reasoning with formal validation.
 *
 * Following Federation Constitution:
 * - Law of Configuration Explicitness: All timeouts and thresholds via ENV vars
 * - Law of Idempotency: Checkpoints for each stage
 * - Observability: Comprehensive event publishing
 */
import { EventBus } from '../event-bus';
import { LoongFlowAdapter } from '../../adapters/loongflow-adapter/src/adapter';
import { OpenEvolveAdapter } from '../../adapters/openevolve-adapter/src/adapter';
import { Problem } from '../../schemas/pes-canonical';
import { HybridExecutionResult } from '../../schemas/hybrid-pes-evolution-canonical';
import { CorrelationContext } from '../correlation-tracker';
export interface MultiStageReasoningWorkflowConfig {
    loongflowAdapter: LoongFlowAdapter;
    openevolveAdapter: OpenEvolveAdapter;
    z3Adapter?: any;
    leanAideAdapter?: any;
    eventBus?: EventBus;
    default_timeout_ms?: number;
    enable_validation?: boolean;
    enable_refinement?: boolean;
    max_refinement_loops?: number;
}
export interface MultiStageReasoningInput {
    problem: Problem;
    validation_system?: 'z3' | 'lean-aide' | 'both';
    refinement_threshold?: number;
    enable_parallel_stages?: boolean;
}
export interface ValidationResult {
    valid: boolean;
    system: 'z3' | 'lean-aide' | 'both';
    confidence: number;
    errors?: string[];
    warnings?: string[];
}
export interface RefinementResult {
    refined: boolean;
    iterations: number;
    final_score: number;
    improvements: string[];
}
export interface StageResult {
    stage_name: string;
    status: 'pending' | 'in_progress' | 'completed' | 'failed';
    result?: any;
    duration_ms: number;
    error?: string;
}
export declare class MultiStageReasoningWorkflow {
    private readonly logger;
    private readonly eventBus;
    private readonly loongflowAdapter;
    private readonly openevolveAdapter;
    private readonly z3Adapter?;
    private readonly leanAideAdapter?;
    private readonly circuitBreaker;
    private readonly DEFAULT_TIMEOUT_MS;
    private readonly ENABLE_VALIDATION;
    private readonly ENABLE_REFINEMENT;
    private readonly MAX_REFINEMENT_LOOPS;
    private readonly DEFAULT_REFINEMENT_THRESHOLD;
    constructor(config: MultiStageReasoningWorkflowConfig);
    /**
     * Execute multi-stage reasoning workflow
     *
     * @param input - Multi-stage reasoning input
     * @param correlationContext - Optional correlation context
     * @returns Hybrid execution result
     */
    executeReasoning(input: MultiStageReasoningInput, correlationContext?: CorrelationContext): Promise<HybridExecutionResult>;
    /**
     * Stage 1: Plan - Create execution plan using LoongFlow
     */
    private stagePlan;
    /**
     * Stage 2: Optimize - Optimize solution using OpenEvolve
     */
    private stageOptimize;
    /**
     * Stage 3: Validate - Validate solution using Z3 or LeanAide
     */
    private stageValidate;
    /**
     * Stage 4: Refine - Refine solution if validation failed
     */
    private stageRefine;
    /**
     * Stage 5: Summarize - Summarize results
     */
    private stageSummarize;
    /**
     * Stage 6: Extract knowledge - Extract and store knowledge
     */
    private stageExtractKnowledge;
    private buildMetrics;
    private sleep;
    private publishWorkflowCompleted;
    private publishWorkflowFailed;
    private publishStageFailed;
}
export declare function createMultiStageReasoningWorkflow(config: MultiStageReasoningWorkflowConfig): MultiStageReasoningWorkflow;
//# sourceMappingURL=multi-stage-reasoning-workflow.d.ts.map