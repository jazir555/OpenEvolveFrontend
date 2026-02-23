/**
 * Adaptive Execution Workflow
 *
 * Monitors execution and dynamically switches between PES and Evolution
 * based on performance metrics and adaptive triggers.
 *
 * Flow:
 * 1. Start with PES execution
 * 2. Monitor confidence and progress
 * 3. If trigger conditions met, switch paradigms
 * 4. Continue with new paradigm
 * 5. Compare results and select best
 *
 * Following Federation Constitution:
 * - Law of Configuration Explicitness: All thresholds via ENV vars
 * - Law of Idempotency: State tracking for safe resume
 * - Observability: Detailed logging of paradigm switches
 */
import { EventBus } from '../event-bus';
import { LoongFlowAdapter } from '../../adapters/loongflow-adapter/src/adapter';
import { OpenEvolveAdapter } from '../../adapters/openevolve-adapter/src/adapter';
import { Problem } from '../../schemas/pes-canonical';
import { HybridExecutionResult, AdaptiveTrigger, AdaptiveAction } from '../../schemas/hybrid-pes-evolution-canonical';
import { CorrelationContext } from '../correlation-tracker';
export interface AdaptiveExecutionWorkflowConfig {
    loongflowAdapter: LoongFlowAdapter;
    openevolveAdapter: OpenEvolveAdapter;
    eventBus?: EventBus;
    default_timeout_ms?: number;
    max_paradigm_switches?: number;
    max_iterations?: number;
}
export interface AdaptiveExecutionInput {
    problem: Problem;
    initial_paradigm?: 'PES' | 'EVOLUTION';
    triggers: AdaptiveTrigger[];
    enable_hybrid_fallback?: boolean;
}
export interface ParadigmExecutionState {
    paradigm: 'PES' | 'EVOLUTION' | 'HYBRID';
    iteration: number;
    start_time: number;
    result?: any;
    metrics?: any;
}
export interface TriggerEvaluation {
    triggered: boolean;
    trigger?: AdaptiveTrigger;
    reason?: string;
    suggested_action?: AdaptiveAction;
}
export declare class AdaptiveExecutionWorkflow {
    private readonly logger;
    private readonly eventBus;
    private readonly loongflowAdapter;
    private readonly openevolveAdapter;
    private readonly circuitBreaker;
    private readonly DEFAULT_TIMEOUT_MS;
    private readonly MAX_PARADIGM_SWITCHES;
    private readonly MAX_ITERATIONS;
    private readonly DEFAULT_CONFIDENCE_THRESHOLD;
    private readonly DEFAULT_STAGNATION_THRESHOLD;
    constructor(config: AdaptiveExecutionWorkflowConfig);
    /**
     * Execute adaptive workflow with paradigm switching
     *
     * @param input - Adaptive execution input
     * @param correlationContext - Optional correlation context
     * @returns Hybrid execution result
     */
    executeAdaptive(input: AdaptiveExecutionInput, correlationContext?: CorrelationContext): Promise<HybridExecutionResult>;
    /**
     * Execute with a specific paradigm
     */
    private executeParadigm;
    /**
     * Execute PES paradigm
     */
    private executePES;
    /**
     * Execute Evolution paradigm
     */
    private executeEvolution;
    /**
     * Execute Hybrid paradigm
     */
    private executeHybrid;
    /**
     * Evaluate adaptive triggers
     */
    private evaluateTriggers;
    /**
     * Evaluate a single trigger
     */
    private evaluateSingleTrigger;
    /**
     * Determine new paradigm based on action
     */
    private determineNewParadigm;
    private extractScore;
    private isConverged;
    private isSuccess;
    private buildIntegrationMetrics;
    private buildAdaptiveSummary;
    private sleep;
    private publishParadigmSwitched;
    private publishAdaptiveCompleted;
    private publishAdaptiveFailed;
}
export declare function createAdaptiveExecutionWorkflow(config: AdaptiveExecutionWorkflowConfig): AdaptiveExecutionWorkflow;
//# sourceMappingURL=adaptive-execution-workflow.d.ts.map