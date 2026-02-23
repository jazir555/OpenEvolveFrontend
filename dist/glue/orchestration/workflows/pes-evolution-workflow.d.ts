/**
 * PES-Evolution Hybrid Workflow
 *
 * This workflow orchestrates LoongFlow's Plan-Execute-Summarize (PES)
 * with OpenEvolve's evolutionary optimization for enhanced problem-solving.
 *
 * Flow:
 * 1. Receive problem
 * 2. LoongFlow plans solution
 * 3. LoongFlow executes initial solution
 * 4. OpenEvolve optimizes the solution
 * 5. LoongFlow summarizes results
 * 6. Extract evolutionary knowledge
 * 7. Store knowledge for future use
 *
 * Following Federation Constitution:
 * - Law of Configuration Explicitness: All timeouts, retries via ENV vars
 * - Law of Idempotency: Checkpoints support resume capability
 * - Law of UTC: All timestamps in ISO-8601 UTC
 * - Failure Management: Circuit breakers on all adapter calls
 * - Observability: Structured logging with correlation IDs
 */
import { EventBus } from '../event-bus';
import { LoongFlowAdapter } from '../../adapters/loongflow-adapter/src/adapter';
import { OpenEvolveAdapter } from '../../adapters/openevolve-adapter/src/adapter';
import { Problem } from '../../schemas/pes-canonical';
import { HybridExecutionResult, EvolutionConfig } from '../../schemas/hybrid-pes-evolution-canonical';
import { CorrelationContext } from '../correlation-tracker';
export interface PESEvolutionWorkflowConfig {
    loongflowAdapter: LoongFlowAdapter;
    openevolveAdapter: OpenEvolveAdapter;
    eventBus?: EventBus;
    checkpoints_enabled?: boolean;
    checkpoint_path?: string;
    default_timeout_ms?: number;
    max_retries?: number;
}
export interface PESEvolutionInput {
    problem: Problem;
    pes_config?: {
        max_iterations?: number;
        target_score?: number;
        concurrency?: number;
    };
    evolution_config?: EvolutionConfig;
    enable_optimization?: boolean;
    enable_knowledge_extraction?: boolean;
}
export interface WorkflowCheckpoint {
    checkpoint_id: string;
    task_id: string;
    stage: 'plan' | 'execute' | 'optimize' | 'summarize' | 'complete';
    timestamp: string;
    data: any;
}
export declare class PESEvolutionWorkflow {
    private readonly logger;
    private readonly eventBus;
    private readonly loongflowAdapter;
    private readonly openevolveAdapter;
    private readonly circuitBreaker;
    private readonly checkpoints;
    private readonly DEFAULT_TIMEOUT_MS;
    private readonly MAX_RETRIES;
    private readonly CHECKPOINTS_ENABLED;
    constructor(config: PESEvolutionWorkflowConfig);
    /**
     * Execute the PES-Evolution hybrid workflow
     *
     * @param input - Workflow input with problem and configurations
     * @param correlationContext - Optional correlation context for distributed tracing
     * @returns Hybrid execution result
     */
    execute(input: PESEvolutionInput, correlationContext?: CorrelationContext): Promise<HybridExecutionResult>;
    /**
     * Stage 1: Plan - Create execution plan using LoongFlow
     */
    private stagePlan;
    /**
     * Stage 2: Execute - Execute plan using LoongFlow PES
     */
    private stageExecute;
    /**
     * Stage 3: Optimize - Optimize solution using OpenEvolve
     */
    private stageOptimize;
    /**
     * Stage 4: Summarize - Summarize results using LoongFlow
     */
    private stageSummarize;
    /**
     * Stage 5: Extract knowledge - Extract and store evolutionary knowledge
     */
    private stageExtractKnowledge;
    /**
     * Calculate integration metrics
     */
    private calculateMetrics;
    /**
     * Calculate synergy score between PES and Evolution
     */
    private calculateSynergyScore;
    /**
     * Save checkpoint for resume capability
     */
    private saveCheckpoint;
    /**
     * Execute function with timeout
     */
    private withTimeout;
    /**
     * Sleep for specified milliseconds
     */
    private sleep;
    private publishWorkflowCompleted;
    private publishWorkflowFailed;
    private publishStageFailed;
    /**
     * Get checkpoint by ID
     */
    getCheckpoint(checkpointId: string): WorkflowCheckpoint | undefined;
    /**
     * Get all checkpoints for a task
     */
    getCheckpointsForTask(taskId: string): WorkflowCheckpoint[];
    /**
     * Resume workflow from checkpoint
     */
    resumeFromCheckpoint(checkpointId: string, correlationContext?: CorrelationContext): Promise<HybridExecutionResult>;
    /**
     * Clear checkpoints for a task
     */
    clearCheckpoints(taskId: string): void;
}
export declare function createPESEvolutionWorkflow(config: PESEvolutionWorkflowConfig): PESEvolutionWorkflow;
//# sourceMappingURL=pes-evolution-workflow.d.ts.map