/**
 * Knowledge Extraction Workflow
 *
 * Extracts evolutionary knowledge from PES executions and formulates
 * new problems based on insights.
 *
 * Flow:
 * 1. Query LoongFlow evolutionary database for best solutions
 * 2. Analyze solution patterns
 * 3. Extract knowledge fragments
 * 4. Store knowledge in Graphiti knowledge graph
 * 5. Vectorize knowledge for similarity search
 * 6. Formulate new problems based on knowledge gaps
 *
 * Following Federation Constitution:
 * - Law of Idempotency: Safe to run multiple times
 * - Law of the Untouchable DB: Read-only access to LoongFlow DB
 * - Law of UTC: All timestamps in ISO-8601 UTC
 * - Observability: Structured logging with correlation IDs
 */
import { EventBus } from '../event-bus';
import { LoongFlowAdapter, Solution } from '../../adapters/loongflow-adapter/src/adapter';
import { EvolutionaryKnowledge, KnowledgeType } from '../../schemas/hybrid-pes-evolution-canonical';
import { CorrelationContext } from '../correlation-tracker';
export interface KnowledgeExtractionWorkflowConfig {
    loongflowAdapter: LoongFlowAdapter;
    eventBus?: EventBus;
    graphitiAdapter?: any;
    vectorDBAdapter?: any;
    enable_graph_storage?: boolean;
    enable_vectorization?: boolean;
    enable_problem_formulation?: boolean;
}
export interface KnowledgeExtractionInput {
    source_solution_id?: string;
    island_id?: number;
    top_k?: number;
    min_score?: number;
    knowledge_types?: KnowledgeType[];
    problem_id?: string;
}
export interface SolutionPattern {
    pattern_id: string;
    pattern_type: string;
    frequency: number;
    success_rate: number;
    avg_score: number;
    examples: Solution[];
}
export interface FormulatedProblem {
    problem_id: string;
    problem_type: string;
    description: string;
    context: Record<string, any>;
    priority: number;
    based_on_knowledge: string[];
}
export declare class KnowledgeExtractionWorkflow {
    private readonly logger;
    private readonly eventBus;
    private readonly loongflowAdapter;
    private readonly graphitiAdapter?;
    private readonly vectorDBAdapter?;
    private readonly circuitBreaker;
    private readonly ENABLE_GRAPH_STORAGE;
    private readonly ENABLE_VECTORIZATION;
    private readonly ENABLE_PROBLEM_FORMULATION;
    private readonly DEFAULT_TOP_K;
    private readonly MIN_SCORE_THRESHOLD;
    constructor(config: KnowledgeExtractionWorkflowConfig);
    /**
     * Execute knowledge extraction workflow
     *
     * @param input - Extraction input parameters
     * @param correlationContext - Optional correlation context
     * @returns Extracted knowledge and formulated problems
     */
    execute(input: KnowledgeExtractionInput, correlationContext?: CorrelationContext): Promise<{
        knowledge: EvolutionaryKnowledge[];
        problems: FormulatedProblem[];
        patterns: SolutionPattern[];
    }>;
    /**
     * Step 1: Retrieve best solutions from LoongFlow evolutionary database
     */
    private stepRetrieveSolutions;
    /**
     * Step 2: Analyze solution patterns
     */
    private stepAnalyzePatterns;
    /**
     * Step 3: Extract knowledge
     */
    private stepExtractKnowledge;
    /**
     * Step 4: Store knowledge in Graphiti (optional)
     */
    private stepStoreInGraphiti;
    /**
     * Step 5: Vectorize knowledge for similarity search (optional)
     */
    private stepVectorizeKnowledge;
    /**
     * Step 6: Formulate new problems based on knowledge gaps (optional)
     */
    private stepFormulateProblems;
    /**
     * Extract pattern key from solution
     * This is a simplified implementation
     */
    private extractPatternKey;
    /**
     * Calculate average score of solutions
     */
    private calculateAverageScore;
    /**
     * Deduplicate knowledge by source_id and knowledge_type
     */
    private deduplicateKnowledge;
    private publishExtractionCompleted;
    private publishExtractionFailed;
    private publishStepFailed;
}
export declare function createKnowledgeExtractionWorkflow(config: KnowledgeExtractionWorkflowConfig): KnowledgeExtractionWorkflow;
//# sourceMappingURL=knowledge-extraction-workflow.d.ts.map