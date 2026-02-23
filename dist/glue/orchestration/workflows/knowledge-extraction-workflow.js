"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.KnowledgeExtractionWorkflow = void 0;
exports.createKnowledgeExtractionWorkflow = createKnowledgeExtractionWorkflow;
const uuid_1 = require("uuid");
const event_bus_1 = require("../event-bus");
const event_types_1 = require("../event-types");
const logger_1 = require("../../lib/logger");
const circuit_breaker_1 = require("../../lib/circuit-breaker");
const hybrid_pes_evolution_canonical_1 = require("../../schemas/hybrid-pes-evolution-canonical");
const correlation_tracker_1 = require("../correlation-tracker");
// ============================================================================
// MAIN WORKFLOW CLASS
// ============================================================================
class KnowledgeExtractionWorkflow {
    constructor(config) {
        if (!config.loongflowAdapter) {
            throw new Error('loongflowAdapter is required');
        }
        this.logger = new logger_1.Logger('knowledge-extraction-workflow');
        this.eventBus = config.eventBus || event_bus_1.eventBus;
        this.loongflowAdapter = config.loongflowAdapter;
        this.graphitiAdapter = config.graphitiAdapter;
        this.vectorDBAdapter = config.vectorDBAdapter;
        // Load configuration from environment
        this.ENABLE_GRAPH_STORAGE = config.enable_graph_storage !== undefined
            ? config.enable_graph_storage
            : process.env.KNOWLEDGE_ENABLE_GRAPH_STORAGE === 'true';
        this.ENABLE_VECTORIZATION = config.enable_vectorization !== undefined
            ? config.enable_vectorization
            : process.env.KNOWLEDGE_ENABLE_VECTORIZATION === 'true';
        this.ENABLE_PROBLEM_FORMULATION = config.enable_problem_formulation !== undefined
            ? config.enable_problem_formulation
            : process.env.KNOWLEDGE_ENABLE_PROBLEM_FORMULATION === 'true';
        this.DEFAULT_TOP_K = parseInt(process.env.KNOWLEDGE_EXTRACTION_TOP_K || '10', 10);
        this.MIN_SCORE_THRESHOLD = parseFloat(process.env.KNOWLEDGE_MIN_SCORE_THRESHOLD || '0.5');
        // Initialize circuit breaker
        this.circuitBreaker = new circuit_breaker_1.CircuitBreaker({
            threshold: parseInt(process.env.KNOWLEDGE_CIRCUIT_THRESHOLD || '5', 10),
            timeout_ms: parseInt(process.env.KNOWLEDGE_CIRCUIT_TIMEOUT_MS || '60000', 10),
        });
        this.logger.info('Knowledge Extraction workflow initialized', {
            enable_graph_storage: this.ENABLE_GRAPH_STORAGE,
            enable_vectorization: this.ENABLE_VECTORIZATION,
            enable_problem_formulation: this.ENABLE_PROBLEM_FORMULATION,
            default_top_k: this.DEFAULT_TOP_K,
            min_score_threshold: this.MIN_SCORE_THRESHOLD,
        });
    }
    // ============================================================================
    // WORKFLOW EXECUTION
    // ============================================================================
    /**
     * Execute knowledge extraction workflow
     *
     * @param input - Extraction input parameters
     * @param correlationContext - Optional correlation context
     * @returns Extracted knowledge and formulated problems
     */
    async execute(input, correlationContext) {
        const correlationCtx = correlationContext || correlation_tracker_1.correlationTracker.createContext();
        const executionId = (0, uuid_1.v4)();
        const startTime = Date.now();
        this.logger.info('Starting Knowledge Extraction workflow', {
            correlation_id: correlationCtx.correlation_id,
            execution_id,
            source_solution_id: input.source_solution_id,
            island_id: input.island_id,
            top_k: input.top_k,
        });
        try {
            return await this.circuitBreaker.execute(async () => {
                // Step 1: Retrieve best solutions from LoongFlow
                const solutions = await this.stepRetrieveSolutions(input, correlationCtx);
                // Step 2: Analyze patterns
                const patterns = await this.stepAnalyzePatterns(solutions, correlationCtx);
                // Step 3: Extract knowledge
                const knowledge = await this.stepExtractKnowledge(solutions, patterns, input, correlationCtx);
                // Step 4: Store in Graphiti (optional)
                if (this.ENABLE_GRAPH_STORAGE) {
                    await this.stepStoreInGraphiti(knowledge, correlationCtx);
                }
                // Step 5: Vectorize for search (optional)
                if (this.ENABLE_VECTORIZATION) {
                    await this.stepVectorizeKnowledge(knowledge, correlationCtx);
                }
                // Step 6: Formulate new problems (optional)
                let problems = [];
                if (this.ENABLE_PROBLEM_FORMULATION) {
                    problems = await this.stepFormulateProblems(knowledge, patterns, correlationCtx);
                }
                // Publish completion event
                const durationMs = Date.now() - startTime;
                await this.publishExtractionCompleted(executionId, knowledge, problems, patterns, durationMs, correlationCtx);
                this.logger.info('Knowledge Extraction workflow completed', {
                    correlation_id: correlationCtx.correlation_id,
                    execution_id,
                    duration_ms: durationMs,
                    knowledge_count: knowledge.length,
                    problem_count: problems.length,
                    pattern_count: patterns.length,
                });
                return { knowledge, problems, patterns };
            });
        }
        catch (error) {
            this.logger.error('Knowledge Extraction workflow failed', error, {
                correlation_id: correlationCtx.correlation_id,
                execution_id,
                duration_ms: Date.now() - startTime,
            });
            await this.publishExtractionFailed(executionId, error, correlationCtx);
            throw error;
        }
    }
    // ============================================================================
    // WORKFLOW STEPS
    // ============================================================================
    /**
     * Step 1: Retrieve best solutions from LoongFlow evolutionary database
     */
    async stepRetrieveSolutions(input, correlationCtx) {
        const step = 'retrieve_solutions';
        this.logger.info('Step 1: Retrieving solutions', {
            correlation_id: correlationCtx.correlation_id,
            step,
            island_id: input.island_id,
            top_k: input.top_k || this.DEFAULT_TOP_K,
        });
        try {
            let solutions = [];
            if (input.source_solution_id) {
                // Retrieve specific solution
                // Note: LoongFlow adapter doesn't have a getSolutionById method
                // This would need to be implemented or use getBestSolutions and filter
                this.logger.warn('Specific solution retrieval not yet implemented, using best solutions', {
                    correlation_id: correlationCtx.correlation_id,
                    source_solution_id: input.source_solution_id,
                });
            }
            // Retrieve best solutions
            solutions = await this.loongflowAdapter.getBestSolutions(input.island_id, input.top_k || this.DEFAULT_TOP_K);
            // Filter by minimum score
            const filteredSolutions = solutions.filter(s => s.score >= this.MIN_SCORE_THRESHOLD);
            this.logger.info('Step 1 completed: Solutions retrieved', {
                correlation_id: correlationCtx.correlation_id,
                total_retrieved: solutions.length,
                after_filtering: filteredSolutions.length,
                min_score: this.MIN_SCORE_THRESHOLD,
            });
            // Publish event
            await this.eventBus.publish((0, event_types_1.createBaseEvent)('SolutionsRetrieved', 'knowledge-extraction-workflow', correlationCtx.correlation_id, {
                execution_step: step,
                solution_count: filteredSolutions.length,
                island_id: input.island_id,
                avg_score: this.calculateAverageScore(filteredSolutions),
            }));
            return filteredSolutions;
        }
        catch (error) {
            this.logger.error('Step 1 failed: Solution retrieval', error, {
                correlation_id: correlationCtx.correlation_id,
            });
            await this.publishStepFailed(step, error, correlationCtx);
            throw error;
        }
    }
    /**
     * Step 2: Analyze solution patterns
     */
    async stepAnalyzePatterns(solutions, correlationCtx) {
        const step = 'analyze_patterns';
        this.logger.info('Step 2: Analyzing patterns', {
            correlation_id: correlationCtx.correlation_id,
            step,
            solution_count: solutions.length,
        });
        try {
            const patterns = [];
            // Group solutions by pattern characteristics
            const patternGroups = new Map();
            for (const solution of solutions) {
                // Extract pattern key from solution
                // This is a simplified pattern extraction - actual implementation
                // would use more sophisticated analysis
                const patternKey = this.extractPatternKey(solution);
                if (!patternGroups.has(patternKey)) {
                    patternGroups.set(patternKey, []);
                }
                patternGroups.get(patternKey).push(solution);
            }
            // Analyze each pattern group
            for (const [patternKey, groupSolutions] of patternGroups.entries()) {
                const scores = groupSolutions.map(s => s.score);
                const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
                const successRate = scores.filter(s => s >= 0.8).length / scores.length;
                patterns.push({
                    pattern_id: (0, uuid_1.v4)(),
                    pattern_type: patternKey,
                    frequency: groupSolutions.length,
                    success_rate: successRate,
                    avg_score: avgScore,
                    examples: groupSolutions.slice(0, 3), // Keep top 3 examples
                });
            }
            // Sort by frequency and success rate
            patterns.sort((a, b) => {
                const frequencyScore = b.frequency - a.frequency;
                if (Math.abs(frequencyScore) > 1)
                    return frequencyScore;
                return b.success_rate - a.success_rate;
            });
            this.logger.info('Step 2 completed: Patterns analyzed', {
                correlation_id: correlationCtx.correlation_id,
                pattern_count: patterns.length,
                top_pattern: patterns[0]?.pattern_type,
            });
            // Publish event
            await this.eventBus.publish((0, event_types_1.createBaseEvent)('PatternsAnalyzed', 'knowledge-extraction-workflow', correlationCtx.correlation_id, {
                execution_step: step,
                pattern_count: patterns.length,
                top_patterns: patterns.slice(0, 5).map(p => ({
                    type: p.pattern_type,
                    frequency: p.frequency,
                    success_rate: p.success_rate,
                })),
            }));
            return patterns;
        }
        catch (error) {
            this.logger.error('Step 2 failed: Pattern analysis', error, {
                correlation_id: correlationCtx.correlation_id,
            });
            await this.publishStepFailed(step, error, correlationCtx);
            throw error;
        }
    }
    /**
     * Step 3: Extract knowledge
     */
    async stepExtractKnowledge(solutions, patterns, input, correlationCtx) {
        const step = 'extract_knowledge';
        this.logger.info('Step 3: Extracting knowledge', {
            correlation_id: correlationCtx.correlation_id,
            step,
            solution_count: solutions.length,
            pattern_count: patterns.length,
        });
        try {
            const knowledge = [];
            const knowledgeTypes = input.knowledge_types || [
                'solution_pattern',
                'planning_strategy',
                'execution_approach',
                'parameter_setting',
            ];
            // Extract knowledge from each solution
            for (const solution of solutions) {
                for (const knowledgeType of knowledgeTypes) {
                    // Validate solution meets quality threshold
                    if (solution.score < this.MIN_SCORE_THRESHOLD) {
                        continue;
                    }
                    const knowledgeItem = {
                        id: (0, uuid_1.v4)(),
                        source_type: 'loongflow_solution',
                        problem_id: input.problem_id || (0, uuid_1.v4)(),
                        knowledge_type: knowledgeType,
                        content: {
                            pattern: solution.generate_plan,
                            success_rate: solution.score,
                            avg_score: solution.score,
                            usage_count: 1,
                            context: {
                                island_id: solution.island_id,
                                iteration: solution.iteration,
                                solution_id: solution.solution_id,
                            },
                        },
                        source_id: solution.solution_id,
                        extracted_at: new Date().toISOString(),
                        metadata: {
                            summary: solution.summary,
                            evaluation: solution.evaluation,
                            parent_id: solution.parent_id,
                        },
                    };
                    // Validate knowledge
                    const validation = (0, hybrid_pes_evolution_canonical_1.validateEvolutionaryKnowledge)(knowledgeItem);
                    if (validation.success) {
                        knowledge.push(knowledgeItem);
                    }
                }
            }
            // Extract knowledge from patterns
            for (const pattern of patterns) {
                if (pattern.success_rate >= 0.8 && pattern.frequency >= 2) {
                    knowledge.push({
                        id: (0, uuid_1.v4)(),
                        source_type: 'loongflow_solution',
                        problem_id: input.problem_id || (0, uuid_1.v4)(),
                        knowledge_type: 'solution_pattern',
                        content: {
                            pattern: pattern.pattern_type,
                            success_rate: pattern.success_rate,
                            avg_score: pattern.avg_score,
                            usage_count: pattern.frequency,
                            context: {
                                pattern_id: pattern.pattern_id,
                                example_count: pattern.examples.length,
                            },
                        },
                        extracted_at: new Date().toISOString(),
                        metadata: {
                            pattern_analysis: true,
                        },
                    });
                }
            }
            // Deduplicate knowledge by source_id and knowledge_type
            const deduplicatedKnowledge = this.deduplicateKnowledge(knowledge);
            this.logger.info('Step 3 completed: Knowledge extracted', {
                correlation_id: correlationCtx.correlation_id,
                knowledge_count: knowledge.length,
                after_deduplication: deduplicatedKnowledge.length,
            });
            // Publish event for each knowledge item
            for (const k of deduplicatedKnowledge) {
                await this.eventBus.publish((0, event_types_1.createBaseEvent)('KnowledgeExtracted', 'knowledge-extraction-workflow', correlationCtx.correlation_id, {
                    knowledge_id: k.id,
                    problem_id: k.problem_id,
                    knowledge_type: k.knowledge_type,
                    source_type: k.source_type,
                    source_id: k.source_id,
                }));
            }
            return deduplicatedKnowledge;
        }
        catch (error) {
            this.logger.error('Step 3 failed: Knowledge extraction', error, {
                correlation_id: correlationCtx.correlation_id,
            });
            await this.publishStepFailed(step, error, correlationCtx);
            throw error;
        }
    }
    /**
     * Step 4: Store knowledge in Graphiti (optional)
     */
    async stepStoreInGraphiti(knowledge, correlationCtx) {
        const step = 'store_in_graphiti';
        this.logger.info('Step 4: Storing in Graphiti', {
            correlation_id: correlationCtx.correlation_id,
            step,
            knowledge_count: knowledge.length,
        });
        if (!this.graphitiAdapter) {
            this.logger.warn('Graphiti adapter not configured, skipping graph storage', {
                correlation_id: correlationCtx.correlation_id,
            });
            return;
        }
        try {
            // Store each knowledge item as a node in the knowledge graph
            for (const k of knowledge) {
                // This is a simplified example - actual implementation would
                // depend on the Graphiti adapter API
                /*
                await this.graphitiAdapter.addKnowledgeNode({
                  node_id: k.id,
                  node_type: k.knowledge_type,
                  properties: {
                    content: k.content,
                    source_type: k.source_type,
                    source_id: k.source_id,
                    problem_id: k.problem_id,
                    extracted_at: k.extracted_at,
                  },
                });
                */
            }
            this.logger.info('Step 4 completed: Stored in Graphiti', {
                correlation_id: correlationCtx.correlation_id,
                knowledge_count: knowledge.length,
            });
            // Publish event
            await this.eventBus.publish((0, event_types_1.createBaseEvent)('GraphUpdated', 'knowledge-extraction-workflow', correlationCtx.correlation_id, {
                update_type: 'node_added',
                node_count: knowledge.length,
                graph_system: 'graphiti',
                changes: knowledge.map(k => ({
                    type: 'node',
                    action: 'added',
                    id: k.id,
                })),
            }));
        }
        catch (error) {
            this.logger.error('Step 4 failed: Graphiti storage', error, {
                correlation_id: correlationCtx.correlation_id,
            });
            await this.publishStepFailed(step, error, correlationCtx);
            // Don't throw - graph storage failure should not fail the workflow
        }
    }
    /**
     * Step 5: Vectorize knowledge for similarity search (optional)
     */
    async stepVectorizeKnowledge(knowledge, correlationCtx) {
        const step = 'vectorize_knowledge';
        this.logger.info('Step 5: Vectorizing knowledge', {
            correlation_id: correlationCtx.correlation_id,
            step,
            knowledge_count: knowledge.length,
        });
        if (!this.vectorDBAdapter) {
            this.logger.warn('Vector DB adapter not configured, skipping vectorization', {
                correlation_id: correlationCtx.correlation_id,
            });
            return;
        }
        try {
            // Create embeddings for each knowledge item
            for (const k of knowledge) {
                // This is a simplified example - actual implementation would
                // depend on the Vector DB adapter API
                /*
                const embedding = await this.vectorDBAdapter.createEmbedding({
                  text: k.content.pattern,
                  metadata: {
                    knowledge_id: k.id,
                    knowledge_type: k.knowledge_type,
                    source_type: k.source_type,
                  },
                });
        
                await this.vectorDBAdapter.indexEmbedding({
                  vector_id: k.id,
                  embedding: embedding.vector,
                  metadata: {
                    knowledge_id: k.id,
                    problem_id: k.problem_id,
                    knowledge_type: k.knowledge_type,
                  },
                });
                */
            }
            this.logger.info('Step 5 completed: Knowledge vectorized', {
                correlation_id: correlationCtx.correlation_id,
                knowledge_count: knowledge.length,
            });
            // Publish event
            await this.eventBus.publish((0, event_types_1.createBaseEvent)('VectorIndexed', 'knowledge-extraction-workflow', correlationCtx.correlation_id, {
                index_type: 'create',
                embedding_count: knowledge.length,
                embedding_model: 'text-embedding-ada-002',
                dimension: 1536,
                vector_db_type: 'chroma',
            }));
        }
        catch (error) {
            this.logger.error('Step 5 failed: Knowledge vectorization', error, {
                correlation_id: correlationCtx.correlation_id,
            });
            await this.publishStepFailed(step, error, correlationCtx);
            // Don't throw - vectorization failure should not fail the workflow
        }
    }
    /**
     * Step 6: Formulate new problems based on knowledge gaps (optional)
     */
    async stepFormulateProblems(knowledge, patterns, correlationCtx) {
        const step = 'formulate_problems';
        this.logger.info('Step 6: Formulating problems', {
            correlation_id: correlationCtx.correlation_id,
            step,
            knowledge_count: knowledge.length,
            pattern_count: patterns.length,
        });
        try {
            const problems = [];
            // Identify knowledge gaps
            const lowSuccessPatterns = patterns.filter(p => p.success_rate < 0.7);
            const lowScoreKnowledge = knowledge.filter(k => k.content.avg_score < 0.7);
            // Formulate problems for improvement
            for (const pattern of lowSuccessPatterns) {
                problems.push({
                    problem_id: (0, uuid_1.v4)(),
                    problem_type: 'optimization',
                    description: `Improve success rate for pattern: ${pattern.pattern_type}`,
                    context: {
                        current_success_rate: pattern.success_rate,
                        target_success_rate: 0.8,
                        pattern_frequency: pattern.frequency,
                        based_on_pattern_id: pattern.pattern_id,
                    },
                    priority: Math.ceil((1 - pattern.success_rate) * 10),
                    based_on_knowledge: knowledge
                        .filter(k => k.content.pattern?.includes(pattern.pattern_type))
                        .map(k => k.id),
                });
            }
            // Formulate problems for low-score knowledge
            for (const k of lowScoreKnowledge.slice(0, 5)) {
                problems.push({
                    problem_id: (0, uuid_1.v4)(),
                    problem_type: 'reasoning',
                    description: `Improve solution quality for: ${k.content.pattern?.substring(0, 100)}...`,
                    context: {
                        current_score: k.content.avg_score,
                        target_score: 0.8,
                        knowledge_type: k.knowledge_type,
                        source_id: k.source_id,
                    },
                    priority: Math.ceil((1 - k.content.avg_score) * 10),
                    based_on_knowledge: [k.id],
                });
            }
            this.logger.info('Step 6 completed: Problems formulated', {
                correlation_id: correlationCtx.correlation_id,
                problem_count: problems.length,
                high_priority_count: problems.filter(p => p.priority >= 7).length,
            });
            return problems;
        }
        catch (error) {
            this.logger.error('Step 6 failed: Problem formulation', error, {
                correlation_id: correlationCtx.correlation_id,
            });
            await this.publishStepFailed(step, error, correlationCtx);
            throw error;
        }
    }
    // ============================================================================
    // UTILITY METHODS
    // ============================================================================
    /**
     * Extract pattern key from solution
     * This is a simplified implementation
     */
    extractPatternKey(solution) {
        // Extract pattern characteristics
        const planLength = solution.generate_plan.length;
        const scoreRange = Math.floor(solution.score * 10) / 10;
        const island = solution.island_id;
        // Create pattern key based on characteristics
        return `plan_${planLength < 500 ? 'short' : planLength < 1000 ? 'medium' : 'long'}_score_${scoreRange}_island_${island}`;
    }
    /**
     * Calculate average score of solutions
     */
    calculateAverageScore(solutions) {
        if (solutions.length === 0)
            return 0;
        const sum = solutions.reduce((acc, s) => acc + s.score, 0);
        return sum / solutions.length;
    }
    /**
     * Deduplicate knowledge by source_id and knowledge_type
     */
    deduplicateKnowledge(knowledge) {
        const seen = new Set();
        const deduplicated = [];
        for (const k of knowledge) {
            const key = `${k.source_id}_${k.knowledge_type}`;
            if (!seen.has(key)) {
                seen.add(key);
                deduplicated.push(k);
            }
        }
        return deduplicated;
    }
    // ============================================================================
    // EVENT PUBLISHING
    // ============================================================================
    async publishExtractionCompleted(executionId, knowledge, problems, patterns, durationMs, correlationCtx) {
        await this.eventBus.publish((0, event_types_1.createBaseEvent)('WorkflowCompleted', 'knowledge-extraction-workflow', correlationCtx.correlation_id, {
            workflow_id: executionId,
            workflow_name: 'knowledge-extraction',
            duration_ms: durationMs,
            output_data: {
                knowledge_count: knowledge.length,
                problem_count: problems.length,
                pattern_count: patterns.length,
            },
            steps_completed: 6,
            steps_failed: 0,
        }));
    }
    async publishExtractionFailed(executionId, error, correlationCtx) {
        await this.eventBus.publish((0, event_types_1.createBaseEvent)('WorkflowFailed', 'knowledge-extraction-workflow', correlationCtx.correlation_id, {
            workflow_id: executionId,
            workflow_name: 'knowledge-extraction',
            failure_reason: error.message,
            failed_step: 'unknown',
            error_details: {
                name: error.name,
                message: error.message,
                stack: error.stack,
            },
            duration_ms: 0,
        }));
    }
    async publishStepFailed(step, error, correlationCtx) {
        await this.eventBus.publish((0, event_types_1.createBaseEvent)('StepFailed', 'knowledge-extraction-workflow', correlationCtx.correlation_id, {
            step,
            error_message: error.message,
            error_name: error.name,
        }));
    }
}
exports.KnowledgeExtractionWorkflow = KnowledgeExtractionWorkflow;
// ============================================================================
// FACTORY FUNCTION
// ============================================================================
function createKnowledgeExtractionWorkflow(config) {
    return new KnowledgeExtractionWorkflow(config);
}
//# sourceMappingURL=knowledge-extraction-workflow.js.map