"use strict";
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * Enhanced ICR Memory Agent
 *
 * High-level memory management interface for ICR Contextual Mode.
 * Provides intelligent memory retrieval, storage, and learning capabilities.
 * Follows the Federation Constitution:
 * - Law of Configuration Explicitness: All config via constructor
 * - Law of UTC: All timestamps in UTC ISO-8601 format
 * - Law of Idempotency: All operations safe to replay
 * - Observability: Structured logging with correlation IDs
 *
 * Architecture:
 * ICR Adapter -> EnhancedICRMemoryAgent -> GraphitiMemoryManager -> Graphiti Adapter
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.EnhancedICRMemoryAgent = void 0;
const uuid_1 = require("uuid");
const graphiti_memory_1 = require("./graphiti-memory");
const DEFAULT_CONFIG = {
    enable_historical_retrieval: true,
    enable_pattern_learning: true,
    enable_cross_session_learning: true,
    default_context_window: 5,
    min_relevance_score: 0.3,
    max_historical_results: 20,
    learning_threshold: 0.7,
    pattern_extraction_min_frequency: 2
};
class MemoryStructuredLogger {
    constructor(sourceService = 'icr-memory-agent') {
        this.sourceService = sourceService;
    }
    log(level, entry) {
        const logEntry = {
            ...entry,
            timestamp_utc: new Date().toISOString(),
            source_service: this.sourceService
        };
        const jsonLine = JSON.stringify({ level, ...logEntry });
        console.log(jsonLine);
    }
    info(entry) {
        this.log('info', entry);
    }
    warn(entry) {
        this.log('warn', entry);
    }
    error(entry) {
        this.log('error', entry);
    }
    debug(entry) {
        if (process.env.DEBUG === 'true') {
            this.log('debug', entry);
        }
    }
}
// ============================================================================
// ENHANCED ICR MEMORY AGENT
// ============================================================================
class EnhancedICRMemoryAgent {
    constructor(config) {
        // In-memory cache for frequently accessed patterns
        this.patternCache = new Map();
        this.cacheTimestamp = Date.now();
        this.CACHE_TTL_MS = 300000; // 5 minutes
        this.config = {
            ...DEFAULT_CONFIG,
            ...config
        };
        this.graphitiMemory = new graphiti_memory_1.GraphitiMemoryManager(this.config.graphiti);
        this.logger = new MemoryStructuredLogger();
        this.logger.info({
            msg: 'ICR Memory Agent initialized',
            enable_historical_retrieval: this.config.enable_historical_retrieval,
            enable_pattern_learning: this.config.enable_pattern_learning,
            default_context_window: this.config.default_context_window
        });
    }
    // ========================================================================
    // HISTORICAL KNOWLEDGE RETRIEVAL
    // ========================================================================
    /**
     * Retrieve historical knowledge relevant to the current query
     *
     * @param query - Natural language query
     * @param contextWindow - Number of recent sessions to consider
     * @param correlationId - Optional correlation ID for tracing
     * @returns Enriched context with historical knowledge
     */
    async retrieveHistoricalKnowledge(query, contextWindow = this.config.default_context_window, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        const startTime = Date.now();
        this.logger.info({
            msg: 'Retrieving historical knowledge',
            correlation_id: cid,
            query,
            context_window: contextWindow
        });
        try {
            // Build memory query
            const memoryQuery = {
                query,
                max_results: this.config.max_historical_results,
                include_failed: false,
                correlation_id: cid
            };
            // Retrieve historical knowledge from Graphiti
            const historicalKnowledge = await this.graphitiMemory.retrieveHistoricalKnowledge(memoryQuery, contextWindow, cid);
            // Extract related patterns
            const relatedPatterns = await this.extractRelatedPatterns(historicalKnowledge, cid);
            // Generate suggested approaches
            const suggestedApproaches = this.generateSuggestedApproaches(historicalKnowledge, relatedPatterns);
            // Identify common pitfalls
            const commonPitfalls = this.identifyCommonPitfalls(historicalKnowledge);
            // Calculate success probability
            const successProbability = this.calculateSuccessProbability(historicalKnowledge);
            // Calculate overall confidence score
            const confidenceScore = this.calculateConfidenceScore(historicalKnowledge, relatedPatterns);
            const enrichedContext = {
                query,
                historical_knowledge: historicalKnowledge,
                related_patterns: relatedPatterns,
                suggested_approaches: suggestedApproaches,
                common_pitfalls: commonPitfalls,
                success_probability: successProbability,
                confidence_score: confidenceScore,
                processing_time_ms: Date.now() - startTime,
                correlation_id: cid,
                timestamp_utc: new Date().toISOString()
            };
            this.logger.info({
                msg: 'Historical knowledge retrieved successfully',
                correlation_id: cid,
                knowledge_count: historicalKnowledge.length,
                pattern_count: relatedPatterns.length,
                confidence_score: confidenceScore,
                processing_time_ms: enrichedContext.processing_time_ms
            });
            return enrichedContext;
        }
        catch (error) {
            this.logger.error({
                msg: 'Failed to retrieve historical knowledge',
                correlation_id: cid,
                error: error instanceof Error ? error.message : String(error)
            });
            // Return empty context on error
            return {
                query,
                historical_knowledge: [],
                related_patterns: [],
                suggested_approaches: [],
                common_pitfalls: [],
                confidence_score: 0,
                processing_time_ms: Date.now() - startTime,
                correlation_id: cid,
                timestamp_utc: new Date().toISOString()
            };
        }
    }
    // ========================================================================
    // MEMORY STORAGE OPERATIONS
    // ========================================================================
    /**
     * Store refinement insights from a completed session
     *
     * @param insights - Refinement insights to store
     * @param sessionId - Session identifier
     * @param correlationId - Optional correlation ID
     * @returns Storage result
     */
    async storeRefinementInsights(insights, sessionId, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        this.logger.info({
            msg: 'Storing refinement insights',
            correlation_id: cid,
            session_id: sessionId,
            iterations: insights.total_iterations,
            outcome: insights.overall_outcome
        });
        try {
            const result = await this.graphitiMemory.storeRefinementInsights(insights, sessionId, cid);
            if (result.success) {
                this.logger.info({
                    msg: 'Refinement insights stored successfully',
                    correlation_id: cid,
                    episode_id: result.episode_id,
                    entities_created: result.entities_created,
                    relationships_created: result.relationships_created
                });
                // Invalidate pattern cache to force refresh
                this.invalidatePatternCache();
            }
            else {
                this.logger.error({
                    msg: 'Failed to store refinement insights',
                    correlation_id: cid,
                    error: result.error
                });
            }
        }
        catch (error) {
            this.logger.error({
                msg: 'Error storing refinement insights',
                correlation_id: cid,
                error: error instanceof Error ? error.message : String(error)
            });
        }
    }
    /**
     * Store contextual session data
     *
     * @param session - Contextual session to store
     * @param correlationId - Optional correlation ID
     * @returns Storage result
     */
    async storeContextualSession(session, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        this.logger.info({
            msg: 'Storing contextual session',
            correlation_id: cid,
            session_id: session.session_id,
            duration_ms: session.duration_ms,
            quality_score: session.quality_score
        });
        try {
            const result = await this.graphitiMemory.storeContextualSession(session, cid);
            if (result.success) {
                this.logger.info({
                    msg: 'Contextual session stored successfully',
                    correlation_id: cid,
                    episode_id: result.episode_id,
                    entities_created: result.entities_created,
                    relationships_created: result.relationships_created
                });
                // Invalidate pattern cache
                this.invalidatePatternCache();
            }
            else {
                this.logger.error({
                    msg: 'Failed to store contextual session',
                    correlation_id: cid,
                    error: result.error
                });
            }
        }
        catch (error) {
            this.logger.error({
                msg: 'Error storing contextual session',
                correlation_id: cid,
                error: error instanceof Error ? error.message : String(error)
            });
        }
    }
    // ========================================================================
    // SESSION MEMORY RETRIEVAL
    // ========================================================================
    /**
     * Get complete memory for a specific session
     *
     * @param sessionId - Session identifier
     * @param correlationId - Optional correlation ID
     * @returns Session memory if found
     */
    async getContextualMemory(sessionId, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        this.logger.info({
            msg: 'Retrieving contextual memory',
            correlation_id: cid,
            session_id: sessionId
        });
        try {
            const session = await this.graphitiMemory.retrieveContextualMemory(sessionId, cid);
            if (!session) {
                this.logger.warn({
                    msg: 'Session not found in memory',
                    correlation_id: cid,
                    session_id: sessionId
                });
                return null;
            }
            // Build session memory object
            const sessionMemory = {
                session_id: sessionId,
                session,
                created_at: new Date().toISOString()
            };
            this.logger.info({
                msg: 'Contextual memory retrieved successfully',
                correlation_id: cid,
                session_id: sessionId
            });
            return sessionMemory;
        }
        catch (error) {
            this.logger.error({
                msg: 'Error retrieving contextual memory',
                correlation_id: cid,
                session_id: sessionId,
                error: error instanceof Error ? error.message : String(error)
            });
            return null;
        }
    }
    // ========================================================================
    // LEARNING OPERATIONS
    // ========================================================================
    /**
     * Learn from a completed session
     *
     * @param session - Completed contextual session
     * @param outcomes - Session outcomes to learn from
     * @param correlationId - Optional correlation ID
     * @returns Learning result with statistics
     */
    async learnFromSession(session, outcomes, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        this.logger.info({
            msg: 'Learning from session',
            correlation_id: cid,
            session_id: session.session_id,
            outcomes_count: outcomes.length
        });
        if (!this.config.enable_pattern_learning) {
            this.logger.debug({
                msg: 'Pattern learning disabled',
                correlation_id: cid
            });
            return {
                success: true,
                patterns_learned: 0,
                patterns_updated: 0,
                new_relationships: 0,
                insights_extracted: 0,
                processing_time_ms: 0,
                correlation_id: cid
            };
        }
        try {
            const result = await this.graphitiMemory.learnFromSession(session, outcomes, cid);
            if (result.success) {
                this.logger.info({
                    msg: 'Learning completed successfully',
                    correlation_id: cid,
                    patterns_learned: result.patterns_learned,
                    patterns_updated: result.patterns_updated,
                    new_relationships: result.new_relationships,
                    insights_extracted: result.insights_extracted,
                    confidence_score: result.confidence_score
                });
                // Invalidate pattern cache after learning
                this.invalidatePatternCache();
            }
            return result;
        }
        catch (error) {
            this.logger.error({
                msg: 'Error learning from session',
                correlation_id: cid,
                error: error instanceof Error ? error.message : String(error)
            });
            return {
                success: false,
                patterns_learned: 0,
                patterns_updated: 0,
                new_relationships: 0,
                insights_extracted: 0,
                processing_time_ms: 0,
                error: error instanceof Error ? error.message : String(error),
                correlation_id: cid
            };
        }
    }
    // ========================================================================
    // PATTERN ANALYSIS
    // ========================================================================
    /**
     * Analyze patterns across multiple sessions
     *
     * @param sessions - Sessions to analyze
     * @param correlationId - Optional correlation ID
     * @returns Pattern relationships discovered
     */
    async analyzePatterns(sessions, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        this.logger.info({
            msg: 'Analyzing patterns across sessions',
            correlation_id: cid,
            session_count: sessions.length
        });
        try {
            // Build contextual graph to extract patterns
            const graph = await this.graphitiMemory.buildContextualGraph(sessions, cid);
            // Extract pattern relationships from graph
            const patterns = this.extractPatternsFromGraph(graph);
            this.logger.info({
                msg: 'Pattern analysis completed',
                correlation_id: cid,
                patterns_discovered: patterns.length
            });
            return patterns;
        }
        catch (error) {
            this.logger.error({
                msg: 'Error analyzing patterns',
                correlation_id: cid,
                error: error instanceof Error ? error.message : String(error)
            });
            return [];
        }
    }
    // ========================================================================
    // PRIVATE HELPER METHODS
    // ========================================================================
    /**
     * Extract related patterns from historical knowledge
     */
    async extractRelatedPatterns(knowledge, correlationId) {
        // Check cache first
        if (this.isPatternCacheValid()) {
            return this.patternCache.get('default') || [];
        }
        const patterns = new Map();
        // Extract patterns from knowledge items
        for (const item of knowledge) {
            for (const pattern of item.applicable_patterns) {
                const patternId = `pattern-${pattern.replace(/\s+/g, '-').toLowerCase()}`;
                if (!patterns.has(patternId)) {
                    patterns.set(patternId, {
                        pattern_id: patternId,
                        pattern_type: this.inferPatternType(pattern),
                        pattern_name: pattern,
                        description: `Discovered pattern: ${pattern}`,
                        related_sessions: [item.session_id],
                        success_rate: item.outcome === 'success' ? 1.0 : 0.5,
                        avg_execution_time_ms: 0,
                        frequency: 1,
                        last_seen_utc: item.timestamp_utc,
                        first_seen_utc: item.timestamp_utc
                    });
                }
                else {
                    const existing = patterns.get(patternId);
                    existing.related_sessions.push(item.session_id);
                    existing.frequency++;
                    existing.last_seen_utc = item.timestamp_utc;
                    // Update success rate
                    const outcomeWeight = item.outcome === 'success' ? 1.0 :
                        item.outcome === 'partial_success' ? 0.5 : 0.0;
                    existing.success_rate = (existing.success_rate + outcomeWeight) / 2;
                }
            }
        }
        const result = Array.from(patterns.values());
        // Update cache
        this.patternCache.set('default', result);
        this.cacheTimestamp = Date.now();
        return result;
    }
    /**
     * Generate suggested approaches from historical knowledge
     */
    generateSuggestedApproaches(knowledge, patterns) {
        const approaches = new Set();
        // Extract approaches from successful knowledge items
        for (const item of knowledge) {
            if (item.outcome === 'success') {
                for (const insight of item.insights) {
                    if (insight.length > 20) { // Filter out very short insights
                        approaches.add(insight);
                    }
                }
            }
        }
        // Add approaches from high-success patterns
        for (const pattern of patterns) {
            if (pattern.success_rate >= 0.8) {
                approaches.add(`Apply pattern: ${pattern.pattern_name}`);
                if (pattern.description) {
                    approaches.add(pattern.description);
                }
            }
        }
        return Array.from(approaches).slice(0, 10); // Limit to top 10
    }
    /**
     * Identify common pitfalls from historical knowledge
     */
    identifyCommonPitfalls(knowledge) {
        const pitfalls = new Map();
        // Extract pitfalls from failed/partially successful knowledge
        for (const item of knowledge) {
            if (item.outcome !== 'success') {
                for (const insight of item.insights) {
                    // Look for negative patterns
                    const lowerInsight = insight.toLowerCase();
                    if (lowerInsight.includes('fail') ||
                        lowerInsight.includes('error') ||
                        lowerInsight.includes('avoid') ||
                        lowerInsight.includes('pitfall')) {
                        pitfalls.set(insight, (pitfalls.get(insight) || 0) + 1);
                    }
                }
            }
        }
        // Sort by frequency and return top pitfalls
        return Array.from(pitfalls.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
            .map(([pitfall]) => pitfall);
    }
    /**
     * Calculate success probability based on historical knowledge
     */
    calculateSuccessProbability(knowledge) {
        if (knowledge.length === 0) {
            return 0.5; // Neutral probability
        }
        let totalWeight = 0;
        let successWeight = 0;
        for (const item of knowledge) {
            const weight = item.relevance_score;
            const outcomeWeight = item.outcome === 'success' ? 1.0 :
                item.outcome === 'partial_success' ? 0.5 : 0.0;
            totalWeight += weight;
            successWeight += weight * outcomeWeight;
        }
        return totalWeight > 0 ? successWeight / totalWeight : 0.5;
    }
    /**
     * Calculate confidence score for enriched context
     */
    calculateConfidenceScore(knowledge, patterns) {
        if (knowledge.length === 0 && patterns.length === 0) {
            return 0.0;
        }
        // Confidence based on:
        // 1. Amount of relevant knowledge
        // 2. Pattern success rates
        // 3. Knowledge relevance scores
        const knowledgeScore = Math.min(1.0, knowledge.length / 10);
        const patternScore = patterns.length > 0
            ? patterns.reduce((sum, p) => sum + p.success_rate, 0) / patterns.length
            : 0.5;
        const avgRelevance = knowledge.length > 0
            ? knowledge.reduce((sum, k) => sum + k.relevance_score, 0) / knowledge.length
            : 0.0;
        // Weighted average
        return (knowledgeScore * 0.4) + (patternScore * 0.4) + (avgRelevance * 0.2);
    }
    /**
     * Extract patterns from memory graph
     */
    extractPatternsFromGraph(graph) {
        const patterns = [];
        for (const node of graph.nodes) {
            if (node.type === 'pattern') {
                // Find edges connected to this pattern
                const connectedEdges = graph.edges.filter(e => e.source_id === node.id || e.target_id === node.id);
                patterns.push({
                    pattern_id: node.id,
                    pattern_type: this.inferPatternType(node.name),
                    pattern_name: node.name,
                    description: node.description,
                    related_sessions: connectedEdges.map(e => e.source_id === node.id ? e.target_id : e.source_id),
                    success_rate: connectedEdges.length > 0
                        ? connectedEdges.reduce((sum, e) => sum + (e.weight || 0.5), 0) / connectedEdges.length
                        : 0.5,
                    avg_execution_time_ms: 0,
                    frequency: connectedEdges.length,
                    last_seen_utc: node.updated_at || node.created_at,
                    first_seen_utc: node.created_at,
                    metadata: node.attributes
                });
            }
        }
        return patterns;
    }
    /**
     * Infer pattern type from pattern name
     */
    inferPatternType(patternName) {
        const name = patternName.toLowerCase();
        if (name.includes('refinement') || name.includes('iteration')) {
            return 'iterative_refinement';
        }
        else if (name.includes('agent') || name.includes('collaboration')) {
            return 'agent_collaboration';
        }
        else if (name.includes('memory') || name.includes('compression')) {
            return 'memory_compression';
        }
        else if (name.includes('quality') || name.includes('improvement')) {
            return 'quality_improvement';
        }
        else if (name.includes('novelty') || name.includes('generative')) {
            return 'novelty_generation';
        }
        else {
            return 'custom';
        }
    }
    /**
     * Check if pattern cache is valid
     */
    isPatternCacheValid() {
        return Date.now() - this.cacheTimestamp < this.CACHE_TTL_MS;
    }
    /**
     * Invalidate pattern cache
     */
    invalidatePatternCache() {
        this.patternCache.clear();
        this.cacheTimestamp = 0;
    }
}
exports.EnhancedICRMemoryAgent = EnhancedICRMemoryAgent;
//# sourceMappingURL=memory-agent.js.map