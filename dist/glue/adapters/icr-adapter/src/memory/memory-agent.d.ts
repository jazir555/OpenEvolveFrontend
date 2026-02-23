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
import { GraphitiMemoryConfig } from './graphiti-memory';
import { EnrichedContext, RefinementInsights, ContextualSession, SessionMemory, SessionOutcome, LearningResult, PatternRelationship } from './canonical';
export interface MemoryAgentConfig {
    graphiti: GraphitiMemoryConfig;
    enable_historical_retrieval?: boolean;
    enable_pattern_learning?: boolean;
    enable_cross_session_learning?: boolean;
    default_context_window?: number;
    min_relevance_score?: number;
    max_historical_results?: number;
    learning_threshold?: number;
    pattern_extraction_min_frequency?: number;
}
export declare class EnhancedICRMemoryAgent {
    private readonly graphitiMemory;
    private readonly config;
    private readonly logger;
    private patternCache;
    private cacheTimestamp;
    private readonly CACHE_TTL_MS;
    constructor(config: MemoryAgentConfig);
    /**
     * Retrieve historical knowledge relevant to the current query
     *
     * @param query - Natural language query
     * @param contextWindow - Number of recent sessions to consider
     * @param correlationId - Optional correlation ID for tracing
     * @returns Enriched context with historical knowledge
     */
    retrieveHistoricalKnowledge(query: string, contextWindow?: number, correlationId?: string): Promise<EnrichedContext>;
    /**
     * Store refinement insights from a completed session
     *
     * @param insights - Refinement insights to store
     * @param sessionId - Session identifier
     * @param correlationId - Optional correlation ID
     * @returns Storage result
     */
    storeRefinementInsights(insights: RefinementInsights, sessionId: string, correlationId?: string): Promise<void>;
    /**
     * Store contextual session data
     *
     * @param session - Contextual session to store
     * @param correlationId - Optional correlation ID
     * @returns Storage result
     */
    storeContextualSession(session: ContextualSession, correlationId?: string): Promise<void>;
    /**
     * Get complete memory for a specific session
     *
     * @param sessionId - Session identifier
     * @param correlationId - Optional correlation ID
     * @returns Session memory if found
     */
    getContextualMemory(sessionId: string, correlationId?: string): Promise<SessionMemory | null>;
    /**
     * Learn from a completed session
     *
     * @param session - Completed contextual session
     * @param outcomes - Session outcomes to learn from
     * @param correlationId - Optional correlation ID
     * @returns Learning result with statistics
     */
    learnFromSession(session: ContextualSession, outcomes: SessionOutcome[], correlationId?: string): Promise<LearningResult>;
    /**
     * Analyze patterns across multiple sessions
     *
     * @param sessions - Sessions to analyze
     * @param correlationId - Optional correlation ID
     * @returns Pattern relationships discovered
     */
    analyzePatterns(sessions: ContextualSession[], correlationId?: string): Promise<PatternRelationship[]>;
    /**
     * Extract related patterns from historical knowledge
     */
    private extractRelatedPatterns;
    /**
     * Generate suggested approaches from historical knowledge
     */
    private generateSuggestedApproaches;
    /**
     * Identify common pitfalls from historical knowledge
     */
    private identifyCommonPitfalls;
    /**
     * Calculate success probability based on historical knowledge
     */
    private calculateSuccessProbability;
    /**
     * Calculate confidence score for enriched context
     */
    private calculateConfidenceScore;
    /**
     * Extract patterns from memory graph
     */
    private extractPatternsFromGraph;
    /**
     * Infer pattern type from pattern name
     */
    private inferPatternType;
    /**
     * Check if pattern cache is valid
     */
    private isPatternCacheValid;
    /**
     * Invalidate pattern cache
     */
    private invalidatePatternCache;
}
//# sourceMappingURL=memory-agent.d.ts.map