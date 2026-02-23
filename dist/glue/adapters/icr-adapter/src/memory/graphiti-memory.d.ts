/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * Graphiti Memory Manager
 *
 * Manages persistent memory storage and retrieval using Graphiti temporal knowledge graph.
 * Follows the Federation Constitution:
 * - Law of Configuration Explicitness: All config via environment variables
 * - Law of UTC: All timestamps in UTC ISO-8601 format
 * - Law of Idempotency: All storage operations safe to replay
 * - Law of Runtime Truth: Verify Graphiti connection before use
 *
 * Architecture:
 * ICR Memory Agent -> GraphitiMemoryManager -> Graphiti Adapter -> Neo4j
 */
import { RefinementInsights, ContextualSession, MemoryQuery, HistoricalKnowledge, MemoryGraph, StorageResult, LearningResult, SessionOutcome } from './canonical';
import { AddEpisodeOperation, AddEpisodeResult, CanonicalSearchResult, EpisodeType } from '../../../schemas/graphiti-canonical';
/**
 * Interface for Graphiti adapter
 * Defined here to avoid direct import from graphiti-adapter (Air Gap compliance)
 */
export interface IGraphitiAdapter {
    isInitialized(): boolean;
    addEpisode(operation: AddEpisodeOperation, correlationId?: string): Promise<AddEpisodeResult>;
    search(query: string, options?: any): Promise<CanonicalSearchResult>;
    getStatistics(): Promise<any>;
}
export interface GraphitiMemoryConfig {
    graphitiAdapter: IGraphitiAdapter;
    default_context_window?: number;
    max_historical_results?: number;
    enable_pattern_learning?: boolean;
    enable_cross_session_learning?: boolean;
    default_episode_type?: EpisodeType;
    update_communities?: boolean;
}
export declare class GraphitiMemoryManager {
    private readonly config;
    constructor(config: GraphitiMemoryConfig);
    /**
     * Store refinement insights from a session into Graphiti
     *
     * @param insights - Refinement insights to store
     * @param sessionId - Session identifier
     * @param correlationId - Optional correlation ID for tracing
     * @returns Storage result with episode ID and statistics
     */
    storeRefinementInsights(insights: RefinementInsights, sessionId: string, correlationId?: string): Promise<StorageResult>;
    /**
     * Store contextual session data into Graphiti
     *
     * @param session - Contextual session to store
     * @param correlationId - Optional correlation ID
     * @returns Storage result
     */
    storeContextualSession(session: ContextualSession, correlationId?: string): Promise<StorageResult>;
    /**
     * Retrieve historical knowledge based on a query
     *
     * @param query - Memory query parameters
     * @param contextWindow - Number of recent sessions to consider
     * @param correlationId - Optional correlation ID
     * @returns Array of historical knowledge items
     */
    retrieveHistoricalKnowledge(query: MemoryQuery, contextWindow: number, correlationId?: string): Promise<HistoricalKnowledge[]>;
    /**
     * Retrieve contextual memory for a specific session
     *
     * @param sessionId - Session identifier
     * @param correlationId - Optional correlation ID
     * @returns Contextual session if found
     */
    retrieveContextualMemory(sessionId: string, correlationId?: string): Promise<ContextualSession | null>;
    /**
     * Build a contextual memory graph from multiple sessions
     *
     * @param sessions - Array of contextual sessions
     * @param correlationId - Optional correlation ID
     * @returns Memory graph with nodes and edges
     */
    buildContextualGraph(sessions: ContextualSession[], correlationId?: string): Promise<MemoryGraph>;
    /**
     * Learn from session outcomes and update patterns
     *
     * @param session - Completed session
     * @param outcomes - Session outcomes
     * @param correlationId - Optional correlation ID
     * @returns Learning result with statistics
     */
    learnFromSession(session: ContextualSession, outcomes: SessionOutcome[], correlationId?: string): Promise<LearningResult>;
    /**
     * Format refinement insights as episode content
     */
    private formatInsightsAsEpisode;
    /**
     * Format contextual session as episode content
     */
    private formatSessionAsEpisode;
    /**
     * Build search query from MemoryQuery
     */
    private buildSearchQuery;
    /**
     * Transform Graphiti edge to HistoricalKnowledge
     */
    private transformEdgeToKnowledge;
    /**
     * Calculate relevance score for knowledge item
     */
    private calculateRelevance;
    /**
     * Filter knowledge based on query parameters
     */
    private filterKnowledge;
    /**
     * Extract session data from Graphiti edge
     */
    private extractSessionFromEdge;
    /**
     * Extract patterns from a session
     */
    private extractPatternsFromSession;
    /**
     * Extract n-grams from text
     */
    private extractNgrams;
    /**
     * Check if a phrase is a pattern candidate
     */
    private isPatternCandidate;
    /**
     * Build pattern relationships from sessions
     */
    private buildPatternRelationships;
    /**
     * Infer pattern type from pattern name
     */
    private inferPatternType;
    /**
     * Store pattern insight as episode
     */
    private storePatternInsight;
    /**
     * Store lesson learned as episode
     */
    private storeLessonLearned;
}
//# sourceMappingURL=graphiti-memory.d.ts.map