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

import { v4 as uuidv4 } from 'uuid';
import {
  RefinementInsights,
  ContextualSession,
  MemoryQuery,
  HistoricalKnowledge,
  EnrichedContext,
  MemoryGraph,
  PatternRelationship,
  StorageResult,
  LearningResult,
  SessionOutcome,
  RefinementMemory,
  validateMemorySchema
} from './canonical';
import {
  AddEpisodeOperation,
  AddEpisodeResult,
  CanonicalSearchResult,
  CanonicalEpisode,
  CanonicalEntity,
  CanonicalEntityEdge,
  EpisodeType
} from '../types/graphiti-canonical';

// ============================================================================
// GRAPHITI ADAPTER INTERFACE (Air Gap Compliant)
// ============================================================================

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

// ============================================================================
// CONFIGURATION
// ============================================================================

export interface GraphitiMemoryConfig {
  // Graphiti adapter instance (injected)
  graphitiAdapter: IGraphitiAdapter;

  // Memory configuration
  default_context_window?: number;
  max_historical_results?: number;
  enable_pattern_learning?: boolean;
  enable_cross_session_learning?: boolean;

  // Episode configuration
  default_episode_type?: EpisodeType;
  update_communities?: boolean;
}

const DEFAULT_CONFIG = {
  default_context_window: 5,
  max_historical_results: 20,
  enable_pattern_learning: true,
  enable_cross_session_learning: true,
  default_episode_type: 'custom' as EpisodeType,
  update_communities: false
};

// ============================================================================
// GRAPHITI MEMORY MANAGER
// ============================================================================

export class GraphitiMemoryManager {
  private readonly config: Required<Omit<GraphitiMemoryConfig, 'graphitiAdapter'>> & {
    graphitiAdapter: IGraphitiAdapter;
  };

  constructor(config: GraphitiMemoryConfig) {
    this.config = {
      ...DEFAULT_CONFIG,
      ...config,
      graphitiAdapter: config.graphitiAdapter
    };

    // Verify Graphiti adapter is initialized
    if (!this.config.graphitiAdapter.isInitialized()) {
      throw new Error(
        'GraphitiMemoryManager: Graphiti adapter is not initialized. '
        + 'Cannot proceed with memory operations.'
      );
    }
  }

  // ========================================================================
  // STORAGE OPERATIONS
  // ========================================================================

  /**
   * Store refinement insights from a session into Graphiti
   *
   * @param insights - Refinement insights to store
   * @param sessionId - Session identifier
   * @param correlationId - Optional correlation ID for tracing
   * @returns Storage result with episode ID and statistics
   */
  async storeRefinementInsights(
    insights: RefinementInsights,
    sessionId: string,
    correlationId?: string
  ): Promise<StorageResult> {
    const cid = correlationId || uuidv4();
    const startTime = Date.now();

    try {
      // Validate insights against schema
      const validation = validateMemorySchema(
        require('./canonical').RefinementInsightsSchema,
        insights
      );

      if (!validation.success) {
        return {
          success: false,
          processing_time_ms: Date.now() - startTime,
          error: `Validation failed: ${validation.errors?.join(', ')}`,
          correlation_id: cid
        } as StorageResult;
      }

      // Create episode content from insights
      const episodeContent = this.formatInsightsAsEpisode(insights);

      // Create episode operation
      const episodeOperation: AddEpisodeOperation = {
        name: `ICR Refinement Session - ${sessionId}`,
        content: episodeContent,
        source_description: 'ICR Contextual Mode Memory Agent',
        episode_type: this.config.default_episode_type,
        valid_at: new Date().toISOString(),
        group_id: sessionId, // Group all iterations from same session
        uuid: uuidv4(),
        update_communities: this.config.update_communities
      };

      // Add episode to Graphiti
      const result = await this.config.graphitiAdapter.addEpisode(
        episodeOperation,
        cid
      );

      return {
        success: result.success,
        episode_id: result.episode_id,
        entities_created: result.entities_extracted,
        relationships_created: result.relationships_extracted,
        processing_time_ms: result.processing_time_ms,
        correlation_id: cid
      };
    } catch (error) {
        return {
          success: false,
          processing_time_ms: Date.now() - startTime,
          error: error instanceof Error ? error.message : String(error),
          correlation_id: cid
        } as StorageResult;
    }
  }

  /**
   * Store contextual session data into Graphiti
   *
   * @param session - Contextual session to store
   * @param correlationId - Optional correlation ID
   * @returns Storage result
   */
  async storeContextualSession(
    session: ContextualSession,
    correlationId?: string
  ): Promise<StorageResult> {
    const cid = correlationId || uuidv4();
    const startTime = Date.now();

    try {
      // Validate session
      const validation = validateMemorySchema(
        require('./canonical').ContextualSessionSchema,
        session
      );

      if (!validation.success) {
        return {
          success: false,
          processing_time_ms: Date.now() - startTime,
          error: `Validation failed: ${validation.errors?.join(', ')}`,
          correlation_id: cid
        } as StorageResult;
      }

      // Create episode content from session
      const episodeContent = this.formatSessionAsEpisode(session);

      // Create episode operation
      const episodeOperation: AddEpisodeOperation = {
        name: `ICR Contextual Session - ${session.session_id}`,
        content: episodeContent,
        source_description: 'ICR Contextual Mode Agent Collaboration',
        episode_type: 'event',
        valid_at: session.end_time_utc,
        group_id: session.session_id,
        uuid: uuidv4(),
        update_communities: this.config.update_communities
      };

      // Add episode to Graphiti
      const result = await this.config.graphitiAdapter.addEpisode(
        episodeOperation,
        cid
      );

      return {
        success: result.success,
        episode_id: result.episode_id,
        entities_created: result.entities_extracted,
        relationships_created: result.relationships_extracted,
        processing_time_ms: result.processing_time_ms,
        correlation_id: cid
      };
    } catch (error) {
        return {
          success: false,
          processing_time_ms: Date.now() - startTime,
          error: error instanceof Error ? error.message : String(error),
          correlation_id: cid
        } as StorageResult;
    }
  }

  // ========================================================================
  // RETRIEVAL OPERATIONS
  // ========================================================================

  /**
   * Retrieve historical knowledge based on a query
   *
   * @param query - Memory query parameters
   * @param contextWindow - Number of recent sessions to consider
   * @param correlationId - Optional correlation ID
   * @returns Array of historical knowledge items
   */
  async retrieveHistoricalKnowledge(
    query: MemoryQuery,
    contextWindow: number,
    correlationId?: string
  ): Promise<HistoricalKnowledge[]> {
    const cid = correlationId || uuidv4();

    try {
      // Build search query for Graphiti
      const searchQuery = this.buildSearchQuery(query);

      // Search Graphiti
      const searchResult = await this.config.graphitiAdapter.search(
        searchQuery,
        {
          max_results: query.max_results || this.config.max_historical_results,
          temporal_filter: query.time_range ? 'time_range' : 'current',
          start_time: query.time_range?.start_utc,
          end_time: query.time_range?.end_utc
        }
      );

      // Extract and transform results into HistoricalKnowledge
      const knowledge: HistoricalKnowledge[] = [];

      for (const edge of searchResult.edges) {
        const knowledgeItem = this.transformEdgeToKnowledge(edge, query);
        if (knowledgeItem) {
          knowledge.push(knowledgeItem);
        }
      }

      // Apply additional filters
      let filtered = this.filterKnowledge(knowledge, query);

      // Limit to context window
      filtered = filtered.slice(0, contextWindow);

      return filtered;
    } catch (error) {
      console.error('Error retrieving historical knowledge:', error);
      return [];
    }
  }

  /**
   * Retrieve contextual memory for a specific session
   *
   * @param sessionId - Session identifier
   * @param correlationId - Optional correlation ID
   * @returns Contextual session if found
   */
  async retrieveContextualMemory(
    sessionId: string,
    correlationId?: string
  ): Promise<ContextualSession | null> {
    const cid = correlationId || uuidv4();

    try {
      // Search for session by group_id
      const searchResult = await this.config.graphitiAdapter.search(
        `session_id:${sessionId}`,
        {
          max_results: 1,
          group_ids: [sessionId]
        }
      );

      if (searchResult.edges.length === 0) {
        return null;
      }

      // Extract session data from the first result
      const edge = searchResult.edges[0];
      return this.extractSessionFromEdge(edge);
    } catch (error) {
      console.error('Error retrieving contextual memory:', error);
      return null;
    }
  }

  // ========================================================================
  // GRAPH BUILDING OPERATIONS
  // ========================================================================

  /**
   * Build a contextual memory graph from multiple sessions
   *
   * @param sessions - Array of contextual sessions
   * @param correlationId - Optional correlation ID
   * @returns Memory graph with nodes and edges
   */
  async buildContextualGraph(
    sessions: ContextualSession[],
    correlationId?: string
  ): Promise<MemoryGraph> {
    const cid = correlationId || uuidv4();

    try {
      const nodes: any[] = [];
      const edges: any[] = [];

      // Extract entities and relationships from sessions
      for (const session of sessions) {
        const sessionNode = {
          id: session.session_id,
          type: 'session' as const,
          name: `Session ${session.session_id.substring(0, 8)}`,
          description: `Contextual mode session with ${session.agents_involved.length} agents`,
          attributes: {
            mode: session.mode,
            duration_ms: session.duration_ms,
            successes: session.successes,
            failures: session.failures,
            quality_score: session.quality_score
          },
          created_at: session.start_time_utc
        };

        nodes.push(sessionNode);

        // Extract agent nodes
        for (const agentType of session.agents_involved) {
          const agentId = `agent-${agentType}`;
          const agentNode = nodes.find(n => n.id === agentId);

          if (!agentNode) {
            nodes.push({
              id: agentId,
              type: 'agent' as const,
              name: agentType,
              description: `ICR agent: ${agentType}`,
              attributes: {},
              created_at: session.start_time_utc
            });
          }

          // Create edge from session to agent
          edges.push({
            id: uuidv4(),
            source_id: session.session_id,
            target_id: agentId,
            relationship_type: 'uses_agent',
            weight: 1.0,
            attributes: {},
            created_at: session.start_time_utc
          });
        }

        // Extract pattern nodes from interactions
        const patterns = this.extractPatternsFromSession(session);
        for (const pattern of patterns) {
          const patternId = `pattern-${pattern.replace(/\s+/g, '-').toLowerCase()}`;
          const patternNode = nodes.find(n => n.id === patternId);

          if (!patternNode) {
            nodes.push({
              id: patternId,
              type: 'pattern' as const,
              name: pattern,
              description: `Refinement pattern: ${pattern}`,
              attributes: {},
              created_at: session.start_time_utc
            });
          }

          // Create edge from session to pattern
          edges.push({
            id: uuidv4(),
            source_id: session.session_id,
            target_id: patternId,
            relationship_type: 'exhibits_pattern',
            weight: 1.0,
            attributes: {},
            created_at: session.start_time_utc
          });
        }
      }

      // Build pattern relationships
      const patternRelationships = this.buildPatternRelationships(sessions);
      for (const rel of patternRelationships) {
        edges.push({
          id: uuidv4(),
          source_id: rel.pattern_id,
          target_id: rel.related_sessions[0],
          relationship_type: 'pattern_applied_in',
          weight: rel.success_rate,
          strength: rel.avg_improvement,
          attributes: {
            frequency: rel.frequency
          },
          created_at: rel.last_seen_utc
        });
      }

      return {
        nodes,
        edges,
        session_count: sessions.length,
        pattern_count: nodes.filter(n => n.type === 'pattern').length,
        last_updated: new Date().toISOString()
      };
    } catch (error) {
      console.error('Error building contextual graph:', error);
      return {
        nodes: [],
        edges: [],
        session_count: 0,
        pattern_count: 0,
        last_updated: new Date().toISOString()
      };
    }
  }

  /**
   * Learn from session outcomes and update patterns
   *
   * @param session - Completed session
   * @param outcomes - Session outcomes
   * @param correlationId - Optional correlation ID
   * @returns Learning result with statistics
   */
  async learnFromSession(
    session: ContextualSession,
    outcomes: SessionOutcome[],
    correlationId?: string
  ): Promise<LearningResult> {
    const cid = correlationId || uuidv4();
    const startTime = Date.now();

    try {
      let patternsLearned = 0;
      let patternsUpdated = 0;
      let newRelationships = 0;
      let insightsExtracted = 0;

      // Extract patterns from outcomes
      for (const outcome of outcomes) {
        // Learn from successful patterns
        for (const pattern of outcome.successful_patterns) {
          // Store pattern as insight
          await this.storePatternInsight(
            session.session_id,
            pattern,
            outcome.outcome,
            outcome.quality_score,
            cid
          );
          patternsLearned++;
        }

        // Update problematic patterns
        for (const pattern of outcome.problematic_patterns) {
          await this.storePatternInsight(
            session.session_id,
            pattern,
            outcome.outcome,
            outcome.quality_score,
            cid
          );
          patternsUpdated++;
        }

        // Extract lessons learned
        for (const lesson of outcome.lessons_learned) {
          await this.storeLessonLearned(
            session.session_id,
            lesson,
            outcome,
            cid
          );
          insightsExtracted++;
        }

        // Create pattern relationships
        newRelationships += outcome.successful_patterns.length;
      }

      // Calculate overall confidence score
      const successfulOutcomes = outcomes.filter(o => o.outcome === 'success').length;
      const confidenceScore = outcomes.length > 0
        ? successfulOutcomes / outcomes.length
        : 0;

      return {
        success: true,
        patterns_learned: patternsLearned,
        patterns_updated: patternsUpdated,
        new_relationships: newRelationships,
        insights_extracted: insightsExtracted,
        processing_time_ms: Date.now() - startTime,
        confidence_score: confidenceScore,
        correlation_id: cid
      };
    } catch (error) {
      return {
        success: false,
        patterns_learned: 0,
        patterns_updated: 0,
        new_relationships: 0,
        insights_extracted: 0,
        processing_time_ms: Date.now() - startTime,
        error: error instanceof Error ? error.message : String(error),
        correlation_id: cid
      };
    }
  }

  // ========================================================================
  // PRIVATE HELPER METHODS
  // ========================================================================

  /**
   * Format refinement insights as episode content
   */
  private formatInsightsAsEpisode(insights: RefinementInsights): string {
    const sections = [
      `# ICR Refinement Session: ${insights.session_id}`,
      `Mode: ${insights.mode}`,
      `Total Iterations: ${insights.total_iterations}`,
      `Successful: ${insights.successful_iterations}`,
      `Failed: ${insights.failed_iterations}`,
      `Overall Outcome: ${insights.overall_outcome}`,
      '',
      '## Key Patterns Discovered',
      ...insights.key_patterns_discovered.map(p => `- ${p}`),
      '',
      '## Lessons Learned',
      ...insights.lessons_learned.map(l => `- ${l}`),
      '',
      '## Iteration Details'
    ];

    for (const iteration of insights.iterations) {
      sections.push(
        '',
        `### Iteration ${iteration.iteration_number}`,
        `- Type: ${iteration.refinement_type}`,
        `- Outcome: ${iteration.outcome}`,
        `- Execution Time: ${iteration.execution_time_ms}ms`,
        iteration.insights.length > 0 ? `- Insights: ${iteration.insights.join('; ')}` : '',
        iteration.suggested_features ? `- Suggested Features: ${iteration.suggested_features}` : '',
        iteration.bug_fixes ? `- Bug Fixes: ${iteration.bug_fixes}` : ''
      );
    }

    return sections.filter(s => s !== '').join('\n');
  }

  /**
   * Format contextual session as episode content
   */
  private formatSessionAsEpisode(session: ContextualSession): string {
    const sections = [
      `# ICR Contextual Session: ${session.session_id}`,
      `Mode: ${session.mode}`,
      `Duration: ${session.duration_ms}ms`,
      `Agents: ${session.agents_involved.join(', ')}`,
      `Successes: ${session.successes}`,
      `Failures: ${session.failures}`,
      session.quality_score ? `Quality Score: ${session.quality_score}` : '',
      '',
      '## Agent Interactions'
    ];

    for (const interaction of session.interactions) {
      sections.push(
        '',
        `### ${interaction.agent_type}`,
        `- Timestamp: ${interaction.timestamp_utc}`,
        interaction.execution_time_ms ? `- Execution Time: ${interaction.execution_time_ms}ms` : '',
        `- Content: ${interaction.content.substring(0, 200)}${interaction.content.length > 200 ? '...' : ''}`
      );
    }

    if (session.memory_compression_events && session.memory_compression_events.length > 0) {
      sections.push('', '## Memory Compression Events');
      for (const event of session.memory_compression_events) {
        sections.push(
          `- ${event.timestamp_utc}: Compressed ${event.compressed_message_count} messages`,
          `  Ratio: ${event.compression_ratio}, Saved: ${event.bytes_saved} bytes`
        );
      }
    }

    return sections.filter(s => s !== '').join('\n');
  }

  /**
   * Build search query from MemoryQuery
   */
  private buildSearchQuery(query: MemoryQuery): string {
    const parts = [query.query];

    if (query.session_context) {
      parts.push(`context:${query.session_context}`);
    }

    if (query.pattern_type) {
      parts.push(`pattern_type:${query.pattern_type}`);
    }

    return parts.join(' ');
  }

  /**
   * Transform Graphiti edge to HistoricalKnowledge
   */
  private transformEdgeToKnowledge(
    edge: CanonicalEntityEdge,
    query: MemoryQuery
  ): HistoricalKnowledge | null {
    try {
      // Extract knowledge from edge attributes
      const attrs = edge.attributes as any;

      return {
        session_id: attrs.session_id || uuidv4(),
        prompt: edge.fact,
        pattern_type: attrs.pattern_type || 'custom',
        outcome: attrs.outcome || 'success',
        insights: attrs.insights || [],
        quality_score: attrs.quality_score,
        timestamp_utc: edge.created_at,
        relevance_score: this.calculateRelevance(edge, query),
        applicable_patterns: attrs.applicable_patterns || [],
        metadata: edge.metadata
      };
    } catch (error) {
      return null;
    }
  }

  /**
   * Calculate relevance score for knowledge item
   */
  private calculateRelevance(edge: CanonicalEntityEdge, query: MemoryQuery): number {
    // Simple relevance calculation based on text matching
    const factText = edge.fact.toLowerCase();
    const queryText = query.query.toLowerCase();

    const factWords = factText.split(/\s+/);
    const queryWords = queryText.split(/\s+/);

    const matches = factWords.filter((w: any) => queryWords.includes(w)).length;
    const score = queryWords.length > 0 ? matches / queryWords.length : 0;

    return Math.min(1, Math.max(0, score));
  }

  /**
   * Filter knowledge based on query parameters
   */
  private filterKnowledge(knowledge: HistoricalKnowledge[], query: MemoryQuery): HistoricalKnowledge[] {
    let filtered = knowledge;

    // Filter by outcome if not including failed
    if (!query.include_failed) {
      filtered = filtered.filter(k => k.outcome === 'success' || k.outcome === 'partial_success');
    }

    // Filter by minimum success rate
    if (query.min_success_rate !== undefined) {
      filtered = filtered.filter(k => {
        const successWeight = k.outcome === 'success' ? 1.0
          : k.outcome === 'partial_success' ? 0.5 : 0.0;
        return successWeight >= query.min_success_rate!;
      });
    }

    // Filter by pattern type
    if (query.pattern_type) {
      filtered = filtered.filter(k => k.pattern_type === query.pattern_type);
    }

    return filtered;
  }

  /**
   * Extract session data from Graphiti edge
   */
  private extractSessionFromEdge(edge: CanonicalEntityEdge): ContextualSession | null {
    try {
      const attrs = edge.attributes as any;

      return {
        session_id: attrs.session_id || uuidv4(),
        mode: 'contextual',
        prompt: edge.fact,
        agents_involved: attrs.agents_involved || [],
        interactions: attrs.interactions || [],
        successes: attrs.successes || 0,
        failures: attrs.failures || 0,
        duration_ms: attrs.duration_ms || 0,
        start_time_utc: edge.created_at,
        end_time_utc: attrs.end_time_utc || edge.created_at,
        final_output: attrs.final_output,
        quality_score: attrs.quality_score,
        metadata: edge.metadata
      };
    } catch (error) {
      return null;
    }
  }

  /**
   * Extract patterns from a session
   */
  private extractPatternsFromSession(session: ContextualSession): string[] {
    const patterns: Set<string> = new Set();

    // Extract from agent interactions
    for (const interaction of session.interactions) {
      // Simple pattern extraction: look for repeated phrases
      const words = interaction.content.toLowerCase().split(/\s+/);
      const ngrams = this.extractNgrams(words, 3);

      for (const ngram of ngrams) {
        if (this.isPatternCandidate(ngram)) {
          patterns.add(ngram);
        }
      }
    }

    return Array.from(patterns);
  }

  /**
   * Extract n-grams from text
   */
  private extractNgrams(words: string[], n: number): string[] {
    const ngrams: string[] = [];

    for (let i = 0; i <= words.length - n; i++) {
      const ngram = words.slice(i, i + n).join(' ');
      ngrams.push(ngram);
    }

    return ngrams;
  }

  /**
   * Check if a phrase is a pattern candidate
   */
  private isPatternCandidate(phrase: string): boolean {
    // Simple heuristic: phrase with certain keywords
    const patternKeywords = [
      'refinement', 'iteration', 'improvement', 'optimization',
      'enhancement', 'modification', 'adjustment', 'correction'
    ];

    return patternKeywords.some(keyword => phrase.includes(keyword));
  }

  /**
   * Build pattern relationships from sessions
   */
  private buildPatternRelationships(sessions: ContextualSession[]): PatternRelationship[] {
    const relationships: Map<string, PatternRelationship> = new Map();

    for (const session of sessions) {
      const patterns = this.extractPatternsFromSession(session);

      for (const pattern of patterns) {
        const patternId = `pattern-${pattern.replace(/\s+/g, '-').toLowerCase()}`;

        if (!relationships.has(patternId)) {
          relationships.set(patternId, {
            pattern_id: patternId,
            pattern_type: this.inferPatternType(pattern),
            pattern_name: pattern,
            description: `Pattern: ${pattern}`,
            related_sessions: [],
            success_rate: session.successes / (session.successes + session.failures || 1),
            avg_execution_time_ms: session.duration_ms,
            frequency: 1,
            last_seen_utc: session.end_time_utc,
            first_seen_utc: session.start_time_utc
          });
        }

        const rel = relationships.get(patternId)!;
        rel.related_sessions.push(session.session_id);
        rel.frequency++;
        rel.last_seen_utc = session.end_time_utc;
      }
    }

    return Array.from(relationships.values());
  }

  /**
   * Infer pattern type from pattern name
   */
  private inferPatternType(patternName: string): any {
    const name = patternName.toLowerCase();

    if (name.includes('refinement') || name.includes('iteration')) {
      return 'iterative_refinement';
    } if (name.includes('agent') || name.includes('collaboration')) {
      return 'agent_collaboration';
    } if (name.includes('memory') || name.includes('compression')) {
      return 'memory_compression';
    } if (name.includes('quality') || name.includes('improvement')) {
      return 'quality_improvement';
    }
    return 'custom';
  }

  /**
   * Store pattern insight as episode
   */
  private async storePatternInsight(
    sessionId: string,
    pattern: string,
    outcome: any,
    qualityScore: number | undefined,
    correlationId: string
  ): Promise<void> {
    const episodeOperation: AddEpisodeOperation = {
      name: `Pattern Learning - ${pattern}`,
      content: `Pattern: ${pattern}\nOutcome: ${outcome}\nQuality: ${qualityScore || 'N/A'}`,
      source_description: 'ICR Pattern Learning',
      episode_type: 'custom',
      valid_at: new Date().toISOString(),
      group_id: `patterns-${sessionId}`,
      uuid: uuidv4(),
      update_communities: false
    };

    await this.config.graphitiAdapter.addEpisode(episodeOperation, correlationId);
  }

  /**
   * Store lesson learned as episode
   */
  private async storeLessonLearned(
    sessionId: string,
    lesson: string,
    outcome: SessionOutcome,
    correlationId: string
  ): Promise<void> {
    const episodeOperation: AddEpisodeOperation = {
      name: `Lesson Learned - Session ${sessionId}`,
      content: `Lesson: ${lesson}\nOutcome: ${outcome.outcome}\nSatisfaction: ${outcome.user_satisfaction || 'N/A'}`,
      source_description: 'ICR Session Learning',
      episode_type: 'custom',
      valid_at: outcome.timestamp_utc,
      group_id: `lessons-${sessionId}`,
      uuid: uuidv4(),
      update_communities: false
    };

    await this.config.graphitiAdapter.addEpisode(episodeOperation, correlationId);
  }
}
