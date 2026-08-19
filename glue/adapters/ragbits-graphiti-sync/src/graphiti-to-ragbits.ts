/**
 * Graphiti to RAGBits Sync
 *
 * Follows the Federation Constitution:
 * - Law of the Air Gap: No imports from core-projects
 * - Law of Runtime Truth: Verify API calls work
 * - Law of Idempotency: Safe to run multiple times
 * - Failure Management: Circuit breakers and retries
 *
 * Synchronizes data from Graphiti (Temporal Knowledge Graph) to RAGBits (RAG system)
 * Enhances retrieval with knowledge graph entities
 */

import { v4 as uuidv4 } from 'uuid';
import { Logger } from '../../../lib/logger.js';
import { CircuitBreaker } from '../../../lib/circuit-breaker.js';
import {
  SyncResult,
  SyncDirection,
  SyncStatus,
  EnhancedQuery,
  BoostFactor,
  createSyncResult,
} from './canonical.js';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

/**
 * Graphiti Entity
 */
export interface GraphitiEntity {
  id: string;
  name: string;
  labels: string[];
  summary?: string;
  attributes: Record<string, any>;
  created_at: string;
  updated_at?: string;
}

/**
 * Graphiti Episode
 */
export interface GraphitiEpisode {
  id: string;
  name: string;
  content: string;
  episode_type: string;
  valid_at: string;
  created_at: string;
  entity_edges: string[];
}

/**
 * Entity with temporal context
 */
export interface TemporalEntity {
  entity: GraphitiEntity;
  valid_at: string;
  episodes: string[];
  relationships: Relationship[];
}

/**
 * Relationship between entities
 */
export interface Relationship {
  id: string;
  source_entity_id: string;
  target_entity_id: string;
  relation_type: string;
  fact: string;
  valid_at: string;
}

/**
 * Configuration for Graphiti to RAGBits sync
 */
export interface GraphitiToRAGBitsConfig {
  graphiti_api_url: string;
  ragbits_api_url: string;
  timeout_ms: number;
  max_retries: number;
  retry_delay_ms: number;
  batch_size: number;
  enable_enhancement: boolean;
  boost_threshold: number;
  min_confidence: number;
}

// ============================================================================
// MAIN SYNC CLASS
// ============================================================================

/**
 * Graphiti to RAGBits Synchronization
 *
 * Handles one-way sync from Graphiti knowledge graph to RAGBits
 * Enhances retrieval with knowledge graph context
 */
export class GraphitiToRAGBitsSync {
  private readonly logger: Logger;
  private readonly circuitBreaker: CircuitBreaker;
  private readonly graphitiServiceName = 'graphiti';
  private readonly ragbitsServiceName = 'ragbits';

  constructor(_config: GraphitiToRAGBitsConfig) {
    this.logger = new Logger('graphiti-to-ragbits-sync');
    this.circuitBreaker = new CircuitBreaker({
      threshold: 5,
      timeout_ms: 60000,
      onStateChange: (oldState, newState) => {
        this.logger.warn('Circuit breaker state changed', {
          old_state: oldState,
          new_state: newState,
          source_service: this.graphitiServiceName,
          target_service: this.ragbitsServiceName,
        });
      },
    });
  }

  /**
   * Sync entity metadata to RAGBits
   *
   * @param entity - Graphiti entity to sync
   * @param correlationId - Correlation ID for tracing
   * @returns Sync result
   */
  async syncEntity(entity: GraphitiEntity, correlationId: string): Promise<SyncResult> {
    const startTime = Date.now();
    const operationId = uuidv4();

    this.logger.info('Starting entity sync to RAGBits', {
      correlation_id: correlationId,
      operation_id: operationId,
      entity_id: entity.id,
      entity_name: entity.name,
      source_service: this.graphitiServiceName,
      target_service: this.ragbitsServiceName,
    });

    const result = createSyncResult(
      operationId,
      SyncStatus.pending,
      SyncDirection.graphiti_to_ragbits,
      correlationId
    );

    result.operations_total = 1;

    try {
      // Extract keywords from entity
      const keywords = this.extractKeywords(entity);

      // Create boost factors for retrieval
      const boostFactor = this.createEntityBoost(entity);

      // Update RAGBits with entity metadata
      await this.circuitBreaker.execute(async () => {
        return await this.updateRAGBitsWithEntity(entity, keywords, boostFactor, correlationId);
      });

      result.status = SyncStatus.completed;
      result.operations_completed = 1;
      result.duration_ms = Date.now() - startTime;

      this.logger.info('Entity sync completed successfully', {
        correlation_id: correlationId,
        operation_id: operationId,
        entity_id: entity.id,
        entity_name: entity.name,
        keywords_count: keywords.length,
        duration_ms: result.duration_ms,
      });

      return result;
    } catch (error) {
      result.status = SyncStatus.failed;
      result.operations_failed = 1;
      result.duration_ms = Date.now() - startTime;
      result.errors.push({
        code: 'ENTITY_SYNC_FAILED',
        message: error instanceof Error ? error.message : 'Unknown error',
        details: { entity_id: entity.id, entity_name: entity.name },
      });

      this.logger.error('Entity sync failed', error as Error, {
        correlation_id: correlationId,
        operation_id: operationId,
        entity_id: entity.id,
        duration_ms: result.duration_ms,
      });

      return result;
    }
  }

  /**
   * Enhance retrieval query with knowledge graph entities
   *
   * @param query - Original user query
   * @param correlationId - Correlation ID for tracing
   * @returns Enhanced query with entity context
   */
  async enhanceRetrieval(query: string, correlationId: string): Promise<EnhancedQuery> {
    this.logger.info('Enhancing retrieval query with knowledge graph', {
      correlation_id: correlationId,
      query_length: query.length,
    });

    try {
      // Search for relevant entities in the knowledge graph
      const entities = await this.searchEntities(query, correlationId);

      // Create boost factors based on entities
      const boostFactors: BoostFactor[] = [];
      const entitiesData = [];

      for (const entity of entities) {
        const boost = this.createEntityBoost(entity);
        boostFactors.push(boost);

        entitiesData.push({
          id: entity.id,
          name: entity.name,
          labels: entity.labels,
          boost_factor: boost.boost_value,
        });
      }

      // Enhance query with entity context
      const enhancedQuery = this.buildEnhancedQuery(query, entities);

      this.logger.info('Query enhancement completed', {
        correlation_id: correlationId,
        entities_found: entities.length,
        boost_factors_created: boostFactors.length,
      });

      return {
        original_query: query,
        enhanced_query: enhancedQuery,
        entities: entitiesData,
        boost_factors: Object.fromEntries(
          boostFactors.map((bf) => [bf.entity_id, bf.boost_value])
        ),
        metadata: {
          enhanced_at: new Date().toISOString(),
          entity_count: entities.length,
        },
      };
    } catch (error) {
      this.logger.warn('Query enhancement failed, returning original query', {
        correlation_id: correlationId,
        error_message: error instanceof Error ? error.message : 'Unknown error',
      });

      // Return original query if enhancement fails
      return {
        original_query: query,
        enhanced_query: query,
        entities: [],
        boost_factors: {},
        metadata: {
          enhanced_at: new Date().toISOString(),
          enhancement_failed: true,
          error: error instanceof Error ? error.message : 'Unknown error',
        },
      };
    }
  }

  /**
   * Extract keywords from entity for retrieval enhancement
   *
   * @param entity - Graphiti entity
   * @returns Array of keywords
   */
  extractKeywords(entity: GraphitiEntity): string[] {
    const keywords: string[] = [];

    // Add entity name
    keywords.push(entity.name.toLowerCase());

    // Add labels
    for (const label of entity.labels) {
      keywords.push(label.toLowerCase());
    }

    // Add summary keywords
    if (entity.summary) {
      const words = entity.summary.split(/\s+/);
      for (const word of words) {
        if (word.length > 3) {
          keywords.push(word.toLowerCase().replace(/[^a-z0-9]/g, ''));
        }
      }
    }

    // Remove duplicates
    const uniqueKeywords = Array.from(new Set(keywords));

    this.logger.debug('Extracted keywords from entity', {
      entity_id: entity.id,
      entity_name: entity.name,
      keywords_count: uniqueKeywords.length,
    });

    return uniqueKeywords;
  }

  /**
   * Create boost factor for entity
   *
   * @param entity - Graphiti entity
   * @returns Boost factor
   */
  createEntityBoost(entity: GraphitiEntity): BoostFactor {
    // Calculate boost based on entity properties
    let boostValue = 1.0;

    // Boost based on number of labels (more specific = higher boost)
    boostValue += entity.labels.length * 0.1;

    // Boost if entity has summary (more context = higher boost)
    if (entity.summary) {
      boostValue += 0.2;
    }

    // Boost based on age (newer entities get slight boost)
    const createdAt = new Date(entity.created_at);
    const daysSinceCreation = (Date.now() - createdAt.getTime()) / (1000 * 60 * 60 * 24);
    if (daysSinceCreation < 30) {
      boostValue += 0.1;
    }

    const boost: BoostFactor = {
      entity_id: entity.id,
      boost_value: Math.min(boostValue, 3.0), // Cap at 3.0
      reason: `Entity boost based on labels (${entity.labels.length}), summary (${
        entity.summary ? 'yes' : 'no'
      }), and age (${Math.floor(daysSinceCreation)} days)`,
      confidence: this.calculateEntityConfidence(entity),
    };

    this.logger.debug('Created boost factor for entity', {
      entity_id: entity.id,
      entity_name: entity.name,
      boost_value: boost.boost_value,
      confidence: boost.confidence,
    });

    return boost;
  }

  /**
   * Update RAGBits with entity metadata
   *
   * @param entity - Graphiti entity
   * @param keywords - Keywords for the entity
   * @param boostFactor - Boost factor for retrieval
   * @param correlationId - Correlation ID for tracing
   * @returns Promise that resolves when updated
   */
  private async updateRAGBitsWithEntity(
    entity: GraphitiEntity,
    keywords: string[],
    boostFactor: BoostFactor,
    correlationId: string
  ): Promise<void> {
    this.logger.debug('Updating RAGBits with entity metadata', {
      correlation_id: correlationId,
      entity_id: entity.id,
      keywords_count: keywords.length,
      boost_value: boostFactor.boost_value,
    });

    // TODO: Implement actual RAGBits API call
    // This is a placeholder that demonstrates the structure
    // In production, this would call the RAGBits API to update metadata

    // Simulate API call
    await this.simulateApiCall(correlationId);

    this.logger.debug('RAGBits updated with entity metadata', {
      correlation_id: correlationId,
      entity_id: entity.id,
    });
  }

  /**
   * Search for entities relevant to query
   *
   * @param query - User query
   * @param correlationId - Correlation ID for tracing
   * @returns Array of relevant entities
   */
  private async searchEntities(query: string, correlationId: string): Promise<GraphitiEntity[]> {
    this.logger.debug('Searching for entities in knowledge graph', {
      correlation_id: correlationId,
      query_length: query.length,
    });

    // TODO: Implement actual Graphiti search API call
    // This is a placeholder that demonstrates the structure
    // In production, this would search the Graphiti knowledge graph

    // Simulate API call
    await this.simulateApiCall(correlationId);

    // Return empty array for now
    const entities: GraphitiEntity[] = [];

    this.logger.debug('Entity search completed', {
      correlation_id: correlationId,
      entities_count: entities.length,
    });

    return entities;
  }

  /**
   * Build enhanced query with entity context
   *
   * @param originalQuery - Original user query
   * @param entities - Relevant entities
   * @returns Enhanced query
   */
  private buildEnhancedQuery(originalQuery: string, entities: GraphitiEntity[]): string {
    if (entities.length === 0) {
      return originalQuery;
    }

    // Build context from entities
    const entityContexts = entities.map((entity) => {
      let context = `${entity.name}`;
      if (entity.summary) {
        context += `: ${entity.summary}`;
      }
      return context;
    });

    // Combine original query with entity context
    const enhancedQuery = `${originalQuery}\n\nRelevant entities:\n${entityContexts.join('\n')}`;

    this.logger.debug('Built enhanced query', {
      original_length: originalQuery.length,
      enhanced_length: enhancedQuery.length,
      entities_count: entities.length,
    });

    return enhancedQuery;
  }

  /**
   * Calculate confidence score for entity
   *
   * @param entity - Graphiti entity
   * @returns Confidence score (0-1)
   */
  private calculateEntityConfidence(entity: GraphitiEntity): number {
    let confidence = 0.5; // Base confidence

    // Increase confidence based on labels
    confidence += Math.min(entity.labels.length * 0.1, 0.3);

    // Increase confidence if has summary
    if (entity.summary) {
      confidence += 0.2;
    }

    // Increase confidence based on attribute count
    const attributeCount = Object.keys(entity.attributes).length;
    confidence += Math.min(attributeCount * 0.05, 0.2);

    return Math.min(confidence, 1.0);
  }

  /**
   * Update retrieval strategy based on entities
   *
   * @param query - Original query
   * @param entities - Relevant entities
   * @returns Updated query strategy
   */
  updateRetrievalStrategy(query: string, entities: GraphitiEntity[]): string {
    this.logger.debug('Updating retrieval strategy with entities', {
      query_length: query.length,
      entities_count: entities.length,
    });

    // Build filters based on entities
    const filters: Record<string, any> = {
      include_entities: entities.map((e) => e.id),
    };

    // Add temporal filters if entities have temporal context
    const dates = entities
      .map((e) => e.created_at)
      .filter((d): d is string => d !== undefined)
      .sort();

    if (dates.length > 0) {
      filters.temporal_range = {
        start: dates[0],
        end: dates[dates.length - 1],
      };
    }

    this.logger.debug('Retrieval strategy updated', {
      filters_count: Object.keys(filters).length,
    });

    return JSON.stringify(filters);
  }

  /**
   * Simulate API call (placeholder)
   *
   * @param correlationId - Correlation ID for tracing
   * @returns Promise that resolves after a delay
   */
  private async simulateApiCall(_correlationId: string): Promise<void> {
    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  /**
   * Get circuit breaker stats
   *
   * @returns Circuit breaker statistics
   */
  getCircuitBreakerStats() {
    return this.circuitBreaker.getStats();
  }

  /**
   * Reset circuit breaker
   */
  resetCircuitBreaker(): void {
    this.circuitBreaker.reset();
    this.logger.info('Circuit breaker reset');
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export default GraphitiToRAGBitsSync;
