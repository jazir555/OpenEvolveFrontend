/**
 * Graphiti Adapter
 *
 * Main adapter implementation for Graphiti temporal knowledge graph integration.
 * Follows the Federation Constitution:
 * - Law of Configuration Explicitness: Environment variables for all config
 * - Law of UTC: All timestamps in UTC
 * - Law of Idempotency: All operations safe to run multiple times
 * - Runtime Truth: Verify connection before marking as initialized
 *
 * Architecture:
 * - Graphiti Client -> Canonical Schema -> Event Bus -> Other Adapters
 */

import { v4 as uuidv4 } from 'uuid';
import { logger, Logger } from '../../lib/logger';
import { CircuitBreaker, CircuitState } from '../../lib/circuit-breaker';
import {
  validateCanonical,
  CanonicalEntity,
  CanonicalEntityEdge,
  CanonicalEpisode,
  CanonicalSearchQuery,
  CanonicalSearchResult,
  AddEpisodeOperation,
  AddEpisodeResult,
  AddTripletOperation,
  AddTripletResult,
  GraphStatistics,
  TemporalFilter,
  EpisodeType,
  CanonicalEpisodeSchema,
  CanonicalEntitySchema,
  CanonicalEntityEdgeSchema,
  AddEpisodeOperationSchema,
  AddTripletOperationSchema,
  CanonicalSearchQuerySchema,
} from '../../schemas/graphiti-canonical';
import { GraphitiClient } from './graph-client';
import { GraphitiTemporalOps } from './temporal-ops';

// ============================================================================
// CONFIGURATION
// ============================================================================

export interface GraphitiAdapterConfig {
  // Required configuration
  graphiti_api_url: string;
  neo4j_uri: string;
  neo4j_user: string;
  neo4j_password: string;

  // Optional configuration with defaults
  timeout_ms?: number;
  max_retries?: number;
  retry_delay_ms?: number;
  circuit_breaker_threshold?: number;
  circuit_breaker_timeout_ms?: number;

  // Optional LLM configuration
  openai_api_key?: string;
  anthropic_api_key?: string;

  // Logging
  logger?: Logger;
}

const DEFAULT_CONFIG = {
  timeout_ms: 30000,
  max_retries: 3,
  retry_delay_ms: 1000,
  circuit_breaker_threshold: 5,
  circuit_breaker_timeout_ms: 60000,
};

// ============================================================================
// ADAPTER IMPLEMENTATION
// ============================================================================

export class GraphitiAdapter {
  private readonly config: Required<Omit<GraphitiAdapterConfig, 'logger' | 'openai_api_key' | 'anthropic_api_key'>> & {
    logger?: Logger;
    openai_api_key?: string;
    anthropic_api_key?: string;
  };

  private readonly client: GraphitiClient;
  private readonly temporalOps: GraphitiTemporalOps;
  private readonly circuitBreaker: CircuitBreaker;
  private readonly log: Logger;
  private initialized: boolean = false;

  constructor(config: GraphitiAdapterConfig) {
    this.config = {
      ...DEFAULT_CONFIG,
      ...config,
    };

    this.log = this.config.logger || new Logger('graphiti-adapter');

    // Initialize Graphiti client
    this.client = new GraphitiClient({
      neo4jUri: this.config.neo4j_uri,
      neo4jUser: this.config.neo4j_user,
      neo4jPassword: this.config.neo4j_password,
      openaiApiKey: this.config.openai_api_key,
      anthropicApiKey: this.config.anthropic_api_key,
      timeoutMs: this.config.timeout_ms,
      logger: this.log,
    });

    // Initialize temporal operations
    this.temporalOps = new GraphitiTemporalOps(this.client, this.log);

    // Initialize circuit breaker
    this.circuitBreaker = new CircuitBreaker({
      threshold: this.config.circuit_breaker_threshold,
      timeout_ms: this.config.circuit_breaker_timeout_ms,
      onStateChange: (oldState, newState) => {
        this.log.warn('Circuit breaker state changed', {
          correlation_id: 'circuit-breaker',
          old_state: oldState,
          new_state: newState,
        });
      },
    });

    this.log.info('GraphitiAdapter initialized', {
      correlation_id: 'adapter-init',
      graphiti_api_url: this.config.graphiti_api_url,
      neo4j_uri: this.config.neo4j_uri,
    });
  }

  // ========================================================================
  // INITIALIZATION
  // ========================================================================

  /**
   * Initialize adapter and verify Graphiti connection
   * Following CLAUDE.md: RUNTIME TRUTH - verify before marking as ready
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      this.log.warn('GraphitiAdapter already initialized', {
        correlation_id: 'adapter-init',
      });
      return;
    }

    const correlationId = uuidv4();

    this.log.info('Initializing GraphitiAdapter', {
      correlation_id: correlationId,
      target_service: 'graphiti-core',
    });

    try {
      // Use circuit breaker for initialization
      await this.circuitBreaker.execute(async () => {
        await this.client.initialize();
        await this.client.buildIndices();

        // Verify basic functionality with a test query
        const stats = await this.client.getStatistics();
        this.log.info('Graphiti connection verified', {
          correlation_id: correlationId,
          entities_count: stats.entities_count,
          relationships_count: stats.relationships_count,
        });
      });

      this.initialized = true;
      this.log.info('GraphitiAdapter initialized successfully', {
        correlation_id: correlationId,
      });
    } catch (error) {
      this.log.error('Failed to initialize GraphitiAdapter', error as Error, {
        correlation_id: correlationId,
      });
      throw new Error(
        `GraphitiAdapter initialization failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  // ========================================================================
  // EPISODE OPERATIONS
  // ========================================================================

  /**
   * Add an episode to the knowledge graph
   * Following CLAUDE.md: Law of Idempotency - safe to run multiple times
   */
  async addEpisode(
    operation: AddEpisodeOperation,
    correlationId?: string
  ): Promise<AddEpisodeResult> {
    const cid = correlationId || uuidv4();

    this.log.info('Adding episode to Graphiti', {
      correlation_id: cid,
      episode_name: operation.name,
      episode_type: operation.episode_type,
    });

    // Validate input
    const validation = validateCanonical(AddEpisodeOperationSchema, operation);
    if (!validation.success) {
      this.log.error('Invalid addEpisode operation', undefined, {
        correlation_id: cid,
        errors: validation.errors,
      });
      return {
        success: false,
        episode_id: uuidv4(), // Placeholder
        entities_extracted: 0,
        relationships_extracted: 0,
        processing_time_ms: 0,
        correlation_id: cid,
        error: `Validation failed: ${validation.errors.join(', ')}`,
      };
    }

    const startTime = Date.now();

    try {
      const result = await this.circuitBreaker.execute(async () => {
        return await this.client.addEpisode(operation, cid);
      });

      const processingTimeMs = Date.now() - startTime;

      this.log.info('Episode added to Graphiti successfully', {
        correlation_id: cid,
        episode_id: result.episode_id,
        entities_extracted: result.entities_extracted,
        relationships_extracted: result.relationships_extracted,
        processing_time_ms: processingTimeMs,
      });

      return {
        ...result,
        processing_time_ms: processingTimeMs,
        correlation_id: cid,
      };
    } catch (error) {
      const processingTimeMs = Date.now() - startTime;

      this.log.error('Failed to add episode to Graphiti', error as Error, {
        correlation_id: cid,
        processing_time_ms: processingTimeMs,
        circuit_state: this.circuitBreaker.getState(),
      });

      return {
        success: false,
        episode_id: uuidv4(),
        entities_extracted: 0,
        relationships_extracted: 0,
        processing_time_ms: processingTimeMs,
        correlation_id: cid,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Add multiple episodes in bulk
   */
  async addEpisodesBulk(
    operations: AddEpisodeOperation[],
    correlationId?: string
  ): Promise<AddEpisodeResult[]> {
    const cid = correlationId || uuidv4();

    this.log.info('Adding episodes in bulk to Graphiti', {
      correlation_id: cid,
      episode_count: operations.length,
    });

    const results: AddEpisodeResult[] = [];

    for (const operation of operations) {
      const result = await this.addEpisode(operation, cid);
      results.push(result);
    }

    const successCount = results.filter((r) => r.success).length;

    this.log.info('Bulk episode addition completed', {
      correlation_id: cid,
      total_count: operations.length,
      success_count: successCount,
      failure_count: operations.length - successCount,
    });

    return results;
  }

  // ========================================================================
  // TRIPLET OPERATIONS
  // ========================================================================

  /**
   * Add a triplet (subject -> predicate -> object) to the graph
   * Following CLAUDE.md: Law of Idempotency - duplicate triplets are merged
   */
  async addTriplet(
    operation: AddTripletOperation,
    correlationId?: string
  ): Promise<AddTripletResult> {
    const cid = correlationId || uuidv4();

    this.log.info('Adding triplet to Graphiti', {
      correlation_id: cid,
      subject: operation.subject.name,
      predicate: operation.predicate.relation_type,
      object: operation.object.name,
    });

    // Validate input
    const validation = validateCanonical(AddTripletOperationSchema, operation);
    if (!validation.success) {
      this.log.error('Invalid addTriplet operation', undefined, {
        correlation_id: cid,
        errors: validation.errors,
      });
      return {
        success: false,
        processing_time_ms: 0,
        correlation_id: cid,
        error: `Validation failed: ${validation.errors.join(', ')}`,
      };
    }

    const startTime = Date.now();

    try {
      const result = await this.circuitBreaker.execute(async () => {
        return await this.client.addTriplet(operation, cid);
      });

      const processingTimeMs = Date.now() - startTime;

      this.log.info('Triplet added to Graphiti successfully', {
        correlation_id: cid,
        subject_uuid: result.subject_uuid,
        object_uuid: result.object_uuid,
        edge_uuid: result.edge_uuid,
        processing_time_ms: processingTimeMs,
      });

      return {
        ...result,
        processing_time_ms: processingTimeMs,
        correlation_id: cid,
      };
    } catch (error) {
      const processingTimeMs = Date.now() - startTime;

      this.log.error('Failed to add triplet to Graphiti', error as Error, {
        correlation_id: cid,
        processing_time_ms: processingTimeMs,
      });

      return {
        success: false,
        processing_time_ms: processingTimeMs,
        correlation_id: cid,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  // ========================================================================
  // SEARCH OPERATIONS
  // ========================================================================

  /**
   * Search the knowledge graph
   */
  async search(
    query: CanonicalSearchQuery,
    correlationId?: string
  ): Promise<CanonicalSearchResult> {
    const cid = correlationId || uuidv4();

    this.log.info('Searching Graphiti', {
      correlation_id: cid,
      query: query.query,
      temporal_filter: query.temporal_filter,
      max_results: query.max_results,
    });

    // Validate input
    const validation = validateCanonical(CanonicalSearchQuerySchema, query);
    if (!validation.success) {
      this.log.error('Invalid search query', undefined, {
        correlation_id: cid,
        errors: validation.errors,
      });
      throw new Error(`Invalid search query: ${validation.errors.join(', ')}`);
    }

    const startTime = Date.now();

    try {
      const result = await this.circuitBreaker.execute(async () => {
        return await this.client.search(query, cid);
      });

      const queryTimeMs = Date.now() - startTime;

      this.log.info('Graphiti search completed', {
        correlation_id: cid,
        results_count: result.edges.length,
        nodes_count: result.nodes.length,
        query_time_ms: queryTimeMs,
      });

      return {
        ...result,
        query_time_ms: queryTimeMs,
      };
    } catch (error) {
      const queryTimeMs = Date.now() - startTime;

      this.log.error('Graphiti search failed', error as Error, {
        correlation_id: cid,
        query_time_ms: queryTimeMs,
      });

      throw error;
    }
  }

  // ========================================================================
  // TEMPORAL OPERATIONS
  // ========================================================================

  /**
   * Query knowledge at a specific point in time
   */
  async queryAtPointInTime(
    query: string,
    timestamp: string,
    maxResults: number = 10,
    correlationId?: string
  ): Promise<CanonicalSearchResult> {
    const cid = correlationId || uuidv4();

    this.log.info('Querying Graphiti at point in time', {
      correlation_id: cid,
      query,
      timestamp,
      max_results: maxResults,
    });

    return await this.temporalOps.queryAtPointInTime(
      query,
      timestamp,
      maxResults,
      cid
    );
  }

  /**
   * Get entity timeline
   */
  async getEntityTimeline(
    entityName: string,
    startTime: string,
    endTime: string,
    correlationId?: string
  ): Promise<any[]> {
    const cid = correlationId || uuidv4();

    this.log.info('Getting entity timeline from Graphiti', {
      correlation_id: cid,
      entity: entityName,
      start_time: startTime,
      end_time: endTime,
    });

    return await this.temporalOps.getEntityTimeline(
      entityName,
      startTime,
      endTime,
      cid
    );
  }

  // ========================================================================
  // ENTITY OPERATIONS
  // ========================================================================

  /**
   * Get an entity by UUID
   */
  async getEntity(uuid: string, correlationId?: string): Promise<CanonicalEntity | null> {
    const cid = correlationId || uuidv4();

    this.log.info('Getting entity from Graphiti', {
      correlation_id: cid,
      entity_uuid: uuid,
    });

    try {
      const entity = await this.circuitBreaker.execute(async () => {
        return await this.client.getEntity(uuid, cid);
      });

      if (!entity) {
        this.log.warn('Entity not found', {
          correlation_id: cid,
          entity_uuid: uuid,
        });
        return null;
      }

      // Validate against canonical schema
      const validation = validateCanonical(CanonicalEntitySchema, entity);
      if (!validation.success) {
        this.log.error('Invalid entity returned from Graphiti', undefined, {
          correlation_id: cid,
          entity_uuid: uuid,
          errors: validation.errors,
        });
        throw new Error(`Invalid entity: ${validation.errors.join(', ')}`);
      }

      this.log.info('Entity retrieved successfully', {
        correlation_id: cid,
        entity_uuid: uuid,
        entity_name: entity.name,
      });

      return validation.data!;
    } catch (error) {
      this.log.error('Failed to get entity from Graphiti', error as Error, {
        correlation_id: cid,
        entity_uuid: uuid,
      });
      throw error;
    }
  }

  // ========================================================================
  // GRAPH STATISTICS
  // ========================================================================

  /**
   * Get graph statistics
   */
  async getStatistics(correlationId?: string): Promise<GraphStatistics> {
    const cid = correlationId || uuidv4();

    this.log.info('Getting Graphiti statistics', {
      correlation_id: cid,
    });

    try {
      const stats = await this.client.getStatistics();

      this.log.info('Graphiti statistics retrieved', {
        correlation_id: cid,
        entities_count: stats.entities_count,
        relationships_count: stats.relationships_count,
        episodes_count: stats.episodes_count,
      });

      return stats;
    } catch (error) {
      this.log.error('Failed to get Graphiti statistics', error as Error, {
        correlation_id: cid,
      });
      throw error;
    }
  }

  // ========================================================================
  // HEALTH CHECK
  // ========================================================================

  /**
   * Check adapter health
   */
  async healthCheck(): Promise<{
    healthy: boolean;
    circuit_state: CircuitState;
    initialized: boolean;
    graphiti_connected: boolean;
  }> {
    const circuitStats = this.circuitBreaker.getStats();

    try {
      if (!this.initialized) {
        return {
          healthy: false,
          circuit_state: circuitStats.state,
          initialized: false,
          graphiti_connected: false,
        };
      }

      // Quick connectivity check
      await this.client.getStatistics();

      return {
        healthy: circuitStats.state === CircuitState.CLOSED,
        circuit_state: circuitStats.state,
        initialized: true,
        graphiti_connected: true,
      };
    } catch (error) {
      return {
        healthy: false,
        circuit_state: circuitStats.state,
        initialized: true,
        graphiti_connected: false,
      };
    }
  }

  // ========================================================================
  // CLEANUP
  // ========================================================================

  /**
   * Close adapter and cleanup resources
   */
  async close(): Promise<void> {
    this.log.info('Closing GraphitiAdapter', {
      correlation_id: 'adapter-close',
    });

    try {
      await this.client.close();
      this.initialized = false;

      this.log.info('GraphitiAdapter closed successfully', {
        correlation_id: 'adapter-close',
      });
    } catch (error) {
      this.log.error('Error closing GraphitiAdapter', error as Error, {
        correlation_id: 'adapter-close',
      });
    }
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export * from './graph-client';
export * from './temporal-ops';
