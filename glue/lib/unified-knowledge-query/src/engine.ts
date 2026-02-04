/**
 * Unified Knowledge Query Engine
 *
 * Main engine for querying multiple knowledge systems.
 *
 * Federation Constitution Compliance:
 * - Configuration Explicitness: All config via environment variables
 * - Failure Management: Circuit breakers, retries, fallbacks
 * - Observability: Structured logging with correlation IDs
 * - Law of UTC: All timestamps in UTC
 */

import { Logger } from '@openevolve/glue-lib';
import {
  validateQuery,
  UnifiedKnowledgeQuery,
  UnifiedQueryResult,
  QueryOptions,
  SystemConfig,
  SystemHealth,
  EngineMetrics,
  SystemSource,
  HealthStatus,
} from './canonical';
import { QueryRouter } from './query-router';
import { ResultFusion } from './result-fusion';
import { FallbackStrategy } from './fallback-strategy';
import { RAGBitsClient, GraphitiClient, VectorDBClient } from './clients';

/**
 * Engine Configuration
 */
interface EngineConfig {
  ragbits: Partial<SystemConfig>;
  graphiti: Partial<SystemConfig>;
  vectordb: Partial<SystemConfig>;
}

/**
 * Engine Constructor Options
 */
interface EngineOptions {
  ragbitsUrl?: string;
  graphitiUrl?: string;
  vectordbUrl?: string;
  timeout?: number;
  enableFallback?: boolean;
  maxResults?: number;
}

/**
 * Unified Knowledge Query Engine
 *
 * Main entry point for querying multiple knowledge systems
 */
export class UnifiedKnowledgeQueryEngine {
  private logger: Logger;
  private router: QueryRouter;
  private fusion: ResultFusion;
  private fallback: FallbackStrategy;

  private systems: Map<SystemSource, SystemConfig>;
  private metrics: EngineMetrics;
  private startTime: number;

  constructor(options: EngineOptions = {}) {
    this.logger = new Logger('unified-knowledge-engine');
    this.router = new QueryRouter();
    this.fusion = new ResultFusion();
    this.fallback = new FallbackStrategy({
      enableFallback: options.enableFallback ?? true,
    });

    this.startTime = Date.now();
    this.systems = new Map();
    this.metrics = {
      totalQueries: 0,
      successfulQueries: 0,
      failedQueries: 0,
      averageQueryTime: 0,
      systemHealth: [],
      uptime: 0,
    };

    // Initialize systems
    this.initializeSystems(options);
  }

  /**
   * Execute unified knowledge query
   *
   * @param query - Query text
   * @param options - Query options
   * @returns Unified query results
   */
  async query(
    query: string,
    options: QueryOptions = {}
  ): Promise<UnifiedQueryResult> {
    const correlationId = options.correlationId || this.generateCorrelationId();
    const startTime = Date.now();

    // Build unified query object
    const unifiedQuery: UnifiedKnowledgeQuery = {
      query,
      domains: options.domains || ['all'],
      queryType: options.queryType || 'hybrid',
      temporalFilter: options.temporalFilter,
      knowledgeTypes: options.knowledgeTypes || ['all'],
      maxResults: options.maxResults || 50,
      minConfidence: options.minConfidence || 0.0,
      maxDepth: options.maxDepth || 2,
      correlationId,
    };

    // Validate query (Law of Configuration Explicitness)
    const validated = validateQuery(unifiedQuery);

    this.logger.info('Query execution started', {
      correlation_id: correlationId,
      query: query.substring(0, 100),
      domains: validated.domains,
      query_type: validated.queryType,
    });

    this.metrics.totalQueries++;

    try {
      // Route query to appropriate systems
      const availableSystems = Array.from(this.systems.values());
      const plan = await this.router.route(validated, availableSystems);

      // Execute queries
      const systemResults = await this.executeQueries(validated, plan);

      // Merge results
      let mergedResult = await this.fusion.merge(
        systemResults,
        query,
        correlationId
      );

      // Apply filters
      mergedResult = this.applyFilters(mergedResult, validated);

      // Update metrics
      this.metrics.successfulQueries++;
      this.updateAverageQueryTime(Date.now() - startTime);

      this.logger.info('Query completed successfully', {
        correlation_id: correlationId,
        result_count: mergedResult.results.length,
        confidence: mergedResult.confidence,
        execution_time: mergedResult.executionTimeMs,
      });

      return mergedResult;
    } catch (error) {
      this.metrics.failedQueries++;
      this.updateAverageQueryTime(Date.now() - startTime);

      this.logger.error('Query failed', error as Error, {
        correlation_id: correlationId,
        query: query.substring(0, 100),
      });

      throw error;
    }
  }

  /**
   * Semantic search query
   */
  async semanticSearch(
    query: string,
    options: QueryOptions = {}
  ): Promise<UnifiedQueryResult> {
    return this.query(query, {
      ...options,
      queryType: 'semantic-search',
      domains: ['vectordb', 'ragbits'],
    });
  }

  /**
   * Temporal query with time filters
   */
  async temporalQuery(
    query: string,
    startDate: string,
    endDate: string,
    options: QueryOptions = {}
  ): Promise<UnifiedQueryResult> {
    return this.query(query, {
      ...options,
      queryType: 'temporal-query',
      domains: ['graphiti'],
      temporalFilter: {
        startDate,
        endDate,
      },
    });
  }

  /**
   * Graph traversal query
   */
  async graphTraversal(
    query: string,
    options: QueryOptions = {}
  ): Promise<UnifiedQueryResult> {
    return this.query(query, {
      ...options,
      queryType: 'graph-traversal',
      domains: ['graphiti'],
    });
  }

  /**
   * Hybrid query across all systems
   */
  async hybridQuery(
    query: string,
    options: QueryOptions = {}
  ): Promise<UnifiedQueryResult> {
    return this.query(query, {
      ...options,
      queryType: 'hybrid',
      domains: ['all'],
    });
  }

  /**
   * Health check for all systems
   */
  async healthCheck(): Promise<SystemHealth[]> {
    const healthChecks: Promise<SystemHealth>[] = [];

    for (const [name, config] of this.systems.entries()) {
      healthChecks.push(this.checkSystemHealth(name, config));
    }

    const results = await Promise.all(healthChecks);

    // Update metrics
    this.metrics.systemHealth = results;

    // Update router and fallback with health info
    this.router.updateSystemHealth(results);
    this.fallback.updateSystemHealth(results);

    this.logger.info('Health check completed', {
      systems: results.map(r => ({
        system: r.system,
        status: r.status,
      })),
    });

    return results;
  }

  /**
   * Get engine metrics
   */
  async getMetrics(): Promise<EngineMetrics> {
    // Update uptime
    this.metrics.uptime = Date.now() - this.startTime;

    // Get latest health
    await this.healthCheck();

    return {
      ...this.metrics,
      averageQueryTime: Math.round(this.metrics.averageQueryTime),
    };
  }

  /**
   * Reset engine metrics
   */
  resetMetrics(): void {
    this.metrics = {
      totalQueries: 0,
      successfulQueries: 0,
      failedQueries: 0,
      averageQueryTime: 0,
      systemHealth: [],
      uptime: Date.now() - this.startTime,
    };

    this.logger.info('Engine metrics reset');
  }

  /**
   * Execute queries according to plan
   */
  private async executeQueries(
    query: UnifiedKnowledgeQuery,
    plan: any
  ): Promise<any[]> {
    const systemResults: any[] = [];
    const correlationId = query.correlationId || this.generateCorrelationId();

    // Check if parallel execution is possible
    if (plan.parallelizable && plan.systems.length > 1) {
      // Execute in parallel
      const promises = plan.systems.map(system =>
        this.executeSystemQuery(system, query, correlationId)
          .catch(error => ({
            system: system.name,
            items: [],
            queryTimeMs: 0,
            success: false,
            error: error.message,
          }))
      );

      const results = await Promise.all(promises);
      systemResults.push(...results);
    } else {
      // Execute sequentially with fallback
      for (const system of plan.systems) {
        const startTime = Date.now();

        try {
          const items = await this.executeSystemQuery(system, query, correlationId);

          systemResults.push({
            system: system.name,
            items,
            queryTimeMs: Date.now() - startTime,
            success: true,
          });

          // If we got good results, don't try other systems
          if (items.length > 0 && items.length >= query.maxResults * 0.5) {
            this.logger.info('Sufficient results obtained', {
              correlation_id: correlationId,
              system: system.name,
              result_count: items.length,
            });
            break;
          }
        } catch (error) {
          systemResults.push({
            system: system.name,
            items: [],
            queryTimeMs: Date.now() - startTime,
            success: false,
            error: (error as Error).message,
          });

          // Try fallback
          const fallbacks = this.fallback.getAvailableFallbacks(
            plan.systems.filter(s => s.name !== system.name)
          );

          if (fallbacks.length > 0) {
            const fallback = fallbacks[0];

            this.logger.info('Trying fallback system', {
              correlation_id: correlationId,
              fallback: fallback.name,
            });

            try {
              const items = await this.executeSystemQuery(
                fallback,
                query,
                correlationId
              );

              systemResults.push({
                system: fallback.name,
                items,
                queryTimeMs: Date.now() - startTime,
                success: true,
              });

              break;
            } catch (fallbackError) {
              this.logger.warn('Fallback system failed', {
                correlation_id: correlationId,
                fallback: fallback.name,
                error: (fallbackError as Error).message,
              });
            }
          }
        }
      }
    }

    return systemResults;
  }

  /**
   * Execute query against a single system
   */
  private async executeSystemQuery(
    system: SystemConfig,
    query: UnifiedKnowledgeQuery,
    correlationId: string
  ): Promise<any[]> {
    const client = this.createClient(system);

    const items = await client.search(query.query, {
      maxResults: query.maxResults,
      temporalFilter: query.temporalFilter,
      knowledgeTypes: query.knowledgeTypes,
      correlationId,
    });

    return items;
  }

  /**
   * Create client for system
   */
  private createClient(system: SystemConfig): any {
    switch (system.name) {
      case 'ragbits':
        return new RAGBitsClient(system);
      case 'graphiti':
        return new GraphitiClient(system);
      case 'vectordb':
        return new VectorDBClient(system);
      default:
        throw new Error(`Unknown system: ${system.name}`);
    }
  }

  /**
   * Apply filters to results
   */
  private applyFilters(
    result: UnifiedQueryResult,
    query: UnifiedKnowledgeQuery
  ): UnifiedQueryResult {
    let filtered = result.results;

    // Filter by confidence
    filtered = this.fusion.filterByConfidence(filtered, query.minConfidence);

    // Filter by type
    filtered = this.fusion.filterByType(filtered, query.knowledgeTypes);

    // Deduplicate
    filtered = this.fusion.deduplicateById(filtered);

    // Limit results
    filtered = this.fusion.limitResults(filtered, query.maxResults);

    return {
      ...result,
      results: filtered,
    };
  }

  /**
   * Check health of a single system
   */
  private async checkSystemHealth(
    name: SystemSource,
    config: SystemConfig
  ): Promise<SystemHealth> {
    const startTime = Date.now();

    try {
      const client = this.createClient(config);
      const isHealthy = await client.healthCheck();

      return {
        system: name,
        status: isHealthy ? 'healthy' : 'unhealthy',
        responseTimeMs: Date.now() - startTime,
        lastCheck: new Date().toISOString(),
      };
    } catch (error) {
      return {
        system: name,
        status: 'unhealthy',
        responseTimeMs: Date.now() - startTime,
        lastCheck: new Date().toISOString(),
        error: (error as Error).message,
      };
    }
  }

  /**
   * Initialize systems from config
   */
  private initializeSystems(options: EngineOptions): void {
    // Validate required environment variables
    const ragbitsUrl = options.ragbitsUrl || process.env.RAGBITS_URL;
    const graphitiUrl = options.graphitiUrl || process.env.GRAPHITI_URL;
    const vectordbUrl = options.vectordbUrl || process.env.VECTORDB_URL;

    // RAGBits
    if (ragbitsUrl) {
      this.systems.set('ragbits', {
        name: 'ragbits',
        enabled: true,
        url: ragbitsUrl,
        timeout: options.timeout || 5000,
        priority: 3,
      });
    }

    // Graphiti
    if (graphitiUrl) {
      this.systems.set('graphiti', {
        name: 'graphiti',
        enabled: true,
        url: graphitiUrl,
        timeout: options.timeout || 5000,
        priority: 2,
      });
    }

    // Vector DB
    if (vectordbUrl) {
      this.systems.set('vectordb', {
        name: 'vectordb',
        enabled: true,
        url: vectordbUrl,
        timeout: options.timeout || 5000,
        priority: 1,
      });
    }

    // Log warning if no systems configured
    if (this.systems.size === 0) {
      this.logger.warn('No knowledge systems configured', {
        required_env_vars: ['RAGBITS_URL', 'GRAPHITI_URL', 'VECTORDB_URL'],
      });
    }

    this.logger.info('Systems initialized', {
      systems: Array.from(this.systems.keys()),
    });
  }

  /**
   * Update average query time
   */
  private updateAverageQueryTime(queryTime: number): void {
    const total = this.metrics.averageQueryTime * (this.metrics.totalQueries - 1);
    this.metrics.averageQueryTime = (total + queryTime) / this.metrics.totalQueries;
  }

  /**
   * Generate correlation ID (UUID v4)
   */
  private generateCorrelationId(): string {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }
}

/**
 * Default engine instance
 */
export const defaultEngine = new UnifiedKnowledgeQueryEngine();
