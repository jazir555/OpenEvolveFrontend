/**
 * KNOWLEDGE RETRIEVAL WORKFLOW
 *
 * A comprehensive workflow for retrieving relevant knowledge from multiple
 * knowledge sources including RAGBits, Graphiti, and Vector DB.
 *
 * This workflow combines:
 * 1. RAGBits semantic document search
 * 2. Graphiti entity/relationship graph queries
 * 3. Vector DB similarity search for historical executions
 * 4. Multi-source result merging and ranking
 *
 * Follows Federation Constitution:
 * - Law of Runtime Truth: Validates all knowledge sources via probe scripts
 * - Law of Configuration Explicitness: All URLs and timeouts from environment
 * - Failure Management: Circuit breakers for each knowledge source
 */

import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import { HttpBubble } from '../service-bubble/http.js';
import { CircuitBreaker } from '../../lib/circuitBreaker.js';
import { NetworkError, ValidationError } from '../../lib/errors.js';

/**
 * Knowledge source configuration
 */
const KnowledgeSourceSchema = z.object({
  ragbits: z
    .boolean()
    .default(true)
    .describe('Enable RAGBits document search'),
  graphiti: z
    .boolean()
    .default(true)
    .describe('Enable Graphiti graph search'),
  vectordb: z
    .boolean()
    .default(true)
    .describe('Enable Vector DB similarity search'),
});

/**
 * Query options for knowledge retrieval
 */
const QueryOptionsSchema = z.object({
  topK: z
    .number()
    .int()
    .min(1)
    .max(100)
    .default(10)
    .describe('Number of results to retrieve per source'),
  minScore: z
    .number()
    .min(0)
    .max(1)
    .default(0.7)
    .describe('Minimum similarity score threshold'),
  filters: z
    .record(z.string(), z.any())
    .optional()
    .default({})
    .describe('Metadata filters for search'),
  timeout: z
    .number()
    .int()
    .min(1000)
    .max(60000)
    .default(10000)
    .describe('Request timeout in milliseconds'),
});

/**
 * Parameters schema for knowledge retrieval workflow
 */
const KnowledgeRetrievalParamsSchema = z.object({
  /**
   * Query string for knowledge retrieval
   */
  query: z
    .string()
    .min(1, 'Query is required')
    .describe('Knowledge retrieval query'),

  /**
   * Knowledge sources to query
   */
  sources: KnowledgeSourceSchema.optional().describe('Enabled knowledge sources'),

  /**
   * Query options
   */
  options: QueryOptionsSchema.optional().describe('Query configuration'),

  /**
   * Maximum number of results to return (after merging)
   */
  maxResults: z
    .number()
    .int()
    .min(1)
    .max(100)
    .default(10)
    .describe('Maximum total results to return'),

  /**
   * Knowledge source endpoints (from environment variables)
   */
  endpoints: z
    .object({
      ragbits: z
        .string()
        .url()
        .optional()
        .describe('RAGBits server URL (from RAGBITS_URL env var)'),
      graphiti: z
        .string()
        .url()
        .optional()
        .describe('Graphiti server URL (from GRAPHITI_URL env var)'),
      vectordb: z
        .string()
        .url()
        .optional()
        .describe('Vector DB URL (from VECTORDB_URL env var)'),
    })
    .optional()
    .describe('Knowledge source endpoints'),

  /**
   * Credentials
   */
  credentials: z
    .record(z.nativeEnum(CredentialType), z.string())
    .optional()
    .describe('Credentials for knowledge sources'),
});

type KnowledgeRetrievalParams = z.input<typeof KnowledgeRetrievalParamsSchema>;

/**
 * Individual knowledge result
 */
const KnowledgeResultSchema = z.object({
  content: z
    .string()
    .describe('Knowledge content'),
  source: z
    .enum(['ragbits', 'graphiti', 'vectordb'])
    .describe('Knowledge source'),
  score: z
    .number()
    .describe('Similarity/relevance score'),
  metadata: z
    .record(z.string(), z.any())
    .optional()
    .describe('Additional metadata'),
});

/**
 * Merged result with combined scores
 */
const MergedResultSchema = z.object({
  content: z.string(),
  sources: z.array(z.enum(['ragbits', 'graphiti', 'vectordb'])),
  aggregatedScore: z.number(),
  rank: z.number(),
  metadata: z.record(z.string(), z.any()).optional(),
});

/**
 * Result schema for knowledge retrieval workflow
 */
const KnowledgeRetrievalResultSchema = z.object({
  success: z.boolean(),
  error: z.string().optional(),

  /**
   * Original query
   */
  query: z
    .string()
    .describe('Original query'),

  /**
   * Retrieved and ranked knowledge results
   */
  results: z
    .array(MergedResultSchema)
    .describe('Top-k merged knowledge results'),

  /**
   * Source breakdown statistics
   */
  sources: z
    .object({
      ragbits: z
        .object({
          queried: z.boolean(),
          success: z.boolean(),
          resultCount: z.number(),
          avgScore: z.number().optional(),
        })
        .optional(),
      graphiti: z
        .object({
          queried: z.boolean(),
          success: z.boolean(),
          resultCount: z.number(),
          avgScore: z.number().optional(),
        })
        .optional(),
      vectordb: z
        .object({
          queried: z.boolean(),
          success: z.boolean(),
          resultCount: z.number(),
          avgScore: z.number().optional(),
        })
        .optional(),
    })
    .optional()
    .describe('Per-source statistics'),

  /**
   * Overall confidence score
   */
  confidence: z
    .number()
    .min(0)
    .max(1)
    .describe('Overall confidence in results'),

  /**
   * Retrieval metadata
   */
  metadata: z
    .object({
      correlationId: z.string().describe('Correlation ID for tracing'),
      retrievalTimestamp: z.date().describe('UTC timestamp of retrieval'),
      processingTime: z.number().describe('Processing time in milliseconds'),
      totalSourcesQueried: z.number().describe('Number of sources queried'),
      successfulSources: z.number().describe('Number of successful queries'),
    })
    .optional(),
});

type KnowledgeRetrievalResult = z.infer<typeof KnowledgeRetrievalResultSchema>;
type KnowledgeResult = z.infer<typeof KnowledgeResultSchema>;
type MergedResult = z.infer<typeof MergedResultSchema>;

/**
 * Knowledge Retrieval Workflow
 *
 * Retrieves relevant knowledge from multiple knowledge sources with
 * intelligent merging and ranking.
 */
export class KnowledgeRetrievalWorkflow extends WorkflowBubble<
  KnowledgeRetrievalParams,
  KnowledgeRetrievalResult
> {
  static readonly type = 'workflow' as const;
  static readonly bubbleName = 'knowledge-retrieval' as const;
  static readonly schema = KnowledgeRetrievalParamsSchema;
  static readonly resultSchema = KnowledgeRetrievalResultSchema;
  static readonly shortDescription =
    'Retrieve knowledge from RAGBits, Graphiti, and Vector DB';
  static readonly longDescription = `
    Retrieves relevant knowledge from multiple knowledge sources and merges results.

    Features:
    - Multi-source knowledge retrieval (RAGBits, Graphiti, Vector DB)
    - Semantic search with configurable similarity thresholds
    - Intelligent result merging and ranking
    - Circuit breaker pattern for fault tolerance
    - Comprehensive source statistics and confidence scoring

    Use cases:
    - Retrieving relevant context for AI workflows
    - Finding similar historical executions
    - Entity and relationship graph queries
    - Document search and retrieval

    Process:
    1. Query enabled knowledge sources in parallel
    2. Collect and normalize results from each source
    3. Merge and rank results by aggregated relevance
    4. Return top-k results with confidence scores
  `;
  static readonly alias = 'retrieve-knowledge';

  // Circuit breakers for each knowledge source
  private ragbitsBreaker: CircuitBreaker;
  private graphitiBreaker: CircuitBreaker;
  private vectordbBreaker: CircuitBreaker;

  constructor(
    params: KnowledgeRetrievalParams,
    context?: BubbleContext,
    instanceId?: string
  ) {
    super(params, context, instanceId);

    // Initialize circuit breakers for each knowledge source
    this.ragbitsBreaker = new CircuitBreaker('ragbits-knowledge', {
      failureThreshold: 3,
      timeout: 30000,
      halfOpenAttempts: 2,
    });

    this.graphitiBreaker = new CircuitBreaker('graphiti-knowledge', {
      failureThreshold: 3,
      timeout: 30000,
      halfOpenAttempts: 2,
    });

    this.vectordbBreaker = new CircuitBreaker('vectordb-knowledge', {
      failureThreshold: 3,
      timeout: 30000,
      halfOpenAttempts: 2,
    });
  }

  protected async performAction(): Promise<KnowledgeRetrievalResult> {
    const startTime = Date.now();
    const correlationId = this.generateCorrelationId();

    console.log(`[KnowledgeRetrieval] Starting knowledge retrieval`);
    console.log(`[KnowledgeRetrieval] Query: ${this.params.query.substring(0, 100)}...`);
    console.log(`[KnowledgeRetrieval] Correlation ID: ${correlationId}`);

    // Validate environment configuration (Law of Configuration Explicitness)
    const endpoints = this.validateAndGetEndpoints();

    const sources = this.params.sources || {};
    const options = this.params.options || {};
    const allResults: KnowledgeResult[] = [];
    const sourceStats: NonNullable<KnowledgeRetrievalResult['sources']> = {};
    let totalSourcesQueried = 0;
    let successfulSources = 0;

    try {
      // Step 1: Query RAGBits for relevant documents
      if (sources.ragbits !== false && endpoints.ragbits) {
        console.log('[KnowledgeRetrieval] Querying RAGBits...');
        totalSourcesQueried++;

        const ragbitsResult = await this.queryRAGBits(
          endpoints.ragbits,
          this.params.query,
          options
        );

        sourceStats.ragbits = {
          queried: true,
          success: ragbitsResult.success,
          resultCount: ragbitsResult.results.length,
          avgScore: this.calculateAverageScore(ragbitsResult.results),
        };

        if (ragbitsResult.success) {
          allResults.push(...ragbitsResult.results);
          successfulSources++;
        }
      }

      // Step 2: Query Graphiti for relevant entities/relationships
      if (sources.graphiti !== false && endpoints.graphiti) {
        console.log('[KnowledgeRetrieval] Querying Graphiti...');
        totalSourcesQueried++;

        const graphitiResult = await this.queryGraphiti(
          endpoints.graphiti,
          this.params.query,
          options
        );

        sourceStats.graphiti = {
          queried: true,
          success: graphitiResult.success,
          resultCount: graphitiResult.results.length,
          avgScore: this.calculateAverageScore(graphitiResult.results),
        };

        if (graphitiResult.success) {
          allResults.push(...graphitiResult.results);
          successfulSources++;
        }
      }

      // Step 3: Query Vector DB for similar historical executions
      if (sources.vectordb !== false && endpoints.vectordb) {
        console.log('[KnowledgeRetrieval] Querying Vector DB...');
        totalSourcesQueried++;

        const vectorResult = await this.queryVectorDB(
          endpoints.vectordb,
          this.params.query,
          options
        );

        sourceStats.vectordb = {
          queried: true,
          success: vectorResult.success,
          resultCount: vectorResult.results.length,
          avgScore: this.calculateAverageScore(vectorResult.results),
        };

        if (vectorResult.success) {
          allResults.push(...vectorResult.results);
          successfulSources++;
        }
      }

      // Step 4: Merge and rank knowledge results
      console.log('[KnowledgeRetrieval] Merging and ranking results...');
      const merged = await this.mergeKnowledgeResults(allResults);

      // Step 5: Return top-k results
      const topK = merged.slice(0, this.params.maxResults || 10);
      const confidence = this.calculateOverallConfidence(topK, successfulSources, totalSourcesQueried);

      const processingTime = Date.now() - startTime;

      console.log(`[KnowledgeRetrieval] Retrieved ${topK.length} results in ${processingTime}ms`);
      console.log(`[KnowledgeRetrieval] Overall confidence: ${confidence.toFixed(2)}`);

      return {
        success: true,
        error: undefined,
        query: this.params.query,
        results: topK,
        sources: sourceStats,
        confidence,
        metadata: {
          correlationId,
          retrievalTimestamp: new Date(), // UTC timestamp (Law of UTC)
          processingTime,
          totalSourcesQueried,
          successfulSources,
        },
      };
    } catch (error) {
      const processingTime = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      console.error('[KnowledgeRetrieval] Workflow failed:', errorMessage);

      return {
        success: false,
        error: `Knowledge retrieval failed: ${errorMessage}`,
        query: this.params.query,
        results: [],
        sources: sourceStats,
        confidence: 0,
        metadata: {
          correlationId,
          retrievalTimestamp: new Date(),
          processingTime,
          totalSourcesQueried,
          successfulSources,
        },
      };
    }
  }

  /**
   * Validate and get knowledge source endpoints from environment
   * Follows Law of Configuration Explicitness - crash if missing required config
   */
  private validateAndGetEndpoints(): NonNullable<KnowledgeRetrievalParams['endpoints']> {
    const endpoints: Record<string, string> = {};

    // Check environment variables for knowledge source URLs
    if (process.env.RAGBITS_URL) {
      endpoints.ragbits = process.env.RAGBITS_URL;
    } else if (this.params.endpoints?.ragbits) {
      endpoints.ragbits = this.params.endpoints.ragbits;
    }

    if (process.env.GRAPHITI_URL) {
      endpoints.graphiti = process.env.GRAPHITI_URL;
    } else if (this.params.endpoints?.graphiti) {
      endpoints.graphiti = this.params.endpoints.graphiti;
    }

    if (process.env.VECTORDB_URL) {
      endpoints.vectordb = process.env.VECTORDB_URL;
    } else if (this.params.endpoints?.vectordb) {
      endpoints.vectordb = this.params.endpoints.vectordb;
    }

    if (Object.keys(endpoints).length === 0) {
      throw new ValidationError(
        'No knowledge source endpoints configured. Please set RAGBITS_URL, GRAPHITI_URL, or VECTORDB_URL environment variables, or provide endpoints in params.',
        undefined,
        { providedEndpoints: this.params.endpoints }
      );
    }

    return endpoints as any;
  }

  /**
   * Query RAGBits for relevant documents
   */
  private async queryRAGBits(
    endpoint: string,
    query: string,
    options: z.infer<typeof QueryOptionsSchema>
  ): Promise<{ success: boolean; results: KnowledgeResult[] }> {
    return this.ragbitsBreaker.execute(async () => {
      try {
        const httpBubble = new HttpBubble(
          {
            url: `${endpoint}/search`,
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: {
              query,
              top_k: options.topK,
              filters: options.filters,
              min_score: options.minScore,
            },
            timeout: options.timeout,
            credentials: this.params.credentials,
          },
          this.context
        );

        const result = await httpBubble.action();

        if (!result.success || !result.data.json) {
          console.warn('[KnowledgeRetrieval] RAGBits query failed');
          return { success: false, results: [] };
        }

        const data = result.data.json as any;
        const results: KnowledgeResult[] = (data.results || []).map((item: any) => ({
          content: item.content || '',
          source: 'ragbits' as const,
          score: item.score || 0,
          metadata: item.metadata || {},
        }));

        console.log(`[KnowledgeRetrieval] RAGBits returned ${results.length} results`);
        return { success: true, results };
      } catch (error) {
        console.error('[KnowledgeRetrieval] RAGBits error:', error);
        throw new NetworkError(
          `RAGBits query failed: ${error instanceof Error ? error.message : 'Unknown error'}`
        );
      }
    });
  }

  /**
   * Query Graphiti for relevant entities/relationships
   */
  private async queryGraphiti(
    endpoint: string,
    query: string,
    options: z.infer<typeof QueryOptionsSchema>
  ): Promise<{ success: boolean; results: KnowledgeResult[] }> {
    return this.graphitiBreaker.execute(async () => {
      try {
        const httpBubble = new HttpBubble(
          {
            url: `${endpoint}/search`,
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: {
              query,
              top_k: options.topK,
              filters: options.filters,
              min_score: options.minScore,
            },
            timeout: options.timeout,
            credentials: this.params.credentials,
          },
          this.context
        );

        const result = await httpBubble.action();

        if (!result.success || !result.data.json) {
          console.warn('[KnowledgeRetrieval] Graphiti query failed');
          return { success: false, results: [] };
        }

        const data = result.data.json as any;
        const results: KnowledgeResult[] = (data.entities || data.results || []).map((item: any) => ({
          content: item.description || item.name || JSON.stringify(item),
          source: 'graphiti' as const,
          score: item.score || item.relevance || 0,
          metadata: {
            entityType: item.entity_type,
            relationships: item.relationships,
          },
        }));

        console.log(`[KnowledgeRetrieval] Graphiti returned ${results.length} results`);
        return { success: true, results };
      } catch (error) {
        console.error('[KnowledgeRetrieval] Graphiti error:', error);
        throw new NetworkError(
          `Graphiti query failed: ${error instanceof Error ? error.message : 'Unknown error'}`
        );
      }
    });
  }

  /**
   * Query Vector DB for similar historical executions
   */
  private async queryVectorDB(
    endpoint: string,
    query: string,
    options: z.infer<typeof QueryOptionsSchema>
  ): Promise<{ success: boolean; results: KnowledgeResult[] }> {
    return this.vectordbBreaker.execute(async () => {
      try {
        const httpBubble = new HttpBubble(
          {
            url: `${endpoint}/search`,
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: {
              query,
              top_k: options.topK,
              filters: options.filters,
              min_score: options.minScore,
            },
            timeout: options.timeout,
            credentials: this.params.credentials,
          },
          this.context
        );

        const result = await httpBubble.action();

        if (!result.success || !result.data.json) {
          console.warn('[KnowledgeRetrieval] Vector DB query failed');
          return { success: false, results: [] };
        }

        const data = result.data.json as any;
        const results: KnowledgeResult[] = (data.results || []).map((item: any) => ({
          content: item.execution_context || item.content || '',
          source: 'vectordb' as const,
          score: item.similarity || item.score || 0,
          metadata: {
            executionId: item.execution_id,
            timestamp: item.timestamp,
            outcome: item.outcome,
          },
        }));

        console.log(`[KnowledgeRetrieval] Vector DB returned ${results.length} results`);
        return { success: true, results };
      } catch (error) {
        console.error('[KnowledgeRetrieval] Vector DB error:', error);
        throw new NetworkError(
          `Vector DB query failed: ${error instanceof Error ? error.message : 'Unknown error'}`
        );
      }
    });
  }

  /**
   * Merge knowledge results from multiple sources
   * Implements reciprocal rank fusion for intelligent merging
   */
  private async mergeKnowledgeResults(
    results: KnowledgeResult[]
  ): Promise<MergedResult[]> {
    // Group results by content (deduplication)
    const contentMap = new Map<string, MergedResult>();

    for (const result of results) {
      const key = this.getContentKey(result.content);

      if (contentMap.has(key)) {
        // Merge with existing result
        const existing = contentMap.get(key)!;
        existing.sources.push(result.source);
        existing.aggregatedScore += result.score;
        if (result.metadata) {
          existing.metadata = { ...existing.metadata, ...result.metadata };
        }
      } else {
        // Create new merged result
        contentMap.set(key, {
          content: result.content,
          sources: [result.source],
          aggregatedScore: result.score,
          rank: 0,
          metadata: result.metadata,
        });
      }
    }

    // Convert to array and sort by aggregated score
    const merged = Array.from(contentMap.values());

    // Normalize scores (divide by number of sources)
    for (const item of merged) {
      item.aggregatedScore = item.aggregatedScore / item.sources.length;
    }

    // Sort by aggregated score (descending)
    merged.sort((a, b) => b.aggregatedScore - a.aggregatedScore);

    // Assign ranks
    merged.forEach((item, index) => {
      item.rank = index + 1;
    });

    return merged;
  }

  /**
   * Generate content key for deduplication
   */
  private getContentKey(content: string): string {
    // Simple normalization: lowercase, trim, remove extra spaces
    return content.toLowerCase().trim().replace(/\s+/g, ' ');
  }

  /**
   * Calculate average score from results
   */
  private calculateAverageScore(results: KnowledgeResult[]): number {
    if (results.length === 0) return 0;
    const sum = results.reduce((acc, r) => acc + r.score, 0);
    return sum / results.length;
  }

  /**
   * Calculate overall confidence score
   * Considers result quality and source success rate
   */
  private calculateOverallConfidence(
    results: MergedResult[],
    successfulSources: number,
    totalSources: number
  ): number {
    if (results.length === 0) return 0;

    // Average score of top results
    const avgScore = results.reduce((acc, r) => acc + r.aggregatedScore, 0) / results.length;

    // Source success rate
    const sourceSuccessRate = totalSources > 0 ? successfulSources / totalSources : 0;

    // Combine: 70% result quality, 30% source success
    return (avgScore * 0.7) + (sourceSuccessRate * 0.3);
  }

  /**
   * Generate correlation ID for tracing
   */
  private generateCorrelationId(): string {
    return `kr-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  }
}
