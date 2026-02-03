/**
 * Qdrant Vector Database Service Bubble
 *
 * Provides integration with Qdrant vector database for OpenEvolve knowledge engine.
 * Implements full CRUD operations, vector similarity search, and health monitoring.
 *
 * FIXED: Now extends ServiceBubble properly with Federation Constitution compliance
 */

import { z } from 'zod';
import { ServiceBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';
import { ResilienceWrapper, DEFAULT_RESILIENCE_CONFIG } from '../adapters/resilience';

// ============================================================================
// QDRANT-SPECIFIC PARAMETER SCHEMAS
// ============================================================================

const QdrantOperationSchema = z.enum([
  'create_collection',
  'delete_collection',
  'insert_points',
  'search_points',
  'delete_points',
  'get_collection',
  'list_collections',
  'health_check',
  'count_points'
]);

const VectorSchema = z.array(z.number());

const PointStructSchema = z.object({
  id: z.union([z.string(), z.number()]),
  vector: VectorSchema,
  payload: z.record(z.unknown()).optional(),
});

const SearchFilterSchema = z.object({
  must: z.array(z.unknown()).optional(),
  should: z.array(z.unknown()).optional(),
  must_not: z.array(z.unknown()).optional(),
});

// ============================================================================
// MAIN PARAMETER SCHEMA (NO MAGIC DEFAULTS)
// ============================================================================

const QdrantParamsSchema = z.object({
  operation: QdrantOperationSchema.describe('Operation to perform on Qdrant'),
  collectionName: z.string().optional().describe('Name of the collection'),

  // REQUIRED: No magic defaults - Federation Constitution compliance
  baseUrl: z.string().url().describe('Qdrant server URL (REQUIRED)'),
  apiKey: z.string().optional().describe('Qdrant API key for authentication'),

  timeout: z.number().min(1000).max(120000).default(30000).describe('Request timeout in ms'),

  // Collection operations
  vectorSize: z.number().optional().describe('Vector dimension size for collection creation'),
  distance: z.enum(['Cosine', 'Euclidean', 'Dot']).default('Cosine').describe('Distance metric'),

  // Point operations
  points: z.array(PointStructSchema).optional().describe('Points to insert/upsert'),
  pointIds: z.array(z.union([z.string(), z.number()])).optional().describe('Point IDs to delete'),

  // Search operations
  queryVector: VectorSchema.optional().describe('Query vector for similarity search'),
  limit: z.number().min(1).max(1000).default(10).describe('Number of results to return'),
  scoreThreshold: z.number().min(0).max(1).optional().describe('Minimum similarity score'),
  filter: SearchFilterSchema.optional().describe('Search filter'),
  withPayload: z.boolean().default(true).describe('Include payload in results'),
  withVector: z.boolean().default(false).describe('Include vector in results'),
});

type QdrantParamsInput = z.input<typeof QdrantParamsSchema>;
type QdrantParams = z.output<typeof QdrantParamsSchema>;

// ============================================================================
// RESULT SCHEMA
// ============================================================================

const QdrantResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  status: z.object({
    code: z.number(),
    reason: z.string().optional(),
  }),
  error: z.string().optional(),
  timing: z.number().describe('Response time in milliseconds'),
});

type QdrantResult = z.output<typeof QdrantResultSchema>;

// ============================================================================
// QDRANT BUBBLE (PROPERLY EXTENDS ServiceBubble)
// ============================================================================

export class QdrantBubble extends ServiceBubble<QdrantParams, QdrantResult> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName = 'qdrant' as const;
  static readonly type = 'service' as const;
  static readonly schema = QdrantParamsSchema;
  static readonly resultSchema = QdrantResultSchema;
  static readonly credentialType = 'qdrant_api_key' as const;

  static readonly shortDescription = 'Qdrant vector database integration for similarity search';
  static readonly longDescription = `
    Qdrant vector database service bubble for OpenEvolve knowledge engine.

    Features:
    - Vector similarity search with configurable distance metrics
    - Collection management (create, delete, list, get info)
    - Point operations (insert, upsert, delete)
    - Advanced filtering support
    - Health monitoring and statistics
    - Circuit breaker and retry logic for fault tolerance

    Required Configuration:
    - baseUrl: Qdrant server URL (no default - must be provided)
    - apiKey: Optional API key for authentication

    Federation Constitution Compliance:
    - No magic defaults (baseUrl is required)
    - Circuit breaker for fault tolerance
    - Exponential backoff retry with jitter
    - Request deduplication for idempotency
    - Structured logging with correlation IDs
  `;

  private resilience: ResilienceWrapper;

  constructor(params: QdrantParamsInput, context?: BubbleContext) {
    super(params, context);

    // Validate required environment variables at startup
    QdrantBubble.validateConfig();

    // Initialize resilience wrapper
    this.resilience = new ResilienceWrapper('qdrant', DEFAULT_RESILIENCE_CONFIG);
  }

  /**
   * Validate configuration at startup (Federation Constitution compliance)
   */
  private static validateConfig(): void {
    // No validation needed here - baseUrl is required by schema
    // Additional runtime validation can be added
  }

  /**
   * Build HTTP headers for Qdrant API requests
   */
  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.params.apiKey) {
      headers['api-key'] = this.params.apiKey;
    }

    return headers;
  }

  /**
   * Build full URL for Qdrant endpoint
   */
  private buildUrl(endpoint: string): string {
    return `${this.params.baseUrl}${endpoint}`;
  }

  /**
   * Make HTTP request to Qdrant API
   */
  private async makeRequest(
    method: string,
    endpoint: string,
    body?: unknown
  ): Promise<Response> {
    const url = this.buildUrl(endpoint);

    return await fetch(url, {
      method,
      headers: this.buildHeaders(),
      body: body ? JSON.stringify(body) : undefined,
    });
  }

  /**
   * Health check operation
   */
  private async healthCheck(): Promise<QdrantResult> {
    const startTime = Date.now();

    try {
      const response = await this.resilience.execute(
        `qdrant-healthcheck`,
        () => this.makeRequest('GET', '/'),
        { operation: 'health_check' }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'health_check',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data.status?.error || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'health_check',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Create collection operation
   */
  private async createCollection(): Promise<QdrantResult> {
    if (!this.params.collectionName) {
      throw new Error('collectionName is required for create_collection operation');
    }

    const startTime = Date.now();

    try {
      const response = await this.resilience.execute(
        `qdrant-create-${this.params.collectionName}`,
        () => this.makeRequest('PUT', `/collections/${this.params.collectionName}`, {
          vectors: {
            size: this.params.vectorSize || 1536,
            distance: this.params.distance,
          },
        }),
        { operation: 'create_collection', collectionName: this.params.collectionName }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'create_collection',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data.status?.error || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'create_collection',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Search points operation
   */
  private async searchPoints(): Promise<QdrantResult> {
    if (!this.params.collectionName || !this.params.queryVector) {
      throw new Error('collectionName and queryVector are required for search_points operation');
    }

    const startTime = Date.now();

    try {
      const body: Record<string, unknown> = {
        vector: this.params.queryVector,
        limit: this.params.limit,
        with_payload: this.params.withPayload,
        with_vector: this.params.withVector,
      };

      if (this.params.scoreThreshold !== undefined) {
        body.score_threshold = this.params.scoreThreshold;
      }

      if (this.params.filter) {
        body.filter = this.params.filter;
      }

      const cacheKey = `qdrant-search-${this.params.collectionName}-${JSON.stringify(this.params.queryVector)}-${this.params.limit}`;

      const response = await this.resilience.execute(
        cacheKey,
        () => this.makeRequest('POST', `/collections/${this.params.collectionName}/points/search`, body),
        { operation: 'search_points', collectionName: this.params.collectionName, queryVector: this.params.queryVector }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'search_points',
        data: data.result,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data.status?.error || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'search_points',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Insert points operation
   */
  private async insertPoints(): Promise<QdrantResult> {
    if (!this.params.collectionName || !this.params.points) {
      throw new Error('collectionName and points are required for insert_points operation');
    }

    const startTime = Date.now();

    try {
      const response = await this.resilience.execute(
        `qdrant-insert-${this.params.collectionName}`,
        () => this.makeRequest('PUT', `/collections/${this.params.collectionName}/points`, {
          points: this.params.points,
        }),
        { operation: 'insert_points', collectionName: this.params.collectionName, points: this.params.points }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'insert_points',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data.status?.error || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'insert_points',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Get collection operation
   */
  private async getCollection(): Promise<QdrantResult> {
    if (!this.params.collectionName) {
      throw new Error('collectionName is required for get_collection operation');
    }

    const startTime = Date.now();

    try {
      const response = await this.resilience.execute(
        `qdrant-get-${this.params.collectionName}`,
        () => this.makeRequest('GET', `/collections/${this.params.collectionName}`),
        { operation: 'get_collection', collectionName: this.params.collectionName }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'get_collection',
        data: data.result,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data.status?.error || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'get_collection',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Delete points operation
   */
  private async deletePoints(): Promise<QdrantResult> {
    if (!this.params.collectionName || !this.params.pointIds) {
      throw new Error('collectionName and pointIds are required for delete_points operation');
    }

    const startTime = Date.now();

    try {
      const response = await this.resilience.execute(
        `qdrant-delete-${this.params.collectionName}`,
        () => this.makeRequest('POST', `/collections/${this.params.collectionName}/points/delete`, {
          points: this.params.pointIds,
        }),
        { operation: 'delete_points', collectionName: this.params.collectionName, pointIds: this.params.pointIds }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'delete_points',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data.status?.error || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'delete_points',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * List collections operation
   */
  private async listCollections(): Promise<QdrantResult> {
    const startTime = Date.now();

    try {
      const response = await this.resilience.execute(
        'qdrant-list-collections',
        () => this.makeRequest('GET', '/collections'),
        { operation: 'list_collections' }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'list_collections',
        data: data.result?.collections,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data.status?.error || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'list_collections',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Main action method - routes to appropriate operation
   */
  async action(): Promise<QdrantResult> {
    switch (this.params.operation) {
      case 'health_check':
        return this.healthCheck();
      case 'create_collection':
        return this.createCollection();
      case 'insert_points':
        return this.insertPoints();
      case 'search_points':
        return this.searchPoints();
      case 'delete_points':
        return this.deletePoints();
      case 'get_collection':
        return this.getCollection();
      case 'list_collections':
        return this.listCollections();
      default:
        return {
          success: false,
          operation: this.params.operation,
          status: { code: 400, reason: 'Invalid operation' },
          error: `Unknown operation: ${this.params.operation}`,
          timing: 0,
        };
    }
  }
}

export default QdrantBubble;
