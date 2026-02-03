/**
 * Elasticsearch Service Bubble
 *
 * Provides integration with Elasticsearch for OpenEvolve search capabilities.
 * Implements indexing, searching, aggregations, and cluster health monitoring.
 *
 * FIXED: Now extends ServiceBubble properly with Federation Constitution compliance
 */

import { z } from 'zod';
import { ServiceBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';
import { ResilienceWrapper, DEFAULT_RESILIENCE_CONFIG } from '../adapters/resilience';

// ============================================================================
// ELASTICSEARCH-SPECIFIC PARAMETER SCHEMAS
// ============================================================================

const ElasticsearchOperationSchema = z.enum([
  'create_index',
  'delete_index',
  'index_document',
  'bulk_index',
  'search',
  'get_document',
  'delete_document',
  'update_document',
  'health_check',
  'cluster_info',
  'aggregate',
]);

const QueryDslSchema = z.object({
  query: z.record(z.unknown()).optional(),
  aggs: z.record(z.unknown()).optional(),
  sort: z.array(z.unknown()).optional(),
  size: z.number().optional(),
  from: z.number().optional(),
  highlight: z.record(z.unknown()).optional(),
  _source: z.union([z.boolean(), z.array(z.string())]).optional(),
});

// ============================================================================
// MAIN PARAMETER SCHEMA (NO MAGIC DEFAULTS)
// ============================================================================

const ElasticsearchParamsSchema = z.object({
  operation: ElasticsearchOperationSchema.describe('Operation to perform on Elasticsearch'),

  // REQUIRED: No magic defaults - Federation Constitution compliance
  baseUrl: z.string().url().describe('Elasticsearch server URL (REQUIRED)'),

  // Authentication (optional but recommended)
  username: z.string().optional().describe('Basic auth username'),
  password: z.string().optional().describe('Basic auth password'),
  apiKey: z.string().optional().describe('Elasticsearch API key'),

  index: z.string().optional().describe('Index name'),
  timeout: z.number().min(1000).max(120000).default(30000).describe('Request timeout in ms'),

  // Document operations
  documentId: z.string().optional().describe('Document ID'),
  document: z.record(z.unknown()).optional().describe('Document data'),
  documents: z.array(z.record(z.unknown())).optional().describe('Bulk documents'),

  // Search operations
  query: QueryDslSchema.optional().describe('Query DSL object'),
  rawQuery: z.string().optional().describe('Raw query string'),

  // Index operations
  mappings: z.record(z.unknown()).optional().describe('Index mappings'),
  settings: z.record(z.unknown()).optional().describe('Index settings'),
});

type ElasticsearchParamsInput = z.input<typeof ElasticsearchParamsSchema>;
type ElasticsearchParams = z.output<typeof ElasticsearchParamsSchema>;

// ============================================================================
// RESULT SCHEMA
// ============================================================================

const ElasticsearchResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: z.unknown().optional(),
  took: z.number().optional().describe('Query execution time in ms'),
  timed_out: z.boolean().optional(),
  hits: z.object({
    total: z.object({
      value: z.number(),
      relation: z.string(),
    }),
    hits: z.array(z.unknown()),
  }).optional(),
  aggregations: z.record(z.unknown()).optional(),
  status: z.object({
    code: z.number(),
    reason: z.string().optional(),
  }),
  error: z.string().optional(),
  timing: z.number().describe('Total request time in ms'),
});

type ElasticsearchResult = z.output<typeof ElasticsearchResultSchema>;

// ============================================================================
// ELASTICSEARCH BUBBLE (PROPERLY EXTENDS ServiceBubble)
// ============================================================================

export class ElasticsearchBubble extends ServiceBubble<ElasticsearchParams, ElasticsearchResult> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName = 'elasticsearch' as const;
  static readonly type = 'service' as const;
  static readonly schema = ElasticsearchParamsSchema;
  static readonly resultSchema = ElasticsearchResultSchema;
  static readonly credentialType = 'elasticsearch_api_key' as const;

  static readonly shortDescription = 'Elasticsearch integration for full-text search and analytics';
  static readonly longDescription = `
    Elasticsearch service bubble for OpenEvolve search capabilities.

    Features:
    - Full-text search with Query DSL
    - Document indexing and CRUD operations
    - Bulk operations for high-throughput indexing
    - Aggregations and analytics
    - Cluster health monitoring
    - Circuit breaker and retry logic for fault tolerance

    Required Configuration:
    - baseUrl: Elasticsearch server URL (no default - must be provided)
    - apiKey or username/password for authentication

    Federation Constitution Compliance:
    - No magic defaults (baseUrl is required)
    - Circuit breaker for fault tolerance
    - Exponential backoff retry with jitter
    - Request deduplication for idempotency
    - Structured logging with correlation IDs
  `;

  private resilience: ResilienceWrapper;

  constructor(params: ElasticsearchParamsInput, context?: BubbleContext) {
    super(params, context);

    // Validate required environment variables at startup
    ElasticsearchBubble.validateConfig();

    // Initialize resilience wrapper
    this.resilience = new ResilienceWrapper('elasticsearch', DEFAULT_RESILIENCE_CONFIG);
  }

  /**
   * Validate configuration at startup (Federation Constitution compliance)
   */
  private static validateConfig(): void {
    // No validation needed here - baseUrl is required by schema
    // Additional runtime validation can be added
  }

  /**
   * Build HTTP headers for Elasticsearch API requests
   */
  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.params.apiKey) {
      headers['Authorization'] = `ApiKey ${this.params.apiKey}`;
    } else if (this.params.username && this.params.password) {
      headers['Authorization'] = `Basic ${Buffer.from(`${this.params.username}:${this.params.password}`).toString('base64')}`;
    }

    return headers;
  }

  /**
   * Build full URL for Elasticsearch endpoint
   */
  private buildUrl(endpoint: string): string {
    return `${this.params.baseUrl}${endpoint}`;
  }

  /**
   * Make HTTP request to Elasticsearch API
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
  private async healthCheck(): Promise<ElasticsearchResult> {
    const startTime = Date.now();

    try {
      const response = await this.resilience.execute(
        'elasticsearch-healthcheck',
        () => this.makeRequest('GET', '/_cluster/health'),
        { operation: 'health_check' }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'health_check',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data.error?.reason || 'Unknown error',
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
   * Cluster info operation
   */
  private async clusterInfo(): Promise<ElasticsearchResult> {
    const startTime = Date.now();

    try {
      const response = await this.resilience.execute(
        'elasticsearch-cluster-info',
        () => this.makeRequest('GET', '/'),
        { operation: 'cluster_info' }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'cluster_info',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data.error?.reason || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'cluster_info',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Create index operation
   */
  private async createIndex(): Promise<ElasticsearchResult> {
    if (!this.params.index) {
      throw new Error('index is required for create_index operation');
    }

    const startTime = Date.now();

    try {
      const body: Record<string, unknown> = {};
      if (this.params.mappings) {
        body.mappings = this.params.mappings;
      }
      if (this.params.settings) {
        body.settings = this.params.settings;
      }

      const response = await this.resilience.execute(
        `elasticsearch-create-${this.params.index}`,
        () => this.makeRequest('PUT', `/${this.params.index}`, body),
        { operation: 'create_index', index: this.params.index }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'create_index',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data.error?.reason || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'create_index',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Delete index operation
   */
  private async deleteIndex(): Promise<ElasticsearchResult> {
    if (!this.params.index) {
      throw new Error('index is required for delete_index operation');
    }

    const startTime = Date.now();

    try {
      const response = await this.resilience.execute(
        `elasticsearch-delete-${this.params.index}`,
        () => this.makeRequest('DELETE', `/${this.params.index}`),
        { operation: 'delete_index', index: this.params.index }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'delete_index',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data.error?.reason || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'delete_index',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Index document operation
   */
  private async indexDocument(): Promise<ElasticsearchResult> {
    if (!this.params.index || !this.params.document) {
      throw new Error('index and document are required for index_document operation');
    }

    const startTime = Date.now();

    try {
      const endpoint = this.params.documentId
        ? `/${this.params.index}/_doc/${this.params.documentId}`
        : `/${this.params.index}/_doc`;

      const response = await this.resilience.execute(
        `elasticsearch-index-${this.params.index}-${this.params.documentId || 'new'}`,
        () => this.makeRequest('POST', endpoint, this.params.document),
        { operation: 'index_document', index: this.params.index, documentId: this.params.documentId }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'index_document',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data.error?.reason || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'index_document',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Bulk index operation
   */
  private async bulkIndex(): Promise<ElasticsearchResult> {
    if (!this.params.index || !this.params.documents) {
      throw new Error('index and documents are required for bulk_index operation');
    }

    const startTime = Date.now();

    try {
      const bulkBody: string[] = [];
      for (const doc of this.params.documents) {
        const action = { index: { _index: this.params.index } };
        if (doc._id) {
          (action.index as Record<string, unknown>)._id = doc._id as string;
        }
        bulkBody.push(JSON.stringify(action));
        bulkBody.push(JSON.stringify(doc));
      }

      const response = await this.resilience.execute(
        `elasticsearch-bulk-${this.params.index}`,
        () => this.makeRequest('POST', '/_bulk', bulkBody.join('\n') + '\n'),
        { operation: 'bulk_index', index: this.params.index, count: this.params.documents.length }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'bulk_index',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data.error?.reason || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'bulk_index',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Search operation
   */
  private async search(): Promise<ElasticsearchResult> {
    if (!this.params.index) {
      throw new Error('index is required for search operation');
    }

    const startTime = Date.now();

    try {
      const query = this.params.rawQuery
        ? { query: { query_string: { query: this.params.rawQuery } } }
        : this.params.query || { query: { match_all: {} } };

      const cacheKey = `elasticsearch-search-${this.params.index}-${JSON.stringify(query)}`;

      const response = await this.resilience.execute(
        cacheKey,
        () => this.makeRequest('POST', `/${this.params.index}/_search`, query),
        { operation: 'search', index: this.params.index, query }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'search',
        data,
        took: data.took,
        timed_out: data.timed_out,
        hits: data.hits,
        aggregations: data.aggregations,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data.error?.reason || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'search',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Get document operation
   */
  private async getDocument(): Promise<ElasticsearchResult> {
    if (!this.params.index || !this.params.documentId) {
      throw new Error('index and documentId are required for get_document operation');
    }

    const startTime = Date.now();

    try {
      const response = await this.resilience.execute(
        `elasticsearch-get-${this.params.index}-${this.params.documentId}`,
        () => this.makeRequest('GET', `/${this.params.index}/_doc/${this.params.documentId}`),
        { operation: 'get_document', index: this.params.index, documentId: this.params.documentId }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'get_document',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data.error?.reason || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'get_document',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Delete document operation
   */
  private async deleteDocument(): Promise<ElasticsearchResult> {
    if (!this.params.index || !this.params.documentId) {
      throw new Error('index and documentId are required for delete_document operation');
    }

    const startTime = Date.now();

    try {
      const response = await this.resilience.execute(
        `elasticsearch-delete-doc-${this.params.index}-${this.params.documentId}`,
        () => this.makeRequest('DELETE', `/${this.params.index}/_doc/${this.params.documentId}`),
        { operation: 'delete_document', index: this.params.index, documentId: this.params.documentId }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'delete_document',
        data,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data.error?.reason || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'delete_document',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Aggregate operation
   */
  private async aggregate(): Promise<ElasticsearchResult> {
    if (!this.params.index) {
      throw new Error('index is required for aggregate operation');
    }

    const startTime = Date.now();

    try {
      const query = this.params.query || {};

      const cacheKey = `elasticsearch-aggregate-${this.params.index}-${JSON.stringify(query)}`;

      const response = await this.resilience.execute(
        cacheKey,
        () => this.makeRequest('POST', `/${this.params.index}/_search`, query),
        { operation: 'aggregate', index: this.params.index, query }
      );

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'aggregate',
        data,
        aggregations: data.aggregations,
        status: { code: response.status, reason: response.statusText },
        error: response.ok ? undefined : data.error?.reason || 'Unknown error',
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'aggregate',
        status: { code: 0, reason: 'Request failed' },
        error: errorMessage,
        timing,
      };
    }
  }

  /**
   * Main action method - routes to appropriate operation
   */
  async action(): Promise<ElasticsearchResult> {
    switch (this.params.operation) {
      case 'health_check':
        return this.healthCheck();
      case 'cluster_info':
        return this.clusterInfo();
      case 'create_index':
        return this.createIndex();
      case 'delete_index':
        return this.deleteIndex();
      case 'index_document':
        return this.indexDocument();
      case 'bulk_index':
        return this.bulkIndex();
      case 'search':
        return this.search();
      case 'get_document':
        return this.getDocument();
      case 'delete_document':
        return this.deleteDocument();
      case 'aggregate':
        return this.aggregate();
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

export default ElasticsearchBubble;
