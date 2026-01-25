import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
import { Client } from '@elastic/elasticsearch';

/**
 * Elasticsearch Bubble - Complete Service Bubble Implementation
 *
 * Full production implementation with 10 operations:
 * 1. createIndex - Create a new index with mappings
 * 2. indexDocument - Add or update a document
 * 3. bulkIndex - Bulk index multiple documents
 * 4. search - Search documents with query DSL
 * 5. getDocument - Retrieve a document by ID
 * 6. updateDocument - Partially update a document
 * 7. deleteDocument - Delete a document by ID
 * 8. deleteIndex - Delete an entire index
 * 9. indexExists - Check if an index exists
 * 10. aggregate - Perform aggregations on data
 */

// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================

const CreateIndexParamsSchema = z.object({
  operation: z.literal('createIndex'),
  indexName: z.string().min(1, 'Index name is required'),
  mappings: z.record(z.unknown()).optional().describe('Index mappings for field types'),
  settings: z.record(z.unknown()).optional().describe('Index settings like shards, replicas'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const IndexDocumentParamsSchema = z.object({
  operation: z.literal('indexDocument'),
  indexName: z.string().min(1, 'Index name is required'),
  documentId: z.string().optional(),
  document: z.record(z.unknown()).describe('Document to index'),
  refresh: z.enum(['true', 'false', 'wait_for']).optional().default('false'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const BulkIndexParamsSchema = z.object({
  operation: z.literal('bulkIndex'),
  indexName: z.string().min(1, 'Index name is required'),
  documents: z.array(z.record(z.unknown())).min(1, 'At least one document is required'),
  refresh: z.enum(['true', 'false', 'wait_for']).optional().default('false'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const SearchParamsSchema = z.object({
  operation: z.literal('search'),
  indexName: z.string().min(1, 'Index name is required'),
  query: z.record(z.unknown()).describe('Elasticsearch query DSL'),
  from: z.number().int().nonnegative().optional().default(0),
  size: z.number().int().positive().optional().default(10),
  sort: z.array(z.unknown()).optional(),
  source: z.boolean().or(z.array(z.string())).optional(),
  aggs: z.record(z.unknown()).optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetDocumentParamsSchema = z.object({
  operation: z.literal('getDocument'),
  indexName: z.string().min(1, 'Index name is required'),
  documentId: z.string().min(1, 'Document ID is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const UpdateDocumentParamsSchema = z.object({
  operation: z.literal('updateDocument'),
  indexName: z.string().min(1, 'Index name is required'),
  documentId: z.string().min(1, 'Document ID is required'),
  doc: z.record(z.unknown()).describe('Partial document updates'),
  refresh: z.enum(['true', 'false', 'wait_for']).optional().default('false'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const DeleteDocumentParamsSchema = z.object({
  operation: z.literal('deleteDocument'),
  indexName: z.string().min(1, 'Index name is required'),
  documentId: z.string().min(1, 'Document ID is required'),
  refresh: z.enum(['true', 'false', 'wait_for']).optional().default('false'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const DeleteIndexParamsSchema = z.object({
  operation: z.literal('deleteIndex'),
  indexName: z.string().min(1, 'Index name is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const IndexExistsParamsSchema = z.object({
  operation: z.literal('indexExists'),
  indexName: z.string().min(1, 'Index name is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const AggregateParamsSchema = z.object({
  operation: z.literal('aggregate'),
  indexName: z.string().min(1, 'Index name is required'),
  aggs: z.record(z.unknown()).describe('Aggregation queries'),
  query: z.record(z.unknown()).optional().describe('Filter query for aggregation'),
  size: z.number().int().nonnegative().optional().default(0),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

// Union of all parameter schemas
const ElasticsearchBubbleParamsSchema = z.discriminatedUnion('operation', [
  CreateIndexParamsSchema,
  IndexDocumentParamsSchema,
  BulkIndexParamsSchema,
  SearchParamsSchema,
  GetDocumentParamsSchema,
  UpdateDocumentParamsSchema,
  DeleteDocumentParamsSchema,
  DeleteIndexParamsSchema,
  IndexExistsParamsSchema,
  AggregateParamsSchema,
]);

type ElasticsearchBubbleParams = z.input<typeof ElasticsearchBubbleParamsSchema>;

// Result schema
const ElasticsearchBubbleResultSchema = z.object({
  success: z.boolean(),
  data: z.unknown().describe('Operation result data'),
  error: z.string(),
  meta: z.object({
    operation: z.string(),
    indexName: z.string().optional(),
    took: z.number().optional(),
  }),
});

type ElasticsearchBubbleResult = z.output<typeof ElasticsearchBubbleResultSchema>;

// ============================================================================
// MAIN BUBBLE CLASS
// ============================================================================

export class ElasticsearchBubble extends ServiceBubble<
  ElasticsearchBubbleParams,
  ElasticsearchBubbleResult
> {
  static readonly service = 'elasticsearch';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'elasticsearch';
  static readonly type = 'service' as const;
  static readonly schema = ElasticsearchBubbleParamsSchema;
  static readonly resultSchema = ElasticsearchBubbleResultSchema;
  static readonly shortDescription =
    'Full-text search and analytics engine with distributed architecture';
  static readonly longDescription = `
    Elasticsearch Bubble for full-text search, logging, and analytics.

    Features:
    - Create and manage indices with custom mappings
    - Index and search documents in near real-time
    - Powerful query DSL for complex searches
    - Bulk operations for high-throughput indexing
    - Aggregations for data analytics
    - Distributed and scalable architecture

    Use cases:
    - Full-text search applications
    - Log analytics and monitoring
    - Metrics and dashboard data
    - Autocomplete and typeahead
    - Geospatial search
  `;
  static readonly alias = 'es';

  private client: Client | null = null;

  constructor(
    params: ElasticsearchBubbleParams,
    context?: BubbleContext,
    instanceId?: string
  ) {
    super(params, context, instanceId);
  }

  protected getCredentialType(): CredentialType {
    return CredentialType.ELASTICSEARCH_CRED;
  }

  protected chooseCredential(): string | undefined {
    const credentials = this.params.credentials;
    if (!credentials || typeof credentials !== 'object') {
      throw new Error('Elasticsearch credentials are required');
    }
    return credentials[CredentialType.ELASTICSEARCH_CRED];
  }

  public async testCredential(): Promise<boolean> {
    try {
      const client = this.getClient();
      await client.ping();
      return true;
    } catch (error) {
      console.error('[Elasticsearch] Credential test failed:', error);
      return false;
    }
  }

  private getClient(): Client {
    if (!this.client) {
      const credential = this.chooseCredential();
      if (!credential) {
        throw new Error('Elasticsearch credentials not found');
      }

      // Parse credential (expected format: JSON string with url, username, password)
      let config: any;
      try {
        config = typeof credential === 'string' ? JSON.parse(credential) : credential;
      } catch {
        throw new Error('Invalid Elasticsearch credentials format. Expected JSON string.');
      }

      if (!config.url) {
        throw new Error('Elasticsearch URL is required in credentials');
      }

      const clientConfig: any = {
        node: config.url,
      };

      // Add authentication if provided
      if (config.username && config.password) {
        clientConfig.auth = {
          username: config.username,
          password: config.password,
        };
      } else if (config.apiKey) {
        clientConfig.auth = {
          apiKey: config.apiKey,
        };
      }

      // Add TLS options for HTTPS
      if (config.url.startsWith('https://')) {
        clientConfig.tls = {
          rejectUnauthorized: config.rejectUnauthorized ?? true,
        };
      }

      this.client = new Client(clientConfig);
      console.log('[Elasticsearch] Client initialized successfully');
    }

    return this.client;
  }

  protected async performAction(
    context?: BubbleContext
  ): Promise<ElasticsearchBubbleResult> {
    void context;
    const startTime = Date.now();

    try {
      const client = this.getClient();
      const operation = this.params.operation;
      let result: any;

      console.log(`[Elasticsearch] Executing operation: ${operation}`);

      switch (operation) {
        case 'createIndex':
          result = await this.createIndex(client);
          break;

        case 'indexDocument':
          result = await this.indexDocument(client);
          break;

        case 'bulkIndex':
          result = await this.bulkIndex(client);
          break;

        case 'search':
          result = await this.search(client);
          break;

        case 'getDocument':
          result = await this.getDocument(client);
          break;

        case 'updateDocument':
          result = await this.updateDocument(client);
          break;

        case 'deleteDocument':
          result = await this.deleteDocument(client);
          break;

        case 'deleteIndex':
          result = await this.deleteIndex(client);
          break;

        case 'indexExists':
          result = await this.indexExists(client);
          break;

        case 'aggregate':
          result = await this.aggregate(client);
          break;

        default:
          throw new Error(`Unknown operation: ${operation}`);
      }

      const took = Date.now() - startTime;

      return {
        success: true,
        data: result,
        error: '', // Empty string for successful operations
        meta: {
          operation,
          indexName: this.extractIndexName(),
          took,
        },
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`[Elasticsearch] Operation failed:`, errorMessage);

      return {
        success: false,
        data: null,
        error: errorMessage,
        meta: {
          operation: this.params.operation,
          indexName: this.extractIndexName(),
        },
      };
    }
  }

  private async createIndex(client: Client): Promise<any> {
    const params = this.params as z.output<typeof CreateIndexParamsSchema>;
    console.log(`[Elasticsearch] Creating index: ${params.indexName}`);

    const indexParams: any = {
      index: params.indexName,
    };

    if (params.mappings) {
      indexParams.mappings = params.mappings;
    }

    if (params.settings) {
      indexParams.settings = params.settings;
    }

    const response = await client.indices.create(indexParams);
    console.log(`[Elasticsearch] Index created successfully: ${params.indexName}`);
    return response;
  }

  private async indexDocument(client: Client): Promise<any> {
    const params = this.params as z.output<typeof IndexDocumentParamsSchema>;

    const indexParams: any = {
      index: params.indexName,
      body: params.document,
      refresh: params.refresh,
    };

    if (params.documentId) {
      indexParams.id = params.documentId;
    }

    const response = await client.index(indexParams);
    console.log(`[Elasticsearch] Document indexed: ${response.result}`);
    return response;
  }

  private async bulkIndex(client: Client): Promise<any> {
    const params = this.params as z.output<typeof BulkIndexParamsSchema>;

    // Build bulk operations array
    const operations: any[] = [];
    for (const doc of params.documents) {
      operations.push({ index: { _index: params.indexName } });
      operations.push(doc);
    }

    const response = await client.bulk({
      operations,
      refresh: params.refresh,
    });

    console.log(
      `[Elasticsearch] Bulk indexed ${response.items.length} documents with ${response.errors ? 'errors' : 'success'}`
    );

    return response;
  }

  private async search(client: Client): Promise<any> {
    const params = this.params as z.output<typeof SearchParamsSchema>;

    const searchParams: any = {
      index: params.indexName,
      body: {
        query: params.query,
        from: params.from,
        size: params.size,
      },
    };

    if (params.sort) {
      searchParams.body.sort = params.sort;
    }

    if (params.source !== undefined) {
      searchParams.body._source = params.source;
    }

    if (params.aggs) {
      searchParams.body.aggs = params.aggs;
    }

    const response = await client.search(searchParams);

    const total = typeof response.hits.total === 'number'
      ? response.hits.total
      : response.hits.total?.value ?? 0;

    console.log(`[Elasticsearch] Search completed: ${total} hits`);

    return {
      took: response.took,
      total,
      hits: response.hits.hits.map((hit: any) => ({
        id: hit._id,
        score: hit._score,
        source: hit._source,
      })),
      aggregations: response.aggregations,
    };
  }

  private async getDocument(client: Client): Promise<any> {
    const params = this.params as z.output<typeof GetDocumentParamsSchema>;

    const response = await client.get({
      index: params.indexName,
      id: params.documentId,
    });

    console.log(`[Elasticsearch] Document retrieved: ${params.documentId}`);

    return {
      id: response._id,
      source: response._source,
      found: response.found,
    };
  }

  private async updateDocument(client: Client): Promise<any> {
    const params = this.params as z.output<typeof UpdateDocumentParamsSchema>;

    const response = await client.update({
      index: params.indexName,
      id: params.documentId,
      body: {
        doc: params.doc,
      },
      refresh: params.refresh,
    });

    console.log(`[Elasticsearch] Document updated: ${params.documentId}`);
    return response;
  }

  private async deleteDocument(client: Client): Promise<any> {
    const params = this.params as z.output<typeof DeleteDocumentParamsSchema>;

    const response = await client.delete({
      index: params.indexName,
      id: params.documentId,
      refresh: params.refresh,
    });

    console.log(`[Elasticsearch] Document deleted: ${params.documentId}`);
    return response;
  }

  private async deleteIndex(client: Client): Promise<any> {
    const params = this.params as z.output<typeof DeleteIndexParamsSchema>;

    const response = await client.indices.delete({
      index: params.indexName,
    });

    console.log(`[Elasticsearch] Index deleted: ${params.indexName}`);
    return response;
  }

  private async indexExists(client: Client): Promise<any> {
    const params = this.params as z.output<typeof IndexExistsParamsSchema>;

    const exists = await client.indices.exists({
      index: params.indexName,
    });

    console.log(`[Elasticsearch] Index ${params.indexName} ${exists ? 'exists' : 'does not exist'}`);

    return {
      indexName: params.indexName,
      exists: Boolean(exists),
    };
  }

  private async aggregate(client: Client): Promise<any> {
    const params = this.params as z.output<typeof AggregateParamsSchema>;

    const body: any = {
      size: params.size,
      aggs: params.aggs,
    };

    if (params.query) {
      body.query = params.query;
    }

    const response = await client.search({
      index: params.indexName,
      body,
    });

    console.log(`[Elasticsearch] Aggregation completed`);

    return {
      took: response.took,
      aggregations: response.aggregations,
    };
  }

  private extractIndexName(): string | undefined {
    const params = this.params as any;
    return params.indexName;
  }
}
