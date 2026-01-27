import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
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
// ============================================================================
// MAIN BUBBLE CLASS
// ============================================================================
export class ElasticsearchBubble extends ServiceBubble {
    static service = 'elasticsearch';
    static authType = 'apikey';
    static bubbleName = 'elasticsearch';
    static type = 'service';
    static schema = ElasticsearchBubbleParamsSchema;
    static resultSchema = ElasticsearchBubbleResultSchema;
    static shortDescription = 'Full-text search and analytics engine with distributed architecture';
    static longDescription = `
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
    static alias = 'es';
    client = null;
    constructor(params, context, instanceId) {
        super(params, context, instanceId);
    }
    getCredentialType() {
        return CredentialType.ELASTICSEARCH_CRED;
    }
    chooseCredential() {
        const credentials = this.params.credentials;
        if (!credentials || typeof credentials !== 'object') {
            throw new Error('Elasticsearch credentials are required');
        }
        return credentials[CredentialType.ELASTICSEARCH_CRED];
    }
    async testCredential() {
        try {
            const client = this.getClient();
            await client.ping();
            return true;
        }
        catch (error) {
            console.error('[Elasticsearch] Credential test failed:', error);
            return false;
        }
    }
    getClient() {
        if (!this.client) {
            const credential = this.chooseCredential();
            if (!credential) {
                throw new Error('Elasticsearch credentials not found');
            }
            // Parse credential (expected format: JSON string with url, username, password)
            let config;
            try {
                config = typeof credential === 'string' ? JSON.parse(credential) : credential;
            }
            catch {
                throw new Error('Invalid Elasticsearch credentials format. Expected JSON string.');
            }
            if (!config.url) {
                throw new Error('Elasticsearch URL is required in credentials');
            }
            const clientConfig = {
                node: config.url,
            };
            // Add authentication if provided
            if (config.username && config.password) {
                clientConfig.auth = {
                    username: config.username,
                    password: config.password,
                };
            }
            else if (config.apiKey) {
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
    async performAction(context) {
        void context;
        const startTime = Date.now();
        try {
            const client = this.getClient();
            const operation = this.params.operation;
            let result;
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
        }
        catch (error) {
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
    async createIndex(client) {
        const params = this.params;
        console.log(`[Elasticsearch] Creating index: ${params.indexName}`);
        const indexParams = {
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
    async indexDocument(client) {
        const params = this.params;
        const indexParams = {
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
    async bulkIndex(client) {
        const params = this.params;
        // Build bulk operations array
        const operations = [];
        for (const doc of params.documents) {
            operations.push({ index: { _index: params.indexName } });
            operations.push(doc);
        }
        const response = await client.bulk({
            operations,
            refresh: params.refresh,
        });
        console.log(`[Elasticsearch] Bulk indexed ${response.items.length} documents with ${response.errors ? 'errors' : 'success'}`);
        return response;
    }
    async search(client) {
        const params = this.params;
        const searchParams = {
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
            hits: response.hits.hits.map((hit) => ({
                id: hit._id,
                score: hit._score,
                source: hit._source,
            })),
            aggregations: response.aggregations,
        };
    }
    async getDocument(client) {
        const params = this.params;
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
    async updateDocument(client) {
        const params = this.params;
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
    async deleteDocument(client) {
        const params = this.params;
        const response = await client.delete({
            index: params.indexName,
            id: params.documentId,
            refresh: params.refresh,
        });
        console.log(`[Elasticsearch] Document deleted: ${params.documentId}`);
        return response;
    }
    async deleteIndex(client) {
        const params = this.params;
        const response = await client.indices.delete({
            index: params.indexName,
        });
        console.log(`[Elasticsearch] Index deleted: ${params.indexName}`);
        return response;
    }
    async indexExists(client) {
        const params = this.params;
        const exists = await client.indices.exists({
            index: params.indexName,
        });
        console.log(`[Elasticsearch] Index ${params.indexName} ${exists ? 'exists' : 'does not exist'}`);
        return {
            indexName: params.indexName,
            exists: Boolean(exists),
        };
    }
    async aggregate(client) {
        const params = this.params;
        const body = {
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
    extractIndexName() {
        const params = this.params;
        return params.indexName;
    }
}
//# sourceMappingURL=elasticsearch-bubble.js.map