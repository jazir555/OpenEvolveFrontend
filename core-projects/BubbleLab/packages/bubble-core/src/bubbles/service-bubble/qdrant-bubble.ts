import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
import { QdrantClient } from '@qdrant/js-client-rest';

/**
 * Qdrant Bubble - Vector Database Service Bubble Implementation
 *
 * Full production implementation with 10 operations:
 * 1. createCollection - Create a new vector collection
 * 2. deleteCollection - Delete a collection
 * 3. collectionExists - Check if a collection exists
 * 4. insertPoints - Insert vectors with payloads
 * 5. searchPoints - Search for similar vectors
 * 6. upsertPoints - Insert or update vectors
 * 7. deletePoints - Delete vectors by IDs
 * 8. getPoint - Retrieve a point by ID
 * 9. scrollPoints - Browse points with filtering
 * 10. updatePayload - Update point payloads
 */

// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================

const CreateCollectionParamsSchema = z.object({
  operation: z.literal('createCollection'),
  collectionName: z.string().min(1, 'Collection name is required'),
  vectorSize: z.number().int().positive().describe('Dimension of vectors (e.g., 1536 for OpenAI embeddings)'),
  distance: z.enum(['Cosine', 'Euclid', 'Dot', 'Manhattan']).optional().default('Cosine'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const DeleteCollectionParamsSchema = z.object({
  operation: z.literal('deleteCollection'),
  collectionName: z.string().min(1, 'Collection name is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const CollectionExistsParamsSchema = z.object({
  operation: z.literal('collectionExists'),
  collectionName: z.string().min(1, 'Collection name is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const InsertPointsParamsSchema = z.object({
  operation: z.literal('insertPoints'),
  collectionName: z.string().min(1, 'Collection name is required'),
  points: z.array(
    z.object({
      id: z.union([z.string(), z.number()]),
      vector: z.array(z.number()),
      payload: z.record(z.unknown()).optional(),
    })
  ).min(1, 'At least one point is required'),
  wait: z.boolean().optional().default(true),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const SearchPointsParamsSchema = z.object({
  operation: z.literal('searchPoints'),
  collectionName: z.string().min(1, 'Collection name is required'),
  vector: z.array(z.number()).min(1, 'Vector is required'),
  limit: z.number().int().positive().optional().default(10),
  scoreThreshold: z.number().min(0).max(1).optional(),
  withPayload: z.boolean().or(z.array(z.string())).optional().default(true),
  withVector: z.boolean().optional().default(false),
  filter: z.record(z.unknown()).optional().describe('Filter conditions'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const UpsertPointsParamsSchema = z.object({
  operation: z.literal('upsertPoints'),
  collectionName: z.string().min(1, 'Collection name is required'),
  points: z.array(
    z.object({
      id: z.union([z.string(), z.number()]),
      vector: z.array(z.number()),
      payload: z.record(z.unknown()).optional(),
    })
  ).min(1, 'At least one point is required'),
  wait: z.boolean().optional().default(true),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const DeletePointsParamsSchema = z.object({
  operation: z.literal('deletePoints'),
  collectionName: z.string().min(1, 'Collection name is required'),
  points: z.array(z.union([z.string(), z.number()])).min(1, 'At least one point ID is required'),
  wait: z.boolean().optional().default(true),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetPointParamsSchema = z.object({
  operation: z.literal('getPoint'),
  collectionName: z.string().min(1, 'Collection name is required'),
  pointId: z.union([z.string(), z.number()]),
  withPayload: z.boolean().or(z.array(z.string())).optional().default(true),
  withVector: z.boolean().optional().default(false),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const ScrollPointsParamsSchema = z.object({
  operation: z.literal('scrollPoints'),
  collectionName: z.string().min(1, 'Collection name is required'),
  limit: z.number().int().positive().optional().default(10),
  offset: z.union([z.string(), z.number()]).optional(),
  filter: z.record(z.unknown()).optional().describe('Filter conditions'),
  withPayload: z.boolean().or(z.array(z.string())).optional().default(true),
  withVector: z.boolean().optional().default(false),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const UpdatePayloadParamsSchema = z.object({
  operation: z.literal('updatePayload'),
  collectionName: z.string().min(1, 'Collection name is required'),
  payload: z.record(z.unknown()).describe('Payload data to set'),
  points: z.array(z.union([z.string(), z.number()])).min(1, 'At least one point ID is required'),
  wait: z.boolean().optional().default(true),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

// Union of all parameter schemas
const QdrantBubbleParamsSchema = z.discriminatedUnion('operation', [
  CreateCollectionParamsSchema,
  DeleteCollectionParamsSchema,
  CollectionExistsParamsSchema,
  InsertPointsParamsSchema,
  SearchPointsParamsSchema,
  UpsertPointsParamsSchema,
  DeletePointsParamsSchema,
  GetPointParamsSchema,
  ScrollPointsParamsSchema,
  UpdatePayloadParamsSchema,
]);

type QdrantBubbleParams = z.input<typeof QdrantBubbleParamsSchema>;

// Result schema
const QdrantBubbleResultSchema = z.object({
  success: z.boolean(),
  data: z.unknown().describe('Operation result data'),
  error: z.string(),
  meta: z.object({
    operation: z.string(),
    collectionName: z.string().optional(),
    time: z.number().optional(),
  }),
});

type QdrantBubbleResult = z.output<typeof QdrantBubbleResultSchema>;

// ============================================================================
// MAIN BUBBLE CLASS
// ============================================================================

export class QdrantBubble extends ServiceBubble<QdrantBubbleParams, QdrantBubbleResult> {
  static readonly service = 'qdrant';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'qdrant';
  static readonly type = 'service' as const;
  static readonly schema = QdrantBubbleParamsSchema;
  static readonly resultSchema = QdrantBubbleResultSchema;
  static readonly shortDescription =
    'High-performance vector database for similarity search and AI applications';
  static readonly longDescription = `
    Qdrant Bubble for vector similarity search and management.

    Features:
    - Create and manage vector collections
    - Insert and search high-dimensional vectors
    - Filter searches by payload metadata
    - Real-time updates with upsert operations
    - Scalable architecture with sharding support
    - Multiple distance metrics (Cosine, Euclidean, Dot, Manhattan)

    Use cases:
    - Semantic search with embeddings
    - Recommendation systems
    - Image and document similarity
    - RAG (Retrieval Augmented Generation)
    - Duplicate detection
    - Clustering and classification
  `;
  static readonly alias = 'vector';

  private client: QdrantClient | null = null;

  constructor(
    params: QdrantBubbleParams,
    context?: BubbleContext,
    instanceId?: string
  ) {
    super(params, context, instanceId);
  }

  protected getCredentialType(): CredentialType {
    return CredentialType.QDRANT_CRED;
  }

  protected chooseCredential(): string | undefined {
    const credentials = this.params.credentials;
    if (!credentials || typeof credentials !== 'object') {
      throw new Error('Qdrant credentials are required');
    }
    return credentials[CredentialType.QDRANT_CRED];
  }

  public async testCredential(): Promise<boolean> {
    try {
      const client = this.getClient();
      await client.getCollections();
      return true; // If no error, credential is valid
    } catch (error) {
      console.error('[Qdrant] Credential test failed:', error);
      return false;
    }
  }

  private getClient(): QdrantClient {
    if (!this.client) {
      const credential = this.chooseCredential();
      if (!credential) {
        throw new Error('Qdrant credentials not found');
      }

      // Parse credential (expected format: JSON string with url, apiKey)
      let config: any;
      try {
        config = typeof credential === 'string' ? JSON.parse(credential) : credential;
      } catch {
        throw new Error('Invalid Qdrant credentials format. Expected JSON string.');
      }

      if (!config.url) {
        throw new Error('Qdrant URL is required in credentials');
      }

      const clientConfig: any = {
        url: config.url,
      };

      // Add API key if provided
      if (config.apiKey) {
        clientConfig.apiKey = config.apiKey;
      }

      this.client = new QdrantClient(clientConfig);
      console.log('[Qdrant] Client initialized successfully');
    }

    return this.client;
  }

  protected async performAction(context?: BubbleContext): Promise<QdrantBubbleResult> {
    void context;
    const startTime = Date.now();

    try {
      const client = this.getClient();
      const operation = this.params.operation;
      let result: any;

      console.log(`[Qdrant] Executing operation: ${operation}`);

      switch (operation) {
        case 'createCollection':
          result = await this.createCollection(client);
          break;

        case 'deleteCollection':
          result = await this.deleteCollection(client);
          break;

        case 'collectionExists':
          result = await this.collectionExists(client);
          break;

        case 'insertPoints':
          result = await this.insertPoints(client);
          break;

        case 'searchPoints':
          result = await this.searchPoints(client);
          break;

        case 'upsertPoints':
          result = await this.upsertPoints(client);
          break;

        case 'deletePoints':
          result = await this.deletePoints(client);
          break;

        case 'getPoint':
          result = await this.getPoint(client);
          break;

        case 'scrollPoints':
          result = await this.scrollPoints(client);
          break;

        case 'updatePayload':
          result = await this.updatePayload(client);
          break;

        default:
          throw new Error(`Unknown operation: ${operation}`);
      }

      const time = Date.now() - startTime;

      return {
        success: true,
        data: result,
        error: '', // Empty string for successful operations,
        meta: {
          operation,
          collectionName: this.extractCollectionName(),
          time,
        },
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`[Qdrant] Operation failed:`, errorMessage);

      return {
        success: false,
        data: null,
        error: errorMessage,
        meta: {
          operation: this.params.operation,
          collectionName: this.extractCollectionName(),
        },
      };
    }
  }

  private async createCollection(client: QdrantClient): Promise<any> {
    const params = this.params as z.output<typeof CreateCollectionParamsSchema>;
    console.log(`[Qdrant] Creating collection: ${params.collectionName}`);

    await client.createCollection(params.collectionName, {
      vectors: {
        size: params.vectorSize,
        distance: params.distance,
      },
    });

    console.log(`[Qdrant] Collection created successfully: ${params.collectionName}`);

    return {
      collectionName: params.collectionName,
      vectorSize: params.vectorSize,
      distance: params.distance,
      status: 'created',
    };
  }

  private async deleteCollection(client: QdrantClient): Promise<any> {
    const params = this.params as z.output<typeof DeleteCollectionParamsSchema>;

    await client.deleteCollection(params.collectionName);

    console.log(`[Qdrant] Collection deleted: ${params.collectionName}`);

    return {
      collectionName: params.collectionName,
      status: 'deleted',
    };
  }

  private async collectionExists(client: QdrantClient): Promise<any> {
    const params = this.params as z.output<typeof CollectionExistsParamsSchema>;

    const collection = await client.getCollection(params.collectionName);

    console.log(`[Qdrant] Collection ${params.collectionName} exists`);

    return {
      collectionName: params.collectionName,
      exists: true,
      vectorsCount: (collection as any).vectors_count || 0,
      pointsCount: collection.points_count || 0,
      status: collection.status,
    };
  }

  private async insertPoints(client: QdrantClient): Promise<any> {
    const params = this.params as z.output<typeof InsertPointsParamsSchema>;

    const response = await client.upsert(params.collectionName, {
      points: params.points,
      wait: params.wait,
    });

    console.log(`[Qdrant] Inserted ${params.points.length} points`);

    return {
      operationId: response.operation_id,
      status: 'completed',
    };
  }

  private async searchPoints(client: QdrantClient): Promise<any> {
    const params = this.params as z.output<typeof SearchPointsParamsSchema>;

    const searchResult = await client.search(params.collectionName, {
      vector: params.vector,
      limit: params.limit,
      score_threshold: params.scoreThreshold,
      with_payload: params.withPayload,
      with_vector: params.withVector,
      filter: params.filter as any,
    });

    console.log(`[Qdrant] Found ${searchResult.length} results`);

    return {
      points: searchResult.map((point: any) => ({
        id: point.id,
        score: point.score,
        payload: point.payload,
      })),
      count: searchResult.length,
    };
  }

  private async upsertPoints(client: QdrantClient): Promise<any> {
    const params = this.params as z.output<typeof UpsertPointsParamsSchema>;

    const response = await client.upsert(params.collectionName, {
      points: params.points,
      wait: params.wait,
    });

    console.log(`[Qdrant] Upserted ${params.points.length} points`);

    return {
      operationId: response.operation_id,
      status: 'completed',
    };
  }

  private async deletePoints(client: QdrantClient): Promise<any> {
    const params = this.params as z.output<typeof DeletePointsParamsSchema>;

    const response = await client.delete(params.collectionName, {
      points: params.points,
      wait: params.wait,
    });

    console.log(`[Qdrant] Deleted ${params.points.length} points`);

    return {
      operationId: response.operation_id,
      status: 'completed',
    };
  }

  private async getPoint(client: QdrantClient): Promise<any> {
    const params = this.params as z.output<typeof GetPointParamsSchema>;

    const point = await client.retrieve(params.collectionName, {
      ids: [params.pointId],
      with_payload: params.withPayload,
      with_vector: params.withVector,
    });

    console.log(`[Qdrant] Retrieved point: ${params.pointId}`);

    return {
      id: point[0]?.id,
      payload: point[0]?.payload,
      vector: point[0]?.vector,
    };
  }

  private async scrollPoints(client: QdrantClient): Promise<any> {
    const params = this.params as z.output<typeof ScrollPointsParamsSchema>;

    const response = await client.scroll(params.collectionName, {
      limit: params.limit,
      offset: typeof params.offset === 'string' ? params.offset : undefined,
      filter: params.filter as any,
      with_payload: params.withPayload,
      with_vector: params.withVector,
    });

    console.log(`[Qdrant] Retrieved ${response.points.length} points`);

    return {
      points: response.points.map((point: any) => ({
        id: point.id,
        payload: point.payload,
      })),
      nextPageOffset: response.next_page_offset,
    };
  }

  private async updatePayload(client: QdrantClient): Promise<any> {
    const params = this.params as z.output<typeof UpdatePayloadParamsSchema>;

    const response = await client.setPayload(params.collectionName, {
      payload: params.payload,
      points: params.points,
      wait: params.wait,
    });

    console.log(`[Qdrant] Updated payload for ${params.points.length} points`);

    return {
      operationId: response.operation_id,
      status: 'completed',
    };
  }

  private extractCollectionName(): string | undefined {
    const params = this.params as any;
    return params.collectionName;
  }
}

