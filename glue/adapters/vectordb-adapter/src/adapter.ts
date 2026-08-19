/**
 * Vector DB Adapter - Main Entry Point
 *
 * Multi-backend vector database adapter supporting:
 * - Qdrant
 * - Pinecone
 * - Chroma
 * - pgvector
 *
 * Following CLAUDE.md Federation Constitution:
 * - Law of Configuration Explicitness: All config via environment variables
 * - Law of Runtime Truth: Validates backend capabilities at runtime
 * - Law of UTC: All timestamps in UTC ISO-8601
 * - JSON Lines logging with correlation_id
 */

import { Logger } from './lib/logger';
import { validateEnvVars } from './lib/env-validator';
import { CircuitBreaker } from './lib/circuit-breaker';
import {
  VectorDBType,
  VectorEntry,
  CollectionConfig,
  SearchQuery,
  SearchResult,
  UpsertRequest,
  UpsertResponse,
  DeleteRequest,
  DeleteResponse,
  CollectionInfo,
  HealthCheckResponse,
  validateVectorDimension,
  validateVectorEntry,
  validateSearchQuery,
  validateUpsertRequest,
  validateDeleteRequest,
  validateCollectionConfig,
} from './schemas/vectordb-canonical';
import { QdrantClient, QdrantClientConfig } from './clients/qdrant-client';
import { PineconeClient, PineconeClientConfig } from './clients/pinecone-client';
import { ChromaClient, ChromaClientConfig } from './clients/chroma-client';
import { PgvectorClient, PgvectorClientConfig } from './clients/pgvector-client';

export interface VectorDBAdapterConfig {
  backendType: VectorDBType;
  url?: string;
  apiKey?: string;
  connectionString?: string;
  timeout?: number;
  maxRetries?: number;
  environment?: string; // For Pinecone
}

/**
 * Vector DB Adapter Class
 *
 * Main adapter that routes operations to the appropriate backend client.
 */
export class VectorDBAdapter {
  private logger: Logger;
  private config: VectorDBAdapterConfig;
  private client: QdrantClient | PineconeClient | ChromaClient | PgvectorClient | null = null;

  constructor(config?: VectorDBAdapterConfig) {
    this.logger = new Logger('vectordb-adapter');
    this.config = config || this.loadConfigFromEnv();
    this.initializeClient();
  }

  /**
   * Load configuration from environment variables
   * Following Law of Configuration Explicitness
   */
  private loadConfigFromEnv(): VectorDBAdapterConfig {
    // Validate required environment variables
    const validation = validateEnvVars(['VECTORDB_TYPE'], process.env);

    if (!validation.valid) {
      const errorMsg = `Missing required environment variables: ${validation.missing.join(', ')}`;
      this.logger.error(errorMsg);
      throw new Error(errorMsg);
    }

    const backendType = process.env.VECTORDB_TYPE as VectorDBType;

    if (!Object.values(VectorDBType).includes(backendType)) {
      const errorMsg = `Invalid VECTORDB_TYPE: ${backendType}. Must be one of: ${Object.values(VectorDBType).join(', ')}`;
      this.logger.error(errorMsg);
      throw new Error(errorMsg);
    }

    const config: VectorDBAdapterConfig = {
      backendType,
      url: process.env.VECTORDB_URL,
      apiKey: process.env.VECTORDB_API_KEY,
      connectionString: process.env.VECTORDB_CONNECTION_STRING,
      timeout: process.env.TIMEOUT_MS ? parseInt(process.env.TIMEOUT_MS) : 5000,
      maxRetries: process.env.MAX_RETRIES ? parseInt(process.env.MAX_RETRIES) : 3,
      environment: process.env.PINECONE_ENVIRONMENT,
    };

    // Backend-specific validation
    switch (backendType) {
      case VectorDBType.QDRANT:
      case VectorDBType.CHROMA:
        if (!config.url) {
          throw new Error(`VECTORDB_URL is required for ${backendType}`);
        }
        break;

      case VectorDBType.PINECONE:
        if (!config.apiKey) {
          throw new Error('VECTORDB_API_KEY is required for Pinecone');
        }
        break;

      case VectorDBType.PGVECTOR:
        if (!config.connectionString) {
          throw new Error('VECTORDB_CONNECTION_STRING is required for pgvector');
        }
        break;
    }

    this.logger.info('VectorDB adapter configuration loaded from environment', {
      backend_type: backendType,
      timeout: config.timeout,
    });

    return config;
  }

  /**
   * Initialize the appropriate backend client
   */
  private initializeClient(): void {
    try {
      switch (this.config.backendType) {
        case VectorDBType.QDRANT:
          if (!this.config.url) {
            throw new Error('Qdrant requires URL configuration');
          }
          this.client = new QdrantClient({
            url: this.config.url,
            apiKey: this.config.apiKey,
            timeout: this.config.timeout,
            maxRetries: this.config.maxRetries,
          } as QdrantClientConfig);
          break;

        case VectorDBType.PINECONE:
          if (!this.config.apiKey) {
            throw new Error('Pinecone requires API key configuration');
          }
          this.client = new PineconeClient({
            apiKey: this.config.apiKey,
            environment: this.config.environment,
            timeout: this.config.timeout,
            maxRetries: this.config.maxRetries,
          } as PineconeClientConfig);
          break;

        case VectorDBType.CHROMA:
          if (!this.config.url) {
            throw new Error('Chroma requires URL configuration');
          }
          this.client = new ChromaClient({
            url: this.config.url,
            timeout: this.config.timeout,
            maxRetries: this.config.maxRetries,
          } as ChromaClientConfig);
          break;

        case VectorDBType.PGVECTOR:
          if (!this.config.connectionString) {
            throw new Error('pgvector requires connection string configuration');
          }
          this.client = new PgvectorClient({
            connectionString: this.config.connectionString,
            timeout: this.config.timeout,
            maxRetries: this.config.maxRetries,
          } as PgvectorClientConfig);
          break;

        default:
          throw new Error(`Unsupported backend type: ${this.config.backendType}`);
      }

      this.logger.info('VectorDB client initialized', {
        backend: this.config.backendType,
      });
    } catch (error) {
      this.logger.error('Failed to initialize VectorDB client', error as Error);
      throw error;
    }
  }

  /**
   * Health check
   * Verifies the backend is accessible and responsive
   */
  async healthCheck(): Promise<HealthCheckResponse> {
    if (!this.client) {
      throw new Error('VectorDB client not initialized');
    }

    this.logger.info('Performing health check', {
      backend: this.config.backendType,
    });

    return this.client.healthCheck();
  }

  /**
   * Create a collection (table/index) in the vector database
   */
  async createCollection(config: CollectionConfig): Promise<void> {
    // Validate collection config
    const validation = validateCollectionConfig(config);
    if (!validation.success) {
      const error = new Error(`Invalid collection config: ${JSON.stringify(validation.error?.issues)}`);
      this.logger.error('Collection config validation failed', error);
      throw error;
    }

    if (!this.client) {
      throw new Error('VectorDB client not initialized');
    }

    this.logger.info('Creating collection', {
      backend: this.config.backendType,
      collection: config.name,
      dimension: config.dimension,
      distance_metric: config.distance_metric,
    });

    return this.client.createCollection(config);
  }

  /**
   * Get information about a collection
   */
  async getCollectionInfo(collectionName: string): Promise<CollectionInfo> {
    if (!this.client) {
      throw new Error('VectorDB client not initialized');
    }

    this.logger.info('Getting collection info', {
      backend: this.config.backendType,
      collection: collectionName,
    });

    return this.client.getCollectionInfo(collectionName);
  }

  /**
   * List all collections
   */
  async listCollections(): Promise<string[]> {
    if (!this.client) {
      throw new Error('VectorDB client not initialized');
    }

    this.logger.info('Listing collections', {
      backend: this.config.backendType,
    });

    return this.client.listCollections();
  }

  /**
   * Upsert vectors to a collection
   * Following Law of Idempotency: Safe to run multiple times
   */
  async upsert(request: UpsertRequest): Promise<UpsertResponse> {
    // Validate upsert request
    const validation = validateUpsertRequest(request);
    if (!validation.success) {
      const error = new Error(`Invalid upsert request: ${JSON.stringify(validation.error?.issues)}`);
      this.logger.error('Upsert request validation failed', error);
      throw error;
    }

    if (!this.client) {
      throw new Error('VectorDB client not initialized');
    }

    // Validate vector dimensions
    const collectionInfo = await this.getCollectionInfo(request.collection_name);
    for (const entry of request.entries) {
      const dimensionValidation = validateVectorDimension(
        entry.vector,
        collectionInfo.dimension
      );

      if (!dimensionValidation.valid) {
        const error = new Error(dimensionValidation.error!);
        this.logger.error('Vector dimension validation failed', error, {
          entry_id: entry.id,
          expected_dimension: collectionInfo.dimension,
        });
        throw error;
      }
    }

    this.logger.info('Upserting vectors', {
      backend: this.config.backendType,
      collection: request.collection_name,
      count: request.entries.length,
    });

    return this.client.upsert(request);
  }

  /**
   * Search for similar vectors
   */
  async search(collectionName: string, query: SearchQuery): Promise<SearchResult[]> {
    // Validate search query
    const validation = validateSearchQuery(query);
    if (!validation.success) {
      const error = new Error(`Invalid search query: ${JSON.stringify(validation.error?.issues)}`);
      this.logger.error('Search query validation failed', error);
      throw error;
    }

    if (!this.client) {
      throw new Error('VectorDB client not initialized');
    }

    this.logger.info('Searching vectors', {
      backend: this.config.backendType,
      collection: collectionName,
      k: query.k,
      score_threshold: query.score_threshold,
    });

    return this.client.search(collectionName, query);
  }

  /**
   * Delete vectors from a collection
   * Following Law of Idempotency: Safe to run multiple times
   */
  async delete(request: DeleteRequest): Promise<DeleteResponse> {
    // Validate delete request
    const validation = validateDeleteRequest(request);
    if (!validation.success) {
      const error = new Error(`Invalid delete request: ${JSON.stringify(validation.error?.issues)}`);
      this.logger.error('Delete request validation failed', error);
      throw error;
    }

    if (!this.client) {
      throw new Error('VectorDB client not initialized');
    }

    if (request.delete_all) {
      this.logger.warn('Deleting all vectors in collection', {
        backend: this.config.backendType,
        collection: request.collection_name,
      });
    } else {
      this.logger.info('Deleting vectors', {
        backend: this.config.backendType,
        collection: request.collection_name,
        count: request.ids.length,
      });
    }

    return this.client.delete(request);
  }

  /**
   * Get the backend type
   */
  getBackendType(): VectorDBType {
    return this.config.backendType;
  }

  /**
   * Close the adapter and release resources
   */
  async close(): Promise<void> {
    this.logger.info('Closing VectorDB adapter', {
      backend: this.config.backendType,
    });

    if (this.client && 'close' in this.client) {
      await (this.client as PgvectorClient).close();
    }

    this.client = null;
  }
}

/**
 * Create a VectorDB adapter instance from environment variables
 */
export function createVectorDBAdapter(): VectorDBAdapter {
  return new VectorDBAdapter();
}

/**
 * Create a VectorDB adapter instance with explicit configuration
 */
export function createVectorDBAdapterWithConfig(config: VectorDBAdapterConfig): VectorDBAdapter {
  return new VectorDBAdapter(config);
}
