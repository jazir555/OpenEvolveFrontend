/**
 * Pinecone Client Implementation
 *
 * Pinecone-specific vector database client with circuit breaker and retry logic.
 */

import { Logger } from '../../../lib/logger';
import { CircuitBreaker } from '../../../lib/circuit-breaker';
import { retryWithJitter } from '../../../lib/retry';
import {
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
  VectorDBType,
  transformPineconeToCanonical,
  transformCanonicalToPinecone,
} from '../../../schemas/vectordb-canonical';

export interface PineconeClientConfig {
  apiKey: string;
  environment?: string;
  timeout?: number;
  maxRetries?: number;
}

export class PineconeClient {
  private logger: Logger;
  private circuitBreaker: CircuitBreaker;
  private config: PineconeClientConfig;
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(config: PineconeClientConfig) {
    this.logger = new Logger('vectordb-adapter:pinecone-client');
    this.config = {
      timeout: 5000,
      maxRetries: 3,
      ...config,
    };

    // Pinecone API URL format
    const environment = this.config.environment || 'us-east1-aws';
    this.baseUrl = `https://controller.${environment}.pinecone.io`;

    this.headers = {
      'Content-Type': 'application/json',
      'Api-Key': this.config.apiKey,
    };

    // Circuit breaker configuration
    this.circuitBreaker = new CircuitBreaker({
      threshold: 5,
      timeout: 60000,
      logger: this.logger,
    });
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<HealthCheckResponse> {
    const startTime = Date.now();

    return this.circuitBreaker.execute(async () => {
      try {
        const response = await fetch(`${this.baseUrl}/databases`, {
          method: 'GET',
          headers: this.headers,
          signal: AbortSignal.timeout(this.config.timeout!),
        });

        if (!response.ok) {
          throw new Error(`Pinecone health check failed: ${response.statusText}`);
        }

        const latency = Date.now() - startTime;

        const result: HealthCheckResponse = {
          status: 'healthy',
          backend_type: VectorDBType.PINECONE,
          connected: true,
          latency_ms: latency,
          timestamp: new Date().toISOString(),
        };

        this.logger.info('Pinecone health check successful', {
          latency_ms: latency,
        });

        return result;
      } catch (error) {
        this.logger.error('Pinecone health check failed', error as Error);
        return {
          status: 'unhealthy',
          backend_type: VectorDBType.PINECONE,
          connected: false,
          error: (error as Error).message,
          timestamp: new Date().toISOString(),
        };
      }
    });
  }

  /**
   * Create index (collection in Pinecone)
   */
  async createCollection(config: CollectionConfig): Promise<void> {
    return this.circuitBreaker.execute(async () => {
      const body = {
        name: config.name,
        dimension: config.dimension,
        metric: config.distance_metric === 'cosine' ? 'cosine' :
                 config.distance_metric === 'euclidean' ? 'euclidean' :
                 'dotproduct',
        pods: 1,
        replicas: 1,
        pod_type: 'p1.x1',
      };

      await retryWithJitter(
        async () => {
          const response = await fetch(`${this.baseUrl}/databases`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(body),
            signal: AbortSignal.timeout(this.config.timeout!),
          });

          if (!response.ok && response.status !== 409) {
            throw new Error(`Failed to create index: ${response.statusText}`);
          }
        },
        this.config.maxRetries!,
        this.logger
      );

      this.logger.info('Pinecone index created', {
        index: config.name,
        dimension: config.dimension,
        distance_metric: config.distance_metric,
      });
    });
  }

  /**
   * Get collection info
   */
  async getCollectionInfo(collectionName: string): Promise<CollectionInfo> {
    return this.circuitBreaker.execute(async () => {
      const response = await fetch(
        `${this.baseUrl}/databases/${collectionName}`,
        {
          method: 'GET',
          headers: this.headers,
          signal: AbortSignal.timeout(this.config.timeout!),
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to get index info: ${response.statusText}`);
      }

      const data = await response.json();

      return {
        name: data.database.name,
        dimension: data.database.dimension,
        vector_count: data.database.total_vector_count,
        distance_metric: data.database.metric.toLowerCase(),
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };
    });
  }

  /**
   * Get index URL for vector operations
   */
  private getIndexUrl(collectionName: string): string {
    const environment = this.config.environment || 'us-east1-aws';
    return `https://${collectionName}-${environment}.pinecone.io`;
  }

  /**
   * Upsert vectors
   */
  async upsert(request: UpsertRequest): Promise<UpsertResponse> {
    return this.circuitBreaker.execute(async () => {
      const vectors = request.entries.map(transformCanonicalToPinecone);
      const indexUrl = this.getIndexUrl(request.collection_name);
      const namespace = request.namespace || '';

      await retryWithJitter(
        async () => {
          const response = await fetch(
            `${indexUrl}/vectors/upsert?namespace=${namespace}`,
            {
              method: 'POST',
              headers: this.headers,
              body: JSON.stringify({ vectors }),
              signal: AbortSignal.timeout(this.config.timeout!),
            }
          );

          if (!response.ok) {
            throw new Error(`Upsert failed: ${response.statusText}`);
          }
        },
        this.config.maxRetries!,
        this.logger
      );

      this.logger.info('Pinecone upsert successful', {
        collection: request.collection_name,
        count: request.entries.length,
        namespace,
      });

      return {
        upserted_count: request.entries.length,
        collection_name: request.collection_name,
        timestamp: new Date().toISOString(),
      };
    });
  }

  /**
   * Search vectors
   */
  async search(collectionName: string, query: SearchQuery): Promise<SearchResult[]> {
    return this.circuitBreaker.execute(async () => {
      const indexUrl = this.getIndexUrl(collectionName);
      const namespace = query.filter?.namespace || '';

      const requestBody: any = {
        vector: Array.isArray(query.vector) ? query.vector : null,
        topK: query.k,
        includeMetadata: true,
        includeValues: true,
        namespace,
      };

      if (query.filter) {
        requestBody.filter = query.filter;
      }

      const response = await fetch(`${indexUrl}/query`, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(requestBody),
        signal: AbortSignal.timeout(this.config.timeout!),
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const data = await response.json();
      const results = data.matches.map((match: any) => ({
        entry: transformPineconeToCanonical(match),
        score: match.score,
        distance: 1 - match.score,
      }));

      this.logger.info('Pinecone search successful', {
        collection: collectionName,
        result_count: results.length,
      });

      return results;
    });
  }

  /**
   * Delete vectors
   */
  async delete(request: DeleteRequest): Promise<DeleteResponse> {
    return this.circuitBreaker.execute(async () => {
      const indexUrl = this.getIndexUrl(request.collection_name);
      const namespace = request.namespace || '';

      if (request.delete_all) {
        // Delete all vectors in namespace
        await retryWithJitter(
          async () => {
            const response = await fetch(
              `${indexUrl}/vectors/delete?deleteAll=true&namespace=${namespace}`,
              {
                method: 'POST',
                headers: this.headers,
                signal: AbortSignal.timeout(this.config.timeout!),
              }
            );

            if (!response.ok) {
              throw new Error(`Delete all failed: ${response.statusText}`);
            }
          },
          this.config.maxRetries!,
          this.logger
        );
      } else {
        // Delete specific IDs
        await retryWithJitter(
          async () => {
            const response = await fetch(
              `${indexUrl}/vectors/delete`,
              {
                method: 'POST',
                headers: this.headers,
                body: JSON.stringify({
                  ids: request.ids,
                  namespace,
                }),
                signal: AbortSignal.timeout(this.config.timeout!),
              }
            );

            if (!response.ok) {
              throw new Error(`Delete failed: ${response.statusText}`);
            }
          },
          this.config.maxRetries!,
          this.logger
        );
      }

      this.logger.info('Pinecone delete successful', {
        collection: request.collection_name,
        count: request.delete_all ? 'all' : request.ids.length,
        namespace,
      });

      return {
        deleted_count: request.delete_all ? -1 : request.ids.length,
        collection_name: request.collection_name,
        timestamp: new Date().toISOString(),
      };
    });
  }

  /**
   * List collections (indexes)
   */
  async listCollections(): Promise<string[]> {
    return this.circuitBreaker.execute(async () => {
      const response = await fetch(`${this.baseUrl}/databases`, {
        method: 'GET',
        headers: this.headers,
        signal: AbortSignal.timeout(this.config.timeout!),
      });

      if (!response.ok) {
        throw new Error(`Failed to list indexes: ${response.statusText}`);
      }

      const data = await response.json();
      return data.databases.map((db: any) => db.name);
    });
  }
}
