/**
 * pgvector Client Implementation
 *
 * pgvector-specific vector database client with circuit breaker and retry logic.
 * Uses PostgreSQL with the pgvector extension for vector similarity search.
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
  transformPgvectorToCanonical,
  transformCanonicalToPgvector,
  validateVectorDimension,
} from '../../../schemas/vectordb-canonical';

export interface PgvectorClientConfig {
  connectionString: string;
  tableName?: string;
  timeout?: number;
  maxRetries?: number;
}

export class PgvectorClient {
  private logger: Logger;
  private circuitBreaker: CircuitBreaker;
  private config: PgvectorClientConfig;
  private tableName: string;
  private pool: any; // Will be initialized when needed

  constructor(config: PgvectorClientConfig) {
    this.logger = new Logger('vectordb-adapter:pgvector-client');
    this.config = {
      timeout: 5000,
      maxRetries: 3,
      ...config,
    };

    this.tableName = this.config.tableName || 'vectors';

    // Circuit breaker configuration
    this.circuitBreaker = new CircuitBreaker({
      threshold: 5,
      timeout: 60000,
      logger: this.logger,
    });
  }

  /**
   * Initialize connection pool
   */
  private async initializePool() {
    if (this.pool) return this.pool;

    // Dynamic import for pg (PostgreSQL client)
    const { Pool } = await import('pg');

    this.pool = new Pool({
      connectionString: this.config.connectionString,
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: this.config.timeout,
    });

    return this.pool;
  }

  /**
   * Execute SQL query
   */
  private async query(text: string, params: any[] = []) {
    const pool = await this.initializePool();
    const result = await pool.query(text, params);
    return result;
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<HealthCheckResponse> {
    const startTime = Date.now();

    return this.circuitBreaker.execute(async () => {
      try {
        await this.query('SELECT 1');
        const latency = Date.now() - startTime;

        const result: HealthCheckResponse = {
          status: 'healthy',
          backend_type: VectorDBType.PGVECTOR,
          connected: true,
          latency_ms: latency,
          timestamp: new Date().toISOString(),
        };

        this.logger.info('pgvector health check successful', {
          latency_ms: latency,
        });

        return result;
      } catch (error) {
        this.logger.error('pgvector health check failed', error as Error);
        return {
          status: 'unhealthy',
          backend_type: VectorDBType.PGVECTOR,
          connected: false,
          error: (error as Error).message,
          timestamp: new Date().toISOString(),
        };
      }
    });
  }

  /**
   * Create collection (table with vector column)
   */
  async createCollection(config: CollectionConfig): Promise<void> {
    return this.circuitBreaker.execute(async () => {
      await retryWithJitter(
        async () => {
          // Create pgvector extension if not exists
          await this.query('CREATE EXTENSION IF NOT EXISTS vector');

          // Create table
          const createTableSQL = `
            CREATE TABLE IF NOT EXISTS ${config.name} (
              id UUID PRIMARY KEY,
              vector vector(${config.dimension}),
              text TEXT,
              metadata JSONB,
              created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
          `;

          await this.query(createTableSQL);

          // Create index based on distance metric
          let operator = '<=>'; // Default: cosine
          if (config.distance_metric === 'euclidean') {
            operator = '<->';
          } else if (config.distance_metric === 'dot_product') {
            operator = '<#>';
          }

          const createIndexSQL = `
            CREATE INDEX IF NOT EXISTS ${config.name}_vector_idx
            ON ${config.name}
            USING hnsw (vector ${operator}_vector_ops)
            WITH (m = 16, ef_construction = 64);
          `;

          await this.query(createIndexSQL);
        },
        this.config.maxRetries!,
        this.logger
      );

      this.logger.info('pgvector table created', {
        table: config.name,
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
      const result = await this.query(`
        SELECT
          table_name as name,
          (SELECT COUNT(*) FROM ${collectionName}) as vector_count,
          obj_description((SELECT oid FROM pg_class WHERE relname = '${collectionName}'), 'pg_class') as distance_metric
        FROM information_schema.tables
        WHERE table_name = $1
      `, [collectionName]);

      if (result.rows.length === 0) {
        throw new Error(`Collection ${collectionName} not found`);
      }

      const row = result.rows[0];

      return {
        name: row.name,
        dimension: 0, // pgvector doesn't store dimension in metadata
        vector_count: parseInt(row.vector_count),
        distance_metric: row.distance_metric || 'cosine',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };
    });
  }

  /**
   * Upsert vectors
   */
  async upsert(request: UpsertRequest): Promise<UpsertResponse> {
    return this.circuitBreaker.execute(async () => {
      let upsertedCount = 0;

      await retryWithJitter(
        async () => {
          const client = await this.initializePool();

          // Use transaction for batch insert
          await client.query('BEGIN');

          try {
            for (const entry of request.entries) {
              const canonical = transformCanonicalToPgvector(entry);
              const vectorStr = Array.isArray(canonical.vector)
                ? `[${canonical.vector.join(',')}]`
                : JSON.stringify(canonical.vector);

              await this.query(
                `INSERT INTO ${request.collection_name} (id, vector, text, metadata, created_at)
                 VALUES ($1, $2::vector, $3, $4, $5)
                 ON CONFLICT (id) DO UPDATE
                 SET vector = EXCLUDED.vector,
                     text = EXCLUDED.text,
                     metadata = EXCLUDED.metadata`,
                [
                  canonical.id,
                  vectorStr,
                  canonical.text,
                  JSON.stringify(canonical.metadata),
                  canonical.created_at,
                ]
              );

              upsertedCount++;
            }

            await client.query('COMMIT');
          } catch (error) {
            await client.query('ROLLBACK');
            throw error;
          }
        },
        this.config.maxRetries!,
        this.logger
      );

      this.logger.info('pgvector upsert successful', {
        table: request.collection_name,
        count: upsertedCount,
      });

      return {
        upserted_count: upsertedCount,
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
      if (!Array.isArray(query.vector)) {
        throw new Error('pgvector search requires dense vectors');
      }

      let operator = '<=>'; // Default: cosine
      if (query.distance_metric === 'euclidean') {
        operator = '<->';
      } else if (query.distance_metric === 'dot_product') {
        operator = '<#>';
      }

      const vectorStr = `[${query.vector.join(',')}]`;
      let whereClause = '';
      const params: any[] = [vectorStr];
      let paramIndex = 2;

      if (query.filter) {
        const filters: string[] = [];
        for (const [key, value] of Object.entries(query.filter)) {
          filters.push(`metadata->>$${paramIndex} = $${paramIndex + 1}`);
          params.push(key, value);
          paramIndex += 2;
        }
        whereClause = ` AND ${filters.join(' AND ')}`;
      }

      const sql = `
        SELECT
          id,
          vector,
          text,
          metadata,
          created_at,
          vector ${operator} $1::vector as distance,
          1 - (vector ${operator} $1::vector) as score
        FROM ${collectionName}
        WHERE true ${whereClause}
        ORDER BY vector ${operator} $1::vector
        LIMIT $${paramIndex}
      `;

      params.push(query.k);

      const result = await this.query(sql, params);

      const results: SearchResult[] = result.rows.map((row: any) => ({
        entry: transformPgvectorToCanonical({
          id: row.id,
          vector: row.vector,
          text: row.text,
          metadata: row.metadata,
          created_at: row.created_at,
        }),
        score: row.score,
        distance: row.distance,
      }));

      // Apply score threshold client-side
      let filteredResults = results;
      if (query.score_threshold !== undefined) {
        filteredResults = results.filter(r => r.score >= query.score_threshold!);
      }

      this.logger.info('pgvector search successful', {
        table: collectionName,
        result_count: filteredResults.length,
      });

      return filteredResults;
    });
  }

  /**
   * Delete vectors
   */
  async delete(request: DeleteRequest): Promise<DeleteResponse> {
    return this.circuitBreaker.execute(async () => {
      let deletedCount = 0;

      await retryWithJitter(
        async () => {
          if (request.delete_all) {
            // Truncate table (delete all rows)
            const result = await this.query(`TRUNCATE TABLE ${request.collection_name}`);
            deletedCount = -1; // Unknown count
          } else {
            // Delete specific IDs
            const result = await this.query(
              `DELETE FROM ${request.collection_name} WHERE id = ANY($1)`,
              [request.ids]
            );
            deletedCount = result.rowCount || 0;
          }
        },
        this.config.maxRetries!,
        this.logger
      );

      this.logger.info('pgvector delete successful', {
        table: request.collection_name,
        count: deletedCount,
      });

      return {
        deleted_count: deletedCount,
        collection_name: request.collection_name,
        timestamp: new Date().toISOString(),
      };
    });
  }

  /**
   * List collections (tables with vector columns)
   */
  async listCollections(): Promise<string[]> {
    return this.circuitBreaker.execute(async () => {
      const result = await this.query(`
        SELECT table_name
        FROM information_schema.columns
        WHERE data_type = 'user-defined'
          AND udt_name = 'vector'
      `);

      return result.rows.map((row: any) => row.table_name);
    });
  }

  /**
   * Close connection pool
   */
  async close(): Promise<void> {
    if (this.pool) {
      await this.pool.end();
      this.pool = null;
    }
  }
}
