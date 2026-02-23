/**
 * Vector Storage Integration for Evolved Code
 *
 * Following CLAUDE.md Federation Constitution:
 * - Law of the Air Gap: No imports from core-projects
 * - Law of Runtime Truth: Verify vector DB connection before use
 * - Law of Idempotency: Safe to run multiple times
 * - Law of Configuration Explicitness: All config via environment variables
 * - Failure Management: Circuit breaker for transient failures
 *
 * Integrates with Vector DB adapter to store evolved code embeddings
 * for semantic search and similarity matching.
 */

import { v4 as uuidv4 } from 'uuid';
import { z } from 'zod';
import { Logger } from '../../logger';
import { CircuitBreaker } from '../circuit-breaker';
import {
  EvolvedCode,
  Problem,
  SimilarSolution,
  StoreWithEmbeddingRequest,
  SearchSimilarRequest,
  validateEvolvedCode,
  validateProblem,
} from './canonical';

// ============================================================================
// CONFIGURATION
// ============================================================================

export interface VectorStorageConfig {
  // Vector DB adapter URL
  vectordb_adapter_url: string;

  // Collection configuration
  collection_name: string;

  // Embedding configuration
  embedding_dimension: number;
  embedding_model?: string; // Name of embedding model if using external service
  embedding_api_key?: string; // API key for embedding service

  // Timeout and retry configuration
  timeout_ms?: number;
  max_retries?: number;
  circuit_breaker_threshold?: number;
  circuit_breaker_timeout_ms?: number;

  // Logging
  logger?: Logger;
}

const DEFAULT_CONFIG = {
  timeout_ms: 30000,
  max_retries: 3,
  circuit_breaker_threshold: 5,
  circuit_breaker_timeout_ms: 60000,
  embedding_dimension: 1536, // OpenAI text-embedding-ada-002 default
};

// ============================================================================
// EMBEDDING GENERATION
// ============================================================================

/**
 * Generate embedding from text
 * This is a placeholder - actual implementation depends on embedding service
 */
export interface EmbeddingGenerator {
  generateEmbedding(text: string): Promise<number[]>;
}

/**
 * Simple embedding generator using character-based hashing
 * This is a fallback for demonstration - production should use proper embeddings
 */
export class SimpleEmbeddingGenerator implements EmbeddingGenerator {
  private readonly dimension: number;

  constructor(dimension: number = 1536) {
    this.dimension = dimension;
  }

  /**
   * Generate a simple hash-based embedding
   * Note: This is NOT semantically meaningful. Use real embeddings in production.
   */
  async generateEmbedding(text: string): Promise<number[]> {
    const embedding: number[] = [];
    const normalized = text.toLowerCase().replace(/\s+/g, ' ');

    for (let i = 0; i < this.dimension; i++) {
      // Simple hash function
      const charCode = i < normalized.length ? normalized.charCodeAt(i) : 0;
      const hash = (charCode * 31 + i * 17) % 1000;
      embedding.push(hash / 1000); // Normalize to [-1, 1] range approximately
    }

    return embedding;
  }
}

/**
 * OpenAI embedding generator (for production use)
 */
export class OpenAIEmbeddingGenerator implements EmbeddingGenerator {
  private readonly apiKey: string;
  private readonly model: string;
  private readonly logger: Logger;

  constructor(apiKey: string, model: string = 'text-embedding-ada-002', logger?: Logger) {
    this.apiKey = apiKey;
    this.model = model;
    this.logger = logger || new Logger('openai-embedding');
  }

  async generateEmbedding(text: string): Promise<number[]> {
    try {
      const response = await fetch('https://api.openai.com/v1/embeddings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          model: this.model,
          input: text,
        }),
        signal: AbortSignal.timeout(30000), // 30 second timeout
      });

      if (!response.ok) {
        throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();

      if (!data.data || !data.data[0] || !data.data[0].embedding) {
        throw new Error('Invalid response format from OpenAI API');
      }

      return data.data[0].embedding;
    } catch (error) {
      this.logger.error('Failed to generate embedding with OpenAI', error as Error, {
        model: this.model,
        text_length: text.length,
      });
      throw error;
    }
  }
}

// ============================================================================
// VECTOR STORAGE CLIENT
// ============================================================================

/**
 * Vector Storage for Evolved Code
 *
 * Integrates with Vector DB adapter to store and search evolved code
 */
export class VectorStorage {
  private readonly config: Required<Omit<VectorStorageConfig, 'logger' | 'embedding_model' | 'embedding_api_key'>> & {
    logger?: Logger;
    embedding_model?: string;
    embedding_api_key?: string;
  };

  private readonly logger: Logger;
  private readonly circuitBreaker: CircuitBreaker;
  private readonly embeddingGenerator: EmbeddingGenerator;
  private initialized: boolean = false;

  // Vector DB adapter HTTP client (simple implementation)
  private readonly httpClient: {
    post: (path: string, body: any) => Promise<any>;
    get: (path: string) => Promise<any>;
  };

  constructor(config: VectorStorageConfig) {
    this.config = {
      ...DEFAULT_CONFIG,
      ...config,
    };

    this.logger = this.config.logger || new Logger('vector-storage');

    // Initialize circuit breaker
    this.circuitBreaker = new CircuitBreaker({
      threshold: this.config.circuit_breaker_threshold,
      timeout_ms: this.config.circuit_breaker_timeout_ms,
      onStateChange: (oldState, newState) => {
        this.logger.warn('Circuit breaker state changed', {
          correlation_id: 'vector-storage-circuit',
          old_state: oldState,
          new_state: newState,
        });
      },
    });

    // Initialize embedding generator
    if (this.config.embedding_api_key) {
      this.embeddingGenerator = new OpenAIEmbeddingGenerator(
        this.config.embedding_api_key,
        this.config.embedding_model,
        this.logger
      );
    } else {
      this.logger.warn('No embedding API key provided, using simple hash-based embeddings (not semantically meaningful)');
      this.embeddingGenerator = new SimpleEmbeddingGenerator(this.config.embedding_dimension);
    }

    // Simple HTTP client for Vector DB adapter
    this.httpClient = {
      post: async (path: string, body: any) => {
        const url = `${this.config.vectordb_adapter_url}${path}`;
        const response = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(this.config.timeout_ms),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return response.json();
      },
      get: async (path: string) => {
        const url = `${this.config.vectordb_adapter_url}${path}`;
        const response = await fetch(url, {
          signal: AbortSignal.timeout(this.config.timeout_ms),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return response.json();
      },
    };

    this.logger.info('VectorStorage initialized', {
      correlation_id: 'vector-storage-init',
      vectordb_adapter_url: this.config.vectordb_adapter_url,
      collection_name: this.config.collection_name,
      embedding_dimension: this.config.embedding_dimension,
    });
  }

  // ========================================================================
  // INITIALIZATION
  // ========================================================================

  /**
   * Initialize vector storage
   * Following CLAUDE.md: RUNTIME TRUTH - verify before marking as ready
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      this.logger.warn('VectorStorage already initialized', {
        correlation_id: 'vector-storage-init',
      });
      return;
    }

    const correlationId = uuidv4();

    this.logger.info('Initializing VectorStorage', {
      correlation_id: correlationId,
      target_service: 'vectordb-adapter',
    });

    try {
      await this.circuitBreaker.execute(async () => {
        // Check if collection exists
        try {
          await this.httpClient.get(`/collections/${this.config.collection_name}`);
          this.logger.info('Collection exists', {
            correlation_id: correlationId,
            collection: this.config.collection_name,
          });
        } catch (error) {
          // Create collection if it doesn't exist
          this.logger.info('Creating collection', {
            correlation_id: correlationId,
            collection: this.config.collection_name,
          });

          await this.httpClient.post('/collections', {
            name: this.config.collection_name,
            dimension: this.config.embedding_dimension,
            distance_metric: 'cosine',
          });
        }
      });

      this.initialized = true;
      this.logger.info('VectorStorage initialized successfully', {
        correlation_id: correlationId,
      });
    } catch (error) {
      this.logger.error('Failed to initialize VectorStorage', error as Error, {
        correlation_id: correlationId,
      });
      throw new Error(
        `VectorStorage initialization failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  // ========================================================================
  // EMBEDDING GENERATION
  // ========================================================================

  /**
   * Generate embedding for evolved code
   * Combines problem description and code for better semantic representation
   */
  async generateEmbedding(evolvedCode: EvolvedCode): Promise<number[]> {
    const text = this.createEmbeddingText(evolvedCode);
    return await this.embeddingGenerator.generateEmbedding(text);
  }

  /**
   * Generate embedding for problem search
   */
  async generateProblemEmbedding(problem: Problem): Promise<number[]> {
    const text = this.createProblemEmbeddingText(problem);
    return await this.embeddingGenerator.generateEmbedding(text);
  }

  /**
   * Create text representation for embedding
   * Combines problem description, code, and metadata
   */
  private createEmbeddingText(evolvedCode: EvolvedCode): string {
    const parts: string[] = [];

    // Problem description
    parts.push(`Problem: ${evolvedCode.problem.description}`);
    parts.push(`Type: ${evolvedCode.problem.type}`);

    if (evolvedCode.problem.constraints) {
      parts.push(`Constraints: ${JSON.stringify(evolvedCode.problem.constraints)}`);
    }

    // Code content
    parts.push(`Code: ${evolvedCode.code}`);
    parts.push(`Language: ${evolvedCode.language}`);

    // Metrics
    parts.push(`Fitness: ${evolvedCode.metrics.fitness_score}`);
    parts.push(`Iterations: ${evolvedCode.metrics.iterations}`);

    return parts.join('\n\n');
  }

  /**
   * Create text representation for problem embedding
   */
  private createProblemEmbeddingText(problem: Problem): string {
    const parts: string[] = [];

    parts.push(`Problem: ${problem.description}`);
    parts.push(`Type: ${problem.type}`);

    if (problem.constraints) {
      parts.push(`Constraints: ${JSON.stringify(problem.constraints)}`);
    }

    if (problem.input_spec) {
      parts.push(`Input: ${problem.input_spec}`);
    }

    if (problem.output_spec) {
      parts.push(`Output: ${problem.output_spec}`);
    }

    if (problem.tags && problem.tags.length > 0) {
      parts.push(`Tags: ${problem.tags.join(', ')}`);
    }

    return parts.join('\n\n');
  }

  // ========================================================================
  // STORAGE OPERATIONS
  // ========================================================================

  /**
   * Store evolved code with embedding
   * Following CLAUDE.md: Law of Idempotency - safe to run multiple times
   */
  async storeWithEmbedding(
    request: StoreWithEmbeddingRequest,
    correlationId?: string
  ): Promise<void> {
    const cid = correlationId || uuidv4();

    this.logger.info('Storing evolved code with embedding', {
      correlation_id: cid,
      code_id: request.evolved_code.id,
      collection: this.config.collection_name,
    });

    // Validate evolved code
    const validation = validateEvolvedCode(request.evolved_code);
    if (!validation.success) {
      throw new Error(`Invalid evolved code: ${validation.errors.join(', ')}`);
    }

    try {
      await this.circuitBreaker.execute(async () => {
        // Generate embedding if not provided
        const embedding = request.embedding
          || await this.generateEmbedding(request.evolved_code);

        // Prepare vector entry
        const vectorEntry = {
          id: request.evolved_code.id,
          vector: embedding,
          payload: {
            code_id: request.evolved_code.id,
            problem_description: request.evolved_code.problem.description,
            problem_type: request.evolved_code.problem.type,
            language: request.evolved_code.language,
            fitness_score: request.evolved_code.metrics.fitness_score,
            timestamp_utc: request.evolved_code.timestamp_utc,
            tags: request.evolvedCode.tags || [],
          },
        };

        // Store in vector database
        await this.httpClient.post(`/collections/${this.config.collection_name}/upsert`, {
          collection_name: this.config.collection_name,
          entries: [vectorEntry],
        });
      });

      this.logger.info('Evolved code stored successfully', {
        correlation_id: cid,
        code_id: request.evolved_code.id,
      });
    } catch (error) {
      this.logger.error('Failed to store evolved code', error as Error, {
        correlation_id: cid,
        code_id: request.evolved_code.id,
      });
      throw error;
    }
  }

  // ========================================================================
  // SEARCH OPERATIONS
  // ========================================================================

  /**
   * Search for similar problems
   * Returns evolved code that solved similar problems
   */
  async searchSimilar(
    request: SearchSimilarRequest,
    correlationId?: string
  ): Promise<SimilarSolution[]> {
    const cid = correlationId || uuidv4();

    this.logger.info('Searching for similar problems', {
      correlation_id: cid,
      problem_type: request.problem.type,
      max_results: request.max_results,
      similarity_threshold: request.similarity_threshold,
    });

    // Validate problem
    const validation = validateProblem(request.problem);
    if (!validation.success) {
      throw new Error(`Invalid problem: ${validation.errors.join(', ')}`);
    }

    try {
      const results = await this.circuitBreaker.execute(async () => {
        // Generate embedding for problem
        const embedding = await this.generateProblemEmbedding(request.problem);

        // Search vector database
        const searchResults = await this.httpClient.post(
          `/collections/${this.config.collection_name}/search`,
          {
            collection_name: this.config.collection_name,
            query: {
              vector: embedding,
              k: request.max_results,
              score_threshold: request.similarity_threshold,
            },
          }
        );

        // Convert to SimilarSolution format
        // Note: In production, you'd fetch full evolved code from storage
        return searchResults.map((result: any) => ({
          evolved_code: {
            id: result.payload.code_id,
            problem: request.problem, // Placeholder - full problem would be fetched
            language: result.payload.language,
            code: '', // Would be fetched from storage
            metrics: {
              iterations: 0,
              fitness_score: result.payload.fitness_score,
              fitness_improvement: 0,
              duration_ms: 0,
            },
            timestamp_utc: result.payload.timestamp_utc,
            is_valid: true,
          } as EvolvedCode,
          similarity_score: result.score,
          similarity_method: 'semantic' as const,
          distance: result.distance,
        }));
      });

      this.logger.info('Search completed', {
        correlation_id: cid,
        results_count: results.length,
      });

      return results;
    } catch (error) {
      this.logger.error('Failed to search similar problems', error as Error, {
        correlation_id: cid,
      });
      throw error;
    }
  }

  // ========================================================================
  // MAINTENANCE OPERATIONS
  // ========================================================================

  /**
   * Delete stale code older than timestamp
   * Following CLAUDE.md: Law of Idempotency - safe to run multiple times
   */
  async deleteStaleCode(timestamp_utc: string, correlationId?: string): Promise<number> {
    const cid = correlationId || uuidv4();

    this.logger.info('Deleting stale code', {
      correlation_id: cid,
      timestamp_utc,
    });

    try {
      const result = await this.circuitBreaker.execute(async () => {
        // Note: This is a simplified implementation
        // Production would use proper date filtering in the vector DB
        const response = await this.httpClient.post(
          `/collections/${this.config.collection_name}/delete`,
          {
            collection_name: this.config.collection_name,
            filter: {
              timestamp_utc: { lt: timestamp_utc },
            },
          }
        );

        return response.deleted_count || 0;
      });

      this.logger.info('Stale code deleted', {
        correlation_id: cid,
        deleted_count: result,
      });

      return result;
    } catch (error) {
      this.logger.error('Failed to delete stale code', error as Error, {
        correlation_id: cid,
      });
      throw error;
    }
  }

  // ========================================================================
  // HEALTH CHECK
  // ========================================================================

  /**
   * Check vector storage health
   */
  async healthCheck(): Promise<{
    healthy: boolean;
    initialized: boolean;
    circuit_state: string;
    collection_exists: boolean;
  }> {
    const circuitStats = this.circuitBreaker.getStats();

    try {
      if (!this.initialized) {
        return {
          healthy: false,
          initialized: false,
          circuit_state: circuitStats.state,
          collection_exists: false,
        };
      }

      // Quick connectivity check
      await this.httpClient.get(`/collections/${this.config.collection_name}`);

      return {
        healthy: circuitStats.state === 'closed',
        initialized: true,
        circuit_state: circuitStats.state,
        collection_exists: true,
      };
    } catch (error) {
      return {
        healthy: false,
        initialized: true,
        circuit_state: circuitStats.state,
        collection_exists: false,
      };
    }
  }

  // ========================================================================
  // CLEANUP
  // ========================================================================

  /**
   * Close vector storage and cleanup resources
   */
  async close(): Promise<void> {
    this.logger.info('Closing VectorStorage', {
      correlation_id: 'vector-storage-close',
    });

    this.initialized = false;
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export { SimpleEmbeddingGenerator, OpenAIEmbeddingGenerator };
export type { EmbeddingGenerator };
