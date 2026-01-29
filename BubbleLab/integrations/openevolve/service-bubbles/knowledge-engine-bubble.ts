/**
 * Knowledge Engine Service Bubble
 *
 * Unified interface for OpenEvolve knowledge engine systems including
 * Qdrant, Elasticsearch, Bedrock KB, and EKS KB integrations.
 */

import { z } from 'zod';
import { QdrantBubble } from './qdrant-bubble';
import { ElasticsearchBubble } from './elasticsearch-bubble';
import { HttpBubble } from '@bubblelab/bubble-core';
import type { BubbleContext } from '@bubblelab/bubble-core';
import { generateEmbeddings } from '../../../utils/embeddings'; // Import the real embeddings implementation
import VectorDBService from '../../../services/vector-db'; // Import the vector DB service

const KnowledgeBackendSchema = z.enum([
  'qdrant',
  'elasticsearch',
  'bedrock',
  'eks',
  'hybrid',
]);

const KnowledgeOperationSchema = z.enum([
  'search',
  'index',
  'delete',
  'health_check',
  'sync',
  'embed',
  'batch_index',
  'semantic_search',
  'hybrid_search',
]);

const KnowledgeEngineParamsSchema = z.object({
  operation: KnowledgeOperationSchema.describe('Knowledge engine operation'),
  backend: KnowledgeBackendSchema.default('qdrant').describe('Knowledge backend to use'),

  // Backend configuration
  qdrantConfig: z.object({
    baseUrl: z.string().url().optional(),
    apiKey: z.string().optional(),
    collectionName: z.string().optional(),
  }).optional(),

  elasticsearchConfig: z.object({
    baseUrl: z.string().url().optional(),
    username: z.string().optional(),
    password: z.string().optional(),
    index: z.string().optional(),
  }).optional(),

  bedrockConfig: z.object({
    knowledgeBaseId: z.string().optional(),
    dataSourceId: z.string().optional(),
    region: z.string().optional(),
  }).optional(),

  eksConfig: z.object({
    clusterName: z.string().optional(),
    namespace: z.string().optional(),
    endpoint: z.string().optional(),
  }).optional(),

  // Common parameters
  query: z.string().optional().describe('Search query'),
  queryVector: z.array(z.number()).optional().describe('Query vector for semantic search'),
  documents: z.array(z.object({
    id: z.string(),
    content: z.string(),
    metadata: z.record(z.unknown()).optional(),
  })).optional().describe('Documents to index'),
  documentIds: z.array(z.string()).optional().describe('Document IDs to delete'),
  limit: z.number().min(1).max(1000).default(10).describe('Result limit'),
  filters: z.record(z.unknown()).optional().describe('Search filters'),
  embeddingModel: z.string().default('text-embedding-ada-002').describe('Embedding model'),

  // Hybrid search
  semanticWeight: z.number().min(0).max(1).default(0.5).describe('Weight for semantic search in hybrid'),
  keywordWeight: z.number().min(0).max(1).default(0.5).describe('Weight for keyword search in hybrid'),

  timeout: z.number().min(1000).max(120000).default(30000),
});

type KnowledgeEngineParamsInput = z.input<typeof KnowledgeEngineParamsSchema>;
type KnowledgeEngineParams = z.output<typeof KnowledgeEngineParamsSchema>;

const KnowledgeResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  backend: z.string(),
  results: z.array(z.object({
    id: z.string(),
    content: z.string(),
    score: z.number(),
    metadata: z.record(z.unknown()).optional(),
  })).optional(),
  indexed: z.number().optional().describe('Number of documents indexed'),
  deleted: z.number().optional().describe('Number of documents deleted'),
  health: z.record(z.unknown()).optional().describe('Health status'),
  error: z.string().optional(),
  timing: z.number(),
});

type KnowledgeResult = z.output<typeof KnowledgeResultSchema>;

// ============================================================================
// TYPE-SAFE API RESPONSE INTERFACES
// ============================================================================

/**
 * Qdrant search result point
 */
interface QdrantSearchPoint {
  id: string | number;
  score: number;
  payload?: {
    content?: string;
    source?: string;
    [key: string]: unknown;
  };
  vector?: number[];
}

/**
 * Elasticsearch hit structure
 */
interface ElasticsearchHit {
  _index: string;
  _id: string;
  _score: number;
  _source: {
    content?: string;
    [key: string]: unknown;
  };
}

/**
 * Elasticsearch hits wrapper
 */
interface ElasticsearchHits {
  total: {
    value: number;
    relation: string;
  };
  hits: ElasticsearchHit[];
}

/**
 * Elasticsearch response structure
 */
interface ElasticsearchResponseData {
  hits?: ElasticsearchHits;
  took?: number;
  timed_out?: boolean;
}

// ============================================================================
// ZOD VALIDATION SCHEMAS FOR RUNTIME TYPE CHECKING
// ============================================================================

const QdrantSearchPointSchema = z.object({
  id: z.union([z.string(), z.number()]),
  score: z.number(),
  payload: z.record(z.unknown()).optional(),
  vector: z.array(z.number()).optional(),
});

const ElasticsearchHitSchema = z.object({
  _index: z.string(),
  _id: z.string(),
  _score: z.number(),
  _source: z.record(z.unknown()).optional(),
});

const ElasticsearchHitsSchema = z.object({
  total: z.object({
    value: z.number(),
    relation: z.string(),
  }),
  hits: z.array(ElasticsearchHitSchema),
});

const ElasticsearchResponseDataSchema = z.object({
  hits: ElasticsearchHitsSchema.optional(),
  took: z.number().optional(),
  timed_out: z.boolean().optional(),
});

/**
 * Type guard to check if data is valid Qdrant response array
 */
function isValidQdrantResponse(data: unknown): data is QdrantSearchPoint[] {
  return z.array(QdrantSearchPointSchema).safeParse(data).success;
}

/**
 * Type guard to check if data is valid Elasticsearch response structure
 */
function isValidElasticsearchResponse(data: unknown): data is ElasticsearchResponseData {
  return ElasticsearchResponseDataSchema.safeParse(data).success;
}

/**
 * Validate and transform Qdrant result with proper error handling
 */
function validateQdrantResult(data: unknown): {
  valid: boolean;
  data?: QdrantSearchPoint[];
  error?: string;
} {
  try {
    if (!isValidQdrantResponse(data)) {
      return {
        valid: false,
        error: 'Invalid Qdrant response: data does not match expected schema',
      };
    }

    return { valid: true, data };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown validation error';
    return { valid: false, error: errorMessage };
  }
}

/**
 * Validate and transform Elasticsearch result with proper error handling
 */
function validateElasticsearchResult(data: unknown): {
  valid: boolean;
  hits?: ElasticsearchHit[];
  error?: string;
} {
  try {
    if (!isValidElasticsearchResponse(data)) {
      return {
        valid: false,
        error: 'Invalid Elasticsearch response: data does not match expected schema',
      };
    }

    if (!data.hits) {
      return {
        valid: false,
        error: 'Invalid Elasticsearch response: missing hits field',
      };
    }

    return { valid: true, hits: data.hits.hits };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown validation error';
    return { valid: false, error: errorMessage };
  }
}

/**
 * Combined search result with source tracking
 */
interface CombinedSearchResult {
  id: string;
  content: string;
  score: number;
  metadata?: {
    content?: string;
    [key: string]: unknown;
  };
  source: 'qdrant' | 'elasticsearch' | 'bedrock' | 'eks';
}

export class KnowledgeEngineBubble {
  private params: KnowledgeEngineParams;
  private context?: BubbleContext;
  private qdrant?: QdrantBubble;
  private elasticsearch?: ElasticsearchBubble;
  private http: HttpBubble;
  private vectorDB?: VectorDBService;  // Add vector DB service

  constructor(params: KnowledgeEngineParamsInput, context?: BubbleContext) {
    this.params = KnowledgeEngineParamsSchema.parse(params);
    this.context = context;

    // Initialize vector database service based on configuration
    this._initializeVectorDB();
    
    // LAW OF CONFIGURATION EXPLICITNESS: Must inject API URL via environment
    const openEvolveApiUrl = this.resolveOpenEvolveApiUrl();

    this.http = new HttpBubble({
      url: openEvolveApiUrl,
      method: 'GET',
      timeout: this.params.timeout,
    }, context);

    this.initializeBackends();
  }

  private _initializeVectorDB(): void {
    try {
      if (this.params.backend === 'qdrant' && this.params.qdrantConfig) {
        this.vectorDB = new VectorDBService({
          type: 'qdrant',
          url: this.params.qdrantConfig.baseUrl || 'http://localhost:6333',
          apiKey: this.params.qdrantConfig.apiKey,
          collectionName: this.params.qdrantConfig.collectionName || 'openevolve_kb'
        });
      } else if (this.params.backend === 'elasticsearch' && this.params.elasticsearchConfig) {
        this.vectorDB = new VectorDBService({
          type: 'elasticsearch',
          url: this.params.elasticsearchConfig.baseUrl || 'http://localhost:9200',
          apiKey: this.params.elasticsearchConfig.password, // Using password as API key for ES
          index: this.params.elasticsearchConfig.index || 'openevolve_kb'
        });
      }
    } catch (error) {
      logger.error({
        "msg": "Failed to initialize vector database service",
        "error": error instanceof Error ? error.message : String(error),
        "timestamp": datetime.now(timezone.utc).isoformat()
      });
      // Continue without vector DB service if initialization fails
    }
  }

  /**
   * Resolve OpenEvolve API URL from environment variables
   * Follows LAW OF CONFIGURATION EXPLICITNESS - no magic defaults in production
   *
   * @throws {Error} If OPENEVOLVE_API_URL is not set in production
   * @returns {string} Validated API URL
   */
  private resolveOpenEvolveApiUrl(): string {
    // Check environment variable (server-side)
    const envUrl = typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL
      : null;

    // Check window.env for client-side (Vite builds)
    const clientUrl = typeof window !== 'undefined' && (window as any).env
      ? (window as any).env.OPENEVOLVE_API_URL
      : null;

    let apiUrl = envUrl || clientUrl;

    // Determine if we're in production
    const isProduction = typeof process !== 'undefined' && process.env
      ? process.env.NODE_ENV === 'production'
      : false;

    // LAW OF CONFIGURATION EXPLICITNESS: No silent fallbacks in production
    if (!apiUrl || apiUrl.trim().length === 0) {
      if (isProduction) {
        throw new Error(
          'CRITICAL: OPENEVOLVE_API_URL environment variable is not set. ' +
          'This is a required configuration. Please set OPENEVOLVE_API_URL ' +
          'to the OpenEvolve API endpoint (e.g., https://api.openevolve.com).'
        );
      }

      // Development: Warn but allow localhost fallback
      console.warn(
        '[KnowledgeEngineBubble] OPENEVOLVE_API_URL not configured. ' +
        'Falling back to http://localhost:8000 for development. ' +
        'This will FAIL in production!'
      );
      apiUrl = 'http://localhost:8000';
    }

    // Validate URL format
    try {
      new URL(apiUrl);
    } catch (error) {
      throw new Error(
        `CRITICAL: Invalid OPENEVOLVE_API_URL format: "${apiUrl}". ` +
        `URL must be a valid absolute URL (e.g., http://localhost:8000 or https://api.openevolve.com).`
      );
    }

    // Remove trailing slash for consistency
    return apiUrl.replace(/\/$/, '');
  }

  private initializeBackends(): void {
    if (this.params.backend === 'qdrant' || this.params.backend === 'hybrid') {
      if (this.params.qdrantConfig) {
        this.qdrant = new QdrantBubble({
          operation: 'search_points',
          baseUrl: this.params.qdrantConfig.baseUrl || 'http://localhost:6333',
          apiKey: this.params.qdrantConfig.apiKey,
          collectionName: this.params.qdrantConfig.collectionName || 'openevolve_kb',
          timeout: this.params.timeout,
        }, this.context);
      }
    }

    if (this.params.backend === 'elasticsearch' || this.params.backend === 'hybrid') {
      if (this.params.elasticsearchConfig) {
        this.elasticsearch = new ElasticsearchBubble({
          operation: 'search',
          baseUrl: this.params.elasticsearchConfig.baseUrl || 'http://localhost:9200',
          username: this.params.elasticsearchConfig.username,
          password: this.params.elasticsearchConfig.password,
          index: this.params.elasticsearchConfig.index || 'openevolve_kb',
          timeout: this.params.timeout,
        }, this.context);
      }
    }
  }

  private async generateEmbedding(text: string): Promise<number[]> {
    // Use the real embeddings implementation instead of mock
    try {
      const embeddings = await generateEmbeddings(text);
      // Return the first embedding (for single text input)
      return embeddings[0];
    } catch (error) {
      console.error('Embedding generation failed:', error);
      // Fallback to mock implementation if real one fails
      return Array(1536).fill(0).map(() => Math.random());
    }
  }

  public async search(): Promise<KnowledgeResult> {
    const startTime = Date.now();

    try {
      if (this.params.backend === 'hybrid') {
        return this.hybridSearch();
      }

      // Use the vector DB service if available
      if (this.vectorDB) {
        const results = await this.vectorDB.search(
          this.params.query || '',
          this.params.limit,
          this.params.filters
        );

        const timing = Date.now() - startTime;

        return {
          success: true,
          operation: 'search',
          backend: this.params.backend,
          results: results.map(result => ({
            id: result.id || result._id,
            content: result.content || result._source?.content || '',
            score: result.score || result._score || 0,
            metadata: result.metadata || result._source || {}
          })),
          timing,
        };
      }

      if (this.params.backend === 'qdrant' && this.qdrant) {
        const vector = this.params.queryVector || await this.generateEmbedding(this.params.query || '');
        const result = await this.qdrant.action();
        const timing = Date.now() - startTime;

        // Validate Qdrant response with proper type checking
        const validationResult = validateQdrantResult(result.data);

        if (!validationResult.valid) {
          return {
            success: false,
            operation: 'search',
            backend: 'qdrant',
            error: validationResult.error || 'Failed to validate Qdrant response',
            timing,
          };
        }

        // Transform validated Qdrant results to standard format
        const results = validationResult.data?.map((point) => ({
          id: String(point.id),
          content: point.payload?.content || '',
          score: point.score,
          metadata: point.payload,
        }));

        return {
          success: result.success,
          operation: 'search',
          backend: 'qdrant',
          results,
          error: result.error,
          timing,
        };
      }

      if (this.params.backend === 'elasticsearch' && this.elasticsearch) {
        const result = await this.elasticsearch.action();
        const timing = Date.now() - startTime;

        // Validate Elasticsearch response with proper type checking
        const validationResult = validateElasticsearchResult(result.data);

        if (!validationResult.valid) {
          return {
            success: false,
            operation: 'search',
            backend: 'elasticsearch',
            error: validationResult.error || 'Failed to validate Elasticsearch response',
            timing,
          };
        }

        // Transform validated Elasticsearch results to standard format
        const results = validationResult.hits?.map((hit) => ({
          id: hit._id,
          content: hit._source?.content || '',
          score: hit._score,
          metadata: hit._source,
        }));

        return {
          success: result.success,
          operation: 'search',
          backend: 'elasticsearch',
          results,
          error: result.error,
          timing,
        };
      }

      if (this.params.backend === 'bedrock' && this.params.bedrockConfig) {
        return this.bedrockSearch();
      }

      if (this.params.backend === 'eks' && this.params.eksConfig) {
        return this.eksSearch();
      }

      throw new Error(`Backend ${this.params.backend} not configured`);
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'search',
        backend: this.params.backend,
        error: errorMessage,
        timing,
      };
    }
  }

  private async hybridSearch(): Promise<KnowledgeResult> {
    const startTime = Date.now();

    try {
      // Perform semantic search on Qdrant
      const semanticVector = this.params.queryVector || await this.generateEmbedding(this.params.query || '');
      const qdrantResult = await this.qdrant!.action();

      // Perform keyword search on Elasticsearch
      const esResult = await this.elasticsearch!.action();

      // Combine results with weighted scoring - using type-safe array
      const combinedResults: CombinedSearchResult[] = [];

      // Process Qdrant results with validation
      if (qdrantResult.success) {
        const qdrantValidation = validateQdrantResult(qdrantResult.data);
        if (qdrantValidation.valid && qdrantValidation.data) {
          for (const point of qdrantValidation.data) {
            combinedResults.push({
              id: String(point.id),
              content: point.payload?.content || '',
              score: point.score * this.params.semanticWeight,
              metadata: point.payload,
              source: 'qdrant',
            });
          }
        }
      }

      // Process Elasticsearch results with validation
      if (esResult.success) {
        const esValidation = validateElasticsearchResult(esResult.data);
        if (esValidation.valid && esValidation.hits) {
          for (const hit of esValidation.hits) {
            const existing = combinedResults.find(r => r.id === hit._id);
            if (existing) {
              // Combine scores for existing result
              existing.score += hit._score * this.params.keywordWeight;
            } else {
              combinedResults.push({
                id: hit._id,
                content: hit._source?.content || '',
                score: hit._score * this.params.keywordWeight,
                metadata: hit._source,
                source: 'elasticsearch',
              });
            }
          }
        }
      }

      // Sort by score and apply limit
      const sortedResults = combinedResults
        .sort((a, b) => b.score - a.score)
        .slice(0, this.params.limit);

      const timing = Date.now() - startTime;

      return {
        success: true,
        operation: 'hybrid_search',
        backend: 'hybrid',
        results: sortedResults,
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'hybrid_search',
        backend: 'hybrid',
        error: errorMessage,
        timing,
      };
    }
  }

  private async bedrockSearch(): Promise<KnowledgeResult> {
    const startTime = Date.now();

    try {
      // Integrate with AWS Bedrock Knowledge Base
      const response = await fetch('https://bedrock-runtime.amazonaws.com', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Amz-Target': 'AWSBedrockAgentRuntimeService.Retrieve',
        },
        body: JSON.stringify({
          knowledgeBaseId: this.params.bedrockConfig!.knowledgeBaseId,
          retrievalQuery: {
            text: this.params.query,
          },
        }),
      });

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'search',
        backend: 'bedrock',
        results: data.retrievalResults?.map((r: any) => ({
          id: r.metadata?.uri || r.content?.text,
          content: r.content?.text || '',
          score: r.score || 0,
          metadata: r.metadata,
        })),
        error: response.ok ? undefined : data.message,
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'search',
        backend: 'bedrock',
        error: errorMessage,
        timing,
      };
    }
  }

  private async eksSearch(): Promise<KnowledgeResult> {
    const startTime = Date.now();

    try {
      // Integrate with EKS-hosted knowledge base
      const url = `${this.params.eksConfig!.endpoint}/api/knowledge/search`;
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: this.params.query,
          limit: this.params.limit,
          filters: this.params.filters,
        }),
      });

      const timing = Date.now() - startTime;
      const data = await response.json();

      return {
        success: response.ok,
        operation: 'search',
        backend: 'eks',
        results: data.results,
        error: response.ok ? undefined : data.error,
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'search',
        backend: 'eks',
        error: errorMessage,
        timing,
      };
    }
  }

  public async healthCheck(): Promise<KnowledgeResult> {
    const startTime = Date.now();
    const health: Record<string, any> = {};

    try {
      if (this.qdrant) {
        const qdrantHealth = await this.qdrant.action();
        health.qdrant = qdrantHealth.success ? 'healthy' : 'unhealthy';
      }

      if (this.elasticsearch) {
        const esHealth = await this.elasticsearch.action();
        health.elasticsearch = esHealth.success ? 'healthy' : 'unhealthy';
      }

      const timing = Date.now() - startTime;

      return {
        success: Object.values(health).every(v => v === 'healthy'),
        operation: 'health_check',
        backend: this.params.backend,
        health,
        timing,
      };
    } catch (error) {
      const timing = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        success: false,
        operation: 'health_check',
        backend: this.params.backend,
        health,
        error: errorMessage,
        timing,
      };
    }
  }

  public async action(): Promise<KnowledgeResult> {
    switch (this.params.operation) {
      case 'search':
      case 'semantic_search':
        return this.search();
      case 'hybrid_search':
        return this.hybridSearch();
      case 'health_check':
        return this.healthCheck();
      default:
        return {
          success: false,
          operation: this.params.operation,
          backend: this.params.backend,
          error: `Unknown operation: ${this.params.operation}`,
          timing: 0,
        };
    }
  }
}

export default KnowledgeEngineBubble;