/**
 * VectorDB Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for vector database interactions.
 * All adapters must normalize their data to/from this format.
 *
 * Law of the "Air Gap": This is the ONLY acceptable format for vector database
 * data in the glue layer. Do not pass raw vector DB API responses between services.
 */

import { z } from 'zod';

/**
 * Vector Metadata Schema
 *
 * Defines the structure for metadata associated with vectors.
 */
export const VectorMetadata = z.record(z.any())
  .describe("Key-value pairs for vector metadata");

export type VectorMetadata = z.infer<typeof VectorMetadata>;

/**
 * Vector Data Schema
 *
 * Represents a single vector with its associated metadata.
 */
export const VectorData = z.object({
  id: z.string()
    .min(1, "Vector ID cannot be empty")
    .describe("Unique identifier for the vector"),

  vector: z.array(z.number())
    .min(1, "Vector cannot be empty")
    .describe("Array of float values representing the embedding"),

  metadata: VectorMetadata.optional()
    .describe("Optional metadata associated with the vector"),

  namespace: z.string().optional()
    .describe("Optional namespace for partitioning vectors"),
});

export type VectorData = z.infer<typeof VectorData>;

/**
 * Collection Info Schema
 *
 * Represents information about a vector collection.
 */
export const CollectionInfo = z.object({
  name: z.string()
    .min(1, "Collection name cannot be empty")
    .describe("Name of the collection"),

  dimension: z.number()
    .int("Dimension must be an integer")
    .positive("Dimension must be positive")
    .describe("Dimension of vectors in the collection"),

  count: z.number()
    .int("Count must be an integer")
    .min(0, "Count must be non-negative")
    .describe("Number of vectors in the collection"),

  metric: z.enum([
    'cosine',
    'euclidean',
    'dotproduct',
    'manhattan',
  ]).optional().describe("Distance metric used for similarity search"),

  metadata: z.record(z.any()).optional()
    .describe("Additional collection metadata"),

  created_at: z.string().datetime().optional()
    .describe("UTC timestamp when collection was created (ISO-8601)"),

  updated_at: z.string().datetime().optional()
    .describe("UTC timestamp when collection was last updated (ISO-8601)"),
});

export type CollectionInfo = z.infer<typeof CollectionInfo>;

/**
 * Vector Upsert Request Schema
 *
 * Represents a request to upsert (insert or update) vectors.
 */
export const VectorUpsertRequest = z.object({
  collection_name: z.string()
    .min(1, "Collection name cannot be empty")
    .describe("Name of the collection to upsert vectors into"),

  vectors: z.array(z.object({
    id: z.string().min(1, "Vector ID cannot be empty"),
    vector: z.array(z.number()).min(1, "Vector cannot be empty"),
    metadata: VectorMetadata.optional(),
  }))
    .min(1, "At least one vector must be provided")
    .max(1000, "Cannot upsert more than 1000 vectors at once")
    .describe("Array of vectors to upsert"),

  namespace: z.string().optional()
    .describe("Optional namespace for the vectors"),

  timeout_ms: z.number()
    .int("Timeout must be an integer")
    .positive("Timeout must be positive")
    .max(60000, "Timeout cannot exceed 60 seconds")
    .describe("Request timeout in milliseconds (MANDATORY)"),

  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),

  metadata: z.record(z.any()).optional()
    .describe("Optional metadata for observability and tracking"),
});

export type VectorUpsertRequest = z.infer<typeof VectorUpsertRequest>;

/**
 * Vector Upsert Response Schema
 *
 * Represents the response after upserting vectors.
 */
export const VectorUpsertResponse = z.object({
  upserted_count: z.number()
    .int("Count must be an integer")
    .min(0, "Count must be non-negative")
    .describe("Number of vectors upserted"),

  collection_name: z.string().describe("Name of the collection"),

  namespace: z.string().optional().describe("Namespace used for the upsert"),

  metadata: z.object({
    processing_time_ms: z.number().optional()
      .describe("Time taken for upsert in milliseconds"),
    total_vectors: z.number().optional()
      .describe("Total number of vectors in collection after upsert"),
  }).optional().describe("Upsert metadata"),

  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),

  timestamp: z.string().datetime()
    .describe("UTC timestamp of the response (ISO-8601)"),
});

export type VectorUpsertResponse = z.infer<typeof VectorUpsertResponse>;

/**
 * Vector Search Request Schema
 *
 * Represents a request to search for similar vectors.
 */
export const VectorSearchRequest = z.object({
  collection_name: z.string()
    .min(1, "Collection name cannot be empty")
    .describe("Name of the collection to search in"),

  query_vector: z.array(z.number())
    .min(1, "Query vector cannot be empty")
    .describe("Vector to search with"),

  top_k: z.number()
    .int("top_k must be an integer")
    .positive("top_k must be positive")
    .max(100, "Cannot retrieve more than 100 results")
    .describe("Number of similar vectors to retrieve"),

  filter: z.object({
    key: z.string().describe("Metadata key to filter on"),
    value: z.any().describe("Value to filter by"),
    operator: z.enum(['=', '!=', '>', '<', '>=', '<=', 'in', 'not_in']).optional()
      .describe("Comparison operator (defaults to '=')"),
  }).array().optional()
    .describe("Optional filters to apply to metadata"),

  namespace: z.string().optional()
    .describe("Optional namespace to search within"),

  include_metadata: z.boolean().optional()
    .describe("Whether to include metadata in results (default: true)"),

  include_vectors: z.boolean().optional()
    .describe("Whether to include vector data in results (default: false)"),

  timeout_ms: z.number()
    .int("Timeout must be an integer")
    .positive("Timeout must be positive")
    .max(30000, "Timeout cannot exceed 30 seconds")
    .describe("Request timeout in milliseconds (MANDATORY)"),

  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),

  metadata: z.record(z.any()).optional()
    .describe("Optional metadata for observability and tracking"),
});

export type VectorSearchRequest = z.infer<typeof VectorSearchRequest>;

/**
 * Vector Search Result Schema
 *
 * Represents a single search result.
 */
export const VectorSearchResult = z.object({
  id: z.string().describe("ID of the matching vector"),

  score: z.number()
    .float("Score must be a float")
    .describe("Similarity score (higher is more similar)"),

  metadata: VectorMetadata.optional()
    .describe("Metadata associated with the vector"),

  vector: z.array(z.number()).optional()
    .describe("Vector data (if requested)"),
});

export type VectorSearchResult = z.infer<typeof VectorSearchResult>;

/**
 * Vector Search Response Schema
 *
 * Represents the response from a vector search.
 */
export const VectorSearchResponse = z.object({
  results: z.array(VectorSearchResult)
    .describe("Array of search results sorted by similarity"),

  collection_name: z.string().describe("Name of the collection searched"),

  namespace: z.string().optional().describe("Namespace searched"),

  metadata: z.object({
    search_time_ms: z.number().optional()
      .describe("Time taken for search in milliseconds"),
    total_vectors: z.number().optional()
      .describe("Total vectors in the searched collection"),
  }).optional().describe("Search metadata"),

  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),

  timestamp: z.string().datetime()
    .describe("UTC timestamp of the response (ISO-8601)"),
});

export type VectorSearchResponse = z.infer<typeof VectorSearchResponse>;

/**
 * Vector Delete Request Schema
 *
 * Represents a request to delete vectors.
 */
export const VectorDeleteRequest = z.object({
  collection_name: z.string()
    .min(1, "Collection name cannot be empty")
    .describe("Name of the collection to delete vectors from"),

  ids: z.array(z.string())
    .min(1, "At least one ID must be provided")
    .max(1000, "Cannot delete more than 1000 vectors at once")
    .describe("Array of vector IDs to delete"),

  namespace: z.string().optional()
    .describe("Optional namespace to delete from"),

  delete_all: z.boolean().optional()
    .describe("If true, delete all vectors (ignores ids parameter)"),

  timeout_ms: z.number()
    .int("Timeout must be an integer")
    .positive("Timeout must be positive")
    .max(60000, "Timeout cannot exceed 60 seconds")
    .describe("Request timeout in milliseconds (MANDATORY)"),

  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),
});

export type VectorDeleteRequest = z.infer<typeof VectorDeleteRequest>;

/**
 * Vector Delete Response Schema
 *
 * Represents the response after deleting vectors.
 */
export const VectorDeleteResponse = z.object({
  deleted_count: z.number()
    .int("Count must be an integer")
    .min(0, "Count must be non-negative")
    .describe("Number of vectors deleted"),

  collection_name: z.string().describe("Name of the collection"),

  namespace: z.string().optional().describe("Namespace deleted from"),

  metadata: z.object({
    processing_time_ms: z.number().optional()
      .describe("Time taken for deletion in milliseconds"),
    remaining_vectors: z.number().optional()
      .describe("Number of vectors remaining in collection"),
  }).optional().describe("Deletion metadata"),

  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),

  timestamp: z.string().datetime()
    .describe("UTC timestamp of the response (ISO-8601)"),
});

export type VectorDeleteResponse = z.infer<typeof VectorDeleteResponse>;

/**
 * Collection Create Request Schema
 *
 * Represents a request to create a new collection.
 */
export const CollectionCreateRequest = z.object({
  name: z.string()
    .min(1, "Collection name cannot be empty")
    .max(255, "Collection name cannot exceed 255 characters")
    .describe("Name of the collection to create"),

  dimension: z.number()
    .int("Dimension must be an integer")
    .positive("Dimension must be positive")
    .max(10000, "Dimension cannot exceed 10000")
    .describe("Dimension of vectors in the collection"),

  metric: z.enum([
    'cosine',
    'euclidean',
    'dotproduct',
    'manhattan',
  ]).optional().describe("Distance metric (default: cosine)"),

  metadata: z.record(z.any()).optional()
    .describe("Optional collection metadata"),

  timeout_ms: z.number()
    .int("Timeout must be an integer")
    .positive("Timeout must be positive")
    .max(30000, "Timeout cannot exceed 30 seconds")
    .describe("Request timeout in milliseconds (MANDATORY)"),

  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),
});

export type CollectionCreateRequest = z.infer<typeof CollectionCreateRequest>;

/**
 * Collection Create Response Schema
 *
 * Represents the response after creating a collection.
 */
export const CollectionCreateResponse = z.object({
  collection: CollectionInfo.describe("Information about the created collection"),

  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),

  timestamp: z.string().datetime()
    .describe("UTC timestamp of the response (ISO-8601)"),
});

export type CollectionCreateResponse = z.infer<typeof CollectionCreateResponse>;

/**
 * Error Model
 *
 * Represents errors that can occur during vector database operations.
 */
export const VectorDBError = z.object({
  code: z.enum([
    'COLLECTION_NOT_FOUND',
    'COLLECTION_ALREADY_EXISTS',
    'INVALID_DIMENSION',
    'INVALID_VECTOR',
    'VECTOR_NOT_FOUND',
    'QUERY_FAILED',
    'TIMEOUT',
    'INVALID_FILTER',
    'QUOTA_EXCEEDED',
    'INVALID_METRIC',
    'UNKNOWN_ERROR',
  ]).describe("Error code for categorization"),

  message: z.string().describe("Human-readable error message"),

  details: z.record(z.any()).optional()
    .describe("Additional error details"),

  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for tracing the error"),

  timestamp: z.string().datetime()
    .describe("UTC timestamp when the error occurred (ISO-8601)"),
});

export type VectorDBError = z.infer<typeof VectorDBError>;

/**
 * Transformation Functions
 */

/**
 * Transform raw vector DB upsert response to canonical format
 */
export function transformUpsertResponseToCanonical(
  rawResponse: any,
  collectionName: string,
  correlationId?: string
): VectorUpsertResponse {
  const timestamp = new Date().toISOString();

  return {
    upserted_count: rawResponse.upserted_count || rawResponse.count || 0,
    collection_name: collectionName,
    namespace: rawResponse.namespace,
    metadata: {
      processing_time_ms: rawResponse.processing_time,
      total_vectors: rawResponse.total_vectors,
    },
    correlation_id: correlationId,
    timestamp,
  };
}

/**
 * Transform canonical upsert request to vector DB API format
 */
export function transformCanonicalToUpsertRequest(
  canonicalRequest: VectorUpsertRequest
): any {
  return {
    collection: canonicalRequest.collection_name,
    vectors: canonicalRequest.vectors,
    namespace: canonicalRequest.namespace,
    metadata: canonicalRequest.metadata,
  };
}

/**
 * Transform raw vector DB search response to canonical format
 */
export function transformSearchResponseToCanonical(
  rawResponse: any,
  collectionName: string,
  correlationId?: string
): VectorSearchResponse {
  const timestamp = new Date().toISOString();

  return {
    results: (rawResponse.results || rawResponse.matches || []).map((match: any) => ({
      id: match.id,
      score: match.score || match.similarity || 0,
      metadata: match.metadata,
      vector: match.vector,
    })),
    collection_name: collectionName,
    namespace: rawResponse.namespace,
    metadata: {
      search_time_ms: rawResponse.search_time,
      total_vectors: rawResponse.total_vectors,
    },
    correlation_id: correlationId,
    timestamp,
  };
}

/**
 * Transform canonical search request to vector DB API format
 */
export function transformCanonicalToSearchRequest(
  canonicalRequest: VectorSearchRequest
): any {
  return {
    collection: canonicalRequest.collection_name,
    vector: canonicalRequest.query_vector,
    top_k: canonicalRequest.top_k,
    filter: canonicalRequest.filter,
    namespace: canonicalRequest.namespace,
    include_metadata: canonicalRequest.include_metadata,
    include_vectors: canonicalRequest.include_vectors,
  };
}

/**
 * Validation Functions
 */

export function validateVectorUpsertRequest(data: unknown): {
  success: boolean;
  data?: VectorUpsertRequest;
  errors?: string[];
} {
  const result = VectorUpsertRequest.safeParse(data);

  if (result.success) {
    return { success: true, data: result.data };
  }

  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

export function validateVectorSearchRequest(data: unknown): {
  success: boolean;
  data?: VectorSearchRequest;
  errors?: string[];
} {
  const result = VectorSearchRequest.safeParse(data);

  if (result.success) {
    return { success: true, data: result.data };
  }

  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

export function validateVectorSearchResponse(data: unknown): {
  success: boolean;
  data?: VectorSearchResponse;
  errors?: string[];
} {
  const result = VectorSearchResponse.safeParse(data);

  if (result.success) {
    return { success: true, data: result.data };
  }

  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

export function validateCollectionInfo(data: unknown): {
  success: boolean;
  data?: CollectionInfo;
  errors?: string[];
} {
  const result = CollectionInfo.safeParse(data);

  if (result.success) {
    return { success: true, data: result.data };
  }

  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

/**
 * Type Guards
 */

export function isVectorSearchRequest(data: unknown): data is VectorSearchRequest {
  return typeof data === 'object' && data !== null &&
    'collection_name' in data && 'query_vector' in data && 'top_k' in data;
}

export function isVectorUpsertRequest(data: unknown): data is VectorUpsertRequest {
  return typeof data === 'object' && data !== null &&
    'collection_name' in data && 'vectors' in data;
}

export function isCollectionInfo(data: unknown): data is CollectionInfo {
  return typeof data === 'object' && data !== null &&
    'name' in data && 'dimension' in data && 'count' in data;
}

/**
 * Example usage and validation examples
 */
export const VectorDBExamples = {
  validVectorUpsertRequest: {
    collection_name: "documents",
    vectors: [
      {
        id: "doc1",
        vector: [0.1, 0.2, 0.3, 0.4, 0.5],
        metadata: {
          title: "Document 1",
          category: "tech",
        },
      },
      {
        id: "doc2",
        vector: [0.6, 0.7, 0.8, 0.9, 1.0],
        metadata: {
          title: "Document 2",
          category: "science",
        },
      },
    ],
    namespace: "production",
    timeout_ms: 5000,
    correlation_id: "550e8400-e29b-41d4-a716-446655440000",
  } as VectorUpsertRequest,

  validVectorUpsertResponse: {
    upserted_count: 2,
    collection_name: "documents",
    namespace: "production",
    metadata: {
      processing_time_ms: 150,
      total_vectors: 1002,
    },
    correlation_id: "550e8400-e29b-41d4-a716-446655440000",
    timestamp: "2025-02-03T12:34:56.789Z",
  } as VectorUpsertResponse,

  validVectorSearchRequest: {
    collection_name: "documents",
    query_vector: [0.15, 0.25, 0.35, 0.45, 0.55],
    top_k: 5,
    filter: [
      {
        key: "category",
        value: "tech",
        operator: "=",
      },
    ],
    namespace: "production",
    include_metadata: true,
    include_vectors: false,
    timeout_ms: 3000,
    correlation_id: "550e8400-e29b-41d4-a716-446655440000",
  } as VectorSearchRequest,

  validVectorSearchResponse: {
    results: [
      {
        id: "doc1",
        score: 0.98,
        metadata: {
          title: "Document 1",
          category: "tech",
        },
      },
      {
        id: "doc3",
        score: 0.87,
        metadata: {
          title: "Document 3",
          category: "tech",
        },
      },
    ],
    collection_name: "documents",
    namespace: "production",
    metadata: {
      search_time_ms: 45,
      total_vectors: 1002,
    },
    correlation_id: "550e8400-e29b-41d4-a716-446655440000",
    timestamp: "2025-02-03T12:34:56.789Z",
  } as VectorSearchResponse,

  validCollectionInfo: {
    name: "documents",
    dimension: 1536,
    count: 1002,
    metric: "cosine",
    metadata: {
      description: "Document embeddings",
      model: "text-embedding-ada-002",
    },
    created_at: "2025-01-15T10:00:00.000Z",
    updated_at: "2025-02-03T12:34:56.000Z",
  } as CollectionInfo,

  validVectorDBError: {
    code: 'COLLECTION_NOT_FOUND',
    message: "The specified collection does not exist",
    details: {
      collection_name: "nonexistent_collection",
    },
    correlation_id: "550e8400-e29b-41d4-a716-446655440000",
    timestamp: "2025-02-03T12:34:56.789Z",
  } as VectorDBError,
};
