"use strict";
/**
 * VectorDB Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for vector database interactions.
 * All adapters must normalize their data to/from this format.
 *
 * Law of the "Air Gap": This is the ONLY acceptable format for vector database
 * data in the glue layer. Do not pass raw vector DB API responses between services.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.VectorDBExamples = exports.VectorDBError = exports.CollectionCreateResponse = exports.CollectionCreateRequest = exports.VectorDeleteResponse = exports.VectorDeleteRequest = exports.VectorSearchResponse = exports.VectorSearchResult = exports.VectorSearchRequest = exports.VectorUpsertResponse = exports.VectorUpsertRequest = exports.CollectionInfo = exports.VectorData = exports.VectorMetadata = void 0;
exports.transformUpsertResponseToCanonical = transformUpsertResponseToCanonical;
exports.transformCanonicalToUpsertRequest = transformCanonicalToUpsertRequest;
exports.transformSearchResponseToCanonical = transformSearchResponseToCanonical;
exports.transformCanonicalToSearchRequest = transformCanonicalToSearchRequest;
exports.validateVectorUpsertRequest = validateVectorUpsertRequest;
exports.validateVectorSearchRequest = validateVectorSearchRequest;
exports.validateVectorSearchResponse = validateVectorSearchResponse;
exports.validateCollectionInfo = validateCollectionInfo;
exports.isVectorSearchRequest = isVectorSearchRequest;
exports.isVectorUpsertRequest = isVectorUpsertRequest;
exports.isCollectionInfo = isCollectionInfo;
const zod_1 = require("zod");
/**
 * Vector Metadata Schema
 *
 * Defines the structure for metadata associated with vectors.
 */
exports.VectorMetadata = zod_1.z.record(zod_1.z.any())
    .describe("Key-value pairs for vector metadata");
/**
 * Vector Data Schema
 *
 * Represents a single vector with its associated metadata.
 */
exports.VectorData = zod_1.z.object({
    id: zod_1.z.string()
        .min(1, "Vector ID cannot be empty")
        .describe("Unique identifier for the vector"),
    vector: zod_1.z.array(zod_1.z.number())
        .min(1, "Vector cannot be empty")
        .describe("Array of float values representing the embedding"),
    metadata: exports.VectorMetadata.optional()
        .describe("Optional metadata associated with the vector"),
    namespace: zod_1.z.string().optional()
        .describe("Optional namespace for partitioning vectors"),
});
/**
 * Collection Info Schema
 *
 * Represents information about a vector collection.
 */
exports.CollectionInfo = zod_1.z.object({
    name: zod_1.z.string()
        .min(1, "Collection name cannot be empty")
        .describe("Name of the collection"),
    dimension: zod_1.z.number()
        .int("Dimension must be an integer")
        .positive("Dimension must be positive")
        .describe("Dimension of vectors in the collection"),
    count: zod_1.z.number()
        .int("Count must be an integer")
        .min(0, "Count must be non-negative")
        .describe("Number of vectors in the collection"),
    metric: zod_1.z.enum([
        'cosine',
        'euclidean',
        'dotproduct',
        'manhattan',
    ]).optional().describe("Distance metric used for similarity search"),
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Additional collection metadata"),
    created_at: zod_1.z.string().datetime().optional()
        .describe("UTC timestamp when collection was created (ISO-8601)"),
    updated_at: zod_1.z.string().datetime().optional()
        .describe("UTC timestamp when collection was last updated (ISO-8601)"),
});
/**
 * Vector Upsert Request Schema
 *
 * Represents a request to upsert (insert or update) vectors.
 */
exports.VectorUpsertRequest = zod_1.z.object({
    collection_name: zod_1.z.string()
        .min(1, "Collection name cannot be empty")
        .describe("Name of the collection to upsert vectors into"),
    vectors: zod_1.z.array(zod_1.z.object({
        id: zod_1.z.string().min(1, "Vector ID cannot be empty"),
        vector: zod_1.z.array(zod_1.z.number()).min(1, "Vector cannot be empty"),
        metadata: exports.VectorMetadata.optional(),
    }))
        .min(1, "At least one vector must be provided")
        .max(1000, "Cannot upsert more than 1000 vectors at once")
        .describe("Array of vectors to upsert"),
    namespace: zod_1.z.string().optional()
        .describe("Optional namespace for the vectors"),
    timeout_ms: zod_1.z.number()
        .int("Timeout must be an integer")
        .positive("Timeout must be positive")
        .max(60000, "Timeout cannot exceed 60 seconds")
        .describe("Request timeout in milliseconds (MANDATORY)"),
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Optional metadata for observability and tracking"),
});
/**
 * Vector Upsert Response Schema
 *
 * Represents the response after upserting vectors.
 */
exports.VectorUpsertResponse = zod_1.z.object({
    upserted_count: zod_1.z.number()
        .int("Count must be an integer")
        .min(0, "Count must be non-negative")
        .describe("Number of vectors upserted"),
    collection_name: zod_1.z.string().describe("Name of the collection"),
    namespace: zod_1.z.string().optional().describe("Namespace used for the upsert"),
    metadata: zod_1.z.object({
        processing_time_ms: zod_1.z.number().optional()
            .describe("Time taken for upsert in milliseconds"),
        total_vectors: zod_1.z.number().optional()
            .describe("Total number of vectors in collection after upsert"),
    }).optional().describe("Upsert metadata"),
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    timestamp: zod_1.z.string().datetime()
        .describe("UTC timestamp of the response (ISO-8601)"),
});
/**
 * Vector Search Request Schema
 *
 * Represents a request to search for similar vectors.
 */
exports.VectorSearchRequest = zod_1.z.object({
    collection_name: zod_1.z.string()
        .min(1, "Collection name cannot be empty")
        .describe("Name of the collection to search in"),
    query_vector: zod_1.z.array(zod_1.z.number())
        .min(1, "Query vector cannot be empty")
        .describe("Vector to search with"),
    top_k: zod_1.z.number()
        .int("top_k must be an integer")
        .positive("top_k must be positive")
        .max(100, "Cannot retrieve more than 100 results")
        .describe("Number of similar vectors to retrieve"),
    filter: zod_1.z.object({
        key: zod_1.z.string().describe("Metadata key to filter on"),
        value: zod_1.z.any().describe("Value to filter by"),
        operator: zod_1.z.enum(['=', '!=', '>', '<', '>=', '<=', 'in', 'not_in']).optional()
            .describe("Comparison operator (defaults to '=')"),
    }).array().optional()
        .describe("Optional filters to apply to metadata"),
    namespace: zod_1.z.string().optional()
        .describe("Optional namespace to search within"),
    include_metadata: zod_1.z.boolean().optional()
        .describe("Whether to include metadata in results (default: true)"),
    include_vectors: zod_1.z.boolean().optional()
        .describe("Whether to include vector data in results (default: false)"),
    timeout_ms: zod_1.z.number()
        .int("Timeout must be an integer")
        .positive("Timeout must be positive")
        .max(30000, "Timeout cannot exceed 30 seconds")
        .describe("Request timeout in milliseconds (MANDATORY)"),
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Optional metadata for observability and tracking"),
});
/**
 * Vector Search Result Schema
 *
 * Represents a single search result.
 */
exports.VectorSearchResult = zod_1.z.object({
    id: zod_1.z.string().describe("ID of the matching vector"),
    score: zod_1.z.number()
        .float("Score must be a float")
        .describe("Similarity score (higher is more similar)"),
    metadata: exports.VectorMetadata.optional()
        .describe("Metadata associated with the vector"),
    vector: zod_1.z.array(zod_1.z.number()).optional()
        .describe("Vector data (if requested)"),
});
/**
 * Vector Search Response Schema
 *
 * Represents the response from a vector search.
 */
exports.VectorSearchResponse = zod_1.z.object({
    results: zod_1.z.array(exports.VectorSearchResult)
        .describe("Array of search results sorted by similarity"),
    collection_name: zod_1.z.string().describe("Name of the collection searched"),
    namespace: zod_1.z.string().optional().describe("Namespace searched"),
    metadata: zod_1.z.object({
        search_time_ms: zod_1.z.number().optional()
            .describe("Time taken for search in milliseconds"),
        total_vectors: zod_1.z.number().optional()
            .describe("Total vectors in the searched collection"),
    }).optional().describe("Search metadata"),
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    timestamp: zod_1.z.string().datetime()
        .describe("UTC timestamp of the response (ISO-8601)"),
});
/**
 * Vector Delete Request Schema
 *
 * Represents a request to delete vectors.
 */
exports.VectorDeleteRequest = zod_1.z.object({
    collection_name: zod_1.z.string()
        .min(1, "Collection name cannot be empty")
        .describe("Name of the collection to delete vectors from"),
    ids: zod_1.z.array(zod_1.z.string())
        .min(1, "At least one ID must be provided")
        .max(1000, "Cannot delete more than 1000 vectors at once")
        .describe("Array of vector IDs to delete"),
    namespace: zod_1.z.string().optional()
        .describe("Optional namespace to delete from"),
    delete_all: zod_1.z.boolean().optional()
        .describe("If true, delete all vectors (ignores ids parameter)"),
    timeout_ms: zod_1.z.number()
        .int("Timeout must be an integer")
        .positive("Timeout must be positive")
        .max(60000, "Timeout cannot exceed 60 seconds")
        .describe("Request timeout in milliseconds (MANDATORY)"),
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
});
/**
 * Vector Delete Response Schema
 *
 * Represents the response after deleting vectors.
 */
exports.VectorDeleteResponse = zod_1.z.object({
    deleted_count: zod_1.z.number()
        .int("Count must be an integer")
        .min(0, "Count must be non-negative")
        .describe("Number of vectors deleted"),
    collection_name: zod_1.z.string().describe("Name of the collection"),
    namespace: zod_1.z.string().optional().describe("Namespace deleted from"),
    metadata: zod_1.z.object({
        processing_time_ms: zod_1.z.number().optional()
            .describe("Time taken for deletion in milliseconds"),
        remaining_vectors: zod_1.z.number().optional()
            .describe("Number of vectors remaining in collection"),
    }).optional().describe("Deletion metadata"),
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    timestamp: zod_1.z.string().datetime()
        .describe("UTC timestamp of the response (ISO-8601)"),
});
/**
 * Collection Create Request Schema
 *
 * Represents a request to create a new collection.
 */
exports.CollectionCreateRequest = zod_1.z.object({
    name: zod_1.z.string()
        .min(1, "Collection name cannot be empty")
        .max(255, "Collection name cannot exceed 255 characters")
        .describe("Name of the collection to create"),
    dimension: zod_1.z.number()
        .int("Dimension must be an integer")
        .positive("Dimension must be positive")
        .max(10000, "Dimension cannot exceed 10000")
        .describe("Dimension of vectors in the collection"),
    metric: zod_1.z.enum([
        'cosine',
        'euclidean',
        'dotproduct',
        'manhattan',
    ]).optional().describe("Distance metric (default: cosine)"),
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Optional collection metadata"),
    timeout_ms: zod_1.z.number()
        .int("Timeout must be an integer")
        .positive("Timeout must be positive")
        .max(30000, "Timeout cannot exceed 30 seconds")
        .describe("Request timeout in milliseconds (MANDATORY)"),
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
});
/**
 * Collection Create Response Schema
 *
 * Represents the response after creating a collection.
 */
exports.CollectionCreateResponse = zod_1.z.object({
    collection: exports.CollectionInfo.describe("Information about the created collection"),
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    timestamp: zod_1.z.string().datetime()
        .describe("UTC timestamp of the response (ISO-8601)"),
});
/**
 * Error Model
 *
 * Represents errors that can occur during vector database operations.
 */
exports.VectorDBError = zod_1.z.object({
    code: zod_1.z.enum([
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
    message: zod_1.z.string().describe("Human-readable error message"),
    details: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Additional error details"),
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for tracing the error"),
    timestamp: zod_1.z.string().datetime()
        .describe("UTC timestamp when the error occurred (ISO-8601)"),
});
/**
 * Transformation Functions
 */
/**
 * Transform raw vector DB upsert response to canonical format
 */
function transformUpsertResponseToCanonical(rawResponse, collectionName, correlationId) {
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
function transformCanonicalToUpsertRequest(canonicalRequest) {
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
function transformSearchResponseToCanonical(rawResponse, collectionName, correlationId) {
    const timestamp = new Date().toISOString();
    return {
        results: (rawResponse.results || rawResponse.matches || []).map((match) => ({
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
function transformCanonicalToSearchRequest(canonicalRequest) {
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
function validateVectorUpsertRequest(data) {
    const result = exports.VectorUpsertRequest.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
function validateVectorSearchRequest(data) {
    const result = exports.VectorSearchRequest.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
function validateVectorSearchResponse(data) {
    const result = exports.VectorSearchResponse.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
function validateCollectionInfo(data) {
    const result = exports.CollectionInfo.safeParse(data);
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
function isVectorSearchRequest(data) {
    return typeof data === 'object' && data !== null &&
        'collection_name' in data && 'query_vector' in data && 'top_k' in data;
}
function isVectorUpsertRequest(data) {
    return typeof data === 'object' && data !== null &&
        'collection_name' in data && 'vectors' in data;
}
function isCollectionInfo(data) {
    return typeof data === 'object' && data !== null &&
        'name' in data && 'dimension' in data && 'count' in data;
}
/**
 * Example usage and validation examples
 */
exports.VectorDBExamples = {
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
    },
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
    },
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
    },
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
    },
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
    },
    validVectorDBError: {
        code: 'COLLECTION_NOT_FOUND',
        message: "The specified collection does not exist",
        details: {
            collection_name: "nonexistent_collection",
        },
        correlation_id: "550e8400-e29b-41d4-a716-446655440000",
        timestamp: "2025-02-03T12:34:56.789Z",
    },
};
//# sourceMappingURL=vectordb-canonical.js.map