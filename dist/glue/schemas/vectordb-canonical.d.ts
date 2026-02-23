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
export declare const VectorMetadata: z.ZodRecord<z.ZodString, z.ZodAny>;
export type VectorMetadata = z.infer<typeof VectorMetadata>;
/**
 * Vector Data Schema
 *
 * Represents a single vector with its associated metadata.
 */
export declare const VectorData: z.ZodObject<{
    id: z.ZodString;
    vector: z.ZodArray<z.ZodNumber, "many">;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    namespace: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    id: string;
    vector: number[];
    metadata?: Record<string, any> | undefined;
    namespace?: string | undefined;
}, {
    id: string;
    vector: number[];
    metadata?: Record<string, any> | undefined;
    namespace?: string | undefined;
}>;
export type VectorData = z.infer<typeof VectorData>;
/**
 * Collection Info Schema
 *
 * Represents information about a vector collection.
 */
export declare const CollectionInfo: z.ZodObject<{
    name: z.ZodString;
    dimension: z.ZodNumber;
    count: z.ZodNumber;
    metric: z.ZodOptional<z.ZodEnum<["cosine", "euclidean", "dotproduct", "manhattan"]>>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    created_at: z.ZodOptional<z.ZodString>;
    updated_at: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    name: string;
    count: number;
    dimension: number;
    metadata?: Record<string, any> | undefined;
    created_at?: string | undefined;
    updated_at?: string | undefined;
    metric?: "euclidean" | "cosine" | "dotproduct" | "manhattan" | undefined;
}, {
    name: string;
    count: number;
    dimension: number;
    metadata?: Record<string, any> | undefined;
    created_at?: string | undefined;
    updated_at?: string | undefined;
    metric?: "euclidean" | "cosine" | "dotproduct" | "manhattan" | undefined;
}>;
export type CollectionInfo = z.infer<typeof CollectionInfo>;
/**
 * Vector Upsert Request Schema
 *
 * Represents a request to upsert (insert or update) vectors.
 */
export declare const VectorUpsertRequest: z.ZodObject<{
    collection_name: z.ZodString;
    vectors: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        vector: z.ZodArray<z.ZodNumber, "many">;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        vector: number[];
        metadata?: Record<string, any> | undefined;
    }, {
        id: string;
        vector: number[];
        metadata?: Record<string, any> | undefined;
    }>, "many">;
    namespace: z.ZodOptional<z.ZodString>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    vectors: {
        id: string;
        vector: number[];
        metadata?: Record<string, any> | undefined;
    }[];
    collection_name: string;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    namespace?: string | undefined;
}, {
    timeout_ms: number;
    vectors: {
        id: string;
        vector: number[];
        metadata?: Record<string, any> | undefined;
    }[];
    collection_name: string;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    namespace?: string | undefined;
}>;
export type VectorUpsertRequest = z.infer<typeof VectorUpsertRequest>;
/**
 * Vector Upsert Response Schema
 *
 * Represents the response after upserting vectors.
 */
export declare const VectorUpsertResponse: z.ZodObject<{
    upserted_count: z.ZodNumber;
    collection_name: z.ZodString;
    namespace: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodObject<{
        processing_time_ms: z.ZodOptional<z.ZodNumber>;
        total_vectors: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        processing_time_ms?: number | undefined;
        total_vectors?: number | undefined;
    }, {
        processing_time_ms?: number | undefined;
        total_vectors?: number | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    collection_name: string;
    upserted_count: number;
    correlation_id?: string | undefined;
    metadata?: {
        processing_time_ms?: number | undefined;
        total_vectors?: number | undefined;
    } | undefined;
    namespace?: string | undefined;
}, {
    timestamp: string;
    collection_name: string;
    upserted_count: number;
    correlation_id?: string | undefined;
    metadata?: {
        processing_time_ms?: number | undefined;
        total_vectors?: number | undefined;
    } | undefined;
    namespace?: string | undefined;
}>;
export type VectorUpsertResponse = z.infer<typeof VectorUpsertResponse>;
/**
 * Vector Search Request Schema
 *
 * Represents a request to search for similar vectors.
 */
export declare const VectorSearchRequest: z.ZodObject<{
    collection_name: z.ZodString;
    query_vector: z.ZodArray<z.ZodNumber, "many">;
    top_k: z.ZodNumber;
    filter: z.ZodOptional<z.ZodArray<z.ZodObject<{
        key: z.ZodString;
        value: z.ZodAny;
        operator: z.ZodOptional<z.ZodEnum<["=", "!=", ">", "<", ">=", "<=", "in", "not_in"]>>;
    }, "strip", z.ZodTypeAny, {
        key: string;
        value?: any;
        operator?: "<" | ">" | "!=" | "=" | ">=" | "<=" | "in" | "not_in" | undefined;
    }, {
        key: string;
        value?: any;
        operator?: "<" | ">" | "!=" | "=" | ">=" | "<=" | "in" | "not_in" | undefined;
    }>, "many">>;
    namespace: z.ZodOptional<z.ZodString>;
    include_metadata: z.ZodOptional<z.ZodBoolean>;
    include_vectors: z.ZodOptional<z.ZodBoolean>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    top_k: number;
    collection_name: string;
    query_vector: number[];
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    filter?: {
        key: string;
        value?: any;
        operator?: "<" | ">" | "!=" | "=" | ">=" | "<=" | "in" | "not_in" | undefined;
    }[] | undefined;
    namespace?: string | undefined;
    include_metadata?: boolean | undefined;
    include_vectors?: boolean | undefined;
}, {
    timeout_ms: number;
    top_k: number;
    collection_name: string;
    query_vector: number[];
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    filter?: {
        key: string;
        value?: any;
        operator?: "<" | ">" | "!=" | "=" | ">=" | "<=" | "in" | "not_in" | undefined;
    }[] | undefined;
    namespace?: string | undefined;
    include_metadata?: boolean | undefined;
    include_vectors?: boolean | undefined;
}>;
export type VectorSearchRequest = z.infer<typeof VectorSearchRequest>;
/**
 * Vector Search Result Schema
 *
 * Represents a single search result.
 */
export declare const VectorSearchResult: z.ZodObject<{
    id: z.ZodString;
    score: any;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    vector: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
}, "strip", z.ZodTypeAny, {
    [x: string]: any;
    id?: unknown;
    score?: unknown;
    metadata?: unknown;
    vector?: unknown;
}, {
    [x: string]: any;
    id?: unknown;
    score?: unknown;
    metadata?: unknown;
    vector?: unknown;
}>;
export type VectorSearchResult = z.infer<typeof VectorSearchResult>;
/**
 * Vector Search Response Schema
 *
 * Represents the response from a vector search.
 */
export declare const VectorSearchResponse: z.ZodObject<{
    results: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        score: any;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        vector: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
    }, "strip", z.ZodTypeAny, {
        [x: string]: any;
        id?: unknown;
        score?: unknown;
        metadata?: unknown;
        vector?: unknown;
    }, {
        [x: string]: any;
        id?: unknown;
        score?: unknown;
        metadata?: unknown;
        vector?: unknown;
    }>, "many">;
    collection_name: z.ZodString;
    namespace: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodObject<{
        search_time_ms: z.ZodOptional<z.ZodNumber>;
        total_vectors: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        total_vectors?: number | undefined;
        search_time_ms?: number | undefined;
    }, {
        total_vectors?: number | undefined;
        search_time_ms?: number | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    results: {
        [x: string]: any;
        id?: unknown;
        score?: unknown;
        metadata?: unknown;
        vector?: unknown;
    }[];
    collection_name: string;
    correlation_id?: string | undefined;
    metadata?: {
        total_vectors?: number | undefined;
        search_time_ms?: number | undefined;
    } | undefined;
    namespace?: string | undefined;
}, {
    timestamp: string;
    results: {
        [x: string]: any;
        id?: unknown;
        score?: unknown;
        metadata?: unknown;
        vector?: unknown;
    }[];
    collection_name: string;
    correlation_id?: string | undefined;
    metadata?: {
        total_vectors?: number | undefined;
        search_time_ms?: number | undefined;
    } | undefined;
    namespace?: string | undefined;
}>;
export type VectorSearchResponse = z.infer<typeof VectorSearchResponse>;
/**
 * Vector Delete Request Schema
 *
 * Represents a request to delete vectors.
 */
export declare const VectorDeleteRequest: z.ZodObject<{
    collection_name: z.ZodString;
    ids: z.ZodArray<z.ZodString, "many">;
    namespace: z.ZodOptional<z.ZodString>;
    delete_all: z.ZodOptional<z.ZodBoolean>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    ids: string[];
    collection_name: string;
    correlation_id?: string | undefined;
    namespace?: string | undefined;
    delete_all?: boolean | undefined;
}, {
    timeout_ms: number;
    ids: string[];
    collection_name: string;
    correlation_id?: string | undefined;
    namespace?: string | undefined;
    delete_all?: boolean | undefined;
}>;
export type VectorDeleteRequest = z.infer<typeof VectorDeleteRequest>;
/**
 * Vector Delete Response Schema
 *
 * Represents the response after deleting vectors.
 */
export declare const VectorDeleteResponse: z.ZodObject<{
    deleted_count: z.ZodNumber;
    collection_name: z.ZodString;
    namespace: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodObject<{
        processing_time_ms: z.ZodOptional<z.ZodNumber>;
        remaining_vectors: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        processing_time_ms?: number | undefined;
        remaining_vectors?: number | undefined;
    }, {
        processing_time_ms?: number | undefined;
        remaining_vectors?: number | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    collection_name: string;
    deleted_count: number;
    correlation_id?: string | undefined;
    metadata?: {
        processing_time_ms?: number | undefined;
        remaining_vectors?: number | undefined;
    } | undefined;
    namespace?: string | undefined;
}, {
    timestamp: string;
    collection_name: string;
    deleted_count: number;
    correlation_id?: string | undefined;
    metadata?: {
        processing_time_ms?: number | undefined;
        remaining_vectors?: number | undefined;
    } | undefined;
    namespace?: string | undefined;
}>;
export type VectorDeleteResponse = z.infer<typeof VectorDeleteResponse>;
/**
 * Collection Create Request Schema
 *
 * Represents a request to create a new collection.
 */
export declare const CollectionCreateRequest: z.ZodObject<{
    name: z.ZodString;
    dimension: z.ZodNumber;
    metric: z.ZodOptional<z.ZodEnum<["cosine", "euclidean", "dotproduct", "manhattan"]>>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    name: string;
    dimension: number;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    metric?: "euclidean" | "cosine" | "dotproduct" | "manhattan" | undefined;
}, {
    timeout_ms: number;
    name: string;
    dimension: number;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    metric?: "euclidean" | "cosine" | "dotproduct" | "manhattan" | undefined;
}>;
export type CollectionCreateRequest = z.infer<typeof CollectionCreateRequest>;
/**
 * Collection Create Response Schema
 *
 * Represents the response after creating a collection.
 */
export declare const CollectionCreateResponse: z.ZodObject<{
    collection: z.ZodObject<{
        name: z.ZodString;
        dimension: z.ZodNumber;
        count: z.ZodNumber;
        metric: z.ZodOptional<z.ZodEnum<["cosine", "euclidean", "dotproduct", "manhattan"]>>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        created_at: z.ZodOptional<z.ZodString>;
        updated_at: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        count: number;
        dimension: number;
        metadata?: Record<string, any> | undefined;
        created_at?: string | undefined;
        updated_at?: string | undefined;
        metric?: "euclidean" | "cosine" | "dotproduct" | "manhattan" | undefined;
    }, {
        name: string;
        count: number;
        dimension: number;
        metadata?: Record<string, any> | undefined;
        created_at?: string | undefined;
        updated_at?: string | undefined;
        metric?: "euclidean" | "cosine" | "dotproduct" | "manhattan" | undefined;
    }>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    collection: {
        name: string;
        count: number;
        dimension: number;
        metadata?: Record<string, any> | undefined;
        created_at?: string | undefined;
        updated_at?: string | undefined;
        metric?: "euclidean" | "cosine" | "dotproduct" | "manhattan" | undefined;
    };
    correlation_id?: string | undefined;
}, {
    timestamp: string;
    collection: {
        name: string;
        count: number;
        dimension: number;
        metadata?: Record<string, any> | undefined;
        created_at?: string | undefined;
        updated_at?: string | undefined;
        metric?: "euclidean" | "cosine" | "dotproduct" | "manhattan" | undefined;
    };
    correlation_id?: string | undefined;
}>;
export type CollectionCreateResponse = z.infer<typeof CollectionCreateResponse>;
/**
 * Error Model
 *
 * Represents errors that can occur during vector database operations.
 */
export declare const VectorDBError: z.ZodObject<{
    code: z.ZodEnum<["COLLECTION_NOT_FOUND", "COLLECTION_ALREADY_EXISTS", "INVALID_DIMENSION", "INVALID_VECTOR", "VECTOR_NOT_FOUND", "QUERY_FAILED", "TIMEOUT", "INVALID_FILTER", "QUOTA_EXCEEDED", "INVALID_METRIC", "UNKNOWN_ERROR"]>;
    message: z.ZodString;
    details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    message: string;
    code: "TIMEOUT" | "INVALID_FILTER" | "UNKNOWN_ERROR" | "COLLECTION_NOT_FOUND" | "COLLECTION_ALREADY_EXISTS" | "INVALID_DIMENSION" | "INVALID_VECTOR" | "VECTOR_NOT_FOUND" | "QUERY_FAILED" | "QUOTA_EXCEEDED" | "INVALID_METRIC";
    correlation_id?: string | undefined;
    details?: Record<string, any> | undefined;
}, {
    timestamp: string;
    message: string;
    code: "TIMEOUT" | "INVALID_FILTER" | "UNKNOWN_ERROR" | "COLLECTION_NOT_FOUND" | "COLLECTION_ALREADY_EXISTS" | "INVALID_DIMENSION" | "INVALID_VECTOR" | "VECTOR_NOT_FOUND" | "QUERY_FAILED" | "QUOTA_EXCEEDED" | "INVALID_METRIC";
    correlation_id?: string | undefined;
    details?: Record<string, any> | undefined;
}>;
export type VectorDBError = z.infer<typeof VectorDBError>;
/**
 * Transformation Functions
 */
/**
 * Transform raw vector DB upsert response to canonical format
 */
export declare function transformUpsertResponseToCanonical(rawResponse: any, collectionName: string, correlationId?: string): VectorUpsertResponse;
/**
 * Transform canonical upsert request to vector DB API format
 */
export declare function transformCanonicalToUpsertRequest(canonicalRequest: VectorUpsertRequest): any;
/**
 * Transform raw vector DB search response to canonical format
 */
export declare function transformSearchResponseToCanonical(rawResponse: any, collectionName: string, correlationId?: string): VectorSearchResponse;
/**
 * Transform canonical search request to vector DB API format
 */
export declare function transformCanonicalToSearchRequest(canonicalRequest: VectorSearchRequest): any;
/**
 * Validation Functions
 */
export declare function validateVectorUpsertRequest(data: unknown): {
    success: boolean;
    data?: VectorUpsertRequest;
    errors?: string[];
};
export declare function validateVectorSearchRequest(data: unknown): {
    success: boolean;
    data?: VectorSearchRequest;
    errors?: string[];
};
export declare function validateVectorSearchResponse(data: unknown): {
    success: boolean;
    data?: VectorSearchResponse;
    errors?: string[];
};
export declare function validateCollectionInfo(data: unknown): {
    success: boolean;
    data?: CollectionInfo;
    errors?: string[];
};
/**
 * Type Guards
 */
export declare function isVectorSearchRequest(data: unknown): data is VectorSearchRequest;
export declare function isVectorUpsertRequest(data: unknown): data is VectorUpsertRequest;
export declare function isCollectionInfo(data: unknown): data is CollectionInfo;
/**
 * Example usage and validation examples
 */
export declare const VectorDBExamples: {
    validVectorUpsertRequest: VectorUpsertRequest;
    validVectorUpsertResponse: VectorUpsertResponse;
    validVectorSearchRequest: VectorSearchRequest;
    validVectorSearchResponse: VectorSearchResponse;
    validCollectionInfo: CollectionInfo;
    validVectorDBError: VectorDBError;
};
//# sourceMappingURL=vectordb-canonical.d.ts.map