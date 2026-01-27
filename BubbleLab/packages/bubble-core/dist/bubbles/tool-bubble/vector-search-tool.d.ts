/**
 * VECTOR SEARCH TOOL
 *
 * A tool bubble for performing vector similarity searches using vector databases.
 * Supports Qdrant and other vector database backends for semantic search,
 * recommendation systems, and similarity matching.
 *
 * Features:
 * - Vector similarity search with configurable top-K
 * - Multiple distance metrics (cosine, euclidean, dot product)
 * - Filtering support for metadata-based queries
 * - Batch search operations
 * - Integration with Qdrant vector database
 */
import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
/**
 * Distance metrics for vector similarity
 */
export declare enum VectorDistanceMetric {
    COSINE = "cosine",
    EUCLIDEAN = "euclidean",
    DOT_PRODUCT = "dot"
}
/**
 * Vector search parameters schema
 */
declare const VectorSearchToolParamsSchema: z.ZodObject<{
    vector: z.ZodArray<z.ZodNumber, "many">;
    topK: z.ZodDefault<z.ZodNumber>;
    scoreThreshold: z.ZodDefault<z.ZodNumber>;
    distanceMetric: z.ZodDefault<z.ZodNativeEnum<typeof VectorDistanceMetric>>;
    collectionName: z.ZodString;
    vectorDimension: z.ZodNumber;
    filter: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    collectionName: string;
    vector: number[];
    scoreThreshold: number;
    topK: number;
    distanceMetric: VectorDistanceMetric;
    vectorDimension: number;
    filter?: Record<string, unknown> | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    collectionName: string;
    vector: number[];
    vectorDimension: number;
    filter?: Record<string, unknown> | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    scoreThreshold?: number | undefined;
    topK?: number | undefined;
    distanceMetric?: VectorDistanceMetric | undefined;
}>;
/**
 * Vector search result schema
 */
declare const VectorSearchToolResultSchema: z.ZodObject<{
    results: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        score: z.ZodNumber;
        payload: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        vector: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        score: number;
        payload?: Record<string, unknown> | undefined;
        vector?: number[] | undefined;
    }, {
        id: string;
        score: number;
        payload?: Record<string, unknown> | undefined;
        vector?: number[] | undefined;
    }>, "many">;
    totalResults: z.ZodNumber;
    queryMetadata: z.ZodObject<{
        collectionName: z.ZodString;
        vectorDimension: z.ZodNumber;
        distanceMetric: z.ZodString;
        searchTime: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        collectionName: string;
        distanceMetric: string;
        vectorDimension: number;
        searchTime: number;
    }, {
        collectionName: string;
        distanceMetric: string;
        vectorDimension: number;
        searchTime: number;
    }>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    results: {
        id: string;
        score: number;
        payload?: Record<string, unknown> | undefined;
        vector?: number[] | undefined;
    }[];
    totalResults: number;
    queryMetadata: {
        collectionName: string;
        distanceMetric: string;
        vectorDimension: number;
        searchTime: number;
    };
}, {
    error: string;
    success: boolean;
    results: {
        id: string;
        score: number;
        payload?: Record<string, unknown> | undefined;
        vector?: number[] | undefined;
    }[];
    totalResults: number;
    queryMetadata: {
        collectionName: string;
        distanceMetric: string;
        vectorDimension: number;
        searchTime: number;
    };
}>;
type VectorSearchToolParams = z.output<typeof VectorSearchToolParamsSchema>;
type VectorSearchToolResult = z.output<typeof VectorSearchToolResultSchema>;
type VectorSearchToolParamsInput = z.input<typeof VectorSearchToolParamsSchema>;
/**
 * Vector Search Tool
 * Performs similarity search in vector databases
 */
export declare class VectorSearchTool extends ToolBubble<VectorSearchToolParams, VectorSearchToolResult> {
    /**
     * REQUIRED STATIC METADATA
     */
    static readonly type: "tool";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        vector: z.ZodArray<z.ZodNumber, "many">;
        topK: z.ZodDefault<z.ZodNumber>;
        scoreThreshold: z.ZodDefault<z.ZodNumber>;
        distanceMetric: z.ZodDefault<z.ZodNativeEnum<typeof VectorDistanceMetric>>;
        collectionName: z.ZodString;
        vectorDimension: z.ZodNumber;
        filter: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        collectionName: string;
        vector: number[];
        scoreThreshold: number;
        topK: number;
        distanceMetric: VectorDistanceMetric;
        vectorDimension: number;
        filter?: Record<string, unknown> | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        collectionName: string;
        vector: number[];
        vectorDimension: number;
        filter?: Record<string, unknown> | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        scoreThreshold?: number | undefined;
        topK?: number | undefined;
        distanceMetric?: VectorDistanceMetric | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        results: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            score: z.ZodNumber;
            payload: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            vector: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
        }, "strip", z.ZodTypeAny, {
            id: string;
            score: number;
            payload?: Record<string, unknown> | undefined;
            vector?: number[] | undefined;
        }, {
            id: string;
            score: number;
            payload?: Record<string, unknown> | undefined;
            vector?: number[] | undefined;
        }>, "many">;
        totalResults: z.ZodNumber;
        queryMetadata: z.ZodObject<{
            collectionName: z.ZodString;
            vectorDimension: z.ZodNumber;
            distanceMetric: z.ZodString;
            searchTime: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            collectionName: string;
            distanceMetric: string;
            vectorDimension: number;
            searchTime: number;
        }, {
            collectionName: string;
            distanceMetric: string;
            vectorDimension: number;
            searchTime: number;
        }>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        results: {
            id: string;
            score: number;
            payload?: Record<string, unknown> | undefined;
            vector?: number[] | undefined;
        }[];
        totalResults: number;
        queryMetadata: {
            collectionName: string;
            distanceMetric: string;
            vectorDimension: number;
            searchTime: number;
        };
    }, {
        error: string;
        success: boolean;
        results: {
            id: string;
            score: number;
            payload?: Record<string, unknown> | undefined;
            vector?: number[] | undefined;
        }[];
        totalResults: number;
        queryMetadata: {
            collectionName: string;
            distanceMetric: string;
            vectorDimension: number;
            searchTime: number;
        };
    }>;
    static readonly shortDescription = "Perform vector similarity search using Qdrant or other vector databases";
    static readonly longDescription = "\n    A powerful tool for performing vector similarity searches in vector databases.\n\n    Features:\n    - High-performance vector similarity search\n    - Support for multiple distance metrics (cosine, euclidean, dot product)\n    - Configurable top-K result retrieval\n    - Score-based filtering for relevant results\n    - Metadata filtering capabilities\n    - Integration with Qdrant vector database\n\n    Use cases:\n    - Semantic search in document repositories\n    - Recommendation systems (content-based filtering)\n    - Image similarity search\n    - Duplicate detection\n    - Natural language query matching\n    - Clustering and nearest neighbor queries\n\n    Distance Metrics:\n    - COSINE: Measures cosine similarity (best for normalized vectors)\n    - EUCLIDEAN: Measures Euclidean distance (L2 norm)\n    - DOT_PRODUCT: Measures dot product similarity\n\n    Requirements:\n    - QDRANT_CRED credential for Qdrant access\n    - Collection must exist in the vector database\n    - Query vector dimension must match collection dimension\n  ";
    static readonly alias = "vector-search";
    constructor(params: VectorSearchToolParamsInput, context?: BubbleContext);
    /**
     * Main action method - performs vector similarity search
     */
    performAction(context?: BubbleContext): Promise<VectorSearchToolResult>;
    /**
     * Validate query vector
     */
    private validateVector;
    /**
     * Perform vector search using Qdrant HTTP API or in-memory computation
     */
    private performQdrantSearch;
    /**
     * Perform in-memory vector similarity search
     * Useful for testing or when no vector database is available
     */
    private performInMemorySearch;
    /**
     * Calculate similarity between two vectors using the specified metric
     */
    private calculateSimilarity;
    /**
     * Calculate cosine similarity between two vectors
     * Returns value in range [-1, 1], where 1 is identical
     */
    private cosineSimilarity;
    /**
     * Calculate Euclidean distance (L2 norm) between two vectors
     * Returns non-negative value, where 0 is identical
     */
    private euclideanDistance;
    /**
     * Calculate dot product of two vectors
     * Returns unnormalized similarity score
     */
    private dotProduct;
    /**
     * Batch compute similarity for multiple query vectors
     * Optimized for performance when searching with many vectors
     */
    private batchSimilarity;
}
export {};
//# sourceMappingURL=vector-search-tool.d.ts.map