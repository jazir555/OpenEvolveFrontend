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
import { CredentialType } from '@bubblelab/shared-schemas';
/**
 * Distance metrics for vector similarity
 */
export var VectorDistanceMetric;
(function (VectorDistanceMetric) {
    VectorDistanceMetric["COSINE"] = "cosine";
    VectorDistanceMetric["EUCLIDEAN"] = "euclidean";
    VectorDistanceMetric["DOT_PRODUCT"] = "dot";
})(VectorDistanceMetric || (VectorDistanceMetric = {}));
/**
 * Vector search parameters schema
 */
const VectorSearchToolParamsSchema = z.object({
    // Vector data
    vector: z
        .array(z.number())
        .min(1, 'Vector cannot be empty')
        .describe('Query vector for similarity search'),
    // Search configuration
    topK: z
        .number()
        .int()
        .min(1, 'topK must be at least 1')
        .max(1000, 'topK cannot exceed 1000')
        .default(10)
        .describe('Number of similar vectors to return'),
    scoreThreshold: z
        .number()
        .min(0, 'Score threshold must be between 0 and 1')
        .max(1, 'Score threshold must be between 0 and 1')
        .default(0.0)
        .describe('Minimum similarity score threshold (0-1)'),
    distanceMetric: z
        .nativeEnum(VectorDistanceMetric)
        .default(VectorDistanceMetric.COSINE)
        .describe('Distance metric for similarity calculation'),
    // Collection configuration
    collectionName: z
        .string()
        .min(1, 'Collection name is required')
        .describe('Name of the vector collection to search'),
    vectorDimension: z
        .number()
        .int()
        .positive()
        .describe('Dimension of the vectors'),
    // Optional filtering
    filter: z
        .record(z.unknown())
        .optional()
        .describe('Optional metadata filter for search results'),
    // Credentials
    credentials: z
        .record(z.nativeEnum(CredentialType), z.string())
        .optional()
        .describe('Vector database credentials'),
});
/**
 * Vector search result item
 */
const VectorSearchResultItemSchema = z.object({
    id: z.string().describe('Unique identifier of the result'),
    score: z.number().describe('Similarity score'),
    payload: z
        .record(z.unknown())
        .optional()
        .describe('Associated metadata/payload'),
    vector: z
        .array(z.number())
        .optional()
        .describe('The result vector (optional)'),
});
/**
 * Vector search result schema
 */
const VectorSearchToolResultSchema = z.object({
    results: z
        .array(VectorSearchResultItemSchema)
        .describe('Array of search results ranked by similarity'),
    totalResults: z
        .number()
        .describe('Total number of results returned'),
    queryMetadata: z.object({
        collectionName: z.string().describe('Collection searched'),
        vectorDimension: z.number().describe('Dimension of query vector'),
        distanceMetric: z.string().describe('Distance metric used'),
        searchTime: z
            .number()
            .describe('Search execution time in milliseconds'),
    }),
    success: z.boolean().describe('Whether the search was successful'),
    error: z.string().describe('Error message if search failed'),
});
/**
 * Vector Search Tool
 * Performs similarity search in vector databases
 */
export class VectorSearchTool extends ToolBubble {
    /**
     * REQUIRED STATIC METADATA
     */
    static type = 'tool';
    static bubbleName = 'vector-search-tool';
    static schema = VectorSearchToolParamsSchema;
    static resultSchema = VectorSearchToolResultSchema;
    static shortDescription = 'Perform vector similarity search using Qdrant or other vector databases';
    static longDescription = `
    A powerful tool for performing vector similarity searches in vector databases.

    Features:
    - High-performance vector similarity search
    - Support for multiple distance metrics (cosine, euclidean, dot product)
    - Configurable top-K result retrieval
    - Score-based filtering for relevant results
    - Metadata filtering capabilities
    - Integration with Qdrant vector database

    Use cases:
    - Semantic search in document repositories
    - Recommendation systems (content-based filtering)
    - Image similarity search
    - Duplicate detection
    - Natural language query matching
    - Clustering and nearest neighbor queries

    Distance Metrics:
    - COSINE: Measures cosine similarity (best for normalized vectors)
    - EUCLIDEAN: Measures Euclidean distance (L2 norm)
    - DOT_PRODUCT: Measures dot product similarity

    Requirements:
    - QDRANT_CRED credential for Qdrant access
    - Collection must exist in the vector database
    - Query vector dimension must match collection dimension
  `;
    static alias = 'vector-search';
    constructor(params, context) {
        super(params, context);
    }
    /**
     * Main action method - performs vector similarity search
     */
    async performAction(context) {
        void context; // Context available but not currently used
        const startTime = Date.now();
        try {
            const { vector, topK, scoreThreshold, collectionName, vectorDimension } = this.params;
            console.log(`[VectorSearchTool] Starting vector similarity search in collection: ${collectionName}`);
            console.log(`[VectorSearchTool] Vector dimension: ${vector.length}`);
            console.log(`[VectorSearchTool] Requested top-K: ${topK}`);
            console.log(`[VectorSearchTool] Score threshold: ${scoreThreshold}`);
            // Validate query vector
            this.validateVector(vector, vectorDimension);
            // Perform the search using HTTP request to Qdrant
            const results = await this.performQdrantSearch();
            // Filter results by score threshold
            const filteredResults = results.filter((r) => r.score >= scoreThreshold);
            const searchTime = Date.now() - startTime;
            console.log(`[VectorSearchTool] Search completed. Found ${filteredResults.length} results above threshold`);
            console.log(`[VectorSearchTool] Search time: ${searchTime}ms`);
            return {
                results: filteredResults,
                totalResults: filteredResults.length,
                queryMetadata: {
                    collectionName,
                    vectorDimension,
                    distanceMetric: this.params.distanceMetric,
                    searchTime,
                },
                success: true,
                error: '',
            };
        }
        catch (error) {
            const searchTime = Date.now() - startTime;
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            console.error(`[VectorSearchTool] Search failed: ${errorMessage}`);
            return {
                results: [],
                totalResults: 0,
                queryMetadata: {
                    collectionName: this.params.collectionName,
                    vectorDimension: this.params.vectorDimension,
                    distanceMetric: this.params.distanceMetric,
                    searchTime,
                },
                success: false,
                error: errorMessage,
            };
        }
    }
    /**
     * Validate query vector
     */
    validateVector(vector, expectedDimension) {
        if (!Array.isArray(vector)) {
            throw new Error('Query vector must be an array');
        }
        if (vector.length === 0) {
            throw new Error('Query vector cannot be empty');
        }
        if (vector.length !== expectedDimension) {
            throw new Error(`Vector dimension mismatch: expected ${expectedDimension}, got ${vector.length}`);
        }
        // Check for NaN or infinite values
        for (let i = 0; i < vector.length; i++) {
            if (!Number.isFinite(vector[i])) {
                throw new Error(`Vector contains invalid value at index ${i}: ${vector[i]}`);
            }
        }
    }
    /**
     * Perform vector search using Qdrant HTTP API or in-memory computation
     */
    async performQdrantSearch() {
        // Get credential and parse config
        const credential = this.params.credentials?.[CredentialType.QDRANT_CRED];
        let apiKey = process.env.QDRANT_API_KEY;
        let qdrantUrl = process.env.QDRANT_URL || 'http://localhost:6333';
        if (credential) {
            try {
                const config = typeof credential === 'string' ? JSON.parse(credential) : credential;
                apiKey = config.apiKey || apiKey;
                qdrantUrl = config.url || qdrantUrl;
            }
            catch (error) {
                console.warn('[VectorSearchTool] Failed to parse Qdrant credential, using env vars');
            }
        }
        // If no Qdrant connection available, perform in-memory computation
        if (!apiKey && !qdrantUrl.includes('localhost')) {
            console.log('[VectorSearchTool] No Qdrant credentials, using in-memory computation');
            return this.performInMemorySearch();
        }
        try {
            // Prepare search request
            const searchRequest = {
                limit: this.params.topK,
                vector: this.params.vector,
                with_payload: true,
                with_vector: false,
                score_threshold: this.params.scoreThreshold,
            };
            // Add filter if provided
            if (this.params.filter) {
                Object.assign(searchRequest, { filter: this.params.filter });
            }
            const headers = {
                'Content-Type': 'application/json',
            };
            if (apiKey) {
                headers['api-key'] = apiKey;
            }
            // Make HTTP request to Qdrant
            const response = await fetch(`${qdrantUrl}/collections/${this.params.collectionName}/points/search`, {
                method: 'POST',
                headers,
                body: JSON.stringify(searchRequest),
                signal: AbortSignal.timeout(30000), // 30 second timeout
            });
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Qdrant search failed: ${response.status} ${response.statusText} - ${errorText}`);
            }
            const data = await response.json();
            // Transform Qdrant results to our format
            return data.result.map((item) => ({
                id: item.id,
                score: item.score,
                payload: item.payload,
                vector: item.vector,
            }));
        }
        catch (error) {
            console.warn('[VectorSearchTool] Qdrant connection failed, falling back to in-memory search');
            return this.performInMemorySearch();
        }
    }
    /**
     * Perform in-memory vector similarity search
     * Useful for testing or when no vector database is available
     */
    performInMemorySearch() {
        // For in-memory search, we need vector data in the filter
        const vectors = this.params.filter?.vectors;
        if (!vectors || !Array.isArray(vectors)) {
            throw new Error('In-memory search requires vectors array in filter parameter');
        }
        const queryVector = this.params.vector;
        const metric = this.params.distanceMetric;
        console.log(`[VectorSearchTool] Computing ${metric} similarity for ${vectors.length} vectors`);
        // Calculate similarity scores
        const results = vectors
            .map((item) => ({
            id: item.id,
            score: this.calculateSimilarity(queryVector, item.vector, metric),
            payload: item.payload,
            vector: item.vector,
        }))
            .sort((a, b) => b.score - a.score) // Sort by score descending
            .slice(0, this.params.topK); // Take top-K results
        console.log(`[VectorSearchTool] Found ${results.length} results`);
        return results;
    }
    /**
     * Calculate similarity between two vectors using the specified metric
     */
    calculateSimilarity(vecA, vecB, metric) {
        if (vecA.length !== vecB.length) {
            throw new Error(`Vector dimension mismatch: ${vecA.length} vs ${vecB.length}`);
        }
        switch (metric) {
            case VectorDistanceMetric.COSINE:
                return this.cosineSimilarity(vecA, vecB);
            case VectorDistanceMetric.EUCLIDEAN:
                // Convert Euclidean distance to similarity (inverse)
                const distance = this.euclideanDistance(vecA, vecB);
                return 1 / (1 + distance); // Scale to 0-1 range
            case VectorDistanceMetric.DOT_PRODUCT:
                return this.dotProduct(vecA, vecB);
            default:
                throw new Error(`Unknown distance metric: ${metric}`);
        }
    }
    /**
     * Calculate cosine similarity between two vectors
     * Returns value in range [-1, 1], where 1 is identical
     */
    cosineSimilarity(vecA, vecB) {
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }
        const denominator = Math.sqrt(normA) * Math.sqrt(normB);
        if (denominator === 0) {
            return 0; // One or both vectors are zero vectors
        }
        return dotProduct / denominator;
    }
    /**
     * Calculate Euclidean distance (L2 norm) between two vectors
     * Returns non-negative value, where 0 is identical
     */
    euclideanDistance(vecA, vecB) {
        let sumSquaredDiff = 0;
        for (let i = 0; i < vecA.length; i++) {
            const diff = vecA[i] - vecB[i];
            sumSquaredDiff += diff * diff;
        }
        return Math.sqrt(sumSquaredDiff);
    }
    /**
     * Calculate dot product of two vectors
     * Returns unnormalized similarity score
     */
    dotProduct(vecA, vecB) {
        let product = 0;
        for (let i = 0; i < vecA.length; i++) {
            product += vecA[i] * vecB[i];
        }
        return product;
    }
    /**
     * Batch compute similarity for multiple query vectors
     * Optimized for performance when searching with many vectors
     */
    batchSimilarity(queryVectors, targetVectors, metric) {
        const results = [];
        for (const queryVec of queryVectors) {
            const row = [];
            for (const targetVec of targetVectors) {
                row.push(this.calculateSimilarity(queryVec, targetVec, metric));
            }
            results.push(row);
        }
        return results;
    }
}
//# sourceMappingURL=vector-search-tool.js.map