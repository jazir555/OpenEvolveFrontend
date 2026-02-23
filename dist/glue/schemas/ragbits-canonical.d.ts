/**
 * RAGbits Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for RAG (Retrieval-Augmented Generation)
 * interactions through the RAGbits system. All adapters must normalize their data to/from
 * this format.
 *
 * Law of the "Air Gap": This is the ONLY acceptable format for RAG data in the glue layer.
 * Do not pass raw RAGbits API responses between services.
 */
import { z } from 'zod';
/**
 * Document Chunk Schema
 *
 * Represents a chunk of document content that can be retrieved and used for context.
 */
export declare const DocumentChunk: z.ZodObject<{
    id: z.ZodString;
    content: z.ZodString;
    source: z.ZodString;
    chunk_index: z.ZodNumber;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    embedding: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    id: string;
    content: string;
    source: string;
    chunk_index: number;
    metadata?: Record<string, any> | undefined;
    embedding?: number[] | undefined;
}, {
    timestamp: string;
    id: string;
    content: string;
    source: string;
    chunk_index: number;
    metadata?: Record<string, any> | undefined;
    embedding?: number[] | undefined;
}>;
export type DocumentChunk = z.infer<typeof DocumentChunk>;
/**
 * RAG Request Schema
 *
 * Represents a request to perform retrieval-augmented generation.
 */
export declare const RAGRequest: z.ZodObject<{
    query: z.ZodString;
    context: z.ZodOptional<z.ZodString>;
    retrieval_count: z.ZodNumber;
    filters: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    query: string;
    retrieval_count: number;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    context?: string | undefined;
    filters?: Record<string, any> | undefined;
}, {
    timeout_ms: number;
    query: string;
    retrieval_count: number;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    context?: string | undefined;
    filters?: Record<string, any> | undefined;
}>;
export type RAGRequest = z.infer<typeof RAGRequest>;
/**
 * RAG Response Schema
 *
 * Represents the response from a RAG query with retrieved context and generated content.
 */
export declare const RAGResponse: z.ZodObject<{
    results: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        content: z.ZodString;
        source: z.ZodString;
        chunk_index: z.ZodNumber;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        embedding: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
        timestamp: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        timestamp: string;
        id: string;
        content: string;
        source: string;
        chunk_index: number;
        metadata?: Record<string, any> | undefined;
        embedding?: number[] | undefined;
    }, {
        timestamp: string;
        id: string;
        content: string;
        source: string;
        chunk_index: number;
        metadata?: Record<string, any> | undefined;
        embedding?: number[] | undefined;
    }>, "many">;
    answer: z.ZodOptional<z.ZodString>;
    embeddings: z.ZodOptional<z.ZodArray<z.ZodArray<z.ZodNumber, "many">, "many">>;
    metadata: z.ZodOptional<z.ZodObject<{
        retrieval_time_ms: z.ZodOptional<z.ZodNumber>;
        generation_time_ms: z.ZodOptional<z.ZodNumber>;
        total_time_ms: z.ZodOptional<z.ZodNumber>;
        model_used: z.ZodOptional<z.ZodString>;
        retrieval_method: z.ZodOptional<z.ZodString>;
        confidence_score: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        retrieval_method?: string | undefined;
        confidence_score?: number | undefined;
        retrieval_time_ms?: number | undefined;
        generation_time_ms?: number | undefined;
        total_time_ms?: number | undefined;
        model_used?: string | undefined;
    }, {
        retrieval_method?: string | undefined;
        confidence_score?: number | undefined;
        retrieval_time_ms?: number | undefined;
        generation_time_ms?: number | undefined;
        total_time_ms?: number | undefined;
        model_used?: string | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    results: {
        timestamp: string;
        id: string;
        content: string;
        source: string;
        chunk_index: number;
        metadata?: Record<string, any> | undefined;
        embedding?: number[] | undefined;
    }[];
    correlation_id?: string | undefined;
    metadata?: {
        retrieval_method?: string | undefined;
        confidence_score?: number | undefined;
        retrieval_time_ms?: number | undefined;
        generation_time_ms?: number | undefined;
        total_time_ms?: number | undefined;
        model_used?: string | undefined;
    } | undefined;
    answer?: string | undefined;
    embeddings?: number[][] | undefined;
}, {
    timestamp: string;
    results: {
        timestamp: string;
        id: string;
        content: string;
        source: string;
        chunk_index: number;
        metadata?: Record<string, any> | undefined;
        embedding?: number[] | undefined;
    }[];
    correlation_id?: string | undefined;
    metadata?: {
        retrieval_method?: string | undefined;
        confidence_score?: number | undefined;
        retrieval_time_ms?: number | undefined;
        generation_time_ms?: number | undefined;
        total_time_ms?: number | undefined;
        model_used?: string | undefined;
    } | undefined;
    answer?: string | undefined;
    embeddings?: number[][] | undefined;
}>;
export type RAGResponse = z.infer<typeof RAGResponse>;
/**
 * Document Ingestion Request Schema
 *
 * Represents a request to ingest a document into the RAG system.
 */
export declare const DocumentIngestionRequest: z.ZodObject<{
    content: z.ZodString;
    source: z.ZodString;
    chunk_size: z.ZodOptional<z.ZodNumber>;
    chunk_overlap: z.ZodOptional<z.ZodNumber>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    content: string;
    source: string;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    chunk_size?: number | undefined;
    chunk_overlap?: number | undefined;
}, {
    timeout_ms: number;
    content: string;
    source: string;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    chunk_size?: number | undefined;
    chunk_overlap?: number | undefined;
}>;
export type DocumentIngestionRequest = z.infer<typeof DocumentIngestionRequest>;
/**
 * Document Ingestion Response Schema
 *
 * Represents the response after ingesting a document.
 */
export declare const DocumentIngestionResponse: z.ZodObject<{
    document_id: z.ZodString;
    chunks_created: z.ZodNumber;
    metadata: z.ZodOptional<z.ZodObject<{
        ingestion_time_ms: z.ZodOptional<z.ZodNumber>;
        embedding_model: z.ZodOptional<z.ZodString>;
        total_chunks: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        embedding_model?: string | undefined;
        ingestion_time_ms?: number | undefined;
        total_chunks?: number | undefined;
    }, {
        embedding_model?: string | undefined;
        ingestion_time_ms?: number | undefined;
        total_chunks?: number | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    document_id: string;
    chunks_created: number;
    correlation_id?: string | undefined;
    metadata?: {
        embedding_model?: string | undefined;
        ingestion_time_ms?: number | undefined;
        total_chunks?: number | undefined;
    } | undefined;
}, {
    timestamp: string;
    document_id: string;
    chunks_created: number;
    correlation_id?: string | undefined;
    metadata?: {
        embedding_model?: string | undefined;
        ingestion_time_ms?: number | undefined;
        total_chunks?: number | undefined;
    } | undefined;
}>;
export type DocumentIngestionResponse = z.infer<typeof DocumentIngestionResponse>;
/**
 * Error Model
 *
 * Represents errors that can occur during RAG operations.
 */
export declare const RAGError: z.ZodObject<{
    code: z.ZodEnum<["QUERY_TOO_LONG", "RETRIEVAL_FAILED", "GENERATION_FAILED", "TIMEOUT", "INVALID_FILTER", "EMBEDDING_ERROR", "DOCUMENT_NOT_FOUND", "INGESTION_FAILED", "UNKNOWN_ERROR"]>;
    message: z.ZodString;
    details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    message: string;
    code: "TIMEOUT" | "QUERY_TOO_LONG" | "RETRIEVAL_FAILED" | "GENERATION_FAILED" | "INVALID_FILTER" | "EMBEDDING_ERROR" | "DOCUMENT_NOT_FOUND" | "INGESTION_FAILED" | "UNKNOWN_ERROR";
    correlation_id?: string | undefined;
    details?: Record<string, any> | undefined;
}, {
    timestamp: string;
    message: string;
    code: "TIMEOUT" | "QUERY_TOO_LONG" | "RETRIEVAL_FAILED" | "GENERATION_FAILED" | "INVALID_FILTER" | "EMBEDDING_ERROR" | "DOCUMENT_NOT_FOUND" | "INGESTION_FAILED" | "UNKNOWN_ERROR";
    correlation_id?: string | undefined;
    details?: Record<string, any> | undefined;
}>;
export type RAGError = z.infer<typeof RAGError>;
/**
 * Transformation Functions
 */
/**
 * Transform raw RAGbits API response to canonical RAGResponse
 */
export declare function transformRAGResponseToCanonical(rawResponse: any, correlationId?: string): RAGResponse;
/**
 * Transform canonical RAGRequest to RAGbits API format
 */
export declare function transformCanonicalToRAGRequest(canonicalRequest: RAGRequest): any;
/**
 * Validate a RAGRequest against the schema
 */
export declare function validateRAGRequest(data: unknown): {
    success: boolean;
    data?: RAGRequest;
    errors?: string[];
};
/**
 * Validate a RAGResponse against the schema
 */
export declare function validateRAGResponse(data: unknown): {
    success: boolean;
    data?: RAGResponse;
    errors?: string[];
};
/**
 * Validate a DocumentChunk against the schema
 */
export declare function validateDocumentChunk(data: unknown): {
    success: boolean;
    data?: DocumentChunk;
    errors?: string[];
};
/**
 * Type Guards
 */
/**
 * Check if data is a valid RAGRequest
 */
export declare function isRAGRequest(data: unknown): data is RAGRequest;
/**
 * Check if data is a valid RAGResponse
 */
export declare function isRAGResponse(data: unknown): data is RAGResponse;
/**
 * Example usage and validation examples
 */
export declare const RAGExamples: {
    validRAGRequest: RAGRequest;
    validRAGResponse: RAGResponse;
    validDocumentIngestionRequest: DocumentIngestionRequest;
    validDocumentIngestionResponse: DocumentIngestionResponse;
    validRAGError: RAGError;
};
//# sourceMappingURL=ragbits-canonical.d.ts.map