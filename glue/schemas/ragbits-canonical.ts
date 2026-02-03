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
export const DocumentChunk = z.object({
  id: z.string().uuid().describe("Unique identifier for the document chunk"),
  content: z.string()
    .min(1, "Document chunk content cannot be empty")
    .describe("The text content of the document chunk"),
  source: z.string()
    .describe("Source document or identifier where this chunk originated"),
  chunk_index: z.number()
    .int("Chunk index must be an integer")
    .min(0, "Chunk index must be non-negative")
    .describe("Index of this chunk within the source document"),
  metadata: z.record(z.any()).optional()
    .describe("Additional metadata about the chunk (e.g., page number, section)"),
  embedding: z.array(z.number()).optional()
    .describe("Vector embedding for semantic search (optional)"),
  timestamp: z.string().datetime()
    .describe("UTC timestamp when chunk was created/processed (ISO-8601)"),
});

export type DocumentChunk = z.infer<typeof DocumentChunk>;

/**
 * RAG Request Schema
 *
 * Represents a request to perform retrieval-augmented generation.
 */
export const RAGRequest = z.object({
  query: z.string()
    .min(1, "Query cannot be empty")
    .max(10000, "Query cannot exceed 10000 characters")
    .describe("The user query or prompt to process"),

  context: z.string().optional()
    .describe("Optional context or conversation history to include"),

  retrieval_count: z.number()
    .int("Retrieval count must be an integer")
    .positive("Retrieval count must be positive")
    .max(100, "Cannot retrieve more than 100 chunks at once")
    .describe("Number of document chunks to retrieve"),

  filters: z.record(z.any()).optional()
    .describe("Optional filters to apply to retrieval (e.g., source, date range)"),

  timeout_ms: z.number()
    .int("Timeout must be an integer")
    .positive("Timeout must be positive")
    .max(300000, "Timeout cannot exceed 5 minutes")
    .describe("Request timeout in milliseconds (MANDATORY)"),

  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),

  metadata: z.record(z.any()).optional()
    .describe("Optional metadata for observability and tracking"),
});

export type RAGRequest = z.infer<typeof RAGRequest>;

/**
 * RAG Response Schema
 *
 * Represents the response from a RAG query with retrieved context and generated content.
 */
export const RAGResponse = z.object({
  results: z.array(DocumentChunk)
    .describe("Array of retrieved document chunks"),

  answer: z.string().optional()
    .describe("Generated answer or response based on retrieved context"),

  embeddings: z.array(z.array(z.number())).optional()
    .describe("Query embedding and/or result embeddings"),

  metadata: z.object({
    retrieval_time_ms: z.number().optional()
      .describe("Time taken for retrieval in milliseconds"),
    generation_time_ms: z.number().optional()
      .describe("Time taken for answer generation in milliseconds"),
    total_time_ms: z.number().optional()
      .describe("Total processing time in milliseconds"),
    model_used: z.string().optional()
      .describe("LLM model used for generation"),
    retrieval_method: z.string().optional()
      .describe("Method used for retrieval (e.g., semantic, keyword)"),
    confidence_score: z.number().min(0).max(1).optional()
      .describe("Confidence score for the answer (0-1)"),
  }).optional().describe("Execution metadata"),

  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),

  timestamp: z.string().datetime()
    .describe("UTC timestamp of the response (ISO-8601)"),
});

export type RAGResponse = z.infer<typeof RAGResponse>;

/**
 * Document Ingestion Request Schema
 *
 * Represents a request to ingest a document into the RAG system.
 */
export const DocumentIngestionRequest = z.object({
  content: z.string()
    .min(1, "Document content cannot be empty")
    .describe("The full text content of the document to ingest"),

  source: z.string()
    .min(1, "Source cannot be empty")
    .describe("Source identifier for the document (e.g., file path, URL)"),

  chunk_size: z.number()
    .int("Chunk size must be an integer")
    .positive("Chunk size must be positive")
    .max(10000, "Chunk size cannot exceed 10000 characters")
    .optional()
    .describe("Target size for document chunks (optional)"),

  chunk_overlap: z.number()
    .int("Chunk overlap must be an integer")
    .min(0, "Chunk overlap must be non-negative")
    .max(1000, "Chunk overlap cannot exceed 1000 characters")
    .optional()
    .describe("Overlap between chunks for context preservation (optional)"),

  metadata: z.record(z.any()).optional()
    .describe("Optional metadata to attach to the document"),

  timeout_ms: z.number()
    .int("Timeout must be an integer")
    .positive("Timeout must be positive")
    .max(300000, "Timeout cannot exceed 5 minutes")
    .describe("Request timeout in milliseconds (MANDATORY)"),

  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),
});

export type DocumentIngestionRequest = z.infer<typeof DocumentIngestionRequest>;

/**
 * Document Ingestion Response Schema
 *
 * Represents the response after ingesting a document.
 */
export const DocumentIngestionResponse = z.object({
  document_id: z.string().uuid()
    .describe("Unique identifier for the ingested document"),

  chunks_created: z.number()
    .int("Chunk count must be an integer")
    .describe("Number of chunks created from the document"),

  metadata: z.object({
    ingestion_time_ms: z.number().optional()
      .describe("Time taken for ingestion in milliseconds"),
    embedding_model: z.string().optional()
      .describe("Embedding model used"),
    total_chunks: z.number().optional()
      .describe("Total chunks in the system after ingestion"),
  }).optional().describe("Ingestion metadata"),

  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),

  timestamp: z.string().datetime()
    .describe("UTC timestamp of the response (ISO-8601)"),
});

export type DocumentIngestionResponse = z.infer<typeof DocumentIngestionResponse>;

/**
 * Error Model
 *
 * Represents errors that can occur during RAG operations.
 */
export const RAGError = z.object({
  code: z.enum([
    'QUERY_TOO_LONG',
    'RETRIEVAL_FAILED',
    'GENERATION_FAILED',
    'TIMEOUT',
    'INVALID_FILTER',
    'EMBEDDING_ERROR',
    'DOCUMENT_NOT_FOUND',
    'INGESTION_FAILED',
    'UNKNOWN_ERROR',
  ]).describe("Error code for categorization"),

  message: z.string()
    .describe("Human-readable error message"),

  details: z.record(z.any()).optional()
    .describe("Additional error details"),

  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for tracing the error"),

  timestamp: z.string().datetime()
    .describe("UTC timestamp when the error occurred (ISO-8601)"),
});

export type RAGError = z.infer<typeof RAGError>;

/**
 * Transformation Functions
 */

/**
 * Transform raw RAGbits API response to canonical RAGResponse
 */
export function transformRAGResponseToCanonical(
  rawResponse: any,
  correlationId?: string
): RAGResponse {
  const timestamp = new Date().toISOString();

  return {
    results: (rawResponse.results || []).map((chunk: any) => ({
      id: chunk.id || generateUUID(),
      content: chunk.content,
      source: chunk.source || 'unknown',
      chunk_index: chunk.chunk_index || 0,
      metadata: chunk.metadata,
      embedding: chunk.embedding,
      timestamp: chunk.timestamp || timestamp,
    })),
    answer: rawResponse.answer,
    embeddings: rawResponse.embeddings,
    metadata: {
      retrieval_time_ms: rawResponse.retrieval_time,
      generation_time_ms: rawResponse.generation_time,
      total_time_ms: rawResponse.total_time,
      model_used: rawResponse.model,
      retrieval_method: rawResponse.retrieval_method,
      confidence_score: rawResponse.confidence,
    },
    correlation_id: correlationId,
    timestamp,
  };
}

/**
 * Transform canonical RAGRequest to RAGbits API format
 */
export function transformCanonicalToRAGRequest(
  canonicalRequest: RAGRequest
): any {
  return {
    query: canonicalRequest.query,
    context: canonicalRequest.context,
    retrieval_count: canonicalRequest.retrieval_count,
    filters: canonicalRequest.filters,
    timeout: canonicalRequest.timeout_ms,
    metadata: canonicalRequest.metadata,
  };
}

/**
 * Validate a RAGRequest against the schema
 */
export function validateRAGRequest(data: unknown): {
  success: boolean;
  data?: RAGRequest;
  errors?: string[];
} {
  const result = RAGRequest.safeParse(data);

  if (result.success) {
    return { success: true, data: result.data };
  }

  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

/**
 * Validate a RAGResponse against the schema
 */
export function validateRAGResponse(data: unknown): {
  success: boolean;
  data?: RAGResponse;
  errors?: string[];
} {
  const result = RAGResponse.safeParse(data);

  if (result.success) {
    return { success: true, data: result.data };
  }

  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

/**
 * Validate a DocumentChunk against the schema
 */
export function validateDocumentChunk(data: unknown): {
  success: boolean;
  data?: DocumentChunk;
  errors?: string[];
} {
  const result = DocumentChunk.safeParse(data);

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

/**
 * Check if data is a valid RAGRequest
 */
export function isRAGRequest(data: unknown): data is RAGRequest {
  return typeof data === 'object' && data !== null &&
    'query' in data && 'retrieval_count' in data && 'timeout_ms' in data;
}

/**
 * Check if data is a valid RAGResponse
 */
export function isRAGResponse(data: unknown): data is RAGResponse {
  return typeof data === 'object' && data !== null &&
    'results' in data && 'timestamp' in data;
}

/**
 * Helper function to generate UUID
 */
function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Example usage and validation examples
 */
export const RAGExamples = {
  validRAGRequest: {
    query: "What are the key principles of machine learning?",
    context: "Previous conversation about AI",
    retrieval_count: 5,
    filters: {
      source: "ml_textbook",
      date_after: "2024-01-01",
    },
    timeout_ms: 10000,
    correlation_id: "550e8400-e29b-41d4-a716-446655440000",
    metadata: {
      user_id: "user123",
      session_id: "session456",
    },
  } as RAGRequest,

  validRAGResponse: {
    results: [
      {
        id: "550e8400-e29b-41d4-a716-446655440001",
        content: "Machine learning is based on several key principles...",
        source: "ml_textbook_chapter1",
        chunk_index: 0,
        metadata: {
          page: 15,
          section: "Introduction",
        },
        timestamp: "2025-02-03T12:34:56.789Z",
      },
      {
        id: "550e8400-e29b-41d4-a716-446655440002",
        content: "The first principle is data-driven learning...",
        source: "ml_textbook_chapter1",
        chunk_index: 1,
        metadata: {
          page: 16,
          section: "Principles",
        },
        timestamp: "2025-02-03T12:34:56.789Z",
      },
    ],
    answer: "The key principles of machine learning include...",
    embeddings: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    metadata: {
      retrieval_time_ms: 150,
      generation_time_ms: 2000,
      total_time_ms: 2150,
      model_used: "gpt-4",
      retrieval_method: "semantic",
      confidence_score: 0.92,
    },
    correlation_id: "550e8400-e29b-41d4-a716-446655440000",
    timestamp: "2025-02-03T12:34:56.789Z",
  } as RAGResponse,

  validDocumentIngestionRequest: {
    content: "This is the full document content to be ingested...",
    source: "/documents/ml_paper.pdf",
    chunk_size: 1000,
    chunk_overlap: 200,
    metadata: {
      author: "John Doe",
      title: "Introduction to ML",
      publication_date: "2024-01-15",
    },
    timeout_ms: 30000,
    correlation_id: "550e8400-e29b-41d4-a716-446655440000",
  } as DocumentIngestionRequest,

  validDocumentIngestionResponse: {
    document_id: "550e8400-e29b-41d4-a716-446655440003",
    chunks_created: 42,
    metadata: {
      ingestion_time_ms: 5000,
      embedding_model: "text-embedding-ada-002",
      total_chunks: 1000,
    },
    correlation_id: "550e8400-e29b-41d4-a716-446655440000",
    timestamp: "2025-02-03T12:34:56.789Z",
  } as DocumentIngestionResponse,

  validRAGError: {
    code: 'RETRIEVAL_FAILED',
    message: "Failed to retrieve relevant documents",
    details: {
      reason: "Vector database connection timeout",
      query: "What is machine learning?",
    },
    correlation_id: "550e8400-e29b-41d4-a716-446655440000",
    timestamp: "2025-02-03T12:34:56.789Z",
  } as RAGError,
};
