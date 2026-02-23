/**
 * RAGBits Adapter - Exports
 *
 * Following Federation Constitution:
 * - AIR GAP: No imports from core-projects
 * - RUNTIME TRUTH: Verify before use
 * - IDEMPOTENCY: Safe to retry
 * - CIRCUIT BREAKER: Fail fast on dead service
 *
 * @module ragbits-adapter
 */
export { RAGClient, type RAGClientConfig, type RAGSearchRequest, type RAGIngestRequest, } from './rag-client';
export { RAGBitsAdapter, } from './adapter';
export { DocumentChunk, RAGRequest, RAGResponse, DocumentIngestionRequest, DocumentIngestionResponse, RAGError, transformRAGResponseToCanonical, transformCanonicalToRAGRequest, validateRAGRequest, validateRAGResponse, validateDocumentChunk, isRAGRequest, isRAGResponse, RAGExamples, type DocumentChunk as DocumentChunkType, type RAGRequest as RAGRequestType, type RAGResponse as RAGResponseType, type DocumentIngestionRequest as DocumentIngestionRequestType, type DocumentIngestionResponse as DocumentIngestionResponseType, type RAGError as RAGErrorType, } from '../../schemas/ragbits-canonical';
//# sourceMappingURL=index.d.ts.map