/**
 * Input/Output type definitions for RAGBits operations
 */

/**
 * RAGBits Ingest Input Interface
 * Input data for document ingestion operations
 */
export interface RAGBitsIngestInput {
  /**
   * Source type for ingestion
   */
  sourceType: 'file' | 'url' | 'text';
  
  /**
   * File paths for file-based ingestion
   */
  filePaths?: string[];
  
  /**
   * URLs for web-based ingestion
   */
  urls?: string[];
  
  /**
   * Raw text content for direct ingestion
   */
  textContent?: string;
  
  /**
   * Authentication credentials
   */
  auth?: {
    username?: string;
    password?: string;
    apiKey?: string;
    headers?: Record<string, string>;
  };
  
  /**
   * Processing options
   */
  options?: {
    chunkSize?: number;
    chunkOverlap?: number;
    metadata?: Record<string, any>;
  };
}

/**
 * RAGBits Ingest Output Interface
 * Output data from document ingestion operations
 */
export interface RAGBitsIngestOutput {
  /**
   * Successfully processed documents
   */
  documents: ProcessedDocument[];
  
  /**
   * Processing statistics
   */
  stats: ProcessingStats;
  
  /**
   * Any errors encountered during processing
   */
  errors?: Error[];
}

/**
 * RAGBits Search Input Interface
 * Input data for semantic search operations
 */
export interface RAGBitsSearchInput {
  /**
   * Search query
   */
  query: string;
  
  /**
   * Search parameters
   */
  params?: {
    topK?: number;
    similarityThreshold?: number;
    filters?: Record<string, any>;
  };
  
  /**
   * Context from previous operations
   */
  context?: any;
}

/**
 * RAGBits Search Output Interface
 * Output data from semantic search operations
 */
export interface RAGBitsSearchOutput {
  /**
   * Search results
   */
  results: SearchResult[];
  
  /**
   * Search statistics
   */
  stats: SearchStats;
  
  /**
   * Query metadata
   */
  queryMetadata: {
    originalQuery: string;
    processedQuery: string;
    timestamp: Date;
  };
}

/**
 * RAGBits Generation Input Interface
 * Input data for response generation operations
 */
export interface RAGBitsGenerationInput {
  /**
   * Context for generation (from search results)
   */
  context: string | SearchResult[];
  
  /**
   * Original query
   */
  query: string;
  
  /**
   * Generation parameters
   */
  params?: {
    model?: string;
    temperature?: number;
    maxTokens?: number;
  };
  
  /**
   * Prompt template
   */
  promptTemplate?: string;
}

/**
 * RAGBits Generation Output Interface
 * Output data from response generation operations
 */
export interface RAGBitsGenerationOutput {
  /**
   * Generated response
   */
  response: string;
  
  /**
   * Generation metadata
   */
  metadata: {
    model: string;
    tokensUsed: number;
    completionTime: number;
    confidenceScore?: number;
  };
  
  /**
   * Context used for generation
   */
  contextSummary: string;
  
  /**
   * Quality metrics
   */
  quality?: {
    coherence?: number;
    relevance?: number;
    completeness?: number;
  };
}

/**
 * RAGBits Index Input Interface
 * Input data for index management operations
 */
export interface RAGBitsIndexInput {
  /**
   * Operation to perform
   */
  operation: 'stats' | 'refresh' | 'clear' | 'optimize';
  
  /**
   * Operation parameters
   */
  params?: {
    indexName?: string;
    force?: boolean;
    timeout?: number;
  };
}

/**
 * RAGBits Index Output Interface
 * Output data from index management operations
 */
export interface RAGBitsIndexOutput {
  /**
   * Operation result
   */
  success: boolean;
  
  /**
   * Operation metadata
   */
  metadata: {
    operation: string;
    timestamp: Date;
    duration: number;
  };
  
  /**
   * Result data (varies by operation)
   */
  data?: {
    stats?: IndexStats;
    message?: string;
    affectedDocuments?: number;
  };
}

/**
 * Processed Document Interface
 * Represents a document after processing
 */
export interface ProcessedDocument {
  /**
   * Document identifier
   */
  id: string;
  
  /**
   * Document content
   */
  content: string;
  
  /**
   * Document metadata
   */
  metadata: Record<string, any>;
  
  /**
   * Document embeddings (if available)
   */
  embeddings?: number[];
  
  /**
   * Processing timestamp
   */
  processedAt: Date;
  
  /**
   * Source information
   */
  source: {
    type: 'file' | 'url' | 'text';
    originalPath?: string;
    originalUrl?: string;
  };
}

/**
 * Processing Statistics Interface
 * Statistics about document processing
 */
export interface ProcessingStats {
  /**
   * Total documents processed
   */
  totalDocuments: number;
  
  /**
   * Successful documents
   */
  successful: number;
  
  /**
   * Failed documents
   */
  failed: number;
  
  /**
   * Total processing time (ms)
   */
  processingTime: number;
  
  /**
   * Average processing time per document (ms)
   */
  avgProcessingTime: number;
  
  /**
   * Total tokens processed
   */
  tokensProcessed?: number;
}

/**
 * Workflow Execution Result Interface
 * Result of workflow execution
 */
export interface WorkflowExecutionResult {
  /**
   * Execution identifier
   */
  executionId: string;
  
  /**
   * Workflow identifier
   */
  workflowId: string;
  
  /**
   * Execution status
   */
  status: 'success' | 'partial' | 'failed';
  
  /**
   * Node results
   */
  nodeResults: Record<string, any>;
  
  /**
   * Execution statistics
   */
  stats: {
    startTime: Date;
    endTime: Date;
    totalDuration: number;
    successfulNodes: number;
    failedNodes: number;
  };
  
  /**
   * Errors encountered
   */
  errors?: Error[];
}

/**
 * Workflow Execution Options Interface
 * Options for workflow execution
 */
export interface WorkflowExecutionOptions {
  /**
   * Timeout for execution (ms)
   */
  timeout?: number;
  
  /**
   * Maximum retries for failed nodes
   */
  maxRetries?: number;
  
  /**
   * Debug mode
   */
  debug?: boolean;
  
  /**
   * Logging level
   */
  logLevel?: 'debug' | 'info' | 'warn' | 'error';
  
  /**
   * Custom context
   */
  context?: Record<string, any>;
}

/**
 * Generation Options Interface
 * Options for response generation
 */
export interface GenerationOptions {
  /**
   * Model to use
   */
  model?: string;
  
  /**
   * Generation parameters
   */
  parameters?: {
    temperature?: number;
    maxTokens?: number;
    topP?: number;
  };
  
  /**
   * Prompt configuration
   */
  prompt?: {
    system?: string;
    user?: string;
    assistant?: string;
  };
}

/**
 * Generated Configuration Interface
 * Configuration generated from workflow
 */
export interface GeneratedConfig {
  /**
   * Configuration type
   */
  type: 'ragbits' | 'typescript' | 'deployment';
  
  /**
   * Configuration content
   */
  content: string | object;
  
  /**
   * Generation metadata
   */
  metadata: {
    generatedAt: Date;
    workflowId: string;
    environment?: string;
  };
}

/**
 * Search Result Interface
 * Individual search result
 */
export interface SearchResult {
  /**
   * Document identifier
   */
  documentId: string;
  
  /**
   * Document content
   */
  content: string;
  
  /**
   * Similarity score
   */
  score: number;
  
  /**
   * Document metadata
   */
  metadata: Record<string, any>;
  
  /**
   * Highlights from the document
   */
  highlights?: string[];
}

/**
 * Search Statistics Interface
 * Statistics about search operation
 */
export interface SearchStats {
  /**
   * Total documents searched
   */
  totalDocuments: number;
  
  /**
   * Documents returned
   */
  documentsReturned: number;
  
  /**
   * Search duration (ms)
   */
  searchTime: number;
  
  /**
   * Reranking time (ms)
   */
  rerankTime?: number;
}

/**
 * Index Statistics Interface
 * Statistics about index
 */
export interface IndexStats {
  /**
   * Total documents in index
   */
  documentCount: number;
  
  /**
   * Index size (bytes)
   */
  indexSize: number;
  
  /**
   * Last updated timestamp
   */
  lastUpdated: Date;
  
  /**
   * Index health status
   */
  health: 'green' | 'yellow' | 'red';
}