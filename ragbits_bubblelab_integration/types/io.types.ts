/**
 * Input/Output Types for Ragbits + BubbleLab Integration
 */

/**
 * Input parameters for the RAGBits document ingestion bubble
 */
export interface RAGBitsIngestInput {
  /** Path to the source document (for file-based ingestion) */
  source?: string;
  /** Text content to ingest directly */
  content?: string;
  /** Additional metadata to associate with the document */
  metadata?: Record<string, any>;
}

/**
 * Output result from the RAGBits document ingestion bubble
 */
export interface RAGBitsIngestOutput {
  /** Whether the ingestion was successful */
  success: boolean;
  /** Unique identifier for the ingested document */
  documentId: string;
  /** Number of text chunks created during ingestion */
  chunksIngested: number;
  /** Time taken to process the document in milliseconds */
  processingTime: number;
  /** Error message if the operation failed */
  error?: string;
}

/**
 * Input parameters for the RAGBits semantic search bubble
 */
export interface RAGBitsSearchInput {
  /** Query text to search for */
  query: string;
  /** Maximum number of results to return (overrides config default) */
  topK?: number;
  /** Minimum similarity score threshold (overrides config default) */
  scoreThreshold?: number;
  /** Additional filters to apply to the search */
  filters?: Record<string, any>;
}

/**
 * Output result from the RAGBits semantic search bubble
 */
export interface RAGBitsSearchOutput {
  /** Whether the search was successful */
  success: boolean;
  /** Array of search results */
  results: Array<{
    /** Unique identifier of the matched document */
    documentId: string;
    /** Content of the matched document/chunk */
    content: string;
    /** Similarity score of the match */
    score: number;
    /** Metadata associated with the document */
    metadata: Record<string, any>;
  }>;
  /** Total number of results returned */
  totalResults: number;
  /** Time taken to execute the search in milliseconds */
  executionTime: number;
  /** Error message if the operation failed */
  error?: string;
}

/**
 * Input parameters for the RAGBits response generation bubble
 */
export interface RAGBitsGenerationInput {
  /** Original query from the user */
  query: string;
  /** Context retrieved from documents to inform the response */
  context?: Array<{
    /** Content of the context document/chunk */
    content: string;
    /** Metadata associated with the context */
    metadata: Record<string, any>;
    /** Relevance score of the context */
    score?: number;
  }>;
  /** Custom prompt to override the default system prompt */
  customPrompt?: string;
}

/**
 * Output result from the RAGBits response generation bubble
 */
export interface RAGBitsGenerationOutput {
  /** Whether the generation was successful */
  success: boolean;
  /** Generated response text */
  response: string;
  /** Time taken to generate the response in milliseconds */
  executionTime: number;
  /** Token usage statistics */
  tokensUsed?: {
    /** Number of tokens in the input prompt */
    input: number;
    /** Number of tokens in the generated output */
    output: number;
    /** Total number of tokens used */
    total: number;
  };
  /** Error message if the operation failed */
  error?: string;
}

/**
 * Input parameters for the RAGBits index management bubble
 */
export interface RAGBitsIndexInput {
  /** Operation to perform on the index */
  operation: 'stats' | 'refresh' | 'clear' | 'optimize';
  /** Whether to force the operation (where applicable) */
  force?: boolean;
}

/**
 * Output result from the RAGBits index management bubble
 */
export interface RAGBitsIndexOutput {
  /** Whether the operation was successful */
  success: boolean;
  /** Operation that was performed */
  operation: string;
  /** Result of the operation */
  result: any;
  /** Time taken to execute the operation in milliseconds */
  executionTime: number;
  /** Error message if the operation failed */
  error?: string;
}

/**
 * Representation of a processed document
 */
export interface ProcessedDocument {
  /** Unique identifier for the document */
  id: string;
  /** Source identifier for the document */
  source: string;
  /** Content of the document (may be truncated) */
  content: string;
  /** Metadata associated with the document */
  metadata: Record<string, any>;
  /** Time taken to process the document in milliseconds */
  processingTime: number;
  /** Whether the processing was successful */
  success: boolean;
  /** Error message if processing failed */
  error?: string;
}

/**
 * Statistics about document processing
 */
export interface ProcessingStats {
  /** Total number of documents processed */
  totalProcessed: number;
  /** Number of successfully processed documents */
  successful: number;
  /** Number of documents that failed to process */
  failed: number;
  /** Average processing time in milliseconds */
  averageProcessingTime: number;
  /** Timestamp of the last processed document */
  lastProcessed: Date;
  /** Current size of the processing queue */
  queueSize: number;
}

/**
 * Result of executing a single workflow node
 */
export interface WorkflowExecutionResult {
  /** Whether the node execution was successful */
  success: boolean;
  /** ID of the node that was executed */
  nodeId: string;
  /** Output produced by the node */
  output: any;
  /** Time taken to execute the node in milliseconds */
  executionTime: number;
  /** Error message if the execution failed */
  error?: string;
}

/**
 * Options for configuring workflow execution
 */
export interface WorkflowExecutionOptions {
  /** Maximum time to wait for a single node to complete (milliseconds) */
  timeout?: number;
  /** Maximum number of retry attempts for failed nodes */
  maxRetries?: number;
  /** Whether to enable logging during execution */
  enableLogging?: boolean;
  /** Minimum level of logs to output */
  logLevel?: 'info' | 'debug' | 'warn' | 'error';
}

/**
 * Options for configuring generation
 */
export interface GenerationOptions {
  /** Whether to include comments in the generated configuration */
  includeComments?: boolean;
  /** Format for the generated configuration */
  format?: 'json' | 'yaml' | 'typescript';
  /** Whether to validate the generated configuration */
  validate?: boolean;
  /** Whether to generate deployment files */
  generateDeploymentFiles?: boolean;
  /** Target environment for the configuration */
  targetEnvironment?: 'development' | 'staging' | 'production';
}

/**
 * Result of configuration generation
 */
export interface GeneratedConfig {
  /** The generated Ragbits configuration */
  ragbitsConfig: RagbitsConfig;
  /** Deployment manifest if requested */
  deploymentManifest?: any;
  /** Environment-specific configuration if applicable */
  environmentConfig?: any;
  /** Validation errors if validation was performed */
  validationErrors?: string[];
}