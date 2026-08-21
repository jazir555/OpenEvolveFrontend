/**
 * Core type definitions for BubbleLab and RAGBits configuration
 */

/**
 * RAGBits Ingest Configuration Interface
 * Defines configuration for document ingestion operations
 */
export interface RAGBitsIngestConfig {
  /**
   * Source type for ingestion (file, url, text)
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
   * Authentication configuration for protected sources
   */
  auth?: {
    username?: string;
    password?: string;
    apiKey?: string;
    headers?: Record<string, string>;
  };
  
  /**
   * Processing options for ingestion
   */
  processingOptions?: {
    chunkSize?: number;
    chunkOverlap?: number;
    metadataExtraction?: boolean;
    textCleaning?: boolean;
    languageDetection?: boolean;
  };
  
  /**
   * Metadata to associate with ingested documents
   */
  metadata?: Record<string, any>;
}

/**
 * RAGBits Search Configuration Interface
 * Defines configuration for semantic search operations
 */
export interface RAGBitsSearchConfig {
  /**
   * Search strategy to use
   */
  searchStrategy: 'semantic' | 'keyword' | 'hybrid';
  
  /**
   * Number of top results to return
   */
  topK: number;
  
  /**
   * Minimum similarity threshold (0-1)
   */
  similarityThreshold?: number;
  
  /**
   * Search filters to apply
   */
  filters?: {
    metadata?: Record<string, any>;
    dateRange?: {
      start?: Date | string;
      end?: Date | string;
    };
    sourceTypes?: string[];
  };
  
  /**
   * Reranking configuration
   */
  rerank?: {
    enabled: boolean;
    model?: string;
    topN?: number;
  };
}

/**
 * RAGBits Generation Configuration Interface
 * Defines configuration for response generation operations
 */
export interface RAGBitsGenerationConfig {
  /**
   * LLM model to use for generation
   */
  model: string;
  
  /**
   * Generation parameters
   */
  parameters?: {
    temperature?: number;
    maxTokens?: number;
    topP?: number;
    presencePenalty?: number;
    frequencyPenalty?: number;
  };
  
  /**
   * Prompt template configuration
   */
  promptTemplate?: {
    system?: string;
    user?: string;
    assistant?: string;
  };
  
  /**
   * Context formatting options
   */
  contextFormatting?: {
    maxContextLength?: number;
    contextTruncation?: 'start' | 'end' | 'middle';
    separator?: string;
  };
}

/**
 * RAGBits Index Configuration Interface
 * Defines configuration for index management operations
 */
export interface RAGBitsIndexConfig {
  /**
   * Index operations to perform
   */
  operations: ('stats' | 'refresh' | 'clear' | 'optimize')[];
  
  /**
   * Index-specific configuration
   */
  indexConfig?: {
    indexName?: string;
    shardCount?: number;
    replicaCount?: number;
    refreshInterval?: string;
  };
  
  /**
   * Optimization parameters
   */
  optimization?: {
    mergeFactor?: number;
    maxMergeDocs?: number;
    maxMergeSize?: string;
  };
}

/**
 * Common Bubble Configuration Interface
 * Base configuration shared by all bubble types
 */
export interface BubbleConfig {
  /**
   * Unique identifier for the bubble
   */
  id: string;
  
  /**
   * Human-readable name for the bubble
   */
  name: string;
  
  /**
   * Description of the bubble's purpose
   */
  description?: string;
  
  /**
   * Tags for categorization
   */
  tags?: string[];
  
  /**
   * Metadata associated with the bubble
   */
  metadata?: Record<string, any>;
}

/**
 * BubbleLab Node Interface
 * Represents a node in the BubbleLab workflow
 */
export interface BubbleLabNode extends BubbleConfig {
  /**
   * Type of the node
   */
  type: string;
  
  /**
   * Position in the workflow canvas
   */
  position?: {
    x: number;
    y: number;
  };
  
  /**
   * Node-specific configuration
   */
  config: any;
}

/**
 * BubbleLab Edge Interface
 * Represents a connection between nodes in the workflow
 */
export interface BubbleLabEdge {
  /**
   * Unique identifier for the edge
   */
  id: string;
  
  /**
   * Source node ID
   */
  source: string;
  
  /**
   * Source handle (output port)
   */
  sourceHandle?: string;
  
  /**
   * Target node ID
   */
  target: string;
  
  /**
   * Target handle (input port)
   */
  targetHandle?: string;
  
  /**
   * Edge metadata
   */
  metadata?: Record<string, any>;
}

/**
 * BubbleLab Workflow Configuration Interface
 * Complete workflow definition
 */
export interface BubbleLabWorkflowConfig {
  /**
   * Workflow identifier
   */
  id: string;
  
  /**
   * Workflow name
   */
  name: string;
  
  /**
   * Workflow description
   */
  description?: string;
  
  /**
   * List of nodes in the workflow
   */
  nodes: BubbleLabNode[];
  
  /**
   * List of edges (connections) in the workflow
   */
  edges: BubbleLabEdge[];
  
  /**
   * Workflow metadata
   */
  metadata?: Record<string, any>;
  
  /**
   * Workflow version
   */
  version?: string;
}

/**
 * Ragbits Node Configuration Interface
 * Configuration for a specific Ragbits node
 */
export interface RagbitsNodeConfig {
  /**
   * Node identifier
   */
  id: string;
  
  /**
   * Node type
   */
  type: 'ingest' | 'search' | 'generation' | 'index';
  
  /**
   * Node-specific configuration
   */
  config: RAGBitsIngestConfig | RAGBitsSearchConfig | RAGBitsGenerationConfig | RAGBitsIndexConfig;
  
  /**
   * Input connections
   */
  inputs?: string[];
  
  /**
   * Output connections
   */
  outputs?: string[];
}

/**
 * Ragbits Connection Interface
 * Represents a connection between Ragbits nodes
 */
export interface RagbitsConnection {
  /**
   * Source node ID
   */
  source: string;
  
  /**
   * Target node ID
   */
  target: string;
  
  /**
   * Connection metadata
   */
  metadata?: Record<string, any>;
}

/**
 * Ragbits Configuration Interface
 * Complete Ragbits workflow configuration
 */
export interface RagbitsConfig {
  /**
   * Configuration identifier
   */
  id: string;
  
  /**
   * Configuration name
   */
  name: string;
  
  /**
   * List of nodes
   */
  nodes: RagbitsNodeConfig[];
  
  /**
   * List of connections
   */
  connections: RagbitsConnection[];
  
  /**
   * Global configuration options
   */
  globalConfig?: {
    logging?: {
      level?: 'debug' | 'info' | 'warn' | 'error';
      file?: string;
    };
    caching?: {
      enabled?: boolean;
      ttl?: number;
    };
    processorConfig?: {
      documentProcessor?: {
        chunkSize?: number;
        chunkOverlap?: number;
        embeddingModel?: string;
        vectorStoreType?: string;
      };
      searchConfig?: {
        topK?: number;
        similarityThreshold?: number;
        rerankModel?: string;
      };
      generationConfig?: {
        model?: string;
        temperature?: number;
        maxTokens?: number;
      };
    };
  };
}