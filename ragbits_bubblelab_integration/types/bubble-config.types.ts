/**
 * Bubble Configuration Types for Ragbits + BubbleLab Integration
 */

/**
 * Base configuration interface for all bubble components
 */
export interface BubbleConfig {
  /** Unique identifier for the bubble */
  id: string;
  /** Human-readable name for the bubble */
  name: string;
  /** Description of the bubble's purpose */
  description: string;
  /** Additional properties allowed for extensibility */
  [key: string]: any;
}

/**
 * Configuration for RAGBits document ingestion bubble
 */
export interface RAGBitsIngestConfig extends BubbleConfig {
  /** Type of source for ingestion */
  sourceType: 'file' | 'url' | 'text';
  /** Path to the source document or directory */
  sourcePath: string;
  /** Additional metadata to attach to ingested documents */
  metadata?: Record<string, any>;
  /** Size of text chunks when processing documents (default: 1000) */
  chunkSize?: number;
  /** Overlap between consecutive chunks (default: 200) */
  chunkOverlap?: number;
}

/**
 * Configuration for RAGBits semantic search bubble
 */
export interface RAGBitsSearchConfig extends BubbleConfig {
  /** Number of results to return (default: 5) */
  topK?: number;
  /** Minimum similarity score threshold (default: 0.0) */
  scoreThreshold?: number;
  /** Whether to enable hybrid search combining semantic and keyword search (default: false) */
  enableHybridSearch?: boolean;
  /** Default filters to apply to all searches */
  defaultFilters?: Record<string, any>;
}

/**
 * Configuration for RAGBits response generation bubble
 */
export interface RAGBitsGenerationConfig extends BubbleConfig {
  /** LLM model to use for generation (default: 'gpt-4o') */
  llmModel?: string;
  /** Sampling temperature for generation (default: 0.7) */
  temperature?: number;
  /** Maximum number of tokens in the generated response (default: 1000) */
  maxTokens?: number;
  /** System prompt to guide the LLM's behavior */
  systemPrompt?: string;
}

/**
 * Configuration for RAGBits index management bubble
 */
export interface RAGBitsIndexConfig extends BubbleConfig {
  /** Type of vector store to use (default: 'memory') */
  vectorStoreType?: 'memory' | 'qdrant';
  /** Embedding model to use for vector generation (default: 'text-embedding-3-small') */
  embeddingModel?: string;
  /** Whether to automatically refresh the index periodically (default: false) */
  autoRefresh?: boolean;
  /** Interval in seconds for automatic index refresh (default: 300) */
  refreshInterval?: number;
}

/**
 * Represents a node in a BubbleLab workflow
 */
export interface BubbleLabNode {
  /** Unique identifier for the node */
  id: string;
  /** Type of the node (determines its functionality) */
  type: string; // 'ragbits-ingest', 'ragbits-search', 'ragbits-generation', 'ragbits-index'
  /** Position of the node in the visual editor */
  position: { x: number; y: number };
  /** Configuration data for the node */
  data: Record<string, any>;
}

/**
 * Represents an edge connecting two nodes in a BubbleLab workflow
 */
export interface BubbleLabEdge {
  /** Unique identifier for the edge */
  id: string;
  /** ID of the source node */
  source: string;
  /** ID of the target node */
  target: string;
  /** Specific output handle on the source node (if applicable) */
  sourceHandle?: string;
  /** Specific input handle on the target node (if applicable) */
  targetHandle?: string;
}

/**
 * Complete configuration for a BubbleLab workflow
 */
export interface BubbleLabWorkflowConfig {
  /** Unique identifier for the workflow */
  id: string;
  /** Human-readable name for the workflow */
  name: string;
  /** Description of the workflow's purpose */
  description: string;
  /** List of nodes in the workflow */
  nodes: Array<BubbleLabNode>;
  /** List of edges connecting the nodes */
  edges: Array<BubbleLabEdge>;
  /** Additional metadata for the workflow */
  metadata: Record<string, any>;
}

/**
 * Configuration for a node in the Ragbits workflow
 */
export interface RagbitsNodeConfig {
  /** Unique identifier for the node */
  id: string;
  /** Type of operation the node performs */
  type: 'ingest' | 'search' | 'generation' | 'index';
  /** Specific configuration for the node's operation */
  config: any;
  /** List of input names the node accepts */
  inputs?: string[];
  /** List of output names the node produces */
  outputs?: string[];
}

/**
 * Represents a connection between two nodes in a Ragbits workflow
 */
export interface RagbitsConnection {
  /** ID of the source node */
  sourceNodeId: string;
  /** Name of the output from the source node */
  sourceOutput: string;
  /** ID of the target node */
  targetNodeId: string;
  /** Name of the input on the target node */
  targetInput: string;
}

/**
 * Complete configuration for a Ragbits system
 */
export interface RagbitsConfig {
  /** Configuration for the document processor component */
  documentProcessor: {
    /** Model to use for generating embeddings */
    embedding_model: string;
    /** Type of vector store to use */
    vector_store_type: 'memory' | 'qdrant';
    /** URL for Qdrant server (if using Qdrant) */
    qdrant_url?: string;
    /** Name of the Qdrant collection (if using Qdrant) */
    qdrant_collection?: string;
    /** Size of text chunks when processing documents */
    chunk_size: number;
    /** Overlap between consecutive chunks */
    chunk_overlap: number;
    /** Minimum size for a text chunk */
    min_chunk_size: number;
  };
  /** Configuration for the search component */
  search: {
    /** Default number of results to return */
    default_top_k: number;
    /** Default minimum similarity score threshold */
    default_score_threshold: number;
    /** Whether to enable hybrid search by default */
    enable_hybrid_search: boolean;
    /** Whether to enable reranking of results */
    enable_reranking: boolean;
  };
  /** Configuration for the generation component */
  generation: {
    /** Default LLM model to use for generation */
    default_model: string;
    /** Default sampling temperature */
    default_temperature: number;
    /** Default maximum number of tokens in responses */
    default_max_tokens: number;
  };
  /** Configuration for the workflow component */
  workflow: {
    /** Name of the workflow */
    name: string;
    /** Description of the workflow */
    description: string;
    /** List of nodes in the workflow */
    nodes: Array<RagbitsNodeConfig>;
    /** List of connections between nodes */
    connections: Array<RagbitsConnection>;
  };
}