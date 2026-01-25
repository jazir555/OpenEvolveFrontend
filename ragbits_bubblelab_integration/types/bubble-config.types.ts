/**
 * Bubble Configuration Types for Ragbits + BubbleLab Integration
 */

export interface BubbleConfig {
  id: string;
  name: string;
  description: string;
  [key: string]: any; // Allow additional properties
}

export interface RAGBitsIngestConfig extends BubbleConfig {
  sourceType: 'file' | 'url' | 'text';
  sourcePath: string;
  metadata?: Record<string, any>;
  chunkSize?: number;
  chunkOverlap?: number;
}

export interface RAGBitsSearchConfig extends BubbleConfig {
  topK?: number;
  scoreThreshold?: number;
  enableHybridSearch?: boolean;
  defaultFilters?: Record<string, any>;
}

export interface RAGBitsGenerationConfig extends BubbleConfig {
  llmModel?: string;
  temperature?: number;
  maxTokens?: number;
  systemPrompt?: string;
}

export interface RAGBitsIndexConfig extends BubbleConfig {
  vectorStoreType?: 'memory' | 'qdrant';
  embeddingModel?: string;
  autoRefresh?: boolean;
  refreshInterval?: number; // in seconds
}

export interface BubbleLabNode {
  id: string;
  type: string; // 'ragbits-ingest', 'ragbits-search', 'ragbits-generation', 'ragbits-index'
  position: { x: number; y: number };
  data: Record<string, any>;
}

export interface BubbleLabEdge {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
}

export interface BubbleLabWorkflowConfig {
  id: string;
  name: string;
  description: string;
  nodes: Array<BubbleLabNode>;
  edges: Array<BubbleLabEdge>;
  metadata: Record<string, any>;
}

export interface RagbitsNodeConfig {
  id: string;
  type: 'ingest' | 'search' | 'generation' | 'index';
  config: any;
  inputs?: string[];
  outputs?: string[];
}

export interface RagbitsConnection {
  sourceNodeId: string;
  sourceOutput: string;
  targetNodeId: string;
  targetInput: string;
}

export interface RagbitsConfig {
  documentProcessor: {
    embedding_model: string;
    vector_store_type: 'memory' | 'qdrant';
    qdrant_url?: string;
    qdrant_collection?: string;
    chunk_size: number;
    chunk_overlap: number;
    min_chunk_size: number;
  };
  search: {
    default_top_k: number;
    default_score_threshold: number;
    enable_hybrid_search: boolean;
    enable_reranking: boolean;
  };
  generation: {
    default_model: string;
    default_temperature: number;
    default_max_tokens: number;
  };
  workflow: {
    name: string;
    description: string;
    nodes: Array<RagbitsNodeConfig>;
    connections: Array<RagbitsConnection>;
  };
}