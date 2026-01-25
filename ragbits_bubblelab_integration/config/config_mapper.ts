/**
 * Configuration Mapper for Ragbits + BubbleLab Integration
 * 
 * This module maps BubbleLab workflow definitions to Ragbits configurations
 */

import { 
  BubbleLabWorkflowConfig, 
  BubbleLabNode, 
  BubbleLabEdge, 
  RagbitsConfig, 
  RagbitsNodeConfig, 
  RagbitsConnection,
  RAGBitsIngestConfig,
  RAGBitsSearchConfig,
  RAGBitsGenerationConfig,
  RAGBitsIndexConfig
} from '../types';

export class ConfigMapper {
  /**
   * Maps a BubbleLab workflow configuration to Ragbits configuration
   */
  static mapBubbleLabToRagbits(bubbleLabConfig: BubbleLabWorkflowConfig): RagbitsConfig {
    const nodes: RagbitsNodeConfig[] = [];
    const connections: RagbitsConnection[] = [];

    // Map each node based on its type
    for (const node of bubbleLabConfig.nodes) {
      const ragbitsNode = this.mapNode(node);
      if (ragbitsNode) {
        nodes.push(ragbitsNode);
      }
    }

    // Map edges to connections
    for (const edge of bubbleLabConfig.edges) {
      const connection: RagbitsConnection = {
        sourceNodeId: edge.source,
        sourceOutput: edge.sourceHandle || 'output',
        targetNodeId: edge.target,
        targetInput: edge.targetHandle || 'input',
      };
      connections.push(connection);
    }

    return {
      documentProcessor: this.extractDocumentProcessorConfig(nodes),
      search: this.extractSearchConfig(nodes),
      generation: this.extractGenerationConfig(nodes),
      workflow: {
        name: bubbleLabConfig.name,
        description: bubbleLabConfig.description,
        nodes,
        connections,
      },
    };
  }

  /**
   * Maps a single BubbleLab node to a Ragbits node configuration
   */
  private static mapNode(node: BubbleLabNode): RagbitsNodeConfig | null {
    switch (node.type) {
      case 'ragbits-ingest':
        return {
          id: node.id,
          type: 'ingest',
          config: this.mapIngestConfig(node.data),
          inputs: ['source', 'content'],
          outputs: ['documentId', 'chunksIngested'],
        };
      case 'ragbits-search':
        return {
          id: node.id,
          type: 'search',
          config: this.mapSearchConfig(node.data),
          inputs: ['query', 'filters'],
          outputs: ['results', 'totalResults'],
        };
      case 'ragbits-generation':
        return {
          id: node.id,
          type: 'generation',
          config: this.mapGenerationConfig(node.data),
          inputs: ['query', 'context'],
          outputs: ['response', 'tokensUsed'],
        };
      case 'ragbits-index':
        return {
          id: node.id,
          type: 'index',
          config: this.mapIndexConfig(node.data),
          inputs: ['operation'],
          outputs: ['result', 'stats'],
        };
      default:
        console.warn(`Unknown node type: ${node.type}`);
        return null;
    }
  }

  /**
   * Maps BubbleLab ingest configuration to Ragbits ingest configuration
   */
  private static mapIngestConfig(data: Record<string, any>): RAGBitsIngestConfig {
    return {
      id: data.id || '',
      name: data.name || 'RAGBits Ingest',
      description: data.description || 'Ingest documents into RAGBits',
      sourceType: data.sourceType || 'file',
      sourcePath: data.sourcePath || '',
      metadata: data.metadata || {},
      chunkSize: data.chunkSize || 1000,
      chunkOverlap: data.chunkOverlap || 200,
    };
  }

  /**
   * Maps BubbleLab search configuration to Ragbits search configuration
   */
  private static mapSearchConfig(data: Record<string, any>): RAGBitsSearchConfig {
    return {
      id: data.id || '',
      name: data.name || 'RAGBits Search',
      description: data.description || 'Perform semantic search with RAGBits',
      topK: data.topK || 5,
      scoreThreshold: data.scoreThreshold || 0.0,
      enableHybridSearch: data.enableHybridSearch || false,
      defaultFilters: data.defaultFilters || {},
    };
  }

  /**
   * Maps BubbleLab generation configuration to Ragbits generation configuration
   */
  private static mapGenerationConfig(data: Record<string, any>): RAGBitsGenerationConfig {
    return {
      id: data.id || '',
      name: data.name || 'RAGBits Generation',
      description: data.description || 'Generate responses with RAGBits',
      llmModel: data.llmModel || 'gpt-4o',
      temperature: data.temperature || 0.7,
      maxTokens: data.maxTokens || 1000,
      systemPrompt: data.systemPrompt || '',
    };
  }

  /**
   * Maps BubbleLab index configuration to Ragbits index configuration
   */
  private static mapIndexConfig(data: Record<string, any>): RAGBitsIndexConfig {
    return {
      id: data.id || '',
      name: data.name || 'RAGBits Index',
      description: data.description || 'Manage RAGBits vector index',
      vectorStoreType: data.vectorStoreType || 'memory',
      embeddingModel: data.embeddingModel || 'text-embedding-3-small',
      autoRefresh: data.autoRefresh || false,
      refreshInterval: data.refreshInterval || 300, // 5 minutes
    };
  }

  /**
   * Extract document processor configuration from nodes
   */
  private static extractDocumentProcessorConfig(nodes: RagbitsNodeConfig[]): RagbitsConfig['documentProcessor'] {
    // Look for index nodes to get vector store and embedding model settings
    const indexNodes = nodes.filter(node => node.type === 'index');
    if (indexNodes.length > 0) {
      const indexConfig = indexNodes[0].config as RAGBitsIndexConfig;
      return {
        embedding_model: indexConfig.embeddingModel || 'text-embedding-3-small',
        vector_store_type: indexConfig.vectorStoreType || 'memory',
        chunk_size: 1000, // default
        chunk_overlap: 200, // default
        min_chunk_size: 100, // default
      };
    }

    // Default configuration
    return {
      embedding_model: 'text-embedding-3-small',
      vector_store_type: 'memory',
      chunk_size: 1000,
      chunk_overlap: 200,
      min_chunk_size: 100,
    };
  }

  /**
   * Extract search configuration from nodes
   */
  private static extractSearchConfig(nodes: RagbitsNodeConfig[]): RagbitsConfig['search'] {
    // Look for search nodes to get search settings
    const searchNodes = nodes.filter(node => node.type === 'search');
    if (searchNodes.length > 0) {
      const searchConfig = searchNodes[0].config as RAGBitsSearchConfig;
      return {
        default_top_k: searchConfig.topK || 5,
        default_score_threshold: searchConfig.scoreThreshold || 0.0,
        enable_hybrid_search: searchConfig.enableHybridSearch || false,
        enable_reranking: false, // Not directly mapped from current config
      };
    }

    // Default configuration
    return {
      default_top_k: 5,
      default_score_threshold: 0.0,
      enable_hybrid_search: false,
      enable_reranking: false,
    };
  }

  /**
   * Extract generation configuration from nodes
   */
  private static extractGenerationConfig(nodes: RagbitsNodeConfig[]): RagbitsConfig['generation'] {
    // Look for generation nodes to get LLM settings
    const generationNodes = nodes.filter(node => node.type === 'generation');
    if (generationNodes.length > 0) {
      const genConfig = generationNodes[0].config as RAGBitsGenerationConfig;
      return {
        default_model: genConfig.llmModel || 'gpt-4o',
        default_temperature: genConfig.temperature || 0.7,
        default_max_tokens: genConfig.maxTokens || 1000,
      };
    }

    // Default configuration
    return {
      default_model: 'gpt-4o',
      default_temperature: 0.7,
      default_max_tokens: 1000,
    };
  }

  /**
   * Validates a BubbleLab workflow configuration
   */
  static validateBubbleLabConfig(config: BubbleLabWorkflowConfig): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Validate nodes
    for (const node of config.nodes) {
      if (!node.id) {
        errors.push(`Node missing ID`);
      }
      if (!node.type) {
        errors.push(`Node ${node.id || 'unknown'} missing type`);
      }
      if (!node.data) {
        errors.push(`Node ${node.id || 'unknown'} missing data`);
      }
    }

    // Validate edges
    for (const edge of config.edges) {
      if (!edge.source) {
        errors.push(`Edge missing source`);
      }
      if (!edge.target) {
        errors.push(`Edge missing target`);
      }

      // Check if referenced nodes exist
      const sourceExists = config.nodes.some(n => n.id === edge.source);
      const targetExists = config.nodes.some(n => n.id === edge.target);
      if (!sourceExists) {
        errors.push(`Edge references non-existent source node: ${edge.source}`);
      }
      if (!targetExists) {
        errors.push(`Edge references non-existent target node: ${edge.target}`);
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }
}