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
import { Logger } from '../utils/common.utils';

export class ConfigMapper {
  private static logger = new Logger({ level: 'info', prefix: 'ConfigMapper' });

  /**
   * Maps a BubbleLab workflow configuration to Ragbits configuration
   */
  static mapBubbleLabToRagbits(bubbleLabConfig: BubbleLabWorkflowConfig): RagbitsConfig {
    this.logger.info(`Mapping BubbleLab workflow to Ragbits configuration: ${bubbleLabConfig.name}`);

    // Validate the configuration first
    const validation = this.validateBubbleLabConfig(bubbleLabConfig);
    if (!validation.isValid) {
      this.logger.error(`Configuration validation failed: ${validation.errors.join(', ')}`);
      throw new Error(`Invalid BubbleLab configuration: ${validation.errors.join(', ')}`);
    }

    const nodes: RagbitsNodeConfig[] = [];
    const connections: RagbitsConnection[] = [];

    // Map each node based on its type
    for (const node of bubbleLabConfig.nodes) {
      const ragbitsNode = this.mapNode(node);
      if (ragbitsNode) {
        nodes.push(ragbitsNode);
        this.logger.debug(`Mapped node ${node.id} (${node.type}) to Ragbits node`);
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
      this.logger.debug(`Mapped edge from ${edge.source} to ${edge.target}`);
    }

    const ragbitsConfig: RagbitsConfig = {
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

    this.logger.info(`Successfully mapped workflow with ${nodes.length} nodes and ${connections.length} connections`);
    return ragbitsConfig;
  }

  /**
   * Maps a single BubbleLab node to a Ragbits node configuration
   */
  private static mapNode(node: BubbleLabNode): RagbitsNodeConfig | null {
    this.logger.debug(`Mapping node ${node.id} of type ${node.type}`);

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
        this.logger.warn(`Unknown node type: ${node.type}`);
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
      this.logger.debug(`Extracted document processor config from index node: ${indexConfig.vectorStoreType}, ${indexConfig.embeddingModel}`);
      return {
        embedding_model: indexConfig.embeddingModel || 'text-embedding-3-small',
        vector_store_type: indexConfig.vectorStoreType || 'memory',
        qdrant_url: indexConfig.vectorStoreType === 'qdrant' ? 'http://localhost:6333' : undefined,
        qdrant_collection: indexConfig.vectorStoreType === 'qdrant' ? 'ragbits_default' : undefined,
        chunk_size: 1000, // default
        chunk_overlap: 200, // default
        min_chunk_size: 100, // default
      };
    }

    // Default configuration
    this.logger.debug('Using default document processor configuration');
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
      this.logger.debug(`Extracted search config from search node: topK=${searchConfig.topK}, threshold=${searchConfig.scoreThreshold}`);
      return {
        default_top_k: searchConfig.topK || 5,
        default_score_threshold: searchConfig.scoreThreshold || 0.0,
        enable_hybrid_search: searchConfig.enableHybridSearch || false,
        enable_reranking: false, // Not directly mapped from current config
      };
    }

    // Default configuration
    this.logger.debug('Using default search configuration');
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
      this.logger.debug(`Extracted generation config from generation node: ${genConfig.llmModel}, temp=${genConfig.temperature}`);
      return {
        default_model: genConfig.llmModel || 'gpt-4o',
        default_temperature: genConfig.temperature || 0.7,
        default_max_tokens: genConfig.maxTokens || 1000,
      };
    }

    // Default configuration
    this.logger.debug('Using default generation configuration');
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
    this.logger.debug(`Validating BubbleLab workflow configuration: ${config.name}`);
    const errors: string[] = [];

    // Validate workflow-level properties
    if (!config.id) {
      errors.push('Workflow configuration must have an id');
    }
    if (!config.name) {
      errors.push('Workflow configuration must have a name');
    }
    if (!config.nodes) {
      errors.push('Workflow configuration must have nodes');
    }
    if (!config.edges) {
      errors.push('Workflow configuration must have edges');
    }

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

    const isValid = errors.length === 0;
    if (!isValid) {
      this.logger.warn(`Configuration validation failed with ${errors.length} errors: ${errors.join('; ')}`);
    } else {
      this.logger.info(`Configuration validation passed for workflow: ${config.name}`);
    }

    return {
      isValid,
      errors,
    };
  }

  /**
   * Add node type detection logic
   */
  static detectNodeType(type: string): 'ingest' | 'search' | 'generation' | 'index' | 'unknown' {
    switch (type) {
      case 'ragbits-ingest':
        return 'ingest';
      case 'ragbits-search':
        return 'search';
      case 'ragbits-generation':
        return 'generation';
      case 'ragbits-index':
        return 'index';
      default:
        return 'unknown';
    }
  }

  /**
   * Add 'ragbits-ingest' mapping
   */
  static mapIngestNode(node: BubbleLabNode): RagbitsNodeConfig | null {
    if (node.type !== 'ragbits-ingest') {
      return null;
    }
    return {
      id: node.id,
      type: 'ingest',
      config: this.mapIngestConfig(node.data),
      inputs: ['source', 'content'],
      outputs: ['documentId', 'chunksIngested'],
    };
  }

  /**
   * Add 'ragbits-search' mapping
   */
  static mapSearchNode(node: BubbleLabNode): RagbitsNodeConfig | null {
    if (node.type !== 'ragbits-search') {
      return null;
    }
    return {
      id: node.id,
      type: 'search',
      config: this.mapSearchConfig(node.data),
      inputs: ['query', 'filters'],
      outputs: ['results', 'totalResults'],
    };
  }

  /**
   * Add 'ragbits-generation' mapping
   */
  static mapGenerationNode(node: BubbleLabNode): RagbitsNodeConfig | null {
    if (node.type !== 'ragbits-generation') {
      return null;
    }
    return {
      id: node.id,
      type: 'generation',
      config: this.mapGenerationConfig(node.data),
      inputs: ['query', 'context'],
      outputs: ['response', 'tokensUsed'],
    };
  }

  /**
   * Add 'ragbits-index' mapping
   */
  static mapIndexNode(node: BubbleLabNode): RagbitsNodeConfig | null {
    if (node.type !== 'ragbits-index') {
      return null;
    }
    return {
      id: node.id,
      type: 'index',
      config: this.mapIndexConfig(node.data),
      inputs: ['operation'],
      outputs: ['result', 'stats'],
    };
  }

  /**
   * Add unknown node type handling
   */
  static handleUnknownNodeType(nodeType: string): void {
    this.logger.warn(`Encountered unknown node type: ${nodeType}`);
  }

  /**
   * Add configuration transformation logic
   */
  static transformConfig(config: any, transformationRules: Record<string, any>): any {
    // Apply transformation rules to the configuration
    const transformedConfig = { ...config };

    for (const [key, value] of Object.entries(transformationRules)) {
      if (transformedConfig.hasOwnProperty(key)) {
        transformedConfig[key] = value;
      }
    }

    return transformedConfig;
  }

  /**
   * Add input/output mapping
   */
  static mapNodeIO(nodeType: string): { inputs: string[], outputs: string[] } {
    switch (nodeType) {
      case 'ragbits-ingest':
        return { inputs: ['source', 'content'], outputs: ['documentId', 'chunksIngested'] };
      case 'ragbits-search':
        return { inputs: ['query', 'filters'], outputs: ['results', 'totalResults'] };
      case 'ragbits-generation':
        return { inputs: ['query', 'context'], outputs: ['response', 'tokensUsed'] };
      case 'ragbits-index':
        return { inputs: ['operation'], outputs: ['result', 'stats'] };
      default:
        return { inputs: [], outputs: [] };
    }
  }

  /**
   * Add metadata preservation
   */
  static preserveMetadata(originalConfig: any, mappedConfig: any): any {
    // Preserve any metadata from the original configuration
    if (originalConfig.metadata) {
      mappedConfig.metadata = originalConfig.metadata;
    }
    return mappedConfig;
  }

  /**
   * Add validation for each node type
   */
  static validateNodeType(node: BubbleLabNode): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    switch (node.type) {
      case 'ragbits-ingest':
        if (!node.data.sourceType) {
          errors.push(`Ingest node ${node.id} missing sourceType`);
        }
        break;
      case 'ragbits-search':
        if (node.data.topK && node.data.topK <= 0) {
          errors.push(`Search node ${node.id} topK must be positive`);
        }
        break;
      case 'ragbits-generation':
        if (node.data.temperature && (node.data.temperature < 0 || node.data.temperature > 2)) {
          errors.push(`Generation node ${node.id} temperature must be between 0 and 2`);
        }
        break;
      case 'ragbits-index':
        if (node.data.vectorStoreType && !['memory', 'qdrant'].includes(node.data.vectorStoreType)) {
          errors.push(`Index node ${node.id} invalid vectorStoreType: ${node.data.vectorStoreType}`);
        }
        break;
      default:
        errors.push(`Unknown node type: ${node.type}`);
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }
}