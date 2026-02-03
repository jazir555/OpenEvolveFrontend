/**
 * Configuration Mapper
 * Maps BubbleLab workflow configurations to RAGBits configurations
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
  RAGBitsIndexConfig,
  isRAGBitsIngestConfig,
  isRAGBitsSearchConfig,
  isRAGBitsGenerationConfig,
  isRAGBitsIndexConfig
} from '../types';

/**
 * ConfigMapper - Maps BubbleLab configurations to RAGBits configurations
 * Handles validation, transformation, and error collection
 */
export class ConfigMapper {
  private validationErrors: string[] = [];
  
  /**
   * Map BubbleLab workflow configuration to RAGBits configuration
   * @param workflowConfig - BubbleLab workflow configuration
   * @returns RAGBits configuration
   */
  public mapBubbleLabToRagbits(workflowConfig: BubbleLabWorkflowConfig): RagbitsConfig {
    this.validationErrors = [];
    
    if (!isBubbleLabWorkflowConfig(workflowConfig)) {
      this.addValidationError('Invalid BubbleLab workflow configuration');
      throw new Error('Invalid workflow configuration: ' + this.validationErrors.join(', '));
    }
    
    try {
      // Map nodes
      const nodes = workflowConfig.nodes.map(node => this.mapNode(node));
      
      // Map edges to connections
      const connections = workflowConfig.edges.map(edge => this.mapEdge(edge));
      
      // Create RAGBits configuration
      const ragbitsConfig: RagbitsConfig = {
        id: workflowConfig.id,
        name: workflowConfig.name,
        nodes: nodes,
        connections: connections,
        globalConfig: {
          logging: {
            level: 'info'
          },
          caching: {
            enabled: true,
            ttl: 300000 // 5 minutes
          }
        }
      };
      
      this.validateRagbitsConfig(ragbitsConfig);
      
      if (this.validationErrors.length > 0) {
        throw new Error('Configuration validation failed: ' + this.validationErrors.join(', '));
      }
      
      return ragbitsConfig;
    } catch (error) {
      this.addValidationError(error instanceof Error ? error.message : 'Unknown error during mapping');
      throw new Error('Configuration mapping failed: ' + this.validationErrors.join(', '));
    }
  }
  
  /**
   * Map individual node
   * @param node - BubbleLab node
   * @returns RAGBits node configuration
   */
  private mapNode(node: BubbleLabNode): RagbitsNodeConfig {
    try {
      // Validate node structure
      if (!node || !node.id || !node.type) {
        throw new Error(`Invalid node: ${node?.id || 'unknown'}`);
      }
      
      // Map based on node type
      switch (node.type) {
        case 'ragbits-ingest':
          return this.mapIngestConfig(node);
        case 'ragbits-search':
          return this.mapSearchConfig(node);
        case 'ragbits-generation':
          return this.mapGenerationConfig(node);
        case 'ragbits-index':
          return this.mapIndexConfig(node);
        default:
          throw new Error(`Unknown node type: ${node.type}`);
      }
    } catch (error) {
      this.addValidationError(`Node ${node.id} mapping failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      throw error;
    }
  }
  
  /**
   * Map ingest node configuration
   * @param node - BubbleLab ingest node
   * @returns RAGBits ingest node configuration
   */
  private mapIngestConfig(node: BubbleLabNode): RagbitsNodeConfig {
    if (!isRAGBitsIngestConfig(node.config)) {
      throw new Error(`Invalid ingest configuration for node ${node.id}`);
    }
    
    return {
      id: node.id,
      type: 'ingest',
      config: this.extractDocumentProcessorConfig(node.config),
      inputs: [],
      outputs: []
    };
  }
  
  /**
   * Map search node configuration
   * @param node - BubbleLab search node
   * @returns RAGBits search node configuration
   */
  private mapSearchConfig(node: BubbleLabNode): RagbitsNodeConfig {
    if (!isRAGBitsSearchConfig(node.config)) {
      throw new Error(`Invalid search configuration for node ${node.id}`);
    }
    
    return {
      id: node.id,
      type: 'search',
      config: this.extractSearchConfig(node.config),
      inputs: [],
      outputs: []
    };
  }
  
  /**
   * Map generation node configuration
   * @param node - BubbleLab generation node
   * @returns RAGBits generation node configuration
   */
  private mapGenerationConfig(node: BubbleLabNode): RagbitsNodeConfig {
    if (!isRAGBitsGenerationConfig(node.config)) {
      throw new Error(`Invalid generation configuration for node ${node.id}`);
    }
    
    return {
      id: node.id,
      type: 'generation',
      config: this.extractGenerationConfig(node.config),
      inputs: [],
      outputs: []
    };
  }
  
  /**
   * Map index node configuration
   * @param node - BubbleLab index node
   * @returns RAGBits index node configuration
   */
  private mapIndexConfig(node: BubbleLabNode): RagbitsNodeConfig {
    if (!isRAGBitsIndexConfig(node.config)) {
      throw new Error(`Invalid index configuration for node ${node.id}`);
    }
    
    return {
      id: node.id,
      type: 'index',
      config: node.config,
      inputs: [],
      outputs: []
    };
  }
  
  /**
   * Map edge to connection
   * @param edge - BubbleLab edge
   * @returns RAGBits connection
   */
  private mapEdge(edge: BubbleLabEdge): RagbitsConnection {
    if (!edge || !edge.source || !edge.target) {
      throw new Error(`Invalid edge: ${edge?.id || 'unknown'}`);
    }
    
    return {
      source: edge.source,
      target: edge.target,
      metadata: edge.metadata || {}
    };
  }
  
  /**
   * Extract document processor configuration
   * @param config - Ingest configuration
   * @returns Processed configuration
   */
  private extractDocumentProcessorConfig(config: RAGBitsIngestConfig): RAGBitsIngestConfig {
    return {
      sourceType: config.sourceType,
      filePaths: config.filePaths,
      urls: config.urls,
      textContent: config.textContent,
      auth: config.auth,
      processingOptions: config.processingOptions || {
        chunkSize: 1000,
        chunkOverlap: 200,
        metadataExtraction: true,
        textCleaning: true,
        languageDetection: false
      },
      metadata: config.metadata
    };
  }
  
  /**
   * Extract search configuration
   * @param config - Search configuration
   * @returns Processed configuration
   */
  private extractSearchConfig(config: RAGBitsSearchConfig): RAGBitsSearchConfig {
    return {
      searchStrategy: config.searchStrategy,
      topK: config.topK || 5,
      similarityThreshold: config.similarityThreshold || 0.7,
      filters: config.filters || {},
      rerank: config.rerank || {
        enabled: false,
        model: 'default',
        topN: 10
      }
    };
  }
  
  /**
   * Extract generation configuration
   * @param config - Generation configuration
   * @returns Processed configuration
   */
  private extractGenerationConfig(config: RAGBitsGenerationConfig): RAGBitsGenerationConfig {
    return {
      model: config.model,
      parameters: config.parameters || {
        temperature: 0.7,
        maxTokens: 500,
        topP: 1.0,
        presencePenalty: 0.0,
        frequencyPenalty: 0.0
      },
      promptTemplate: config.promptTemplate || {
        system: 'You are a helpful AI assistant.',
        user: 'Answer the question based on the context: {query}\\nContext: {context}',
        assistant: ''
      },
      contextFormatting: config.contextFormatting || {
        maxContextLength: 2000,
        contextTruncation: 'end',
        separator: '\n\n'
      }
    };
  }
  
  /**
   * Validate RAGBits configuration
   * @param config - RAGBits configuration to validate
   */
  private validateRagbitsConfig(config: RagbitsConfig): void {
    if (!config.id || typeof config.id !== 'string') {
      this.addValidationError('Configuration ID is required');
    }
    
    if (!config.name || typeof config.name !== 'string') {
      this.addValidationError('Configuration name is required');
    }
    
    if (!config.nodes || !Array.isArray(config.nodes) || config.nodes.length === 0) {
      this.addValidationError('At least one node is required');
    }
    
    // Validate nodes
    config.nodes.forEach(node => {
      this.validateNode(node);
    });
    
    // Validate connections
    if (config.connections) {
      config.connections.forEach(connection => {
        this.validateConnection(connection, config.nodes);
      });
    }
    
    // Check for circular dependencies
    this.checkCircularDependencies(config);
    
    // Check for orphaned nodes
    this.checkOrphanedNodes(config);
  }
  
  /**
   * Validate individual node
   * @param node - Node to validate
   */
  private validateNode(node: RagbitsNodeConfig): void {
    if (!node.id || typeof node.id !== 'string') {
      this.addValidationError(`Node ${node.id || 'unknown'} has invalid ID`);
    }
    
    if (!node.type || !['ingest', 'search', 'generation', 'index'].includes(node.type)) {
      this.addValidationError(`Node ${node.id} has invalid type`);
    }
    
    if (!node.config || typeof node.config !== 'object') {
      this.addValidationError(`Node ${node.id} has invalid configuration`);
    }
    
    // Type-specific validation
    switch (node.type) {
      case 'ingest':
        if (!isRAGBitsIngestConfig(node.config)) {
          this.addValidationError(`Node ${node.id} has invalid ingest configuration`);
        }
        break;
      case 'search':
        if (!isRAGBitsSearchConfig(node.config)) {
          this.addValidationError(`Node ${node.id} has invalid search configuration`);
        }
        break;
      case 'generation':
        if (!isRAGBitsGenerationConfig(node.config)) {
          this.addValidationError(`Node ${node.id} has invalid generation configuration`);
        }
        break;
      case 'index':
        if (!isRAGBitsIndexConfig(node.config)) {
          this.addValidationError(`Node ${node.id} has invalid index configuration`);
        }
        break;
    }
  }
  
  /**
   * Validate connection
   * @param connection - Connection to validate
   * @param nodes - All nodes in configuration
   */
  private validateConnection(connection: RagbitsConnection, nodes: RagbitsNodeConfig[]): void {
    if (!connection.source || !connection.target) {
      this.addValidationError('Connection has invalid source or target');
      return;
    }
    
    const sourceNode = nodes.find(n => n.id === connection.source);
    const targetNode = nodes.find(n => n.id === connection.target);
    
    if (!sourceNode) {
      this.addValidationError(`Connection source node ${connection.source} not found`);
    }
    
    if (!targetNode) {
      this.addValidationError(`Connection target node ${connection.target} not found`);
    }
    
    if (sourceNode && targetNode && sourceNode.id === targetNode.id) {
      this.addValidationError(`Connection from ${connection.source} to itself is not allowed`);
    }
  }
  
  /**
   * Check for circular dependencies
   * @param config - RAGBits configuration
   */
  private checkCircularDependencies(config: RagbitsConfig): void {
    // Simple cycle detection - would need more sophisticated algorithm for complex graphs
    const nodeIds = config.nodes.map(n => n.id);
    const connectionMap: Record<string, string[]> = {};
    
    // Build connection map
    config.connections.forEach(conn => {
      if (!connectionMap[conn.source]) {
        connectionMap[conn.source] = [];
      }
      connectionMap[conn.source].push(conn.target);
    });
    
    // Check for obvious cycles (this is simplified - real implementation would need DFS)
    for (const nodeId of nodeIds) {
      if (connectionMap[nodeId] && connectionMap[nodeId].includes(nodeId)) {
        this.addValidationError(`Circular dependency detected: ${nodeId} -> ${nodeId}`);
      }
    }
  }
  
  /**
   * Check for orphaned nodes
   * @param config - RAGBits configuration
   */
  private checkOrphanedNodes(config: RagbitsConfig): void {
    const allNodeIds = new Set(config.nodes.map(n => n.id));
    const connectedNodeIds = new Set<string>();
    
    // Add all nodes that are either sources or targets in connections
    config.connections.forEach(conn => {
      connectedNodeIds.add(conn.source);
      connectedNodeIds.add(conn.target);
    });
    
    // Find orphaned nodes (nodes not connected to anything)
    const orphanedNodes = Array.from(allNodeIds).filter(id => !connectedNodeIds.has(id));
    
    if (orphanedNodes.length > 0) {
      this.addValidationError(`Orphaned nodes detected: ${orphanedNodes.join(', ')}`);
    }
  }
  
  /**
   * Add validation error
   * @param error - Error message
   */
  private addValidationError(error: string): void {
    this.validationErrors.push(error);
  }
  
  /**
   * Get validation errors
   * @returns Array of validation errors
   */
  public getValidationErrors(): string[] {
    return [...this.validationErrors];
  }
  
  /**
   * Clear validation errors
   */
  public clearValidationErrors(): void {
    this.validationErrors = [];
  }
}

/**
 * Validate BubbleLab workflow configuration
 * @param config - Configuration to validate
 * @returns True if valid, false otherwise
 */
export function isBubbleLabWorkflowConfig(config: any): config is BubbleLabWorkflowConfig {
  return config && 
         typeof config.id === 'string' && 
         typeof config.name === 'string' &&
         Array.isArray(config.nodes) &&
         Array.isArray(config.edges);
}