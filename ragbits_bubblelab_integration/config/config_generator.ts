/**
 * Enhanced Configuration Generator for Ragbits + BubbleLab Integration
 * 
 * This module generates Ragbits configurations from BubbleLab workflow definitions
 */

import { BubbleLabWorkflowConfig, RagbitsConfig, RagbitsNodeConfig, GenerationOptions, GeneratedConfig } from '../types';
import { ConfigMapper } from './config_mapper';

export class ConfigGenerator {
  /**
   * Generates a Ragbits configuration from a BubbleLab workflow
   */
  static generate(
    bubbleLabConfig: BubbleLabWorkflowConfig,
    options: GenerationOptions = {}
  ): GeneratedConfig {
    const opts = {
      includeComments: true,
      format: 'json' as const,
      validate: true,
      generateDeploymentFiles: false,
      targetEnvironment: 'development' as const,
      ...options
    };

    // Generate the base Ragbits configuration
    const ragbitsConfig: RagbitsConfig = {
      documentProcessor: {
        embedding_model: 'text-embedding-3-small',
        vector_store_type: 'memory',
        chunk_size: 1000,
        chunk_overlap: 200,
        min_chunk_size: 100,
      },
      search: {
        default_top_k: 5,
        default_score_threshold: 0.0,
        enable_hybrid_search: false,
        enable_reranking: false,
      },
      generation: {
        default_model: 'gpt-4o',
        default_temperature: 0.7,
        default_max_tokens: 1000,
      },
      workflow: {
        name: bubbleLabConfig.name,
        description: bubbleLabConfig.description,
        nodes: [],
        connections: [],
      },
    };

    // Process each node in the workflow
    for (const node of bubbleLabConfig.nodes) {
      const ragbitsNode = this.convertNode(node);
      if (ragbitsNode) {
        ragbitsConfig.workflow.nodes.push(ragbitsNode);
      }
    }

    // Process connections
    for (const edge of bubbleLabConfig.edges) {
      ragbitsConfig.workflow.connections.push({
        sourceNodeId: edge.source,
        sourceOutput: edge.sourceHandle || 'output',
        targetNodeId: edge.target,
        targetInput: edge.targetHandle || 'input',
      });
    }

    // Add environment-specific configurations
    if (opts.targetEnvironment) {
      this.applyEnvironmentSettings(ragbitsConfig, opts.targetEnvironment);
    }

    // Validate if requested
    let validationErrors: string[] | undefined;
    if (opts.validate) {
      validationErrors = this.validateConfig(ragbitsConfig);
    }

    // Generate deployment manifest if requested
    let deploymentManifest: any;
    if (opts.generateDeploymentFiles) {
      deploymentManifest = this.generateDeploymentManifest(bubbleLabConfig, opts);
    }

    return {
      ragbitsConfig,
      deploymentManifest,
      validationErrors,
    };
  }

  /**
   * Converts a BubbleLab node to a Ragbits node configuration
   */
  private static convertNode(node: any): RagbitsNodeConfig | null {
    switch (node.type) {
      case 'ragbits-ingest':
        return {
          id: node.id,
          type: 'ingest',
          config: {
            sourceType: node.data.sourceType || 'file',
            sourcePath: node.data.sourcePath || '',
            metadata: node.data.metadata || {},
            chunkSize: node.data.chunkSize || 1000,
            chunkOverlap: node.data.chunkOverlap || 200,
          },
          inputs: ['source', 'content'],
          outputs: ['documentId', 'chunksIngested'],
        };

      case 'ragbits-search':
        return {
          id: node.id,
          type: 'search',
          config: {
            topK: node.data.topK || 5,
            scoreThreshold: node.data.scoreThreshold || 0.0,
            enableHybridSearch: node.data.enableHybridSearch || false,
            defaultFilters: node.data.defaultFilters || {},
          },
          inputs: ['query', 'filters'],
          outputs: ['results', 'totalResults'],
        };

      case 'ragbits-generation':
        return {
          id: node.id,
          type: 'generation',
          config: {
            llmModel: node.data.llmModel || 'gpt-4o',
            temperature: node.data.temperature || 0.7,
            maxTokens: node.data.maxTokens || 1000,
            systemPrompt: node.data.systemPrompt || '',
          },
          inputs: ['query', 'context'],
          outputs: ['response', 'tokensUsed'],
        };

      case 'ragbits-index':
        return {
          id: node.id,
          type: 'index',
          config: {
            vectorStoreType: node.data.vectorStoreType || 'memory',
            embeddingModel: node.data.embeddingModel || 'text-embedding-3-small',
            autoRefresh: node.data.autoRefresh || false,
            refreshInterval: node.data.refreshInterval || 300,
          },
          inputs: ['operation'],
          outputs: ['result', 'stats'],
        };

      default:
        console.warn(`Unknown node type: ${node.type}`);
        return null;
    }
  }

  /**
   * Applies environment-specific settings to the configuration
   */
  private static applyEnvironmentSettings(config: RagbitsConfig, environment: string): void {
    switch (environment) {
      case 'production':
        // Use more robust settings for production
        config.documentProcessor.vector_store_type = 'qdrant'; // More scalable than memory
        config.documentProcessor.chunk_size = 500; // Smaller chunks for better precision
        config.search.default_top_k = 10;
        config.search.enable_hybrid_search = true;
        config.generation.default_temperature = 0.3; // More deterministic
        config.generation.default_max_tokens = 2000;
        break;

      case 'staging':
        // Medium settings for staging
        config.documentProcessor.vector_store_type = 'qdrant';
        config.documentProcessor.chunk_size = 750;
        config.search.default_top_k = 7;
        config.search.enable_hybrid_search = true;
        config.generation.default_temperature = 0.5;
        config.generation.default_max_tokens = 1500;
        break;

      case 'development':
      default:
        // Default settings remain as initialized
        break;
    }
  }

  /**
   * Validates the generated configuration
   */
  private static validateConfig(config: RagbitsConfig): string[] {
    const errors: string[] = [];

    // Validate document processor settings
    if (!config.documentProcessor.embedding_model) {
      errors.push('Document processor: embedding_model is required');
    }

    if (!['memory', 'qdrant'].includes(config.documentProcessor.vector_store_type)) {
      errors.push(`Document processor: invalid vector_store_type '${config.documentProcessor.vector_store_type}'`);
    }

    if (config.documentProcessor.chunk_size <= 0) {
      errors.push('Document processor: chunk_size must be positive');
    }

    if (config.documentProcessor.chunk_overlap < 0) {
      errors.push('Document processor: chunk_overlap cannot be negative');
    }

    if (config.documentProcessor.chunk_overlap >= config.documentProcessor.chunk_size) {
      errors.push('Document processor: chunk_overlap must be less than chunk_size');
    }

    // Validate search settings
    if (config.search.default_top_k <= 0) {
      errors.push('Search: default_top_k must be positive');
    }

    if (config.search.default_score_threshold < 0 || config.search.default_score_threshold > 1) {
      errors.push('Search: default_score_threshold must be between 0 and 1');
    }

    // Validate generation settings
    if (!config.generation.default_model) {
      errors.push('Generation: default_model is required');
    }

    if (config.generation.default_temperature < 0 || config.generation.default_temperature > 2) {
      errors.push('Generation: default_temperature should be between 0 and 2');
    }

    if (config.generation.default_max_tokens <= 0) {
      errors.push('Generation: default_max_tokens must be positive');
    }

    // Validate workflow settings
    if (!config.workflow.name) {
      errors.push('Workflow: name is required');
    }

    // Validate nodes
    for (const node of config.workflow.nodes) {
      if (!node.id) {
        errors.push(`Workflow node: missing id`);
      }
      if (!['ingest', 'search', 'generation', 'index'].includes(node.type)) {
        errors.push(`Workflow node ${node.id}: invalid type '${node.type}'`);
      }
    }

    return errors;
  }

  /**
   * Generates a deployment manifest for the workflow
   */
  private static generateDeploymentManifest(
    bubbleLabConfig: BubbleLabWorkflowConfig,
    options: GenerationOptions
  ): any {
    return {
      apiVersion: 'ragbits/v1',
      kind: 'RAGWorkflow',
      metadata: {
        name: bubbleLabConfig.name.toLowerCase().replace(/\s+/g, '-'),
        labels: {
          'app.kubernetes.io/name': bubbleLabConfig.name.toLowerCase().replace(/\s+/g, '-'),
          'app.kubernetes.io/part-of': 'ragbits',
          'app.kubernetes.io/managed-by': 'bubblelab',
        },
      },
      spec: {
        workflow: {
          name: bubbleLabConfig.name,
          description: bubbleLabConfig.description,
          nodes: bubbleLabConfig.nodes.map(node => ({
            id: node.id,
            type: node.type,
            position: node.position,
          })),
          edges: bubbleLabConfig.edges.map(edge => ({
            id: edge.id,
            source: edge.source,
            target: edge.target,
          })),
        },
        resources: {
          requests: {
            cpu: '500m',
            memory: '1Gi',
          },
          limits: {
            cpu: '1000m',
            memory: '2Gi',
          },
        },
        environment: options.targetEnvironment,
        replicas: options.targetEnvironment === 'production' ? 3 : 1,
        autoscaling: {
          enabled: options.targetEnvironment === 'production',
          minReplicas: 1,
          maxReplicas: 10,
          targetCPUUtilizationPercentage: 80,
        },
        monitoring: {
          enabled: true,
          loki: {
            enabled: true,
          },
          prometheus: {
            enabled: true,
          },
        },
      },
    };
  }

  /**
   * Formats the configuration in the requested format
   */
  static formatConfig(config: GeneratedConfig, format: 'json' | 'yaml' | 'typescript'): string {
    switch (format) {
      case 'json':
        return JSON.stringify(config.ragbitsConfig, null, 2);
      
      case 'yaml':
        // Since we're in TypeScript, we'll return a JSON string that represents YAML structure
        // In a real implementation, we'd use a YAML library
        return JSON.stringify(config.ragbitsConfig, null, 2);
      
      case 'typescript':
        return this.generateTypeScriptConfig(config.ragbitsConfig);
      
      default:
        return JSON.stringify(config.ragbitsConfig, null, 2);
    }
  }

  /**
   * Generates a TypeScript configuration file
   */
  private static generateTypeScriptConfig(config: RagbitsConfig): string {
    let tsConfig = `// Ragbits Configuration for ${config.workflow.name}\n\n`;
    
    tsConfig += 'import { RagbitsConfig } from \'@ragbits/core\';\n\n';
    
    tsConfig += 'export const ragbitsConfig: RagbitsConfig = {\n';
    tsConfig += `  // Document processor configuration\n`;
    tsConfig += `  documentProcessor: {\n`;
    tsConfig += `    embedding_model: "${config.documentProcessor.embedding_model}",\n`;
    tsConfig += `    vector_store_type: "${config.documentProcessor.vector_store_type}",\n`;
    tsConfig += `    chunk_size: ${config.documentProcessor.chunk_size},\n`;
    tsConfig += `    chunk_overlap: ${config.documentProcessor.chunk_overlap},\n`;
    tsConfig += `    min_chunk_size: ${config.documentProcessor.min_chunk_size},\n`;
    tsConfig += `  },\n\n`;
    
    tsConfig += `  // Search configuration\n`;
    tsConfig += `  search: {\n`;
    tsConfig += `    default_top_k: ${config.search.default_top_k},\n`;
    tsConfig += `    default_score_threshold: ${config.search.default_score_threshold},\n`;
    tsConfig += `    enable_hybrid_search: ${config.search.enable_hybrid_search},\n`;
    tsConfig += `    enable_reranking: ${config.search.enable_reranking},\n`;
    tsConfig += `  },\n\n`;
    
    tsConfig += `  // Generation configuration\n`;
    tsConfig += `  generation: {\n`;
    tsConfig += `    default_model: "${config.generation.default_model}",\n`;
    tsConfig += `    default_temperature: ${config.generation.default_temperature},\n`;
    tsConfig += `    default_max_tokens: ${config.generation.default_max_tokens},\n`;
    tsConfig += `  },\n\n`;
    
    tsConfig += `  // Workflow configuration\n`;
    tsConfig += `  workflow: {\n`;
    tsConfig += `    name: "${config.workflow.name}",\n`;
    tsConfig += `    description: "${config.workflow.description}",\n`;
    tsConfig += `    nodes: [\n`;
    
    for (const node of config.workflow.nodes) {
      tsConfig += `      {\n`;
      tsConfig += `        id: "${node.id}",\n`;
      tsConfig += `        type: "${node.type}",\n`;
      tsConfig += `        config: ${JSON.stringify(node.config, null, 4).replace(/\n/g, '\n        ')},\n`;
      tsConfig += `      },\n`;
    }
    
    tsConfig += `    ],\n`;
    tsConfig += `    connections: [\n`;
    
    for (const conn of config.workflow.connections) {
      tsConfig += `      {\n`;
      tsConfig += `        sourceNodeId: "${conn.sourceNodeId}",\n`;
      tsConfig += `        sourceOutput: "${conn.sourceOutput}",\n`;
      tsConfig += `        targetNodeId: "${conn.targetNodeId}",\n`;
      tsConfig += `        targetInput: "${conn.targetInput}",\n`;
      tsConfig += `      },\n`;
    }
    
    tsConfig += `    ],\n`;
    tsConfig += `  },\n`;
    tsConfig += `};\n\n`;
    
    tsConfig += `export default ragbitsConfig;\n`;
    
    return tsConfig;
  }
}