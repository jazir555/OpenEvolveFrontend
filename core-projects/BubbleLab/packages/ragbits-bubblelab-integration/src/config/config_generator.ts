/**
 * Configuration Generator
 * Generates various configuration formats from BubbleLab workflows
 */

import {
  BubbleLabWorkflowConfig,
  BubbleLabNode,
  GeneratedConfig,
  RagbitsConfig,
  RAGBitsIngestConfig,
  RAGBitsSearchConfig,
  RAGBitsGenerationConfig,
  RAGBitsIndexConfig
} from '../types';
import { ConfigMapper } from './config_mapper';

/**
 * ConfigGenerator - Generates configurations from BubbleLab workflows
 * Supports RAGBits config, TypeScript code, and deployment manifests
 */
export class ConfigGenerator {
  private configMapper: ConfigMapper;
  
  /**
   * Constructor
   */
  constructor() {
    this.configMapper = new ConfigMapper();
  }
  
  /**
   * Generate configuration
   * @param workflowConfig - BubbleLab workflow configuration
   * @param type - Configuration type to generate
   * @param environment - Target environment
   * @returns Generated configuration
   */
  public generate(
    workflowConfig: BubbleLabWorkflowConfig,
    type: 'ragbits' | 'typescript' | 'deployment' = 'ragbits',
    environment: 'development' | 'staging' | 'production' = 'development'
  ): GeneratedConfig {
    try {
      switch (type) {
        case 'ragbits':
          return this.generateRagbitsConfig(workflowConfig, environment);
        case 'typescript':
          return this.generateTypeScriptConfig(workflowConfig, environment);
        case 'deployment':
          return this.generateDeploymentManifest(workflowConfig, environment);
        default:
          throw new Error(`Unknown configuration type: ${type}`);
      }
    } catch (error) {
      throw new Error(`Configuration generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
  
  /**
   * Generate RAGBits configuration
   * @param workflowConfig - BubbleLab workflow configuration
   * @param environment - Target environment
   * @returns Generated RAGBits configuration
   */
  private generateRagbitsConfig(
    workflowConfig: BubbleLabWorkflowConfig,
    environment: 'development' | 'staging' | 'production'
  ): GeneratedConfig {
    // Map to RAGBits configuration
    const ragbitsConfig = this.configMapper.mapBubbleLabToRagbits(workflowConfig);
    
    // Apply environment settings
    this.applyEnvironmentSettings(ragbitsConfig, environment);
    
    return {
      type: 'ragbits',
      content: ragbitsConfig,
      metadata: {
        generatedAt: new Date(),
        workflowId: workflowConfig.id,
        environment
      }
    };
  }
  
  /**
   * Generate TypeScript configuration
   * @param workflowConfig - BubbleLab workflow configuration
   * @param environment - Target environment
   * @returns Generated TypeScript configuration
   */
  private generateTypeScriptConfig(
    workflowConfig: BubbleLabWorkflowConfig,
    environment: 'development' | 'staging' | 'production'
  ): GeneratedConfig {
    // First generate RAGBits config
    const ragbitsConfig = this.configMapper.mapBubbleLabToRagbits(workflowConfig);
    this.applyEnvironmentSettings(ragbitsConfig, environment);
    
    // Convert to TypeScript code
    const typescriptCode = this.convertToTypeScript(ragbitsConfig);
    
    return {
      type: 'typescript',
      content: typescriptCode,
      metadata: {
        generatedAt: new Date(),
        workflowId: workflowConfig.id,
        environment
      }
    };
  }
  
  /**
   * Generate deployment manifest
   * @param workflowConfig - BubbleLab workflow configuration
   * @param environment - Target environment
   * @returns Generated deployment manifest
   */
  private generateDeploymentManifest(
    workflowConfig: BubbleLabWorkflowConfig,
    environment: 'development' | 'staging' | 'production'
  ): GeneratedConfig {
    // Generate RAGBits config first
    const ragbitsConfig = this.configMapper.mapBubbleLabToRagbits(workflowConfig);
    this.applyEnvironmentSettings(ragbitsConfig, environment);
    
    // Create deployment manifest
    const manifest = this.createKubernetesManifest(workflowConfig, ragbitsConfig, environment);
    
    return {
      type: 'deployment',
      content: manifest,
      metadata: {
        generatedAt: new Date(),
        workflowId: workflowConfig.id,
        environment
      }
    };
  }
  
  /**
   * Apply environment-specific settings
   * @param config - RAGBits configuration
   * @param environment - Target environment
   */
  private applyEnvironmentSettings(
    config: RagbitsConfig,
    environment: 'development' | 'staging' | 'production'
  ): void {
    // Apply global settings based on environment
    if (!config.globalConfig) {
      config.globalConfig = {};
    }
    
    switch (environment) {
      case 'development':
        config.globalConfig.logging = {
          level: 'debug',
          file: 'ragbits-development.log'
        };
        config.globalConfig.caching = {
          enabled: false
        };
        break;
      
      case 'staging':
        config.globalConfig.logging = {
          level: 'info',
          file: 'ragbits-staging.log'
        };
        config.globalConfig.caching = {
          enabled: true,
          ttl: 600000 // 10 minutes
        };
        break;
      
      case 'production':
        config.globalConfig.logging = {
          level: 'warn',
          file: 'ragbits-production.log'
        };
        config.globalConfig.caching = {
          enabled: true,
          ttl: 1800000 // 30 minutes
        };
        break;
    }
    
    // Apply node-specific environment settings
    config.nodes.forEach(node => {
      this.applyNodeEnvironmentSettings(node, environment);
    });
  }
  
  /**
   * Apply environment settings to individual node
   * @param node - RAGBits node configuration
   * @param environment - Target environment
   */
  private applyNodeEnvironmentSettings(
    node: any,
    environment: 'development' | 'staging' | 'production'
  ): void {
    switch (node.type) {
      case 'ingest':
        this.applyIngestEnvironmentSettings(node.config as RAGBitsIngestConfig, environment);
        break;
      case 'search':
        this.applySearchEnvironmentSettings(node.config as RAGBitsSearchConfig, environment);
        break;
      case 'generation':
        this.applyGenerationEnvironmentSettings(node.config as RAGBitsGenerationConfig, environment);
        break;
      case 'index':
        this.applyIndexEnvironmentSettings(node.config as RAGBitsIndexConfig, environment);
        break;
    }
  }
  
  /**
   * Apply environment settings to ingest configuration
   * @param config - Ingest configuration
   * @param environment - Target environment
   */
  private applyIngestEnvironmentSettings(
    config: RAGBitsIngestConfig,
    environment: 'development' | 'staging' | 'production'
  ): void {
    if (!config.processingOptions) {
      config.processingOptions = {};
    }
    
    switch (environment) {
      case 'development':
        config.processingOptions.chunkSize = 500;
        config.processingOptions.chunkOverlap = 100;
        break;
      case 'staging':
        config.processingOptions.chunkSize = 1000;
        config.processingOptions.chunkOverlap = 200;
        break;
      case 'production':
        config.processingOptions.chunkSize = 2000;
        config.processingOptions.chunkOverlap = 400;
        break;
    }
  }
  
  /**
   * Apply environment settings to search configuration
   * @param config - Search configuration
   * @param environment - Target environment
   */
  private applySearchEnvironmentSettings(
    config: RAGBitsSearchConfig,
    environment: 'development' | 'staging' | 'production'
  ): void {
    switch (environment) {
      case 'development':
        config.topK = 3;
        config.similarityThreshold = 0.6;
        break;
      case 'staging':
        config.topK = 5;
        config.similarityThreshold = 0.7;
        break;
      case 'production':
        config.topK = 10;
        config.similarityThreshold = 0.8;
        break;
    }
  }
  
  /**
   * Apply environment settings to generation configuration
   * @param config - Generation configuration
   * @param environment - Target environment
   */
  private applyGenerationEnvironmentSettings(
    config: RAGBitsGenerationConfig,
    environment: 'development' | 'staging' | 'production'
  ): void {
    if (!config.parameters) {
      config.parameters = {};
    }
    
    switch (environment) {
      case 'development':
        config.parameters.temperature = 0.9;
        config.parameters.maxTokens = 250;
        break;
      case 'staging':
        config.parameters.temperature = 0.7;
        config.parameters.maxTokens = 500;
        break;
      case 'production':
        config.parameters.temperature = 0.5;
        config.parameters.maxTokens = 1000;
        break;
    }
  }
  
  /**
   * Apply environment settings to index configuration
   * @param config - Index configuration
   * @param environment - Target environment
   */
  private applyIndexEnvironmentSettings(
    config: RAGBitsIndexConfig,
    environment: 'development' | 'staging' | 'production'
  ): void {
    if (!config.indexConfig) {
      config.indexConfig = {};
    }
    
    switch (environment) {
      case 'development':
        config.indexConfig.shardCount = 1;
        config.indexConfig.replicaCount = 0;
        break;
      case 'staging':
        config.indexConfig.shardCount = 2;
        config.indexConfig.replicaCount = 1;
        break;
      case 'production':
        config.indexConfig.shardCount = 4;
        config.indexConfig.replicaCount = 2;
        break;
    }
  }
  
  /**
   * Convert configuration to TypeScript code
   * @param config - RAGBits configuration
   * @returns TypeScript code string
   */
  private convertToTypeScript(config: RagbitsConfig): string {
    const importStatements = `import { RAGBitsDocumentProcessor } from 'ragbits';
import { RAGBitsWorkflowEngine } from 'ragbits-bubblelab-integration';
`;
    
    const configVariable = `const ragbitsConfig = ${JSON.stringify(config, null, 2)};`;
    
    const processorSetup = `
// Initialize the document processor
const processor = new RAGBitsDocumentProcessor({
  processorType: 'ragbits',
  processorConfig: {
    documentProcessor: {
      chunkSize: ${config.globalConfig?.processorConfig?.documentProcessor?.chunkSize || 1000},
      chunkOverlap: ${config.globalConfig?.processorConfig?.documentProcessor?.chunkOverlap || 200},
      embeddingModel: '${config.globalConfig?.processorConfig?.documentProcessor?.embeddingModel || 'text-embedding-ada-002'}',
      vectorStoreType: '${config.globalConfig?.processorConfig?.documentProcessor?.vectorStoreType || 'faiss'}'
    },
    searchConfig: {
      topK: ${config.globalConfig?.processorConfig?.searchConfig?.topK || 5},
      similarityThreshold: ${config.globalConfig?.processorConfig?.searchConfig?.similarityThreshold || 0.7},
      rerankModel: '${config.globalConfig?.processorConfig?.searchConfig?.rerankModel || 'default'}'
    },
    generationConfig: {
      model: '${config.globalConfig?.processorConfig?.generationConfig?.model || 'gpt-3.5-turbo'}',
      temperature: ${config.globalConfig?.processorConfig?.generationConfig?.temperature || 0.7},
      maxTokens: ${config.globalConfig?.processorConfig?.generationConfig?.maxTokens || 500}
    }
  }
});
`;
    
    const workflowSetup = `
// Initialize the workflow engine
const workflowEngine = new RAGBitsWorkflowEngine(ragbitsConfig, {}, processor);

// Execute the workflow
export async function executeWorkflow() {
  await workflowEngine.initialize();
  const result = await workflowEngine.executeWorkflow();
  return result;
}
`;
    
    return `${importStatements}

${configVariable}

${processorSetup}

${workflowSetup}`;
  }
  
  /**
   * Create Kubernetes deployment manifest
   * @param workflowConfig - BubbleLab workflow configuration
   * @param ragbitsConfig - RAGBits configuration
   * @param environment - Target environment
   * @returns Kubernetes manifest
   */
  private createKubernetesManifest(
    workflowConfig: BubbleLabWorkflowConfig,
    ragbitsConfig: RagbitsConfig,
    environment: 'development' | 'staging' | 'production'
  ): any {
    const sanitizedName = workflowConfig.name.toLowerCase().replace(/\s+/g, '-');
    const appName = `ragbits-${sanitizedName}`;
    
    return {
      apiVersion: 'apps/v1',
      kind: 'Deployment',
      metadata: {
        name: appName,
        labels: {
          app: appName,
          environment,
          workflow: workflowConfig.id
        }
      },
      spec: {
        replicas: this.getReplicaCount(environment),
        selector: {
          matchLabels: {
            app: appName
          }
        },
        template: {
          metadata: {
            labels: {
              app: appName
            }
          },
          spec: {
            containers: [
              {
                name: 'ragbits-workflow',
                image: this.getDockerImage(environment),
                ports: [
                  {
                    containerPort: 3000
                  }
                ],
                env: [
                  {
                    name: 'NODE_ENV',
                    value: environment
                  },
                  {
                    name: 'WORKFLOW_ID',
                    value: workflowConfig.id
                  },
                  {
                    name: 'WORKFLOW_NAME',
                    value: workflowConfig.name
                  }
                ],
                resources: this.getResourceLimits(environment),
                volumeMounts: [
                  {
                    name: 'config-volume',
                    mountPath: '/app/config'
                  }
                ]
              }
            ],
            volumes: [
              {
                name: 'config-volume',
                configMap: {
                  name: `${appName}-config`
                }
              }
            ]
          }
        }
      }
    };
  }
  
  /**
   * Get replica count based on environment
   * @param environment - Target environment
   * @returns Replica count
   */
  private getReplicaCount(environment: 'development' | 'staging' | 'production'): number {
    switch (environment) {
      case 'development': return 1;
      case 'staging': return 2;
      case 'production': return 4;
      default: return 1;
    }
  }
  
  /**
   * Get Docker image based on environment
   * @param environment - Target environment
   * @returns Docker image name
   */
  private getDockerImage(environment: 'development' | 'staging' | 'production'): string {
    switch (environment) {
      case 'development': return 'ragbits-workflow:dev';
      case 'staging': return 'ragbits-workflow:staging';
      case 'production': return 'ragbits-workflow:prod';
      default: return 'ragbits-workflow:latest';
    }
  }
  
  /**
   * Get resource limits based on environment
   * @param environment - Target environment
   * @returns Resource limits
   */
  private getResourceLimits(environment: 'development' | 'staging' | 'production'): any {
    switch (environment) {
      case 'development':
        return {
          requests: {
            cpu: '500m',
            memory: '512Mi'
          },
          limits: {
            cpu: '1000m',
            memory: '1024Mi'
          }
        };
      case 'staging':
        return {
          requests: {
            cpu: '1000m',
            memory: '1024Mi'
          },
          limits: {
            cpu: '2000m',
            memory: '2048Mi'
          }
        };
      case 'production':
        return {
          requests: {
            cpu: '2000m',
            memory: '2048Mi'
          },
          limits: {
            cpu: '4000m',
            memory: '4096Mi'
          }
        };
      default:
        return {
          requests: {
            cpu: '500m',
            memory: '512Mi'
          },
          limits: {
            cpu: '1000m',
            memory: '1024Mi'
          }
        };
    }
  }
  
  /**
   * Validate configuration before generation
   * @param workflowConfig - BubbleLab workflow configuration
   * @throws Error if configuration is invalid
   */
  public validateConfig(workflowConfig: BubbleLabWorkflowConfig): void {
    // Use the config mapper to validate
    try {
      this.configMapper.mapBubbleLabToRagbits(workflowConfig);
    } catch (error) {
      throw new Error(`Configuration validation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
  
  /**
   * Format configuration as JSON
   * @param config - Configuration to format
   * @returns Formatted JSON string
   */
  public formatConfig(config: any): string {
    return JSON.stringify(config, null, 2);
  }
  
  /**
   * Generate TypeScript configuration from JSON
   * @param jsonConfig - JSON configuration
   * @returns TypeScript code
   */
  public generateTypeScriptFromJson(jsonConfig: string): string {
    try {
      const config = JSON.parse(jsonConfig);
      return this.convertToTypeScript(config);
    } catch (error) {
      throw new Error(`Failed to parse JSON: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
}