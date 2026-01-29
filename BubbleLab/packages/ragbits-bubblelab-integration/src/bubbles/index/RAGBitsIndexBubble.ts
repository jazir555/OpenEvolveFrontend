/**
 * RAGBits Index Bubble
 * Bubble component for index management operations
 */

import { BaseBubble } from '../BaseBubble';
import { RAGBitsIndexConfig, RAGBitsIndexInput, RAGBitsIndexOutput, BubbleLabNode } from '../../types';
import { RAGBitsDocumentProcessor } from '../../integration/RagbitsProcessorIntegration';

/**
 * RAGBitsIndexBubble - Bubble for index management
 * Handles index statistics, refresh, clear, and optimize operations
 */
export class RAGBitsIndexBubble extends BaseBubble<RAGBitsIndexConfig> {
  private processor: RAGBitsDocumentProcessor | null = null;
  
  /**
   * Constructor
   * @param config - Index configuration
   */
  constructor(config: RAGBitsIndexConfig) {
    super(config);
  }
  
  /**
   * Initialize the bubble with processor
   * @param node - BubbleLab node
   * @param processor - RAGBits document processor
   */
  public async initialize(node: BubbleLabNode, processor?: RAGBitsDocumentProcessor): Promise<void> {
    await super.initialize(node);
    
    if (processor) {
      this.processor = processor;
      this.log('info', 'Document processor initialized');
    } else {
      this.log('warn', 'No document processor provided - using mock mode');
    }
  }
  
  /**
   * Validate index configuration
   * @param config - Configuration to validate
   * @returns Validated configuration
   */
  protected validateConfig(config: RAGBitsIndexConfig): RAGBitsIndexConfig {
    const baseConfig = super.validateConfig(config);
    
    // Validate operations
    if (!baseConfig.operations || baseConfig.operations.length === 0) {
      throw new Error('operations must be a non-empty array');
    }
    
    // Validate operation types
    const validOperations = ['stats', 'refresh', 'clear', 'optimize'];
    for (const op of baseConfig.operations) {
      if (!validOperations.includes(op)) {
        throw new Error(`Invalid operation: ${op}`);
      }
    }
    
    // Validate index configuration
    if (baseConfig.indexConfig) {
      if (baseConfig.indexConfig.shardCount !== undefined && baseConfig.indexConfig.shardCount <= 0) {
        throw new Error('shardCount must be a positive number');
      }
      
      if (baseConfig.indexConfig.replicaCount !== undefined && baseConfig.indexConfig.replicaCount < 0) {
        throw new Error('replicaCount must be a non-negative number');
      }
    }
    
    return baseConfig;
  }
  
  /**
   * Execute index operations
   * @param input - Index input data
   * @returns Promise with index output
   */
  public async action(input: RAGBitsIndexInput): Promise<RAGBitsIndexOutput> {
    if (!this.initialized) {
      throw new Error('Bubble not initialized');
    }
    
    // Merge input with config
    const mergedInput: RAGBitsIndexInput = {
      operation: input.operation || this.config.operations[0],
      params: input.params || this.config.indexConfig
    };
    
    return this.measureExecutionTime(async () => {
      try {
        this.log('info', `Executing index operation: ${mergedInput.operation}`);
        
        if (!this.processor) {
          // Mock mode - return simulated results
          return this.mockIndexOperation(mergedInput);
        }
        
        // Real processor mode
        switch (mergedInput.operation) {
          case 'stats':
            return this.processor.getIndexStats();
          case 'refresh':
            return this.processor.refreshIndex();
          case 'clear':
            return this.processor.clearStore();
          case 'optimize':
            return this.processor.optimizeIndex();
          default:
            throw new Error(`Unsupported operation: ${mergedInput.operation}`);
        }
      } catch (error) {
        this.handleError(error as Error, 'index operation');
        throw error; // This line will never be reached due to handleError throwing
      }
    }).then(({ result, time }) => {
      this.updateMetrics(true, time);
      this.log('info', `Index operation completed in ${time}ms`);
      return result;
    }).catch(error => {
      this.updateMetrics(false, 0);
      throw error;
    });
  }
  
  /**
   * Mock index operation for testing without processor
   * @param input - Index input
   * @returns Simulated index output
   */
  private mockIndexOperation(input: RAGBitsIndexInput): RAGBitsIndexOutput {
    this.log('warn', 'Using mock index operation - no real document processor available');
    
    const operation = input.operation || 'stats';
    
    switch (operation) {
      case 'stats':
        return {
          success: true,
          metadata: {
            operation: 'stats',
            timestamp: new Date(),
            duration: 20
          },
          data: {
            stats: {
              documentCount: 1000,
              indexSize: 50 * 1024 * 1024, // 50 MB
              lastUpdated: new Date(Date.now() - 86400000), // Yesterday
              health: 'green'
            }
          }
        };
      
      case 'refresh':
        return {
          success: true,
          metadata: {
            operation: 'refresh',
            timestamp: new Date(),
            duration: 50
          },
          data: {
            message: 'Index refreshed successfully',
            affectedDocuments: 1000
          }
        };
      
      case 'clear':
        return {
          success: true,
          metadata: {
            operation: 'clear',
            timestamp: new Date(),
            duration: 100
          },
          data: {
            message: 'Index cleared successfully',
            affectedDocuments: 1000
          }
        };
      
      case 'optimize':
        return {
          success: true,
          metadata: {
            operation: 'optimize',
            timestamp: new Date(),
            duration: 200
          },
          data: {
            message: 'Index optimized successfully',
            affectedDocuments: 1000
          }
        };
      
      default:
        throw new Error(`Unsupported operation: ${operation}`);
    }
  }
  
  /**
   * Execute stats operation
   * @returns Promise with stats output
   */
  private async executeStatsOperation(): Promise<RAGBitsIndexOutput> {
    this.log('info', 'Executing index stats operation');
    
    if (!this.processor) {
      return this.mockIndexOperation({ operation: 'stats' });
    }
    
    return this.processor.getIndexStats();
  }
  
  /**
   * Execute refresh operation
   * @returns Promise with refresh output
   */
  private async executeRefreshOperation(): Promise<RAGBitsIndexOutput> {
    this.log('info', 'Executing index refresh operation');
    
    if (!this.processor) {
      return this.mockIndexOperation({ operation: 'refresh' });
    }
    
    return this.processor.refreshIndex();
  }
  
  /**
   * Execute clear operation
   * @returns Promise with clear output
   */
  private async executeClearOperation(): Promise<RAGBitsIndexOutput> {
    this.log('info', 'Executing index clear operation');
    
    if (!this.processor) {
      return this.mockIndexOperation({ operation: 'clear' });
    }
    
    return this.processor.clearStore();
  }
  
  /**
   * Execute optimize operation
   * @returns Promise with optimize output
   */
  private async executeOptimizeOperation(): Promise<RAGBitsIndexOutput> {
    this.log('info', 'Executing index optimize operation');
    
    if (!this.processor) {
      return this.mockIndexOperation({ operation: 'optimize' });
    }
    
    return this.processor.optimizeIndex();
  }
  
  /**
   * Get processor status
   * @returns True if processor is available, false otherwise
   */
  public hasProcessor(): boolean {
    return !!this.processor;
  }
  
  /**
   * Set processor instance
   * @param processor - RAGBits document processor
   */
  public setProcessor(processor: RAGBitsDocumentProcessor): void {
    this.processor = processor;
    this.log('info', 'Document processor set');
  }
  
  /**
   * Get current processor
   * @returns Current processor or null
   */
  public getProcessor(): RAGBitsDocumentProcessor | null {
    return this.processor;
  }
}