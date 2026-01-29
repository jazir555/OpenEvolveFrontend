/**
 * RAGBitsIndexBubble - Index Management Bubble for BubbleLab
 *
 * This bubble manages the RAGBits vector index
 */

import { BaseBubble } from './BaseBubble';
import { RAGBitsIndexConfig, RAGBitsIndexInput, RAGBitsIndexOutput } from '../types';
import { RAGBitsDocumentProcessor } from '../../knowledge_engine/ragbits_document_processor';
import { Logger, generateId } from '../utils/common.utils';

export class RAGBitsIndexBubble extends BaseBubble<
  RAGBitsIndexConfig,
  RAGBitsIndexInput,
  RAGBitsIndexOutput
> {
  private processor: RAGBitsDocumentProcessor;
  private autoRefreshIntervalId: NodeJS.Timeout | null = null;

  constructor(config: RAGBitsIndexConfig) {
    super(config);
    this.processor = new RAGBitsDocumentProcessor({
      vector_store_type: config.vectorStoreType || 'memory',
      embedding_model: config.embeddingModel || 'text-embedding-3-small'
    });
  }

  async initialize(): Promise<void> {
    await this.processor.initialize();
    this.logger.info('RAGBitsIndexBubble initialized');

    // Set up auto-refresh if enabled
    if (this.config.autoRefresh && this.config.refreshInterval) {
      this.logger.info(`Setting up auto-refresh every ${this.config.refreshInterval} seconds`);
      this.autoRefreshIntervalId = setInterval(async () => {
        try {
          const result = await this.action({ operation: 'refresh' });
          this.logger.debug(`Auto-refresh completed: ${result.success ? 'SUCCESS' : 'FAILED'}`);
        } catch (error) {
          this.logger.error(`Auto-refresh failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
      }, this.config.refreshInterval! * 1000);
    }
  }

  async action(input: RAGBitsIndexInput): Promise<RAGBitsIndexOutput> {
    const startTime = Date.now();
    const operationId = generateId('index-operation');

    this.logger.debug(`Starting index operation ${operationId}: ${input.operation}`);

    try {
      let result;
      switch (input.operation) {
        case 'stats':
          this.logger.debug(`Executing stats operation for ${operationId}`);
          result = await this.processor.get_statistics();
          break;
        case 'refresh':
          this.logger.debug(`Executing refresh operation for ${operationId}`);
          // For now, just return stats as refresh is handled internally by Ragbits
          result = await this.processor.get_statistics();
          break;
        case 'clear':
          this.logger.debug(`Executing clear operation for ${operationId}`);
          result = await this.processor.clear();
          break;
        case 'optimize':
          this.logger.debug(`Executing optimize operation for ${operationId}`);
          // Ragbits handles optimization internally, return stats
          result = await this.processor.get_statistics();
          break;
        default:
          throw new Error(`Unsupported operation: ${input.operation}`);
      }

      const executionTime = Date.now() - startTime;

      this.logger.info(`Index operation ${operationId} (${input.operation}) completed in ${executionTime}ms`);

      return {
        success: true,
        operation: input.operation,
        result,
        executionTime,
        error: undefined
      };
    } catch (error) {
      const executionTime = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      this.logger.error(`Index operation ${operationId} (${input.operation}) failed after ${executionTime}ms: ${errorMessage}`);

      return {
        success: false,
        operation: input.operation,
        result: null,
        executionTime,
        error: errorMessage
      };
    }
  }

  /**
   * Add stats operation logic
   */
  async getStats(): Promise<RAGBitsIndexOutput> {
    return this.action({ operation: 'stats' });
  }

  /**
   * Add refresh operation logic
   */
  async refreshIndex(): Promise<RAGBitsIndexOutput> {
    return this.action({ operation: 'refresh' });
  }

  /**
   * Add clear operation logic
   */
  async clearIndex(): Promise<RAGBitsIndexOutput> {
    return this.action({ operation: 'clear' });
  }

  /**
   * Add optimize operation logic
   */
  async optimizeIndex(): Promise<RAGBitsIndexOutput> {
    return this.action({ operation: 'optimize' });
  }

  /**
   * Override error handling for index operations
   */
  override handleError(error: unknown, context?: string): Error {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    const fullContext = context ? ` in ${context}` : '';
    this.logger.error(`Index error${fullContext}: ${errorMessage}`);
    return new Error(errorMessage);
  }

  /**
   * Add logging for index operations
   */
  private logIndexOperation(operation: string, operationId: string, status: 'start' | 'complete' | 'error'): void {
    this.logger.debug(`Index operation ${operationId} (${operation}) ${status}`);
  }

  /**
   * Add metrics collection for index operations
   */
  private collectMetrics(result: RAGBitsIndexOutput, executionTime: number): void {
    // In a real implementation, this would collect metrics
    this.logger.debug(`Collected metrics for index operation: ${result.operation} in ${executionTime}ms`);
  }

  /**
   * Perform a custom index operation
   */
  async performOperation(operation: 'stats' | 'refresh' | 'clear' | 'optimize', force?: boolean): Promise<RAGBitsIndexOutput> {
    return this.action({ operation, force });
  }

  /**
   * Get index status information
   */
  async getStatus(): Promise<RAGBitsIndexOutput> {
    return this.action({ operation: 'stats' });
  }

  /**
   * Rebuild the entire index
   */
  async rebuildIndex(): Promise<RAGBitsIndexOutput> {
    // First clear the index
    const clearResult = await this.action({ operation: 'clear' });
    if (!clearResult.success) {
      return clearResult;
    }

    // Then potentially refresh/rebuild
    return this.action({ operation: 'refresh' });
  }

  /**
   * Clean up resources
   */
  async dispose(): Promise<void> {
    if (this.autoRefreshIntervalId) {
      clearInterval(this.autoRefreshIntervalId);
      this.logger.info('Auto-refresh interval cleared');
    }
    await super.dispose();
  }
}