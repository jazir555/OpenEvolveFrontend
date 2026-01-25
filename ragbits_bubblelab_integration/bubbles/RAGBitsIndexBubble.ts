/**
 * RAGBitsIndexBubble - Index Management Bubble for BubbleLab
 *
 * This bubble manages the RAGBits vector index
 */

import { BaseBubble } from './BaseBubble';
import { RAGBitsIndexConfig, RAGBitsIndexInput, RAGBitsIndexOutput } from '../types';
import { RAGBitsDocumentProcessor } from '../../knowledge_engine/ragbits_document_processor';

export class RAGBitsIndexBubble extends BaseBubble<
  RAGBitsIndexConfig,
  RAGBitsIndexInput,
  RAGBitsIndexOutput
> {
  private processor: RAGBitsDocumentProcessor;

  constructor(config: RAGBitsIndexConfig) {
    super(config);
    this.processor = new RAGBitsDocumentProcessor({
      vector_store_type: config.vectorStoreType || 'memory',
      embedding_model: config.embeddingModel || 'text-embedding-3-small'
    });
  }

  async initialize(): Promise<void> {
    await this.processor.initialize();
    console.log('RAGBitsIndexBubble initialized');

    // Set up auto-refresh if enabled
    if (this.config.autoRefresh && this.config.refreshInterval) {
      setInterval(async () => {
        try {
          await this.action({ operation: 'refresh' });
        } catch (error) {
          console.error('Auto-refresh failed:', error);
        }
      }, this.config.refreshInterval * 1000);
    }
  }

  async action(input: RAGBitsIndexInput): Promise<RAGBitsIndexOutput> {
    try {
      const startTime = Date.now();

      let result;
      switch (input.operation) {
        case 'stats':
          result = await this.processor.get_statistics();
          break;
        case 'refresh':
          // For now, just return stats as refresh is handled internally by Ragbits
          result = await this.processor.get_statistics();
          break;
        case 'clear':
          result = await this.processor.clear();
          break;
        case 'optimize':
          // Ragbits handles optimization internally, return stats
          result = await this.processor.get_statistics();
          break;
        default:
          throw new Error(`Unsupported operation: ${input.operation}`);
      }

      const executionTime = Date.now() - startTime;

      return {
        success: true,
        operation: input.operation,
        result,
        executionTime,
        error: undefined
      };
    } catch (error) {
      console.error('RAGBitsIndexBubble error:', error);
      return {
        success: false,
        operation: input.operation,
        result: null,
        executionTime: 0,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }
}