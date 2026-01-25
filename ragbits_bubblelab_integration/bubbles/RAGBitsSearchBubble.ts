/**
 * RAGBitsSearchBubble - Semantic Search Bubble for BubbleLab
 *
 * This bubble performs semantic search using the RAGBits system
 */

import { BaseBubble } from './BaseBubble';
import { RAGBitsSearchConfig, RAGBitsSearchInput, RAGBitsSearchOutput } from '../types';
import { RAGBitsDocumentProcessor } from '../../knowledge_engine/ragbits_document_processor';

export class RAGBitsSearchBubble extends BaseBubble<
  RAGBitsSearchConfig,
  RAGBitsSearchInput,
  RAGBitsSearchOutput
> {
  private processor: RAGBitsDocumentProcessor;

  constructor(config: RAGBitsSearchConfig) {
    super(config);
    this.processor = new RAGBitsDocumentProcessor();
  }

  async initialize(): Promise<void> {
    await this.processor.initialize();
    console.log('RAGBitsSearchBubble initialized');
  }

  async action(input: RAGBitsSearchInput): Promise<RAGBitsSearchOutput> {
    try {
      const startTime = Date.now();

      // Use input values or fall back to config defaults
      const topK = input.topK || this.config.topK || 5;
      const scoreThreshold = input.scoreThreshold || this.config.scoreThreshold || 0.0;
      const filters = { ...this.config.defaultFilters, ...input.filters };

      // Perform search
      const results = await this.processor.search(
        input.query,
        topK,
        filters,
        scoreThreshold
      );

      const executionTime = Date.now() - startTime;

      return {
        success: true,
        results: results.map(result => ({
          documentId: result.metadata.document_id || 'unknown',
          content: result.content,
          score: result.score,
          metadata: result.metadata
        })),
        totalResults: results.length,
        executionTime,
        error: undefined
      };
    } catch (error) {
      console.error('RAGBitsSearchBubble error:', error);
      return {
        success: false,
        results: [],
        totalResults: 0,
        executionTime: 0,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }
}