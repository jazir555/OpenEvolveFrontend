/**
 * RAGBitsSearchBubble - Semantic Search Bubble for BubbleLab
 *
 * This bubble performs semantic search using the RAGBits system
 */

import { BaseBubble } from './BaseBubble';
import { RAGBitsSearchConfig, RAGBitsSearchInput, RAGBitsSearchOutput } from '../types';
import { RAGBitsDocumentProcessor } from '../../knowledge_engine/ragbits_document_processor';
import { Logger, generateId } from '../utils/common.utils';

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
    this.logger.info('RAGBitsSearchBubble initialized');
  }

  async action(input: RAGBitsSearchInput): Promise<RAGBitsSearchOutput> {
    const startTime = Date.now();
    const operationId = generateId('search-operation');

    this.logger.debug(`Starting search operation ${operationId} for query: "${input.query.substring(0, 50)}..."`);

    try {
      // Use input values or fall back to config defaults
      const topK = input.topK ?? this.config.topK ?? 5;
      const scoreThreshold = input.scoreThreshold ?? this.config.scoreThreshold ?? 0.0;
      const filters = { ...this.config.defaultFilters, ...input.filters };

      this.logger.debug(`Search parameters - topK: ${topK}, scoreThreshold: ${scoreThreshold}, filters: ${JSON.stringify(filters)}`);

      // Perform search
      const results = await this.processor.search(
        input.query,
        topK,
        filters,
        scoreThreshold
      );

      const executionTime = Date.now() - startTime;

      this.logger.info(`Search operation ${operationId} completed with ${results.length} results in ${executionTime}ms`);

      return {
        success: true,
        results: results.map(result => ({
          documentId: result.metadata.document_id || generateId('doc'),
          content: result.content,
          score: result.score,
          metadata: result.metadata
        })),
        totalResults: results.length,
        executionTime,
        error: undefined
      };
    } catch (error) {
      const executionTime = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      this.logger.error(`Search operation ${operationId} failed after ${executionTime}ms: ${errorMessage}`);

      return {
        success: false,
        results: [],
        totalResults: 0,
        executionTime,
        error: errorMessage
      };
    }
  }

  /**
   * Add query processing logic
   */
  private processQuery(query: string): string {
    // In a real implementation, this would normalize and preprocess the query
    return query.trim();
  }

  /**
   * Add filter application logic
   */
  private applyFilters(results: RAGBitsSearchOutput['results'], filters: Record<string, any>): RAGBitsSearchOutput['results'] {
    if (!filters || Object.keys(filters).length === 0) {
      return results;
    }

    return results.filter(result => {
      return Object.entries(filters).every(([key, value]) => {
        if (typeof value === 'string') {
          return result.metadata[key]?.toString().toLowerCase().includes(value.toLowerCase());
        }
        return result.metadata[key] === value;
      });
    });
  }

  /**
   * Add result formatting logic
   */
  private formatResults(results: RAGBitsSearchOutput['results']): RAGBitsSearchOutput['results'] {
    // In a real implementation, this would format results according to specific requirements
    return results;
  }

  /**
   * Override error handling for search failures
   */
  override handleError(error: unknown, context?: string): Error {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    const fullContext = context ? ` in ${context}` : '';
    this.logger.error(`Search error${fullContext}: ${errorMessage}`);
    return new Error(errorMessage);
  }

  /**
   * Add logging for search process
   */
  private logSearchProgress(progress: number, operationId: string): void {
    this.logger.debug(`Search operation ${operationId} progress: ${progress}%`);
  }

  /**
   * Add metrics collection for search
   */
  private collectMetrics(result: RAGBitsSearchOutput, executionTime: number): void {
    // In a real implementation, this would collect metrics
    this.logger.debug(`Collected metrics for search: ${result.totalResults} results in ${executionTime}ms`);
  }

  /**
   * Perform a search with additional options
   */
  async searchWithOptions(query: string, options: {
    topK?: number;
    scoreThreshold?: number;
    filters?: Record<string, any>;
  }): Promise<RAGBitsSearchOutput> {
    return this.action({
      query,
      topK: options.topK,
      scoreThreshold: options.scoreThreshold,
      filters: options.filters
    });
  }

  /**
   * Search with specific document IDs
   */
  async searchInDocuments(query: string, documentIds: string[], options?: {
    topK?: number;
    scoreThreshold?: number;
  }): Promise<RAGBitsSearchOutput> {
    // In a real implementation, this would restrict search to specific documents
    const filters = { document_ids: documentIds };
    return this.action({
      query,
      topK: options?.topK,
      scoreThreshold: options?.scoreThreshold,
      filters
    });
  }
}