/**
 * RAGBitsIngestBubble - Document Ingestion Bubble for BubbleLab
 *
 * This bubble handles document ingestion into the RAGBits system
 */

import { BaseBubble } from './BaseBubble';
import { RAGBitsIngestConfig, RAGBitsIngestInput, RAGBitsIngestOutput } from '../types';
import { RAGBitsDocumentProcessor } from '../../knowledge_engine/ragbits_document_processor';
import { Logger, generateId } from '../utils/common.utils';

export class RAGBitsIngestBubble extends BaseBubble<
  RAGBitsIngestConfig,
  RAGBitsIngestInput,
  RAGBitsIngestOutput
> {
  private processor: RAGBitsDocumentProcessor;

  constructor(config: RAGBitsIngestConfig) {
    super(config);
    this.processor = new RAGBitsDocumentProcessor({
      chunk_size: config.chunkSize || 1000,
      chunk_overlap: config.chunkOverlap || 200
    });
  }

  async initialize(): Promise<void> {
    await this.processor.initialize();
    this.logger.info('RAGBitsIngestBubble initialized');
  }

  async action(input: RAGBitsIngestInput): Promise<RAGBitsIngestOutput> {
    const startTime = Date.now();
    const operationId = generateId('ingest-operation');

    this.logger.debug(`Starting ingestion operation ${operationId}`);

    try {
      let result;

      // Determine ingestion method based on input and config
      if (input.content) {
        // Ingest text content directly
        this.logger.debug(`Ingesting text content for operation ${operationId}`);
        result = await this.processor.ingest_text(
          input.content,
          { ...this.config.metadata, ...input.metadata, operationId },
          input.source || this.config.sourcePath || 'inline-content'
        );
      } else if (input.source) {
        // Ingest from file source
        this.logger.debug(`Ingesting from file source: ${input.source} for operation ${operationId}`);
        result = await this.processor.ingest_file(
          input.source,
          { ...this.config.metadata, ...input.metadata, operationId }
        );
      } else if (this.config.sourcePath) {
        // Use configured source path
        this.logger.debug(`Ingesting from configured source: ${this.config.sourcePath} for operation ${operationId}`);
        result = await this.processor.ingest_file(
          this.config.sourcePath,
          { ...this.config.metadata, operationId }
        );
      } else {
        throw new Error('No source provided for ingestion - either input.content, input.source, or config.sourcePath must be specified');
      }

      const processingTime = Date.now() - startTime;

      this.logger.info(`Ingestion operation ${operationId} completed successfully in ${processingTime}ms`);

      return {
        success: result.success,
        documentId: result.document_id || generateId('doc'),
        chunksIngested: result.chunks_ingested || 0,
        processingTime,
        error: result.error
      };
    } catch (error) {
      const processingTime = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      this.logger.error(`Ingestion operation ${operationId} failed after ${processingTime}ms: ${errorMessage}`);

      return {
        success: false,
        documentId: '',
        chunksIngested: 0,
        processingTime,
        error: errorMessage
      };
    }
  }

  /**
   * Add file ingestion functionality
   */
  async ingestFile(filePath: string, metadata?: Record<string, any>): Promise<RAGBitsIngestOutput> {
    return this.action({ source: filePath, metadata });
  }

  /**
   * Add URL ingestion functionality
   */
  async ingestUrl(url: string, metadata?: Record<string, any>): Promise<RAGBitsIngestOutput> {
    // In a real implementation, this would handle URL ingestion
    // For now, we'll treat it similarly to file ingestion
    return this.action({ source: url, metadata });
  }

  /**
   * Add text ingestion functionality
   */
  async ingestText(content: string, metadata?: Record<string, any>): Promise<RAGBitsIngestOutput> {
    return this.action({ content, metadata });
  }

  /**
   * Add metadata handling functionality
   */
  addMetadata(metadata: Record<string, any>): void {
    this.config.metadata = { ...this.config.metadata, ...metadata };
  }

  /**
   * Override error handling for ingestion failures
   */
  override handleError(error: unknown, context?: string): Error {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    const fullContext = context ? ` in ${context}` : '';
    this.logger.error(`Ingestion error${fullContext}: ${errorMessage}`);
    return new Error(errorMessage);
  }

  /**
   * Add logging for ingestion process
   */
  private logIngestionProgress(progress: number, operationId: string): void {
    this.logger.debug(`Ingestion operation ${operationId} progress: ${progress}%`);
  }

  /**
   * Add metrics collection for ingestion
   */
  private collectMetrics(result: RAGBitsIngestOutput, processingTime: number): void {
    // In a real implementation, this would collect metrics
    // For example, sending to a metrics service
    this.logger.debug(`Collected metrics for ingestion: ${result.chunksIngested} chunks in ${processingTime}ms`);
  }
}