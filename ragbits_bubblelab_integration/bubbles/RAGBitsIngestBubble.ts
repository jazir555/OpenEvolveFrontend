/**
 * RAGBitsIngestBubble - Document Ingestion Bubble for BubbleLab
 *
 * This bubble handles document ingestion into the RAGBits system
 */

import { BaseBubble } from './BaseBubble';
import { RAGBitsIngestConfig, RAGBitsIngestInput, RAGBitsIngestOutput } from '../types';
import { RAGBitsDocumentProcessor } from '../../knowledge_engine/ragbits_document_processor';

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
    console.log('RAGBitsIngestBubble initialized');
  }

  async action(input: RAGBitsIngestInput): Promise<RAGBitsIngestOutput> {
    try {
      const startTime = Date.now();

      let result;
      if (input.content) {
        // Ingest text content
        result = await this.processor.ingest_text(
          input.content,
          input.metadata || this.config.metadata,
          input.source || this.config.sourcePath
        );
      } else if (input.source) {
        // Ingest file
        result = await this.processor.ingest_file(
          input.source,
          input.metadata || this.config.metadata
        );
      } else {
        // Use configured source
        if (this.config.sourceType === 'file') {
          result = await this.processor.ingest_file(
            this.config.sourcePath,
            this.config.metadata
          );
        } else if (this.config.sourceType === 'text') {
          // For text type, we need content in the input
          throw new Error('Text content required for text source type');
        } else {
          throw new Error(`Unsupported source type: ${this.config.sourceType}`);
        }
      }

      const processingTime = Date.now() - startTime;

      return {
        success: result.success,
        documentId: result.document_id,
        chunksIngested: result.chunks_ingested,
        processingTime,
        error: result.error
      };
    } catch (error) {
      console.error('RAGBitsIngestBubble error:', error);
      return {
        success: false,
        documentId: '',
        chunksIngested: 0,
        processingTime: 0,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }
}