/**
 * Ragbits Document Processor Integration for BubbleLab
 * 
 * This module provides integration between BubbleLab and the existing Ragbits document processor
 */

import { RAGBitsDocumentProcessor, RAGBitsProcessorConfig, DocumentProcessingResult } from '../../knowledge_engine/ragbits_document_processor';
import { ProcessedDocument, ProcessingStats, ProcessorIntegrationConfig } from '../types';

export class RagbitsProcessorIntegration {
  private processor: RAGBitsDocumentProcessor;
  private config: ProcessorIntegrationConfig;
  private processingQueue: Array<{
    id: string;
    source: string;
    content: string;
    metadata: Record<string, any>;
    resolve: (result: ProcessedDocument) => void;
    reject: (error: Error) => void;
  }>;
  private isProcessing: boolean;
  private stats: ProcessingStats;
  private cache: Map<string, { data: any; timestamp: number }>;

  constructor(config?: ProcessorIntegrationConfig) {
    this.config = {
      enableAutoIndexing: true,
      autoIndexInterval: 300, // 5 minutes
      batchSize: 10,
      enableCaching: true,
      cacheTTL: 3600, // 1 hour
      enableMonitoring: true,
      maxConcurrentProcesses: 5,
      ...config
    };

    // Create processor with default config
    this.processor = new RAGBitsDocumentProcessor();
    this.processingQueue = [];
    this.isProcessing = false;
    this.stats = {
      totalProcessed: 0,
      successful: 0,
      failed: 0,
      averageProcessingTime: 0,
      lastProcessed: new Date(0),
      queueSize: 0
    };
    this.cache = new Map();
  }

  /**
   * Initializes the processor and starts auto-indexing if enabled
   */
  async initialize(): Promise<void> {
    try {
      // Initialize the Ragbits processor
      const initialized = await this.processor.initialize();
      
      if (!initialized) {
        throw new Error('Failed to initialize Ragbits document processor');
      }

      console.log('RagbitsProcessorIntegration initialized successfully');

      // Start auto-indexing if enabled
      if (this.config.enableAutoIndexing && this.config.autoIndexInterval) {
        setInterval(() => {
          this.processQueue();
        }, this.config.autoIndexInterval * 1000);
      }
    } catch (error) {
      console.error('Failed to initialize RagbitsProcessorIntegration:', error);
      throw error;
    }
  }

  /**
   * Adds a document to the processing queue
   */
  async addDocument(
    source: string,
    content: string,
    metadata: Record<string, any> = {}
  ): Promise<ProcessedDocument> {
    return new Promise((resolve, reject) => {
      const id = this.generateId();
      
      this.processingQueue.push({
        id,
        source,
        content,
        metadata,
        resolve,
        reject
      });
      
      this.stats.queueSize = this.processingQueue.length;
      
      // Process immediately if auto-indexing is disabled
      if (!this.config.enableAutoIndexing) {
        this.processQueue();
      }
    });
  }

  /**
   * Processes a single document directly
   */
  async processDocument(
    source: string,
    content: string,
    metadata: Record<string, any> = {}
  ): Promise<ProcessedDocument> {
    const startTime = Date.now();
    
    try {
      // Check cache first if enabled
      const cacheKey = this.getCacheKey('process', { source, metadata });
      if (this.config.enableCaching) {
        const cached = this.getCachedResult(cacheKey);
        if (cached) {
          return cached;
        }
      }

      // Process the document
      const result = await this.processor.ingest_text(content, metadata, source);

      const processingTime = Date.now() - startTime;
      
      const processedDoc: ProcessedDocument = {
        id: result.document_id,
        source,
        content: content.substring(0, 100) + (content.length > 100 ? '...' : ''), // Truncate for performance
        metadata,
        processingTime,
        success: result.success,
        error: result.error
      };

      // Update stats
      this.updateStats(processedDoc, processingTime);

      // Cache result if enabled
      if (this.config.enableCaching) {
        this.setCachedResult(cacheKey, processedDoc);
      }

      return processedDoc;
    } catch (error) {
      const processingTime = Date.now() - startTime;
      
      const processedDoc: ProcessedDocument = {
        id: this.generateId(),
        source,
        content: content.substring(0, 100) + (content.length > 100 ? '...' : ''),
        metadata,
        processingTime,
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };

      // Update stats
      this.updateStats(processedDoc, processingTime);

      throw error;
    }
  }

  /**
   * Processes documents in the queue in batches
   */
  async processQueue(): Promise<void> {
    if (this.isProcessing || this.processingQueue.length === 0) {
      return;
    }

    this.isProcessing = true;

    try {
      // Process in batches
      const batch = this.processingQueue.splice(0, this.config.batchSize || 10);
      this.stats.queueSize = this.processingQueue.length;

      // Process each document in the batch
      for (const item of batch) {
        try {
          const result = await this.processDocument(item.source, item.content, item.metadata);
          item.resolve(result);
        } catch (error) {
          item.reject(error);
        }
      }
    } finally {
      this.isProcessing = false;
    }
  }

  /**
   * Searches for documents using the Ragbits processor
   */
  async search(
    query: string,
    topK: number = 5,
    filters?: Record<string, any>,
    minScore: number = 0.0
  ): Promise<Array<{
    documentId: string;
    content: string;
    metadata: Record<string, any>;
    score: number;
  }>> {
    // Check cache first if enabled
    const cacheKey = this.getCacheKey('search', { query, topK, filters, minScore });
    if (this.config.enableCaching) {
      const cached = this.getCachedResult(cacheKey);
      if (cached) {
        return cached;
      }
    }

    try {
      const results = await this.processor.search(query, topK, filters, minScore);

      // Transform to expected format
      const transformedResults = results.map(result => ({
        documentId: result.metadata.document_id || 'unknown',
        content: result.content,
        metadata: result.metadata,
        score: result.score
      }));

      // Cache result if enabled
      if (this.config.enableCaching) {
        this.setCachedResult(cacheKey, transformedResults);
      }

      return transformedResults;
    } catch (error) {
      console.error('Search failed:', error);
      throw error;
    }
  }

  /**
   * Gets statistics about document processing
   */
  getStats(): ProcessingStats {
    return { ...this.stats };
  }

  /**
   * Gets the number of documents in the processing queue
   */
  getQueueSize(): number {
    return this.processingQueue.length;
  }

  /**
   * Clears the processing queue
   */
  clearQueue(): void {
    // Reject all pending promises
    for (const item of this.processingQueue) {
      item.reject(new Error('Queue cleared'));
    }
    this.processingQueue = [];
    this.stats.queueSize = 0;
  }

  /**
   * Clears the vector store
   */
  async clearStore(): Promise<boolean> {
    try {
      const result = await this.processor.clear();
      return result;
    } catch (error) {
      console.error('Failed to clear store:', error);
      return false;
    }
  }

  /**
   * Gets index statistics from the processor
   */
  async getIndexStats(): Promise<any> {
    try {
      return await this.processor.get_statistics();
    } catch (error) {
      console.error('Failed to get index stats:', error);
      throw error;
    }
  }

  /**
   * Updates processing statistics
   */
  private updateStats(result: ProcessedDocument, processingTime: number): void {
    this.stats.totalProcessed++;
    
    if (result.success) {
      this.stats.successful++;
    } else {
      this.stats.failed++;
    }

    // Update average processing time
    this.stats.averageProcessingTime = 
      ((this.stats.averageProcessingTime * (this.stats.totalProcessed - 1)) + processingTime) / 
      this.stats.totalProcessed;

    this.stats.lastProcessed = new Date();
  }

  /**
   * Generates a unique ID
   */
  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Gets a cache key for the given parameters
   */
  private getCacheKey(operation: string, params: any): string {
    return `${operation}:${JSON.stringify(params)}`;
  }

  /**
   * Gets a cached result if it exists and hasn't expired
   */
  private getCachedResult<T>(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) {
      return null;
    }

    const age = (Date.now() - entry.timestamp) / 1000;
    if (age > (this.config.cacheTTL || 3600)) {
      this.cache.delete(key);
      return null;
    }

    return entry.data as T;
  }

  /**
   * Sets a cached result
   */
  private setCachedResult<T>(key: string, data: T): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now()
    });
  }

  /**
   * Disposes of resources
   */
  async dispose(): Promise<void> {
    this.clearQueue();
    if (this.processor && typeof this.processor.close === 'function') {
      await this.processor.close();
    }
    this.cache.clear();
  }
}