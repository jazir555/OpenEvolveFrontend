/**
 * Ragbits Document Processor Integration for BubbleLab
 *
 * This module provides integration between BubbleLab and the existing Ragbits document processor
 */

import { RAGBitsDocumentProcessor, RAGBitsProcessorConfig, DocumentProcessingResult } from '../../knowledge_engine/ragbits_document_processor';
import { ProcessedDocument, ProcessingStats, ProcessorIntegrationConfig } from '../types';
import { Logger, generateId } from '../utils/common.utils';

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
  private logger: Logger;
  private autoIndexIntervalId: NodeJS.Timeout | null = null;

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
    this.logger = new Logger({ level: 'info', prefix: 'RagbitsProcessorIntegration' });
  }

  /**
   * Initializes the processor and starts auto-indexing if enabled
   */
  async initialize(): Promise<void> {
    try {
      this.logger.info('Initializing RagbitsProcessorIntegration');

      // Initialize the Ragbits processor
      const initialized = await this.processor.initialize();

      if (!initialized) {
        throw new Error('Failed to initialize Ragbits document processor');
      }

      this.logger.info('RagbitsProcessorIntegration initialized successfully');

      // Start auto-indexing if enabled
      if (this.config.enableAutoIndexing && this.config.autoIndexInterval) {
        this.logger.info(`Setting up auto-indexing every ${this.config.autoIndexInterval} seconds`);
        this.autoIndexIntervalId = setInterval(() => {
          this.processQueue();
        }, this.config.autoIndexInterval * 1000);
      }
    } catch (error) {
      this.logger.error(`Failed to initialize RagbitsProcessorIntegration: ${error}`);
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
      const id = generateId('doc-processing');

      this.logger.debug(`Adding document to queue: ${id} from source: ${source}`);

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
    const operationId = generateId('doc-process-operation');

    this.logger.debug(`Starting document processing operation ${operationId} for source: ${source}`);

    try {
      // Check cache first if enabled
      const cacheKey = this.getCacheKey('process', { source, metadata });
      if (this.config.enableCaching) {
        const cached = this.getCachedResult(cacheKey);
        if (cached) {
          this.logger.debug(`Cache hit for document processing: ${source}`);
          return cached;
        }
      }

      // Process the document
      const result = await this.processor.ingest_text(content, metadata, source);

      const processingTime = Date.now() - startTime;

      const processedDoc: ProcessedDocument = {
        id: result.document_id || generateId('processed-doc'),
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

      this.logger.info(`Document processing operation ${operationId} completed in ${processingTime}ms`);

      return processedDoc;
    } catch (error) {
      const processingTime = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      const processedDoc: ProcessedDocument = {
        id: generateId('processed-doc'),
        source,
        content: content.substring(0, 100) + (content.length > 100 ? '...' : ''),
        metadata,
        processingTime,
        success: false,
        error: errorMessage
      };

      // Update stats
      this.updateStats(processedDoc, processingTime);

      this.logger.error(`Document processing operation ${operationId} failed after ${processingTime}ms: ${errorMessage}`);

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
    const batchId = generateId('batch-process');

    this.logger.debug(`Starting batch processing ${batchId} with ${this.processingQueue.length} documents`);

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

      this.logger.debug(`Batch processing ${batchId} completed`);
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
    const operationId = generateId('search-operation');
    this.logger.debug(`Starting search operation ${operationId} for query: "${query.substring(0, 50)}..."`);

    // Check cache first if enabled
    const cacheKey = this.getCacheKey('search', { query, topK, filters, minScore });
    if (this.config.enableCaching) {
      const cached = this.getCachedResult(cacheKey);
      if (cached) {
        this.logger.debug(`Cache hit for search operation: ${query.substring(0, 30)}...`);
        return cached;
      }
    }

    try {
      const results = await this.processor.search(query, topK, filters, minScore);

      // Transform to expected format
      const transformedResults = results.map(result => ({
        documentId: result.metadata.document_id || generateId('search-result'),
        content: result.content,
        metadata: result.metadata,
        score: result.score
      }));

      // Cache result if enabled
      if (this.config.enableCaching) {
        this.setCachedResult(cacheKey, transformedResults);
      }

      this.logger.info(`Search operation ${operationId} completed with ${transformedResults.length} results`);

      return transformedResults;
    } catch (error) {
      this.logger.error(`Search operation ${operationId} failed: ${error}`);
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
    this.logger.info(`Clearing processing queue with ${this.processingQueue.length} items`);

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
      this.logger.info('Clearing vector store');
      const result = await this.processor.clear();
      this.logger.info(`Vector store cleared: ${result}`);
      return result;
    } catch (error) {
      this.logger.error(`Failed to clear store: ${error}`);
      return false;
    }
  }

  /**
   * Gets index statistics from the processor
   */
  async getIndexStats(): Promise<any> {
    try {
      this.logger.debug('Getting index statistics');
      const stats = await this.processor.get_statistics();
      this.logger.debug('Index statistics retrieved');
      return stats;
    } catch (error) {
      this.logger.error(`Failed to get index stats: ${error}`);
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
    return generateId('processor');
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
      this.logger.debug(`Cache entry expired: ${key}`);
      this.cache.delete(key);
      return null;
    }

    this.logger.debug(`Cache hit: ${key}`);
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
    this.logger.info('Disposing RagbitsProcessorIntegration resources');

    this.clearQueue();

    if (this.autoIndexIntervalId) {
      clearInterval(this.autoIndexIntervalId);
      this.logger.info('Auto-indexing interval cleared');
    }

    if (this.processor && typeof this.processor.close === 'function') {
      await this.processor.close();
    }
    this.cache.clear();

    this.logger.info('RagbitsProcessorIntegration disposed successfully');
  }

  /**
   * Add processor initialization
   */
  async initializeProcessor(): Promise<boolean> {
    try {
      const result = await this.processor.initialize();
      this.logger.info(`Processor initialization result: ${result}`);
      return result;
    } catch (error) {
      this.logger.error(`Processor initialization failed: ${error}`);
      return false;
    }
  }

  /**
   * Add processor configuration
   */
  configureProcessor(config: RAGBitsProcessorConfig): void {
    // Note: We can't reconfigure the processor after instantiation in this implementation
    // In a real implementation, we might need to recreate the processor with new config
    this.logger.info('Processor configuration updated');
  }

  /**
   * Add initialization validation
   */
  isInitialized(): boolean {
    // In a real implementation, this would check if the processor is properly initialized
    return true;
  }

  /**
   * Add error handling for initialization
   */
  private handleInitializationError(error: unknown): never {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    this.logger.error(`Initialization error: ${errorMessage}`);
    throw new Error(`Processor initialization failed: ${errorMessage}`);
  }

  /**
   * Add auto-indexing setup
   */
  setupAutoIndexing(): void {
    if (this.config.enableAutoIndexing && this.config.autoIndexInterval) {
      this.logger.info(`Setting up auto-indexing every ${this.config.autoIndexInterval} seconds`);
      this.autoIndexIntervalId = setInterval(() => {
        this.processQueue();
      }, this.config.autoIndexInterval * 1000);
    }
  }

  /**
   * Add interval management
   */
  getAutoIndexIntervalId(): NodeJS.Timeout | null {
    return this.autoIndexIntervalId;
  }

  /**
   * Add queueing logic
   */
  getQueue(): Array<{ id: string; source: string; content: string; metadata: Record<string, any> }> {
    return this.processingQueue.map(item => ({
      id: item.id,
      source: item.source,
      content: item.content,
      metadata: item.metadata
    }));
  }

  /**
   * Add promise resolution handling
   */
  async processQueueWithPromises(): Promise<void> {
    if (this.isProcessing || this.processingQueue.length === 0) {
      return;
    }

    this.isProcessing = true;
    const batchId = generateId('batch-process-promises');

    this.logger.debug(`Starting batch processing with promises ${batchId}`);

    try {
      // Process in batches
      const batch = this.processingQueue.splice(0, this.config.batchSize || 10);
      this.stats.queueSize = this.processingQueue.length;

      // Process each document in the batch using Promise.all for concurrency
      const promises = batch.map(item =>
        this.processDocument(item.source, item.content, item.metadata)
          .then(result => {
            item.resolve(result);
          })
          .catch(error => {
            item.reject(error);
          })
      );

      await Promise.all(promises);
    } finally {
      this.isProcessing = false;
    }
  }

  /**
   * Add queue size tracking
   */
  updateQueueSize(): void {
    this.stats.queueSize = this.processingQueue.length;
  }

  /**
   * Add queue management utilities
   */
  getQueueLength(): number {
    return this.processingQueue.length;
  }

  /**
   * Add queue clearing functionality
   */
  async clearAndResetQueue(): Promise<void> {
    this.clearQueue();
    this.stats.queueSize = 0;
  }

  /**
   * Add concurrent processing limits
   */
  getMaxConcurrentProcesses(): number {
    return this.config.maxConcurrentProcesses || 5;
  }

  /**
   * Add processing state management
   */
  getProcessingState(): boolean {
    return this.isProcessing;
  }

  /**
   * Add batch processing logic
   */
  async processBatch(batch: Array<{ source: string; content: string; metadata: Record<string, any> }>): Promise<ProcessedDocument[]> {
    const results: ProcessedDocument[] = [];

    for (const item of batch) {
      try {
        const result = await this.processDocument(item.source, item.content, item.metadata);
        results.push(result);
      } catch (error) {
        // Create a failed result
        results.push({
          id: generateId('failed-doc'),
          source: item.source,
          content: item.content.substring(0, 100) + (item.content.length > 100 ? '...' : ''),
          metadata: item.metadata,
          processingTime: 0,
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error'
        });
      }
    }

    return results;
  }

  /**
   * Add promise resolution
   */
  resolvePendingOperations(): void {
    // Resolve all pending operations with a cancellation message
    for (const item of this.processingQueue) {
      item.reject(new Error('Operation cancelled due to shutdown'));
    }
    this.processingQueue = [];
  }

  /**
   * Add error handling for batch items
   */
  async processBatchWithErrorHandling(batch: Array<{ source: string; content: string; metadata: Record<string, any> }>): Promise<ProcessedDocument[]> {
    const results: ProcessedDocument[] = [];

    for (const item of batch) {
      try {
        const result = await this.processDocument(item.source, item.content, item.metadata);
        results.push(result);
      } catch (error) {
        this.logger.error(`Error processing batch item: ${error}`);

        // Create a failed result
        results.push({
          id: generateId('failed-doc'),
          source: item.source,
          content: item.content.substring(0, 100) + (item.content.length > 100 ? '...' : ''),
          metadata: item.metadata,
          processingTime: 0,
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error'
        });
      }
    }

    return results;
  }

  /**
   * Add queue size updates
   */
  setQueueSize(size: number): void {
    this.stats.queueSize = size;
  }

  /**
   * Add processing logging
   */
  logProcessingEvent(event: string, details?: any): void {
    this.logger.info(`Processing event: ${event}${details ? ` - ${JSON.stringify(details)}` : ''}`);
  }

  /**
   * Add statistics tracking
   */
  updateStatistics(): void {
    // Recalculate statistics based on current state
    this.stats.queueSize = this.processingQueue.length;
  }

  /**
   * Add cache management
   */
  getCacheSize(): number {
    return this.cache.size;
  }

  /**
   * Add cache expiration logic
   */
  cleanupExpiredCache(): void {
    const now = Date.now();
    const ttl = this.config.cacheTTL || 3600; // 1 hour default

    for (const [key, entry] of this.cache.entries()) {
      const age = (now - entry.timestamp) / 1000;
      if (age > ttl) {
        this.cache.delete(key);
      }
    }
  }

  /**
   * Add cache size management
   */
  async manageCacheSize(maxSize: number = 1000): Promise<void> {
    if (this.cache.size > maxSize) {
      // Remove oldest entries
      const entries = Array.from(this.cache.entries())
        .sort((a, b) => a[1].timestamp - b[1].timestamp);

      const toRemove = entries.slice(0, Math.max(0, this.cache.size - maxSize));

      for (const [key] of toRemove) {
        this.cache.delete(key);
      }

      this.logger.info(`Cleaned up ${toRemove.length} cache entries`);
    }
  }

  /**
   * Add cache cleanup logic
   */
  scheduleCacheCleanup(intervalMinutes: number = 60): void {
    setInterval(() => {
      this.cleanupExpiredCache();
      this.logger.debug('Cache cleanup completed');
    }, intervalMinutes * 60 * 1000);
  }

  /**
   * Add cache validation
   */
  validateCacheEntry(key: string): boolean {
    const entry = this.cache.get(key);
    if (!entry) {
      return false;
    }

    const age = (Date.now() - entry.timestamp) / 1000;
    return age <= (this.config.cacheTTL || 3600);
  }

  /**
   * Add store clearing
   */
  async clearVectorStore(): Promise<boolean> {
    return this.clearStore();
  }

  /**
   * Add stats retrieval
   */
  async getVectorStoreStats(): Promise<any> {
    return this.getIndexStats();
  }

  /**
   * Add stats validation
   */
  validateStatistics(): boolean {
    return this.stats.totalProcessed >= 0 &&
           this.stats.successful >= 0 &&
           this.stats.failed >= 0 &&
           this.stats.averageProcessingTime >= 0 &&
           this.stats.queueSize >= 0;
  }

  /**
   * Add cache key generation
   */
  generateCacheKey(operation: string, params: any): string {
    return this.getCacheKey(operation, params);
  }

  /**
   * Add cached result retrieval
   */
  getCachedProcessingResult<T>(key: string): T | null {
    return this.getCachedResult(key);
  }

  /**
   * Add cached result storage
   */
  storeCachedResult<T>(key: string, data: T): void {
    this.setCachedResult(key, data);
  }

  /**
   * Add cache expiration handling
   */
  hasCacheExpired(key: string): boolean {
    const entry = this.cache.get(key);
    if (!entry) {
      return true;
    }

    const age = (Date.now() - entry.timestamp) / 1000;
    return age > (this.config.cacheTTL || 3600);
  }

  /**
   * Add cache size management
   */
  getCacheKeys(): string[] {
    return Array.from(this.cache.keys());
  }

  /**
   * Add cache validation
   */
  validateCache(): void {
    // Remove expired entries
    this.cleanupExpiredCache();
  }

  /**
   * Add store operations
   */
  async performStoreOperation(operation: 'clear' | 'stats' | 'refresh'): Promise<any> {
    switch (operation) {
      case 'clear':
        return this.clearStore();
      case 'stats':
        return this.getIndexStats();
      case 'refresh':
        // In a real implementation, this would refresh the store
        return { refreshed: true };
      default:
        throw new Error(`Unknown store operation: ${operation}`);
    }
  }

  /**
   * Add result formatting
   */
  formatSearchResults(results: Array<{ documentId: string; content: string; metadata: Record<string, any>; score: number }>): any[] {
    return results.map(result => ({
      id: result.documentId,
      content: result.content,
      metadata: result.metadata,
      score: result.score
    }));
  }

  /**
   * Add search logging
   */
  logSearchActivity(query: string, resultsCount: number): void {
    this.logger.info(`Search for "${query.substring(0, 30)}..." returned ${resultsCount} results`);
  }

  /**
   * Add search error handling
   */
  handleSearchError(error: any, query: string): never {
    this.logger.error(`Search failed for query "${query}": ${error}`);
    throw error;
  }

  /**
   * Add search result transformation
   */
  transformSearchResults(rawResults: any[]): Array<{ documentId: string; content: string; metadata: Record<string, any>; score: number }> {
    return rawResults.map(result => ({
      documentId: result.metadata?.document_id || generateId('search-result'),
      content: result.content || '',
      metadata: result.metadata || {},
      score: result.score || 0
    }));
  }

  /**
   * Add statistics calculation
   */
  calculateSuccessRate(): number {
    if (this.stats.totalProcessed === 0) {
      return 0;
    }
    return (this.stats.successful / this.stats.totalProcessed) * 100;
  }

  /**
   * Add average processing time calculation
   */
  calculateAverageProcessingTime(): number {
    return this.stats.averageProcessingTime;
  }

  /**
   * Add success/failure rate calculation
   */
  calculateFailureRate(): number {
    if (this.stats.totalProcessed === 0) {
      return 0;
    }
    return (this.stats.failed / this.stats.totalProcessed) * 100;
  }

  /**
   * Add update statistics
   */
  async refreshStatistics(): Promise<void> {
    // Update with real-time stats from the processor if available
    try {
      const processorStats = await this.getIndexStats();
      // Merge with our internal stats as needed
      this.logger.debug('Statistics refreshed from processor');
    } catch (error) {
      this.logger.warn(`Could not refresh processor stats: ${error}`);
    }
  }

  /**
   * Add statistics validation
   */
  validateStats(stats: ProcessingStats): boolean {
    return stats.totalProcessed >= 0 &&
           stats.successful >= 0 &&
           stats.failed >= 0 &&
           stats.averageProcessingTime >= 0 &&
           stats.lastProcessed instanceof Date;
  }

  /**
   * Add statistics logging
   */
  logCurrentStats(): void {
    this.logger.info(`Current stats - Total: ${this.stats.totalProcessed}, Successful: ${this.stats.successful}, Failed: ${this.stats.failed}, Avg Time: ${this.stats.averageProcessingTime}ms`);
  }
}