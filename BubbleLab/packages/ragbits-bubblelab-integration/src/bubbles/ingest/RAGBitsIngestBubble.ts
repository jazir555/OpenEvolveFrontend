/**
 * RAGBits Ingest Bubble
 * Bubble component for document ingestion operations
 */

import { BaseBubble } from '../BaseBubble';
import { RAGBitsIngestConfig, RAGBitsIngestInput, RAGBitsIngestOutput, BubbleLabNode } from '../../types';
import { RAGBitsDocumentProcessor } from '../../integration/RagbitsProcessorIntegration';

/**
 * RAGBitsIngestBubble - Bubble for document ingestion
 * Handles file, URL, and text ingestion with processing options
 */
export class RAGBitsIngestBubble extends BaseBubble<RAGBitsIngestConfig> {
  private processor: RAGBitsDocumentProcessor | null = null;
  
  /**
   * Constructor
   * @param config - Ingest configuration
   */
  constructor(config: RAGBitsIngestConfig) {
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
   * Validate ingest configuration
   * @param config - Configuration to validate
   * @returns Validated configuration
   */
  protected validateConfig(config: RAGBitsIngestConfig): RAGBitsIngestConfig {
    const baseConfig = super.validateConfig(config);
    
    // Validate source type
    if (!['file', 'url', 'text'].includes(baseConfig.sourceType)) {
      throw new Error(`Invalid sourceType: ${baseConfig.sourceType}`);
    }
    
    // Validate source-specific requirements
    switch (baseConfig.sourceType) {
      case 'file':
        if (!baseConfig.filePaths || baseConfig.filePaths.length === 0) {
          throw new Error('filePaths is required for file source type');
        }
        break;
      case 'url':
        if (!baseConfig.urls || baseConfig.urls.length === 0) {
          throw new Error('urls is required for url source type');
        }
        break;
      case 'text':
        if (!baseConfig.textContent || baseConfig.textContent.trim() === '') {
          throw new Error('textContent is required for text source type');
        }
        break;
    }
    
    return baseConfig;
  }
  
  /**
   * Execute document ingestion
   * @param input - Ingest input data
   * @returns Promise with ingest output
   */
  public async action(input: RAGBitsIngestInput): Promise<RAGBitsIngestOutput> {
    if (!this.initialized) {
      throw new Error('Bubble not initialized');
    }
    
    // Merge input with config
    const mergedInput: RAGBitsIngestInput = {
      ...this.config,
      ...input,
      sourceType: this.config.sourceType,
      filePaths: input.filePaths || this.config.filePaths,
      urls: input.urls || this.config.urls,
      textContent: input.textContent || this.config.textContent,
      auth: input.auth || this.config.auth,
      options: input.options || this.config.processingOptions
    };
    
    return this.measureExecutionTime(async () => {
      try {
        this.log('info', `Starting ingestion from ${mergedInput.sourceType}`);
        
        if (!this.processor) {
          // Mock mode - return simulated results
          return this.mockIngestion(mergedInput);
        }
        
        // Real processor mode
        return this.processor.ingestDocuments(mergedInput);
      } catch (error) {
        this.handleError(error as Error, 'document ingestion');
        throw error; // This line will never be reached due to handleError throwing
      }
    }).then(({ result, time }) => {
      this.updateMetrics(true, time);
      this.log('info', `Ingestion completed in ${time}ms`);
      return result;
    }).catch(error => {
      this.updateMetrics(false, 0);
      throw error;
    });
  }
  
  /**
   * Mock ingestion for testing without processor
   * @param input - Ingest input
   * @returns Simulated ingest output
   */
  private mockIngestion(input: RAGBitsIngestInput): RAGBitsIngestOutput {
    this.log('warn', 'Using mock ingestion - no real document processor available');
    
    // Simulate processing based on source type
    let documentCount = 0;
    
    switch (input.sourceType) {
      case 'file':
        documentCount = input.filePaths?.length || 0;
        break;
      case 'url':
        documentCount = input.urls?.length || 0;
        break;
      case 'text':
        documentCount = 1;
        break;
    }
    
    return {
      documents: Array(documentCount).fill(0).map((_, i) => ({
        id: `mock-doc-${i}`,
        content: `Mock document content ${i + 1}`,
        metadata: {
          source: input.sourceType,
          processed: true,
          mock: true
        },
        processedAt: new Date(),
        source: { type: input.sourceType }
      })),
      stats: {
        totalDocuments: documentCount,
        successful: documentCount,
        failed: 0,
        processingTime: 100, // Simulated processing time
        avgProcessingTime: 100 / documentCount,
        tokensProcessed: documentCount * 100 // Simulated token count
      }
    };
  }
  
  /**
   * Handle file ingestion
   * @param filePaths - Array of file paths
   * @returns Promise with ingest results
   */
  private async handleFileIngestion(filePaths: string[]): Promise<RAGBitsIngestOutput> {
    this.log('info', `Processing ${filePaths.length} files`);
    
    if (!this.processor) {
      return this.mockIngestion({ sourceType: 'file', filePaths });
    }
    
    return this.processor.ingestDocuments({
      sourceType: 'file',
      filePaths,
      auth: this.config.auth,
      options: this.config.processingOptions
    });
  }
  
  /**
   * Handle URL ingestion
   * @param urls - Array of URLs
   * @returns Promise with ingest results
   */
  private async handleUrlIngestion(urls: string[]): Promise<RAGBitsIngestOutput> {
    this.log('info', `Processing ${urls.length} URLs`);
    
    if (!this.processor) {
      return this.mockIngestion({ sourceType: 'url', urls });
    }
    
    return this.processor.ingestDocuments({
      sourceType: 'url',
      urls,
      auth: this.config.auth,
      options: this.config.processingOptions
    });
  }
  
  /**
   * Handle text ingestion
   * @param textContent - Text content to ingest
   * @returns Promise with ingest results
   */
  private async handleTextIngestion(textContent: string): Promise<RAGBitsIngestOutput> {
    this.log('info', 'Processing text content');
    
    if (!this.processor) {
      return this.mockIngestion({ sourceType: 'text', textContent });
    }
    
    return this.processor.ingestDocuments({
      sourceType: 'text',
      textContent,
      options: this.config.processingOptions
    });
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