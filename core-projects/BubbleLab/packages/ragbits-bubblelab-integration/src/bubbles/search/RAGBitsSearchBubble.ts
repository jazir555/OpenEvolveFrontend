/**
 * RAGBits Search Bubble
 * Bubble component for semantic search operations
 */

import { BaseBubble } from '../BaseBubble';
import { RAGBitsSearchConfig, RAGBitsSearchInput, RAGBitsSearchOutput, BubbleLabNode } from '../../types';
import { RAGBitsDocumentProcessor } from '../../integration/RagbitsProcessorIntegration';

/**
 * RAGBitsSearchBubble - Bubble for semantic search
 * Handles query processing, filtering, and result formatting
 */
export class RAGBitsSearchBubble extends BaseBubble<RAGBitsSearchConfig> {
  private processor: RAGBitsDocumentProcessor | null = null;
  
  /**
   * Constructor
   * @param config - Search configuration
   */
  constructor(config: RAGBitsSearchConfig) {
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
   * Validate search configuration
   * @param config - Configuration to validate
   * @returns Validated configuration
   */
  protected validateConfig(config: RAGBitsSearchConfig): RAGBitsSearchConfig {
    const baseConfig = super.validateConfig(config);
    
    // Validate search strategy
    if (!['semantic', 'keyword', 'hybrid'].includes(baseConfig.searchStrategy)) {
      throw new Error(`Invalid searchStrategy: ${baseConfig.searchStrategy}`);
    }
    
    // Validate topK
    if (!baseConfig.topK || baseConfig.topK <= 0) {
      throw new Error('topK must be a positive number');
    }
    
    // Validate similarity threshold
    if (baseConfig.similarityThreshold !== undefined && 
        (baseConfig.similarityThreshold < 0 || baseConfig.similarityThreshold > 1)) {
      throw new Error('similarityThreshold must be between 0 and 1');
    }
    
    return baseConfig;
  }
  
  /**
   * Execute semantic search
   * @param input - Search input data
   * @returns Promise with search output
   */
  public async action(input: RAGBitsSearchInput): Promise<RAGBitsSearchOutput> {
    if (!this.initialized) {
      throw new Error('Bubble not initialized');
    }
    
    // Merge input with config
    const mergedInput: RAGBitsSearchInput = {
      query: input.query,
      params: {
        topK: input.params?.topK || this.config.topK,
        similarityThreshold: input.params?.similarityThreshold || this.config.similarityThreshold,
        filters: input.params?.filters || this.config.filters
      },
      context: input.context
    };
    
    return this.measureExecutionTime(async () => {
      try {
        this.log('info', `Executing ${this.config.searchStrategy} search for query: "${mergedInput.query}"`);
        
        if (!this.processor) {
          // Mock mode - return simulated results
          return this.mockSearch(mergedInput);
        }
        
        // Real processor mode
        return this.processor.search(mergedInput);
      } catch (error) {
        this.handleError(error as Error, 'search execution');
        throw error; // This line will never be reached due to handleError throwing
      }
    }).then(({ result, time }) => {
      this.updateMetrics(true, time);
      this.log('info', `Search completed in ${time}ms, returned ${result.results.length} results`);
      return result;
    }).catch(error => {
      this.updateMetrics(false, 0);
      throw error;
    });
  }
  
  /**
   * Mock search for testing without processor
   * @param input - Search input
   * @returns Simulated search output
   */
  private mockSearch(input: RAGBitsSearchInput): RAGBitsSearchOutput {
    this.log('warn', 'Using mock search - no real document processor available');
    
    const topK = input.params?.topK || this.config.topK || 5;
    const query = input.query || 'sample query';
    
    return {
      results: Array(topK).fill(0).map((_, i) => ({
        documentId: `mock-doc-${i}`,
        content: `This is a mock search result for query "${query}". Result ${i + 1} of ${topK}.`,
        score: 1 - (i * 0.1), // Decreasing scores
        metadata: {
          source: 'mock',
          mock: true,
          rank: i + 1
        },
        highlights: [`Mock highlight ${i + 1} for "${query}"`]
      })),
      stats: {
        totalDocuments: 100, // Simulated total documents
        documentsReturned: topK,
        searchTime: 50, // Simulated search time
        rerankTime: input.params?.filters?.rerank?.enabled ? 20 : 0
      },
      queryMetadata: {
        originalQuery: query,
        processedQuery: query.toLowerCase(),
        timestamp: new Date()
      }
    };
  }
  
  /**
   * Apply filters to search results
   * @param results - Original search results
   * @param filters - Filters to apply
   * @returns Filtered results
   */
  private applyFilters(results: any[], filters?: any): any[] {
    if (!filters) {
      return results;
    }
    
    let filteredResults = [...results];
    
    // Apply metadata filters
    if (filters.metadata) {
      filteredResults = filteredResults.filter(result => {
        return Object.entries(filters.metadata).every(([key, value]) => {
          return result.metadata?.[key] === value;
        });
      });
    }
    
    // Apply date range filters
    if (filters.dateRange) {
      const startDate = filters.dateRange.start ? new Date(filters.dateRange.start) : null;
      const endDate = filters.dateRange.end ? new Date(filters.dateRange.end) : null;
      
      filteredResults = filteredResults.filter(result => {
        const docDate = result.metadata?.date ? new Date(result.metadata.date) : null;
        
        if (!docDate) return true;
        
        if (startDate && docDate < startDate) return false;
        if (endDate && docDate > endDate) return false;
        
        return true;
      });
    }
    
    // Apply source type filters
    if (filters.sourceTypes && filters.sourceTypes.length > 0) {
      filteredResults = filteredResults.filter(result => {
        return filters.sourceTypes.includes(result.metadata?.sourceType);
      });
    }
    
    return filteredResults;
  }
  
  /**
   * Format search results
   * @param results - Raw search results
   * @returns Formatted results
   */
  private formatResults(results: any[]): any[] {
    return results.map(result => ({
      documentId: result.id || result.documentId,
      content: result.content || result.text || '',
      score: result.score || result.similarity || 0,
      metadata: result.metadata || {},
      highlights: result.highlights || []
    }));
  }
  
  /**
   * Process query for search
   * @param query - Original query
   * @returns Processed query
   */
  private processQuery(query: string): string {
    // Basic query processing
    return query
      .trim()
      .toLowerCase()
      .replace(/\s+/, ' ');
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