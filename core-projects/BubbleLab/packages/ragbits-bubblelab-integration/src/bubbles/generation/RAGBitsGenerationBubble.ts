/**
 * RAGBits Generation Bubble
 * Bubble component for response generation operations
 */

import { BaseBubble } from '../BaseBubble';
import { RAGBitsGenerationConfig, RAGBitsGenerationInput, RAGBitsGenerationOutput, BubbleLabNode } from '../../types';
import { RAGBitsDocumentProcessor } from '../../integration/RagbitsProcessorIntegration';

/**
 * RAGBitsGenerationBubble - Bubble for response generation
 * Handles context formatting, prompt construction, and LLM integration
 */
export class RAGBitsGenerationBubble extends BaseBubble<RAGBitsGenerationConfig> {
  private processor: RAGBitsDocumentProcessor | null = null;
  
  /**
   * Constructor
   * @param config - Generation configuration
   */
  constructor(config: RAGBitsGenerationConfig) {
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
   * Validate generation configuration
   * @param config - Configuration to validate
   * @returns Validated configuration
   */
  protected validateConfig(config: RAGBitsGenerationConfig): RAGBitsGenerationConfig {
    const baseConfig = super.validateConfig(config);
    
    // Validate model
    if (!baseConfig.model || typeof baseConfig.model !== 'string') {
      throw new Error('model is required and must be a string');
    }
    
    // Validate parameters
    if (baseConfig.parameters) {
      if (baseConfig.parameters.temperature !== undefined && 
          (baseConfig.parameters.temperature < 0 || baseConfig.parameters.temperature > 2)) {
        throw new Error('temperature must be between 0 and 2');
      }
      
      if (baseConfig.parameters.maxTokens !== undefined && 
          (baseConfig.parameters.maxTokens <= 0 || baseConfig.parameters.maxTokens > 4096)) {
        throw new Error('maxTokens must be between 1 and 4096');
      }
      
      if (baseConfig.parameters.topP !== undefined && 
          (baseConfig.parameters.topP < 0 || baseConfig.parameters.topP > 1)) {
        throw new Error('topP must be between 0 and 1');
      }
    }
    
    return baseConfig;
  }
  
  /**
   * Execute response generation
   * @param input - Generation input data
   * @returns Promise with generation output
   */
  public async action(input: RAGBitsGenerationInput): Promise<RAGBitsGenerationOutput> {
    if (!this.initialized) {
      throw new Error('Bubble not initialized');
    }
    
    // Merge input with config
    const mergedInput: RAGBitsGenerationInput = {
      context: input.context,
      query: input.query,
      params: {
        model: input.params?.model || this.config.model,
        temperature: input.params?.temperature || this.config.parameters?.temperature,
        maxTokens: input.params?.maxTokens || this.config.parameters?.maxTokens
      },
      promptTemplate: input.promptTemplate || this.config.promptTemplate?.user
    };
    
    return this.measureExecutionTime(async () => {
      try {
        this.log('info', `Generating response using model: ${mergedInput.params?.model}`);
        
        if (!this.processor) {
          // Mock mode - return simulated results
          return this.mockGeneration(mergedInput);
        }
        
        // Real processor mode
        return this.processor.generate(mergedInput);
      } catch (error) {
        this.handleError(error as Error, 'response generation');
        throw error; // This line will never be reached due to handleError throwing
      }
    }).then(({ result, time }) => {
      this.updateMetrics(true, time);
      this.log('info', `Generation completed in ${time}ms`);
      return result;
    }).catch(error => {
      this.updateMetrics(false, 0);
      throw error;
    });
  }
  
  /**
   * Mock generation for testing without processor
   * @param input - Generation input
   * @returns Simulated generation output
   */
  private mockGeneration(input: RAGBitsGenerationInput): RAGBitsGenerationOutput {
    this.log('warn', 'Using mock generation - no real document processor available');
    
    const contextSummary = this.formatContextSummary(input.context);
    const query = input.query || 'sample query';
    const model = input.params?.model || this.config.model || 'mock-llm';
    
    return {
      response: `This is a mock generated response based on the query "${query}" and context: ${contextSummary}. ` +
                `The response demonstrates how a real LLM would combine the query intent with relevant context ` +
                `to produce a coherent and informative answer.`,
      metadata: {
        model: model,
        tokensUsed: 150, // Simulated token usage
        completionTime: 200, // Simulated completion time
        confidenceScore: 0.95 // Simulated confidence
      },
      contextSummary: contextSummary,
      quality: {
        coherence: 0.9,
        relevance: 0.95,
        completeness: 0.85
      }
    };
  }
  
  /**
   * Format context for generation
   * @param context - Raw context (string or search results)
   * @returns Formatted context string
   */
  private formatContextSummary(context: string | any[]): string {
    if (typeof context === 'string') {
      return context.length > 200 ? context.substring(0, 200) + '...' : context;
    }
    
    if (Array.isArray(context)) {
      const summaries = context.slice(0, 3).map((item, i) => {
        if (typeof item === 'string') {
          return item;
        } else if (item.content) {
          return item.content.substring(0, 100) + '...';
        } else {
          return `Document ${i + 1}`;
        }
      });
      
      return summaries.join(' | ');
    }
    
    return 'No context provided';
  }
  
  /**
   * Construct prompt for generation
   * @param query - User query
   * @param context - Context from search results
   * @returns Formatted prompt
   */
  private constructPrompt(query: string, context: string | any[]): string {
    const contextText = this.formatContextSummary(context);
    
    // Use template if available
    if (this.config.promptTemplate?.user) {
      return this.config.promptTemplate.user
        .replace('{query}', query)
        .replace('{context}', contextText);
    }
    
    // Default prompt template
    return `Answer the following question based on the provided context:

Question: ${query}

Context: ${contextText}

Answer:`;
  }
  
  /**
   * Format generation output
   * @param response - Raw LLM response
   * @param metadata - Generation metadata
   * @returns Formatted output
   */
  private formatGenerationOutput(response: string, metadata: any): RAGBitsGenerationOutput {
    return {
      response: response.trim(),
      metadata: {
        model: metadata.model || this.config.model,
        tokensUsed: metadata.tokensUsed || 0,
        completionTime: metadata.completionTime || 0,
        confidenceScore: metadata.confidenceScore
      },
      contextSummary: metadata.contextSummary || 'Generated from provided context',
      quality: metadata.quality || {
        coherence: 0.8,
        relevance: 0.8,
        completeness: 0.7
      }
    };
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