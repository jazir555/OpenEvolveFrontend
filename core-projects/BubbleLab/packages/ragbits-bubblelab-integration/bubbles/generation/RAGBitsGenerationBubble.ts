import { z } from 'zod';
import { ServiceBubble } from '../../../bubble-core/dist/types/service-bubble-class.js';
import type { BubbleContext } from '../../../bubble-core/dist/types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';

// Define the parameters schema for the RAGBits generation bubble
const RAGBitsGenerationParamsSchema = z.object({
  query: z.string().min(1, 'Query is required').describe('Query for RAG generation'),
  context: z.string().optional().describe('Context to provide to the generation model'),
  llmModel: z.string().default('gpt-4o').describe('Language model to use for generation'),
  temperature: z.number().min(0).max(1).default(0.7).describe('Temperature for generation'),
  maxTokens: z.number().min(1).max(4096).default(1000).describe('Maximum tokens in response'),
  searchQuery: z.string().optional().describe('Separate query for semantic search if different from generation query'),
  topK: z.number().min(1).max(100).default(5).describe('Number of search results to include as context'),
  serverUrl: z.string().url('Server URL must be a valid URL').describe('RAGBits server URL'),
  apiKey: z.string().optional().describe('API key for RAGBits server authentication'),
  credentials: z
    .record(z.nativeEnum(CredentialType), z.string())
    .optional()
    .describe('Object mapping credential types to values (injected at runtime)'),
});

// Input type (what the constructor accepts)
type RAGBitsGenerationParamsInput = z.input<typeof RAGBitsGenerationParamsSchema>;
// Output type (what gets validated and stored)
type RAGBitsGenerationParams = z.output<typeof RAGBitsGenerationParamsSchema>;

// Define the result schema for validation
const RAGBitsGenerationResultSchema = z.object({
  response: z.string().describe('Generated response from the model'),
  success: z.boolean().describe('Whether the generation was successful'),
  searchResultsUsed: z.number().describe('Number of search results used as context'),
  model: z.string().describe('Model that was used for generation'),
  error: z.string().optional().describe('Error message if operation failed'),
});

type RAGBitsGenerationResult = z.output<typeof RAGBitsGenerationResultSchema>;

export class RAGBitsGenerationBubble extends ServiceBubble<
  RAGBitsGenerationParams,
  RAGBitsGenerationResult
> {
  static readonly service = 'ragbits';
  static readonly authType = 'apiKey' as const;
  static readonly bubbleName = 'ragbits-generation';
  static readonly type = 'service' as const;
  static readonly schema = RAGBitsGenerationParamsSchema;
  static readonly resultSchema = RAGBitsGenerationResultSchema;
  static readonly shortDescription = 'Generate responses using RAGBits with semantic search';
  static readonly longDescription = `
    A bubble that generates responses using RAGBits with semantic search context.
    Use cases:
    - Generating answers based on retrieved documents
    - Creating content with knowledge base context
    - Building RAG-based assistants
  `;
  static readonly alias = 'ragbits-generation';

  constructor(
    params: RAGBitsGenerationParamsInput,
    context?: BubbleContext
  ) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    // Use the API key from params or return undefined if not provided
    return this.params.apiKey;
  }

  public async testCredential(): Promise<boolean> {
    // Test connection to RAGBits server
    try {
      const response = await fetch(`${this.params.serverUrl}/health`);
      return response.ok;
    } catch (error) {
      console.error('Error testing RAGBits connection:', error);
      return false;
    }
  }

  protected async performAction(
    context?: BubbleContext
  ): Promise<RAGBitsGenerationResult> {
    // Context is available but not currently used in this implementation
    void context;

    try {
      // Call the RAGBits server's generation endpoint which handles both search and generation
      const response = await fetch(`${this.params.serverUrl}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.params.apiKey ? { 'Authorization': `Bearer ${this.params.apiKey}` } : {}),
        },
        body: JSON.stringify({
          query: this.params.query,
          context: this.params.context,
          search_query: this.params.searchQuery || this.params.query,
          top_k: this.params.topK,
          llm_model: this.params.llmModel,
          temperature: this.params.temperature,
          max_tokens: this.params.maxTokens,
          filters: {} // Default empty filters, could be extended
        }),
      });

      if (!response.ok) {
        const errorData = await response.text();
        return {
          response: '',
          success: false,
          searchResultsUsed: 0,
          model: this.params.llmModel,
          error: `Generation failed: HTTP ${response.status}: ${errorData}`
        };
      }

      const result = await response.json();

      return {
        response: result.response || result.generated_text || 'No response generated',
        success: result.success !== false, // Success is true unless explicitly false
        searchResultsUsed: result.search_results_used || result.sources?.length || 0,
        model: result.model_used || this.params.llmModel,
        error: undefined
      };
    } catch (error) {
      return {
        response: '',
        success: false,
        searchResultsUsed: 0,
        model: this.params.llmModel,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }
}
