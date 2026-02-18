import { z } from 'zod';
import { ServiceBubble } from '../../../bubble-core/dist/types/service-bubble-class.js';
import type { BubbleContext } from '../../../bubble-core/dist/types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';

// Define the parameters schema for the RAGBits search bubble
const RAGBitsSearchParamsSchema = z.object({
  query: z.string().min(1, 'Query is required').describe('Search query for semantic search'),
  topK: z.number().min(1).max(100).default(5).describe('Number of results to return'),
  filters: z.record(z.string(), z.any()).optional().default({}).describe('Metadata filters for search'),
  minScore: z.number().min(0).max(1).default(0).describe('Minimum similarity score threshold'),
  serverUrl: z.string().url('Server URL must be a valid URL').describe('RAGBits server URL'),
  apiKey: z.string().optional().describe('API key for RAGBits server authentication'),
  credentials: z
    .record(z.nativeEnum(CredentialType), z.string())
    .optional()
    .describe('Object mapping credential types to values (injected at runtime)'),
});

// Input type (what the constructor accepts)
type RAGBitsSearchParamsInput = z.input<typeof RAGBitsSearchParamsSchema>;
// Output type (what gets validated and stored)
type RAGBitsSearchParams = z.output<typeof RAGBitsSearchParamsSchema>;

// Define the result schema for validation
const RAGBitsSearchResultSchema = z.object({
  results: z.array(z.object({
    content: z.string().describe('Content of the search result'),
    score: z.number().describe('Similarity score of the result'),
    metadata: z.record(z.string(), z.any()).describe('Metadata associated with the result'),
  })).describe('Array of search results'),
  totalResults: z.number().describe('Total number of results found'),
  query: z.string().describe('Original search query'),
  success: z.boolean().describe('Whether the search was successful'),
  error: z.string().optional().describe('Error message if operation failed'),
});

type RAGBitsSearchResult = z.output<typeof RAGBitsSearchResultSchema>;

export class RAGBitsSearchBubble extends ServiceBubble<
  RAGBitsSearchParams,
  RAGBitsSearchResult
> {
  static readonly service = 'ragbits';
  static readonly authType = 'apiKey' as const;
  static readonly bubbleName = 'ragbits-search';
  static readonly type = 'service' as const;
  static readonly schema = RAGBitsSearchParamsSchema;
  static readonly resultSchema = RAGBitsSearchResultSchema;
  static readonly shortDescription = 'Perform semantic search using RAGBits';
  static readonly longDescription = `
    A bubble that performs semantic search on documents in the RAGBits vector store.
    Use cases:
    - Retrieving relevant documents based on semantic similarity
    - Finding related content for RAG workflows
    - Knowledge base queries
  `;
  static readonly alias = 'ragbits-search';

  constructor(
    params: RAGBitsSearchParamsInput,
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
  ): Promise<RAGBitsSearchResult> {
    // Context is available but not currently used in this implementation
    void context;

    try {
      const response = await fetch(`${this.params.serverUrl}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.params.apiKey ? { 'Authorization': `Bearer ${this.params.apiKey}` } : {}),
        },
        body: JSON.stringify({
          query: this.params.query,
          top_k: this.params.topK,
          filters: this.params.filters,
          min_score: this.params.minScore
        }),
      });

      if (!response.ok) {
        const errorData = await response.text();
        return {
          results: [],
          totalResults: 0,
          query: this.params.query,
          success: false,
          error: `HTTP ${response.status}: ${errorData}`
        };
      }

      const result = await response.json();
      
      return {
        results: result.results,
        totalResults: result.total_results,
        query: result.query,
        success: true,
        error: undefined
      };
    } catch (error) {
      return {
        results: [],
        totalResults: 0,
        query: this.params.query,
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }
}
