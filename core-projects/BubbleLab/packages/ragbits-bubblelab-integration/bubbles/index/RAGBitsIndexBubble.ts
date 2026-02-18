import { z } from 'zod';
import { ServiceBubble } from '../../../bubble-core/dist/types/service-bubble-class.js';
import type { BubbleContext } from '../../../bubble-core/dist/types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';

// Define the parameters schema for the RAGBits index bubble
const RAGBitsIndexParamsSchema = z.object({
  vectorStoreType: z.enum(['memory', 'qdrant']).default('memory').describe('Type of vector store to use'),
  embeddingModel: z.string().default('text-embedding-3-small').describe('Embedding model to use'),
  qdrantUrl: z.string().url().optional().describe('Qdrant URL if using Qdrant vector store'),
  qdrantCollection: z.string().optional().default('knowledge_engine').describe('Qdrant collection name'),
  chunkSize: z.number().min(100).max(2000).default(1000).describe('Size of document chunks'),
  chunkOverlap: z.number().min(0).max(500).default(200).describe('Overlap between chunks'),
  serverUrl: z.string().url('Server URL must be a valid URL').describe('RAGBits server URL'),
  apiKey: z.string().optional().describe('API key for RAGBits server authentication'),
  credentials: z
    .record(z.nativeEnum(CredentialType), z.string())
    .optional()
    .describe('Object mapping credential types to values (injected at runtime)'),
});

// Input type (what the constructor accepts)
type RAGBitsIndexParamsInput = z.input<typeof RAGBitsIndexParamsSchema>;
// Output type (what gets validated and stored)
type RAGBitsIndexParams = z.output<typeof RAGBitsIndexParamsSchema>;

// Define the result schema for validation
const RAGBitsIndexResultSchema = z.object({
  success: z.boolean().describe('Whether the index operation was successful'),
  message: z.string().describe('Status message'),
  vectorStoreType: z.string().describe('Type of vector store used'),
  embeddingModel: z.string().describe('Embedding model used'),
  error: z.string().optional().describe('Error message if operation failed'),
});

type RAGBitsIndexResult = z.output<typeof RAGBitsIndexResultSchema>;

export class RAGBitsIndexBubble extends ServiceBubble<
  RAGBitsIndexParams,
  RAGBitsIndexResult
> {
  static readonly service = 'ragbits';
  static readonly authType = 'apiKey' as const;
  static readonly bubbleName = 'ragbits-index';
  static readonly type = 'service' as const;
  static readonly schema = RAGBitsIndexParamsSchema;
  static readonly resultSchema = RAGBitsIndexResultSchema;
  static readonly shortDescription = 'Manage RAGBits vector index configuration';
  static readonly longDescription = `
    A bubble that manages the RAGBits vector index configuration.
    Use cases:
    - Configuring vector store settings
    - Setting up embedding models
    - Managing index parameters
  `;
  static readonly alias = 'ragbits-index';

  constructor(
    params: RAGBitsIndexParamsInput,
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
  ): Promise<RAGBitsIndexResult> {
    // Context is available but not currently used in this implementation
    void context;

    try {
      // For index configuration, we'll just validate the configuration by getting stats
      const response = await fetch(`${this.params.serverUrl}/stats`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          ...(this.params.apiKey ? { 'Authorization': `Bearer ${this.params.apiKey}` } : {}),
        },
      });

      if (!response.ok) {
        const errorData = await response.text();
        return {
          success: false,
          message: `HTTP ${response.status}: ${errorData}`,
          vectorStoreType: this.params.vectorStoreType,
          embeddingModel: this.params.embeddingModel,
          error: `HTTP ${response.status}: ${errorData}`
        };
      }

      const result = await response.json();
      
      return {
        success: true,
        message: `Index configuration validated. Vector store: ${result.vector_store_type}, Embedding model: ${result.embedding_model}`,
        vectorStoreType: result.vector_store_type,
        embeddingModel: result.embedding_model,
        error: undefined
      };
    } catch (error) {
      return {
        success: false,
        message: 'Failed to validate index configuration',
        vectorStoreType: this.params.vectorStoreType,
        embeddingModel: this.params.embeddingModel,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }
}
