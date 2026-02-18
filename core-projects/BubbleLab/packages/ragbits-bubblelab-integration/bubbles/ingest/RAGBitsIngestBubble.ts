import { z } from 'zod';
import { ServiceBubble } from '../../../bubble-core/dist/types/service-bubble-class.js';
import type { BubbleContext } from '../../../bubble-core/dist/types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';

// Define the parameters schema for the RAGBits ingest bubble
const RAGBitsIngestParamsSchema = z.object({
  content: z.string().min(1, 'Content is required').describe('Document content to ingest'),
  source: z.string().optional().default('manual').describe('Document source identifier'),
  metadata: z.record(z.string(), z.any()).optional().default({}).describe('Additional metadata for the document'),
  serverUrl: z.string().url('Server URL must be a valid URL').describe('RAGBits server URL'),
  apiKey: z.string().optional().describe('API key for RAGBits server authentication'),
  credentials: z
    .record(z.nativeEnum(CredentialType), z.string())
    .optional()
    .describe('Object mapping credential types to values (injected at runtime)'),
});

// Input type (what the constructor accepts)
type RAGBitsIngestParamsInput = z.input<typeof RAGBitsIngestParamsSchema>;
// Output type (what gets validated and stored)
type RAGBitsIngestParams = z.output<typeof RAGBitsIngestParamsSchema>;

// Define the result schema for validation
const RAGBitsIngestResultSchema = z.object({
  success: z.boolean().describe('Whether the ingestion was successful'),
  documentId: z.string().describe('ID of the ingested document'),
  chunksIngested: z.number().describe('Number of chunks ingested'),
  processingTime: z.number().describe('Time taken for processing in seconds'),
  error: z.string().optional().describe('Error message if operation failed'),
});

type RAGBitsIngestResult = z.output<typeof RAGBitsIngestResultSchema>;

export class RAGBitsIngestBubble extends ServiceBubble<
  RAGBitsIngestParams,
  RAGBitsIngestResult
> {
  static readonly service = 'ragbits';
  static readonly authType = 'apiKey' as const;
  static readonly bubbleName = 'ragbits-ingest';
  static readonly type = 'service' as const;
  static readonly schema = RAGBitsIngestParamsSchema;
  static readonly resultSchema = RAGBitsIngestResultSchema;
  static readonly shortDescription = 'Ingest documents into RAGBits vector store';
  static readonly longDescription = `
    A bubble that ingests documents into the RAGBits vector store for semantic search.
    Use cases:
    - Adding documents to knowledge base for retrieval
    - Indexing content for semantic search
    - Building document collections for RAG workflows
  `;
  static readonly alias = 'ragbits-ingest';

  constructor(
    params: RAGBitsIngestParamsInput,
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
  ): Promise<RAGBitsIngestResult> {
    // Context is available but not currently used in this implementation
    void context;

    try {
      const response = await fetch(`${this.params.serverUrl}/ingest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.params.apiKey ? { 'Authorization': `Bearer ${this.params.apiKey}` } : {}),
        },
        body: JSON.stringify({
          content: this.params.content,
          metadata: this.params.metadata,
          source: this.params.source
        }),
      });

      if (!response.ok) {
        const errorData = await response.text();
        return {
          success: false,
          documentId: '',
          chunksIngested: 0,
          processingTime: 0,
          error: `HTTP ${response.status}: ${errorData}`
        };
      }

      const result = await response.json();
      
      return {
        success: result.success,
        documentId: result.document_id,
        chunksIngested: result.chunks_ingested,
        processingTime: result.processing_time,
        error: result.error
      };
    } catch (error) {
      return {
        success: false,
        documentId: '',
        chunksIngested: 0,
        processingTime: 0,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }
}
