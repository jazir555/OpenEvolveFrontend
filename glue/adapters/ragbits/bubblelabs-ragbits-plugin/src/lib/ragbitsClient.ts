// RAGBits Client
// Handles HTTP communication with the RAGBits server
// Updated with structured logging, validation, and proper error handling

import { ragbitsLogger, LogContext } from '../../../../../lib/structuredLogger';

export interface RagbitsClientConfig {
  serverUrl: string;
  apiKey?: string;
  timeout?: number;
}

export class RagbitsClient {
  private config: RagbitsClientConfig;
  private correlationId: string;

  constructor(config: RagbitsClientConfig) {
    this.config = config;
    this.correlationId = `ragbits-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
  }

  /**
   * Update client configuration
   */
  configure(config: Partial<RagbitsClientConfig>): void {
    this.config = { ...this.config, ...config };
    ragbitsLogger.info('RAGBits client configuration updated', {
      correlation_id: this.correlationId,
      target_service: 'ragbits',
      has_api_key: !!this.config.apiKey
    });
  }

  /**
   * Test connection to RAGBits server
   */
  async testConnection(): Promise<boolean> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'ragbits-plugin',
      target_service: 'ragbits-server',
      operation: 'test_connection'
    };

    try {
      ragbitsLogger.info('Testing RAGBits connection', context);

      const response = await this.fetch('/health', {
        method: 'GET',
      });

      if (response.ok) {
        ragbitsLogger.info('RAGBits connection successful', context);
      } else {
        ragbitsLogger.warn('RAGBits connection failed', {
          ...context,
          status: response.status,
          status_text: response.statusText
        });
      }

      return response.ok;
    } catch (error) {
      ragbitsLogger.error('RAGBits connection test failed', error as Error, context);
      return false;
    }
  }

  /**
   * Search for documents
   */
  async search(request: {
    query: string;
    topK?: number;
    scoreThreshold?: number;
    filter?: Record<string, any>;
    enableHybridSearch?: boolean;
    enableReranking?: boolean;
  }): Promise<any> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'ragbits-plugin',
      target_service: 'ragbits-server',
      operation: 'search',
      query_length: request.query?.length
    };

    try {
      ragbitsLogger.info('Executing RAGBits search', context);

      const response = await this.fetch('/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorText = await response.text();
        ragbitsLogger.error('RAGBits search failed', new Error(errorText), {
          ...context,
          status: response.status,
          status_text: response.statusText
        });
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const result = await response.json();
      ragbitsLogger.info('RAGBits search successful', {
        ...context,
        result_count: result.results?.length || 0
      });

      return result;
    } catch (error) {
      ragbitsLogger.error('RAGBits search error', error as Error, context);
      throw error;
    }
  }

  /**
   * Ingest a document
   */
  async ingest(request: {
    content: string;
    metadata: Record<string, any>;
  }): Promise<any> {
    const response = await this.fetch('/ingest', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Ingest failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Batch ingest documents
   */
  async batchIngest(requests: Array<{
    content: string;
    metadata: Record<string, any>;
  }>): Promise<any[]> {
    const response = await this.fetch('/ingest/batch', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ documents: requests }),
    });

    if (!response.ok) {
      throw new Error(`Batch ingest failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get index statistics
   */
  async getIndexStats(): Promise<any> {
    const response = await this.fetch('/index/stats', {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error(`Failed to get index stats: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Clear cache
   */
  async clearCache(): Promise<any> {
    const response = await this.fetch('/cache/clear', {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to clear cache: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Perform HTTP request with timeout and error handling
   */
  private async fetch(
    path: string,
    options: RequestInit = {}
  ): Promise<Response> {
    const url = `${this.config.serverUrl}${path}`;
    const timeout = this.config.timeout || 30000;

    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      // Add API key header if provided
      const headers: HeadersInit = {
        ...options.headers,
      };

      if (this.config.apiKey) {
        headers['Authorization'] = `Bearer ${this.config.apiKey}`;
      }

      const response = await fetch(url, {
        ...options,
        headers,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error(`Request timeout after ${timeout}ms`);
        }
        throw error;
      }

      throw new Error('Unknown error occurred');
    }
  }
}
