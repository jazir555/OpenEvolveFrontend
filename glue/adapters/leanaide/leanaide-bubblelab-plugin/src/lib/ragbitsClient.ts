import { ApiHttpError } from './apiTypes';

export interface RagbitsConfig {
  serverUrl: string;
  apiKey?: string;
}

export interface RagbitsSearchRequest {
  query: string;
  filters?: Record<string, unknown>;
  topK?: number;
}

export interface RagbitsSearchResult {
  content: string;
  metadata?: Record<string, unknown>;
  score?: number;
}

export interface RagbitsSearchResponse {
  success: boolean;
  results: RagbitsSearchResult[];
  error?: string;
}

export interface RagbitsIngestRequest {
  content: string;
  metadata?: Record<string, unknown>;
}

export interface RagbitsIngestResponse {
  success: boolean;
  artifactId?: string;
  error?: string;
}

export class RagbitsClient {
  private serverUrl: string;
  private apiKey?: string;

  constructor(config: RagbitsConfig) {
    this.serverUrl = config.serverUrl;
    this.apiKey = config.apiKey;
  }

  async search(request: RagbitsSearchRequest): Promise<RagbitsSearchResponse> {
    return this.request('/search', request);
  }

  async ingest(request: RagbitsIngestRequest): Promise<RagbitsIngestResponse> {
    return this.request('/ingest', request);
  }

  private async request<T>(path: string, payload: unknown): Promise<T> {
    const url = `${this.serverUrl}${path}`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers.Authorization = `Bearer ${this.apiKey}`;
    }

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });

    let data: unknown = null;
    try {
      data = await response.json();
    } catch {
      data = { error: 'Invalid JSON response from RAGBits server' };
    }

    if (!response.ok) {
      throw new ApiHttpError(response.status, data);
    }

    return data as T;
  }
}
