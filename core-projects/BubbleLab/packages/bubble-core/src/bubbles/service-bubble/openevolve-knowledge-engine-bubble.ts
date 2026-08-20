import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';

const KnowledgeBackendSchema = z.enum([
  'qdrant',
  'elasticsearch',
  'bedrock',
  'eks',
  'hybrid',
]);

const resolveOpenEvolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_KNOWLEDGE_API_URL ||
        process.env.OPENEVOLVE_API_URL ||
        process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const KnowledgeOperationSchema = z.enum([
  'search',
  'semantic_search',
  'hybrid_search',
  'index',
  'batch_index',
  'delete',
  'embed',
  'health_check',
]);

const KnowledgeEngineParamsSchema = z.object({
  operation: KnowledgeOperationSchema,
  backend: KnowledgeBackendSchema.default('qdrant'),

  qdrant: z
    .object({
      base_url: z.string().url(),
      api_key: z.string().optional(),
      collection_name: z.string().min(1),
    })
    .optional(),

  elasticsearch: z
    .object({
      base_url: z.string().url(),
      username: z.string().optional(),
      password: z.string().optional(),
      index: z.string().min(1),
    })
    .optional(),

  bedrock: z
    .object({
      knowledge_base_id: z.string().min(1),
      data_source_id: z.string().optional(),
      region: z.string().optional(),
    })
    .optional(),

  eks: z
    .object({
      endpoint: z.string().url(),
    })
    .optional(),

  query: z.string().optional(),
  query_vector: z.array(z.number()).optional(),
  documents: z
    .array(
      z.object({
        id: z.string(),
        content: z.string(),
        metadata: z.record(z.unknown()).optional(),
      })
    )
    .optional(),
  document_ids: z.array(z.string()).optional(),

  limit: z.number().min(1).max(1000).default(10),
  filters: z.record(z.unknown()).optional(),

  embedding_model: z.string().default('text-embedding-3-small'),
  openai_api_key: z.string().optional(),

  semantic_weight: z.number().min(0).max(1).default(0.5),
  keyword_weight: z.number().min(0).max(1).default(0.5),

  timeout: z.number().int().min(1000).max(120000).default(30000),
});

type KnowledgeEngineParams = z.input<typeof KnowledgeEngineParamsSchema>;

const KnowledgeEngineResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  backend: z.string(),
  results: z
    .array(
      z.object({
        id: z.string(),
        content: z.string(),
        score: z.number().optional(),
        metadata: z.record(z.unknown()).optional(),
      })
    )
    .optional(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type KnowledgeEngineResult = z.output<typeof KnowledgeEngineResultSchema>;

type KnowledgeDocument = {
  id: string;
  content: string;
  metadata?: Record<string, unknown>;
};

type SearchResult = {
  id: string;
  content: string;
  score?: number;
  metadata?: Record<string, unknown>;
};

export class OpenEvolveKnowledgeEngineBubble extends ServiceBubble<
  KnowledgeEngineParams,
  KnowledgeEngineResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName =
    'openevolve-knowledge-engine' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = KnowledgeEngineParamsSchema;
  static readonly resultSchema = KnowledgeEngineResultSchema;
  static readonly shortDescription =
    'OpenEvolve knowledge engine (semantic/hybrid search and indexing)';
  static readonly longDescription = `
    Unified knowledge engine bubble for OpenEvolve.

    Supports:
    - Qdrant semantic search and indexing
    - Elasticsearch keyword search and indexing
    - Hybrid search across semantic + keyword backends
    - Embedding generation (OpenAI)
  `;
  static readonly alias = 'openevolve-knowledge-engine';

  constructor(params: KnowledgeEngineParams, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.openai_api_key;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<KnowledgeEngineResult> {
    const startTime = Date.now();
    try {
      switch (this.params.operation) {
        case 'search':
        case 'semantic_search':
          return await this.semanticSearch(startTime);
        case 'hybrid_search':
          return await this.hybridSearch(startTime);
        case 'index':
          return await this.indexDocuments(startTime, false);
        case 'batch_index':
          return await this.indexDocuments(startTime, true);
        case 'delete':
          return await this.deleteDocuments(startTime);
        case 'embed':
          return await this.embedOnly(startTime);
        case 'health_check':
          return await this.healthCheck(startTime);
        default:
          return {
            success: false,
            operation: this.params.operation,
            backend: this.params.backend,
            error: `Unsupported operation: ${this.params.operation}`,
            timing: Date.now() - startTime,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: this.params.operation,
        backend: this.params.backend,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }

  private resolveOpenAiKey(): string {
    const key =
      this.params.openai_api_key ||
      (typeof process !== 'undefined' && process.env
        ? process.env.OPENAI_API_KEY || ''
        : '');
    if (!key) {
      throw new Error(
        'OpenAI API key is required for embedding generation (openai_api_key or OPENAI_API_KEY).'
      );
    }
    return key;
  }

  private async generateEmbedding(text: string): Promise<number[]> {
    const apiKey = this.resolveOpenAiKey();
    const response = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: this.params.embedding_model,
        input: text,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data?.error?.message || 'Embedding request failed');
    }
    const vector = data?.data?.[0]?.embedding;
    if (!Array.isArray(vector)) {
      throw new Error('Embedding response missing vector');
    }
    return vector as number[];
  }

  private async semanticSearch(startTime: number): Promise<KnowledgeEngineResult> {
    if (this.params.backend === 'elasticsearch') {
      const results = await this.searchElasticsearch();
      return {
        success: true,
        operation: this.params.operation,
        backend: 'elasticsearch',
        results,
        timing: Date.now() - startTime,
      };
    }

    if (this.params.backend === 'bedrock') {
      const results = await this.searchBedrock();
      return {
        success: true,
        operation: this.params.operation,
        backend: 'bedrock',
        results,
        timing: Date.now() - startTime,
      };
    }

    if (this.params.backend === 'eks') {
      const results = await this.searchEks();
      return {
        success: true,
        operation: this.params.operation,
        backend: 'eks',
        results,
        timing: Date.now() - startTime,
      };
    }

    const results = await this.searchQdrant();
    return {
      success: true,
      operation: this.params.operation,
      backend: 'qdrant',
      results,
      timing: Date.now() - startTime,
    };
  }

  private async hybridSearch(startTime: number): Promise<KnowledgeEngineResult> {
    const [semanticResults, keywordResults] = await Promise.all([
      this.searchQdrant(),
      this.searchElasticsearch().catch(() => [] as SearchResult[]),
    ]);

    const combined: SearchResult[] = [];
    const byId = new Map<string, SearchResult>();

    for (const item of semanticResults) {
      const weighted = {
        ...item,
        score: (item.score || 0) * this.params.semantic_weight,
      };
      byId.set(item.id, weighted);
      combined.push(weighted);
    }

    for (const item of keywordResults) {
      const existing = byId.get(item.id);
      if (existing) {
        existing.score = (existing.score || 0) + (item.score || 0) * this.params.keyword_weight;
      } else {
        const weighted = {
          ...item,
          score: (item.score || 0) * this.params.keyword_weight,
        };
        byId.set(item.id, weighted);
        combined.push(weighted);
      }
    }

    combined.sort((a, b) => (b.score || 0) - (a.score || 0));

    return {
      success: true,
      operation: this.params.operation,
      backend: 'hybrid',
      results: combined.slice(0, this.params.limit),
      timing: Date.now() - startTime,
    };
  }

  private async indexDocuments(
    startTime: number,
    isBatch: boolean
  ): Promise<KnowledgeEngineResult> {
    const documents = this.params.documents || [];
    if (documents.length === 0) {
      throw new Error('documents are required for index operations');
    }

    if (this.params.backend === 'elasticsearch') {
      await this.indexElasticsearch(documents, isBatch);
      return {
        success: true,
        operation: this.params.operation,
        backend: 'elasticsearch',
        data: { indexed: documents.length },
        timing: Date.now() - startTime,
      };
    }

    await this.indexQdrant(documents);
    return {
      success: true,
      operation: this.params.operation,
      backend: 'qdrant',
      data: { indexed: documents.length },
      timing: Date.now() - startTime,
    };
  }

  private async deleteDocuments(startTime: number): Promise<KnowledgeEngineResult> {
    const ids = this.params.document_ids || [];
    if (ids.length === 0) {
      throw new Error('document_ids are required for delete');
    }

    if (this.params.backend === 'elasticsearch') {
      await this.deleteElasticsearch(ids);
      return {
        success: true,
        operation: this.params.operation,
        backend: 'elasticsearch',
        data: { deleted: ids.length },
        timing: Date.now() - startTime,
      };
    }

    await this.deleteQdrant(ids);
    return {
      success: true,
      operation: this.params.operation,
      backend: 'qdrant',
      data: { deleted: ids.length },
      timing: Date.now() - startTime,
    };
  }

  private async embedOnly(startTime: number): Promise<KnowledgeEngineResult> {
    if (this.params.query) {
      const vector = await this.generateEmbedding(this.params.query);
      return {
        success: true,
        operation: this.params.operation,
        backend: 'embedding',
        data: { vector },
        timing: Date.now() - startTime,
      };
    }

    if (this.params.documents && this.params.documents.length > 0) {
      const vectors = await Promise.all(
        this.params.documents.map(async (doc) => ({
          id: doc.id,
          vector: await this.generateEmbedding(doc.content),
        }))
      );
      return {
        success: true,
        operation: this.params.operation,
        backend: 'embedding',
        data: { vectors },
        timing: Date.now() - startTime,
      };
    }

    throw new Error('query or documents are required for embed');
  }

  private async healthCheck(startTime: number): Promise<KnowledgeEngineResult> {
    // Health must reflect the real OpenEvolve backend: never fake success.
    // `success` is derived from the HTTP response, and transport failures
    // (unreachable server, timeout/abort) resolve to success: false.
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.params.timeout);
    const url = `${resolveOpenEvolveBaseUrl()}/health`;
    // `backend` carries a zod default, so narrow it to a plain string here.
    const backend = this.params.backend ?? 'qdrant';

    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      const detail = await response.json().catch(() => undefined);

      return {
        success: response.ok,
        operation: this.params.operation,
        backend,
        data: {
          status: response.ok ? 'ok' : 'unhealthy',
          http_status: response.status,
          url,
          detail,
        },
        error: response.ok
          ? undefined
          : `OpenEvolve health check failed: ${response.status} ${response.statusText}`,
        timing: Date.now() - startTime,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: this.params.operation,
        backend,
        data: { status: 'unhealthy', url },
        error: `OpenEvolve server unreachable: ${message}`,
        timing: Date.now() - startTime,
      };
    }
  }

  private getQdrantConfig() {
    if (!this.params.qdrant) {
      throw new Error('qdrant config is required');
    }
    return this.params.qdrant;
  }

  private async searchQdrant(): Promise<SearchResult[]> {
    const config = this.getQdrantConfig();
    const vector =
      this.params.query_vector ||
      (this.params.query ? await this.generateEmbedding(this.params.query) : undefined);

    if (!vector) {
      throw new Error('query_vector or query is required for semantic search');
    }

    const response = await fetch(
      `${config.base_url.replace(/\/$/, '')}/collections/${config.collection_name}/points/search`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(config.api_key ? { 'api-key': config.api_key } : {}),
        },
        body: JSON.stringify({
          vector,
          limit: this.params.limit,
          filter: this.params.filters,
          with_payload: true,
          with_vector: false,
        }),
      }
    );

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data?.status?.error || 'Qdrant search failed');
    }

    const points = data?.result || [];
    return points.map((point: any) => ({
      id: String(point.id),
      content: point.payload?.content || '',
      score: point.score,
      metadata: point.payload,
    }));
  }

  private async indexQdrant(documents: KnowledgeDocument[]): Promise<void> {
    const config = this.getQdrantConfig();
    const points = await Promise.all(
      documents.map(async (doc) => ({
        id: doc.id,
        vector: await this.generateEmbedding(doc.content),
        payload: { content: doc.content, ...doc.metadata },
      }))
    );

    const response = await fetch(
      `${config.base_url.replace(/\/$/, '')}/collections/${config.collection_name}/points?wait=true`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(config.api_key ? { 'api-key': config.api_key } : {}),
        },
        body: JSON.stringify({ points }),
      }
    );

    if (!response.ok) {
      const data = await response.json();
      throw new Error(data?.status?.error || 'Qdrant index failed');
    }
  }

  private async deleteQdrant(ids: string[]): Promise<void> {
    const config = this.getQdrantConfig();
    const response = await fetch(
      `${config.base_url.replace(/\/$/, '')}/collections/${config.collection_name}/points/delete`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(config.api_key ? { 'api-key': config.api_key } : {}),
        },
        body: JSON.stringify({ points: ids }),
      }
    );

    if (!response.ok) {
      const data = await response.json();
      throw new Error(data?.status?.error || 'Qdrant delete failed');
    }
  }

  private getElasticsearchConfig() {
    if (!this.params.elasticsearch) {
      throw new Error('elasticsearch config is required');
    }
    return this.params.elasticsearch;
  }

  private buildElasticsearchHeaders(config: {
    username?: string;
    password?: string;
  }): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (config.username && config.password) {
      const raw = `${config.username}:${config.password}`;
      const token =
        typeof Buffer !== 'undefined'
          ? Buffer.from(raw).toString('base64')
          : btoa(raw);
      headers.Authorization = `Basic ${token}`;
    }

    return headers;
  }

  private async searchElasticsearch(): Promise<SearchResult[]> {
    const config = this.getElasticsearchConfig();
    if (!this.params.query) {
      throw new Error('query is required for elasticsearch search');
    }

    const response = await fetch(
      `${config.base_url.replace(/\/$/, '')}/${config.index}/_search`,
      {
        method: 'POST',
        headers: this.buildElasticsearchHeaders(config),
        body: JSON.stringify({
          query: { match: { content: this.params.query } },
          size: this.params.limit,
        }),
      }
    );

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data?.error?.reason || 'Elasticsearch search failed');
    }

    const hits = data?.hits?.hits || [];
    return hits.map((hit: any) => ({
      id: String(hit._id),
      content: hit._source?.content || '',
      score: hit._score,
      metadata: hit._source,
    }));
  }

  private async indexElasticsearch(documents: KnowledgeDocument[], isBatch: boolean): Promise<void> {
    const config = this.getElasticsearchConfig();
    if (isBatch) {
      // Simple batch with sequential requests to avoid huge payloads
      for (const doc of documents) {
        await this.indexElasticsearch([doc], false);
      }
      return;
    }

    const doc = documents[0];
    const response = await fetch(
      `${config.base_url.replace(/\/$/, '')}/${config.index}/_doc/${doc.id}`,
      {
        method: 'PUT',
        headers: this.buildElasticsearchHeaders(config),
        body: JSON.stringify({ content: doc.content, ...doc.metadata }),
      }
    );

    if (!response.ok) {
      const data = await response.json();
      throw new Error(data?.error?.reason || 'Elasticsearch index failed');
    }
  }

  private async deleteElasticsearch(ids: string[]): Promise<void> {
    const config = this.getElasticsearchConfig();
    for (const id of ids) {
      const response = await fetch(
        `${config.base_url.replace(/\/$/, '')}/${config.index}/_doc/${id}`,
        {
          method: 'DELETE',
          headers: this.buildElasticsearchHeaders(config),
        }
      );

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data?.error?.reason || 'Elasticsearch delete failed');
      }
    }
  }

  private async searchBedrock(): Promise<SearchResult[]> {
    if (!this.params.bedrock) {
      throw new Error('bedrock config is required');
    }
    if (!this.params.query) {
      throw new Error('query is required for bedrock search');
    }

    const response = await fetch('https://bedrock-runtime.amazonaws.com', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Amz-Target': 'AWSBedrockAgentRuntimeService.Retrieve',
      },
      body: JSON.stringify({
        knowledgeBaseId: this.params.bedrock.knowledge_base_id,
        retrievalQuery: { text: this.params.query },
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data?.message || 'Bedrock search failed');
    }

    return (data.retrievalResults || []).map((item: any) => ({
      id: item.metadata?.uri || item.content?.text || 'unknown',
      content: item.content?.text || '',
      score: item.score || 0,
      metadata: item.metadata,
    }));
  }

  private async searchEks(): Promise<SearchResult[]> {
    if (!this.params.eks) {
      throw new Error('eks config is required');
    }
    if (!this.params.query) {
      throw new Error('query is required for eks search');
    }

    const response = await fetch(
      `${this.params.eks.endpoint.replace(/\/$/, '')}/api/knowledge/search`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: this.params.query,
          limit: this.params.limit,
          filters: this.params.filters,
        }),
      }
    );

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data?.error || 'EKS search failed');
    }

    return (data.results || []).map((item: any) => ({
      id: String(item.id ?? item.document_id ?? ''),
      content: item.content || '',
      score: item.score,
      metadata: item.metadata,
    }));
  }
}

export default OpenEvolveKnowledgeEngineBubble;
