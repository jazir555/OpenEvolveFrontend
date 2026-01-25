import QdrantIntegration from './qdrant';
import ElasticsearchIntegration from './elasticsearch';

type VectorDBType = 'qdrant' | 'elasticsearch';

interface VectorDBConfig {
  type: VectorDBType;
  url: string;
  apiKey?: string;
  collectionName?: string;
  index?: string;
}

class VectorDBService {
  private db: QdrantIntegration | ElasticsearchIntegration;
  private type: VectorDBType;

  constructor(config: VectorDBConfig) {
    this.type = config.type;
    
    if (config.type === 'qdrant') {
      this.db = new QdrantIntegration({
        url: config.url,
        apiKey: config.apiKey,
        collectionName: config.collectionName || 'knowledge_base'
      });
    } else {
      this.db = new ElasticsearchIntegration({
        node: config.url,
        auth: config.apiKey ? { username: 'elastic', password: config.apiKey } : undefined,
        index: config.index || 'knowledge_base'
      });
    }
  }

  async search(
    query: string,
    limit: number = 10,
    filters?: Record<string, any>
  ): Promise<any[]> {
    if (this.type === 'qdrant') {
      // For Qdrant, we need to generate embeddings first
      const { generateEmbeddings } = await import('../utils/embeddings');
      const embeddings = await generateEmbeddings(query);
      return await (this.db as QdrantIntegration).search(embeddings[0], limit, filters);
    } else {
      return await (this.db as ElasticsearchIntegration).search(query, limit, filters);
    }
  }

  async store(
    id: string,
    content: string,
    metadata: Record<string, any>,
    embedding?: number[]
  ): Promise<void> {
    if (this.type === 'qdrant') {
      if (!embedding) {
        const { generateEmbeddings } = await import('../utils/embeddings');
        const embeddings = await generateEmbeddings(content);
        embedding = embeddings[0];
      }
      await (this.db as QdrantIntegration).storeEmbedding(id, embedding, { content, ...metadata });
    } else {
      await (this.db as ElasticsearchIntegration).indexDocument(id, content, { content, ...metadata });
    }
  }
}

export default VectorDBService;
export { VectorDBService, type VectorDBType, type VectorDBConfig };