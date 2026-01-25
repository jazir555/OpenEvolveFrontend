import { QdrantClient } from '@qdrant/qdrant-js';

class QdrantIntegration {
  private client: QdrantClient;
  private collectionName: string;

  constructor(config: { url: string; apiKey?: string; collectionName: string }) {
    this.client = new QdrantClient({
      url: config.url,
      apiKey: config.apiKey,
    });
    this.collectionName = config.collectionName;
  }

  async search(
    vector: number[],
    limit: number = 10,
    filters?: Record<string, any>
  ): Promise<any[]> {
    const searchResponse = await this.client.search(this.collectionName, {
      vector,
      limit,
      filter: filters,
    });

    return searchResponse;
  }

  async storeEmbedding(
    id: string,
    vector: number[],
    payload: Record<string, any>
  ): Promise<void> {
    await this.client.upsert(this.collectionName, {
      points: [
        {
          id,
          vector,
          payload,
        },
      ],
    });
  }
}

export default QdrantIntegration;