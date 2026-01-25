import { Client } from '@elastic/elasticsearch';

class ElasticsearchIntegration {
  private client: Client;

  constructor(config: { node: string; auth?: { username: string; password: string }; index: string }) {
    this.client = new Client({
      node: config.node,
      auth: config.auth,
    });
    this.index = config.index;
  }

  async search(
    query: string,
    limit: number = 10,
    filters?: Record<string, any>
  ): Promise<any[]> {
    const searchResponse = await this.client.search({
      index: this.index,
      body: {
        query: {
          bool: {
            should: [
              { match: { content: query } },
              { match_phrase: { content: query } }
            ],
            filter: filters || []
          }
        },
        size: limit
      }
    });

    return searchResponse.body.hits.hits;
  }

  async indexDocument(
    id: string,
    content: string,
    metadata: Record<string, any>
  ): Promise<void> {
    await this.client.index({
      index: this.index,
      id,
      body: {
        content,
        ...metadata
      }
    });
  }
}

export default ElasticsearchIntegration;