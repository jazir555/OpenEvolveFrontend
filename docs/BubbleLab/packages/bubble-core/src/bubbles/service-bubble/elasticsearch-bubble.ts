import { ServiceBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

export class ElasticsearchBubble extends ServiceBubble<ElasticsearchParams, ElasticsearchResult> {
  bubbleName = 'elasticsearch';
  type = 'service';
  alias = 'Elasticsearch';
  credentialType = 'elasticsearch_api_key';

  params = {
    apiKey: z.string().min(1),
    baseUrl: z.string().url(),
    timeout: z.number().int().positive().default(30000)
  };

  private client: any = null;

  async connect() {
    const { Client } = await import('@elastic/elasticsearch');
    this.client = new Client({
      node: this.params.baseUrl,
      auth: {
        apiKey: this.params.apiKey
      }
    });
  }

  async createIndex(params: { name: string; mappings?: any }): Promise<ElasticsearchResult> {
    try {
      await this.client.indices.create({ index: params.name, body: { mappings: params.mappings } });
      return { success: true, index: params.name };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async indexDocument(params: { index: string; id: string; document: any }): Promise<ElasticsearchResult> {
    try {
      await this.client.index({ index: params.index, id: params.id, body: params.document });
      return { success: true };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async search(params: { index: string; query: any }): Promise<ElasticsearchResult> {
    try {
      const response = await this.client.search({ index: params.index, body: params.query });
      return { success: true, results: response.hits };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async deleteIndex(params: { index: string }): Promise<ElasticsearchResult> {
    try {
      await this.client.indices.delete({ index: params.index });
      return { success: true, index: params.index };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface ElasticsearchParams {
  apiKey: string;
  baseUrl: string;
  timeout?: number;
}

export interface ElasticsearchResult {
  success: boolean;
  index?: string;
  results?: any;
  error?: string;
}
