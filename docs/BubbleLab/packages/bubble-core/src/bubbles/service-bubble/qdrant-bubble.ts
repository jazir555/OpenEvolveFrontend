import { ServiceBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * QdrantBubble - Vector database operations for similarity search
 */
export class QdrantBubble extends ServiceBubble<QdrantParams, QdrantResult> {
  bubbleName = 'qdrant';
  type = 'service';
  alias = 'Vector Database';
  schema = z.object({
    apiKey: z.string().min(1),
    baseUrl: z.string().url(),
    collection: z.string().optional()
  });
  resultSchema = z.object({
    success: z.boolean(),
    points: z.array(z.any()).optional(),
    error: z.string().optional()
  });
  descriptions = {
    createCollection: 'Create a new vector collection',
    insertPoints: 'Insert vectors into collection',
    searchPoints: 'Search for similar vectors',
    deletePoints: 'Delete vectors by ID'
  };

  credentialType = 'qdrant_api_key';

  params = {
    apiKey: z.string().min(1),
    baseUrl: z.string().url(),
    timeout: z.number().int().positive().default(30000)
  };

  private client: any = null;

  async connect() {
    const { QdrantClient } = await import('@qdrant/js-client-rest');
    this.client = new QdrantClient({
      url: this.params.baseUrl,
      apiKey: this.params.apiKey,
      timeout: this.params.timeout
    });
  }

  async createCollection(params: {
    name: string;
    vectorSize: number;
    distance?: 'Cosine' | 'Euclid' | 'Dot';
  }): Promise<QdrantResult> {
    try {
      await this.client.createCollection(params.name, {
        vectors: {
          size: params.vectorSize,
          distance: params.distance || 'Cosine'
        }
      });
      return { success: true, collection: params.name };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async insertPoints(params: {
    collection: string;
    points: Array<{ id: string | number; vector: number[]; payload?: any }>;
  }): Promise<QdrantResult> {
    try {
      await this.client.upsert(params.collection, {
        points: params.points
      });
      return { success: true, inserted: params.points.length };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async searchPoints(params: {
    collection: string;
    vector: number[];
    limit: number;
    scoreThreshold?: number;
  }): Promise<QdrantResult> {
    try {
      const response = await this.client.search(params.collection, {
        vector: params.vector,
        limit: params.limit,
        score_threshold: params.scoreThreshold
      });
      return { success: true, points: response.result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async deletePoints(params: {
    collection: string;
    ids: Array<string | number>;
  }): Promise<QdrantResult> {
    try {
      await this.client.delete(params.collection, {
        points: params.ids
      });
      return { success: true, deleted: params.ids.length };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getCollection(params: { collection: string }): Promise<QdrantResult> {
    try {
      const collection = await this.client.getCollection(params.collection);
      return { success: true, collection };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async deleteCollection(params: { collection: string }): Promise<QdrantResult> {
    try {
      await this.client.deleteCollection(params.collection);
      return { success: true, collection: params.collection };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async updatePayload(params: {
    collection: string;
    id: string | number;
    payload: any;
  }): Promise<QdrantResult> {
    try {
      await this.client.setPayload(params.collection, {
        payload: params.payload,
        points: [params.id]
      });
      return { success: true };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async scroll(params: {
    collection: string;
    limit: number;
    offset?: number;
  }): Promise<QdrantResult> {
    try {
      const response = await this.client.scroll(params.collection, {
        limit: params.limit,
        offset: params.offset
      });
      return { success: true, points: response.result.points };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async count(params: { collection: string }): Promise<QdrantResult> {
    try {
      const collection = await this.client.getCollection(params.collection);
      return { success: true, count: collection.result.points_count };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async recreateCollection(params: {
    name: string;
    vectorSize: number;
  }): Promise<QdrantResult> {
    try {
      await this.client.recreateCollection(params.name, {
        vectors: { size: params.vectorSize }
      });
      return { success: true, collection: params.name };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface QdrantParams {
  apiKey: string;
  baseUrl: string;
  timeout?: number;
}

export interface QdrantResult {
  success: boolean;
  collection?: string;
  points?: any[];
  inserted?: number;
  deleted?: number;
  count?: number;
  error?: string;
}
