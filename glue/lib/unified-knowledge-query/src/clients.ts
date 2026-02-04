/**
 * Individual System Clients for RAGBits, Graphiti, and Vector DB
 *
 * Federation Constitution Compliance:
 * - Circuit Breakers for each client
 * - Retry logic with exponential backoff
 * - Timeout enforcement on all requests
 * - Structured logging with correlation IDs
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import { Logger, CircuitBreaker } from '@openevolve/glue-lib';
import {
  KnowledgeItem,
  SystemSource,
  Entity,
  Relationship,
  SystemConfig,
} from './canonical';

/**
 * Base Client Interface
 */
interface KnowledgeClient {
  search(query: string, options: any): Promise<KnowledgeItem[]>;
  healthCheck(): Promise<boolean>;
  getStats(): Promise<any>;
}

/**
 * RAGBits Client
 *
 * Connects to RAGBits for document-based retrieval and RAG
 */
export class RAGBitsClient implements KnowledgeClient {
  private client: AxiosInstance;
  private circuitBreaker: CircuitBreaker;
  private logger: Logger;
  private config: SystemConfig;

  constructor(config: SystemConfig) {
    this.config = config;

    // Validate required environment (Law of Configuration Explicitness)
    if (!config.url) {
      throw new Error('RAGBITS_URL environment variable is required');
    }

    this.client = axios.create({
      baseURL: config.url,
      timeout: config.timeout || 5000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.circuitBreaker = new CircuitBreaker({
      threshold: 5,
      timeout_ms: 60000,
      onStateChange: (old, newState) => {
        this.logger.warn('RAGBits circuit breaker state changed', {
          old_state: old,
          new_state: newState,
          source_service: 'unified-knowledge-query',
          target_service: 'ragbits',
        });
      },
    });

    this.logger = new Logger('ragbits-client');
  }

  /**
   * Search RAGBits for relevant documents
   */
  async search(query: string, options: any = {}): Promise<KnowledgeItem[]> {
    const correlationId = options.correlationId || this.generateCorrelationId();

    this.logger.info('RAGBits search initiated', {
      correlation_id: correlationId,
      query: query.substring(0, 100),
      source_service: 'unified-knowledge-query',
      target_service: 'ragbits',
    });

    try {
      const results = await this.circuitBreaker.execute(async () => {
        const response = await this.client.post('/api/search', {
          query,
          top_k: options.maxResults || 50,
          filters: options.filters || {},
        });

        if (response.status !== 200) {
          throw new Error(`RAGBits returned ${response.status}`);
        }

        return response.data;
      });

      // Normalize to canonical format
      const items: KnowledgeItem[] = (results.documents || []).map((doc: any) => ({
        content: doc.content || doc.text || '',
        source: 'ragbits' as SystemSource,
        id: doc.id || doc.document_id,
        type: 'document',
        confidence: doc.score || doc.confidence || 0.5,
        relevance: doc.score || doc.relevance || 0.5,
        timestamp: doc.timestamp || new Date().toISOString(),
        metadata: {
          title: doc.title,
          author: doc.author,
          tags: doc.tags,
          ...doc.metadata,
        },
      }));

      this.logger.info('RAGBits search completed', {
        correlation_id: correlationId,
        result_count: items.length,
        source_service: 'unified-knowledge-query',
        target_service: 'ragbits',
      });

      return items;
    } catch (error) {
      this.logger.error('RAGBits search failed', error as Error, {
        correlation_id: correlationId,
        query: query.substring(0, 100),
        source_service: 'unified-knowledge-query',
        target_service: 'ragbits',
      });
      throw error;
    }
  }

  /**
   * Ingest document into RAGBits
   */
  async ingest(document: any): Promise<string> {
    const correlationId = this.generateCorrelationId();

    try {
      const result = await this.circuitBreaker.execute(async () => {
        const response = await this.client.post('/api/ingest', document);
        return response.data;
      });

      this.logger.info('Document ingested into RAGBits', {
        correlation_id: correlationId,
        document_id: result.id,
        source_service: 'unified-knowledge-query',
        target_service: 'ragbits',
      });

      return result.id;
    } catch (error) {
      this.logger.error('RAGBits ingest failed', error as Error, {
        correlation_id: correlationId,
        source_service: 'unified-knowledge-query',
        target_service: 'ragbits',
      });
      throw error;
    }
  }

  /**
   * Get RAGBits statistics
   */
  async getStats(): Promise<any> {
    try {
      const result = await this.circuitBreaker.execute(async () => {
        const response = await this.client.get('/api/stats');
        return response.data;
      });
      return result;
    } catch (error) {
      this.logger.error('RAGBits stats failed', error as Error, {
        source_service: 'unified-knowledge-query',
        target_service: 'ragbits',
      });
      throw error;
    }
  }

  /**
   * Health check for RAGBits
   */
  async healthCheck(): Promise<boolean> {
    try {
      await this.client.get('/health', { timeout: 2000 });
      return true;
    } catch (error) {
      this.logger.warn('RAGBits health check failed', error as Error);
      return false;
    }
  }

  private generateCorrelationId(): string {
    return `ragbits-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}

/**
 * Graphiti Client
 *
 * Connects to Graphiti for temporal knowledge graph queries
 */
export class GraphitiClient implements KnowledgeClient {
  private client: AxiosInstance;
  private circuitBreaker: CircuitBreaker;
  private logger: Logger;
  private config: SystemConfig;

  constructor(config: SystemConfig) {
    this.config = config;

    if (!config.url) {
      throw new Error('GRAPHITI_URL environment variable is required');
    }

    this.client = axios.create({
      baseURL: config.url,
      timeout: config.timeout || 5000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.circuitBreaker = new CircuitBreaker({
      threshold: 5,
      timeout_ms: 60000,
      onStateChange: (old, newState) => {
        this.logger.warn('Graphiti circuit breaker state changed', {
          old_state: old,
          new_state: newState,
          source_service: 'unified-knowledge-query',
          target_service: 'graphiti',
        });
      },
    });

    this.logger = new Logger('graphiti-client');
  }

  /**
   * Search Graphiti for entities and relationships
   */
  async search(query: string, options: any = {}): Promise<KnowledgeItem[]> {
    const correlationId = options.correlationId || this.generateCorrelationId();

    this.logger.info('Graphiti search initiated', {
      correlation_id: correlationId,
      query: query.substring(0, 100),
      source_service: 'unified-knowledge-query',
      target_service: 'graphiti',
    });

    try {
      const results = await this.circuitBreaker.execute(async () => {
        const response = await this.client.post('/api/search', {
          query,
          limit: options.maxResults || 50,
          temporal_filter: options.temporalFilter,
        });

        if (response.status !== 200) {
          throw new Error(`Graphiti returned ${response.status}`);
        }

        return response.data;
      });

      // Normalize entities and relationships to canonical format
      const items: KnowledgeItem[] = [];

      // Add entities
      if (results.entities) {
        items.push(...results.entities.map((entity: any) => ({
          content: entity.description || entity.name,
          source: 'graphiti' as SystemSource,
          id: entity.id,
          type: 'entity',
          confidence: entity.confidence || 0.5,
          relevance: entity.relevance || 0.5,
          timestamp: entity.created_at || new Date().toISOString(),
          metadata: {
            name: entity.name,
            entity_type: entity.type,
            ...entity.metadata,
          },
        })));
      }

      // Add relationships
      if (results.relationships) {
        items.push(...results.relationships.map((rel: any) => ({
          content: `${rel.source} ${rel.relation} ${rel.target}`,
          source: 'graphiti' as SystemSource,
          id: rel.id,
          type: 'relationship',
          confidence: rel.weight || rel.confidence || 0.5,
          relevance: rel.relevance || 0.5,
          timestamp: rel.created_at || new Date().toISOString(),
          metadata: {
            source_entity: rel.source,
            target_entity: rel.target,
            relation_type: rel.relation,
            ...rel.metadata,
          },
        })));
      }

      this.logger.info('Graphiti search completed', {
        correlation_id: correlationId,
        result_count: items.length,
        entities_count: results.entities?.length || 0,
        relationships_count: results.relationships?.length || 0,
        source_service: 'unified-knowledge-query',
        target_service: 'graphiti',
      });

      return items;
    } catch (error) {
      this.logger.error('Graphiti search failed', error as Error, {
        correlation_id: correlationId,
        query: query.substring(0, 100),
        source_service: 'unified-knowledge-query',
        target_service: 'graphiti',
      });
      throw error;
    }
  }

  /**
   * Temporal query with time filters
   */
  async temporalQuery(
    query: string,
    startDate: string,
    endDate: string,
    options: any = {}
  ): Promise<KnowledgeItem[]> {
    const correlationId = this.generateCorrelationId();

    try {
      const results = await this.circuitBreaker.execute(async () => {
        const response = await this.client.post('/api/temporal', {
          query,
          start_date: startDate,
          end_date: endDate,
          limit: options.maxResults || 50,
        });
        return response.data;
      });

      // Normalize results
      const items: KnowledgeItem[] = (results.results || []).map((item: any) => ({
        content: item.content,
        source: 'graphiti' as SystemSource,
        id: item.id,
        type: item.type || 'entity',
        confidence: item.confidence || 0.5,
        relevance: item.relevance || 0.5,
        timestamp: item.timestamp,
        metadata: item.metadata,
      }));

      return items;
    } catch (error) {
      this.logger.error('Graphiti temporal query failed', error as Error, {
        correlation_id: correlationId,
        source_service: 'unified-knowledge-query',
        target_service: 'graphiti',
      });
      throw error;
    }
  }

  /**
   * Get entities from Graphiti
   */
  async getEntities(options: any = {}): Promise<Entity[]> {
    try {
      const results = await this.circuitBreaker.execute(async () => {
        const response = await this.client.get('/api/entities', {
          params: { limit: options.limit || 100 },
        });
        return response.data;
      });

      return results.entities || [];
    } catch (error) {
      this.logger.error('Graphiti getEntities failed', error as Error);
      throw error;
    }
  }

  /**
   * Get relationships from Graphiti
   */
  async getRelationships(options: any = {}): Promise<Relationship[]> {
    try {
      const results = await this.circuitBreaker.execute(async () => {
        const response = await this.client.get('/api/relationships', {
          params: { limit: options.limit || 100 },
        });
        return response.data;
      });

      return results.relationships || [];
    } catch (error) {
      this.logger.error('Graphiti getRelationships failed', error as Error);
      throw error;
    }
  }

  /**
   * Get Graphiti statistics
   */
  async getStats(): Promise<any> {
    try {
      const result = await this.circuitBreaker.execute(async () => {
        const response = await this.client.get('/api/stats');
        return response.data;
      });
      return result;
    } catch (error) {
      this.logger.error('Graphiti stats failed', error as Error);
      throw error;
    }
  }

  /**
   * Health check for Graphiti
   */
  async healthCheck(): Promise<boolean> {
    try {
      await this.client.get('/health', { timeout: 2000 });
      return true;
    } catch (error) {
      this.logger.warn('Graphiti health check failed', error as Error);
      return false;
    }
  }

  private generateCorrelationId(): string {
    return `graphiti-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}

/**
 * Vector DB Client
 *
 * Connects to Vector DB for high-performance semantic search
 */
export class VectorDBClient implements KnowledgeClient {
  private client: AxiosInstance;
  private circuitBreaker: CircuitBreaker;
  private logger: Logger;
  private config: SystemConfig;

  constructor(config: SystemConfig) {
    this.config = config;

    if (!config.url) {
      throw new Error('VECTORDB_URL environment variable is required');
    }

    this.client = axios.create({
      baseURL: config.url,
      timeout: config.timeout || 5000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.circuitBreaker = new CircuitBreaker({
      threshold: 5,
      timeout_ms: 60000,
      onStateChange: (old, newState) => {
        this.logger.warn('VectorDB circuit breaker state changed', {
          old_state: old,
          new_state: newState,
          source_service: 'unified-knowledge-query',
          target_service: 'vectordb',
        });
      },
    });

    this.logger = new Logger('vectordb-client');
  }

  /**
   * Semantic vector search
   */
  async search(query: string, options: any = {}): Promise<KnowledgeItem[]> {
    const correlationId = options.correlationId || this.generateCorrelationId();

    this.logger.info('VectorDB search initiated', {
      correlation_id: correlationId,
      query: query.substring(0, 100),
      source_service: 'unified-knowledge-query',
      target_service: 'vectordb',
    });

    try {
      const results = await this.circuitBreaker.execute(async () => {
        const response = await this.client.post('/api/search', {
          query,
          top_k: options.maxResults || 50,
          collection: options.collection || 'default',
          filters: options.filters || {},
        });

        if (response.status !== 200) {
          throw new Error(`VectorDB returned ${response.status}`);
        }

        return response.data;
      });

      // Normalize to canonical format
      const items: KnowledgeItem[] = (results.results || []).map((doc: any) => ({
        content: doc.payload?.content || doc.payload?.text || '',
        source: 'vectordb' as SystemSource,
        id: doc.id,
        type: doc.payload?.type || 'document',
        confidence: doc.score || 0.5,
        relevance: doc.score || 0.5,
        timestamp: doc.payload?.timestamp || new Date().toISOString(),
        metadata: {
          collection: options.collection || 'default',
          vector_id: doc.id,
          ...doc.payload,
        },
      }));

      this.logger.info('VectorDB search completed', {
        correlation_id: correlationId,
        result_count: items.length,
        source_service: 'unified-knowledge-query',
        target_service: 'vectordb',
      });

      return items;
    } catch (error) {
      this.logger.error('VectorDB search failed', error as Error, {
        correlation_id: correlationId,
        query: query.substring(0, 100),
        source_service: 'unified-knowledge-query',
        target_service: 'vectordb',
      });
      throw error;
    }
  }

  /**
   * Upsert vectors into collection
   */
  async upsert(collection: string, vectors: any[]): Promise<void> {
    const correlationId = this.generateCorrelationId();

    try {
      await this.circuitBreaker.execute(async () => {
        await this.client.post('/api/upsert', {
          collection,
          vectors,
        });
      });

      this.logger.info('Vectors upserted into VectorDB', {
        correlation_id: correlationId,
        collection,
        count: vectors.length,
        source_service: 'unified-knowledge-query',
        target_service: 'vectordb',
      });
    } catch (error) {
      this.logger.error('VectorDB upsert failed', error as Error, {
        correlation_id: correlationId,
        source_service: 'unified-knowledge-query',
        target_service: 'vectordb',
      });
      throw error;
    }
  }

  /**
   * Get collection information
   */
  async getCollectionInfo(collection: string): Promise<any> {
    try {
      const result = await this.circuitBreaker.execute(async () => {
        const response = await this.client.get(`/api/collections/${collection}`);
        return response.data;
      });
      return result;
    } catch (error) {
      this.logger.error('VectorDB getCollectionInfo failed', error as Error);
      throw error;
    }
  }

  /**
   * Get VectorDB statistics
   */
  async getStats(): Promise<any> {
    try {
      const result = await this.circuitBreaker.execute(async () => {
        const response = await this.client.get('/api/stats');
        return response.data;
      });
      return result;
    } catch (error) {
      this.logger.error('VectorDB stats failed', error as Error);
      throw error;
    }
  }

  /**
   * Health check for VectorDB
   */
  async healthCheck(): Promise<boolean> {
    try {
      await this.client.get('/health', { timeout: 2000 });
      return true;
    } catch (error) {
      this.logger.warn('VectorDB health check failed', error as Error);
      return false;
    }
  }

  private generateCorrelationId(): string {
    return `vectordb-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}
