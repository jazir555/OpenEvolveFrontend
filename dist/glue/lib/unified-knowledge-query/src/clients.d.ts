/**
 * Individual System Clients for RAGBits, Graphiti, and Vector DB
 *
 * Federation Constitution Compliance:
 * - Circuit Breakers for each client
 * - Retry logic with exponential backoff
 * - Timeout enforcement on all requests
 * - Structured logging with correlation IDs
 */
import { KnowledgeItem, Entity, Relationship, SystemConfig } from './canonical';
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
export declare class RAGBitsClient implements KnowledgeClient {
    private client;
    private circuitBreaker;
    private logger;
    private config;
    constructor(config: SystemConfig);
    /**
     * Search RAGBits for relevant documents
     */
    search(query: string, options?: any): Promise<KnowledgeItem[]>;
    /**
     * Ingest document into RAGBits
     */
    ingest(document: any): Promise<string>;
    /**
     * Get RAGBits statistics
     */
    getStats(): Promise<any>;
    /**
     * Health check for RAGBits
     */
    healthCheck(): Promise<boolean>;
    private generateCorrelationId;
}
/**
 * Graphiti Client
 *
 * Connects to Graphiti for temporal knowledge graph queries
 */
export declare class GraphitiClient implements KnowledgeClient {
    private client;
    private circuitBreaker;
    private logger;
    private config;
    constructor(config: SystemConfig);
    /**
     * Search Graphiti for entities and relationships
     */
    search(query: string, options?: any): Promise<KnowledgeItem[]>;
    /**
     * Temporal query with time filters
     */
    temporalQuery(query: string, startDate: string, endDate: string, options?: any): Promise<KnowledgeItem[]>;
    /**
     * Get entities from Graphiti
     */
    getEntities(options?: any): Promise<Entity[]>;
    /**
     * Get relationships from Graphiti
     */
    getRelationships(options?: any): Promise<Relationship[]>;
    /**
     * Get Graphiti statistics
     */
    getStats(): Promise<any>;
    /**
     * Health check for Graphiti
     */
    healthCheck(): Promise<boolean>;
    private generateCorrelationId;
}
/**
 * Vector DB Client
 *
 * Connects to Vector DB for high-performance semantic search
 */
export declare class VectorDBClient implements KnowledgeClient {
    private client;
    private circuitBreaker;
    private logger;
    private config;
    constructor(config: SystemConfig);
    /**
     * Semantic vector search
     */
    search(query: string, options?: any): Promise<KnowledgeItem[]>;
    /**
     * Upsert vectors into collection
     */
    upsert(collection: string, vectors: any[]): Promise<void>;
    /**
     * Get collection information
     */
    getCollectionInfo(collection: string): Promise<any>;
    /**
     * Get VectorDB statistics
     */
    getStats(): Promise<any>;
    /**
     * Health check for VectorDB
     */
    healthCheck(): Promise<boolean>;
    private generateCorrelationId;
}
export {};
//# sourceMappingURL=clients.d.ts.map