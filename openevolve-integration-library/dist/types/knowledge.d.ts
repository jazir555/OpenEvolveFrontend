import { ExecutionConfig } from './common';
export interface KnowledgeInputs {
    operation: 'query' | 'extract' | 'search' | 'stats';
    input: GraphQuery | ExtractionInput | SearchInput;
    config?: ExecutionConfig;
}
export interface GraphQuery {
    type: 'sparql' | 'cypher' | 'gremlin' | 'natural';
    query: string;
    parameters?: Record<string, any>;
    format?: 'json' | 'xml' | 'csv';
}
export interface ExtractionInput {
    document: string;
    documentType: 'text' | 'pdf' | 'html' | 'markdown';
    strategy?: 'ner' | 're' | 'ie' | 'custom';
}
export interface SearchInput {
    query: string;
    type: 'keyword' | 'semantic' | 'hybrid';
    maxResults?: number;
    filters?: SearchFilter[];
    sort?: 'relevance' | 'date' | 'citations';
}
export interface SearchFilter {
    field: string;
    operator: 'eq' | 'ne' | 'gt' | 'lt' | 'contains' | 'matches';
    value: any;
}
export interface GraphResult {
    results: any[];
    count: number;
    executionTime: number;
    metadata?: GraphResultMetadata;
}
export interface GraphResultMetadata {
    queryPlan?: string;
}
export interface ExtractionResult {
    entities: ExtractedEntity[];
    relationships: ExtractedRelationship[];
    confidence: number;
    metadata: ExtractionMetadata;
}
export interface ExtractedEntity {
    id: string;
    type: string;
    text: string;
    confidence: number;
    properties?: Record<string, any>;
    position?: {
        start: number;
        end: number;
    };
}
export interface ExtractedRelationship {
    id: string;
    source: string;
    target: string;
    type: string;
    confidence: number;
    properties?: Record<string, any>;
}
export interface ExtractionMetadata {
    documentId: string;
    extractionTime: number;
}
export interface SearchResult {
    results: SearchItem[];
    total: number;
    executionTime: number;
    metadata: SearchMetadata;
}
export interface SearchItem {
    id: string;
    type: 'entity' | 'relation' | 'triple' | 'document';
    content: any;
    score: number;
    highlight?: string[];
}
export interface SearchMetadata {
    query: string;
    searchType: string;
    indexName: string;
    filtersApplied: string[];
}
export interface GraphStats {
    nodes: NodeStats;
    relationships: RelationshipStats;
    storage: StorageStats;
    queries: QueryStats;
}
export interface NodeStats {
    total: number;
    byType: Record<string, number>;
    byLabel: Record<string, number>;
}
export interface RelationshipStats {
    total: number;
    byType: Record<string, number>;
}
export interface StorageStats {
    totalSize: number;
    nodesSize: number;
    relationshipsSize: number;
    indexesSize: number;
}
export interface QueryStats {
    totalQueries: number;
    averageQueryTime: number;
    byType: Record<string, number>;
}
export interface KnowledgeResult {
    type: 'query' | 'extract' | 'search' | 'stats';
    result: GraphResult | ExtractionResult | SearchResult | GraphStats;
    metadata: {
        executionTime: number;
        timestamp: string;
        apiVersion: string;
    };
}
//# sourceMappingURL=knowledge.d.ts.map