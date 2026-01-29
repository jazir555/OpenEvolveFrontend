import { KnowledgeArtifact, KnowledgeGraph } from '../stores/knowledgeStore';
/**
 * Knowledge query parameters
 */
export interface KnowledgeQueryParams {
    query: string;
    context?: string;
    limit?: number;
    threshold?: number;
}
/**
 * Knowledge query result
 */
export interface KnowledgeQueryResult {
    artifact_id: string;
    relevance_score: number;
    artifact: KnowledgeArtifact;
    matched_sections: Array<{
        content: string;
        score: number;
    }>;
}
/**
 * Knowledge ingestion parameters
 */
export interface KnowledgeIngestParams {
    content: string;
    title: string;
    language?: string;
    tags?: string[];
    metadata?: Record<string, any>;
}
/**
 * Knowledge graph query
 */
export interface GraphQueryParams {
    source?: string;
    relation_type?: string;
    depth?: number;
    max_nodes?: number;
}
/**
 * Knowledge state
 */
export interface KnowledgeEngineState {
    data: any;
    loading: boolean;
    error: Error | null;
    progress: number;
}
/**
 * Custom hook for knowledge engine operations
 * Manages knowledge graph and artifact operations
 */
export declare function useKnowledgeEngine(): {
    artifacts: KnowledgeArtifact[];
    graphData: KnowledgeGraph;
    query: (params: KnowledgeQueryParams) => Promise<KnowledgeQueryResult[]>;
    ingest: (params: KnowledgeIngestParams) => Promise<KnowledgeArtifact | null>;
    getGraph: (params?: GraphQueryParams) => Promise<KnowledgeGraph | null>;
    getArtifacts: () => Promise<KnowledgeArtifact[]>;
    getArtifact: (artifactId: string) => Promise<KnowledgeArtifact | null>;
    updateArtifact: (artifactId: string, updates: Partial<KnowledgeArtifact>) => Promise<void>;
    deleteArtifact: (artifactId: string) => Promise<void>;
    getRelationships: (artifactId: string) => Promise<Array<{
        from: string;
        to: string;
        type: string;
        weight: number;
    }>>;
    semanticSearch: (query: string, limit?: number) => Promise<KnowledgeQueryResult[]>;
    cancel: () => void;
    reset: () => void;
    data: any;
    loading: boolean;
    error: Error | null;
    progress: number;
};
/**
 * Knowledge analytics hook
 */
export declare function useKnowledgeAnalytics(): {
    refetch: () => Promise<void>;
    data: {
        totalArtifacts: number;
        totalRelationships: number;
        growthRate: number;
        topTags: Array<{
            tag: string;
            count: number;
        }>;
    } | null;
    loading: boolean;
    error: Error | null;
};
export { useKnowledgeEngine as useKnowledge };
