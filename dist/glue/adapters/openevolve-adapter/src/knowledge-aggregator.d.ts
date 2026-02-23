/**
 * Knowledge Aggregator
 *
 * Aggregates knowledge from all integrated sources:
 * - Z3 Prover: Proofs, lemmas, theorems
 * - LeanAide: Tactic libraries, proof patterns
 * - RAGBits: Document embeddings, semantic knowledge
 * - Vector DB: Vector representations
 * - Graphiti: Graph knowledge entities
 * - KarateClub: ML embeddings, clusters
 *
 * The aggregator provides:
 * - Unified knowledge query interface
 * - Cross-source knowledge fusion
 * - Knowledge artifact extraction
 * - Semantic search across all sources
 * - Knowledge graph construction
 *
 * Environment Variables:
 *   KNOWLEDGE_AGGREGATION_TIMEOUT_MS - Query timeout
 *   MAX_KNOWLEDGE_ARTIFACTS - Maximum artifacts to return
 *   SEMANTIC_SIMILARITY_THRESHOLD - Minimum similarity score
 */
import { OpenEvolveAdapter, KnowledgeArtifact } from './adapter';
import { IntegrationCoordinator } from './integration-coordinator';
export interface KnowledgeQuery {
    query: string;
    domain?: string;
    problem_type?: string;
    sources?: string[];
    max_results?: number;
    similarity_threshold?: number;
    include_embeddings?: boolean;
}
export interface KnowledgeResult {
    artifact_id: string;
    source: string;
    source_type: string;
    content: any;
    relevance_score: number;
    metadata: Record<string, any>;
    extracted_at: string;
}
export interface KnowledgeFusionResult {
    query: string;
    total_results: number;
    results_by_source: Map<string, KnowledgeResult[]>;
    fused_results: KnowledgeResult[];
    fusion_method: 'semantic' | 'graph' | 'hybrid';
    execution_time_ms: number;
}
export interface KnowledgeExtractionRequest {
    workflow_id: string;
    extraction_types: ('solution_pattern' | 'problem_solution_mapping' | 'critique_insight' | 'team_performance' | 'gauntlet_effectiveness')[];
    domain?: string;
    problem_type?: string;
    max_artifacts?: number;
}
export interface KnowledgeGraphEdge {
    source_id: string;
    target_id: string;
    edge_type: string;
    weight: number;
    metadata?: Record<string, any>;
}
export interface KnowledgeGraphNode {
    id: string;
    artifact: KnowledgeArtifact;
    connections: KnowledgeGraphEdge[];
    centrality_score?: number;
    cluster_id?: string;
}
export declare class KnowledgeAggregator {
    private readonly openEvolveAdapter;
    private readonly integrationCoordinator;
    private readonly timeout_ms;
    private readonly logger;
    private readonly correlationId;
    private readonly httpClient;
    private readonly knowledgeCache;
    private readonly cacheTimeout;
    constructor(openEvolveAdapter: OpenEvolveAdapter, integrationCoordinator: IntegrationCoordinator, timeout_ms?: number, cache_timeout_ms?: number);
    queryKnowledge(query: KnowledgeQuery): Promise<KnowledgeFusionResult>;
    private querySource;
    private fuseResults;
    private groupResultsBySource;
    extractKnowledge(request: KnowledgeExtractionRequest): Promise<KnowledgeArtifact[]>;
    private extractKnowledgeByType;
    buildKnowledgeGraph(artifacts: KnowledgeArtifact[]): Promise<{
        nodes: KnowledgeGraphNode[];
        edges: KnowledgeGraphEdge[];
    }>;
    private calculateSemanticSimilarity;
    private generateCacheKey;
    private getAllSources;
    private generateCorrelationId;
    getCacheStats(): {
        size: number;
        keys: string[];
    };
    clearCache(): void;
}
export declare function createKnowledgeAggregator(openEvolveAdapter: OpenEvolveAdapter, integrationCoordinator: IntegrationCoordinator, timeout_ms?: number, cache_timeout_ms?: number): KnowledgeAggregator;
//# sourceMappingURL=knowledge-aggregator.d.ts.map