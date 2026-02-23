/**
 * KarateClub Algorithm Registry
 *
 * Complete registry of all 51 KarateClub algorithms with their parameters,
 * papers, and metadata. Used for validation and documentation.
 *
 * Reference: core-projects/karateclub/
 */
import { NodeEmbeddingAlgorithm, CommunityAlgorithm, GraphEmbeddingAlgorithm, AlgorithmCategory } from '../../schemas/karateclub-canonical';
export interface AlgorithmInfo {
    name: string;
    description: string;
    category: AlgorithmCategory;
    paper?: string;
    year?: number;
    parameters: ParameterInfo[];
    defaultTimeout: number;
}
export interface ParameterInfo {
    name: string;
    type: 'number' | 'integer' | 'boolean' | 'string';
    default?: any;
    description: string;
    required?: boolean;
}
/**
 * Community Detection Algorithms (10 total)
 */
export declare const COMMUNITY_ALGORITHMS: Record<CommunityAlgorithm, AlgorithmInfo>;
/**
 * Node Embedding Algorithms (32 total)
 */
export declare const NODE_EMBEDDING_ALGORITHMS: Partial<Record<NodeEmbeddingAlgorithm, AlgorithmInfo>>;
/**
 * Graph Embedding Algorithms (10 total)
 */
export declare const GRAPH_EMBEDDING_ALGORITHMS: Partial<Record<GraphEmbeddingAlgorithm, AlgorithmInfo>>;
/**
 * Get algorithm information by name and category
 */
export declare function getAlgorithmInfo(algorithm: string, category: AlgorithmCategory): AlgorithmInfo | undefined;
/**
 * Get all algorithms by category
 */
export declare function getAlgorithmsByCategory(category: AlgorithmCategory): string[];
/**
 * Get default timeout for algorithm
 */
export declare function getDefaultTimeout(algorithm: string, category: AlgorithmCategory): number;
//# sourceMappingURL=algorithms.d.ts.map