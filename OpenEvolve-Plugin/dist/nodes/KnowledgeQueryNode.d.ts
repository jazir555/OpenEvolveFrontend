import { OpenEvolveBaseNode, NodeInputs, NodeResult, ExecutionContext, ValidationError, ParameterSchema } from './OpenEvolveBaseNode';
/**
 * Query types for knowledge retrieval
 */
export type QueryType = 'semantic' | 'entity' | 'relationship' | 'path' | 'aggregation';
/**
 * Knowledge query node configuration
 */
export interface KnowledgeQueryNodeConfig {
    queryType?: QueryType;
    maxResults?: number;
    minRelevance?: number;
    enableReasoning?: boolean;
    includeMetadata?: boolean;
}
/**
 * Knowledge entity
 */
export interface KnowledgeEntity {
    id: string;
    type: string;
    label: string;
    properties: Record<string, any>;
    confidence: number;
}
/**
 * Knowledge relationship
 */
export interface KnowledgeRelationship {
    id: string;
    source: string;
    target: string;
    type: string;
    properties: Record<string, any>;
    confidence: number;
}
/**
 * Knowledge path result
 */
export interface KnowledgePath {
    entities: KnowledgeEntity[];
    relationships: KnowledgeRelationship[];
    length: number;
    confidence: number;
}
/**
 * Query result
 */
export interface KnowledgeQueryResult {
    queryId: string;
    queryType: QueryType;
    query: string;
    results: Array<{
        content: string;
        relevance: number;
        entities?: KnowledgeEntity[];
        source?: string;
        metadata?: Record<string, any>;
    }>;
    entities: KnowledgeEntity[];
    relationships: KnowledgeRelationship[];
    paths?: KnowledgePath[];
    summary: {
        totalResults: number;
        avgRelevance: number;
        entityCount: number;
        relationshipCount: number;
    };
    reasoning?: string;
    metadata: {
        executedAt: Date;
        executionTime: number;
        parameters: {
            maxResults: number;
            minRelevance: number;
            enableReasoning: boolean;
        };
    };
}
/**
 * Knowledge Query Node
 *
 * Queries the knowledge graph for relevant information.
 * Supports multiple query types and advanced reasoning.
 */
export declare class KnowledgeQueryNode extends OpenEvolveBaseNode {
    static readonly DISPLAY_NAME = "Knowledge Query";
    static readonly DESCRIPTION = "Query knowledge graph for semantic information, entities, and relationships";
    static readonly ICON = "knowledge";
    static readonly CATEGORY = "knowledge";
    static readonly VERSION = "1.0.0";
    constructor(id: string, config?: KnowledgeQueryNodeConfig);
    /**
     * Execute knowledge query
     *
     * @param inputs - Must contain 'query' string
     * @param context - Execution context
     * @returns Promise resolving to query result
     */
    execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult>;
    /**
     * Process query result from API
     *
     * @param response - API response
     * @param query - Original query
     * @param queryType - Query type
     * @param startTime - Execution start time
     * @param enableReasoning - Whether reasoning was enabled
     * @returns Processed query result
     */
    private processQueryResult;
    /**
     * Validate input data
     *
     * @param inputs - Input data to validate
     * @returns Array of validation errors
     */
    validateInputs(inputs: NodeInputs): ValidationError[];
    /**
     * Get JSON Schema for configuration parameters
     *
     * @returns Parameter schema
     */
    getParameterSchema(): ParameterSchema;
    /**
     * Add knowledge to the graph
     *
     * @param data - Knowledge data to add
     * @returns Promise resolving to add result
     */
    addKnowledge(data: {
        content: string;
        entities?: Array<{
            type: string;
            label: string;
            properties?: Record<string, any>;
        }>;
        relationships?: Array<{
            source: string;
            target: string;
            type: string;
            properties?: Record<string, any>;
        }>;
        metadata?: Record<string, any>;
    }): Promise<NodeResult>;
    /**
     * Update knowledge in the graph
     *
     * @param knowledgeId - Knowledge ID to update
     * @param data - Updated knowledge data
     * @returns Promise resolving to update result
     */
    updateKnowledge(knowledgeId: string, data: any): Promise<NodeResult>;
    /**
     * Delete knowledge from the graph
     *
     * @param knowledgeId - Knowledge ID to delete
     * @returns Promise resolving to deletion result
     */
    deleteKnowledge(knowledgeId: string): Promise<NodeResult>;
    /**
     * List knowledge artifacts
     *
     * @param params - Optional query parameters
     * @returns Promise resolving to list of knowledge artifacts
     */
    listKnowledge(params?: {
        type?: string;
        limit?: number;
        offset?: number;
    }): Promise<NodeResult>;
    /**
     * Get knowledge statistics
     *
     * @returns Promise resolving to knowledge graph statistics
     */
    getStatistics(): Promise<NodeResult>;
}
export default KnowledgeQueryNode;
