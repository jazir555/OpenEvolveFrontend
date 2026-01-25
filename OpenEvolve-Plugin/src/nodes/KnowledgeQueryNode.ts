/**
 * Knowledge Query Node
 *
 * Knowledge graph query node for intelligent information retrieval.
 * Supports semantic search, entity extraction, and relationship analysis.
 *
 * @module nodes
 */

import {
  OpenEvolveBaseNode,
  NodeInputs,
  NodeResult,
  ExecutionContext,
  ValidationError,
  ParameterSchema
} from './OpenEvolveBaseNode';
import { knowledgeApi } from '@/services/api/endpoints';

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
export class KnowledgeQueryNode extends OpenEvolveBaseNode {
  static readonly DISPLAY_NAME = 'Knowledge Query';
  static readonly DESCRIPTION = 'Query knowledge graph for semantic information, entities, and relationships';
  static readonly ICON = 'knowledge';
  static readonly CATEGORY = 'knowledge';
  static readonly VERSION = '1.0.0';

  constructor(id: string, config: KnowledgeQueryNodeConfig = {}) {
    super(id, {
      queryType: 'semantic',
      maxResults: 10,
      minRelevance: 0.5,
      enableReasoning: false,
      includeMetadata: true,
      ...config
    });
  }

  /**
   * Execute knowledge query
   *
   * @param inputs - Must contain 'query' string
   * @param context - Execution context
   * @returns Promise resolving to query result
   */
  async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    try {
      const startTime = Date.now();

      // Extract inputs
      const query = inputs.query as string;
      const queryType = (inputs.queryType as QueryType) || (this.config.queryType as QueryType);
      const maxResults = (inputs.maxResults as number) || this.config.maxResults as number;
      const minRelevance = (inputs.minRelevance as number) || this.config.minRelevance as number;
      const enableReasoning = inputs.enableReasoning as boolean ?? this.config.enableReasoning as boolean;
      const filters = inputs.filters as Record<string, any> | undefined;

      // Validate required inputs
      if (!query || query.trim().length === 0) {
        return this.createErrorResult('Query is required and cannot be empty');
      }

      context.updateProgress(10, 'Preparing knowledge query');

      // Prepare query parameters
      const queryParams: Record<string, any> = {
        query,
        query_type: queryType,
        max_results: maxResults,
        min_relevance: minRelevance,
        enable_reasoning: enableReasoning,
        include_metadata: this.config.includeMetadata as boolean
      };

      if (filters) {
        queryParams.filters = filters;
      }

      context.updateProgress(30, 'Executing knowledge query');

      // Execute query via API
      let response: any;
      if (queryType === 'semantic') {
        // Use RAG search for semantic queries
        const ragResults = await knowledgeApi.searchRag({ query });
        
        // Filter and limit results
        const filteredResults = (ragResults || [])
          .filter((r: any) => {
            const score = r.score !== undefined ? r.score : (r.relevance !== undefined ? r.relevance : 0.5);
            return score >= minRelevance;
          })
          .slice(0, maxResults);
          
        response = { results: filteredResults };
      } else if (['relationship', 'path'].includes(queryType)) {
        // Use Graphiti for graph-based queries
        const graphResult = await knowledgeApi.searchGraphiti({ query, num_results: maxResults });
        
        if (graphResult && graphResult.nodes) {
          // Transform graph result to expected format
          response = {
            results: [],
            entities: (graphResult.nodes || []).map((n: any) => ({ label: n.id, ...n })),
            relationships: (graphResult.edges || []).map((e: any) => ({ 
               source: e.source, 
               target: e.target, 
               type: e.label, 
               confidence: 1.0 
            }))
          };
        } else {
          response = { results: [], entities: [], relationships: [] };
        }
      } else {
        // Fallback or generic query (implementation pending backend support)
        // For now, try RAG as a safe default
        const ragResults = await knowledgeApi.searchRag({ query });
        response = { results: ragResults };
      }

      context.updateProgress(70, 'Processing query results');

      // Process results
      const result = this.processQueryResult(response, query, queryType, startTime, enableReasoning);

      context.updateProgress(100, `Query complete: ${result.summary.totalResults} results found`);

      return this.createSuccessResult(result);

    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Unknown error during knowledge query'
      );
    }
  }

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
  private processQueryResult(
    response: any,
    query: string,
    queryType: QueryType,
    startTime: number,
    enableReasoning: boolean
  ): KnowledgeQueryResult {
    const executionTime = Date.now() - startTime;
    const safeResponse = response || {};

    // Extract results from response
    const results = (safeResponse.results || []).map((r: any) => ({
      content: r.content || '',
      relevance: r.relevance !== undefined ? r.relevance : (r.score !== undefined ? r.score : 0.5),
      entities: r.entities || [],
      source: r.source,
      metadata: r.metadata
    }));

    // Extract entities and relationships
    const entities = (safeResponse.entities || []).map((e: any) => ({
      id: e.id || '',
      type: e.type || 'unknown',
      label: e.label || '',
      properties: e.properties || {},
      confidence: e.confidence || 0.5
    }));

    const relationships = (safeResponse.relationships || []).map((r: any) => ({
      id: r.id || '',
      source: r.source || '',
      target: r.target || '',
      type: r.type || 'related_to',
      properties: r.properties || {},
      confidence: r.confidence || 0.5
    }));

    // Extract paths if available
    const paths = safeResponse.paths?.map((p: any) => ({
      entities: p.entities || [],
      relationships: p.relationships || [],
      length: p.length || 0,
      confidence: p.confidence || 0.5
    }));

    // Calculate summary statistics
    const totalResults = results.length;
    const avgRelevance = totalResults > 0
      ? results.reduce((sum, r) => sum + r.relevance, 0) / totalResults
      : 0;

    const summary = {
      totalResults,
      avgRelevance,
      entityCount: entities.length,
      relationshipCount: relationships.length
    };

    return {
      queryId: response.query_id || `query-${Date.now()}`,
      queryType,
      query,
      results,
      entities,
      relationships,
      paths,
      summary,
      reasoning: enableReasoning ? response.reasoning : undefined,
      metadata: {
        executedAt: new Date(),
        executionTime,
        parameters: {
          maxResults: this.config.maxResults as number,
          minRelevance: this.config.minRelevance as number,
          enableReasoning
        }
      }
    };
  }

  /**
   * Validate input data
   *
   * @param inputs - Input data to validate
   * @returns Array of validation errors
   */
  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];

    if (!inputs.query) {
      errors.push({
        field: 'query',
        message: 'Query is required',
        severity: 'error'
      });
    }

    if (inputs.query && typeof inputs.query !== 'string') {
      errors.push({
        field: 'query',
        message: 'Query must be a string',
        severity: 'error'
      });
    }

    if (inputs.query && inputs.query.length < 3) {
      errors.push({
        field: 'query',
        message: 'Query is too short (minimum 3 characters)',
        severity: 'warning'
      });
    }

    // Validate query type
    if (inputs.queryType && typeof inputs.queryType === 'string') {
      const validTypes = ['semantic', 'entity', 'relationship', 'path', 'aggregation'];
      if (!validTypes.includes(inputs.queryType)) {
        errors.push({
          field: 'queryType',
          message: `Query type must be one of: ${validTypes.join(', ')}`,
          severity: 'error'
        });
      }
    }

    // Validate max results
    if (inputs.maxResults && typeof inputs.maxResults === 'number') {
      if (inputs.maxResults < 1 || inputs.maxResults > 100) {
        errors.push({
          field: 'maxResults',
          message: 'Max results must be between 1 and 100',
          severity: 'error'
        });
      }
    }

    // Validate min relevance
    if (inputs.minRelevance && typeof inputs.minRelevance === 'number') {
      if (inputs.minRelevance < 0 || inputs.minRelevance > 1) {
        errors.push({
          field: 'minRelevance',
          message: 'Min relevance must be between 0 and 1',
          severity: 'error'
        });
      }
    }

    return errors;
  }

  /**
   * Get JSON Schema for configuration parameters
   *
   * @returns Parameter schema
   */
  getParameterSchema(): ParameterSchema {
    return {
      type: 'object',
      properties: {
        queryType: {
          type: 'string',
          description: 'Type of knowledge query to execute',
          enum: ['semantic', 'entity', 'relationship', 'path', 'aggregation'],
          default: 'semantic'
        },
        maxResults: {
          type: 'number',
          description: 'Maximum number of results to return',
          minimum: 1,
          maximum: 100,
          default: 10
        },
        minRelevance: {
          type: 'number',
          description: 'Minimum relevance threshold for results (0-1)',
          minimum: 0,
          maximum: 1,
          default: 0.5
        },
        enableReasoning: {
          type: 'boolean',
          description: 'Enable reasoning and explanation',
          default: false
        },
        includeMetadata: {
          type: 'boolean',
          description: 'Include metadata in results',
          default: true
        }
      },
      required: []
    };
  }

  /**
   * Add knowledge to the graph
   *
   * @param data - Knowledge data to add
   * @returns Promise resolving to add result
   */
  async addKnowledge(data: {
    content: string;
    entities?: Array<{ type: string; label: string; properties?: Record<string, any> }>;
    relationships?: Array<{ source: string; target: string; type: string; properties?: Record<string, any> }>;
    metadata?: Record<string, any>;
  }): Promise<NodeResult> {
    try {
      const response = await knowledgeApi.add(data);
      return this.createSuccessResult(response);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to add knowledge'
      );
    }
  }

  /**
   * Update knowledge in the graph
   *
   * @param knowledgeId - Knowledge ID to update
   * @param data - Updated knowledge data
   * @returns Promise resolving to update result
   */
  async updateKnowledge(knowledgeId: string, data: any): Promise<NodeResult> {
    // Backend support pending for update operations
    return this.createErrorResult('Update operation not supported by backend yet');
  }

  /**
   * Delete knowledge from the graph
   *
   * @param knowledgeId - Knowledge ID to delete
   * @returns Promise resolving to deletion result
   */
  async deleteKnowledge(knowledgeId: string): Promise<NodeResult> {
    // Backend support pending for delete operations
    return this.createErrorResult('Delete operation not supported by backend yet');
  }

  /**
   * List knowledge artifacts
   *
   * @param params - Optional query parameters
   * @returns Promise resolving to list of knowledge artifacts
   */
  async listKnowledge(params?: {
    type?: string;
    limit?: number;
    offset?: number;
  }): Promise<NodeResult> {
    try {
      const response = await knowledgeApi.list(params);
      return this.createSuccessResult(response);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to list knowledge'
      );
    }
  }

  /**
   * Get knowledge statistics
   *
   * @returns Promise resolving to knowledge graph statistics
   */
  async getStatistics(): Promise<NodeResult> {
    try {
      const response = await knowledgeApi.getStatistics();
      return this.createSuccessResult(response);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to get statistics'
      );
    }
  }
}

export default KnowledgeQueryNode;
