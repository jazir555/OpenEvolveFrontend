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

import axios, { AxiosInstance } from 'axios';
import { OpenEvolveAdapter, KnowledgeArtifact, StructuredLogger, LogContext } from './adapter';
import { IntegrationCoordinator } from './integration-coordinator';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface KnowledgeQuery {
  query: string;
  domain?: string;
  problem_type?: string;
  sources?: string[]; // Specific adapters to query
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

// ============================================================================
// KNOWLEDGE AGGREGATOR CLASS
// ============================================================================

export class KnowledgeAggregator {
  private readonly logger: StructuredLogger;
  private readonly correlationId: string;
  private readonly httpClient: AxiosInstance;
  private readonly knowledgeCache: Map<string, KnowledgeResult[]> = new Map();
  private readonly cacheTimeout: number;

  constructor(
    private readonly openEvolveAdapter: OpenEvolveAdapter,
    private readonly integrationCoordinator: IntegrationCoordinator,
    private readonly timeout_ms: number = 10000,
    cache_timeout_ms: number = 300000, // 5 minutes
  ) {
    this.logger = new StructuredLogger('knowledge-aggregator');
    this.correlationId = this.generateCorrelationId();
    this.cacheTimeout = cache_timeout_ms;

    // Initialize HTTP client
    this.httpClient = axios.create({
      timeout: this.timeout_ms,
      headers: {
        'Content-Type': 'application/json',
        'X-Correlation-ID': this.correlationId,
      },
    });

    this.logger.info('Knowledge aggregator initialized', {
      correlation_id: this.correlationId,
      timeout_ms: this.timeout_ms,
      cache_timeout_ms: this.cacheTimeout,
    });
  }

  // ==========================================================================
  // KNOWLEDGE QUERIES
  // ==========================================================================

  async queryKnowledge(query: KnowledgeQuery): Promise<KnowledgeFusionResult> {
    const startTime = Date.now();

    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'knowledge-aggregator',
      query: query.query,
      domain: query.domain,
    };

    this.logger.info('Querying knowledge', {
      ...context,
      sources: query.sources,
      max_results: query.max_results,
    });

    // Check cache first
    const cacheKey = this.generateCacheKey(query);
    const cachedResults = this.knowledgeCache.get(cacheKey);

    if (cachedResults) {
      this.logger.info('Returning cached knowledge results', {
        ...context,
        result_count: cachedResults.length,
      });

      return {
        query: query.query,
        total_results: cachedResults.length,
        results_by_source: this.groupResultsBySource(cachedResults),
        fused_results: cachedResults,
        fusion_method: 'semantic',
        execution_time_ms: Date.now() - startTime,
      };
    }

    // Query each knowledge source
    const sourcesToQuery = query.sources || this.getAllSources();
    const resultsBySource = new Map<string, KnowledgeResult[]>();

    const queryPromises = sourcesToQuery.map(source =>
      this.querySource(source, query).catch(error => {
        this.logger.warn('Source query failed', {
          ...context,
          source,
          error: error instanceof Error ? error.message : String(error),
        });
        return [] as KnowledgeResult[];
      })
    );

    const sourceResults = await Promise.all(queryPromises);

    for (let i = 0; i < sourcesToQuery.length; i++) {
      const source = sourcesToQuery[i];
      const results = sourceResults[i];

      if (results.length > 0) {
        resultsBySource.set(source, results);
      }
    }

    // Fuse results from all sources
    const allResults = Array.from(resultsBySource.values()).flat();
    const fusedResults = await this.fuseResults(allResults, query);

    // Cache results
    this.knowledgeCache.set(cacheKey, fusedResults);

    // Schedule cache invalidation
    setTimeout(() => {
      this.knowledgeCache.delete(cacheKey);
    }, this.cacheTimeout);

    const executionTime = Date.now() - startTime;

    this.logger.info('Knowledge query completed', {
      ...context,
      source_count: resultsBySource.size,
      total_results: fusedResults.length,
      execution_time_ms: executionTime,
    });

    return {
      query: query.query,
      total_results: fusedResults.length,
      results_by_source: resultsBySource,
      fused_results: fusedResults,
      fusion_method: 'hybrid',
      execution_time_ms: executionTime,
    };
  }

  private async querySource(source: string, query: KnowledgeQuery): Promise<KnowledgeResult[]> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'knowledge-aggregator',
      target_service: source,
      query: query.query,
    };

    this.logger.debug('Querying source', context);

    // Get adapter endpoint
    const adapter = this.integrationCoordinator.getAdapterByName(source);
    if (!adapter) {
      this.logger.warn('Adapter not found', { ...context, source });
      return [];
    }

    try {
      // Make query to adapter
      const response = await this.httpClient.post(
        `${adapter.url}/knowledge/query`,
        {
          query: query.query,
          domain: query.domain,
          max_results: query.max_results || 10,
          similarity_threshold: query.similarity_threshold || 0.5,
        },
        {
          timeout: this.timeout_ms,
        }
      );

      // Transform response to KnowledgeResult format
      const results: KnowledgeResult[] = (response.data.results || []).map((item: any) => ({
        artifact_id: item.id || `${source}-${Date.now()}-${Math.random()}`,
        source,
        source_type: adapter.type,
        content: item.content || item,
        relevance_score: item.score || item.relevance || 0.5,
        metadata: item.metadata || {},
        extracted_at: new Date().toISOString(),
      }));

      this.logger.debug('Source query successful', {
        ...context,
        result_count: results.length,
      });

      return results;
    } catch (error) {
      this.logger.error('Source query failed', {
        ...context,
        error: error instanceof Error ? error.message : String(error),
      });
      return [];
    }
  }

  private async fuseResults(results: KnowledgeResult[], query: KnowledgeQuery): Promise<KnowledgeResult[]> {
    // Sort by relevance score
    const sorted = results.sort((a, b) => b.relevance_score - a.relevance_score);

    // Apply max_results limit
    const maxResults = query.max_results || 50;
    const limited = sorted.slice(0, maxResults);

    // Apply similarity threshold
    const threshold = query.similarity_threshold || 0.0;
    const filtered = limited.filter(r => r.relevance_score >= threshold);

    return filtered;
  }

  private groupResultsBySource(results: KnowledgeResult[]): Map<string, KnowledgeResult[]> {
    const grouped = new Map<string, KnowledgeResult[]>();

    for (const result of results) {
      const sourceResults = grouped.get(result.source) || [];
      sourceResults.push(result);
      grouped.set(result.source, sourceResults);
    }

    return grouped;
  }

  // ==========================================================================
  // KNOWLEDGE EXTRACTION
  // ==========================================================================

  async extractKnowledge(request: KnowledgeExtractionRequest): Promise<KnowledgeArtifact[]> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'knowledge-aggregator',
      workflow_id: request.workflow_id,
    };

    this.logger.info('Extracting knowledge artifacts', {
      ...context,
      extraction_types: request.extraction_types,
      domain: request.domain,
    });

    const artifacts: KnowledgeArtifact[] = [];

    // Get workflow state
    const workflowState = await this.openEvolveAdapter.getWorkflowStatus(request.workflow_id);

    // Extract based on requested types
    for (const extractionType of request.extraction_types) {
      const typeArtifacts = await this.extractKnowledgeByType(
        extractionType,
        workflowState,
        request.domain,
        request.problem_type,
      );

      artifacts.push(...typeArtifacts);
    }

    this.logger.info('Knowledge extraction completed', {
      ...context,
      artifact_count: artifacts.length,
    });

    return artifacts;
  }

  private async extractKnowledgeByType(
    type: string,
    workflowState: any,
    domain?: string,
    problemType?: string,
  ): Promise<KnowledgeArtifact[]> {
    const artifacts: KnowledgeArtifact[] = [];

    switch (type) {
      case 'solution_pattern':
        // Extract solution patterns from workflow
        if (workflowState.final_solution) {
          artifacts.push({
            id: `solution-pattern-${workflowState.workflow_id}-${Date.now()}`,
            artifact_type: 'solution_pattern',
            content: {
              solution: workflowState.final_solution,
              problem_statement: workflowState.problem_statement,
              sub_problems: workflowState.decomposition_plan?.sub_problems || [],
            },
            source_workflow_id: workflowState.workflow_id,
            extraction_timestamp: Date.now(),
            domain,
            problem_type: problemType,
            usage_count: 0,
            effectiveness_score: 0.0,
            related_artifacts: [],
          });
        }
        break;

      case 'team_performance':
        // Extract team performance metrics
        if (workflowState.performance_metrics) {
          artifacts.push({
            id: `team-performance-${workflowState.workflow_id}-${Date.now()}`,
            artifact_type: 'team_performance',
            content: {
              metrics: workflowState.performance_metrics,
              teams: [
                workflowState.content_analyzer_team,
                workflowState.planner_team,
                workflowState.solver_team,
                workflowState.patcher_team,
                workflowState.assembler_team,
              ].filter(Boolean),
            },
            source_workflow_id: workflowState.workflow_id,
            extraction_timestamp: Date.now(),
            domain,
            problem_type: problemType,
          });
        }
        break;

      case 'gauntlet_effectiveness':
        // Extract gauntlet effectiveness data
        if (workflowState.all_critique_reports || workflowState.all_verification_reports) {
          artifacts.push({
            id: `gauntlet-effectiveness-${workflowState.workflow_id}-${Date.now()}`,
            artifact_type: 'gauntlet_effectiveness',
            content: {
              critique_reports: workflowState.all_critique_reports || [],
              verification_reports: workflowState.all_verification_reports || [],
            },
            source_workflow_id: workflowState.workflow_id,
            extraction_timestamp: Date.now(),
            domain,
            problem_type: problemType,
          });
        }
        break;

      default:
        this.logger.warn('Unknown extraction type', {
          extraction_type: type,
        });
    }

    return artifacts;
  }

  // ==========================================================================
  // KNOWLEDGE GRAPH CONSTRUCTION
  // ==========================================================================

  async buildKnowledgeGraph(artifacts: KnowledgeArtifact[]): Promise<{
    nodes: KnowledgeGraphNode[];
    edges: KnowledgeGraphEdge[];
  }> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'knowledge-aggregator',
    };

    this.logger.info('Building knowledge graph', {
      ...context,
      artifact_count: artifacts.length,
    });

    // Create nodes from artifacts
    const nodes: KnowledgeGraphNode[] = artifacts.map(artifact => ({
      id: artifact.id,
      artifact,
      connections: [],
    }));

    // Create edges based on artifact relationships
    const edges: KnowledgeGraphEdge[] = [];

    for (const node of nodes) {
      for (const relatedId of node.artifact.related_artifacts || []) {
        const targetNode = nodes.find(n => n.id === relatedId);
        if (targetNode) {
          edges.push({
            source_id: node.id,
            target_id: targetNode.id,
            edge_type: 'related',
            weight: 1.0,
          });
        }
      }
    }

    // Add semantic similarity edges
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const similarity = this.calculateSemanticSimilarity(nodes[i].artifact, nodes[j].artifact);

        if (similarity > 0.7) {
          edges.push({
            source_id: nodes[i].id,
            target_id: nodes[j].id,
            edge_type: 'semantic_similar',
            weight: similarity,
          });
        }
      }
    }

    // Update node connections
    for (const edge of edges) {
      const sourceNode = nodes.find(n => n.id === edge.source_id);
      if (sourceNode) {
        sourceNode.connections.push(edge);
      }
    }

    this.logger.info('Knowledge graph built', {
      ...context,
      node_count: nodes.length,
      edge_count: edges.length,
    });

    return { nodes, edges };
  }

  private calculateSemanticSimilarity(artifact1: KnowledgeArtifact, artifact2: KnowledgeArtifact): number {
    // Simple similarity based on domain and problem type
    let similarity = 0.0;

    if (artifact1.domain === artifact2.domain) {
      similarity += 0.3;
    }

    if (artifact1.problem_type === artifact2.problem_type) {
      similarity += 0.3;
    }

    if (artifact1.artifact_type === artifact2.artifact_type) {
      similarity += 0.2;
    }

    // Check for overlapping content (simplified)
    const content1 = JSON.stringify(artifact1.content);
    const content2 = JSON.stringify(artifact2.content);
    const words1 = new Set(content1.toLowerCase().split(/\s+/));
    const words2 = new Set(content2.toLowerCase().split(/\s+/));

    const intersection = new Set([...words1].filter(x => words2.has(x)));
    const union = new Set([...words1, ...words2]);

    const jaccard = intersection.size / union.size;
    similarity += jaccard * 0.2;

    return Math.min(similarity, 1.0);
  }

  // ==========================================================================
  // UTILITY METHODS
  // ==========================================================================

  private generateCacheKey(query: KnowledgeQuery): string {
    const key = {
      query: query.query,
      domain: query.domain,
      problem_type: query.problem_type,
      sources: query.sources?.sort() || [],
      max_results: query.max_results,
      similarity_threshold: query.similarity_threshold,
    };

    return Buffer.from(JSON.stringify(key)).toString('base64');
  }

  private getAllSources(): string[] {
    return this.integrationCoordinator
      .getRegisteredAdapters()
      .map(a => a.name);
  }

  private generateCorrelationId(): string {
    return `knowledge-${Date.now()}-${Math.random().toString(36).substring(7)}`;
  }

  getCacheStats(): { size: number; keys: string[] } {
    return {
      size: this.knowledgeCache.size,
      keys: Array.from(this.knowledgeCache.keys()),
    };
  }

  clearCache(): void {
    this.knowledgeCache.clear();
    this.logger.info('Knowledge cache cleared', {
      correlation_id: this.correlationId,
    });
  }
}

// ============================================================================
// FACTORY FUNCTION
// ============================================================================

export function createKnowledgeAggregator(
  openEvolveAdapter: OpenEvolveAdapter,
  integrationCoordinator: IntegrationCoordinator,
  timeout_ms?: number,
  cache_timeout_ms?: number,
): KnowledgeAggregator {
  return new KnowledgeAggregator(openEvolveAdapter, integrationCoordinator, timeout_ms, cache_timeout_ms);
}
