/**
 * KarateClub Adapter
 *
 * Main adapter for KarateClub graph ML operations.
 * Follows CLAUDE.md principles:
 * - Law of Air Gap: No imports from core-projects
 * - Runtime Truth: Validate against actual KarateClub
 * - Configuration Explicitness: All config via environment variables
 * - UTC Timestamps: All times in UTC
 * - Idempotent Operations: Safe to retry
 *
 * Architecture:
 * [Core OpenEvolve] --> [KarateClub Adapter (Canonical Layer)] --> [KarateClub Python Engine]
 */

import { v4 as uuidv4 } from 'uuid';

import {
  NodeEmbeddingRequest,
  NodeEmbeddingResponse,
  CommunityDetectionRequest,
  CommunityDetectionResponse,
  GraphEmbeddingRequest,
  GraphEmbeddingResponse,
  GraphAnalysisRequest,
  GraphAnalysisResponse,
  GraphStructure,
  validateNodeEmbeddingRequest,
  validateCommunityDetectionRequest,
  validateGraphEmbeddingRequest,
  validateGraphAnalysisRequest,
} from '../../schemas/karateclub-canonical';

import { KarateClubMLClient, KarateClubClientConfig } from './ml-client';

export interface AdapterConfig extends KarateClubClientConfig {
  enableMetrics?: boolean;
  logLevel?: 'debug' | 'info' | 'warn' | 'error';
}

export class KarateClubAdapter {
  private client: KarateClubMLClient;
  private config: AdapterConfig;
  private metrics: {
    totalRequests: number;
    successfulRequests: number;
    failedRequests: number;
    totalExecutionTimeMs: number;
  };

  constructor(config: AdapterConfig = {}) {
    // Validate environment variables
    this.validateEnvironment();

    this.config = {
      ...config,
      enableMetrics: config.enableMetrics ?? true,
      logLevel: config.logLevel ?? 'info',
    };

    this.client = new KarateClubMLClient(config);

    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      totalExecutionTimeMs: 0,
    };

    this.log('info', 'KarateClub adapter initialized', {
      python_path: this.config.pythonPath,
      timeout_ms: this.config.timeoutMs,
    });
  }

  /**
   * Validate environment variables (Law of Configuration Explicitness)
   */
  private validateEnvironment(): void {
    const required: string[] = [];

    // Optional but recommended
    if (!process.env.TIMEOUT_MS) {
      this.log('warn', 'TIMEOUT_MS not set, using default 60000ms');
    }

    if (required.length > 0) {
      throw new Error(`Missing required environment variables: ${required.join(', ')}`);
    }
  }

  /**
   * Structured logging (JSON Lines)
   */
  private log(level: string, msg: string, metadata?: Record<string, any>): void {
    if (this.shouldLog(level)) {
      const logEntry = {
        level,
        msg,
        timestamp: new Date().toISOString(),
        source_service: 'karateclub-adapter',
        adapter_version: '1.0.0',
        correlation_id: uuidv4(),
        ...metadata,
      };
      console.log(JSON.stringify(logEntry));
    }
  }

  /**
   * Check if message should be logged based on log level
   */
  private shouldLog(level: string): boolean {
    const levels = ['debug', 'info', 'warn', 'error'];
    const currentLevel = levels.indexOf(this.config.logLevel ?? 'info');
    const msgLevel = levels.indexOf(level);
    return msgLevel >= currentLevel;
  }

  /**
   * Update metrics
   */
  private updateMetrics(success: boolean, executionTimeMs: number): void {
    if (this.config.enableMetrics) {
      this.metrics.totalRequests++;
      this.metrics.totalExecutionTimeMs += executionTimeMs;

      if (success) {
        this.metrics.successfulRequests++;
      } else {
        this.metrics.failedRequests++;
      }
    }
  }

  /**
   * Get current metrics
   */
  getMetrics() {
    return {
      ...this.metrics,
      averageExecutionTimeMs: this.metrics.totalRequests > 0
        ? this.metrics.totalExecutionTimeMs / this.metrics.totalRequests
        : 0,
      successRate: this.metrics.totalRequests > 0
        ? this.metrics.successfulRequests / this.metrics.totalRequests
        : 0,
    };
  }

  /**
   * Generate node embeddings
   */
  async generateNodeEmbeddings(request: NodeEmbeddingRequest): Promise<NodeEmbeddingResponse> {
    const startTime = Date.now();
    const correlationId = request.correlation_id ?? uuidv4();

    this.log('info', 'Node embedding request received', {
      algorithm: request.algorithm,
      num_nodes: request.graph.nodes.length,
      timeout_ms: request.timeout_ms,
      correlation_id: correlationId,
    });

    // Validate request
    const validation = validateNodeEmbeddingRequest(request);
    if (!validation.success) {
      this.log('error', 'Invalid node embedding request', {
        errors: validation.error.errors,
        correlation_id: correlationId,
      });

      return {
        success: false,
        error: `Validation failed: ${validation.error.errors.map(e => e.message).join(', ')}`,
        dimensions: 0,
        algorithm: request.algorithm,
        metadata: {
          num_nodes: request.graph.nodes.length,
          training_time_ms: 0,
        },
        timestamp: new Date().toISOString(),
        correlation_id: correlationId,
      };
    }

    try {
      // Execute via client
      const response = await this.client.generateNodeEmbeddings(request);

      const executionTime = Date.now() - startTime;
      this.updateMetrics(response.success, executionTime);

      this.log('info', 'Node embedding completed', {
        success: response.success,
        dimensions: response.dimensions,
        execution_time_ms: executionTime,
        correlation_id: correlationId,
      });

      return response;
    } catch (error) {
      const executionTime = Date.now() - startTime;
      this.updateMetrics(false, executionTime);

      this.log('error', 'Node embedding error', {
        error: (error as Error).message,
        execution_time_ms: executionTime,
        correlation_id: correlationId,
      });

      return {
        success: false,
        error: (error as Error).message,
        dimensions: 0,
        algorithm: request.algorithm,
        metadata: {
          num_nodes: request.graph.nodes.length,
          training_time_ms: executionTime,
        },
        timestamp: new Date().toISOString(),
        correlation_id: correlationId,
      };
    }
  }

  /**
   * Detect communities
   */
  async detectCommunities(request: CommunityDetectionRequest): Promise<CommunityDetectionResponse> {
    const startTime = Date.now();
    const correlationId = request.correlation_id ?? uuidv4();

    this.log('info', 'Community detection request received', {
      algorithm: request.algorithm,
      num_nodes: request.graph.nodes.length,
      num_edges: request.graph.edges.length,
      timeout_ms: request.timeout_ms,
      correlation_id: correlationId,
    });

    // Validate request
    const validation = validateCommunityDetectionRequest(request);
    if (!validation.success) {
      this.log('error', 'Invalid community detection request', {
        errors: validation.error.errors,
        correlation_id: correlationId,
      });

      return {
        success: false,
        error: `Validation failed: ${validation.error.errors.map(e => e.message).join(', ')}`,
        algorithm: request.algorithm,
        metadata: {
          detection_time_ms: 0,
        },
        timestamp: new Date().toISOString(),
        correlation_id: correlationId,
      };
    }

    try {
      // Execute via client
      const response = await this.client.detectCommunities(request);

      const executionTime = Date.now() - startTime;
      this.updateMetrics(response.success, executionTime);

      this.log('info', 'Community detection completed', {
        success: response.success,
        num_communities: response.num_communities,
        execution_time_ms: executionTime,
        correlation_id: correlationId,
      });

      return response;
    } catch (error) {
      const executionTime = Date.now() - startTime;
      this.updateMetrics(false, executionTime);

      this.log('error', 'Community detection error', {
        error: (error as Error).message,
        execution_time_ms: executionTime,
        correlation_id: correlationId,
      });

      return {
        success: false,
        error: (error as Error).message,
        algorithm: request.algorithm,
        metadata: {
          detection_time_ms: executionTime,
        },
        timestamp: new Date().toISOString(),
        correlation_id: correlationId,
      };
    }
  }

  /**
   * Perform comprehensive graph analysis
   */
  async analyzeGraph(request: GraphAnalysisRequest): Promise<GraphAnalysisResponse> {
    const startTime = Date.now();
    const correlationId = request.correlation_id ?? uuidv4();

    this.log('info', 'Graph analysis request received', {
      analyses: request.analyses,
      num_nodes: request.graph.nodes.length,
      num_edges: request.graph.edges.length,
      timeout_ms: request.timeout_ms,
      correlation_id: correlationId,
    });

    // Validate request
    const validation = validateGraphAnalysisRequest(request);
    if (!validation.success) {
      this.log('error', 'Invalid graph analysis request', {
        errors: validation.error.errors,
        correlation_id: correlationId,
      });

      return {
        success: false,
        error: `Validation failed: ${validation.error.errors.map(e => e.message).join(', ')}`,
        execution_time_ms: 0,
        timestamp: new Date().toISOString(),
        correlation_id: correlationId,
      };
    }

    const results: any = {};
    const algorithmsUsed: any = {};
    let hasError = false;
    let firstError: string | undefined;

    try {
      // Node embeddings
      if (request.analyses.includes('node_embeddings')) {
        const algorithm = request.node_embedding_algorithm ?? 'node2vec';

        const embeddingResponse = await this.generateNodeEmbeddings({
          algorithm,
          graph: request.graph,
          parameters: {
            dimensions: request.parameters?.embedding_dimensions ?? 128,
          },
          timeout_ms: Math.min(request.timeout_ms, 120000),
          correlation_id: correlationId,
        });

        if (embeddingResponse.success) {
          results.node_embeddings = embeddingResponse.embeddings;
          algorithmsUsed.node_embedding = algorithm;
        } else {
          hasError = true;
          firstError = firstError ?? embeddingResponse.error;
        }
      }

      // Community detection
      if (request.analyses.includes('community_detection')) {
        const algorithm = request.community_algorithm ?? 'label_propagation';

        const communityResponse = await this.detectCommunities({
          algorithm,
          graph: request.graph,
          timeout_ms: Math.min(request.timeout_ms, 60000),
          correlation_id: correlationId,
        });

        if (communityResponse.success) {
          results.communities = communityResponse.memberships;
          algorithmsUsed.community_detection = algorithm;
        } else {
          hasError = true;
          firstError = firstError ?? communityResponse.error;
        }
      }

      // Graph statistics
      if (request.analyses.includes('graph_statistics')) {
        results.graph_statistics = this.calculateGraphStatistics(request.graph);
      }

      // Centrality (requires NetworkX in Python)
      if (request.analyses.includes('centrality')) {
        // This would require additional Python implementation
        // For now, we'll skip it
        this.log('warn', 'Centrality analysis not yet implemented', {
          correlation_id: correlationId,
        });
      }

      const executionTime = Date.now() - startTime;
      this.updateMetrics(!hasError, executionTime);

      this.log('info', 'Graph analysis completed', {
        success: !hasError,
        execution_time_ms: executionTime,
        algorithms_used: algorithmsUsed,
        correlation_id: correlationId,
      });

      return {
        success: !hasError,
        results: results,
        algorithms_used: algorithmsUsed,
        execution_time_ms: executionTime,
        error: hasError ? firstError : undefined,
        timestamp: new Date().toISOString(),
        correlation_id: correlationId,
      };
    } catch (error) {
      const executionTime = Date.now() - startTime;
      this.updateMetrics(false, executionTime);

      this.log('error', 'Graph analysis error', {
        error: (error as Error).message,
        execution_time_ms: executionTime,
        correlation_id: correlationId,
      });

      return {
        success: false,
        error: (error as Error).message,
        execution_time_ms: executionTime,
        timestamp: new Date().toISOString(),
        correlation_id: correlationId,
      };
    }
  }

  /**
   * Calculate basic graph statistics
   */
  private calculateGraphStatistics(graph: GraphStructure) {
    const numNodes = graph.nodes.length;
    const numEdges = graph.edges.length;

    // Density (undirected graph)
    const maxEdges = (numNodes * (numNodes - 1)) / 2;
    const density = maxEdges > 0 ? numEdges / maxEdges : 0;

    // Average degree
    const degrees: Record<string, number> = {};
    graph.nodes.forEach(node => {
      degrees[node.id] = 0;
    });
    graph.edges.forEach(edge => {
      degrees[edge.source] = (degrees[edge.source] || 0) + 1;
      degrees[edge.target] = (degrees[edge.target] || 0) + 1;
    });

    const avgDegree = numNodes > 0
      ? Object.values(degrees).reduce((sum, deg) => sum + deg, 0) / numNodes
      : 0;

    return {
      num_nodes: numNodes,
      num_edges: numEdges,
      density: density,
      is_connected: true, // Would need full connected components analysis
      avg_degree: avgDegree,
    };
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<{ healthy: boolean; version?: string; error?: string }> {
    this.log('info', 'Health check requested');

    try {
      const health = await this.client.healthCheck();

      this.log('info', 'Health check completed', {
        healthy: health.healthy,
        version: health.version,
      });

      return health;
    } catch (error) {
      this.log('error', 'Health check failed', {
        error: (error as Error).message,
      });

      return {
        healthy: false,
        error: (error as Error).message,
      };
    }
  }

  /**
   * Reset metrics
   */
  resetMetrics(): void {
    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      totalExecutionTimeMs: 0,
    };

    this.log('info', 'Metrics reset');
  }
}

// Export singleton instance
let defaultAdapter: KarateClubAdapter | null = null;

export function getDefaultAdapter(config?: AdapterConfig): KarateClubAdapter {
  if (!defaultAdapter) {
    defaultAdapter = new KarateClubAdapter(config);
  }
  return defaultAdapter;
}

export function createAdapter(config?: AdapterConfig): KarateClubAdapter {
  return new KarateClubAdapter(config);
}
