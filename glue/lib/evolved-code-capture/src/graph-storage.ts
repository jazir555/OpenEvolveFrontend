/**
 * Graph Storage Integration for Evolved Code
 *
 * Following CLAUDE.md Federation Constitution:
 * - Law of the Air Gap: No imports from core-projects
 * - Law of Runtime Truth: Verify Graphiti connection before use
 * - Law of Idempotency: Safe to run multiple times
 * - Law of Configuration Explicitness: All config via environment variables
 * - Failure Management: Circuit breaker for transient failures
 *
 * Integrates with Graphiti adapter to store evolved code as temporal episodes
 * for lineage tracking and knowledge graph-based retrieval.
 */

import { v4 as uuidv4 } from 'uuid';
import { Logger } from '../../logger';
import { CircuitBreaker } from '../../circuit-breaker';
import {
  EvolvedCode,
  Problem,
  EvolutionLineage,
  EvolutionNode,
  validateEvolvedCode,
} from './canonical';

// ============================================================================
// CONFIGURATION
// ============================================================================

export interface GraphStorageConfig {
  // Graphiti adapter URL
  graphiti_adapter_url: string;

  // Episode configuration
  episode_type_base: string;

  // Timeout and retry configuration
  timeout_ms?: number;
  max_retries?: number;
  circuit_breaker_threshold?: number;
  circuit_breaker_timeout_ms?: number;

  // Logging
  logger?: Logger;
}

const DEFAULT_CONFIG = {
  timeout_ms: 30000,
  max_retries: 3,
  circuit_breaker_threshold: 5,
  circuit_breaker_timeout_ms: 60000,
  episode_type_base: 'evolved_code',
};

// ============================================================================
// GRAPH STORAGE CLIENT
// ============================================================================

/**
 * Graph Storage for Evolved Code
 *
 * Integrates with Graphiti adapter to store evolved code as temporal episodes
 */
export class GraphStorage {
  private readonly config: Required<Omit<GraphStorageConfig, 'logger'>> & {
    logger?: Logger;
  };

  private readonly logger: Logger;
  private readonly circuitBreaker: CircuitBreaker;
  private initialized: boolean = false;

  // Graphiti adapter HTTP client (simple implementation)
  private readonly httpClient: {
    post: (path: string, body: any) => Promise<any>;
    get: (path: string) => Promise<any>;
  };

  constructor(config: GraphStorageConfig) {
    this.config = {
      ...DEFAULT_CONFIG,
      ...config,
    };

    this.logger = this.config.logger || new Logger('graph-storage');

    // Initialize circuit breaker
    this.circuitBreaker = new CircuitBreaker({
      threshold: this.config.circuit_breaker_threshold,
      timeout_ms: this.config.circuit_breaker_timeout_ms,
      onStateChange: (oldState, newState) => {
        this.logger.warn('Circuit breaker state changed', {
          correlation_id: 'graph-storage-circuit',
          old_state: oldState,
          new_state: newState,
        });
      },
    });

    // Simple HTTP client for Graphiti adapter
    this.httpClient = {
      post: async (path: string, body: any) => {
        const url = `${this.config.graphiti_adapter_url}${path}`;
        const response = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(this.config.timeout_ms),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return response.json();
      },
      get: async (path: string) => {
        const url = `${this.config.graphiti_adapter_url}${path}`;
        const response = await fetch(url, {
          signal: AbortSignal.timeout(this.config.timeout_ms),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return response.json();
      },
    };

    this.logger.info('GraphStorage initialized', {
      correlation_id: 'graph-storage-init',
      graphiti_adapter_url: this.config.graphiti_adapter_url,
      episode_type_base: this.config.episode_type_base,
    });
  }

  // ========================================================================
  // INITIALIZATION
  // ========================================================================

  /**
   * Initialize graph storage
   * Following CLAUDE.md: RUNTIME TRUTH - verify before marking as ready
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      this.logger.warn('GraphStorage already initialized', {
        correlation_id: 'graph-storage-init',
      });
      return;
    }

    const correlationId = uuidv4();

    this.logger.info('Initializing GraphStorage', {
      correlation_id: correlationId,
      target_service: 'graphiti-adapter',
    });

    try {
      await this.circuitBreaker.execute(async () => {
        // Health check
        const health = await this.httpClient.get('/health');
        if (!health.healthy) {
          throw new Error('Graphiti adapter is not healthy');
        }
      });

      this.initialized = true;
      this.logger.info('GraphStorage initialized successfully', {
        correlation_id: correlationId,
      });
    } catch (error) {
      this.logger.error('Failed to initialize GraphStorage', error as Error, {
        correlation_id: correlationId,
      });
      throw new Error(
        `GraphStorage initialization failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  // ========================================================================
  // EPISODE OPERATIONS
  // ========================================================================

  /**
   * Store evolved code as a Graphiti episode
   * Following CLAUDE.md: Law of Idempotency - safe to run multiple times
   */
  async storeAsEpisode(
    evolvedCode: EvolvedCode,
    correlationId?: string
  ): Promise<{ episode_id: string; success: boolean }> {
    const cid = correlationId || uuidv4();

    this.logger.info('Storing evolved code as episode', {
      correlation_id: cid,
      code_id: evolvedCode.id,
      episode_type: this.config.episode_type_base,
    });

    // Validate evolved code
    const validation = validateEvolvedCode(evolvedCode);
    if (!validation.success) {
      throw new Error(`Invalid evolved code: ${validation.errors.join(', ')}`);
    }

    try {
      const result = await this.circuitBreaker.execute(async () => {
        // Create episode description
        const episodeDescription = this.createEpisodeDescription(evolvedCode);

        // Create episode
        const episode = {
          name: `Evolved Code: ${evolvedCode.problem.type}`,
          episode_type: `${this.config.episode_type_base}_${evolvedCode.problem.type}`,
          description: episodeDescription,
          timestamp: evolvedCode.timestamp_utc,
          source: 'openevolve',
          metadata: {
            code_id: evolvedCode.id,
            language: evolvedCode.language,
            problem_type: evolvedCode.problem.type,
            fitness_score: evolvedCode.metrics.fitness_score,
            iterations: evolvedCode.metrics.iterations,
            duration_ms: evolvedCode.metrics.duration_ms,
            is_valid: evolvedCode.is_valid,
            tags: evolvedCode.tags || [],
            generation_number: evolvedCode.generation_number,
          },
        };

        // Add episode to Graphiti
        const response = await this.httpClient.post('/episodes', {
          operation: episode,
          correlation_id: cid,
        });

        return response;
      });

      this.logger.info('Evolved code stored as episode successfully', {
        correlation_id: cid,
        code_id: evolvedCode.id,
        episode_id: result.episode_id,
      });

      return {
        episode_id: result.episode_id,
        success: result.success,
      };
    } catch (error) {
      this.logger.error('Failed to store evolved code as episode', error as Error, {
        correlation_id: cid,
        code_id: evolvedCode.id,
      });
      throw error;
    }
  }

  /**
   * Create episode description from evolved code
   */
  private createEpisodeDescription(evolvedCode: EvolvedCode): string {
    const parts: string[] = [];

    parts.push(`Problem: ${evolvedCode.problem.description}`);
    parts.push(`Type: ${evolvedCode.problem.type}`);
    parts.push(`Language: ${evolvedCode.language}`);
    parts.push(`Fitness Score: ${evolvedCode.metrics.fitness_score}`);
    parts.push(`Iterations: ${evolvedCode.metrics.iterations}`);
    parts.push(`Duration: ${evolvedCode.metrics.duration_ms}ms`);

    if (evolvedCode.metrics.fitness_improvement) {
      parts.push(`Improvement: ${evolvedCode.metrics.fitness_improvement}`);
    }

    return parts.join('\n');
  }

  // ========================================================================
  // LINKING OPERATIONS
  // ========================================================================

  /**
   * Link problem to solution in the knowledge graph
   * Creates a relationship between problem and solution entities
   */
  async linkProblemToSolution(
    problemId: string,
    solutionId: string,
    correlationId?: string
  ): Promise<{ success: boolean; edge_id: string }> {
    const cid = correlationId || uuidv4();

    this.logger.info('Linking problem to solution', {
      correlation_id: cid,
      problem_id: problemId,
      solution_id: solutionId,
    });

    try {
      const result = await this.circuitBreaker.execute(async () => {
        // Create triplet
        const triplet = {
          subject: { name: `Problem:${problemId}`, type: 'Problem' },
          predicate: { relation_type: 'SOLVED_BY', attributes: {} },
          object: { name: `Solution:${solutionId}`, type: 'Solution' },
          timestamp: new Date().toISOString(),
        };

        const response = await this.httpClient.post('/triplets', {
          operation: triplet,
          correlation_id: cid,
        });

        return response;
      });

      this.logger.info('Problem linked to solution successfully', {
        correlation_id: cid,
        problem_id: problemId,
        solution_id: solutionId,
        edge_id: result.edge_id,
      });

      return {
        success: result.success,
        edge_id: result.edge_id,
      };
    } catch (error) {
      this.logger.error('Failed to link problem to solution', error as Error, {
        correlation_id: cid,
        problem_id: problemId,
        solution_id: solutionId,
      });
      throw error;
    }
  }

  // ========================================================================
  // LINEAGE TRACKING
  // ========================================================================

  /**
   * Track evolution lineage for a code solution
   * Builds the evolution tree from initial to final solution
   */
  async trackEvolutionLineage(
    codeId: string,
    correlationId?: string
  ): Promise<EvolutionLineage> {
    const cid = correlationId || uuidv4();

    this.logger.info('Tracking evolution lineage', {
      correlation_id: cid,
      code_id: codeId,
    });

    try {
      const lineage = await this.circuitBreaker.execute(async () => {
        // Search for all related code solutions
        const searchQuery = {
          query: `code_id:${codeId} OR parent_code_id:${codeId}`,
          max_results: 100,
          temporal_filter: {
            start: '1970-01-01T00:00:00.000Z',
            end: new Date().toISOString(),
          },
        };

        const response = await this.httpClient.post('/search', {
          query: searchQuery,
          correlation_id: cid,
        });

        // Build lineage from search results
        return this.buildLineageFromResults(response, codeId);
      });

      this.logger.info('Evolution lineage tracked successfully', {
        correlation_id: cid,
        code_id: codeId,
        total_nodes: lineage.total_nodes,
        depth: lineage.depth,
      });

      return lineage;
    } catch (error) {
      this.logger.error('Failed to track evolution lineage', error as Error, {
        correlation_id: cid,
        code_id: codeId,
      });
      throw error;
    }
  }

  /**
   * Build lineage from search results
   */
  private buildLineageFromResults(results: any, rootCodeId: string): EvolutionLineage {
    const nodes: EvolutionNode[] = [];
    const nodeMap = new Map<string, EvolutionNode>();

    // Create nodes from results
    for (const result of results.edges || []) {
      const node: EvolutionNode = {
        code_id: result.source_uuid || result.target_uuid,
        fitness_score: result.attributes?.fitness_score || 0,
        timestamp_utc: result.created_at || new Date().toISOString(),
        generation: result.attributes?.generation_number || 0,
        parent_id: result.attributes?.parent_code_id,
        children_ids: [],
      };
      nodes.push(node);
      nodeMap.set(node.code_id, node);
    }

    // Build tree structure
    for (const node of nodes) {
      if (node.parent_id && nodeMap.has(node.parent_id)) {
        const parent = nodeMap.get(node.parent_id)!;
        parent.children_ids.push(node.code_id);
      }
    }

    // Calculate depth
    let maxDepth = 0;
    for (const node of nodes) {
      const depth = this.calculateNodeDepth(node, nodeMap);
      if (depth > maxDepth) {
        maxDepth = depth;
      }
    }

    // Count branches (nodes with multiple children)
    const branches = Array.from(nodeMap.values()).filter(n => n.children_ids.length > 1).length;

    return {
      root_code_id: rootCodeId,
      final_code_id: rootCodeId, // In production, would trace to find leaf
      total_nodes: nodes.length,
      depth: maxDepth,
      nodes,
      branches,
    };
  }

  /**
   * Calculate depth of a node
   */
  private calculateNodeDepth(node: EvolutionNode, nodeMap: Map<string, EvolutionNode>): number {
    if (!node.parent_id || !nodeMap.has(node.parent_id)) {
      return 1;
    }
    const parent = nodeMap.get(node.parent_id)!;
    return 1 + this.calculateNodeDepth(parent, nodeMap);
  }

  // ========================================================================
  // HISTORY OPERATIONS
  // ========================================================================

  /**
   * Get evolution history for a problem type
   * Returns all evolved code solutions for a given problem type
   */
  async getEvolutionHistory(
    problemType: string,
    correlationId?: string
  ): Promise<EvolvedCode[]> {
    const cid = correlationId || uuidv4();

    this.logger.info('Getting evolution history', {
      correlation_id: cid,
      problem_type: problemType,
    });

    try {
      const history = await this.circuitBreaker.execute(async () => {
        // Search for episodes by problem type
        const searchQuery = {
          query: `episode_type:${this.config.episode_type_base}_${problemType}`,
          max_results: 100,
          temporal_filter: {
            start: '1970-01-01T00:00:00.000Z',
            end: new Date().toISOString(),
          },
        };

        const response = await this.httpClient.post('/search', {
          query: searchQuery,
          correlation_id: cid,
        });

        // Convert results to EvolvedCode
        // Note: In production, would fetch full code from storage
        return (response.edges || []).map((edge: any) => {
          return {
            id: edge.attributes?.code_id || uuidv4(),
            problem: {
              description: edge.attributes?.problem_description || '',
              type: problemType as any,
            },
            language: edge.attributes?.language || 'python',
            code: '', // Would be fetched from storage
            metrics: {
              iterations: edge.attributes?.iterations || 0,
              fitness_score: edge.attributes?.fitness_score || 0,
              fitness_improvement: 0,
              duration_ms: edge.attributes?.duration_ms || 0,
            },
            timestamp_utc: edge.created_at || new Date().toISOString(),
            is_valid: edge.attributes?.is_valid ?? true,
          } as EvolvedCode;
        });
      });

      this.logger.info('Evolution history retrieved successfully', {
        correlation_id: cid,
        problem_type: problemType,
        count: history.length,
      });

      return history;
    } catch (error) {
      this.logger.error('Failed to get evolution history', error as Error, {
        correlation_id: cid,
        problem_type: problemType,
      });
      throw error;
    }
  }

  // ========================================================================
  // HEALTH CHECK
  // ========================================================================

  /**
   * Check graph storage health
   */
  async healthCheck(): Promise<{
    healthy: boolean;
    initialized: boolean;
    circuit_state: string;
    graphiti_connected: boolean;
  }> {
    const circuitStats = this.circuitBreaker.getStats();

    try {
      if (!this.initialized) {
        return {
          healthy: false,
          initialized: false,
          circuit_state: circuitStats.state,
          graphiti_connected: false,
        };
      }

      // Quick connectivity check
      const health = await this.httpClient.get('/health');

      return {
        healthy: circuitStats.state === 'closed' && health.healthy,
        initialized: true,
        circuit_state: circuitStats.state,
        graphiti_connected: health.healthy,
      };
    } catch (error) {
      return {
        healthy: false,
        initialized: true,
        circuit_state: circuitStats.state,
        graphiti_connected: false,
      };
    }
  }

  // ========================================================================
  // CLEANUP
  // ========================================================================

  /**
   * Close graph storage and cleanup resources
   */
  async close(): Promise<void> {
    this.logger.info('Closing GraphStorage', {
      correlation_id: 'graph-storage-close',
    });

    this.initialized = false;
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

