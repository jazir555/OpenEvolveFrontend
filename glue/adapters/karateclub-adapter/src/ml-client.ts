/**
 * KarateClub ML Client
 *
 * Python-based client for executing KarateClub algorithms.
 * Follows CLAUDE.md principles:
 * - Runtime Truth: Execute against live KarateClub
 * - Configuration Explicitness: All parameters via environment/config
 * - UTC Timestamps: All times in UTC
 * - Circuit Breaker: Handle failures gracefully
 * - Retry Logic: Fewer retries for long ML operations
 *
 * This client spawns Python subprocesses to run KarateClub operations.
 */

import { spawn, ChildProcess } from 'child_process';
import { promisify } from 'util';
import { writeFile, unlink } from 'fs/promises';
import { join } from 'path';
import { v4 as uuidv4 } from 'uuid';

import {
  NodeEmbeddingRequest,
  NodeEmbeddingResponse,
  CommunityDetectionRequest,
  CommunityDetectionResponse,
  GraphEmbeddingRequest,
  GraphEmbeddingResponse,
  GraphStructure,
} from '../../../schemas/karateclub-canonical';

import { getAlgorithmInfo, getDefaultTimeout } from './algorithms';

// Circuit breaker state
enum CircuitState {
  CLOSED = 'closed',     // Normal operation
  OPEN = 'open',         // Failing, reject requests
  HALF_OPEN = 'half_open', // Testing if recovered
}

interface CircuitBreakerConfig {
  failureThreshold: number;
  successThreshold: number;
  timeout: number;       // milliseconds to stay open
  halfOpenMaxCalls: number;
}

interface CircuitBreakerState {
  state: CircuitState;
  failureCount: number;
  successCount: number;
  lastFailureTime: number;
  halfOpenCalls: number;
}

export interface KarateClubClientConfig {
  apiUrl?: string;           // Optional: Python API URL
  pythonPath?: string;       // Path to Python executable
  timeoutMs?: number;        // Default timeout
  maxRetries?: number;       // Maximum retry attempts
  tempDir?: string;          // Temporary directory for graph files
  circuitBreaker?: Partial<CircuitBreakerConfig>;
}

export class KarateClubMLClient {
  private config: Required<KarateClubClientConfig>;
  private circuitBreaker: CircuitBreakerState;
  private circuitBreakerConfig: CircuitBreakerConfig;

  constructor(config: KarateClubClientConfig = {}) {
    this.config = {
      apiUrl: config.apiUrl ?? process.env.KARATECLUB_API_URL ?? 'http://localhost:8000',
      pythonPath: config.pythonPath ?? process.env.PYTHON_PATH ?? 'python3',
      timeoutMs: config.timeoutMs ?? parseInt(process.env.TIMEOUT_MS ?? '60000'),
      maxRetries: config.maxRetries ?? parseInt(process.env.MAX_RETRIES ?? '2'),
      tempDir: config.tempDir ?? process.env.TEMP_DIR ?? '/tmp/karateclub',
      circuitBreaker: config.circuitBreaker ?? {},
    };

    this.circuitBreakerConfig = {
      failureThreshold: this.config.circuitBreaker.failureThreshold ?? 5,
      successThreshold: this.config.circuitBreaker.successThreshold ?? 2,
      timeout: this.config.circuitBreaker.timeout ?? 60000,
      halfOpenMaxCalls: this.config.circuitBreaker.halfOpenMaxCalls ?? 1,
    };

    this.circuitBreaker = {
      state: CircuitState.CLOSED,
      failureCount: 0,
      successCount: 0,
      lastFailureTime: 0,
      halfOpenCalls: 0,
    };
  }

  /**
   * Check circuit breaker before making request
   */
  private async checkCircuitBreaker(): Promise<boolean> {
    const now = Date.now();

    // If OPEN and timeout expired, move to HALF_OPEN
    if (
      this.circuitBreaker.state === CircuitState.OPEN
      && now - this.circuitBreaker.lastFailureTime > this.circuitBreakerConfig.timeout
    ) {
      this.circuitBreaker.state = CircuitState.HALF_OPEN;
      this.circuitBreaker.halfOpenCalls = 0;
      this.log('warn', 'Circuit breaker moved to HALF_OPEN');
    }

    // Reject if OPEN
    if (this.circuitBreaker.state === CircuitState.OPEN) {
      this.log('error', 'Circuit breaker is OPEN - rejecting request');
      return false;
    }

    // Check HALF_OPEN call limit
    if (
      this.circuitBreaker.state === CircuitState.HALF_OPEN
      && this.circuitBreaker.halfOpenCalls >= this.circuitBreakerConfig.halfOpenMaxCalls
    ) {
      this.log('warn', 'Circuit breaker HALF_OPEN call limit exceeded');
      return false;
    }

    return true;
  }

  /**
   * Record success in circuit breaker
   */
  private recordSuccess(): void {
    if (this.circuitBreaker.state === CircuitState.HALF_OPEN) {
      this.circuitBreaker.halfOpenCalls++;
      this.circuitBreaker.successCount++;

      if (this.circuitBreaker.successCount >= this.circuitBreakerConfig.successThreshold) {
        this.circuitBreaker.state = CircuitState.CLOSED;
        this.circuitBreaker.failureCount = 0;
        this.circuitBreaker.successCount = 0;
        this.log('info', 'Circuit breaker moved to CLOSED');
      }
    } else if (this.circuitBreaker.state === CircuitState.CLOSED) {
      this.circuitBreaker.failureCount = 0;
    }
  }

  /**
   * Record failure in circuit breaker
   */
  private recordFailure(): void {
    this.circuitBreaker.failureCount++;
    this.circuitBreaker.lastFailureTime = Date.now();
    this.circuitBreaker.successCount = 0;

    if (
      this.circuitBreaker.state === CircuitState.HALF_OPEN
      || this.circuitBreaker.failureCount >= this.circuitBreakerConfig.failureThreshold
    ) {
      this.circuitBreaker.state = CircuitState.OPEN;
      this.log('error', `Circuit breaker moved to OPEN (failures: ${this.circuitBreaker.failureCount})`);
    }
  }

  /**
   * Structured logging (JSON Lines)
   */
  private log(level: string, msg: string, metadata?: Record<string, any>): void {
    const logEntry = {
      level,
      msg,
      timestamp: new Date().toISOString(),
      source_service: 'karateclub-adapter',
      circuit_state: this.circuitBreaker.state,
      ...metadata,
    };
    console.log(JSON.stringify(logEntry));
  }

  /**
   * Execute Python script with timeout
   */
  private async executePython(
    script: string,
    args: string[],
    timeoutMs: number
  ): Promise<{ stdout: string; stderr: string; exitCode: number | null }> {
    return new Promise((resolve, reject) => {
      const startTime = Date.now();
      const correlationId = uuidv4();

      this.log('info', 'Executing Python script', {
        script,
        args,
        timeout_ms: timeoutMs,
        correlation_id: correlationId,
      });

      const proc = spawn(this.config.pythonPath, [script, ...args], {
        env: {
          ...process.env,
          PYTHONPATH: process.env.PYTHONPATH,
        },
      });

      let stdout = '';
      let stderr = '';

      proc.stdout?.on('data', (data) => {
        stdout += data.toString();
      });

      proc.stderr?.on('data', (data) => {
        stderr += data.toString();
      });

      const timeout = setTimeout(() => {
        proc.kill('SIGKILL');
        const elapsed = Date.now() - startTime;
        this.log('error', 'Python script timeout', {
          timeout_ms: timeoutMs,
          elapsed_ms: elapsed,
          correlation_id: correlationId,
        });
        reject(new Error(`Script execution timeout after ${timeoutMs}ms`));
      }, timeoutMs);

      proc.on('close', (code) => {
        clearTimeout(timeout);
        const elapsed = Date.now() - startTime;
        this.log('info', 'Python script completed', {
          exit_code: code,
          elapsed_ms: elapsed,
          correlation_id: correlationId,
        });
        resolve({ stdout, stderr, exitCode: code });
      });

      proc.on('error', (err) => {
        clearTimeout(timeout);
        this.log('error', 'Python script error', {
          error: err.message,
          correlation_id: correlationId,
        });
        reject(err);
      });
    });
  }

  /**
   * Convert graph structure to temporary JSON file
   */
  private async writeGraphFile(graph: GraphStructure): Promise<string> {
    const filename = join(this.config.tempDir, `graph_${uuidv4()}.json`);
    await writeFile(filename, JSON.stringify(graph), 'utf-8');
    return filename;
  }

  /**
   * Generate Python script for node embedding
   */
  private generateNodeEmbeddingScript(
    algorithm: string,
    parameters: Record<string, any>
  ): string {
    return `
import sys
import json
import networkx as nx
import numpy as np

try:
    from karateclub import ${algorithm}
except ImportError:
    print(json.dumps({"error": f"Algorithm {algorithm} not available"}))
    sys.exit(1)

def main():
    # Load graph
    graph_file = sys.argv[1]
    with open(graph_file, 'r') as f:
        graph_data = json.load(f)

    # Create NetworkX graph
    G = nx.Graph()
    for node in graph_data['nodes']:
        G.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
    for edge in graph_data['edges']:
        G.add_edge(edge['source'], edge['target'], **{k: v for k, v in edge.items() if k not in ['source', 'target']})

    # Initialize model with parameters
    params = ${JSON.stringify(parameters)}
    model = ${algorithm}(**params)

    # Fit model
    model.fit(G)

    # Get embeddings
    embedding_matrix = model.get_embedding()

    # Create output dict
    node_list = list(G.nodes())
    embeddings = {}
    for i, node in enumerate(node_list):
        if i < len(embedding_matrix):
            embeddings[node] = embedding_matrix[i].tolist()

    # Output result
    result = {
        "success": True,
        "embeddings": embeddings,
        "dimensions": len(embedding_matrix[0]) if len(embedding_matrix) > 0 else 0,
        "num_nodes": len(embeddings)
    }

    print(json.dumps(result))

if __name__ == "__main__":
    main()
`;
  }

  /**
   * Generate Python script for community detection
   */
  private generateCommunityScript(algorithm: string, parameters: Record<string, any>): string {
    // Import mapping for community algorithms
    const importMap: Record<string, string> = {
      label_propagation: 'LabelPropagation',
      bigclam: 'BigClam',
      danmf: 'DANMF',
      gemsec: 'GEMSEC',
      edmot: 'EdMot',
      scd: 'SCD',
    };

    const className = importMap[algorithm] || algorithm;

    return `
import sys
import json
import networkx as nx

try:
    from karateclub import ${className}
except ImportError:
    # Try different import paths
    try:
        from karateclub.community_detection.non_overlapping import ${className}
    except ImportError:
        try:
            from karateclub.community_detection.overlapping import ${className}
        except ImportError:
            print(json.dumps({"error": f"Algorithm ${className} not available"}))
            sys.exit(1)

def main():
    # Load graph
    graph_file = sys.argv[1]
    with open(graph_file, 'r') as f:
        graph_data = json.load(f)

    # Create NetworkX graph (undirected for community detection)
    G = nx.Graph()
    for node in graph_data['nodes']:
        G.add_node(node['id'])
    for edge in graph_data['edges']:
        G.add_edge(edge['source'], edge['target'])

    # Initialize model
    params = ${JSON.stringify(parameters)}
    model = ${className}(**params)

    # Fit model
    model.fit(G)

    # Get memberships
    try:
        membership = model.get_memberships()
    except AttributeError:
        # Some algorithms return memberships differently
        membership = {}

    # Count communities
    num_communities = len(set(membership.values())) if membership else 0

    # Calculate community sizes
    community_sizes = {}
    for node_id, comm_id in membership.items():
        community_sizes[str(comm_id)] = community_sizes.get(str(comm_id), 0) + 1

    # Output result
    result = {
        "success": True,
        "memberships": membership,
        "num_communities": num_communities,
        "community_sizes": community_sizes
    }

    print(json.dumps(result))

if __name__ == "__main__":
    main()
`;
  }

  /**
   * Execute with retry logic
   */
  private async executeWithRetry<T>(
    fn: () => Promise<T>,
    maxRetries: number,
    operation: string
  ): Promise<T> {
    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const result = await fn();
        this.recordSuccess();
        return result;
      } catch (error) {
        lastError = error as Error;
        this.log('warn', `${operation} failed (attempt ${attempt + 1}/${maxRetries + 1})`, {
          error: lastError.message,
        });

        if (attempt < maxRetries) {
          // Exponential backoff with jitter
          const delay = Math.min(1000 * Math.pow(2, attempt) + Math.random() * 100, 5000);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }

    this.recordFailure();
    throw lastError;
  }

  /**
   * Generate node embeddings
   */
  async generateNodeEmbeddings(
    request: NodeEmbeddingRequest
  ): Promise<NodeEmbeddingResponse> {
    // Check circuit breaker
    if (!(await this.checkCircuitBreaker())) {
      return {
        success: false,
        error: 'Circuit breaker is OPEN',
        dimensions: 0,
        algorithm: request.algorithm,
        metadata: {
          num_nodes: request.graph.nodes.length,
          training_time_ms: 0,
        },
        timestamp: new Date().toISOString(),
      };
    }

    const startTime = Date.now();

    try {
      // Get algorithm info
      const algoInfo = getAlgorithmInfo(request.algorithm, 'node_embedding');
      if (!algoInfo) {
        throw new Error(`Unknown algorithm: ${request.algorithm}`);
      }

      // Write graph to file
      const graphFile = await this.writeGraphFile(request.graph);

      // Generate Python script
      const script = this.generateNodeEmbeddingScript(
        request.algorithm,
        request.parameters ?? {}
      );

      // Write script to file
      const scriptFile = join(this.config.tempDir, `embedding_${uuidv4()}.py`);
      await writeFile(scriptFile, script, 'utf-8');

      try {
        // Execute with retry
        const result = await this.executeWithRetry(
          async () => {
            return await this.executePython(
              scriptFile,
              [graphFile],
              request.timeout_ms
            );
          },
          this.config.maxRetries,
          'node_embedding'
        );

        // Parse output
        const output = JSON.parse(result.stdout);

        if (!output.success) {
          throw new Error(output.error ?? 'Node embedding failed');
        }

        const trainingTime = Date.now() - startTime;

        this.log('info', 'Node embedding completed', {
          algorithm: request.algorithm,
          num_nodes: output.num_nodes,
          dimensions: output.dimensions,
          training_time_ms: trainingTime,
        });

        return {
          success: true,
          embeddings: output.embeddings,
          dimensions: output.dimensions,
          algorithm: request.algorithm,
          metadata: {
            num_nodes: output.num_nodes,
            training_time_ms: trainingTime,
          },
          timestamp: new Date().toISOString(),
          correlation_id: request.correlation_id,
        };
      } finally {
        // Cleanup
        await unlink(graphFile).catch(() => {});
        await unlink(scriptFile).catch(() => {});
      }
    } catch (error) {
      this.recordFailure();
      const trainingTime = Date.now() - startTime;

      this.log('error', 'Node embedding failed', {
        error: (error as Error).message,
        algorithm: request.algorithm,
      });

      return {
        success: false,
        error: (error as Error).message,
        dimensions: 0,
        algorithm: request.algorithm,
        metadata: {
          num_nodes: request.graph.nodes.length,
          training_time_ms: trainingTime,
        },
        timestamp: new Date().toISOString(),
        correlation_id: request.correlation_id,
      };
    }
  }

  /**
   * Detect communities
   */
  async detectCommunities(
    request: CommunityDetectionRequest
  ): Promise<CommunityDetectionResponse> {
    // Check circuit breaker
    if (!(await this.checkCircuitBreaker())) {
      return {
        success: false,
        error: 'Circuit breaker is OPEN',
        algorithm: request.algorithm,
        metadata: {
          detection_time_ms: 0,
        },
        timestamp: new Date().toISOString(),
      };
    }

    const startTime = Date.now();

    try {
      // Get algorithm info
      const algoInfo = getAlgorithmInfo(request.algorithm, 'community');
      if (!algoInfo) {
        throw new Error(`Unknown algorithm: ${request.algorithm}`);
      }

      // Write graph to file
      const graphFile = await this.writeGraphFile(request.graph);

      // Generate Python script
      const script = this.generateCommunityScript(
        request.algorithm,
        request.parameters ?? {}
      );

      // Write script to file
      const scriptFile = join(this.config.tempDir, `community_${uuidv4()}.py`);
      await writeFile(scriptFile, script, 'utf-8');

      try {
        // Execute with retry (fewer retries for ML operations)
        const result = await this.executeWithRetry(
          async () => {
            return await this.executePython(
              scriptFile,
              [graphFile],
              request.timeout_ms
            );
          },
          Math.max(1, this.config.maxRetries - 1), // Fewer retries for community detection
          'community_detection'
        );

        // Parse output
        const output = JSON.parse(result.stdout);

        if (!output.success) {
          throw new Error(output.error ?? 'Community detection failed');
        }

        const detectionTime = Date.now() - startTime;

        this.log('info', 'Community detection completed', {
          algorithm: request.algorithm,
          num_communities: output.num_communities,
          detection_time_ms: detectionTime,
        });

        return {
          success: true,
          memberships: output.memberships,
          num_communities: output.num_communities,
          community_sizes: output.community_sizes,
          algorithm: request.algorithm,
          metadata: {
            detection_time_ms: detectionTime,
          },
          timestamp: new Date().toISOString(),
          correlation_id: request.correlation_id,
        };
      } finally {
        // Cleanup
        await unlink(graphFile).catch(() => {});
        await unlink(scriptFile).catch(() => {});
      }
    } catch (error) {
      this.recordFailure();
      const detectionTime = Date.now() - startTime;

      this.log('error', 'Community detection failed', {
        error: (error as Error).message,
        algorithm: request.algorithm,
      });

      return {
        success: false,
        error: (error as Error).message,
        algorithm: request.algorithm,
        metadata: {
          detection_time_ms: detectionTime,
        },
        timestamp: new Date().toISOString(),
        correlation_id: request.correlation_id,
      };
    }
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<{ healthy: boolean; version?: string; error?: string }> {
    try {
      const script = `
import sys
import karateclub
print(json.dumps({"version": karateclub.__version__, "healthy": True}))
`;
      const scriptFile = join(this.config.tempDir, `health_${uuidv4()}.py`);
      await writeFile(scriptFile, script, 'utf-8');

      try {
        const result = await this.executePython(scriptFile, [], 5000);
        const output = JSON.parse(result.stdout);
        return output;
      } finally {
        await unlink(scriptFile).catch(() => {});
      }
    } catch (error) {
      return {
        healthy: false,
        error: (error as Error).message,
      };
    }
  }
}
