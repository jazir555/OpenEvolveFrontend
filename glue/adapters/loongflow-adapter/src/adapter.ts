/**
 * LoongFlow Adapter
 *
 * This adapter integrates the LoongFlow PES (Plan-Execute-Summary) evolutionary
 * AI framework into the OpenEvolve federation.
 *
 * Architecture:
 * - LoongFlow is a Python library, not an HTTP API
 * - This adapter communicates with a Python sidecar service via HTTP
 * - The sidecar runs LoongFlow and exposes REST endpoints
 *
 * Environment Variables (Law of Configuration Explicitness):
 *   LOONGFLOW_API_URL - Base URL of LoongFlow sidecar (required)
 *   LOONGFLOW_TIMEOUT_MS - Request timeout in ms (default: 30000)
 *   LOONGFLOW_MAX_RETRIES - Max retry attempts (default: 3)
 *   LOG_LEVEL - Logging level (default: info)
 *
 * Following Federation Constitution:
 * - Law of Air Gap: No imports from core-projects/LoongFlow
 * - Law of Runtime Truth: All operations verified via probes
 * - Law of Idempotency: All operations safe to retry
 * - Law of UTC: All timestamps in UTC ISO-8601
 * - Law of Configuration Explicitness: Required env vars crash service
 * - Observability: Structured JSON logging with correlation_id
 */

import axios, { AxiosInstance } from 'axios';
import { v4 as uuidv4 } from 'uuid';

// Import shared utilities from lib
// @ts-ignore - lib is compiled separately
import { Logger } from '../../../lib/logger';
// @ts-ignore - lib is compiled separately
import { CircuitBreaker } from '../../../lib/circuit-breaker';
// @ts-ignore - lib is compiled separately
import { retryWithBackoff } from '../../../lib/retry';

// ============================================================================
// TYPE DEFINITIONS - Canonical Schemas
// ============================================================================

/**
 * PES Agent configuration
 */
export interface PESAgentConfig {
  task: string;
  max_iterations: number;
  target_score: number;
  concurrency: number;
  workspace_path?: string;
  initial_code?: string;
  initial_score?: number;
  initial_evaluation?: string;
  checkpoint_path?: string;
  metadata?: Record<string, any>;
}

/**
 * PES Agent state
 */
export interface PESAgentState {
  agent_id: string;
  status: 'idle' | 'running' | 'interrupted' | 'completed' | 'failed';
  current_iteration: number;
  max_iterations: number;
  target_score: number;
  best_score: number;
  start_time: string;
  end_time?: string;
  completion_count: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_cost: number;
}

/**
 * Solution from LoongFlow evolutionary database
 */
export interface Solution {
  solution_id: string;
  solution: string;
  evaluation: string;
  score: number;
  parent_id?: string;
  island_id: number;
  iteration: number;
  generate_plan: string;
  summary: string;
  created_at: string;
}

/**
 * Evolutionary database status
 */
export interface DatabaseStatus {
  global_status: {
    current_iteration: number;
    best_score: number;
    total_solutions: number;
  };
  island_status?: Record<number, {
    best_score: number;
    total_solutions: number;
  }>;
}

/**
 * Checkpoint information
 */
export interface CheckpointInfo {
  checkpoint_path: string;
  tag: string;
  created_at: string;
  iteration: number;
  completion_count: number;
}

/**
 * Problem submission request
 */
export interface SubmitProblemRequest {
  task: string;
  max_iterations?: number;
  target_score?: number;
  concurrency?: number;
  initial_code?: string;
  initial_score?: number;
  initial_evaluation?: string;
  workspace_path?: string;
  metadata?: Record<string, any>;
}

/**
 * Problem submission response
 */
export interface SubmitProblemResponse {
  agent_id: string;
  status: string;
  message: string;
}

/**
 * Execution result
 */
export interface ExecutionResult {
  agent_id: string;
  status: string;
  final_solution?: string;
  final_score?: number;
  best_solutions?: Solution[];
  total_iterations: number;
  total_tokens: number;
  total_cost: number;
  was_interrupted: boolean;
  start_time: string;
  end_time: string;
}

// ============================================================================
// CIRCUIT BREAKER WRAPPER
// ============================================================================

/**
 * Wrapper for executing operations through circuit breaker with retry
 */
async function executeWithResilience<T>(
  operation: string,
  circuitBreaker: CircuitBreaker,
  fn: () => Promise<T>,
  logger: Logger,
  context: Record<string, any>
): Promise<T> {
  return retryWithBackoff(
    async () => {
      return circuitBreaker.execute(async () => {
        logger.debug(`Executing ${operation}`, context);
        return await fn();
      });
    },
    {
      max_retries: 3,
      base_delay_ms: 1000,
      max_delay_ms: 10000,
      jitter_ms: 500,
      onRetry: (attempt, error) => {
        logger.warn(`Retrying ${operation} after error`, {
          ...context,
          attempt,
          error_message: error.message,
        });
      },
    }
  );
}

// ============================================================================
// MAIN ADAPTER CLASS
// ============================================================================

export interface LoongFlowAdapterConfig {
  api_url: string;
  timeout_ms: number;
  max_retries?: number;
  log_level?: string;
  circuit_breaker?: Partial<{
    threshold: number;
    timeout_ms: number;
    reset_timeout_ms: number;
  }>;
}

export class LoongFlowAdapter {
  private readonly api: AxiosInstance;
  private readonly logger: Logger;
  private readonly circuitBreaker: CircuitBreaker;
  private readonly correlationId: string;

  constructor(config: LoongFlowAdapterConfig) {
    // Validate required environment (Law of Configuration Explicitness)
    if (!config.api_url) {
      throw new Error('LOONGFLOW_API_URL is required and cannot have a default value');
    }
    if (!config.timeout_ms) {
      throw new Error('LOONGFLOW_TIMEOUT_MS is required and cannot have a default value');
    }

    // Create correlation ID for this adapter instance
    this.correlationId = uuidv4();

    // Initialize structured logger
    this.logger = new Logger('loongflow-adapter');

    // Initialize axios instance
    this.api = axios.create({
      baseURL: config.api_url,
      timeout: config.timeout_ms,
      headers: {
        'Content-Type': 'application/json',
        'X-Correlation-ID': this.correlationId,
      },
    });

    // Initialize circuit breaker
    this.circuitBreaker = new CircuitBreaker({
      threshold: config.circuit_breaker?.threshold ?? 5,
      timeout_ms: config.circuit_breaker?.timeout_ms ?? 60000,
      reset_timeout_ms: config.circuit_breaker?.reset_timeout_ms ?? 10000,
      onStateChange: (oldState, newState) => {
        this.logger.warn('Circuit breaker state changed', {
          correlation_id: this.correlationId,
          old_state: oldState,
          new_state: newState,
        });
      },
    });

    this.logger.info('LoongFlow adapter initialized', {
      correlation_id: this.correlationId,
      api_url: config.api_url,
      timeout_ms: config.timeout_ms,
    });
  }

  // ============================================================================
  // HEALTH CHECKS
  // ============================================================================

  /**
   * Check if LoongFlow sidecar is healthy
   */
  async healthCheck(): Promise<{
    status: string;
    timestamp: string;
    version?: string;
  }> {
    const context = {
      correlation_id: this.correlationId,
      source_service: 'loongflow-adapter',
      target_service: 'loongflow-sidecar',
    };

    return executeWithResilience(
      'healthCheck',
      this.circuitBreaker,
      async () => {
        const response = await this.api.get('/health');
        this.logger.info('Health check successful', context);
        return response.data;
      },
      this.logger,
      context
    );
  }

  // ============================================================================
  // PES AGENT MANAGEMENT
  // ============================================================================

  /**
   * Submit a problem to the PES Agent for evolution
   * This is idempotent - submitting the same task_id will return the existing agent
   */
  async submitProblem(request: SubmitProblemRequest): Promise<SubmitProblemResponse> {
    const context = {
      correlation_id: this.correlationId,
      source_service: 'loongflow-adapter',
      target_service: 'loongflow-sidecar',
      task: request.task.substring(0, 100), // Truncate for logging
    };

    this.logger.info('Submitting problem to PES Agent', context);

    return executeWithResilience(
      'submitProblem',
      this.circuitBreaker,
      async () => {
        const response = await this.api.post('/pes/submit', request);
        this.logger.info('Problem submitted successfully', {
          ...context,
          agent_id: response.data.agent_id,
        });
        return response.data;
      },
      this.logger,
      context
    );
  }

  /**
   * Get the current state of a PES Agent
   */
  async getAgentState(agentId: string): Promise<PESAgentState> {
    const context = {
      correlation_id: this.correlationId,
      source_service: 'loongflow-adapter',
      target_service: 'loongflow-sidecar',
      agent_id: agentId,
    };

    return executeWithResilience(
      'getAgentState',
      this.circuitBreaker,
      async () => {
        const response = await this.api.get(`/pes/agents/${agentId}/state`);
        this.logger.debug('Agent state retrieved', {
          ...context,
          status: response.data.status,
          iteration: response.data.current_iteration,
        });
        return response.data;
      },
      this.logger,
      context
    );
  }

  /**
   * Interrupt a running PES Agent
   * This is idempotent - interrupting an already stopped agent is a no-op
   */
  async interruptAgent(agentId: string): Promise<{ message: string }> {
    const context = {
      correlation_id: this.correlationId,
      source_service: 'loongflow-adapter',
      target_service: 'loongflow-sidecar',
      agent_id: agentId,
    };

    this.logger.info('Interrupting PES Agent', context);

    return executeWithResilience(
      'interruptAgent',
      this.circuitBreaker,
      async () => {
        const response = await this.api.post(`/pes/agents/${agentId}/interrupt`);
        this.logger.info('Agent interrupted successfully', context);
        return response.data;
      },
      this.logger,
      context
    );
  }

  /**
   * Get the final execution result of a PES Agent
   */
  async getExecutionResult(agentId: string): Promise<ExecutionResult> {
    const context = {
      correlation_id: this.correlationId,
      source_service: 'loongflow-adapter',
      target_service: 'loongflow-sidecar',
      agent_id: agentId,
    };

    return executeWithResilience(
      'getExecutionResult',
      this.circuitBreaker,
      async () => {
        const response = await this.api.get(`/pes/agents/${agentId}/result`);
        this.logger.info('Execution result retrieved', {
          ...context,
          status: response.data.status,
          final_score: response.data.final_score,
        });
        return response.data;
      },
      this.logger,
      context
    );
  }

  // ============================================================================
  // EVOLUTIONARY DATABASE OPERATIONS
  // ============================================================================

  /**
   * Sample a solution from the evolutionary database
   */
  async sampleSolution(islandId?: number): Promise<Solution | {}> {
    const context = {
      correlation_id: this.correlationId,
      source_service: 'loongflow-adapter',
      target_service: 'loongflow-sidecar',
      island_id: islandId,
    };

    return executeWithResilience(
      'sampleSolution',
      this.circuitBreaker,
      async () => {
        const params = islandId !== undefined ? { island_id: islandId } : {};
        const response = await this.api.get('/database/sample', { params });
        this.logger.debug('Solution sampled', context);
        return response.data;
      },
      this.logger,
      context
    );
  }

  /**
   * Add a solution to the evolutionary database
   * This is idempotent if solution_id is the same
   */
  async addSolution(solution: Omit<Solution, 'solution_id' | 'created_at'>): Promise<string> {
    const context = {
      correlation_id: this.correlationId,
      source_service: 'loongflow-adapter',
      target_service: 'loongflow-sidecar',
      score: solution.score,
      island_id: solution.island_id,
    };

    this.logger.info('Adding solution to database', context);

    return executeWithResilience(
      'addSolution',
      this.circuitBreaker,
      async () => {
        const response = await this.api.post('/database/solutions', solution);
        this.logger.info('Solution added successfully', {
          ...context,
          solution_id: response.data.solution_id,
        });
        return response.data.solution_id;
      },
      this.logger,
      context
    );
  }

  /**
   * Update a solution in the evolutionary database
   */
  async updateSolution(solutionId: string, updates: Partial<Solution>): Promise<string> {
    const context = {
      correlation_id: this.correlationId,
      source_service: 'loongflow-adapter',
      target_service: 'loongflow-sidecar',
      solution_id: solutionId,
    };

    this.logger.info('Updating solution in database', context);

    return executeWithResilience(
      'updateSolution',
      this.circuitBreaker,
      async () => {
        const response = await this.api.put(`/database/solutions/${solutionId}`, updates);
        this.logger.info('Solution updated successfully', context);
        return response.data.solution_id;
      },
      this.logger,
      context
    );
  }

  /**
   * Get the best solutions from the evolutionary database
   */
  async getBestSolutions(islandId?: number, topK?: number): Promise<Solution[]> {
    const context = {
      correlation_id: this.correlationId,
      source_service: 'loongflow-adapter',
      target_service: 'loongflow-sidecar',
      island_id: islandId,
      top_k: topK,
    };

    return executeWithResilience(
      'getBestSolutions',
      this.circuitBreaker,
      async () => {
        const params: any = {};
        if (islandId !== undefined) params.island_id = islandId;
        if (topK !== undefined) params.top_k = topK;

        const response = await this.api.get('/database/best', { params });
        this.logger.info('Best solutions retrieved', {
          ...context,
          count: response.data.length,
        });
        return response.data;
      },
      this.logger,
      context
    );
  }

  /**
   * Get database status
   */
  async getDatabaseStatus(islandId?: number): Promise<DatabaseStatus> {
    const context = {
      correlation_id: this.correlationId,
      source_service: 'loongflow-adapter',
      target_service: 'loongflow-sidecar',
      island_id: islandId,
    };

    return executeWithResilience(
      'getDatabaseStatus',
      this.circuitBreaker,
      async () => {
        const params = islandId !== undefined ? { island_id: islandId } : {};
        const response = await this.api.get('/database/status', { params });
        this.logger.debug('Database status retrieved', context);
        return response.data;
      },
      this.logger,
      context
    );
  }

  // ============================================================================
  // CHECKPOINT OPERATIONS
  // ============================================================================

  /**
   * Save a checkpoint of the current evolutionary state
   */
  async saveCheckpoint(checkpointPath: string, tag: string): Promise<CheckpointInfo> {
    const context = {
      correlation_id: this.correlationId,
      source_service: 'loongflow-adapter',
      target_service: 'loongflow-sidecar',
      checkpoint_path: checkpointPath,
      tag,
    };

    this.logger.info('Saving checkpoint', context);

    return executeWithResilience(
      'saveCheckpoint',
      this.circuitBreaker,
      async () => {
        const response = await this.api.post('/database/checkpoints', {
          checkpoint_path: checkpointPath,
          tag,
        });
        this.logger.info('Checkpoint saved successfully', context);
        return response.data;
      },
      this.logger,
      context
    );
  }

  /**
   * Load a checkpoint
   */
  async loadCheckpoint(checkpointPath: string): Promise<{ message: string }> {
    const context = {
      correlation_id: this.correlationId,
      source_service: 'loongflow-adapter',
      target_service: 'loongflow-sidecar',
      checkpoint_path: checkpointPath,
    };

    this.logger.info('Loading checkpoint', context);

    return executeWithResilience(
      'loadCheckpoint',
      this.circuitBreaker,
      async () => {
        const response = await this.api.post('/database/checkpoints/load', {
          checkpoint_path: checkpointPath,
        });
        this.logger.info('Checkpoint loaded successfully', context);
        return response.data;
      },
      this.logger,
      context
    );
  }

  /**
   * List available checkpoints
   */
  async listCheckpoints(checkpointPath: string): Promise<CheckpointInfo[]> {
    const context = {
      correlation_id: this.correlationId,
      source_service: 'loongflow-adapter',
      target_service: 'loongflow-sidecar',
      checkpoint_path: checkpointPath,
    };

    return executeWithResilience(
      'listCheckpoints',
      this.circuitBreaker,
      async () => {
        const response = await this.api.get('/database/checkpoints', {
          params: { checkpoint_path: checkpointPath },
        });
        this.logger.info('Checkpoints listed', {
          ...context,
          count: response.data.length,
        });
        return response.data;
      },
      this.logger,
      context
    );
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  /**
   * Get circuit breaker state (for monitoring)
   */
  getCircuitBreakerState() {
    return this.circuitBreaker.getStats();
  }

  /**
   * Manually reset circuit breaker (for recovery)
   */
  resetCircuitBreaker(): void {
    this.circuitBreaker.reset();
    this.logger.info('Circuit breaker manually reset', {
      correlation_id: this.correlationId,
    });
  }
}

// ============================================================================
// FACTORY FUNCTION
// ============================================================================

export function createLoongFlowAdapter(config: LoongFlowAdapterConfig): LoongFlowAdapter {
  return new LoongFlowAdapter(config);
}

// Export default
export default LoongFlowAdapter;
