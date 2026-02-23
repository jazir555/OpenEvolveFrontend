"use strict";
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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.LoongFlowAdapter = void 0;
exports.createLoongFlowAdapter = createLoongFlowAdapter;
const axios_1 = __importDefault(require("axios"));
const uuid_1 = require("uuid");
// Import shared utilities from lib
// @ts-ignore - lib is compiled separately
const logger_1 = require("../../../lib/logger");
// @ts-ignore - lib is compiled separately
const circuit_breaker_1 = require("../../../lib/circuit-breaker");
// @ts-ignore - lib is compiled separately
const retry_1 = require("../../../lib/retry");
// ============================================================================
// CIRCUIT BREAKER WRAPPER
// ============================================================================
/**
 * Wrapper for executing operations through circuit breaker with retry
 */
async function executeWithResilience(operation, circuitBreaker, fn, logger, context) {
    return (0, retry_1.retryWithBackoff)(async () => {
        return circuitBreaker.execute(async () => {
            logger.debug(`Executing ${operation}`, context);
            return await fn();
        });
    }, {
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
    });
}
class LoongFlowAdapter {
    constructor(config) {
        // Validate required environment (Law of Configuration Explicitness)
        if (!config.api_url) {
            throw new Error('LOONGFLOW_API_URL is required and cannot have a default value');
        }
        if (!config.timeout_ms) {
            throw new Error('LOONGFLOW_TIMEOUT_MS is required and cannot have a default value');
        }
        // Create correlation ID for this adapter instance
        this.correlationId = (0, uuid_1.v4)();
        // Initialize structured logger
        this.logger = new logger_1.Logger('loongflow-adapter');
        // Initialize axios instance
        this.api = axios_1.default.create({
            baseURL: config.api_url,
            timeout: config.timeout_ms,
            headers: {
                'Content-Type': 'application/json',
                'X-Correlation-ID': this.correlationId,
            },
        });
        // Initialize circuit breaker
        this.circuitBreaker = new circuit_breaker_1.CircuitBreaker({
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
    async healthCheck() {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'loongflow-adapter',
            target_service: 'loongflow-sidecar',
        };
        return executeWithResilience('healthCheck', this.circuitBreaker, async () => {
            const response = await this.api.get('/health');
            this.logger.info('Health check successful', context);
            return response.data;
        }, this.logger, context);
    }
    // ============================================================================
    // PES AGENT MANAGEMENT
    // ============================================================================
    /**
     * Submit a problem to the PES Agent for evolution
     * This is idempotent - submitting the same task_id will return the existing agent
     */
    async submitProblem(request) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'loongflow-adapter',
            target_service: 'loongflow-sidecar',
            task: request.task.substring(0, 100), // Truncate for logging
        };
        this.logger.info('Submitting problem to PES Agent', context);
        return executeWithResilience('submitProblem', this.circuitBreaker, async () => {
            const response = await this.api.post('/pes/submit', request);
            this.logger.info('Problem submitted successfully', {
                ...context,
                agent_id: response.data.agent_id,
            });
            return response.data;
        }, this.logger, context);
    }
    /**
     * Get the current state of a PES Agent
     */
    async getAgentState(agentId) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'loongflow-adapter',
            target_service: 'loongflow-sidecar',
            agent_id: agentId,
        };
        return executeWithResilience('getAgentState', this.circuitBreaker, async () => {
            const response = await this.api.get(`/pes/agents/${agentId}/state`);
            this.logger.debug('Agent state retrieved', {
                ...context,
                status: response.data.status,
                iteration: response.data.current_iteration,
            });
            return response.data;
        }, this.logger, context);
    }
    /**
     * Interrupt a running PES Agent
     * This is idempotent - interrupting an already stopped agent is a no-op
     */
    async interruptAgent(agentId) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'loongflow-adapter',
            target_service: 'loongflow-sidecar',
            agent_id: agentId,
        };
        this.logger.info('Interrupting PES Agent', context);
        return executeWithResilience('interruptAgent', this.circuitBreaker, async () => {
            const response = await this.api.post(`/pes/agents/${agentId}/interrupt`);
            this.logger.info('Agent interrupted successfully', context);
            return response.data;
        }, this.logger, context);
    }
    /**
     * Get the final execution result of a PES Agent
     */
    async getExecutionResult(agentId) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'loongflow-adapter',
            target_service: 'loongflow-sidecar',
            agent_id: agentId,
        };
        return executeWithResilience('getExecutionResult', this.circuitBreaker, async () => {
            const response = await this.api.get(`/pes/agents/${agentId}/result`);
            this.logger.info('Execution result retrieved', {
                ...context,
                status: response.data.status,
                final_score: response.data.final_score,
            });
            return response.data;
        }, this.logger, context);
    }
    // ============================================================================
    // EVOLUTIONARY DATABASE OPERATIONS
    // ============================================================================
    /**
     * Sample a solution from the evolutionary database
     */
    async sampleSolution(islandId) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'loongflow-adapter',
            target_service: 'loongflow-sidecar',
            island_id: islandId,
        };
        return executeWithResilience('sampleSolution', this.circuitBreaker, async () => {
            const params = islandId !== undefined ? { island_id: islandId } : {};
            const response = await this.api.get('/database/sample', { params });
            this.logger.debug('Solution sampled', context);
            return response.data;
        }, this.logger, context);
    }
    /**
     * Add a solution to the evolutionary database
     * This is idempotent if solution_id is the same
     */
    async addSolution(solution) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'loongflow-adapter',
            target_service: 'loongflow-sidecar',
            score: solution.score,
            island_id: solution.island_id,
        };
        this.logger.info('Adding solution to database', context);
        return executeWithResilience('addSolution', this.circuitBreaker, async () => {
            const response = await this.api.post('/database/solutions', solution);
            this.logger.info('Solution added successfully', {
                ...context,
                solution_id: response.data.solution_id,
            });
            return response.data.solution_id;
        }, this.logger, context);
    }
    /**
     * Update a solution in the evolutionary database
     */
    async updateSolution(solutionId, updates) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'loongflow-adapter',
            target_service: 'loongflow-sidecar',
            solution_id: solutionId,
        };
        this.logger.info('Updating solution in database', context);
        return executeWithResilience('updateSolution', this.circuitBreaker, async () => {
            const response = await this.api.put(`/database/solutions/${solutionId}`, updates);
            this.logger.info('Solution updated successfully', context);
            return response.data.solution_id;
        }, this.logger, context);
    }
    /**
     * Get the best solutions from the evolutionary database
     */
    async getBestSolutions(islandId, topK) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'loongflow-adapter',
            target_service: 'loongflow-sidecar',
            island_id: islandId,
            top_k: topK,
        };
        return executeWithResilience('getBestSolutions', this.circuitBreaker, async () => {
            const params = {};
            if (islandId !== undefined)
                params.island_id = islandId;
            if (topK !== undefined)
                params.top_k = topK;
            const response = await this.api.get('/database/best', { params });
            this.logger.info('Best solutions retrieved', {
                ...context,
                count: response.data.length,
            });
            return response.data;
        }, this.logger, context);
    }
    /**
     * Get database status
     */
    async getDatabaseStatus(islandId) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'loongflow-adapter',
            target_service: 'loongflow-sidecar',
            island_id: islandId,
        };
        return executeWithResilience('getDatabaseStatus', this.circuitBreaker, async () => {
            const params = islandId !== undefined ? { island_id: islandId } : {};
            const response = await this.api.get('/database/status', { params });
            this.logger.debug('Database status retrieved', context);
            return response.data;
        }, this.logger, context);
    }
    // ============================================================================
    // CHECKPOINT OPERATIONS
    // ============================================================================
    /**
     * Save a checkpoint of the current evolutionary state
     */
    async saveCheckpoint(checkpointPath, tag) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'loongflow-adapter',
            target_service: 'loongflow-sidecar',
            checkpoint_path: checkpointPath,
            tag: tag,
        };
        this.logger.info('Saving checkpoint', context);
        return executeWithResilience('saveCheckpoint', this.circuitBreaker, async () => {
            const response = await this.api.post('/database/checkpoints', {
                checkpoint_path: checkpointPath,
                tag: tag,
            });
            this.logger.info('Checkpoint saved successfully', context);
            return response.data;
        }, this.logger, context);
    }
    /**
     * Load a checkpoint
     */
    async loadCheckpoint(checkpointPath) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'loongflow-adapter',
            target_service: 'loongflow-sidecar',
            checkpoint_path: checkpointPath,
        };
        this.logger.info('Loading checkpoint', context);
        return executeWithResilience('loadCheckpoint', this.circuitBreaker, async () => {
            const response = await this.api.post('/database/checkpoints/load', {
                checkpoint_path: checkpointPath,
            });
            this.logger.info('Checkpoint loaded successfully', context);
            return response.data;
        }, this.logger, context);
    }
    /**
     * List available checkpoints
     */
    async listCheckpoints(checkpointPath) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'loongflow-adapter',
            target_service: 'loongflow-sidecar',
            checkpoint_path: checkpointPath,
        };
        return executeWithResilience('listCheckpoints', this.circuitBreaker, async () => {
            const response = await this.api.get('/database/checkpoints', {
                params: { checkpoint_path: checkpointPath },
            });
            this.logger.info('Checkpoints listed', {
                ...context,
                count: response.data.length,
            });
            return response.data;
        }, this.logger, context);
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
    resetCircuitBreaker() {
        this.circuitBreaker.reset();
        this.logger.info('Circuit breaker manually reset', {
            correlation_id: this.correlationId,
        });
    }
}
exports.LoongFlowAdapter = LoongFlowAdapter;
// ============================================================================
// FACTORY FUNCTION
// ============================================================================
function createLoongFlowAdapter(config) {
    return new LoongFlowAdapter(config);
}
// Export default
exports.default = LoongFlowAdapter;
//# sourceMappingURL=adapter.js.map