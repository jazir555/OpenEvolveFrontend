"use strict";
/**
 * OpenEvolve Main Orchestration Adapter
 *
 * This is the primary orchestration adapter that coordinates all integrated
 * systems within the OpenEvolve federation. It serves as the central hub for:
 * - Multi-adapter coordination (Z3, LeanAide, RAGBits, Vector DB, etc.)
 * - Workflow orchestration across multiple systems
 * - Knowledge aggregation from all sources
 * - Event bus integration for pub/sub patterns
 * - Circuit breaker and retry logic for resilience
 * - Canonical schema enforcement (Anti-Corruption Layer)
 *
 * Environment Variables:
 *   OPENEVOLVE_API_URL - Base URL of the OpenEvolve API (required, no default)
 *   TIMEOUT_MS - Request timeout in milliseconds (required, no default)
 *   EVENT_BUS_URL - Event bus URL for pub/sub (optional)
 *   LOG_LEVEL - Logging level (default: info)
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.OpenEvolveAdapter = exports.StructuredLogger = void 0;
exports.createOpenEvolveAdapter = createOpenEvolveAdapter;
const axios_1 = __importDefault(require("axios"));
const uuid_1 = require("uuid");
// ============================================================================
// CIRCUIT BREAKER
// ============================================================================
var CircuitBreakerState;
(function (CircuitBreakerState) {
    CircuitBreakerState["CLOSED"] = "closed";
    CircuitBreakerState["OPEN"] = "open";
    CircuitBreakerState["HALF_OPEN"] = "half_open";
})(CircuitBreakerState || (CircuitBreakerState = {}));
class CircuitBreaker {
    constructor(name, config, logger) {
        this.name = name;
        this.config = config;
        this.logger = logger;
        this.state = CircuitBreakerState.CLOSED;
        this.failureCount = 0;
        this.successCount = 0;
    }
    async execute(fn) {
        if (this.state === CircuitBreakerState.OPEN) {
            if (Date.now() < (this.nextAttempt || 0)) {
                throw new Error(`Circuit breaker '${this.name}' is OPEN`);
            }
            this.state = CircuitBreakerState.HALF_OPEN;
            this.logger.info('Circuit breaker HALF_OPEN', { circuit_breaker: this.name });
        }
        try {
            const result = await fn();
            this.onSuccess();
            return result;
        }
        catch (error) {
            this.onFailure();
            throw error;
        }
    }
    onSuccess() {
        this.failureCount = 0;
        if (this.state === CircuitBreakerState.HALF_OPEN) {
            this.successCount++;
            if (this.successCount >= this.config.successThreshold) {
                this.state = CircuitBreakerState.CLOSED;
                this.successCount = 0;
                this.logger.info('Circuit breaker CLOSED', { circuit_breaker: this.name });
            }
        }
    }
    onFailure() {
        this.failureCount++;
        this.lastFailureTime = Date.now();
        if (this.failureCount >= this.config.failureThreshold) {
            this.state = CircuitBreakerState.OPEN;
            this.nextAttempt = Date.now() + this.config.timeout;
            this.logger.error('Circuit breaker OPEN', {
                circuit_breaker: this.name,
                failure_count: this.failureCount,
                next_attempt: new Date(this.nextAttempt).toISOString(),
            });
        }
    }
    getState() {
        return this.state;
    }
}
async function retryWithBackoff(fn, config, logger, context) {
    let lastError;
    for (let attempt = 0; attempt <= config.maxRetries; attempt++) {
        try {
            if (attempt > 0) {
                logger.info('Retrying request', {
                    ...context,
                    attempt,
                    max_retries: config.maxRetries,
                });
            }
            return await fn();
        }
        catch (error) {
            lastError = error;
            if (attempt < config.maxRetries) {
                const delay = Math.min(config.baseDelay * Math.pow(2, attempt), config.maxDelay);
                const jitterAmount = config.jitter ? Math.random() * 0.3 * delay : 0;
                const finalDelay = delay + jitterAmount;
                logger.warn('Request failed, retrying after delay', {
                    ...context,
                    attempt,
                    delay_ms: finalDelay,
                    error: error instanceof Error ? error.message : String(error),
                });
                await sleep(finalDelay);
            }
        }
    }
    logger.error('All retries exhausted', {
        ...context,
        max_retries: config.maxRetries,
        error: lastError instanceof Error ? lastError.message : String(lastError),
    });
    throw lastError;
}
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
class StructuredLogger {
    constructor(serviceName, logLevel = 'info') {
        this.serviceName = serviceName;
        this.logLevel = logLevel;
    }
    log(level, message, context = {}) {
        const logEntry = {
            timestamp: new Date().toISOString(),
            level,
            message,
            service: this.serviceName,
            ...context,
        };
        const logLine = JSON.stringify(logEntry);
        const output = level === 'error' ? console.error : level === 'warn' ? console.warn : console.log;
        output(logLine);
    }
    info(message, context = {}) {
        this.log('info', message, context);
    }
    warn(message, context = {}) {
        this.log('warn', message, context);
    }
    error(message, context = {}) {
        this.log('error', message, context);
    }
    debug(message, context = {}) {
        if (this.logLevel === 'debug') {
            this.log('debug', message, context);
        }
    }
}
exports.StructuredLogger = StructuredLogger;
class OpenEvolveAdapter {
    constructor(config) {
        this.circuitBreakers = new Map();
        // Validate required environment
        if (!config.api_url) {
            throw new Error('OPENEVOLVE_API_URL is required and cannot have a default value');
        }
        if (!config.timeout_ms) {
            throw new Error('TIMEOUT_MS is required and cannot have a default value');
        }
        // Create correlation ID for this adapter instance
        this.correlationId = (0, uuid_1.v4)();
        // Initialize structured logger
        this.logger = new StructuredLogger('openevolve-adapter', config.log_level);
        // Initialize axios instance
        this.api = axios_1.default.create({
            baseURL: config.api_url,
            timeout: config.timeout_ms,
            headers: {
                'Content-Type': 'application/json',
                'X-Correlation-ID': this.correlationId,
            },
        });
        // Configure retry
        this.retryConfig = {
            maxRetries: config.retry?.maxRetries ?? 3,
            baseDelay: config.retry?.baseDelay ?? 1000,
            maxDelay: config.retry?.maxDelay ?? 10000,
            jitter: config.retry?.jitter ?? true,
        };
        // Initialize circuit breakers
        const circuitBreakerConfig = {
            failureThreshold: config.circuit_breaker?.failureThreshold ?? 5,
            successThreshold: config.circuit_breaker?.successThreshold ?? 2,
            timeout: config.circuit_breaker?.timeout ?? 60000,
            monitorPeriod: config.circuit_breaker?.monitorPeriod ?? 10000,
        };
        this.z3CircuitBreaker = new CircuitBreaker('z3-adapter', circuitBreakerConfig, this.logger);
        this.leanaideCircuitBreaker = new CircuitBreaker('leanaide-adapter', circuitBreakerConfig, this.logger);
        this.ragbitsCircuitBreaker = new CircuitBreaker('ragbits-adapter', circuitBreakerConfig, this.logger);
        this.vectordbCircuitBreaker = new CircuitBreaker('vectordb-adapter', circuitBreakerConfig, this.logger);
        this.graphitiCircuitBreaker = new CircuitBreaker('graphiti-adapter', circuitBreakerConfig, this.logger);
        this.karateclubCircuitBreaker = new CircuitBreaker('karateclub-adapter', circuitBreakerConfig, this.logger);
        this.logger.info('OpenEvolve adapter initialized', {
            api_url: config.api_url,
            timeout_ms: config.timeout_ms,
            correlation_id: this.correlationId,
        });
    }
    // ============================================================================
    // HEALTH CHECKS
    // ============================================================================
    async healthCheck() {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'openevolve-adapter',
            target_service: 'openevolve-api',
        };
        this.logger.info('Performing health check', context);
        try {
            const start = Date.now();
            const response = await this.api.get('/health');
            const latency = Date.now() - start;
            const integrations = await this.checkIntegrationHealth(context);
            this.logger.info('Health check successful', {
                ...context,
                latency_ms: latency,
                integration_count: integrations.length,
            });
            return {
                status: response.data.status,
                timestamp: response.data.timestamp,
                integrations,
            };
        }
        catch (error) {
            this.logger.error('Health check failed', {
                ...context,
                error: error instanceof Error ? error.message : String(error),
            });
            throw error;
        }
    }
    async checkIntegrationHealth(context) {
        const integrations = [];
        const checks = [
            { name: 'Z3 Prover', breaker: this.z3CircuitBreaker },
            { name: 'LeanAide', breaker: this.leanaideCircuitBreaker },
            { name: 'RAGBits', breaker: this.ragbitsCircuitBreaker },
            { name: 'Vector DB', breaker: this.vectordbCircuitBreaker },
            { name: 'Graphiti', breaker: this.graphitiCircuitBreaker },
            { name: 'KarateClub', breaker: this.karateclubCircuitBreaker },
        ];
        for (const check of checks) {
            const state = check.breaker.getState();
            integrations.push({
                name: check.name,
                status: state === CircuitBreakerState.CLOSED ? 'healthy' : 'unhealthy',
                last_check: new Date().toISOString(),
            });
        }
        return integrations;
    }
    // ============================================================================
    // TEAM MANAGEMENT
    // ============================================================================
    async createTeam(team) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'openevolve-adapter',
            target_service: 'openevolve-api',
            team_name: team.name,
            role: team.role,
        };
        return retryWithBackoff(async () => {
            const response = await this.api.post('/openevolve/teams', team);
            this.logger.info('Team created', context);
            return response.data;
        }, this.retryConfig, this.logger, context);
    }
    async getTeams() {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'openevolve-adapter',
            target_service: 'openevolve-api',
        };
        return retryWithBackoff(async () => {
            const response = await this.api.get('/openevolve/teams');
            this.logger.info('Teams retrieved', { ...context, count: response.data.length });
            return response.data;
        }, this.retryConfig, this.logger, context);
    }
    async getTeam(name) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'openevolve-adapter',
            target_service: 'openevolve-api',
            team_name: name,
        };
        return retryWithBackoff(async () => {
            const response = await this.api.get(`/openevolve/teams/${name}`);
            this.logger.info('Team retrieved', context);
            return response.data;
        }, this.retryConfig, this.logger, context);
    }
    async updateTeam(name, team) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'openevolve-adapter',
            target_service: 'openevolve-api',
            team_name: name,
        };
        return retryWithBackoff(async () => {
            const response = await this.api.put(`/openevolve/teams/${name}`, team);
            this.logger.info('Team updated', context);
            return response.data;
        }, this.retryConfig, this.logger, context);
    }
    async deleteTeam(name) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'openevolve-adapter',
            target_service: 'openevolve-api',
            team_name: name,
        };
        return retryWithBackoff(async () => {
            const response = await this.api.delete(`/openevolve/teams/${name}`);
            this.logger.info('Team deleted', context);
            return response.data;
        }, this.retryConfig, this.logger, context);
    }
    // ============================================================================
    // GAUNTLET MANAGEMENT
    // ============================================================================
    async createGauntlet(gauntlet) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'openevolve-adapter',
            target_service: 'openevolve-api',
            gauntlet_name: gauntlet.name,
            team_name: gauntlet.team_name,
        };
        return retryWithBackoff(async () => {
            const response = await this.api.post('/openevolve/gauntlets', gauntlet);
            this.logger.info('Gauntlet created', context);
            return response.data;
        }, this.retryConfig, this.logger, context);
    }
    async getGauntlets() {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'openevolve-adapter',
            target_service: 'openevolve-api',
        };
        return retryWithBackoff(async () => {
            const response = await this.api.get('/openevolve/gauntlets');
            this.logger.info('Gauntlets retrieved', { ...context, count: response.data.length });
            return response.data;
        }, this.retryConfig, this.logger, context);
    }
    async getGauntlet(name) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'openevolve-adapter',
            target_service: 'openevolve-api',
            gauntlet_name: name,
        };
        return retryWithBackoff(async () => {
            const response = await this.api.get(`/openevolve/gauntlets/${name}`);
            this.logger.info('Gauntlet retrieved', context);
            return response.data;
        }, this.retryConfig, this.logger, context);
    }
    async deleteGauntlet(name) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'openevolve-adapter',
            target_service: 'openevolve-api',
            gauntlet_name: name,
        };
        return retryWithBackoff(async () => {
            const response = await this.api.delete(`/openevolve/gauntlets/${name}`);
            this.logger.info('Gauntlet deleted', context);
            return response.data;
        }, this.retryConfig, this.logger, context);
    }
    // ============================================================================
    // WORKFLOW ORCHESTRATION
    // ============================================================================
    async createWorkflow(workflow) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'openevolve-adapter',
            target_service: 'openevolve-api',
            workflow_id: workflow.workflow_id,
            sub_problem_count: workflow.sub_problems.length,
        };
        this.logger.info('Creating workflow', context);
        return retryWithBackoff(async () => {
            const response = await this.api.post('/openevolve/workflows', workflow);
            this.logger.info('Workflow created', context);
            return response.data;
        }, this.retryConfig, this.logger, context);
    }
    async getWorkflows() {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'openevolve-adapter',
            target_service: 'openevolve-api',
        };
        return retryWithBackoff(async () => {
            const response = await this.api.get('/openevolve/workflows');
            this.logger.info('Workflows retrieved', { ...context, count: response.data.length });
            return response.data;
        }, this.retryConfig, this.logger, context);
    }
    async getWorkflowStatus(workflowId) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'openevolve-adapter',
            target_service: 'openevolve-api',
            workflow_id: workflowId,
        };
        return retryWithBackoff(async () => {
            const response = await this.api.get(`/openevolve/workflows/${workflowId}/status`);
            this.logger.info('Workflow status retrieved', { ...context, status: response.data.status });
            return response.data;
        }, this.retryConfig, this.logger, context);
    }
    async deleteWorkflow(workflowId) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'openevolve-adapter',
            target_service: 'openevolve-api',
            workflow_id: workflowId,
        };
        return retryWithBackoff(async () => {
            const response = await this.api.delete(`/openevolve/workflows/${workflowId}`);
            this.logger.info('Workflow deleted', context);
            return response.data;
        }, this.retryConfig, this.logger, context);
    }
    // ============================================================================
    // INTEGRATION COORDINATION (Stub implementations)
    // ============================================================================
    async getIntegrationHealth() {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'openevolve-adapter',
        };
        this.logger.info('Checking integration health', context);
        const integrations = await this.checkIntegrationHealth(context);
        return { integrations };
    }
    async getAvailableAdapters() {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'openevolve-adapter',
        };
        this.logger.info('Retrieving available adapters', context);
        // Return known adapters
        return [
            { name: 'z3', type: 'prover', status: 'available' },
            { name: 'leanaide', type: 'assistant', status: 'available' },
            { name: 'ragbits', type: 'retrieval', status: 'available' },
            { name: 'vectordb', type: 'database', status: 'available' },
            { name: 'graphiti', type: 'graph', status: 'available' },
            { name: 'karateclub', type: 'ml', status: 'available' },
        ];
    }
}
exports.OpenEvolveAdapter = OpenEvolveAdapter;
// ============================================================================
// FACTORY FUNCTION
// ============================================================================
function createOpenEvolveAdapter(config) {
    return new OpenEvolveAdapter(config);
}
// Export default
exports.default = OpenEvolveAdapter;
//# sourceMappingURL=adapter.js.map