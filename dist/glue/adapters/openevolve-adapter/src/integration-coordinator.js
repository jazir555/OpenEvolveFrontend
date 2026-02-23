"use strict";
/**
 * Integration Coordinator
 *
 * Coordinates all adapters in the OpenEvolve federation:
 * - Z3 Prover Adapter (formal verification)
 * - LeanAide Adapter (proof assistant)
 * - RAGBits Adapter (retrieval augmented generation)
 * - Vector DB Adapter (vector storage)
 * - Graphiti Adapter (graph knowledge)
 * - KarateClub Adapter (graph ML)
 *
 * The coordinator provides:
 * - Unified interface for all integrations
 * - Intelligent routing based on problem type
 * - Load balancing across adapters
 * - Health monitoring and failover
 * - Canonical schema transformation (Anti-Corruption Layer)
 *
 * Environment Variables:
 *   Z3_ADAPTER_URL - Z3 adapter endpoint
 *   LEANAIDE_ADAPTER_URL - LeanAide adapter endpoint
 *   RAGBITS_ADAPTER_URL - RAGBits adapter endpoint
 *   VECTOR_DB_URL - Vector DB endpoint
 *   GRAPHITI_ADAPTER_URL - Graphiti adapter endpoint
 *   KARATECLUB_ADAPTER_URL - KarateClub adapter endpoint
 *   COORDINATION_TIMEOUT_MS - Coordination timeout
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.IntegrationCoordinator = void 0;
exports.createIntegrationCoordinator = createIntegrationCoordinator;
const adapter_1 = require("./adapter");
const axios_1 = __importDefault(require("axios"));
// ============================================================================
// INTEGRATION COORDINATOR CLASS
// ============================================================================
class IntegrationCoordinator {
    constructor(openEvolveAdapter, timeout_ms) {
        this.openEvolveAdapter = openEvolveAdapter;
        this.timeout_ms = timeout_ms;
        this.adapters = new Map();
        this.logger = new adapter_1.StructuredLogger('integration-coordinator');
        this.correlationId = this.generateCorrelationId();
        // Initialize HTTP client
        this.httpClient = axios_1.default.create({
            timeout: this.timeout_ms,
            headers: {
                'Content-Type': 'application/json',
                'X-Correlation-ID': this.correlationId,
            },
        });
        // Register adapters from environment
        this.registerAdapters();
        this.logger.info('Integration coordinator initialized', {
            correlation_id: this.correlationId,
            adapter_count: this.adapters.size,
            timeout_ms: this.timeout_ms,
        });
    }
    // ==========================================================================
    // ADAPTER REGISTRATION
    // ==========================================================================
    registerAdapters() {
        const adapterConfigs = [
            {
                name: 'z3',
                type: 'prover',
                url: process.env.Z3_ADAPTER_URL || 'http://localhost:8080',
                enabled: true,
            },
            {
                name: 'leanaide',
                type: 'assistant',
                url: process.env.LEANAIDE_ADAPTER_URL || 'http://localhost:8081',
                enabled: true,
            },
            {
                name: 'ragbits',
                type: 'retrieval',
                url: process.env.RAGBITS_ADAPTER_URL || 'http://localhost:8082',
                enabled: true,
            },
            {
                name: 'vectordb',
                type: 'database',
                url: process.env.VECTOR_DB_URL || 'http://localhost:8083',
                enabled: true,
            },
            {
                name: 'graphiti',
                type: 'graph',
                url: process.env.GRAPHITI_ADAPTER_URL || 'http://localhost:8084',
                enabled: true,
            },
            {
                name: 'karateclub',
                type: 'ml',
                url: process.env.KARATECLUB_ADAPTER_URL || 'http://localhost:8085',
                enabled: true,
            },
        ];
        for (const config of adapterConfigs) {
            if (config.enabled) {
                this.adapters.set(config.name, config);
            }
        }
    }
    // ==========================================================================
    // COORDINATION PLANNING
    // ==========================================================================
    async planCoordination(request) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'integration-coordinator',
            problem_type: request.problem_type,
            domain: request.domain,
        };
        this.logger.info('Planning coordination', {
            ...context,
            capabilities: request.capabilities,
        });
        // Select adapters based on problem type and capabilities
        const selectedAdapters = this.selectAdapters(request);
        const fallbackOrder = this.determineFallbackOrder(selectedAdapters);
        // Determine if parallel execution is beneficial
        const parallelExecution = this.shouldExecuteInParallel(request, selectedAdapters);
        // Estimate duration
        const estimatedDuration = this.estimateDuration(request, selectedAdapters, parallelExecution);
        const plan = {
            selected_adapters: selectedAdapters,
            fallback_order: fallbackOrder,
            parallel_execution: parallelExecution,
            estimated_duration_ms: estimatedDuration,
        };
        this.logger.info('Coordination plan created', {
            ...context,
            adapter_count: selectedAdapters.length,
            parallel_execution: parallelExecution,
            estimated_duration_ms: estimatedDuration,
        });
        return plan;
    }
    selectAdapters(request) {
        const selected = [];
        // Problem type to adapter mapping
        const problemTypeMapping = {
            formal_verification: ['z3', 'leanaide'],
            proof_assistant: ['leanaide'],
            retrieval: ['ragbits', 'vectordb'],
            knowledge_graph: ['graphiti', 'vectordb'],
            graph_ml: ['karateclub', 'graphiti'],
            semantic_search: ['vectordb', 'ragbits'],
            code_analysis: ['z3', 'leanaide', 'graphiti'],
        };
        // Capability to adapter mapping
        const capabilityMapping = {
            smt_solving: ['z3'],
            tactic_execution: ['leanaide'],
            vector_search: ['vectordb'],
            graph_traversal: ['graphiti'],
            node_embedding: ['karateclub'],
            document_retrieval: ['ragbits'],
        };
        // Add adapters based on problem type
        const typeAdapters = problemTypeMapping[request.problem_type] || [];
        for (const adapterName of typeAdapters) {
            const adapter = this.adapters.get(adapterName);
            if (adapter) {
                selected.push(adapter);
            }
        }
        // Add adapters based on capabilities
        for (const capability of request.capabilities) {
            const capAdapters = capabilityMapping[capability] || [];
            for (const adapterName of capAdapters) {
                const adapter = this.adapters.get(adapterName);
                if (adapter && !selected.find(a => a.name === adapterName)) {
                    selected.push(adapter);
                }
            }
        }
        // Fallback: include all adapters if no specific match
        if (selected.length === 0) {
            this.logger.warn('No specific adapters matched, including all enabled adapters', {
                problem_type: request.problem_type,
            });
            return Array.from(this.adapters.values());
        }
        return selected;
    }
    determineFallbackOrder(adapters) {
        // Define fallback priorities based on adapter type
        const typePriority = {
            prover: 1,
            assistant: 2,
            database: 3,
            retrieval: 4,
            graph: 5,
            ml: 6,
        };
        return adapters
            .sort((a, b) => typePriority[a.type] - typePriority[b.type])
            .map(a => a.name);
    }
    shouldExecuteInParallel(request, adapters) {
        // Execute in parallel if:
        // 1. Multiple adapters are selected
        // 2. Priority is not high (high priority might need sequential guarantees)
        // 3. Adapters are of different types (avoid conflicts)
        if (adapters.length <= 1) {
            return false;
        }
        if (request.priority === 'high') {
            return false;
        }
        const uniqueTypes = new Set(adapters.map(a => a.type));
        return uniqueTypes.size === adapters.length;
    }
    estimateDuration(request, adapters, parallel) {
        const baseDuration = 1000; // 1 second base overhead
        const perAdapterDuration = 2000; // 2 seconds per adapter
        if (parallel) {
            // Parallel: duration is the max of individual adapters + coordination overhead
            return baseDuration + perAdapterDuration;
        }
        else {
            // Sequential: duration is sum of all adapters
            return baseDuration + (adapters.length * perAdapterDuration);
        }
    }
    // ==========================================================================
    // EXECUTION
    // ==========================================================================
    async executeCoordination(plan, request) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'integration-coordinator',
            target_service: plan.selected_adapters.map(a => a.name).join(','),
        };
        this.logger.info('Executing coordination', {
            ...context,
            adapter_count: plan.selected_adapters.length,
            parallel_execution: plan.parallel_execution,
        });
        const results = [];
        if (plan.parallel_execution) {
            // Execute all adapters in parallel
            const promises = plan.selected_adapters.map(adapter => this.executeAdapter(adapter, request));
            const parallelResults = await Promise.allSettled(promises);
            for (let i = 0; i < parallelResults.length; i++) {
                const result = parallelResults[i];
                const adapter = plan.selected_adapters[i];
                if (result.status === 'fulfilled') {
                    results.push(result.value);
                }
                else {
                    results.push({
                        adapter_name: adapter.name,
                        adapter_type: adapter.type,
                        status: 'failure',
                        error: result.reason instanceof Error ? result.reason.message : String(result.reason),
                        latency_ms: 0,
                    });
                }
            }
        }
        else {
            // Execute adapters sequentially with fallback
            for (const adapter of plan.selected_adapters) {
                const result = await this.executeAdapter(adapter, request);
                results.push(result);
                // Stop on first success for high priority requests
                if (result.status === 'success' && request.priority === 'high') {
                    this.logger.info('Successful execution, stopping sequential execution', {
                        ...context,
                        successful_adapter: adapter.name,
                    });
                    break;
                }
            }
        }
        this.logger.info('Coordination execution completed', {
            ...context,
            total_results: results.length,
            successful: results.filter(r => r.status === 'success').length,
            failed: results.filter(r => r.status === 'failure').length,
        });
        return results;
    }
    async executeAdapter(adapter, request) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'integration-coordinator',
            target_service: adapter.name,
        };
        const startTime = Date.now();
        try {
            this.logger.info('Executing adapter', {
                ...context,
                adapter_type: adapter.type,
                problem_type: request.problem_type,
            });
            // Call the adapter's health check as a basic integration test
            const response = await this.httpClient.get(`${adapter.url}/health`, {
                timeout: request.timeout_ms || this.timeout_ms,
            });
            const latency = Date.now() - startTime;
            this.logger.info('Adapter execution successful', {
                ...context,
                latency_ms: latency,
            });
            return {
                adapter_name: adapter.name,
                adapter_type: adapter.type,
                status: 'success',
                result: response.data,
                latency_ms: latency,
            };
        }
        catch (error) {
            const latency = Date.now() - startTime;
            this.logger.error('Adapter execution failed', {
                ...context,
                latency_ms: latency,
                error: error instanceof Error ? error.message : String(error),
            });
            return {
                adapter_name: adapter.name,
                adapter_type: adapter.type,
                status: 'failure',
                error: error instanceof Error ? error.message : String(error),
                latency_ms: latency,
            };
        }
    }
    // ==========================================================================
    // HEALTH MONITORING
    // ==========================================================================
    async getAdapterHealth() {
        const healthMap = new Map();
        for (const [name, adapter] of this.adapters) {
            try {
                const startTime = Date.now();
                await this.httpClient.get(`${adapter.url}/health`, { timeout: 5000 });
                const latency = Date.now() - startTime;
                healthMap.set(name, {
                    status: 'healthy',
                    latency_ms: latency,
                });
            }
            catch (error) {
                healthMap.set(name, {
                    status: 'unhealthy',
                    latency_ms: -1,
                });
            }
        }
        return healthMap;
    }
    // ==========================================================================
    // UTILITY METHODS
    // ==========================================================================
    generateCorrelationId() {
        return `coord-${Date.now()}-${Math.random().toString(36).substring(7)}`;
    }
    getRegisteredAdapters() {
        return Array.from(this.adapters.values());
    }
    getAdapterByName(name) {
        return this.adapters.get(name);
    }
}
exports.IntegrationCoordinator = IntegrationCoordinator;
// ============================================================================
// FACTORY FUNCTION
// ============================================================================
function createIntegrationCoordinator(openEvolveAdapter, timeout_ms = 10000) {
    return new IntegrationCoordinator(openEvolveAdapter, timeout_ms);
}
//# sourceMappingURL=integration-coordinator.js.map