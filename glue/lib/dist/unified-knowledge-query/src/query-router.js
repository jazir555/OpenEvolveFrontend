"use strict";
/**
 * Query Router
 *
 * Determines which systems to query and how to execute the query.
 *
 * Federation Constitution Compliance:
 * - Runtime Truth: Routes based on actual system capabilities
 * - Failure Management: Considers system health when routing
 * - Configuration Explicitness: No magic defaults
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.queryRouter = exports.QueryRouter = void 0;
const glue_lib_1 = require("@openevolve/glue-lib");
/**
 * Query Router Class
 */
class QueryRouter {
    constructor() {
        this.logger = new glue_lib_1.Logger('query-router');
        this.routingContext = {
            systemHealth: new Map(),
            historicalPerformance: new Map(),
        };
    }
    /**
     * Route query to appropriate systems
     */
    async route(query, availableSystems) {
        this.logger.info('Routing query', {
            correlation_id: query.correlationId,
            query_type: query.queryType,
            domains: query.domains,
        });
        // Determine query strategy
        const strategy = this.determineStrategy(query);
        // Select systems to query
        const systems = this.selectSystems(query.domains, availableSystems);
        // Create execution plan
        const plan = {
            query,
            strategy: strategy.type,
            systems,
            estimatedCost: strategy.estimatedCost.timeMs,
            parallelizable: strategy.parallelizable,
        };
        this.logger.info('Query routed', {
            correlation_id: query.correlationId,
            strategy: strategy.type,
            systems: systems.map(s => s.name),
            parallelizable: strategy.parallelizable,
            estimated_cost: strategy.estimatedCost.timeMs,
        });
        return plan;
    }
    /**
     * Update routing context with system health
     */
    updateSystemHealth(health) {
        for (const h of health) {
            this.routingContext.systemHealth.set(h.system, h);
        }
        this.logger.debug('System health updated', {
            systems: health.map(h => ({
                system: h.system,
                status: h.status,
                response_time: h.responseTimeMs,
            })),
        });
    }
    /**
     * Determine query strategy based on query characteristics
     */
    determineStrategy(query) {
        // Explicit query type
        if (query.queryType && query.queryType !== 'hybrid') {
            return this.getStrategyForType(query.queryType);
        }
        // Temporal filter present -> use temporal query
        if (query.temporalFilter) {
            return this.getStrategyForType('temporal-query');
        }
        // Entity/relationship types -> use graph traversal
        if (query.knowledgeTypes.includes('entity') ||
            query.knowledgeTypes.includes('relationship')) {
            return this.getStrategyForType('graph-traversal');
        }
        // Document types -> use semantic search
        if (query.knowledgeTypes.includes('document')) {
            return this.getStrategyForType('semantic-search');
        }
        // Default to hybrid
        return this.getStrategyForType('hybrid');
    }
    /**
     * Get strategy for specific query type
     */
    getStrategyForType(type) {
        switch (type) {
            case 'semantic-search':
                return {
                    type: 'semantic-search',
                    parallelizable: true,
                    estimatedCost: {
                        timeMs: 2000,
                        complexity: 'low',
                        resources: ['vectordb', 'ragbits'],
                    },
                };
            case 'temporal-query':
                return {
                    type: 'temporal-query',
                    parallelizable: false,
                    estimatedCost: {
                        timeMs: 5000,
                        complexity: 'medium',
                        resources: ['graphiti'],
                    },
                };
            case 'graph-traversal':
                return {
                    type: 'graph-traversal',
                    parallelizable: false,
                    estimatedCost: {
                        timeMs: 7000,
                        complexity: 'high',
                        resources: ['graphiti'],
                    },
                };
            case 'hybrid':
                return {
                    type: 'hybrid',
                    parallelizable: true,
                    estimatedCost: {
                        timeMs: 8000,
                        complexity: 'high',
                        resources: ['ragbits', 'graphiti', 'vectordb'],
                    },
                };
            case 'fallback':
                return {
                    type: 'fallback',
                    parallelizable: false,
                    estimatedCost: {
                        timeMs: 10000,
                        complexity: 'low',
                        resources: [],
                    },
                };
            default:
                return {
                    type: 'hybrid',
                    parallelizable: true,
                    estimatedCost: {
                        timeMs: 5000,
                        complexity: 'medium',
                        resources: ['ragbits', 'graphiti', 'vectordb'],
                    },
                };
        }
    }
    /**
     * Select systems based on domains and health
     */
    selectSystems(domains, availableSystems) {
        const selected = [];
        // Map domains to system names
        for (const domain of domains) {
            if (domain === 'all') {
                // Select all enabled systems
                for (const system of availableSystems) {
                    if (system.enabled && this.isSystemHealthy(system.name)) {
                        selected.push(system);
                    }
                }
                break;
            }
            else {
                // Select specific system
                const systemName = this.domainToSystemName(domain);
                const system = availableSystems.find(s => s.name === systemName);
                if (system && system.enabled && this.isSystemHealthy(systemName)) {
                    selected.push(system);
                }
            }
        }
        // Sort by priority
        selected.sort((a, b) => b.priority - a.priority);
        this.logger.debug('Systems selected', {
            requested_domains: domains,
            selected_systems: selected.map(s => s.name),
        });
        return selected;
    }
    /**
     * Check if system is healthy
     */
    isSystemHealthy(system) {
        const health = this.routingContext.systemHealth.get(system);
        // No health info yet, assume healthy
        if (!health) {
            return true;
        }
        // System is healthy or degraded (degraded still works)
        return health.status === 'healthy' || health.status === 'degraded';
    }
    /**
     * Map domain to system name
     */
    domainToSystemName(domain) {
        const mapping = {
            ragbits: 'ragbits',
            graphiti: 'graphiti',
            vectordb: 'vectordb',
            all: 'fused', // Placeholder
        };
        return mapping[domain];
    }
    /**
     * Estimate cost of query execution
     */
    estimateCost(plan) {
        let baseCost = 0;
        // Add cost for each system
        for (const system of plan.systems) {
            baseCost += this.getSystemCost(system.name);
        }
        // Add strategy overhead
        const strategyOverhead = {
            'semantic-search': 1.0,
            'temporal-query': 1.5,
            'graph-traversal': 2.0,
            'hybrid': 1.2,
            'fallback': 0.5,
        };
        baseCost *= strategyOverhead[plan.strategy] || 1.0;
        return Math.round(baseCost);
    }
    /**
     * Get estimated cost for system
     */
    getSystemCost(system) {
        const costs = {
            ragbits: 2000,
            graphiti: 3000,
            vectordb: 1500,
            fused: 4000,
        };
        return costs[system] || 2000;
    }
    /**
     * Get current routing context
     */
    getRoutingContext() {
        return this.routingContext;
    }
    /**
     * Reset routing context
     */
    reset() {
        this.routingContext.systemHealth.clear();
        this.routingContext.historicalPerformance.clear();
        this.logger.info('Routing context reset');
    }
}
exports.QueryRouter = QueryRouter;
/**
 * Default router instance
 */
exports.queryRouter = new QueryRouter();
//# sourceMappingURL=query-router.js.map