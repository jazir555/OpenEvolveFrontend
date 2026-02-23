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
import { UnifiedKnowledgeQuery, QueryPlan, SystemConfig, SystemSource, SystemHealth } from './canonical';
/**
 * Routing Context
 */
interface RoutingContext {
    systemHealth: Map<SystemSource, SystemHealth>;
    historicalPerformance: Map<SystemSource, number>;
}
/**
 * Query Router Class
 */
export declare class QueryRouter {
    private logger;
    private routingContext;
    constructor();
    /**
     * Route query to appropriate systems
     */
    route(query: UnifiedKnowledgeQuery, availableSystems: SystemConfig[]): Promise<QueryPlan>;
    /**
     * Update routing context with system health
     */
    updateSystemHealth(health: SystemHealth[]): void;
    /**
     * Determine query strategy based on query characteristics
     */
    private determineStrategy;
    /**
     * Get strategy for specific query type
     */
    private getStrategyForType;
    /**
     * Select systems based on domains and health
     */
    private selectSystems;
    /**
     * Check if system is healthy
     */
    private isSystemHealthy;
    /**
     * Map domain to system name
     */
    private domainToSystemName;
    /**
     * Estimate cost of query execution
     */
    private estimateCost;
    /**
     * Get estimated cost for system
     */
    private getSystemCost;
    /**
     * Get current routing context
     */
    getRoutingContext(): RoutingContext;
    /**
     * Reset routing context
     */
    reset(): void;
}
/**
 * Default router instance
 */
export declare const queryRouter: QueryRouter;
export {};
//# sourceMappingURL=query-router.d.ts.map