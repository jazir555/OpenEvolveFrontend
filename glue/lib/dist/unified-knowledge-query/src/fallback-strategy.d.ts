/**
 * Fallback Strategy
 *
 * Implements graceful degradation when systems are unavailable.
 *
 * Federation Constitution Compliance:
 * - Failure Management: Transient failures trigger fallback
 * - Circuit Breaker: System failures trigger circuit opening
 * - Law of Idempotency: Fallback queries are safe to retry
 */
import { SystemConfig, SystemSource, QueryPlan, KnowledgeItem, SystemHealth } from './canonical';
/**
 * Fallback Configuration
 */
interface FallbackConfig {
    maxAttempts: number;
    retryDelayMs: number;
    enableFallback: boolean;
    fallbackPriority: SystemSource[];
}
/**
 * Fallback Execution Result
 */
interface FallbackResult {
    items: KnowledgeItem[];
    systemUsed: SystemSource;
    attempt: number;
    wasFallback: boolean;
}
/**
 * Fallback Strategy Class
 */
export declare class FallbackStrategy {
    private logger;
    private config;
    private systemHealth;
    constructor(config?: Partial<FallbackConfig>);
    /**
     * Execute query with fallback strategy
     */
    executeWithFallback(plan: QueryPlan, primary: SystemConfig, fallbacks: SystemConfig[], query: string, options?: any): Promise<FallbackResult>;
    /**
     * Execute query against multiple systems in parallel
     * Use first successful response
     */
    executeParallel(systems: SystemConfig[], query: string, options?: any): Promise<FallbackResult>;
    /**
     * Try executing query against a system
     */
    private trySystem;
    /**
     * Create client for system
     */
    private createClient;
    /**
     * Determine if error should trigger fallback
     */
    private shouldFallback;
    /**
     * Select best fallback from available systems
     */
    selectFallback(available: SystemConfig[]): SystemConfig;
    /**
     * Get numeric score for health status
     */
    private getHealthScore;
    /**
     * Update system health
     */
    updateSystemHealth(health: SystemHealth[]): void;
    /**
     * Get available fallbacks
     */
    getAvailableFallbacks(allSystems: SystemConfig[]): SystemConfig[];
    /**
     * Delay helper
     */
    private delay;
    /**
     * Get current config
     */
    getConfig(): FallbackConfig;
    /**
     * Update config
     */
    updateConfig(updates: Partial<FallbackConfig>): void;
    /**
     * Reset health status
     */
    reset(): void;
}
/**
 * Default fallback strategy instance
 */
export declare const fallbackStrategy: FallbackStrategy;
export {};
//# sourceMappingURL=fallback-strategy.d.ts.map