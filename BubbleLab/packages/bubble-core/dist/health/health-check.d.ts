/**
 * Health Check System for BubbleLab
 *
 * Provides comprehensive health check endpoints for monitoring system status
 * including dependencies, circuit breakers, and bubble initialization status
 */
export interface HealthCheckResult {
    name: string;
    status: 'healthy' | 'degraded' | 'unhealthy';
    message?: string;
    responseTime?: number;
    lastCheck?: string;
}
export interface SystemHealth {
    status: 'healthy' | 'degraded' | 'unhealthy';
    timestamp: string;
    checks: Record<string, HealthCheckResult>;
    uptime: number;
}
export interface ReadinessCheck {
    ready: boolean;
    checks: Record<string, boolean>;
    message?: string;
}
declare class HealthCheckRegistry {
    private checks;
    private startTime;
    /**
     * Register a health check
     */
    register(name: string, check: () => Promise<HealthCheckResult>): void;
    /**
     * Unregister a health check
     */
    unregister(name: string): void;
    /**
     * Run a specific health check
     */
    runCheck(name: string): Promise<HealthCheckResult>;
    /**
     * Run all health checks
     */
    runAllChecks(): Promise<Record<string, HealthCheckResult>>;
    /**
     * Get overall system health
     */
    getSystemHealth(): Promise<SystemHealth>;
    /**
     * Get uptime in seconds
     */
    getUptime(): number;
}
declare const registry: HealthCheckRegistry;
/**
 * Database health check
 */
export declare function checkDatabase(queryFn: () => Promise<unknown>): Promise<HealthCheckResult>;
/**
 * Redis health check
 */
export declare function checkRedis(pingFn: () => Promise<string>): Promise<HealthCheckResult>;
/**
 * Qdrant health check
 */
export declare function checkQdrant(healthFn: () => Promise<{
    status: string;
} | unknown>): Promise<HealthCheckResult>;
/**
 * Elasticsearch health check
 */
export declare function checkElasticsearch(healthFn: () => Promise<{
    status: string;
} | unknown>): Promise<HealthCheckResult>;
/**
 * HTTP endpoint health check
 */
export declare function checkHttpEndpoint(url: string, timeout?: number): Promise<HealthCheckResult>;
/**
 * Custom health check
 */
export declare function createCustomCheck(name: string, checkFn: () => Promise<boolean>, healthyMessage?: string, unhealthyMessage?: string): () => Promise<HealthCheckResult>;
/**
 * Check if all bubbles are initialized
 */
export declare function checkBubblesInitialized(getBubbleStatus: () => Promise<Record<string, boolean>>): Promise<boolean>;
/**
 * Check if all circuit breakers are in a valid state
 */
export declare function checkCircuitBreakers(getCircuitBreakerStatus: () => Promise<Record<string, boolean>>): Promise<boolean>;
/**
 * Get readiness status
 */
export declare function getReadinessStatus(checks: Record<string, () => Promise<boolean>>): Promise<ReadinessCheck>;
import { Request, Response } from 'express';
/**
 * Health check endpoint middleware
 */
export declare function healthCheckEndpoint(_req: Request, res: Response): Promise<void>;
/**
 * Readiness check endpoint middleware
 */
export declare function readinessCheckEndpoint(checks: Record<string, () => Promise<boolean>>): Promise<(_req: Request, res: Response) => Promise<void>>;
/**
 * Liveness check endpoint middleware
 */
export declare function livenessCheckEndpoint(_req: Request, res: Response): Promise<void>;
export declare function registerHealthCheck(name: string, check: () => Promise<HealthCheckResult>): void;
export declare function unregisterHealthCheck(name: string): void;
export { registry };
export default registry;
//# sourceMappingURL=health-check.d.ts.map