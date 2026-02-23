/**
 * Health Check Library
 *
 * Following CLAUDE.md principles:
 * - Law of Configuration Explicitness: All config via environment variables
 * - Law of Runtime Truth: Verify dependencies are actually accessible
 * - Structured Logging: JSON format with correlation_id
 *
 * Usage:
 * ```typescript
 * import { createHealthCheckServer, checkDependency } from './health-check';
 *
 * const server = createHealthCheckServer({
 *   port: parseInt(process.env.SERVICE_PORT || '3000'),
 *   checks: [
 *     async () => checkDependency('http://core-service:8000', 'Core Service'),
 *     async () => checkDatabase(process.env.DB_URL)
 *   ]
 * });
 *
 * server.start();
 * ```
 */
export interface HealthCheckResult {
    name: string;
    status: 'healthy' | 'unhealthy' | 'degraded';
    message?: string;
    responseTime?: number;
    timestamp: string;
}
export interface HealthCheckResponse {
    status: 'healthy' | 'unhealthy' | 'degraded';
    timestamp: string;
    uptime: number;
    checks: HealthCheckResult[];
    version?: string;
}
export interface HealthCheckOptions {
    port: number;
    checks?: Array<() => Promise<HealthCheckResult>>;
    version?: string;
    timeout?: number;
}
/**
 * Check if an HTTP/HTTPS dependency is accessible
 */
export declare function checkDependency(url: string, name: string, timeout?: number): Promise<HealthCheckResult>;
/**
 * Check database connectivity (generic SQL check)
 */
export declare function checkDatabase(connectionString: string, name?: string): Promise<HealthCheckResult>;
/**
 * Check environment variable configuration
 */
export declare function checkEnvVariables(requiredVars: Record<string, string>, name?: string): Promise<HealthCheckResult>;
/**
 * Create a health check HTTP server
 */
export declare function createHealthCheckServer(options: HealthCheckOptions): {
    start: () => Promise<void>;
    stop: () => Promise<void>;
};
/**
 * Startup validator - crashes process if required environment is missing
 * Following Law of Configuration Explicitness
 */
export declare function validateRequiredEnv(requiredVars: Record<string, string>): void;
/**
 * Default health check that can be used in adapters
 */
export declare function defaultHealthCheck(): Promise<HealthCheckResult>;
//# sourceMappingURL=health-check.d.ts.map