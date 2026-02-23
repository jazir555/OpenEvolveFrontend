/**
 * Health Checker
 *
 * Follows the Federation Constitution:
 * - Law of Configuration Explicitness: All config via environment variables
 * - Runtime Truth: Health checks verify actual service availability
 * - Failure Management: Graceful degradation
 *
 * Provides:
 * - HTTP health endpoints for each service
 * - Aggregated health status
 * - Readiness/liveness probes
 * - Dependency health checks
 */
export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy';
export interface HealthCheckResult {
    name: string;
    status: HealthStatus;
    message?: string;
    timestamp: string;
    response_time_ms?: number;
    dependencies?: HealthCheckResult[];
    metadata?: Record<string, any>;
}
export interface HealthCheckOptions {
    timeout?: number;
    critical?: boolean;
}
export type HealthCheckFunction = () => Promise<HealthCheckResult> | HealthCheckResult;
/**
 * Health Checker class
 *
 * Manages health checks for services and their dependencies
 */
export declare class HealthChecker {
    private checks;
    private serviceName;
    constructor(serviceName: string);
    /**
     * Register a health check
     */
    register(name: string, checkFn: HealthCheckFunction): void;
    /**
     * Unregister a health check
     */
    unregister(name: string): void;
    /**
     * Execute a single health check
     */
    private executeCheck;
    /**
     * Execute all registered health checks
     */
    checkHealth(): Promise<HealthCheckResult>;
    /**
     * Execute specific health check
     */
    checkSpecific(checkName: string): Promise<HealthCheckResult>;
    /**
     * Calculate overall health status from results
     */
    private calculateOverallStatus;
    /**
     * Get liveness status (is the service running?)
     * This should always return true if the process is alive
     */
    getLiveness(): Promise<HealthCheckResult>;
    /**
     * Get readiness status (is the service ready to handle traffic?)
     * Checks if all critical dependencies are healthy
     */
    getReadiness(): Promise<HealthCheckResult>;
}
/**
 * HTTP Health Endpoint Response
 */
export interface HealthEndpointResponse {
    status: number;
    body: HealthCheckResult;
    headers: Record<string, string>;
}
/**
 * Create HTTP health endpoint handlers
 */
export declare class HealthEndpointHandler {
    private healthChecker;
    constructor(healthChecker: HealthChecker);
    /**
     * Handle GET /health
     * Returns overall health status
     */
    handleHealth(): Promise<HealthEndpointResponse>;
    /**
     * Handle GET /health/live
     * Liveness probe (Kubernetes style)
     */
    handleLiveness(): Promise<HealthEndpointResponse>;
    /**
     * Handle GET /health/ready
     * Readiness probe (Kubernetes style)
     */
    handleReadiness(): Promise<HealthEndpointResponse>;
    /**
     * Handle GET /health/:checkName
     * Specific health check
     */
    handleSpecificCheck(checkName: string): Promise<HealthEndpointResponse>;
    /**
     * Convert health status to HTTP status code
     */
    private statusToStatusCode;
}
/**
 * Helper function to create HTTP health check
 */
export declare function createHttpHealthCheck(url: string, options?: {
    timeout?: number;
    expectedStatus?: number;
}): HealthCheckFunction;
/**
 * Helper function to create TCP health check
 */
export declare function createTcpHealthCheck(host: string, port: number, options?: {
    timeout?: number;
}): HealthCheckFunction;
/**
 * Helper function to create database health check
 */
export declare function createDatabaseHealthCheck(checkFn: () => Promise<void>, options?: {
    timeout?: number;
}): HealthCheckFunction;
/**
 * Example usage:
 *
 * ```typescript
 * import { HealthChecker, HealthEndpointHandler, createHttpHealthCheck } from './health-checker';
 *
 * // Create health checker
 * const healthChecker = new HealthChecker('crm-adapter');
 *
 * // Register health checks
 * healthChecker.register('api', createHttpHealthCheck('http://crm-core:8000/health', {
 *   timeout: 5000,
 *   expectedStatus: 200,
 * }));
 *
 * healthChecker.register('database', async () => {
 *   await database.query('SELECT 1');
 *   return {
 *     name: 'database',
 *     status: 'healthy',
 *     message: 'Database OK',
 *     timestamp: new Date().toISOString(),
 *   };
 * });
 *
 * // Create endpoint handler
 * const handler = new HealthEndpointHandler(healthChecker);
 *
 * // Use in Express
 * app.get('/health', async (req, res) => {
 *   const result = await handler.handleHealth();
 *   res.status(result.status).json(result.body);
 * });
 *
 * // Or use directly
 * const health = await healthChecker.checkHealth();
 * console.log(health);
 * ```
 */
//# sourceMappingURL=health-checker.d.ts.map