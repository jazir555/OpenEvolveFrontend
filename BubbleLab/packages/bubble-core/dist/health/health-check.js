/**
 * Health Check System for BubbleLab
 *
 * Provides comprehensive health check endpoints for monitoring system status
 * including dependencies, circuit breakers, and bubble initialization status
 */
// ============================================================================
// HEALTH CHECK REGISTRY
// ============================================================================
class HealthCheckRegistry {
    checks = new Map();
    startTime = Date.now();
    /**
     * Register a health check
     */
    register(name, check) {
        this.checks.set(name, check);
    }
    /**
     * Unregister a health check
     */
    unregister(name) {
        this.checks.delete(name);
    }
    /**
     * Run a specific health check
     */
    async runCheck(name) {
        const check = this.checks.get(name);
        if (!check) {
            return {
                name,
                status: 'unhealthy',
                message: 'Health check not found'
            };
        }
        try {
            return await check();
        }
        catch (error) {
            return {
                name,
                status: 'unhealthy',
                message: error instanceof Error ? error.message : 'Unknown error'
            };
        }
    }
    /**
     * Run all health checks
     */
    async runAllChecks() {
        const results = {};
        const promises = Array.from(this.checks.entries()).map(async ([name, check]) => {
            try {
                const result = await check();
                results[name] = result;
            }
            catch (error) {
                results[name] = {
                    name,
                    status: 'unhealthy',
                    message: error instanceof Error ? error.message : 'Unknown error'
                };
            }
        });
        await Promise.all(promises);
        return results;
    }
    /**
     * Get overall system health
     */
    async getSystemHealth() {
        const checks = await this.runAllChecks();
        const statuses = Object.values(checks).map((c) => c.status);
        let overallStatus = 'healthy';
        if (statuses.some((s) => s === 'unhealthy')) {
            overallStatus = 'unhealthy';
        }
        else if (statuses.some((s) => s === 'degraded')) {
            overallStatus = 'degraded';
        }
        return {
            status: overallStatus,
            timestamp: new Date().toISOString(),
            checks,
            uptime: Date.now() - this.startTime
        };
    }
    /**
     * Get uptime in seconds
     */
    getUptime() {
        return Math.floor((Date.now() - this.startTime) / 1000);
    }
}
// Global registry instance
const registry = new HealthCheckRegistry();
// ============================================================================
// PREDEFINED HEALTH CHECKS
// ============================================================================
/**
 * Database health check
 */
export async function checkDatabase(queryFn) {
    const start = Date.now();
    try {
        await queryFn();
        const responseTime = Date.now() - start;
        return {
            name: 'database',
            status: 'healthy',
            message: 'Database connection successful',
            responseTime,
            lastCheck: new Date().toISOString()
        };
    }
    catch (error) {
        return {
            name: 'database',
            status: 'unhealthy',
            message: error instanceof Error ? error.message : 'Database connection failed',
            responseTime: Date.now() - start,
            lastCheck: new Date().toISOString()
        };
    }
}
/**
 * Redis health check
 */
export async function checkRedis(pingFn) {
    const start = Date.now();
    try {
        const result = await pingFn();
        const responseTime = Date.now() - start;
        if (result === 'PONG') {
            return {
                name: 'redis',
                status: 'healthy',
                message: 'Redis connection successful',
                responseTime,
                lastCheck: new Date().toISOString()
            };
        }
        else {
            return {
                name: 'redis',
                status: 'degraded',
                message: `Unexpected Redis response: ${result}`,
                responseTime,
                lastCheck: new Date().toISOString()
            };
        }
    }
    catch (error) {
        return {
            name: 'redis',
            status: 'unhealthy',
            message: error instanceof Error ? error.message : 'Redis connection failed',
            responseTime: Date.now() - start,
            lastCheck: new Date().toISOString()
        };
    }
}
/**
 * Qdrant health check
 */
export async function checkQdrant(healthFn) {
    const start = Date.now();
    try {
        const result = await healthFn();
        const responseTime = Date.now() - start;
        if (result?.status === 'ok' || result?.status === 'green') {
            return {
                name: 'qdrant',
                status: 'healthy',
                message: 'Qdrant connection successful',
                responseTime,
                lastCheck: new Date().toISOString()
            };
        }
        else {
            return {
                name: 'qdrant',
                status: 'degraded',
                message: `Qdrant status: ${result?.status || 'unknown'}`,
                responseTime,
                lastCheck: new Date().toISOString()
            };
        }
    }
    catch (error) {
        return {
            name: 'qdrant',
            status: 'unhealthy',
            message: error instanceof Error ? error.message : 'Qdrant connection failed',
            responseTime: Date.now() - start,
            lastCheck: new Date().toISOString()
        };
    }
}
/**
 * Elasticsearch health check
 */
export async function checkElasticsearch(healthFn) {
    const start = Date.now();
    try {
        const result = await healthFn();
        const responseTime = Date.now() - start;
        if (result?.status === 'green' || result?.status === 'yellow') {
            return {
                name: 'elasticsearch',
                status: result.status === 'green' ? 'healthy' : 'degraded',
                message: `Elasticsearch status: ${result.status}`,
                responseTime,
                lastCheck: new Date().toISOString()
            };
        }
        else {
            return {
                name: 'elasticsearch',
                status: 'unhealthy',
                message: `Elasticsearch status: ${result?.status || 'unknown'}`,
                responseTime,
                lastCheck: new Date().toISOString()
            };
        }
    }
    catch (error) {
        return {
            name: 'elasticsearch',
            status: 'unhealthy',
            message: error instanceof Error ? error.message : 'Elasticsearch connection failed',
            responseTime: Date.now() - start,
            lastCheck: new Date().toISOString()
        };
    }
}
/**
 * HTTP endpoint health check
 */
export async function checkHttpEndpoint(url, timeout = 5000) {
    const start = Date.now();
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    try {
        const response = await fetch(url, {
            method: 'GET',
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        const responseTime = Date.now() - start;
        if (response.ok) {
            return {
                name: url,
                status: 'healthy',
                message: `HTTP ${response.status}: ${response.statusText}`,
                responseTime,
                lastCheck: new Date().toISOString()
            };
        }
        else {
            return {
                name: url,
                status: response.status >= 500 ? 'unhealthy' : 'degraded',
                message: `HTTP ${response.status}: ${response.statusText}`,
                responseTime,
                lastCheck: new Date().toISOString()
            };
        }
    }
    catch (error) {
        clearTimeout(timeoutId);
        return {
            name: url,
            status: 'unhealthy',
            message: error instanceof Error ? error.message : 'HTTP endpoint check failed',
            responseTime: Date.now() - start,
            lastCheck: new Date().toISOString()
        };
    }
}
/**
 * Custom health check
 */
export function createCustomCheck(name, checkFn, healthyMessage = 'Check passed', unhealthyMessage = 'Check failed') {
    return async () => {
        const start = Date.now();
        try {
            const result = await checkFn();
            const responseTime = Date.now() - start;
            return {
                name,
                status: result ? 'healthy' : 'unhealthy',
                message: result ? healthyMessage : unhealthyMessage,
                responseTime,
                lastCheck: new Date().toISOString()
            };
        }
        catch (error) {
            return {
                name,
                status: 'unhealthy',
                message: error instanceof Error ? error.message : 'Custom check failed',
                responseTime: Date.now() - start,
                lastCheck: new Date().toISOString()
            };
        }
    };
}
// ============================================================================
// READINESS CHECKS
// ============================================================================
/**
 * Check if all bubbles are initialized
 */
export async function checkBubblesInitialized(getBubbleStatus) {
    try {
        const status = await getBubbleStatus();
        const allInitialized = Object.values(status).every((s) => s);
        return allInitialized;
    }
    catch {
        return false;
    }
}
/**
 * Check if all circuit breakers are in a valid state
 */
export async function checkCircuitBreakers(getCircuitBreakerStatus) {
    try {
        const status = await getCircuitBreakerStatus();
        const allValid = Object.values(status).every((s) => s);
        return allValid;
    }
    catch {
        return false;
    }
}
/**
 * Get readiness status
 */
export async function getReadinessStatus(checks) {
    const results = {};
    for (const [name, check] of Object.entries(checks)) {
        try {
            results[name] = await check();
        }
        catch {
            results[name] = false;
        }
    }
    const allReady = Object.values(results).every((r) => r);
    return {
        ready: allReady,
        checks: results,
        message: allReady ? 'System is ready' : 'System is not ready'
    };
}
/**
 * Health check endpoint middleware
 */
export async function healthCheckEndpoint(_req, res) {
    const health = await registry.getSystemHealth();
    const statusCode = health.status === 'healthy' ? 200 : health.status === 'degraded' ? 200 : 503;
    res.status(statusCode).json(health);
}
/**
 * Readiness check endpoint middleware
 */
export async function readinessCheckEndpoint(checks) {
    return async (_req, res) => {
        const readiness = await getReadinessStatus(checks);
        const statusCode = readiness.ready ? 200 : 503;
        res.status(statusCode).json(readiness);
    };
}
/**
 * Liveness check endpoint middleware
 */
export async function livenessCheckEndpoint(_req, res) {
    res.status(200).json({
        status: 'alive',
        timestamp: new Date().toISOString(),
        uptime: registry.getUptime()
    });
}
// ============================================================================
// REGISTRY EXPORTS
// ============================================================================
export function registerHealthCheck(name, check) {
    registry.register(name, check);
}
export function unregisterHealthCheck(name) {
    registry.unregister(name);
}
export { registry };
export default registry;
//# sourceMappingURL=health-check.js.map