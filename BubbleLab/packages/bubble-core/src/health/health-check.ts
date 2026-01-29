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

// ============================================================================
// HEALTH CHECK REGISTRY
// ============================================================================

class HealthCheckRegistry {
  private checks: Map<string, () => Promise<HealthCheckResult>> = new Map();
  private startTime: number = Date.now();

  /**
   * Register a health check
   */
  register(name: string, check: () => Promise<HealthCheckResult>): void {
    this.checks.set(name, check);
  }

  /**
   * Unregister a health check
   */
  unregister(name: string): void {
    this.checks.delete(name);
  }

  /**
   * Run a specific health check
   */
  async runCheck(name: string): Promise<HealthCheckResult> {
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
    } catch (error) {
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
  async runAllChecks(): Promise<Record<string, HealthCheckResult>> {
    const results: Record<string, HealthCheckResult> = {};

    const promises = Array.from(this.checks.entries()).map(
      async ([name, check]) => {
        try {
          const result = await check();
          results[name] = result;
        } catch (error) {
          results[name] = {
            name,
            status: 'unhealthy',
            message: error instanceof Error ? error.message : 'Unknown error'
          };
        }
      }
    );

    await Promise.all(promises);
    return results;
  }

  /**
   * Get overall system health
   */
  async getSystemHealth(): Promise<SystemHealth> {
    const checks = await this.runAllChecks();
    const statuses = Object.values(checks).map((c) => c.status);

    let overallStatus: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';

    if (statuses.some((s) => s === 'unhealthy')) {
      overallStatus = 'unhealthy';
    } else if (statuses.some((s) => s === 'degraded')) {
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
  getUptime(): number {
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
export async function checkDatabase(
  queryFn: () => Promise<unknown>
): Promise<HealthCheckResult> {
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
  } catch (error) {
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
export async function checkRedis(
  pingFn: () => Promise<string>
): Promise<HealthCheckResult> {
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
    } else {
      return {
        name: 'redis',
        status: 'degraded',
        message: `Unexpected Redis response: ${result}`,
        responseTime,
        lastCheck: new Date().toISOString()
      };
    }
  } catch (error) {
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
export async function checkQdrant(
  healthFn: () => Promise<{ status: string } | unknown>
): Promise<HealthCheckResult> {
  const start = Date.now();

  try {
    const result = await healthFn() as { status?: string };
    const responseTime = Date.now() - start;

    if (result?.status === 'ok' || result?.status === 'green') {
      return {
        name: 'qdrant',
        status: 'healthy',
        message: 'Qdrant connection successful',
        responseTime,
        lastCheck: new Date().toISOString()
      };
    } else {
      return {
        name: 'qdrant',
        status: 'degraded',
        message: `Qdrant status: ${result?.status || 'unknown'}`,
        responseTime,
        lastCheck: new Date().toISOString()
      };
    }
  } catch (error) {
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
export async function checkElasticsearch(
  healthFn: () => Promise<{ status: string } | unknown>
): Promise<HealthCheckResult> {
  const start = Date.now();

  try {
    const result = await healthFn() as { status?: string };
    const responseTime = Date.now() - start;

    if (result?.status === 'green' || result?.status === 'yellow') {
      return {
        name: 'elasticsearch',
        status: result.status === 'green' ? 'healthy' : 'degraded',
        message: `Elasticsearch status: ${result.status}`,
        responseTime,
        lastCheck: new Date().toISOString()
      };
    } else {
      return {
        name: 'elasticsearch',
        status: 'unhealthy',
        message: `Elasticsearch status: ${result?.status || 'unknown'}`,
        responseTime,
        lastCheck: new Date().toISOString()
      };
    }
  } catch (error) {
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
export async function checkHttpEndpoint(
  url: string,
  timeout: number = 5000
): Promise<HealthCheckResult> {
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
    } else {
      return {
        name: url,
        status: response.status >= 500 ? 'unhealthy' : 'degraded',
        message: `HTTP ${response.status}: ${response.statusText}`,
        responseTime,
        lastCheck: new Date().toISOString()
      };
    }
  } catch (error) {
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
export function createCustomCheck(
  name: string,
  checkFn: () => Promise<boolean>,
  healthyMessage: string = 'Check passed',
  unhealthyMessage: string = 'Check failed'
): () => Promise<HealthCheckResult> {
  return async (): Promise<HealthCheckResult> => {
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
    } catch (error) {
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
export async function checkBubblesInitialized(
  getBubbleStatus: () => Promise<Record<string, boolean>>
): Promise<boolean> {
  try {
    const status = await getBubbleStatus();
    const allInitialized = Object.values(status).every((s) => s);
    return allInitialized;
  } catch {
    return false;
  }
}

/**
 * Check if all circuit breakers are in a valid state
 */
export async function checkCircuitBreakers(
  getCircuitBreakerStatus: () => Promise<Record<string, boolean>>
): Promise<boolean> {
  try {
    const status = await getCircuitBreakerStatus();
    const allValid = Object.values(status).every((s) => s);
    return allValid;
  } catch {
    return false;
  }
}

/**
 * Get readiness status
 */
export async function getReadinessStatus(
  checks: Record<string, () => Promise<boolean>>
): Promise<ReadinessCheck> {
  const results: Record<string, boolean> = {};

  for (const [name, check] of Object.entries(checks)) {
    try {
      results[name] = await check();
    } catch {
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

// ============================================================================
// EXPRESS MIDDLEWARE
// ============================================================================

import { Request, Response } from 'express';

/**
 * Health check endpoint middleware
 */
export async function healthCheckEndpoint(_req: Request, res: Response): Promise<void> {
  const health = await registry.getSystemHealth();
  const statusCode = health.status === 'healthy' ? 200 : health.status === 'degraded' ? 200 : 503;

  res.status(statusCode).json(health);
}

/**
 * Readiness check endpoint middleware
 */
export async function readinessCheckEndpoint(
  checks: Record<string, () => Promise<boolean>>
) {
  return async (_req: Request, res: Response): Promise<void> => {
    const readiness = await getReadinessStatus(checks);
    const statusCode = readiness.ready ? 200 : 503;

    res.status(statusCode).json(readiness);
  };
}

/**
 * Liveness check endpoint middleware
 */
export async function livenessCheckEndpoint(_req: Request, res: Response): Promise<void> {
  res.status(200).json({
    status: 'alive',
    timestamp: new Date().toISOString(),
    uptime: registry.getUptime()
  });
}

// ============================================================================
// REGISTRY EXPORTS
// ============================================================================

export function registerHealthCheck(
  name: string,
  check: () => Promise<HealthCheckResult>
): void {
  registry.register(name, check);
}

export function unregisterHealthCheck(name: string): void {
  registry.unregister(name);
}

export { registry };
export default registry;
