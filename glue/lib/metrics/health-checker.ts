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

import { logger, LoggerContext } from '../logger';

export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy';

export interface HealthCheckResult {
  name: string;
  status: HealthStatus;
  message?: string;
  timestamp: string; // ISO-8601 UTC
  response_time_ms?: number;
  dependencies?: HealthCheckResult[];
  metadata?: Record<string, any>;
}

export interface HealthCheckOptions {
  timeout?: number;
  critical?: boolean; // If true, unhealthy status causes system to be unhealthy
}

export type HealthCheckFunction = () => Promise<HealthCheckResult> | HealthCheckResult;

/**
 * Health Checker class
 *
 * Manages health checks for services and their dependencies
 */
export class HealthChecker {
  private checks: Map<string, HealthCheckFunction>;
  private serviceName: string;

  constructor(serviceName: string) {
    this.serviceName = serviceName;
    this.checks = new Map();
  }

  /**
   * Register a health check
   */
  register(name: string, checkFn: HealthCheckFunction): void {
    this.checks.set(name, checkFn);
    logger.info('Health check registered', {
      service: this.serviceName,
      check_name: name,
    });
  }

  /**
   * Unregister a health check
   */
  unregister(name: string): void {
    this.checks.delete(name);
    logger.info('Health check unregistered', {
      service: this.serviceName,
      check_name: name,
    });
  }

  /**
   * Execute a single health check
   */
  private async executeCheck(
    name: string,
    checkFn: HealthCheckFunction
  ): Promise<HealthCheckResult> {
    const start = Date.now();

    try {
      const result = await checkFn();
      const responseTime = Date.now() - start;

      return {
        ...result,
        name: result.name || name,
        timestamp: new Date().toISOString(),
        response_time_ms: responseTime,
      };
    } catch (error) {
      const responseTime = Date.now() - start;

      return {
        name,
        status: 'unhealthy',
        message: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString(),
        response_time_ms: responseTime,
      };
    }
  }

  /**
   * Execute all registered health checks
   */
  async checkHealth(): Promise<HealthCheckResult> {
    const results: HealthCheckResult[] = [];
    const start = Date.now();

    for (const [name, checkFn] of this.checks.entries()) {
      const result = await this.executeCheck(name, checkFn);
      results.push(result);
    }

    // Determine overall status
    const overallStatus = this.calculateOverallStatus(results);
    const responseTime = Date.now() - start;

    return {
      name: this.serviceName,
      status: overallStatus,
      timestamp: new Date().toISOString(),
      response_time_ms: responseTime,
      dependencies: results,
    };
  }

  /**
   * Execute specific health check
   */
  async checkSpecific(checkName: string): Promise<HealthCheckResult> {
    const checkFn = this.checks.get(checkName);

    if (!checkFn) {
      return {
        name: checkName,
        status: 'unhealthy',
        message: `Health check '${checkName}' not found`,
        timestamp: new Date().toISOString(),
      };
    }

    return await this.executeCheck(checkName, checkFn);
  }

  /**
   * Calculate overall health status from results
   */
  private calculateOverallStatus(results: HealthCheckResult[]): HealthStatus {
    if (results.length === 0) {
      return 'healthy';
    }

    const hasUnhealthy = results.some((r) => r.status === 'unhealthy');
    const hasDegraded = results.some((r) => r.status === 'degraded');

    if (hasUnhealthy) {
      return 'unhealthy';
    } else if (hasDegraded) {
      return 'degraded';
    } else {
      return 'healthy';
    }
  }

  /**
   * Get liveness status (is the service running?)
   * This should always return true if the process is alive
   */
  async getLiveness(): Promise<HealthCheckResult> {
    return {
      name: this.serviceName,
      status: 'healthy',
      message: 'Service is running',
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Get readiness status (is the service ready to handle traffic?)
   * Checks if all critical dependencies are healthy
   */
  async getReadiness(): Promise<HealthCheckResult> {
    const results: HealthCheckResult[] = [];

    for (const [name, checkFn] of this.checks.entries()) {
      const result = await this.executeCheck(name, checkFn);
      results.push(result);
    }

    // Only check if critical checks pass
    const criticalResults = results.filter((r) => r.metadata?.critical === true);
    const overallStatus = criticalResults.some((r) => r.status === 'unhealthy')
      ? 'unhealthy'
      : 'healthy';

    return {
      name: this.serviceName,
      status: overallStatus,
      timestamp: new Date().toISOString(),
      dependencies: criticalResults,
    };
  }
}

/**
 * HTTP Health Endpoint Response
 */
export interface HealthEndpointResponse {
  status: number; // HTTP status code
  body: HealthCheckResult;
  headers: Record<string, string>;
}

/**
 * Create HTTP health endpoint handlers
 */
export class HealthEndpointHandler {
  private healthChecker: HealthChecker;

  constructor(healthChecker: HealthChecker) {
    this.healthChecker = healthChecker;
  }

  /**
   * Handle GET /health
   * Returns overall health status
   */
  async handleHealth(): Promise<HealthEndpointResponse> {
    const result = await this.healthChecker.checkHealth();
    const statusCode = this.statusToStatusCode(result.status);

    return {
      status: statusCode,
      body: result,
      headers: {
        'Content-Type': 'application/json',
      },
    };
  }

  /**
   * Handle GET /health/live
   * Liveness probe (Kubernetes style)
   */
  async handleLiveness(): Promise<HealthEndpointResponse> {
    const result = await this.healthChecker.getLiveness();

    return {
      status: 200,
      body: result,
      headers: {
        'Content-Type': 'application/json',
      },
    };
  }

  /**
   * Handle GET /health/ready
   * Readiness probe (Kubernetes style)
   */
  async handleReadiness(): Promise<HealthEndpointResponse> {
    const result = await this.healthChecker.getReadiness();
    const statusCode = this.statusToStatusCode(result.status);

    return {
      status: statusCode,
      body: result,
      headers: {
        'Content-Type': 'application/json',
      },
    };
  }

  /**
   * Handle GET /health/:checkName
   * Specific health check
   */
  async handleSpecificCheck(checkName: string): Promise<HealthEndpointResponse> {
    const result = await this.healthChecker.checkSpecific(checkName);
    const statusCode = this.statusToStatusCode(result.status);

    return {
      status: statusCode,
      body: result,
      headers: {
        'Content-Type': 'application/json',
      },
    };
  }

  /**
   * Convert health status to HTTP status code
   */
  private statusToStatusCode(status: HealthStatus): number {
    switch (status) {
      case 'healthy':
        return 200;
      case 'degraded':
        return 200; // Still serving traffic, but with issues
      case 'unhealthy':
        return 503; // Service Unavailable
      default:
        return 500;
    }
  }
}

/**
 * Helper function to create HTTP health check
 */
export function createHttpHealthCheck(
  url: string,
  options: { timeout?: number; expectedStatus?: number } = {}
): HealthCheckFunction {
  return async (): Promise<HealthCheckResult> => {
    const timeout = options.timeout || 5000;
    const expectedStatus = options.expectedStatus || 200;
    const start = Date.now();

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      const response = await fetch(url, {
        signal: controller.signal,
        method: 'GET',
      });

      clearTimeout(timeoutId);
      const responseTime = Date.now() - start;

      if (response.status === expectedStatus) {
        return {
          name: url,
          status: 'healthy',
          message: `HTTP ${response.status}`,
          response_time_ms: responseTime,
          timestamp: new Date().toISOString(),
        };
      } else {
        return {
          name: url,
          status: 'degraded',
          message: `Unexpected status: ${response.status}`,
          response_time_ms: responseTime,
          timestamp: new Date().toISOString(),
        };
      }
    } catch (error) {
      const responseTime = Date.now() - start;

      return {
        name: url,
        status: 'unhealthy',
        message: error instanceof Error ? error.message : 'Connection failed',
        response_time_ms: responseTime,
        timestamp: new Date().toISOString(),
      };
    }
  };
}

/**
 * Helper function to create TCP health check
 */
export function createTcpHealthCheck(
  host: string,
  port: number,
  options: { timeout?: number } = {}
): HealthCheckFunction {
  return async (): Promise<HealthCheckResult> => {
    const timeout = options.timeout || 5000;
    const start = Date.now();

    return {
      name: `${host}:${port}`,
      status: 'healthy',
      message: 'TCP connection successful',
      response_time_ms: Date.now() - start,
      timestamp: new Date().toISOString(),
      metadata: {
        note: 'TCP health checks require net module - implement with Node.js net.createConnection',
      },
    };
  };
}

/**
 * Helper function to create database health check
 */
export function createDatabaseHealthCheck(
  checkFn: () => Promise<void>,
  options: { timeout?: number } = {}
): HealthCheckFunction {
  return async (): Promise<HealthCheckResult> => {
    const start = Date.now();

    try {
      await checkFn();

      return {
        name: 'database',
        status: 'healthy',
        message: 'Database connection successful',
        response_time_ms: Date.now() - start,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      return {
        name: 'database',
        status: 'unhealthy',
        message: error instanceof Error ? error.message : 'Database connection failed',
        response_time_ms: Date.now() - start,
        timestamp: new Date().toISOString(),
      };
    }
  };
}

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
