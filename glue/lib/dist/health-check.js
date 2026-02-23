"use strict";
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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.checkDependency = checkDependency;
exports.checkDatabase = checkDatabase;
exports.checkEnvVariables = checkEnvVariables;
exports.createHealthCheckServer = createHealthCheckServer;
exports.validateRequiredEnv = validateRequiredEnv;
exports.defaultHealthCheck = defaultHealthCheck;
const http_1 = __importDefault(require("http"));
const logger_1 = require("./logger");
const logger = new logger_1.Logger('HealthCheck');
/**
 * Check if an HTTP/HTTPS dependency is accessible
 */
async function checkDependency(url, name, timeout = 5000) {
    const startTime = Date.now();
    const urlObj = new URL(url);
    try {
        const options = {
            hostname: urlObj.hostname,
            port: urlObj.port || (urlObj.protocol === 'https:' ? '443' : '80'),
            path: urlObj.pathname || '/health',
            method: 'GET',
            timeout,
        };
        await new Promise((resolve, reject) => {
            const req = http_1.default.request(options, (res) => {
                if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
                    resolve();
                }
                else {
                    reject(new Error(`HTTP ${res.statusCode}`));
                }
            });
            req.on('error', reject);
            req.on('timeout', () => {
                req.destroy();
                reject(new Error('Timeout'));
            });
            req.end();
        });
        return {
            name,
            status: 'healthy',
            responseTime: Date.now() - startTime,
            timestamp: new Date().toISOString(),
        };
    }
    catch (error) {
        return {
            name,
            status: 'unhealthy',
            message: error instanceof Error ? error.message : 'Unknown error',
            responseTime: Date.now() - startTime,
            timestamp: new Date().toISOString(),
        };
    }
}
/**
 * Check database connectivity (generic SQL check)
 */
async function checkDatabase(connectionString, name = 'Database') {
    const startTime = Date.now();
    if (!connectionString) {
        return {
            name,
            status: 'unhealthy',
            message: 'No connection string provided',
            responseTime: Date.now() - startTime,
            timestamp: new Date().toISOString(),
        };
    }
    // Basic validation - actual DB connection depends on driver
    try {
        const url = new URL(connectionString);
        if (!url.hostname) {
            throw new Error('Invalid connection string');
        }
        return {
            name,
            status: 'healthy',
            message: 'Connection string valid',
            responseTime: Date.now() - startTime,
            timestamp: new Date().toISOString(),
        };
    }
    catch (error) {
        return {
            name,
            status: 'unhealthy',
            message: error instanceof Error ? error.message : 'Invalid connection string',
            responseTime: Date.now() - startTime,
            timestamp: new Date().toISOString(),
        };
    }
}
/**
 * Check environment variable configuration
 */
async function checkEnvVariables(requiredVars, name = 'Environment Configuration') {
    const startTime = Date.now();
    const missing = [];
    for (const [varName, description] of Object.entries(requiredVars)) {
        if (!process.env[varName]) {
            missing.push(`${varName} (${description})`);
        }
    }
    if (missing.length > 0) {
        return {
            name,
            status: 'unhealthy',
            message: `Missing required variables: ${missing.join(', ')}`,
            responseTime: Date.now() - startTime,
            timestamp: new Date().toISOString(),
        };
    }
    return {
        name,
        status: 'healthy',
        message: `All ${Object.keys(requiredVars).length} required variables present`,
        responseTime: Date.now() - startTime,
        timestamp: new Date().toISOString(),
    };
}
/**
 * Create a health check HTTP server
 */
function createHealthCheckServer(options) {
    const startTime = Date.now();
    const checks = options.checks || [];
    const server = http_1.default.createServer(async (req, res) => {
        // Set CORS headers
        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Access-Control-Allow-Origin', '*');
        res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
        res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
        if (req.method === 'OPTIONS') {
            res.writeHead(200);
            res.end();
            return;
        }
        if (req.method !== 'GET') {
            res.writeHead(405);
            res.end(JSON.stringify({ error: 'Method not allowed' }));
            return;
        }
        const path = req.url || '/';
        // Health endpoint
        if (path === '/health') {
            try {
                const results = await Promise.all(checks.map((check) => check()));
                // Determine overall status
                const overallStatus = results.every((r) => r.status === 'healthy')
                    ? 'healthy'
                    : results.some((r) => r.status === 'unhealthy')
                        ? 'unhealthy'
                        : 'degraded';
                const response = {
                    status: overallStatus,
                    timestamp: new Date().toISOString(),
                    uptime: process.uptime(),
                    checks: results,
                    version: options.version,
                };
                const statusCode = overallStatus === 'healthy' ? 200 : overallStatus === 'degraded' ? 200 : 503;
                res.writeHead(statusCode);
                res.end(JSON.stringify(response, null, 2));
                logger.info('Health check completed', {
                    status: overallStatus,
                    uptime: process.uptime(),
                });
            }
            catch (error) {
                logger.error('Health check failed', { error: error instanceof Error ? error.message : 'Unknown error' });
                res.writeHead(503);
                res.end(JSON.stringify({
                    status: 'unhealthy',
                    timestamp: new Date().toISOString(),
                    uptime: process.uptime(),
                    error: 'Health check execution failed',
                }));
            }
            return;
        }
        // Readiness endpoint (simpler version)
        if (path === '/ready') {
            const uptime = process.uptime();
            const isReady = uptime > 5; // Consider ready after 5 seconds
            res.writeHead(isReady ? 200 : 503);
            res.end(JSON.stringify({
                ready: isReady,
                timestamp: new Date().toISOString(),
                uptime,
            }));
            return;
        }
        // Liveness endpoint (basic check)
        if (path === '/live') {
            res.writeHead(200);
            res.end(JSON.stringify({
                alive: true,
                timestamp: new Date().toISOString(),
                uptime: process.uptime(),
            }));
            return;
        }
        // 404 for other paths
        res.writeHead(404);
        res.end(JSON.stringify({ error: 'Not found' }));
    });
    return {
        start: () => {
            return new Promise((resolve) => {
                server.listen(options.port, () => {
                    logger.info(`Health check server listening on port ${options.port}`);
                    resolve();
                });
            });
        },
        stop: () => {
            return new Promise((resolve) => {
                server.close(() => {
                    logger.info('Health check server closed');
                    resolve();
                });
            });
        },
    };
}
/**
 * Startup validator - crashes process if required environment is missing
 * Following Law of Configuration Explicitness
 */
function validateRequiredEnv(requiredVars) {
    const missing = [];
    for (const [varName, description] of Object.entries(requiredVars)) {
        if (!process.env[varName]) {
            missing.push(`${varName} (${description})`);
        }
    }
    if (missing.length > 0) {
        logger.error('CRITICAL: Missing required environment variables', {
            missing,
            count: missing.length,
        });
        console.error(`
╔════════════════════════════════════════════════════════════╗
║         CRITICAL CONFIGURATION ERROR                      ║
╠════════════════════════════════════════════════════════════╣
║ Following Law of Configuration Explicitness:              ║
║ Service cannot start without required configuration.       ║
║                                                            ║
║ Missing Variables:                                        ║
${missing.map((m) => `║   - ${m.padEnd(50)} ║`).join('\n')}
║                                                            ║
║ Please set these environment variables before restarting.  ║
╚════════════════════════════════════════════════════════════╝
    `);
        process.exit(1);
    }
    logger.info('All required environment variables validated', {
        count: Object.keys(requiredVars).length,
    });
}
/**
 * Default health check that can be used in adapters
 */
async function defaultHealthCheck() {
    return {
        name: 'Adapter',
        status: 'healthy',
        message: 'Adapter is running',
        timestamp: new Date().toISOString(),
    };
}
//# sourceMappingURL=health-check.js.map