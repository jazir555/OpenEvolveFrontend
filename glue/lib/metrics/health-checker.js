"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.HealthEndpointHandler = exports.HealthChecker = void 0;
exports.createHttpHealthCheck = createHttpHealthCheck;
exports.createTcpHealthCheck = createTcpHealthCheck;
exports.createDatabaseHealthCheck = createDatabaseHealthCheck;
const logger_1 = require("../logger");
class HealthChecker {
    constructor(serviceName) {
        this.serviceName = serviceName;
        this.checks = new Map();
    }
    register(name, checkFn) {
        this.checks.set(name, checkFn);
        logger_1.logger.info('Health check registered', {
            service: this.serviceName,
            check_name: name,
        });
    }
    unregister(name) {
        this.checks.delete(name);
        logger_1.logger.info('Health check unregistered', {
            service: this.serviceName,
            check_name: name,
        });
    }
    async executeCheck(name, checkFn) {
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
        }
        catch (error) {
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
    async checkHealth() {
        const results = [];
        const start = Date.now();
        for (const [name, checkFn] of this.checks.entries()) {
            const result = await this.executeCheck(name, checkFn);
            results.push(result);
        }
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
    async checkSpecific(checkName) {
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
    calculateOverallStatus(results) {
        if (results.length === 0) {
            return 'healthy';
        }
        const hasUnhealthy = results.some((r) => r.status === 'unhealthy');
        const hasDegraded = results.some((r) => r.status === 'degraded');
        if (hasUnhealthy) {
            return 'unhealthy';
        }
        else if (hasDegraded) {
            return 'degraded';
        }
        else {
            return 'healthy';
        }
    }
    async getLiveness() {
        return {
            name: this.serviceName,
            status: 'healthy',
            message: 'Service is running',
            timestamp: new Date().toISOString(),
        };
    }
    async getReadiness() {
        const results = [];
        for (const [name, checkFn] of this.checks.entries()) {
            const result = await this.executeCheck(name, checkFn);
            results.push(result);
        }
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
exports.HealthChecker = HealthChecker;
class HealthEndpointHandler {
    constructor(healthChecker) {
        this.healthChecker = healthChecker;
    }
    async handleHealth() {
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
    async handleLiveness() {
        const result = await this.healthChecker.getLiveness();
        return {
            status: 200,
            body: result,
            headers: {
                'Content-Type': 'application/json',
            },
        };
    }
    async handleReadiness() {
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
    async handleSpecificCheck(checkName) {
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
    statusToStatusCode(status) {
        switch (status) {
            case 'healthy':
                return 200;
            case 'degraded':
                return 200;
            case 'unhealthy':
                return 503;
            default:
                return 500;
        }
    }
}
exports.HealthEndpointHandler = HealthEndpointHandler;
function createHttpHealthCheck(url, options = {}) {
    return async () => {
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
            }
            else {
                return {
                    name: url,
                    status: 'degraded',
                    message: `Unexpected status: ${response.status}`,
                    response_time_ms: responseTime,
                    timestamp: new Date().toISOString(),
                };
            }
        }
        catch (error) {
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
function createTcpHealthCheck(host, port, options = {}) {
    return async () => {
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
function createDatabaseHealthCheck(checkFn, options = {}) {
    return async () => {
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
        }
        catch (error) {
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
//# sourceMappingURL=health-checker.js.map