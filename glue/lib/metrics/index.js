"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getAlertManager = exports.AlertManager = exports.getTracer = exports.Tracer = exports.HealthEndpointHandler = exports.HealthChecker = exports.getMetricsCollector = exports.MetricsCollector = void 0;
exports.initializeMonitoring = initializeMonitoring;
exports.createMetricsMiddleware = createMetricsMiddleware;
exports.createHealthMiddleware = createHealthMiddleware;
exports.createRequestTrackingMiddleware = createRequestTrackingMiddleware;
const metrics_collector_1 = require("./metrics-collector");
Object.defineProperty(exports, "MetricsCollector", { enumerable: true, get: function () { return metrics_collector_1.MetricsCollector; } });
Object.defineProperty(exports, "getMetricsCollector", { enumerable: true, get: function () { return metrics_collector_1.getMetricsCollector; } });
const health_checker_1 = require("./health-checker");
Object.defineProperty(exports, "HealthChecker", { enumerable: true, get: function () { return health_checker_1.HealthChecker; } });
Object.defineProperty(exports, "HealthEndpointHandler", { enumerable: true, get: function () { return health_checker_1.HealthEndpointHandler; } });
const tracer_1 = require("./tracer");
Object.defineProperty(exports, "Tracer", { enumerable: true, get: function () { return tracer_1.Tracer; } });
Object.defineProperty(exports, "getTracer", { enumerable: true, get: function () { return tracer_1.getTracer; } });
const alert_manager_1 = require("./alert-manager");
Object.defineProperty(exports, "AlertManager", { enumerable: true, get: function () { return alert_manager_1.AlertManager; } });
Object.defineProperty(exports, "getAlertManager", { enumerable: true, get: function () { return alert_manager_1.getAlertManager; } });
const logger_1 = require("../logger");
async function initializeMonitoring(config) {
    validateEnvironment();
    const metrics = (0, metrics_collector_1.getMetricsCollector)();
    const health = new health_checker_1.HealthChecker(config.serviceName);
    const tracer = (0, tracer_1.getTracer)(config.serviceName);
    const alerts = (0, alert_manager_1.getAlertManager)(config.serviceName);
    if (config.health?.enabled !== false) {
        registerDefaultHealthChecks(health);
    }
    logger_1.logger.info('Monitoring system initialized', {
        service: config.serviceName,
        prometheus_prefix: config.prometheus?.prefix,
        otel_endpoint: config.otel?.endpoint,
        health_enabled: config.health?.enabled !== false,
        alerts_enabled: config.alerts?.enabled !== false,
    });
    return {
        metrics,
        health,
        tracer,
        alerts,
    };
}
function validateEnvironment() {
    const required = [];
    const optional = {
        PROMETHEUS_PORT: '9090',
        OTEL_EXPORTER_OTLP_ENDPOINT: 'http://localhost:4317',
        SERVICE_NAME: 'unknown-service',
    };
    for (const envVar of required) {
        if (!process.env[envVar]) {
            throw new Error(`Required environment variable ${envVar} is not set`);
        }
    }
    for (const [envVar, defaultValue] of Object.entries(optional)) {
        if (!process.env[envVar]) {
            process.env[envVar] = defaultValue;
        }
    }
}
function registerDefaultHealthChecks(health) {
    health.register('memory', async () => {
        const used = process.memoryUsage();
        const heapUsedMB = used.heapUsed / 1024 / 1024;
        const heapTotalMB = used.heapTotal / 1024 / 1024;
        const usagePercent = (heapUsedMB / heapTotalMB) * 100;
        return {
            name: 'memory',
            status: usagePercent > 90 ? 'unhealthy' : usagePercent > 75 ? 'degraded' : 'healthy',
            message: `Memory usage: ${heapUsedMB.toFixed(2)}MB / ${heapTotalMB.toFixed(2)}MB (${usagePercent.toFixed(1)}%)`,
            timestamp: new Date().toISOString(),
            metadata: {
                heap_used_mb: heapUsedMB,
                heap_total_mb: heapTotalMB,
                usage_percent: usagePercent,
                critical: true,
            },
        };
    });
    health.register('event_loop', async () => {
        const start = process.hrtime.bigint();
        await new Promise((resolve) => setImmediate(resolve));
        const lag = Number(process.hrtime.bigint() - start) / 1000000;
        return {
            name: 'event_loop',
            status: lag > 100 ? 'unhealthy' : lag > 50 ? 'degraded' : 'healthy',
            message: `Event loop lag: ${lag.toFixed(2)}ms`,
            timestamp: new Date().toISOString(),
            metadata: {
                lag_ms: lag,
            },
        };
    });
}
function createMetricsMiddleware() {
    const metrics = (0, metrics_collector_1.getMetricsCollector)();
    return async (req, res, next) => {
        if (req.path === '/metrics') {
            try {
                const metricsText = await metrics.getMetrics();
                res.set('Content-Type', 'text/plain');
                res.send(metricsText);
            }
            catch (error) {
                logger_1.logger.error('Failed to collect metrics', error);
                res.status(500).send('Error collecting metrics');
            }
        }
        else {
            next();
        }
    };
}
function createHealthMiddleware(health) {
    const handler = new health_checker_1.HealthEndpointHandler(health);
    return async (req, res, next) => {
        if (req.path === '/health') {
            const result = await handler.handleHealth();
            res.status(result.status).json(result.body);
        }
        else if (req.path === '/health/live') {
            const result = await handler.handleLiveness();
            res.status(result.status).json(result.body);
        }
        else if (req.path === '/health/ready') {
            const result = await handler.handleReadiness();
            res.status(result.status).json(result.body);
        }
        else if (req.path.startsWith('/health/')) {
            const checkName = req.path.replace('/health/', '');
            const result = await handler.handleSpecificCheck(checkName);
            res.status(result.status).json(result.body);
        }
        else {
            next();
        }
    };
}
function createRequestTrackingMiddleware(serviceName) {
    const metrics = (0, metrics_collector_1.getMetricsCollector)();
    const tracer = (0, tracer_1.getTracer)();
    return async (req, res, next) => {
        const start = Date.now();
        const correlationId = req.headers['x-correlation-id'] || req.headers['x-request-id'] || undefined;
        metrics.setHttpRequestsInProgress(serviceName || 'api', 1);
        res.on('finish', async () => {
            const duration = (Date.now() - start) / 1000;
            const status = res.statusCode;
            const statusCategory = Math.floor(status / 100);
            metrics.recordHttpRequestDuration({
                service: serviceName || 'api',
                operation: req.method,
                status: statusCategory.toString(),
            }, duration);
            metrics.incrementHttpRequests({
                service: serviceName || 'api',
                operation: req.method,
                status: `${statusCategory}xx`,
            });
            metrics.setHttpRequestsInProgress(serviceName || 'api', -1);
            if (status >= 500) {
                metrics.recordError({
                    service: serviceName || 'api',
                    operation: req.method,
                    error_type: 'http_error',
                });
            }
        });
        next();
    };
}
//# sourceMappingURL=index.js.map