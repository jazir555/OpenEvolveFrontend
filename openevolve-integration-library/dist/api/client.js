"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(exports, p)) __createBinding(exports, m, p);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.BackendClient = exports.OpenEvolveClient = exports.IntegrationName = void 0;
exports.createOpenEvolveClient = createOpenEvolveClient;
const uuid_1 = require("uuid");
const backend_1 = require("./backend");
const errors_1 = require("./errors");
const integrations_1 = require("../integrations");
var IntegrationName;
(function (IntegrationName) {
    IntegrationName["LEANAIDE"] = "leanaide";
    IntegrationName["EVOLUTION"] = "evolution";
    IntegrationName["KNOWLEDGE"] = "knowledge";
    IntegrationName["MAKER"] = "maker";
    IntegrationName["HEPHAESTUS"] = "hephaestus";
    IntegrationName["DECOMPOSITION"] = "decomposition";
    IntegrationName["VERIFICATION"] = "verification";
    IntegrationName["ASSEMBLY"] = "assembly";
    IntegrationName["SOLUTION"] = "solution";
})(IntegrationName || (exports.IntegrationName = IntegrationName = {}));
const DEFAULT_RETRY_CONFIG = {
    maxAttempts: 3,
    initialDelay: 1000,
    maxDelay: 10000,
    backoffMultiplier: 2,
    retryOn4xx: false,
    retryOn5xx: true,
    retryableStatusCodes: [408, 429, 500, 502, 503, 504],
};
const MAX_METRICS_SIZE = 1000;
class OpenEvolveClient {
    constructor(config) {
        this.connectionState = 'disconnected';
        this.executionMetrics = new Map();
        this.progressCallbacks = new Map();
        this.errorHandlers = new Set();
        this.middleware = [];
        this.healthCheckTimer = null;
        this.integrationAdapters = new Map();
        this.executionMetrics = new Map();
        this.progressCallbacks = new Map();
        this.errorHandlers = new Set();
        if (config.onError) {
            this.errorHandlers.add(config.onError);
        }
        this.middleware = config.middleware || [];
        this.retryConfig = DEFAULT_RETRY_CONFIG;
        this.circuitBreakerConfig = config.circuitBreakerConfig || {};
        this.debug = config.debug || false;
        this.config = config;
        try {
            this.backend = new backend_1.BackendClient({
                baseUrl: config.baseUrl,
                timeout: config.timeout || 30000,
                apiKey: config.apiKey,
                debug: this.debug,
                headers: config.headers,
                requestTransform: config.requestTransform,
                responseTransform: config.responseTransform,
            });
            this.loadIntegrations();
            if (config.enableWebSocket !== false) {
                try {
                    this.setupWebSocket();
                }
                catch (wsError) {
                    this.log('Failed to setup WebSocket, client will continue in HTTP-only mode', wsError);
                }
            }
            this.log('OpenEvolve client initialized');
            if (config.healthCheckInterval && config.healthCheckInterval > 0) {
                this.startHealthCheck(config.healthCheckInterval);
            }
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            console.error(`[OpenEvolveClient] Critical failure during initialization: ${errorMessage}`);
            throw (0, errors_1.createIntegrationError)('client_init', error);
        }
    }
    loadIntegrations() {
        const retryConfig = this.config.retryConfig || DEFAULT_RETRY_CONFIG;
        const cbConfig = this.circuitBreakerConfig;
        const integrations = [
            { name: IntegrationName.LEANAIDE, adapter: new integrations_1.LeanAideIntegration(this.backend, retryConfig, cbConfig) },
            { name: IntegrationName.EVOLUTION, adapter: new integrations_1.EvolutionIntegration(this.backend, retryConfig, cbConfig) },
            { name: IntegrationName.KNOWLEDGE, adapter: new integrations_1.KnowledgeIntegration(this.backend, retryConfig, cbConfig) },
            { name: IntegrationName.MAKER, adapter: new integrations_1.MakerIntegration(this.backend, retryConfig, cbConfig) },
            { name: IntegrationName.HEPHAESTUS, adapter: new integrations_1.HephaestusIntegration(this.backend, retryConfig, cbConfig) },
            { name: IntegrationName.DECOMPOSITION, adapter: new integrations_1.DecompositionIntegration(this.backend, retryConfig, cbConfig) },
            { name: IntegrationName.VERIFICATION, adapter: new integrations_1.VerificationIntegration(this.backend, retryConfig, cbConfig) },
            { name: IntegrationName.ASSEMBLY, adapter: new integrations_1.AssemblyIntegration(this.backend, retryConfig, cbConfig) },
            { name: IntegrationName.SOLUTION, adapter: new integrations_1.SolutionIntegration(this.backend, retryConfig, cbConfig) },
        ];
        integrations.forEach(({ name, adapter }) => {
            adapter.setGlobalErrorHandler((error) => {
                this.handleExecutionError('direct-call', error, name);
            });
            this.integrationAdapters.set(name, adapter);
        });
        this.log('Integration adapters loaded');
    }
    setupWebSocket() {
        const handlers = {
            onConnect: () => {
                this.connectionState = 'connected';
                this.log('WebSocket connected');
            },
            onDisconnect: (reason) => {
                this.connectionState = 'disconnected';
                this.log('WebSocket disconnected:', { reason });
            },
            onError: (error) => {
                this.log('WebSocket error:', error);
            },
            onMessage: (message) => {
                this.handleWebSocketMessage(message);
            },
            onReconnect: (attemptNumber) => {
                this.connectionState = 'reconnecting';
                this.log('WebSocket reconnecting:', { attemptNumber });
            },
        };
        this.backend.websocket('/ws', handlers);
        this.connectionState = 'connecting';
    }
    startHealthCheck(interval) {
        this.stopHealthCheck();
        this.healthCheckTimer = setInterval(async () => {
            try {
                const health = await this.healthCheck();
                if (health.status === 'unhealthy') {
                    this.handleExecutionError('background-health', new Error('Backend reported unhealthy status'), 'backend');
                }
            }
            catch (error) {
                this.log('Background health check failed', error);
            }
        }, interval);
        if (this.healthCheckTimer.unref) {
            this.healthCheckTimer.unref();
        }
    }
    stopHealthCheck() {
        if (this.healthCheckTimer) {
            clearInterval(this.healthCheckTimer);
            this.healthCheckTimer = null;
        }
    }
    addErrorHandler(handler) {
        this.errorHandlers.add(handler);
    }
    removeErrorHandler(handler) {
        this.errorHandlers.delete(handler);
    }
    handleWebSocketMessage(message) {
        try {
            this.log('WebSocket message received:', message);
            switch (message.type) {
                case 'progress':
                    this.handleProgressUpdate(message.data);
                    break;
                case 'complete':
                    this.handleExecutionComplete(message.executionId, message.data);
                    break;
                case 'error':
                    this.handleExecutionError(message.executionId, message.data, message.integration);
                    break;
                case 'status':
                    break;
            }
        }
        catch (error) {
            this.log('Error handling WebSocket message:', error);
        }
    }
    handleProgressUpdate(update) {
        try {
            this.log('Progress update:', update);
            const callback = this.progressCallbacks.get(update.executionId);
            if (callback) {
                callback(update);
            }
        }
        catch (error) {
            this.log('Error in progress callback:', error);
        }
    }
    handleExecutionComplete(executionId, result) {
        try {
            this.log('Execution complete:', { executionId, result });
            const metrics = this.executionMetrics.get(executionId);
            if (metrics) {
                metrics.endTime = new Date().toISOString();
                metrics.duration = Date.now() - new Date(metrics.startTime).getTime();
                metrics.success = true;
                this.executionMetrics.set(executionId, metrics);
            }
        }
        catch (error) {
            this.log('Error handling execution complete:', error);
        }
    }
    handleExecutionSuccess(executionId) {
        try {
            this.log('Execution success:', { executionId });
            const metrics = this.executionMetrics.get(executionId);
            if (metrics) {
                metrics.endTime = new Date().toISOString();
                metrics.duration = Date.now() - new Date(metrics.startTime).getTime();
                metrics.success = true;
                this.executionMetrics.set(executionId, metrics);
            }
        }
        catch (error) {
            this.log('Error handling execution success:', error);
        }
    }
    handleExecutionError(executionId, error, integration) {
        try {
            const integrationError = (0, errors_1.createIntegrationError)(integration || 'unknown', error);
            if (!error || !error._isMetricsTracked) {
                const metrics = this.executionMetrics.get(executionId);
                if (metrics) {
                    metrics.endTime = new Date().toISOString();
                    metrics.duration = Date.now() - new Date(metrics.startTime).getTime();
                    metrics.success = false;
                    metrics.error = integrationError.message;
                    this.executionMetrics.set(executionId, metrics);
                }
                if (error && typeof error === 'object')
                    error._isMetricsTracked = true;
            }
            this.log('Execution error:', { executionId, error: integrationError });
            this.errorHandlers.forEach(handler => {
                try {
                    handler(integrationError);
                }
                catch (cbError) {
                    this.log('Error in global error handler:', cbError);
                }
            });
        }
        catch (err) {
            this.log('Error handling execution error:', err);
        }
    }
    async execute(integration, inputs, options) {
        const executionId = options?.executionId || (0, uuid_1.v4)();
        const startTime = new Date().toISOString();
        this.executionMetrics.set(executionId, {
            requestId: executionId,
            integration: integration,
            startTime,
            endTime: '',
            duration: 0,
            retries: 0,
            success: false,
        });
        this.log('Executing integration:', { integration, executionId, inputs });
        this.clearOldMetrics();
        try {
            const result = await this.runMiddleware({ integration: integration, inputs, options, executionId }, async () => {
                await this.validateInputs(integration, inputs);
                const adapter = this.getIntegration(integration);
                const executionOptions = { ...options, executionId };
                const execResult = await adapter.execute(inputs, executionOptions);
                if (options?.onComplete) {
                    try {
                        options.onComplete(execResult);
                    }
                    catch (cbError) {
                        this.log('Error in onComplete callback:', cbError);
                    }
                }
                return execResult;
            });
            this.handleExecutionSuccess(executionId);
            return result;
        }
        catch (error) {
            const integrationError = (0, errors_1.createIntegrationError)(integration, error);
            this.handleExecutionError(executionId, integrationError, integration);
            if (options?.onError) {
                try {
                    options.onError(integrationError);
                }
                catch (cbError) {
                    this.log('Error in onError callback:', cbError);
                }
            }
            throw integrationError;
        }
    }
    async runMiddleware(context, finalAction) {
        let index = -1;
        const dispatch = async (i) => {
            if (i <= index) {
                throw new Error('next() called multiple times in middleware pipeline');
            }
            index = i;
            try {
                const fn = i === this.middleware.length ? finalAction : this.middleware[i];
                if (!fn) {
                    if (i === this.middleware.length) {
                        throw new Error('Final action missing in middleware pipeline');
                    }
                    return await dispatch(i + 1);
                }
                if (i === this.middleware.length) {
                    return await fn();
                }
                return await fn(context, () => dispatch(i + 1));
            }
            catch (error) {
                throw (0, errors_1.createIntegrationError)(context.integration, error);
            }
        };
        return await dispatch(0);
    }
    async executeStream(integration, inputs, onProgress, options) {
        const executionId = options?.executionId || (0, uuid_1.v4)();
        const startTime = new Date().toISOString();
        this.executionMetrics.set(executionId, {
            requestId: executionId,
            integration: integration,
            startTime,
            endTime: '',
            duration: 0,
            retries: 0,
            success: false,
        });
        this.log('Executing integration with streaming:', { integration, executionId });
        this.clearOldMetrics();
        try {
            const result = await this.runMiddleware({ integration, inputs, options, executionId }, async () => {
                await this.validateInputs(integration, inputs);
                const adapter = this.getIntegration(integration);
                this.progressCallbacks.set(executionId, onProgress);
                try {
                    return await adapter.executeStream(inputs, onProgress, { ...options, executionId });
                }
                finally {
                    this.progressCallbacks.delete(executionId);
                }
            });
            this.handleExecutionSuccess(executionId);
            return result;
        }
        catch (error) {
            this.handleExecutionError(executionId, error, integration);
            throw (0, errors_1.createIntegrationError)(integration, error);
        }
    }
    async executeBatch(requests) {
        if (!Array.isArray(requests)) {
            this.log('Invalid batch request: expected array');
            return [];
        }
        this.log('Executing batch:', { count: requests.length });
        const results = await Promise.allSettled(requests.map(async (request) => {
            const startTime = Date.now();
            const requestId = request.id || (0, uuid_1.v4)();
            try {
                if (!request.integration) {
                    throw new Error('Integration name is required for batch request');
                }
                const result = await this.execute(request.integration, request.inputs, { ...request.options, executionId: requestId });
                return {
                    id: requestId,
                    result,
                    error: null,
                    executionTime: Date.now() - startTime,
                    success: true,
                };
            }
            catch (error) {
                return {
                    id: requestId,
                    result: null,
                    error: (0, errors_1.createIntegrationError)(request.integration || 'batch', error),
                    executionTime: Date.now() - startTime,
                    success: false,
                };
            }
        })).then((settledResults) => settledResults.map((result, index) => {
            if (result.status === 'fulfilled') {
                return result.value;
            }
            else {
                const request = requests[index];
                return {
                    id: request?.id || `failed-${index}`,
                    result: null,
                    error: (0, errors_1.createIntegrationError)(request?.integration || 'batch', result.reason),
                    executionTime: 0,
                    success: false,
                };
            }
        }));
        return results;
    }
    async healthCheck() {
        this.log('Performing health check');
        const backendStatus = await this.backend.getStatus();
        const integrationNames = Array.from(this.integrationAdapters.keys());
        const healthResults = await Promise.all(Array.from(this.integrationAdapters.values()).map(integration => integration.healthCheck().catch(error => ({
            name: integration.name,
            status: 'unavailable',
            responseTime: 0,
            lastError: error instanceof Error ? error.message : String(error),
            endpoints: [],
        }))));
        const integrationHealth = {};
        healthResults.forEach((result, index) => {
            integrationHealth[integrationNames[index]] = result;
        });
        return {
            status: backendStatus.online ? 'healthy' : 'unhealthy',
            backend: backendStatus,
            integrations: integrationHealth,
            timestamp: new Date().toISOString(),
        };
    }
    async connect() {
        this.log('Connecting to backend');
        this.connectionState = 'connecting';
        try {
            const isOnline = await this.backend.ping();
            if (!isOnline) {
                throw new errors_1.ConnectionError('backend', 'Backend is not responding');
            }
            if (this.config.enableWebSocket !== false && !this.backend.isWebSocketConnected()) {
                this.setupWebSocket();
            }
            this.connectionState = 'connected';
            this.log('Connected successfully');
        }
        catch (error) {
            this.connectionState = 'disconnected';
            throw (0, errors_1.createIntegrationError)('backend', error);
        }
    }
    async disconnect() {
        this.log('Disconnecting from backend');
        this.connectionState = 'disconnecting';
        this.stopHealthCheck();
        try {
            this.backend.disconnectWebSocket();
            this.connectionState = 'disconnected';
            this.log('Disconnected successfully');
        }
        catch (error) {
            throw (0, errors_1.createIntegrationError)('backend', error);
        }
    }
    getVersions() {
        const versions = {};
        for (const [name, adapter] of this.integrationAdapters.entries()) {
            versions[name] = adapter.getVersion();
        }
        return versions;
    }
    isConnected() {
        return this.connectionState === 'connected';
    }
    getConnectionState() {
        return this.connectionState;
    }
    async validateInputs(integration, inputs) {
        const adapter = this.integrationAdapters.get(integration);
        if (!adapter) {
            throw new errors_1.IntegrationError(integration, 'INTEGRATION_NOT_FOUND', `Integration '${integration}' not found in registry`);
        }
        const result = await adapter.validate(inputs);
        if (!result.valid) {
            throw new errors_1.ValidationError(integration, result.errors);
        }
    }
    getIntegration(name) {
        const integration = this.integrationAdapters.get(name);
        if (!integration) {
            throw new errors_1.IntegrationError(name, 'INTEGRATION_NOT_FOUND', `Integration '${name}' not found`);
        }
        return integration;
    }
    get integrations() {
        return {
            leanaide: this.getIntegration(IntegrationName.LEANAIDE),
            evolution: this.getIntegration(IntegrationName.EVOLUTION),
            knowledge: this.getIntegration(IntegrationName.KNOWLEDGE),
            maker: this.getIntegration(IntegrationName.MAKER),
            hephaestus: this.getIntegration(IntegrationName.HEPHAESTUS),
            decomposition: this.getIntegration(IntegrationName.DECOMPOSITION),
            verification: this.getIntegration(IntegrationName.VERIFICATION),
            assembly: this.getIntegration(IntegrationName.ASSEMBLY),
            solution: this.getIntegration(IntegrationName.SOLUTION),
        };
    }
    getMetrics(executionId) {
        return this.executionMetrics.get(executionId) || null;
    }
    getAllMetrics() {
        return this.executionMetrics;
    }
    getMetricsSummary() {
        const metrics = Array.from(this.executionMetrics.values());
        const totalRequests = metrics.length;
        if (totalRequests === 0) {
            return { totalRequests: 0, successRate: 0, averageDuration: 0, totalRetries: 0 };
        }
        const successes = metrics.filter(m => m.success).length;
        const totalDuration = metrics.reduce((sum, m) => sum + m.duration, 0);
        const totalRetries = metrics.reduce((sum, m) => sum + (m.retries || 0), 0);
        return {
            totalRequests,
            successRate: successes / totalRequests,
            averageDuration: totalDuration / totalRequests,
            totalRetries,
        };
    }
    clearMetrics() {
        this.executionMetrics.clear();
    }
    updateRetryConfig(config) {
        this.retryConfig = { ...this.retryConfig, ...config };
        this.log('Retry configuration updated:', this.retryConfig);
    }
    log(message, data) {
        if (this.debug) {
            if (data) {
                console.log(`[OpenEvolveClient] ${message}`, data);
            }
            else {
                console.log(`[OpenEvolveClient] ${message}`);
            }
        }
    }
    clearOldMetrics() {
        if (this.executionMetrics.size >= MAX_METRICS_SIZE) {
            const keysToRemove = Array.from(this.executionMetrics.keys()).slice(0, 100);
            for (const key of keysToRemove) {
                this.executionMetrics.delete(key);
            }
        }
    }
    getBackend() {
        return this.backend;
    }
}
exports.OpenEvolveClient = OpenEvolveClient;
OpenEvolveClient.VERSION = '1.1.0';
function createOpenEvolveClient(baseUrl) {
    return new OpenEvolveClient({
        baseUrl,
        timeout: 30000,
        retryAttempts: 3,
        enableWebSocket: true,
        debug: false,
    });
}
__exportStar(require("./types"), exports);
__exportStar(require("./errors"), exports);
var backend_2 = require("./backend");
Object.defineProperty(exports, "BackendClient", { enumerable: true, get: function () { return backend_2.BackendClient; } });
//# sourceMappingURL=client.js.map