"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.BaseIntegrationAdapter = void 0;
const errors_1 = require("../api/errors");
const helpers_1 = require("../utils/helpers");
class BaseIntegrationAdapter {
    constructor(client, name, version, description, retryConfig, circuitBreakerConfig) {
        this.circuitState = 'closed';
        this.failureCount = 0;
        this.successCount = 0;
        this.lastFailureTime = 0;
        this.client = client;
        this.name = name;
        this.version = version;
        this.description = description;
        this.retryConfig = retryConfig;
        this.circuitBreakerConfig = {
            enabled: circuitBreakerConfig?.enabled ?? true,
            failureThreshold: circuitBreakerConfig?.failureThreshold ?? 5,
            resetTimeout: circuitBreakerConfig?.resetTimeout ?? 30000,
            successThreshold: circuitBreakerConfig?.successThreshold ?? 2,
        };
    }
    setGlobalErrorHandler(handler) {
        this.onGlobalError = handler;
    }
    checkCircuit() {
        if (!this.circuitBreakerConfig.enabled)
            return;
        if (this.circuitState === 'open') {
            const now = Date.now();
            if (now - this.lastFailureTime > this.circuitBreakerConfig.resetTimeout) {
                this.client.log?.(`[${this.name}] Circuit breaker moving to half-open state`);
                this.circuitState = 'half-open';
                this.successCount = 0;
            }
            else {
                throw new errors_1.CircuitBreakerError(this.name, {
                    cooldownRemaining: this.circuitBreakerConfig.resetTimeout - (now - this.lastFailureTime)
                });
            }
        }
    }
    recordSuccess() {
        if (!this.circuitBreakerConfig.enabled)
            return;
        if (this.circuitState === 'half-open') {
            this.successCount++;
            if (this.successCount >= this.circuitBreakerConfig.successThreshold) {
                this.client.log?.(`[${this.name}] Circuit breaker closed`);
                this.circuitState = 'closed';
                this.failureCount = 0;
            }
        }
        else if (this.circuitState === 'closed') {
            this.failureCount = 0;
        }
    }
    recordFailure(error) {
        if (!this.circuitBreakerConfig.enabled)
            return;
        const integrationError = error instanceof errors_1.IntegrationError ? error : this.handleError(error);
        const criticalCodes = ['CONNECTION_ERROR', 'TIMEOUT_ERROR', 'NETWORK_ERROR', 'EXECUTION_ERROR'];
        if (!criticalCodes.includes(integrationError.code)) {
            return;
        }
        this.failureCount++;
        this.lastFailureTime = Date.now();
        if (this.circuitState === 'closed' && this.failureCount >= this.circuitBreakerConfig.failureThreshold) {
            this.client.log?.(`[${this.name}] Circuit breaker opened for ${this.name}`);
            this.circuitState = 'open';
        }
        else if (this.circuitState === 'half-open') {
            this.client.log?.(`[${this.name}] Circuit breaker re-opened for ${this.name}`);
            this.circuitState = 'open';
        }
    }
    getCircuitState() {
        return this.circuitState;
    }
    getName() {
        return this.name;
    }
    getVersion() {
        return this.version;
    }
    getDescription() {
        return this.description;
    }
    async validate(inputs) {
        return (0, helpers_1.validateInputs)(inputs, this.getSchema());
    }
    async executeStream(inputs, _onProgress, options) {
        const validation = await this.validate(inputs);
        if (!validation.valid) {
            throw new errors_1.ValidationError(this.name, validation.errors);
        }
        return this.execute(inputs, options);
    }
    async healthCheck() {
        const startTime = Date.now();
        try {
            if (!this.client) {
                throw new Error('Backend client not initialized');
            }
            const isOnline = await this.client.ping();
            return {
                name: this.name,
                status: isOnline ? 'available' : 'unavailable',
                responseTime: Date.now() - startTime,
                lastError: undefined,
                endpoints: this.getEndpoints(),
            };
        }
        catch (error) {
            return {
                name: this.name,
                status: 'unavailable',
                responseTime: Date.now() - startTime,
                lastError: error instanceof Error ? error.message : String(error || 'Unknown health check error'),
                endpoints: [],
            };
        }
    }
    async requestBackend(method, endpoint, data, options) {
        this.checkCircuit();
        const maxRetries = options?.retries ?? this.retryConfig?.maxAttempts ?? 3;
        try {
            const result = await (0, helpers_1.retryWithBackoff)(async () => {
                const abortController = new AbortController();
                let timeoutId;
                if (options?.timeout) {
                    timeoutId = setTimeout(() => {
                        abortController.abort();
                    }, options.timeout);
                }
                try {
                    const transformedData = data ? this.transformRequest(data) : undefined;
                    let response;
                    const axiosConfig = {
                        signal: options?.signal || abortController.signal,
                        timeout: options?.timeout,
                    };
                    switch (method) {
                        case 'GET':
                            response = await this.client.get(endpoint, axiosConfig);
                            break;
                        case 'POST':
                            response = await this.client.post(endpoint, transformedData, axiosConfig);
                            break;
                        case 'PUT':
                            response = await this.client.put(endpoint, transformedData, axiosConfig);
                            break;
                        case 'DELETE':
                            response = await this.client.delete(endpoint, axiosConfig);
                            break;
                        case 'PATCH':
                            response = await this.client.patch(endpoint, transformedData, axiosConfig);
                            break;
                    }
                    if (timeoutId)
                        clearTimeout(timeoutId);
                    try {
                        const result = this.transformResponse(response);
                        const validation = this.validateResponse(result);
                        if (!validation.valid) {
                            throw new errors_1.ParseError(this.name, 'Response validation failed', { errors: validation.errors });
                        }
                        return result;
                    }
                    catch (parseError) {
                        if (parseError instanceof errors_1.IntegrationError)
                            throw parseError;
                        throw new errors_1.ParseError(this.name, 'Failed to parse or transform backend response', {
                            originalError: parseError,
                            responseData: response
                        });
                    }
                }
                catch (error) {
                    if (timeoutId)
                        clearTimeout(timeoutId);
                    if (error.name === 'AbortError' || error.code === 'ECONNABORTED') {
                        throw new errors_1.TimeoutError(this.name, options?.timeout || 0);
                    }
                    throw this.handleError(error);
                }
            }, maxRetries > 0 ? maxRetries - 1 : 0, this.retryConfig?.initialDelay || 1000, (error) => {
                const integrationError = error instanceof errors_1.IntegrationError ? error : this.handleError(error);
                const retryableCodes = ['NETWORK_ERROR', 'TIMEOUT_ERROR', 'RATE_LIMIT_ERROR', 'CONNECTION_ERROR'];
                const shouldRetry = retryableCodes.includes(integrationError.code) ||
                    (integrationError.code === 'EXECUTION_ERROR' && integrationError.message.includes('Server error'));
                return shouldRetry;
            }, (error, attempt, delay) => {
                if (options?.onRetry) {
                    try {
                        options.onRetry(error instanceof errors_1.IntegrationError ? error : this.handleError(error), attempt, delay);
                    }
                    catch (cbError) {
                    }
                }
            });
            this.recordSuccess();
            return result;
        }
        catch (error) {
            this.recordFailure(error);
            throw error;
        }
    }
    async executeBackend(endpoint, request, executionId, options) {
        return this.requestBackend('POST', endpoint, executionId ? { ...request, executionId } : request, options);
    }
    async streamExecute(endpoint, request, onProgress, options) {
        this.checkCircuit();
        return new Promise((resolve, reject) => {
            const executionId = request.executionId || (0, helpers_1.generateId)();
            let isFinalized = false;
            let timeoutId;
            const finalize = (fn) => {
                if (isFinalized)
                    return;
                isFinalized = true;
                if (timeoutId)
                    clearTimeout(timeoutId);
                if (ws)
                    ws.disconnect();
                fn();
            };
            const handlers = {
                onConnect: () => {
                    this.client.log?.(`[${this.name}] WebSocket connected for execution ${executionId}`);
                },
                onError: (error) => {
                    this.recordFailure(error);
                    finalize(() => reject(this.handleError(error)));
                },
                onMessage: (message) => {
                    if (message.executionId !== executionId)
                        return;
                    if (message.type === 'progress') {
                        onProgress(message.data);
                    }
                    else if (message.type === 'complete') {
                        try {
                            const result = this.transformResponse(message.data);
                            const validation = this.validateResponse(result);
                            if (!validation.valid) {
                                throw new errors_1.ParseError(this.name, 'Stream response validation failed', { errors: validation.errors });
                            }
                            this.recordSuccess();
                            finalize(() => resolve(result));
                        }
                        catch (parseError) {
                            const integrationError = parseError instanceof errors_1.IntegrationError
                                ? parseError
                                : new errors_1.ParseError(this.name, 'Failed to parse or transform stream response', { originalError: parseError });
                            this.recordFailure(integrationError);
                            finalize(() => reject(integrationError));
                        }
                    }
                    else if (message.type === 'error') {
                        const errorMessage = message.data?.message || 'Unknown stream error';
                        const error = new Error(errorMessage);
                        this.recordFailure(error);
                        finalize(() => reject(this.handleError(error)));
                    }
                }
            };
            const ws = this.client.websocket(`/ws/${this.name}/${executionId}`, handlers);
            this.client.post(endpoint, { ...request, executionId }, { signal: options?.signal }).catch((error) => {
                finalize(() => reject(this.handleError(error)));
            });
            if (options?.signal) {
                options.signal.addEventListener('abort', () => {
                    finalize(() => reject(new errors_1.CancellationError(this.name, executionId)));
                });
            }
            if (options?.timeout) {
                timeoutId = setTimeout(() => {
                    finalize(() => reject(new errors_1.TimeoutError(this.name, options.timeout)));
                }, options.timeout);
            }
        });
    }
    handleError(error) {
        const integrationError = (0, errors_1.createIntegrationError)(this.name, error);
        if (this.onGlobalError) {
            try {
                this.onGlobalError(integrationError);
            }
            catch (cbError) {
            }
        }
        throw integrationError;
    }
    transformRequest(data) {
        return data;
    }
    transformResponse(data) {
        return data;
    }
    validateResponse(_data) {
        return { valid: true };
    }
    validateRequired(inputs, requiredFields) {
        const errors = [];
        if (!inputs || typeof inputs !== 'object') {
            return ['Inputs must be an object'];
        }
        for (const field of requiredFields) {
            if (inputs[field] === undefined || inputs[field] === null) {
                errors.push(`Required field '${String(field)}' is missing`);
            }
        }
        return errors;
    }
    validateTypes(inputs, typeDefinitions) {
        const errors = [];
        for (const [field, expectedType] of Object.entries(typeDefinitions)) {
            const value = inputs[field];
            if (value !== undefined && value !== null) {
                let isValid = false;
                const actualType = typeof value;
                if (expectedType === 'array') {
                    isValid = Array.isArray(value);
                }
                else if (expectedType === 'object') {
                    isValid = actualType === 'object' && !Array.isArray(value);
                }
                else {
                    isValid = actualType === expectedType;
                }
                if (!isValid) {
                    errors.push(`Field '${field}' has invalid type: expected ${expectedType}, got ${Array.isArray(value) ? 'array' : actualType}`);
                }
            }
        }
        return errors;
    }
    validateEnum(inputs, enumDefinitions) {
        const errors = [];
        if (!inputs || typeof inputs !== 'object') {
            return [];
        }
        for (const [field, validValues] of Object.entries(enumDefinitions)) {
            const values = validValues;
            const value = inputs[field];
            if (value !== undefined && !values.includes(value)) {
                const displayValue = (0, helpers_1.isPlainObject)(value) ? JSON.stringify(value) : String(value);
                errors.push(`Field '${field}' has invalid value: ${displayValue}. Valid values are: ${values.join(', ')}`);
            }
        }
        return errors;
    }
    validateRanges(inputs, rangeDefinitions) {
        const errors = [];
        for (const [field, rangeDef] of Object.entries(rangeDefinitions)) {
            const range = rangeDef;
            const value = inputs[field];
            if (typeof value === 'number') {
                if (range.min !== undefined && value < range.min) {
                    errors.push(`Field '${field}' must be at least ${range.min}, got ${value}`);
                }
                if (range.max !== undefined && value > range.max) {
                    errors.push(`Field '${field}' must be at most ${range.max}, got ${value}`);
                }
            }
        }
        return errors;
    }
}
exports.BaseIntegrationAdapter = BaseIntegrationAdapter;
//# sourceMappingURL=base.js.map