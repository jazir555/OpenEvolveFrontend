"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TimeoutError = exports.ExecutionError = exports.ValidationError = exports.IntegrationError = void 0;
class IntegrationError extends Error {
    constructor(message, code, details) {
        super(message);
        this.code = code;
        this.details = details;
        this.name = 'IntegrationError';
    }
}
exports.IntegrationError = IntegrationError;
class ValidationError extends IntegrationError {
    constructor(message, errors) {
        super(message, 'VALIDATION_ERROR', { errors });
        this.errors = errors;
        this.name = 'ValidationError';
    }
}
exports.ValidationError = ValidationError;
class ExecutionError extends IntegrationError {
    constructor(message, executionId) {
        super(message, 'EXECUTION_ERROR', { executionId });
        this.executionId = executionId;
        this.name = 'ExecutionError';
    }
}
exports.ExecutionError = ExecutionError;
class TimeoutError extends IntegrationError {
    constructor(message, timeout) {
        super(message, 'TIMEOUT_ERROR', { timeout });
        this.timeout = timeout;
        this.name = 'TimeoutError';
    }
}
exports.TimeoutError = TimeoutError;
//# sourceMappingURL=common.js.map