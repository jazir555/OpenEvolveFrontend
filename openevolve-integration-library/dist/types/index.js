"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ExecutionError = exports.ValidationError = exports.NetworkError = exports.OpenEvolveError = exports.BaseIntegration = void 0;
class BaseIntegration {
    validate(inputs) {
        return {
            valid: true,
            errors: [],
            warnings: []
        };
    }
    async executeStream(inputs, onUpdate) {
        return this.execute(inputs);
    }
}
exports.BaseIntegration = BaseIntegration;
class OpenEvolveError extends Error {
    constructor(code, message, details) {
        super(message);
        this.code = code;
        this.details = details;
        this.name = 'OpenEvolveError';
    }
}
exports.OpenEvolveError = OpenEvolveError;
class NetworkError extends OpenEvolveError {
    constructor(message, details) {
        super('NETWORK_ERROR', message, details);
        this.name = 'NetworkError';
    }
}
exports.NetworkError = NetworkError;
class ValidationError extends OpenEvolveError {
    constructor(message, details) {
        super('VALIDATION_ERROR', message, details);
        this.name = 'ValidationError';
    }
}
exports.ValidationError = ValidationError;
class ExecutionError extends OpenEvolveError {
    constructor(message, details) {
        super('EXECUTION_ERROR', message, details);
        this.name = 'ExecutionError';
    }
}
exports.ExecutionError = ExecutionError;
//# sourceMappingURL=index.js.map