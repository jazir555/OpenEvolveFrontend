"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.validateInputs = validateInputs;
exports.deepMerge = deepMerge;
exports.generateId = generateId;
exports.formatDuration = formatDuration;
exports.retryWithBackoff = retryWithBackoff;
exports.sleep = sleep;
exports.debounce = debounce;
exports.throttle = throttle;
exports.parseDuration = parseDuration;
exports.isPlainObject = isPlainObject;
exports.deepClone = deepClone;
exports.pick = pick;
exports.omit = omit;
function validateInputs(inputs, schema) {
    const errors = [];
    const warnings = [];
    if (!inputs || typeof inputs !== 'object' || Array.isArray(inputs)) {
        return {
            valid: false,
            errors: [{
                    field: 'root',
                    message: 'Inputs must be an object',
                    code: 'INVALID_INPUT_TYPE'
                }],
            warnings: []
        };
    }
    if (schema.required) {
        for (const field of schema.required) {
            if (!(field in inputs) || inputs[field] === undefined) {
                errors.push({
                    field,
                    message: `Required field '${field}' is missing`,
                    code: 'REQUIRED_FIELD_MISSING'
                });
            }
        }
    }
    if (schema.properties) {
        for (const [fieldName, value] of Object.entries(inputs)) {
            const property = schema.properties[fieldName];
            if (!property) {
                warnings.push({
                    field: fieldName,
                    message: `Unknown field '${fieldName}'`,
                    code: 'UNKNOWN_FIELD'
                });
                continue;
            }
            if (value !== null && value !== undefined) {
                const typeError = validateType(fieldName, value, property);
                if (typeError) {
                    errors.push(typeError);
                }
                if (property.enum && !property.enum.includes(value)) {
                    errors.push({
                        field: fieldName,
                        message: `Value must be one of: ${property.enum.join(', ')}`,
                        code: 'INVALID_ENUM_VALUE'
                    });
                }
                if (typeof value === 'number') {
                    if (property.minimum !== undefined && value < property.minimum) {
                        errors.push({
                            field: fieldName,
                            message: `Value must be at least ${property.minimum}`,
                            code: 'VALUE_TOO_SMALL'
                        });
                    }
                    if (property.maximum !== undefined && value > property.maximum) {
                        errors.push({
                            field: fieldName,
                            message: `Value must be at most ${property.maximum}`,
                            code: 'VALUE_TOO_LARGE'
                        });
                    }
                }
                if (typeof value === 'string' && property.pattern) {
                    try {
                        const regex = new RegExp(property.pattern);
                        if (!regex.test(value)) {
                            errors.push({
                                field: fieldName,
                                message: `Value does not match required pattern`,
                                code: 'PATTERN_MISMATCH'
                            });
                        }
                    }
                    catch (e) {
                        warnings.push({
                            field: fieldName,
                            message: `Invalid regex pattern: ${property.pattern}`,
                            code: 'INVALID_PATTERN'
                        });
                    }
                }
            }
        }
    }
    return {
        valid: errors.length === 0,
        errors,
        warnings
    };
}
function validateType(fieldName, value, property) {
    const expectedType = property.type;
    if (value === null) {
        if (expectedType === 'null')
            return null;
        return {
            field: fieldName,
            message: `Expected ${expectedType}, got null`,
            code: 'TYPE_MISMATCH'
        };
    }
    if (expectedType === 'array') {
        if (!Array.isArray(value)) {
            return {
                field: fieldName,
                message: `Expected array, got ${typeof value}`,
                code: 'TYPE_MISMATCH'
            };
        }
    }
    if (expectedType === 'object') {
        if (typeof value !== 'object' || Array.isArray(value)) {
            return {
                field: fieldName,
                message: `Expected object, got ${typeof value}`,
                code: 'TYPE_MISMATCH'
            };
        }
    }
    if (expectedType === 'string' && typeof value !== 'string') {
        return {
            field: fieldName,
            message: `Expected string, got ${typeof value}`,
            code: 'TYPE_MISMATCH'
        };
    }
    if (expectedType === 'number' && (typeof value !== 'number' || isNaN(value))) {
        return {
            field: fieldName,
            message: `Expected number, got ${typeof value}`,
            code: 'TYPE_MISMATCH'
        };
    }
    if (expectedType === 'integer') {
        if (!Number.isInteger(value)) {
            return {
                field: fieldName,
                message: `Expected integer, got ${typeof value === 'number' && isNaN(value) ? 'NaN' : typeof value}`,
                code: 'TYPE_MISMATCH'
            };
        }
    }
    if (expectedType === 'boolean' && typeof value !== 'boolean') {
        return {
            field: fieldName,
            message: `Expected boolean, got ${typeof value}`,
            code: 'TYPE_MISMATCH'
        };
    }
    return null;
}
function deepMerge(target, source) {
    const result = { ...target };
    for (const key in source) {
        if (key === '__proto__' || key === 'constructor' || key === 'prototype') {
            continue;
        }
        const sourceValue = source[key];
        if (sourceValue === undefined) {
            continue;
        }
        const targetValue = target[key];
        if (isPlainObject(sourceValue) && isPlainObject(targetValue)) {
            result[key] = deepMerge(targetValue, sourceValue);
        }
        else if (Array.isArray(sourceValue)) {
            result[key] = [...sourceValue];
        }
        else {
            result[key] = sourceValue;
        }
    }
    return result;
}
function generateId() {
    return Math.random().toString(36).substring(2, 15) +
        Math.random().toString(36).substring(2, 15);
}
function formatDuration(ms) {
    if (ms < 1000) {
        return `${ms}ms`;
    }
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    if (hours > 0) {
        return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
    }
    else if (minutes > 0) {
        return `${minutes}m ${seconds % 60}s`;
    }
    else {
        return `${seconds}s`;
    }
}
async function retryWithBackoff(fn, maxRetries = 3, baseDelay = 1000, shouldRetry, onRetry) {
    let lastError;
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            return await fn();
        }
        catch (error) {
            lastError = error;
            if (attempt < maxRetries && (!shouldRetry || shouldRetry(error))) {
                let delay = baseDelay * Math.pow(2, attempt);
                if (error.name === 'RateLimitError' && error.getRetryAfterMs) {
                    delay = Math.max(delay, error.getRetryAfterMs());
                }
                else if (error.details?.retryAfter) {
                    delay = Math.max(delay, error.details.retryAfter * 1000);
                }
                const jitter = delay * 0.2 * (Math.random() * 2 - 1);
                delay = Math.max(0, delay + jitter);
                if (onRetry) {
                    onRetry(error, attempt + 1, delay);
                }
                await sleep(delay);
            }
            else {
                throw error;
            }
        }
    }
    throw lastError;
}
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
function debounce(fn, delay) {
    let timeoutId;
    return (...args) => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn(...args), delay);
    };
}
function throttle(fn, limit) {
    let lastArgs = null;
    let inThrottle = false;
    return (...args) => {
        if (!inThrottle) {
            fn(...args);
            inThrottle = true;
            setTimeout(() => {
                inThrottle = false;
                if (lastArgs) {
                    fn(...lastArgs);
                    lastArgs = null;
                }
            }, limit);
        }
        else {
            lastArgs = args;
        }
    };
}
function parseDuration(duration) {
    const match = duration.match(/^(\d+)(ms|s|m|h)$/);
    if (!match) {
        throw new Error(`Invalid duration format: ${duration}`);
    }
    const value = parseInt(match[1], 10);
    const unit = match[2];
    switch (unit) {
        case 'ms':
            return value;
        case 's':
            return value * 1000;
        case 'm':
            return value * 60 * 1000;
        case 'h':
            return value * 60 * 60 * 1000;
        default:
            throw new Error(`Invalid duration unit: ${unit}`);
    }
}
function isPlainObject(value) {
    if (value === null || typeof value !== 'object') {
        return false;
    }
    const proto = Object.getPrototypeOf(value);
    return proto === null || proto === Object.prototype;
}
function deepClone(obj) {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }
    if (obj instanceof Date) {
        return new Date(obj.getTime());
    }
    if (obj instanceof RegExp) {
        return new RegExp(obj.source, obj.flags);
    }
    if (obj instanceof Set) {
        const result = new Set();
        obj.forEach(value => result.add(deepClone(value)));
        return result;
    }
    if (obj instanceof Map) {
        const result = new Map();
        obj.forEach((value, key) => result.set(key, deepClone(value)));
        return result;
    }
    if (Array.isArray(obj)) {
        return obj.map(item => deepClone(item));
    }
    const cloned = {};
    for (const key in obj) {
        if (Object.prototype.hasOwnProperty.call(obj, key)) {
            cloned[key] = deepClone(obj[key]);
        }
    }
    return cloned;
}
function pick(obj, keys) {
    const result = {};
    for (const key of keys) {
        if (key in obj) {
            result[key] = obj[key];
        }
    }
    return result;
}
function omit(obj, keys) {
    const result = { ...obj };
    for (const key of keys) {
        delete result[key];
    }
    return result;
}
//# sourceMappingURL=helpers.js.map