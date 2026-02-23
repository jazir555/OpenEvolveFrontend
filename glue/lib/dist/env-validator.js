"use strict";
/**
 * Environment Validator
 *
 * Follows the Federation Constitution:
 * - Law of Configuration Explicitness: No magic defaults
 * - All configurable values must be via environment variables
 * - Crashes immediately if required vars are missing
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.validateEnv = validateEnv;
exports.validateEnvWithTypes = validateEnvWithTypes;
exports.getEnv = getEnv;
var logger_1 = require("./logger");
/**
 * Validate environment variables
 *
 * Crashes immediately if missing required vars (Law of Configuration Explicitness)
 * Validates types (URLs, ports, numbers)
 *
 * @param required - Array of required environment variable names
 * @throws Error if validation fails
 */
function validateEnv(required) {
    var errors = [];
    for (var _i = 0, required_1 = required; _i < required_1.length; _i++) {
        var varName = required_1[_i];
        var value = process.env[varName];
        if (!value || value.trim() === '') {
            errors.push("Missing required environment variable: ".concat(varName));
        }
    }
    if (errors.length > 0) {
        var errorMessage = "Environment validation failed:\n".concat(errors.join('\n'));
        logger_1.logger.error(errorMessage, undefined, {
            validation_errors: errors,
            missing_vars: errors.length,
        });
        // Crash immediately (Law of Configuration Explicitness)
        throw new Error(errorMessage);
    }
    logger_1.logger.info('Environment validation passed', {
        validated_vars: required.length,
    });
}
/**
 * Validate environment variables with type checking
 *
 * @param vars - Array of environment variable definitions
 * @returns Object with validated and parsed values
 * @throws Error if validation fails
 */
function validateEnvWithTypes(vars) {
    var errors = [];
    var result = {};
    for (var _i = 0, vars_1 = vars; _i < vars_1.length; _i++) {
        var envVar = vars_1[_i];
        var value = process.env[envVar.name];
        // Check if required variable is missing
        if (envVar.required && (!value || value.trim() === '')) {
            errors.push("Missing required environment variable: ".concat(envVar.name));
            continue;
        }
        // Use default for optional variables
        if (!value && envVar.default !== undefined) {
            result[envVar.name] = envVar.default;
            continue;
        }
        if (!value) {
            continue;
        }
        // Type validation
        try {
            switch (envVar.type) {
                case 'string':
                    result[envVar.name] = value;
                    break;
                case 'number':
                    var num = Number(value);
                    if (isNaN(num)) {
                        errors.push("".concat(envVar.name, ": \"").concat(value, "\" is not a valid number"));
                    }
                    else {
                        result[envVar.name] = num;
                    }
                    break;
                case 'boolean':
                    if (value.toLowerCase() === 'true' || value === '1') {
                        result[envVar.name] = true;
                    }
                    else if (value.toLowerCase() === 'false' || value === '0') {
                        result[envVar.name] = false;
                    }
                    else {
                        errors.push("".concat(envVar.name, ": \"").concat(value, "\" is not a valid boolean (use true/false or 1/0)"));
                    }
                    break;
                case 'url':
                    try {
                        new URL(value);
                        result[envVar.name] = value;
                    }
                    catch (_a) {
                        errors.push("".concat(envVar.name, ": \"").concat(value, "\" is not a valid URL"));
                    }
                    break;
                case 'port':
                    var port = Number(value);
                    if (isNaN(port) || port < 1 || port > 65535) {
                        errors.push("".concat(envVar.name, ": \"").concat(value, "\" is not a valid port (1-65535)"));
                    }
                    else {
                        result[envVar.name] = port;
                    }
                    break;
            }
        }
        catch (error) {
            errors.push("".concat(envVar.name, ": Validation error - ").concat(error instanceof Error ? error.message : String(error)));
        }
    }
    if (errors.length > 0) {
        var errorMessage = "Environment validation failed:\n".concat(errors.join('\n'));
        logger_1.logger.error(errorMessage, undefined, {
            validation_errors: errors,
            error_count: errors.length,
        });
        // Crash immediately (Law of Configuration Explicitness)
        throw new Error(errorMessage);
    }
    logger_1.logger.info('Environment validation passed', {
        validated_vars: Object.keys(result).length,
    });
    return result;
}
/**
 * Get environment variable or throw error
 *
 * Convenience function to get a single required env var with type checking
 */
function getEnv(name, type) {
    if (type === void 0) { type = 'string'; }
    var value = process.env[name];
    if (!value || value.trim() === '') {
        throw new Error("Missing required environment variable: ".concat(name));
    }
    switch (type) {
        case 'string':
            return value;
        case 'number':
            var num = Number(value);
            if (isNaN(num)) {
                throw new Error("".concat(name, ": \"").concat(value, "\" is not a valid number"));
            }
            return num;
        case 'boolean':
            if (value.toLowerCase() === 'true' || value === '1') {
                return true;
            }
            else if (value.toLowerCase() === 'false' || value === '0') {
                return false;
            }
            throw new Error("".concat(name, ": \"").concat(value, "\" is not a valid boolean (use true/false or 1/0)"));
        case 'url':
            try {
                new URL(value);
                return value;
            }
            catch (_a) {
                throw new Error("".concat(name, ": \"").concat(value, "\" is not a valid URL"));
            }
        case 'port':
            var port = Number(value);
            if (isNaN(port) || port < 1 || port > 65535) {
                throw new Error("".concat(name, ": \"").concat(value, "\" is not a valid port (1-65535)"));
            }
            return port;
        default:
            return value;
    }
}
/**
 * Example usage:
 *
 * ```typescript
 * import { validateEnv, validateEnvWithTypes, getEnv } from './env-validator';
 *
 * // Simple validation (just checks presence)
 * validateEnv([
 *   'DATABASE_URL',
 *   'API_KEY',
 *   'SERVICE_PORT',
 * ]);
 *
 * // Validation with type checking
 * const config = validateEnvWithTypes([
 *   { name: 'DATABASE_URL', type: 'url', required: true },
 *   { name: 'SERVICE_PORT', type: 'port', required: true },
 *   { name: 'MAX_RETRIES', type: 'number', required: false, default: 3 },
 *   { name: 'DEBUG_MODE', type: 'boolean', required: false, default: false },
 *   { name: 'API_TIMEOUT_MS', type: 'number', required: false, default: 5000 },
 * ]);
 *
 * console.log(config.DATABASE_URL);  // "postgresql://..."
 * console.log(config.SERVICE_PORT);  // 8000 (number)
 * console.log(config.MAX_RETRIES);   // 3 (number)
 * console.log(config.DEBUG_MODE);    // false (boolean)
 *
 * // Get single variable
 * const dbUrl = getEnv('DATABASE_URL', 'url');
 * const port = getEnv('SERVICE_PORT', 'port');
 *
 * // In adapter startup (Law of Configuration Explicitness):
 * try {
 *   const config = validateEnvWithTypes([
 *     { name: 'TARGET_API_URL', type: 'url', required: true },
 *     { name: 'TIMEOUT_MS', type: 'number', required: false, default: 5000 },
 *   ]);
 *   // Start service
 * } catch (error) {
 *   // Service crashes immediately with clear error message
 *   process.exit(1);
 * }
 * ```
 */
