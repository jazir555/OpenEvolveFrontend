"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.validateEnv = validateEnv;
exports.validateEnvWithTypes = validateEnvWithTypes;
exports.getEnv = getEnv;
const logger_1 = require("./logger");
function validateEnv(required) {
    const errors = [];
    for (const varName of required) {
        const value = process.env[varName];
        if (!value || value.trim() === '') {
            errors.push(`Missing required environment variable: ${varName}`);
        }
    }
    if (errors.length > 0) {
        const errorMessage = `Environment validation failed:\n${errors.join('\n')}`;
        logger_1.logger.error(errorMessage, undefined, {
            validation_errors: errors,
            missing_vars: errors.length,
        });
        throw new Error(errorMessage);
    }
    logger_1.logger.info('Environment validation passed', {
        validated_vars: required.length,
    });
}
function validateEnvWithTypes(vars) {
    const errors = [];
    const result = {};
    for (const envVar of vars) {
        const value = process.env[envVar.name];
        if (envVar.required && (!value || value.trim() === '')) {
            errors.push(`Missing required environment variable: ${envVar.name}`);
            continue;
        }
        if (!value && envVar.default !== undefined) {
            result[envVar.name] = envVar.default;
            continue;
        }
        if (!value) {
            continue;
        }
        try {
            switch (envVar.type) {
                case 'string':
                    result[envVar.name] = value;
                    break;
                case 'number':
                    const num = Number(value);
                    if (isNaN(num)) {
                        errors.push(`${envVar.name}: "${value}" is not a valid number`);
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
                        errors.push(`${envVar.name}: "${value}" is not a valid boolean (use true/false or 1/0)`);
                    }
                    break;
                case 'url':
                    try {
                        new URL(value);
                        result[envVar.name] = value;
                    }
                    catch {
                        errors.push(`${envVar.name}: "${value}" is not a valid URL`);
                    }
                    break;
                case 'port':
                    const port = Number(value);
                    if (isNaN(port) || port < 1 || port > 65535) {
                        errors.push(`${envVar.name}: "${value}" is not a valid port (1-65535)`);
                    }
                    else {
                        result[envVar.name] = port;
                    }
                    break;
            }
        }
        catch (error) {
            errors.push(`${envVar.name}: Validation error - ${error instanceof Error ? error.message : String(error)}`);
        }
    }
    if (errors.length > 0) {
        const errorMessage = `Environment validation failed:\n${errors.join('\n')}`;
        logger_1.logger.error(errorMessage, undefined, {
            validation_errors: errors,
            error_count: errors.length,
        });
        throw new Error(errorMessage);
    }
    logger_1.logger.info('Environment validation passed', {
        validated_vars: Object.keys(result).length,
    });
    return result;
}
function getEnv(name, type = 'string') {
    const value = process.env[name];
    if (!value || value.trim() === '') {
        throw new Error(`Missing required environment variable: ${name}`);
    }
    switch (type) {
        case 'string':
            return value;
        case 'number':
            const num = Number(value);
            if (isNaN(num)) {
                throw new Error(`${name}: "${value}" is not a valid number`);
            }
            return num;
        case 'boolean':
            if (value.toLowerCase() === 'true' || value === '1') {
                return true;
            }
            else if (value.toLowerCase() === 'false' || value === '0') {
                return false;
            }
            throw new Error(`${name}: "${value}" is not a valid boolean (use true/false or 1/0)`);
        case 'url':
            try {
                new URL(value);
                return value;
            }
            catch {
                throw new Error(`${name}: "${value}" is not a valid URL`);
            }
        case 'port':
            const port = Number(value);
            if (isNaN(port) || port < 1 || port > 65535) {
                throw new Error(`${name}: "${value}" is not a valid port (1-65535)`);
            }
            return port;
        default:
            return value;
    }
}
//# sourceMappingURL=env-validator.js.map