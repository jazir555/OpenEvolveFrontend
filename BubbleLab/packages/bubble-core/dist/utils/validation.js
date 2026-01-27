/**
 * Validation Utilities
 *
 * Provides helpers for validating and parsing data using Zod schemas
 * with consistent error handling and reporting.
 */
import { z } from 'zod';
import { err } from './result.js';
/**
 * Custom validation error
 */
export class ValidationError extends Error {
    errors;
    constructor(message, errors) {
        super(message);
        this.errors = errors;
        this.name = 'ValidationError';
    }
}
/**
 * Validate and parse data using Zod schema
 * Throws ValidationError if validation fails
 */
export function validateAndParse(schema, data, errorMessage) {
    try {
        return schema.parse(data);
    }
    catch (error) {
        if (error instanceof z.ZodError) {
            const errors = error.errors.map((e) => ({
                path: e.path.map(String),
                message: e.message,
            }));
            const issues = errors.map((e) => `${e.path.join('.') || 'field'}: ${e.message}`).join(', ');
            throw new ValidationError(errorMessage || 'Validation failed', errors);
        }
        throw error;
    }
}
/**
 * Safely validate data using Zod schema
 * Returns Result type instead of throwing
 */
export function safeValidate(schema, data) {
    try {
        const parsed = schema.parse(data);
        return { success: true, data: parsed };
    }
    catch (error) {
        if (error instanceof z.ZodError) {
            const errors = error.errors.map((e) => ({
                path: e.path.map(String),
                message: e.message,
            }));
            return err(new ValidationError('Validation failed', errors));
        }
        return err(new ValidationError('Unknown validation error', []));
    }
}
export function validateWithDetails(schema, data) {
    const result = schema.safeParse(data);
    if (result.success) {
        return {
            success: true,
            data: result.data,
        };
    }
    const errors = result.error.errors.map((e) => ({
        field: e.path.join('.') || 'unknown',
        message: e.message,
        code: e.code,
    }));
    return {
        success: false,
        errors,
    };
}
/**
 * Create a validator function from a schema
 */
export function createValidator(schema) {
    return {
        validate: (data) => validateAndParse(schema, data),
        safeValidate: (data) => safeValidate(schema, data),
        validateWithDetails: (data) => validateWithDetails(schema, data),
    };
}
/**
 * Common validation schemas
 */
export const CommonSchemas = {
    // Email validation
    email: z.string().email('Invalid email format'),
    // URL validation
    url: z.string().url('Invalid URL format'),
    // UUID validation
    uuid: z.string().uuid('Invalid UUID format'),
    // Date validation
    date: z.string().or(z.date()).transform((val) => new Date(val)),
    // Positive number
    positiveNumber: z.number().positive('Must be a positive number'),
    // Non-negative number
    nonNegativeNumber: z.number().nonnegative('Must be a non-negative number'),
    // Pagination
    pagination: z.object({
        page: z.number().int().positive().default(1),
        pageSize: z.number().int().positive().max(500).default(50),
    }),
    // ID parameter
    id: z.string().min(1, 'ID is required'),
    // Timestamp
    timestamp: z.number().int().positive(),
    // Boolean string
    booleanString: z
        .enum(['true', 'false', 'True', 'False', 'TRUE', 'FALSE', '1', '0'])
        .transform((val) => ['true', 'True', 'TRUE', '1'].includes(val)),
};
/**
 * Sanitization helpers
 */
export const Sanitizers = {
    /**
     * Trim and sanitize string input
     */
    string: (value) => {
        if (typeof value !== 'string')
            return null;
        return value.trim();
    },
    /**
     * Sanitize email (lowercase and trim)
     */
    email: (value) => {
        if (typeof value !== 'string')
            return null;
        return value.trim().toLowerCase();
    },
    /**
     * Sanitize URL
     */
    url: (value) => {
        if (typeof value !== 'string')
            return null;
        return value.trim();
    },
    /**
     * Sanitize array of strings
     */
    stringArray: (value) => {
        if (!Array.isArray(value))
            return null;
        const strings = value.filter((item) => typeof item === 'string');
        return strings.map((s) => s.trim());
    },
};
/**
 * Type guards for runtime validation
 */
export const TypeGuards = {
    isString: (value) => typeof value === 'string',
    isNumber: (value) => typeof value === 'number' && !isNaN(value),
    isBoolean: (value) => typeof value === 'boolean',
    isArray: (value) => Array.isArray(value),
    isObject: (value) => typeof value === 'object' && value !== null && !Array.isArray(value),
    isDate: (value) => value instanceof Date,
    isEmpty: (value) => {
        if (value === null || value === undefined)
            return true;
        if (typeof value === 'string')
            return value.trim().length === 0;
        if (Array.isArray(value))
            return value.length === 0;
        if (typeof value === 'object')
            return Object.keys(value).length === 0;
        return false;
    },
};
/**
 * Assert utilities for development
 */
export const assert = {
    isDefined: (value, message = 'Value must be defined') => {
        if (value === null || value === undefined) {
            throw new Error(message);
        }
        return value;
    },
    isString: (value, message = 'Value must be a string') => {
        if (typeof value !== 'string') {
            throw new Error(message);
        }
        return value;
    },
    isNumber: (value, message = 'Value must be a number') => {
        if (typeof value !== 'number' || isNaN(value)) {
            throw new Error(message);
        }
        return value;
    },
    isArray: (value, message = 'Value must be an array') => {
        if (!Array.isArray(value)) {
            throw new Error(message);
        }
        return value;
    },
};
//# sourceMappingURL=validation.js.map