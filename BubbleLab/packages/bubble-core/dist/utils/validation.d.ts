/**
 * Validation Utilities
 *
 * Provides helpers for validating and parsing data using Zod schemas
 * with consistent error handling and reporting.
 */
import { z } from 'zod';
import { Result } from './result.js';
/**
 * Custom validation error
 */
export declare class ValidationError extends Error {
    errors: Array<{
        path: string[];
        message: string;
    }>;
    constructor(message: string, errors: Array<{
        path: string[];
        message: string;
    }>);
}
/**
 * Validate and parse data using Zod schema
 * Throws ValidationError if validation fails
 */
export declare function validateAndParse<T>(schema: z.ZodSchema<T>, data: unknown, errorMessage?: string): T;
/**
 * Safely validate data using Zod schema
 * Returns Result type instead of throwing
 */
export declare function safeValidate<T>(schema: z.ZodSchema<T>, data: unknown): Result<T, ValidationError>;
/**
 * Validate data and return detailed error information
 */
export interface ValidationResult<T> {
    success: boolean;
    data?: T;
    errors?: Array<{
        field: string;
        message: string;
        code: string;
    }>;
}
export declare function validateWithDetails<T>(schema: z.ZodSchema<T>, data: unknown): ValidationResult<T>;
/**
 * Create a validator function from a schema
 */
export declare function createValidator<T>(schema: z.ZodSchema<T>): {
    validate: (data: unknown) => T;
    safeValidate: (data: unknown) => Result<T, ValidationError>;
    validateWithDetails: (data: unknown) => ValidationResult<T>;
};
/**
 * Common validation schemas
 */
export declare const CommonSchemas: {
    email: z.ZodString;
    url: z.ZodString;
    uuid: z.ZodString;
    date: z.ZodEffects<z.ZodUnion<[z.ZodString, z.ZodDate]>, Date, string | Date>;
    positiveNumber: z.ZodNumber;
    nonNegativeNumber: z.ZodNumber;
    pagination: z.ZodObject<{
        page: z.ZodDefault<z.ZodNumber>;
        pageSize: z.ZodDefault<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        page: number;
        pageSize: number;
    }, {
        page?: number | undefined;
        pageSize?: number | undefined;
    }>;
    id: z.ZodString;
    timestamp: z.ZodNumber;
    booleanString: z.ZodEffects<z.ZodEnum<["true", "false", "True", "False", "TRUE", "FALSE", "1", "0"]>, boolean, "0" | "1" | "true" | "false" | "TRUE" | "FALSE" | "True" | "False">;
};
/**
 * Sanitization helpers
 */
export declare const Sanitizers: {
    /**
     * Trim and sanitize string input
     */
    string: (value: unknown) => string | null;
    /**
     * Sanitize email (lowercase and trim)
     */
    email: (value: unknown) => string | null;
    /**
     * Sanitize URL
     */
    url: (value: unknown) => string | null;
    /**
     * Sanitize array of strings
     */
    stringArray: (value: unknown) => string[] | null;
};
/**
 * Type guards for runtime validation
 */
export declare const TypeGuards: {
    isString: (value: unknown) => value is string;
    isNumber: (value: unknown) => value is number;
    isBoolean: (value: unknown) => value is boolean;
    isArray: (value: unknown) => value is unknown[];
    isObject: (value: unknown) => value is Record<string, unknown>;
    isDate: (value: unknown) => value is Date;
    isEmpty: (value: unknown) => boolean;
};
/**
 * Assert utilities for development
 */
export declare const assert: {
    isDefined: <T>(value: T | null | undefined, message?: string) => T;
    isString: (value: unknown, message?: string) => string;
    isNumber: (value: unknown, message?: string) => number;
    isArray: <T>(value: unknown, message?: string) => T[];
};
//# sourceMappingURL=validation.d.ts.map