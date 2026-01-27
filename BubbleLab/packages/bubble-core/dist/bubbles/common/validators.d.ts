/**
 * Common validation utilities for Bubble implementations
 * Provides reusable validation functions to ensure data integrity and security
 */
import { z } from 'zod';
/**
 * Email validation regex
 * Follows RFC 5322 standard with practical simplifications
 */
export declare const EMAIL_REGEX: RegExp;
/**
 * URL validation regex
 * Allows http, https protocols
 */
export declare const URL_REGEX: RegExp;
/**
 * ISO 8601 timestamp validation regex
 */
export declare const ISO_TIMESTAMP_REGEX: RegExp;
/**
 * Common validation error class
 */
export declare class ValidationError extends Error {
    field?: string | undefined;
    constructor(message: string, field?: string | undefined);
}
/**
 * Validate email address
 * @param email - Email address to validate
 * @returns True if email is valid
 * @throws ValidationError if email is invalid
 */
export declare function validateEmail(email: string): boolean;
/**
 * Validate URL
 * @param url - URL to validate
 * @param allowedProtocols - Array of allowed protocols (default: ['http', 'https'])
 * @returns True if URL is valid
 * @throws ValidationError if URL is invalid
 */
export declare function validateUrl(url: string, allowedProtocols?: string[]): boolean;
/**
 * Validate timestamp in ISO 8601 format
 * @param timestamp - Timestamp string to validate
 * @returns True if timestamp is valid
 * @throws ValidationError if timestamp is invalid
 */
export declare function validateTimestamp(timestamp: string): boolean;
/**
 * Validate that a string is not empty or just whitespace
 * @param value - String value to validate
 * @param fieldName - Name of the field for error messages
 * @returns True if value is valid
 * @throws ValidationError if value is empty
 */
export declare function validateNonEmptyString(value: string, fieldName?: string): boolean;
/**
 * Validate number is within range
 * @param value - Number to validate
 * @param min - Minimum allowed value (inclusive)
 * @param max - Maximum allowed value (inclusive)
 * @param fieldName - Name of the field for error messages
 * @returns True if value is within range
 * @throws ValidationError if value is out of range
 */
export declare function validateNumberRange(value: number, min: number, max: number, fieldName?: string): boolean;
/**
 * Validate array length
 * @param array - Array to validate
 * @param minLength - Minimum allowed length (default: 0)
 * @param maxLength - Maximum allowed length
 * @param fieldName - Name of the field for error messages
 * @returns True if array length is valid
 * @throws ValidationError if array length is invalid
 */
export declare function validateArrayLength<T>(array: T[], minLength: number | undefined, maxLength: number, fieldName?: string): boolean;
/**
 * Validate object has required properties
 * @param obj - Object to validate
 * @param requiredProps - Array of required property names
 * @param fieldName - Name of the field for error messages
 * @returns True if object has all required properties
 * @throws ValidationError if object is missing required properties
 */
export declare function validateRequiredProperties(obj: Record<string, unknown>, requiredProps: string[], fieldName?: string): boolean;
/**
 * Sanitize string input to prevent injection attacks
 * @param input - String to sanitize
 * @returns Sanitized string
 */
export declare function sanitizeString(input: string): string;
/**
 * Validate and sanitize file path to prevent path traversal
 * @param filePath - File path to validate
 * @param allowAbsolutePaths - Whether to allow absolute paths (default: false)
 * @returns True if path is valid
 * @throws ValidationError if path contains dangerous patterns
 */
export declare function validateFilePath(filePath: string, allowAbsolutePaths?: boolean): boolean;
/**
 * Create a Zod schema that validates a non-empty string
 * @param fieldName - Name of the field for error messages
 * @returns Zod schema
 */
export declare function createNonEmptyStringSchema(fieldName?: string): z.ZodEffects<z.ZodString, string, string>;
/**
 * Create a Zod schema that validates an email
 * @returns Zod schema
 */
export declare function createEmailSchema(): z.ZodEffects<z.ZodString, string, string>;
/**
 * Create a Zod schema that validates a URL
 * @param allowedProtocols - Array of allowed protocols
 * @returns Zod schema
 */
export declare function createUrlSchema(allowedProtocols?: string[]): z.ZodEffects<z.ZodString, string, string>;
/**
 * Batch validate multiple values
 * @param validations - Array of validation functions to execute
 * @returns Object with validation results
 */
export declare function batchValidate(validations: Array<{
    fn: () => boolean;
    field: string;
}>): {
    isValid: boolean;
    errors: Array<{
        field: string;
        message: string;
    }>;
};
//# sourceMappingURL=validators.d.ts.map