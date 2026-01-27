/**
 * Common type definitions for Bubble implementations
 * Provides shared types, interfaces, and type guards
 */
import { z } from 'zod';
/**
 * Create a successful result
 */
export function ok(data) {
    return { success: true, data };
}
/**
 * Create a failed result
 */
export function err(error) {
    return { success: false, error };
}
/**
 * Unwrap a result, throwing if it's an error
 */
export function unwrap(result) {
    if (result.success) {
        return result.data;
    }
    throw result.error;
}
/**
 * Common credential types
 */
export var CredentialType;
(function (CredentialType) {
    CredentialType["API_KEY"] = "api_key";
    CredentialType["OAUTH_TOKEN"] = "oauth_token";
    CredentialType["BASIC_AUTH"] = "basic_auth";
    CredentialType["BEARER_TOKEN"] = "bearer_token";
    CredentialType["DATABASE_CRED"] = "database_cred";
    CredentialType["CUSTOM_AUTH_KEY"] = "custom_auth_key";
    CredentialType["SLACK_CRED"] = "slack_cred";
    CredentialType["STRIPE_CRED"] = "stripe_cred";
    CredentialType["AIRTABLE_CRED"] = "airtable_cred";
    CredentialType["GMAIL_CRED"] = "gmail_cred";
    CredentialType["GOOGLE_CALENDAR_CRED"] = "google_calendar_cred";
    CredentialType["SHEETS_CRED"] = "sheets_cred";
    CredentialType["NOTION_CRED"] = "notion_cred";
    CredentialType["POSTGRES_CRED"] = "postgres_cred";
    CredentialType["REDIS_CRED"] = "redis_cred";
    CredentialType["MONGODB_CRED"] = "mongodb_cred";
    CredentialType["S3_CRED"] = "s3_cred";
    CredentialType["AWS_CRED"] = "aws_cred";
})(CredentialType || (CredentialType = {}));
/**
 * Type guard to check if a value is a Result
 */
export function isResult(value) {
    return (typeof value === 'object' &&
        value !== null &&
        'success' in value &&
        typeof value.success === 'boolean');
}
/**
 * Type guard to check if a value is a successful Result
 */
export function isOk(value) {
    return isResult(value) && value.success === true;
}
/**
 * Type guard to check if a value is an error Result
 */
export function isErr(value) {
    return isResult(value) && value.success === false;
}
/**
 * Type guard to check if a value is a plain object
 */
export function isPlainObject(value) {
    return (typeof value === 'object' &&
        value !== null &&
        !Array.isArray(value) &&
        !(value instanceof Date) &&
        !(value instanceof RegExp) &&
        !(value instanceof Error));
}
/**
 * Type guard to check if a string is a valid ISO 8601 timestamp
 */
export function isIsoTimestamp(value) {
    if (typeof value !== 'string')
        return false;
    const isoRegex = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$/;
    return isoRegex.test(value);
}
/**
 * Type guard to check if a value is a non-empty string
 */
export function isNonEmptyString(value) {
    return typeof value === 'string' && value.trim().length > 0;
}
/**
 * Type guard to check if a value is a positive number
 */
export function isPositiveNumber(value) {
    return typeof value === 'number' && !isNaN(value) && value > 0;
}
/**
 * Type guard to check if a value is an array
 */
export function isArray(value) {
    return Array.isArray(value);
}
/**
 * Create a Zod schema for Money type
 */
export function createMoneySchema() {
    return z.object({
        amount: z.number().int().min(0),
        currency: z.string().length(3).regex(/^[A-Z]{3}$/)
    });
}
/**
 * Create a Zod schema for Coordinate type
 */
export function createCoordinateSchema() {
    return z.object({
        latitude: z.number().min(-90).max(90),
        longitude: z.number().min(-180).max(180)
    });
}
/**
 * Create a Zod schema for PersonName type
 */
export function createPersonNameSchema() {
    return z.object({
        prefix: z.string().optional(),
        firstName: z.string().optional(),
        middleName: z.string().optional(),
        lastName: z.string().optional(),
        suffix: z.string().optional(),
        fullName: z.string().optional()
    });
}
/**
 * Deep clone an object (simple implementation)
 */
export function deepClone(obj) {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }
    if (obj instanceof Date) {
        return new Date(obj.getTime());
    }
    if (Array.isArray(obj)) {
        return obj.map(item => deepClone(item));
    }
    const clonedObj = {};
    for (const key in obj) {
        if (Object.prototype.hasOwnProperty.call(obj, key)) {
            clonedObj[key] = deepClone(obj[key]);
        }
    }
    return clonedObj;
}
/**
 * Merge two objects deeply (simple implementation)
 */
export function deepMerge(target, source) {
    const result = { ...target };
    for (const key in source) {
        if (Object.prototype.hasOwnProperty.call(source, key)) {
            const sourceValue = source[key];
            const targetValue = result[key];
            if (isPlainObject(sourceValue) && isPlainObject(targetValue)) {
                result[key] = deepMerge(targetValue, sourceValue);
            }
            else {
                result[key] = sourceValue;
            }
        }
    }
    return result;
}
//# sourceMappingURL=types.js.map